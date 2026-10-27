"""
lecture_01_comprehensive_cost_engine.py
A 股量化投资摩擦成本、主动管理基本法则 (FLAM) 与真实净收益归因引擎
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class MarketFrictionConfig:
    stamp_duty_sell: float = 0.0005     # 印花税 (2023年8月后口径: 卖方单边 0.05%)
    broker_commission: float = 0.00015  # 券商综合佣金 (双边各万分之1.5)
    transfer_fee: float = 0.00001       # 证券过户费 (双边各十万分之一)
    slippage_one_way: float = 0.0004    # 单边买卖盘口买卖价差与冲击滑点 (单边万分之4)
    im_annual_basis_cost: float = 0.075 # 中证1000期指 (IM) 平均年化贴水损耗 (7.5%)
    ic_annual_basis_cost: float = 0.055 # 中证500期指 (IC) 平均年化贴水损耗 (5.5%)
    if_annual_basis_cost: float = 0.025 # 沪深300期指 (IF) 平均年化贴水损耗 (2.5%)


@dataclass
class FLAMParameters:
    ic: float = 0.045                   # 信号截面平均 Rank IC (例如 0.045)
    breadth_universe_size: int = 1000   # 股票池广度 (如中证1000为 1000 只)
    rebalance_frequency_days: int = 5   # 调仓周期 (天)
    transfer_coefficient: float = 0.45  # 约束转化系数 TC (受限于做空受限、涨跌停、行业偏离约束)
    active_risk_vol: float = 0.08       # 目标年化主动跟踪误差波动率 TE (8%)


class QuantCostAttributionEngine:
    def __init__(self, config: MarketFrictionConfig = MarketFrictionConfig()):
        self.cfg = config

    def calculate_flam_theoretical_metrics(self, params: FLAMParameters) -> dict:
        """
        基于 Grinold & Kahn (1999) 主动管理基本法则 (Fundamental Law of Active Management)
        计算理论信息比率 (IR) 与理论毛超额收益 (Gross Alpha)
        公式:
          N_annual = (Universe_Size) * (250 / Rebalance_Days)
          IR = IC * sqrt(N_annual) * TC
          Gross_Alpha = IR * Active_Risk
        """
        decisions_per_year = (250.0 / params.rebalance_frequency_days)
        annual_breadth = params.breadth_universe_size * decisions_per_year
        theoretical_ir = params.ic * np.sqrt(annual_breadth) * params.transfer_coefficient
        gross_alpha = theoretical_ir * params.active_risk_vol

        return {
            "rank_ic": params.ic,
            "universe_size": params.breadth_universe_size,
            "annual_decisions": int(annual_breadth),
            "transfer_coefficient_tc": params.transfer_coefficient,
            "active_risk_te": params.active_risk_vol,
            "theoretical_ir": theoretical_ir,
            "theoretical_gross_alpha": gross_alpha
        }

    def calculate_one_round_friction(self, custom_slippage: float = None) -> float:
        """
        计算单次完整换仓（100% 卖出老组合并 100% 买入新组合）的摩擦总成本率
        包含：卖出印花税 + 买卖双边佣金 + 买卖双边过户费 + 买卖双边滑点
        """
        slip = self.cfg.slippage_one_way if custom_slippage is None else custom_slippage
        total_friction = (
            self.cfg.stamp_duty_sell +
            (self.cfg.broker_commission * 2) +
            (self.cfg.transfer_fee * 2) +
            (slip * 2)
        )
        return total_friction

    def simulate_strategy_net_returns(
        self,
        gross_alpha_annual: float,
        turnover_annual: float,
        strategy_type: str = "index_enhancement",
        hedge_index: str = "IM",
        custom_slippage: float = None
    ) -> dict:
        """
        全量推演量化策略的净超额与收益衰减
        :param gross_alpha_annual: 理论毛超额收益率 (例如 0.20 代表 20%)
        :param turnover_annual: 单边年化换手率 (例如 30 代表一年持仓轮换 30 次，平均持有约 8 天)
        :param strategy_type: "index_enhancement" (指增) 或 "market_neutral" (中性)
        :param hedge_index: 对冲标的 "IM" (中证1000), "IC" (中证500), "IF" (沪深300)
        :param custom_slippage: 自定义单边滑点率
        """
        round_trip_cost = self.calculate_one_round_friction(custom_slippage)
        total_trading_cost = turnover_annual * round_trip_cost

        basis_cost = 0.0
        if strategy_type == "market_neutral":
            if hedge_index == "IM":
                basis_cost = self.cfg.im_annual_basis_cost
            elif hedge_index == "IC":
                basis_cost = self.cfg.ic_annual_basis_cost
            else:
                basis_cost = self.cfg.if_annual_basis_cost

        net_alpha = gross_alpha_annual - total_trading_cost - basis_cost
        cost_erosion_rate = (1.0 - (net_alpha / gross_alpha_annual)) if gross_alpha_annual > 0 else 1.0

        return {
            "strategy_type": strategy_type,
            "hedge_index": hedge_index if strategy_type == "market_neutral" else "None",
            "gross_alpha": gross_alpha_annual,
            "annual_turnover": turnover_annual,
            "round_trip_friction_bps": round_trip_cost * 10000,
            "annual_trading_friction": total_trading_cost,
            "annual_basis_cost": basis_cost,
            "net_alpha": net_alpha,
            "erosion_percentage": cost_erosion_rate * 100.0,
            "is_viable": net_alpha > 0.05
        }

    def generate_sensitivity_table(self, gross_alphas: list, turnovers: list) -> pd.DataFrame:
        """
        生成换手率与毛收益的敏感性分析矩阵 (以中证1000指增为例)
        """
        rows = []
        for g_alpha in gross_alphas:
            row = {"Gross_Alpha": f"{g_alpha*100:.0f}%"}
            for t in turnovers:
                res = self.simulate_strategy_net_returns(g_alpha, t, "index_enhancement")
                row[f"Turnover_{t}x"] = f"{res['net_alpha']*100:.2f}% (损耗{res['erosion_percentage']:.0f}%)"
            rows.append(row)
        return pd.DataFrame(rows)


if __name__ == "__main__":
    engine = QuantCostAttributionEngine()
    
    print("=" * 75)
    print("0. 主动管理基本法则 (FLAM) 理论超额测算 (以中证1000周频调仓为例)")
    print("=" * 75)
    flam_res = engine.calculate_flam_theoretical_metrics(FLAMParameters())
    for k, v in flam_res.items():
        if isinstance(v, float):
            print(f"  {k:28s} : {v:.4f}")
        else:
            print(f"  {k:28s} : {v}")

    print("\n" + "=" * 75)
    print("案例 1：中证 1000 指数增强策略 (中频价量模型，年化换手 35 倍)")
    print("=" * 75)
    res_enhanced = engine.simulate_strategy_net_returns(
        gross_alpha_annual=flam_res["theoretical_gross_alpha"],
        turnover_annual=35.0,
        strategy_type="index_enhancement"
    )
    for k, v in res_enhanced.items():
        if isinstance(v, float):
            print(f"  {k:28s} : {v:.4f}")
        else:
            print(f"  {k:28s} : {v}")

    print("\n" + "=" * 75)
    print("案例 2：相同 Alpha 引擎构造的市场中性策略 (IM 期指对冲，年化贴水 7.5%)")
    print("=" * 75)
    res_neutral = engine.simulate_strategy_net_returns(
        gross_alpha_annual=flam_res["theoretical_gross_alpha"],
        turnover_annual=35.0,
        strategy_type="market_neutral",
        hedge_index="IM"
    )
    for k, v in res_neutral.items():
        if isinstance(v, float):
            print(f"  {k:28s} : {v:.4f}")
        else:
            print(f"  {k:28s} : {v}")

    print("\n" + "=" * 75)
    print("敏感性矩阵：不同换手率对 A 股指增策略净超额的影响")
    print("=" * 75)
    df_sens = engine.generate_sensitivity_table(
        gross_alphas=[0.12, 0.18, 0.25],
        turnovers=[10.0, 25.0, 40.0, 60.0]
    )
    print(df_sens.to_string(index=False))

