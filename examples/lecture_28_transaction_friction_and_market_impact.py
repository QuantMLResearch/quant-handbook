"""显式费率与平方根冲击假设下的订单成本、等额日调仓容量情景。"""
from dataclasses import dataclass
import numpy as np
import polars as pl


@dataclass(frozen=True)
class TradeFrictionDetail:
    symbol: str
    trade_amount: float
    direction: str
    stamp_duty: float
    transfer_fee: float
    commission: float
    impact_cost: float
    total_friction: float
    friction_bps: float


class AshareFrictionEngine:
    """费率为教学配置，非实时税率服务；历史回测须按有效日期选择费率。"""
    def __init__(self, stamp_duty_rate=.0005, commission_rate=.00015,
                 min_commission=5., transfer_fee_rate=.00001, impact_coef_y=.6):
        values = [stamp_duty_rate, commission_rate, min_commission, transfer_fee_rate, impact_coef_y]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError('费率、保底和冲击参数须有限且非负')
        self.stamp_duty, self.commission, self.min_comm, self.transfer_fee, self.impact_y = values

    def calculate_trade_friction(self, symbol, direction, trade_amount, adv_20d,
                                  daily_volatility=.025):
        if direction not in ('BUY', 'SELL'):
            raise ValueError('direction 必须为 BUY/SELL')
        if not np.isfinite([trade_amount, adv_20d, daily_volatility]).all() or trade_amount < 0 or adv_20d <= 0 or daily_volatility < 0:
            raise ValueError('金额须非负、ADV 须为正、波动率须非负，且均须有限')
        if trade_amount == 0:
            return TradeFrictionDetail(symbol, 0., direction, 0., 0., 0., 0., 0., 0.)
        duty = trade_amount * self.stamp_duty if direction == 'SELL' else 0.
        transfer = trade_amount * self.transfer_fee
        commission = max(self.min_comm, trade_amount * self.commission)
        impact = trade_amount * self.impact_y * daily_volatility * np.sqrt(trade_amount / adv_20d)
        total = duty + transfer + commission + impact
        return TradeFrictionDetail(symbol, trade_amount, direction, duty, transfer, commission,
                                   impact, total, total / trade_amount * 10000)


class AUMCapacityEstimator:
    """均匀分散的日调仓假设；单边年换手 τ 对应买/卖各 AUM*τ。

    每日每股每侧一笔订单，因此保底佣金、买卖税差和冲击均通过同一个成本引擎计算。
    不模拟收益随容量变化、订单拆分或流动性动态；结果只是参数情景。
    """
    def __init__(self, gross_alpha_annual=.22, annual_turnover=30., n_stocks=200,
                 mean_adv=80_000_000., mean_vol=.025, trading_days=250,
                 friction_engine=None, max_participation=.1):
        if not np.isfinite([gross_alpha_annual, annual_turnover, mean_adv, mean_vol, max_participation]).all() or annual_turnover < 0 or mean_adv <= 0 or mean_vol < 0 or not 0 < max_participation <= 1:
            raise ValueError('容量参数范围不合法')
        if not isinstance(n_stocks, int) or n_stocks < 1 or not isinstance(trading_days, int) or trading_days < 1:
            raise ValueError('股票数与交易日数须为正整数')
        self.gross_alpha, self.turnover, self.n_stocks = gross_alpha_annual, annual_turnover, n_stocks
        self.mean_adv, self.mean_vol, self.days = mean_adv, mean_vol, trading_days
        self.engine = friction_engine or AshareFrictionEngine()
        self.max_participation = max_participation

    def estimate_capacity_curve(self, aum_candidates):
        if not len(aum_candidates) or not np.isfinite(aum_candidates).all() or min(aum_candidates) <= 0:
            raise ValueError('AUM 候选须非空、有限且为正')
        rows = []
        for aum in aum_candidates:
            q = aum * self.turnover / (self.days * self.n_stocks)
            # 两侧总参与量，不把超过模型适用边界的候选称为可执行容量。
            participation = 2 * q / self.mean_adv
            if participation > self.max_participation:
                raise ValueError('候选的每日双边参与量超出配置上限，请调整执行假设')
            buy = self.engine.calculate_trade_friction('POOL', 'BUY', q, self.mean_adv, self.mean_vol)
            sell = self.engine.calculate_trade_friction('POOL', 'SELL', q, self.mean_adv, self.mean_vol)
            annual_cost = self.days * self.n_stocks * (buy.total_friction + sell.total_friction) / aum
            net_alpha = self.gross_alpha - annual_cost
            rows.append(dict(aum_cny=float(aum), aum_yi=aum / 1e8, daily_two_sided_participation=participation,
                             annual_friction=annual_cost, net_alpha=net_alpha,
                             annual_profit_cny=aum * net_alpha))
        return pl.DataFrame(rows)


if __name__ == '__main__':
    engine = AshareFrictionEngine()
    print(engine.calculate_trade_friction('DEMO', 'SELL', 1e6, 5e7, .02))
    curve = AUMCapacityEstimator(friction_engine=engine).estimate_capacity_curve(
        [1e7, 5e7, 2e8, 5e8, 1e9, 2e9, 4e9])
    print(curve)
    best = curve.sort('annual_profit_cny', descending=True).row(0, named=True)
    print(f'候选网格内最大模型利润：AUM={best["aum_yi"]:.2f}亿元，利润={best["annual_profit_cny"] / 1e4:.2f}万元')
    print('负利润保留；网格最优不是经真实执行验证的策略容量。')
