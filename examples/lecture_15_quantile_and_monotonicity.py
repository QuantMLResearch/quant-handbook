"""
A 股教学示例：因子分层回测引擎 (AshareQuantileBacktestEngine)
支持：5/10 分层收益计算、净值曲线追踪、单调性得分与多头超额解耦
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import polars as pl
from scipy import stats
from examples.factor_utils import validate_panel, annualized_ratio


@dataclass
class QuantileSummaryReport:
    """分层回测评估报告"""
    factor_name: str
    n_quantiles: int
    group_annual_returns: Dict[int, float]
    group_sharpe_ratios: Dict[int, float]
    group_max_drawdowns: Dict[int, float]
    long_short_annual_ret: float
    long_short_sharpe: float
    long_short_mdd: float
    long_only_excess_annual_ret: float
    long_only_excess_sharpe: float
    long_contribution_ratio: float
    monotonicity_score: float
    step_ratio: float

    def print_summary(self):
        """格式化输出报告"""
        print("=" * 70)
        print(f"📈 因子分层回测深度剖析报告: [{self.factor_name}] ({self.n_quantiles} 分位)")
        print("=" * 70)
        print(f"{'组别 (Quantile)':<18} | {'年化复合收益':<14} | {'年化夏普比率':<14} | {'历史最大回撤':<14}")
        print("-" * 70)
        for q in range(1, self.n_quantiles + 1):
            ann_ret = self.group_annual_returns[q]
            shp = self.group_sharpe_ratios[q]
            mdd = self.group_max_drawdowns[q]
            tag = " (Bottom 劣质组)" if q == 1 else (" (Top 优质多头组)" if q == self.n_quantiles else "")
            print(f"Group {q:<2}{tag:<10} | {ann_ret * 100:10.2f}%   | {shp:12.3f}   | {mdd * 100:10.2f}%")
        print("-" * 70)
        print("【多空对冲与多头超额分离度量】")
        print(f"  • 多空对冲年化利差 (L/S Spread) : {self.long_short_annual_ret * 100:8.2f}%")
        print(f"  • 多空夏普比率 (L/S Sharpe)     : {self.long_short_sharpe:8.3f}")
        print(f"  • 多空最大回撤 (L/S MDD)        : {self.long_short_mdd * 100:8.2f}%")
        print(f"  • 多头超额算术年化 (Top-Bench): {self.long_only_excess_annual_ret * 100:8.2f}%")
        print(f"  • 纯多头信息比率 (Active IR)    : {self.long_only_excess_sharpe:8.3f}")
        print(f"  • 多头收益贡献占比              : {self.long_contribution_ratio * 100:8.2f}%")
        print("-" * 70)
        print("【单调性稳健性检验 (Monotonicity)】")
        print(f"  • 斯皮尔曼单调性得分 (MonoScore) : {self.monotonicity_score:8.3f}")
        print(f"  • 台阶递增满足率 (StepRatio)    : {self.step_ratio * 100:8.2f}%")
        print("=" * 70)


class AshareQuantileBacktestEngine:
    """
    A 股分层回测引擎
    """

    def __init__(
        self,
        n_quantiles: int = 5,
        date_col: str = "trade_date",
        symbol_col: str = "symbol",
        annual_trading_days: float = 250.0
    ):
        if n_quantiles < 2:
            raise ValueError("至少需要两个分组")
        self.n_quantiles = n_quantiles
        self.date_col = date_col
        self.symbol_col = symbol_col
        self.annual_days = annual_trading_days

    def _calc_mdd(self, net_values: np.ndarray) -> float:
        """计算净值序列的最大回撤"""
        net_values = np.r_[1.0, np.asarray(net_values, dtype=float)]
        peak = np.maximum.accumulate(net_values)
        drawdown = (net_values - peak) / (peak + 1e-12)
        return float(np.min(drawdown))

    def run_quantile_backtest(
        self,
        df: pl.DataFrame,
        factor_col: str,
        forward_ret_col: str
    ) -> Tuple[pl.DataFrame, QuantileSummaryReport]:
        """
        执行分层回测核心运算
        """
        validate_panel(df, self.date_col, self.symbol_col)
        # 只用当时已知的因子决定候选池；标签缺失不能反过来改变选股。
        valid_df = df.filter(pl.col(factor_col).is_finite()).sort([self.date_col, self.symbol_col])
        if valid_df.is_empty() or valid_df[self.date_col].n_unique() != df[self.date_col].n_unique():
            raise ValueError("存在没有有效因子的日期")
        if valid_df.select((~pl.col(forward_ret_col).is_finite()).fill_null(True).any()).item():
            raise ValueError("候选池前瞻收益缺失；请补齐标签或显式裁剪完整评估区间")
        if valid_df.filter(pl.col(forward_ret_col) < -1).height:
            raise ValueError("股票简单收益率不能小于 -100%")
        counts = valid_df.group_by(self.date_col).len()
        if counts["len"].min() < self.n_quantiles:
            raise ValueError("截面股票数少于分组数")
        # 同分按 symbol 稳定破同分；此时组间差异不代表因子能区分同分股票。
        binned_df = valid_df.with_columns(
            (((pl.col(factor_col).rank(method="ordinal").over(self.date_col) - 1) /
               pl.len().over(self.date_col) * self.n_quantiles).floor().cast(pl.Int32) + 1)
            .alias("quantile_group")
        )

        # 限制边界防止浮点溢出
        binned_df = binned_df.with_columns(
            pl.col("quantile_group").clip(1, self.n_quantiles)
        )

        # 2. 计算每日各组等权收益率
        group_daily_returns = (
            binned_df.group_by([self.date_col, "quantile_group"])
            .agg(pl.col(forward_ret_col).mean().alias("daily_ret"))
            .sort([self.date_col, "quantile_group"])
        )

        # 3. 计算每日全市场等权基准收益率
        bench_daily = (
            valid_df.group_by(self.date_col)
            .agg(pl.col(forward_ret_col).mean().alias("bench_ret"))
            .sort(self.date_col)
        )

        # 4. 透视转换为宽表格式 (每列一个 Group 收益)
        pivot_df = group_daily_returns.pivot(
            on="quantile_group",
            index=self.date_col,
            values="daily_ret"
        ).sort(self.date_col)

        pivot_df = pivot_df.join(bench_daily, on=self.date_col, how="left")

        # 重命名分组列为 G1, G2 ...
        for q in range(1, self.n_quantiles + 1):
            pivot_df = pivot_df.rename({str(q): f"G{q}"})

        # 5. 添加多空利差列与多头超额列
        top_col = f"G{self.n_quantiles}"
        bottom_col = "G1"
        pivot_df = pivot_df.with_columns([
            (pl.col(top_col) - pl.col(bottom_col)).alias("long_short_spread"),
            (pl.col(top_col) - pl.col("bench_ret")).alias("top_excess_ret")
        ])

        # 6. 计算各组统计量
        T = pivot_df.height
        ann_rets = {}
        sharpes = {}
        mdds = {}
        raw_ann_rets = []

        for q in range(1, self.n_quantiles + 1):
            g_col = f"G{q}"
            ret_series = pivot_df[g_col].to_numpy()
            
            # 年化几何复合收益率
            wealth = np.cumprod(1.0 + ret_series)
            ann_ret = float(wealth[-1] ** (self.annual_days / T) - 1.0)
            ann_vol = float(np.std(ret_series, ddof=1) * np.sqrt(self.annual_days))
            sharpe = annualized_ratio(ret_series, self.annual_days, annual_rf=.02)
            mdd = self._calc_mdd(wealth)

            ann_rets[q] = ann_ret
            sharpes[q] = sharpe
            mdds[q] = mdd
            raw_ann_rets.append(float(np.mean(ret_series)))

        # 多空对冲统计
        ls_series = pivot_df["long_short_spread"].to_numpy()
        ls_wealth = np.cumprod(1.0 + ls_series)
        ls_ann_ret = float(ls_wealth[-1] ** (self.annual_days / T) - 1.0) if np.all(ls_series > -1) else float("nan")
        ls_ann_vol = float(np.std(ls_series, ddof=1) * np.sqrt(self.annual_days))
        ls_sharpe = annualized_ratio(ls_series, self.annual_days)
        ls_mdd = self._calc_mdd(ls_wealth) if np.all(ls_series > -1) else float("nan")

        # 多头超额统计
        excess_series = pivot_df["top_excess_ret"].to_numpy()
        excess_wealth = np.cumprod(1.0 + excess_series)
        excess_ann_ret = float(np.mean(excess_series) * self.annual_days) # 算术年化主动收益
        excess_ann_vol = float(np.std(excess_series, ddof=1) * np.sqrt(self.annual_days))
        excess_ir = annualized_ratio(excess_series, self.annual_days)

        # 同一算术均值口径，避免用 CAGR 差值冒充收益贡献。
        spread_mean = float(np.mean(ls_series))
        long_contrib = float(np.mean(excess_series) / spread_mean) if abs(spread_mean) > 1e-12 else float("nan")

        # 单调性打分
        group_ids = np.arange(1, self.n_quantiles + 1)
        mono_score, _ = stats.spearmanr(group_ids, raw_ann_rets)
        
        # 台阶递增满足率
        step_diffs = np.diff(raw_ann_rets)
        step_ratio = float(np.mean(step_diffs > 0))

        report = QuantileSummaryReport(
            factor_name=factor_col,
            n_quantiles=self.n_quantiles,
            group_annual_returns=ann_rets,
            group_sharpe_ratios=sharpes,
            group_max_drawdowns=mdds,
            long_short_annual_ret=ls_ann_ret,
            long_short_sharpe=ls_sharpe,
            long_short_mdd=ls_mdd,
            long_only_excess_annual_ret=excess_ann_ret,
            long_only_excess_sharpe=excess_ir,
            long_contribution_ratio=long_contrib,
            monotonicity_score=float(mono_score),
            step_ratio=step_ratio
        )

        return pivot_df, report


# ================= 合成数据演示（统计结果不是实盘收益证明） =================
if __name__ == "__main__":
    np.random.seed(42)
    n_days = 250
    n_stocks = 600

    import pandas as pd
    dates = pd.bdate_range("2025-01-01", periods=n_days).date.tolist()
    symbols = [f"{s:06d}.SH" for s in range(n_stocks)]

    records = []
    # 模拟真实市场中的高单调性 Alpha 因子
    for d in dates:
        # 市场大盘 Beta 收益
        market_beta = np.random.normal(0.0004, 0.012)
        # 真实的个股纯 Alpha 打分
        true_latent_alpha = np.random.normal(0, 1.0, n_stocks)
        
        # 收益与因子呈现高单调性对应：Top 组收益明显抬高，Bottom 组收益明显拉低
        stock_returns = market_beta + true_latent_alpha * 0.006 + np.random.normal(0, 0.015, n_stocks)
        
        # 观测到的因子值 (包含测量噪声)
        observed_factor = true_latent_alpha + np.random.normal(0, 0.5, n_stocks)

        for s_idx, s in enumerate(symbols):
            records.append({
                "trade_date": d,
                "symbol": s,
                "factor_score": observed_factor[s_idx],
                "ret_fwd_1d": stock_returns[s_idx]
            })

    sim_df = pl.DataFrame(records)

    # 运行 5 分位分层回测
    engine = AshareQuantileBacktestEngine(n_quantiles=5)
    daily_res, report = engine.run_quantile_backtest(
        sim_df,
        factor_col="factor_score",
        forward_ret_col="ret_fwd_1d"
    )

    print(">>> 各组每日收益率透视表 (前 5 行):")
    print(daily_res.head(5))

    # 打印买方标准分层报告
    report.print_summary()
