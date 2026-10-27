"""
A 股教学示例：因子信息系数 (IC) 评估引擎
支持：Pearson IC / Spearman Rank IC 截面并行计算、时序统计量与假设检验
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import polars as pl
from scipy import stats
from examples.factor_utils import validate_panel, mean_test


@dataclass
class ICReport:
    """因子 IC 评估汇总报告"""
    factor_name: str
    sample_days: int
    mean_ic: float
    ic_std: float
    icir: float
    annualized_icir: float
    t_stat: float
    p_value: float
    positive_ratio: float
    mean_rank_ic: float
    rank_ic_std: float
    rank_icir: float
    annualized_rank_icir: float
    rank_t_stat: float
    rank_p_value: float
    rank_positive_ratio: float
    skewness: float
    kurtosis: float
    excluded_days: int
    hac_lags: int

    def print_summary(self):
        """格式化打印评估报告"""
        print("=" * 65)
        print(f"📊 因子全面 IC 评估报告: [{self.factor_name}] (评估交易日: {self.sample_days} 天)")
        print("=" * 65)
        print(f"无效截面: {self.excluded_days}；均值检验 HAC lag={self.hac_lags}（0 表示 iid t）")
        print(f"【Spearman Rank IC (买方核心)】")
        print(f"  • Rank IC 均值         : {self.mean_rank_ic:8.4f}")
        print(f"  • Rank IC 标准差       : {self.rank_ic_std:8.4f}")
        print(f"  • 日频 Rank ICIR       : {self.rank_icir:8.4f}")
        print(f"  • 年化 Rank ICIR       : {self.annualized_rank_icir:8.4f}")
        print(f"  • 均值检验统计量   : {self.rank_t_stat:8.4f} (p值: {self.rank_p_value:.4e})")
        print(f"  • 正 IC 比例 (胜率)    : {self.rank_positive_ratio * 100:7.2f}%")
        print("-" * 65)
        print(f"【Pearson Normal IC (线性参考)】")
        print(f"  • Pearson IC 均值      : {self.mean_ic:8.4f}")
        print(f"  • 年化 Pearson ICIR    : {self.annualized_icir:8.4f}")
        print(f"  • t 统计量             : {self.t_stat:8.4f} (p值: {self.p_value:.4e})")
        print(f"  • 胜率                 : {self.positive_ratio * 100:7.2f}%")
        print("-" * 65)
        print(f"【分布形态与稳健性】")
        print(f"  • IC 偏度 (Skewness)   : {self.skewness:8.4f}")
        print(f"  • IC 峰度 (Kurtosis)   : {self.kurtosis:8.4f}")
        print("=" * 65)


class AshareFactorICEngine:
    """
    A 股因子信息系数评估引擎
    """

    def __init__(
        self,
        date_col: str = "trade_date",
        symbol_col: str = "symbol",
        annual_factor: float = 250.0,
        hac_lags: int = 0
    ):
        self.date_col = date_col
        self.symbol_col = symbol_col
        self.annual_factor = annual_factor
        self.hac_lags = hac_lags

    def compute_daily_ic(
        self,
        df: pl.DataFrame,
        factor_col: str,
        forward_ret_col: str
    ) -> pl.DataFrame:
        """
        利用 Polars 截面表达式并行高效计算每日的 Pearson IC 与 Spearman Rank IC
        """
        validate_panel(df, self.date_col, self.symbol_col)
        # 1. 过滤空值、NaN 和无穷值
        valid_df = df.filter(
            pl.col(factor_col).is_not_null() & 
            pl.col(forward_ret_col).is_not_null() &
            pl.col(factor_col).is_finite() &
            pl.col(forward_ret_col).is_finite()
        )

        # 2. 在每个日期截面上进行 Rank 排序
        ranked_df = valid_df.with_columns([
            pl.col(factor_col).rank().over(self.date_col).alias("factor_rank"),
            pl.col(forward_ret_col).rank().over(self.date_col).alias("ret_rank")
        ])

        # 3. 按日期分组聚合计算相关系数
        daily_ic_df = (
            ranked_df.group_by(self.date_col)
            .agg([
                # 计算股票样本数
                pl.len().alias("stock_count"),
                # 皮尔逊线性相关系数
                pl.corr(factor_col, forward_ret_col).alias("ic"),
                # 斯皮尔曼秩相关系数 (排序后的皮尔逊相关系数)
                pl.corr("factor_rank", "ret_rank").alias("rank_ic")
            ])
            .sort(self.date_col)
        )

        # 保留所有日期并显式标记无效截面；累计线不被 NaN 污染。
        daily_ic_df = df.select(self.date_col).unique().join(daily_ic_df, on=self.date_col, how="left").sort(self.date_col)
        daily_ic_df = daily_ic_df.with_columns([
            pl.col("stock_count").fill_null(0),
            *[pl.when((pl.col("stock_count") >= 3) & pl.col(c).is_finite())
              .then(pl.col(c)).otherwise(None).alias(c) for c in ["ic", "rank_ic"]]
        ]).with_columns([
            (pl.col("ic").is_not_null() & pl.col("rank_ic").is_not_null()).alias("valid_ic"),
            pl.col("ic").fill_null(0).cum_sum().alias("cum_ic"),
            pl.col("rank_ic").fill_null(0).cum_sum().alias("cum_rank_ic")
        ])

        return daily_ic_df

    def evaluate(
        self,
        daily_ic_df: pl.DataFrame,
        factor_name: str
    ) -> ICReport:
        """
        基于日度 IC 序列计算全套时序统计量与置信度检验
        """
        valid = daily_ic_df.filter(pl.col("ic").is_finite() & pl.col("rank_ic").is_finite())
        if self.hac_lags and valid.height != daily_ic_df.height:
            raise ValueError("HAC 需要连续有效截面；请选择完整区间，不能删行压缩时间")
        ic_series = valid["ic"].to_numpy()
        rank_ic_series = valid["rank_ic"].to_numpy()

        T = len(rank_ic_series)
        if T < 5:
            raise ValueError("至少需要 5 个有效截面交易日")

        # --- Pearson IC 统计量 ---
        mean_ic = float(np.mean(ic_series))
        ic_std = float(np.std(ic_series, ddof=1))
        icir = mean_ic / ic_std if ic_std > 1e-12 else float("nan")
        annualized_icir = icir * np.sqrt(self.annual_factor)
        t_stat, p_val = mean_test(ic_series, self.hac_lags)
        pos_ratio = float(np.mean(ic_series > 0))

        # --- Spearman Rank IC 统计量 ---
        mean_rank_ic = float(np.mean(rank_ic_series))
        rank_ic_std = float(np.std(rank_ic_series, ddof=1))
        rank_icir = mean_rank_ic / rank_ic_std if rank_ic_std > 1e-12 else float("nan")
        annualized_rank_icir = rank_icir * np.sqrt(self.annual_factor)
        rank_t_stat, rank_p_val = mean_test(rank_ic_series, self.hac_lags)
        rank_pos_ratio = float(np.mean(rank_ic_series > 0))

        # 偏度与峰度
        skewness = float(stats.skew(rank_ic_series)) if rank_ic_std > 1e-12 else float("nan")
        kurtosis = float(stats.kurtosis(rank_ic_series)) if rank_ic_std > 1e-12 else float("nan")

        report = ICReport(
            factor_name=factor_name,
            sample_days=T,
            mean_ic=mean_ic,
            ic_std=ic_std,
            icir=icir,
            annualized_icir=annualized_icir,
            t_stat=t_stat,
            p_value=p_val,
            positive_ratio=pos_ratio,
            mean_rank_ic=mean_rank_ic,
            rank_ic_std=rank_ic_std,
            rank_icir=rank_icir,
            annualized_rank_icir=annualized_rank_icir,
            rank_t_stat=rank_t_stat,
            rank_p_value=rank_p_val,
            rank_positive_ratio=rank_pos_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            excluded_days=daily_ic_df.height - T,
            hac_lags=self.hac_lags
        )

        return report


# ================= 合成数据演示（统计结果不是实盘收益证明） =================
if __name__ == "__main__":
    np.random.seed(42)
    n_days = 250   # 模拟 1 年交易日
    n_stocks = 500 # 类似中证 500 股票池

    import pandas as pd
    dates = pd.bdate_range("2025-01-01", periods=n_days).date.tolist()
    symbols = [f"{s:06d}.SH" for s in range(n_stocks)]

    records = []
    # 模拟真实市场：构建一个真实的稳健 Alpha 因子
    # 特征：真实 Rank IC 均值约为 0.045, 日频存在噪声
    for d in dates:
        # 当天的真实截面收益率
        true_cross_return = np.random.normal(0.0005, 0.025, n_stocks)
        
        # 因子带有真实预测信号 (系数 0.05) + 市场噪声 (0.95)
        factor_signal = true_cross_return * 2.0 + np.random.normal(0, 1.0, n_stocks)
        
        # 偶发性极端离群点 (模拟 A 股突发重组连板拉偏 Pearson)
        outlier_idx = np.random.choice(n_stocks, 2, replace=False)
        factor_signal[outlier_idx[0]] += 50.0  # 极端因子飞鸟点

        for s_idx, s in enumerate(symbols):
            records.append({
                "trade_date": d,
                "symbol": s,
                "alpha_factor": factor_signal[s_idx],
                "ret_fwd_1d": true_cross_return[s_idx]
            })

    test_df = pl.DataFrame(records)

    # 初始化评估引擎
    engine = AshareFactorICEngine()
    
    # 计算日度 IC
    daily_ic_df = engine.compute_daily_ic(
        test_df,
        factor_col="alpha_factor",
        forward_ret_col="ret_fwd_1d"
    )

    print(">>> 每日 IC 时序与累积 IC (前 5 行):")
    print(daily_ic_df.head(5))

    # 生成买方标准评估报告
    report = engine.evaluate(daily_ic_df, factor_name="alpha_robust_momentum")
    report.print_summary()
