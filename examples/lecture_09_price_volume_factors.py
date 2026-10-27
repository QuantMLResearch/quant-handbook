"""
lecture_09_price_volume_engine.py
工业级 A 股全市场量价因子工程化计算引擎 (基于 Polars 表达式)
"""
import time
import numpy as np
import polars as pl
import pandas as pd

class AsharePriceVolumeFactorEngine:
    def __init__(self):
        pass

    def compute_all_factors(self, daily_bars_df: pl.DataFrame) -> pl.DataFrame:
        """
        全量批量计算量价特征矩阵 (纯多核向量化)
        输入必须包含: [symbol, trade_date, open, high, low, close, close_hfq, volume, turnover]
        """
        print("[*] 正在启动 Polars 多核并行量价特征流水线...")
        t0 = time.perf_counter()

        prices = ["open", "high", "low", "close", "close_hfq"]
        if daily_bars_df.select(pl.any_horizontal([
            (~pl.col(c).is_finite() | (pl.col(c) <= 0)).fill_null(True) for c in prices
        ]).any()).item():
            raise ValueError("原始和复权价格必须为有限正数")
        if daily_bars_df.select(pl.struct("symbol", "trade_date").is_duplicated().any()).item():
            raise ValueError("行情的股票日期键必须唯一")
        # 转换为惰性执行图
        lazy_df = daily_bars_df.lazy().sort(["symbol", "trade_date"])

        factors_lazy = lazy_df.with_columns([
            # ================= 1. 动量与反转体系 =================
            # 1日收益率 (日收益)
            (pl.col("close_hfq") / pl.col("close_hfq").shift(1).over("symbol") - 1.0).alias("ret_1d"),
            # 5日动量
            (pl.col("close_hfq") / pl.col("close_hfq").shift(5).over("symbol") - 1.0).alias("mom_5d"),
            # 20日动量
            (pl.col("close_hfq") / pl.col("close_hfq").shift(20).over("symbol") - 1.0).alias("mom_20d"),
            # 60日中周期动量
            (pl.col("close_hfq") / pl.col("close_hfq").shift(60).over("symbol") - 1.0).alias("mom_60d"),

            # 5日与20日短期反转因子 (取负号)
            (-(pl.col("close_hfq") / pl.col("close_hfq").shift(5).over("symbol") - 1.0)).alias("rev_5d"),
            (-(pl.col("close_hfq") / pl.col("close_hfq").shift(20).over("symbol") - 1.0)).alias("rev_20d"),

            # ================= 2. 隔夜与日内微观解耦 =================
            # 隔夜收益 (开盘 / 前收 - 1)
            ((pl.col("open") * pl.col("close_hfq") / pl.col("close")) / pl.col("close_hfq").shift(1).over("symbol") - 1.0).alias("ret_overnight"),
            # 日内收益 (收盘 / 开盘 - 1)
            (pl.col("close") / pl.col("open") - 1.0).alias("ret_intraday"),

            # ================= 3. 波动与极端收益体系 (MAX) =================
            # 20日年化波动率
            (
                (pl.col("close_hfq") / pl.col("close_hfq").shift(1).over("symbol") - 1.0)
                .rolling_std(window_size=20)
                .over("symbol") * (252.0 ** 0.5)
            ).alias("volatility_20d"),

            # 日内真实振幅 (High / Low - 1)
            (pl.col("high") / pl.col("low") - 1.0).alias("amplitude_daily"),
            # 20日平均振幅
            ((pl.col("high") / pl.col("low") - 1.0).rolling_mean(20).over("symbol")).alias("amplitude_20d"),

            # MAX 异象因子: 过去 20 日单日最大收益率 (取负号，预期负 Alpha)
            (
                -(pl.col("close_hfq") / pl.col("close_hfq").shift(1).over("symbol") - 1.0)
                .rolling_max(window_size=20)
                .over("symbol")
            ).alias("neg_max_20d"),

            # ================= 4. 流动性与换手活跃度体系 =================
            # 20日成交量移动平均
            (pl.col("volume").rolling_mean(20).over("symbol")).alias("vol_ma20"),
            # 成交量爆发比率 (今日量 / 20日均量)
            (pl.col("volume") / (pl.col("volume").rolling_mean(20).over("symbol") + 1e-8)).alias("vol_surge_ratio"),

            # Amihud 冲击成本因子 (过去20天 |收益| / 成交额 的均值)
            (
                (
                    (pl.col("close_hfq") / pl.col("close_hfq").shift(1).over("symbol") - 1.0).abs() /
                    pl.when(pl.col("turnover") > 0).then(pl.col("turnover") / 1e8).otherwise(None) # 换算为亿元成交额
                )
                .rolling_mean(window_size=20)
                .over("symbol")
            ).alias("amihud_illiq_20d")
        ])

        # 触发物理计算
        res_df = factors_lazy.collect()
        elapsed = time.perf_counter() - t0
        print(f"✅ 量价特征计算完毕！总耗时: {elapsed:.3f} 秒 (包含 {len(res_df.columns)} 个特征字段)")
        return res_df

# =========================================================================
# 单元验证与吞吐量测试
# =========================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n_stocks = 500
    n_days = 250
    total_len = n_stocks * n_days
    print(f"[*] 正在构造规模为 {total_len:,} 行的仿真 A 股行情面板...")

    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    symbols = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]

    records = []
    for s in symbols:
        ret = np.random.normal(0.0003, 0.02, size=n_days)
        prices = 10.0 * np.exp(np.cumsum(ret))
        opens = prices * (1 + np.random.normal(0, 0.004, size=n_days))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.005, size=n_days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.005, size=n_days)))
        vols = np.random.uniform(500000, 5000000, size=n_days)
        turnovers = vols * prices

        for i in range(n_days):
            records.append({
                "symbol": s,
                "trade_date": dates[i].date(),
                "open": opens[i],
                "high": highs[i],
                "low": lows[i],
                "close": prices[i],
                "close_hfq": prices[i], # 合成行情无分红拆股
                "volume": vols[i],
                "turnover": turnovers[i]
            })

    mock_bars = pl.DataFrame(records)

    # 运行特征引擎
    engine = AsharePriceVolumeFactorEngine()
    features_df = engine.compute_all_factors(mock_bars)

    # 抽样打印结果
    sample_sym = "000001.SZ"
    sample_res = features_df.filter(pl.col("symbol") == sample_sym).tail(3)
    
    print("\n" + "=" * 90)
    print(f"标的 {sample_sym} 最新特征计算切片抽样：")
    print("=" * 90)
    cols_to_show = ["symbol", "trade_date", "close", "rev_20d", "ret_overnight", "ret_intraday", "volatility_20d", "neg_max_20d", "amihud_illiq_20d"]
    print(sample_res.select(cols_to_show))
