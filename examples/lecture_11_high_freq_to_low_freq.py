"""
lecture_11_high_freq_ofi_engine.py
Level-2 订单簿订单流不平衡 (OFI) 模拟与高频低频化聚合计算引擎
"""
from datetime import datetime, timedelta, date
import time
import numpy as np
import polars as pl
import pandas as pd

class AshareHighFreqOFIEngine:
    def __init__(self):
        pass

    def compute_intraday_ofi(self, l2_snapshots_df: pl.DataFrame) -> pl.DataFrame:
        """
        计算单只股票全天日内快照的 OFI 序列 (微观状态机)
        输入包含: [symbol, timestamp, bid_price, bid_vol, ask_price, ask_vol]
        """
        if (l2_snapshots_df.schema["timestamp"] != pl.Datetime
                or l2_snapshots_df.schema["timestamp"].time_zone is not None):
            raise ValueError("timestamp 必须为上海本地无时区 Datetime")
        if l2_snapshots_df.select(pl.struct("symbol", "timestamp").is_duplicated().any()).item():
            raise ValueError("快照的股票时间键必须唯一")
        if l2_snapshots_df.select(pl.any_horizontal([
            (~pl.col(c).is_finite() | (pl.col(c) <= 0)).fill_null(True)
            for c in ["bid_price", "ask_price", "bid_vol", "ask_vol"]
        ]).any()).item():
            raise ValueError("报价和深度必须为有限正数；空盘口需上游处理")
        l2_snapshots_df = l2_snapshots_df.with_columns(pl.col("timestamp").dt.date().alias("trade_date"))
        # 利用 Polars 滞后函数提取上一个快照的状态 (shift 1)
        ofi_calc = (
            l2_snapshots_df.lazy()
            .sort(["symbol", "timestamp"])
            .with_columns([
                pl.col("bid_price").shift(1).over(["symbol", "trade_date"]).alias("prev_bid_p"),
                pl.col("bid_vol").shift(1).over(["symbol", "trade_date"]).alias("prev_bid_v"),
                pl.col("ask_price").shift(1).over(["symbol", "trade_date"]).alias("prev_ask_p"),
                pl.col("ask_vol").shift(1).over(["symbol", "trade_date"]).alias("prev_ask_v")
            ])
            # 计算买方增量 I_bid
            .with_columns([
                pl.when(pl.col("bid_price") > pl.col("prev_bid_p"))
                .then(pl.col("bid_vol"))
                .when(pl.col("bid_price") == pl.col("prev_bid_p"))
                .then(pl.col("bid_vol") - pl.col("prev_bid_v"))
                .when(pl.col("bid_price") < pl.col("prev_bid_p"))
                .then(-pl.col("prev_bid_v"))
                .otherwise(0.0) # 每日第一帧作为基线
                .alias("i_bid"),

                # 计算卖方增量 I_ask
                pl.when(pl.col("ask_price") > pl.col("prev_ask_p"))
                .then(-pl.col("prev_ask_v"))
                .when(pl.col("ask_price") == pl.col("prev_ask_p"))
                .then(pl.col("ask_vol") - pl.col("prev_ask_v"))
                .when(pl.col("ask_price") < pl.col("prev_ask_p"))
                .then(pl.col("ask_vol"))
                .otherwise(0.0)
                .alias("i_ask")
            ])
            # 计算每个快照瞬间的微观 OFI
            .with_columns([
                (pl.col("i_bid") - pl.col("i_ask")).alias("snapshot_ofi"),
                (pl.col("bid_vol") + pl.col("ask_vol")).alias("depth_sum")
            ])
            .collect()
        )
        return ofi_calc

    def aggregate_to_daily_factors(self, ofi_snapshots_df: pl.DataFrame, trade_date: str | None = None) -> pl.DataFrame:
        """
        将日内数千个快照降维聚合成【日频低频化高价值 Alpha 因子】
        """
        if trade_date is not None:
            requested = date.fromisoformat(trade_date)
            if ofi_snapshots_df["trade_date"].unique().to_list() != [requested]:
                raise ValueError("指定日期必须与全部快照日期一致；多日输入请省略 trade_date")
        daily_features = (
            ofi_snapshots_df.group_by(["symbol", "trade_date"])
            .agg([
                # 1. 全天净累计 OFI
                pl.col("snapshot_ofi").sum().alias("daily_cum_ofi"),
                # 2. 全天挂单平均深度
                pl.col("depth_sum").mean().alias("avg_lob_depth"),
                # 3. 标准化相对 OFI 强度 (除以挂单深度)
                (pl.col("snapshot_ofi").sum() / (pl.col("depth_sum").mean() + 1e-5)).alias("factor_norm_ofi"),
                # 4. OFI 波动率 (日内订单流博弈剧烈度)
                pl.col("snapshot_ofi").std().alias("factor_ofi_volatility")
            ])
            .sort("factor_norm_ofi", descending=True)
        )
        return daily_features

# =========================================================================
# 单元验证测试：模拟全天 4,800 个 L2 快照推演
# =========================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n_snapshots = 4800 # 模拟全天完整 3秒一个的 L2 快照流
    print(f"[*] 正在生成高密度 Level-2 订单簿盘口快照仿真流 (共 {n_snapshots} 条快照)...")

    # 模拟两只标的：
    # 标的 A: 机构大单持续向上挂单吃筹 (买强卖弱)
    # 标的 B: 散户被动挂单，大单持续下砸出货 (卖强买弱)
    symbols = ["000001.SZ", "600519.SH"]
    all_records = []

    for sym in symbols:
        p_base = 100.0 if sym == "600519.SH" else 15.0
        # 标的 A 存在微弱向上漂移，标的 B 存在向下抛压
        drift = 0.00002 if sym == "600519.SH" else -0.00002

        curr_p = p_base
        for i in range(n_snapshots):
            curr_p += np.random.normal(drift, 0.01)
            bid_p = np.round(curr_p - 0.01, 2)
            ask_p = np.round(curr_p + 0.01, 2)
            
            # 茅台模拟多头主动盘口增挂
            bid_v = np.random.randint(100, 500) if sym == "600519.SH" else np.random.randint(50, 200)
            ask_v = np.random.randint(50, 200) if sym == "600519.SH" else np.random.randint(100, 500)

            all_records.append({
                "symbol": sym,
                "timestamp": datetime(2023, 10, 16, 9, 30) + timedelta(seconds=3*i + (5400 if i >= 2400 else 0)),
                "bid_price": bid_p,
                "bid_vol": float(bid_v),
                "ask_price": ask_p,
                "ask_vol": float(ask_v)
            })

    df_l2 = pl.DataFrame(all_records)

    # 运行高频特征引擎
    engine = AshareHighFreqOFIEngine()
    t0 = time.perf_counter()
    df_ofi_snaps = engine.compute_intraday_ofi(df_l2)
    daily_res = engine.aggregate_to_daily_factors(df_ofi_snaps, trade_date="2023-10-16")
    elapsed = time.perf_counter() - t0

    print(f"✅ 高频低频化聚合计算完毕！耗时: {elapsed:.3f} 秒")
    print("\n" + "=" * 80)
    print("日频标准化订单流不平衡 (Norm OFI) 降维提取结果：")
    print("=" * 80)
    print(daily_res)
