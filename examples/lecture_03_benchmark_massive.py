"""
lecture_03_benchmark_massive.py
A 股全市场千万级面板数据：Pandas vs Polars 终极性能压测
"""
import time
from typing import Tuple
import numpy as np
import pandas as pd
import polars as pl

def generate_massive_universe(n_stocks: int = 2500, n_days: int = 800) -> Tuple[pd.DataFrame, pl.DataFrame]:
    """
    生成 200 万行高密度 A 股面板仿真数据 (兼顾普通笔记本内存压力)
    """
    total_records = n_stocks * n_days
    print(f"[*] 正在生成规模为 {total_records:,} 行的仿真 A 股数据集...")
    np.random.seed(42)
    
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    stock_ids = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    
    # 批量构建连续内存数据
    all_symbols = np.repeat(stock_ids, n_days)
    all_dates = np.tile(dates, n_stocks)
    # 几何布朗运动随机游走价格
    returns = np.random.normal(0.0004, 0.02, size=total_records)
    # 价格序列
    # 每只股票独立累积，避免跨股票串联导致指数溢出。
    prices = (10.0 * np.exp(np.cumsum(returns.reshape(n_stocks, n_days), axis=1))).ravel()

    df_pd = pd.DataFrame({
        "symbol": all_symbols,
        "trade_date": all_dates,
        "close": prices
    })
    
    print("[*] 正在将数据无缝映射至 Apache Arrow 格式 (Polars)...")
    df_pl = pl.from_pandas(df_pd)
    print(f"✅ 数据准备就绪！内存行数: {len(df_pd):,} 行\n")
    return df_pd, df_pl

def run_pandas_factor_pipeline(df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """
    Pandas 方案：
    1. 按股票分组计算 20 日动量
    2. 按交易日分组计算截面 Rank
    """
    t0 = time.perf_counter()
    
    # 1. 时序动量 (基于股票分组)
    df["mom20"] = df.groupby("symbol")["close"].pct_change(periods=20)
    
    # 2. 截面百分比排名 (基于交易日分组)
    df["cs_rank"] = df.groupby("trade_date")["mom20"].rank(pct=True)
    
    elapsed = time.perf_counter() - t0
    return elapsed, df

def run_polars_factor_pipeline(df: pl.DataFrame) -> Tuple[float, pl.DataFrame]:
    """
    Polars 方案：
    利用 Lazy 计算图优化 + Rayon 多核并行原生表达式
    """
    t0 = time.perf_counter()
    
    result = (
        df.lazy()
        .with_columns([
            # 时序动量计算
            (pl.col("close") / pl.col("close").shift(20).over("symbol") - 1.0).alias("mom20")
        ])
        .with_columns([
            # 截面百分比排名计算 (rank / count)
            (
                pl.col("mom20").rank(method="average").over("trade_date") /
                pl.col("mom20").count().over("trade_date")
            ).alias("cs_rank")
        ])
        .collect() # 触发多线程执行引擎
    )
    
    elapsed = time.perf_counter() - t0
    return elapsed, result

if __name__ == "__main__":
    print("=" * 70)
    print("现代化量化算力对决：Pandas vs Polars 性能与吞吐量压测")
    print("=" * 70)
    
    # 启动数据生成
    df_pd, df_pl = generate_massive_universe(n_stocks=2000, n_days=500) # 100 万行基准压测
    
    print(">>> 正在启动 Pandas 计算流水线 (单线程执行)...")
    time_pd, res_pd = run_pandas_factor_pipeline(df_pd)
    print(f"  Pandas 耗时 : {time_pd:.3f} 秒")
    
    print("\n>>> 正在启动 Polars 现代化并发流水线 (多核并行执行)...")
    time_pl, res_pl = run_polars_factor_pipeline(df_pl)
    print(f"  Polars 耗时 : {time_pl:.3f} 秒")
    
    speedup = time_pd / (time_pl + 1e-8)
    print("-" * 70)
    for column in ["mom20", "cs_rank"]:
        np.testing.assert_allclose(res_pd[column], res_pl[column].to_numpy(), equal_nan=True)
    print(f"结果一致；本次 Pandas / Polars 耗时比：{speedup:.2f}（小于 1 表示 Pandas 更快）")
    print("该比值仅适用于本次数据规模、硬件、版本和计算任务。")
    print("=" * 70)
