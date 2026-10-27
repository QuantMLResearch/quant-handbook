"""
lecture_10_fundamental_factor_engine.py
买方级基本面与杜邦质量成长因子向量化计算引擎
"""
import time
from datetime import date
import numpy as np
import polars as pl
import pandas as pd

class AshareFundamentalFactorEngine:
    def __init__(self):
        pass

    def compute_fundamental_factors(
        self,
        snapshot_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        输入必须包含经过 PIT 对齐的截面宽表:
        [symbol, trade_date, market_cap, net_profit_ttm, revenue_ttm, 
         total_equity, total_assets, ocf_ttm, net_profit_q, net_profit_ly_same_q]
        """
        print("[*] 正在计算买方基本面与杜邦质量特征矩阵...")
        t0 = time.perf_counter()

        factors_df = snapshot_df.lazy().with_columns([
            # ================= 1. 估值价值体系 (EP / BP / OCFP) =================
            # 市盈率倒数 EP (盈利收益率)
            (pl.col("net_profit_ttm") / pl.when(pl.col("market_cap") > 0).then(pl.col("market_cap")).otherwise(None)).alias("factor_ep_ttm"),
            # 市净率倒数 BP (账面市值比)
            (pl.col("total_equity") / pl.when(pl.col("market_cap") > 0).then(pl.col("market_cap")).otherwise(None)).alias("factor_bp"),
            # 经营现金流收益率 OCFP
            (pl.col("ocf_ttm") / pl.when(pl.col("market_cap") > 0).then(pl.col("market_cap")).otherwise(None)).alias("factor_ocfp"),

            # ================= 2. 杜邦质量分析三因子拆解 =================
            # 销售净利率 (Net Profit Margin)
            (pl.col("net_profit_ttm") / pl.when(pl.col("revenue_ttm") > 0).then(pl.col("revenue_ttm")).otherwise(None)).alias("dupont_margin"),
            # 总资产周转率 (Asset Turnover)
            (pl.col("revenue_ttm") / pl.when(pl.col("total_assets") > 0).then(pl.col("total_assets")).otherwise(None)).alias("dupont_turnover"),
            # 权益乘数 (Financial Leverage)
            (pl.col("total_assets") / pl.when(pl.col("total_equity") > 0).then(pl.col("total_equity")).otherwise(None)).alias("dupont_leverage"),
            # 最终 ROE 综合指标
            (pl.col("net_profit_ttm") / pl.when(pl.col("total_equity") > 0).then(pl.col("total_equity")).otherwise(None)).alias("roe_ttm"),

            # ================= 3. 盈余真实性与应计质量 (排雷因子) =================
            # 正利润公司的现金流/利润比；不能据此判定财务真实性
            (pl.col("ocf_ttm") / pl.when(pl.col("net_profit_ttm") > 0).then(pl.col("net_profit_ttm")).otherwise(None)).alias("cash_to_profit_ratio"),
            # 应计利润占总资产比重 (取负号，越低越好)
            (
                -(pl.col("net_profit_ttm") - pl.col("ocf_ttm")) / pl.when(pl.col("total_assets") > 0).then(pl.col("total_assets")).otherwise(None)
            ).alias("neg_accrual_ratio"),

            # ================= 4. 同季度同比增长（不是标准化 SUE） =================
            # 单季度净利润同比增速
            (
                (pl.col("net_profit_q") - pl.col("net_profit_ly_same_q")) /
                pl.when(pl.col("net_profit_ly_same_q").abs() > 0).then(pl.col("net_profit_ly_same_q").abs()).otherwise(None)
            ).alias("profit_yoy_growth")
        ]).collect()

        elapsed = time.perf_counter() - t0
        print(f"✅ 基本面特征矩阵计算完成！耗时: {elapsed:.3f} 秒")
        return factors_df

# =========================================================================
# 单元验证测试
# =========================================================================
if __name__ == "__main__":
    # 模拟构建 3 家商业模式完全不同的典型公司
    # 公司 A: 合成高利润企业 (超高净利率、高周转、低杠杆)
    # 公司 B: 某传统高周转超市 (薄利高销、高周转、中杠杆)
    # 公司 C: 某暴雷高杠杆房企 (亏损边缘、巨额负债加杠杆维持 ROE、现金流为负)
    
    mock_pit_data = pl.DataFrame([
        {
            "symbol": "DEMO_A", # 虚构标的
            "trade_date": date(2023, 10, 16),
            "market_cap": 22000.0, # 2.2 万亿
            "net_profit_ttm": 700.0,
            "revenue_ttm": 1400.0,
            "total_equity": 2100.0,
            "total_assets": 2600.0,
            "ocf_ttm": 780.0, # 现金流极其充沛
            "net_profit_q": 175.0,
            "net_profit_ly_same_q": 145.0
        },
        {
            "symbol": "DEMO_B", # 虚构零售企业
            "trade_date": date(2023, 10, 16),
            "market_cap": 300.0,
            "net_profit_ttm": 6.0,
            "revenue_ttm": 900.0,
            "total_equity": 80.0,
            "total_assets": 450.0,
            "ocf_ttm": 35.0,
            "net_profit_q": 1.5,
            "net_profit_ly_same_q": 1.125
        },
        {
            "symbol": "DEMO_C", # 虚构高杠杆企业
            "trade_date": date(2023, 10, 16),
            "market_cap": 200.0,
            "net_profit_ttm": 15.0,
            "revenue_ttm": 500.0,
            "total_equity": 30.0,  # 净资产极薄
            "total_assets": 1200.0, # 负债上万亿
            "ocf_ttm": -80.0, # 现金流枯竭
            "net_profit_q": 3.75,
            "net_profit_ly_same_q": 10.0
        }
    ])

    engine = AshareFundamentalFactorEngine()
    result = engine.compute_fundamental_factors(mock_pit_data)

    print("=" * 95)
    print("三类不同商业质量标的的杜邦拆解与基本面特征对比：")
    print("=" * 95)
    cols = ["symbol", "factor_ep_ttm", "roe_ttm", "dupont_margin", "dupont_turnover", "dupont_leverage", "cash_to_profit_ratio"]
    print(result.select(cols))
