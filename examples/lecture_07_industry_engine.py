"""行业 PIT：按当时可见的生效事件匹配，分类版本显式指定，未知行业单列。"""
import polars as pl

from examples.common import close_time, latest_state, require_columns, require_unique

SW_2021_NAMES = (
    "农林牧渔", "基础化工", "钢铁", "有色金属", "电子", "家用电器", "食品饮料",
    "纺织服饰", "轻工制造", "医药生物", "公用事业", "交通运输", "房地产",
    "商贸零售", "社会服务", "银行", "非银金融", "汽车", "机械设备", "国防军工",
    "计算机", "传媒", "通信", "煤炭", "石油石化", "环保", "美容护理", "电力设备",
    "建筑材料", "建筑装饰", "综合",
)


class AshareIndustryEngine:
    def __init__(self, taxonomy: str, industry_names=SW_2021_NAMES):
        self.taxonomy = taxonomy
        self.industry_names = tuple(industry_names)
        if len(set(self.industry_names)) != len(self.industry_names) or "未知" in self.industry_names:
            raise ValueError("行业字典重复或使用了保留名称“未知”")

    def build_industry_dummy_matrix(self, stock_universe_df: pl.DataFrame,
                                    industry_mapping_df: pl.DataFrame,
                                    trade_date: str) -> pl.DataFrame:
        require_unique(stock_universe_df, ["symbol"])
        require_columns(industry_mapping_df, ["taxonomy", "industry_name"])
        events = industry_mapping_df.filter(pl.col("taxonomy") == self.taxonomy)
        valid = latest_state(events, close_time(trade_date)).select("symbol", "industry_name")
        merged = stock_universe_df.join(valid, on="symbol", how="left", validate="1:1")
        if merged.filter(pl.col("industry_name").is_not_null() &
                         ~pl.col("industry_name").is_in(self.industry_names)).height:
            raise ValueError("行业名称不在所选版本字典中，请检查分类版本")
        merged = merged.with_columns(
            pl.col("industry_name").is_not_null().alias("industry_known"),
            pl.col("industry_name").fill_null("未知"),
        )
        return merged.with_columns([
            (pl.col("industry_name") == name).cast(pl.Int8).alias(f"ind_{name}")
            for name in (*self.industry_names, "未知")
        ])


if __name__ == "__main__":
    from examples.fixtures import sample_data
    data = sample_data()
    engine = AshareIndustryEngine("demo-v1")
    result = engine.build_industry_dummy_matrix(
        data["basics"].select("symbol"), data["industries"], "2023-04-14")
    columns = [c for c in result.columns if c.startswith("ind_")]
    assert (result.select(pl.sum_horizontal(columns)).to_series() == 1).all()
    print(result.select("symbol", "industry_name", "industry_known"))
