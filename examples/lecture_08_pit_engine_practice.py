"""整合第 5–7 讲的收盘截面参考实现；信号最早在下一交易时段执行。"""
from dataclasses import dataclass
from datetime import date

import polars as pl

from examples.common import close_time, require_columns, require_unique
from examples.lecture_05_dynamic_universe_builder import DynamicUniverseBuilder, UniverseFilterConfig
from examples.lecture_06_pit_financial_engine import AsharePITFinancialEngine
from examples.lecture_07_industry_engine import AshareIndustryEngine


@dataclass(frozen=True)
class EngineConfig:
    min_listed_days: int = 60
    exclude_st: bool = True
    exclude_suspended: bool = True
    taxonomy: str = "SW2021"


class AsharePITEngine:
    def __init__(self, daily_bars, stock_basics, pit_financials, industry_mapping,
                 trade_calendar, status_history, config: EngineConfig | None = None):
        self.cfg = config or EngineConfig()
        self.bars = daily_bars
        self.basics = stock_basics
        self.fins = pit_financials
        self.industries = industry_mapping
        self.calendar = trade_calendar
        self.status = status_history
        self.universe = DynamicUniverseBuilder(UniverseFilterConfig(
            self.cfg.min_listed_days, self.cfg.exclude_st, self.cfg.exclude_suspended))
        self.financial_engine = AsharePITFinancialEngine()
        self.industry_engine = AshareIndustryEngine(self.cfg.taxonomy)

    def get_market_snapshot(self, trade_date: str, symbols: list[str] | None = None) -> pl.DataFrame:
        day = date.fromisoformat(trade_date)
        if day not in self.calendar["trade_date"].to_list():
            raise ValueError(f"{day} 不在输入交易日历中；不会自动回退到前一日")
        require_columns(self.bars, ["symbol", "trade_date", "close_hfq", "volume"])
        basics = self.basics.filter((pl.col("list_date") <= day) &
                                   (pl.col("delist_date").is_null() | (pl.col("delist_date") > day)))
        if symbols is not None:
            missing = set(symbols) - set(basics["symbol"])
            if missing:
                raise ValueError(f"查询日不存在有效上市档案: {sorted(missing)}")
            basics = basics.filter(pl.col("symbol").is_in(symbols))
        bars = self.bars.filter((pl.col("trade_date") <= day) &
                               pl.col("symbol").is_in(basics["symbol"].to_list()))
        require_unique(bars, ["symbol", "trade_date"])
        today = bars.filter(pl.col("trade_date") == day)
        if today.is_empty():
            raise ValueError(f"{day} 没有行情数据")
        if basics.join(today, on="symbol", how="anti").height:
            raise ValueError("上市股票缺少当日行情；停牌应显式提供零成交量记录")
        universe = self.universe.build_dynamic_universe(today, basics, self.calendar, self.status)
        features = (
            bars.sort(["symbol", "trade_date"]).with_columns(
                (pl.col("close_hfq") / pl.col("close_hfq").shift(20).over("symbol") - 1).alias("mom_20d"),
                (pl.col("close_hfq").pct_change().rolling_std(20).over("symbol") * 252**.5).alias("vol_20d"),
            ).filter(pl.col("trade_date") == day)
            .select("symbol", "mom_20d", "vol_20d")
        )
        financials = self.financial_engine.snapshot(self.fins, close_time(day))
        industries = self.industry_engine.build_industry_dummy_matrix(
            today.select("symbol"), self.industries, trade_date)
        result = (universe.join(features, on="symbol", validate="1:1")
                  .join(financials, on="symbol", how="left", validate="1:1")
                  .join(industries, on="symbol", how="left", validate="1:1")
                  .with_columns(pl.lit(close_time(day)).alias("decision_at")))
        self.financial_engine.assert_zero_future_leakage(result)
        return result.sort("symbol")

    @staticmethod
    def run_lookahead_penetration_tests(engine: "AsharePITEngine"):
        snapshot = engine.get_market_snapshot("2023-04-14")
        old = snapshot.filter(pl.col("symbol") == "OLD.SH")
        assert old["report_period"][0] == date(2022, 12, 31)
        assert old["in_universe"][0]
        assert not snapshot.filter(pl.col("symbol") == "HALT.SZ")["in_universe"][0]
        try:
            engine.get_market_snapshot("2023-04-15")
        except ValueError as error:
            assert "交易日历" in str(error)
        else:
            raise AssertionError("非交易日必须显式拒绝")
        print(snapshot.select("symbol", "trade_date", "in_universe", "report_period", "industry_name"))
        print("合成样例通过；完整边界验证请运行 python -m unittest discover -s tests -v")


if __name__ == "__main__":
    from examples.fixtures import sample_data
    data = sample_data()
    engine = AsharePITEngine(data["bars"], data["basics"], data["financials"],
                            data["industries"], data["calendar"], data["status"],
                            EngineConfig(taxonomy="demo-v1"))
    AsharePITEngine.run_lookahead_penetration_tests(engine)
