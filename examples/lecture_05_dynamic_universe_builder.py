"""日级股票池：上市日 + 交易日历 + 历史状态事件；查询口径为收盘后。"""
from dataclasses import dataclass
from datetime import date

import polars as pl

from examples.common import close_time, latest_state, require_columns, require_unique


@dataclass(frozen=True)
class UniverseFilterConfig:
    min_listed_days: int = 60
    exclude_st: bool = True
    exclude_suspended: bool = True


class DynamicUniverseBuilder:
    def __init__(self, config: UniverseFilterConfig | None = None):
        self.cfg = config or UniverseFilterConfig()
        if self.cfg.min_listed_days < 0:
            raise ValueError("min_listed_days 不得为负")

    def build_dynamic_universe(self, daily_bars_df: pl.DataFrame,
                               stock_basics_df: pl.DataFrame,
                               trade_calendar: pl.DataFrame,
                               status_history: pl.DataFrame) -> pl.DataFrame:
        require_unique(daily_bars_df, ["symbol", "trade_date"])
        require_unique(stock_basics_df, ["symbol"])
        require_columns(stock_basics_df, ["list_date", "delist_date"])
        require_columns(status_history, ["is_st", "is_delisting"])
        require_columns(daily_bars_df, ["volume"])
        require_unique(trade_calendar, ["trade_date"])
        if trade_calendar.is_empty():
            raise ValueError("交易日历不能为空")
        calendar = trade_calendar.select("trade_date").sort("trade_date")
        sessions = calendar["trade_date"].to_list()
        if stock_basics_df["list_date"].null_count():
            raise ValueError("必须提供真实上市日，不能用行情起点代替")
        if stock_basics_df.filter(pl.col("list_date") < sessions[0]).height:
            raise ValueError("交易日历必须覆盖所有上市日；请补齐日历")
        if daily_bars_df.join(calendar, on="trade_date", how="anti").height:
            raise ValueError("行情包含日历未覆盖的日期或非交易日")
        if daily_bars_df.join(stock_basics_df, on="symbol", how="anti").height:
            raise ValueError("存在缺少上市档案的股票")

        # 按真实交易所开市日计数，停牌日仍计入上市年龄；与加载多少根 Bar 无关。
        from bisect import bisect_left, bisect_right
        results = []
        for day in sorted(daily_bars_df["trade_date"].unique()):
            state = latest_state(status_history, close_time(day)).select(
                "symbol", "is_st", "is_delisting")
            panel = daily_bars_df.filter(pl.col("trade_date") == day).join(
                stock_basics_df, on="symbol", how="left", validate="m:1")
            if panel.filter(pl.col("list_date") > day).height:
                raise ValueError("存在上市前行情")
            ages = [bisect_right(sessions, day) - bisect_left(sessions, listed)
                    for listed in panel["list_date"]]
            panel = panel.with_columns(pl.Series("trading_days_since_ipo", ages)).join(
                state, on="symbol", how="left", validate="m:1")
            panel = panel.with_columns(
                (pl.col("is_st").is_not_null() & pl.col("is_delisting").is_not_null()).alias("status_known"),
                (pl.col("volume").is_null() | ~pl.col("volume").is_finite() |
                 (pl.col("volume") <= 0)).alias("is_suspended"),
            ).with_columns((
                (pl.col("trading_days_since_ipo") > self.cfg.min_listed_days) &
                (pl.col("delist_date").is_null() | (pl.col("delist_date") > day)) &
                pl.col("status_known") & ~pl.col("is_delisting").fill_null(True) &
                (~pl.col("is_st").fill_null(True) if self.cfg.exclude_st else pl.lit(True)) &
                (~pl.col("is_suspended") if self.cfg.exclude_suspended else pl.lit(True))
            ).alias("in_universe"))
            results.append(panel)
        if not results:
            raise ValueError("没有行情可供构建股票池")
        return pl.concat(results).sort(["trade_date", "symbol"])

    def generate_coverage_report(self, universe_df: pl.DataFrame) -> pl.DataFrame:
        return universe_df.group_by("trade_date").agg(
            pl.len().alias("total_stocks_recorded"),
            pl.col("in_universe").sum().alias("valid_universe_count"),
            (~pl.col("status_known")).sum().alias("unknown_status_count"),
        ).sort("trade_date")


if __name__ == "__main__":
    from examples.fixtures import sample_data
    data = sample_data()
    result = DynamicUniverseBuilder().build_dynamic_universe(
        data["bars"], data["basics"], data["calendar"], data["status"])
    day = result.filter(pl.col("trade_date") == date(2023, 4, 14))
    print(day.select("symbol", "trading_days_since_ipo", "in_universe"))
    assert day.filter(pl.col("symbol") == "OLD.SH")["in_universe"][0]
    assert not day.filter(pl.col("symbol") == "NEW.SZ")["in_universe"][0]
    assert not day.filter(pl.col("symbol") == "HALT.SZ")["in_universe"][0]
