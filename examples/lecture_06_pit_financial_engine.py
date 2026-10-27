"""财报 PIT：先按可用时刻过滤，再按报告期选取当时可见的更正版本。"""
from datetime import datetime

import polars as pl

from examples.common import close_time, require_columns, require_local_timestamp, require_unique


class AsharePITFinancialEngine:
    def snapshot(self, financials: pl.DataFrame, cutoff: datetime,
                 report_period=None) -> pl.DataFrame:
        require_unique(financials, ["symbol", "report_period", "available_at"])
        require_local_timestamp(financials, "available_at")
        visible = financials.filter(pl.col("available_at") <= cutoff)
        # 先还原每个报告期在查询时点可见的版本，保留原始披露和所有更正。
        versions = visible.sort(["symbol", "report_period", "available_at"]).unique(
            subset=["symbol", "report_period"], keep="last", maintain_order=True)
        if report_period is not None:
            return versions.filter(pl.col("report_period") == report_period)
        # 默认取最新报告期；旧年报的晚到更正不能覆盖更近的一季报。
        return versions.sort(["symbol", "report_period"]).unique(
            subset=["symbol"], keep="last", maintain_order=True)

    def align_prices_and_financials(self, daily_bars_df: pl.DataFrame,
                                    pit_financials_df: pl.DataFrame) -> pl.DataFrame:
        require_unique(daily_bars_df, ["symbol", "trade_date"])
        parts = []
        for day in sorted(daily_bars_df["trade_date"].unique()):
            cutoff = close_time(day)
            parts.append(daily_bars_df.filter(pl.col("trade_date") == day).join(
                self.snapshot(pit_financials_df, cutoff), on="symbol", how="left",
                validate="m:1").with_columns(pl.lit(cutoff).alias("decision_at")))
        if not parts:
            raise ValueError("行情为空")
        return pl.concat(parts).sort(["trade_date", "symbol"])

    def assert_zero_future_leakage(self, aligned_df: pl.DataFrame) -> None:
        require_columns(aligned_df, ["available_at", "decision_at"])
        if aligned_df.filter(pl.col("available_at") > pl.col("decision_at")).height:
            raise ValueError("检测到尚不可用的财务数据")


if __name__ == "__main__":
    from examples.fixtures import sample_data
    data = sample_data()
    engine = AsharePITFinancialEngine()
    result = engine.align_prices_and_financials(data["bars"], data["financials"])
    engine.assert_zero_future_leakage(result)
    print(result.filter((pl.col("symbol") == "OLD.SH") &
                       (pl.col("trade_date").is_in([
                           close_time("2023-03-31").date(), close_time("2023-04-26").date(),
                           close_time("2023-04-27").date()])))
          .select("trade_date", "report_period", "available_at", "roe"))
