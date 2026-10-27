"""小型确定性合成数据。证券、财报数值、状态均为教学虚构，不代表真实市场。"""
from datetime import date, datetime, timedelta

import polars as pl


def sample_data() -> dict[str, pl.DataFrame]:
    # 显式标注为模拟日历；生产环境必须换成经核验的交易所日历。
    days = [date(2023, 1, 2) + timedelta(days=i) for i in range(124)]
    days = [day for day in days if day.weekday() < 5]
    calendar = pl.DataFrame({"trade_date": days})
    basics = pl.DataFrame({
        "symbol": ["OLD.SH", "HALT.SZ", "NEW.SZ"],
        "list_date": [date(2023, 1, 2), date(2023, 1, 2), date(2023, 4, 10)],
        "delist_date": [None, None, None],
    }, schema_overrides={"delist_date": pl.Date})
    rows = []
    for symbol, listed, _ in basics.iter_rows():
        for i, day in enumerate(days):
            if day < max(listed, date(2023, 3, 1)):
                continue  # 行情仅从 3 月加载；上市天数仍应从 1 月计算。
            volume = 0.0 if symbol == "HALT.SZ" and date(2023, 4, 10) <= day <= date(2023, 4, 20) else 1000.0
            rows.append({"symbol": symbol, "trade_date": day, "open": 100.0 + i,
                         "high": 102.0 + i, "low": 99.0 + i, "close": 101.0 + i,
                         "close_hfq": 101.0 + i, "volume": volume, "turnover": volume * (101.0 + i)})
    status = pl.DataFrame([
        {"symbol": symbol, "effective_date": listed,
         "available_at": datetime.combine(listed, datetime.min.time()),
         "is_st": False, "is_delisting": False}
        for symbol, listed, _ in basics.iter_rows()
    ])
    financials = pl.DataFrame([
        {"symbol": "OLD.SH", "report_period": date(2022, 12, 31),
         "available_at": datetime(2023, 3, 31, 20), "roe": .30, "net_profit_yoy": .19},
        {"symbol": "OLD.SH", "report_period": date(2023, 3, 31),
         "available_at": datetime(2023, 4, 26, 20), "roe": .09, "net_profit_yoy": .20},
        # 较旧报告期的晚到更正不应把当前一季报覆盖成年报。
        {"symbol": "OLD.SH", "report_period": date(2022, 12, 31),
         "available_at": datetime(2023, 4, 27, 10), "roe": .25, "net_profit_yoy": .15},
    ])
    industries = pl.DataFrame([
        {"symbol": symbol, "effective_date": listed,
         "available_at": datetime.combine(listed, datetime.min.time()),
         "taxonomy": "demo-v1", "industry_name": industry}
        for (symbol, listed, _), industry in zip(basics.iter_rows(), ["食品饮料", "房地产", "电子"])
    ])
    return dict(calendar=calendar, basics=basics, bars=pl.DataFrame(rows),
                status=status, financials=financials, industries=industries)
