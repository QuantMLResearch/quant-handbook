"""第 4–8 讲共享契约。时间戳均为 Asia/Shanghai 本地无时区时间。"""
from datetime import date, datetime, time

import polars as pl


def close_time(value: str | date) -> datetime:
    day = date.fromisoformat(value) if isinstance(value, str) else value
    return datetime.combine(day, time(15))


def require_columns(frame: pl.DataFrame, columns: list[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"缺少字段: {sorted(missing)}")


def require_unique(frame: pl.DataFrame, keys: list[str]) -> None:
    require_columns(frame, keys)
    if frame.select(pl.any_horizontal(pl.col(keys).is_null()).any()).item():
        raise ValueError(f"主键不得为空: {keys}")
    if frame.select(keys).is_duplicated().any():
        raise ValueError(f"主键重复，必须明确消解冲突: {keys}")


def require_local_timestamp(frame: pl.DataFrame, column: str) -> None:
    require_columns(frame, [column])
    dtype = frame.schema[column]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone is not None:
        raise ValueError(f"{column} 必须是上海本地无时区 Datetime；请先转换时区")
    if frame[column].null_count():
        raise ValueError(f"{column} 不得为空")


def latest_state(events: pl.DataFrame, cutoff: datetime) -> pl.DataFrame:
    """状态事件只追加；同一生效日的更正以当时可见的最新版本为准。"""
    require_unique(events, ["symbol", "effective_date", "available_at"])
    require_local_timestamp(events, "available_at")
    return (
        events.filter((pl.col("effective_date") <= cutoff.date()) &
                      (pl.col("available_at") <= cutoff))
        .sort(["symbol", "effective_date", "available_at"])
        .unique(subset=["symbol"], keep="last", maintain_order=True)
    )
