"""AkShare 接入适配与 Parquet 落库。默认不联网，使用 --fetch 才下载样本。"""
from datetime import datetime
from pathlib import Path
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import polars as pl

from examples.common import require_columns, require_unique


def local_now() -> datetime:
    return datetime.now(ZoneInfo('Asia/Shanghai')).replace(tzinfo=None)


def atomic_write(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        frame.write_parquet(temporary, compression='snappy')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class IndustrialAshareDataPipeline:
    def __init__(self, data_root='data_lake', api=None):
        self.root = Path(data_root)
        self.api = api
        self.meta_dir = self.root / 'metadata'
        self.daily_dir = self.root / 'daily_bars'
        self.index_dir = self.root / 'index_constituents'

    def _api(self):
        if self.api is None:
            import akshare
            self.api = akshare
        return self.api

    @staticmethod
    def fetch_with_retry(fetch_func, max_retries=3, delay=1.0, **kwargs):
        for attempt in range(max_retries):
            try:
                return fetch_func(**kwargs)
            except Exception:
                if attempt + 1 == max_retries:
                    raise
                time.sleep(delay * 2**attempt)
        raise ValueError('max_retries 必须大于零')

    def sync_trading_calendar(self) -> pl.DataFrame:
        # 数据来源为新浪适配接口；用于生产前应与交易所日历核验覆盖期。
        raw = self.fetch_with_retry(self._api().tool_trade_date_hist_sina)
        calendar = pl.from_pandas(raw).select(pl.col('trade_date').cast(pl.Date)).unique().sort('trade_date')
        require_unique(calendar, ['trade_date'])
        if calendar.is_empty():
            raise ValueError('日历为空')
        atomic_write(calendar, self.meta_dir / 'trade_calendar.parquet')
        return calendar

    def sync_stock_basics(self, symbols: list[str]) -> pl.DataFrame:
        # 显式传入带市场后缀的股票；不把当前上市名录冒充历史全市场全集。
        rows = []
        for symbol in symbols:
            code, exchange = symbol.split('.')
            if exchange not in {'SH', 'SZ', 'BJ'} or len(code) != 6 or not code.isdigit():
                raise ValueError(f'股票代码需包含正确交易所后缀: {symbol}')
            raw = self.fetch_with_retry(self._api().stock_individual_info_em, symbol=code)
            info = dict(zip(raw['item'], raw['value']))
            listed = datetime.strptime(str(info['上市时间']).removesuffix('.0'), '%Y%m%d').date()
            rows.append(dict(symbol=symbol, display_name=str(info['股票简称']),
                             list_date=listed, delist_date=None))
        result = pl.DataFrame(rows, schema_overrides={'list_date': pl.Date, 'delist_date': pl.Date})
        require_unique(result, ['symbol'])
        path = self.meta_dir / 'stock_basic.parquet'
        if path.exists():
            existing = pl.read_parquet(path)
            # 不删除旧股票，也不清除人工补入的退市日期。
            existing_dates = existing.select('symbol', pl.col('delist_date').alias('_known_delist_date'))
            result = result.join(existing_dates, on='symbol', how='left', validate='1:1').with_columns(
                pl.coalesce('delist_date', '_known_delist_date').alias('delist_date')).drop('_known_delist_date')
            result = pl.concat([existing.join(result.select('symbol'), on='symbol', how='anti'), result])
        atomic_write(result.sort('symbol'), path)
        return result

    def store_index_snapshot(self, index_code: str, weights: pl.DataFrame,
                             observed_at: datetime) -> Path:
        require_unique(weights, ['symbol'])
        require_columns(weights, ['asof_date', 'weight'])
        if weights.is_empty() or weights['asof_date'].n_unique() != 1:
            raise ValueError('一次写入必须是同一日期的完整成分股快照')
        if weights['asof_date'].null_count() or weights['asof_date'].max() > observed_at.date():
            raise ValueError('快照日期为空或晚于观测时间')
        if observed_at.tzinfo is not None:
            raise ValueError('observed_at 需先转换为上海本地无时区时间')
        if weights.filter(pl.col('weight').is_null() | ~pl.col('weight').is_finite() |
                          (pl.col('weight') < 0) | (pl.col('weight') > 1)).height:
            raise ValueError('权重需是 0–1 范围内的有限数值')
        batch = weights.with_columns(pl.lit(observed_at).alias('available_at'))
        # 新观测另存一份；历史文件不会被新版快照覆盖。
        path = self.index_dir / index_code / f'{observed_at:%Y%m%dT%H%M%S%f}-{uuid4().hex[:8]}.parquet'
        atomic_write(batch, path)
        return path

    def sync_index_weights(self, index_code='000852') -> Path:
        raw = self.fetch_with_retry(self._api().index_stock_cons_weight_csindex, symbol=index_code)
        frame = pl.from_pandas(raw)
        suffixes = {'上海证券交易所': 'SH', '深圳证券交易所': 'SZ', '北京证券交易所': 'BJ'}
        rows = []
        for row in frame.to_dicts():
            if row['交易所'] not in suffixes:
                raise ValueError(f"未识别的交易所: {row['交易所']}")
            code = str(row['成分券代码']).removesuffix('.0').zfill(6)
            rows.append(dict(symbol=f"{code}.{suffixes[row['交易所']]}",
                             asof_date=row['日期'], weight=float(row['权重']) / 100))
        weights = pl.DataFrame(rows).with_columns(pl.col('asof_date').cast(pl.Date))
        return self.store_index_snapshot(index_code, weights, local_now())

    def index_weights_asof(self, index_code: str, cutoff: datetime) -> pl.DataFrame:
        candidates = []
        for path in sorted((self.index_dir / index_code).glob('*.parquet')):
            frame = pl.read_parquet(path)
            available_at = frame['available_at'][0]
            asof_date = frame['asof_date'][0]
            if available_at <= cutoff and asof_date <= cutoff.date():
                candidates.append((asof_date, available_at, frame))
        if not candidates:
            raise ValueError('查询时点没有已观测的指数快照；不能回填今日成分股')
        # 选择整批快照，而非逐股票取最新，避免保留已被剔除的成分股。
        return max(candidates, key=lambda item: item[:2])[2]

    @staticmethod
    def validate_bars(frame: pl.DataFrame) -> None:
        require_unique(frame, ['symbol', 'trade_date'])
        prices = ['open', 'high', 'low', 'close', 'close_hfq']
        require_columns(frame, [*prices, 'volume', 'turnover'])
        invalid = [pl.col(c).is_null() | ~pl.col(c).is_finite() | (pl.col(c) <= 0) for c in prices]
        invalid += [pl.col(c).is_null() | ~pl.col(c).is_finite() | (pl.col(c) < 0)
                    for c in ['volume', 'turnover']]
        invalid += [pl.col('high') < pl.max_horizontal('open', 'close', 'low'),
                    pl.col('low') > pl.min_horizontal('open', 'close', 'high')]
        if frame.filter(pl.any_horizontal(invalid)).height:
            raise ValueError('价格、成交量、成交额非有限值或 OHLC 关系异常')

    def upsert_daily_bars(self, frame: pl.DataFrame) -> None:
        self.validate_bars(frame)
        for year in sorted(frame['trade_date'].dt.year().unique()):
            incoming = frame.filter(pl.col('trade_date').dt.year() == year)
            path = self.daily_dir / f'year={year}.parquet'
            if path.exists():
                old = pl.read_parquet(path)
                old = old.join(incoming.select('symbol', 'trade_date'),
                               on=['symbol', 'trade_date'], how='anti')
                incoming = pl.concat([old, incoming])
            self.validate_bars(incoming)
            atomic_write(incoming.sort(['symbol', 'trade_date']), path)

    def download_and_partition_daily_bars(self, symbols, start_year=2023, end_year=2024):
        frames = []
        for symbol in symbols:
            kwargs = dict(symbol=symbol.split('.')[0], period='daily',
                          start_date=f'{start_year}0101', end_date=f'{end_year}1231')
            raw = pl.from_pandas(self.fetch_with_retry(self._api().stock_zh_a_hist, **kwargs, adjust=''))
            adjusted = pl.from_pandas(self.fetch_with_retry(self._api().stock_zh_a_hist, **kwargs, adjust='hfq'))
            if raw.is_empty() or adjusted.is_empty():
                raise ValueError(f'{symbol} 行情为空；本批次尚未写入')
            raw = raw.rename({'日期': 'trade_date', '开盘': 'open', '最高': 'high', '最低': 'low',
                              '收盘': 'close', '成交量': 'volume', '成交额': 'turnover'})
            raw = raw.select(pl.lit(symbol).alias('symbol'), pl.col('trade_date').cast(pl.Date),
                             pl.col('open', 'high', 'low', 'close', 'turnover').cast(pl.Float64),
                             (pl.col('volume').cast(pl.Float64) * 100).alias('volume'))  # 手 -> 股
            adjusted = adjusted.select(pl.col('日期').cast(pl.Date).alias('trade_date'),
                                       pl.col('收盘').cast(pl.Float64).alias('close_hfq'))
            require_unique(adjusted, ['trade_date'])
            frames.append(raw.join(adjusted, on='trade_date', how='left', validate='1:1'))
        if not frames:
            raise ValueError('股票列表为空')
        self.upsert_daily_bars(pl.concat(frames))

    def assert_data_integrity(self, file_path: Path):
        self.validate_bars(pl.read_parquet(file_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help='联网下载四只样本股；不代表历史全市场数据')
    args = parser.parse_args()
    if args.fetch:
        pipeline = IndustrialAshareDataPipeline()
        symbols = ['600519.SH', '300750.SZ', '000001.SZ', '688981.SH']
        pipeline.sync_trading_calendar()
        pipeline.sync_stock_basics(symbols)
        pipeline.sync_index_weights()
        pipeline.download_and_partition_daily_bars(symbols)
        print('样本已写入 data_lake；历史状态、退市档案、财报和行业事件须另外接入。')
    else:
        print('默认不联网。使用 --fetch 下载样本；使用 unittest 运行离线回归。')
