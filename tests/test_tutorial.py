"""离线回归测试：只使用合成数据和本地临时目录。"""
import contextlib
import importlib
import io
import unittest
import tempfile
from pathlib import Path
from datetime import date, datetime
from polars.testing import assert_frame_equal

import numpy as np
import polars as pl


class OriginalRegressions(unittest.TestCase):
    def test_lecture_03_import_and_finite_prices(self):
        mod = importlib.import_module('examples.lecture_03_benchmark_massive')
        with contextlib.redirect_stdout(io.StringIO()):
            pandas_df, polars_df = mod.generate_massive_universe(4, 80)
        self.assertTrue(np.isfinite(pandas_df['close']).all())
        _, pandas_result = mod.run_pandas_factor_pipeline(pandas_df)
        _, polars_result = mod.run_polars_factor_pipeline(polars_df)
        for col in ['mom20', 'cs_rank']:
            np.testing.assert_allclose(pandas_result[col], polars_result[col].to_numpy(),
                                       equal_nan=True, atol=1e-12)

    def test_lecture_08_original_demo(self):
        import runpy
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_module('examples.lecture_08_pit_engine_practice', run_name='__main__')


class PITRegressions(unittest.TestCase):
    def setUp(self):
        from examples.fixtures import sample_data
        self.data = sample_data()

    def engine(self):
        from examples.lecture_08_pit_engine_practice import AsharePITEngine, EngineConfig
        d = self.data
        return AsharePITEngine(d['bars'], d['basics'], d['financials'], d['industries'],
                              d['calendar'], d['status'], EngineConfig(taxonomy='demo-v1'))

    def test_weekend_and_missing_bar_are_explicit_errors(self):
        with self.assertRaisesRegex(ValueError, '交易日历'):
            self.engine().get_market_snapshot('2023-04-15')
        self.data['bars'] = self.data['bars'].filter(
            ~((pl.col('symbol') == 'OLD.SH') & (pl.col('trade_date') == date(2023, 4, 14))))
        with self.assertRaisesRegex(ValueError, '缺少当日行情'):
            self.engine().get_market_snapshot('2023-04-14')

    def test_listing_age_is_independent_of_loaded_bars(self):
        full = self.engine().get_market_snapshot('2023-04-14')
        self.data['bars'] = self.data['bars'].filter(pl.col('trade_date') == date(2023, 4, 14))
        short = self.engine().get_market_snapshot('2023-04-14')
        assert_frame_equal(full.select('symbol', 'in_universe', 'trading_days_since_ipo'),
                           short.select('symbol', 'in_universe', 'trading_days_since_ipo'))
        self.assertTrue(short.filter(pl.col('symbol') == 'OLD.SH')['in_universe'][0])
        self.assertFalse(short.filter(pl.col('symbol') == 'NEW.SZ')['in_universe'][0])

    def test_st_effective_date_and_information_time(self):
        event = pl.DataFrame([dict(symbol='OLD.SH', effective_date=date(2023, 4, 17),
                                  available_at=datetime(2023, 4, 14, 20),
                                  is_st=True, is_delisting=False)])
        self.data['status'] = pl.concat([self.data['status'], event])
        before = self.engine().get_market_snapshot('2023-04-14', ['OLD.SH'])
        after = self.engine().get_market_snapshot('2023-04-17', ['OLD.SH'])
        self.assertTrue(before['in_universe'][0])
        self.assertFalse(after['in_universe'][0])

    def test_missing_status_does_not_default_to_tradable(self):
        self.data['status'] = self.data['status'].filter(pl.col('symbol') != 'OLD.SH')
        result = self.engine().get_market_snapshot('2023-04-14', ['OLD.SH'])
        self.assertFalse(result['status_known'][0])
        self.assertFalse(result['in_universe'][0])

    def test_after_close_report_is_not_available_same_day(self):
        before = self.engine().get_market_snapshot('2023-04-26', ['OLD.SH'])
        after = self.engine().get_market_snapshot('2023-04-27', ['OLD.SH'])
        self.assertEqual(before['report_period'][0], date(2022, 12, 31))
        self.assertEqual(after['report_period'][0], date(2023, 3, 31))

    def test_revision_preserves_history_and_does_not_downgrade_period(self):
        from examples.lecture_06_pit_financial_engine import AsharePITFinancialEngine
        engine = AsharePITFinancialEngine()
        financials = self.data['financials']
        before = engine.snapshot(financials, datetime(2023, 4, 26, 15), date(2022, 12, 31))
        after = engine.snapshot(financials, datetime(2023, 4, 27, 15), date(2022, 12, 31))
        self.assertEqual(before['roe'][0], .30)
        self.assertEqual(after['roe'][0], .25)
        self.assertEqual(engine.snapshot(financials, datetime(2023, 4, 27, 15))['roe'][0], .09)

    def test_future_injection_cannot_change_past_snapshot(self):
        engine = self.engine()
        original = engine.get_market_snapshot('2023-04-14')
        engine.bars = engine.bars.with_columns(
            pl.when(pl.col('trade_date') > date(2023, 4, 14)).then(1e9)
            .otherwise(pl.col('close_hfq')).alias('close_hfq'))
        engine.fins = engine.fins.with_columns(
            pl.when(pl.col('available_at') > datetime(2023, 4, 14, 15)).then(-999.)
            .otherwise(pl.col('roe')).alias('roe'))
        assert_frame_equal(original, engine.get_market_snapshot('2023-04-14'))

    def test_unknown_industry_and_conflicting_mapping(self):
        self.data['industries'] = self.data['industries'].filter(pl.col('symbol') != 'OLD.SH')
        result = self.engine().get_market_snapshot('2023-04-14', ['OLD.SH'])
        self.assertEqual(result['ind_未知'][0], 1)
        self.assertEqual(result['ind_综合'][0], 0)
        self.data['industries'] = pl.concat([self.data['industries'], self.data['industries'].head(1)])
        with self.assertRaisesRegex(ValueError, '主键重复'):
            self.engine().get_market_snapshot('2023-04-14')

    def test_short_calendar_rejected(self):
        self.data['calendar'] = self.data['calendar'].filter(pl.col('trade_date') >= date(2023, 3, 1))
        with self.assertRaisesRegex(ValueError, '覆盖所有上市日'):
            self.engine().get_market_snapshot('2023-04-14')

    def test_same_timestamp_two_reports_select_latest_period(self):
        self.data['financials'] = self.data['financials'].with_columns(
            pl.lit(datetime(2023, 4, 14, 10)).alias('available_at')).unique(
                subset=['symbol', 'report_period'], keep='first')
        result = self.engine().get_market_snapshot('2023-04-14', ['OLD.SH'])
        self.assertEqual(result['report_period'][0], date(2023, 3, 31))


class PipelineRegressions(unittest.TestCase):
    def setUp(self):
        from examples.fixtures import sample_data
        from examples.lecture_04_industrial_data_pipeline import IndustrialAshareDataPipeline
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pipeline = IndustrialAshareDataPipeline(self.tmp.name)
        self.bars = sample_data()['bars']

    def test_incremental_download_preserves_other_symbols(self):
        first = self.bars.filter(pl.col('symbol') == 'OLD.SH')
        second = self.bars.filter(pl.col('symbol') == 'NEW.SZ')
        self.pipeline.upsert_daily_bars(first)
        self.pipeline.upsert_daily_bars(second)
        self.pipeline.upsert_daily_bars(second)
        stored = pl.read_parquet(Path(self.tmp.name) / 'daily_bars/year=2023.parquet')
        self.assertEqual(stored.height, first.height + second.height)
        self.assertEqual(stored.select('symbol', 'trade_date').n_unique(), stored.height)

    def test_nonfinite_and_inverted_prices_rejected(self):
        for col, value in [('close', float('nan')), ('high', float('inf')), ('low', 1e9), ('volume', -1.)]:
            with self.subTest(col=col), self.assertRaises(ValueError):
                self.pipeline.upsert_daily_bars(self.bars.with_columns(pl.lit(value).alias(col)))
        self.assertFalse((Path(self.tmp.name) / 'daily_bars/year=2023.parquet').exists())

    def test_index_snapshots_preserve_history_and_whole_membership(self):
        before = pl.DataFrame(dict(symbol=['A.SH', 'B.SH'],
                                   asof_date=[date(2023, 4, 1)] * 2, weight=[.5, .5]))
        after = pl.DataFrame(dict(symbol=['A.SH', 'C.SH'],
                                  asof_date=[date(2023, 4, 2)] * 2, weight=[.4, .6]))
        self.pipeline.store_index_snapshot('demo', before, datetime(2023, 4, 3, 18))
        self.pipeline.store_index_snapshot('demo', after, datetime(2023, 4, 4, 18))
        with self.assertRaisesRegex(ValueError, '没有已观测'):
            self.pipeline.index_weights_asof('demo', datetime(2023, 4, 3, 15))
        old = self.pipeline.index_weights_asof('demo', datetime(2023, 4, 4, 15))
        new = self.pipeline.index_weights_asof('demo', datetime(2023, 4, 5, 15))
        self.assertEqual(set(old['symbol']), {'A.SH', 'B.SH'})
        self.assertEqual(set(new['symbol']), {'A.SH', 'C.SH'})

    def test_stock_listing_date_is_saved_and_retired_symbols_preserved(self):
        import pandas as pd
        class FakeAPI:
            def stock_individual_info_em(self, symbol):
                return pd.DataFrame({'item': ['股票简称', '上市时间'], 'value': ['示例股', 20010102]})
        self.pipeline.api = FakeAPI()
        self.pipeline.sync_stock_basics(['600001.SH'])
        path = Path(self.tmp.name) / 'metadata/stock_basic.parquet'
        existing = pl.read_parquet(path).with_columns(pl.lit(date(2023, 4, 1)).alias('delist_date'))
        existing.write_parquet(path)
        self.pipeline.sync_stock_basics(['000001.SZ'])
        result = self.pipeline.sync_stock_basics(['600001.SH'])
        self.assertEqual(result.height, 2)
        self.assertEqual(result.filter(pl.col('symbol') == '600001.SH')['list_date'][0], date(2001, 1, 2))
        self.assertEqual(result.filter(pl.col('symbol') == '600001.SH')['delist_date'][0], date(2023, 4, 1))

    def test_api_adapter_keeps_raw_and_adjusted_prices_separate(self):
        import pandas as pd
        class FakeAPI:
            def stock_zh_a_hist(self, **kwargs):
                if kwargs['adjust'] == 'hfq':
                    return pd.DataFrame({'日期': [date(2023, 4, 14)], '收盘': [110.]})
                return pd.DataFrame({'日期': [date(2023, 4, 14)], '开盘': [10.],
                                     '最高': [12.], '最低': [9.], '收盘': [11.],
                                     '成交量': [2.], '成交额': [2200.]})
        self.pipeline.api = FakeAPI()
        self.pipeline.download_and_partition_daily_bars(['600001.SH'])
        frame = pl.read_parquet(Path(self.tmp.name) / 'daily_bars/year=2023.parquet')
        self.assertEqual(frame['close'][0], 11.)
        self.assertEqual(frame['close_hfq'][0], 110.)
        self.assertEqual(frame['volume'][0], 200.)


class BacktestAccountingRegressions(unittest.TestCase):
    def test_liquidation_still_charges_costs_when_all_positions_closed(self):
        import pandas as pd
        from examples.lecture_02_backtest_sanitizer import AshareBacktestSanitizer
        bars = pd.DataFrame(dict(
            trade_date=pd.date_range('2023-04-10', periods=3), symbol=['A'] * 3,
            open=[100., 100., 90.], close=[100., 90., 90.], returns=[.1, -.1, 0.],
            is_limit_up=[False] * 3, is_limit_down=[False] * 3))
        result = AshareBacktestSanitizer().run_backtest_engine(bars)
        self.assertLess(result['returns_series'][-1], 0.)

    def test_buy_does_not_receive_pre_open_gap(self):
        import pandas as pd
        from examples.lecture_02_backtest_sanitizer import AshareBacktestSanitizer, TradeExecutionRules
        rules = TradeExecutionRules(one_way_commission=0., stamp_duty=0., slippage_bps=0.)
        bars = pd.DataFrame(dict(
            trade_date=pd.date_range('2023-04-10', periods=2), symbol=['A'] * 2,
            open=[100., 110.], close=[100., 110.], returns=[.1, .1],
            is_limit_up=[False] * 2, is_limit_down=[False] * 2))
        result = AshareBacktestSanitizer(rules).run_backtest_engine(bars)
        self.assertAlmostEqual(result['cumulative_return'], 0.)

    def test_dsr_requires_trial_dispersion_for_multiple_tests(self):
        from examples.lecture_02_backtest_sanitizer import AshareBacktestSanitizer
        returns = np.array([.01, -.02, .03, -.01, .02])
        with self.assertRaises(ValueError):
            AshareBacktestSanitizer.calculate_deflated_sharpe_ratio(returns, num_trials=100)


if __name__ == '__main__':
    unittest.main()
