"""第 9–17 讲的数值回归；使用小型合成面板，不访问行情服务。"""
import contextlib
import importlib
import io
import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl


def lecture(n, name):
    return importlib.import_module(f"examples.lecture_{n:02d}_{name}")


class FeatureRegressions(unittest.TestCase):
    def test_corporate_action_does_not_create_returns(self):
        engine = lecture(9, 'price_volume_factors').AsharePriceVolumeFactorEngine()
        prices = [100.] * 15 + [50.] * 15
        bars = pl.DataFrame(dict(symbol=['A'] * 30, trade_date=list(range(30)),
                                 open=prices, high=prices, low=prices, close=prices,
                                 close_hfq=[100.] * 30, volume=[1000.] * 30,
                                 turnover=[100000.] * 30))
        with contextlib.redirect_stdout(io.StringIO()):
            result = engine.compute_all_factors(bars)
        for col, idx in [('ret_1d', 15), ('ret_overnight', 15), ('mom_20d', -1),
                         ('volatility_20d', -1), ('amihud_illiq_20d', -1)]:
            self.assertAlmostEqual(result[col][idx], 0.)

    def test_quarter_growth_uses_comparable_periods(self):
        engine = lecture(10, 'fundamental_factors').AshareFundamentalFactorEngine()
        row = dict(symbol='A', trade_date='2023-04-14', market_cap=1000.,
                   net_profit_ttm=400., revenue_ttm=1000., total_equity=1000.,
                   total_assets=2000., ocf_ttm=400., net_profit_q=100.,
                   net_profit_ly_same_q=100.)
        result = engine.compute_fundamental_factors(pl.DataFrame([row]))
        self.assertAlmostEqual(result['profit_yoy_growth'][0], 0.)
        self.assertNotIn('sue_profit_growth_proxy', result.columns)

    def test_ofi_all_price_branches_and_session_boundary(self):
        engine = lecture(11, 'high_freq_to_low_freq').AshareHighFreqOFIEngine()
        rows = []
        # (bid, bid size, ask, ask size, expected event OFI)
        cases = [(99., 80., 101., 70., -100.), (100., 100., 100.5, 60., -60.),
                 (100.5, 80., 101., 70., 80.), (100., 100., 102., 60., 70.),
                 (100., 110., 101., 60., 20.)]
        for i, (bp, bv, ap, av, expected) in enumerate(cases):
            for t, values in enumerate([(100., 100., 101., 70.), (bp, bv, ap, av)]):
                rows.append(dict(symbol=str(i), timestamp=datetime(2025, 1, 2, 10) + timedelta(seconds=t),
                                 bid_price=values[0], bid_vol=values[1], ask_price=values[2], ask_vol=values[3]))
        result = engine.compute_intraday_ofi(pl.DataFrame(rows))
        for i, case in enumerate(cases):
            self.assertAlmostEqual(result.filter(pl.col('symbol') == str(i))['snapshot_ofi'][1], case[-1])
        next_day = pl.DataFrame([dict(rows[0], timestamp=datetime(2025, 1, 3, 10), bid_price=90.)])
        result = engine.compute_intraday_ofi(pl.concat([pl.DataFrame(rows[:2]), next_day]))
        self.assertEqual(result['snapshot_ofi'][-1], 0.)

    def test_preprocess_imputation_and_weighted_orthogonality(self):
        p = lecture(12, 'factor_preprocessing_pipeline').AshareFactorPreprocessor()
        filled = p.impute_industry_median(np.array([1., np.nan, 3., np.nan, np.nan]),
                                         np.array(['A', 'A', 'B', 'C', 'C']))
        np.testing.assert_allclose(filled, [1., 1., 3., 2., 2.])
        rng = np.random.default_rng(7)
        n = 100
        caps = rng.normal(size=n)
        groups = np.eye(3)[np.arange(n) % 3]
        weights = np.exp(caps / 2)
        y = caps * 2 + rng.normal(size=n)
        residual = p.neutralize_factor_wls(y, caps, groups, weights)
        X = np.column_stack([caps, groups])
        np.testing.assert_allclose(X.T @ (weights * residual), 0., atol=1e-10)
        with self.assertRaises(ValueError):
            p.neutralize_factor_wls(y[:2], caps[:2], groups[:2], weights[:2])

    def test_rank_deficient_factors_are_rejected(self):
        engine = lecture(13, 'factor_orthogonalization').AshareFactorOrthogonalizer()
        x = np.linspace(-1, 1, 100)
        with self.assertRaisesRegex(ValueError, '秩|共线'):
            engine.fit_transform_matrix(np.column_stack([x, -x]), ['mom', 'rev'])

    def test_orthogonalization_centers_before_whitening(self):
        engine = lecture(13, 'factor_orthogonalization').AshareFactorOrthogonalizer()
        rng = np.random.default_rng(8)
        F = rng.normal(size=(100, 3)) + [10, 20, 30]
        out, _ = engine.fit_transform_matrix(F, ['a', 'b', 'c'])
        np.testing.assert_allclose(np.corrcoef(out, rowvar=False), np.eye(3), atol=1e-10)

    def test_orthogonalization_supports_one_factor_and_both_methods(self):
        module = lecture(13, 'factor_orthogonalization')
        x = np.arange(10., 30.)[:, None]
        for method in ['symmetric', 'gram_schmidt']:
            engine = module.AshareFactorOrthogonalizer(method=method)
            out, diagnostics = engine.fit_transform_matrix(x, ['one'])
            self.assertAlmostEqual(out.mean(), 0.)
            self.assertAlmostEqual(out.std(ddof=1), 1.)
            self.assertAlmostEqual(diagnostics['max_abs_off_diagonal_corr'], 0.)
            with self.assertRaises(ValueError):
                engine.fit_transform_matrix(np.ones((20, 2)), ['one', 'two'])

    def test_zero_profit_base_is_missing_not_infinite_growth(self):
        engine = lecture(10, 'fundamental_factors').AshareFundamentalFactorEngine()
        row = dict(symbol='A', trade_date='2023-04-14', market_cap=1000.,
                   net_profit_ttm=-4., revenue_ttm=1000., total_equity=1000.,
                   total_assets=2000., ocf_ttm=400., net_profit_q=100.,
                   net_profit_ly_same_q=0.)
        result = engine.compute_fundamental_factors(pl.DataFrame([row]))
        self.assertIsNone(result['profit_yoy_growth'][0])
        self.assertIsNone(result['cash_to_profit_ratio'][0])


class EvaluationRegressions(unittest.TestCase):
    def test_sharpe_uses_arithmetic_excess_and_hac_reduces_false_precision(self):
        from examples.factor_utils import annualized_ratio, mean_test
        from scipy import stats
        returns = np.array([.1, -.1, .03, .02])
        daily_rf = 1.02 ** (1/250) - 1
        expected = (returns.mean() - daily_rf) / returns.std(ddof=1) * np.sqrt(250)
        self.assertAlmostEqual(annualized_ratio(returns, annual_rf=.02), expected)
        self.assertTrue(np.isnan(annualized_ratio(np.zeros(10))))
        x = np.array([0., 0., 0., 2., 2., 2.])
        iid, _ = mean_test(x)
        hac, _ = mean_test(x, hac_lags=1)
        self.assertAlmostEqual(iid, stats.ttest_1samp(x, 0).statistic)
        self.assertAlmostEqual(hac, 1.8257418583505538)
        self.assertLess(hac, iid)

    def test_constant_cross_section_does_not_poison_ic(self):
        engine = lecture(14, 'factor_ic_metrics').AshareFactorICEngine(date_col='date')
        rows = [dict(date=d, symbol=str(s), f=float(s if d else 1), r=float(s+s*s))
                for d in range(6) for s in range(5)]
        daily = engine.compute_daily_ic(pl.DataFrame(rows), 'f', 'r')
        report = engine.evaluate(daily, 'f')
        self.assertEqual(report.sample_days, 5)
        self.assertAlmostEqual(report.mean_rank_ic, 1.)
        self.assertAlmostEqual(daily['cum_rank_ic'][-1], 5.)
        with self.assertRaisesRegex(ValueError, '连续'):
            lecture(14, 'factor_ic_metrics').AshareFactorICEngine(date_col='date', hac_lags=1).evaluate(daily, 'f')

    def test_quantiles_handle_ties_and_keep_initial_nav(self):
        engine = lecture(15, 'quantile_and_monotonicity').AshareQuantileBacktestEngine(date_col='date')
        rows = [dict(date=d, symbol=f'{s:02d}', f=1., r=s*.001)
                for d in range(6) for s in range(10)]
        daily, _ = engine.run_quantile_backtest(pl.DataFrame(rows), 'f', 'r')
        self.assertTrue(all(f'G{q}' in daily.columns for q in range(1, 6)))
        shuffled, _ = engine.run_quantile_backtest(pl.DataFrame(rows).reverse(), 'f', 'r')
        self.assertEqual(daily.to_dicts(), shuffled.to_dicts())
        self.assertAlmostEqual(engine._calc_mdd(np.array([.9, .95])), -.1)

    def test_missing_forward_return_cannot_change_selection(self):
        engine = lecture(15, 'quantile_and_monotonicity').AshareQuantileBacktestEngine(date_col='date')
        rows = [dict(date=d, symbol=str(s), f=float(s), r=s*.001)
                for d in range(6) for s in range(10)]
        rows[0]['r'] = None
        with self.assertRaisesRegex(ValueError, '收益|缺失'):
            engine.run_quantile_backtest(pl.DataFrame(rows), 'f', 'r')

    def test_report_sharpe_and_long_contribution_use_daily_means(self):
        from examples.factor_utils import annualized_ratio
        engine = lecture(15, 'quantile_and_monotonicity').AshareQuantileBacktestEngine(n_quantiles=2, date_col='date')
        df = pl.DataFrame([dict(date=d, symbol=str(s), f=float(s), r=(s-1)*.01 + (-1)**d*.02)
                           for d in range(10) for s in range(4)])
        daily, report = engine.run_quantile_backtest(df, 'f', 'r')
        self.assertAlmostEqual(report.group_sharpe_ratios[2], annualized_ratio(daily['G2'], annual_rf=.02))
        self.assertAlmostEqual(report.long_contribution_ratio, .5)
        self.assertAlmostEqual(report.long_only_excess_annual_ret, daily['top_excess_ret'].mean()*250)

    def test_negative_ic_has_no_fabricated_half_life(self):
        engine = lecture(16, 'turnover_and_alpha_decay').AshareFactorDecayEngine(date_col='date')
        df = pl.DataFrame([dict(date=d, symbol=str(s), f=float(s), r=-s*.001)
                           for d in range(30) for s in range(30)])
        report = engine.analyze_decay(df, 'f', 'r', horizons=[1, 2, 3, 5, 10, 20])
        self.assertIsNone(report.half_life_days)

    def test_forward_returns_compound_and_stay_with_symbol(self):
        engine = lecture(16, 'turnover_and_alpha_decay').AshareFactorDecayEngine(date_col='date')
        df = pl.DataFrame(dict(symbol=['A']*3 + ['B']*3, date=[0,1,2]*2,
                               r=[0., .1, -.1, 0., .2, .2]))
        out = engine.forward_returns(df, 'r', [2])
        self.assertAlmostEqual(out['ret_fwd_2d'][0], -.01)
        self.assertAlmostEqual(out['ret_fwd_2d'][3], .44)
        self.assertIsNone(out['ret_fwd_2d'][1])
        with self.assertRaisesRegex(ValueError, '完整'):
            engine.forward_returns(df.filter(~((pl.col('symbol') == 'A') & (pl.col('date') == 1))), 'r', [2])

    def test_half_life_uses_marginal_curve_and_explicit_direction(self):
        engine = lecture(16, 'turnover_and_alpha_decay').AshareFactorDecayEngine()
        curve = {h: -.1 * 2**(-h/4) for h in [1, 2, 3, 5, 10]}
        half_life, _ = engine.fit_half_life(curve, direction=-1)
        self.assertAlmostEqual(half_life, 4.)
        self.assertIsNone(engine.fit_half_life({1:.1, 2:.1, 3:.1})[0])
        self.assertIsNone(engine.fit_half_life({1:.1, 2:-.1, 3:.01})[0])

    def test_execution_preserves_cash_and_all_blocked_days(self):
        engine = lecture(17, 'fake_alpha_forensics').AshareFakeAlphaSanitizer(date_col='date')
        rows = [dict(date=d, symbol=f'{s:02d}', market_cap=float(10+s*s), f=float(s),
                     ret_fwd_1d=.1 if s == 39 else .02, is_limit_up_next_day=int(s != 39))
                for d in range(3) for s in range(40)]
        report = engine.audit_factor(pl.DataFrame(rows), 'f')
        self.assertAlmostEqual(report.realistic_tradeable_top_ret, .1/4*250)
        df = pl.DataFrame(rows).with_columns(pl.lit(1).alias('is_limit_up_next_day'))
        report = engine.audit_factor(df, 'f')
        self.assertEqual(report.realistic_tradeable_top_ret, 0.)

    def test_all_blocked_day_is_not_removed_from_return_average(self):
        engine = lecture(17, 'fake_alpha_forensics').AshareFakeAlphaSanitizer(date_col='date')
        df = pl.DataFrame([dict(date=d, symbol=str(s), market_cap=float(s*s+10), f=float(s),
                                ret_fwd_1d=.1, is_limit_up_next_day=int(d == 0))
                           for d in range(2) for s in range(40)])
        report = engine.audit_factor(df, 'f')
        self.assertAlmostEqual(report.realistic_tradeable_top_ret, .05*250)
        self.assertEqual(report.verdict, 'REVIEW')


if __name__ == '__main__':
    unittest.main()
