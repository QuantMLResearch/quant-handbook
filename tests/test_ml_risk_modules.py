"""第 18–26 讲的离线回归；完整运行需要 ml、deep、optimization extras。"""
import importlib
import unittest
import subprocess
import sys
import textwrap

import numpy as np
import polars as pl
from examples.ml_utils import demo_dates, purged_holdout, RankReturnCalibrator


def lecture(number, name):
    return importlib.import_module(f"examples.lecture_{number}_{name}")


class TemporalRegression(unittest.TestCase):
    def test_demo_calendar_is_unique_valid_and_ordered(self):
        dates = demo_dates(750)
        self.assertEqual(len(set(dates)), 750)
        self.assertEqual(dates, sorted(dates))

    def test_variable_label_intervals_preserve_date_groups(self):
        dates = np.repeat(np.arange(40), 2)
        ends = dates + np.tile([1, 7], 40)
        splitter = lecture(18, 'financial_cv_purging_embargo').CombinatorialPurgedCV(
            n_blocks=4, n_test_blocks=1, embargo_pct=0)
        for train, test, _ in splitter.split(dates, ends):
            self.assertFalse(((dates[train, None] <= ends[test]) & (ends[train, None] >= dates[test])).any())
            for day in np.unique(dates[train]):
                self.assertEqual(np.sum(dates[train] == day), 2)

    def test_holdout_purges_labels_and_future_maturity(self):
        df = pl.DataFrame(dict(trade_date=list(range(20)), label_end_date=list(range(5, 25)),
                               ret_fwd_5d=np.ones(20)))
        train, val = purged_holdout(df, 10, 17, 20)
        self.assertEqual(train['trade_date'].to_list(), list(range(5)))
        self.assertEqual(val['trade_date'].to_list(), list(range(10, 15)))

    def test_walk_forward_never_trains_on_future(self):
        dates = np.repeat(np.arange(30), 2)
        splitter = lecture(18, 'financial_cv_purging_embargo').PurgedGroupTimeSeriesSplit(
            n_splits=3, forward_horizon=5, embargo_pct=0)
        for train, test in splitter.split(dates):
            self.assertLess(dates[train].max() + 5, dates[test].min())

    def test_cpcv_purges_both_label_interval_edges(self):
        dates = np.arange(30)
        splitter = lecture(18, 'financial_cv_purging_embargo').CombinatorialPurgedCV(
            n_blocks=6, n_test_blocks=1, forward_horizon=5, embargo_pct=0)
        for train, test, _ in splitter.split(dates):
            overlaps = ((dates[train, None] <= dates[test] + 5)
                        & (dates[train, None] + 5 >= dates[test]))
            self.assertFalse(overlaps.any())


class RiskRegression(unittest.TestCase):
    def test_historical_exposures_are_used_each_day(self):
        engine = lecture(23, 'barra_risk_model_architecture').AshareBarraRiskEngine(shrinkage_intensity=0)
        X = np.stack([np.array([[1. + t], [2. - .1 * t], [3.]]) for t in range(10)])
        returns = X[:, :, 0] * .01
        result = engine.estimate_risk_model(returns, X, np.ones((10, 3)), ['factor'], list('ABC'),
                                            current_exposure_matrix=X[-1] * 2)
        np.testing.assert_allclose(result.specific_variance, 0, atol=1e-12)
        np.testing.assert_allclose(result.exposure_matrix, X[-1] * 2)
        with self.assertRaises(ValueError):
            engine.estimate_risk_model(returns, X[-1], np.ones(3), ['factor'], list('ABC'))

    def test_impact_preserves_absolute_liquidity_and_aum(self):
        optimizer = lecture(25, 'turnover_constraints_and_costs').AshareTurnoverConstrainedOptimizer()
        adv, vol = np.array([1e6, 2e6]), np.array([.02, .03])
        gamma = optimizer.impact_coefficients(adv, vol, 1e8)
        np.testing.assert_allclose(optimizer.impact_coefficients(adv * 100, vol, 1e8), gamma / 100)
        np.testing.assert_allclose(optimizer.impact_coefficients(adv, vol, 2e8), gamma * 2)
        with self.assertRaises(ValueError):
            optimizer.impact_coefficients([0, 1], vol, 1e8)

    def test_turnover_cost_units_and_constraints(self):
        optimizer = lecture(25, 'turnover_constraints_and_costs').AshareTurnoverConstrainedOptimizer(
            max_active_weight=.4, linear_cost_rate=.001, max_tracking_error=.1)
        r = optimizer.optimize(np.array([.03, -.03]), np.array([.5, .5]), np.array([.5, .5]),
            np.column_stack([np.ones(2), np.zeros(2)]), np.eye(2) * .01, np.ones(2) * .04,
            np.array([1e7, 1e7]), aum=1e8, daily_volatility=np.array([.02, .02]),
            max_turnover=.05, n_industries=1, size_col_idx=1)
        self.assertLessEqual(r.turnover, .05 + 1e-6)
        self.assertLessEqual(r.annualized_tracking_error, .1 + 1e-6)
        self.assertAlmostEqual(r.linear_cost, .001 * 2 * r.turnover)
        self.assertAlmostEqual(r.net_active_return, r.gross_active_return - r.linear_cost - r.impact_cost)
        self.assertAlmostEqual(r.optimal_weights.sum(), 1)

    def test_hrp_does_not_claim_erc(self):
        engine = lecture(26, 'risk_parity_and_hrp').AshareHierarchicalRiskParity()
        cov = np.diag([.04, .16])
        weights = np.array(list(engine.compute_hrp_weights(cov).values()))
        _, trc, vol = engine.evaluate_risk_contributions(weights, cov)
        np.testing.assert_allclose(weights, [.8, .2], atol=1e-9)
        np.testing.assert_allclose(trc / vol, [.8, .2], atol=1e-9)

    def test_specific_variance_removes_residual_mean(self):
        engine = lecture(23, 'barra_risk_model_architecture').AshareBarraRiskEngine(
            shrinkage_intensity=0)
        result = engine.estimate_risk_model(
            np.tile([.01, -.01, .02, -.02], (20, 1)),
            np.ones((4, 1)), np.ones(4), ['market'], list('ABCD'), assume_static=True)
        np.testing.assert_allclose(result.specific_variance, 0, atol=1e-12)

    def test_risk_contributions_sum_to_returned_volatility(self):
        engine = lecture(26, 'risk_parity_and_hrp').AshareHierarchicalRiskParity()
        mrc, trc, vol = engine.evaluate_risk_contributions(
            np.array([.5, .5]), np.eye(2) * .0004)
        self.assertAlmostEqual(trc.sum(), vol)
        np.testing.assert_allclose(trc, .5 * mrc)

    def test_hrp_single_asset(self):
        engine = lecture(26, 'risk_parity_and_hrp').AshareHierarchicalRiskParity()
        self.assertEqual(engine.compute_hrp_weights(np.array([[.04]]), ['A']), {'A': 1.})

    def test_te_limit_is_enforced(self):
        optimizer = lecture(24, 'cvxpy_portfolio_optimization').AshareIndexEnhancementOptimizer(
            risk_aversion=.01, max_active_weight=.49)
        result = optimizer.optimize(np.array([1., -1.]), np.array([.5, .5]),
            np.column_stack([np.ones(2), np.zeros(2)]), np.eye(2), np.ones(2) * 4,
            n_industries=1, size_col_idx=1)
        self.assertLessEqual(result.annualized_tracking_error, .045 + 1e-6)


class MLRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = lecture(22, 'ml_stock_selection_pipeline')
        cls.raw, cls.features, cls.calendar = cls.module.demo_panel(n_days=50, n_stocks=20)
        cls.pipeline = cls.module.IndustrialMLStockPickerPipeline(cls.features, top_k=5)
        cls.panel = cls.pipeline.preprocess_cross_section(cls.raw)
        cls.pipeline.train_purged_ranker(cls.panel, cls.calendar[15],
            validation_end=cls.calendar[25], as_of=cls.calendar[30], early_stopping_rounds=3)

    def test_future_labels_cannot_change_training(self):
        changed = self.panel.with_columns(pl.when(pl.col('label_end_date') >= self.calendar[30])
            .then(pl.lit(99.)).otherwise(pl.col('ret_fwd_5d')).alias('ret_fwd_5d'))
        other = self.module.IndustrialMLStockPickerPipeline(self.features, top_k=5)
        other.train_purged_ranker(changed, self.calendar[15], validation_end=self.calendar[25],
                                 as_of=self.calendar[30], early_stopping_rounds=3)
        today = self.panel.filter(pl.col('trade_date') == self.calendar[40])
        np.testing.assert_allclose(self.pipeline.score(today)['ml_score'], other.score(today)['ml_score'])
        self.assertLess(self.pipeline.training_audit['train_label_end'], self.calendar[15])
        self.assertLess(self.pipeline.training_audit['validation_label_end'], self.calendar[30])

    def test_historical_signal_rejected_and_empty_portfolio_is_cash(self):
        with self.assertRaises(ValueError):
            self.pipeline.score(self.panel.head(20))
        all_st = self.panel.with_columns(pl.lit(1).alias('is_st'))
        self.assertTrue(self.pipeline.generate_daily_portfolio(all_st, self.calendar[40]).is_empty())
        with self.assertRaises(ValueError):
            self.pipeline.generate_daily_portfolio(self.panel.drop('is_suspended'), self.calendar[40])

    def test_close_signal_does_not_use_future_open_status(self):
        before = self.pipeline.generate_daily_portfolio(self.panel, self.calendar[40])
        after = self.pipeline.generate_daily_portfolio(
            self.panel.with_columns(pl.lit(1).alias('is_limit_up_at_open')), self.calendar[40])
        self.assertEqual(before.to_dicts(), after.to_dicts())
        self.assertAlmostEqual(before['target_weight'].sum(), 1.)

    def test_rank_return_calibration_and_maturity(self):
        data = pl.DataFrame(dict(trade_date=[10] * 4 + [11] * 4, label_end_date=[15] * 4 + [16] * 4,
                                ml_score=[0., 1., 2., 3.] * 2,
                                excess_return=[-.03, -.01, .01, .03] * 2))
        c = RankReturnCalibrator().fit(data, model_available_at=10, as_of=17, horizon_days=5)
        today = data.head(4).with_columns(pl.lit(17).alias('trade_date'))
        np.testing.assert_allclose(c.predict(today), [-.03, -.01, .01, .03])
        np.testing.assert_allclose(c.predict(today.with_columns((pl.col('ml_score') * 100 + 9))), c.predict(today))
        with self.assertRaises(ValueError):
            RankReturnCalibrator().fit(data, model_available_at=10, as_of=16, horizon_days=5)
        with self.assertRaises(ValueError):
            c.predict(data)
        with self.assertRaises(ValueError):
            RankReturnCalibrator().fit(data.with_columns(pl.lit(float('nan')).alias('ml_score')),
                                       model_available_at=10, as_of=17, horizon_days=5)

    def test_frozen_predictions_calibrate_then_enter_optimizer(self):
        past = self.pipeline.score(self.panel.filter(
            (pl.col('trade_date') >= self.calendar[30]) & (pl.col('trade_date') < self.calendar[35])))
        past = past.with_columns((pl.col('ret_fwd_5d') - pl.col('ret_fwd_5d').mean().over('trade_date')).alias('excess_return'))
        calibrator = RankReturnCalibrator().fit(past, model_available_at=self.calendar[30],
                                                as_of=self.calendar[40], horizon_days=5)
        today = self.pipeline.score(self.panel.filter(pl.col('trade_date') == self.calendar[40]))
        alpha = calibrator.predict(today)
        n = today.height
        X = np.column_stack([np.eye(5)[np.arange(n) % 5], np.zeros(n)])
        optimizer = lecture(24, 'cvxpy_portfolio_optimization').AshareIndexEnhancementOptimizer(
            horizon_days=calibrator.horizon_days)
        result = optimizer.optimize(alpha, np.ones(n) / n, X, np.eye(6) * .01,
                                    np.ones(n) * .04, n_industries=5, size_col_idx=5)
        self.assertEqual(result.horizon_days, 5)
        self.assertLessEqual(result.annualized_tracking_error, .045 + 1e-6)
        self.assertAlmostEqual(result.expected_active_return, alpha @ (result.optimal_weights - 1 / n))

    def test_shap_additivity_and_unique_stock(self):
        self.run_isolated("""
            import numpy as np
            import polars as pl
            import lightgbm as lgb
            from examples.ml_utils import demo_dates
            from examples.lecture_20_shap_interpretability import AshareTreeSHAPExplainer
            rng = np.random.default_rng(7)
            X = rng.normal(size=(100, 2))
            model = lgb.train(dict(objective='regression', num_leaves=4, min_data_in_leaf=5,
                                  num_threads=2, verbosity=-1), lgb.Dataset(X, label=X[:, 0], feature_name=['a', 'b']), num_boost_round=5)
            explainer = AshareTreeSHAPExplainer(model, ['a', 'b'])
            data = pl.DataFrame(dict(trade_date=[demo_dates(1)[0]] * 100, symbol=[str(i) for i in range(100)], a=X[:, 0], b=X[:, 1]))
            values, _ = explainer.explain_dataset(data)
            np.testing.assert_allclose(values.sum(axis=1) + explainer.expected_value, model.predict(X, raw_score=True), atol=1e-8)
            assert len(explainer.explain_single_stock(data, '0', data['trade_date'][0])) == 2
        """)

    def test_deep_model_forward_and_gradient(self):
        self.run_isolated("""
            import torch
            from examples.lecture_21_deep_learning_boundaries import AshareTemporalCNNGRU
            torch.set_num_threads(2)
            model = AshareTemporalCNNGRU(n_features=3)
            pred = model(torch.randn(4, 20, 3))
            assert tuple(pred.shape) == (4,)
            pred.square().mean().backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        """)

    def run_isolated(self, script):
        # 独立解释器隔离可选原生运行库；本机在已训练 LightGBM 的进程中运行 Torch 卷积可阻塞。
        result = subprocess.run([sys.executable, '-c', textwrap.dedent(script)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
