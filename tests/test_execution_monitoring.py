"""第 27–31 讲：回归已复现错误，并检查账户/归因守恒及风险控制行为。"""
import contextlib
import io
import unittest
from dataclasses import replace
import numpy as np
from examples.lecture_27_vectorized_vs_event_driven_backtest import AshareEventDrivenBacktester, OrderStatus
from examples.lecture_28_transaction_friction_and_market_impact import AshareFrictionEngine, AUMCapacityEstimator
from examples.lecture_29_matching_rules_limits_suspensions import (
    DailyMarketBar, LotRule, CorporateAction, CorporateActionBook, AshareExecutionMatchingSimulator)
from examples.lecture_30_performance_attribution_brinson_barra import AsharePerformanceAttributor
from examples.lecture_31_live_monitoring_risk_dashboard import AshareRiskMonitoringDashboard

DAYS = ['2025-01-02', '2025-01-03', '2025-01-06', '2025-01-07', '2025-01-08']


def bar(price=10., **kwargs):
    return replace(DailyMarketBar('A', price, price, price, price, None, None,
                   opening_buy_liquidity=100000, opening_sell_liquidity=100000), **kwargs)


def free_costs():
    return AshareFrictionEngine(0, 0, 0, 0, 0)


def account(**kwargs):
    return AshareEventDrivenBacktester(10000, trading_calendar=DAYS,
        slippage_fixed=0, friction_engine=kwargs.pop('friction_engine', free_costs()), **kwargs)


def rebalance(engine, weights, index=0, market=None, **kwargs):
    return engine.execute_rebalance(weights, {'A': bar()} if market is None else market,
        DAYS[index], signal_date='2024-12-31' if index == 0 else DAYS[index-1], **kwargs)


def report(monitor=None, returns=(0., 0.), day='2025-01-03', **kwargs):
    return (monitor or AshareRiskMonitoringDashboard()).evaluate_live_status(
        day, returns, kwargs.pop('ex_ante_te', .05), kwargs.pop('style_exposures', {}),
        kwargs.pop('baseline_features', {}), kwargs.pop('live_features', {}), **kwargs)


class AccountTests(unittest.TestCase):
    def test_new_position_has_market_value_and_close_mark(self):
        e = account()
        rebalance(e, {'A': .5})
        self.assertEqual(e.total_equity, 10000)
        snap = e.close_day(DAYS[0], {'A': 11})
        self.assertEqual(snap['total_equity'], 10500)
        self.assertAlmostEqual(snap['net_return'], .05)

    def test_same_day_does_not_unlock_next_trading_day_does(self):
        e = account()
        rebalance(e, {'A': .5})
        rejected = rebalance(e, {})[-1]
        self.assertEqual(rejected.status, OrderStatus.REJECTED)
        self.assertEqual(e.positions['A'].total_shares, 500)
        e.close_day(DAYS[0], {'A': 10})
        rebalance(e, {}, 1)
        self.assertEqual(e.positions['A'].total_shares, 0)
        self.assertEqual(e.cash, 10000)

    def test_full_weight_reserves_cost_and_reconciles(self):
        e = account(friction_engine=AshareFrictionEngine())
        order = rebalance(e, {'A': 1})[-1]
        self.assertGreater(order.shares, 0)
        self.assertGreaterEqual(e.cash, 0)
        snap = e.close_day(DAYS[0], {'A': 10})
        self.assertAlmostEqual(snap['total_equity'], 10000 - order.total_friction)
        self.assertEqual(order.status, OrderStatus.PARTIAL)

    def test_partial_fills_cannot_reuse_opening_liquidity(self):
        e = account()
        market = {'A': bar(opening_buy_liquidity=250)}
        self.assertEqual(rebalance(e, {'A': 1}, market=market)[-1].shares, 200)
        self.assertEqual(rebalance(e, {'A': 1}, market=market)[-1].shares, 0)
        self.assertEqual(e.positions['A'].total_shares, 200)

    def test_calendar_signal_missing_prices_and_duplicate_close(self):
        e = account()
        with self.assertRaises(ValueError):
            e.execute_rebalance({'A': 1}, {'A': bar()}, DAYS[0], signal_date=DAYS[0])
        with self.assertRaises(ValueError):
            e.execute_rebalance({}, {}, '2025-01-04', signal_date=DAYS[0])
        rebalance(e, {'A': .5})
        with self.assertRaises(ValueError):
            e.close_day(DAYS[0], {})
        e.close_day(DAYS[0], {'A': 10})
        with self.assertRaises(ValueError):
            e.close_day(DAYS[0], {'A': 10})
        with self.assertRaises(ValueError):
            rebalance(e, {}, 2)
        with self.assertRaises(ValueError):
            rebalance(e, {}, 1, market={})

    def test_dividend_entitlement_receivable_payment_and_listing(self):
        action = CorporateAction('DIV', 'A', DAYS[0], DAYS[1], DAYS[2], DAYS[3], 1., .1)
        e = account(corporate_actions=[action])
        rebalance(e, {'A': .1})  # 登记日持有 100 股
        e.close_day(DAYS[0], {'A': 10})
        ex_price = 9 / 1.1
        rebalance(e, {'A': 1}, 1, market={'A': bar(ex_price)}, allow_new_buys=False)
        self.assertEqual(e.positions['A'].total_shares, 110)
        self.assertEqual(e.positions['A'].available_shares, 100)
        self.assertEqual(e.dividend_receivable, 100)
        self.assertAlmostEqual(e.total_equity, 10000)
        e.close_day(DAYS[1], {'A': ex_price})
        rebalance(e, {}, 2, market={'A': bar(ex_price)})  # 原股份卖出，送股仍未上市
        self.assertEqual(e.positions['A'].total_shares, 10)
        self.assertEqual(e.positions['A'].available_shares, 0)
        self.assertEqual(e.dividend_receivable, 0)
        before = e.cash
        rebalance(e, {}, 2, market={'A': bar(ex_price)})
        self.assertEqual(e.cash, before)
        e.close_day(DAYS[2], {'A': ex_price})
        order = rebalance(e, {}, 3, market={'A': bar(ex_price)})[-1]
        self.assertEqual(order.shares, 10)  # 不足一手的余股全部清仓
        self.assertAlmostEqual(e.cash, 10000)
        e.close_day(DAYS[3], {})
        rebalance(e, {}, 4, market={})  # 历史送股已卖完，不再强求行情
        self.assertAlmostEqual(e.close_day(DAYS[4], {})['total_equity'], 10000)

    def test_missing_record_snapshot_fails_before_cash_changes(self):
        a = CorporateAction('DIV', 'A', '2024-12-31', DAYS[0], DAYS[1], cash_div_per_share=1)
        e = account(corporate_actions=[a])
        with self.assertRaises(ValueError):
            rebalance(e, {})
        self.assertEqual(e.cash, 10000)
        self.assertEqual(e.actions.processed, set())

    def test_monitor_controls_block_purchase_in_execution(self):
        r = report(returns=[-.11, 0])
        e = account()
        order = rebalance(e, {'A': .5}, allow_new_buys=r.controls.allow_new_buys)[-1]
        self.assertEqual(order.shares, 0)
        self.assertEqual(order.reason, 'RISK_CONTROL')
        self.assertEqual(e.cash, 10000)


class MatchingAndCorporateTests(unittest.TestCase):
    def test_open_fill_invariant_to_future_high_low_close(self):
        matcher = AshareExecutionMatchingSimulator(0)
        for price, direction in [(11, 'BUY'), (9, 'SELL'), (10, 'BUY'), (10, 'SELL')]:
            a = bar(price, limit_up_price=11, limit_down_price=9)
            b = replace(a, high=999, low=.1, close=5)
            self.assertEqual(matcher.simulate_order_execution('A', direction, 100, a),
                             matcher.simulate_order_execution('A', direction, 100, b))

    def test_suspension_unknown_liquidity_and_price_limits(self):
        m = AshareExecutionMatchingSimulator(.02)
        for b in [bar(is_suspended=True), bar(opening_buy_liquidity=0),
                  bar(10.99, limit_up_price=11)]:
            self.assertFalse(m.simulate_order_execution('A', 'BUY', 100, b)[0])

    def test_configurable_lot_and_partial_liquidity(self):
        m = AshareExecutionMatchingSimulator(0)
        fill = m.simulate_order_execution('A', 'BUY', 250, bar(opening_buy_liquidity=231),
                                          lot_rule=LotRule(200, 1))
        self.assertEqual(fill[1], 231)
        self.assertFalse(m.simulate_order_execution('A', 'BUY', 199, bar(), lot_rule=LotRule(200, 1))[0])

    def test_record_date_freezes_entitlement_and_phases_idempotent(self):
        a = CorporateAction('DIV', 'A', DAYS[0], DAYS[1], DAYS[2], DAYS[3], 1., .2)
        book = CorporateActionBook([a])
        book.capture_record_date(DAYS[0], {'A': 100})
        with self.assertRaises(ValueError):
            book.capture_record_date(DAYS[0], {'A': 200})
        ex = book.process_day(DAYS[1])
        self.assertEqual((ex.cash_delta, ex.receivable_delta, ex.shares_delta['A']), (0, 100, 20))
        self.assertEqual(book.process_day(DAYS[1]).receivable_delta, 0)
        pay = book.process_day(DAYS[2])
        self.assertEqual((pay.cash_delta, pay.receivable_delta), (100, -100))
        self.assertEqual(book.process_day(DAYS[3]).listed_delta, {'A': 20})
        self.assertEqual(book.process_day(DAYS[3]).listed_delta, {})

    def test_zero_entitlement_and_fractional_share_rejection(self):
        a = CorporateAction('DIV', 'A', DAYS[0], DAYS[1], share_listing_date=DAYS[1], bonus_share_ratio=.15)
        book = CorporateActionBook([a])
        book.capture_record_date(DAYS[0], {})
        self.assertEqual(book.process_day(DAYS[1]).shares_delta['A'], 0)
        book = CorporateActionBook([a])
        book.capture_record_date(DAYS[0], {'A': 1})
        with self.assertRaises(ValueError):
            book.process_day(DAYS[1])
        self.assertFalse(book.processed)


class CostAndAttributionTests(unittest.TestCase):
    def test_fee_direction_zero_order_and_impact_scaling(self):
        f = AshareFrictionEngine()
        buy = f.calculate_trade_friction('A', 'BUY', 10000, 1e8)
        sell = f.calculate_trade_friction('A', 'SELL', 10000, 1e8)
        self.assertAlmostEqual(sell.total_friction - buy.total_friction, 5)
        self.assertEqual(f.calculate_trade_friction('A', 'BUY', 0, 1e8).total_friction, 0)
        four = f.calculate_trade_friction('A', 'BUY', 40000, 1e8)
        self.assertAlmostEqual(four.impact_cost / buy.impact_cost, 8)
        with self.assertRaises(ValueError):
            f.calculate_trade_friction('A', 'BUY', 100, 0)

    def test_capacity_days_two_sides_negative_profit(self):
        f = AshareFrictionEngine(.001, .002, 0, .003, .5)
        aum, tau, days, n, adv, vol = 1e6, 5, 250, 10, 1e8, .02
        row = AUMCapacityEstimator(.01, tau, n, adv, vol, days, f).estimate_capacity_curve([aum]).row(0, named=True)
        q = aum * tau / (days * n)
        expected = tau * (.001 + 2 * (.002 + .003) + 2 * .5 * vol * np.sqrt(q / adv))
        self.assertAlmostEqual(row['annual_friction'], expected)
        self.assertAlmostEqual(row['annual_profit_cny'], aum * (.01 - expected))
        self.assertLess(row['annual_profit_cny'], 0)
        self.assertAlmostEqual(row['daily_two_sided_participation'], 2*q/adv)

    def test_capacity_commission_floor_and_participation(self):
        row = AUMCapacityEstimator(annual_turnover=1, n_stocks=1,
                friction_engine=AshareFrictionEngine(0, 0, 5, 0, 0)).estimate_capacity_curve([10000]).row(0, named=True)
        self.assertAlmostEqual(row['annual_friction'], .25)  # 250 天×双边×5 元
        with self.assertRaises(ValueError):
            AUMCapacityEstimator(mean_adv=100).estimate_capacity_curve([1e8])

    def test_brinson_cash_and_fees_reconcile(self):
        r = AsharePerformanceAttributor().compute_brinson_attribution(
            [.5], [1.], [.1], ['industry'], cash_return=.02, portfolio_cost=.003, benchmark_cost=.001)
        self.assertAlmostEqual(r.total_active_return, -.042)
        self.assertAlmostEqual(r.allocation_effect + r.selection_effect + r.interaction_effect + r.cost_effect,
                               r.total_active_return)
        self.assertAlmostEqual(r.reconciliation_error, 0)
        self.assertIn('__CASH__', r.industry_details['industry'].to_list())

    def test_brinson_empty_benchmark_sector_closes(self):
        r = AsharePerformanceAttributor().compute_brinson_attribution([.5, .5], [1., 0], [.1, -.2], ['a', 'b'])
        self.assertAlmostEqual(r.total_active_return, -.15)
        self.assertAlmostEqual(r.reconciliation_error, 0)

    def test_factor_cash_fees_and_residual_are_separate(self):
        r = AsharePerformanceAttributor().compute_barra_attribution(
            [.5], [1.], [.1], [[1, 2]], [.01, .02], ['industry', 'style'],
            n_industries=1, cash_return=.02, portfolio_cost=.003, benchmark_cost=.001)
        self.assertAlmostEqual(r.residual_pnl, -.025)
        self.assertAlmostEqual(r.cash_pnl, .01)
        self.assertAlmostEqual(r.total_active_return, -.042)
        self.assertAlmostEqual(r.reconciliation_error, 0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            r.print_summary()
        self.assertNotIn('健康', output.getvalue())


class MonitoringTests(unittest.TestCase):
    def test_first_day_loss_and_relative_wealth(self):
        self.assertAlmostEqual(report(returns=[-.11, 0]).current_active_drawdown, .11)
        self.assertEqual(report(returns=[-.10, 0]).circuit_breaker_tier, 'TIER_3_EMERGENCY')
        r = report(returns=[-.1, 0], benchmark_returns_history=[.1, 0])
        self.assertAlmostEqual(r.current_active_drawdown, 1 - 1/1.1)

    def test_emergency_is_latched_and_requires_explicit_reset(self):
        m = AshareRiskMonitoringDashboard()
        self.assertEqual(report(m, [-.11, 0]).circuit_breaker_tier, 'TIER_3_EMERGENCY')
        with self.assertRaises(ValueError):
            m.reset_emergency('approval')
        r = report(m, [-.11, 0, .13], day='2025-01-06')
        self.assertEqual(r.circuit_breaker_tier, 'TIER_3_EMERGENCY')
        self.assertFalse(r.controls.allow_new_buys)
        with self.assertRaises(ValueError):
            m.reset_emergency('')
        m.reset_emergency('DEMO-001')
        self.assertEqual(report(m, [-.11, 0, .13], day='2025-01-07').circuit_breaker_tier, 'NORMAL')
        self.assertEqual(len(m.restart_log), 1)

    def test_hysteresis_retains_tier_until_recovery_threshold(self):
        m = AshareRiskMonitoringDashboard()
        self.assertEqual(report(m, [-.08, 0]).circuit_breaker_tier, 'TIER_2_THROTTLE')
        self.assertEqual(report(m, [-.08, 0, .03]).circuit_breaker_tier, 'TIER_2_THROTTLE')
        self.assertEqual(report(m, [-.08, 0, .06]).circuit_breaker_tier, 'TIER_1_WARNING')
        self.assertEqual(report(m, [-.08, 0, .08]).circuit_breaker_tier, 'NORMAL')

    def test_constant_and_discrete_psi(self):
        m = AshareRiskMonitoringDashboard()
        self.assertGreater(m.calculate_psi(np.ones(100), np.ones(100)*2), .25)
        self.assertGreater(m.calculate_psi(np.ones(100), np.zeros(100)), .25)
        self.assertEqual(m.calculate_psi(np.ones(100), np.ones(100)), 0)
        self.assertTrue(np.isfinite(m.calculate_psi(np.r_[np.zeros(50), np.ones(50)], np.arange(100))))

    def test_invalid_returns_fail_closed_and_latch_survives(self):
        for returns in ([], [0], [np.nan, 0], [0, np.inf], [-1, 0]):
            r = report(returns=returns)
            self.assertFalse(r.data_valid)
            self.assertEqual(r.circuit_breaker_tier, 'DATA_ERROR')
            self.assertFalse(r.controls.allow_new_buys)
        m = AshareRiskMonitoringDashboard()
        report(m, [-.11, 0])
        self.assertEqual(report(m, []).circuit_breaker_tier, 'TIER_3_EMERGENCY')
        m = AshareRiskMonitoringDashboard()
        report(m, [])
        self.assertEqual(report(m, [-.11, 0, .13]).circuit_breaker_tier, 'TIER_3_EMERGENCY')

    def test_missing_features_styles_nan_and_missing_exante(self):
        cases = [dict(baseline_features={'x': np.ones(10)}), dict(style_exposures={'Size': np.nan}),
                 dict(ex_ante_te=np.nan), dict(ex_ante_te='missing'), dict(style_exposures={'Size': 'missing'}),
                 dict(baseline_features={'x': np.ones(10)}, live_features={'x': []})]
        for kwargs in cases:
            r = report(**kwargs)
            self.assertFalse(r.data_valid)
            self.assertFalse(r.controls.allow_new_buys)
        self.assertFalse(report(AshareRiskMonitoringDashboard(required_styles=('Size',))).data_valid)

    def test_target_te_window_and_configured_renderer(self):
        m = AshareRiskMonitoringDashboard(target_te=.01, style_threshold=.2, te_window=2, psi_warning=.2, psi_critical=.4)
        r = report(m, [0.1, -.1, .001, -.001], style_exposures={'Size': .15})
        self.assertAlmostEqual(r.realized_tracking_error, np.std([.001, -.001], ddof=1)*np.sqrt(250))
        self.assertTrue(any(a.category == 'TRACKING_ERROR' for a in r.alerts))
        self.assertFalse(any(a.category == 'STYLE_DRIFT' for a in r.alerts))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r.render_terminal_dashboard()
        self.assertIn('±0.2', out.getvalue())
        self.assertIn('0.2/0.4', out.getvalue())

    def test_psi_alert_is_distribution_not_concept_drift(self):
        r = report(baseline_features={'x': np.ones(100)}, live_features={'x': np.ones(100)*2})
        self.assertEqual(r.circuit_breaker_tier, 'FEATURE_REVIEW')
        self.assertFalse(r.controls.allow_new_buys)
        self.assertTrue(any(a.category == 'FEATURE_DISTRIBUTION' for a in r.alerts))


if __name__ == '__main__':
    unittest.main()
