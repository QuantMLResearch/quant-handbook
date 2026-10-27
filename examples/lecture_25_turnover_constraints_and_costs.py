"""H 日收益、H 日风险与单次交易成本口径一致的教学优化器。"""
from dataclasses import dataclass
import cvxpy as cp
import numpy as np
from examples.lecture_24_cvxpy_portfolio_optimization import AshareIndexEnhancementOptimizer
from examples.portfolio_utils import validate_risk_inputs, solve_checked


@dataclass
class TurnoverOptimizationResult:
    optimal_weights: np.ndarray
    gross_active_return: float   # H 日预期主动收益
    linear_cost: float           # 本次调仓占净资产比例
    impact_cost: float           # 本次调仓占净资产比例
    net_active_return: float     # H 日毛收益减本次成本；非全年净收益
    turnover: float
    annualized_tracking_error: float
    horizon_days: int
    status: str


class AshareTurnoverConstrainedOptimizer(AshareIndexEnhancementOptimizer):
    def __init__(self, risk_aversion=2.5, linear_cost_rate=.0015, impact_penalty_coef=1.,
                 max_active_weight=.015, size_tolerance=.05, max_tracking_error=.045,
                 horizon_days=5, annual_factor=250.):
        super().__init__(risk_aversion, max_active_weight, size_tolerance,
                         max_tracking_error, horizon_days, annual_factor)
        if linear_cost_rate < 0 or impact_penalty_coef < 0:
            raise ValueError('成本系数不能为负')
        self.linear_cost, self.impact_coef = linear_cost_rate, impact_penalty_coef

    def impact_coefficients(self, adv_amount, daily_volatility, aum):
        """gamma_i = k * sigma_daily_i * AUM / ADV_amount_i。

        线性价格冲击的二次总成本近似；k 必须在历史执行数据上校准。
        AUM 和 ADV 都以同一币种金额计，不是股数。无全截面再归一化。
        """
        adv = np.asarray(adv_amount, dtype=float)
        vol = np.asarray(daily_volatility, dtype=float)
        if adv.ndim != 1 or vol.shape != adv.shape or not np.isfinite(aum) or aum <= 0:
            raise ValueError('ADV/波动率形状必须一致，AUM 须为正')
        if not np.isfinite(adv).all() or not np.isfinite(vol).all() or np.any(adv <= 0) or np.any(vol < 0):
            raise ValueError('ADV 须为正，日波动率须非负，且均为有限数值')
        return self.impact_coef * vol * aum / adv

    def optimize(self, expected_returns, benchmark_weights, prev_weights, exposure_matrix,
                 factor_covariance, specific_variance, adv_amount, *, aum, daily_volatility,
                 max_turnover=None, n_industries=6, size_col_idx=6):
        expected_returns, benchmark_weights, exposure_matrix, factor_covariance, specific_variance = validate_risk_inputs(
            expected_returns, benchmark_weights, exposure_matrix, factor_covariance, specific_variance, n_industries, size_col_idx)
        n = len(expected_returns)
        prev_weights = np.asarray(prev_weights, dtype=float)
        if prev_weights.shape != (n,) or not np.isfinite(prev_weights).all() or np.any(prev_weights < 0) or not np.isclose(prev_weights.sum(), 1):
            raise ValueError('旧仓位须为同一证券顺序的非负满仓权重；本例未建模现金与池外仓位')
        if max_turnover is not None and not 0 <= max_turnover <= 1:
            raise ValueError('单边换手上限须在 [0,1]')
        gamma = self.impact_coefficients(adv_amount, daily_volatility, aum)
        if gamma.shape != (n,):
            raise ValueError('流动性输入必须与证券顺序对齐')
        w = cp.Variable(n)
        h, delta = w - benchmark_weights, w - prev_weights
        y = exposure_matrix.T @ h
        annual_risk = cp.quad_form(y, factor_covariance) + cp.sum(cp.multiply(specific_variance, cp.square(h)))
        period_risk = annual_risk * self.horizon_days / self.annual_factor
        linear_cost = self.linear_cost * cp.norm1(delta)
        impact_cost = .5 * cp.sum(cp.multiply(gamma, cp.square(delta)))
        objective = cp.Maximize(expected_returns @ h - self.risk_aversion / 2 * period_risk - linear_cost - impact_cost)
        constraints = [w >= 0, cp.sum(w) == 1,
            exposure_matrix[:, :n_industries].T @ h == 0,
            cp.abs(exposure_matrix[:, size_col_idx] @ h) <= self.size_tolerance,
            h <= self.max_active_weight, annual_risk <= self.max_tracking_error ** 2]
        if max_turnover is not None:
            constraints.append(.5 * cp.norm1(delta) <= max_turnover)
        problem = cp.Problem(objective, constraints)
        result = solve_checked(problem, w, constraints)
        actual_delta, active = result - prev_weights, result - benchmark_weights
        gross = float(expected_returns @ active)
        linear = float(self.linear_cost * np.abs(actual_delta).sum())
        impact = float(.5 * np.sum(gamma * actual_delta ** 2))
        beta = exposure_matrix.T @ active
        te = float(np.sqrt(max(0., beta @ factor_covariance @ beta + np.sum(specific_variance * active ** 2))))
        return TurnoverOptimizationResult(result, gross, linear, impact, gross - linear - impact,
            float(.5 * np.abs(actual_delta).sum()), te, self.horizon_days, problem.status)

    def trace_pareto_frontier(self, expected_returns, benchmark_weights, prev_weights,
                             exposure_matrix, factor_covariance, specific_variance, adv_amount,
                             *, aum, daily_volatility, turnover_caps=None,
                             n_industries=6, size_col_idx=6):
        if turnover_caps is None:
            turnover_caps = [.05, .10, .15, .20, .30, .50]
        return [self.optimize(expected_returns, benchmark_weights, prev_weights, exposure_matrix,
                factor_covariance, specific_variance, adv_amount, aum=aum,
                daily_volatility=daily_volatility, max_turnover=cap,
                n_industries=n_industries, size_col_idx=size_col_idx) for cap in turnover_caps]


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    n = 400
    benchmark = rng.uniform(.1, .4, n)
    benchmark /= benchmark.sum()
    # 合规初始仓位，确保包括很小换手上限的各情景有可行解。
    prev = benchmark.copy()
    X = np.column_stack([np.eye(6)[rng.integers(0, 6, n)], rng.normal(size=n)])
    F, specific = np.eye(7) * .04, rng.uniform(.04, .10, n)
    adv = rng.uniform(1e6, 5e7, n)
    alpha_5d = rng.normal(.03, .06, n) * 5 / 250
    optimizer = AshareTurnoverConstrainedOptimizer(linear_cost_rate=.0005)
    caps = [.05, .10, .15, .20, .30, .50]
    frontier = optimizer.trace_pareto_frontier(alpha_5d, benchmark, prev, X, F, specific, adv,
        aum=1e8, daily_volatility=np.full(n, .02), turnover_caps=caps)
    print('上限 | 实际单边换手 | 5日毛超额 | 本次成本 | 5日扣费预期 | 年化TE')
    for cap, r in zip(caps, frontier):
        print(f'{cap:.1%} | {r.turnover:.2%} | {r.gross_active_return:.4%} | '
              f'{r.linear_cost + r.impact_cost:.4%} | {r.net_active_return:.4%} | {r.annualized_tracking_error:.2%}')
    print('这是模型内敏感性表；放宽上限不会强制交易，不存在预设的 15%～20% 最佳换手率。')
