"""单期、期初权重的 Brinson 与因子归因；显式包含现金、费用及闭合检查。"""
from dataclasses import dataclass
import numpy as np
import polars as pl


def inputs(portfolio_weights, benchmark_weights, stock_returns, cash_return, portfolio_cost, benchmark_cost):
    p, b, r = [np.asarray(x, dtype=float) for x in (portfolio_weights, benchmark_weights, stock_returns)]
    if p.ndim != 1 or not len(p) or p.shape != b.shape or p.shape != r.shape:
        raise ValueError('股票权重与收益须为等长非空向量')
    if not all(np.isfinite(x).all() for x in (p, b, r)) or not np.isfinite([cash_return, portfolio_cost, benchmark_cost]).all():
        raise ValueError('归因输入须全部有限')
    if min(p.min(), b.min(), portfolio_cost, benchmark_cost) < 0 or p.sum() > 1 + 1e-10 or b.sum() > 1 + 1e-10:
        raise ValueError('本例仅支持非负、无杠杆权重；费用须非负')
    if r.min() < -1 or cash_return < -1:
        raise ValueError('简单收益不能低于 -100%')
    cp, cb = max(0., 1 - p.sum()), max(0., 1 - b.sum())
    gross = float((p - b) @ r + (cp - cb) * cash_return)
    cost_effect = benchmark_cost - portfolio_cost
    return p, b, r, cp, cb, gross, cost_effect


@dataclass
class BrinsonAttributionReport:
    total_active_return: float          # 扣费后主动收益
    gross_active_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    cost_effect: float
    reconciliation_error: float
    industry_details: pl.DataFrame

    def print_summary(self):
        print('Brinson 单期归因（含现金类别，费用单列）')
        print(self.industry_details)
        print(f'毛主动收益={self.gross_active_return:.4%}，费用贡献={self.cost_effect:.4%}，'
              f'净主动收益={self.total_active_return:.4%}，闭合误差={self.reconciliation_error:.3g}')


@dataclass
class BarraAttributionReport:
    total_active_return: float
    industry_pnl: float
    style_pnl_by_factor: dict
    total_style_pnl: float
    residual_pnl: float
    cash_pnl: float
    cost_effect: float
    reconciliation_error: float

    def print_summary(self):
        print('因子归因：行业、风格、未解释股票残差、现金与费用')
        print(f'主动收益={self.total_active_return:.4%}，行业={self.industry_pnl:.4%}，'
              f'风格={self.total_style_pnl:.4%}，股票残差={self.residual_pnl:.4%}，'
              f'现金={self.cash_pnl:.4%}，费用={self.cost_effect:.4%}')
        print(self.style_pnl_by_factor)
        print(f'闭合误差={self.reconciliation_error:.3g}；残差不是已证明可持续的选股能力。')


class AsharePerformanceAttributor:
    def compute_brinson_attribution(self, portfolio_weights, benchmark_weights, stock_returns,
                                    industry_names, *, cash_return=0., portfolio_cost=0., benchmark_cost=0.):
        """期初权重、同区间总收益；现金权重由剩余权重补齐，成本为占期初资产的比例。

        期内有交易时应按成交事件切分子期，再选择明确的多期链接方法；不可用期末权重倒算。
        对基准未覆盖的行业，rb=0 仅为分解约定，不等于真实行业基准收益。
        """
        p, b, r, cp, cb, gross, fee = inputs(portfolio_weights, benchmark_weights, stock_returns,
                                            cash_return, portfolio_cost, benchmark_cost)
        if len(industry_names) != len(p) or any(not isinstance(s, str) or not s or s == '__CASH__' for s in industry_names):
            raise ValueError('行业名称须完整，且不能使用保留名称 __CASH__')
        p, b, r = np.append(p, cp), np.append(b, cb), np.append(r, cash_return)
        names = np.array([*industry_names, '__CASH__'])
        rb_total = float(b @ r)
        rows = []
        for name in sorted(set(names)):
            mask = names == name
            wp, wb = p[mask].sum(), b[mask].sum()
            rb = float(b[mask] @ r[mask] / wb) if wb > 0 else 0.
            rp = float(p[mask] @ r[mask] / wp) if wp > 0 else rb
            allocation = (wp - wb) * (rb - rb_total)
            selection = wb * (rp - rb)
            interaction = (wp - wb) * (rp - rb)
            rows.append(dict(industry=name, port_weight=wp, bench_weight=wb,
                allocation=allocation, selection=selection, interaction=interaction,
                total_contrib=allocation + selection + interaction))
        details = pl.DataFrame(rows).sort('total_contrib', descending=True)
        a, s, i = [float(details[c].sum()) for c in ('allocation', 'selection', 'interaction')]
        error = gross + fee - (a + s + i + fee)
        if abs(error) > 1e-10:
            raise RuntimeError('Brinson 归因不闭合')
        return BrinsonAttributionReport(gross + fee, gross, a, s, i, fee, error, details)

    def compute_barra_attribution(self, portfolio_weights, benchmark_weights, stock_returns,
                                  exposure_matrix, factor_returns, factor_names, n_industries=6,
                                  *, cash_return=0., portfolio_cost=0., benchmark_cost=0.):
        p, b, r, cp, cb, gross, fee = inputs(portfolio_weights, benchmark_weights, stock_returns,
                                            cash_return, portfolio_cost, benchmark_cost)
        X, f = np.asarray(exposure_matrix, float), np.asarray(factor_returns, float)
        if X.ndim != 2 or X.shape[0] != len(p) or f.shape != (X.shape[1],) or len(factor_names) != len(f):
            raise ValueError('暴露、因子收益、名称必须按证券和因子顺序对齐')
        if len(set(factor_names)) != len(f) or not isinstance(n_industries, int) or not 0 <= n_industries <= len(f) or not np.isfinite(X).all() or not np.isfinite(f).all():
            raise ValueError('因子输入须有限，名称唯一且行业列数合法')
        h = p - b
        factor_pnl = (X.T @ h) * f
        industry = float(factor_pnl[:n_industries].sum())
        styles = dict(zip(factor_names[n_industries:], map(float, factor_pnl[n_industries:])))
        style = sum(styles.values())
        residual = float(h @ (r - X @ f))
        cash = float((cp - cb) * cash_return)
        error = gross + fee - (industry + style + residual + cash + fee)
        if abs(error) > 1e-10:
            raise RuntimeError('因子归因不闭合')
        return BarraAttributionReport(gross + fee, industry, styles, style, residual, cash, fee, error)


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    n = 100
    names = ['行业A', '行业B']
    industries = [names[i % 2] for i in range(n)]
    X = np.column_stack([np.eye(2)[np.arange(n) % 2], rng.normal(size=n)])
    f = np.array([.002, -.001, .003])
    returns = X @ f + rng.normal(0, .01, n)
    benchmark = np.full(n, 1 / n)
    portfolio = rng.uniform(.1, 1, n)
    portfolio *= .9 / portfolio.sum()  # 10% 现金也必须归因
    engine = AsharePerformanceAttributor()
    engine.compute_brinson_attribution(portfolio, benchmark, returns, industries,
                                      portfolio_cost=.0002).print_summary()
    engine.compute_barra_attribution(portfolio, benchmark, returns, X, f,
        names + ['Size'], n_industries=2, portfolio_cost=.0002).print_summary()
