"""优化器共用的输入与求解结果校验；协方差输入统一为年化值。"""
import numpy as np


def validate_risk_inputs(alpha, benchmark, exposure, factor_cov, specific_var,
                         n_industries, size_col_idx):
    arrays = [np.asarray(a, dtype=float) for a in (alpha, benchmark, exposure, factor_cov, specific_var)]
    alpha, benchmark, exposure, factor_cov, specific_var = arrays
    n = len(alpha)
    if alpha.shape != (n,) or n == 0 or benchmark.shape != (n,) or specific_var.shape != (n,):
        raise ValueError('收益、权重、特异性方差必须为同长度非空向量')
    if exposure.ndim != 2 or exposure.shape[0] != n or factor_cov.shape != (exposure.shape[1],) * 2:
        raise ValueError('暴露/因子协方差形状不匹配')
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('优化输入必须全部为有限数值')
    if np.any(benchmark < 0) or not np.isclose(benchmark.sum(), 1, atol=1e-8):
        raise ValueError('基准必须非负且权重和为 1')
    if not 1 <= n_industries <= exposure.shape[1] or not 0 <= size_col_idx < exposure.shape[1]:
        raise ValueError('行业/风格列索引越界')
    if not np.allclose(factor_cov, factor_cov.T, atol=1e-10) or np.linalg.eigvalsh(factor_cov).min() < -1e-10 or np.any(specific_var < 0):
        raise ValueError('风险协方差必须对称半正定，特异性方差必须非负')
    return arrays


def solve_checked(problem, weights, constraints, tolerance=1e-6):
    import cvxpy as cp
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-9, tol_feas=1e-9, tol_gap_rel=1e-9)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or weights.value is None:
        raise RuntimeError(f'优化不可行或求解失败: {problem.status}')
    result = np.asarray(weights.value).ravel()
    violation = max(float(np.max(c.violation())) for c in constraints)
    if not np.isfinite(result).all() or violation > tolerance:
        raise RuntimeError(f'求解结果违反约束，最大残差 {violation:g}')
    # 不裁剪/重新归一化：那会破坏行业、TE 或换手约束。
    return result
