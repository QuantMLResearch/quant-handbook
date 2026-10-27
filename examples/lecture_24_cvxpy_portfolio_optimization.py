"""
教学用 A 股指数增强凸优化器 (AshareIndexEnhancementOptimizer)
基于 cvxpy 与 Barra 结构化风险模型构建
实现：行业硬中性、市值约束、个股上下界限制与二次跟踪误差惩罚
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cvxpy as cp
import numpy as np
from examples.portfolio_utils import validate_risk_inputs, solve_checked


@dataclass
class OptimizationResult:
    """优化结果与组合全景体检报告"""
    horizon_days: int
    optimal_weights: np.ndarray
    expected_active_return: float
    annualized_tracking_error: float
    active_share: float
    effective_stocks_count: int
    industry_max_deviation: float
    size_style_deviation: float
    status: str

    def print_dossier(self):
        """格式化打印优化报告"""
        print("=" * 70)
        print(f"🎯 指数增强凸优化求解报告 [求解状态: {self.status}]")
        print("=" * 70)
        print("【核心收益与主动风险】")
        print(f"  • 预期持有期主动超额 (Alpha)  : {self.expected_active_return * 100:8.2f}%")
        print(f"  • 预期年化跟踪误差 (TE)     : {self.annualized_tracking_error * 100:8.2f}%")
        print(f"  • 收益预测持有期             : {self.horizon_days} 个交易日")
        print(f"  • 主动份额 (Active Share)    : {self.active_share * 100:8.2f}%")
        print(f"  • 有效持仓股票数 (ENP)      : {self.effective_stocks_count:8d} 支")
        print("-" * 70)
        print("【风控合规穿透检查】")
        print(f"  • 给定行业最大净暴露偏差     : {self.industry_max_deviation * 100:8.4f}% ")
        print(f"  • 对数市值风格净偏离 (Size) : {self.size_style_deviation:8.4f} ")
        print("=" * 70)


class AshareIndexEnhancementOptimizer:
    """
    A 股指数增强二次凸优化器
    """

    def __init__(
        self,
        risk_aversion: float = 2.0,      # 风险厌恶系数 lambda
        max_active_weight: float = 0.015, # 单只股票最大主动超配权重 +1.5%
        size_tolerance: float = 0.05,     # 市值风格偏离硬容忍上限
        max_tracking_error: float = 0.045,
        horizon_days: int = 5,
        annual_factor: float = 250.
    ):
        if risk_aversion < 0 or max_active_weight < 0 or size_tolerance < 0 or max_tracking_error <= 0 or horizon_days < 1 or annual_factor <= 0:
            raise ValueError("风险/约束参数范围不合法")
        self.max_tracking_error = max_tracking_error
        self.horizon_days, self.annual_factor = horizon_days, annual_factor
        self.risk_aversion = risk_aversion
        self.max_active_weight = max_active_weight
        self.size_tolerance = size_tolerance

    def optimize(
        self,
        expected_returns: np.ndarray,          # (N,) H 日预期超额收益率；不能传入未经校准的排序分数
        benchmark_weights: np.ndarray,     # (N,) 基准指数成分股权重 (sum=1)
        exposure_matrix: np.ndarray,       # (N, K) 因子暴露矩阵 (前 K_ind 列为行业, 之后为风格)
        factor_covariance: np.ndarray,     # (K, K) 因子协方差矩阵 F
        specific_variance: np.ndarray,     # (N,) 特异性方差对角元素 Δ
        n_industries: int = 6,             # 行业因子列数
        size_col_idx: int = 6              # 市值因子列索引
    ) -> OptimizationResult:
        """
        构建带二次风险约束的凸优化问题 问题并求解最优持仓权重
        """
        expected_returns, benchmark_weights, exposure_matrix, factor_covariance, specific_variance = validate_risk_inputs(
            expected_returns, benchmark_weights, exposure_matrix, factor_covariance, specific_variance, n_industries, size_col_idx)
        N = len(expected_returns)

        # 1. 定义优化决策变量: 投资组合最终权重 w (N,)
        w = cp.Variable(N, nonneg=True) # 单边做多约束: w_i >= 0
        h = w - benchmark_weights       # 主动偏离向量 h

        # 2. 目标函数: max h^T alpha - (lambda / 2) * (h^T Σ h)
        # 利用 Barra 形式降低维度: h^T Σ h = y^T F y + h^T Δ h
        y = exposure_matrix.T @ h       # 主动因子暴露 (K,)
        
        # 风险项: 因子风险 + 特异性风险
        factor_risk = cp.quad_form(y, factor_covariance)
        specific_risk = cp.sum(cp.multiply(specific_variance, cp.square(h)))
        portfolio_risk = factor_risk + specific_risk

        expected_return = expected_returns @ h
        period_risk = portfolio_risk * self.horizon_days / self.annual_factor
        objective = cp.Maximize(expected_return - (self.risk_aversion / 2.0) * period_risk)

        # 3. 约束条件构建
        constraints = [
            cp.sum(w) == 1.0,           # 全额投资预算约束
            w >= 0,
            portfolio_risk <= self.max_tracking_error ** 2,
        ]

        # 3.1 行业硬中性约束: X_ind^T (w - w_b) = 0
        X_ind = exposure_matrix[:, :n_industries]
        constraints.append(X_ind.T @ h == 0.0)

        # 3.2 市值风格偏离软容忍约束: |X_size^T h| <= delta
        size_exp = exposure_matrix[:, size_col_idx]
        constraints.append(cp.abs(size_exp @ h) <= self.size_tolerance)

        # 3.3 个股主动权重上限约束: w_i - w_bi <= max_active_weight
        constraints.append(h <= self.max_active_weight)

        # 二次 TE 硬约束需要锥求解器；求解后检查所有约束，不裁剪权重。
        prob = cp.Problem(objective, constraints)
        opt_w = solve_checked(prob, w, constraints)

        # 5. 体检指标测算
        active_w = opt_w - benchmark_weights
        exp_ret = float(expected_returns @ active_w)
        
        # 跟踪误差
        y_opt = exposure_matrix.T @ active_w
        f_risk = float(y_opt.T @ factor_covariance @ y_opt)
        s_risk = float(np.sum(specific_variance * (active_w ** 2)))
        te_ann = np.sqrt(f_risk + s_risk)

        # 主动份额 (Active Share = 1/2 * sum(|w - w_b|))
        active_share = float(0.5 * np.sum(np.abs(active_w)))

        # 有效持仓股票数 (Herfindahl-Hirschman: 1 / sum(w_i^2))
        enp = int(np.round(1.0 / np.sum(opt_w ** 2)))

        # 行业最大暴露偏差
        ind_bias = np.abs(X_ind.T @ active_w)
        max_ind_dev = float(np.max(ind_bias))

        # 市值暴露偏差
        size_dev = float(size_exp @ active_w)

        return OptimizationResult(
            horizon_days=self.horizon_days,
            optimal_weights=opt_w,
            expected_active_return=exp_ret,
            annualized_tracking_error=te_ann,
            active_share=active_share,
            effective_stocks_count=enp,
            industry_max_deviation=max_ind_dev,
            size_style_deviation=size_dev,
            status=prob.status
        )


# ================= 单元测试与指增组合优化实战 =================
if __name__ == "__main__":
    np.random.seed(42)
    n_stocks = 500  # 模拟中证 500 成分股池
    n_factors = 7   # 6 个行业因子 + 1 个 Size 因子

    # 1. 模拟基准指数权重 (如中证 500 等权偏离，各成分股权重在 0.1% ~ 0.5% 之间)
    raw_b = np.random.uniform(0.1, 0.5, n_stocks)
    w_benchmark = raw_b / np.sum(raw_b)

    # 2. 模拟 Barra 因子暴露
    X_mat = np.zeros((n_stocks, n_factors))
    ind_assign = np.random.choice(6, n_stocks)
    for i in range(n_stocks):
        X_mat[i, ind_assign[i]] = 1.0  # 行业 One-Hot
    # 最后一列为标准化市值因子
    X_mat[:, 6] = np.random.normal(0, 1.0, n_stocks)

    # 3. 模拟 Barra 风险参数
    F_cov = np.eye(n_factors) * 0.04
    spec_var = np.random.uniform(0.04, 0.12, n_stocks) # 个股年化特异方差

    # 4. 模拟持有期收益率预测，独立于排序模型分数
    alpha = np.random.normal(0.02, 0.05, n_stocks) * 5 / 250  # 合成 5 日预期收益，不是排序分数

    # 5. 实例化并求解指数增强优化
    optimizer = AshareIndexEnhancementOptimizer(
        risk_aversion=3.0,
        max_active_weight=0.012, # 单只股票最多超配 1.2%
        size_tolerance=0.03      # 市值偏离严格限制在 0.03 以内
    )

    report = optimizer.optimize(
        expected_returns=alpha,
        benchmark_weights=w_benchmark,
        exposure_matrix=X_mat,
        factor_covariance=F_cov,
        specific_variance=spec_var,
        n_industries=6,
        size_col_idx=6
    )

    # 输出买方体检卷宗
    report.print_dossier()
