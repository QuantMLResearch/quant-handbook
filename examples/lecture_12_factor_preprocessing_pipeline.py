"""截面预处理：MAD、行业中位数填充、Z-Score、WLS 残差。

WLS 用 NumPy 求解 sqrt(W)X beta ≈ sqrt(W)y，无额外统计包依赖。
本实现要求行业已知；未知行业应由上游显式编码成单独类别。
"""
import numpy as np


class AshareFactorPreprocessor:
    @staticmethod
    def winsorize_mad(series: np.ndarray, n_sigmas: float = 3.0) -> np.ndarray:
        values = np.asarray(series, dtype=float).copy()
        values[~np.isfinite(values)] = np.nan
        if not np.isfinite(values).any():
            raise ValueError("截面没有有效因子值")
        med = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - med))
        # MAD=0 时保留有限值，不凭空构造尺度；后续标准化会拒绝常数截面。
        if mad == 0:
            return values
        return np.clip(values, med - n_sigmas * 1.4826 * mad, med + n_sigmas * 1.4826 * mad)

    @staticmethod
    def impute_industry_median(series: np.ndarray, industries: np.ndarray) -> np.ndarray:
        values = np.asarray(series, dtype=float).copy()
        groups = np.asarray(industries)
        if values.ndim != 1 or groups.shape != values.shape:
            raise ValueError("行业和因子必须为等长向量")
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in groups):
            raise ValueError("未知行业请显式编码为独立类别")
        valid = np.isfinite(values)
        if not valid.any():
            raise ValueError("全截面缺失，不能填充")
        fallback = np.median(values[valid])
        for industry in np.unique(groups):
            members = groups == industry
            observed = values[members & valid]
            median = np.median(observed) if len(observed) else fallback
            values[members & ~valid] = median
        return values

    @staticmethod
    def standardize_zscore(series: np.ndarray) -> np.ndarray:
        values = np.asarray(series, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("标准化前须完成缺失值填充")
        std = values.std(ddof=0)
        if std <= 1e-12:
            raise ValueError("常数截面无可排序信号")
        return (values - values.mean()) / std

    def neutralize_factor_wls(self, raw_factor, log_mcap, industry_dummies, weights):
        y = np.asarray(raw_factor, dtype=float)
        cap = np.asarray(log_mcap, dtype=float)
        industries = np.asarray(industry_dummies, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if y.ndim != 1 or cap.shape != y.shape or weights.shape != y.shape:
            raise ValueError("因子、市值、权重必须为等长向量")
        if industries.ndim != 2 or industries.shape[0] != len(y):
            raise ValueError("行业矩阵行数必须等于股票数")
        if not np.isfinite(industries).all() or not np.all(np.isin(industries, [0, 1])) or not np.all(industries.sum(axis=1) == 1):
            raise ValueError("每只股票必须对应一个已知行业哑变量")
        # 保留全部行业虚拟变量，不再增加截距；删除当前截面不存在的行业列。
        industries = industries[:, industries.any(axis=0)]
        X = np.column_stack([cap, industries])
        valid = np.isfinite(y) & np.isfinite(X).all(axis=1) & np.isfinite(weights) & (weights > 0)
        if valid.sum() <= X.shape[1]:
            raise ValueError("有效样本不足，不能进行中性化")
        Xv, yv, wv = X[valid], y[valid], weights[valid]
        sqrt_w = np.sqrt(wv)
        beta, _, rank, _ = np.linalg.lstsq(Xv * sqrt_w[:, None], yv * sqrt_w, rcond=None)
        if rank < Xv.shape[1]:
            raise ValueError("中性化设计矩阵秩不足")
        residuals = np.full(len(y), np.nan)
        residuals[valid] = yv - Xv @ beta
        # 正交条件是 X.T W residual = 0，不是无权相关系数必为 0。
        return residuals

    def transform(self, raw_factor, log_mcap, industries, weights):
        """按明确顺序执行四步；返回残差及各步数据，便于审计。"""
        winsorized = self.winsorize_mad(raw_factor)
        imputed = self.impute_industry_median(winsorized, industries)
        standardized = self.standardize_zscore(imputed)
        _, codes = np.unique(industries, return_inverse=True)
        dummies = np.eye(codes.max() + 1)[codes]
        residuals = self.neutralize_factor_wls(standardized, log_mcap, dummies, weights)
        return residuals, dict(winsorized=winsorized, imputed=imputed,
                               standardized=standardized, industry_dummies=dummies)


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    n = 500
    log_mcap = rng.normal(10, 1.2, n)
    industries = np.arange(n) % 5
    weights = np.exp(log_mcap / 2)  # 研究权重选择，不宣称等于误差方差倒数
    raw = .5 * log_mcap + rng.normal(size=n)
    raw[:4] = [500., -300., np.nan, np.inf]
    residuals, steps = AshareFactorPreprocessor().transform(raw, log_mcap, industries, weights)
    X = np.column_stack([log_mcap, steps['industry_dummies']])
    assert np.isfinite(residuals).all()
    np.testing.assert_allclose(X.T @ (weights * residuals), 0., atol=1e-7)
    print('四步预处理完成；缺失值已填充，WLS 加权正交条件通过。')
    print('无权市值相关系数（不要求为零）：', np.corrcoef(residuals, log_mcap)[0, 1])
