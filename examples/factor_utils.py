"""因子评估共用的小型统计与面板校验工具。"""
import numpy as np
import polars as pl
from scipy import stats


def validate_panel(df, date_col, symbol_col):
    if df.is_empty():
        raise ValueError("面板不能为空")
    if df.select(pl.any_horizontal(pl.col(date_col).is_null(), pl.col(symbol_col).is_null()).any()).item():
        raise ValueError("股票与日期不得为空")
    if df.select(pl.struct(date_col, symbol_col).is_duplicated().any()).item():
        raise ValueError("股票日期键必须唯一")


def annualized_ratio(returns, annual_days=250., annual_rf=0.):
    """算术平均日超额收益 / 日样本标准差 × sqrt(A)；零波动时未定义。"""
    values = np.asarray(returns, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return float('nan')
    daily_rf = (1 + annual_rf) ** (1 / annual_days) - 1
    excess = values - daily_rf
    std = excess.std(ddof=1)
    return float(excess.mean() / std * np.sqrt(annual_days)) if std > 1e-12 else float('nan')


def mean_test(values, hac_lags=0):
    """均值检验：lag=0 为 iid Student t；lag>0 为 Bartlett HAC 渐近 z。

    HAC 带宽由调用者按标签重叠长度等指定，不能解决数据窥探和多次试验。
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if not 0 <= hac_lags < n:
        raise ValueError("HAC 滞后数必须非负且小于有效样本数")
    u = x - x.mean()
    variance = float(u @ u / (n - 1))
    if hac_lags:
        variance += 2 * sum((1 - k / (hac_lags + 1)) * (u[k:] @ u[:-k]) / (n - 1)
                            for k in range(1, hac_lags + 1))
    if variance <= 1e-24:
        return float('nan'), float('nan')
    statistic = float(x.mean() / np.sqrt(variance / n))
    p = 2 * (stats.norm.sf(abs(statistic)) if hac_lags else stats.t.sf(abs(statistic), n - 1))
    return statistic, float(p)
