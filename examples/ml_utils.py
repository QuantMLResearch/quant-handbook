"""时间区间与排序校准工具。所有日期均来自调用方的同一交易日历。"""
import numpy as np
import pandas as pd
import polars as pl


def demo_dates(n, start='2025-01-02'):
    """唯一、有序的工作日，仅为合成示例，不是交易所日历。"""
    return pd.bdate_range(start, periods=n).date.tolist()


def interval_coordinates(dates, label_end_dates=None, forward_horizon=5):
    dates = np.asarray(dates)
    if dates.ndim != 1 or not len(dates) or any(x is None for x in dates):
        raise ValueError('dates 必须为非空且无缺失的一维日期向量')
    unique, starts = np.unique(dates, return_inverse=True)
    if label_end_dates is None:
        if not isinstance(forward_horizon, int) or forward_horizon < 0:
            raise ValueError('forward_horizon 必须为非负整数')
        ends = starts + forward_horizon
    else:
        raw_ends = np.asarray(label_end_dates)
        if raw_ends.shape != dates.shape or any(x is None for x in raw_ends):
            raise ValueError('label_end_dates 必须逐行提供且不能缺失')
        if np.any(raw_ends < dates):
            raise ValueError('标签结束日期不能早于特征日期')
        # 映射到最后一个不晚于结算日的观测日期；缺失日历时不能用固定 H 推断。
        ends = np.searchsorted(unique, raw_ends, side='right') - 1
    group_ends = np.zeros(len(unique), dtype=int)
    np.maximum.at(group_ends, starts, ends)
    return unique, starts, group_ends


def purged_holdout(df, split_date, validation_end, as_of, date_col='trade_date',
                   label_col='ret_fwd_5d', label_end_col='label_end_date'):
    """训练标签严格早于验证开始，验证标签严格早于模型可用日期。"""
    required = {date_col, label_col, label_end_col}
    if not required <= set(df.columns):
        raise ValueError(f'缺少字段: {sorted(required - set(df.columns))}')
    if not split_date < validation_end <= as_of:
        raise ValueError('必须满足 split_date < validation_end <= as_of')
    valid = df.filter(pl.col(label_col).is_finite() & pl.col(label_end_col).is_not_null())
    if valid.filter(pl.col(label_end_col) < pl.col(date_col)).height:
        raise ValueError('标签结束日期不能早于特征日期')
    # 按日期整组清洗，避免同日长短标签拆散至不同阶段。
    valid = valid.with_columns(pl.col(label_end_col).max().over(date_col).alias('_group_end'))
    train = valid.filter((pl.col(date_col) < split_date) & (pl.col('_group_end') < split_date))
    val = valid.filter((pl.col(date_col) >= split_date) & (pl.col(date_col) < validation_end)
                       & (pl.col('_group_end') < as_of))
    if not train.height or not val.height:
        raise ValueError('清洗后训练集或验证集为空')
    return train.drop('_group_end').sort(date_col), val.drop('_group_end').sort(date_col)


def rank_ic(df, score_col, label_col, date_col='trade_date'):
    valid = df.filter(pl.col(score_col).is_finite() & pl.col(label_col).is_finite())
    ranked = valid.with_columns(pl.col(score_col).rank().over(date_col).alias('_pred_rank'),
                               pl.col(label_col).rank().over(date_col).alias('_true_rank'))
    values = ranked.group_by(date_col).agg(pl.corr('_pred_rank', '_true_rank').alias('ic'))
    finite = values.filter(pl.col('ic').is_finite())
    return float(finite['ic'].mean()) if finite.height else float('nan')


class RankReturnCalibrator:
    """在冻结模型的历史样本外预测上拟合截面百分位→H日超额收益。

    调用方须保证 score 是用 model_available_at 时已冻结的模型生成；
    此类检查时间边界，不能替代上游预测溯源。输出不做自动年化。
    """
    def __init__(self):
        self.coefficients = None

    @staticmethod
    def percentiles(df, score_col, date_col):
        return df.select(((pl.col(score_col).rank().over(date_col) - .5)
                          / pl.len().over(date_col) - .5).alias('rank')).to_series().to_numpy()

    def fit(self, forecasts, *, model_available_at, as_of, horizon_days,
            score_col='ml_score', excess_return_col='excess_return',
            date_col='trade_date', label_end_col='label_end_date'):
        if horizon_days < 1 or not forecasts.height:
            raise ValueError('校准需要正数持有期及非空样本外预测')
        required = [score_col, excess_return_col, date_col, label_end_col]
        if not set(required) <= set(forecasts.columns) or any(forecasts[c].null_count() for c in required):
            raise ValueError('校准字段须完整且不能缺失')
        if not forecasts[score_col].is_finite().all() or not forecasts[excess_return_col].is_finite().all():
            raise ValueError('校准分数与收益必须为有限值')
        if forecasts.filter(pl.col(label_end_col) < pl.col(date_col)).height:
            raise ValueError('校准标签结束日不能早于预测日期')
        if forecasts[date_col].min() < model_available_at:
            raise ValueError('校准日期必须在模型冻结后')
        if forecasts[label_end_col].null_count() or forecasts[label_end_col].max() >= as_of:
            raise ValueError('校准标签尚未在 as_of 前成熟')
        x = self.percentiles(forecasts, score_col, date_col)
        y = forecasts[excess_return_col].to_numpy()
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.std(x) < 1e-12:
            raise ValueError('校准输入必须有限且排序有变化')
        self.coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        self.as_of, self.horizon_days = as_of, horizon_days
        return self

    def predict(self, df, score_col='ml_score', date_col='trade_date'):
        if self.coefficients is None:
            raise ValueError('校准器尚未拟合')
        if df.is_empty() or df[score_col].null_count() or not df[score_col].is_finite().all():
            raise ValueError('预测分数必须非空且有限')
        if df[date_col].min() < self.as_of:
            raise ValueError('不能将未来拟合的校准器用于历史决策')
        x = self.percentiles(df, score_col, date_col)
        return self.coefficients[0] + self.coefficients[1] * x
