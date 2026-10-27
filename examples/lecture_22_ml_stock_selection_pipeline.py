"""已对齐特征面板→清洗→训练/早停→独立预测→收盘目标权重的教学流水线。

不负责原始 PIT 数据生成、WLS 中性化、下一日订单撮合；这些边界必须由上游/下游接入。
"""
import numpy as np
import polars as pl
from examples.lecture_19_lightgbm_catboost_ranking import AshareLightGBMRanker
from examples.ml_utils import demo_dates, purged_holdout, rank_ic, RankReturnCalibrator


class IndustrialMLStockPickerPipeline:
    def __init__(self, feature_cols, label_col='ret_fwd_5d', date_col='trade_date',
                 symbol_col='symbol', industry_col='industry', cap_col='market_cap',
                 top_k=50, random_state=42):
        if top_k < 1:
            raise ValueError('top_k 必须为正数')
        self.feature_cols, self.label_col = feature_cols, label_col
        self.date_col, self.symbol_col = date_col, symbol_col
        self.industry_col, self.cap_col = industry_col, cap_col
        self.top_k, self.random_state = top_k, random_state
        self.model = None
        self.available_at = None

    def preprocess_cross_section(self, df):
        if df.select(pl.struct(self.date_col, self.symbol_col).is_duplicated().any()).item():
            raise ValueError('股票/日期键重复')
        exprs = []
        for col in self.feature_cols:
            value = pl.when(pl.col(col).is_finite()).then(pl.col(col)).otherwise(None)
            median = value.median().over(self.date_col)
            mad = (value - median).abs().median().over(self.date_col)
            # 常数截面保留；缺失值以该日中位数填充，全缺失截面填零。
            clipped = value.fill_null(median).clip(median - 3 * 1.4826 * mad,
                                                   median + 3 * 1.4826 * mad)
            std = clipped.std().over(self.date_col)
            z = (clipped - clipped.mean().over(self.date_col)) / std
            exprs.append(pl.when(std > 1e-12).then(z).otherwise(0.).fill_null(0.).alias(col))
        return df.with_columns(exprs)

    def train_purged_ranker(self, df, split_date, *, validation_end, as_of,
                            early_stopping_rounds=30, label_end_col='label_end_date'):
        train, val = purged_holdout(df, split_date, validation_end, as_of,
                                    self.date_col, self.label_col, label_end_col)
        self.ranker = AshareLightGBMRanker(min_child_samples=30, n_estimators=120,
                                          random_state=self.random_state)
        self.ranker.fit(train, val, self.feature_cols, self.label_col,
                        date_col=self.date_col, label_end_col=label_end_col,
                        early_stopping_rounds=early_stopping_rounds, available_at=as_of)
        self.model, self.available_at = self.ranker.model, as_of
        self.training_audit = {'train_end': train[self.date_col].max(),
                               'train_label_end': train[label_end_col].max(),
                               'validation_label_end': val[label_end_col].max(), 'available_at': as_of}
        scored = val.with_columns(pl.Series('ml_score', self.ranker.predict(val)))
        return rank_ic(scored, 'ml_score', self.label_col, self.date_col)

    def score(self, df):
        if self.model is None or self.available_at is None:
            raise ValueError('模型尚未训练')
        if df.is_empty():
            return df.with_columns(pl.Series('ml_score', [], dtype=pl.Float64))
        if df[self.date_col].min() < self.available_at:
            raise ValueError('不能用后来训练的模型生成历史交易信号')
        return df.with_columns(pl.Series('ml_score', self.model.predict(df.select(self.feature_cols).to_numpy())))

    def generate_daily_portfolio(self, current_cross_section_df, trade_date, filter_st=True):
        """收盘信号权重。仅使用当日已知 ST/停牌状态，不预知次日开盘涨停。

        全部过滤时返回空持仓清单（现金），不代表可以卖出被停牌的已有仓位。
        """
        today = current_cross_section_df.filter(pl.col(self.date_col) == trade_date)
        if today.is_empty():
            raise ValueError('没有指定日期的截面数据')
        required = ['is_suspended'] + (['is_st'] if filter_st else [])
        if not set(required) <= set(today.columns):
            raise ValueError(f'缺少必需状态列: {required}')
        if today[self.symbol_col].n_unique() != today.height:
            raise ValueError('股票/日期键重复')
        clean = today.filter(pl.all_horizontal([pl.col(c) == 0 for c in required]))
        scored = self.score(clean).sort(['ml_score', self.symbol_col], descending=[True, False]).head(self.top_k)
        result = scored.with_columns(
            pl.lit(1. / scored.height if scored.height else 0.).alias('target_weight'),
            pl.int_range(1, scored.height + 1).alias('rank_order'))
        cols = ['rank_order', self.symbol_col, self.industry_col, 'ml_score', 'target_weight']
        if self.cap_col in result.columns:
            cols.append(self.cap_col)
        return result.select(cols)


def demo_panel(n_days=130, n_stocks=100):
    rng = np.random.default_rng(42)
    calendar = demo_dates(n_days + 5)
    features = ['momentum', 'value', 'quality']
    records = []
    for t, day in enumerate(calendar[:n_days]):
        x = rng.normal(size=(n_stocks, len(features)))
        y = .005 * x[:, 0] + .006 * x[:, 2] + rng.normal(0, .025, n_stocks)
        for i in range(n_stocks):
            records.append(dict(trade_date=day, symbol=f'{i:06d}.SH', industry=f'行业{i % 5}',
                market_cap=float(np.exp(rng.normal(5, 1))), is_st=int(i % 37 == 0), is_suspended=0,
                label_end_date=calendar[t + 5], ret_fwd_5d=y[i], **dict(zip(features, x[i]))))
    return pl.DataFrame(records), features, calendar


if __name__ == '__main__':
    raw, features, calendar = demo_panel()
    pipeline = IndustrialMLStockPickerPipeline(features, top_k=10)
    panel = pipeline.preprocess_cross_section(raw)
    val_ic = pipeline.train_purged_ranker(panel, calendar[70], validation_end=calendar[90], as_of=calendar[100])
    print(f'早停验证 Rank IC（非最终业绩）: {val_ic:.4f}；时间审计: {pipeline.training_audit}')
    # 模型冻结后 100～109 日的预测：只用于校准，标签在 115 日前成熟。
    calibration = pipeline.score(panel.filter((pl.col('trade_date') >= calendar[100])
                                               & (pl.col('trade_date') < calendar[110])))
    calibration = calibration.with_columns((pl.col('ret_fwd_5d') - pl.col('ret_fwd_5d').mean().over('trade_date')).alias('excess_return'))
    calibrator = RankReturnCalibrator().fit(calibration, model_available_at=calendar[100],
                                            as_of=calendar[115], horizon_days=5)
    # 115～123 日留作最终诊断，不参加早停或校准；在 129 日标签均已成熟。
    test = pipeline.score(panel.filter((pl.col('trade_date') >= calendar[115])
                                       & (pl.col('label_end_date') < calendar[129])))
    print(f'独立测试 Rank IC（合成数据）: {rank_ic(test, "ml_score", "ret_fwd_5d"):.4f}')
    today = pipeline.score(panel.filter(pl.col('trade_date') == calendar[129]))
    alpha_5d = calibrator.predict(today)
    print(f'校准后 5 日预期超额范围: {alpha_5d.min():.4%}～{alpha_5d.max():.4%}；未自动年化')
    print(pipeline.generate_daily_portfolio(panel, calendar[129]))
    print('以上为收盘目标权重；次日未成交订单保留现金/原持仓，由执行引擎处理。')
