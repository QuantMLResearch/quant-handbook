"""累计持有期 IC、单日滞后 IC 与调仓成本情景分析。

输入 ret_1d[t] 是截至 t 的后向日收益。因子 f[t] 只配对 t+1 及以后收益。
示例需要完整、等间隔交易日面板；停牌和退市估值应上游处理。
情景 Sharpe 由明确假设给出，不是持仓回测或实盘最优调仓建议。
"""
from dataclasses import dataclass
from typing import Dict
import numpy as np
import polars as pl
from scipy import stats
from examples.factor_utils import validate_panel


@dataclass
class DecayAnalysisReport:
    factor_name: str
    ic_by_horizon: Dict[int, float]
    marginal_ic_by_lag: Dict[int, float]
    half_life_days: float | None
    fit_status: str
    lag1_autocorr: float
    lag5_autocorr: float
    daily_top_membership_turnover: float

    def print_summary(self):
        print(f'因子诊断：{self.factor_name}')
        print('累计持有期 IC：', self.ic_by_horizon)
        print('单日滞后 IC（用于衰减拟合）：', self.marginal_ic_by_lag)
        print('半衰期：', self.half_life_days, '；拟合状态：', self.fit_status)
        print('1/5 日秩自相关：', self.lag1_autocorr, self.lag5_autocorr)
        print('Top 名单替换率（未计权重漂移）：', self.daily_top_membership_turnover)


class AshareFactorDecayEngine:
    def __init__(self, date_col='trade_date', symbol_col='symbol', roundtrip_cost=.0035, annual_days=250.):
        if not 0 <= roundtrip_cost < 1 or annual_days <= 0:
            raise ValueError('成本和年化天数无效')
        self.date_col = date_col
        self.symbol_col = symbol_col
        self.roundtrip_cost = roundtrip_cost
        self.annual_days = annual_days

    def forward_returns(self, df, ret_col, horizons):
        """精确计算 prod(1+r[t+1:t+h+1])-1；尾部不足 h 天保持 null。"""
        validate_panel(df, self.date_col, self.symbol_col)
        if not horizons or any(not isinstance(h, int) or h < 1 for h in horizons):
            raise ValueError('horizons 必须为正整数列表')
        n_dates = df[self.date_col].n_unique()
        if df.group_by(self.symbol_col).len()['len'].min() != n_dates:
            raise ValueError('需要完整交易日面板，不能跨缺失日期直接 shift')
        if df.select((~pl.col(ret_col).is_finite() | (pl.col(ret_col) < -1)).fill_null(True).any()).item():
            raise ValueError('日收益必须有限且不小于 -100%，缺失估值需显式处理')
        work = df.sort([self.symbol_col, self.date_col])
        for h in sorted(set(horizons)):
            gross = pl.lit(1.)
            for lag in range(1, h + 1):
                gross = gross * (1 + pl.col(ret_col).shift(-lag).over(self.symbol_col))
            work = work.with_columns((gross - 1).alias(f'ret_fwd_{h}d'))
        return work

    def _rank_ic(self, df, factor, target):
        valid = df.filter(pl.col(factor).is_finite() & pl.col(target).is_finite())
        daily = (valid.with_columns(pl.col(factor).rank().over(self.date_col).alias('_f'),
                                     pl.col(target).rank().over(self.date_col).alias('_r'))
                 .group_by(self.date_col).agg(pl.corr('_f', '_r').alias('ic'), pl.len().alias('n')))
        values = daily.filter((pl.col('n') >= 3) & pl.col('ic').is_finite())['ic']
        return float(values.mean()) if len(values) else float('nan')

    @staticmethod
    def fit_half_life(marginal_ic, direction=1):
        """方向须事先指定；不筛掉不利点来强行凑出衰减模型。"""
        if direction not in [-1, 1]:
            raise ValueError('direction 必须为事先指定的 +1 或 -1')
        hs = np.array(sorted(marginal_ic), dtype=float)
        values = direction * np.array([marginal_ic[int(h)] for h in hs])
        if len(hs) < 3 or not np.isfinite(values).all() or np.any(values <= 0):
            return None, '未拟合：需要至少三个有限、同方向的单日滞后 IC'
        slope, _, _, _, _ = stats.linregress(hs, np.log(values))
        if slope >= -1e-10:
            return None, '未拟合：未观察到指数衰减（平坦或上升）'
        return float(np.log(2) / -slope), '描述性指数拟合；未给出置信区间'

    def analyze_decay(self, df, factor_col, ret_1d_col, horizons=None, top_ratio=.10, direction=1):
        horizons = [1, 2, 3, 5, 10, 20] if horizons is None else list(horizons)
        if not 0 < top_ratio <= 1 or direction not in [-1, 1]:
            raise ValueError('top_ratio 或方向无效')
        work = self.forward_returns(df, ret_1d_col, horizons)
        if work[self.date_col].n_unique() <= max(horizons):
            raise ValueError('样本长度必须大于最大前瞻期')
        if work.select((~pl.col(factor_col).is_finite()).fill_null(True).any()).item():
            raise ValueError('因子缺失请先执行截面预处理')
        cumulative, marginal = {}, {}
        for h in sorted(set(horizons)):
            cumulative[h] = self._rank_ic(work, factor_col, f'ret_fwd_{h}d')
            work = work.with_columns(pl.col(ret_1d_col).shift(-h).over(self.symbol_col).alias('_lag_ret'))
            marginal[h] = self._rank_ic(work, factor_col, '_lag_ret')
        half_life, fit_status = self.fit_half_life(marginal, direction)
        autocorr = {}
        for lag in [1, 5]:
            work = work.with_columns(pl.col(factor_col).shift(lag).over(self.symbol_col).alias('_lag_factor'))
            autocorr[lag] = self._rank_ic(work, factor_col, '_lag_factor')
        # 等权目标名单替换比例，不冒充计入涨跌漂移的成交换手率。
        previous, replacement = None, []
        for day in work[self.date_col].unique().sort():
            cross = work.filter(pl.col(self.date_col) == day).sort(
                [factor_col, self.symbol_col], descending=[direction == 1, False])
            k = max(1, int(np.ceil(cross.height * top_ratio)))
            selected = set(cross.head(k)[self.symbol_col].to_list())
            if previous is not None:
                replacement.append(len(selected - previous) / k)
            previous = selected
        return DecayAnalysisReport(factor_col, cumulative, marginal, half_life, fit_status,
                                   autocorr[1], autocorr[5], float(np.mean(replacement)))

    def assumption_scenarios(self, report, *, return_scale, assumed_annual_vol,
                             turnover_exponent=.4, annual_rf=.02, direction=1):
        """纯假设情景表：所有未由持仓回测得到的参数都公开；不输出最优周期。"""
        if return_scale < 0 or assumed_annual_vol <= 0 or direction not in [-1, 1]:
            raise ValueError('情景参数无效')
        rows = []
        for p, ic in report.ic_by_horizon.items():
            if not np.isfinite(ic):
                continue
            single_turnover = min(1., report.daily_top_membership_turnover * p**turnover_exponent)
            cost = self.annual_days / p * single_turnover * self.roundtrip_cost
            gross = direction * ic * np.sqrt(self.annual_days / p) * return_scale
            rows.append(dict(period=p, assumed_gross_return=gross, assumed_cost=cost,
                             assumed_annual_vol=assumed_annual_vol,
                             scenario_sharpe=(gross - cost - annual_rf) / assumed_annual_vol))
        return pl.DataFrame(rows)


if __name__ == '__main__':
    import pandas as pd
    rng = np.random.default_rng(42)
    dates = pd.bdate_range('2025-01-01', periods=200).date.tolist()
    n = 400
    latent = np.zeros((len(dates), n))
    returns = np.zeros_like(latent)
    # AR(1) 信号的理论衰减半衰期为 4 天；收益依赖上一期信号。
    rho = 2 ** (-1 / 4)
    latent[0] = rng.normal(size=n)
    for t in range(1, len(dates)):
        returns[t] = .005 * latent[t-1] + rng.normal(0, .02, n)
        latent[t] = rho * latent[t-1] + np.sqrt(1-rho*rho) * rng.normal(size=n)
    frame = pl.DataFrame([dict(trade_date=d, symbol=f'DEMO_{s:04}',
                               factor_reversal=latent[t, s], ret_1d=returns[t, s])
                          for t, d in enumerate(dates) for s in range(n)])
    engine = AshareFactorDecayEngine()
    report = engine.analyze_decay(frame, 'factor_reversal', 'ret_1d', horizons=[1, 2, 3, 5, 10])
    report.print_summary()
    print('以下为假设情景，不是调仓回测：')
    print(engine.assumption_scenarios(report, return_scale=.18, assumed_annual_vol=.16))
