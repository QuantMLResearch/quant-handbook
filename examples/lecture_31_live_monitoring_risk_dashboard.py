"""教学监控器：相对净值回撤、滚动 TE、特征分布告警与进程内状态。

active_returns_history 为每日 Rp - Rb；回撤按组合/基准相对财富计算。
未传 benchmark_returns_history 时仅适用于基准每日收益为零的演示。
历史必须截至评估日、按交易日排序；该数组接口无法核验源数据的时戳。
"""
from dataclasses import dataclass
from datetime import date
from numbers import Real
import numpy as np


def finite_number(value):
    return isinstance(value, Real) and np.isfinite(value)


@dataclass(frozen=True)
class RiskAlert:
    level: str
    category: str
    message: str


@dataclass(frozen=True)
class RiskControls:
    allow_new_buys: bool
    freeze_model_updates: bool
    risk_aversion_multiplier: float
    active_weight_limit_multiplier: float
    tracking_error_limit: float
    require_manual_recovery: bool


@dataclass
class DashboardTelemetryReport:
    trade_date: str
    realized_tracking_error: float | None
    ex_ante_tracking_error: float
    style_exposures: dict
    current_active_drawdown: float | None
    max_active_drawdown_history: float | None
    circuit_breaker_tier: str
    feature_psi_scores: dict
    alerts: list[RiskAlert]
    controls: RiskControls
    data_valid: bool
    style_threshold: float
    psi_warning: float
    psi_critical: float

    def render_terminal_dashboard(self):
        def percent(x):
            return '不可用' if not finite_number(x) else f'{x:.2%}'
        print(f'[{self.trade_date}] 状态：{self.circuit_breaker_tier}；数据有效：{self.data_valid}')
        print(f'相对净值当前回撤 {percent(self.current_active_drawdown)} / 历史最大 {percent(self.max_active_drawdown_history)}')
        print(f'滚动年化 TE {percent(self.realized_tracking_error)} / 事前 TE {percent(self.ex_ante_tracking_error)}')
        print(f'风格主动暴露阈值 ±{self.style_threshold:g}：{self.style_exposures}')
        print(f'PSI 阈值 {self.psi_warning:g}/{self.psi_critical:g}：{self.feature_psi_scores}')
        print(f'控制指令：{self.controls}')
        for alert in self.alerts:
            print(f'[{alert.level}][{alert.category}] {alert.message}')
        if not self.alerts:
            print('本次已配置检查未触发告警。')


class AshareRiskMonitoringDashboard:
    def __init__(self, target_te=.05, style_threshold=.08,
                 tier1_dd=.04, tier2_dd=.07, tier3_dd=.10, *,
                 recovery1_dd=.02, recovery2_dd=.04, te_window=60,
                 annualization=250, psi_warning=.10, psi_critical=.25,
                 required_styles=()):
        values = [target_te, style_threshold, tier1_dd, tier2_dd, tier3_dd,
                  recovery1_dd, recovery2_dd, annualization, psi_warning, psi_critical]
        if not np.isfinite(values).all() or target_te <= 0 or style_threshold <= 0:
            raise ValueError('阈值必须有限，TE 和风格阈值必须为正')
        if not (0 <= recovery1_dd < tier1_dd <= recovery2_dd < tier2_dd < tier3_dd < 1):
            raise ValueError('回撤阈值及恢复阈值顺序不合法')
        if not isinstance(te_window, int) or te_window < 2 or annualization <= 0:
            raise ValueError('TE 窗口至少 2，年化交易日数为正')
        if not 0 < psi_warning < psi_critical:
            raise ValueError('PSI 阈值顺序不合法')
        self.target_te, self.style_threshold = target_te, style_threshold
        self.tier1_dd, self.tier2_dd, self.tier3_dd = tier1_dd, tier2_dd, tier3_dd
        self.recovery1_dd, self.recovery2_dd = recovery1_dd, recovery2_dd
        self.te_window, self.annualization = te_window, annualization
        self.psi_warning, self.psi_critical = psi_warning, psi_critical
        self.required_styles = set(required_styles)
        self.tier = 0
        self.last_date = None
        self.last_report = None
        self.has_valid_history = False
        self.restart_log = []

    @staticmethod
    def _array(values, name):
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1 or len(arr) < 2 or not np.isfinite(arr).all():
            raise ValueError(f'{name} 需要至少两个有限的一维观测')
        return arr

    def calculate_psi(self, baseline_samples, production_samples, n_bins=10):
        base = self._array(baseline_samples, '基准特征')
        live = self._array(production_samples, '实盘特征')
        if not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError('分箱数至少 2')
        if np.all(base == base[0]):
            c = base[0]
            # 常数基准保留独立点箱，向上/向下移动都不会被吞入同一个箱。
            edges = np.array([-np.inf, np.nextafter(c, -np.inf),
                              np.nextafter(c, np.inf), np.inf])
        else:
            interior = np.unique(np.quantile(base, np.linspace(0, 1, n_bins + 1)[1:-1]))
            edges = np.r_[-np.inf, interior, np.inf]
        bc = np.histogram(base, bins=edges)[0]
        lc = np.histogram(live, bins=edges)[0]
        # 对每箱加入 0.5 伪计数后重新归一化；空箱无 log(0)。
        q = (bc + .5) / (len(base) + .5 * len(bc))
        p = (lc + .5) / (len(live) + .5 * len(lc))
        return float(np.sum((p - q) * np.log(p / q)))

    def _advance_tier(self, dd):
        # 百分比边界允许浮点误差，例如 1 - 0.9 略小于 0.1。
        reaches = lambda threshold: dd >= threshold - 1e-12
        if self.tier == 3 or reaches(self.tier3_dd):
            self.tier = 3
        elif reaches(self.tier2_dd) or (self.tier == 2 and reaches(self.recovery2_dd)):
            self.tier = 2
        elif reaches(self.tier1_dd) or (self.tier >= 1 and reaches(self.recovery1_dd)):
            self.tier = 1
        else:
            self.tier = 0

    def reset_emergency(self, approval_reference):
        """教学中的显式复位入口；外部系统仍须做身份鉴权、持久化和审批。"""
        if not isinstance(approval_reference, str) or not approval_reference.strip():
            raise ValueError('必须提供人工复位审批引用')
        report = self.last_report
        if self.tier != 3 or report is None or not report.data_valid:
            raise ValueError('仅允许在数据有效的三级熔断状态复位')
        if report.current_active_drawdown > self.recovery1_dd + 1e-12:
            raise ValueError('尚未恢复至人工复位的回撤阈值')
        if any(a.level == 'CRITICAL' and a.category != 'DRAWDOWN' for a in report.alerts):
            raise ValueError('尚有其他严重告警，不能复位')
        self.restart_log.append((report.trade_date, approval_reference))
        self.tier = 0

    def evaluate_live_status(self, trade_date, active_returns_history, ex_ante_te,
                             style_exposures, baseline_features, live_features, *,
                             benchmark_returns_history=None):
        day = date.fromisoformat(trade_date)
        if self.last_date is not None and day < self.last_date:
            raise ValueError('状态机不能倒退评估日期')
        self.last_date = day
        alerts, psi = [], {}
        realized_te = current_dd = max_dd = None
        data_valid = True

        def invalid(message):
            nonlocal data_valid
            data_valid = False
            alerts.append(RiskAlert('CRITICAL', 'DATA_QUALITY', message))

        try:
            active = self._array(active_returns_history, '主动收益历史')
            benchmark = (np.zeros_like(active) if benchmark_returns_history is None else
                         self._array(benchmark_returns_history, '基准收益历史'))
            if benchmark.shape != active.shape or np.any(benchmark <= -1) or np.any(active + benchmark <= -1):
                raise ValueError('基准与组合收益必须等长且大于 -100%')
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                relative_nav = np.r_[1., np.cumprod((1 + active + benchmark) / (1 + benchmark))]
                if not np.isfinite(relative_nav).all() or np.any(relative_nav <= 0):
                    raise ValueError('相对净值溢出或失效')
                dd = 1 - relative_nav / np.maximum.accumulate(relative_nav)
                realized_te = float(np.std(active[-self.te_window:], ddof=1) * np.sqrt(self.annualization))
            current_dd, max_dd = float(dd[-1]), float(dd.max())
            # 首次装载完整历史时，也不能漏掉中途已越过紧急线的事件。
            if not self.has_valid_history and max_dd >= self.tier3_dd - 1e-12:
                self.tier = 3
            self.has_valid_history = True
            self._advance_tier(current_dd)
        except (ValueError, TypeError, FloatingPointError) as exc:
            invalid(str(exc))
        if not finite_number(ex_ante_te) or ex_ante_te <= 0:
            invalid('事前 TE 必须是有限正数')
        else:
            if ex_ante_te > self.target_te or (realized_te is not None and realized_te > self.target_te):
                alerts.append(RiskAlert('WARNING', 'TRACKING_ERROR', f'预测或实现 TE 超过目标 {self.target_te:.2%}'))
            if realized_te is not None and realized_te > ex_ante_te * 1.5:
                alerts.append(RiskAlert('WARNING', 'TRACKING_ERROR', '实现 TE 超过预测 TE 的 1.5 倍'))
        for missing in sorted(self.required_styles - style_exposures.keys()):
            invalid(f'缺少要求的风格暴露：{missing}')
        for name, exposure in style_exposures.items():
            if not finite_number(exposure):
                invalid(f'风格暴露 {name} 不是有限数值')
            elif abs(exposure) > self.style_threshold:
                alerts.append(RiskAlert('WARNING', 'STYLE_DRIFT', f'{name} 暴露 {exposure:+.3f} 超过 ±{self.style_threshold:g}'))
        for name, baseline in baseline_features.items():
            if name not in live_features:
                invalid(f'实盘缺少基准特征：{name}')
                continue
            try:
                psi[name] = self.calculate_psi(baseline, live_features[name])
                if psi[name] >= self.psi_warning:
                    level = 'CRITICAL' if psi[name] >= self.psi_critical else 'WARNING'
                    alerts.append(RiskAlert(level, 'FEATURE_DISTRIBUTION',
                                           f'{name} PSI={psi[name]:.4f}；需排查数据与分布变化，不能据此认定概念漂移'))
            except (ValueError, TypeError, FloatingPointError) as exc:
                invalid(f'{name}：{exc}')
        if live_features.keys() - baseline_features.keys():
            invalid('出现未配置基准分布的实盘特征')
        tier_names = ['NORMAL', 'TIER_1_WARNING', 'TIER_2_THROTTLE', 'TIER_3_EMERGENCY']
        if self.tier:
            alerts.append(RiskAlert('CRITICAL' if self.tier == 3 else 'WARNING', 'DRAWDOWN',
                                   f'回撤控制级别 {self.tier}；三级需显式审批复位'))
        critical = any(a.level == 'CRITICAL' for a in alerts)
        controls = RiskControls(
            allow_new_buys=not critical,
            freeze_model_updates=self.tier >= 1 or critical,
            risk_aversion_multiplier=2. if self.tier >= 2 else 1.,
            active_weight_limit_multiplier=.5 if self.tier >= 2 else 1.,
            tracking_error_limit=min(self.target_te, .03) if self.tier >= 2 else self.target_te,
            require_manual_recovery=self.tier == 3)
        status = tier_names[self.tier] if data_valid or self.tier == 3 else 'DATA_ERROR'
        if critical and self.tier == 0 and data_valid:
            status = 'FEATURE_REVIEW'
        report = DashboardTelemetryReport(trade_date, realized_te, ex_ante_te,
            dict(style_exposures), current_dd, max_dd, status, psi, alerts, controls,
            data_valid, self.style_threshold, self.psi_warning, self.psi_critical)
        self.last_report = report
        return report


if __name__ == '__main__':
    monitor = AshareRiskMonitoringDashboard(required_styles=('Size',))
    for day, returns in [('2025-01-03', [-.11, 0]), ('2025-01-06', [-.11, 0, .13])]:
        report = monitor.evaluate_live_status(day, returns, .045, {'Size': .03},
                                             {'factor': np.ones(100)}, {'factor': np.ones(100)})
        report.render_terminal_dashboard()
    monitor.reset_emergency('DEMO-APPROVAL-001')
    print('显式复位日志：', monitor.restart_log)
    print('执行适配器可传 report.controls.allow_new_buys 给第 27 讲；降风险参数仍需优化器重算目标。')
