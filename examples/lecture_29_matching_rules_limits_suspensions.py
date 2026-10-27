"""保守开盘撮合与登记日权益账本。日内排队、税补扣和零碎股分配不在本例范围。"""
from dataclasses import dataclass, field
from datetime import date
import numpy as np


def as_date(value):
    return value if type(value) is date else date.fromisoformat(str(value))


@dataclass(frozen=True)
class LotRule:
    min_buy: int = 100
    step: int = 100

    def __post_init__(self):
        if not isinstance(self.min_buy, int) or not isinstance(self.step, int) or min(self.min_buy, self.step) < 1:
            raise ValueError('最小申报量和步长须为正整数')

    def floor_buy(self, shares):
        return 0 if shares < self.min_buy else self.min_buy + int((shares - self.min_buy) // self.step) * self.step


@dataclass(frozen=True)
class DailyMarketBar:
    """撮合仅访问 open/限价/停牌及显式开盘可成交量；high/low/close 用于收盘展示。"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    limit_up_price: float | None
    limit_down_price: float | None
    is_suspended: bool = False
    opening_buy_liquidity: int = 0
    opening_sell_liquidity: int = 0
    adv_20d: float = 1e8       # 决策前已知的历史成交额
    daily_volatility: float = .02

    def __post_init__(self):
        if not np.isfinite([self.open, self.adv_20d, self.daily_volatility]).all() or min(self.open, self.adv_20d) <= 0 or self.daily_volatility < 0:
            raise ValueError('开盘价、ADV 须为正，波动率非负且有限')
        for q in (self.opening_buy_liquidity, self.opening_sell_liquidity):
            if not isinstance(q, int) or q < 0:
                raise ValueError('开盘可成交量须为非负整数，未知时为 0')
        for bound in (self.limit_up_price, self.limit_down_price):
            if bound is not None and (not np.isfinite(bound) or bound <= 0):
                raise ValueError('涨跌停价须为正；无限价时显式传 None')
        if self.limit_up_price is not None and self.open > self.limit_up_price + 1e-8:
            raise ValueError('开盘价超过涨停价')
        if self.limit_down_price is not None and self.open < self.limit_down_price - 1e-8:
            raise ValueError('开盘价低于跌停价')


class AshareExecutionMatchingSimulator:
    """不掌管持仓或现金。开盘触及不利限价时保守拒单，不推断盘中是否开板。"""
    def __init__(self, price_slippage=.01):
        if not np.isfinite(price_slippage) or price_slippage < 0:
            raise ValueError('固定滑点须非负且有限')
        self.slippage = price_slippage

    def simulate_order_execution(self, symbol, direction, order_shares, market_bar,
                                 *, lot_rule=None, used_liquidity=0):
        rule = lot_rule or LotRule()
        if symbol != market_bar.symbol or direction not in ('BUY', 'SELL'):
            raise ValueError('证券或交易方向不匹配')
        if not isinstance(order_shares, int) or order_shares <= 0 or used_liquidity < 0:
            raise ValueError('委托数量须为正整数')
        if market_bar.is_suspended:
            return False, 0, 0., 'SUSPENDED'
        upper, lower = market_bar.limit_up_price, market_bar.limit_down_price
        if direction == 'BUY':
            if upper is not None and market_bar.open >= upper - 1e-8:
                return False, 0, 0., 'OPEN_AT_LIMIT_UP'
            quantity = rule.floor_buy(min(order_shares, max(0, market_bar.opening_buy_liquidity - used_liquidity)))
            price = market_bar.open + self.slippage
            if upper is not None and price > upper:
                return False, 0, 0., 'SLIPPAGE_OUTSIDE_LIMIT'
        else:
            if lower is not None and market_bar.open <= lower + 1e-8:
                return False, 0, 0., 'OPEN_AT_LIMIT_DOWN'
            quantity = min(order_shares, max(0, market_bar.opening_sell_liquidity - used_liquidity))
            price = market_bar.open - self.slippage
            if price <= 0 or (lower is not None and price < lower):
                return False, 0, 0., 'SLIPPAGE_OUTSIDE_LIMIT'
        if quantity == 0:
            return False, 0, 0., 'NO_OPENING_LIQUIDITY_OR_LOT'
        return True, quantity, price, 'FILLED' if quantity == order_shares else 'PARTIAL'


@dataclass(frozen=True)
class CorporateAction:
    action_id: str
    symbol: str
    record_date: str
    ex_date: str
    cash_payment_date: str | None = None
    share_listing_date: str | None = None
    cash_div_per_share: float = 0.
    bonus_share_ratio: float = 0.

    def __post_init__(self):
        if not self.action_id or not self.symbol or as_date(self.record_date) >= as_date(self.ex_date):
            raise ValueError('事件须有唯一 ID，且登记日早于除权日')
        if not np.isfinite([self.cash_div_per_share, self.bonus_share_ratio]).all() or min(self.cash_div_per_share, self.bonus_share_ratio) < 0:
            raise ValueError('现金和股份权益须非负且有限')
        for amount, when in ((self.cash_div_per_share, self.cash_payment_date), (self.bonus_share_ratio, self.share_listing_date)):
            if amount and (when is None or as_date(when) < as_date(self.ex_date)):
                raise ValueError('必须提供不早于除权日的实际到账/上市日期')


@dataclass
class CorporateSettlement:
    cash_delta: float = 0.
    receivable_delta: float = 0.
    shares_delta: dict = field(default_factory=dict)
    listed_delta: dict = field(default_factory=dict)


class CorporateActionBook:
    """登记日收盘冻结数量；除权日确认应收/股份；支付与上市分开，重复调用无重复到账。"""
    def __init__(self, actions=()):
        self.actions = list(actions)
        if len({a.action_id for a in self.actions}) != len(self.actions):
            raise ValueError('公司行为 ID 重复')
        self.entitlements = {}
        self.processed = set()
        self.last_processed_date = None

    def capture_record_date(self, current_date, holdings):
        day = as_date(current_date)
        for a in self.actions:
            if as_date(a.record_date) == day:
                q = holdings.get(a.symbol, 0)
                if not isinstance(q, int) or q < 0:
                    raise ValueError('登记日持股数须为非负整数')
                if a.action_id in self.entitlements and self.entitlements[a.action_id] != q:
                    raise ValueError('不能覆盖已冻结的登记日权益')
                self.entitlements[a.action_id] = q

    def process_day(self, current_date):
        day = as_date(current_date)
        if self.last_processed_date is not None and day < self.last_processed_date:
            raise ValueError('公司行为账本不允许时间倒退')
        due = [a for a in self.actions if as_date(a.ex_date) <= day]
        # 先验证全部事件，避免异常造成半次结算。
        for a in due:
            if a.action_id not in self.entitlements:
                raise ValueError(f'缺少登记日持仓快照: {a.action_id}')
            q = self.entitlements[a.action_id] * a.bonus_share_ratio
            if not np.isclose(q, round(q), atol=1e-8, rtol=0):
                raise ValueError('零碎股需上游提供登记结算后的分配结果，本例不自动截断')
        result = CorporateSettlement()
        for a in due:
            quantity = self.entitlements[a.action_id]
            cash, shares = quantity * a.cash_div_per_share, round(quantity * a.bonus_share_ratio)
            for phase, when in [('EX', a.ex_date), ('PAY', a.cash_payment_date), ('LIST', a.share_listing_date)]:
                key = (a.action_id, phase)
                if when is None or as_date(when) > day or key in self.processed:
                    continue
                if phase == 'EX':
                    result.receivable_delta += cash
                    result.shares_delta[a.symbol] = result.shares_delta.get(a.symbol, 0) + shares
                elif phase == 'PAY':
                    result.cash_delta += cash
                    result.receivable_delta -= cash
                else:
                    result.listed_delta[a.symbol] = result.listed_delta.get(a.symbol, 0) + shares
                self.processed.add(key)
        self.last_processed_date = day
        return result


if __name__ == '__main__':
    matcher = AshareExecutionMatchingSimulator()
    for low in (11., 10.5):
        bar = DailyMarketBar('A', 11, 11, low, 10.8, 11, 9, opening_buy_liquidity=10000)
        print('后续最低价', low, matcher.simulate_order_execution('A', 'BUY', 100, bar))
    action = CorporateAction('A-DIV', 'A', '2025-01-02', '2025-01-03',
                             '2025-01-06', '2025-01-07', 1., .2)
    book = CorporateActionBook([action])
    book.capture_record_date('2025-01-02', {'A': 100})
    for day in ['2025-01-03', '2025-01-06', '2025-01-06', '2025-01-07']:
        print(day, book.process_day(day))
