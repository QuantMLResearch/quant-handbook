"""日频账户参考实现：前日信号→保守开盘撮合→收盘盯市与登记日权益快照。"""
from dataclasses import dataclass
from enum import Enum
import numpy as np
import polars as pl
from examples.lecture_28_transaction_friction_and_market_impact import AshareFrictionEngine
from examples.lecture_29_matching_rules_limits_suspensions import (
    AshareExecutionMatchingSimulator, CorporateActionBook, DailyMarketBar, LotRule, as_date)


class OrderDirection(Enum):
    BUY = 'BUY'
    SELL = 'SELL'


class OrderStatus(Enum):
    FILLED = 'FILLED'
    PARTIAL = 'PARTIAL'
    REJECTED = 'REJECTED'


@dataclass
class Order:
    order_id: int
    trade_date: str
    symbol: str
    direction: OrderDirection
    requested_shares: int
    shares: int
    price: float
    total_friction: float
    status: OrderStatus
    reason: str


@dataclass
class Position:
    symbol: str
    total_shares: int = 0
    available_shares: int = 0
    unlisted_shares: int = 0
    last_price: float = 0.

    @property
    def market_value(self):
        return self.total_shares * self.last_price


class AshareEventDrivenBacktester:
    def __init__(self, initial_cash=10_000_000., *, trading_calendar,
                 friction_engine=None, lot_rules=None, corporate_actions=(), slippage_fixed=.01):
        if not np.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError('初始现金须有限且为正')
        calendar = [as_date(d) for d in trading_calendar]
        if not calendar or calendar != sorted(set(calendar)):
            raise ValueError('必须提供唯一、递增的经核验交易日历')
        self.calendar = calendar
        self.initial_cash = self.cash = float(initial_cash)
        self.dividend_receivable = 0.
        self.friction = friction_engine or AshareFrictionEngine()
        self.matcher = AshareExecutionMatchingSimulator(slippage_fixed)
        self.actions = CorporateActionBook(corporate_actions)
        self.lot_rules = lot_rules or {}
        self.positions, self.orders, self.daily_snapshots = {}, [], []
        self.current_date = None
        self.closed = False
        self.used_liquidity = {}
        self.daily_fees = 0.

    def _position(self, symbol):
        return self.positions.setdefault(symbol, Position(symbol))

    @property
    def total_equity(self):
        return self.cash + self.dividend_receivable + sum(p.market_value for p in self.positions.values())

    def _begin_day(self, day, bars):
        if day not in self.calendar or (self.current_date is not None and day < self.current_date):
            raise ValueError('非交易日或日期倒退')
        if self.current_date == day:
            if self.closed:
                raise ValueError('当日已收盘，不能再执行开盘订单')
            return
        if self.current_date is not None:
            if not self.closed or self.calendar.index(day) != self.calendar.index(self.current_date) + 1:
                raise ValueError('必须先逐日收盘记账，不能跳过交易日')
        required = {s for s, p in self.positions.items() if p.total_shares}
        required |= {a.symbol for a in self.actions.actions if as_date(a.ex_date) <= day and a.bonus_share_ratio
                     and self.actions.entitlements.get(a.action_id, 0) > 0
                     and (a.action_id, 'EX') not in self.actions.processed}
        if not required <= set(bars):
            raise ValueError('缺少持仓/送股证券的当日估值，停牌也须提供明确估值')
        settlement = self.actions.process_day(day)
        # 只在交易日推进时解冻，未上市送股始终锁定。
        for pos in self.positions.values():
            pos.available_shares = pos.total_shares - pos.unlisted_shares
        self.cash += settlement.cash_delta
        self.dividend_receivable += settlement.receivable_delta
        for symbol, qty in settlement.shares_delta.items():
            pos = self._position(symbol)
            pos.total_shares += qty
            pos.unlisted_shares += qty
        for symbol, qty in settlement.listed_delta.items():
            pos = self._position(symbol)
            pos.unlisted_shares -= qty
            pos.available_shares += qty
        self.current_date, self.closed = day, False
        self.used_liquidity, self.daily_fees = {}, 0.

    def execute_rebalance(self, target_weights, market_bars, trade_date, *, signal_date,
                          allow_new_buys=True):
        """开盘执行。目标来自更早日期；未知成交量为 0。买单缩量留足全部成本。

        日内再次调用不会解冻，也不会重复使用已消耗的开盘流动性。
        allow_new_buys 可由第 31 讲的数据质量/风险控制建议传入。
        """
        day = as_date(trade_date)
        if as_date(signal_date) >= day:
            raise ValueError('开盘信号必须在之前日期已知')
        if not all(np.isfinite(w) and w >= 0 for w in target_weights.values()) or sum(target_weights.values()) > 1 + 1e-10:
            raise ValueError('目标权重须非负有限且合计不超过 1')
        required = set(target_weights) | {s for s, p in self.positions.items() if p.total_shares}
        if not required <= set(market_bars):
            raise ValueError('缺少当日开盘行情；不能用旧价格伪造成交')
        if any(not isinstance(b, DailyMarketBar) or s != b.symbol for s, b in market_bars.items()):
            raise ValueError('需要证券代码匹配的 DailyMarketBar')
        self._begin_day(day, market_bars)
        for s, bar in market_bars.items():
            if s in self.positions:
                self.positions[s].last_price = bar.open
        equity = self.total_equity
        start = len(self.orders)
        # 股票缺省目标为零；资金不足只缩小买入，不卖出今日新买持仓。
        for symbol in sorted(set(target_weights) | set(self.positions)):
            pos = self._position(symbol)
            if not pos.total_shares:
                continue
            bar = market_bars[symbol]
            rule = self.lot_rules.get(symbol, LotRule())
            target = rule.floor_buy(equity * target_weights.get(symbol, 0) / bar.open)
            requested = max(0, pos.total_shares - target)
            if requested:
                available = min(requested, pos.available_shares)
                quantity = available if available == pos.total_shares else rule.floor_buy(available)
                self._trade(symbol, 'SELL', requested, quantity, bar, rule)
        for symbol, weight in sorted(target_weights.items()):
            bar = market_bars[symbol]
            pos = self._position(symbol)
            rule = self.lot_rules.get(symbol, LotRule())
            target = rule.floor_buy(equity * weight / bar.open)
            requested = rule.floor_buy(max(0, target - pos.total_shares))
            if requested:
                self._trade(symbol, 'BUY', requested, requested if allow_new_buys else 0, bar, rule,
                            blocked_reason='RISK_CONTROL' if not allow_new_buys else None)
        self._check_account()
        return self.orders[start:]

    def _trade(self, symbol, direction, requested, quantity, bar, rule, blocked_reason=None):
        if quantity == 0:
            ok, filled, price, reason = False, 0, 0., blocked_reason or 'T_PLUS_ONE_OR_LOT'
        else:
            ok, filled, price, reason = self.matcher.simulate_order_execution(
                symbol, direction, quantity, bar, lot_rule=rule,
                used_liquidity=self.used_liquidity.get((symbol, direction), 0))
        fee = 0.
        if ok and direction == 'BUY':
            # 二分搜索可负担的申报数量；成本单调包括保底、过户与非线性冲击。
            lo, hi, affordable = 0, (filled - rule.min_buy) // rule.step, 0
            while lo <= hi:
                mid = (lo + hi) // 2
                q = rule.min_buy + mid * rule.step
                cost = self.friction.calculate_trade_friction(symbol, direction, q * price, bar.adv_20d, bar.daily_volatility)
                if q * price + cost.total_friction <= self.cash:
                    affordable, lo = q, mid + 1
                else:
                    hi = mid - 1
            if affordable < filled:
                reason = 'CASH_SIZED' if affordable else 'INSUFFICIENT_CASH'
            filled = affordable
            ok = filled > 0
        if ok:
            fee = self.friction.calculate_trade_friction(symbol, direction, filled * price,
                                                         bar.adv_20d, bar.daily_volatility).total_friction
            pos = self._position(symbol)
            pos.last_price = bar.open
            if direction == 'BUY':
                self.cash -= filled * price + fee
                pos.total_shares += filled
            else:
                self.cash += filled * price - fee
                pos.total_shares -= filled
                pos.available_shares -= filled
            self.used_liquidity[(symbol, direction)] = self.used_liquidity.get((symbol, direction), 0) + filled
            self.daily_fees += fee
        status = OrderStatus.REJECTED if not ok else OrderStatus.FILLED if filled == requested else OrderStatus.PARTIAL
        self.orders.append(Order(len(self.orders) + 1, self.current_date.isoformat(), symbol,
            OrderDirection(direction), requested, filled, price, fee, status, reason))

    def _check_account(self):
        if not np.isfinite(self.total_equity) or self.cash < -1e-7 or self.dividend_receivable < -1e-7:
            raise RuntimeError('账户资金恒等式或现金非负约束被破坏')
        for p in self.positions.values():
            if not 0 <= p.available_shares <= p.total_shares - p.unlisted_shares or p.unlisted_shares < 0:
                raise RuntimeError('可卖/锁定股份与总持仓不一致')

    def close_day(self, trade_date, closing_prices):
        if self.current_date != as_date(trade_date) or self.closed:
            raise ValueError('需要先执行当天开盘阶段，且每个交易日只收盘一次')
        held = {s for s, p in self.positions.items() if p.total_shares}
        if not held <= set(closing_prices) or any(not np.isfinite(closing_prices[s]) or closing_prices[s] <= 0 for s in held):
            raise ValueError('每个持仓必须有有限、正数的当日收盘估值')
        for s in held:
            self.positions[s].last_price = closing_prices[s]
        self.actions.capture_record_date(self.current_date, {s: p.total_shares for s, p in self.positions.items()})
        self._check_account()
        previous = self.daily_snapshots[-1]['total_equity'] if self.daily_snapshots else self.initial_cash
        snapshot = dict(trade_date=self.current_date, total_equity=self.total_equity,
            cash=self.cash, dividend_receivable=self.dividend_receivable,
            market_value=sum(p.market_value for p in self.positions.values()),
            fees=self.daily_fees, net_return=self.total_equity / previous - 1,
            cumulative_return=self.total_equity / self.initial_cash - 1)
        self.daily_snapshots.append(snapshot)
        self.closed = True
        return snapshot

    def get_equity_curve_dataframe(self):
        return pl.DataFrame(self.daily_snapshots)


if __name__ == '__main__':
    calendar = ['2025-01-02', '2025-01-03']
    engine = AshareEventDrivenBacktester(initial_cash=100000, trading_calendar=calendar)
    for day, signal, price in [('2025-01-02', '2024-12-31', 10.), ('2025-01-03', '2025-01-02', 10.2)]:
        bar = DailyMarketBar('A', price, price, price, price, 11., 9.,
                             opening_buy_liquidity=100000, opening_sell_liquidity=100000)
        for order in engine.execute_rebalance({'A': 1.}, {'A': bar}, day, signal_date=signal):
            print(order)
        engine.close_day(day, {'A': price})
    print(engine.get_equity_curve_dataframe())
    print('账户包含第 28 讲成本和第 29 讲撮合；未模拟连续竞价队列或券商柜台。')
