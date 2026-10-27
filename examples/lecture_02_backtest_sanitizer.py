"""
lecture_02_backtest_sanitizer.py
合成日频案例：比较信号时点与撮合约束，演示现金持仓核算和 DSR
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass(frozen=True)
class TradeExecutionRules:
    can_buy_on_limit_up: bool = False   # 严格禁止涨停买入
    can_sell_on_limit_down: bool = False # 严格禁止跌停卖出
    one_way_commission: float = 0.00015  # 单边万1.5
    stamp_duty: float = 0.0005           # 卖方印花税 0.05%
    slippage_bps: float = 0.0005         # 基础滑点万分之5

class AshareBacktestSanitizer:
    def __init__(self, rules: TradeExecutionRules | None = None):
        self.rules = rules or TradeExecutionRules()

    def simulate_mock_ashare_universe(self, n_days: int = 500, n_stocks: int = 5) -> pd.DataFrame:
        """
        生成教学合成行情；不代表交易所真实日历、限价规则或退市过程。
        """
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        all_dfs = []

        for stock_id in [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]:
            # 基础随机游走收益率
            base_ret = np.random.normal(0.0003, 0.022, size=n_days)
            
            # 人工注入 3% 概率的一字涨停与 2% 概率的封死跌停
            limit_up_mask = np.random.binomial(1, 0.03, size=n_days).astype(bool)
            limit_down_mask = np.random.binomial(1, 0.02, size=n_days).astype(bool) & ~limit_up_mask
            
            base_ret[limit_up_mask] = 0.0998  # 封死涨停
            base_ret[limit_down_mask] = -0.0998 # 封死跌停

            close = 10.0 * np.cumprod(1 + base_ret)
            open_p = close * (1 + np.random.normal(0, 0.003, size=n_days))
            # 保证价格物理合理性
            high_p = np.maximum(open_p, close) * (1 + np.abs(np.random.normal(0, 0.005, size=n_days)))
            low_p = np.minimum(open_p, close) * (1 - np.abs(np.random.normal(0, 0.005, size=n_days)))

            locked = limit_up_mask | limit_down_mask
            open_p[locked] = high_p[locked] = low_p[locked] = close[locked]

            df = pd.DataFrame({
                "trade_date": dates,
                "symbol": stock_id,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close,
                "returns": base_ret,
                "is_limit_up": limit_up_mask,
                "is_limit_down": limit_down_mask
            })
            all_dfs.append(df)

        return pd.concat(all_dfs, ignore_index=True)

    def run_backtest_engine(
        self,
        df: pd.DataFrame,
        use_future_leakage: bool = False,
        enforce_liquidity_constraints: bool = True
    ) -> Dict:
        """
        回测主推进引擎：支持在合规模式与漏洞模式之间切换对比
        """
        if df.duplicated(["symbol", "trade_date"]).any():
            raise ValueError("股票日期主键重复")
        df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
        symbols = df["symbol"].unique()
        dates = df["trade_date"].unique()
        if len(dates) < 2 or len(df) != len(dates) * len(symbols):
            raise ValueError("示例要求至少两天且每个交易日覆盖相同股票全集")
        if not np.isfinite(df[["open", "close", "returns"]].to_numpy()).all() or (df[["open", "close"]] <= 0).any().any():
            raise ValueError("行情需为有效有限数值")
        df["signal"] = (df["returns"] > 0).astype(int)
        if not use_future_leakage:
            df["signal"] = df.groupby("symbol")["signal"].shift(1).fillna(0).astype(int)
        # 固定初始资金分仓，仓间不再平衡；未投资部分保留现金。
        # 每只股票每天只在开盘做一次交易，新买入持仓不会同日卖出。
        cash = {symbol: 1.0 / len(symbols) for symbol in symbols}
        shares = {symbol: 0.0 for symbol in symbols}
        previous_nav = 1.0
        portfolio_daily_returns = []
        buy_cost = self.rules.one_way_commission + self.rules.slippage_bps
        sell_cost = buy_cost + self.rules.stamp_duty
        if min(buy_cost, sell_cost) < 0 or sell_cost >= 1:
            raise ValueError("交易成本率不合法")
        for day in dates[1:]:
            nav = 0.0
            for _, row in df[df["trade_date"] == day].iterrows():
                symbol = row["symbol"]
                buy = row["signal"] == 1 and shares[symbol] == 0
                sell = row["signal"] == 0 and shares[symbol] > 0
                suspended = ("volume" in row and (pd.isna(row["volume"]) or row["volume"] <= 0))
                if enforce_liquidity_constraints:
                    buy = buy and not suspended and (not row["is_limit_up"] or self.rules.can_buy_on_limit_up)
                    sell = sell and not suspended and (not row["is_limit_down"] or self.rules.can_sell_on_limit_down)
                if buy:
                    shares[symbol] = cash[symbol] / (row["open"] * (1 + buy_cost))
                    cash[symbol] = 0.0
                elif sell:
                    cash[symbol] = shares[symbol] * row["open"] * (1 - sell_cost)
                    shares[symbol] = 0.0
                nav += cash[symbol] + shares[symbol] * row["close"]
            portfolio_daily_returns.append(nav / previous_nav - 1)
            previous_nav = nav
        returns_arr = np.asarray(portfolio_daily_returns)
        std_ret = np.std(returns_arr, ddof=1) if len(returns_arr) > 1 else 0.0
        annual_sharpe = np.sqrt(252) * np.mean(returns_arr) / std_ret if std_ret > 0 else 0.0
        return dict(sharpe_ratio=annual_sharpe, cumulative_return=previous_nav - 1,
                    mean_daily_return=np.mean(returns_arr), returns_series=returns_arr)

    @staticmethod
    def calculate_deflated_sharpe_ratio(
        returns: np.ndarray, num_trials: int = 1, benchmark_sharpe: float = 0.0,
        trial_sharpe_std: float | None = None,
    ) -> float:
        """DSR 参考公式。夏普、基准与试验间标准差均为单期口径。

        num_trials 为独立试验次数假设；多次试验必须提供完整搜索的夏普离散程度。
        结果不是“策略真实”的贝叶斯后验概率，不单独处理收益自相关。
        来源：https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
        """
        returns = np.asarray(returns, dtype=float)
        if returns.ndim != 1 or len(returns) < 3 or not np.isfinite(returns).all():
            raise ValueError("至少需要三个有限的单期收益观测")
        if not isinstance(num_trials, int) or num_trials < 1 or not np.isfinite(benchmark_sharpe):
            raise ValueError("试验次数和基准不合法")
        standard_deviation = np.std(returns, ddof=1)
        if standard_deviation <= 0:
            raise ValueError("常量收益没有可估计的夏普标准误")
        sr = np.mean(returns) / standard_deviation
        threshold = benchmark_sharpe
        if num_trials > 1:
            if trial_sharpe_std is None or not np.isfinite(trial_sharpe_std) or trial_sharpe_std < 0:
                raise ValueError("多次试验需要提供单期夏普的试验间标准差")
            gamma = 0.5772156649
            max_z = ((1 - gamma) * norm.ppf(1 - 1 / num_trials) +
                     gamma * norm.ppf(1 - 1 / (num_trials * np.e)))
            threshold += trial_sharpe_std * max_z
        variance_term = 1 - skew(returns, bias=False) * sr + (kurtosis(returns, fisher=False, bias=False) - 1) * sr**2 / 4
        if not np.isfinite(variance_term) or variance_term <= 0:
            raise ValueError("无法估计有效夏普标准误")
        return float(norm.cdf((sr - threshold) * np.sqrt(len(returns) - 1) / np.sqrt(variance_term)))

if __name__ == "__main__":
    sanitizer = AshareBacktestSanitizer()
    data = sanitizer.simulate_mock_ashare_universe(n_days=600, n_stocks=10)

    print("=" * 70)
    print("实验 1：包含未来函数 + 忽略涨跌停限制的【业余漏洞版本】")
    print("=" * 70)
    res_flawed = sanitizer.run_backtest_engine(
        data, use_future_leakage=True, enforce_liquidity_constraints=False
    )
    print(f"  虚假回测夏普比率 (Sharpe Ratio) : {res_flawed['sharpe_ratio']:.2f}")
    print(f"  虚假累计总收益 (Cumulative Ret): {res_flawed['cumulative_return']*100:.2f}%")

    print("\n" + "=" * 70)
    print("实验 2：买方防御型【严格时序 + 涨跌停封板撮合拦截版本】")
    print("=" * 70)
    res_clean = sanitizer.run_backtest_engine(
        data, use_future_leakage=False, enforce_liquidity_constraints=True
    )
    print(f"  约束版本夏普比率 (Clean Sharpe) : {res_clean['sharpe_ratio']:.2f}")
    print(f"  约束版本累计收益 (Clean Return)    : {res_clean['cumulative_return']*100:.2f}%")

    print("\n" + "=" * 70)
    print("实验 3：过拟合与数据窥探检验 (Deflated Sharpe Ratio)")
    print("=" * 70)
    # 假设该策略是在遍历测试了 500 次不同参数后挑选出来的最好结果
    dsr_val = AshareBacktestSanitizer.calculate_deflated_sharpe_ratio(
        res_clean["returns_series"], num_trials=500, trial_sharpe_std=0.02  # 仅为演示假设，不是实测搜索结果
    )
    print(f"  单次回测名义夏普值               : {res_clean['sharpe_ratio']:.2f}")
    print(f"  假设 500 次独立试验、单期夏普标准差 0.02 的 DSR: {dsr_val:.4f} (仅示范公式，不能据此判定策略真实概率)")
