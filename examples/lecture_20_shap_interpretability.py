"""
教学用 A 股模型可解释性与归因引擎 (AshareTreeSHAPExplainer)
包含：TreeSHAP 全局特征贡献度、截面个股预测瀑布分解与因子交互检测
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
import lightgbm as lgb
import shap
from examples.ml_utils import demo_dates


class AshareTreeSHAPExplainer:
    """
    基于 TreeSHAP 的 A 股机器学习模型归因器
    """

    def __init__(self, booster: lgb.Booster, feature_names: List[str]):
        """
        初始化 SHAP 树解释器
        """
        self.booster = booster
        self.feature_names = feature_names
        # 预热 TreeExplainer
        self.explainer = shap.TreeExplainer(self.booster, model_output="raw", feature_perturbation="tree_path_dependent")
        self.expected_value = float(np.ravel(self.explainer.expected_value)[0])

    def explain_dataset(self, df: pl.DataFrame) -> Tuple[np.ndarray, pl.DataFrame]:
        """
        对整批数据集计算 SHAP 矩阵
        返回: (shap_values_matrix, global_importance_df)
        """
        X = df.select(self.feature_names).to_numpy()
        shap_values = self.explainer.shap_values(X)
        
        self.expected_value = float(np.ravel(self.explainer.expected_value)[0])
        # 本例仅支持单输出回归或排序模型
        if isinstance(shap_values, list):
            shap_mat = shap_values[0]
        else:
            shap_mat = shap_values

        if np.asarray(shap_mat).shape != X.shape:
            raise ValueError("本示例只支持单输出回归或排序模型")

        # 计算全局平均绝对 SHAP 值
        mean_abs_shap = np.mean(np.abs(shap_mat), axis=0)
        
        global_imp_df = pl.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort("mean_abs_shap", descending=True)

        return shap_mat, global_imp_df

    def explain_single_stock(
        self,
        df: pl.DataFrame,
        symbol: str,
        target_date: str,
        date_col: str = "trade_date",
        symbol_col: str = "symbol"
    ) -> Dict[str, float]:
        """
        对特定日期、特定单只股票进行个股预测归因瀑布拆解
        """
        row_df = df.filter(
            (pl.col(date_col) == target_date) & 
            (pl.col(symbol_col) == symbol)
        )
        assert row_df.height == 1, f"未找到股票 {symbol} 在 {target_date} 的唯一记录！"

        X_single = row_df.select(self.feature_names).to_numpy()
        single_shap = self.explainer.shap_values(X_single)
        if isinstance(single_shap, list):
            single_shap = single_shap[0]
        
        shap_dict = {
            self.feature_names[i]: float(single_shap[0, i])
            for i in range(len(self.feature_names))
        }

        # 按贡献绝对值排序
        sorted_shap = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return sorted_shap

    def detect_interactions(
        self,
        df: pl.DataFrame,
        sample_size: int = 500
    ) -> pl.DataFrame:
        """
        抽取部分截面样本计算双因子交互作用矩阵 (SHAP Interaction Values)
        """
        sampled_df = df.sample(min(df.height, sample_size), seed=42)
        X_sample = sampled_df.select(self.feature_names).to_numpy()
        
        # 计算交互矩阵 (N, M, M)
        interaction_values = self.explainer.shap_interaction_values(X_sample)
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[0]

        # 统计非对角线上的平均绝对交互强度
        M = len(self.feature_names)
        records = []
        for i in range(M):
            for j in range(i + 1, M):
                inter_strength = float(np.mean(np.abs(interaction_values[:, i, j])))
                records.append({
                    "factor_A": self.feature_names[i],
                    "factor_B": self.feature_names[j],
                    "interaction_strength": inter_strength
                })

        return pl.DataFrame(records).sort("interaction_strength", descending=True)


# ================= 单元测试与实战归因演示 =================
if __name__ == "__main__":
    np.random.seed(42)
    n_stocks = 200
    n_days = 50

    dates = demo_dates(n_days)
    symbols = [f"{s:06d}.SH" for s in range(n_stocks)]
    factor_names = ["mom_20d", "ep_ratio", "bp_ratio", "amihud_illiq", "turnover_vol"]

    # 构造带有明确业务逻辑的特征数据
    records = []
    for d in dates:
        f_mat = np.random.normal(0, 1.0, (n_stocks, len(factor_names)))
        
        # 预设非线性逻辑：
        # 1. mom_20d 是主力正向因子 (+0.008)
        # 2. ep_ratio 仅在 mom_20d > 0 时强力爆发 (交互项)
        # 3. turnover_vol 是负向惩罚因子 (-0.005)
        y = (
            0.008 * f_mat[:, 0] +
            0.010 * f_mat[:, 1] * (f_mat[:, 0] > 0.2).astype(float) -
            0.005 * f_mat[:, 4] +
            np.random.normal(0, 0.015, n_stocks)
        )

        for s_idx, s in enumerate(symbols):
            row = {"trade_date": d, "symbol": s, "ret_fwd": y[s_idx]}
            for f_idx, f_name in enumerate(factor_names):
                row[f_name] = f_mat[s_idx, f_idx]
            records.append(row)

    sample_df = pl.DataFrame(records)

    # 快速训练一个轻量 LightGBM 模型
    X_train = sample_df.select(factor_names).to_numpy()
    y_train = sample_df["ret_fwd"].to_numpy()
    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=factor_names)
    
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "num_threads": 2,
        "verbose": -1
    }
    booster = lgb.train(params, train_ds, num_boost_round=100)

    # 1. 初始化 TreeSHAP 解释器
    explainer = AshareTreeSHAPExplainer(booster, factor_names)
    print(f"解释器参考基线 E[f(X)]: {explainer.expected_value:8.5f}")

    # 2. 全局特征贡献度
    _, global_imp = explainer.explain_dataset(sample_df)
    print("\n>>> [全局归因] 全市场平均绝对 SHAP 贡献值排名:")
    print(global_imp)

    # 3. 单只个股选股逻辑瀑布拆解
    target_sym = "000001.SH"
    target_dt = dates[4]
    stock_attr = explainer.explain_single_stock(sample_df, symbol=target_sym, target_date=target_dt)
    
    print(f"\n>>> [局部归因] 股票 {target_sym} 在 {target_dt} 的预测打分驱动拆解:")
    sum_contributions = 0.0
    for f_name, shap_val in stock_attr.items():
        direction = "🟢 增益助力" if shap_val > 0 else "🔴 拖累抑制"
        print(f"  • {f_name:14s} -> SHAP 贡献: {shap_val:+8.5f} | {direction}")
        sum_contributions += shap_val

    predicted_val = explainer.expected_value + sum_contributions
    row = sample_df.filter((pl.col("trade_date") == target_dt) & (pl.col("symbol") == target_sym))
    raw_prediction = booster.predict(row.select(factor_names).to_numpy(), raw_score=True)[0]
    np.testing.assert_allclose(predicted_val, raw_prediction, atol=1e-8)
    print(f"  ──────────────────────────────────────────")
    print(f"  • 基准均值 (Base)  : {explainer.expected_value:+8.5f}")
    print(f"  • 特征合力 (Shap)  : {sum_contributions:+8.5f}")
    print(f"  • 最终预测 (Score) : {predicted_val:+8.5f} (加和效率性严格闭合！)")

    # 4. 双因子交互检测
    print("\n>>> [交互归因] 双因子非线性耦合强度 Top 3:")
    inter_df = explainer.detect_interactions(sample_df, sample_size=300)
    print(inter_df.head(3))
