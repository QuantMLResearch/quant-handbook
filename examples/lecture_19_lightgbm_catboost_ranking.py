"""
教学用 A 股 LightGBM 截面排序器 (AshareLightGBMRanker)
支持：LambdaMART 排序学习与鲁棒 Huber 回归目标、日期 Group 自动聚合、标签区间检查与早停验证
"""

from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import polars as pl
import lightgbm as lgb
from examples.ml_utils import demo_dates, purged_holdout


class AshareLightGBMRanker:
    """
    A 股截面多因子 LightGBM 学习器
    """

    def __init__(
        self,
        objective: Literal["lambdarank", "regression"] = "lambdarank",
        learning_rate: float = 0.03,
        num_leaves: int = 15,
        max_depth: int = 4,
        min_child_samples: int = 100,
        feature_fraction: float = 0.70,
        bagging_fraction: float = 0.80,
        reg_alpha: float = 1.0,
        reg_lambda: float = 5.0,
        n_estimators: int = 500,
        random_state: int = 42
    ):
        self.objective = objective
        self.params = {
            "boosting_type": "gbdt",
            "objective": objective if objective != "regression" else "huber",
            "metric": "ndcg" if objective == "lambdarank" else "l2",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": 1,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": 2,
            "verbose": -1
        }
        if objective == "lambdarank":
            # 关注 Top 20 / Top 50 选股排序质量
            self.params["eval_at"] = [20, 50]

        self.n_estimators = n_estimators
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []

    def _prepare_lgb_dataset(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str,
        date_col: str = "trade_date"
    ) -> Tuple[lgb.Dataset, np.ndarray]:
        """
        构造 LightGBM 数据集并按日期提取 Group 切片大小
        """
        # 必须确保数据严格按日期排序以构造连续 Group
        sorted_df = df.sort(date_col)
        
        # 提取 Group 大小 (每日股票数)
        group_counts = (
            sorted_df.group_by(date_col, maintain_order=True)
            .agg(pl.len().alias("count"))["count"]
            .to_numpy()
        )

        X = sorted_df.select(feature_cols).to_numpy()

        if self.objective == "lambdarank":
            # LambdaMART 要求标签为非负整数相关度等级 (0, 1, 2, 3, 4)
            # 每天在截面上将未来收益等分映射为 5 个离散档位
            ranked_label = (
                (sorted_df.select(
                    ((pl.col(label_col).rank().over(date_col) - 1.0) /
                     pl.len().over(date_col) * 5.0).floor().cast(pl.Int32)
                )).to_series().clip(0, 4).to_numpy()
            )
            y = ranked_label
            dataset = lgb.Dataset(X, label=y, group=group_counts, feature_name=feature_cols)
        else:
            y = sorted_df[label_col].to_numpy()
            dataset = lgb.Dataset(X, label=y, feature_name=feature_cols)

        return dataset, y

    def fit(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str,
        date_col: str = "trade_date",
        early_stopping_rounds: int = 30,
        label_end_col: str = "label_end_date",
        available_at=None
    ):
        """
        执行带验证集 Early Stopping 的模型训练
        """
        if train_df.is_empty() or val_df.is_empty() or label_end_col not in train_df.columns or label_end_col not in val_df.columns:
            raise ValueError("训练/验证集不能为空，且必须提供 label_end_date")
        if available_at is None:
            raise ValueError("必须提供模型可用日期 available_at")
        for frame in (train_df, val_df):
            if any(frame[c].null_count() for c in (date_col, label_col, label_end_col)) or frame.filter(~pl.col(label_col).is_finite()).height:
                raise ValueError("训练标签必须有限且已提供结算日期")
            if frame.filter(pl.col(label_end_col) < pl.col(date_col)).height:
                raise ValueError("标签结算日期不能早于特征日期")
        if train_df[date_col].max() >= val_df[date_col].min() or train_df[label_end_col].max() >= val_df[date_col].min():
            raise ValueError("训练标签与验证区间重叠，请先执行 Purging")
        if val_df[label_end_col].max() >= available_at:
            raise ValueError("早停验证标签在模型可用日期之前尚未成熟")
        self.feature_names = feature_cols
        self.available_at = available_at

        train_ds, _ = self._prepare_lgb_dataset(train_df, feature_cols, label_col, date_col)
        val_ds, _ = self._prepare_lgb_dataset(val_df, feature_cols, label_col, date_col)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0) # 静默模式
        ]

        self.model = lgb.train(
            self.params,
            train_set=train_ds,
            num_boost_round=self.n_estimators,
            valid_sets=[val_ds],
            callbacks=callbacks
        )

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """
        使用训练好的模型预测截面打分
        """
        assert self.model is not None, "模型尚未训练，请先调用 fit()!"
        X = df.select(self.feature_names).to_numpy()
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def get_feature_importance(self) -> pl.DataFrame:
        """
        提取特征分裂收益 (Gain) 重要性
        """
        assert self.model is not None, "模型尚未训练!"
        gains = self.model.feature_importance(importance_type="gain")
        splits = self.model.feature_importance(importance_type="split")
        
        return pl.DataFrame({
            "feature": self.feature_names,
            "gain_importance": gains,
            "split_count": splits
        }).sort("gain_importance", descending=True)


# ================= 单元测试与排序模型实战 =================
if __name__ == "__main__":
    np.random.seed(42)
    n_days = 200
    n_stocks = 300

    calendar = demo_dates(n_days + 6, "2024-01-02")
    dates = calendar[:n_days]
    symbols = [f"{s:06d}.SH" for s in range(n_stocks)]
    features = [f"factor_{k}" for k in range(8)]

    records = []
    # 模拟多因子非线性市场：
    # factor_0 和 factor_1 存在强交互非线性效应，factor_2 具有阶梯截断
    for d_idx, d in enumerate(dates):
        f_mat = np.random.normal(0, 1.0, (n_stocks, len(features)))
        
        # 真实的收益生成方程 (包含非线性与交互)
        # y = 0.003 * f0 + 0.005 * f1 * (f0 > 0.5) + noise
        y = (
            0.004 * f_mat[:, 0] +
            0.006 * f_mat[:, 1] * (f_mat[:, 0] > 0.5).astype(float) -
            0.003 * (f_mat[:, 2] ** 2) +
            np.random.normal(0, 0.02, n_stocks)
        )

        for s_idx, s in enumerate(symbols):
            row = {
                "trade_date": d,
                "symbol": s,
                "ret_fwd_5d": y[s_idx],
                "label_end_date": calendar[d_idx + 5]
            }
            for f_idx, f_name in enumerate(features):
                row[f_name] = f_mat[s_idx, f_idx]
            records.append(row)

    full_df = pl.DataFrame(records)

    # 验证用于早停，不作为最终样本外业绩；按标签结束日清洗训练边界
    unique_dates = full_df["trade_date"].unique().sort().to_list()
    split_date = unique_dates[150]

    available_at = calendar[-1]
    train_data, val_data = purged_holdout(full_df, split_date, dates[-1], available_at)

    print(f"训练集规模: {train_data.height:,} 行 | 验证集规模: {val_data.height:,} 行")

    # 1. 训练 LambdaMART 排序学习器
    print("\n>>> 开始训练 LightGBM LambdaMART 截面排序器...")
    ranker = AshareLightGBMRanker(
        objective="lambdarank",
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=50
    )
    ranker.fit(train_data, val_data, feature_cols=features, label_col="ret_fwd_5d", available_at=available_at)

    # 2. 验证集推理与 Rank IC 评估
    val_preds = ranker.predict(val_data)
    val_with_pred = val_data.with_columns(pl.Series("pred_score", val_preds))

    # 计算每日验证集的 Rank IC
    ranked_val = val_with_pred.with_columns([
        pl.col("pred_score").rank().over("trade_date").alias("p_rank"),
        pl.col("ret_fwd_5d").rank().over("trade_date").alias("y_rank")
    ])
    daily_ic = ranked_val.group_by("trade_date").agg(pl.corr("p_rank", "y_rank").alias("rank_ic"))
    mean_val_ic = float(daily_ic["rank_ic"].mean())

    print(f"✅ 验证集平均 Rank IC: {mean_val_ic:8.4f} (树模型最佳迭代步数: {ranker.model.best_iteration})")

    # 3. 查看特征重要性
    print("\n>>> 模型特征分裂增益 (Gain) 排名:")
    print(ranker.get_feature_importance())
