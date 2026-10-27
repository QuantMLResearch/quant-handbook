"""
A 股教学示例：因子正交化处理器 (AshareFactorOrthogonalizer)
包含：对称正交化 (Löwdin) 与 施密特顺序正交化 (Gram-Schmidt)
"""

from typing import List, Literal, Tuple, Dict
from datetime import date
import numpy as np
import polars as pl


class AshareFactorOrthogonalizer:
    """
    A 股截面因子正交化引擎
    """

    def __init__(
        self,
        method: Literal["symmetric", "gram_schmidt"] = "symmetric",
        regularization: float = 1e-6
    ):
        """
        参数:
            method: 正交化算法，可选 'symmetric' (对称正交化/Löwdin) 或 'gram_schmidt' (施密特顺序正交化)
            regularization: 最小相对特征值容忍度；病态/共线输入明确拒绝
        """
        self.method = method
        if not 0 < regularization < 1:
            raise ValueError("相对特征值容忍度必须在 0 和 1 之间")
        self.regularization = regularization

    def _symmetric_orthogonalize(self, F: np.ndarray) -> np.ndarray:
        """
        Löwdin 对称正交化实现
        输入 F: (N, K) 截面矩阵 (已均值0, 方差1)
        输出 F_star: (N, K) 正交矩阵
        """
        N, K = F.shape
        # 计算样本相关性矩阵 S = (F^T @ F) / (N - 1)
        # 为保持与公式完全一致，直接用 S = F^T @ F
        S = F.T @ F

        # 特征值与特征向量分解 S = V @ diag(eigenvals) @ V^T
        # eigh 专用于对称/厄米特矩阵，数值稳定性远高于常规 eig
        eigenvals, V = np.linalg.eigh(S)

        # 拒绝病态矩阵；不截断特征值伪造满秩
        if eigenvals[0] <= self.regularization * eigenvals[-1]:
            raise ValueError("因子矩阵秩不足或近似共线，请删除冗余因子或先降维")

        # 计算 S^{-1/2} = V @ diag(1 / sqrt(lambda)) @ V^T
        diag_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvals))
        S_inv_sqrt = V @ diag_inv_sqrt @ V.T

        # F* = F @ S^{-1/2}
        F_star = F @ S_inv_sqrt

        # 重新对各列做尺度归一化 (均值 0, 方差 1)，便于后续与收益率配对打分
        # 注意：乘以常数倍不改变列之间的 0 相关性
        col_stds = np.std(F_star, axis=0, ddof=1)
        col_stds = np.where(col_stds < 1e-8, 1.0, col_stds)
        F_star = (F_star - np.mean(F_star, axis=0)) / col_stds

        return F_star

    def _gram_schmidt_orthogonalize(self, F: np.ndarray) -> np.ndarray:
        """
        经典施密特正交化实现 (依照列的顺序逐步剔除投影残差)
        """
        N, K = F.shape
        Q = np.zeros_like(F)

        for j in range(K):
            v = F[:, j].copy()
            # 依次扣除在之前所有正交基上的投影
            for i in range(j):
                q_i = Q[:, i]
                proj = np.dot(v, q_i) / (np.dot(q_i, q_i) + 1e-12)
                v = v - proj * q_i

            norm = np.linalg.norm(v)
            if norm < 1e-8:
                Q[:, j] = v  # 几乎线性相关全零向量
            else:
                Q[:, j] = v / norm

        # 标准化为单位方差
        col_stds = np.std(Q, axis=0, ddof=1)
        col_stds = np.where(col_stds < 1e-8, 1.0, col_stds)
        Q = (Q - np.mean(Q, axis=0)) / col_stds
        return Q

    def fit_transform_matrix(
        self,
        F: np.ndarray,
        factor_names: List[str]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        核心矩阵正交化运算与诊断指标输出
        """
        F = np.asarray(F, dtype=float)
        if F.ndim != 2 or not np.isfinite(F).all():
            raise ValueError("因子矩阵必须为有限的二维数组")
        N, K = F.shape
        if K < 1 or N <= K or len(factor_names) != K or len(set(factor_names)) != K:
            raise ValueError("需要 N > K >= 1，且因子名称唯一、数量等于列数")
        F = F - F.mean(axis=0)
        std = F.std(axis=0, ddof=1)
        if np.any(std <= 1e-12):
            raise ValueError("常数列导致矩阵秩不足")
        F = F / std
        eigenvalues = np.linalg.eigvalsh(F.T @ F)
        if eigenvalues[0] <= self.regularization * eigenvalues[-1]:
            raise ValueError("因子矩阵秩不足或近似共线，请删除冗余因子或先降维")

        # 1. 原始矩阵健康度诊断
        orig_corr = np.atleast_2d(np.corrcoef(F, rowvar=False))
        orig_cond = np.linalg.cond(orig_corr)

        # 2. 正交化计算
        if self.method == "symmetric":
            F_ortho = self._symmetric_orthogonalize(F)
        elif self.method == "gram_schmidt":
            F_ortho = self._gram_schmidt_orthogonalize(F)
        else:
            raise ValueError(f"未知的正交化方法: {self.method}")

        # 3. 正交后矩阵验证
        ortho_corr = np.atleast_2d(np.corrcoef(F_ortho, rowvar=False))
        ortho_cond = np.linalg.cond(ortho_corr)

        # 4. 计算每个新因子与原对应母因子的皮尔逊保留度 (Retention Correlation)
        retentions = {}
        for idx, name in enumerate(factor_names):
            ret_corr = np.corrcoef(F[:, idx], F_ortho[:, idx])[0, 1]
            retentions[name] = float(ret_corr)

        diagnostics = {
            "original_condition_number": float(orig_cond),
            "orthogonal_condition_number": float(ortho_cond),
            "retention_correlations": retentions,
            "max_abs_off_diagonal_corr": float(np.max(np.abs(ortho_corr - np.eye(K))))
        }

        return F_ortho, diagnostics

    def transform_polars(
        self,
        df: pl.DataFrame,
        factor_cols: List[str],
        date_col: str = "trade_date"
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        按日期截面遍历并执行正交化
        返回: (带有正交因子列的新 DataFrame, 截面诊断指标 DataFrame)
        """
        ortho_cols = [f"{col}_ortho" for col in factor_cols]
        diag_records = []
        result_dfs = []

        # 按截面日期分组正交化
        dates = df[date_col].unique().sort().to_list()

        for cur_date in dates:
            sub_df = df.filter(pl.col(date_col) == cur_date)
            # 提取因子矩阵
            f_mat = sub_df.select(factor_cols).to_numpy()

            # 执行正交化
            f_ortho, diag = self.fit_transform_matrix(f_mat, factor_cols)

            # 诊断日志
            diag_record = {
                date_col: cur_date,
                "orig_cond": diag["original_condition_number"],
                "ortho_cond": diag["orthogonal_condition_number"],
                "max_residual_corr": diag["max_abs_off_diagonal_corr"]
            }
            # 记录各因子保留度
            for col in factor_cols:
                diag_record[f"retention_{col}"] = diag["retention_correlations"][col]
            diag_records.append(diag_record)

            # 拼接回 Polars DataFrame
            ortho_pl_dict = {
                ortho_cols[i]: f_ortho[:, i] for i in range(len(factor_cols))
            }
            sub_ortho_df = sub_df.with_columns([
                pl.Series(k, v) for k, v in ortho_pl_dict.items()
            ])
            result_dfs.append(sub_ortho_df)

        final_df = pl.concat(result_dfs)
        diag_df = pl.DataFrame(diag_records)
        return final_df, diag_df


# ================= 合成数据演示（统计结果不是实盘收益证明） =================
if __name__ == "__main__":
    np.random.seed(42)
    n_stocks = 800  # 类似中证 800 股票池截面

    # 1. 模拟一组具有相关性的合成因子
    # 假设底层存在一个共同的隐变量 (例如未被中性化干净的中小市值成长动量风格)
    latent_growth = np.random.normal(0, 1, n_stocks)
    latent_value = np.random.normal(0, 1, n_stocks)

    # 构造三个强共线性的动量类因子
    mom_20d = latent_growth * 0.8 + np.random.normal(0, 0.4, n_stocks)
    mom_60d = latent_growth * 0.75 + np.random.normal(0, 0.45, n_stocks)
    macd_power = latent_growth * 0.7 + np.random.normal(0, 0.5, n_stocks)

    # 构造两个强共线性的价值因子
    ep_ratio = latent_value * 0.85 + np.random.normal(0, 0.35, n_stocks)
    bp_ratio = latent_value * 0.80 + np.random.normal(0, 0.40, n_stocks)

    raw_data = {
        "trade_date": [date(2026, 3, 31)] * n_stocks,
        "symbol": [f"{i:06d}.SH" for i in range(n_stocks)],
        "mom_20d": mom_20d,
        "mom_60d": mom_60d,
        "macd_power": macd_power,
        "ep_ratio": ep_ratio,
        "bp_ratio": bp_ratio,
    }
    sample_df = pl.DataFrame(raw_data)
    factor_list = ["mom_20d", "mom_60d", "macd_power", "ep_ratio", "bp_ratio"]

    print(">>> 原始因子截面相关系数矩阵:")
    orig_corr = np.corrcoef(sample_df.select(factor_list).to_numpy(), rowvar=False)
    for i, f1 in enumerate(factor_list):
        row_str = " ".join([f"{orig_corr[i, j]:6.3f}" for j in range(len(factor_list))])
        print(f"  {f1:12s} | {row_str}")

    # 2. 运行 Löwdin 对称正交化
    engine = AshareFactorOrthogonalizer(method="symmetric")
    ortho_df, diag_df = engine.transform_polars(sample_df, factor_list)

    print("\n>>> 对称正交化后的因子相关系数矩阵:")
    ortho_cols = [f"{f}_ortho" for f in factor_list]
    ortho_corr = np.corrcoef(ortho_df.select(ortho_cols).to_numpy(), rowvar=False)
    for i, f1 in enumerate(ortho_cols):
        row_str = " ".join([f"{ortho_corr[i, j]:6.3f}" for j in range(len(ortho_cols))])
        print(f"  {f1:18s} | {row_str}")

    print("\n>>> 正交化诊断指标报告:")
    print(diag_df)

    # 3. 对比施密特正交化
    gs_engine = AshareFactorOrthogonalizer(method="gram_schmidt")
    gs_ortho_df, gs_diag_df = gs_engine.transform_polars(sample_df, factor_list)
    print("\n>>> 施密特正交化信息保留度 (越排在后面丢失越严重):")
    for f in factor_list:
        sym_ret = diag_df[f"retention_{f}"][0]
        gs_ret = gs_diag_df[f"retention_{f}"][0]
        print(f"  因子 {f:12s} -> 对称正交化保留度: {sym_ret:6.3f} | 施密特保留度: {gs_ret:6.3f}")
