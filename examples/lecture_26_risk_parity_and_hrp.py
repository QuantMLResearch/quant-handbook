"""
教学用层级风险平价配置引擎 (AshareHierarchicalRiskParity)
采用相关距离直接 single-linkage 的 HRP 变体；不保证 ERC
包含：距离矩阵聚类、拟对角化重排、递归二分风险反比分配与 TRC 风险贡献度量
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform


class AshareHierarchicalRiskParity:
    """
    层级风险平价 (HRP) 资产配置优化器
    """

    def __init__(self, annual_factor: float = 250.0):
        if annual_factor <= 0:
            raise ValueError("annual_factor 必须为正")
        self.annual_factor = annual_factor

    def _correl_distance(self, corr: np.ndarray) -> np.ndarray:
        """
        计算相关系数距离矩阵 d_ij = sqrt(0.5 * (1 - rho_ij))
        """
        dist = np.sqrt(0.5 * np.maximum(0.0, 1.0 - corr))
        np.fill_diagonal(dist, 0.0)
        return dist

    def _get_quasi_diag_order(self, link_matrix: np.ndarray) -> List[int]:
        """
        从层次聚类树链接矩阵中提取叶子节点的拟对角化顺序
        """
        root = to_tree(link_matrix, rd=False)
        order = []

        def _traverse(node):
            if node.is_leaf():
                order.append(node.id)
            else:
                _traverse(node.left)
                _traverse(node.right)

        _traverse(root)
        return order

    def _get_cluster_var(self, cov: np.ndarray, cluster_items: List[int]) -> float:
        """
        计算某个子簇的反方差加权综合风险方差
        """
        sub_cov = cov[np.ix_(cluster_items, cluster_items)]
        # 反方差加权
        diag_inv = 1.0 / np.maximum(np.diag(sub_cov), 1e-8)
        w = diag_inv / np.sum(diag_inv)
        cluster_var = float(w.T @ sub_cov @ w)
        return cluster_var

    def _recursive_bisection(
        self,
        cov: np.ndarray,
        ordered_indices: List[int]
    ) -> np.ndarray:
        """
        自顶向下递归二分分配权重
        """
        weights = np.ones(len(ordered_indices))
        clusters = [ordered_indices]

        while len(clusters) > 0:
            next_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    # 将当前簇二分为前半段与后半段
                    mid = len(cluster) // 2
                    c1 = cluster[:mid]
                    c2 = cluster[mid:]

                    # 计算两子簇综合方差
                    v1 = self._get_cluster_var(cov, c1)
                    v2 = self._get_cluster_var(cov, c2)

                    # 计算反方差权重分配系数 alpha
                    alpha1 = v2 / (v1 + v2 + 1e-12)
                    alpha2 = 1.0 - alpha1

                    # 缩放子簇对应原始位置的权重
                    weights[c1] *= alpha1
                    weights[c2] *= alpha2

                    next_clusters.append(c1)
                    next_clusters.append(c2)
            clusters = next_clusters

        return weights

    def compute_hrp_weights(
        self,
        cov_matrix: np.ndarray,
        asset_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        计算全套 HRP 配置权重
        """
        cov_matrix = np.asarray(cov_matrix, dtype=float)
        if cov_matrix.ndim != 2 or not len(cov_matrix) or cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError("协方差须为非空方阵")
        if not np.isfinite(cov_matrix).all() or not np.allclose(cov_matrix, cov_matrix.T) or np.linalg.eigvalsh(cov_matrix).min() < -1e-10 or np.any(np.diag(cov_matrix) <= 0):
            raise ValueError("协方差须对称半正定，各资产方差为正")
        N = cov_matrix.shape[0]
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(N)]

        if len(asset_names) != N or len(set(asset_names)) != N:
            raise ValueError("资产名须唯一且与矩阵维度一致")
        if N == 1:
            return {asset_names[0]: 1.}

        # 1. 计算相关系数矩阵与相关距离
        vols = np.sqrt(np.diag(cov_matrix))
        outer_vols = np.outer(vols, vols)
        corr = cov_matrix / np.maximum(outer_vols, 1e-8)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        dist_mat = self._correl_distance(corr)

        # 2. 层次聚类 (Single Linkage)
        condensed_dist = squareform(dist_mat, checks=False)
        link = linkage(condensed_dist, method="single")

        # 3. 拟对角化叶子节点排序
        ordered_ids = self._get_quasi_diag_order(link)

        # 4. 递归二分风险反比赋权
        raw_weights = self._recursive_bisection(cov_matrix, ordered_ids)

        # 映射回原始资产顺序并归一化
        final_weights = raw_weights / np.sum(raw_weights)
        
        weight_dict = {
            asset_names[i]: float(final_weights[i])
            for i in range(N)
        }
        return weight_dict

    def evaluate_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        计算边际风险贡献 (MRC) 与总风险贡献 (TRC)
        输入日频协方差；返回全部年化的 (MRC向量, TRC向量, 组合波动率)
        """
        port_var = float(weights.T @ cov_matrix @ weights)
        if port_var <= 0:
            raise ValueError("零或负组合方差时波动率风险贡献未定义")
        port_vol = np.sqrt(port_var)

        # MRC = (Σ w) / sigma_p
        mrc = (cov_matrix @ weights) / port_vol
        # TRC = w * MRC
        trc = weights * mrc

        ann_port_vol = port_vol * np.sqrt(self.annual_factor)
        return mrc * np.sqrt(self.annual_factor), trc * np.sqrt(self.annual_factor), ann_port_vol


# ================= 单元测试与全天候资产配置演示 =================
if __name__ == "__main__":
    np.random.seed(42)

    # 模拟包含 6 大经典资产类别的全天候多资产池：
    # 1. 股票资产 (A股大盘 300, 中证 1000) - 年化波动 ~22%
    # 2. 债券资产 (十年期国债 ETF, 信用债) - 年化波动 ~4%
    # 3. 大宗商品 (黄金 ETF, 南华商品指数) - 年化波动 ~15%
    asset_classes = [
        "沪深300_ETF", "中证1000_ETF", 
        "十年国债_ETF", "企业信用债_ETF", 
        "黄金_ETF", "南华商品_ETF"
    ]
    N = len(asset_classes)

    # 设定逼近真实市场的年化波动率与相关性
    vols_annual = np.array([0.20, 0.26, 0.035, 0.045, 0.16, 0.18])
    corr_matrix = np.array([
        [ 1.00,  0.82, -0.15, -0.05,  0.08,  0.35],
        [ 0.82,  1.00, -0.18, -0.08,  0.05,  0.42],
        [-0.15, -0.18,  1.00,  0.65,  0.25, -0.12],
        [-0.05, -0.08,  0.65,  1.00,  0.18, -0.08],
        [ 0.08,  0.05,  0.25,  0.18,  1.00,  0.30],
        [ 0.35,  0.42, -0.12, -0.08,  0.30,  1.00]
    ])

    # 重构日频协方差矩阵
    outer_v = np.outer(vols_annual / np.sqrt(250.0), vols_annual / np.sqrt(250.0))
    daily_cov = corr_matrix * outer_v

    # 1. 运行 HRP 引擎
    hrp_engine = AshareHierarchicalRiskParity(annual_factor=250.0)
    hrp_weights_dict = hrp_engine.compute_hrp_weights(daily_cov, asset_classes)
    hrp_w = np.array(list(hrp_weights_dict.values()))

    # 2. 对比等权组合 (Equal Weight)
    ew_w = np.ones(N) / N

    ivp_w = 1 / np.diag(daily_cov)
    ivp_w /= ivp_w.sum()
    _, ivp_trc, ivp_vol = hrp_engine.evaluate_risk_contributions(ivp_w, daily_cov)

    # 3. 评估风险贡献
    _, ew_trc, ew_vol = hrp_engine.evaluate_risk_contributions(ew_w, daily_cov)
    _, hrp_trc, hrp_vol = hrp_engine.evaluate_risk_contributions(hrp_w, daily_cov)

    print("=" * 80)
    print(f"📊 全天候多资产 HRP 风险平价对比报告")
    print("=" * 80)
    print(f"{'资产名称':<16} | {'等权资金比例':<12} | {'等权风险占比':<12} | {'HRP资金权重':<12} | {'HRP风险占比':<12}")
    print("-" * 80)
    for i, name in enumerate(asset_classes):
        ew_r_share = ew_trc[i] / np.sum(ew_trc) * 100
        hrp_r_share = hrp_trc[i] / np.sum(hrp_trc) * 100
        print(f"{name:<16} | {ew_w[i]*100:10.2f}%   | {ew_r_share:10.2f}%   | {hrp_w[i]*100:10.2f}%   | {hrp_r_share:10.2f}%")
    print("-" * 80)
    print(f"IVP 权重: {np.round(ivp_w, 4)}；风险占比: {np.round(ivp_trc / ivp_vol, 4)}；年化波动: {ivp_vol:.2%}")
    np.testing.assert_allclose(hrp_trc.sum(), hrp_vol)
    print("HRP 不保证资产风险贡献相等；未提供收益序列，故不计算夏普比率。")
    print(f"【组合总年化波动率对比】")
    print(f"  • 传统等权组合 (EW) 年化波动  : {ew_vol * 100:6.2f}%")
    print(f"  • 层级风险平价 (HRP) 年化波动 : {hrp_vol * 100:6.2f}%")
    print("=" * 80)
