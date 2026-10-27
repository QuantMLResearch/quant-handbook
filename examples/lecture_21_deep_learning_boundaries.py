"""
A 股日频时序深度学习模型简析 (1D-CNN + GRU 混合架构)
演示：合成三维张量上的 CNN/GRU 前向与训练循环；不是模型优劣实验
"""

from typing import Tuple
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class AshareTemporalCNNGRU(nn.Module):
    """
    轻量级时序卷积 + 门控循环单元混合网络
    输入形状: (Batch_size, Sequence_length, N_features)
    输出形状: (Batch_size, 1) 预测收益
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 20,
        cnn_channels: int = 32,
        gru_hidden: int = 32,
        dropout: float = 0.30
    ):
        super().__init__()
        self.seq_len = seq_len

        # 1. 1D 卷积层：提取局部短周期量价微观形态 (如连续 3 天的 K 线组合)
        # Conv1d 期望输入形状: (Batch, Channels/Features, Seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=cnn_channels,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()

        # 2. 门控循环单元 (GRU)：捕获中周期跨期依赖与记忆遗忘
        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True
        )

        # 3. 正则化与全连接预测头
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (B, T, F) -> 转置为 (B, F, T) 供 Conv1d 使用
        x_trans = x.transpose(1, 2)
        conv_out = self.relu(self.bn1(self.conv1(x_trans)))

        # 转回 (B, T, Channels) 供 GRU 处理
        gru_in = conv_out.transpose(1, 2)
        gru_out, _ = self.gru(gru_in)

        # 提取序列最后一个时间步的隐状态
        last_hidden = gru_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)

        # 输出预测标量
        pred = self.head(last_hidden)
        return pred.squeeze(-1)


# ================= 合成张量训练演示 =================
if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)

    # 模拟输入参数
    batch_size = 64
    seq_len = 20     # 观察过去 20 个交易日的量价历史窗口
    n_features = 6   # 6 个基础量价指标 (开高低收量额)
    n_samples = 2000

    # 构造时序三维张量: (样本数, 20天窗口, 6特征)
    X_sim = np.random.normal(0, 1.0, (n_samples, seq_len, n_features)).astype(np.float32)
    
    # 模拟真实微弱信号 (最后 3 天的动量微弱主导)
    true_signal = (X_sim[:, -1, 0] - X_sim[:, -3, 0]) * 0.006
    y_sim = (true_signal + np.random.normal(0, 0.02, n_samples)).astype(np.float32)

    # 划分 80% 训练，20% 验证
    split_idx = int(n_samples * 0.8)
    train_ds = TensorDataset(torch.from_numpy(X_sim[:split_idx]), torch.from_numpy(y_sim[:split_idx]))
    val_ds = TensorDataset(torch.from_numpy(X_sim[split_idx:]), torch.from_numpy(y_sim[split_idx:]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AshareTemporalCNNGRU(n_features=n_features, seq_len=seq_len)
    criterion = nn.HuberLoss() # 使用鲁棒 Huber 损失防御异常跳动
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)

    print(">>> 开始训练 1D-CNN + GRU 神经网络 (观察训练集 vs 验证集误差演变)...")
    for epoch in range(1, 6):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                preds = model(bx)
                val_loss += criterion(preds, by).item() * len(by)
        val_loss /= len(val_ds)

        print(f"  Epoch {epoch:2d} | 训练集 Loss: {train_loss:.6f} | 验证集 Loss: {val_loss:.6f}")

    print("\n组件训练演示完成。本例没有真实股票滑窗、GBDT 对照或样本外收益验证，不能据此判断模型优劣。")
