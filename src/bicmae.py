"""
拓扑感知双向因果掩码自编码器 (Topology-Aware BiCMAE)
"""

import torch
import torch.nn as nn
import numpy as np


class CausalEncoder(nn.Module):
    """因果编码器 - 提取潜在状态 s（近似外生噪声 / 因果表示）"""

    def __init__(self, input_dim, hidden_dims=[256, 128], latent_dim=64):
        super(CausalEncoder, self).__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """x: (batch_size, obs_dim) -> s: (batch_size, latent_dim)"""
        return self.encoder(x)


class TopologyAwareBidirectionalForwardModel(nn.Module):
    """拓扑感知的双向因果掩码前向动力学模型"""

    def __init__(self, latent_dim, action_dim,
                 ancestors_matrix, descendants_matrix,
                 hidden_dims=[128, 128], device='cpu'):
        super(TopologyAwareBidirectionalForwardModel, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.device = device

        # 转换为 torch tensor
        if isinstance(ancestors_matrix, np.ndarray):
            anc = torch.tensor(ancestors_matrix.astype(float),
                               dtype=torch.float32, device=self.device)
        else:
            anc = ancestors_matrix.float().to(self.device)

        if isinstance(descendants_matrix, np.ndarray):
            des = torch.tensor(descendants_matrix.astype(float),
                               dtype=torch.float32, device=self.device)
        else:
            des = descendants_matrix.float().to(self.device)

        self.register_buffer('ancestors', anc)
        self.register_buffer('descendants', des)

        # Forward path
        layers_fwd = []
        prev_dim = latent_dim + action_dim
        for h in hidden_dims:
            layers_fwd += [
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.LayerNorm(h),
                nn.Dropout(0.1)
            ]
            prev_dim = h
        layers_fwd.append(nn.Linear(prev_dim, latent_dim))
        self.forward_path = nn.Sequential(*layers_fwd)

        # Backward path
        layers_bwd = []
        prev_dim = latent_dim + action_dim
        for h in hidden_dims:
            layers_bwd += [
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.LayerNorm(h),
                nn.Dropout(0.1)
            ]
            prev_dim = h
        layers_bwd.append(nn.Linear(prev_dim, latent_dim))
        self.backward_path = nn.Sequential(*layers_bwd)

    def build_forward_mask(self, batch_size):
        """Forward Mask: mask 祖先"""
        B = batch_size
        D = self.latent_dim
        targets = torch.randint(0, D, (B,), device=self.device)
        cols = self.ancestors[:, targets].t()
        mask = 1.0 - cols
        return mask

    def build_backward_mask(self, batch_size):
        """Backward Mask: mask 后代"""
        B = batch_size
        D = self.latent_dim
        targets = torch.randint(0, D, (B,), device=self.device)
        rows = self.descendants[targets, :]
        mask = 1.0 - rows
        return mask

    def forward(self, s_t, a_t):
        """双向因果掩码前向预测"""
        B = s_t.shape[0]

        mask_fwd = self.build_forward_mask(B)
        mask_bwd = self.build_backward_mask(B)

        s_fwd = s_t * mask_fwd
        s_bwd = s_t * mask_bwd

        inp_fwd = torch.cat([s_fwd, a_t], dim=-1)
        inp_bwd = torch.cat([s_bwd, a_t], dim=-1)

        s_next_fwd = self.forward_path(inp_fwd)
        s_next_bwd = self.backward_path(inp_bwd)

        return s_next_fwd, s_next_bwd


class BiCMAEDynamicsModel(nn.Module):
    """完整的 BiCMAE 模型"""

    def __init__(self, obs_dim, latent_dim, action_dim,
                 ancestors_matrix, descendants_matrix,
                 enc_hidden=[256, 128], fwd_hidden=[128, 128],
                 device='cpu'):
        super(BiCMAEDynamicsModel, self).__init__()
        self.encoder = CausalEncoder(obs_dim, enc_hidden, latent_dim)
        self.forward_model = TopologyAwareBidirectionalForwardModel(
            latent_dim, action_dim,
            ancestors_matrix, descendants_matrix,
            fwd_hidden, device
        )

    def encode(self, o):
        return self.encoder(o)

    def predict_next_state(self, o_t, a_t):
        """从观测和动作预测下一状态"""
        s_t = self.encoder(o_t)
        s_next_fwd, s_next_bwd = self.forward_model(s_t, a_t)
        return s_next_fwd, s_next_bwd


class BiCMAETrainer:
    """BiCMAE 训练器"""

    def __init__(self, model, device='cuda', lambda_fwd=1.0, lambda_bwd=1.0):
        self.model = model.to(device)
        self.device = device
        self.lambda_fwd = lambda_fwd
        self.lambda_bwd = lambda_bwd

    def train_epoch(self, dataloader, optimizer):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_loss_fwd = 0.0
        total_loss_bwd = 0.0
        n_batches = 0

        for batch in dataloader:
            o_t, a_t, o_tp1 = batch
            o_t = o_t.to(self.device)
            a_t = a_t.to(self.device)
            o_tp1 = o_tp1.to(self.device)

            optimizer.zero_grad()

            s_tp1_true = self.model.encode(o_tp1)
            s_tp1_fwd, s_tp1_bwd = self.model.predict_next_state(o_t, a_t)

            loss_fwd = torch.mean((s_tp1_fwd - s_tp1_true) ** 2)
            loss_bwd = torch.mean((s_tp1_bwd - s_tp1_true) ** 2)

            loss = self.lambda_fwd * loss_fwd + self.lambda_bwd * loss_bwd

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_fwd += loss_fwd.item()
            total_loss_bwd += loss_bwd.item()
            n_batches += 1

        return {
            'total_loss': total_loss / n_batches,
            'loss_fwd': total_loss_fwd / n_batches,
            'loss_bwd': total_loss_bwd / n_batches
        }
