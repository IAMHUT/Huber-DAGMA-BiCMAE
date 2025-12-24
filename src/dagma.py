"""
DAGMA: 通过M-矩阵和对数行列式无环性约束学习DAG
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


class DAGMA:
    """DAGMA: 通过M-矩阵和对数行列式无环性约束学习DAG"""

    def __init__(self, d, s=1.0, device='cpu'):
        self.d = d
        self.s = s
        self.device = device

    def h_logdet(self, W):
        """计算对数行列式无环性函数 h(W)"""
        W_squared = W * W
        M = self.s * torch.eye(self.d, device=self.device) - W_squared
        h = -torch.logdet(M) + self.d * np.log(self.s)
        return h

    def grad_h_logdet(self, W):
        """计算无环性函数 h(W) 的梯度 ∂h/∂W"""
        W_squared = W * W
        M = self.s * torch.eye(self.d, device=self.device) - W_squared
        M_inv_T = torch.inverse(M).T
        grad = 2 * M_inv_T * W
        return grad


class LinearDAGMA(nn.Module):
    """线性 SEM 的 DAGMA 实现 - 使用 Huber 损失"""

    def __init__(self, d, device='cpu'):
        super(LinearDAGMA, self).__init__()
        self.d = d
        self.device = device
        self.W = nn.Parameter(torch.zeros(d, d, device=device))

    def forward(self, X):
        """线性结构方程: X_pred = X @ W"""
        return X @ self.W

    def huber_loss(self, X, delta=1.0):
        """Huber 损失函数，对异常值更鲁棒"""
        n = X.shape[0]
        X_pred = self.forward(X)
        residual = X - X_pred
        abs_residual = torch.abs(residual)

        quadratic_part = 0.5 * residual ** 2
        linear_part = delta * (abs_residual - 0.5 * delta)

        loss_elementwise = torch.where(
            abs_residual <= delta,
            quadratic_part,
            linear_part
        )

        loss = torch.sum(loss_elementwise) / n
        return loss

    def l1_penalty(self):
        """L1 正则化，鼓励稀疏"""
        return torch.sum(torch.abs(self.W))


def dagma_linear_huber(
        X,
        lambda1=0.05,
        delta=1.0,
        s_values=[1.0, 0.9, 0.8, 0.7],
        mu_init=1.0,
        mu_factor=0.1,
        lr=3e-4,
        max_iter=20000,
        tol=1e-6,
        device='cpu'
):
    """DAGMA 主算法（线性 + Huber 损失）"""
    X = torch.tensor(X, dtype=torch.float32, device=device)
    n, d = X.shape

    model = LinearDAGMA(d, device=device)
    dagma = DAGMA(d, device=device)

    T = len(s_values)
    mu = mu_init

    print(f"\n使用Huber损失 (δ={delta})")

    for t in range(T):
        dagma.s = s_values[t]
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.99, 0.999))

        if t < T - 1:
            current_max_iter = max_iter
        else:
            current_max_iter = int(3.5 * max_iter)

        print(f"\n迭代 {t + 1}/{T}, μ={mu:.4f}, s={dagma.s}, δ={delta}")

        prev_loss = float('inf')
        for iter in range(current_max_iter):
            optimizer.zero_grad()

            loss_score = model.huber_loss(X, delta=delta)
            loss_l1 = model.l1_penalty()
            loss_h = dagma.h_logdet(model.W)

            total_loss = mu * (loss_score + lambda1 * loss_l1) + loss_h

            total_loss.backward()
            optimizer.step()

            if iter % 100 == 0:
                with torch.no_grad():
                    h_val = dagma.h_logdet(model.W).item()
                    print(f"  步 {iter}: 总损失={total_loss.item():.6f}, h(W)={h_val:.6f}")

            if abs(total_loss.item() - prev_loss) / (abs(prev_loss) + 1e-8) < tol:
                print(f"  在步 {iter} 收敛")
                break
            prev_loss = total_loss.item()

        mu = mu * mu_factor

    W_est = model.W.detach().cpu().numpy()
    W_est[np.abs(W_est) < 0.3] = 0

    return W_est
