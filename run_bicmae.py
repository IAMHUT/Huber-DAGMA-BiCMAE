"""
步骤2: 训练BiCMAE模型
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.bicmae import BiCMAEDynamicsModel, BiCMAETrainer
from src.utils import (
    simulate_controlled_dynamics,
    TransitionDataset,
    visualize_bicmae_losses,
    visualize_masking_mechanism,
    ensure_dir
)


def main():
    print("=" * 80)
    print("步骤2: 训练 BiCMAE 模型")
    print("=" * 80)

    # 确保数据目录存在
    ensure_dir('data/figures')
    ensure_dir('data/results')

    # 加载DAGMA结果
    print("\n[1] 加载 DAGMA 因果发现结果...")
    W_est = np.load('data/results/W_est.npy')
    ancestors = np.load('data/results/ancestors.npy')
    descendants = np.load('data/results/descendants.npy')
    print(f"✓ 加载因果图: {W_est.shape}")
    print(f"✓ 祖先矩阵: {ancestors.shape}")
    print(f"✓ 后代矩阵: {descendants.shape}")

    # 生成动力学数据
    print("\n[2] 生成受控动力学系统数据...")
    n_steps = 5000
    action_dim = 3
    seed = 42

    observations, actions, states = simulate_controlled_dynamics(
        W_est,
        n_steps=n_steps,
        action_dim=action_dim,
        seed=seed
    )
    print(f"✓ 生成 {n_steps} 步转移数据")
    print(f"✓ 观测维度: {observations.shape[1]}")
    print(f"✓ 动作维度: {action_dim}")

    # 准备数据集
    print("\n[3] 准备训练数据...")
    dataset = TransitionDataset(observations, actions)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"✓ 训练集大小: {train_size}")
    print(f"✓ 验证集大小: {val_size}")
    print(f"✓ Batch size: {batch_size}")

    # 初始化模型
    print("\n[4] 初始化 BiCMAE 模型...")
    device = 'cpu' if torch.cuda.is_available() else 'cuda'
    print(f"✓ 使用设备: {device}")

    # 初始化模型
    print("\n[4] 初始化 BiCMAE 模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ 使用设备: {device}")

    obs_dim = observations.shape[1]
    latent_dim = obs_dim  # 🔥 修改这里：使潜在维度等于观测维度

    model = BiCMAEDynamicsModel(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
        ancestors_matrix=ancestors,
        descendants_matrix=descendants,
        enc_hidden=[256, 128],
        fwd_hidden=[128, 128],
        device=device
    )

    print(f"✓ 观测维度: {obs_dim}")
    print(f"✓ 潜在维度: {latent_dim}")
    print(f"✓ 动作维度: {action_dim}")

    # 训练模型
    print("\n[5] 训练 BiCMAE...")
    trainer = BiCMAETrainer(model, device=device, lambda_fwd=1.0, lambda_bwd=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    n_epochs = 200
    loss_history = {
        'total_loss': [],
        'loss_fwd': [],
        'loss_bwd': []
    }

    print(f"✓ 训练 {n_epochs} 个 epoch...")

    for epoch in range(n_epochs):
        # 训练
        train_metrics = trainer.train_epoch(train_loader, optimizer)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                o_t, a_t, o_tp1 = batch
                o_t = o_t.to(device)
                a_t = a_t.to(device)
                o_tp1 = o_tp1.to(device)

                s_tp1_true = model.encode(o_tp1)
                s_tp1_fwd, s_tp1_bwd = model.predict_next_state(o_t, a_t)

                loss = torch.mean((s_tp1_fwd - s_tp1_true) ** 2 +
                                  (s_tp1_bwd - s_tp1_true) ** 2)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # 记录损失
        loss_history['total_loss'].append(train_metrics['total_loss'])
        loss_history['loss_fwd'].append(train_metrics['loss_fwd'])
        loss_history['loss_bwd'].append(train_metrics['loss_bwd'])

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} - "
                  f"Train Loss: {train_metrics['total_loss']:.6f}, "
                  f"Val Loss: {val_loss:.6f}")

    # 可视化训练过程
    print("\n[6] 生成可视化...")
    visualize_bicmae_losses(loss_history)
    visualize_masking_mechanism(ancestors, descendants, sample_idx=5)

    # 保存模型
    print("\n[7] 保存模型...")
    torch.save(model.state_dict(), 'data/results/bicmae_model.pth')
    np.save('data/results/loss_history.npy', loss_history)

    print("\n" + "=" * 80)
    print("✓ BiCMAE 训练完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
