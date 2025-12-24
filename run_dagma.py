"""
步骤1: 运行DAGMA因果发现算法
"""

import os
import numpy as np
import torch
from src.dagma import dagma_linear_huber
from src.utils import (
    generate_linear_sem,
    compute_metrics,
    visualize_causal_matrices,
    visualize_dag_structure,
    compute_ancestors_descendants,
    ensure_dir
)


def main():
    print("=" * 80)
    print("步骤1: DAGMA 因果发现")
    print("=" * 80)

    # 确保数据目录存在
    ensure_dir('data/figures')
    ensure_dir('data/results')

    # 生成数据
    print("\n[1] 生成线性 SEM 数据...")
    n_samples = 2000
    n_vars = 20
    seed = 42

    X, W_true, noise = generate_linear_sem(
        n=n_samples,
        d=n_vars,
        graph_type='ER',
        degree=1,
        seed=seed
    )
    print(f"✓ 生成 {n_samples} 个样本，{n_vars} 个变量")
    print(f"✓ 真实因果边数量: {np.sum(np.abs(W_true) > 0.3)}")

    # 运行 DAGMA
    print("\n[2] 使用 DAGMA 学习因果结构...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ 使用设备: {device}")

    W_est = dagma_linear_huber(
        X,
        lambda1=0.05,
        delta=1.0,
        s_values=[1.0, 0.9, 0.8, 0.7],
        mu_init=1.0,
        mu_factor=0.1,
        lr=3e-4,
        max_iter=10000,
        device=device
    )
    # 评估结果
    print("\n[3] 评估因果发现结果...")
    metrics = compute_metrics(W_true, W_est, threshold=0.3)
    print(f"✓ Precision: {metrics['precision']:.4f}")
    print(f"✓ Recall: {metrics['recall']:.4f}")
    print(f"✓ F1-Score: {metrics['f1']:.4f}")
    print(f"✓ SHD: {metrics['shd']}")
    print(f"✓ 估计边数: {metrics['n_est_edges']} (真实: {metrics['n_true_edges']})")

    # 可视化
    print("\n[4] 生成可视化...")
    visualize_causal_matrices(W_true, W_est)
    ancestors, descendants = visualize_dag_structure(W_est)

    # 保存结果
    print("\n[5] 保存结果...")
    np.save('data/results/W_true.npy', W_true)
    np.save('data/results/W_est.npy', W_est)
    np.save('data/results/ancestors.npy', ancestors)
    np.save('data/results/descendants.npy', descendants)
    np.save('data/results/X_data.npy', X)

    print("\n" + "=" * 80)
    print("✓ DAGMA 因果发现完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

