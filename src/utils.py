"""
工具函数：数据生成、评估、可视化
"""

import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_linear_sem(n=2000, d=20, graph_type='ER', degree=2, seed=None):
    """生成线性 SEM 数据"""
    if seed is not None:
        np.random.seed(seed)

    if graph_type == 'ER':
        prob = degree / (d - 1)
        B = np.random.binomial(1, prob, size=(d, d))
        perm = np.random.permutation(d)
        B = B[np.ix_(perm, perm)]
        B = np.tril(B, k=-1)
        B = B[np.argsort(perm), :][:, np.argsort(perm)]

    W_true = np.zeros((d, d))
    num_edges = np.sum(B != 0)

    weights = np.random.uniform(0.5, 2.0, size=num_edges)
    signs = np.random.choice([-1, 1], size=num_edges)
    W_true[B != 0] = weights * signs

    exogenous_noise = np.random.randn(n, d)
    X = np.linalg.solve(np.eye(d) - W_true.T, exogenous_noise.T).T

    return X, W_true, exogenous_noise


def simulate_controlled_dynamics(W_true, n_steps=1000, action_dim=3, seed=None):
    """模拟受控动力学系统"""
    if seed is not None:
        np.random.seed(seed)

    d = W_true.shape[0]
    states = np.zeros((n_steps + 1, d))
    states[0] = np.random.randn(d) * 0.1

    actions = np.random.randn(n_steps, action_dim) * 0.5
    B = np.random.randn(d, action_dim) * 0.3

    for t in range(n_steps):
        noise = np.random.randn(d) * 0.1
        states[t + 1] = W_true @ states[t] + B @ actions[t] + noise

    observations = states + np.random.randn(n_steps + 1, d) * 0.05

    return observations, actions, states


def compute_metrics(W_true, W_est, threshold=0.3):
    """计算 DAG 评估指标"""
    B_true = (np.abs(W_true) > threshold).astype(int).flatten()
    B_est = (np.abs(W_est) > threshold).astype(int).flatten()

    precision = precision_score(B_true, B_est, zero_division=0)
    recall = recall_score(B_true, B_est, zero_division=0)
    f1 = f1_score(B_true, B_est, zero_division=0)

    shd = np.sum(np.abs(B_true.reshape(W_true.shape) - B_est.reshape(W_est.shape)))

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'shd': shd,
        'n_true_edges': np.sum(B_true),
        'n_est_edges': np.sum(B_est)
    }


def compute_ancestors_descendants(W, threshold=0.3):
    """计算祖先矩阵和后代矩阵"""
    d = W.shape[0]
    adj = (np.abs(W) > threshold).astype(float)

    reachability = adj.copy()
    for k in range(d):
        for i in range(d):
            for j in range(d):
                if reachability[i, k] and reachability[k, j]:
                    reachability[i, j] = 1

    ancestors = reachability.T
    descendants = reachability

    return ancestors, descendants


class TransitionDataset(torch.utils.data.Dataset):
    """状态转移数据集"""

    def __init__(self, observations, actions):
        assert len(observations) == len(actions) + 1
        self.o_t = observations[:-1]
        self.a_t = actions
        self.o_tp1 = observations[1:]

    def __len__(self):
        return len(self.o_t)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.o_t[idx]),
            torch.FloatTensor(self.a_t[idx]),
            torch.FloatTensor(self.o_tp1[idx])
        )


def visualize_causal_matrices(W_true, W_est, save_dir='data/figures'):
    """可视化因果矩阵"""
    ensure_dir(save_dir)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    vmax = max(np.abs(W_true).max(), np.abs(W_est).max())
    vmin = -vmax

    sns.heatmap(W_true, ax=axes[0], cmap='RdBu_r', center=0,
                vmin=vmin, vmax=vmax, square=True,
                cbar_kws={'label': 'weight'})
    axes[0].set_title('True Causality Matrix', fontsize=14, fontweight='bold')

    sns.heatmap(W_est, ax=axes[1], cmap='RdBu_r', center=0,
                vmin=vmin, vmax=vmax, square=True,
                cbar_kws={'label': 'weight'})
    axes[1].set_title('Estimated Causality Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'causal_matrices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {save_path}")


def visualize_dag_structure(W, threshold=0.3, save_dir='data/figures'):
    """可视化DAG结构"""
    ensure_dir(save_dir)

    d = W.shape[0]
    G = nx.DiGraph()

    for i in range(d):
        G.add_node(i)

    for i in range(d):
        for j in range(d):
            if np.abs(W[i, j]) > threshold:
                G.add_edge(i, j, weight=W[i, j])

    try:
        pos = nx.spring_layout(G, k=2, iterations=50)
    except:
        pos = nx.circular_layout(G)

    fig, axes = plt.subplots(1, 2, figsize=(12, 10))

    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                           node_size=800, ax=axes[0])
    nx.draw_networkx_labels(G, pos, font_size=10,
                            font_weight='bold', ax=axes[0])

    edges = G.edges()
    pos_edges = [(u, v) for u, v in edges if G[u][v]['weight'] > 0]
    neg_edges = [(u, v) for u, v in edges if G[u][v]['weight'] < 0]

    nx.draw_networkx_edges(G, pos, pos_edges, edge_color='blue',
                           arrows=True, arrowsize=20, width=2,
                           ax=axes[0], connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edges(G, pos, neg_edges, edge_color='red',
                           arrows=True, arrowsize=20, width=2,
                           ax=axes[0], connectionstyle='arc3,rad=0.1')

    axes[0].set_title('DAG Structure', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    ancestors, descendants = compute_ancestors_descendants(W, threshold)
    sns.heatmap(descendants, annot=True, fmt='.0f', cmap='YlOrRd',
                cbar_kws={'label': 'Reachability'}, ax=axes[1],
                xticklabels=range(d), yticklabels=range(d))
    axes[1].set_title('Descendants Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'dag_structure.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {save_path}")

    return ancestors, descendants


def visualize_bicmae_losses(loss_history, save_dir='data/figures'):
    """可视化BiCMAE损失曲线"""
    ensure_dir(save_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(loss_history['total_loss']) + 1)

    axes[0].plot(epochs, loss_history['total_loss'], linewidth=2, color='purple')
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Total Loss", fontsize=12)
    axes[0].set_title("BiCMAE Total Loss", fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, loss_history['loss_fwd'], linewidth=2, color='blue')
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Forward Mask Loss", fontsize=12)
    axes[1].set_title("Forward Mask Loss", fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, loss_history['loss_bwd'], linewidth=2, color='red')
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("Backward Mask Loss", fontsize=12)
    axes[2].set_title("Backward Mask Loss", fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'bicmae_losses.png')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 保存: {save_path}")


def visualize_masking_mechanism(ancestors, descendants, sample_idx=0, save_dir='data/figures'):
    """可视化掩码机制"""
    ensure_dir(save_dir)

    d = ancestors.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    target = sample_idx
    fwd_mask = 1.0 - ancestors[:, target]

    axes[0].bar(range(d), fwd_mask, color='blue', alpha=0.7)
    axes[0].axvline(x=target, color='red', linestyle='--', linewidth=2, label=f'Target {target}')
    axes[0].set_xlabel('Node Index', fontsize=12)
    axes[0].set_ylabel('Mask Value', fontsize=12)
    axes[0].set_title(f'Forward Mask (mask ancestors of node {target})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    bwd_mask = 1.0 - descendants[target, :]

    axes[1].bar(range(d), bwd_mask, color='red', alpha=0.7)
    axes[1].axvline(x=target, color='red', linestyle='--', linewidth=2, label=f'Target {target}')
    axes[1].set_xlabel('Node Index', fontsize=12)
    axes[1].set_ylabel('Mask Value', fontsize=12)
    axes[1].set_title(f'Backward Mask (mask descendants of node {target})', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'masking_mechanism.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {save_path}")
