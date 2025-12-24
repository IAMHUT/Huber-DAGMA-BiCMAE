"""
拓扑感知双向因果掩码自编码器 (Topology-Aware BiCMAE)
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .dagma import DAGMA, LinearDAGMA, dagma_linear_huber
from .bicmae import (
    CausalEncoder,
    TopologyAwareBidirectionalForwardModel,
    BiCMAEDynamicsModel,
    BiCMAETrainer
)
from .utils import (
    generate_linear_sem,
    simulate_controlled_dynamics,
    compute_metrics,
    compute_ancestors_descendants,
    TransitionDataset
)

__all__ = [
    'DAGMA',
    'LinearDAGMA',
    'dagma_linear_huber',
    'CausalEncoder',
    'TopologyAwareBidirectionalForwardModel',
    'BiCMAEDynamicsModel',
    'BiCMAETrainer',
    'generate_linear_sem',
    'simulate_controlled_dynamics',
    'compute_metrics',
    'compute_ancestors_descendants',
    'TransitionDataset'
]
