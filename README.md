# Topology-Aware BiCMAE: Causal Discovery meets Bidirectional Masked Autoencoders

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular framework that combines **DAGMA causal discovery** with **Bidirectional Causal Masked Autoencoders (BiCMAE)** for learning interpretable causal dynamics from observational data.

## 🔥 Key Features

- **🎯 Causal Discovery**: DAGMA algorithm with Huber loss for robust DAG learning
- **🔄 Bidirectional Masking**: Novel forward/backward masking strategy guided by causal topology
- **📊 Modular Design**: Clean separation of causal discovery, dynamics modeling, and analysis
- **🚀 Easy to Use**: Three-step pipeline from raw data to trained models
- **📈 Comprehensive Evaluation**: Built-in metrics and visualization tools

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/causal-bicmae.git
cd causal-bicmae

# Create virtual environment (recommended)
conda create -n causal-env python=3.8
conda activate causal-env

# Install dependencies
pip install -r requirements.txt
