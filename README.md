# GraphEPI

GraphEPI: A Graph Neural Network Approach for Enhancer-Promoter Interaction Prediction

## Overview

GraphEPI is a deep learning framework for predicting enhancer-promoter interactions (EPI) using graph neural networks. The model leverages multi-scale CNN feature extraction and graph attention networks (GAT) to learn the complex relationships between enhancers and promoters.

## Architecture

The model architecture consists of:

1. **Multi-Scale CNN Feature Extractor**: Extracts features from enhancer and promoter sequences using multiple kernel sizes (3, 7, 9, 11).
2. **Graph Construction**: Builds graphs based on cosine similarity between feature embeddings using Top-K neighbors.
3. **Graph Attention Network (GAT)**: Learns node representations through multi-head attention mechanisms.
4. **CNN+Transformer Classifier**: Combines enhancer and promoter embeddings for interaction prediction.

## Installation

### Prerequisites

- Python >= 3.8
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

For PyTorch Geometric installation, please refer to the [official documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your specific PyTorch and CUDA version.

## Quick Start

### Training with Sample Data

```bash
cd /home/liuyue/LiuProject/GraphEPI
python train_multiview.py \
    --hdf5 datasets/test_100_samples_processed.h5 \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-3 \
    --device cuda
```

### Basic Usage in Python

```python
from EPI import get_dataloaders, create_model, create_trainer
import torch

# Load data
train_loader, val_loader, test_loader, info = get_dataloaders(
    hdf5_path='datasets/test_100_samples_processed.h5',
    batch_size=16,
    val_ratio=0.15,
    test_ratio=0.15
)

# Create model
model = create_model(
    hidden_dim=128,
    embed_dim=128,
    num_cnn_layers=2,
    num_gat_layers=2,
    num_heads=4,
    device='cuda'
)

# Create trainer
trainer = create_trainer(
    model=model,
    device=torch.device('cuda'),
    lr=1e-3,
    weight_decay=1e-5
)

# Train
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    patience=15
)
```

## Input Data Format

### Raw Data (CSV)

The raw data should contain the following columns:

- `enhancer_chrom`: Enhancer chromosome
- `enhancer_start`: Enhancer start position
- `enhancer_end`: Enhancer end position
- `promoter_chrom`: Promoter chromosome
- `promoter_start`: Promoter start position
- `promoter_end`: Promoter end position
- `label`: Interaction label (0 or 1)

See `datasets/test_100_samples.csv` for an example.

### Preprocessed Data (HDF5)

The preprocessed HDF5 file should contain:

- `enhancer`: (N, enhancer_length, num_channels) enhancer features
- `promoter`: (N, promoter_length, num_channels) promoter features
- `label`: (N,) interaction labels
- Attributes:
  - `enhancer_length`: Length of enhancer sequences
  - `promoter_length`: Length of promoter sequences
  - `num_channels`: Number of feature channels (default: 10)
  - `feature_names`: List of feature names

## Features

The model uses 10 feature channels:

1. DNA sequence (one-hot encoding: A, C, G, T)
2. Histone modifications (H3K4me1, H3K4me3, H3K27ac, H3K27me3)
3. Chromatin accessibility (ATAC-seq/DNase-seq)
4. DNA methylation
5. Conservation scores

## Output

The training script produces:

- `outputs/best_model.pt`: Best model checkpoint
- `outputs/results.json`: Training and test results

## Citation

If you use GraphEPI in your research, please cite:

```bibtex
~ to do
```

## License

This project is licensed under the MIT License.

