#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script for Multi-View GAT EPI model.

Usage:
    python -m EPI.train_multiview \
        --hdf5 datasets/test_data/debug_processed.h5 \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-3 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# Add repo root to path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from EPI.model_multiview import create_model
from EPI.data_loader import get_dataloaders, compute_pos_weight
from EPI.trainer_multiview import create_trainer


def main():
    parser = argparse.ArgumentParser(
        description='Train Multi-View GAT EPI model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument(
        '--hdf5',
        type=str,
        required=True,
        help='Path to preprocessed HDF5 file'
    )

    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--embed-dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num-cnn-layers', type=int, default=2, help='Number of CNN layers')
    parser.add_argument('--num-gat-layers', type=int, default=2, help='Number of GAT layers')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--enhancer-topk', type=int, default=8, help='Top-K neighbors for enhancer graph')
    parser.add_argument('--promoter-topk', type=int, default=8, help='Top-K neighbors for promoter graph')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')

    # Data split
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')

    # Output
    parser.add_argument('--output', type=str, default='./outputs', help='Output directory')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Multi-View GAT EPI Training")
    print("=" * 60)
    print(f"\nArguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # Load data
    print(f"\n{'='*60}")
    print("Loading data...")
    print("=" * 60)

    train_loader, val_loader, test_loader, info = get_dataloaders(
        hdf5_path=args.hdf5,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers
    )

    # Compute pos weight
    # Load labels from HDF5
    import h5py
    with h5py.File(args.hdf5, 'r') as hf:
        labels = hf['label'][:]
    pos_weight = compute_pos_weight(labels)
    print(f"\nPos weight for BCE loss: {pos_weight:.3f}")

    # Create model
    print(f"\n{'='*60}")
    print("Creating model...")
    print("=" * 60)

    model = create_model(
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_cnn_layers=args.num_cnn_layers,
        num_gat_layers=args.num_gat_layers,
        num_heads=args.num_heads,
        enhancer_topk=args.enhancer_topk,
        promoter_topk=args.promoter_topk,
        dropout=args.dropout,
        device=args.device
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = create_trainer(
        model=model,
        device=torch.device(args.device),
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        amp=not args.no_amp
    )

    # Train
    print(f"\n{'='*60}")
    print("Training...")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output,
        save_best=True
    )

    # Test
    print(f"\n{'='*60}")
    print("Testing...")
    print("=" * 60)

    # Load best model
    best_model_path = os.path.join(args.output, 'best_model.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))

    trainer.model = model.to(args.device)
    test_metrics = trainer.evaluate(test_loader)

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics.loss:.4f}")
    print(f"  Accuracy: {test_metrics.acc:.3f}")
    if test_metrics.auroc is not None:
        print(f"  AUROC: {test_metrics.auroc:.3f}")
    if test_metrics.auprc is not None:
        print(f"  AUPRC: {test_metrics.auprc:.3f}")

    # Save results
    results = {
        'args': {arg: str(getattr(args, arg)) for arg in vars(args)},
        'history': {k: [float(v) if v is not None else None for v in vals]
                   for k, vals in history.items()},
        'test_metrics': {
            'loss': float(test_metrics.loss),
            'acc': float(test_metrics.acc),
            'auroc': float(test_metrics.auroc) if test_metrics.auroc else None,
            'auprc': float(test_metrics.auprc) if test_metrics.auprc else None,
        }
    }

    results_path = os.path.join(args.output, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
