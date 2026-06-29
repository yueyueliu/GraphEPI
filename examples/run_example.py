#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for training GraphEPI model.

This script demonstrates how to use GraphEPI for enhancer-promoter interaction prediction.
"""

import os
import sys

# Add project root to path
_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from EPI import get_dataloaders, create_model, create_trainer
import torch


def main():
    """Example training script."""

    print("=" * 60)
    print("GraphEPI Example Training")
    print("=" * 60)

    # Parameters
    hdf5_path = 'datasets/test_100_samples_processed.h5'
    batch_size = 16
    epochs = 10
    lr = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Data: {hdf5_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Device: {device}")

    # Load data
    print(f"\n{'='*60}")
    print("Loading data...")
    print("=" * 60)

    train_loader, val_loader, test_loader, info = get_dataloaders(
        hdf5_path=hdf5_path,
        batch_size=batch_size,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    # Compute pos weight
    import h5py
    with h5py.File(hdf5_path, 'r') as hf:
        labels = hf['label'][:]

    pos_count = (labels == 1).sum()
    neg_count = (labels == 0).sum()
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    print(f"\nLabel distribution:")
    print(f"  Positive: {pos_count}")
    print(f"  Negative: {neg_count}")
    print(f"  Pos weight: {pos_weight:.3f}")

    # Create model
    print(f"\n{'='*60}")
    print("Creating model...")
    print("=" * 60)

    model = create_model(
        hidden_dim=128,
        embed_dim=128,
        num_cnn_layers=2,
        num_gat_layers=2,
        num_heads=4,
        enhancer_topk=8,
        promoter_topk=8,
        dropout=0.1,
        device=device
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = create_trainer(
        model=model,
        device=torch.device(device),
        lr=lr,
        weight_decay=1e-5,
        pos_weight=pos_weight,
        amp=(device == 'cuda')
    )

    # Train
    print(f"\n{'='*60}")
    print("Training...")
    print("=" * 60)

    os.makedirs('outputs', exist_ok=True)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=5,
        output_dir='outputs',
        save_best=True
    )

    # Test
    print(f"\n{'='*60}")
    print("Testing...")
    print("=" * 60)

    test_metrics = trainer.evaluate(test_loader)

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics.loss:.4f}")
    print(f"  Accuracy: {test_metrics.acc:.3f}")
    if test_metrics.auroc is not None:
        print(f"  AUROC: {test_metrics.auroc:.3f}")
    if test_metrics.auprc is not None:
        print(f"  AUPRC: {test_metrics.auprc:.3f}")

    print("\n" + "=" * 60)
    print("Example training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()