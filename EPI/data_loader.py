#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loader for Multi-View GAT EPI prediction.

Loads preprocessed HDF5 data and creates PyTorch DataLoaders.
Supports both memory-efficient loading and full in-memory loading.

Usage:
    from EPI.data_loader import get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders(
        hdf5_path='datasets/test_data/debug_processed.h5',
        batch_size=32,
        val_ratio=0.15,
        test_ratio=0.15
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


@dataclass
class EPIDatasetInfo:
    """Dataset metadata."""
    num_samples: int
    enhancer_length: int
    promoter_length: int
    num_channels: int
    feature_names: List[str]


class EPIHDF5Dataset(Dataset):
    """
    Dataset for EPI prediction from HDF5.

    Loads enhancer-promoter pairs from preprocessed HDF5 file.
    """

    def __init__(
        self,
        hdf5_path: str,
        load_in_memory: bool = True
    ):
        """
        Args:
            hdf5_path: Path to HDF5 file
            load_in_memory: If True, load all data into memory
        """
        self.hdf5_path = hdf5_path
        self.load_in_memory = load_in_memory

        # Read metadata
        with h5py.File(hdf5_path, 'r') as hf:
            self.enhancer_length = int(hf.attrs['enhancer_length'])
            self.promoter_length = int(hf.attrs['promoter_length'])
            self.num_channels = int(hf.attrs['num_channels'])
            self.feature_names = [fn.decode() for fn in hf.attrs['feature_names']]
            self.num_samples = hf['label'].shape[0]

        if load_in_memory:
            # Load all data into memory
            with h5py.File(hdf5_path, 'r') as hf:
                self.enhancer_data = hf['enhancer'][:]
                self.promoter_data = hf['promoter'][:]
                self.labels = hf['label'][:]
            self.hf = None
        else:
            # Keep file handle open for lazy loading
            self.hf = h5py.File(hdf5_path, 'r')
            self.enhancer_data = self.hf['enhancer']
            self.promoter_data = self.hf['promoter']
            self.labels = self.hf['label']

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Returns:
            enhancer: (enhancer_length, num_channels)
            promoter: (promoter_length, num_channels)
            label: int
        """
        return (
            self.enhancer_data[idx],
            self.promoter_data[idx],
            int(self.labels[idx])
        )

    def get_info(self) -> EPIDatasetInfo:
        """Get dataset info."""
        return EPIDatasetInfo(
            num_samples=self.num_samples,
            enhancer_length=self.enhancer_length,
            promoter_length=self.promoter_length,
            num_channels=self.num_channels,
            feature_names=self.feature_names
        )

    def close(self):
        """Close file handle if open."""
        if self.hf is not None:
            self.hf.close()
            self.hf = None

    def __del__(self):
        self.close()


class EPICollate:
    """Collate function for DataLoader."""

    def __call__(self, batch: List[Tuple]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: List of (enhancer, promoter, label) tuples

        Returns:
            Dict with 'enhancer', 'promoter', 'label' tensors
        """
        enhancers, promoters, labels = zip(*batch)

        return {
            'enhancer': torch.tensor(np.array(enhancers), dtype=torch.float32),
            'promoter': torch.tensor(np.array(promoters), dtype=torch.float32),
            'label': torch.tensor(labels, dtype=torch.float32)
        }


def stratified_split(
    labels: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified split maintaining label distribution.

    Args:
        labels: Array of labels
        n_train, n_val, n_test: Number of samples for each split
        seed: Random seed

    Returns:
        train_indices, val_indices, test_indices
    """
    np.random.seed(seed)
    n = len(labels)

    # Get indices by label
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    # Shuffle
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    # Calculate proportions
    pos_ratio = len(pos_idx) / n
    neg_ratio = len(neg_idx) / n

    # Split each class
    n_pos_train = int(round(n_train * pos_ratio))
    n_neg_train = int(round(n_train * neg_ratio))
    n_pos_val = int(round(n_val * pos_ratio))
    n_neg_val = int(round(n_val * neg_ratio))

    # Ensure we don't exceed available samples
    n_pos_train = min(n_pos_train, len(pos_idx))
    n_neg_train = min(n_neg_train, len(neg_idx))
    n_pos_val = min(n_pos_val, len(pos_idx) - n_pos_train)
    n_neg_val = min(n_neg_val, len(neg_idx) - n_neg_train)

    # Build splits
    train_idx = list(pos_idx[:n_pos_train]) + list(neg_idx[:n_neg_train])
    val_idx = list(pos_idx[n_pos_train:n_pos_train+n_pos_val]) + \
              list(neg_idx[n_neg_train:n_neg_train+n_neg_val])
    test_idx = list(pos_idx[n_pos_train+n_pos_val:]) + \
               list(neg_idx[n_neg_train+n_neg_val:])

    # Adjust test size
    if len(test_idx) > n_test:
        np.random.shuffle(test_idx)
        # Move extra to val
        extra = test_idx[n_test:]
        test_idx = test_idx[:n_test]
        val_idx.extend(extra)
    elif len(test_idx) < n_test:
        # Move from val to test
        need = n_test - len(test_idx)
        if len(val_idx) > need:
            np.random.shuffle(val_idx)
            test_idx.extend(val_idx[:need])
            val_idx = val_idx[need:]

    # Shuffle each split
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    return list(train_idx), list(val_idx), list(test_idx)


def get_dataloaders(
    hdf5_path: str,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    load_in_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, EPIDatasetInfo]:
    """
    Create train/val/test DataLoaders.

    Args:
        hdf5_path: Path to HDF5 file
        batch_size: Batch size
        val_ratio: Validation ratio
        test_ratio: Test ratio
        seed: Random seed
        num_workers: Number of DataLoader workers
        load_in_memory: Load all data into memory

    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    # Load full dataset
    dataset = EPIHDF5Dataset(hdf5_path, load_in_memory=load_in_memory)
    info = dataset.get_info()

    print(f"Dataset info:")
    print(f"  Total samples: {info.num_samples}")
    print(f"  Enhancer length: {info.enhancer_length}")
    print(f"  Promoter length: {info.promoter_length}")
    print(f"  Num channels: {info.num_channels}")
    print(f"  Features: {info.feature_names}")

    # Calculate split sizes
    n = info.num_samples
    n_train = int(round(n * (1 - val_ratio - test_ratio)))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    # Stratified split
    labels = dataset.labels[:] if load_in_memory else dataset.labels[:]
    train_idx, val_idx, test_idx = stratified_split(
        labels, n_train, n_val, n_test, seed
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val: {len(val_idx)}")
    print(f"  Test: {len(test_idx)}")

    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Create dataloaders
    collate = EPICollate()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, info


def compute_pos_weight(labels: np.ndarray) -> float:
    """Compute positive weight for BCE loss."""
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos < 1e-6:
        return 1.0
    return max(n_neg / n_pos, 1e-3)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <hdf5_path>")
        sys.exit(1)

    hdf5_path = sys.argv[1]

    train_loader, val_loader, test_loader, info = get_dataloaders(
        hdf5_path,
        batch_size=4,
        val_ratio=0.15,
        test_ratio=0.15
    )

    print("\nTesting DataLoader...")
    for batch in train_loader:
        print(f"Batch shapes:")
        print(f"  enhancer: {batch['enhancer'].shape}")
        print(f"  promoter: {batch['promoter'].shape}")
        print(f"  label: {batch['label'].shape}")
        print(f"Labels: {batch['label']}")
        break
