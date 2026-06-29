#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer for Multi-View GAT EPI model.

Features:
- Training and validation loops
- Metrics: Loss, Accuracy, AUROC, AUPRC
- Early stopping
- Learning rate scheduling
- Mixed precision training (AMP)
- Model checkpointing
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainMetrics:
    """Training/validation metrics."""
    loss: float
    acc: float
    auroc: Optional[float] = None
    auprc: Optional[float] = None


def compute_auc(labels: torch.Tensor, scores: torch.Tensor) -> Optional[float]:
    """Compute AUROC using rank-based method."""
    y = labels.detach().float().view(-1)
    s = scores.detach().view(-1)
    n = y.numel()

    if n < 2:
        return None

    pos = (y > 0.5).sum().item()
    neg = n - pos

    if pos == 0 or neg == 0:
        return None

    # Rank-based AUC
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float64, device=s.device)

    sum_ranks_pos = ranks[y > 0.5].sum()
    auc = (sum_ranks_pos - pos * (pos + 1) / 2) / (pos * neg)

    return float(auc.item())


def compute_auprc(labels: torch.Tensor, scores: torch.Tensor) -> Optional[float]:
    """Compute AUPRC (Average Precision)."""
    y = labels.detach().float().view(-1)
    s = scores.detach().float().view(-1)

    if y.numel() < 2:
        return None

    y_bin = (y > 0.5).to(torch.long)
    pos = int((y_bin == 1).sum().item())

    if pos == 0 or pos == int(y_bin.numel()):
        return None

    # Sort by decreasing score
    order = torch.argsort(s, descending=True)
    y_sorted = y_bin[order]

    tp_cum = torch.cumsum(y_sorted == 1, dim=0).to(torch.float64)
    fp_cum = torch.cumsum(y_sorted == 0, dim=0).to(torch.float64)
    precision = tp_cum / (tp_cum + fp_cum + 1e-12)
    recall = tp_cum / float(pos)

    # Stepwise AP
    ap = 0.0
    recall_prev = 0.0

    for i in range(int(y_sorted.numel())):
        if int(y_sorted[i].item()) == 1:
            r = float(recall[i].item())
            p = float(precision[i].item())
            ap += p * max(r - recall_prev, 0.0)
            recall_prev = r

    return float(ap)


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Trainer for Multi-View GAT EPI model.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        pos_weight: Optional[float] = None,
        amp: bool = True,
        grad_clip: float = 5.0
    ):
        """
        Args:
            model: The model to train
            device: Device to use
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            pos_weight: Positive weight for BCE loss
            amp: Use mixed precision training
            grad_clip: Gradient clipping value
        """
        self.model = model.to(device)
        self.device = device
        self.amp = amp and (device.type == 'cuda')
        self.grad_clip = grad_clip

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Loss function
        pw = torch.tensor([pos_weight], device=device) if pos_weight else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def _prepare_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch data."""
        enhancer = batch['enhancer'].to(self.device)
        promoter = batch['promoter'].to(self.device)
        labels = batch['label'].to(self.device)

        # For now, each sample is a separate node
        # pair indices are just range(batch_size)
        batch_size = enhancer.shape[0]
        pair_e_idx = torch.arange(batch_size, device=self.device)
        pair_p_idx = torch.arange(batch_size, device=self.device)

        return enhancer, promoter, pair_e_idx, pair_p_idx, labels

    def train_epoch(self, loader: DataLoader) -> TrainMetrics:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_n = 0
        correct = 0
        all_labels = []
        all_scores = []

        pbar = tqdm(loader, desc="Training", leave=False)

        for batch in pbar:
            enhancer, promoter, pair_e_idx, pair_p_idx, labels = self._prepare_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)

            if self.amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(enhancer, promoter, pair_e_idx, pair_p_idx)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(enhancer, promoter, pair_e_idx, pair_p_idx)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # Metrics
            bs = labels.numel()
            total_loss += float(loss.item()) * bs
            total_n += bs

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += int((preds == labels).sum().item())

            all_labels.append(labels.detach().cpu())
            all_scores.append(probs.detach().cpu())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Compute final metrics
        y = torch.cat(all_labels)
        s = torch.cat(all_scores)

        return TrainMetrics(
            loss=total_loss / max(total_n, 1),
            acc=correct / max(total_n, 1),
            auroc=compute_auc(y, s),
            auprc=compute_auprc(y, s)
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> TrainMetrics:
        """Evaluate on validation/test set."""
        self.model.eval()

        total_loss = 0.0
        total_n = 0
        correct = 0
        all_labels = []
        all_scores = []

        for batch in loader:
            enhancer, promoter, pair_e_idx, pair_p_idx, labels = self._prepare_batch(batch)

            if self.amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(enhancer, promoter, pair_e_idx, pair_p_idx)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(enhancer, promoter, pair_e_idx, pair_p_idx)
                loss = self.criterion(logits, labels)

            bs = labels.numel()
            total_loss += float(loss.item()) * bs
            total_n += bs

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += int((preds == labels).sum().item())

            all_labels.append(labels.detach().cpu())
            all_scores.append(probs.detach().cpu())

        y = torch.cat(all_labels)
        s = torch.cat(all_scores)

        return TrainMetrics(
            loss=total_loss / max(total_n, 1),
            acc=correct / max(total_n, 1),
            auroc=compute_auc(y, s),
            auprc=compute_auprc(y, s)
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 15,
        output_dir: Optional[str] = None,
        save_best: bool = True
    ) -> Dict:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            patience: Early stopping patience
            output_dir: Directory to save checkpoints
            save_best: Save best model

        Returns:
            Training history dict
        """
        early_stopping = EarlyStopping(patience=patience)

        history = {
            'train_loss': [],
            'train_acc': [],
            'train_auroc': [],
            'train_auprc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auroc': [],
            'val_auprc': [],
        }

        best_val_loss = float('inf')
        best_state = None

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.evaluate(val_loader)

            # Update scheduler
            self.scheduler.step(val_metrics.loss)

            # Record history
            history['train_loss'].append(train_metrics.loss)
            history['train_acc'].append(train_metrics.acc)
            history['train_auroc'].append(train_metrics.auroc)
            history['train_auprc'].append(train_metrics.auprc)
            history['val_loss'].append(val_metrics.loss)
            history['val_acc'].append(val_metrics.acc)
            history['val_auroc'].append(val_metrics.auroc)
            history['val_auprc'].append(val_metrics.auprc)

            # Print progress
            print(f"Epoch {epoch:03d}")
            print(f"  Train: loss={train_metrics.loss:.4f}, acc={train_metrics.acc:.3f}", end='')
            if train_metrics.auroc is not None:
                print(f", auroc={train_metrics.auroc:.3f}", end='')
            if train_metrics.auprc is not None:
                print(f", auprc={train_metrics.auprc:.3f}", end='')
            print()

            print(f"  Val:   loss={val_metrics.loss:.4f}, acc={val_metrics.acc:.3f}", end='')
            if val_metrics.auroc is not None:
                print(f", auroc={val_metrics.auroc:.3f}", end='')
            if val_metrics.auprc is not None:
                print(f", auprc={val_metrics.auprc:.3f}", end='')
            print()

            # Save best model
            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                if output_dir and save_best:
                    torch.save(best_state, os.path.join(output_dir, 'best_model.pt'))
                    print(f"  Saved best model (val_loss={val_metrics.loss:.4f})")

            # Early stopping
            if early_stopping(val_metrics.loss):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\nLoaded best model (val_loss={best_val_loss:.4f})")

        return history


def create_trainer(
    model: nn.Module,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    pos_weight: Optional[float] = None,
    amp: bool = True
) -> Trainer:
    """Create a trainer instance."""
    return Trainer(
        model=model,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        pos_weight=pos_weight,
        amp=amp
    )
