#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified GAT Model for Enhancer-Promoter Interaction Prediction.

Architecture:
1. Multi-scale CNN: 直接处理所有10通道特征
2. Graph Construction: 基于特征相似度构建一个图
3. GAT Encoding: 学习节点表示
4. CNN+Transformer Classifier: 分类预测
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
except ImportError:
    raise ImportError("Please install torch-geometric: pip install torch-geometric")


# =============================================================================
# Multi-Scale CNN Feature Extractor
# =============================================================================

class MultiScaleCNN(nn.Module):
    """
    Multi-scale CNN for feature extraction.

    Uses multiple kernel sizes to capture patterns at different scales.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [3, 5, 7, 9],
        dropout: float = 0.1
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        num_kernels = len(kernel_sizes)

        # Each kernel produces out_channels // num_kernels channels
        base_channels = out_channels // num_kernels
        extra_channels = out_channels % num_kernels

        for i, k in enumerate(kernel_sizes):
            curr_channels = base_channels + (1 if i < extra_channels else 0)
            conv = nn.Sequential(
                nn.Conv1d(in_channels, curr_channels, k, padding=k // 2),
                nn.BatchNorm1d(curr_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.convs.append(conv)

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)

        Returns:
            (batch, out_channels, seq_len)
        """
        conv_outs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)
        out = out + self.residual(x)
        return out


class ViewEncoder(nn.Module):
    """
    Single view encoder: CNN layers + pooling.

    Processes all input channels together.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        current_dim = in_channels

        for i in range(num_layers):
            layers.append(MultiScaleCNN(
                current_dim,
                hidden_dim if i < num_layers - 1 else out_dim,
                kernel_sizes=kernel_sizes,
                dropout=dropout
            ))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
            current_dim = hidden_dim if i < num_layers - 1 else out_dim

        self.cnn = nn.Sequential(*layers)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_channels)

        Returns:
            (batch, out_dim) - node embedding
        """
        # (batch, in_channels, seq_len)
        x = x.transpose(1, 2)

        # CNN
        x = self.cnn(x)  # (batch, out_dim, seq_len)

        # Global pooling
        x = self.pool(x).squeeze(-1)  # (batch, out_dim)

        return x


# =============================================================================
# Graph Construction
# =============================================================================

def build_cosine_similarity_graph(
    x: torch.Tensor,
    k: int,
    self_loop: bool = True
) -> torch.Tensor:
    """
    Build graph from cosine similarity (Top-K neighbors).

    Args:
        x: (N, D) node features
        k: Number of neighbors
        self_loop: Whether to add self-loops

    Returns:
        edge_index: (2, E) edge indices
    """
    n = x.shape[0]

    if n <= 1:
        if self_loop:
            return torch.tensor([[0], [0]], dtype=torch.long, device=x.device)
        else:
            return torch.zeros((2, 0), dtype=torch.long, device=x.device)

    # Normalize
    xn = F.normalize(x, p=2, dim=1, eps=1e-8)

    # Compute similarity matrix
    sim = xn @ xn.t()  # (N, N)

    # Set diagonal to -inf to exclude self
    sim.fill_diagonal_(-1e9)

    # Get top-k neighbors
    k_actual = min(k, n - 1)
    _, indices = torch.topk(sim, k=k_actual, dim=1)  # (N, k)

    # Build edge list
    sources = torch.arange(n, device=x.device).unsqueeze(1).expand(-1, k_actual)
    targets = indices

    edges = torch.stack([sources.flatten(), targets.flatten()], dim=0)  # (2, N*k)

    # Make undirected
    edges_rev = torch.stack([edges[1], edges[0]], dim=0)
    edges = torch.cat([edges, edges_rev], dim=1)

    # Remove duplicates
    edges = torch.unique(edges, dim=1)

    # Add self-loops
    if self_loop:
        self_loops = torch.arange(n, device=x.device).unsqueeze(0).expand(2, -1)
        edges = torch.cat([edges, self_loops], dim=1)
        edges = torch.unique(edges, dim=1)

    return edges


# =============================================================================
# GAT Encoder
# =============================================================================

class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder.

    Multiple GAT layers with residual connections.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer: output dimension
                self.gat_layers.append(
                    GATConv(hidden_dim, out_dim, heads=1, dropout=dropout, concat=False)
                )
            else:
                # Intermediate layers
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                           dropout=dropout, concat=True)
                )
            self.norms.append(nn.LayerNorm(hidden_dim if i < num_layers - 1 else out_dim))

        # Output projection if needed
        if hidden_dim != out_dim:
            self.out_proj = nn.Linear(hidden_dim, out_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) edges

        Returns:
            (N, out_dim) node embeddings
        """
        h = self.in_proj(x)

        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            h_prev = h
            h = gat(h, edge_index)
            h = norm(h)

            # Residual (if dimensions match)
            if h.shape == h_prev.shape:
                h = h + h_prev

            if i < self.num_layers - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h


# =============================================================================
# Simplified GAT for Single Region Type (Enhancer or Promoter)
# =============================================================================

class SimplifiedGATRegion(nn.Module):
    """
    Simplified GAT for a single region type (enhancer or promoter).

    直接使用10通道输入，构建一个图，不再使用多视图图。

    Process:
    1. Multi-scale CNN encoding (所有10通道一起)
    2. Graph construction (基于整体特征相似度)
    3. GAT encoding
    """

    def __init__(
        self,
        in_channels: int = 10,  # 直接使用所有10个通道
        hidden_dim: int = 128,
        embed_dim: int = 128,
        num_cnn_layers: int = 2,
        num_gat_layers: int = 2,
        num_heads: int = 4,
        topk: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.topk = topk
        self.embed_dim = embed_dim

        # Multi-scale CNN encoder (直接处理所有通道)
        self.cnn_encoder = ViewEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            num_layers=num_cnn_layers,
            dropout=dropout
        )

        # GAT encoder
        self.gat_encoder = GATEncoder(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            num_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            x: (N, seq_len, C) node features for all N nodes
               C = 10 channels (DNA + epigenetic + conservation)
            return_attention_weights: Whether to return attention weights

        Returns:
            h: (N, embed_dim) final node embeddings
            attn_dict: Optional dict with attention weights
        """
        N = x.shape[0]

        # Multi-scale CNN encoding
        h = self.cnn_encoder(x)  # (N, embed_dim)

        # Build single graph based on feature similarity
        edge_index = build_cosine_similarity_graph(h, k=self.topk, self_loop=True)

        # GAT encoding
        h = self.gat_encoder(h, edge_index)  # (N, embed_dim)

        # Return attention weights if requested
        attn_dict = None
        if return_attention_weights:
            attn_dict = {
                'edge_index': edge_index,
            }

        return h, attn_dict


# =============================================================================
# Full Simplified GAT Model for EPI Prediction
# =============================================================================

class SimplifiedGAT_EPI(nn.Module):
    """
    Simplified GAT model for Enhancer-Promoter Interaction prediction.

    使用简化的架构：
    1. Multi-scale CNN特征提取（所有10通道）
    2. 构建单个图（基于特征相似度）
    3. GAT编码
    4. CNN+Transformer分类
    """

    def __init__(
        self,
        num_channels: int = 10,
        hidden_dim: int = 128,
        embed_dim: int = 128,
        num_cnn_layers: int = 2,
        num_gat_layers: int = 2,
        num_heads: int = 4,
        enhancer_topk: int = 8,
        promoter_topk: int = 8,
        dropout: float = 0.1,
        classifier_type: str = 'cnn_transformer',
        **kwargs
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Enhancer encoder
        self.enhancer_encoder = SimplifiedGATRegion(
            in_channels=num_channels,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_cnn_layers=num_cnn_layers,
            num_gat_layers=num_gat_layers,
            num_heads=num_heads,
            topk=enhancer_topk,
            dropout=dropout
        )

        # Promoter encoder
        self.promoter_encoder = SimplifiedGATRegion(
            in_channels=num_channels,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_cnn_layers=num_cnn_layers,
            num_gat_layers=num_gat_layers,
            num_heads=num_heads,
            topk=promoter_topk,
            dropout=dropout
        )

        # Classifier - CNN + Transformer
        classifier_input_dim = embed_dim * 4

        self.classifier_type = classifier_type

        if self.classifier_type == 'cnn_transformer':
            # CNN处理局部模式
            self.cnn_encoder = nn.Sequential(
                nn.Conv1d(classifier_input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # 投影层
            self.proj_e = nn.Linear(embed_dim, hidden_dim)
            self.proj_p = nn.Linear(embed_dim, hidden_dim)

            # Transformer全局交互
            transformer_num_heads = kwargs.get('num_heads', num_heads)
            transformer_num_layers = kwargs.get('num_layers', 2)

            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=transformer_num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=transformer_num_layers
            )

            # 分类头
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

        elif self.classifier_type == 'mlp':
            # 简单MLP分类器
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")

    def forward(
        self,
        enhancer_x: torch.Tensor,
        promoter_x: torch.Tensor,
        pair_e_idx: torch.Tensor,
        pair_p_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            enhancer_x: (N_e, seq_len_e, C) all enhancer features
            promoter_x: (N_p, seq_len_p, C) all promoter features
            pair_e_idx: (batch,) indices of enhancers in pairs
            pair_p_idx: (batch,) indices of promoters in pairs

        Returns:
            logits: (batch,) prediction logits
        """
        # Encode enhancers
        h_e, _ = self.enhancer_encoder(enhancer_x)  # (N_e, embed_dim)

        # Encode promoters
        h_p, _ = self.promoter_encoder(promoter_x)  # (N_p, embed_dim)

        # Get paired embeddings
        h_e_pair = h_e[pair_e_idx]  # (batch, embed_dim)
        h_p_pair = h_p[pair_p_idx]  # (batch, embed_dim)

        # Classification
        if self.classifier_type == 'cnn_transformer':
            # 特征组合
            h_combined = torch.cat([
                h_e_pair,
                h_p_pair,
                h_e_pair * h_p_pair,
                torch.abs(h_e_pair - h_p_pair)
            ], dim=-1)

            # CNN处理局部模式
            h_seq = h_combined.unsqueeze(1).transpose(1, 2)
            h_cnn = self.cnn_encoder(h_seq).squeeze(-1)

            # 投影原始特征
            h_e_proj = self.proj_e(h_e_pair)
            h_p_proj = self.proj_p(h_p_pair)

            # 构建Transformer序列
            h_transformer_input = torch.stack([h_e_proj, h_p_proj, h_cnn], dim=1)

            # Transformer编码
            h_transformed = self.transformer_encoder(h_transformer_input)

            # 分类
            logits = self.classifier(h_transformed[:, 0, :]).squeeze(-1)

        elif self.classifier_type == 'mlp':
            h_combined = torch.cat([
                h_e_pair,
                h_p_pair,
                h_e_pair * h_p_pair,
                torch.abs(h_e_pair - h_p_pair)
            ], dim=-1)

            logits = self.classifier(h_combined).squeeze(-1)

        return logits

    def predict_proba(
        self,
        enhancer_x: torch.Tensor,
        promoter_x: torch.Tensor,
        pair_e_idx: torch.Tensor,
        pair_p_idx: torch.Tensor
    ) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(enhancer_x, promoter_x, pair_e_idx, pair_p_idx)
        return torch.sigmoid(logits)


def create_model(
    hidden_dim: int = 128,
    embed_dim: int = 128,
    num_cnn_layers: int = 2,
    num_gat_layers: int = 2,
    num_heads: int = 4,
    enhancer_topk: int = 8,
    promoter_topk: int = 8,
    dropout: float = 0.1,
    classifier_type: str = 'cnn_transformer',
    device: str = 'cuda'
) -> SimplifiedGAT_EPI:
    """Create and initialize model."""
    model = SimplifiedGAT_EPI(
        num_channels=10,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_cnn_layers=num_cnn_layers,
        num_gat_layers=num_gat_layers,
        num_heads=num_heads,
        enhancer_topk=enhancer_topk,
        promoter_topk=promoter_topk,
        dropout=dropout,
        classifier_type=classifier_type
    )
    model = model.to(device)
    return model


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("Testing SimplifiedGAT_EPI model...")

    # Create dummy data
    N_enhancers = 50
    N_promoters = 50
    enhancer_len = 3000
    promoter_len = 2000
    num_channels = 10
    batch_size = 8

    # Random features
    enhancer_x = torch.randn(N_enhancers, enhancer_len, num_channels)
    promoter_x = torch.randn(N_promoters, promoter_len, num_channels)

    # Random pairs
    pair_e_idx = torch.randint(0, N_enhancers, (batch_size,))
    pair_p_idx = torch.randint(0, N_promoters, (batch_size,))

    # Create model
    model = create_model(device='cpu')

    # Forward
    logits = model(enhancer_x, promoter_x, pair_e_idx, pair_p_idx)
    proba = model.predict_proba(enhancer_x, promoter_x, pair_e_idx, pair_p_idx)

    print(f"\nInput shapes:")
    print(f"  enhancer_x: {enhancer_x.shape}")
    print(f"  promoter_x: {promoter_x.shape}")
    print(f"  pair_e_idx: {pair_e_idx.shape}")
    print(f"  pair_p_idx: {pair_p_idx.shape}")

    print(f"\nOutput shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  proba: {proba.shape}")

    print(f"\nSample outputs:")
    print(f"  logits: {logits[:5]}")
    print(f"  proba: {proba[:5]}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")