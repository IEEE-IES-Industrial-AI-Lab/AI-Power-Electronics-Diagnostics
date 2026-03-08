"""
Patch-based Transformer encoder for power electronics signal classification.

Architecture inspired by PatchTST (Nie et al., 2022) and ViT:
  Input: (batch, n_channels, window_size)
  → Split into non-overlapping patches of size patch_size
  → Linear patch embedding + positional encoding
  → N× Transformer encoder blocks (multi-head self-attention + FFN)
  → [CLS] token → Dropout → Linear classifier

Each channel is processed independently with shared weights (channel-
independent patching), then representations are aggregated by mean pooling
across channels before the classifier.

Reference:
  Nie, Y., et al. (2022). A time series is worth 64 words: Long-term
  forecasting with Transformers. ICLR 2023.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Split a 1-D signal into patches and project to embedding dimension."""

    def __init__(
        self,
        window_size: int,
        patch_size: int,
        d_model: int,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        assert window_size % patch_size == 0, (
            f"window_size ({window_size}) must be divisible by patch_size ({patch_size})"
        )
        self.n_patches = window_size // patch_size
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * in_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, C, T)

        Returns
        -------
        patches : torch.Tensor, shape (B, n_patches, d_model)
        """
        B, C, T = x.shape
        # Reshape to (B, n_patches, patch_size * C)
        x = x.reshape(B, C, self.n_patches, self.patch_size)
        x = x.permute(0, 2, 3, 1)  # (B, n_patches, patch_size, C)
        x = x.reshape(B, self.n_patches, self.patch_size * C)
        return self.projection(x)  # (B, n_patches, d_model)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for patch sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block (pre-norm formulation)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual (pre-norm)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerSignalClassifier(nn.Module):
    """Patch-based Transformer for fault classification.

    Parameters
    ----------
    n_channels : int
        Number of input signal channels.
    n_classes : int
        Number of fault classes.
    window_size : int
        Signal length in samples.
    patch_size : int
        Patch size in samples. Must divide ``window_size``.
    d_model : int
        Transformer embedding dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer encoder blocks.
    ffn_dim : int
        Feed-forward hidden dimension. Defaults to 4 × d_model.
    dropout : float
        Dropout rate.

    Examples
    --------
    >>> model = TransformerSignalClassifier(n_channels=6, n_classes=9)
    >>> x = torch.randn(32, 6, 1024)
    >>> logits = model(x)  # (32, 9)
    """

    def __init__(
        self,
        n_channels: int = 6,
        n_classes: int = 9,
        window_size: int = 1024,
        patch_size: int = 64,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        ffn_dim = ffn_dim or d_model * 4

        # Channel-independent patching: process each channel separately
        self.patch_embed = PatchEmbedding(
            window_size=window_size,
            patch_size=patch_size,
            d_model=d_model,
            in_channels=1,  # each channel processed alone
        )
        self.pos_enc = PositionalEncoding(d_model, max_len=window_size // patch_size + 1, dropout=dropout)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, T)

        Returns
        -------
        logits : torch.Tensor, shape (B, n_classes)
        """
        B, C, T = x.shape

        # Process each channel independently, average patch embeddings
        channel_tokens = []
        for c in range(C):
            ch = x[:, c : c + 1, :]  # (B, 1, T)
            patches = self.patch_embed(ch)  # (B, n_patches, d_model)

            # Prepend CLS token
            cls = self.cls_token.expand(B, -1, -1)
            patches = torch.cat([cls, patches], dim=1)  # (B, 1+n_patches, d_model)
            patches = self.pos_enc(patches)

            for block in self.encoder:
                patches = block(patches)

            channel_tokens.append(patches[:, 0])  # CLS token: (B, d_model)

        # Aggregate channels
        x = torch.stack(channel_tokens, dim=1).mean(dim=1)  # (B, d_model)
        x = self.norm(x)
        return self.classifier(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return d_model embeddings before the classifier."""
        B, C, T = x.shape
        channel_tokens = []
        for c in range(C):
            ch = x[:, c : c + 1, :]
            patches = self.patch_embed(ch)
            cls = self.cls_token.expand(B, -1, -1)
            patches = torch.cat([cls, patches], dim=1)
            patches = self.pos_enc(patches)
            for block in self.encoder:
                patches = block(patches)
            channel_tokens.append(patches[:, 0])
        x = torch.stack(channel_tokens, dim=1).mean(dim=1)
        return self.norm(x)

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
