"""
1-D CNN classifier for raw time-domain power electronics waveforms.

Architecture:
  Input: (batch, n_channels, window_size)
  → 4× [Conv1d → BatchNorm → ReLU → MaxPool]
  → Global Average Pooling
  → Dropout → Linear → Softmax

This model processes raw signal waveforms directly, learning
discriminative time-domain patterns without hand-crafted features.
It is fast to train and serves as the primary baseline.

Reference:
  Wang, Z., Yan, W., & Oates, T. (2017). Time series classification
  from scratch with deep neural networks: A strong baseline.
  IJCNN 2017.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """1-D residual block with two Conv layers and skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class CNN1DWaveformClassifier(nn.Module):
    """1-D residual CNN for multi-class fault classification.

    Parameters
    ----------
    n_channels : int
        Number of input signal channels (e.g., 6 for Va/Vb/Vc/Ia/Ib/Ic).
    n_classes : int
        Number of fault classes.
    window_size : int
        Input signal length (samples).
    base_filters : int
        Base number of filters, doubled at each stage.
    kernel_size : int
        Convolutional kernel size.
    dropout : float
        Dropout rate before the classifier head.

    Examples
    --------
    >>> model = CNN1DWaveformClassifier(n_channels=6, n_classes=9)
    >>> x = torch.randn(32, 6, 1024)
    >>> logits = model(x)  # shape: (32, 9)
    """

    def __init__(
        self,
        n_channels: int = 6,
        n_classes: int = 9,
        window_size: int = 1024,
        base_filters: int = 64,
        kernel_size: int = 7,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Stem: initial projection
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, base_filters, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # 4 residual stages with progressive downsampling and filter doubling
        self.stage1 = ResidualBlock1D(base_filters, base_filters, kernel_size, dropout=dropout)
        self.pool1 = nn.MaxPool1d(2)

        self.stage2 = ResidualBlock1D(base_filters, base_filters * 2, kernel_size, dropout=dropout)
        self.pool2 = nn.MaxPool1d(2)

        self.stage3 = ResidualBlock1D(base_filters * 2, base_filters * 4, kernel_size, dropout=dropout)
        self.pool3 = nn.MaxPool1d(2)

        self.stage4 = ResidualBlock1D(base_filters * 4, base_filters * 8, kernel_size, dropout=dropout)

        # Global average pooling + classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_filters * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
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
        x = self.stem(x)
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.stage4(x)
        x = self.gap(x)
        return self.classifier(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return GAP embeddings before the classifier head."""
        x = self.stem(x)
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.stage4(x)
        return self.gap(x).squeeze(-1)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
