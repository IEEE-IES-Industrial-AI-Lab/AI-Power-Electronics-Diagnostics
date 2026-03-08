"""
2-D CNN (ResNet-18 backbone) for STFT spectrogram-based fault classification.

Input: (batch, n_channels, H, W) — STFT spectrograms per signal channel.
The model treats each spectrogram channel like an image channel.

For multi-channel signals, all channel spectrograms are stacked along
the channel dimension of the ResNet input.

Architecture:
  - Custom stem for arbitrary input channels (not fixed to 3-channel RGB)
  - 4 ResNet stages with BasicBlocks
  - Global Average Pooling → Dropout → Linear classifier
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock2D(nn.Module):
    """Standard ResNet BasicBlock for 2-D inputs."""

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class SpectrogramCNN(nn.Module):
    """ResNet-18 style 2-D CNN for spectrogram-based fault detection.

    Parameters
    ----------
    in_channels : int
        Number of spectrogram channels (= number of signal channels).
        For 6-channel power signals this is 6.
    n_classes : int
        Number of fault classes.
    dropout : float
        Dropout before the final linear layer.

    Examples
    --------
    >>> model = SpectrogramCNN(in_channels=6, n_classes=9)
    >>> x = torch.randn(16, 6, 128, 128)   # batch of spectrograms
    >>> logits = model(x)                   # (16, 9)
    """

    _LAYER_CONFIGS = [
        (64,  64,  2, 1),
        (64,  128, 2, 2),
        (128, 256, 2, 2),
        (256, 512, 2, 2),
    ]

    def __init__(
        self,
        in_channels: int = 6,
        n_classes: int = 9,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        # Flexible stem (not hard-coded to 3 channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, 64, n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, H, W)

        Returns
        -------
        logits : torch.Tensor, shape (B, n_classes)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.classifier(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 512-D embeddings before the classifier."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).squeeze(-1).squeeze(-1)

    @staticmethod
    def _make_layer(
        in_planes: int, out_planes: int, n_blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [BasicBlock2D(in_planes, out_planes, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock2D(out_planes, out_planes, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
