"""
1-D Convolutional Autoencoder for unsupervised anomaly detection.

Operational principle:
  - Train ONLY on healthy (normal) power electronics signals.
  - At inference, the reconstruction error is used as an anomaly score.
  - A faulty signal will have high reconstruction error because the
    autoencoder has never seen fault patterns during training.

Architecture:
  Encoder: Conv1d × 4 with strided downsampling → bottleneck
  Decoder: ConvTranspose1d × 4 with upsampling → reconstructed signal

Anomaly scoring:
  - Per-sample MSE reconstruction error
  - Optional threshold-based binary classification (healthy / fault)
  - Channel-wise anomaly maps for fault localization
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int = 9, stride: int = 2
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int = 9, stride: int = 2
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        output_padding = stride - 1
        self.block = nn.Sequential(
            nn.ConvTranspose1d(
                in_ch, out_ch, kernel_size,
                stride=stride, padding=padding, output_padding=output_padding, bias=False
            ),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AutoencoderAnomalyDetector(nn.Module):
    """Convolutional Autoencoder for unsupervised power electronics fault detection.

    Parameters
    ----------
    n_channels : int
        Number of input signal channels.
    window_size : int
        Input signal length.
    latent_channels : int
        Number of channels in the bottleneck representation.
    base_filters : int
        Number of filters in first encoder layer (doubled each stage).
    anomaly_threshold : float or None
        Reconstruction MSE threshold for binary anomaly classification.
        Set via ``set_threshold()`` on a validation set.

    Examples
    --------
    >>> model = AutoencoderAnomalyDetector(n_channels=6)
    >>> x = torch.randn(32, 6, 1024)
    >>> x_hat = model(x)            # reconstruction
    >>> scores = model.anomaly_score(x)   # per-sample MSE
    """

    def __init__(
        self,
        n_channels: int = 6,
        window_size: int = 1024,
        latent_channels: int = 32,
        base_filters: int = 64,
        anomaly_threshold: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold

        f = base_filters

        # Encoder: 4 stages, each halves temporal dimension
        self.encoder = nn.Sequential(
            EncoderBlock(n_channels, f),       # T → T/2
            EncoderBlock(f, f * 2),            # T/2 → T/4
            EncoderBlock(f * 2, f * 4),        # T/4 → T/8
            EncoderBlock(f * 4, latent_channels),  # T/8 → T/16
        )

        # Decoder: 4 stages, each doubles temporal dimension
        self.decoder = nn.Sequential(
            DecoderBlock(latent_channels, f * 4),
            DecoderBlock(f * 4, f * 2),
            DecoderBlock(f * 2, f),
            nn.ConvTranspose1d(
                f, n_channels, kernel_size=9, stride=2, padding=4, output_padding=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and decode to produce reconstruction.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, T)

        Returns
        -------
        x_hat : torch.Tensor, shape (B, C, T)  — reconstructed signal
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)

        # Match output length to input (in case of rounding mismatches)
        if x_hat.shape[-1] != x.shape[-1]:
            x_hat = F.interpolate(x_hat, size=x.shape[-1], mode="linear", align_corners=False)

        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return bottleneck latent representation."""
        return self.encoder(x)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction MSE.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, T)

        Returns
        -------
        scores : torch.Tensor, shape (B,)
        """
        with torch.no_grad():
            x_hat = self(x)
        mse = F.mse_loss(x_hat, x, reduction="none")
        return mse.mean(dim=(-2, -1))  # mean over channels and time → (B,)

    def channel_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel reconstruction error for fault localization.

        Returns
        -------
        maps : torch.Tensor, shape (B, C)
        """
        with torch.no_grad():
            x_hat = self(x)
        mse = F.mse_loss(x_hat, x, reduction="none")
        return mse.mean(dim=-1)  # mean over time → (B, C)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Binary anomaly prediction using the stored threshold.

        Returns
        -------
        labels : torch.Tensor, shape (B,)  — 0=healthy, 1=fault
        """
        if self.anomaly_threshold is None:
            raise RuntimeError(
                "anomaly_threshold is not set. Call set_threshold() first."
            )
        scores = self.anomaly_score(x)
        return (scores > self.anomaly_threshold).long()

    def set_threshold(
        self,
        healthy_signals: torch.Tensor,
        percentile: float = 95.0,
    ) -> float:
        """Calibrate anomaly threshold on a set of healthy training signals.

        The threshold is set at the ``percentile``-th percentile of
        reconstruction errors on healthy signals.

        Parameters
        ----------
        healthy_signals : torch.Tensor, shape (N, C, T)
        percentile : float
            Percentile of healthy reconstruction errors to use as threshold.

        Returns
        -------
        threshold : float
        """
        scores = self.anomaly_score(healthy_signals)
        threshold = float(torch.quantile(scores, percentile / 100.0))
        self.anomaly_threshold = threshold
        return threshold

    def reconstruction_loss(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """MSE reconstruction loss for training."""
        x_hat = self(x)
        return F.mse_loss(x_hat, x, reduction=reduction)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
