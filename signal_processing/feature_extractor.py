"""
Unified signal feature extraction pipeline.

``SignalFeatureExtractor`` is the single entry point for the training
pipeline.  It combines FFT, wavelet, and harmonic features into one
consistent interface, regardless of signal source (synthetic or real).

Output modes:
  - 'raw'         : return raw windowed signal (C, T) — for CNN/LSTM/Transformer
  - 'spectrogram' : return STFT spectrogram (C, H, W) — for 2-D CNN
  - 'features'    : return 1-D hand-crafted feature vector — for baselines
  - 'scalogram'   : return CWT scalogram (C, H, W) — for wavelets CNN
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np

from signal_processing.fft_analysis import FFTAnalyzer
from signal_processing.stft_spectrogram import STFTSpectrogram
from signal_processing.wavelet_features import WaveletFeatureExtractor
from signal_processing.harmonic_analysis import HarmonicAnalyzer

OutputMode = Literal["raw", "spectrogram", "features", "scalogram"]


class SignalFeatureExtractor:
    """Unified feature extraction pipeline for power electronics signals.

    Parameters
    ----------
    f_sample : float
        Sampling frequency [Hz].
    f_fund : float
        Fundamental frequency [Hz].
    output_mode : str
        One of 'raw', 'spectrogram', 'features', 'scalogram'.
    spectrogram_size : tuple(int, int)
        Output size for spectrogram / scalogram. Only used for 2-D output modes.
    wavelet : str
        Wavelet name for DWT / CWT features.
    normalize : bool
        Z-score normalize the output.

    Examples
    --------
    >>> extractor = SignalFeatureExtractor(f_sample=50_000, output_mode='spectrogram')
    >>> X_spec = extractor.transform_batch(X_raw)  # (N, C, 128, 128)

    >>> extractor = SignalFeatureExtractor(f_sample=50_000, output_mode='features')
    >>> X_feat = extractor.transform_batch(X_raw)  # (N, n_features)
    """

    def __init__(
        self,
        f_sample: float = 50_000.0,
        f_fund: float = 50.0,
        output_mode: OutputMode = "raw",
        spectrogram_size: Tuple[int, int] = (128, 128),
        wavelet: str = "db4",
        normalize: bool = True,
    ) -> None:
        self.f_sample = f_sample
        self.f_fund = f_fund
        self.output_mode = output_mode
        self.spectrogram_size = spectrogram_size
        self.normalize = normalize

        # Sub-analyzers (lazily constructed)
        self._fft = FFTAnalyzer(f_sample=f_sample)
        self._stft = STFTSpectrogram(f_sample=f_sample, output_size=spectrogram_size)
        self._wavelet = WaveletFeatureExtractor(wavelet=wavelet)
        self._harmonic = HarmonicAnalyzer(f_sample=f_sample, f_fund_nominal=f_fund)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Transform a single multi-channel signal.

        Parameters
        ----------
        signal : np.ndarray, shape (C, T)

        Returns
        -------
        output : np.ndarray
            Shape depends on ``output_mode``:
            - 'raw'         : (C, T)
            - 'spectrogram' : (C, H, W)
            - 'scalogram'   : (C, H, W)
            - 'features'    : (n_features,)
        """
        if self.output_mode == "raw":
            out = signal.astype(np.float32)
            if self.normalize:
                out = self._znorm(out)
            return out

        elif self.output_mode == "spectrogram":
            spec = self._stft.compute_multichannel(signal)
            if self.normalize:
                spec = self._znorm_2d(spec)
            return spec

        elif self.output_mode == "scalogram":
            scales = np.arange(1, self.spectrogram_size[0] + 1)
            channels = []
            for c in range(signal.shape[0]):
                sc = self._wavelet.cwt_scalogram(
                    signal[c], scales=scales, output_size=self.spectrogram_size
                )
                channels.append(sc)
            out = np.stack(channels).astype(np.float32)
            if self.normalize:
                out = self._znorm_2d(out)
            return out

        elif self.output_mode == "features":
            return self._extract_hand_features(signal)

        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode!r}")

    def transform_batch(self, signals: np.ndarray) -> np.ndarray:
        """Transform a batch of signals.

        Parameters
        ----------
        signals : np.ndarray, shape (N, C, T)

        Returns
        -------
        np.ndarray
        """
        return np.stack([self.transform(s) for s in signals])

    def feature_names(self, n_channels: int = 6) -> list:
        """Return descriptive feature names for the 'features' output mode."""
        names = []
        for c in range(n_channels):
            prefix = f"ch{c}"
            # FFT features
            for i in range(8):
                names.append(f"{prefix}_fft_band{i}_energy")
            names.append(f"{prefix}_fft_centroid")
            names.append(f"{prefix}_fft_spread")
            names.append(f"{prefix}_fft_rolloff")
            for h in range(1, 6):
                names.append(f"{prefix}_harmonic_{h}_amp")
            names.append(f"{prefix}_thd")
            # Wavelet features (level=5, 3 features per band)
            for lvl in range(6):
                names.append(f"{prefix}_dwt_energy_l{lvl}")
                names.append(f"{prefix}_dwt_rel_energy_l{lvl}")
                names.append(f"{prefix}_dwt_entropy_l{lvl}")
                for stat in ["mae", "std", "kurt", "skew"]:
                    names.append(f"{prefix}_dwt_{stat}_l{lvl}")
        return names

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_hand_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract combined FFT + wavelet + harmonic feature vector."""
        channel_features = []
        for c in range(signal.shape[0]):
            ch = signal[c]
            fft_feats = self._fft.extract_features(
                ch, n_bands=8, f_fund=self.f_fund, n_harmonics=5
            )
            dwt_feats = self._wavelet.extract_dwt_features(ch, include_stats=True)
            channel_features.append(np.concatenate([fft_feats, dwt_feats]))

        return np.concatenate(channel_features).astype(np.float32)

    @staticmethod
    def _znorm(x: np.ndarray) -> np.ndarray:
        """Z-score normalize along last axis (per channel)."""
        mu = x.mean(axis=-1, keepdims=True)
        sigma = x.std(axis=-1, keepdims=True) + 1e-8
        return (x - mu) / sigma

    @staticmethod
    def _znorm_2d(x: np.ndarray) -> np.ndarray:
        """Z-score normalize each (H, W) image independently."""
        mu = x.mean(axis=(-2, -1), keepdims=True)
        sigma = x.std(axis=(-2, -1), keepdims=True) + 1e-8
        return (x - mu) / sigma

    def __repr__(self) -> str:
        return (
            f"SignalFeatureExtractor("
            f"mode={self.output_mode!r}, "
            f"f_sample={self.f_sample}, "
            f"wavelet={self._wavelet.wavelet!r})"
        )
