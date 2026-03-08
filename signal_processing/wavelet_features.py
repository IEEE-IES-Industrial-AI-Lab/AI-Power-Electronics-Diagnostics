"""
Wavelet-based feature extraction for power electronics fault signals.

Implements:
  - Discrete Wavelet Transform (DWT) multi-resolution analysis
  - Continuous Wavelet Transform (CWT) scalogram
  - DWT energy features per sub-band
  - Wavelet packet decomposition (WPD) energy tree

Wavelet analysis is particularly effective for detecting transient faults
(switching events, impulse spikes) that are non-stationary in nature.

References:
  Cusidó, J., et al. (2008). Fault detection in induction motors using
  power spectral density in wavelet decomposition. Electric Power Systems
  Research, 78(7), 1262–1270.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pywt


@dataclass
class DWTResult:
    """Container for DWT multi-resolution analysis."""
    coefficients: List[np.ndarray]   # [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    level: int
    wavelet: str
    sub_band_energies: np.ndarray    # shape (level + 1,) — energy per sub-band
    sub_band_entropy: np.ndarray     # Shannon entropy per sub-band
    relative_energies: np.ndarray   # Normalized energies


class WaveletFeatureExtractor:
    """Extract wavelet-domain features from power electronics signals.

    Parameters
    ----------
    wavelet : str
        Wavelet family. Common choices for electrical signals:
        'db4' (Daubechies-4), 'sym5' (Symlet-5), 'coif3' (Coiflet-3).
    level : int or None
        Decomposition level. Defaults to ``pywt.dwt_max_level(len, wavelet)``.
    mode : str
        Signal extension mode (boundary handling). Default: 'periodization'.

    Examples
    --------
    >>> extractor = WaveletFeatureExtractor(wavelet='db4', level=5)
    >>> features = extractor.extract_dwt_features(signal)
    """

    def __init__(
        self,
        wavelet: str = "db4",
        level: Optional[int] = None,
        mode: str = "periodization",
    ) -> None:
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self._wavelet_obj = pywt.Wavelet(wavelet)

    # ------------------------------------------------------------------
    # DWT analysis
    # ------------------------------------------------------------------

    def dwt_decompose(self, signal: np.ndarray) -> DWTResult:
        """Multi-level DWT decomposition.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)

        Returns
        -------
        DWTResult
        """
        max_level = pywt.dwt_max_level(len(signal), self.wavelet)
        level = min(self.level or max_level, max_level)

        coeffs = pywt.wavedec(signal, self.wavelet, mode=self.mode, level=level)
        # coeffs = [cA_level, cD_level, cD_{level-1}, ..., cD_1]

        energies = np.array([np.sum(c ** 2) for c in coeffs], dtype=np.float32)
        total_energy = energies.sum() + 1e-12
        relative_energies = energies / total_energy

        entropy = np.array(
            [self._shannon_entropy(c) for c in coeffs], dtype=np.float32
        )

        return DWTResult(
            coefficients=coeffs,
            level=level,
            wavelet=self.wavelet,
            sub_band_energies=energies,
            sub_band_entropy=entropy,
            relative_energies=relative_energies,
        )

    def extract_dwt_features(
        self,
        signal: np.ndarray,
        include_stats: bool = True,
    ) -> np.ndarray:
        """Extract a fixed-length DWT feature vector.

        Per sub-band features:
        - Energy, relative energy, Shannon entropy
        - If ``include_stats``: mean absolute value, std, kurtosis, skewness

        Parameters
        ----------
        include_stats : bool
            Include per-sub-band statistical moments.

        Returns
        -------
        features : np.ndarray, shape (n_features,)
        """
        result = self.dwt_decompose(signal)
        features = []

        for i, coeffs in enumerate(result.coefficients):
            features.append(float(result.sub_band_energies[i]))
            features.append(float(result.relative_energies[i]))
            features.append(float(result.sub_band_entropy[i]))

            if include_stats:
                features.append(float(np.mean(np.abs(coeffs))))
                features.append(float(np.std(coeffs)))
                features.append(float(self._kurtosis(coeffs)))
                features.append(float(self._skewness(coeffs)))

        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------
    # CWT scalogram
    # ------------------------------------------------------------------

    def cwt_scalogram(
        self,
        signal: np.ndarray,
        scales: Optional[np.ndarray] = None,
        wavelet: str = "cmor1.5-1.0",
        output_size: Optional[Tuple[int, int]] = (64, 64),
    ) -> np.ndarray:
        """Compute CWT scalogram (time-scale representation).

        Parameters
        ----------
        scales : np.ndarray or None
            CWT scales. Defaults to np.arange(1, 65) (64 scales).
        wavelet : str
            Complex wavelet for CWT. 'cmor1.5-1.0' (complex Morlet) or
            'mexh' (Mexican hat) are typical choices.
        output_size : tuple(int, int) or None
            Resize scalogram to (scales, time) shape.

        Returns
        -------
        scalogram : np.ndarray, shape output_size or (len(scales), len(signal))
        """
        if scales is None:
            scales = np.arange(1, 65)

        coefficients, _ = pywt.cwt(signal, scales, wavelet)
        scalogram = np.abs(coefficients).astype(np.float32)

        if output_size is not None:
            from scipy.ndimage import zoom
            zoom_r = output_size[0] / scalogram.shape[0]
            zoom_c = output_size[1] / scalogram.shape[1]
            scalogram = zoom(scalogram, (zoom_r, zoom_c), order=1)

        return scalogram

    # ------------------------------------------------------------------
    # Wavelet Packet Decomposition
    # ------------------------------------------------------------------

    def wpd_energy_features(
        self, signal: np.ndarray, level: int = 4
    ) -> np.ndarray:
        """Wavelet Packet Decomposition (WPD) energy at all leaf nodes.

        WPD decomposes both approximation AND detail at every level,
        providing finer frequency resolution than standard DWT.

        Parameters
        ----------
        level : int
            Decomposition level. Produces 2^level leaf nodes.

        Returns
        -------
        features : np.ndarray, shape (2^level,)  — energy per leaf node
        """
        wp = pywt.WaveletPacket(signal, wavelet=self.wavelet, mode=self.mode, maxlevel=level)
        nodes = [node.path for node in wp.get_level(level, "natural")]
        energies = np.array(
            [np.sum(wp[node].data ** 2) for node in nodes], dtype=np.float32
        )
        total = energies.sum() + 1e-12
        return energies / total  # return relative energies

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def extract_batch(self, signals: np.ndarray) -> np.ndarray:
        """Extract DWT features for a batch of single-channel signals.

        Parameters
        ----------
        signals : np.ndarray, shape (N, T)

        Returns
        -------
        features : np.ndarray, shape (N, n_features)
        """
        return np.stack([self.extract_dwt_features(s) for s in signals])

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shannon_entropy(coefficients: np.ndarray) -> float:
        """Shannon entropy of wavelet coefficients."""
        p = coefficients ** 2
        total = p.sum()
        if total < 1e-12:
            return 0.0
        p = p / total
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """Excess kurtosis."""
        mu = x.mean()
        sigma = x.std()
        if sigma < 1e-12:
            return 0.0
        return float(((x - mu) ** 4).mean() / sigma ** 4 - 3)

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        """Skewness."""
        mu = x.mean()
        sigma = x.std()
        if sigma < 1e-12:
            return 0.0
        return float(((x - mu) ** 3).mean() / sigma ** 3)

    @staticmethod
    def available_wavelets() -> Dict[str, List[str]]:
        """Return available wavelet families from PyWavelets."""
        return {family: pywt.wavelist(family) for family in pywt.families()}
