"""
FFT-based spectral analysis for power electronics signals.

Provides:
  - Single-sided amplitude and power spectrum
  - Power spectral density (PSD) via Welch's method
  - Dominant frequency identification
  - Spectral feature vector extraction (for ML input)

All methods accept 1-D numpy arrays (single channel).  For multi-channel
signals iterate over channels or use ``SignalFeatureExtractor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import welch


@dataclass
class FFTResult:
    """Container for FFT analysis outputs."""
    frequencies: np.ndarray      # Frequency axis [Hz]
    amplitude_spectrum: np.ndarray  # Single-sided amplitude [same unit as signal]
    power_spectrum: np.ndarray   # Single-sided power (amplitude²)
    dominant_freq: float         # Frequency with highest amplitude [Hz]
    dominant_amp: float          # Amplitude at dominant frequency
    thd: float                   # Total Harmonic Distortion [%]


class FFTAnalyzer:
    """Compute FFT-based spectral features for power electronics signals.

    Parameters
    ----------
    f_sample : float
        Sampling frequency [Hz].
    n_fft : int or None
        FFT size. Defaults to signal length (no zero-padding).
    window : str
        Scipy/numpy window function name applied before FFT.
        Common choices: 'hann', 'hamming', 'blackman', 'boxcar'.

    Examples
    --------
    >>> analyzer = FFTAnalyzer(f_sample=50_000.0)
    >>> result = analyzer.compute(signal)
    >>> print(f"Dominant freq: {result.dominant_freq:.1f} Hz")
    """

    def __init__(
        self,
        f_sample: float = 50_000.0,
        n_fft: Optional[int] = None,
        window: str = "hann",
    ) -> None:
        self.f_sample = f_sample
        self.n_fft = n_fft
        self.window = window

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self, signal: np.ndarray) -> FFTResult:
        """Compute the single-sided amplitude spectrum.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Time-domain signal.

        Returns
        -------
        FFTResult
        """
        n = len(signal)
        n_fft = self.n_fft or n

        # Apply window to reduce spectral leakage
        win = self._make_window(n)
        windowed = signal * win

        # Real FFT — returns only positive frequencies
        spectrum = np.fft.rfft(windowed, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.f_sample)

        # Single-sided amplitude spectrum (compensate for window scaling)
        amp = np.abs(spectrum) / (win.sum() / 2)
        amp[0] /= 2       # DC component
        if n_fft % 2 == 0:
            amp[-1] /= 2  # Nyquist component

        power = amp ** 2
        dom_idx = int(np.argmax(amp[1:]) + 1)  # skip DC

        thd = self._compute_thd(amp, freqs)

        return FFTResult(
            frequencies=freqs,
            amplitude_spectrum=amp,
            power_spectrum=power,
            dominant_freq=float(freqs[dom_idx]),
            dominant_amp=float(amp[dom_idx]),
            thd=thd,
        )

    def psd_welch(
        self,
        signal: np.ndarray,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Power Spectral Density using Welch's method.

        Parameters
        ----------
        nperseg : int
            Length of each segment.
        noverlap : int or None
            Overlap between segments. Defaults to nperseg // 2.

        Returns
        -------
        freqs : np.ndarray
        psd   : np.ndarray
        """
        freqs, psd = welch(
            signal,
            fs=self.f_sample,
            nperseg=nperseg,
            noverlap=noverlap,
            window=self.window,
        )
        return freqs, psd

    def extract_features(
        self,
        signal: np.ndarray,
        n_bands: int = 8,
        f_fund: Optional[float] = None,
        n_harmonics: int = 5,
    ) -> np.ndarray:
        """Extract a fixed-length spectral feature vector.

        Features included:
        - Band energy for ``n_bands`` equal-width frequency bands
        - Spectral centroid
        - Spectral spread (variance)
        - Spectral rolloff (95 %)
        - Fundamental amplitude + N harmonic amplitudes (if f_fund given)
        - THD

        Parameters
        ----------
        n_bands : int
            Number of frequency bands for band energy.
        f_fund : float or None
            Fundamental frequency. If provided, harmonic amplitudes are added.
        n_harmonics : int
            Number of harmonics to extract (only used when f_fund is given).

        Returns
        -------
        features : np.ndarray, shape (n_features,)
        """
        result = self.compute(signal)
        amp = result.amplitude_spectrum
        freqs = result.frequencies

        features = []

        # Band energies
        band_edges = np.linspace(freqs[0], freqs[-1], n_bands + 1)
        for i in range(n_bands):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            features.append(float(amp[mask].sum()))

        # Spectral moments
        total_power = (amp ** 2).sum() + 1e-12
        centroid = float((freqs * amp ** 2).sum() / total_power)
        spread = float(np.sqrt(((freqs - centroid) ** 2 * amp ** 2).sum() / total_power))
        features.extend([centroid, spread])

        # Spectral rolloff (frequency below which 95% of energy is contained)
        cumulative = np.cumsum(amp ** 2)
        rolloff_threshold = 0.95 * cumulative[-1]
        rolloff_idx = int(np.searchsorted(cumulative, rolloff_threshold))
        features.append(float(freqs[min(rolloff_idx, len(freqs) - 1)]))

        # Harmonic amplitudes
        if f_fund is not None:
            for h in range(1, n_harmonics + 1):
                f_h = h * f_fund
                idx = int(np.argmin(np.abs(freqs - f_h)))
                features.append(float(amp[idx]))

        # THD
        features.append(result.thd)

        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_window(self, n: int) -> np.ndarray:
        window_funcs = {
            "hann": np.hanning,
            "hamming": np.hamming,
            "blackman": np.blackman,
            "boxcar": np.ones,
        }
        fn = window_funcs.get(self.window, np.hanning)
        return fn(n)

    @staticmethod
    def _compute_thd(amp: np.ndarray, freqs: np.ndarray) -> float:
        """Compute THD as ratio of harmonic power to fundamental power [%].

        Uses the first peak as fundamental; harmonics 2–10 are summed.
        """
        if len(amp) < 2:
            return 0.0

        # Find fundamental (highest amplitude, ignoring DC at index 0)
        fund_idx = int(np.argmax(amp[1:])) + 1
        fund_amp = amp[fund_idx]

        if fund_amp < 1e-12:
            return 0.0

        fund_freq = freqs[fund_idx]
        harmonic_power = 0.0
        for h in range(2, 11):
            h_freq = h * fund_freq
            h_idx = int(np.argmin(np.abs(freqs - h_freq)))
            harmonic_power += amp[h_idx] ** 2

        thd = 100.0 * np.sqrt(harmonic_power) / fund_amp
        return float(thd)

    def frequency_resolution(self) -> float:
        """Frequency bin width [Hz] for default FFT size."""
        return self.f_sample / (self.n_fft or 1024)
