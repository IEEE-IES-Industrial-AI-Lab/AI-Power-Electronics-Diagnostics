"""
Harmonic analysis for power electronics signals.

Provides:
  - Total Harmonic Distortion (THD-F and THD-R)
  - Individual harmonic amplitude and phase extraction
  - Inter-harmonic detection
  - Sequence component analysis (positive / negative / zero)
  - Harmonic feature vector for ML models

IEEE 519-2022 defines THD limits for power quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class HarmonicAnalysisResult:
    """Container for harmonic analysis results."""
    fundamental_freq: float                  # Detected fundamental [Hz]
    fundamental_amplitude: float             # Fundamental amplitude
    harmonics: Dict[int, float]              # {order: amplitude}
    harmonic_phases: Dict[int, float]        # {order: phase [rad]}
    thd_f: float                             # THD-F [%] — relative to fundamental
    thd_r: float                             # THD-R [%] — relative to total RMS
    total_rms: float                         # Total signal RMS
    individual_hdrs: Dict[int, float]        # Individual harmonic distortion ratios [%]
    inter_harmonics: List[float]             # Detected inter-harmonic frequencies [Hz]


class HarmonicAnalyzer:
    """Analyze harmonic content of power electronics voltage/current signals.

    Parameters
    ----------
    f_sample : float
        Sampling frequency [Hz].
    f_fund_nominal : float
        Nominal fundamental frequency [Hz] (50 or 60 Hz).
    n_harmonics : int
        Maximum harmonic order to analyze.
    freq_tolerance : float
        Frequency bin search window as fraction of bin width.

    Examples
    --------
    >>> analyzer = HarmonicAnalyzer(f_sample=50_000.0, f_fund_nominal=50.0)
    >>> result = analyzer.analyze(voltage_signal)
    >>> print(f"THD-F: {result.thd_f:.2f}%")
    """

    def __init__(
        self,
        f_sample: float = 50_000.0,
        f_fund_nominal: float = 50.0,
        n_harmonics: int = 20,
        freq_tolerance: float = 0.1,
    ) -> None:
        self.f_sample = f_sample
        self.f_fund_nominal = f_fund_nominal
        self.n_harmonics = n_harmonics
        self.freq_tolerance = freq_tolerance

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(self, signal: np.ndarray) -> HarmonicAnalysisResult:
        """Full harmonic analysis of a 1-D signal.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)

        Returns
        -------
        HarmonicAnalysisResult
        """
        n = len(signal)
        win = np.hanning(n)
        spectrum = np.fft.rfft(signal * win) / (win.sum() / 2)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.f_sample)

        # Locate fundamental (search around nominal ±10%)
        fund_freq, fund_amp, fund_phase = self._find_fundamental(spectrum, freqs)

        harmonics: Dict[int, float] = {}
        harmonic_phases: Dict[int, float] = {}
        harmonic_power = 0.0

        for h in range(2, self.n_harmonics + 1):
            amp, phase = self._find_harmonic(spectrum, freqs, fund_freq, h)
            harmonics[h] = amp
            harmonic_phases[h] = phase
            harmonic_power += amp ** 2

        harmonics[1] = fund_amp
        harmonic_phases[1] = fund_phase

        # THD-F: relative to fundamental
        thd_f = 100.0 * np.sqrt(harmonic_power) / (fund_amp + 1e-12)

        # Total RMS (all components)
        total_rms = float(np.sqrt(np.mean(signal ** 2)))

        # THD-R: relative to total RMS
        thd_r = 100.0 * np.sqrt(harmonic_power) / (total_rms + 1e-12)

        # Individual HDR per harmonic
        individual_hdrs = {
            h: 100.0 * amp / (fund_amp + 1e-12) for h, amp in harmonics.items() if h > 1
        }

        # Inter-harmonic detection
        inter_harmonics = self._detect_inter_harmonics(spectrum, freqs, fund_freq)

        return HarmonicAnalysisResult(
            fundamental_freq=fund_freq,
            fundamental_amplitude=fund_amp,
            harmonics=harmonics,
            harmonic_phases=harmonic_phases,
            thd_f=float(thd_f),
            thd_r=float(thd_r),
            total_rms=total_rms,
            individual_hdrs=individual_hdrs,
            inter_harmonics=inter_harmonics,
        )

    def extract_features(
        self,
        signal: np.ndarray,
        n_harmonics: Optional[int] = None,
    ) -> np.ndarray:
        """Compact harmonic feature vector for ML models.

        Features:
        - Fundamental amplitude
        - THD-F, THD-R
        - Amplitude of harmonics 2 … n_harmonics
        - Phase of harmonics 1 … n_harmonics (cosine encoded)
        - Total RMS
        - Crest factor
        - Number of significant inter-harmonics

        Returns
        -------
        features : np.ndarray
        """
        n_h = n_harmonics or self.n_harmonics
        result = self.analyze(signal)

        features = [
            result.fundamental_amplitude,
            result.thd_f,
            result.thd_r,
            result.total_rms,
            float(np.max(np.abs(signal))) / (result.total_rms + 1e-12),  # crest factor
            float(len(result.inter_harmonics)),
        ]

        for h in range(2, n_h + 1):
            features.append(result.harmonics.get(h, 0.0))

        # Encode phase via cosine / sine to avoid phase wrapping
        for h in range(1, n_h + 1):
            phase = result.harmonic_phases.get(h, 0.0)
            features.append(float(np.cos(phase)))
            features.append(float(np.sin(phase)))

        return np.array(features, dtype=np.float32)

    def three_phase_sequence(
        self,
        ia: np.ndarray,
        ib: np.ndarray,
        ic: np.ndarray,
    ) -> Dict[str, float]:
        """Compute symmetrical sequence components (positive, negative, zero).

        Returns the amplitude of each sequence component at the fundamental.
        Negative-sequence component is a key indicator of phase imbalance
        and inter-turn short circuits.

        Returns
        -------
        dict with keys 'positive', 'negative', 'zero' (amplitudes)
        """
        n = min(len(ia), len(ib), len(ic))
        win = np.hanning(n)

        Sa = np.fft.rfft(ia[:n] * win) / (win.sum() / 2)
        Sb = np.fft.rfft(ib[:n] * win) / (win.sum() / 2)
        Sc = np.fft.rfft(ic[:n] * win) / (win.sum() / 2)

        freqs = np.fft.rfftfreq(n, d=1.0 / self.f_sample)
        fund_idx = int(np.argmin(np.abs(freqs - self.f_fund_nominal)))

        a = complex(np.exp(1j * 2 * np.pi / 3))
        a2 = a ** 2

        I_pos = (Sa[fund_idx] + a * Sb[fund_idx] + a2 * Sc[fund_idx]) / 3
        I_neg = (Sa[fund_idx] + a2 * Sb[fund_idx] + a * Sc[fund_idx]) / 3
        I_zero = (Sa[fund_idx] + Sb[fund_idx] + Sc[fund_idx]) / 3

        return {
            "positive": float(np.abs(I_pos)),
            "negative": float(np.abs(I_neg)),
            "zero": float(np.abs(I_zero)),
            "unbalance_factor": float(np.abs(I_neg) / (np.abs(I_pos) + 1e-12)),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_fundamental(
        self, spectrum: np.ndarray, freqs: np.ndarray
    ) -> Tuple[float, float, float]:
        """Locate fundamental frequency near f_fund_nominal."""
        search_low = self.f_fund_nominal * (1 - 0.1)
        search_high = self.f_fund_nominal * (1 + 0.1)
        mask = (freqs >= search_low) & (freqs <= search_high)
        sub_amp = np.abs(spectrum[mask])
        sub_phase = np.angle(spectrum[mask])
        sub_freqs = freqs[mask]

        if len(sub_amp) == 0:
            return self.f_fund_nominal, 0.0, 0.0

        peak_idx = int(np.argmax(sub_amp))
        return float(sub_freqs[peak_idx]), float(sub_amp[peak_idx]), float(sub_phase[peak_idx])

    def _find_harmonic(
        self,
        spectrum: np.ndarray,
        freqs: np.ndarray,
        f_fund: float,
        order: int,
    ) -> Tuple[float, float]:
        """Find the amplitude and phase of the h-th harmonic."""
        f_target = order * f_fund
        bin_width = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        tolerance = max(1, int(self.freq_tolerance * f_target / bin_width))
        idx = int(np.argmin(np.abs(freqs - f_target)))
        search_range = slice(max(0, idx - tolerance), min(len(spectrum), idx + tolerance + 1))
        sub = np.abs(spectrum[search_range])
        sub_phase = np.angle(spectrum[search_range])
        if len(sub) == 0:
            return 0.0, 0.0
        peak = int(np.argmax(sub))
        return float(sub[peak]), float(sub_phase[peak])

    def _detect_inter_harmonics(
        self,
        spectrum: np.ndarray,
        freqs: np.ndarray,
        f_fund: float,
        threshold_frac: float = 0.02,
    ) -> List[float]:
        """Detect significant inter-harmonic components.

        Inter-harmonics are spectral peaks that fall between integer multiples
        of the fundamental and exceed ``threshold_frac`` of fundamental amplitude.
        """
        fund_idx = int(np.argmin(np.abs(freqs - f_fund)))
        fund_amp = float(np.abs(spectrum[fund_idx]))
        threshold = threshold_frac * fund_amp

        # Build set of harmonic bin indices
        harmonic_bins = set()
        for h in range(1, self.n_harmonics + 2):
            h_freq = h * f_fund
            h_idx = int(np.argmin(np.abs(freqs - h_freq)))
            for offset in range(-3, 4):
                harmonic_bins.add(max(0, min(len(freqs) - 1, h_idx + offset)))

        inter_harmonics = []
        amp = np.abs(spectrum)
        for i, (f, a) in enumerate(zip(freqs, amp)):
            if f < f_fund * 1.5:
                continue
            if i not in harmonic_bins and a > threshold:
                inter_harmonics.append(float(f))

        return inter_harmonics
