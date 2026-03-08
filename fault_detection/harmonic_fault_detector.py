"""
Harmonic distortion fault detection pipeline.

Detects power quality degradation and motor faults via spectral analysis:
  - Excessive THD (IEEE 519-2022 limit violations)
  - Inter-turn short circuit (ITSC) via sideband detection
  - Inter-harmonic components (non-integer multiples of fundamental)
  - Voltage / current unbalance (negative sequence component)

This module operates primarily via rule-based thresholds on harmonic
features, with an optional ML model for multi-class classification.

IEEE 519-2022 voltage THD limits:
  - Vn ≤ 1 kV : 8%
  - 1 kV < Vn ≤ 69 kV : 5%
  - 69 kV < Vn ≤ 161 kV : 2.5%
  - Vn > 161 kV : 1.5%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from signal_processing.harmonic_analysis import HarmonicAnalyzer

logger = logging.getLogger(__name__)

# IEEE 519-2022 THD limits by voltage class
IEEE519_THD_LIMITS = {
    "LV":  8.0,   # ≤ 1 kV
    "MV":  5.0,   # 1–69 kV
    "HV":  2.5,   # 69–161 kV
    "EHV": 1.5,   # > 161 kV
}

# Typical individual harmonic distortion limits (% of fundamental)
IEEE519_IHD_LIMITS = {
    3: 5.0, 5: 4.0, 7: 4.0, 9: 1.5, 11: 2.0,
    13: 2.0, 15: 0.3, 17: 1.5, 19: 1.5, 21: 0.3,
    23: 0.7, 25: 0.7,
}


@dataclass
class HarmonicFaultResult:
    """Result of harmonic fault analysis."""
    thd_f: float
    thd_r: float
    ieee519_violated: bool
    violated_harmonics: List[int]         # harmonic orders exceeding IHD limits
    itsc_detected: bool
    itsc_sideband_amplitude: float
    inter_harmonics: List[float]
    unbalance_factor: float               # negative / positive sequence ratio
    fault_type: str                       # 'healthy' | 'high_thd' | 'itsc' | 'unbalance'
    severity: str                         # 'normal' | 'warning' | 'critical'
    feature_vector: np.ndarray


class HarmonicFaultDetector:
    """Detect harmonic-related power quality and motor faults.

    Parameters
    ----------
    f_sample : float
        Sampling frequency [Hz].
    f_fund : float
        Nominal fundamental frequency [Hz].
    voltage_class : str
        One of 'LV', 'MV', 'HV', 'EHV' — selects IEEE 519 THD limit.
    thd_warning_frac : float
        Issue a warning at ``thd_warning_frac`` × IEEE limit (e.g., 0.8 = 80%).
    itsc_sideband_threshold : float
        Relative amplitude of (1±2s)f sidebands to flag ITSC [%].
    slip : float
        Motor slip (used for ITSC sideband frequency calculation).

    Examples
    --------
    >>> detector = HarmonicFaultDetector(f_sample=50_000, f_fund=50.0)
    >>> result = detector.analyze(voltage_signal)
    >>> print(result.fault_type, result.thd_f)
    """

    def __init__(
        self,
        f_sample: float = 50_000.0,
        f_fund: float = 50.0,
        voltage_class: str = "LV",
        thd_warning_frac: float = 0.8,
        itsc_sideband_threshold: float = 2.0,
        slip: float = 0.03,
    ) -> None:
        self.f_sample = f_sample
        self.f_fund = f_fund
        self.thd_limit = IEEE519_THD_LIMITS[voltage_class]
        self.thd_warning = thd_warning_frac * self.thd_limit
        self.itsc_threshold = itsc_sideband_threshold
        self.slip = slip

        self._analyzer = HarmonicAnalyzer(
            f_sample=f_sample,
            f_fund_nominal=f_fund,
            n_harmonics=25,
        )

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        signal: np.ndarray,
        ia: Optional[np.ndarray] = None,
        ib: Optional[np.ndarray] = None,
        ic: Optional[np.ndarray] = None,
    ) -> HarmonicFaultResult:
        """Full harmonic fault analysis of a single-phase signal.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Primary signal to analyze (voltage or current phase A).
        ia, ib, ic : np.ndarray or None
            All three phase currents. If provided, sequence analysis is computed.

        Returns
        -------
        HarmonicFaultResult
        """
        harm_result = self._analyzer.analyze(signal)
        features = self._analyzer.extract_features(signal)

        # Check IEEE 519 limits
        violated_harmonics = []
        for order, limit in IEEE519_IHD_LIMITS.items():
            idr = harm_result.individual_hdrs.get(order, 0.0)
            if idr > limit:
                violated_harmonics.append(order)

        ieee519_violated = (
            harm_result.thd_f > self.thd_limit or len(violated_harmonics) > 0
        )

        # ITSC detection via sideband check
        itsc_detected, itsc_amp = self._check_itsc(signal)

        # Sequence unbalance
        unbalance_factor = 0.0
        if ia is not None and ib is not None and ic is not None:
            seq = self._analyzer.three_phase_sequence(ia, ib, ic)
            unbalance_factor = seq["unbalance_factor"]

        # Determine fault type and severity
        fault_type, severity = self._classify(
            harm_result.thd_f,
            ieee519_violated,
            itsc_detected,
            unbalance_factor,
        )

        return HarmonicFaultResult(
            thd_f=harm_result.thd_f,
            thd_r=harm_result.thd_r,
            ieee519_violated=ieee519_violated,
            violated_harmonics=violated_harmonics,
            itsc_detected=itsc_detected,
            itsc_sideband_amplitude=itsc_amp,
            inter_harmonics=harm_result.inter_harmonics,
            unbalance_factor=unbalance_factor,
            fault_type=fault_type,
            severity=severity,
            feature_vector=features,
        )

    def analyze_three_phase(
        self,
        ia: np.ndarray,
        ib: np.ndarray,
        ic: np.ndarray,
    ) -> Dict[str, HarmonicFaultResult]:
        """Analyze all three phases and return per-phase results."""
        return {
            "ia": self.analyze(ia, ia, ib, ic),
            "ib": self.analyze(ib, ia, ib, ic),
            "ic": self.analyze(ic, ia, ib, ic),
        }

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_itsc(self, signal: np.ndarray) -> Tuple[bool, float]:
        """Check for ITSC via (1 ± 2s)·f sideband detection."""
        n = len(signal)
        win = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(signal * win) / (win.sum() / 2))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.f_sample)

        fund_idx = int(np.argmin(np.abs(freqs - self.f_fund)))
        fund_amp = float(spectrum[fund_idx])

        if fund_amp < 1e-9:
            return False, 0.0

        sideband_freqs = [
            self.f_fund * (1 - 2 * self.slip),
            self.f_fund * (1 + 2 * self.slip),
        ]

        max_sideband_pct = 0.0
        for f_sb in sideband_freqs:
            idx = int(np.argmin(np.abs(freqs - f_sb)))
            sideband_pct = 100.0 * spectrum[idx] / fund_amp
            max_sideband_pct = max(max_sideband_pct, sideband_pct)

        itsc_detected = max_sideband_pct > self.itsc_threshold
        return itsc_detected, float(max_sideband_pct)

    def _classify(
        self,
        thd_f: float,
        ieee_violated: bool,
        itsc: bool,
        unbalance: float,
    ) -> Tuple[str, str]:
        """Simple rule-based fault classification."""
        if itsc:
            severity = "critical" if itsc else "warning"
            return "itsc", severity

        if unbalance > 0.05:
            severity = "critical" if unbalance > 0.10 else "warning"
            return "unbalance", severity

        if ieee_violated:
            severity = "critical" if thd_f > self.thd_limit * 1.5 else "warning"
            return "high_thd", severity

        if thd_f > self.thd_warning:
            return "high_thd", "warning"

        return "healthy", "normal"
