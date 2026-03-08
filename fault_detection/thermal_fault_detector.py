"""
Thermal / overtemperature fault detection pipeline.

Detects overheating conditions in motor windings and power electronics
components using two complementary approaches:

1. **Autoencoder-based** (unsupervised):
   Trains on healthy current signatures; elevated reconstruction error
   signals thermal degradation (increasing winding resistance changes
   current amplitude and shape).

2. **Statistical trend analysis** (interpretable, no ML required):
   Monitors current amplitude, RMS trend, and spectral features over
   time for drift indicating thermal degradation.

Motor thermal fault signatures:
  - Increasing winding resistance → decreasing current amplitude
  - Gradual amplitude modulation due to expanding conductor geometry
  - Subtle change in harmonic content (3rd / 5th harmonic amplitude)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThermalFaultResult:
    """Result of thermal fault analysis on one window."""
    rms_current: float            # RMS of current window
    rms_trend: float              # Slope of recent RMS values (A / window)
    amplitude_change_pct: float   # % change from baseline
    autoencoder_score: float      # Reconstruction MSE (if AE available)
    is_fault: bool
    severity: str                 # 'normal' | 'warning' | 'critical'
    fault_type: str               # 'healthy' | 'overtemperature'


class ThermalFaultDetector:
    """Detect motor overtemperature faults via current signal monitoring.

    Parameters
    ----------
    f_sample : float
        Sampling frequency [Hz].
    window_size : int
        Signal window size.
    amplitude_drop_threshold : float
        Percentage drop in current RMS from baseline to flag warning.
    amplitude_critical_threshold : float
        Percentage drop in current RMS from baseline to flag critical.
    trend_window : int
        Number of recent windows used for trend estimation.
    autoencoder : nn.Module or None
        Trained AutoencoderAnomalyDetector. If provided, uses reconstruction
        error as an additional anomaly signal.
    ae_threshold : float or None
        Anomaly score threshold for the autoencoder.
    device : str
        'cpu' or 'cuda' (for autoencoder inference).

    Examples
    --------
    >>> detector = ThermalFaultDetector(f_sample=50_000)
    >>> detector.set_baseline(healthy_signal)
    >>> result = detector.detect(new_signal)
    """

    def __init__(
        self,
        f_sample: float = 50_000.0,
        window_size: int = 1024,
        amplitude_drop_threshold: float = 10.0,
        amplitude_critical_threshold: float = 25.0,
        trend_window: int = 20,
        autoencoder=None,
        ae_threshold: Optional[float] = None,
        device: str = "cpu",
    ) -> None:
        self.f_sample = f_sample
        self.window_size = window_size
        self.amp_warn = amplitude_drop_threshold
        self.amp_crit = amplitude_critical_threshold
        self.trend_window = trend_window
        self.device = device

        self._baseline_rms: Optional[float] = None
        self._rms_history: Deque[float] = deque(maxlen=trend_window)

        # Autoencoder (optional)
        self._ae = None
        self._ae_threshold = ae_threshold
        if autoencoder is not None:
            import torch
            self._ae = autoencoder.to(device).eval()
            self._ae_threshold = ae_threshold

    # ------------------------------------------------------------------
    # Baseline calibration
    # ------------------------------------------------------------------

    def set_baseline(self, signal: np.ndarray) -> float:
        """Calibrate baseline RMS from a known-healthy signal segment.

        Parameters
        ----------
        signal : np.ndarray, shape (n_channels, T) or (T,)

        Returns
        -------
        baseline_rms : float
        """
        rms = float(np.sqrt(np.mean(signal ** 2)))
        self._baseline_rms = rms
        logger.info("Thermal detector baseline set: RMS = %.4f", rms)
        return rms

    def calibrate_autoencoder(
        self,
        healthy_signals: "np.ndarray",
        percentile: float = 95.0,
    ) -> float:
        """Set autoencoder threshold from healthy training data.

        Parameters
        ----------
        healthy_signals : np.ndarray, shape (N, C, T)
        percentile : float
            Percentile of reconstruction error to use as threshold.
        """
        if self._ae is None:
            raise RuntimeError("No autoencoder model loaded.")

        import torch
        tensor = torch.from_numpy(healthy_signals.astype(np.float32)).to(self.device)
        threshold = self._ae.set_threshold(tensor, percentile=percentile)
        self._ae_threshold = threshold
        logger.info("Autoencoder threshold calibrated: %.6f", threshold)
        return threshold

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, signal: np.ndarray) -> ThermalFaultResult:
        """Detect thermal fault in one signal window.

        Parameters
        ----------
        signal : np.ndarray, shape (n_channels, T) or (T,)

        Returns
        -------
        ThermalFaultResult
        """
        rms = float(np.sqrt(np.mean(signal ** 2)))
        self._rms_history.append(rms)

        # Amplitude change from baseline
        amplitude_change_pct = 0.0
        if self._baseline_rms is not None and self._baseline_rms > 1e-9:
            amplitude_change_pct = 100.0 * (self._baseline_rms - rms) / self._baseline_rms

        # Trend: linear regression slope over recent RMS values
        rms_trend = self._compute_rms_trend()

        # Autoencoder score
        ae_score = 0.0
        if self._ae is not None:
            ae_score = self._ae_score(signal)

        # Classification
        is_fault, severity = self._classify(
            amplitude_change_pct, rms_trend, ae_score
        )
        fault_type = "overtemperature" if is_fault else "healthy"

        return ThermalFaultResult(
            rms_current=rms,
            rms_trend=rms_trend,
            amplitude_change_pct=amplitude_change_pct,
            autoencoder_score=ae_score,
            is_fault=is_fault,
            severity=severity,
            fault_type=fault_type,
        )

    def detect_sequence(
        self, signals: np.ndarray, hop_size: Optional[int] = None
    ) -> List[ThermalFaultResult]:
        """Detect thermal faults across a time series of windows.

        Parameters
        ----------
        signals : np.ndarray, shape (N, T) or (N, C, T)
        """
        results = []
        for s in signals:
            results.append(self.detect(s))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_rms_trend(self) -> float:
        """Estimate slope of RMS over recent windows via linear regression."""
        if len(self._rms_history) < 3:
            return 0.0
        y = np.array(self._rms_history, dtype=float)
        x = np.arange(len(y), dtype=float)
        # Simple slope estimate
        slope = (len(y) * np.dot(x, y) - x.sum() * y.sum()) / (
            len(y) * (x ** 2).sum() - x.sum() ** 2 + 1e-12
        )
        return float(slope)

    def _ae_score(self, signal: np.ndarray) -> float:
        """Compute autoencoder reconstruction error."""
        import torch
        s = signal.astype(np.float32)
        if s.ndim == 1:
            s = s[np.newaxis, np.newaxis, :]
        elif s.ndim == 2:
            s = s[np.newaxis, :]
        tensor = torch.from_numpy(s).to(self.device)
        score = float(self._ae.anomaly_score(tensor).cpu().numpy()[0])
        return score

    def _classify(
        self,
        amp_change_pct: float,
        trend: float,
        ae_score: float,
    ) -> Tuple[bool, str]:
        """Rule-based thermal severity classification."""
        is_fault = False
        severity = "normal"

        # Critical: large amplitude drop
        if amp_change_pct > self.amp_crit:
            is_fault = True
            severity = "critical"
        # Warning: moderate amplitude drop or negative trend
        elif amp_change_pct > self.amp_warn or (trend < -0.002 and len(self._rms_history) >= 5):
            is_fault = True
            severity = "warning"

        # Autoencoder reinforcement
        if self._ae is not None and self._ae_threshold is not None:
            if ae_score > self._ae_threshold:
                is_fault = True
                if severity == "normal":
                    severity = "warning"
                elif severity == "warning":
                    severity = "critical"

        return is_fault, severity

    @property
    def baseline_rms(self) -> Optional[float]:
        return self._baseline_rms

    def reset_history(self) -> None:
        """Clear accumulated RMS history (e.g., at start of new session)."""
        self._rms_history.clear()
        self._baseline_rms = None
