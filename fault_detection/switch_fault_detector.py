"""
IGBT open-circuit and short-circuit fault detection pipeline.

Combines signal processing (FFT + STFT) with a trained classifier model
(CNN or Transformer) to identify:
  - Which switch has failed (T1–T6 open circuit)
  - Short-circuit / shoot-through events
  - Healthy operation

The detector wraps a trained PyTorch model and provides a clean
inference interface for streaming and batch use cases.

Diagnostic methodology:
  - Open-circuit: characteristic asymmetry in phase currents / voltages;
    half-cycle amplitude clamped to zero (one polarity disappears).
  - Short-circuit: sudden impulsive overcurrent spike across all phases.

Reference:
  Trabelsi, M., et al. (2012). FPGA-based real-time power converter
  switch failure diagnosis using lookup-table method. IEEE Trans.
  Industrial Electronics, 59(4), 2187–2195.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

SWITCH_FAULT_CLASSES = [
    "healthy",
    "open_circuit_T1",
    "open_circuit_T2",
    "open_circuit_T3",
    "open_circuit_T4",
    "open_circuit_T5",
    "open_circuit_T6",
    "short_circuit",
    "dc_undervoltage",
]


class SwitchFaultDetector:
    """End-to-end inverter switch fault detection pipeline.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch classifier (CNN1D, Transformer, etc.).
    f_sample : float
        Signal sampling frequency [Hz].
    window_size : int
        Expected input window length.
    device : str
        'cpu' or 'cuda'.
    confidence_threshold : float
        Minimum softmax probability to accept a prediction.
        Below this, prediction is marked as 'uncertain'.

    Examples
    --------
    >>> from models import CNN1DWaveformClassifier
    >>> model = CNN1DWaveformClassifier(n_channels=6, n_classes=9)
    >>> detector = SwitchFaultDetector(model, f_sample=100_000)
    >>> result = detector.detect(signal_window)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        f_sample: float = 100_000.0,
        window_size: int = 1024,
        device: str = "cpu",
        confidence_threshold: float = 0.6,
    ) -> None:
        self.model = model.to(device).eval()
        self.f_sample = f_sample
        self.window_size = window_size
        self.device = device
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, signal: np.ndarray) -> Dict:
        """Detect switch fault from a single signal window.

        Parameters
        ----------
        signal : np.ndarray, shape (n_channels, window_size)

        Returns
        -------
        result : dict with keys:
            - 'fault_type' : str
            - 'label'      : int
            - 'confidence' : float
            - 'probabilities' : dict {fault_name: probability}
            - 'uncertain'  : bool
        """
        signal = self._preprocess(signal)
        tensor = torch.from_numpy(signal).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        return {
            "fault_type": SWITCH_FAULT_CLASSES[pred_idx],
            "label": pred_idx,
            "confidence": confidence,
            "probabilities": {
                name: float(p) for name, p in zip(SWITCH_FAULT_CLASSES, probs)
            },
            "uncertain": confidence < self.confidence_threshold,
        }

    def detect_batch(self, signals: np.ndarray) -> List[Dict]:
        """Detect faults in a batch of signal windows.

        Parameters
        ----------
        signals : np.ndarray, shape (N, n_channels, window_size)

        Returns
        -------
        results : list of N detection result dicts
        """
        processed = np.stack([self._preprocess(s) for s in signals])
        tensor = torch.from_numpy(processed).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for i, p in enumerate(probs):
            pred_idx = int(np.argmax(p))
            results.append({
                "fault_type": SWITCH_FAULT_CLASSES[pred_idx],
                "label": pred_idx,
                "confidence": float(p[pred_idx]),
                "probabilities": {
                    name: float(prob) for name, prob in zip(SWITCH_FAULT_CLASSES, p)
                },
                "uncertain": float(p[pred_idx]) < self.confidence_threshold,
            })
        return results

    def streaming_detect(
        self,
        full_signal: np.ndarray,
        hop_size: Optional[int] = None,
    ) -> List[Dict]:
        """Sliding-window detection over a continuous signal stream.

        Parameters
        ----------
        full_signal : np.ndarray, shape (n_channels, T)
        hop_size : int or None
            Stride between windows. Defaults to window_size // 2.

        Returns
        -------
        detections : list of dicts, one per window
        """
        hop = hop_size or self.window_size // 2
        _, T = full_signal.shape
        detections = []

        for start in range(0, T - self.window_size + 1, hop):
            window = full_signal[:, start : start + self.window_size]
            result = self.detect(window)
            result["window_start"] = start
            result["window_end"] = start + self.window_size
            result["time_start_s"] = start / self.f_sample
            detections.append(result)

        return detections

    # ------------------------------------------------------------------
    # Rule-based pre-screening (physics-based heuristics)
    # ------------------------------------------------------------------

    @staticmethod
    def rule_based_screen(
        signal: np.ndarray,
        f_sample: float,
    ) -> Dict:
        """Quick rule-based pre-screening before running the ML model.

        Checks:
        1. Phase asymmetry (indicator of open-circuit fault)
        2. Impulsive spike (indicator of short-circuit)
        3. DC component (indicator of DC bus issue)

        Parameters
        ----------
        signal : np.ndarray, shape (n_channels, T)
            Rows: [Va, Vb, Vc, Ia, Ib, Ic]

        Returns
        -------
        dict with 'flags': list of triggered rule names
        """
        flags = []

        # Phase current amplitudes (channels 3–5)
        if signal.shape[0] >= 6:
            curr = signal[3:6]
            rms = np.sqrt(np.mean(curr ** 2, axis=-1))
            max_rms = rms.max()
            min_rms = rms.min()
            if max_rms > 1e-6 and (max_rms - min_rms) / max_rms > 0.3:
                flags.append("phase_asymmetry")

        # Short-circuit: peak/RMS (crest factor) > 5
        for ch in signal:
            rms_ch = np.sqrt(np.mean(ch ** 2))
            if rms_ch > 1e-6 and np.max(np.abs(ch)) / rms_ch > 5.0:
                flags.append("impulsive_spike")
                break

        # DC offset check
        for ch in signal:
            dc = np.abs(ch.mean())
            rms_ch = np.sqrt(np.mean(ch ** 2))
            if rms_ch > 1e-6 and dc / rms_ch > 0.15:
                flags.append("dc_offset")
                break

        return {
            "flags": flags,
            "fault_suspected": len(flags) > 0,
            "flag_count": len(flags),
        }

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load model state dict."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info("Model loaded from %s", path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _preprocess(self, signal: np.ndarray) -> np.ndarray:
        """Z-score normalize and ensure correct dtype."""
        signal = signal.astype(np.float32)
        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True) + 1e-8
        return (signal - mean) / std
