"""
Permanent Magnet Synchronous Motor (PMSM) drive fault signal simulator.

Generates 3-phase stator current waveforms under:
  - Healthy operation
  - Phase current imbalance (phase loss)
  - Inter-turn short circuit (ITSC) — introduces sub-harmonic sidebands
  - Bearing fault — adds characteristic fault frequency ripple
  - Overtemperature — increases winding resistance, reduces current amplitude

Signals are shaped (3, n_samples): [ia, ib, ic].

Reference:
  Nandi, S., Toliyat, H. A., & Li, X. (2005). Condition monitoring and
  fault diagnosis of electrical motors — A review. IEEE Trans. Energy
  Conversion, 20(4), 719–729.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

MOTOR_FAULT_LABELS: Dict[str, int] = {
    "healthy": 0,
    "phase_loss": 1,
    "itsc": 2,           # Inter-turn short circuit
    "bearing": 3,
    "overtemperature": 4,
}


@dataclass
class MotorConfig:
    f_supply: float = 50.0         # Supply frequency [Hz]
    f_sample: float = 50_000.0     # Sampling frequency [Hz]
    n_cycles: int = 20             # Number of supply cycles
    i_rated: float = 10.0          # Rated peak current [A]
    n_poles: int = 4               # Number of poles
    slip: float = 0.03             # Slip (for bearing fault calc)
    noise_std: float = 0.05        # Gaussian noise std [A]
    random_seed: int = 42


class MotorDriveSimulator:
    """Simulate 3-phase PMSM stator currents with common fault signatures.

    Parameters
    ----------
    config : MotorConfig
        Motor and simulation parameters. Uses defaults if not provided.

    Examples
    --------
    >>> sim = MotorDriveSimulator()
    >>> signals, label = sim.generate("bearing")
    >>> signals.shape   # (3, n_samples)  — ia, ib, ic
    """

    def __init__(self, config: MotorConfig | None = None) -> None:
        self.cfg = config or MotorConfig()
        self._rng = np.random.default_rng(self.cfg.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self, fault_type: str = "healthy"
    ) -> Tuple[np.ndarray, int]:
        """Generate one sample of 3-phase stator current waveforms.

        Parameters
        ----------
        fault_type : str
            One of the keys in ``MOTOR_FAULT_LABELS``.

        Returns
        -------
        signals : np.ndarray, shape (3, n_samples)
        label   : int
        """
        if fault_type not in MOTOR_FAULT_LABELS:
            raise ValueError(
                f"Unknown fault type '{fault_type}'. "
                f"Valid options: {list(MOTOR_FAULT_LABELS)}"
            )

        t = self._time_axis()
        ia, ib, ic = self._healthy_currents(t)

        if fault_type == "healthy":
            pass
        elif fault_type == "phase_loss":
            ia, ib, ic = self._apply_phase_loss(ia, ib, ic, t)
        elif fault_type == "itsc":
            ia, ib, ic = self._apply_itsc(ia, ib, ic, t)
        elif fault_type == "bearing":
            ia, ib, ic = self._apply_bearing_fault(ia, ib, ic, t)
        elif fault_type == "overtemperature":
            ia, ib, ic = self._apply_overtemperature(ia, ib, ic, t)

        noise = self._rng.normal(0, self.cfg.noise_std, (3, len(t)))
        signals = np.stack([ia, ib, ic]) + noise

        return signals, MOTOR_FAULT_LABELS[fault_type]

    def generate_dataset(
        self,
        n_per_class: int = 200,
        window_size: int = 1024,
        fault_types: List[str] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a windowed, multi-class dataset.

        Returns
        -------
        X : np.ndarray, shape (N, 3, window_size)
        y : np.ndarray, shape (N,)
        """
        if fault_types is None:
            fault_types = list(MOTOR_FAULT_LABELS)

        X_list, y_list = [], []
        for fault in fault_types:
            for _ in range(n_per_class):
                signals, label = self.generate(fault)
                window = self._random_window(signals, window_size)
                X_list.append(window)
                y_list.append(label)

        return np.stack(X_list), np.array(y_list, dtype=np.int64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _time_axis(self) -> np.ndarray:
        n = int(self.cfg.f_sample * self.cfg.n_cycles / self.cfg.f_supply)
        return np.linspace(0, self.cfg.n_cycles / self.cfg.f_supply, n)

    def _healthy_currents(
        self, t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        omega = 2 * np.pi * self.cfg.f_supply
        ia = self.cfg.i_rated * np.sin(omega * t)
        ib = self.cfg.i_rated * np.sin(omega * t - 2 * np.pi / 3)
        ic = self.cfg.i_rated * np.sin(omega * t + 2 * np.pi / 3)
        return ia, ib, ic

    # ------------------------------------------------------------------
    # Fault injection
    # ------------------------------------------------------------------

    def _apply_phase_loss(
        self,
        ia: np.ndarray,
        ib: np.ndarray,
        ic: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Phase C loss after mid-signal — remaining phases increase by √3."""
        mid = len(t) // 2
        ic_fault = ic.copy()
        ic_fault[mid:] = 0.0
        ia_fault = ia.copy()
        ib_fault = ib.copy()
        ia_fault[mid:] *= np.sqrt(3)
        ib_fault[mid:] *= np.sqrt(3)
        return ia_fault, ib_fault, ic_fault

    def _apply_itsc(
        self,
        ia: np.ndarray,
        ib: np.ndarray,
        ic: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inter-turn short circuit: introduces (1±2s)f sidebands in ia.

        Reference sideband frequencies: f_s ± 2·slip·f_s
        """
        f_sb1 = self.cfg.f_supply * (1 - 2 * self.cfg.slip)  # lower sideband
        f_sb2 = self.cfg.f_supply * (1 + 2 * self.cfg.slip)  # upper sideband
        sideband_amp = 0.08 * self.cfg.i_rated

        ia_fault = ia + sideband_amp * np.sin(2 * np.pi * f_sb1 * t)
        ia_fault += sideband_amp * np.sin(2 * np.pi * f_sb2 * t)
        return ia_fault, ib, ic

    def _apply_bearing_fault(
        self,
        ia: np.ndarray,
        ib: np.ndarray,
        ic: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bearing fault: modulates current at bearing characteristic freq.

        Simplified bearing fault frequency (outer race):
          f_bpfo ≈ 0.4 * n_balls * (1 - slip) * f_supply / (n_poles/2)
        Here a typical BPFO ≈ 3.58 * f_supply is used.
        """
        f_bearing = 3.58 * self.cfg.f_supply * (1 - self.cfg.slip)
        bearing_amp = 0.06 * self.cfg.i_rated

        # Amplitude modulation at bearing frequency
        mod = 1 + bearing_amp * np.abs(np.sin(2 * np.pi * f_bearing * t))
        ia_fault = ia * mod
        ib_fault = ib * mod
        ic_fault = ic * mod
        return ia_fault, ib_fault, ic_fault

    def _apply_overtemperature(
        self,
        ia: np.ndarray,
        ib: np.ndarray,
        ic: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Overtemperature: winding resistance increases, reducing current.

        Models a linear resistance rise from R0 to 1.4·R0 over signal duration,
        which reduces the current amplitude proportionally.
        """
        resistance_scale = np.linspace(1.0, 1.0 / 1.4, len(t))
        return ia * resistance_scale, ib * resistance_scale, ic * resistance_scale

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _random_window(self, signals: np.ndarray, window_size: int) -> np.ndarray:
        n = signals.shape[1]
        if window_size >= n:
            return signals[:, :window_size]
        start = self._rng.integers(0, n - window_size)
        return signals[:, start : start + window_size]

    @staticmethod
    def fault_labels() -> Dict[str, int]:
        return MOTOR_FAULT_LABELS
