"""
3-Phase Voltage Source Inverter (VSI) fault signal simulator.

Generates realistic 3-phase voltage and current waveforms under:
  - Healthy operation
  - Open-circuit IGBT fault (T1–T6)
  - Short-circuit / shoot-through fault
  - DC bus undervoltage

All signals are returned as numpy arrays shaped (n_samples,) or
(n_channels, n_samples) depending on the method called.

Reference conditions (defaults):
  - Fundamental frequency : 50 Hz
  - Switching frequency   : 10 kHz
  - DC bus voltage (Vdc)  : 400 V
  - Sampling frequency    : 100 kHz
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Fault label mapping used across the whole repo
INVERTER_FAULT_LABELS: Dict[str, int] = {
    "healthy": 0,
    "open_circuit_T1": 1,
    "open_circuit_T2": 2,
    "open_circuit_T3": 3,
    "open_circuit_T4": 4,
    "open_circuit_T5": 5,
    "open_circuit_T6": 6,
    "short_circuit": 7,
    "dc_undervoltage": 8,
}


@dataclass
class InverterConfig:
    f_fund: float = 50.0          # Fundamental frequency [Hz]
    f_sw: float = 10_000.0        # Switching frequency [Hz]
    f_sample: float = 100_000.0   # Sampling frequency [Hz]
    vdc: float = 400.0            # DC bus voltage [V]
    load_r: float = 10.0          # Load resistance [Ω]
    load_l: float = 5e-3          # Load inductance [H]
    n_cycles: int = 10            # Number of fundamental cycles to simulate
    noise_std: float = 0.5        # Gaussian noise std [V or A]
    random_seed: int = 42


class InverterFaultSimulator:
    """Simulate 3-phase VSI waveforms with injected IGBT faults.

    Parameters
    ----------
    config : InverterConfig
        Simulation parameters. Uses defaults if not provided.

    Examples
    --------
    >>> sim = InverterFaultSimulator()
    >>> signals, label = sim.generate("healthy")
    >>> signals.shape   # (6, n_samples)  — va, vb, vc, ia, ib, ic
    """

    def __init__(self, config: InverterConfig | None = None) -> None:
        self.cfg = config or InverterConfig()
        self._rng = np.random.default_rng(self.cfg.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self, fault_type: str = "healthy"
    ) -> Tuple[np.ndarray, int]:
        """Generate one sample of 3-phase VSI signals.

        Parameters
        ----------
        fault_type : str
            One of the keys in ``INVERTER_FAULT_LABELS``.

        Returns
        -------
        signals : np.ndarray, shape (6, n_samples)
            Rows: [Va, Vb, Vc, Ia, Ib, Ic].
        label : int
            Integer class label.
        """
        if fault_type not in INVERTER_FAULT_LABELS:
            raise ValueError(
                f"Unknown fault type '{fault_type}'. "
                f"Valid options: {list(INVERTER_FAULT_LABELS)}"
            )

        t, va, vb, vc = self._generate_pwm_voltages()

        # Apply fault modification to voltages
        if fault_type == "healthy":
            pass
        elif fault_type.startswith("open_circuit"):
            switch_id = int(fault_type[-1])  # T1 … T6
            va, vb, vc = self._apply_open_circuit(va, vb, vc, t, switch_id)
        elif fault_type == "short_circuit":
            va, vb, vc = self._apply_short_circuit(va, vb, vc, t)
        elif fault_type == "dc_undervoltage":
            va, vb, vc = self._apply_dc_undervoltage(va, vb, vc, t)

        ia, ib, ic = self._simulate_currents(t, va, vb, vc)

        # Add measurement noise
        noise = self._rng.normal(0, self.cfg.noise_std, (6, len(t)))
        signals = np.stack([va, vb, vc, ia, ib, ic]) + noise

        return signals, INVERTER_FAULT_LABELS[fault_type]

    def generate_dataset(
        self,
        n_per_class: int = 200,
        window_size: int = 1024,
        fault_types: List[str] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a windowed dataset for all fault classes.

        Parameters
        ----------
        n_per_class : int
            Number of windows to extract per fault type.
        window_size : int
            Number of samples per window (must be ≤ total signal length).
        fault_types : list[str] or None
            Subset of fault types. Defaults to all.

        Returns
        -------
        X : np.ndarray, shape (N, 6, window_size)
        y : np.ndarray, shape (N,)
        """
        if fault_types is None:
            fault_types = list(INVERTER_FAULT_LABELS)

        X_list, y_list = [], []
        for fault in fault_types:
            for _ in range(n_per_class):
                signals, label = self.generate(fault)
                window = self._random_window(signals, window_size)
                X_list.append(window)
                y_list.append(label)

        X = np.stack(X_list)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    # ------------------------------------------------------------------
    # Internal signal generation helpers
    # ------------------------------------------------------------------

    def _time_axis(self) -> np.ndarray:
        n_samples = int(self.cfg.f_sample * self.cfg.n_cycles / self.cfg.f_fund)
        return np.linspace(0, self.cfg.n_cycles / self.cfg.f_fund, n_samples)

    def _generate_pwm_voltages(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate ideal sinusoidal PWM phase-to-neutral voltages."""
        t = self._time_axis()
        omega = 2 * np.pi * self.cfg.f_fund
        vdc_half = self.cfg.vdc / 2

        # Sinusoidal reference modulated against triangular carrier
        va = vdc_half * np.sin(omega * t)
        vb = vdc_half * np.sin(omega * t - 2 * np.pi / 3)
        vc = vdc_half * np.sin(omega * t + 2 * np.pi / 3)

        # Simplified PWM effect via harmonic ripple at switching freq
        v_ripple = 0.05 * vdc_half * np.sin(2 * np.pi * self.cfg.f_sw * t)
        va += v_ripple
        vb += v_ripple
        vc += v_ripple

        return t, va, vb, vc

    def _simulate_currents(
        self,
        t: np.ndarray,
        va: np.ndarray,
        vb: np.ndarray,
        vc: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """First-order RL load model: I(s) = V(s) / (R + sL)."""
        dt = t[1] - t[0]
        tau = self.cfg.load_l / self.cfg.load_r

        def rl_filter(v: np.ndarray) -> np.ndarray:
            i = np.zeros_like(v)
            for k in range(1, len(v)):
                i[k] = i[k - 1] + (dt / tau) * (v[k] / self.cfg.load_r - i[k - 1])
            return i

        return rl_filter(va), rl_filter(vb), rl_filter(vc)

    # ------------------------------------------------------------------
    # Fault injection methods
    # ------------------------------------------------------------------

    def _apply_open_circuit(
        self,
        va: np.ndarray,
        vb: np.ndarray,
        vc: np.ndarray,
        t: np.ndarray,
        switch_id: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Open-circuit fault on switch T{switch_id}.

        Each switch is associated with one phase leg and polarity:
          T1/T4 → phase A (upper/lower),  T3/T6 → phase B,  T5/T2 → phase C
        When a switch opens, the affected half-cycle is clamped to zero.
        """
        phase_map = {1: 0, 4: 0, 3: 1, 6: 1, 5: 2, 2: 2}
        upper_map = {1, 3, 5}  # upper arm switches
        phases = [va, vb, vc]

        p_idx = phase_map.get(switch_id, 0)
        v = phases[p_idx].copy()

        if switch_id in upper_map:
            # Clamp positive half-cycle
            v[v > 0] = 0.0
        else:
            # Clamp negative half-cycle
            v[v < 0] = 0.0

        phases[p_idx] = v
        return phases[0], phases[1], phases[2]

    def _apply_short_circuit(
        self,
        va: np.ndarray,
        vb: np.ndarray,
        vc: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shoot-through: brief overcurrent spike at a random instant."""
        spike_start = self._rng.integers(0, len(t) // 2)
        spike_len = max(1, int(0.001 * self.cfg.f_sample))  # 1 ms spike

        spike = np.zeros(len(t))
        spike[spike_start : spike_start + spike_len] = self.cfg.vdc * 1.5

        return va + spike, vb + spike, vc + spike

    def _apply_dc_undervoltage(
        self,
        va: np.ndarray,
        vb: np.ndarray,
        vc: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """DC bus sags to 60 % of nominal after half the signal duration."""
        sag_factor = 0.6
        mid = len(t) // 2
        va[mid:] *= sag_factor
        vb[mid:] *= sag_factor
        vc[mid:] *= sag_factor
        return va, vb, vc

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _random_window(self, signals: np.ndarray, window_size: int) -> np.ndarray:
        n_samples = signals.shape[1]
        if window_size >= n_samples:
            return signals[:, :window_size]
        start = self._rng.integers(0, n_samples - window_size)
        return signals[:, start : start + window_size]

    @staticmethod
    def fault_labels() -> Dict[str, int]:
        return INVERTER_FAULT_LABELS
