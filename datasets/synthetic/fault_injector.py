"""
Generic fault injector for arbitrary 1-D or multi-channel signals.

Provides standalone transforms that can be applied to any numpy signal
array, independent of its source (simulated or real).  Useful for
data augmentation during training.

Fault transforms:
  - Impulsive spike       : models capacitor discharge or overcurrent pulse
  - Amplitude dropout     : models phase loss or sensor failure
  - Additive harmonic     : injects a harmonic component at N×f_fund
  - White noise burst     : models EMI interference window
  - Amplitude modulation  : models bearing or eccentricity effects
  - DC offset drift       : models sensor bias or partial demagnetization
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class FaultInjector:
    """Apply parameterized fault transforms to signal arrays.

    All methods accept and return numpy arrays of shape
    ``(n_channels, n_samples)`` or ``(n_samples,)`` and are pure
    (non-in-place) — they always return a new array.

    Parameters
    ----------
    random_seed : int
        Seed for reproducible stochastic transforms.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self._rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Individual fault transforms
    # ------------------------------------------------------------------

    def impulsive_spike(
        self,
        signal: np.ndarray,
        amplitude: float = 5.0,
        width_frac: float = 0.005,
        n_spikes: int = 1,
    ) -> np.ndarray:
        """Inject one or more impulse spikes simulating shoot-through events.

        Parameters
        ----------
        amplitude : float
            Spike amplitude relative to signal RMS.
        width_frac : float
            Spike width as a fraction of total signal length.
        n_spikes : int
            Number of impulse events.
        """
        out = signal.copy().astype(float)
        n = out.shape[-1]
        spike_len = max(1, int(n * width_frac))
        signal_rms = np.sqrt(np.mean(out ** 2))

        for _ in range(n_spikes):
            pos = self._rng.integers(0, n - spike_len)
            spike = amplitude * signal_rms * np.ones(spike_len)
            if out.ndim == 1:
                out[pos : pos + spike_len] += spike
            else:
                channel = self._rng.integers(0, out.shape[0])
                out[channel, pos : pos + spike_len] += spike

        return out

    def amplitude_dropout(
        self,
        signal: np.ndarray,
        dropout_frac: float = 0.3,
        channel: Optional[int] = None,
    ) -> np.ndarray:
        """Zero out a contiguous segment (phase loss simulation).

        Parameters
        ----------
        dropout_frac : float
            Fraction of signal length to zero.
        channel : int or None
            Which channel to drop. Randomly selected if None.
        """
        out = signal.copy().astype(float)
        n = out.shape[-1]
        drop_len = int(n * dropout_frac)
        start = self._rng.integers(0, n - drop_len)

        if out.ndim == 1:
            out[start : start + drop_len] = 0.0
        else:
            ch = channel if channel is not None else self._rng.integers(0, out.shape[0])
            out[ch, start : start + drop_len] = 0.0

        return out

    def additive_harmonic(
        self,
        signal: np.ndarray,
        f_sample: float,
        f_fund: float,
        harmonic_order: int = 5,
        relative_amp: float = 0.1,
    ) -> np.ndarray:
        """Inject a harmonic component (e.g., 5th harmonic distortion).

        Parameters
        ----------
        f_sample : float
            Sampling frequency [Hz].
        f_fund : float
            Fundamental frequency of the signal [Hz].
        harmonic_order : int
            Order of harmonic to inject (e.g., 3, 5, 7).
        relative_amp : float
            Amplitude relative to signal peak.
        """
        out = signal.copy().astype(float)
        n = out.shape[-1]
        t = np.arange(n) / f_sample
        harmonic = relative_amp * np.abs(out).max(-1, keepdims=True if out.ndim > 1 else False)

        if out.ndim == 1:
            harmonic_wave = float(harmonic) * np.sin(2 * np.pi * harmonic_order * f_fund * t)
            out += harmonic_wave
        else:
            for ch in range(out.shape[0]):
                peak = np.abs(out[ch]).max()
                harmonic_wave = relative_amp * peak * np.sin(
                    2 * np.pi * harmonic_order * f_fund * t
                )
                out[ch] += harmonic_wave

        return out

    def noise_burst(
        self,
        signal: np.ndarray,
        burst_frac: float = 0.1,
        snr_db: float = 10.0,
    ) -> np.ndarray:
        """Inject a burst of white noise (EMI window simulation).

        Parameters
        ----------
        burst_frac : float
            Fraction of signal length affected.
        snr_db : float
            Signal-to-noise ratio within the burst [dB].
        """
        out = signal.copy().astype(float)
        n = out.shape[-1]
        burst_len = int(n * burst_frac)
        start = self._rng.integers(0, n - burst_len)

        signal_power = np.mean(out ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self._rng.normal(0, np.sqrt(noise_power), out.shape[:-1] + (burst_len,))

        if out.ndim == 1:
            out[start : start + burst_len] += noise
        else:
            out[:, start : start + burst_len] += noise

        return out

    def amplitude_modulation(
        self,
        signal: np.ndarray,
        f_sample: float,
        f_mod: float,
        mod_depth: float = 0.15,
    ) -> np.ndarray:
        """Amplitude modulation simulating bearing or eccentricity effects.

        Parameters
        ----------
        f_mod : float
            Modulation frequency [Hz] (bearing characteristic frequency).
        mod_depth : float
            Modulation depth (0–1).
        """
        out = signal.copy().astype(float)
        n = out.shape[-1]
        t = np.arange(n) / f_sample
        envelope = 1 + mod_depth * np.sin(2 * np.pi * f_mod * t)

        if out.ndim == 1:
            return out * envelope
        return out * envelope[np.newaxis, :]

    def dc_offset_drift(
        self,
        signal: np.ndarray,
        max_drift_frac: float = 0.05,
    ) -> np.ndarray:
        """Add a linearly increasing DC offset (sensor bias / demagnetization).

        Parameters
        ----------
        max_drift_frac : float
            Maximum offset as fraction of signal peak amplitude.
        """
        out = signal.copy().astype(float)
        n = out.shape[-1]
        peak = np.abs(out).max()
        drift = np.linspace(0, max_drift_frac * peak, n)

        if out.ndim == 1:
            return out + drift
        return out + drift[np.newaxis, :]

    # ------------------------------------------------------------------
    # Compound / random augmentation
    # ------------------------------------------------------------------

    def random_augment(
        self,
        signal: np.ndarray,
        f_sample: float = 50_000.0,
        f_fund: float = 50.0,
        p_each: float = 0.3,
    ) -> np.ndarray:
        """Apply a random combination of fault transforms for augmentation.

        Each transform is applied independently with probability ``p_each``.

        Parameters
        ----------
        p_each : float
            Probability (0–1) of applying each individual transform.
        """
        out = signal.copy()
        transforms = [
            lambda s: self.impulsive_spike(s),
            lambda s: self.amplitude_dropout(s),
            lambda s: self.additive_harmonic(s, f_sample, f_fund),
            lambda s: self.noise_burst(s),
            lambda s: self.amplitude_modulation(s, f_sample, f_sample * 0.005),
            lambda s: self.dc_offset_drift(s),
        ]
        for transform in transforms:
            if self._rng.random() < p_each:
                out = transform(out)
        return out
