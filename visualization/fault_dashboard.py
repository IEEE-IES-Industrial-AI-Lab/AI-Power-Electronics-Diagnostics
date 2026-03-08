"""
Multi-panel fault analysis dashboard.

Creates a comprehensive single-figure diagnostic view combining:
  - Raw waveform (all channels)
  - FFT amplitude spectrum
  - STFT spectrogram
  - DWT sub-band energy
  - Fault probability bar chart (from model output)
  - Harmonic distortion chart
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class FaultDashboard:
    """Multi-panel diagnostic dashboard for a single fault detection event.

    Parameters
    ----------
    f_sample : float
        Sampling frequency [Hz].
    f_fund : float
        Fundamental frequency [Hz].
    class_names : list[str]
        Fault class names (for probability bar chart).
    """

    def __init__(
        self,
        f_sample: float = 50_000.0,
        f_fund: float = 50.0,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.f_sample = f_sample
        self.f_fund = f_fund
        self.class_names = class_names or []

    def plot(
        self,
        signal: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        predicted_fault: Optional[str] = None,
        channel_idx: int = 3,
        f_max: float = 5_000.0,
        title: str = "Fault Diagnostic Dashboard",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Generate the full multi-panel dashboard.

        Parameters
        ----------
        signal : np.ndarray, shape (C, T)
        probabilities : np.ndarray or None, shape (n_classes,)
            Model output softmax probabilities.
        predicted_fault : str or None
            Predicted fault label string.
        channel_idx : int
            Primary channel for 1-D plots.
        f_max : float
            Maximum frequency for spectrum plots.
        """
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(
            f"{title}" + (f"  |  Predicted: {predicted_fault}" if predicted_fault else ""),
            fontsize=15, fontweight="bold", y=0.98,
        )

        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        # ── Panel 1: All waveforms (spans 2 columns) ─────────────────────
        ax_wave = fig.add_subplot(gs[0, :2])
        self._plot_waveforms(ax_wave, signal)

        # ── Panel 2: Probability bar (top right) ──────────────────────────
        ax_prob = fig.add_subplot(gs[0, 2])
        if probabilities is not None:
            self._plot_probabilities(ax_prob, probabilities, predicted_fault)
        else:
            ax_prob.set_visible(False)

        # ── Panel 3: FFT spectrum ─────────────────────────────────────────
        ax_fft = fig.add_subplot(gs[1, 0])
        self._plot_fft(ax_fft, signal[channel_idx], f_max)

        # ── Panel 4: STFT spectrogram ─────────────────────────────────────
        ax_spec = fig.add_subplot(gs[1, 1])
        self._plot_stft(ax_spec, signal[channel_idx], f_max)

        # ── Panel 5: DWT sub-band energy ──────────────────────────────────
        ax_dwt = fig.add_subplot(gs[1, 2])
        self._plot_dwt(ax_dwt, signal[channel_idx])

        # ── Panel 6: Phase current amplitudes (RMS per channel) ───────────
        ax_rms = fig.add_subplot(gs[2, 0])
        self._plot_channel_rms(ax_rms, signal)

        # ── Panel 7: Harmonic content ─────────────────────────────────────
        ax_harm = fig.add_subplot(gs[2, 1])
        self._plot_harmonics(ax_harm, signal[channel_idx])

        # ── Panel 8: Phase portrait (ia vs ib) ───────────────────────────
        ax_phase = fig.add_subplot(gs[2, 2])
        self._plot_phase_portrait(ax_phase, signal)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Panel helpers
    # ------------------------------------------------------------------

    def _plot_waveforms(self, ax: plt.Axes, signal: np.ndarray) -> None:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        labels_v = [r"$V_a$", r"$V_b$", r"$V_c$"]
        labels_i = [r"$I_a$", r"$I_b$", r"$I_c$"]
        T = signal.shape[1]
        t = np.arange(T) / self.f_sample * 1000

        for c in range(min(signal.shape[0], 6)):
            label = (labels_v + labels_i)[c]
            ax.plot(t, signal[c], color=colors[c % len(colors)],
                    linewidth=0.7, alpha=0.85, label=label)

        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.set_title("Signal Waveforms", fontsize=11, fontweight="bold")
        ax.legend(ncol=6, fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25, linestyle="--")

    def _plot_probabilities(
        self,
        ax: plt.Axes,
        probs: np.ndarray,
        predicted: Optional[str],
    ) -> None:
        names = self.class_names or [f"Class {i}" for i in range(len(probs))]
        colors = ["#d62728" if n == predicted else "#1f77b4" for n in names]
        bars = ax.barh(names, probs * 100, color=colors, edgecolor="k", linewidth=0.4)
        ax.set_xlabel("Probability (%)", fontsize=10)
        ax.set_title("Fault Probabilities", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 105)
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        for bar, p in zip(bars, probs):
            if p > 0.05:
                ax.text(p * 100 + 1, bar.get_y() + bar.get_height() / 2,
                        f"{p*100:.1f}%", va="center", fontsize=7)

    def _plot_fft(self, ax: plt.Axes, signal: np.ndarray, f_max: float) -> None:
        n = len(signal)
        win = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(signal * win) / (win.sum() / 2))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.f_sample)
        mask = freqs <= f_max
        ax.semilogy(freqs[mask], spectrum[mask] + 1e-9, linewidth=0.7, color="#1f77b4")

        for h in range(1, 11):
            f_h = h * self.f_fund
            if f_h <= f_max:
                ax.axvline(f_h, color="#d62728", alpha=0.4, linewidth=0.6)

        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Amplitude (log)", fontsize=10)
        ax.set_title("FFT Spectrum", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")

    def _plot_stft(self, ax: plt.Axes, signal: np.ndarray, f_max: float) -> None:
        from scipy.signal import stft as scipy_stft
        n_fft = 256
        freqs, times, Zxx = scipy_stft(
            signal, fs=self.f_sample, nperseg=n_fft,
            noverlap=n_fft // 2, window="hann",
        )
        magnitude = 20 * np.log10(np.abs(Zxx) + 1e-9)
        mask = freqs <= f_max

        ax.pcolormesh(times * 1000, freqs[mask], magnitude[mask],
                      shading="gouraud", cmap="viridis")
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_ylabel("Freq (Hz)", fontsize=10)
        ax.set_title("STFT Spectrogram", fontsize=11, fontweight="bold")

    def _plot_dwt(self, ax: plt.Axes, signal: np.ndarray) -> None:
        import pywt
        coeffs = pywt.wavedec(signal, "db4", level=5)
        energies = np.array([np.sum(c ** 2) for c in coeffs])
        energies = energies / (energies.sum() + 1e-12) * 100
        labels = ["cA5"] + [f"cD{5-i}" for i in range(len(energies) - 1)]
        ax.bar(labels, energies, color="#2ca02c", edgecolor="k", linewidth=0.4)
        ax.set_xlabel("Sub-band", fontsize=10)
        ax.set_ylabel("Energy (%)", fontsize=10)
        ax.set_title("DWT Sub-band Energy", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    def _plot_channel_rms(self, ax: plt.Axes, signal: np.ndarray) -> None:
        labels = [r"$V_a$", r"$V_b$", r"$V_c$", r"$I_a$", r"$I_b$", r"$I_c$"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        rms = np.sqrt(np.mean(signal ** 2, axis=-1))
        n = len(rms)
        ax.bar(labels[:n], rms, color=colors[:n], edgecolor="k", linewidth=0.4)
        ax.set_ylabel("RMS", fontsize=10)
        ax.set_title("Per-Channel RMS", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    def _plot_harmonics(self, ax: plt.Axes, signal: np.ndarray) -> None:
        n = len(signal)
        win = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(signal * win) / (win.sum() / 2))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.f_sample)

        fund_idx = int(np.argmin(np.abs(freqs - self.f_fund)))
        fund_amp = spectrum[fund_idx] + 1e-12

        orders = list(range(1, 11))
        amps = []
        for h in orders:
            idx = int(np.argmin(np.abs(freqs - h * self.f_fund)))
            amps.append(100.0 * spectrum[idx] / fund_amp)

        colors = ["#1f77b4" if h == 1 else "#ff7f0e" for h in orders]
        ax.bar([str(h) for h in orders], amps, color=colors, edgecolor="k", linewidth=0.4)
        ax.axhline(5.0, color="#d62728", linestyle="--", linewidth=0.8, label="IEEE 519 5%")
        ax.set_xlabel("Harmonic Order", fontsize=10)
        ax.set_ylabel("% of fundamental", fontsize=10)
        ax.set_title("Harmonic Content", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    def _plot_phase_portrait(self, ax: plt.Axes, signal: np.ndarray) -> None:
        """Phase portrait: Ia vs Ib (lissajous figure)."""
        if signal.shape[0] < 5:
            ax.set_visible(False)
            return
        ia, ib = signal[3], signal[4]
        ax.plot(ia, ib, linewidth=0.5, color="#9467bd", alpha=0.7)
        ax.set_xlabel(r"$I_a$ (A)", fontsize=10)
        ax.set_ylabel(r"$I_b$ (A)", fontsize=10)
        ax.set_title("Phase Portrait (Ia vs Ib)", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_aspect("equal", "box")
