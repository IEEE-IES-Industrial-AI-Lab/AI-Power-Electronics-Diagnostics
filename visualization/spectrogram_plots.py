"""
Frequency-domain and spectrogram visualization.

Provides:
  - FFT amplitude spectrum plot
  - STFT spectrogram (time-frequency heatmap)
  - CWT scalogram
  - Harmonic bar chart
  - DWT sub-band energy visualization
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


class SpectrogramPlotter:
    """Visualization tools for frequency-domain signal analysis.

    Parameters
    ----------
    f_sample : float
        Sampling frequency [Hz].
    figsize : tuple
        Default figure size.
    colormap : str
        Colormap for spectrograms. 'viridis' or 'jet' are standard choices.
    """

    def __init__(
        self,
        f_sample: float = 50_000.0,
        figsize: Tuple[float, float] = (12, 5),
        colormap: str = "viridis",
    ) -> None:
        self.f_sample = f_sample
        self.figsize = figsize
        self.colormap = colormap

    # ------------------------------------------------------------------
    # FFT spectrum
    # ------------------------------------------------------------------

    def plot_fft(
        self,
        signal: np.ndarray,
        f_max: Optional[float] = None,
        title: str = "FFT Amplitude Spectrum",
        mark_harmonics: Optional[float] = None,
        n_harmonics: int = 10,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot single-sided FFT amplitude spectrum.

        Parameters
        ----------
        f_max : float or None
            Maximum frequency to display. Defaults to Nyquist.
        mark_harmonics : float or None
            If given, mark harmonic lines at N × mark_harmonics Hz.
        """
        n = len(signal)
        win = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(signal * win) / (win.sum() / 2))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.f_sample)

        if f_max is None:
            f_max = self.f_sample / 2

        mask = freqs <= f_max
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(freqs[mask], spectrum[mask], linewidth=0.8, color="#1f77b4")

        if mark_harmonics is not None:
            for h in range(1, n_harmonics + 1):
                f_h = h * mark_harmonics
                if f_h <= f_max:
                    ax.axvline(f_h, color="#d62728", alpha=0.5, linestyle="--",
                               linewidth=0.8, label=f"H{h}" if h <= 5 else "")

        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Amplitude", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(0, f_max)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_yscale("log")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_psd(
        self,
        signal: np.ndarray,
        nperseg: int = 1024,
        title: str = "Power Spectral Density (Welch)",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot Power Spectral Density using Welch's method."""
        from scipy.signal import welch
        freqs, psd = welch(signal, fs=self.f_sample, nperseg=nperseg)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.semilogy(freqs, psd, linewidth=0.9, color="#2ca02c")
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("PSD (V²/Hz or A²/Hz)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # STFT spectrogram
    # ------------------------------------------------------------------

    def plot_spectrogram(
        self,
        signal: np.ndarray,
        n_fft: int = 512,
        hop_length: Optional[int] = None,
        f_max: Optional[float] = None,
        log_scale: bool = True,
        title: str = "STFT Spectrogram",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot STFT magnitude spectrogram."""
        from scipy.signal import stft as scipy_stft
        hop = hop_length or n_fft // 4
        freqs, times, Zxx = scipy_stft(
            signal, fs=self.f_sample,
            nperseg=n_fft, noverlap=n_fft - hop,
            window="hann",
        )
        magnitude = np.abs(Zxx)

        if log_scale:
            magnitude = 20 * np.log10(magnitude + 1e-9)

        if f_max is not None:
            freq_mask = freqs <= f_max
            freqs = freqs[freq_mask]
            magnitude = magnitude[freq_mask]

        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.pcolormesh(
            times * 1000, freqs, magnitude,
            shading="gouraud", cmap=self.colormap,
        )
        plt.colorbar(im, ax=ax, label="dB" if log_scale else "Amplitude")
        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.set_ylabel("Frequency (Hz)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_spectrogram_comparison(
        self,
        signals: Dict[str, np.ndarray],
        n_fft: int = 512,
        f_max: float = 5000.0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot STFT spectrograms for multiple fault types side by side."""
        from scipy.signal import stft as scipy_stft
        n = len(signals)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharex=False)
        if n == 1:
            axes = [axes]

        hop = n_fft // 4
        for ax, (fault_name, signal) in zip(axes, signals.items()):
            freqs, times, Zxx = scipy_stft(
                signal, fs=self.f_sample,
                nperseg=n_fft, noverlap=n_fft - hop, window="hann",
            )
            magnitude = 20 * np.log10(np.abs(Zxx) + 1e-9)
            mask = freqs <= f_max
            im = ax.pcolormesh(
                times * 1000, freqs[mask], magnitude[mask],
                shading="gouraud", cmap=self.colormap,
            )
            plt.colorbar(im, ax=ax, label="dB")
            ax.set_title(fault_name.replace("_", " ").title(), fontsize=10)
            ax.set_xlabel("Time (ms)", fontsize=9)
            ax.set_ylabel("Freq (Hz)", fontsize=9)

        fig.suptitle("Spectrogram Comparison by Fault Type",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # CWT scalogram
    # ------------------------------------------------------------------

    def plot_scalogram(
        self,
        signal: np.ndarray,
        scales: Optional[np.ndarray] = None,
        wavelet: str = "cmor1.5-1.0",
        title: str = "CWT Scalogram",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot Continuous Wavelet Transform scalogram."""
        import pywt
        if scales is None:
            scales = np.arange(1, 129)

        coefficients, freqs = pywt.cwt(signal, scales, wavelet, 1.0 / self.f_sample)
        scalogram = np.abs(coefficients)

        t = np.arange(len(signal)) / self.f_sample * 1000

        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.pcolormesh(
            t, freqs, scalogram, shading="gouraud", cmap=self.colormap
        )
        plt.colorbar(im, ax=ax, label="Magnitude")
        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.set_ylabel("Frequency (Hz)", fontsize=12)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Harmonic bar chart
    # ------------------------------------------------------------------

    def plot_harmonics(
        self,
        harmonics: Dict[int, float],
        f_fund: float,
        title: str = "Harmonic Spectrum",
        thd_f: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Bar chart of harmonic amplitudes normalized to fundamental.

        Parameters
        ----------
        harmonics : dict {order: amplitude}
        """
        orders = sorted(harmonics.keys())
        amplitudes = [harmonics[o] for o in orders]
        fundamental = harmonics.get(1, 1.0) + 1e-12
        normalized = [a / fundamental * 100 for a in amplitudes]

        colors = ["#1f77b4" if o == 1 else "#ff7f0e" if o % 2 == 0 else "#2ca02c"
                  for o in orders]

        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.bar([str(o) for o in orders], normalized, color=colors, edgecolor="k",
                      linewidth=0.5)
        ax.set_xlabel("Harmonic Order", fontsize=12)
        ax.set_ylabel("Amplitude (% of fundamental)", fontsize=12)
        label = f"{title}"
        if thd_f is not None:
            label += f" | THD-F = {thd_f:.2f}%"
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

        # IEEE 519 limit reference lines
        ax.axhline(5.0, color="#d62728", linestyle="--", linewidth=1.0,
                   label="IEEE 519 limit (5%)")
        ax.legend(fontsize=9)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # DWT sub-band energy
    # ------------------------------------------------------------------

    def plot_dwt_energy(
        self,
        energies: np.ndarray,
        title: str = "DWT Sub-band Energy",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Bar chart of relative DWT sub-band energies."""
        levels = [f"cA{len(energies)-1}"] + [f"cD{len(energies)-1-i}" for i in range(len(energies)-1)]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(levels, energies * 100, color="#1f77b4", edgecolor="k", linewidth=0.5)
        ax.set_xlabel("DWT Sub-band", fontsize=12)
        ax.set_ylabel("Relative Energy (%)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
