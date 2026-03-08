"""
Time-domain waveform visualization for power electronics signals.

Provides publication-quality plots suitable for IEEE papers:
  - 3-phase voltage / current waveforms with fault overlays
  - Side-by-side healthy vs. faulty comparison
  - Phase current imbalance visualization
  - Attention weight overlay on waveforms
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# IEEE-style default aesthetics
COLORS = {
    "phase_a": "#1f77b4",
    "phase_b": "#ff7f0e",
    "phase_c": "#2ca02c",
    "fault_region": "#d62728",
    "healthy": "#2ca02c",
    "fault": "#d62728",
    "neutral": "#7f7f7f",
}

PHASE_LABELS = {
    0: r"$V_a$", 1: r"$V_b$", 2: r"$V_c$",
    3: r"$I_a$", 4: r"$I_b$", 5: r"$I_c$",
}


class WaveformPlotter:
    """Generate time-domain waveform plots for power electronics signals.

    Parameters
    ----------
    f_sample : float
        Sampling frequency [Hz] — used to build the time axis.
    figsize : tuple(float, float)
        Default figure size in inches.
    style : str
        Matplotlib style sheet. Use 'default' or 'seaborn-v0_8-whitegrid'.
    """

    def __init__(
        self,
        f_sample: float = 50_000.0,
        figsize: Tuple[float, float] = (14, 8),
        style: str = "default",
    ) -> None:
        self.f_sample = f_sample
        self.figsize = figsize
        try:
            plt.style.use(style)
        except OSError:
            pass  # Fallback to default if style not found

    # ------------------------------------------------------------------
    # Core plots
    # ------------------------------------------------------------------

    def plot_three_phase(
        self,
        signal: np.ndarray,
        title: str = "3-Phase Signal",
        channel_labels: Optional[List[str]] = None,
        fault_regions: Optional[List[Tuple[int, int]]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot up to 6 signal channels (Va/Vb/Vc and Ia/Ib/Ic).

        Parameters
        ----------
        signal : np.ndarray, shape (C, T)
        fault_regions : list of (start, end) sample indices to shade red
        """
        C, T = signal.shape
        t = np.arange(T) / self.f_sample * 1000  # ms

        n_rows = 2 if C >= 6 else 1
        n_cols = min(C, 3)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize, sharex=True)
        axes = np.array(axes).reshape(n_rows, n_cols)

        phase_colors = [COLORS["phase_a"], COLORS["phase_b"], COLORS["phase_c"]]
        labels = channel_labels or [PHASE_LABELS.get(c, f"Ch{c}") for c in range(C)]

        for c in range(C):
            row = c // 3
            col = c % 3
            ax = axes[row, col] if n_rows > 1 else axes[0, col]

            ax.plot(t, signal[c], color=phase_colors[col], linewidth=0.8, label=labels[c])

            if fault_regions:
                for start, end in fault_regions:
                    t_start = start / self.f_sample * 1000
                    t_end = end / self.f_sample * 1000
                    ax.axvspan(t_start, t_end, alpha=0.2, color=COLORS["fault_region"],
                               label="Fault region" if c == 0 else "")

            ax.set_ylabel(labels[c], fontsize=11)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(loc="upper right", fontsize=9)

        for ax in axes[-1]:
            ax.set_xlabel("Time (ms)", fontsize=11)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_fault_comparison(
        self,
        healthy_signal: np.ndarray,
        faulty_signal: np.ndarray,
        fault_name: str,
        channel_idx: int = 3,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Side-by-side healthy vs. faulty signal comparison.

        Parameters
        ----------
        channel_idx : int
            Which channel to compare (default 3 = Ia).
        """
        T = healthy_signal.shape[1]
        t = np.arange(T) / self.f_sample * 1000

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        ch_label = PHASE_LABELS.get(channel_idx, f"Ch{channel_idx}")

        ax1.plot(t, healthy_signal[channel_idx], color=COLORS["healthy"],
                 linewidth=0.8, label="Healthy")
        ax1.set_ylabel(f"{ch_label} — Healthy", fontsize=11)
        ax1.set_title("Healthy Signal", fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.legend(fontsize=9)

        ax2.plot(t, faulty_signal[channel_idx], color=COLORS["fault"],
                 linewidth=0.8, label=fault_name)
        ax2.set_ylabel(f"{ch_label} — {fault_name}", fontsize=11)
        ax2.set_xlabel("Time (ms)", fontsize=11)
        ax2.set_title(f"Fault: {fault_name}", fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle="--")
        ax2.legend(fontsize=9)

        fig.suptitle(f"Fault Comparison: Healthy vs. {fault_name}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_all_fault_types(
        self,
        signals_dict: Dict[str, np.ndarray],
        channel_idx: int = 3,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Grid plot of all fault type waveforms for one channel.

        Parameters
        ----------
        signals_dict : dict mapping fault_name → signal array (C, T)
        """
        n = len(signals_dict)
        n_cols = 3
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), sharex=True)
        axes = np.array(axes).flatten()

        for i, (fault_name, signal) in enumerate(signals_dict.items()):
            T = signal.shape[-1]
            t = np.arange(T) / self.f_sample * 1000
            color = COLORS["healthy"] if fault_name == "healthy" else COLORS["fault"]
            axes[i].plot(t, signal[channel_idx], color=color, linewidth=0.7)
            axes[i].set_title(fault_name.replace("_", " ").title(), fontsize=10)
            axes[i].grid(True, alpha=0.3, linestyle="--")
            axes[i].set_ylabel(PHASE_LABELS.get(channel_idx, f"Ch{channel_idx}"), fontsize=9)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        for ax in axes[max(0, n - n_cols) : n]:
            ax.set_xlabel("Time (ms)", fontsize=9)

        ch_label = PHASE_LABELS.get(channel_idx, f"Ch{channel_idx}")
        fig.suptitle(f"All Fault Types — Channel {ch_label}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_attention_overlay(
        self,
        signal: np.ndarray,
        attention_weights: np.ndarray,
        channel_idx: int = 3,
        title: str = "BiLSTM Attention",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Overlay LSTM attention weights on the waveform.

        Parameters
        ----------
        attention_weights : np.ndarray, shape (T,) — normalized to [0, 1]
        """
        T = len(signal[channel_idx])
        t = np.arange(T) / self.f_sample * 1000

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})

        ax1.plot(t, signal[channel_idx], color=COLORS["phase_a"], linewidth=0.8)
        ax1.set_ylabel(PHASE_LABELS.get(channel_idx, "Signal"), fontsize=11)
        ax1.set_title(title, fontsize=13, fontweight="bold")
        ax1.grid(True, alpha=0.3, linestyle="--")

        # Attention heatmap
        weights = np.interp(
            np.arange(T),
            np.linspace(0, T - 1, len(attention_weights)),
            attention_weights,
        )
        ax2.fill_between(t, weights, alpha=0.7, color=COLORS["fault_region"])
        ax2.set_ylabel("Attention", fontsize=10)
        ax2.set_xlabel("Time (ms)", fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_rms_trend(
        self,
        rms_values: np.ndarray,
        fault_onset: Optional[int] = None,
        title: str = "Current RMS Trend",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot temporal RMS trend for thermal fault monitoring."""
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(rms_values, color=COLORS["phase_a"], linewidth=1.2, label="Current RMS")

        if fault_onset is not None:
            ax.axvline(fault_onset, color=COLORS["fault_region"], linestyle="--",
                       linewidth=1.5, label=f"Fault onset (window {fault_onset})")

        ax.set_xlabel("Window Index", fontsize=11)
        ax.set_ylabel("RMS [A]", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
