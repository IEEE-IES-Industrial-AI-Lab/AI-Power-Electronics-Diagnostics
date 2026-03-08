"""
Short-Time Fourier Transform (STFT) spectrogram generation.

Converts 1-D time-domain signals into 2-D time-frequency representations
suitable for input to convolutional neural networks.

Features:
  - Standard STFT spectrogram (linear or log scale)
  - Mel-scale spectrogram
  - Fixed-size output via bilinear resize
  - Batch processing for multi-channel signals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import stft as scipy_stft


@dataclass
class SpectrogramResult:
    """Container for STFT spectrogram output."""
    spectrogram: np.ndarray      # (freq_bins, time_frames) magnitude spectrogram
    frequencies: np.ndarray      # Frequency axis [Hz]
    times: np.ndarray            # Time axis [s]
    log_scale: bool              # Whether spectrogram is in log (dB) scale


class STFTSpectrogram:
    """Generate STFT spectrograms from 1-D power electronics signals.

    Parameters
    ----------
    f_sample : float
        Sampling frequency [Hz].
    n_fft : int
        FFT size (also segment length).
    hop_length : int or None
        Hop length in samples. Defaults to ``n_fft // 4``.
    window : str
        Window function: 'hann', 'hamming', 'blackman'.
    output_size : tuple(int, int) or None
        If given, resize spectrogram to (freq, time) via bilinear interpolation.
        Useful for fixed-size CNN input (e.g., (128, 128)).
    log_scale : bool
        Convert amplitude to dB scale: 20·log10(amp + eps).

    Examples
    --------
    >>> stft = STFTSpectrogram(f_sample=50_000.0, output_size=(128, 128))
    >>> result = stft.compute(signal)
    >>> print(result.spectrogram.shape)  # (128, 128)
    """

    def __init__(
        self,
        f_sample: float = 50_000.0,
        n_fft: int = 512,
        hop_length: Optional[int] = None,
        window: str = "hann",
        output_size: Optional[Tuple[int, int]] = (128, 128),
        log_scale: bool = True,
    ) -> None:
        self.f_sample = f_sample
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.window = window
        self.output_size = output_size
        self.log_scale = log_scale

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self, signal: np.ndarray) -> SpectrogramResult:
        """Compute STFT spectrogram for a 1-D signal.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)

        Returns
        -------
        SpectrogramResult
        """
        freqs, times, Zxx = scipy_stft(
            signal,
            fs=self.f_sample,
            window=self.window,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            return_onesided=True,
        )
        magnitude = np.abs(Zxx)  # (freq_bins, time_frames)

        if self.log_scale:
            magnitude = 20 * np.log10(magnitude + 1e-9)

        if self.output_size is not None:
            magnitude = self._resize(magnitude, self.output_size)

        return SpectrogramResult(
            spectrogram=magnitude.astype(np.float32),
            frequencies=freqs,
            times=times,
            log_scale=self.log_scale,
        )

    def compute_batch(self, signals: np.ndarray) -> np.ndarray:
        """Compute spectrograms for a batch of multi-channel signals.

        Parameters
        ----------
        signals : np.ndarray, shape (N, C, T)
            Batch of N samples, C channels, T time steps.

        Returns
        -------
        spectrograms : np.ndarray, shape (N, C, H, W)
            Where H×W = ``output_size`` or (n_fft//2+1, time_frames).
        """
        N, C, T = signals.shape
        specs = []
        for n in range(N):
            channel_specs = []
            for c in range(C):
                result = self.compute(signals[n, c])
                channel_specs.append(result.spectrogram)
            specs.append(np.stack(channel_specs))
        return np.stack(specs)  # (N, C, H, W)

    def compute_multichannel(self, signal: np.ndarray) -> np.ndarray:
        """Compute spectrogram for each channel of a multi-channel signal.

        Parameters
        ----------
        signal : np.ndarray, shape (C, T)

        Returns
        -------
        spectrograms : np.ndarray, shape (C, H, W)
        """
        return np.stack([self.compute(signal[c]).spectrogram for c in range(signal.shape[0])])

    def mel_spectrogram(
        self,
        signal: np.ndarray,
        n_mels: int = 128,
    ) -> np.ndarray:
        """Convert STFT magnitude to mel-scale spectrogram.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
        n_mels : int
            Number of mel filter banks.

        Returns
        -------
        mel_spec : np.ndarray, shape (n_mels, time_frames)
        """
        result = self.compute(signal)
        # Use linear magnitude (undo log if applied)
        if self.log_scale:
            magnitude = 10 ** (result.spectrogram / 20)
        else:
            magnitude = result.spectrogram

        mel_fb = self._mel_filterbank(n_mels, magnitude.shape[0])
        mel_spec = mel_fb @ magnitude  # (n_mels, time_frames)

        if self.log_scale:
            mel_spec = 20 * np.log10(mel_spec + 1e-9)

        if self.output_size is not None:
            mel_spec = self._resize(mel_spec, self.output_size)

        return mel_spec.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mel_filterbank(self, n_mels: int, n_fft_bins: int) -> np.ndarray:
        """Construct a mel filter bank matrix.

        Returns
        -------
        filterbank : np.ndarray, shape (n_mels, n_fft_bins)
        """
        f_min, f_max = 0.0, self.f_sample / 2

        def hz_to_mel(f: float) -> float:
            return 2595 * np.log10(1 + f / 700)

        def mel_to_hz(m: float) -> float:
            return 700 * (10 ** (m / 2595) - 1)

        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = np.array([mel_to_hz(m) for m in mel_points])

        fft_freqs = np.linspace(0, self.f_sample / 2, n_fft_bins)
        filterbank = np.zeros((n_mels, n_fft_bins))
        for m in range(1, n_mels + 1):
            f_m_minus = hz_points[m - 1]
            f_m = hz_points[m]
            f_m_plus = hz_points[m + 1]

            for k, f in enumerate(fft_freqs):
                if f_m_minus <= f < f_m:
                    filterbank[m - 1, k] = (f - f_m_minus) / (f_m - f_m_minus)
                elif f_m <= f <= f_m_plus:
                    filterbank[m - 1, k] = (f_m_plus - f) / (f_m_plus - f_m)

        return filterbank

    @staticmethod
    def _resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Bilinear resize a 2-D array to ``size = (rows, cols)``."""
        from scipy.ndimage import zoom
        zoom_r = size[0] / image.shape[0]
        zoom_c = size[1] / image.shape[1]
        return zoom(image, (zoom_r, zoom_c), order=1)

    @property
    def freq_resolution(self) -> float:
        """Frequency bin spacing [Hz]."""
        return self.f_sample / self.n_fft

    @property
    def time_resolution(self) -> float:
        """Time frame spacing [s]."""
        return self.hop_length / self.f_sample
