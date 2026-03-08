"""
Abstract base class for all dataset loaders in this repository.

Every concrete loader must implement:
  - ``load()``     : return (X, y) numpy arrays
  - ``info()``     : return a dict describing the dataset
  - ``class_names``: property listing class label strings

The unified interface allows training pipelines to swap datasets
without modifying model code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DatasetSplit:
    """Container for a train/val/test split."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: List[str] = field(default_factory=list)


class BaseDatasetLoader(ABC):
    """Abstract base class for power electronics dataset loaders.

    Parameters
    ----------
    data_dir : str or Path
        Root directory where the dataset is stored (or will be downloaded).
    window_size : int
        Number of samples per input window.
    normalize : bool
        Whether to z-score normalize signals per channel.
    """

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int = 1024,
        normalize: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the full dataset.

        Returns
        -------
        X : np.ndarray, shape (N, C, window_size)
        y : np.ndarray, shape (N,)  — integer class labels
        """

    @abstractmethod
    def info(self) -> Dict:
        """Return a dictionary with dataset metadata."""

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """List of class name strings ordered by label index."""

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        random_seed: int = 42,
    ) -> DatasetSplit:
        """Stratified train / val / test split.

        Parameters
        ----------
        train_frac : float
            Fraction of data for training.
        val_frac : float
            Fraction of data for validation (remainder goes to test).
        """
        from sklearn.model_selection import train_test_split

        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=1 - train_frac, stratify=y, random_state=random_seed
        )
        relative_val = val_frac / (1 - train_frac)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=1 - relative_val, stratify=y_tmp, random_state=random_seed
        )
        return DatasetSplit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            class_names=self.class_names,
        )

    @staticmethod
    def normalize_signals(X: np.ndarray) -> np.ndarray:
        """Z-score normalize each channel independently.

        Parameters
        ----------
        X : np.ndarray, shape (N, C, T)
        """
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True) + 1e-8
        return (X - mean) / std

    @staticmethod
    def extract_windows(
        signal: np.ndarray,
        window_size: int,
        hop_size: Optional[int] = None,
    ) -> np.ndarray:
        """Sliding-window segmentation of a (C, T) signal.

        Parameters
        ----------
        signal : np.ndarray, shape (C, T)
        window_size : int
        hop_size : int or None
            Step between windows. Defaults to ``window_size`` (no overlap).

        Returns
        -------
        windows : np.ndarray, shape (n_windows, C, window_size)
        """
        if hop_size is None:
            hop_size = window_size
        C, T = signal.shape
        starts = range(0, T - window_size + 1, hop_size)
        windows = np.stack([signal[:, s : s + window_size] for s in starts])
        return windows

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"data_dir={self.data_dir}, "
            f"window_size={self.window_size}, "
            f"normalize={self.normalize})"
        )
