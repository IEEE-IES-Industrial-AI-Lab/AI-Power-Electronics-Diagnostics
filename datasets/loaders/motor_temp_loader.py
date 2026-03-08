"""
Loader for the Kaggle Electric Motor Temperature dataset.

Dataset: "Electric Motor Temperature"
Source  : https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature
Paper   : Kirchgässner, W., Wallscheid, O., & Böcker, J. (2019).
          Empirical evaluation of exponentially weighted moving averages
          for simple linear thermal modeling of permanent magnet
          synchronous machines. IEEE IEMDC 2019.

Contains multi-variate time-series measurements from a PMSM on a
test bench:
  - u_q, u_d  : Stator voltages [V]
  - i_q, i_d  : Stator currents [A]
  - motor_speed: [rpm]
  - torque     : [Nm]
  - ambient    : Ambient temperature [°C]
  - coolant    : Coolant temperature [°C]
  - pm         : Permanent magnet temperature (TARGET) [°C]
  - stator_winding / stator_yoke / stator_tooth : temperatures [°C]

This loader frames the dataset as a FAULT DETECTION problem by
thresholding the permanent magnet temperature:
  pm < T_normal  → healthy (0)
  pm >= T_normal → overtemperature (1)

Usage
-----
>>> from datasets.loaders.motor_temp_loader import MotorTemperatureLoader
>>> loader = MotorTemperatureLoader(data_dir="datasets/raw/motor_temp")
>>> X, y = loader.load()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from datasets.loaders.base_loader import BaseDatasetLoader

logger = logging.getLogger(__name__)

# Columns used as input features (electrical + thermal signals)
FEATURE_COLUMNS = ["u_q", "u_d", "i_q", "i_d", "motor_speed", "torque",
                   "ambient", "coolant"]
TARGET_COLUMN = "pm"  # Permanent magnet temperature


class MotorTemperatureLoader(BaseDatasetLoader):
    """Load and window the Kaggle Electric Motor Temperature dataset.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``measures_v2.csv`` (the Kaggle dataset file).
    window_size : int
        Samples per input window (default 1024, at 2 Hz → ~512 s windows).
    normalize : bool
        Z-score normalize each feature channel.
    temp_threshold : float
        PM temperature threshold [°C] for healthy/overtemperature split.
    hop_size : int or None
        Sliding window step. Defaults to ``window_size // 2`` (50% overlap).
    """

    DATASET_FILE = "measures_v2.csv"
    CLASS_NAMES = ["healthy", "overtemperature"]

    def __init__(
        self,
        data_dir: str | Path = "datasets/raw/motor_temp",
        window_size: int = 1024,
        normalize: bool = True,
        temp_threshold: float = 100.0,
        hop_size: Optional[int] = None,
    ) -> None:
        super().__init__(data_dir, window_size, normalize)
        self.temp_threshold = temp_threshold
        self.hop_size = hop_size or window_size // 2

    # ------------------------------------------------------------------
    # BaseDatasetLoader interface
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> List[str]:
        return self.CLASS_NAMES

    def info(self) -> Dict:
        return {
            "name": "Electric Motor Temperature",
            "source": "https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature",
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "task": "Binary fault detection (overtemperature)",
            "temp_threshold_C": self.temp_threshold,
            "window_size": self.window_size,
        }

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load, window, and label the dataset.

        Returns
        -------
        X : np.ndarray, shape (N, n_features, window_size)
        y : np.ndarray, shape (N,)  — 0 = healthy, 1 = overtemperature
        """
        csv_path = self.data_dir / self.DATASET_FILE
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found at {csv_path}.\n"
                f"Download it via:\n"
                f"  bash datasets/download_scripts/download_motor_temp.sh\n"
                f"or run:\n"
                f"  python datasets/download_scripts/setup_datasets.py --dataset motor_temp"
            )

        logger.info("Loading %s ...", csv_path)
        df = pd.read_csv(csv_path)

        # Validate required columns
        missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        X_list, y_list = [], []

        # Process each measurement profile_id independently to avoid cross-session windows
        profile_col = "profile_id" if "profile_id" in df.columns else None
        groups = df.groupby(profile_col) if profile_col else [("all", df)]

        for _, group in groups:
            features = group[FEATURE_COLUMNS].values.T.astype(np.float32)  # (C, T)
            labels_raw = (group[TARGET_COLUMN].values >= self.temp_threshold).astype(np.int64)

            windows = self.extract_windows(features, self.window_size, self.hop_size)
            # Majority-vote label per window
            for i, win in enumerate(windows):
                start = i * self.hop_size
                end = start + self.window_size
                window_labels = labels_raw[start : end]
                y_win = int(window_labels.mean() >= 0.5)
                X_list.append(win)
                y_list.append(y_win)

        X = np.stack(X_list)
        y = np.array(y_list, dtype=np.int64)

        if self.normalize:
            X = self.normalize_signals(X)

        logger.info(
            "Loaded %d windows | healthy: %d | overtemperature: %d",
            len(y), (y == 0).sum(), (y == 1).sum(),
        )
        return X, y

    # ------------------------------------------------------------------
    # Regression mode (predict PM temperature directly)
    # ------------------------------------------------------------------

    def load_regression(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset for PM temperature regression.

        Returns
        -------
        X : np.ndarray, shape (N, n_features, window_size)
        y : np.ndarray, shape (N,)  — mean PM temperature per window [°C]
        """
        csv_path = self.data_dir / self.DATASET_FILE
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {csv_path}.")

        df = pd.read_csv(csv_path)
        X_list, y_list = [], []

        profile_col = "profile_id" if "profile_id" in df.columns else None
        groups = df.groupby(profile_col) if profile_col else [("all", df)]

        for _, group in groups:
            features = group[FEATURE_COLUMNS].values.T.astype(np.float32)
            temps = group[TARGET_COLUMN].values.astype(np.float32)

            windows = self.extract_windows(features, self.window_size, self.hop_size)
            for i, win in enumerate(windows):
                start = i * self.hop_size
                end = start + self.window_size
                y_win = float(temps[start:end].mean())
                X_list.append(win)
                y_list.append(y_win)

        X = np.stack(X_list)
        y = np.array(y_list, dtype=np.float32)

        if self.normalize:
            X = self.normalize_signals(X)

        return X, y
