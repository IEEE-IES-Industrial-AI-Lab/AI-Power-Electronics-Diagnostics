"""
Training utilities: early stopping, checkpointing, LR scheduling helpers,
data loaders, and miscellaneous training aids.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when a monitored metric has stopped improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement after which training stops.
    min_delta : float
        Minimum change to qualify as an improvement.
    monitor : str
        Metric direction: 'min' (val_loss) or 'max' (val_acc).
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        monitor: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_value = float("inf") if monitor == "min" else float("-inf")
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Update and return True if training should stop."""
        improved = (
            value < self.best_value - self.min_delta
            if self.monitor == "min"
            else value > self.best_value + self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self.patience,
                )

        return self.should_stop


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """Save and load model checkpoints.

    Keeps track of the best checkpoint based on a monitored metric,
    plus an optional 'last' checkpoint.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        experiment_name: str,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.dir = Path(checkpoint_dir) / experiment_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_path: Optional[Path] = None

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        is_last: bool = False,
    ) -> Optional[Path]:
        """Save checkpoint if it improves the monitored metric.

        Returns
        -------
        path : Path or None — path where checkpoint was saved, or None.
        """
        value = metrics.get(self.monitor, None)
        is_best = False

        if value is not None:
            is_best = (
                value < self.best_value if self.mode == "min" else value > self.best_value
            )
            if is_best:
                self.best_value = value

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        saved_path = None
        if is_best:
            path = self.dir / "best.pt"
            torch.save(state, path)
            self.best_path = path
            saved_path = path
            logger.info("Epoch %d: New best %s=%.4f → saved to %s", epoch, self.monitor, value, path)

        if is_last:
            path = self.dir / "last.pt"
            torch.save(state, path)
            saved_path = saved_path or path

        return saved_path

    def load_best(self, model: nn.Module, device: str = "cpu") -> Dict:
        """Load the best checkpoint into ``model``."""
        if self.best_path is None or not self.best_path.exists():
            raise FileNotFoundError("No best checkpoint found.")
        state = torch.load(self.best_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        return state


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 4,
    class_weights: str = "balanced",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders from numpy arrays.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    X_val_t = torch.from_numpy(X_val.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.int64))
    X_test_t = torch.from_numpy(X_test.astype(np.float32))
    y_test_t = torch.from_numpy(y_test.astype(np.int64))

    train_sampler = None
    if class_weights == "balanced":
        train_sampler = _make_balanced_sampler(y_train_t)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def _make_balanced_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for class-balanced mini-batches."""
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / (class_counts.float() + 1e-8)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to a torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Construct optimizer by name."""
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": lambda params, **kw: torch.optim.SGD(params, momentum=0.9, **kw),
    }
    cls = optimizers.get(optimizer_name.lower())
    if cls is None:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return cls(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    config: Dict,
    epochs: int,
) -> Optional[Any]:
    """Construct LR scheduler from config dict."""
    name = scheduler_name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("T_max", epochs),
            eta_min=config.get("eta_min", 1e-6),
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 30),
            gamma=config.get("gamma", 0.1),
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=config.get("patience", 10),
            factor=config.get("factor", 0.5),
        )
    return None


class AverageMeter:
    """Running mean tracker for a scalar metric (loss, accuracy, etc.)."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def generate_experiment_name(model_name: str, dataset: str) -> str:
    """Generate a unique experiment name with timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{dataset}_{timestamp}"
