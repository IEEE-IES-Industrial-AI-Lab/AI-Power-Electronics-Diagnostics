"""
Automated benchmark runner for all AI Power Electronics Diagnostics models.

Trains each model on each dataset with N independent runs and records:
  - Accuracy, Macro F1, Weighted F1, Cohen Kappa
  - Training time per epoch
  - Model parameter count
  - Mean ± std across runs

Results are saved to ``benchmarks/results/benchmark_results.csv``.

Usage
-----
  # Run full benchmark (may take several hours on CPU)
  python benchmarks/benchmark_all_models.py

  # Quick smoke-test (3 epochs, 50 samples/class)
  python benchmarks/benchmark_all_models.py --quick

  # Single model
  python benchmarks/benchmark_all_models.py --models cnn_waveform --datasets inverter_synthetic

  # Help
  python benchmarks/benchmark_all_models.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    fault_domain: str,
    n_per_class: int,
    window_size: int,
    random_seed: int,
):
    """Return (X, y, class_names) for a given fault domain."""
    if fault_domain == "inverter":
        from datasets.synthetic import InverterFaultSimulator
        sim = InverterFaultSimulator()
        sim.cfg.random_seed = random_seed
        X, y = sim.generate_dataset(n_per_class=n_per_class, window_size=window_size)
        class_names = list(sim.fault_labels().keys())
    else:
        from datasets.synthetic import MotorDriveSimulator
        sim = MotorDriveSimulator()
        sim.cfg.random_seed = random_seed
        X, y = sim.generate_dataset(n_per_class=n_per_class, window_size=window_size)
        class_names = list(sim.fault_labels().keys())

    return X, y, class_names


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_name: str, n_channels: int, n_classes: int, window_size: int):
    from models import MODEL_REGISTRY
    kwargs = dict(n_channels=n_channels, n_classes=n_classes)
    if model_name in ("cnn_waveform", "transformer"):
        kwargs["window_size"] = window_size
    return MODEL_REGISTRY[model_name](**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    device: torch.device,
    random_seed: int = 42,
) -> Dict:
    """Train one model on one dataset and return metrics dict."""
    n_channels = X.shape[1]
    n_classes = len(np.unique(y))
    window_size = X.shape[-1]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=random_seed
    )

    def make_loader(Xd, yd, shuffle=False):
        return DataLoader(
            TensorDataset(
                torch.from_numpy(Xd.astype(np.float32)),
                torch.from_numpy(yd.astype(np.int64)),
            ),
            batch_size=cfg["batch_size"],
            shuffle=shuffle,
            num_workers=min(cfg["num_workers"], 2),
            drop_last=False,
        )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    test_loader = make_loader(X_test, y_test)

    model = build_model(model_name, n_channels, n_classes, window_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )

    t_start = time.time()
    epoch_times = []

    for epoch in range(cfg["epochs"]):
        t_ep = time.time()
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        epoch_times.append(time.time() - t_ep)

    total_time = time.time() - t_start

    # Evaluation
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            logits = model(X_b.to(device))
            preds.extend(logits.argmax(-1).cpu().numpy().tolist())
            labels.extend(y_b.numpy().tolist())

    y_true = np.array(labels)
    y_pred = np.array(preds)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "total_train_time_s": round(total_time, 2),
        "avg_epoch_time_s": round(float(np.mean(epoch_times)), 3),
        "n_params": model.count_parameters(),
        "n_test_samples": len(y_true),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(args: argparse.Namespace) -> None:
    cfg_path = REPO_ROOT / "benchmarks" / "configs" / "benchmark_config.yaml"
    with open(cfg_path) as f:
        full_cfg = yaml.safe_load(f)

    bm = full_cfg["benchmark"]
    if args.quick:
        bm["epochs"] = 5
        bm["n_per_class"] = 50
        bm["n_runs"] = 1
        logger.info("Quick mode: epochs=5, n_per_class=50, n_runs=1")

    results_dir = REPO_ROOT / full_cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / full_cfg["output"]["csv_file"]

    device_str = bm.get("device", "auto")
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) else "cpu")
    logger.info("Benchmark device: %s", device)

    models_to_run = args.models if args.models else [m for m in full_cfg["models"]]
    datasets_to_run = args.datasets if args.datasets else [
        d["name"] for d in full_cfg["datasets"]
    ]

    dataset_configs = {d["name"]: d for d in full_cfg["datasets"]}
    all_rows = []

    for ds_name in datasets_to_run:
        ds_cfg = dataset_configs[ds_name]
        fault_domain = ds_cfg["fault_domain"]
        n_channels = ds_cfg["n_channels"]

        logger.info("\n=== Dataset: %s ===", ds_name)
        X, y, class_names = build_dataset(
            fault_domain,
            bm["n_per_class"],
            bm["window_size"],
            bm["random_seed"],
        )
        logger.info("Samples: %d | Classes: %d | Channels: %d", len(y), len(class_names), n_channels)

        for model_name in models_to_run:
            logger.info("  Model: %s (%d runs)", model_name, bm["n_runs"])
            run_metrics = []

            for run in range(bm["n_runs"]):
                seed = bm["random_seed"] + run * 100
                torch.manual_seed(seed)
                np.random.seed(seed)

                try:
                    m = train_and_evaluate(
                        model_name, X, y,
                        cfg={
                            "epochs": bm["epochs"],
                            "batch_size": bm["batch_size"],
                            "lr": bm["lr"],
                            "weight_decay": bm["weight_decay"],
                            "num_workers": bm["num_workers"],
                        },
                        device=device,
                        random_seed=seed,
                    )
                    run_metrics.append(m)
                    logger.info("    Run %d/%d: acc=%.3f  f1=%.4f",
                                run + 1, bm["n_runs"], m["accuracy"], m["macro_f1"])
                except Exception as e:
                    logger.error("    Run %d failed: %s", run + 1, e)
                    continue

            if not run_metrics:
                continue

            # Aggregate across runs
            row = {
                "dataset": ds_name,
                "model": model_name,
                "n_runs": len(run_metrics),
                "n_params": run_metrics[0]["n_params"],
            }
            for metric in ["accuracy", "macro_f1", "weighted_f1", "cohen_kappa",
                           "total_train_time_s", "avg_epoch_time_s"]:
                values = [r[metric] for r in run_metrics]
                row[f"{metric}_mean"] = round(float(np.mean(values)), 4)
                row[f"{metric}_std"] = round(float(np.std(values)), 4)

            all_rows.append(row)
            logger.info("  → acc=%.3f±%.3f  f1=%.4f±%.4f",
                        row["accuracy_mean"], row["accuracy_std"],
                        row["macro_f1_mean"], row["macro_f1_std"])

    # Save CSV
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info("\nResults saved to: %s", csv_path)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Dataset':<22} {'Model':<18} {'Accuracy':>10} {'Macro F1':>10} {'Params':>12}")
    print("=" * 90)
    for row in all_rows:
        print(
            f"{row['dataset']:<22} {row['model']:<18} "
            f"{row['accuracy_mean']*100:>8.2f}%  "
            f"{row['macro_f1_mean']:>10.4f}  "
            f"{row['n_params']:>12,}"
        )
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all models on power electronics datasets")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke-test (5 epochs, 50 samples/class)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to run (default: all)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to run (default: all)")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
