"""
Evaluation script: compute classification metrics, confusion matrix,
per-class F1 scores, and produce a summary report.

Usage
-----
  python training/evaluate.py \\
      --checkpoint training/checkpoints/my_experiment/best.pt \\
      --model cnn_waveform \\
      --dataset synthetic \\
      --config training/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: "DataLoader",
    device: torch.device,
    class_names: list,
) -> dict:
    """Run inference and compute all metrics.

    Returns
    -------
    metrics : dict with keys:
        accuracy, macro_f1, weighted_f1, cohen_kappa,
        per_class_f1, confusion_matrix, classification_report
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.numpy().tolist())
            all_probs.extend(probs.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
        "per_class_f1": {class_names[i]: f for i, f in enumerate(per_class)},
        "confusion_matrix": cm,
        "classification_report": report,
        "n_samples": len(y_true),
    }


def print_results(metrics: dict) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy      : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1      : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1   : {metrics['weighted_f1']:.4f}")
    print(f"  Cohen Kappa   : {metrics['cohen_kappa']:.4f}")
    print(f"  Samples       : {metrics['n_samples']}")
    print("\nPer-class F1:")
    for cls_name, f1 in metrics["per_class_f1"].items():
        print(f"    {cls_name:<25s} {f1:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained fault detection model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="cnn_waveform",
                        choices=["cnn_waveform", "spectrogram_cnn", "transformer",
                                 "bilstm", "autoencoder"])
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "motor_temp"])
    parser.add_argument("--fault_domain", type=str, default="inverter",
                        choices=["inverter", "motor"])
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional path to save results as JSON")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from training.utils import get_device
    device = get_device(cfg["training"]["device"])
    logger.info("Device: %s", device)

    # Build model
    from models import MODEL_REGISTRY
    model_cfg = cfg["model"].get(args.model.replace("-", "_"), {})
    n_classes = cfg["model"]["n_classes"]
    n_channels = cfg["dataset"]["n_channels"]
    window_size = cfg["dataset"]["window_size"]

    model = MODEL_REGISTRY[args.model](
        n_channels=n_channels,
        n_classes=n_classes,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    logger.info("Loaded checkpoint: %s (epoch %d)", args.checkpoint, state.get("epoch", -1))

    # Build dataset
    if args.dataset == "synthetic":
        if args.fault_domain == "inverter":
            from datasets.synthetic import InverterFaultSimulator
            sim = InverterFaultSimulator()
            X, y = sim.generate_dataset(
                n_per_class=cfg["dataset"]["n_per_class"],
                window_size=window_size,
            )
        else:
            from datasets.synthetic import MotorDriveSimulator
            sim = MotorDriveSimulator()
            X, y = sim.generate_dataset(
                n_per_class=cfg["dataset"]["n_per_class"],
                window_size=window_size,
            )

        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
    else:
        logger.error("Only 'synthetic' supported in this evaluation script.")
        sys.exit(1)

    from torch.utils.data import DataLoader, TensorDataset
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test.astype(np.float32)),
            torch.from_numpy(y_test.astype(np.int64)),
        ),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    from datasets.synthetic.inverter_fault_sim import INVERTER_FAULT_LABELS
    class_names = list(INVERTER_FAULT_LABELS.keys())

    metrics = evaluate_model(model, test_loader, device, class_names)
    print_results(metrics)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {k: v for k, v in metrics.items() if k != "classification_report"}
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
