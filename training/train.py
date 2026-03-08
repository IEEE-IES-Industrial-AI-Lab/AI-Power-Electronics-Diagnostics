"""
Training CLI for AI-Power-Electronics-Diagnostics models.

Usage
-----
  # Train 1D CNN on synthetic inverter data
  python training/train.py --model cnn_waveform --dataset synthetic

  # Train Transformer on motor drive data
  python training/train.py \\
      --model transformer \\
      --dataset synthetic \\
      --fault_domain motor \\
      --epochs 100 \\
      --lr 5e-4

  # Train autoencoder (unsupervised)
  python training/train.py --model autoencoder --dataset synthetic

  # Full help
  python training/train.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(cfg: dict, fault_domain: str, model_name: str):
    """Generate or load the dataset and return train/val/test splits."""
    dataset_cfg = cfg["dataset"]
    window_size = dataset_cfg["window_size"]
    n_per_class = dataset_cfg["n_per_class"]

    if dataset_cfg["source"] == "synthetic":
        if fault_domain == "inverter":
            from datasets.synthetic import InverterFaultSimulator
            sim = InverterFaultSimulator()
            X, y = sim.generate_dataset(n_per_class=n_per_class, window_size=window_size)
            class_names = list(sim.fault_labels().keys())
        else:
            from datasets.synthetic import MotorDriveSimulator
            sim = MotorDriveSimulator()
            X, y = sim.generate_dataset(n_per_class=n_per_class, window_size=window_size)
            class_names = list(sim.fault_labels().keys())

    elif dataset_cfg["source"] == "motor_temp":
        from datasets.loaders import MotorTemperatureLoader
        loader = MotorTemperatureLoader(
            data_dir=REPO_ROOT / "datasets" / "raw" / "motor_temp",
            window_size=window_size,
        )
        X, y = loader.load()
        class_names = loader.class_names
    else:
        raise ValueError(f"Unknown dataset source: {dataset_cfg['source']}")

    # Spectrogram mode: convert raw to spectrograms
    if model_name == "spectrogram_cnn":
        from signal_processing import STFTSpectrogram
        stft_cfg = cfg["model"].get("spectrogram_cnn", {})
        size = tuple(stft_cfg.get("spectrogram_size", [128, 128]))
        stft = STFTSpectrogram(
            f_sample=50_000.0,
            output_size=size,
            log_scale=stft_cfg.get("log_scale", True),
        )
        logger.info("Converting %d signals to STFT spectrograms %s ...", len(X), size)
        X = stft.compute_batch(X)  # (N, C, H, W)

    from datasets.loaders.base_loader import BaseDatasetLoader
    split = BaseDatasetLoader.train_val_test_split(
        None, X, y,
        train_frac=dataset_cfg["train_frac"],
        val_frac=dataset_cfg["val_frac"],
    )
    split.class_names = class_names
    return split


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, model_name: str, n_classes: int, n_channels: int, window_size: int):
    from models import MODEL_REGISTRY
    model_cfg = cfg["model"]
    dropout = model_cfg.get("dropout", 0.3)

    kwargs = dict(n_channels=n_channels, dropout=dropout)

    if model_name != "autoencoder":
        kwargs["n_classes"] = n_classes

    if model_name == "cnn_waveform":
        sub = model_cfg.get("cnn_waveform", {})
        kwargs.update(dict(
            window_size=window_size,
            base_filters=sub.get("base_filters", 64),
            kernel_size=sub.get("kernel_size", 7),
        ))
    elif model_name == "spectrogram_cnn":
        pass  # uses default
    elif model_name == "transformer":
        sub = model_cfg.get("transformer", {})
        kwargs.update(dict(
            window_size=window_size,
            patch_size=sub.get("patch_size", 64),
            d_model=sub.get("d_model", 128),
            n_heads=sub.get("n_heads", 8),
            n_layers=sub.get("n_layers", 4),
            ffn_dim=sub.get("ffn_dim", 512),
        ))
    elif model_name == "bilstm":
        sub = model_cfg.get("bilstm", {})
        kwargs.update(dict(
            hidden_size=sub.get("hidden_size", 256),
            n_layers=sub.get("n_layers", 2),
        ))
    elif model_name == "autoencoder":
        sub = model_cfg.get("autoencoder", {})
        kwargs.update(dict(
            window_size=window_size,
            latent_channels=sub.get("latent_channels", 32),
            base_filters=sub.get("base_filters", 64),
        ))

    return MODEL_REGISTRY[model_name](**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Training loops
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: dict,
    device: torch.device,
    experiment_name: str,
) -> None:
    """Standard supervised classification training loop."""
    from training.utils import (
        AverageMeter, build_optimizer, build_scheduler,
        CheckpointManager, EarlyStopping,
    )

    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    optimizer = build_optimizer(
        model,
        train_cfg["optimizer"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer,
        train_cfg["scheduler"],
        cfg.get(train_cfg["scheduler"], {}),
        epochs=train_cfg["epochs"],
    )
    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(
        patience=train_cfg["early_stopping"]["patience"],
        monitor="min",
    )
    ckpt_mgr = CheckpointManager(
        out_cfg["checkpoint_dir"],
        experiment_name,
        monitor="val_loss",
    )

    # TensorBoard
    writer = None
    if out_cfg.get("tensorboard"):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(Path(out_cfg["log_dir"]) / experiment_name))

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        loss_meter = AverageMeter("train_loss")
        correct = total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_meter.update(loss.item(), X_batch.size(0))
            preds = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_acc = correct / total

        # Validation
        val_loss, val_acc = _validate(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        metrics = {
            "train_loss": loss_meter.avg,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d  loss=%.4f  acc=%.3f  val_loss=%.4f  val_acc=%.3f  lr=%.2e",
                epoch, train_cfg["epochs"],
                loss_meter.avg, train_acc, val_loss, val_acc,
                optimizer.param_groups[0]["lr"],
            )

        if writer:
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)

        ckpt_mgr.save(epoch, model, optimizer, metrics,
                      is_last=(epoch == train_cfg["epochs"]))

        if train_cfg["early_stopping"]["enabled"] and early_stop.step(val_loss):
            logger.info("Early stopping at epoch %d.", epoch)
            break

    if writer:
        writer.close()

    logger.info("Training complete. Best val_loss=%.4f", early_stop.best_value)


def train_autoencoder(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: dict,
    device: torch.device,
    experiment_name: str,
) -> None:
    """Unsupervised autoencoder training on healthy signals only."""
    from training.utils import (
        AverageMeter, build_optimizer, build_scheduler,
        CheckpointManager, EarlyStopping,
    )

    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    optimizer = build_optimizer(
        model, train_cfg["optimizer"],
        lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, train_cfg["scheduler"],
                                 cfg, epochs=train_cfg["epochs"])
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=train_cfg["early_stopping"]["patience"])
    ckpt_mgr = CheckpointManager(out_cfg["checkpoint_dir"], experiment_name)

    logger.info("Autoencoder training — filtering to healthy (label=0) samples only.")

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        loss_meter = AverageMeter("train_loss")

        for X_batch, y_batch in train_loader:
            # Only train on healthy samples
            healthy_mask = y_batch == 0
            if healthy_mask.sum() == 0:
                continue
            X_healthy = X_batch[healthy_mask].to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_healthy), X_healthy)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), X_healthy.size(0))

        # Validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                healthy_mask = y_batch == 0
                if healthy_mask.sum() == 0:
                    continue
                X_h = X_batch[healthy_mask].to(device)
                val_losses.append(criterion(model(X_h), X_h).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            logger.info("Epoch %3d  train_loss=%.6f  val_loss=%.6f",
                        epoch, loss_meter.avg, val_loss)

        metrics = {"val_loss": val_loss, "train_loss": loss_meter.avg}
        ckpt_mgr.save(epoch, model, optimizer, metrics,
                      is_last=(epoch == train_cfg["epochs"]))

        if train_cfg["early_stopping"]["enabled"] and early_stop.step(val_loss):
            logger.info("Early stopping at epoch %d.", epoch)
            break

    logger.info("Autoencoder training complete.")


def _validate(model, loader, criterion, device):
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            losses.append(criterion(logits, y).item())
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)
    return float(np.mean(losses)), correct / (total + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train power electronics fault detection model")
    parser.add_argument("--model", type=str, default="cnn_waveform",
                        choices=["cnn_waveform", "spectrogram_cnn", "transformer",
                                 "bilstm", "autoencoder"])
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "motor_temp"])
    parser.add_argument("--fault_domain", type=str, default="inverter",
                        choices=["inverter", "motor"])
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.epochs: cfg["training"]["epochs"] = args.epochs
    if args.lr: cfg["training"]["lr"] = args.lr
    if args.batch_size: cfg["training"]["batch_size"] = args.batch_size
    if args.device: cfg["training"]["device"] = args.device
    cfg["dataset"]["source"] = args.dataset

    from training.utils import get_device, generate_experiment_name, make_dataloaders
    device = get_device(cfg["training"]["device"])
    logger.info("Device: %s", device)

    logger.info("Building dataset (%s / %s)...", args.dataset, args.fault_domain)
    split = build_dataset(cfg, args.fault_domain, args.model)
    n_classes = len(split.class_names)
    n_channels = split.X_train.shape[1]
    window_size = split.X_train.shape[-1]
    logger.info("Classes: %s | Train: %d | Val: %d | Test: %d",
                split.class_names, len(split.y_train), len(split.y_val), len(split.y_test))

    cfg["model"]["n_classes"] = n_classes

    logger.info("Building model: %s", args.model)
    model = build_model(cfg, args.model, n_classes, n_channels, window_size).to(device)
    logger.info("Parameters: %s", f"{model.count_parameters():,}")

    train_loader, val_loader, test_loader = make_dataloaders(
        split.X_train, split.y_train,
        split.X_val, split.y_val,
        split.X_test, split.y_test,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        class_weights=cfg["training"]["class_weights"],
    )

    exp_name = generate_experiment_name(args.model, args.dataset)
    if not cfg["output"].get("experiment_name"):
        cfg["output"]["experiment_name"] = exp_name

    logger.info("Experiment: %s", exp_name)

    if args.model == "autoencoder":
        train_autoencoder(model, train_loader, val_loader, cfg, device, exp_name)
    else:
        train_classifier(model, train_loader, val_loader, cfg, device, exp_name)

    logger.info("Done. Run evaluation with:")
    logger.info(
        "  python training/evaluate.py --checkpoint training/checkpoints/%s/best.pt "
        "--model %s --dataset %s",
        exp_name, args.model, args.dataset,
    )


if __name__ == "__main__":
    main()
