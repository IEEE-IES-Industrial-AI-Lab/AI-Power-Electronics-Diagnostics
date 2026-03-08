"""
One-click dataset setup script.

Usage
-----
  # Download all datasets
  python datasets/download_scripts/setup_datasets.py

  # Download a specific dataset
  python datasets/download_scripts/setup_datasets.py --dataset motor_temp

  # List available datasets
  python datasets/download_scripts/setup_datasets.py --list

Supported datasets
------------------
  motor_temp  : Kaggle Electric Motor Temperature (requires Kaggle API key)

Kaggle setup:
  1. Go to https://www.kaggle.com/settings/account → "Create new token"
  2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
  3. chmod 600 ~/.kaggle/kaggle.json
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "motor_temp": {
        "description": "Kaggle Electric Motor Temperature (PMSM test bench)",
        "kaggle_dataset": "wkirgsn/electric-motor-temperature",
        "target_dir": REPO_ROOT / "datasets" / "raw" / "motor_temp",
        "method": "kaggle",
    },
}


def list_datasets() -> None:
    print("\nAvailable datasets:\n")
    for name, meta in DATASETS.items():
        print(f"  {name:20s} — {meta['description']}")
    print()


def download_kaggle(dataset_id: str, target_dir: Path) -> None:
    """Download and unzip a Kaggle dataset."""
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle  # noqa: F401
    except ImportError:
        logger.error(
            "kaggle package not found. Install it with:  pip install kaggle"
        )
        sys.exit(1)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.error(
            "Kaggle API credentials not found at %s.\n"
            "  1. Go to https://www.kaggle.com/settings → 'Create new token'\n"
            "  2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json\n"
            "  3. Run: chmod 600 ~/.kaggle/kaggle.json",
            kaggle_json,
        )
        sys.exit(1)

    logger.info("Downloading '%s' → %s", dataset_id, target_dir)
    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", dataset_id,
        "-p", str(target_dir),
        "--unzip",
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("Kaggle download failed (exit code %d).", result.returncode)
        sys.exit(result.returncode)

    logger.info("Download complete: %s", target_dir)


def setup_dataset(name: str) -> None:
    if name not in DATASETS:
        logger.error("Unknown dataset '%s'. Use --list to see available datasets.", name)
        sys.exit(1)

    meta = DATASETS[name]
    method = meta["method"]

    if method == "kaggle":
        download_kaggle(meta["kaggle_dataset"], meta["target_dir"])
    else:
        logger.error("Unknown download method: %s", method)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and set up datasets for AI-Power-Electronics-Diagnostics"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to download (default: all)",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    targets = [args.dataset] if args.dataset else list(DATASETS)
    for name in targets:
        logger.info("Setting up dataset: %s", name)
        setup_dataset(name)

    logger.info("All requested datasets are ready.")


if __name__ == "__main__":
    main()
