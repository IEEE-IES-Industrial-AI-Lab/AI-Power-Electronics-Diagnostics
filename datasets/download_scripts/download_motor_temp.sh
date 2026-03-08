#!/usr/bin/env bash
# Download the Kaggle Electric Motor Temperature dataset.
#
# Prerequisites:
#   pip install kaggle
#   Place kaggle.json at ~/.kaggle/kaggle.json with chmod 600

set -e

TARGET_DIR="$(git rev-parse --show-toplevel)/datasets/raw/motor_temp"
mkdir -p "$TARGET_DIR"

echo "Downloading Electric Motor Temperature dataset..."
kaggle datasets download -d wkirgsn/electric-motor-temperature \
    -p "$TARGET_DIR" --unzip

echo "Done. Files saved to: $TARGET_DIR"
ls "$TARGET_DIR"
