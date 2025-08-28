#!/usr/bin/env bash
set -euo pipefail

# Download a YOLOv11 weights file for logo detection. Replace URL with your custom model if needed.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

MODEL_URL=${MODEL_URL:-"https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"}
MODEL_DST=${MODEL_DST:-"$ROOT_DIR/yolo11-logo.pt"}

mkdir -p "$(dirname "$MODEL_DST")"
echo "Downloading YOLOv11 weights to $MODEL_DST"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 -o "$MODEL_DST" "$MODEL_URL"
else
  wget -O "$MODEL_DST" "$MODEL_URL"
fi
echo "Done."


