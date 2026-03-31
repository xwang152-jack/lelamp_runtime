#!/usr/bin/env bash
# Download edge AI model files for LeLamp vision features
# Usage: bash scripts/download_models.sh

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$MODELS_DIR"

BASE_URL="https://storage.googleapis.com/mediapipe-models"

declare -A MODELS=(
  ["blaze_face_full_range.tflite"]="face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
  ["hand_landmarker.task"]="hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
  ["gesture_recognizer.task"]="gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
  ["efficientdet_lite0.tflite"]="object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite"
)

for name in "${!MODELS[@]}"; do
  dest="$MODELS_DIR/$name"
  if [ -f "$dest" ]; then
    echo "✓ $name already exists, skipping"
    continue
  fi
  url="$BASE_URL/${MODELS[$name]}"
  echo "Downloading $name ..."
  curl -fSL -o "$dest" "$url"
  echo "✓ Downloaded $name ($(du -h "$dest" | cut -f1))"
done

echo ""
echo "All models ready in models/"
ls -lh "$MODELS_DIR"/*.tflite "$MODELS_DIR"/*.task 2>/dev/null
