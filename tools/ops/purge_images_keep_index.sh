#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="data/raw"

echo "Cleaning raw images in ${RAW_DIR}..."

if [ ! -d "$RAW_DIR" ]; then
  echo "Directory $RAW_DIR does not exist. Nothing to clean."
  exit 0
fi

# Delete image files only
find "$RAW_DIR" -type f \( \
  -iname "*.jpg" -o \
  -iname "*.jpeg" -o \
  -iname "*.png" -o \
  -iname "*.webp" -o \
  -iname "*.tif" -o \
  -iname "*.tiff" \
\) -print -delete

echo "Raw images deleted. Indexes preserved."