#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
OUTPUT_FILE="$ROOT_DIR/scripts_bundle.txt"

echo "Generating bundle at: $OUTPUT_FILE"
echo "Repo root: $ROOT_DIR"

# Clean previous bundle
rm -f "$OUTPUT_FILE"

# Header
{
  echo "========================================"
  echo "SCRIPTS BUNDLE"
  echo "Generated: $(date)"
  echo "Root: $ROOT_DIR"
  echo "========================================"
  echo ""
} >> "$OUTPUT_FILE"

# Find all .py and .sh excluding heavy/irrelevant dirs
find . \
  -type f \
  \( -name "*.py" -o -name "*.sh" \) \
  ! -path "./.venv/*" \
  ! -path "./data/*" \
  ! -path "./runs/*" \
  ! -path "./exports/*" \
  ! -path "./models/*" \
  ! -path "./__pycache__/*" \
  ! -path "./.git/*" \
  | sort \
  | while read -r file; do

    echo "========================================" >> "$OUTPUT_FILE"
    echo "FILE: $file" >> "$OUTPUT_FILE"
    echo "========================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    cat "$file" >> "$OUTPUT_FILE"

    echo "" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

done

echo "Bundle created: $OUTPUT_FILE"