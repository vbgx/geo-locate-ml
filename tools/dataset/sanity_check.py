#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sanity check for dataset parquet(s).")
    ap.add_argument("--parquet", default="data/index/images.parquet")
    ap.add_argument("--labels", default="data/index/labels.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    path = Path(args.parquet)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run tools/dataset/build_index.py + merge_splits.py first.")

    df = pd.read_parquet(path)

    # Always-required columns (both full + kept)
    required_base = ["id", "lat", "lon", "path", "h3_id", "split", "line_idx"]
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in parquet: {missing}")

    # If this is the kept dataset, we expect label_idx
    is_kept = ("is_kept" in df.columns) or ("cell_count_split" in df.columns)
    has_label_idx = "label_idx" in df.columns

    # file existence
    missing_files = df[~df["path"].map(lambda s: Path(str(s)).exists())]
    if len(missing_files) > 0:
        print(f"WARN: {len(missing_files)} rows point to missing files.")

    # split distribution
    print("\nSplit distribution:")
    print(df["split"].value_counts())

    # classes
    print("\nCells/classes:")
    vc = df["h3_id"].value_counts()
    print(f"- num cells: {vc.shape[0]}")
    print(f"- samples min/median/max: {int(vc.min())}/{float(vc.median()):.1f}/{int(vc.max())}")

    # label_idx presence (only strict for kept)
    if is_kept and not has_label_idx:
        raise RuntimeError(
            f"{path} looks like a kept/training dataset but is missing 'label_idx'. "
            "Fix merge_splits to write label_idx into the kept parquet."
        )
    if not is_kept and not has_label_idx:
        print("\nNOTE: 'label_idx' missing (OK for full dataset). Sanity will not validate label mapping.")

    # leakage check (sequence_id should not be in multiple splits)
    if "sequence_id" in df.columns:
        s = df.dropna(subset=["sequence_id"])
        if len(s) > 0:
            leak = s.groupby("sequence_id")["split"].nunique().reset_index(name="n_splits")
            leaked = leak[leak["n_splits"] > 1]
            print("\nLeakage check (sequence_id across multiple splits):")
            if len(leaked) == 0:
                print("✅ OK: no sequence appears in multiple splits.")
            else:
                print(f"❌ FAIL: {len(leaked)} sequences appear in multiple splits.")
        else:
            print("\nLeakage check: sequence_id missing/empty -> cannot verify.")
    else:
        print("\nLeakage check: sequence_id column missing -> cannot verify.")

    print("\nSanity check done.")


if __name__ == "__main__":
    main()
