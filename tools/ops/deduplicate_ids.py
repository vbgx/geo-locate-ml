#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Remove duplicate image ids (keep first occurrence).")
    ap.add_argument("--parquet", default="data/index/images.parquet")
    ap.add_argument("--dry-run", action="store_true", help="Only report duplicates, do not modify file.")
    args = ap.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_parquet(path)

    if "id" not in df.columns:
        raise RuntimeError("Column 'id' not found in parquet.")

    df["id"] = df["id"].astype(str)

    total_rows = len(df)

    # mark duplicates (keep first)
    duplicated_mask = df.duplicated(subset=["id"], keep="first")
    n_dups = int(duplicated_mask.sum())

    print(f"Total rows: {total_rows}")
    print(f"Duplicate rows to remove: {n_dups}")

    if n_dups == 0:
        print("Nothing to do.")
        return

    if args.dry_run:
        print("Dry run enabled. No file modified.")
        return

    # backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(f".backup_{timestamp}.parquet")
    df.to_parquet(backup_path, index=False)
    print(f"Backup saved to: {backup_path}")

    # remove duplicates
    df_clean = df[~duplicated_mask].copy()
    df_clean.to_parquet(path, index=False)

    print(f"Cleaned file written to: {path}")
    print(f"New row count: {len(df_clean)}")
    print(f"Removed rows: {total_rows - len(df_clean)}")


if __name__ == "__main__":
    main()
