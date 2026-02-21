#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PARQUET_PATH = Path("data/index/images.parquet")
SPLITS_PATH = Path("data/index/splits.parquet")
OUT_PATH = Path("data/index/images.parquet")
LABELS_PATH = Path("data/index/labels.json")


def main() -> None:
    if not PARQUET_PATH.exists():
        raise RuntimeError(f"Missing {PARQUET_PATH}. Run: python tools/dataset/build_index.py ...")
    if not SPLITS_PATH.exists():
        raise RuntimeError(f"Missing {SPLITS_PATH}. Run: python tools/dataset/make_splits.py")

    df = pd.read_parquet(PARQUET_PATH)
    splits = pd.read_parquet(SPLITS_PATH)

    if "line_idx" not in df.columns:
        raise RuntimeError("images.parquet missing line_idx")
    if "line_idx" not in splits.columns:
        raise RuntimeError("splits.parquet missing line_idx (regenerate splits with updated make_splits.py)")

    needed = {"line_idx", "split"}
    missing = needed - set(splits.columns)
    if missing:
        raise RuntimeError(f"splits.parquet missing columns: {sorted(missing)}")

    # Ensure line_idx is unique in splits (it should be 1:1 with parquet rows)
    before = len(splits)
    splits = splits.drop_duplicates(subset=["line_idx"], keep="first").reset_index(drop=True)
    dups = before - len(splits)
    if dups > 0:
        print(f"WARN: splits.parquet had {dups} duplicate line_idx rows. Kept first occurrence.")

    if "split" in df.columns:
        df = df.drop(columns=["split"])

    df = df.merge(splits[["line_idx", "split"]], on="line_idx", how="left", validate="one_to_one")

    if df["split"].isna().any():
        n = int(df["split"].isna().sum())
        bad = df.loc[df["split"].isna(), ["line_idx", "id"]].head(10).to_dict("records")
        raise RuntimeError(f"{n} rows missing split after merge. Example: {bad}")

    if "h3_id" not in df.columns:
        raise RuntimeError("images.parquet missing h3_id")

    labels = sorted(df["h3_id"].dropna().unique())
    label_to_idx = {h3_id: i for i, h3_id in enumerate(labels)}
    df["label_idx"] = df["h3_id"].map(label_to_idx).fillna(-1).astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(
        json.dumps(
            {
                "num_classes": len(labels),
                "labels": labels,
                "label_to_idx": label_to_idx,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {OUT_PATH}")
    print(f"Wrote: {LABELS_PATH}")
    print("\nSplit distribution:")
    print(df["split"].value_counts())
    print("\nNum classes:", len(labels))


if __name__ == "__main__":
    main()
