#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge splits + build labels (kept-only safe).")
    ap.add_argument("--parquet", required=True, help="Input images.parquet (from build_index).")
    ap.add_argument("--splits", required=True, help="Input splits.parquet (from make_splits).")
    ap.add_argument("--out", required=True, help="Output full dataset parquet (debug/inspection).")
    ap.add_argument("--out-kept", required=True, help="Output kept-only dataset parquet (training).")
    ap.add_argument("--labels-out", required=True, help="Output labels.json (built from kept-only).")
    ap.add_argument("--min-cell-samples", type=int, default=30, help="Min samples per (split, h3_id) to keep.")
    ap.add_argument("--drop-missing-files", action="store_true", help="Drop rows whose path does not exist.")
    return ap.parse_args()


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")


def main() -> None:
    args = parse_args()

    parquet_path = Path(args.parquet)
    splits_path = Path(args.splits)
    out_path = Path(args.out)
    out_kept_path = Path(args.out_kept)
    labels_path = Path(args.labels_out)

    if not parquet_path.exists():
        raise RuntimeError(f"Missing {parquet_path}")
    if not splits_path.exists():
        raise RuntimeError(f"Missing {splits_path}")

    df = pd.read_parquet(parquet_path)
    splits = pd.read_parquet(splits_path)

    _require_cols(df, ["line_idx", "id", "lat", "lon", "path", "h3_id"], "images parquet")
    _require_cols(splits, ["line_idx", "split"], "splits parquet")

    # Ensure 1:1 mapping line_idx -> split
    before = len(splits)
    splits = splits.drop_duplicates(subset=["line_idx"], keep="first").reset_index(drop=True)
    dropped = before - len(splits)
    if dropped > 0:
        print(f"WARN: splits.parquet had {dropped} duplicate line_idx rows. Kept first occurrence.")

    # Merge split column
    if "split" in df.columns:
        df = df.drop(columns=["split"])

    df = df.merge(
        splits[["line_idx", "split"]],
        on="line_idx",
        how="left",
        validate="one_to_one",
    )

    if df["split"].isna().any():
        n = int(df["split"].isna().sum())
        ex = df.loc[df["split"].isna(), ["line_idx", "id"]].head(10).to_dict("records")
        raise RuntimeError(f"{n} rows missing split after merge. Examples: {ex}")

    # -------------------------------------
    # FULL DATASET WRITE (debug/inspection)
    # -------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote full dataset: {out_path}")

    # -------------------------------------
    # KEPT-ONLY FILTER (per split per cell)
    # -------------------------------------
    counts = (
        df.groupby(["split", "h3_id"])
        .size()
        .reset_index(name="cell_count_split")
    )

    df2 = df.merge(counts, on=["split", "h3_id"], how="left")
    df2["cell_count_split"] = df2["cell_count_split"].fillna(0).astype(int)
    df2["is_kept"] = df2["cell_count_split"] >= int(args.min_cell_samples)

    df_kept = df2[df2["is_kept"]].copy()

    # Optional: drop missing files (on kept only)
    if args.drop_missing_files:
        missing_mask = ~df_kept["path"].map(lambda s: Path(str(s)).exists())
        if missing_mask.any():
            print(f"Dropping {int(missing_mask.sum())} kept rows with missing files")
            df_kept = df_kept.loc[~missing_mask].copy()

    if df_kept.empty:
        raise RuntimeError(
            "Kept-only dataset is empty. "
            "Lower --min-cell-samples or download more data."
        )

    # -------------------------------------
    # BUILD LABELS FROM KEPT ONLY
    # -------------------------------------
    labels = sorted(df_kept["h3_id"].dropna().astype(str).unique().tolist())
    if len(labels) <= 1:
        raise RuntimeError(
            f"Num kept classes looks wrong: {len(labels)}. "
            "Check your h3_id generation and min-cell-samples per split."
        )

    label_to_idx = {h3_id: i for i, h3_id in enumerate(labels)}
    df_kept["label_idx"] = df_kept["h3_id"].astype(str).map(label_to_idx).astype(int)

    # -------------------------------------
    # WRITE KEPT DATASET
    # -------------------------------------
    out_kept_path.parent.mkdir(parents=True, exist_ok=True)
    df_kept.to_parquet(out_kept_path, index=False)
    print(f"Wrote kept-only dataset: {out_kept_path}")

    # -------------------------------------
    # WRITE LABELS
    # -------------------------------------
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(
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
    print(f"Wrote labels: {labels_path}")

    # -------------------------------------
    # SANITY PRINT
    # -------------------------------------
    print("\nFull split distribution:")
    print(df["split"].value_counts())

    print("\nKept split distribution:")
    print(df_kept["split"].value_counts())

    print("\nNum kept classes:", len(labels))
    print("Samples per cell (kept):")
    print(df_kept.groupby("h3_id").size().describe())


if __name__ == "__main__":
    main()
