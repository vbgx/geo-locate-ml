#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _print_head(title: str, df: pd.DataFrame, n: int = 20) -> None:
    print("\n" + title)
    print("-" * len(title))
    if len(df) == 0:
        print("(none)")
        return
    with pd.option_context("display.max_rows", n, "display.max_columns", 200, "display.width", 140):
        print(df.head(n).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Check duplicates and split leakage in data/index/images.parquet")
    ap.add_argument("--parquet", default="data/index/images.parquet", help="Path to images.parquet")
    ap.add_argument("--show", type=int, default=30, help="How many rows to show for each report section")
    args = ap.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_parquet(path)

    required = ["id", "split", "label_idx", "h3_id", "path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in parquet: {missing}")

    # Normalize types
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["split"] = df["split"].astype(str)

    print(f"Loaded: {path}")
    print(f"Rows: {len(df)}")
    print("Split distribution:")
    print(df["split"].value_counts().to_string())

    # 1) Duplicate ids (same image id appears multiple times)
    dup_ids = df.groupby("id").size().reset_index(name="n").sort_values("n", ascending=False)
    dup_ids = dup_ids[dup_ids["n"] > 1]

    _print_head("Duplicate image ids (id appears >1 times)", dup_ids, n=int(args.show))

    if len(dup_ids) > 0:
        top_dup = dup_ids.head(min(10, len(dup_ids)))["id"].tolist()
        sub = df[df["id"].isin(top_dup)].sort_values(["id", "split", "label_idx"])
        cols = [c for c in ["id", "split", "label_idx", "h3_id", "sequence_id", "city", "captured_at", "path"] if c in sub.columns]
        _print_head("Examples for top duplicate ids", sub[cols], n=int(args.show))

    # 2) Same id across multiple splits (leakage)
    splits_per_id = df.groupby("id")["split"].nunique().reset_index(name="split_n")
    leak_ids = splits_per_id[splits_per_id["split_n"] > 1].sort_values("split_n", ascending=False)
    _print_head("IDs present in multiple splits (leakage)", leak_ids, n=int(args.show))

    if len(leak_ids) > 0:
        leak_top = leak_ids.head(min(20, len(leak_ids)))["id"].tolist()
        sub = df[df["id"].isin(leak_top)].sort_values(["id", "split"])
        cols = [c for c in ["id", "split", "label_idx", "h3_id", "sequence_id", "path"] if c in sub.columns]
        _print_head("Leakage examples (same id across splits)", sub[cols], n=int(args.show))

    # 3) Sequence leakage (sequence_id in multiple splits)
    if "sequence_id" in df.columns:
        s = df.dropna(subset=["sequence_id"]).copy()
        s["sequence_id"] = s["sequence_id"].astype(str)

        if len(s) == 0:
            print("\nSequence leakage: sequence_id column exists but no non-null values.")
        else:
            seq_splits = s.groupby("sequence_id")["split"].nunique().reset_index(name="split_n")
            seq_leaks = seq_splits[seq_splits["split_n"] > 1].sort_values(["split_n"], ascending=False)
            _print_head("Sequences present in multiple splits (sequence leakage)", seq_leaks, n=int(args.show))

            if len(seq_leaks) > 0:
                leak_seq_top = seq_leaks.head(min(20, len(seq_leaks)))["sequence_id"].tolist()
                sub = s[s["sequence_id"].isin(leak_seq_top)].sort_values(["sequence_id", "split"])
                cols = [c for c in ["sequence_id", "id", "split", "label_idx", "h3_id", "path"] if c in sub.columns]
                _print_head("Sequence leakage examples", sub[cols], n=int(args.show))
    else:
        print("\nSequence leakage: sequence_id column missing -> cannot check.")

    # 4) Same file path mapped to multiple ids (inconsistent index)
    # (can happen if path column points to processed dir in multiple ways)
    if "path" in df.columns:
        p = df.copy()
        p["path"] = p["path"].astype(str)
        ids_per_path = p.groupby("path")["id"].nunique().reset_index(name="id_n")
        bad_paths = ids_per_path[ids_per_path["id_n"] > 1].sort_values("id_n", ascending=False)
        _print_head("Paths pointing to multiple ids (index inconsistency)", bad_paths, n=int(args.show))

        if len(bad_paths) > 0:
            top_paths = bad_paths.head(min(20, len(bad_paths)))["path"].tolist()
            sub = p[p["path"].isin(top_paths)].sort_values(["path", "id", "split"])
            cols = [c for c in ["path", "id", "split", "label_idx", "h3_id"] if c in sub.columns]
            _print_head("Path inconsistency examples", sub[cols], n=int(args.show))

    # Summary exit hints
    print("\nSummary:")
    print(f"- duplicate ids: {len(dup_ids)}")
    print(f"- ids across multiple splits: {len(leak_ids)}")
    if "sequence_id" in df.columns:
        s = df.dropna(subset=["sequence_id"])
        if len(s) > 0:
            seq_splits = s.groupby("sequence_id")["split"].nunique()
            seq_leaks_n = int((seq_splits > 1).sum())
            print(f"- sequence leakage count: {seq_leaks_n}")
        else:
            print("- sequence leakage count: (n/a, no sequence_id values)")
    else:
        print("- sequence leakage count: (n/a, missing column)")

    print("\nDone.")


if __name__ == "__main__":
    main()
