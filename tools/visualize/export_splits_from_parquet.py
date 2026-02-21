#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Export id->split (train/val/...) from dataset parquet.")
    ap.add_argument("--parquet", required=True, help="Dataset parquet path used for training.")
    ap.add_argument("--out", default="data/index/splits.jsonl", help="Output JSONL path.")
    ap.add_argument("--id-col", default="image_id", help="ID column name (try: image_id or id).")
    ap.add_argument("--split-col", default="split", help="Split column name (default: split).")
    args = ap.parse_args()

    pq = Path(args.parquet)
    if not pq.exists():
        raise SystemExit(f"Missing parquet: {pq}")

    df = pd.read_parquet(pq, columns=[args.id_col, args.split_col])

    if args.id_col not in df.columns:
        raise SystemExit(f"Missing id column '{args.id_col}' in parquet. Available: {list(df.columns)}")
    if args.split_col not in df.columns:
        raise SystemExit(f"Missing split column '{args.split_col}' in parquet. Available: {list(df.columns)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    counts = {}
    with out.open("w", encoding="utf-8") as f:
        for _id, sp in zip(df[args.id_col].astype(str), df[args.split_col].astype(str)):
            sp = sp.strip() or "unknown"
            f.write(json.dumps({"id": _id, "split": sp}, ensure_ascii=False) + "\n")
            n += 1
            counts[sp] = counts.get(sp, 0) + 1

    print(f"Wrote {n} rows -> {out}")
    print("Counts:", counts)


if __name__ == "__main__":
    main()
