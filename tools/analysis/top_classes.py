#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

def pick_val_topk() -> Path:
    cands = list(Path("runs").glob("**/val_topk.parquet"))
    if not cands:
        raise FileNotFoundError("No runs/**/val_topk.parquet found.")
    latest = Path("runs/latest/val_topk.parquet")
    if latest.exists():
        return latest
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

VAL_TOPK = pick_val_topk()
print("Using:", VAL_TOPK)

df = pd.read_parquet(VAL_TOPK)

print("\nTop predicted classes (val):")
print(df["pred_idx"].value_counts().head(15))

print("\nTop true classes (val):")
print(df["true_idx"].value_counts().head(15))

conf = df[df["true_idx"] != df["pred_idx"]]
print("\nConfusion rate:", len(conf), "/", len(df), f"= {len(conf)/max(1,len(df)):.3f}")

pairs = conf.groupby(["true_idx", "pred_idx"]).size().sort_values(ascending=False)
print("\nMost frequent confusion pairs:")
print(pairs.head(20))
