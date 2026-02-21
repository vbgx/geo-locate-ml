#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h3
import pandas as pd

from src.labels import LabelSpace, build_label_space


PARQUET_PATH = Path("data/index/images.parquet")
SPLITS_PATH = Path("data/index/splits.parquet")
OUT_PATH = Path("data/index/images.parquet")
LABELS_PATH = Path("data/index/labels.json")
LABELS_FULL_PATH = Path("data/index/labels_full.json")
MERGE_MAP_PATH = Path("data/index/merge_map.parquet")


def _load_far_rate_map(path: str, labels_path: str) -> dict[str, float]:
    if not path:
        return {}
    pth = Path(path)
    if not pth.exists():
        print(f"[merge] far_rate file not found: {pth} (skipping)")
        return {}

    if pth.suffix.lower() == ".parquet":
        df = pd.read_parquet(pth)
    else:
        df = pd.read_csv(pth)

    if "far_rate" not in df.columns:
        print(f"[merge] far_rate column missing in {pth} (skipping)")
        return {}

    if "h3_id" in df.columns:
        return {str(h): float(fr) for h, fr in zip(df["h3_id"].tolist(), df["far_rate"].tolist())}

    if "true_idx" in df.columns:
        if not labels_path:
            print(f"[merge] far_rate file has true_idx but labels_path is empty (skipping)")
            return {}
        lp = Path(labels_path)
        if not lp.exists():
            print(f"[merge] labels_path not found: {lp} (skipping)")
            return {}
        labels = LabelSpace.from_json(lp.read_text(encoding="utf-8"))
        idx_to_h3 = labels.idx_to_h3
        out: dict[str, float] = {}
        for idx, fr in zip(df["true_idx"].tolist(), df["far_rate"].tolist()):
            try:
                h = idx_to_h3[int(idx)]
            except Exception:
                continue
            out[str(h)] = float(fr)
        return out

    print(f"[merge] far_rate file missing h3_id/true_idx columns: {pth} (skipping)")
    return {}


def _merge_toxic_cells(
    df: pd.DataFrame,
    *,
    min_cell_count: int,
    parent_res: int,
    far_rate_map: dict[str, float],
    far_rate_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    counts = df["h3_id"].value_counts()
    merge_set = set(counts[counts < int(min_cell_count)].index.tolist())

    if far_rate_map:
        for h, fr in far_rate_map.items():
            if float(fr) > float(far_rate_threshold):
                merge_set.add(str(h))

    mapping_rows = []
    for h3_id, cnt in counts.items():
        fr = float(far_rate_map.get(str(h3_id), float("nan"))) if far_rate_map else float("nan")
        reasons: list[str] = []
        if int(cnt) < int(min_cell_count):
            reasons.append(f"count<{int(min_cell_count)}")
        if far_rate_map and not math.isnan(fr) and fr > float(far_rate_threshold):
            reasons.append(f"far_rate>{float(far_rate_threshold)}")
        merged = str(h3_id) in merge_set
        if merged:
            parent = str(h3.cell_to_parent(str(h3_id), int(parent_res)))
        else:
            parent = str(h3_id)
        mapping_rows.append(
            {
                "h3_id_raw": str(h3_id),
                "h3_id_merged": parent,
                "cell_count": int(cnt),
                "far_rate": fr if far_rate_map else float("nan"),
                "merged": bool(merged),
                "reason": "|".join(reasons),
            }
        )

    if merge_set:
        df.loc[df["h3_id"].isin(merge_set), "h3_id"] = df.loc[df["h3_id"].isin(merge_set), "h3_id"].map(
            lambda h: str(h3.cell_to_parent(str(h), int(parent_res)))
        )

    mapping = pd.DataFrame(mapping_rows)
    return df, mapping


def _filter_min_samples_per_split(
    df: pd.DataFrame,
    *,
    min_samples: int,
    split_col: str = "split",
    label_col: str = "h3_id",
) -> tuple[pd.DataFrame, list[str]]:
    counts = df.groupby([label_col, split_col]).size().unstack(fill_value=0)
    keep = counts[(counts >= int(min_samples)).all(axis=1)].index.tolist()
    out = df[df[label_col].isin(keep)].copy()
    return out, keep


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge splits + build labels (with r6/r7 hierarchy).")
    ap.add_argument("--parquet", default=str(PARQUET_PATH))
    ap.add_argument("--splits", default=str(SPLITS_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--labels-out", default=str(LABELS_PATH))
    ap.add_argument("--labels-full-out", default=str(LABELS_FULL_PATH))
    ap.add_argument("--merge-map-out", default=str(MERGE_MAP_PATH))
    ap.add_argument("--min-cell-samples", type=int, default=30)
    ap.add_argument("--merge-parent-res", type=int, default=6)
    ap.add_argument("--merge-min-count", type=int, default=15)
    ap.add_argument("--merge-far-rate", type=float, default=0.8)
    ap.add_argument("--far-classes", default="", help="Optional classes_far.csv/parquet with far_rate")
    ap.add_argument("--far-labels", default="", help="labels.json for mapping true_idx->h3_id when needed")
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    splits_path = Path(args.splits)
    out_path = Path(args.out)
    labels_out = Path(args.labels_out)
    labels_full_out = Path(args.labels_full_out)
    merge_map_out = Path(args.merge_map_out)

    if not parquet_path.exists():
        raise RuntimeError(f"Missing {parquet_path}. Run: python tools/dataset/build_index.py ...")
    if not splits_path.exists():
        raise RuntimeError(f"Missing {splits_path}. Run: python tools/dataset/make_splits.py")

    df = pd.read_parquet(parquet_path)
    splits = pd.read_parquet(splits_path)

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

    # filter invalid rows
    if "is_valid" in df.columns:
        df = df[df["is_valid"].astype(bool)].copy()
    else:
        df = df[df["h3_id"].notna()].copy()

    far_rate_map = _load_far_rate_map(str(args.far_classes), str(args.far_labels))
    df, merge_map = _merge_toxic_cells(
        df,
        min_cell_count=int(args.merge_min_count),
        parent_res=int(args.merge_parent_res),
        far_rate_map=far_rate_map,
        far_rate_threshold=float(args.merge_far_rate),
    )

    # Hard filter after split
    df, _kept = _filter_min_samples_per_split(df, min_samples=int(args.min_cell_samples))

    # Build label space (includes r6 hierarchy)
    labels = build_label_space(df, parent_res=int(args.merge_parent_res))
    df["label_idx"] = df["h3_id"].map(labels.h3_to_idx).astype(int)
    df["label_r6_idx"] = df["label_idx"].map(labels.r7_to_r6).astype(int)
    df["h3_id_r6"] = df["label_r6_idx"].map(labels.idx_to_h3_r6)

    # Recompute cell_count on final labels
    cell_counts = df["h3_id"].value_counts()
    df["cell_count"] = df["h3_id"].map(cell_counts).fillna(0).astype(int)
    df["cell_kept"] = True
    df["is_kept"] = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    labels_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.write_text(
        json.dumps(
            {
                "num_classes": len(labels.h3_ids),
                "labels": labels.h3_ids,
                "label_to_idx": labels.h3_to_idx,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    labels_full_out.parent.mkdir(parents=True, exist_ok=True)
    labels_full_out.write_text(labels.to_json() + "\n", encoding="utf-8")

    merge_map_out.parent.mkdir(parents=True, exist_ok=True)
    merge_map.to_parquet(merge_map_out, index=False)

    print(f"Wrote: {out_path}")
    print(f"Wrote: {labels_out}")
    print(f"Wrote: {labels_full_out}")
    print(f"Wrote: {merge_map_out}")
    print("\nSplit distribution:")
    print(df["split"].value_counts())
    print("\nNum classes:", len(labels.h3_ids))


if __name__ == "__main__":
    main()
