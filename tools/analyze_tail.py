#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _safe_percentile(a: np.ndarray, p: float) -> float:
    if a.size == 0:
        return 0.0
    return float(np.percentile(a, p))


def _far_rate(a: np.ndarray, thr: float) -> float:
    if a.size == 0:
        return 0.0
    return float((a > float(thr)).mean())


def _summarize_dist(dist: np.ndarray) -> dict:
    dist = np.asarray(dist, dtype=np.float64)
    if dist.size == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "far200": 0.0,
            "far500": 0.0,
            "far1000": 0.0,
        }
    return {
        "n": int(dist.size),
        "mean": float(dist.mean()),
        "median": float(np.median(dist)),
        "p90": _safe_percentile(dist, 90),
        "p95": _safe_percentile(dist, 95),
        "far200": _far_rate(dist, 200.0),
        "far500": _far_rate(dist, 500.0),
        "far1000": _far_rate(dist, 1000.0),
    }


def _load_val_errors(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "val_errors.parquet"
    if not p.exists():
        raise SystemExit(f"Missing {p}. Run training with updated train.py to generate it.")
    df = pd.read_parquet(p)
    required = {"image_id", "true_idx", "pred_idx", "dist_km"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise SystemExit(f"val_errors.parquet missing columns: {missing}")
    return df


def _maybe_join_index(df: pd.DataFrame, index_parquet: Path) -> pd.DataFrame:
    if not index_parquet.exists():
        return df

    idx = pd.read_parquet(index_parquet)

    if "id" not in idx.columns:
        return df

    # 🔥 CRITICAL FIX: enforce string type on both sides
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)

    idx = idx.copy()
    idx["id"] = idx["id"].astype(str)

    keep_cols = [
        c for c in [
            "id",
            "split",
            "h3_id",
            "sequence_id",
            "city",
            "captured_at",
            "path",
            "lat",
            "lon",
        ] if c in idx.columns
    ]

    idx = idx[keep_cols]

    out = df.merge(idx, left_on="image_id", right_on="id", how="left")
    return out


def _top_classes_far(df: pd.DataFrame, far_thr: float, topn: int) -> pd.DataFrame:
    # by true class
    g = df.groupby("true_idx").agg(
        n=("dist_km", "size"),
        mean=("dist_km", "mean"),
        median=("dist_km", "median"),
        p90=("dist_km", lambda s: np.percentile(np.asarray(s), 90) if len(s) else 0.0),
        far_rate=("dist_km", lambda s: float((np.asarray(s) > far_thr).mean()) if len(s) else 0.0),
        far_n=("dist_km", lambda s: int((np.asarray(s) > far_thr).sum()) if len(s) else 0),
    )
    g = g.reset_index().sort_values(["far_rate", "far_n", "p90"], ascending=False)
    return g.head(topn)


def _top_confusion_pairs(df: pd.DataFrame, topn: int) -> pd.DataFrame:
    # pairs true->pred with biggest mean distance and enough support
    gp = df.groupby(["true_idx", "pred_idx"]).agg(
        n=("dist_km", "size"),
        mean_km=("dist_km", "mean"),
        median_km=("dist_km", "median"),
        p90_km=("dist_km", lambda s: np.percentile(np.asarray(s), 90) if len(s) else 0.0),
        max_km=("dist_km", "max"),
    ).reset_index()

    # avoid tiny-count noise
    gp = gp[gp["n"] >= 5].copy()
    gp = gp.sort_values(["mean_km", "max_km", "n"], ascending=False)
    return gp.head(topn)


def _top_images(df: pd.DataFrame, topn: int) -> pd.DataFrame:
    return df.sort_values("dist_km", ascending=False).head(topn).copy()


def _write_text_report(out_path: Path, title: str, blocks: list[Tuple[str, dict]]) -> None:
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    for name, stats in blocks:
        lines.append(f"{name}")
        lines.append("-" * len(name))
        for k, v in stats.items():
            lines.append(f"{k}: {v}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze tail geo errors to reduce P90 (uses val_errors.parquet).")
    ap.add_argument("--run-dir", required=True, help="Run directory containing val_errors.parquet")
    ap.add_argument("--index-parquet", default="data/index/images.parquet", help="Optional: join metadata from index parquet")
    ap.add_argument("--out", default=None, help="Output directory (default: <run-dir>/tail_analysis)")
    ap.add_argument("--far-thr", type=float, default=500.0, help="Far threshold in km (default 500)")
    ap.add_argument("--top-classes", type=int, default=50, help="How many classes to list (default 50)")
    ap.add_argument("--top-pairs", type=int, default=50, help="How many confusion pairs to list (default 50)")
    ap.add_argument("--top-images", type=int, default=200, help="How many worst images to list (default 200)")
    ap.add_argument("--top-pct", type=float, default=1.0, help="Also export top X%% rows (default 1%%)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else (run_dir / "tail_analysis")
    out.mkdir(parents=True, exist_ok=True)

    df = _load_val_errors(run_dir)
    df = _maybe_join_index(df, Path(args.index_parquet))

    dist = df["dist_km"].to_numpy(dtype=np.float64)
    overall = _summarize_dist(dist)

    # Top pct
    n = len(df)
    cut = max(1, int(n * (float(args.top_pct) / 100.0)))
    df_top_pct = df.sort_values("dist_km", ascending=False).head(cut).copy()

    # Top classes and pairs
    top_classes = _top_classes_far(df, far_thr=float(args.far_thr), topn=int(args.top_classes))
    top_pairs = _top_confusion_pairs(df, topn=int(args.top_pairs))
    top_images = _top_images(df, topn=int(args.top_images))

    # Optional per-split summary if split exists
    blocks = [("overall", overall)]
    if "split" in df.columns:
        for sp, sdf in df.groupby("split"):
            blocks.append((f"split={sp}", _summarize_dist(sdf["dist_km"].to_numpy(dtype=np.float64))))

    # Save outputs
    (out / "overall_stats.txt").write_text(
        "\n".join([f"{k}: {v}" for k, v in overall.items()]) + "\n",
        encoding="utf-8",
    )
    _write_text_report(out / "summary.txt", "Tail Analysis Summary", blocks)

    top_classes.to_csv(out / "classes_far.csv", index=False)
    top_pairs.to_csv(out / "confusion_pairs_far.csv", index=False)
    top_images.to_csv(out / "top_images.csv", index=False)
    df_top_pct.to_csv(out / "top_pct.csv", index=False)

    # also write parquet versions for fast reuse
    try:
        top_classes.to_parquet(out / "classes_far.parquet", index=False)
        top_pairs.to_parquet(out / "confusion_pairs_far.parquet", index=False)
        top_images.to_parquet(out / "top_images.parquet", index=False)
        df_top_pct.to_parquet(out / "top_pct.parquet", index=False)
    except Exception:
        # parquet optional
        pass

    print("Wrote:")
    print(f" - {out}")
    print(f" - {out/'summary.txt'}")
    print(f" - {out/'classes_far.csv'}")
    print(f" - {out/'confusion_pairs_far.csv'}")
    print(f" - {out/'top_images.csv'}")
    print(f" - {out/'top_pct.csv'}")

    # Print key headline for convenience
    print("\nHeadline:")
    print(f"  n={overall['n']}  median={overall['median']:.2f}km  p90={overall['p90']:.2f}km  far500={overall['far500']:.3f}")


if __name__ == "__main__":
    main()
