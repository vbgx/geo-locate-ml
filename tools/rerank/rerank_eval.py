from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.geo import haversine_km
from src.labels import LabelSpace
from src.paths import p
from tools.rerank.rerank_features import (
    FeatureRerankConfig,
    build_feature_tables,
    load_h3_features,
    rerank_topk_with_features,
)


@dataclass(frozen=True)
class Report:
    n: int
    median_km: float
    p90_km: float
    p95_km: float
    mean_km: float
    far200: float
    far500: float

    def fmt(self, title: str) -> str:
        return (
            f"{title}\n"
            f"  n={self.n}\n"
            f"  mean_km={self.mean_km:.2f}\n"
            f"  median_km={self.median_km:.2f}\n"
            f"  p90_km={self.p90_km:.2f}\n"
            f"  p95_km={self.p95_km:.2f}\n"
            f"  far>200km={self.far200:.3f}\n"
            f"  far>500km={self.far500:.3f}\n"
        )


def _metrics_from_err(errs_km: np.ndarray) -> Report:
    a = np.asarray(errs_km, dtype=np.float64)
    if a.size == 0:
        return Report(n=0, mean_km=0.0, median_km=0.0, p90_km=0.0, p95_km=0.0, far200=0.0, far500=0.0)
    return Report(
        n=int(a.size),
        mean_km=float(a.mean()),
        median_km=float(np.median(a)),
        p90_km=float(np.percentile(a, 90)),
        p95_km=float(np.percentile(a, 95)),
        far200=float((a > 200.0).mean()),
        far500=float((a > 500.0).mean()),
    )


def _load_labels_from_best_meta() -> LabelSpace:
    meta_path = p("models", "best.json")
    if not meta_path.exists():
        raise FileNotFoundError("models/best.json not found. Run training to produce it.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    labels_path = Path(meta["labels_path"])
    return LabelSpace.from_json(labels_path.read_text(encoding="utf-8"))


def _extract_topk_cols(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    idx_cols = sorted(
        [c for c in df.columns if c.startswith("top") and c.endswith("_idx")],
        key=lambda x: int(x.replace("top", "").replace("_idx", "")),
    )
    logit_cols = sorted(
        [c for c in df.columns if c.startswith("top") and c.endswith("_logit")],
        key=lambda x: int(x.replace("top", "").replace("_logit", "")),
    )
    if not idx_cols or not logit_cols or len(idx_cols) != len(logit_cols):
        raise RuntimeError("Missing topk columns in parquet. Expected top1_idx/top1_logit ...")
    topk_idx = df[idx_cols].to_numpy(dtype=np.int64)
    topk_logit = df[logit_cols].to_numpy(dtype=np.float64)
    return topk_idx, topk_logit


def _geo_err_km(
    y_pred: np.ndarray,
    lat_true: np.ndarray,
    lon_true: np.ndarray,
    idx_to_centroid: Dict[int, Tuple[float, float]],
) -> np.ndarray:
    errs = np.zeros(len(y_pred), dtype=np.float64)
    for i, (yp, lat, lon) in enumerate(zip(y_pred, lat_true, lon_true)):
        plat, plon = idx_to_centroid[int(yp)]
        errs[i] = haversine_km(float(lat), float(lon), float(plat), float(plon))
    return errs


def main():
    ap = argparse.ArgumentParser(description="Evaluate offline reranking on saved val_topk.parquet")
    ap.add_argument("--topk_parquet", required=True, help="Path to val_topk.parquet")
    ap.add_argument("--parquet_index", default=None, help="Dataset parquet index path (default from models/best.json)")

    ap.add_argument("--h3_features", required=True, help="Path to data/index/h3_features.parquet to enable feature rerank.")
    ap.add_argument("--alpha", type=float, default=0.20)
    ap.add_argument("--min_margin", type=float, default=0.50)
    ap.add_argument("--min_top1_prob", type=float, default=0.35)

    ap.add_argument("--w_coast", type=float, default=0.15)
    ap.add_argument("--w_elev", type=float, default=0.05)
    ap.add_argument("--w_pop", type=float, default=0.10)
    ap.add_argument("--w_lc_water", type=float, default=0.05)

    args = ap.parse_args()

    labels = _load_labels_from_best_meta()

    topk_path = Path(args.topk_parquet)
    if not topk_path.exists():
        raise FileNotFoundError(f"TopK parquet not found: {topk_path}")

    df = pd.read_parquet(topk_path)

    meta_path = p("models", "best.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dataset_parquet = args.parquet_index or meta["parquet_path"]

    ds = pd.read_parquet(dataset_parquet, columns=["path", "lat", "lon", "split"])
    ds["image_id"] = ds["path"].apply(lambda s: Path(str(s)).stem)

    merged = df.merge(ds[["image_id", "lat", "lon", "split"]], on="image_id", how="left")
    if merged["lat"].isna().any() or merged["lon"].isna().any():
        missing = int(merged["lat"].isna().sum())
        raise RuntimeError(f"Missing lat/lon for {missing} rows after merge. Check image_id consistency.")

    lat_true = merged["lat"].to_numpy(dtype=np.float64)
    lon_true = merged["lon"].to_numpy(dtype=np.float64)

    y_true = merged["true_idx"].to_numpy(dtype=np.int64)
    y_pred_base = merged["pred_idx"].to_numpy(dtype=np.int64)

    errs_base = _geo_err_km(y_pred_base, lat_true, lon_true, labels.idx_to_centroid)
    rep_base = _metrics_from_err(errs_base)
    acc_base = float((y_pred_base == y_true).mean()) if y_true.size else 0.0

    topk_idx, topk_logit = _extract_topk_cols(merged)

    h3f = load_h3_features(Path(args.h3_features))
    tables = build_feature_tables(h3f)

    cfg = FeatureRerankConfig(
        alpha=float(args.alpha),
        min_margin=float(args.min_margin),
        min_top1_prob=float(args.min_top1_prob),
        w_coast=float(args.w_coast),
        w_elev=float(args.w_elev),
        w_pop=float(args.w_pop),
        w_lc_water=float(args.w_lc_water),
    )

    y_pred_feat = rerank_topk_with_features(topk_idx, topk_logit, tables, cfg=cfg)
    errs_feat = _geo_err_km(y_pred_feat, lat_true, lon_true, labels.idx_to_centroid)
    rep_feat = _metrics_from_err(errs_feat)
    acc_feat = float((y_pred_feat == y_true).mean()) if y_true.size else 0.0

    print(rep_base.fmt("BASELINE (model pred_idx)"))
    print(f"  acc@1={acc_base:.4f}\n")

    print(rep_feat.fmt("RERANK (SAFE gated features-only)"))
    print(f"  acc@1={acc_feat:.4f}\n")

    print("DELTA (rerank - baseline)")
    print(f"  acc@1:   {acc_feat - acc_base:+.4f}")
    print(f"  median:  {rep_feat.median_km - rep_base.median_km:+.2f} km")
    print(f"  p90:     {rep_feat.p90_km - rep_base.p90_km:+.2f} km")
    print(f"  p95:     {rep_feat.p95_km - rep_base.p95_km:+.2f} km")
    print(f"  far200:  {rep_feat.far200 - rep_base.far200:+.3f}")
    print(f"  far500:  {rep_feat.far500 - rep_base.far500:+.3f}")


if __name__ == "__main__":
    main()
