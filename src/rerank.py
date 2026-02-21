from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .geo import haversine_km
from .data import p


# ============================================================================
# Simple prior-based rerank
# ============================================================================

@dataclass(frozen=True)
class RerankConfig:
    beta: float = 0.35  # weight of log-prior
    eps: float = 1e-12


def rerank_topk(
    topk_idx: np.ndarray,      # shape: [N, K]
    topk_logits: np.ndarray,   # shape: [N, K]
    class_prior: np.ndarray,   # shape: [C] (probabilities, sum=1)
    cfg: RerankConfig = RerankConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - new_idx: [N, K] indices reordered by rerank score
      - new_scores: [N, K] rerank scores (higher better)
    """
    if topk_idx.shape != topk_logits.shape:
        raise ValueError("topk_idx and topk_logits must have same shape")

    N, K = topk_idx.shape
    C = class_prior.shape[0]

    if np.any(topk_idx < 0) or np.any(topk_idx >= C):
        raise ValueError("topk_idx contains out-of-range class indices")

    prior = class_prior[topk_idx]  # [N, K]
    log_prior = np.log(np.clip(prior, cfg.eps, 1.0))
    scores = topk_logits + float(cfg.beta) * log_prior

    order = np.argsort(-scores, axis=1)
    new_idx = np.take_along_axis(topk_idx, order, axis=1)
    new_scores = np.take_along_axis(scores, order, axis=1)
    return new_idx, new_scores


def build_class_prior_from_counts(counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    counts = counts.astype(np.float64)
    counts = np.clip(counts, 0.0, None)
    s = float(counts.sum())
    if s <= 0:
        raise ValueError("counts sum must be > 0")
    p0 = counts / s
    p0 = np.clip(p0, eps, 1.0)
    p0 = p0 / p0.sum()
    return p0


# ============================================================================
# Feature-based safe rerank (ex rerank_features.py)
# ============================================================================

@dataclass(frozen=True)
class FeatureRerankConfig:
    # Only rerank when model is uncertain (protects median)
    min_margin: float = 0.50          # logit(top1)-logit(top2) below this => uncertain
    min_top1_prob: float = 0.35       # if prob(top1) below => uncertain
    alpha: float = 0.20               # how much to mix feature score (0=no effect)

    # feature weights inside feature score
    w_coast: float = 0.15
    w_elev: float = 0.05
    w_pop: float = 0.10
    w_lc_water: float = 0.05

    eps: float = 1e-12


def load_h3_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"class_idx", "dist_coast_km", "elev_m", "pop", "landcover"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"h3_features missing columns: {sorted(missing)}")
    return df


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def build_feature_tables(h3_features: pd.DataFrame) -> Dict[str, np.ndarray]:
    df = h3_features.sort_values("class_idx").reset_index(drop=True)

    dist_coast = df["dist_coast_km"].to_numpy(dtype=np.float64)
    elev = df["elev_m"].to_numpy(dtype=np.float64)
    pop = df["pop"].to_numpy(dtype=np.float64)
    lc = df["landcover"].to_numpy(dtype=np.float64)

    z_coast = _zscore(dist_coast)
    z_elev = _zscore(elev)
    z_pop = _zscore(pop)

    is_water = (lc == 80).astype(np.float64)
    z_water = is_water

    return {
        "z_coast": z_coast,
        "z_elev": z_elev,
        "z_pop": z_pop,
        "z_water": z_water,
    }


def rerank_topk_with_features(
    topk_idx: np.ndarray,     # [N,K]
    topk_logit: np.ndarray,   # [N,K]
    tables: Dict[str, np.ndarray],
    cfg: FeatureRerankConfig = FeatureRerankConfig(),
) -> np.ndarray:
    """
    Safe rerank: only adjust uncertain predictions, and only a little (alpha).
    Feature-score is generic, meant to help tail without breaking easy cases.
    """
    if topk_idx.shape != topk_logit.shape:
        raise ValueError("topk_idx and topk_logit must have same shape")

    top1 = topk_logit[:, 0]
    top2 = topk_logit[:, 1] if topk_logit.shape[1] > 1 else (top1 - 999.0)
    margin = top1 - top2

    # local softmax over topk
    x = topk_logit - topk_logit.max(axis=1, keepdims=True)
    ex = np.exp(x)
    probs = ex / np.clip(ex.sum(axis=1, keepdims=True), float(cfg.eps), None)
    top1_prob = probs[:, 0]

    uncertain = (margin < float(cfg.min_margin)) | (top1_prob < float(cfg.min_top1_prob))

    z_coast = tables["z_coast"][topk_idx]
    z_elev = tables["z_elev"][topk_idx]
    z_pop = tables["z_pop"][topk_idx]
    z_water = tables["z_water"][topk_idx]

    feature_score = (
        - float(cfg.w_coast) * z_coast
        - float(cfg.w_elev) * np.abs(z_elev)
        + float(cfg.w_pop) * z_pop
        + float(cfg.w_lc_water) * z_water
    )

    scores = topk_logit.copy()
    scores[uncertain] = scores[uncertain] + float(cfg.alpha) * feature_score[uncertain]

    best_pos = np.argmax(scores, axis=1)
    return topk_idx[np.arange(topk_idx.shape[0]), best_pos]


# ============================================================================
# Offline eval CLI (ex rerank_eval.py)
# ============================================================================

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


def _load_labels_from_best_meta() -> Tuple[Dict[int, Tuple[float, float]], str]:
    meta_path = p("models", "best.json")
    if not meta_path.exists():
        raise FileNotFoundError("models/best.json not found. Run training to produce it.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    labels_path = Path(meta["labels_path"])
    obj = json.loads(labels_path.read_text(encoding="utf-8"))

    labels_list = obj.get("labels") or obj.get("h3_ids")
    if not isinstance(labels_list, list) or not labels_list:
        raise RuntimeError(f"Invalid labels file: missing 'labels' list: {labels_path}")

    import h3 as _h3
    idx_to_centroid: Dict[int, Tuple[float, float]] = {}
    for i, h3_id in enumerate([str(x) for x in labels_list]):
        lat, lon = _h3.cell_to_latlng(str(h3_id))
        idx_to_centroid[int(i)] = (float(lat), float(lon))

    parquet_path = str(meta.get("train_parquet") or meta.get("parquet_path") or "")
    return idx_to_centroid, parquet_path


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


def run_offline_feature_rerank_eval(
    *,
    topk_parquet: Path,
    h3_features: Path,
    dataset_parquet: Optional[Path] = None,
    cfg: FeatureRerankConfig = FeatureRerankConfig(),
) -> Dict[str, Any]:
    idx_to_centroid, default_dataset_parquet = _load_labels_from_best_meta()

    topk_path = Path(topk_parquet)
    if not topk_path.exists():
        raise FileNotFoundError(f"TopK parquet not found: {topk_path}")

    df = pd.read_parquet(topk_path)

    ds_path = Path(dataset_parquet) if dataset_parquet is not None else Path(default_dataset_parquet)
    if not ds_path.exists():
        raise FileNotFoundError(
            f"Dataset parquet not found: {ds_path}. "
            f"Pass --parquet_index or ensure models/best.json includes train_parquet/parquet_path."
        )

    ds = pd.read_parquet(ds_path, columns=["path", "lat", "lon", "split"])
    ds["image_id"] = ds["path"].apply(lambda s: Path(str(s)).stem)

    merged = df.merge(ds[["image_id", "lat", "lon", "split"]], on="image_id", how="left")
    if merged["lat"].isna().any() or merged["lon"].isna().any():
        missing = int(merged["lat"].isna().sum())
        raise RuntimeError(f"Missing lat/lon for {missing} rows after merge. Check image_id consistency.")

    lat_true = merged["lat"].to_numpy(dtype=np.float64)
    lon_true = merged["lon"].to_numpy(dtype=np.float64)

    y_true = merged["true_idx"].to_numpy(dtype=np.int64)
    y_pred_base = merged["pred_idx"].to_numpy(dtype=np.int64)

    errs_base = _geo_err_km(y_pred_base, lat_true, lon_true, idx_to_centroid)
    rep_base = _metrics_from_err(errs_base)
    acc_base = float((y_pred_base == y_true).mean()) if y_true.size else 0.0

    topk_idx, topk_logit = _extract_topk_cols(merged)

    h3f = load_h3_features(Path(h3_features))
    tables = build_feature_tables(h3f)

    y_pred_feat = rerank_topk_with_features(topk_idx, topk_logit, tables, cfg=cfg)
    errs_feat = _geo_err_km(y_pred_feat, lat_true, lon_true, idx_to_centroid)
    rep_feat = _metrics_from_err(errs_feat)
    acc_feat = float((y_pred_feat == y_true).mean()) if y_true.size else 0.0

    return {
        "baseline": {"acc1": acc_base, "report": rep_base},
        "rerank": {"acc1": acc_feat, "report": rep_feat},
        "delta": {
            "acc1": acc_feat - acc_base,
            "median_km": rep_feat.median_km - rep_base.median_km,
            "p90_km": rep_feat.p90_km - rep_base.p90_km,
            "p95_km": rep_feat.p95_km - rep_base.p95_km,
            "far200": rep_feat.far200 - rep_base.far200,
            "far500": rep_feat.far500 - rep_base.far500,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate offline feature reranking on saved val_topk.parquet")
    ap.add_argument("--topk_parquet", required=True, help="Path to val_topk.parquet")
    ap.add_argument("--parquet_index", default=None, help="Dataset parquet path (default from models/best.json)")
    ap.add_argument("--h3_features", required=True, help="Path to data/index/h3_features.parquet")

    ap.add_argument("--alpha", type=float, default=0.20)
    ap.add_argument("--min_margin", type=float, default=0.50)
    ap.add_argument("--min_top1_prob", type=float, default=0.35)

    ap.add_argument("--w_coast", type=float, default=0.15)
    ap.add_argument("--w_elev", type=float, default=0.05)
    ap.add_argument("--w_pop", type=float, default=0.10)
    ap.add_argument("--w_lc_water", type=float, default=0.05)

    args = ap.parse_args()

    cfg = FeatureRerankConfig(
        alpha=float(args.alpha),
        min_margin=float(args.min_margin),
        min_top1_prob=float(args.min_top1_prob),
        w_coast=float(args.w_coast),
        w_elev=float(args.w_elev),
        w_pop=float(args.w_pop),
        w_lc_water=float(args.w_lc_water),
    )

    out = run_offline_feature_rerank_eval(
        topk_parquet=Path(args.topk_parquet),
        h3_features=Path(args.h3_features),
        dataset_parquet=Path(args.parquet_index) if args.parquet_index else None,
        cfg=cfg,
    )

    rep_base: Report = out["baseline"]["report"]
    rep_feat: Report = out["rerank"]["report"]
    acc_base = float(out["baseline"]["acc1"])
    acc_feat = float(out["rerank"]["acc1"])
    delta = out["delta"]

    print(rep_base.fmt("BASELINE (model pred_idx)"))
    print(f"  acc@1={acc_base:.4f}\n")

    print(rep_feat.fmt("RERANK (SAFE gated features-only)"))
    print(f"  acc@1={acc_feat:.4f}\n")

    print("DELTA (rerank - baseline)")
    print(f"  acc@1:   {delta['acc1']:+.4f}")
    print(f"  median:  {delta['median_km']:+.2f} km")
    print(f"  p90:     {delta['p90_km']:+.2f} km")
    print(f"  p95:     {delta['p95_km']:+.2f} km")
    print(f"  far200:  {delta['far200']:+.3f}")
    print(f"  far500:  {delta['far500']:+.3f}")


if __name__ == "__main__":
    main()
