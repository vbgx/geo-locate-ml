from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


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

    # Compute uncertainty from logits
    # margin = top1 - top2 (higher => more confident)
    # prob from softmax on topk logits (approx, but enough for gating)
    top1 = topk_logit[:, 0]
    top2 = topk_logit[:, 1] if topk_logit.shape[1] > 1 else (top1 - 999.0)
    margin = top1 - top2

    # local softmax over topk
    x = topk_logit - topk_logit.max(axis=1, keepdims=True)
    ex = np.exp(x)
    probs = ex / np.clip(ex.sum(axis=1, keepdims=True), cfg.eps, None)
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

    # mix: base logits + alpha * feature_score (only if uncertain)
    scores = topk_logit.copy()
    scores[uncertain] = scores[uncertain] + float(cfg.alpha) * feature_score[uncertain]

    best_pos = np.argmax(scores, axis=1)
    return topk_idx[np.arange(topk_idx.shape[0]), best_pos]
