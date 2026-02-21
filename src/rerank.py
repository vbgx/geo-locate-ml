from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


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

    # log prior per candidate
    prior = class_prior[topk_idx]  # [N, K]
    log_prior = np.log(np.clip(prior, cfg.eps, 1.0))

    scores = topk_logits + cfg.beta * log_prior

    # sort descending within each row
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
    p = counts / s
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return p
