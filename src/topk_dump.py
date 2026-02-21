from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TopKDumpConfig:
    k: int = 10


def dump_topk_parquet(
    out_path: Path,
    image_ids: list[str],
    true_idx: np.ndarray,          # [N]
    topk_idx: np.ndarray,          # [N, K]
    topk_scores: np.ndarray,       # [N, K] (logits or probs)
    pred_idx: np.ndarray,          # [N]
    geo_err_km: Optional[np.ndarray] = None,  # [N]
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    N = len(image_ids)
    if true_idx.shape[0] != N or pred_idx.shape[0] != N:
        raise ValueError("N mismatch in arrays")
    if topk_idx.shape[0] != N or topk_scores.shape[0] != N:
        raise ValueError("N mismatch in topk arrays")

    K = topk_idx.shape[1]

    df = pd.DataFrame(
        {
            "image_id": image_ids,
            "true_idx": true_idx.astype(int),
            "pred_idx": pred_idx.astype(int),
        }
    )

    for j in range(K):
        df[f"top{j+1}_idx"] = topk_idx[:, j].astype(int)
        df[f"top{j+1}_score"] = topk_scores[:, j].astype(float)

    if geo_err_km is not None:
        df["geo_err_km"] = geo_err_km.astype(float)

    df.to_parquet(out_path, index=False)
