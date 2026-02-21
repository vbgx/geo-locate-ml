from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import torch

def compute_class_weights_from_parquet(
    parquet_path: str,
    split: str = "train",
    num_classes: Optional[int] = None,
    *,
    label_column: str = "label_idx",
) -> torch.Tensor:
    df = pd.read_parquet(parquet_path)
    if "split" in df.columns:
        df = df[df["split"] == split]
    if label_column not in df.columns:
        raise RuntimeError(f"Missing label column '{label_column}' in parquet: {parquet_path}")
    y = df[label_column].to_numpy(dtype=np.int64)

    C = int(num_classes) if num_classes is not None else int(y.max() + 1)
    counts = np.bincount(y, minlength=C).astype(np.float64)

    # Inverse frequency with smoothing to avoid huge weights
    # w_c ∝ 1 / sqrt(count_c)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    w = w / w.mean()  # normalize around 1.0

    return torch.tensor(w, dtype=torch.float32)
