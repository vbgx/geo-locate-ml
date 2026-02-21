from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model (ex model.py)
# ============================================================================

class MultiScaleCNN(nn.Module):
    """
    Resolution-agnostic CNN using AdaptiveAvgPool2d so 64/128/192 all work.

    Optionally predicts proxy attributes (multi-task):
      proxies: [p_water, p_urban, p_coastal, p_high_pop, p_high_elev] in [0,1]

    Outputs:
      - hierarchical_enabled=False:
          logits_r7                           OR (logits_r7, proxy)
      - hierarchical_enabled=True:
          (logits_r6, logits_r7)              OR (logits_r6, logits_r7, proxy)
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.30,
        num_proxies: int = 0,
        *,
        num_classes_r6: int = 0,
        hierarchical_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_proxies = int(num_proxies)
        self.hierarchical_enabled = bool(hierarchical_enabled)
        self.num_classes_r6 = int(num_classes_r6) if self.hierarchical_enabled else 0

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(float(dropout))

        # r7 head
        self.fc = nn.Linear(128, self.num_classes)

        # r6 head (optional)
        if self.hierarchical_enabled:
            if self.num_classes_r6 <= 0:
                raise ValueError("hierarchical_enabled=True requires num_classes_r6 > 0")
            self.fc_r6 = nn.Linear(128, self.num_classes_r6)
        else:
            self.fc_r6 = None

        # proxy head (optional)
        if self.num_proxies > 0:
            self.proxy_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(float(dropout) * 0.5),
                nn.Linear(64, self.num_proxies),
            )
        else:
            self.proxy_head = None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)  # [N,128]
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor):
        z = self._encode(x)

        logits_r7 = self.fc(z)

        if self.hierarchical_enabled:
            if self.fc_r6 is None:
                raise RuntimeError("hierarchical_enabled=True but fc_r6 is None")
            logits_r6 = self.fc_r6(z)

            if self.proxy_head is None:
                return logits_r6, logits_r7

            proxy_logits = self.proxy_head(z)
            proxy = torch.sigmoid(proxy_logits)  # [N,P] in [0,1]
            return logits_r6, logits_r7, proxy

        if self.proxy_head is None:
            return logits_r7

        proxy_logits = self.proxy_head(z)
        proxy = torch.sigmoid(proxy_logits)
        return logits_r7, proxy


# ============================================================================
# Loss (ex geo_loss.py)
# ============================================================================

class GeoSoftTargetLoss(nn.Module):
    """
    Distance-aware soft target loss:

      For each target y, define soft targets:
        p_j ∝ exp(-D[y,j] / tau)
      then minimize KL( p || softmax(logits) ).

    Args:
      distance_km: [C, C] distances (km)
      tau_km: temperature scale (km)
    """

    def __init__(self, distance_km: torch.Tensor, tau_km: float) -> None:
        super().__init__()
        if distance_km.dim() != 2 or distance_km.size(0) != distance_km.size(1):
            raise ValueError("distance_km must be square [C,C]")
        self.register_buffer("D", distance_km.float())
        self.tau = float(tau_km)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError("logits must be [B,C]")
        if targets.dim() != 1:
            raise ValueError("targets must be [B]")

        # logits: [B, C], targets: [B]
        B, C = logits.shape
        if self.D.size(0) != C:
            raise ValueError(f"D shape {tuple(self.D.shape)} incompatible with logits C={C}")

        # gather distances for each target row: [B, C]
        Dy = self.D[targets]  # buffer indexing
        p = torch.exp(-Dy / self.tau)
        p = p / (p.sum(dim=1, keepdim=True) + 1e-12)

        log_q = F.log_softmax(logits, dim=1)
        loss = -(p * log_q).sum(dim=1).mean()
        return loss


# ============================================================================
# Class weights (ex class_weights.py)
# ============================================================================

def compute_class_weights_from_parquet(
    parquet_path: str,
    split: str = "train",
    num_classes: Optional[int] = None,
    *,
    label_column: str = "label_idx",
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a dataset parquet.

    - Filters by split if the parquet includes a 'split' column.
    - Uses 1/sqrt(count) smoothing and normalizes weights to mean=1.

    Returns: torch.FloatTensor [C]
    """
    df = pd.read_parquet(parquet_path)

    if "split" in df.columns:
        df = df[df["split"] == split]

    if label_column not in df.columns:
        raise RuntimeError(f"Missing label column '{label_column}' in parquet: {parquet_path}")

    y = df[label_column].to_numpy(dtype=np.int64)

    if y.size == 0:
        raise RuntimeError(f"No rows found for split='{split}' in parquet: {parquet_path}")

    C = int(num_classes) if num_classes is not None else int(y.max() + 1)
    counts = np.bincount(y, minlength=C).astype(np.float64)

    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    w = w / (w.mean() + 1e-12)

    return torch.tensor(w, dtype=torch.float32)
