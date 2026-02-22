from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model
# ============================================================================


class MultiScaleCNN(nn.Module):
    """
    Resolution-agnostic CNN using AdaptiveAvgPool2d so 64/128/192 all work.

    Optionally predicts proxy attributes (multi-task).

    Proxy head is generic regression (float targets). You can mix:
      - z-scored continuous values (recommended)
      - [0,1] fractions/scores (also OK as regression)

    Outputs:
      - hierarchical_enabled=False:
          logits_r7                           OR (logits_r7, proxy_pred)
      - hierarchical_enabled=True:
          (logits_r6, logits_r7)              OR (logits_r6, logits_r7, proxy_pred)
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.30,
        num_proxies: int = 0,
        *,
        num_classes_r6: int = 0,
        hierarchical_enabled: bool = False,
        proxy_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_proxies = int(num_proxies)
        self.hierarchical_enabled = bool(hierarchical_enabled)
        self.num_classes_r6 = int(num_classes_r6) if self.hierarchical_enabled else 0

        # Simple, stable backbone (keep it lightweight; your dataset is big)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

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

        # proxy head (optional) - regression (NO sigmoid)
        if self.num_proxies > 0:
            h = int(proxy_hidden)
            h = 32 if h < 32 else h
            self.proxy_head = nn.Sequential(
                nn.Linear(128, h),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout) * 0.5),
                nn.Linear(h, self.num_proxies),
            )
        else:
            self.proxy_head = None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x), inplace=True))
        x = self.pool(F.relu(self.conv2(x), inplace=True))
        x = self.pool(F.relu(self.conv3(x), inplace=True))
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

            proxy_pred = self.proxy_head(z)  # regression (no sigmoid)
            return logits_r6, logits_r7, proxy_pred

        if self.proxy_head is None:
            return logits_r7

        proxy_pred = self.proxy_head(z)  # regression (no sigmoid)
        return logits_r7, proxy_pred


# ============================================================================
# Proxy loss
# ============================================================================


@dataclass(frozen=True)
class ProxyLossConfig:
    """
    Weighted masked regression loss for proxies.

    - Uses SmoothL1 (Huber) by default to avoid proxy scale spikes.
    - `mask` is 0/1 per proxy dimension to ignore missing values.
    - `per_proxy_weights` lets you overweight some proxies.
    """

    enabled: bool = False
    weight: float = 1.0  # global multiplier
    huber_beta: float = 1.0
    # if provided: list/1D tensor of shape [P]
    per_proxy_weights: Optional[torch.Tensor] = None


class ProxyRegressionLoss(nn.Module):
    def __init__(self, cfg: ProxyLossConfig, num_proxies: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_proxies = int(num_proxies)
        if self.num_proxies <= 0:
            raise ValueError("ProxyRegressionLoss requires num_proxies > 0")

        if cfg.per_proxy_weights is not None:
            w = cfg.per_proxy_weights.detach().float().view(-1)
            if int(w.numel()) != int(self.num_proxies):
                raise ValueError(
                    f"per_proxy_weights shape mismatch: got {int(w.numel())} expected {self.num_proxies}"
                )
            self.register_buffer("w", w)
        else:
            self.register_buffer("w", torch.ones((self.num_proxies,), dtype=torch.float32))

    def forward(
        self,
        pred: torch.Tensor,  # [B,P]
        target: torch.Tensor,  # [B,P]
        mask: Optional[torch.Tensor] = None,  # [B,P] 0/1
    ) -> torch.Tensor:
        if pred.dim() != 2 or pred.size(1) != self.num_proxies:
            raise ValueError(f"pred must be [B,{self.num_proxies}]")
        if target.shape != pred.shape:
            raise ValueError("target must match pred shape")

        if mask is None:
            mask_t = torch.ones_like(pred, dtype=torch.float32)
        else:
            if mask.shape != pred.shape:
                raise ValueError("mask must match pred shape")
            mask_t = mask.float()

        # SmoothL1 per-element -> apply weights/mask -> normalize by number of present values
        loss_el = F.smooth_l1_loss(pred, target, beta=float(self.cfg.huber_beta), reduction="none")
        loss_el = loss_el * self.w.view(1, -1) * mask_t

        denom = torch.clamp(mask_t.sum(), min=1.0)
        return loss_el.sum() / denom


# ============================================================================
# Loss
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

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        allowed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError("logits must be [B,C]")
        if targets.dim() != 1:
            raise ValueError("targets must be [B]")

        B, C = logits.shape
        if self.D.size(0) != C:
            raise ValueError(f"D shape {tuple(self.D.shape)} incompatible with logits C={C}")

        Dy = self.D[targets]  # [B, C]
        p = torch.exp(-Dy / self.tau)  # [B, C]

        if allowed_mask is not None:
            if allowed_mask.shape != p.shape:
                raise ValueError(
                    f"allowed_mask must be [B,C], got {tuple(allowed_mask.shape)} vs {tuple(p.shape)}"
                )
            allowed = allowed_mask.to(dtype=torch.bool, device=p.device)
            # zero-out forbidden mass and renormalize over allowed classes only
            p = p.masked_fill(~allowed, 0.0)
            p = p / (p.sum(dim=1, keepdim=True) + 1e-12)
        else:
            p = p / (p.sum(dim=1, keepdim=True) + 1e-12)

        log_q = F.log_softmax(logits, dim=1)
        loss = -(p * log_q).sum(dim=1).mean()
        return loss


# ============================================================================
# Class weights
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