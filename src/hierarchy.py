from __future__ import annotations

from typing import Tuple

import torch

from .labels import LabelSpace


def r7_parent_tensor(labels: LabelSpace, device: str | torch.device | None = None) -> torch.Tensor:
    parents = [int(labels.r7_to_r6[i]) for i in range(len(labels.h3_ids))]
    t = torch.tensor(parents, dtype=torch.long)
    if device is not None:
        t = t.to(device)
    return t


def mask_r7_logits(
    logits_r7: torch.Tensor,
    r6_idx: torch.Tensor,
    r7_parent: torch.Tensor,
    *,
    fill_value: float = -1e9,
) -> torch.Tensor:
    # logits_r7: [N, C], r6_idx: [N], r7_parent: [C]
    mask = r7_parent.unsqueeze(0).eq(r6_idx.unsqueeze(1))
    return logits_r7.masked_fill(~mask, float(fill_value))


def hierarchical_predict(
    logits_r6: torch.Tensor,
    logits_r7: torch.Tensor,
    r7_parent: torch.Tensor,
    *,
    fill_value: float = -1e9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_r6 = torch.argmax(logits_r6, dim=1)
    masked = mask_r7_logits(logits_r7, pred_r6, r7_parent, fill_value=fill_value)
    pred_r7 = torch.argmax(masked, dim=1)
    return pred_r6, pred_r7, masked
