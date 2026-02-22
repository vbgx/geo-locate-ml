from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


# ============================================================================
# Geo math
# ============================================================================


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two lat/lon points in kilometers.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2.0) ** 2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


# ============================================================================
# Hierarchy helpers
# ============================================================================


def r7_parent_tensor(
    r7_to_r6: Dict[int, int],
    num_r7: int,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """
    Build tensor of shape [C7] mapping r7_idx -> r6_idx.

    Requirements:
      - keys cover [0..num_r7-1]
    """
    C7 = int(num_r7)
    parents = [int(r7_to_r6[i]) for i in range(C7)]
    t = torch.tensor(parents, dtype=torch.long)
    if device is not None:
        t = t.to(device)
    return t


@torch.no_grad()
def validate_hierarchy_batch(
    *,
    y7: torch.Tensor,
    y6: torch.Tensor,
    r7_parent: torch.Tensor,
    fail_hard: bool = True,
    max_print: int = 12,
) -> bool:
    """
    Validates that for each sample i:
      r7_parent[y7[i]] == y6[i]

    Returns True if OK, False otherwise (unless fail_hard=True then raises).
    """
    if y7.ndim != 1 or y6.ndim != 1:
        raise ValueError(f"validate_hierarchy_batch expects 1D y7/y6, got y7={tuple(y7.shape)} y6={tuple(y6.shape)}")
    if y7.numel() != y6.numel():
        raise ValueError(f"validate_hierarchy_batch expects same batch size, got {y7.numel()} vs {y6.numel()}")
    if r7_parent.ndim != 1:
        raise ValueError(f"validate_hierarchy_batch expects 1D r7_parent, got {tuple(r7_parent.shape)}")

    # ensure dtype
    y7i = y7.to(dtype=torch.long)
    y6i = y6.to(dtype=torch.long)
    p = r7_parent.to(dtype=torch.long)

    parent_of_y = p[y7i]  # (B,)
    bad = parent_of_y.ne(y6i)

    if not bool(bad.any().item()):
        return True

    idxs = bad.nonzero(as_tuple=False).view(-1).tolist()
    msg_lines = [
        f"❌ hierarchy mismatch: {len(idxs)}/{int(y7i.numel())} samples have r7_parent[y7] != y6"
    ]
    for i in idxs[: int(max_print)]:
        msg_lines.append(
            f"  i={i} y7={int(y7i[i])} y6={int(y6i[i])} parent(y7)={int(parent_of_y[i])}"
        )
    msg = "\n".join(msg_lines)

    if fail_hard:
        raise RuntimeError(msg)

    print(msg)
    return False


def mask_r7_logits(
    logits_r7: torch.Tensor,
    r6_idx: torch.Tensor,
    r7_parent: torch.Tensor,
    *,
    fill_value: float = -1e9,
    keep_target: bool = False,
    y7_target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mask logits_r7 so only classes whose parent == r6_idx remain unmasked.

    logits_r7: [B, C7]
    r6_idx:    [B]
    r7_parent: [C7] mapping r7 -> r6

    fill_value: value used to fill masked entries (default -1e9)
    keep_target: if True, ensures the provided y7_target stays unmasked even if mapping mismatch
                 (THIS IS FOR DEBUGGING ONLY).
    y7_target: required if keep_target=True
    """
    if logits_r7.ndim != 2:
        raise ValueError(f"mask_r7_logits expects logits [B,C], got {tuple(logits_r7.shape)}")
    B, C7 = logits_r7.shape

    r6 = r6_idx.to(device=logits_r7.device, dtype=torch.long)
    p = r7_parent.to(device=logits_r7.device, dtype=torch.long)

    if p.numel() != C7:
        raise ValueError(f"mask_r7_logits r7_parent length {p.numel()} != C7 {C7}")

    # mask True for allowed classes
    # shape: [B, C7]
    allowed = p.unsqueeze(0).eq(r6.unsqueeze(1))

    masked = logits_r7.masked_fill(~allowed, float(fill_value))

    if keep_target:
        if y7_target is None:
            raise ValueError("keep_target=True requires y7_target")
        y7t = y7_target.to(device=logits_r7.device, dtype=torch.long)
        if y7t.numel() != B:
            raise ValueError(f"y7_target batch mismatch: {y7t.numel()} != {B}")
        # force the target index to remain original logit (unmasked)
        masked[torch.arange(B, device=logits_r7.device), y7t] = logits_r7[
            torch.arange(B, device=logits_r7.device), y7t
        ]

    return masked


def hierarchical_predict(
    logits_r6: torch.Tensor,
    logits_r7: torch.Tensor,
    r7_parent: torch.Tensor,
    *,
    fill_value: float = -1e9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hierarchical prediction:
      1) predict r6 via argmax(logits_r6)
      2) mask r7 logits to predicted r6 bucket
      3) predict r7 via argmax(masked_r7)

    Returns:
      pred_r6: [B]
      pred_r7: [B]
      masked_logits_r7: [B,C7]
    """
    if logits_r6.ndim != 2 or logits_r7.ndim != 2:
        raise ValueError("hierarchical_predict expects 2D logits for both r6 and r7")
    pred_r6 = torch.argmax(logits_r6, dim=1)
    masked = mask_r7_logits(logits_r7, pred_r6, r7_parent, fill_value=fill_value)
    pred_r7 = torch.argmax(masked, dim=1)
    return pred_r6, pred_r7, masked


# ============================================================================
# Hard negative mining
# ============================================================================


@dataclass
class HardNegConfig:
    """
    Maintain a pool of sample_ids considered "hard" (e.g. val dist_km > threshold_km).
    During training, oversample those ids using a WeightedRandomSampler.

    Pool is capped for stability.
    """
    enabled: bool = True
    threshold_km: float = 500.0
    boost: float = 4.0
    max_pool: int = 20_000
    min_count_to_enable: int = 100


def load_pool(path: Path) -> List[str]:
    """
    Accepts:
      - ["id1", "id2", ...]
      - {"ids": ["id1", ...]}
    """
    if not Path(path).exists():
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, dict):
        ids = data.get("ids", [])
        if isinstance(ids, list):
            return [str(x) for x in ids if str(x)]
        return []

    if isinstance(data, list):
        return [str(x) for x in data if str(x)]

    return []


def save_pool(path: Path, ids: Sequence[str]) -> None:
    """
    Saves pool as {"ids": [...]}
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ids": [str(x) for x in ids if str(x)]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_pool(prev: Sequence[str], new_ids: Iterable[str], cfg: HardNegConfig) -> List[str]:
    """
    Merge new ids into previous pool, keeping recency order (new first), unique, capped.
    """
    merged: List[str] = []
    seen: set[str] = set()

    for x in list(new_ids):  # treat iterable as newest-first
        sx = str(x)
        if sx and sx not in seen:
            merged.append(sx)
            seen.add(sx)

    for x in prev:
        sx = str(x)
        if sx and sx not in seen:
            merged.append(sx)
            seen.add(sx)

    cap = int(cfg.max_pool) if cfg.max_pool else 0
    if cap > 0 and len(merged) > cap:
        merged = merged[:cap]

    return merged


def build_sample_weights(all_ids: Sequence[str], pool_ids: Sequence[str], cfg: HardNegConfig) -> Dict[str, float]:
    """
    Weight = cfg.boost if id in pool else 1.0
    """
    pool = set(str(x) for x in pool_ids if str(x))
    boost = float(cfg.boost)
    out: Dict[str, float] = {}
    for sid in all_ids:
        s = str(sid)
        out[s] = boost if s in pool else 1.0
    return out