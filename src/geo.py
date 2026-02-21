from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


# ============================================================================
# Geo math (ex geo.py)
# ============================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================================
# Hierarchy helpers (ex hierarchy.py)
#   NOTE: To avoid circular imports, functions accept raw mapping dicts.
#   If you still have a LabelSpace/LabelsLite, pass its r7_to_r6 dict.
# ============================================================================

def r7_parent_tensor(
    r7_to_r6: Dict[int, int],
    num_r7: int,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Build tensor of shape [C7] mapping r7_idx -> r6_idx.
    """
    parents = [int(r7_to_r6[i]) for i in range(int(num_r7))]
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
    """
    logits_r7: [N, C7]
    r6_idx:    [N]
    r7_parent: [C7] (r7 -> r6)
    """
    mask = r7_parent.unsqueeze(0).eq(r6_idx.unsqueeze(1))
    return logits_r7.masked_fill(~mask, float(fill_value))


def hierarchical_predict(
    logits_r6: torch.Tensor,
    logits_r7: torch.Tensor,
    r7_parent: torch.Tensor,
    *,
    fill_value: float = -1e9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      pred_r6: [N]
      pred_r7: [N]
      masked_logits_r7: [N,C7]
    """
    pred_r6 = torch.argmax(logits_r6, dim=1)
    masked = mask_r7_logits(logits_r7, pred_r6, r7_parent, fill_value=fill_value)
    pred_r7 = torch.argmax(masked, dim=1)
    return pred_r6, pred_r7, masked


# ============================================================================
# Hard negative mining (ex hardneg.py)
# ============================================================================

@dataclass
class HardNegConfig:
    """
    Hard-negative mining configuration.

    We maintain a pool of sample_ids (image_id strings) that are considered "hard"
    (e.g., val dist_km > threshold_km). During training, oversample those ids
    using a WeightedRandomSampler.

    This targets the tail (P90) by repeatedly showing catastrophic mistakes.
    Pool is capped to avoid training on only hard examples.
    """
    enabled: bool = True
    threshold_km: float = 500.0
    boost: float = 4.0
    max_pool: int = 20_000
    min_count_to_enable: int = 100


def load_pool(path: Path) -> List[str]:
    """
    Loads a pool file.
    Accepts either:
      - a list: ["id1", "id2", ...]
      - a dict: {"ids": ["id1", ...]}
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "ids" in data and isinstance(data["ids"], list):
            return [str(x) for x in data["ids"] if str(x)]
        if isinstance(data, list):
            return [str(x) for x in data if str(x)]
    except Exception:
        return []
    return []


def save_pool(path: Path, ids: List[str]) -> None:
    """
    Saves pool as: {"ids": [...]}
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ids": [str(x) for x in ids if str(x)]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_pool(prev: List[str], new_ids: Iterable[str], cfg: HardNegConfig) -> List[str]:
    """
    Merge new ids into previous pool, keeping recency order (new first), unique, capped.
    """
    merged: List[str] = []
    seen = set()

    for x in list(new_ids):  # newest first
        sx = str(x)
        if sx and sx not in seen:
            merged.append(sx)
            seen.add(sx)

    for x in prev:
        sx = str(x)
        if sx and sx not in seen:
            merged.append(sx)
            seen.add(sx)

    if cfg.max_pool and len(merged) > int(cfg.max_pool):
        merged = merged[: int(cfg.max_pool)]

    return merged


def build_sample_weights(all_ids: List[str], pool_ids: List[str], cfg: HardNegConfig) -> Dict[str, float]:
    """
    Returns per-sample weights:
      - 1.0 for normal samples
      - cfg.boost for samples whose id is in pool_ids
    """
    pool = set(str(x) for x in pool_ids)
    boost = float(cfg.boost)
    out: Dict[str, float] = {}
    for sid in all_ids:
        s = str(sid)
        out[s] = boost if s in pool else 1.0
    return out
