from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class HardNegConfig:
    """
    Hard-negative mining configuration.

    Behavior:
      - We maintain a pool of sample_ids (image_id strings) that are considered "hard"
        (e.g., val dist_km > threshold_km).
      - During training, we oversample those ids using a WeightedRandomSampler.

    Notes:
      - This approach targets the tail (P90) by repeatedly showing catastrophic mistakes.
      - Keep pool capped to avoid turning training into "only hard examples".
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

    # newest first
    for x in list(new_ids):
        sx = str(x)
        if sx and sx not in seen:
            merged.append(sx)
            seen.add(sx)

    # then previous
    for x in prev:
        sx = str(x)
        if sx and sx not in seen:
            merged.append(sx)
            seen.add(sx)

    # cap
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
