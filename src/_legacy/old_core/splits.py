from __future__ import annotations
import hashlib
from typing import Tuple

import pandas as pd

def _stable_hash_to_unit_interval(s: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{s}".encode("utf-8")).hexdigest()
    # take first 12 hex chars -> int -> [0,1)
    v = int(h[:12], 16)
    return (v % 10_000_000) / 10_000_000.0

def assign_split_by_sequence(
    df: pd.DataFrame,
    seed: int,
    p_train: float,
    p_val: float,
    p_test: float,
) -> pd.DataFrame:
    assert abs((p_train + p_val + p_test) - 1.0) < 1e-6, "splits must sum to 1.0"
    df = df.copy()

    # sequence_id can be None; fallback to image id in that case (still deterministic)
    seq = df["sequence_id"].fillna(df["id"]).astype(str)

    def pick(sid: str) -> str:
        u = _stable_hash_to_unit_interval(sid, seed)
        if u < p_train:
            return "train"
        if u < p_train + p_val:
            return "val"
        return "test"

    df["split"] = seq.map(pick)
    return df
