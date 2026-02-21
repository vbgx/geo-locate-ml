from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import h3
import pandas as pd

@dataclass
class LabelSpace:
    h3_ids: List[str]
    h3_to_idx: Dict[str, int]
    idx_to_h3: Dict[int, str]
    idx_to_centroid: Dict[int, Tuple[float, float]]  # (lat, lon)
    parent_res: int = 6
    h3_ids_r6: List[str] = field(default_factory=list)
    h3_r6_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_h3_r6: Dict[int, str] = field(default_factory=dict)
    idx_to_centroid_r6: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    r7_to_r6: Dict[int, int] = field(default_factory=dict)
    r6_to_r7: Dict[int, List[int]] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "h3_ids": self.h3_ids,
            "h3_to_idx": self.h3_to_idx,
            "idx_to_h3": {str(k): v for k, v in self.idx_to_h3.items()},
            "idx_to_centroid": {str(k): [lat, lon] for k, (lat, lon) in self.idx_to_centroid.items()},
            "parent_res": int(self.parent_res),
            "h3_ids_r6": self.h3_ids_r6,
            "h3_r6_to_idx": self.h3_r6_to_idx,
            "idx_to_h3_r6": {str(k): v for k, v in self.idx_to_h3_r6.items()},
            "idx_to_centroid_r6": {str(k): [lat, lon] for k, (lat, lon) in self.idx_to_centroid_r6.items()},
            "r7_to_r6": {str(k): v for k, v in self.r7_to_r6.items()},
            "r6_to_r7": {str(k): v for k, v in self.r6_to_r7.items()},
        }
        return json.dumps(payload, indent=2)

    @staticmethod
    def from_json(text: str) -> "LabelSpace":
        obj = json.loads(text)
        h3_ids = obj["h3_ids"]
        h3_to_idx = {k: int(v) for k, v in obj["h3_to_idx"].items()}
        idx_to_h3 = {int(k): v for k, v in obj["idx_to_h3"].items()}
        idx_to_centroid = {int(k): (float(v[0]), float(v[1])) for k, v in obj["idx_to_centroid"].items()}
        parent_res = int(obj.get("parent_res", 6))

        if "h3_ids_r6" in obj and "h3_r6_to_idx" in obj and "r7_to_r6" in obj:
            h3_ids_r6 = list(obj["h3_ids_r6"])
            h3_r6_to_idx = {k: int(v) for k, v in obj["h3_r6_to_idx"].items()}
            idx_to_h3_r6 = {int(k): v for k, v in obj["idx_to_h3_r6"].items()}
            idx_to_centroid_r6 = {int(k): (float(v[0]), float(v[1])) for k, v in obj["idx_to_centroid_r6"].items()}
            r7_to_r6 = {int(k): int(v) for k, v in obj["r7_to_r6"].items()}
            r6_to_r7 = {int(k): [int(x) for x in v] for k, v in obj["r6_to_r7"].items()}
        else:
            (
                h3_ids_r6,
                h3_r6_to_idx,
                idx_to_h3_r6,
                idx_to_centroid_r6,
                r7_to_r6,
                r6_to_r7,
            ) = build_h3_hierarchy(h3_ids, parent_res=parent_res)
        return LabelSpace(
            h3_ids=h3_ids,
            h3_to_idx=h3_to_idx,
            idx_to_h3=idx_to_h3,
            idx_to_centroid=idx_to_centroid,
            parent_res=parent_res,
            h3_ids_r6=h3_ids_r6,
            h3_r6_to_idx=h3_r6_to_idx,
            idx_to_h3_r6=idx_to_h3_r6,
            idx_to_centroid_r6=idx_to_centroid_r6,
            r7_to_r6=r7_to_r6,
            r6_to_r7=r6_to_r7,
        )

def compute_h3(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    df = df.copy()
    df["h3_id"] = df.apply(lambda r: h3.latlng_to_cell(r["lat"], r["lon"], resolution), axis=1)
    return df

def filter_sparse_cells(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    counts = df["h3_id"].value_counts()
    keep = set(counts[counts >= min_samples].index.tolist())
    return df[df["h3_id"].isin(keep)].copy()

def build_h3_hierarchy(
    h3_ids: List[str],
    parent_res: int = 6,
) -> Tuple[
    List[str],
    Dict[str, int],
    Dict[int, str],
    Dict[int, Tuple[float, float]],
    Dict[int, int],
    Dict[int, List[int]],
]:
    parents: List[str] = []
    for h in h3_ids:
        try:
            p = h3.cell_to_parent(h, int(parent_res))
        except Exception:
            p = h
        parents.append(str(p))

    h3_ids_r6 = sorted(set(parents))
    h3_r6_to_idx = {h: i for i, h in enumerate(h3_ids_r6)}
    idx_to_h3_r6 = {i: h for h, i in h3_r6_to_idx.items()}

    idx_to_centroid_r6: Dict[int, Tuple[float, float]] = {}
    for i, h in idx_to_h3_r6.items():
        lat, lon = h3.cell_to_latlng(h)
        idx_to_centroid_r6[i] = (float(lat), float(lon))

    r7_to_r6: Dict[int, int] = {}
    r6_to_r7: Dict[int, List[int]] = {i: [] for i in idx_to_h3_r6.keys()}
    for r7_idx, parent in enumerate(parents):
        r6_idx = int(h3_r6_to_idx[parent])
        r7_to_r6[int(r7_idx)] = r6_idx
        r6_to_r7[r6_idx].append(int(r7_idx))

    return h3_ids_r6, h3_r6_to_idx, idx_to_h3_r6, idx_to_centroid_r6, r7_to_r6, r6_to_r7


def build_label_space(df: pd.DataFrame, parent_res: int = 6) -> LabelSpace:
    h3_ids = sorted(df["h3_id"].unique().tolist())
    h3_to_idx = {h: i for i, h in enumerate(h3_ids)}
    idx_to_h3 = {i: h for h, i in h3_to_idx.items()}

    idx_to_centroid = {}
    for i, h in idx_to_h3.items():
        lat, lon = h3.cell_to_latlng(h)
        idx_to_centroid[i] = (float(lat), float(lon))

    (
        h3_ids_r6,
        h3_r6_to_idx,
        idx_to_h3_r6,
        idx_to_centroid_r6,
        r7_to_r6,
        r6_to_r7,
    ) = build_h3_hierarchy(h3_ids, parent_res=parent_res)

    return LabelSpace(
        h3_ids=h3_ids,
        h3_to_idx=h3_to_idx,
        idx_to_h3=idx_to_h3,
        idx_to_centroid=idx_to_centroid,
        parent_res=int(parent_res),
        h3_ids_r6=h3_ids_r6,
        h3_r6_to_idx=h3_r6_to_idx,
        idx_to_h3_r6=idx_to_h3_r6,
        idx_to_centroid_r6=idx_to_centroid_r6,
        r7_to_r6=r7_to_r6,
        r6_to_r7=r6_to_r7,
    )
