from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import h3


# ============================================================================
# Paths (ex paths.py)
# ============================================================================

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def p(*parts: str) -> Path:
    return repo_root().joinpath(*parts)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def update_latest_symlink(runs_dir: Path, run_dir: Path) -> None:
    latest = runs_dir / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir, target_is_directory=True)


# ============================================================================
# Splits (ex splits.py + tools/dataset/make_splits.py)
# ============================================================================

def _stable_hash_to_unit_interval(s: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{s}".encode("utf-8")).hexdigest()
    v = int(h[:12], 16)
    return (v % 10_000_000) / 10_000_000.0


def assign_split_by_sequence(
    df: pd.DataFrame,
    seed: int,
    p_train: float,
    p_val: float,
    p_test: float,
) -> pd.DataFrame:
    """
    Deterministic split assignment based on sequence_id (or id fallback).
    Produces a 'split' column with values: train/val/test.
    """
    assert abs((p_train + p_val + p_test) - 1.0) < 1e-6, "splits must sum to 1.0"
    df = df.copy()

    if "sequence_id" not in df.columns:
        df["sequence_id"] = None

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


def make_splits_parquet(
    *,
    in_parquet: Path,
    out_parquet: Path,
    seed: int = 42,
    train: float = 0.80,
    val: float = 0.15,
    test: float = 0.05,
) -> Path:
    """
    Equivalent to tools/dataset/make_splits.py but as a library function.
    Reads images.parquet (must contain line_idx) and writes splits.parquet.
    """
    in_parquet = Path(in_parquet)
    out_parquet = Path(out_parquet)

    if not in_parquet.exists():
        raise FileNotFoundError(f"Missing {in_parquet}. Build index first.")

    df = pd.read_parquet(in_parquet)

    if "line_idx" not in df.columns:
        raise RuntimeError(
            "images.parquet missing line_idx. Rebuild index with parity-strict build_index.py."
        )

    df = assign_split_by_sequence(df, seed=seed, p_train=train, p_val=val, p_test=test)

    out = df[["line_idx", "id", "sequence_id", "split"]].copy()

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)

    return out_parquet


# ============================================================================
# Labels / H3 (ex labels.py)
# ============================================================================

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
            "idx_to_centroid": {
                str(k): [lat, lon] for k, (lat, lon) in self.idx_to_centroid.items()
            },
            "parent_res": int(self.parent_res),
            "h3_ids_r6": self.h3_ids_r6,
            "h3_r6_to_idx": self.h3_r6_to_idx,
            "idx_to_h3_r6": {str(k): v for k, v in self.idx_to_h3_r6.items()},
            "idx_to_centroid_r6": {
                str(k): [lat, lon] for k, (lat, lon) in self.idx_to_centroid_r6.items()
            },
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
        idx_to_centroid = {
            int(k): (float(v[0]), float(v[1])) for k, v in obj["idx_to_centroid"].items()
        }
        parent_res = int(obj.get("parent_res", 6))

        if "h3_ids_r6" in obj and "h3_r6_to_idx" in obj and "r7_to_r6" in obj:
            h3_ids_r6 = list(obj["h3_ids_r6"])
            h3_r6_to_idx = {k: int(v) for k, v in obj["h3_r6_to_idx"].items()}
            idx_to_h3_r6 = {int(k): v for k, v in obj["idx_to_h3_r6"].items()}
            idx_to_centroid_r6 = {
                int(k): (float(v[0]), float(v[1])) for k, v in obj["idx_to_centroid_r6"].items()
            }
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
            p_ = h3.cell_to_parent(h, int(parent_res))
        except Exception:
            p_ = h
        parents.append(str(p_))

    h3_ids_r6 = sorted(set(parents))
    h3_r6_to_idx = {h_: i for i, h_ in enumerate(h3_ids_r6)}
    idx_to_h3_r6 = {i: h_ for h_, i in h3_r6_to_idx.items()}

    idx_to_centroid_r6: Dict[int, Tuple[float, float]] = {}
    for i, h_ in idx_to_h3_r6.items():
        lat, lon = h3.cell_to_latlng(h_)
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
    h3_to_idx = {h_: i for i, h_ in enumerate(h3_ids)}
    idx_to_h3 = {i: h_ for h_, i in h3_to_idx.items()}

    idx_to_centroid: Dict[int, Tuple[float, float]] = {}
    for i, h_ in idx_to_h3.items():
        lat, lon = h3.cell_to_latlng(h_)
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


# ============================================================================
# Dataset (ex dataset.py)
# ============================================================================

class GeoDataset(Dataset):
    """
    Processed-only dataset.

    Returns:
      - non-hierarchical:
          (x, y, lat, lon, image_id)
      - hierarchical:
          (x, y_r6, y_r7, lat, lon, image_id)

    Absolutely NO raw handling.
    Absolutely NO proxy / h3_features.
    """

    def __init__(
        self,
        parquet_path: str,
        split: str,
        image_size: int,
        *,
        hierarchical_enabled: bool = False,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.split = str(split)
        self.image_size = int(image_size)
        self.hierarchical_enabled = bool(hierarchical_enabled)

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {self.parquet_path}")

        df = pd.read_parquet(self.parquet_path)

        required = {"path", "label_idx", "lat", "lon", "split", "id"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"Parquet missing columns: {sorted(missing)}")

        if self.hierarchical_enabled:
            hier_required = {"label_r6_idx"}
            missing_h = hier_required - set(df.columns)
            if missing_h:
                raise RuntimeError(
                    f"hierarchical_enabled=True but missing columns: {sorted(missing_h)}"
                )

        df = df[df["split"] == self.split].reset_index(drop=True)
        if df.empty:
            raise RuntimeError(
                f"No samples for split='{self.split}' in {self.parquet_path}"
            )

        # processed-only path validation
        paths = df["path"].astype(str)
        missing_files = [p_ for p_ in paths if not Path(p_).exists()]
        if missing_files:
            raise RuntimeError(
                f"{len(missing_files)} processed images are missing on disk. "
                f"Example: {missing_files[0]}"
            )

        self.df = df

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]

        img_path = Path(row["path"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {img_path}") from e

        x = self.transform(img)

        lat = float(row["lat"])
        lon = float(row["lon"])
        image_id = str(row["id"])

        if self.hierarchical_enabled:
            y_r6 = int(row["label_r6_idx"])
            y_r7 = int(row["label_idx"])
            return (
                x,
                torch.tensor(y_r6, dtype=torch.long),
                torch.tensor(y_r7, dtype=torch.long),
                torch.tensor(lat, dtype=torch.float32),
                torch.tensor(lon, dtype=torch.float32),
                image_id,
            )

        y = int(row["label_idx"])
        return (
            x,
            torch.tensor(y, dtype=torch.long),
            torch.tensor(lat, dtype=torch.float32),
            torch.tensor(lon, dtype=torch.float32),
            image_id,
        )

    def all_ids(self) -> List[str]:
        return [str(v) for v in self.df["id"].tolist()]


# ============================================================================
# Merge splits + kept-only + labels (ex merge_splits.py)
# ============================================================================

def _require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")


def merge_splits_and_build_training_parquet(
    *,
    parquet: Path,
    splits: Path,
    out_full: Path,
    out_kept: Path,
    labels_out: Path,
    min_cell_samples: int = 30,
    drop_missing_files: bool = False,
    parent_res: int = 6,
) -> Dict[str, Path]:
    """
    Merge split column, then create a kept-only training parquet where each (split, h3_id)
    has at least min_cell_samples. Builds labels.json from kept-only classes and adds label_idx.
    Also adds label_r6_idx for hierarchical training.
    """
    parquet = Path(parquet)
    splits = Path(splits)
    out_full = Path(out_full)
    out_kept = Path(out_kept)
    labels_out = Path(labels_out)

    if not parquet.exists():
        raise RuntimeError(f"Missing {parquet}")
    if not splits.exists():
        raise RuntimeError(f"Missing {splits}")

    df = pd.read_parquet(parquet)
    sp = pd.read_parquet(splits)

    _require_cols(df, ["line_idx", "id", "lat", "lon", "path", "h3_id"], "images parquet")
    _require_cols(sp, ["line_idx", "split"], "splits parquet")

    # Ensure 1:1 mapping line_idx -> split
    before = len(sp)
    sp = sp.drop_duplicates(subset=["line_idx"], keep="first").reset_index(drop=True)
    dropped = before - len(sp)
    if dropped > 0:
        print(f"WARN: splits.parquet had {dropped} duplicate line_idx rows. Kept first occurrence.")

    if "split" in df.columns:
        df = df.drop(columns=["split"])

    df = df.merge(
        sp[["line_idx", "split"]],
        on="line_idx",
        how="left",
        validate="one_to_one",
    )

    if df["split"].isna().any():
        n = int(df["split"].isna().sum())
        ex = df.loc[df["split"].isna(), ["line_idx", "id"]].head(10).to_dict("records")
        raise RuntimeError(f"{n} rows missing split after merge. Examples: {ex}")

    # full write
    out_full.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_full, index=False)
    print(f"Wrote full dataset: {out_full}")

    # kept-only per (split, h3_id)
    counts = df.groupby(["split", "h3_id"]).size().reset_index(name="cell_count_split")
    df2 = df.merge(counts, on=["split", "h3_id"], how="left")
    df2["cell_count_split"] = df2["cell_count_split"].fillna(0).astype(int)
    df2["is_kept"] = df2["cell_count_split"] >= int(min_cell_samples)

    df_kept = df2[df2["is_kept"]].copy()

    if drop_missing_files:
        missing_mask = ~df_kept["path"].map(lambda s: Path(str(s)).exists())
        if missing_mask.any():
            print(f"Dropping {int(missing_mask.sum())} kept rows with missing files")
            df_kept = df_kept.loc[~missing_mask].copy()

    if df_kept.empty:
        raise RuntimeError(
            "Kept-only dataset is empty. Lower min_cell_samples or download more data."
        )

    # labels from kept-only
    labels = sorted(df_kept["h3_id"].dropna().astype(str).unique().tolist())
    if len(labels) <= 1:
        raise RuntimeError(
            f"Num kept classes looks wrong: {len(labels)}. "
            "Check h3_id generation and min-cell-samples per split."
        )

    label_to_idx = {h_: i for i, h_ in enumerate(labels)}
    df_kept["label_idx"] = df_kept["h3_id"].astype(str).map(label_to_idx).astype(int)

    # hierarchical r6 labels
    # Build r7->r6 mapping in the SAME order as labels list (r7 index = label_idx)
    # We compute r6 index from parent cell list.
    parents = []
    for h_ in labels:
        try:
            p_ = h3.cell_to_parent(h_, int(parent_res))
        except Exception:
            p_ = h_
        parents.append(str(p_))
    h3_ids_r6 = sorted(set(parents))
    h3_r6_to_idx = {h_: i for i, h_ in enumerate(h3_ids_r6)}
    r7_to_r6 = {i: int(h3_r6_to_idx[parents[i]]) for i in range(len(parents))}

    df_kept["label_r6_idx"] = df_kept["label_idx"].map(r7_to_r6).astype(int)

    # write kept parquet
    out_kept.parent.mkdir(parents=True, exist_ok=True)
    df_kept.to_parquet(out_kept, index=False)
    print(f"Wrote kept-only dataset: {out_kept}")

    # write labels.json (compatible with your existing format)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.write_text(
        json.dumps(
            {
                "num_classes": len(labels),
                "labels": labels,
                "label_to_idx": label_to_idx,
                "parent_res": int(parent_res),
                "h3_ids_r6": h3_ids_r6,
                "h3_r6_to_idx": h3_r6_to_idx,
                "r7_to_r6": r7_to_r6,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote labels: {labels_out}")

    print("\nFull split distribution:")
    print(df["split"].value_counts())

    print("\nKept split distribution:")
    print(df_kept["split"].value_counts())

    print("\nNum kept classes:", len(labels))
    print("Samples per cell (kept):")
    print(df_kept.groupby("h3_id").size().describe())

    return {
        "out_full": out_full,
        "out_kept": out_kept,
        "labels_out": labels_out,
    }


# ============================================================================
# Split coverage check (ex check_split_coverage.py)
# ============================================================================

def check_split_coverage_jsonl(
    *,
    images_jsonl: Path = Path("data/index/images.jsonl"),
    splits_jsonl: Path = Path("data/index/splits.jsonl"),
) -> Dict[str, int]:
    images_jsonl = Path(images_jsonl)
    splits_jsonl = Path(splits_jsonl)

    if not images_jsonl.exists():
        raise FileNotFoundError(f"Missing {images_jsonl}")
    if not splits_jsonl.exists():
        raise FileNotFoundError(f"Missing {splits_jsonl}")

    split_map: Dict[str, str] = {}
    with splits_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            _id = str(o.get("id") or "")
            sp = str(o.get("split") or "")
            if _id and sp:
                split_map[_id] = sp

    counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    total = 0

    with images_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            _id = str(o.get("id") or "")
            sp = split_map.get(_id, "unknown")
            counts[sp] = counts.get(sp, 0) + 1
            total += 1

    known = total - counts.get("unknown", 0)
    print("Total images.jsonl:", total)
    print("Split rows:", len(split_map))
    print("Known (in splits):", known)
    print("Counts by split:", dict(sorted(counts.items(), key=lambda kv: kv[0])))
    print("Unknown:", counts.get("unknown", 0))
    return counts


# ============================================================================
# Export splits from parquet (ex export_splits_from_parquet.py)
# ============================================================================

def export_splits_jsonl_from_parquet(
    *,
    parquet: Path,
    out: Path = Path("data/index/splits.jsonl"),
    id_col: str = "id",
    split_col: str = "split",
) -> Path:
    parquet = Path(parquet)
    out = Path(out)

    if not parquet.exists():
        raise FileNotFoundError(f"Missing parquet: {parquet}")

    df = pd.read_parquet(parquet, columns=[id_col, split_col])

    if id_col not in df.columns:
        raise RuntimeError(f"Missing id column '{id_col}' in parquet. Available: {list(df.columns)}")
    if split_col not in df.columns:
        raise RuntimeError(f"Missing split column '{split_col}' in parquet. Available: {list(df.columns)}")

    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    counts: Dict[str, int] = {}
    with out.open("w", encoding="utf-8") as f:
        for _id, sp in zip(df[id_col].astype(str), df[split_col].astype(str)):
            sp = sp.strip() or "unknown"
            f.write(json.dumps({"id": _id, "split": sp}, ensure_ascii=False) + "\n")
            n += 1
            counts[sp] = counts.get(sp, 0) + 1

    print(f"Wrote {n} rows -> {out}")
    print("Counts:", counts)
    return out


# ============================================================================
# Optional: tiny CLI wrapper (so you can do: python -m src.data ...)
# ============================================================================

def _cmd_make_splits(args: argparse.Namespace) -> None:
    out = make_splits_parquet(
        in_parquet=Path(args.in_parquet),
        out_parquet=Path(args.out_parquet),
        seed=int(args.seed),
        train=float(args.train),
        val=float(args.val),
        test=float(args.test),
    )
    print("Wrote:", out)
    df = pd.read_parquet(out)
    print(df["split"].value_counts())


def _cmd_merge_splits(args: argparse.Namespace) -> None:
    merge_splits_and_build_training_parquet(
        parquet=Path(args.parquet),
        splits=Path(args.splits),
        out_full=Path(args.out),
        out_kept=Path(args.out_kept),
        labels_out=Path(args.labels_out),
        min_cell_samples=int(args.min_cell_samples),
        drop_missing_files=bool(args.drop_missing_files),
        parent_res=int(args.parent_res),
    )


def _cmd_check_coverage(args: argparse.Namespace) -> None:
    check_split_coverage_jsonl(
        images_jsonl=Path(args.images_jsonl),
        splits_jsonl=Path(args.splits_jsonl),
    )


def _cmd_export_splits(args: argparse.Namespace) -> None:
    export_splits_jsonl_from_parquet(
        parquet=Path(args.parquet),
        out=Path(args.out),
        id_col=str(args.id_col),
        split_col=str(args.split_col),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Data utilities (splits, labels, dataset io).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("make-splits", help="Create splits.parquet deterministically.")
    sp.add_argument("--in-parquet", dest="in_parquet", default="data/index/images.parquet")
    sp.add_argument("--out-parquet", dest="out_parquet", default="data/index/splits.parquet")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--train", type=float, default=0.80)
    sp.add_argument("--val", type=float, default=0.15)
    sp.add_argument("--test", type=float, default=0.05)
    sp.set_defaults(func=_cmd_make_splits)

    mg = sub.add_parser("merge-splits", help="Merge splits + build kept-only training parquet + labels.")
    mg.add_argument("--parquet", required=True)
    mg.add_argument("--splits", required=True)
    mg.add_argument("--out", required=True)
    mg.add_argument("--out-kept", dest="out_kept", required=True)
    mg.add_argument("--labels-out", dest="labels_out", required=True)
    mg.add_argument("--min-cell-samples", type=int, default=30)
    mg.add_argument("--drop-missing-files", action="store_true")
    mg.add_argument("--parent-res", type=int, default=6)
    mg.set_defaults(func=_cmd_merge_splits)

    cc = sub.add_parser("check-coverage", help="Check images.jsonl vs splits.jsonl coverage.")
    cc.add_argument("--images-jsonl", default="data/index/images.jsonl")
    cc.add_argument("--splits-jsonl", default="data/index/splits.jsonl")
    cc.set_defaults(func=_cmd_check_coverage)

    ex = sub.add_parser("export-splits", help="Export splits.jsonl from dataset parquet.")
    ex.add_argument("--parquet", required=True)
    ex.add_argument("--out", default="data/index/splits.jsonl")
    ex.add_argument("--id-col", default="id")
    ex.add_argument("--split-col", default="split")
    ex.set_defaults(func=_cmd_export_splits)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


def build_transform(image_size: int):
    """
    Torchvision transform used both for training and inference.
    Kept here so predict.py doesn't depend on GeoDataset internals.
    """
    from torchvision import transforms

    image_size = int(image_size)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
