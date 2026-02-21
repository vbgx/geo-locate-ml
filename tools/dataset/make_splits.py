import argparse
import hashlib
from pathlib import Path

import pandas as pd


def _stable_hash_to_unit_interval(s: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{s}".encode("utf-8")).hexdigest()
    v = int(h[:12], 16)
    return (v % 10_000_000) / 10_000_000.0


def assign_split_by_sequence(df: pd.DataFrame, seed: int, p_train: float, p_val: float, p_test: float) -> pd.DataFrame:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", default="data/index/images.parquet")
    ap.add_argument("--out-parquet", default="data/index/splits.parquet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.05)
    args = ap.parse_args()

    in_path = Path(args.in_parquet)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run tools/dataset/build_index.py first.")

    df = pd.read_parquet(in_path)

    if "line_idx" not in df.columns:
        raise RuntimeError("images.parquet missing line_idx. Rebuild index with parity-strict build_index.py.")

    # assign split deterministically by sequence_id (or id)
    df = assign_split_by_sequence(df, seed=args.seed, p_train=args.train, p_val=args.val, p_test=args.test)

    # keep 1 row per parquet row: merge later on line_idx (not id!)
    out = df[["line_idx", "id", "sequence_id", "split"]].copy()

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print("Wrote:", out_path)
    print(out["split"].value_counts())


if __name__ == "__main__":
    main()
