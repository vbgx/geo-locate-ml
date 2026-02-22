from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def remake_splits_stratified_inplace(
    df: pd.DataFrame,
    *,
    seed: int,
    val_frac: float,
    test_frac: float,
    min_total_for_holdout: int,
) -> pd.DataFrame:
    """
    Recrée df["split"] de manière stratifiée par h3_id, en conservant toutes les autres colonnes.

    Règles:
    - si n < min_total_for_holdout: train-only
    - sinon: ~test_frac en test, ~val_frac en val, reste en train
    - garantit val/test ⊆ train (par construction)
    """
    rng = np.random.default_rng(seed)

    out = df.copy()
    out["split"] = "train"

    groups = out.groupby("h3_id").groups
    for _h3, idx in groups.items():
        idx = np.fromiter(idx, dtype=np.int64)
        n = int(idx.size)

        if n < min_total_for_holdout:
            continue

        rng.shuffle(idx)

        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))

        # garde au moins 1 en train
        if n_test + n_val >= n:
            overflow = (n_test + n_val) - (n - 1)
            take = min(overflow, n_test)
            n_test -= take
            overflow -= take
            if overflow > 0:
                n_val = max(0, n_val - overflow)

        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]

        if n_test > 0:
            out.loc[test_idx, "split"] = "test"
        if n_val > 0:
            out.loc[val_idx, "split"] = "val"

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/index/images_kept.parquet")
    ap.add_argument("--output", default="data/index/images_kept_splits.parquet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--min-total-for-holdout", type=int, default=20)
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)

    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    df = pd.read_parquet(inp)

    required = {"id", "lat", "lon", "path", "h3_id", "label_idx"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {inp}: {sorted(missing)}")

    df2 = remake_splits_stratified_inplace(
        df,
        seed=int(args.seed),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        min_total_for_holdout=int(args.min_total_for_holdout),
    )

    print("\n=== SPLIT DISTRIBUTION (new) ===")
    print(df2["split"].value_counts())

    # Sanity checks
    train_classes = set(df2[df2["split"] == "train"]["h3_id"].unique())
    val_classes = set(df2[df2["split"] == "val"]["h3_id"].unique())
    test_classes = set(df2[df2["split"] == "test"]["h3_id"].unique())

    print("\n=== CLASS COVERAGE CHECK ===")
    print("val ⊆ train:", val_classes.issubset(train_classes))
    print("test ⊆ train:", test_classes.issubset(train_classes))

    per_class = df2.groupby("h3_id").size()
    rare = per_class[per_class < int(args.min_total_for_holdout)]

    print("\n=== CLASSES ===")
    print("Unique classes:", df2["h3_id"].nunique())
    print("\nSamples per class stats:")
    print(per_class.describe())

    print(f"\nRare classes (<{int(args.min_total_for_holdout)} samples): {len(rare)}")

    outp.parent.mkdir(parents=True, exist_ok=True)
    df2.to_parquet(outp, index=False)
    print(f"\nWrote: {outp} rows={len(df2)}")


if __name__ == "__main__":
    main()
