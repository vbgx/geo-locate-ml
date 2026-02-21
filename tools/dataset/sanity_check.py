import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/index/images.parquet")
    args = ap.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run tools/build_index.py first.")

    df = pd.read_parquet(path)

    required = ["id", "lat", "lon", "path", "h3_id", "label_idx", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in parquet: {missing}")

    # file existence
    missing_files = df[~df["path"].map(lambda s: Path(s).exists())]
    if len(missing_files) > 0:
        print(f"WARN: {len(missing_files)} rows point to missing files.")

    # split distribution
    print("\nSplit distribution:")
    print(df["split"].value_counts())

    # classes
    print("\nCells/classes:")
    vc = df["h3_id"].value_counts()
    print(f"- num cells: {vc.shape[0]}")
    print(f"- samples min/median/max: {int(vc.min())}/{float(vc.median()):.1f}/{int(vc.max())}")

    # leakage check (sequence_id should not be in multiple splits)
    if "sequence_id" in df.columns:
        s = df.dropna(subset=["sequence_id"])
        if len(s) > 0:
            leak = (
                s.groupby("sequence_id")["split"]
                .nunique()
                .reset_index()
            )
            leaked = leak[leak["split"] > 1]
            print("\nLeakage check (sequence_id across multiple splits):")
            if len(leaked) == 0:
                print("✅ OK: no sequence appears in multiple splits.")
            else:
                print(f"❌ FAIL: {len(leaked)} sequences appear in multiple splits.")
        else:
            print("\nLeakage check: sequence_id missing/empty -> cannot verify.")
    else:
        print("\nLeakage check: sequence_id column missing -> cannot verify.")

    print("\nSanity check done.")


if __name__ == "__main__":
    main()
