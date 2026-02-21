from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import pandas as pd


WORLD_COVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
OUT_DIR = Path("data/external/worldcover_tiles_2021")


def tile_name(lat: float, lon: float) -> str:
    """
    WorldCover uses 3x3 degree tiles.
    Tile naming format:
      ESA_WorldCover_10m_2021_v200_N51E003_Map.tif
    """

    lat_floor = math.floor(lat / 3) * 3
    lon_floor = math.floor(lon / 3) * 3

    lat_prefix = "N" if lat_floor >= 0 else "S"
    lon_prefix = "E" if lon_floor >= 0 else "W"

    return (
        f"ESA_WorldCover_10m_2021_v200_"
        f"{lat_prefix}{abs(lat_floor):02d}"
        f"{lon_prefix}{abs(lon_floor):03d}"
        f"_Map.tif"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default="data/index/images.parquet",
        help="Path to parquet index with lat/lon columns",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading parquet...")
    df = pd.read_parquet(args.parquet, columns=["lat", "lon"])

    # France métropole bounding box (approx)
    df = df[
        (df["lat"] >= 41.0)
        & (df["lat"] <= 51.5)
        & (df["lon"] >= -5.5)
        & (df["lon"] <= 10.0)
    ].copy()

    if df.empty:
        print("No images inside France bounding box.")
        return

    print(f"{len(df)} images inside France bbox")

    tiles = set()
    for lat, lon in zip(df["lat"], df["lon"]):
        tiles.add(tile_name(float(lat), float(lon)))

    print(f"{len(tiles)} unique WorldCover tiles needed\n")

    for t in sorted(tiles):
        url = f"{WORLD_COVER_BASE}/{t}"
        out_path = OUT_DIR / t

        if out_path.exists():
            print(f"✓ Already exists: {t}")
            continue

        print(f"Downloading: {t}")
        subprocess.run(
            ["curl", "-L", url, "-o", str(out_path)],
            check=False,
        )

    print("\nDone.")
    print(f"Tiles stored in: {OUT_DIR}")


if __name__ == "__main__":
    main()
