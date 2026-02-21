from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from typing import Iterable, Set, Tuple

import pandas as pd

# Copernicus DEM GLO-90 on AWS Open Data (public bucket, no auth needed)
# Folder pattern documented in the bucket readme:
#   Copernicus_DSM_COG_[resolution]_[northing]_[easting]_DEM/
# For GLO-90, resolution = 30 (arc-seconds), bucket = copernicus-dem-90m
BASE = "https://copernicus-dem-90m.s3.amazonaws.com"

OUT_DIR = Path("data/external/dem_tiles_90m")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# France métropole bbox (approx, same as earlier)
FR_BBOX = (-5.5, 41.0, 10.0, 51.5)  # (min_lon, min_lat, max_lon, max_lat)


def _fmt_ns(lat_deg: int) -> str:
    return f"{'N' if lat_deg >= 0 else 'S'}{abs(lat_deg):02d}_00"


def _fmt_ew(lon_deg: int) -> str:
    return f"{'E' if lon_deg >= 0 else 'W'}{abs(lon_deg):03d}_00"


def tile_folder_for_latlon(lat: float, lon: float) -> str:
    """
    Copernicus DEM tiles are 1x1 degree in an original grid.
    We map a (lat, lon) to the SW 1-degree tile by flooring.
    Folder example (GLO-90 => 30 arcsec):
      Copernicus_DSM_COG_30_N48_00_E002_00_DEM
    """
    lat_deg = math.floor(float(lat))
    lon_deg = math.floor(float(lon))
    return f"Copernicus_DSM_COG_30_{_fmt_ns(lat_deg)}_{_fmt_ew(lon_deg)}_DEM"


def tiles_from_points(points: Iterable[Tuple[float, float]]) -> Set[str]:
    out: Set[str] = set()
    for lat, lon in points:
        out.add(tile_folder_for_latlon(lat, lon))
    return out


def curl_download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # -f fail on HTTP errors, -L follow redirects, --retry for transient
    cmd = [
        "curl",
        "-fL",
        "--retry",
        "5",
        "--retry-delay",
        "1",
        "-o",
        str(out_path),
        url,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Copernicus DEM (GLO-90) tiles needed for France métropole dataset.")
    ap.add_argument("--parquet", default="data/index/images.parquet", help="Parquet with lat/lon columns (and ideally your dataset).")
    ap.add_argument("--out_dir", default=str(OUT_DIR), help="Where to store DEM tiles.")
    ap.add_argument("--no_bbox_filter", action="store_true", help="Do not filter points to France bbox (use all points).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    if "lat" not in df.columns or "lon" not in df.columns:
        raise SystemExit("Parquet must contain columns: lat, lon")

    if not args.no_bbox_filter:
        min_lon, min_lat, max_lon, max_lat = FR_BBOX
        df = df[
            (df["lat"] >= min_lat)
            & (df["lat"] <= max_lat)
            & (df["lon"] >= min_lon)
            & (df["lon"] <= max_lon)
        ].copy()

    if df.empty:
        raise SystemExit("No points found after filtering. Check bbox or parquet content.")

    points = list(zip(df["lat"].astype(float).tolist(), df["lon"].astype(float).tolist()))
    tiles = sorted(tiles_from_points(points))

    print(f"Points considered: {len(points)}")
    print(f"Unique DEM tiles needed: {len(tiles)}")
    print(f"Output dir: {out_dir}\n")

    ok = 0
    skipped = 0
    failed = 0

    for folder in tiles:
        tif_name = f"{folder}.tif"
        url = f"{BASE}/{folder}/{tif_name}"
        out_path = out_dir / tif_name

        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"✓ exists  {tif_name}")
            skipped += 1
            continue

        try:
            print(f"↓ fetch   {tif_name}")
            curl_download(url, out_path)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"✗ FAILED {tif_name}  ({e})")

    print("\nSummary:")
    print(f"  downloaded: {ok}")
    print(f"  skipped:    {skipped}")
    print(f"  failed:     {failed}")

    print("\nNext (when GDAL is installed):")
    print("  gdalbuildvrt data/external/dem.vrt data/external/dem_tiles_90m/*.tif")
    print("  gdal_translate -co COMPRESS=DEFLATE -co TILED=YES data/external/dem.vrt data/external/dem.tif")


if __name__ == "__main__":
    main()
