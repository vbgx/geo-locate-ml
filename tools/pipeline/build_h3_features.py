from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


def load_labels_from_best() -> Tuple[List[str], Dict[int, Tuple[float, float]]]:
    best = Path("models/best.json")
    if not best.exists():
        raise SystemExit("models/best.json not found. Train once to create it.")
    meta = json.loads(best.read_text(encoding="utf-8"))
    labels_path = Path(meta["labels_path"])
    if not labels_path.exists():
        raise SystemExit(f"labels_path not found: {labels_path}")

    obj = json.loads(labels_path.read_text(encoding="utf-8"))
    h3_ids = list(obj["h3_ids"])
    idx_to_centroid = {int(k): (float(v[0]), float(v[1])) for k, v in obj["idx_to_centroid"].items()}
    return h3_ids, idx_to_centroid


def sample_raster(tif_path: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    import rasterio

    with rasterio.open(tif_path) as ds:
        # rasterio expects (lon, lat)
        pts = list(zip(lons.tolist(), lats.tolist()))
        out = np.array([v[0] if v is not None else np.nan for v in ds.sample(pts)], dtype=np.float64)
        # convert nodata to nan
        if ds.nodata is not None:
            out = np.where(out == ds.nodata, np.nan, out)
        return out


def load_coastline_geom(coastline_shp: Path):
    import geopandas as gpd

    gdf = gpd.read_file(coastline_shp)
    # Ensure WGS84
    if gdf.crs is None:
        # NaturalEarth usually has WGS84, but be safe
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    geom = gdf.geometry
    # dissolve into one geometry (MultiLineString)
    merged = geom.union_all()
    return merged


def dist_to_coast_km(coast_geom, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Compute approximate distance to coastline using EPSG:3857 projection for metric distance.
    Good enough for feature engineering.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    # Points in WGS84
    pts = [Point(float(lon), float(lat)) for lat, lon in zip(lats, lons)]
    gpts = gpd.GeoSeries(pts, crs="EPSG:4326").to_crs("EPSG:3857")

    coast = gpd.GeoSeries([coast_geom], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

    # distance in meters -> km
    d_m = gpts.distance(coast).to_numpy(dtype=np.float64)
    return d_m / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem_tif", default="data/external/dem.tif")
    ap.add_argument("--worldcover_tif", default="data/external/worldcover.tif")
    ap.add_argument("--ghsl_pop_tif", default="data/external/ghsl_pop.tif")
    ap.add_argument("--coastline_shp", default="data/external/ne_coastline/ne_10m_coastline.shp")
    ap.add_argument("--out", default="data/index/h3_features.parquet")
    args = ap.parse_args()

    dem_tif = Path(args.dem_tif)
    wc_tif = Path(args.worldcover_tif)
    pop_tif = Path(args.ghsl_pop_tif)
    coast_shp = Path(args.coastline_shp)
    out_path = Path(args.out)

    for fp in [dem_tif, wc_tif, pop_tif, coast_shp]:
        if not fp.exists():
            raise SystemExit(f"Missing required file: {fp}")

    h3_ids, idx_to_centroid = load_labels_from_best()

    # Build base table (one row per class_idx)
    C = len(h3_ids)
    class_idx = np.arange(C, dtype=np.int64)
    lat = np.array([idx_to_centroid[i][0] for i in class_idx], dtype=np.float64)
    lon = np.array([idx_to_centroid[i][1] for i in class_idx], dtype=np.float64)

    print(f"Classes: {C}")
    print("Sampling rasters...")

    elev = sample_raster(dem_tif, lat, lon)
    landcover = sample_raster(wc_tif, lat, lon)
    pop = sample_raster(pop_tif, lat, lon)

    print("Loading coastline geometry...")
    coast_geom = load_coastline_geom(coast_shp)
    print("Computing distance to coast...")
    dist_coast = dist_to_coast_km(coast_geom, lat, lon)

    df = pd.DataFrame(
        {
            "class_idx": class_idx,
            "h3_id": h3_ids,
            "lat": lat,
            "lon": lon,
            "elev_m": elev,
            "pop": pop,
            "landcover": landcover,
            "dist_coast_km": dist_coast,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n✅ Wrote: {out_path}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
