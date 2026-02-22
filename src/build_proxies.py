# src/build_proxies.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# ---- Optional heavy deps: imported lazily where needed ----


# ============================================================================
# WorldCover codes (ESA WorldCover v200)
# ============================================================================
WC_BUILT = 50
WC_WATER = 80


# ============================================================================
# Spec
# ============================================================================
@dataclass(frozen=True)
class ProxySpec:
    radius_m: float = 500.0
    coastal_decay_km: float = 20.0


# ============================================================================
# Utils
# ============================================================================
def _safe_float(x, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _deg_from_meters_lat(m: float) -> float:
    # ~111.32km per degree latitude
    return float(m) / 111_320.0


def _deg_from_meters_lon(m: float, lat_deg: float) -> float:
    # lon degree shrinks with cos(lat)
    return float(m) / (111_320.0 * max(1e-6, math.cos(math.radians(float(lat_deg)))))


def _coastal_score(dist_km: np.ndarray, decay_km: float) -> np.ndarray:
    decay_km = max(1e-6, float(decay_km))
    d = np.clip(dist_km.astype(np.float32), 0.0, 1e9)
    return np.exp(-d / decay_km).astype(np.float32)


class RunningStats:
    """Numerically stable running mean/std (Welford) ignoring NaNs."""
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update_many(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        for v in x:
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self.m2 += delta * delta2

    def final(self) -> Tuple[float, float]:
        if self.n <= 1:
            return 0.0, 1.0
        var = self.m2 / (self.n - 1)
        return float(self.mean), float(math.sqrt(var) + 1e-6)


# ============================================================================
# Raster sampling (fast + memory-safe)
#   IMPORTANT: Your rasters are EPSG:4326 according to your smoke output.
#   So we do NOT reproject for sampling. We sample directly in lon/lat.
# ============================================================================
class RasterSampler:
    """
    Safe sampler for EPSG:4326 rasters.
    - sample(lon,lat): single pixel value
    - mean(lon,lat,radius_m): approximate mean via small grid sampling (NOT window read)
    """
    def __init__(self, path: Path, name: str, *, gdal_cache_mb: int = 64) -> None:
        import rasterio  # type: ignore

        if not path.exists():
            raise FileNotFoundError(f"{name} raster not found: {path}")

        self.name = name
        self.path = path
        self._env = rasterio.Env(GDAL_CACHEMAX=int(gdal_cache_mb))
        self._env.__enter__()
        self.ds = rasterio.open(str(path))

        if self.ds.crs is None:
            raise RuntimeError(f"{name}: raster has no CRS")
        if str(self.ds.crs).upper() != "EPSG:4326":
            # You *can* support reproj later, but for now keep it explicit.
            raise RuntimeError(f"{name}: expected EPSG:4326, got {self.ds.crs}")

        self.nodata = self.ds.nodata

    def close(self) -> None:
        try:
            self.ds.close()
        finally:
            try:
                self._env.__exit__(None, None, None)
            except Exception:
                pass

    def _is_nodata(self, v: float) -> bool:
        if not math.isfinite(v):
            return True
        if self.nodata is None:
            return False
        try:
            return float(v) == float(self.nodata)
        except Exception:
            return False

    def sample(self, lon: float, lat: float) -> float:
        # rasterio.sample expects [(x,y)] with x=lon y=lat in EPSG:4326
        try:
            arr = next(self.ds.sample([(float(lon), float(lat))]))
            v = float(arr[0])
            if self._is_nodata(v):
                return float("nan")
            return v
        except Exception:
            return float("nan")

    def mean(self, lon: float, lat: float, radius_m: float, *, grid: int = 5) -> float:
        """
        Approximate mean in a radius box by sampling a small grid.
        This avoids huge window reads (which can OOM/kill on big rasters).
        """
        g = int(grid)
        g = max(3, min(g, 11))
        half_lat = _deg_from_meters_lat(radius_m)
        half_lon = _deg_from_meters_lon(radius_m, lat)

        # grid points in box centered at (lon,lat)
        xs = np.linspace(float(lon) - half_lon, float(lon) + half_lon, g, dtype=np.float64)
        ys = np.linspace(float(lat) - half_lat, float(lat) + half_lat, g, dtype=np.float64)

        pts = [(float(x), float(y)) for y in ys for x in xs]
        try:
            vals = []
            for a in self.ds.sample(pts):
                v = float(a[0])
                if not self._is_nodata(v):
                    vals.append(v)
            if not vals:
                return float("nan")
            return float(np.mean(np.asarray(vals, dtype=np.float64)))
        except Exception:
            return float("nan")


class WorldCoverSampler(RasterSampler):
    def __init__(self, path: Path, *, gdal_cache_mb: int = 64) -> None:
        super().__init__(path, "WorldCover", gdal_cache_mb=gdal_cache_mb)

    def fractions(self, lon: float, lat: float, radius_m: float, *, pixel_only: bool) -> Tuple[float, float]:
        if pixel_only:
            v = self.sample(lon, lat)
            if not math.isfinite(v):
                return float("nan"), float("nan")
            vi = int(v)
            return float(vi == WC_WATER), float(vi == WC_BUILT)

        # approximate fractions by grid sampling
        g = 7
        half_lat = _deg_from_meters_lat(radius_m)
        half_lon = _deg_from_meters_lon(radius_m, lat)
        xs = np.linspace(float(lon) - half_lon, float(lon) + half_lon, g, dtype=np.float64)
        ys = np.linspace(float(lat) - half_lat, float(lat) + half_lat, g, dtype=np.float64)
        pts = [(float(x), float(y)) for y in ys for x in xs]

        try:
            vals = []
            for a in self.ds.sample(pts):
                v = float(a[0])
                if math.isfinite(v) and not self._is_nodata(v):
                    vals.append(int(v))
            if not vals:
                return float("nan"), float("nan")
            arr = np.asarray(vals, dtype=np.int32)
            return float(np.mean(arr == WC_WATER)), float(np.mean(arr == WC_BUILT))
        except Exception:
            return float("nan"), float("nan")


# ============================================================================
# Parquet helpers (streaming, schema-stable)
# ============================================================================
def _read_base_schema_and_names(in_parquet: Path):
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    pf = pq.ParquetFile(in_parquet)

    # This is the file schema (the only one that matters)
    base_schema = pf.schema_arrow

    # IMPORTANT: sometimes a column might be stored as "null" in some batches if entirely null there.
    # We will CAST each batch to base_schema to force stable types.
    base_names = [f.name for f in base_schema]

    return pf, base_schema, base_names


def _make_out_schema(base_schema):
    import pyarrow as pa  # type: ignore

    proxy_fields = [
        pa.field("proxy_elev_m", pa.float32()),
        pa.field("proxy_pop", pa.float32()),
        pa.field("proxy_water_frac", pa.float32()),
        pa.field("proxy_built_frac", pa.float32()),
        pa.field("proxy_coast_dist_km", pa.float32()),
        pa.field("proxy_coastal_score", pa.float32()),
        pa.field("proxy_elev_log1p", pa.float32()),
        pa.field("proxy_pop_log1p", pa.float32()),
        pa.field("proxy_elev_log1p_z", pa.float32()),
        pa.field("proxy_pop_log1p_z", pa.float32()),
    ]
    return pa.schema(list(base_schema) + proxy_fields)


# ============================================================================
# Main pipeline: two-pass streaming
#   Pass1: compute raw proxies + log1p, accumulate train stats, write tmp parquet
#   Pass2: read tmp parquet, compute z using train stats, write final parquet
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Build proxy columns into a parquet (streaming, safe).")

    ap.add_argument("--in-parquet", default="data/index/images_kept.parquet")
    ap.add_argument("--out-parquet", default="data/index/images_kept_with_proxies.parquet")
    ap.add_argument("--stats-out", default="data/index/proxies_stats.json")

    ap.add_argument("--dem", default="data/external/dem.vrt")
    ap.add_argument("--pop", default="data/external/ghsl_pop.tif")
    ap.add_argument("--worldcover", default="data/external/worldcover.vrt")

    ap.add_argument("--radius-m", type=float, default=500.0)
    ap.add_argument("--coastal-decay-km", type=float, default=20.0)

    ap.add_argument("--batch-rows", type=int, default=256)
    ap.add_argument("--limit-rows", type=int, default=0)

    ap.add_argument("--gdal-cache-mb", type=int, default=64)

    ap.add_argument("--pixel-only", action="store_true", help="Fast mode: sample single pixel (no windows/grids).")
    ap.add_argument("--skip-dem", action="store_true")
    ap.add_argument("--skip-pop", action="store_true")
    ap.add_argument("--skip-worldcover", action="store_true")

    ap.add_argument("--parquet-only", action="store_true", help="Only iterate parquet batches (debug).")
    ap.add_argument("--smoke", action="store_true", help="Open rasters and exit (debug).")

    # Coast is intentionally omitted here (you can add later). Keep columns but fill NaN.
    ap.add_argument("--no-coast", action="store_true", help="Ignored (coast not computed yet).", default=True)

    args = ap.parse_args()

    in_parquet = Path(args.in_parquet)
    out_parquet = Path(args.out_parquet)
    stats_out = Path(args.stats_out)

    if not in_parquet.exists():
        raise FileNotFoundError(f"Missing parquet: {in_parquet}")

    # --- Parquet-only / schema discovery ---
    pf, base_schema, base_names = _read_base_schema_and_names(in_parquet)

    if args.parquet_only:
        from tqdm.auto import tqdm  # type: ignore

        seen = 0
        for batch in tqdm(pf.iter_batches(batch_size=int(args.batch_rows)), desc="Parquet only"):
            seen += batch.num_rows
            if args.limit_rows and seen >= int(args.limit_rows):
                break
        print(f"✅ Parquet iteration OK. rows_seen={seen}")
        return

    # --- Open rasters (lazy) ---
    dem_sampler: Optional[RasterSampler] = None
    pop_sampler: Optional[RasterSampler] = None
    wc_sampler: Optional[WorldCoverSampler] = None

    if args.smoke:
        if not args.skip_dem:
            dem_sampler = RasterSampler(Path(args.dem), "DEM", gdal_cache_mb=int(args.gdal_cache_mb))
            print(f"DEM: crs={dem_sampler.ds.crs} size={dem_sampler.ds.width}x{dem_sampler.ds.height} res={dem_sampler.ds.res} dtype={dem_sampler.ds.dtypes}")
            dem_sampler.close()
        else:
            print("DEM: SKIPPED")

        if not args.skip_pop:
            pop_sampler = RasterSampler(Path(args.pop), "POP", gdal_cache_mb=int(args.gdal_cache_mb))
            print(f"POP: crs={pop_sampler.ds.crs} size={pop_sampler.ds.width}x{pop_sampler.ds.height} res={pop_sampler.ds.res} dtype={pop_sampler.ds.dtypes}")
            pop_sampler.close()
        else:
            print("POP: SKIPPED")

        if not args.skip_worldcover:
            wc_sampler = WorldCoverSampler(Path(args.worldcover), gdal_cache_mb=int(args.gdal_cache_mb))
            print(f"WORLDCOVER: crs={wc_sampler.ds.crs} size={wc_sampler.ds.width}x{wc_sampler.ds.height} res={wc_sampler.ds.res} dtype={wc_sampler.ds.dtypes}")
            wc_sampler.close()
        else:
            print("WORLDCOVER: SKIPPED")

        print("✅ Smoke OK (opened rasters).")
        return

    # Real run: open requested samplers
    try:
        if not args.skip_dem:
            dem_sampler = RasterSampler(Path(args.dem), "DEM", gdal_cache_mb=int(args.gdal_cache_mb))
        if not args.skip_pop:
            pop_sampler = RasterSampler(Path(args.pop), "POP", gdal_cache_mb=int(args.gdal_cache_mb))
        if not args.skip_worldcover:
            wc_sampler = WorldCoverSampler(Path(args.worldcover), gdal_cache_mb=int(args.gdal_cache_mb))

        spec = ProxySpec(radius_m=float(args.radius_m), coastal_decay_km=float(args.coastal_decay_km))

        # Two-pass paths
        tmp_parquet = out_parquet.with_suffix(".pass1.parquet")

        # --------------------------------------------------------------------
        # PASS 1
        # --------------------------------------------------------------------
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        from tqdm.auto import tqdm  # type: ignore

        for pth in (tmp_parquet, out_parquet):
            if pth.exists():
                pth.unlink()
        tmp_parquet.parent.mkdir(parents=True, exist_ok=True)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)

        out_schema = _make_out_schema(base_schema)

        elev_stats = RunningStats()
        pop_stats = RunningStats()

        writer: Optional[pq.ParquetWriter] = None
        rows_seen = 0

        for batch in tqdm(pf.iter_batches(batch_size=int(args.batch_rows)), desc="Pass1 (raw proxies)", dynamic_ncols=True):
            t_raw = pa.Table.from_batches([batch])

            # KEY: enforce stable types (fixes your sequence_id null-vs-string crash)
            t = t_raw.cast(base_schema, safe=False)

            lat = t["lat"].to_numpy(zero_copy_only=False).astype(np.float64)
            lon = t["lon"].to_numpy(zero_copy_only=False).astype(np.float64)
            n = int(len(lat))

            elev = np.full((n,), np.nan, dtype=np.float32)
            pop = np.full((n,), np.nan, dtype=np.float32)
            water = np.full((n,), np.nan, dtype=np.float32)
            built = np.full((n,), np.nan, dtype=np.float32)
            coast_km = np.full((n,), np.nan, dtype=np.float32)  # not computed yet

            r = float(spec.radius_m)
            pixel_only = bool(args.pixel_only)

            for i in range(n):
                la = float(lat[i])
                lo = float(lon[i])

                if dem_sampler is not None:
                    elev[i] = _safe_float(dem_sampler.sample(lo, la) if pixel_only else dem_sampler.mean(lo, la, r))
                if pop_sampler is not None:
                    pop[i] = _safe_float(pop_sampler.sample(lo, la) if pixel_only else pop_sampler.mean(lo, la, r))
                if wc_sampler is not None:
                    wv, bv = wc_sampler.fractions(lo, la, r, pixel_only=pixel_only)
                    water[i] = _safe_float(wv)
                    built[i] = _safe_float(bv)

            elev_log = np.log1p(np.clip(elev, 0.0, 1e7)).astype(np.float32)
            pop_log = np.log1p(np.clip(pop, 0.0, 1e12)).astype(np.float32)

            if "split" in base_names:
                split = t["split"].to_numpy(zero_copy_only=False).astype(str)
                tr = split == "train"
            else:
                tr = np.ones((n,), dtype=bool)

            elev_stats.update_many(elev_log[tr])
            pop_stats.update_many(pop_log[tr])

            arrays = [t[name] for name in base_names]
            arrays += [
                pa.array(elev, type=pa.float32()),
                pa.array(pop, type=pa.float32()),
                pa.array(water, type=pa.float32()),
                pa.array(built, type=pa.float32()),
                pa.array(coast_km, type=pa.float32()),
                pa.array(_coastal_score(coast_km, spec.coastal_decay_km), type=pa.float32()),
                pa.array(elev_log, type=pa.float32()),
                pa.array(pop_log, type=pa.float32()),
                pa.array(np.zeros((n,), dtype=np.float32), type=pa.float32()),  # z placeholders
                pa.array(np.zeros((n,), dtype=np.float32), type=pa.float32()),
            ]

            out_table = pa.Table.from_arrays(arrays, schema=out_schema)

            if writer is None:
                writer = pq.ParquetWriter(str(tmp_parquet), out_schema, compression="snappy")
            writer.write_table(out_table)

            rows_seen += n
            if args.limit_rows and rows_seen >= int(args.limit_rows):
                break

        if writer is None:
            raise RuntimeError("Pass1 wrote no rows.")
        writer.close()

        elev_mu, elev_sd = elev_stats.final()
        pop_mu, pop_sd = pop_stats.final()

        # --------------------------------------------------------------------
        # PASS 2
        # --------------------------------------------------------------------
        pf2 = pq.ParquetFile(str(tmp_parquet))

        writer2: Optional[pq.ParquetWriter] = None
        rows_seen2 = 0

        for batch in tqdm(pf2.iter_batches(batch_size=int(args.batch_rows)), desc="Pass2 (zscore)", dynamic_ncols=True):
            t = pa.Table.from_batches([batch]).cast(out_schema, safe=False)

            elev_log = t["proxy_elev_log1p"].to_numpy(zero_copy_only=False).astype(np.float32)
            pop_log = t["proxy_pop_log1p"].to_numpy(zero_copy_only=False).astype(np.float32)

            elev_z = ((elev_log - float(elev_mu)) / float(elev_sd)).astype(np.float32)
            pop_z = ((pop_log - float(pop_mu)) / float(pop_sd)).astype(np.float32)

            # replace z columns (last two proxy cols)
            cols = list(t.columns)
            # out_schema order: ... proxy_elev_log1p_z, proxy_pop_log1p_z at the end
            cols[-2] = pa.array(elev_z, type=pa.float32())
            cols[-1] = pa.array(pop_z, type=pa.float32())

            t2 = pa.Table.from_arrays(cols, schema=out_schema)

            if writer2 is None:
                writer2 = pq.ParquetWriter(str(out_parquet), out_schema, compression="snappy")
            writer2.write_table(t2)

            rows_seen2 += t2.num_rows
            if args.limit_rows and rows_seen2 >= int(args.limit_rows):
                break

        if writer2 is None:
            raise RuntimeError("Pass2 wrote no rows.")
        writer2.close()

        # Cleanup tmp
        try:
            tmp_parquet.unlink()
        except Exception:
            pass

        # Stats JSON
        stats_payload = {
            "spec": {"radius_m": float(spec.radius_m), "coastal_decay_km": float(spec.coastal_decay_km)},
            "train_stats": {
                "proxy_elev_log1p": {"mean": float(elev_mu), "std": float(elev_sd)},
                "proxy_pop_log1p": {"mean": float(pop_mu), "std": float(pop_sd)},
            },
            "notes": {
                "coast": "not computed yet (proxy_coast_dist_km is NaN).",
                "window": "mean/fractions use grid sampling unless --pixel-only is set.",
            },
        }
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats_out.write_text(json.dumps(stats_payload, indent=2) + "\n", encoding="utf-8")

        print("✅ Wrote:", out_parquet)
        print("✅ Wrote stats:", stats_out)
        print(f"rows: {rows_seen2}")

    finally:
        # Close rasters
        try:
            if dem_sampler is not None:
                dem_sampler.close()
        except Exception:
            pass
        try:
            if pop_sampler is not None:
                pop_sampler.close()
        except Exception:
            pass
        try:
            if wc_sampler is not None:
                wc_sampler.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()