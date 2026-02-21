#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

SRC = Path("data/cities/cities_fr.csv")
OUT_DIR = Path("data/cities")
OUT_ALL = OUT_DIR / "cities_fr_50_bboxes.csv"

BATCH_PREFIX = "cities_batch_"
BATCH_SIZE = 5

DEFAULT_RADIUS_KM = 8.0
DEFAULT_LIMIT = 2000


@dataclass(frozen=True)
class CityPoint:
    name: str
    lat: float
    lon: float
    radius_km: float
    limit: int


@dataclass(frozen=True)
class CityBBox:
    name: str
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    limit: int


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("’", "'")
    # very light "slug": keep ascii letters/numbers/underscore/hyphen -> underscore
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def km_to_deg_lat(km: float) -> float:
    return km / 111.0


def km_to_deg_lon(km: float, lat_deg: float) -> float:
    cosv = math.cos(math.radians(lat_deg))
    cosv = max(1e-6, cosv)
    return km / (111.0 * cosv)


def point_to_bbox(c: CityPoint) -> CityBBox:
    dlat = km_to_deg_lat(c.radius_km)
    dlon = km_to_deg_lon(c.radius_km, c.lat)
    return CityBBox(
        name=c.name,
        lon_min=c.lon - dlon,
        lat_min=c.lat - dlat,
        lon_max=c.lon + dlon,
        lat_max=c.lat + dlat,
        limit=c.limit,
    )


def read_rows(path: Path) -> Tuple[List[str], List[dict]]:
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        headers = list(r.fieldnames or [])
        rows = list(r)
    return headers, rows


def as_float(row: dict, key: str) -> Optional[float]:
    v = row.get(key)
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    return float(s)


def as_int(row: dict, key: str) -> Optional[int]:
    v = row.get(key)
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    return int(float(s))


def detect_and_parse(path: Path) -> List[CityBBox]:
    headers, rows = read_rows(path)
    hs = set(h.strip() for h in headers)

    out: List[CityBBox] = []

    # Schema 1: bbox
    if {"name", "lon_min", "lat_min", "lon_max", "lat_max"}.issubset(hs):
        for row in rows:
            raw = (row.get("name") or "").strip()
            if not raw:
                continue
            name = slugify(raw)
            limit = as_int(row, "limit") or DEFAULT_LIMIT
            out.append(
                CityBBox(
                    name=name,
                    lon_min=float(row["lon_min"]),
                    lat_min=float(row["lat_min"]),
                    lon_max=float(row["lon_max"]),
                    lat_max=float(row["lat_max"]),
                    limit=limit,
                )
            )
        return out

    # Schema 2: point (lat/lon)
    if {"name", "lat", "lon"}.issubset(hs):
        for row in rows:
            raw = (row.get("name") or "").strip()
            if not raw:
                continue
            name = slugify(raw)
            lat = as_float(row, "lat")
            lon = as_float(row, "lon")
            if lat is None or lon is None:
                continue
            radius = as_float(row, "radius_km") or DEFAULT_RADIUS_KM
            limit = as_int(row, "limit") or DEFAULT_LIMIT
            out.append(point_to_bbox(CityPoint(name=name, lat=lat, lon=lon, radius_km=radius, limit=limit)))
        return out

    raise SystemExit(
        f"{path} missing required columns.\n"
        f"Supported schemas:\n"
        f" - name,lat,lon[,radius_km][,limit]\n"
        f" - name,lon_min,lat_min,lon_max,lat_max[,limit]\n"
        f"Found headers: {headers}"
    )


def dedupe_by_name(bboxes: List[CityBBox]) -> List[CityBBox]:
    seen: Dict[str, CityBBox] = {}
    for b in bboxes:
        # keep first occurrence
        if b.name not in seen:
            seen[b.name] = b
    return list(seen.values())


def write_bbox_csv(path: Path, rows: List[CityBBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "lon_min", "lat_min", "lon_max", "lat_max", "limit"])
        for c in rows:
            w.writerow([c.name, c.lon_min, c.lat_min, c.lon_max, c.lat_max, c.limit])


def main() -> None:
    bboxes = dedupe_by_name(detect_and_parse(SRC))

    # Keep first 50 (after normalization/dedupe)
    bboxes = bboxes[:50]
    if len(bboxes) < 50:
        print(f"WARN: only {len(bboxes)} cities found (expected 50). Proceeding anyway.")

    write_bbox_csv(OUT_ALL, bboxes)
    print(f"Wrote: {OUT_ALL} ({len(bboxes)} cities)")

    batches = [bboxes[i : i + BATCH_SIZE] for i in range(0, len(bboxes), BATCH_SIZE)]
    for i, batch in enumerate(batches, start=1):
        out = OUT_DIR / f"{BATCH_PREFIX}{i:02d}.csv"
        write_bbox_csv(out, batch)
        print(f"Wrote: {out} ({len(batch)} cities)")


if __name__ == "__main__":
    main()
