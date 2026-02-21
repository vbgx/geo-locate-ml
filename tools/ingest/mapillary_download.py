#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

GRAPH = "https://graph.mapillary.com"

DEFAULT_FIELDS = [
    "id",
    "captured_at",
    "sequence",
    "computed_geometry",
    "geometry",
    "thumb_1024_url",
]

MAX_TILE_AREA = 0.010  # Mapillary constraint (square degrees)

CANON_KEYS = [
    "id",
    "lat",
    "lon",
    "captured_at",
    "sequence_id",
    "thumb_1024_url",
    "bbox",
    "city",
    "path",
]


def canonical_obj(obj: Dict) -> "OrderedDict[str, object]":
    out: "OrderedDict[str, object]" = OrderedDict()
    for k in CANON_KEYS:
        out[k] = obj.get(k, None)
    return out


@dataclass(frozen=True)
class ImageItem:
    id: str
    lat: float
    lon: float
    captured_at: Optional[str]
    sequence_id: Optional[str]
    thumb_url: str
    bbox: str
    city: Optional[str]


# -----------------------
# bbox helpers
# -----------------------
def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError('bbox must be "lon_min,lat_min,lon_max,lat_max"')
    lon_min, lat_min, lon_max, lat_max = map(float, parts)
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError("invalid bbox ordering")
    return lon_min, lat_min, lon_max, lat_max


def bbox_area(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> float:
    return abs((lon_max - lon_min) * (lat_max - lat_min))


def bbox_dims(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    lon_min, lat_min, lon_max, lat_max = b
    return abs(lon_max - lon_min), abs(lat_max - lat_min)


def bbox_center(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    lon_min, lat_min, lon_max, lat_max = b
    return (lon_min + lon_max) / 2.0, (lat_min + lat_max) / 2.0


def expand_bbox_to_dims(
    b: Tuple[float, float, float, float],
    target_w: float,
    target_h: float,
) -> Tuple[float, float, float, float]:
    cx, cy = bbox_center(b)
    half_w = target_w / 2.0
    half_h = target_h / 2.0
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def next_expanded_bbox(
    b: Tuple[float, float, float, float],
    factor: float,
    max_deg: float,
) -> Tuple[float, float, float, float]:
    w, h = bbox_dims(b)
    new_w = min(max_deg, w * factor)
    new_h = min(max_deg, h * factor)
    return expand_bbox_to_dims(b, new_w, new_h)


def split_bbox_into_tiles(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    max_tile_area: float,
) -> List[Tuple[float, float, float, float]]:
    area = bbox_area(lon_min, lat_min, lon_max, lat_max)
    if area <= max_tile_area:
        return [(lon_min, lat_min, lon_max, lat_max)]

    tiles_needed = int(area / max_tile_area) + 1
    nx = max(1, int(tiles_needed**0.5))
    ny = nx
    while True:
        tile_w = (lon_max - lon_min) / nx
        tile_h = (lat_max - lat_min) / ny
        if abs(tile_w * tile_h) <= max_tile_area:
            break
        nx += 1
        ny = nx

    tile_w = (lon_max - lon_min) / nx
    tile_h = (lat_max - lat_min) / ny

    tiles: List[Tuple[float, float, float, float]] = []
    for ix in range(nx):
        for iy in range(ny):
            a = lon_min + ix * tile_w
            b = lat_min + iy * tile_h
            c = lon_min + (ix + 1) * tile_w
            d = lat_min + (iy + 1) * tile_h
            tiles.append((a, b, c, d))
    return tiles


# -----------------------
# auth / http
# -----------------------
def get_token(cli_token: Optional[str]) -> str:
    load_dotenv()
    token = cli_token or os.environ.get("MAPILLARY_TOKEN") or os.environ.get("MLY_TOKEN")
    if not token:
        raise RuntimeError("Missing token. Set MAPILLARY_TOKEN/MLY_TOKEN or pass --token.")
    return token


def requests_get_json(url: str, token: str, timeout_s: float, max_retries: int = 6) -> Dict:
    headers = {"Authorization": f"OAuth {token}"}
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout_s)
            if r.status_code == 200:
                return r.json()

            if r.status_code in (429, 500, 502, 503, 504):
                msg = r.text[:300].replace("\n", " ")
                print(f"WARN: HTTP {r.status_code}. Retry {attempt}/{max_retries} :: {msg}", flush=True)
                time.sleep(min(60.0, (2.0 ** (attempt - 1)) + random.random()))
                continue

            raise RuntimeError(f"HTTP {r.status_code} for {url}\n{r.text[:800]}")
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            last_err = e
            print(f"WARN: timeout. Retry {attempt}/{max_retries} :: {e}", flush=True)
            time.sleep(min(60.0, (2.0 ** (attempt - 1)) + random.random()))
        except requests.exceptions.RequestException as e:
            last_err = e
            print(f"WARN: request error. Retry {attempt}/{max_retries} :: {e}", flush=True)
            time.sleep(min(60.0, (2.0 ** (attempt - 1)) + random.random()))

    raise RuntimeError(f"Failed after {max_retries} attempts: {url}") from last_err


def safe_extract_latlon(item: Dict) -> Tuple[float, float]:
    geom = item.get("computed_geometry") or item.get("geometry")
    if not geom or "coordinates" not in geom:
        raise KeyError("missing geometry.coordinates")
    lon, lat = geom["coordinates"]
    return float(lat), float(lon)


# -----------------------
# index + download
# -----------------------
def load_existing_ids(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    ids: set[str] = set()
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = obj.get("id")
            if mid is not None:
                ids.add(str(mid))
    return ids


def append_jsonl(index_path: Path, obj: Dict) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def download_file(url: str, out_path: Path, token: str, timeout_s: float) -> None:
    headers = {"Authorization": f"OAuth {token}"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with requests.get(url, headers=headers, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    tmp.replace(out_path)


# -----------------------
# search (tile paging)
# -----------------------
def iter_images_in_tile(
    token: str,
    tile: Tuple[float, float, float, float],
    page_limit: int,
    sleep_s: float,
    timeout_s: float,
    fields: List[str],
    city: Optional[str],
    scan_budget_ref: List[int],
) -> Iterator[ImageItem]:
    """
    Yields items from Mapillary in this tile.
    Decrements scan_budget_ref[0] for each item returned by the API (seen).
    Stops when budget is exhausted.
    """
    lon_min, lat_min, lon_max, lat_max = tile
    bbox_str = f"{lon_min:.5f},{lat_min:.5f},{lon_max:.5f},{lat_max:.5f}"
    fields_str = ",".join(fields)
    after: Optional[str] = None

    while scan_budget_ref[0] > 0:
        this_limit = min(page_limit, scan_budget_ref[0])
        url = (
            f"{GRAPH}/images"
            f"?bbox={lon_min},{lat_min},{lon_max},{lat_max}"
            f"&fields={fields_str}"
            f"&limit={this_limit}"
        )
        if after:
            url += f"&after={after}"

        data = requests_get_json(url, token=token, timeout_s=timeout_s)
        items = data.get("data", []) or []
        paging = data.get("paging", {}) or {}
        cursors = paging.get("cursors", {}) or {}
        after = cursors.get("after")

        if not items:
            break

        for it in items:
            if scan_budget_ref[0] <= 0:
                break
            scan_budget_ref[0] -= 1

            try:
                img_id = str(it["id"])
                lat, lon = safe_extract_latlon(it)
                captured_at = it.get("captured_at")
                seq = it.get("sequence")
                sequence_id: Optional[str] = None
                if isinstance(seq, dict):
                    sequence_id = seq.get("id")
                thumb = it.get("thumb_1024_url")
                if not thumb:
                    continue

                yield ImageItem(
                    id=img_id,
                    lat=lat,
                    lon=lon,
                    captured_at=captured_at,
                    sequence_id=sequence_id,
                    thumb_url=thumb,
                    bbox=bbox_str,
                    city=city,
                )
            except Exception:
                continue

        if not after:
            break

        if sleep_s > 0:
            time.sleep(sleep_s)


# -----------------------
# aggressive coverage (controlled)
# -----------------------
def download_aggressive_controlled(
    token: str,
    base_bbox: Tuple[float, float, float, float],
    need_new: int,
    max_deg: float,
    expand_factor: float,
    page_limit: int,
    sleep_s: float,
    timeout_s: float,
    fields: List[str],
    city: Optional[str],
    out_dir: Path,
    index_path: Path,
    scan_budget_max: int,
    pbar: tqdm,
    debug: bool,
) -> Tuple[int, bool]:
    """
    Expand up to max_deg, tile, stop as soon as need_new new images are indexed.
    Scan budget is a MAX cap (compat with old --limit).
    Returns (new_indexed, found_any).
    """
    existing_ids = load_existing_ids(index_path)

    new_indexed = 0
    found_any = False
    bbox = base_bbox

    # Budget ref: decremented for each API item seen.
    scan_budget_ref = [max(1, int(scan_budget_max))]

    attempt = 0
    while True:
        attempt += 1
        w, h = bbox_dims(bbox)
        lon_min, lat_min, lon_max, lat_max = bbox
        area = bbox_area(lon_min, lat_min, lon_max, lat_max)
        tiles = split_bbox_into_tiles(lon_min, lat_min, lon_max, lat_max, MAX_TILE_AREA)

        # Use tqdm.write to not break the bar
        tqdm.write(
            f"TRY {attempt}: w={w:.6f} h={h:.6f} area={area:.6f} tiles={len(tiles)} "
            f"need_new={need_new-new_indexed} budget={scan_budget_ref[0]}",
        )

        # Shuffle tiles to avoid always hitting same first pages/places (helps when lots already indexed)
        random.shuffle(tiles)

        for tile in tiles:
            if new_indexed >= need_new or scan_budget_ref[0] <= 0:
                return new_indexed, found_any

            for item in iter_images_in_tile(
                token=token,
                tile=tile,
                page_limit=page_limit,
                sleep_s=sleep_s,
                timeout_s=timeout_s,
                fields=fields,
                city=city,
                scan_budget_ref=scan_budget_ref,
            ):
                found_any = True

                if item.id in existing_ids:
                    continue

                img_path = out_dir / f"{item.id}.jpg"
                if not img_path.exists():
                    try:
                        download_file(item.thumb_url, img_path, token=token, timeout_s=timeout_s)
                    except Exception:
                        continue

                obj = {
                    "id": item.id,
                    "lat": item.lat,
                    "lon": item.lon,
                    "captured_at": item.captured_at,
                    "sequence_id": item.sequence_id,
                    "thumb_1024_url": item.thumb_url,
                    "bbox": item.bbox,
                    "city": item.city,
                    "path": str(img_path),
                }
                append_jsonl(index_path, canonical_obj(obj))
                existing_ids.add(item.id)

                new_indexed += 1
                pbar.update(1)  # LIVE progress

                if debug:
                    tqdm.write(f"NEW {new_indexed}/{need_new}: id={item.id} city={city}")

                if new_indexed >= need_new:
                    return new_indexed, found_any

                if scan_budget_ref[0] <= 0:
                    return new_indexed, found_any

        # if bbox already at cap in both dimensions, stop expanding
        cur_w, cur_h = bbox_dims(bbox)
        if cur_w >= max_deg and cur_h >= max_deg:
            return new_indexed, found_any

        bbox = next_expanded_bbox(bbox, factor=expand_factor, max_deg=max_deg)


# -----------------------
# Main
# -----------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Mapillary downloader (compat + controlled aggressive coverage): "
            "expand bbox up to max_deg, tile, stop once N NEW images indexed."
        )
    )
    p.add_argument("--bbox", required=True, help='lon_min,lat_min,lon_max,lat_max')

    # Backward compatibility: your wrapper calls --limit
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Backward-compatible. Max scan budget (items seen from API). If omitted, uses --scan-budget.",
    )

    p.add_argument("--need-new", type=int, default=10, help="Stop once this many NEW images are indexed.")
    p.add_argument("--scan-budget", type=int, default=8000, help="Max items to scan from API before stopping.")
    p.add_argument("--max-deg", type=float, default=0.5, help="Max bbox width/height in degrees (e.g. 0.5).")
    p.add_argument("--expand-factor", type=float, default=2.0, help="Expansion factor per step.")
    p.add_argument("--page-limit", type=int, default=500, help="Per-request limit for Mapillary paging.")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep between API calls.")
    p.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds.")
    p.add_argument("--token", default=None, help="Mapillary token (MAPILLARY_TOKEN env/.env).")
    p.add_argument("--city", default=None, help="City name stored in index (optional).")
    p.add_argument("--out-dir", default="data/raw/mapillary", help="Where images are stored.")
    p.add_argument("--index", default="data/index/images.jsonl", help="JSONL index path.")
    p.add_argument("--fields", default=",".join(DEFAULT_FIELDS), help="Graph API fields to request.")
    p.add_argument("--debug", action="store_true", help="Verbose debug logs.")
    args = p.parse_args()

    token = get_token(args.token)
    out_dir = Path(args.out_dir)
    index_path = Path(args.index)
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    base_bbox = parse_bbox(args.bbox)

    scan_budget_max = int(args.limit) if args.limit is not None else int(args.scan_budget)
    scan_budget_max = max(1, scan_budget_max)

    print(f"images -> {out_dir}", flush=True)
    print(f"index  -> {index_path}", flush=True)
    print(
        f"need_new={args.need_new} scan_budget={scan_budget_max} max_deg={args.max_deg} "
        f"expand_factor={args.expand_factor} page_limit={args.page_limit}",
        flush=True,
    )

    pbar = tqdm(total=args.need_new, desc="new_indexed", dynamic_ncols=True, file=sys.stdout)

    new_indexed, found_any = download_aggressive_controlled(
        token=token,
        base_bbox=base_bbox,
        need_new=args.need_new,
        max_deg=args.max_deg,
        expand_factor=args.expand_factor,
        page_limit=args.page_limit,
        sleep_s=args.sleep,
        timeout_s=args.timeout,
        fields=fields,
        city=args.city,
        out_dir=out_dir,
        index_path=index_path,
        scan_budget_max=scan_budget_max,
        pbar=pbar,
        debug=args.debug,
    )

    pbar.close()

    print("\nDone.", flush=True)
    print(f"Found any images: {found_any}", flush=True)
    print(f"New indexed: {new_indexed}", flush=True)
    print(f"Index: {index_path}", flush=True)
    print(f"Images: {out_dir}", flush=True)

    if not found_any:
        raise SystemExit(2)
    raise SystemExit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
        raise SystemExit(1)