#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests

JSONL = Path("data/index/images.jsonl")
RAW_DIR = Path("data/raw/mapillary")

ENV_FILE_CANDIDATES = [Path(".env"), Path(".env.local")]
MAPILLARY_URL = "https://graph.mapillary.com/images"

DEFAULT_STATE_DIR = Path("data/state/fill_communes_to_100")
DEFAULT_STUCK_CSV = DEFAULT_STATE_DIR / "stuck_communes.csv"
DEFAULT_DONE_CSV = DEFAULT_STATE_DIR / "done_communes.csv"


# -----------------------
# Utils
# -----------------------
def die(msg: str) -> None:
    raise SystemExit(msg)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def load_env_token() -> Optional[str]:
    tok = (os.environ.get("MAPILLARY_TOKEN") or "").strip()
    if tok:
        return tok

    for p in ENV_FILE_CANDIDATES:
        if not p.exists():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("export "):
                    s = s[len("export ") :].strip()
                if not s.startswith("MAPILLARY_TOKEN"):
                    continue
                if "=" not in s:
                    continue
                k, v = s.split("=", 1)
                if k.strip() != "MAPILLARY_TOKEN":
                    continue
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
        except Exception:
            continue
    return None


def ensure_files(state_dir: Path, stuck_csv: Path, done_csv: Path) -> None:
    JSONL.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    JSONL.touch(exist_ok=True)

    state_dir.mkdir(parents=True, exist_ok=True)

    if not stuck_csv.exists():
        stuck_csv.parent.mkdir(parents=True, exist_ok=True)
        stuck_csv.write_text("city,have,target,bbox,reason,ts\n", encoding="utf-8")

    if not done_csv.exists():
        done_csv.parent.mkdir(parents=True, exist_ok=True)
        done_csv.write_text("city,have,target,bbox,ts\n", encoding="utf-8")


def safe_get(row: Dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def bbox_from_row(row: Dict[str, str]) -> str:
    lon_min = safe_get(row, "lon_min")
    lat_min = safe_get(row, "lat_min")
    lon_max = safe_get(row, "lon_max")
    lat_max = safe_get(row, "lat_max")
    if not (lon_min and lat_min and lon_max and lat_max):
        die(f"Row missing bbox fields: {row}")
    return f"{lon_min},{lat_min},{lon_max},{lat_max}"


# -----------------------
# Global download throttle
# -----------------------
_DL_LOCK = threading.Lock()
_NEXT_DL_T = 0.0


def throttle_download(img_rate: float) -> None:
    global _NEXT_DL_T
    if img_rate <= 0:
        return
    gap = 1.0 / img_rate
    with _DL_LOCK:
        now = time.time()
        if now < _NEXT_DL_T:
            time.sleep(_NEXT_DL_T - now)
        _NEXT_DL_T = max(_NEXT_DL_T, time.time()) + gap


# -----------------------
# State / resume helpers
# -----------------------
def load_city_set(csv_path: Path) -> Set[str]:
    out: Set[str] = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return out
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            if not r.fieldnames or "city" not in r.fieldnames:
                return out
            for row in r:
                c = (row.get("city") or "").strip()
                if c:
                    out.add(c)
    except Exception:
        return out
    return out


def append_state_row(path: Path, header: List[str], row: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})
        f.flush()


# -----------------------
# JSONL index: existing ids + per-city counts
# -----------------------
def load_existing_ids_and_city_counts() -> Tuple[Set[str], Dict[str, int]]:
    ids: Set[str] = set()
    counts: Dict[str, int] = {}
    if not JSONL.exists():
        return ids, counts

    with JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue

            img_id = o.get("id")
            if img_id:
                ids.add(str(img_id))

            city = o.get("city")
            if city:
                c = str(city)
                counts[c] = counts.get(c, 0) + 1

    return ids, counts


def append_jsonl_line(obj: Dict[str, object]) -> None:
    with JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")


# -----------------------
# Mapillary API (single page at a time)
# -----------------------
def mapillary_fetch_page(
    *,
    session: requests.Session,
    token: str,
    bbox: str,
    limit: int,
    after: Optional[str],
    timeout_connect: float,
    timeout_read: float,
    graph_sleep_s: float,
    retries: int,
    backoff: float,
) -> Tuple[List[dict], Optional[str], float]:
    """
    Fetch one page. Returns (items, next_after, dt_seconds).
    """
    fields = "id,computed_geometry,captured_at,sequence_id,thumb_1024_url"
    params = {"bbox": bbox, "fields": fields, "limit": str(limit), "access_token": token}
    if after:
        params["after"] = after

    for attempt in range(1, retries + 1):
        if graph_sleep_s > 0:
            time.sleep(graph_sleep_s)

        t0 = time.time()
        try:
            r = session.get(MAPILLARY_URL, params=params, timeout=(timeout_connect, timeout_read))
            dt = time.time() - t0
            sc = r.status_code

            if sc in (401, 403):
                raise ValueError(f"auth_http_{sc}")
            if sc in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"http_{sc}", response=r)
            if sc >= 400:
                raise ValueError(f"http_{sc}")

            data = r.json()
            items = data.get("data") or []
            if not isinstance(items, list):
                items = []

            paging = data.get("paging") or {}
            cursors = paging.get("cursors") or {}
            next_after = cursors.get("after") or None
            return items, next_after, dt

        except ValueError:
            # auth or hard error: stop commune quickly
            return [], None, 0.0
        except (requests.HTTPError, requests.RequestException):
            if attempt < retries:
                jitter = random.uniform(0.85, 1.15)
                time.sleep(min(20.0, backoff * attempt * jitter))
                continue
            return [], None, 0.0

    return [], None, 0.0


def extract_lat_lon(img: dict) -> Optional[Tuple[float, float]]:
    geom = img.get("computed_geometry")
    if not isinstance(geom, dict):
        return None
    coords = geom.get("coordinates")
    if not (isinstance(coords, list) and len(coords) >= 2):
        return None
    lon = float(coords[0])
    lat = float(coords[1])
    return lat, lon


def download_thumb(
    *,
    session: requests.Session,
    url: str,
    out_path: Path,
    timeout_connect: float,
    timeout_read: float,
    retries: int,
    backoff: float,
) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True

    tmp = out_path.with_suffix(out_path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, stream=True, timeout=(timeout_connect, timeout_read))
            if r.status_code >= 400:
                raise requests.HTTPError(f"thumb_http_{r.status_code}", response=r)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            tmp.replace(out_path)
            return True

        except (requests.RequestException, OSError):
            if attempt < retries:
                jitter = random.uniform(0.85, 1.15)
                time.sleep(min(20.0, backoff * attempt * jitter))
                continue
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return False

    return False


# -----------------------
# CSV input
# -----------------------
def iter_communes_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not safe_get(row, "nom"):
                continue
            yield row


# -----------------------
# Main
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fill each commune to N unique images (Mapillary Graph API).\n"
            "Behavior optimized to minimize API calls:\n"
            "- Aim to add exactly K images per fetch round (default 20)\n"
            "- If a fetch round adds 0 new images -> mark commune STUCK and move on immediately.\n"
        )
    )

    ap.add_argument("--input-csv", default="data/index/coverage/good_communes.csv")
    ap.add_argument("--target", type=int, default=100)

    # polite mode
    ap.add_argument("--img-rate", type=float, default=1.0, help="Max downloaded images per second (default 1.0).")
    ap.add_argument("--graph-sleep", type=float, default=0.6, help="Sleep before each Graph API request (default 0.6).")

    ap.add_argument("--timeout-connect", type=float, default=5.0)
    ap.add_argument("--timeout-read", type=float, default=20.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=2.0)

    # paging/fetch policy
    ap.add_argument("--per-page", type=int, default=50, help="Graph API limit per page (default 50).")
    ap.add_argument("--max-pages-per-round", type=int, default=5, help="Max pages to scan per round (default 5).")
    ap.add_argument("--per-fetch-add", type=int, default=20, help="How many NEW images to add per fetch round (default 20).")

    # downloads
    ap.add_argument("--thumb-retries", type=int, default=3)
    ap.add_argument("--thumb-backoff", type=float, default=2.0)

    # flow
    ap.add_argument("--pause-between-communes", type=float, default=0.2)

    # state
    ap.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    ap.add_argument("--stuck-csv", default="")
    ap.add_argument("--done-csv", default="")

    # logging
    ap.add_argument("--log-every", type=int, default=20)

    args = ap.parse_args()

    token = load_env_token()
    if not token:
        die("Missing MAPILLARY_TOKEN (set env or .env: MAPILLARY_TOKEN=...)")

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        die(f"Missing input CSV: {input_csv}")

    state_dir = Path(args.state_dir)
    stuck_csv = Path(args.stuck_csv) if args.stuck_csv else (state_dir / "stuck_communes.csv")
    done_csv = Path(args.done_csv) if args.done_csv else (state_dir / "done_communes.csv")

    ensure_files(state_dir, stuck_csv, done_csv)

    existing_ids, city_counts = load_existing_ids_and_city_counts()
    stuck_set = load_city_set(stuck_csv)
    done_set = load_city_set(done_csv)

    graph = requests.Session()
    graph.headers.update({"User-Agent": "geo-locate-ml/fill_communes_to_100(graph)"})

    dl = requests.Session()
    dl.headers.update({"User-Agent": "geo-locate-ml/fill_communes_to_100(download)"})

    rows = list(iter_communes_csv(input_csv))
    total_rows = len(rows)

    print(f"== fill_communes_to_{args.target} ==", flush=True)
    print(f"input={input_csv}", flush=True)
    print(f"jsonl={JSONL}", flush=True)
    print(f"raw_dir={RAW_DIR}", flush=True)
    print(f"existing_ids={len(existing_ids)}", flush=True)
    print(f"resume: stuck={len(stuck_set)} done={len(done_set)}", flush=True)
    print(
        f"policy: per_fetch_add={args.per_fetch_add} per_page={args.per_page} max_pages_per_round={args.max_pages_per_round}",
        flush=True,
    )
    print(
        f"polite: img_rate={args.img_rate}/s graph_sleep={args.graph_sleep}s retries={args.retries}",
        flush=True,
    )
    print("", flush=True)

    stuck_header = ["city", "have", "target", "bbox", "reason", "ts"]
    done_header = ["city", "have", "target", "bbox", "ts"]

    for i, row in enumerate(rows, start=1):
        city = safe_get(row, "nom")
        bbox = bbox_from_row(row)
        have = int(city_counts.get(city, 0))

        if city in stuck_set:
            print(f"[{i}/{total_rows}] SKIP(stuck) {city} have={have} target={args.target}", flush=True)
            continue
        if city in done_set:
            print(f"[{i}/{total_rows}] SKIP(done) {city} have={have} target={args.target}", flush=True)
            continue

        if have >= args.target:
            print(f"[{i}/{total_rows}] DONE {city} have={have} target={args.target}", flush=True)
            done_set.add(city)
            append_state_row(
                done_csv,
                done_header,
                {"city": city, "have": str(have), "target": str(args.target), "bbox": bbox, "ts": now_ts()},
            )
            continue

        print(f"\n[{i}/{total_rows}] START {city} have={have} target={args.target} bbox={bbox}", flush=True)

        added_total = 0

        while have < args.target:
            need = min(int(args.per_fetch_add), int(args.target) - have)

            # One "round" = scan pages until we add `need` new images OR pages exhausted.
            round_added = 0
            round_fetched = 0
            after: Optional[str] = None
            round_pages = 0

            while round_added < need and round_pages < int(args.max_pages_per_round):
                round_pages += 1
                items, next_after, dt = mapillary_fetch_page(
                    session=graph,
                    token=token,
                    bbox=bbox,
                    limit=int(args.per_page),
                    after=after,
                    timeout_connect=float(args.timeout_connect),
                    timeout_read=float(args.timeout_read),
                    graph_sleep_s=float(args.graph_sleep),
                    retries=int(args.retries),
                    backoff=float(args.backoff),
                )

                if not items:
                    # No results from Graph: treat as immediate stuck for this commune.
                    print(f"{now_ts()} WARN {city}: no results from Graph dt={dt:.1f}s", flush=True)
                    reason = "no_results_from_graph"
                    print(f"{now_ts()} STUCK {city}: {reason} at {have}/{args.target}", flush=True)
                    stuck_set.add(city)
                    append_state_row(
                        stuck_csv,
                        stuck_header,
                        {"city": city, "have": str(have), "target": str(args.target), "bbox": bbox, "reason": reason, "ts": now_ts()},
                    )
                    round_added = 0
                    round_fetched = 0
                    after = None
                    round_pages = int(args.max_pages_per_round)  # force exit
                    break

                round_fetched += len(items)

                for img in items:
                    if have >= args.target or round_added >= need:
                        break

                    img_id = str(img.get("id") or "").strip()
                    if not img_id or img_id in existing_ids:
                        continue

                    ll = extract_lat_lon(img)
                    if not ll:
                        continue
                    lat, lon = ll

                    captured_at = img.get("captured_at")
                    try:
                        captured_at_int = int(captured_at) if captured_at is not None else None
                    except Exception:
                        captured_at_int = None

                    sequence_id = img.get("sequence_id")
                    if sequence_id is not None:
                        sequence_id = str(sequence_id)

                    thumb = str(img.get("thumb_1024_url") or "").strip()
                    if not thumb:
                        continue

                    out_path = RAW_DIR / f"{img_id}.jpg"

                    throttle_download(float(args.img_rate))

                    ok = download_thumb(
                        session=dl,
                        url=thumb,
                        out_path=out_path,
                        timeout_connect=float(args.timeout_connect),
                        timeout_read=float(args.timeout_read),
                        retries=int(args.thumb_retries),
                        backoff=float(args.thumb_backoff),
                    )
                    if not ok:
                        continue

                    obj = {
                        "id": img_id,
                        "lat": lat,
                        "lon": lon,
                        "captured_at": captured_at_int,
                        "sequence_id": sequence_id if sequence_id else None,
                        "thumb_1024_url": thumb,
                        "bbox": bbox,
                        "city": city,
                        "path": str(out_path),
                    }
                    append_jsonl_line(obj)

                    existing_ids.add(img_id)
                    city_counts[city] = city_counts.get(city, 0) + 1
                    have = int(city_counts[city])

                    round_added += 1
                    added_total += 1

                    if int(args.log_every) > 0 and added_total % int(args.log_every) == 0:
                        print(f"{now_ts()} COUNT {city}: have={have}/{args.target} (+{added_total} total)", flush=True)

                after = next_after
                if not after:
                    break

            # Decision after one fetch round:
            if round_added == 0:
                # Your rule: one 0-new => assume 0-new for subsequent => move on.
                reason = "zero_new_after_fetch_round"
                print(
                    f"{now_ts()} STUCK {city}: fetched={round_fetched} added=0 -> {reason} at {have}/{args.target}",
                    flush=True,
                )
                stuck_set.add(city)
                append_state_row(
                    stuck_csv,
                    stuck_header,
                    {"city": city, "have": str(have), "target": str(args.target), "bbox": bbox, "reason": reason, "ts": now_ts()},
                )
                break

            print(f"{now_ts()} PROGRESS {city}: +{round_added} -> have={have}/{args.target}", flush=True)

        if have >= args.target and city not in done_set:
            done_set.add(city)
            append_state_row(
                done_csv,
                done_header,
                {"city": city, "have": str(have), "target": str(args.target), "bbox": bbox, "ts": now_ts()},
            )
            print(f"{now_ts()} DONE {city}: have={have}/{args.target}", flush=True)

        if args.pause_between_communes > 0:
            time.sleep(float(args.pause_between_communes))

    print("\nDONE.", flush=True)
    print(f"state: stuck_csv={stuck_csv} done_csv={done_csv}", flush=True)


if __name__ == "__main__":
    main()