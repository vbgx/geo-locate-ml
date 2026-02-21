#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

MAPILLARY_URL = "https://graph.mapillary.com/images"
THREAD_LOCAL = threading.local()

# -----------------------
# Thread-local HTTP session (bigger pool)
# -----------------------
def get_session() -> requests.Session:
    s = getattr(THREAD_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=0)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        # optional UA
        s.headers.update({"User-Agent": "geo-locate-ml/scan_communes_coverage_mt"})
        THREAD_LOCAL.session = s
    return s


def safe_get(row: Dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def bbox_from_row(row: Dict[str, str]) -> str:
    return f"{safe_get(row,'lon_min')},{safe_get(row,'lat_min')},{safe_get(row,'lon_max')},{safe_get(row,'lat_max')}"


def commune_key(row: Dict[str, str]) -> str:
    insee = safe_get(row, "insee")
    if insee:
        return f"insee:{insee}"
    nom = safe_get(row, "nom")
    return f"nom:{nom}"


def load_processed_keys(path: Path) -> Set[str]:
    keys: Set[str] = set()
    if not path.exists() or path.stat().st_size == 0:
        return keys
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                insee = safe_get(row, "insee")
                nom = safe_get(row, "nom")
                if insee:
                    keys.add(f"insee:{insee}")
                elif nom:
                    keys.add(f"nom:{nom}")
    except Exception:
        pass
    return keys


def read_batches(batch_dir: Path, start: int, end: int, processed: Set[str]) -> List[Tuple[str, Dict[str, str]]]:
    rows: List[Tuple[str, Dict[str, str]]] = []
    for i in range(start, end + 1):
        p = batch_dir / f"communes_batch_{i:03d}.csv"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if not safe_get(row, "nom"):
                    continue
                ok = True
                for k in ("lon_min", "lat_min", "lon_max", "lat_max"):
                    if not safe_get(row, k):
                        ok = False
                        break
                if not ok:
                    continue
                k = commune_key(row)
                if k in processed:
                    continue
                rows.append((p.name, row))
    return rows


# -----------------------
# Global rate limiter (shared across threads)
# -----------------------
class RateLimiter:
    """
    Simple global rate limiter: ensures average <= rate requests/second across all threads.
    rate<=0 disables.
    """
    def __init__(self, rate_per_s: float) -> None:
        self.rate = float(rate_per_s)
        self.lock = threading.Lock()
        self.next_ts = time.monotonic()

    def wait(self) -> None:
        if self.rate <= 0:
            return
        gap = 1.0 / self.rate
        with self.lock:
            now = time.monotonic()
            if now < self.next_ts:
                sleep_s = self.next_ts - now
                self.next_ts += gap
            else:
                sleep_s = 0.0
                self.next_ts = now + gap
        if sleep_s > 0:
            time.sleep(sleep_s)


def check_bbox_status(
    bbox: str,
    *,
    token: str,
    limit: int,
    timeout_connect: float,
    timeout_read: float,
    retries: int,
    backoff_s: float,
    limiter: Optional[RateLimiter],
) -> str:
    session = get_session()
    params = {"bbox": bbox, "fields": "id", "limit": str(limit), "access_token": token}

    for attempt in range(1, retries + 1):
        if limiter is not None:
            limiter.wait()

        try:
            r = session.get(MAPILLARY_URL, params=params, timeout=(timeout_connect, timeout_read))

            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    return "unknown"
                return "good" if bool(data.get("data")) else "dead"

            # auth: stop fast
            if r.status_code in (401, 403):
                return "unknown"

            # retryable
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt < retries:
                    # exponential-ish backoff + jitter
                    time.sleep(min(10.0, backoff_s * attempt + random.uniform(0, 0.35)))
                    continue
                return "unknown"

            # other 4xx -> unknown
            return "unknown"

        except requests.RequestException:
            if attempt < retries:
                time.sleep(min(10.0, backoff_s * attempt + random.uniform(0, 0.35)))
                continue
            return "unknown"

    return "unknown"


# -----------------------
# Writer thread
# -----------------------
@dataclass
class OutItem:
    status: str
    out_row: Dict[str, str]


def main() -> None:
    ap = argparse.ArgumentParser(description="MT scan commune coverage with resume + bounded queue + writer + rate limiter.")
    ap.add_argument("--batch-dir", default="data/communes/batches")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=509)
    ap.add_argument("--workers", type=int, default=6)

    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--timeout-connect", type=float, default=5.0)
    ap.add_argument("--timeout-read", type=float, default=20.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=1.5)

    # keep your old knob, but now it's "extra sleep after a request"
    ap.add_argument("--sleep", type=float, default=0.0, help="extra delay AFTER each task (normally 0)")

    # NEW: global rate limit (req/s across all workers). 0 disables.
    ap.add_argument("--rate", type=float, default=0.0, help="global requests/sec across all workers (recommended 1.0–2.5).")

    ap.add_argument("--out-dir", default="data/index/coverage")
    ap.add_argument("--print-every", type=int, default=100)
    ap.add_argument("--heartbeat-s", type=int, default=5)
    ap.add_argument("--flush-every", type=int, default=200, help="flush CSV files every N writes (default 200)")

    args = ap.parse_args()

    token = os.environ.get("MAPILLARY_TOKEN", "").strip()
    if not token:
        print("ERROR: MAPILLARY_TOKEN missing (expected in .env).", flush=True)
        raise SystemExit(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    good_csv = out_dir / "good_communes.csv"
    dead_csv = out_dir / "dead_communes.csv"
    unknown_csv = out_dir / "unknown_communes.csv"

    # Resume: load already processed
    processed: Set[str] = set()
    processed |= load_processed_keys(good_csv)
    processed |= load_processed_keys(dead_csv)
    processed |= load_processed_keys(unknown_csv)

    items = read_batches(Path(args.batch_dir), args.start, args.end, processed)
    total = len(items)

    print(f"Resume: already processed = {len(processed)}", flush=True)
    print(f"Rows to scan: {total}", flush=True)
    print(f"Output: {good_csv} / {dead_csv} / {unknown_csv}", flush=True)
    if args.rate and args.rate > 0:
        print(f"Rate limit: {args.rate:.2f} req/s (global)", flush=True)

    if total == 0:
        print("Nothing to do.", flush=True)
        return

    fieldnames = ["batch", "insee", "nom", "lon_min", "lat_min", "lon_max", "lat_max", "radius_km", "limit", "min_target"]

    def open_append(path: Path):
        exists = path.exists()
        f = path.open("a", encoding="utf-8", newline="")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if (not exists) or path.stat().st_size == 0:
            w.writeheader()
            f.flush()
        return w, f

    w_good, fg = open_append(good_csv)
    w_dead, fd = open_append(dead_csv)
    w_unk, fu = open_append(unknown_csv)

    # Queues
    work_q: "queue.Queue[Optional[Tuple[str, Dict[str, str]]]]" = queue.Queue(maxsize=5000)
    out_q: "queue.Queue[Optional[OutItem]]" = queue.Queue(maxsize=5000)

    # Counters (writer thread owns them -> no locks)
    done = 0
    good_n = 0
    dead_n = 0
    unk_n = 0
    flush_counter = 0
    print_counter = 0

    t0 = time.time()
    last_hb = t0

    limiter = RateLimiter(args.rate) if args.rate and args.rate > 0 else None

    def to_out_row(batch_name: str, row: Dict[str, str]) -> Dict[str, str]:
        d = {k: safe_get(row, k) for k in fieldnames if k != "batch"}
        d["batch"] = batch_name
        return d

    # Writer thread
    def writer_loop() -> None:
        nonlocal done, good_n, dead_n, unk_n, flush_counter, print_counter, last_hb
        try:
            while True:
                item = out_q.get()
                if item is None:
                    break

                done += 1
                if item.status == "good":
                    w_good.writerow(item.out_row); good_n += 1
                elif item.status == "dead":
                    w_dead.writerow(item.out_row); dead_n += 1
                else:
                    w_unk.writerow(item.out_row); unk_n += 1

                flush_counter += 1
                print_counter += 1

                now = time.time()
                if flush_counter >= args.flush_every:
                    fg.flush(); fd.flush(); fu.flush()
                    flush_counter = 0

                if args.print_every and print_counter >= args.print_every:
                    rate = done / max(1e-6, (now - t0))
                    eta_s = (total - done) / max(1e-6, rate)
                    print(
                        f"{done}/{total} rate={rate:.2f}/s good={good_n} dead={dead_n} unk={unk_n} eta={eta_s/3600:.2f}h",
                        flush=True,
                    )
                    print_counter = 0

                if now - last_hb >= args.heartbeat_s:
                    rate = done / max(1e-6, (now - t0))
                    print(
                        f"Heartbeat: {done}/{total} ({rate:.2f}/s) good={good_n} dead={dead_n} unk={unk_n}",
                        flush=True,
                    )
                    last_hb = now
        finally:
            try:
                fg.flush(); fd.flush(); fu.flush()
            except Exception:
                pass

    wt = threading.Thread(target=writer_loop, daemon=True)
    wt.start()

    # Workers (consume work_q, produce out_q)
    def worker_loop(worker_id: int) -> None:
        while True:
            item = work_q.get()
            if item is None:
                return
            batch_name, row = item
            bbox = bbox_from_row(row)
            status = check_bbox_status(
                bbox,
                token=token,
                limit=args.limit,
                timeout_connect=args.timeout_connect,
                timeout_read=args.timeout_read,
                retries=args.retries,
                backoff_s=args.backoff,
                limiter=limiter,
            )
            if args.sleep and args.sleep > 0:
                time.sleep(args.sleep)
            out_q.put(OutItem(status=status, out_row=to_out_row(batch_name, row)))

    workers = [threading.Thread(target=worker_loop, args=(i,), daemon=True) for i in range(max(1, args.workers))]
    for t in workers:
        t.start()

    try:
        # Producer: feed queue progressively (no 120k futures)
        for batch_name, row in items:
            work_q.put((batch_name, row))
    finally:
        # Stop workers
        for _ in workers:
            work_q.put(None)
        for t in workers:
            t.join()

        # Stop writer
        out_q.put(None)
        wt.join(timeout=30)

        try: fg.close()
        except Exception: pass
        try: fd.close()
        except Exception: pass
        try: fu.close()
        except Exception: pass

    dt = max(1e-6, time.time() - t0)
    print("DONE.", flush=True)
    print(f"added good={good_n} dead={dead_n} unknown={unk_n}  rate={done/dt:.2f}/s", flush=True)


if __name__ == "__main__":
    main()