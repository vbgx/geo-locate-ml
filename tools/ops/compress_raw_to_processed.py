#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image

RAW_DIR = Path("data/raw/mapillary")
PROCESSED_DIR = Path("data/processed/mapillary")
JSONL = Path("data/index/images.jsonl")


def iter_image_ids(jsonl_path: Path) -> Iterable[str]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            img_id = str(o.get("id", "")).strip()
            if img_id:
                yield img_id


def compress_one(
    img_id: str,
    max_size: int,
    quality: int,
    delete_raw: bool,
) -> Tuple[str, str]:
    """
    Returns (status, img_id)
    status in: ok, skip_missing_raw, skip_exists, error
    """
    src = RAW_DIR / f"{img_id}.jpg"
    dst = PROCESSED_DIR / f"{img_id}.jpg"

    if not src.exists():
        return ("skip_missing_raw", img_id)

    # Skip if processed already exists
    if dst.exists():
        return ("skip_exists", img_id)

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((max_size, max_size))
            im.save(dst, "JPEG", quality=quality, optimize=True)

        if delete_raw:
            try:
                src.unlink()
            except FileNotFoundError:
                pass

        return ("ok", img_id)

    except Exception:
        # best effort cleanup of partial file
        try:
            if dst.exists():
                dst.unlink()
        except Exception:
            pass
        return ("error", img_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compress raw Mapillary JPGs into processed thumbnails (multi-thread).")
    ap.add_argument("--max-size", type=int, default=512, help="Max width/height in pixels (default 512)")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality 1-95 (default 80)")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2), help="Thread workers (default ~cpu/2)")
    ap.add_argument("--delete-raw", action="store_true", help="Delete raw JPG after successful processed write")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = no limit)")
    args = ap.parse_args()

    if not JSONL.exists():
        raise SystemExit(f"Missing {JSONL}. Download images first.")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ids = list(iter_image_ids(JSONL))
    if args.limit and args.limit > 0:
        ids = ids[: args.limit]

    print(f"Input jsonl: {JSONL} (ids={len(ids)})")
    print(f"Raw dir:     {RAW_DIR}")
    print(f"Processed:   {PROCESSED_DIR}")
    print(f"max_size={args.max_size} quality={args.quality} workers={args.workers} delete_raw={args.delete_raw}")

    stats = {"ok": 0, "skip_missing_raw": 0, "skip_exists": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(compress_one, img_id, args.max_size, args.quality, args.delete_raw)
            for img_id in ids
        ]
        done = 0
        for fut in as_completed(futs):
            status, _img_id = fut.result()
            stats[status] += 1
            done += 1
            if done % 500 == 0:
                print(f"Processed {done}/{len(ids)} — ok={stats['ok']} skip_exists={stats['skip_exists']} missing_raw={stats['skip_missing_raw']} err={stats['error']}")

    print("Done.")
    print("Stats:", stats)


if __name__ == "__main__":
    main()
