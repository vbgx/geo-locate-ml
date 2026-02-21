#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

IMAGES = Path("data/index/images.jsonl")
SPLITS = Path("data/index/splits.jsonl")

def main() -> None:
    if not IMAGES.exists():
        raise SystemExit(f"Missing {IMAGES}")
    if not SPLITS.exists():
        raise SystemExit(f"Missing {SPLITS}")

    split_map: dict[str, str] = {}
    with SPLITS.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            _id = str(o.get("id") or "")
            sp = str(o.get("split") or "")
            if _id and sp:
                split_map[_id] = sp

    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    total = 0

    with IMAGES.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            _id = str(o.get("id") or "")
            sp = split_map.get(_id, "unknown")
            counts[sp] = counts.get(sp, 0) + 1
            total += 1

    known = total - counts.get("unknown", 0)
    print("Total images.jsonl:", total)
    print("Split rows:", len(split_map))
    print("Known (in splits):", known)
    print("Counts by split:", dict(sorted(counts.items(), key=lambda kv: kv[0])))
    print("Unknown:", counts.get("unknown", 0))

if __name__ == "__main__":
    main()
