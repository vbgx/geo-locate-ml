#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h3
import pandas as pd

JSONL = Path("data/index/images.jsonl")
OUT_PARQUET = Path("data/index/images.parquet")
OUT_STATS = Path("data/index/stats.json")

RAW_DIR = Path("data/raw/mapillary")
PROCESSED_DIR = Path("data/processed/mapillary")


@dataclass(frozen=True)
class IndexRow:
    line_idx: int

    id: str
    lat: Optional[float]
    lon: Optional[float]
    path: str
    h3_id: Optional[str]
    city: Optional[str]
    sequence_id: Optional[str]
    captured_at: Optional[str]

    is_duplicate_jsonl: bool
    drop_reason: str
    is_valid: bool
    cell_count: int
    cell_kept: bool
    is_kept: bool


def _as_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "null", "None"):
        return None
    return s


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def choose_path(image_id: str, prefer_processed: bool = True) -> Path:
    p = PROCESSED_DIR / f"{image_id}.jpg"
    r = RAW_DIR / f"{image_id}.jpg"

    if prefer_processed:
        if p.exists():
            return p
        if r.exists():
            return r
        return p
    else:
        if r.exists():
            return r
        if p.exists():
            return p
        return r


def load_jsonl_keep_all_lines(path: Path) -> List[Tuple[int, Optional[Dict[str, Any]], str]]:
    """
    Returns list of (line_idx, obj_or_none, kind)
      - kind="json" for parsed json object
      - kind="empty" for empty/whitespace-only line
      - kind="bad_json" for unparsable json line (obj is None)
    We KEEP every physical line so wc -l parity is achievable.
    """
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run download first.")

    rows: List[Tuple[int, Optional[Dict[str, Any]], str]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            line = raw.strip()
            if not line:
                rows.append((i, None, "empty"))
                continue
            try:
                rows.append((i, json.loads(line), "json"))
            except Exception:
                rows.append((i, None, "bad_json"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build parquet index from images.jsonl (parity-strict, keep all lines).")
    ap.add_argument("--jsonl", default=str(JSONL))
    ap.add_argument("--out-parquet", default=str(OUT_PARQUET))
    ap.add_argument("--out-stats", default=str(OUT_STATS))
    ap.add_argument("--h3-res", type=int, required=True)
    ap.add_argument("--min-cell-samples", type=int, required=True)
    ap.add_argument("--prefer-processed", action="store_true", default=True)
    ap.add_argument("--prefer-raw", action="store_true", help="Override: prefer raw over processed")
    args = ap.parse_args()

    prefer_processed = True
    if args.prefer_raw:
        prefer_processed = False

    jsonl_path = Path(args.jsonl)
    out_parquet = Path(args.out_parquet)
    out_stats = Path(args.out_stats)

    raw = load_jsonl_keep_all_lines(jsonl_path)

    out: List[IndexRow] = []
    skipped_missing_coords = 0
    skipped_missing_id = 0
    empty_lines = 0
    bad_json_lines = 0

    seen: dict[str, int] = {}

    for line_idx, o, kind in raw:
        if kind == "empty":
            empty_lines += 1
            out.append(
                IndexRow(
                    line_idx=int(line_idx),
                    id="",
                    lat=None,
                    lon=None,
                    path=str(PROCESSED_DIR / "MISSING.jpg"),
                    h3_id=None,
                    city=None,
                    sequence_id=None,
                    captured_at=None,
                    is_duplicate_jsonl=False,
                    drop_reason="empty_line",
                    is_valid=False,
                    cell_count=0,
                    cell_kept=False,
                    is_kept=False,
                )
            )
            continue

        if kind == "bad_json":
            bad_json_lines += 1
            out.append(
                IndexRow(
                    line_idx=int(line_idx),
                    id="",
                    lat=None,
                    lon=None,
                    path=str(PROCESSED_DIR / "MISSING.jpg"),
                    h3_id=None,
                    city=None,
                    sequence_id=None,
                    captured_at=None,
                    is_duplicate_jsonl=False,
                    drop_reason="bad_json",
                    is_valid=False,
                    cell_count=0,
                    cell_kept=False,
                    is_kept=False,
                )
            )
            continue

        assert o is not None

        image_id = _as_str(o.get("id")) or ""
        if not image_id:
            skipped_missing_id += 1

        lat = _as_float(o.get("lat"))
        lon = _as_float(o.get("lon"))
        if lat is None or lon is None:
            skipped_missing_coords += 1

        is_dup = False
        if image_id:
            seen[image_id] = seen.get(image_id, 0) + 1
            is_dup = seen[image_id] > 1

        h3_id: Optional[str] = None
        if lat is not None and lon is not None:
            h3_id = str(h3.latlng_to_cell(lat, lon, int(args.h3_res)))

        city = _as_str(o.get("city"))
        seq = _as_str(o.get("sequence_id") or o.get("sequence"))
        captured = _as_str(o.get("captured_at"))

        p = choose_path(image_id, prefer_processed=prefer_processed) if image_id else (PROCESSED_DIR / "MISSING.jpg")

        reasons: list[str] = []
        if not image_id:
            reasons.append("missing_id")
        if lat is None or lon is None:
            reasons.append("missing_latlon")

        drop_reason = "|".join(reasons)
        is_valid = (drop_reason == "") and (h3_id is not None)

        out.append(
            IndexRow(
                line_idx=int(line_idx),
                id=str(image_id),
                lat=lat,
                lon=lon,
                path=str(p),
                h3_id=h3_id,
                city=city,
                sequence_id=seq,
                captured_at=captured,
                is_duplicate_jsonl=bool(is_dup),
                drop_reason=drop_reason,
                is_valid=bool(is_valid),
                cell_count=0,
                cell_kept=False,
                is_kept=False,
            )
        )

    df = pd.DataFrame([asdict(r) for r in out])

    valid_mask = df["is_valid"].astype(bool) & df["h3_id"].notna()
    cell_counts = df.loc[valid_mask, "h3_id"].value_counts()

    df["cell_count"] = df["h3_id"].map(cell_counts).fillna(0).astype(int)
    df["cell_kept"] = df["cell_count"] >= int(args.min_cell_samples)
    df["is_kept"] = df["is_valid"].astype(bool) & df["cell_kept"].astype(bool)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    stats = {
        "rows_jsonl_physical": int(len(raw)),  # should match wc -l
        "rows_total": int(len(df)),
        "rows_valid": int(df["is_valid"].sum()),
        "rows_kept_for_training": int(df["is_kept"].sum()),
        "cells_total_valid": int(cell_counts.shape[0]),
        "cells_kept": int(df.loc[df["is_kept"], "h3_id"].nunique()),
        "h3_res": int(args.h3_res),
        "min_cell_samples": int(args.min_cell_samples),
        "skipped_missing_coords": int(skipped_missing_coords),
        "skipped_missing_id": int(skipped_missing_id),
        "empty_lines": int(empty_lines),
        "bad_json_lines": int(bad_json_lines),
        "prefer_processed": bool(prefer_processed),
        "raw_dir": str(RAW_DIR),
        "processed_dir": str(PROCESSED_DIR),
    }
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Wrote:")
    print(f" - {out_parquet}")
    print(f" - {out_stats}")
    print(f"Rows(parquet)={len(df)}  Valid={int(df['is_valid'].sum())}  Kept={int(df['is_kept'].sum())}")


if __name__ == "__main__":
    main()
