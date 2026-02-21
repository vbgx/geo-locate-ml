#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# =============================================================================
# Defaults = "the command that works"
# =============================================================================
DEFAULT_LEVEL = 5
DEFAULT_NO_DUPLICATES = True
DEFAULT_EDGE_SIDE = 128
DEFAULT_BACKUP_INDEX = True
DEFAULT_WRITE_INDEX = True
DEFAULT_LOG_EVERY = 25

DEFAULT_PROCESSED_DIR = "data/processed/mapillary"
DEFAULT_INDEX_JSONL = "data/index/images.jsonl"
DEFAULT_REPORT_JSONL = "reports/scan_processed_quality.jsonl"


# =============================================================================
# Utils
# =============================================================================
def log(msg: str) -> None:
    print(msg, flush=True)


def is_image_path(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED_EXT


def scan_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if is_image_path(p):
            yield p


def extract_id_from_path(p: Path) -> str:
    return p.stem


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def now_s() -> float:
    return time.perf_counter()


def fmt_eta(done: int, total: int, elapsed_s: float) -> str:
    if done <= 0:
        return "ETA: ?"
    rate = elapsed_s / done
    rem = max(0.0, (total - done) * rate)
    m, s = divmod(int(rem), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"ETA: {h}h{m:02d}m"
    return f"ETA: {m}m{s:02d}s"


def backup_file(path: Path) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = path.with_name(path.name + f".bak-{ts}")
    ensure_parent_dir(bak)
    bak.write_bytes(path.read_bytes())
    return bak


# =============================================================================
# Index JSONL I/O
# =============================================================================
def load_index_jsonl(index_path: Path) -> Tuple[List[dict], Dict[str, int]]:
    rows: List[dict] = []
    id_to_idx: Dict[str, int] = {}

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                log(f"[warn] invalid json at line {line_no}, skipping")
                continue
            rows.append(obj)
            _id = obj.get("id")
            if isinstance(_id, str) and _id and _id not in id_to_idx:
                id_to_idx[_id] = len(rows) - 1

    return rows, id_to_idx


def write_index_jsonl_atomic(index_path: Path, rows: List[dict]) -> None:
    tmp = index_path.with_suffix(index_path.suffix + ".tmp")
    ensure_parent_dir(tmp)
    with tmp.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(index_path)


# =============================================================================
# Image metrics
# =============================================================================
def to_gray_uint8(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.uint8)


def laplacian_var(gray: np.ndarray) -> float:
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.int16)
    g = gray.astype(np.int16)
    p = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    lap = (
        k[0, 0] * p[:-2, :-2] + k[0, 1] * p[:-2, 1:-1] + k[0, 2] * p[:-2, 2:] +
        k[1, 0] * p[1:-1, :-2] + k[1, 1] * p[1:-1, 1:-1] + k[1, 2] * p[1:-1, 2:] +
        k[2, 0] * p[2:, :-2] + k[2, 1] * p[2:, 1:-1] + k[2, 2] * p[2:, 2:]
    )
    return float(lap.var())


def image_entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def sobel_edge_density(gray: np.ndarray) -> float:
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.int16)
    gy = np.array([[-1, -2, -1],
                   [0,  0,  0],
                   [1,  2,  1]], dtype=np.int16)
    g = gray.astype(np.int16)
    p = np.pad(g, ((1, 1), (1, 1)), mode="edge")

    sx = (
        gx[0, 0] * p[:-2, :-2] + gx[0, 1] * p[:-2, 1:-1] + gx[0, 2] * p[:-2, 2:] +
        gx[1, 0] * p[1:-1, :-2] + gx[1, 1] * p[1:-1, 1:-1] + gx[1, 2] * p[1:-1, 2:] +
        gx[2, 0] * p[2:, :-2] + gx[2, 1] * p[2:, 1:-1] + gx[2, 2] * p[2:, 2:]
    )
    sy = (
        gy[0, 0] * p[:-2, :-2] + gy[0, 1] * p[:-2, 1:-1] + gy[0, 2] * p[:-2, 2:] +
        gy[1, 0] * p[1:-1, :-2] + gy[1, 1] * p[1:-1, 1:-1] + gy[1, 2] * p[1:-1, 2:] +
        gy[2, 0] * p[2:, :-2] + gy[2, 1] * p[2:, 1:-1] + gy[2, 2] * p[2:, 2:]
    )
    mag = np.sqrt(sx.astype(np.float32) ** 2 + sy.astype(np.float32) ** 2)
    thr = np.percentile(mag, 90) * 0.25
    if thr <= 0:
        return 0.0
    return float((mag >= thr).mean())


def preprocess_compat_check(
    im: Image.Image,
    *,
    input_size: int,
    resize_shorter_to: int,
    crop: str,
    require_rgb: bool,
) -> None:
    im = ImageOps.exif_transpose(im)
    if require_rgb:
        if "A" in im.getbands():
            raise ValueError("has_alpha")
        if im.mode != "RGB":
            im = im.convert("RGB")

    w, h = im.size
    if w <= 0 or h <= 0:
        raise ValueError("invalid_dimensions")

    if w < h:
        new_w = resize_shorter_to
        new_h = int(round(h * (resize_shorter_to / w)))
    else:
        new_h = resize_shorter_to
        new_w = int(round(w * (resize_shorter_to / h)))
    im = im.resize((new_w, new_h), resample=Image.BILINEAR)

    w, h = im.size
    if w < input_size or h < input_size:
        raise ValueError("too_small_after_resize")

    if crop == "center":
        left = (w - input_size) // 2
        top = (h - input_size) // 2
    elif crop == "random":
        left = int(np.random.randint(0, w - input_size + 1))
        top = int(np.random.randint(0, h - input_size + 1))
    else:
        raise ValueError("invalid_crop")

    im = im.crop((left, top, left + input_size, top + input_size))
    arr = np.asarray(im, dtype=np.uint8)
    if arr.shape[:2] != (input_size, input_size):
        raise ValueError("bad_crop_size")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("bad_channels")


# =============================================================================
# Verdict
# =============================================================================
@dataclass
class Verdict:
    quality: str
    reasons: List[str]
    metrics: Dict[str, float]


def build_comment(reasons: List[str], metrics: Dict[str, float]) -> str:
    comment = ";".join(reasons) if reasons else ""
    extras = []
    for k in ["w", "h", "aspect", "lap_var", "std", "entropy", "edge_density"]:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                extras.append(f"{k}={v:.4f}" if k == "edge_density" else f"{k}={v:.2f}")
    if extras:
        comment = (comment + " | " if comment else "") + " ".join(extras)
    return comment


def assess_image(
    path: Path,
    *,
    level: int,
    require_rgb: bool,
    min_size: int,
    max_aspect: float,
    blur_thr: float,
    dark_pct_thr: float,
    bright_pct_thr: float,
    low_contrast_thr: float,
    input_size: int,
    resize_shorter_to: int,
    crop: str,
    entropy_thr: float,
    edge_density_thr: float,
    edge_side: int,
) -> Verdict:
    reasons: List[str] = []
    metrics: Dict[str, float] = {}

    try:
        with Image.open(path) as im_v:
            im_v.verify()
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im.load()

            # L1
            w, h = im.size
            metrics["w"] = float(w)
            metrics["h"] = float(h)

            if min(w, h) < min_size:
                reasons.append(f"too_small:{w}x{h}<min{min_size}")

            aspect = max(w / h, h / w) if (w > 0 and h > 0) else 999.0
            metrics["aspect"] = float(aspect)
            if aspect > max_aspect:
                reasons.append(f"extreme_aspect:{w}x{h}>{max_aspect:.2f}")

            bands = im.getbands()
            if require_rgb:
                if "A" in bands:
                    reasons.append(f"has_alpha:{bands}")
                if im.mode != "RGB":
                    reasons.append(f"non_rgb_mode:{im.mode}")

            # L2
            if level >= 2:
                gray = to_gray_uint8(im)
                lv = laplacian_var(gray)
                metrics["lap_var"] = float(lv)
                if lv < blur_thr:
                    reasons.append(f"blurry:lap_var{lv:.1f}<thr{blur_thr}")

                dark = float((gray <= 10).mean())
                bright = float((gray >= 245).mean())
                std = float(gray.std())
                metrics["dark_pct"] = float(dark)
                metrics["bright_pct"] = float(bright)
                metrics["std"] = float(std)

                if dark > dark_pct_thr:
                    reasons.append(f"too_dark:{dark:.3f}>{dark_pct_thr}")
                if bright > bright_pct_thr:
                    reasons.append(f"too_bright:{bright:.3f}>{bright_pct_thr}")
                if std < low_contrast_thr:
                    reasons.append(f"low_contrast:std{std:.1f}<thr{low_contrast_thr}")

            # L3
            if level >= 3:
                try:
                    preprocess_compat_check(
                        im,
                        input_size=input_size,
                        resize_shorter_to=resize_shorter_to,
                        crop=crop,
                        require_rgb=require_rgb,
                    )
                except Exception as e:
                    reasons.append(f"preprocess_fail:{type(e).__name__}:{e}")

            # L4
            if level >= 4:
                gray = to_gray_uint8(im)
                ent = image_entropy(gray)
                metrics["entropy"] = float(ent)
                if ent < entropy_thr:
                    reasons.append(f"low_entropy:{ent:.2f}<thr{entropy_thr}")

            # L5 (fast: downscale)
            if level >= 5:
                gim = ImageOps.exif_transpose(im).convert("L").resize((edge_side, edge_side), Image.BILINEAR)
                g = np.asarray(gim, dtype=np.uint8)
                ed = sobel_edge_density(g)
                metrics["edge_density"] = float(ed)
                if ed < edge_density_thr:
                    reasons.append(f"low_edge_density:{ed:.4f}<thr{edge_density_thr}")

    except Exception as e:
        return Verdict("BAD", [f"decode_fail:{type(e).__name__}:{e}"], metrics)

    hard_prefixes = ("decode_fail", "preprocess_fail", "has_alpha", "too_small", "extreme_aspect")
    mid_prefixes = ("blurry", "too_dark", "too_bright", "low_contrast", "low_entropy", "low_edge_density", "non_rgb_mode")

    hard = any(r.startswith(hard_prefixes) for r in reasons)
    mid = any(r.startswith(mid_prefixes) for r in reasons)

    if hard:
        q = "BAD"
    elif mid:
        q = "OK"
    else:
        q = "GOOD"

    return Verdict(q, reasons, metrics)


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    p = argparse.ArgumentParser(
        description="Scan processed images and write quality fields into index JSONL. Defaults are tuned to run fast and safely."
    )

    p.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--index-jsonl", default=DEFAULT_INDEX_JSONL)
    p.add_argument("--report-jsonl", default=DEFAULT_REPORT_JSONL)

    # "just run it" defaults:
    p.add_argument("--level", type=int, default=DEFAULT_LEVEL, choices=[1, 2, 3, 4, 5])
    p.add_argument("--no-duplicates", action="store_true", default=DEFAULT_NO_DUPLICATES)  # kept for compatibility, but unused now
    p.add_argument("--edge-side", type=int, default=DEFAULT_EDGE_SIDE)
    p.add_argument("--backup-index", action="store_true", default=DEFAULT_BACKUP_INDEX)
    p.add_argument("--write-index", action="store_true", default=DEFAULT_WRITE_INDEX)
    p.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)

    # optional controls
    p.add_argument("--dry-run", action="store_true", help="Do not write index, only produce report and logs")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--verbose", action="store_true")

    # thresholds / model params
    p.add_argument("--require-rgb", action="store_true", default=True)
    p.add_argument("--min-size", type=int, default=160)
    p.add_argument("--max-aspect", type=float, default=3.0)
    p.add_argument("--blur-thr", type=float, default=140.0)
    p.add_argument("--dark-pct", type=float, default=0.20)
    p.add_argument("--bright-pct", type=float, default=0.20)
    p.add_argument("--low-contrast", type=float, default=25.0)

    p.add_argument("--input-size", type=int, default=128)
    p.add_argument("--resize-shorter-to", type=int, default=160)
    p.add_argument("--crop", choices=["center", "random"], default="center")

    p.add_argument("--entropy-thr", type=float, default=4.0)
    p.add_argument("--edge-density-thr", type=float, default=0.020)

    args = p.parse_args()

    processed_root = Path(args.processed_dir).resolve()
    index_path = Path(args.index_jsonl).resolve()
    report_path = Path(args.report_jsonl).resolve()
    ensure_parent_dir(report_path)

    if not processed_root.exists():
        log(f"ERROR: processed dir not found: {processed_root}")
        return 2
    if not index_path.exists():
        log(f"ERROR: index jsonl not found: {index_path}")
        return 2

    rows, id_to_idx = load_index_jsonl(index_path)

    items: List[Tuple[str, Path]] = []
    for img_path in scan_images(processed_root):
        _id = extract_id_from_path(img_path)
        if _id in id_to_idx:
            items.append((_id, img_path))
            if args.limit and len(items) >= args.limit:
                break

    if not items:
        log("No processed images matched to index by id.")
        return 0

    n_total = len(items)

    # In this simplified version, duplicates are intentionally NOT computed.
    # (--no-duplicates kept to match your preferred command; it's a no-op.)

    log(f"Processed dir : {processed_root}")
    log(f"Index JSONL   : {index_path}")
    log(f"Report JSONL  : {report_path}")
    log(f"Level         : {args.level}")
    log(f"Edge side     : {args.edge_side}")
    log(f"Mode          : {'DRY-RUN' if args.dry_run else 'LIVE'} | write_index={bool(args.write_index)} | backup_index={bool(args.backup_index)}")
    log(f"Images matched: {n_total}")
    log("-" * 80)

    counts = {"GOOD": 0, "OK": 0, "BAD": 0}
    updated = 0

    wall0 = now_s()
    with report_path.open("w", encoding="utf-8") as rep:
        for i, (_id, img_path) in enumerate(items, start=1):
            idx = id_to_idx[_id]

            v = assess_image(
                img_path,
                level=args.level,
                require_rgb=bool(args.require_rgb),
                min_size=args.min_size,
                max_aspect=args.max_aspect,
                blur_thr=args.blur_thr,
                dark_pct_thr=args.dark_pct,
                bright_pct_thr=args.bright_pct,
                low_contrast_thr=args.low_contrast,
                input_size=args.input_size,
                resize_shorter_to=args.resize_shorter_to,
                crop=args.crop,
                entropy_thr=args.entropy_thr,
                edge_density_thr=args.edge_density_thr,
                edge_side=args.edge_side,
            )

            counts[v.quality] += 1
            comment = build_comment(v.reasons, v.metrics)

            rep.write(json.dumps({
                "id": _id,
                "path": str(img_path),
                "quality": v.quality,
                "quality_level": args.level,
                "comments": comment,
            }, ensure_ascii=False) + "\n")

            # Always overwrite fields (both directions)
            rows[idx]["quality"] = v.quality
            rows[idx]["quality_level"] = args.level
            rows[idx]["comments"] = comment
            updated += 1

            if args.verbose:
                reason = v.reasons[0] if v.reasons else "ok"
                log(f"[{i}/{n_total}] L{args.level} {v.quality:4s} id={_id} file={img_path.name} reason={reason}")

            if args.log_every and (i == 1 or i % args.log_every == 0 or i == n_total):
                elapsed = now_s() - wall0
                avg_ms = (elapsed / i) * 1000.0
                log(f"[progress] {i}/{n_total} | avg={avg_ms:.1f}ms/img | GOOD={counts['GOOD']} OK={counts['OK']} BAD={counts['BAD']} | {fmt_eta(i, n_total, elapsed)}")

    wall_scan = now_s() - wall0

    log("\n" + "=" * 80)
    log("SUMMARY")
    log("=" * 80)
    log(f"GOOD                 : {counts['GOOD']}")
    log(f"OK                   : {counts['OK']}")
    log(f"BAD                  : {counts['BAD']}")
    log(f"Updated rows (index) : {updated}")
    log(f"Wall scan (s)        : {wall_scan:.2f}")
    log(f"Report JSONL         : {report_path}")
    log("=" * 80)

    # Write index?
    if args.dry_run:
        log("NOTE: dry-run enabled -> index not written.")
        return 0
    if not args.write_index:
        log("NOTE: write-index disabled -> index not written.")
        return 0

    if args.backup_index:
        bak = backup_file(index_path)
        log(f"BACKUP: {bak}")

    write_index_jsonl_atomic(index_path, rows)
    log(f"WROTE: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
