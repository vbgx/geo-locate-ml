#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ScanResult:
    ok: bool
    reason: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


def is_image_path(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED_EXT


def extract_id_from_path(img_path: Path) -> str:
    return img_path.stem


def scan_images(raw_root: Path) -> Iterable[Path]:
    for p in raw_root.rglob("*"):
        if is_image_path(p):
            yield p


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normpath_str(p: str) -> str:
    return os.path.normpath(p)


def remove_index_lines(index_path: Path, image_id: str, raw_path: Optional[Path] = None) -> Tuple[int, int]:
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    removed = 0
    kept = 0

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    raw_path_str = str(raw_path) if raw_path is not None else None
    raw_path_norm = normpath_str(raw_path_str) if raw_path_str is not None else None

    with index_path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                fout.write(line)
                kept += 1
                continue

            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                fout.write(line)
                kept += 1
                continue

            match = False
            obj_id = obj.get("id")
            if isinstance(obj_id, str) and obj_id == image_id:
                match = True

            if not match and raw_path_norm is not None:
                p = obj.get("path")
                if isinstance(p, str) and p and normpath_str(p) == raw_path_norm:
                    match = True

            if not match:
                p = obj.get("path")
                if isinstance(p, str) and p and Path(p).stem == image_id:
                    match = True

            if match:
                removed += 1
                continue

            fout.write(line)
            kept += 1

    tmp_path.replace(index_path)
    return removed, kept


def copy_to_processed(raw_path: Path, raw_root: Path, processed_root: Path) -> Path:
    rel = raw_path.relative_to(raw_root)
    dst = processed_root / rel
    ensure_parent_dir(dst)
    shutil.copy2(raw_path, dst)
    return dst


def _to_gray_uint8(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.uint8)


def _laplacian_var(gray: np.ndarray) -> float:
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


def preprocess_dry_run(
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
            raise ValueError(f"has_alpha bands={im.getbands()}")
        if im.mode != "RGB":
            im = im.convert("RGB")

    w, h = im.size
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid_dimensions {w}x{h}")

    if w < h:
        new_w = resize_shorter_to
        new_h = int(round(h * (resize_shorter_to / w)))
    else:
        new_h = resize_shorter_to
        new_w = int(round(w * (resize_shorter_to / h)))
    im = im.resize((new_w, new_h), resample=Image.BILINEAR)
    w, h = im.size

    if w < input_size or h < input_size:
        raise ValueError(f"too_small_after_resize {w}x{h} (<{input_size})")

    if crop == "center":
        left = (w - input_size) // 2
        top = (h - input_size) // 2
    elif crop == "random":
        left = int(np.random.randint(0, w - input_size + 1))
        top = int(np.random.randint(0, h - input_size + 1))
    else:
        raise ValueError(f"invalid_crop_mode {crop}")

    im = im.crop((left, top, left + input_size, top + input_size))
    arr = np.asarray(im, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"bad_tensor_shape {arr.shape}")
    if arr.shape[0] != input_size or arr.shape[1] != input_size:
        raise ValueError(f"bad_tensor_size {arr.shape}")


def make_thumb_data_uri(img_path: Path, max_side: int = 900) -> str:
    """
    For carousel: generate a larger preview. This can be heavy; page may take time to load.
    Use --limit while reviewing to keep it smooth.
    """
    try:
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im)
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=82)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


def try_open_image(
    img_path: Path,
    *,
    level: int,
    min_size: int,
    require_rgb: bool,
    max_aspect: float,
    blur_thr: float,
    dark_pct_thr: float,
    bright_pct_thr: float,
    low_contrast_thr: float,
    input_size: int,
    resize_shorter_to: int,
    crop: str,
) -> ScanResult:
    metrics: Dict[str, float] = {}
    try:
        with Image.open(img_path) as im:
            im.verify()

        with Image.open(img_path) as im2:
            im2.load()
            w, h = im2.size
            metrics["w"] = float(w)
            metrics["h"] = float(h)

            if level >= 1:
                if min(w, h) < min_size:
                    return ScanResult(False, f"too_small {w}x{h} (min<{min_size})", metrics)

                aspect = max(w / h, h / w)
                metrics["aspect"] = float(aspect)
                if aspect > max_aspect:
                    return ScanResult(False, f"extreme_aspect {w}x{h} (>{max_aspect:.2f})", metrics)

                if require_rgb and "A" in im2.getbands():
                    return ScanResult(False, f"has_alpha bands={im2.getbands()}", metrics)

            if level >= 2:
                gray = _to_gray_uint8(im2)
                lv = _laplacian_var(gray)
                metrics["lap_var"] = float(lv)
                if lv < blur_thr:
                    return ScanResult(False, f"blurry lap_var={lv:.2f} (<{blur_thr})", metrics)

                dark = float((gray <= 10).mean())
                bright = float((gray >= 245).mean())
                std = float(gray.std())
                metrics["dark_pct"] = float(dark)
                metrics["bright_pct"] = float(bright)
                metrics["std"] = float(std)

                if dark > dark_pct_thr:
                    return ScanResult(False, f"too_dark dark_pct={dark:.3f} (>{dark_pct_thr})", metrics)
                if bright > bright_pct_thr:
                    return ScanResult(False, f"too_bright bright_pct={bright:.3f} (>{bright_pct_thr})", metrics)
                if std < low_contrast_thr:
                    return ScanResult(False, f"low_contrast std={std:.2f} (<{low_contrast_thr})", metrics)

            if level >= 3:
                with Image.open(img_path) as im3:
                    im3.load()
                    preprocess_dry_run(
                        im3,
                        input_size=input_size,
                        resize_shorter_to=resize_shorter_to,
                        crop=crop,
                        require_rgb=require_rgb,
                    )

        return ScanResult(True, None, metrics)

    except Exception as e:
        return ScanResult(False, f"{type(e).__name__}: {e}", metrics)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_carousel_html(items: List[dict], out_html: Path) -> None:
    ensure_parent_dir(out_html)
    data_json = json.dumps(items)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Mapillary Review — Carousel</title>
<style>
  body{{margin:0;background:#0b0f14;color:#e6edf3;font-family:-apple-system,system-ui,Segoe UI,Roboto,Arial}}
  header{{position:sticky;top:0;background:#0b0f14cc;backdrop-filter:blur(8px);border-bottom:1px solid #1f2a37;padding:12px 16px;z-index:10}}
  .row{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
  button{{background:#111827;color:#e6edf3;border:1px solid #243244;border-radius:12px;padding:10px 12px;cursor:pointer}}
  button.keep{{border-color:#14532d}}
  button.skip{{border-color:#334155}}
  button.delete{{border-color:#7f1d1d}}
  .pill{{display:inline-block;font-size:12px;padding:4px 10px;border-radius:999px;border:1px solid #334155;color:#cbd5e1}}
  .meta{{padding:10px 16px;color:#9ca3af;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:70vw}}
  .wrap{{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;padding:16px}}
  .frame{{width:min(1100px, 96vw);background:#060a10;border:1px solid #1f2a37;border-radius:18px;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.55)}}
  .imgbox{{display:flex;align-items:center;justify-content:center;min-height:68vh;padding:10px}}
  img{{max-width:100%;max-height:68vh;object-fit:contain;border-radius:12px}}
  .bottom{{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:12px 14px;border-top:1px solid #1f2a37;flex-wrap:wrap}}
  .small{{color:#9ca3af;font-size:12px}}
</style>
</head>
<body>
<header>
  <div class="row">
    <div style="font-weight:900">Mapillary Review — Carousel</div>
    <span class="pill" id="pos"></span>
    <span class="pill" id="decided"></span>
    <button id="prev">← Prev</button>
    <button id="next">Next →</button>
    <button id="export">Export actions.jsonl</button>
    <button id="clear">Clear local</button>
    <span class="small">Keys: 1=KEEP · 2=SKIP · 3=DELETE · ←/→ navigate</span>
  </div>
</header>

<div class="wrap">
  <div class="frame">
    <div class="imgbox">
      <img id="img" alt="preview"/>
    </div>
    <div class="bottom">
      <div>
        <button class="keep" id="keep">KEEP</button>
        <button class="skip" id="skip">SKIP</button>
        <button class="delete" id="delete">DELETE</button>
      </div>
      <div class="meta" id="meta"></div>
    </div>
  </div>
</div>

<script>
const ITEMS = {data_json};
const KEY = "scan_mapillary_carousel_v1"; // id -> action
let decisions = {{}};
try {{
  decisions = JSON.parse(localStorage.getItem(KEY) || "{{}}");
}} catch (e) {{
  decisions = {{}};
}}

let idx = 0;

function persist() {{
  localStorage.setItem(KEY, JSON.stringify(decisions));
}}

function decidedCount() {{
  return Object.keys(decisions).length;
}}

function setIdx(i) {{
  if (!ITEMS.length) return;
  idx = Math.max(0, Math.min(ITEMS.length - 1, i));
  render();
}}

function mark(action) {{
  const it = ITEMS[idx];
  decisions[it.id] = action; // KEEP/SKIP/DELETE
  persist();
  // advance
  if (idx < ITEMS.length - 1) idx += 1;
  render();
}}

function exportActions() {{
  const lines = [];
  for (const it of ITEMS) {{
    const action = decisions[it.id];
    if (!action) continue;
    lines.push(JSON.stringify({{
      id: it.id,
      action,
      reason: it.reason || "",
      path: it.path || "",
      metrics: it.metrics || {{}}
    }}));
  }}
  const blob = new Blob([lines.join("\\n") + (lines.length ? "\\n" : "")], {{type:'text/plain'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'actions.jsonl';
  a.click();
  URL.revokeObjectURL(url);
}}

function clearLocal() {{
  if (!confirm("Clear saved decisions in this browser?")) return;
  decisions = {{}};
  persist();
  render();
}}

function render() {{
  const it = ITEMS[idx];
  document.getElementById('img').src = it.thumb || "";
  document.getElementById('pos').textContent = (idx + 1) + " / " + ITEMS.length;
  document.getElementById('decided').textContent = decidedCount() + " decided";
  const action = decisions[it.id] || "undecided";
  document.getElementById('meta').textContent =
    it.id + " · " + action + " · " + it.reason + " · " + it.path;

  // disable prev/next
  document.getElementById('prev').disabled = (idx === 0);
  document.getElementById('next').disabled = (idx === ITEMS.length - 1);
}}

document.getElementById('keep').onclick = () => mark("KEEP");
document.getElementById('skip').onclick = () => mark("SKIP");
document.getElementById('delete').onclick = () => mark("DELETE");

document.getElementById('prev').onclick = () => setIdx(idx - 1);
document.getElementById('next').onclick = () => setIdx(idx + 1);

document.getElementById('export').onclick = exportActions;
document.getElementById('clear').onclick = clearLocal;

window.addEventListener('keydown', (e) => {{
  if (e.key === "ArrowLeft") setIdx(idx - 1);
  if (e.key === "ArrowRight") setIdx(idx + 1);
  if (e.key === "1") mark("KEEP");
  if (e.key === "2") mark("SKIP");
  if (e.key === "3") mark("DELETE");
}});

if (!ITEMS.length) {{
  document.getElementById('meta').textContent = "No problematic items found.";
}} else {{
  render();
}}
</script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def _normalize_action(a: str) -> str:
    if not a:
        return ""
    x = str(a).strip().lower()
    if x in {"keep", "send"}:
        return "KEEP"
    if x in {"skip", "k"}:
        return "SKIP"
    if x in {"delete", "d", "del", "rm", "remove"}:
        return "DELETE"
    return ""


def load_actions(actions_path: Path) -> List[dict]:
    actions: List[dict] = []
    with actions_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            actions.append(json.loads(s))
    return actions


def apply_actions(
    actions: List[dict],
    *,
    raw_root: Path,
    processed_root: Path,
    index_path: Path,
    dry_run: bool,
) -> None:
    id_to_path: Dict[str, Path] = {extract_id_from_path(p): p for p in scan_images(raw_root)}

    keep_n = skip_n = delete_n = 0
    missing = unknown = 0
    index_removed_total = 0

    for row in actions:
        image_id = row.get("id")
        if not isinstance(image_id, str) or not image_id:
            print(f"[warn] bad row (missing id): {row}", file=sys.stderr)
            continue

        act = _normalize_action(row.get("action", ""))
        if not act:
            unknown += 1
            print(f"[warn] unknown action for id={image_id}: {row.get('action')!r}", file=sys.stderr)
            continue

        img_path = id_to_path.get(image_id)
        if img_path is None:
            missing += 1
            print(f"[warn] id not found in raw dir: {image_id}", file=sys.stderr)
            continue

        if act == "SKIP":
            skip_n += 1
            continue

        if act == "KEEP":
            if dry_run:
                print(f"DRY-RUN KEEP: would copy -> processed: {img_path}")
            else:
                dst = copy_to_processed(img_path, raw_root, processed_root)
                print(f"KEEP: copied to {dst}")
            keep_n += 1
            continue

        if act == "DELETE":
            if dry_run:
                print(f"DRY-RUN DELETE: would delete file: {img_path}")
                print(f"DRY-RUN DELETE: would remove from index: {index_path} (id={image_id})")
            else:
                try:
                    img_path.unlink()
                    print(f"DELETE: removed {img_path}")
                except FileNotFoundError:
                    print(f"[warn] already missing on disk: {img_path}", file=sys.stderr)

                removed, kept = remove_index_lines(index_path, image_id, raw_path=img_path)
                index_removed_total += removed
                print(f"DELETE: index updated (removed {removed} lines, kept {kept}) for id={image_id}")
            delete_n += 1

    print("\n" + "=" * 80)
    print("APPLY ACTIONS SUMMARY")
    print("=" * 80)
    print(f"KEEP   : {keep_n}")
    print(f"SKIP   : {skip_n}")
    print(f"DELETE : {delete_n}")
    print(f"Missing IDs    : {missing}")
    print(f"Unknown action : {unknown}")
    if not dry_run:
        print(f"Index removed  : {index_removed_total}")
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan Mapillary and generate a carousel reviewer (KEEP/SKIP/DELETE). Optionally apply actions."
    )
    parser.add_argument("--raw-dir", default="data/raw/mapillary")
    parser.add_argument("--processed-dir", default="data/processed/mapillary")
    parser.add_argument("--index-jsonl", default="data/index/images.jsonl")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--level", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--min-size", type=int, default=160)
    parser.add_argument("--require-rgb", action="store_true")
    parser.add_argument("--max-aspect", type=float, default=3.0)

    parser.add_argument("--blur-thr", type=float, default=140.0)
    parser.add_argument("--dark-pct", type=float, default=0.20)
    parser.add_argument("--bright-pct", type=float, default=0.20)
    parser.add_argument("--low-contrast", type=float, default=25.0)

    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--resize-shorter-to", type=int, default=160)
    parser.add_argument("--crop", choices=["center", "random"], default="center")

    parser.add_argument("--perf", action="store_true")

    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--thumb-side", type=int, default=900, help="Bigger thumb for carousel (heavier HTML)")

    parser.add_argument("--apply-actions", default="", help="Path to actions.jsonl exported from carousel")

    args = parser.parse_args()

    raw_root = Path(args.raw_dir).resolve()
    processed_root = Path(args.processed_dir).resolve()
    index_path = Path(args.index_jsonl).resolve()
    report_dir = Path(args.report_dir).resolve()

    if args.apply_actions:
        actions_path = Path(args.apply_actions).resolve()
        actions = load_actions(actions_path)
        apply_actions(actions, raw_root=raw_root, processed_root=processed_root, index_path=index_path, dry_run=args.dry_run)
        return 0

    if not raw_root.exists():
        print(f"ERROR: raw dir not found: {raw_root}", file=sys.stderr)
        return 2

    print(f"Scanning   : {raw_root}")
    print(f"Processed  : {processed_root}")
    print(f"Index JSONL: {index_path}")
    print(f"Mode       : {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print(f"Level      : {args.level}")
    print(f"Model      : input_size={args.input_size}")
    print(f"Report dir : {report_dir}")
    if args.html:
        print("HTML       : enabled (carousel reviewer)")

    total = ok_count = bad_count = 0
    problems: List[dict] = []

    t0 = time.perf_counter()
    dt_total = 0.0

    for img_path in scan_images(raw_root):
        total += 1
        t_img0 = time.perf_counter()

        res = try_open_image(
            img_path,
            level=args.level,
            min_size=args.min_size,
            require_rgb=bool(args.require_rgb),
            max_aspect=args.max_aspect,
            blur_thr=args.blur_thr,
            dark_pct_thr=args.dark_pct,
            bright_pct_thr=args.bright_pct,
            low_contrast_thr=args.low_contrast,
            input_size=args.input_size,
            resize_shorter_to=args.resize_shorter_to,
            crop=args.crop,
        )

        dt_total += (time.perf_counter() - t_img0)

        if res.ok:
            ok_count += 1
            continue

        bad_count += 1
        image_id = extract_id_from_path(img_path)
        reason = res.reason or "unknown_error"

        item = {"id": image_id, "path": str(img_path), "reason": reason, "metrics": res.metrics or {}}
        if args.html:
            item["thumb"] = make_thumb_data_uri(img_path, max_side=args.thumb_side)
        problems.append(item)

        if args.limit and bad_count >= args.limit:
            print(f"Limit reached ({args.limit} problematic images).")
            break

    wall = time.perf_counter() - t0

    if args.html:
        out_jsonl = report_dir / "scan_mapillary.jsonl"
        out_html = report_dir / "scan_mapillary_carousel.html"
        write_jsonl(out_jsonl, problems)
        write_carousel_html(problems, out_html)
        print(f"\nReport JSONL: {out_jsonl}")
        print(f"Report HTML : {out_html}")
        print("Open with:  open reports/scan_mapillary_carousel.html")

    print("\n" + "=" * 80)
    print("SCAN SUMMARY")
    print("=" * 80)
    print(f"Total images scanned : {total}")
    print(f"OK                  : {ok_count}")
    print(f"Problematic         : {bad_count}")
    print("=" * 80)

    if args.perf and total > 0:
        ms_total = (dt_total / total) * 1000.0
        print("\nPERF")
        print("-" * 80)
        print(f"Avg ms/img (total)  : {ms_total:.2f}")
        print(f"Wall time (s)       : {wall:.2f}")
        print("-" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
