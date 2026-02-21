#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed/mapillary")


def resolve_image(image_id: str) -> Path:
    # single source of truth: processed images
    return PROCESSED_DIR / f"{image_id}.jpg"


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect tail errors (top pct) by copying images + exporting a report.")
    ap.add_argument("--run-dir", required=True, help="Path to a run directory containing val_errors.parquet")
    ap.add_argument("--out", default=None, help="Output directory (default: <run-dir>/inspect_top_errors)")
    ap.add_argument("--top-pct", type=float, default=1.0, help="Take top X%% by dist_km (default: 1%%)")
    ap.add_argument("--take", type=int, default=100, help="Copy up to N images (default: 100)")
    ap.add_argument("--min-km", type=float, default=None, help="Optional: only keep errors >= this km")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    valp = run_dir / "val_errors.parquet"
    if not valp.exists():
        raise SystemExit(f"Missing {valp}. Run training with the updated train.py to generate it.")

    out = Path(args.out) if args.out else (run_dir / "inspect_top_errors")
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(valp)
    if "dist_km" not in df.columns or "image_id" not in df.columns:
        raise SystemExit("val_errors.parquet missing required columns: image_id, dist_km")

    # sort by distance descending (worst first)
    df = df.sort_values("dist_km", ascending=False).reset_index(drop=True)

    if args.min_km is not None:
        df = df[df["dist_km"] >= float(args.min_km)].reset_index(drop=True)

    n = len(df)
    if n == 0:
        raise SystemExit("No rows to inspect after filtering.")

    cut = max(1, int(n * (float(args.top_pct) / 100.0)))
    top = df.head(cut).copy()

    # take first N for copy/inspection
    top = top.head(int(args.take)).copy()

    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    copied_rows = []
    copied = 0

    for i, r in top.iterrows():
        image_id = str(r["image_id"])
        dist_km = float(r["dist_km"])
        src = resolve_image(image_id)

        # filename is sortable, includes distance
        dst = img_dir / f"{copied:03d}__{image_id}__{dist_km:.1f}km.jpg"

        if src.exists():
            shutil.copy2(src, dst)
            rr = r.to_dict()
            rr["copied_path"] = str(dst)
            copied_rows.append(rr)
            copied += 1
        else:
            rr = r.to_dict()
            rr["copied_path"] = ""
            rr["missing_file"] = True
            copied_rows.append(rr)

        if copied >= int(args.take):
            break

    rep = pd.DataFrame(copied_rows)

    # Write report CSV
    rep_path = out / "report.csv"
    rep.to_csv(rep_path, index=False)

    # Write a tiny HTML index for fast eyeballing
    html_path = out / "index.html"
    rows_html = []
    for _, r in rep.iterrows():
        image_id = str(r.get("image_id", ""))
        dist_km = float(r.get("dist_km", 0.0))
        rel = ""
        if r.get("copied_path"):
            rel = f"images/{Path(str(r['copied_path'])).name}"
            img_tag = f'<img src="{rel}" style="max-width: 520px; height: auto; border-radius: 8px;" />'
        else:
            img_tag = "<div style='color:#b00;'>missing image file</div>"

        rows_html.append(
            f"""
            <div style="display:flex; gap:16px; padding:12px; border:1px solid #eee; border-radius:12px; margin:10px 0;">
              <div>{img_tag}</div>
              <div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">
                <div><b>{image_id}</b></div>
                <div>dist_km: <b>{dist_km:.2f}</b></div>
                <div>true_idx: {r.get('true_idx','')}</div>
                <div>pred_idx: {r.get('pred_idx','')}</div>
                <div>lat_true: {r.get('lat_true','')}</div>
                <div>lon_true: {r.get('lon_true','')}</div>
              </div>
            </div>
            """
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Top errors inspection</title>
</head>
<body style="max-width: 1100px; margin: 24px auto; padding: 0 16px;">
  <h1 style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;">
    Top errors inspection
  </h1>
  <p style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; color:#444;">
    Source: {valp}<br/>
    Filter: top {float(args.top_pct):g}% {("and dist>="+str(args.min_km)) if args.min_km is not None else ""}<br/>
    Copied: {copied} images
  </p>
  {''.join(rows_html)}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")

    print("Wrote:")
    print(f" - {out}")
    print(f" - {rep_path}")
    print(f" - {html_path}")
    print(f"Copied images: {copied} (requested {int(args.take)})")


if __name__ == "__main__":
    main()
