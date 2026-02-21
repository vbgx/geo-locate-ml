#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


PLOTS = [
    "metrics_loss.png",
    "metrics_valacc.png",
    "geo_error.png",
    "confusion_matrix.png",
]

TABLES = [
    "val_topk.parquet",
    "val_errors.parquet",
    "classes_far.parquet",
    "confusion_pairs_far.parquet",
]

DATA = [
    "dist_km.pt",
    "hardneg_pool.json",
]


def _safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _safe_move(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return True


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _metric_if_present(row: pd.Series, col: str) -> float:
    return _safe_float(row[col]) if col in row.index else float("nan")


def build_summary(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"

    cfg = _read_json(cfg_path) if cfg_path.exists() else {}
    df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    best = {}
    last = {}

    if not df.empty:
        # best by val_acc
        if "val_acc" in df.columns:
            best_acc_row = df.loc[df["val_acc"].astype(float).idxmax()]
        else:
            best_acc_row = df.iloc[-1]

        # best by geo p90
        if "geo_p90_km" in df.columns:
            best_p90_row = df.loc[df["geo_p90_km"].astype(float).idxmin()]
        else:
            best_p90_row = df.iloc[-1]

        last_row = df.iloc[-1]

        best = {
            "best_val_acc": _metric_if_present(best_acc_row, "val_acc"),
            "best_val_acc_epoch": int(_safe_float(best_acc_row.get("epoch", -1))),
            "best_val_acc_size": int(_safe_float(best_acc_row.get("image_size", -1))),
            "best_p90_km": _metric_if_present(best_p90_row, "geo_p90_km"),
            "best_p90_epoch": int(_safe_float(best_p90_row.get("epoch", -1))),
            "best_p90_size": int(_safe_float(best_p90_row.get("image_size", -1))),
        }

        last = {
            "last_epoch": int(_safe_float(last_row.get("epoch", len(df)))),
            "last_val_acc": _metric_if_present(last_row, "val_acc"),
            "last_geo_median_km": _metric_if_present(last_row, "geo_median_km"),
            "last_geo_p90_km": _metric_if_present(last_row, "geo_p90_km"),
            "last_geo_p95_km": _metric_if_present(last_row, "geo_p95_km"),
            "last_train_loss": _metric_if_present(last_row, "train_loss"),
        }

    summary = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "has_metrics": (run_dir / "metrics.csv").exists(),
        "has_val_topk": (run_dir / "artifacts" / "tables" / "val_topk.parquet").exists() or (run_dir / "val_topk.parquet").exists(),
        "has_val_errors": (run_dir / "artifacts" / "tables" / "val_errors.parquet").exists() or (run_dir / "val_errors.parquet").exists(),
        "config": {
            "lr": cfg.get("lr"),
            "dropout": cfg.get("dropout"),
            "weight_decay": cfg.get("weight_decay"),
            "geo_loss_enabled": bool(cfg.get("geo_loss_enabled", False)),
            "geo_tau_km": cfg.get("geo_tau_km"),
            "geo_mix_ce": cfg.get("geo_mix_ce"),
            "hardneg_enabled": bool(cfg.get("hardneg_enabled", False)),
            "hierarchical_enabled": bool(cfg.get("hierarchical_enabled", False)),
        },
        **best,
        **last,
    }
    return summary


def write_dashboard(run_dir: Path, summary: Dict[str, Any]) -> Path:
    """
    Lightweight single-page dashboard:
    - header + key metrics
    - links to artifacts
    """
    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    plots_dir = Path("artifacts/plots")
    tables_dir = Path("artifacts/tables")

    imgs = []
    for fn in PLOTS:
        p = run_dir / plots_dir / fn
        if p.exists():
            imgs.append(f'<div style="margin:12px 0;"><div><b>{esc(fn)}</b></div><img src="{esc(str(plots_dir / fn))}" style="max-width:100%;border:1px solid #ddd;"/></div>')

    links = []
    for fn in TABLES:
        p = run_dir / tables_dir / fn
        if p.exists():
            links.append(f'<li><a href="{esc(str(tables_dir / fn))}">{esc(fn)}</a></li>')

    ckpt_links = []
    for fn in ["checkpoints/best.pt", "checkpoints/last.pt", "config.json", "REPORT.md", "summary.json"]:
        p = run_dir / fn
        if p.exists():
            ckpt_links.append(f'<li><a href="{esc(fn)}">{esc(fn)}</a></li>')

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Run {esc(summary.get("run_id",""))} — geo-locate-ml</title>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 18px; }}
    code {{ background:#f2f2f2; padding:2px 4px; border-radius:4px; }}
    .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .card {{ border:1px solid #ddd; border-radius:10px; padding:12px; }}
    h1 {{ margin: 0 0 8px 0; }}
    ul {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <h1>Run <code>{esc(summary.get("run_id",""))}</code></h1>
  <div class="grid">
    <div class="card">
      <h3>Key metrics</h3>
      <ul>
        <li>best_val_acc: <code>{summary.get("best_val_acc")}</code> (epoch <code>{summary.get("best_val_acc_epoch")}</code>)</li>
        <li>best_p90_km: <code>{summary.get("best_p90_km")}</code> (epoch <code>{summary.get("best_p90_epoch")}</code>)</li>
        <li>last_val_acc: <code>{summary.get("last_val_acc")}</code></li>
        <li>last_geo_median_km: <code>{summary.get("last_geo_median_km")}</code></li>
      </ul>
    </div>
    <div class="card">
      <h3>Config (selected)</h3>
      <pre style="white-space:pre-wrap;margin:0;">{esc(json.dumps(summary.get("config",{}), indent=2))}</pre>
    </div>
  </div>

  <div class="card" style="margin-top:12px;">
    <h3>Files</h3>
    <div class="grid">
      <div>
        <b>Tables</b>
        <ul>{''.join(links) if links else '<li><i>none</i></li>'}</ul>
      </div>
      <div>
        <b>Other</b>
        <ul>{''.join(ckpt_links) if ckpt_links else '<li><i>none</i></li>'}</ul>
      </div>
    </div>
  </div>

  <h2 style="margin-top:18px;">Plots</h2>
  {''.join(imgs) if imgs else '<p><i>No plots found.</i></p>'}
</body>
</html>
"""
    out = run_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


def normalize_one_run(run_dir: Path, *, move: bool = False) -> None:
    """
    move=False => copy files into artifacts/ while keeping legacy files in place.
    move=True  => move files into artifacts/ (destructive but clean).
    """
    if not run_dir.exists():
        raise SystemExit(f"Missing run_dir: {run_dir}")

    (run_dir / "artifacts" / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts" / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts" / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    op = _safe_move if move else _safe_copy

    # plots at root -> artifacts/plots
    for fn in PLOTS:
        src = run_dir / fn
        dst = run_dir / "artifacts" / "plots" / fn
        op(src, dst)

    # tables at root
    for fn in ["val_topk.parquet", "val_errors.parquet"]:
        src = run_dir / fn
        dst = run_dir / "artifacts" / "tables" / fn
        op(src, dst)

    # tail_analysis -> artifacts/tables
    tail = run_dir / "tail_analysis"
    if tail.exists() and tail.is_dir():
        for fn in ["classes_far.parquet", "confusion_pairs_far.parquet"]:
            src = tail / fn
            dst = run_dir / "artifacts" / "tables" / fn
            op(src, dst)

    # data
    for fn in ["dist_km.pt", "hardneg_pool.json"]:
        src = run_dir / fn
        dst = run_dir / "artifacts" / "data" / fn
        op(src, dst)

    # checkpoints already under checkpoints/ in your tree -> keep as-is (but ensure present)
    # If some runs have best.pt/last.pt at root (not expected), we also capture them:
    op(run_dir / "best.pt", run_dir / "checkpoints" / "best.pt")
    op(run_dir / "last.pt", run_dir / "checkpoints" / "last.pt")

    # summary + dashboard
    summary = build_summary(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    dash = write_dashboard(run_dir, summary)

    print(f"✅ normalized: {run_dir.name}")
    print(f"  - summary: {run_dir/'summary.json'}")
    print(f"  - dashboard: {dash}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize runs/<run_id>/ into a stable artifacts layout + summary + dashboard.")
    ap.add_argument("--runs-dir", default="runs", help="Runs directory (default: runs)")
    ap.add_argument("--run-id", default=None, help="Normalize one run id only (e.g. 2026-02-21_03-36-36)")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying (cleaner but destructive).")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise SystemExit(f"Missing runs dir: {runs_dir}")

    if args.run_id:
        run_dir = runs_dir / str(args.run_id)
        normalize_one_run(run_dir, move=bool(args.move))
        return

    # normalize all real run dirs
    for p in sorted(runs_dir.iterdir()):
        if p.name in {"latest", "template_run", "_aggregate"} or p.is_symlink():
            continue
        if not p.is_dir():
            continue
        if not (p / "metrics.csv").exists() or not (p / "config.json").exists():
            continue
        normalize_one_run(p, move=bool(args.move))


if __name__ == "__main__":
    main()
