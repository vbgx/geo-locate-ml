#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        v = float(x)
        if math.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        v = int(float(x))
        return v
    except Exception:
        return int(default)


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _is_real_run_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    if p.is_symlink():
        return False
    if p.name in {"latest", "template_run", "_aggregate"}:
        return False
    # v2 contract: summary.json present
    return (p / "summary.json").exists() and (p / "dashboard.html").exists()


# -----------------------------------------------------------------------------
# Scoring (multi-criteria)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreWeights:
    # Normalize each metric into [0,1] (higher is better) then combine.
    w_acc: float = 0.55       # best_val_acc (higher better)
    w_p90: float = 0.35       # best_p90_km (lower better)
    w_median: float = 0.10    # last_geo_median_km (lower better)

    # Penalties (subtracted from final score)
    penalty_missing_val_errors: float = 0.03
    penalty_missing_val_topk: float = 0.01


def _minmax_norm_high(x: np.ndarray) -> np.ndarray:
    """Higher is better -> map to [0,1]."""
    a = np.asarray(x, dtype=np.float64)
    m = np.nanmin(a)
    M = np.nanmax(a)
    if not np.isfinite(m) or not np.isfinite(M) or abs(M - m) < 1e-12:
        return np.where(np.isfinite(a), 1.0, np.nan)
    return (a - m) / (M - m)


def _minmax_norm_low(x: np.ndarray) -> np.ndarray:
    """Lower is better -> map to [0,1]."""
    a = np.asarray(x, dtype=np.float64)
    m = np.nanmin(a)
    M = np.nanmax(a)
    if not np.isfinite(m) or not np.isfinite(M) or abs(M - m) < 1e-12:
        return np.where(np.isfinite(a), 1.0, np.nan)
    return (M - a) / (M - m)


def compute_composite_scores(df: pd.DataFrame, w: ScoreWeights) -> pd.DataFrame:
    out = df.copy()

    acc = out["best_val_acc"].to_numpy(dtype=np.float64) if "best_val_acc" in out.columns else np.full((len(out),), np.nan)
    p90 = out["best_p90_km"].to_numpy(dtype=np.float64) if "best_p90_km" in out.columns else np.full((len(out),), np.nan)
    med = out["last_geo_median_km"].to_numpy(dtype=np.float64) if "last_geo_median_km" in out.columns else np.full((len(out),), np.nan)

    acc_n = _minmax_norm_high(acc)
    p90_n = _minmax_norm_low(p90)
    med_n = _minmax_norm_low(med)

    score = (
        float(w.w_acc) * np.nan_to_num(acc_n, nan=0.0)
        + float(w.w_p90) * np.nan_to_num(p90_n, nan=0.0)
        + float(w.w_median) * np.nan_to_num(med_n, nan=0.0)
    )

    # penalties (prefer runs with richer artifacts)
    has_val_errors = out.get("has_val_errors", False)
    has_val_topk = out.get("has_val_topk", False)

    if isinstance(has_val_errors, pd.Series):
        score = score - (~has_val_errors.astype(bool)).astype(np.float64) * float(w.penalty_missing_val_errors)
    if isinstance(has_val_topk, pd.Series):
        score = score - (~has_val_topk.astype(bool)).astype(np.float64) * float(w.penalty_missing_val_topk)

    out["score_acc_norm"] = acc_n
    out["score_p90_norm"] = p90_n
    out["score_median_norm"] = med_n
    out["score_composite"] = score

    return out


# -----------------------------------------------------------------------------
# Dashboard + plots
# -----------------------------------------------------------------------------

def _plot_bar(df: pd.DataFrame, x: str, y: str, out_png: Path, title: str, y_label: str) -> Optional[Path]:
    if df.empty or y not in df.columns or x not in df.columns:
        return None
    dd = df.copy()
    dd = dd.sort_values(y, ascending=False, na_position="last")
    vals = dd[y].to_numpy(dtype=np.float64)
    if not np.isfinite(vals).any():
        return None

    plt.figure(figsize=(10, max(3.5, 0.28 * len(dd))))
    plt.barh(dd[x].astype(str).tolist()[::-1], vals[::-1])
    plt.xlabel(y_label)
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()
    return out_png


def write_dashboard(out_dir: Path, df: pd.DataFrame, plots_rel: Dict[str, str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "dashboard.html"

    # clickable run links
    view = df.copy()

    def run_link(run_id: str) -> str:
        return f'<a href="../{_html_escape(run_id)}/dashboard.html">{_html_escape(run_id)}</a>'

    view.insert(0, "run", view["run_id"].astype(str).map(run_link))

    # compact columns
    cols = [
        "run",
        "score_composite",
        "best_val_acc",
        "best_p90_km",
        "last_geo_median_km",
        "best_val_acc_epoch",
        "best_p90_epoch",
        "has_val_errors",
        "has_val_topk",
        "config.lr",
        "config.dropout",
        "config.geo_loss_enabled",
        "config.geo_tau_km",
        "config.geo_mix_ce",
        "config.hardneg_enabled",
        "config.hierarchical_enabled",
    ]
    cols = [c for c in cols if c in view.columns]

    table_html = view[cols].to_html(index=False, escape=False)

    plot_html = ""
    for k, rel in plots_rel.items():
        plot_html += (
            f'<div style="margin:12px 0;">'
            f'<div><b>{_html_escape(k)}</b></div>'
            f'<img src="{_html_escape(rel)}" style="max-width:100%; height:auto; border:1px solid #ddd;"/>'
            f"</div>\n"
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>geo-locate-ml — Runs aggregate</title>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 18px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f6f6f6; position: sticky; top: 0; }}
    code {{ background:#f2f2f2; padding:2px 4px; border-radius:4px; }}
  </style>
</head>
<body>
  <h1>geo-locate-ml — Runs aggregate</h1>
  <p>Generated in <code>{_html_escape(str(out_dir))}</code></p>

  <h2>Ranking (multi-criteria)</h2>
  <p>
    Score = acc (higher) + p90 (lower) + median (lower),
    with small penalties when val_errors/val_topk are missing.
  </p>

  <h2>Overview plots</h2>
  {plot_html if plot_html else "<p><i>No plots.</i></p>"}

  <h2>Runs table</h2>
  {table_html}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate runs by reading summary.json, build multi-criteria ranking + dashboard.")
    ap.add_argument("--runs-dir", default="runs", help="Runs directory (default: runs)")
    ap.add_argument("--out-dir", default="runs/_aggregate", help="Output directory (default: runs/_aggregate)")

    # scoring weights
    ap.add_argument("--w-acc", type=float, default=0.55)
    ap.add_argument("--w-p90", type=float, default=0.35)
    ap.add_argument("--w-median", type=float, default=0.10)
    ap.add_argument("--penalty-missing-val-errors", type=float, default=0.03)
    ap.add_argument("--penalty-missing-val-topk", type=float, default=0.01)

    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    if not runs_dir.exists():
        raise SystemExit(f"Missing runs dir: {runs_dir}")

    rows: List[Dict[str, Any]] = []
    for p in sorted(runs_dir.iterdir()):
        if not _is_real_run_dir(p):
            continue
        try:
            s = _read_json(p / "summary.json")
        except Exception as e:
            print(f"WARN: skip {p.name}: cannot read summary.json: {e}")
            continue

        # flatten into stable schema (explicit, no surprises)
        cfg = s.get("config") or {}
        rows.append(
            {
                "run_id": str(s.get("run_id") or p.name),
                "run_dir": str(s.get("run_dir") or str(p)),
                "generated_at": str(s.get("generated_at") or ""),
                "has_metrics": bool(s.get("has_metrics", False)),
                "has_val_topk": bool(s.get("has_val_topk", False)),
                "has_val_errors": bool(s.get("has_val_errors", False)),

                "best_val_acc": _safe_float(s.get("best_val_acc")),
                "best_val_acc_epoch": _safe_int(s.get("best_val_acc_epoch")),
                "best_p90_km": _safe_float(s.get("best_p90_km")),
                "best_p90_epoch": _safe_int(s.get("best_p90_epoch")),

                "last_val_acc": _safe_float(s.get("last_val_acc")),
                "last_geo_median_km": _safe_float(s.get("last_geo_median_km")),

                # config (selected)
                "config.lr": _safe_float(cfg.get("lr")),
                "config.dropout": _safe_float(cfg.get("dropout")),
                "config.weight_decay": _safe_float(cfg.get("weight_decay")),
                "config.geo_loss_enabled": bool(cfg.get("geo_loss_enabled", False)),
                "config.geo_tau_km": _safe_float(cfg.get("geo_tau_km")),
                "config.geo_mix_ce": _safe_float(cfg.get("geo_mix_ce")),
                "config.hardneg_enabled": bool(cfg.get("hardneg_enabled", False)),
                "config.hierarchical_enabled": bool(cfg.get("hierarchical_enabled", False)),
            }
        )

    if not rows:
        raise SystemExit("No normalized runs found (need runs/<run_id>/summary.json + dashboard.html).")

    df = pd.DataFrame(rows)

    w = ScoreWeights(
        w_acc=float(args.w_acc),
        w_p90=float(args.w_p90),
        w_median=float(args.w_median),
        penalty_missing_val_errors=float(args.penalty_missing_val_errors),
        penalty_missing_val_topk=float(args.penalty_missing_val_topk),
    )
    df = compute_composite_scores(df, w=w)

    # rank
    df = df.sort_values(
        by=["score_composite", "best_val_acc", "best_p90_km"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1, dtype=np.int64))

    out_dir.mkdir(parents=True, exist_ok=True)

    # write tables
    csv_path = out_dir / "runs_summary.csv"
    pq_path = out_dir / "runs_summary.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)

    # plots
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plots_rel: Dict[str, str] = {}

    p1 = _plot_bar(
        df,
        x="run_id",
        y="score_composite",
        out_png=plots_dir / "score_composite.png",
        title="Composite score per run (higher is better)",
        y_label="score_composite",
    )
    if p1 and p1.exists():
        plots_rel["score_composite"] = str(Path("plots") / p1.name)

    p2 = _plot_bar(
        df,
        x="run_id",
        y="best_val_acc",
        out_png=plots_dir / "best_val_acc.png",
        title="Best val_acc per run",
        y_label="best_val_acc",
    )
    if p2 and p2.exists():
        plots_rel["best_val_acc"] = str(Path("plots") / p2.name)

    p3 = _plot_bar(
        df,
        x="run_id",
        y="best_p90_km",
        out_png=plots_dir / "best_p90_km.png",
        title="Best p90_km per run (lower is better) — plotted as higher is better ranking via sorting",
        y_label="best_p90_km (lower better)",
    )
    if p3 and p3.exists():
        plots_rel["best_p90_km"] = str(Path("plots") / p3.name)

    dash = write_dashboard(out_dir, df, plots_rel)

    print("✅ Aggregated runs:", len(df))
    print("Summary CSV:", csv_path)
    print("Summary Parquet:", pq_path)
    print("Dashboard:", dash)


if __name__ == "__main__":
    main()
