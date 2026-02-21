from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import h3
import pandas as pd
import torch

from .config import TrainConfig
from .paths import p, ensure_dir, update_latest_symlink
from .labels import LabelSpace, build_label_space, compute_h3
from .splits import assign_split_by_sequence
from .train import run_training
from .plots import plot_metrics_csv
from .analysis import confusion_matrix_png, geo_error_plot_png
from .geo import haversine_km


def compute_distance_matrix_km(labels) -> torch.Tensor:
    """
    Build a (C,C) distance matrix between class centroids in kilometers.
    """
    C = len(labels.h3_ids)
    D = torch.zeros((C, C), dtype=torch.float32)
    for i in range(C):
        lat1, lon1 = labels.idx_to_centroid[i]
        for j in range(C):
            lat2, lon2 = labels.idx_to_centroid[j]
            D[i, j] = float(haversine_km(lat1, lon1, lat2, lon2))
    return D


def maybe_update_global_best(
    models_dir: Path,
    best_run_ckpt: str,
    best_val_acc: float,
    meta: dict,
) -> bool:
    best_pt = models_dir / "best.pt"
    best_json = models_dir / "best.json"

    prev = -1.0
    if best_json.exists():
        try:
            prev = float(json.loads(best_json.read_text(encoding="utf-8")).get("best_val_acc", -1.0))
        except Exception:
            prev = -1.0

    if best_val_acc > prev:
        shutil.copy2(best_run_ckpt, best_pt)
        best_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"✅ New GLOBAL BEST: {best_val_acc:.4f} (updated models/best.pt)")
        return True

    print(f"🧹 Not better than global best ({prev:.4f}). Global best unchanged.")
    return False


def write_report(run_dir: Path, cfg: TrainConfig, summary, artifacts: dict) -> Path:
    rp = run_dir / "REPORT.md"
    lines: list[str] = []
    lines.append("# geo-locate-ml — Run report\n\n")
    lines.append(f"- Timestamp: `{artifacts['timestamp']}`\n")
    lines.append(f"- Device: `{summary.device}`\n")
    lines.append(f"- Best val acc: `{summary.best_val_acc:.4f}`\n")
    lines.append(f"- Best epoch: `{summary.best_epoch}`\n")
    lines.append(f"- Best image_size: `{summary.best_image_size}`\n")
    lines.append(f"- Best geo median (val): `{summary.best_median_km:.2f} km`\n")
    lines.append(f"- Best geo p90 (val): `{summary.best_p90_km:.2f} km`\n")
    lines.append(f"- Num classes: `{artifacts['num_classes']}`\n")
    lines.append(f"- H3 resolution: `{cfg.h3_resolution}`\n")
    lines.append(f"- Min cell samples: `{cfg.min_cell_samples}`\n")
    lines.append(f"- Geo loss enabled: `{cfg.geo_loss_enabled}`\n")
    if cfg.geo_loss_enabled:
        lines.append(f"- Geo tau (km): `{cfg.geo_tau_km}`\n")
        lines.append(f"- Geo mix CE: `{cfg.geo_mix_ce}`\n")
    lines.append(
        f"- Early stopping: `{cfg.early_stopping_enabled}` "
        f"({cfg.early_stop_metric}/{cfg.early_stop_mode}, patience={cfg.early_stop_patience})\n"
    )

    lines.append("\n## Artifacts\n")
    for k, v in artifacts.items():
        if k == "timestamp":
            continue
        lines.append(f"- {k}: `{v}`\n")

    lines.append("\n## Config\n```json\n")
    lines.append(json.dumps(cfg.to_dict(), indent=2))
    lines.append("\n```\n")

    rp.write_text("".join(lines), encoding="utf-8")
    return rp


def _max_label_idx(df: pd.DataFrame) -> int:
    if "label_idx" not in df.columns or df.empty:
        return -1
    return int(df["label_idx"].max())


def _num_classes_from_df(df: pd.DataFrame) -> int:
    if "label_idx" not in df.columns or df.empty:
        return 0
    return int(df["label_idx"].nunique())


def _ensure_geo_distance_matrix(
    cfg: TrainConfig,
    run_dir: Path,
    labels,
) -> torch.Tensor:
    """
    Always build a fresh dist_km.pt for THIS run, aligned to current labels.
    Also: protect against any mismatch (defensive programming).
    """
    num_classes = len(labels.h3_ids)
    D = compute_distance_matrix_km(labels)

    if D.shape[0] != num_classes or D.shape[1] != num_classes:
        raise RuntimeError(
            f"Distance matrix shape mismatch: D={tuple(D.shape)} expected=({num_classes},{num_classes})."
        )

    dist_path = run_dir / "dist_km.pt"
    torch.save(D, dist_path)
    return D


def main() -> None:
    cfg = TrainConfig()

    runs_dir = p("runs")
    models_dir = p("models")
    ensure_dir(runs_dir)
    ensure_dir(models_dir)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs_dir / ts
    ensure_dir(run_dir)

    jsonl_path = p(*cfg.jsonl_index.split("/"))
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing {jsonl_path}. Run tools/mapillary_download.py first.")

    # Load jsonl -> df
    df = pd.read_json(jsonl_path, lines=True)

    # Required columns for building label space
    need = {"id", "lat", "lon", "path"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in jsonl: {missing}. Fix your downloader/indexer.")

    df = df.dropna(subset=["id", "lat", "lon", "path"]).reset_index(drop=True)

    # normalize optional fields
    if "sequence_id" not in df.columns:
        df["sequence_id"] = None

    # H3 + filtering
    df = compute_h3(df, cfg.h3_resolution)
    df = filter_sparse_cells(df, cfg.min_cell_samples)

    if len(df) < 100:
        raise RuntimeError("Too few samples after filtering. Lower min_cell_samples or download more images.")

    # Splits (sequence-aware)
    df = assign_split_by_sequence(
        df,
        seed=cfg.seed,
        p_train=cfg.split_train,
        p_val=cfg.split_val,
        p_test=cfg.split_test,
    )

    # Label space
    labels = build_label_space(df)
    df["label_idx"] = df["h3_id"].map(labels.h3_to_idx).astype(int)

    # Defensive: ensure label_idx covers 0..C-1
    C = len(labels.h3_ids)
    mx = _max_label_idx(df)
    if mx >= C:
        raise RuntimeError(f"label_idx max={mx} but num_classes={C}. Label mapping is inconsistent.")

    # Write parquet (canonical) + run parquet snapshot
    parquet_path = p(*cfg.parquet_index.split("/"))
    ensure_dir(parquet_path.parent)
    df.to_parquet(parquet_path, index=False)

    run_parquet = run_dir / "images.parquet"
    df.to_parquet(run_parquet, index=False)

    # Write labels + config
    labels_path = run_dir / "labels.json"
    labels_path.write_text(labels.to_json(), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    # Geo distance matrix aligned to this run's labels (fixes the crash you hit)
    dist_km: torch.Tensor | None = None
    if cfg.geo_loss_enabled:
        dist_km = _ensure_geo_distance_matrix(cfg, run_dir, labels)

    # Train
    summary = run_training(
        cfg,
        parquet_path=str(parquet_path),
        num_classes=C,
        run_dir=run_dir,
        idx_to_centroid=labels.idx_to_centroid,
        distance_km=dist_km if cfg.geo_loss_enabled else None,
    )

    # Plots from metrics.csv
    plots = plot_metrics_csv(summary.metrics_csv, str(run_dir / "metrics.png"))

    # Diagnostics
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diag_size = max(cfg.image_sizes)

    cm_path = run_dir / "confusion_matrix.png"
    geo_path = run_dir / "geo_error.png"

    if cfg.make_confusion_matrix:
        confusion_matrix_png(
            parquet_path=str(parquet_path),
            ckpt_path=summary.best_ckpt,
            labels=labels,
            image_size=diag_size,
            out_path=str(cm_path),
            dropout=cfg.dropout,
            device=device,
        )

    if cfg.make_geo_error_plot:
        geo_error_plot_png(
            parquet_path=str(parquet_path),
            ckpt_path=summary.best_ckpt,
            labels=labels,
            image_size=diag_size,
            out_path=str(geo_path),
            dropout=cfg.dropout,
            device=device,
        )

    # latest symlink
    update_latest_symlink(runs_dir, run_dir)

    dist_path = str(run_dir / "dist_km.pt") if (run_dir / "dist_km.pt").exists() else ""

    artifacts = {
        "timestamp": ts,
        "run_dir": str(run_dir),
        "latest": str(runs_dir / "latest"),
        "parquet_path": str(parquet_path),
        "run_parquet": str(run_parquet),
        "labels_path": str(labels_path),
        "dist_km_pt": dist_path,
        "metrics_csv": summary.metrics_csv,
        "metrics_loss_png": plots.get("loss_png", ""),
        "metrics_valacc_png": plots.get("valacc_png", ""),
        "best_ckpt": summary.best_ckpt,
        "last_ckpt": summary.last_ckpt,
        "confusion_matrix_png": str(cm_path) if cm_path.exists() else "",
        "geo_error_png": str(geo_path) if geo_path.exists() else "",
        "num_classes": C,
        "label_idx_max": mx,
    }

    report = write_report(run_dir, cfg, summary, artifacts)

    meta = {
        "best_val_acc": summary.best_val_acc,
        "timestamp": ts,
        "run_dir": str(run_dir),
        "labels_path": str(labels_path),
        "parquet_path": str(parquet_path),
        "dropout": cfg.dropout,
        "geo_loss_enabled": cfg.geo_loss_enabled,
        "geo_tau_km": cfg.geo_tau_km,
        "geo_mix_ce": cfg.geo_mix_ce,
        "best_geo_median_km": summary.best_median_km,
        "best_geo_p90_km": summary.best_p90_km,
        "early_stopping_enabled": cfg.early_stopping_enabled,
        "early_stop_metric": cfg.early_stop_metric,
        "num_classes": C,
        "label_idx_max": mx,
    }

    maybe_update_global_best(models_dir, summary.best_ckpt, summary.best_val_acc, meta)

    print("\nDone.")
    print("Report:", report)
    print("Runs latest:", runs_dir / "latest")
    print("Global best:", models_dir / "best.pt")


if __name__ == "__main__":
    main()
