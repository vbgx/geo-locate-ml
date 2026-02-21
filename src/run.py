from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch

from .config import TrainConfig
from .geo import haversine_km
from .labels import LabelSpace
from .paths import ensure_dir, p, update_latest_symlink
from .reporting import plot_metrics_csv
from .train_loop import run_training


def compute_distance_matrix_km(labels: LabelSpace) -> torch.Tensor:
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


def _ensure_geo_distance_matrix(run_dir: Path, labels: LabelSpace) -> torch.Tensor:
    num_classes = len(labels.h3_ids)
    D = compute_distance_matrix_km(labels)
    if D.shape != (num_classes, num_classes):
        raise RuntimeError(
            f"Distance matrix shape mismatch: D={tuple(D.shape)} expected=({num_classes},{num_classes})."
        )
    dist_path = run_dir / "dist_km.pt"
    torch.save(D, dist_path)
    return D


def _load_labels_json_compat(path: Path) -> LabelSpace:
    """
    Load labels.json from tools/dataset/merge_splits.py (simple format) OR LabelSpace.to_json format.

    Supports:
      A) LabelSpace.to_json():
         { "h3_ids": [...], "h3_to_idx": {...}, "idx_to_h3": {...}, "idx_to_centroid": {...}, ... }

      B) merge_splits.py (simple):
         { "num_classes": int, "labels": [...], "label_to_idx": {...} }
    """
    obj = json.loads(path.read_text(encoding="utf-8"))

    # Case A: full LabelSpace format
    if isinstance(obj, dict) and "h3_ids" in obj and "h3_to_idx" in obj and "idx_to_centroid" in obj:
        return LabelSpace.from_json(json.dumps(obj))

    # Case B: simple labels file
    labels_list = obj.get("labels")
    if not isinstance(labels_list, list) or not labels_list:
        raise RuntimeError(f"Invalid labels file: missing 'labels' list: {path}")

    labels_list = [str(h) for h in labels_list]

    h3_to_idx_obj = obj.get("label_to_idx")
    if isinstance(h3_to_idx_obj, dict) and h3_to_idx_obj:
        h3_to_idx = {str(k): int(v) for k, v in h3_to_idx_obj.items()}
    else:
        h3_to_idx = {str(h): int(i) for i, h in enumerate(labels_list)}

    idx_to_h3 = {int(i): str(h) for i, h in enumerate(labels_list)}

    import h3 as _h3  # local import

    idx_to_centroid: Dict[int, tuple[float, float]] = {}
    for i, h3_id in enumerate(labels_list):
        lat, lon = _h3.cell_to_latlng(str(h3_id))
        idx_to_centroid[int(i)] = (float(lat), float(lon))

    return LabelSpace(
        h3_ids=labels_list,
        h3_to_idx=h3_to_idx,
        idx_to_h3=idx_to_h3,
        idx_to_centroid=idx_to_centroid,
    )


def _maybe_update_global_best(models_dir: Path, best_run_ckpt: str, best_val_acc: float, meta: dict) -> bool:
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


def _write_report(run_dir: Path, cfg: TrainConfig, summary, artifacts: dict) -> Path:
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
    lines.append(f"- Train parquet: `{artifacts['train_parquet']}`\n")

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


def main() -> None:
    cfg = TrainConfig()

    runs_dir = p("runs")
    models_dir = p("models")
    ensure_dir(runs_dir)
    ensure_dir(models_dir)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs_dir / ts
    ensure_dir(run_dir)

    parquet_full = p(*cfg.parquet_index.split("/"))
    parquet_kept = parquet_full.parent / "images_kept.parquet"

    parquet_train = parquet_kept if parquet_kept.exists() else parquet_full
    if not parquet_train.exists():
        raise FileNotFoundError(
            f"Missing {parquet_train}.\n"
            f"Expected one of:\n"
            f" - {parquet_kept}\n"
            f" - {parquet_full}\n"
            f"Run: make rebuild"
        )

    labels_path = parquet_full.parent / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}. Run: make rebuild (merge-splits step).")

    df = pd.read_parquet(parquet_train)
    required_cols = {"id", "lat", "lon", "path", "h3_id", "label_idx", "split"}
    missing = sorted(list(required_cols - set(df.columns)))
    if missing:
        raise RuntimeError(f"{parquet_train} missing columns: {missing}")

    labels = _load_labels_json_compat(labels_path)

    C = int(len(labels.h3_ids))
    if C <= 1:
        raise RuntimeError(f"Num classes looks wrong (C={C}). Check labels.json + parquet.")

    mx = int(df["label_idx"].max()) if len(df) else -1
    if mx >= C:
        raise RuntimeError(f"label_idx max={mx} but num_classes={C}. labels.json and parquet are inconsistent.")

    # Snapshots
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "labels.json").write_text(labels_path.read_text(encoding="utf-8"), encoding="utf-8")
    shutil.copy2(parquet_train, run_dir / "images_train.parquet")

    dist_km: Optional[torch.Tensor] = None
    if cfg.geo_loss_enabled:
        dist_km = _ensure_geo_distance_matrix(run_dir, labels)

    summary = run_training(
        cfg,
        parquet_path=str(parquet_train),
        num_classes=C,
        run_dir=run_dir,
        idx_to_centroid=labels.idx_to_centroid,
        distance_km=dist_km if cfg.geo_loss_enabled else None,
    )

    plots = plot_metrics_csv(summary.metrics_csv, str(run_dir / "metrics"))
    update_latest_symlink(runs_dir, run_dir)

    artifacts: Dict[str, Any] = {
        "timestamp": ts,
        "run_dir": str(run_dir),
        "latest": str(runs_dir / "latest"),
        "train_parquet": str(parquet_train),
        "labels_path": str(labels_path),
        "dist_km_pt": str(run_dir / "dist_km.pt") if (run_dir / "dist_km.pt").exists() else "",
        "metrics_csv": summary.metrics_csv,
        "metrics_loss_png": plots.get("loss_png", "") if isinstance(plots, dict) else "",
        "metrics_valacc_png": plots.get("valacc_png", "") if isinstance(plots, dict) else "",
        "best_ckpt": summary.best_ckpt,
        "last_ckpt": summary.last_ckpt,
        "num_classes": C,
        "label_idx_max": mx,
    }

    report = _write_report(run_dir, cfg, summary, artifacts)

    meta = {
        "best_val_acc": summary.best_val_acc,
        "timestamp": ts,
        "run_dir": str(run_dir),
        "labels_path": str(labels_path),
        "train_parquet": str(parquet_train),
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
        "hierarchical_enabled": bool(getattr(cfg, "hierarchical_enabled", False)),
    }

    _maybe_update_global_best(models_dir, summary.best_ckpt, summary.best_val_acc, meta)

    print("\nDone.")
    print("Report:", report)
    print("Runs latest:", runs_dir / "latest")
    print("Global best:", models_dir / "best.pt")


if __name__ == "__main__":
    main()
