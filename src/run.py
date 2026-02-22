from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from .config import TrainConfig
from .geo import haversine_km
from .labels import LabelSpace
from .paths import ensure_dir, p, update_latest_symlink
from .reporting import plot_metrics_csv
from .train_loop import run_training


# ============================================================================
# Distances
# ============================================================================


def compute_distance_matrix_km(idx_to_centroid: Dict[int, Tuple[float, float]]) -> torch.Tensor:
    """
    Build a (C,C) distance matrix between class centroids in kilometers.
    idx_to_centroid must contain contiguous keys [0..C-1].
    """
    C = len(idx_to_centroid)
    for i in range(C):
        if i not in idx_to_centroid:
            raise RuntimeError(f"idx_to_centroid missing key {i} (expected contiguous 0..{C-1})")

    D = torch.zeros((C, C), dtype=torch.float32)
    for i in range(C):
        lat1, lon1 = idx_to_centroid[i]
        for j in range(C):
            lat2, lon2 = idx_to_centroid[j]
            D[i, j] = float(haversine_km(lat1, lon1, lat2, lon2))
    return D


def _ensure_geo_distance_matrix(run_dir: Path, idx_to_centroid: Dict[int, Tuple[float, float]]) -> torch.Tensor:
    """
    Always recompute distances for the CURRENT run_dir / CURRENT labels.
    Never read from runs/latest (can be stale).
    """
    C = len(idx_to_centroid)
    D = compute_distance_matrix_km(idx_to_centroid)
    if D.shape != (C, C):
        raise RuntimeError(f"Distance matrix shape mismatch: D={tuple(D.shape)} expected=({C},{C}).")

    dist_path = run_dir / "dist_km.pt"
    torch.save(D, dist_path)
    return D


# ============================================================================
# Labels loading (compat)
# ============================================================================


def _load_labels_json_compat(path: Path) -> Tuple[LabelSpace, Optional[Dict[int, int]], int]:
    """
    Load labels.json from either:

    A) LabelSpace.to_json():
       { "h3_ids": [...], "h3_to_idx": {...}, "idx_to_h3": {...}, "idx_to_centroid": {...}, ... }

    B) merge_splits_and_build_training_parquet (simple):
       {
         "num_classes": int,
         "labels": [...],
         "label_to_idx": {...},
         "parent_res": 6,
         "r7_to_r6": { "0": 12, "1": 12, ... }   # may exist
       }

    Returns:
      (labels_space, r7_to_r6 (optional), parent_res)
    """
    obj = json.loads(path.read_text(encoding="utf-8"))

    # Case A: full LabelSpace format
    if isinstance(obj, dict) and "h3_ids" in obj and "h3_to_idx" in obj and "idx_to_centroid" in obj:
        ls = LabelSpace.from_json(json.dumps(obj))
        r7_to_r6 = getattr(ls, "r7_to_r6", None)
        parent_res = int(getattr(ls, "parent_res", obj.get("parent_res", 6)))
        return ls, r7_to_r6, parent_res

    # Case B: simple labels file
    labels_list = obj.get("labels")
    if not isinstance(labels_list, list) or not labels_list:
        raise RuntimeError(f"Invalid labels file: missing 'labels' list: {path}")

    labels_list = [str(h) for h in labels_list]

    label_to_idx_raw = obj.get("label_to_idx") or {}
    if isinstance(label_to_idx_raw, dict) and label_to_idx_raw:
        h3_to_idx = {str(k): int(v) for k, v in label_to_idx_raw.items()}
    else:
        h3_to_idx = {str(h): int(i) for i, h in enumerate(labels_list)}

    idx_to_h3 = {int(i): str(h) for i, h in enumerate(labels_list)}

    import h3 as _h3  # local import

    idx_to_centroid: Dict[int, Tuple[float, float]] = {}
    for i, h3_id in enumerate(labels_list):
        lat, lon = _h3.cell_to_latlng(str(h3_id))
        idx_to_centroid[int(i)] = (float(lat), float(lon))

    parent_res = int(obj.get("parent_res", 6))

    r7_to_r6_raw = obj.get("r7_to_r6")
    r7_to_r6: Optional[Dict[int, int]] = None
    if isinstance(r7_to_r6_raw, dict) and r7_to_r6_raw:
        r7_to_r6 = {int(k): int(v) for k, v in r7_to_r6_raw.items()}

    ls = LabelSpace(
        h3_ids=labels_list,
        h3_to_idx=h3_to_idx,
        idx_to_h3=idx_to_h3,
        idx_to_centroid=idx_to_centroid,
    )

    return ls, r7_to_r6, parent_res


# ============================================================================
# Proxies (validation)
# ============================================================================


def _validate_proxies_columns(cfg: TrainConfig, df: pd.DataFrame, parquet_path: Path) -> int:
    """
    Ensure proxy columns exist when proxy loss is enabled.
    Returns:
      - num_proxies (len(cfg.proxy_cols)) if enabled
      - 0 otherwise
    """
    proxy_enabled = bool(getattr(cfg, "proxy_loss_enabled", False)) and float(
        getattr(cfg, "proxy_loss_weight", 0.0)
    ) > 0.0
    if not proxy_enabled:
        return 0

    proxy_cols = list(getattr(cfg, "proxy_cols", []) or [])
    if not proxy_cols:
        raise RuntimeError("proxy_loss_enabled=True but cfg.proxy_cols is empty.")

    missing = [c for c in proxy_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            "proxy_loss_enabled=True but parquet is missing proxy columns:\n"
            f"  parquet: {parquet_path}\n"
            f"  missing: {missing}\n"
            "Build proxies first and point cfg.parquet_index to the parquet that contains them."
        )

    return int(len(proxy_cols))


# ============================================================================
# Global best + report
# ============================================================================


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
    lines: List[str] = []
    lines.append("# geo-locate-ml — Run report\n\n")
    lines.append(f"- Timestamp: `{artifacts['timestamp']}`\n")
    lines.append(f"- Device: `{summary.device}`\n")
    lines.append(f"- Best val acc: `{summary.best_val_acc:.4f}`\n")
    lines.append(f"- Best epoch: `{summary.best_epoch}`\n")
    lines.append(f"- Best image_size: `{summary.best_image_size}`\n")
    lines.append(f"- Best geo median (val): `{summary.best_median_km:.2f} km`\n")
    lines.append(f"- Best geo p90 (val): `{summary.best_p90_km:.2f} km`\n")
    lines.append(f"- Num classes: `{artifacts['num_classes']}`\n")
    lines.append(
        f"- Proxies enabled: `{artifacts.get('proxy_loss_enabled', False)}` "
        f"(num_proxies={artifacts.get('num_proxies', 0)})\n"
    )
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


# ============================================================================
# Main
# ============================================================================


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
            "Run: make rebuild"
        )

    labels_path = parquet_full.parent / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}. Run: make rebuild (merge-splits step).")

    df = pd.read_parquet(parquet_train)

    required_cols = {"id", "lat", "lon", "path", "h3_id", "label_idx", "split"}
    missing = sorted(list(required_cols - set(df.columns)))
    if missing:
        raise RuntimeError(f"{parquet_train} missing columns: {missing}")

    hierarchical_enabled = bool(getattr(cfg, "hierarchical_enabled", False))
    if hierarchical_enabled and "label_r6_idx" not in df.columns:
        raise RuntimeError(
            "hierarchical_enabled=True but parquet is missing 'label_r6_idx'. "
            "Rebuild via merge_splits_and_build_training_parquet."
        )

    labels, r7_to_r6, parent_res = _load_labels_json_compat(labels_path)

    C = int(len(labels.h3_ids))
    if C <= 1:
        raise RuntimeError(f"Num classes looks wrong (C={C}). Check labels.json + parquet.")

    mx = int(df["label_idx"].max()) if len(df) else -1
    if mx >= C:
        raise RuntimeError(f"label_idx max={mx} but num_classes={C}. labels.json and parquet are inconsistent.")

    # Proxies validation (only if enabled)
    num_proxies = _validate_proxies_columns(cfg, df, parquet_train)
    proxy_loss_enabled = bool(getattr(cfg, "proxy_loss_enabled", False)) and float(
        getattr(cfg, "proxy_loss_weight", 0.0)
    ) > 0.0

    # Snapshots (make the run self-contained)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "labels.json").write_text(labels_path.read_text(encoding="utf-8"), encoding="utf-8")
    shutil.copy2(parquet_train, run_dir / "images_train.parquet")

    # Geo distances — ALWAYS compute for this run_dir with this labels set
    dist_km: Optional[torch.Tensor] = None
    if bool(getattr(cfg, "geo_loss_enabled", False)):
        dist_km = _ensure_geo_distance_matrix(run_dir, labels.idx_to_centroid)
        if dist_km.shape != (C, C):
            raise RuntimeError(f"dist_km shape {tuple(dist_km.shape)} != ({C},{C}) — stale distance matrix")

    # --- Hierarchy wiring (required if hierarchical_enabled) ---
    num_classes_r6 = 0
    r7_parent_t: Optional[torch.Tensor] = None

    if hierarchical_enabled:
        if r7_to_r6 is None or len(r7_to_r6) != C:
            raise RuntimeError(
                "hierarchical_enabled=True but labels.json missing r7_to_r6 (or wrong size). "
                "Rebuild labels via merge step that writes r7_to_r6."
            )

        for i in range(C):
            if i not in r7_to_r6:
                raise RuntimeError(f"labels.json r7_to_r6 missing key {i} (expected 0..{C-1})")

        num_classes_r6 = int(max(r7_to_r6.values()) + 1) if r7_to_r6 else 0
        if num_classes_r6 <= 1:
            raise RuntimeError("hierarchical_enabled=True but num_classes_r6 looks invalid")

        r7_parent = torch.empty((C,), dtype=torch.long)
        for k, v in r7_to_r6.items():
            r7_parent[int(k)] = int(v)
        r7_parent_t = r7_parent

    summary = run_training(
        cfg,
        parquet_path=str(parquet_train),
        num_classes=C,
        run_dir=run_dir,
        idx_to_centroid=labels.idx_to_centroid,
        distance_km=dist_km if bool(getattr(cfg, "geo_loss_enabled", False)) else None,
        num_proxies=int(num_proxies),
        num_classes_r6=num_classes_r6,
        r7_parent=r7_parent_t,
    )

    plots = plot_metrics_csv(summary.metrics_csv, str(run_dir / "metrics"))
    update_latest_symlink(runs_dir, run_dir)

    artifacts: Dict[str, Any] = {
        "timestamp": ts,
        "run_dir": str(run_dir),
        "latest": str(runs_dir / "latest"),
        "train_parquet": str(parquet_train),
        "labels_path": str(labels_path),
        "proxy_loss_enabled": bool(proxy_loss_enabled),
        "proxy_loss_weight": float(getattr(cfg, "proxy_loss_weight", 0.0)),
        "num_proxies": int(num_proxies),
        "proxy_cols": list(getattr(cfg, "proxy_cols", []) or []),
        "dist_km_pt": str(run_dir / "dist_km.pt") if (run_dir / "dist_km.pt").exists() else "",
        "metrics_csv": summary.metrics_csv,
        "metrics_loss_png": plots.get("loss_png", "") if isinstance(plots, dict) else "",
        "metrics_valacc_png": plots.get("valacc_png", "") if isinstance(plots, dict) else "",
        "best_ckpt": summary.best_ckpt,
        "last_ckpt": summary.last_ckpt,
        "num_classes": C,
        "label_idx_max": mx,
        "hierarchical_enabled": hierarchical_enabled,
        "num_classes_r6": int(num_classes_r6),
        "parent_res": int(parent_res),
    }

    report = _write_report(run_dir, cfg, summary, artifacts)

    meta = {
        "best_val_acc": summary.best_val_acc,
        "timestamp": ts,
        "run_dir": str(run_dir),
        "latest": str(runs_dir / "latest"),
        "train_parquet": str(parquet_train),
        "labels_path": str(labels_path),
        "proxy_loss_enabled": bool(proxy_loss_enabled),
        "proxy_loss_weight": float(getattr(cfg, "proxy_loss_weight", 0.0)),
        "num_proxies": int(num_proxies),
        "proxy_cols": list(getattr(cfg, "proxy_cols", []) or []),
        "dropout": float(cfg.dropout),
        "geo_loss_enabled": bool(getattr(cfg, "geo_loss_enabled", False)),
        "geo_tau_km": float(getattr(cfg, "geo_tau_km", 0.0)),
        "geo_mix_ce": float(getattr(cfg, "geo_mix_ce", 0.0)),
        "best_geo_median_km": summary.best_median_km,
        "best_geo_p90_km": summary.best_p90_km,
        "early_stopping_enabled": bool(getattr(cfg, "early_stopping_enabled", False)),
        "early_stop_metric": str(getattr(cfg, "early_stop_metric", "")),
        "hierarchical_enabled": hierarchical_enabled,
        "num_classes": C,
        "label_idx_max": mx,
        "num_classes_r6": int(num_classes_r6),
        "parent_res": int(parent_res),
    }

    _maybe_update_global_best(models_dir, summary.best_ckpt, summary.best_val_acc, meta)

    print("\nDone.")
    print("Report:", report)
    print("Runs latest:", runs_dir / "latest")
    print("Global best:", models_dir / "best.pt")
    if num_proxies:
        print(
            f"Proxies: enabled (num_proxies={num_proxies}) "
            f"weight={float(getattr(cfg, 'proxy_loss_weight', 0.0))}"
        )
        print(f"Proxy cols: {list(getattr(cfg, 'proxy_cols', []) or [])}")


if __name__ == "__main__":
    main()