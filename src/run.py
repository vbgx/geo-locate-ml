from __future__ import annotations

import json
import math
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


def _load_far_rate_map(path: str, labels_path: str) -> dict[str, float]:
    if not path:
        return {}
    pth = Path(path)
    if not pth.exists():
        print(f"[merge] far_rate file not found: {pth} (skipping)")
        return {}

    if pth.suffix.lower() == ".parquet":
        df = pd.read_parquet(pth)
    else:
        df = pd.read_csv(pth)

    if "far_rate" not in df.columns:
        print(f"[merge] far_rate column missing in {pth} (skipping)")
        return {}

    if "h3_id" in df.columns:
        return {str(h): float(fr) for h, fr in zip(df["h3_id"].tolist(), df["far_rate"].tolist())}

    if "true_idx" in df.columns:
        if not labels_path:
            print(f"[merge] far_rate file has true_idx but labels_path is empty (skipping)")
            return {}
        lp = Path(labels_path)
        if not lp.exists():
            print(f"[merge] labels_path not found: {lp} (skipping)")
            return {}
        labels = LabelSpace.from_json(lp.read_text(encoding="utf-8"))
        idx_to_h3 = labels.idx_to_h3
        out: dict[str, float] = {}
        for idx, fr in zip(df["true_idx"].tolist(), df["far_rate"].tolist()):
            try:
                h = idx_to_h3[int(idx)]
            except Exception:
                continue
            out[str(h)] = float(fr)
        return out

    print(f"[merge] far_rate file missing h3_id/true_idx columns: {pth} (skipping)")
    return {}


def _merge_toxic_cells(
    df: pd.DataFrame,
    *,
    min_cell_count: int,
    parent_res: int,
    far_rate_map: dict[str, float],
    far_rate_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    counts = df["h3_id"].value_counts()
    merge_set = set(counts[counts < int(min_cell_count)].index.tolist())

    if far_rate_map:
        for h, fr in far_rate_map.items():
            if float(fr) > float(far_rate_threshold):
                merge_set.add(str(h))

    mapping_rows = []
    for h3_id, cnt in counts.items():
        fr = float(far_rate_map.get(str(h3_id), float("nan"))) if far_rate_map else float("nan")
        reasons: list[str] = []
        if int(cnt) < int(min_cell_count):
            reasons.append(f"count<{int(min_cell_count)}")
        if far_rate_map and not math.isnan(fr) and fr > float(far_rate_threshold):
            reasons.append(f"far_rate>{float(far_rate_threshold)}")
        merged = str(h3_id) in merge_set
        if merged:
            parent = str(h3.cell_to_parent(str(h3_id), int(parent_res)))
        else:
            parent = str(h3_id)
        mapping_rows.append(
            {
                "h3_id_raw": str(h3_id),
                "h3_id_merged": parent,
                "cell_count": int(cnt),
                "far_rate": fr if far_rate_map else float("nan"),
                "merged": bool(merged),
                "reason": "|".join(reasons),
            }
        )

    if merge_set:
        df.loc[df["h3_id"].isin(merge_set), "h3_id"] = df.loc[df["h3_id"].isin(merge_set), "h3_id"].map(
            lambda h: str(h3.cell_to_parent(str(h), int(parent_res)))
        )

    mapping = pd.DataFrame(mapping_rows)
    return df, mapping


def _filter_min_samples_per_split(
    df: pd.DataFrame,
    *,
    min_samples: int,
    split_col: str = "split",
    label_col: str = "h3_id",
) -> tuple[pd.DataFrame, list[str]]:
    counts = df.groupby([label_col, split_col]).size().unstack(fill_value=0)
    keep = counts[(counts >= int(min_samples)).all(axis=1)].index.tolist()
    out = df[df[label_col].isin(keep)].copy()
    return out, keep


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
    lines.append(f"- Hierarchical enabled: `{cfg.hierarchical_enabled}`\n")
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

    # H3 (r7)
    df = compute_h3(df, cfg.h3_resolution)

    # Merge toxic micro-classes (r7 -> r6)
    far_rate_map = _load_far_rate_map(
        getattr(cfg, "merge_far_rate_path", ""),
        getattr(cfg, "merge_far_labels_path", ""),
    )
    df, merge_map = _merge_toxic_cells(
        df,
        min_cell_count=int(getattr(cfg, "merge_min_cell_count", 15)),
        parent_res=int(getattr(cfg, "merge_parent_res", 6)),
        far_rate_map=far_rate_map,
        far_rate_threshold=float(getattr(cfg, "merge_far_rate_threshold", 0.8)),
    )

    # Splits (sequence-aware)
    df = assign_split_by_sequence(
        df,
        seed=cfg.seed,
        p_train=cfg.split_train,
        p_val=cfg.split_val,
        p_test=cfg.split_test,
    )

    # Hard filter: min samples per split AFTER split
    df, _kept_labels = _filter_min_samples_per_split(df, min_samples=cfg.min_cell_samples)

    if len(df) < 100:
        raise RuntimeError("Too few samples after filtering. Lower min_cell_samples or download more images.")

    # Label space (includes r6 hierarchy)
    labels = build_label_space(df, parent_res=int(getattr(cfg, "merge_parent_res", 6)))
    df["label_idx"] = df["h3_id"].map(labels.h3_to_idx).astype(int)
    df["label_r6_idx"] = df["label_idx"].map(labels.r7_to_r6).astype(int)
    df["h3_id_r6"] = df["label_r6_idx"].map(labels.idx_to_h3_r6)

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

    merge_map_path = run_dir / "merge_map.parquet"
    try:
        merge_map.to_parquet(merge_map_path, index=False)
    except Exception:
        merge_map_path = None

    # Write labels + config
    labels_path = run_dir / "labels.json"
    labels_path.write_text(labels.to_json(), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    # Geo distance matrix aligned to this run's labels (fixes the crash you hit)
    dist_km: torch.Tensor | None = None
    if cfg.geo_loss_enabled:
        dist_km = _ensure_geo_distance_matrix(cfg, run_dir, labels)

    # Train
    r7_parent = None
    if bool(cfg.hierarchical_enabled):
        r7_parent = torch.tensor([labels.r7_to_r6[i] for i in range(len(labels.h3_ids))], dtype=torch.long)

    summary = run_training(
        cfg,
        parquet_path=str(parquet_path),
        num_classes=C,
        run_dir=run_dir,
        idx_to_centroid=labels.idx_to_centroid,
        distance_km=dist_km if cfg.geo_loss_enabled else None,
        num_classes_r6=len(labels.h3_ids_r6),
        r7_parent=r7_parent,
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
        "merge_map": str(merge_map_path) if merge_map_path else "",
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
        "hierarchical_enabled": bool(cfg.hierarchical_enabled),
        "num_classes_r6": len(labels.h3_ids_r6),
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
