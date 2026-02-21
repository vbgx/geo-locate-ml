from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from .data import GeoDataset
from .geo import haversine_km
from .modeling import MultiScaleCNN


# ============================================================================
# Simple run charts (metrics.csv)
# ============================================================================

def plot_metrics_csv(metrics_csv: str, out_base: str) -> dict:
    df = pd.read_csv(metrics_csv)

    out_base = Path(out_base)
    loss_png = out_base.with_name(out_base.stem + "_loss.png")
    acc_png = out_base.with_name(out_base.stem + "_valacc.png")

    # Loss
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"])
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(loss_png, dpi=160)
    plt.close()

    # Val acc
    plt.figure()
    plt.plot(df["epoch"], df["val_acc"])
    plt.xlabel("epoch")
    plt.ylabel("val_acc")
    plt.title("Validation accuracy")
    plt.tight_layout()
    plt.savefig(acc_png, dpi=160)
    plt.close()

    return {"loss_png": str(loss_png), "valacc_png": str(acc_png)}


# ============================================================================
# Labels loading (works with both old + new labels.json)
# ============================================================================

@dataclass(frozen=True)
class LabelsLite:
    h3_ids: List[str]
    idx_to_centroid: Dict[int, Tuple[float, float]]
    h3_ids_r6: List[str]
    r7_to_r6: Dict[int, int]

    @property
    def num_classes(self) -> int:
        return int(len(self.h3_ids))

    @property
    def num_classes_r6(self) -> int:
        return int(len(self.h3_ids_r6))


def load_labels_json(path: Path) -> LabelsLite:
    """
    Supports:
      - legacy LabelSpace.to_json()
      - merge_splits labels.json: {"labels":[...], "label_to_idx":{...}}
      - extended labels.json: includes r7_to_r6, h3_ids_r6, etc.
    """
    obj = json.loads(path.read_text(encoding="utf-8"))

    labels_list = obj.get("labels") or obj.get("h3_ids")
    if not isinstance(labels_list, list) or not labels_list:
        raise RuntimeError(f"Invalid labels file: missing labels list: {path}")
    h3_ids = [str(x) for x in labels_list]

    import h3 as _h3
    idx_to_centroid: Dict[int, Tuple[float, float]] = {}
    for i, h3_id in enumerate(h3_ids):
        lat, lon = _h3.cell_to_latlng(str(h3_id))
        idx_to_centroid[int(i)] = (float(lat), float(lon))

    # r6 hierarchy (optional)
    h3_ids_r6: List[str] = []
    if isinstance(obj.get("h3_ids_r6"), list) and obj["h3_ids_r6"]:
        h3_ids_r6 = [str(x) for x in obj["h3_ids_r6"]]
    else:
        # derive from parents if missing
        parent_res = int(obj.get("parent_res", 6))
        parents = []
        for h in h3_ids:
            try:
                parents.append(str(_h3.cell_to_parent(str(h), parent_res)))
            except Exception:
                parents.append(str(h))
        h3_ids_r6 = sorted(set(parents))

    r7_to_r6: Dict[int, int] = {}
    raw = obj.get("r7_to_r6")
    if isinstance(raw, dict) and raw:
        r7_to_r6 = {int(k): int(v) for k, v in raw.items()}
    else:
        # derive mapping from parent membership
        parent_res = int(obj.get("parent_res", 6))
        h3_r6_to_idx = {h: i for i, h in enumerate(h3_ids_r6)}
        for r7_idx, h in enumerate(h3_ids):
            try:
                p = str(_h3.cell_to_parent(str(h), parent_res))
            except Exception:
                p = str(h)
            r7_to_r6[int(r7_idx)] = int(h3_r6_to_idx.get(p, 0))

    return LabelsLite(
        h3_ids=h3_ids,
        idx_to_centroid=idx_to_centroid,
        h3_ids_r6=h3_ids_r6,
        r7_to_r6=r7_to_r6,
    )


# ============================================================================
# Model loading (checkpoint-aware)
# ============================================================================

def _ckpt_has_proxy_head(state: dict) -> bool:
    return any(str(k).startswith("proxy_head.") for k in state.keys())


def _ckpt_has_r6_head(state: dict) -> bool:
    return any(str(k).startswith("fc_r6.") for k in state.keys())


def load_model(
    ckpt_path: str,
    *,
    num_classes: int,
    num_classes_r6: int,
    dropout: float,
    device: str,
    hierarchical_enabled: bool,
) -> MultiScaleCNN:
    state = torch.load(ckpt_path, map_location=device)
    num_proxies = 5 if _ckpt_has_proxy_head(state) else 0
    use_hier = bool(hierarchical_enabled) or _ckpt_has_r6_head(state)

    m = MultiScaleCNN(
        num_classes=int(num_classes),
        dropout=float(dropout),
        num_proxies=int(num_proxies),
        num_classes_r6=int(num_classes_r6),
        hierarchical_enabled=bool(use_hier),
    ).to(device)

    m.load_state_dict(state, strict=True)
    m.eval()
    return m


# ============================================================================
# Hierarchy helpers (local, to avoid extra files)
# ============================================================================

def _mask_r7_logits(logits_r7: torch.Tensor, y6: torch.Tensor, r7_parent: torch.Tensor) -> torch.Tensor:
    """
    logits_r7: [B,C7]
    y6: [B] (r6 target idx)
    r7_parent: [C7] mapping r7->r6
    Mask logits so only children of y6 are valid.
    """
    if logits_r7.dim() != 2:
        raise ValueError("logits_r7 must be [B,C]")
    B, C = logits_r7.shape
    if r7_parent.numel() != C:
        raise ValueError("r7_parent size mismatch")
    # mask positions where parent != y6
    parent_of_each_class = r7_parent.view(1, -1).expand(B, C)  # [B,C]
    y6_expand = y6.view(-1, 1).expand(B, C)
    mask = parent_of_each_class != y6_expand
    out = logits_r7.clone()
    out[mask] = -1e9
    return out


def _hierarchical_predict(
    logits_r6: torch.Tensor,
    logits_r7: torch.Tensor,
    r7_parent: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      pred_r6: [B]
      pred_r7: [B]
      logits_r7_masked: [B,C7]
    """
    pred_r6 = torch.argmax(logits_r6, dim=1)
    logits_r7_masked = _mask_r7_logits(logits_r7, pred_r6, r7_parent)
    pred_r7 = torch.argmax(logits_r7_masked, dim=1)
    return pred_r6, pred_r7, logits_r7_masked


# ============================================================================
# Confusion matrix + geo error plots
# ============================================================================

@torch.no_grad()
def confusion_matrix_png(
    parquet_path: str,
    ckpt_path: str,
    labels_path: str,
    image_size: int,
    out_path: str,
    dropout: float,
    device: str,
    max_classes: int = 60,
    batch_size: int = 64,
) -> None:
    labels = load_labels_json(Path(labels_path))

    ds = GeoDataset(parquet_path, "val", int(image_size), hierarchical_enabled=False)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    y_true: List[int] = []
    y_pred: List[int] = []

    model = load_model(
        ckpt_path,
        num_classes=labels.num_classes,
        num_classes_r6=labels.num_classes_r6,
        dropout=dropout,
        device=device,
        hierarchical_enabled=False,
    )

    r7_parent = torch.tensor([labels.r7_to_r6[i] for i in range(labels.num_classes)], dtype=torch.long).to(device)

    for batch in dl:
        # expected: (x,y,lat,lon,image_id) or (x,y6,y7,lat,lon,image_id)
        if len(batch) >= 6:
            x = batch[0]
            y = batch[2]  # y7
        else:
            x, y = batch[0], batch[1]

        x = x.to(device, non_blocking=True)

        out = model(x)
        if model.hierarchical_enabled:
            if isinstance(out, tuple) and len(out) >= 2:
                logits_r6, logits_r7 = out[0], out[1]
            else:
                raise RuntimeError("Checkpoint indicates hierarchy but model output is not (logits_r6, logits_r7)")
            _pred_r6, preds, _masked = _hierarchical_predict(logits_r6, logits_r7, r7_parent)
            p = preds.cpu().numpy().tolist()
        else:
            logits = out[0] if isinstance(out, tuple) else out
            p = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        y_pred.extend(p)
        y_true.extend(y.numpy().tolist())

    df = pd.read_parquet(parquet_path)
    v = df[df["split"] == "val"]["label_idx"].value_counts().head(int(max_classes)).index.tolist()
    keep = set(int(i) for i in v)

    filt_true: List[int] = []
    filt_pred: List[int] = []
    for yt, yp in zip(y_true, y_pred):
        if int(yt) in keep:
            filt_true.append(int(yt))
            filt_pred.append(int(yp) if int(yp) in keep else int(yt))

    if not filt_true:
        return

    labels_sorted = sorted(list(keep))
    cm = confusion_matrix(filt_true, filt_pred, labels=labels_sorted, normalize="true")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[str(i) for i in labels_sorted],
    )
    plt.figure(figsize=(12, 12))
    disp.plot(include_values=False, xticks_rotation="vertical")
    plt.title("Confusion matrix (val) — top frequent classes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


@torch.no_grad()
def geo_error_plot_png(
    parquet_path: str,
    ckpt_path: str,
    labels_path: str,
    image_size: int,
    out_path: str,
    dropout: float,
    device: str,
    max_points: int = 5000,
    batch_size: int = 64,
) -> None:
    labels = load_labels_json(Path(labels_path))

    ds = GeoDataset(parquet_path, "val", int(image_size), hierarchical_enabled=False)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    model = load_model(
        ckpt_path,
        num_classes=labels.num_classes,
        num_classes_r6=labels.num_classes_r6,
        dropout=dropout,
        device=device,
        hierarchical_enabled=False,
    )
    r7_parent = torch.tensor([labels.r7_to_r6[i] for i in range(labels.num_classes)], dtype=torch.long).to(device)

    errors: List[float] = []
    n = 0

    for batch in dl:
        if len(batch) >= 6:
            x = batch[0]
            y = batch[2]
            lat = batch[3]
            lon = batch[4]
        else:
            x, y, lat, lon = batch[0], batch[1], batch[2], batch[3]

        x = x.to(device, non_blocking=True)

        out = model(x)
        if model.hierarchical_enabled:
            if isinstance(out, tuple) and len(out) >= 2:
                logits_r6, logits_r7 = out[0], out[1]
            else:
                raise RuntimeError("Checkpoint indicates hierarchy but model output is not (logits_r6, logits_r7)")
            _pred_r6, preds, _masked = _hierarchical_predict(logits_r6, logits_r7, r7_parent)
            pred = preds.cpu().numpy()
        else:
            logits = out[0] if isinstance(out, tuple) else out
            pred = torch.argmax(logits, dim=1).cpu().numpy()

        y_np = y.numpy()
        lat_np = lat.numpy()
        lon_np = lon.numpy()

        for yi, pi, lati, loni in zip(y_np, pred, lat_np, lon_np):
            plat, plon = labels.idx_to_centroid[int(pi)]
            err = haversine_km(float(lati), float(loni), float(plat), float(plon))
            errors.append(float(err))
            n += 1
            if n >= int(max_points):
                break
        if n >= int(max_points):
            break

    if not errors:
        return

    plt.figure()
    plt.hist(errors, bins=50)
    plt.xlabel("geo error (km)")
    plt.ylabel("count")
    plt.title("Geo error distribution (val)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
