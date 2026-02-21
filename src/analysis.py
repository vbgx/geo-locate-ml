from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from .dataset import GeoDataset
from .geo import haversine_km
from .labels import LabelSpace
from .hierarchy import hierarchical_predict
from .model import MultiScaleCNN


def _ckpt_has_proxy_head(state: dict) -> bool:
    return any(str(k).startswith("proxy_head.") for k in state.keys())


def _ckpt_has_r6_head(state: dict) -> bool:
    return any(str(k).startswith("fc_r6.") for k in state.keys())


def load_model(ckpt_path: str, num_classes: int, num_classes_r6: int, dropout: float, device: str, hierarchical_enabled: bool):
    state = torch.load(ckpt_path, map_location=device)
    num_proxies = 5 if _ckpt_has_proxy_head(state) else 0
    use_hier = bool(hierarchical_enabled) or _ckpt_has_r6_head(state)

    m = MultiScaleCNN(
        num_classes=num_classes,
        dropout=dropout,
        num_proxies=num_proxies,
        num_classes_r6=int(num_classes_r6),
        hierarchical_enabled=use_hier,
    ).to(device)
    # strict=True is fine because we matched architecture to checkpoint
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


@torch.no_grad()
def confusion_matrix_png(
    parquet_path: str,
    ckpt_path: str,
    labels: LabelSpace,
    image_size: int,
    out_path: str,
    dropout: float,
    device: str,
    max_classes: int = 60,
    batch_size: int = 64,
):
    ds = GeoDataset(parquet_path, "val", image_size)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    y_true: List[int] = []
    y_pred: List[int] = []

    model = load_model(
        ckpt_path,
        num_classes=len(labels.h3_ids),
        num_classes_r6=len(labels.h3_ids_r6),
        dropout=dropout,
        device=device,
        hierarchical_enabled=False,
    )
    r7_parent = torch.tensor([labels.r7_to_r6[i] for i in range(len(labels.h3_ids))], dtype=torch.long).to(device)

    for batch in dl:
        # dataset can return 5-tuple (x,y,lat,lon,image_id) or 6-tuple with proxies
        if len(batch) == 5:
            x, y, _lat, _lon, _img_id = batch
        else:
            x, y, _lat, _lon, _img_id, _proxy_t = batch

        x = x.to(device, non_blocking=True)
        out = model(x)
        if model.hierarchical_enabled:
            if isinstance(out, tuple) and len(out) == 3:
                logits_r6, logits_r7, _proxy = out
            else:
                logits_r6, logits_r7 = out
            _pred_r6, preds, _masked = hierarchical_predict(logits_r6, logits_r7, r7_parent)
            p = preds.cpu().numpy().tolist()
        else:
            logits = out[0] if isinstance(out, tuple) else out
            p = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_pred.extend(p)
        y_true.extend(y.numpy().tolist())

    # Confusion matrix can be huge; show only top frequent classes in val
    df = pd.read_parquet(parquet_path)
    v = df[df["split"] == "val"]["label_idx"].value_counts().head(int(max_classes)).index.tolist()
    keep = set(int(i) for i in v)

    filt_true: List[int] = []
    filt_pred: List[int] = []
    for yt, yp in zip(y_true, y_pred):
        if yt in keep:
            filt_true.append(int(yt))
            # clamp to keep-space for display
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
    labels: LabelSpace,
    image_size: int,
    out_path: str,
    dropout: float,
    device: str,
    max_points: int = 5000,
    batch_size: int = 64,
):
    ds = GeoDataset(parquet_path, "val", image_size)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    model = load_model(
        ckpt_path,
        num_classes=len(labels.h3_ids),
        num_classes_r6=len(labels.h3_ids_r6),
        dropout=dropout,
        device=device,
        hierarchical_enabled=False,
    )
    r7_parent = torch.tensor([labels.r7_to_r6[i] for i in range(len(labels.h3_ids))], dtype=torch.long).to(device)

    errors: List[float] = []
    n = 0

    for batch in dl:
        if len(batch) == 5:
            x, y, lat, lon, _img_id = batch
        else:
            x, y, lat, lon, _img_id, _proxy_t = batch

        x = x.to(device, non_blocking=True)
        out = model(x)
        if model.hierarchical_enabled:
            if isinstance(out, tuple) and len(out) == 3:
                logits_r6, logits_r7, _proxy = out
            else:
                logits_r6, logits_r7 = out
            _pred_r6, preds, _masked = hierarchical_predict(logits_r6, logits_r7, r7_parent)
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
