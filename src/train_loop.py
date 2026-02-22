from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from .config import TrainConfig
from .data import GeoDataset, ensure_dir, p, update_latest_symlink
from .geo import (
    HardNegConfig,
    build_sample_weights,
    haversine_km,
    hierarchical_predict,
    load_pool,
    save_pool,
    update_pool,
)
from .metrics_geo import compute_geo_kpi
from .modeling import GeoSoftTargetLoss, MultiScaleCNN, compute_class_weights_from_parquet


# ============================================================================
# Train summary
# ============================================================================


@dataclass
class TrainSummary:
    best_val_acc: float
    best_epoch: int
    best_image_size: int
    device: str
    metrics_csv: str
    best_ckpt: str
    last_ckpt: str
    best_p90_km: float
    best_median_km: float


# ============================================================================
# Helpers
# ============================================================================


def _accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.numel() == 0:
        return 0.0
    return float((preds == targets).float().mean().item())


def _topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    if targets.numel() == 0 or logits.numel() == 0:
        return 0.0
    k = min(int(k), int(logits.size(1)))
    topk = torch.topk(logits, k=k, dim=1).indices
    ok = (topk == targets.view(-1, 1)).any(dim=1)
    return float(ok.float().mean().item())


def _is_improved(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return current < (best - float(min_delta))
    if mode == "max":
        return current > (best + float(min_delta))
    raise ValueError("mode must be 'min' or 'max'")


def _dump_val_topk_parquet(
    out_path: Path,
    image_ids: list[str],
    targets: torch.Tensor,
    preds: torch.Tensor,
    logits_all: torch.Tensor,
    k: int,
) -> None:
    if logits_all.numel() == 0:
        return

    n, c = logits_all.shape
    k = min(int(k), int(c))
    topk_vals, topk_idxs = torch.topk(logits_all, k=k, dim=1)

    if len(image_ids) != int(n):
        image_ids = [f"val_{i:08d}" for i in range(int(n))]

    df = pd.DataFrame(
        {
            "image_id": image_ids,
            "true_idx": targets.cpu().numpy().astype(int),
            "pred_idx": preds.cpu().numpy().astype(int),
        }
    )
    for j in range(k):
        df[f"top{j+1}_idx"] = topk_idxs[:, j].cpu().numpy().astype(int)
        df[f"top{j+1}_logit"] = topk_vals[:, j].cpu().numpy().astype(float)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def _dump_val_errors_parquet(
    out_path: Path,
    image_ids: list[str],
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    lat_true: np.ndarray,
    lon_true: np.ndarray,
    idx_to_centroid: Dict[int, Tuple[float, float]],
) -> np.ndarray:
    n = len(image_ids)
    dist = np.zeros((n,), dtype=np.float64)

    for i in range(n):
        yp = int(y_pred[i])
        plat, plon = idx_to_centroid[int(yp)]
        dist[i] = haversine_km(
            float(lat_true[i]),
            float(lon_true[i]),
            float(plat),
            float(plon),
        )

    df = pd.DataFrame(
        {
            "image_id": image_ids,
            "true_idx": y_true.cpu().numpy().astype(int),
            "pred_idx": y_pred.cpu().numpy().astype(int),
            "lat_true": lat_true.astype(float),
            "lon_true": lon_true.astype(float),
            "dist_km": dist.astype(float),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return dist


def _is_str_list(x: Any) -> bool:
    if isinstance(x, list) and (len(x) == 0 or isinstance(x[0], str)):
        return True
    if isinstance(x, tuple) and (len(x) == 0 or isinstance(x[0], str)):
        return True
    return False


def _to_1d_float_tensor(x: Any) -> Optional[torch.Tensor]:
    """
    Convert a batch field to a 1D float tensor if plausible, else None.
    Accepts torch tensors, numpy arrays, python lists.
    """
    try:
        if torch.is_tensor(x):
            t = x
        else:
            t = torch.as_tensor(x)
        if t.numel() == 0:
            return None
        if not (t.dtype.is_floating_point or t.dtype in (torch.int16, torch.int32, torch.int64, torch.uint8)):
            return None
        t = t.detach()
        if t.dim() == 1:
            return t.float()
        if t.dim() == 2 and t.size(1) == 1:
            return t[:, 0].float()
        return None
    except Exception:
        return None


def _looks_like_lat(t: torch.Tensor) -> bool:
    if t.numel() == 0:
        return False
    vmin = float(t.min().cpu())
    vmax = float(t.max().cpu())
    return (vmin >= -90.5) and (vmax <= 90.5)


def _looks_like_lon(t: torch.Tensor) -> bool:
    if t.numel() == 0:
        return False
    vmin = float(t.min().cpu())
    vmax = float(t.max().cpu())
    return (vmin >= -180.5) and (vmax <= 180.5)


def _extract_lat_lon_img_id(batch: Sequence[Any], *, B: int) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Robustly find (lat, lon, img_id) regardless of proxy insertion/order.
    - img_id: first list/tuple[str] when scanning from the end.
    - lat/lon: two 1D numeric tensors length B with plausible ranges.
    """
    img_id: List[str] = []
    for x in reversed(batch):
        if _is_str_list(x):
            img_id = [str(s) for s in x]
            break
    if not img_id:
        last = batch[-1]
        if torch.is_tensor(last) and last.dim() == 1 and int(last.numel()) == int(B):
            img_id = [str(int(v)) for v in last.detach().cpu().tolist()]
        else:
            img_id = [f"val_{i:08d}" for i in range(B)]

    cands: List[torch.Tensor] = []
    for x in batch:
        t = _to_1d_float_tensor(x)
        if t is None:
            continue
        if int(t.numel()) != int(B):
            continue
        cands.append(t)

    lat_t: Optional[torch.Tensor] = None
    lon_t: Optional[torch.Tensor] = None
    for t in cands:
        if lat_t is None and _looks_like_lat(t):
            lat_t = t
            continue
        if lon_t is None and _looks_like_lon(t):
            lon_t = t
            continue

    if lat_t is None or lon_t is None:
        dbg: List[str] = []
        for i, x in enumerate(batch):
            if torch.is_tensor(x):
                dbg.append(f"{i}: tensor shape={tuple(x.shape)} dtype={x.dtype}")
            elif _is_str_list(x):
                dbg.append(f"{i}: str_list len={len(x)}")
            else:
                dbg.append(f"{i}: {type(x).__name__}")
        raise RuntimeError("Could not infer lat/lon from batch. Batch layout:\n" + "\n".join(dbg))

    return lat_t, lon_t, img_id


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """SmoothL1 with per-element mask (0/1). Returns a scalar."""
    if pred.shape != target.shape:
        raise RuntimeError(f"proxy pred/target shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if mask.shape != target.shape:
        raise RuntimeError(f"proxy mask shape mismatch: mask={tuple(mask.shape)} target={tuple(target.shape)}")
    diff = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    diff = diff * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom



def _stats(t: torch.Tensor) -> str:
    t0 = t.detach()
    return f"shape={tuple(t0.shape)} min={float(t0.min().cpu()):.4f} max={float(t0.max().cpu()):.4f}"


def _debug_first_batch(
    *,
    epoch: int,
    n_batches: int,
    hierarchical_enabled: bool,
    loss: torch.Tensor,
    logits_r6: Optional[torch.Tensor] = None,
    logits_r7: Optional[torch.Tensor] = None,
) -> None:
    if not (epoch == 1 and n_batches == 0):
        return

    tqdm.write("DEBUG first batch:")
    tqdm.write(f"  hierarchical_enabled: {hierarchical_enabled}")
    tqdm.write(f"  loss: {float(loss.detach().cpu()):.6f}")
    if hierarchical_enabled:
        if logits_r6 is not None:
            tqdm.write(f"  logits_r6: {_stats(logits_r6)}")
        if logits_r7 is not None:
            tqdm.write(f"  logits_r7(masked): {_stats(logits_r7)}")
    else:
        if logits_r7 is not None:
            tqdm.write(f"  logits: {_stats(logits_r7)}")


# ============================================================================
# Training core
# ============================================================================


def run_training(
    cfg: TrainConfig,
    parquet_path: str,
    num_classes: int,
    run_dir: Path,
    idx_to_centroid: Dict[int, Tuple[float, float]],
    distance_km: Optional[torch.Tensor] = None,
    *,
    num_proxies: int = 1,
    num_classes_r6: int = 0,
    r7_parent: Optional[torch.Tensor] = None,
) -> TrainSummary:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    hierarchical_enabled = bool(getattr(cfg, "hierarchical_enabled", False))
    if hierarchical_enabled:
        if int(num_classes_r6) <= 0:
            raise RuntimeError("hierarchical_enabled=True but num_classes_r6 is missing/invalid")
        if r7_parent is None:
            raise RuntimeError("hierarchical_enabled=True but r7_parent mapping is missing")

    num_proxies = int(num_proxies or 0)
    proxy_loss_enabled = bool(getattr(cfg, "proxy_loss_enabled", False)) and float(
        getattr(cfg, "proxy_loss_weight", 0.0)
    ) > 0.0
    proxy_cols: List[str] = list(getattr(cfg, "proxy_cols", []) or [])

    if proxy_loss_enabled and num_proxies <= 0:
        raise RuntimeError("proxy_loss_enabled=True but num_proxies==0 (run.py should pass len(cfg.proxy_cols)).")
    if proxy_loss_enabled and len(proxy_cols) != num_proxies:
        raise RuntimeError(f"proxy_cols length mismatch: len(proxy_cols)={len(proxy_cols)} num_proxies={num_proxies}")

    dump_k = int(getattr(cfg, "dump_topk", 10))
    dump_k = max(dump_k, 10)
    dump_k = max(dump_k, int(getattr(cfg, "topk", 5)))

    proxy_loss_weight = float(getattr(cfg, "proxy_loss_weight", 0.35)) if proxy_loss_enabled else 0.0

    hard_cfg = HardNegConfig(
        enabled=bool(getattr(cfg, "hardneg_enabled", False)),
        threshold_km=float(getattr(cfg, "hardneg_threshold_km", 500.0)),
        boost=float(getattr(cfg, "hardneg_boost", 4.0)),
        max_pool=int(getattr(cfg, "hardneg_max_pool", 20000)),
        min_count_to_enable=int(getattr(cfg, "hardneg_min_count", 100)),
    )
    hard_pool_path = run_dir / "hardneg_pool.json"
    hard_pool_ids = load_pool(hard_pool_path)

    model = MultiScaleCNN(
        num_classes=int(num_classes),
        dropout=float(cfg.dropout),
        num_proxies=int(num_proxies),
        num_classes_r6=int(num_classes_r6),
        hierarchical_enabled=bool(hierarchical_enabled),
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    class_w = compute_class_weights_from_parquet(
        parquet_path,
        split="train",
        num_classes=int(num_classes),
        label_column="label_idx",
    ).to(device)
    ce = nn.CrossEntropyLoss(weight=class_w)

    ce_r6: Optional[nn.CrossEntropyLoss] = None
    if hierarchical_enabled:
        class_w_r6 = compute_class_weights_from_parquet(
            parquet_path,
            split="train",
            num_classes=int(num_classes_r6),
            label_column="label_r6_idx",
        ).to(device)
        ce_r6 = nn.CrossEntropyLoss(weight=class_w_r6)

    geo_loss: Optional[GeoSoftTargetLoss] = None
    if bool(getattr(cfg, "geo_loss_enabled", False)):
        if distance_km is None:
            raise RuntimeError("geo_loss_enabled=True but distance_km is None")
        geo_loss = GeoSoftTargetLoss(
            distance_km.to(device),
            tau_km=float(getattr(cfg, "geo_tau_km", 1.8)),
        )

    proxy_loss_fn: Optional[nn.Module] = None
    if proxy_loss_enabled:
        proxy_loss_fn = nn.SmoothL1Loss(reduction="mean")

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"
    metrics_csv = run_dir / "metrics.csv"

    r7_parent_t = r7_parent.to(device) if (hierarchical_enabled and r7_parent is not None) else None

    best_acc = -1.0
    best_epoch = -1
    best_size = -1
    best_p90 = float("inf")
    best_median = float("inf")

    best_ckpt_metric = float("inf") if cfg.early_stop_mode == "min" else -float("inf")
    best_es_value = float("inf") if cfg.early_stop_mode == "min" else -float("inf")
    es_best = float("inf") if cfg.early_stop_mode == "min" else -float("inf")
    es_bad_epochs = 0

    use_cuda = device == "cuda"
    pin = bool(use_cuda)
    persistent = bool(cfg.num_workers and int(cfg.num_workers) > 0)

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "epoch",
                "image_size",
                "batch_size",
                "train_loss",
                "train_loss_main",
                "train_loss_proxy",
                "proxy_weight",
                "val_acc",
                "val_top5",
                "val_top10",
                "geo_mean_km",
                "geo_median_km",
                "geo_p90_km",
                "geo_p95_km",
                "geo_far_200_rate",
                "geo_far_500_rate",
                "early_stop_metric",
                "hardneg_pool_size",
                "hardneg_added_this_epoch",
            ]
        )

        epoch_pbar = tqdm(range(1, int(cfg.epochs) + 1), desc="Training", dynamic_ncols=True)

        for epoch in epoch_pbar:
            size_idx = (epoch - 1) % len(cfg.image_sizes)
            image_size = int(cfg.image_sizes[size_idx])
            batch_size = int(cfg.batch_sizes[size_idx])

            train_ds = GeoDataset(
                parquet_path,
                "train",
                image_size,
                hierarchical_enabled=hierarchical_enabled,
                proxies_enabled=bool(proxy_loss_enabled),
                proxy_columns=proxy_cols if proxy_loss_enabled else None,
            )
            val_ds = GeoDataset(
                parquet_path,
                "val",
                image_size,
                hierarchical_enabled=hierarchical_enabled,
                proxies_enabled=bool(proxy_loss_enabled),
                proxy_columns=proxy_cols if proxy_loss_enabled else None,
            )

            sampler = None
            shuffle = True
            if hard_cfg.enabled and len(hard_pool_ids) >= hard_cfg.min_count_to_enable:
                all_ids = train_ds.all_ids()
                w_by_id = build_sample_weights(all_ids, hard_pool_ids, hard_cfg)
                weights = torch.tensor([w_by_id.get(sid, 1.0) for sid in all_ids], dtype=torch.double)
                sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
                shuffle = False

            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=int(cfg.num_workers),
                pin_memory=pin,
                persistent_workers=persistent,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(cfg.num_workers),
                pin_memory=pin,
                persistent_workers=persistent,
            )

            # --------------------
            # Train
            # --------------------
            model.train()
            total_loss = 0.0
            total_main = 0.0
            total_proxy = 0.0
            n_batches = 0
            n_proxy_batches = 0

            train_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch:02d}/{int(cfg.epochs)} [train] size={image_size} bs={batch_size}",
                leave=False,
                dynamic_ncols=True,
            )

            for batch in train_pbar:
                if hierarchical_enabled:
                    # Expected minimal layout:
                    # (x, y6, y7, lat, lon, ..., img_id)
                    x = batch[0]
                    y6 = batch[1]
                    y = batch[2]
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    y6 = y6.to(device, non_blocking=True)
                else:
                    # Expected minimal layout:
                    # (x, y7, lat, lon, ..., img_id)
                    x = batch[0]
                    y = batch[1]
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                proxy_t = None
                proxy_m = None
                if proxy_loss_enabled:
                    # Dataset appends (proxy_targets, proxy_mask) at the end of each sample.
                    proxy_t = batch[-2].to(device, non_blocking=True).float()
                    proxy_m = batch[-1].to(device, non_blocking=True).float()

                optimizer.zero_grad(set_to_none=True)
                out = model(x)

                proxy_pred = None

                if hierarchical_enabled:
                    if not (isinstance(out, tuple) and len(out) >= 2):
                        raise RuntimeError("hierarchical_enabled=True but model did not return (logits_r6, logits_r7)")
                    logits_r6, logits_r7 = out[0], out[1]
                    if proxy_loss_enabled:
                        if len(out) < 3:
                            raise RuntimeError("proxy_loss_enabled=True but model did not return proxy head output")
                        proxy_pred = out[2]

                    if ce_r6 is None:
                        raise RuntimeError("hierarchical_enabled=True but ce_r6 is None")
                    if r7_parent_t is None:
                        raise RuntimeError("hierarchical_enabled=True but r7_parent_t is None")

                    loss_r6 = ce_r6(logits_r6, y6)

                    allowed_mask = r7_parent_t.unsqueeze(0).eq(y6.unsqueeze(1))  # [B,C]
                    logits_r7_masked = logits_r7.masked_fill(~allowed_mask, -1e9)

                    if geo_loss is None:
                        loss_r7 = ce(logits_r7_masked, y)
                    else:
                        loss_geo = geo_loss(logits_r7_masked, y, allowed_mask=allowed_mask)
                        loss_ce = ce(logits_r7_masked, y)
                        loss_r7 = (1.0 - float(cfg.geo_mix_ce)) * loss_geo + float(cfg.geo_mix_ce) * loss_ce

                    loss_main = loss_r6 + loss_r7

                    _debug_first_batch(
                        epoch=epoch,
                        n_batches=n_batches,
                        hierarchical_enabled=True,
                        loss=loss_main,
                        logits_r6=logits_r6,
                        logits_r7=logits_r7_masked,
                    )
                else:
                    if isinstance(out, tuple):
                        logits = out[0]
                        if proxy_loss_enabled:
                            if len(out) < 2:
                                raise RuntimeError("proxy_loss_enabled=True but model did not return proxy head output")
                            proxy_pred = out[1]
                    else:
                        logits = out

                    if geo_loss is None:
                        loss_main = ce(logits, y)
                    else:
                        loss_geo = geo_loss(logits, y)
                        loss_ce = ce(logits, y)
                        loss_main = (1.0 - float(cfg.geo_mix_ce)) * loss_geo + float(cfg.geo_mix_ce) * loss_ce

                    _debug_first_batch(
                        epoch=epoch,
                        n_batches=n_batches,
                        hierarchical_enabled=False,
                        loss=loss_main,
                        logits_r7=logits,
                    )

                loss_proxy = torch.tensor(0.0, device=device)
                if proxy_loss_enabled:
                    if proxy_loss_fn is None:
                        raise RuntimeError("proxy_loss_enabled=True but proxy_loss_fn is None")
                    if proxy_pred is None or proxy_t is None:
                        raise RuntimeError("proxy_loss_enabled=True but missing proxy_pred/proxy_t")
                    loss_proxy = _masked_smooth_l1(proxy_pred, proxy_t, proxy_m)

                loss = loss_main + float(proxy_loss_weight) * loss_proxy

                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_main += float(loss_main.item())
                if proxy_loss_enabled:
                    total_proxy += float(loss_proxy.detach().item())
                    n_proxy_batches += 1
                n_batches += 1

                train_pbar.set_postfix(
                    {
                        "loss": f"{(total_loss / max(1, n_batches)):.4f}",
                        "main": f"{(total_main / max(1, n_batches)):.4f}",
                    }
                )

            train_loss = total_loss / max(1, n_batches)
            train_loss_main = total_main / max(1, n_batches)
            train_loss_proxy = (total_proxy / max(1, n_proxy_batches)) if n_proxy_batches else 0.0

            # --------------------
            # Val
            # --------------------
            model.eval()
            all_logits: list[torch.Tensor] = []
            all_preds: list[torch.Tensor] = []
            all_targets: list[torch.Tensor] = []
            all_lat: list[torch.Tensor] = []
            all_lon: list[torch.Tensor] = []
            all_img_ids: list[str] = []

            val_pbar = tqdm(
                val_loader,
                desc=f"Epoch {epoch:02d}/{int(cfg.epochs)} [val]",
                leave=False,
                dynamic_ncols=True,
            )

            with torch.no_grad():
                for batch in val_pbar:
                    if hierarchical_enabled:
                        x = batch[0]
                        y6 = batch[1]
                        y = batch[2]
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        y6 = y6.to(device, non_blocking=True)

                        B = int(y.numel())
                        lat, lon, img_id = _extract_lat_lon_img_id(batch, B=B)
                        lat = lat.to(device, non_blocking=True)
                        lon = lon.to(device, non_blocking=True)

                        out = model(x)
                        if not (isinstance(out, tuple) and len(out) >= 2):
                            raise RuntimeError(
                                "hierarchical_enabled=True but model did not return (logits_r6, logits_r7)"
                            )
                        logits_r6, logits_r7 = out[0], out[1]

                        if r7_parent_t is None:
                            raise RuntimeError("hierarchical_enabled=True but r7_parent_t is None")

                        _pred_r6, preds, logits_masked = hierarchical_predict(
                            logits_r6,
                            logits_r7,
                            r7_parent_t,
                        )
                        logits = logits_masked
                    else:
                        x = batch[0]
                        y = batch[1]
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)

                        B = int(y.numel())
                        lat, lon, img_id = _extract_lat_lon_img_id(batch, B=B)
                        lat = lat.to(device, non_blocking=True)
                        lon = lon.to(device, non_blocking=True)

                        out = model(x)
                        logits = out[0] if isinstance(out, tuple) else out
                        preds = torch.argmax(logits, dim=1)

                    all_logits.append(logits.detach().cpu())
                    all_preds.append(preds.detach().cpu())
                    all_targets.append(y.detach().cpu())
                    all_lat.append(lat.reshape(-1).detach().cpu())
                    all_lon.append(lon.reshape(-1).detach().cpu())
                    all_img_ids.extend([str(s) for s in img_id])

                    batch_acc = float((preds == y.detach().cpu()).float().mean().item())
                    val_pbar.set_postfix(acc=f"{batch_acc:.3f}")

            logits_all = (
                torch.cat(all_logits, dim=0) if all_logits else torch.empty((0, num_classes), dtype=torch.float32)
            )
            preds = torch.cat(all_preds, dim=0) if all_preds else torch.empty((0,), dtype=torch.long)
            targets = torch.cat(all_targets, dim=0) if all_targets else torch.empty((0,), dtype=torch.long)

            # ---- Robust lat/lon handling (proxies may change batch layout) ----
            N = int(targets.numel())
            if all_lat and all_lon:
                lat_true = torch.cat(all_lat, dim=0).reshape(-1).numpy()
                lon_true = torch.cat(all_lon, dim=0).reshape(-1).numpy()
            else:
                lat_true = np.array([], dtype=np.float32)
                lon_true = np.array([], dtype=np.float32)

            geo_ok = True
            if N > 0:
                if lat_true.size != N or lon_true.size != N:
                    geo_ok = False
                    tqdm.write(
                        "⚠️  Geo metrics skipped: lat/lon missing or mismatch "
                        f"(lat={int(lat_true.size)} lon={int(lon_true.size)} vs y_true={N}). "
                        "This usually means the batch layout changed (e.g. proxies inserted) "
                        "and lat/lon were not extracted correctly."
                    )

            val_acc = _accuracy(preds, targets)
            val_top5 = _topk_accuracy(logits_all, targets, k=5)
            val_top10 = _topk_accuracy(logits_all, targets, k=10)

            if geo_ok:
                kpi = compute_geo_kpi(
                    y_true=targets.numpy() if targets.numel() else np.array([], dtype=np.int64),
                    y_pred=preds.numpy() if preds.numel() else np.array([], dtype=np.int64),
                    lat_true=lat_true,
                    lon_true=lon_true,
                    idx_to_centroid=idx_to_centroid,
                )
            else:
                from .metrics_geo import GeoKPI

                kpi = GeoKPI(
                    mean_km=0.0,
                    median_km=0.0,
                    p90_km=0.0,
                    p95_km=0.0,
                    far_error_rate_200=0.0,
                    far_error_rate_500=0.0,
                )

            hard_added = 0
            if geo_ok and len(all_img_ids) == int(targets.numel()) and int(targets.numel()) > 0:
                val_errors_path = run_dir / "val_errors.parquet"
                dist_km_arr = _dump_val_errors_parquet(
                    val_errors_path,
                    all_img_ids,
                    targets,
                    preds,
                    lat_true,
                    lon_true,
                    idx_to_centroid,
                )

                if hard_cfg.enabled and dist_km_arr.size > 0:
                    new_hard = [
                        all_img_ids[i]
                        for i in range(len(all_img_ids))
                        if float(dist_km_arr[i]) > float(hard_cfg.threshold_km)
                    ]
                    hard_added = len(new_hard)
                    hard_pool_ids = update_pool(hard_pool_ids, new_hard, hard_cfg)
                    save_pool(hard_pool_path, hard_pool_ids)
            else:
                dist_km_arr = np.array([], dtype=np.float64)

            if cfg.early_stop_metric == "val_acc":
                es_value = float(val_acc)
            elif cfg.early_stop_metric == "p90_km":
                es_value = float(kpi.p90_km)
            elif cfg.early_stop_metric == "median_km":
                es_value = float(kpi.median_km)
            else:
                raise ValueError("early_stop_metric must be one of: val_acc, p90_km, median_km")

            w.writerow(
                [
                    epoch,
                    image_size,
                    batch_size,
                    f"{train_loss:.6f}",
                    f"{train_loss_main:.6f}",
                    f"{train_loss_proxy:.6f}",
                    f"{proxy_loss_weight if proxy_loss_enabled else 0.0:.6f}",
                    f"{val_acc:.6f}",
                    f"{val_top5:.6f}",
                    f"{val_top10:.6f}",
                    f"{kpi.mean_km:.6f}",
                    f"{kpi.median_km:.6f}",
                    f"{kpi.p90_km:.6f}",
                    f"{kpi.p95_km:.6f}",
                    f"{kpi.far_error_rate_200:.6f}",
                    f"{kpi.far_error_rate_500:.6f}",
                    f"{es_value:.6f}",
                    int(len(hard_pool_ids)),
                    int(hard_added),
                ]
            )
            f.flush()

            torch.save(model.state_dict(), last_ckpt)

            if float(val_acc) > float(best_acc):
                best_acc = float(val_acc)
                best_epoch = int(epoch)
                best_size = int(image_size)

            if _is_improved(float(es_value), float(best_ckpt_metric), cfg.early_stop_mode, cfg.early_stop_min_delta):
                best_ckpt_metric = float(es_value)
                torch.save(model.state_dict(), best_ckpt)

            if float(kpi.p90_km) < best_p90:
                best_p90 = float(kpi.p90_km)
            if float(kpi.median_km) < best_median:
                best_median = float(kpi.median_km)

            if _is_improved(float(es_value), float(best_es_value), cfg.early_stop_mode, cfg.early_stop_min_delta):
                best_es_value = float(es_value)
                out_path = run_dir / "val_topk.parquet"
                _dump_val_topk_parquet(out_path, all_img_ids, targets, preds, logits_all, k=dump_k)
                tqdm.write(f"💾 Saved val topk dump: {out_path.resolve()} (k={dump_k})")

            tag = "geo" if geo_ok and geo_loss is not None else ("ce" if geo_ok else "no-geo")
            hn_tag = f" hardneg(pool={len(hard_pool_ids)},+{hard_added})" if hard_cfg.enabled else ""
            tqdm.write(
                f"Epoch {epoch:02d} | size={image_size:3d} bs={batch_size:3d} | "
                f"loss={train_loss:.4f} (main={train_loss_main:.4f} proxy={train_loss_proxy:.4f}) | "
                f"val_acc={val_acc:.4f} | top5={val_top5:.4f} top10={val_top10:.4f} | "
                f"median={kpi.median_km:.2f}km p90={kpi.p90_km:.2f}km | "
                f"far200={kpi.far_error_rate_200:.3f} far500={kpi.far_error_rate_500:.3f} | "
                f"{tag}{hn_tag}"
            )

            epoch_pbar.set_postfix(
                loss=f"{train_loss:.4f}",
                acc=f"{val_acc:.4f}",
                top5=f"{val_top5:.4f}",
                median_km=f"{kpi.median_km:.1f}",
                p90_km=f"{kpi.p90_km:.1f}",
            )

            if bool(getattr(cfg, "early_stopping_enabled", False)):
                improved = _is_improved(float(es_value), float(es_best), cfg.early_stop_mode, cfg.early_stop_min_delta)
                if improved:
                    es_best = float(es_value)
                    es_bad_epochs = 0
                else:
                    es_bad_epochs += 1
                    if es_bad_epochs >= int(cfg.early_stop_patience):
                        tqdm.write(
                            f"⏹ Early stopping: no improvement on {cfg.early_stop_metric} "
                            f"for {int(cfg.early_stop_patience)} epochs. Best={es_best:.6f}"
                        )
                        break

    return TrainSummary(
        best_val_acc=float(best_acc),
        best_epoch=int(best_epoch),
        best_image_size=int(best_size),
        device=device,
        metrics_csv=str(metrics_csv),
        best_ckpt=str(best_ckpt),
        last_ckpt=str(last_ckpt),
        best_p90_km=float(best_p90),
        best_median_km=float(best_median),
    )


# ============================================================================
# Orchestration (run once) — kept for backward-compat / CLI usage
# ============================================================================


def _compute_distance_matrix_km(idx_to_centroid: Dict[int, Tuple[float, float]]) -> torch.Tensor:
    c = len(idx_to_centroid)
    d = torch.zeros((c, c), dtype=torch.float32)
    for i in range(c):
        lat1, lon1 = idx_to_centroid[i]
        for j in range(c):
            lat2, lon2 = idx_to_centroid[j]
            d[i, j] = float(haversine_km(lat1, lon1, lat2, lon2))
    return d


def _ensure_geo_distance_matrix(run_dir: Path, idx_to_centroid: Dict[int, Tuple[float, float]]) -> torch.Tensor:
    c = len(idx_to_centroid)
    d = _compute_distance_matrix_km(idx_to_centroid)
    if d.shape != (c, c):
        raise RuntimeError(f"Distance matrix shape mismatch: D={tuple(d.shape)} expected=({c},{c}).")
    dist_path = run_dir / "dist_km.pt"
    torch.save(d, dist_path)
    return d


def _load_labels_json(path: Path) -> Tuple[list[str], Dict[int, Tuple[float, float]], Optional[Dict[int, int]], int]:
    obj = json.loads(path.read_text(encoding="utf-8"))

    labels_list = obj.get("labels") or obj.get("h3_ids")
    if not isinstance(labels_list, list) or not labels_list:
        raise RuntimeError(f"Invalid labels file: missing labels list: {path}")

    labels_list = [str(h) for h in labels_list]

    import h3 as _h3

    idx_to_centroid: Dict[int, Tuple[float, float]] = {}
    for i, h3_id in enumerate(labels_list):
        lat, lon = _h3.cell_to_latlng(str(h3_id))
        idx_to_centroid[int(i)] = (float(lat), float(lon))

    r7_to_r6_raw = obj.get("r7_to_r6")
    r7_to_r6: Optional[Dict[int, int]] = None
    if isinstance(r7_to_r6_raw, dict) and r7_to_r6_raw:
        r7_to_r6 = {int(k): int(v) for k, v in r7_to_r6_raw.items()}

    parent_res = int(obj.get("parent_res", 6))
    return labels_list, idx_to_centroid, r7_to_r6, parent_res


def run_once(cfg: Optional[TrainConfig] = None) -> Path:
    cfg = cfg or TrainConfig()

    runs_dir = p("runs")
    ensure_dir(runs_dir)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs_dir / ts
    ensure_dir(run_dir)

    parquet_full = p(*str(cfg.parquet_index).split("/"))
    parquet_kept = parquet_full.parent / "images_kept.parquet"
    parquet_train = parquet_kept if parquet_kept.exists() else parquet_full

    if not parquet_train.exists():
        raise FileNotFoundError(f"Missing {parquet_train}. Run: make rebuild")

    labels_path = parquet_full.parent / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}. Run: make rebuild")

    df = pd.read_parquet(parquet_train)
    required_cols = {"id", "lat", "lon", "path", "h3_id", "label_idx", "split"}
    missing = sorted(list(required_cols - set(df.columns)))

    proxy_loss_weight = float(getattr(cfg, "proxy_loss_weight", 0.0))
    proxy_loss_enabled = bool(getattr(cfg, "proxy_loss_enabled", False)) and (proxy_loss_weight > 0.0)
    proxy_cols = list(getattr(cfg, "proxy_cols", [])) or [
        "proxy_elev_log1p_z",
        "proxy_pop_log1p_z",
        "proxy_water_frac",
        "proxy_built_frac",
        "proxy_coastal_score",
    ]
    if proxy_loss_enabled:
        missing_proxy = sorted([c for c in proxy_cols if c not in df.columns])
        if missing_proxy:
            raise RuntimeError(f"proxy_loss_enabled=True but {parquet_train} missing columns: {missing_proxy}")

    if missing:
        raise RuntimeError(f"{parquet_train} missing columns: {missing}")

    hierarchical_enabled = bool(getattr(cfg, "hierarchical_enabled", False))
    if hierarchical_enabled and "label_r6_idx" not in df.columns:
        raise RuntimeError("hierarchical_enabled=True but parquet is missing label_r6_idx. Rebuild parquet.")

    labels_list, idx_to_centroid, r7_to_r6, parent_res = _load_labels_json(labels_path)

    c = int(len(labels_list))
    if c <= 1:
        raise RuntimeError(f"Num classes looks wrong (C={c}). Check labels.json + parquet.")
    mx = int(df["label_idx"].max()) if len(df) else -1
    if mx >= c:
        raise RuntimeError(f"label_idx max={mx} but num_classes={c}. labels.json and parquet are inconsistent.")

    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "labels.json").write_text(labels_path.read_text(encoding="utf-8"), encoding="utf-8")
    shutil.copy2(parquet_train, run_dir / "images_train.parquet")

    dist_km: Optional[torch.Tensor] = None
    if bool(getattr(cfg, "geo_loss_enabled", False)):
        dist_km = _ensure_geo_distance_matrix(run_dir, idx_to_centroid)

    num_classes_r6 = 0
    r7_parent_t: Optional[torch.Tensor] = None
    if hierarchical_enabled:
        if r7_to_r6 is None or len(r7_to_r6) != c:
            raise RuntimeError("hierarchical_enabled=True but labels.json missing r7_to_r6 (or wrong size).")

        num_classes_r6 = int(max(r7_to_r6.values()) + 1) if r7_to_r6 else 0
        if num_classes_r6 <= 1:
            raise RuntimeError("hierarchical_enabled=True but num_classes_r6 looks invalid")

        r7_parent = torch.zeros((c,), dtype=torch.long)
        for k, v in r7_to_r6.items():
            r7_parent[int(k)] = int(v)
        r7_parent_t = r7_parent

    summary = run_training(
        cfg,
        parquet_path=str(parquet_train),
        num_classes=c,
        run_dir=run_dir,
        idx_to_centroid=idx_to_centroid,
        distance_km=dist_km if bool(getattr(cfg, "geo_loss_enabled", False)) else None,
        num_classes_r6=num_classes_r6,
        r7_parent=r7_parent_t,
    )

    update_latest_symlink(runs_dir, run_dir)

    meta: Dict[str, Any] = {
        "timestamp": ts,
        "run_dir": str(run_dir),
        "train_parquet": str(parquet_train),
        "labels_path": str(labels_path),
        "num_classes": int(c),
        "label_idx_max": int(mx),
        "hierarchical_enabled": bool(hierarchical_enabled),
        "num_classes_r6": int(num_classes_r6),
        "parent_res": int(parent_res),
        "proxy_loss_enabled": bool(proxy_loss_enabled),
        "proxy_loss_weight": float(proxy_loss_weight),
        "proxy_cols": proxy_cols if proxy_loss_enabled else [],
        "best_val_acc": float(summary.best_val_acc),
        "best_epoch": int(summary.best_epoch),
        "best_image_size": int(summary.best_image_size),
        "best_geo_median_km": float(summary.best_median_km),
        "best_geo_p90_km": float(summary.best_p90_km),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    tqdm.write("\nDone.")
    tqdm.write(f"Run dir: {run_dir}")
    tqdm.write(f"Latest: {runs_dir / 'latest'}")
    tqdm.write(f"Metrics: {run_dir / 'metrics.csv'}")
    tqdm.write(f"Best ckpt: {run_dir / 'checkpoints' / 'best.pt'}")
    tqdm.write(f"Last ckpt: {run_dir / 'checkpoints' / 'last.pt'}")

    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Train loop runner (one run).")
    ap.add_argument("--run", action="store_true", help="Run one training session.")
    args = ap.parse_args()

    if args.run:
        run_once()
        return

    run_once()


if __name__ == "__main__":
    main()