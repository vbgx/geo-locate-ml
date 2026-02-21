from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from .class_weights import compute_class_weights_from_parquet
from .dataset import GeoDataset
from .geo import haversine_km
from .hierarchy import hierarchical_predict, mask_r7_logits
from .geo_loss import GeoSoftTargetLoss
from .hardneg import HardNegConfig, build_sample_weights, load_pool, save_pool, update_pool
from .metrics_geo import compute_geo_kpi
from .model import MultiScaleCNN


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
        return current < (best - min_delta)
    if mode == "max":
        return current > (best + min_delta)
    raise ValueError("mode must be 'min' or 'max'")


def _dump_val_topk_parquet(
    out_path: Path,
    image_ids: list[str],
    targets: torch.Tensor,
    preds: torch.Tensor,
    logits_all: torch.Tensor,
    k: int,
) -> None:
    import pandas as pd

    if logits_all.numel() == 0:
        return

    N, C = logits_all.shape
    k = min(int(k), int(C))
    topk_vals, topk_idxs = torch.topk(logits_all, k=k, dim=1)

    if len(image_ids) != int(N):
        image_ids = [f"val_{i:08d}" for i in range(int(N))]

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
    import pandas as pd

    N = len(image_ids)
    dist = np.zeros((N,), dtype=np.float64)

    for i in range(N):
        yp = int(y_pred[i])
        plat, plon = idx_to_centroid[int(yp)]
        dist[i] = haversine_km(float(lat_true[i]), float(lon_true[i]), float(plat), float(plon))

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


def run_training(
    cfg,
    parquet_path: str,
    num_classes: int,
    run_dir: Path,
    idx_to_centroid: Dict[int, Tuple[float, float]],
    distance_km: Optional[torch.Tensor] = None,
    *,
    num_classes_r6: int = 0,
    r7_parent: Optional[torch.Tensor] = None,
) -> TrainSummary:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    dump_k = int(getattr(cfg, "dump_topk", 10))
    dump_k = max(dump_k, 10)
    dump_k = max(dump_k, int(getattr(cfg, "topk", 5)))

    proxy_loss_weight = float(getattr(cfg, "proxy_loss_weight", 0.35))  # kept for metrics compat

    hierarchical_enabled = bool(getattr(cfg, "hierarchical_enabled", False))
    if hierarchical_enabled:
        if int(num_classes_r6) <= 0:
            raise RuntimeError("hierarchical_enabled=True but num_classes_r6 is missing/invalid")
        if r7_parent is None:
            raise RuntimeError("hierarchical_enabled=True but r7_parent mapping is missing")

    hard_cfg = HardNegConfig(
        enabled=bool(getattr(cfg, "hardneg_enabled", False)),
        threshold_km=float(getattr(cfg, "hardneg_threshold_km", 500.0)),
        boost=float(getattr(cfg, "hardneg_boost", 4.0)),
        max_pool=int(getattr(cfg, "hardneg_max_pool", 20000)),
        min_count_to_enable=int(getattr(cfg, "hardneg_min_count", 100)),
    )
    hard_pool_path = run_dir / "hardneg_pool.json"
    hard_pool_ids = load_pool(hard_pool_path)

    # NOTE:
    # We intentionally DO NOT pass h3_features_path to GeoDataset anymore.
    # Dataset is processed-only and may not support proxy features at all.
    # Model still supports proxies via num_proxies=0 (disabled).

    num_proxies = 0
    model = MultiScaleCNN(
        num_classes=num_classes,
        dropout=cfg.dropout,
        num_proxies=num_proxies,
        num_classes_r6=int(num_classes_r6),
        hierarchical_enabled=hierarchical_enabled,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    class_w = compute_class_weights_from_parquet(
        parquet_path,
        split="train",
        num_classes=num_classes,
        label_column="label_idx",
    ).to(device)
    ce = nn.CrossEntropyLoss(weight=class_w)

    ce_r6 = None
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
        geo_loss = GeoSoftTargetLoss(distance_km.to(device), tau_km=float(getattr(cfg, "geo_tau_km", 1.8)))

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

    es_best = float("inf") if cfg.early_stop_mode == "min" else -float("inf")
    es_bad_epochs = 0
    best_es_value = float("inf") if cfg.early_stop_mode == "min" else -float("inf")

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

        epoch_pbar = tqdm(range(1, cfg.epochs + 1), desc="Training", dynamic_ncols=True)

        for epoch in epoch_pbar:
            size_idx = (epoch - 1) % len(cfg.image_sizes)
            image_size = int(cfg.image_sizes[size_idx])
            batch_size = int(cfg.batch_sizes[size_idx])

            # IMPORTANT: no h3_features_path kwarg here
            train_ds = GeoDataset(
                parquet_path,
                "train",
                image_size,
                hierarchical_enabled=hierarchical_enabled,
            )
            val_ds = GeoDataset(
                parquet_path,
                "val",
                image_size,
                hierarchical_enabled=hierarchical_enabled,
            )

            sampler = None
            shuffle = True
            if hard_cfg.enabled and len(hard_pool_ids) >= hard_cfg.min_count_to_enable:
                all_ids = train_ds.all_ids() if hasattr(train_ds, "all_ids") else list(getattr(train_ds, "image_ids"))
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
            total_proxy = 0.0  # always 0 (proxies disabled)
            n_batches = 0
            n_proxy_batches = 0

            train_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch:02d}/{cfg.epochs} [train] size={image_size} bs={batch_size}",
                leave=False,
                dynamic_ncols=True,
            )

            for batch in train_pbar:
                if hierarchical_enabled:
                    # expected: (x, y6, y7, lat, lon, img_id) (proxy ignored)
                    x, y6, y, _lat, _lon, _img_id = batch[:6]
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    y6 = y6.to(device, non_blocking=True)
                else:
                    # expected: (x, y, lat, lon, img_id)
                    x, y, _lat, _lon, _img_id = batch[:5]
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                out = model(x)

                if hierarchical_enabled:
                    # model returns (logits_r6, logits_r7) when proxies disabled
                    if isinstance(out, tuple) and len(out) >= 2:
                        logits_r6, logits_r7 = out[0], out[1]
                    else:
                        raise RuntimeError("hierarchical_enabled=True but model did not return (logits_r6, logits_r7)")

                    loss_r6 = ce_r6(logits_r6, y6) if ce_r6 is not None else 0.0
                    if r7_parent_t is None:
                        raise RuntimeError("hierarchical_enabled=True but r7_parent_t is None")
                    logits_r7_masked = mask_r7_logits(logits_r7, y6, r7_parent_t)

                    if geo_loss is None:
                        loss_r7 = ce(logits_r7_masked, y)
                    else:
                        loss_geo = geo_loss(logits_r7_masked, y)
                        loss_ce = ce(logits_r7_masked, y)
                        loss_r7 = (1.0 - float(cfg.geo_mix_ce)) * loss_geo + float(cfg.geo_mix_ce) * loss_ce

                    loss_main = loss_r6 + loss_r7
                else:
                    logits = out[0] if isinstance(out, tuple) else out
                    if geo_loss is None:
                        loss_main = ce(logits, y)
                    else:
                        loss_geo = geo_loss(logits, y)
                        loss_ce = ce(logits, y)
                        loss_main = (1.0 - float(cfg.geo_mix_ce)) * loss_geo + float(cfg.geo_mix_ce) * loss_ce

                loss = loss_main
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_main += float(loss_main.item())
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
            all_logits = []
            all_preds = []
            all_targets = []
            all_lat = []
            all_lon = []
            all_img_ids: list[str] = []

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs} [val]", leave=False, dynamic_ncols=True)

            with torch.no_grad():
                for batch in val_pbar:
                    if hierarchical_enabled:
                        x, y6, y, lat, lon, img_id = batch[:6]
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        y6 = y6.to(device, non_blocking=True)

                        out = model(x)
                        if not (isinstance(out, tuple) and len(out) >= 2):
                            raise RuntimeError("hierarchical_enabled=True but model did not return (logits_r6, logits_r7)")
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
                        x, y, lat, lon, img_id = batch[:5]
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)

                        out = model(x)
                        logits = out[0] if isinstance(out, tuple) else out
                        preds = torch.argmax(logits, dim=1)

                    all_logits.append(logits.detach().cpu())
                    all_preds.append(preds.detach().cpu())
                    all_targets.append(y.detach().cpu())
                    all_lat.append(lat.detach().cpu())
                    all_lon.append(lon.detach().cpu())
                    all_img_ids.extend([str(s) for s in img_id])

                    batch_acc = float((preds == y).float().mean().item())
                    val_pbar.set_postfix(acc=f"{batch_acc:.3f}")

            logits_all = torch.cat(all_logits, dim=0) if all_logits else torch.empty((0, num_classes), dtype=torch.float32)
            preds = torch.cat(all_preds, dim=0) if all_preds else torch.empty((0,), dtype=torch.long)
            targets = torch.cat(all_targets, dim=0) if all_targets else torch.empty((0,), dtype=torch.long)

            lat_true = torch.cat(all_lat, dim=0).numpy() if all_lat else np.array([], dtype=np.float32)
            lon_true = torch.cat(all_lon, dim=0).numpy() if all_lon else np.array([], dtype=np.float32)

            val_acc = _accuracy(preds, targets)
            val_top5 = _topk_accuracy(logits_all, targets, k=5)
            val_top10 = _topk_accuracy(logits_all, targets, k=10)

            kpi = compute_geo_kpi(
                y_true=targets.numpy() if targets.numel() else np.array([], dtype=np.int64),
                y_pred=preds.numpy() if preds.numel() else np.array([], dtype=np.int64),
                lat_true=lat_true,
                lon_true=lon_true,
                idx_to_centroid=idx_to_centroid,
            )

            hard_added = 0
            if len(all_img_ids) == int(targets.numel()) and int(targets.numel()) > 0:
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
                    new_hard = [all_img_ids[i] for i in range(len(all_img_ids)) if float(dist_km_arr[i]) > hard_cfg.threshold_km]
                    hard_added = len(new_hard)
                    hard_pool_ids = update_pool(hard_pool_ids, new_hard, hard_cfg)
                    save_pool(hard_pool_path, hard_pool_ids)

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
                    f"{proxy_loss_weight if num_proxies else 0.0:.6f}",
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
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_size = image_size
                torch.save(model.state_dict(), best_ckpt)

            if kpi.p90_km < best_p90:
                best_p90 = float(kpi.p90_km)
            if kpi.median_km < best_median:
                best_median = float(kpi.median_km)

            if _is_improved(es_value, best_es_value, cfg.early_stop_mode, cfg.early_stop_min_delta):
                best_es_value = es_value
                out_path = run_dir / "val_topk.parquet"
                _dump_val_topk_parquet(out_path, all_img_ids, targets, preds, logits_all, k=dump_k)
                tqdm.write(f"💾 Saved val topk dump: {out_path.resolve()} (k={dump_k})")

            tag = "geo" if geo_loss is not None else "ce"
            hn_tag = f" hardneg(pool={len(hard_pool_ids)},+{hard_added})" if hard_cfg.enabled else ""
            tqdm.write(
                f"Epoch {epoch:02d} | size={image_size:3d} bs={batch_size:3d} | "
                f"loss={train_loss:.4f} (main={train_loss_main:.4f}) | "
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

            if cfg.early_stopping_enabled:
                improved = _is_improved(es_value, es_best, cfg.early_stop_mode, cfg.early_stop_min_delta)
                if improved:
                    es_best = es_value
                    es_bad_epochs = 0
                else:
                    es_bad_epochs += 1
                    if es_bad_epochs >= cfg.early_stop_patience:
                        tqdm.write(
                            f"⏹ Early stopping: no improvement on {cfg.early_stop_metric} "
                            f"for {cfg.early_stop_patience} epochs. Best={es_best:.4f}"
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
