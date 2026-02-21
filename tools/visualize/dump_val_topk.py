from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import GeoDataset
from src.hierarchy import hierarchical_predict
from src.labels import LabelSpace
from src.model import MultiScaleCNN
from src.paths import p


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out", default=str(p("runs", "latest", "val_topk.parquet")))
    args = ap.parse_args()

    meta_path = p("models", "best.json")
    if not meta_path.exists():
        raise FileNotFoundError("models/best.json not found.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    parquet_path = meta["parquet_path"]
    labels_path = Path(meta["labels_path"])
    dropout = float(meta.get("dropout", 0.30))

    labels = LabelSpace.from_json(labels_path.read_text(encoding="utf-8"))
    num_classes = len(labels.h3_ids)

    ckpt = p("models", "best.pt")
    if not ckpt.exists():
        raise FileNotFoundError("models/best.pt not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hierarchical_enabled = bool(meta.get("hierarchical_enabled", False))

    ds = GeoDataset(parquet_path, "val", args.image_size, hierarchical_enabled=hierarchical_enabled)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device == "cuda"),
    )

    model = MultiScaleCNN(
        num_classes=num_classes,
        dropout=dropout,
        num_classes_r6=len(labels.h3_ids_r6),
        hierarchical_enabled=hierarchical_enabled,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    r7_parent = None
    if hierarchical_enabled:
        r7_parent = torch.tensor([labels.r7_to_r6[i] for i in range(len(labels.h3_ids))], dtype=torch.long).to(device)

    image_ids: list[str] = []
    true_idx: list[int] = []
    pred_idx: list[int] = []
    topk_idx_rows: list[np.ndarray] = []
    topk_logit_rows: list[np.ndarray] = []

    k = int(args.k)

    for batch in dl:
        # dataset returns x,y,lat,lon (and optionally image_id / proxy) or hierarchical variant
        if hierarchical_enabled:
            if len(batch) == 6:
                x, y6, y, _lat, _lon, ids = batch
            else:
                x, y6, y, _lat, _lon, ids, *_ = batch
        else:
            if len(batch) == 4:
                x, y, _lat, _lon = batch
                # can't dump without image_id -> fallback to row index
                ids = [None] * x.size(0)
            elif len(batch) == 5:
                x, y, _lat, _lon, ids = batch
            else:
                # if proxies enabled: x,y,lat,lon,image_id,proxy
                x, y, _lat, _lon, ids, *_ = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        if hierarchical_enabled:
            if isinstance(out, tuple) and len(out) == 3:
                logits_r6, logits_r7, _proxy = out
            else:
                logits_r6, logits_r7 = out
            _pred_r6, preds, logits_masked = hierarchical_predict(logits_r6, logits_r7, r7_parent)
            logits = logits_masked
        else:
            logits = out[0] if isinstance(out, tuple) else out
            preds = torch.argmax(logits, dim=1)

        kk = min(k, logits.size(1))
        vals, idxs = torch.topk(logits, k=kk, dim=1)

        for i in range(y.size(0)):
            image_ids.append(str(ids[i]) if ids[i] is not None else f"val_{len(image_ids):08d}")
            true_idx.append(int(y[i].item()))
            pred_idx.append(int(preds[i].item()))
            topk_idx_rows.append(idxs[i].detach().cpu().numpy().astype(np.int64))
            topk_logit_rows.append(vals[i].detach().cpu().numpy().astype(np.float64))

    # build dataframe
    df = pd.DataFrame(
        {
            "image_id": image_ids,
            "true_idx": true_idx,
            "pred_idx": pred_idx,
        }
    )

    # expand topk columns
    kk = len(topk_idx_rows[0]) if topk_idx_rows else 0
    topk_idx_arr = np.stack(topk_idx_rows, axis=0) if kk else np.zeros((0, 0), dtype=np.int64)
    topk_logit_arr = np.stack(topk_logit_rows, axis=0) if kk else np.zeros((0, 0), dtype=np.float64)

    for j in range(kk):
        df[f"top{j+1}_idx"] = topk_idx_arr[:, j]
        df[f"top{j+1}_logit"] = topk_logit_arr[:, j]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"✅ wrote {out_path} rows={len(df)} k={kk}")


if __name__ == "__main__":
    main()
