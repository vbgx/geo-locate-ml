from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_transform
from .geo import hierarchical_predict
from .labels import LabelSpace
from .modeling import MultiScaleCNN
from .paths import p


def load_best_metadata() -> dict:
    meta_path = p("models", "best.json")
    if not meta_path.exists():
        raise FileNotFoundError("models/best.json not found. Run: python -m src.run")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2) + math.cos(phi1) * math.cos(phi2) * (math.sin(dlmb / 2.0) ** 2)
    return 2.0 * R * math.asin(math.sqrt(a))


def _h3_center_latlng(h3_id: str) -> tuple[float, float]:
    try:
        import h3
    except Exception as e:
        raise RuntimeError(
            "Distance requested but the `h3` package is not available. Install it (e.g. `pip install h3`)."
        ) from e

    if hasattr(h3, "cell_to_latlng"):
        lat, lng = h3.cell_to_latlng(h3_id)
    else:
        lat, lng = h3.h3_to_geo(h3_id)
    return float(lat), float(lng)


def _h3_dist_km(a: str, b: str) -> float:
    lat1, lon1 = _h3_center_latlng(a)
    lat2, lon2 = _h3_center_latlng(b)
    return _haversine_km(lat1, lon1, lat2, lon2)


def pick_random_from_parquet(parquet_path: str, split: str = "any") -> tuple[Path, str, str]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if split and split != "any":
        df = df[df["split"] == split]
        if len(df) == 0:
            raise RuntimeError(f"No rows found for split='{split}' in parquet: {parquet_path}")

    row = df.sample(n=1, random_state=random.randint(0, 10_000)).iloc[0]
    return Path(row["path"]), str(row.get("h3_id")), str(row.get("split"))


def _ckpt_has_proxy_head(state: dict) -> bool:
    return any(str(k).startswith("proxy_head.") for k in state.keys())


def _ckpt_has_r6_head(state: dict) -> bool:
    return any(str(k).startswith("fc_r6.") for k in state.keys())


@torch.no_grad()
def _predict_logits(model: MultiScaleCNN, device: str, pil_img: Image.Image, size: int):
    x = build_transform(int(size))(pil_img).unsqueeze(0).to(device)
    return model(x)


def topk(classes: list[str], probs: torch.Tensor, k: int = 5) -> list[tuple[str, float]]:
    k = min(int(k), int(probs.numel()))
    vals, idxs = torch.topk(probs, k=k)
    return [(classes[int(i)], float(v)) for v, i in zip(vals, idxs)]


def _rank_of_true(probs: torch.Tensor, true_idx: int) -> int:
    # 1-based rank: 1 means best
    # rank = 1 + number of classes with strictly greater probability
    true_p = float(probs[true_idx].item())
    return 1 + int((probs > true_p).sum().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path", nargs="?", default=None)
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--sizes", default="64,128,192")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--split", default="any", choices=["any", "train", "val", "test"])
    ap.add_argument("--dist", action="store_true", help="Print distance (km) to true H3 for each top-k (requires true label).")
    args = ap.parse_args()

    meta = load_best_metadata()
    best_pt = p("models", "best.pt")
    labels_path = Path(meta["labels_path"])
    parquet_path = meta.get("train_parquet") or meta.get("parquet_path")
    if not parquet_path:
        raise RuntimeError("best.json missing train_parquet/parquet_path")

    labels = LabelSpace.from_json(labels_path.read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(best_pt, map_location=device)

    num_proxies = 5 if _ckpt_has_proxy_head(state) else 0
    hierarchical_enabled = bool(meta.get("hierarchical_enabled", False)) or _ckpt_has_r6_head(state)

    model = MultiScaleCNN(
        num_classes=len(labels.h3_ids),
        dropout=float(meta.get("dropout", 0.30)),
        num_proxies=num_proxies,
        num_classes_r6=len(labels.h3_ids_r6),
        hierarchical_enabled=hierarchical_enabled,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    true_h3: Optional[str] = None
    split: Optional[str] = None
    if args.image_path is None:
        img_path, true_h3, split = pick_random_from_parquet(str(parquet_path), split=args.split)
        print(f"\nPicked: {img_path}")
        print(f"True h3: {true_h3} (split={split})")
    else:
        img_path = Path(args.image_path)

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    pil = Image.open(img_path).convert("RGB")
    sizes = [int(s.strip()) for s in str(args.sizes).split(",") if s.strip()]
    if not sizes:
        raise RuntimeError("--sizes must contain at least one integer size, e.g. 128 or 64,128,192")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    pred_r6_h3: Optional[str] = None
    if hierarchical_enabled:
        r7_parent = torch.tensor(
            [int(labels.r7_to_r6[i]) for i in range(len(labels.h3_ids))],
            dtype=torch.long,
            device=device,
        )

        if args.ensemble:
            logits_r6_list = []
            logits_r7_list = []
            for s in sizes:
                out = _predict_logits(model, device, pil, s)
                if isinstance(out, tuple) and len(out) == 3:
                    lr6, lr7, _proxy = out
                else:
                    lr6, lr7 = out
                logits_r6_list.append(lr6)
                logits_r7_list.append(lr7)

            logits_r6 = torch.stack(logits_r6_list, dim=0).mean(dim=0)
            logits_r7 = torch.stack(logits_r7_list, dim=0).mean(dim=0)
            print(f"\nEnsemble sizes: {sizes}")
        else:
            out = _predict_logits(model, device, pil, sizes[-1])
            if isinstance(out, tuple) and len(out) == 3:
                logits_r6, logits_r7, _proxy = out
            else:
                logits_r6, logits_r7 = out
            print(f"\nSingle size: {sizes[-1]}")

        pred_r6, pred_r7, masked = hierarchical_predict(logits_r6, logits_r7, r7_parent)
        probs = F.softmax(masked, dim=1)[0]
        pred_r6_h3 = labels.idx_to_h3_r6[int(pred_r6.item())]
        print(f"Pred r6: {pred_r6_h3}")
    else:
        if args.ensemble:
            logits_list = []
            for s in sizes:
                out = _predict_logits(model, device, pil, s)
                logits = out[0] if isinstance(out, tuple) else out
                logits_list.append(logits)
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
            print(f"\nEnsemble sizes: {sizes}")
        else:
            out = _predict_logits(model, device, pil, sizes[-1])
            logits = out[0] if isinstance(out, tuple) else out
            print(f"\nSingle size: {sizes[-1]}")

        probs = F.softmax(logits, dim=1)[0]

    # ------------------------------------------------------------------
    # True rank diagnostics
    # ------------------------------------------------------------------
    if true_h3 and true_h3 in labels.h3_ids:
        true_idx = labels.h3_ids.index(true_h3)
        true_rank = _rank_of_true(probs, true_idx)
        true_p = float(probs[true_idx].item())
        in_topk = true_rank <= int(args.topk)
        print(f"\nTrue rank: {true_rank}   true_p={true_p:.6f}   in_top{args.topk}={in_topk}")

        if hierarchical_enabled and pred_r6_h3 is not None:
            try:
                true_r6_idx = int(labels.r7_to_r6[true_idx])
                true_r6_h3 = labels.idx_to_h3_r6[true_r6_idx]
                print(f"True r6: {true_r6_h3}   r6_match={true_r6_h3 == pred_r6_h3}")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Print top-k
    # ------------------------------------------------------------------
    print("\nTop-k (model probs):")
    out = topk(labels.h3_ids, probs, k=args.topk)
    for i, (h, pr) in enumerate(out, start=1):
        if args.dist and true_h3:
            try:
                dkm = _h3_dist_km(h, true_h3)
                print(f"{i:>2}. {h}   p={pr:.4f}   dist_km={dkm:.2f}")
            except Exception as e:
                print(f"{i:>2}. {h}   p={pr:.4f}   dist_km=? ({type(e).__name__})")
        else:
            print(f"{i:>2}. {h}   p={pr:.4f}")


if __name__ == "__main__":
    main()
