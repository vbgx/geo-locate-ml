from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .paths import p
from .geo import hierarchical_predict
from .modeling import MultiScaleCNN
from .data import build_transform
from .labels import LabelSpace


def load_best_metadata():
    meta_path = p("models", "best.json")
    if not meta_path.exists():
        raise FileNotFoundError("models/best.json not found. Run: python -m src.run")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta


def pick_random_from_parquet(parquet_path: str) -> Path:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    row = df.sample(n=1, random_state=random.randint(0, 10_000)).iloc[0]
    return Path(row["path"]), str(row.get("h3_id")), str(row.get("split"))


def _ckpt_has_proxy_head(state: dict) -> bool:
    return any(str(k).startswith("proxy_head.") for k in state.keys())


def _ckpt_has_r6_head(state: dict) -> bool:
    return any(str(k).startswith("fc_r6.") for k in state.keys())


@torch.no_grad()
def _predict_logits(model, device, pil_img, size: int):
    x = build_transform(size)(pil_img).unsqueeze(0).to(device)
    return model(x)


def topk(classes, probs, k=5):
    k = min(k, len(classes))
    vals, idxs = torch.topk(probs, k=k)
    return [(classes[int(i)], float(v)) for v, i in zip(vals, idxs)]


def build_class_prior_from_parquet(parquet_path: str, labels: LabelSpace) -> torch.Tensor:
    """
    Build a class prior aligned with labels.h3_ids from the TRAIN split of the parquet.
    Returns a torch Tensor of shape [C] that sums to 1.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["h3_id", "split"])
    df = df[df["split"] == "train"]

    counts = df["h3_id"].value_counts(dropna=False).to_dict()
    C = len(labels.h3_ids)

    prior = torch.zeros(C, dtype=torch.float64)
    # labels.h3_ids is the canonical ordering
    for i, h3 in enumerate(labels.h3_ids):
        prior[i] = float(counts.get(h3, 0.0))

    s = float(prior.sum().item())
    if s <= 0:
        # fallback: uniform
        prior = torch.ones(C, dtype=torch.float64) / float(C)
        return prior.to(dtype=torch.float32)

    prior = prior / s
    # avoid log(0) later
    eps = 1e-12
    prior = torch.clamp(prior, min=eps)
    prior = prior / prior.sum()
    return prior.to(dtype=torch.float32)


def rerank_topk_prior_only(
    probs: torch.Tensor,
    class_prior: torch.Tensor,
    k: int,
    beta: float = 0.35,
):
    """
    Rerank TopK candidates by: score = log(prob) + beta * log(prior)
    Returns list of tuples: (h3_id, prob, score)
    """
    k = min(k, probs.numel())
    vals, idxs = torch.topk(probs, k=k)

    # candidates
    cand_probs = vals
    cand_idxs = idxs

    # compute rerank scores
    eps = 1e-12
    logp = torch.log(torch.clamp(cand_probs, min=eps))
    cand_prior = class_prior[cand_idxs]
    logprior = torch.log(torch.clamp(cand_prior, min=eps))

    scores = logp + float(beta) * logprior
    order = torch.argsort(scores, descending=True)

    out = []
    for j in order.tolist():
        out.append((int(cand_idxs[j]), float(cand_probs[j]), float(scores[j])))
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("image_path", nargs="?", default=None)
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--sizes", default="64,128,192")
    ap.add_argument("--topk", type=int, default=5)

    # rerank options
    ap.add_argument("--rerank", action="store_true", help="Rerank TopK using a train-split class prior (prior-only).")
    ap.add_argument("--beta", type=float, default=0.35, help="Weight of log-prior in rerank score.")

    args = ap.parse_args()

    meta = load_best_metadata()
    best_pt = p("models", "best.pt")
    labels_path = Path(meta["labels_path"])
    parquet_path = meta["parquet_path"]

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

    if args.image_path is None:
        img_path, true_h3, split = pick_random_from_parquet(parquet_path)
        print(f"\nPicked: {img_path}")
        print(f"True h3: {true_h3} (split={split})")
    else:
        img_path = Path(args.image_path)
        true_h3 = None

    pil = Image.open(img_path).convert("RGB")

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]

    if hierarchical_enabled:
        r7_parent = torch.tensor([labels.r7_to_r6[i] for i in range(len(labels.h3_ids))], dtype=torch.long).to(device)

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

        pred_r6, _pred_r7, masked = hierarchical_predict(logits_r6, logits_r7, r7_parent)
        probs = F.softmax(masked, dim=1)[0]
        print(f"Pred r6: {labels.idx_to_h3_r6[int(pred_r6.item())]}")
    else:
        if args.ensemble:
            logits_list = []
            for s in sizes:
                out = _predict_logits(model, device, pil, s)
                if isinstance(out, tuple):
                    out = out[0]
                logits_list.append(out)
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
            print(f"\nEnsemble sizes: {sizes}")
        else:
            logits = _predict_logits(model, device, pil, sizes[-1])
            print(f"\nSingle size: {sizes[-1]}")

        if isinstance(logits, tuple):
            logits = logits[0]
        probs = F.softmax(logits, dim=1)[0]

    print("\nTop-k (model probs):")
    out = topk(labels.h3_ids, probs, k=args.topk)
    for i, (h, pr) in enumerate(out, start=1):
        print(f"{i:>2}. {h}   p={pr:.4f}")

    if args.rerank:
        print("\nBuilding train prior from parquet (split=train)...")
        class_prior = build_class_prior_from_parquet(parquet_path, labels)

        reranked = rerank_topk_prior_only(
            probs=probs.cpu(),
            class_prior=class_prior.cpu(),
            k=args.topk,
            beta=float(args.beta),
        )

        print(f"\nTop-k (reranked, beta={args.beta:.3f}):")
        for rank, (idx, pr, score) in enumerate(reranked, start=1):
            h3 = labels.h3_ids[idx]
            print(f"{rank:>2}. {h3}   p={pr:.4f}   score={score:.4f}")


if __name__ == "__main__":
    main()
