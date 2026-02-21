#!/usr/bin/env python3
"""
mapview.py — single-entry visualization + split utilities for geo-locate-ml.

Goal
- Replace tools/visualize/* with one script that can:
  1) Generate an interactive folium map HTML (default: ./mapview.html)
     - points layer (optionally colored by split)
     - optional MarkerCluster + HeatMap
     - optional VAL far-errors layer from runs/**/val_topk.parquet
     - optional H3 dominant-class layer (if h3 is installed)
  2) Export splits.jsonl from a dataset parquet
  3) Check split coverage between images.jsonl and splits.jsonl

Typical usage (from repo root)
- Build map:
    python mapview.py build --out mapview.html

- Build map from explicit paths:
    python mapview.py build \
      --images-jsonl data/index/images.jsonl \
      --splits-jsonl data/index/splits.jsonl \
      --labels-json data/index/labels.json \
      --val-topk runs/latest/val_topk.parquet \
      --out mapview.html

- Export splits:
    python mapview.py export-splits --parquet data/dataset.parquet --out data/index/splits.jsonl

- Check coverage:
    python mapview.py check-coverage --images-jsonl data/index/images.jsonl --splits-jsonl data/index/splits.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import folium
from folium.plugins import HeatMap, MarkerCluster

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: pandas. Install with: pip install pandas") from e


# ----------------------------
# Helpers
# ----------------------------
def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # great-circle distance (km)
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def pick_latest_val_topk(runs_dir: Path = Path("runs")) -> Path:
    latest = runs_dir / "latest" / "val_topk.parquet"
    if latest.exists():
        return latest
    cands = list(runs_dir.glob("**/val_topk.parquet"))
    if not cands:
        raise FileNotFoundError("No runs/**/val_topk.parquet found. Provide --val-topk explicitly.")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def split_color(split: str) -> str:
    # folium expects CSS color strings
    return {
        "train": "#1f77b4",   # blue
        "val": "#2ca02c",     # green
        "test": "#9467bd",    # purple
        "unknown": "#7f7f7f", # gray
    }.get(split, "#7f7f7f")


@dataclass(frozen=True)
class CentroidLookup:
    idx_to_centroid: dict

    @classmethod
    def load(cls, labels_json: Path) -> "CentroidLookup":
        with labels_json.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(idx_to_centroid=obj)

    def centroid(self, idx: str) -> Optional[Tuple[float, float]]:
        c = self.idx_to_centroid.get(idx)
        if c is None:
            return None
        if isinstance(c, dict):
            return float(c["lat"]), float(c["lon"])
        # allow list/tuple
        return float(c[0]), float(c[1])


def load_images(images_jsonl: Path) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, float, float]]]:
    id_to_coord: Dict[str, Tuple[float, float]] = {}
    points: List[Tuple[str, float, float]] = []
    for o in iter_jsonl(images_jsonl):
        _id = str(o["id"])
        lat = float(o["lat"])
        lon = float(o["lon"])
        id_to_coord[_id] = (lat, lon)
        points.append((_id, lat, lon))
    return id_to_coord, points


def load_splits(splits_jsonl: Optional[Path]) -> Dict[str, str]:
    if not splits_jsonl:
        return {}
    if not splits_jsonl.exists():
        return {}
    m: Dict[str, str] = {}
    for o in iter_jsonl(splits_jsonl):
        _id = str(o.get("id") or "")
        sp = str(o.get("split") or "")
        if _id:
            m[_id] = (sp.strip() or "unknown")
    return m


# ----------------------------
# Map building
# ----------------------------
def add_points_layer(
    m: folium.Map,
    points: List[Tuple[str, float, float]],
    split_map: Dict[str, str],
    *,
    color_by: str,
    cluster: bool,
    radius: float,
    opacity: float,
    max_points: int,
    layer_name: str = "Points",
    show: bool = True,
) -> None:
    if max_points and max_points > 0:
        points = points[:max_points]

    fg = folium.FeatureGroup(name=layer_name, show=show)

    container = fg
    if cluster:
        container = MarkerCluster(name=f"{layer_name} (cluster)").add_to(fg)

    for _id, lat, lon in points:
        sp = split_map.get(_id, "unknown")
        if color_by == "split":
            color = split_color(sp)
            popup = f"id={_id} split={sp}"
        else:
            color = "#1f77b4"
            popup = f"id={_id}"

        folium.CircleMarker(
            location=(lat, lon),
            radius=radius,
            weight=0,
            fill=True,
            fill_opacity=opacity,
            color=color,
            popup=popup,
        ).add_to(container)

    fg.add_to(m)


def add_heat_layer(
    m: folium.Map,
    points: List[Tuple[str, float, float]],
    *,
    max_points: int,
    layer_name: str = "Heatmap",
    show: bool = False,
) -> None:
    if max_points and max_points > 0:
        points = points[:max_points]
    heat_pts = [[lat, lon, 1.0] for _, lat, lon in points]
    if not heat_pts:
        return
    HeatMap(heat_pts, name=layer_name, show=show).add_to(m)


def add_val_far_errors_layer(
    m: folium.Map,
    *,
    id_to_coord: Dict[str, Tuple[float, float]],
    centroid_lookup: CentroidLookup,
    val_topk: Path,
    thresh_km: float,
    add_heatmap: bool,
    show: bool = True,
) -> None:
    df = pd.read_parquet(val_topk)

    fg = folium.FeatureGroup(name=f"VAL far errors > {thresh_km:.0f}km", show=show)
    heat_pts: List[List[float]] = []
    errors = 0
    skipped = 0

    for _, row in df.iterrows():
        img_id = str(row["image_id"])
        pred_idx = str(row["pred_idx"])
        true_idx = str(row["true_idx"])

        if img_id not in id_to_coord:
            skipped += 1
            continue
        cp = centroid_lookup.centroid(pred_idx)
        if cp is None:
            skipped += 1
            continue

        lat_true, lon_true = id_to_coord[img_id]
        lat_pred, lon_pred = cp
        d = haversine_km(lat_true, lon_true, lat_pred, lon_pred)

        if d > thresh_km:
            errors += 1
            heat_pts.append([lat_true, lon_true, min(d / 800.0, 1.0)])
            folium.CircleMarker(
                location=(lat_true, lon_true),
                radius=4,
                weight=1,
                color="red",
                fill=True,
                fill_opacity=0.75,
                popup=f"id={img_id} d={d:.1f}km true={true_idx} pred={pred_idx}",
            ).add_to(fg)

    fg.add_to(m)

    if add_heatmap and heat_pts:
        HeatMap(heat_pts, name=f"Heatmap far errors > {thresh_km:.0f}km", show=False).add_to(m)

    print(f"[far errors] val_topk={val_topk} | far={errors} | skipped={skipped}")


def add_h3_dominant_layer(
    m: folium.Map,
    *,
    id_to_coord: Dict[str, Tuple[float, float]],
    val_topk: Path,
    h3_res: int,
    min_cell_count: int,
    mode: str,
    show: bool = False,
) -> None:
    # optional dependency
    try:
        import h3  # type: ignore
    except Exception as e:
        print("H3 not available (pip install h3). Skipping H3 layer.", e)
        return

    df = pd.read_parquet(val_topk)

    cell_counts: dict[str, Counter] = defaultdict(Counter)
    used = 0

    for _, row in df.iterrows():
        img_id = str(row["image_id"])
        if img_id not in id_to_coord:
            continue
        lat, lon = id_to_coord[img_id]
        cell = h3.latlng_to_cell(lat, lon, h3_res)
        cls = str(row["pred_idx"] if mode == "pred" else row["true_idx"])
        cell_counts[cell][cls] += 1
        used += 1

    totals = {cell: sum(c.values()) for cell, c in cell_counts.items()}
    kept = {cell: c for cell, c in cell_counts.items() if totals[cell] >= min_cell_count}

    if not kept:
        print("[h3] No cells kept (try lowering --h3-min-cell-count or changing --h3-res).")
        return

    max_total = max(totals[cell] for cell in kept.keys())

    fg = folium.FeatureGroup(name=f"H3 dominant ({mode}) r={h3_res}", show=show)

    def color_for(frac: float) -> str:
        # simple 4-step ramp
        if frac < 0.25:
            return "#4e79a7"
        if frac < 0.5:
            return "#59a14f"
        if frac < 0.75:
            return "#f28e2b"
        return "#e15759"

    for cell, counter in kept.items():
        total = totals[cell]
        dom_cls, dom_n = counter.most_common(1)[0]
        frac_total = total / max_total

        boundary = h3.cell_to_boundary(cell)  # list of (lat,lon)
        poly = [(lat, lon) for lat, lon in boundary]

        popup = (
            f"cell={cell}<br>"
            f"total={total}<br>"
            f"dominant_{mode}={dom_cls} ({dom_n}, {dom_n/total:.1%})"
        )

        folium.Polygon(
            locations=poly,
            color=color_for(frac_total),
            weight=1,
            fill=True,
            fill_opacity=0.25,
            popup=folium.Popup(popup, max_width=350),
        ).add_to(fg)

    fg.add_to(m)
    print(f"[h3] val_topk={val_topk} | used={used} | cells={len(kept)} (min_count={min_cell_count})")


def cmd_build(args: argparse.Namespace) -> None:
    images_jsonl = Path(args.images_jsonl)
    if not images_jsonl.exists():
        raise SystemExit(f"Missing images jsonl: {images_jsonl}")

    splits_jsonl = Path(args.splits_jsonl) if args.splits_jsonl else None
    labels_json = Path(args.labels_json) if args.labels_json else None

    id_to_coord, points = load_images(images_jsonl)
    split_map = load_splits(splits_jsonl)

    center = (args.center_lat, args.center_lon)
    m = folium.Map(location=list(center), zoom_start=args.zoom, tiles=args.tiles)

    add_points_layer(
        m,
        points,
        split_map,
        color_by=args.color_by,
        cluster=args.cluster,
        radius=args.radius,
        opacity=args.opacity,
        max_points=args.max_points,
        layer_name="Points",
        show=True,
    )

    if args.heat:
        add_heat_layer(m, points, max_points=args.max_points, layer_name="Heatmap", show=False)

    # val layers require labels + val_topk
    if args.add_val_layers:
        if labels_json is None:
            raise SystemExit("--labels-json is required when --add-val-layers is enabled.")
        if not labels_json.exists():
            raise SystemExit(f"Missing labels json: {labels_json}")

        val_topk = Path(args.val_topk) if args.val_topk else pick_latest_val_topk()
        if not val_topk.exists():
            raise SystemExit(f"Missing val_topk parquet: {val_topk}")

        centroid_lookup = CentroidLookup.load(labels_json)

        add_val_far_errors_layer(
            m,
            id_to_coord=id_to_coord,
            centroid_lookup=centroid_lookup,
            val_topk=val_topk,
            thresh_km=args.far_thresh_km,
            add_heatmap=args.far_heatmap,
            show=True,
        )

        if args.h3:
            add_h3_dominant_layer(
                m,
                id_to_coord=id_to_coord,
                val_topk=val_topk,
                h3_res=args.h3_res,
                min_cell_count=args.h3_min_cell_count,
                mode=args.h3_mode,
                show=False,
            )

    folium.LayerControl(collapsed=False).add_to(m)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print("Saved:", out.resolve())


# ----------------------------
# Split utilities
# ----------------------------
def cmd_export_splits(args: argparse.Namespace) -> None:
    pq = Path(args.parquet)
    if not pq.exists():
        raise SystemExit(f"Missing parquet: {pq}")

    df = pd.read_parquet(pq, columns=[args.id_col, args.split_col])

    if args.id_col not in df.columns:
        raise SystemExit(f"Missing id column '{args.id_col}' in parquet. Available: {list(df.columns)}")
    if args.split_col not in df.columns:
        raise SystemExit(f"Missing split column '{args.split_col}' in parquet. Available: {list(df.columns)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    counts: Dict[str, int] = {}
    with out.open("w", encoding="utf-8") as f:
        for _id, sp in zip(df[args.id_col].astype(str), df[args.split_col].astype(str)):
            sp = sp.strip() or "unknown"
            f.write(json.dumps({"id": _id, "split": sp}, ensure_ascii=False) + "\n")
            n += 1
            counts[sp] = counts.get(sp, 0) + 1

    print(f"Wrote {n} rows -> {out}")
    print("Counts:", dict(sorted(counts.items(), key=lambda kv: kv[0])))


def cmd_check_coverage(args: argparse.Namespace) -> None:
    images = Path(args.images_jsonl)
    splits = Path(args.splits_jsonl)

    if not images.exists():
        raise SystemExit(f"Missing {images}")
    if not splits.exists():
        raise SystemExit(f"Missing {splits}")

    split_map: Dict[str, str] = {}
    for o in iter_jsonl(splits):
        _id = str(o.get("id") or "")
        sp = str(o.get("split") or "")
        if _id and sp:
            split_map[_id] = sp

    counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    total = 0

    for o in iter_jsonl(images):
        _id = str(o.get("id") or "")
        sp = split_map.get(_id, "unknown")
        counts[sp] = counts.get(sp, 0) + 1
        total += 1

    known = total - counts.get("unknown", 0)
    print("Total images.jsonl:", total)
    print("Split rows:", len(split_map))
    print("Known (in splits):", known)
    print("Counts by split:", dict(sorted(counts.items(), key=lambda kv: kv[0])))
    print("Unknown:", counts.get("unknown", 0))


# ----------------------------
# CLI
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mapview.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Generate a folium HTML map (default: ./mapview.html)")
    b.add_argument("--images-jsonl", default="data/index/images.jsonl")
    b.add_argument("--splits-jsonl", default="data/index/splits.jsonl", help="Optional; missing is OK")
    b.add_argument("--labels-json", default="data/index/labels.json", help="Required for --add-val-layers")
    b.add_argument("--val-topk", default="", help="Optional; otherwise auto-picks runs/latest/val_topk.parquet")
    b.add_argument("--out", default="mapview.html")

    b.add_argument("--center-lat", type=float, default=46.5)
    b.add_argument("--center-lon", type=float, default=2.5)
    b.add_argument("--zoom", type=int, default=6)
    b.add_argument("--tiles", default="cartodbpositron")

    b.add_argument("--color-by", default="split", choices=["none", "split"])
    b.add_argument("--cluster", action="store_true", help="Use MarkerCluster for points")
    b.add_argument("--heat", action="store_true", help="Add HeatMap layer of points")
    b.add_argument("--max-points", type=int, default=0, help="0 = all points (can be heavy)")
    b.add_argument("--radius", type=float, default=2.0)
    b.add_argument("--opacity", type=float, default=0.65)

    b.add_argument("--add-val-layers", action="store_true", help="Add layers from val_topk (far errors, optional H3)")
    b.add_argument("--far-thresh-km", type=float, default=300.0)
    b.add_argument("--far-heatmap", action="store_true", help="Add heatmap for far errors (val)")

    b.add_argument("--h3", action="store_true", help="Add H3 dominant layer (requires `pip install h3`)")
    b.add_argument("--h3-res", type=int, default=6)
    b.add_argument("--h3-min-cell-count", type=int, default=20)
    b.add_argument("--h3-mode", default="pred", choices=["pred", "true"])
    b.set_defaults(func=cmd_build)

    ex = sub.add_parser("export-splits", help="Export id->split JSONL from dataset parquet")
    ex.add_argument("--parquet", required=True)
    ex.add_argument("--out", default="data/index/splits.jsonl")
    ex.add_argument("--id-col", default="image_id")
    ex.add_argument("--split-col", default="split")
    ex.set_defaults(func=cmd_export_splits)

    cc = sub.add_parser("check-coverage", help="Check split coverage between images.jsonl and splits.jsonl")
    cc.add_argument("--images-jsonl", default="data/index/images.jsonl")
    cc.add_argument("--splits-jsonl", default="data/index/splits.jsonl")
    cc.set_defaults(func=cmd_check_coverage)

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
