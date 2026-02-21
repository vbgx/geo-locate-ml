#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import folium
from folium.plugins import MarkerCluster, HeatMap
from tqdm import tqdm


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_split_map(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        raise SystemExit(f"Missing split jsonl: {path}")
    m: Dict[str, str] = {}
    for o in iter_jsonl(path):
        _id = str(o.get("id") or "")
        sp = str(o.get("split") or "")
        if _id and sp:
            m[_id] = sp
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Map images.jsonl points on a folium map.")
    ap.add_argument("--images-jsonl", default="data/index/images.jsonl")
    ap.add_argument("--split-jsonl", default="", help="Optional: data/index/splits.jsonl (id -> split)")
    ap.add_argument("--out", default="data/index/map_preview.html")

    ap.add_argument("--color-by", default="none", choices=["none", "split"])
    ap.add_argument("--cluster", action="store_true", help="Use MarkerCluster")
    ap.add_argument("--heat", action="store_true", help="Add HeatMap layer")
    ap.add_argument("--max-points", type=int, default=0, help="0 = all points (can be heavy)")
    ap.add_argument("--radius", type=float, default=1.5, help="CircleMarker radius")
    ap.add_argument("--opacity", type=float, default=0.6, help="CircleMarker fill opacity")
    args = ap.parse_args()

    images_path = Path(args.images_jsonl)
    if not images_path.exists():
        raise SystemExit(f"Missing images jsonl: {images_path}")

    split_path = Path(args.split_jsonl) if args.split_jsonl else None
    split_map = load_split_map(split_path)

    # Stable colors
    color_for_split = {
        "train": "#2b8cbe",   # blue-ish
        "val": "#31a354",     # green
        "test": "#756bb1",    # purple
        "unknown": "#636363", # gray
    }

    pts: List[Tuple[float, float, str]] = []
    n = 0
    for o in tqdm(iter_jsonl(images_path), desc="Loading", unit="pt"):
        lat = o.get("lat")
        lon = o.get("lon")
        if lat is None or lon is None:
            continue
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            continue

        _id = str(o.get("id") or "")
        sp = split_map.get(_id, "unknown") if args.color_by == "split" else "unknown"
        pts.append((lat_f, lon_f, sp))

        n += 1
        if args.max_points and n >= int(args.max_points):
            break

    if not pts:
        raise SystemExit("No points loaded.")

    # Center map by median-ish (simple average)
    lat0 = sum(p[0] for p in pts) / len(pts)
    lon0 = sum(p[1] for p in pts) / len(pts)

    m = folium.Map(location=[lat0, lon0], zoom_start=6, tiles="cartodbpositron")

    # Layers per split (so you can toggle them)
    layers = {}
    if args.color_by == "split":
        for sp in ["train", "val", "test", "unknown"]:
            layers[sp] = folium.FeatureGroup(name=f"split:{sp}", show=True).add_to(m)
    else:
        layers["all"] = folium.FeatureGroup(name="points", show=True).add_to(m)

    # Cluster container (optional)
    cluster_obj = None
    if args.cluster:
        cluster_obj = MarkerCluster(name="cluster").add_to(m)

    # Heat layer (optional)
    if args.heat:
        heat_pts = [[p[0], p[1]] for p in pts]
        HeatMap(heat_pts, name="heat", radius=12, blur=18, min_opacity=0.2).add_to(m)

    # Draw points
    for lat, lon, sp in tqdm(pts, desc="Rendering", unit="pt"):
        color = color_for_split.get(sp, "#1f78b4")
        cm = folium.CircleMarker(
            location=(lat, lon),
            radius=float(args.radius),
            weight=0,
            fill=True,
            fill_color=color,
            fill_opacity=float(args.opacity),
        )

        if cluster_obj is not None:
            cm.add_to(cluster_obj)
        else:
            if args.color_by == "split":
                cm.add_to(layers.get(sp, layers["unknown"]))
            else:
                cm.add_to(layers["all"])

    folium.LayerControl(collapsed=False).add_to(m)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))

    # Quick stats
    counts: Dict[str, int] = {}
    for _lat, _lon, sp in pts:
        counts[sp] = counts.get(sp, 0) + 1

    print("Saved:", out_path)
    print("Points:", len(pts))
    print("Counts:", dict(sorted(counts.items(), key=lambda kv: kv[0])))


if __name__ == "__main__":
    main()
