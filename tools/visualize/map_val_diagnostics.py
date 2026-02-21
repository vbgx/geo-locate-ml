#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap

# -------------------
# Paths (project defaults)
# -------------------
IMAGES = Path("data/index/images.jsonl")
SPLITS = Path("data/index/splits.jsonl")
LABELS = Path("data/index/labels.json")      # idx -> centroid (lat,lon)
OUT = Path("data/index/map_val_diagnostics.html")

# -------------------
# Options
# -------------------
CENTER = (46.5, 2.5)
ZOOM = 6

FAR_THRESH_KM = 300
ADD_FAR_HEATMAP = True

H3_RES = 6                 # 5..7 typiquement. 6 = bon compromis France
H3_MIN_CELL_COUNT = 20     # ignore les cellules trop petites
H3_MODE = "pred"           # "pred" ou "true" (classe dominante par cell)
# -------------------

def pick_val_topk() -> Path:
    latest = Path("runs/latest/val_topk.parquet")
    if latest.exists():
        return latest
    cands = list(Path("runs").glob("**/val_topk.parquet"))
    if not cands:
        raise FileNotFoundError("No runs/**/val_topk.parquet found.")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def load_images():
    id_to_coord = {}
    points = []  # (id, lat, lon)
    with IMAGES.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            _id = str(o["id"])
            lat = float(o["lat"])
            lon = float(o["lon"])
            id_to_coord[_id] = (lat, lon)
            points.append((_id, lat, lon))
    return id_to_coord, points

def load_splits():
    split_map = {}
    if not SPLITS.exists():
        return split_map
    with SPLITS.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            split_map[str(o["id"])] = str(o["split"])
    return split_map

def load_centroids():
    with LABELS.open("r", encoding="utf-8") as f:
        idx_to_centroid = json.load(f)

    def centroid(idx: str):
        c = idx_to_centroid.get(idx)
        if c is None:
            return None
        if isinstance(c, dict):
            return float(c["lat"]), float(c["lon"])
        return float(c[0]), float(c[1])

    return centroid

def split_color(split: str) -> str:
    # folium expects CSS color strings
    return {
        "train": "#1f77b4",  # blue
        "val":   "#2ca02c",  # green
        "test":  "#9467bd",  # purple
        "unknown": "#7f7f7f" # gray
    }.get(split, "#7f7f7f")

def make_map():
    return folium.Map(location=list(CENTER), zoom_start=ZOOM, tiles="cartodbpositron")

def add_split_points(m, points, split_map):
    fg = folium.FeatureGroup(name="Points (split)", show=True)
    cluster = MarkerCluster(name="Cluster (split)").add_to(fg)

    for _id, lat, lon in points:
        sp = split_map.get(_id, "unknown")
        folium.CircleMarker(
            location=(lat, lon),
            radius=2,
            weight=0,
            fill=True,
            fill_opacity=0.7,
            color=split_color(sp),
            popup=f"id={_id} split={sp}",
        ).add_to(cluster)

    fg.add_to(m)

def add_far_errors(m, id_to_coord, centroid):
    VAL_TOPK = pick_val_topk()
    df = pd.read_parquet(VAL_TOPK)

    fg = folium.FeatureGroup(name=f"VAL far errors > {FAR_THRESH_KM}km", show=True)
    errors = []
    heat_pts = []

    skipped = 0
    for _, row in df.iterrows():
        img_id = str(row["image_id"])
        pred_idx = str(row["pred_idx"])
        true_idx = str(row["true_idx"])

        if img_id not in id_to_coord:
            skipped += 1
            continue

        cp = centroid(pred_idx)
        if cp is None:
            skipped += 1
            continue

        lat_true, lon_true = id_to_coord[img_id]
        lat_pred, lon_pred = cp
        d = haversine(lat_true, lon_true, lat_pred, lon_pred)
        if d > FAR_THRESH_KM:
            errors.append((img_id, lat_true, lon_true, d, true_idx, pred_idx))
            heat_pts.append([lat_true, lon_true, min(d / 800.0, 1.0)])

    # markers
    for img_id, lat, lon, d, t, p in errors:
        folium.CircleMarker(
            location=(lat, lon),
            radius=4,
            weight=1,
            color="red",
            fill=True,
            fill_opacity=0.75,
            popup=f"id={img_id} d={d:.1f}km true={t} pred={p}",
        ).add_to(fg)

    fg.add_to(m)

    if ADD_FAR_HEATMAP and heat_pts:
        HeatMap(heat_pts, name=f"Heatmap far errors > {FAR_THRESH_KM}km", show=False).add_to(m)

    print(f"[far errors] val_topk={VAL_TOPK} | far={len(errors)} | skipped={skipped}")

def add_h3_dominant(m, id_to_coord):
    # optional dependency
    try:
        import h3
    except Exception as e:
        print("H3 not available (pip install h3). Skipping H3 layer.", e)
        return

    VAL_TOPK = pick_val_topk()
    df = pd.read_parquet(VAL_TOPK)

    # cell -> Counter(class)
    cell_counts = defaultdict(Counter)

    used = 0
    for _, row in df.iterrows():
        img_id = str(row["image_id"])
        if img_id not in id_to_coord:
            continue
        lat, lon = id_to_coord[img_id]
        cell = h3.latlng_to_cell(lat, lon, H3_RES)

        cls = str(row["pred_idx"] if H3_MODE == "pred" else row["true_idx"])
        cell_counts[cell][cls] += 1
        used += 1

    # normalize for coloring
    totals = {cell: sum(c.values()) for cell, c in cell_counts.items()}
    kept = {cell: c for cell, c in cell_counts.items() if totals[cell] >= H3_MIN_CELL_COUNT}

    if not kept:
        print("[h3] No cells kept (try lowering H3_MIN_CELL_COUNT or resolution).")
        return

    max_total = max(totals[cell] for cell in kept.keys())

    fg = folium.FeatureGroup(name=f"H3 dominant ({H3_MODE}) r={H3_RES}", show=False)

    def color_for(frac: float) -> str:
        # simple blue -> red ramp
        # frac in [0,1]
        # low density => light blue, high => orange/red
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
            f"dominant_{H3_MODE}={dom_cls} ({dom_n}, {dom_n/total:.1%})"
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
    print(f"[h3] val_topk={VAL_TOPK} | used={used} | cells={len(kept)} (min_count={H3_MIN_CELL_COUNT})")

def main():
    assert IMAGES.exists(), f"Missing {IMAGES}"
    assert LABELS.exists(), f"Missing {LABELS}"

    id_to_coord, points = load_images()
    split_map = load_splits()
    centroid = load_centroids()

    m = make_map()

    # layers
    add_split_points(m, points, split_map)
    add_far_errors(m, id_to_coord, centroid)
    add_h3_dominant(m, id_to_coord)

    folium.LayerControl(collapsed=False).add_to(m)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT))
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
