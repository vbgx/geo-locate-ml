#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import math

import pandas as pd
import folium
from tqdm import tqdm

THRESH_KM = 300
OUT = Path("data/index/map_val_far_errors.html")
IMAGES = Path("data/index/images.jsonl")
LABELS = Path("data/index/labels.json")

def pick_val_topk() -> Path:
    cands = list(Path("runs").glob("**/val_topk.parquet"))
    if not cands:
        raise FileNotFoundError("No runs/**/val_topk.parquet found. Did you run training with dump enabled?")
    # prefer runs/latest if it exists and contains the file
    latest = Path("runs/latest/val_topk.parquet")
    if latest.exists():
        return latest
    # else pick most recently modified
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

VAL_TOPK = pick_val_topk()
print("Using:", VAL_TOPK)

print("Loading val_topk...")
df = pd.read_parquet(VAL_TOPK)

print("Loading labels...")
with open(LABELS, "r", encoding="utf-8") as f:
    idx_to_centroid = json.load(f)

# Accept both { "123": [lat,lon] } and { "123": {"lat":..,"lon":..} }
def centroid(idx: str):
    c = idx_to_centroid.get(idx)
    if c is None:
        return None
    if isinstance(c, dict):
        return float(c["lat"]), float(c["lon"])
    return float(c[0]), float(c[1])

print("Loading image lat/lon...")
id_to_coord = {}
with open(IMAGES, "r", encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        id_to_coord[str(o["id"])] = (float(o["lat"]), float(o["lon"]))

errors = []
skipped = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_id = str(row["image_id"])
    true_idx = str(row["true_idx"])
    pred_idx = str(row["pred_idx"])

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
    if d > THRESH_KM:
        errors.append((lat_true, lon_true, d, img_id, true_idx, pred_idx))

print("Far errors:", len(errors), "| skipped:", skipped)

m = folium.Map(location=[46.5, 2.5], zoom_start=6, tiles="cartodbpositron")

for lat, lon, d, img_id, t, p in errors:
    folium.CircleMarker(
        location=(lat, lon),
        radius=4,
        fill=True,
        color="red",
        fill_opacity=0.7,
        popup=f"id={img_id} {d:.1f}km true={t} pred={p}",
    ).add_to(m)

m.save(OUT)
print("Saved to", OUT)
