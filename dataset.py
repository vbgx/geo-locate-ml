from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


# 🚫 RAW REMOVED — training uses processed only
PROCESSED_DIR = Path("data/processed/mapillary")


def resolve_image_path(image_id: str) -> Path:
    p = PROCESSED_DIR / f"{image_id}.jpg"
    if not p.exists():
        raise FileNotFoundError(f"Processed image not found for id={image_id}")
    return p


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _to_float01(v: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(v.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(v, 0.0, 1.0)


class GeoDataset(Dataset):
    """
    Returns:
      - without proxies: (x, y, lat, lon, image_id)
      - with proxies:    (x, y, lat, lon, image_id, proxy_t)

    Option B (semantic proxies):
      proxy_t = [p_water, p_urban, p_coastal, p_high_pop, p_high_elev] in [0,1]

    Proxies are derived from class-level H3 features (h3_features.parquet) joined by class_idx.
    This means proxy_t depends on the target label (label_idx), not on the image_id.

    Expected columns in h3_features.parquet:
      - class_idx (int)  [required]
      - landcover        [optional but used for water/urban if present]
      - dist_coast_km    [optional but used for coastal if present]
      - pop              [optional but used for high_pop if present]
      - elev_m           [optional but used for high_elev if present]
    """

    def __init__(
        self,
        parquet_path: str,
        split: str,
        image_size: int,
        h3_features_path: Optional[str] = None,
        *,
        strict_proxy: bool = False,
        # thresholds / scales (tune later; these are sane defaults)
        pop_log_mid: float = 8.5,          # sigmoid(log1p(pop)) midpoint (~4.9k)
        pop_log_scale: float = 1.0,        # slope (bigger = softer)
        elev_mid_m: float = 800.0,         # midpoint for "high elevation"
        elev_scale_m: float = 300.0,       # slope in meters
        coast_mid_km: float = 25.0,        # midpoint for "coastalness" (closer => higher)
        coast_scale_km: float = 15.0,      # slope in km
        landcover_water_tokens: tuple[str, ...] = ("water", "ocean", "sea", "lake", "river", "wetland"),
        landcover_urban_tokens: tuple[str, ...] = ("urban", "built", "built-up", "impervious", "settlement", "city"),
    ):
        self.parquet_path = str(parquet_path)
        self.split = str(split)
        self.image_size = int(image_size)
        self.max_missing_tries = 25

        df = pd.read_parquet(self.parquet_path)
        df = df[df["split"] == self.split].reset_index(drop=True)

        self.paths = df["path"].astype(str).tolist()
        self.labels = df["label_idx"].astype(int).tolist()
        self.lats = df["lat"].astype(float).tolist()
        self.lons = df["lon"].astype(float).tolist()
        self.image_ids = [Path(p).stem for p in self.paths]

        # --------------------
        # Semantic proxies (class-level)
        # --------------------
        self.strict_proxy = bool(strict_proxy)
        self._proxy_by_class: Optional[dict[int, np.ndarray]] = None

        if h3_features_path is not None:
            fdf = pd.read_parquet(str(h3_features_path))

            if "class_idx" not in fdf.columns:
                raise ValueError(
                    f"h3_features_path provided but missing required column 'class_idx'. "
                    f"Columns={list(fdf.columns)}"
                )

            # pull available columns
            has_landcover = "landcover" in fdf.columns
            has_coast = "dist_coast_km" in fdf.columns
            has_pop = "pop" in fdf.columns
            has_elev = "elev_m" in fdf.columns

            if not (has_landcover or has_coast or has_pop or has_elev):
                raise ValueError(
                    "h3_features_path provided but none of the expected feature columns are present "
                    "(landcover, dist_coast_km, pop, elev_m). "
                    f"Columns={list(fdf.columns)}"
                )

            # --- derive proxies per class ---
            class_idx = fdf["class_idx"].astype(int).to_numpy()

            # p_water / p_urban from landcover text/category (best-effort)
            if has_landcover:
                lc = fdf["landcover"].astype(str).fillna("").str.lower()
                water_mask = lc.apply(lambda s: any(tok in s for tok in landcover_water_tokens)).to_numpy(dtype=np.float32)
                urban_mask = lc.apply(lambda s: any(tok in s for tok in landcover_urban_tokens)).to_numpy(dtype=np.float32)
                p_water = water_mask
                p_urban = urban_mask
            else:
                p_water = np.zeros((len(fdf),), dtype=np.float32)
                p_urban = np.zeros((len(fdf),), dtype=np.float32)

            # p_coastal from dist_coast_km (closer => higher), smooth sigmoid
            if has_coast:
                d = fdf["dist_coast_km"].astype(np.float32).to_numpy()
                d = np.nan_to_num(d, nan=1e6, posinf=1e6, neginf=1e6)
                # coastalness = sigmoid((mid - d)/scale)
                p_coastal = _sigmoid((coast_mid_km - d) / max(1e-6, coast_scale_km)).astype(np.float32)
                p_coastal = _to_float01(p_coastal)
            else:
                p_coastal = np.zeros((len(fdf),), dtype=np.float32)

            # p_high_pop from pop (log1p then sigmoid)
            if has_pop:
                pop = fdf["pop"].astype(np.float32).to_numpy()
                pop = np.nan_to_num(pop, nan=0.0, posinf=0.0, neginf=0.0)
                pop_log = np.log1p(np.maximum(pop, 0.0))
                p_high_pop = _sigmoid((pop_log - pop_log_mid) / max(1e-6, pop_log_scale)).astype(np.float32)
                p_high_pop = _to_float01(p_high_pop)
            else:
                p_high_pop = np.zeros((len(fdf),), dtype=np.float32)

            # p_high_elev from elev_m (sigmoid)
            if has_elev:
                elev = fdf["elev_m"].astype(np.float32).to_numpy()
                elev = np.nan_to_num(elev, nan=0.0, posinf=0.0, neginf=0.0)
                p_high_elev = _sigmoid((elev - elev_mid_m) / max(1e-6, elev_scale_m)).astype(np.float32)
                p_high_elev = _to_float01(p_high_elev)
            else:
                p_high_elev = np.zeros((len(fdf),), dtype=np.float32)

            proxies = np.stack([p_water, p_urban, p_coastal, p_high_pop, p_high_elev], axis=1).astype(np.float32)
            proxies = _to_float01(proxies)

            # Build map: class_idx -> proxy vector
            self._proxy_by_class = {}
            for i, c in enumerate(class_idx.tolist()):
                self._proxy_by_class[int(c)] = proxies[i]

        # --------------------
        # Transforms
        # --------------------
        self.train_tf = T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(0.15, 0.15, 0.15, 0.03),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.eval_tf = T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.labels)

    def _get_proxy_for_class(self, class_idx: int) -> Optional[torch.Tensor]:
        if self._proxy_by_class is None:
            return None

        v = self._proxy_by_class.get(int(class_idx))
        if v is None:
            if self.strict_proxy:
                raise KeyError(f"No proxy features for class_idx={class_idx}")
            v = np.zeros((5,), dtype=np.float32)

        return torch.from_numpy(v).to(dtype=torch.float32)

    def __getitem__(self, idx: int):
        for _ in range(self.max_missing_tries):
            try:
                image_id = self.image_ids[idx]
                pth = resolve_image_path(image_id)

                img = Image.open(pth).convert("RGB")
                tf = self.train_tf if self.split == "train" else self.eval_tf
                x = tf(img)

                y_int = int(self.labels[idx])
                y = torch.tensor(y_int, dtype=torch.long)
                lat = torch.tensor(self.lats[idx], dtype=torch.float32)
                lon = torch.tensor(self.lons[idx], dtype=torch.float32)

                proxy_t = self._get_proxy_for_class(y_int)
                if proxy_t is None:
                    return x, y, lat, lon, image_id

                return x, y, lat, lon, image_id, proxy_t

            except FileNotFoundError:
                idx = (idx * 1103515245 + 12345) % len(self)

        raise RuntimeError("Too many missing processed images.")
