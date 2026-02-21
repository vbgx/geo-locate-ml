from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class GeoDataset(Dataset):
    """
    Processed-only dataset.

    Returns:
      - non-hierarchical:
          (x, y, lat, lon, image_id)
      - hierarchical:
          (x, y_r6, y_r7, lat, lon, image_id)

    Absolutely NO raw handling.
    Absolutely NO proxy / h3_features.
    """

    def __init__(
        self,
        parquet_path: str,
        split: str,
        image_size: int,
        *,
        hierarchical_enabled: bool = False,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.split = str(split)
        self.image_size = int(image_size)
        self.hierarchical_enabled = bool(hierarchical_enabled)

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {self.parquet_path}")

        df = pd.read_parquet(self.parquet_path)

        # -------------------------
        # Required columns
        # -------------------------
        required = {"path", "label_idx", "lat", "lon", "split", "id"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"Parquet missing columns: {sorted(missing)}")

        if self.hierarchical_enabled:
            hier_required = {"label_r6_idx"}
            missing_h = hier_required - set(df.columns)
            if missing_h:
                raise RuntimeError(
                    f"hierarchical_enabled=True but missing columns: {sorted(missing_h)}"
                )

        # -------------------------
        # Split filter
        # -------------------------
        df = df[df["split"] == self.split].reset_index(drop=True)

        if df.empty:
            raise RuntimeError(f"No samples for split='{self.split}' in {self.parquet_path}")

        # -------------------------
        # Path validation (processed only)
        # -------------------------
        paths = df["path"].astype(str)
        missing_files = [p for p in paths if not Path(p).exists()]
        if missing_files:
            raise RuntimeError(
                f"{len(missing_files)} processed images are missing on disk. "
                f"Example: {missing_files[0]}"
            )

        self.df = df

        # -------------------------
        # Torch transforms
        # -------------------------
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]

        img_path = Path(row["path"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {img_path}") from e

        x = self.transform(img)

        lat = float(row["lat"])
        lon = float(row["lon"])
        image_id = str(row["id"])

        if self.hierarchical_enabled:
            y_r6 = int(row["label_r6_idx"])
            y_r7 = int(row["label_idx"])
            return (
                x,
                torch.tensor(y_r6, dtype=torch.long),
                torch.tensor(y_r7, dtype=torch.long),
                torch.tensor(lat, dtype=torch.float32),
                torch.tensor(lon, dtype=torch.float32),
                image_id,
            )

        y = int(row["label_idx"])
        return (
            x,
            torch.tensor(y, dtype=torch.long),
            torch.tensor(lat, dtype=torch.float32),
            torch.tensor(lon, dtype=torch.float32),
            image_id,
        )

    # ------------------------------------------------------------------
    # Optional helpers (used by hard-neg sampler)
    # ------------------------------------------------------------------
    def all_ids(self) -> List[str]:
        return [str(v) for v in self.df["id"].tolist()]

