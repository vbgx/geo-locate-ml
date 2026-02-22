from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .geo import haversine_km


def far_error_rate(geo_err_km: np.ndarray, threshold_km: float) -> float:
    """
    Fraction of samples with geo error strictly greater than threshold_km.
    """
    a = np.asarray(geo_err_km, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float((a > float(threshold_km)).mean())


@dataclass
class GeoKPI:
    mean_km: float
    median_km: float
    p90_km: float
    p95_km: float
    far_error_rate_200: float = 0.0
    far_error_rate_500: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_km": float(self.mean_km),
            "median_km": float(self.median_km),
            "p90_km": float(self.p90_km),
            "p95_km": float(self.p95_km),
            "far_error_rate_200": float(self.far_error_rate_200),
            "far_error_rate_500": float(self.far_error_rate_500),
        }


def compute_geo_kpi(
    y_true,
    y_pred,
    lat_true,
    lon_true,
    idx_to_centroid: Dict[int, Tuple[float, float]],
) -> GeoKPI:
    # Normalize shapes
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    lat_true = np.asarray(lat_true, dtype=np.float64).reshape(-1)
    lon_true = np.asarray(lon_true, dtype=np.float64).reshape(-1)

    n = int(y_true.size)
    if y_pred.size != n:
        raise RuntimeError(f"compute_geo_kpi: y_pred size mismatch: {y_pred.size} vs y_true {n}")
    if lat_true.size != n or lon_true.size != n:
        raise RuntimeError(
            f"compute_geo_kpi: lat/lon missing or mismatch: "
            f"lat={lat_true.size} lon={lon_true.size} vs y_true={n}"
        )

    errs: List[float] = []
    for yt, yp, lat, lon in zip(y_true, y_pred, lat_true, lon_true):
        plat, plon = idx_to_centroid[int(yp)]
        errs.append(haversine_km(float(lat), float(lon), float(plat), float(plon)))

    if len(errs) == 0:
        return GeoKPI(
            mean_km=0.0,
            median_km=0.0,
            p90_km=0.0,
            p95_km=0.0,
            far_error_rate_200=0.0,
            far_error_rate_500=0.0,
        )

    a = np.array(errs, dtype=np.float64)

    return GeoKPI(
        mean_km=float(a.mean()),
        median_km=float(np.median(a)),
        p90_km=float(np.percentile(a, 90)),
        p95_km=float(np.percentile(a, 95)),
        far_error_rate_200=far_error_rate(a, 200.0),
        far_error_rate_500=far_error_rate(a, 500.0),
    )