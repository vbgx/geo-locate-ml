from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class TrainConfig:
    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    jsonl_index: str = "data/index/images.jsonl"

    # IMPORTANT:
    # Training must use kept-only dataset
    parquet_index: str = "data/index/images_kept.parquet"

    h3_resolution: int = 7
    min_cell_samples: int = 30

    # --------------------------------------------------
    # Hierarchical / Merge settings
    # --------------------------------------------------
    # Run B: keep disabled unless your merge step already writes label_r6_idx + r7_to_r6
    hierarchical_enabled: bool = True

    # Parent resolution (r6 if r7 base)
    merge_parent_res: int = 6

    # Toxic class thresholds
    merge_min_cell_count: int = 15
    merge_far_rate_threshold: float = 0.8

    # Optional external far-rate inputs
    merge_far_rate_path: str = ""
    merge_far_labels_path: str = ""

    # --------------------------------------------------
    # Splits (sequence-aware)
    # --------------------------------------------------
    split_train: float = 0.80
    split_val: float = 0.15
    split_test: float = 0.05
    seed: int = 42

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    image_sizes: Tuple[int, ...] = (128,)
    batch_sizes: Tuple[int, ...] = (32,)

    epochs: int = 60

    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.30
    num_workers: int = 6

    # --------------------------------------------------
    # Distance-aware loss (geo)
    # --------------------------------------------------
    geo_loss_enabled: bool = True

    # Run B: keep tau stable (change later if needed)
    geo_tau_km: float = 1.2

    # 0 -> only geo loss
    # 1 -> only cross-entropy
    # Run B: push more geo to reduce far tail
    geo_mix_ce: float = 0.30

    # --------------------------------------------------
    # Hard-negative mining (tail reducer)
    # --------------------------------------------------
    hardneg_enabled: bool = True

    # Run B: more aggressive tail pressure
    hardneg_threshold_km: float = 300.0
    hardneg_boost: float = 4.0
    hardneg_max_pool: int = 60000
    hardneg_min_count: int = 50

    # --------------------------------------------------
    # Early stopping
    # --------------------------------------------------
    early_stopping_enabled: bool = True

    # "median_km" | "p90_km" | "val_acc"
    # Run B: optimize tail directly
    early_stop_metric: str = "p90_km"

    # "min" for km metrics, "max" for accuracy
    early_stop_mode: str = "min"

    # Run B: give the tail time to improve (still capped by epochs=60)
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.0

    # --------------------------------------------------
    # Outputs / Logging
    # --------------------------------------------------
    keep_only_global_best: bool = True
    make_confusion_matrix: bool = True
    make_geo_error_plot: bool = True

    topk: int = 5
    dump_topk: int = 10  # number of samples dumped for val inspection

    # --------------------------------------------------
    # Proxy learning (NEW)
    # --------------------------------------------------
    # Backwards-compatible: proxy learning stays OFF unless:
    #   proxy_loss_enabled=True AND proxy_loss_weight>0
    proxy_loss_enabled: bool = True

    # scalar multiplier applied in train_loop:
    #   loss = loss_main + proxy_loss_weight * proxy_loss
    proxy_loss_weight: float = 2.0

    # Which columns to read from the training parquet when proxy loss is enabled.
    # Default matches build_proxies.py output.
    proxy_cols: List[str] = field(
        default_factory=lambda: [
            "proxy_elev_log1p_z",
            "proxy_pop_log1p_z",
            "proxy_water_frac",
            "proxy_built_frac",
            "proxy_coastal_score",
        ]
    )

    # Optional per-proxy weights.
    # - dict: {"proxy_elev_log1p_z": 2.0, ...}
    # - list: [w0, w1, ...] aligned with proxy_cols
    proxy_weights: Optional[Union[Dict[str, float], List[float]]] = None

    # Regression loss shaping (Huber is robust; beta is transition between L2 and L1)
    proxy_huber_beta: float = 1.0

    # How GeoDataset should treat missing proxy values:
    # - "nan": return NaN in targets + mask=0 (default)
    # - "zero": fill missing with 0.0 but still mask=0 (so it's ignored)
    proxy_missing_policy: str = "nan"

    # --------------------------------------------------
    # Serialization
    # --------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**d)