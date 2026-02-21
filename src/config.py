from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class TrainConfig:
    # -------------------------
    # Data
    # -------------------------
    jsonl_index: str = "data/index/images.jsonl"
    parquet_index: str = "data/index/images.parquet"

    h3_resolution: int = 7
    min_cell_samples: int = 30
    hierarchical_enabled: bool = False
    merge_parent_res: int = 6
    merge_min_cell_count: int = 15
    merge_far_rate_threshold: float = 0.8
    merge_far_rate_path: str = ""
    merge_far_labels_path: str = ""

    # -------------------------
    # Splits (sequence-aware)
    # -------------------------
    split_train: float = 0.80
    split_val: float = 0.15
    split_test: float = 0.05
    seed: int = 42

    # -------------------------
    # Training
    # -------------------------
    image_sizes: List[int] = (128,)
    batch_sizes: List[int] = (32,)
    epochs: int = 6

    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.30
    num_workers: int = 2

    # -------------------------
    # Distance-aware loss (geo)
    # -------------------------
    geo_loss_enabled: bool = True
    geo_tau_km: float = 1.8
    geo_mix_ce: float = 0.45  # 0 -> only geo loss, 1 -> only CE

    # -------------------------
    # Hard-negative mining (P90 killer)
    # -------------------------
    hardneg_enabled: bool = False
    hardneg_threshold_km: float = 500.0
    hardneg_boost: float = 2.0
    hardneg_max_pool: int = 20000
    hardneg_min_count: int = 100  # don't activate sampler if pool too small

    # -------------------------
    # Early stopping
    # -------------------------
    early_stopping_enabled: bool = True
    early_stop_metric: str = "median_km"   # "median_km" | "p90_km" | "val_acc"
    early_stop_mode: str = "min"           # "min" for km metrics, "max" for accuracy
    early_stop_patience: int = 6
    early_stop_min_delta: float = 0.0

    # -------------------------
    # Outputs
    # -------------------------
    keep_only_global_best: bool = True
    make_confusion_matrix: bool = True
    make_geo_error_plot: bool = True
    topk: int = 5
    dump_topk: int = 10  # for val_topk.parquet dump

    def to_dict(self) -> dict:
        return asdict(self)
