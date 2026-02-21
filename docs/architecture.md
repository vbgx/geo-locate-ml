geo-locate-ml — Architecture
1. Overview

geo-locate-ml is a structured computer vision system for large-scale geo-localization.

The system predicts an H3 cell (r7) from a single image, with:

Optional hierarchical prediction (r6 → r7 masking)

Optional geo-aware loss (distance-soft targets)

Hard-negative mining (tail improvement)

Feature-based reranking (post-hoc)

Fully normalized run artifacts + reporting system

The project is designed around:

Deterministic training runs

Stable run artifacts

Aggregatable experiment tracking

Minimal module duplication

2. High-Level Architecture
                ┌───────────────────────┐
                │     Data Layer        │
                │  (Parquet + H3 index) │
                └─────────────┬─────────┘
                              │
                              ▼
                ┌───────────────────────┐
                │     Modeling Layer    │
                │  CNN + GeoSoftLoss    │
                └─────────────┬─────────┘
                              │
                              ▼
                ┌───────────────────────┐
                │     Training Loop     │
                │  hardneg + hierarchy  │
                └─────────────┬─────────┘
                              │
                              ▼
                ┌───────────────────────┐
                │     Run Artifacts     │
                │  summary + dashboard  │
                └─────────────┬─────────┘
                              │
                              ▼
                ┌───────────────────────┐
                │  Global Aggregation   │
                │  multi-criteria rank  │
                └───────────────────────┘
3. Source Code Structure (src/)

All runtime logic lives under src/.

3.1 Data Layer

src/data.py

Responsibilities:

GeoDataset

Train/val split loading

Parquet reading

Transform builder

Label index handling

This is the single source of truth for dataset logic.

3.2 Modeling Layer

src/modeling.py

Contains:

MultiScaleCNN

GeoSoftTargetLoss

compute_class_weights_from_parquet

No duplicate model definitions elsewhere.

3.3 Geo & Hierarchical Logic

src/geo.py

Contains:

haversine_km

hierarchical_predict

mask_r7_logits

HardNegConfig

Hard-negative pool utilities

This module centralizes:

spatial logic

hierarchical masking

hard negative mechanics

3.4 Training

src/train_loop.py

Responsibilities:

Full training loop

Loss selection (CE / GeoSoft)

Hard-negative oversampling

Hierarchical masking

Metrics logging

src/run.py orchestrates:

Config parsing

Model creation

Train invocation

Artifact generation

3.5 Reporting

src/reporting.py

Generates:

metrics_loss.png

metrics_valacc.png

geo_error.png

confusion_matrix.png

No plotting logic outside this module.

3.6 Reranking

src/rerank.py

Contains:

Prior-based reranking

Feature-based gated reranking

Evaluation logic

Designed to improve tail (P90/P95) without breaking median.

4. Runs Architecture

Each training run is fully self-contained.

runs/<run_id>/
  summary.json
  dashboard.html
  artifacts/
    plots/
    tables/
    data/
  checkpoints/
  config.json
  REPORT.md
4.1 summary.json

Compact, comparable across runs.
Contains:

best_val_acc

best_p90_km

last metrics

config subset

artifact flags

Stable schema.

4.2 dashboard.html

Single-page visual overview:

Key metrics

Plots

Artifact links

Human-readable experiment page.

5. Aggregation Layer

tools/reporting/aggregate_runs.py

Reads all:

runs/<run_id>/summary.json

Generates:

runs/_aggregate/
  runs_summary.csv
  runs_summary.parquet
  dashboard.html

Ranking is multi-criteria:

Score combines:

best_val_acc (higher better)

best_p90_km (lower better)

last_geo_median_km (lower better)

penalties for missing artifacts

This allows rational experiment comparison.

6. Data Architecture
data/
  raw/              # unprocessed Mapillary
  processed/        # cleaned images
  index/            # canonical parquet + H3 features
  external/         # DEM, coastline, worldcover
  state/            # pipeline progress

The index folder is canonical:

images.parquet

splits.parquet

h3_features.parquet

labels.json

No duplication elsewhere.

7. Design Principles
7.1 One Source of Truth

Model defined once

Dataset defined once

Geo logic defined once

No duplicated logic in legacy files

7.2 Run = Product

Each run is:

Reproducible

Self-contained

Inspectable

Comparable

7.3 Strict Separation

Runtime → src/

Tooling → tools/

Data → data/

Experiments → runs/

8. Future Evolution

Potential improvements:

Lightning/Fabric migration

WandB-style online tracking (optional)

Spatial attention backbone

Better feature rerank gating

Automated hyperparameter sweeps

Dataset balancing strategies

9. Summary

geo-locate-ml is structured as:

A clean modular ML system

With deterministic training

With stable experiment artifacts

With multi-run rational ranking

With zero duplicated core logic

The repository is organized to minimize architectural drift and maximize long-term maintainability.