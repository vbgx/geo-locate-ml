# geo-locate-ml — Architecture Overview

This document describes the technical architecture of the project.

The system predicts the geographic location of an image using supervised learning on Mapillary street-level imagery.

---

# 1. High-Level Pipeline

The project follows a deterministic data → training → export pipeline:

1. Image acquisition (Mapillary API)
2. Dataset indexing (H3 spatial binning)
3. Train/validation/test split (sequence-aware)
4. Model training
5. Evaluation (accuracy + geographic metrics)
6. Snapshot export
7. Optional raw image cleanup

All steps are reproducible from the command line.

---

# 2. Data Layer

## 2.1 Raw Images

Location:

data/raw/mapillary/


Contains downloaded `.jpg` files from Mapillary.

Raw images are temporary and can be safely deleted after training.

---

## 2.2 Index (Canonical Dataset State)

Location:

data/index/


Files:

- `images.jsonl` → raw metadata
- `images.parquet` → structured dataset
- `splits.parquet` → train/val/test assignment
- `stats.json` → dataset statistics

This folder defines the current dataset state.

---

# 3. Spatial Representation

## 3.1 H3 Grid

The Earth surface is discretized using H3 hexagonal cells.

Parameters:
- `h3_resolution`
- `min_cell_samples`

Each image is assigned to an H3 cell.
Each H3 cell becomes a classification class.

This converts geographic regression into classification over spatial bins.

---

# 4. Splitting Strategy

Splits are **sequence-aware**:

- Images from the same Mapillary sequence
- Never appear in multiple splits

This prevents leakage across train/validation/test.

Ratios:
- 80% train
- 15% validation
- 5% test

---

# 5. Model Architecture

Defined in:

src/model.py


Current design:
- CNN backbone
- Dropout regularization
- Final classification layer over H3 cells

The architecture is intentionally lightweight to:
- Allow CPU training
- Support incremental dataset growth
- Avoid overfitting on small batches

---

# 6. Training System

Entry point:

python -m src.run


Core components:
- `dataset.py`
- `train.py`
- `geo_loss.py`
- `metrics_geo.py`

## 6.1 Loss Functions

Two modes:

### 1. Cross-Entropy
Standard classification loss.

### 2. Distance-Aware Loss (optional)

Encourages geographically closer predictions by:

- Computing distance between predicted cell centroid and ground-truth coordinates
- Applying exponential distance weighting (`geo_tau_km`)
- Mixing with cross-entropy (`geo_mix_ce`)

This improves geographic realism.

---

# 7. Metrics

Training reports:

- Validation accuracy
- Median geographic error (km)
- 90th percentile error (km)

Why:

Accuracy alone does not reflect geographic quality.
Distance metrics measure real-world usefulness.

---

# 8. Artifacts

## 8.1 Training Runs

Location:

runs/<timestamp>/


Contains:
- checkpoints
- plots
- metrics.csv
- REPORT.md

A symlink:

runs/latest

Points to most recent run.

---

## 8.2 Global Best Model

Location:

models/best.pt
models/best.json


Updated only when performance improves.

---

## 8.3 Snapshots

Location:

exports/


Each snapshot contains:
- Run artifacts
- Dataset index
- Best model at that time

Snapshots are immutable.

---

# 9. Automation Layer

Controlled by:


Makefile


Main flows:
- `make download`
- `make rebuild`
- `make train`
- `make export`
- `make batch`
- `make loop`

The Makefile ensures reproducible execution.

---

# 10. Storage Strategy

Images are large.
Index files are small.

Design principle:
- Raw images are disposable.
- Index + model artifacts are persistent.

This allows incremental dataset growth without disk exhaustion.

---

# 11. Design Philosophy

The system prioritizes:

- Reproducibility
- Determinism
- Geographic realism
- Controlled disk usage
- Clear separation between data, training, and artifacts

It is designed to scale gradually from:
- 10k images
- to 100k+
- to multi-region datasets

without changing core architecture.

---

# 12. Future Extensions

Possible evolution paths:

- Higher H3 resolution (finer localization)
- Backbone upgrades (ResNet / ViT)
- Pretrained initialization
- Regression over lat/lon instead of classification
- Multi-scale spatial modeling
- Embedding-based nearest neighbor inference

---

# 13. Summary

The project is structured as:

Data acquisition → Spatial indexing → Supervised training → Geographic evaluation → Snapshot export

Each layer is isolated and replaceable.


