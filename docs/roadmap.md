# geo-locate-ml — Roadmap

This document defines the long-term direction of the project.

Goal:
Train a vision model capable of predicting where a photo was taken,
using large-scale Mapillary data and distance-aware learning.

---

# Phase 0 — Infrastructure Stabilization

Objective: Make the pipeline fully reliable.

- [x] Robust Mapillary downloader with retries
- [x] Batch city CSV system
- [x] Index builder (H3 grid)
- [x] Sequence-aware splits (no leakage)
- [x] Distance-aware loss (geo loss)
- [x] Metrics: accuracy + median km + p90 km
- [x] Early stopping
- [x] Export snapshot system
- [x] Disk purge system
- [ ] Harden Makefile & scripts
- [ ] Add dependency requirements.txt
- [ ] Add logging instead of print()

Exit criteria:
Pipeline runs from download → train → export without manual intervention.

---

# Phase 1 — Dataset Expansion (France)

Objective: Increase geographic diversity.

- [ ] Process 50 cities (10 batches)
- [ ] Ensure balanced H3 cells
- [ ] Monitor class imbalance
- [ ] Track dataset growth metrics
- [ ] Visualize geographic coverage map

Target:
> 100k images  
> 50–100 H3 cells  
> Median error < 2km  

---

# Phase 2 — Model Architecture Upgrade

Objective: Move beyond simple CNN.

Options:

- [ ] Replace custom CNN with ResNet18
- [ ] Try EfficientNet-lite
- [ ] Add pretrained ImageNet weights
- [ ] Compare frozen backbone vs fine-tune
- [ ] Add multi-scale training

Research directions:

- Geo-classification vs regression
- Hybrid classification + coordinate regression
- Multi-head architecture (cell + offset)

Exit criteria:
Clear architecture outperforming baseline by >20%.

---

# Phase 3 — Better Geographic Intelligence

Objective: Make the model spatially aware.

- [ ] Hierarchical H3 (coarse → fine prediction)
- [ ] Distance-aware soft targets
- [ ] Top-k geographic smoothing
- [ ] Graph-based cell adjacency regularization
- [ ] Sequence-aware inference (use consecutive frames)

Target:
Median error < 1km
p90 < 5km

---

# Phase 4 — Scaling Strategy

Objective: Prepare for large-scale training.

- [ ] Dataset caching system
- [ ] Image resizing pipeline
- [ ] Mixed precision training
- [ ] Multi-GPU support
- [ ] Cloud training option

Optional:
- Self-supervised pretraining
- Contrastive geo-embedding model

---

# Phase 5 — Evaluation & Productization

Objective: Turn research into usable system.

- [ ] CLI prediction tool
- [ ] Local FastAPI inference server
- [ ] TorchScript export
- [ ] ONNX export
- [ ] Web demo

Metrics to track:

- Accuracy
- Median km
- p90 km
- Inference speed
- Model size

---

# Phase 6 — Research Ambition

Long-term direction:

- Continental-scale model
- City-level street discrimination
- Terrain-type awareness
- Seasonal robustness
- Domain adaptation (satellite + street)

Ultimate ambition:

Given an image,
predict location with < 500m error in dense urban areas.

---

# Guiding Principles

1. Data quality > model complexity.
2. Prevent data leakage at all costs.
3. Always measure geographic distance, not just accuracy.
4. Automate everything.
5. Keep disk usage controlled.

---

# Current Status

Baseline:
- H3 resolution: 7
- ~5k images
- 7 cells
- Median error: ~1–7km
- Accuracy: ~0.35–0.55

We are still in early dataset scaling phase.

