geo-locate-ml — Commands Reference

This document lists all operational commands used in the project.

The workflow is:

Build / update dataset

Train

Normalize run

Aggregate experiments

Analyze / rerank

1. Environment
1.1 Python

This project requires:

Python 3.10+

torch

torchvision

pandas

matplotlib

scikit-learn

pyarrow

Verify Python:

python3 --version
2. Dataset Pipeline

All dataset operations live in tools/.

2.1 Build Index
python3 tools/dataset/build_index.py

Generates:

data/index/images.parquet
data/index/splits.parquet
data/index/labels.json
2.2 Create Splits
python3 tools/dataset/make_splits.py
2.3 Build H3 Feature Table
python3 tools/pipeline/build_h3_features.py

Outputs:

data/index/h3_features.parquet
3. Training

Training is orchestrated by src/run.py.

3.1 Basic Training
python3 -m src.run

Or with explicit config overrides:

python3 -m src.run \
  --lr 0.001 \
  --dropout 0.3 \
  --image_size 128
3.2 Enable GeoSoft Loss
python3 -m src.run \
  --geo_loss_enabled true \
  --geo_tau_km 250 \
  --geo_mix_ce 0.3
3.3 Enable Hard Negative Mining
python3 -m src.run \
  --hardneg_enabled true
3.4 Enable Hierarchical Prediction
python3 -m src.run \
  --hierarchical_enabled true
4. Runs Management

Each training produces:

runs/<run_id>/

To normalize all runs into stable artifacts:

python3 tools/reporting/normalize_run.py

To normalize one specific run:

python3 tools/reporting/normalize_run.py --run-id 2026-02-21_03-36-36

To move (instead of copy) artifacts into canonical layout:

python3 tools/reporting/normalize_run.py --move
5. Aggregation

Aggregate all normalized runs:

python3 tools/reporting/aggregate_runs.py

Outputs:

runs/_aggregate/dashboard.html
runs/_aggregate/runs_summary.csv
runs/_aggregate/runs_summary.parquet
6. Unified Reporting (Recommended)

Run full pipeline:

make report

This performs:

Normalize runs

Aggregate runs

Generate dashboards

Open:

runs/_aggregate/dashboard.html
7. Evaluation & Analysis
7.1 Feature-Based Rerank Evaluation
python3 -m src.rerank_eval \
  --topk_parquet runs/<run_id>/artifacts/tables/val_topk.parquet \
  --h3_features data/index/h3_features.parquet
7.2 Inspect Hard Errors
python3 tools/analysis/inspect_top_errors.py
7.3 Visual Diagnostics
python3 tools/visualize/map_val_diagnostics.py
8. Quality Gates

Compile all source files:

python3 -m compileall src

Search for legacy imports:

rg "from \.(dataset|model|geo_loss|class_weights|hardneg|hierarchy|plots) import" -n src

Should return nothing outside _legacy/.

9. Quick Workflow

Typical iteration:

# train
python3 -m src.run

# normalize + aggregate
make report

# open results
open runs/_aggregate/dashboard.html
10. Cleaning

Remove all runs:

rm -rf runs/2026-*

Remove aggregate dashboard:

rm -rf runs/_aggregate
Summary

Core commands:

python3 -m src.run
make report

Everything else is optional tooling.