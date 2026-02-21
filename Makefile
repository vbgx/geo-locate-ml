SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -euo pipefail -c

# =============================================================================
# geo-locate-ml — Makefile (pipeline-safe)
# =============================================================================

# -----------------------
# Interpreter (always use venv)
# -----------------------
PY ?= .venv/bin/python3

# -----------------------
# Mode: communes (default) or cities
# -----------------------
MODE ?= communes

# -----------------------
# Download (cities)
# -----------------------
CITIES_BATCH ?= data/cities/cities_batch_01.csv
SLEEP ?= 1.5
PAUSE_BETWEEN_CITIES ?= 25
MAX_RETRIES_CITY ?= 3

# -----------------------
# Download (communes)
# -----------------------
COMMUNES_BATCH ?= data/communes/batches/communes_batch_001.csv
BATCH_DIR ?= data/communes/batches
STATE_DIR ?= data/state
LOG_DIR ?= logs/download

PAUSE_BETWEEN_COMMUNES ?= 0
MIN_PER_COMMUNE ?= 10
MAX_RETRIES_COMMUNE ?= 3

# forwarded as --sleep
DOWNLOADER_SLEEP ?= $(SLEEP)

# aggressive fill behavior
FILL_LIMIT ?= 500
FILL_MIN_NEED ?= 1

# -----------------------
# Dataset build
# -----------------------
H3_RES ?= 7
MIN_CELL_SAMPLES ?= 30

# -----------------------
# Training
# -----------------------
TRAIN_CMD ?= $(PY) -m src.run

# -----------------------
# Loop range (pipeline_all_batches.sh)
# -----------------------
START ?= 1
END ?= 10

# -----------------------
# Compression raw -> processed
# -----------------------
COMPRESS_MAX_SIZE ?= 512
COMPRESS_QUALITY  ?= 80
COMPRESS_WORKERS  ?= 6
COMPRESS_ARGS ?=

# -----------------------
# Tool paths (single source of truth)
# -----------------------
CITY_DOWNLOADER_SH := tools/ingest/cities_batch_download.sh

COMMUNES_RUNNER_SH := tools/pipeline/download_communes_resume.sh
COMMUNES_ENSURE_PY := tools/pipeline/ensure_min_images_per_commune_then_download.py

GEN_CITY_BATCHES_PY := tools/ingest/generate_city_batches.py

BUILD_INDEX_PY := tools/dataset/build_index.py
MAKE_SPLITS_PY := tools/dataset/make_splits.py
MERGE_SPLITS_PY := tools/dataset/merge_splits.py
SANITY_PY := tools/dataset/sanity_check.py

EXPORT_SH := tools/ops/export_snapshot.sh
PURGE_RAW_SH := tools/ops/purge_images_keep_index.sh

PIPELINE_ALL_SH := tools/pipeline/pipeline_all_batches.sh

COMPRESS_PY := tools/ops/compress_raw_to_processed.py

# =============================================================================
# Help
# =============================================================================
.PHONY: help
help:
	@echo "geo-locate-ml — Targets"
	@echo ""
	@echo "Download (MODE=communes|cities):"
	@echo "  make download                                  # route by MODE (default: communes)"
	@echo "  make download-communes                         # resume-safe: next batch + logs + state"
	@echo "  make download-communes-once COMMUNES_BATCH=... # one batch only (no resume)"
	@echo "  make download-cities   CITIES_BATCH=...         # one cities CSV batch"
	@echo ""
	@echo "Batch generation:"
	@echo "  make generate_city_batches                     # regenerate city bboxes/batches"
	@echo ""
	@echo "Dataset:"
	@echo "  make rebuild                                   # build index + splits + merge + sanity"
	@echo "  make build-index                               # jsonl -> images.parquet (full)"
	@echo "  make make-splits                               # images.parquet -> splits.parquet"
	@echo "  make merge-splits                              # images.parquet + splits -> images.parquet + images_kept.parquet + labels.json"
	@echo "  make sanity                                    # sanity check on images_kept.parquet"
	@echo ""
	@echo "Training:"
	@echo "  make train                                     # run training ($(TRAIN_CMD))"
	@echo ""
	@echo "Exports / cleanup:"
	@echo "  make export                                    # snapshot (runs/latest + data/index + models/best.*)"
	@echo "  make clean-raw                                 # delete raw images only (keeps indexes)"
	@echo ""
	@echo "Inspect:"
	@echo "  make sizes                                     # disk usage summary"
	@echo "  make tree                                      # show repo tree (excluding heavy dirs)"
	@echo "  make counts                                    # index vs on-disk counts"
	@echo ""
	@echo "Pipelines:"
	@echo "  make batch                                     # generate_city_batches + download + rebuild + train + export"
	@echo "  make batch-clean                               # batch + clean-raw"
	@echo "  make loop START=1 END=10                       # run pipeline_all_batches.sh"
	@echo ""
	@echo "Compression:"
	@echo "  make compress                                  # raw -> processed"
	@echo "  make compress COMPRESS_ARGS=\"--limit 50\"        # pass args to tool"
	@echo "  make compress-delete                           # raw -> processed + delete raw"
	@echo ""
	@echo "Key variables:"
	@echo "  PY=$(PY)"
	@echo "  MODE=$(MODE)"
	@echo "  CITIES_BATCH=$(CITIES_BATCH)"
	@echo "  COMMUNES_BATCH=$(COMMUNES_BATCH)"
	@echo "  BATCH_DIR=$(BATCH_DIR) STATE_DIR=$(STATE_DIR) LOG_DIR=$(LOG_DIR)"
	@echo "  SLEEP=$(SLEEP) DOWNLOADER_SLEEP=$(DOWNLOADER_SLEEP)"
	@echo "  MIN_PER_COMMUNE=$(MIN_PER_COMMUNE) FILL_LIMIT=$(FILL_LIMIT) FILL_MIN_NEED=$(FILL_MIN_NEED)"
	@echo "  H3_RES=$(H3_RES) MIN_CELL_SAMPLES=$(MIN_CELL_SAMPLES)"
	@echo "  TRAIN_CMD=$(TRAIN_CMD)"

# =============================================================================
# Safety checks
# =============================================================================
.PHONY: check-venv check-tools
check-venv:
	@if [ ! -x "$(PY)" ]; then \
	  echo "ERROR: venv Python not found at $(PY)"; \
	  echo ""; \
	  echo "Create it:"; \
	  echo "  python3 -m venv .venv"; \
	  echo "  source .venv/bin/activate"; \
	  echo "  $(PY) -m pip install -r requirements.txt"; \
	  exit 1; \
	fi

check-tools:
	@missing=0; \
	for f in "$(CITY_DOWNLOADER_SH)" "$(COMMUNES_RUNNER_SH)" "$(COMMUNES_ENSURE_PY)" "$(GEN_CITY_BATCHES_PY)" \
	         "$(BUILD_INDEX_PY)" "$(MAKE_SPLITS_PY)" "$(MERGE_SPLITS_PY)" "$(SANITY_PY)" \
	         "$(EXPORT_SH)" "$(PURGE_RAW_SH)" "$(PIPELINE_ALL_SH)" "$(COMPRESS_PY)"; do \
	  if [ ! -e "$$f" ]; then echo "ERROR: missing $$f"; missing=1; fi; \
	done; \
	if [ "$$missing" -ne 0 ]; then exit 1; fi

# =============================================================================
# Download (router)
# =============================================================================
.PHONY: download
download: check-venv
	@if [ "$(MODE)" = "cities" ]; then \
	  $(MAKE) download-cities; \
	elif [ "$(MODE)" = "communes" ]; then \
	  $(MAKE) download-communes; \
	else \
	  echo "ERROR: MODE must be 'communes' or 'cities' (got: $(MODE))"; \
	  exit 2; \
	fi

# =============================================================================
# Download — cities
# =============================================================================
.PHONY: download-cities
download-cities: check-venv
	@if [ ! -x "$(CITY_DOWNLOADER_SH)" ]; then \
	  echo "ERROR: missing or not executable: $(CITY_DOWNLOADER_SH)"; \
	  exit 1; \
	fi
	@echo "Downloading CITIES from: $(CITIES_BATCH)"
	@echo "PY=$(PY) sleep=$(SLEEP) pause_between_cities=$(PAUSE_BETWEEN_CITIES) max_retries_city=$(MAX_RETRIES_CITY)"
	@PY="$(PY)" SLEEP="$(SLEEP)" PAUSE_BETWEEN_CITIES="$(PAUSE_BETWEEN_CITIES)" MAX_RETRIES_CITY="$(MAX_RETRIES_CITY)" \
	  "$(CITY_DOWNLOADER_SH)" "$(CITIES_BATCH)"

# =============================================================================
# Download — communes (resume-safe default)
# =============================================================================
.PHONY: download-communes
download-communes: check-venv
	@if [ ! -x "$(COMMUNES_RUNNER_SH)" ]; then \
	  echo "ERROR: missing or not executable: $(COMMUNES_RUNNER_SH)"; \
	  echo "You must create it: tools/pipeline/download_communes_resume.sh"; \
	  exit 1; \
	fi
	@echo "Downloading COMMUNES (resume mode) from: $(BATCH_DIR)/*.csv"
	@PY="$(PY)" \
	  BATCH_DIR="$(BATCH_DIR)" STATE_DIR="$(STATE_DIR)" LOG_DIR="$(LOG_DIR)" \
	  MIN_PER_COMMUNE="$(MIN_PER_COMMUNE)" \
	  PAUSE_BETWEEN_COMMUNES="$(PAUSE_BETWEEN_COMMUNES)" \
	  MAX_RETRIES_COMMUNE="$(MAX_RETRIES_COMMUNE)" \
	  DOWNLOADER_SLEEP="$(DOWNLOADER_SLEEP)" \
	  FILL_LIMIT="$(FILL_LIMIT)" \
	  FILL_MIN_NEED="$(FILL_MIN_NEED)" \
	  "$(COMMUNES_RUNNER_SH)"

# =============================================================================
# Download — communes (one-shot explicit CSV)
# =============================================================================
.PHONY: download-communes-once
download-communes-once: check-venv
	@if [ ! -f "$(COMMUNES_ENSURE_PY)" ]; then \
	  echo "ERROR: missing $(COMMUNES_ENSURE_PY)"; \
	  exit 1; \
	fi
	@echo "Downloading COMMUNES from: $(COMMUNES_BATCH) (one-shot)"
	@echo "PY=$(PY) sleep=$(DOWNLOADER_SLEEP) pause_between_communes=$(PAUSE_BETWEEN_COMMUNES) max_retries_commune=$(MAX_RETRIES_COMMUNE) min_per_commune=$(MIN_PER_COMMUNE) fill_limit=$(FILL_LIMIT)"
	@PY="$(PY)" SLEEP="$(DOWNLOADER_SLEEP)" \
	  "$(PY)" -u "$(COMMUNES_ENSURE_PY)" \
	  "$(COMMUNES_BATCH)" \
	  --min-per-commune "$(MIN_PER_COMMUNE)" \
	  --pause-between-communes "$(PAUSE_BETWEEN_COMMUNES)" \
	  --max-retries "$(MAX_RETRIES_COMMUNE)" \
	  --sleep "$(DOWNLOADER_SLEEP)" \
	  --fill-limit "$(FILL_LIMIT)" \
	  --fill-min-need "$(FILL_MIN_NEED)"

# =============================================================================
# Batch generation
# =============================================================================
.PHONY: generate_city_batches
generate_city_batches: check-venv
	@echo "Generating city batches (data/cities/cities_fr.csv -> data/cities/...)"
	@$(PY) "$(GEN_CITY_BATCHES_PY)"

# =============================================================================
# Dataset build
# =============================================================================
.PHONY: build-index make-splits merge-splits sanity rebuild

build-index: check-venv
	@echo "Building full index -> data/index/images.parquet"
	@$(PY) "$(BUILD_INDEX_PY)" --h3-res "$(H3_RES)" --min-cell-samples "$(MIN_CELL_SAMPLES)"

make-splits: check-venv
	@echo "Building splits -> data/index/splits.parquet"
	@$(PY) -m tools.dataset.make_splits \
	  --in-parquet data/index/images.parquet \
	  --out-parquet data/index/splits.parquet

merge-splits: check-venv
	@echo "Merging splits + writing kept-only dataset + labels"
	@$(PY) -m tools.dataset.merge_splits \
	  --parquet data/index/images.parquet \
	  --splits data/index/splits.parquet \
	  --out data/index/images.parquet \
	  --out-kept data/index/images_kept.parquet \
	  --labels-out data/index/labels.json \
	  --min-cell-samples "$(MIN_CELL_SAMPLES)" \
	  --drop-missing-files

sanity: check-venv
	@$(PY) "$(SANITY_PY)" --parquet data/index/images_kept.parquet

rebuild: build-index make-splits merge-splits sanity
	@echo "✅ rebuild done"

# =============================================================================
# Train
# =============================================================================
.PHONY: train
train: check-venv
	@echo "Starting training: $(TRAIN_CMD)"
	@$(TRAIN_CMD)

# =============================================================================
# Export (ensure runs/latest symlink)
# =============================================================================
.PHONY: ensure-latest export
ensure-latest:
	@latest_run=$$(ls -1dt runs/* 2>/dev/null | head -n 1 || true); \
	if [ -n "$$latest_run" ]; then \
	  ln -sfn "$$latest_run" runs/latest; \
	  echo "runs/latest -> $$latest_run"; \
	else \
	  echo "No runs/* found. Run: make train"; \
	fi

export: ensure-latest
	@if [ ! -x "$(EXPORT_SH)" ]; then \
	  echo "ERROR: missing or not executable: $(EXPORT_SH)"; \
	  exit 1; \
	fi
	@$(EXPORT_SH)

# =============================================================================
# Cleanup raw images
# =============================================================================
.PHONY: clean-raw
clean-raw:
	@if [ ! -x "$(PURGE_RAW_SH)" ]; then \
	  echo "ERROR: missing or not executable: $(PURGE_RAW_SH)"; \
	  exit 1; \
	fi
	@echo "Purging raw images under data/raw/mapillary (keeps indexes)..."
	@$(PURGE_RAW_SH)

# =============================================================================
# Inspect
# =============================================================================
.PHONY: sizes tree counts
sizes:
	@du -sh data/raw/mapillary data/processed/mapillary runs models data/index exports 2>/dev/null || true

tree:
	@command -v tree >/dev/null 2>&1 || { echo "tree not installed (brew install tree)"; exit 1; }
	@tree -I "data/raw|data/processed|__pycache__|*.egg-info|.venv|*.jpg|*.csv|*tif"

counts: check-venv
	@echo "Indexed jsonl lines:" $$(wc -l < data/index/images.jsonl 2>/dev/null || echo 0)
	@echo "Indexed parquet rows:" $$($(PY) -c "import pathlib; import pandas as pd; p=pathlib.Path('data/index/images.parquet'); print(len(pd.read_parquet(p)) if p.exists() else 0)")
	@echo "Kept parquet rows:" $$($(PY) -c "import pathlib; import pandas as pd; p=pathlib.Path('data/index/images_kept.parquet'); print(len(pd.read_parquet(p)) if p.exists() else 0)")
	@echo "On-disk JPG files:" $$(find data/raw/mapillary -type f -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')

# =============================================================================
# Compression raw -> processed
# =============================================================================
.PHONY: compress compress-delete
compress: check-venv
	@$(PY) "$(COMPRESS_PY)" \
	  --max-size "$(COMPRESS_MAX_SIZE)" \
	  --quality "$(COMPRESS_QUALITY)" \
	  --workers "$(COMPRESS_WORKERS)" \
	  $(COMPRESS_ARGS)

compress-delete: check-venv
	@$(PY) "$(COMPRESS_PY)" \
	  --max-size "$(COMPRESS_MAX_SIZE)" \
	  --quality "$(COMPRESS_QUALITY)" \
	  --workers "$(COMPRESS_WORKERS)" \
	  --delete-raw \
	  $(COMPRESS_ARGS)

# =============================================================================
# Pipelines
# =============================================================================
.PHONY: batch batch-clean loop
batch: check-venv generate_city_batches download rebuild train export
	@echo "✅ Batch complete."

batch-clean: batch clean-raw
	@echo "✅ Batch complete + raw images removed."

loop: check-venv
	@if [ ! -x "$(PIPELINE_ALL_SH)" ]; then \
	  echo "ERROR: missing or not executable: $(PIPELINE_ALL_SH)"; \
	  exit 1; \
	fi
	@START="$(START)" END="$(END)" \
	PY="$(PY)" MODE="$(MODE)" \
	SLEEP="$(SLEEP)" DOWNLOADER_SLEEP="$(DOWNLOADER_SLEEP)" \
	PAUSE_BETWEEN_CITIES="$(PAUSE_BETWEEN_CITIES)" MAX_RETRIES_CITY="$(MAX_RETRIES_CITY)" \
	PAUSE_BETWEEN_COMMUNES="$(PAUSE_BETWEEN_COMMUNES)" MAX_RETRIES_COMMUNE="$(MAX_RETRIES_COMMUNE)" MIN_PER_COMMUNE="$(MIN_PER_COMMUNE)" \
	FILL_LIMIT="$(FILL_LIMIT)" FILL_MIN_NEED="$(FILL_MIN_NEED)" \
	H3_RES="$(H3_RES)" MIN_CELL_SAMPLES="$(MIN_CELL_SAMPLES)" \
	TRAIN_CMD="$(TRAIN_CMD)" \
	"$(PIPELINE_ALL_SH)"

# =============================================================================
# Default
# =============================================================================
.PHONY: default
default: help

.PHONY: report report-clean

report:
	@echo "==> Normalize runs ..."
	python3 tools/reporting/normalize_run.py
	@echo "==> Aggregate runs ..."
	python3 tools/reporting/aggregate_runs.py
	@echo ""
	@echo "✅ Done."
	@echo "Open: runs/_aggregate/dashboard.html"

report-clean:
	@echo "==> Normalize runs (MOVE mode) ..."
	python3 tools/reporting/normalize_run.py --move
	@echo "==> Aggregate runs ..."
	python3 tools/reporting/aggregate_runs.py
	@echo ""
	@echo "✅ Done."
	@echo "Open: runs/_aggregate/dashboard.html"
