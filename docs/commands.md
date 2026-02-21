# geo-locate-ml — Commands

This page is a practical cheat sheet: what to run, in what order, and what each command does.

---

## Prerequisites 

Activate the virtual environment:

```bash
source .venv/bin/activate
```

All commands assume you run them from the repository root.

Quick start (one batch)

Run a full cycle for one cities CSV file:

make batch BATCH=data/cities/cities_batch_01.csv

If you need to free disk space afterwards (deletes only raw images):

make clean-raw
Inspect project state

Show the repository tree (excluding raw images):

make tree

Show disk usage:

make sizes
Download images

Download images for one batch CSV:

make download BATCH=data/cities/cities_batch_01.csv

Tune pacing (reduce API errors):

SLEEP=1.5 PAUSE_BETWEEN_CITIES=25 make download BATCH=data/cities/cities_batch_01.csv
Build dataset (index + splits)

Rebuild the dataset index and splits:

make rebuild

Change H3 parameters:

H3_RES=8 MIN_CELL_SAMPLES=40 make rebuild

Run only the sanity checks:

make sanity

Outputs (canonical dataset state):

data/index/images.jsonl

data/index/images.parquet

data/index/splits.parquet

data/index/stats.json

Train

Train using the configured training entrypoint:

make train

Training outputs:

runs/<timestamp>/ (artifacts for this run)

runs/latest (symlink to the most recent run)

models/best.pt and models/best.json (global best)

Export snapshot

Export a snapshot of the latest run + dataset index + best model:

make export

Creates:

exports/run_<timestamp>/

exports/index_<timestamp>/

exports/best_<timestamp>.pt

exports/best_<timestamp>.json

Remove raw images (keep indexes)

Delete only the downloaded images in data/raw/mapillary/:

make clean-raw

This keeps dataset index files intact.

Run multiple batches (01 → 10)

Process batches in sequence (does not delete raw images automatically):

make loop START=1 END=10

You can pause and manually delete raw images anytime:

make clean-raw
Recommended routine

For each batch:

make batch BATCH=data/cities/cities_batch_01.csv
make clean-raw

Then move to the next file (cities_batch_02.csv, etc).

Troubleshooting

If downloads fail:

Increase SLEEP

Increase PAUSE_BETWEEN_CITIES

If training looks wrong:

make sanity

Then inspect:

runs/latest/REPORT.md



python tools/pipeline/fill_communes_to_100.py \
  --input-csv data/index/coverage/good_communes.csv \
  --target 100


python tools/pipeline/scan_communes_coverage_mt.py \
  --workers 16 \
  --rate 2.0 \
  --limit 1 \
  --retries 2 \
  --backoff 1.2 \
  --timeout-connect 4 \
  --timeout-read 10 \
  --sleep 2 \
  --heartbeat-s 10 \
  --print-every 500 \
  --flush-every 200

find data/raw/mapillary -type f -iname "*.jpg" | wc -l

python -m src.rerank_eval --beta 0.35