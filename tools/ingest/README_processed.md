# scan_processed_quality.py — quick usage

This repo includes a small utility to **scan `data/processed/mapillary/`** and **write per-image quality tags** into the index file `data/index/images.jsonl`.

The scanner adds/overwrites **three fields** on each indexed image row:

- `quality` — `GOOD | OK | BAD`
- `quality_level` — the scan level used (1..5)
- `comments` — compact explanation + a few metrics

> Important: the index file is written **atomically** (via a `.tmp` file then replace), so you will **not** see it “change line-by-line” while the scan runs.

---

## What it scans

- Folder: `data/processed/mapillary/**`
- Matches images to index rows by **ID = filename stem**  
  Example: `data/processed/mapillary/1408239167678528.jpg` → `id="1408239167678528"`

Supported image extensions: `.jpg .jpeg .png .webp .bmp .tif .tiff`

---

## Levels (cumulative)

`--level N` runs all checks up to N.

- **L1**: decode + dimensions + aspect ratio + RGB/alpha checks
- **L2**: blur (laplacian var) + dark/bright saturation + low contrast
- **L3**: preprocessing compatibility (resize + crop to model input size)
- **L4**: entropy (information content)
- **L5**: edge density (fast mode: computed on a downscaled grayscale image)

---

## Most common commands

### 1) Normal run (writes results + makes a backup)
Pick the level you want:

```bash
python tools/ingest/scan_processed_quality.py --level 2
python tools/ingest/scan_processed_quality.py --level 5
2) Dry-run (no write)

Useful to test thresholds and see counts:

python tools/ingest/scan_processed_quality.py --level 5 --dry-run
3) Show progress more often
python tools/ingest/scan_processed_quality.py --level 5 --log-every 5
4) Verbose (one line per image)
python tools/ingest/scan_processed_quality.py --level 3 --verbose
Output files
Index updated

data/index/images.jsonl (updated in-place, atomically)

Report file

reports/scan_processed_quality.jsonl
One row per scanned image, includes:

id

path

quality

quality_level

comments

Backups and restore

If --backup-index is enabled, the script creates:

data/index/images.jsonl.bak-YYYYMMDD-HHMMSS

List backups:

ls -lt data/index/images.jsonl.bak-*

Restore a backup:

cp data/index/images.jsonl.bak-YYYYMMDD-HHMMSS data/index/images.jsonl
How to exclude bad samples in training

Typical filters:

Exclude all BAD:

quality != "BAD"

Keep only strict scans:

quality_level >= 5 and quality != "BAD"

Exact filtering depends on your training pipeline (pandas, dataset loader, etc.).

Notes / gotchas

The scanner writes the index at the end (atomic replace). This is expected.

If you run level 2 then later run level 5, the latest run overwrites:

quality

quality_level

comments

If you see “no output for a while”, increase logging frequency:

--log-every 1

or run with --verbose

Quick sanity checks

Count results in index:

grep -c '"quality": "GOOD"' data/index/images.jsonl
grep -c '"quality": "OK"'   data/index/images.jsonl
grep -c '"quality": "BAD"'  data/index/images.jsonl

See a few BAD comments:

grep '"quality": "BAD"' data/index/images.jsonl | head