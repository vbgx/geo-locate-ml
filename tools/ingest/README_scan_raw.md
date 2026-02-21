# Mapillary Review Tool

Location:
tools/ingest/scan_mapillary.py

Purpose:
Scan raw images, review problematic ones in a carousel, and apply KEEP / SKIP / DELETE decisions.

---

# 1️⃣ Generate Review Interface

Recommended:

python tools/ingest/scan_mapillary.py \
  --dry-run \
  --level 3 \
  --require-rgb \
  --html \
  --limit 300

Open:

open reports/scan_mapillary_carousel.html

Use --limit to keep the UI responsive.

---

# 2️⃣ Carousel Controls

Three buttons:

KEEP  
SKIP  
DELETE  

Keyboard shortcuts:

1 → KEEP  
2 → SKIP  
3 → DELETE  

ArrowLeft  → previous image  
ArrowRight → next image  

Each decision:
- Is saved immediately
- Automatically moves to the next image

---

# 3️⃣ Export Decisions

Click:

Export actions.jsonl

File is always saved to:

tools/ingest/actions.jsonl

(No browser download folder ambiguity.)

---

# 4️⃣ Apply Decisions

Dry run first:

python tools/ingest/scan_mapillary.py \
  --dry-run \
  --apply-actions tools/ingest/actions.jsonl

Apply for real:

python tools/ingest/scan_mapillary.py \
  --apply-actions tools/ingest/actions.jsonl

---

# 5️⃣ What Actions Do

KEEP
→ Copy image to data/processed/mapillary

SKIP
→ Do nothing

DELETE
→ Remove raw image
→ Remove matching entry in data/index/images.jsonl (match by id or path)

---

# 6️⃣ Default Paths

Raw:
data/raw/mapillary

Processed:
data/processed/mapillary

Index:
data/index/images.jsonl

---

# 7️⃣ Recommended Workflow

Batch review:

1. Scan 300 images
2. Review in carousel
3. Export actions.jsonl
4. Apply
5. Repeat

---

Always dry-run before destructive apply.