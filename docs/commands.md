# geo-locate-ml — Command Reference

This document lists the canonical commands to operate the project.

The project is organized as:

- `src/` → core runtime (train, predict, report)
- `tools/` → offline utilities (dataset build, analysis, rerank, reporting)
- `runs/` → experiment outputs
- `models/` → global best checkpoint

---

# 1️⃣ Dataset Pipeline

Rebuild dataset index + labels:

    make rebuild

Or manually:

    python -m tools.dataset.build_index
    python -m tools.dataset.make_splits
    python -m tools.dataset.merge_splits

Expected outputs:
- data/index/images.parquet
- data/index/images_kept.parquet
- data/index/labels.json

---

# 2️⃣ Train

Run a full training experiment:

    python -m src.run

This will:

- create a new directory in `runs/<timestamp>/`
- train model
- compute geo metrics
- generate plots
- update `runs/latest`
- update `models/best.pt` if improved

Outputs:

runs/<timestamp>/
- config.json
- labels.json
- images_train.parquet
- dist_km.pt (if geo_loss enabled)
- checkpoints/
- metrics.csv
- metrics_loss.png
- metrics_valacc.png
- REPORT.md

---

# 3️⃣ Predict

Predict on a specific image:

    python -m src.predict path/to/image.jpg

Predict random sample from training parquet:

    python -m src.predict

With ensemble over multiple sizes:

    python -m src.predict path/to/image.jpg --ensemble --sizes 64,128,192

---

# 4️⃣ Reports

Generate run-level report artifacts:

    make report

Aggregate all runs into a global dashboard:

    python -m tools.reporting.aggregate_runs

Outputs:

runs/_aggregate/
- dashboard.html
- runs_summary.parquet
- plots/

---

# 5️⃣ Rerank (offline only)

Rerank is NOT part of core runtime.

Evaluate feature-based reranking:

    python -m tools.rerank.rerank_eval \
        --topk_parquet runs/<run_id>/val_topk.parquet \
        --h3_features data/index/h3_features.parquet

This does not modify models or checkpoints.
It is analysis-only.

---

# 6️⃣ Clean

Remove Python cache:

    find . -type d -name "__pycache__" -prune -exec rm -rf {} +

Recompile modules:

    python -m compileall src
    python -m compileall tools

---

# 7️⃣ Git Workflow

Train on feature branch:

    git checkout -b feature/xyz

Merge into main:

    git checkout main
    git merge --no-ff feature/xyz

Delete branch:

    git branch -d feature/xyz
    git push origin --delete feature/xyz

---

# 8️⃣ Debug Checklist

If `python -m src.run` fails:

- Check labels.json exists
- Check images_kept.parquet exists
- Verify label_idx max < num_classes
- Ensure models/best.json format is valid

---

# 9️⃣ Project Philosophy

- `src/` must remain minimal and stable.
- No experimental logic inside core.
- All analysis tools go in `tools/`.
- `_legacy/` is read-only reference.
