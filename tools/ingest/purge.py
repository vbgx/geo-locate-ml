from __future__ import annotations

import json
import shutil
from pathlib import Path


def main() -> None:
    # --- Chemins ---
    jsonl_path = Path("data/index/images.jsonl")
    processed_dir = Path("data/processed/mapillary")
    target_dir = Path("data/legacy_bad_data")

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {jsonl_path}")

    if not processed_dir.exists():
        raise FileNotFoundError(f"Missing processed directory: {processed_dir}")

    # Crée le dossier cible si nécessaire
    target_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    missing = 0
    total = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            quality = row.get("quality")
            if quality in {"OK", "BAD"}:
                img_id = row.get("id")
                if not img_id:
                    continue

                src = processed_dir / f"{img_id}.jpg"
                dst = target_dir / f"{img_id}.jpg"

                if src.exists():
                    shutil.move(str(src), str(dst))
                    moved += 1
                else:
                    missing += 1

    print("\nPurge terminé.")
    print(f"Lignes traitées : {total}")
    print(f"Fichiers déplacés : {moved}")
    print(f"Fichiers manquants : {missing}")


if __name__ == "__main__":
    main()
