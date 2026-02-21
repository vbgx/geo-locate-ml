#!/usr/bin/env bash
set -euo pipefail

CSV="${1:-data/cities/cities_fr.csv}"

# Safe settings (override via env)
SLEEP="${SLEEP:-1.20}"                       # request pacing
PAUSE_BETWEEN_CITIES="${PAUSE_BETWEEN_CITIES:-20}"  # pause between cities
MAX_RETRIES_CITY="${MAX_RETRIES_CITY:-3}"     # retries per city

# Python interpreter (override via env)
PY="${PY:-python3}"
DL="$PY tools/mapillary_download.py"

if [ ! -f "$CSV" ]; then
  echo "Missing CSV: $CSV" >&2
  exit 1
fi

echo "Batch download from: $CSV"
echo "PY=$PY"
echo "sleep=$SLEEP pause_between_cities=$PAUSE_BETWEEN_CITIES max_retries_city=$MAX_RETRIES_CITY"
echo ""

# Skip header, read CSV: name,lon_min,lat_min,lon_max,lat_max,limit
tail -n +2 "$CSV" | while IFS=, read -r name lon_min lat_min lon_max lat_max limit; do
  name="$(echo "$name" | xargs)"
  bbox="${lon_min},${lat_min},${lon_max},${lat_max}"
  limit="$(echo "$limit" | xargs)"

  echo "======================================="
  echo "CITY: $name"
  echo "bbox=$bbox limit=$limit sleep=$SLEEP"
  echo "======================================="

  attempt=1
  while true; do
    set +e
    ${DL} --bbox "$bbox" --limit "$limit" --sleep "$SLEEP" --city "$name"
    code=$?
    set -e

    if [ $code -eq 0 ]; then
      echo "OK: $name"
      break
    fi

    if [ $attempt -ge $MAX_RETRIES_CITY ]; then
      echo "FAIL: $name after $attempt attempts (exit $code). Skipping."
      break
    fi

    backoff=$((60 * attempt))
    echo "WARN: $name failed (exit $code). Backoff ${backoff}s then retry (${attempt}/${MAX_RETRIES_CITY})..."
    sleep "$backoff"
    attempt=$((attempt+1))
  done

  echo "Index lines so far:"
  wc -l data/index/images.jsonl || true

  echo "Pause between cities: ${PAUSE_BETWEEN_CITIES}s"
  sleep "$PAUSE_BETWEEN_CITIES"
done

echo ""
echo "DONE."
wc -l data/index/images.jsonl || true
