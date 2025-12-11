#!/usr/bin/env bash
set -euo pipefail

# Backfill GDELT RAW -> hourly parquet for a date range, day by day (UTC).
# Usage: bash jobs/backfill_gdelt_gap.sh 2025-12-01 2025-12-06
# Notes:
# - Cleans RAW tmp each day to control disk usage on t3.micro.
# - After loop, rebuilds gdelt_hourly.parquet via run_gdelt_parquet_refresh.sh.

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <start-date-UTC YYYY-MM-DD> <end-date-UTC YYYY-MM-DD>" >&2
  echo "Example: $0 2025-12-01 2025-12-06" >&2
  exit 1
fi

START_DATE="$1"
END_DATE="$2"

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/wti}"
RAW_DIR="$PROJECT_ROOT/data/gdelt_raw_tmp"
OUT_DIR="$PROJECT_ROOT/data/gdelt_hourly_monthly"

cd "$PROJECT_ROOT"
mkdir -p "$RAW_DIR" "$OUT_DIR"

start_ts=$(date -u -d "$START_DATE" +%s)
end_ts=$(date -u -d "$END_DATE" +%s)

if [ "$start_ts" -ge "$end_ts" ]; then
  echo "[ERROR] start-date must be earlier than end-date" >&2
  exit 1
fi

echo "[INFO] Backfill window UTC: $START_DATE -> $END_DATE (exclusive of END)"

cur_ts="$start_ts"
while [ "$cur_ts" -lt "$end_ts" ]; do
  day_start_iso=$(date -u -d "@$cur_ts" +"%Y-%m-%dT00:00")
  next_ts=$((cur_ts + 86400))
  day_end_iso=$(date -u -d "@$((next_ts-60))" +"%Y-%m-%dT%H:%M")

  echo "[INFO] === $day_start_iso -> $day_end_iso ==="

  echo "[INFO] [1/3] Downloading RAW ..."
  /usr/bin/python3 jobs/pull_gdelt_http_to_csv.py \
    --from "$day_start_iso" \
    --to "$day_end_iso" \
    --output "data/gdelt_raw_tmp" \
    --log-level INFO

  echo "[INFO] [2/3] Aggregating buckets ..."
  /usr/bin/python3 gdelt_gkg_bucket_aggregator.py \
    --start "$day_start_iso" \
    --end   "$day_end_iso" \
    --raw-dir "data/gdelt_raw_tmp" \
    --out-dir "data/gdelt_hourly_monthly"

  echo "[INFO] [3/3] Cleaning RAW tmp ..."
  rm -f "$RAW_DIR"/* || true

  cur_ts="$next_ts"
done

echo "[INFO] Rebuilding gdelt_hourly.parquet ..."
bash jobs/run_gdelt_parquet_refresh.sh

echo "[INFO] Backfill completed for $START_DATE -> $END_DATE."
