#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ec2-user/Data"
VENV_PY="$PROJECT_ROOT/warehouse/.venv/bin/python"

RAW_DIR="$PROJECT_ROOT/data/gdelt_raw_tmp"
OUT_DIR="$PROJECT_ROOT/data/gdelt_hourly_monthly"

cd "$PROJECT_ROOT"

# ??????????UTC?
# ????? 16:10?"1 hour ago" ? 15:10?
# HOUR_START = 15:00, HOUR_END = 15:59
HOUR_START=$(date -u -d "1 hour ago" "+%Y-%m-%dT%H:00")
HOUR_END=$(date -u -d "1 hour ago" "+%Y-%m-%dT%H:59")

echo "[INFO] === GDELT hourly incremental ==="
echo "[INFO] WINDOW UTC: $HOUR_START -> $HOUR_END"
echo "[INFO] RAW_DIR    = $RAW_DIR"
echo "[INFO] OUT_DIR    = $OUT_DIR"

mkdir -p "$RAW_DIR" "$OUT_DIR"

# 1) ??????? GDELT RAW ?????
echo "[INFO] Downloading GDELT RAW for window $HOUR_START -> $HOUR_END ..."
"$VENV_PY" jobs/pull_gdelt_http_to_csv.py \
  --from "$HOUR_START" \
  --to "$HOUR_END" \
  --output "data/gdelt_raw_tmp" \
  --log-level INFO

# 2) ? aggregator ??????? ALL bucket ???????? parquet?
echo "[INFO] Aggregating buckets for window $HOUR_START -> $HOUR_END ..."
"$VENV_PY" jobs/gdelt_gkg_bucket_aggregator.py \
  --start "$HOUR_START" \
  --end   "$HOUR_END" \
  --raw-dir "data/gdelt_raw_tmp" \
  --out-dir "data/gdelt_hourly_monthly"

# 3) ???? RAW??????????
echo "[INFO] Cleaning RAW_DIR=$RAW_DIR ..."
rm -f "$RAW_DIR"/* || true

echo "[INFO] GDELT hourly incremental DONE."
