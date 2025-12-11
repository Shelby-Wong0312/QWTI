#!/usr/bin/env bash
set -euo pipefail

# 每天跑一次：抓昨天的 GDELT RAW + 用 gdelt_gkg_bucket_aggregator.py 聚合進 monthly parquet

PROJECT_ROOT="/home/ec2-user/Data"
VENV_PY="$PROJECT_ROOT/warehouse/.venv/bin/python"

RAW_DIR="$PROJECT_ROOT/data/gdelt_raw_tmp"        # 暫存 RAW，用完就清
OUT_DIR="$PROJECT_ROOT/data/gdelt_hourly_monthly"

YDATE=$(date -u -d "yesterday" "+%Y-%m-%d")       # 昨天 (UTC)，例如 2025-11-30

echo "[INFO] === GDELT daily refresh for $YDATE ==="
echo "[INFO] PROJECT_ROOT = $PROJECT_ROOT"
echo "[INFO] RAW_DIR      = $RAW_DIR"
echo "[INFO] OUT_DIR      = $OUT_DIR"

mkdir -p "$RAW_DIR" "$OUT_DIR"
cd "$PROJECT_ROOT"

# 1) 抓昨天一整天的 GDELT RAW 到 RAW_DIR
echo "[INFO] Downloading GDELT RAW for $YDATE ..."
"$VENV_PY" jobs/pull_gdelt_http_to_csv.py \
  --from "$YDATE" \
  --to "$YDATE" \
  --output "$RAW_DIR" \
  --log-level INFO

# 2) 用 Codex 給的 ALL bucket 聚合腳本：gdelt_gkg_bucket_aggregator.py
#    CLI: python3 gdelt_gkg_bucket_aggregator.py --start YYYY-MM-DD --end YYYY-MM-DD \
#                                               --raw-dir data/gdelt_raw --out-dir data/gdelt_hourly_monthly
#    這裡我們把 raw-dir 指到 RAW_DIR（暫存），out-dir 指到 OUT_DIR
echo "[INFO] Aggregating buckets for $YDATE ..."
"$VENV_PY" jobs/gdelt_gkg_bucket_aggregator.py \
  --start "$YDATE" \
  --end "$YDATE" \
  --raw-dir "data/gdelt_raw_tmp" \
  --out-dir "data/gdelt_hourly_monthly"

# 3) 清掉暫存 RAW，避免磁碟爆掉
echo "[INFO] Cleaning RAW_DIR=$RAW_DIR ..."
rm -f "$RAW_DIR"/* || true

echo "[INFO] GDELT daily refresh for $YDATE done."
