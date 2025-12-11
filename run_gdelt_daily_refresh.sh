#!/usr/bin/env bash
set -euo pipefail

# 每天跑一次：抓昨天的 GDELT RAW + 聚合進 monthly parquet
PROJECT_ROOT="/home/ec2-user/Data"
VENV_PY="$PROJECT_ROOT/warehouse/.venv/bin/python"

RAW_DIR="$PROJECT_ROOT/data/gdelt_raw_tmp"
OUT_DIR="$PROJECT_ROOT/data/gdelt_hourly_monthly"

YDATE=$(date -u -d "yesterday" "+%Y-%m-%d")   # 昨天（UTC）
YM=$(date -u -d "$YDATE" "+%Y-%m")             # 對應年月，例如 2025-11

echo "[INFO] === GDELT daily refresh for $YDATE (month $YM) ==="
echo "[INFO] PROJECT_ROOT = $PROJECT_ROOT"
echo "[INFO] RAW_DIR      = $RAW_DIR"
echo "[INFO] OUT_DIR      = $OUT_DIR"

mkdir -p "$RAW_DIR" "$OUT_DIR"
cd "$PROJECT_ROOT"

# 1) 抓昨天一整天的 GDELT RAW（只存暫存目錄，跑完就清）
echo "[INFO] Downloading GDELT RAW for $YDATE ..."
"$VENV_PY" jobs/pull_gdelt_http_to_csv.py \
  --from "$YDATE" \
  --to   "$YDATE" \
  --output "$RAW_DIR" \
  --log-level INFO

# 2) 用 Codex 那顆「ALL bucket 聚合腳本」把 RAW 聚合成 hourly parquet（這行一定要你改）
MONTH_PARQUET="$OUT_DIR/gdelt_hourly_${YM}.parquet"

# ===== 必改區開始：用 Codex 問清楚聚合腳本名稱 & 參數 =====
# 例：jobs/gdelt_agg_all_bucket.py 之類的
# 請你把下面這行的腳本名稱和參數，改成 Codex 跟你說的正確版本
"$VENV_PY" jobs/YOUR_CODEX_GDELT_AGG_SCRIPT.py \
  --from-date "$YDATE" \
  --to-date "$YDATE" \
  --raw-dir "$RAW_DIR" \
  --out-parquet "$MONTH_PARQUET"
# ===== 必改區結束 =====

echo "[INFO] Cleaning RAW_DIR=$RAW_DIR ..."
rm -f "$RAW_DIR"/* || true

echo "[INFO] GDELT daily refresh for $YDATE done."
