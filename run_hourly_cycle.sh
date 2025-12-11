#!/usr/bin/env bash
set -euo pipefail

# --- 基本設定 ---
PROJECT_ROOT="/home/ec2-user/Data"
VENV_PATH="$PROJECT_ROOT/warehouse/.venv"
LOG_DIR="$PROJECT_ROOT/warehouse/monitoring"
LOG_FILE="$LOG_DIR/hourly_monitor.log"

mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT"

# 啟動 venv（如果存在）
if [ -d "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

# 執行每小時監控
python warehouse/monitoring/hourly_monitor.py \
  --features-path features_hourly_with_term.parquet \
  >> "$LOG_FILE" 2>&1