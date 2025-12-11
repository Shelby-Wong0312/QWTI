#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ec2-user/Data"
cd "$PROJECT_ROOT"

LOG_DIR="capital_wti_downloader/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/capital_refresh.log"

{
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] START CAPITAL REFRESH"

  echo "[1/4] Download WTI hourly from Capital.com (DEMO) ..."
  warehouse/.venv/bin/python capital_wti_downloader/main.py

  echo "[2/4] Rebuild term_crack_ovx_hourly from local prices ..."
  warehouse/.venv/bin/python jobs/make_term_crack_ovx_from_local.py

  echo "[3/4] Rebuild features_hourly_v2.csv from term_crack_ovx + GDELT ..."
  warehouse/.venv/bin/python jobs/features_term_crack_ovx.py

  echo "[4/4] Update features_hourly_with_term.parquet for monitor ..."
  warehouse/.venv/bin/python warehouse/monitoring/update_features_snapshot.py

  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] DONE CAPITAL REFRESH"
} >> "$LOG_FILE" 2>&1