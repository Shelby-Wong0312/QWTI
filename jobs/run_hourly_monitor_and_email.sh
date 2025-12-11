#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ec2-user/Data"
VENV_PY="$PROJECT_ROOT/warehouse/.venv/bin/python"

cd "$PROJECT_ROOT"

echo "[INFO] === Hourly WTI monitor + email ==="

# 1) GDELT incremental (hourly window)
bash jobs/run_gdelt_hourly_incremental.sh

# 2) Refresh gdelt hourly parquet aggregation
bash jobs/run_gdelt_parquet_refresh.sh

# 3) Run hourly monitor (prediction, position, IC, Hard gate check)
echo "[INFO] Running hourly_monitor.py..."
"$VENV_PY" warehouse/monitoring/hourly_monitor.py

# 4) Recompute rolling IC/IR/PMR metrics for email/monitoring
"$VENV_PY" jobs/run_hourly_metrics_refresh.py

# 5) Send hourly email
if [ -f "send_hourly_email.py" ]; then
  echo "[INFO] Using send_hourly_email.py at repo root"
  "$VENV_PY" send_hourly_email.py
elif [ -f "jobs/send_hourly_email.py" ]; then
  echo "[INFO] Using jobs/send_hourly_email.py"
  "$VENV_PY" jobs/send_hourly_email.py
else
  echo "[ERROR] send_hourly_email.py not found (tried ./ and ./jobs/)" >&2
  exit 1
fi

echo "[INFO] Hourly WTI monitor done."
