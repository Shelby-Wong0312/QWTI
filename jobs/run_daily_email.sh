#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ec2-user/Data"
VENV_PY="$PROJECT_ROOT/warehouse/.venv/bin/python"

cd "$PROJECT_ROOT"

echo "[INFO] === Daily WTI report email ==="

if [ -f "send_daily_email.py" ]; then
  echo "[INFO] Using send_daily_email.py at repo root"
  "$VENV_PY" send_daily_email.py
elif [ -f "jobs/send_daily_email.py" ]; then
  echo "[INFO] Using jobs/send_daily_email.py"
  "$VENV_PY" jobs/send_daily_email.py
else
  echo "[ERROR] send_daily_email.py not found (tried ./ and ./jobs/)" >&2
  exit 1
fi

echo "[INFO] Daily WTI report email done."
