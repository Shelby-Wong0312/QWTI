#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ec2-user/Data"
VENV_PY="$PROJECT_ROOT/warehouse/.venv/bin/python"

OUT_DIR="$PROJECT_ROOT/data/gdelt_hourly_monthly"
AGG_PARQUET="$PROJECT_ROOT/data/gdelt_hourly.parquet"
AGG_CSV="$PROJECT_ROOT/data/gdelt_hourly.csv"

echo "[INFO] GDELT parquet-only refresh start"
echo "[INFO] PROJECT_ROOT = $PROJECT_ROOT"
echo "[INFO] OUT_DIR      = $OUT_DIR"

cd "$PROJECT_ROOT"

# 1) ??? monthly parquet ?????? gdelt_hourly.parquet
echo "[INFO] Concatenating monthly parquets into $AGG_PARQUET ..."

"$VENV_PY" - << 'PY'
import pandas as pd
from pathlib import Path
from pandas.api.types import is_datetime64_any_dtype

in_dir = Path("data/gdelt_hourly_monthly")
out_parq = Path("data/gdelt_hourly.parquet")
out_csv  = Path("data/gdelt_hourly.csv")

files = sorted(in_dir.glob("gdelt_hourly_20*.parquet"))
if not files:
    print("[PY] ERROR: no monthly parquet files under", in_dir)
    raise SystemExit(1)

print("[PY] concatenating", len(files), "files:")
for f in files:
    print("   ", f)

dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# ?????? ts ?????? datetime index
if "ts" in df.columns and is_datetime64_any_dtype(df["ts"]):
    df = df.sort_values("ts")
elif isinstance(df.index, pd.DatetimeIndex):
    df = df.sort_index()

df.to_parquet(out_parq, index=False)
print("[PY] wrote parquet", out_parq, "rows =", len(df))

# 2) ? parquet DataFrame ?? CSV???? features_term_crack_ovx.py ?
# ??? ts ??
if isinstance(df.index, pd.DatetimeIndex) and "ts" not in df.columns:
    df = df.reset_index().rename(columns={df.columns[0]: "ts"})
elif "ts" not in df.columns:
    dt_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df = df.rename(columns={dt_cols[0]: "ts"})
    else:
        df = df.rename(columns={df.columns[0]: "ts"})

# ???????art_cnt / tone_avg / tone_pos_ratio
df["art_cnt"]        = df["ALL_art_cnt"]
df["tone_avg"]       = df["ALL_tone_avg"]
df["tone_pos_ratio"] = df["ALL_tone_pos_ratio"]

df.to_csv(out_csv, index=False)
print("[PY] wrote CSV", out_csv, "rows =", len(df))
print("[PY] min ts =", df["ts"].min())
print("[PY] max ts =", df["ts"].max())
PY

echo "[INFO] GDELT parquet-only refresh DONE."
