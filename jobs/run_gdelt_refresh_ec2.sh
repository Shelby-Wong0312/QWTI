#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# GDELT Refresh on EC2 (HTTP -> hourly parquet -> merged parquet + CSV)
# ==========================================

PROJECT_ROOT="/home/ec2-user/Data"
VENV_PY="$PROJECT_ROOT/warehouse/.venv/bin/python"

RAW_DIR="$PROJECT_ROOT/data/gdelt_raw_tmp"          # 臨時原始資料，跑完會清
OUT_DIR="$PROJECT_ROOT/data/gdelt_hourly_monthly"  # 每個 month 的 hourly parquet
AGG_PARQUET="$PROJECT_ROOT/data/gdelt_hourly.parquet"
AGG_CSV="$PROJECT_ROOT/data/gdelt_hourly.csv"

START_MONTH="${1:-2025-10}"   # e.g. 2025-10
END_MONTH="${2:-2025-12}"     # e.g. 2025-12

echo "[INFO] GDELT refresh start"
echo "[INFO] PROJECT_ROOT = $PROJECT_ROOT"
echo "[INFO] START_MONTH  = $START_MONTH"
echo "[INFO] END_MONTH    = $END_MONTH"

mkdir -p "$RAW_DIR" "$OUT_DIR"

cd "$PROJECT_ROOT"

# ------------------------------------------
# 1) 逐月抓 GDELT 並轉成 hourly parquet
#    ※ 這裡假設 pull_gdelt_http_to_csv.py 的介面：
#       --year-month YYYY-MM
#       --raw-dir    原始 zip/csv 存放處
#       --out-parquet 該月份的 hourly parquet
#    如果你本機是別的 flag 名稱，就把下面那行改成你本機那一行即可。
# ------------------------------------------

cur="${START_MONTH}-01"
end="${END_MONTH}-01"

while true; do
  ym=$(date -u -d "$cur" +%Y-%m)
  echo "[INFO] === Processing month $ym ==="

  month_parquet="$OUT_DIR/gdelt_hourly_${ym}.parquet"

  # 如果該月份 parquet 已存在，就先跳過（保守作法）
  if [[ -f "$month_parquet" ]]; then
    echo "[INFO] Monthly parquet already exists for $ym: $month_parquet (skip download)"
  else
    echo "[INFO] Running pull_gdelt_http_to_csv.py for $ym ..."
    "$VENV_PY" jobs/pull_gdelt_http_to_csv.py \
      --year-month "$ym" \
      --raw-dir "$RAW_DIR" \
      --out-parquet "$month_parquet"
  fi

  # 跑完這個 month 就把 RAW_DIR 清空，避免磁碟爆掉
  echo "[INFO] Cleaning RAW_DIR=$RAW_DIR for month $ym ..."
  rm -f "$RAW_DIR"/* || true

  # 判斷是否已經處理到 END_MONTH
  if [[ "$ym" == "$END_MONTH" ]]; then
    break
  fi
  # 下一個月
  cur=$(date -u -d "$cur +1 month" +%Y-%m-01)
done

# ------------------------------------------
# 2) 合併所有 monthly parquet -> data/gdelt_hourly.parquet
#    這裡用你剛丟上去的 concat_gdelt_hourly_parquet.py
# ------------------------------------------
echo "[INFO] Concatenating monthly parquets into $AGG_PARQUET ..."
"$VENV_PY" jobs/concat_gdelt_hourly_parquet.py \
  --in-dir "$OUT_DIR" \
  --out-parquet "$AGG_PARQUET"

# ------------------------------------------
# 3) 從 parquet => CSV，並補上 features_term_crack_ovx.py 期望的欄位
#    ts / art_cnt / tone_avg / tone_pos_ratio
#    這裡用 ALL_* 作為整體新聞量與情緒來源
# ------------------------------------------
echo "[INFO] Building $AGG_CSV from $AGG_PARQUET ..."

"$VENV_PY" - << 'PY'
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

path_parq = "data/gdelt_hourly.parquet"
path_csv  = "data/gdelt_hourly.csv"

df = pd.read_parquet(path_parq)

# 1) 確保有 ts 欄位
if isinstance(df.index, pd.DatetimeIndex):
    df = df.reset_index().rename(columns={df.columns[0]: "ts"})
elif "ts" not in df.columns:
    dt_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df = df.rename(columns={dt_cols[0]: "ts"})
    else:
        df = df.rename(columns={df.columns[0]: "ts"})

# 2) 給 features_term_crack_ovx.py 用的三個欄位
df["art_cnt"]        = df["ALL_art_cnt"]
df["tone_avg"]       = df["ALL_tone_avg"]
df["tone_pos_ratio"] = df["ALL_tone_pos_ratio"]

df.to_csv(path_csv, index=False)
print("[PY] wrote", path_csv)
print("[PY] rows =", len(df))
print("[PY] min ts =", df["ts"].min())
print("[PY] max ts =", df["ts"].max())
PY

echo "[INFO] GDELT refresh DONE."
