import pandas as pd
import numpy as np
from pathlib import Path
import shutil

ROOT = Path(".").resolve()

print("Root:", ROOT)

# 1) ? features_hourly_with_term.parquet
feat_candidates = list(ROOT.rglob("features_hourly_with_term.parquet"))
if not feat_candidates:
    raise FileNotFoundError("??? features_hourly_with_term.parquet?????? Data ???")

FEAT_PATH = feat_candidates[0]
print("Using features file:", FEAT_PATH)

df = pd.read_parquet(FEAT_PATH)

# 2) ????????????????
ts_series = None
for c in ["ts_utc", "ts", "timestamp", "time", "datetime"]:
    if c in df.columns:
        ts_series = pd.to_datetime(df[c], utc=True, errors="coerce")
        print(f"Using time column: {c}")
        break

if ts_series is None and isinstance(df.index, pd.DatetimeIndex):
    ts_series = pd.to_datetime(df.index, utc=True, errors="coerce")
    print("Using DatetimeIndex as time column")

if ts_series is None:
    cand = [
        c for c in df.columns
        if any(s in str(c).lower() for s in ["ts", "time", "date"])
    ]
    if len(cand) == 1:
        col = cand[0]
        ts_series = pd.to_datetime(df[col], utc=True, errors="coerce")
        print(f"Using fuzzy time column: {col}")

if ts_series is None:
    print("Columns:", list(df.columns))
    raise RuntimeError("????????ts_utc/ts/timestamp/time/datetime ???")

df["ts"] = ts_series
df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
print("Total rows after ts cleaning:", len(df))

# 3) ? momentum_24h?? wti_returns / ret_1h / returns_1h
ret_col = None
for c in ["wti_returns", "ret_1h", "returns_1h"]:
    if c in df.columns:
        ret_col = c
        break

price_col = None
if ret_col is None:
    for c in ["wti_close", "close_mid", "close", "mid", "price"]:
        if c in df.columns:
            price_col = c
            break

if ret_col is None and price_col is None:
    print("Columns:", list(df.columns))
    raise RuntimeError("??? wti_returns/ret_1h/returns_1h ???????????? momentum_24h")

if ret_col is None and price_col is not None:
    print(f"Using price column '{price_col}' to derive 1h returns for momentum_24h")
    df = df.sort_values("ts")
    df["__tmp_ret"] = df[price_col].pct_change().shift(-1)
    ret_col = "__tmp_ret"
else:
    print(f"Using return column '{ret_col}' for momentum_24h")

ret = df[ret_col]

# 24 ??????? (1+ret) ? -1
mom24 = (1.0 + ret).rolling(24, min_periods=24).apply(
    lambda x: np.prod(x), raw=True
) - 1.0

df["momentum_24h"] = mom24
print("momentum_24h filled (non-null):", int(df["momentum_24h"].notna().sum()))

# 4) ? cl_bz_spread?? term_crack_ovx_hourly.* ? merge
term_candidates = []
for pattern in ["term_crack_ovx_hourly.csv", "term_crack_ovx_hourly.parquet"]:
    term_candidates.extend(ROOT.rglob(pattern))

if not term_candidates:
    raise FileNotFoundError(
        "??? term_crack_ovx_hourly.csv / .parquet\n"
        "??? EC2 /home/ec2-user/wti/data/term_crack_ovx_hourly.csv ?????? Data ???????"
    )

TERM_PATH = term_candidates[0]
print("Using term/crack/ovx file:", TERM_PATH)

if TERM_PATH.suffix == ".csv":
    term = pd.read_csv(TERM_PATH)
else:
    term = pd.read_parquet(TERM_PATH)

# ? term ??????
t_ts = None
for c in ["ts_utc", "ts", "timestamp", "time", "datetime"]:
    if c in term.columns:
        t_ts = pd.to_datetime(term[c], utc=True, errors="coerce")
        print(f"Term file time column: {c}")
        break

if t_ts is None and isinstance(term.index, pd.DatetimeIndex):
    t_ts = pd.to_datetime(term.index, utc=True, errors="coerce")
    print("Term file using DatetimeIndex as time column")

if t_ts is None:
    cand = [
        c for c in term.columns
        if any(s in str(c).lower() for s in ["ts", "time", "date"])
    ]
    if len(cand) == 1:
        col = cand[0]
        t_ts = pd.to_datetime(term[col], utc=True, errors="coerce")
        print(f"Term file fuzzy time column: {col}")

if t_ts is None:
    print("Term columns:", list(term.columns))
    raise RuntimeError("term_crack_ovx_hourly.* ???????")

term["ts"] = t_ts
term = term.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

if "cl_bz_spread" not in term.columns:
    print("Term columns:", list(term.columns))
    raise RuntimeError("term_crack_ovx_hourly.* ??? cl_bz_spread ?????? merge")

term_small = term[["ts", "cl_bz_spread"]].dropna()

print("Term rows with cl_bz_spread:", len(term_small))

# ??? features ??? cl_bz_spread?????
if "cl_bz_spread" in df.columns:
    df = df.drop(columns=["cl_bz_spread"])

df = df.merge(term_small, on="ts", how="left")

print("After merge, cl_bz_spread non-null:", int(df["cl_bz_spread"].notna().sum()))

# ?? 2024-10~2025-12 ???????
mask_range = (df["ts"] >= pd.Timestamp("2024-10-01", tz="UTC")) & (
    df["ts"] <= pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
)
print("In 2024-10~2025-12 range:")
print("  rows:", int(mask_range.sum()))
print("  momentum_24h non-null:", int(df.loc[mask_range, "momentum_24h"].notna().sum()))
print("  cl_bz_spread non-null:", int(df.loc[mask_range, "cl_bz_spread"].notna().sum()))

# 5) ?? + ??
backup_path = FEAT_PATH.with_suffix(FEAT_PATH.suffix + ".backup")
if not backup_path.exists():
    shutil.copy2(FEAT_PATH, backup_path)
    print("Backup created:", backup_path)
else:
    print("Backup already exists:", backup_path)

df.to_parquet(FEAT_PATH)
print("Updated features saved to:", FEAT_PATH)

print("Done rebuilding cl_bz_spread & momentum_24h.")
