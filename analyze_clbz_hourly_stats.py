import pandas as pd
import numpy as np
from pathlib import Path
import pickle

ROOT = Path(".").resolve()

print("Scanning for model and features under:", ROOT)

# 1) ???
model_candidates = list(ROOT.rglob("base_seed202_clbz_h1.pkl"))
if not model_candidates:
    raise FileNotFoundError("??? base_seed202_clbz_h1.pkl")
MODEL_PATH = model_candidates[0]
print("Using model:", MODEL_PATH)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
feature_cols = list(model.feature_names_in_)
print("Model feature count:", len(feature_cols))

# 2) ? features_hourly_with_term.parquet
feat_candidates = list(ROOT.rglob("features_hourly_with_term.parquet"))
if not feat_candidates:
    raise FileNotFoundError("??? features_hourly_with_term.parquet")
FEAT_PATH = feat_candidates[0]
print("Using features file:", FEAT_PATH)

df = pd.read_parquet(FEAT_PATH)

# 3) ??????????????
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

# 4) ??????2024-10 ~ 2025-12????????
DATE_FROM = pd.Timestamp("2024-10-01", tz="UTC")
DATE_TO   = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
df = df[(df["ts"] >= DATE_FROM) & (df["ts"] <= DATE_TO)].copy()
print("Rows in range 2024-10~2025-12:", len(df))

if len(df) == 0:
    raise RuntimeError("?????????")

# 5) ????
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    print("WARNING: Missing feature cols, fill 0.0:", missing)
    for c in missing:
        df[c] = 0.0

X = df[feature_cols].fillna(0.0)

print("Generating predictions...")
df["prediction"] = model.predict(X)

# 6) ??????
ret_col = None
for c in ["ret_1h", "wti_returns", "returns_1h"]:
    if c in df.columns:
        ret_col = c
        break

if ret_col is None:
    price_col = None
    for c in ["close_mid", "wti_close", "close", "mid", "price"]:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        print("Columns:", list(df.columns))
        raise RuntimeError("?? ret_1h / wti_returns???????????????")

    print(f"Using price column '{price_col}' to compute 1h returns")
    df = df.sort_values("ts")
    df["act_ret"] = df[price_col].pct_change().shift(-1)
else:
    print(f"Using return column: {ret_col}")
    df["act_ret"] = df[ret_col]

df = df.dropna(subset=["act_ret", "prediction"]).reset_index(drop=True)

# 7) ?? & ???
df["pred_dir"] = np.where(df["prediction"] > 0, "UP", "DN")
df["act_dir"]  = np.where(df["act_ret"] > 0, "UP", "DN")
df["correct"]  = df["pred_dir"] == df["act_dir"]

print("")
print(f"Total samples after cleaning: {len(df)}")

# === A. ??????? ===
df["hour"] = df["ts"].dt.hour

def ic(g):
    return g["prediction"].corr(g["act_ret"])

hourly = (
    df.groupby("hour")
    .apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "acc": g["correct"].mean() * 100.0,
                "ic": ic(g),
            }
        )
    )
    .sort_index()
)

print("")
print("=" * 80)
print("Accuracy / IC by HOUR (UTC, 0-23)")
print("=" * 80)
print(hourly.to_string(float_format=lambda x: f"{x:6.2f}"))

print("")
print("Top 5 hours by accuracy:")
print(
    hourly.sort_values("acc", ascending=False)
    .head(5)
    .to_string(float_format=lambda x: f"{x:6.2f}")
)

print("")
print("Bottom 5 hours by accuracy:")
print(
    hourly.sort_values("acc", ascending=True)
    .head(5)
    .to_string(float_format=lambda x: f"{x:6.2f}")
)

# === B. ???????? ===
df["dow"] = df["ts"].dt.dayofweek  # 0=Mon ... 6=Sun
dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

dow_stats = (
    df.groupby("dow")
    .apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "acc": g["correct"].mean() * 100.0,
                "ic": ic(g),
            }
        )
    )
    .sort_index()
)
dow_stats.index = dow_stats.index.map(dow_map)

print("")
print("=" * 80)
print("Accuracy / IC by DAY OF WEEK")
print("=" * 80)
print(dow_stats.to_string(float_format=lambda x: f"{x:6.2f}"))

print("")
print("Done hourly / DOW analysis.")
