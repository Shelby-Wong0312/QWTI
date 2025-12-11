import pandas as pd
import numpy as np
from pathlib import Path
import pickle

ROOT = Path(".").resolve()

print("Scanning for model and features under:", ROOT)

# ========== ?????? ==========

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

# 3) ??????
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

# 4) ??????2024-10 ~ 2025-12
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

# 6) ????
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

# 7) ??????? / abs_bucket / hour / dow
df["pred_dir"] = np.where(df["prediction"] > 0, "UP", "DN")
df["act_dir"]  = np.where(df["act_ret"] > 0, "UP", "DN")
df["correct"]  = df["pred_dir"] == df["act_dir"]
df["abs_pred"] = df["prediction"].abs()

# abs_bucket: 0=??, 4=??
N_BUCKETS = 5
ranks = df["abs_pred"].rank(method="average", pct=True)
df["abs_bucket"] = pd.qcut(ranks, N_BUCKETS, labels=False, duplicates="drop").astype(int)

df["hour"] = df["ts"].dt.hour
df["dow"]  = df["ts"].dt.dayofweek  # 0=Mon ... 6=Sun
dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
df["dow_name"] = df["dow"].map(dow_map)

print("")
print(f"Total samples after cleaning: {len(df)}")
overall_acc = df["correct"].mean() * 100
overall_ic  = df["prediction"].corr(df["act_ret"])
print(f"Overall acc : {overall_acc:.2f}%")
print(f"Overall IC  : {overall_ic:.4f}")

# ========== ???? ==========

def compute_stats(name: str, sub: pd.DataFrame) -> None:
    print("")
    print("=" * 80)
    print(f"FILTER: {name}")
    print("=" * 80)

    n = len(sub)
    print(f"Samples: {n}")
    if n == 0:
        print("No data under this filter.")
        return
    if n < 30:
        print("WARNING: samples < 30, ??????????")

    acc = sub["correct"].mean() * 100.0
    ic_pearson  = sub["prediction"].corr(sub["act_ret"])
    ic_spearman = sub["prediction"].corr(sub["act_ret"], method="spearman")

    pred_up = int((sub["pred_dir"] == "UP").sum())
    pred_dn = int((sub["pred_dir"] == "DN").sum())
    act_up  = int((sub["act_dir"]  == "UP").sum())
    act_dn  = int((sub["act_dir"]  == "DN").sum())

    print(f"Accuracy      : {acc:.2f}%")
    print(f"IC (Pearson)  : {ic_pearson:.4f}")
    print(f"IC (Spearman) : {ic_spearman:.4f}")
    print("")
    print("Direction counts:")
    print(f"  Pred UP : {pred_up:5d}")
    print(f"  Pred DN : {pred_dn:5d}")
    print(f"  Act  UP : {act_up:5d}")
    print(f"  Act  DN : {act_dn:5d}")

    # ?? IC / IR
    sub = sub.copy()
    sub["date"] = sub["ts"].dt.floor("D")
    daily_ic = (
        sub.groupby("date")[["prediction", "act_ret"]]
        .apply(lambda g: g["prediction"].corr(g["act_ret"]))
        .dropna()
    )
    if len(daily_ic) > 1:
        ic_mean = daily_ic.mean()
        ic_std  = daily_ic.std(ddof=0)
        ir      = ic_mean / ic_std if ic_std > 0 else np.nan
    else:
        ic_mean = np.nan
        ic_std  = np.nan
        ir      = np.nan

    print("")
    print("Daily IC & IR (by calendar date):")
    print(f"  Days with IC  : {len(daily_ic)}")
    print(f"  Daily IC mean : {ic_mean:.4f}")
    print(f"  Daily IC std  : {ic_std:.4f}")
    print(f"  IR (mean/std) : {ir:.4f}")

    # ????
    sub["ym"] = sub["ts"].dt.to_period("M").astype(str)
    def group_ic(g):
        return g["prediction"].corr(g["act_ret"])

    monthly = (
        sub.groupby("ym")
        .apply(
            lambda g: pd.Series(
                {
                    "n":   len(g),
                    "acc": g["correct"].mean() * 100.0,
                    "ic":  group_ic(g),
                }
            )
        )
        .sort_index()
    )

    print("")
    print("Monthly summary (ym, n, acc%, ic):")
    print(monthly.to_string(float_format=lambda x: f"{x:6.2f}"))

# ========== ?? filter ==========

filters = []

# baseline????
filters.append(("ALL", np.ones(len(df), dtype=bool)))

# 1) abs_bucket >= 3
filters.append((
    "abs_bucket >= 3",
    df["abs_bucket"] >= 3
))

# 2) abs_bucket >= 3 & TueThu (dow 1,2,3)
filters.append((
    "abs_bucket >= 3 & TueThu",
    (df["abs_bucket"] >= 3) & (df["dow"].isin([1, 2, 3]))
))

# 3) abs_bucket >= 3 & hours in [3,9,13,22] (UTC)
strong_hours = [3, 9, 13, 22]
filters.append((
    f"abs_bucket >= 3 & hour in {strong_hours} (UTC)",
    (df["abs_bucket"] >= 3) & (df["hour"].isin(strong_hours))
))

# 4) abs_bucket >= 3 & TueThu & hours in [3,9,13,22] (UTC)
filters.append((
    f"abs_bucket >= 3 & TueThu & hour in {strong_hours} (UTC)",
    (df["abs_bucket"] >= 3) &
    (df["dow"].isin([1, 2, 3])) &
    (df["hour"].isin(strong_hours))
))

# ========== ??? filter ==========

for name, mask in filters:
    sub = df[mask].copy()
    compute_stats(name, sub)

print("")
print("Done filtered backtest.")
