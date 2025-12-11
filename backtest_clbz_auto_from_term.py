import pandas as pd
import numpy as np
from pathlib import Path
import pickle

ROOT = Path(".").resolve()

print("Scanning for model and features under:", ROOT)

# 1) ??? base_seed202_clbz_h1.pkl??? repo ????
model_candidates = list(ROOT.rglob("base_seed202_clbz_h1.pkl"))
if not model_candidates:
    raise FileNotFoundError("??????????? base_seed202_clbz_h1.pkl")

MODEL_PATH = model_candidates[0]
print("Using model:", MODEL_PATH)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
feature_cols = list(model.feature_names_in_)
print("Model feature count:", len(feature_cols))

# 2) ? features_hourly_with_term.parquet??? repo ????
feat_candidates = list(ROOT.rglob("features_hourly_with_term.parquet"))
if not feat_candidates:
    raise FileNotFoundError("??????????? features_hourly_with_term.parquet")

FEAT_PATH = feat_candidates[0]
print("Using features file:", FEAT_PATH)

df = pd.read_parquet(FEAT_PATH)

# 3) ????????????? index??????
ts_series = None

# 3-1 ????
for c in ["ts_utc", "ts", "timestamp", "time", "datetime"]:
    if c in df.columns:
        ts_series = pd.to_datetime(df[c], utc=True, errors="coerce")
        print(f"Using time column: {c}")
        break

# 3-2 ??????? DatetimeIndex
if ts_series is None and isinstance(df.index, pd.DatetimeIndex):
    ts_series = pd.to_datetime(df.index, utc=True, errors="coerce")
    print("Using DatetimeIndex as time column")

# 3-3 ????????????? ts/time/date?
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
    print("?????", list(df.columns))
    raise RuntimeError("?????????? ts_utc/ts/timestamp/time/datetime???? DatetimeIndex????? ts/time/date ???")

df["ts"] = ts_series
df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

# ??? 2024-10 ~ 2025-12
DATE_FROM = pd.Timestamp("2024-10-01", tz="UTC")
DATE_TO   = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
df = df[(df["ts"] >= DATE_FROM) & (df["ts"] <= DATE_TO)].copy()
print("Rows in range 2024-10~2025-12:", len(df))

if len(df) == 0:
    raise RuntimeError("??????????2024-10~2025-12?")

# 4) ??????????
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    print("WARNING: Missing feature cols, will fill 0.0:", missing)
    for c in missing:
        df[c] = 0.0

X = df[feature_cols].fillna(0.0)

print("Generating predictions...")
df["prediction"] = model.predict(X)

# 5) ????????? ret_1h / wti_returns???????
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
        print("?????", list(df.columns))
        raise RuntimeError("??? ret_1h / wti_returns?????????(close_mid ?)????????")

    print(f"ret_1h ???????????? '{price_col}' ?? 1h ???")
    df = df.sort_values("ts")
    df["act_ret"] = df[price_col].pct_change().shift(-1)
else:
    print(f"Using return column: {ret_col}")
    df["act_ret"] = df[ret_col]

df = df.dropna(subset=["act_ret", "prediction"]).reset_index(drop=True)

# 6) ?? & ???
df["pred_dir"] = np.where(df["prediction"] > 0, "UP", "DN")
df["act_dir"]  = np.where(df["act_ret"] > 0, "UP", "DN")
df["correct"]  = df["pred_dir"] == df["act_dir"]

total   = len(df)
correct = int(df["correct"].sum())
acc     = df["correct"].mean() * 100.0
pred_up = int((df["pred_dir"] == "UP").sum())
pred_dn = int((df["pred_dir"] == "DN").sum())
act_up  = int((df["act_dir"]  == "UP").sum())
act_dn  = int((df["act_dir"]  == "DN").sum())

# 7) ?? IC
ic_pearson  = df["prediction"].corr(df["act_ret"])
ic_spearman = df["prediction"].corr(df["act_ret"], method="spearman")

# 8) ?? IC / IR
df["date"] = df["ts"].dt.floor("D")
daily_ic = (
    df.groupby("date")[["prediction", "act_ret"]]
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

# 9) ????
df["ym"] = df["ts"].dt.to_period("M").astype(str)
def group_ic(g):
    return g["prediction"].corr(g["act_ret"])

monthly = (
    df.groupby("ym")
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
print("=" * 80)
print("CLBZ FULL BACKTEST RESULT (2024-10 ~ 2025-12, auto-found features_hourly_with_term)")
print("=" * 80)
print(f"Features path : {FEAT_PATH}")
print(f"Range         : {df['ts'].min()}  ->  {df['ts'].max()}")
print(f"Total samples : {total}")
print(f"Correct       : {correct}")
print(f"Accuracy      : {acc:.2f}%")
print("")
print("Prediction vs Actual direction counts:")
print(f"  Pred UP : {pred_up:5d}")
print(f"  Pred DN : {pred_dn:5d}")
print(f"  Act  UP : {act_up:5d}")
print(f"  Act  DN : {act_dn:5d}")
print("")
print("Information Coefficient (all hours):")
print(f"  IC (Pearson)  : {ic_pearson:.4f}")
print(f"  IC (Spearman) : {ic_spearman:.4f}")
print("")
print("Daily IC & IR (by calendar date):")
print(f"  Days with IC  : {len(daily_ic)}")
print(f"  Daily IC mean : {ic_mean:.4f}")
print(f"  Daily IC std  : {ic_std:.4f}")
print(f"  IR (mean/std) : {ir:.4f}")
print("")
print("Monthly summary (ym, n, acc%, ic):")
print(monthly.to_string(float_format=lambda x: f"{x:6.2f}"))
print("")
print("Last 20 rows preview:")
print(
    df[[
        "ts",
        "prediction",
        "act_ret",
        "pred_dir",
        "act_dir",
        "correct",
    ]]
    .tail(20)
    .to_string(index=False)
)
print("")
print("Done.")
