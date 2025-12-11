import pandas as pd
import numpy as np
from pathlib import Path
import pickle

ROOT = Path(".").resolve()

# ???
MODEL_CANDIDATES = [
    ROOT / "models" / "base_seed202_clbz_h1.pkl",
    ROOT / "wti" / "models" / "base_seed202_clbz_h1.pkl",
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)
if MODEL_PATH is None:
    raise FileNotFoundError("??? base_seed202_clbz_h1.pkl?????? models/ ? wti/models/ ???")

print("Using model:", MODEL_PATH)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
feature_cols = list(model.feature_names_in_)
print("Model feature count:", len(feature_cols))

# ? GDELT
GDELT_CANDIDATES = [
    ROOT / "data" / "gdelt_hourly.parquet",
    ROOT / "wti" / "data" / "gdelt_hourly.parquet",
]
GDELT_PATH = next((p for p in GDELT_CANDIDATES if p.exists()), None)
if GDELT_PATH is None:
    raise FileNotFoundError("??? gdelt_hourly.parquet????? data/ ? wti/data/ ???")

print("Using GDELT:", GDELT_PATH)
gd = pd.read_parquet(GDELT_PATH)

ts_col_gd = None
for c in ["ts_utc", "ts", "timestamp"]:
    if c in gd.columns:
        ts_col_gd = c
        break
if ts_col_gd is None:
    raise RuntimeError("GDELT ????????(ts_utc/ts/timestamp)")

gd["ts"] = pd.to_datetime(gd[ts_col_gd], utc=True, errors="coerce")
gd = gd.dropna(subset=["ts"])

DATE_FROM = pd.Timestamp("2024-10-01", tz="UTC")
DATE_TO   = pd.Timestamp("2025-12-10 23:59:59", tz="UTC")
gd = gd[(gd["ts"] >= DATE_FROM) & (gd["ts"] <= DATE_TO)].copy()
print("GDELT rows in range:", len(gd))

# ???
PRICE_CANDIDATES = [
    ROOT / "data" / "wti_hourly_capital.parquet",
    ROOT / "wti" / "data" / "wti_hourly_capital.parquet",
]
PRICE_PATH = next((p for p in PRICE_CANDIDATES if p.exists()), None)
if PRICE_PATH is None:
    raise FileNotFoundError("??? wti_hourly_capital.parquet????? data/ ? wti/data/ ???")

print("Using price:", PRICE_PATH)
px = pd.read_parquet(PRICE_PATH)

ts_col_px = None
for c in ["ts_utc", "ts", "timestamp"]:
    if c in px.columns:
        ts_col_px = c
        break
if ts_col_px is None:
    raise RuntimeError("WTI ??????????(ts_utc/ts/timestamp)")

px["ts"] = pd.to_datetime(px[ts_col_px], utc=True, errors="coerce")

price_col = None
for c in ["close_mid", "wti_close", "close", "mid", "price"]:
    if c in px.columns:
        price_col = c
        break
if price_col is None:
    raise RuntimeError("WTI ??????????(close_mid/wti_close/close/mid/price)")

px = px.dropna(subset=["ts"]).sort_values("ts")
px["ret_1h"] = px[price_col].pct_change().shift(-1)
px = px[(px["ts"] >= DATE_FROM) & (px["ts"] <= DATE_TO)].copy()
print("Price rows in range:", len(px))

# merge
merged = gd.merge(px[["ts", price_col, "ret_1h"]], on="ts", how="inner")
print("Merged rows:", len(merged))

# ts ?? naive UTC
merged["ts"] = merged["ts"].dt.tz_convert(None)

# ??????????
missing = [c for c in feature_cols if c not in merged.columns]
if missing:
    print("Missing feature cols, will fill 0.0:", missing)
    for c in missing:
        merged[c] = 0.0

cols = ["ts", "ret_1h"] + feature_cols
merged = merged[cols].sort_values("ts").reset_index(drop=True)

OUT_PATH = ROOT / "data" / "features_clbz_202410_202512.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
print("Saving features to:", OUT_PATH)
merged.to_parquet(OUT_PATH)
print("Done. Rows:", len(merged))
