#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


TS_CANDIDATES = ("ts", "time", "timestamp", "datetime")
CLOSE_CANDIDATES = ("close", "price", "settle", "settlement")

ROOT_DIR = Path(__file__).resolve().parents[1]
PRICE_DIR = ROOT_DIR / "capital_wti_downloader" / "hourly"
TERM_PATH = ROOT_DIR / "data" / "term_crack_ovx_hourly.csv"
FEATURES_PATH = ROOT_DIR / "data" / "features_hourly.parquet"

SIGNALS_PATH = ROOT_DIR / "warehouse" / "signals_hourly_exp3.csv"
SUMMARY_PATH = ROOT_DIR / "warehouse" / "monitor_signals_summary.csv"


def _detect_column(columns: Iterable[str], candidates: Iterable[str], label: str) -> str:
    lookup = {col.strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Unable to detect {label} column among {list(columns)}")


def _latest_price_path() -> Path:
    files = sorted(PRICE_DIR.glob("OIL_CRUDE_HOUR_*_clean.parquet"))
    if not files:
        raise FileNotFoundError(f"No clean price files found in {PRICE_DIR}")
    return files[-1]


def _normalize_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.floor("H")


def load_price() -> pd.DataFrame:
    price_path = _latest_price_path()
    df = pd.read_parquet(price_path)
    ts_col = _detect_column(df.columns, TS_CANDIDATES, "timestamp")
    close_col = _detect_column(df.columns, CLOSE_CANDIDATES, "close")
    price = pd.DataFrame(
        {
            "ts": _normalize_ts(df[ts_col]),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    )
    price = price.dropna(subset=["ts", "close"])
    price = price.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)
    return price


def load_term() -> pd.DataFrame:
    if not TERM_PATH.exists():
        raise FileNotFoundError(f"Term proxy file not found: {TERM_PATH}")
    df = pd.read_csv(TERM_PATH)
    ts_col = _detect_column(df.columns, TS_CANDIDATES, "timestamp")
    term = pd.DataFrame(
        {
            "ts": _normalize_ts(df[ts_col]),
            "cl1_cl2": pd.to_numeric(df["cl1_cl2"], errors="coerce"),
        }
    )
    term = term.dropna(subset=["ts", "cl1_cl2"])
    term = term.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)
    return term


def load_features() -> Tuple[pd.DataFrame, bool]:
    if not FEATURES_PATH.exists():
        price = load_price()
        term = load_term()
        merged = price.merge(term, on="ts", how="inner")
        return merged, False

    df = pd.read_parquet(FEATURES_PATH)
    if "ts" not in df.columns:
        raise ValueError(f"'ts' column missing in features file: {FEATURES_PATH}")

    close_col = _detect_column(df.columns, CLOSE_CANDIDATES, "close")
    base_cols = ["ts", close_col, "cl1_cl2"]
    news_cols = [col for col in df.columns if col.startswith("news_")]
    subset = df[base_cols + news_cols].copy()
    subset = subset.rename(columns={close_col: "close"})
    subset["close"] = pd.to_numeric(subset["close"], errors="coerce")
    subset["cl1_cl2"] = pd.to_numeric(subset["cl1_cl2"], errors="coerce")
    for col in news_cols:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
    subset["ts"] = _normalize_ts(subset["ts"])
    subset = subset.dropna(subset=["ts", "close", "cl1_cl2"])
    subset = subset.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)

    news_available = "news_tone_avg" in subset.columns
    return subset, news_available


def compute_signals(base: pd.DataFrame, news_available: bool) -> Tuple[pd.DataFrame, str]:
    df = base.sort_values("ts").reset_index(drop=True).copy()
    df["ret_1h"] = df["close"].pct_change().fillna(0.0)

    term_mean_24 = df["cl1_cl2"].rolling(window=24, min_periods=24).mean()
    term_std_24 = df["cl1_cl2"].rolling(window=24, min_periods=24).std(ddof=0).clip(lower=1e-6)
    df["term_z"] = ((df["cl1_cl2"] - term_mean_24) / term_std_24).fillna(0.0)

    vol_24 = df["ret_1h"].rolling(window=24, min_periods=24).std(ddof=0).clip(lower=1e-6).fillna(1e-6)
    df["vol_24"] = vol_24
    ret_avg_6 = df["ret_1h"].rolling(window=6, min_periods=6).mean().fillna(0.0)

    w_term = np.tanh(0.75 * df["term_z"])
    w_mom = np.tanh(3.0 * ret_avg_6 / vol_24)

    if news_available and "news_tone_avg" in df.columns:
        news_mean_24 = df["news_tone_avg"].rolling(window=24, min_periods=24).mean()
        news_std_24 = (
            df["news_tone_avg"].rolling(window=24, min_periods=24).std(ddof=0).clip(lower=1e-6)
        )
        df["news_z"] = ((df["news_tone_avg"] - news_mean_24) / news_std_24).fillna(0.0)
        w_news = np.tanh(0.8 * df["news_z"])
        note = "with_news"
    else:
        df["news_z"] = 0.0
        w_news = pd.Series(0.0, index=df.index)
        note = "no_news"

    w_t = 0.5 * w_term + 0.35 * w_mom + 0.15 * w_news
    w_t = np.clip(w_t, -1.0, 1.0)

    lookback_hours = 24 * 90
    vol_q90 = vol_24.rolling(window=lookback_hours, min_periods=24).quantile(0.9)
    risk_mask = (vol_24 > vol_q90) & vol_q90.notna()
    w_t = w_t.where(~risk_mask, w_t * 0.6)

    signals = pd.DataFrame(
        {
            "ts": df["ts"],
            "w_t": w_t,
            "w_term": w_term,
            "w_mom": w_mom,
            "w_news": w_news,
            "ret_1h": df["ret_1h"],
            "term_z": df["term_z"],
            "news_z": df["news_z"],
            "vol_24": vol_24,
        }
    )
    return signals, note


def compute_summaries(signals: pd.DataFrame, note: str) -> pd.DataFrame:
    windows = (7, 30, 90)
    rows = []
    for days in windows:
        window_hours = days * 24
        subset = signals.tail(window_hours)
        count = len(subset)
        if count == 0:
            zero_ratio = np.nan
            turnover = np.nan
        else:
            zero_ratio = float((subset["w_t"] == 0).mean())
            turnover = float(subset["w_t"].diff().abs().sum()) / days
        rows.append(
            {
                "window": f"{days}d",
                "rows": count,
                "zero_ratio": zero_ratio,
                "turnover": turnover,
                "note": note,
            }
        )
    return pd.DataFrame(rows, columns=["window", "rows", "zero_ratio", "turnover", "note"])


def main() -> None:
    base, news_available = load_features()
    signals, note = compute_signals(base, news_available)

    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    out = signals.copy()
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(SIGNALS_PATH, index=False)

    summary = compute_summaries(signals, note)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"[INFO] Wrote signals: {SIGNALS_PATH} rows={len(out)} note={note}")
    print(f"[INFO] Wrote summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
