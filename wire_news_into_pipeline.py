#!/usr/bin/env python3
"""
Integrate hourly GDELT features with price and term data, producing a unified
feature set for the online experiment pipeline.
"""
from __future__ import annotations

# --- No-Drift preflight ---
from pathlib import Path
import sys
sys.path.insert(0, r"C:\Users\niuji\Documents\Data\warehouse\policy\utils")
from nodrift_preflight import enforce
# --- End preflight ---

from typing import Iterable

import numpy as np
import pandas as pd


PRICE_DIR = Path("capital_wti_downloader/hourly")
TERM_PATH = Path("data/term_crack_ovx_hourly.csv")
NEWS_PATH = Path("data/gdelt_hourly.parquet")
OUTPUT_PATH = Path("data/features_hourly.parquet")

TS_CANDIDATES = ("ts", "time", "timestamp", "datetime")


def _detect_column(columns: Iterable[str], candidates: Iterable[str], label: str) -> str:
    lookup = {col.strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Unable to locate {label} column in {list(columns)}")


def _latest_price_file() -> Path:
    if not PRICE_DIR.exists():
        raise FileNotFoundError(f"Price directory not found: {PRICE_DIR}")
    files = sorted(PRICE_DIR.glob("OIL_CRUDE_HOUR_*_clean.parquet"))
    if not files:
        raise FileNotFoundError(f"No *_clean.parquet price files in {PRICE_DIR}")
    return files[-1]


def _normalize_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    ts = ts.dropna()
    return ts.dt.floor("H")


def load_price() -> pd.DataFrame:
    price_path = _latest_price_file()
    df = pd.read_parquet(price_path)
    ts_col = _detect_column(df.columns, TS_CANDIDATES, "timestamp")
    price = df.copy()
    price["ts"] = pd.to_datetime(price[ts_col], utc=True, errors="coerce").dt.floor("H")
    price = price.dropna(subset=["ts"])
    price = price.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)
    return price


def load_term() -> pd.DataFrame:
    if not TERM_PATH.exists():
        raise FileNotFoundError(f"Term proxy data not found: {TERM_PATH}")
    df = pd.read_csv(TERM_PATH)
    ts_col = _detect_column(df.columns, TS_CANDIDATES, "timestamp")
    term = df.copy()
    term["ts"] = pd.to_datetime(term[ts_col], utc=True, errors="coerce").dt.floor("H")
    term = term.dropna(subset=["ts"])
    term = term.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)
    return term


def load_news() -> pd.DataFrame:
    if not NEWS_PATH.exists():
        print(f"[INFO] News parquet not found at {NEWS_PATH}; skipping integration.")
        return pd.DataFrame()
    df = pd.read_parquet(NEWS_PATH)
    ts_col = _detect_column(df.columns, TS_CANDIDATES, "timestamp")
    news = df.copy()
    news["ts"] = pd.to_datetime(news[ts_col], utc=True, errors="coerce").dt.floor("H")
    news = news.dropna(subset=["ts"])
    news = news.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)
    rename_map = {
        col: f"news_{col}"
        for col in news.columns
        if col != "ts"
    }
    news = news.rename(columns=rename_map)
    return news


def compute_alignment(features: pd.DataFrame, price_rows: int) -> None:
    if price_rows == 0:
        print("[WARN] Price dataset empty; cannot compute alignment ratio.")
        return
    ratio = len(features) / price_rows if price_rows else float("nan")
    print(f"[INFO] Alignment ratio (features vs. price): {len(features)}/{price_rows} = {ratio:.2%}")


def report_news_coverage(features: pd.DataFrame) -> None:
    news_cols = [col for col in features.columns if col.startswith("news_")]
    if not news_cols:
        print("[INFO] No news columns present after merge.")
        return
    coverage = {col: float(features[col].notna().mean()) for col in news_cols}
    coverage_str = ", ".join(f"{col}: {pct:.2%}" for col, pct in coverage.items())
    print(f"[INFO] News column non-null ratios -> {coverage_str}")


def enforce_no_drift(coverage_snapshot: pd.DataFrame, price_rows: int) -> None:
    """Run the policy preflight using observed coverage metrics."""
    mapped_ratio = float(len(coverage_snapshot) / price_rows) if price_rows else 0.0
    art_col = "news_art_cnt"
    if art_col in coverage_snapshot.columns and not coverage_snapshot.empty:
        all_art_cnt = float(coverage_snapshot[art_col].median())
    else:
        all_art_cnt = 0.0
    tone_col = "news_tone_avg"
    tone_nonnull = bool(
        tone_col in coverage_snapshot.columns
        and not coverage_snapshot.empty
        and coverage_snapshot[tone_col].notna().all()
    )
    observed = dict(
        mode="hard_kpi",
        mapped_ratio=mapped_ratio,
        all_art_cnt=all_art_cnt,
        tone_nonnull=tone_nonnull,
        skip_ratio=0.0,
    )
    enforce(observed)


def main() -> None:
    price = load_price()
    price_rows = len(price)
    term = load_term()
    news = load_news()
    if news.empty:
        print("[INFO] No news data available; exiting without creating features.")
        return

    merged = price.merge(term, on="ts", how="inner", suffixes=("", "_term"))
    merged = merged.merge(news, on="ts", how="inner")
    merged = merged.sort_values("ts").reset_index(drop=True)

    coverage_snapshot = merged.copy()
    enforce_no_drift(coverage_snapshot, price_rows)

    for col in ("news_tone_avg", "news_tone_pos_ratio"):
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)

    compute_alignment(merged, price_rows)
    report_news_coverage(coverage_snapshot)

    print(f"[INFO] Wrote features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
