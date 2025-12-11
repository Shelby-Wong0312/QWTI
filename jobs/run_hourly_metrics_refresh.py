#!/usr/bin/env python3
"""
Compute rolling IC/IR/PMR metrics from hourly metrics/runlog and write to monitoring metrics file.

Inputs:
- warehouse/monitoring/base_seed202_lean7_metrics.csv  (must contain timestamp + ic column)
  Expected time column candidates: ts, timestamp, time, as_of, as_of_utc
  Expected IC column candidates: ic, IC

Outputs:
- warehouse/monitoring/base_seed202_lean7_metrics.csv  (overwritten with added rolling columns)
- warehouse/monitoring/hourly_metrics.parquet          (same content in parquet)

Rolling windows (time-based, hourly freq):
- 15d -> IC_15D, IR_15D, PMR_15D
- 30d -> IC_30D, IR_30D, PMR_30D
- 60d -> IC_60D, IR_60D, PMR_60D

Compatible with send_hourly_email.py / send_daily_email.py which read IC_15D/IR_15D/PMR_15D.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONITOR_DIR = PROJECT_ROOT / "warehouse" / "monitoring"
METRICS_CSV = MONITOR_DIR / "base_seed202_lean7_metrics.csv"
METRICS_PARQUET = MONITOR_DIR / "hourly_metrics.parquet"


def detect_ts_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("ts", "timestamp", "time", "as_of", "as_of_utc"):
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def detect_ic_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("ic", "IC"):
        if c in df.columns:
            return c
    return None


def compute_rolling(df: pd.DataFrame, ic_col: str, days: int) -> pd.DataFrame:
    hours = days * 24
    # time-based rolling over hourly index
    roll = df[ic_col].rolling(f"{hours}H", min_periods=1)
    ic_mean = roll.mean()
    ic_std = roll.std(ddof=0)
    pmr = roll.apply(lambda x: float((x > 0).mean()) if len(x) else np.nan, raw=False)
    ir = ic_mean / ic_std.replace(0, np.nan)
    df[f"IC_{days}D"] = ic_mean
    df[f"IR_{days}D"] = ir
    df[f"PMR_{days}D"] = pmr
    return df


def refresh_metrics(log_level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    if not METRICS_CSV.exists():
        logging.error("Metrics CSV not found: %s", METRICS_CSV)
        return

    df = pd.read_csv(METRICS_CSV)
    if df.empty:
        logging.error("Metrics CSV is empty: %s", METRICS_CSV)
        return

    ts_col = detect_ts_column(df)
    ic_col = detect_ic_column(df)
    if ts_col is None:
        logging.error("No timestamp column found in metrics CSV.")
        return
    if ic_col is None:
        logging.error("No IC column found in metrics CSV.")
        return

    df["_ts_utc"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=["_ts_utc"]).sort_values("_ts_utc")
    df = df.set_index("_ts_utc")

    for days in (15, 30, 60):
        df = compute_rolling(df, ic_col=ic_col, days=days)

    df = df.reset_index().rename(columns={"_ts_utc": ts_col})

    # Ensure send_hourly_email / send_daily_email pick up IC/IR/PMR 15D columns
    needed_cols: List[str] = [f"{p}_{d}D" for d in (15, 30, 60) for p in ("IC", "IR", "PMR")]
    for c in needed_cols:
        if c not in df.columns:
            df[c] = np.nan

    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = METRICS_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp_csv, index=False)
    tmp_csv.replace(METRICS_CSV)
    logging.info("Updated metrics CSV with rolling IC/IR/PMR: %s (rows=%d)", METRICS_CSV, len(df))

    tmp_parq = METRICS_PARQUET.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_parq, index=False)
    tmp_parq.replace(METRICS_PARQUET)
    logging.info("Wrote parquet snapshot: %s", METRICS_PARQUET)


def main():
    ap = argparse.ArgumentParser(description="Refresh rolling IC/IR/PMR metrics for monitoring emails.")
    ap.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    args = ap.parse_args()
    refresh_metrics(log_level=args.log_level)


if __name__ == "__main__":
    main()
