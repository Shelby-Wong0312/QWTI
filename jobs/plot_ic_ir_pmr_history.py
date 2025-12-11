#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot historical IC / IR / PMR time series from monitoring metrics.

Inputs:
- warehouse/monitoring/base_seed202_lean7_metrics.csv
  - must contain timestamp column (candidates: ts, timestamp, time, as_of, as_of_utc)
  - must contain raw IC column (candidates: ic, IC)
  - if rolling columns are missing, they will be computed (15d/30d/60d, hourly indexed)

Outputs:
- warehouse/monitoring/ic_ir_pmr_history.png
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONITOR_DIR = PROJECT_ROOT / "warehouse" / "monitoring"
METRICS_CSV = MONITOR_DIR / "base_seed202_lean7_metrics.csv"
PLOT_PATH = MONITOR_DIR / "ic_ir_pmr_history.png"


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


def compute_rollings(df: pd.DataFrame, ic_col: str) -> pd.DataFrame:
    df = df.copy()
    for days in (15, 30, 60):
        hours = days * 24
        roll = df[ic_col].rolling(f"{hours}H", min_periods=1)
        ic_mean = roll.mean()
        ic_std = roll.std(ddof=0)
        pmr = roll.apply(lambda x: float((x > 0).mean()) if len(x) else np.nan, raw=False)
        ir = ic_mean / ic_std.replace(0, np.nan)
        df[f"IC_{days}D"] = ic_mean
        df[f"IR_{days}D"] = ir
        df[f"PMR_{days}D"] = pmr
    return df


def plot_series(df: pd.DataFrame) -> None:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    # IC (rolling means)
    for days, color in [(15, "orange"), (30, "deepskyblue"), (60, "violet")]:
        col = f"IC_{days}D"
        if col in df.columns:
            axes[0].plot(df.index, df[col], label=f"IC {days}D", color=color, alpha=0.9)
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_ylabel("IC")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.2)

    # IR
    for days, color in [(15, "orange"), (30, "deepskyblue"), (60, "violet")]:
        col = f"IR_{days}D"
        if col in df.columns:
            axes[1].plot(df.index, df[col], label=f"IR {days}D", color=color, alpha=0.9)
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_ylabel("IR")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.2)

    # PMR
    for days, color in [(15, "orange"), (30, "deepskyblue"), (60, "violet")]:
        col = f"PMR_{days}D"
        if col in df.columns:
            axes[2].plot(df.index, df[col], label=f"PMR {days}D", color=color, alpha=0.9)
    axes[2].axhline(0.5, color="gray", lw=0.8, ls="--")
    axes[2].set_ylabel("PMR")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.2)

    fig.autofmt_xdate()
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    logging.info("Saved plot: %s", PLOT_PATH)


def main():
    ap = argparse.ArgumentParser(description="Plot historical IC/IR/PMR from monitoring metrics.")
    ap.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    if not METRICS_CSV.exists():
        logging.error("Metrics CSV not found: %s", METRICS_CSV)
        return

    df = pd.read_csv(METRICS_CSV)
    if df.empty:
        logging.error("Metrics CSV is empty: %s", METRICS_CSV)
        return

    ts_col = detect_ts_column(df)
    ic_col = detect_ic_column(df)
    if ts_col is None or ic_col is None:
        logging.error("Missing timestamp or IC column in metrics CSV.")
        return

    df["_ts_utc"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=["_ts_utc"]).sort_values("_ts_utc").set_index("_ts_utc")
    df = compute_rollings(df, ic_col=ic_col)

    plot_series(df)


if __name__ == "__main__":
    main()
