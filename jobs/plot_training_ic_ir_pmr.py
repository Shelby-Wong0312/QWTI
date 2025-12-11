#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot training-period IC/IR/PMR time series from a predictions+labels dataset.

Expected input (CSV or Parquet):
- A time column (default: ts) parsable to UTC
- A prediction column (default: prediction)
- A label/actual return column (default: label)

Rolling metrics (time-based windows):
- 15d, 30d, 60d IC: rolling Pearson corr(prediction, label)
- IR: rolling mean(IC) / std(IC)
- PMR: rolling proportion of correct direction (sign match)

Example:
  python jobs/plot_training_ic_ir_pmr.py \
    --data data/training_predictions.parquet \
    --ts-col ts --pred-col prediction --label-col label \
    --out warehouse/monitoring/training_ic_ir_pmr.png
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_OUT = Path("warehouse/monitoring/training_ic_ir_pmr.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training IC/IR/PMR from predictions + labels.")
    p.add_argument("--data", required=True, help="Input CSV/Parquet with ts/prediction/label columns.")
    p.add_argument("--ts-col", default="ts", help="Timestamp column name (default: ts)")
    p.add_argument("--pred-col", default="prediction", help="Prediction column name (default: prediction)")
    p.add_argument("--label-col", default="label", help="Label/actual return column name (default: label)")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output PNG path (default: warehouse/monitoring/training_ic_ir_pmr.png)")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p.parse_args()


def load_frame(path: Path, ts_col: str, pred_col: str, label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"data not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    missing = [c for c in (ts_col, pred_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing} in {path}")
    df = df.copy()
    df["_ts_utc"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=["_ts_utc"])
    df = df.sort_values("_ts_utc").set_index("_ts_utc")
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[pred_col, label_col])
    return df


def compute_rollings(df: pd.DataFrame, pred_col: str, label_col: str) -> pd.DataFrame:
    out = df.copy()
    for days in (15, 30, 60):
        window = f"{days}D"
        ic = out[pred_col].rolling(window, min_periods=2).corr(out[label_col])
        ic_mean = ic.rolling(window, min_periods=2).mean()
        ic_std = ic.rolling(window, min_periods=2).std(ddof=0)
        ir = ic_mean / ic_std.replace(0, np.nan)
        pmr = out[[pred_col, label_col]].rolling(window, min_periods=2).apply(
            lambda x: float((np.sign(x[:, 0]) == np.sign(x[:, 1])).mean()) if len(x) else np.nan,
            raw=True,
        )
        out[f"IC_{days}D"] = ic
        out[f"IR_{days}D"] = ir
        out[f"PMR_{days}D"] = pmr
    return out


def plot_series(df: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    for days, color in [(15, "orange"), (30, "deepskyblue"), (60, "violet")]:
        col = f"IC_{days}D"
        if col in df.columns:
            axes[0].plot(df.index, df[col], label=f"IC {days}D", color=color, alpha=0.9)
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_ylabel("IC")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.2)

    for days, color in [(15, "orange"), (30, "deepskyblue"), (60, "violet")]:
        col = f"IR_{days}D"
        if col in df.columns:
            axes[1].plot(df.index, df[col], label=f"IR {days}D", color=color, alpha=0.9)
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_ylabel("IR")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.2)

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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    logging.info("Saved plot: %s", out_path)


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    data_path = Path(args.data)
    out_path = Path(args.out)

    df = load_frame(data_path, args.ts_col, args.pred_col, args.label_col)
    if df.empty:
        logging.error("No rows after parsing input.")
        return

    df_roll = compute_rollings(df, pred_col=args.pred_col, label_col=args.label_col)
    plot_series(df_roll, out_path)


if __name__ == "__main__":
    main()
