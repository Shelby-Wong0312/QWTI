#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Concat monthly gdelt_hourly_YYYY-MM.parquet into data/gdelt_hourly.parquet
Designed for EC2 /home/ec2-user/Data layout.
"""

import glob
import os
import sys

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_PATH = os.path.join(DATA_DIR, "gdelt_hourly.parquet")


def main():
    pattern = os.path.join(DATA_DIR, "gdelt_hourly_202*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[ERROR] No files matched pattern: {pattern}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Found monthly parquet files:")
    for f in files:
        print("  -", f)

    dfs = []
    for f in files:
        print(f"[INFO] Reading {f} ...")
        df = pd.read_parquet(f)
        dfs.append(df)

    df_all = pd.concat(dfs)

    # If index is datetime, keep it; otherwise try to set index from first column if datetime
    if isinstance(df_all.index, pd.DatetimeIndex):
        df_all = df_all.sort_index()
    else:
        first_col = df_all.columns[0]
        if pd.api.types.is_datetime64_any_dtype(df_all[first_col]):
            df_all = df_all.set_index(first_col).sort_index()

    df_all.to_parquet(OUT_PATH)
    print(f"[INFO] Wrote {OUT_PATH}")
    print(f"[INFO] rows = {len(df_all)}")
    idx = df_all.index
    print(f"[INFO] ts range = {idx.min()}  ~  {idx.max()}")


if __name__ == "__main__":
    main()