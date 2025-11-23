#!/usr/bin/env python3
"""
Rebuild gdelt_hourly.parquet from monthly files
Fix the PMR=0 issue by properly merging monthly data with bucket columns
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("Rebuild GDELT Total File from Monthly Files")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Paths
MONTHLY_DIR = Path("data/gdelt_hourly_monthly")
OUTPUT_PATH = Path("data/gdelt_hourly.parquet")
BACKUP_PATH = Path("data/gdelt_hourly.parquet.backup")

# Backup existing file
if OUTPUT_PATH.exists():
    print(f"### Backing up existing file ###")
    OUTPUT_PATH.rename(BACKUP_PATH)
    print(f"Backed up to: {BACKUP_PATH}")
    print()

# Load all monthly files
print(f"### Loading Monthly Files ###")
monthly_files = sorted(MONTHLY_DIR.glob("gdelt_hourly_*.parquet"))
print(f"Found {len(monthly_files)} monthly files")

if len(monthly_files) == 0:
    print("[ERROR] No monthly files found!")
    exit(1)

dfs = []
for f in monthly_files:
    print(f"Loading {f.name}...")
    df = pd.read_parquet(f)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['ts_utc'].min()} to {df['ts_utc'].max()}")

    # Check bucket columns
    bucket_cols = [c for c in df.columns if any(b in c for b in ['OIL_CORE', 'USD_RATE', 'GEOPOL', 'SUPPLY', 'MACRO', 'ESG'])]
    if bucket_cols:
        non_null = sum(df[c].notna().sum() for c in bucket_cols)
        print(f"  Bucket columns: {len(bucket_cols)}, non-null values: {non_null}")

    dfs.append(df)

print()

# Concatenate all monthly files
print(f"### Concatenating Files ###")
total_df = pd.concat(dfs, ignore_index=True)
print(f"Combined shape: {total_df.shape}")
print(f"Combined date range: {total_df['ts_utc'].min()} to {total_df['ts_utc'].max()}")
print()

# Sort by time
total_df = total_df.sort_values('ts_utc').reset_index(drop=True)

# Check for duplicates
print(f"### Checking for Duplicates ###")
duplicates = total_df['ts_utc'].duplicated().sum()
if duplicates > 0:
    print(f"Found {duplicates} duplicate timestamps")
    print(f"Keeping last occurrence of each timestamp")
    total_df = total_df.drop_duplicates(subset='ts_utc', keep='last').reset_index(drop=True)
    print(f"Shape after deduplication: {total_df.shape}")
else:
    print(f"No duplicates found")
print()

# Verify bucket columns have data
print(f"### Verifying Bucket Columns ###")
bucket_names = ['OIL_CORE', 'USD_RATE', 'GEOPOL', 'SUPPLY_CHAIN', 'MACRO', 'ESG_POLICY']
for bucket in bucket_names:
    bucket_cols = [c for c in total_df.columns if bucket in c]
    if bucket_cols:
        non_null = sum(total_df[c].notna().sum() for c in bucket_cols)
        total_values = len(total_df) * len(bucket_cols)
        pct = 100 * non_null / total_values if total_values > 0 else 0
        print(f"  {bucket}: {len(bucket_cols)} columns, {non_null}/{total_values} non-null ({pct:.1f}%)")
print()

# Save new total file
print(f"### Saving New Total File ###")
total_df.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved to: {OUTPUT_PATH}")
print(f"Final shape: {total_df.shape}")
print(f"Final columns: {len(total_df.columns)}")
print()

# Create summary report
print(f"### Summary Report ###")
print(f"Columns in final file:")
for col in total_df.columns:
    dtype = total_df[col].dtype
    null_count = total_df[col].isna().sum()
    null_pct = 100 * null_count / len(total_df)
    print(f"  {col}: {dtype}, {len(total_df)-null_count} non-null ({100-null_pct:.1f}%)")

print()
print("=" * 80)
print(f"Rebuild Complete! End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()
print("NEXT STEPS:")
print("1. Re-run debug_pmr_zero.py to verify bucket data is now available")
print("2. Re-run IC evaluation to calculate PMR")
print("3. Check if PMR > 0 after using real bucket data")
