#!/usr/bin/env python3
"""
Post-backfill pipeline:
1. Rebuild gdelt_hourly.parquet from monthly files
2. Run IC/PMR evaluation with H=[1,2,3], lag=1h
3. Check if Hard IC threshold is achieved
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import sys

print("=" * 80)
print("POST-BACKFILL PIPELINE")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: Rebuild gdelt_hourly.parquet from monthly files
# ============================================================================
print("### STEP 1: Rebuild gdelt_hourly.parquet ###")
print()

MONTHLY_DIR = Path("data/gdelt_hourly_monthly")
OUTPUT_PATH = Path("data/gdelt_hourly.parquet")
BACKUP_PATH = Path("data/gdelt_hourly.parquet.backup_pre_backfill")

# Backup old file
if OUTPUT_PATH.exists():
    print(f"Backing up existing file to {BACKUP_PATH}")
    if BACKUP_PATH.exists():
        BACKUP_PATH.unlink()
    OUTPUT_PATH.rename(BACKUP_PATH)

# Load all monthly files
monthly_files = sorted(MONTHLY_DIR.glob("gdelt_hourly_*.parquet"))
print(f"Found {len(monthly_files)} monthly files")

dfs = []
for f in monthly_files:
    print(f"  Loading {f.name}...")
    df = pd.read_parquet(f)
    print(f"    Shape: {df.shape}, Date range: {df['ts_utc'].min()} to {df['ts_utc'].max()}")
    dfs.append(df)

# Concatenate
total_df = pd.concat(dfs, ignore_index=True)
total_df = total_df.sort_values('ts_utc').reset_index(drop=True)

# Check duplicates
duplicates = total_df['ts_utc'].duplicated().sum()
if duplicates > 0:
    print(f"Found {duplicates} duplicate timestamps, keeping last")
    total_df = total_df.drop_duplicates(subset='ts_utc', keep='last').reset_index(drop=True)

# Save
total_df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nWrote {len(total_df)} rows to {OUTPUT_PATH}")
print(f"Columns: {len(total_df.columns)}")
print(f"Date range: {total_df['ts_utc'].min()} to {total_df['ts_utc'].max()}")
print()

# Verify bucket data coverage
bucket_cols = [c for c in total_df.columns if 'OIL_CORE_art_cnt' in c or 'GEOPOL_art_cnt' in c]
if bucket_cols:
    non_null = total_df[bucket_cols[0]].notna().sum()
    pct = 100 * non_null / len(total_df)
    print(f"Bucket data coverage: {non_null}/{len(total_df)} rows ({pct:.1f}%)")
print()

# ============================================================================
# STEP 2: Run IC/PMR Evaluation
# ============================================================================
print("### STEP 2: Run IC/PMR Evaluation (H=[1,2,3], lag=1h) ###")
print()

# Find the IC evaluation script
eval_script = None
possible_scripts = [
    "jobs/evaluate_ic_pmr.py",
    "evaluate_composite_ic.py",
    "warehouse/scripts/evaluate_ic.py"
]

for script_path in possible_scripts:
    if Path(script_path).exists():
        eval_script = script_path
        break

if eval_script is None:
    print("[WARN] IC evaluation script not found, will need manual run")
    print("Expected locations:", possible_scripts)
    print()
else:
    print(f"Running {eval_script}...")
    try:
        result = subprocess.run(
            [sys.executable, eval_script],
            capture_output=True,
            text=True,
            timeout=600
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print()
    except subprocess.TimeoutExpired:
        print("[ERROR] Evaluation script timed out after 10 minutes")
    except Exception as e:
        print(f"[ERROR] Failed to run evaluation: {e}")
    print()

# ============================================================================
# STEP 3: Check Hard IC Threshold
# ============================================================================
print("### STEP 3: Check Hard IC Threshold ###")
print()

# Look for composite results
composite_files = list(Path("warehouse/ic").glob("composite*_summary_short.csv"))
if not composite_files:
    print("[WARN] No composite IC results found in warehouse/ic/")
    print()
else:
    print(f"Found {len(composite_files)} composite result files")
    for f in composite_files:
        print(f"\n## {f.name} ##")
        df = pd.read_csv(f)
        print(df.to_string(index=False))

        # Check Hard threshold
        if 'IC' in df.columns and 'IR' in df.columns and 'PMR' in df.columns:
            hard_candidates = df[
                (df['IC'] >= 0.02) &
                (df['IR'] >= 0.5) &
                (df['PMR'] >= 0.55)
            ]
            if not hard_candidates.empty:
                print(f"\n*** HARD IC THRESHOLD ACHIEVED! ***")
                print(hard_candidates.to_string(index=False))
            else:
                print(f"\nNo Hard IC candidates (need IC>=0.02, IR>=0.5, PMR>=0.55)")
    print()

# ============================================================================
# STEP 4: Summary Report
# ============================================================================
print("### STEP 4: Summary Report ###")
print()

print(f"GDELT hourly file: {OUTPUT_PATH}")
print(f"  Rows: {len(total_df)}")
print(f"  Columns: {len(total_df.columns)}")
print(f"  Date range: {total_df['ts_utc'].min()} to {total_df['ts_utc'].max()}")
print()

# Check features_hourly.parquet
features_path = Path("data/features_hourly.parquet")
if features_path.exists():
    features = pd.read_parquet(features_path)
    print(f"Features hourly file: {features_path}")
    print(f"  Rows: {len(features)}")
    print(f"  Columns: {features.columns.tolist()}")
    print(f"  Date range: {features['ts_utc'].min()} to {features['ts_utc'].max()}")

    # Check price data
    if 'ret_1h' in features.columns:
        non_null_ret = features['ret_1h'].notna().sum()
        non_zero_ret = ((features['ret_1h'] != 0) & features['ret_1h'].notna()).sum()
        print(f"  ret_1h non-null: {non_null_ret}/{len(features)} ({100*non_null_ret/len(features):.1f}%)")
        print(f"  ret_1h non-zero: {non_zero_ret}/{len(features)} ({100*non_zero_ret/len(features):.1f}%)")
else:
    print(f"[WARN] Features file not found: {features_path}")
print()

print("=" * 80)
print(f"PIPELINE COMPLETE! End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
