"""
Integration Script: Merge term structure and OVX with GDELT features
LEAN 7-FEATURE CONFIGURATION (crack spreads excluded - zero importance)

Goal: Validate that lean configuration still exceeds IR≥0.5 threshold.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
GDELT_HOURLY_PATH = Path('data/gdelt_hourly.parquet')
FEATURES_HOURLY_PATH = Path('data/features_hourly.parquet')
TERM_CRACK_OVX_PATH = Path('data/term_crack_ovx_hourly.csv')
OUTPUT_PATH = Path('features_hourly_with_term.parquet')

# Keep only 5 baseline GDELT bucket features
GDELT_FEATURES = [
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt'
]

# Lean market features (crack spreads removed - zero importance)
MARKET_FEATURES = [
    'cl1_cl2',
    'ovx'
]

def main():
    print("="*60)
    print("LEAN 7-FEATURE INTEGRATION: GDELT + TERM STRUCTURE + OVX")
    print("="*60)
    print("(Crack spreads excluded - zero importance in 9-feature model)")

    # 1. Load GDELT features
    print("\n[1/6] Loading GDELT features from parquet...")
    df_gdelt = pd.read_parquet(GDELT_HOURLY_PATH)
    print(f"  - Shape: {df_gdelt.shape}")
    print(f"  - Columns: {len(df_gdelt.columns)}")

    # Keep only timestamp and 5 bucket features
    keep_cols = ['ts_utc'] + GDELT_FEATURES
    df_gdelt = df_gdelt[keep_cols]
    print(f"  - Kept columns: {list(df_gdelt.columns)}")
    print(f"  - Date range: {df_gdelt['ts_utc'].min()} to {df_gdelt['ts_utc'].max()}")

    # 2. Load price returns
    print("\n[2/6] Loading price returns from parquet...")
    df_prices = pd.read_parquet(FEATURES_HOURLY_PATH)
    print(f"  - Shape: {df_prices.shape}")
    print(f"  - Columns: {list(df_prices.columns)}")
    print(f"  - Date range: {df_prices['ts_utc'].min()} to {df_prices['ts_utc'].max()}")

    # 3. Load term/crack/OVX data
    print("\n[3/6] Loading term structure / crack spread / OVX data...")
    df_term = pd.read_csv(TERM_CRACK_OVX_PATH)
    print(f"  - Shape: {df_term.shape}")
    print(f"  - Columns: {list(df_term.columns)}")

    # Parse timestamp and keep only cl1_cl2 and ovx
    df_term['ts_utc'] = pd.to_datetime(df_term['ts'], utc=True)
    df_term = df_term[['ts_utc', 'cl1_cl2', 'ovx']]  # Drop crack spreads

    print(f"  - Date range: {df_term['ts_utc'].min()} to {df_term['ts_utc'].max()}")
    print(f"  - Missing values:")
    print(df_term.isnull().sum())

    # 4. Merge all three datasets on ts_utc
    print("\n[4/6] Merging datasets on ts_utc timestamp...")

    # Start with prices (has the most complete timestamp coverage)
    df_merged = df_prices.copy()
    print(f"  - Starting with prices: {df_merged.shape}")

    # Merge GDELT features
    df_merged = df_merged.merge(df_gdelt, on='ts_utc', how='left')
    print(f"  - After GDELT merge: {df_merged.shape}")

    # Merge term/crack/OVX features
    df_merged = df_merged.merge(df_term, on='ts_utc', how='left')
    print(f"  - After term/crack/OVX merge: {df_merged.shape}")

    # 5. Handle missing values
    print("\n[5/6] Handling missing values...")
    print("  - Missing values before filling:")
    print(df_merged.isnull().sum())

    # Fill GDELT features with 0 (no news activity)
    for feat in GDELT_FEATURES:
        df_merged[feat] = df_merged[feat].fillna(0)

    # Fill market features
    # cl1_cl2: 0 = no contango/backwardation
    # ovx: forward fill then backward fill
    df_merged['cl1_cl2'] = df_merged['cl1_cl2'].fillna(0)
    df_merged['ovx'] = df_merged['ovx'].ffill().bfill()

    print("\n  - Missing values after filling:")
    print(df_merged.isnull().sum())

    # Rename ret_1h to wti_returns for consistency
    df_merged = df_merged.rename(columns={'ret_1h': 'wti_returns'})

    # 6. Save integrated dataset
    print(f"\n[6/6] Saving integrated dataset to {OUTPUT_PATH}...")

    # Set timestamp as index
    df_merged = df_merged.set_index('ts_utc')
    df_merged.to_parquet(OUTPUT_PATH)

    print("\n" + "="*60)
    print("INTEGRATION COMPLETE")
    print("="*60)
    print(f"Output: {OUTPUT_PATH}")
    print(f"  - Shape: {df_merged.shape}")
    print(f"  - GDELT features: {len(GDELT_FEATURES)}")
    print(f"  - Market features: {len(MARKET_FEATURES)} (cl1_cl2, ovx)")
    print(f"  - Target: wti_returns")
    print(f"  - Total features: {len(GDELT_FEATURES) + len(MARKET_FEATURES)} (LEAN CONFIG)")
    print(f"  - Date range: {df_merged.index.min()} to {df_merged.index.max()}")
    print(f"  - Total samples: {len(df_merged)}")

    # Show feature summary
    print("\nFeature list (LEAN 7-FEATURE CONFIGURATION):")
    print("  GDELT (5):", GDELT_FEATURES)
    print("  Market (2):", MARKET_FEATURES)
    print("  Target (1):", ['wti_returns'])
    print("\n  Excluded (zero importance):", ['crack_rb', 'crack_ho'])

    # Show sample
    print("\nSample of integrated data (last 5 rows):")
    display_cols = GDELT_FEATURES + MARKET_FEATURES + ['wti_returns']
    print(df_merged[display_cols].tail())

    return df_merged

if __name__ == '__main__':
    df = main()
