#!/usr/bin/env python3
"""
Build Features with Regime - 正式特徵管線

將 Regime 特徵加入 features_hourly_with_term.parquet：
- vol_regime_high: 波動率高位旗標
- ovx_high: OVX 高位旗標
- momentum_24h: 24h 累積報酬

輸出: features_hourly_with_regime.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
INPUT_PATH = Path("features_hourly_with_term.parquet")
EVENTS_PATH = Path("data/events_calendar.csv")
OUTPUT_PATH = Path("features_hourly_with_regime.parquet")

# Original features
BASE_FEATURES = [
    "OIL_CORE_norm_art_cnt",
    "GEOPOL_norm_art_cnt",
    "USD_RATE_norm_art_cnt",
    "SUPPLY_CHAIN_norm_art_cnt",
    "MACRO_norm_art_cnt",
    "cl1_cl2",
    "ovx",
]

TARGET_COL = "wti_returns"


def load_base_features():
    """Load base feature data"""
    print("[1/5] Loading base features...")
    df = pd.read_parquet(INPUT_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    print(f"   Rows: {len(df)}")
    print(f"   Range: {df.index.min()} to {df.index.max()}")
    print(f"   Columns: {list(df.columns)}")
    return df


def load_events():
    """Load EIA event calendar"""
    print("\n[2/5] Loading event calendar...")
    if not EVENTS_PATH.exists():
        print("   WARNING: Event calendar not found")
        return pd.DataFrame()

    events = pd.read_csv(EVENTS_PATH)
    events['ts'] = pd.to_datetime(events['ts'], utc=True)
    print(f"   Events loaded: {len(events)} records")
    return events


def add_eia_features(df, events):
    """Add EIA event window flags"""
    print("\n[3/5] Adding EIA event features...")

    df = df.copy()

    # Initialize
    df['eia_pre_4h'] = 0
    df['eia_post_4h'] = 0
    df['eia_day'] = 0

    if events.empty:
        return df

    eia_events = pd.to_datetime(events[events['event_type'] == 'EIA']['ts'], utc=True)

    for event_ts in eia_events:
        hours_diff = (df.index - event_ts).total_seconds() / 3600
        df.loc[(hours_diff >= -4) & (hours_diff < 0), 'eia_pre_4h'] = 1
        df.loc[(hours_diff >= 0) & (hours_diff <= 4), 'eia_post_4h'] = 1
        df.loc[df.index.date == event_ts.date(), 'eia_day'] = 1

    print(f"   EIA pre-4h: {df['eia_pre_4h'].sum()} flags")
    print(f"   EIA post-4h: {df['eia_post_4h'].sum()} flags")
    print(f"   EIA day: {df['eia_day'].sum()} flags")

    return df


def add_regime_features(df):
    """Add regime features"""
    print("\n[4/5] Adding regime features...")

    df = df.copy()

    # Volatility regime (rolling 24h std)
    df['vol_24h'] = df[TARGET_COL].rolling(24, min_periods=1).std()
    vol_75 = df['vol_24h'].quantile(0.75)
    vol_25 = df['vol_24h'].quantile(0.25)
    df['vol_regime_high'] = (df['vol_24h'] > vol_75).astype(int)
    df['vol_regime_low'] = (df['vol_24h'] < vol_25).astype(int)
    print(f"   vol_regime_high threshold: {vol_75:.6f}")
    print(f"   vol_regime_high flags: {df['vol_regime_high'].sum()}")

    # OVX regime
    df['ovx_high'] = (df['ovx'] > 0.7).astype(int)
    df['ovx_low'] = (df['ovx'] < 0.3).astype(int)
    print(f"   ovx_high flags: {df['ovx_high'].sum()}")

    # Momentum (24h cumulative return)
    df['momentum_24h'] = df[TARGET_COL].rolling(24, min_periods=1).sum()
    print(f"   momentum_24h range: [{df['momentum_24h'].min():.4f}, {df['momentum_24h'].max():.4f}]")

    # GDELT intensity
    gdelt_cols = [c for c in BASE_FEATURES if 'norm_art_cnt' in c]
    df['gdelt_intensity'] = df[gdelt_cols].sum(axis=1)
    gdelt_75 = df['gdelt_intensity'].quantile(0.75)
    df['gdelt_high'] = (df['gdelt_intensity'] > gdelt_75).astype(int)
    print(f"   gdelt_high flags: {df['gdelt_high'].sum()}")

    # Drop intermediate columns
    df = df.drop(columns=['vol_24h', 'gdelt_intensity'], errors='ignore')

    return df


def save_features(df):
    """Save to parquet"""
    print("\n[5/5] Saving features...")

    # Ensure proper column order
    feature_cols = BASE_FEATURES + [
        'eia_pre_4h', 'eia_post_4h', 'eia_day',
        'vol_regime_high', 'vol_regime_low',
        'ovx_high', 'ovx_low',
        'momentum_24h', 'gdelt_high',
    ]

    # Keep only existing columns
    existing_cols = [c for c in feature_cols if c in df.columns]
    output_cols = existing_cols + [TARGET_COL]

    df_out = df[output_cols]

    # Save
    df_out.to_parquet(OUTPUT_PATH)

    print(f"   Output: {OUTPUT_PATH}")
    print(f"   Shape: {df_out.shape}")
    print(f"   Columns: {list(df_out.columns)}")

    return df_out


def main():
    print("=" * 70)
    print("BUILD FEATURES WITH REGIME")
    print("=" * 70)

    # Load
    df = load_base_features()
    events = load_events()

    # Add features
    df = add_eia_features(df, events)
    df = add_regime_features(df)

    # Save
    df_out = save_features(df)

    # Summary
    print("\n" + "=" * 70)
    print("FEATURE PIPELINE COMPLETE")
    print("=" * 70)

    print("\nFeature Groups:")
    print("  Base GDELT (5):", [c for c in BASE_FEATURES if 'norm_art_cnt' in c])
    print("  Base Market (2):", ['cl1_cl2', 'ovx'])
    print("  EIA Events (3):", ['eia_pre_4h', 'eia_post_4h', 'eia_day'])
    print("  Regime (6):", ['vol_regime_high', 'vol_regime_low', 'ovx_high', 'ovx_low', 'momentum_24h', 'gdelt_high'])
    print("  Target (1):", [TARGET_COL])

    print(f"\nTotal features: {len(df_out.columns) - 1}")
    print(f"Output file: {OUTPUT_PATH}")

    return df_out


if __name__ == "__main__":
    df = main()
