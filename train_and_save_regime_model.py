#!/usr/bin/env python3
"""
Train and Save base_7_plus_regime Model

Trains LightGBM on specified date range and saves to models/.
Supports command-line arguments for flexible training periods.

Usage:
  python train_and_save_regime_model.py
  python train_and_save_regime_model.py --start 2024-10-01 --end 2025-12-01
  python train_and_save_regime_model.py --features features_hourly_with_regime.parquet --out models/base_seed202_regime_h1.pkl
"""

import argparse
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
import json

# Default paths
DEFAULT_FEATURES_PATH = Path("features_hourly_with_regime.parquet")
DEFAULT_MODEL_PATH = Path("models/base_seed202_regime_h1.pkl")
DEFAULT_CONFIG_PATH = Path("models/base_seed202_regime_h1_config.json")

# Model config
MODEL_CONFIG = {
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "num_leaves": 31,
    "random_state": 202,
    "verbosity": -1,
    "n_jobs": -1,
}

# Feature columns (base_7 + regime)
FEATURE_COLS = [
    "OIL_CORE_norm_art_cnt",
    "GEOPOL_norm_art_cnt",
    "USD_RATE_norm_art_cnt",
    "SUPPLY_CHAIN_norm_art_cnt",
    "MACRO_norm_art_cnt",
    "cl1_cl2",
    "ovx",
    "vol_regime_high",
    "ovx_high",
    "momentum_24h",
]

TARGET_COL = "wti_returns"

DEFAULT_TRAIN_HOURS = 720  # 30 days


def winsorize(series, lower=0.01, upper=0.99):
    """Winsorize at percentiles"""
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and save base_seed202_regime_h1 model")
    parser.add_argument("--start", type=str, default=None,
                        help="Training start date (YYYY-MM-DD), default: use last 720 hours")
    parser.add_argument("--end", type=str, default=None,
                        help="Training end date (YYYY-MM-DD), default: latest available")
    parser.add_argument("--features", type=str, default=str(DEFAULT_FEATURES_PATH),
                        help=f"Features parquet file (default: {DEFAULT_FEATURES_PATH})")
    parser.add_argument("--out", type=str, default=str(DEFAULT_MODEL_PATH),
                        help=f"Output model path (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--train-hours", type=int, default=DEFAULT_TRAIN_HOURS,
                        help=f"Training window in hours (default: {DEFAULT_TRAIN_HOURS})")
    return parser.parse_args()


def main():
    args = parse_args()

    features_path = Path(args.features)
    model_path = Path(args.out)
    config_path = model_path.with_suffix('.json').parent / (model_path.stem + '_config.json')

    print("=" * 70)
    print("TRAIN AND SAVE: base_seed202_regime_h1")
    print("=" * 70)
    print(f"Features: {features_path}")
    print(f"Output:   {model_path}")
    if args.start:
        print(f"Start:    {args.start}")
    if args.end:
        print(f"End:      {args.end}")

    # Load data
    print("\n[1/4] Loading features...")
    df = pd.read_parquet(features_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    print(f"   Total rows: {len(df)}")
    print(f"   Range: {df.index.min()} to {df.index.max()}")

    # Filter by date range if specified
    print("\n[2/4] Preparing training data...")
    df = df.sort_index()

    if args.start:
        start_dt = pd.to_datetime(args.start, utc=True)
        df = df[df.index >= start_dt]
        print(f"   Filtered from: {start_dt}")

    if args.end:
        end_dt = pd.to_datetime(args.end, utc=True)
        df = df[df.index <= end_dt]
        print(f"   Filtered to: {end_dt}")

    # Get data with valid features
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    # Use specified training hours or all available data
    if args.start and args.end:
        train_df = df.copy()
        train_hours = len(train_df)
    else:
        train_hours = args.train_hours
        train_df = df.iloc[-train_hours:]

    print(f"   Training rows: {len(train_df)}")
    print(f"   Training range: {train_df.index.min()} to {train_df.index.max()}")

    # Winsorize
    train_df = train_df.copy()
    for col in FEATURE_COLS:
        train_df[col] = winsorize(train_df[col])
    train_df[TARGET_COL] = winsorize(train_df[TARGET_COL])

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    # Train model
    print("\n[3/4] Training LightGBM...")
    model = lgb.LGBMRegressor(**MODEL_CONFIG)
    model.fit(X_train, y_train)

    print("   Feature importance:")
    for feat, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
        print(f"     {feat:<35} {imp:.1f}")

    # Save model
    print("\n[4/4] Saving model...")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   Model saved: {model_path}")

    # Save config
    config = {
        "model_name": "base_seed202_regime_h1",
        "model_type": "LightGBM",
        "model_params": MODEL_CONFIG,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "train_hours": len(train_df),
        "train_start": train_df.index.min().isoformat(),
        "train_end": train_df.index.max().isoformat(),
        "created_at": datetime.now().isoformat(),
        "features_file": str(features_path.name),
        "n_features": len(FEATURE_COLS),
        "args": {
            "start": args.start,
            "end": args.end,
            "features": str(features_path),
            "out": str(model_path),
        }
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   Config saved: {config_path}")

    # Verify model can be loaded
    print("\n   Verifying model load...")
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    test_pred = loaded_model.predict(X_train.iloc[:5])
    print(f"   Test predictions: {test_pred[:5]}")

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Config: {config_path}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Training period: {config['train_start']} to {config['train_end']}")
    print(f"Training rows: {len(train_df)}")

    return model, config


if __name__ == "__main__":
    model, config = main()
