#!/usr/bin/env python3
"""
Drift Period Validation: Train and validate LightGBM on 2025-07~11 data only

Goal: Check if model can recover Hard Gate metrics when trained on drift period data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import lightgbm as lgb
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Paths
FEATURES_PATH = Path("features_hourly_with_term.parquet")
OUTPUT_DIR = Path("warehouse/ic")

# Model config (same as base_seed202_lean7_h1)
MODEL_CONFIG = {
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "num_leaves": 31,
    "random_state": 202,
    "verbosity": -1,
    "n_jobs": -1,
}

FEATURE_COLS = [
    "OIL_CORE_norm_art_cnt",
    "GEOPOL_norm_art_cnt",
    "USD_RATE_norm_art_cnt",
    "SUPPLY_CHAIN_norm_art_cnt",
    "MACRO_norm_art_cnt",
    "cl1_cl2",
    "ovx",
]

TARGET_COL = "wti_returns"

# Period definitions
DRIFT_PERIOD = ("2025-07-01", "2025-11-30")
GOOD_PERIOD = ("2025-02-01", "2025-05-31")

# Rolling window settings (same as stability validation)
TRAIN_HOURS = 1440  # 60 days
TEST_HOURS = 360    # 15 days (non-overlapping windows)


def load_data():
    """Load and prepare data"""
    print("[1/7] Loading data...")
    df = pd.read_parquet(FEATURES_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    return df


def get_period_data(df, start, end):
    """Extract data for a specific period"""
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC")
    return df[(df.index >= start_dt) & (df.index <= end_dt)]


def winsorize(series, lower=0.01, upper=0.99):
    """Winsorize at percentiles"""
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)


def run_rolling_validation(df, period_name):
    """Run rolling window validation on a period"""
    print(f"\n   Running validation on {period_name}...")

    # Winsorize
    df = df.copy()
    for col in FEATURE_COLS:
        df[col] = winsorize(df[col])
    df[TARGET_COL] = winsorize(df[TARGET_COL])

    n_samples = len(df)
    if n_samples < TRAIN_HOURS + TEST_HOURS:
        print(f"   WARNING: Not enough data ({n_samples} < {TRAIN_HOURS + TEST_HOURS})")
        return None

    results = []
    window_idx = 0
    start_idx = TRAIN_HOURS

    while start_idx + TEST_HOURS <= n_samples:
        train_df = df.iloc[start_idx - TRAIN_HOURS:start_idx]
        test_df = df.iloc[start_idx:start_idx + TEST_HOURS]

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]
        X_test = test_df[FEATURE_COLS]
        y_test = test_df[TARGET_COL]

        # Train
        model = lgb.LGBMRegressor(**MODEL_CONFIG)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate IC
        ic, _ = spearmanr(y_pred, y_test)

        # Get test month
        test_month = test_df.index[len(test_df)//2].strftime("%Y-%m")

        results.append({
            'window': window_idx + 1,
            'test_month': test_month,
            'test_start': test_df.index.min(),
            'test_end': test_df.index.max(),
            'ic': ic,
            'n_train': len(train_df),
            'n_test': len(test_df),
        })

        window_idx += 1
        start_idx += TEST_HOURS

    return pd.DataFrame(results)


def calculate_metrics(results_df):
    """Calculate IC/IR/PMR from window results"""
    if results_df is None or len(results_df) == 0:
        return None

    ic_values = results_df['ic'].dropna()

    metrics = {
        'n_windows': len(results_df),
        'ic_mean': ic_values.mean(),
        'ic_median': ic_values.median(),
        'ic_std': ic_values.std(),
        'ir': ic_values.mean() / ic_values.std() if ic_values.std() > 0 else np.nan,
        'pmr': (ic_values > 0).mean(),
        'ic_min': ic_values.min(),
        'ic_max': ic_values.max(),
    }

    # Hard Gate check
    metrics['ic_pass'] = metrics['ic_median'] >= 0.02
    metrics['ir_pass'] = metrics['ir'] >= 0.5
    metrics['pmr_pass'] = metrics['pmr'] >= 0.55
    metrics['all_pass'] = metrics['ic_pass'] and metrics['ir_pass'] and metrics['pmr_pass']

    return metrics


def main():
    print("=" * 70)
    print("DRIFT PERIOD VALIDATION")
    print("=" * 70)
    print(f"Model: LightGBM (depth=5, lr=0.1, n=100, leaves=31, seed=202)")
    print(f"Features: {len(FEATURE_COLS)} (same as base_seed202_lean7_h1)")
    print(f"Train: {TRAIN_HOURS}h ({TRAIN_HOURS//24}d), Test: {TEST_HOURS}h ({TEST_HOURS//24}d)")
    print("=" * 70)

    # Load data
    df = load_data()

    # Get period data
    df_drift = get_period_data(df, *DRIFT_PERIOD)
    df_good = get_period_data(df, *GOOD_PERIOD)

    print(f"\n   Drift period ({DRIFT_PERIOD[0]} to {DRIFT_PERIOD[1]}): {len(df_drift)} rows")
    print(f"   Good period ({GOOD_PERIOD[0]} to {GOOD_PERIOD[1]}): {len(df_good)} rows")

    # Run validation on drift period
    print("\n[2/7] Validating on DRIFT period (2025-07~11)...")
    results_drift = run_rolling_validation(df_drift, "Drift (Jul-Nov)")
    metrics_drift = calculate_metrics(results_drift)

    # Run validation on good period for comparison
    print("\n[3/7] Validating on GOOD period (2025-02~05) for comparison...")
    results_good = run_rolling_validation(df_good, "Good (Feb-May)")
    metrics_good = calculate_metrics(results_good)

    # Print window-by-window results
    print("\n[4/7] Window-by-Window Results")
    print("=" * 70)

    if results_drift is not None:
        print("\nDRIFT Period Windows:")
        print(f"{'Window':<8} {'Month':<10} {'IC':<12} {'Status':<8}")
        print("-" * 40)
        for _, row in results_drift.iterrows():
            status = "OK" if row['ic'] > 0 else "X"
            hard = "HARD" if row['ic'] >= 0.02 else ""
            print(f"{row['window']:<8} {row['test_month']:<10} {row['ic']:<12.6f} {status:<8} {hard}")

    if results_good is not None:
        print("\nGOOD Period Windows:")
        print(f"{'Window':<8} {'Month':<10} {'IC':<12} {'Status':<8}")
        print("-" * 40)
        for _, row in results_good.iterrows():
            status = "OK" if row['ic'] > 0 else "X"
            hard = "HARD" if row['ic'] >= 0.02 else ""
            print(f"{row['window']:<8} {row['test_month']:<10} {row['ic']:<12.6f} {status:<8} {hard}")

    # Print metrics comparison
    print("\n[5/7] Metrics Comparison")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Drift (Jul-Nov)':<20} {'Good (Feb-May)':<20} {'Threshold':<15}")
    print("-" * 75)

    if metrics_drift and metrics_good:
        for key, threshold in [('ic_median', '>=0.02'), ('ir', '>=0.5'), ('pmr', '>=0.55')]:
            d = metrics_drift[key]
            g = metrics_good[key]
            d_pass = metrics_drift[f'{key.split("_")[0]}_pass'] if key != 'pmr' else metrics_drift['pmr_pass']
            g_pass = metrics_good[f'{key.split("_")[0]}_pass'] if key != 'pmr' else metrics_good['pmr_pass']
            d_status = "PASS" if d_pass else "FAIL"
            g_status = "PASS" if g_pass else "FAIL"
            print(f"{key:<20} {d:<12.4f} {d_status:<8} {g:<12.4f} {g_status:<8} {threshold}")

        print(f"\n{'n_windows':<20} {metrics_drift['n_windows']:<20} {metrics_good['n_windows']:<20}")
        print(f"{'ic_range':<20} [{metrics_drift['ic_min']:.4f}, {metrics_drift['ic_max']:.4f}]      [{metrics_good['ic_min']:.4f}, {metrics_good['ic_max']:.4f}]")

    # Hard Gate Summary
    print("\n[6/7] Hard Gate Summary")
    print("=" * 70)

    if metrics_drift:
        print(f"\nDRIFT Period (2025-07~11):")
        print(f"   IC median >= 0.02: {metrics_drift['ic_median']:.4f} {'PASS' if metrics_drift['ic_pass'] else 'FAIL'}")
        print(f"   IR >= 0.5:         {metrics_drift['ir']:.4f} {'PASS' if metrics_drift['ir_pass'] else 'FAIL'}")
        print(f"   PMR >= 0.55:       {metrics_drift['pmr']:.2%} {'PASS' if metrics_drift['pmr_pass'] else 'FAIL'}")
        print(f"   Overall:           {'ALL PASS' if metrics_drift['all_pass'] else 'FAILED'}")

    if metrics_good:
        print(f"\nGOOD Period (2025-02~05):")
        print(f"   IC median >= 0.02: {metrics_good['ic_median']:.4f} {'PASS' if metrics_good['ic_pass'] else 'FAIL'}")
        print(f"   IR >= 0.5:         {metrics_good['ir']:.4f} {'PASS' if metrics_good['ir_pass'] else 'FAIL'}")
        print(f"   PMR >= 0.55:       {metrics_good['pmr']:.2%} {'PASS' if metrics_good['pmr_pass'] else 'FAIL'}")
        print(f"   Overall:           {'ALL PASS' if metrics_good['all_pass'] else 'FAILED'}")

    # Conclusion and Recommendation
    print("\n[7/7] Conclusion & Recommendation")
    print("=" * 70)

    if metrics_drift and metrics_drift['all_pass']:
        print("\n[OK] DRIFT PERIOD MODEL PASSES HARD GATE")
        print("   Model trained on recent data can achieve Hard thresholds.")
        print("\n   RECOMMENDATION: Implement regime-aware routing")
        print("   - Good regime (Feb-May pattern): Use original base_seed202_lean7_h1")
        print("   - Drift regime (Jul-Nov pattern): Use drift-retrained model")
        print("   - Regime detection: Monitor IC rolling 15d, switch if < 0.01 for 5 consecutive")
    elif metrics_drift and metrics_drift['ic_median'] > 0:
        print("\n[~] DRIFT PERIOD MODEL SHOWS IMPROVEMENT BUT NOT HARD PASS")
        print(f"   IC median: {metrics_drift['ic_median']:.4f} (need >=0.02)")
        print(f"   IR: {metrics_drift['ir']:.4f} (need >=0.5)")
        print(f"   PMR: {metrics_drift['pmr']:.2%} (need >=55%)")
        print("\n   RECOMMENDATION: ")
        print("   - Consider shorter training window (30d instead of 60d)")
        print("   - Add regime-specific features (volatility regime, news intensity)")
        print("   - Reduce position sizing during drift regime")
    else:
        print("\n[X] DRIFT PERIOD MODEL FAILS TO RECOVER")
        print("   Even with recent data, model cannot achieve positive IC.")
        print("\n   RECOMMENDATION:")
        print("   - Market structure may have fundamentally changed")
        print("   - Consider halting strategy or using ensemble with diverse features")
        print("   - Investigate if GDELT signal relevance has degraded")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if results_drift is not None:
        drift_path = OUTPUT_DIR / f"drift_period_windows_{timestamp}.csv"
        results_drift.to_csv(drift_path, index=False)
        print(f"\n   Drift windows saved: {drift_path}")

    if results_good is not None:
        good_path = OUTPUT_DIR / f"good_period_windows_{timestamp}.csv"
        results_good.to_csv(good_path, index=False)
        print(f"   Good windows saved: {good_path}")

    # Save summary
    summary = {
        'period': ['drift', 'good'],
        'start': [DRIFT_PERIOD[0], GOOD_PERIOD[0]],
        'end': [DRIFT_PERIOD[1], GOOD_PERIOD[1]],
    }
    if metrics_drift:
        for k, v in metrics_drift.items():
            if k not in summary:
                summary[k] = [v, metrics_good.get(k) if metrics_good else None]

    summary_df = pd.DataFrame(summary)
    summary_path = OUTPUT_DIR / f"drift_period_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"   Summary saved: {summary_path}")

    print("\n" + "=" * 70)
    print("DRIFT PERIOD VALIDATION COMPLETE")
    print("=" * 70)

    return {
        'results_drift': results_drift,
        'results_good': results_good,
        'metrics_drift': metrics_drift,
        'metrics_good': metrics_good,
    }


if __name__ == "__main__":
    results = main()
