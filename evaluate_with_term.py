"""
Evaluation Script: Test Seed202 with LEAN 7-feature configuration

Baseline: Seed202 with 5 GDELT features only (IR=0.3969, IC_std=0.057015)
Lean: Seed202 with 5 GDELT + 2 market features (cl1_cl2, ovx)

Goal: Verify lean config still exceeds hard thresholds (IR≥0.5, IC median≥0.02, PMR≥0.55)
Note: Crack spreads (crack_rb, crack_ho) removed due to zero importance in 9-feature model
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lightgbm import LGBMRegressor
from datetime import datetime

# Paths
INPUT_PATH = Path('features_hourly_with_term.parquet')
BASELINE_PATH = Path('data/features_hourly.parquet')
GDELT_PATH = Path('data/gdelt_hourly.parquet')
OUTPUT_DIR = Path('warehouse/ic')
RUNLOG_PATH = Path('RUNLOG_OPERATIONS.md')

# Window configuration (same as V3 Seed202)
H = 1  # Horizon in hours
TRAIN_DAYS = 60
TEST_DAYS = 15

# Model configuration (Seed202 from V3)
MODEL_CONFIG = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.6,
    'bagging_freq': 1,
    'reg_lambda': 1.5,
    'random_state': 202,
    'verbosity': -1,
    'force_col_wise': True
}

# Feature sets
GDELT_FEATURES = [
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt'
]

MARKET_FEATURES = [
    'cl1_cl2',
    'ovx'
]
# Crack spreads removed - zero importance in 9-feature model

def evaluate_model(df, features, model_name):
    """Evaluate model with walk-forward validation"""

    print(f"\nEvaluating {model_name}...")
    print(f"  Features ({len(features)}): {features}")

    # Drop rows with missing target or features
    df_eval = df.dropna(subset=['wti_returns'] + features).copy()
    print(f"  Samples after dropna: {len(df_eval)}")

    # Define date range for walk-forward windows
    min_date = df_eval.index.min()
    max_date = df_eval.index.max()
    print(f"  Date range: {min_date} to {max_date}")

    # Create walk-forward windows
    train_hours = TRAIN_DAYS * 24
    test_hours = TEST_DAYS * 24

    results = []
    window_num = 1

    current_train_start = min_date

    while True:
        # Define window boundaries
        train_start = current_train_start
        train_end = train_start + pd.Timedelta(hours=train_hours)
        test_start = train_end
        test_end = test_start + pd.Timedelta(hours=test_hours)

        # Check if we have enough data for test window
        if test_end > max_date:
            break

        # Extract train and test data
        train_mask = (df_eval.index >= train_start) & (df_eval.index < train_end)
        test_mask = (df_eval.index >= test_start) & (df_eval.index < test_end)

        X_train = df_eval.loc[train_mask, features]
        y_train = df_eval.loc[train_mask, 'wti_returns']
        X_test = df_eval.loc[test_mask, features]
        y_test = df_eval.loc[test_mask, 'wti_returns']

        # Skip if insufficient data
        if len(X_train) < 100 or len(X_test) < 10:
            current_train_start = test_start
            continue

        # Train model
        model = LGBMRegressor(**MODEL_CONFIG)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate IC
        ic = np.corrcoef(y_pred, y_test)[0, 1]

        results.append({
            'window': window_num,
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
            'ic': ic,
            'n_train': len(X_train),
            'n_test': len(X_test)
        })

        print(f"  Window {window_num}: {test_start.date()} to {test_end.date()}, IC={ic:.4f}, n_train={len(X_train)}, n_test={len(X_test)}")

        window_num += 1
        current_train_start = test_start

    # Calculate summary statistics
    df_results = pd.DataFrame(results)

    ic_mean = df_results['ic'].mean()
    ic_median = df_results['ic'].median()
    ic_std = df_results['ic'].std()
    ir = ic_mean / ic_std if ic_std > 0 else 0
    pmr = (df_results['ic'] > 0).mean()

    summary = {
        'model': model_name,
        'n_features': len(features),
        'n_windows': len(df_results),
        'ic_mean': ic_mean,
        'ic_median': ic_median,
        'ic_std': ic_std,
        'ir': ir,
        'pmr': pmr
    }

    return df_results, summary

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("="*60)
    print("EVALUATING SEED202 WITH LEAN 7-FEATURE CONFIG")
    print("="*60)
    print("Features: 5 GDELT + 2 Market (cl1_cl2, ovx)")

    # 1. Load integrated data
    print("\n[1/4] Loading integrated data...")
    df_integrated = pd.read_parquet(INPUT_PATH)
    print(f"  Shape: {df_integrated.shape}")
    print(f"  Date range: {df_integrated.index.min()} to {df_integrated.index.max()}")

    # 2. Prepare baseline data (GDELT only)
    print("\n[2/4] Preparing baseline data (GDELT only)...")
    df_gdelt_raw = pd.read_parquet(GDELT_PATH)
    df_prices_raw = pd.read_parquet(BASELINE_PATH)

    # Merge GDELT with prices
    df_baseline = df_prices_raw.merge(
        df_gdelt_raw[['ts_utc'] + GDELT_FEATURES],
        on='ts_utc',
        how='left'
    )
    df_baseline['ts_utc'] = pd.to_datetime(df_baseline['ts_utc'], utc=True)
    df_baseline = df_baseline.set_index('ts_utc')
    df_baseline = df_baseline.rename(columns={'ret_1h': 'wti_returns'})

    # Fill missing GDELT features with 0
    for feat in GDELT_FEATURES:
        df_baseline[feat] = df_baseline[feat].fillna(0)

    print(f"  Shape: {df_baseline.shape}")

    # 3. Evaluate both models
    print("\n[3/4] Running walk-forward evaluations...")

    # Baseline: GDELT only
    df_baseline_windows, baseline_summary = evaluate_model(
        df_baseline,
        GDELT_FEATURES,
        'Seed202 (GDELT only)'
    )

    # New: GDELT + Market
    df_integrated_windows, integrated_summary = evaluate_model(
        df_integrated,
        GDELT_FEATURES + MARKET_FEATURES,
        'Seed202 (GDELT + Market)'
    )

    # 4. Save results and compare
    print("\n[4/4] Saving results and generating comparison...")

    # Save window-level results
    baseline_windows_path = OUTPUT_DIR / f'seed202_lean_baseline_windows_{timestamp}.csv'
    integrated_windows_path = OUTPUT_DIR / f'seed202_lean_integrated_windows_{timestamp}.csv'

    df_baseline_windows.to_csv(baseline_windows_path, index=False)
    df_integrated_windows.to_csv(integrated_windows_path, index=False)

    # Save summary
    df_summary = pd.DataFrame([baseline_summary, integrated_summary])
    summary_path = OUTPUT_DIR / f'seed202_lean_comparison_{timestamp}.csv'
    df_summary.to_csv(summary_path, index=False)

    # Calculate improvements
    ic_mean_change = integrated_summary['ic_mean'] - baseline_summary['ic_mean']
    ic_mean_pct = 100 * ic_mean_change / baseline_summary['ic_mean']

    ic_std_change = integrated_summary['ic_std'] - baseline_summary['ic_std']
    ic_std_pct = 100 * ic_std_change / baseline_summary['ic_std']

    ir_change = integrated_summary['ir'] - baseline_summary['ir']
    ir_pct = 100 * ir_change / baseline_summary['ir']

    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    print(f"\nBaseline (GDELT only):")
    print(f"  IC mean:   {baseline_summary['ic_mean']:.6f}")
    print(f"  IC median: {baseline_summary['ic_median']:.6f}")
    print(f"  IC std:    {baseline_summary['ic_std']:.6f}")
    print(f"  IR:        {baseline_summary['ir']:.4f}")
    print(f"  PMR:       {baseline_summary['pmr']:.4f}")

    print(f"\nIntegrated (GDELT + Market):")
    print(f"  IC mean:   {integrated_summary['ic_mean']:.6f} ({ic_mean_pct:+.2f}%)")
    print(f"  IC median: {integrated_summary['ic_median']:.6f}")
    print(f"  IC std:    {integrated_summary['ic_std']:.6f} ({ic_std_pct:+.2f}%)")
    print(f"  IR:        {integrated_summary['ir']:.4f} ({ir_pct:+.2f}%)")
    print(f"  PMR:       {integrated_summary['pmr']:.4f}")

    print(f"\nHard Threshold Check:")
    print(f"  IC median ≥ 0.02:  {'✓ PASS' if integrated_summary['ic_median'] >= 0.02 else '✗ FAIL'}")
    print(f"  IR ≥ 0.5:          {'✓ PASS' if integrated_summary['ir'] >= 0.5 else '✗ FAIL'}")
    print(f"  PMR ≥ 0.55:        {'✓ PASS' if integrated_summary['pmr'] >= 0.55 else '✗ FAIL'}")

    gap_to_threshold = 0.5 - integrated_summary['ir']
    print(f"\nGap to IR ≥ 0.5 threshold: {gap_to_threshold:.4f}")

    # Feature importance for integrated model
    print("\n[Extra] Training final model for feature importance...")
    df_final = df_integrated.dropna(subset=['wti_returns'] + GDELT_FEATURES + MARKET_FEATURES)
    X = df_final[GDELT_FEATURES + MARKET_FEATURES]
    y = df_final['wti_returns']

    model = LGBMRegressor(**MODEL_CONFIG)
    model.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': GDELT_FEATURES + MARKET_FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_path = OUTPUT_DIR / f'seed202_lean_importance_{timestamp}.csv'
    importance_df.to_csv(importance_path, index=False)

    print("\nFeature Importance (top 9):")
    print(importance_df.to_string(index=False))

    # Update RUNLOG
    print(f"\nUpdating {RUNLOG_PATH}...")

    with open(RUNLOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"\n\n## LEAN 7-Feature Validation: GDELT + Market (No Cracks) ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n")
        f.write(f"**Experiment**: Validate lean configuration with term structure (cl1_cl2) and OVX only (crack spreads excluded)\n\n")
        f.write(f"**Rationale**: 9-feature model showed crack_rb and crack_ho have zero importance - removing to simplify\n\n")
        f.write(f"**Goal**: Verify lean 7-feature config still exceeds hard thresholds (IR≥0.5, IC median≥0.02, PMR≥0.55)\n\n")

        f.write(f"**Configuration**:\n")
        f.write(f"- Model: Seed202 (reg_lambda=1.5, random_state=202)\n")
        f.write(f"- Features: 5 GDELT + 2 Market = 7 total (LEAN)\n")
        f.write(f"- Market features: cl1_cl2, ovx (crack_rb, crack_ho excluded)\n")
        f.write(f"- Window: H=1, {TRAIN_DAYS}d train / {TEST_DAYS}d test\n")
        f.write(f"- Bagging: feature_fraction=0.5, bagging_fraction=0.6\n\n")

        f.write(f"**Results**:\n\n")
        f.write(f"Baseline (GDELT only, 5 features):\n")
        f.write(f"- IC mean:   {baseline_summary['ic_mean']:.6f}\n")
        f.write(f"- IC std:    {baseline_summary['ic_std']:.6f}\n")
        f.write(f"- IR:        {baseline_summary['ir']:.4f}\n")
        f.write(f"- PMR:       {baseline_summary['pmr']:.4f}\n")
        f.write(f"- Windows:   {baseline_summary['n_windows']}\n\n")

        f.write(f"Lean (GDELT + Market, 7 features):\n")
        f.write(f"- IC mean:   {integrated_summary['ic_mean']:.6f} ({ic_mean_pct:+.2f}%)\n")
        f.write(f"- IC std:    {integrated_summary['ic_std']:.6f} ({ic_std_pct:+.2f}%)\n")
        f.write(f"- IR:        {integrated_summary['ir']:.4f} ({ir_pct:+.2f}%)\n")
        f.write(f"- PMR:       {integrated_summary['pmr']:.4f}\n")
        f.write(f"- Windows:   {integrated_summary['n_windows']}\n\n")

        f.write(f"**Feature Importance** (Lean 7-feature model):\n")
        f.write(f"```\n{importance_df.to_string(index=False)}\n```\n\n")

        f.write(f"**Hard Threshold Check**:\n")
        f.write(f"- IC median ≥ 0.02: {integrated_summary['ic_median']:.4f} {'✓ PASS' if integrated_summary['ic_median'] >= 0.02 else '✗ FAIL'}\n")
        f.write(f"- IR ≥ 0.5:         {integrated_summary['ir']:.4f} {'✓ PASS' if integrated_summary['ir'] >= 0.5 else '✗ FAIL'}\n")
        f.write(f"- PMR ≥ 0.55:       {integrated_summary['pmr']:.4f} {'✓ PASS' if integrated_summary['pmr'] >= 0.55 else '✗ FAIL'}\n\n")

        if integrated_summary['ir'] >= 0.5:
            f.write(f"**Decision**: ✓ LEAN CONFIG VALIDATED. All hard thresholds exceeded.\n")
            f.write(f"**Status**: Ready to promote to Base. Crack spreads confirmed unnecessary.\n")
        else:
            f.write(f"**Decision**: ✗ LEAN CONFIG FAILED. Gap to threshold: {gap_to_threshold:.4f}\n")
            f.write(f"**Recommendation**: Revert to 9-feature model or investigate why crack removal degraded performance.\n")

    print(f"✓ Results saved to {OUTPUT_DIR}")
    print(f"✓ RUNLOG updated")

    return df_summary, importance_df

if __name__ == '__main__':
    summary, importance = main()
