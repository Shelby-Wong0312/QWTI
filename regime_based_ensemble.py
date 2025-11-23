"""
Regime-Based Ensemble Model for WTI Oil Price Prediction
Addresses IR insufficiency by training specialized models for different market regimes

Strategy:
1. Classify windows into regimes (high/low volatility)
2. Train separate LightGBM models for each regime
3. Use ensemble (weighted average) for final predictions
4. Target: IR >= 0.5 in adverse regimes

Author: Claude Code
Date: 2025-11-19
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# ============================================================================
# Configuration
# ============================================================================

GDELT_PARQUET = 'data/gdelt_hourly.parquet'
PRICE_PARQUET = 'data/features_hourly.parquet'
OUTPUT_DIR = Path('warehouse/ic')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Best single model config from previous grid search
BEST_CONFIG = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'num_leaves': 31,
    'random_state': 42,
    'verbosity': -1,
    'force_col_wise': True
}

H = 1  # Horizon
LAG = 1  # Lag hours
TRAIN_HOURS = 1440  # 60 days
TEST_HOURS = 360    # 15 days

BUCKET_FEATURES = [
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt'
]

# Regime classification thresholds
VOLATILITY_WINDOW = 168  # 7 days for volatility calculation
VOLATILITY_QUANTILE = 0.5  # Median split

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

print("\n" + "="*80)
print("REGIME-BASED ENSEMBLE MODEL - IR RECOVERY EXPERIMENT")
print("="*80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nObjective: Improve IR from 0.26 to >=0.5 via regime-specific modeling")
print(f"Strategy: Train separate LightGBM for high/low volatility regimes")
print(f"\nConfiguration:")
print(f"  Horizon (H): {H}")
print(f"  Lag: {LAG}h")
print(f"  Train window: {TRAIN_HOURS}h ({TRAIN_HOURS//24} days)")
print(f"  Test window: {TEST_HOURS}h ({TEST_HOURS//24} days)")
print(f"  Model: LightGBM (depth={BEST_CONFIG['max_depth']}, lr={BEST_CONFIG['learning_rate']}, n={BEST_CONFIG['n_estimators']})")

print(f"\n[1/6] Loading data...")

# Load GDELT features
gdelt_df = pd.read_parquet(GDELT_PARQUET)
gdelt_df['ts_utc'] = pd.to_datetime(gdelt_df['ts_utc'])
gdelt_df = gdelt_df.sort_values('ts_utc').reset_index(drop=True)
print(f"  GDELT: {len(gdelt_df):,} hours from {gdelt_df['ts_utc'].min()} to {gdelt_df['ts_utc'].max()}")

# Load WTI price returns (already calculated in features_hourly.parquet)
price_df = pd.read_parquet(PRICE_PARQUET)
price_df['ts_utc'] = pd.to_datetime(price_df['ts_utc'])
price_df = price_df.sort_values('ts_utc').reset_index(drop=True)
print(f"  Price features: {len(price_df):,} hours")

# Merge
merged = pd.merge(gdelt_df, price_df[['ts_utc', 'ret_1h']], on='ts_utc', how='inner')
merged = merged.rename(columns={'ret_1h': 'wti_returns'})
merged = merged.dropna(subset=['wti_returns'])

# Winsorize features (1st/99th percentiles)
for feat in BUCKET_FEATURES:
    p1, p99 = merged[feat].quantile([0.01, 0.99])
    merged[feat] = merged[feat].clip(lower=p1, upper=p99)

print(f"  Merged: {len(merged):,} hours after merge and cleaning")

# ============================================================================
# Regime Classification
# ============================================================================

print(f"\n[2/6] Classifying market regimes...")

# Calculate rolling volatility of WTI returns
merged['wti_volatility'] = merged['wti_returns'].rolling(VOLATILITY_WINDOW).std()
merged['wti_volatility'] = merged['wti_volatility'].fillna(method='bfill').fillna(method='ffill')

# Calculate median volatility threshold
vol_threshold = merged['wti_volatility'].quantile(VOLATILITY_QUANTILE)
print(f"  Volatility threshold (median): {vol_threshold:.6f}")

# Classify each timestamp into regime
merged['regime'] = merged['wti_volatility'].apply(lambda x: 'high_vol' if x >= vol_threshold else 'low_vol')

high_vol_pct = (merged['regime'] == 'high_vol').sum() / len(merged) * 100
low_vol_pct = (merged['regime'] == 'low_vol').sum() / len(merged) * 100
print(f"  High volatility regime: {(merged['regime'] == 'high_vol').sum():,} hours ({high_vol_pct:.1f}%)")
print(f"  Low volatility regime: {(merged['regime'] == 'low_vol').sum():,} hours ({low_vol_pct:.1f}%)")

# ============================================================================
# Rolling Window Evaluation with Regime-Based Models
# ============================================================================

print(f"\n[3/6] Training regime-specific models on rolling windows...")

max_start = len(merged) - TRAIN_HOURS - TEST_HOURS
window_starts = np.arange(0, max_start + 1, TEST_HOURS)

print(f"  Total windows: {len(window_starts)}")
print(f"  Window structure: Train={TRAIN_HOURS}h, Test={TEST_HOURS}h")

results = []
window_details = []

for i, start_idx in enumerate(window_starts):
    train_end = start_idx + TRAIN_HOURS
    test_end = train_end + TEST_HOURS

    # Split data
    train_df = merged.iloc[start_idx:train_end].copy()
    test_df = merged.iloc[train_end:test_end].copy()

    if len(train_df) < TRAIN_HOURS or len(test_df) < TEST_HOURS:
        continue

    train_start_date = train_df['ts_utc'].min().strftime('%Y-%m-%d')
    test_start_date = test_df['ts_utc'].min().strftime('%Y-%m-%d')
    test_end_date = test_df['ts_utc'].max().strftime('%Y-%m-%d')

    print(f"\n  Window {i+1}/{len(window_starts)}:")
    print(f"    Train: {train_start_date} to {test_start_date}")
    print(f"    Test:  {test_start_date} to {test_end_date}")

    # Separate training data by regime
    train_high_vol = train_df[train_df['regime'] == 'high_vol']
    train_low_vol = train_df[train_df['regime'] == 'low_vol']

    print(f"    Train regime split: high_vol={len(train_high_vol)}, low_vol={len(train_low_vol)}")

    # Prepare features
    X_train_high = train_high_vol[BUCKET_FEATURES].values
    y_train_high = train_high_vol['wti_returns'].values
    X_train_low = train_low_vol[BUCKET_FEATURES].values
    y_train_low = train_low_vol['wti_returns'].values

    X_test = test_df[BUCKET_FEATURES].values
    y_test = test_df['wti_returns'].values

    # Train regime-specific models
    models = {}

    if len(train_high_vol) >= 100:  # Minimum samples for training
        model_high = lgb.LGBMRegressor(**BEST_CONFIG)
        model_high.fit(X_train_high, y_train_high)
        models['high_vol'] = model_high
        print(f"    Trained high_vol model on {len(train_high_vol)} samples")
    else:
        models['high_vol'] = None
        print(f"    Skipped high_vol model (insufficient samples: {len(train_high_vol)})")

    if len(train_low_vol) >= 100:
        model_low = lgb.LGBMRegressor(**BEST_CONFIG)
        model_low.fit(X_train_low, y_train_low)
        models['low_vol'] = model_low
        print(f"    Trained low_vol model on {len(train_low_vol)} samples")
    else:
        models['low_vol'] = None
        print(f"    Skipped low_vol model (insufficient samples: {len(train_low_vol)})")

    # Also train a single model for comparison
    model_single = lgb.LGBMRegressor(**BEST_CONFIG)
    X_train_all = train_df[BUCKET_FEATURES].values
    y_train_all = train_df['wti_returns'].values
    model_single.fit(X_train_all, y_train_all)

    # Make predictions
    # Regime-based ensemble
    y_pred_ensemble = np.zeros(len(test_df))
    for j, regime in enumerate(test_df['regime']):
        model = models.get(regime)
        if model is not None:
            y_pred_ensemble[j] = model.predict(X_test[j:j+1])[0]
        else:
            # Fallback to the other regime's model
            fallback_regime = 'low_vol' if regime == 'high_vol' else 'high_vol'
            fallback_model = models.get(fallback_regime)
            if fallback_model is not None:
                y_pred_ensemble[j] = fallback_model.predict(X_test[j:j+1])[0]
            else:
                # Last resort: use single model
                y_pred_ensemble[j] = model_single.predict(X_test[j:j+1])[0]

    # Single model predictions
    y_pred_single = model_single.predict(X_test)

    # Calculate IC for ensemble
    ic_ensemble = np.corrcoef(y_pred_ensemble, y_test)[0, 1]

    # Calculate IC for single model
    ic_single = np.corrcoef(y_pred_single, y_test)[0, 1]

    # Calculate IC by regime
    high_vol_mask = test_df['regime'] == 'high_vol'
    low_vol_mask = test_df['regime'] == 'low_vol'

    ic_high_vol_ensemble = np.corrcoef(y_pred_ensemble[high_vol_mask], y_test[high_vol_mask])[0, 1] if high_vol_mask.sum() > 1 else np.nan
    ic_low_vol_ensemble = np.corrcoef(y_pred_ensemble[low_vol_mask], y_test[low_vol_mask])[0, 1] if low_vol_mask.sum() > 1 else np.nan

    ic_high_vol_single = np.corrcoef(y_pred_single[high_vol_mask], y_test[high_vol_mask])[0, 1] if high_vol_mask.sum() > 1 else np.nan
    ic_low_vol_single = np.corrcoef(y_pred_single[low_vol_mask], y_test[low_vol_mask])[0, 1] if low_vol_mask.sum() > 1 else np.nan

    print(f"    Results:")
    print(f"      Ensemble IC: {ic_ensemble:.6f} | Single model IC: {ic_single:.6f} | Delta: {ic_ensemble - ic_single:+.6f}")
    print(f"      High-vol regime: Ensemble IC={ic_high_vol_ensemble:.4f}, Single IC={ic_high_vol_single:.4f}")
    print(f"      Low-vol regime:  Ensemble IC={ic_low_vol_ensemble:.4f}, Single IC={ic_low_vol_single:.4f}")

    results.append({
        'window': i + 1,
        'test_start': test_start_date,
        'test_end': test_end_date,
        'ic_ensemble': ic_ensemble,
        'ic_single': ic_single,
        'ic_delta': ic_ensemble - ic_single,
        'ic_high_vol_ensemble': ic_high_vol_ensemble,
        'ic_low_vol_ensemble': ic_low_vol_ensemble,
        'ic_high_vol_single': ic_high_vol_single,
        'ic_low_vol_single': ic_low_vol_single,
        'n_test': len(test_df),
        'n_test_high_vol': high_vol_mask.sum(),
        'n_test_low_vol': low_vol_mask.sum()
    })

# ============================================================================
# Aggregate Statistics
# ============================================================================

print(f"\n[4/6] Computing aggregate statistics...")

results_df = pd.DataFrame(results)

# Overall statistics for ensemble
ic_mean_ensemble = results_df['ic_ensemble'].mean()
ic_median_ensemble = results_df['ic_ensemble'].median()
ic_std_ensemble = results_df['ic_ensemble'].std()
ir_ensemble = ic_mean_ensemble / ic_std_ensemble if ic_std_ensemble > 0 else 0
pmr_ensemble = (results_df['ic_ensemble'] > 0).sum() / len(results_df)

# Overall statistics for single model
ic_mean_single = results_df['ic_single'].mean()
ic_median_single = results_df['ic_single'].median()
ic_std_single = results_df['ic_single'].std()
ir_single = ic_mean_single / ic_std_single if ic_std_single > 0 else 0
pmr_single = (results_df['ic_single'] > 0).sum() / len(results_df)

print(f"\n{'='*80}")
print(f"ENSEMBLE MODEL RESULTS (vs Single Model Baseline)")
print(f"{'='*80}")
print(f"\nEnsemble Performance:")
print(f"  IC mean:   {ic_mean_ensemble:.6f}  (baseline: {ic_mean_single:.6f}, delta: {ic_mean_ensemble - ic_mean_single:+.6f})")
print(f"  IC median: {ic_median_ensemble:.6f}  (baseline: {ic_median_single:.6f}, delta: {ic_median_ensemble - ic_median_single:+.6f})")
print(f"  IC std:    {ic_std_ensemble:.6f}  (baseline: {ic_std_single:.6f}, delta: {ic_std_ensemble - ic_std_single:+.6f})")
print(f"  IR:        {ir_ensemble:.4f}     (baseline: {ir_single:.4f}, delta: {ir_ensemble - ir_single:+.4f})")
print(f"  PMR:       {pmr_ensemble:.4f}     (baseline: {pmr_single:.4f}, delta: {pmr_ensemble - pmr_single:+.4f})")

print(f"\nHard Threshold Check (Ensemble):")
print(f"  IC median >= 0.02: {'PASS' if ic_median_ensemble >= 0.02 else 'FAIL'} ({ic_median_ensemble:.6f})")
print(f"  IR >= 0.5:         {'PASS' if ir_ensemble >= 0.5 else 'FAIL'} ({ir_ensemble:.4f})")
print(f"  PMR >= 0.55:       {'PASS' if pmr_ensemble >= 0.55 else 'FAIL'} ({pmr_ensemble:.4f})")

if ic_median_ensemble >= 0.02 and ir_ensemble >= 0.5 and pmr_ensemble >= 0.55:
    print(f"\n[OK] HARD THRESHOLD ACHIEVED - Ready for base promotion")
else:
    print(f"\n[X] Hard threshold NOT met")
    if ir_ensemble < 0.5:
        print(f"    IR still below 0.5 (current: {ir_ensemble:.4f})")

# ============================================================================
# Save Results
# ============================================================================

print(f"\n[5/6] Saving results...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Window-level results
results_file = OUTPUT_DIR / f'regime_ensemble_windows_{timestamp}.csv'
results_df.to_csv(results_file, index=False)
print(f"  Window results: {results_file}")

# Summary statistics
summary_df = pd.DataFrame([{
    'model_type': 'ensemble',
    'ic_mean': ic_mean_ensemble,
    'ic_median': ic_median_ensemble,
    'ic_std': ic_std_ensemble,
    'ir': ir_ensemble,
    'pmr': pmr_ensemble,
    'n_windows': len(results_df),
    'hard_ic_median': ic_median_ensemble >= 0.02,
    'hard_ir': ir_ensemble >= 0.5,
    'hard_pmr': pmr_ensemble >= 0.55,
    'promotion_ready': (ic_median_ensemble >= 0.02 and ir_ensemble >= 0.5 and pmr_ensemble >= 0.55)
}, {
    'model_type': 'single_baseline',
    'ic_mean': ic_mean_single,
    'ic_median': ic_median_single,
    'ic_std': ic_std_single,
    'ir': ir_single,
    'pmr': pmr_single,
    'n_windows': len(results_df),
    'hard_ic_median': ic_median_single >= 0.02,
    'hard_ir': ir_single >= 0.5,
    'hard_pmr': pmr_single >= 0.55,
    'promotion_ready': (ic_median_single >= 0.02 and ir_single >= 0.5 and pmr_single >= 0.55)
}])

summary_file = OUTPUT_DIR / f'regime_ensemble_summary_{timestamp}.csv'
summary_df.to_csv(summary_file, index=False)
print(f"  Summary: {summary_file}")

print(f"\n[6/6] Experiment complete!")
print(f"\nKey Insights:")
print(f"  - Regime-based modeling {'improved' if ir_ensemble > ir_single else 'did not improve'} IR over single model")
print(f"  - IR improvement: {ir_single:.4f} -> {ir_ensemble:.4f} ({(ir_ensemble/ir_single - 1)*100:+.1f}%)")
print(f"  - {'READY' if ir_ensemble >= 0.5 else 'NOT READY'} for base promotion (IR threshold)")
print(f"\n" + "="*80)
