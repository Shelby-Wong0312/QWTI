"""
Stacking Ensemble (3xLightGBM) Stability Validation
Compares against approved base_seed202_lean7_h1 configuration

Strategy:
1. Train 3 LightGBM base learners with different seeds and regularization
2. Use Ridge meta-model for stacking
3. Same lag=1, H=1 as base configuration
4. Same gap tolerance for market closures (weekends, holidays)

Author: Claude Code
Date: 2025-12-03
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import lightgbm as lgb
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

# ============================================================================
# Configuration
# ============================================================================

GDELT_PARQUET = 'data/gdelt_hourly.parquet'
PRICE_PARQUET = 'data/features_hourly.parquet'
OUTPUT_DIR = Path('warehouse/ic')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3 Base Learners with different seeds and regularization
BASE_LEARNERS = {
    'lgb_seed42': {
        'name': 'LGB Seed42 (L=0.5)',
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'num_leaves': 31,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'reg_lambda': 0.5,
        'random_state': 42,
        'verbosity': -1,
        'force_col_wise': True
    },
    'lgb_seed202': {
        'name': 'LGB Seed202 (L=1.0)',
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'num_leaves': 31,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'reg_lambda': 1.0,
        'random_state': 202,
        'verbosity': -1,
        'force_col_wise': True
    },
    'lgb_seed303': {
        'name': 'LGB Seed303 (L=2.0)',
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'num_leaves': 31,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'reg_lambda': 2.0,
        'random_state': 303,
        'verbosity': -1,
        'force_col_wise': True
    }
}

# Same parameters as base_seed202_lean7_h1
H = 1  # Horizon
LAG = 1  # Lag hours (No-Drift compliance)
TRAIN_HOURS = 1440  # 60 days
TEST_HOURS = 360    # 15 days

# Hard thresholds (same as base)
HARD_IC_MIN = 0.02
HARD_IR_MIN = 0.5
HARD_PMR_MIN = 0.55

BUCKET_FEATURES = [
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt'
]

# Ridge alpha for meta-model
RIDGE_ALPHA = 1.0

# ============================================================================
# Header
# ============================================================================

print("="*80)
print("Stacking Ensemble (3xLightGBM) Stability Validation")
print("="*80)
print(f"Config: 3 LightGBM base learners + Ridge meta-model")
print(f"H={H}, lag={LAG}h (No-Drift compliance)")
print(f"Train: {TRAIN_HOURS}h ({TRAIN_HOURS//24}d), Test: {TEST_HOURS}h ({TEST_HOURS//24}d)")
print(f"Hard IC thresholds: IC>={HARD_IC_MIN}, IR>={HARD_IR_MIN}, PMR>={HARD_PMR_MIN}")
print("="*80)

# ============================================================================
# Data Loading
# ============================================================================

print("\n[1/8] Loading data...")

gdelt_df = pd.read_parquet(GDELT_PARQUET)
gdelt_df['ts_utc'] = pd.to_datetime(gdelt_df['ts_utc'])
gdelt_df = gdelt_df.sort_values('ts_utc').reset_index(drop=True)
print(f"   GDELT: {len(gdelt_df):,} hours")

price_df = pd.read_parquet(PRICE_PARQUET)
price_df['ts_utc'] = pd.to_datetime(price_df['ts_utc'])
price_df = price_df.sort_values('ts_utc').reset_index(drop=True)
print(f"   Price: {len(price_df):,} hours")

merged = pd.merge(gdelt_df, price_df[['ts_utc', 'ret_1h']], on='ts_utc', how='inner')
merged = merged.rename(columns={'ret_1h': 'wti_returns'})
merged = merged.dropna(subset=['wti_returns'])
print(f"   Merged: {len(merged):,} hours")

# ============================================================================
# Find Longest Continuous Segment (same logic as stability_validation)
# ============================================================================

print("\n[2/8] Finding longest continuous segment...")

merged = merged.sort_values('ts_utc').reset_index(drop=True)
merged['time_diff'] = merged['ts_utc'].diff().dt.total_seconds() / 3600

# Allow regular market closures (same as stability_validation_best_lightgbm.py)
# - Daily halt: 2h (20:00-22:00 UTC)
# - Weekend: ~50h (Fri 20:00 ~ Sun 22:00 UTC)
# - Holidays: up to ~75h
# Only flag truly abnormal gaps > 80h
merged['is_gap'] = merged['time_diff'] > 80

segment_ids = merged['is_gap'].cumsum()
segment_lengths = segment_ids.value_counts().sort_index()
longest_segment_id = segment_lengths.idxmax()
segment_data = merged[segment_ids == longest_segment_id].copy()

print(f"   Longest segment: {len(segment_data)} hours")
print(f"   Date range: {segment_data['ts_utc'].min()} to {segment_data['ts_utc'].max()}")

# ============================================================================
# Winsorization
# ============================================================================

print("\n[3/8] Applying winsorization (1st/99th percentiles)...")

for col in BUCKET_FEATURES + ['wti_returns']:
    if col in segment_data.columns:
        p1 = segment_data[col].quantile(0.01)
        p99 = segment_data[col].quantile(0.99)
        segment_data[col] = segment_data[col].clip(p1, p99)

# ============================================================================
# Rolling Window Evaluation with Stacking Ensemble
# ============================================================================

print(f"\n[4/8] Running rolling window evaluation...")

X = segment_data[BUCKET_FEATURES].values
y_forward = segment_data['wti_returns'].shift(-H).values

# Remove NaN rows
valid_idx = ~np.isnan(y_forward)
X_valid = X[valid_idx]
y_valid = y_forward[valid_idx]
ts_valid = segment_data['ts_utc'].values[valid_idx]

if len(X_valid) < TRAIN_HOURS + TEST_HOURS:
    print(f"   ERROR: Insufficient data ({len(X_valid)} < {TRAIN_HOURS + TEST_HOURS})")
    exit(1)

# Calculate window starts
n_samples = len(X_valid)
max_start = n_samples - TRAIN_HOURS - TEST_HOURS
window_starts = np.arange(0, max_start + 1, TEST_HOURS)

print(f"   Total samples: {n_samples}")
print(f"   Max windows: {len(window_starts)}")
print(f"   Window step: {TEST_HOURS}h (non-overlapping)")

window_results = []

for w_idx, start in enumerate(window_starts):
    train_end = start + TRAIN_HOURS
    test_end = train_end + TEST_HOURS

    X_train = X_valid[start:train_end]
    y_train = y_valid[start:train_end]
    X_test = X_valid[train_end:test_end]
    y_test = y_valid[train_end:test_end]

    test_month = pd.Timestamp(ts_valid[train_end]).strftime('%Y-%m')

    # Split training for meta-model
    train_split = int(len(X_train) * 0.8)
    X_train_base = X_train[:train_split]
    y_train_base = y_train[:train_split]
    X_val = X_train[train_split:]
    y_val = y_train[train_split:]

    # Train 3 base learners
    base_preds_val = {}
    base_preds_test = {}
    base_ics = {}

    for name, config in BASE_LEARNERS.items():
        cfg = {k: v for k, v in config.items() if k != 'name'}
        model = lgb.LGBMRegressor(**cfg)
        model.fit(X_train_base, y_train_base)

        base_preds_val[name] = model.predict(X_val)
        base_preds_test[name] = model.predict(X_test)

        # Individual IC
        ic, _ = spearmanr(base_preds_test[name], y_test)
        base_ics[name] = ic

    # Stacking: Ridge meta-model
    meta_X_train = np.column_stack([base_preds_val[k] for k in BASE_LEARNERS.keys()])
    meta_X_test = np.column_stack([base_preds_test[k] for k in BASE_LEARNERS.keys()])

    meta_model = Ridge(alpha=RIDGE_ALPHA)
    meta_model.fit(meta_X_train, y_val)

    stacking_pred = meta_model.predict(meta_X_test)
    stacking_ic, _ = spearmanr(stacking_pred, y_test)

    # Simple average ensemble
    simple_avg_pred = np.mean([base_preds_test[k] for k in BASE_LEARNERS.keys()], axis=0)
    simple_avg_ic, _ = spearmanr(simple_avg_pred, y_test)

    # Check if IC passes threshold
    ic_pass = "OK" if stacking_ic >= HARD_IC_MIN else "X"

    print(f"   Window {w_idx+1}/{len(window_starts)}: month={test_month}, "
          f"Stack IC={stacking_ic:.6f}, Avg IC={simple_avg_ic:.6f}, {ic_pass}")

    window_results.append({
        'window': w_idx + 1,
        'test_month': test_month,
        'test_start': pd.Timestamp(ts_valid[train_end]).isoformat(),
        'test_end': pd.Timestamp(ts_valid[test_end-1]).isoformat(),
        'ic_stacking': stacking_ic,
        'ic_simple_avg': simple_avg_ic,
        'ic_lgb_seed42': base_ics['lgb_seed42'],
        'ic_lgb_seed202': base_ics['lgb_seed202'],
        'ic_lgb_seed303': base_ics['lgb_seed303'],
        'n_test': len(y_test),
        'ridge_coef_seed42': meta_model.coef_[0],
        'ridge_coef_seed202': meta_model.coef_[1],
        'ridge_coef_seed303': meta_model.coef_[2],
    })

# ============================================================================
# Analyze Results
# ============================================================================

print(f"\n[5/8] Analyzing stability across {len(window_results)} windows...")

results_df = pd.DataFrame(window_results)

# Stacking ensemble stats
ic_stacking = results_df['ic_stacking']
ic_mean = ic_stacking.mean()
ic_median = ic_stacking.median()
ic_std = ic_stacking.std()
ir = ic_mean / ic_std if ic_std > 0 else 0
pmr = (ic_stacking > 0).sum() / len(ic_stacking)

print(f"\n   Stacking Ensemble Performance:")
print(f"     IC mean:   {ic_mean:.6f}")
print(f"     IC median: {ic_median:.6f} {'OK' if ic_median >= HARD_IC_MIN else 'FAIL'}")
print(f"     IC std:    {ic_std:.6f}")
print(f"     IR:        {ir:.2f} {'OK' if ir >= HARD_IR_MIN else 'FAIL'}")
print(f"     PMR:       {pmr:.2f} {'OK' if pmr >= HARD_PMR_MIN else 'FAIL'}")
print(f"     Hard threshold: {ic_median >= HARD_IC_MIN and ir >= HARD_IR_MIN and pmr >= HARD_PMR_MIN}")

# Simple average stats
ic_avg = results_df['ic_simple_avg']
ic_avg_mean = ic_avg.mean()
ic_avg_median = ic_avg.median()
ic_avg_std = ic_avg.std()
ir_avg = ic_avg_mean / ic_avg_std if ic_avg_std > 0 else 0
pmr_avg = (ic_avg > 0).sum() / len(ic_avg)

print(f"\n   Simple Average Ensemble Performance:")
print(f"     IC mean:   {ic_avg_mean:.6f}")
print(f"     IC median: {ic_avg_median:.6f}")
print(f"     IC std:    {ic_avg_std:.6f}")
print(f"     IR:        {ir_avg:.2f}")
print(f"     PMR:       {pmr_avg:.2f}")

# Individual base learner stats
print(f"\n   Individual Base Learner Performance:")
for name in BASE_LEARNERS.keys():
    ic_col = results_df[f'ic_{name}']
    print(f"     {name}: IC mean={ic_col.mean():.6f}, IR={ic_col.mean()/ic_col.std():.2f}")

# Consecutive positive windows
consecutive_positive = 0
max_consecutive = 0
for ic in ic_stacking:
    if ic > 0:
        consecutive_positive += 1
        max_consecutive = max(max_consecutive, consecutive_positive)
    else:
        consecutive_positive = 0

# Consecutive hard-meeting windows
consecutive_hard = 0
max_consecutive_hard = 0
for ic in ic_stacking:
    if ic >= HARD_IC_MIN:
        consecutive_hard += 1
        max_consecutive_hard = max(max_consecutive_hard, consecutive_hard)
    else:
        consecutive_hard = 0

print(f"\n   Stability Metrics:")
print(f"     IC min:    {ic_stacking.min():.6f}")
print(f"     IC max:    {ic_stacking.max():.6f}")
print(f"     IC range:  {ic_stacking.max() - ic_stacking.min():.6f}")
print(f"     Max consecutive positive windows: {max_consecutive}")
print(f"     Consecutive hard-meeting windows: {max_consecutive_hard}")

# ============================================================================
# Monthly Analysis
# ============================================================================

print(f"\n[6/8] Analyzing by month...")

monthly = results_df.groupby('test_month').agg({
    'ic_stacking': ['mean', 'median', 'std', 'count']
}).round(6)
monthly.columns = ['IC_mean', 'IC_median', 'IC_std', 'n_windows']
monthly['PMR'] = results_df.groupby('test_month').apply(
    lambda x: (x['ic_stacking'] > 0).sum() / len(x)
)

print(f"\n   Monthly Performance:")
print(monthly.to_string())

best_month = monthly['IC_median'].idxmax()
worst_month = monthly['IC_median'].idxmin()
print(f"\n   Best month:  {best_month} (IC median={monthly.loc[best_month, 'IC_median']:.6f})")
print(f"   Worst month: {worst_month} (IC median={monthly.loc[worst_month, 'IC_median']:.6f})")

# ============================================================================
# Comparison with base_seed202_lean7_h1
# ============================================================================

print(f"\n[7/8] Comparing with base_seed202_lean7_h1...")

# Base configuration results (from stability_validation)
BASE_IC_MEDIAN = 0.050117
BASE_IR = 1.07
BASE_PMR = 0.86

print(f"\n   Comparison:")
print(f"   {'Metric':<12} {'base_seed202_lean7_h1':>20} {'Stacking 3xLGB':>20} {'Diff':>12}")
print(f"   {'-'*12} {'-'*20} {'-'*20} {'-'*12}")
print(f"   {'IC median':<12} {BASE_IC_MEDIAN:>20.6f} {ic_median:>20.6f} {ic_median - BASE_IC_MEDIAN:>+12.6f}")
print(f"   {'IR':<12} {BASE_IR:>20.2f} {ir:>20.2f} {ir - BASE_IR:>+12.2f}")
print(f"   {'PMR':<12} {BASE_PMR:>20.2f} {pmr:>20.2f} {pmr - BASE_PMR:>+12.2f}")

stacking_better = (ic_median > BASE_IC_MEDIAN and ir > BASE_IR)

if stacking_better:
    print(f"\n   [RESULT] Stacking 3xLGB OUTPERFORMS base_seed202_lean7_h1")
else:
    print(f"\n   [RESULT] Stacking 3xLGB does NOT outperform base_seed202_lean7_h1")

# ============================================================================
# Save Results
# ============================================================================

print(f"\n[8/8] Saving results...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Window results
windows_file = OUTPUT_DIR / f'stacking_3lgb_windows_{timestamp}.csv'
results_df.to_csv(windows_file, index=False)
print(f"   Window results: {windows_file}")

# Summary
summary = {
    'model': 'Stacking_3xLightGBM',
    'timestamp': timestamp,
    'n_windows': len(window_results),
    'ic_mean': ic_mean,
    'ic_median': ic_median,
    'ic_std': ic_std,
    'ir': ir,
    'pmr': pmr,
    'hard_ic_pass': ic_median >= HARD_IC_MIN,
    'hard_ir_pass': ir >= HARD_IR_MIN,
    'hard_pmr_pass': pmr >= HARD_PMR_MIN,
    'all_hard_pass': ic_median >= HARD_IC_MIN and ir >= HARD_IR_MIN and pmr >= HARD_PMR_MIN,
    'max_consecutive_positive': max_consecutive,
    'max_consecutive_hard': max_consecutive_hard,
    'best_month': best_month,
    'worst_month': worst_month,
    'vs_base_ic_diff': ic_median - BASE_IC_MEDIAN,
    'vs_base_ir_diff': ir - BASE_IR,
    'outperforms_base': stacking_better,
    'ridge_alpha': RIDGE_ALPHA,
}

summary_file = OUTPUT_DIR / f'stacking_3lgb_summary_{timestamp}.csv'
pd.DataFrame([summary]).to_csv(summary_file, index=False)
print(f"   Summary: {summary_file}")

# Monthly breakdown
monthly_file = OUTPUT_DIR / f'stacking_3lgb_monthly_{timestamp}.csv'
monthly.to_csv(monthly_file)
print(f"   Monthly: {monthly_file}")

# ============================================================================
# Final Report
# ============================================================================

print("\n" + "="*80)
print("STACKING ENSEMBLE (3xLightGBM) VALIDATION REPORT")
print("="*80)

print(f"\nConfiguration:")
print(f"  Base learners: 3 LightGBM (seeds: 42, 202, 303)")
print(f"  Meta-model: Ridge (alpha={RIDGE_ALPHA})")
print(f"  H={H}, lag={LAG}h")

print(f"\nPerformance (n={len(window_results)} windows):")
print(f"  IC:  {ic_mean:.6f} +/- {ic_std:.6f} (median={ic_median:.6f})")
print(f"  IR:  {ir:.2f}")
print(f"  PMR: {pmr:.2f}")

print(f"\nHard Gate Check:")
print(f"  IC median >= {HARD_IC_MIN}: {'PASS' if ic_median >= HARD_IC_MIN else 'FAIL'} ({ic_median:.6f})")
print(f"  IR >= {HARD_IR_MIN}: {'PASS' if ir >= HARD_IR_MIN else 'FAIL'} ({ir:.2f})")
print(f"  PMR >= {HARD_PMR_MIN}: {'PASS' if pmr >= HARD_PMR_MIN else 'FAIL'} ({pmr:.2f})")

print(f"\nComparison with base_seed202_lean7_h1:")
print(f"  IC median: {ic_median:.6f} vs {BASE_IC_MEDIAN:.6f} ({ic_median - BASE_IC_MEDIAN:+.6f})")
print(f"  IR: {ir:.2f} vs {BASE_IR:.2f} ({ir - BASE_IR:+.2f})")
print(f"  PMR: {pmr:.2f} vs {BASE_PMR:.2f} ({pmr - BASE_PMR:+.2f})")

if stacking_better:
    print(f"\n[CONCLUSION] Stacking 3xLGB OUTPERFORMS base - consider for promotion")
else:
    print(f"\n[CONCLUSION] Keep base_seed202_lean7_h1 as selected source")

print("\n" + "="*80)
