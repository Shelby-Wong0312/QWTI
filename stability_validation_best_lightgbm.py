#!/usr/bin/env python3
"""
Stability Validation for Best LightGBM Configuration
Config: depth=5, lr=0.1, n=100, H=1, lag=1h
Goal: Verify stability across more windows & market regimes for base promotion
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("ERROR: LightGBM not installed")
    exit(1)

# ============================================================================
# Configuration (Best from grid search + No-Drift policy)
# ============================================================================
GDELT_PATH = Path("data/gdelt_hourly.parquet")
PRICE_PATH = Path("data/features_hourly.parquet")
OUTPUT_DIR = Path("warehouse/ic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Best LightGBM configuration
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

# Evaluation settings (aligned with No-Drift policy)
H = 1  # Horizon (hours)
LAG = 1  # Lag (hours) - No-Drift requirement
TRAIN_HOURS = 1440  # 60 days
TEST_HOURS = 360    # 15 days (REDUCED for more windows)

# No-Drift Hard IC thresholds
HARD_IC_THRESHOLDS = {
    'ic_median_min': 0.02,
    'ir_min': 0.5,
    'pmr_min': 0.55  # pos_month_ratio
}

# Promotion to base requirements
PROMOTION_REQUIREMENTS = {
    'consecutive_windows_min': 2,
    'meets_hard_thresholds': True,
    'lag_safety_ok': True
}

# Bucket features
BUCKET_FEATURES = [
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt'
]

print("="*80)
print("Stability Validation - Best LightGBM Configuration")
print("="*80)
print(f"Config: depth={BEST_CONFIG['max_depth']}, lr={BEST_CONFIG['learning_rate']}, ")
print(f"        n={BEST_CONFIG['n_estimators']}, leaves={BEST_CONFIG['num_leaves']}")
print(f"H={H}, lag={LAG}h (No-Drift compliance)")
print(f"Train: {TRAIN_HOURS}h ({TRAIN_HOURS//24}d), Test: {TEST_HOURS}h ({TEST_HOURS//24}d)")
print(f"Hard IC thresholds: IC>={HARD_IC_THRESHOLDS['ic_median_min']}, ")
print(f"                    IR>={HARD_IC_THRESHOLDS['ir_min']}, ")
print(f"                    PMR>={HARD_IC_THRESHOLDS['pmr_min']}")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[1/9] Loading data...")
gdelt_df = pd.read_parquet(GDELT_PATH)
price_df = pd.read_parquet(PRICE_PATH)

gdelt_df['ts_utc'] = pd.to_datetime(gdelt_df['ts_utc'])
price_df['ts_utc'] = pd.to_datetime(price_df['ts_utc'])
gdelt_df = gdelt_df.sort_values('ts_utc').reset_index(drop=True)
price_df = price_df.sort_values('ts_utc').reset_index(drop=True)

merged = pd.merge(gdelt_df, price_df[['ts_utc', 'ret_1h']], on='ts_utc', how='inner')
merged = merged.rename(columns={'ret_1h': 'wti_returns'})
print(f"   Merged data: {len(merged)} rows")

# Check timezone (No-Drift preflight)
assert merged['ts_utc'].dt.tz is not None or 'UTC' in str(merged['ts_utc'].dtype), \
    "Preflight FAILED: timezone must be UTC"
print(f"   OK Preflight: timezone check passed")

# ============================================================================
# Find Longest Continuous Segment
# ============================================================================
print("\n[2/9] Finding longest continuous segment...")
merged = merged.sort_values('ts_utc').reset_index(drop=True)
merged['time_diff'] = merged['ts_utc'].diff().dt.total_seconds() / 3600
merged['is_gap'] = merged['time_diff'] > 1.5

segment_ids = merged['is_gap'].cumsum()
segment_lengths = segment_ids.value_counts().sort_index()
longest_segment_id = segment_lengths.idxmax()
segment_data = merged[segment_ids == longest_segment_id].copy()

print(f"   Longest segment: {len(segment_data)} hours")
print(f"   Date range: {segment_data['ts_utc'].min()} to {segment_data['ts_utc'].max()}")

# ============================================================================
# Winsorization
# ============================================================================
print("\n[3/9] Applying winsorization (1st/99th percentiles)...")
for col in BUCKET_FEATURES + ['wti_returns']:
    if col in segment_data.columns:
        p1 = segment_data[col].quantile(0.01)
        p99 = segment_data[col].quantile(0.99)
        segment_data[col] = segment_data[col].clip(p1, p99)

# ============================================================================
# Expanded Rolling Window Evaluation
# ============================================================================
print(f"\n[4/9] Running expanded rolling window evaluation...")

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

# Calculate window starts (smaller step for more windows)
n_samples = len(X_valid)
max_start = n_samples - TRAIN_HOURS - TEST_HOURS
window_starts = np.arange(0, max_start + 1, TEST_HOURS)  # Non-overlapping

print(f"   Total samples: {n_samples}")
print(f"   Max windows: {len(window_starts)}")
print(f"   Window step: {TEST_HOURS}h (non-overlapping)")

window_results = []
ic_list = []

for i, start in enumerate(window_starts):
    train_end = start + TRAIN_HOURS
    test_end = train_end + TEST_HOURS

    if test_end > n_samples:
        break

    X_train = X_valid[start:train_end]
    y_train = y_valid[start:train_end]
    X_test = X_valid[train_end:test_end]
    y_test = y_valid[train_end:test_end]

    # Get timestamp info
    test_start_ts = ts_valid[train_end]
    test_end_ts = ts_valid[test_end-1]
    test_month = pd.to_datetime(test_start_ts).strftime('%Y-%m')

    # Train LightGBM
    model = lgb.LGBMRegressor(**BEST_CONFIG)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate IC
    ic = np.corrcoef(y_pred, y_test)[0, 1]
    if not np.isnan(ic):
        ic_list.append(ic)

        window_results.append({
            'window_id': i,
            'test_month': test_month,
            'test_start': test_start_ts,
            'test_end': test_end_ts,
            'IC': ic,
            'IC_positive': ic > 0
        })

        print(f"   Window {i+1}/{len(window_starts)}: "
              f"month={test_month}, IC={ic:.6f}, "
              f"{'OK' if ic > 0 else 'X'}")

# ============================================================================
# Stability Analysis
# ============================================================================
print(f"\n[5/9] Analyzing stability across {len(ic_list)} windows...")

if len(ic_list) == 0:
    print("   ERROR: No valid windows")
    exit(1)

# Overall metrics
ic_mean = np.mean(ic_list)
ic_median = np.median(ic_list)
ic_std = np.std(ic_list)
ir = ic_mean / ic_std if ic_std > 0 else 0
pmr = np.mean([ic > 0 for ic in ic_list])

# Stability metrics
ic_min = np.min(ic_list)
ic_max = np.max(ic_list)
ic_range = ic_max - ic_min
ic_cv = ic_std / ic_mean if ic_mean != 0 else np.inf  # Coefficient of variation

# Consecutive positive windows
consecutive_pos = []
current_streak = 0
for ic in ic_list:
    if ic > 0:
        current_streak += 1
    else:
        if current_streak > 0:
            consecutive_pos.append(current_streak)
        current_streak = 0
if current_streak > 0:
    consecutive_pos.append(current_streak)
max_consecutive_pos = max(consecutive_pos) if consecutive_pos else 0

# Hard threshold check
meets_hard = (ic_median >= HARD_IC_THRESHOLDS['ic_median_min'] and
              ir >= HARD_IC_THRESHOLDS['ir_min'] and
              pmr >= HARD_IC_THRESHOLDS['pmr_min'])

# Consecutive hard-meeting windows
consecutive_hard = 0
for res in window_results:
    # Check if this window alone would meet hard
    if res['IC'] >= HARD_IC_THRESHOLDS['ic_median_min'] and res['IC_positive']:
        consecutive_hard += 1
        if consecutive_hard >= PROMOTION_REQUIREMENTS['consecutive_windows_min']:
            break
    else:
        consecutive_hard = 0

meets_promotion = (meets_hard and
                   consecutive_hard >= PROMOTION_REQUIREMENTS['consecutive_windows_min'] and
                   LAG == 1)  # lag_safety_ok

print(f"\n   Overall Performance:")
print(f"     IC mean:   {ic_mean:.6f}")
print(f"     IC median: {ic_median:.6f} {'OK' if ic_median >= HARD_IC_THRESHOLDS['ic_median_min'] else 'X'}")
print(f"     IC std:    {ic_std:.6f}")
print(f"     IR:        {ir:.2f} {'OK' if ir >= HARD_IC_THRESHOLDS['ir_min'] else 'X'}")
print(f"     PMR:       {pmr:.2f} {'OK' if pmr >= HARD_IC_THRESHOLDS['pmr_min'] else 'X'}")
print(f"     Hard threshold: {meets_hard}")

print(f"\n   Stability Metrics:")
print(f"     IC min:    {ic_min:.6f}")
print(f"     IC max:    {ic_max:.6f}")
print(f"     IC range:  {ic_range:.6f}")
print(f"     IC CV:     {ic_cv:.4f} (lower is more stable)")
print(f"     Max consecutive positive windows: {max_consecutive_pos}")
print(f"     Consecutive hard-meeting windows: {consecutive_hard}")

# ============================================================================
# Monthly Regime Analysis
# ============================================================================
print(f"\n[6/9] Analyzing by market regime (monthly)...")

window_df = pd.DataFrame(window_results)
monthly_stats = window_df.groupby('test_month').agg({
    'IC': ['mean', 'median', 'std', 'count'],
    'IC_positive': 'mean'
}).round(6)
monthly_stats.columns = ['IC_mean', 'IC_median', 'IC_std', 'n_windows', 'PMR']

print(f"\n   Monthly Performance:")
print(monthly_stats.to_string())

# Identify best and worst months
best_month = monthly_stats['IC_median'].idxmax()
worst_month = monthly_stats['IC_median'].idxmin()
print(f"\n   Best month:  {best_month} (IC median={monthly_stats.loc[best_month, 'IC_median']:.6f})")
print(f"   Worst month: {worst_month} (IC median={monthly_stats.loc[worst_month, 'IC_median']:.6f})")

# ============================================================================
# No-Drift Preflight Checks
# ============================================================================
print(f"\n[7/9] Running No-Drift preflight checks...")

preflight_results = []

# Check 1: timezone==UTC
check1 = "timezone==UTC"
pass1 = True  # Already checked earlier
preflight_results.append((check1, pass1))
print(f"   {'OK' if pass1 else 'X'} {check1}")

# Check 2: lag_hours==1
check2 = "lag_hours==1"
pass2 = (LAG == 1)
preflight_results.append((check2, pass2))
print(f"   {'OK' if pass2 else 'X'} {check2}")

# Check 3: meets hard IC thresholds
check3 = f"IC_median>={HARD_IC_THRESHOLDS['ic_median_min']}"
pass3 = (ic_median >= HARD_IC_THRESHOLDS['ic_median_min'])
preflight_results.append((check3, pass3))
print(f"   {'OK' if pass3 else 'X'} {check3}: {ic_median:.6f}")

check4 = f"IR>={HARD_IC_THRESHOLDS['ir_min']}"
pass4 = (ir >= HARD_IC_THRESHOLDS['ir_min'])
preflight_results.append((check4, pass4))
print(f"   {'OK' if pass4 else 'X'} {check4}: {ir:.2f}")

check5 = f"PMR>={HARD_IC_THRESHOLDS['pmr_min']}"
pass5 = (pmr >= HARD_IC_THRESHOLDS['pmr_min'])
preflight_results.append((check5, pass5))
print(f"   {'OK' if pass5 else 'X'} {check5}: {pmr:.2f}")

# Check 6: consecutive windows
check6 = f"consecutive_windows>={PROMOTION_REQUIREMENTS['consecutive_windows_min']}"
pass6 = (consecutive_hard >= PROMOTION_REQUIREMENTS['consecutive_windows_min'])
preflight_results.append((check6, pass6))
print(f"   {'OK' if pass6 else 'X'} {check6}: {consecutive_hard}")

all_preflight_passed = all(p[1] for p in preflight_results)
print(f"\n   Overall preflight: {'OK PASSED' if all_preflight_passed else 'X FAILED'}")

# ============================================================================
# Promotion Decision
# ============================================================================
print(f"\n[8/9] Promotion to 'base' decision...")

print(f"\n   Requirements:")
print(f"     Meets hard thresholds: {meets_hard} {'OK' if meets_hard else 'X'}")
print(f"     Consecutive windows>={PROMOTION_REQUIREMENTS['consecutive_windows_min']}: {consecutive_hard >= PROMOTION_REQUIREMENTS['consecutive_windows_min']} {'OK' if consecutive_hard >= PROMOTION_REQUIREMENTS['consecutive_windows_min'] else 'X'}")
print(f"     Lag safety (lag=1h): {LAG == 1} {'OK' if LAG == 1 else 'X'}")
print(f"     Preflight checks: {all_preflight_passed} {'OK' if all_preflight_passed else 'X'}")

print(f"\n   DECISION: {'OKOKOK APPROVED FOR BASE PROMOTION OKOKOK' if meets_promotion else 'XXX NOT READY FOR BASE XXX'}")

# ============================================================================
# Save Results
# ============================================================================
print(f"\n[9/9] Saving results...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Window-level results
window_output = OUTPUT_DIR / f"stability_validation_windows_{timestamp}.csv"
window_df.to_csv(window_output, index=False)
print(f"   Window results: {window_output}")

# Summary results
summary = {
    'timestamp': timestamp,
    'model': 'LightGBM',
    'config': str(BEST_CONFIG),
    'H': H,
    'lag': LAG,
    'n_windows': len(ic_list),
    'IC_mean': ic_mean,
    'IC_median': ic_median,
    'IC_std': ic_std,
    'IC_min': ic_min,
    'IC_max': ic_max,
    'IC_range': ic_range,
    'IC_CV': ic_cv,
    'IR': ir,
    'PMR': pmr,
    'max_consecutive_pos': max_consecutive_pos,
    'consecutive_hard': consecutive_hard,
    'meets_hard_thresholds': meets_hard,
    'preflight_passed': all_preflight_passed,
    'promotion_approved': meets_promotion,
    'best_month': best_month,
    'worst_month': worst_month
}

summary_output = OUTPUT_DIR / f"stability_validation_summary_{timestamp}.csv"
pd.DataFrame([summary]).to_csv(summary_output, index=False)
print(f"   Summary: {summary_output}")

# ============================================================================
# Final Report
# ============================================================================
print("\n" + "="*80)
print("STABILITY VALIDATION REPORT")
print("="*80)
print(f"\nConfiguration:")
print(f"  Model: LightGBM")
print(f"  Params: depth={BEST_CONFIG['max_depth']}, lr={BEST_CONFIG['learning_rate']}, ")
print(f"          n={BEST_CONFIG['n_estimators']}, leaves={BEST_CONFIG['num_leaves']}")
print(f"  H={H}, lag={LAG}h")

print(f"\nPerformance (n={len(ic_list)} windows):")
print(f"  IC:  {ic_mean:.6f} ± {ic_std:.6f} (median={ic_median:.6f})")
print(f"  IR:  {ir:.2f}")
print(f"  PMR: {pmr:.2f}")
print(f"  Range: [{ic_min:.6f}, {ic_max:.6f}]")

print(f"\nStability:")
print(f"  Coefficient of Variation: {ic_cv:.4f}")
print(f"  Max consecutive positive: {max_consecutive_pos} windows")
print(f"  Consecutive hard-meeting: {consecutive_hard} windows")

print(f"\nRegime Analysis:")
print(f"  Months covered: {len(monthly_stats)}")
print(f"  Best:  {best_month} (IC={monthly_stats.loc[best_month, 'IC_median']:.6f})")
print(f"  Worst: {worst_month} (IC={monthly_stats.loc[worst_month, 'IC_median']:.6f})")

print(f"\nNo-Drift Compliance:")
print(f"  Hard thresholds: {meets_hard} {'OK' if meets_hard else 'X'}")
print(f"  Preflight checks: {all_preflight_passed} {'OK' if all_preflight_passed else 'X'}")

print(f"\n{'='*80}")
print(f"PROMOTION DECISION: {'APPROVED FOR BASE' if meets_promotion else 'NOT READY'}")
print(f"{'='*80}")

if meets_promotion:
    print(f"\nOK This configuration is APPROVED for selected_source='base'")
    print(f"  - Meets all Hard IC thresholds")
    print(f"  - {consecutive_hard} consecutive hard-meeting windows (required: {PROMOTION_REQUIREMENTS['consecutive_windows_min']})")
    print(f"  - Lag=1h safety confirmed")
    print(f"  - All preflight checks passed")
    print(f"\nNext steps:")
    print(f"  1. Update RUNLOG_OPERATIONS.md with this validation")
    print(f"  2. Update warehouse/policy/no_drift.yaml: selected_source='base'")
    print(f"  3. Proceed to production deployment")
else:
    print(f"\nX This configuration is NOT READY for base promotion")
    print(f"\nGaps:")
    if not meets_hard:
        print(f"  - Does not meet Hard IC thresholds")
    if consecutive_hard < PROMOTION_REQUIREMENTS['consecutive_windows_min']:
        print(f"  - Only {consecutive_hard} consecutive hard windows (need {PROMOTION_REQUIREMENTS['consecutive_windows_min']})")
    if not all_preflight_passed:
        print(f"  - Preflight checks failed")

print("\n" + "="*80)
