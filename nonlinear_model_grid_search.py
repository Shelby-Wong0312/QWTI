#!/usr/bin/env python3
"""
Nonlinear Model Grid Search (XGBoost + LightGBM)
Strategy: Test tree-based models to capture nonlinear relationships
Goal: Push H=1 IC from 0.012 (Lasso) to >=0.02 (Hard threshold)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Check and import libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("WARNING: LightGBM not installed. Install with: pip install lightgbm")

# ============================================================================
# Configuration
# ============================================================================
GDELT_PATH = Path("data/gdelt_hourly.parquet")
PRICE_PATH = Path("data/features_hourly.parquet")
OUTPUT_DIR = Path("warehouse/ic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations (curated for interpretability + performance)
XGBOOST_CONFIGS = [
    {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 'subsample': 0.8},  # Fast, conservative
    {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 100, 'subsample': 0.8},  # Default-ish
    {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 150, 'subsample': 0.8}, # Moderate
    {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 150, 'subsample': 0.8}, # Deeper
    {'max_depth': 5, 'learning_rate': 0.01, 'n_estimators': 200, 'subsample': 0.8}, # Slower learning
]

LIGHTGBM_CONFIGS = [
    {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 'subsample': 0.8, 'num_leaves': 7},   # Fast
    {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 100, 'subsample': 0.8, 'num_leaves': 31},  # Default
    {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 150, 'subsample': 0.8, 'num_leaves': 31}, # Moderate
    {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 150, 'subsample': 0.8, 'num_leaves': 63}, # Deeper
    {'max_depth': 5, 'learning_rate': 0.01, 'n_estimators': 200, 'subsample': 0.8, 'num_leaves': 31}, # Slower
]

# Evaluation settings
HORIZONS = [1, 3]  # H=1 (target), H=3 (reference)
LAG = 1
TRAIN_HOURS = 1440  # 60 days
TEST_HOURS = 720    # 30 days
STANDARDIZE = False  # Raw features (proven better for linear models)

# Bucket features
BUCKET_FEATURES = [
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt'
]

print("="*80)
print("Nonlinear Model Grid Search (XGBoost + LightGBM)")
print("="*80)
print(f"Goal: Push H=1 IC from 0.012 (Lasso) to >=0.02 (Hard threshold)")
print(f"Strategy: Tree-based models to capture nonlinear relationships")
print(f"XGBoost configs: {len(XGBOOST_CONFIGS) if XGBOOST_AVAILABLE else 0}")
print(f"LightGBM configs: {len(LIGHTGBM_CONFIGS) if LIGHTGBM_AVAILABLE else 0}")
print(f"Total: {(len(XGBOOST_CONFIGS) if XGBOOST_AVAILABLE else 0) + (len(LIGHTGBM_CONFIGS) if LIGHTGBM_AVAILABLE else 0)} configs x {len(HORIZONS)} horizons = {((len(XGBOOST_CONFIGS) if XGBOOST_AVAILABLE else 0) + (len(LIGHTGBM_CONFIGS) if LIGHTGBM_AVAILABLE else 0)) * len(HORIZONS)} runs")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[1/6] Loading data...")
gdelt_df = pd.read_parquet(GDELT_PATH)
price_df = pd.read_parquet(PRICE_PATH)

gdelt_df['ts_utc'] = pd.to_datetime(gdelt_df['ts_utc'])
price_df['ts_utc'] = pd.to_datetime(price_df['ts_utc'])
gdelt_df = gdelt_df.sort_values('ts_utc').reset_index(drop=True)
price_df = price_df.sort_values('ts_utc').reset_index(drop=True)

merged = pd.merge(gdelt_df, price_df[['ts_utc', 'ret_1h']], on='ts_utc', how='inner')
merged = merged.rename(columns={'ret_1h': 'wti_returns'})
print(f"   Merged data: {len(merged)} rows")

# Check for missing features
missing_cols = [c for c in BUCKET_FEATURES if c not in merged.columns]
if missing_cols:
    print(f"   WARNING: Missing columns: {missing_cols}")
    BUCKET_FEATURES = [c for c in BUCKET_FEATURES if c in merged.columns]
    print(f"   Using available features: {BUCKET_FEATURES}")

# ============================================================================
# Find Longest Continuous Segment
# ============================================================================
print("\n[2/6] Finding longest continuous segment...")
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
print("\n[3/6] Applying winsorization (1st/99th percentiles)...")
for col in BUCKET_FEATURES + ['wti_returns']:
    if col in segment_data.columns:
        p1 = segment_data[col].quantile(0.01)
        p99 = segment_data[col].quantile(0.99)
        segment_data[col] = segment_data[col].clip(p1, p99)

# ============================================================================
# Grid Search
# ============================================================================
print(f"\n[4/6] Running grid search...")

results = []
config_num = 0
total_configs = ((len(XGBOOST_CONFIGS) if XGBOOST_AVAILABLE else 0) +
                 (len(LIGHTGBM_CONFIGS) if LIGHTGBM_AVAILABLE else 0)) * len(HORIZONS)

for H in HORIZONS:
    print(f"\n--- Horizon H={H} ---")

    # Prepare features and target
    X = segment_data[BUCKET_FEATURES].values
    y_forward = segment_data['wti_returns'].shift(-H).values

    # Remove NaN rows
    valid_idx = ~np.isnan(y_forward)
    X_valid = X[valid_idx]
    y_valid = y_forward[valid_idx]

    if len(X_valid) < TRAIN_HOURS + TEST_HOURS:
        print(f"   SKIP: Insufficient data ({len(X_valid)} < {TRAIN_HOURS + TEST_HOURS})")
        continue

    # XGBoost configs
    if XGBOOST_AVAILABLE:
        for cfg in XGBOOST_CONFIGS:
            config_num += 1
            print(f"\n   [{config_num}/{total_configs}] XGBoost: depth={cfg['max_depth']}, lr={cfg['learning_rate']}, n={cfg['n_estimators']}")

            # Rolling window evaluation
            n_samples = len(X_valid)
            max_start = n_samples - TRAIN_HOURS - TEST_HOURS
            window_starts = np.arange(0, max_start + 1, TEST_HOURS)

            ic_list = []

            for start in window_starts:
                train_end = start + TRAIN_HOURS
                test_end = train_end + TEST_HOURS

                if test_end > n_samples:
                    break

                X_train = X_valid[start:train_end]
                y_train = y_valid[start:train_end]
                X_test = X_valid[train_end:test_end]
                y_test = y_valid[train_end:test_end]

                # Train XGBoost
                model = xgb.XGBRegressor(
                    max_depth=cfg['max_depth'],
                    learning_rate=cfg['learning_rate'],
                    n_estimators=cfg['n_estimators'],
                    subsample=cfg['subsample'],
                    random_state=42,
                    verbosity=0,
                    objective='reg:squarederror'
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate IC
                ic = np.corrcoef(y_pred, y_test)[0, 1]
                ic_list.append(ic)

            # Aggregate metrics
            ic_list = [ic for ic in ic_list if not np.isnan(ic)]

            if len(ic_list) == 0:
                print(f"      SKIP: No valid windows")
                continue

            ic_mean = np.mean(ic_list)
            ic_std = np.std(ic_list)
            ir = ic_mean / ic_std if ic_std > 0 else 0
            pmr = np.mean([ic > 0 for ic in ic_list])
            n_windows = len(ic_list)

            # Check Hard threshold
            is_hard = (ic_mean >= 0.02) and (ir >= 0.5) and (pmr >= 0.55)

            print(f"      IC={ic_mean:.6f}, IR={ir:.2f}, PMR={pmr:.2f}, n={n_windows}, Hard={is_hard}")

            results.append({
                'H': H,
                'model': 'XGBoost',
                'max_depth': cfg['max_depth'],
                'learning_rate': cfg['learning_rate'],
                'n_estimators': cfg['n_estimators'],
                'subsample': cfg['subsample'],
                'IC': ic_mean,
                'IC_std': ic_std,
                'IR': ir,
                'PMR': pmr,
                'n_windows': n_windows,
                'is_hard': is_hard
            })

    # LightGBM configs
    if LIGHTGBM_AVAILABLE:
        for cfg in LIGHTGBM_CONFIGS:
            config_num += 1
            print(f"\n   [{config_num}/{total_configs}] LightGBM: depth={cfg['max_depth']}, lr={cfg['learning_rate']}, n={cfg['n_estimators']}, leaves={cfg['num_leaves']}")

            # Rolling window evaluation
            n_samples = len(X_valid)
            max_start = n_samples - TRAIN_HOURS - TEST_HOURS
            window_starts = np.arange(0, max_start + 1, TEST_HOURS)

            ic_list = []

            for start in window_starts:
                train_end = start + TRAIN_HOURS
                test_end = train_end + TEST_HOURS

                if test_end > n_samples:
                    break

                X_train = X_valid[start:train_end]
                y_train = y_valid[start:train_end]
                X_test = X_valid[train_end:test_end]
                y_test = y_valid[train_end:test_end]

                # Train LightGBM
                model = lgb.LGBMRegressor(
                    max_depth=cfg['max_depth'],
                    learning_rate=cfg['learning_rate'],
                    n_estimators=cfg['n_estimators'],
                    subsample=cfg['subsample'],
                    num_leaves=cfg['num_leaves'],
                    random_state=42,
                    verbosity=-1,
                    force_col_wise=True
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate IC
                ic = np.corrcoef(y_pred, y_test)[0, 1]
                ic_list.append(ic)

            # Aggregate metrics
            ic_list = [ic for ic in ic_list if not np.isnan(ic)]

            if len(ic_list) == 0:
                print(f"      SKIP: No valid windows")
                continue

            ic_mean = np.mean(ic_list)
            ic_std = np.std(ic_list)
            ir = ic_mean / ic_std if ic_std > 0 else 0
            pmr = np.mean([ic > 0 for ic in ic_list])
            n_windows = len(ic_list)

            # Check Hard threshold
            is_hard = (ic_mean >= 0.02) and (ir >= 0.5) and (pmr >= 0.55)

            print(f"      IC={ic_mean:.6f}, IR={ir:.2f}, PMR={pmr:.2f}, n={n_windows}, Hard={is_hard}")

            results.append({
                'H': H,
                'model': 'LightGBM',
                'max_depth': cfg['max_depth'],
                'learning_rate': cfg['learning_rate'],
                'n_estimators': cfg['n_estimators'],
                'subsample': cfg['subsample'],
                'num_leaves': cfg.get('num_leaves', None),
                'IC': ic_mean,
                'IC_std': ic_std,
                'IR': ir,
                'PMR': pmr,
                'n_windows': n_windows,
                'is_hard': is_hard
            })

# ============================================================================
# Save Results
# ============================================================================
print(f"\n[5/6] Saving results...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = OUTPUT_DIR / f"nonlinear_model_grid_search_{timestamp}.csv"

results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"   Saved to: {output_file}")

# ============================================================================
# Summary Analysis
# ============================================================================
print("\n[6/6] " + "="*80)
print("SUMMARY: Nonlinear vs Linear Comparison")
print("="*80)

# Best overall
best_overall = results_df.loc[results_df['IC'].idxmax()]
print(f"\nBest overall:")
print(f"  Model: {best_overall['model']}")
print(f"  H={int(best_overall['H'])}")
print(f"  Params: depth={int(best_overall['max_depth'])}, lr={best_overall['learning_rate']}, n={int(best_overall['n_estimators'])}")
print(f"  IC={best_overall['IC']:.6f}, IR={best_overall['IR']:.2f}, PMR={best_overall['PMR']:.2f}")
print(f"  Hard: {best_overall['is_hard']}")

for H in HORIZONS:
    h_results = results_df[results_df['H'] == H]
    if len(h_results) == 0:
        continue

    print(f"\n--- H={H} ---")

    # Best by model
    for model in ['XGBoost', 'LightGBM']:
        model_results = h_results[h_results['model'] == model]
        if len(model_results) == 0:
            continue

        best = model_results.loc[model_results['IC'].idxmax()]
        print(f"\nBest {model}:")
        print(f"  Params: depth={int(best['max_depth'])}, lr={best['learning_rate']}, n={int(best['n_estimators'])}")
        print(f"  IC={best['IC']:.6f}, IR={best['IR']:.2f}, PMR={best['PMR']:.2f}")
        print(f"  Hard: {best['is_hard']}")

    # Compare to Lasso baseline
    lasso_baseline_h1 = 0.012062
    lasso_baseline_h3 = 0.006325
    baseline = lasso_baseline_h1 if H == 1 else lasso_baseline_h3

    best_h = h_results.loc[h_results['IC'].idxmax()]
    improvement = best_h['IC'] - baseline
    improvement_pct = (improvement / baseline * 100) if baseline != 0 else 0

    print(f"\nImprovement vs Lasso (alpha=0.005, IC={baseline:.6f}):")
    print(f"  Best nonlinear IC: {best_h['IC']:.6f}")
    print(f"  Delta: {improvement:+.6f} ({improvement_pct:+.1f}%)")
    print(f"  Model: {best_h['model']}")

# Check Hard threshold
hard_achievers = results_df[results_df['is_hard']]
if len(hard_achievers) > 0:
    print(f"\n*** HARD THRESHOLD ACHIEVED ({len(hard_achievers)} configs) ***")
    for _, row in hard_achievers.iterrows():
        print(f"  H={int(row['H'])}, {row['model']}: IC={row['IC']:.6f}, IR={row['IR']:.2f}, PMR={row['PMR']:.2f}")
        print(f"    Params: depth={int(row['max_depth'])}, lr={row['learning_rate']}, n={int(row['n_estimators'])}")
else:
    print(f"\nNo Hard threshold achieved (IC>=0.02 AND IR>=0.5 AND PMR>=0.55)")

    # Show closest
    results_df['dist_to_hard'] = np.sqrt(
        (np.maximum(0, 0.02 - results_df['IC']))**2 +
        (np.maximum(0, 0.5 - results_df['IR']))**2 +
        (np.maximum(0, 0.55 - results_df['PMR']))**2
    )
    closest = results_df.loc[results_df['dist_to_hard'].idxmin()]
    print(f"\nClosest to Hard:")
    print(f"  H={int(closest['H'])}, {closest['model']}")
    print(f"  IC={closest['IC']:.6f} (gap: {max(0, 0.02-closest['IC']):.6f})")
    print(f"  IR={closest['IR']:.2f} (gap: {max(0, 0.5-closest['IR']):.2f})")
    print(f"  PMR={closest['PMR']:.2f} (gap: {max(0, 0.55-closest['PMR']):.2f})")

print("\n" + "="*80)
print("Nonlinear model grid search complete!")
print("="*80)
