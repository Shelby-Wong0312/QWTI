#!/usr/bin/env python3
"""
LightGBM Hyperparameter Sweep with Regime Features
Data source: features_hourly_with_regime.parquet (2017-05~2025-12)
Features: cl1_cl2/ovx + regime flags + EIA flags + GDELT buckets
Target: H=1 prediction
Rolling window: 60d train / 15d test (1440/360 hours)
Goal: Find best IC/IR with PMR >= 0.55
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import itertools
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("ERROR: LightGBM not installed. Install with: pip install lightgbm")
    exit(1)

# ============================================================================
# Configuration
# ============================================================================
FEATURES_PATH = Path("features_hourly_with_regime.parquet")
OUTPUT_DIR = Path("warehouse/ic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Rolling window settings (60d train / 15d test)
TRAIN_HOURS = 1440  # 60 days
TEST_HOURS = 360    # 15 days
H = 1  # Prediction horizon

# Hyperparameter grid for ~40 combinations
PARAM_GRID = {
    'max_depth': [3, 5, 7, 9],
    'num_leaves': [7, 15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
}

# Generate all combinations (4*4*3*3 = 144, we'll sample ~40)
all_combos = list(itertools.product(
    PARAM_GRID['max_depth'],
    PARAM_GRID['num_leaves'],
    PARAM_GRID['learning_rate'],
    PARAM_GRID['subsample']
))

# Smart sampling: prioritize diverse combinations
np.random.seed(42)
selected_indices = np.random.choice(len(all_combos), size=min(40, len(all_combos)), replace=False)
LIGHTGBM_CONFIGS = []
for idx in selected_indices:
    combo = all_combos[idx]
    LIGHTGBM_CONFIGS.append({
        'max_depth': combo[0],
        'num_leaves': combo[1],
        'learning_rate': combo[2],
        'subsample': combo[3],
        'n_estimators': 150,  # Fixed
    })

# Features to use (all available in regime parquet)
FEATURE_COLS = [
    # GDELT buckets
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt',
    # Term structure & volatility
    'cl1_cl2',
    'ovx',
    # Regime flags
    'vol_regime_high',
    'vol_regime_low',
    'ovx_high',
    'ovx_low',
    'momentum_24h',
    'gdelt_high',
    # EIA flags
    'eia_pre_4h',
    'eia_post_4h',
    'eia_day',
]

print("="*80)
print("LightGBM Hyperparameter Sweep with Regime Features")
print("="*80)
print(f"Data source: {FEATURES_PATH}")
print(f"Features: {len(FEATURE_COLS)} columns")
print(f"Rolling window: {TRAIN_HOURS}h train / {TEST_HOURS}h test (60d/15d)")
print(f"Target horizon: H={H}")
print(f"Configs to test: {len(LIGHTGBM_CONFIGS)}")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[1/5] Loading data...")
df = pd.read_parquet(FEATURES_PATH)
print(f"   Loaded: {len(df)} rows")
print(f"   Date range: {df.index.min()} to {df.index.max()}")
print(f"   Columns: {df.columns.tolist()}")

# Check for target column
if 'wti_returns' not in df.columns:
    print("ERROR: 'wti_returns' column not found!")
    exit(1)

# Check for missing features
available_features = [c for c in FEATURE_COLS if c in df.columns]
missing_features = [c for c in FEATURE_COLS if c not in df.columns]
if missing_features:
    print(f"   WARNING: Missing features: {missing_features}")
print(f"   Using {len(available_features)} features: {available_features}")

FEATURE_COLS = available_features

# ============================================================================
# Prepare Data
# ============================================================================
print("\n[2/5] Preparing data...")

# Reset index if needed
if not isinstance(df.index, pd.DatetimeIndex):
    if 'ts_utc' in df.columns:
        df = df.set_index('ts_utc')
    else:
        df = df.reset_index(drop=True)

# Sort by time
df = df.sort_index()

# Handle NaN in features
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

# Create forward return (target)
df['target'] = df['wti_returns'].shift(-H)

# Drop rows with NaN target
df = df.dropna(subset=['target'])

print(f"   After preparation: {len(df)} rows")

# Winsorization (1st/99th percentiles)
print("   Applying winsorization...")
for col in FEATURE_COLS + ['target']:
    if col in df.columns:
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(p1, p99)

# ============================================================================
# Grid Search with Rolling Window
# ============================================================================
print(f"\n[3/5] Running grid search ({len(LIGHTGBM_CONFIGS)} configs)...")

results = []
X = df[FEATURE_COLS].values
y = df['target'].values
timestamps = df.index.tolist()

n_samples = len(X)
max_start = n_samples - TRAIN_HOURS - TEST_HOURS
window_starts = np.arange(0, max_start + 1, TEST_HOURS)

print(f"   Total samples: {n_samples}")
print(f"   Rolling windows: {len(window_starts)}")

for cfg_idx, cfg in enumerate(LIGHTGBM_CONFIGS):
    print(f"\n   [{cfg_idx+1}/{len(LIGHTGBM_CONFIGS)}] depth={cfg['max_depth']}, leaves={cfg['num_leaves']}, lr={cfg['learning_rate']}, subsample={cfg['subsample']}")

    ic_list = []
    window_details = []

    for start in window_starts:
        train_end = start + TRAIN_HOURS
        test_end = train_end + TEST_HOURS

        if test_end > n_samples:
            break

        X_train = X[start:train_end]
        y_train = y[start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]

        # Train LightGBM
        model = lgb.LGBMRegressor(
            max_depth=cfg['max_depth'],
            num_leaves=cfg['num_leaves'],
            learning_rate=cfg['learning_rate'],
            subsample=cfg['subsample'],
            n_estimators=cfg['n_estimators'],
            random_state=42,
            verbosity=-1,
            force_col_wise=True
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate IC (Spearman correlation)
        if np.std(y_pred) > 0 and np.std(y_test) > 0:
            ic = np.corrcoef(y_pred, y_test)[0, 1]
        else:
            ic = 0

        ic_list.append(ic)
        window_details.append({
            'window_start': timestamps[start] if isinstance(timestamps[0], (pd.Timestamp, datetime)) else start,
            'window_end': timestamps[test_end-1] if isinstance(timestamps[0], (pd.Timestamp, datetime)) else test_end-1,
            'ic': ic
        })

    # Filter valid ICs
    valid_ics = [ic for ic in ic_list if not np.isnan(ic)]

    if len(valid_ics) == 0:
        print(f"      SKIP: No valid windows")
        continue

    # Calculate metrics
    ic_mean = np.mean(valid_ics)
    ic_std = np.std(valid_ics)
    ir = ic_mean / ic_std if ic_std > 0 else 0
    pmr = np.mean([ic > 0 for ic in valid_ics])
    n_windows = len(valid_ics)

    # Check thresholds
    is_pmr_ok = pmr >= 0.55
    is_hard = (ic_mean >= 0.02) and (ir >= 0.5) and (pmr >= 0.55)

    status = "HARD" if is_hard else ("PMR_OK" if is_pmr_ok else "")
    print(f"      IC={ic_mean:.6f}, IR={ir:.3f}, PMR={pmr:.3f}, n={n_windows} {status}")

    results.append({
        'config_id': cfg_idx + 1,
        'max_depth': cfg['max_depth'],
        'num_leaves': cfg['num_leaves'],
        'learning_rate': cfg['learning_rate'],
        'subsample': cfg['subsample'],
        'n_estimators': cfg['n_estimators'],
        'IC': ic_mean,
        'IC_std': ic_std,
        'IR': ir,
        'PMR': pmr,
        'n_windows': n_windows,
        'is_pmr_ok': is_pmr_ok,
        'is_hard': is_hard
    })

# ============================================================================
# Save Results
# ============================================================================
print(f"\n[4/5] Saving results...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = OUTPUT_DIR / f"lightgbm_regime_sweep_{timestamp}.csv"

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('IC', ascending=False)
results_df.to_csv(output_file, index=False)
print(f"   Saved to: {output_file}")

# ============================================================================
# Summary Analysis
# ============================================================================
print(f"\n[5/5] " + "="*80)
print("SUMMARY: LightGBM Regime Sweep Results")
print("="*80)

if len(results_df) == 0:
    print("No valid results!")
    exit(1)

# Top 10 by IC
print("\n--- Top 10 by IC ---")
top10 = results_df.head(10)
for idx, row in top10.iterrows():
    pmr_status = "PMR_OK" if row['is_pmr_ok'] else ""
    hard_status = "HARD" if row['is_hard'] else ""
    status = hard_status if hard_status else pmr_status
    print(f"  #{int(row['config_id']):2d}: IC={row['IC']:.6f}, IR={row['IR']:.3f}, PMR={row['PMR']:.3f} | depth={int(row['max_depth'])}, leaves={int(row['num_leaves'])}, lr={row['learning_rate']}, sub={row['subsample']} {status}")

# Best with PMR >= 0.55
pmr_ok = results_df[results_df['is_pmr_ok']]
if len(pmr_ok) > 0:
    print(f"\n--- Best with PMR >= 0.55 ({len(pmr_ok)} configs) ---")
    best_pmr_ok = pmr_ok.iloc[0]  # Already sorted by IC
    print(f"  Best: IC={best_pmr_ok['IC']:.6f}, IR={best_pmr_ok['IR']:.3f}, PMR={best_pmr_ok['PMR']:.3f}")
    print(f"  Params: depth={int(best_pmr_ok['max_depth'])}, leaves={int(best_pmr_ok['num_leaves'])}, lr={best_pmr_ok['learning_rate']}, subsample={best_pmr_ok['subsample']}")
else:
    print("\n--- No configs achieved PMR >= 0.55 ---")
    # Show closest to PMR threshold
    closest_pmr = results_df.iloc[(results_df['PMR'] - 0.55).abs().argsort()[:3]]
    print("Closest to PMR >= 0.55:")
    for _, row in closest_pmr.iterrows():
        print(f"  IC={row['IC']:.6f}, IR={row['IR']:.3f}, PMR={row['PMR']:.3f}")

# Hard threshold achievers
hard_achievers = results_df[results_df['is_hard']]
if len(hard_achievers) > 0:
    print(f"\n*** HARD THRESHOLD ACHIEVED ({len(hard_achievers)} configs) ***")
    for _, row in hard_achievers.iterrows():
        print(f"  IC={row['IC']:.6f}, IR={row['IR']:.3f}, PMR={row['PMR']:.3f}")
        print(f"  Params: depth={int(row['max_depth'])}, leaves={int(row['num_leaves'])}, lr={row['learning_rate']}, subsample={row['subsample']}")
else:
    print("\n--- Hard threshold (IC>=0.02, IR>=0.5, PMR>=0.55) not achieved ---")

# Recommendation for new base
print("\n" + "="*80)
print("RECOMMENDATION FOR NEW BASE")
print("="*80)

# Select best IC with PMR >= 0.55, or best overall if none qualify
if len(pmr_ok) > 0:
    best = pmr_ok.iloc[0]
    print(f"\nRecommended config (best IC with PMR >= 0.55):")
else:
    best = results_df.iloc[0]
    print(f"\nRecommended config (best IC overall, PMR below threshold):")

print(f"  IC: {best['IC']:.6f}")
print(f"  IR: {best['IR']:.3f}")
print(f"  PMR: {best['PMR']:.3f}")
print(f"  Config:")
print(f"    max_depth: {int(best['max_depth'])}")
print(f"    num_leaves: {int(best['num_leaves'])}")
print(f"    learning_rate: {best['learning_rate']}")
print(f"    subsample: {best['subsample']}")
print(f"    n_estimators: {int(best['n_estimators'])}")

print("\n" + "="*80)
print("LightGBM regime sweep complete!")
print(f"Results saved to: {output_file}")
print("="*80)
