#!/usr/bin/env python3
"""
LightGBM Sweep on Clean Data Period (2024-10-01+)
- Uses only complete data period where GDELT buckets have real values
- 6 base features: ovx + 5 GDELT buckets (cl1_cl2 excluded - all zeros)
- Rolling window: 60d train / 15d test
- Goal: Recover IC/IR to 0.04+/1.0+ by avoiding noisy historical data
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
    print("ERROR: LightGBM not installed")
    exit(1)

# ============================================================================
# Configuration
# ============================================================================
FEATURES_PATH = Path("features_hourly_with_regime.parquet")
OUTPUT_DIR = Path("warehouse/ic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Time filter - only use clean data period
START_DATE = "2024-10-01"

# Rolling window (60d/15d)
TRAIN_HOURS = 1440  # 60 days
TEST_HOURS = 360    # 15 days
H = 1  # Prediction horizon

# 6 base features (excluding cl1_cl2 which is all zeros)
FEATURE_COLS = [
    'ovx',
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt',
]

# Hyperparameter grid - focused sweep
PARAM_GRID = {
    'max_depth': [3, 5, 7],
    'num_leaves': [7, 15, 31],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9],
    'n_estimators': [100, 150, 200],
}

# Generate combinations (3*3*3*3*3 = 243, sample 40)
all_combos = list(itertools.product(
    PARAM_GRID['max_depth'],
    PARAM_GRID['num_leaves'],
    PARAM_GRID['learning_rate'],
    PARAM_GRID['subsample'],
    PARAM_GRID['n_estimators'],
))

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
        'n_estimators': combo[4],
    })

print("="*80)
print("LightGBM Sweep on Clean Data Period (2024-10-01+)")
print("="*80)
print(f"Data source: {FEATURES_PATH}")
print(f"Time filter: >= {START_DATE}")
print(f"Features: {len(FEATURE_COLS)} (ovx + 5 GDELT buckets)")
print(f"Rolling window: {TRAIN_HOURS}h train / {TEST_HOURS}h test")
print(f"Configs to test: {len(LIGHTGBM_CONFIGS)}")
print("="*80)

# ============================================================================
# Load and Filter Data
# ============================================================================
print("\n[1/5] Loading and filtering data...")
df = pd.read_parquet(FEATURES_PATH)
print(f"   Full data: {len(df)} rows")

# Apply time filter
df = df[df.index >= START_DATE]
print(f"   After time filter (>= {START_DATE}): {len(df)} rows")
print(f"   Date range: {df.index.min()} to {df.index.max()}")

# Check data availability
total_hours = len(df)
min_required = TRAIN_HOURS + TEST_HOURS
print(f"   Minimum required: {min_required}h, Available: {total_hours}h")

if total_hours < min_required:
    print(f"ERROR: Insufficient data! Need {min_required}h, have {total_hours}h")
    exit(1)

# Check feature quality
print("\n   Feature coverage:")
for col in FEATURE_COLS:
    if col in df.columns:
        non_zero = (df[col] != 0).sum()
        pct = non_zero / len(df) * 100
        print(f"     {col}: {pct:.1f}% non-zero")
    else:
        print(f"     {col}: MISSING!")

# ============================================================================
# Prepare Data
# ============================================================================
print("\n[2/5] Preparing data...")

# Ensure sorted by time
df = df.sort_index()

# Fill NaN with 0 for features
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

# Create target (forward return)
df['target'] = df['wti_returns'].shift(-H)
df = df.dropna(subset=['target'])

print(f"   After preparation: {len(df)} rows")

# Winsorization
print("   Applying winsorization (1st/99th)...")
for col in FEATURE_COLS + ['target']:
    p1 = df[col].quantile(0.01)
    p99 = df[col].quantile(0.99)
    df[col] = df[col].clip(p1, p99)

# ============================================================================
# Grid Search
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

if len(window_starts) < 3:
    print("WARNING: Very few windows. Results may be unstable.")

for cfg_idx, cfg in enumerate(LIGHTGBM_CONFIGS):
    print(f"\n   [{cfg_idx+1}/{len(LIGHTGBM_CONFIGS)}] depth={cfg['max_depth']}, leaves={cfg['num_leaves']}, lr={cfg['learning_rate']}, sub={cfg['subsample']}, n_est={cfg['n_estimators']}")

    ic_list = []

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

        # Calculate IC
        if np.std(y_pred) > 0 and np.std(y_test) > 0:
            ic = np.corrcoef(y_pred, y_test)[0, 1]
        else:
            ic = 0

        ic_list.append(ic)

    # Filter valid ICs
    valid_ics = [ic for ic in ic_list if not np.isnan(ic)]

    if len(valid_ics) == 0:
        print(f"      SKIP: No valid windows")
        continue

    # Metrics
    ic_mean = np.mean(valid_ics)
    ic_std = np.std(valid_ics)
    ir = ic_mean / ic_std if ic_std > 0 else 0
    pmr = np.mean([ic > 0 for ic in valid_ics])
    n_windows = len(valid_ics)

    # Thresholds
    is_pmr_ok = pmr >= 0.55
    is_hard = (ic_mean >= 0.02) and (ir >= 0.5) and (pmr >= 0.55)

    status = "HARD!" if is_hard else ("PMR_OK" if is_pmr_ok else "")
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
output_file = OUTPUT_DIR / f"lightgbm_clean_period_sweep_{timestamp}.csv"

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('IC', ascending=False)
results_df.to_csv(output_file, index=False)
print(f"   Saved to: {output_file}")

# ============================================================================
# Summary
# ============================================================================
print(f"\n[5/5] " + "="*80)
print("SUMMARY: Clean Period LightGBM Sweep")
print("="*80)

if len(results_df) == 0:
    print("No valid results!")
    exit(1)

# Top 10
print("\n--- Top 10 by IC ---")
top10 = results_df.head(10)
for _, row in top10.iterrows():
    status = "HARD!" if row['is_hard'] else ("PMR_OK" if row['is_pmr_ok'] else "")
    print(f"  IC={row['IC']:.6f}, IR={row['IR']:.3f}, PMR={row['PMR']:.3f} | d={int(row['max_depth'])}, l={int(row['num_leaves'])}, lr={row['learning_rate']}, sub={row['subsample']}, n={int(row['n_estimators'])} {status}")

# Hard achievers
hard_achievers = results_df[results_df['is_hard']]
if len(hard_achievers) > 0:
    print(f"\n*** HARD THRESHOLD ACHIEVED ({len(hard_achievers)} configs) ***")
    for _, row in hard_achievers.iterrows():
        print(f"  IC={row['IC']:.6f}, IR={row['IR']:.3f}, PMR={row['PMR']:.3f}")
        print(f"  Config: depth={int(row['max_depth'])}, leaves={int(row['num_leaves'])}, lr={row['learning_rate']}, sub={row['subsample']}, n_est={int(row['n_estimators'])}")
else:
    print("\n--- Hard threshold NOT achieved ---")
    # Show gap to Hard
    best = results_df.iloc[0]
    print(f"  Best IC: {best['IC']:.6f} (gap to 0.02: {0.02 - best['IC']:.6f})")
    print(f"  Best IR: {best['IR']:.3f} (gap to 0.5: {max(0, 0.5 - best['IR']):.3f})")

# PMR OK
pmr_ok = results_df[results_df['is_pmr_ok']]
print(f"\n--- PMR >= 0.55: {len(pmr_ok)} configs ---")
if len(pmr_ok) > 0:
    best_pmr = pmr_ok.iloc[0]
    print(f"  Best: IC={best_pmr['IC']:.6f}, IR={best_pmr['IR']:.3f}, PMR={best_pmr['PMR']:.3f}")

# Recommendation
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
if len(hard_achievers) > 0:
    best = hard_achievers.iloc[0]
    print(f"Use HARD-achieving config:")
elif len(pmr_ok) > 0:
    best = pmr_ok.iloc[0]
    print(f"Use best PMR>=0.55 config:")
else:
    best = results_df.iloc[0]
    print(f"Use best IC config (PMR below threshold):")

print(f"""
LGBMRegressor(
    max_depth={int(best['max_depth'])},
    num_leaves={int(best['num_leaves'])},
    learning_rate={best['learning_rate']},
    subsample={best['subsample']},
    n_estimators={int(best['n_estimators'])},
    random_state=42
)

Metrics: IC={best['IC']:.6f}, IR={best['IR']:.3f}, PMR={best['PMR']:.3f}
""")

print("="*80)
print(f"Results saved to: {output_file}")
print("="*80)
