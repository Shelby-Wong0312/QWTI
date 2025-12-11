#!/usr/bin/env python3
"""
LightGBM Stability Validation (2024-10+)
- Uses recommended config from sweep: depth=3, leaves=15, lr=0.05, sub=0.8, n=100
- 7 features: ovx + 5 GDELT buckets + momentum_24h
- Rolling window: 60d train / 15d test
- Goal: Validate IC/IR stability and push IR over 0.5 threshold
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: LightGBM not installed")
    exit(1)

# ============================================================================
# Configuration
# ============================================================================
FEATURES_PATH = Path("features_hourly_with_regime.parquet")
OUTPUT_DIR = Path("warehouse/ic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2024-10-01"
TRAIN_HOURS = 1440  # 60 days
TEST_HOURS = 360    # 15 days
H = 1

# 7 available features (cl1_cl2 excluded - all zeros)
FEATURE_COLS = [
    'ovx',
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt',
    'momentum_24h',  # Added as term structure proxy
]

# Recommended config from sweep
BEST_CONFIG = {
    'max_depth': 3,
    'num_leaves': 15,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'n_estimators': 100,
}

# Also test a few variations to potentially improve IR
CONFIGS_TO_TEST = [
    # Base recommended
    {'max_depth': 3, 'num_leaves': 15, 'learning_rate': 0.05, 'subsample': 0.8, 'n_estimators': 100, 'name': 'base_recommended'},
    # Lower variance (might improve IR)
    {'max_depth': 3, 'num_leaves': 15, 'learning_rate': 0.03, 'subsample': 0.8, 'n_estimators': 150, 'name': 'lower_lr'},
    # More trees
    {'max_depth': 3, 'num_leaves': 15, 'learning_rate': 0.05, 'subsample': 0.8, 'n_estimators': 200, 'name': 'more_trees'},
    # Higher subsample (more stable)
    {'max_depth': 3, 'num_leaves': 15, 'learning_rate': 0.05, 'subsample': 0.9, 'n_estimators': 100, 'name': 'higher_subsample'},
    # Fewer leaves (simpler model)
    {'max_depth': 3, 'num_leaves': 7, 'learning_rate': 0.05, 'subsample': 0.8, 'n_estimators': 100, 'name': 'fewer_leaves'},
    # Combined: lower lr + more trees
    {'max_depth': 3, 'num_leaves': 15, 'learning_rate': 0.03, 'subsample': 0.85, 'n_estimators': 200, 'name': 'conservative'},
]

print("="*80)
print("LightGBM Stability Validation (2024-10+)")
print("="*80)
print(f"Data: {FEATURES_PATH} (filtered >= {START_DATE})")
print(f"Features: {len(FEATURE_COLS)}")
print(f"Configs to test: {len(CONFIGS_TO_TEST)}")
print(f"Target: Push IR >= 0.5 while maintaining IC >= 0.02, PMR >= 0.55")
print("="*80)

# ============================================================================
# Load and Prepare Data
# ============================================================================
print("\n[1/4] Loading data...")
df = pd.read_parquet(FEATURES_PATH)
df = df[df.index >= START_DATE]
print(f"   Filtered data: {len(df)} rows")
print(f"   Date range: {df.index.min()} to {df.index.max()}")

# Prepare
df = df.sort_index()
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
df['target'] = df['wti_returns'].shift(-H)
df = df.dropna(subset=['target'])

# Winsorize
for col in FEATURE_COLS + ['target']:
    p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(p1, p99)

print(f"   After preparation: {len(df)} rows")

# Check feature coverage
print("\n   Feature coverage:")
for col in FEATURE_COLS:
    non_zero = (df[col] != 0).sum()
    pct = non_zero / len(df) * 100
    print(f"     {col}: {pct:.1f}%")

# ============================================================================
# Stability Validation
# ============================================================================
print(f"\n[2/4] Running stability validation...")

X = df[FEATURE_COLS].values
y = df['target'].values
timestamps = df.index.tolist()

n_samples = len(X)
max_start = n_samples - TRAIN_HOURS - TEST_HOURS
window_starts = np.arange(0, max_start + 1, TEST_HOURS)

print(f"   Total samples: {n_samples}")
print(f"   Rolling windows: {len(window_starts)}")

results = []

for cfg in CONFIGS_TO_TEST:
    name = cfg['name']
    print(f"\n   --- {name} ---")

    ic_list = []
    window_results = []

    for w_idx, start in enumerate(window_starts):
        train_end = start + TRAIN_HOURS
        test_end = train_end + TEST_HOURS

        if test_end > n_samples:
            break

        X_train = X[start:train_end]
        y_train = y[start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]

        # Train
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

        # IC
        if np.std(y_pred) > 0 and np.std(y_test) > 0:
            ic = np.corrcoef(y_pred, y_test)[0, 1]
        else:
            ic = 0

        ic_list.append(ic)
        window_results.append({
            'window': w_idx + 1,
            'start': timestamps[start],
            'end': timestamps[test_end-1],
            'ic': ic
        })

    # Metrics
    valid_ics = [ic for ic in ic_list if not np.isnan(ic)]
    ic_mean = np.mean(valid_ics)
    ic_std = np.std(valid_ics)
    ir = ic_mean / ic_std if ic_std > 0 else 0
    pmr = np.mean([ic > 0 for ic in valid_ics])
    n_windows = len(valid_ics)

    # Thresholds
    ic_ok = ic_mean >= 0.02
    ir_ok = ir >= 0.5
    pmr_ok = pmr >= 0.55
    is_hard = ic_ok and ir_ok and pmr_ok

    status = "HARD!" if is_hard else ""
    print(f"   IC={ic_mean:.6f} {'OK' if ic_ok else 'X'}, IR={ir:.3f} {'OK' if ir_ok else 'X'}, PMR={pmr:.3f} {'OK' if pmr_ok else 'X'} {status}")

    # Window details
    print(f"   Window ICs: {[f'{ic:.4f}' for ic in valid_ics[:5]]}...")

    results.append({
        'name': name,
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
        'ic_ok': ic_ok,
        'ir_ok': ir_ok,
        'pmr_ok': pmr_ok,
        'is_hard': is_hard,
        'window_ics': valid_ics
    })

# ============================================================================
# Save Results
# ============================================================================
print(f"\n[3/4] Saving results...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Summary CSV
results_df = pd.DataFrame([{k:v for k,v in r.items() if k != 'window_ics'} for r in results])
results_df = results_df.sort_values('IR', ascending=False)
output_file = OUTPUT_DIR / f"lightgbm_stability_validation_{timestamp}.csv"
results_df.to_csv(output_file, index=False)
print(f"   Saved to: {output_file}")

# ============================================================================
# Summary
# ============================================================================
print(f"\n[4/4] " + "="*80)
print("SUMMARY: Stability Validation Results")
print("="*80)

print("\n--- All Configs Ranked by IR ---")
for _, row in results_df.iterrows():
    ic_s = "OK" if row['ic_ok'] else "X "
    ir_s = "OK" if row['ir_ok'] else "X "
    pmr_s = "OK" if row['pmr_ok'] else "X "
    hard = " HARD!" if row['is_hard'] else ""
    print(f"  {row['name']:20s}: IC={row['IC']:.6f}[{ic_s}] IR={row['IR']:.3f}[{ir_s}] PMR={row['PMR']:.3f}[{pmr_s}]{hard}")

# Hard achievers
hard_achievers = results_df[results_df['is_hard']]
if len(hard_achievers) > 0:
    print(f"\n*** HARD THRESHOLD ACHIEVED ({len(hard_achievers)} configs) ***")
    for _, row in hard_achievers.iterrows():
        print(f"\n  Config: {row['name']}")
        print(f"    IC={row['IC']:.6f}, IR={row['IR']:.3f}, PMR={row['PMR']:.3f}")
        print(f"    Params: depth={int(row['max_depth'])}, leaves={int(row['num_leaves'])}, lr={row['learning_rate']}, sub={row['subsample']}, n={int(row['n_estimators'])}")
else:
    print("\n--- Hard threshold NOT achieved ---")
    best = results_df.iloc[0]
    print(f"  Best IR: {best['IR']:.3f} (gap: {0.5 - best['IR']:.3f})")
    print(f"  IC: {best['IC']:.6f}, PMR: {best['PMR']:.3f}")

# Recommendation
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

best = results_df.iloc[0]
if best['is_hard']:
    print(f"\nUse HARD-achieving config: {best['name']}")
elif best['ir_ok']:
    print(f"\nUse IR-best config: {best['name']}")
else:
    print(f"\nBest available (IR not yet at 0.5): {best['name']}")

print(f"""
LGBMRegressor(
    max_depth={int(best['max_depth'])},
    num_leaves={int(best['num_leaves'])},
    learning_rate={best['learning_rate']},
    subsample={best['subsample']},
    n_estimators={int(best['n_estimators'])},
    random_state=42
)

Metrics:
  IC:  {best['IC']:.6f} (threshold: 0.02) {'OK' if best['ic_ok'] else 'BELOW'}
  IR:  {best['IR']:.3f} (threshold: 0.5) {'OK' if best['ir_ok'] else 'BELOW'}
  PMR: {best['PMR']:.3f} (threshold: 0.55) {'OK' if best['pmr_ok'] else 'BELOW'}
""")

print("="*80)
print(f"Results saved to: {output_file}")
print("="*80)
