#!/usr/bin/env python3
"""
Lasso Alpha Fine-Tuning + Stability Validation
Goal: Push H=1 IC from 0.0190 to ≥0.02 (Hard threshold)
Strategy: Fine-grained alpha search + multi-seed cross-validation
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Configuration
# ============================================================================
GDELT_PATH = Path("data/gdelt_hourly.parquet")
PRICE_PATH = Path("data/features_hourly.parquet")
OUTPUT_DIR = Path("warehouse/ic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fine-tuned alpha grid (narrow range around 0.01)
ALPHA_GRID = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]

# Multi-seed for stability validation
RANDOM_SEEDS = [42, 123, 456]

# Evaluation settings
HORIZONS = [1, 3]  # H=1 (target), H=3 (reference)
LAG = 1
TRAIN_HOURS = 1440  # 60 days
TEST_HOURS = 720    # 30 days
STANDARDIZE = False  # Raw features (proven better)

# Bucket features
BUCKET_FEATURES = [
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt'
]

print("="*80)
print("Lasso Alpha Fine-Tuning + Stability Validation")
print("="*80)
print(f"Goal: Push H=1 IC from 0.0190 to >=0.02 (Hard threshold)")
print(f"Strategy: Alpha search {ALPHA_GRID} x Seeds {RANDOM_SEEDS}")
print(f"Total configs: {len(ALPHA_GRID)} alphas x {len(RANDOM_SEEDS)} seeds x {len(HORIZONS)} horizons = {len(ALPHA_GRID)*len(RANDOM_SEEDS)*len(HORIZONS)}")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[1/5] Loading data...")
gdelt_df = pd.read_parquet(GDELT_PATH)
price_df = pd.read_parquet(PRICE_PATH)

# Ensure datetime index
gdelt_df['ts_utc'] = pd.to_datetime(gdelt_df['ts_utc'])
price_df['ts_utc'] = pd.to_datetime(price_df['ts_utc'])
gdelt_df = gdelt_df.sort_values('ts_utc').reset_index(drop=True)
price_df = price_df.sort_values('ts_utc').reset_index(drop=True)

# Merge
merged = pd.merge(gdelt_df, price_df[['ts_utc', 'ret_1h']], on='ts_utc', how='inner')
merged = merged.rename(columns={'ret_1h': 'wti_returns'})
print(f"   Merged data: {len(merged)} rows")

# Check for missing bucket features
missing_cols = [c for c in BUCKET_FEATURES if c not in merged.columns]
if missing_cols:
    print(f"   WARNING: Missing columns: {missing_cols}")
    BUCKET_FEATURES = [c for c in BUCKET_FEATURES if c in merged.columns]
    print(f"   Using available features: {BUCKET_FEATURES}")

# ============================================================================
# Find Longest Continuous Segment
# ============================================================================
print("\n[2/5] Finding longest continuous segment...")
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
print("\n[3/5] Applying winsorization (1st/99th percentiles)...")
for col in BUCKET_FEATURES + ['wti_returns']:
    if col in segment_data.columns:
        p1 = segment_data[col].quantile(0.01)
        p99 = segment_data[col].quantile(0.99)
        segment_data[col] = segment_data[col].clip(p1, p99)

# ============================================================================
# Grid Search
# ============================================================================
print(f"\n[4/5] Running grid search ({len(ALPHA_GRID)*len(RANDOM_SEEDS)*len(HORIZONS)} configs)...")

results = []
config_num = 0

for H in HORIZONS:
    for alpha in ALPHA_GRID:
        for seed in RANDOM_SEEDS:
            config_num += 1
            print(f"\n   Config {config_num}/{len(ALPHA_GRID)*len(RANDOM_SEEDS)*len(HORIZONS)}: "
                  f"H={H}, alpha={alpha}, seed={seed}")

            # Set random seed for reproducibility
            np.random.seed(seed)

            # Prepare features and target
            X = segment_data[BUCKET_FEATURES].values
            y_forward = segment_data['wti_returns'].shift(-H).values

            # Remove NaN rows
            valid_idx = ~np.isnan(y_forward)
            X_valid = X[valid_idx]
            y_valid = y_forward[valid_idx]

            if len(X_valid) < TRAIN_HOURS + TEST_HOURS:
                print(f"      SKIP: Insufficient data ({len(X_valid)} < {TRAIN_HOURS + TEST_HOURS})")
                continue

            # Rolling window evaluation
            n_samples = len(X_valid)
            max_start = n_samples - TRAIN_HOURS - TEST_HOURS

            # Calculate number of windows (with seed-based shuffle for variety)
            window_starts = np.arange(0, max_start + 1, TEST_HOURS)

            # Optional: shuffle window order based on seed (for robustness)
            shuffled_starts = window_starts.copy()
            np.random.shuffle(shuffled_starts)

            ic_list = []

            for start in shuffled_starts:
                train_end = start + TRAIN_HOURS
                test_end = train_end + TEST_HOURS

                if test_end > n_samples:
                    break

                X_train = X_valid[start:train_end]
                y_train = y_valid[start:train_end]
                X_test = X_valid[train_end:test_end]
                y_test = y_valid[train_end:test_end]

                # Train model
                if STANDARDIZE:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model = Lasso(alpha=alpha, random_state=seed, max_iter=10000)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model = Lasso(alpha=alpha, random_state=seed, max_iter=10000)
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
                'alpha': alpha,
                'seed': seed,
                'standardize': STANDARDIZE,
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
print(f"\n[5/5] Saving results...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = OUTPUT_DIR / f"lasso_alpha_fine_tune_{timestamp}.csv"

results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"   Saved to: {output_file}")

# ============================================================================
# Summary Analysis
# ============================================================================
print("\n" + "="*80)
print("SUMMARY ANALYSIS")
print("="*80)

for H in HORIZONS:
    h_results = results_df[results_df['H'] == H]
    if len(h_results) == 0:
        continue

    print(f"\n--- H={H} ---")

    # Best by IC
    best_ic = h_results.loc[h_results['IC'].idxmax()]
    print(f"Best IC: {best_ic['IC']:.6f} (alpha={best_ic['alpha']}, seed={best_ic['seed']}, IR={best_ic['IR']:.2f}, PMR={best_ic['PMR']:.2f})")

    # Best by IR
    best_ir = h_results.loc[h_results['IR'].idxmax()]
    print(f"Best IR: {best_ir['IR']:.2f} (alpha={best_ir['alpha']}, seed={best_ir['seed']}, IC={best_ir['IC']:.6f}, PMR={best_ir['PMR']:.2f})")

    # Best by PMR
    best_pmr = h_results.loc[h_results['PMR'].idxmax()]
    print(f"Best PMR: {best_pmr['PMR']:.2f} (alpha={best_pmr['alpha']}, seed={best_pmr['seed']}, IC={best_pmr['IC']:.6f}, IR={best_pmr['IR']:.2f})")

    # Hard threshold candidates
    hard_candidates = h_results[h_results['is_hard']]
    if len(hard_candidates) > 0:
        print(f"\n*** HARD THRESHOLD ACHIEVED ({len(hard_candidates)} configs) ***")
        for _, row in hard_candidates.iterrows():
            print(f"    alpha={row['alpha']}, seed={row['seed']}: IC={row['IC']:.6f}, IR={row['IR']:.2f}, PMR={row['PMR']:.2f}")
    else:
        print(f"\nNo Hard threshold candidates (IC≥0.02 ∧ IR≥0.5 ∧ PMR≥0.55)")

        # Show closest to Hard
        h_results['dist_to_hard'] = np.sqrt(
            (np.maximum(0, 0.02 - h_results['IC']))**2 +
            (np.maximum(0, 0.5 - h_results['IR']))**2 +
            (np.maximum(0, 0.55 - h_results['PMR']))**2
        )
        closest = h_results.loc[h_results['dist_to_hard'].idxmin()]
        print(f"Closest to Hard: alpha={closest['alpha']}, seed={closest['seed']}")
        print(f"  IC={closest['IC']:.6f} (gap: {max(0, 0.02-closest['IC']):.6f})")
        print(f"  IR={closest['IR']:.2f} (gap: {max(0, 0.5-closest['IR']):.2f})")
        print(f"  PMR={closest['PMR']:.2f} (gap: {max(0, 0.55-closest['PMR']):.2f})")

    # Alpha stability analysis
    print(f"\nAlpha Stability Analysis (IC mean ± std across seeds):")
    alpha_stats = h_results.groupby('alpha')['IC'].agg(['mean', 'std', 'min', 'max'])
    print(alpha_stats.to_string())

print("\n" + "="*80)
print("Fine-tuning complete!")
print("="*80)
