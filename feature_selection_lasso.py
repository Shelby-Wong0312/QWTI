#!/usr/bin/env python3
"""
Lasso Feature Selection + Retraining
Strategy: Remove zero/negative coefficient features from best Lasso model
Goal: Push H=1 IC from 0.012 to >=0.02
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

# Best alpha values from previous experiments
ALPHA_VALUES = [0.005, 0.01]  # Test both best performers
HORIZONS = [1, 3]
LAG = 1
TRAIN_HOURS = 1440
TEST_HOURS = 720
STANDARDIZE = False  # Raw features proven better

# Original bucket features
BUCKET_FEATURES = [
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt'
]

print("="*80)
print("Lasso Feature Selection + Retraining")
print("="*80)
print(f"Strategy: Remove zero/negative coefficient features")
print(f"Goal: Push H=1 IC from 0.012 to >=0.02")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[1/7] Loading data...")
gdelt_df = pd.read_parquet(GDELT_PATH)
price_df = pd.read_parquet(PRICE_PATH)

gdelt_df['ts_utc'] = pd.to_datetime(gdelt_df['ts_utc'])
price_df['ts_utc'] = pd.to_datetime(price_df['ts_utc'])
gdelt_df = gdelt_df.sort_values('ts_utc').reset_index(drop=True)
price_df = price_df.sort_values('ts_utc').reset_index(drop=True)

merged = pd.merge(gdelt_df, price_df[['ts_utc', 'ret_1h']], on='ts_utc', how='inner')
merged = merged.rename(columns={'ret_1h': 'wti_returns'})
print(f"   Merged data: {len(merged)} rows")

# ============================================================================
# Find Longest Continuous Segment
# ============================================================================
print("\n[2/7] Finding longest continuous segment...")
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
print("\n[3/7] Applying winsorization...")
for col in BUCKET_FEATURES + ['wti_returns']:
    if col in segment_data.columns:
        p1 = segment_data[col].quantile(0.01)
        p99 = segment_data[col].quantile(0.99)
        segment_data[col] = segment_data[col].clip(p1, p99)

# ============================================================================
# Feature Selection via Lasso Coefficients
# ============================================================================
print("\n[4/7] Analyzing Lasso coefficients for feature selection...")

feature_selection_results = {}

for H in HORIZONS:
    print(f"\n--- Horizon H={H} ---")

    # Prepare data
    X = segment_data[BUCKET_FEATURES].values
    y_forward = segment_data['wti_returns'].shift(-H).values
    valid_idx = ~np.isnan(y_forward)
    X_valid = X[valid_idx]
    y_valid = y_forward[valid_idx]

    if len(X_valid) < TRAIN_HOURS + TEST_HOURS:
        print(f"   SKIP: Insufficient data")
        continue

    # Use first window to analyze coefficients
    X_train = X_valid[:TRAIN_HOURS]
    y_train = y_valid[:TRAIN_HOURS]

    for alpha in ALPHA_VALUES:
        print(f"\n  Alpha={alpha}:")

        # Train Lasso
        model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
        model.fit(X_train, y_train)

        # Analyze coefficients
        coeffs = model.coef_
        print(f"    Coefficients:")
        for i, (feat, coef) in enumerate(zip(BUCKET_FEATURES, coeffs)):
            status = "ZERO" if abs(coef) < 1e-10 else ("NEG" if coef < 0 else "POS")
            print(f"      {feat:30s}: {coef:+.6f}  [{status}]")

        # Identify features to keep (positive coefficients only)
        positive_features = [feat for feat, coef in zip(BUCKET_FEATURES, coeffs) if coef > 1e-10]
        removed_features = [feat for feat in BUCKET_FEATURES if feat not in positive_features]

        print(f"\n    Features to KEEP ({len(positive_features)}): {positive_features}")
        print(f"    Features to REMOVE ({len(removed_features)}): {removed_features}")

        # Store for later use
        feature_selection_results[(H, alpha)] = {
            'coefficients': dict(zip(BUCKET_FEATURES, coeffs)),
            'positive_features': positive_features,
            'removed_features': removed_features
        }

# ============================================================================
# Retrain with Selected Features
# ============================================================================
print("\n[5/7] Retraining with selected features...")

comparison_results = []

for H in HORIZONS:
    print(f"\n--- Horizon H={H} ---")

    X_full = segment_data[BUCKET_FEATURES].values
    y_forward = segment_data['wti_returns'].shift(-H).values
    valid_idx = ~np.isnan(y_forward)
    X_valid_full = X_full[valid_idx]
    y_valid = y_forward[valid_idx]

    if len(X_valid_full) < TRAIN_HOURS + TEST_HOURS:
        continue

    for alpha in ALPHA_VALUES:
        print(f"\n  Alpha={alpha}:")

        selected_features = feature_selection_results[(H, alpha)]['positive_features']

        if len(selected_features) == 0:
            print(f"    SKIP: No positive features selected")
            continue

        # Get selected feature indices
        selected_indices = [BUCKET_FEATURES.index(f) for f in selected_features]
        X_valid_selected = X_valid_full[:, selected_indices]

        # Rolling window evaluation (BEFORE feature selection - using all features)
        n_samples = len(X_valid_full)
        max_start = n_samples - TRAIN_HOURS - TEST_HOURS
        window_starts = np.arange(0, max_start + 1, TEST_HOURS)

        ic_list_before = []
        ic_list_after = []

        for start in window_starts:
            train_end = start + TRAIN_HOURS
            test_end = train_end + TEST_HOURS
            if test_end > n_samples:
                break

            # BEFORE: All features
            X_train_full = X_valid_full[start:train_end]
            y_train = y_valid[start:train_end]
            X_test_full = X_valid_full[train_end:test_end]
            y_test = y_valid[train_end:test_end]

            model_before = Lasso(alpha=alpha, random_state=42, max_iter=10000)
            model_before.fit(X_train_full, y_train)
            y_pred_before = model_before.predict(X_test_full)
            ic_before = np.corrcoef(y_pred_before, y_test)[0, 1]
            ic_list_before.append(ic_before)

            # AFTER: Selected features only
            X_train_selected = X_valid_selected[start:train_end]
            X_test_selected = X_valid_selected[train_end:test_end]

            model_after = Lasso(alpha=alpha, random_state=42, max_iter=10000)
            model_after.fit(X_train_selected, y_train)
            y_pred_after = model_after.predict(X_test_selected)
            ic_after = np.corrcoef(y_pred_after, y_test)[0, 1]
            ic_list_after.append(ic_after)

        # Calculate metrics
        ic_list_before = [ic for ic in ic_list_before if not np.isnan(ic)]
        ic_list_after = [ic for ic in ic_list_after if not np.isnan(ic)]

        if len(ic_list_before) == 0 or len(ic_list_after) == 0:
            print(f"    SKIP: No valid windows")
            continue

        # BEFORE metrics
        ic_mean_before = np.mean(ic_list_before)
        ic_std_before = np.std(ic_list_before)
        ir_before = ic_mean_before / ic_std_before if ic_std_before > 0 else 0
        pmr_before = np.mean([ic > 0 for ic in ic_list_before])

        # AFTER metrics
        ic_mean_after = np.mean(ic_list_after)
        ic_std_after = np.std(ic_list_after)
        ir_after = ic_mean_after / ic_std_after if ic_std_after > 0 else 0
        pmr_after = np.mean([ic > 0 for ic in ic_list_after])

        # Hard threshold check
        is_hard_after = (ic_mean_after >= 0.02) and (ir_after >= 0.5) and (pmr_after >= 0.55)

        print(f"    BEFORE (all {len(BUCKET_FEATURES)} features):")
        print(f"      IC={ic_mean_before:.6f}, IR={ir_before:.2f}, PMR={pmr_before:.2f}, n={len(ic_list_before)}")
        print(f"    AFTER (selected {len(selected_features)} features):")
        print(f"      IC={ic_mean_after:.6f}, IR={ir_after:.2f}, PMR={pmr_after:.2f}, n={len(ic_list_after)}")
        print(f"      Hard threshold: {is_hard_after}")
        print(f"    DELTA:")
        print(f"      IC: {ic_mean_after-ic_mean_before:+.6f} ({(ic_mean_after/ic_mean_before-1)*100 if ic_mean_before!=0 else 0:+.1f}%)")
        print(f"      IR: {ir_after-ir_before:+.2f}")
        print(f"      PMR: {pmr_after-pmr_before:+.2f}")

        comparison_results.append({
            'H': H,
            'alpha': alpha,
            'n_features_before': len(BUCKET_FEATURES),
            'n_features_after': len(selected_features),
            'features_removed': ','.join(feature_selection_results[(H, alpha)]['removed_features']),
            'IC_before': ic_mean_before,
            'IR_before': ir_before,
            'PMR_before': pmr_before,
            'IC_after': ic_mean_after,
            'IR_after': ir_after,
            'PMR_after': pmr_after,
            'IC_delta': ic_mean_after - ic_mean_before,
            'IC_delta_pct': (ic_mean_after/ic_mean_before-1)*100 if ic_mean_before!=0 else 0,
            'n_windows': len(ic_list_after),
            'is_hard_after': is_hard_after
        })

# ============================================================================
# Save Results
# ============================================================================
print(f"\n[6/7] Saving results...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = OUTPUT_DIR / f"feature_selection_lasso_{timestamp}.csv"

results_df = pd.DataFrame(comparison_results)
results_df.to_csv(output_file, index=False)
print(f"   Saved to: {output_file}")

# ============================================================================
# Summary
# ============================================================================
print("\n[7/7] " + "="*80)
print("SUMMARY: Feature Selection Impact")
print("="*80)

for H in HORIZONS:
    h_results = results_df[results_df['H'] == H]
    if len(h_results) == 0:
        continue

    print(f"\n--- H={H} ---")

    for _, row in h_results.iterrows():
        print(f"\nAlpha={row['alpha']}:")
        print(f"  Features: {row['n_features_before']} -> {row['n_features_after']} (removed: {row['features_removed']})")
        print(f"  IC:  {row['IC_before']:.6f} -> {row['IC_after']:.6f}  (delta: {row['IC_delta']:+.6f} / {row['IC_delta_pct']:+.1f}%)")
        print(f"  IR:  {row['IR_before']:.2f} -> {row['IR_after']:.2f}")
        print(f"  PMR: {row['PMR_before']:.2f} -> {row['PMR_after']:.2f}")
        print(f"  Hard threshold achieved: {row['is_hard_after']}")

# Check if any configuration achieved Hard threshold
hard_achievers = results_df[results_df['is_hard_after']]
if len(hard_achievers) > 0:
    print(f"\n*** HARD THRESHOLD ACHIEVED ({len(hard_achievers)} configs) ***")
    for _, row in hard_achievers.iterrows():
        print(f"  H={row['H']}, alpha={row['alpha']}: IC={row['IC_after']:.6f}, IR={row['IR_after']:.2f}, PMR={row['PMR_after']:.2f}")
else:
    print(f"\nNo Hard threshold achieved (IC>=0.02 AND IR>=0.5 AND PMR>=0.55)")

    # Show best improvement
    best_ic = results_df.loc[results_df['IC_after'].idxmax()]
    print(f"\nBest IC after selection:")
    print(f"  H={best_ic['H']}, alpha={best_ic['alpha']}: IC={best_ic['IC_after']:.6f} (was {best_ic['IC_before']:.6f})")
    print(f"  Features kept: {best_ic['n_features_after']}/{best_ic['n_features_before']}")
    print(f"  Removed: {best_ic['features_removed']}")

print("\n" + "="*80)
print("Feature selection complete!")
print("="*80)
