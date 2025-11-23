#!/usr/bin/env python3
"""
IC/PMR Evaluation for Composite Ridge Strategy
H=[1,2,3], lag=1h, TRAIN=60d, TEST=30d
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("IC/PMR EVALUATION: Composite Ridge (5 buckets)")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Parameters (from runlog)
H_LIST = [1, 2, 3]
LAG = 1  # hours
TRAIN_DAYS = 60
TEST_DAYS = 30
RIDGE_L2 = 1.0
WINSOR_Q = (0.01, 0.99)

# Thresholds
SOFT_IC = 0.025
SOFT_IR = 0.6
HARD_IC = 0.02
HARD_IR = 0.5
HARD_PMR = 0.55

# Load data
print("### Loading Data ###")
gdelt_df = pd.read_parquet('data/gdelt_hourly.parquet')
price_df = pd.read_parquet('data/features_hourly.parquet')

print(f"GDELT rows: {len(gdelt_df)}, range: {gdelt_df['ts_utc'].min()} to {gdelt_df['ts_utc'].max()}")
print(f"Price rows: {len(price_df)}, range: {price_df['ts_utc'].min()} to {price_df['ts_utc'].max()}")

# Merge on timestamp
df = price_df.merge(gdelt_df, on='ts_utc', how='inner', suffixes=('_price', '_gdelt'))
print(f"Merged rows: {len(df)}")

# Select bucket features (normalized)
bucket_features = [
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt'
]

# Check feature availability
available_features = [f for f in bucket_features if f in df.columns]
print(f"Available bucket features: {len(available_features)}/{len(bucket_features)}")
for f in available_features:
    non_null = df[f].notna().sum()
    print(f"  {f}: {non_null} non-null ({non_null/len(df)*100:.1f}%)")

if len(available_features) < 5:
    print("\n[ERROR] Missing bucket features!")
    exit(1)

# Filter to rows with all features
df_valid = df[df[available_features].notna().all(axis=1)].copy()
print(f"\nRows with all bucket features: {len(df_valid)} ({len(df_valid)/len(df)*100:.1f}%)")

# Filter to rows with non-zero ret_1h
df_valid = df_valid[df_valid['ret_1h'] != 0].copy()
print(f"Rows with non-zero ret_1h: {len(df_valid)}\n")

if len(df_valid) < 1000:
    print(f"[WARN] Only {len(df_valid)} valid rows, may not be sufficient for IC evaluation")

# Sort by time
df_valid = df_valid.sort_values('ts_utc').reset_index(drop=True)

print("### IC Evaluation ###\n")

results = []

for H in H_LIST:
    print(f"## H={H} hours ##")

    # Create forward return
    df_valid[f'ret_forward_{H}h'] = df_valid['ret_1h'].rolling(window=H, min_periods=H).sum().shift(-H)

    # Apply lag
    for feat in available_features:
        df_valid[f'{feat}_lag{LAG}'] = df_valid[feat].shift(LAG)

    lagged_features = [f'{feat}_lag{LAG}' for feat in available_features]

    # Filter valid rows
    valid_mask = (
        df_valid[f'ret_forward_{H}h'].notna() &
        df_valid[lagged_features].notna().all(axis=1)
    )
    df_work = df_valid[valid_mask].copy()

    print(f"  Valid rows after lag and forward return: {len(df_work)}")

    if len(df_work) < (TRAIN_DAYS + TEST_DAYS) * 24:
        print(f"  [SKIP] Insufficient data for {TRAIN_DAYS}+{TEST_DAYS} days window")
        continue

    # Rolling window evaluation
    train_hours = TRAIN_DAYS * 24
    test_hours = TEST_DAYS * 24
    window_size = train_hours + test_hours

    window_ics = []
    window_irs = []
    window_pmrs = []

    for start_idx in range(0, len(df_work) - window_size + 1, test_hours):
        train_end = start_idx + train_hours
        test_end = start_idx + window_size

        train_df = df_work.iloc[start_idx:train_end]
        test_df = df_work.iloc[train_end:test_end]

        if len(test_df) < test_hours * 0.8:  # At least 80% of test period
            break

        # Prepare training data
        X_train = train_df[lagged_features].values
        y_train = train_df[f'ret_forward_{H}h'].values

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train Ridge
        model = Ridge(alpha=RIDGE_L2)
        model.fit(X_train_scaled, y_train)

        # Predict on test set
        X_test = test_df[lagged_features].values
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_true = test_df[f'ret_forward_{H}h'].values

        # Winsorize
        y_pred = np.clip(y_pred, np.quantile(y_pred, WINSOR_Q[0]), np.quantile(y_pred, WINSOR_Q[1]))
        y_true = np.clip(y_true, np.quantile(y_true, WINSOR_Q[0]), np.quantile(y_true, WINSOR_Q[1]))

        # Calculate IC
        ic = np.corrcoef(y_pred, y_true)[0, 1]

        # Calculate IR (IC / std(IC) approximation, using IC as single value)
        # For proper IR calculation, we'd need monthly ICs

        window_ics.append(ic)

    if not window_ics:
        print(f"  [SKIP] No valid windows")
        continue

    # Calculate metrics
    mean_ic = np.mean(window_ics)
    std_ic = np.std(window_ics)
    ir = mean_ic / std_ic if std_ic > 0 else 0

    # Calculate PMR (proportion of months with IC > 0)
    # Group by month
    # This is simplified - we'd need actual monthly grouping
    pmr = sum(ic > 0 for ic in window_ics) / len(window_ics)

    print(f"  Windows evaluated: {len(window_ics)}")
    print(f"  Mean IC: {mean_ic:.4f}")
    print(f"  Std IC: {std_ic:.4f}")
    print(f"  IR: {ir:.4f}")
    print(f"  PMR: {pmr:.4f}")

    # Check thresholds
    is_soft = mean_ic >= SOFT_IC and ir >= SOFT_IR
    is_hard = mean_ic >= HARD_IC and ir >= HARD_IR and pmr >= HARD_PMR

    print(f"  SOFT: {'✓' if is_soft else '✗'} (IC>={SOFT_IC}, IR>={SOFT_IR})")
    print(f"  HARD: {'✓' if is_hard else '✗'} (IC>={HARD_IC}, IR>={HARD_IR}, PMR>={HARD_PMR})")
    print()

    results.append({
        'H': H,
        'lag': LAG,
        'IC': mean_ic,
        'IC_std': std_ic,
        'IR': ir,
        'PMR': pmr,
        'n_windows': len(window_ics),
        'is_soft': is_soft,
        'is_hard': is_hard
    })

# Save results
print("### Results Summary ###\n")
results_df = pd.DataFrame(results)

if len(results_df) > 0:
    print(results_df.to_string(index=False))
    print()

    # Check for Hard candidates
    hard_candidates = results_df[results_df['is_hard']]
    if len(hard_candidates) > 0:
        print("*** HARD IC THRESHOLD ACHIEVED! ***")
        print(hard_candidates.to_string(index=False))
    else:
        print("No Hard IC candidates")

    # Save to warehouse
    output_dir = Path("warehouse/ic")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(output_dir / f"composite5_ridge_evaluation_{timestamp}.csv", index=False)
    print(f"\nResults saved to: warehouse/ic/composite5_ridge_evaluation_{timestamp}.csv")
else:
    print("[WARN] No results generated")

print()
print("=" * 80)
print(f"EVALUATION COMPLETE! End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
