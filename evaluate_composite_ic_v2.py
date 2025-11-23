#!/usr/bin/env python3
"""
IC/PMR Evaluation for Composite Ridge Strategy (v2 - Keep all data points)
H=[1,2,3], lag=1h, TRAIN=60d, TEST=30d
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import sys
import io
import warnings
warnings.filterwarnings('ignore')

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("IC/PMR EVALUATION v2: Composite Ridge (Keep all timepoints)")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Parameters
H_LIST = [1, 2, 3]
LAG = 1  # hours
TRAIN_DAYS = 60
TEST_DAYS = 30
RIDGE_L2 = 1.0
WINSOR_Q = (0.01, 0.99)

# Thresholds
HARD_IC = 0.02
HARD_IR = 0.5
HARD_PMR = 0.55

# Load data
print("### Loading Data ###")
gdelt_df = pd.read_parquet('data/gdelt_hourly.parquet')
price_df = pd.read_parquet('data/features_hourly.parquet')

print(f"GDELT rows: {len(gdelt_df)}")
print(f"Price rows: {len(price_df)}")

# Merge
df = price_df.merge(gdelt_df, on='ts_utc', how='inner', suffixes=('_price', '_gdelt'))
print(f"Merged rows: {len(df)}")

# Bucket features
bucket_features = [
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt'
]

# Filter to rows with all bucket features (DON'T filter ret_1h)
df_valid = df[df[bucket_features].notna().all(axis=1)].copy()
print(f"Rows with all bucket features: {len(df_valid)} ({len(df_valid)/len(df)*100:.1f}%)\n")

# Sort by time
df_valid = df_valid.sort_values('ts_utc').reset_index(drop=True)

# Check continuous coverage
time_diffs = df_valid['ts_utc'].diff()
hourly_continuous = (time_diffs == pd.Timedelta(hours=1)).sum()
print(f"Hourly continuous timestamps: {hourly_continuous}/{len(df_valid)-1} ({hourly_continuous/(len(df_valid)-1)*100:.1f}%)")

# Find longest continuous segment
segments = []
current_start = 0
for i in range(1, len(df_valid)):
    if time_diffs.iloc[i] != pd.Timedelta(hours=1):
        if i - current_start > 0:
            segments.append((current_start, i-1, i-current_start))
        current_start = i
segments.append((current_start, len(df_valid)-1, len(df_valid)-current_start))

segments.sort(key=lambda x: x[2], reverse=True)
longest_segment = segments[0]
print(f"Longest continuous segment: {longest_segment[2]} hours ({longest_segment[2]/24:.1f} days)")
print(f"  From {df_valid.iloc[longest_segment[0]]['ts_utc']} to {df_valid.iloc[longest_segment[1]]['ts_utc']}")
print()

if longest_segment[2] < (TRAIN_DAYS + TEST_DAYS) * 24:
    print(f"[WARN] Longest segment ({longest_segment[2]/24:.1f} days) < required ({TRAIN_DAYS+TEST_DAYS} days)")
    print(f"[INFO] Will use the longest available segment for evaluation\n")

print("### IC Evaluation ###\n")

results = []

for H in H_LIST:
    print(f"## H={H} hours ##")

    # Create forward return (for all timepoints)
    df_valid[f'ret_forward_{H}h'] = df_valid['ret_1h'].rolling(window=H, min_periods=H).sum().shift(-H)

    # Apply lag
    for feat in bucket_features:
        df_valid[f'{feat}_lag{LAG}'] = df_valid[feat].shift(LAG)

    lagged_features = [f'{feat}_lag{LAG}' for feat in bucket_features]

    # Use longest continuous segment
    start_idx, end_idx, segment_length = longest_segment
    df_segment = df_valid.iloc[start_idx:end_idx+1].copy()

    # Filter valid rows within segment
    valid_mask = (
        df_segment[f'ret_forward_{H}h'].notna() &
        df_segment[lagged_features].notna().all(axis=1) &
        df_segment['ret_1h'].notna()  # Ensure current ret_1h exists
    )
    df_work = df_segment[valid_mask].copy()

    print(f"  Longest segment length: {segment_length} hours ({segment_length/24:.1f} days)")
    print(f"  Valid rows after lag/forward: {len(df_work)}")

    if len(df_work) < 500:
        print(f"  [SKIP] Insufficient valid data ({len(df_work)} < 500 hours)")
        continue

    # Use available data for rolling windows (smaller window if needed)
    train_hours = min(TRAIN_DAYS * 24, int(len(df_work) * 0.6))
    test_hours = min(TEST_DAYS * 24, int(len(df_work) * 0.3))
    window_size = train_hours + test_hours

    print(f"  Adjusted window: train={train_hours}h ({train_hours/24:.1f}d), test={test_hours}h ({test_hours/24:.1f}d)")

    window_ics = []
    window_dates = []

    for start in range(0, len(df_work) - window_size + 1, test_hours):
        train_end = start + train_hours
        test_end = start + window_size

        train_df = df_work.iloc[start:train_end]
        test_df = df_work.iloc[train_end:test_end]

        if len(test_df) < test_hours * 0.8:
            break

        # Prepare data
        X_train = train_df[lagged_features].values
        y_train = train_df[f'ret_forward_{H}h'].values

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train Ridge
        model = Ridge(alpha=RIDGE_L2)
        model.fit(X_train_scaled, y_train)

        # Predict
        X_test = test_df[lagged_features].values
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_true = test_df[f'ret_forward_{H}h'].values

        # Winsorize
        y_pred = np.clip(y_pred, np.quantile(y_pred, WINSOR_Q[0]), np.quantile(y_pred, WINSOR_Q[1]))
        y_true = np.clip(y_true, np.quantile(y_true, WINSOR_Q[0]), np.quantile(y_true, WINSOR_Q[1]))

        # Calculate IC
        if len(y_pred) > 1 and len(y_true) > 1 and np.std(y_pred) > 0 and np.std(y_true) > 0:
            ic = np.corrcoef(y_pred, y_true)[0, 1]
            if not np.isnan(ic):
                window_ics.append(ic)
                window_dates.append(test_df.iloc[0]['ts_utc'])

    if not window_ics:
        print(f"  [SKIP] No valid windows")
        continue

    # Calculate metrics
    mean_ic = np.mean(window_ics)
    std_ic = np.std(window_ics)
    ir = mean_ic / std_ic if std_ic > 0 else 0
    pmr = sum(ic > 0 for ic in window_ics) / len(window_ics)

    print(f"  Windows evaluated: {len(window_ics)}")
    print(f"  Mean IC: {mean_ic:.4f}")
    print(f"  Std IC: {std_ic:.4f}")
    print(f"  IR: {ir:.4f}")
    print(f"  PMR: {pmr:.4f}")

    # Check Hard threshold
    is_hard = mean_ic >= HARD_IC and ir >= HARD_IR and pmr >= HARD_PMR

    print(f"  HARD: {'✓ ACHIEVED!' if is_hard else '✗'} (IC>={HARD_IC}, IR>={HARD_IR}, PMR>={HARD_PMR})")
    print()

    results.append({
        'H': H,
        'lag': LAG,
        'IC': mean_ic,
        'IC_std': std_ic,
        'IR': ir,
        'PMR': pmr,
        'n_windows': len(window_ics),
        'is_hard': is_hard,
        'segment_days': segment_length / 24,
        'train_hours': train_hours,
        'test_hours': test_hours
    })

# Save results
print("### Results Summary ###\n")
results_df = pd.DataFrame(results)

if len(results_df) > 0:
    print(results_df[['H', 'IC', 'IR', 'PMR', 'is_hard', 'n_windows']].to_string(index=False))
    print()

    # Check for Hard candidates
    hard_candidates = results_df[results_df['is_hard']]
    if len(hard_candidates) > 0:
        print("🎉 *** HARD IC THRESHOLD ACHIEVED FOR THE FIRST TIME! *** 🎉")
        print(hard_candidates[['H', 'IC', 'IR', 'PMR']].to_string(index=False))
    else:
        print("No Hard IC candidates yet")
        print("\nClosest to Hard threshold:")
        results_df['distance_to_hard'] = (
            (results_df['IC'] - HARD_IC) +
            (results_df['IR'] - HARD_IR) +
            (results_df['PMR'] - HARD_PMR)
        )
        print(results_df.nlargest(3, 'distance_to_hard')[['H', 'IC', 'IR', 'PMR', 'distance_to_hard']].to_string(index=False))

    # Save
    output_dir = Path("warehouse/ic")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(output_dir / f"composite5_ridge_evaluation_v2_{timestamp}.csv", index=False)
    print(f"\nResults saved to: warehouse/ic/composite5_ridge_evaluation_v2_{timestamp}.csv")
else:
    print("[WARN] No results generated")

print()
print("=" * 80)
print(f"EVALUATION COMPLETE! End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
