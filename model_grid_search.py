#!/usr/bin/env python3
"""
Model Hyperparameter Grid Search for IC Optimization
Tests: Ridge/Lasso/ElasticNet with various alphas
       Standardized vs Raw features
Goal: IC from -0.0005 → positive, break Hard threshold
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
import sys
import io
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("MODEL HYPERPARAMETER GRID SEARCH")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Parameters
H_LIST = [1, 2, 3]
LAG = 1
TRAIN_DAYS = 60
TEST_DAYS = 30
WINSOR_Q = (0.01, 0.99)

# Thresholds
HARD_IC = 0.02
HARD_IR = 0.5
HARD_PMR = 0.55

# Grid Search Space
ALPHA_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # Ridge/Lasso alphas
L1_RATIO_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]  # ElasticNet l1_ratio
STANDARDIZE_OPTIONS = [True, False]  # With/without standardization

# Load data
print("### Loading Data ###")
gdelt_df = pd.read_parquet('data/gdelt_hourly.parquet')
price_df = pd.read_parquet('data/features_hourly.parquet')

print(f"GDELT rows: {len(gdelt_df)}")
print(f"Price rows: {len(price_df)}")

df = price_df.merge(gdelt_df, on='ts_utc', how='inner', suffixes=('_price', '_gdelt'))
print(f"Merged rows: {len(df)}\n")

bucket_features = [
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt'
]

df_valid = df[df[bucket_features].notna().all(axis=1)].copy()
print(f"Rows with all bucket features: {len(df_valid)} ({len(df_valid)/len(df)*100:.1f}%)\n")

df_valid = df_valid.sort_values('ts_utc').reset_index(drop=True)

# Find longest continuous segment
time_diffs = df_valid['ts_utc'].diff()
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
print(f"  From {df_valid.iloc[longest_segment[0]]['ts_utc']} to {df_valid.iloc[longest_segment[1]]['ts_utc']}\n")

# Grid Search Results
all_results = []

print("### Starting Grid Search ###\n")
total_configs = len(H_LIST) * (
    len(ALPHA_GRID) * len(STANDARDIZE_OPTIONS) * 2 +  # Ridge + Lasso
    len(ALPHA_GRID) * len(L1_RATIO_GRID) * len(STANDARDIZE_OPTIONS)  # ElasticNet
)
print(f"Total configurations to test: {total_configs}\n")

config_counter = 0

for H in H_LIST:
    print(f"## H={H} hours ##\n")

    # Create forward return
    df_valid[f'ret_forward_{H}h'] = df_valid['ret_1h'].rolling(window=H, min_periods=H).sum().shift(-H)

    # Apply lag
    for feat in bucket_features:
        df_valid[f'{feat}_lag{LAG}'] = df_valid[feat].shift(LAG)

    lagged_features = [f'{feat}_lag{LAG}' for feat in bucket_features]

    # Use longest continuous segment
    start_idx, end_idx, segment_length = longest_segment
    df_segment = df_valid.iloc[start_idx:end_idx+1].copy()

    valid_mask = (
        df_segment[f'ret_forward_{H}h'].notna() &
        df_segment[lagged_features].notna().all(axis=1) &
        df_segment['ret_1h'].notna()
    )
    df_work = df_segment[valid_mask].copy()

    print(f"Valid rows after lag/forward: {len(df_work)}")

    if len(df_work) < 500:
        print(f"[SKIP] Insufficient data ({len(df_work)} < 500 hours)\n")
        continue

    # Adjusted window
    train_hours = min(TRAIN_DAYS * 24, int(len(df_work) * 0.6))
    test_hours = min(TEST_DAYS * 24, int(len(df_work) * 0.3))
    window_size = train_hours + test_hours

    print(f"Adjusted window: train={train_hours}h, test={test_hours}h")
    print(f"Testing {len(ALPHA_GRID)*2 + len(ALPHA_GRID)*len(L1_RATIO_GRID)} model configs × {len(STANDARDIZE_OPTIONS)} standardization options\n")

    # Test Ridge models
    for alpha in ALPHA_GRID:
        for standardize in STANDARDIZE_OPTIONS:
            config_counter += 1
            model_name = f"Ridge(α={alpha}, std={standardize})"
            print(f"[{config_counter}/{total_configs}] Testing {model_name}...", end=' ')

            window_ics = []

            for start in range(0, len(df_work) - window_size + 1, test_hours):
                train_end = start + train_hours
                test_end = start + window_size

                train_df = df_work.iloc[start:train_end]
                test_df = df_work.iloc[train_end:test_end]

                if len(test_df) < test_hours * 0.8:
                    break

                X_train = train_df[lagged_features].values
                y_train = train_df[f'ret_forward_{H}h'].values
                X_test = test_df[lagged_features].values
                y_true = test_df[f'ret_forward_{H}h'].values

                if standardize:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                model = Ridge(alpha=alpha)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                y_pred = np.clip(y_pred, np.quantile(y_pred, WINSOR_Q[0]), np.quantile(y_pred, WINSOR_Q[1]))
                y_true = np.clip(y_true, np.quantile(y_true, WINSOR_Q[0]), np.quantile(y_true, WINSOR_Q[1]))

                if len(y_pred) > 1 and np.std(y_pred) > 0 and np.std(y_true) > 0:
                    ic = np.corrcoef(y_pred, y_true)[0, 1]
                    if not np.isnan(ic):
                        window_ics.append(ic)

            if window_ics:
                mean_ic = np.mean(window_ics)
                std_ic = np.std(window_ics)
                ir = mean_ic / std_ic if std_ic > 0 else 0
                pmr = sum(ic > 0 for ic in window_ics) / len(window_ics)
                is_hard = mean_ic >= HARD_IC and ir >= HARD_IR and pmr >= HARD_PMR

                print(f"IC={mean_ic:.4f}, IR={ir:.2f}, PMR={pmr:.2f} {'✓ HARD!' if is_hard else ''}")

                all_results.append({
                    'H': H,
                    'model': 'Ridge',
                    'alpha': alpha,
                    'l1_ratio': np.nan,
                    'standardize': standardize,
                    'IC': mean_ic,
                    'IC_std': std_ic,
                    'IR': ir,
                    'PMR': pmr,
                    'n_windows': len(window_ics),
                    'is_hard': is_hard
                })
            else:
                print("No valid windows")

    # Test Lasso models
    for alpha in ALPHA_GRID:
        for standardize in STANDARDIZE_OPTIONS:
            config_counter += 1
            model_name = f"Lasso(α={alpha}, std={standardize})"
            print(f"[{config_counter}/{total_configs}] Testing {model_name}...", end=' ')

            window_ics = []

            for start in range(0, len(df_work) - window_size + 1, test_hours):
                train_end = start + train_hours
                test_end = start + window_size

                train_df = df_work.iloc[start:train_end]
                test_df = df_work.iloc[train_end:test_end]

                if len(test_df) < test_hours * 0.8:
                    break

                X_train = train_df[lagged_features].values
                y_train = train_df[f'ret_forward_{H}h'].values
                X_test = test_df[lagged_features].values
                y_true = test_df[f'ret_forward_{H}h'].values

                if standardize:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                model = Lasso(alpha=alpha, max_iter=10000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                y_pred = np.clip(y_pred, np.quantile(y_pred, WINSOR_Q[0]), np.quantile(y_pred, WINSOR_Q[1]))
                y_true = np.clip(y_true, np.quantile(y_true, WINSOR_Q[0]), np.quantile(y_true, WINSOR_Q[1]))

                if len(y_pred) > 1 and np.std(y_pred) > 0 and np.std(y_true) > 0:
                    ic = np.corrcoef(y_pred, y_true)[0, 1]
                    if not np.isnan(ic):
                        window_ics.append(ic)

            if window_ics:
                mean_ic = np.mean(window_ics)
                std_ic = np.std(window_ics)
                ir = mean_ic / std_ic if std_ic > 0 else 0
                pmr = sum(ic > 0 for ic in window_ics) / len(window_ics)
                is_hard = mean_ic >= HARD_IC and ir >= HARD_IR and pmr >= HARD_PMR

                print(f"IC={mean_ic:.4f}, IR={ir:.2f}, PMR={pmr:.2f} {'✓ HARD!' if is_hard else ''}")

                all_results.append({
                    'H': H,
                    'model': 'Lasso',
                    'alpha': alpha,
                    'l1_ratio': np.nan,
                    'standardize': standardize,
                    'IC': mean_ic,
                    'IC_std': std_ic,
                    'IR': ir,
                    'PMR': pmr,
                    'n_windows': len(window_ics),
                    'is_hard': is_hard
                })
            else:
                print("No valid windows")

    # Test ElasticNet models
    for alpha in ALPHA_GRID:
        for l1_ratio in L1_RATIO_GRID:
            for standardize in STANDARDIZE_OPTIONS:
                config_counter += 1
                model_name = f"ElasticNet(α={alpha}, l1={l1_ratio}, std={standardize})"
                print(f"[{config_counter}/{total_configs}] Testing {model_name}...", end=' ')

                window_ics = []

                for start in range(0, len(df_work) - window_size + 1, test_hours):
                    train_end = start + train_hours
                    test_end = start + window_size

                    train_df = df_work.iloc[start:train_end]
                    test_df = df_work.iloc[train_end:test_end]

                    if len(test_df) < test_hours * 0.8:
                        break

                    X_train = train_df[lagged_features].values
                    y_train = train_df[f'ret_forward_{H}h'].values
                    X_test = test_df[lagged_features].values
                    y_true = test_df[f'ret_forward_{H}h'].values

                    if standardize:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    y_pred = np.clip(y_pred, np.quantile(y_pred, WINSOR_Q[0]), np.quantile(y_pred, WINSOR_Q[1]))
                    y_true = np.clip(y_true, np.quantile(y_true, WINSOR_Q[0]), np.quantile(y_true, WINSOR_Q[1]))

                    if len(y_pred) > 1 and np.std(y_pred) > 0 and np.std(y_true) > 0:
                        ic = np.corrcoef(y_pred, y_true)[0, 1]
                        if not np.isnan(ic):
                            window_ics.append(ic)

                if window_ics:
                    mean_ic = np.mean(window_ics)
                    std_ic = np.std(window_ics)
                    ir = mean_ic / std_ic if std_ic > 0 else 0
                    pmr = sum(ic > 0 for ic in window_ics) / len(window_ics)
                    is_hard = mean_ic >= HARD_IC and ir >= HARD_IR and pmr >= HARD_PMR

                    print(f"IC={mean_ic:.4f}, IR={ir:.2f}, PMR={pmr:.2f} {'✓ HARD!' if is_hard else ''}")

                    all_results.append({
                        'H': H,
                        'model': 'ElasticNet',
                        'alpha': alpha,
                        'l1_ratio': l1_ratio,
                        'standardize': standardize,
                        'IC': mean_ic,
                        'IC_std': std_ic,
                        'IR': ir,
                        'PMR': pmr,
                        'n_windows': len(window_ics),
                        'is_hard': is_hard
                    })
                else:
                    print("No valid windows")

    print()

# Analyze Results
print("\n" + "=" * 80)
print("GRID SEARCH RESULTS SUMMARY")
print("=" * 80 + "\n")

results_df = pd.DataFrame(all_results)

if len(results_df) > 0:
    # Save full results
    output_dir = Path("warehouse/ic")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(output_dir / f"model_grid_search_{timestamp}.csv", index=False)
    print(f"Full results saved to: warehouse/ic/model_grid_search_{timestamp}.csv\n")

    # Check for Hard candidates
    hard_candidates = results_df[results_df['is_hard']]
    if len(hard_candidates) > 0:
        print("🎉 *** HARD IC THRESHOLD ACHIEVED! *** 🎉\n")
        print(hard_candidates.sort_values('IC', ascending=False)[['H', 'model', 'alpha', 'l1_ratio', 'standardize', 'IC', 'IR', 'PMR']].to_string(index=False))
    else:
        print("No Hard IC candidates found.\n")

    # Show Top 10 configurations
    print("\n### Top 10 Configurations by IC ###\n")
    top10 = results_df.nlargest(10, 'IC')[['H', 'model', 'alpha', 'l1_ratio', 'standardize', 'IC', 'IR', 'PMR', 'is_hard']]
    print(top10.to_string(index=False))

    # Best per horizon
    print("\n### Best Configuration per Horizon ###\n")
    for h in H_LIST:
        best = results_df[results_df['H'] == h].nlargest(1, 'IC')
        if len(best) > 0:
            row = best.iloc[0]
            print(f"H={h}: {row['model']}(α={row['alpha']}, l1={row['l1_ratio'] if not pd.isna(row['l1_ratio']) else 'N/A'}, std={row['standardize']}) → IC={row['IC']:.4f}, PMR={row['PMR']:.2f}")

    # Standardize vs Raw comparison
    print("\n### Standardization Impact ###\n")
    std_true = results_df[results_df['standardize'] == True]['IC'].mean()
    std_false = results_df[results_df['standardize'] == False]['IC'].mean()
    print(f"Average IC with standardization: {std_true:.4f}")
    print(f"Average IC without standardization: {std_false:.4f}")
    print(f"Difference: {std_true - std_false:.4f} ({'Standardization helps' if std_true > std_false else 'Raw features better'})")

    # Model comparison
    print("\n### Model Performance Comparison ###\n")
    model_avg = results_df.groupby('model')['IC'].agg(['mean', 'std', 'max'])
    print(model_avg.to_string())

else:
    print("[WARN] No results generated")

print("\n" + "=" * 80)
print(f"GRID SEARCH COMPLETE! End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
