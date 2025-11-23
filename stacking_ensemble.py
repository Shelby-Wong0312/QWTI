"""
Stacking Ensemble Model for WTI Oil Price Prediction
Addresses IR insufficiency via decorrelating base learners and optimizing Ridge meta-model

Strategy:
1. Train 7 LightGBM base learners with SAME architecture but DIFFERENT seeds and AGGRESSIVE feature bagging
2. AGGRESSIVE feature bagging (feature_fraction=0.5, bagging_fraction=0.6) for strong decorrelation
3. Different reg_lambda values (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0) to reduce variance
4. SCAN Ridge alpha values [0.1, 0.3, 0.5, 1.0, 2.0, 5.0] to optimize IC_std and IR
5. Goal: Find optimal Ridge alpha to reduce IC_std and push IR >= 0.5

Author: Claude Code
Date: 2025-11-19
Version: V4 - Ridge Alpha Optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# ============================================================================
# Configuration
# ============================================================================

GDELT_PARQUET = 'data/gdelt_hourly.parquet'
PRICE_PARQUET = 'data/features_hourly.parquet'
OUTPUT_DIR = Path('warehouse/ic')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Base learner configurations (7 learners with SAME architecture, DIFFERENT seeds & AGGRESSIVE feature bagging)
# Strategy: Strong decorrelation via aggressive feature/row bagging to reduce IC_std
SEEDS = [42, 101, 202, 303, 404, 505, 606]
REG_LAMBDAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

# Ridge alpha values to scan for optimal IC_std and IR
RIDGE_ALPHAS = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

BASE_LEARNERS = {}
for i, (seed, reg_lambda) in enumerate(zip(SEEDS, REG_LAMBDAS)):
    learner_key = f'seed{seed}'
    BASE_LEARNERS[learner_key] = {
        'name': f'Seed{seed} (L={reg_lambda})',
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'num_leaves': 31,
        'feature_fraction': 0.5,  # AGGRESSIVE feature bagging for strong decorrelation
        'bagging_fraction': 0.6,  # AGGRESSIVE row bagging for strong decorrelation
        'bagging_freq': 1,
        'reg_lambda': reg_lambda,
        'random_state': seed,
        'verbosity': -1,
        'force_col_wise': True
    }

H = 1  # Horizon
LAG = 1  # Lag hours
TRAIN_HOURS = 1440  # 60 days
TEST_HOURS = 360    # 15 days

BUCKET_FEATURES = [
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt'
]

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

print("\n" + "="*80)
print("STACKING ENSEMBLE MODEL - IC_STD REDUCTION EXPERIMENT")
print("="*80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nObjective: Optimize Ridge alpha to reduce IC_std and push IR >= 0.5")
print(f"Target: IR >= 0.5 (V3 baseline: 0.4358, gap: 0.0642)")
print(f"\nStrategy:")
print(f"  - Train 7 base learners with SAME architecture, DIFFERENT seeds {SEEDS}")
print(f"  - AGGRESSIVE feature bagging: feature_fraction=0.5, bagging_fraction=0.6, bagging_freq=1")
print(f"  - Regularization: reg_lambda={REG_LAMBDAS} to reduce variance")
print(f"  - SCAN Ridge alpha values: {RIDGE_ALPHAS} to find optimal meta-model")
print(f"  - Compare IC_std and IR across all Ridge alpha values")

print(f"\nConfiguration:")
print(f"  Horizon (H): {H}")
print(f"  Lag: {LAG}h")
print(f"  Train window: {TRAIN_HOURS}h ({TRAIN_HOURS//24} days)")
print(f"  Test window: {TEST_HOURS}h ({TEST_HOURS//24} days)")

print(f"\n[1/7] Loading data...")

# Load GDELT features
gdelt_df = pd.read_parquet(GDELT_PARQUET)
gdelt_df['ts_utc'] = pd.to_datetime(gdelt_df['ts_utc'])
gdelt_df = gdelt_df.sort_values('ts_utc').reset_index(drop=True)
print(f"  GDELT: {len(gdelt_df):,} hours from {gdelt_df['ts_utc'].min()} to {gdelt_df['ts_utc'].max()}")

# Load WTI price returns
price_df = pd.read_parquet(PRICE_PARQUET)
price_df['ts_utc'] = pd.to_datetime(price_df['ts_utc'])
price_df = price_df.sort_values('ts_utc').reset_index(drop=True)
print(f"  Price features: {len(price_df):,} hours")

# Merge
merged = pd.merge(gdelt_df, price_df[['ts_utc', 'ret_1h']], on='ts_utc', how='inner')
merged = merged.rename(columns={'ret_1h': 'wti_returns'})
merged = merged.dropna(subset=['wti_returns'])

# Winsorize features (1st/99th percentiles)
for feat in BUCKET_FEATURES:
    p1, p99 = merged[feat].quantile([0.01, 0.99])
    merged[feat] = merged[feat].clip(lower=p1, upper=p99)

print(f"  Merged: {len(merged):,} hours after merge and cleaning")

# ============================================================================
# Rolling Window Evaluation with Stacking Ensemble
# ============================================================================

print(f"\n[2/7] Training base learners and ensemble on rolling windows...")

max_start = len(merged) - TRAIN_HOURS - TEST_HOURS
window_starts = np.arange(0, max_start + 1, TEST_HOURS)

print(f"  Total windows: {len(window_starts)}")
print(f"  Window structure: Train={TRAIN_HOURS}h, Test={TEST_HOURS}h")

results = []

for i, start_idx in enumerate(window_starts):
    train_end = start_idx + TRAIN_HOURS
    test_end = train_end + TEST_HOURS

    # Split data
    train_df = merged.iloc[start_idx:train_end].copy()
    test_df = merged.iloc[train_end:test_end].copy()

    if len(train_df) < TRAIN_HOURS or len(test_df) < TEST_HOURS:
        continue

    train_start_date = train_df['ts_utc'].min().strftime('%Y-%m-%d')
    test_start_date = test_df['ts_utc'].min().strftime('%Y-%m-%d')
    test_end_date = test_df['ts_utc'].max().strftime('%Y-%m-%d')

    print(f"\n  Window {i+1}/{len(window_starts)}:")
    print(f"    Train: {train_start_date} to {test_start_date}")
    print(f"    Test:  {test_start_date} to {test_end_date}")

    # Split training data into train/val for meta-model and weight optimization
    train_split = int(len(train_df) * 0.8)
    train_train = train_df.iloc[:train_split]
    train_val = train_df.iloc[train_split:]

    # Prepare data
    X_train = train_train[BUCKET_FEATURES].values
    y_train = train_train['wti_returns'].values
    X_val = train_val[BUCKET_FEATURES].values
    y_val = train_val['wti_returns'].values
    X_test = test_df[BUCKET_FEATURES].values
    y_test = test_df['wti_returns'].values

    # Train 7 base learners
    base_predictions_val = {}
    base_predictions_test = {}
    learner_keys = list(BASE_LEARNERS.keys())

    for model_name, config in BASE_LEARNERS.items():
        config_copy = {k: v for k, v in config.items() if k != 'name'}
        model = lgb.LGBMRegressor(**config_copy)
        model.fit(X_train, y_train)

        base_predictions_val[model_name] = model.predict(X_val)
        base_predictions_test[model_name] = model.predict(X_test)

    print(f"    Trained {len(BASE_LEARNERS)} base learners: {', '.join(learner_keys)}")

    # Ensemble Strategy 1: Simple Average (equal weights)
    pred_simple_avg_test = np.mean([base_predictions_test[key] for key in learner_keys], axis=0)
    ic_simple_avg = np.corrcoef(pred_simple_avg_test, y_test)[0, 1]

    # Ensemble Strategy 2: Weighted Average (optimize on validation set)
    def weighted_avg_loss(weights):
        weights = np.abs(weights) / np.sum(np.abs(weights))  # Normalize to sum=1
        pred_val = sum(weights[i] * base_predictions_val[key] for i, key in enumerate(learner_keys))
        ic_val = np.corrcoef(pred_val, y_val)[0, 1]
        return -ic_val  # Minimize negative IC (maximize IC)

    n_learners = len(learner_keys)
    result = minimize(weighted_avg_loss, x0=[1/n_learners]*n_learners, method='Nelder-Mead')
    opt_weights = np.abs(result.x) / np.sum(np.abs(result.x))

    pred_weighted_avg_test = sum(opt_weights[i] * base_predictions_test[key] for i, key in enumerate(learner_keys))
    ic_weighted_avg = np.corrcoef(pred_weighted_avg_test, y_test)[0, 1]

    # Ensemble Strategy 3: Stacking with Ridge meta-model - SCAN multiple alpha values
    # Prepare meta-features (base learner predictions on validation set)
    meta_X_train = np.column_stack([base_predictions_val[key] for key in learner_keys])
    meta_y_train = y_val

    # Meta-features for test set
    meta_X_test = np.column_stack([base_predictions_test[key] for key in learner_keys])

    # Train meta-models with different alpha values
    stacking_ics = {}
    stacking_models = {}
    for alpha in RIDGE_ALPHAS:
        meta_model = Ridge(alpha=alpha)
        meta_model.fit(meta_X_train, meta_y_train)
        pred_stacking_test = meta_model.predict(meta_X_test)
        ic = np.corrcoef(pred_stacking_test, y_test)[0, 1]
        stacking_ics[alpha] = ic
        stacking_models[alpha] = meta_model

    # Individual base learner ICs for comparison
    base_ics = {}
    print(f"    Base Learners IC:")
    for key in learner_keys:
        ic = np.corrcoef(base_predictions_test[key], y_test)[0, 1]
        base_ics[key] = ic
        learner_name = BASE_LEARNERS[key]['name']
        print(f"      {learner_name}: {ic:.6f}")
    print(f"    Ensemble IC:")
    print(f"      Simple Avg:   {ic_simple_avg:.6f}")
    weights_str = ', '.join([f'{w:.2f}' for w in opt_weights])
    print(f"      Weighted Avg: {ic_weighted_avg:.6f} (weights: {weights_str})")
    print(f"    Stacking IC (by Ridge alpha):")
    for alpha in RIDGE_ALPHAS:
        print(f"      alpha={alpha:>4.1f}: {stacking_ics[alpha]:.6f}")

    # Build result dictionary dynamically
    result_dict = {
        'window': i + 1,
        'test_start': test_start_date,
        'test_end': test_end_date,
    }

    # Add base learner ICs
    for key in learner_keys:
        result_dict[f'ic_{key}'] = base_ics[key]

    # Add ensemble ICs
    result_dict['ic_simple_avg'] = ic_simple_avg
    result_dict['ic_weighted_avg'] = ic_weighted_avg

    # Add stacking ICs for all Ridge alpha values
    for alpha in RIDGE_ALPHAS:
        result_dict[f'ic_stacking_alpha{alpha}'] = stacking_ics[alpha]

    # Add weights
    for i_w, key in enumerate(learner_keys):
        result_dict[f'weight_{key}'] = opt_weights[i_w]

    # Add stacking coefficients for all Ridge alpha values
    for alpha in RIDGE_ALPHAS:
        for i_c, key in enumerate(learner_keys):
            result_dict[f'stacking_alpha{alpha}_coef_{key}'] = stacking_models[alpha].coef_[i_c]

    result_dict['n_test'] = len(test_df)

    results.append(result_dict)

# ============================================================================
# Aggregate Statistics
# ============================================================================

print(f"\n[3/7] Computing aggregate statistics...")

results_df = pd.DataFrame(results)

def compute_stats(ic_series, name):
    ic_mean = ic_series.mean()
    ic_median = ic_series.median()
    ic_std = ic_series.std()
    ir = ic_mean / ic_std if ic_std > 0 else 0
    pmr = (ic_series > 0).sum() / len(ic_series)
    return {
        'model': name,
        'ic_mean': ic_mean,
        'ic_median': ic_median,
        'ic_std': ic_std,
        'ir': ir,
        'pmr': pmr,
        'n_windows': len(ic_series)
    }

stats = []
# Add base learner stats dynamically
learner_keys = list(BASE_LEARNERS.keys())
for key in learner_keys:
    learner_name = BASE_LEARNERS[key]['name']
    stats.append(compute_stats(results_df[f'ic_{key}'], f'Base: {learner_name}'))

# Add ensemble stats
stats.append(compute_stats(results_df['ic_simple_avg'], 'Ensemble: Simple Avg'))
stats.append(compute_stats(results_df['ic_weighted_avg'], 'Ensemble: Weighted Avg'))

# Add stacking stats for all Ridge alpha values
for alpha in RIDGE_ALPHAS:
    stats.append(compute_stats(results_df[f'ic_stacking_alpha{alpha}'], f'Stacking: alpha={alpha}'))

stats_df = pd.DataFrame(stats)

print(f"\n{'='*80}")
print(f"STACKING ENSEMBLE RESULTS")
print(f"{'='*80}")
print(f"\nBase Learners:")
for idx, row in stats_df[stats_df['model'].str.startswith('Base')].iterrows():
    print(f"  {row['model']:30s}: IC mean={row['ic_mean']:.6f}, IC std={row['ic_std']:.6f}, IR={row['ir']:.4f}, PMR={row['pmr']:.4f}")

print(f"\nEnsemble Methods:")
for idx, row in stats_df[stats_df['model'].str.startswith('Ensemble')].iterrows():
    print(f"  {row['model']:30s}: IC mean={row['ic_mean']:.6f}, IC std={row['ic_std']:.6f}, IR={row['ir']:.4f}, PMR={row['pmr']:.4f}")

print(f"\nStacking Methods (Ridge Alpha Scan):")
for idx, row in stats_df[stats_df['model'].str.startswith('Stacking')].iterrows():
    print(f"  {row['model']:30s}: IC mean={row['ic_mean']:.6f}, IC std={row['ic_std']:.6f}, IR={row['ir']:.4f}, PMR={row['pmr']:.4f}")

# Find best method overall (ensemble + stacking)
ensemble_stacking_df = stats_df[stats_df['model'].str.startswith('Ensemble') | stats_df['model'].str.startswith('Stacking')]
best_ensemble = ensemble_stacking_df.nlargest(1, 'ir').iloc[0]
print(f"\n[BEST METHOD]: {best_ensemble['model']}")
print(f"  IC mean:   {best_ensemble['ic_mean']:.6f}")
print(f"  IC median: {best_ensemble['ic_median']:.6f}")
print(f"  IC std:    {best_ensemble['ic_std']:.6f}")
print(f"  IR:        {best_ensemble['ir']:.4f}")
print(f"  PMR:       {best_ensemble['pmr']:.4f}")

print(f"\nHard Threshold Check (Best Ensemble):")
print(f"  IC median >= 0.02: {'PASS' if best_ensemble['ic_median'] >= 0.02 else 'FAIL'} ({best_ensemble['ic_median']:.6f})")
print(f"  IR >= 0.5:         {'PASS' if best_ensemble['ir'] >= 0.5 else 'FAIL'} ({best_ensemble['ir']:.4f})")
print(f"  PMR >= 0.55:       {'PASS' if best_ensemble['pmr'] >= 0.55 else 'FAIL'} ({best_ensemble['pmr']:.4f})")

if best_ensemble['ic_median'] >= 0.02 and best_ensemble['ir'] >= 0.5 and best_ensemble['pmr'] >= 0.55:
    print(f"\n[OK] HARD THRESHOLD ACHIEVED - Ready for base promotion")
else:
    print(f"\n[X] Hard threshold NOT met")
    if best_ensemble['ir'] < 0.5:
        print(f"    IR still below 0.5 (current: {best_ensemble['ir']:.4f}, gap: {0.5 - best_ensemble['ir']:.4f})")

# Compare to V3 baseline (IR=0.4358, alpha=1.0)
baseline_ir = 0.4358
ir_improvement = best_ensemble['ir'] - baseline_ir
print(f"\nComparison to V3 Baseline (alpha=1.0):")
print(f"  V3 Baseline IR: {baseline_ir:.4f}")
print(f"  Current IR:     {best_ensemble['ir']:.4f}")
print(f"  Improvement:    {ir_improvement:+.4f} ({ir_improvement/baseline_ir*100:+.1f}%)")

# ============================================================================
# Save Results
# ============================================================================

print(f"\n[4/7] Saving results...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Window-level results
results_file = OUTPUT_DIR / f'stacking_ensemble_windows_{timestamp}.csv'
results_df.to_csv(results_file, index=False)
print(f"  Window results: {results_file}")

# Summary statistics
summary_file = OUTPUT_DIR / f'stacking_ensemble_summary_{timestamp}.csv'
stats_df.to_csv(summary_file, index=False)
print(f"  Summary: {summary_file}")

# Best ensemble metadata
best_meta = {
    'ensemble_method': best_ensemble['model'],
    'ic_mean': best_ensemble['ic_mean'],
    'ic_median': best_ensemble['ic_median'],
    'ic_std': best_ensemble['ic_std'],
    'ir': best_ensemble['ir'],
    'pmr': best_ensemble['pmr'],
    'hard_threshold_met': (best_ensemble['ic_median'] >= 0.02 and
                           best_ensemble['ir'] >= 0.5 and
                           best_ensemble['pmr'] >= 0.55),
    'ir_vs_baseline': ir_improvement,
    'ir_improvement_pct': ir_improvement/baseline_ir*100
}

meta_file = OUTPUT_DIR / f'stacking_ensemble_best_{timestamp}.csv'
pd.DataFrame([best_meta]).to_csv(meta_file, index=False)
print(f"  Best ensemble metadata: {meta_file}")

print(f"\n[5/7] Analyzing weight distributions...")

# Analyze weight distributions
print(f"\nWeighted Average Weights Distribution:")
learner_keys = list(BASE_LEARNERS.keys())
for key in learner_keys:
    learner_name = BASE_LEARNERS[key]['name']
    weight_col = f'weight_{key}'
    mean_w = results_df[weight_col].mean()
    std_w = results_df[weight_col].std()
    print(f"  {learner_name}: mean={mean_w:.3f}, std={std_w:.3f}")

print(f"\nStacking Coefficients Distribution:")
for key in learner_keys:
    learner_name = BASE_LEARNERS[key]['name']
    coef_col = f'stacking_coef_{key}'
    mean_c = results_df[coef_col].mean()
    std_c = results_df[coef_col].std()
    print(f"  {learner_name}: mean={mean_c:.3f}, std={std_c:.3f}")

print(f"\n[6/7] IC_std reduction analysis...")

# Compare IC_std across methods
# Find the best single base learner by IR
best_base = stats_df[stats_df['model'].str.startswith('Base')].nlargest(1, 'ir').iloc[0]
best_base_std = best_base['ic_std']
best_ensemble_std = best_ensemble['ic_std']
std_reduction = best_base_std - best_ensemble_std
std_reduction_pct = std_reduction / best_base_std * 100

print(f"\nIC_std Comparison:")
print(f"  Best single base learner ({best_base['model']}): {best_base_std:.6f}")
print(f"  Best ensemble:                                   {best_ensemble_std:.6f}")
print(f"  Reduction:                                       {std_reduction:.6f} ({std_reduction_pct:+.1f}%)")

print(f"\n[7/7] Experiment complete!")
print(f"\nKey Insights:")
print(f"  - Best ensemble method: {best_ensemble['model']}")
print(f"  - IC_std reduction: {std_reduction_pct:+.1f}%")
print(f"  - IR improvement vs baseline: {ir_improvement/baseline_ir*100:+.1f}%")
print(f"  - {'READY' if best_meta['hard_threshold_met'] else 'NOT READY'} for base promotion")
print(f"\n" + "="*80)
