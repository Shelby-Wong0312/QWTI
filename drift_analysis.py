#!/usr/bin/env python3
"""
Drift Analysis: Compare 2025/02-05 (good period) vs 2025/07-11 (bad period)

Outputs:
- Period-wise IC/IR/PMR comparison
- Feature importance changes
- Feature distribution drift (KS test)
- Diagnosis report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import lightgbm as lgb
from scipy.stats import spearmanr, ks_2samp
import warnings
warnings.filterwarnings('ignore')

# Paths
FEATURES_PATH = Path("features_hourly_with_term.parquet")
OUTPUT_DIR = Path("warehouse/ic")

# Model config
MODEL_CONFIG = {
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "num_leaves": 31,
    "random_state": 202,
    "verbosity": -1,
    "n_jobs": -1,
}

FEATURE_COLS = [
    "OIL_CORE_norm_art_cnt",
    "GEOPOL_norm_art_cnt",
    "USD_RATE_norm_art_cnt",
    "SUPPLY_CHAIN_norm_art_cnt",
    "MACRO_norm_art_cnt",
    "cl1_cl2",
    "ovx",
]

TARGET_COL = "wti_returns"

# Period definitions
GOOD_PERIOD = ("2025-02-01", "2025-05-31")  # Good performance
BAD_PERIOD = ("2025-07-01", "2025-11-30")   # Poor performance
TRAIN_HOURS = 1440  # 60 days


def load_data():
    """Load feature data"""
    print("[1/6] Loading data...")
    df = pd.read_parquet(FEATURES_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    print(f"   Total rows: {len(df)}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    return df


def get_period_data(df, start, end):
    """Extract data for a specific period"""
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC")
    return df[(df.index >= start_dt) & (df.index <= end_dt)]


def train_and_evaluate(df, period_name):
    """Train model and get metrics + feature importance"""
    print(f"\n   Training on {period_name}...")

    # Use first 60 days as training, rest as test
    if len(df) < TRAIN_HOURS + 100:
        print(f"   WARNING: Not enough data ({len(df)} rows)")
        return None, None, None

    train_df = df.iloc[:TRAIN_HOURS]
    test_df = df.iloc[TRAIN_HOURS:]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    # Train
    model = lgb.LGBMRegressor(**MODEL_CONFIG)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    ic, _ = spearmanr(y_pred, y_test)

    # Daily IC for IR/PMR
    test_df = test_df.copy()
    test_df['pred'] = y_pred
    test_df['date'] = test_df.index.date

    daily_ic = test_df.groupby('date').apply(
        lambda x: spearmanr(x['pred'], x[TARGET_COL])[0] if len(x) > 1 else np.nan
    ).dropna()

    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    ir = ic_mean / ic_std if ic_std > 0 else np.nan
    pmr = (daily_ic > 0).mean()

    metrics = {
        'period': period_name,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'ic_overall': ic,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ir': ir,
        'pmr': pmr,
    }

    # Feature importance
    importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    return metrics, importance, model


def analyze_distribution_drift(df_good, df_bad):
    """Analyze feature distribution drift using KS test"""
    print("\n[3/6] Analyzing distribution drift...")

    drift_results = []
    for col in FEATURE_COLS + [TARGET_COL]:
        good_vals = df_good[col].dropna()
        bad_vals = df_bad[col].dropna()

        # KS test
        ks_stat, ks_pval = ks_2samp(good_vals, bad_vals)

        # Basic stats
        drift_results.append({
            'feature': col,
            'good_mean': good_vals.mean(),
            'bad_mean': bad_vals.mean(),
            'mean_change': bad_vals.mean() - good_vals.mean(),
            'mean_change_pct': (bad_vals.mean() - good_vals.mean()) / (abs(good_vals.mean()) + 1e-10) * 100,
            'good_std': good_vals.std(),
            'bad_std': bad_vals.std(),
            'std_change_pct': (bad_vals.std() - good_vals.std()) / (good_vals.std() + 1e-10) * 100,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'significant_drift': ks_pval < 0.05,
        })

    return pd.DataFrame(drift_results)


def analyze_target_predictability(df_good, df_bad):
    """Analyze if target behavior changed"""
    print("\n[4/6] Analyzing target predictability...")

    results = {}

    for name, df in [('good', df_good), ('bad', df_bad)]:
        target = df[TARGET_COL]

        # Autocorrelation
        ac1 = target.autocorr(lag=1)
        ac24 = target.autocorr(lag=24)

        # Volatility
        vol = target.std()

        # Feature correlations with target
        corrs = {}
        for col in FEATURE_COLS:
            corr, _ = spearmanr(df[col], target)
            corrs[col] = corr

        results[name] = {
            'autocorr_1h': ac1,
            'autocorr_24h': ac24,
            'volatility': vol,
            'feature_correlations': corrs,
        }

    return results


def generate_plots(df_good, df_bad, importance_good, importance_bad, drift_df, output_dir):
    """Generate diagnostic plots"""
    print("\n[5/6] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Feature Importance Comparison
    ax1 = axes[0, 0]
    x = np.arange(len(FEATURE_COLS))
    width = 0.35

    imp_good = importance_good.set_index('feature').loc[FEATURE_COLS, 'importance'].values
    imp_bad = importance_bad.set_index('feature').loc[FEATURE_COLS, 'importance'].values

    ax1.bar(x - width/2, imp_good, width, label='Good (Feb-May)', color='green', alpha=0.7)
    ax1.bar(x + width/2, imp_bad, width, label='Bad (Jul-Nov)', color='red', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.replace('_norm_art_cnt', '').replace('_', '\n') for f in FEATURE_COLS], rotation=0, fontsize=8)
    ax1.set_ylabel('Importance')
    ax1.set_title('Feature Importance: Good vs Bad Period')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distribution Drift (KS statistic)
    ax2 = axes[0, 1]
    drift_sorted = drift_df.sort_values('ks_statistic', ascending=True)
    colors = ['red' if sig else 'gray' for sig in drift_sorted['significant_drift']]
    ax2.barh(drift_sorted['feature'].str.replace('_norm_art_cnt', ''), drift_sorted['ks_statistic'], color=colors)
    ax2.axvline(x=0.1, color='orange', linestyle='--', label='Moderate drift')
    ax2.set_xlabel('KS Statistic')
    ax2.set_title('Distribution Drift (KS Test)\nRed = Significant (p<0.05)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Target Distribution
    ax3 = axes[1, 0]
    ax3.hist(df_good[TARGET_COL], bins=50, alpha=0.5, label='Good (Feb-May)', color='green', density=True)
    ax3.hist(df_bad[TARGET_COL], bins=50, alpha=0.5, label='Bad (Jul-Nov)', color='red', density=True)
    ax3.set_xlabel('WTI Returns')
    ax3.set_ylabel('Density')
    ax3.set_title('Target Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Feature Mean Changes
    ax4 = axes[1, 1]
    feat_drift = drift_df[drift_df['feature'] != TARGET_COL].copy()
    colors = ['red' if abs(v) > 20 else 'orange' if abs(v) > 10 else 'green' for v in feat_drift['mean_change_pct']]
    ax4.barh(feat_drift['feature'].str.replace('_norm_art_cnt', ''), feat_drift['mean_change_pct'], color=colors)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Mean Change (%)')
    ax4.set_title('Feature Mean Drift (%)\nRed=|>20%|, Orange=|>10%|')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"drift_analysis_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Plot saved: {plot_path}")
    return plot_path


def main():
    print("=" * 70)
    print("DRIFT ANALYSIS: 2025/02-05 (Good) vs 2025/07-11 (Bad)")
    print("=" * 70)

    # Load data
    df = load_data()

    # Extract periods
    df_good = get_period_data(df, *GOOD_PERIOD)
    df_bad = get_period_data(df, *BAD_PERIOD)

    print(f"\n   Good period ({GOOD_PERIOD[0]} to {GOOD_PERIOD[1]}): {len(df_good)} rows")
    print(f"   Bad period ({BAD_PERIOD[0]} to {BAD_PERIOD[1]}): {len(df_bad)} rows")

    # Train and evaluate on each period
    print("\n[2/6] Training period-specific models...")
    metrics_good, importance_good, model_good = train_and_evaluate(df_good, "Good (Feb-May)")
    metrics_bad, importance_bad, model_bad = train_and_evaluate(df_bad, "Bad (Jul-Nov)")

    # Distribution drift analysis
    drift_df = analyze_distribution_drift(df_good, df_bad)

    # Target predictability analysis
    pred_analysis = analyze_target_predictability(df_good, df_bad)

    # Generate plots
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = generate_plots(df_good, df_bad, importance_good, importance_bad, drift_df, OUTPUT_DIR)

    # Print results
    print("\n[6/6] Results Summary")
    print("=" * 70)

    print("\n### Period Performance Comparison ###")
    print(f"\n{'Metric':<20} {'Good (Feb-May)':<20} {'Bad (Jul-Nov)':<20} {'Change':<15}")
    print("-" * 75)

    if metrics_good and metrics_bad:
        for key in ['ic_overall', 'ic_mean', 'ir', 'pmr']:
            g = metrics_good[key]
            b = metrics_bad[key]
            change = b - g
            pct = f"({change/abs(g)*100:+.1f}%)" if g != 0 else ""
            print(f"{key:<20} {g:<20.4f} {b:<20.4f} {change:+.4f} {pct}")

    print("\n### Feature Importance Changes ###")
    print(f"\n{'Feature':<30} {'Good':<12} {'Bad':<12} {'Change':<12} {'Change %':<12}")
    print("-" * 78)

    for feat in FEATURE_COLS:
        g = importance_good[importance_good['feature'] == feat]['importance'].values[0]
        b = importance_bad[importance_bad['feature'] == feat]['importance'].values[0]
        change = b - g
        pct = (change / g * 100) if g > 0 else 0
        print(f"{feat:<30} {g:<12.1f} {b:<12.1f} {change:+12.1f} {pct:+12.1f}%")

    print("\n### Distribution Drift (KS Test) ###")
    print(f"\n{'Feature':<30} {'KS Stat':<12} {'P-value':<12} {'Significant':<12} {'Mean Chg%':<12}")
    print("-" * 78)

    for _, row in drift_df.iterrows():
        sig = "YES" if row['significant_drift'] else "no"
        print(f"{row['feature']:<30} {row['ks_statistic']:<12.4f} {row['ks_pvalue']:<12.4f} {sig:<12} {row['mean_change_pct']:+12.1f}%")

    print("\n### Target Predictability Analysis ###")
    print(f"\n{'Metric':<30} {'Good':<15} {'Bad':<15} {'Change':<15}")
    print("-" * 75)

    for metric in ['autocorr_1h', 'autocorr_24h', 'volatility']:
        g = pred_analysis['good'][metric]
        b = pred_analysis['bad'][metric]
        change = b - g
        print(f"{metric:<30} {g:<15.4f} {b:<15.4f} {change:+15.4f}")

    print("\n### Feature-Target Correlations ###")
    print(f"\n{'Feature':<30} {'Good':<12} {'Bad':<12} {'Change':<12}")
    print("-" * 66)

    for feat in FEATURE_COLS:
        g = pred_analysis['good']['feature_correlations'][feat]
        b = pred_analysis['bad']['feature_correlations'][feat]
        change = b - g
        print(f"{feat:<30} {g:+12.4f} {b:+12.4f} {change:+12.4f}")

    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    # Check for significant drift
    n_sig_drift = drift_df['significant_drift'].sum()
    max_drift_feat = drift_df.loc[drift_df['ks_statistic'].idxmax(), 'feature']
    max_drift_val = drift_df['ks_statistic'].max()

    # Check importance changes
    imp_changes = []
    for feat in FEATURE_COLS:
        g = importance_good[importance_good['feature'] == feat]['importance'].values[0]
        b = importance_bad[importance_bad['feature'] == feat]['importance'].values[0]
        if g > 0:
            imp_changes.append((feat, (b - g) / g * 100))

    max_imp_change = max(imp_changes, key=lambda x: abs(x[1]))

    # Check correlation changes
    corr_changes = []
    for feat in FEATURE_COLS:
        g = pred_analysis['good']['feature_correlations'][feat]
        b = pred_analysis['bad']['feature_correlations'][feat]
        corr_changes.append((feat, b - g))

    max_corr_change = max(corr_changes, key=lambda x: abs(x[1]))

    print(f"\n1. Distribution Drift:")
    print(f"   - {n_sig_drift}/{len(FEATURE_COLS)+1} features show significant drift (p<0.05)")
    print(f"   - Largest drift: {max_drift_feat} (KS={max_drift_val:.4f})")

    print(f"\n2. Feature Importance Shift:")
    print(f"   - Largest change: {max_imp_change[0]} ({max_imp_change[1]:+.1f}%)")

    print(f"\n3. Feature-Target Correlation Changes:")
    print(f"   - Largest change: {max_corr_change[0]} ({max_corr_change[1]:+.4f})")

    print(f"\n4. Target Behavior:")
    ac_change = pred_analysis['bad']['autocorr_1h'] - pred_analysis['good']['autocorr_1h']
    vol_change = pred_analysis['bad']['volatility'] - pred_analysis['good']['volatility']
    print(f"   - Autocorr(1h) change: {ac_change:+.4f}")
    print(f"   - Volatility change: {vol_change:+.6f}")

    # Final diagnosis
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if n_sig_drift > 3:
        print("\n[!] SIGNIFICANT DATA DRIFT DETECTED")
        print("   Multiple features show distribution shifts.")
        print("   Recommendation: RETRAIN model with recent data or add drift-adaptive mechanism.")
    elif abs(max_corr_change[1]) > 0.05:
        print("\n[!] FEATURE-TARGET RELATIONSHIP CHANGED")
        print(f"   {max_corr_change[0]} correlation shifted by {max_corr_change[1]:+.4f}")
        print("   Recommendation: Investigate market regime change, consider adding new features.")
    elif abs(ac_change) > 0.1:
        print("\n[!] TARGET DYNAMICS CHANGED")
        print("   Return autocorrelation structure shifted significantly.")
        print("   Recommendation: Market has become less predictable; consider reducing position sizing.")
    else:
        print("\n[OK] No single dominant drift factor identified.")
        print("   Model may be overfitting to early data patterns.")
        print("   Recommendation: Increase regularization or use shorter training window.")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save metrics comparison
    if metrics_good and metrics_bad:
        metrics_df = pd.DataFrame([metrics_good, metrics_bad])
        metrics_path = OUTPUT_DIR / f"drift_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n   Metrics saved: {metrics_path}")

    # Save drift analysis
    drift_path = OUTPUT_DIR / f"drift_distribution_{timestamp}.csv"
    drift_df.to_csv(drift_path, index=False)
    print(f"   Drift analysis saved: {drift_path}")

    # Save importance comparison
    imp_compare = importance_good.merge(importance_bad, on='feature', suffixes=('_good', '_bad'))
    imp_compare['change_pct'] = (imp_compare['importance_bad'] - imp_compare['importance_good']) / imp_compare['importance_good'] * 100
    imp_path = OUTPUT_DIR / f"drift_importance_{timestamp}.csv"
    imp_compare.to_csv(imp_path, index=False)
    print(f"   Importance comparison saved: {imp_path}")

    print("\n" + "=" * 70)
    print("DRIFT ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        'metrics_good': metrics_good,
        'metrics_bad': metrics_bad,
        'drift_df': drift_df,
        'importance_good': importance_good,
        'importance_bad': importance_bad,
    }


if __name__ == "__main__":
    results = main()
