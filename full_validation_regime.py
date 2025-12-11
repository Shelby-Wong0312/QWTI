#!/usr/bin/env python3
"""
Full Validation for base_7_plus_regime Model

- Features: 10 (base_7 + vol_regime_high + ovx_high + momentum_24h)
- Train: 30d (720h)
- Test: 14d (336h)
- Period: 2024-10 to 2025-12
- Output: warehouse/ic/

Compares with original base_seed202_lean7_h1 (7 features, 60d train)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import lightgbm as lgb
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Paths
FEATURES_PATH = Path("features_hourly_with_regime.parquet")
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

# Feature sets
BASE_7_FEATURES = [
    "OIL_CORE_norm_art_cnt",
    "GEOPOL_norm_art_cnt",
    "USD_RATE_norm_art_cnt",
    "SUPPLY_CHAIN_norm_art_cnt",
    "MACRO_norm_art_cnt",
    "cl1_cl2",
    "ovx",
]

REGIME_FEATURES = [
    "vol_regime_high",
    "ovx_high",
    "momentum_24h",
]

BASE_7_PLUS_REGIME = BASE_7_FEATURES + REGIME_FEATURES

TARGET_COL = "wti_returns"

# Period
FULL_PERIOD = ("2024-10-01", "2025-12-01")

# Window settings
TRAIN_HOURS = 720   # 30 days
TEST_HOURS = 336    # 14 days


def load_data():
    """Load feature data"""
    print("[1/7] Loading features...")
    df = pd.read_parquet(FEATURES_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    # Filter to period
    start_dt = pd.Timestamp(FULL_PERIOD[0], tz="UTC")
    end_dt = pd.Timestamp(FULL_PERIOD[1], tz="UTC")
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

    print(f"   Rows: {len(df)}")
    print(f"   Range: {df.index.min()} to {df.index.max()}")

    return df


def winsorize(series, lower=0.01, upper=0.99):
    """Winsorize at percentiles"""
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)


def run_rolling_validation(df, features, config_name):
    """Run rolling window validation"""
    print(f"\n[2/7] Running validation: {config_name}...")
    print(f"   Features: {len(features)}")
    print(f"   Train: {TRAIN_HOURS}h, Test: {TEST_HOURS}h")

    df = df.copy()

    # Winsorize
    for col in features:
        if col in df.columns:
            df[col] = winsorize(df[col])
    df[TARGET_COL] = winsorize(df[TARGET_COL])

    # Drop NaN
    df = df.dropna(subset=features + [TARGET_COL])

    n_samples = len(df)
    print(f"   Samples after cleaning: {n_samples}")

    if n_samples < TRAIN_HOURS + TEST_HOURS:
        print(f"   ERROR: Not enough data")
        return None, None

    results = []
    predictions = []
    window_idx = 0
    start_idx = TRAIN_HOURS

    while start_idx + TEST_HOURS <= n_samples:
        train_df = df.iloc[start_idx - TRAIN_HOURS:start_idx]
        test_df = df.iloc[start_idx:start_idx + TEST_HOURS]

        X_train = train_df[features]
        y_train = train_df[TARGET_COL]
        X_test = test_df[features]
        y_test = test_df[TARGET_COL]

        # Train
        model = lgb.LGBMRegressor(**MODEL_CONFIG)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # IC for this window
        ic, _ = spearmanr(y_pred, y_test)

        test_month = test_df.index[len(test_df)//2].strftime("%Y-%m")

        results.append({
            'window': window_idx + 1,
            'test_month': test_month,
            'test_start': test_df.index.min(),
            'test_end': test_df.index.max(),
            'ic': ic,
            'n_test': len(test_df),
        })

        # Store predictions for PnL
        # Position formula from Readme/Dashboard:
        # position = base_weight × sign(pred) × min(1, |pred| / 0.005)
        base_weight = 0.15
        threshold = 0.005
        for ts, pred, actual in zip(test_df.index, y_pred, y_test):
            position = base_weight * np.sign(pred) * min(1.0, abs(pred) / threshold)
            pnl = position * actual
            predictions.append({
                'timestamp': ts,
                'prediction': pred,
                'actual': actual,
                'position': position,
                'pnl': pnl,
            })

        window_idx += 1
        start_idx += TEST_HOURS

    print(f"   Windows completed: {window_idx}")

    # Get feature importance from last model
    importance = dict(zip(features, model.feature_importances_))

    return pd.DataFrame(results), pd.DataFrame(predictions), importance


def calculate_metrics(windows_df):
    """Calculate IC/IR/PMR metrics"""
    if windows_df is None or len(windows_df) == 0:
        return None

    ic_values = windows_df['ic'].dropna()

    metrics = {
        'n_windows': len(windows_df),
        'ic_mean': ic_values.mean(),
        'ic_median': ic_values.median(),
        'ic_std': ic_values.std(),
        'ir': ic_values.mean() / ic_values.std() if ic_values.std() > 0 else np.nan,
        'pmr': (ic_values > 0).mean(),
        'ic_min': ic_values.min(),
        'ic_max': ic_values.max(),
    }

    # Hard Gate check
    metrics['ic_pass'] = metrics['ic_median'] >= 0.02
    metrics['ir_pass'] = metrics['ir'] >= 0.5
    metrics['pmr_pass'] = metrics['pmr'] >= 0.55
    metrics['all_pass'] = metrics['ic_pass'] and metrics['ir_pass'] and metrics['pmr_pass']

    return metrics


def calculate_pnl_metrics(predictions_df):
    """Calculate PnL metrics"""
    if predictions_df is None or len(predictions_df) == 0:
        return None

    cumulative_pnl = predictions_df['pnl'].cumsum()
    total_pnl = cumulative_pnl.iloc[-1]

    # Sharpe
    hourly_pnl = predictions_df['pnl']
    sharpe = (hourly_pnl.mean() / hourly_pnl.std()) * np.sqrt(24 * 252) if hourly_pnl.std() > 0 else np.nan

    # Max drawdown
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_dd = drawdown.min()

    # Hit rate
    hit_rate = ((predictions_df['prediction'] * predictions_df['actual']) > 0).mean()

    return {
        'total_pnl': total_pnl,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'hit_rate': hit_rate,
    }


def generate_plots(predictions_df, windows_df, metrics, config_name, output_dir):
    """Generate diagnostic plots"""
    print(f"\n[4/7] Generating plots for {config_name}...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cumulative PnL
    ax1 = axes[0, 0]
    cumulative_pnl = predictions_df['pnl'].cumsum()
    ax1.plot(predictions_df['timestamp'], cumulative_pnl, 'b-', linewidth=1)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title(f'Cumulative PnL - {config_name}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative PnL')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Window IC
    ax2 = axes[0, 1]
    colors = ['green' if ic > 0.02 else 'orange' if ic > 0 else 'red' for ic in windows_df['ic']]
    ax2.bar(range(len(windows_df)), windows_df['ic'], color=colors)
    ax2.axhline(y=0.02, color='green', linestyle='--', label='Hard Gate (0.02)', alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title(f'Window IC - {config_name}')
    ax2.set_xlabel('Window')
    ax2.set_ylabel('IC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Monthly IC
    ax3 = axes[1, 0]
    monthly = windows_df.groupby('test_month')['ic'].mean()
    colors = ['green' if ic > 0 else 'red' for ic in monthly]
    ax3.bar(range(len(monthly)), monthly.values, color=colors)
    ax3.set_xticks(range(len(monthly)))
    ax3.set_xticklabels(monthly.index, rotation=45)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title(f'Monthly IC - {config_name}')
    ax3.set_ylabel('IC')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Position distribution
    ax4 = axes[1, 1]
    ax4.hist(predictions_df['position'], bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax4.set_title(f'Position Distribution - {config_name}')
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"regime_validation_{config_name}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Plot saved: {plot_path}")
    return plot_path


def main():
    print("=" * 70)
    print("FULL VALIDATION: base_7_plus_regime")
    print("=" * 70)
    print(f"Period: {FULL_PERIOD[0]} to {FULL_PERIOD[1]}")
    print(f"Train: {TRAIN_HOURS}h ({TRAIN_HOURS//24}d)")
    print(f"Test: {TEST_HOURS}h ({TEST_HOURS//24}d)")
    print("=" * 70)

    # Load data
    df = load_data()

    # Run validation
    windows_df, predictions_df, importance = run_rolling_validation(
        df, BASE_7_PLUS_REGIME, "base_7_plus_regime"
    )

    # Calculate metrics
    print("\n[3/7] Calculating metrics...")
    metrics = calculate_metrics(windows_df)
    pnl_metrics = calculate_pnl_metrics(predictions_df)

    # Generate plots
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = generate_plots(predictions_df, windows_df, metrics, "base_7_plus_regime", OUTPUT_DIR)

    # Print results
    print("\n[5/7] Results Summary")
    print("=" * 70)

    print("\n### Window-by-Window IC ###")
    print(f"{'Window':<8} {'Month':<10} {'IC':<12} {'Status':<10}")
    print("-" * 40)
    for _, row in windows_df.iterrows():
        status = "HARD" if row['ic'] >= 0.02 else ("OK" if row['ic'] > 0 else "X")
        print(f"{row['window']:<8} {row['test_month']:<10} {row['ic']:<12.6f} {status}")

    print("\n### Monthly IC Summary ###")
    monthly = windows_df.groupby('test_month')['ic'].agg(['mean', 'count'])
    for month, row in monthly.iterrows():
        status = "OK" if row['mean'] > 0 else "X"
        print(f"   {month}: IC={row['mean']:.4f} (n={int(row['count'])}) {status}")

    print("\n### Hard Gate Check ###")
    print(f"   IC median >= 0.02: {metrics['ic_median']:.4f} {'PASS' if metrics['ic_pass'] else 'FAIL'}")
    print(f"   IR >= 0.5:         {metrics['ir']:.4f} {'PASS' if metrics['ir_pass'] else 'FAIL'}")
    print(f"   PMR >= 0.55:       {metrics['pmr']:.2%} {'PASS' if metrics['pmr_pass'] else 'FAIL'}")
    print(f"\n   Overall: {'ALL PASS - APPROVED' if metrics['all_pass'] else 'FAILED'}")

    print("\n### PnL Metrics ###")
    print(f"   Total PnL:    {pnl_metrics['total_pnl']:.6f}")
    print(f"   Sharpe:       {pnl_metrics['sharpe']:.4f}")
    print(f"   Max Drawdown: {pnl_metrics['max_drawdown']:.6f}")
    print(f"   Hit Rate:     {pnl_metrics['hit_rate']:.2%}")

    print("\n### Feature Importance ###")
    for feat, val in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        print(f"   {feat:<35} {val:.1f}")

    # Compare with original base
    print("\n[6/7] Comparison with Original base_seed202_lean7_h1")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'base_7_plus_regime':<20} {'Original (approx)':<20} {'Better?':<10}")
    print("-" * 70)

    # Original metrics from earlier validation
    original = {
        'ic_median': 0.050,  # from 2024-10 to 2025-06
        'ir': 1.07,
        'pmr': 0.86,
        'sharpe': -0.29,  # full period was negative
    }

    print(f"{'IC median':<20} {metrics['ic_median']:<20.4f} {original['ic_median']:<20.4f} {'YES' if metrics['ic_median'] > original['ic_median'] else 'NO'}")
    print(f"{'IR':<20} {metrics['ir']:<20.4f} {original['ir']:<20.4f} {'YES' if metrics['ir'] > original['ir'] else 'NO'}")
    print(f"{'PMR':<20} {metrics['pmr']:<20.2%} {original['pmr']:<20.2%} {'YES' if metrics['pmr'] > original['pmr'] else 'NO'}")
    print(f"{'Sharpe':<20} {pnl_metrics['sharpe']:<20.4f} {original['sharpe']:<20.4f} {'YES' if pnl_metrics['sharpe'] > original['sharpe'] else 'NO'}")

    # Save results
    print("\n[7/7] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Windows
    windows_path = OUTPUT_DIR / f"regime_full_windows_{timestamp}.csv"
    windows_df.to_csv(windows_path, index=False)
    print(f"   Windows: {windows_path}")

    # Predictions
    pred_path = OUTPUT_DIR / f"regime_full_predictions_{timestamp}.csv"
    predictions_df.to_csv(pred_path, index=False)
    print(f"   Predictions: {pred_path}")

    # Summary
    summary = {
        'config': 'base_7_plus_regime',
        'n_features': len(BASE_7_PLUS_REGIME),
        'train_hours': TRAIN_HOURS,
        'test_hours': TEST_HOURS,
        'period_start': FULL_PERIOD[0],
        'period_end': FULL_PERIOD[1],
    }
    summary.update(metrics)
    summary.update(pnl_metrics)

    summary_df = pd.DataFrame([summary])
    summary_path = OUTPUT_DIR / f"regime_full_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"   Summary: {summary_path}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if metrics['all_pass']:
        print("\n[OK] base_7_plus_regime PASSES ALL HARD GATES")
        print("\n   RECOMMENDATION: REPLACE base_seed202_lean7_h1 with base_7_plus_regime")
        print("   - Use 10 features (7 base + 3 regime)")
        print("   - Use 30d training window (instead of 60d)")
        print("   - Update hourly_monitor.py configuration")
    else:
        print("\n[X] base_7_plus_regime DOES NOT PASS ALL HARD GATES")
        if metrics['ic_median'] > 0:
            print("   - Shows positive IC, but below thresholds")
            print("   - Consider as experimental, not production-ready")
        else:
            print("   - Keep original base_seed202_lean7_h1")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    return {
        'windows': windows_df,
        'predictions': predictions_df,
        'metrics': metrics,
        'pnl_metrics': pnl_metrics,
        'importance': importance,
    }


if __name__ == "__main__":
    results = main()
