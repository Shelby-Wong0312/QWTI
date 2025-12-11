#!/usr/bin/env python3
"""
Base Weight Scan for base_seed202_regime_h1

Scans different base_weight values (10%, 12.5%, 15%) with 5 bps transaction cost.
Evaluates net Sharpe and MaxDD to find optimal weight.

Position formula: position = base_weight * sign(pred) * min(1, |pred| / 0.005)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
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

# Feature columns
FEATURE_COLS = [
    "OIL_CORE_norm_art_cnt",
    "GEOPOL_norm_art_cnt",
    "USD_RATE_norm_art_cnt",
    "SUPPLY_CHAIN_norm_art_cnt",
    "MACRO_norm_art_cnt",
    "cl1_cl2",
    "ovx",
    "vol_regime_high",
    "ovx_high",
    "momentum_24h",
]

TARGET_COL = "wti_returns"

# Period and windows
FULL_PERIOD = ("2024-10-01", "2025-12-01")
TRAIN_HOURS = 720   # 30 days
TEST_HOURS = 336    # 14 days

# Cost assumption
COST_BPS = 5
THRESHOLD = 0.005

# Base weights to scan
BASE_WEIGHTS = [0.10, 0.125, 0.15]


def winsorize(series, lower=0.01, upper=0.99):
    """Winsorize at percentiles"""
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)


def calculate_position(pred, base_weight):
    """Calculate position using production formula"""
    return base_weight * np.sign(pred) * min(1.0, abs(pred) / THRESHOLD)


def run_validation_with_weight(df, base_weight, cost_bps=5):
    """Run full validation with specific base_weight and transaction cost"""

    df = df.copy()

    # Winsorize
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = winsorize(df[col])
    df[TARGET_COL] = winsorize(df[TARGET_COL])

    # Drop NaN
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    n_samples = len(df)
    if n_samples < TRAIN_HOURS + TEST_HOURS:
        return None

    predictions = []
    window_idx = 0
    start_idx = TRAIN_HOURS

    while start_idx + TEST_HOURS <= n_samples:
        train_df = df.iloc[start_idx - TRAIN_HOURS:start_idx]
        test_df = df.iloc[start_idx:start_idx + TEST_HOURS]

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]
        X_test = test_df[FEATURE_COLS]
        y_test = test_df[TARGET_COL]

        # Train
        model = lgb.LGBMRegressor(**MODEL_CONFIG)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Store predictions
        for ts, pred, actual in zip(test_df.index, y_pred, y_test):
            position = calculate_position(pred, base_weight)
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

    pred_df = pd.DataFrame(predictions)

    # Calculate turnover and costs
    pred_df['pos_change'] = pred_df['position'].diff().abs()
    pred_df.loc[pred_df.index[0], 'pos_change'] = abs(pred_df['position'].iloc[0])

    cost_per_trade = cost_bps / 10000
    pred_df['cost'] = pred_df['pos_change'] * cost_per_trade
    pred_df['net_pnl'] = pred_df['pnl'] - pred_df['cost']

    return pred_df


def calculate_metrics(pred_df):
    """Calculate comprehensive metrics"""

    # Gross metrics
    gross_pnl = pred_df['pnl'].sum()

    # Net metrics
    net_pnl = pred_df['net_pnl'].sum()
    total_cost = pred_df['cost'].sum()

    # Turnover
    total_turnover = pred_df['pos_change'].sum()
    n_hours = len(pred_df)
    years = n_hours / (252 * 24)
    annual_turnover = total_turnover / years

    # Sharpe (daily)
    pred_df['date'] = pd.to_datetime(pred_df['timestamp']).dt.date

    daily_gross = pred_df.groupby('date')['pnl'].sum()
    daily_net = pred_df.groupby('date')['net_pnl'].sum()

    gross_sharpe = (daily_gross.mean() / daily_gross.std()) * np.sqrt(252) if daily_gross.std() > 0 else 0
    net_sharpe = (daily_net.mean() / daily_net.std()) * np.sqrt(252) if daily_net.std() > 0 else 0

    # Max Drawdown
    cumulative_gross = pred_df['pnl'].cumsum()
    cumulative_net = pred_df['net_pnl'].cumsum()

    gross_max_dd = (cumulative_gross - cumulative_gross.cummax()).min()
    net_max_dd = (cumulative_net - cumulative_net.cummax()).min()

    # Win rate
    positive_days = (daily_net > 0).sum()
    total_days = len(daily_net)
    win_rate = positive_days / total_days

    # Position stats
    mean_pos = pred_df['position'].abs().mean()
    max_pos = pred_df['position'].abs().max()

    return {
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'total_cost': total_cost,
        'annual_turnover': annual_turnover,
        'gross_sharpe': gross_sharpe,
        'net_sharpe': net_sharpe,
        'gross_max_dd': gross_max_dd,
        'net_max_dd': net_max_dd,
        'win_rate': win_rate,
        'positive_days': positive_days,
        'total_days': total_days,
        'mean_position': mean_pos,
        'max_position': max_pos,
    }


def main():
    print("=" * 70)
    print("BASE WEIGHT SCAN: base_seed202_regime_h1")
    print("=" * 70)
    print(f"Base weights to scan: {[f'{w*100:.1f}%' for w in BASE_WEIGHTS]}")
    print(f"Transaction cost: {COST_BPS} bps")
    print(f"Position formula: base_weight * sign(pred) * min(1, |pred|/{THRESHOLD})")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading features...")
    df = pd.read_parquet(FEATURES_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    # Filter to period
    start_dt = pd.Timestamp(FULL_PERIOD[0], tz="UTC")
    end_dt = pd.Timestamp(FULL_PERIOD[1], tz="UTC")
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

    print(f"   Rows: {len(df)}")
    print(f"   Range: {df.index.min()} to {df.index.max()}")

    # Run validation for each weight
    print("\n[2/3] Running validation for each base_weight...")
    results = []

    for base_weight in BASE_WEIGHTS:
        print(f"\n   Testing base_weight = {base_weight*100:.1f}%...")

        pred_df = run_validation_with_weight(df.copy(), base_weight, COST_BPS)

        if pred_df is None:
            print(f"   ERROR: Not enough data")
            continue

        metrics = calculate_metrics(pred_df)
        metrics['base_weight'] = base_weight
        results.append(metrics)

        print(f"      Net Sharpe: {metrics['net_sharpe']:.2f}")
        print(f"      Net MaxDD:  {metrics['net_max_dd']:.6f}")
        print(f"      Net PnL:    {metrics['net_pnl']:.6f}")

    # Results summary
    print("\n" + "=" * 70)
    print("[3/3] RESULTS SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)

    print("\n### Base Weight Comparison (5 bps cost) ###")
    print(f"\n{'Weight':<10} {'Net Sharpe':<12} {'Net MaxDD':<12} {'Net PnL':<12} {'Win Rate':<10} {'Turnover/yr':<12}")
    print("-" * 70)

    for _, row in results_df.iterrows():
        print(f"{row['base_weight']*100:.1f}%      "
              f"{row['net_sharpe']:<12.2f} "
              f"{row['net_max_dd']:<12.6f} "
              f"{row['net_pnl']:<12.6f} "
              f"{row['win_rate']*100:<10.1f}% "
              f"{row['annual_turnover']:<12.1f}x")

    # Find optimal weight
    print("\n### Optimal Weight Selection ###")

    # Criteria: maximize net Sharpe while keeping MaxDD reasonable
    best_sharpe_idx = results_df['net_sharpe'].idxmax()
    best_sharpe_row = results_df.loc[best_sharpe_idx]

    # Also check MaxDD-adjusted (Sharpe / |MaxDD|)
    results_df['sharpe_dd_ratio'] = results_df['net_sharpe'] / results_df['net_max_dd'].abs()
    best_ratio_idx = results_df['sharpe_dd_ratio'].idxmax()
    best_ratio_row = results_df.loc[best_ratio_idx]

    print(f"\n   Best Net Sharpe:      {best_sharpe_row['base_weight']*100:.1f}% (Sharpe={best_sharpe_row['net_sharpe']:.2f})")
    print(f"   Best Sharpe/DD ratio: {best_ratio_row['base_weight']*100:.1f}% (ratio={best_ratio_row['sharpe_dd_ratio']:.1f})")

    # Select optimal (prefer Sharpe/DD ratio for risk-adjusted)
    optimal = best_ratio_row

    print(f"\n   SELECTED: base_weight = {optimal['base_weight']*100:.1f}%")
    print(f"      Net Sharpe: {optimal['net_sharpe']:.2f}")
    print(f"      Net MaxDD:  {optimal['net_max_dd']:.6f}")
    print(f"      Net PnL:    {optimal['net_pnl']:.6f}")
    print(f"      Win Rate:   {optimal['win_rate']*100:.1f}%")

    # Save results
    print("\n### Saving Results ###")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = OUTPUT_DIR / f"base_weight_scan_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"   Results saved: {results_path}")

    print("\n" + "=" * 70)
    print("BASE WEIGHT SCAN COMPLETE")
    print("=" * 70)
    print(f"\nRECOMMENDATION: Set base_weight = {optimal['base_weight']*100:.1f}%")

    return results_df, optimal


if __name__ == "__main__":
    results_df, optimal = main()
