#!/usr/bin/env python3
"""
Full Period Backtest for base_seed202_lean7_h1

Uses features_hourly_with_term.parquet (2024-10 to 2025-12-01)
Outputs IC/IR/PMR metrics and cumulative PnL curve to warehouse/ic/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import lightgbm as lgb
from scipy.stats import spearmanr

# Paths
FEATURES_PATH = Path("features_hourly_with_term.parquet")
OUTPUT_DIR = Path("warehouse/ic")

# Model config (base_seed202_lean7_h1)
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
]

TARGET_COL = "wti_returns"

# Rolling window settings
TRAIN_HOURS = 1440  # 60 days
TEST_HOURS = 24     # 1 day (for daily rebalancing backtest)


def load_data():
    """Load and prepare data"""
    print("[1/6] Loading data...")
    df = pd.read_parquet(FEATURES_PATH)

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    # Filter to 2024-10 onwards (where GDELT data exists)
    start_date = pd.Timestamp("2024-10-01", tz="UTC")
    df = df[df.index >= start_date]

    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")

    return df


def find_continuous_segments(df, max_gap_hours=80):
    """Find continuous data segments allowing for market closures"""
    df = df.sort_index()
    time_diff = df.index.to_series().diff().dt.total_seconds() / 3600
    gap_mask = time_diff > max_gap_hours
    segment_id = gap_mask.cumsum()

    segments = []
    for seg_id in segment_id.unique():
        seg_df = df[segment_id == seg_id]
        if len(seg_df) >= TRAIN_HOURS + TEST_HOURS:
            segments.append(seg_df)

    return segments


def winsorize(series, lower=0.01, upper=0.99):
    """Winsorize series at given percentiles"""
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)


def run_backtest(df):
    """Run rolling window backtest"""
    print("[2/6] Running rolling backtest...")

    # Winsorize features
    for col in FEATURE_COLS:
        df[col] = winsorize(df[col])
    df[TARGET_COL] = winsorize(df[TARGET_COL])

    results = []
    positions = []

    n_samples = len(df)
    start_idx = TRAIN_HOURS

    total_windows = (n_samples - TRAIN_HOURS) // TEST_HOURS
    print(f"   Total samples: {n_samples}")
    print(f"   Windows to evaluate: {total_windows}")

    window_count = 0
    while start_idx + TEST_HOURS <= n_samples:
        train_df = df.iloc[start_idx - TRAIN_HOURS:start_idx]
        test_df = df.iloc[start_idx:start_idx + TEST_HOURS]

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df[TARGET_COL].values
        X_test = test_df[FEATURE_COLS].values
        y_test = test_df[TARGET_COL].values

        # Train model
        model = lgb.LGBMRegressor(**MODEL_CONFIG)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate IC for this window
        if len(y_test) > 1:
            ic, _ = spearmanr(y_pred, y_test)
        else:
            ic = np.nan

        # Store results
        for i, (ts, pred, actual) in enumerate(zip(test_df.index, y_pred, y_test)):
            # Position sizing: tanh scaling with 15% max
            position = np.tanh(pred * 100) * 0.15
            pnl = position * actual

            results.append({
                "timestamp": ts,
                "prediction": pred,
                "actual": actual,
                "position": position,
                "pnl": pnl,
                "ic": ic,
            })

        window_count += 1
        if window_count % 50 == 0:
            print(f"   Window {window_count}/{total_windows} completed...")

        start_idx += TEST_HOURS

    return pd.DataFrame(results)


def calculate_metrics(results_df):
    """Calculate IC, IR, PMR and other metrics"""
    print("[3/6] Calculating metrics...")

    # Daily IC (aggregate by day)
    results_df["date"] = results_df["timestamp"].dt.date
    daily_ic = results_df.groupby("date").apply(
        lambda x: spearmanr(x["prediction"], x["actual"])[0] if len(x) > 1 else np.nan
    )

    # Overall metrics
    ic_mean = daily_ic.mean()
    ic_median = daily_ic.median()
    ic_std = daily_ic.std()
    ir = ic_mean / ic_std if ic_std > 0 else np.nan
    pmr = (daily_ic > 0).mean()

    # Monthly breakdown
    results_df["month"] = results_df["timestamp"].dt.to_period("M")
    monthly_ic = results_df.groupby("month").apply(
        lambda x: spearmanr(x["prediction"], x["actual"])[0] if len(x) > 1 else np.nan
    )

    # PnL metrics
    cumulative_pnl = results_df["pnl"].cumsum()
    total_pnl = cumulative_pnl.iloc[-1]

    # Sharpe (annualized, assuming hourly)
    hourly_pnl = results_df["pnl"]
    sharpe = (hourly_pnl.mean() / hourly_pnl.std()) * np.sqrt(24 * 252) if hourly_pnl.std() > 0 else np.nan

    # Max drawdown
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_drawdown = drawdown.min()

    # Hit rate
    hit_rate = ((results_df["prediction"] * results_df["actual"]) > 0).mean()

    metrics = {
        "ic_mean": ic_mean,
        "ic_median": ic_median,
        "ic_std": ic_std,
        "ir": ir,
        "pmr": pmr,
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "n_days": len(daily_ic),
        "n_hours": len(results_df),
    }

    return metrics, daily_ic, monthly_ic


def plot_results(results_df, metrics, daily_ic, output_dir):
    """Generate and save plots"""
    print("[4/6] Generating plots...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Cumulative PnL
    cumulative_pnl = results_df["pnl"].cumsum()
    results_df["cumulative_pnl"] = cumulative_pnl

    ax1 = axes[0]
    ax1.plot(results_df["timestamp"], cumulative_pnl, "b-", linewidth=1)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title(f"Cumulative PnL (Total: {metrics['total_pnl']:.4f}, Sharpe: {metrics['sharpe']:.2f})")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative PnL")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Rolling IC (15-day)
    ax2 = axes[1]
    results_df["date"] = results_df["timestamp"].dt.date
    daily_data = results_df.groupby("date").agg({
        "prediction": "mean",
        "actual": "mean",
    }).reset_index()

    # Calculate rolling IC
    rolling_ic = []
    for i in range(len(daily_data)):
        start = max(0, i - 14)
        window = daily_data.iloc[start:i+1]
        if len(window) >= 5:
            ic, _ = spearmanr(window["prediction"], window["actual"])
            rolling_ic.append(ic)
        else:
            rolling_ic.append(np.nan)

    daily_data["rolling_ic"] = rolling_ic
    ax2.plot(pd.to_datetime(daily_data["date"]), daily_data["rolling_ic"], "g-", linewidth=1)
    ax2.axhline(y=0.02, color="red", linestyle="--", label="Hard Gate (0.02)", alpha=0.7)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title(f"Rolling 15-Day IC (Median: {metrics['ic_median']:.4f}, IR: {metrics['ir']:.2f})")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("IC")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Position distribution
    ax3 = axes[2]
    ax3.hist(results_df["position"], bins=50, edgecolor="black", alpha=0.7)
    ax3.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    ax3.set_title(f"Position Distribution (Hit Rate: {metrics['hit_rate']:.2%})")
    ax3.set_xlabel("Position")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"backtest_pnl_curve_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"   Plot saved: {plot_path}")
    return plot_path


def save_results(results_df, metrics, daily_ic, monthly_ic, output_dir):
    """Save results to CSV files"""
    print("[5/6] Saving results...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save hourly results
    hourly_path = output_dir / f"backtest_hourly_{timestamp}.csv"
    results_df.to_csv(hourly_path, index=False)
    print(f"   Hourly results: {hourly_path}")

    # Save daily IC
    daily_ic_df = pd.DataFrame({
        "date": daily_ic.index,
        "ic": daily_ic.values
    })
    daily_path = output_dir / f"backtest_daily_ic_{timestamp}.csv"
    daily_ic_df.to_csv(daily_path, index=False)
    print(f"   Daily IC: {daily_path}")

    # Save monthly IC
    monthly_ic_df = pd.DataFrame({
        "month": monthly_ic.index.astype(str),
        "ic": monthly_ic.values
    })
    monthly_path = output_dir / f"backtest_monthly_ic_{timestamp}.csv"
    monthly_ic_df.to_csv(monthly_path, index=False)
    print(f"   Monthly IC: {monthly_path}")

    # Save summary
    summary_df = pd.DataFrame([metrics])
    summary_path = output_dir / f"backtest_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"   Summary: {summary_path}")

    return {
        "hourly": hourly_path,
        "daily": daily_path,
        "monthly": monthly_path,
        "summary": summary_path,
    }


def main():
    print("=" * 70)
    print("FULL PERIOD BACKTEST - base_seed202_lean7_h1")
    print("=" * 70)
    print(f"Model: LightGBM (depth=5, lr=0.1, n=100, leaves=31, seed=202)")
    print(f"Features: {len(FEATURE_COLS)} (5 GDELT + cl1_cl2 + ovx)")
    print(f"Train window: {TRAIN_HOURS}h ({TRAIN_HOURS//24}d)")
    print(f"Test step: {TEST_HOURS}h")
    print("=" * 70)

    # Load data
    df = load_data()

    # Find continuous segments
    segments = find_continuous_segments(df)
    print(f"\n   Found {len(segments)} continuous segment(s)")

    if not segments:
        print("ERROR: No valid continuous segments found!")
        return

    # Use longest segment
    longest_seg = max(segments, key=len)
    print(f"   Using longest segment: {len(longest_seg)} hours")
    print(f"   Range: {longest_seg.index.min()} to {longest_seg.index.max()}")

    # Run backtest
    results_df = run_backtest(longest_seg)

    # Calculate metrics
    metrics, daily_ic, monthly_ic = calculate_metrics(results_df)

    # Print results
    print("\n[6/6] Results Summary")
    print("=" * 70)
    print("\nPerformance Metrics:")
    print(f"   IC mean:     {metrics['ic_mean']:.6f}")
    print(f"   IC median:   {metrics['ic_median']:.6f}")
    print(f"   IC std:      {metrics['ic_std']:.6f}")
    print(f"   IR:          {metrics['ir']:.4f}")
    print(f"   PMR:         {metrics['pmr']:.2%}")

    print("\nPnL Metrics:")
    print(f"   Total PnL:   {metrics['total_pnl']:.6f}")
    print(f"   Sharpe:      {metrics['sharpe']:.4f}")
    print(f"   Max DD:      {metrics['max_drawdown']:.6f}")
    print(f"   Hit Rate:    {metrics['hit_rate']:.2%}")

    print("\nCoverage:")
    print(f"   Days:        {metrics['n_days']}")
    print(f"   Hours:       {metrics['n_hours']}")

    # Check Hard Gate
    print("\n" + "=" * 70)
    print("HARD GATE CHECK")
    print("=" * 70)
    ic_pass = metrics['ic_median'] >= 0.02
    ir_pass = metrics['ir'] >= 0.5
    pmr_pass = metrics['pmr'] >= 0.55

    print(f"   IC median >= 0.02: {metrics['ic_median']:.4f} {'OK' if ic_pass else 'FAIL'}")
    print(f"   IR >= 0.5:         {metrics['ir']:.4f} {'OK' if ir_pass else 'FAIL'}")
    print(f"   PMR >= 0.55:       {metrics['pmr']:.2%} {'OK' if pmr_pass else 'FAIL'}")

    all_pass = ic_pass and ir_pass and pmr_pass
    print(f"\n   Overall: {'PASSED' if all_pass else 'FAILED'}")

    # Monthly breakdown
    print("\nMonthly IC Breakdown:")
    for month, ic in monthly_ic.items():
        status = "OK" if ic > 0 else "X"
        print(f"   {month}: {ic:.6f} {status}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = save_results(results_df, metrics, daily_ic, monthly_ic, OUTPUT_DIR)

    # Generate plots
    plot_path = plot_results(results_df, metrics, daily_ic, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    for name, path in paths.items():
        print(f"   {name}: {path}")
    print(f"   plot: {plot_path}")

    return results_df, metrics


if __name__ == "__main__":
    results, metrics = main()
