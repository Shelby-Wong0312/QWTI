#!/usr/bin/env python3
"""
Drift Period Short Window Validation

- Train: 30d (720h)
- Test: 14d (336h)
- New features: EIA event flags, OVX regime, volatility regime
- Period: 2025-07~11
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import lightgbm as lgb
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Paths
FEATURES_PATH = Path("features_hourly_with_term.parquet")
EVENTS_PATH = Path("data/events_calendar.csv")
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

# Original features
BASE_FEATURES = [
    "OIL_CORE_norm_art_cnt",
    "GEOPOL_norm_art_cnt",
    "USD_RATE_norm_art_cnt",
    "SUPPLY_CHAIN_norm_art_cnt",
    "MACRO_norm_art_cnt",
    "cl1_cl2",
    "ovx",
]

TARGET_COL = "wti_returns"

# Period
DRIFT_PERIOD = ("2025-07-01", "2025-11-30")

# Short window settings
TRAIN_HOURS = 720   # 30 days
TEST_HOURS = 336    # 14 days


def load_base_data():
    """Load base feature data"""
    print("[1/8] Loading base features...")
    df = pd.read_parquet(FEATURES_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    print(f"   Base data: {len(df)} rows")
    return df


def load_events():
    """Load EIA/OPEC event calendar"""
    print("[2/8] Loading event calendar...")
    if not EVENTS_PATH.exists():
        print("   WARNING: Event calendar not found")
        return pd.DataFrame()

    events = pd.read_csv(EVENTS_PATH)
    events['ts'] = pd.to_datetime(events['ts'], utc=True)
    print(f"   Events loaded: {len(events)} records")
    print(f"   Event types: {events['event_type'].unique().tolist()}")
    return events


def add_event_features(df, events):
    """Add EIA event window flags"""
    print("[3/8] Adding event features...")

    df = df.copy()

    # Initialize event columns
    df['eia_pre_4h'] = 0  # 4 hours before EIA
    df['eia_post_4h'] = 0  # 4 hours after EIA
    df['eia_day'] = 0     # Same day as EIA

    if events.empty:
        return df

    # Get EIA events and ensure UTC timezone
    eia_events = pd.to_datetime(events[events['event_type'] == 'EIA']['ts'], utc=True)

    # Vectorized approach for efficiency
    for event_ts in eia_events:
        # Calculate hours difference for all rows at once
        hours_diff = (df.index - event_ts).total_seconds() / 3600

        # Pre-event window (4h before)
        df.loc[(hours_diff >= -4) & (hours_diff < 0), 'eia_pre_4h'] = 1
        # Post-event window (4h after)
        df.loc[(hours_diff >= 0) & (hours_diff <= 4), 'eia_post_4h'] = 1
        # Same day
        df.loc[df.index.date == event_ts.date(), 'eia_day'] = 1

    print(f"   EIA pre-4h flags: {df['eia_pre_4h'].sum()}")
    print(f"   EIA post-4h flags: {df['eia_post_4h'].sum()}")
    print(f"   EIA day flags: {df['eia_day'].sum()}")

    return df


def add_regime_features(df):
    """Add volatility and OVX regime features"""
    print("[4/8] Adding regime features...")

    df = df.copy()

    # Volatility regime (rolling 24h std of returns)
    df['vol_24h'] = df[TARGET_COL].rolling(24, min_periods=1).std()
    df['vol_regime_high'] = (df['vol_24h'] > df['vol_24h'].quantile(0.75)).astype(int)
    df['vol_regime_low'] = (df['vol_24h'] < df['vol_24h'].quantile(0.25)).astype(int)

    # OVX regime
    df['ovx_high'] = (df['ovx'] > 0.7).astype(int)
    df['ovx_low'] = (df['ovx'] < 0.3).astype(int)

    # Momentum (24h return)
    df['momentum_24h'] = df[TARGET_COL].rolling(24, min_periods=1).sum()

    # GDELT intensity (sum of all buckets)
    gdelt_cols = [c for c in BASE_FEATURES if 'norm_art_cnt' in c]
    df['gdelt_intensity'] = df[gdelt_cols].sum(axis=1)
    df['gdelt_high'] = (df['gdelt_intensity'] > df['gdelt_intensity'].quantile(0.75)).astype(int)

    print(f"   Vol regime high: {df['vol_regime_high'].sum()}")
    print(f"   OVX high: {df['ovx_high'].sum()}")
    print(f"   GDELT high: {df['gdelt_high'].sum()}")

    return df


def get_period_data(df, start, end):
    """Extract data for a specific period"""
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC")
    return df[(df.index >= start_dt) & (df.index <= end_dt)]


def winsorize(series, lower=0.01, upper=0.99):
    """Winsorize at percentiles"""
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)


def run_validation(df, feature_sets, period_name):
    """Run rolling window validation with different feature sets"""
    print(f"\n[5/8] Running validation on {period_name}...")

    results = {}

    for set_name, features in feature_sets.items():
        print(f"\n   Testing feature set: {set_name} ({len(features)} features)")

        df_test = df.copy()

        # Winsorize
        for col in features:
            if col in df_test.columns:
                df_test[col] = winsorize(df_test[col])
        df_test[TARGET_COL] = winsorize(df_test[TARGET_COL])

        # Drop rows with NaN in features
        df_test = df_test.dropna(subset=features + [TARGET_COL])

        n_samples = len(df_test)
        if n_samples < TRAIN_HOURS + TEST_HOURS:
            print(f"   WARNING: Not enough data ({n_samples} rows)")
            continue

        window_results = []
        window_idx = 0
        start_idx = TRAIN_HOURS

        while start_idx + TEST_HOURS <= n_samples:
            train_df = df_test.iloc[start_idx - TRAIN_HOURS:start_idx]
            test_df = df_test.iloc[start_idx:start_idx + TEST_HOURS]

            X_train = train_df[features]
            y_train = train_df[TARGET_COL]
            X_test = test_df[features]
            y_test = test_df[TARGET_COL]

            # Train
            model = lgb.LGBMRegressor(**MODEL_CONFIG)
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Calculate IC
            ic, _ = spearmanr(y_pred, y_test)

            test_month = test_df.index[len(test_df)//2].strftime("%Y-%m")

            window_results.append({
                'window': window_idx + 1,
                'test_month': test_month,
                'ic': ic,
                'n_test': len(test_df),
            })

            window_idx += 1
            start_idx += TEST_HOURS

        if window_results:
            results[set_name] = {
                'windows': pd.DataFrame(window_results),
                'features': features,
                'importance': dict(zip(features, model.feature_importances_)) if window_results else {},
            }

    return results


def calculate_metrics(windows_df):
    """Calculate IC/IR/PMR"""
    if windows_df is None or len(windows_df) == 0:
        return None

    ic_values = windows_df['ic'].dropna()

    return {
        'n_windows': len(windows_df),
        'ic_mean': ic_values.mean(),
        'ic_median': ic_values.median(),
        'ic_std': ic_values.std(),
        'ir': ic_values.mean() / ic_values.std() if ic_values.std() > 0 else np.nan,
        'pmr': (ic_values > 0).mean(),
        'ic_min': ic_values.min(),
        'ic_max': ic_values.max(),
    }


def main():
    print("=" * 70)
    print("DRIFT PERIOD SHORT WINDOW VALIDATION")
    print("=" * 70)
    print(f"Train: {TRAIN_HOURS}h ({TRAIN_HOURS//24}d), Test: {TEST_HOURS}h ({TEST_HOURS//24}d)")
    print(f"Period: {DRIFT_PERIOD[0]} to {DRIFT_PERIOD[1]}")
    print("=" * 70)

    # Load data
    df = load_base_data()
    events = load_events()

    # Add features
    df = add_event_features(df, events)
    df = add_regime_features(df)

    # Get drift period
    df_drift = get_period_data(df, *DRIFT_PERIOD)
    print(f"\n   Drift period data: {len(df_drift)} rows")

    # Define feature sets to test
    NEW_FEATURES = [
        'eia_pre_4h', 'eia_post_4h', 'eia_day',
        'vol_regime_high', 'vol_regime_low',
        'ovx_high', 'ovx_low',
        'momentum_24h', 'gdelt_high',
    ]

    feature_sets = {
        'base_7': BASE_FEATURES,
        'base_7_short': BASE_FEATURES,  # Same features, shorter window
        'base_7_plus_eia': BASE_FEATURES + ['eia_pre_4h', 'eia_post_4h', 'eia_day'],
        'base_7_plus_regime': BASE_FEATURES + ['vol_regime_high', 'ovx_high', 'momentum_24h'],
        'full_16': BASE_FEATURES + NEW_FEATURES,
    }

    # Run validation
    results = run_validation(df_drift, feature_sets, "Drift Period")

    # Print results
    print("\n[6/8] Results Summary")
    print("=" * 70)

    print(f"\n{'Feature Set':<25} {'Windows':<10} {'IC Median':<12} {'IR':<12} {'PMR':<10} {'Status':<10}")
    print("-" * 79)

    all_metrics = {}
    for set_name, data in results.items():
        metrics = calculate_metrics(data['windows'])
        if metrics:
            all_metrics[set_name] = metrics
            ic_status = "PASS" if metrics['ic_median'] >= 0.02 else "FAIL"
            ir_status = "PASS" if metrics['ir'] >= 0.5 else "FAIL"
            pmr_status = "PASS" if metrics['pmr'] >= 0.55 else "FAIL"
            overall = "HARD" if ic_status == "PASS" and ir_status == "PASS" and pmr_status == "PASS" else ""

            print(f"{set_name:<25} {metrics['n_windows']:<10} {metrics['ic_median']:<12.4f} {metrics['ir']:<12.4f} {metrics['pmr']:<10.2%} {overall}")

    # Window-by-window for best set
    print("\n[7/8] Window Details (Best Feature Set)")
    print("=" * 70)

    # Find best by IC median
    if all_metrics:
        best_set = max(all_metrics.keys(), key=lambda x: all_metrics[x]['ic_median'])
        print(f"\nBest set: {best_set}")
        print(f"\n{'Window':<10} {'Month':<12} {'IC':<12} {'Status':<10}")
        print("-" * 44)

        for _, row in results[best_set]['windows'].iterrows():
            status = "OK" if row['ic'] > 0 else "X"
            hard = "HARD" if row['ic'] >= 0.02 else ""
            print(f"{row['window']:<10} {row['test_month']:<12} {row['ic']:<12.6f} {status:<10} {hard}")

        # Feature importance
        print(f"\nFeature Importance ({best_set}):")
        imp = results[best_set]['importance']
        for feat, val in sorted(imp.items(), key=lambda x: -x[1])[:10]:
            print(f"   {feat:<35} {val:.1f}")

    # Conclusion
    print("\n[8/8] Conclusion")
    print("=" * 70)

    # Check if any set passes Hard Gate
    passing_sets = [s for s, m in all_metrics.items()
                   if m['ic_median'] >= 0.02 and m['ir'] >= 0.5 and m['pmr'] >= 0.55]

    positive_ic_sets = [s for s, m in all_metrics.items() if m['ic_median'] > 0]

    if passing_sets:
        print(f"\n[OK] HARD GATE PASSED with: {', '.join(passing_sets)}")
        print("   RECOMMENDATION: Deploy regime-aware model with these features")
    elif positive_ic_sets:
        best = max(positive_ic_sets, key=lambda x: all_metrics[x]['ic_median'])
        print(f"\n[~] POSITIVE IC achieved with: {best}")
        print(f"   IC median: {all_metrics[best]['ic_median']:.4f}")
        print(f"   IR: {all_metrics[best]['ir']:.4f}")
        print(f"   PMR: {all_metrics[best]['pmr']:.2%}")
        print("\n   RECOMMENDATION:")
        print("   - Model shows weak but positive signal")
        print("   - Consider reduced position sizing (50% of normal)")
        print("   - Continue monitoring; may recover with more data")
    else:
        print("\n[X] ALL FEATURE SETS SHOW NEGATIVE IC")
        print("   RECOMMENDATION:")
        print("   - HALT STRATEGY for drift period")
        print("   - GDELT signal appears fundamentally broken for recent data")
        print("   - Investigate alternative data sources or model architectures")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all window results
    for set_name, data in results.items():
        path = OUTPUT_DIR / f"drift_short_{set_name}_{timestamp}.csv"
        data['windows'].to_csv(path, index=False)

    # Save summary
    summary_rows = []
    for set_name, metrics in all_metrics.items():
        row = {'feature_set': set_name, 'n_features': len(results[set_name]['features'])}
        row.update(metrics)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / f"drift_short_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n   Summary saved: {summary_path}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    return results, all_metrics


if __name__ == "__main__":
    results, metrics = main()
