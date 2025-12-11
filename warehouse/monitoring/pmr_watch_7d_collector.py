#!/usr/bin/env python3
"""
7-Day PMR/IR/IC Stability Collector

Collects hourly metrics for 7 days to generate a formal stability report.
Special focus on low-liquidity period PMR fluctuations.

Output files:
- warehouse/monitoring/pmr_watch_7d.csv: Raw hourly data
- warehouse/monitoring/pmr_watch_7d_report.json: Stability assessment

Run daily to append new data and regenerate report.
After 7 days of stable operation, model can be locked as long-term base.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import sys

# Paths
FEATURES_PATH = Path('features_hourly_with_clbz.parquet')
MODEL_PATH = Path('models/base_seed202_clbz_h1.pkl')
CONFIG_PATH = Path('warehouse/base_monitoring_config.json')
PMR_7D_CSV = Path('warehouse/monitoring/pmr_watch_7d.csv')
PMR_7D_REPORT = Path('warehouse/monitoring/pmr_watch_7d_report.json')

# Feature columns
FEATURE_COLS = [
    'cl_bz_spread', 'ovx',
    'OIL_CORE_norm_art_cnt', 'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt', 'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt', 'momentum_24h'
]

# Thresholds
IC_THRESHOLD = 0.02
IR_THRESHOLD = 0.5
PMR_THRESHOLD = 0.55
PMR_WATCH_THRESHOLD = 0.55
PMR_HALT_THRESHOLD = 0.548


def classify_liquidity_period(hour_utc: int) -> str:
    """Classify hour into liquidity period"""
    if 20 <= hour_utc or hour_utc < 8:
        return 'LOW'
    elif 13 <= hour_utc < 20:
        return 'HIGH'
    else:
        return 'MEDIUM'


def calculate_rolling_metrics(predictions: list, actuals: list, window: int = 360):
    """Calculate IC, IR, PMR for a rolling window"""
    if len(predictions) < window:
        window = len(predictions)

    if window < 10:
        return {'ic': np.nan, 'ir': np.nan, 'pmr': np.nan}

    recent_preds = predictions[-window:]
    recent_actuals = actuals[-window:]

    # IC (Pearson correlation)
    if np.std(recent_preds) > 0 and np.std(recent_actuals) > 0:
        ic = np.corrcoef(recent_preds, recent_actuals)[0, 1]
    else:
        ic = 0

    # IR (IC / std of IC over sub-windows)
    ic_values = []
    for i in range(24, window, 24):
        sub_preds = recent_preds[-i:]
        sub_actuals = recent_actuals[-i:]
        if len(sub_preds) > 1 and np.std(sub_preds) > 0 and np.std(sub_actuals) > 0:
            sub_ic = np.corrcoef(sub_preds, sub_actuals)[0, 1]
            ic_values.append(sub_ic)

    ic_std = np.std(ic_values) if ic_values else 0.001
    ir = ic / ic_std if ic_std > 0 else 0

    # PMR (Positive Mean Ratio)
    pmr = np.mean([1 if (p > 0 and a > 0) or (p < 0 and a < 0) else 0
                   for p, a in zip(recent_preds, recent_actuals)])

    return {'ic': ic, 'ir': ir, 'pmr': pmr}


def classify_pmr_zone(pmr: float) -> str:
    """Classify PMR into zone"""
    if pd.isna(pmr):
        return 'NO_DATA'
    if pmr >= PMR_WATCH_THRESHOLD:
        return 'NORMAL'
    elif pmr >= PMR_HALT_THRESHOLD:
        return 'WATCH'
    else:
        return 'HALT'


def collect_7d_data():
    """Collect 7 days of hourly metrics data"""
    print("="*70)
    print("7-DAY PMR/IR/IC STABILITY COLLECTOR")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model and config
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    print(f"Model: {MODEL_PATH.name}")
    print(f"Strategy: {config['strategy_name']} v{config['version']}")

    # Load features
    df = pd.read_parquet(FEATURES_PATH)
    print(f"Features loaded: {len(df)} rows")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Prepare target
    df['target'] = df['wti_returns'].shift(-1)
    df = df.dropna(subset=['target'])

    # Calculate window: 7 days test + 15 days lookback
    test_hours = 7 * 24  # 168 hours
    lookback_hours = 360  # 15 days for rolling metrics

    end_ts = df.index.max()
    start_ts = end_ts - timedelta(hours=lookback_hours + test_hours)
    df_window = df[df.index >= start_ts].copy()

    print(f"\nAnalysis window: {df_window.index.min()} to {df_window.index.max()}")
    print(f"Total hours in window: {len(df_window)}")

    # Split into lookback and test periods
    test_start_idx = len(df_window) - test_hours

    # Generate predictions for lookback period
    lookback_df = df_window.iloc[:test_start_idx]
    lookback_preds = []
    lookback_actuals = []

    print(f"\nProcessing lookback period ({len(lookback_df)} hours)...")
    for idx in lookback_df.index:
        row = lookback_df.loc[idx]
        X = pd.DataFrame([row[FEATURE_COLS]])[FEATURE_COLS]
        pred = float(model.predict(X)[0])
        lookback_preds.append(pred)
        lookback_actuals.append(row['target'])

    # Generate predictions and metrics for test period (7 days)
    test_df = df_window.iloc[test_start_idx:]
    results = []

    print(f"\nProcessing 7-day test period ({len(test_df)} hours)...")

    all_preds = lookback_preds.copy()
    all_actuals = lookback_actuals.copy()

    for i, (idx, row) in enumerate(test_df.iterrows()):
        # Predict
        X = pd.DataFrame([row[FEATURE_COLS]])[FEATURE_COLS]
        pred = float(model.predict(X)[0])
        actual = row['target']

        all_preds.append(pred)
        all_actuals.append(actual)

        # Calculate rolling metrics
        metrics = calculate_rolling_metrics(all_preds, all_actuals, window=360)

        # Classify
        hour_utc = idx.hour
        liquidity_period = classify_liquidity_period(hour_utc)
        pmr_zone = classify_pmr_zone(metrics['pmr'])

        # Check thresholds
        ic_ok = metrics['ic'] >= IC_THRESHOLD if not pd.isna(metrics['ic']) else False
        ir_ok = metrics['ir'] >= IR_THRESHOLD if not pd.isna(metrics['ir']) else False
        pmr_ok = metrics['pmr'] >= PMR_THRESHOLD if not pd.isna(metrics['pmr']) else False
        hard_gate_passed = ic_ok and ir_ok and pmr_ok

        results.append({
            'timestamp': idx,
            'hour_utc': hour_utc,
            'day_of_week': idx.dayofweek,
            'liquidity_period': liquidity_period,
            'prediction': pred,
            'actual': actual,
            'ic': metrics['ic'],
            'ir': metrics['ir'],
            'pmr': metrics['pmr'],
            'pmr_zone': pmr_zone,
            'ic_ok': ic_ok,
            'ir_ok': ir_ok,
            'pmr_ok': pmr_ok,
            'hard_gate_passed': hard_gate_passed
        })

        # Progress update every day
        if (i + 1) % 24 == 0:
            day_num = (i + 1) // 24
            print(f"  Day {day_num}/7: IC={metrics['ic']:.4f}, IR={metrics['ir']:.2f}, PMR={metrics['pmr']:.3f}, Zone={pmr_zone}")

    # Save raw data
    df_results = pd.DataFrame(results)
    PMR_7D_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(PMR_7D_CSV, index=False)
    print(f"\nRaw data saved: {PMR_7D_CSV}")

    return df_results


def generate_stability_report(df: pd.DataFrame):
    """Generate formal stability report"""
    print("\n" + "="*70)
    print("GENERATING STABILITY REPORT")
    print("="*70)

    # Overall metrics
    overall = {
        'ic_mean': df['ic'].mean(),
        'ic_std': df['ic'].std(),
        'ic_min': df['ic'].min(),
        'ic_max': df['ic'].max(),
        'ir_mean': df['ir'].mean(),
        'ir_std': df['ir'].std(),
        'ir_min': df['ir'].min(),
        'ir_max': df['ir'].max(),
        'pmr_mean': df['pmr'].mean(),
        'pmr_std': df['pmr'].std(),
        'pmr_min': df['pmr'].min(),
        'pmr_max': df['pmr'].max()
    }

    # Compliance rates
    compliance = {
        'ic_compliance': df['ic_ok'].mean(),
        'ir_compliance': df['ir_ok'].mean(),
        'pmr_compliance': df['pmr_ok'].mean(),
        'hard_gate_compliance': df['hard_gate_passed'].mean()
    }

    # PMR zone distribution
    pmr_zones = df['pmr_zone'].value_counts(normalize=True).to_dict()

    # Liquidity period analysis
    liquidity_analysis = {}
    for period in ['LOW', 'MEDIUM', 'HIGH']:
        period_df = df[df['liquidity_period'] == period]
        if len(period_df) > 0:
            liquidity_analysis[period] = {
                'n_hours': len(period_df),
                'pmr_mean': period_df['pmr'].mean(),
                'pmr_std': period_df['pmr'].std(),
                'pmr_min': period_df['pmr'].min(),
                'pmr_compliance': period_df['pmr_ok'].mean(),
                'hard_gate_compliance': period_df['hard_gate_passed'].mean(),
                'watch_zone_hours': len(period_df[period_df['pmr_zone'] == 'WATCH']),
                'halt_zone_hours': len(period_df[period_df['pmr_zone'] == 'HALT'])
            }

    # Day-of-week analysis
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_analysis = {}
    for dow in range(7):
        dow_df = df[df['day_of_week'] == dow]
        if len(dow_df) > 0:
            dow_analysis[dow_names[dow]] = {
                'n_hours': len(dow_df),
                'pmr_mean': dow_df['pmr'].mean(),
                'hard_gate_compliance': dow_df['hard_gate_passed'].mean()
            }

    # Violations analysis
    violations = df[~df['hard_gate_passed']]
    violation_analysis = {
        'total_violations': len(violations),
        'violation_rate': len(violations) / len(df) if len(df) > 0 else 0,
        'pmr_violations': len(df[~df['pmr_ok']]),
        'ic_violations': len(df[~df['ic_ok']]),
        'ir_violations': len(df[~df['ir_ok']])
    }

    # Low liquidity specific analysis
    low_liq = df[df['liquidity_period'] == 'LOW']
    low_liq_analysis = {
        'n_hours': len(low_liq),
        'pmr_mean': low_liq['pmr'].mean() if len(low_liq) > 0 else None,
        'pmr_std': low_liq['pmr'].std() if len(low_liq) > 0 else None,
        'violations_in_low_liq': len(low_liq[~low_liq['hard_gate_passed']]) if len(low_liq) > 0 else 0,
        'pct_of_all_violations': len(low_liq[~low_liq['hard_gate_passed']]) / len(violations) if len(violations) > 0 else 0
    }

    # Stability assessment
    is_stable = (
        compliance['hard_gate_compliance'] >= 0.85 and  # 85%+ compliance
        overall['pmr_min'] >= PMR_HALT_THRESHOLD and    # Never hit HALT zone
        overall['ic_mean'] >= IC_THRESHOLD and          # Mean IC above threshold
        overall['ir_mean'] >= IR_THRESHOLD              # Mean IR above threshold
    )

    stability_verdict = 'STABLE' if is_stable else 'NEEDS_REVIEW'

    # Recommendation
    if is_stable:
        recommendation = "Model stable for 7 days. Ready to lock as long-term base."
    elif compliance['hard_gate_compliance'] >= 0.80:
        recommendation = "Model marginally stable. Continue monitoring for another 7 days."
    else:
        recommendation = "Model unstable. Review required before promotion."

    # Build report
    report = {
        'report_generated': datetime.now().isoformat(),
        'model': str(MODEL_PATH),
        'period': {
            'start': str(df['timestamp'].min()),
            'end': str(df['timestamp'].max()),
            'total_hours': len(df)
        },
        'overall_metrics': overall,
        'compliance': compliance,
        'pmr_zone_distribution': pmr_zones,
        'liquidity_period_analysis': liquidity_analysis,
        'day_of_week_analysis': dow_analysis,
        'violation_analysis': violation_analysis,
        'low_liquidity_focus': low_liq_analysis,
        'stability_assessment': {
            'verdict': stability_verdict,
            'is_stable': is_stable,
            'recommendation': recommendation
        },
        'thresholds': {
            'IC': IC_THRESHOLD,
            'IR': IR_THRESHOLD,
            'PMR': PMR_THRESHOLD,
            'PMR_WATCH': PMR_WATCH_THRESHOLD,
            'PMR_HALT': PMR_HALT_THRESHOLD
        }
    }

    # Save report
    with open(PMR_7D_REPORT, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved: {PMR_7D_REPORT}")

    return report


def print_summary(report: dict):
    """Print human-readable summary"""
    print("\n" + "="*70)
    print("7-DAY STABILITY SUMMARY")
    print("="*70)

    print(f"""
Period: {report['period']['start']} to {report['period']['end']}
Total Hours: {report['period']['total_hours']}

OVERALL METRICS:
  IC:  {report['overall_metrics']['ic_mean']:.4f} +/- {report['overall_metrics']['ic_std']:.4f} (range: {report['overall_metrics']['ic_min']:.4f} to {report['overall_metrics']['ic_max']:.4f})
  IR:  {report['overall_metrics']['ir_mean']:.2f} +/- {report['overall_metrics']['ir_std']:.2f} (range: {report['overall_metrics']['ir_min']:.2f} to {report['overall_metrics']['ir_max']:.2f})
  PMR: {report['overall_metrics']['pmr_mean']:.3f} +/- {report['overall_metrics']['pmr_std']:.3f} (range: {report['overall_metrics']['pmr_min']:.3f} to {report['overall_metrics']['pmr_max']:.3f})

COMPLIANCE RATES:
  IC compliance:  {report['compliance']['ic_compliance']:.1%}
  IR compliance:  {report['compliance']['ir_compliance']:.1%}
  PMR compliance: {report['compliance']['pmr_compliance']:.1%}
  Hard gate:      {report['compliance']['hard_gate_compliance']:.1%}

PMR ZONE DISTRIBUTION:""")

    for zone, pct in report['pmr_zone_distribution'].items():
        print(f"  {zone}: {pct:.1%}")

    print("\nLIQUIDITY PERIOD ANALYSIS:")
    for period, data in report['liquidity_period_analysis'].items():
        print(f"  {period}: {data['n_hours']}h, PMR={data['pmr_mean']:.3f}+/-{data['pmr_std']:.3f}, Compliance={data['hard_gate_compliance']:.1%}")

    print("\nLOW LIQUIDITY FOCUS:")
    low_liq = report['low_liquidity_focus']
    print(f"  Hours in low liquidity: {low_liq['n_hours']}")
    print(f"  PMR in low liquidity: {low_liq['pmr_mean']:.3f} +/- {low_liq['pmr_std']:.3f}")
    print(f"  Violations in low liquidity: {low_liq['violations_in_low_liq']} ({low_liq['pct_of_all_violations']:.1%} of all violations)")

    print(f"""
VIOLATIONS:
  Total: {report['violation_analysis']['total_violations']} ({report['violation_analysis']['violation_rate']:.1%})
  PMR violations: {report['violation_analysis']['pmr_violations']}
  IC violations: {report['violation_analysis']['ic_violations']}
  IR violations: {report['violation_analysis']['ir_violations']}

STABILITY ASSESSMENT:
  Verdict: {report['stability_assessment']['verdict']}
  Recommendation: {report['stability_assessment']['recommendation']}
""")

    print("="*70)


def main():
    """Main entry point"""
    try:
        # Collect data
        df = collect_7d_data()

        # Generate report
        report = generate_stability_report(df)

        # Print summary
        print_summary(report)

        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return 0
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
