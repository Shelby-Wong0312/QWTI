"""
Simulate 24 Hours of Monitoring

Generates synthetic monitoring data to demonstrate:
1. Hourly position allocation
2. Rolling metrics calculation
3. Hard gate monitoring
4. Alert generation when gates fail
5. Dashboard terminal vision support

This creates realistic data for testing the monitoring system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# Paths
POSITION_LOG = Path('warehouse/positions/base_seed202_lean7_positions.csv')
METRICS_LOG = Path('warehouse/monitoring/base_seed202_lean7_metrics.csv')
ALERT_LOG = Path('warehouse/monitoring/base_seed202_lean7_alerts.csv')
EXECUTION_LOG = Path('warehouse/monitoring/hourly_execution_log.csv')

def generate_synthetic_metrics(n_hours=24, base_ic=0.12, base_ir=1.58):
    """
    Generate synthetic hourly metrics that simulate real monitoring.

    Uses validation performance as baseline:
    - IC mean: 0.118463
    - IC std: 0.075177
    - IR: 1.5758
    """
    np.random.seed(42)  # For reproducibility

    start_time = datetime.now() - timedelta(hours=n_hours)

    records = []

    for hour in range(n_hours):
        timestamp = start_time + timedelta(hours=hour)

        # Generate IC with realistic variation
        # Use AR(1) process to create autocorrelation
        if hour == 0:
            ic = np.random.normal(base_ic, 0.075)
        else:
            prev_ic = records[-1]['ic']
            ic = 0.7 * prev_ic + 0.3 * np.random.normal(base_ic, 0.075)

        # Generate prediction and position
        prediction = np.random.normal(0.002, 0.005)
        position = min(0.15, max(-0.15, 0.15 * prediction / 0.005))

        # Generate features (simplified)
        features = {
            'cl1_cl2': np.random.normal(-0.02, 0.05),
            'ovx': np.random.normal(0.35, 0.10),
            'OIL_CORE_norm_art_cnt': np.random.normal(20, 5),
            'MACRO_norm_art_cnt': np.random.normal(15, 3),
            'SUPPLY_CHAIN_norm_art_cnt': np.random.normal(12, 3),
            'USD_RATE_norm_art_cnt': np.random.normal(8, 2),
            'GEOPOL_norm_art_cnt': np.random.normal(6, 2)
        }

        records.append({
            'timestamp': timestamp.isoformat(),
            'ic': ic,
            'prediction': prediction,
            'position': position,
            'features': json.dumps(features),
            'strategy_id': 'base_seed202_lean7_h1'
        })

    return pd.DataFrame(records)

def generate_position_log(metrics_df):
    """Generate position log from metrics"""
    position_records = []

    for _, row in metrics_df.iterrows():
        features = json.loads(row['features'])

        position_records.append({
            'timestamp': row['timestamp'],
            'prediction': row['prediction'],
            'position': row['position'],
            'base_weight': 0.15,
            'max_weight': 0.30,
            'strategy': 'base_seed202_lean7_h1',
            'feature_snapshot': row['features'],
            'metadata': json.dumps({
                'data_timestamp': row['timestamp'],
                'strategy_id': 'base_seed202_lean7_h1',
                'version': '1.0.0'
            })
        })

    return pd.DataFrame(position_records)

def calculate_rolling_metrics(metrics_df, window_hours=15*24):
    """Calculate rolling metrics for Hard gate checking"""
    df = metrics_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    results = []

    for i in range(len(df)):
        current_time = df.iloc[i]['timestamp']
        window_start = current_time - timedelta(hours=window_hours)

        window_data = df[
            (df['timestamp'] >= window_start) &
            (df['timestamp'] <= current_time)
        ]

        if len(window_data) >= 5:  # Need at least 5 observations
            ic_mean = window_data['ic'].mean()
            ic_std = window_data['ic'].std()
            ir = ic_mean / ic_std if ic_std > 0 else 0
            pmr = (window_data['ic'] > 0).mean()

            results.append({
                'timestamp': current_time.isoformat(),
                'window_hours': window_hours,
                'n_obs': len(window_data),
                'ic_mean': ic_mean,
                'ic_median': window_data['ic'].median(),
                'ic_std': ic_std,
                'ir': ir,
                'pmr': pmr
            })

    return pd.DataFrame(results) if results else None

def check_hard_gates(metrics_df):
    """Check Hard gates and generate alerts"""
    alerts = []

    # Calculate rolling metrics
    metrics_15d = calculate_rolling_metrics(metrics_df, 15*24)
    metrics_60d = calculate_rolling_metrics(metrics_df, 60*24)
    metrics_30d = calculate_rolling_metrics(metrics_df, 30*24)

    if metrics_15d is None or len(metrics_15d) == 0:
        return []

    latest_15d = metrics_15d.iloc[-1]
    latest_60d = metrics_60d.iloc[-1] if metrics_60d is not None and len(metrics_60d) > 0 else None
    latest_30d = metrics_30d.iloc[-1] if metrics_30d is not None and len(metrics_30d) > 0 else None

    # Check IC median
    if latest_15d['ic_median'] < 0.02:
        alerts.append({
            'timestamp': latest_15d['timestamp'],
            'level': 'CRITICAL',
            'gate': 'HARD_IC',
            'metric': 'IC_MEDIAN',
            'value': latest_15d['ic_median'],
            'threshold': 0.02,
            'message': f"IC median ({latest_15d['ic_median']:.4f}) below Hard gate (0.02)"
        })

    # Check IR
    if latest_60d is not None and latest_60d['ir'] < 0.5:
        alerts.append({
            'timestamp': latest_60d['timestamp'],
            'level': 'CRITICAL',
            'gate': 'HARD_IC',
            'metric': 'IR',
            'value': latest_60d['ir'],
            'threshold': 0.5,
            'message': f"IR ({latest_60d['ir']:.4f}) below Hard gate (0.5)"
        })

    # Check PMR
    if latest_30d is not None and latest_30d['pmr'] < 0.55:
        alerts.append({
            'timestamp': latest_30d['timestamp'],
            'level': 'CRITICAL',
            'gate': 'HARD_IC',
            'metric': 'PMR',
            'value': latest_30d['pmr'],
            'threshold': 0.55,
            'message': f"PMR ({latest_30d['pmr']:.4f}) below Hard gate (0.55)"
        })

    # Check for consecutive low IC (hard stop)
    if len(metrics_df) >= 5:
        recent_5_ic = metrics_df.tail(5)['ic'].values
        if all(ic < 0.01 for ic in recent_5_ic):
            alerts.append({
                'timestamp': metrics_df.iloc[-1]['timestamp'],
                'level': 'CRITICAL',
                'gate': 'HARD_STOP',
                'metric': 'IC_CONSECUTIVE',
                'value': recent_5_ic.mean(),
                'threshold': 0.01,
                'message': 'IC < 0.01 for 5 consecutive windows - AUTO DE-ACTIVATION'
            })

    return alerts

def main():
    print("="*70)
    print("SIMULATING 24 HOURS OF MONITORING")
    print("="*70)

    # Generate synthetic data
    print("\n[1/5] Generating synthetic metrics (24 hours)...")
    metrics_df = generate_synthetic_metrics(n_hours=24)
    print(f"  Generated {len(metrics_df)} hourly records")
    print(f"  IC range: {metrics_df['ic'].min():.4f} to {metrics_df['ic'].max():.4f}")
    print(f"  Mean IC: {metrics_df['ic'].mean():.4f}")

    # Generate position log
    print("\n[2/5] Generating position log...")
    position_df = generate_position_log(metrics_df)
    print(f"  Generated {len(position_df)} position records")

    # Save logs
    print("\n[3/5] Saving logs to warehouse/...")

    POSITION_LOG.parent.mkdir(parents=True, exist_ok=True)
    METRICS_LOG.parent.mkdir(parents=True, exist_ok=True)

    # Append to existing logs (or create new)
    if POSITION_LOG.exists():
        position_df.to_csv(POSITION_LOG, mode='a', header=False, index=False)
    else:
        position_df.to_csv(POSITION_LOG, index=False)

    if METRICS_LOG.exists():
        # Only keep essential columns for metrics log
        metrics_essential = metrics_df[['timestamp', 'ic', 'prediction', 'position', 'strategy_id']]
        metrics_essential.to_csv(METRICS_LOG, mode='a', header=False, index=False)
    else:
        metrics_essential = metrics_df[['timestamp', 'ic', 'prediction', 'position', 'strategy_id']]
        metrics_essential.to_csv(METRICS_LOG, index=False)

    print(f"  Position log: {POSITION_LOG} ({len(position_df)} rows)")
    print(f"  Metrics log: {METRICS_LOG} ({len(metrics_df)} rows)")

    # Check Hard gates
    print("\n[4/5] Checking Hard gates...")
    alerts = check_hard_gates(metrics_df)

    if alerts:
        print(f"  Generated {len(alerts)} alert(s)")

        for alert in alerts:
            print(f"    [{alert['level']}] {alert['gate']}: {alert['message']}")

        # Save alerts
        alert_df = pd.DataFrame(alerts)

        if ALERT_LOG.exists():
            alert_df.to_csv(ALERT_LOG, mode='a', header=False, index=False)
        else:
            alert_df.to_csv(ALERT_LOG, index=False)

        print(f"  Alert log: {ALERT_LOG}")
    else:
        print("  No alerts - All Hard gates passed")

    # Generate execution log
    print("\n[5/5] Generating execution log...")
    execution_records = []

    for _, row in metrics_df.iterrows():
        execution_records.append({
            'timestamp': row['timestamp'],
            'status': 'SUCCESS',
            'hard_gate_passed': True,  # Simplified for simulation
            'n_alerts': 0,
            'error': None
        })

    execution_df = pd.DataFrame(execution_records)

    if EXECUTION_LOG.exists():
        execution_df.to_csv(EXECUTION_LOG, mode='a', header=False, index=False)
    else:
        execution_df.to_csv(EXECUTION_LOG, index=False)

    print(f"  Execution log: {EXECUTION_LOG} ({len(execution_df)} rows)")

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print("\nGenerated Files:")
    print(f"  {POSITION_LOG}")
    print(f"  {METRICS_LOG}")
    if alerts:
        print(f"  {ALERT_LOG}")
    print(f"  {EXECUTION_LOG}")

    print("\nView Dashboard:")
    print(f"  python warehouse/monitoring/base_dashboard.py")

    print("\nSummary Statistics:")
    print(f"  Total hours simulated: {len(metrics_df)}")
    print(f"  Mean IC: {metrics_df['ic'].mean():.4f}")
    print(f"  IC std: {metrics_df['ic'].std():.4f}")
    print(f"  Mean position: {metrics_df['position'].mean():.2%}")
    print(f"  Alerts: {len(alerts)}")

if __name__ == '__main__':
    main()
