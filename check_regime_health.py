#!/usr/bin/env python3
"""
Quick health check for base_seed202_regime model

Run periodically to monitor Hard Gate status after production switch.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

METRICS_PATH = Path('warehouse/monitoring/base_seed202_regime_metrics.csv')
ALERTS_PATH = Path('warehouse/monitoring/base_seed202_regime_alerts.csv')
POSITIONS_PATH = Path('warehouse/positions/base_seed202_regime_positions.csv')

# Hard Gate thresholds
THRESHOLDS = {
    'ic_median_min': 0.02,
    'ir_min': 0.5,
    'pmr_min': 0.55,
}

def check_health():
    print("=" * 60)
    print(f"REGIME MODEL HEALTH CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Check metrics file
    if not METRICS_PATH.exists():
        print("\n[WARN] No metrics file found. Model has not run yet.")
        return

    df = pd.read_csv(METRICS_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    n_records = len(df)

    print(f"\n[INFO] Total records: {n_records}")
    print(f"[INFO] First record: {df['timestamp'].min()}")
    print(f"[INFO] Last record:  {df['timestamp'].max()}")

    if n_records < 15:
        print(f"\n[WAIT] Need {15 - n_records} more records for valid 15d metrics")
        print(f"       Current IC values: {df['ic'].tolist()}")
        return

    # Calculate 15d rolling metrics
    recent = df.tail(15)
    ic_median = recent['ic'].median()
    ic_mean = recent['ic'].mean()
    ic_std = recent['ic'].std()
    ir = ic_mean / ic_std if ic_std > 0 else 0
    pmr = (recent['ic'] > 0).mean()

    print("\n" + "-" * 40)
    print("15-DAY ROLLING METRICS")
    print("-" * 40)

    # IC check
    ic_status = "PASS" if ic_median >= THRESHOLDS['ic_median_min'] else "FAIL"
    print(f"IC median: {ic_median:.4f} (threshold: {THRESHOLDS['ic_median_min']}) [{ic_status}]")

    # IR check
    ir_status = "PASS" if ir >= THRESHOLDS['ir_min'] else "FAIL"
    print(f"IR:        {ir:.4f} (threshold: {THRESHOLDS['ir_min']}) [{ir_status}]")

    # PMR check
    pmr_status = "PASS" if pmr >= THRESHOLDS['pmr_min'] else "FAIL"
    print(f"PMR:       {pmr:.2%} (threshold: {THRESHOLDS['pmr_min']:.0%}) [{pmr_status}]")

    # Overall status
    print("\n" + "-" * 40)
    all_pass = ic_status == "PASS" and ir_status == "PASS" and pmr_status == "PASS"

    if all_pass:
        print("[OK] HARD GATE: HEALTHY")
    else:
        print("[CRITICAL] HARD GATE: FAILED")
        print("\nRECOMMENDED ACTIONS:")
        if ic_status == "FAIL":
            print("  - IC below threshold: consider model retrain")
        if ir_status == "FAIL":
            print("  - IR below threshold: reduce position size or pause")
        if pmr_status == "FAIL":
            print("  - PMR below threshold: check for regime shift")

    # Recent positions
    if POSITIONS_PATH.exists():
        pos_df = pd.read_csv(POSITIONS_PATH)
        print("\n" + "-" * 40)
        print("RECENT POSITIONS (last 5)")
        print("-" * 40)
        for _, row in pos_df.tail(5).iterrows():
            ts = row['timestamp'][:19]
            pred = row['prediction']
            pos = row['position']
            print(f"  {ts}  pred={pred:+.4f}  pos={pos:+.2%}")

    # Check alerts
    if ALERTS_PATH.exists():
        alerts_df = pd.read_csv(ALERTS_PATH)
        recent_alerts = alerts_df.tail(3)
        if len(recent_alerts) > 0:
            print("\n" + "-" * 40)
            print("RECENT ALERTS")
            print("-" * 40)
            for _, row in recent_alerts.iterrows():
                print(f"  {row['timestamp'][:19]}: {row['status']}")

if __name__ == "__main__":
    check_health()
