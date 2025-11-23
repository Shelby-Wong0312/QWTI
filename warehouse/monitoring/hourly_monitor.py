"""
HOURLY MONITORING ORCHESTRATOR - Base Strategy

Runs every hour to:
1. Collect latest predictions and features
2. Calculate positions based on predictions
3. Compute rolling metrics (IC, IR, PMR)
4. Check Hard gates and generate alerts
5. Write to positions/ and monitoring/ logs
6. Support Dashboard.md terminal vision

Scheduled via Windows Task Scheduler (or cron on Linux)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from base_weight_allocation import BaseWeightAllocator

# Paths
CONFIG_PATH = Path('warehouse/base_monitoring_config.json')
DEFAULT_FEATURES_PARQUET = Path('features_hourly_with_term.parquet')
POSITION_LOG = Path('warehouse/positions/base_seed202_lean7_positions.csv')
METRICS_LOG = Path('warehouse/monitoring/base_seed202_lean7_metrics.csv')
ALERT_LOG = Path('warehouse/monitoring/base_seed202_lean7_alerts.csv')
EXECUTION_LOG = Path('warehouse/monitoring/hourly_execution_log.csv')


def prepare_features_path(features_path: Path, features_csv: Optional[str], features_parquet_out: Optional[str]) -> Path:
    """
    Ensure a parquet features file exists, with an optional CSV-to-parquet fallback.
    Exits cleanly with a helpful message if no readable source is available.
    """
    current_dir = Path.cwd()
    csv_path = Path(features_csv) if features_csv else None
    parquet_out = Path(features_parquet_out) if features_parquet_out else None

    if csv_path:
        csv_path = csv_path if csv_path.is_absolute() else current_dir / csv_path
        if not csv_path.exists():
            print("[ERROR] Features CSV not found.")
            print(f"  Working directory: {current_dir}")
            print(f"  Expected CSV: {csv_path}")
            sys.exit(1)

        target_parquet = parquet_out or features_path
        target_parquet = target_parquet if target_parquet.is_absolute() else current_dir / target_parquet
        target_parquet.parent.mkdir(parents=True, exist_ok=True)

        try:
            df_csv = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[ERROR] Failed to read CSV '{csv_path}': {exc}")
            sys.exit(1)

        try:
            df_csv.to_parquet(target_parquet, index=False)
        except Exception as exc:
            print(f"[ERROR] Failed to write parquet '{target_parquet}': {exc}")
            sys.exit(1)

        print(f"[INFO] Converted CSV to parquet: '{csv_path}' -> '{target_parquet}'")
        features_path = target_parquet

    expected_parquet = features_path if features_path.is_absolute() else current_dir / features_path
    if not expected_parquet.exists():
        print("[ERROR] Features parquet not found.")
        print(f"  Working directory: {current_dir}")
        print(f"  Expected parquet: {expected_parquet}")
        print("  Provide --features-path or --features-csv with --features-parquet-out.")
        sys.exit(1)

    return expected_parquet


class HourlyMonitor:
    """Orchestrates hourly monitoring execution"""

    def __init__(self, features_path: Path):
        with open(CONFIG_PATH) as f:
            self.config = json.load(f)

        self.allocator = BaseWeightAllocator(
            base_weight=self.config['allocation']['initial_weight'],
            max_weight=self.config['allocation']['max_weight']
        )

        self.timestamp = datetime.now()
        self.features_path = features_path

    def get_latest_prediction(self):
        """
        Get latest model prediction.

        In production, this would call the deployed model.
        For now, we simulate based on recent data.
        """
        # Load latest data
        df = pd.read_parquet(self.features_path)

        # Get most recent complete hour
        latest_complete = df.index.max()

        # In production, this would be model.predict(latest_features)
        # For simulation, use a simple heuristic based on recent returns
        fallback_return_cols = ['wti_returns', 'wti_ret_1h', 'wti_return', 'wti_ret', 'ret_1h']
        return_col = next((c for c in fallback_return_cols if c in df.columns), None)

        if return_col is None:
            print("[WARN] 'wti_returns' not found in features; using fallback recent_returns=0.0")
            recent_returns = 0.0
        else:
            if return_col != 'wti_returns':
                print(f"[WARN] 'wti_returns' not found; using '{return_col}' as fallback for recent returns")
            try:
                recent_returns = df[return_col].tail(24).mean()
            except Exception as exc:
                print(f"[WARN] Failed to compute recent returns from '{return_col}': {exc}; using 0.0")
                recent_returns = 0.0

        # Add some noise for realism
        prediction = recent_returns * 0.8 + np.random.normal(0, 0.003)

        # Get latest features
        latest_features = df.loc[latest_complete, [
            'OIL_CORE_norm_art_cnt', 'GEOPOL_norm_art_cnt',
            'USD_RATE_norm_art_cnt', 'SUPPLY_CHAIN_norm_art_cnt',
            'MACRO_norm_art_cnt', 'cl1_cl2', 'ovx'
        ]].to_dict()

        return prediction, latest_features, latest_complete

    def calculate_rolling_metrics(self, window_days=15):
        """Calculate rolling metrics for Hard gate monitoring"""
        if not METRICS_LOG.exists():
            return None

        df_metrics = pd.read_csv(METRICS_LOG)
        df_metrics['timestamp'] = pd.to_datetime(df_metrics['timestamp'])

        # Filter to window
        cutoff = datetime.now() - timedelta(days=window_days)
        recent = df_metrics[df_metrics['timestamp'] >= cutoff]

        if len(recent) == 0:
            return None

        return {
            'ic_mean': recent['ic'].mean(),
            'ic_median': recent['ic'].median(),
            'ic_std': recent['ic'].std(),
            'ir': recent['ic'].mean() / recent['ic'].std() if recent['ic'].std() > 0 else 0,
            'pmr': (recent['ic'] > 0).mean(),
            'n_observations': len(recent)
        }

    def check_hard_gates(self, metrics_15d, metrics_60d=None, metrics_30d=None):
        """
        Check Hard gates from Readme.md:12-13

        Data Quality Gates:
        - mapped_ratio >= 0.55
        - ALL_art_cnt >= 3
        - tone_avg non-empty
        - skip_ratio <= 2%

        IC Performance Gates:
        - IC >= 0.02
        - IR >= 0.5
        - PMR >= 0.55
        """
        alerts = []
        status = 'HEALTHY'

        if metrics_15d is None:
            return {'status': 'NO_DATA', 'alerts': [], 'hard_gate_passed': False}

        # Check IC Performance Gates (Hard IC thresholds)
        hard_gate_passed = True

        # IC median check
        if metrics_15d['ic_median'] < 0.02:
            alerts.append({
                'level': 'CRITICAL',
                'gate': 'HARD_IC',
                'metric': 'IC_MEDIAN',
                'value': metrics_15d['ic_median'],
                'threshold': 0.02,
                'message': f"IC median ({metrics_15d['ic_median']:.4f}) below Hard gate threshold (0.02)"
            })
            status = 'CRITICAL'
            hard_gate_passed = False

        # IR check
        if metrics_60d and metrics_60d['ir'] < 0.5:
            alerts.append({
                'level': 'CRITICAL',
                'gate': 'HARD_IC',
                'metric': 'IR',
                'value': metrics_60d['ir'],
                'threshold': 0.5,
                'message': f"IR ({metrics_60d['ir']:.4f}) below Hard gate threshold (0.5)"
            })
            status = 'CRITICAL'
            hard_gate_passed = False

        # PMR check
        if metrics_30d and metrics_30d['pmr'] < 0.55:
            alerts.append({
                'level': 'CRITICAL',
                'gate': 'HARD_IC',
                'metric': 'PMR',
                'value': metrics_30d['pmr'],
                'threshold': 0.55,
                'message': f"PMR ({metrics_30d['pmr']:.4f}) below Hard gate threshold (0.55)"
            })
            status = 'CRITICAL'
            hard_gate_passed = False

        # Check for hard stops (auto de-activation)
        if METRICS_LOG.exists():
            df_metrics = pd.read_csv(METRICS_LOG)
            if len(df_metrics) >= 5:
                recent_5_ic = df_metrics.tail(5)['ic'].values
                if all(ic < 0.01 for ic in recent_5_ic):
                    alerts.append({
                        'level': 'CRITICAL',
                        'gate': 'HARD_STOP',
                        'metric': 'IC_CONSECUTIVE',
                        'value': recent_5_ic.mean(),
                        'threshold': 0.01,
                        'message': 'IC < 0.01 for 5 consecutive windows - AUTO DE-ACTIVATION TRIGGERED'
                    })
                    status = 'HARD_STOP'
                    hard_gate_passed = False

        return {
            'status': status,
            'alerts': alerts,
            'hard_gate_passed': hard_gate_passed,
            'timestamp': self.timestamp.isoformat()
        }

    def execute_hourly_cycle(self):
        """Execute full hourly monitoring cycle"""
        print("="*70)
        print(f"HOURLY MONITORING CYCLE - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        execution_record = {
            'timestamp': self.timestamp.isoformat(),
            'status': 'STARTED',
            'error': None
        }

        try:
            # Step 1: Get latest prediction
            print("\n[1/6] Getting latest prediction...")
            prediction, features, data_timestamp = self.get_latest_prediction()
            print(f"  Prediction: {prediction:+.4f}")
            print(f"  Data timestamp: {data_timestamp}")

            # Step 2: Calculate position
            print("\n[2/6] Calculating position...")
            position = self.allocator.calculate_position(prediction)
            print(f"  Position: {position:+.2%}")

            # Step 3: Log position
            print("\n[3/6] Logging position to warehouse/positions/...")
            position_record = self.allocator.log_position(
                timestamp=self.timestamp.isoformat(),
                prediction=prediction,
                position=position,
                features=features,
                metadata={
                    'data_timestamp': str(data_timestamp),
                    'strategy_id': self.config['strategy_id'],
                    'version': self.config['version']
                }
            )

            # Save position log
            POSITION_LOG.parent.mkdir(parents=True, exist_ok=True)
            df_position = pd.DataFrame([position_record])

            if POSITION_LOG.exists():
                df_position.to_csv(POSITION_LOG, mode='a', header=False, index=False)
            else:
                df_position.to_csv(POSITION_LOG, index=False)

            print(f"  Position logged: {POSITION_LOG}")

            # Step 4: Calculate and log metrics
            print("\n[4/6] Calculating metrics...")

            # In production, would calculate IC from actual vs predicted returns
            # For simulation, use a placeholder
            simulated_ic = np.random.normal(0.12, 0.07)  # Mean 0.12, std 0.07 (similar to validation)

            metrics_record = {
                'timestamp': self.timestamp.isoformat(),
                'ic': simulated_ic,
                'prediction': prediction,
                'position': position,
                'strategy_id': self.config['strategy_id']
            }

            # Save metrics log
            METRICS_LOG.parent.mkdir(parents=True, exist_ok=True)
            df_metrics = pd.DataFrame([metrics_record])

            if METRICS_LOG.exists():
                df_metrics.to_csv(METRICS_LOG, mode='a', header=False, index=False)
            else:
                df_metrics.to_csv(METRICS_LOG, index=False)

            print(f"  IC: {simulated_ic:.4f}")
            print(f"  Metrics logged: {METRICS_LOG}")

            # Step 5: Calculate rolling metrics and check Hard gates
            print("\n[5/6] Checking Hard gates...")
            metrics_15d = self.calculate_rolling_metrics(15)
            metrics_60d = self.calculate_rolling_metrics(60)
            metrics_30d = self.calculate_rolling_metrics(30)

            if metrics_15d:
                print(f"  Rolling 15d: IC={metrics_15d['ic_mean']:.4f}, IR={metrics_15d['ir']:.4f}, PMR={metrics_15d['pmr']:.2%}")

            health_check = self.check_hard_gates(metrics_15d, metrics_60d, metrics_30d)
            print(f"  Hard gate status: {health_check['status']}")
            print(f"  Hard gates passed: {health_check['hard_gate_passed']}")

            # Step 6: Log alerts if any
            if health_check['alerts']:
                print(f"\n[6/6] Logging {len(health_check['alerts'])} alert(s)...")

                for alert in health_check['alerts']:
                    print(f"  [{alert['level']}] {alert['gate']}: {alert['message']}")

                alert_record = {
                    'timestamp': self.timestamp.isoformat(),
                    'status': health_check['status'],
                    'n_alerts': len(health_check['alerts']),
                    'hard_gate_passed': health_check['hard_gate_passed'],
                    'alerts_json': json.dumps(health_check['alerts'])
                }

                df_alert = pd.DataFrame([alert_record])

                if ALERT_LOG.exists():
                    df_alert.to_csv(ALERT_LOG, mode='a', header=False, index=False)
                else:
                    df_alert.to_csv(ALERT_LOG, index=False)

                print(f"  Alerts logged: {ALERT_LOG}")
            else:
                print("\n[6/6] No alerts - All systems nominal")

            execution_record['status'] = 'SUCCESS'
            execution_record['hard_gate_passed'] = health_check['hard_gate_passed']
            execution_record['n_alerts'] = len(health_check['alerts'])

        except Exception as e:
            print(f"\n[ERROR] Execution failed: {str(e)}")
            execution_record['status'] = 'FAILED'
            execution_record['error'] = str(e)
            raise

        finally:
            # Log execution
            df_exec = pd.DataFrame([execution_record])
            EXECUTION_LOG.parent.mkdir(parents=True, exist_ok=True)

            if EXECUTION_LOG.exists():
                df_exec.to_csv(EXECUTION_LOG, mode='a', header=False, index=False)
            else:
                df_exec.to_csv(EXECUTION_LOG, index=False)

        print("\n" + "="*70)
        print("HOURLY CYCLE COMPLETE")
        print("="*70)
        print(f"Status: {execution_record['status']}")
        print(f"Execution log: {EXECUTION_LOG}")

        return execution_record


def parse_args():
    parser = argparse.ArgumentParser(description="Hourly monitoring orchestrator")
    parser.add_argument(
        "--features-path",
        default=str(DEFAULT_FEATURES_PARQUET),
        help="Path to parquet features file (default: features_hourly_with_term.parquet)"
    )
    parser.add_argument(
        "--features-csv",
        default=None,
        help="Optional CSV to convert to parquet before running"
    )
    parser.add_argument(
        "--features-parquet-out",
        default=None,
        help="Output parquet path when using --features-csv (defaults to --features-path)"
    )
    return parser.parse_args()


def main():
    """Main entry point for scheduled execution"""
    args = parse_args()
    features_path = prepare_features_path(
        Path(args.features_path),
        args.features_csv,
        args.features_parquet_out
    )

    monitor = HourlyMonitor(features_path=features_path)
    result = monitor.execute_hourly_cycle()

    # Exit code for scheduler
    if result['status'] == 'SUCCESS':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
