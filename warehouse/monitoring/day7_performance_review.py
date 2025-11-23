"""
Day-7 Performance and Risk Control Review

Validates Base strategy performance after first week of monitoring:
1. Hard gate compliance (IC median >= 0.02, IR >= 0.5, PMR >= 0.55)
2. Drawdown analysis (max drawdown, current drawdown)
3. Alert summary (count, severity, resolution status)
4. Position statistics (utilization, turnover, extremes)
5. Data integrity checks (no gaps, timestamp consistency)

Generates comprehensive audit report for RUNLOG_OPERATIONS.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys

# Paths
METRICS_LOG = Path('warehouse/monitoring/base_seed202_lean7_metrics.csv')
POSITION_LOG = Path('warehouse/positions/base_seed202_lean7_positions.csv')
ALERT_LOG = Path('warehouse/monitoring/base_seed202_lean7_alerts.csv')
EXECUTION_LOG = Path('warehouse/monitoring/hourly_execution_log.csv')
MONITORING_CONFIG = Path('warehouse/base_monitoring_config.json')

class Day7PerformanceReview:
    """Comprehensive performance review for Day-7 audit"""

    def __init__(self):
        self.timestamp = datetime.now()
        self.report = {
            'review_date': self.timestamp.isoformat(),
            'review_type': 'Day-7 First Performance Review',
            'status': 'PENDING'
        }

        # Load config
        with open(MONITORING_CONFIG) as f:
            self.config = json.load(f)

        # Load data
        self.load_monitoring_data()

    def load_monitoring_data(self):
        """Load all monitoring data files"""
        print("[1/8] Loading monitoring data files...")

        if not METRICS_LOG.exists():
            raise FileNotFoundError(f"Metrics log not found: {METRICS_LOG}")

        self.metrics_df = pd.read_csv(METRICS_LOG)
        self.metrics_df['timestamp'] = pd.to_datetime(self.metrics_df['timestamp'])
        self.metrics_df = self.metrics_df.sort_values('timestamp')

        print(f"  Metrics: {len(self.metrics_df)} records")
        print(f"  Date range: {self.metrics_df['timestamp'].min()} to {self.metrics_df['timestamp'].max()}")

        if POSITION_LOG.exists():
            self.positions_df = pd.read_csv(POSITION_LOG)
            self.positions_df['timestamp'] = pd.to_datetime(self.positions_df['timestamp'])
            print(f"  Positions: {len(self.positions_df)} records")
        else:
            self.positions_df = None
            print("  Positions: No data")

        if ALERT_LOG.exists():
            self.alerts_df = pd.read_csv(ALERT_LOG)
            self.alerts_df['timestamp'] = pd.to_datetime(self.alerts_df['timestamp'])
            print(f"  Alerts: {len(self.alerts_df)} records")
        else:
            self.alerts_df = None
            print("  Alerts: No data")

        if EXECUTION_LOG.exists():
            self.execution_df = pd.read_csv(EXECUTION_LOG)
            self.execution_df['timestamp'] = pd.to_datetime(self.execution_df['timestamp'])
            print(f"  Execution: {len(self.execution_df)} records")
        else:
            self.execution_df = None
            print("  Execution: No data")

    def calculate_rolling_metrics(self, window_hours):
        """Calculate rolling metrics for a given window"""
        if len(self.metrics_df) == 0:
            return None

        cutoff = self.metrics_df['timestamp'].max() - timedelta(hours=window_hours)
        window_data = self.metrics_df[self.metrics_df['timestamp'] >= cutoff]

        if len(window_data) < 5:
            return None

        ic_mean = window_data['ic'].mean()
        ic_std = window_data['ic'].std()
        ic_median = window_data['ic'].median()
        ir = ic_mean / ic_std if ic_std > 0 else 0
        pmr = (window_data['ic'] > 0).mean()

        return {
            'window_hours': window_hours,
            'window_days': window_hours / 24,
            'n_observations': len(window_data),
            'ic_mean': ic_mean,
            'ic_median': ic_median,
            'ic_std': ic_std,
            'ir': ir,
            'pmr': pmr,
            'ic_min': window_data['ic'].min(),
            'ic_max': window_data['ic'].max()
        }

    def check_hard_gates(self):
        """Check Hard gate compliance"""
        print("\n[2/8] Checking Hard gate compliance...")

        # Calculate rolling metrics for different windows
        metrics_15d = self.calculate_rolling_metrics(15 * 24)  # 15 days = 360 hours
        metrics_30d = self.calculate_rolling_metrics(30 * 24)  # 30 days = 720 hours
        metrics_60d = self.calculate_rolling_metrics(60 * 24)  # 60 days = 1440 hours

        # For current data (likely < 7 days), use available data
        metrics_available = self.calculate_rolling_metrics(len(self.metrics_df))

        hard_gates = {
            'ic_median_gate': {
                'threshold': 0.02,
                'description': 'IC median >= 0.02',
                'status': 'N/A',
                'value': None
            },
            'ir_gate': {
                'threshold': 0.5,
                'description': 'IR >= 0.5',
                'status': 'N/A',
                'value': None
            },
            'pmr_gate': {
                'threshold': 0.55,
                'description': 'PMR >= 0.55',
                'status': 'N/A',
                'value': None
            }
        }

        # Use best available window
        metrics_to_check = metrics_15d or metrics_available

        if metrics_to_check:
            # IC median check
            ic_median = metrics_to_check['ic_median']
            hard_gates['ic_median_gate']['value'] = ic_median
            hard_gates['ic_median_gate']['status'] = 'PASS' if ic_median >= 0.02 else 'FAIL'

            # IR check (prefer 60d if available)
            ir_metrics = metrics_60d or metrics_to_check
            ir = ir_metrics['ir']
            hard_gates['ir_gate']['value'] = ir
            hard_gates['ir_gate']['status'] = 'PASS' if ir >= 0.5 else 'FAIL'

            # PMR check (prefer 30d if available)
            pmr_metrics = metrics_30d or metrics_to_check
            pmr = pmr_metrics['pmr']
            hard_gates['pmr_gate']['value'] = pmr
            hard_gates['pmr_gate']['status'] = 'PASS' if pmr >= 0.55 else 'FAIL'

        overall_status = 'PASS' if all(
            g['status'] == 'PASS' for g in hard_gates.values() if g['status'] != 'N/A'
        ) else 'FAIL'

        for gate_name, gate_info in hard_gates.items():
            status_symbol = '[PASS]' if gate_info['status'] == 'PASS' else '[FAIL]' if gate_info['status'] == 'FAIL' else '[N/A]'
            value_str = f"{gate_info['value']:.4f}" if gate_info['value'] is not None else "N/A"
            print(f"  {status_symbol} {gate_info['description']}: {value_str}")

        print(f"\n  Overall Hard gate status: {overall_status}")

        self.report['hard_gates'] = {
            'gates': hard_gates,
            'overall_status': overall_status,
            'metrics_15d': metrics_15d,
            'metrics_30d': metrics_30d,
            'metrics_60d': metrics_60d,
            'metrics_available': metrics_available
        }

        return overall_status

    def calculate_drawdown(self):
        """Calculate drawdown metrics"""
        print("\n[3/8] Calculating drawdown metrics...")

        if len(self.metrics_df) < 2:
            print("  Insufficient data for drawdown calculation")
            self.report['drawdown'] = {'status': 'INSUFFICIENT_DATA'}
            return

        # Calculate cumulative IC (proxy for cumulative returns)
        cumulative_ic = self.metrics_df['ic'].cumsum()

        # Calculate running maximum
        running_max = cumulative_ic.expanding().max()

        # Calculate drawdown
        drawdown = cumulative_ic - running_max

        # Calculate drawdown percentage (relative to running max)
        drawdown_pct = (drawdown / running_max.replace(0, np.nan)) * 100

        max_drawdown = drawdown.min()
        max_drawdown_pct = drawdown_pct.min() if not np.isnan(drawdown_pct.min()) else 0
        current_drawdown = drawdown.iloc[-1]
        current_drawdown_pct = drawdown_pct.iloc[-1] if not np.isnan(drawdown_pct.iloc[-1]) else 0

        # Find max drawdown date
        max_dd_idx = drawdown.idxmin()
        max_dd_date = self.metrics_df.loc[max_dd_idx, 'timestamp']

        print(f"  Max drawdown: {max_drawdown:.4f} ({max_drawdown_pct:.2f}%) on {max_dd_date}")
        print(f"  Current drawdown: {current_drawdown:.4f} ({current_drawdown_pct:.2f}%)")
        print(f"  Cumulative IC: {cumulative_ic.iloc[-1]:.4f}")

        # Check if in drawdown
        in_drawdown = current_drawdown < 0
        print(f"  Currently in drawdown: {'Yes' if in_drawdown else 'No'}")

        self.report['drawdown'] = {
            'max_drawdown': float(max_drawdown),
            'max_drawdown_pct': float(max_drawdown_pct),
            'max_drawdown_date': max_dd_date.isoformat(),
            'current_drawdown': float(current_drawdown),
            'current_drawdown_pct': float(current_drawdown_pct),
            'cumulative_ic': float(cumulative_ic.iloc[-1]),
            'in_drawdown': in_drawdown
        }

    def analyze_positions(self):
        """Analyze position statistics"""
        print("\n[4/8] Analyzing position statistics...")

        if self.positions_df is None or len(self.positions_df) == 0:
            print("  No position data available")
            self.report['positions'] = {'status': 'NO_DATA'}
            return

        positions = self.positions_df['position']
        base_weight = self.config['allocation']['initial_weight']
        max_weight = self.config['allocation']['max_weight']

        # Position statistics
        mean_position = positions.mean()
        std_position = positions.std()
        min_position = positions.min()
        max_position = positions.max()

        # Utilization (as % of max weight)
        utilization = positions.abs().mean() / max_weight * 100

        # Count extremes (at max weight)
        n_max_long = (positions >= max_weight * 0.99).sum()
        n_max_short = (positions <= -max_weight * 0.99).sum()
        pct_extremes = (n_max_long + n_max_short) / len(positions) * 100

        # Turnover (absolute position changes)
        if len(positions) > 1:
            position_changes = positions.diff().abs()
            avg_turnover = position_changes.mean()
            max_turnover = position_changes.max()
        else:
            avg_turnover = 0
            max_turnover = 0

        print(f"  Mean position: {mean_position:.4f} ({mean_position/max_weight*100:.1f}% of max)")
        print(f"  Position std: {std_position:.4f}")
        print(f"  Position range: [{min_position:.4f}, {max_position:.4f}]")
        print(f"  Utilization: {utilization:.1f}% of max weight")
        print(f"  Extreme positions: {n_max_long} max long, {n_max_short} max short ({pct_extremes:.1f}%)")
        print(f"  Avg turnover: {avg_turnover:.4f} per hour")

        self.report['positions'] = {
            'n_records': len(positions),
            'mean_position': float(mean_position),
            'std_position': float(std_position),
            'min_position': float(min_position),
            'max_position': float(max_position),
            'utilization_pct': float(utilization),
            'n_max_long': int(n_max_long),
            'n_max_short': int(n_max_short),
            'pct_extremes': float(pct_extremes),
            'avg_turnover': float(avg_turnover),
            'max_turnover': float(max_turnover)
        }

    def analyze_alerts(self):
        """Analyze alert history"""
        print("\n[5/8] Analyzing alert history...")

        if self.alerts_df is None or len(self.alerts_df) == 0:
            print("  No alerts recorded - System healthy")
            self.report['alerts'] = {
                'total_alerts': 0,
                'status': 'HEALTHY'
            }
            return

        total_alerts = len(self.alerts_df)

        # Count by level
        if 'level' in self.alerts_df.columns:
            alert_counts = self.alerts_df['level'].value_counts().to_dict()
        else:
            alert_counts = {}

        # Recent alerts (last 24 hours)
        recent_cutoff = self.timestamp - timedelta(hours=24)
        recent_alerts = self.alerts_df[self.alerts_df['timestamp'] >= recent_cutoff]

        print(f"  Total alerts: {total_alerts}")
        for level, count in alert_counts.items():
            print(f"    {level}: {count}")
        print(f"  Recent alerts (24h): {len(recent_alerts)}")

        self.report['alerts'] = {
            'total_alerts': total_alerts,
            'alert_counts': alert_counts,
            'recent_alerts_24h': len(recent_alerts),
            'status': 'WARNINGS_PRESENT' if total_alerts > 0 else 'HEALTHY'
        }

    def check_data_integrity(self):
        """Check data integrity (gaps, consistency)"""
        print("\n[6/8] Checking data integrity...")

        issues = []

        # Check for timestamp gaps in metrics
        if len(self.metrics_df) > 1:
            time_diffs = self.metrics_df['timestamp'].diff()
            expected_diff = timedelta(hours=1)

            # Allow some tolerance (e.g., 1.5 hours)
            gaps = time_diffs[time_diffs > timedelta(hours=1.5)]

            if len(gaps) > 0:
                issues.append(f"Found {len(gaps)} time gaps > 1.5 hours in metrics")
                print(f"  [WARNING] {issues[-1]}")
                for idx, gap in gaps.items():
                    gap_hours = gap.total_seconds() / 3600
                    timestamp = self.metrics_df.loc[idx, 'timestamp']
                    print(f"    Gap of {gap_hours:.1f} hours at {timestamp}")
            else:
                print("  [OK] No significant time gaps in metrics")

        # Check for duplicate timestamps
        duplicates = self.metrics_df['timestamp'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate timestamps")
            print(f"  [WARNING] {issues[-1]}")
        else:
            print("  [OK] No duplicate timestamps")

        # Check for missing values
        missing_ic = self.metrics_df['ic'].isna().sum()
        missing_position = self.metrics_df['position'].isna().sum()

        if missing_ic > 0 or missing_position > 0:
            issues.append(f"Missing values: IC={missing_ic}, position={missing_position}")
            print(f"  [WARNING] {issues[-1]}")
        else:
            print("  [OK] No missing critical values")

        # Check execution log success rate
        if self.execution_df is not None and len(self.execution_df) > 0:
            success_rate = (self.execution_df['status'] == 'SUCCESS').mean() * 100
            print(f"  [OK] Execution success rate: {success_rate:.1f}%")

            if success_rate < 95:
                issues.append(f"Low execution success rate: {success_rate:.1f}%")

        integrity_status = 'CLEAN' if len(issues) == 0 else 'ISSUES_FOUND'
        print(f"\n  Data integrity status: {integrity_status}")

        self.report['data_integrity'] = {
            'status': integrity_status,
            'issues': issues,
            'n_metrics_records': len(self.metrics_df),
            'n_position_records': len(self.positions_df) if self.positions_df is not None else 0
        }

    def generate_summary(self):
        """Generate overall summary"""
        print("\n[7/8] Generating performance summary...")

        # Overall health score
        health_score = 100

        # Deduct points for issues
        if self.report['hard_gates']['overall_status'] == 'FAIL':
            health_score -= 50
            print("  [-50] Hard gates not passing")

        if self.report.get('alerts', {}).get('total_alerts', 0) > 0:
            health_score -= 10
            print("  [-10] Alerts present")

        if self.report.get('data_integrity', {}).get('status') == 'ISSUES_FOUND':
            health_score -= 20
            print("  [-20] Data integrity issues")

        if self.report.get('drawdown', {}).get('in_drawdown', False):
            health_score -= 10
            print("  [-10] Currently in drawdown")

        # Determine overall status
        if health_score >= 90:
            overall_status = 'EXCELLENT'
        elif health_score >= 70:
            overall_status = 'GOOD'
        elif health_score >= 50:
            overall_status = 'ACCEPTABLE'
        else:
            overall_status = 'NEEDS_ATTENTION'

        print(f"\n  Health score: {health_score}/100")
        print(f"  Overall status: {overall_status}")

        self.report['summary'] = {
            'health_score': health_score,
            'overall_status': overall_status,
            'review_complete': True
        }

        self.report['status'] = overall_status

    def export_report(self):
        """Export report to JSON"""
        print("\n[8/8] Exporting audit report...")

        output_path = Path('warehouse/monitoring/day7_audit_report.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)

        print(f"  Report saved: {output_path}")

        return output_path

    def print_executive_summary(self):
        """Print executive summary for quick review"""
        print("\n" + "="*70)
        print(" "*20 + "DAY-7 PERFORMANCE REVIEW")
        print("="*70)
        print(f"Review Date: {self.report['review_date']}")
        print(f"Strategy: {self.config['strategy_id']}")
        print(f"Version: {self.config['version']}")
        print("="*70)

        print("\n[OVERALL STATUS]")
        print(f"  Health Score: {self.report['summary']['health_score']}/100")
        print(f"  Status: {self.report['summary']['overall_status']}")

        print("\n[HARD GATES]")
        print(f"  Overall: {self.report['hard_gates']['overall_status']}")
        for gate_name, gate_info in self.report['hard_gates']['gates'].items():
            status_symbol = '[PASS]' if gate_info['status'] == 'PASS' else '[FAIL]' if gate_info['status'] == 'FAIL' else '[N/A]'
            value_str = f"{gate_info['value']:.4f}" if gate_info['value'] is not None else "N/A"
            print(f"    {status_symbol} {gate_info['description']}: {value_str}")

        print("\n[DRAWDOWN]")
        if 'drawdown' in self.report and self.report['drawdown'].get('status') != 'INSUFFICIENT_DATA':
            dd = self.report['drawdown']
            print(f"  Max Drawdown: {dd['max_drawdown']:.4f} ({dd['max_drawdown_pct']:.2f}%)")
            print(f"  Current Drawdown: {dd['current_drawdown']:.4f} ({dd['current_drawdown_pct']:.2f}%)")
            print(f"  Cumulative IC: {dd['cumulative_ic']:.4f}")
        else:
            print("  Insufficient data")

        print("\n[POSITIONS]")
        if 'positions' in self.report and self.report['positions'].get('status') != 'NO_DATA':
            pos = self.report['positions']
            print(f"  Mean position: {pos['mean_position']:.4f}")
            print(f"  Utilization: {pos['utilization_pct']:.1f}% of max weight")
            print(f"  Extremes: {pos['pct_extremes']:.1f}% at limits")
        else:
            print("  No position data")

        print("\n[ALERTS]")
        alerts = self.report.get('alerts', {})
        print(f"  Total: {alerts.get('total_alerts', 0)}")
        print(f"  Recent (24h): {alerts.get('recent_alerts_24h', 0)}")

        print("\n[DATA INTEGRITY]")
        integrity = self.report.get('data_integrity', {})
        print(f"  Status: {integrity.get('status', 'UNKNOWN')}")
        if integrity.get('issues'):
            print("  Issues:")
            for issue in integrity['issues']:
                print(f"    - {issue}")

        print("\n" + "="*70)

        # Recommendation
        print("\n[RECOMMENDATION]")
        if self.report['summary']['overall_status'] in ['EXCELLENT', 'GOOD']:
            print("  Continue monitoring. Strategy performing as expected.")
        elif self.report['summary']['overall_status'] == 'ACCEPTABLE':
            print("  Continue with caution. Monitor closely for improvements.")
        else:
            print("  ATTENTION REQUIRED. Review hard gate failures and alerts.")

        print("\n" + "="*70)

    def run_full_review(self):
        """Run complete Day-7 review"""
        print("="*70)
        print(" "*15 + "DAY-7 PERFORMANCE AND RISK CONTROL REVIEW")
        print("="*70)
        print(f"Strategy: {self.config['strategy_id']}")
        print(f"Review Date: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        try:
            # Run all checks
            self.check_hard_gates()
            self.calculate_drawdown()
            self.analyze_positions()
            self.analyze_alerts()
            self.check_data_integrity()
            self.generate_summary()

            # Export report
            report_path = self.export_report()

            # Print summary
            self.print_executive_summary()

            print(f"\nFull audit report: {report_path}")
            print("\nNext steps:")
            print("  1. Review dashboard snapshot")
            print("  2. Update RUNLOG_OPERATIONS.md with audit results")
            print("  3. Schedule Day-14 review")

            return True

        except Exception as e:
            print(f"\n[ERROR] Review failed: {e}")
            import traceback
            traceback.print_exc()
            self.report['status'] = 'FAILED'
            self.report['error'] = str(e)
            return False


def main():
    """Run Day-7 performance review"""
    reviewer = Day7PerformanceReview()
    success = reviewer.run_full_review()

    if not success:
        sys.exit(1)

    return reviewer.report


if __name__ == '__main__':
    report = main()
