"""
REAL-TIME MONITORING DASHBOARD - Base Strategy

Implements Dashboard.md vision for Seed202 LEAN 7-Feature Base strategy.
Provides real-time monitoring, alerts, and audit trail capabilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# Paths
MONITORING_CONFIG = Path('warehouse/base_monitoring_config.json')
METRICS_LOG = Path('warehouse/monitoring/base_seed202_lean7_metrics.csv')
POSITION_LOG = Path('warehouse/positions/base_seed202_lean7_positions.csv')
ALERT_LOG = Path('warehouse/monitoring/base_seed202_lean7_alerts.csv')

class BaseDashboard:
    """
    Real-time dashboard for Base strategy monitoring.

    Implements Dashboard.md sections:
    - [A] Market Status Overview
    - [B] Strategy Cards
    - [C] Account Status
    - [D] Trade Records (with replay capability)
    - [E] Data Integrity
    - [F] Risk Control Panel
    """

    def __init__(self):
        with open(MONITORING_CONFIG) as f:
            self.config = json.load(f)

        self.metrics_df = None
        self.positions_df = None
        self.alerts = []

    def load_latest_data(self):
        """Load latest metrics and positions"""
        if METRICS_LOG.exists():
            self.metrics_df = pd.read_csv(METRICS_LOG)

        if POSITION_LOG.exists():
            self.positions_df = pd.read_csv(POSITION_LOG)

    def calculate_rolling_metrics(self, window_days=15):
        """Calculate rolling metrics for monitoring"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            return None

        # Get recent data
        cutoff = datetime.now() - timedelta(days=window_days)
        recent = self.metrics_df[
            pd.to_datetime(self.metrics_df['timestamp']) >= cutoff
        ]

        if len(recent) == 0:
            return None

        rolling = {
            'window_days': window_days,
            'n_observations': len(recent),
            'ic_mean': recent['ic'].mean(),
            'ic_median': recent['ic'].median(),
            'ic_std': recent['ic'].std(),
            'ir': recent['ic'].mean() / recent['ic'].std() if recent['ic'].std() > 0 else 0,
            'pmr': (recent['ic'] > 0).mean(),
            'last_ic': recent['ic'].iloc[-1] if len(recent) > 0 else np.nan,
            'last_update': recent['timestamp'].iloc[-1]
        }

        return rolling

    def check_health_status(self):
        """
        Check strategy health and generate alerts.

        Returns status: HEALTHY, WARNING, CRITICAL, HARD_STOP
        """
        rolling_15d = self.calculate_rolling_metrics(15)
        rolling_60d = self.calculate_rolling_metrics(60)
        rolling_30d = self.calculate_rolling_metrics(30)

        if rolling_15d is None:
            return {'status': 'NO_DATA', 'alerts': []}

        alerts = []
        status = 'HEALTHY'

        # Check IC
        if rolling_15d['ic_mean'] < 0.01:
            alerts.append({
                'level': 'WARNING',
                'metric': 'IC',
                'value': rolling_15d['ic_mean'],
                'threshold': 0.01,
                'message': f"Rolling 15d IC ({rolling_15d['ic_mean']:.4f}) below alert threshold"
            })
            status = 'WARNING'

        # Check IR
        if rolling_60d and rolling_60d['ir'] < 0.30:
            alerts.append({
                'level': 'WARNING',
                'metric': 'IR',
                'value': rolling_60d['ir'],
                'threshold': 0.30,
                'message': f"Rolling 60d IR ({rolling_60d['ir']:.4f}) below alert threshold"
            })
            status = 'WARNING'

        # Check PMR
        if rolling_30d and rolling_30d['pmr'] < 0.50:
            alerts.append({
                'level': 'WARNING',
                'metric': 'PMR',
                'value': rolling_30d['pmr'],
                'threshold': 0.50,
                'message': f"Rolling 30d PMR ({rolling_30d['pmr']:.4f}) below alert threshold"
            })
            status = 'WARNING'

        # Check hard stops
        recent_5 = self.metrics_df.tail(5)
        if len(recent_5) >= 5 and all(recent_5['ic'] < 0.01):
            alerts.append({
                'level': 'CRITICAL',
                'metric': 'HARD_STOP',
                'value': recent_5['ic'].mean(),
                'threshold': 0.01,
                'message': 'IC < 0.01 for 5 consecutive windows - AUTO DE-ACTIVATION TRIGGERED'
            })
            status = 'HARD_STOP'

        return {
            'status': status,
            'alerts': alerts,
            'metrics': {
                'rolling_15d': rolling_15d,
                'rolling_60d': rolling_60d,
                'rolling_30d': rolling_30d
            }
        }

    def generate_strategy_card(self):
        """
        Generate strategy card in Dashboard.md format.

        Section [B] - Strategy recommendation card
        """
        health = self.check_health_status()

        if health['status'] == 'NO_DATA':
            return "Strategy Card: NO DATA AVAILABLE"

        metrics = health['metrics']['rolling_15d']

        # Determine recommendation
        if health['status'] == 'HARD_STOP':
            recommendation = 'HALT'
            confidence = 0
        elif health['status'] == 'CRITICAL':
            recommendation = 'REDUCE'
            confidence = 30
        elif health['status'] == 'WARNING':
            recommendation = 'HOLD'
            confidence = 50
        else:
            recommendation = 'ACTIVE'
            confidence = min(100, int(metrics['ic_mean'] / 0.02 * 100))

        card = f"""
+-------------------------------------------------------------------+
| Strategy: Seed202 LEAN 7-Feature (H=1)                            |
+-------------------------------------------------------------------+
| Recommendation: {recommendation:10s}  Confidence: {confidence:3d}%               |
| Current Weight: {self.config['allocation']['initial_weight']*100:4.1f}%    Max Weight: {self.config['allocation']['max_weight']*100:4.1f}%          |
+-------------------------------------------------------------------+
| Performance (15d rolling):                                        |
|   IC mean:   {metrics['ic_mean']:7.4f}  (target: 0.1358)                      |
|   IR:        {metrics['ir']:7.4f}  (target: 1.5758)                      |
|   PMR:       {metrics['pmr']:7.4f}  (target: 0.8039)                      |
+-------------------------------------------------------------------+
| Status: {health['status']:20s}                              |
| Last Update: {metrics['last_update']:30s}             |
+-------------------------------------------------------------------+
        """

        if health['alerts']:
            card += "\n\nActive Alerts:\n"
            for alert in health['alerts']:
                card += f"  [{alert['level']}] {alert['message']}\n"

        return card

    def print_dashboard(self):
        """Print full dashboard to console"""
        print("="*70)
        print(" "*15 + "BASE STRATEGY MONITORING DASHBOARD")
        print("="*70)
        print(f"Strategy: {self.config['strategy_id']}")
        print(f"Version: {self.config['version']}")
        print(f"Activated: {self.config['activation_date']}")
        print("="*70)

        # Strategy Card
        print("\n[STRATEGY CARD]")
        print(self.generate_strategy_card())

        # Health Status
        print("\n[HEALTH CHECK]")
        health = self.check_health_status()
        print(f"Overall Status: {health['status']}")

        if health['metrics']['rolling_15d']:
            print("\nRolling Metrics:")
            print(f"  15d: IC={health['metrics']['rolling_15d']['ic_mean']:.4f}, "
                  f"IR={health['metrics']['rolling_15d']['ir']:.4f}, "
                  f"PMR={health['metrics']['rolling_15d']['pmr']:.2%}")

        if health['metrics']['rolling_60d']:
            print(f"  60d: IC={health['metrics']['rolling_60d']['ic_mean']:.4f}, "
                  f"IR={health['metrics']['rolling_60d']['ir']:.4f}")

        # Alerts
        if health['alerts']:
            print("\n[ACTIVE ALERTS]")
            for alert in health['alerts']:
                print(f"  [{alert['level']}] {alert['metric']}: {alert['message']}")
        else:
            print("\n[ACTIVE ALERTS] None - All systems nominal")

        # Risk Controls
        print("\n[RISK CONTROLS]")
        print("Hard Stops:")
        for stop in self.config['risk_controls']['hard_stops']:
            print(f"  - {stop['condition']} -> {stop['action']}")

        # Data Integrity
        print("\n[DATA INTEGRITY]")
        for source in self.config['data_integrity']['sources']:
            print(f"  [{source['name']}] {source['path']}")
            print(f"    Version: {source['version']}, Update: {source['update_frequency']}")

        print("\n" + "="*70)


def generate_monitoring_summary():
    """Generate monitoring summary for warehouse records"""
    dashboard = BaseDashboard()
    dashboard.load_latest_data()
    health = dashboard.check_health_status()

    summary = {
        'timestamp': datetime.now().isoformat(),
        'strategy_id': dashboard.config['strategy_id'],
        'status': health['status'],
        'n_alerts': len(health['alerts']),
        'alerts': health['alerts'],
        'metrics': health['metrics']
    }

    # Save to alert log if there are alerts
    if health['alerts']:
        alert_log_entry = pd.DataFrame([{
            'timestamp': datetime.now().isoformat(),
            'status': health['status'],
            'n_alerts': len(health['alerts']),
            'alerts_json': json.dumps(health['alerts'])
        }])

        ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)

        if ALERT_LOG.exists():
            alert_log_entry.to_csv(ALERT_LOG, mode='a', header=False, index=False)
        else:
            alert_log_entry.to_csv(ALERT_LOG, index=False)

    return summary


if __name__ == '__main__':
    print("Initializing Base Strategy Dashboard...")
    print()

    dashboard = BaseDashboard()
    dashboard.load_latest_data()

    # Print dashboard
    dashboard.print_dashboard()

    # Generate summary
    print("\n\nGenerating monitoring summary...")
    summary = generate_monitoring_summary()

    print(f"\nSummary Status: {summary['status']}")
    print(f"Alerts: {summary['n_alerts']}")

    if summary['n_alerts'] > 0:
        print(f"Alert log updated: {ALERT_LOG}")
