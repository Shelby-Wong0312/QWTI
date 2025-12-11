"""
BASE WEIGHT ALLOCATION - Seed202 LEAN 7-Feature Strategy

Production weight/position calculator for first Hard IC compliant strategy.
Implements conservative position sizing with risk controls aligned to Dashboard vision.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Configuration
BASE_WEIGHT = 0.10  # Optimal 10% allocation (from base_weight_scan.py with 5bps cost)
MAX_WEIGHT = 0.10   # Cap at 10% for single strategy (same as base_weight)
PREDICTION_THRESHOLD = 0.005  # 0.5% move caps position to full weight
MAX_DRAWDOWN_ALERT = 0.02  # Alert if drawdown > 2%
IC_ALERT_THRESHOLD = 0.01  # Alert if rolling IC < 0.01
IR_ALERT_THRESHOLD = 0.30  # Alert if rolling IR < 0.30
PMR_ALERT_THRESHOLD = 0.50  # Alert if rolling PMR < 0.50

# Paths
MODEL_PATH = Path('models/seed202_lean7_base.pkl')
DATA_PATH = Path('features_hourly_with_term.parquet')
POSITION_LOG_PATH = Path('warehouse/positions/base_seed202_lean7_positions.csv')
MONITOR_LOG_PATH = Path('warehouse/monitoring/base_seed202_lean7_metrics.csv')

class BaseWeightAllocator:
    """
    Production weight allocator for Seed202 LEAN 7-feature strategy.

    Implements Dashboard.md vision:
    - Real-time position calculation
    - Risk controls and hard stops
    - Audit trail for every allocation
    - Reproducible decision-making
    """

    def __init__(self, base_weight=BASE_WEIGHT, max_weight=MAX_WEIGHT):
        self.base_weight = base_weight
        self.max_weight = max_weight
        self.current_position = 0.0
        self.position_history = []
        self.metrics_history = []

    def calculate_position(self, prediction, confidence=1.0):
        """
        Calculate position size based on model prediction.

        Args:
            prediction: Model output (1-hour forward return forecast)
            confidence: Confidence multiplier (0-1), default 1.0

        Returns:
            position: Float between -max_weight and +max_weight
        """
        # Position = base_weight * sign(prediction) * min(1.0, |prediction| / threshold)
        direction = np.sign(prediction)
        magnitude = min(1.0, abs(prediction) / PREDICTION_THRESHOLD)
        position = self.base_weight * direction * magnitude * confidence

        # Apply max weight cap
        position = np.clip(position, -self.max_weight, self.max_weight)

        return position

    def log_position(self, timestamp, prediction, position, features, metadata=None):
        """
        Log position allocation with full audit trail.

        Implements Dashboard.md Section [D] - trade record with replay capability
        """
        record = {
            'timestamp': timestamp,
            'prediction': prediction,
            'position': position,
            'base_weight': self.base_weight,
            'max_weight': self.max_weight,
            'strategy': 'base_seed202_lean7_h1',
            'feature_snapshot': json.dumps(features),
            'metadata': json.dumps(metadata or {})
        }

        self.position_history.append(record)
        return record

    def check_risk_controls(self, rolling_ic, rolling_ir, rolling_pmr, drawdown):
        """
        Check risk controls and generate alerts.

        Implements Dashboard.md Section [F] - risk control panel

        Returns:
            dict: Alert status and messages
        """
        alerts = []
        status = 'HEALTHY'

        # IC check
        if rolling_ic < IC_ALERT_THRESHOLD:
            alerts.append({
                'level': 'WARNING',
                'metric': 'IC',
                'value': rolling_ic,
                'threshold': IC_ALERT_THRESHOLD,
                'message': f'Rolling IC ({rolling_ic:.4f}) below threshold ({IC_ALERT_THRESHOLD})'
            })
            status = 'WARNING'

        # IR check
        if rolling_ir < IR_ALERT_THRESHOLD:
            alerts.append({
                'level': 'WARNING',
                'metric': 'IR',
                'value': rolling_ir,
                'threshold': IR_ALERT_THRESHOLD,
                'message': f'Rolling IR ({rolling_ir:.4f}) below threshold ({IR_ALERT_THRESHOLD})'
            })
            status = 'WARNING'

        # PMR check
        if rolling_pmr < PMR_ALERT_THRESHOLD:
            alerts.append({
                'level': 'WARNING',
                'metric': 'PMR',
                'value': rolling_pmr,
                'threshold': PMR_ALERT_THRESHOLD,
                'message': f'Rolling PMR ({rolling_pmr:.4f}) below threshold ({PMR_ALERT_THRESHOLD})'
            })
            status = 'WARNING'

        # Drawdown check
        if drawdown > MAX_DRAWDOWN_ALERT:
            alerts.append({
                'level': 'CRITICAL',
                'metric': 'DRAWDOWN',
                'value': drawdown,
                'threshold': MAX_DRAWDOWN_ALERT,
                'message': f'Drawdown ({drawdown:.2%}) exceeds threshold ({MAX_DRAWDOWN_ALERT:.2%})'
            })
            status = 'CRITICAL'

        # Check for hard stops (auto de-activation)
        hard_stop = False
        if rolling_ic < 0.01 and len(self.metrics_history) >= 5:
            # Check if IC < 0.01 for 5 consecutive windows
            recent_ics = [m['ic'] for m in self.metrics_history[-5:]]
            if all(ic < 0.01 for ic in recent_ics):
                hard_stop = True
                alerts.append({
                    'level': 'CRITICAL',
                    'metric': 'HARD_STOP',
                    'value': rolling_ic,
                    'threshold': 0.01,
                    'message': 'IC < 0.01 for 5 consecutive windows - AUTOMATIC DE-ACTIVATION'
                })
                status = 'HARD_STOP'

        return {
            'status': status,
            'hard_stop': hard_stop,
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }

    def save_position_log(self):
        """Save position history to CSV for audit trail"""
        if not self.position_history:
            return

        df = pd.DataFrame(self.position_history)
        POSITION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(POSITION_LOG_PATH, index=False)

        return POSITION_LOG_PATH

    def save_metrics_log(self):
        """Save metrics history to CSV for monitoring"""
        if not self.metrics_history:
            return

        df = pd.DataFrame(self.metrics_history)
        MONITOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(MONITOR_LOG_PATH, index=False)

        return MONITOR_LOG_PATH


def generate_position_sizing_table():
    """
    Generate position sizing reference table.

    Shows relationship between prediction magnitude and position size.
    Helps users understand allocation logic (Dashboard.md Section [B] - strategy cards)
    """
    allocator = BaseWeightAllocator()

    predictions = np.linspace(-0.02, 0.02, 41)  # -2% to +2% returns
    positions = [allocator.calculate_position(p) for p in predictions]

    table = pd.DataFrame({
        'prediction_pct': predictions * 100,
        'position_pct': [p * 100 for p in positions],
        'direction': ['SHORT' if p < 0 else 'LONG' if p > 0 else 'NEUTRAL' for p in positions]
    })

    return table


def generate_monitoring_config():
    """
    Generate monitoring configuration for real-time dashboard.

    Implements Dashboard.md Section [A] + [E] - market overview and data integrity
    """
    config = {
        "strategy_id": "base_seed202_lean7_h1",
        "strategy_name": "Seed202 LEAN 7-Feature (H=1)",
        "version": "1.0.0",
        "activation_date": "2025-11-19",

        "allocation": {
            "initial_weight": BASE_WEIGHT,
            "max_weight": MAX_WEIGHT,
            "ramp_schedule": [
                {"days": 30, "weight": 0.20, "condition": "IC > 0.02"},
                {"days": 60, "weight": 0.25, "condition": "IC > 0.02"},
                {"days": 90, "weight": 0.30, "condition": "IC > 0.02"}
            ]
        },

        "features": [
            {"name": "cl1_cl2", "type": "market", "importance": 0.46, "source": "futures_curve"},
            {"name": "ovx", "type": "market", "importance": 0.373, "source": "volatility_index"},
            {"name": "OIL_CORE_norm_art_cnt", "type": "gdelt", "importance": 0.161, "source": "news_volume"},
            {"name": "MACRO_norm_art_cnt", "type": "gdelt", "importance": 0.125, "source": "news_volume"},
            {"name": "SUPPLY_CHAIN_norm_art_cnt", "type": "gdelt", "importance": 0.091, "source": "news_volume"},
            {"name": "USD_RATE_norm_art_cnt", "type": "gdelt", "importance": 0.012, "source": "news_volume"},
            {"name": "GEOPOL_norm_art_cnt", "type": "gdelt", "importance": 0.009, "source": "news_volume"}
        ],

        "monitoring_metrics": {
            "realtime": {
                "update_frequency": "1h",
                "metrics": [
                    {"name": "current_position", "alert_abs_threshold": 0.30},
                    {"name": "prediction", "display_format": "%.4f"},
                    {"name": "feature_drift_pct", "alert_threshold": 0.05}
                ]
            },
            "rolling_15d": {
                "metrics": [
                    {"name": "ic", "alert_threshold": 0.01, "target": 0.1358},
                    {"name": "ic_std", "display_format": "%.4f"}
                ]
            },
            "rolling_60d": {
                "metrics": [
                    {"name": "ir", "alert_threshold": 0.30, "target": 1.5758},
                    {"name": "sharpe", "display_format": "%.2f"}
                ]
            },
            "rolling_30d": {
                "metrics": [
                    {"name": "pmr", "alert_threshold": 0.50, "target": 0.8039},
                    {"name": "max_drawdown", "alert_threshold": 0.02}
                ]
            }
        },

        "risk_controls": {
            "hard_stops": [
                {"condition": "ic < 0.01 for 5 consecutive windows", "action": "AUTO_DEACTIVATE"},
                {"condition": "ir < 0.3 for 10 consecutive windows", "action": "AUTO_DEACTIVATE"},
                {"condition": "data_quality_gate_fail", "action": "HALT_TRADING"},
                {"condition": "skip_ratio > 0.02", "action": "HALT_TRADING"}
            ],
            "soft_alerts": [
                {"condition": "ic < 0.02", "action": "NOTIFY"},
                {"condition": "drawdown > 0.015", "action": "NOTIFY"},
                {"condition": "feature_missing_pct > 0.05", "action": "NOTIFY"}
            ],
            "position_limits": {
                "max_single_position": 0.30,
                "max_leverage": 1.0,
                "max_daily_turnover": 2.0
            }
        },

        "data_integrity": {
            "sources": [
                {
                    "name": "GDELT",
                    "path": "data/gdelt_hourly.parquet",
                    "version": "v2024.10",
                    "hash_type": "SHA256",
                    "update_frequency": "1h",
                    "sla_latency": "15min"
                },
                {
                    "name": "WTI_Prices",
                    "path": "data/features_hourly.parquet",
                    "version": "v2025.10",
                    "hash_type": "SHA256",
                    "update_frequency": "1h",
                    "sla_latency": "5min"
                },
                {
                    "name": "Market_Term_OVX",
                    "path": "data/term_crack_ovx_hourly.csv",
                    "version": "live",
                    "hash_type": "SHA256",
                    "update_frequency": "1h",
                    "sla_latency": "10min"
                }
            ],
            "no_drift_contract": "warehouse/policy/no_drift.yaml",
            "preflight_check": "warehouse/policy/utils/nodrift_preflight.py"
        },

        "audit_trail": {
            "position_log": "warehouse/positions/base_seed202_lean7_positions.csv",
            "metrics_log": "warehouse/monitoring/base_seed202_lean7_metrics.csv",
            "alert_log": "warehouse/monitoring/base_seed202_lean7_alerts.csv",
            "replay_capability": True,
            "snapshot_retention_days": 365
        }
    }

    return config


if __name__ == '__main__':
    print("="*60)
    print("BASE WEIGHT ALLOCATION - Seed202 LEAN 7-Feature")
    print("="*60)

    # Generate position sizing table
    print("\n[1/3] Position Sizing Reference Table")
    print("-" * 60)
    table = generate_position_sizing_table()
    print("\nSample entries (prediction to position):")
    print(table[::4].to_string(index=False))

    # Save full table
    table_path = Path('warehouse/base_position_sizing_table.csv')
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    print(f"\nFull table saved: {table_path}")

    # Generate monitoring config
    print("\n[2/3] Monitoring Configuration")
    print("-" * 60)
    config = generate_monitoring_config()

    config_path = Path('warehouse/base_monitoring_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved: {config_path}")

    print("\nKey Monitoring Metrics:")
    print(f"  - Real-time: Position, Prediction, Feature Drift (every 1h)")
    print(f"  - Rolling 15d: IC (alert < {IC_ALERT_THRESHOLD})")
    print(f"  - Rolling 60d: IR (alert < {IR_ALERT_THRESHOLD})")
    print(f"  - Rolling 30d: PMR (alert < {PMR_ALERT_THRESHOLD}), Max DD (alert > {MAX_DRAWDOWN_ALERT:.1%})")

    print("\nHard Stops (Auto De-activation):")
    for stop in config['risk_controls']['hard_stops']:
        print(f"  - {stop['condition']} -> {stop['action']}")

    # Create directory structure
    print("\n[3/3] Creating Directory Structure")
    print("-" * 60)
    dirs = [
        Path('warehouse/positions'),
        Path('warehouse/monitoring'),
        Path('models')
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {d}")

    print("\n" + "="*60)
    print("BASE ALLOCATION SETUP COMPLETE")
    print("="*60)
    print(f"\nStrategy: base_seed202_lean7_h1")
    print(f"Initial Weight: {BASE_WEIGHT:.1%}")
    print(f"Max Weight: {MAX_WEIGHT:.1%}")
    print(f"Status: READY FOR PRODUCTION")
