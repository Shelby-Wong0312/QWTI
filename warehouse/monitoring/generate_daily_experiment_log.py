"""
DAILY EXPERIMENT LOG GENERATOR

Reads hourly monitoring data and generates a daily markdown report.

Sources:
- hourly_runlog.jsonl
- base_seed202_lean7_metrics.csv
- base_seed202_lean7_alerts.csv
- base_seed202_lean7_positions.csv

Output:
- warehouse/monitoring/daily_experiment_log/YYYY-MM-DD.md
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
import argparse

# Paths
MONITORING_DIR = Path('warehouse/monitoring')
POSITIONS_DIR = Path('warehouse/positions')
RUNLOG_PATH = MONITORING_DIR / 'hourly_runlog.jsonl'
METRICS_PATH = MONITORING_DIR / 'base_seed202_lean7_metrics.csv'
ALERTS_PATH = MONITORING_DIR / 'base_seed202_lean7_alerts.csv'
POSITIONS_PATH = POSITIONS_DIR / 'base_seed202_lean7_positions.csv'
OUTPUT_DIR = MONITORING_DIR / 'daily_experiment_log'


def load_runlog_for_date(target_date: date) -> List[Dict[str, Any]]:
    """Load runlog entries for a specific date"""
    if not RUNLOG_PATH.exists():
        return []

    records = []
    with open(RUNLOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                ts_run = record.get('ts_run', '')
                if ts_run:
                    record_date = datetime.fromisoformat(ts_run.replace('Z', '+00:00')).date()
                    if record_date == target_date:
                        records.append(record)
            except (json.JSONDecodeError, ValueError):
                continue
    return records


def load_csv_for_date(csv_path: Path, target_date: date, ts_col: str = 'timestamp') -> pd.DataFrame:
    """Load CSV entries for a specific date"""
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if ts_col not in df.columns:
        return pd.DataFrame()

    df[ts_col] = pd.to_datetime(df[ts_col])
    df['date'] = df[ts_col].dt.date
    return df[df['date'] == target_date].copy()


def compute_execution_stats(runlog: List[Dict]) -> Dict[str, Any]:
    """Compute execution statistics from runlog"""
    if not runlog:
        return {
            'total_runs': 0,
            'success_count': 0,
            'failed_count': 0,
            'success_rate': 0.0,
            'error_types': {}
        }

    total = len(runlog)
    success = sum(1 for r in runlog if r.get('status') == 'SUCCESS')
    failed = total - success

    error_types = {}
    for r in runlog:
        if r.get('status') == 'FAILED':
            err_type = r.get('error_type', 'Unknown')
            error_types[err_type] = error_types.get(err_type, 0) + 1

    return {
        'total_runs': total,
        'success_count': success,
        'failed_count': failed,
        'success_rate': success / total if total > 0 else 0.0,
        'error_types': error_types
    }


def compute_metric_stats(runlog: List[Dict]) -> Dict[str, Any]:
    """Compute IC/IR/PMR statistics from runlog"""
    ic_15d_values = [r['ic_15d'] for r in runlog if r.get('ic_15d') is not None]
    ir_15d_values = [r['ir_15d'] for r in runlog if r.get('ir_15d') is not None]
    pmr_15d_values = [r['pmr_15d'] for r in runlog if r.get('pmr_15d') is not None]
    ic_60d_values = [r['ic_60d'] for r in runlog if r.get('ic_60d') is not None]
    ir_60d_values = [r['ir_60d'] for r in runlog if r.get('ir_60d') is not None]

    def stats(values):
        if not values:
            return {'min': None, 'max': None, 'mean': None, 'std': None}
        return {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)) if len(values) > 1 else 0.0
        }

    return {
        'ic_15d': stats(ic_15d_values),
        'ir_15d': stats(ir_15d_values),
        'pmr_15d': stats(pmr_15d_values),
        'ic_60d': stats(ic_60d_values),
        'ir_60d': stats(ir_60d_values)
    }


def compute_position_stats(runlog: List[Dict]) -> Dict[str, Any]:
    """Compute position statistics from runlog"""
    positions = [r['position'] for r in runlog if r.get('position') is not None]
    predictions = [r['prediction'] for r in runlog if r.get('prediction') is not None]

    if not positions:
        return {
            'position_min': None,
            'position_max': None,
            'position_mean': None,
            'position_extreme': None,
            'prediction_min': None,
            'prediction_max': None
        }

    pos_abs = [abs(p) for p in positions]
    extreme_idx = pos_abs.index(max(pos_abs))

    return {
        'position_min': float(min(positions)),
        'position_max': float(max(positions)),
        'position_mean': float(np.mean(positions)),
        'position_extreme': float(positions[extreme_idx]),
        'prediction_min': float(min(predictions)) if predictions else None,
        'prediction_max': float(max(predictions)) if predictions else None
    }


def compute_alert_stats(runlog: List[Dict]) -> Dict[str, Any]:
    """Compute alert statistics from runlog"""
    all_alerts = []
    for r in runlog:
        alerts = r.get('alerts', [])
        if isinstance(alerts, list):
            all_alerts.extend(alerts)

    level_counts = {}
    gate_counts = {}

    for alert in all_alerts:
        level = alert.get('level', 'UNKNOWN')
        gate = alert.get('gate', 'UNKNOWN')
        level_counts[level] = level_counts.get(level, 0) + 1
        gate_counts[gate] = gate_counts.get(gate, 0) + 1

    hard_gate_statuses = [r.get('hard_gate_status') for r in runlog if r.get('hard_gate_status')]

    return {
        'total_alerts': len(all_alerts),
        'level_counts': level_counts,
        'gate_counts': gate_counts,
        'hard_gate_statuses': list(set(hard_gate_statuses)),
        'had_critical': 'CRITICAL' in level_counts,
        'had_hard_stop': 'HARD_STOP' in gate_counts
    }


def compute_data_integrity(runlog: List[Dict]) -> Dict[str, Any]:
    """Compute data integrity stats from runlog"""
    hosts = list(set(r.get('source_host') for r in runlog if r.get('source_host')))
    python_versions = list(set(r.get('python_version') for r in runlog if r.get('python_version')))
    pandas_versions = list(set(r.get('pandas_version') for r in runlog if r.get('pandas_version')))
    features_versions = list(set(r.get('features_version') for r in runlog if r.get('features_version')))

    data_ts_values = [r.get('data_ts') for r in runlog if r.get('data_ts')]

    return {
        'source_hosts': hosts,
        'python_versions': python_versions,
        'pandas_versions': pandas_versions,
        'features_versions': features_versions,
        'data_timestamps': data_ts_values[:5] if len(data_ts_values) > 5 else data_ts_values,
        'n_unique_data_ts': len(set(data_ts_values))
    }


def format_stat(value, fmt='.4f'):
    """Format a stat value for display"""
    if value is None:
        return 'N/A'
    return f'{value:{fmt}}'


def generate_markdown_report(target_date: date, runlog: List[Dict]) -> str:
    """Generate the full markdown report"""
    exec_stats = compute_execution_stats(runlog)
    metric_stats = compute_metric_stats(runlog)
    pos_stats = compute_position_stats(runlog)
    alert_stats = compute_alert_stats(runlog)
    data_stats = compute_data_integrity(runlog)

    # Build markdown
    lines = []
    lines.append(f"# Daily Experiment Log: {target_date.isoformat()}")
    lines.append("")
    lines.append(f"> Auto-generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Execution Summary
    lines.append("## 1. Execution Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Runs | {exec_stats['total_runs']} |")
    lines.append(f"| Success | {exec_stats['success_count']} |")
    lines.append(f"| Failed | {exec_stats['failed_count']} |")
    lines.append(f"| Success Rate | {exec_stats['success_rate']:.1%} |")
    lines.append("")

    if exec_stats['error_types']:
        lines.append("### Error Types")
        lines.append("")
        for err_type, count in exec_stats['error_types'].items():
            lines.append(f"- `{err_type}`: {count} occurrence(s)")
        lines.append("")

    # IC/IR/PMR Distribution
    lines.append("## 2. IC/IR/PMR Distribution")
    lines.append("")
    lines.append("### Rolling 15-Day Metrics")
    lines.append("")
    lines.append("| Metric | Min | Max | Mean | Std |")
    lines.append("|--------|-----|-----|------|-----|")
    for metric_name in ['ic_15d', 'ir_15d', 'pmr_15d']:
        s = metric_stats[metric_name]
        lines.append(f"| {metric_name.upper()} | {format_stat(s['min'])} | {format_stat(s['max'])} | {format_stat(s['mean'])} | {format_stat(s['std'])} |")
    lines.append("")

    lines.append("### Rolling 60-Day Metrics")
    lines.append("")
    lines.append("| Metric | Min | Max | Mean | Std |")
    lines.append("|--------|-----|-----|------|-----|")
    for metric_name in ['ic_60d', 'ir_60d']:
        s = metric_stats[metric_name]
        lines.append(f"| {metric_name.upper()} | {format_stat(s['min'])} | {format_stat(s['max'])} | {format_stat(s['mean'])} | {format_stat(s['std'])} |")
    lines.append("")

    # Position Summary
    lines.append("## 3. Position Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Min Position | {format_stat(pos_stats['position_min'], '.2%') if pos_stats['position_min'] is not None else 'N/A'} |")
    lines.append(f"| Max Position | {format_stat(pos_stats['position_max'], '.2%') if pos_stats['position_max'] is not None else 'N/A'} |")
    lines.append(f"| Mean Position | {format_stat(pos_stats['position_mean'], '.2%') if pos_stats['position_mean'] is not None else 'N/A'} |")
    lines.append(f"| Most Extreme | {format_stat(pos_stats['position_extreme'], '.2%') if pos_stats['position_extreme'] is not None else 'N/A'} |")
    lines.append(f"| Prediction Range | [{format_stat(pos_stats['prediction_min'])}  ~  {format_stat(pos_stats['prediction_max'])}] |")
    lines.append("")

    # Alert Summary
    lines.append("## 4. Alert Summary")
    lines.append("")
    lines.append(f"**Total Alerts**: {alert_stats['total_alerts']}")
    lines.append("")

    if alert_stats['level_counts']:
        lines.append("### By Level")
        lines.append("")
        for level, count in sorted(alert_stats['level_counts'].items()):
            emoji = {'CRITICAL': '🔴', 'WARNING': '🟡', 'INFO': '🟢'}.get(level, '⚪')
            lines.append(f"- {emoji} **{level}**: {count}")
        lines.append("")

    if alert_stats['gate_counts']:
        lines.append("### By Gate")
        lines.append("")
        for gate, count in sorted(alert_stats['gate_counts'].items()):
            lines.append(f"- `{gate}`: {count}")
        lines.append("")

    lines.append(f"**Hard Gate Statuses Observed**: {', '.join(alert_stats['hard_gate_statuses']) if alert_stats['hard_gate_statuses'] else 'None'}")
    lines.append("")

    if alert_stats['had_critical']:
        lines.append("> ⚠️ **CRITICAL alerts occurred today - review required**")
        lines.append("")

    if alert_stats['had_hard_stop']:
        lines.append("> 🛑 **HARD_STOP triggered - strategy may have been auto-deactivated**")
        lines.append("")

    # Data Integrity
    lines.append("## 5. Data Integrity Check")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
    lines.append(f"| Source Host(s) | {', '.join(data_stats['source_hosts']) if data_stats['source_hosts'] else 'N/A'} |")
    lines.append(f"| Python Version(s) | {', '.join(data_stats['python_versions']) if data_stats['python_versions'] else 'N/A'} |")
    lines.append(f"| Pandas Version(s) | {', '.join(data_stats['pandas_versions']) if data_stats['pandas_versions'] else 'N/A'} |")
    lines.append(f"| Features File(s) | {', '.join(data_stats['features_versions']) if data_stats['features_versions'] else 'N/A'} |")
    lines.append(f"| Unique Data Timestamps | {data_stats['n_unique_data_ts']} |")
    lines.append("")

    # Observations section (for human input)
    lines.append("## 6. Today's Observations / TODO")
    lines.append("")
    lines.append("<!-- Manual section - fill in your observations below -->")
    lines.append("")
    lines.append("### Observations")
    lines.append("")
    lines.append("- [ ] (填寫今日觀察)")
    lines.append("")
    lines.append("### Action Items")
    lines.append("")
    lines.append("- [ ] (填寫待辦事項)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated from {len(runlog)} runlog entries*")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate daily experiment log")
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Target date in YYYY-MM-DD format (default: today)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(OUTPUT_DIR),
        help='Output directory for daily logs'
    )
    args = parser.parse_args()

    # Parse target date
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        target_date = date.today()

    print(f"Generating daily experiment log for {target_date.isoformat()}...")

    # Load runlog data
    runlog = load_runlog_for_date(target_date)
    print(f"  Found {len(runlog)} runlog entries")

    # Generate report
    report = generate_markdown_report(target_date, runlog)

    # Write output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{target_date.isoformat()}.md"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  Report written to: {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()
