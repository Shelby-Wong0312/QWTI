#!/usr/bin/env python3
"""
24-Hour Dry Run of Hourly Monitor

Simulates 24 hours of streaming conditions:
1. Iterates through recent 24 hours of data
2. Runs predictions using the HARD model
3. Calculates rolling IC/IR/PMR
4. Verifies thresholds stay above Hard gates
5. Logs results to warehouse/monitoring/

This validates the production pipeline before full deployment.
"""
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')

# Paths
FEATURES_PATH = Path('features_hourly_with_clbz.parquet')
MODEL_PATH = Path('models/base_seed202_clbz_h1.pkl')
CONFIG_PATH = Path('warehouse/base_monitoring_config.json')
DRYRUN_LOG = Path('warehouse/monitoring/dryrun_24h_log.csv')
DRYRUN_SUMMARY = Path('warehouse/monitoring/dryrun_24h_summary.json')

# Feature columns
FEATURE_COLS = [
    'cl_bz_spread',
    'ovx',
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt',
    'momentum_24h',
]

# Thresholds
IC_THRESHOLD = 0.02
IR_THRESHOLD = 0.5
PMR_THRESHOLD = 0.55

print("="*80)
print("24-HOUR DRY RUN - HARD Model Validation")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Features: {FEATURES_PATH}")
print(f"Model: {MODEL_PATH}")

# ============================================================================
# Load Model and Data
# ============================================================================
print("\n[1/5] Loading model and data...")

# Load config
with open(CONFIG_PATH) as f:
    config = json.load(f)

print(f"   Strategy: {config['strategy_name']}")
print(f"   Version: {config['version']}")

# Load model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
print(f"   Model loaded: {MODEL_PATH.name}")

# Load features
df = pd.read_parquet(FEATURES_PATH)
print(f"   Data loaded: {len(df)} rows")
print(f"   Date range: {df.index.min()} to {df.index.max()}")

# Filter to recent data for streaming simulation
# Use last 360 hours (15 days) for rolling window + 24 hours for test
test_window = 24  # hours to simulate
lookback = 360  # hours for IC/IR calculation

end_ts = df.index.max()
start_ts = end_ts - timedelta(hours=lookback + test_window)
df_sim = df[df.index >= start_ts].copy()

print(f"   Simulation period: {df_sim.index.min()} to {df_sim.index.max()}")
print(f"   Simulation rows: {len(df_sim)}")

# ============================================================================
# Run Streaming Simulation
# ============================================================================
print("\n[2/5] Running streaming simulation (24 hours)...")

# Prepare for streaming
df_sim['target'] = df_sim['wti_returns'].shift(-1)
df_sim = df_sim.dropna(subset=['target'])

# Get timestamps for last 24 hours
test_start_idx = len(df_sim) - test_window
test_timestamps = df_sim.index[test_start_idx:]

results = []
predictions = []
actuals = []

print(f"\n   Simulating {len(test_timestamps)} hourly cycles...")

for i, ts in enumerate(test_timestamps):
    # Get features at this timestamp
    row = df_sim.loc[ts]
    X = pd.DataFrame([row[FEATURE_COLS]])[FEATURE_COLS]

    # Make prediction
    pred = float(model.predict(X)[0])
    actual = row['target']

    predictions.append(pred)
    actuals.append(actual)

    # Calculate position (same logic as allocator)
    base_weight = config['allocation']['base_weight']
    threshold = config['allocation']['threshold']

    if abs(pred) < threshold:
        position = 0
    else:
        direction = np.sign(pred)
        magnitude = min(1.0, abs(pred) / threshold)
        position = base_weight * direction * magnitude

    results.append({
        'timestamp': ts,
        'prediction': pred,
        'actual': actual,
        'position': position,
        'hour_in_sim': i + 1
    })

    # Progress update every 6 hours
    if (i + 1) % 6 == 0:
        print(f"   Hour {i+1:2d}/{len(test_timestamps)}: pred={pred:+.6f}, actual={actual:+.6f}, pos={position:+.2%}")

print(f"\n   Completed {len(results)} hourly cycles")

# ============================================================================
# Calculate Rolling Metrics
# ============================================================================
print("\n[3/5] Calculating rolling metrics...")

# Calculate IC for each prediction-actual pair
ics = []
for pred, actual in zip(predictions, actuals):
    # Single point IC approximation (direction agreement)
    ic_point = 1 if (pred > 0 and actual > 0) or (pred < 0 and actual < 0) else -1
    ics.append(ic_point)

# Use full rolling window for proper IC calculation
# Get all historical predictions for the lookback period
full_hist = df_sim.iloc[:test_start_idx]
hist_preds = []
hist_actuals = []

for idx in full_hist.index:
    row = full_hist.loc[idx]
    X = pd.DataFrame([row[FEATURE_COLS]])[FEATURE_COLS]
    pred = float(model.predict(X)[0])
    hist_preds.append(pred)
    hist_actuals.append(row['target'])

# Combine historical + test for rolling calculation
all_preds = hist_preds + predictions
all_actuals = hist_actuals + actuals

# Calculate correlation-based IC over windows
def rolling_ic(preds, actuals, window=360):
    """Calculate rolling IC with Pearson correlation"""
    if len(preds) < window:
        return np.nan
    recent_preds = preds[-window:]
    recent_actuals = actuals[-window:]
    if np.std(recent_preds) > 0 and np.std(recent_actuals) > 0:
        return np.corrcoef(recent_preds, recent_actuals)[0, 1]
    return 0

# Calculate metrics at end of simulation
final_ic = rolling_ic(all_preds, all_actuals, window=360)
final_ic_std = np.std([rolling_ic(all_preds[:i], all_actuals[:i], 360)
                       for i in range(360, len(all_preds), 24) if i <= len(all_preds)])
final_ir = final_ic / final_ic_std if final_ic_std > 0 else 0
final_pmr = np.mean([1 if (p > 0 and a > 0) or (p < 0 and a < 0) else 0
                     for p, a in zip(all_preds[-360:], all_actuals[-360:])])

print(f"   Final IC (360h window): {final_ic:.6f}")
print(f"   Final IR: {final_ir:.3f}")
print(f"   Final PMR: {final_pmr:.3f}")

# Calculate metrics at each hour of simulation
hourly_metrics = []
for i in range(len(predictions)):
    # Use all history up to this point
    current_preds = hist_preds + predictions[:i+1]
    current_actuals = hist_actuals + actuals[:i+1]

    ic = rolling_ic(current_preds, current_actuals, min(360, len(current_preds)))

    # Calculate IR and PMR for available window
    window_size = min(360, len(current_preds))
    recent_preds = current_preds[-window_size:]
    recent_actuals = current_actuals[-window_size:]

    # IC std for IR
    ic_values = []
    for j in range(24, window_size, 24):
        sub_ic = np.corrcoef(recent_preds[-j:], recent_actuals[-j:])[0, 1] if len(recent_preds[-j:]) > 1 else 0
        ic_values.append(sub_ic)

    ic_std = np.std(ic_values) if ic_values else 0.001
    ir = ic / ic_std if ic_std > 0 else 0

    # PMR
    pmr = np.mean([1 if (p > 0 and a > 0) or (p < 0 and a < 0) else 0
                   for p, a in zip(recent_preds, recent_actuals)])

    hourly_metrics.append({
        'hour': i + 1,
        'timestamp': test_timestamps[i],
        'ic': ic,
        'ir': ir,
        'pmr': pmr,
        'ic_ok': ic >= IC_THRESHOLD,
        'ir_ok': ir >= IR_THRESHOLD,
        'pmr_ok': pmr >= PMR_THRESHOLD,
        'hard_gate_passed': (ic >= IC_THRESHOLD) and (ir >= IR_THRESHOLD) and (pmr >= PMR_THRESHOLD)
    })

# ============================================================================
# Verify Thresholds
# ============================================================================
print("\n[4/5] Verifying Hard gate thresholds...")

# Check threshold compliance
ic_compliance = sum(1 for m in hourly_metrics if m['ic_ok']) / len(hourly_metrics)
ir_compliance = sum(1 for m in hourly_metrics if m['ir_ok']) / len(hourly_metrics)
pmr_compliance = sum(1 for m in hourly_metrics if m['pmr_ok']) / len(hourly_metrics)
hard_gate_compliance = sum(1 for m in hourly_metrics if m['hard_gate_passed']) / len(hourly_metrics)

print(f"   IC compliance (>={IC_THRESHOLD}): {ic_compliance:.1%}")
print(f"   IR compliance (>={IR_THRESHOLD}): {ir_compliance:.1%}")
print(f"   PMR compliance (>={PMR_THRESHOLD}): {pmr_compliance:.1%}")
print(f"   Full Hard gate compliance: {hard_gate_compliance:.1%}")

# Check for any threshold violations
violations = []
for m in hourly_metrics:
    if not m['hard_gate_passed']:
        violations.append({
            'hour': m['hour'],
            'timestamp': str(m['timestamp']),
            'ic': m['ic'],
            'ir': m['ir'],
            'pmr': m['pmr']
        })

if violations:
    print(f"\n   WARNING: {len(violations)} threshold violations detected")
    for v in violations[:5]:  # Show first 5
        print(f"     Hour {v['hour']}: IC={v['ic']:.4f}, IR={v['ir']:.3f}, PMR={v['pmr']:.3f}")
else:
    print("\n   All 24 hours passed Hard gate thresholds!")

# ============================================================================
# Save Results
# ============================================================================
print("\n[5/5] Saving results...")

# Save hourly log
df_results = pd.DataFrame(results)
df_results['ic'] = [m['ic'] for m in hourly_metrics]
df_results['ir'] = [m['ir'] for m in hourly_metrics]
df_results['pmr'] = [m['pmr'] for m in hourly_metrics]
df_results['hard_gate_passed'] = [m['hard_gate_passed'] for m in hourly_metrics]

DRYRUN_LOG.parent.mkdir(parents=True, exist_ok=True)
df_results.to_csv(DRYRUN_LOG, index=False)
print(f"   Hourly log saved: {DRYRUN_LOG}")

# Save summary
summary = {
    'run_timestamp': datetime.now().isoformat(),
    'model': str(MODEL_PATH),
    'features': str(FEATURES_PATH),
    'strategy_name': config['strategy_name'],
    'version': config['version'],
    'simulation_hours': len(test_timestamps),
    'lookback_hours': lookback,
    'final_metrics': {
        'IC': round(final_ic, 6),
        'IR': round(final_ir, 3),
        'PMR': round(final_pmr, 3)
    },
    'thresholds': {
        'IC': IC_THRESHOLD,
        'IR': IR_THRESHOLD,
        'PMR': PMR_THRESHOLD
    },
    'compliance': {
        'IC': round(ic_compliance, 3),
        'IR': round(ir_compliance, 3),
        'PMR': round(pmr_compliance, 3),
        'hard_gate': round(hard_gate_compliance, 3)
    },
    'n_violations': len(violations),
    'violations': violations[:10],  # First 10 violations
    'pass': hard_gate_compliance >= 0.9,  # Pass if 90%+ hours comply
    'ready_for_production': len(violations) == 0
}

with open(DRYRUN_SUMMARY, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"   Summary saved: {DRYRUN_SUMMARY}")

# ============================================================================
# Final Report
# ============================================================================
print("\n" + "="*80)
print("DRY RUN SUMMARY")
print("="*80)

print(f"""
Strategy: {config['strategy_name']} v{config['version']}
Period: {df_sim.index.min()} to {df_sim.index.max()}
Simulation: {len(test_timestamps)} hourly cycles

FINAL METRICS:
  IC:  {final_ic:.6f} (threshold: {IC_THRESHOLD}) {'PASS' if final_ic >= IC_THRESHOLD else 'FAIL'}
  IR:  {final_ir:.3f} (threshold: {IR_THRESHOLD}) {'PASS' if final_ir >= IR_THRESHOLD else 'FAIL'}
  PMR: {final_pmr:.3f} (threshold: {PMR_THRESHOLD}) {'PASS' if final_pmr >= PMR_THRESHOLD else 'FAIL'}

COMPLIANCE RATES:
  IC compliance:  {ic_compliance:.1%}
  IR compliance:  {ir_compliance:.1%}
  PMR compliance: {pmr_compliance:.1%}
  Full Hard gate: {hard_gate_compliance:.1%}

VIOLATIONS: {len(violations)} / {len(hourly_metrics)} hours

VERDICT: {'READY FOR PRODUCTION' if len(violations) == 0 else 'NEEDS REVIEW - ' + str(len(violations)) + ' violations'}
""")

print("="*80)
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
