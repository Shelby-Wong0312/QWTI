#!/usr/bin/env python3
"""
Fix cl1_cl2 using CL-BZ spread and run stability validation
- Uses Yahoo Finance CL=F (WTI) and BZ=F (Brent) hourly data
- CL-BZ spread serves as term structure proxy (contango/backwardation indicator)
- Rebuilds features_hourly_with_regime.parquet
- Runs stability validation with recommended config
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import lightgbm as lgb

OUTPUT_DIR = Path("warehouse/ic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Fix cl1_cl2 and Run Stability Validation")
print("="*80)

# ============================================================================
# Step 1: Download CL-BZ spread from Yahoo Finance
# ============================================================================
print("\n[1/5] Downloading CL-BZ spread from Yahoo Finance...")

cl = yf.download('CL=F', period='max', interval='1h', progress=False)
bz = yf.download('BZ=F', period='max', interval='1h', progress=False)

# Handle multi-level columns
if isinstance(cl.columns, pd.MultiIndex):
    cl.columns = cl.columns.droplevel(1)
if isinstance(bz.columns, pd.MultiIndex):
    bz.columns = bz.columns.droplevel(1)

print(f"   CL=F: {len(cl)} rows")
print(f"   BZ=F: {len(bz)} rows")

# Extract Close prices
cl = cl[['Close']].rename(columns={'Close': 'cl_close'})
bz = bz[['Close']].rename(columns={'Close': 'bz_close'})

# Merge and calculate spread
spread_df = cl.join(bz, how='inner')
spread_df['cl_bz_spread'] = spread_df['cl_close'] - spread_df['bz_close']

# Convert index to UTC
spread_df.index = spread_df.index.tz_convert('UTC')
spread_df = spread_df.reset_index().rename(columns={'Datetime': 'ts'})
spread_df['ts'] = pd.to_datetime(spread_df['ts'], utc=True)

print(f"   Merged spread: {len(spread_df)} rows")
print(f"   Date range: {spread_df['ts'].min()} to {spread_df['ts'].max()}")
print(f"   CL-BZ spread mean: {spread_df['cl_bz_spread'].mean():.3f}")

# ============================================================================
# Step 2: Update term_crack_ovx_hourly.csv
# ============================================================================
print("\n[2/5] Updating term_crack_ovx_hourly.csv...")

# Read existing file
term_path = Path("data/term_crack_ovx_hourly.csv")
term_df = pd.read_csv(term_path)
term_df['ts'] = pd.to_datetime(term_df['ts'], utc=True)

print(f"   Original term file: {len(term_df)} rows")
print(f"   Original cl1_cl2 non-zero: {(term_df['cl1_cl2'] != 0).sum()}")

# Merge CL-BZ spread as cl1_cl2
spread_for_merge = spread_df[['ts', 'cl_bz_spread']].copy()
spread_for_merge['ts'] = spread_for_merge['ts'].dt.floor('h')  # Round to hour
spread_for_merge = spread_for_merge.drop_duplicates('ts', keep='last')

# Update cl1_cl2 column
term_df = term_df.merge(spread_for_merge, on='ts', how='left')
term_df['cl1_cl2'] = term_df['cl_bz_spread'].fillna(0)
term_df = term_df.drop(columns=['cl_bz_spread'])

print(f"   Updated cl1_cl2 non-zero: {(term_df['cl1_cl2'] != 0).sum()}")

# Check 2024-10+ coverage
term_2024 = term_df[term_df['ts'] >= '2024-10-01']
print(f"   Since 2024-10-01: {(term_2024['cl1_cl2'] != 0).sum()}/{len(term_2024)} non-zero")

# Save
term_df.to_csv(term_path, index=False)
print(f"   Saved to: {term_path}")

# ============================================================================
# Step 3: Rebuild features_hourly_with_regime.parquet
# ============================================================================
print("\n[3/5] Rebuilding features_hourly_with_regime.parquet...")

# Read GDELT data
gdelt_path = Path("data/gdelt_hourly.parquet")
gdelt_df = pd.read_parquet(gdelt_path)
gdelt_df['ts_utc'] = pd.to_datetime(gdelt_df['ts_utc'], utc=True)
print(f"   GDELT: {len(gdelt_df)} rows")

# Read price data
price_path = Path("data/features_hourly.parquet")
price_df = pd.read_parquet(price_path)
price_df['ts_utc'] = pd.to_datetime(price_df['ts_utc'], utc=True)
print(f"   Price: {len(price_df)} rows")

# Read updated term data
term_df = pd.read_csv(term_path)
term_df['ts'] = pd.to_datetime(term_df['ts'], utc=True)
term_df = term_df.rename(columns={'ts': 'ts_utc'})
print(f"   Term: {len(term_df)} rows")

# Merge all
merged = gdelt_df.merge(price_df[['ts_utc', 'ret_1h']], on='ts_utc', how='inner')
merged = merged.rename(columns={'ret_1h': 'wti_returns'})
merged = merged.merge(term_df[['ts_utc', 'cl1_cl2', 'ovx']], on='ts_utc', how='left')

print(f"   Merged: {len(merged)} rows")

# Add EIA flags
def add_eia_flags(df):
    df = df.copy()
    df['hour'] = df['ts_utc'].dt.hour
    df['weekday'] = df['ts_utc'].dt.weekday

    # EIA report typically Wednesday 10:30 AM ET = 14:30 UTC (winter) or 15:30 UTC (summer)
    df['eia_day'] = ((df['weekday'] == 2)).astype(int)  # Wednesday
    df['eia_pre_4h'] = ((df['weekday'] == 2) & (df['hour'] >= 10) & (df['hour'] < 14)).astype(int)
    df['eia_post_4h'] = ((df['weekday'] == 2) & (df['hour'] >= 14) & (df['hour'] < 18)).astype(int)

    return df.drop(columns=['hour', 'weekday'])

merged = add_eia_flags(merged)

# Add regime flags
def add_regime_flags(df):
    df = df.copy()

    # Vol regime based on rolling volatility
    df['vol_24h'] = df['wti_returns'].rolling(24).std()
    vol_median = df['vol_24h'].median()
    df['vol_regime_high'] = (df['vol_24h'] > vol_median * 1.5).astype(int)
    df['vol_regime_low'] = (df['vol_24h'] < vol_median * 0.5).astype(int)

    # OVX regime
    ovx_median = df['ovx'].median()
    df['ovx_high'] = (df['ovx'] > ovx_median * 1.2).astype(int)
    df['ovx_low'] = (df['ovx'] < ovx_median * 0.8).astype(int)

    # Momentum
    df['momentum_24h'] = df['wti_returns'].rolling(24).mean()

    # GDELT high activity
    gdelt_cols = ['OIL_CORE_norm_art_cnt', 'GEOPOL_norm_art_cnt', 'USD_RATE_norm_art_cnt',
                  'SUPPLY_CHAIN_norm_art_cnt', 'MACRO_norm_art_cnt']
    available_cols = [c for c in gdelt_cols if c in df.columns]
    if available_cols:
        df['gdelt_total'] = df[available_cols].sum(axis=1)
        gdelt_median = df['gdelt_total'].median()
        df['gdelt_high'] = (df['gdelt_total'] > gdelt_median * 2).astype(int)
        df = df.drop(columns=['gdelt_total'])
    else:
        df['gdelt_high'] = 0

    df = df.drop(columns=['vol_24h'], errors='ignore')

    return df

merged = add_regime_flags(merged)

# Set index and save
merged = merged.set_index('ts_utc').sort_index()

# Select final columns
final_cols = [
    'OIL_CORE_norm_art_cnt', 'GEOPOL_norm_art_cnt', 'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt', 'MACRO_norm_art_cnt',
    'cl1_cl2', 'ovx',
    'eia_pre_4h', 'eia_post_4h', 'eia_day',
    'vol_regime_high', 'vol_regime_low', 'ovx_high', 'ovx_low',
    'momentum_24h', 'gdelt_high', 'wti_returns'
]
final_cols = [c for c in final_cols if c in merged.columns]
merged = merged[final_cols]

# Fill NaN
merged = merged.fillna(0)

# Save
output_path = Path("features_hourly_with_regime.parquet")
merged.to_parquet(output_path)
print(f"   Saved to: {output_path}")
print(f"   Final shape: {merged.shape}")

# Check cl1_cl2 coverage
merged_2024 = merged[merged.index >= '2024-10-01']
print(f"   2024-10+ cl1_cl2 non-zero: {(merged_2024['cl1_cl2'] != 0).sum()}/{len(merged_2024)}")

# ============================================================================
# Step 4: Run Stability Validation
# ============================================================================
print("\n[4/5] Running stability validation...")

START_DATE = "2024-10-01"
TRAIN_HOURS = 1440
TEST_HOURS = 360
H = 1

# 8 features now (including cl1_cl2)
FEATURE_COLS = [
    'cl1_cl2',  # Now has real values!
    'ovx',
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt',
    'momentum_24h',
]

# Filter to clean period
df = merged[merged.index >= START_DATE].copy()
print(f"   Filtered data: {len(df)} rows")

# Prepare
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
df['target'] = df['wti_returns'].shift(-H)
df = df.dropna(subset=['target'])

# Winsorize
for col in FEATURE_COLS + ['target']:
    p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(p1, p99)

print(f"   After preparation: {len(df)} rows")

# Feature coverage
print("\n   Feature coverage:")
for col in FEATURE_COLS:
    non_zero = (df[col] != 0).sum()
    pct = non_zero / len(df) * 100
    print(f"     {col}: {pct:.1f}%")

# Configs to test
CONFIGS = [
    {'max_depth': 3, 'num_leaves': 15, 'learning_rate': 0.05, 'subsample': 0.8, 'n_estimators': 200, 'name': 'recommended'},
    {'max_depth': 3, 'num_leaves': 15, 'learning_rate': 0.03, 'subsample': 0.85, 'n_estimators': 250, 'name': 'conservative'},
    {'max_depth': 3, 'num_leaves': 15, 'learning_rate': 0.05, 'subsample': 0.9, 'n_estimators': 200, 'name': 'high_subsample'},
]

X = df[FEATURE_COLS].values
y = df['target'].values
timestamps = df.index.tolist()

n_samples = len(X)
window_starts = np.arange(0, n_samples - TRAIN_HOURS - TEST_HOURS + 1, TEST_HOURS)

print(f"\n   Rolling windows: {len(window_starts)}")

results = []

for cfg in CONFIGS:
    name = cfg['name']
    print(f"\n   --- {name} ---")

    ic_list = []

    for start in window_starts:
        train_end = start + TRAIN_HOURS
        test_end = train_end + TEST_HOURS
        if test_end > n_samples:
            break

        X_train, y_train = X[start:train_end], y[start:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]

        model = lgb.LGBMRegressor(
            max_depth=cfg['max_depth'],
            num_leaves=cfg['num_leaves'],
            learning_rate=cfg['learning_rate'],
            subsample=cfg['subsample'],
            n_estimators=cfg['n_estimators'],
            random_state=42,
            verbosity=-1,
            force_col_wise=True
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        ic = np.corrcoef(y_pred, y_test)[0, 1] if np.std(y_pred) > 0 else 0
        ic_list.append(ic)

    # Metrics
    valid_ics = [ic for ic in ic_list if not np.isnan(ic)]
    ic_mean = np.mean(valid_ics)
    ic_std = np.std(valid_ics)
    ir = ic_mean / ic_std if ic_std > 0 else 0
    pmr = np.mean([ic > 0 for ic in valid_ics])

    ic_ok = ic_mean >= 0.02
    ir_ok = ir >= 0.5
    pmr_ok = pmr >= 0.55
    is_hard = ic_ok and ir_ok and pmr_ok

    status = "HARD!" if is_hard else ""
    print(f"   IC={ic_mean:.6f} {'OK' if ic_ok else 'X'}, IR={ir:.3f} {'OK' if ir_ok else 'X'}, PMR={pmr:.3f} {'OK' if pmr_ok else 'X'} {status}")

    results.append({
        'name': name,
        'IC': ic_mean,
        'IC_std': ic_std,
        'IR': ir,
        'PMR': pmr,
        'ic_ok': ic_ok,
        'ir_ok': ir_ok,
        'pmr_ok': pmr_ok,
        'is_hard': is_hard,
        'window_ics': valid_ics,
        **{k:v for k,v in cfg.items() if k != 'name'}
    })

# ============================================================================
# Step 5: Save Results
# ============================================================================
print("\n[5/5] Saving results...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df = pd.DataFrame([{k:v for k,v in r.items() if k != 'window_ics'} for r in results])
results_df = results_df.sort_values('IR', ascending=False)

output_file = OUTPUT_DIR / f"cl1_cl2_fixed_validation_{timestamp}.csv"
results_df.to_csv(output_file, index=False)
print(f"   Saved to: {output_file}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

for r in results:
    ic_s = "OK" if r['ic_ok'] else "X "
    ir_s = "OK" if r['ir_ok'] else "X "
    pmr_s = "OK" if r['pmr_ok'] else "X "
    hard = " *** HARD ***" if r['is_hard'] else ""
    print(f"  {r['name']:20s}: IC={r['IC']:.6f}[{ic_s}] IR={r['IR']:.3f}[{ir_s}] PMR={r['PMR']:.3f}[{pmr_s}]{hard}")

# Check for HARD achievers
hard_achievers = [r for r in results if r['is_hard']]
if hard_achievers:
    print(f"\n*** HARD THRESHOLD ACHIEVED! ({len(hard_achievers)} configs) ***")
    best = hard_achievers[0]
    print(f"\nBest config: {best['name']}")
    print(f"  IC:  {best['IC']:.6f}")
    print(f"  IR:  {best['IR']:.3f}")
    print(f"  PMR: {best['PMR']:.3f}")
else:
    print("\n--- Hard threshold NOT achieved ---")
    best = results[0]
    print(f"Best IR: {best['IR']:.3f} (gap: {0.5 - best['IR']:.3f})")

print("\n" + "="*80)
print(f"Results saved to: {output_file}")
print("="*80)
