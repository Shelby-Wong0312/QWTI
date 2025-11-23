"""
Diagnose why we only have 111 valid rows after lag+forward return filtering
"""
import pandas as pd
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load data
gdelt_df = pd.read_parquet('data/gdelt_hourly.parquet')
price_df = pd.read_parquet('data/features_hourly.parquet')

# Merge
df = price_df.merge(gdelt_df, on='ts_utc', how='inner', suffixes=('_price', '_gdelt'))

# Bucket features
bucket_features = [
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt'
]

# Filter to rows with all features
df_valid = df[df[bucket_features].notna().all(axis=1)].copy()
print(f"Step 1 - Has all bucket features: {len(df_valid)} rows")

# Filter to non-zero ret_1h
df_valid = df_valid[df_valid['ret_1h'] != 0].copy()
print(f"Step 2 - Non-zero ret_1h: {len(df_valid)} rows")

# Sort by time
df_valid = df_valid.sort_values('ts_utc').reset_index(drop=True)

# Check time distribution
print(f"\nTime range: {df_valid['ts_utc'].min()} to {df_valid['ts_utc'].max()}")
print(f"Time span: {(df_valid['ts_utc'].max() - df_valid['ts_utc'].min()).days} days")

# Check for gaps
df_valid['time_diff'] = df_valid['ts_utc'].diff()
large_gaps = df_valid[df_valid['time_diff'] > pd.Timedelta(hours=48)]
print(f"\nLarge gaps (>48h): {len(large_gaps)}")
if len(large_gaps) > 0:
    print("Largest gaps:")
    for i, row in large_gaps.head(10).iterrows():
        print(f"  {row['ts_utc']}: gap = {row['time_diff']}")

# Create H=1 forward return
H = 1
LAG = 1
df_valid[f'ret_forward_{H}h'] = df_valid['ret_1h'].rolling(window=H, min_periods=H).sum().shift(-H)

# Apply lag
for feat in bucket_features:
    df_valid[f'{feat}_lag{LAG}'] = df_valid[feat].shift(LAG)

lagged_features = [f'{feat}_lag{LAG}' for feat in bucket_features]

# Check validity
df_valid['has_forward_ret'] = df_valid[f'ret_forward_{H}h'].notna()
df_valid['has_lagged_features'] = df_valid[lagged_features].notna().all(axis=1)
df_valid['is_valid'] = df_valid['has_forward_ret'] & df_valid['has_lagged_features']

print(f"\nStep 3 - After lag={LAG} and forward_ret={H}h:")
print(f"  Has forward return: {df_valid['has_forward_ret'].sum()}")
print(f"  Has lagged features: {df_valid['has_lagged_features'].sum()}")
print(f"  Both valid: {df_valid['is_valid'].sum()}")

# Analyze why forward return is missing
print(f"\nWhy forward return is missing?")
df_valid['ret_1h_next'] = df_valid['ret_1h'].shift(-1)
df_valid['ret_1h_next2'] = df_valid['ret_1h'].shift(-2)

has_next = df_valid['ret_1h_next'].notna().sum()
has_next2 = df_valid['ret_1h_next2'].notna().sum()
print(f"  Has next hour ret: {has_next}/{len(df_valid)}")
print(f"  Has +2 hour ret: {has_next2}/{len(df_valid)}")

# Check consecutive hours
df_valid['next_hour_consecutive'] = (df_valid['ts_utc'].shift(-1) - df_valid['ts_utc']) == pd.Timedelta(hours=1)
consecutive = df_valid['next_hour_consecutive'].sum()
print(f"  Consecutive hours: {consecutive}/{len(df_valid)} ({consecutive/len(df_valid)*100:.1f}%)")

# Monthly breakdown
df_valid['month'] = df_valid['ts_utc'].dt.to_period('M')
monthly_counts = df_valid.groupby('month').agg({
    'ts_utc': 'count',
    'next_hour_consecutive': 'sum',
    'is_valid': 'sum'
})
monthly_counts.columns = ['total_rows', 'consecutive_hours', 'valid_for_IC']
print(f"\nMonthly breakdown:")
print(monthly_counts.to_string())

# Check continuous segments
segments = []
current_segment_start = df_valid.iloc[0]['ts_utc']
current_segment_count = 1

for i in range(1, len(df_valid)):
    if (df_valid.iloc[i]['ts_utc'] - df_valid.iloc[i-1]['ts_utc']) == pd.Timedelta(hours=1):
        current_segment_count += 1
    else:
        segments.append({'start': current_segment_start, 'count': current_segment_count})
        current_segment_start = df_valid.iloc[i]['ts_utc']
        current_segment_count = 1

segments.append({'start': current_segment_start, 'count': current_segment_count})

print(f"\nContinuous segments: {len(segments)}")
print("Top 10 longest segments:")
segments_df = pd.DataFrame(segments).sort_values('count', ascending=False)
print(segments_df.head(10).to_string(index=False))

print(f"\nLongest segment: {segments_df.iloc[0]['count']} hours = {segments_df.iloc[0]['count']/24:.1f} days")
print(f"Need for IC evaluation: {60+30} days = {(60+30)*24} hours")
