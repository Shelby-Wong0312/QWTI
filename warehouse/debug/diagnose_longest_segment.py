"""
Diagnose the longest 75-day continuous segment
"""
import pandas as pd
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load data
gdelt_df = pd.read_parquet('data/gdelt_hourly.parquet')
price_df = pd.read_parquet('data/features_hourly.parquet')
df = price_df.merge(gdelt_df, on='ts_utc', how='inner', suffixes=('_price', '_gdelt'))

# Bucket features
bucket_features = [
    'OIL_CORE_norm_art_cnt',
    'GEOPOL_norm_art_cnt',
    'USD_RATE_norm_art_cnt',
    'SUPPLY_CHAIN_norm_art_cnt',
    'MACRO_norm_art_cnt'
]

# Filter to rows with all bucket features
df_valid = df[df[bucket_features].notna().all(axis=1)].copy()
df_valid = df_valid.sort_values('ts_utc').reset_index(drop=True)

# Find longest segment
time_diffs = df_valid['ts_utc'].diff()
segments = []
current_start = 0
for i in range(1, len(df_valid)):
    if time_diffs.iloc[i] != pd.Timedelta(hours=1):
        if i - current_start > 0:
            segments.append((current_start, i-1, i-current_start))
        current_start = i
segments.append((current_start, len(df_valid)-1, len(df_valid)-current_start))
segments.sort(key=lambda x: x[2], reverse=True)

start_idx, end_idx, segment_length = segments[0]
df_segment = df_valid.iloc[start_idx:end_idx+1].copy()

print(f"=== Analyzing Longest Segment: {segment_length} hours ({segment_length/24:.1f} days) ===")
print(f"From: {df_segment.iloc[0]['ts_utc']}")
print(f"To: {df_segment.iloc[-1]['ts_utc']}")
print(f"Total rows: {len(df_segment)}\n")

# Check ret_1h distribution
print("ret_1h statistics:")
print(f"  Non-null: {df_segment['ret_1h'].notna().sum()}/{len(df_segment)} ({df_segment['ret_1h'].notna().sum()/len(df_segment)*100:.1f}%)")
print(f"  Non-zero: {(df_segment['ret_1h'] != 0).sum()}/{len(df_segment)} ({(df_segment['ret_1h'] != 0).sum()/len(df_segment)*100:.1f}%)")
print(f"  Zero: {(df_segment['ret_1h'] == 0).sum()}/{len(df_segment)} ({(df_segment['ret_1h'] == 0).sum()/len(df_segment)*100:.1f}%)")
print()

# Create H=1 forward return and lag
H = 1
LAG = 1
df_segment[f'ret_forward_{H}h'] = df_segment['ret_1h'].rolling(window=H, min_periods=H).sum().shift(-H)

for feat in bucket_features:
    df_segment[f'{feat}_lag{LAG}'] = df_segment[feat].shift(LAG)

lagged_features = [f'{feat}_lag{LAG}' for feat in bucket_features]

# Check validity
df_segment['has_ret_1h'] = df_segment['ret_1h'].notna()
df_segment['has_forward'] = df_segment[f'ret_forward_{H}h'].notna()
df_segment['has_lagged'] = df_segment[lagged_features].notna().all(axis=1)
df_segment['all_valid'] = df_segment['has_ret_1h'] & df_segment['has_forward'] & df_segment['has_lagged']

print("After lag=1 and forward return=1h:")
print(f"  Has ret_1h: {df_segment['has_ret_1h'].sum()}")
print(f"  Has forward ret: {df_segment['has_forward'].sum()}")
print(f"  Has lagged features: {df_segment['has_lagged'].sum()}")
print(f"  All valid: {df_segment['all_valid'].sum()}\n")

# Find continuous valid sequences
valid_df = df_segment[df_segment['all_valid']].copy()
print(f"Valid rows: {len(valid_df)}")

if len(valid_df) > 0:
    # Check if valid rows are consecutive
    valid_df = valid_df.reset_index(drop=True)
    valid_time_diffs = valid_df['ts_utc'].diff()
    consecutive_valid = (valid_time_diffs == pd.Timedelta(hours=1)).sum()
    print(f"Consecutive valid timestamps: {consecutive_valid}/{len(valid_df)-1} ({consecutive_valid/(len(valid_df)-1)*100:.1f}%)\n")

    # Find continuous valid segments
    valid_segments = []
    current_start = 0
    for i in range(1, len(valid_df)):
        if valid_time_diffs.iloc[i] != pd.Timedelta(hours=1):
            if i - current_start > 0:
                valid_segments.append((current_start, i-1, i-current_start))
            current_start = i
    valid_segments.append((current_start, len(valid_df)-1, len(valid_df)-current_start))
    valid_segments.sort(key=lambda x: x[2], reverse=True)

    print(f"Continuous valid segments: {len(valid_segments)}")
    print("Top 10 longest:")
    for i, (s, e, length) in enumerate(valid_segments[:10]):
        print(f"  {i+1}. Length: {length} hours ({length/24:.1f} days) - {valid_df.iloc[s]['ts_utc']} to {valid_df.iloc[e]['ts_utc']}")

    longest_valid = valid_segments[0]
    print(f"\nLongest valid segment: {longest_valid[2]} hours ({longest_valid[2]/24:.1f} days)")
    print(f"Need for IC evaluation: 90 days = {90*24} hours")

    if longest_valid[2] >= 500:
        print(f"\n✓ Sufficient for reduced-window IC evaluation!")
        print(f"  Can use train={int(longest_valid[2]*0.6)}h, test={int(longest_valid[2]*0.3)}h")
