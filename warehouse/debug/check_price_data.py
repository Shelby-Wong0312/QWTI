"""
Check price data time range and overlap with GDELT
"""
import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load price data
price_df = pd.read_parquet('data/features_hourly.parquet')
print("=== Price Data (features_hourly.parquet) ===")
print(f"  Time range: {price_df['ts_utc'].min()} to {price_df['ts_utc'].max()}")
print(f"  Total rows: {len(price_df):,}")
non_zero_ret = (price_df['ret_1h'] != 0).sum()
print(f"  Non-zero ret_1h: {non_zero_ret:,} ({non_zero_ret/len(price_df)*100:.1f}%)")

# Load GDELT data (current)
gdelt_df = pd.read_parquet('data/gdelt_hourly.parquet')
print("\n=== Current GDELT Data (gdelt_hourly.parquet) ===")
print(f"  Time range: {gdelt_df['ts_utc'].min()} to {gdelt_df['ts_utc'].max()}")
print(f"  Total rows: {len(gdelt_df):,}")

# Check overlap
merged = price_df.merge(gdelt_df, on='ts_utc', how='inner', suffixes=('_price', '_gdelt'))
print(f"\n=== Overlap (current) ===")
print(f"  Overlapping timestamps: {len(merged):,}")

# Now check if we rebuild from monthly files
import glob
monthly_files = sorted(glob.glob('data/gdelt_hourly_monthly/*.parquet'))
monthly_dfs = []
for f in monthly_files:
    df = pd.read_parquet(f)
    monthly_dfs.append(df)

combined_gdelt = pd.concat(monthly_dfs, ignore_index=True).sort_values('ts_utc')
print(f"\n=== Rebuilt GDELT from monthly files ===")
print(f"  Time range: {combined_gdelt['ts_utc'].min()} to {combined_gdelt['ts_utc'].max()}")
print(f"  Total rows: {len(combined_gdelt):,}")

# Check which months have bucket data
bucket_cols = ['OIL_CORE_norm_art_cnt', 'GEOPOL_norm_art_cnt', 'USD_RATE_norm_art_cnt']
has_bucket = combined_gdelt[bucket_cols[0]].notna() if bucket_cols[0] in combined_gdelt.columns else pd.Series([False]*len(combined_gdelt))
print(f"  Rows with bucket data: {has_bucket.sum():,} ({has_bucket.sum()/len(combined_gdelt)*100:.1f}%)")

# Check overlap with rebuilt GDELT
merged_new = price_df.merge(combined_gdelt[has_bucket], on='ts_utc', how='inner', suffixes=('_price', '_gdelt'))
print(f"\n=== Overlap (rebuilt, bucket data only) ===")
print(f"  Overlapping timestamps: {len(merged_new):,}")
if len(merged_new) > 0:
    print(f"  Time range: {merged_new['ts_utc'].min()} to {merged_new['ts_utc'].max()}")
    non_zero_in_overlap = (merged_new['ret_1h'] != 0).sum()
    print(f"  Non-zero ret_1h in overlap: {non_zero_in_overlap:,} ({non_zero_in_overlap/len(merged_new)*100:.1f}%)")
