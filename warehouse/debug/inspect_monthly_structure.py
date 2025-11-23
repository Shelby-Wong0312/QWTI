"""
Check monthly file structure and raw data availability
"""
import pandas as pd
import glob
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Check first monthly file
monthly_files = sorted(glob.glob('data/gdelt_hourly_monthly/*.parquet'))
first_file = monthly_files[0]
print(f"Checking: {first_file}\n")

df = pd.read_parquet(first_file)
print("Columns:")
print(df.columns.tolist())
print(f"\nShape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nNull counts:")
print(df.isnull().sum())

# Check if raw GDELT data exists
print("\n" + "="*60)
print("Checking for raw GDELT data...")
raw_monthly = glob.glob('data/gdelt_raw/monthly/*.parquet')
raw_daily = glob.glob('data/gdelt_raw/daily/*.parquet')
raw_other = glob.glob('data/gdelt_raw/*.parquet')

print(f"Raw monthly files: {len(raw_monthly)}")
print(f"Raw daily files: {len(raw_daily)}")
print(f"Raw other files: {len(raw_other)}")

if raw_monthly:
    print("\nSample raw monthly files:")
    for f in raw_monthly[:3]:
        print(f"  {f}")
if raw_daily:
    print("\nSample raw daily files:")
    for f in raw_daily[:3]:
        print(f"  {f}")
if raw_other:
    print("\nOther raw files:")
    for f in raw_other[:3]:
        print(f"  {f}")
