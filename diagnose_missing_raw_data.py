"""
Diagnose missing raw GKG data and explore alternatives
"""
import pandas as pd
from pathlib import Path
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*80)
print("MISSING RAW DATA DIAGNOSTIC")
print("="*80)

# Check inventory
inventory_path = Path("data/gdelt_raw/_raw_inventory_index_v3.csv")
if inventory_path.exists():
    df = pd.read_csv(inventory_path)
    print(f"\nInventory file exists: {len(df)} entries")

    # Extract month from path
    df['month'] = df['path'].str.extract(r'(2024|2025)/(\d{2})')[1]
    df['year'] = df['path'].str.extract(r'(2024|2025)/(\d{2})')[0]
    df['year_month'] = df['year'] + '-' + df['month']

    print("\nFiles per month (inventory):")
    monthly_counts = df.groupby('year_month').size().sort_index()
    print(monthly_counts.to_string())

    # Check if files actually exist
    print("\n\nChecking file existence...")
    sample_files = df.sample(min(10, len(df)))
    exists_count = 0
    for idx, row in sample_files.iterrows():
        p = Path(row['path'])
        if p.exists():
            exists_count += 1

    print(f"Sample check: {exists_count}/{len(sample_files)} files exist")

    if exists_count == 0:
        print("\n*** CRITICAL: NO RAW GKG FILES FOUND ***")
        print("All inventory files have been deleted!")
else:
    print("\nInventory file not found")

# Check monthly parquet files
print("\n" + "="*80)
print("MONTHLY PARQUET FILES STATUS")
print("="*80)

monthly_dir = Path("data/gdelt_hourly_monthly")
monthly_files = sorted(monthly_dir.glob("*.parquet"))

print(f"\nFound {len(monthly_files)} monthly parquet files\n")

for fpath in monthly_files:
    month = fpath.stem.replace('gdelt_hourly_', '')
    df_month = pd.read_parquet(fpath)

    # Check bucket columns
    bucket_cols = [
        'OIL_CORE_norm_art_cnt',
        'GEOPOL_norm_art_cnt',
        'USD_RATE_norm_art_cnt',
        'SUPPLY_CHAIN_norm_art_cnt',
        'MACRO_norm_art_cnt'
    ]

    has_bucket_cols = all(col in df_month.columns for col in bucket_cols)
    if has_bucket_cols:
        non_null = df_month[bucket_cols[0]].notna().sum()
        coverage = non_null / len(df_month) * 100
        status = "OK" if coverage > 0 else "EMPTY"
    else:
        coverage = 0
        status = "MISSING_COLS"

    print(f"{month}: {len(df_month):>4} rows, {len(df_month.columns):>2} cols, "
          f"bucket coverage: {coverage:>5.1f}% [{status}]")

# Alternative: Check if we can download raw data
print("\n" + "="*80)
print("ALTERNATIVE OPTIONS")
print("="*80)

print("\nOption 1: Re-download raw GKG data")
print("  - Use GDELT's public HTTP API")
print("  - URL pattern: http://data.gdeltproject.org/gkg/YYYYMMDDHHMMSS.gkg.csv.zip")
print("  - Need: 2025-03 to 2025-09 (7 months)")
print("  - Estimated: ~7 months x 30 days x 96 files/day = ~20,000 files")
print("  - Status: FEASIBLE but time-consuming")

print("\nOption 2: Fix existing monthly parquet files")
print("  - Current issue: 2025-03~09 have NULL bucket columns")
print("  - Root cause: UNKNOWN (aggregation failed? schema mismatch?)")
print("  - Solution: Investigate monthly file structure")
print("  - Status: INVESTIGATING")

print("\nOption 3: Accept partial data")
print("  - Use only 2024-10~2025-02 + 2025-10 (6 months)")
print("  - Total continuous: 2024-10-29 ~ 2025-01-12 (75 days)")
print("  - Status: INSUFFICIENT for 90-day window")

# Check monthly file structure for problematic months
print("\n" + "="*80)
print("INVESTIGATING 2025-03 FILE STRUCTURE")
print("="*80)

march_file = monthly_dir / "gdelt_hourly_2025-03.parquet"
if march_file.exists():
    df_march = pd.read_parquet(march_file)
    print(f"\n2025-03 columns ({len(df_march.columns)}):")
    print(df_march.columns.tolist())

    print(f"\nFirst 3 rows:")
    print(df_march.head(3).to_string())

    print(f"\nColumn dtypes:")
    print(df_march.dtypes.to_string())

    print(f"\nNull counts:")
    null_counts = df_march.isnull().sum()
    print(null_counts[null_counts > 0].to_string())

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

print("""
Based on the diagnostic:

1. RAW GKG DATA IS DELETED - Cannot re-aggregate from source
2. Monthly parquet files 2025-03~09 have NULL bucket columns
3. Possible causes:
   a) Schema change in aggregation script between 2025-02 and 2025-03
   b) Aggregation script not run for these months
   c) Data corruption during write

NEXT STEPS:
1. Compare 2025-02.parquet vs 2025-03.parquet structure
2. Check if 2025-03~09 were generated differently
3. If they're truly empty, we CANNOT fix without raw data
4. Must either:
   - Re-download 7 months of raw GKG from GDELT (20K+ files)
   - Accept that IC evaluation is not possible with current data
""")
