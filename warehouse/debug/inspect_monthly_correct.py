"""
Check monthly GDELT files with correct bucket column names
"""
import pandas as pd
import glob
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

monthly_files = sorted(glob.glob('data/gdelt_hourly_monthly/*.parquet'))

# Correct normalized bucket columns
bucket_cols = {
    'OIL': 'OIL_CORE_norm_art_cnt',
    'GEOPOL': 'GEOPOL_norm_art_cnt',
    'USD_RATE': 'USD_RATE_norm_art_cnt',
    'SUPPLY': 'SUPPLY_CHAIN_norm_art_cnt',
    'ECON': 'MACRO_norm_art_cnt',
    'ENERGY': 'ESG_POLICY_norm_art_cnt',
    'OTHER': 'OTHER_norm_art_cnt'
}

print(f"=== Checking {len(monthly_files)} monthly files with CORRECT column names ===\n")

total_stats = []

for fpath in monthly_files:
    from pathlib import Path
    month_name = Path(fpath).stem.replace('gdelt_hourly_', '')
    df = pd.read_parquet(fpath)

    total_rows = len(df)

    # Check each bucket
    bucket_coverage = {}
    for bucket_name, col_name in bucket_cols.items():
        if col_name in df.columns:
            non_null = df[col_name].notna().sum()
            non_zero = (df[col_name] > 0).sum()
            bucket_coverage[bucket_name] = {
                'non_zero': non_zero,
                'pct': non_zero / total_rows * 100
            }

    # Count rows with ANY bucket data
    norm_cols = [col for col in bucket_cols.values() if col in df.columns]
    has_any_bucket = (df[norm_cols] > 0).any(axis=1).sum() if norm_cols else 0
    has_any_pct = has_any_bucket / total_rows * 100

    print(f"【{month_name}】")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Has bucket data: {has_any_bucket:,} ({has_any_pct:.1f}%)")
    print(f"  ALL_art_cnt avg: {df['ALL_art_cnt'].mean():.0f}")
    if 'mapped_ratio' in df.columns:
        print(f"  mapped_ratio avg: {df['mapped_ratio'].mean():.3f}")
    else:
        print(f"  mapped_ratio: N/A (column not found)")

    for bucket_name, stats in bucket_coverage.items():
        print(f"    {bucket_name:10s}: {stats['non_zero']:>6,} rows ({stats['pct']:>5.1f}%)")

    print()

    total_stats.append({
        'month': month_name,
        'total_rows': total_rows,
        'has_bucket': has_any_bucket,
        'coverage_pct': has_any_pct,
        'avg_articles': df['ALL_art_cnt'].mean(),
        'mapped_ratio': df['mapped_ratio'].mean() if 'mapped_ratio' in df.columns else 0.0
    })

# Summary
print("\n=== SUMMARY ===")
stats_df = pd.DataFrame(total_stats)
print(stats_df.to_string(index=False))
print(f"\nTotal rows: {stats_df['total_rows'].sum():,}")
print(f"Total with bucket data: {stats_df['has_bucket'].sum():,}")
print(f"Average coverage: {stats_df['coverage_pct'].mean():.1f}%")
print(f"Average mapped_ratio: {stats_df['mapped_ratio'].mean():.3f}")
