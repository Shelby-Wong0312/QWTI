"""
Inspect 12-month GDELT monthly files quality and coverage
"""
import pandas as pd
import glob
from pathlib import Path
import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 找到所有月度檔案
monthly_files = sorted(glob.glob('data/gdelt_hourly_monthly/*.parquet'))

print(f"=== 找到 {len(monthly_files)} 個月度檔案 ===\n")

bucket_cols = ['OIL', 'USD_RATE', 'GEOPOL', 'ECON', 'SUPPLY', 'ENERGY', 'OTHER']
total_stats = []

for fpath in monthly_files:
    month_name = Path(fpath).stem.replace('gdelt_hourly_', '')
    df = pd.read_parquet(fpath)

    # 基本統計
    total_rows = len(df)

    # Bucket coverage
    bucket_coverage = {}
    for col in bucket_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            non_zero = (df[col] > 0).sum()
            bucket_coverage[col] = {
                'non_null': non_null,
                'non_zero': non_zero,
                'non_null_pct': non_null / total_rows * 100,
                'non_zero_pct': non_zero / total_rows * 100
            }

    # 計算至少有一個bucket有數據的行數
    bucket_data = df[bucket_cols] if all(c in df.columns for c in bucket_cols) else pd.DataFrame()
    if not bucket_data.empty:
        has_any_bucket = (bucket_data.notna().any(axis=1)).sum()
        has_any_bucket_pct = has_any_bucket / total_rows * 100
    else:
        has_any_bucket = 0
        has_any_bucket_pct = 0

    print(f"【{month_name}】")
    print(f"  總行數: {total_rows:,}")
    print(f"  有Bucket數據: {has_any_bucket:,} ({has_any_bucket_pct:.1f}%)")

    # 顯示每個bucket的覆蓋率
    for col in bucket_cols:
        if col in bucket_coverage:
            cov = bucket_coverage[col]
            print(f"    {col:10s}: {cov['non_zero']:>6,} 非零 ({cov['non_zero_pct']:>5.1f}%)")

    print()

    total_stats.append({
        'month': month_name,
        'total_rows': total_rows,
        'has_bucket_rows': has_any_bucket,
        'coverage_pct': has_any_bucket_pct,
        'date_range': f"{df['dt'].min()} ~ {df['dt'].max()}" if 'dt' in df.columns else 'N/A'
    })

# 總覽
print("=== 總覽統計 ===")
stats_df = pd.DataFrame(total_stats)
print(stats_df.to_string(index=False))
print(f"\n總計行數: {stats_df['total_rows'].sum():,}")
print(f"總計有Bucket數據: {stats_df['has_bucket_rows'].sum():,}")
print(f"平均覆蓋率: {stats_df['coverage_pct'].mean():.1f}%")
