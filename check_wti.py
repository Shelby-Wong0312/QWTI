import pandas as pd
df = pd.read_parquet('data/wti_hourly_capital.parquet')
print('rows:', len(df))
print('columns:', list(df.columns))
print('ts range:', df['ts_utc'].min(), '->', df['ts_utc'].max())
print()
print(df.head(10))
