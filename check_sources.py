import pandas as pd

print("=== GDELT hourly ===")
try:
    g = pd.read_parquet('data/gdelt_hourly.parquet')
    print('rows:', len(g))
    print('columns:', list(g.columns))
    print('ts range:', g['ts_utc'].min() if 'ts_utc' in g.columns else g.iloc[:,0].min(), 
          '->', g['ts_utc'].max() if 'ts_utc' in g.columns else g.iloc[:,0].max())
except Exception as e:
    print('ERROR:', e)

print("\n=== features_hourly_with_term ===")
try:
    f = pd.read_parquet('data/features_hourly_with_term.parquet')
    print('rows:', len(f))
    print('columns:', list(f.columns))
    print('ts range:', f['ts_utc'].min(), '->', f['ts_utc'].max())
except Exception as e:
    print('ERROR:', e)

print("\n=== WTI hourly ===")
try:
    w = pd.read_parquet('data/wti_hourly_capital.parquet')
    print('rows:', len(w))
    print('columns:', list(w.columns))
    print('ts range:', w['ts_utc'].min(), '->', w['ts_utc'].max())
except Exception as e:
    print('ERROR:', e)
