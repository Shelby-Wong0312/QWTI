import pandas as pd
from datetime import datetime, timezone, timedelta

start = pd.Timestamp("2023-01-01", tz="UTC")
end   = pd.Timestamp("2025-10-01", tz="UTC")

# 每週三 15:30 UTC（簡化；DST 未微調，足夠回測用）
all_days = pd.date_range(start=start, end=end, freq="W-WED", tz="UTC")
eia_times = [d + pd.Timedelta(hours=15, minutes=30) for d in all_days]

df = pd.DataFrame({
    "event_time_utc":[t.isoformat() for t in eia_times],
    "event_type":["EIA"]*len(eia_times),
    "event_subtype":["WPSR"]*len(eia_times),
    "url":[""]*len(eia_times),
    "notes":["auto"]*len(eia_times),
    "weight":[1.0]*len(eia_times)
})
df.to_csv("data/events_calendar.csv", index=False)
print("EIA events written:", len(df))
