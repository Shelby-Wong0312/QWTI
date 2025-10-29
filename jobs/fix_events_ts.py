import pandas as pd, numpy as np, os, sys
from pathlib import Path

src = Path("data/events_calendar.csv")
if not src.exists():
    print("[ERR] data/events_calendar.csv 不存在"); sys.exit(0)

df = pd.read_csv(src)
orig_cols = set(df.columns)
# 已有 ts 就標準化
cands = [
    "ts","timestamp","time","datetime","event_time","start","start_time",
    "utc_time","updateTimeUTC","date","Date","DATE"
]
ts = None
for c in cands:
    if c in df.columns:
        s = pd.to_datetime(df[c], utc=True, errors="coerce")
        if s.notna().sum()>0:
            ts = s.dt.floor("h")
            break

if ts is None:
    # 嘗試任何包含 time/date 的欄位
    found = None
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            s = pd.to_datetime(df[c], utc=True, errors="coerce")
            if s.notna().sum()>0:
                found = c; ts = s.dt.floor("h"); break
    if ts is None:
        # 無法解析  輸出空，讓上游看見
        print("[WARN] 找不到可解析的時間欄，寫回空 ts 的檔案。")
        df = pd.DataFrame(columns=["ts","event","type"])
        df.to_csv(src, index=False); sys.exit(0)

df_out = df.copy()
df_out["ts"] = ts
if "event" not in df_out.columns: df_out["event"] = "EIA"
if "type"  not in df_out.columns: df_out["type"]  = "EIA"
df_out = df_out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

# 備份 & 覆蓋
bak = src.with_suffix(".orig.csv")
try:
    if not bak.exists(): os.replace(src, bak)
except Exception: pass
df_out.to_csv(src, index=False)
print(f"[OK] events_calendar.csv 標準化完成 rows={len(df_out)}  cols={list(df_out.columns)}")
