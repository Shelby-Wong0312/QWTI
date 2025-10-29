import pandas as pd, numpy as np
from pathlib import Path

def read(p, keep=None):
    p=Path(p)
    if not p.exists(): return pd.DataFrame()
    df=pd.read_csv(p)
    if "ts" in df.columns:
        df["ts"]=pd.to_datetime(df["ts"], utc=True, errors="coerce").dt.floor("h")
        df=df.dropna(subset=["ts"])
    return df if keep is None else df[[c for c in keep if c in df.columns]]

feat=read("warehouse/features_hourly_v2.csv")
term=read("data/term_crack_ovx_hourly.csv",["ts","cl1_cl2","crack_rb","crack_ho"])
news=read("data/gdelt_hourly.csv",["ts","art_cnt","tone_avg","tone_pos_ratio"])
sig =read("warehouse/signals_hourly_exp3.csv")  # 讀全部，之後判斷欄位
ev  =read("data/events_calendar.csv",["ts"])

# 統一時間索引
idx = pd.DatetimeIndex([], tz="UTC")
for d in (feat,term,news,sig):
    if not d.empty:
        idx = idx.union(pd.DatetimeIndex(d["ts"]))

def align_rate(df):
    if df.empty or idx.size==0: return np.nan
    return df.set_index("ts").reindex(idx).notna().all(axis=1).mean()

row={}
row["align_features"]=align_rate(feat)
row["align_term"    ]=align_rate(term)
row["align_news"    ]=align_rate(news)
row["align_signals" ]=align_rate(sig)
row["share_cl1cl2_zero"]=float((term["cl1_cl2"].fillna(0)==0).mean()) if not term.empty else np.nan
row["share_crack_zero" ]=float(((term["crack_rb"].fillna(0)==0)&(term["crack_ho"].fillna(0)==0)).mean()) if not term.empty else np.nan
row["share_news_hour_has_article"]=float((news["art_cnt"].fillna(0)>0).mean()) if not news.empty else np.nan
if not sig.empty and "w_t" in sig.columns:
    row["share_wt_zero"]=float((sig["w_t"].fillna(0).abs()<1e-6).mean())
    row["avg_turnover"] =float(sig["w_t"].diff().abs().mean())
else:
    row["share_wt_zero"]=np.nan; row["avg_turnover"]=np.nan
row["events_total"]=int(len(ev))
if not ev.empty and not sig.empty and "in_event" in sig.columns:
    row["events_paired_share"]=float(sig.set_index("ts").reindex(ev["ts"])["in_event"].fillna(False).mean())
else:
    row["events_paired_share"]=np.nan

out = pd.DataFrame([row])
Path("warehouse").mkdir(parents=True, exist_ok=True)
out.to_csv("warehouse/data_gaps_report.csv", index=False)
print(out.to_string(index=False))
print("[WROTE] warehouse/data_gaps_report.csv")
