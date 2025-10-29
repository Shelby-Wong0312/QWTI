import pandas as pd, numpy as np
from pathlib import Path

events_p = Path("data/events_calendar.csv")
sig_p    = Path("warehouse/signals_hourly_exp3.csv")
price_dir= Path("capital_wti_downloader/output")
out_p    = Path("warehouse/event_impact_report.csv"); out_p.parent.mkdir(parents=True, exist_ok=True)

# 讀事件
if not events_p.exists():
    pd.DataFrame().to_csv(out_p, index=False); print("[WARN] no events file"); raise SystemExit(0)
ev = pd.read_csv(events_p)
if "ts" not in ev.columns:
    pd.DataFrame().to_csv(out_p, index=False); print("[WARN] events has no ts"); raise SystemExit(0)
ev["ts"] = pd.to_datetime(ev["ts"], utc=True, errors="coerce").dt.floor("h")
ev = ev.dropna(subset=["ts"]).sort_values("ts")
if ev.empty:
    pd.DataFrame().to_csv(out_p, index=False); print("[WARN] events empty after ts parsing"); raise SystemExit(0)
if "event" not in ev.columns: ev["event"]="EIA"
if "type"  not in ev.columns: ev["type"]="EIA"

# 讀 signals（保底來源）
sig = pd.read_csv(sig_p) if sig_p.exists() else pd.DataFrame(columns=["ts"])
if not sig.empty:
    sig["ts"] = pd.to_datetime(sig["ts"], utc=True, errors="coerce").dt.floor("h")
    sig = sig.dropna(subset=["ts"]).sort_values("ts")

# 讀價格 close（盡力找出 ts/close；若仍找不到，就用 signals 當保底）
def load_price_ret():
    if price_dir.exists():
        cands = sorted(price_dir.glob("OIL_CRUDE_HOUR_*.csv"))
        if cands:
            df = pd.read_csv(cands[-1])
            # 找時間欄
            tcol = None
            for c in df.columns:
                cl = c.lower()
                if cl in ("ts","timestamp","datetime","time","date"):
                    tcol=c; break
            # 找收盤欄
            ccol = None
            for c in df.columns:
                cl = c.lower()
                if cl in ("close","price","c"):
                    ccol=c; break
            if tcol is not None and ccol is not None:
                df["ts"]=pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.floor("h")
                df["close"]=pd.to_numeric(df[ccol], errors="coerce")
                df = df[["ts","close"]].dropna().drop_duplicates().sort_values("ts")
                if not df.empty:
                    df["ret_1h"]=df["close"].pct_change()
                    df["ret_3h"]=df["close"].pct_change(3)
                    df["ret_24h"]=df["close"].pct_change(24)
                    return df
    return pd.DataFrame(columns=["ts"])

price = load_price_ret()

# 若沒有 price，就嘗試用 signals 當保底（以 pnl/w_t 的號向作 proxy）
fallback = False
if price.empty:
    if not sig.empty and "pnl_1h" in sig.columns and "w_t" in sig.columns:
        s = sig[["ts","pnl_1h","w_t"]].copy()
        s["ret_1h"] = np.sign(s["pnl_1h"].fillna(0)) * (np.sign(s["w_t"].fillna(0))!=0)
        s = s.set_index("ts").sort_index()
        price = s[["ret_1h"]].copy()
        price["ret_3h"]  = price["ret_1h"].rolling(3 , min_periods=1).sum()
        price["ret_24h"] = price["ret_1h"].rolling(24, min_periods=1).sum()
        price = price.reset_index()
        fallback = True
    else:
        price = pd.DataFrame(columns=["ts"])  # 仍無法保底

base = ev[["ts","event","type"]].copy()
df = base.merge(price, on="ts", how="left")
if not sig.empty:
    keep = [c for c in ("ts","w_t","pnl_1h","in_event","threshold") if c in sig.columns]
    if keep: df = df.merge(sig[keep], on="ts", how="left")

for col in ("ret_1h","ret_3h","ret_24h"):
    if col in price.columns:
        x = price[["ts",col]].rename(columns={col:f"after_{col}"})
        df = df.merge(x, on="ts", how="left")

for c in ("after_ret_1h","after_ret_3h","after_ret_24h"):
    if c in df.columns:
        df[f"hit_{c}"] = (pd.to_numeric(df[c], errors="coerce")>0).astype(float)

df["year"]=df["ts"].dt.year
agg={}
for c in ("after_ret_1h","after_ret_3h","after_ret_24h"):
    if c in df.columns: agg[c]="mean"
for c in ("hit_after_ret_1h","hit_after_ret_3h","hit_after_ret_24h"):
    if c in df.columns: agg[c]="mean"
if "pnl_1h" in df.columns: agg["pnl_1h"]="mean"
summary = df.groupby(["year","type"], dropna=False).agg(agg).reset_index() if agg else pd.DataFrame()

out = pd.concat([
    pd.DataFrame({"section":["per_event_detail"]}).assign(**{c:np.nan for c in df.columns}),
    df,
    pd.DataFrame({"section":["yearly_summary"]}).assign(**{c:np.nan for c in summary.columns}) if not summary.empty else pd.DataFrame(),
    summary
], ignore_index=True)
out.to_csv(out_p, index=False)
flag = " (price=fallback_by_signals)" if fallback else ""
print(f"[WROTE] warehouse/event_impact_report.csv rows={len(out)}{flag}")
