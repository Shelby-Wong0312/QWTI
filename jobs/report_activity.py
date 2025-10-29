import pandas as pd, numpy as np
from pathlib import Path
sig_p = Path("warehouse/signals_hourly_exp3.csv")
out_p = Path("warehouse/activity_report.csv"); out_p.parent.mkdir(parents=True, exist_ok=True)
if not sig_p.exists():
    print("[WARN] signals_hourly_exp3.csv 不存在"); pd.DataFrame().to_csv(out_p, index=False); raise SystemExit(0)
sig = pd.read_csv(sig_p)
if "ts" not in sig.columns:
    print("[WARN] signals 無 ts 欄"); pd.DataFrame().to_csv(out_p, index=False); raise SystemExit(0)
sig["ts"]=pd.to_datetime(sig["ts"], utc=True, errors="coerce").dt.floor("h")
sig = sig.dropna(subset=["ts"]).sort_values("ts")
def window_stats(df, days):
    end = df["ts"].max(); start = end - pd.Timedelta(days=days)
    x = df[df["ts"]>=start].copy()
    if x.empty: return {"days":days,"rows":0}
    x["dw"]= x["w_t"].diff().abs() if "w_t" in x.columns else np.nan
    return {
        "days":days, "rows":len(x),
        "wt_zero_share": float((x["w_t"].abs()<1e-6).mean()) if "w_t" in x.columns else np.nan,
        "avg_threshold": float(x["threshold"].mean()) if "threshold" in x.columns else np.nan,
        "avg_turnover" : float(x["dw"].mean()) if "w_t" in x.columns else np.nan,
        "trades_approx": int((x["dw"]>0.01).sum()) if "w_t" in x.columns else 0,
        "pnl_mean_1h"  : float(x["pnl_1h"].mean()) if "pnl_1h" in x.columns else np.nan,
        "pnl_std_1h"   : float(x["pnl_1h"].std())  if "pnl_1h" in x.columns else np.nan,
        "hit_rate_1h"  : float((x["pnl_1h"]>0).mean()) if "pnl_1h" in x.columns else np.nan
    }
report = pd.DataFrame([window_stats(sig,30), window_stats(sig,90)])
report.to_csv(out_p, index=False)
print(report.to_string(index=False))
print("[WROTE] warehouse/activity_report.csv")
