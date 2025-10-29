import os
import pandas as pd
from pathlib import Path

OUTDIR = Path("warehouse")
OUTDIR.mkdir(parents=True, exist_ok=True)

SIG = Path("warehouse/signals_hourly_exp3.csv")

def ensure_utc_naive(s):
    s = pd.to_datetime(s, utc=True, errors="coerce")
    s = s.dropna()
    return s.dt.tz_convert("UTC").dt.tz_localize(None)

def load_signals():
    if not SIG.exists():
        raise SystemExit("signals_hourly_exp3.csv 不存在")
    df = pd.read_csv(SIG)
    df["ts"] = ensure_utc_naive(df["ts"])
    df = df.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
    # 兼容欄位
    if "w_t" not in df.columns and "w" in df.columns:
        df = df.rename(columns={"w":"w_t"})
    return df[["w_t"]]

def determine_reference_time(sig):
    # 取「現在UTC整點」與「資料最後時間」的較小者
    now_utc = pd.Timestamp.now(tz="UTC").floor("h").tz_convert("UTC").tz_localize(None)
    last_ts = sig.index.max()
    return min(now_utc, last_ts)

def make_today_list(sig):
    ref = determine_reference_time(sig)
    today_last = sig.loc[:ref].tail(1).copy()
    if today_last.empty:
        return pd.DataFrame(columns=["ts","w_t","note"])
    out = today_last.reset_index().rename(columns={"ts":"ts"})
    out["note"] = "latest <= ref_utc"
    return out[["ts","w_t","note"]]

def nearest_within(sig, target_ts, tol_min=90):
    # 找 target_tstol_min 分鐘內的最近一筆
    lo = target_ts - pd.Timedelta(minutes=tol_min)
    hi = target_ts + pd.Timedelta(minutes=tol_min)
    window = sig.loc[lo:hi]
    if window.empty:
        return None
    idx = (__import__("numpy").abs(window.index - target_ts)).argmin()
    row = window.iloc[[idx]].copy()
    row.index = [target_ts]  # 以目標時間標示
    return row

def make_22utc_list(sig):
    # 每日 22:00 UTC 精確匹配；沒有就找90 分鐘內最近一筆
    days = pd.to_datetime(sig.index.date).unique()
    rows = []
    for d in days:
        t = pd.Timestamp(d) + pd.Timedelta(hours=22)  # tz-naive UTC 時刻
        exact = sig.loc[sig.index == t]
        if not exact.empty:
            rows.append(exact.iloc[[0]])
        else:
            near = nearest_within(sig, t, 90)
            if near is not None:
                rows.append(near)
    if not rows:
        return pd.DataFrame(columns=["ts","w_t"])
    out = pd.concat(rows).reset_index().rename(columns={"index":"ts"})
    return out[["ts","w_t"]]

def main():
    sig = load_signals()
    today = make_today_list(sig)
    daily22 = make_22utc_list(sig)

    today.to_csv(OUTDIR/"trade_list_today.csv", index=False)
    daily22.to_csv(OUTDIR/"trade_list_22utc.csv", index=False)

    print("today_last:", 0 if today.empty else today.iloc[0]["ts"], "w_t=", 0.0 if today.empty else float(today.iloc[0]["w_t"]))
    print("22utc_rows:", len(daily22), "first_ts:", "" if daily22.empty else daily22.iloc[0]["ts"])

if __name__ == "__main__":
    main()



