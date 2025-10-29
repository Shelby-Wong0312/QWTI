import numpy as np
import pandas as pd
from pathlib import Path

# ---------- 路徑與輸入 ----------
price_csv = sorted(Path("capital_wti_downloader/output").glob("*_HOUR_*.csv"))
if not price_csv:
    raise SystemExit("找不到價格 CSV（capital_wti_downloader/output/*_HOUR_*.csv）")
price_csv = price_csv[-1]

events_csv = Path("data/events_calendar.csv")
gdelt_csv  = Path("data/gdelt_hourly.csv")
if not events_csv.exists():
    raise SystemExit("找不到 data/events_calendar.csv")
if not gdelt_csv.exists():
    raise SystemExit("找不到 data/gdelt_hourly.csv")

def ensure_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx.tz_convert("UTC").tz_localize(None)


# ---------- 讀價格 ----------
p = pd.read_csv(price_csv, parse_dates=["snapshotTimeUTC"]).sort_values("snapshotTimeUTC")
p = p.rename(columns={"snapshotTimeUTC": "ts"})
p["mid_close"] = (p["close_bid"] + p["close_ask"]) / 2
p["ret_1h"] = np.log(p["mid_close"]).diff()
p = p[["ts", "ret_1h"]].set_index("ts")
p.index = ensure_utc_index(p.index)

# ---------- 讀 GDELT（逐小時聚合） ----------
gd = pd.read_csv(gdelt_csv, parse_dates=["ts"]).sort_values("ts").set_index("ts")
gd.index = ensure_utc_index(gd.index)
gd = gd[["art_cnt", "tone_avg", "tone_pos_ratio"]]

# robust z-score for art_cnt
med = gd["art_cnt"].median()
mad = (gd["art_cnt"] - med).abs().median()
scale = 1.4826 * mad if mad and not np.isnan(mad) else (gd["art_cnt"].std() or 1.0)
gd["gdelt_art_cnt_z"] = (gd["art_cnt"] - med) / (scale if scale != 0 else 1.0)

gd = gd.rename(columns={
    "tone_pos_ratio":"gdelt_tone_pos_ratio",
    "tone_avg":"gdelt_tone_avg",
})

# ---------- 讀事件並做指數衰減 ----------
ev = pd.read_csv(events_csv, parse_dates=["event_time_utc"]).sort_values("event_time_utc")
full_index = p.index.union(gd.index).drop_duplicates().sort_values()
event_strength = pd.Series(0.0, index=full_index)

tau = {"GEOPOLITICAL":72, "OPEC":36, "EIA":12}
for _, r in ev.iterrows():
    t0 = pd.Timestamp(r["event_time_utc"], tz=None)
    w  = float(r.get("weight",1.0))
    et = str(r.get("event_type","")).upper()
    decay = tau.get(et, 12)
    # 對 t >= t0 做指數衰減
    idx = full_index.searchsorted(t0, side="left")
    if idx < len(full_index):
        dt_h = (full_index[idx:] - t0).total_seconds()/3600.0
        event_strength.iloc[idx:] = event_strength.iloc[idx:] + w * np.exp(-dt_h/decay)

# ---------- 對齊到共同逐小時索引 ----------
df = pd.DataFrame(index=full_index)
df = df.join(p["ret_1h"], how="left")
df = df.join(gd[["gdelt_art_cnt_z","gdelt_tone_pos_ratio","gdelt_tone_avg"]], how="left")
df["event_strength"] = event_strength

# 缺值處理：前向/後向填
df[["gdelt_art_cnt_z","gdelt_tone_pos_ratio","gdelt_tone_avg"]] = (
    df[["gdelt_art_cnt_z","gdelt_tone_pos_ratio","gdelt_tone_avg"]]
    .ffill().bfill()
)

# ---------- 波動縮放 lambda ----------
rv = df["ret_1h"].rolling(24).std().bfill()
target = rv.rolling(240).median().fillna(rv.median())
lam = (target/rv).clip(upper=1.0).fillna(0.0)
df["lambda"] = lam

# ---------- 權重（升級小配方） ----------
beta0, beta1, beta2, beta3, beta4 = 0.0, 0.8, 0.3, 0.25, 0.15
base = (
    beta0
    + beta1*df["event_strength"].fillna(0.0)
    + beta2*df["gdelt_art_cnt_z"].fillna(0.0)
    + beta3*df["gdelt_tone_pos_ratio"].fillna(0.0)
    + beta4*df["gdelt_tone_avg"].fillna(0.0)
)
w_t = np.tanh(base) * df["lambda"].fillna(0.0)

# 不對稱：近 5h 最大 event_strength > 全樣本 70% 分位  放大 1.3
thr = np.nanpercentile(df["event_strength"].values, 70)
neg_mask = df["event_strength"].rolling(5, min_periods=1).max() > thr
w_t = w_t * np.where(neg_mask, 1.3, 1.0)
df["w_t"] = w_t.clip(-1, 1)

# ---------- 指標（小時 & 隔日方向） ----------
pnl_1h = (df["ret_1h"].shift(-1) * df["w_t"]).fillna(0.0)
eq_1h  = (1.0 + pnl_1h).cumprod()

# 小時 Sharpe（粗略）
sharpe_h = (pnl_1h.mean() / (pnl_1h.std() + 1e-12)) * np.sqrt(24*252)

# 方向命中率（小時）
hit_h = (np.sign(df["w_t"]) == np.sign(df["ret_1h"].shift(-1))).astype(float).mean()

# 隔日方向：取每日 22:00 UTC 的 bar
daily_idx = df.index.indexer_between_time("22:00","22:00")
w_d = df["w_t"].iloc[daily_idx]
r_d = df["ret_1h"].iloc[daily_idx]
nextday_hit = (np.sign(w_d) == np.sign(r_d.shift(-1))).astype(float).mean()

# ---------- 輸出 ----------
Path("warehouse").mkdir(parents=True, exist_ok=True)
out_feat = Path("warehouse/features_hourly.csv")
out_sig  = Path("warehouse/signals_hourly.csv")
df_out = df[["event_strength","gdelt_art_cnt_z","gdelt_tone_pos_ratio","gdelt_tone_avg","lambda","w_t"]].copy()
df_out.reset_index(names="ts").to_csv(out_feat, index=False)
pd.DataFrame({"ts":df.index, "pnl_1h":pnl_1h.values, "equity_1h":eq_1h.values, "w_t":df["w_t"].values}).to_csv(out_sig, index=False)

print(f"rows={len(df)}  Hit(1h)={hit_h:.3f}  Sharpe(hourly)={sharpe_h:.3f}  Next-day Hit={nextday_hit:.3f}")
