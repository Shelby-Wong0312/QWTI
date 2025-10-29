import pandas as pd, numpy as np
from pathlib import Path

feat = pd.read_csv("warehouse/features_hourly_v2.csv", parse_dates=["ts"]).set_index("ts")
sig  = pd.read_csv("warehouse/signals_hourly_v2.csv",  parse_dates=["ts"]).set_index("ts")

df = feat.join(sig[["pnl_1h","w_t"]], how="inner", rsuffix="_sig")
df["w_t_sig"] = df["w_t_sig"].fillna(df["w_t"])
df["pnl_1h"]  = df["pnl_1h"].fillna(0.0)

# 係數（和 v2 權重一致）
B = dict(β0=0.0, β1=0.8, β2=0.3, β3=0.25, β4=0.15, β5=0.25, β6=0.25, β7=0.35)

# 重建 base
combo_crack = 0.5*df["crack_rb_z"].fillna(0.0) + 0.5*df["crack_ho_z"].fillna(0.0)
base = (
    B["β0"]
  + B["β1"]*df["event_strength"].fillna(0.0)
  + B["β2"]*df["gdelt_art_cnt_z"].fillna(0.0)
  + B["β3"]*df["gdelt_tone_pos_ratio"].fillna(0.0)
  + B["β4"]*df["gdelt_tone_avg"].fillna(0.0)
  + B["β5"]*df["k_t"].fillna(0.0)
  + B["β6"]*combo_crack
  - B["β7"]*df["ovx_q"].fillna(0.0)
)

# dw/dbase = (1 - tanh(base)^2) * lambda
lam  = df["lambda"].fillna(0.0)
tanhb = np.tanh(base)
sens = (1 - tanhb**2) * lam

# 各特徵貢獻到權重，再分攤到 pnl（線性近似）
terms = {
    "event_strength":   (B["β1"], df["event_strength"].fillna(0.0)),
    "gdelt_art_cnt_z":  (B["β2"], df["gdelt_art_cnt_z"].fillna(0.0)),
    "gdelt_tone_pos":   (B["β3"], df["gdelt_tone_pos_ratio"].fillna(0.0)),
    "gdelt_tone_avg":   (B["β4"], df["gdelt_tone_avg"].fillna(0.0)),
    "k_t":              (B["β5"], df["k_t"].fillna(0.0)),
    "crack_z":          (B["β6"], combo_crack.fillna(0.0)),
    "ovx_q(-)":         (-B["β7"], df["ovx_q"].fillna(0.0)),
}

# 避免除零：用權重的絕對和來分攤
eps = 1e-12
den = (sum(abs(coef*X) for coef, X in terms.values()) + eps)

contrib_rows = []
for name, (coef, X) in terms.items():
    weight_part = sens * (coef * X)
    share = (abs(coef*X) / den)
    contrib = share * df["pnl_1h"]  # 分攤 pnl
    contrib_rows.append((name, contrib))

# 匯總
agg = []
for name, contrib in contrib_rows:
    agg.append([name, float(contrib.sum()), float(contrib.mean())])
contrib_df = pd.DataFrame(agg, columns=["feature","total_contrib","mean_contrib"]).sort_values("total_contrib", ascending=False)

# 最近 50 筆事件（event_strength 大於 0.2）的摘要
recent_ev = df[df["event_strength"]>0.2].tail(50).copy()
# 當下最大貢獻者（用權重份額 share 判定）
shares = {name: (abs(coef*X) / den) for name,(coef,X) in terms.items()}
share_mat = pd.DataFrame({k:v for k,v in shares.items()})
top_feat = share_mat.loc[recent_ev.index].idxmax(axis=1)
recent_tbl = pd.DataFrame({
    "ts": recent_ev.index,
    "w_t": recent_ev["w_t_sig"],
    "pnl_1h": recent_ev["pnl_1h"],
    "top_feature": top_feat.values
})

# 輸出
Path("warehouse").mkdir(exist_ok=True, parents=True)
contrib_df.to_csv("warehouse/contrib_features.csv", index=False)
recent_tbl.to_csv("warehouse/contrib_recent_events.csv", index=False)

# 摘要
top3 = ", ".join(contrib_df.head(3)["feature"].tolist())
print("Top3 features by total_contrib:", top3)
