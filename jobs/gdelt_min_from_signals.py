import pandas as pd
from pathlib import Path
SIG = Path("warehouse/signals_hourly_exp3.csv")
OUT = Path("data/gdelt_hourly.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)
cols = ["ts","art_cnt","tone_avg","tone_pos_ratio"]
if not SIG.exists():
    pd.DataFrame(columns=cols).to_csv(OUT, index=False)
    print("[WROTE]", OUT, "rows=0 (no signals)"); raise SystemExit(0)
s = pd.read_csv(SIG)
if "ts" not in s.columns:
    pd.DataFrame(columns=cols).to_csv(OUT, index=False)
    print("[WROTE]", OUT, "rows=0 (signals no ts)"); raise SystemExit(0)
s["ts"] = pd.to_datetime(s["ts"], utc=True, errors="coerce").dt.floor("h")
s = s.dropna(subset=["ts"]).drop_duplicates(subset=["ts"]).sort_values("ts")
out = pd.DataFrame({"ts": s["ts"].astype(str), "art_cnt": 0, "tone_avg": 0.0, "tone_pos_ratio": 0.0})
out.to_csv(OUT, index=False)
print("[WROTE]", OUT, "rows=", len(out))
