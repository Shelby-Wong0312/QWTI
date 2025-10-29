print("__WTI_TAG__", "make_term_crack_ovx_from_capital.py", "hotfix3")
import os, logging, pandas as pd, requests
from datetime import datetime, timedelta, timezone

UTC = timezone.utc

def iso_noz(dt): 
    return dt.replace(tzinfo=UTC).isoformat().replace('+00:00','')

class CapClient:
    def __init__(self):
        base = os.environ.get("CAPITAL_BASE_URL")
        if not base: raise RuntimeError("CAPITAL_BASE_URL not set")
        self.base = base.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Accept":"application/json",
            "CST": os.environ.get("CAPITAL_CST",""),
            "X-SECURITY-TOKEN": os.environ.get("CAPITAL_XST","")
        })
        if os.getenv("CAPITAL_API_KEY"):
            self.session.headers["X-CAP-API-KEY"] = os.getenv("CAPITAL_API_KEY")
        self.auth_failed = False

    def get(self, path, params=None):
        r = self.session.get(f"{self.base}{path}", params=params, timeout=60)
        if r.status_code == 401:
            self.auth_failed = True
            logging.warning("401 on %s, fallback to offline", path)
            return None
        if 200 <= r.status_code < 300:
            return r
        logging.warning("GET %s failed: %s", path, r.status_code)
        return None

    def prices_hourly(self, epic, start_dt, end_dt):
        # 第一槍做可用性探針：最近 24h
        probe_from = end_dt - timedelta(hours=24)
        if not self.get(f"/api/v1/prices/{epic}", {"resolution": os.getenv("CAPITAL_RESOLUTION","HOUR"),"from":iso_noz(probe_from),"to":iso_noz(end_dt)}):
            return pd.DataFrame(columns=["ts","close"])  # 無權或無資料  空
        # 正式分段（<=90天一段）
        out, cur = [], start_dt
        while cur < end_dt and not self.auth_failed:
            to = min(cur + timedelta(days=90), end_dt)
            r = self.get(f"/api/v1/prices/{epic}", {"resolution": os.getenv("CAPITAL_RESOLUTION","HOUR"),"from":iso_noz(cur),"to":iso_noz(to)})
            if r is None:
                cur = to; continue
            try:
                arr = (r.json().get("prices") or r.json().get("candles") or [])
                for it in arr:
                    t = it.get("snapshotTimeUTC") or it.get("snapshotTime") or it.get("time")
                    cp = it.get("closePrice",{}).get("mid") if isinstance(it.get("closePrice"),dict) else it.get("closePrice")
                    if t is None or cp is None: continue
                    ts = pd.to_datetime(t, utc=True).floor("h")
                    out.append((ts, float(cp)))
            except Exception as e:
                logging.warning("parse error: %s", e)
            cur = to
        if not out: return pd.DataFrame(columns=["ts","close"])
        df = pd.DataFrame(out, columns=["ts","close"]).drop_duplicates("ts").sort_values("ts")
        return df

def load_ovx():
    p = os.path.join("data","term_crack_ovx_hourly.csv")
    if not os.path.exists(p): return pd.DataFrame(columns=["ts","ovx"])
    try:
        df = pd.read_csv(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df[["ts","ovx"]]
    except Exception:
        return pd.DataFrame(columns=["ts","ovx"])

def write_zeros(start_dt, end_dt, msg):
    logging.warning(msg)
    idx = pd.date_range(start=start_dt, end=end_dt, freq="h", tz="UTC")
    out = pd.DataFrame({"ts":idx, "cl1_cl2":0.0, "crack_rb":0.0, "crack_ho":0.0})
    ovx = load_ovx()
    out = out.merge(ovx, on="ts", how="left")
    os.makedirs("data", exist_ok=True)
    out.to_csv(os.path.join("data","term_crack_ovx_hourly.csv"), index=False)
    print("WROTE data/term_crack_ovx_hourly.csv (zeros)")

def main():
    logging.basicConfig(level=logging.INFO)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-02-19")
    ap.add_argument("--end", default=datetime.now(UTC).date().isoformat())
    args = ap.parse_args()
    start_dt = pd.to_datetime(args.start, utc=True)
    end_dt   = pd.to_datetime(args.end,   utc=True) + pd.Timedelta(hours=1)

    try:
        client = CapClient()
    except Exception as e:
        write_zeros(start_dt, end_dt, f"client init failed: {e}")
        return

    CL1 = os.getenv("CL1_EPIC") or "OIL_CRUDE"
    print(f"CL1: {CL1} [HOTFIX], CL2/RB/HO set 0")
    cl1 = client.prices_hourly(CL1, start_dt, end_dt)
    if cl1.empty:
        write_zeros(start_dt, end_dt, "prices empty or unauthorized")
        return

    idx = pd.date_range(start=cl1["ts"].min(), end=cl1["ts"].max(), freq="h", tz="UTC")
    cl1 = cl1.set_index("ts").reindex(idx).rename_axis("ts").reset_index()
    out = cl1.copy()
    out["cl1_cl2"] = 0.0
    out["crack_rb"] = 0.0
    out["crack_ho"] = 0.0
    out = out[["ts","cl1_cl2","crack_rb","crack_ho"]]
    ovx = load_ovx()
    out = out.merge(ovx, on="ts", how="left")
    os.makedirs("data", exist_ok=True)
    out.to_csv(os.path.join("data","term_crack_ovx_hourly.csv"), index=False)
    print("WROTE data/term_crack_ovx_hourly.csv")

if __name__ == "__main__":
    main()

