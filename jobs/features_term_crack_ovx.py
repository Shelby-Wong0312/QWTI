from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

DECAY_MAP = {"EIA": 12.0, "OPEC": 36.0, "GEOPOLITICAL": 72.0}
TOL = 1e-12


def robust_zscore(series: pd.Series) -> pd.Series:
    ser = series.astype(float)
    med = ser.median()
    if pd.isna(med):
        return pd.Series(np.nan, index=series.index, dtype=float)
    mad = (ser - med).abs().median()
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        alt = ser.std(ddof=0)
        scale = alt if np.isfinite(alt) and alt != 0 else 1.0
    return (ser - med) / scale


def ensure_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True)
    return ts.dt.tz_convert("UTC")


def normalize_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx.tz_convert("UTC").tz_localize(None)


def load_prices() -> pd.DataFrame:
    price_files = sorted(Path("capital_wti_downloader/output").glob("*_HOUR_*.csv"))
    if not price_files:
        raise SystemExit("æ‰¾ä¸åˆ°åƒ¹æ ¼ CSVï¼ˆcapital_wti_downloader/output/*_HOUR_*.csvï¼‰")
    frames = [pd.read_csv(path, parse_dates=["snapshotTimeUTC"]) for path in price_files]
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.drop_duplicates(subset=["snapshotTimeUTC"]).sort_values("snapshotTimeUTC")
    prices = prices.rename(columns={"snapshotTimeUTC": "ts"})
    prices["ts"] = ensure_utc(prices["ts"])
    prices["mid_close"] = (prices["close_bid"] + prices["close_ask"]) / 2.0
    prices["ret_1h"] = np.log(prices["mid_close"]).diff()
    prices = prices[["ts", "ret_1h"]].set_index("ts")
    prices.index = normalize_index(prices.index)
    return prices


def load_gdelt() -> pd.DataFrame:
    gdelt_csv = Path("data/gdelt_hourly.csv")
    if not gdelt_csv.exists():
        raise SystemExit("æ‰¾ä¸åˆ° data/gdelt_hourly.csv")
    gd = pd.read_csv(gdelt_csv, parse_dates=["ts"]).sort_values("ts")
    gd["ts"] = ensure_utc(gd["ts"])
    gd = gd.set_index("ts")
    gd.index = normalize_index(gd.index)
    gd = gd[["art_cnt", "tone_avg", "tone_pos_ratio"]]
    gd["gdelt_art_cnt_z"] = robust_zscore(gd["art_cnt"])
    gd = gd.rename(
        columns={
            "tone_avg": "gdelt_tone_avg",
            "tone_pos_ratio": "gdelt_tone_pos_ratio",
        }
    )
    return gd[["gdelt_art_cnt_z", "gdelt_tone_pos_ratio", "gdelt_tone_avg"]]


def load_term_crack_ovx() -> Tuple[pd.DataFrame, pd.DataFrame]:
    term_csv = Path("data/term_crack_ovx_hourly.csv")
    if not term_csv.exists():
        raise SystemExit("æ‰¾ä¸åˆ° data/term_crack_ovx_hourly.csv")
    term = pd.read_csv(term_csv); term["ts"]=pd.to_datetime(term["ts"], utc=True, errors="coerce"); term=term.dropna(subset=["ts"]).sort_values("ts")
    term["ts"] = ensure_utc(term["ts"])
    term = term.set_index("ts")
    term.index = normalize_index(term.index)

    features = pd.DataFrame(index=term.index)
    features["k_t"] = robust_zscore(term["cl1_cl2"])
    features["crack_rb_z"] = robust_zscore(term["crack_rb"])
    features["crack_ho_z"] = robust_zscore(term["crack_ho"])

    ovx = term["ovx"].astype(float)
    ovx_valid = ovx.dropna()
    ovx_q = pd.Series(np.nan, index=ovx.index, dtype=float)
    if len(ovx_valid) > 1:
        ranks = ovx_valid.rank(method="average") - 1.0
        denom = len(ovx_valid) - 1.0
        ovx_q.update((ranks / denom).clip(0.0, 1.0))
    elif len(ovx_valid) == 1:
        ovx_q.loc[ovx_valid.index] = 0.5
    features["ovx_q"] = ovx_q

    term_raw = term[["cl1_cl2", "crack_rb", "crack_ho", "ovx"]].copy()
    return features, term_raw


def load_events() -> Optional[pd.DataFrame]:
    events_csv = Path("data/events_calendar.csv")
    if not events_csv.exists():
        return None
    events = pd.read_csv(events_csv)
    if "event_time_utc" not in events.columns:
        return None
    events["event_time_utc"] = pd.to_datetime(events["event_time_utc"], utc=True, errors="coerce")
    events = events.dropna(subset=["event_time_utc"])
    if events.empty:
        return None
    events["event_time_utc"] = events["event_time_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    return events


def build_event_strength(full_index: pd.DatetimeIndex, events: Optional[pd.DataFrame]) -> pd.Series:
    strength = np.zeros(len(full_index), dtype=float)
    if events is None or full_index.empty:
        return pd.Series(strength, index=full_index)
    base = full_index.values.astype("datetime64[ns]")
    for row in events.itertuples(index=False):
        t0 = getattr(row, "event_time_utc", None)
        if not isinstance(t0, datetime):
            continue
        event_type = str(getattr(row, "event_type", "")).upper()
        decay = DECAY_MAP.get(event_type, 12.0)
        weight = getattr(row, "weight", 1.0)
        try:
            w = float(weight)
        except (TypeError, ValueError):
            w = 1.0
        t0v = np.datetime64(t0)
        idx = base.searchsorted(t0v, side="left")
        if idx >= len(base):
            continue
        dt_hours = (base[idx:] - t0v) / np.timedelta64(1, "h")
        strength[idx:] += w * np.exp(-dt_hours / decay)
    return pd.Series(strength, index=full_index)


def compute_lambda(df: pd.DataFrame) -> pd.Series:
    ret = df["ret_1h"].fillna(0.0)
    rv = ret.rolling(window=24, min_periods=1).std(ddof=0)
    target = rv.rolling(window=240, min_periods=1).median()
    target = target.fillna(rv.median())
    rv_values = rv.to_numpy()
    target_values = target.to_numpy()
    lam = np.divide(target_values, rv_values, out=np.zeros_like(target_values), where=rv_values > 0)
    lam = np.clip(lam, 0.0, 1.0)
    return pd.Series(lam, index=df.index, dtype=float)


def main() -> None:
    prices = load_prices()
    gdelt = load_gdelt()
    term_features, term_raw = load_term_crack_ovx()
    events = load_events()

    base_index = prices.index.intersection(term_features.index).sort_values()
    if base_index.empty:
        raise SystemExit("æ²’æœ‰è¶³å¤ çš„è³‡æ–™å»ºç«‹æŒ‡æ¨™")

    df = pd.DataFrame(index=base_index)
    df["ret_1h"] = prices["ret_1h"].reindex(base_index)
    df = df.join(gdelt.reindex(base_index), how="left")
    df = df.join(term_features.reindex(base_index), how="left")
    df["event_strength"] = build_event_strength(base_index, events)

    feature_cols = [
        "event_strength",
        "gdelt_art_cnt_z",
        "gdelt_tone_pos_ratio",
        "gdelt_tone_avg",
        "k_t",
        "crack_rb_z",
        "crack_ho_z",
        "ovx_q",
    ]
    df[feature_cols] = df[feature_cols].ffill().bfill()

    lam = compute_lambda(df)
    is_weekend = pd.Index(df.index).weekday >= 5
    lam = lam.where(~is_weekend, lam * 0.5)
    ret_zero = df["ret_1h"].abs().fillna(0.0) < TOL
    lam = lam.where(~ret_zero, lam * 0.7)
    df["lambda"] = lam

    beta0 = 0.0
    beta1 = 0.9
    beta2 = 0.3
    beta3 = 0.25
    beta4 = 0.15
    beta5 = 0.4
    beta6 = 0.35
    beta7 = 0.5

    cl_ratio = float((term_raw["cl1_cl2"].reindex(base_index).fillna(0.0).abs() > TOL).mean())
    rb_ratio = float((term_raw["crack_rb"].reindex(base_index).fillna(0.0).abs() > TOL).mean())
    ho_ratio = float((term_raw["crack_ho"].reindex(base_index).fillna(0.0).abs() > TOL).mean())

    cl_disabled = False
    crack_disabled = False
    if cl_ratio < 0.2:
        beta5 = 0.0
        cl_disabled = True
    if rb_ratio < 0.2 and ho_ratio < 0.2:
        beta6 = 0.0
        crack_disabled = True

    crack_combo = 0.5 * df["crack_rb_z"].fillna(0.0) + 0.5 * df["crack_ho_z"].fillna(0.0)
    base = (
        beta0
        + beta1 * df["event_strength"].fillna(0.0)
        + beta2 * df["gdelt_art_cnt_z"].fillna(0.0)
        + beta3 * df["gdelt_tone_pos_ratio"].fillna(0.0)
        + beta4 * df["gdelt_tone_avg"].fillna(0.0)
        + beta5 * df["k_t"].fillna(0.0)
        + beta6 * crack_combo
        - beta7 * df["ovx_q"].fillna(0.0)
    )
    w_t = np.tanh(base) * df["lambda"].fillna(0.0)

    threshold = np.nanpercentile(df["event_strength"].to_numpy(), 70)
    recent_max = df["event_strength"].rolling(window=5, min_periods=1).max()
    amp = np.where(recent_max > threshold, 1.3, 1.0)
    df["w_t"] = np.clip(w_t * amp, -1.0, 1.0)

    pnl_1h = (df["w_t"] * df["ret_1h"].shift(-1)).fillna(0.0)
    equity_1h = (1.0 + pnl_1h).cumprod()

    pnl_mean = pnl_1h.mean()
    pnl_std = pnl_1h.std()
    sharpe_hourly = (pnl_mean / pnl_std) * np.sqrt(24 * 252) if pnl_std and pnl_std > 0 else np.nan

    mask_22 = df.index.hour == 22
    next_ret_same_hour = df["ret_1h"].shift(-24)
    w_22 = df.loc[mask_22, "w_t"]
    r_22 = next_ret_same_hour.loc[mask_22]
    valid_22 = r_22.notna()
    next_day_hit = (
        (np.sign(w_22[valid_22]) == np.sign(r_22[valid_22])).astype(float).mean() if valid_22.any() else np.nan
    )

    future_ret = df["ret_1h"].shift(-1)
    valid_hit = future_ret.notna()
    hit_hourly = (
        (np.sign(df.loc[valid_hit, "w_t"]) == np.sign(future_ret[valid_hit])).astype(float).mean()
        if valid_hit.any()
        else np.nan
    )

    Path("warehouse").mkdir(parents=True, exist_ok=True)
    df_features = df[
        [
            "event_strength",
            "gdelt_art_cnt_z",
            "gdelt_tone_pos_ratio",
            "gdelt_tone_avg",
            "k_t",
            "crack_rb_z",
            "crack_ho_z",
            "ovx_q",
            "lambda",
            "w_t",
        ]
    ].reset_index(names="ts")
    df_features.to_csv("warehouse/features_hourly_v2.csv", index=False)

    df_signals = pd.DataFrame(
        {
            "ts": df.index,
            "pnl_1h": pnl_1h.values,
            "equity_1h": equity_1h.values,
            "w_t": df["w_t"].values,
        }
    )
    df_signals.to_csv("warehouse/signals_hourly_v2.csv", index=False)

    def fmt(val: float) -> str:
        return "nan" if pd.isna(val) else f"{val:.3f}"

    print(
        f"rows={len(df)}  Hit(1h)={fmt(hit_hourly)}  Sharpe(hourly)={fmt(sharpe_hourly)}  Next-day Hit={fmt(next_day_hit)}"
    )
    if cl_disabled:
        print("INFO: Î²5 disabled due to insufficient cl1_cl2 signal")
    if crack_disabled:
        print("INFO: Î²6 disabled due to insufficient crack signals")


if __name__ == "__main__":
    main()




