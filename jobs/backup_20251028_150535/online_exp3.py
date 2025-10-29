import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ARM_NAMES = ("news", "term", "crack")
WINDOW_HOURS_30D = 24 * 30
WINDOW_HOURS_180D = 24 * 180
EMA_ALPHA = 0.2
TURNOVER_COST = 2e-5
BASE_OUTPUT_PATH = Path("warehouse/signals_hourly_exp3.csv")
PRICE_GLOB = Path("capital_wti_downloader/output/OIL_CRUDE_HOUR_*.csv")


@dataclass
class Exp3State:
    weights: np.ndarray
    prev_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic EXP3-style online weighting for hourly signals")
    parser.add_argument("--since", type=str, help="Recompute from this UTC date (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="Recompute for the entire available history")
    parser.add_argument("--base-gamma", type=float, default=0.07, help="Base gamma parameter for EXP3 (default: 0.07)")
    parser.add_argument(
        "--write-mode",
        type=str,
        choices=("overwrite", "append"),
        default="overwrite",
        help="Output write mode (overwrite | append)",
    )
    return parser.parse_args()


def find_latest_price_file() -> Path:
    files = sorted(PRICE_GLOB.parent.glob(PRICE_GLOB.name))
    if not files:
        raise SystemExit("找不到資料檔：capital_wti_downloader/output/OIL_CRUDE_HOUR_*.csv")
    return files[-1]


def parse_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts


def load_price_returns(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    time_col = None
    for candidate in ("snapshotTimeUTC", "snapshotTime", "ts", "time"):
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        raise SystemExit(f"價格檔案缺少時間欄位: {path}")
    df["ts"] = parse_ts(df[time_col])
    df = df.dropna(subset=["ts"])
    price_col = None
    for candidate in ("close", "mid_close", "CLOSE"):
        if candidate in df.columns:
            price_col = candidate
            break
    if price_col is None:
        if {"close_bid", "close_ask"}.issubset(df.columns):
            df["close_mid_tmp"] = (df["close_bid"].astype(float) + df["close_ask"].astype(float)) / 2.0
            price_col = "close_mid_tmp"
        else:
            raise SystemExit(f"價格檔案缺少 close 欄位: {path}")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    prices = df.set_index("ts")[price_col]
    ret = prices.pct_change().fillna(0.0)
    return ret


def load_indexed_csv(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"找不到資料檔：{path}")
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise SystemExit(f"{path} 缺少 ts 欄位")
    df["ts"] = parse_ts(df["ts"])
    df = df.dropna(subset=["ts"]).drop_duplicates(subset=["ts"]).sort_values("ts")
    df = df.set_index("ts")
    if columns:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            for col in missing:
                df[col] = np.nan
        return df[list(columns)]
    return df


def load_events(path: Path) -> List[Tuple[pd.Timestamp, str]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty:
        return []
    time_col = None
    for candidate in (
        "event_time_utc",
        "event_time",
        "time_utc",
        "datetime_utc",
        "datetime",
        "ts",
        "time",
    ):
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        return []
    df[time_col] = parse_ts(df[time_col])
    df = df.dropna(subset=[time_col])
    if df.empty:
        return []
    type_col = None
    for candidate in ("event_type", "type", "event", "category", "name", "description"):
        if candidate in df.columns:
            type_col = candidate
            break
    if type_col is None:
        df["_event_type"] = ""
        type_col = "_event_type"
    df[type_col] = df[type_col].fillna("").astype(str).str.upper()
    events: List[Tuple[pd.Timestamp, str]] = []
    for ts, event_type in zip(df[time_col], df[type_col]):
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        events.append((ts, event_type))
    return events


def build_event_flag(index: pd.DatetimeIndex, events: Sequence[Tuple[pd.Timestamp, str]]) -> pd.Series:
    if len(index) == 0:
        return pd.Series(dtype=bool)
    flag = pd.Series(False, index=index)
    for event_time, event_type in events:
        if "EIA" in event_type:
            window = pd.Timedelta(hours=3)
        elif "OPEC" in event_type:
            window = pd.Timedelta(hours=6)
        else:
            continue
        event_naive = event_time.tz_convert("UTC")
        mask = (index >= event_naive - window) & (index <= event_naive + window)
        if mask.any():
            flag.loc[mask] = True
    return flag


def zscore_clip(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or std < 1e-9:
        return pd.Series(0.0, index=series.index)
    z = (values - mean) / std
    return z.clip(-3.0, 3.0)


def fill_arm_nan(series: pd.Series) -> pd.Series:
    median = series.median()
    if pd.isna(median):
        median = 0.0
    return series.replace([np.inf, -np.inf], np.nan).fillna(median).fillna(0.0)


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    signals = pd.DataFrame(index=df.index, columns=ARM_NAMES, dtype=float)
    tone_z = zscore_clip(df["tone_avg"])
    news_signal = tone_z * (df["tone_pos_ratio"].fillna(0.5) - 0.5)
    signals["news"] = fill_arm_nan(news_signal)
    term_signal = -zscore_clip(df["cl1_cl2"])
    signals["term"] = fill_arm_nan(term_signal)
    crack_base = 0.5 * (df["crack_rb"] + df["crack_ho"])
    crack_signal = zscore_clip(crack_base)
    signals["crack"] = fill_arm_nan(crack_signal)
    return signals.fillna(0.0)


def compute_vol_metrics(ret_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    rolling_std = ret_series.rolling(WINDOW_HOURS_30D, min_periods=24).std(ddof=0)
    annualize = math.sqrt(24 * 365)
    rv30 = rolling_std * annualize
    rv_anchor = rv30.rolling(WINDOW_HOURS_180D, min_periods=WINDOW_HOURS_30D).median()
    return rv30, rv_anchor


def compute_gamma(rv30: float, rv_anchor: float, base_gamma: float) -> float:
    if pd.isna(rv30) or rv30 <= 1e-9:
        return base_gamma
    anchor = rv_anchor if not pd.isna(rv_anchor) and rv_anchor > 0 else rv30
    gamma = base_gamma * (anchor / rv30)
    return float(np.clip(gamma, 0.05, base_gamma))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    cummax = equity.cummax()
    drawdown = equity - cummax
    return float(drawdown.min())


def apply_write_mode(result: pd.DataFrame, mode: str, output_path: Path, since: Optional[pd.Timestamp]) -> None:
    if mode == "append" and output_path.exists():
        existing = pd.read_csv(output_path)
        if existing.empty or "ts" not in existing.columns:
            combined = result
        else:
            existing["ts"] = parse_ts(existing["ts"])
            existing = existing.dropna(subset=["ts"])
            if since is not None:
                existing = existing[existing["ts"] < since]
            combined = pd.concat(
                [existing, result],
                ignore_index=True,
            )
    else:
        combined = result
    combined = combined.drop_duplicates(subset=["ts"]).sort_values("ts")
    output_df = combined.copy()
    output_df["ts"] = output_df["ts"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    price_path = find_latest_price_file()
    ret_series = load_price_returns(price_path)

    term_df = load_indexed_csv(
        Path("data/term_crack_ovx_hourly.csv"),
        columns=("cl1_cl2", "crack_rb", "crack_ho", "ovx"),
    )
    gdelt_df = load_indexed_csv(
        Path("data/gdelt_hourly.csv"),
        columns=("art_cnt", "tone_avg", "tone_pos_ratio", "tone_avg_se", "low_confidence"),
    )

    common_index = ret_series.index.intersection(term_df.index).intersection(gdelt_df.index)
    common_index = common_index.sort_values()
    if len(common_index) == 0:
        raise SystemExit("資料時間範圍無交集，無法計算")

    if args.all:
        since_ts = None
    else:
        since_ts = None
        if args.since:
            since_ts = pd.Timestamp(args.since, tz="UTC")
        if since_ts is not None:
            common_index = common_index[common_index >= since_ts]
        if len(common_index) == 0:
            raise SystemExit("指定區間內無資料")

    df = pd.DataFrame(index=common_index)
    df["ret_1h"] = ret_series.reindex(common_index).fillna(0.0)
    df = df.join(term_df.reindex(common_index))
    df = df.join(gdelt_df.reindex(common_index))
    df["tone_pos_ratio"] = df["tone_pos_ratio"].fillna(0.5)
    df = df.dropna(subset=["ret_1h"])

    signals = compute_signals(df)
    events = load_events(Path("data/events_calendar.csv"))
    in_event = build_event_flag(df.index, events)

    rv30, rv_anchor = compute_vol_metrics(df["ret_1h"].fillna(0.0))
    rv30 = rv30.reindex(df.index)
    rv_anchor = rv_anchor.reindex(df.index)

    state = Exp3State(weights=np.ones(len(ARM_NAMES), dtype=float), prev_weight=0.0)
    gamma_history: List[float] = []
    threshold_history: List[float] = []
    w_raw_history: List[float] = []
    w_history: List[float] = []
    pnl_history: List[float] = []
    turnover_history: List[float] = []
    rv30_history: List[float] = []
    abs_score_history: List[float] = []

    base_gamma = args.base_gamma
    arm_matrix = signals[ARM_NAMES].to_numpy()
    ret_values = df["ret_1h"].to_numpy()
    in_event_values = in_event.to_numpy()

    for idx, ts in enumerate(df.index):
        arm_vec = arm_matrix[idx]
        rv30_val = float(rv30.iloc[idx]) if not pd.isna(rv30.iloc[idx]) else float("nan")
        rv_anchor_val = float(rv_anchor.iloc[idx]) if not pd.isna(rv_anchor.iloc[idx]) else float("nan")
        gamma_t = compute_gamma(rv30_val, rv_anchor_val, base_gamma)
        weights = state.weights
        weight_sum = weights.sum()
        if weight_sum <= 0:
            weights = np.ones_like(weights)
            weight_sum = weights.sum()
        probs = (1 - gamma_t) * (weights / weight_sum) + (gamma_t / len(weights))
        probs = probs / probs.sum()
        if in_event_values[idx]:
            boosted = probs.copy()
            boosted[0] *= 1.3
            probs = boosted / boosted.sum()
        score_pre = float(np.dot(probs, arm_vec))
        abs_score_history.append(abs(score_pre))
        recent_abs = abs_score_history[-WINDOW_HOURS_30D:]
        quantile_level = 0.5 if in_event_values[idx] else 0.45
        if recent_abs:
            q_val = float(np.quantile(recent_abs, quantile_level))
        else:
            q_val = 0.0
        threshold = max(q_val, 0.05)
        w_raw = float(np.tanh(score_pre))
        gate_output = w_raw if abs(score_pre) >= threshold else 0.0
        smoothed_weight = EMA_ALPHA * gate_output + (1.0 - EMA_ALPHA) * state.prev_weight
        turnover = abs(smoothed_weight - state.prev_weight)
        cost = TURNOVER_COST * turnover
        pnl = state.prev_weight * ret_values[idx] - cost

        reward = np.clip(ret_values[idx] * arm_vec, -0.05, 0.05)
        update = np.exp(gamma_t * reward)
        weights = weights * update
        if not np.isfinite(weights).all():
            weights = np.ones_like(weights)
        weights = np.clip(weights, 1e-8, None)

        state.weights = weights
        state.prev_weight = smoothed_weight

        gamma_history.append(gamma_t)
        threshold_history.append(threshold)
        w_raw_history.append(w_raw)
        w_history.append(smoothed_weight)
        pnl_history.append(pnl)
        turnover_history.append(turnover)
        rv30_history.append(rv30_val)

    pnl_series = pd.Series(pnl_history, index=df.index)
    equity_series = pnl_series.cumsum()
    turnover_series = pd.Series(turnover_history, index=df.index)

    result = pd.DataFrame(
        {
            "ts": df.index,
            "w_raw": w_raw_history,
            "w_t": w_history,
            "pnl_1h": pnl_history,
            "equity_1h": equity_series.values,
            "turnover": turnover_series.values,
            "rv30": rv30_history,
            "gamma_t": gamma_history,
            "threshold": threshold_history,
            "in_event": in_event.values,
        }
    )

    apply_write_mode(result.copy(), args.write_mode, BASE_OUTPUT_PATH, since_ts if not args.all else None)

    recent = result.tail(5)[["ts", "gamma_t", "threshold", "in_event"]]
    print("Recent gamma/threshold/in_event sample (last 5 rows):")
    printable = recent.copy()
    printable["ts"] = printable["ts"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M")
    print(printable.to_string(index=False))

    hit_weights = np.concatenate(([0.0], result["w_t"].values[:-1]))
    hit_ratio = float(((hit_weights * ret_values) > 0).mean())
    pnl_std = pnl_series.std(ddof=0)
    sharpe_hourly = float(pnl_series.mean() / (pnl_std + 1e-12))
    sharpe_hourly *= math.sqrt(24 * 365)
    max_dd = max_drawdown(equity_series)
    in_event_ratio = float(result["in_event"].astype(float).mean())

    turnover_mean = float(turnover_series.mean())
    turnover_7d = float(turnover_series.tail(24 * 7).mean())

    print(
        (
            f"in_event ratio={in_event_ratio:.3f}  "
            f"Sharpe(hourly)={sharpe_hourly:.3f}  "
            f"Hit(1h)={hit_ratio:.3f}  "
            f"MaxDD={max_dd:.4f}"
        )
    )
    print(
        f"Turnover |Δw_t| mean (7D)={turnover_7d:.6f}  overall={turnover_mean:.6f}",
    )


if __name__ == "__main__":
    main()
