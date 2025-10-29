import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

UTC = dt.timezone.utc
PRICE_GLOB = Path("capital_wti_downloader/output/OIL_CRUDE_HOUR_*.csv")
FEATURES_PATH = Path("warehouse/features_hourly_v2.csv")
TERM_PATH = Path("data/term_crack_ovx_hourly.csv")
GDELT_PATH = Path("data/gdelt_hourly.csv")
SIGNALS_PATH = Path("warehouse/signals_hourly_exp3.csv")
QUALITY_MONITOR_PATH = Path("warehouse/monitor_data_quality.csv")
KPI_MONITOR_PATH = Path("warehouse/monitor_kpi_7d.csv")
RET_THRESHOLD_DIFF = 0.05
RET_MIN_CORR = 0.2
ANNUALIZATION_FACTOR = math.sqrt(24 * 365)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data quality and KPI monitors")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (unused placeholder)")
    return parser.parse_args()


def find_latest_price_file() -> Optional[Path]:
    files = sorted(PRICE_GLOB.parent.glob(PRICE_GLOB.name))
    return files[-1] if files else None


def parse_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts


def load_price_series(path: Path) -> Tuple[pd.DatetimeIndex, pd.Series]:
    df = pd.read_csv(path)
    time_col = None
    for candidate in ("snapshotTimeUTC", "snapshotTime", "ts", "time"):
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        raise SystemExit(f"價格檔案缺少時間欄位: {path}")
    df["ts"] = parse_ts(df[time_col])
    df = df.dropna(subset=["ts"]).drop_duplicates(subset=["ts"]).sort_values("ts")
    price_col = None
    for candidate in ("close", "mid_close"):
        if candidate in df.columns:
            price_col = candidate
            break
    if price_col is None and {"close_bid", "close_ask"}.issubset(df.columns):
        df["mid_close"] = (df["close_bid"].astype(float) + df["close_ask"].astype(float)) / 2.0
        price_col = "mid_close"
    if price_col is None:
        raise SystemExit(f"價格檔案缺少 close 欄位: {path}")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])
    index = df["ts"].dt.tz_convert("UTC")
    prices = df.set_index("ts")[price_col]
    returns = prices.pct_change().fillna(0.0)
    return index, returns


def load_dataset(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["ts"]).set_index(pd.Index([], dtype="datetime64[ns]"))
    if "ts" not in df.columns:
        return None
    df["ts"] = parse_ts(df["ts"])
    df = df.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
    return df


def compute_duplicates(df: Optional[pd.DataFrame]) -> int:
    if df is None or df.empty:
        return 0
    dup_count = int(df.index.duplicated().sum())
    return dup_count


def compute_illegal_values(df: Optional[pd.DataFrame]) -> int:
    if df is None or df.empty:
        return 0
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return 0
    mask = ~np.isfinite(numeric_df.to_numpy())
    return int(mask.sum())


def detect_lookahead(
    df: Optional[pd.DataFrame],
    baseline_returns: pd.Series,
) -> bool:
    if df is None or df.empty:
        return True
    candidate_cols = [
        col
        for col in df.columns
        if any(
            token in col.lower()
            for token in ("ret", "target", "y_", "y")
        )
    ]
    if not candidate_cols:
        return True
    baseline_returns = baseline_returns.sort_index()
    baseline_future = baseline_returns.shift(-1)
    for col in candidate_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            continue
        aligned = series.reindex(baseline_returns.index)
        if aligned.count() < 10:
            continue
        same_corr = aligned.corr(baseline_returns)
        future_corr = aligned.corr(baseline_future)
        if (
            future_corr is not None
            and same_corr is not None
            and not np.isnan(future_corr)
            and not np.isnan(same_corr)
            and future_corr - same_corr > RET_THRESHOLD_DIFF
            and abs(future_corr) > RET_MIN_CORR
        ):
            return False
    return True


def append_row(path: Path, columns: List[str], values: Dict[str, object]) -> None:
    row = pd.DataFrame([values], columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    row.to_csv(path, mode="a", header=write_header, index=False)


def run_data_quality_monitor() -> None:
    price_path = find_latest_price_file()
    if not price_path:
        raise SystemExit("找不到 OIL_CRUDE 小時線資料")
    price_index, baseline_returns = load_price_series(price_path)

    features_df = load_dataset(FEATURES_PATH)
    term_df = load_dataset(TERM_PATH)
    gdelt_df = load_dataset(GDELT_PATH)

    datasets = [features_df, term_df, gdelt_df]
    available = [df is not None and not df.empty for df in datasets]

    if all(available):
        base_set = set(price_index)
        intersect_set = base_set.copy()
        for df in datasets:
            intersect_set &= set(df.index)
        align_rate = len(intersect_set) / len(base_set) if base_set else np.nan
    elif any(available):
        align_rate = np.nan
    else:
        align_rate = np.nan

    miss_rate = np.nan if np.isnan(align_rate) else max(0.0, 1.0 - align_rate)

    dup_bars = sum(compute_duplicates(df) for df in datasets if df is not None)
    illegal_values = sum(compute_illegal_values(df) for df in datasets if df is not None)

    lookahead_pass = detect_lookahead(features_df, baseline_returns)
    lookahead_status = "PASS" if lookahead_pass else "FAIL"

    now_ts = dt.datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    columns = ["now_ts", "align_rate", "miss_rate", "dup_bars", "illegal_values", "lookahead_check"]
    values = {
        "now_ts": now_ts,
        "align_rate": align_rate,
        "miss_rate": miss_rate,
        "dup_bars": dup_bars,
        "illegal_values": illegal_values,
        "lookahead_check": lookahead_status,
    }
    append_row(QUALITY_MONITOR_PATH, columns, values)


def compute_max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max.replace(0, np.nan)
    return float(drawdown.min())


def run_kpi_monitor() -> None:
    if not SIGNALS_PATH.exists():
        return
    df = pd.read_csv(SIGNALS_PATH)
    if df.empty or "ts" not in df.columns:
        return
    df["ts"] = parse_ts(df["ts"])
    df = df.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
    cutoff = df.index.max() - pd.Timedelta(hours=168)
    subset = df[df.index > cutoff]
    if subset.empty:
        subset = df.tail(168)
    if subset.empty:
        return
    pnl = pd.to_numeric(subset.get("pnl_1h"), errors="coerce")
    equity = pd.to_numeric(subset.get("equity_1h"), errors="coerce")
    w_t = pd.to_numeric(subset.get("w_t"), errors="coerce")
    pnl_mean = pnl.mean()
    pnl_std = pnl.std(ddof=0)
    if pnl_std and not math.isclose(pnl_std, 0.0):
        sharpe = (pnl_mean / pnl_std) * ANNUALIZATION_FACTOR
    else:
        sharpe = float("nan")
    hit_rate = float((pnl > 0).mean())
    if equity.isna().all():
        maxdd = float("nan")
    else:
        maxdd = compute_max_drawdown(equity.fillna(method="ffill"))
    if w_t.isna().all():
        turnover = float("nan")
    else:
        turnover = float(w_t.diff().abs().mean())
    now_ts = dt.datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    columns = ["now_ts", "sharpe_annual", "hit_rate", "maxdd_7d", "turnover_7d"]
    values = {
        "now_ts": now_ts,
        "sharpe_annual": sharpe,
        "hit_rate": hit_rate,
        "maxdd_7d": maxdd,
        "turnover_7d": turnover,
    }
    append_row(KPI_MONITOR_PATH, columns, values)


def main() -> None:
    parse_args()
    run_data_quality_monitor()
    run_kpi_monitor()


if __name__ == "__main__":
    main()
