import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build term/crack/OVX features from local CL1 data only")
    parser.add_argument("--output", type=str, default="data/term_crack_ovx_hourly.csv", help="Output CSV path")
    return parser.parse_args()


def load_latest_price_file() -> Path:
    files = sorted(Path("capital_wti_downloader/output").glob("*_HOUR_*.csv"))
    if not files:
        raise SystemExit("找不到 capital_wti_downloader/output/*_HOUR_*.csv")
    return files[-1]


def ensure_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True)
    return ts.dt.tz_convert("UTC")


def compute_ovx_quantile(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if len(valid) == 0:
        return pd.Series(np.nan, index=series.index)
    if len(valid) == 1:
        return pd.Series([0.5], index=valid.index).reindex(series.index)
    ranks = valid.rank(method="average") - 1.0
    denom = len(valid) - 1.0
    quantiles = (ranks / denom).clip(0.0, 1.0)
    return quantiles.reindex(series.index)


def main() -> None:
    args = parse_args()
    price_file = load_latest_price_file()

    prices = pd.read_csv(price_file, parse_dates=["snapshotTimeUTC"]).sort_values("snapshotTimeUTC")
    prices["ts"] = ensure_utc(prices["snapshotTimeUTC"])
    prices["mid_close"] = (prices["close_bid"] + prices["close_ask"]) / 2.0
    prices = prices.drop_duplicates(subset=["ts"]).set_index("ts")

    start = prices.index.min().floor("h")
    end = prices.index.max().ceil("H")
    hourly_index = pd.date_range(start=start, end=end, freq="h", tz="UTC")

    mid_series = prices["mid_close"].reindex(hourly_index).ffill()
    log_price = np.log(mid_series)
    ret_1h = log_price.diff()
    rv_24h = ret_1h.rolling(window=24, min_periods=1).std(ddof=0)
    ovx = compute_ovx_quantile(rv_24h)

    output = pd.DataFrame(
        {
            "ts": hourly_index.tz_convert(None).strftime("%Y-%m-%dT%H:%M:%S"),
            "cl1_cl2": 0.0,
            "crack_rb": 0.0,
            "crack_ho": 0.0,
            "ovx": ovx.values,
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()


