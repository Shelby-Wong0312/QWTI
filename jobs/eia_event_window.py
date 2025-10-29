import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze EIA event window performance")
    parser.add_argument(
        "--features",
        type=str,
        default="warehouse/signals_hourly_exp3.csv",
        help="Optional strategy PnL CSV (ts, w_t, pnl_1h)",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="warehouse/eia_window_summary.csv",
        help="Summary CSV output path",
    )
    parser.add_argument(
        "--output-panels",
        type=str,
        default="warehouse/eia_window_panels.csv",
        help="Panel CSV output path",
    )
    return parser.parse_args()


def load_prices() -> pd.DataFrame:
    price_files = sorted(Path("capital_wti_downloader/output").glob("*_HOUR_*.csv"))
    if not price_files:
        raise SystemExit("找不到 capital_wti_downloader/output/*_HOUR_*.csv")
    price_path = price_files[-1]
    prices = pd.read_csv(price_path, parse_dates=["snapshotTimeUTC"])
    prices = prices.drop_duplicates(subset=["snapshotTimeUTC"]).sort_values("snapshotTimeUTC")
    prices["ts"] = prices["snapshotTimeUTC"].dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize(None)
    prices["mid_close"] = (prices["close_bid"] + prices["close_ask"]) / 2.0
    prices = prices.dropna(subset=["mid_close"])
    prices = prices.set_index("ts")["mid_close"]
    return prices


def load_events() -> pd.DataFrame:
    events_path = Path("data/events_calendar.csv")
    if not events_path.exists():
        raise SystemExit("找不到 data/events_calendar.csv")
    events = pd.read_csv(events_path)
    events["event_time_utc"] = pd.to_datetime(events["event_time_utc"], utc=True, errors="coerce")
    events = events.dropna(subset=["event_time_utc"])
    events["event_time_utc"] = events["event_time_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    mask = (events["event_type"].str.upper() == "EIA") & (
        events["event_subtype"].str.upper().isin(["WPSR", "WEEKLY"])
    )
    events = events.loc[mask].copy()
    events = events.sort_values("event_time_utc")
    return events


def load_strategy(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    strat = pd.read_csv(path, parse_dates=["ts"])
    strat = strat.sort_values("ts").drop_duplicates(subset=["ts"])
    strat["ts"] = strat["ts"].dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize(None)
    if "pnl_1h" in strat.columns:
        return strat.set_index("ts")["pnl_1h"]
    return pd.Series(dtype=float)


def compute_window_panels(
    prices: pd.Series, events: pd.DataFrame, strat_pnl: pd.Series
) -> pd.DataFrame:
    ret_1h = np.log(prices).diff().fillna(0.0)
    panels = []
    for ts in events["event_time_utc"]:
        window_index = pd.date_range(ts - pd.Timedelta(hours=3), ts + pd.Timedelta(hours=3), freq="h")
        window_returns = ret_1h.reindex(window_index).fillna(0.0)
        cum_returns = window_returns.cumsum()
        strat_window = strat_pnl.reindex(window_index).fillna(0.0).cumsum()
        rel_hours = np.arange(-3, 4)
        for rel_h, px_ret, strat_val in zip(rel_hours, cum_returns.values, strat_window.values):
            panels.append(
                {
                    "event_ts": ts,
                    "rel_h": rel_h,
                    "px_ret": px_ret,
                    "strat_pnl": strat_val if not strat_pnl.empty else np.nan,
                }
            )
    return pd.DataFrame(panels)


def summarize_panels(panels: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for rel_h in [1, 2, 3]:
        subset = panels.loc[panels["rel_h"] == rel_h]
        if subset.empty:
            continue
        summary_rows.append(
            {
                "horizon_h": rel_h * 60,
                "px_ret_mean": subset["px_ret"].mean(),
                "px_ret_median": subset["px_ret"].median(),
                "px_ret_hit": (subset["px_ret"] > 0).mean(),
                "strat_mean": subset["strat_pnl"].mean(),
                "strat_median": subset["strat_pnl"].median(),
                "strat_hit": (subset["strat_pnl"] > 0).mean(),
                "count": len(subset),
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()

    prices = load_prices()
    events = load_events()
    strat_pnl = load_strategy(Path(args.features)) if args.features else pd.Series(dtype=float)

    if events.empty:
        print("沒有符合條件的 EIA 事件")
        return

    panels = compute_window_panels(prices, events, strat_pnl)
    summary = summarize_panels(panels)

    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary, index=False)
    panels.to_csv(args.output_panels, index=False)

    def fmt(val: float) -> str:
        return "nan" if pd.isna(val) else f"{val:.4f}"

    rows = len(events)
    if not summary.empty:
        last_row = summary.iloc[-1]
        print(
            f"events={rows}  px_ret_180m_mean={fmt(last_row['px_ret_mean'])}  "
            f"px_ret_180m_median={fmt(last_row['px_ret_median'])}  strat_180m_mean={fmt(last_row['strat_mean'])}  "
            f"strat_180m_median={fmt(last_row['strat_median'])}"
        )
    else:
        print(f"events={rows} (無有效資料)")


if __name__ == "__main__":
    main()

