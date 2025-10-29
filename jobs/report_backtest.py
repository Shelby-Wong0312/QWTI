import numpy as np
import pandas as pd
from pathlib import Path


def ensure_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True)
    return ts.dt.tz_convert("UTC")


def load_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit("找不到 warehouse/signals_hourly_v2.csv")
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = ensure_utc(df["ts"])
    df = df.set_index("ts").sort_index()
    return df


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit("找不到 warehouse/features_hourly_v2.csv")
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = ensure_utc(df["ts"])
    df = df.set_index("ts").sort_index()
    return df


def reconstruct_ret(signals: pd.DataFrame) -> pd.Series:
    pnl = signals["pnl_1h"].astype(float)
    weights = signals["w_t"].astype(float)
    pnl_values = pnl.to_numpy(dtype=float)
    weight_values = weights.to_numpy(dtype=float)
    denom_mask = np.abs(weight_values) > 1e-12
    ret_forward = np.zeros_like(pnl_values, dtype=float)
    np.divide(pnl_values, weight_values, out=ret_forward, where=denom_mask)
    ret_forward = np.where(np.isfinite(ret_forward), ret_forward, 0.0)
    ret_series = pd.Series(ret_forward, index=signals.index, dtype=float)
    ret_series = ret_series.shift(1).fillna(0.0)
    return ret_series


def compute_max_drawdown(equity: pd.Series) -> float:
    equity = equity.astype(float).ffill()
    if equity.empty:
        return np.nan
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min()


def compute_sharpe(pnl: pd.Series, periods_per_year: float) -> float:
    mean = pnl.mean()
    std = pnl.std()
    if not np.isfinite(std) or std <= 0:
        return np.nan
    return (mean / std) * np.sqrt(periods_per_year)


def main() -> None:
    signals_path = Path("warehouse/signals_hourly_v2.csv")
    features_path = Path("warehouse/features_hourly_v2.csv")

    signals = load_signals(signals_path)
    features = load_features(features_path)

    features = features.reindex(signals.index)
    features["w_t"] = signals["w_t"]

    pnl_hourly = signals["pnl_1h"].astype(float)
    equity_hourly = signals["equity_1h"].astype(float)
    weights = signals["w_t"].astype(float)

    sharpe_hourly = compute_sharpe(pnl_hourly, periods_per_year=24 * 252)
    mdd_hourly = compute_max_drawdown(equity_hourly)
    turnover_hourly = weights.diff().abs().mean()

    ret_hourly = reconstruct_ret(signals)
    mask_22 = ret_hourly.index.hour == 22
    weights_22 = weights.loc[mask_22]
    ret_nextday = ret_hourly.shift(-24).loc[mask_22]
    valid_nextday = weights_22.notna() & ret_nextday.notna()
    pnl_nextday = (weights_22 * ret_nextday).loc[valid_nextday]

    if valid_nextday.any():
        hit_nextday = (np.sign(weights_22[valid_nextday]) == np.sign(ret_nextday[valid_nextday])).astype(float).mean()
        sharpe_nextday = compute_sharpe(pnl_nextday, periods_per_year=252)
    else:
        hit_nextday = np.nan
        sharpe_nextday = np.nan

    ret_nextweek = ret_hourly.shift(-24 * 5).loc[mask_22]
    valid_nextweek = weights_22.notna() & ret_nextweek.notna()
    if valid_nextweek.any():
        hit_nextweek = (
            (np.sign(weights_22[valid_nextweek]) == np.sign(ret_nextweek[valid_nextweek]))
            .astype(float)
            .mean()
        )
    else:
        hit_nextweek = np.nan

    Path("warehouse").mkdir(parents=True, exist_ok=True)

    metrics = [
        ("Sharpe_hourly", sharpe_hourly),
        ("MDD_hourly", mdd_hourly),
        ("Turnover_hourly", turnover_hourly),
        ("Hit_nextday", hit_nextday),
        ("Sharpe_nextday", sharpe_nextday),
        ("Hit_nextweek", hit_nextweek),
    ]
    metrics_df = pd.DataFrame(metrics, columns=["key", "value"])
    metrics_df.to_csv(Path("warehouse/report_metrics.csv"), index=False)

    pd.DataFrame({"ts": signals.index, "equity_1h": equity_hourly.values}).to_csv(
        Path("warehouse/equity_hourly.csv"), index=False
    )
    pd.DataFrame({"ts": signals.index, "w_t": weights.values}).to_csv(
        Path("warehouse/weights_hourly.csv"), index=False
    )

    def fmt(val: float) -> str:
        return "nan" if pd.isna(val) else f"{val:.3f}"

    print(
        "Sharpe_hourly={s_h}  MDD_hourly={mdd}  Turnover_hourly={turn}  "
        "Hit_nextday={hit_d}  Sharpe_nextday={s_d}  Hit_nextweek={hit_w}".format(
            s_h=fmt(sharpe_hourly),
            mdd=fmt(mdd_hourly),
            turn=fmt(turnover_hourly),
            hit_d=fmt(hit_nextday),
            s_d=fmt(sharpe_nextday),
            hit_w=fmt(hit_nextweek),
        )
    )


if __name__ == "__main__":
    main()

