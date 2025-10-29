import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate GDELT raw data into hourly metrics")
    parser.add_argument("--raw-root", type=Path, default=Path("warehouse/gdelt_raw"), help="Raw data root directory")
    parser.add_argument("--out", type=Path, default=Path("data/gdelt_hourly.csv"), help="Output CSV path")
    parser.add_argument("--log-every", type=int, default=100, help="Log progress every N files (default: 100)")
    return parser.parse_args()


def list_raw_files(root: Path) -> List[Path]:
    if not root.exists():
        raise SystemExit(f"Raw directory not found: {root}")
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file():
            lower = path.name.lower()
            if lower.endswith(".parquet") or lower.endswith(".csv") or lower.endswith(".csv.gz"):
                files.append(path)
    if not files:
        raise SystemExit(f"No raw files found under {root}")
    files.sort()
    return files


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".gz" and "".join(path.suffixes[-2:]).lower() == ".csv.gz":
        df = pd.read_csv(path, compression="gzip")
    else:
        df = pd.read_csv(path)
    return df


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ts" not in df.columns:
        if "DATE" in df.columns:
            ts_series = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d%H%M%S", errors="coerce", utc=True)
            df["ts"] = ts_series.dt.floor("H")
        else:
            df["ts"] = pd.NaT
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "tone" not in df.columns:
        if "V2Tone" in df.columns:
            df["tone"] = df["V2Tone"].apply(_parse_tone)
        else:
            df["tone"] = np.nan
    df["tone"] = pd.to_numeric(df["tone"], errors="coerce")
    if "domain" not in df.columns:
        df["domain"] = np.nan
    if "DocumentIdentifier" not in df.columns:
        df["DocumentIdentifier"] = np.nan
    return df


def _parse_tone(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    head = str(value).split(",", 1)[0].strip()
    if not head:
        return None
    try:
        return float(head)
    except ValueError:
        return None


def aggregate_files(files: List[Path], log_every: int) -> Dict[pd.Timestamp, Dict[str, float]]:
    aggregates: Dict[pd.Timestamp, Dict[str, float]] = defaultdict(lambda: {"count": 0.0, "sum": 0.0, "sum_sq": 0.0, "count_pos": 0.0})
    for idx, path in enumerate(files, start=1):
        df = load_frame(path)
        if df.empty:
            continue
        df = ensure_columns(df)
        df = df.dropna(subset=["ts", "tone"])
        if df.empty:
            continue
        if {"domain", "DocumentIdentifier"}.issubset(df.columns):
            df = df.drop_duplicates(subset=["domain", "DocumentIdentifier"])
        grouped = df.groupby("ts")
        for ts, grp in grouped:
            grp_tone = grp["tone"].dropna()
            if grp_tone.empty:
                continue
            n = float(len(grp_tone))
            tone_sum = float(grp_tone.sum())
            tone_sq = float((grp_tone ** 2).sum())
            pos = float((grp_tone > 0).sum())
            slot = aggregates[ts]
            slot["count"] += n
            slot["sum"] += tone_sum
            slot["sum_sq"] += tone_sq
            slot["count_pos"] += pos
        if idx % max(log_every, 1) == 0:
            print(f"Processed {idx} files / {len(files)}")
    return aggregates


def build_output(aggregates: Dict[pd.Timestamp, Dict[str, float]]) -> pd.DataFrame:
    if not aggregates:
        return pd.DataFrame(columns=["ts", "art_cnt", "tone_avg", "tone_pos_ratio", "se_tone", "low_confidence"])
    records = []
    for ts in sorted(aggregates.keys()):
        slot = aggregates[ts]
        count = int(slot["count"])
        if count <= 0:
            continue
        tone_sum = slot["sum"]
        tone_sq = slot["sum_sq"]
        pos = slot["count_pos"]
        mean = tone_sum / count
        if count > 1:
            variance = (tone_sq - (tone_sum ** 2) / count) / (count - 1)
            variance = max(variance, 0.0)
            std = math.sqrt(variance)
            se = std / math.sqrt(count)
        else:
            se = float("nan")
        tone_pos_ratio = pos / count
        low_conf = (not math.isnan(se) and se > 0.15) or (count < 10)
        records.append(
            {
                "ts": ts,
                "art_cnt": count,
                "tone_avg": mean,
                "tone_pos_ratio": tone_pos_ratio,
                "se_tone": se,
                "low_confidence": bool(low_conf),
            }
        )
    df = pd.DataFrame.from_records(records)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"])
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df


def main() -> None:
    args = parse_args()
    files = list_raw_files(args.raw_root)
    aggregates = aggregate_files(files, args.log_every)
    output = build_output(aggregates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"Aggregated {len(output)} hourly rows into {args.out}")


if __name__ == "__main__":
    main()
