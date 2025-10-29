import argparse
import io
import logging
import math
import re
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

THEME_REGEX = re.compile(r"(opec|crudeoil|oil_price|refinery|oilstocks|petroleum|brent|wti)", re.IGNORECASE)
URL_REGEX = re.compile(r"(crude|wti|opec|oil)", re.IGNORECASE)
FIFTEEN_MINUTES = timedelta(minutes=15)


def parse_cli_date(value: str) -> datetime:
    dt = datetime.strptime(value, "%Y-%m-%d")
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)


def determine_range(from_str: str, to_str: str) -> Tuple[datetime, datetime, datetime]:
    start_dt = parse_cli_date(from_str)
    end_day = parse_cli_date(to_str)
    end_dt_15 = end_day + timedelta(days=1) - FIFTEEN_MINUTES
    hour_end = end_day + timedelta(days=1) - timedelta(hours=1)
    if hour_end < start_dt:
        raise ValueError("'--to' must be later than '--from'")
    return start_dt, end_dt_15, hour_end


def iter_times(start_dt: datetime, end_dt: datetime, max_files: int) -> Sequence[datetime]:
    total_steps = int(math.floor((end_dt - start_dt) / FIFTEEN_MINUTES)) + 1
    steps = min(total_steps, max_files)
    return [start_dt + i * FIFTEEN_MINUTES for i in range(steps)]


def build_url(ts: datetime, scheme: str) -> str:
    stamp = ts.strftime("%Y%m%d%H%M") + "00"
    return f"{scheme}://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"


def create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_file(session: requests.Session, ts: datetime, timeout: float) -> Optional[bytes]:
    for scheme in ("https", "http"):
        url = build_url(ts, scheme)
        try:
            response = session.get(url, timeout=timeout, verify=False)
        except requests.exceptions.SSLError as exc:
            logging.warning("SSL error for %s: %s", url, exc)
            if scheme == "https":
                logging.info("Retrying %s via HTTP", ts.strftime("%Y-%m-%d %H:%M"))
                continue
            return None
        except requests.RequestException as exc:
            logging.warning("Request failed for %s: %s", url, exc)
            if scheme == "http":
                return None
            continue
        if response.status_code == 404:
            logging.warning("File not found (404): %s", url)
            return None
        if response.status_code != 200:
            logging.warning("HTTP status %s for %s", response.status_code, url)
            return None
        return response.content
    logging.warning("Unable to retrieve file for %s via HTTPS and HTTP", ts.strftime("%Y-%m-%d %H:%M"))
    return None


def read_gkg(content: bytes, ts: datetime) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = zf.namelist()
        if not names:
            return pd.DataFrame(columns=["themes", "tone", "url", "ts"])
        with zf.open(names[0]) as fh:
            df_raw = pd.read_csv(
                fh,
                sep="\t",
                header=None,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                engine="python",
            )
    if df_raw.empty:
        return pd.DataFrame(columns=["themes", "tone", "url", "ts"])

    header_map = None
    first_row = [str(x).strip().lower() for x in df_raw.iloc[0].tolist()]
    has_header = any(val in first_row for val in ("v2themes", "v2tone", "documentidentifier"))
    if has_header:
        header_map = {name: idx for idx, name in enumerate(first_row)}
        df_data = df_raw.iloc[1:].reset_index(drop=True)
    else:
        df_data = df_raw

    if df_data.empty:
        return pd.DataFrame(columns=["themes", "tone", "url", "ts"])

    def get_col(name: str, fallback_idx: int) -> pd.Series:
        if header_map and name in header_map and header_map[name] < df_data.shape[1]:
            return df_data.iloc[:, header_map[name]].astype(str)
        idx = fallback_idx if fallback_idx < df_data.shape[1] else df_data.shape[1] - 1
        return df_data.iloc[:, idx].astype(str)

    themes = get_col("v2themes", 6)
    tone = get_col("v2tone", 33)
    url = get_col("documentidentifier", df_data.shape[1] - 1)
    df = pd.DataFrame({"themes": themes, "tone": tone, "url": url})
    df["ts"] = pd.Timestamp(ts)
    return df


def parse_tone(value: str) -> float:
    try:
        first = value.split(",")[0].strip()
        result = float(first)
        return result if np.isfinite(result) else 0.0
    except (ValueError, IndexError):
        return 0.0


def filter_records(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["themes"].map(lambda x: bool(THEME_REGEX.search(str(x)))) | df["url"].map(
        lambda x: bool(URL_REGEX.search(str(x)))
    )
    filtered = df.loc[mask].copy()
    if filtered.empty:
        return filtered
    filtered["tone_value"] = filtered["tone"].map(parse_tone)
    filtered["ts"] = pd.to_datetime(filtered["ts"], utc=True)
    return filtered[["ts", "tone_value"]]


def process_range(start_dt: datetime, end_dt_15: datetime, max_files: int, timeout: float, sleep_sec: float) -> pd.DataFrame:
    session = create_session()
    records: List[pd.DataFrame] = []
    for idx, ts in enumerate(iter_times(start_dt, end_dt_15, max_files), start=1):
        content = fetch_file(session, ts, timeout)
        if not content:
            continue
        try:
            df_raw = read_gkg(content, ts)
        except (zipfile.BadZipFile, pd.errors.ParserError) as exc:
            logging.warning("Failed to parse archive for %s: %s", ts.strftime("%Y-%m-%d %H:%M"), exc)
            continue
        filtered = filter_records(df_raw)
        if not filtered.empty:
            records.append(filtered)
        if idx % 20 == 0:
            logging.info("Processed %s files", idx)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    if not records:
        return pd.DataFrame(columns=["ts", "tone_value"])
    return pd.concat(records, ignore_index=True)


def aggregate_hourly(data: pd.DataFrame, start_dt: datetime, hour_end: datetime) -> pd.DataFrame:
    index = pd.date_range(start=start_dt, end=hour_end, freq="H", tz=timezone.utc)
    if data.empty:
        return pd.DataFrame({"ts": index, "art_cnt": 0, "tone_avg": 0.0, "tone_pos_ratio": 0.5})
    grouped = (
        data.set_index("ts")
        .sort_index()
        .resample("H")
        .agg(
            art_cnt=("tone_value", "count"),
            tone_avg=("tone_value", "mean"),
            tone_pos_ratio=("tone_value", lambda x: (x > 0).mean() if len(x) else np.nan),
        )
    )
    grouped = grouped.reindex(index)
    grouped["art_cnt"] = grouped["art_cnt"].fillna(0).astype(int)
    grouped["tone_avg"] = grouped["tone_avg"].fillna(0.0)
    grouped["tone_pos_ratio"] = np.where(grouped["art_cnt"] > 0, grouped["tone_pos_ratio"].fillna(0.5), 0.5)
    grouped = grouped.reset_index().rename(columns={"index": "ts"})
    return grouped


def offline_fill(start_dt: datetime, hour_end: datetime) -> pd.DataFrame:
    index = pd.date_range(start=start_dt, end=hour_end, freq="H", tz=timezone.utc)
    return pd.DataFrame({"ts": index, "art_cnt": 0, "tone_avg": 0.0, "tone_pos_ratio": 0.5})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch GDELT GKG master (15-min) and aggregate to hourly CSV")
    parser.add_argument("--from", dest="from_date", required=True, help="Start date (UTC) in YYYY-MM-DD format")
    parser.add_argument("--to", dest="to_date", required=True, help="End date (UTC) in YYYY-MM-DD format")
    parser.add_argument("--output", type=str, default="data/gdelt_hourly.csv", help="Output CSV path")
    parser.add_argument("--max-files", type=int, default=200, help="Maximum number of 15-minute files to process")
    parser.add_argument("--timeout", type=float, default=8.0, help="Per-request timeout in seconds")
    parser.add_argument("--sleep", type=float, default=0.02, help="Sleep between file downloads in seconds")
    parser.add_argument("--offline-fill", action="store_true", help="Skip downloads and output neutral hourly data")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    try:
        start_dt, end_dt_15, hour_end = determine_range(args.from_date, args.to_date)
    except ValueError as exc:
        logging.error("Invalid date range: %s", exc)
        sys.exit(1)

    if args.offline_fill:
        hourly = offline_fill(start_dt, hour_end)
    else:
        data = process_range(start_dt, end_dt_15, args.max_files, args.timeout, args.sleep)
        hourly = aggregate_hourly(data, start_dt, hour_end)

    hourly["ts"] = hourly["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly[["ts", "art_cnt", "tone_avg", "tone_pos_ratio"]].to_csv(output_path, index=False)
    logging.info("Saved %s rows to %s", len(hourly), output_path)


if __name__ == "__main__":
    main()
