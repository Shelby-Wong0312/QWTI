import argparse
import io
import logging
import math
import re
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def determine_range(from_str: Optional[str], to_str: Optional[str]) -> Tuple[datetime, datetime, datetime]:
    today = datetime.now(timezone.utc)
    start_dt = parse_cli_date(from_str) if from_str else datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=7)
    end_day = parse_cli_date(to_str) if to_str else datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    end_dt_15 = end_day + timedelta(days=1) - FIFTEEN_MINUTES
    hour_end = end_day + timedelta(days=1) - timedelta(hours=1)
    if hour_end < start_dt:
        raise ValueError("'--to' must be later than '--from'")
    return start_dt, end_dt_15, hour_end


def iter_times(start_dt: datetime, end_dt: datetime) -> Sequence[datetime]:
    steps = int(math.floor((end_dt - start_dt) / FIFTEEN_MINUTES)) + 1
    return [start_dt + i * FIFTEEN_MINUTES for i in range(steps)]


def build_url(ts: datetime, scheme: str) -> str:
    return f"{scheme}://data.gdeltproject.org/gdeltv2/{ts.strftime('%Y%m%d%H%M')}.gkg.csv.zip"


def create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_column(df: pd.DataFrame, header_map: Optional[Dict[str, int]], key: str, fallback_idx: Optional[int]) -> pd.Series:
    if header_map and key in header_map:
        idx = header_map[key]
        if idx < df.shape[1]:
            return df.iloc[:, idx]
    if fallback_idx is not None:
        idx = fallback_idx if fallback_idx < df.shape[1] else df.shape[1] - 1
        return df.iloc[:, idx]
    return df.iloc[:, -1]


def read_gkg_file(content: bytes, ts: datetime) -> pd.DataFrame:
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

    header_map: Optional[Dict[str, int]] = None
    first_row_lower = [str(x).strip().lower() for x in df_raw.iloc[0].tolist()]
    if "v2themes" in first_row_lower or "v2tone" in first_row_lower or "documentidentifier" in first_row_lower:
        header_map = {name: idx for idx, name in enumerate(first_row_lower)}
        df_data = df_raw.iloc[1:].reset_index(drop=True)
    else:
        df_data = df_raw

    if df_data.empty:
        return pd.DataFrame(columns=["themes", "tone", "url", "ts"])

    themes = fetch_column(df_data, header_map, "v2themes", 7)
    tone = fetch_column(df_data, header_map, "v2tone", 34)
    url = fetch_column(df_data, header_map, "documentidentifier", None)

    df = pd.DataFrame(
        {
            "themes": themes.astype(str),
            "tone": tone.astype(str),
            "url": url.astype(str),
        }
    )
    df["ts"] = pd.Timestamp(ts)
    return df


def tone_to_float(value: str) -> float:
    try:
        first = value.split(",")[0].strip()
        return float(first)
    except (ValueError, IndexError):
        return 0.0


def fetch_file(session: requests.Session, ts: datetime) -> Optional[bytes]:
    for scheme in ("https", "http"):
        url = build_url(ts, scheme)
        try:
            response = session.get(url, timeout=30, verify=False)
        except requests.exceptions.SSLError as exc:
            logging.warning("SSL error for %s: %s", url, exc)
            if scheme == "https":
                logging.info("Retrying with HTTP")
                continue
            return None
        except requests.RequestException as exc:
            logging.warning("Request failed for %s: %s", url, exc)
            if scheme == "http":
                return None
            continue
        if response.status_code == 404:
            logging.warning("File not found (404): %s", url)
            if scheme == "http":
                return None
            continue
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            logging.warning("HTTP error for %s: %s", url, exc)
            if scheme == "http":
                return None
            continue
        return response.content
    logging.warning("Failed to retrieve file for %s via HTTPS/HTTP", ts.strftime("%Y-%m-%d %H:%M"))
    return None


def process_files(start_dt: datetime, end_dt_15: datetime) -> pd.DataFrame:
    session = create_session()
    records: List[pd.DataFrame] = []
    counter = 0
    for ts in iter_times(start_dt, end_dt_15):
        content = fetch_file(session, ts)
        if content is None:
            continue
        try:
            df = read_gkg_file(content, ts)
        except (zipfile.BadZipFile, pd.errors.ParserError) as exc:
            logging.warning("Failed to parse file for %s: %s", ts.strftime("%Y-%m-%d %H:%M"), exc)
            continue
        if df.empty:
            continue

        mask = THEME_REGEX.search
        url_mask = URL_REGEX.search
        filtered = df[
            df["themes"].map(lambda x: bool(mask(str(x)))) | df["url"].map(lambda x: bool(url_mask(str(x))))
        ].copy()
        if filtered.empty:
            continue
        filtered["tone_value"] = filtered["tone"].map(tone_to_float)
        filtered["ts"] = pd.to_datetime(filtered["ts"], utc=True)
        filtered = filtered[["ts", "tone_value"]]
        if not filtered.empty:
            records.append(filtered)

        counter += 1
        if counter % 20 == 0:
            logging.info("Processed %s files", counter)
    if not records:
        return pd.DataFrame(columns=["ts", "tone_value"])
    return pd.concat(records, ignore_index=True)


def aggregate_hourly(data: pd.DataFrame, start_dt: datetime, hour_end: datetime) -> pd.DataFrame:
    if data.empty:
        idx = pd.date_range(start=start_dt, end=hour_end, freq="H", tz=timezone.utc)
        return pd.DataFrame({"ts": idx, "art_cnt": 0, "tone_avg": 0.0, "tone_pos_ratio": 0.5})

    resampled = (
        data.set_index("ts")
        .sort_index()
        .resample("H")
        .agg(
            art_cnt=("tone_value", "count"),
            tone_avg=("tone_value", "mean"),
            tone_pos_ratio=("tone_value", lambda x: (x > 0).mean() if len(x) > 0 else np.nan),
        )
    )
    idx = pd.date_range(start=start_dt, end=hour_end, freq="H", tz=timezone.utc)
    resampled = resampled.reindex(idx)
    resampled["art_cnt"] = resampled["art_cnt"].fillna(0).astype(int)
    resampled["tone_avg"] = resampled["tone_avg"].fillna(0.0)
    resampled["tone_pos_ratio"] = np.where(
        resampled["art_cnt"] > 0,
        resampled["tone_pos_ratio"].fillna(0.5),
        0.5,
    )
    resampled = resampled.reset_index().rename(columns={"index": "ts"})
    return resampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch GDELT GKG via HTTP and aggregate to hourly CSV")
    parser.add_argument("--from", dest="from_date", type=str, help="Start date (UTC) in YYYY-MM-DD format")
    parser.add_argument("--to", dest="to_date", type=str, help="End date (UTC) in YYYY-MM-DD format")
    parser.add_argument("--output", type=str, default="data/gdelt_hourly.csv", help="Output CSV path")
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

    data = process_files(start_dt, end_dt_15)
    hourly = aggregate_hourly(data, start_dt, hour_end)
    hourly["ts"] = hourly["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly[["ts", "art_cnt", "tone_avg", "tone_pos_ratio"]].to_csv(output_path, index=False)
    logging.info("Saved %s rows to %s", len(hourly), output_path)


if __name__ == "__main__":
    main()




