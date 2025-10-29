import argparse
import io
import logging
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

import pandas as pd
import requests

UTC = timezone.utc
GDELT_URL_TEMPLATE = "https://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
TARGET_COLUMNS = ["DATE", "SourceCollectionIdentifier", "DocumentIdentifier", "V2Tone"]


@dataclass
class DownloadStats:
    success: int = 0
    failed: int = 0
    skipped: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill GDELT GKG data into monthly raw partitions")
    parser.add_argument("--start", type=str, required=True, help="Start datetime (UTC), e.g. 2019-02-01")
    parser.add_argument("--end", type=str, required=True, help="End datetime (UTC), inclusive")
    parser.add_argument("--step-days", type=int, default=30, help="Chunk size in days (default: 30)")
    parser.add_argument("--out", type=Path, default=Path("warehouse/gdelt_raw"), help="Output root directory")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum download retries (default: 3)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def normalize_timestamp(value: str, *, is_end: bool) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    else:
        ts = ts.tz_convert(UTC)
    if not _has_time_component(value):
        if is_end:
            ts = ts + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    ts = ts.floor("15min")
    return ts.to_pydatetime()


def _has_time_component(value: str) -> bool:
    return any(sep in value for sep in ("T", " ", ":"))


def iter_quarter_hours(start: datetime, end: datetime) -> Iterable[datetime]:
    current = start
    while current <= end:
        yield current
        current += timedelta(minutes=15)


def build_url(ts: datetime) -> str:
    return GDELT_URL_TEMPLATE.format(stamp=ts.strftime("%Y%m%d%H%M%S"))


def download_zip(url: str, max_retries: int) -> Optional[bytes]:
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=60)
        except requests.RequestException as exc:
            logging.warning("Download error (%s/%s): %s", attempt, max_retries, exc)
            continue
        if response.status_code == 200:
            return response.content
        if response.status_code == 404:
            logging.debug("File not found (404): %s", url)
            return None
        logging.warning("HTTP %s for %s (attempt %s/%s)", response.status_code, url, attempt, max_retries)
    return None


def extract_dataframe(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        inner_name = _select_inner_file(zf)
        if inner_name is None:
            return pd.DataFrame(columns=TARGET_COLUMNS)
        with zf.open(inner_name) as fh:
            try:
                df = pd.read_csv(
                    fh,
                    sep="\t",
                    dtype=str,
                    engine="python",
                    usecols=lambda col: col in TARGET_COLUMNS,
                    on_bad_lines="skip",
                )
            except ValueError:
                fh.seek(0)
                df = pd.read_csv(
                    fh,
                    sep="\t",
                    dtype=str,
                    engine="python",
                    names=None,
                    on_bad_lines="skip",
                )
    if not df.empty:
        df = df[[col for col in TARGET_COLUMNS if col in df.columns]]
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[TARGET_COLUMNS]


def _select_inner_file(zf: zipfile.ZipFile) -> Optional[str]:
    for name in zf.namelist():
        lower = name.lower()
        if lower.endswith(".csv") or lower.endswith(".gkg"):
            return name
    return zf.namelist()[0] if zf.namelist() else None


def parse_domain(url: str) -> Optional[str]:
    if not isinstance(url, str) or not url:
        return None
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    if not host and parsed.path:
        # Handle URLs missing scheme
        parsed = urlparse("http://" + url.strip())
        host = parsed.netloc.lower()
    host = host.split("@")[-1]
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host or None


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(ts=pd.NaT, tone=pd.NA, domain=pd.NA)
    df = df.copy()
    df["DATE"] = pd.to_numeric(df["DATE"], errors="coerce").astype("Int64")
    df["ts"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d%H%M%S", errors="coerce", utc=True)
    df["ts"] = df["ts"].dt.floor("H")
    df["tone"] = df["V2Tone"].apply(_parse_tone)
    df["DocumentIdentifier"] = df["DocumentIdentifier"].astype(str).str.strip()
    df["domain"] = df["DocumentIdentifier"].apply(parse_domain)
    df = df.dropna(subset=["ts", "DocumentIdentifier", "domain"])
    df = df.drop_duplicates(subset=["domain", "DocumentIdentifier"])
    return df


def _parse_tone(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).split(",", 1)[0].strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def write_partition(df: pd.DataFrame, slot: datetime, out_root: Path) -> Path:
    month_dir = out_root / slot.strftime("%Y-%m")
    month_dir.mkdir(parents=True, exist_ok=True)
    file_stub = slot.strftime("%Y%m%d%H%M%S")
    parquet_path = month_dir / f"{file_stub}.parquet"
    csv_path = month_dir / f"{file_stub}.csv"
    try:
        df.to_parquet(parquet_path, index=False)
        if csv_path.exists():
            csv_path.unlink()
        return parquet_path
    except (ImportError, ValueError):
        df.to_csv(csv_path, index=False)
        if parquet_path.exists():
            parquet_path.unlink()
        return csv_path


def process_slot(slot: datetime, args: argparse.Namespace, stats: DownloadStats) -> None:
    url = build_url(slot)
    month_dir = args.out / slot.strftime("%Y-%m")
    exists = any(
        (month_dir / f"{slot.strftime('%Y%m%d%H%M%S')}{suffix}").exists()
        for suffix in (".parquet", ".csv")
    )
    if exists:
        stats.skipped += 1
        logging.debug("Skip existing %s", slot.strftime("%Y-%m-%d %H:%M"))
        return
    blob = download_zip(url, args.max_retries)
    if blob is None:
        stats.failed += 1
        logging.warning("Failed to download %s", url)
        return
    df = extract_dataframe(blob)
    df = enrich_dataframe(df)
    write_partition(df, slot, args.out)
    stats.success += 1


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    start_ts = normalize_timestamp(args.start, is_end=False)
    end_ts = normalize_timestamp(args.end, is_end=True)
    if end_ts < start_ts:
        raise SystemExit("End timestamp must be after start timestamp")
    stats = DownloadStats()
    step = timedelta(days=args.step_days)
    current = start_ts
    while current <= end_ts:
        chunk_end = min(current + step - timedelta(minutes=15), end_ts)
        for slot in iter_quarter_hours(current, chunk_end):
            process_slot(slot, args, stats)
        logging.info(
            "Processed chunk %s - %s (success=%s, failed=%s, skipped=%s)",
            current.strftime("%Y-%m-%d %H:%M"),
            chunk_end.strftime("%Y-%m-%d %H:%M"),
            stats.success,
            stats.failed,
            stats.skipped,
        )
        current = chunk_end + timedelta(minutes=15)
    print(
        f"GDELT backfill complete: success={stats.success} failed={stats.failed} skipped={stats.skipped}",
    )


if __name__ == "__main__":
    main()
