#!/usr/bin/env python3
"""
Bulk download and aggregate GDELT 2.1 GKG 15-minute feeds into hourly features.
"""
from __future__ import annotations

# --- No-Drift preflight ---
from pathlib import Path
import sys
sys.path.insert(0, r"C:\Users\niuji\Documents\Data\warehouse\policy\utils")
from nodrift_preflight import enforce
# --- End preflight ---

import argparse
import math
import re
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator, Optional
import zipfile

import numpy as np
import pandas as pd
import requests
from requests import RequestException
from tqdm import tqdm


BASE_URL = "http://data.gdeltproject.org/gdeltv2"
TONE_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?,\d+,\d+,-?\d+(?:\.\d+)?,\d+,\d+$")
URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


@dataclass
class Job:
    timestamp: datetime  # UTC
    url: str
    path: Path


@dataclass
class HourStats:
    art_cnt: int = 0
    tone_sum: float = 0.0
    tone_n: int = 0
    tone_pos_cnt: int = 0
    topics: set[str] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--raw-dir", default="data/gdelt_raw", help="Directory to store raw downloads")
    parser.add_argument("--out-parquet", default="data/gdelt_hourly.parquet", help="Output parquet path")
    parser.add_argument("--out-csv", default="data/gdelt_hourly.csv", help="Output CSV path")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent download workers")
    parser.add_argument("--only-missing", action="store_true", help="Skip downloads for existing files")
    parser.add_argument("--chunk-hours", type=int, default=24, help="Processing chunk size in hours")
    return parser.parse_args()


def daterange_quarter_hours(start: datetime, end: datetime) -> Iterator[datetime]:
    current = start
    delta = timedelta(minutes=15)
    while current <= end:
        yield current
        current += delta


def build_jobs(start: datetime, end: datetime, raw_dir: Path) -> list[Job]:
    jobs = []
    for ts in daterange_quarter_hours(start, end):
        stamp = ts.strftime("%Y%m%d%H%M%S")
        url = f"{BASE_URL}/{stamp}.gkg.csv.zip"
        dest = raw_dir / ts.strftime("%Y/%m") / f"{stamp}.gkg.csv.zip"
        jobs.append(Job(timestamp=ts, url=url, path=dest))
    return jobs


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_job(job: Job, only_missing: bool, retries: int = 3) -> str:
    if only_missing and job.path.exists():
        return "skipped"
    ensure_directory(job.path)
    backoff = 1.0
    for attempt in range(1, retries + 1):
        try:
            with requests.get(job.url, stream=True, timeout=(10, 60)) as resp:
                if resp.status_code == 200:
                    with job.path.open("wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            if chunk:
                                fh.write(chunk)
                    return "downloaded"
                if 400 <= resp.status_code < 500:
                    warnings.warn(f"HTTP {resp.status_code} for {job.url}; skipping.")
                    return "failed"
                warnings.warn(f"HTTP {resp.status_code} for {job.url}; retrying ({attempt}/{retries}).")
        except RequestException as exc:
            warnings.warn(f"Download error for {job.url}: {exc}; retrying ({attempt}/{retries}).")
        if attempt < retries:
            time.sleep(backoff)
            backoff *= 2
    return "failed"


def download_jobs(jobs: list[Job], only_missing: bool, workers: int) -> Counter:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    counts: Counter = Counter()
    max_workers = max(1, workers or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_job, job, only_missing): job for job in jobs}
        for future in tqdm(as_completed(futures), total=len(jobs), desc="Downloading", unit="file"):
            result = future.result()
            counts[result] += 1
    return counts


def detect_tone(value: str) -> Optional[float]:
    if value is None:
        return None
    if not TONE_PATTERN.match(value.strip()):
        return None
    parts = value.split(",")
    try:
        tone = float(parts[0])
    except ValueError:
        return None
    return tone


def detect_themes(value: str) -> set[str]:
    if not value or ";" not in value:
        return set()
    letters = sum(ch.isalpha() for ch in value)
    if letters / max(len(value), 1) < 0.1:
        return set()
    pieces = [piece.strip() for piece in value.split(";")]
    themes = {piece for piece in pieces if piece}
    return themes


def extract_fields(row: list[str]) -> tuple[Optional[float], set[str], Optional[str]]:
    tone = None
    themes: set[str] = set()
    url = None
    for field in row:
        if tone is None:
            tone = detect_tone(field)
        if not themes:
            candidate = detect_themes(field)
            if candidate:
                themes = candidate
        if url is None and field and URL_PATTERN.match(field):
            url = field
        if tone is not None and themes and url is not None:
            break
    return tone, themes, url


def parse_zip(path: Path, expected_ts: datetime) -> list[tuple[datetime, Optional[float], set[str]]]:
    try:
        with zipfile.ZipFile(path) as zf:
            inner_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            inner = inner_names[0] if inner_names else zf.namelist()[0]
            with zf.open(inner) as fh:
                rows: list[tuple[datetime, Optional[float], set[str]]] = []
                for line in fh:
                    try:
                        decoded = line.decode("utf-8", errors="ignore").rstrip("\n")
                    except UnicodeDecodeError:
                        continue
                    columns = decoded.split("\t")
                    tone, themes, url = extract_fields(columns)
                    ts = expected_ts.replace(minute=0, second=0, microsecond=0)
                    rows.append((ts, tone, themes))
    except (zipfile.BadZipFile, FileNotFoundError, IndexError, KeyError) as exc:
        warnings.warn(f"Failed to parse {path}: {exc}")
        return []
    return rows


def chunk_jobs(jobs: list[Job], chunk_hours: int) -> Iterator[list[Job]]:
    jobs_sorted = sorted(jobs, key=lambda job: job.timestamp)
    chunk_size = max(1, chunk_hours * 4)  # 4 quarter-hour files per hour
    for i in range(0, len(jobs_sorted), chunk_size):
        yield jobs_sorted[i : i + chunk_size]


def process_jobs(jobs: list[Job], chunk_hours: int) -> tuple[pd.DataFrame, int]:
    stats: defaultdict[datetime, HourStats] = defaultdict(HourStats)
    parsed_files = 0
    chunk_size = max(1, chunk_hours * 4)
    total_chunks = math.ceil(len(jobs) / chunk_size)
    for chunk in tqdm(chunk_jobs(jobs, chunk_hours), total=total_chunks, desc="Parsing", unit="chunk"):
        for job in chunk:
            if not job.path.exists():
                continue
            records = parse_zip(job.path, job.timestamp)
            if not records:
                continue
            parsed_files += 1
            for ts, tone, themes in records:
                hour_stats = stats[ts]
                hour_stats.art_cnt += 1
                if tone is not None:
                    hour_stats.tone_sum += tone
                    hour_stats.tone_n += 1
                    if tone > 0:
                        hour_stats.tone_pos_cnt += 1
                if themes:
                    hour_stats.topics.update(themes)
    rows = []
    for ts in sorted(stats.keys()):
        stat = stats[ts]
        tone_avg = float(stat.tone_sum / stat.tone_n) if stat.tone_n else np.nan
        tone_pos_ratio = float(stat.tone_pos_cnt / stat.tone_n) if stat.tone_n else np.nan
        rows.append(
            {
                "ts": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "art_cnt": int(stat.art_cnt),
                "tone_avg": tone_avg,
                "tone_pos_ratio": tone_pos_ratio,
                "topic_cnt": int(len(stat.topics)),
            }
        )
    hourly = pd.DataFrame(rows, columns=["ts", "art_cnt", "tone_avg", "tone_pos_ratio", "topic_cnt"])
    return hourly, parsed_files


def build_observed_metrics(
    hourly_df: pd.DataFrame,
    start_ts: datetime,
    end_ts: datetime,
    available_files: int,
    parsed_files: int,
) -> dict[str, float]:
    """Derive coverage/quality stats for No-Drift enforcement."""
    end_hour = end_ts.replace(minute=0, second=0, microsecond=0)
    total_hours = (
        int(((end_hour - start_ts).total_seconds() / 3600) + 1) if end_hour >= start_ts else 0
    )
    mapped_ratio = float(len(hourly_df) / total_hours) if total_hours > 0 else 0.0
    all_art_cnt = float(hourly_df["art_cnt"].median()) if not hourly_df.empty else 0.0
    tone_nonnull = bool(not hourly_df.empty and hourly_df["tone_avg"].notna().all())
    if available_files <= 0:
        skip_ratio = 1.0
    else:
        skip_ratio = max(0.0, 1.0 - (parsed_files / available_files))
    return dict(
        mode="hard_kpi",
        mapped_ratio=mapped_ratio,
        all_art_cnt=all_art_cnt,
        tone_nonnull=tone_nonnull,
        skip_ratio=skip_ratio,
    )


def main() -> None:
    args = parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1) - timedelta(seconds=1)
    start_ts = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_ts = end_date.replace(minute=45, second=0, microsecond=0)

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(start_ts, end_ts, raw_dir)
    print(f"[INFO] Total intervals: {len(jobs)}")

    download_counts = download_jobs(jobs, only_missing=args.only_missing, workers=args.workers)
    available_files = sum(1 for job in jobs if job.path.exists())
    print(
        "[INFO] Download stats | downloaded:",
        download_counts.get("downloaded", 0),
        "skipped:",
        download_counts.get("skipped", 0),
        "failed:",
        download_counts.get("failed", 0),
    )
    print(f"[INFO] Files available locally: {available_files}")

    hourly_df, parsed_files = process_jobs(jobs, args.chunk_hours)
    print(f"[INFO] Parsed files with data: {parsed_files}")
    print(f"[INFO] Hourly rows produced: {len(hourly_df)}")

    observed = build_observed_metrics(hourly_df, start_ts, end_ts, available_files, parsed_files)
    enforce(observed)

    out_parquet = Path(args.out_parquet)
    out_csv = Path(args.out_csv)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not hourly_df.empty:
        hourly_df.to_parquet(out_parquet, index=False)
        hourly_df.to_csv(out_csv, index=False)
        print(f"[INFO] Wrote {out_parquet} and {out_csv}")
    else:
        warnings.warn("No hourly data produced; outputs not written.")


if __name__ == "__main__":
    main()




