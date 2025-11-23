# #!/usr/bin/env python3  # Commented out for Windows compatibility
"""
Enhanced GDELT GKG aggregator with bucket mapping and 1/k normalization.
Processes RAW GKG files and generates monthly parquet files with bucket features.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import warnings
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


BASE_URL = "http://data.gdeltproject.org/gkg"
TONE_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?,\d+,\d+,-?\d+(?:\.\d+)?,\d+,\d+$")
THEME_MAP_PATH = Path("warehouse/theme_map.json")
FILTER_CONFIG_PATH = Path("warehouse/filter_config_v2.json")

# Six core buckets
BUCKETS = ["OIL_CORE", "GEOPOL", "USD_RATE", "SUPPLY_CHAIN", "MACRO", "ESG_POLICY"]


@dataclass
class Job:
    timestamp: datetime  # UTC
    path: Path


@dataclass
class BucketStats:
    """Stats for a single bucket in a single hour"""
    art_cnt: int = 0
    tone_sum: float = 0.0
    tone_n: int = 0
    tone_pos_cnt: int = 0
    topics: set[str] = field(default_factory=set)


@dataclass
class HourStats:
    """Stats for all buckets + ALL in a single hour"""
    buckets: dict[str, BucketStats] = field(default_factory=dict)
    all_stats: BucketStats = field(default_factory=BucketStats)


def load_theme_map() -> dict[str, list[str]]:
    """Load bucket keyword mappings from theme_map.json"""
    if not THEME_MAP_PATH.exists():
        raise FileNotFoundError(f"Theme map not found: {THEME_MAP_PATH}")
    with open(THEME_MAP_PATH, 'r') as f:
        theme_map = json.load(f)
    return theme_map


def load_filter_config() -> Optional[dict]:
    """Load bucket filtering rules from filter_config_v2.json"""
    if not FILTER_CONFIG_PATH.exists():
        return None
    with open(FILTER_CONFIG_PATH, 'r') as f:
        filter_config = json.load(f)
    return filter_config


def detect_tone(value: str) -> Optional[float]:
    """Extract tone value from V2Tone field"""
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
    """Extract theme names from V2Themes field"""
    if not value or ";" not in value:
        return set()
    letters = sum(ch.isalpha() for ch in value)
    if letters / max(len(value), 1) < 0.1:
        return set()
    pieces = [piece.strip() for piece in value.split(";")]
    themes = {piece for piece in pieces if piece}
    return themes


def map_themes_to_buckets(themes: set[str], theme_map: dict[str, list[str]]) -> set[str]:
    """Map article themes to buckets using keyword matching"""
    matched_buckets = set()
    for bucket, keywords in theme_map.items():
        for theme in themes:
            theme_upper = theme.upper()
            for keyword in keywords:
                if keyword.upper() in theme_upper:
                    matched_buckets.add(bucket)
                    break
            if bucket in matched_buckets:
                break
    return matched_buckets


def apply_bucket_filters(
    bucket: str,
    tone: Optional[float],
    themes: set[str],
    filter_config: Optional[dict]
) -> bool:
    """
    Check if article passes filters for a specific bucket.
    Returns True if article should be included in bucket stats.
    """
    if filter_config is None or bucket not in filter_config:
        return True

    bucket_filter = filter_config.get(bucket, {})

    # If bucket_filter is not a dict (e.g., metadata fields), skip filtering
    if not isinstance(bucket_filter, dict):
        return True

    # Apply tone filter
    tone_filter = bucket_filter.get("tone_filter", {})
    if tone_filter.get("enabled", False):
        threshold = tone_filter.get("threshold", 0.0)
        operator = tone_filter.get("operator", "<")

        # If tone is None, article fails the filter
        if tone is None:
            return False

        # Apply threshold based on operator
        if operator == "<":
            if tone >= threshold:
                return False
        elif operator == ">":
            if tone <= threshold:
                return False

    # Apply co-occurrence filter
    co_occurrence = bucket_filter.get("co_occurrence", {})
    if co_occurrence.get("enabled", False):
        required_keywords = co_occurrence.get("require_any_of", [])

        # Check if any required keyword appears in article themes
        found = False
        for theme in themes:
            theme_upper = theme.upper()
            for keyword in required_keywords:
                if keyword.upper() in theme_upper:
                    found = True
                    break
            if found:
                break

        if not found:
            return False

    return True


def extract_fields(row: list[str], theme_map: dict[str, list[str]]) -> tuple[Optional[float], set[str], set[str]]:
    """Extract tone, themes, and matched buckets from GKG row"""
    tone = None
    themes: set[str] = set()

    for field in row:
        if tone is None:
            tone = detect_tone(field)
        if not themes:
            candidate = detect_themes(field)
            if candidate:
                themes = candidate
        if tone is not None and themes:
            break

    buckets = map_themes_to_buckets(themes, theme_map) if themes else set()
    return tone, themes, buckets


def parse_zip(path: Path, expected_ts: datetime, theme_map: dict[str, list[str]]) -> list[tuple[datetime, Optional[float], set[str], set[str]]]:
    """Parse a GKG zip file and extract records with bucket mappings"""
    try:
        with zipfile.ZipFile(path) as zf:
            inner_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            inner = inner_names[0] if inner_names else zf.namelist()[0]
            with zf.open(inner) as fh:
                rows: list[tuple[datetime, Optional[float], set[str], set[str]]] = []
                for line in fh:
                    try:
                        decoded = line.decode("utf-8", errors="ignore").rstrip("\n")
                    except UnicodeDecodeError:
                        continue
                    columns = decoded.split("\t")
                    tone, themes, buckets = extract_fields(columns, theme_map)
                    ts = expected_ts.replace(minute=0, second=0, microsecond=0)
                    rows.append((ts, tone, themes, buckets))
    except (zipfile.BadZipFile, FileNotFoundError, IndexError, KeyError) as exc:
        warnings.warn(f"Failed to parse {path}: {exc}")
        return []
    return rows


def build_jobs_from_raw(raw_dir: Path, start: datetime, end: datetime) -> list[Job]:
    """Build job list from existing RAW GKG files"""
    jobs = []
    delta = timedelta(minutes=15)
    current = start
    while current <= end:
        stamp = current.strftime("%Y%m%d%H%M%S")
        path = raw_dir / current.strftime("%Y/%m") / f"{stamp}.gkg.csv.zip"
        if path.exists():
            jobs.append(Job(timestamp=current, path=path))
        current += delta
    return jobs


def process_month(jobs: list[Job], theme_map: dict[str, list[str]], filter_config: Optional[dict] = None) -> pd.DataFrame:
    """Process all jobs for a month and aggregate to hourly bucket features"""
    stats: defaultdict[datetime, HourStats] = defaultdict(HourStats)

    for job in tqdm(jobs, desc="Processing files", unit="file"):
        if not job.path.exists():
            continue

        records = parse_zip(job.path, job.timestamp, theme_map)
        if not records:
            continue

        for ts, tone, themes, matched_buckets in records:
            hour_stats = stats[ts]

            # Update ALL bucket stats (no filtering)
            hour_stats.all_stats.art_cnt += 1
            if tone is not None:
                hour_stats.all_stats.tone_sum += tone
                hour_stats.all_stats.tone_n += 1
                if tone > 0:
                    hour_stats.all_stats.tone_pos_cnt += 1
            if themes:
                hour_stats.all_stats.topics.update(themes)

            # Update per-bucket stats (with filtering)
            for bucket in matched_buckets:
                # Apply bucket-specific filters
                if not apply_bucket_filters(bucket, tone, themes, filter_config):
                    continue

                if bucket not in hour_stats.buckets:
                    hour_stats.buckets[bucket] = BucketStats()
                bucket_stats = hour_stats.buckets[bucket]
                bucket_stats.art_cnt += 1
                if tone is not None:
                    bucket_stats.tone_sum += tone
                    bucket_stats.tone_n += 1
                    if tone > 0:
                        bucket_stats.tone_pos_cnt += 1
                if themes:
                    bucket_stats.topics.update(themes)

    # Build output DataFrame
    rows = []
    for ts in sorted(stats.keys()):
        hour_stats = stats[ts]
        row = {"ts_utc": ts}

        # ALL bucket (raw stats)
        all_st = hour_stats.all_stats
        tone_avg = float(all_st.tone_sum / all_st.tone_n) if all_st.tone_n else np.nan
        tone_pos_ratio = float(all_st.tone_pos_cnt / all_st.tone_n) if all_st.tone_n else np.nan
        row["ALL_art_cnt"] = int(all_st.art_cnt)
        row["ALL_tone_avg"] = tone_avg
        row["ALL_tone_pos_ratio"] = tone_pos_ratio
        row["ALL_topic_cnt"] = int(len(all_st.topics))

        # Per-bucket raw stats
        for bucket in BUCKETS:
            bucket_st = hour_stats.buckets.get(bucket, BucketStats())
            b_tone_avg = float(bucket_st.tone_sum / bucket_st.tone_n) if bucket_st.tone_n else np.nan
            b_tone_pos_ratio = float(bucket_st.tone_pos_cnt / bucket_st.tone_n) if bucket_st.tone_n else np.nan
            row[f"{bucket}_art_cnt"] = int(bucket_st.art_cnt)
            row[f"{bucket}_tone_avg"] = b_tone_avg
            row[f"{bucket}_tone_pos_ratio"] = b_tone_pos_ratio
            row[f"{bucket}_topic_cnt"] = int(len(bucket_st.topics))

        # Apply 1/k normalization
        k = len([b for b in BUCKETS if hour_stats.buckets.get(b, BucketStats()).art_cnt > 0])
        k = max(k, 1)  # Avoid division by zero

        # Normalized ALL
        row["ALL_norm_topic_cnt"] = row["ALL_topic_cnt"] / k

        # Normalized per-bucket
        total_mapped_art_cnt = sum(hour_stats.buckets.get(b, BucketStats()).art_cnt for b in BUCKETS)
        for bucket in BUCKETS:
            bucket_st = hour_stats.buckets.get(bucket, BucketStats())
            row[f"{bucket}_norm_art_cnt"] = bucket_st.art_cnt / k
            row[f"{bucket}_norm_tone_avg"] = (bucket_st.tone_sum / bucket_st.tone_n / k) if bucket_st.tone_n else np.nan
            row[f"{bucket}_norm_tone_pos_ratio"] = (bucket_st.tone_pos_cnt / bucket_st.tone_n / k) if bucket_st.tone_n else np.nan
            row[f"{bucket}_norm_topic_cnt"] = len(bucket_st.topics) / k

        # Metadata
        row["ALL_art_cnt_from_raw"] = row["ALL_art_cnt"]
        row["ALL_mapped_art_cnt"] = total_mapped_art_cnt
        row["mapped_ratio"] = total_mapped_art_cnt / row["ALL_art_cnt"] if row["ALL_art_cnt"] > 0 else 0.0

        # OTHER bucket (articles not mapped to any bucket)
        other_art_cnt = row["ALL_art_cnt"] - total_mapped_art_cnt
        row["OTHER_norm_art_cnt"] = other_art_cnt / k
        row["OTHER_norm_tone_avg"] = np.nan  # Can't calculate tone for OTHER
        row["OTHER_norm_tone_pos_ratio"] = np.nan

        rows.append(row)

    # Create DataFrame with proper column order (60 columns matching Oct 2025 structure)
    df = pd.DataFrame(rows)

    # Ensure proper column order
    col_order = ["ts_utc"]
    # ALL raw
    col_order += ["ALL_art_cnt", "ALL_tone_avg", "ALL_tone_pos_ratio", "ALL_topic_cnt"]
    # Bucket raw
    for bucket in BUCKETS:
        col_order += [f"{bucket}_art_cnt", f"{bucket}_tone_avg", f"{bucket}_tone_pos_ratio", f"{bucket}_topic_cnt"]
    # ALL norm
    col_order += ["ALL_norm_topic_cnt"]
    # Bucket norm
    for bucket in BUCKETS:
        col_order += [f"{bucket}_norm_art_cnt", f"{bucket}_norm_tone_avg", f"{bucket}_norm_tone_pos_ratio", f"{bucket}_norm_topic_cnt"]
    # Metadata
    col_order += ["ALL_art_cnt_from_raw", "ALL_mapped_art_cnt", "mapped_ratio"]
    # OTHER
    col_order += ["OTHER_norm_art_cnt", "OTHER_norm_tone_avg", "OTHER_norm_tone_pos_ratio"]

    # Reorder columns
    df = df[col_order]

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--raw-dir", default="data/gdelt_raw", help="Directory containing raw GKG files")
    parser.add_argument("--out-dir", default="data/gdelt_hourly_monthly", help="Output directory for monthly parquet files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load theme map
    print(f"[INFO] Loading theme map from {THEME_MAP_PATH}")
    theme_map = load_theme_map()
    print(f"[INFO] Loaded {len(theme_map)} bucket mappings: {list(theme_map.keys())}")

    # Load filter config (v2 precision filtering)
    filter_config = load_filter_config()
    if filter_config:
        print(f"[INFO] Loaded filter config from {FILTER_CONFIG_PATH}")
        print(f"[INFO] Filter config version: {filter_config.get('version', 'unknown')}")
        enabled_filters = [b for b in BUCKETS if filter_config.get(b, {}).get('tone_filter', {}).get('enabled') or filter_config.get(b, {}).get('co_occurrence', {}).get('enabled')]
        print(f"[INFO] Buckets with filters enabled: {enabled_filters}")
    else:
        print(f"[INFO] No filter config found, using v1 keyword-only filtering")

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process month by month
    current_month = start_date.replace(day=1)
    end_month = end_date.replace(day=1)

    while current_month <= end_month:
        # Get month boundaries
        month_start = current_month
        if current_month.month == 12:
            month_end = current_month.replace(year=current_month.year + 1, month=1, day=1) - timedelta(seconds=1)
        else:
            month_end = current_month.replace(month=current_month.month + 1, day=1) - timedelta(seconds=1)

        # Clip to user-specified range
        month_start = max(month_start, start_date)
        month_end = min(month_end, end_date)

        month_str = current_month.strftime("%Y-%m")
        out_path = out_dir / f"gdelt_hourly_{month_str}.parquet"

        print(f"\n[INFO] Processing {month_str} ({month_start.date()} to {month_end.date()})")

        # Build jobs for this month
        jobs = build_jobs_from_raw(raw_dir, month_start, month_end)
        print(f"[INFO] Found {len(jobs)} RAW GKG files for {month_str}")

        if not jobs:
            print(f"[WARN] No RAW files found for {month_str}, skipping")
            # Move to next month
            if current_month.month == 12:
                current_month = current_month.replace(year=current_month.year + 1, month=1)
            else:
                current_month = current_month.replace(month=current_month.month + 1)
            continue

        # Process and save
        df = process_month(jobs, theme_map, filter_config)
        df.to_parquet(out_path, index=False)
        print(f"[INFO] Wrote {len(df)} hourly rows to {out_path}")
        print(f"[INFO] Columns: {len(df.columns)}, Sample mapped_ratio: {df['mapped_ratio'].mean():.2%}")

        # Move to next month
        if current_month.month == 12:
            current_month = current_month.replace(year=current_month.year + 1, month=1)
        else:
            current_month = current_month.replace(month=current_month.month + 1)

    print("\n[INFO] Backfill complete!")


if __name__ == "__main__":
    main()
