#!/usr/bin/env python3
"""
Enhanced GDELT GKG bucket aggregator with partial-window support.

Key features:
- Accepts date-only or datetime start/end (UTC, ISO-like):
  * YYYY-MM-DD           -> whole day (00:00:00 ~ 23:59:59)
  * YYYY-MM-DDTHH:MM     -> from that minute
- Aggregates only the required 15-minute RAW files in the window (no full-month sweep).
- Writes/updates monthly parquet in out-dir, merging by ts (hourly) without rewriting the whole month.
- Schema compatible with existing 60-column bucket layout.

Examples:
  python3 gdelt_gkg_bucket_aggregator.py \
    --start 2025-11-03 --end 2025-11-03 \
    --raw-dir data/gdelt_raw --out-dir data/gdelt_hourly_monthly

  python3 gdelt_gkg_bucket_aggregator.py \
    --start 2025-11-03T23:00 --end 2025-11-04T02:00 \
    --raw-dir data/gdelt_raw_tmp --out-dir data/gdelt_hourly_monthly

  # Hourly incremental (cron-friendly)
  python3 gdelt_gkg_bucket_aggregator.py \
    --start 2025-11-03T13:00 --end 2025-11-03T14:00 \
    --raw-dir data/gdelt_raw_tmp --out-dir data/gdelt_hourly_monthly
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import warnings
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------------
# Constants / Config
# ----------------------------
BASE_URL = "http://data.gdeltproject.org/gkg"  # kept for reference; we read from raw-dir
TONE_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?,\d+,\d+,-?\d+(?:\.\d+)?,\d+,\d+$")
THEME_MAP_PATH = Path("warehouse/theme_map.json")
FILTER_CONFIG_PATH = Path("warehouse/filter_config_v2.json")
BUCKETS = ["OIL_CORE", "GEOPOL", "USD_RATE", "SUPPLY_CHAIN", "MACRO", "ESG_POLICY"]
FILENAME_FMT = "%Y%m%d%H%M%S"  # 20251103133000

# ----------------------------
# Data classes
# ----------------------------


@dataclass
class Job:
    timestamp: datetime  # UTC, 15-min slot
    path: Path


@dataclass
class BucketStats:
    art_cnt: int = 0
    tone_sum: float = 0.0
    tone_n: int = 0
    tone_pos_cnt: int = 0
    topics: set[str] = field(default_factory=set)


@dataclass
class HourStats:
    buckets: Dict[str, BucketStats] = field(default_factory=dict)
    all_stats: BucketStats = field(default_factory=BucketStats)


# ----------------------------
# Helpers: parsing & time windows
# ----------------------------


def parse_cli_timestamp(value: str) -> datetime:
    """
    Parse CLI timestamp:
    - YYYY-MM-DD           -> date at 00:00 UTC
    - YYYY-MM-DDTHH:MM     -> datetime minute precision, UTC
    Raises ValueError on invalid format.
    """
    formats = ("%Y-%m-%dT%H:%M", "%Y-%m-%d")
    last_err = None
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError as exc:
            last_err = exc
    raise ValueError(f"Invalid date/time format '{value}'. Supported: YYYY-MM-DD or YYYY-MM-DDTHH:MM") from last_err


def floor_to_15(dt: datetime) -> datetime:
    minute = (dt.minute // 15) * 15
    return dt.replace(minute=minute, second=0, microsecond=0)


def normalize_window(start_raw: str, end_raw: str) -> Tuple[datetime, datetime, datetime, datetime]:
    """
    Return (start_dt, end_dt, start_slot, end_slot) in UTC.
    Rules:
    - Date-only end -> treated as end of that day 23:59:59 (so last slot 23:45).
    - Date-only start -> 00:00 of that day.
    - Time-aware values honored as-is.
    - Slots are floored to 15m; end_slot inclusive.
    - Error if end < start.
    """
    start_dt = parse_cli_timestamp(start_raw)
    end_dt = parse_cli_timestamp(end_raw)

    # If end provided without time, it was parsed at 00:00; move to end of day
    if "T" not in end_raw:
        end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=0)
    if "T" not in start_raw:
        start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    if end_dt < start_dt:
        raise ValueError(f"end ({end_dt}) must be >= start ({start_dt})")

    start_slot = floor_to_15(start_dt)
    end_slot = floor_to_15(end_dt)
    if end_slot < start_slot:
        raise ValueError("End slot earlier than start slot after 15m alignment")

    logging.info("Window parsed: start=%s end=%s (UTC)", start_dt.isoformat(), end_dt.isoformat())
    logging.info("Aligned slots : start_slot=%s end_slot=%s (UTC, 15m, inclusive)", start_slot.isoformat(), end_slot.isoformat())
    return start_dt, end_dt, start_slot, end_slot


def month_key(ts: datetime) -> str:
    return ts.strftime("%Y-%m")


def iter_quarter_hours(start_slot: datetime, end_slot: datetime) -> Iterator[datetime]:
    cur = start_slot
    delta = timedelta(minutes=15)
    while cur <= end_slot:
        yield cur
        cur += delta


def build_jobs_by_month(raw_dir: Path, start_slot: datetime, end_slot: datetime) -> Tuple[Dict[str, List[Job]], int, int, Optional[datetime], Optional[datetime]]:
    """
    Build jobs grouped by month for the requested window.
    Returns (jobs_by_month, total_slots, available_slots, min_found_ts, max_found_ts)
    """
    jobs: Dict[str, List[Job]] = defaultdict(list)
    total = 0
    available = 0
    min_found = None
    max_found = None

    for ts in iter_quarter_hours(start_slot, end_slot):
        total += 1
        stamp = ts.strftime(FILENAME_FMT)
        path = raw_dir / f"{ts.year:04d}" / f"{ts.month:02d}" / f"{stamp}.gkg.csv.zip"
        jobs[month_key(ts)].append(Job(timestamp=ts, path=path))
        if path.exists():
            available += 1
            min_found = ts if min_found is None else min(min_found, ts)
            max_found = ts if max_found is None else max(max_found, ts)

    logging.info("Quarter-hour slots requested: %s, available files: %s", total, available)
    if available:
        logging.info("Available RAW ts range: %s -> %s (UTC)", min_found.isoformat(), max_found.isoformat())
    else:
        logging.warning("No RAW files found in the requested window.")
    return jobs, total, available, min_found, max_found


# ----------------------------
# Helpers: bucket mapping
# ----------------------------


def load_theme_map() -> dict[str, list[str]]:
    if not THEME_MAP_PATH.exists():
        raise FileNotFoundError(f"Theme map not found: {THEME_MAP_PATH}")
    with open(THEME_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_filter_config() -> Optional[dict]:
    if not FILTER_CONFIG_PATH.exists():
        return None
    with open(FILTER_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_tone(value: str) -> Optional[float]:
    if value is None:
        return None
    if not TONE_PATTERN.match(value.strip()):
        return None
    parts = value.split(",")
    try:
        return float(parts[0])
    except ValueError:
        return None


def detect_themes(value: str) -> set[str]:
    if not value or ";" not in value:
        return set()
    letters = sum(ch.isalpha() for ch in value)
    if letters / max(len(value), 1) < 0.1:
        return set()
    return {piece.strip() for piece in value.split(";") if piece.strip()}


def map_themes_to_buckets(themes: set[str], theme_map: dict[str, list[str]]) -> set[str]:
    matched = set()
    for bucket, keywords in theme_map.items():
        for theme in themes:
            theme_upper = theme.upper()
            if any(keyword.upper() in theme_upper for keyword in keywords):
                matched.add(bucket)
                break
    return matched


def apply_bucket_filters(bucket: str, tone: Optional[float], themes: set[str], filter_config: Optional[dict]) -> bool:
    if filter_config is None or bucket not in filter_config or not isinstance(filter_config.get(bucket), dict):
        return True
    cfg = filter_config[bucket]

    tone_filter = cfg.get("tone_filter", {})
    if tone_filter.get("enabled") and tone is not None:
        threshold = tone_filter.get("threshold", 0.0)
        op = tone_filter.get("operator", "<")
        if op == "<" and tone >= threshold:
            return False
        if op == ">" and tone <= threshold:
            return False
        if tone is None:
            return False

    co = cfg.get("co_occurrence", {})
    if co.get("enabled"):
        required = co.get("require_any_of", [])
        found = any(any(req.upper() in t.upper() for req in required) for t in themes)
        if not found:
            return False
    return True


# ----------------------------
# Parsing and aggregation
# ----------------------------


def extract_fields(row: list[str], theme_map: dict[str, list[str]]) -> tuple[Optional[float], set[str], set[str]]:
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
        logging.warning("Failed to parse %s: %s", path, exc)
        return []
    return rows


def process_jobs(jobs: List[Job], theme_map: dict[str, list[str]], filter_config: Optional[dict]) -> Tuple[pd.DataFrame, int]:
    stats: defaultdict[datetime, HourStats] = defaultdict(HourStats)
    parsed_files = 0
    for job in tqdm(jobs, desc="Processing files", unit="file"):
        if not job.path.exists():
            continue
        records = parse_zip(job.path, job.timestamp, theme_map)
        if not records:
            continue
        parsed_files += 1
        for ts, tone, themes, matched_buckets in records:
            hour_stats = stats[ts]
            # ALL
            hour_stats.all_stats.art_cnt += 1
            if tone is not None:
                hour_stats.all_stats.tone_sum += tone
                hour_stats.all_stats.tone_n += 1
                if tone > 0:
                    hour_stats.all_stats.tone_pos_cnt += 1
            if themes:
                hour_stats.all_stats.topics.update(themes)
            # buckets
            for bucket in matched_buckets:
                if not apply_bucket_filters(bucket, tone, themes, filter_config):
                    continue
                bucket_st = hour_stats.buckets.setdefault(bucket, BucketStats())
                bucket_st.art_cnt += 1
                if tone is not None:
                    bucket_st.tone_sum += tone
                    bucket_st.tone_n += 1
                    if tone > 0:
                        bucket_st.tone_pos_cnt += 1
                if themes:
                    bucket_st.topics.update(themes)

    rows = []
    for ts in sorted(stats.keys()):
        hour_stats = stats[ts]
        row: Dict[str, object] = {"ts_utc": ts}
        all_st = hour_stats.all_stats
        tone_avg = float(all_st.tone_sum / all_st.tone_n) if all_st.tone_n else np.nan
        tone_pos_ratio = float(all_st.tone_pos_cnt / all_st.tone_n) if all_st.tone_n else np.nan
        row["ALL_art_cnt"] = int(all_st.art_cnt)
        row["ALL_tone_avg"] = tone_avg
        row["ALL_tone_pos_ratio"] = tone_pos_ratio
        row["ALL_topic_cnt"] = int(len(all_st.topics))

        for bucket in BUCKETS:
            bucket_st = hour_stats.buckets.get(bucket, BucketStats())
            b_tone_avg = float(bucket_st.tone_sum / bucket_st.tone_n) if bucket_st.tone_n else np.nan
            b_tone_pos_ratio = float(bucket_st.tone_pos_cnt / bucket_st.tone_n) if bucket_st.tone_n else np.nan
            row[f"{bucket}_art_cnt"] = int(bucket_st.art_cnt)
            row[f"{bucket}_tone_avg"] = b_tone_avg
            row[f"{bucket}_tone_pos_ratio"] = b_tone_pos_ratio
            row[f"{bucket}_topic_cnt"] = int(len(bucket_st.topics))

        k = len([b for b in BUCKETS if hour_stats.buckets.get(b, BucketStats()).art_cnt > 0])
        k = max(k, 1)
        total_mapped_art_cnt = sum(hour_stats.buckets.get(b, BucketStats()).art_cnt for b in BUCKETS)

        row["ALL_norm_topic_cnt"] = row["ALL_topic_cnt"] / k
        for bucket in BUCKETS:
            bucket_st = hour_stats.buckets.get(bucket, BucketStats())
            row[f"{bucket}_norm_art_cnt"] = bucket_st.art_cnt / k
            row[f"{bucket}_norm_tone_avg"] = (bucket_st.tone_sum / bucket_st.tone_n / k) if bucket_st.tone_n else np.nan
            row[f"{bucket}_norm_tone_pos_ratio"] = (bucket_st.tone_pos_cnt / bucket_st.tone_n / k) if bucket_st.tone_n else np.nan
            row[f"{bucket}_norm_topic_cnt"] = len(bucket_st.topics) / k

        row["ALL_art_cnt_from_raw"] = row["ALL_art_cnt"]
        row["ALL_mapped_art_cnt"] = total_mapped_art_cnt
        row["mapped_ratio"] = total_mapped_art_cnt / row["ALL_art_cnt"] if row["ALL_art_cnt"] else 0.0

        other_art_cnt = row["ALL_art_cnt"] - total_mapped_art_cnt
        row["OTHER_norm_art_cnt"] = other_art_cnt / k
        row["OTHER_norm_tone_avg"] = np.nan
        row["OTHER_norm_tone_pos_ratio"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, parsed_files


def column_order() -> List[str]:
    cols = ["ts_utc"]
    cols += ["ALL_art_cnt", "ALL_tone_avg", "ALL_tone_pos_ratio", "ALL_topic_cnt"]
    for bucket in BUCKETS:
        cols += [f"{bucket}_art_cnt", f"{bucket}_tone_avg", f"{bucket}_tone_pos_ratio", f"{bucket}_topic_cnt"]
    cols += ["ALL_norm_topic_cnt"]
    for bucket in BUCKETS:
        cols += [f"{bucket}_norm_art_cnt", f"{bucket}_norm_tone_avg", f"{bucket}_norm_tone_pos_ratio", f"{bucket}_norm_topic_cnt"]
    cols += ["ALL_art_cnt_from_raw", "ALL_mapped_art_cnt", "mapped_ratio"]
    cols += ["OTHER_norm_art_cnt", "OTHER_norm_tone_avg", "OTHER_norm_tone_pos_ratio"]
    return cols


# ----------------------------
# IO helpers
# ----------------------------


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[cols]


def merge_month_parquet(out_dir: Path, month_key_str: str, df_new: pd.DataFrame, cols: List[str]) -> None:
    out_path = out_dir / f"gdelt_hourly_{month_key_str}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_new = ensure_columns(df_new, cols)
    df_new["ts_utc"] = pd.to_datetime(df_new["ts_utc"], utc=True, errors="coerce")
    df_new = df_new.dropna(subset=["ts_utc"]).sort_values("ts_utc")

    if out_path.exists():
        df_old = pd.read_parquet(out_path)
        ts_col = "ts_utc" if "ts_utc" in df_old.columns else df_old.columns[0]
        df_old["ts_utc"] = pd.to_datetime(df_old[ts_col], utc=True, errors="coerce")
        df_old = ensure_columns(df_old, cols)
        df_old = df_old.dropna(subset=["ts_utc"])
        old_ts = set(df_old["ts_utc"])
        new_ts = set(df_new["ts_utc"])
        replaced = len(old_ts & new_ts)
        added = len(new_ts - old_ts)
        combined = pd.concat([df_old, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts_utc"], keep="last").sort_values("ts_utc")
        tmp_path = out_path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp_path, index=False)
        tmp_path.replace(out_path)
        logging.info(
            "[%s] updated parquet: total=%d (added=%d, replaced=%d) -> %s",
            month_key_str,
            len(combined),
            added,
            replaced,
            out_path,
        )
    else:
        tmp_path = out_path.with_suffix(".parquet.tmp")
        df_new.to_parquet(tmp_path, index=False)
        tmp_path.replace(out_path)
        logging.info("[%s] wrote new parquet: rows=%d -> %s", month_key_str, len(df_new), out_path)


# ----------------------------
# CLI
# ----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True, help="Start time (UTC) YYYY-MM-DD or YYYY-MM-DDTHH:MM")
    p.add_argument("--end", required=True, help="End time (UTC) YYYY-MM-DD or YYYY-MM-DDTHH:MM")
    p.add_argument("--raw-dir", default="data/gdelt_raw", help="Root dir of GKG RAW (YYYY/MM/*.gkg.csv.zip)")
    p.add_argument("--out-dir", default="data/gdelt_hourly_monthly", help="Output dir for monthly parquet")
    p.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    start_dt, end_dt, start_slot, end_slot = normalize_window(args.start, args.end)
    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    theme_map = load_theme_map()
    filter_config = load_filter_config()
    if filter_config:
        logging.info("Loaded filter config: %s", FILTER_CONFIG_PATH)

    jobs_by_month, total_slots, available_slots, min_found, max_found = build_jobs_by_month(raw_dir, start_slot, end_slot)
    if available_slots == 0:
        logging.warning("No RAW files to process in the requested window; exiting without changes.")
        return

    cols = column_order()
    total_parsed = 0
    total_rows_written = 0

    for mk, jobs in sorted(jobs_by_month.items()):
        # Filter jobs to existing files to speed parse; still log missing
        month_existing = [job for job in jobs if job.path.exists()]
        missing = len(jobs) - len(month_existing)
        if missing:
            logging.warning("[%s] Missing RAW files: %d (out of %d slots in window for this month)", mk, missing, len(jobs))

        if not month_existing:
            continue

        df_month, parsed_files = process_jobs(month_existing, theme_map, filter_config)
        total_parsed += parsed_files
        if df_month.empty:
            logging.warning("[%s] No hourly data produced; skipping parquet update.", mk)
            continue
        merge_month_parquet(out_dir, mk, df_month, cols)
        total_rows_written += len(df_month)

    logging.info("Done. Parsed files: %d, hourly rows produced: %d", total_parsed, total_rows_written)


if __name__ == "__main__":
    main()
