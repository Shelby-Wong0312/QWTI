#!/usr/bin/env python3
"""
Download GDELT v2 GKG 15-minute ZIPs for a given time window (UTC) into a raw directory.

Time window semantics (aligned with gdelt_gkg_bucket_aggregator.py):
- Accepts YYYY-MM-DD (whole day 00:00:00~23:59:59) or YYYY-MM-DDTHH:MM (from that minute).
- If end has no time, it is treated as that day's 23:59:59; start without time is 00:00.
- Slots are floored to 15 minutes; end slot is inclusive.

Examples:
  # Whole day
  python jobs/pull_gdelt_http_to_csv.py --from 2025-11-03 --to 2025-11-03 --output data/gdelt_raw_tmp
  # Cross-day window
  python jobs/pull_gdelt_http_to_csv.py --from 2025-11-03T23:00 --to 2025-11-04T02:00 --output data/gdelt_raw_tmp
  # Hourly incremental (cron)
  python jobs/pull_gdelt_http_to_csv.py --from 2025-11-03T13:00 --to 2025-11-03T14:00 --output data/gdelt_raw_tmp
"""
from __future__ import annotations

import argparse
import logging
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = "http://data.gdeltproject.org/gdeltv2"


# ----------------------------
# Time window helpers (aligned with gdelt_gkg_bucket_aggregator.py)
# ----------------------------
def parse_cli_timestamp(value: str) -> datetime:
    """Parse YYYY-MM-DD or YYYY-MM-DDTHH:MM into UTC datetime."""
    formats = ("%Y-%m-%dT%H:%M", "%Y-%m-%d")
    last_err = None
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError as exc:
            last_err = exc
    raise ValueError(f"Invalid date/time '{value}'. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM") from last_err


def floor_to_15(dt: datetime) -> datetime:
    minute = (dt.minute // 15) * 15
    return dt.replace(minute=minute, second=0, microsecond=0)


def normalize_window(start_raw: str, end_raw: str) -> tuple[datetime, datetime, datetime, datetime]:
    """
    Return (start_dt, end_dt, start_slot, end_slot) in UTC.
    - Date-only end -> set to 23:59:59 of that day.
    - Date-only start -> set to 00:00:00 of that day.
    - Slots floored to 15m; end_slot inclusive.
    """
    start_dt = parse_cli_timestamp(start_raw)
    end_dt = parse_cli_timestamp(end_raw)
    if "T" not in end_raw:
        end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=0)
    if "T" not in start_raw:
        start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if end_dt < start_dt:
        raise ValueError(f"--end ({end_dt}) must be >= --start ({start_dt})")

    start_slot = floor_to_15(start_dt)
    end_slot = floor_to_15(end_dt)
    if end_slot < start_slot:
        raise ValueError("end slot earlier than start slot after 15m alignment")
    return start_dt, end_dt, start_slot, end_slot


def iter_slots(start_slot: datetime, end_slot: datetime):
    cur = start_slot
    delta = timedelta(minutes=15)
    while cur <= end_slot:
        yield cur
        cur += delta


# ----------------------------
# IO helpers
# ----------------------------
def fpath(base_dir: Path, dt: datetime) -> Path:
    # base_dir / YYYY / MM / YYYYMMDDHHMMSS.gkg.csv.zip
    return base_dir / dt.strftime("%Y") / dt.strftime("%m") / (dt.strftime("%Y%m%d%H%M%S") + ".gkg.csv.zip")


def is_ok(p: Path) -> bool:
    if (not p.exists()) or p.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(p, "r") as zf:
            bad = zf.testzip()
        return bad is None
    except Exception:
        return False


def dl_one(base_dir: Path, dt: datetime, retries: int = 3):
    """Download a single 15-min slot. Returns (success: bool, reason: 'ok'|'404'|'error')."""
    stamp = dt.strftime("%Y%m%d%H%M%S")
    outp = fpath(base_dir, dt)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".part")
    url = f"{BASE}/{stamp}.gkg.csv.zip"

    for i in range(retries):
        try:
            req = Request(url)
            with urlopen(req, timeout=45) as r:
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 17)
                        if not chunk:
                            break
                        f.write(chunk)
            if outp.exists():
                outp.unlink()
            tmp.replace(outp)
            if is_ok(outp):
                return True, "ok"
            else:
                if outp.exists():
                    outp.unlink()
        except HTTPError as e:
            if e.code == 404:
                if tmp.exists():
                    tmp.unlink()
                return False, "404"
            logging.warning("HTTPError %s for %s (try %d)", e.code, url, i + 1)
            time.sleep(1.5 * (i + 1))
        except URLError as e:
            logging.warning("URLError %s for %s (try %d)", e, url, i + 1)
            time.sleep(1.5 * (i + 1))
        except Exception as e:
            logging.warning("Other error %s for %s (try %d)", e, url, i + 1)
            time.sleep(1.5 * (i + 1))

    if tmp.exists():
        tmp.unlink()
    return False, "error"


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from", dest="from_date", required=True, help="Start (UTC) YYYY-MM-DD or YYYY-MM-DDTHH:MM")
    p.add_argument("--to", dest="to_date", required=True, help="End (UTC) YYYY-MM-DD or YYYY-MM-DDTHH:MM")
    p.add_argument("--output", required=True, help="RAW output dir (e.g., data/gdelt_raw_tmp)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        start_dt, end_dt, start_slot, end_slot = normalize_window(args.from_date, args.to_date)
    except ValueError as exc:
        logging.error("Invalid time window: %s", exc)
        return

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("GDELT RAW output dir: %s", out_dir)
    logging.info("Parsed window (UTC): start=%s end=%s", start_dt.isoformat(), end_dt.isoformat())
    logging.info("Aligned slots (UTC): first=%s last=%s", start_slot.isoformat(), end_slot.isoformat())

    slots = list(iter_slots(start_slot, end_slot))
    total = len(slots)
    logging.info("Total slots: %d (15-min, inclusive)", total)

    downloaded = skipped_ok = errors = missing404 = 0

    for i, dt in enumerate(slots, start=1):
        p = fpath(out_dir, dt)
        if is_ok(p):
            skipped_ok += 1
        else:
            ok, reason = dl_one(out_dir, dt)
            if ok:
                downloaded += 1
            else:
                if reason == "404":
                    missing404 += 1
                else:
                    errors += 1

        if i % 50 == 0 or i == total:
            logging.info(
                "Progress %d/%d dl=%d ok=%d 404=%d err=%d",
                i, total, downloaded, skipped_ok, missing404, errors
            )

    logging.info(
        "DONE download: slots=%d downloaded=%d skipped_ok=%d 404=%d errors=%d",
        total, downloaded, skipped_ok, missing404, errors,
    )


if __name__ == "__main__":
    main()
