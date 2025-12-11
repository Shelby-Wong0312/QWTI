#!/usr/bin/env python
import argparse
import datetime as dt
import logging
import re
from pathlib import Path
from typing import List

import zipfile
import pandas as pd

logger = logging.getLogger(__name__)

FILENAME_RE = re.compile(r"(\d{14})\.gkg\.csv\.zip$")  # 20251101120000.gkg.csv.zip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate GDELT GKG raw (15-min) into hourly monthly parquet (簡化版：只算 art_cnt，tone 用預設值)."
    )
    p.add_argument("--year-month", required=True, help="YYYY-MM, e.g. 2025-11")
    p.add_argument("--raw-dir", required=True, help="Root dir of gdelt_raw (YYYY/MM/...).")
    p.add_argument(
        "--out-parquet",
        required=True,
        help="Output parquet path, e.g. data/gdelt_hourly_monthly/gdelt_hourly_2025-11.parquet",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def find_month_files(raw_root: Path, year: int, month: int) -> List[Path]:
    """在 raw_root/YYYY/MM/ 底下找該月所有 .gkg.csv.zip"""
    y_dir = raw_root / f"{year:04d}" / f"{month:02d}"
    if not y_dir.exists():
        logger.warning("Raw dir for %04d-%02d not found: %s", year, month, y_dir)
        return []
    files = sorted(p for p in y_dir.glob("*.gkg.csv.zip") if FILENAME_RE.search(p.name))
    logger.info("Found %d raw files under %s", len(files), y_dir)
    return files


def extract_ts_from_filename(path: Path) -> dt.datetime:
    m = FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse timestamp from filename: {path}")
    ts_str = m.group(1)  # 20251101120000
    return dt.datetime.strptime(ts_str, "%Y%m%d%H%M%S")


def count_rows_in_zip(path: Path) -> int:
    """讀一個 .gkg.csv.zip，回傳裡面 CSV 的列數（文章數）"""
    with zipfile.ZipFile(path, "r") as zf:
        inner_name = None
        for name in zf.namelist():
            if name.endswith(".gkg.csv"):
                inner_name = name
                break
        if inner_name is None:
            logger.warning("No .gkg.csv inside %s, treat as 0 rows", path)
            return 0
        with zf.open(inner_name) as f:
            # 不讀完整內容，只是為了算列數
            # low_memory=False 是為了避免 dtype infer 警告
            df = pd.read_csv(f, sep="\t", header=None, low_memory=False)
    return len(df)


def aggregate_files_for_hour(files: List[Path]) -> pd.DataFrame:
    """
    簡化版聚合邏輯：
      - 每個 15 分鐘檔案：算出 (ts, art_cnt_file)
      - 其中 ts 從檔名解析（精確到分鐘）
      - 再把 ts.floor('H') 當成 hour key groupby：
          ALL_art_cnt = sum(art_cnt_file)
          ALL_tone_avg = 0.0（預設）
          ALL_tone_pos_ratio = 0.5（預設）
    """
    if not files:
        return pd.DataFrame(
            columns=["ts", "ALL_art_cnt", "ALL_tone_avg", "ALL_tone_pos_ratio"]
        )

    records = []
    for p in files:
        try:
            ts = extract_ts_from_filename(p)
            art_cnt = count_rows_in_zip(p)
            records.append({"ts": ts, "art_cnt_file": art_cnt})
        except Exception as e:
            logger.warning("Failed to process %s: %s", p, e)

    if not records:
        return pd.DataFrame(
            columns=["ts", "ALL_art_cnt", "ALL_tone_avg", "ALL_tone_pos_ratio"]
        )

    df = pd.DataFrame.from_records(records)
    df["hour"] = df["ts"].dt.floor("H")

    grp = (
        df.groupby("hour")
        .agg(ALL_art_cnt=("art_cnt_file", "sum"))
        .reset_index()
        .rename(columns={"hour": "ts"})
    )

    # 簡化：tone 用預設值
    grp["ALL_tone_avg"] = 0.0
    grp["ALL_tone_pos_ratio"] = 0.5

    grp["ts"] = pd.to_datetime(grp["ts"])
    return grp


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ym = args.year_month
    year, month = map(int, ym.split("-"))
    raw_root = Path(args.raw_dir).resolve()
    out_parquet = Path(args.out_parquet).resolve()
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building hourly monthly parquet for %s", ym)
    logger.info("RAW root    : %s", raw_root)
    logger.info("OUT parquet : %s", out_parquet)

    files = find_month_files(raw_root, year, month)
    df_hourly = aggregate_files_for_hour(files)

    if df_hourly.empty:
        logger.warning("No data aggregated for %s, writing empty parquet", ym)

    df_hourly = df_hourly.sort_values("ts")
    df_hourly.to_parquet(out_parquet, index=False)
    logger.info("Wrote %s rows to %s", len(df_hourly), out_parquet)


if __name__ == "__main__":
    main()