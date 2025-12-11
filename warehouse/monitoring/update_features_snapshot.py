#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
update_features_snapshot.py

用途：
- 從 CSV 來源讀取最新的小時特徵資料
- 清理 / 轉型後，寫成 parquet：features_hourly_with_term.parquet
- 給 hourly_monitor.py 使用

假設：
- CSV 來源：warehouse/features_hourly_v2.csv
- parquet 輸出：/home/ec2-user/Data/features_hourly_with_term.parquet
"""

import os
import sys
from typing import Optional, List

import pandas as pd

# --- 基本路徑設定 -----------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE_DIR = PROJECT_ROOT
WAREHOUSE_DIR = os.path.join(BASE_DIR, "warehouse")

CSV_SOURCE = os.path.join(WAREHOUSE_DIR, "features_hourly_v2.csv")
PARQUET_OUTPUT = os.path.join(BASE_DIR, "features_hourly_with_term.parquet")

# hourly_monitor 期望看到但 CSV 可能沒有的欄位
REQUIRED_EXTRA_COLS: List[str] = [
    "OIL_CORE_norm_art_cnt",
    "GEOPOL_norm_art_cnt",
    "USD_RATE_norm_art_cnt",
    "SUPPLY_CHAIN_norm_art_cnt",
    "MACRO_norm_art_cnt",
    "cl1_cl2",
    "ovx",
]


# --- 工具函式：自動找出時間欄位 -------------------------------------------

def pick_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """
    優先順序：
    1. 明確名稱匹配：timestamp / as_of_utc / as_of / time / datetime / date ...
    2. 名稱包含關鍵字：timestamp / datetime / date / time
    3. 嘗試對每個欄位做 to_datetime，挑「可解析比例最高」的那一欄
    """
    # 1) 明確候選
    primary_candidates: List[str] = [
        "timestamp",
        "as_of_utc",
        "as_of",
        "time",
        "datetime",
        "datetime_utc",
        "dt",
        "bar_time",
        "bar_timestamp",
        "bar_end_time",
        "trade_time",
        "date",
        "ts",  # 你現在 CSV 用的欄位
    ]

    cols_lower = {c.lower(): c for c in df.columns}

    # 1. 直接全名匹配
    for name in primary_candidates:
        lower = name.lower()
        if lower in cols_lower:
            print(f"[INFO] 以欄位名稱直接匹配到時間欄位：{cols_lower[lower]}")
            return cols_lower[lower]

    # 2. 名稱包含關鍵字
    keywords = ["timestamp", "datetime", "date", "time"]
    for kw in keywords:
        for c in df.columns:
            if kw.lower() in c.lower():
                print(f"[INFO] 以關鍵字 '{kw}' 匹配到時間欄位：{c}")
                return c

    # 3. 嘗試對每個欄位做 to_datetime，看哪一欄「可解析比例最高」
    best_col = None
    best_ratio = 0.0

    sample_df = df.head(500)  # 不用全部，取前 500 筆加速

    for c in df.columns:
        try:
            parsed = pd.to_datetime(sample_df[c], errors="coerce", utc=True)
            ok_ratio = parsed.notna().mean()
        except Exception:
            continue

        if ok_ratio > best_ratio:
            best_ratio = ok_ratio
            best_col = c

    # 要求至少有一半以上的值可以被當成時間，才認為是時間欄位
    if best_col is not None and best_ratio >= 0.5:
        print(f"[INFO] 透過 to_datetime 推斷時間欄位：{best_col} (可解析比例 {best_ratio:.2%})")
        return best_col

    return None


# --- 主流程 -----------------------------------------------------------------

def main():
    print(f"[INFO] 專案根目錄：{BASE_DIR}")
    print(f"[INFO] 來源 CSV：{CSV_SOURCE}")
    print(f"[INFO] 輸出 parquet：{PARQUET_OUTPUT}")

    if not os.path.exists(CSV_SOURCE):
        print(f"[ERROR] 找不到來源檔案：{CSV_SOURCE}")
        sys.exit(1)

    df = pd.read_csv(CSV_SOURCE)
    print(f"[INFO] CSV 讀入完成，欄位：{list(df.columns)}")
    if df.empty:
        print("[ERROR] 來源 CSV 沒有資料，停止更新。")
        sys.exit(1)

    # 1) 找時間欄位
    ts_col = pick_timestamp_column(df)
    if ts_col is None:
        print("[ERROR] 無法從欄位名稱 / 內容自動判斷時間欄位。")
        print("[ERROR] 請確認 CSV 至少有一個包含 datetime 的欄位，或改名成 'timestamp'.")
        sys.exit(1)

    print(f"[INFO] 使用時間欄位：{ts_col}")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col])

    if df.empty:
        print("[ERROR] 轉型後沒有有效 timestamp，停止更新。")
        sys.exit(1)

    # 2) 補上 hourly_monitor 期望但來源沒有的欄位
    for col in REQUIRED_EXTRA_COLS:
        if col not in df.columns:
            if col == "ovx" and "ovx_q" in df.columns:
                print(f"[WARN] CSV 缺少欄位 {col}，以 ovx_q 的數值複製補上。")
                df[col] = df["ovx_q"]
            else:
                print(f"[WARN] CSV 缺少欄位 {col}，以常數 0.0 補上。")
                df[col] = 0.0

    # 3) 去重 + 排序
    df = df.drop_duplicates(subset=[ts_col]).sort_values(ts_col)

    # 4) 寫 parquet
    df.to_parquet(PARQUET_OUTPUT, index=False)
    print(f"[INFO] 已更新 parquet：{PARQUET_OUTPUT}")
    print(f"[INFO] 共 {len(df)} 筆列，時間範圍：{df[ts_col].min()}  ~  {df[ts_col].max()}")


if __name__ == "__main__":
    main()
