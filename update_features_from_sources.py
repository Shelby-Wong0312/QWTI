#!/usr/bin/env python
"""
Merge GDELT hourly + WTI hourly -> features_hourly_with_term.parquet
每小時自動跑，不需人工介入
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

DATA_DIR = Path("/home/ec2-user/Data/data")
GDELT_PATH = DATA_DIR / "gdelt_hourly.parquet"
WTI_PATH = DATA_DIR / "wti_hourly_capital.parquet"
OUTPUT_PATH = DATA_DIR / "features_hourly_with_term.parquet"


def main():
    logging.info("=== update_features_from_sources ===")

    # 1. 讀 GDELT hourly
    if not GDELT_PATH.exists():
        raise SystemExit(f"[ERROR] GDELT not found: {GDELT_PATH}")
    gdelt = pd.read_parquet(GDELT_PATH)
    gdelt["ts_utc"] = pd.to_datetime(gdelt["ts_utc"], utc=True)
    logging.info("GDELT rows=%d, ts=%s -> %s", len(gdelt), gdelt["ts_utc"].min(), gdelt["ts_utc"].max())

    # 2. 讀 WTI hourly
    if not WTI_PATH.exists():
        raise SystemExit(f"[ERROR] WTI not found: {WTI_PATH}")
    wti = pd.read_parquet(WTI_PATH)
    wti["ts_utc"] = pd.to_datetime(wti["ts_utc"], utc=True)
    logging.info("WTI rows=%d, ts=%s -> %s", len(wti), wti["ts_utc"].min(), wti["ts_utc"].max())

    # 3. 算 WTI returns (1h)
    wti = wti.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"])
    wti["wti_returns"] = wti["wti_close"].pct_change()
    wti["ret_1h"] = wti["wti_returns"]  # alias for hourly_monitor compatibility

    # 4. Merge GDELT + WTI on ts_utc
    merged = pd.merge(gdelt, wti[["ts_utc", "wti_close", "wti_returns", "ret_1h"]], 
                      on="ts_utc", how="left")
    logging.info("Merged rows=%d", len(merged))

    # 5. 如果舊的 features 存在，合併歷史（保留舊的，append 新的）
    if OUTPUT_PATH.exists():
        old = pd.read_parquet(OUTPUT_PATH)
        old["ts_utc"] = pd.to_datetime(old["ts_utc"], utc=True)
        logging.info("Old features rows=%d, ts=%s -> %s", len(old), old["ts_utc"].min(), old["ts_utc"].max())
        
        # 找出 merged 裡有、old 裡沒有的時間點
        old_ts = set(old["ts_utc"])
        new_rows = merged[~merged["ts_utc"].isin(old_ts)]
        logging.info("New rows to append: %d", len(new_rows))
        
        if len(new_rows) > 0:
            # 確保欄位一致（新欄位補 NaN，舊欄位沒有的也補）
            all_cols = list(set(old.columns) | set(new_rows.columns))
            for c in all_cols:
                if c not in old.columns:
                    old[c] = np.nan
                if c not in new_rows.columns:
                    new_rows[c] = np.nan
            merged = pd.concat([old, new_rows[old.columns]], ignore_index=True)
        else:
            merged = old
    
    # 6. 排序 + 去重
    merged = merged.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"]).reset_index(drop=True)
    
    # 7. 寫出
    merged.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved features: %s (rows=%d, ts=%s -> %s)",
                 OUTPUT_PATH, len(merged), merged["ts_utc"].min(), merged["ts_utc"].max())
    logging.info("=== DONE ===")


if __name__ == "__main__":
    main()
