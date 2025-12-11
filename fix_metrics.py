import pandas as pd
from pathlib import Path

metrics_path = Path("warehouse/monitoring/base_seed202_lean7_metrics.csv")
backup_path = Path("warehouse/monitoring/base_seed202_lean7_metrics.csv.bak")

# 讀備份（有歷史資料）
old_df = pd.read_csv(backup_path)
print(f"Backup has {len(old_df)} rows")

# 修復 header：找到正確的 timestamp 欄位
# 保留第一個 timestamp，重命名其他
cols = list(old_df.columns)
new_cols = []
ts_count = 0
for c in cols:
    if c == "timestamp" or c.startswith("timestamp."):
        if ts_count == 0:
            new_cols.append("timestamp")
        else:
            new_cols.append(f"_drop_{ts_count}")
        ts_count += 1
    else:
        new_cols.append(c)

old_df.columns = new_cols

# 刪除多餘的 timestamp 欄位
drop_cols = [c for c in old_df.columns if c.startswith("_drop_")]
old_df = old_df.drop(columns=drop_cols)
print(f"After cleanup: {list(old_df.columns)}")

# 讀新的 metrics（剛從 runlog 建的）
new_df = pd.read_csv(metrics_path)
print(f"New metrics has {len(new_df)} rows")

# 合併：舊資料 + 新資料（去重）
old_df["timestamp"] = pd.to_datetime(old_df["timestamp"], format="mixed", utc=True)
new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], format="mixed", utc=True)

combined = pd.concat([old_df, new_df], ignore_index=True)
combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
combined = combined.sort_values("timestamp").reset_index(drop=True)

# 清除 rolling 欄位（讓 refresh 重算）
for col in ["IC_15D", "IR_15D", "PMR_15D", "IC_30D", "IR_30D", "PMR_30D", "IC_60D", "IR_60D", "PMR_60D"]:
    if col in combined.columns:
        combined[col] = None

combined.to_csv(metrics_path, index=False)
print(f"Combined metrics: {len(combined)} rows")
print(f"Timestamp range: {combined['timestamp'].min()} -> {combined['timestamp'].max()}")
