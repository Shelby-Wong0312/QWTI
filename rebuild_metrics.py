import json
import pandas as pd
from pathlib import Path

runlog_path = Path("warehouse/monitoring/hourly_runlog.jsonl")
metrics_path = Path("warehouse/monitoring/base_seed202_lean7_metrics.csv")

# 備份舊檔
if metrics_path.exists():
    metrics_path.rename(metrics_path.with_suffix(".csv.bak"))
    print("Backed up old metrics CSV")

# 從 runlog 讀取所有 SUCCESS 記錄
records = []
with open(runlog_path) as f:
    for line in f:
        r = json.loads(line.strip())
        if r.get("status") == "SUCCESS":
            records.append({
                "timestamp": r["ts_run"],
                "ic": r.get("ic_15d", 0),  # 用當時的 ic
                "prediction": r.get("prediction", 0),
                "position": r.get("position", 0),
                "strategy_id": r.get("strategy_id", "base_seed202_lean7_h1"),
            })

if not records:
    print("No SUCCESS records found, keeping backup")
else:
    df = pd.DataFrame(records)
    df["IC_15D"] = None
    df["IR_15D"] = None
    df["PMR_15D"] = None
    df["IC_30D"] = None
    df["IR_30D"] = None
    df["PMR_30D"] = None
    df["IC_60D"] = None
    df["IR_60D"] = None
    df["PMR_60D"] = None
    
    df.to_csv(metrics_path, index=False)
    print(f"Rebuilt metrics CSV with {len(df)} records")
    print(f"Latest timestamp: {df['timestamp'].iloc[-1]}")
