# WTI 每小時策略監控 – EC2 Pipeline 完整文件

> **最後更新**：2025-12-03 13:40 UTC+8  
> **目的**：讓任意 AI / 工程師在新對話裡一次看完就能接手，不用再翻之前的對話紀錄。

---

## 0. 專案總目標

在 AWS EC2 上建立**全自動化 WTI 原油策略監控系統**：

1. **每小時自動抓取 WTI 價格**（Capital.com REST API）
2. **每小時抓取 GDELT 新聞資料**（15 分鐘 RAW 檔）
3. **聚合成 ALL bucket → 更新 GDELT hourly parquet**
4. **合併 WTI + GDELT → 更新 features parquet**
5. **執行策略模型 → 產生 prediction / position / metrics**
6. **計算 rolling IC / IR / PMR（15 / 30 / 60 天）**
7. **發送「看得懂」的 Email 報告**

---

## 1. 當前進度狀態（2025-12-03）

### ✅ 已完成

| 項目 | 狀態 | 說明 |
|------|------|------|
| WTI 價格抓取 | ✅ | 純 requests，不依賴 SDK |
| GDELT 增量抓取 | ✅ | 每小時抓 4 個 15 分鐘檔 |
| GDELT bucket 聚合 | ✅ | ALL bucket + 各主題 bucket |
| Features 即時更新 | ✅ | 合併 GDELT + WTI → features_hourly_with_term.parquet |
| hourly_monitor.py | ✅ | 已修復 timestamp 解析問題 |
| Metrics rolling 計算 | ✅ | IC_15D / IR_15D / PMR_15D 等 |
| Email 發送 | ✅ | 時間戳已正確更新 |
| Cron 自動化 | ✅ | 每小時 :05 分執行 |

### ⚠️ 潛在風險

| 項目 | 風險 | 說明 |
|------|------|------|
| metrics CSV header | 中 | `hourly_monitor.py` 寫入時可能產生重複欄位 |
| 中文字體 | 低 | 圖表中文字顯示為方塊（不影響功能） |
| rm 目錄 warning | 低 | 清暫存時有 warning 但不影響 |

---

## 2. 系統架構

### 2.1 EC2 環境

```
EC2 Instance: Amazon Linux 2 / Python 3.9.24
專案根目錄: /home/ec2-user/Data
虛擬環境: /home/ec2-user/Data/warehouse/.venv
```

### 2.2 目錄結構

```
/home/ec2-user/Data/
├── data/
│   ├── wti_hourly_capital.parquet      # WTI 價格（72 小時）
│   ├── wti_hourly_capital.csv
│   ├── gdelt_hourly.parquet            # GDELT 全期間合併檔
│   ├── gdelt_hourly.csv
│   ├── gdelt_hourly_monthly/           # GDELT 月份 parquet
│   │   ├── gdelt_hourly_2024-10.parquet
│   │   ├── ...
│   │   └── gdelt_hourly_2025-12.parquet
│   ├── gdelt_raw_tmp/                  # GDELT RAW 暫存
│   ├── features_hourly.parquet         # 舊版 features
│   └── features_hourly_with_term.parquet  # 新版 features（含 GDELT + WTI）
│
├── warehouse/
│   ├── .venv/                          # Python 虛擬環境
│   ├── theme_map.json                  # GDELT 主題 → bucket 映射
│   ├── monitoring/
│   │   ├── hourly_monitor.py           # 主要監控腳本
│   │   ├── base_seed202_lean7_metrics.csv  # Metrics 記錄
│   │   ├── hourly_metrics.parquet      # Metrics parquet
│   │   ├── hourly_metrics_plot.png     # IC/IR/PMR 圖表
│   │   ├── hourly_runlog.jsonl         # 執行日誌
│   │   └── base_seed202_lean7_alerts.csv  # 警報記錄
│   └── positions/
│       └── base_seed202_lean7_positions.csv  # 部位記錄
│
├── jobs/
│   ├── run_hourly_monitor_and_email.sh # 總控腳本（一條龍）
│   ├── run_wti_hourly_refresh.sh       # WTI 抓價
│   ├── run_gdelt_hourly_incremental.sh # GDELT 增量
│   ├── run_gdelt_parquet_refresh.sh    # GDELT 合併
│   ├── pull_wti_hourly_capital.py      # WTI 抓價（Python）
│   ├── pull_gdelt_http_to_csv.py       # GDELT 下載
│   ├── gdelt_gkg_bucket_aggregator.py  # GDELT 聚合
│   ├── update_features_from_sources.py # Features 更新
│   └── run_hourly_metrics_refresh.py   # Rolling metrics 計算
│
├── send_hourly_email.py                # Email 發送
├── logs/
│   └── hourly_cron.log                 # Cron 執行日誌
└── check_wti.py                        # 驗證腳本
```

---

## 3. 關鍵腳本說明

### 3.1 總控腳本：`run_hourly_monitor_and_email.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /home/ec2-user/Data

# Step 0: WTI 抓價
jobs/run_wti_hourly_refresh.sh

# Step 1: GDELT 增量抓取
jobs/run_gdelt_hourly_incremental.sh

# Step 2: GDELT parquet 合併
jobs/run_gdelt_parquet_refresh.sh

# Step 3: Features 更新（GDELT + WTI → features）
warehouse/.venv/bin/python jobs/update_features_from_sources.py

# Step 4: 執行 hourly_monitor（產生 prediction / metrics）
warehouse/.venv/bin/python warehouse/monitoring/hourly_monitor.py || echo "[WARN] hourly_monitor had issues"

# Step 5: Rolling metrics 刷新
warehouse/.venv/bin/python jobs/run_hourly_metrics_refresh.py

# Step 6: 發送 Email
warehouse/.venv/bin/python send_hourly_email.py
```

### 3.2 WTI 抓價：`pull_wti_hourly_capital.py`

- **純 requests**，不依賴任何第三方 Capital SDK
- API 流程：
  1. `POST /api/v1/session` → 取得 CST + X-SECURITY-TOKEN
  2. `GET /api/v1/prices/{epic}?resolution=HOUR&max=72`
  3. 解析 closePrice.bid/ask 算中價 → wti_close
- 輸出：`data/wti_hourly_capital.parquet` / `.csv`

### 3.3 Features 更新：`update_features_from_sources.py`

- 讀取 `data/gdelt_hourly.parquet` + `data/wti_hourly_capital.parquet`
- 從 WTI 價格算 `ret_1h` / `wti_returns`
- Merge by `ts_utc`
- 輸出：`data/features_hourly_with_term.parquet`

### 3.4 Rolling Metrics：`run_hourly_metrics_refresh.py`

- 從 metrics CSV 讀取所有記錄
- 以時間窗計算 rolling：
  - 15 天 → IC_15D, IR_15D, PMR_15D
  - 30 天 → IC_30D, IR_30D, PMR_30D
  - 60 天 → IC_60D, IR_60D, PMR_60D
- 輸出：更新後的 CSV + `hourly_metrics.parquet`

---

## 4. Capital.com API 設定

```bash
export CAPITAL_API_KEY="JLOovvv0QqcklBnG"
export CAPITAL_IDENTIFIER="niujinheitaizi@gmail.com"
export CAPITAL_API_PASSWORD="@Nickatnyte3"
export CAPITAL_DEMO_MODE="True"
export CAPITAL_WTI_EPIC="OIL_CRUDE"
```

> **注意**：目前硬寫在 shell script 裡，正式上線前應改為 `.env` 或 secrets manager。

---

## 5. Cron 設定

```cron
# Daily jobs（UTC 時間）
5 2 * * *  cd /home/ec2-user/Data && jobs/run_gdelt_daily_refresh.sh >> logs/gdelt_daily.log 2>&1
35 2 * * * cd /home/ec2-user/Data && jobs/run_gdelt_parquet_refresh.sh >> logs/gdelt_parquet_daily.log 2>&1
45 2 * * * cd /home/ec2-user/Data && jobs/run_daily_email.sh >> logs/daily_email.log 2>&1

# Hourly pipeline（每小時 :05 分）
5 * * * * cd /home/ec2-user/Data && ./jobs/run_hourly_monitor_and_email.sh >> logs/hourly_cron.log 2>&1
```

---

## 6. 資料狀態快照（2025-12-03）

| 資料 | 筆數 | 時間範圍 |
|------|------|----------|
| WTI hourly | 72 | 2025-11-27 12:00 → 2025-12-03 04:00 UTC |
| GDELT hourly | 9052 | 2024-10-01 00:00 → 2025-12-03 03:00 UTC |
| Features | 9052 | 2024-10-01 00:00 → 2025-12-03 03:00 UTC |
| Metrics | 36 | 2025-11-18 16:04 → 2025-12-03 05:05 UTC |

---

## 7. 已知問題與修復記錄

### 7.1 `hourly_monitor.py` timestamp 解析（已修復）

**問題**：metrics CSV 中 timestamp 格式混雜，導致 `pd.to_datetime()` 失敗

**修復**：
```python
# 第 201 行
df_metrics['timestamp'] = pd.to_datetime(df_metrics['timestamp'], format='mixed', utc=True)

# 第 204 行
cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
```

### 7.2 Metrics CSV header 重複（已修復）

**問題**：header 出現多個 `timestamp` 欄位

**修復**：執行 `fix_metrics.py` 重建乾淨的 CSV

### 7.3 Features 不即時（已修復）

**問題**：`features_hourly_with_term.parquet` 沒有包含最新的 GDELT + WTI

**修復**：新增 `update_features_from_sources.py`，在 pipeline 中自動更新

---

## 8. 快速指令

### 手動執行完整 pipeline

```bash
ssh -i "$KEY_PATH" "ec2-user@${EC2_IP}" \
  "cd /home/ec2-user/Data && ./jobs/run_hourly_monitor_and_email.sh"
```

### 檢查最新資料

```bash
# WTI
ssh -i "$KEY_PATH" "ec2-user@${EC2_IP}" \
  "cd /home/ec2-user/Data && warehouse/.venv/bin/python check_wti.py"

# GDELT
ssh -i "$KEY_PATH" "ec2-user@${EC2_IP}" \
  "tail -3 /home/ec2-user/Data/data/gdelt_hourly.csv"

# Metrics
ssh -i "$KEY_PATH" "ec2-user@${EC2_IP}" \
  "tail -5 /home/ec2-user/Data/warehouse/monitoring/base_seed202_lean7_metrics.csv"
```

### 檢查 Cron 執行狀態

```bash
ssh -i "$KEY_PATH" "ec2-user@${EC2_IP}" \
  "tail -50 /home/ec2-user/Data/logs/hourly_cron.log"
```

### 檢查 metrics header 是否正常

```bash
ssh -i "$KEY_PATH" "ec2-user@${EC2_IP}" \
  "head -1 /home/ec2-user/Data/warehouse/monitoring/base_seed202_lean7_metrics.csv"
```

---

## 9. 後續方向（Roadmap）

### 9.1 歷年 IC / IR / PMR 圖表

**目標**：產生從 2024-10 到現在所有聚合資料的 IC / IR / PMR 完整歷史圖表

**工作項目**：
- 從 `base_seed202_lean7_metrics.csv` 讀取全部歷史記錄
- 繪製長時間序列圖（不只最近 7 天）
- 加入 30 天 / 60 天 rolling 曲線
- 輸出高解析度 PNG 或互動式 HTML 圖表
- 可選：加入 WTI 價格走勢對照

### 9.2 全系統健康檢查

**目標**：確保系統可以無人值守長期運行

**檢查項目**：
- [ ] Cron 是否正常執行（檢查 `hourly_cron.log` 最近 24 小時）
- [ ] Metrics CSV header 是否乾淨（無重複欄位）
- [ ] Features parquet 是否持續更新
- [ ] GDELT 月份 parquet 是否完整（檢查 2025-11 是否存在）
- [ ] Email 是否正常發送
- [ ] 磁碟空間是否足夠
- [ ] API key 是否即將過期

**輸出**：
- 健康檢查腳本 `jobs/health_check.py`
- 產生 JSON 格式的健康狀態報告
- 若有異常，自動發送警報 email

### 9.3 缺失資料補回 + 本機同步

**目標**：補回所有缺失資料，並同步到本機做備份

**工作項目**：

#### 缺失資料識別
- 檢查 GDELT 月份 parquet 是否有缺（特別是 2025-11）
- 檢查 WTI 價格是否有時間斷點
- 檢查 metrics 是否有時間空缺

#### 缺失資料補回
- 補抓 GDELT RAW（指定時間範圍）
- 重新聚合缺失時段
- 更新 features parquet

#### 本機同步機制
- 建立 `sync_to_local.ps1` 腳本
- 定期（每日）將 EC2 上的關鍵資料同步到本機：
  ```
  C:\Users\niuji\Documents\Data\
  ├── sync/
  │   ├── wti_hourly_capital.parquet
  │   ├── gdelt_hourly.parquet
  │   ├── features_hourly_with_term.parquet
  │   ├── base_seed202_lean7_metrics.csv
  │   └── hourly_cron.log
  └── logs/
      └── sync_log_YYYY-MM-DD.txt
  ```
- 本機 log 記錄同步時間、檔案大小、資料筆數

---

## 10. 本機環境（Windows）

```
本機路徑：C:\Users\niuji\Documents\Data
PowerShell 變數：
  $KEY_PATH = "~/.ssh/your-key.pem"
  $EC2_IP = "your-ec2-ip"
```

---

## 11. 聯絡資訊

- **Email**：niujinheitaizi@gmail.com
- **EC2**：需要 SSH key 才能存取

---

## 附錄：指標說明

| 指標 | 全名 | 意義 | 正常範圍 |
|------|------|------|----------|
| IC | Information Coefficient | 預測與實際報酬的相關性 | > 0 為好 |
| IR | Information Ratio | IC mean / IC std，穩定度 | > 0.5 表示穩定 |
| PMR | Positive Match Ratio | 預測方向猜對的比例 | > 50% 為好 |

---

*本文件由 AI 自動產生，請定期更新以保持準確性。*
