# AWS EC2 Day-0 Smoke Test 總結

**日期**: 2025-11-19 23:45 UTC+8
**狀態**: ✅ 準備完成，等待雲端執行
**依據**: Readme.md Section 8.1
**執行者**: Claude Code

---

## 🎯 Day-0 目標 (Readme.md §8.1)

確認「EC2 + venv + hourly_monitor + Dashboard」在雲端能正常跑，至少連續幾個小時沒問題。

**成功條件**:
1. EC2 上所有 Python 程式能正常執行
2. Dashboard 在雲端可開啟，並正確讀取 log
3. 沒有明顯權限 / 路徑 / 相依套件錯誤

---

## 📦 已完成的準備工作

### 1. AWS EC2 部署指南 ✅

**檔案**: `AWS_EC2_DAY0_DEPLOYMENT.md` (20 KB)

**內容涵蓋**:
- ✅ AWS Free Tier EC2 建立（t2/t3.micro）
- ✅ Security Group 設定（SSH + Dashboard ports）
- ✅ SSH Key Pair 管理與權限設定
- ✅ VS Code Remote-SSH 完整設定
- ✅ 專案傳輸（SCP / Git）
- ✅ Python venv 與套件安裝
- ✅ hourly_monitor.py 執行步驟
- ✅ Dashboard 啟動與驗證
- ✅ 多輪測試指南（3-4 小時）
- ✅ 5 個常見問題疑難排解
- ✅ Day-1 自動化準備（cron）

**特色**:
- 支援 Amazon Linux 2023 和 Ubuntu 22.04
- 完整的 Readme.md §8.1 實作細節
- 包含成本監控與 Free Tier 驗證

---

### 2. 執行檢查清單 ✅

**檔案**: `DAY0_AWS_CHECKLIST.md` (11 KB)

**內容涵蓋**:
- ✅ 7 個 Phase 的執行步驟（checkbox 格式）
- ✅ EC2 建立詳細步驟
- ✅ SSH + VS Code Remote-SSH 設定
- ✅ 專案部署與環境設定
- ✅ hourly_monitor.py 執行與驗證
- ✅ Dashboard 啟動（可選）
- ✅ 多輪測試（Readme.md §8.1 Step 5）
- ✅ Day-0 成功標準清單（對應 Readme.md）
- ✅ 執行結果記錄表格
- ✅ 4 個常見問題疑難排解
- ✅ Day-1 後續步驟

**特色**:
- 全中文使用者友善介面
- 完全對應 Readme.md §8.1 要求
- 互動式檢查清單
- 預估時間：30-45 分鐘設定 + 3-4 小時測試

---

## 🚀 執行流程概覽

### 快速開始（總計 ~4-5 小時）

```
Phase 1: 建立 EC2 Instance (15 分鐘)
    ↓
    Launch t3.micro → 設定 Security Group → 下載 Key Pair

Phase 2: 設定 SSH (10 分鐘)
    ↓
    設定 Key 權限 → 測試連線 → 設定 SSH Config

Phase 3: VS Code Remote-SSH (5 分鐘)
    ↓
    安裝 Extension → 連線到 EC2

Phase 4: 安裝依賴與專案 (15 分鐘)
    ↓
    更新系統 → 傳輸 Data → 建立 venv → 安裝套件

Phase 5: 執行首次監控 (5 分鐘) ← Readme.md §8.1 Step 3
    ↓
    python warehouse/monitoring/hourly_monitor.py
    驗證: positions.csv, metrics.csv, execution_log.csv ✅

Phase 6: (可選) 啟動 Dashboard (5 分鐘) ← Readme.md §8.1 Step 4
    ↓
    streamlit run ... → 瀏覽器訪問 http://<IP>:8501

Phase 7: 多輪測試 (3-4 小時) ← Readme.md §8.1 Step 5
    ↓
    每小時執行一次（手動或 cron）
    驗證 Dashboard 時間序列一致性
```

---

## ✅ Day-0 成功標準 (Readme.md §8.1)

**完全對應 Readme.md §8.1 定義**:

### 必須達成

- [ ] **EC2 上所有 Python 程式能正常執行**
  - `hourly_monitor.py` 無錯誤
  - Python 環境正常
  - 無路徑、權限問題

- [ ] **Dashboard 在雲端可開啟，並正確讀取 log**
  - `http://<EC2_IP>:8501` 可訪問
  - 圖表顯示正確
  - 數據與 CSV 一致

- [ ] **沒有明顯權限 / 路徑 / 相依套件錯誤**
  - CSV 檔案正常寫入
  - 無 Permission denied
  - 無 ModuleNotFoundError

### Readme.md §8.1 Step 3 檢查項目

- [ ] `positions/base_seed202_lean7_positions.csv` 新增一列
- [ ] `monitoring/base_seed202_lean7_metrics.csv` 新增一列
- [ ] `monitoring/base_seed202_lean7_alerts.csv` 如 Hard Gate fail 會有記錄
- [ ] `monitoring/hourly_execution_log.csv` status=SUCCESS

---

## 📁 建立的檔案總覽

| 檔案 | 大小 | 用途 | 推薦閱讀順序 |
|------|------|------|-------------|
| **`DAY0_AWS_CHECKLIST.md`** | 11 KB | 執行清單 | ⭐ **1. 執行時用這個** |
| `AWS_EC2_DAY0_DEPLOYMENT.md` | 20 KB | 技術手冊 | 2. 詳細參考 |
| `DAY0_AWS_SUMMARY.md` | 本檔案 | 總覽摘要 | 0. 快速了解 |

**總文件大小**: ~31 KB

---

## 🔑 關鍵技術點

### 1. AWS Free Tier 規格

```
EC2 Instance:
- t3.micro: 2 vCPU, 1 GB RAM (推薦)
- t2.micro: 1 vCPU, 1 GB RAM
- 750 小時/月（前 12 個月免費）
```

### 2. 完全對應 Readme.md §8.1

所有步驟嚴格按照 Readme.md Section 8.1 設計：

- ✅ **Step 1**: EC2 + SSH / VS Code 連線
- ✅ **Step 2**: 安裝依賴 & clone 專案（§6.2）
- ✅ **Step 3**: 手動跑 hourly_monitor.py，檢查 CSV 檔案
- ✅ **Step 4**: 手動啟動 Dashboard，瀏覽器驗證
- ✅ **Step 5**: 多跑幾輪（3-4 小時），Dashboard 一致性驗證

### 3. VS Code Remote-SSH 工作流程

```
本機 VS Code ←SSH→ AWS EC2
    ↓                  ↓
只做顯示            實際執行
可以關機            持續運行
```

### 4. 驗證檔案 (Readme.md §8.1 明確要求)

```bash
# Readme.md §8.1 Step 3 檢查清單
✅ positions/base_seed202_lean7_positions.csv 新增一列
✅ monitoring/base_seed202_lean7_metrics.csv 新增一列
✅ monitoring/hourly_execution_log.csv status=SUCCESS
✅ monitoring/base_seed202_lean7_alerts.csv (如有 alert)
```

---

## 📝 執行步驟速查

### 最快路徑（有經驗用戶）

```bash
# 1. AWS Console: Launch t3.micro, 設定 Security Group
# 2. 下載 key pair, 設定權限
# 3. SSH 測試連線
ssh -i ~/.ssh/wti-gdelt-key.pem ec2-user@<IP>

# 4. 更新系統 (Amazon Linux)
sudo dnf update -y && sudo dnf install -y git python3 python3-devel gcc

# 5. 傳輸專案 (本機)
tar -czf Data.tar.gz Data && scp Data.tar.gz aws-wti:~/

# 6. EC2 設定
cd ~ && tar -xzf Data.tar.gz && cd Data
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# 7. 執行監控 (Readme.md §8.1 Step 3)
python warehouse/monitoring/hourly_monitor.py

# 8. 驗證檔案
tail -1 warehouse/monitoring/hourly_execution_log.csv
ls -lh warehouse/positions/*.csv warehouse/monitoring/*.csv

# 9. (可選) 啟動 Dashboard (Readme.md §8.1 Step 4)
streamlit run warehouse/dashboard/app.py --server.port=8501 --server.address=0.0.0.0

# 10. 多輪測試 (Readme.md §8.1 Step 5)
# 每小時執行一次，共 3-4 次
```

### 詳細路徑（初次使用）

參考 `DAY0_AWS_CHECKLIST.md`，按照 checkbox 逐步執行。

---

## 🔄 Day-1 準備（Readme.md §8.2）

Day-0 成功後，參考 Readme.md Section 8.2：

### 自動化（Readme.md §6.3）

```bash
crontab -e
```

新增：

```cron
0 * * * * cd ~/Data && . .venv/bin/activate && python warehouse/monitoring/hourly_monitor.py >> warehouse/monitoring/hourly_cron.log 2>&1
```

### 24 小時監控測試

- 檢查 `hourly_cron.log` 無錯誤
- 檢查 `hourly_execution_log.csv` 每小時新增
- EC2 資源壓力可接受（CPU / RAM 未長期 100%）

### Dashboard 常駐（Readme.md §6.4）

設定 systemd service 讓 Dashboard 常駐。

---

## 📊 預期成果

### 系統資訊

```bash
# 在 EC2 上執行
uname -a
# Linux ip-172-31-x-x ... x86_64 GNU/Linux

free -h
#               total        used        free
# Mem:           985Mi       300Mi       400Mi
# Swap:          2.0Gi         0B       2.0Gi

df -h
# /dev/xvda1       30G  4.0G   26G  14% /
```

### 監控執行結果

```bash
# Readme.md §8.1 Step 3 驗證
tail -1 warehouse/monitoring/hourly_execution_log.csv
# 2025-11-19T23:45:00,SUCCESS,,True,0

tail -1 warehouse/monitoring/base_seed202_lean7_metrics.csv
# 2025-11-19T23:45:00,0.1234,0.0023,0.069,base_seed202_lean7_h1

tail -1 warehouse/positions/base_seed202_lean7_positions.csv
# 2025-11-19T23:45:00,0.0023,0.069,0.15,0.3,base_seed202_lean7_h1,...
```

### Dashboard (Readme.md §8.1 Step 4)

```
URL: http://<EC2_IP>:8501
圖表:
  - IC/IR/PMR 時間序列 ✅
  - Alerts 列表 ✅
  - Positions 顯示 ✅
  - 與 CSV 一致 ✅
```

---

## ⚠️ 重要提醒

### 1. Readme.md §8.1 嚴格遵循

所有步驟、檢查項目、成功標準完全依照 Readme.md Section 8.1 設計。

### 2. Security Group 設定

必須開放：
- **Port 22**: SSH 連線
- **Port 8501**: Streamlit Dashboard (如需要)

建議 Source 設為 "My IP" 而非 "Anywhere"（更安全）。

### 3. Key Pair 管理

- `.pem` 檔案僅下載一次，無法重新下載
- 必須設定正確權限（400 或 600）
- 妥善保存，遺失無法恢復

### 4. 費用控制

- AWS Free Tier: 前 12 個月，750 小時/月
- 只運行 1 台 t2/t3.micro 完全免費
- 設定 Billing Alert ($1) 預防超額

### 5. swap 記憶體

t2/t3.micro 只有 1GB RAM，建議設定 2GB swap 避免 OOM。

---

## 📞 疑難排解快速參考

| 問題 | 快速解決 | 詳細文件 |
|------|---------|---------|
| SSH 連線被拒 | 檢查 Security Group port 22 | AWS_EC2_DAY0_DEPLOYMENT.md Phase 7 |
| Key 權限錯誤 | `chmod 400 wti-gdelt-key.pem` | DAY0_AWS_CHECKLIST.md Phase 2 |
| Python 套件失敗 | 安裝 python3-devel gcc | AWS_EC2_DAY0_DEPLOYMENT.md 疑難排解 |
| 記憶體不足 | 建立 2GB swap | DAY0_AWS_CHECKLIST.md 疑難排解 |
| Dashboard 無法訪問 | 檢查 Security Group port 8501 | AWS_EC2_DAY0_DEPLOYMENT.md 疑難排解 |

---

## 📚 參考文件

### 專案文件
- **`Readme.md` Section 8.1**: Day-0 需求定義 ⭐
- `Readme.md` Section 6.2: 安裝步驟
- `Readme.md` Section 6.3: Hourly Monitor cron
- `Readme.md` Section 6.4: Dashboard systemd
- `Readme.md` Section 8.2: Day-1 自動化

### 部署文件
- `DAY0_AWS_CHECKLIST.md`: 執行清單
- `AWS_EC2_DAY0_DEPLOYMENT.md`: 技術手冊

### 程式碼
- `warehouse/monitoring/hourly_monitor.py`: 監控主程式
- `warehouse/base_monitoring_config.json`: 策略配置

---

## ✅ 準備狀態確認

- [x] 本機監控測試通過（2025-11-19 已測試）
- [x] AWS EC2 部署指南完整（20 KB）
- [x] 執行檢查清單建立（11 KB）
- [x] 完全對應 Readme.md §8.1 要求
- [x] 疑難排解指南準備

**狀態**: 🟢 **100% 就緒，可以開始執行**

---

## 📋 下一步行動

### 立即執行（建議）

1. **開啟 `DAY0_AWS_CHECKLIST.md`**
2. **按照 Readme.md §8.1 逐步執行**
3. **記錄執行結果到 checklist 表格**
4. **執行成功後更新 `RUNLOG_OPERATIONS.md`**

### 執行中重點

- ✅ **Readme.md §8.1 Step 3**: 驗證 4 個 CSV 檔案
- ✅ **Readme.md §8.1 Step 4**: Dashboard 可開啟並正確顯示
- ✅ **Readme.md §8.1 Step 5**: 多輪測試 3-4 小時

### Day-1 行動（Readme.md §8.2）

1. 設定 cron 自動化
2. 監控 24 小時
3. 檢查執行日誌與資源使用

---

**準備者**: Claude Code
**完成時間**: 2025-11-19 23:45 UTC+8
**文件總數**: 3 份（~31 KB）
**依據**: Readme.md Section 8.1
**預估執行時間**: 30-45 分鐘設定 + 3-4 小時測試
**狀態**: ✅ **Ready for AWS Deployment**

---

**建議**: 從 `DAY0_AWS_CHECKLIST.md` 開始，嚴格按照 Readme.md §8.1 執行！
