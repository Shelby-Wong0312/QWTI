# AWS EC2 Day-0 Smoke Test 執行清單

**目標**: 在 AWS EC2 上驗證監控管道，用 VS Code Remote-SSH 管理
**依據**: Readme.md Section 8.1
**預計時間**: 30-45 分鐘（首次設定）+ 3-4 小時（多輪測試）
**日期**: 2025-11-19

---

## 📋 前置準備

- [ ] AWS 帳號（Free Tier 可用）
- [ ] 本機有 Data 專案完整檔案
- [ ] VS Code 已安裝
- [ ] 本機網路穩定

---

## 🚀 Phase 1: 建立 EC2 Instance（~15 分鐘）

### Step 1.1: Launch EC2

1. 登入 AWS Console → EC2 Dashboard
2. 點擊 **Launch Instance**

### Step 1.2: 設定 Instance

**Name**: `wti-gdelt-monitor-01`

**AMI** (選一種):
- [ ] **Amazon Linux 2023** (推薦，username: `ec2-user`)
- [ ] **Ubuntu 22.04 LTS** (username: `ubuntu`)

**Instance type**:
- [ ] **t3.micro** (推薦) 或 **t2.micro**
- [ ] ✅ 確認顯示 "Free tier eligible"

**Key pair**:
- [ ] 建立新 key pair: `wti-gdelt-key` (ED25519, .pem 格式)
- [ ] **下載 .pem 檔案並妥善保存**

**Network settings**:
- [ ] Auto-assign public IP: **Enable**
- [ ] Security group: 新建 `wti-gdelt-sg`

**Security group rules**:
- [ ] Rule 1: SSH (port 22), Source: **My IP**
- [ ] Rule 2: Custom TCP (port 8501), Source: **My IP** (Dashboard 用)

**Storage**:
- [ ] Size: **30 GB** (Free Tier 最大)

### Step 1.3: Launch 並記錄資訊

- [ ] 點擊 **Launch instance**
- [ ] 等待 Instance State = "Running"

**記錄以下資訊**:
- Public IP: `__________________`
- Username: `ec2-user` 或 `ubuntu`
- Key path: `~/.ssh/wti-gdelt-key.pem`

✅ **檢查點**: EC2 instance 狀態 = Running

---

## 🔐 Phase 2: 設定 SSH 連線（~10 分鐘）

### Step 2.1: 設定 Key 權限

**Windows** (PowerShell as Admin):

```powershell
Move-Item ~\Downloads\wti-gdelt-key.pem ~\.ssh\
icacls $env:USERPROFILE\.ssh\wti-gdelt-key.pem /inheritance:r
icacls $env:USERPROFILE\.ssh\wti-gdelt-key.pem /grant:r "$env:USERNAME:R"
```

**macOS/Linux**:

```bash
mv ~/Downloads/wti-gdelt-key.pem ~/.ssh/
chmod 400 ~/.ssh/wti-gdelt-key.pem
```

### Step 2.2: 測試 SSH 連線

```bash
# Amazon Linux:
ssh -i ~/.ssh/wti-gdelt-key.pem ec2-user@<PUBLIC_IP>

# Ubuntu:
ssh -i ~/.ssh/wti-gdelt-key.pem ubuntu@<PUBLIC_IP>
```

✅ **檢查點**: 成功登入 EC2，看到 shell prompt

### Step 2.3: 設定 SSH Config

編輯 `~/.ssh/config` (Windows: `C:\Users\<username>\.ssh\config`)

**Amazon Linux**:
```ssh-config
Host aws-wti
    HostName <PUBLIC_IP>
    User ec2-user
    IdentityFile ~/.ssh/wti-gdelt-key.pem
    ServerAliveInterval 60
```

**Ubuntu**:
```ssh-config
Host aws-wti
    HostName <PUBLIC_IP>
    User ubuntu
    IdentityFile ~/.ssh/wti-gdelt-key.pem
    ServerAliveInterval 60
```

測試簡化連線：

```bash
ssh aws-wti
```

✅ **檢查點**: `ssh aws-wti` 可直接登入

---

## 💻 Phase 3: VS Code Remote-SSH（~5 分鐘）

### Step 3.1: 安裝 Extension

1. VS Code → Extensions (Ctrl+Shift+X)
2. 搜尋 "Remote - SSH"
3. 安裝 Microsoft 官方版本

### Step 3.2: 連線到 EC2

1. VS Code 左下角 **><** → "Connect to Host..."
2. 選擇 `aws-wti`
3. 選擇平台: **Linux**
4. 等待 VS Code Server 安裝 (~1-2 分鐘)

✅ **檢查點**: 左下角顯示 "SSH: aws-wti"

### Step 3.3: 開啟資料夾

1. File → Open Folder
2. 輸入 `/home/ec2-user/Data` 或 `/home/ubuntu/Data`
3. 點擊 OK

---

## 📦 Phase 4: 安裝依賴與專案（~15 分鐘）

### Step 4.1: 更新系統

在 VS Code Remote Terminal (或 SSH):

**Amazon Linux**:
```bash
sudo dnf update -y
sudo dnf install -y git python3 python3-pip python3-devel gcc
python3 --version
```

**Ubuntu**:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-pip python3-venv build-essential
python3 --version
```

### Step 4.2: 傳輸專案

**本機** PowerShell:

```powershell
cd C:\Users\niuji\Documents
tar -czf Data.tar.gz Data
scp Data.tar.gz aws-wti:~/
```

**EC2** Terminal:

```bash
cd ~
tar -xzf Data.tar.gz
rm Data.tar.gz
ls -lh Data/
```

✅ **檢查點**: `~/Data/` 目錄存在且有完整檔案

### Step 4.3: 建立 Python venv

```bash
cd ~/Data
python3 -m venv .venv
source .venv/bin/activate
which python  # 確認在 venv
```

### Step 4.4: 安裝套件

```bash
pip install --upgrade pip
pip install -r requirements.txt

# 如需 Dashboard
pip install streamlit

# 驗證
python -c "import pandas, numpy, pyarrow; print('OK')"
```

✅ **檢查點**: 所有套件安裝成功

---

## ▶️ Phase 5: 執行首次監控（~5 分鐘）

**Readme.md §8.1 Step 3**

### Step 5.1: 執行 hourly_monitor.py

```bash
cd ~/Data
source .venv/bin/activate
python warehouse/monitoring/hourly_monitor.py
```

**預期輸出**:

```
======================================================================
HOURLY MONITORING CYCLE - 2025-11-19 XX:XX:XX
======================================================================

[1/6] Getting latest prediction...
[2/6] Calculating position...
[3/6] Logging position to warehouse/positions/...
[4/6] Calculating metrics...
[5/6] Checking Hard gates...
[6/6] No alerts - All systems nominal

======================================================================
HOURLY CYCLE COMPLETE
======================================================================
Status: SUCCESS
```

✅ **檢查點**: 顯示 "Status: SUCCESS"

### Step 5.2: 驗證檔案寫入 (Readme.md §8.1 檢查項目)

```bash
# 檢查檔案存在
ls -lh warehouse/positions/base_seed202_lean7_positions.csv
ls -lh warehouse/monitoring/base_seed202_lean7_metrics.csv
ls -lh warehouse/monitoring/hourly_execution_log.csv

# 查看內容
tail -1 warehouse/monitoring/hourly_execution_log.csv
# 期望: ...,SUCCESS,,True,0

tail -1 warehouse/monitoring/base_seed202_lean7_metrics.csv
# 期望: timestamp,ic,prediction,position,strategy_id

tail -1 warehouse/positions/base_seed202_lean7_positions.csv
# 期望: 完整 position 記錄
```

**Readme.md §8.1 Step 3 檢查清單**:
- [ ] `positions/base_seed202_lean7_positions.csv` 新增一列 ✅
- [ ] `monitoring/base_seed202_lean7_metrics.csv` 新增一列 ✅
- [ ] `monitoring/hourly_execution_log.csv` status=SUCCESS ✅
- [ ] 無錯誤訊息 ✅

---

## 📊 Phase 6: (可選) 啟動 Dashboard

**Readme.md §8.1 Step 4**

### Step 6.1: 檢查 Dashboard 檔案

```bash
# 找 Dashboard 檔案
ls warehouse/dashboard/*.py
ls warehouse/monitoring/base_dashboard.py
```

### Step 6.2: 啟動 Streamlit

```bash
cd ~/Data
source .venv/bin/activate

# 依實際檔案調整
streamlit run warehouse/dashboard/app.py --server.port=8501 --server.address=0.0.0.0

# 或
streamlit run warehouse/monitoring/base_dashboard.py --server.port=8501 --server.address=0.0.0.0
```

### Step 6.3: 瀏覽器訪問

開啟瀏覽器，輸入：

```
http://<EC2_PUBLIC_IP>:8501
```

**Readme.md §8.1 Step 4 驗證**:
- [ ] 能看到 IC/IR/PMR 基本圖表 ✅
- [ ] 能看到 alerts 資訊 ✅
- [ ] 能看到持倉 (positions) 資訊 ✅

**停止 Dashboard**: Terminal 按 `Ctrl+C`

---

## 🔄 Phase 7: 多輪測試（3-4 小時）

**Readme.md §8.1 Step 5**

### Step 7.1: 設定臨時 cron（可選）

```bash
crontab -e
```

新增：

```cron
0 * * * * cd ~/Data && . .venv/bin/activate && python warehouse/monitoring/hourly_monitor.py
```

或手動每小時執行一次。

### Step 7.2: 驗證數據累積

執行 3-4 次後：

```bash
# 檢查行數增加
wc -l warehouse/monitoring/hourly_execution_log.csv
wc -l warehouse/monitoring/base_seed202_lean7_metrics.csv

# 查看最近記錄
tail -5 warehouse/monitoring/hourly_execution_log.csv
```

### Step 7.3: Dashboard 一致性

如果 Dashboard 有運行：

- [ ] 重新整理瀏覽器
- [ ] 時間序列圖更新
- [ ] 數據與 CSV 一致

---

## ✅ Day-0 成功標準 (Readme.md §8.1)

**Readme.md §8.1 定義的成功條件**:

### 必須達成

- [ ] **EC2 上所有 Python 程式能正常執行**
  - `hourly_monitor.py` 無錯誤
  - Python 環境正常
  - 無路徑、權限問題

- [ ] **Dashboard 在雲端可開啟，並正確讀取 log** (如有執行)
  - `http://<EC2_IP>:8501` 可訪問
  - 圖表顯示正確
  - 數據與 CSV 一致

- [ ] **沒有明顯權限 / 路徑 / 相依套件錯誤**
  - CSV 檔案正常寫入
  - 無 Permission denied
  - 無 ModuleNotFoundError

### 額外驗證

- [ ] 所有 `hourly_execution_log.csv` 記錄 status=SUCCESS
- [ ] `metrics.csv` 和 `positions.csv` 每次都新增
- [ ] Hard Gate status: HEALTHY
- [ ] 無 CRITICAL alerts

---

## 📊 執行結果記錄

**EC2 Instance 資訊**:
```
Instance Type: t3.micro / t2.micro
AMI: Amazon Linux 2023 / Ubuntu 22.04
Public IP: ___________________
Instance ID: ___________________
```

**首次執行結果**:
```
執行時間: ___________________
狀態: SUCCESS / FAILED
IC 值: ___________________
Position: ___________________
Hard Gate: PASSED / FAILED
```

**多輪測試**:
```
執行次數: ___ 次
成功次數: ___ 次
失敗次數: ___ 次
總時長: ___ 小時
```

**檔案驗證**:
```bash
wc -l warehouse/monitoring/hourly_execution_log.csv
# 行數: ___________________

wc -l warehouse/monitoring/base_seed202_lean7_metrics.csv
# 行數: ___________________

wc -l warehouse/positions/base_seed202_lean7_positions.csv
# 行數: ___________________
```

**Dashboard 驗證** (如有執行):
```
URL: http://<IP>:8501
訪問: 成功 / 失敗
圖表: 正常 / 異常
數據一致: 是 / 否
```

---

## 🔧 疑難排解

### 問題 1: SSH 連線失敗

```bash
# 檢查 Security Group port 22 是否開放
# 檢查 Key 權限
chmod 400 ~/.ssh/wti-gdelt-key.pem

# Verbose 模式找問題
ssh -v -i ~/.ssh/wti-gdelt-key.pem ec2-user@<IP>
```

### 問題 2: Python 套件安裝失敗

```bash
# Amazon Linux
sudo dnf install -y python3-devel gcc

# Ubuntu
sudo apt install -y python3-dev build-essential

pip install --upgrade pip
pip install -r requirements.txt
```

### 問題 3: 記憶體不足

```bash
# 建立 2GB swap
sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 驗證
free -h
```

### 問題 4: Dashboard 無法訪問

```bash
# 檢查 Security Group port 8501 是否開放
# 確認 Streamlit 使用 --server.address=0.0.0.0

# Amazon Linux 防火牆
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload

# Ubuntu 防火牆
sudo ufw allow 8501/tcp
```

---

## 🔄 Day-1 後續

Day-0 成功後，參考 Readme.md §8.2：

### 正式設定 cron

```bash
crontab -e
```

新增：

```cron
0 * * * * cd ~/Data && . .venv/bin/activate && python warehouse/monitoring/hourly_monitor.py >> warehouse/monitoring/hourly_cron.log 2>&1
```

### 監控 24 小時

檢查：
- [ ] `hourly_cron.log` 無錯誤
- [ ] `hourly_execution_log.csv` 每小時新增
- [ ] EC2 CPU/RAM 正常 (用 `htop`)

### Dashboard 常駐

參考 Readme.md §6.4 設定 systemd service。

---

## 💰 成本提醒

- AWS Free Tier: 750 小時/月（前 12 個月）
- 只運行 1 台 t2/t3.micro 完全免費
- 設定 Billing Alert ($1) 預防超額

---

## 📚 參考文件

- `AWS_EC2_DAY0_DEPLOYMENT.md`: 詳細技術手冊
- `Readme.md` Section 8.1: Day-0 需求定義
- `Readme.md` Section 6.2: 安裝步驟
- `warehouse/monitoring/hourly_monitor.py`: 監控主程式

---

**狀態**: Ready for Execution
**建立者**: Claude Code
**日期**: 2025-11-19
**版本**: 1.0
**依據**: Readme.md §8.1
