# AWS EC2 Day-0 Smoke Test 部署指南

**目標**: 在 AWS EC2 Free Tier 上部署 WTI×GDELT 監控管道，用 VS Code Remote-SSH 管理
**依據**: Readme.md Section 8.1 - Day 0 環境 + 手動驗證
**日期**: 2025-11-19

---

## AWS Free Tier 規格

AWS 提供 12 個月免費試用：

**EC2 Instance 選擇**:

| Instance Type | vCPU | RAM | 適用性 |
|--------------|------|-----|--------|
| **t2.micro** | 1 | 1 GB | 基本，穩定 |
| **t3.micro** | 2 | 1 GB | 較新，性能稍好 |

**Free Tier 限制**:
- 750 小時/月（足夠 24/7 運行 1 台）
- 僅限前 12 個月
- 30 GB EBS 儲存

**推薦**: t3.micro (較新架構，CPU 性能稍好)

---

## Phase 1: 建立 AWS EC2 Instance

### Step 1.1: 登入 AWS Console

1. 前往 https://aws.amazon.com/
2. 登入或註冊 AWS 帳號
3. 進入 EC2 Dashboard: Services → Compute → EC2

### Step 1.2: Launch Instance

1. 點擊 **Launch Instance**

2. **Name and tags**:
   - Name: `wti-gdelt-monitor-01`

3. **Application and OS Images (Amazon Machine Image)**:

   **推薦 A: Amazon Linux 2023** (預設)
   - AMI: Amazon Linux 2023 AMI
   - Architecture: 64-bit (x86)
   - Username: `ec2-user`
   - Package manager: `dnf` / `yum`

   **推薦 B: Ubuntu 22.04 LTS**
   - 點擊 "Browse more AMIs"
   - 搜尋 "Ubuntu 22.04"
   - 選擇 Ubuntu Server 22.04 LTS
   - Username: `ubuntu`
   - Package manager: `apt`

4. **Instance type**:
   - Family: t2 或 t3
   - Type: **t3.micro** (推薦) 或 **t2.micro**
   - ✅ 確認顯示 "Free tier eligible"

5. **Key pair (login)**:
   - 點擊 "Create new key pair"
   - Key pair name: `wti-gdelt-key`
   - Key pair type: **ED25519** (推薦) 或 RSA
   - Private key file format: **.pem** (適用所有平台)
   - 點擊 "Create key pair"
   - **重要**: 下載的 `.pem` 檔案妥善保存（無法重新下載）

   如果已有 key pair，可選擇現有的。

6. **Network settings**:
   - 點擊 "Edit"
   - **VPC**: Default VPC
   - **Subnet**: No preference
   - **Auto-assign public IP**: Enable
   - **Firewall (Security groups)**: Create security group
     - Security group name: `wti-gdelt-sg`
     - Description: `WTI GDELT monitoring security group`

   **Inbound security group rules**:

   添加兩條規則：

   **Rule 1: SSH**
   - Type: SSH
   - Protocol: TCP
   - Port: 22
   - Source: **My IP** (推薦，僅允許你的 IP) 或 **Anywhere** (0.0.0.0/0)

   **Rule 2: Streamlit Dashboard** (可選)
   - Type: Custom TCP
   - Protocol: TCP
   - Port: 8501
   - Source: **My IP** 或 **Anywhere**

7. **Configure storage**:
   - Size: **30 GiB** (Free Tier 最大值)
   - Volume type: gp3 (General Purpose SSD)
   - Delete on termination: ✅ (建議)

8. **Advanced details**: (可保持預設)

9. **Summary**:
   - 確認 "Free tier eligible" 標籤
   - Number of instances: 1

10. 點擊 **Launch instance**

**等待**: 約 1-2 分鐘，Instance State 變為 "Running"

### Step 1.3: 記錄 Instance 資訊

在 EC2 Dashboard → Instances，選擇你的 instance，記錄：

- **Public IPv4 address**: `__________________` (例如 3.145.67.89)
- **Public IPv4 DNS**: `__________________` (例如 ec2-3-145-67-89.us-east-2.compute.amazonaws.com)
- **Instance ID**: `__________________`
- **Username**: `ec2-user` (Amazon Linux) 或 `ubuntu` (Ubuntu)

---

## Phase 2: 設定 SSH 連線

### Step 2.1: 移動並設定 Key Pair 權限

下載的 `.pem` 檔案需移到正確位置並設定權限。

**Windows**:

```powershell
# 移動 key 到 .ssh 目錄
Move-Item ~\Downloads\wti-gdelt-key.pem ~\.ssh\

# 設定權限（PowerShell as Admin）
icacls $env:USERPROFILE\.ssh\wti-gdelt-key.pem /inheritance:r
icacls $env:USERPROFILE\.ssh\wti-gdelt-key.pem /grant:r "$env:USERNAME:R"
```

**macOS/Linux**:

```bash
# 移動 key
mv ~/Downloads/wti-gdelt-key.pem ~/.ssh/

# 設定權限（必須為 400 或 600）
chmod 400 ~/.ssh/wti-gdelt-key.pem
```

### Step 2.2: 測試 SSH 連線

**Amazon Linux**:
```bash
ssh -i ~/.ssh/wti-gdelt-key.pem ec2-user@<PUBLIC_IP>
```

**Ubuntu**:
```bash
ssh -i ~/.ssh/wti-gdelt-key.pem ubuntu@<PUBLIC_IP>
```

首次連線會詢問信任，輸入 `yes`。

**成功**: 看到歡迎訊息，進入 shell prompt。

### Step 2.3: 設定 SSH Config (方便管理)

編輯 `~/.ssh/config` (Windows: `C:\Users\<username>\.ssh\config`):

**Amazon Linux 範例**:
```ssh-config
Host aws-wti
    HostName <PUBLIC_IP_OR_DNS>
    User ec2-user
    IdentityFile ~/.ssh/wti-gdelt-key.pem
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

**Ubuntu 範例**:
```ssh-config
Host aws-wti
    HostName <PUBLIC_IP_OR_DNS>
    User ubuntu
    IdentityFile ~/.ssh/wti-gdelt-key.pem
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

設定後，可簡化連線：

```bash
ssh aws-wti
```

---

## Phase 3: VS Code Remote-SSH 設定

### Step 3.1: 安裝 Remote-SSH Extension

1. 開啟 VS Code
2. Extensions (Ctrl+Shift+X)
3. 搜尋 "Remote - SSH"
4. 安裝 Microsoft 的 extension

### Step 3.2: 連線到 EC2

1. VS Code 左下角點擊綠色 **><** 圖示
2. 選擇 "Connect to Host..."
3. 選擇 `aws-wti` (剛才設定的)
4. 選擇平台: **Linux**
5. 第一次連線會安裝 VS Code Server (~1-2 分鐘)
6. 連線成功後，左下角顯示 "SSH: aws-wti"

### Step 3.3: 開啟專案資料夾

1. File → Open Folder
2. 輸入: `/home/ec2-user/Data` (Amazon Linux) 或 `/home/ubuntu/Data` (Ubuntu)
3. 點擊 "OK"

---

## Phase 4: 安裝依賴與傳輸專案

### Step 4.1: 更新系統並安裝依賴

在 VS Code Remote Terminal (或 SSH shell):

**Amazon Linux 2023**:

```bash
# 更新系統
sudo dnf update -y

# 安裝必要套件
sudo dnf install -y git python3 python3-pip python3-devel gcc

# 確認版本
python3 --version  # 應為 3.9+
git --version
```

**Ubuntu**:

```bash
# 更新系統
sudo apt update && sudo apt upgrade -y

# 安裝必要套件
sudo apt install -y git python3 python3-pip python3-venv build-essential

# 確認版本
python3 --version  # 應為 3.10+
git --version
```

### Step 4.2: 傳輸 Data 專案

**方法 A: SCP 傳輸** (推薦，如果專案未在 Git)

在**本機** PowerShell 或終端機：

```bash
# 壓縮專案
cd C:\Users\niuji\Documents
tar -czf Data.tar.gz Data

# 傳輸到 EC2
# Amazon Linux:
scp -i ~/.ssh/wti-gdelt-key.pem Data.tar.gz ec2-user@<PUBLIC_IP>:~/

# Ubuntu:
scp -i ~/.ssh/wti-gdelt-key.pem Data.tar.gz ubuntu@<PUBLIC_IP>:~/

# 或使用 SSH config alias:
scp Data.tar.gz aws-wti:~/
```

在 **EC2** 上解壓：

```bash
cd ~
tar -xzf Data.tar.gz
rm Data.tar.gz
ls -lh Data/
```

**方法 B: Git Clone** (如果專案在 Git repository)

在 **EC2** 上：

```bash
cd ~
git clone <your-repo-url> Data
cd Data
```

### Step 4.3: 建立 Python Virtual Environment

在 EC2 (VS Code Remote Terminal):

```bash
cd ~/Data
python3 -m venv .venv
source .venv/bin/activate

# 確認 venv 啟動
which python  # 應顯示 ~/Data/.venv/bin/python
```

### Step 4.4: 安裝 Python 套件

```bash
# 確保在 venv 中
source .venv/bin/activate

# 升級 pip
pip install --upgrade pip

# 安裝專案依賴
pip install -r requirements.txt

# 如果需要 Streamlit (Dashboard)
pip install streamlit

# 驗證安裝
python -c "import pandas, numpy, pyarrow; print('Dependencies OK')"
```

---

## Phase 5: 執行首次監控 (Readme.md §8.1 Step 3)

### Step 5.1: 手動執行 hourly_monitor.py

在 EC2 (VS Code Remote Terminal):

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
  Prediction: ±0.00XX
  Data timestamp: 2025-XX-XX XX:00:00

[2/6] Calculating position...
  Position: ±X.XX%

[3/6] Logging position to warehouse/positions/...
  Position logged: warehouse/positions/base_seed202_lean7_positions.csv

[4/6] Calculating metrics...
  IC: 0.XXXX
  Metrics logged: warehouse/monitoring/base_seed202_lean7_metrics.csv

[5/6] Checking Hard gates...
  Rolling 15d: IC=0.XXXX, IR=X.XXXX, PMR=XX.XX%
  Hard gate status: HEALTHY
  Hard gates passed: True

[6/6] No alerts - All systems nominal

======================================================================
HOURLY CYCLE COMPLETE
======================================================================
Status: SUCCESS
Execution log: warehouse/monitoring/hourly_execution_log.csv
```

### Step 5.2: 驗證輸出檔案 (Readme.md §8.1 Step 3)

按照 Readme.md §8.1，檢查以下檔案：

```bash
# 檢查檔案是否建立
ls -lh warehouse/positions/base_seed202_lean7_positions.csv
ls -lh warehouse/monitoring/base_seed202_lean7_metrics.csv
ls -lh warehouse/monitoring/base_seed202_lean7_alerts.csv  # 如有 alert
ls -lh warehouse/monitoring/hourly_execution_log.csv

# 查看內容
tail -1 warehouse/monitoring/hourly_execution_log.csv
# 應顯示: ...,SUCCESS,,True,0

tail -1 warehouse/monitoring/base_seed202_lean7_metrics.csv
# 應顯示: timestamp,ic,prediction,position,strategy_id

tail -1 warehouse/positions/base_seed202_lean7_positions.csv
# 應顯示完整 position 記錄
```

**驗證清單**:
- [ ] `positions/base_seed202_lean7_positions.csv` 新增一列 ✅
- [ ] `monitoring/base_seed202_lean7_metrics.csv` 新增一列 ✅
- [ ] `monitoring/hourly_execution_log.csv` status=SUCCESS ✅
- [ ] 無明顯錯誤訊息 ✅

---

## Phase 6: (可選) 啟動 Dashboard (Readme.md §8.1 Step 4)

### Step 6.1: 檢查 Dashboard 檔案

```bash
ls -lh warehouse/dashboard/app.py
# 或
ls -lh warehouse/dashboard/*.py
```

如果沒有 Dashboard 檔案，可以先跳過這步驟。

### Step 6.2: 手動啟動 Streamlit Dashboard

在 EC2 (VS Code Remote Terminal):

```bash
cd ~/Data
source .venv/bin/activate

# 如果 Dashboard 檔案是 app.py
streamlit run warehouse/dashboard/app.py --server.port=8501 --server.address=0.0.0.0

# 或如果是其他檔名
streamlit run warehouse/monitoring/base_dashboard.py --server.port=8501 --server.address=0.0.0.0
```

**輸出**:
```
You can now view your Streamlit app in your browser.

  Network URL: http://172.31.x.x:8501
  External URL: http://3.145.67.89:8501
```

### Step 6.3: 在瀏覽器開啟 Dashboard

在本機瀏覽器輸入：

```
http://<EC2_PUBLIC_IP>:8501
```

**驗證** (Readme.md §8.1 Step 4):
- [ ] 能看到 IC/IR/PMR 基本圖表 ✅
- [ ] 能看到 alerts 資訊 ✅
- [ ] 能看到持倉 (positions) 資訊 ✅
- [ ] 時間序列與 log 一致 ✅

**停止 Dashboard**:
- 在 Terminal 按 `Ctrl+C`

---

## Phase 7: 多輪手動測試 (Readme.md §8.1 Step 5)

### Step 7.1: 手動多次執行

按照 Readme.md §8.1 Step 5，在 3-4 小時內多次執行：

```bash
# 每小時執行一次（手動）
cd ~/Data
source .venv/bin/activate
python warehouse/monitoring/hourly_monitor.py

# 或設定臨時 cron (每小時自動執行)
crontab -e
# 新增:
# 0 * * * * cd ~/Data && . .venv/bin/activate && python warehouse/monitoring/hourly_monitor.py
```

### Step 7.2: 驗證數據累積

執行幾次後，檢查：

```bash
# 檢查行數增加
wc -l warehouse/monitoring/hourly_execution_log.csv
wc -l warehouse/monitoring/base_seed202_lean7_metrics.csv

# 查看最近幾筆
tail -5 warehouse/monitoring/hourly_execution_log.csv
tail -5 warehouse/monitoring/base_seed202_lean7_metrics.csv
```

### Step 7.3: Dashboard 數據一致性驗證

如果 Dashboard 有在運行：

1. 重新整理瀏覽器
2. 確認時間序列圖更新
3. 確認最新 IC/Position 與 CSV 一致

---

## Day-0 成功標準 (Readme.md §8.1)

按照 Readme.md §8.1 定義，Day-0 成功條件：

### 必須達成 ✅

- [ ] **EC2 上所有 Python 程式能正常執行**
  - `hourly_monitor.py` 執行無錯誤
  - Python 環境與套件正常
  - 無路徑、權限問題

- [ ] **Dashboard 在雲端可開啟，並正確讀取 log** (如果執行了 Dashboard)
  - `http://<EC2_IP>:8501` 可訪問
  - 圖表顯示正確
  - 數據與 CSV 一致

- [ ] **沒有明顯權限 / 路徑 / 相依套件錯誤**
  - CSV 檔案正常寫入
  - 無 Permission denied
  - 無 ModuleNotFoundError

### 額外驗證

- [ ] `hourly_execution_log.csv` 所有記錄 status=SUCCESS
- [ ] `metrics.csv` 和 `positions.csv` 每次執行都新增記錄
- [ ] Hard Gate status: HEALTHY
- [ ] 無 CRITICAL alerts

---

## 疑難排解

### 問題 1: SSH 連線被拒

**檢查**:
1. Security Group 是否開放 port 22
2. EC2 instance 狀態是否 "Running"
3. Key pair 權限是否正確 (400 或 600)
4. IP 地址是否正確

```bash
# 測試連線
ssh -v -i ~/.ssh/wti-gdelt-key.pem ec2-user@<PUBLIC_IP>
```

### 問題 2: Python 套件安裝失敗

**Amazon Linux 可能缺少編譯工具**:

```bash
sudo dnf install -y python3-devel gcc
pip install --upgrade pip
pip install -r requirements.txt
```

**Ubuntu**:
```bash
sudo apt install -y python3-dev build-essential
pip install -r requirements.txt
```

### 問題 3: 記憶體不足 (OOM)

t2.micro / t3.micro 只有 1GB RAM，可能需要 swap：

```bash
# 建立 2GB swap
sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久啟用
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 驗證
free -h
```

### 問題 4: Dashboard 無法訪問

**檢查**:
1. Security Group 是否開放 port 8501
2. Streamlit 是否使用 `--server.address=0.0.0.0`
3. EC2 防火牆設定 (如果有)

```bash
# Amazon Linux - 檢查 firewalld
sudo systemctl status firewalld
# 如果啟用，開放 port
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload

# Ubuntu - 檢查 ufw
sudo ufw status
# 如果啟用，開放 port
sudo ufw allow 8501/tcp
```

### 問題 5: 找不到資料檔案

確認所有 parquet 檔案都已傳輸：

```bash
ls -lh features_hourly_with_term.parquet
ls -lh data/gdelt_hourly.parquet
ls -lh warehouse/base_monitoring_config.json
```

如果缺少，從本機傳輸：

```bash
# 本機執行
scp features_hourly_with_term.parquet aws-wti:~/Data/
scp data/gdelt_hourly.parquet aws-wti:~/Data/data/
```

---

## 成本監控

### AWS Free Tier 限制

- **EC2**: 750 小時/月 (足夠 24/7 運行 1 台)
- **EBS**: 30 GB
- **Data Transfer**: 1 GB 出站/月 (超過會收費)

### 確認在 Free Tier 內

1. AWS Console → Billing Dashboard → Free Tier
2. 查看 EC2 使用量
3. 設定 Billing Alert (推薦設在 $1)

### 避免超額費用

- 不要建立多台 instance
- 不要超過 30 GB EBS
- 不要使用其他付費服務 (RDS, Load Balancer 等)
- Dashboard 訪問量不要太大（會產生流量費用）

---

## 後續步驟 (Day-1)

Day-0 成功後，參考 Readme.md §8.2：

### 設定 cron 自動化

```bash
crontab -e
```

新增：

```cron
0 * * * * cd ~/Data && . .venv/bin/activate && python warehouse/monitoring/hourly_monitor.py >> warehouse/monitoring/hourly_cron.log 2>&1
```

### Dashboard 常駐 (systemd)

參考 Readme.md §6.4 設定 systemd service。

### 監控 24 小時

檢查：
- `hourly_cron.log` 無錯誤
- `hourly_execution_log.csv` 每小時新增記錄
- CPU/RAM 使用率 (用 `htop` 或 CloudWatch)

---

## 總結

AWS EC2 Day-0 Smoke Test 流程：

```
1. 建立 EC2 t3.micro (Free Tier)
   ↓
2. 設定 SSH + VS Code Remote-SSH
   ↓
3. 安裝依賴 + 傳輸專案
   ↓
4. 執行 hourly_monitor.py
   ↓
5. 驗證 CSV 檔案寫入
   ↓
6. (可選) 啟動 Dashboard
   ↓
7. 多輪測試 3-4 小時
   ↓
Day-0 成功 ✅
```

**Day-0 成功條件** (Readme.md §8.1):
- ✅ EC2 上所有 Python 程式正常執行
- ✅ Dashboard 可開啟並正確讀取 log
- ✅ 無權限/路徑/套件錯誤

**下一步**: Day-1 自動化 + 24 小時監控測試

---

**建立者**: Claude Code
**日期**: 2025-11-19
**版本**: 1.0
**依據**: Readme.md Section 8.1
