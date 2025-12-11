# EC2 部署資訊填寫單（填好後照表執行）

請在每個欄位填入實際值，作為部署腳本/手冊的唯一依據。

## 基本帳號與網路
- AWS Account ID：419693044425
- IAM User / Role（含權限範圍，例如 `AmazonEC2FullAccess` + S3 讀寫）：暫無指派 Role（若需 S3/CloudWatch 再掛載）
- Region（例：`ap-northeast-1`）：us-east-1 (N. Virginia)
- Default VPC / Subnet（是否可用；若無，需建立）：VPC vpc-084d072d91e447013 / Subnet subnet-09fd558801a3e64df（已存在）
- Security Group 名稱 + 規則：需至 Console 確認 SG 名稱；目前計畫開 22（SSH，僅限本機 IP），80/443 視 Dashboard 需求開啟
  - 允許入站 SSH (22) IP 清單：僅你的 IP（建議）
  - 允許入站 HTTP/HTTPS (80/443) IP 清單（如部署 Dashboard/健康檢查）：視需求（預設不開）
- Key Pair 名稱（AWS console 建的）：data-ec2-key
- 本機私鑰路徑（用於 SSH）：~/.ssh/data-ec2-key.pem
- Elastic IP 是否需要綁定（Y/N）：N（目前使用自動指派公網 IP 3.236.235.113）

## EC2 規格
- Instance Type（例：t3.micro / t3.small）：t3.micro
- AMI（例：Amazon Linux 2023 / Ubuntu 22.04）：Amazon Linux 2023 (ami-0fa3fe0fa7920f68e / al2023-ami-2023.9.20251117.1)
- Root Volume 大小 / 類型（例：20GB gp3）：需確認（預設 8GB gp3，建議 20–30GB gp3）
- 其他 EBS / 資料盤需求（Y/N，大小，掛載點）：N
- 需預先安裝的套件（Python 版本、gcc、make、git、unzip、tmux/screen）：python3, pip, git, tmux（AL2023 內建 Py3.9）

## 原始碼與模型資產
- Repo 來源（git URL 或壓縮包路徑）：（請填入 Git URL；若用 SCP 壓縮包，標註檔名）
- 部署目錄（例：`~/wti`）：~/wti
- 需要上傳的檔案（絕對或相對路徑列表）：
  - `models/base_seed202_clbz_h1.pkl`
  - `models/base_seed202_clbz_h1_config.json`
  - `features_hourly_with_clbz.parquet`
  - 監控設定：`warehouse/base_monitoring_config.json`
  - 監控腳本：`warehouse/monitoring/hourly_monitor.py`
  - 監控排程/報告：`warehouse/monitoring/pmr_watch_7d_collector.py` 等
- 其他依賴檔案（requirements.txt / vendor 資料等）：
- 其他：確認是否需同步 `warehouse/monitoring/hourly_runlog.jsonl` 歷史 log（可選）

## 環境變數與憑證
- 需要的環境變數（例：`PYTHONPATH`、自訂 API keys）：PYTHONPATH=~/wti（其餘 API Key/SMTP 若有請補）
- 郵件或告警用憑證（若有）：（待補，如 SMTP/SES）
- 日誌/資料目錄寫入路徑（需具寫權限）：~/wti/logs, ~/wti/warehouse/monitoring

## 運行與排程
- Python 版本（例：3.10）與安裝方式（yum/apt/pyenv）：3.9（AL2023 預設）或自行安裝 3.10；yum 安裝即可
- 依賴安裝命令（例：`pip install -r requirements.txt`）：pip install -r requirements.txt
- 每小時監控命令（含工作目錄）：例
  - `cd ~/wti && /usr/bin/python3 warehouse/monitoring/hourly_monitor.py --features-path features_hourly_with_clbz.parquet --model-path models/base_seed202_clbz_h1.pkl`
- 排程方式（cron / systemd / supervisor）：cron（預設）；如需 systemd 可追加
  - 若 cron：排程條目（例 `5 * * * * cd ~/wti && /usr/bin/python3 warehouse/monitoring/hourly_monitor.py --features-path features_hourly_with_clbz.parquet --model-path models/base_seed202_clbz_h1.pkl >> ~/wti/logs/hourly_monitor.log 2>&1`）：
  - 若 systemd：單元檔名稱與路徑（例 `/etc/systemd/system/wti-monitor.service`）：（暫不使用）
- 監控報表輸出路徑（csv/json/png）：~/wti/warehouse/monitoring/

## 網路/端點（如需 Dashboard）
- 是否部署 Web Dashboard（Y/N）：N（t3.micro 資源有限，暫不開）
- 埠號（例：8501/8000/443）：-
- 是否需 HTTPS / 憑證取得方式（Let’s Encrypt / ACM / 自簽）：-
- 反向代理需求（Nginx/Traefik）：-

## 日誌與觀測
- 應收集的日誌檔案（例：`logs/hourly_monitor.log`，`warehouse/monitoring/*`）：logs/hourly_monitor.log, warehouse/monitoring/*.csv（metrics/alerts/runlog 等）
- 旋轉/保留策略（天數/大小）：保留 7 天，超過輪替（待實作 logrotate/cron）
- 故障告警管道（Email/Slack/Webhook；憑證/URL）：（待補 SMTP/Slack/Webhook）

## 安全與存取
- SSH Key 管理（是否限制特定 IP、是否啟用 MFA）：SSH 僅允許固定來源 IP；key 為 data-ec2-key.pem（建議本機備份與權限 400）
- 最低權限原則：是否需要額外 IAM policy（S3/CloudWatch/Secrets Manager）：目前無角色；若需備份/監控，需掛 IAM Role (S3 rw, CloudWatch Logs)
- 備份策略（模型/特徵檔是否同步到 S3；路徑、Bucket 名）：建議設定每日/每週同步 logs & CSV 至 S3（Bucket/前綴待定）

## 驗收清單（Go-Live 前）
- [ ] 用新 keypair 成功 SSH
- [ ] git/程式/模型/特徵 檔案已到位
- [ ] `pip install -r requirements.txt` 成功
- [ ] 手動執行 hourly_monitor 成功並產出 log/csv
- [ ] 排程（cron/systemd）啟用且無錯誤
- [ ] Security Group 僅允許必要 IP/埠
- [ ] Elastic IP（若需）已綁定並可連線
- [ ] 觀測檔案（pmr_watch_7d.csv 等）可寫入且路徑正確
