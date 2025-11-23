# 15 秒 TL;DR

* **唯一目標**：用 WTI × GDELT（+ 期貨價差 cl1_cl2 + OVX）建一套 **可即時監控、可審計** 的動態權重系統，每小時產生 (w_t)，並長期維持 **Hard IC 口徑**：
  (H \le 3,\ \text{lag}=1h,\ IC \ge 0.02,\ IR \ge 0.5,\ PMR \ge 0.55)

* **策略現況**：

  * 已完成資料治理與 No-Drift 合約，上鎖在 ingestion（Hard / Soft 分軌）
  * 已有 **Hard 達標策略**：`base_seed202_lean7_h1`（7 特徵：5 個 GDELT + cl1_cl2 + ovx），通過全部 Hard Gate，並正式升級為 Base 策略、進入實際權重引擎
  * 監控管線與 dashboard（terminal 版）已在本機完成，並支援 Hard Gate 連續監控

* **雲端架構（本文件新增）**：

  * **Dashboard = 純程式**（Python；可用 Streamlit / FastAPI 包裝 `base_dashboard.py` 邏輯），部署在 **AWS EC2 Free Tier** 小型實例上（t2.micro / t3.micro 一台常駐）。
  * EC2 上同時跑：

    1. 每小時的監控腳本 `hourly_monitor.py`（排程）
    2. 常駐的 Web Dashboard 服務（監控 IC/IR/PMR / 回撤 / 告警）。
  * 架構設計為 **一台 EC2 24/7 常開** 也能落在 Free Tier：新帳號在 12 個月內，每月有約 750 小時 t2.micro/t3.micro 類實例可用。

* **VS Code = 指揮中心**：

  * 用 VS Code 的 **Remote-SSH** 直接連到 EC2，所有程式、log、排程、dashboard 都在雲端修改與重啟。
  * 本機關機 / 睡眠 **完全不影響** 雲端 pipeline 與 Dashboard。

* **短期計畫**：

  * 先在 AWS 上做 **1–2 天的雲端 smoke test**：

    * Day 0：建 EC2 + 安裝環境 + 手動跑 2–3 次 hourly monitor + 開 dashboard。
    * Day 1：改成 cron + systemd 常駐，跑滿 24 小時，檢查 IC/IR/PMR 軌跡與告警。
  * 測試穩定後，再啟用 Day-7 / Day-14 / Day-30 週期審核流程（已由 `day7/day14_performance_review.py` 定義）。

---

# 1. 目標與口徑（What / Why / How）

## 1.1 Why

* 建一套 **可即時監控、可審計** 的動態權重系統：

  * 每小時產生 (w_t)（權重建議）
  * 有完整 audit trail（資料 → 特徵 → 預測 → 持倉）
  * 有 Hard Gate / Hard Stop 機制，自動風控與停用

## 1.2 Hard Gate（資料）

硬性資料門檻（任何一條不過 → 不計 KPI，策略視為失格）：

* `mapped_ratio ≥ 0.55`（GDELT bucket mapping 成功率）
* `ALL_art_cnt ≥ 3`（當小時新聞篇數，下限門檻）
* `tone_avg` 非空（情緒指標有填）
* RAW `skip_ratio ≤ 2%`（原始 GKG 解析錯誤率）

## 1.3 Hard IC（策略）

短窗訊號的硬性門檻（IC 口徑）：

* 時間窗：H ∈ {1, 2, 3}，目前主力為 **H=1**
* Lag：1 小時（預測下一小時報酬）
* 門檻：`IC ≥ 0.02 ∧ IR ≥ 0.5 ∧ PMR ≥ 0.55`
* KPI 只看 **Hard + Base**，任何 Shadow / 放寬實驗不入正式 KPI。

## 1.4 不變口徑

* 時間軸：UTC（含 GDELT × WTI 資料）
* 新聞對齊：對齊 **下一整點**（新聞在 10:xx 出現，對齊 11:00 這小時的 return）
* 非交易小時：`ret_1h=0`，只對齊，不記 KPI
* KPI / 權重：只取 `selected_source=base` 的 Hard 訊號輸出。

---

# 2. 資料與檔案（Single Source of Truth）

## 2.1 目錄結構

**本機（Windows）舊路徑**：

* 根目錄：`C:\Users\niuji\Documents\Data\`

**雲端（AWS EC2）建議路徑（Linux）**：

* 根目錄：`/home/ec2-user/Data/` 或 `~/Data/`
* 由 Git clone 同一個 repo，**保持相同的相對路徑**，僅 OS 路徑分隔符不同。

## 2.2 關鍵檔案（資料層）

* `data/gdelt_hourly_monthly/gdelt_hourly_YYYY-MM.parquet`

  * 月檔；ALL + 六桶 Raw/Norm 指標
* `data/gdelt_hourly.parquet`

  * 總檔；對齊 GDELT × 市場特徵的主資料表
* `data/features_hourly.parquet`

  * 價格特徵：WTI 期貨口徑、`ret_1h`、`is_trading_hour`
* `data/gdelt_raw/YYYY/MM/*.gkg.csv.zip`

  * 原始 GKG（tab 分隔）

## 2.3 治理與政策（Governance）

* `warehouse/policy/no_drift.yaml` / `no_drift_schema.json`

  * No-Drift 合約：規定資料口徑、欄位、版本、轉換流程
* `warehouse/policy/utils/nodrift_preflight.py`

  * `enforce(observed)`：寫檔前 preflight，違規直接 fail-fast

## 2.4 IC / 候選輸出

* `warehouse/ic/*.csv`

  * Hard / Soft IC 結果、不同策略視窗的 summary

---

# 3. 策略現況：Base 策略與特徵

## 3.1 Base 策略定義（Seed202 LEAN 7-Feature, H=1）

* **Strategy ID**：`base_seed202_lean7_h1`

* **Model**：LightGBM Regressor（Seed=202，reg_lambda=1.5 等設定）

* **特徵（7 個）**

  * **Market（2）**：

    * `cl1_cl2`：近月 – 次近月期貨價差（term structure）
    * `ovx`：Oil VIX（WTI 波動指數）
  * **GDELT（5）**：

    * `OIL_CORE_norm_art_cnt`
    * `MACRO_norm_art_cnt`
    * `SUPPLY_CHAIN_norm_art_cnt`
    * `USD_RATE_norm_art_cnt`
    * `GEOPOL_norm_art_cnt`

* **Hard 門檻達成**：

  * IC median ≈ 0.1358（> 0.02 的 ~6.8x）
  * IR ≈ 1.58（> 0.5 的 ~3.2x）
  * PMR ≈ 0.80（> 0.55 的 ~1.5x）
  * Walk-forward 51 個窗口全測，穩定通過 Hard 門檻

* **特徵重要性（關鍵結論）**

  * `cl1_cl2 + ovx` ≈ 83% 的重要性 → 市場結構與波動是主訊號來源
  * 5 個 GDELT 桶合計約 17% → 新聞當「輔助 signal」，不是主因。

---

# 4. 日內運行與監控（Hourly Monitor）

## 4.1 Hourly Monitor 流程（`hourly_monitor.py`）

每小時執行一次，流程概要：

1. **載入最新特徵**：GDELT + futures term + OVX。
2. **產生預測**：`pred`。
3. **計算持倉**：

   * `position = base_weight × sign(pred) × min(1, |pred| / 0.005)`
   * `base_weight` 初始 15%。
4. **寫入持倉 log**：`warehouse/positions/base_seed202_lean7_positions.csv`
5. **計算 metrics**：IC、IR、PMR（滾動視窗） → `warehouse/monitoring/base_seed202_lean7_metrics.csv`。
6. **檢查 Hard Gate / Hard Stop**：

   * IC / IR / PMR 是否維持在門檻
   * 資料 Hard Gate 是否通過
7. **寫入 Alerts**：`warehouse/monitoring/base_seed202_lean7_alerts.csv`
8. **寫入 Execution Log**：`warehouse/monitoring/hourly_execution_log.csv`

---

# 5. Dashboard：從 Terminal 到 雲端 Web

## 5.1 Terminal Dashboard（既有）

* `warehouse/monitoring/base_dashboard.py`：

  * 在本機終端輸出：

    * 策略卡：ACTIVE / HOLD / REDUCE / HALT + 信心分數 + 權重
    * 15d 滾動 IC / IR / PMR
    * Hard Gate / Hard Stop 狀態
  * 已驗證能在 24 小時模擬中顯示正確結果

## 5.2 雲端 Web Dashboard（本次新增）

**目標**：

* 保留 `base_dashboard.py` 的邏輯（指標與健康判斷），
* 但改成 **Web Dashboard（純程式）**，部署在 AWS EC2 上，讓任何設備只要有瀏覽器就能查看。

**建議做法**：

* 新增目錄：`warehouse/dashboard/`

  * `app.py`：Streamlit / FastAPI + 前端框架（擇一），以 Python 純程式組裝 UI：

    * 讀取 `metrics.csv / positions.csv / alerts.csv / ...`
    * 畫出：

      * IC / IR / PMR 的時間序列圖
      * 健康分數 / Hard Gate 狀態
      * 回撤曲線
      * 最新持倉表
      * 告警列表
  * `components.py`：封裝共用模組（例如「策略卡」、「風控面板」）。

* 功能對應 Dashboard.md 的七大區塊：

  * [A] Market Status Overview：OVX + cl1_cl2 regime 判讀
  * [B] Strategy Cards：策略建議 + 信心分數 + Hard Gate 狀態
  * [C] Account Status：持倉與 P&L 概覽
  * [D] Trade Records with Replay：完整 trade log + features 快照
  * [E] Data Integrity：版本、Hash、No-Drift 狀態
  * [F] Risk Control Panel：Hard Stop / Drawdown / Position 限制
  * [G] Operations：Scheduler / 執行狀態 / 最近告警

---

# 6. AWS 雲端部署架構（Dashboard = 純程式 + 上雲端跑）

## 6.1 架構概觀

* **雲端平台**：AWS
* **運行節點**：1 台 EC2（Free Tier eligible）

  * 類型：t2.micro / t3.micro（視 region 而定；皆可在 Free Tier 內使用 ~750 小時/月，為期 ~12 個月）
* **OS**：Amazon Linux 2 / Ubuntu 22.04（擇一）
* **目錄**：`~/Data/`（clone 專案）
* **服務**：

  1. **Hourly Monitor**：cron job / systemd timer，每小時執行 `python warehouse/monitoring/hourly_monitor.py`。
  2. **Dashboard Web 服務**：systemd 服務常駐 `streamlit run warehouse/dashboard/app.py`（或 FastAPI + uvicorn），對外開一個 HTTP port（例如 8501 或 80）。

## 6.2 安裝步驟（概略）

在 EC2（首次）：

```bash
# 1. 更新並安裝必要套件
sudo yum update -y  # 或 apt
sudo yum install -y git python3 python3-venv

# 2. clone 專案
cd ~
git clone <your-repo-url> Data
cd Data

# 3. 建立 venv
python3 -m venv .venv
source .venv/bin/activate

# 4. 安裝依賴
pip install -r requirements.txt  # 需含 streamlit / lightgbm / pandas 等
```

## 6.3 Hourly Monitor 排程（cron）

在 EC2 中（已進 venv 且確認 `hourly_monitor.py` 正常執行一次）：

```bash
crontab -e
```

新增：

```cron
0 * * * * cd ~/Data && . .venv/bin/activate && python warehouse/monitoring/hourly_monitor.py >> warehouse/monitoring/hourly_cron.log 2>&1
```

→ 雲端每小時自動跑一次，不依賴你的筆電。

## 6.4 Dashboard 常駐（systemd）

建立 `/etc/systemd/system/base_dashboard.service`：

```ini
[Unit]
Description=Base Strategy Dashboard (Streamlit)
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/Data
ExecStart=/home/ec2-user/Data/.venv/bin/streamlit run warehouse/dashboard/app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

啟用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable base_dashboard.service
sudo systemctl start base_dashboard.service
```

Security Group 中開放對應 port（例如 8501 或 80），可限制來源 IP 只允許你自己的 IP。

---

# 7. VS Code 當指揮中心

## 7.1 Remote-SSH 設定

* 在 VS Code 裝 **Remote - SSH** extension。
* 本機 `~/.ssh/config` 加入：

```ssh-config
Host wti-aws
    HostName <EC2-Public-IP>
    User ec2-user
    IdentityFile ~/.ssh/<your-key>.pem
```

* 在 VS Code 左下角 → `><` → `Remote-SSH: Connect to Host...` → 選 `wti-aws`

連線後：

* 直接開啟 `/home/ec2-user/Data` 當 workspace。
* 用 VS Code Terminal：

```bash
cd ~/Data
source .venv/bin/activate
```

* 在 VS Code 裡：

  * 編輯 `hourly_monitor.py`、`dashboard/app.py`、`base_dashboard.py`
  * `git status / git commit / git push`
  * `sudo systemctl restart base_dashboard.service`
  * 檢查 log：`tail -n 100 warehouse/monitoring/hourly_cron.log`

=> 所有「指揮」都在 VS Code 完成；EC2 才是真正在跑的機器。

---

# 8. 1–2 天 AWS 雲端測試計畫（Smoke Test）

## 8.1 Day 0：環境 + 手動驗證

**目標**：確認「EC2 + venv + hourly_monitor + Dashboard」在雲端能正常跑，至少連續幾個小時沒問題。

步驟：

1. **建 EC2 + SSH / VS Code 連線**

   * 使用 Free Tier eligible 的 t2.micro / t3.micro，指定 security group 開 22 + dashboard port。

2. **安裝依賴 & clone 專案**（見 §6.2）。

3. **手動跑一次 Hourly Monitor**

   ```bash
   cd ~/Data
   source .venv/bin/activate
   python warehouse/monitoring/hourly_monitor.py
   ```

   檢查：

   * `positions/base_seed202_lean7_positions.csv` 新增一列
   * `monitoring/base_seed202_lean7_metrics.csv` 新增一列
   * `monitoring/base_seed202_lean7_alerts.csv` 如 Hard Gate fail 會有紀錄
   * `monitoring/hourly_execution_log.csv` status=SUCCESS

4. **手動啟動 Dashboard Web**

   ```bash
   streamlit run warehouse/dashboard/app.py --server.port=8501 --server.address=0.0.0.0
   ```

   用瀏覽器打 `http://<EC2-IP>:8501`：

   * 確認能看到 IC/IR/PMR / alerts / 持倉等基本圖表。

5. **手動多跑幾輪（3–4 小時內）**

   * 每小時再跑一次 `hourly_monitor.py`（手動或暫時設 cron）
   * 看 Dashboard 的時間序列是否跟 log 一致。

**Day 0 成功條件**：

* EC2 上所有 Python 程式能正常執行。
* Dashboard 在雲端可開啟，並正確讀取 log。
* 沒有明顯權限 / 路徑 / 相依套件錯誤。

---

## 8.2 Day 1：自動化 + 24 小時監控測試

**目標**：驗證「完全無人干預」下，系統能在雲端連續跑滿 ~24 小時。

步驟：

1. **啟用 cron 的 Hourly Monitor**（§6.3 已設定）：

   * `crontab -l` 確認排程存在。
   * 等 2–3 小時，檢查：

     * `hourly_execution_log.csv` 每小時都有新紀錄。
     * `metrics.csv` / `positions.csv` 行數有增加。

2. **Dashboard 改為 systemd 常駐**（§6.4 已設定）：

   * `sudo systemctl status base_dashboard.service` 應為 active (running)。
   * 重新連線瀏覽器，確認正常。

3. **連續觀察 ~24 小時**：

   * 期間可不登入 EC2（模擬你筆電關機 / 睡眠）。
   * 隔日回來：

     * 查看 `hourly_execution_log.csv` 是否有中斷。
     * Dashboard 上是否完整呈現 24 筆以上資料。
     * 是否有 Hard Gate fail / CRITICAL alerts。

4. **資源與費用安全檢查**：

   * EC2 上：

     * `top` / `htop` 檢查 CPU / Memory 使用率。
   * AWS Console：

     * 檢查 Free Tier 用量是否合理（僅 1 台 t2.micro / t3.micro 長開，一般會落在 750 小時內）。

**Day 1 成功條件**：

* 連續 24 小時內 cron 沒中斷。
* `hourly_execution_log.csv` 無大量 FAIL；Hard Gate 大致維持 PASS。
* Dashboard 能回放整個 24h 的 IC / IR / PMR / alerts 曲線。
* EC2 資源壓力可接受（CPU / RAM 未長期 100%）。

---

## 8.3 Day 2（選擇性）：接上 Day-7 / Day-14 審核節奏

Day 2 不一定要立刻做，但可以預先規劃：

* 在 EC2 上同步部署：

  * `warehouse/monitoring/day7_performance_review.py`
  * `warehouse/monitoring/day14_performance_review.py`
* 透過 cron / systemd timer 定期執行，產生：

  * `day7_audit_report.json`
  * `day14_audit_report.json`
* 在 Dashboard 加一個「審核卡片」：顯示最近一次 Day-7 / Day-14 審核結果、健康分數與權重決策（是否從 15% → 20%）。

---

# 9. 後續擴充（簡述）

* 將 Dashboard 容器化（Docker），改用 ECS / Fargate 或 Cloud Run 類似服務（後續專案）。
* 接上 SNS / Email / Line Bot 等告警通知（CRITICAL alert 觸發時）。
* 多策略支援：Base + Shadow 策略在同一 Dashboard 上切換。

---

以上就是整合版 Readme：

* 保留原本 Hard 口徑與資料治理精神，
* 把 **Dashboard = 純程式 + 上雲端跑** 具體化到 AWS 架構，
* 並規劃好 1–2 天在 AWS 上的 smoke test 流程，
* 同時把 VS Code 正式定義為唯一「指揮中心」。
