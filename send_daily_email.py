#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
send_daily_email.py

功能：寄出「WTI 每日綜合報告」
- 統計「今天」所有 hourly_monitor 執行結果
- 統計今日警報數
- 抓最新 15 日 IC / IR / PMR 做健康度判斷
- 畫最近 7 天 IC / IR / PMR 圖
"""

import json
import logging
import smtplib
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
WAREHOUSE_DIR = PROJECT_ROOT / "warehouse"
MONITOR_DIR = WAREHOUSE_DIR / "monitoring"

METRICS_CSV = MONITOR_DIR / "base_seed202_lean7_metrics.csv"
ALERTS_CSV = MONITOR_DIR / "base_seed202_lean7_alerts.csv"
EXEC_LOG_CSV = MONITOR_DIR / "hourly_execution_log.csv"

PLOT_PATH = MONITOR_DIR / "daily_metrics_plot.png"
EMAIL_CONFIG_PATH = PROJECT_ROOT / "email_config.json"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)


def load_email_config(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"email 設定檔不存在：{path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = [
        "smtp_server",
        "smtp_port",
        "username",
        "password",
        "from_email",
        "to_email",
    ]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"email_config.json 缺少欄位: {missing}")
    return cfg


def detect_ts_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["ts", "timestamp", "as_of", "as_of_utc", "time"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def to_utc_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def pick_metric(row: pd.Series, candidates: List[str]) -> Optional[float]:
    for c in candidates:
        if c in row and pd.notna(row[c]):
            return float(row[c])
    return None


@dataclass
class DailyStats:
    total_runs: int
    success_runs: int
    failed_runs: int
    alerts_today: int
    ic_15d: Optional[float]
    ir_15d: Optional[float]
    pmr_15d: Optional[float]
    last_monitor_ts_utc: Optional[datetime]


def get_today_window_utc(offset_hours: int = 8) -> (datetime, datetime, datetime):
    """以本地 (UTC+offset) 的「今天」做日區間，回傳 (start_utc, end_utc, now_local)"""
    local_tz = timezone(timedelta(hours=offset_hours))
    now_local = datetime.now(local_tz)
    d: date = now_local.date()
    start_local = datetime.combine(d, time(0, 0), tzinfo=local_tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(
        timezone.utc
    ), now_local


def summarize_today() -> DailyStats:
    start_utc, end_utc, _ = get_today_window_utc(offset_hours=8)

    # 1) execution log：統計今日執行次數 / 成功 / 失敗
    total_runs = success_runs = failed_runs = 0
    last_monitor_ts_utc: Optional[datetime] = None

    if EXEC_LOG_CSV.exists():
        df_exec = pd.read_csv(EXEC_LOG_CSV)
        if not df_exec.empty:
            ts_col = detect_ts_column(df_exec)
            if ts_col:
                df_exec["_ts_utc"] = to_utc_series(df_exec[ts_col])
                mask = (df_exec["_ts_utc"] >= start_utc) & (
                    df_exec["_ts_utc"] < end_utc
                )
                df_today = df_exec[mask]
            else:
                df_today = df_exec.iloc[0:0]

            total_runs = len(df_today)

            # 嘗試讀取 status 或 success 欄位
            if "status" in df_today.columns:
                success_runs = (df_today["status"].astype(str) == "success").sum()
                failed_runs = (df_today["status"].astype(str) != "success").sum()
            elif "success" in df_today.columns:
                success_runs = df_today["success"].astype(bool).sum()
                failed_runs = total_runs - success_runs
            else:
                success_runs = 0
                failed_runs = 0

            if not df_today.empty:
                last_ts = df_today.get("_ts_utc")
                if last_ts is not None:
                    last_ts = last_ts.max()
                    if pd.notna(last_ts):
                        last_monitor_ts_utc = last_ts.to_pydatetime()

    # 2) 今日警報數
    alerts_today = 0
    if ALERTS_CSV.exists():
        df_alerts = pd.read_csv(ALERTS_CSV)
        if not df_alerts.empty:
            ts_col = detect_ts_column(df_alerts)
            if ts_col:
                df_alerts["_ts_utc"] = to_utc_series(df_alerts[ts_col])
                mask = (df_alerts["_ts_utc"] >= start_utc) & (
                    df_alerts["_ts_utc"] < end_utc
                )
                alerts_today = int(mask.sum())

    # 3) 最新一筆 15 日 IC / IR / PMR
    ic_15d = ir_15d = pmr_15d = None
    if METRICS_CSV.exists():
        df_metrics = pd.read_csv(METRICS_CSV)
        if not df_metrics.empty:
            ts_col = detect_ts_column(df_metrics)
            if ts_col:
                df_metrics["_ts_utc"] = to_utc_series(df_metrics[ts_col])
                df_metrics = df_metrics.sort_values("_ts_utc")
            latest = df_metrics.iloc[-1]
            ic_15d = pick_metric(latest, ["IC_15D", "ic_15d", "ic_15d_rolling"])
            ir_15d = pick_metric(latest, ["IR_15D", "ir_15d", "ir_15d_rolling"])
            pmr_15d = pick_metric(latest, ["PMR_15D", "pmr_15d", "pmr_15d_rolling"])

    return DailyStats(
        total_runs=total_runs,
        success_runs=success_runs,
        failed_runs=failed_runs,
        alerts_today=alerts_today,
        ic_15d=ic_15d,
        ir_15d=ir_15d,
        pmr_15d=pmr_15d,
        last_monitor_ts_utc=last_monitor_ts_utc,
    )


def plot_recent_metrics(days: int = 7) -> Optional[Path]:
    if not METRICS_CSV.exists():
        logging.warning("找不到 metrics 檔案，無法畫圖。")
        return None

    df = pd.read_csv(METRICS_CSV)
    if df.empty:
        logging.warning("metrics 為空，無法畫圖。")
        return None

    ts_col = detect_ts_column(df)
    if ts_col is None:
        logging.warning("metrics 無時間欄位，無法畫圖。")
        return None

    df["_ts_utc"] = to_utc_series(df[ts_col])
    df = df.dropna(subset=["_ts_utc"]).sort_values("_ts_utc")

    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - timedelta(days=days)
    df_recent = df[df["_ts_utc"] >= window_start]

    if df_recent.empty:
        logging.warning("最近 %d 天沒有資料，略過畫圖。", days)
        return None

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 4))

    x = df_recent["_ts_utc"]
    plotted = False

    if "IC_15D" in df_recent.columns:
        ax.plot(x, df_recent["IC_15D"], label="IC 15D")
        plotted = True
    elif "ic_15d" in df_recent.columns:
        ax.plot(x, df_recent["ic_15d"], label="IC 15D")
        plotted = True
    elif "ic_15d_rolling" in df_recent.columns:
        ax.plot(x, df_recent["ic_15d_rolling"], label="IC 15D")
        plotted = True

    if "IR_15D" in df_recent.columns:
        ax.plot(x, df_recent["IR_15D"], label="IR 15D")
        plotted = True
    elif "ir_15d" in df_recent.columns:
        ax.plot(x, df_recent["ir_15d"], label="IR 15D")
        plotted = True
    elif "ir_15d_rolling" in df_recent.columns:
        ax.plot(x, df_recent["ir_15d_rolling"], label="IR 15D")
        plotted = True

    if "PMR_15D" in df_recent.columns:
        ax.plot(x, df_recent["PMR_15D"], label="PMR 15D")
        plotted = True
    elif "pmr_15d" in df_recent.columns:
        ax.plot(x, df_recent["pmr_15d"], label="PMR 15D")
        plotted = True
    elif "pmr_15d_rolling" in df_recent.columns:
        ax.plot(x, df_recent["pmr_15d_rolling"], label="PMR 15D")
        plotted = True

    if not plotted:
        plt.close(fig)
        logging.warning("沒有 IC/IR/PMR 欄位，略過畫圖。")
        return None

    ax.set_xlabel("時間 (UTC)")
    ax.set_ylabel("指標值")
    ax.legend(loc="upper left")
    fig.tight_layout()

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logging.info("7 日指標圖已輸出：%s", PLOT_PATH)
    return PLOT_PATH


def build_daily_html(stats: DailyStats, plot_cid: Optional[str]) -> str:
    start_utc, end_utc, now_local = get_today_window_utc(offset_hours=8)

    send_time_local_str = now_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    monitor_time_str = (
        stats.last_monitor_ts_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
        if stats.last_monitor_ts_utc
        else "（尚無今日監控紀錄）"
    )

    success_rate = (
        stats.success_runs / stats.total_runs if stats.total_runs > 0 else None
    )
    success_rate_str = f"{success_rate:.1%}" if success_rate is not None else "—"

    ic_str = f"{stats.ic_15d:.4f}" if stats.ic_15d is not None else "—"
    ir_str = f"{stats.ir_15d:.4f}" if stats.ir_15d is not None else "—"
    pmr_str = f"{stats.pmr_15d:.2%}" if stats.pmr_15d is not None else "—"

    # 健康度一句話
    if stats.ic_15d is None:
        health = "UNKNOWN — 指標資料不足，無法判斷策略健康度。"
    elif stats.ic_15d < 0:
        health = "⚠️ 最近 15 日 IC 為負，策略預測方向表現偏弱，建議降低部位或暫停。"
    elif stats.ic_15d < 0.05:
        health = "🟡 最近 15 日 IC 為正但偏弱，可維持小倉位觀察。"
    else:
        health = "🟢 最近 15 日 IC 穩定為正，策略整體表現良好。"

    alerts_str = str(stats.alerts_today)

    metrics_img_html = (
        f'<img src="cid:{plot_cid}" alt="IC/IR/PMR 最近 7 天走勢" '
        f'style="width:100%;max-width:900px;border-radius:8px;" />'
        if plot_cid
        else "<p>暫無可用圖表。</p>"
    )

    html = f"""
<html>
  <body style="background-color:#111827;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#E5E7EB;">
    <div style="max-width:960px;margin:0 auto;padding:24px;">
      <h2 style="color:#F9FAFB;margin-bottom:4px;">【WTI每日綜合報告】策略日終總結</h2>
      <p style="color:#9CA3AF;margin-top:0;">寄送時間（本地）: {send_time_local_str}</p>
      <p style="color:#9CA3AF;margin-top:0;">最近一次監控執行時間（UTC）: {monitor_time_str}</p>

      <div style="background:linear-gradient(135deg,#0f172a,#111827);border-radius:16px;padding:16px 20px;margin-top:16px;border:1px solid #1F2937;">
        <h3 style="color:#F9FAFB;margin-top:0;margin-bottom:8px;">📌 今日白話總結</h3>
        <p style="color:#E5E7EB;line-height:1.6;">
          今日共執行 {stats.total_runs} 次監控，成功 {stats.success_runs} 次，失敗 {stats.failed_runs} 次，成功率 {success_rate_str}。<br/>
          今日警報數量：{alerts_str}。
        </p>
        <p style="color:#E5E7EB;line-height:1.6;">{health}</p>
      </div>

      <div style="margin-top:20px;background-color:#020617;border-radius:16px;padding:16px 20px;border:1px solid #1F2937;">
        <h3 style="color:#F9FAFB;margin-top:0;margin-bottom:8px;">🕒 今日執行統計</h3>
        <table style="width:100%;border-collapse:collapse;font-size:14px;">
          <thead>
            <tr>
              <th style="text-align:left;color:#9CA3AF;padding:4px 8px;border-bottom:1px solid #1F2937;">項目</th>
              <th style="text-align:right;color:#9CA3AF;padding:4px 8px;border-bottom:1px solid #1F2937;">數值</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding:4px 8px;">執行次數</td>
              <td style="padding:4px 8px;text-align:right;">{stats.total_runs}</td>
            </tr>
            <tr>
              <td style="padding:4px 8px;">成功次數</td>
              <td style="padding:4px 8px;text-align:right;">{stats.success_runs}</td>
            </tr>
            <tr>
              <td style="padding:4px 8px;">失敗次數</td>
              <td style="padding:4px 8px;text-align:right;">{stats.failed_runs}</td>
            </tr>
            <tr>
              <td style="padding:4px 8px;">成功率</td>
              <td style="padding:4px 8px;text-align:right;">{success_rate_str}</td>
            </tr>
            <tr>
              <td style="padding:4px 8px;">今日警報數</td>
              <td style="padding:4px 8px;text-align:right;">{alerts_str}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div style="margin-top:20px;background-color:#020617;border-radius:16px;padding:16px 20px;border:1px solid #1F2937;">
        <h3 style="color:#F9FAFB;margin-top:0;margin-bottom:8px;">📈 近期指標走勢（最近約 7 天）</h3>
        {metrics_img_html}
      </div>

      <div style="margin-top:20px;font-size:12px;color:#6B7280;">
        <p style="margin:0;">IC（Information Coefficient）：預測與實際報酬的相關性，越接近 1 越好。</p>
        <p style="margin:0;">IR（Information Ratio）：IC 經過波動調整後的穩定度指標。</p>
        <p style="margin:0;">PMR（Positive Match Ratio）：方向判對的比例，越高越好。</p>
      </div>
    </div>
  </body>
</html>
"""
    return html


def send_email(subject: str, html_body: str, img_path: Optional[Path]) -> None:
    cfg = load_email_config(EMAIL_CONFIG_PATH)

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = cfg["from_email"]
    msg["To"] = cfg["to_email"]

    alt = MIMEMultipart("alternative")
    msg.attach(alt)

    alt.attach(MIMEText("此郵件為 HTML 格式，請使用支援 HTML 的郵件客戶端查看。", "plain", "utf-8"))
    alt.attach(MIMEText(html_body, "html", "utf-8"))

    if img_path is not None and img_path.exists():
        with img_path.open("rb") as f:
            img = MIMEImage(f.read())
        img.add_header("Content-ID", "<daily_metrics_plot>")
        img.add_header("Content-Disposition", "inline", filename=img_path.name)
        msg.attach(img)

    with smtplib.SMTP_SSL(cfg["smtp_server"], cfg["smtp_port"]) as server:
        server.login(cfg["username"], cfg["password"])
        server.send_message(msg)

    logging.info("每日綜合報告已寄出。")


def main() -> None:
    logging.info("彙總今日執行統計與警報...")
    stats = summarize_today()

    logging.info("讀取 IC / IR / PMR 指標...")
    logging.info("繪製 7 日指標線圖...")
    img_path = plot_recent_metrics(days=7)

    html = build_daily_html(stats, "daily_metrics_plot" if img_path else None)

    _, _, now_local = get_today_window_utc(offset_hours=8)
    date_str = now_local.strftime("%Y-%m-%d")
    subject = f"【WTI每日綜合報告】 每日綜合報告 - {date_str}"

    logging.info("組每日綜合報告 HTML...")
    logging.info("寄送 email...")
    send_email(subject, html, img_path)


if __name__ == "__main__":
    main()