#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
send_hourly_email.py

功能：寄出「WTI 每小時監控」報告
- 讀取最新一筆監控指標（metrics）
- 讀取最近 24 小時內的警報
- 畫最近 7 天 IC / IR / PMR 走勢圖
- 用 email_config.json 的設定寄出 HTML 報告
"""

import json
import logging
import smtplib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------
# 路徑與 logging 設定
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
WAREHOUSE_DIR = PROJECT_ROOT / "warehouse"
MONITOR_DIR = WAREHOUSE_DIR / "monitoring"
POSITIONS_DIR = WAREHOUSE_DIR / "positions"

METRICS_CSV = MONITOR_DIR / "base_seed202_lean7_metrics.csv"
ALERTS_CSV = MONITOR_DIR / "base_seed202_lean7_alerts.csv"
EXEC_LOG_CSV = MONITOR_DIR / "hourly_execution_log.csv"

HOURLY_PLOT_PATH = MONITOR_DIR / "hourly_metrics_plot.png"
EMAIL_CONFIG_PATH = PROJECT_ROOT / "email_config.json"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------
# 共用小工具
# ---------------------------------------------------------------------


def load_email_config(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"email 設定檔不存在：{path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_keys = [
        "smtp_server",
        "smtp_port",
        "username",
        "password",
        "from_email",
        "to_email",
    ]
    missing = [k for k in required_keys if not cfg.get(k)]
    if missing:
        raise ValueError(f"email_config.json 缺少必要欄位: {missing}")
    return cfg


def detect_ts_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["ts", "timestamp", "as_of", "as_of_utc", "time"]
    for c in candidates:
        if c in df.columns:
            return c
    # 再試一次：找 dtype 是 datetime 的欄位
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def to_utc_series(series: pd.Series) -> pd.Series:
    """一律轉成 UTC aware，避免 tz-naive / tz-aware 比較錯誤"""
    s = pd.to_datetime(series, errors="coerce", utc=True)
    return s


def pick_metric(row: pd.Series, candidates: List[str]) -> Optional[float]:
    for c in candidates:
        if c in row and pd.notna(row[c]):
            return float(row[c])
    return None


@dataclass
class HourlySnapshot:
    data_ts: Optional[datetime]
    ic_15d: Optional[float]
    ir_15d: Optional[float]
    pmr_15d: Optional[float]


# ---------------------------------------------------------------------
# 資料讀取與摘要邏輯
# ---------------------------------------------------------------------


def load_latest_metrics() -> HourlySnapshot:
    if not METRICS_CSV.exists():
        logging.warning("找不到 metrics 檔案：%s", METRICS_CSV)
        return HourlySnapshot(None, None, None, None)

    df = pd.read_csv(METRICS_CSV)
    if df.empty:
        logging.warning("metrics 檔案為空：%s", METRICS_CSV)
        return HourlySnapshot(None, None, None, None)

    ts_col = detect_ts_column(df)
    if ts_col is None:
        logging.warning("metrics 無時間欄位，無法判斷最新一筆")
        latest = df.iloc[-1]
        ts_val = None
    else:
        df["_ts_utc"] = to_utc_series(df[ts_col])
        df = df.sort_values("_ts_utc")
        latest = df.iloc[-1]
        ts_val = latest["_ts_utc"]
        if pd.isna(ts_val):
            ts_val = None

    ic_15d = pick_metric(latest, ["IC_15D", "ic_15d", "ic_15d_rolling"])
    ir_15d = pick_metric(latest, ["IR_15D", "ir_15d", "ir_15d_rolling"])
    pmr_15d = pick_metric(latest, ["PMR_15D", "pmr_15d", "pmr_15d_rolling"])

    return HourlySnapshot(
        data_ts=ts_val.to_pydatetime() if isinstance(ts_val, pd.Timestamp) else ts_val,
        ic_15d=ic_15d,
        ir_15d=ir_15d,
        pmr_15d=pmr_15d,
    )


def load_recent_alerts(hours: int = 24) -> pd.DataFrame:
    if not ALERTS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(ALERTS_CSV)
    if df.empty:
        return df

    ts_col = detect_ts_column(df)
    if ts_col is None:
        return df.iloc[0:0]

    df["_ts_utc"] = to_utc_series(df[ts_col])
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=hours)
    recent = df[df["_ts_utc"] >= cutoff].sort_values("_ts_utc", ascending=False)
    return recent


def plot_recent_metrics(days: int = 7) -> Optional[Path]:
    if not METRICS_CSV.exists():
        logging.warning("找不到 metrics 檔案，無法畫圖：%s", METRICS_CSV)
        return None

    df = pd.read_csv(METRICS_CSV)
    if df.empty:
        logging.warning("metrics 檔案為空，無法畫圖")
        return None

    ts_col = detect_ts_column(df)
    if ts_col is None:
        logging.warning("metrics 無時間欄位，無法畫圖")
        return None

    df["_ts_utc"] = to_utc_series(df[ts_col])
    df = df.dropna(subset=["_ts_utc"]).sort_values("_ts_utc")

    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - timedelta(days=days)
    df_recent = df[df["_ts_utc"] >= window_start]

    if df_recent.empty:
        logging.warning("最近 %d 天沒有 metrics 資料，略過畫圖", days)
        return None

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 4))

    x = df_recent["_ts_utc"]

    plotted_any = False

    ic = pick_metric(df_recent.iloc[-1], ["IC_15D", "ic_15d", "ic_15d_rolling"])
    if "IC_15D" in df_recent.columns:
        ax.plot(x, df_recent["IC_15D"], label="IC 15D")
        plotted_any = True
    elif "ic_15d" in df_recent.columns:
        ax.plot(x, df_recent["ic_15d"], label="IC 15D")
        plotted_any = True
    elif "ic_15d_rolling" in df_recent.columns:
        ax.plot(x, df_recent["ic_15d_rolling"], label="IC 15D")
        plotted_any = True

    if "IR_15D" in df_recent.columns:
        ax.plot(x, df_recent["IR_15D"], label="IR 15D")
        plotted_any = True
    elif "ir_15d" in df_recent.columns:
        ax.plot(x, df_recent["ir_15d"], label="IR 15D")
        plotted_any = True
    elif "ir_15d_rolling" in df_recent.columns:
        ax.plot(x, df_recent["ir_15d_rolling"], label="IR 15D")
        plotted_any = True

    if "PMR_15D" in df_recent.columns:
        ax.plot(x, df_recent["PMR_15D"], label="PMR 15D")
        plotted_any = True
    elif "pmr_15d" in df_recent.columns:
        ax.plot(x, df_recent["pmr_15d"], label="PMR 15D")
        plotted_any = True
    elif "pmr_15d_rolling" in df_recent.columns:
        ax.plot(x, df_recent["pmr_15d_rolling"], label="PMR 15D")
        plotted_any = True

    if not plotted_any:
        logging.warning("找不到 IC/IR/PMR 欄位，略過畫圖")
        plt.close(fig)
        return None

    ax.set_xlabel("時間 (UTC)")
    ax.set_ylabel("指標值")
    ax.legend(loc="upper left")
    fig.tight_layout()

    HOURLY_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(HOURLY_PLOT_PATH, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    logging.info("每小時圖表已輸出：%s", HOURLY_PLOT_PATH)
    return HOURLY_PLOT_PATH


# ---------------------------------------------------------------------
# Email 組裝與寄送
# ---------------------------------------------------------------------


def build_hourly_html(
    snapshot: HourlySnapshot,
    recent_alerts: pd.DataFrame,
    metrics_img_cid: Optional[str],
) -> str:
    now_local = datetime.now(timezone(timedelta(hours=8)))
    send_time_local_str = now_local.strftime("%Y-%m-%d %H:%M:%S %Z")

    if snapshot.data_ts is not None:
        data_ts_utc = snapshot.data_ts.replace(tzinfo=timezone.utc)
        data_ts_str = data_ts_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        data_ts_str = "（無資料）"

    ic_str = f"{snapshot.ic_15d:.4f}" if snapshot.ic_15d is not None else "—"
    ir_str = f"{snapshot.ir_15d:.4f}" if snapshot.ir_15d is not None else "—"
    pmr_str = f"{snapshot.pmr_15d:.2%}" if snapshot.pmr_15d is not None else "—"

    # 判斷一句話健康度（簡單版）
    if snapshot.ic_15d is None:
        health = "UNKNOWN — 指標資料不足，無法判斷策略健康度。"
    elif snapshot.ic_15d < 0:
        health = "⚠️ 當前 15 日 IC 為負，預測方向表現偏弱，請留意。"
    elif snapshot.ic_15d < 0.05:
        health = "🟡 當前 15 日 IC 為正但偏弱，暫時維持觀望或小倉位。"
    else:
        health = "🟢 當前 15 日 IC 穩定為正，策略預測表現良好。"

    # 警報文字
    if recent_alerts.empty:
        alerts_html = "<p>最近 24 小時尚無新警報。</p>"
    else:
        rows = []
        ts_col = detect_ts_column(recent_alerts) or "_ts_utc"
        for _, r in recent_alerts.head(5).iterrows():
            ts_val = r.get("_ts_utc") or r.get(ts_col)
            ts_val = pd.to_datetime(ts_val, errors="coerce", utc=True)
            ts_str = (
                ts_val.astimezone(timezone(timedelta(hours=8))).strftime(
                    "%Y-%m-%d %H:%M"
                )
                if not pd.isna(ts_val)
                else ""
            )
            level = r.get("level", "")
            message = r.get("message", r.get("alert", ""))
            rows.append(f"<li><b>{ts_str}</b> [{level}] {message}</li>")
        alerts_html = "<ul>" + "".join(rows) + "</ul>"

    metrics_img_html = (
        f'<img src="cid:{metrics_img_cid}" alt="IC/IR/PMR 最近 7 天走勢" '
        f'style="width:100%;max-width:900px;border-radius:8px;" />'
        if metrics_img_cid
        else "<p>暫無可用圖表。</p>"
    )

    html = f"""
<html>
  <body style="background-color:#111827;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#E5E7EB;">
    <div style="max-width:960px;margin:0 auto;padding:24px;">
      <h2 style="color:#F9FAFB;margin-bottom:4px;">【WTI每小時監控】策略狀態快照</h2>
      <p style="color:#9CA3AF;margin-top:0;">寄送時間（本地）: {send_time_local_str}</p>
      <p style="color:#9CA3AF;margin-top:0;">最近一次監控資料時間（UTC）: {data_ts_str}</p>

      <div style="background:linear-gradient(135deg,#0f172a,#111827);border-radius:16px;padding:16px 20px;margin-top:16px;border:1px solid #1F2937;">
        <h3 style="color:#F9FAFB;margin-top:0;margin-bottom:8px;">📌 最近一小時白話總結</h3>
        <p style="color:#E5E7EB;line-height:1.6;">{health}</p>
      </div>

      <div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:16px;">
        <div style="flex:1 1 160px;background-color:#111827;border-radius:12px;padding:12px 14px;border:1px solid #1F2937;">
          <div style="font-size:12px;color:#9CA3AF;">15日 IC</div>
          <div style="font-size:22px;font-weight:600;color:#F9FAFB;margin-top:4px;">{ic_str}</div>
        </div>
        <div style="flex:1 1 160px;background-color:#111827;border-radius:12px;padding:12px 14px;border:1px solid #1F2937;">
          <div style="font-size:12px;color:#9CA3AF;">15日 IR</div>
          <div style="font-size:22px;font-weight:600;color:#F9FAFB;margin-top:4px;">{ir_str}</div>
        </div>
        <div style="flex:1 1 160px;background-color:#111827;border-radius:12px;padding:12px 14px;border:1px solid #1F2937;">
          <div style="font-size:12px;color:#9CA3AF;">15日 PMR</div>
          <div style="font-size:22px;font-weight:600;color:#F9FAFB;margin-top:4px;">{pmr_str}</div>
        </div>
      </div>

      <div style="margin-top:24px;background-color:#020617;border-radius:16px;padding:16px 20px;border:1px solid #1F2937;">
        <h3 style="color:#F9FAFB;margin-top:0;margin-bottom:8px;">📈 最近 7 天 IC / IR / PMR 走勢</h3>
        {metrics_img_html}
      </div>

      <div style="margin-top:24px;background-color:#020617;border-radius:16px;padding:16px 20px;border:1px solid #1F2937;">
        <h3 style="color:#F9FAFB;margin-top:0;margin-bottom:8px;">⚠️ 最近 24 小時警報</h3>
        {alerts_html}
      </div>

      <div style="margin-top:24px;font-size:12px;color:#6B7280;">
        <p style="margin:0;">本郵件為系統自動發送，用於 WTI 策略每小時健康度監控。</p>
      </div>
    </div>
  </body>
</html>
"""
    return html


def send_email(subject: str, html_body: str, image_path: Optional[Path]) -> None:
    cfg = load_email_config(EMAIL_CONFIG_PATH)

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = cfg["from_email"]
    msg["To"] = cfg["to_email"]

    alt = MIMEMultipart("alternative")
    msg.attach(alt)

    alt.attach(MIMEText("此郵件為 HTML 格式，請使用支援 HTML 的郵件客戶端查看。", "plain", "utf-8"))
    alt.attach(MIMEText(html_body, "html", "utf-8"))

    if image_path is not None and image_path.exists():
        with image_path.open("rb") as f:
            img = MIMEImage(f.read())
        img.add_header("Content-ID", "<metrics_plot>")
        img.add_header("Content-Disposition", "inline", filename=image_path.name)
        msg.attach(img)

    with smtplib.SMTP_SSL(cfg["smtp_server"], cfg["smtp_port"]) as server:
        server.login(cfg["username"], cfg["password"])
        server.send_message(msg)

    logging.info("每小時監控郵件已寄出。")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main() -> None:
    logging.info("讀取最新監控指標 ...")
    snapshot = load_latest_metrics()

    logging.info("載入最近 24 小時警報 ...")
    recent_alerts = load_recent_alerts(hours=24)

    logging.info("繪製最近 7 天 IC/IR/PMR 圖表 ...")
    img_path = plot_recent_metrics(days=7)

    html = build_hourly_html(snapshot, recent_alerts, "metrics_plot" if img_path else None)

    now_local = datetime.now(timezone(timedelta(hours=8)))
    date_str = now_local.strftime("%Y-%m-%d")
    subject = f"【WTI每小時監控】 每小時監控摘要 - {date_str}"

    logging.info("寄送 email ...")
    send_email(subject, html, img_path)


if __name__ == "__main__":
    main()
