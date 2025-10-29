import argparse
from typing import Optional
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

import pandas as pd
from google.cloud import bigquery


SQL = """
WITH base AS (
  SELECT
    TIMESTAMP_TRUNC(TIMESTAMP(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(Date AS STRING))), HOUR) AS ts_hour_utc,
    SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
    LOWER(V2Themes) AS themes,
    DocumentIdentifier AS url
  FROM `gdelt-bq.gdeltv2.gkg`
  WHERE Date BETWEEN CAST(FORMAT_TIMESTAMP('%Y%m%d%H%M%S', TIMESTAMP(@from)) AS INT64)
                  AND CAST(FORMAT_TIMESTAMP('%Y%m%d%H%M%S', TIMESTAMP(@to)) AS INT64)
),
oil AS (
  SELECT ts_hour_utc, tone, url
  FROM base
  WHERE REGEXP_CONTAINS(themes, r'(opec|crudeoil|oil_price|refinery|oilstocks|petroleum|brent|wti)')
     OR REGEXP_CONTAINS(url, r'(crude|wti|opec|oil)')
)
SELECT
  ts_hour_utc AS ts,
  COUNT(*) AS art_cnt,
  AVG(tone) AS tone_avg,
  COUNTIF(tone > 0) / COUNT(*) AS tone_pos_ratio
FROM oil
GROUP BY ts
ORDER BY ts
"""


def parse_cli_date(value: str) -> datetime:
    dt = datetime.strptime(value, "%Y-%m-%d")
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)


def determine_range(from_str: Optional[str], to_str: Optional[str]) -> tuple[datetime, datetime]:
    today = datetime.now(timezone.utc)
    end_dt = parse_cli_date(to_str) if to_str else datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    start_dt = parse_cli_date(from_str) if from_str else end_dt - pd.Timedelta(days=7)
    if end_dt <= start_dt:
        raise ValueError("'--to' must be later than '--from'")
    return start_dt, end_dt


def run_query(client: bigquery.Client, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("from", "TIMESTAMP", start_dt),
            bigquery.ScalarQueryParameter("to", "TIMESTAMP", end_dt),
        ]
    )
    query_job = client.query(SQL, job_config=job_config)
    df = query_job.result().to_dataframe(create_bqstorage_client=True)
    if df.empty:
        return pd.DataFrame(columns=["ts", "art_cnt", "tone_avg", "tone_pos_ratio"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def fill_missing_hours(df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    idx = pd.date_range(start=start_dt, end=end_dt, freq="H", tz=timezone.utc)
    df = df.set_index("ts").reindex(idx)
    df.index.name = "ts"
    df["art_cnt"] = df["art_cnt"].fillna(0).astype(int)
    df["tone_avg"] = df["tone_avg"].fillna(0.0)
    df["tone_pos_ratio"] = df["tone_pos_ratio"].fillna(0.5)
    return df.reset_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch GDELT hourly oil sentiment data into CSV")
    parser.add_argument("--from", dest="from_date", type=str, help="Start date (UTC) in YYYY-MM-DD format")
    parser.add_argument("--to", dest="to_date", type=str, help="End date (UTC) in YYYY-MM-DD format (exclusive)")
    parser.add_argument("--output", type=str, default="data/gdelt_hourly.csv", help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_dt, end_dt = determine_range(args.from_date, args.to_date)
    client = bigquery.Client()
    df = run_query(client, start_dt, end_dt)
    df = fill_missing_hours(df, start_dt, end_dt)
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[["ts", "art_cnt", "tone_avg", "tone_pos_ratio"]].to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

