#!/usr/bin/env python
"""
Pull WTI hourly prices from Capital.com REST API (pure requests, no third-party SDK)
"""
import os
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd


LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def get_env(name: str, required: bool = True, default=None):
    val = os.environ.get(name, default)
    if required and not val:
        raise SystemExit(f"[ERROR] Missing env var: {name}")
    return val


class CapitalClient:
    """Simple Capital.com REST API client using requests."""

    def __init__(self, api_key: str, identifier: str, password: str, demo: bool = True):
        self.api_key = api_key
        self.identifier = identifier
        self.password = password
        self.demo = demo

        if demo:
            self.base_url = "https://demo-api-capital.backend-capital.com"
        else:
            self.base_url = "https://api-capital.backend-capital.com"

        self.cst = None
        self.security_token = None

    def _start_session(self):
        url = f"{self.base_url}/api/v1/session"
        headers = {
            "X-CAP-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "identifier": self.identifier,
            "password": self.password,
            "encryptedPassword": False,
        }

        logging.info("Starting session at %s (demo=%s)", self.base_url, self.demo)
        resp = requests.post(url, json=payload, headers=headers, timeout=30)

        if resp.status_code != 200:
            raise SystemExit(
                f"[ERROR] POST /session failed: {resp.status_code} {resp.text}"
            )

        self.cst = resp.headers.get("CST")
        self.security_token = resp.headers.get("X-SECURITY-TOKEN")

        if not self.cst or not self.security_token:
            raise SystemExit(
                "[ERROR] No CST or X-SECURITY-TOKEN in response headers. "
                "Check your API key/password."
            )

        logging.info("Session started. CST=%s..., TOKEN=%s...",
                     self.cst[:8] if self.cst else "N/A",
                     self.security_token[:8] if self.security_token else "N/A")

    def _auth_headers(self) -> dict:
        if not self.cst or not self.security_token:
            self._start_session()
        return {
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.security_token,
        }

    def search_markets(self, search_term: str) -> list:
        url = f"{self.base_url}/api/v1/markets"
        params = {"searchTerm": search_term}
        resp = requests.get(url, params=params, headers=self._auth_headers(), timeout=30)

        if resp.status_code != 200:
            raise SystemExit(f"[ERROR] GET /markets failed: {resp.status_code} {resp.text}")

        data = resp.json()
        return data.get("markets", [])

    def get_prices(self, epic: str, resolution: str = "HOUR", max_bars: int = 48) -> dict:
        url = f"{self.base_url}/api/v1/prices/{epic}"
        params = {
            "resolution": resolution,
            "max": max_bars,
        }
        resp = requests.get(url, params=params, headers=self._auth_headers(), timeout=30)

        if resp.status_code != 200:
            raise SystemExit(f"[ERROR] GET /prices/{epic} failed: {resp.status_code} {resp.text}")

        return resp.json()


def resolve_wti_epic(client: CapitalClient) -> str:
    epic_env = os.environ.get("CAPITAL_WTI_EPIC")
    if epic_env:
        logging.info("Using WTI epic from env CAPITAL_WTI_EPIC=%s", epic_env)
        return epic_env

    search_term = os.environ.get("CAPITAL_WTI_SEARCH", "OIL_CRUDE")
    logging.info("CAPITAL_WTI_EPIC not set, searching: %s", search_term)

    markets = client.search_markets(search_term)
    if not markets:
        raise SystemExit(
            f"[ERROR] search_markets('{search_term}') returned nothing. "
            "Set CAPITAL_WTI_EPIC explicitly."
        )

    m0 = markets[0]
    epic = m0["epic"]
    name = m0.get("instrumentName", "")
    logging.info("Auto-selected epic=%s name=%s", epic, name)
    logging.info("Tip: set CAPITAL_WTI_EPIC=%s to skip search next time", epic)
    return epic


def parse_timestamp(s: str) -> datetime:
    if not s:
        raise ValueError("empty timestamp")

    s = s.strip()
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def extract_close(price_obj: dict):
    if not price_obj:
        return None

    close_struct = price_obj.get("closePrice", {})
    bid = close_struct.get("bid")
    ask = close_struct.get("ask")

    if bid is not None and ask is not None:
        return (float(bid) + float(ask)) / 2.0
    if bid is not None:
        return float(bid)
    if ask is not None:
        return float(ask)
    return None


def fetch_hourly_prices(client: CapitalClient, epic: str, hours_back: int) -> pd.DataFrame:
    logging.info("Fetching hourly prices: epic=%s, hours_back=%d", epic, hours_back)

    data = client.get_prices(epic, resolution="HOUR", max_bars=hours_back)
    prices = data.get("prices", [])

    if not prices:
        raise SystemExit("[ERROR] No prices returned. Check epic / API credentials.")

    rows = []
    for p in prices:
        ts_str = p.get("snapshotTimeUTC") or p.get("snapshotTime")
        if not ts_str:
            continue
        try:
            ts = parse_timestamp(ts_str)
        except Exception as exc:
            logging.warning("skip bad ts=%s: %s", ts_str, exc)
            continue

        close = extract_close(p)
        if close is None:
            logging.warning("skip ts=%s: no close price", ts)
            continue

        rows.append({"ts_utc": ts, "wti_close": close})

    if not rows:
        raise SystemExit("[ERROR] No valid price bars after parsing")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    logging.info("Fetched %d hourly bars: %s -> %s",
                 len(df), df["ts_utc"].min(), df["ts_utc"].max())
    return df


def save_output(df: pd.DataFrame, out_parquet: Path, out_csv: Path):
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    logging.info("Saved parquet: %s", out_parquet)
    logging.info("Saved csv    : %s", out_csv)


def main():
    parser = argparse.ArgumentParser(
        description="Pull WTI hourly prices from Capital.com REST API"
    )
    parser.add_argument("--hours-back", type=int, default=48)
    parser.add_argument("--out-parquet", type=str, default="data/wti_hourly_capital.parquet")
    parser.add_argument("--out-csv", type=str, default="data/wti_hourly_capital.csv")
    parser.add_argument("--log-level", type=str, default="INFO", choices=LOG_LEVELS.keys())
    args = parser.parse_args()

    logging.basicConfig(
        level=LOG_LEVELS[args.log_level],
        format="%(asctime)s %(levelname)s %(message)s",
    )

    api_key = get_env("CAPITAL_API_KEY")
    identifier = get_env("CAPITAL_IDENTIFIER")
    password = get_env("CAPITAL_API_PASSWORD")
    demo_flag = get_env("CAPITAL_DEMO_MODE", required=False, default="True")
    demo = str(demo_flag).lower() in {"1", "true", "yes", "y"}

    client = CapitalClient(api_key, identifier, password, demo=demo)
    epic = resolve_wti_epic(client)
    df = fetch_hourly_prices(client, epic, args.hours_back)
    save_output(df, Path(args.out_parquet), Path(args.out_csv))


if __name__ == "__main__":
    main()
