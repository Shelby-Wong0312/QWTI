"""Download 15y hourly prices from Capital.com API (DEMO by default) and save to CSV."""
import argparse
import csv
import logging
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import requests
from dotenv import dotenv_values

# -------- Endpoints --------
DEMO_BASE_URL = "https://demo-api-capital.backend-capital.com"
LIVE_BASE_URL = "https://api-capital.backend-capital.com"

# -------- Defaults --------
WTI_SEARCH_TERMS = ("WTI", "US Crude Oil", "US Crude Oil Spot", "OIL_CRUDE", "Oil - Crude")
DEFAULT_EPIC = "OIL_CRUDE"
DEFAULT_RESOLUTION = "HOUR"
MAX_POINTS_PER_CALL = 1000
POINTS_PER_SLICE = 900           # 900 * 1 hour ≈ 37.5 days per chunk
REQUEST_SLEEP_SECONDS = 0.12
MAX_RETRIES = 7
DEFAULT_LOOKBACK_YEARS = 15
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_CANDIDATES = [
    PROJECT_ROOT / ".env",
    SCRIPT_DIR / ".env",
    Path.cwd() / ".env",
]
OUTPUT_DIR = SCRIPT_DIR / "output"

RESOLUTION_TO_DELTA = {
    "MINUTE": timedelta(minutes=1),
    "MINUTE_2": timedelta(minutes=2),
    "MINUTE_3": timedelta(minutes=3),
    "MINUTE_5": timedelta(minutes=5),
    "MINUTE_10": timedelta(minutes=10),
    "MINUTE_15": timedelta(minutes=15),
    "MINUTE_30": timedelta(minutes=30),
    "HOUR": timedelta(hours=1),
    "HOUR_2": timedelta(hours=2),
    "HOUR_3": timedelta(hours=3),
    "HOUR_4": timedelta(hours=4),
    "DAY": timedelta(days=1),
    "WEEK": timedelta(days=7),
    "MONTH": timedelta(days=30),
}

# -------- Helpers --------
def locate_env_file() -> Path:
    for candidate in ENV_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Unable to locate .env file with Capital.com credentials")

def load_env(path: Path) -> Dict[str, str]:
    values = dotenv_values(str(path))
    return {k: v for k, v in values.items() if k and v is not None}

def normalize_resolution(value: str) -> str:
    r = value.upper()
    if r not in RESOLUTION_TO_DELTA:
        supported = ", ".join(sorted(RESOLUTION_TO_DELTA.keys()))
        raise ValueError(f"Unsupported resolution '{value}'. Supported: {supported}")
    return r

def resolution_slice_delta(resolution: str) -> timedelta:
    return RESOLUTION_TO_DELTA[resolution] * POINTS_PER_SLICE

def format_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def align_to_midnight(dt: datetime) -> datetime:
    dt_utc = dt.astimezone(timezone.utc)
    return datetime(dt_utc.year, dt_utc.month, dt_utc.day, tzinfo=timezone.utc)

def parse_cli_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    dt = datetime.strptime(value, "%Y-%m-%d")
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)

def determine_date_range(from_date: Optional[str], to_date: Optional[str], years: int) -> Tuple[datetime, datetime]:
    now_utc = datetime.now(timezone.utc)
    end_dt = parse_cli_date(to_date) or align_to_midnight(now_utc)
    if from_date:
        start_dt = parse_cli_date(from_date)
    else:
        # years → approx days; +5 accounts for leap days and inclusive rounding
        start_dt = align_to_midnight(end_dt - timedelta(days=years * 365 + math.floor(years / 4) + 5))
    if not start_dt or end_dt <= start_dt:
        raise ValueError(f"'--to' ({end_dt}) must be later than '--from' ({start_dt})")
    return start_dt, end_dt

def load_checkpoint(path: Path) -> Optional[datetime]:
    try:
        content = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not content:
        return None
    try:
        dt = datetime.fromisoformat(content)
    except ValueError:
        logging.warning("Ignoring invalid checkpoint file %s", path)
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def save_checkpoint(path: Path, dt: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dt.isoformat(), encoding="utf-8")

def generate_output_path(epic: str, resolution: str, start_dt: datetime, end_dt: datetime, is_demo: bool) -> Path:
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    env_tag = "DEMO" if is_demo else "LIVE"
    return OUTPUT_DIR / f"{epic}_{resolution}_{env_tag}_{start_str}_{end_str}.csv"

# -------- Client --------
class CapitalClient:
    """Minimal Capital.com REST client for authenticated price requests."""

    def __init__(self, api_key: str, identifier: str, password: str, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-CAP-API-KEY": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        self._identifier = identifier
        self._password = password
        self._authenticate()

    def _sleep_with_jitter(self, base: float) -> None:
        time.sleep(base + random.random() * base)

    def _authenticate(self) -> None:
        payload = {
            "identifier": self._identifier,
            "password": self._password,
            "encryptedPassword": False,
        }
        resp = self.session.post(f"{self.base_url}/api/v1/session", json=payload, timeout=30)
        time.sleep(REQUEST_SLEEP_SECONDS)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            logging.error("Authentication failed: %s", resp.text)
            raise exc
        cst = resp.headers.get("CST")
        sec = resp.headers.get("X-SECURITY-TOKEN")
        if not cst or not sec:
            raise RuntimeError("Missing authentication tokens in Capital.com response")
        self.session.headers.update({"CST": cst, "X-SECURITY-TOKEN": sec})

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, object]] = None,
        json: Optional[Dict[str, object]] = None,
        allow_without_to: bool = False,
    ) -> requests.Response:
        use_to = True
        for attempt in range(1, MAX_RETRIES + 1):
            p = dict(params) if params else None
            if allow_without_to and not use_to and p:
                p.pop("to", None)
            resp = self.session.request(
                method,
                f"{self.base_url}{path}",
                params=p,
                json=json,
                timeout=30,
            )
            time.sleep(REQUEST_SLEEP_SECONDS)
            if resp.status_code == 401:
                logging.warning("401 from Capital.com, refreshing session (attempt %s)", attempt)
                self._authenticate()
                continue
            if resp.status_code == 429:
                # exponential backoff with jitter
                wait = min(2 ** (attempt - 1), 60)
                logging.warning("Rate limited (429). Sleeping ~%ss", wait)
                self._sleep_with_jitter(wait)
                continue
            if resp.status_code == 400 and allow_without_to and use_to:
                logging.warning("400 response; retrying without 'to' parameter")
                use_to = False
                continue
            if resp.status_code == 404:
                return resp
            try:
                resp.raise_for_status()
                return resp
            except requests.HTTPError:
                logging.error("Request failed (%s - %s)", resp.status_code, resp.text)
                if attempt == MAX_RETRIES:
                    raise
                self._sleep_with_jitter(1.0)
        raise RuntimeError("Exceeded retry attempts while calling Capital.com")

    def discover_epic(self, search_terms: Iterable[str], fallback: str) -> str:
        keywords = ("crude", "wti")
        for term in search_terms:
            r = self._request("GET", "/api/v1/markets", params={"searchTerm": term})
            if r.status_code == 404:
                continue
            for m in r.json().get("markets", []):
                epic = m.get("epic")
                if not epic:
                    continue
                label = " ".join(
                    str(v) for v in (m.get("instrumentName"), m.get("symbol"), epic) if v
                ).lower()
                if any(k in label for k in keywords):
                    logging.info("Discovered WTI epic %s via searchTerm=%s", epic, term)
                    return epic
        logging.info("Falling back to default WTI epic %s", fallback)
        return fallback

    def _fetch_prices_slice(
        self,
        epic: str,
        start: datetime,
        end: datetime,
        *,
        resolution: str,
    ) -> List[Dict[str, object]]:
        params = {
            "resolution": resolution,
            "from": format_timestamp(start),
            "to": format_timestamp(end),
            "max": min(POINTS_PER_SLICE, MAX_POINTS_PER_CALL),
        }
        r = self._request("GET", f"/api/v1/prices/{epic}", params=params, allow_without_to=True)
        if r.status_code == 404:
            return []
        return r.json().get("prices", [])

    def fetch_prices(
        self,
        epic: str,
        start_dt: datetime,
        end_dt: datetime,
        resolution: str,
        *,
        on_chunk_complete: Optional[Callable[[datetime], None]] = None,
    ) -> List[Dict[str, object]]:
        if end_dt <= start_dt:
            return []
        combined: Dict[str, Dict[str, object]] = {}
        current = start_dt
        slice_delta = resolution_slice_delta(resolution)
        chunk_index = 0
        while current < end_dt:
            chunk_index += 1
            slice_end = min(current + slice_delta, end_dt)
            logging.info(
                "Fetching %s prices chunk %s: %s -> %s",
                resolution, chunk_index, current.isoformat(), slice_end.isoformat(),
            )
            prices = self._fetch_prices_slice(epic, current, slice_end, resolution=resolution)
            # Upsert by timestamp to dedup
            for p in prices:
                key = p.get("snapshotTimeUTC") or p.get("snapshotTime")
                if key:
                    combined[key] = p
            if on_chunk_complete:
                on_chunk_complete(slice_end)
            current = slice_end
        # Return sorted by key
        return [combined[k] for k in sorted(combined.keys())]

# -------- Transform / IO --------
def transform_price(p: Dict[str, object]) -> Dict[str, object]:
    open_bid = (p.get("openPrice") or {}).get("bid")
    open_ask = (p.get("openPrice") or {}).get("ask")
    high_bid = (p.get("highPrice") or {}).get("bid")
    high_ask = (p.get("highPrice") or {}).get("ask")
    low_bid = (p.get("lowPrice") or {}).get("bid")
    low_ask = (p.get("lowPrice") or {}).get("ask")
    close_bid = (p.get("closePrice") or {}).get("bid")
    close_ask = (p.get("closePrice") or {}).get("ask")

    def mid(a, b):
        try:
            return (float(a) + float(b)) / 2.0 if a is not None and b is not None else None
        except Exception:
            return None

    return {
        "snapshotTimeUTC": p.get("snapshotTimeUTC"),
        "open_bid": open_bid,
        "open_ask": open_ask,
        "high_bid": high_bid,
        "high_ask": high_ask,
        "low_bid": low_bid,
        "low_ask": low_ask,
        "close_bid": close_bid,
        "close_ask": close_ask,
        "open_mid": mid(open_bid, open_ask),
        "high_mid": mid(high_bid, high_ask),
        "low_mid": mid(low_bid, low_ask),
        "close_mid": mid(close_bid, close_ask),
        "lastTradedVolume": p.get("lastTradedVolume"),
    }

def write_prices_to_csv(records: List[Dict[str, object]], output_path: Path) -> None:
    if not records:
        raise RuntimeError("No price data retrieved from Capital.com")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

# -------- CLI --------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download 15y hourly history from Capital.com API")
    p.add_argument("--from", dest="from_date", type=str, help="Start date (UTC) YYYY-MM-DD")
    p.add_argument("--to", dest="to_date", type=str, help="End date (UTC) YYYY-MM-DD")
    p.add_argument("--years", type=int, default=DEFAULT_LOOKBACK_YEARS, help="Lookback years if --from not set (default: 15)")
    p.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION, help=f"Candle resolution (default: {DEFAULT_RESOLUTION})")
    p.add_argument("--checkpoint", type=Path, help="Path to checkpoint file for resuming")
    p.add_argument("--epic", type=str, default=None, help="Override epic (skip discovery)")
    p.add_argument("--live", action="store_true", help="Use LIVE API (default uses DEMO)")
    return p.parse_args()

# -------- Main --------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    env_path = locate_env_file()
    cfg = load_env(env_path)
    api_key = cfg.get("CAPITAL_API_KEY")
    identifier = cfg.get("CAPITAL_IDENTIFIER")
    password = cfg.get("CAPITAL_API_KEY_PASSWORD") or cfg.get("CAPITAL_API_PASSWORD")
    missing = [n for n, v in [("CAPITAL_API_KEY", api_key), ("CAPITAL_IDENTIFIER", identifier), ("CAPITAL_API_KEY_PASSWORD", password)] if not v]
    if missing:
        raise RuntimeError(f"Missing required credentials in {env_path}: {', '.join(missing)}")

    resolution = normalize_resolution(args.resolution or DEFAULT_RESOLUTION)
    start_dt, end_dt = determine_date_range(args.from_date, args.to_date, years=args.years)

    checkpoint_path: Optional[Path] = args.checkpoint
    if checkpoint_path:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        cp_dt = load_checkpoint(checkpoint_path)
        if cp_dt:
            if cp_dt >= end_dt:
                logging.info("Checkpoint covers requested range; no data to fetch")
                start_dt = end_dt
            else:
                start_dt = max(start_dt, cp_dt)
                logging.info("Resuming from checkpoint %s", start_dt.isoformat())

    if start_dt >= end_dt:
        logging.info("No data to download for the requested range")
        return

    base_url = LIVE_BASE_URL if args.live else DEMO_BASE_URL
    client = CapitalClient(api_key=api_key, identifier=identifier, password=password, base_url=base_url)

    epic = args.epic or client.discover_epic(WTI_SEARCH_TERMS, DEFAULT_EPIC)
    logging.info("Using epic %s (%s)", epic, "LIVE" if args.live else "DEMO")

    output_path = generate_output_path(epic, resolution, start_dt, end_dt, is_demo=not args.live)

    def checkpoint_callback(chunk_end: datetime) -> None:
        if checkpoint_path:
            save_checkpoint(checkpoint_path, chunk_end)

    prices = client.fetch_prices(
        epic, start_dt, end_dt, resolution, on_chunk_complete=checkpoint_callback
    )
    # Transform, sort (safety), dedup by timestamp
    seen = set()
    ordered = []
    for p in prices:
        ts = p.get("snapshotTimeUTC") or p.get("snapshotTime")
        if not ts or ts in seen:
            continue
        seen.add(ts)
        ordered.append(transform_price(p))

    # Basic sanity check: expect ~ years * 365 * 24 points (允許市場休市/缺漏)
    exp = args.years * 365 * 24
    if len(ordered) < int(exp * 0.6):  # 寬鬆檢查，避免 demo 環境或市場節假日造成誤報
        logging.warning("Rows (%s) are far less than expected (~%s). Check EPIC / environment / product history.",
                        len(ordered), exp)

    write_prices_to_csv(ordered, output_path)
    logging.info("Saved %s rows to %s", len(ordered), output_path)
    logging.info("Finished export for %s: %s rows -> %s", epic, len(ordered), output_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.error("Script terminated with error: %s", exc)
        sys.exit(1)
