import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UTC = timezone.utc
REQUEST_PAUSE_SECONDS = 0.2
MAX_POINTS_PER_SLICE = 900
RESOLUTION = "HOUR"
PRICE_LOOKBACK_HOURS = 24
DEFAULT_START = datetime(2019, 2, 19, tzinfo=UTC)
WTI_SEARCH_TERMS = ("WTI", "US Crude Oil", "Crude Oil", "CL")
RB_CANDIDATES = ("RBOB", "RBOB Gasoline", "Gasoline")
HO_CANDIDATES = ("Heating Oil", "Ultra Low Sulfur Diesel", "HO")

MONTH_NAME_TO_NUM = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
MONTH_REGEX = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b", re.IGNORECASE)
YYYY_MM_REGEX = re.compile(r"\b(\d{4})-(\d{2})\b")
YYYY_MON_REGEX = re.compile(r"\b(\d{4})(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b", re.IGNORECASE)


@dataclass
class MarketChoice:
    epic: str
    name: str
    expiry: Optional[datetime]
    raw: Dict[str, object]


class CapitalClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        identifier: Optional[str] = None,
        password: Optional[str] = None,
        cst: Optional[str] = None,
        x_security_token: Optional[str] = None,
        verify: bool = True,
        request_timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.identifier = identifier
        self.password = password
        self.request_timeout = request_timeout
        self.verify = verify
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        headers = {"Accept": "application/json"}
        if api_key:
            headers["X-CAP-API-KEY"] = api_key
        if cst:
            headers["CST"] = cst
        if x_security_token:
            headers["X-SECURITY-TOKEN"] = x_security_token
        self.session.headers.update(headers)
        self.can_authenticate = bool(identifier and password)
        self.tokens_provided = bool(cst and x_security_token)
        if not self.tokens_provided and self.can_authenticate:
            self._authenticate()

    def _authenticate(self) -> None:
        payload = {"identifier": self.identifier, "password": self.password, "encryptedPassword": False}
        response = self.session.post(
            f"{self.base_url}/api/v1/session",
            json=payload,
            timeout=self.request_timeout,
            verify=self.verify,
        )
        time.sleep(REQUEST_PAUSE_SECONDS)
        response.raise_for_status()
        cst_header = response.headers.get("CST")
        xst_header = response.headers.get("X-SECURITY-TOKEN")
        if not cst_header or not xst_header:
            raise RuntimeError("Missing CST or X-SECURITY-TOKEN from authentication response")
        self.session.headers["CST"] = cst_header
        self.session.headers["X-SECURITY-TOKEN"] = xst_header

    def request(self, method: str, path: str, *, params: Optional[Dict[str, object]] = None) -> requests.Response:
        url = f"{self.base_url}{path}"
        for attempt in range(5):
            response = self.session.request(
                method,
                url,
                params=params,
                timeout=self.request_timeout,
                verify=self.verify,
            )
            time.sleep(REQUEST_PAUSE_SECONDS)
            if response.status_code == 401 and self.can_authenticate:
                self._authenticate()
                continue
            if response.status_code == 429:
                wait = min(2 ** attempt, 30)
                time.sleep(wait)
                continue
            if response.status_code == 404:
                return response
            response.raise_for_status()
            return response
        raise RuntimeError(f"Failed request to {path}: exceeded retries")

    def search_markets(self, term: str) -> List[Dict[str, object]]:
        response = self.request("GET", "/api/v1/markets", params={"searchTerm": term})
        if response.status_code == 404:
            return []
        return response.json().get("markets", [])

    def fetch_prices_slice(self, epic: str, start: datetime, end: datetime) -> List[Dict[str, object]]:
        params = {
            "resolution": RESOLUTION,
            "from": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "to": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "max": MAX_POINTS_PER_SLICE,
        }
        response = self.request("GET", f"/api/v1/prices/{epic}", params=params)
        if response.status_code == 404:
            return []
        return response.json().get("prices", [])

    def fetch_prices(self, epic: str, start: datetime, end: datetime) -> List[Dict[str, object]]:
        if end <= start:
            return []
        combined: Dict[str, Dict[str, object]] = {}
        current = start
        delta = timedelta(hours=MAX_POINTS_PER_SLICE)
        while current < end:
            slice_end = min(current + delta, end)
            prices = self.fetch_prices_slice(epic, current, slice_end)
            for price in prices:
                ts = price.get("snapshotTimeUTC") or price.get("snapshotTime")
                if ts:
                    combined[ts] = price
            current = slice_end
        return [combined[key] for key in sorted(combined.keys())]


def extract_close(price: Dict[str, object]) -> Optional[float]:
    close_price = price.get("closePrice")
    if isinstance(close_price, dict):
        last_traded = close_price.get("lastTraded")
        if last_traded is not None:
            return float(last_traded)
        bid = close_price.get("bid")
        ask = close_price.get("ask")
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2.0
        if bid is not None:
            return float(bid)
        if ask is not None:
            return float(ask)
    if "close" in price and price["close"] is not None:
        return float(price["close"])
    return None


def parse_expiry(market: Dict[str, object]) -> Optional[datetime]:
    expiry_val = market.get("expiry")
    if expiry_val:
        text = str(expiry_val).replace("Z", "")
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            else:
                dt = dt.astimezone(UTC)
            return dt
        except ValueError:
            pass
        match = YYYY_MM_REGEX.search(text)
        if match:
            year, month = match.groups()
            return datetime(int(year), int(month), 1, tzinfo=UTC)
    name = str(market.get("instrumentName") or "")
    match = MONTH_REGEX.search(name)
    if match:
        month_name, year = match.groups()
        month = MONTH_NAME_TO_NUM.get(month_name.lower()[:3])
        if month:
            return datetime(int(year), month, 1, tzinfo=UTC)
    match = YYYY_MM_REGEX.search(name)
    if match:
        year, month = match.groups()
        return datetime(int(year), int(month), 1, tzinfo=UTC)
    match = YYYY_MON_REGEX.search(name)
    if match:
        year, month_name = match.groups()
        month = MONTH_NAME_TO_NUM.get(month_name.lower()[:3])
        if month:
            return datetime(int(year), month, 1, tzinfo=UTC)
    return None


def has_recent_hourly_data(client: CapitalClient, epic: str) -> bool:
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=PRICE_LOOKBACK_HOURS)
    prices = client.fetch_prices_slice(epic, start, end)
    for price in prices:
        if extract_close(price) is not None:
            return True
    return False


def discover_wti_markets(client: CapitalClient) -> List[MarketChoice]:
    seen_epics = set()
    candidates: List[MarketChoice] = []
    for term in WTI_SEARCH_TERMS:
        markets = client.search_markets(term)
        for market in markets:
            epic = str(market.get("epic") or "").strip()
            if not epic or epic in seen_epics:
                continue
            expiry = parse_expiry(market)
            if not expiry:
                continue
            seen_epics.add(epic)
            candidates.append(
                MarketChoice(
                    epic=epic,
                    name=str(market.get("instrumentName") or epic),
                    expiry=expiry,
                    raw=market,
                )
            )
    candidates.sort(key=lambda item: item.expiry or datetime.max.replace(tzinfo=UTC))
    return candidates


def select_cl_contracts(client: CapitalClient, candidates: Sequence[MarketChoice]) -> Sequence[Optional[MarketChoice]]:
    cl1: Optional[MarketChoice] = None
    cl2: Optional[MarketChoice] = None
    for candidate in candidates:
        if not has_recent_hourly_data(client, candidate.epic):
            continue
        if cl1 is None:
            cl1 = candidate
            continue
        if cl2 is None:
            cl2 = candidate
            break
    return cl1, cl2


def select_single_contract(
    client: CapitalClient,
    search_terms: Iterable[str],
    label: str,
) -> Optional[MarketChoice]:
    seen_epics = set()
    for term in search_terms:
        markets = client.search_markets(term)
        sorted_markets = sorted(
            (
                MarketChoice(
                    epic=str(m.get("epic") or "").strip(),
                    name=str(m.get("instrumentName") or m.get("epic") or ""),
                    expiry=parse_expiry(m),
                    raw=m,
                )
                for m in markets
            ),
            key=lambda item: (
                item.expiry or datetime.max.replace(tzinfo=UTC),
                item.name,
            ),
        )
        for market in sorted_markets:
            if not market.epic or market.epic in seen_epics:
                continue
            seen_epics.add(market.epic)
            if has_recent_hourly_data(client, market.epic):
                return market
    return None


def load_offline_cl1_series(start_dt: datetime, end_dt: datetime) -> pd.Series:
    files = sorted(Path("capital_wti_downloader/output").glob("OIL_CRUDE_HOUR_*.csv"))
    for path in reversed(files):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        time_col = None
        for candidate in ("snapshotTimeUTC", "snapshotTime", "ts", "time"):
            if candidate in df.columns:
                time_col = candidate
                break
        if time_col is None:
            continue
        df["ts"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"])
        price_col = None
        for candidate in ("close", "mid_close"):
            if candidate in df.columns:
                price_col = candidate
                break
        if price_col is None and {"close_bid", "close_ask"}.issubset(df.columns):
            df["mid_close"] = (pd.to_numeric(df["close_bid"], errors="coerce") + pd.to_numeric(df["close_ask"], errors="coerce")) / 2.0
            price_col = "mid_close"
        if price_col is None:
            continue
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        df = df.dropna(subset=[price_col])
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
        mask = (df["ts"] >= start_dt) & (df["ts"] < end_dt)
        df = df.loc[mask]
        if df.empty:
            continue
        return df.set_index("ts")[price_col]
    return pd.Series(dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build term crack OVX dataset from Capital.com")
    parser.add_argument("--start", type=str, help="Start date (UTC) in YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date (UTC) in YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="Bypass any cached data")
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    dt = datetime.strptime(value, "%Y-%m-%d")
    return datetime(dt.year, dt.month, dt.day, tzinfo=UTC)


def determine_range(start_str: Optional[str], end_str: Optional[str]) -> Sequence[datetime]:
    start_dt = parse_date(start_str) if start_str else DEFAULT_START
    end_dt_exclusive: datetime
    if end_str:
        end_date = parse_date(end_str)
        end_dt_exclusive = end_date + timedelta(days=1)
    else:
        now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        end_dt_exclusive = now + timedelta(hours=1)
    if end_dt_exclusive <= start_dt:
        raise ValueError("結束時間必須晚於開始時間")
    return start_dt, end_dt_exclusive


def build_hourly_index(start_dt: datetime, end_dt_exclusive: datetime) -> pd.DatetimeIndex:
    return pd.date_range(start=start_dt, end=end_dt_exclusive, freq="h", tz=UTC, inclusive="left")


def fetch_hourly_series(client: CapitalClient, epic: str, start_dt: datetime, end_dt: datetime) -> pd.Series:
    prices = client.fetch_prices(epic, start_dt, end_dt)
    if not prices:
        return pd.Series(dtype=float)
    rows: List[Dict[str, object]] = []
    for price in prices:
        ts = price.get("snapshotTimeUTC") or price.get("snapshotTime")
        close = extract_close(price)
        if not ts or close is None:
            continue
        rows.append({"ts": ts, "close": close})
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows).drop_duplicates(subset=["ts"]).sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(UTC)
    return df.set_index("ts")["close"]


def reindex_series(series: Optional[pd.Series], index: pd.DatetimeIndex) -> Optional[pd.Series]:
    if series is None:
        return None
    return series.reindex(index)


def load_existing_ovx(index: pd.DatetimeIndex, path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series([float("nan")] * len(index), index=index)
    existing = pd.read_csv(path)
    if "ovx" not in existing.columns:
        return pd.Series([float("nan")] * len(index), index=index)
    existing["ts"] = pd.to_datetime(existing["ts"], utc=True)
    existing = existing.drop_duplicates(subset=["ts"]).set_index("ts")
    series = existing["ovx"]
    return series.reindex(index)


def gather_credentials() -> Dict[str, Optional[str]]:
    env = os.environ
    return {
        "base_url": env.get("CAPITAL_BASE_URL", "https://api-capital.backend-capital.com"),
        "api_key": env.get("CAPITAL_API_KEY"),
        "identifier": env.get("CAPITAL_IDENTIFIER") or env.get("CAPITAL_USERNAME"),
        "password": env.get("CAPITAL_API_PASSWORD") or env.get("CAPITAL_PASSWORD") or env.get("CAPITAL_API_KEY_PASSWORD"),
        "cst": env.get("CAPITAL_CST") or env.get("CST"),
        "xst": env.get("CAPITAL_X_SECURITY_TOKEN") or env.get("CAPITAL_XST") or env.get("XST"),
    }


def ensure_credentials(creds: Dict[str, Optional[str]]) -> None:
    has_tokens = bool(creds["cst"] and creds["xst"])
    has_login = bool(creds["identifier"] and creds["password"])
    if not has_tokens and not has_login:
        raise RuntimeError("缺少 Capital.com 憑證，請設定 CST/XST 或帳號密碼")


def build_client() -> CapitalClient:
    creds = gather_credentials()
    ensure_credentials(creds)
    return CapitalClient(
        base_url=creds["base_url"],
        api_key=creds["api_key"],
        identifier=creds["identifier"],
        password=creds["password"],
        cst=creds["cst"],
        x_security_token=creds["xst"],
    )


def build_dataframe(
    index: pd.DatetimeIndex,
    cl1_series: pd.Series,
    cl2_series: Optional[pd.Series],
    rb_series: Optional[pd.Series],
    ho_series: Optional[pd.Series],
    ovx_series: pd.Series,
) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    df["cl1"] = cl1_series
    if cl2_series is not None:
        df["cl2"] = cl2_series
    if rb_series is not None:
        df["rb"] = rb_series
    if ho_series is not None:
        df["ho"] = ho_series

    if "cl2" in df:
        df["cl1_cl2"] = df["cl1"] - df["cl2"]
    else:
        df["cl1_cl2"] = 0.0

    if "rb" in df:
        df["crack_rb"] = df["rb"] - df["cl1"]
    else:
        df["crack_rb"] = 0.0

    if "ho" in df:
        df["crack_ho"] = df["ho"] - df["cl1"]
    else:
        df["crack_ho"] = 0.0

    df["ovx"] = ovx_series
    output = df[["cl1_cl2", "crack_rb", "crack_ho", "ovx"]].reset_index()
    output["ts"] = output["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return output[["ts", "cl1_cl2", "crack_rb", "crack_ho", "ovx"]]


def main() -> None:
    args = parse_args()
    try:
        start_dt, end_dt = determine_range(args.start, args.end)
    except ValueError as exc:
        print(f"無效日期範圍: {exc}", file=sys.stderr)
        sys.exit(1)

    client = build_client()

    wti_candidates = discover_wti_markets(client)
    cl1_choice, auto_cl2_choice = select_cl_contracts(client, wti_candidates)

    rb_choice = select_single_contract(client, RB_CANDIDATES, "RB")
    ho_choice = select_single_contract(client, HO_CANDIDATES, "HO")

    env_cl1_epic = os.environ.get("CL1_EPIC")
    env_cl2_epic = os.environ.get("CL2_EPIC")

    cl1_mode = "api"
    cl1_epic: Optional[str] = None
    cl1_name: str = ""
    cl1_offline_series: Optional[pd.Series] = None

    if cl1_choice is not None:
        cl1_epic = cl1_choice.epic
        cl1_name = cl1_choice.name or cl1_choice.epic
    else:
        if env_cl1_epic:
            if has_recent_hourly_data(client, env_cl1_epic):
                cl1_epic = env_cl1_epic
                cl1_name = env_cl1_epic
                cl1_mode = "api_env"
            else:
                print(f"INFO: CL1_EPIC={env_cl1_epic} 無法取價，嘗試後續回退")
        if cl1_epic is None and has_recent_hourly_data(client, "OIL_CRUDE"):
            cl1_epic = "OIL_CRUDE"
            cl1_name = "OIL_CRUDE"
            cl1_mode = "api_oil"
        if cl1_epic is None:
            offline_series = load_offline_cl1_series(start_dt, end_dt)
            if not offline_series.empty:
                cl1_epic = "OIL_CRUDE_OFFLINE"
                cl1_name = "Local CSV"
                cl1_mode = "offline"
                cl1_offline_series = offline_series
            else:
                cl1_epic = "CL1_ZERO"
                cl1_name = "Zero"
                cl1_mode = "zero"

    cl2_epic: Optional[str] = None
    cl2_name: str = ""
    cl2_mode = "zero"
    if env_cl2_epic:
        if has_recent_hourly_data(client, env_cl2_epic):
            cl2_epic = env_cl2_epic
            cl2_name = env_cl2_epic
            cl2_mode = "env"
        else:
            print(f"INFO: CL2_EPIC={env_cl2_epic} 無法取價，嘗試其它來源")
    if cl2_epic is None and auto_cl2_choice is not None:
        cl2_epic = auto_cl2_choice.epic
        cl2_name = auto_cl2_choice.name or auto_cl2_choice.epic
        cl2_mode = "api"
    if cl2_epic is None:
        cl2_mode = "zero"

    if cl1_mode == "api":
        print(f"CL1: {cl1_epic} - {cl1_name}")
    elif cl1_mode == "api_env":
        print(f"CL1: {cl1_epic} - {cl1_name} [FALLBACK:ENV]")
    elif cl1_mode == "api_oil":
        print(f"CL1: {cl1_epic} - {cl1_name} [FALLBACK:OIL_CRUDE]")
    elif cl1_mode == "offline":
        print(f"CL1: {cl1_epic} - {cl1_name} [OFFLINE]")
    else:
        print(f"CL1: {cl1_epic} - {cl1_name} [OFFLINE ZERO]")

    if cl2_mode == "api":
        print(f"CL2: {cl2_epic} - {cl2_name}")
    elif cl2_mode == "env":
        print(f"CL2: {cl2_epic} - {cl2_name} [FALLBACK:ENV]")
    else:
        print("INFO: CL2=0 無可取價合約，cl1_cl2 將為 0")

    if rb_choice:
        print(f"RB: {rb_choice.epic} - {rb_choice.name}")
    else:
        print("INFO: RB 無可取價合約，crack_rb 將為 0")
    if ho_choice:
        print(f"HO: {ho_choice.epic} - {ho_choice.name}")
    else:
        print("INFO: HO 無可取價合約，crack_ho 將為 0")

    index = build_hourly_index(start_dt, end_dt)

    if cl1_mode in {"api", "api_env", "api_oil"} and cl1_epic:
        cl1_raw = fetch_hourly_series(client, cl1_epic, start_dt, end_dt)
    elif cl1_mode == "offline" and cl1_offline_series is not None:
        cl1_raw = cl1_offline_series
    else:
        cl1_raw = pd.Series(0.0, index=index)
    cl1_series = reindex_series(cl1_raw, index)
    if cl1_series is None:
        cl1_series = pd.Series(0.0, index=index)

    cl2_series = None
    if cl2_mode in {"api", "env"} and cl2_epic:
        cl2_series = reindex_series(fetch_hourly_series(client, cl2_epic, start_dt, end_dt), index)

    rb_series = None
    if rb_choice:
        rb_series = reindex_series(fetch_hourly_series(client, rb_choice.epic, start_dt, end_dt), index)

    ho_series = None
    if ho_choice:
        ho_series = reindex_series(fetch_hourly_series(client, ho_choice.epic, start_dt, end_dt), index)

    ovx_series = load_existing_ovx(index, Path("data/term_crack_ovx_hourly.csv"))

    output = build_dataframe(index, cl1_series, cl2_series, rb_series, ho_series, ovx_series)
    output_path = Path("data/term_crack_ovx_hourly.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
