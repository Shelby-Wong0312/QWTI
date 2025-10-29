# Capital WTI Downloader

Python script that authenticates against the Capital.com DEMO API and downloads the latest 1-minute OHLC (bid/ask) candles for WTI (US Crude Oil) covering the last three months. The script reads API credentials from the existing `.env` file.

## Prerequisites

- Python 3.9+
- `pip install -r capital_wti_downloader/requirements.txt`
- `.env` file at the workspace root containing the Capital.com demo credentials (already provided by the user):
  - `CAPITAL_API_KEY`
  - `CAPITAL_API_KEY_PASSWORD` (falls back to `CAPITAL_API_PASSWORD` if present)
  - `CAPITAL_IDENTIFIER`
- *(optional)* `CAPITAL_ENV=demo` (script forces DEMO even if omitted)

The script looks up the WTI epic via `GET /api/v1/markets?searchTerm=...` (keywords: WTI -> US Crude Oil -> US Crude Oil Spot -> OIL_CRUDE -> Oil - Crude) and falls back to `OIL_CRUDE` if no match is found.
Historical 1-minute prices are pulled in 900-minute slices (`max=1000`) and stitched together in order.

Rate limits from the Capital.com REST doc: 10 req/s for general endpoints, 1 req/s for `POST /api/v1/session`, and REST sessions expire ~10 minutes after the last request.

## Usage

Run from the project root:

```bash
python capital_wti_downloader/main.py
```

Optional arguments:

- `--output PATH` - override the default CSV location (default: `capital_wti_downloader/output/{epic}_1m_last_3mo.csv`).

The script logs progress for each API slice and writes the consolidated CSV once all data is collected.
