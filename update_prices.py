#!/usr/bin/env python3
"""
update_prices.py — Append missing daily BTC/USD closes to BitcoinPricesDaily.csv,
then re-execute SP.ipynb so model_data.pkl is refreshed.

Usage:
    python3 update_prices.py            # fetch, append, run notebook
    python3 update_prices.py --dry-run  # preview only; no file changes

Data sources (tried in order):
  1. Binance klines  — daily closes, 8 decimal places
  2. CoinGecko       — market_chart/range fallback; works where Binance is geo-blocked
"""

import csv
import datetime
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.resolve()
CSV_PATH  = REPO_ROOT / "BitcoinPricesDaily.csv"
NOTEBOOK  = REPO_ROOT / "SP.ipynb"

def _find_jupyter() -> Path:
    """Locate the jupyter executable: venv → pipx → PATH."""
    candidates = [
        REPO_ROOT / "btc_venv" / "bin" / "jupyter",
        Path.home() / ".local" / "share" / "pipx" / "venvs" / "jupyter" / "bin" / "jupyter",
        Path.home() / ".local" / "bin" / "jupyter",
    ]
    for c in candidates:
        if c.is_file():
            return c
    found = shutil.which("jupyter")
    if found:
        return Path(found)
    raise FileNotFoundError("jupyter not found — install with: pipx install jupyter")

DRY_RUN   = "--dry-run" in sys.argv


# ── Date helpers ─────────────────────────────────────────────────────────────

def date_fmt(d: datetime.date) -> str:
    """Format as M/D/YY (no leading zeros) to match CSV convention."""
    return f"{d.month}/{d.day}/{d.year % 100:02d}"


def parse_last_date(path: Path) -> datetime.date:
    """Return the date of the last row in the CSV."""
    last_row = None
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if row and row[0] != "Date":
                last_row = row
    if last_row is None:
        raise ValueError("CSV has no data rows.")
    return datetime.datetime.strptime(last_row[0].strip(), "%m/%d/%y").date()


# ── Price fetchers ───────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "quantoshi/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_binance(start: datetime.date, end: datetime.date) -> dict:
    """Fetch daily close prices from Binance klines.  Returns {date: price_str}."""
    start_ms = int(datetime.datetime(start.year, start.month, start.day,
                                     tzinfo=datetime.timezone.utc).timestamp() * 1000)
    end_ms   = int(datetime.datetime(end.year, end.month, end.day,
                                     23, 59, 59,
                                     tzinfo=datetime.timezone.utc).timestamp() * 1000)
    url  = (
        "https://api.binance.com/api/v3/klines"
        f"?symbol=BTCUSDT&interval=1d"
        f"&startTime={start_ms}&endTime={end_ms}&limit=1000"
    )
    data   = json.loads(_get(url))
    result = {}
    for candle in data:
        d = datetime.datetime.fromtimestamp(
            candle[0] / 1000, tz=datetime.timezone.utc
        ).date()
        result[d] = candle[4]          # close price as string from API
    return result


def fetch_coingecko(start: datetime.date, end: datetime.date) -> dict:
    """Fetch from CoinGecko market_chart/range.  Returns {date: price_str}.

    For ranges ≤ 90 days CoinGecko returns hourly points; we keep the last
    point per UTC calendar day (closest to the daily close).
    """
    from_ts = int(datetime.datetime(start.year, start.month, start.day,
                                    tzinfo=datetime.timezone.utc).timestamp())
    to_ts   = int(datetime.datetime(end.year, end.month, end.day,
                                    23, 59, 59,
                                    tzinfo=datetime.timezone.utc).timestamp())
    url  = (
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        f"?vs_currency=usd&from={from_ts}&to={to_ts}"
    )
    data     = json.loads(_get(url, timeout=30))
    by_date  = {}                      # date → (ts_ms, price_str)
    for ts_ms, price in data.get("prices", []):
        d = datetime.datetime.fromtimestamp(
            ts_ms / 1000, tz=datetime.timezone.utc
        ).date()
        if d not in by_date or ts_ms > by_date[d][0]:
            by_date[d] = (ts_ms, str(price))
    return {d: v[1] for d, v in by_date.items()}


def fetch_prices(start: datetime.date, end: datetime.date) -> dict:
    """Try Binance first, fall back to CoinGecko.  Returns {date: price_str}."""
    for label, fn in [("Binance", fetch_binance), ("CoinGecko", fetch_coingecko)]:
        try:
            print(f"  Trying {label}...", end=" ", flush=True)
            result = fn(start, end)
            print(f"OK  ({len(result)} daily closes returned)")
            return result
        except Exception as exc:
            print(f"FAILED  ({exc})")
    raise RuntimeError("All price sources failed — check your network connection.")


# ── Notebook runner ───────────────────────────────────────────────────────────

def run_notebook() -> None:
    print("\nRe-executing SP.ipynb …")
    jupyter = _find_jupyter()
    print(f"  Using jupyter: {jupyter}")
    cmd = [
        str(jupyter), "nbconvert",
        "--to", "notebook",
        "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=600",
        str(NOTEBOOK),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("NOTEBOOK FAILED — stderr (last 3 000 chars):")
        print(res.stderr[-3000:])
        sys.exit(1)
    print("Notebook executed successfully — model_data.pkl updated.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Bitcoin Prices Updater")
    print("=" * 60)
    print(f"CSV   : {CSV_PATH}")
    print(f"Mode  : {'DRY RUN (no changes)' if DRY_RUN else 'LIVE'}\n")

    last_date  = parse_last_date(CSV_PATH)
    # Skip the 8 most recent days — data sources can be revised for up to ~7
    # days after the fact; only include prices that have had time to settle.
    SETTLE_DAYS = 8
    fetch_end  = datetime.date.today() - datetime.timedelta(days=SETTLE_DAYS)

    print(f"Last CSV entry : {date_fmt(last_date)}  ({last_date.isoformat()})")
    print(f"Fetch through  : {date_fmt(fetch_end)}  ({fetch_end.isoformat()})"
          f"  (settling period: {SETTLE_DAYS} days)")

    fetch_start = last_date + datetime.timedelta(days=1)
    if fetch_start > fetch_end:
        print("\nCSV is already up to date (within settling window). Nothing to do.")
        return

    n_needed = (fetch_end - fetch_start).days + 1
    print(f"\nFetching {n_needed} missing day(s)…")

    prices = fetch_prices(fetch_start, fetch_end)

    # Collect rows in date order; flag any gaps
    new_rows  = []
    no_data   = []
    d = fetch_start
    while d <= fetch_end:
        if d in prices:
            new_rows.append((d, prices[d]))
        else:
            no_data.append(d)
        d += datetime.timedelta(days=1)

    if no_data:
        print(f"\nWARNING: no price returned for {len(no_data)} date(s):")
        for md in no_data:
            print(f"  {date_fmt(md)}  ({md.isoformat()})")
        print("These dates will be skipped.")

    if not new_rows:
        print("\nNo new rows to append. Done.")
        return

    # ── Preview ──────────────────────────────────────────────────────────────
    print(f"\n{'Date':<12}  {'Close (USD)':>22}")
    print("-" * 36)
    for row_d, row_p in new_rows:
        print(f"{date_fmt(row_d):<12}  {float(row_p):>22.10f}")
    print(f"\n{len(new_rows)} new row(s) ready to append.")

    if DRY_RUN:
        print("\n[--dry-run]  CSV and notebook were NOT modified.")
        return

    # ── Append to CSV ─────────────────────────────────────────────────────────
    # Ensure the file ends with a newline before appending.
    with open(CSV_PATH, "rb") as f:
        f.seek(0, 2)
        if f.tell() > 0:
            f.seek(-1, 2)
            needs_newline = f.read(1) != b"\n"
        else:
            needs_newline = False

    with open(CSV_PATH, "a") as f:
        if needs_newline:
            f.write("\n")
        for row_d, row_p in new_rows:
            # Write price preserving the original string representation where
            # possible; fall back to a 10-decimal float string.
            try:
                price_str = str(float(row_p))   # drops trailing zeros
            except ValueError:
                price_str = row_p
            f.write(f"{date_fmt(row_d)},{price_str}\n")

    print(f"\nAppended {len(new_rows)} row(s) → {CSV_PATH.name}")

    # ── Re-run notebook ───────────────────────────────────────────────────────
    run_notebook()

    # ── Done ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("UPDATE COMPLETE")
    print()
    print("Review the new rows above, then deploy when ready:")
    print()
    print("  git add BitcoinPricesDaily.csv btc_app/model_data.pkl")
    print("  git commit -m 'Update price data'")
    print("  git push origin master")
    print("  ssh root@89.167.70.45 \\")
    print("    'cd /opt/quantoshi && git pull && systemctl restart quantoshi'")
    print("=" * 60)


if __name__ == "__main__":
    main()
