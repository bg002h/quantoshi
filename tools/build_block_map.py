#!/usr/bin/env python3
"""Generate BitcoinBlocksDaily.csv from local bitcoind RPC.

Usage:
    python3 tools/build_block_map.py --full     # one-time build
    python3 tools/build_block_map.py --append   # fill gaps in existing CSV
    python3 tools/build_block_map.py --verify   # sanity check

Auth resolution: BITCOIN_RPC_URL env var > ~/.bitcoin/.cookie > ~/.bitcoin/bitcoin.conf.
Runs on dev only; prod has no bitcoind and reads the committed CSV as static data.
"""
from __future__ import annotations

import argparse
import base64
import bisect
import http.client
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

_LOG = logging.getLogger("build_block_map")

_ROOT = Path(__file__).resolve().parent.parent
_PRICE_CSV = _ROOT / "BitcoinPricesDaily.csv"
_BLOCK_CSV = _ROOT / "BitcoinBlocksDaily.csv"
_COOKIE_PATH = Path.home() / ".bitcoin" / ".cookie"
_CONF_PATH = Path.home() / ".bitcoin" / "bitcoin.conf"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8332
_BATCH_SIZE = 500

_AUTH_CACHE: dict = {}


# ──────────────────────────────────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────────────────────────────────

def _resolve_auth() -> tuple[str, str, str, int]:
    """Return (user, password, host, port). Exit 2 if none available."""
    env = os.environ.get("BITCOIN_RPC_URL")
    if env:
        m = re.match(r"^https?://([^:]+):([^@]+)@([^:/]+)(?::(\d+))?/?$", env)
        if not m:
            _LOG.error("BITCOIN_RPC_URL must be http://user:pass@host[:port]/")
            sys.exit(2)
        user, pw, host = m.group(1), m.group(2), m.group(3)
        port = int(m.group(4) or _DEFAULT_PORT)
        return user, pw, host, port

    if _COOKIE_PATH.exists():
        txt = _COOKIE_PATH.read_text().strip()
        if ":" in txt:
            user, pw = txt.split(":", 1)
            return user, pw, _DEFAULT_HOST, _DEFAULT_PORT

    if _CONF_PATH.exists():
        txt = _CONF_PATH.read_text()
        u = re.search(r"^rpcuser=(.+)$", txt, re.M)
        p = re.search(r"^rpcpassword=(.+)$", txt, re.M)
        port_m = re.search(r"^rpcport=(\d+)$", txt, re.M)
        if u and p:
            return (u.group(1).strip(), p.group(1).strip(),
                    _DEFAULT_HOST,
                    int(port_m.group(1)) if port_m else _DEFAULT_PORT)

    _LOG.error(
        "bitcoind credentials not found. Set BITCOIN_RPC_URL or ensure "
        "~/.bitcoin/.cookie (modern) or ~/.bitcoin/bitcoin.conf (legacy) exists."
    )
    sys.exit(2)


# ──────────────────────────────────────────────────────────────────────────
# JSON-RPC (stdlib http.client, batched)
# ──────────────────────────────────────────────────────────────────────────

def _get_conn():
    if "conn" not in _AUTH_CACHE:
        user, pw, host, port = _resolve_auth()
        _AUTH_CACHE["user"] = user
        _AUTH_CACHE["pw"] = pw
        _AUTH_CACHE["conn"] = http.client.HTTPConnection(host, port, timeout=120)
    return _AUTH_CACHE["conn"]


def _auth_header() -> str:
    user = _AUTH_CACHE["user"]
    pw = _AUTH_CACHE["pw"]
    creds = base64.b64encode(f"{user}:{pw}".encode()).decode()
    return f"Basic {creds}"


def _rpc(method, *params):
    conn = _get_conn()
    body = json.dumps({
        "jsonrpc": "1.0", "id": "bm", "method": method, "params": list(params),
    })
    conn.request(
        "POST", "/", body,
        {"Authorization": _auth_header(), "Content-Type": "application/json"},
    )
    resp = json.loads(conn.getresponse().read())
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(f"bitcoind error: {resp['error']}")
    return resp["result"]


def _rpc_batch(calls: list[tuple[str, list]]) -> list:
    conn = _get_conn()
    body = json.dumps([
        {"jsonrpc": "1.0", "id": i, "method": m, "params": p}
        for i, (m, p) in enumerate(calls)
    ])
    conn.request(
        "POST", "/", body,
        {"Authorization": _auth_header(), "Content-Type": "application/json"},
    )
    resp = json.loads(conn.getresponse().read())
    # bitcoind batch replies are returned by id; sort defensively
    resp_sorted = sorted(resp, key=lambda r: r["id"])
    for r in resp_sorted:
        if r.get("error"):
            raise RuntimeError(f"bitcoind batch error: {r['error']}")
    return [r["result"] for r in resp_sorted]


# ──────────────────────────────────────────────────────────────────────────
# Running-max algorithm (safe against non-monotonic block timestamps per BIP113)
# ──────────────────────────────────────────────────────────────────────────

def _compute_running_max(times: list[int]) -> list[int]:
    """Cumulative max. times[i] is block[i].time; result[i] = max(times[0..i])."""
    out = []
    cur = -1
    for t in times:
        if t > cur:
            cur = t
        out.append(cur)
    return out


def _lookup_height_before(heights: list[int], running_max: list[int],
                           target: int) -> int:
    """Return heights[i] for the largest i where running_max[i] < target.
    Uses binary search since running_max is monotonic by construction."""
    idx = bisect.bisect_left(running_max, target)
    if idx == 0:
        return heights[0] - 1  # no block fits; return "before start"
    return heights[idx - 1]


# ──────────────────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────────────────

def _load_price_dates() -> pd.DatetimeIndex:
    df = pd.read_csv(_PRICE_CSV)
    return pd.to_datetime(df["Date"], format="%m/%d/%y")


def _midnight_utc_next(date: pd.Timestamp) -> int:
    """Unix ts at 00:00 UTC the day AFTER `date`."""
    next_day = (date + pd.Timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    if next_day.tz is None:
        next_day = next_day.tz_localize("UTC")
    return int(next_day.timestamp())


def _fetch_headers_batched(h_start: int, h_end: int) -> tuple[list[int], list[int]]:
    """Fetch block times from h_start to h_end inclusive via batched RPC.
    Returns (heights, times). Walks in _BATCH_SIZE chunks."""
    heights, times = [], []
    for chunk_start in range(h_start, h_end + 1, _BATCH_SIZE):
        chunk_end = min(chunk_start + _BATCH_SIZE - 1, h_end)
        hash_calls = [("getblockhash", [h]) for h in range(chunk_start, chunk_end + 1)]
        hashes = _rpc_batch(hash_calls)
        hdr_calls = [("getblockheader", [hsh]) for hsh in hashes]
        hdrs = _rpc_batch(hdr_calls)
        for h, hdr in zip(range(chunk_start, chunk_end + 1), hdrs):
            heights.append(h)
            times.append(int(hdr["time"]))
        if chunk_end % 10000 < _BATCH_SIZE:
            _LOG.info("fetched headers up to block %d", chunk_end)
    return heights, times


def _build_block_map(dates: pd.DatetimeIndex,
                      heights: list[int],
                      times: list[int]) -> pd.DataFrame:
    """For each date in `dates`, find the blockheight at midnight_utc(date+1)
    using the running-max algorithm."""
    running_max = _compute_running_max(times)
    rows = []
    for date in dates:
        target = _midnight_utc_next(date)
        h = _lookup_height_before(heights, running_max, target)
        rows.append({"date": date.strftime("%Y-%m-%d"),
                      "blockheight": max(0, h)})
    return pd.DataFrame(rows)


def main_full():
    dates = _load_price_dates()
    tip = _rpc("getblockcount")
    # Always start from 0 for a full rebuild. Bitcoin's average is ~150 blocks/day
    # which gives h_start ≈ first price date's actual block height — no margin.
    h_start = 0
    _LOG.info("fetching headers %d..%d (%d blocks)", h_start, tip, tip - h_start + 1)
    heights, times = _fetch_headers_batched(h_start, tip)
    df = _build_block_map(dates, heights, times)
    tmp_path = _BLOCK_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(_BLOCK_CSV)
    _LOG.info("wrote %d rows to %s", len(df), _BLOCK_CSV)


def main_append():
    if not _BLOCK_CSV.exists():
        _LOG.error("no existing block CSV — run --full first")
        sys.exit(2)
    existing = pd.read_csv(_BLOCK_CSV)
    existing_dates = set(existing["date"].tolist())
    price_dates = _load_price_dates()
    price_date_strs = {d.strftime("%Y-%m-%d") for d in price_dates}
    missing = sorted(price_date_strs - existing_dates)
    if not missing:
        _LOG.info("block CSV up to date")
        return
    _LOG.info("filling %d missing rows", len(missing))
    tip = _rpc("getblockcount")
    missing_ts = pd.to_datetime(missing)
    days_span = (pd.Timestamp.today().normalize() - missing_ts.min()).days
    h_start = max(0, tip - days_span * 150 - 1000)
    heights, times = _fetch_headers_batched(h_start, tip)
    new_rows = _build_block_map(missing_ts, heights, times)
    combined = (pd.concat([existing, new_rows])
                  .sort_values("date")
                  .drop_duplicates("date", keep="last")
                  .reset_index(drop=True))
    tmp = _BLOCK_CSV.with_suffix(".csv.tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(_BLOCK_CSV)
    _LOG.info("appended %d rows", len(missing))


def main_verify():
    df = pd.read_csv(_BLOCK_CSV)
    n = len(df)
    indices = set(range(min(5, n))) | set(range(max(0, n - 5), n))
    random.seed(42)
    indices.update(random.sample(range(n), min(10, n)))
    indices = sorted(indices)

    sampled = df.iloc[indices]
    min_h = int(sampled["blockheight"].min())
    max_h = int(sampled["blockheight"].max())
    heights, times = _fetch_headers_batched(max(0, min_h - 500), max_h + 500)
    running_max = _compute_running_max(times)
    errors = []
    for i in indices:
        row = df.iloc[i]
        date = pd.Timestamp(row["date"])
        target = _midnight_utc_next(date)
        h_expected = _lookup_height_before(heights, running_max, target)
        if int(row["blockheight"]) != max(0, h_expected):
            errors.append((row["date"], int(row["blockheight"]), h_expected))
    if errors:
        for e in errors:
            _LOG.error("mismatch: date=%s csv=%d expected=%d", *e)
        sys.exit(3)
    _LOG.info("verified %d sample rows OK", len(indices))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--full", action="store_true")
    g.add_argument("--append", action="store_true")
    g.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    try:
        if args.full:
            main_full()
        elif args.append:
            main_append()
        elif args.verify:
            main_verify()
    except ConnectionError as e:
        _LOG.error(
            "bitcoind unreachable: %s. Start it with 'bitcoind -daemon' "
            "or set BITCOIN_RPC_URL.", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
