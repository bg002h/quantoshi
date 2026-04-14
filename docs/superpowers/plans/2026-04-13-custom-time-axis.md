# Custom Time Axis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Tab 1 panel that lets the user redefine time (calendar OR blockheight, custom t₀, optional weighting) and refit a 4-model subset (PL, QR, BM-floor, Exponential) live on the server without touching other tabs or caches.

**Architecture:** New `btc_web/engines/custom_fit.py` module with pure fit functions on cached in-memory price + block arrays. New `btc_web/callbacks/custom_time.py` server callback co-owns `bubble-graph.figure` with the existing `update_bubble` via a `dcc.Store("bub-redraw-tick")` router. Block data from `BitcoinBlocksDaily.csv` generated on dev from local bitcoind (prod has no bitcoind). Custom dates capped at 2016-01-01 to guarantee fit data.

**Tech Stack:** Dash 4.0.0, scipy.stats (linregress, gaussian_kde), statsmodels (QuantReg), pandas, numpy. bitcoind JSON-RPC via stdlib `http.client`. Reuses `tools/model_toolkit/support.py::fit_support()` for BM-floor.

**Spec:** `docs/superpowers/specs/2026-04-13-custom-time-axis-design.md`

---

## File Manifest

### Create
| Path | Responsibility |
|---|---|
| `btc_web/_custom_time_presets.py` | Frozen tuples of calendar + block t₀ presets |
| `btc_web/engines/custom_fit.py` | Pure fit functions (PL / QR / BM-floor / Exp), weight computer, cached arrays, `_PriceDataShim` |
| `btc_web/layout/custom_time.py` | Collapsible panel with 8 controls |
| `btc_web/callbacks/custom_time.py` | Server callback writing figure + status + tick |
| `btc_web/test_custom_time.py` | Unit tests for fit engine, weights, presets, duplicates, exception wrapper |
| `btc_web/test_custom_time_integration.py` | Direct callback invocation tests (Case O handoff, error paths) |
| `btc_web/test_custom_time_snapshot.py` | Snapshot roundtrip tests (bitmask freeze, all-16 combos, forward-compat) |
| `btc_web/test_custom_time_e2e.py` | Playwright + Firefox E2E |
| `btc_web/test_custom_time_baseline.py` | Regression baseline as Python dict |
| `btc_web/test_block_map_cli.py` | Mocked-RPC tests for `build_block_map.py` |
| `tools/build_block_map.py` | bitcoind → CSV generator (dev only) |
| `tools/find_nonmonotonic_blocks.py` | One-off fixture discovery script |
| `BitcoinBlocksDaily.csv` | Committed date→blockheight data file |

### Modify
| Path | Change |
|---|---|
| `btc_web/callbacks/charts.py:953` (`update_bubble`) | Add `Input("bub-redraw-tick", "data")` + `State("cta-active", "value")` + top-of-body guard |
| `btc_web/layout/bubble.py:78` | Insert `custom_time_panel()` after `display_models_panel` |
| `btc_web/snapshot.py` | Register 8 new control IDs in `_SNAPSHOT_CONTROLS` + `_CHECKLIST_OPTIONS["cta-active"]` + `_CHECKLIST_OPTIONS["cta-models"]` |
| `btc_web/callbacks/routing.py` (`_TAB_CONTROLS["bubble"]`) | Add 8 new IDs to the bubble set |
| `btc_web/callbacks/__init__.py` | Import new `custom_time` callback module |
| `btc_web/layout/__init__.py` | Register `cta-body` / `bub-redraw-tick` Store at layout root |
| `btc_web/app.py` | Extend `/health` JSON response with `block_map_loaded: bool` |
| `btc-web.service` | Add `StartLimitIntervalSec=300` + `StartLimitBurst=5` |
| `update_prices.py` | Add dev-only subprocess call to `build_block_map.py --append` |

---

## Task 0: Prerequisites check

**Purpose:** Confirm bitcoind is accessible and establish the known-good blockheight corresponding to 2010-07-17 (earliest price CSV date) to avoid surprises later.

**Files:** none (read-only RPC probing)

- [ ] **Step 1: Verify bitcoind up**

```bash
bitcoin-cli getblockcount
```
Expected: integer ≥ 944000 (today's tip).

- [ ] **Step 2: Verify RPC credentials readable**

```bash
cat ~/.bitcoin/bitcoin.conf | grep -E "^rpc(user|password)"
```
Expected: `rpcuser=` and `rpcpassword=` lines present (or `.cookie` file exists).

- [ ] **Step 3: Note the block at `2010-07-17 00:00:00 UTC`** for the regression baseline later. Record as a spec comment in `BitcoinBlocksDaily.csv` once generated.

---

## Task 1: `tools/find_nonmonotonic_blocks.py` — fixture discovery

**Purpose:** Find a real non-monotonic timestamp pair in early Bitcoin history to hardcode into the running-max algorithm test fixture. This is a blocker for writing Task 9's first test.

**Files:**
- Create: `tools/find_nonmonotonic_blocks.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Find a non-monotonic block-timestamp pair in early Bitcoin history.

Output is used as a test fixture for tools/build_block_map.py's running-max
algorithm. Runs once against local bitcoind; the discovered pair (height +
both timestamps) gets hardcoded into btc_web/test_block_map_cli.py.
"""
import json
import subprocess

START, END = 30000, 80000

def rpc(method, *params):
    out = subprocess.run(
        ["bitcoin-cli", method] + [str(p) for p in params],
        capture_output=True, text=True, check=True)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return out.stdout.strip()

def main():
    prev_time = None
    prev_h = None
    for h in range(START, END + 1):
        hsh = rpc("getblockhash", h)
        hdr = rpc("getblockheader", hsh)
        t = hdr["time"]
        if prev_time is not None and t < prev_time:
            print(f"Non-monotonic pair found:")
            print(f"  block {prev_h}: time={prev_time}")
            print(f"  block {h}:     time={t}")
            print(f"  delta = {t - prev_time} seconds")
            return
        prev_time = t
        prev_h = h
    print(f"No non-monotonic pair in [{START}, {END}]")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python3 tools/find_nonmonotonic_blocks.py
```
Expected: output like
```
Non-monotonic pair found:
  block 4xxxx: time=12xxxxxxxx
  block 4xxxx: time=12xxxxxxxx
  delta = -NNNN seconds
```
Record the output.

- [ ] **Step 3: Commit the script**

```bash
git add tools/find_nonmonotonic_blocks.py
git commit -m "tools: add one-off non-monotonic block discovery script"
```

---

## Task 2: `tools/build_block_map.py` — the main CLI

**Purpose:** Generate `BitcoinBlocksDaily.csv` with one row per calendar date aligned to `BitcoinPricesDaily.csv`, using a running-max time table that correctly handles non-monotonic timestamps.

**Files:**
- Create: `tools/build_block_map.py`
- Create: `btc_web/test_block_map_cli.py`

- [ ] **Step 1: Write the failing test for `_rpc` auth resolution**

`btc_web/test_block_map_cli.py`:
```python
"""Tests for tools/build_block_map.py. All bitcoind RPC is monkeypatched."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Import the module under test (no package: tools/ is a flat dir)
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools"))
import build_block_map as bm  # noqa: E402


class _FakeConn:
    """Collects requests; returns canned responses from a method dispatcher."""
    def __init__(self, dispatch):
        self.dispatch = dispatch
        self.requests = []

    def request(self, method, path, body, headers):
        self.requests.append(json.loads(body))

    def getresponse(self):
        req = self.requests[-1]
        if isinstance(req, list):  # batch
            result = [{"result": self.dispatch(r["method"], r["params"]),
                        "error": None, "id": r["id"]} for r in req]
        else:
            result = {"result": self.dispatch(req["method"], req["params"]),
                      "error": None, "id": req["id"]}
        class R:
            def read(_): return json.dumps(result).encode()
        return R()


def test_auth_order_env_var_wins(monkeypatch, tmp_path):
    """BITCOIN_RPC_URL takes precedence over cookie and conf."""
    monkeypatch.setenv("BITCOIN_RPC_URL", "http://u:p@localhost:9999/")
    monkeypatch.setattr(bm, "_COOKIE_PATH", tmp_path / ".cookie")
    monkeypatch.setattr(bm, "_CONF_PATH", tmp_path / "bitcoin.conf")
    # Cookie file exists but should be ignored
    (tmp_path / ".cookie").write_text("cookie:should_not_use")
    result = bm._resolve_auth()
    assert result == ("u", "p", "localhost", 9999)


def test_auth_order_cookie_over_conf(monkeypatch, tmp_path):
    """Cookie file wins over bitcoin.conf when no env var."""
    monkeypatch.delenv("BITCOIN_RPC_URL", raising=False)
    cookie = tmp_path / ".cookie"
    cookie.write_text("__cookie__:abcdef123")
    conf = tmp_path / "bitcoin.conf"
    conf.write_text("rpcuser=baduser\nrpcpassword=badpass\n")
    monkeypatch.setattr(bm, "_COOKIE_PATH", cookie)
    monkeypatch.setattr(bm, "_CONF_PATH", conf)
    user, pw, host, port = bm._resolve_auth()
    assert user == "__cookie__"
    assert pw == "abcdef123"


def test_auth_order_conf_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("BITCOIN_RPC_URL", raising=False)
    conf = tmp_path / "bitcoin.conf"
    conf.write_text("rpcuser=alice\nrpcpassword=secret\nrpcport=8332\n")
    monkeypatch.setattr(bm, "_COOKIE_PATH", tmp_path / "missing.cookie")
    monkeypatch.setattr(bm, "_CONF_PATH", conf)
    user, pw, host, port = bm._resolve_auth()
    assert user == "alice"
    assert pw == "secret"


def test_auth_none_available_exits(monkeypatch, tmp_path):
    monkeypatch.delenv("BITCOIN_RPC_URL", raising=False)
    monkeypatch.setattr(bm, "_COOKIE_PATH", tmp_path / "missing.cookie")
    monkeypatch.setattr(bm, "_CONF_PATH", tmp_path / "missing.conf")
    with pytest.raises(SystemExit) as exc:
        bm._resolve_auth()
    assert exc.value.code == 2
```

- [ ] **Step 2: Run the auth tests → expect import error (module doesn't exist)**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/test_block_map_cli.py -v 2>&1 | tail
```
Expected: `ModuleNotFoundError: build_block_map`

- [ ] **Step 3: Write `tools/build_block_map.py` shell + `_resolve_auth`**

```python
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
import http.client
import json
import logging
import os
import random
import re
import sys
from datetime import datetime, timezone
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
                    _DEFAULT_HOST, int(port_m.group(1)) if port_m else _DEFAULT_PORT)

    _LOG.error(
        "bitcoind credentials not found. Set BITCOIN_RPC_URL or ensure "
        "~/.bitcoin/.cookie (modern) or ~/.bitcoin/bitcoin.conf (legacy) exists."
    )
    sys.exit(2)
```

- [ ] **Step 4: Run the auth tests → pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_block_map_cli.py -k auth -v 2>&1 | tail
```
Expected: 4 passed.

- [ ] **Step 5: Write failing tests for `_rpc_batch` against `_FakeConn`**

Append to `test_block_map_cli.py`:
```python
def test_rpc_single(monkeypatch):
    def dispatch(method, params):
        assert method == "getblockcount"
        return 944790
    fake = _FakeConn(dispatch)
    monkeypatch.setattr(bm, "_get_conn", lambda: fake)
    result = bm._rpc("getblockcount")
    assert result == 944790


def test_rpc_batch_preserves_order(monkeypatch):
    def dispatch(method, params):
        assert method == "getblockhash"
        return f"hash-{params[0]}"
    fake = _FakeConn(dispatch)
    monkeypatch.setattr(bm, "_get_conn", lambda: fake)
    result = bm._rpc_batch([("getblockhash", [i]) for i in range(5)])
    assert result == [f"hash-{i}" for i in range(5)]
```

- [ ] **Step 6: Run → expect AttributeError on `_rpc`**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_block_map_cli.py -k rpc -v 2>&1 | tail
```

- [ ] **Step 7: Implement `_rpc` + `_rpc_batch` + `_get_conn`**

Append to `tools/build_block_map.py`:
```python
_AUTH_CACHE: dict = {}


def _get_conn():
    if "conn" not in _AUTH_CACHE:
        user, pw, host, port = _resolve_auth()
        _AUTH_CACHE["user"] = user
        _AUTH_CACHE["pw"] = pw
        _AUTH_CACHE["conn"] = http.client.HTTPConnection(host, port, timeout=30)
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
    conn.request("POST", "/", body,
                 {"Authorization": _auth_header(), "Content-Type": "application/json"})
    resp = json.loads(conn.getresponse().read())
    if resp.get("error"):
        raise RuntimeError(f"bitcoind error: {resp['error']}")
    return resp["result"]


def _rpc_batch(calls: list[tuple[str, list]]) -> list:
    conn = _get_conn()
    body = json.dumps([
        {"jsonrpc": "1.0", "id": i, "method": m, "params": p}
        for i, (m, p) in enumerate(calls)
    ])
    conn.request("POST", "/", body,
                 {"Authorization": _auth_header(), "Content-Type": "application/json"})
    resp = json.loads(conn.getresponse().read())
    # bitcoind batch replies are returned in request order; assert anyway
    resp_sorted = sorted(resp, key=lambda r: r["id"])
    for r in resp_sorted:
        if r.get("error"):
            raise RuntimeError(f"bitcoind batch error: {r['error']}")
    return [r["result"] for r in resp_sorted]
```

- [ ] **Step 8: Run → pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_block_map_cli.py -k rpc -v 2>&1 | tail
```
Expected: 2 passed.

- [ ] **Step 9: Write failing test for `_running_max_lookup` (the non-monotonic core)**

Append:
```python
def test_running_max_lookup_handles_nonmonotonic():
    """The running-max algorithm finds the highest h where the chain tip
    time (accumulated max) was less than target. Block timestamps can go
    backward by up to ~2 hours per BIP113; running-max linearizes this."""
    # Use the real early-Bitcoin non-monotonic pair from tools/find_nonmonotonic_blocks.py.
    # REPLACE the placeholders below with the actual values discovered in Task 1.
    heights = [40000, 40001, 40002, 40003, 40004]
    times   = [1266000000, 1266001000, 1265999500,  # block 40002 is earlier than 40001
               1266002000, 1266003000]
    running_max = bm._compute_running_max(times)
    assert running_max == [1266000000, 1266001000, 1266001000, 1266002000, 1266003000]

    target = 1266001500  # between 40001 and 40002 by wall-clock
    h = bm._lookup_height_before(heights, running_max, target)
    assert h == 40001  # NOT 40002 — running max is still 1266001000 at 40002
```

(During execution, REPLACE the placeholder `heights`/`times` with the actual discovery from Task 1 before running the test. Document the replacement in the commit message.)

- [ ] **Step 10: Run → fail (functions don't exist)**

- [ ] **Step 11: Implement `_compute_running_max` + `_lookup_height_before`**

```python
def _compute_running_max(times: list[int]) -> list[int]:
    """Cumulative max. times[i] is block[i].time; result[i] = max(times[0..i])."""
    out = []
    cur = -1
    for t in times:
        if t > cur:
            cur = t
        out.append(cur)
    return out


def _lookup_height_before(heights: list[int], running_max: list[int], target: int) -> int:
    """Return heights[i] for the largest i where running_max[i] < target.
    Uses binary search since running_max is monotonic by construction."""
    import bisect
    # bisect_left finds the first index where running_max[i] >= target
    idx = bisect.bisect_left(running_max, target)
    if idx == 0:
        return heights[0] - 1  # no block fits; return "before start"
    return heights[idx - 1]
```

- [ ] **Step 12: Run → pass**

- [ ] **Step 13: Commit progress**

```bash
git add tools/build_block_map.py btc_web/test_block_map_cli.py
git commit -m "feat(block_map): auth resolution + running-max core"
```

- [ ] **Step 14: Write failing tests for the `--full` build flow**

Append:
```python
def test_full_build_writes_atomic_csv(monkeypatch, tmp_path):
    """--full writes a temp file and atomically renames to the target."""
    # Synthetic bitcoind: block h has time = 1266000000 + h*600 (10min intervals)
    heights = list(range(0, 200))
    hashes = [f"hash{h:06d}" for h in heights]
    times = [1266000000 + h * 600 for h in heights]

    def dispatch(method, params):
        if method == "getblockcount":
            return max(heights)
        if method == "getblockhash":
            return hashes[params[0]]
        if method == "getblockheader":
            h = hashes.index(params[0])
            return {"height": h, "time": times[h]}
        raise ValueError(method)

    monkeypatch.setattr(bm, "_get_conn", lambda: _FakeConn(dispatch))
    _AUTH_CACHE_STUB = {"user": "u", "pw": "p", "conn": _FakeConn(dispatch)}
    monkeypatch.setattr(bm, "_AUTH_CACHE", _AUTH_CACHE_STUB)

    # Fake price CSV with 3 rows
    price_csv = tmp_path / "BitcoinPricesDaily.csv"
    price_csv.write_text("Date,Price\n2/12/10,100\n2/13/10,101\n2/14/10,102\n")
    block_csv = tmp_path / "BitcoinBlocksDaily.csv"
    monkeypatch.setattr(bm, "_PRICE_CSV", price_csv)
    monkeypatch.setattr(bm, "_BLOCK_CSV", block_csv)

    bm.main_full()

    assert block_csv.exists()
    df = pd.read_csv(block_csv)
    assert list(df.columns) == ["date", "blockheight"]
    assert len(df) == 3
    # Atomic temp file should NOT linger
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())
```

- [ ] **Step 15: Run → fail**

- [ ] **Step 16: Implement `main_full` + `_load_price_dates` + `_fetch_headers_batched` + `_build_block_map`**

```python
def _load_price_dates() -> pd.DatetimeIndex:
    """Read the price CSV and return parsed dates (M/D/YY format)."""
    df = pd.read_csv(_PRICE_CSV)
    return pd.to_datetime(df["Date"], format="%m/%d/%y")


def _midnight_utc_next(date: pd.Timestamp) -> int:
    """Unix ts at 00:00 UTC the day AFTER `date`."""
    next_day = (date + pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                                       microsecond=0)
    return int(next_day.tz_localize("UTC").timestamp())


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
        rows.append({"date": date.strftime("%Y-%m-%d"), "blockheight": max(0, h)})
    return pd.DataFrame(rows)


def main_full():
    dates = _load_price_dates()
    tip = _rpc("getblockcount")
    # Estimate h_start: overshoot by ~150 blocks/day
    days_since_first = (pd.Timestamp.today().normalize() - dates[0]).days
    h_start = max(0, tip - days_since_first * 150 - 1000)
    _LOG.info("fetching headers %d..%d", h_start, tip)
    heights, times = _fetch_headers_batched(h_start, tip)
    df = _build_block_map(dates, heights, times)
    tmp_path = _BLOCK_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(_BLOCK_CSV)
    _LOG.info("wrote %d rows to %s", len(df), _BLOCK_CSV)
```

- [ ] **Step 17: Run → pass**

- [ ] **Step 18: Write failing tests for `--append` (gap fill) + `--verify` (mismatch detection) + alignment**

Append:
```python
def test_append_fills_middle_gap(monkeypatch, tmp_path):
    """--append finds missing date rows and fills them."""
    price_csv = tmp_path / "BitcoinPricesDaily.csv"
    price_csv.write_text(
        "Date,Price\n2/12/10,100\n2/13/10,101\n2/14/10,102\n2/15/10,103\n"
    )
    block_csv = tmp_path / "BitcoinBlocksDaily.csv"
    # Pre-seed with missing middle row 2010-02-13
    block_csv.write_text("date,blockheight\n2010-02-12,40000\n2010-02-14,40288\n")

    def dispatch(method, params):
        if method == "getblockcount": return 50000
        if method == "getblockhash":  return f"hash{params[0]:06d}"
        if method == "getblockheader":
            h = int(params[0].replace("hash", ""))
            return {"height": h, "time": 1266000000 + h * 600}
        raise ValueError(method)

    monkeypatch.setattr(bm, "_get_conn", lambda: _FakeConn(dispatch))
    monkeypatch.setattr(bm, "_AUTH_CACHE",
                         {"user":"u","pw":"p","conn":_FakeConn(dispatch)})
    monkeypatch.setattr(bm, "_PRICE_CSV", price_csv)
    monkeypatch.setattr(bm, "_BLOCK_CSV", block_csv)

    bm.main_append()

    df = pd.read_csv(block_csv)
    assert len(df) == 4
    assert set(df["date"]) == {"2010-02-12", "2010-02-13",
                                "2010-02-14", "2010-02-15"}


def test_verify_catches_corruption(monkeypatch, tmp_path):
    price_csv = tmp_path / "BitcoinPricesDaily.csv"
    price_csv.write_text("Date,Price\n2/12/10,100\n")
    block_csv = tmp_path / "BitcoinBlocksDaily.csv"
    block_csv.write_text("date,blockheight\n2010-02-12,999999\n")  # wrong

    def dispatch(method, params):
        if method == "getblockcount": return 50000
        if method == "getblockhash":  return f"hash{params[0]:06d}"
        if method == "getblockheader":
            h = int(params[0].replace("hash", ""))
            return {"height": h, "time": 1266000000 + h * 600}
        raise ValueError(method)

    monkeypatch.setattr(bm, "_get_conn", lambda: _FakeConn(dispatch))
    monkeypatch.setattr(bm, "_AUTH_CACHE",
                         {"user":"u","pw":"p","conn":_FakeConn(dispatch)})
    monkeypatch.setattr(bm, "_PRICE_CSV", price_csv)
    monkeypatch.setattr(bm, "_BLOCK_CSV", block_csv)

    with pytest.raises(SystemExit) as exc:
        bm.main_verify()
    assert exc.value.code == 3
```

- [ ] **Step 19: Run → fail**

- [ ] **Step 20: Implement `main_append` + `main_verify`**

```python
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
    # Fetch a window covering all missing dates
    missing_ts = pd.to_datetime(missing)
    h_start = max(0, tip - (pd.Timestamp.today().normalize() -
                              missing_ts.min()).days * 150 - 1000)
    heights, times = _fetch_headers_batched(h_start, tip)
    new_rows = _build_block_map(missing_ts, heights, times)
    combined = pd.concat([existing, new_rows]).sort_values("date").drop_duplicates(
        "date", keep="last").reset_index(drop=True)
    tmp = _BLOCK_CSV.with_suffix(".csv.tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(_BLOCK_CSV)
    _LOG.info("appended %d rows", len(missing))


def main_verify():
    df = pd.read_csv(_BLOCK_CSV)
    # Sample first 5 + last 5 + 10 random
    n = len(df)
    indices = set(range(min(5, n))) | set(range(max(0, n-5), n))
    random.seed(42)
    indices.update(random.sample(range(n), min(10, n)))
    tip = _rpc("getblockcount")
    heights, times = _fetch_headers_batched(
        max(0, min(df.iloc[list(indices)]["blockheight"]) - 500),
        max(df.iloc[list(indices)]["blockheight"]) + 500,
    )
    running_max = _compute_running_max(times)
    errors = []
    for i in sorted(indices):
        row = df.iloc[i]
        date = pd.Timestamp(row["date"])
        target = _midnight_utc_next(date)
        h_expected = _lookup_height_before(heights, running_max, target)
        if int(row["blockheight"]) != max(0, h_expected):
            errors.append((row["date"], row["blockheight"], h_expected))
    if errors:
        for e in errors:
            _LOG.error("mismatch: date=%s csv=%d expected=%d", *e)
        sys.exit(3)
    _LOG.info("verified %d sample rows OK", len(indices))
```

- [ ] **Step 21: Run all tests → pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_block_map_cli.py -v 2>&1 | tail
```

- [ ] **Step 22: Add the main() dispatcher + logging setup + CLI**

Append:
```python
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser()
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
        _LOG.error("bitcoind unreachable: %s. Start it with 'bitcoind -daemon' "
                    "or set BITCOIN_RPC_URL.", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 23: Commit**

```bash
git add tools/build_block_map.py btc_web/test_block_map_cli.py
git commit -m "feat(block_map): full/append/verify with running-max algorithm"
```

---

## Task 3: Generate the real `BitcoinBlocksDaily.csv`

**Purpose:** Run `build_block_map.py --full` against local bitcoind. Commit the output. This is the single "real bitcoind" step.

**Files:**
- Create: `BitcoinBlocksDaily.csv`

- [ ] **Step 1: Run the full build**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 tools/build_block_map.py --full 2>&1 | tail -20
```
Expected: ~5-10 minutes, final log line `wrote NNNN rows to BitcoinBlocksDaily.csv`. The `~800k` headers are batched at 500/req → ~1600 HTTP round trips.

- [ ] **Step 2: Sanity check**

```bash
head -3 BitcoinBlocksDaily.csv
tail -3 BitcoinBlocksDaily.csv
wc -l BitcoinBlocksDaily.csv BitcoinPricesDaily.csv
```
Expected: same row count (header + ~5743 data rows in both).

- [ ] **Step 3: Run verify**

```bash
btc_venv/bin/python3 tools/build_block_map.py --verify 2>&1 | tail
```
Expected: `verified 20 sample rows OK`.

- [ ] **Step 4: Commit the CSV**

```bash
git add BitcoinBlocksDaily.csv
git commit -m "data: add BitcoinBlocksDaily.csv (generated from local bitcoind)"
```

---

## Task 4: `btc_web/_custom_time_presets.py` — frozen constants

**Files:**
- Create: `btc_web/_custom_time_presets.py`

- [ ] **Step 1: Write the module**

```python
"""Frozen t₀ preset tables for the Custom Time Axis panel.

Load-bearing properties (enforced by btc_web/test_custom_time.py):
- Calendar presets MUST be < 2016-01-01 (fitting-data floor).
- Block presets MUST be < _BLOCK_CAP (computed at custom_fit import time).
- len(CAL_PRESETS) == 6, len(BLK_PRESETS) == 5. Whitepaper has no block equivalent.
"""
from __future__ import annotations

from datetime import date

# (key, date, display label)
CAL_PRESETS: tuple[tuple[str, date, str], ...] = (
    ("whitepaper",  date(2008, 10, 31), "Bitcoin whitepaper (2008-10-31)"),
    ("genesis",     date(2009,  1,  3), "Genesis block (2009-01-03)"),
    ("optimal",     date(2009,  7, 25), "Current optimal (2009-07-25)"),
    ("nls",         date(2009, 10,  5), "New Liberty Standard (2009-10-05)"),
    ("pizza",       date(2010,  5, 22), "Bitcoin Pizza Day (2010-05-22)"),
    ("mtgox",       date(2010,  7, 17), "Mt. Gox launch (2010-07-17)"),
)

# (key, blockheight, display label)
BLK_PRESETS: tuple[tuple[str, int, str], ...] = (
    ("block_0",     0,       "Block 0 (genesis)"),
    ("block_3300",  3300,    "Block 3300 (≈ 2009-07-25)"),
    ("block_32000", 32000,   "Block 32000 (≈ first dollar trade)"),
    ("block_67700", 67700,   "Block 67700 (≈ Pizza Day)"),
    ("block_70000", 70000,   "Block 70000 (≈ Mt. Gox)"),
)

CAL_PRESET_BY_KEY = {k: (d, lbl) for k, d, lbl in CAL_PRESETS}
BLK_PRESET_BY_KEY = {k: (h, lbl) for k, h, lbl in BLK_PRESETS}
```

- [ ] **Step 2: Commit**

```bash
git add btc_web/_custom_time_presets.py
git commit -m "feat(custom_time): add preset constants module"
```

---

## Task 5: `btc_web/engines/custom_fit.py` — fit engine (TDD)

**Purpose:** Pure fit functions per section 3 of the spec. No Dash dependencies. Cached arrays at module import.

**Files:**
- Create: `btc_web/engines/custom_fit.py`
- Create: `btc_web/test_custom_time.py` (unit tests)

### Task 5a: Data classes + `_compute_weights`

- [ ] **Step 1: Write failing tests**

`btc_web/test_custom_time.py`:
```python
"""Unit tests for btc_web/engines/custom_fit.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "btc_web"))
sys.path.insert(0, str(_ROOT))

from engines import custom_fit as cf  # noqa: E402


def test_compute_weights_none_returns_ones():
    t = np.linspace(1.0, 10.0, 100)
    w, degraded = cf._compute_weights(t, "none")
    assert np.allclose(w, 1.0)
    assert degraded is False


def test_compute_weights_inv_t_monotone_and_mean_one():
    t = np.linspace(1.0, 10.0, 100)
    w, degraded = cf._compute_weights(t, "inv_t")
    assert w[0] > w[-1]  # early > late
    assert abs(w.mean() - 1.0) < 1e-9
    assert degraded is False


def test_compute_weights_inv_sqrt_t_mean_one():
    t = np.linspace(1.0, 10.0, 100)
    w, _ = cf._compute_weights(t, "inv_sqrt_t")
    assert abs(w.mean() - 1.0) < 1e-9


def test_compute_weights_log_density_mean_one():
    rng = np.random.default_rng(0)
    t = rng.uniform(1.0, 100.0, 500)  # uniform in linear t → dense in log t recent half
    w, degraded = cf._compute_weights(t, "log_density")
    assert abs(w.mean() - 1.0) < 1e-6
    assert degraded is False


def test_compute_weights_small_n_falls_back_to_uniform():
    t = np.linspace(1.0, 5.0, 20)  # n=20 < 30
    w, degraded = cf._compute_weights(t, "log_density")
    assert np.allclose(w, 1.0)
    assert degraded is True


def test_compute_weights_unknown_mode_falls_back():
    t = np.linspace(1.0, 10.0, 100)
    w, degraded = cf._compute_weights(t, "nonsense_mode")
    assert np.allclose(w, 1.0)
    # unknown mode reported as degraded with a note at the FitResult level; the
    # weight-computer just returns uniform + degraded=False (caller notes it).
```

- [ ] **Step 2: Run → ImportError**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/test_custom_time.py -v 2>&1 | tail
```

- [ ] **Step 3: Write `btc_web/engines/custom_fit.py` skeleton**

```python
"""Custom Time Axis — fit engine.

Pure functions on cached numpy arrays. No Dash dependencies.
See docs/superpowers/specs/2026-04-13-custom-time-axis-design.md section 3.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.stats

_LOG = logging.getLogger("custom_fit")


@dataclass(frozen=True)
class FitInput:
    t: np.ndarray            # user-chosen scale (years or raw block-units)
    price: np.ndarray        # aligned with t
    weighting: str           # "none" | "inv_t" | "inv_sqrt_t" | "log_density"


@dataclass
class FitResult:
    name: str
    params: dict
    t_plot: np.ndarray
    y_plot: Union[np.ndarray, dict]     # array for PL/Exp/BM-floor, dict for QR
    n_samples: int
    r2: float
    elapsed_ms: float
    note: Optional[str] = None


def _compute_weights(t_positive: np.ndarray, mode: str) -> tuple[np.ndarray, bool]:
    """Return (weights, degraded_flag). Weights normalized to mean=1.0.
    `degraded=True` when KDE fell back to uniform due to small-n."""
    n = len(t_positive)
    if mode == "none":
        return np.ones(n), False
    if n < 30:
        return np.ones(n), True
    try:
        if mode == "inv_t":
            w = 1.0 / t_positive
        elif mode == "inv_sqrt_t":
            w = 1.0 / np.sqrt(t_positive)
        elif mode == "log_density":
            log_t = np.log10(t_positive)
            kde = scipy.stats.gaussian_kde(log_t)  # Scott's rule default
            dens = kde(log_t)
            w = 1.0 / np.maximum(dens, 1e-9)
        else:
            return np.ones(n), False
    except Exception as exc:
        _LOG.warning("custom_fit weight compute failed: %s", exc)
        return np.ones(n), True
    # Normalize to mean=1
    w = w * (n / w.sum())
    return w, False
```

- [ ] **Step 4: Run → pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_custom_time.py -v 2>&1 | tail
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/custom_fit.py btc_web/test_custom_time.py
git commit -m "feat(custom_fit): data classes + weight computer"
```

### Task 5b: `fit_pl`, `fit_exp`, `fit_qr`

- [ ] **Step 1: Write failing tests**

Append to `test_custom_time.py`:
```python
def _synth_pl(slope=5.0, intercept=-1.5, n=1000, seed=0):
    """log10(price) = slope * log10(t) + intercept + noise"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.5, 16.0, n)
    log_p = slope * np.log10(t) + intercept + rng.normal(0, 0.05, n)
    return t, 10 ** log_p


def test_fit_pl_recovers_slope():
    t, p = _synth_pl(slope=5.73, intercept=-1.20)
    fi = cf.FitInput(t=t, price=p, weighting="none")
    r = cf.fit_pl(fi)
    assert r is not None
    assert abs(r.params["slope"] - 5.73) < 0.05
    assert abs(r.params["intercept"] - (-1.20)) < 0.05
    assert r.r2 > 0.99
    assert r.n_samples < 1000  # t<=0 mask dropped ~0 here since start=0.5


def test_fit_pl_returns_none_when_insufficient_samples():
    fi = cf.FitInput(t=np.array([1.0, 2.0]), price=np.array([10.0, 20.0]),
                      weighting="none")
    assert cf.fit_pl(fi) is None


def test_fit_exp_recovers_and_keeps_negative_t():
    rng = np.random.default_rng(0)
    n = 500
    t = np.linspace(-5.0, 10.0, n)  # negative t allowed for Exp
    log_p = 0.35 * t + 2.0 + rng.normal(0, 0.05, n)
    fi = cf.FitInput(t=t, price=10 ** log_p, weighting="none")
    r = cf.fit_exp(fi)
    assert abs(r.params["slope"] - 0.35) < 0.02
    assert abs(r.params["intercept"] - 2.0) < 0.1
    assert r.n_samples == n  # no mask


def test_fit_qr_recovers_median_slope():
    t, p = _synth_pl(slope=5.5, n=500)
    fi = cf.FitInput(t=t, price=p, weighting="none")
    r = cf.fit_qr(fi)
    assert 0.50 in r.y_plot  # dict keyed by quantile
    # Median slope should be close to PL slope
    assert "slopes" in r.params
    assert abs(r.params["slopes"][0.50] - 5.5) < 0.1


def test_fit_qr_reduced_quantiles_when_10_le_n_lt_30():
    rng = np.random.default_rng(0)
    n = 20
    t = np.linspace(1.0, 15.0, n)
    p = 10 ** (3.0 * np.log10(t) + rng.normal(0, 0.05, n))
    fi = cf.FitInput(t=t, price=p, weighting="none")
    r = cf.fit_qr(fi)
    assert r is not None
    assert set(r.y_plot.keys()) == {0.25, 0.50, 0.75}


def test_fit_qr_returns_none_when_n_below_10():
    t = np.linspace(1.0, 5.0, 5)
    p = 10 ** (3.0 * np.log10(t))
    fi = cf.FitInput(t=t, price=p, weighting="none")
    assert cf.fit_qr(fi) is None
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement `fit_pl`, `fit_exp`, `fit_qr`**

Append to `custom_fit.py`:
```python
_T_PLOT_POINTS = 400

_QR_FULL = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
_QR_REDUCED = (0.25, 0.50, 0.75)


def fit_pl(fi: FitInput) -> Optional[FitResult]:
    t0 = time.perf_counter()
    mask = fi.t > 0
    t = fi.t[mask]
    price = fi.price[mask]
    n = len(t)
    if n < 3:
        return None
    log_t = np.log10(t)
    log_p = np.log10(price)
    weights, degraded = _compute_weights(t, fi.weighting)
    if fi.weighting == "none":
        res = scipy.stats.linregress(log_t, log_p)
        slope, intercept = res.slope, res.intercept
        r2 = res.rvalue ** 2
    else:
        # np.polyfit uses w as sqrt-of-weights per the loss Σ w[j]² · r[j]²
        slope, intercept = np.polyfit(log_t, log_p, 1, w=np.sqrt(weights))
        pred = slope * log_t + intercept
        wbar = (weights * log_p).sum() / weights.sum()
        ss_res = (weights * (log_p - pred) ** 2).sum()
        ss_tot = (weights * (log_p - wbar) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    t_plot = np.logspace(np.log10(max(t.min(), 1e-6)),
                          np.log10(t.max() * 1.1), _T_PLOT_POINTS)
    y_plot = slope * np.log10(t_plot) + intercept
    note = "weighting degraded (n<30)" if degraded else None
    return FitResult(
        name="PL", params={"slope": slope, "intercept": intercept},
        t_plot=t_plot, y_plot=y_plot, n_samples=n, r2=r2,
        elapsed_ms=(time.perf_counter() - t0) * 1000, note=note)


def fit_exp(fi: FitInput) -> Optional[FitResult]:
    t0 = time.perf_counter()
    n = len(fi.t)
    if n < 3:
        return None
    log_p = np.log10(fi.price)
    res = scipy.stats.linregress(fi.t, log_p)
    t_plot = np.linspace(fi.t.min(), fi.t.max() * 1.1, _T_PLOT_POINTS)
    y_plot = res.slope * t_plot + res.intercept
    return FitResult(
        name="Exp", params={"slope": res.slope, "intercept": res.intercept},
        t_plot=t_plot, y_plot=y_plot, n_samples=n, r2=res.rvalue ** 2,
        elapsed_ms=(time.perf_counter() - t0) * 1000)


def fit_qr(fi: FitInput) -> Optional[FitResult]:
    import statsmodels.api as sm
    t0 = time.perf_counter()
    mask = fi.t > 0
    t = fi.t[mask]
    price = fi.price[mask]
    n = len(t)
    if n < 10:
        return None
    quantiles = _QR_FULL if n >= 30 else _QR_REDUCED

    log_t = np.log10(t)
    log_p = np.log10(price)

    # Weighted resampling (statsmodels.QuantReg does not accept sample weights)
    if fi.weighting != "none":
        weights, degraded = _compute_weights(t, fi.weighting)
        rng = np.random.default_rng(0)
        probs = weights / weights.sum()
        idx = rng.choice(n, size=5 * n, replace=True, p=probs)
        log_t_fit = log_t[idx]
        log_p_fit = log_p[idx]
        resample_note = "QR weighted via resampling (approximate)"
    else:
        log_t_fit = log_t
        log_p_fit = log_p
        resample_note = None
        degraded = False

    X = sm.add_constant(log_t_fit)
    y_plot: dict = {}
    slopes: dict = {}
    intercepts: dict = {}
    try:
        for q in quantiles:
            mdl = sm.QuantReg(log_p_fit, X).fit(q=q)
            intercept, slope = mdl.params
            slopes[q] = slope
            intercepts[q] = intercept
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return FitResult(
            name="QR", params={}, t_plot=np.array([]), y_plot={},
            n_samples=n, r2=float("nan"), elapsed_ms=elapsed,
            note=f"Fit failed: {type(exc).__name__}")

    t_plot = np.logspace(np.log10(max(t.min(), 1e-6)),
                          np.log10(t.max() * 1.1), _T_PLOT_POINTS)
    log_t_plot = np.log10(t_plot)
    for q in quantiles:
        y_plot[q] = slopes[q] * log_t_plot + intercepts[q]

    note_parts = [resample_note, "weighting degraded (n<30)" if degraded else None,
                   f"reduced quantiles (n<30)" if n < 30 else None]
    note = " | ".join(p for p in note_parts if p) or None

    return FitResult(
        name="QR", params={"slopes": slopes, "intercepts": intercepts},
        t_plot=t_plot, y_plot=y_plot, n_samples=n, r2=float("nan"),
        elapsed_ms=(time.perf_counter() - t0) * 1000, note=note)
```

- [ ] **Step 4: Run → pass**

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/custom_fit.py btc_web/test_custom_time.py
git commit -m "feat(custom_fit): add PL, Exp, QR fit functions"
```

### Task 5c: `fit_bm_floor` via `_PriceDataShim`

- [ ] **Step 1: Write failing test**

Append:
```python
def test_fit_bm_floor_matches_fit_support_on_slice():
    """The shim route should produce the same slope/intercept as calling
    fit_support directly on a PriceData built from the same slice."""
    # Synthetic series with a clear support line
    t = np.linspace(0.5, 15.0, 500)
    rng = np.random.default_rng(0)
    log_p = 5.5 * np.log10(t) + 0.5 + rng.uniform(-0.1, 0.6, 500)  # skewed upward
    p = 10 ** log_p
    fi = cf.FitInput(t=t, price=p, weighting="none")
    r = cf.fit_bm_floor(fi)
    assert r is not None
    assert "slope" in r.params
    assert "intercept" in r.params
    assert r.r2 is not None
    assert r.n_samples > 100


def test_fit_bm_floor_skips_when_n_lt_50():
    t = np.linspace(1.0, 10.0, 30)
    p = 10 ** (3.0 * np.log10(t))
    fi = cf.FitInput(t=t, price=p, weighting="none")
    assert cf.fit_bm_floor(fi) is None
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement `_PriceDataShim` + `fit_bm_floor`**

Append:
```python
class _PriceDataShim:
    """Duck-types the attributes `tools/model_toolkit/support.py::fit_support`
    reads from a real PriceData. In block mode the column is 'log_years' but
    holds log10(block_offset) — misleading name preserved to match upstream."""
    def __init__(self, t_positive: np.ndarray, price_positive: np.ndarray):
        mask = (t_positive > (1.0 / 365.25)) & (price_positive > 0)
        t = t_positive[mask]
        p = price_positive[mask]
        self.log_years = np.log10(t)
        self.log_prices = np.log10(p)
        self.df_full = pd.DataFrame({
            "log_years": self.log_years,
            "log_price": self.log_prices,
        })


def fit_bm_floor(fi: FitInput) -> Optional[FitResult]:
    # Imports here to avoid hard dependency if toolkit path differs in tests
    import sys
    from pathlib import Path
    _toolkit = Path(__file__).resolve().parent.parent.parent / "tools"
    if str(_toolkit) not in sys.path:
        sys.path.insert(0, str(_toolkit))
    from model_toolkit.support import fit_support  # type: ignore

    t0 = time.perf_counter()
    mask = fi.t > 0
    t = fi.t[mask]
    price = fi.price[mask]
    n = len(t)
    if n < 50:
        return None

    shim = _PriceDataShim(t, price)
    try:
        fit = fit_support(shim, percentile=0.20)
    except Exception as exc:
        return FitResult(
            name="BM floor", params={}, t_plot=np.array([]), y_plot=np.array([]),
            n_samples=n, r2=float("nan"),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            note=f"Fit failed: {type(exc).__name__}")

    slope = float(fit["slope"])
    intercept = float(fit["intercept"])
    t_plot = np.logspace(np.log10(max(t.min(), 1e-6)),
                          np.log10(t.max() * 1.1), _T_PLOT_POINTS)
    y_plot = slope * np.log10(t_plot) + intercept
    return FitResult(
        name="BM floor", params={"slope": slope, "intercept": intercept},
        t_plot=t_plot, y_plot=y_plot, n_samples=n,
        r2=float(fit.get("r2", float("nan"))),
        elapsed_ms=(time.perf_counter() - t0) * 1000)
```

- [ ] **Step 4: Verify `fit_support`'s actual return shape**

```bash
btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'tools')
from model_toolkit.support import fit_support
import inspect
print(inspect.getsource(fit_support))" 2>&1 | head -60
```
If the returned dict has different keys than `slope`/`intercept`/`r2`, adjust `fit_bm_floor` accordingly.

- [ ] **Step 5: Run test → pass**

- [ ] **Step 6: Commit**

```bash
git add btc_web/engines/custom_fit.py btc_web/test_custom_time.py
git commit -m "feat(custom_fit): add BM-floor via PriceData shim"
```

### Task 5d: `build_fit_input` + cached arrays

- [ ] **Step 1: Write failing test**

Append:
```python
def test_build_fit_input_calendar():
    """Calendar mode: t = (dates - t0).days / 365.25"""
    import _app_ctx as ctx
    # Use real M.price_years / price_prices via the cached loader
    fi = cf.build_fit_input(scale="calendar", t0="2015-01-01", weighting="none")
    assert fi.t[0] < 0  # 2010-07-17 vs 2015-01-01 is negative
    assert fi.t[-1] > 10  # latest row should be years past 2015
    assert fi.weighting == "none"


def test_build_fit_input_block_mode(monkeypatch):
    """Block mode: t = blocks - t0_block"""
    # Monkeypatch the cached block array for isolation
    fake_blocks = np.arange(0, 5000, dtype=np.int64) * 2  # 0, 2, 4, ..., 9998
    monkeypatch.setattr(cf, "_BLOCKS", fake_blocks)
    fake_prices = np.linspace(1.0, 1000.0, len(fake_blocks))
    monkeypatch.setattr(cf, "_PRICES", fake_prices)
    monkeypatch.setattr(cf, "_DATES",
                         pd.date_range("2010-07-17", periods=len(fake_blocks)))
    fi = cf.build_fit_input(scale="block", t0=1000, weighting="inv_t")
    assert fi.t[0] == -1000  # first block (0) minus t0=1000
    assert fi.weighting == "inv_t"
```

- [ ] **Step 2: Run → fail (functions don't exist)**

- [ ] **Step 3: Implement cached-array loaders + `build_fit_input`**

Append to `custom_fit.py`:
```python
# ──────────────────────────────────────────────────────────────────────────
# Cached price + block arrays (loaded once per worker at import time)
# ──────────────────────────────────────────────────────────────────────────

_DATES: Optional[pd.DatetimeIndex] = None
_PRICES: Optional[np.ndarray] = None
_BLOCKS: Optional[np.ndarray] = None
_BLOCK_CAP: Optional[int] = None
_BLOCK_MAP_LOADED: bool = False


def _load_price_arrays_once() -> None:
    global _DATES, _PRICES
    if _DATES is not None:
        return
    import _app_ctx
    if _app_ctx.M is None:
        raise RuntimeError(
            "custom_fit import before _app_ctx.M populated. "
            "Ensure callbacks.custom_time is imported after app.py sets _app_ctx.M."
        )
    # Reuse the already-loaded price series
    _DATES = pd.to_datetime(_app_ctx.M.dates)
    _PRICES = np.asarray(_app_ctx.M.price_prices, dtype=np.float64)


def _load_block_array_once() -> None:
    global _BLOCKS, _BLOCK_CAP, _BLOCK_MAP_LOADED
    from pathlib import Path
    csv_path = Path(__file__).resolve().parent.parent.parent / "BitcoinBlocksDaily.csv"
    if not csv_path.exists():
        _LOG.error("custom_fit block map missing at %s; block mode disabled", csv_path)
        return
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        assert len(df) == len(_DATES), (
            f"block/price row count mismatch: {len(df)} vs {len(_DATES)}")
        assert (df["date"].values == _DATES.values).all(), \
            "block/price dates not aligned row-for-row"
        _BLOCKS = df["blockheight"].to_numpy(dtype=np.int64)
        # Compute _BLOCK_CAP from 2015-12-31
        cap_row = df[df["date"] <= pd.Timestamp("2015-12-31")].tail(1)
        _BLOCK_CAP = int(cap_row["blockheight"].iloc[0]) if len(cap_row) else None
        _BLOCK_MAP_LOADED = True
        _LOG.info("custom_fit loaded %d block rows, cap=%s", len(df), _BLOCK_CAP)
    except (FileNotFoundError,) as exc:
        _LOG.error("custom_fit block map file error: %s", exc)
    except AssertionError:
        # Data corruption — hard-fail per section 5 case P
        raise


def build_fit_input(scale: str, t0, weighting: str) -> FitInput:
    if _DATES is None:
        _load_price_arrays_once()
        _load_block_array_once()
    assert _DATES is not None and _PRICES is not None

    if scale == "calendar":
        t0_ts = pd.Timestamp(t0)
        t_raw = (_DATES - t0_ts).days.values.astype(np.float64) / 365.25
    elif scale == "block":
        if _BLOCKS is None:
            raise RuntimeError("block map unavailable; cannot build block-mode FitInput")
        t_raw = (_BLOCKS - int(t0)).astype(np.float64)
    else:
        raise ValueError(f"unknown scale {scale!r}")
    return FitInput(t=t_raw, price=_PRICES, weighting=weighting)
```

- [ ] **Step 4: Run tests → pass (may need to mock _app_ctx.M for the calendar test)**

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/custom_fit.py btc_web/test_custom_time.py
git commit -m "feat(custom_fit): cached arrays + build_fit_input"
```

### Task 5e: Preset drift guard + duplicate-t + slow warning tests

- [ ] **Step 1: Add preset tests**

Append:
```python
def test_cal_presets_all_before_2016():
    from _custom_time_presets import CAL_PRESETS
    from datetime import date
    for key, d, _ in CAL_PRESETS:
        assert d < date(2016, 1, 1), f"preset {key} is on/after 2016"


def test_cal_presets_count_frozen():
    from _custom_time_presets import CAL_PRESETS, BLK_PRESETS
    assert len(CAL_PRESETS) == 6
    assert len(BLK_PRESETS) == 5


def test_presets_are_tuples_not_lists():
    from _custom_time_presets import CAL_PRESETS, BLK_PRESETS
    assert isinstance(CAL_PRESETS, tuple)
    assert isinstance(BLK_PRESETS, tuple)


def test_duplicate_t_values_fit_ok():
    """Block mode produces duplicate-t for forward-filled days; fits handle it."""
    t = np.array([1.0]*20 + list(np.linspace(2.0, 20.0, 30)))
    rng = np.random.default_rng(0)
    p = 10 ** (5.0 * np.log10(t) + rng.normal(0, 0.05, len(t)))
    fi = cf.FitInput(t=t, price=p, weighting="none")
    # All 4 (where applicable) should return finite params
    r_pl = cf.fit_pl(fi); assert r_pl is not None and math.isfinite(r_pl.r2)
    r_exp = cf.fit_exp(fi); assert r_exp is not None and math.isfinite(r_exp.r2)
    r_qr = cf.fit_qr(fi); assert r_qr is not None  # 50 samples → full 9q
    # BM-floor needs ≥50 samples — len(t)=50, borderline
    r_bm = cf.fit_bm_floor(fi)
    assert r_bm is not None
```

- [ ] **Step 2: Run → pass**

- [ ] **Step 3: Commit**

```bash
git add btc_web/test_custom_time.py
git commit -m "test(custom_fit): preset drift + duplicate-t regression"
```

---

## Task 6: `btc_web/layout/custom_time.py` — panel UI

**Files:**
- Create: `btc_web/layout/custom_time.py`

- [ ] **Step 1: Write the layout module**

```python
"""Custom Time Axis panel — Tab 1 only.

See spec: docs/superpowers/specs/2026-04-13-custom-time-axis-design.md §2.
"""
from __future__ import annotations

import pandas as pd
from dash import dcc, html
import dash_bootstrap_components as dbc

from colors import DIM_TEXT, UI_FONT_SM, UI_FONT_MD
from layout.common import (_section_card, _row, _lbl, _STYLE_HIDDEN, _CB_MARGIN,
                            _INFO_ICON)
from _custom_time_presets import CAL_PRESETS, BLK_PRESETS


def _cal_dropdown_options():
    return [{"label": lbl, "value": key} for key, _d, lbl in CAL_PRESETS] + [
        {"label": "Custom…", "value": "custom"},
    ]


def _blk_dropdown_options(block_cap: int | None):
    opts = [{"label": lbl, "value": key} for key, _h, lbl in BLK_PRESETS]
    if block_cap is not None:
        opts.append({"label": "Custom…", "value": "custom"})
    return opts


def custom_time_panel():
    """Collapsible Custom Time Axis panel for Tab 1."""
    from engines.custom_fit import _BLOCK_CAP  # avoids circular at import time

    body = html.Div(id="cta-body", style=_STYLE_HIDDEN, children=[
        html.Small(
            "⚠ Custom Time Axis changes only affect this tab. Other tabs stay "
            "on the default axis (years since 2009-07-25).",
            style={"color": DIM_TEXT, "fontSize": UI_FONT_SM,
                    "display": "block", "marginBottom": "6px"},
        ),
        _row(
            html.Div([
                _lbl("Time scale"),
                dcc.RadioItems(
                    id="cta-scale",
                    options=[{"label": " Calendar (years)", "value": "calendar"},
                              {"label": " Blockheight (blocks)", "value": "block"}],
                    value="calendar",
                    labelStyle={"display": "inline-block", "marginRight": "12px"},
                    inputStyle=_CB_MARGIN,
                ),
            ]),
        ),
        html.Div(id="cta-t0-cal-wrap", children=[
            _lbl("t₀ (calendar)"),
            dcc.Dropdown(id="cta-t0-cal", options=_cal_dropdown_options(),
                          value="optimal", clearable=False),
            html.Div(id="cta-t0-cal-custom-wrap", style=_STYLE_HIDDEN, children=[
                _lbl("Custom date"),
                dcc.DatePickerSingle(
                    id="cta-t0-cal-custom",
                    min_date_allowed="2008-10-31",
                    max_date_allowed="2015-12-31",
                    display_format="YYYY-MM-DD",
                    initial_visible_month="2015-12-01",
                    placeholder="YYYY-MM-DD",
                ),
            ]),
        ]),
        html.Div(id="cta-t0-blk-wrap", style=_STYLE_HIDDEN, children=[
            _lbl("t₀ (block)"),
            dcc.Dropdown(id="cta-t0-blk",
                          options=_blk_dropdown_options(_BLOCK_CAP),
                          value="block_0", clearable=False),
            html.Div(id="cta-t0-blk-custom-wrap", style=_STYLE_HIDDEN, children=[
                _lbl("Custom block"),
                dbc.Input(id="cta-t0-blk-custom", type="number",
                           min=0, step=1, max=_BLOCK_CAP or 10**9,
                           debounce=True, placeholder="block number"),
            ]),
        ]),
        _row(
            html.Div([
                _lbl("Weighting"),
                dcc.Dropdown(
                    id="cta-weighting",
                    options=[
                        {"label": "Unweighted", "value": "none"},
                        {"label": "1/t", "value": "inv_t"},
                        {"label": "1/√t", "value": "inv_sqrt_t"},
                        {"label": "Uniform log-t density", "value": "log_density"},
                    ],
                    value="none",
                    clearable=False,
                ),
                html.Small("Applies to PL, QR, BM-floor. Exponential ignores.",
                            style={"color": DIM_TEXT, "fontSize": UI_FONT_SM}),
            ]),
        ),
        _row(
            html.Div([
                _lbl("Models"),
                dcc.Checklist(
                    id="cta-models",
                    options=[
                        {"label": " Power Law", "value": "pl"},
                        {"label": " Quantile Regression", "value": "qr"},
                        {"label": " BM floor", "value": "bm_floor"},
                        {"label": " Exponential", "value": "exp"},
                    ],
                    value=["pl", "qr", "bm_floor", "exp"],
                    labelStyle={"display": "block"},
                    inputStyle=_CB_MARGIN,
                ),
            ]),
        ),
        html.Div(id="cta-status",
                  style={"color": DIM_TEXT, "fontSize": UI_FONT_SM,
                          "marginTop": "8px"},
                  children="Press Activate to fit."),
    ])

    return _section_card(
        "Custom Time Axis",
        dcc.Checklist(
            id="cta-active",
            options=[{"label": " Activate Custom Time Axis", "value": "yes"}],
            value=[],
            labelStyle={"display": "block", "fontWeight": "bold"},
            inputStyle=_CB_MARGIN,
        ),
        body,
        no_hover=True,
    )
```

- [ ] **Step 2: Smoke-test import**

```bash
cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "
from layout.custom_time import custom_time_panel
print(custom_time_panel())" 2>&1 | head
```
Expected: a `SectionCard` representation prints, no errors.

- [ ] **Step 3: Commit**

```bash
git add btc_web/layout/custom_time.py
git commit -m "feat(custom_time): add panel layout"
```

---

## Task 7: `btc_web/callbacks/custom_time.py` — server callback + Store router

**Files:**
- Create: `btc_web/callbacks/custom_time.py`

- [ ] **Step 1: Write the callback module**

```python
"""Custom Time Axis callbacks — section 5 error cases encoded here.

See docs/superpowers/specs/2026-04-13-custom-time-axis-design.md §5.
"""
from __future__ import annotations

import logging
import time
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
from dash.exceptions import PreventUpdate

import _app_ctx
from colors import (
    MODEL_TRACE_COLORS, LINK, DIM_TEXT, UI_FONT_SM,
    TRACE_WIDTH, TRACE_WIDTH_SUPPORT,
)
from engines import custom_fit as cf
from _custom_time_presets import CAL_PRESET_BY_KEY, BLK_PRESET_BY_KEY

_LOG = logging.getLogger("custom_time")
_CAP_DATE = date(2016, 1, 1)


# Clientside: toggle cta-body visibility from the activate checklist
_app_ctx.app.clientside_callback(
    "function(v) { return (v && v.length) ? {} : {display:'none'}; }",
    Output("cta-body", "style"),
    Input("cta-active", "value"),
)

# Clientside: toggle cta-t0-cal-wrap ↔ cta-t0-blk-wrap on scale radio
_app_ctx.app.clientside_callback(
    """
    function(scale) {
        var show = {};
        var hide = {display: 'none'};
        var isCal = (scale === 'calendar');
        return [isCal ? show : hide, isCal ? hide : show];
    }
    """,
    Output("cta-t0-cal-wrap", "style"),
    Output("cta-t0-blk-wrap", "style"),
    Input("cta-scale", "value"),
)

# Clientside: reveal custom-date picker when dropdown == "custom"
_app_ctx.app.clientside_callback(
    "function(v) { return (v === 'custom') ? {} : {display:'none'}; }",
    Output("cta-t0-cal-custom-wrap", "style"),
    Input("cta-t0-cal", "value"),
)
_app_ctx.app.clientside_callback(
    "function(v) { return (v === 'custom') ? {} : {display:'none'}; }",
    Output("cta-t0-blk-custom-wrap", "style"),
    Input("cta-t0-blk", "value"),
)


def _resolve_t0(scale, cal_preset, cal_custom, blk_preset, blk_custom):
    """Return the resolved t0 value or raise PreventUpdate-inducing error."""
    if scale == "calendar":
        if cal_preset == "custom":
            if not cal_custom:
                return None, "Enter a date to fit."
            ts = pd.Timestamp(cal_custom).date()
            if ts >= _CAP_DATE:
                return None, "Custom t₀ must be before 2016-01-01."
            return ts.isoformat(), None
        return CAL_PRESET_BY_KEY[cal_preset][0].isoformat(), None
    else:
        if cf._BLOCKS is None:
            return None, "⚠ Block mode unavailable: BitcoinBlocksDaily.csv missing."
        if blk_preset == "custom":
            if blk_custom is None:
                return None, "Enter a block number to fit."
            if cf._BLOCK_CAP is not None and int(blk_custom) > cf._BLOCK_CAP:
                return None, f"Custom block must be ≤ {cf._BLOCK_CAP} (before 2016)."
            return int(blk_custom), None
        return BLK_PRESET_BY_KEY[blk_preset][0], None


def _build_figure(results: dict, scale: str, t0_label: str) -> go.Figure:
    fig = go.Figure()
    # Raw price series (muted gray)
    fig.add_trace(go.Scatter(
        x=(cf._DATES if scale == "calendar" else cf._BLOCKS),
        y=cf._PRICES, mode="markers",
        marker=dict(size=3, color=DIM_TEXT, opacity=0.5),
        name=f"Price (n={len(cf._PRICES):,})",
    ))
    colors = {
        "PL":       MODEL_TRACE_COLORS.get("pl", "#1B3352"),
        "QR":       MODEL_TRACE_COLORS.get("qr", "#9B2244"),
        "BM floor": MODEL_TRACE_COLORS.get("bub", "#C48209"),
        "Exp":      MODEL_TRACE_COLORS.get("exp", "#555555"),
    }
    for r in results.values():
        if r is None:
            continue
        color = colors.get(r.name, "#444")
        label_n = f"{r.n_samples:,}"
        if isinstance(r.y_plot, dict):
            # QR: one trace per quantile
            for q, y in r.y_plot.items():
                fig.add_trace(go.Scatter(
                    x=r.t_plot, y=10**y, mode="lines",
                    line=dict(color=color, width=TRACE_WIDTH),
                    name=f"{r.name} Q{int(q*100)}% (n={label_n})",
                    legendgroup=r.name,
                ))
        else:
            fig.add_trace(go.Scatter(
                x=r.t_plot, y=10**r.y_plot, mode="lines",
                line=dict(color=color, width=TRACE_WIDTH),
                name=f"{r.name} (n={label_n})",
            ))
    fig.update_layout(
        yaxis=dict(type="log", title="USD"),
        xaxis=dict(
            type="log",
            title=("Years since " + t0_label if scale == "calendar"
                    else f"Blockheight since block {t0_label}"),
        ),
        title=f"Custom Time Axis — t₀ = {t0_label}",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig


@callback(
    Output("bubble-graph", "figure", allow_duplicate=True),
    Output("cta-status", "children"),
    Output("bub-redraw-tick", "data"),
    Input("cta-active", "value"),
    Input("cta-scale", "value"),
    Input("cta-t0-cal", "value"),
    Input("cta-t0-cal-custom", "date"),
    Input("cta-t0-blk", "value"),
    Input("cta-t0-blk-custom", "value"),
    Input("cta-weighting", "value"),
    Input("cta-models", "value"),
    State("bub-redraw-tick", "data"),
    prevent_initial_call=True,
)
def custom_time_callback(active, scale, cal_preset, cal_custom,
                          blk_preset, blk_custom, weighting, models, tick):
    try:
        # 1. Deactivate → bump tick, preserve figure, restore status
        if not active or "yes" not in active:
            return no_update, "Standard view restored.", (tick or 0) + 1

        # 2. Module readiness (block-mode only)
        if scale == "block" and cf._BLOCKS is None:
            return no_update, "⚠ Block mode unavailable: BitcoinBlocksDaily.csv missing.", no_update

        # 3. Input validity
        if not models:
            return no_update, "Select at least one model to fit.", no_update
        t0, err = _resolve_t0(scale, cal_preset, cal_custom, blk_preset, blk_custom)
        if err:
            return no_update, err, no_update

        # 4. Build fit input
        t_start = time.perf_counter()
        fi = cf.build_fit_input(scale=scale, t0=t0, weighting=weighting)

        # 5. Run each selected model
        results = {}
        if "pl" in models:       results["pl"] = cf.fit_pl(fi)
        if "qr" in models:       results["qr"] = cf.fit_qr(fi)
        if "bm_floor" in models: results["bm_floor"] = cf.fit_bm_floor(fi)
        if "exp" in models:      results["exp"] = cf.fit_exp(fi)

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        if elapsed_ms > 5000:
            _LOG.warning("custom_fit slow: %dms params=%s", elapsed_ms,
                          {"scale": scale, "t0": t0, "weighting": weighting})

        # 6. Build figure + status
        fig = _build_figure(results, scale, str(t0))
        total_n = max((r.n_samples for r in results.values() if r is not None),
                       default=0)
        skipped = sum(1 for r in results.values() if r is None)
        status = (
            f"Fit on {total_n:,} samples from t₀={t0}. "
            f"{elapsed_ms}ms. " +
            (f"{skipped} model(s) skipped (see legend)." if skipped else "")
        )
        return fig, status, no_update

    except Exception as exc:
        _LOG.error("custom_fit crash: %s", exc, exc_info=True)
        return no_update, f"⚠ Internal error: {type(exc).__name__}", no_update
```

- [ ] **Step 2: Smoke-test import**

```bash
cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "
import app
print('OK')" 2>&1 | tail -5
```
Expected: `OK`. If it fails, the most likely cause is that `bub-redraw-tick` Store isn't yet registered in the layout — that's Task 9. Import should still succeed (the callback module is imported by `callbacks/__init__.py` only after layout).

- [ ] **Step 3: Commit**

```bash
git add btc_web/callbacks/custom_time.py
git commit -m "feat(custom_time): server callback + Store router clientside toggles"
```

---

## Task 8: Snapshot + routing registration

**Files:**
- Modify: `btc_web/snapshot.py`
- Modify: `btc_web/callbacks/routing.py`
- Create: `btc_web/test_custom_time_snapshot.py`

- [ ] **Step 1: Add 8 IDs to `_SNAPSHOT_CONTROLS` and `_CHECKLIST_OPTIONS`**

In `btc_web/snapshot.py`, find `_SNAPSHOT_CONTROLS` list and append:
```python
# Custom Time Axis (Tab 1)
("cta-active",         "value"),
("cta-scale",          "value"),
("cta-t0-cal",         "value"),
("cta-t0-cal-custom",  "date"),
("cta-t0-blk",         "value"),
("cta-t0-blk-custom",  "value"),
("cta-weighting",      "value"),
("cta-models",         "value"),
```

Find `_CHECKLIST_OPTIONS` dict and append:
```python
# cta-active: single-option checklist used as a toggle.
"cta-active": ["yes"],
# cta-models: order is LOAD-BEARING for bitmask encoding.
# Freeze this list. Reordering or removing entries breaks old share links.
"cta-models": ["pl", "qr", "bm_floor", "exp"],
```

- [ ] **Step 2: Add 8 IDs to `_TAB_CONTROLS["bubble"]`**

In `btc_web/callbacks/routing.py`, find `_TAB_CONTROLS["bubble"]` set and add the 8 strings.

- [ ] **Step 3: Write snapshot roundtrip tests**

`btc_web/test_custom_time_snapshot.py`:
```python
"""Snapshot encode/decode tests for Custom Time Axis controls."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "btc_web"))

from snapshot import (_encode_snapshot, _decode_snapshot,  # noqa: E402
                       _SNAPSHOT_CONTROLS, _CHECKLIST_OPTIONS)


def _base_state():
    """Minimal state dict with defaults for all snapshot controls."""
    return {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}


def test_cta_ids_registered():
    ids = {cid for cid, _ in _SNAPSHOT_CONTROLS}
    for expected in ["cta-active", "cta-scale", "cta-t0-cal", "cta-t0-cal-custom",
                      "cta-t0-blk", "cta-t0-blk-custom", "cta-weighting", "cta-models"]:
        assert expected in ids, f"missing {expected} in _SNAPSHOT_CONTROLS"


def test_cta_models_order_frozen():
    assert _CHECKLIST_OPTIONS["cta-models"] == ["pl", "qr", "bm_floor", "exp"]
    assert _CHECKLIST_OPTIONS["cta-active"] == ["yes"]


def test_cta_active_roundtrip_empty():
    state = _base_state()
    state["cta-active:value"] = []
    encoded = _encode_snapshot(state)
    decoded = _decode_snapshot(encoded)
    assert decoded["cta-active:value"] in (None, [])


def test_cta_active_roundtrip_set():
    state = _base_state()
    state["cta-active:value"] = ["yes"]
    encoded = _encode_snapshot(state)
    decoded = _decode_snapshot(encoded)
    assert decoded["cta-active:value"] == ["yes"]


def test_cta_models_bitmask_all_combinations():
    import itertools
    for combo in itertools.chain.from_iterable(
        itertools.combinations(["pl", "qr", "bm_floor", "exp"], r)
        for r in range(5)
    ):
        state = _base_state()
        state["cta-models:value"] = list(combo)
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert set(decoded["cta-models:value"] or []) == set(combo)


def test_unknown_weighting_forward_compat():
    """Old server reading a new snapshot doesn't crash."""
    state = _base_state()
    state["cta-weighting:value"] = "future_mode"
    encoded = _encode_snapshot(state)
    decoded = _decode_snapshot(encoded)
    assert decoded["cta-weighting:value"] == "future_mode"


def test_all_cta_ids_in_bubble_tab_controls():
    from callbacks.routing import _TAB_CONTROLS
    bubble_set = _TAB_CONTROLS["bubble"]
    for expected in ["cta-active", "cta-scale", "cta-t0-cal", "cta-t0-cal-custom",
                      "cta-t0-blk", "cta-t0-blk-custom", "cta-weighting", "cta-models"]:
        assert expected in bubble_set, f"{expected} not in _TAB_CONTROLS['bubble']"
```

- [ ] **Step 4: Run → pass**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/test_custom_time_snapshot.py -v 2>&1 | tail
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/snapshot.py btc_web/callbacks/routing.py btc_web/test_custom_time_snapshot.py
git commit -m "feat(custom_time): register 8 snapshot controls + tab routing"
```

---

## Task 9: Wire up `update_bubble` guard + Store

**Files:**
- Modify: `btc_web/callbacks/charts.py:953` (`update_bubble`)
- Modify: `btc_web/layout/__init__.py` (add `dcc.Store("bub-redraw-tick")`)
- Modify: `btc_web/callbacks/__init__.py` (import `custom_time`)

- [ ] **Step 1: Add `dcc.Store("bub-redraw-tick", data=0)` to layout root**

Find where other top-level stores are registered in `btc_web/layout/__init__.py` and add:
```python
dcc.Store(id="bub-redraw-tick", data=0),
```

- [ ] **Step 2: Modify `update_bubble` signature + guard**

Add to the `@callback` decorator arguments (after the last existing Input):
```python
Input("bub-redraw-tick",  "data"),
State("cta-active",       "value"),
```

Rename the function signature to accept the two new positional args. At the very top of the function body, insert:
```python
def update_bubble(..., _redraw_tick, cta_active):
    if cta_active and "yes" in cta_active:
        raise PreventUpdate
    ...
```

- [ ] **Step 3: Import `custom_time` in `callbacks/__init__.py`**

```python
from callbacks import custom_time  # noqa: F401
```

- [ ] **Step 4: Smoke-test**

```bash
cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "import app; print('OK')" 2>&1 | tail
```
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/layout/__init__.py btc_web/callbacks/__init__.py
git commit -m "feat(custom_time): wire Store router into update_bubble"
```

---

## Task 10: Integrate panel into `layout/bubble.py`

**Files:**
- Modify: `btc_web/layout/bubble.py`

- [ ] **Step 1: Import and insert `custom_time_panel()`**

At top of `layout/bubble.py`:
```python
from layout.custom_time import custom_time_panel
```

After `display_models_panel("bub", ...)` (around line 80), insert:
```python
custom_time_panel(),
```

- [ ] **Step 2: Smoke-test**

```bash
cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "import app; print('OK')" 2>&1 | tail
```

- [ ] **Step 3: Commit**

```bash
git add btc_web/layout/bubble.py
git commit -m "feat(custom_time): insert panel into Tab 1 layout"
```

---

## Task 11: Health check extension

**Files:**
- Modify: `btc_web/app.py`

- [ ] **Step 1: Find the `/health` route and add `block_map_loaded`**

Search for `/health` in `app.py`, then add to the returned JSON dict:
```python
try:
    from engines.custom_fit import _BLOCK_MAP_LOADED
    block_map_loaded = bool(_BLOCK_MAP_LOADED)
except Exception:
    block_map_loaded = False
return {
    ...,
    "block_map_loaded": block_map_loaded,
}
```

- [ ] **Step 2: Smoke-test**

```bash
cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "import app; print('OK')" 2>&1 | tail
```

- [ ] **Step 3: Commit**

```bash
git add btc_web/app.py
git commit -m "feat(custom_time): add block_map_loaded to /health"
```

---

## Task 12: Full test suite + fix regressions

- [ ] **Step 1: Run full unit suite (excluding E2E)**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py' 2>&1 | tail -30
```

Expected: all new tests pass. If any existing test breaks, fix the regression before continuing.

- [ ] **Step 2: Fix any regressions**

Most likely regression target is `test_snapshot.py:241-246` if a control ID was missed. Cross-check `_SNAPSHOT_CONTROLS` vs `_TAB_CONTROLS["bubble"]` alignment.

- [ ] **Step 3: Commit fixes if any**

---

## Task 13: Local dev server smoke test

- [ ] **Step 1: Start dev server**

```bash
lsof -ti :8050 | xargs -r kill -9
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 5 && tail /tmp/quantoshi_dev.log
```

Expected: `Dash is running on http://0.0.0.0:8050/`.

- [ ] **Step 2: Probe `/health`**

```bash
curl -s http://localhost:8050/health | python3 -m json.tool
```
Expected: JSON with `block_map_loaded: true`.

- [ ] **Step 3: Probe `/1` and check panel renders**

```bash
curl -s http://localhost:8050/1 | grep -c "cta-active"
```
Expected: non-zero (the panel HTML is in the initial DOM).

- [ ] **Step 4: Stop dev server**

```bash
lsof -ti :8050 | xargs -r kill -9
```

---

## Task 14: Deploy

**Files:**
- None (deploy only; Task 15 handles `update_prices.py` extension)

- [ ] **Step 1: Verify clean git status for the feature branch**

```bash
git status --short
```

- [ ] **Step 2: Push**

```bash
git push origin master 2>&1 | tail -5
```

- [ ] **Step 3: Pull on prod + flush Redis + restart**

```bash
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi && systemctl is-active quantoshi" 2>&1 | tail -10
```
Expected: `active`.

- [ ] **Step 4: Probe prod `/health`**

```bash
curl -s https://quantoshi.xyz/health | python3 -m json.tool | grep block_map_loaded
```
Expected: `"block_map_loaded": true`.

- [ ] **Step 5: Hit `/1` on prod**

```bash
curl -s https://quantoshi.xyz/1 | grep -c "cta-active"
```
Expected: non-zero.

---

## Task 15: Deferred items (document only, do not execute tonight)

The following are recorded in `UrgentTodoItems.md` under the deferred feature section and will be handled later:

- **E2E tests** (`test_custom_time_e2e.py`) — requires Playwright setup and a known-good baseline screenshot. Deferred until morning review.
- **Regression baseline** (`test_custom_time_baseline.py`) — requires running fits across all (model, weighting, preset) combinations and recording outputs. Deferred until morning verification.
- **`update_prices.py` integration** — adds `subprocess.run(["build_block_map.py", "--append"])` post-price-update. Deferred until manual verification of dev workflow.
- **`btc-web.service` restart limit** — requires root access on prod to edit systemd unit, plus `systemctl daemon-reload`. Deferred.
- **Integration tests** (`test_custom_time_integration.py`) — direct callback invocation pattern; adds ~3 sec to suite. Deferred pending refinement of `_patch_ctx` fixture.
- **Scan mode** — already in deferred UrgentTodoItems.

---

## Self-Review Checklist

- **Spec coverage:** all 6 spec sections have tasks (§1 arch → Task 7/8/9/10, §2 UX → Task 6, §3 fit engine → Task 5, §4 block map → Task 1/2/3, §5 error handling → Task 7 callback body, §6 testing → Task 5/8/12).
- **Placeholders:** `TBD`/`TODO`/`FIXME` grep on plan file → none.
- **Type consistency:** `FitInput`, `FitResult`, `_PriceDataShim`, `custom_time_callback`, `_resolve_t0`, `_build_figure` referenced consistently throughout.
- **Missing deps:** `fit_support` return-shape check in Task 5c step 4 confirms the real signature before commit.
