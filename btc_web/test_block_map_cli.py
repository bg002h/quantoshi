"""Tests for tools/build_block_map.py. All bitcoind RPC is monkeypatched."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

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

        class _R:
            def read(_self):
                return json.dumps(result).encode()
        return _R()


def _stub_auth_cache(monkeypatch, conn):
    """Populate the auth cache with a pre-built fake connection."""
    monkeypatch.setattr(bm, "_AUTH_CACHE",
                        {"user": "u", "pw": "p", "conn": conn})


# ── Auth resolution ─────────────────────────────────────────────────────────

def test_auth_order_env_var_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("BITCOIN_RPC_URL", "http://u:p@localhost:9999/")
    monkeypatch.setattr(bm, "_COOKIE_PATH", tmp_path / ".cookie")
    monkeypatch.setattr(bm, "_CONF_PATH", tmp_path / "bitcoin.conf")
    (tmp_path / ".cookie").write_text("cookie:should_not_use")
    user, pw, host, port = bm._resolve_auth()
    assert (user, pw, host, port) == ("u", "p", "localhost", 9999)


def test_auth_order_cookie_over_conf(monkeypatch, tmp_path):
    monkeypatch.delenv("BITCOIN_RPC_URL", raising=False)
    cookie = tmp_path / ".cookie"
    cookie.write_text("__cookie__:abcdef123")
    conf = tmp_path / "bitcoin.conf"
    conf.write_text("rpcuser=baduser\nrpcpassword=badpass\n")
    monkeypatch.setattr(bm, "_COOKIE_PATH", cookie)
    monkeypatch.setattr(bm, "_CONF_PATH", conf)
    user, pw, _host, _port = bm._resolve_auth()
    assert (user, pw) == ("__cookie__", "abcdef123")


def test_auth_order_conf_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("BITCOIN_RPC_URL", raising=False)
    conf = tmp_path / "bitcoin.conf"
    conf.write_text("rpcuser=alice\nrpcpassword=secret\nrpcport=8332\n")
    monkeypatch.setattr(bm, "_COOKIE_PATH", tmp_path / "missing.cookie")
    monkeypatch.setattr(bm, "_CONF_PATH", conf)
    user, pw, _host, _port = bm._resolve_auth()
    assert (user, pw) == ("alice", "secret")


def test_auth_none_available_exits(monkeypatch, tmp_path):
    monkeypatch.delenv("BITCOIN_RPC_URL", raising=False)
    monkeypatch.setattr(bm, "_COOKIE_PATH", tmp_path / "missing.cookie")
    monkeypatch.setattr(bm, "_CONF_PATH", tmp_path / "missing.conf")
    with pytest.raises(SystemExit) as exc:
        bm._resolve_auth()
    assert exc.value.code == 2


# ── RPC dispatch ────────────────────────────────────────────────────────────

def test_rpc_single(monkeypatch):
    def dispatch(method, _params):
        assert method == "getblockcount"
        return 944790
    fake = _FakeConn(dispatch)
    _stub_auth_cache(monkeypatch, fake)
    monkeypatch.setattr(bm, "_get_conn", lambda: fake)
    assert bm._rpc("getblockcount") == 944790


def test_rpc_batch_preserves_order(monkeypatch):
    def dispatch(method, params):
        assert method == "getblockhash"
        return f"hash-{params[0]}"
    fake = _FakeConn(dispatch)
    _stub_auth_cache(monkeypatch, fake)
    monkeypatch.setattr(bm, "_get_conn", lambda: fake)
    result = bm._rpc_batch([("getblockhash", [i]) for i in range(5)])
    assert result == [f"hash-{i}" for i in range(5)]


# ── Running-max algorithm ───────────────────────────────────────────────────

def test_running_max_handles_nonmonotonic():
    """Real non-monotonic pair from tools/find_nonmonotonic_blocks.py:
    block 30158 time=1261090051, block 30159 time=1261089909 (delta -142s).

    The running_max stays at 1261090051 at both indices 2 and 3 because
    block 30159's declared time (1261089909) is earlier than the previous
    block's time. The lookup therefore correctly treats block 30159 as
    arriving "at" the same chain-tip time as block 30158."""
    heights = [30156, 30157, 30158, 30159, 30160]
    times = [1261089000, 1261089500, 1261090051, 1261089909, 1261090500]
    running_max = bm._compute_running_max(times)
    assert running_max == [1261089000, 1261089500, 1261090051,
                            1261090051, 1261090500]

    # Target AFTER block 30159's real time but BEFORE block 30160's:
    # running_max is 1261090051 at both 30158 and 30159; the highest such
    # height is 30159, so lookup returns 30159.
    assert bm._lookup_height_before(heights, running_max, 1261090200) == 30159
    # Target BEFORE the non-monotonic pair settled: between 30157 and 30158.
    assert bm._lookup_height_before(heights, running_max, 1261089700) == 30157
    # Target after 30160's time: returns 30160
    assert bm._lookup_height_before(heights, running_max, 1261090600) == 30160
    # Target before everything: returns heights[0]-1
    assert bm._lookup_height_before(heights, running_max, 1261088000) == 30155


# ── Full build ──────────────────────────────────────────────────────────────

def _make_dispatch(heights, hashes, times):
    def dispatch(method, params):
        if method == "getblockcount":
            return max(heights)
        if method == "getblockhash":
            return hashes[params[0] - heights[0]]
        if method == "getblockheader":
            idx = hashes.index(params[0])
            return {"height": heights[idx], "time": times[idx]}
        raise ValueError(method)
    return dispatch


def test_full_build_writes_atomic_csv(monkeypatch, tmp_path):
    # Synthetic chain: block 0..499 with 10-min intervals, starting 2010-02-12 00:00 UTC
    start_ts = 1265932800  # 2010-02-12 00:00:00 UTC
    heights = list(range(0, 500))
    hashes = [f"hash{h:06d}" for h in heights]
    times = [start_ts + h * 600 for h in heights]
    dispatch = _make_dispatch(heights, hashes, times)

    conn = _FakeConn(dispatch)
    _stub_auth_cache(monkeypatch, conn)
    monkeypatch.setattr(bm, "_get_conn", lambda: conn)

    # Fake price CSV with 3 rows (M/D/YY format matches real BitcoinPricesDaily.csv)
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
    # No .tmp file left behind
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())
    # Blockheights should be monotone non-decreasing
    assert list(df["blockheight"]) == sorted(df["blockheight"])


# ── Append (gap fill) ───────────────────────────────────────────────────────

def test_append_fills_middle_gap(monkeypatch, tmp_path):
    start_ts = 1265932800
    heights = list(range(0, 700))
    hashes = [f"hash{h:06d}" for h in heights]
    times = [start_ts + h * 600 for h in heights]
    dispatch = _make_dispatch(heights, hashes, times)
    conn = _FakeConn(dispatch)
    _stub_auth_cache(monkeypatch, conn)
    monkeypatch.setattr(bm, "_get_conn", lambda: conn)

    price_csv = tmp_path / "BitcoinPricesDaily.csv"
    price_csv.write_text(
        "Date,Price\n2/12/10,100\n2/13/10,101\n2/14/10,102\n2/15/10,103\n"
    )
    block_csv = tmp_path / "BitcoinBlocksDaily.csv"
    block_csv.write_text(
        "date,blockheight\n2010-02-12,140\n2010-02-14,430\n")
    monkeypatch.setattr(bm, "_PRICE_CSV", price_csv)
    monkeypatch.setattr(bm, "_BLOCK_CSV", block_csv)

    bm.main_append()

    df = pd.read_csv(block_csv)
    assert len(df) == 4
    assert set(df["date"]) == {
        "2010-02-12", "2010-02-13", "2010-02-14", "2010-02-15"}


# ── Verify ──────────────────────────────────────────────────────────────────

def test_verify_catches_corruption(monkeypatch, tmp_path):
    # Synthetic chain spans 2000 blocks so the verify fetch window
    # (±500 around the CSV's blockheights) stays in-range.
    start_ts = 1265932800
    heights = list(range(0, 2000))
    hashes = [f"hash{h:06d}" for h in heights]
    times = [start_ts + h * 600 for h in heights]
    dispatch = _make_dispatch(heights, hashes, times)
    conn = _FakeConn(dispatch)
    _stub_auth_cache(monkeypatch, conn)
    monkeypatch.setattr(bm, "_get_conn", lambda: conn)

    price_csv = tmp_path / "BitcoinPricesDaily.csv"
    price_csv.write_text("Date,Price\n2/12/10,100\n")
    block_csv = tmp_path / "BitcoinBlocksDaily.csv"
    # True value for 2010-02-12 with 10-min blocks starting at midnight UTC
    # is block 143 (144 * 600 sec = 86400 sec = 1 day). CSV says 999 — wrong.
    block_csv.write_text("date,blockheight\n2010-02-12,999\n")
    monkeypatch.setattr(bm, "_PRICE_CSV", price_csv)
    monkeypatch.setattr(bm, "_BLOCK_CSV", block_csv)

    with pytest.raises(SystemExit) as exc:
        bm.main_verify()
    assert exc.value.code == 3
