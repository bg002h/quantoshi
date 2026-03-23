"""Utility functions: float quantization, LRU figure caches, price fetching."""

import json
import logging
import math
import time
import urllib.request
from functools import lru_cache
from datetime import date

logger = logging.getLogger(__name__)

import _app_ctx

from btc_core import today_t, _find_lot_percentile
from figures import (build_bubble_figure, build_heatmap_figure,
                     build_mc_heatmap_figure,
                     build_dca_figure, build_retire_figure,
                     build_supercharge_figure)

# ── quantize floats to 3 significant figures for cache-friendly keys ───────────
def _q3(x):
    """Round a number to 3 significant figures."""
    if x is None or x == 0:
        return x
    exp = math.floor(math.log10(abs(x)))
    factor = 10 ** (exp - 2)
    return round(x / factor) * factor

_NO_QUANTIZE_KEYS = {"selected_qs", "exit_qs", "active_models"}

_SORT_LIST_KEYS = {"active_models", "selected_qs", "exit_qs", "delays"}

def _quantize_params(p: dict) -> dict:
    """Round all float values in a param dict to 3 sig figs.
    Sort list-valued keys for cache-key stability."""
    out = {}
    for k, v in p.items():
        if k in _NO_QUANTIZE_KEYS:
            out[k] = sorted(v) if k in _SORT_LIST_KEYS and isinstance(v, list) else v
        elif isinstance(v, float) and v != 0:
            out[k] = _q3(v)
        elif isinstance(v, list):
            normed = [_q3(x) if isinstance(x, float) and x != 0 else x for x in v]
            out[k] = sorted(normed) if k in _SORT_LIST_KEYS else normed
        else:
            out[k] = v
    return out

# ── LRU figure caches (maxsize=64 per tab, ~32 MB/worker) ─────────────────────
# Each @lru_cache takes a JSON string key → go.Figure.  Bubble includes today's
# date in the key so the "today" line stays fresh (natural daily expiry).
# Server restarts on deploy clear all caches.

def _make_cached_builder(builder_fn, maxsize=64):
    @lru_cache(maxsize=maxsize)
    def _cached(key: str):
        return builder_fn(_app_ctx.M, json.loads(key))
    return _cached

_cached_bubble_fig      = _make_cached_builder(build_bubble_figure)
_cached_heatmap_fig     = _make_cached_builder(build_heatmap_figure)
_cached_dca_fig         = _make_cached_builder(build_dca_figure)
_cached_retire_fig      = _make_cached_builder(build_retire_figure)
_cached_supercharge_fig = _make_cached_builder(build_supercharge_figure)
_cached_mc_heatmap_fig  = _make_cached_builder(build_mc_heatmap_figure)

def _get_bubble_fig(p: dict):
    p = _quantize_params(p)
    p['_day'] = str(date.today())
    return _cached_bubble_fig(json.dumps(p, sort_keys=True, default=str))

def _get_mc_or_cached(p: dict, builder_fn, cache_fn, always_mc=False):
    """Route to live MC build (bypass LRU) or LRU cache based on mc_enabled.

    Free-tier MC (mc_free_tier=True) goes through LRU cache — mc_cached is
    dropped so the key is JSON-serializable.  The overlay function falls
    through to get_cached_paths() for pre-computed data.

    When MC is not active, all mc_* params are stripped from the cache key
    so that users with different MC settings but identical QR settings share
    the same cache slot.

    always_mc: if True, skip mc_enabled check (used for MC-only heatmap).
    """
    mc_cached = p.pop("mc_cached", None)
    is_free = p.pop("mc_free_tier", False)
    mc_active = always_mc or (p.get("mc_enabled") and _app_ctx._HAS_MARKOV)
    if not mc_active:
        for k in [k for k in p if k.startswith("mc_")]:
            p.pop(k)
    p_q = _quantize_params(p)
    if mc_active:
        if is_free:
            return cache_fn(json.dumps(p_q, sort_keys=True, default=str))
        p_q["mc_cached"] = mc_cached
        return builder_fn(_app_ctx.M, p_q)
    return cache_fn(json.dumps(p_q, sort_keys=True, default=str))

def _get_dca_fig(p: dict):
    return _get_mc_or_cached(p, build_dca_figure, _cached_dca_fig)

def _get_retire_fig(p: dict):
    return _get_mc_or_cached(p, build_retire_figure, _cached_retire_fig)

def _get_supercharge_fig(p: dict):
    return _get_mc_or_cached(p, build_supercharge_figure, _cached_supercharge_fig)

def _get_heatmap_fig(p: dict):
    for k in [k for k in p if k.startswith("mc_")]:
        p.pop(k)
    p_q = _quantize_params(p)
    return _cached_heatmap_fig(json.dumps(p_q, sort_keys=True, default=str))


def _get_mc_heatmap_fig(p: dict):
    return _get_mc_or_cached(p, build_mc_heatmap_figure,
                             _cached_mc_heatmap_fig, always_mc=True)

def _nearest_quantile(target, qs):
    """Snap a percentile value to the nearest available quantile."""
    return min(qs, key=lambda q: abs(q - target))


_price_cache = {"price": None, "ts": 0}
_PRICE_TTL = 60  # seconds — avoid hammering upstream APIs from multiple workers

# ── 24h sparkline cache ──────────────────────────────────────────────────────
_spark_cache = {"svg": "", "ts": 0}
_SPARK_TTL = 300  # refresh sparkline every 5 min


def _fetch_sparkline_svg(width=60, height=18):
    """Fetch 24h hourly prices from Binance and build a tiny SVG sparkline."""
    now = time.time()
    if _spark_cache["svg"] and now - _spark_cache["ts"] < _SPARK_TTL:
        return _spark_cache["svg"]
    try:
        url = ("https://api.binance.com/api/v3/klines"
               "?symbol=BTCUSDT&interval=1h&limit=24")
        with urllib.request.urlopen(url, timeout=5) as r:
            klines = json.loads(r.read())
        closes = [float(k[4]) for k in klines]
        if len(closes) < 2:
            return _spark_cache["svg"]
    except Exception:
        return _spark_cache["svg"]

    lo, hi = min(closes), max(closes)
    rng = hi - lo or 1
    pad = 1  # 1px padding top/bottom
    yscale = (height - 2 * pad) / rng
    xstep = width / (len(closes) - 1)

    points = []
    for i, c in enumerate(closes):
        x = round(i * xstep, 1)
        y = round(pad + (hi - c) * yscale, 1)
        points.append(f"{x},{y}")

    # Color: green if up over 24h, red if down
    color = "#4cff88" if closes[-1] >= closes[0] else "#ff6b6b"
    polyline = " ".join(points)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
           f' viewBox="0 0 {width} {height}">'
           f'<polyline points="{polyline}" fill="none" stroke="{color}"'
           f' stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
           f'</svg>')
    import base64
    b64 = base64.b64encode(svg.encode()).decode()
    data_uri = f"data:image/svg+xml;base64,{b64}"
    _spark_cache.update({"svg": data_uri, "ts": now})
    return data_uri
_fail_streak = 0
_circuit_open_until = 0

def _fetch_btc_price():
    """Fetch current BTC price from multiple sources with fallback chain.

    Returns cached price if fetched within _PRICE_TTL seconds.
    After 3 consecutive all-source failures, skips fetches for 1 hour.
    """
    global _fail_streak, _circuit_open_until
    now = time.time()

    # TTL cache — return recent price without hitting APIs
    if _price_cache["price"] is not None and now - _price_cache["ts"] < _PRICE_TTL:
        return _price_cache["price"]

    # Circuit breaker — skip fetch if all sources repeatedly failed
    if now < _circuit_open_until:
        return _price_cache["price"]

    sources = [
        ("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
         lambda d: float(d["price"])),
        ("https://mempool.space/api/v1/prices",
         lambda d: float(d["USD"])),
        ("https://api.blockchain.info/ticker",
         lambda d: float(d["USD"]["last"])),
        ("https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
         lambda d: float(d["result"]["XXBTZUSD"]["c"][0])),
        ("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
         lambda d: float(d["bitcoin"]["usd"])),
    ]
    for url, parse in sources:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                price = parse(json.loads(r.read()))
                _price_cache.update({"price": price, "ts": now})
                _fail_streak = 0
                return price
        except Exception as exc:
            logger.debug("Price fetch failed for %s: %s", url.split("/")[2], exc)
            continue

    _fail_streak += 1
    if _fail_streak >= 3:
        _circuit_open_until = now + 3600
        logger.warning("All price sources failed %d times, circuit open for 1hr", _fail_streak)
    else:
        logger.warning("All price sources failed (streak %d)", _fail_streak)
    return _price_cache["price"]  # stale price better than None


def _startup_heatmap_defaults():
    """Fetch live BTC price at startup; return entry percentile (0–100 scale)."""
    price = _fetch_btc_price()
    if price is not None:
        pct = _find_lot_percentile(today_t(_app_ctx.M.genesis), price, _app_ctx.M.qr_fits)
        if pct is not None:
            return round(pct * 100, 1)   # e.g. 7.5
    return 50.0   # fallback
