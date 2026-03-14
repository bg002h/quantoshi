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

from btc_core import _find_lot_percentile, today_t
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

_NO_QUANTIZE_KEYS = {"selected_qs", "exit_qs"}  # must match qr_fits keys exactly

def _quantize_params(p: dict) -> dict:
    """Round all float values in a param dict to 3 sig figs."""
    out = {}
    for k, v in p.items():
        if k in _NO_QUANTIZE_KEYS:
            out[k] = v
        elif isinstance(v, float) and v != 0:
            out[k] = _q3(v)
        elif isinstance(v, list):
            out[k] = [_q3(x) if isinstance(x, float) and x != 0 else x for x in v]
        else:
            out[k] = v
    return out

# ── LRU figure caches (maxsize=8 per tab) ─────────────────────────────────────
# Each @lru_cache takes a JSON string key → go.Figure.  Bubble includes today's
# date in the key so the "today" line stays fresh (natural daily expiry).
# Server restarts on deploy clear all caches.

def _make_cached_builder(builder_fn, maxsize=64):
    @lru_cache(maxsize=maxsize)
    def _cached(key: str):
        return builder_fn(json.loads(key))
    return _cached

_cached_bubble_fig      = _make_cached_builder(build_bubble_figure, 16)
_cached_heatmap_fig     = _make_cached_builder(build_heatmap_figure, 32)
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

    always_mc: if True, skip mc_enabled check (used for MC-only heatmap).
    """
    mc_cached = p.pop("mc_cached", None)
    is_free = p.pop("mc_free_tier", False)
    p_q = _quantize_params(p)
    if always_mc or (p.get("mc_enabled") and _app_ctx._HAS_MARKOV):
        if is_free:
            return cache_fn(json.dumps(p_q, sort_keys=True, default=str))
        p_q["mc_cached"] = mc_cached
        return builder_fn(p_q)
    return cache_fn(json.dumps(p_q, sort_keys=True, default=str))

def _get_dca_fig(p: dict):
    return _get_mc_or_cached(p, build_dca_figure, _cached_dca_fig)

def _get_retire_fig(p: dict):
    return _get_mc_or_cached(p, build_retire_figure, _cached_retire_fig)

def _get_supercharge_fig(p: dict):
    return _get_mc_or_cached(p, build_supercharge_figure, _cached_supercharge_fig)

def _get_heatmap_fig(p: dict):
    p.pop("mc_cached", None)  # not used for QR heatmap
    p.pop("mc_free_tier", None)
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
        pct = _find_lot_percentile(today_t(_app_ctx.model.genesis), price, _app_ctx.model.mc_fits)
        if pct is not None:
            return round(pct * 100, 1)   # e.g. 7.5
    return 50.0   # fallback
