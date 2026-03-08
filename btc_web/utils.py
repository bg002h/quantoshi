"""Utility functions: float quantization, LRU figure caches, price fetching."""

import json
import logging
import math
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

@lru_cache(maxsize=8)
def _cached_bubble_fig(key: str):
    return build_bubble_figure(_app_ctx.M, json.loads(key))

@lru_cache(maxsize=8)
def _cached_heatmap_fig(key: str):
    return build_heatmap_figure(_app_ctx.M, json.loads(key))

@lru_cache(maxsize=8)
def _cached_dca_fig(key: str):
    return build_dca_figure(_app_ctx.M, json.loads(key))

@lru_cache(maxsize=8)
def _cached_retire_fig(key: str):
    return build_retire_figure(_app_ctx.M, json.loads(key))

@lru_cache(maxsize=8)
def _cached_supercharge_fig(key: str):
    return build_supercharge_figure(_app_ctx.M, json.loads(key))

def _get_bubble_fig(p: dict):
    p = _quantize_params(p)
    p['_day'] = str(date.today())
    return _cached_bubble_fig(json.dumps(p, sort_keys=True, default=str))

def _get_mc_or_cached(p: dict, builder_fn, cache_fn):
    """Route to live MC build (bypass LRU) or LRU cache based on mc_enabled."""
    mc_cached = p.pop("mc_cached", None)
    p_q = _quantize_params(p)
    if p.get("mc_enabled") and _app_ctx._HAS_MARKOV:
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
    p.pop("mc_cached", None)  # not used for QR heatmap
    p_q = _quantize_params(p)
    return _cached_heatmap_fig(json.dumps(p_q, sort_keys=True, default=str))


def _get_mc_heatmap_fig(p: dict):
    mc_cached = p.pop("mc_cached", None)
    p_q = _quantize_params(p)
    p_q["mc_cached"] = mc_cached
    return build_mc_heatmap_figure(_app_ctx.M, p_q)

def _nearest_quantile(target, qs):
    """Snap a percentile value to the nearest available quantile."""
    return min(qs, key=lambda q: abs(q - target))


def _fetch_btc_price():
    """Fetch current BTC price from multiple sources with fallback chain."""
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
                return parse(json.loads(r.read()))
        except Exception as exc:
            logger.debug("Price fetch failed for %s: %s", url.split("/")[2], exc)
            continue
    logger.warning("All price sources failed")
    return None


def _startup_heatmap_defaults():
    """Fetch live BTC price at startup; return entry percentile (0–100 scale)."""
    price = _fetch_btc_price()
    if price is not None:
        pct = _find_lot_percentile(today_t(_app_ctx.M.genesis), price, _app_ctx.M.qr_fits)
        if pct is not None:
            return round(pct * 100, 1)   # e.g. 7.5
    return 50.0   # fallback
