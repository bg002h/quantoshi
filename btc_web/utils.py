"""Utility functions: float quantization, LRU figure caches, price fetching."""

import hashlib
import json
import logging
import time
import urllib.request
from functools import lru_cache
from datetime import date

logger = logging.getLogger(__name__)

import _app_ctx
from colors import SPARKLINE_UP, UCL_LINE_COLOR as _SPARKLINE_DOWN

from btc_core import today_t, _find_lot_percentile
from figures import (build_bubble_figure, build_heatmap_figure,
                     build_mc_heatmap_figure,
                     build_dca_figure, build_retire_figure,
                     build_supercharge_figure, build_citadel_figure)
from figures.heatmap import build_cagr_line_figure
from figures.residuals import build_residuals_figure

# ── quantize floats to 3 significant figures for cache-friendly keys ───────────
from _app_ctx import _q3

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

def _make_cached_builder(builder_fn, maxsize=64, prefix="fig"):
    """Create an LRU-cached figure builder with optional Redis L2.

    Three cache layers:
      L0: pinned dict for prewarm defaults (never evicted, clears on restart)
      L1: per-worker @lru_cache (fast, no I/O, LRU eviction)
      L2: shared Redis cache (shared across workers, model-fingerprinted)
    """
    _pinned = {}  # L0: default figures, never evicted by LRU

    @lru_cache(maxsize=maxsize)
    def _cached(key: str):
        # L0: check pinned defaults (never evicted)
        if key in _pinned:
            return _pinned[key]

        # L2: check Redis before computing
        try:
            from cache import get_cached, redis_available
            if redis_available():
                hit = get_cached(prefix, key)
                if hit is not None:
                    import plotly.io as pio
                    fig_data = hit.get("figure")
                    mc_data = hit.get("mc_result")
                    is_tuple = hit.get("is_tuple", False)
                    if fig_data:
                        fig = pio.from_json(fig_data)
                        return (fig, mc_data) if is_tuple else fig
        except Exception:
            pass

        # L2 miss — compute
        result = builder_fn(_app_ctx.M, json.loads(key))

        # Store in Redis L2
        try:
            from cache import set_cached, redis_available
            if redis_available():
                is_tuple = isinstance(result, tuple)
                fig = result[0] if is_tuple else result
                mc = result[1] if is_tuple else None
                data = {"figure": fig.to_json() if hasattr(fig, 'to_json') else None,
                        "mc_result": mc,
                        "is_tuple": is_tuple}
                set_cached(prefix, key, data)
        except Exception:
            pass

        return result

    def _pin(key: str, result):
        """Pin a result so it's never evicted by LRU. Used by prewarm."""
        _pinned[key] = result

    _cached.pin = _pin
    _cached.pinned = _pinned
    return _cached

_cached_bubble_fig      = _make_cached_builder(build_bubble_figure, prefix="bub")
_cached_heatmap_fig     = _make_cached_builder(build_heatmap_figure, prefix="hm")
_cached_dca_fig         = _make_cached_builder(build_dca_figure, prefix="dca")
_cached_retire_fig      = _make_cached_builder(build_retire_figure, prefix="ret")
_cached_supercharge_fig = _make_cached_builder(build_supercharge_figure, prefix="sc")
_cached_mc_heatmap_fig  = _make_cached_builder(build_mc_heatmap_figure, prefix="hm_mc")
_cached_citadel_fig     = _make_cached_builder(build_citadel_figure, prefix="cp")
_cached_cagr_fig        = _make_cached_builder(build_cagr_line_figure, prefix="cagr")
_cached_resid_fig       = _make_cached_builder(build_residuals_figure, prefix="resid")

_ALL_CACHES = {
    "bubble": _cached_bubble_fig,
    "heatmap": _cached_heatmap_fig,
    "dca": _cached_dca_fig,
    "retire": _cached_retire_fig,
    "supercharge": _cached_supercharge_fig,
    "mc_heatmap": _cached_mc_heatmap_fig,
    "citadel": _cached_citadel_fig,
    "cagr": _cached_cagr_fig,
    "resid": _cached_resid_fig,
}

def _log_cache_stats():
    """Log LRU cache hit rates for all figure caches."""
    for name, cache in _ALL_CACHES.items():
        info = cache.cache_info()
        total = info.hits + info.misses
        rate = f"{info.hits/total:.0%}" if total else "n/a"
        logger.info("cache/%s: hits=%d misses=%d size=%d/%d rate=%s",
                     name, info.hits, info.misses, info.currsize,
                     info.maxsize, rate)

def _get_bubble_fig(p: dict):
    _try_flush_l0()
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
    _try_flush_l0()
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

def _get_mc_bubble_fig(p: dict):
    """MC-aware bubble cache wrapper. Routes through _get_mc_or_cached
    so mc_cached dicts are stripped from the JSON cache key when MC is
    disabled and passed directly to the builder when MC is enabled.
    update_bubble switches to this wrapper only when mc_enabled is truthy;
    the non-MC fast path stays on _get_bubble_fig."""
    return _get_mc_or_cached(p, build_bubble_figure, _cached_bubble_fig)

def _get_dca_fig(p: dict):
    return _get_mc_or_cached(p, build_dca_figure, _cached_dca_fig)

def _get_retire_fig(p: dict):
    return _get_mc_or_cached(p, build_retire_figure, _cached_retire_fig)

def _get_supercharge_fig(p: dict):
    return _get_mc_or_cached(p, build_supercharge_figure, _cached_supercharge_fig)

def _get_heatmap_fig(p: dict):
    _try_flush_l0()
    for k in [k for k in p if k.startswith("mc_")]:
        p.pop(k)
    # live_price is a per-minute ticker value; dropping it from the cache
    # key trades display precision (entry-price title can lag by up to an
    # hour) for near-100% cache hit rate on first-visit across users.
    p.pop("live_price", None)
    p_q = _quantize_params(p)
    return _cached_heatmap_fig(json.dumps(p_q, sort_keys=True, default=str))


def _get_mc_heatmap_fig(p: dict):
    return _get_mc_or_cached(p, build_mc_heatmap_figure,
                             _cached_mc_heatmap_fig, always_mc=True)

def _get_citadel_fig(p: dict):
    _try_flush_l0()
    p_q = _quantize_params(p)
    return _cached_citadel_fig(json.dumps(p_q, sort_keys=True, default=str))

def _get_cagr_fig(p: dict):
    _try_flush_l0()
    p_q = _quantize_params(p)
    p_q['_day'] = str(date.today())
    return _cached_cagr_fig(json.dumps(p_q, sort_keys=True, default=str))

def _get_resid_fig(p: dict):
    _try_flush_l0()
    p_q = _quantize_params(p)
    p_q['_day'] = str(date.today())
    return _cached_resid_fig(json.dumps(p_q, sort_keys=True, default=str))


# ── L0 prewarm helpers ───────────────────────────────────────────────────────
import plotly.io as _pio

_l0_needs_flush = False  # set True when Redis was down at boot

_L0_BUILDERS = {
    "bub":  (_cached_bubble_fig, _get_bubble_fig),
    "hm":   (_cached_heatmap_fig, _get_heatmap_fig),
    "dca":  (_cached_dca_fig, _get_dca_fig),
    "ret":  (_cached_retire_fig, _get_retire_fig),
    "sc":   (_cached_supercharge_fig, _get_supercharge_fig),
    "cp":   (_cached_citadel_fig, _get_citadel_fig),
}


def _compute_cache_key(prefix: str, p: dict) -> str:
    """Compute the JSON cache key for a param dict, same as _get_*_fig would.

    NOTE: Only correct for prewarm defaults (no mc_* keys). For params with
    mc_* keys, the strip order differs from _get_mc_or_cached. This is fine
    because L0 only stores prewarm defaults.
    """
    p_q = _quantize_params(p)
    if not p_q.get("mc_enabled"):
        p_q = {k: v for k, v in p_q.items() if not k.startswith("mc_")}
    p_q.pop("mc_cached", None)
    p_q.pop("mc_free_tier", None)
    if prefix == "bub":
        p_q["_day"] = str(date.today())
    return json.dumps(p_q, sort_keys=True, default=str)


def _serialize_result(result) -> str:
    """Serialize a figure (or fig+mc tuple) to JSON for Redis storage."""
    is_tuple = isinstance(result, tuple)
    fig = result[0] if is_tuple else result
    mc = result[1] if is_tuple else None
    return json.dumps({
        "figure": fig.to_json() if hasattr(fig, 'to_json') else None,
        "mc_result": mc,
        "is_tuple": is_tuple,
    }, default=str)


def _deserialize_result(json_str: str):
    """Deserialize a figure (or fig+mc tuple) from Redis JSON."""
    data = json.loads(json_str)
    fig = _pio.from_json(data["figure"]) if data.get("figure") else None
    if data.get("is_tuple"):
        return (fig, data.get("mc_result"))
    return fig


def _try_flush_l0():
    """One-shot: write in-memory L0 to Redis if not written at boot."""
    global _l0_needs_flush
    if not _l0_needs_flush:
        return
    try:
        from cache import set_l0, redis_available
        if not redis_available():
            return
        count = 0
        for prefix, (cache_fn, _) in _L0_BUILDERS.items():
            for key, result in cache_fn.pinned.items():
                params_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
                set_l0(prefix, params_hash, _serialize_result(result))
                count += 1
        _l0_needs_flush = False
        logger.info("L0 deferred flush: wrote %d entries to Redis", count)
    except Exception as e:
        logger.debug("L0 deferred flush failed: %s", e)


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
    color = SPARKLINE_UP if closes[-1] >= closes[0] else _SPARKLINE_DOWN
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

    # Check Redis first (populated by Celery Beat every 20min)
    try:
        from cache import _REDIS, _HAS_REDIS
        if _HAS_REDIS:
            cached = _REDIS.get("btc:price")
            if cached:
                import json as _json
                price = _json.loads(cached).get("price")
                if price:
                    _price_cache["price"] = price
                    _price_cache["ts"] = now
                    return price
    except Exception:
        pass

    # In-process TTL cache — return recent price without hitting APIs
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
