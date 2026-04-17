"""Shared Redis-backed figure cache with per-worker LRU fallback.

Two-layer cache:
  L1: per-worker functools.lru_cache (existing, fast, no I/O)
  L2: Redis shared across all workers (slower, but shared)

If Redis is unavailable, falls back to L1-only (pre-Celery behavior).

All singleton flags (_HAS_REDIS, _MODEL_FP, etc.) live in _app_ctx.py.
This module reads them from there — no independent evaluation.
"""
from __future__ import annotations

import hashlib
import json
import logging

import _app_ctx

logger = logging.getLogger(__name__)

_REDIS = _app_ctx._REDIS
_HAS_REDIS = _app_ctx._HAS_REDIS
_MODEL_FP = _app_ctx._MODEL_FP
logger.info("Model fingerprint: %s (Redis: %s)", _MODEL_FP, _HAS_REDIS)

# No TTL — cache is invalidated only by model data changes (fingerprint)
_REDIS_TTL = 0  # 0 = no expiry (Redis LRU eviction handles memory pressure)


def _cache_key(prefix: str, params_json: str) -> str:
    """Deterministic cache key including model fingerprint.
    When model_data.pkl changes, all keys miss automatically."""
    h = hashlib.sha256(params_json.encode()).hexdigest()[:32]
    return f"fig:{_MODEL_FP}:{prefix}:{h}"


def get_cached(prefix: str, params_json: str) -> dict | None:
    """Get a cached figure from Redis. Returns None on miss or error.

    Storage format: figure JSON is the primary key's raw value (no outer
    json.dumps wrapping). If the result was a (fig, mc_result) tuple, a
    sibling ``{key}:mc`` holds the JSON-encoded mc_result — a literal
    "null" string when the tuple's mc_result was None, so the reader
    can distinguish "tuple with None mc" from "non-tuple single fig".
    This avoids the 2x escape overhead of embedding a JSON string inside
    another JSON object.
    """
    if not _HAS_REDIS:
        return None
    try:
        key = _cache_key(prefix, params_json)
        fig_raw = _REDIS.get(key)
        if fig_raw is None:
            return None
        if isinstance(fig_raw, bytes):
            fig_raw = fig_raw.decode()
        mc_raw = _REDIS.get(key + ":mc")
        if mc_raw is None:
            # No sibling key → this entry was written as a single figure.
            return {"figure": fig_raw, "mc_result": None, "is_tuple": False}
        if isinstance(mc_raw, bytes):
            mc_raw = mc_raw.decode()
        # Sibling exists → was stored as tuple. mc_raw may decode to None.
        return {"figure": fig_raw, "mc_result": json.loads(mc_raw),
                "is_tuple": True}
    except Exception:
        return None


def set_cached(prefix: str, params_json: str, data: dict) -> None:
    """Store a figure in Redis. No TTL — persists until model changes
    (fingerprint in key) or Redis LRU eviction. Non-fatal on error.

    See get_cached for the two-key storage layout. When the result was
    a tuple (builder returned ``(fig, mc)``), the ``{key}:mc`` sibling
    is ALWAYS written — even if mc is None — so the reader can tell
    the difference between "tuple with None mc" (DCA/Retire/SC with
    MC-disabled) and "non-tuple single fig" (Bubble / Heatmap). Absent
    that sibling on None-valued tuples, callers that destructure as
    ``fig, mc = _get_*_fig(...)`` blow up with ValueError on cache hit.
    """
    if not _HAS_REDIS:
        return
    try:
        fig_str = data.get("figure")
        if not fig_str:
            return
        key = _cache_key(prefix, params_json)
        pipe = _REDIS.pipeline()
        pipe.set(key, fig_str)
        if data.get("is_tuple"):
            # ALWAYS write the sibling for tuple results, including None.
            pipe.set(key + ":mc", json.dumps(data.get("mc_result"),
                                              default=str))
        pipe.execute()
    except Exception as e:
        logger.debug("Redis set failed: %s", e)


def get_citadel_cached(cache_key: str) -> dict | None:
    """Get a pre-computed Citadel result from Redis."""
    if not _HAS_REDIS:
        return None
    try:
        data = _REDIS.get(f"citadel:{_MODEL_FP}:{cache_key}")
        return json.loads(data) if data else None
    except Exception:
        return None


def set_citadel_cached(cache_key: str, data: dict) -> None:
    """Store a pre-computed Citadel result in Redis. No TTL — persists
    until model changes (fingerprint in key) or LRU eviction."""
    if not _HAS_REDIS:
        return
    try:
        _REDIS.set(f"citadel:{_MODEL_FP}:{cache_key}",
                    json.dumps(data, default=str))
    except Exception as e:
        logger.debug("Redis citadel set failed: %s", e)


def redis_available() -> bool:
    """Check if Redis is connected."""
    return _app_ctx.redis_available()


# ── L0 persistent cache (pinned defaults in Redis) ──────────────────────────
from tab_defaults import _DEFAULTS_HASH

_L0_FINGERPRINT = hashlib.md5(
    f"{_MODEL_FP}:{_DEFAULTS_HASH}".encode()
).hexdigest()[:12]
logger.info("L0 fingerprint: %s (model=%s, defaults=%s)",
            _L0_FINGERPRINT, _MODEL_FP, _DEFAULTS_HASH)

_L0_TTL = 7 * 24 * 3600  # 7 days


def _l0_key(prefix: str, params_hash: str) -> str:
    """Redis key for an L0 pinned entry."""
    return f"l0:{_L0_FINGERPRINT}:{prefix}:{params_hash}"


def get_l0(prefix: str, params_hash: str) -> str | None:
    """Get an L0 entry from Redis. Returns raw JSON string or None."""
    if not _HAS_REDIS:
        return None
    try:
        data = _REDIS.get(_l0_key(prefix, params_hash))
        return data.decode() if data else None
    except Exception:
        return None


def set_l0(prefix: str, params_hash: str, json_str: str) -> bool:
    """Store an L0 entry in Redis with 7-day TTL. Returns True on success."""
    if not _HAS_REDIS:
        return False
    try:
        _REDIS.setex(_l0_key(prefix, params_hash), _L0_TTL, json_str)
        return True
    except Exception as e:
        logger.debug("Redis L0 set failed: %s", e)
        return False


def scan_l0_keys() -> list[str]:
    """Return all L0 keys matching current fingerprint. Diagnostic utility."""
    if not _HAS_REDIS:
        return []
    try:
        pattern = f"l0:{_L0_FINGERPRINT}:*"
        return [k.decode() for k in _REDIS.scan_iter(match=pattern, count=100)]
    except Exception:
        return []
