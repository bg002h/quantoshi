"""Shared Redis-backed figure cache with per-worker LRU fallback.

Two-layer cache:
  L1: per-worker functools.lru_cache (existing, fast, no I/O)
  L2: Redis shared across all workers (slower, but shared)

If Redis is unavailable, falls back to L1-only (pre-Celery behavior).
"""
from __future__ import annotations

import hashlib
import json
import logging

logger = logging.getLogger(__name__)

try:
    import redis
    _REDIS = redis.Redis(host='localhost', port=6379, db=0,
                         socket_timeout=1, socket_connect_timeout=1)
    _REDIS.ping()
    _HAS_REDIS = True
    logger.info("Redis connected for shared figure cache")
except Exception:
    _REDIS = None
    _HAS_REDIS = False
    logger.info("Redis not available — using per-worker LRU only")

# ── Model fingerprint: invalidates cache when model_data.pkl changes ────────
# Cache entries are keyed by this fingerprint. When new price data arrives
# and the notebook regenerates model_data.pkl, all cache keys change
# automatically — no TTL needed, no manual flush needed.

def _compute_model_fingerprint() -> str:
    """Fingerprint based on model_data.pkl mtime + size.
    Changes whenever the pkl is regenerated (new price data)."""
    import os
    for path in ("btc_app/model_data.pkl", "archive/btc_app/model_data.pkl"):
        if os.path.exists(path):
            st = os.stat(path)
            return hashlib.md5(f"{st.st_mtime}:{st.st_size}".encode()).hexdigest()[:8]
    return "unknown"

_MODEL_FP = _compute_model_fingerprint()
logger.info("Model fingerprint: %s", _MODEL_FP)

# No TTL — cache is invalidated only by model data changes (fingerprint)
_REDIS_TTL = 0  # 0 = no expiry (Redis LRU eviction handles memory pressure)


def _cache_key(prefix: str, params_json: str) -> str:
    """Deterministic cache key including model fingerprint.
    When model_data.pkl changes, all keys miss automatically."""
    h = hashlib.sha256(params_json.encode()).hexdigest()[:32]
    return f"fig:{_MODEL_FP}:{prefix}:{h}"


def get_cached(prefix: str, params_json: str) -> dict | None:
    """Get a cached figure from Redis. Returns None on miss or error."""
    if not _HAS_REDIS:
        return None
    try:
        data = _REDIS.get(_cache_key(prefix, params_json))
        return json.loads(data) if data else None
    except Exception:
        return None


def set_cached(prefix: str, params_json: str, data: dict) -> None:
    """Store a figure in Redis. No TTL — persists until model changes
    (fingerprint in key) or Redis LRU eviction. Non-fatal on error."""
    if not _HAS_REDIS:
        return
    try:
        _REDIS.set(_cache_key(prefix, params_json),
                   json.dumps(data, default=str))
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
    return _HAS_REDIS


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
