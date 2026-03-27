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

_FIGURE_TTL = 3600       # 1 hour default
_CITADEL_TTL = 86400     # 24 hours for pre-computed citadel results


def _cache_key(prefix: str, params_json: str) -> str:
    """Deterministic cache key from prefix + JSON params string."""
    h = hashlib.sha256(params_json.encode()).hexdigest()[:16]
    return f"fig:{prefix}:{h}"


def get_cached(prefix: str, params_json: str) -> dict | None:
    """Get a cached figure from Redis. Returns None on miss or error."""
    if not _HAS_REDIS:
        return None
    try:
        data = _REDIS.get(_cache_key(prefix, params_json))
        return json.loads(data) if data else None
    except Exception:
        return None


def set_cached(prefix: str, params_json: str, data: dict,
               ttl: int = _FIGURE_TTL) -> None:
    """Store a figure in Redis. Non-fatal on error."""
    if not _HAS_REDIS:
        return
    try:
        _REDIS.setex(_cache_key(prefix, params_json), ttl,
                     json.dumps(data, default=str))
    except Exception as e:
        logger.debug("Redis set failed: %s", e)


def get_citadel_cached(cache_key: str) -> dict | None:
    """Get a pre-computed Citadel result from Redis."""
    if not _HAS_REDIS:
        return None
    try:
        data = _REDIS.get(f"citadel:{cache_key}")
        return json.loads(data) if data else None
    except Exception:
        return None


def set_citadel_cached(cache_key: str, data: dict,
                       ttl: int = _CITADEL_TTL) -> None:
    """Store a pre-computed Citadel result in Redis."""
    if not _HAS_REDIS:
        return
    try:
        _REDIS.setex(f"citadel:{cache_key}", ttl,
                     json.dumps(data, default=str))
    except Exception as e:
        logger.debug("Redis citadel set failed: %s", e)


def redis_available() -> bool:
    """Check if Redis is connected."""
    return _HAS_REDIS
