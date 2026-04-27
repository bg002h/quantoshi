"""Phase 1 integration tests — time_basis plumbing across modules."""
from __future__ import annotations
import hashlib

import pytest


def test_cache_l1_prefix_includes_time_basis():
    """Cache key prefix carries the axis so calendar/block won't collide.

    cache.py uses neighbor-import (`import _app_ctx`, no btc_web. prefix)
    because gunicorn runs with btc_web/ on sys.path. Tests must match.
    """
    import sys
    sys.path.insert(0, "btc_web")
    import cache  # noqa: E402
    from time_basis import TIME_BASIS  # noqa: E402
    key = cache._cache_key("bub", '{"a": 1, "b": 2}')
    assert key.startswith(f"fig:{TIME_BASIS}:"), (
        f"cache key {key!r} should start with fig:{TIME_BASIS}:")


def test_cache_l0_fingerprint_includes_time_basis():
    """L0 pinned fingerprint hash input includes TIME_BASIS.

    Slice is [:12] (matches existing cache.py:134 — do NOT shorten to [:8]).
    """
    import sys
    sys.path.insert(0, "btc_web")
    import cache  # noqa: E402
    from time_basis import TIME_BASIS  # noqa: E402
    from tab_defaults import _DEFAULTS_HASH  # noqa: E402
    expected_input = f"{TIME_BASIS}:{cache._MODEL_FP}:{_DEFAULTS_HASH}"
    expected_fp = hashlib.md5(expected_input.encode()).hexdigest()[:12]
    assert cache._L0_FINGERPRINT == expected_fp


def test_calendar_block_cache_keys_differ(monkeypatch):
    """Same params but different TIME_BASIS yield different cache keys."""
    import sys
    sys.path.insert(0, "btc_web")
    import cache  # noqa: E402
    monkeypatch.setattr(cache, "TIME_BASIS", "calendar", raising=False)
    cal = cache._cache_key("bub", '{"a": 1}')
    monkeypatch.setattr(cache, "TIME_BASIS", "block", raising=False)
    blk = cache._cache_key("bub", '{"a": 1}')
    assert cal != blk
    assert ":calendar:" in cal
    assert ":block:" in blk


def test_snapshot_fingerprint_changes_when_time_basis_changes(monkeypatch):
    """Reserving the TIME_BASIS slot in the snapshot fingerprint hash."""
    from btc_web import snapshot_defaults as sd
    from btc_web import time_basis as tb
    monkeypatch.setattr(tb, "TIME_BASIS", "calendar")
    monkeypatch.setattr(sd, "TIME_BASIS", "calendar", raising=False)
    cal_fp = sd._compute_snapshot_defaults_fingerprint()
    monkeypatch.setattr(tb, "TIME_BASIS", "block")
    monkeypatch.setattr(sd, "TIME_BASIS", "block", raising=False)
    blk_fp = sd._compute_snapshot_defaults_fingerprint()
    assert cal_fp != blk_fp
    assert len(cal_fp) == 8
    assert len(blk_fp) == 8


def test_snapshot_fingerprint_calendar_value_is_stable():
    """Calendar mode fingerprint is deterministic given the current registry."""
    from btc_web import snapshot_defaults as sd
    fp1 = sd._compute_snapshot_defaults_fingerprint()
    fp2 = sd._compute_snapshot_defaults_fingerprint()
    assert fp1 == fp2
