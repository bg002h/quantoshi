# L0 Persistent Cache — Design Spec

**Date:** 2026-03-27
**Branch:** `DefaultUpdateSanity`
**Goal:** Make the L0 (pinned) figure cache persistent across worker restarts via Redis, so prewarm only recomputes when the model or defaults actually change.

---

## Problem

`_prewarm_caches()` runs on every gunicorn worker boot (~15s), computing default figures for all 7 chart tabs. This happens even when nothing has changed — same model data, same defaults. On a 4-worker deploy, that's ~60s of redundant computation.

The L0 cache layer (`_pinned` dict in `_make_cached_builder`) was designed to hold prewarm results permanently in-process, but it clears on restart and is never populated from a persistent store.

---

## Solution

### Fingerprint

A combined hash that changes when either the model or defaults change:

```python
_L0_FINGERPRINT = md5(model_fp + defaults_hash)[:12]
```

- `model_fp` — existing `_MODEL_FP` from `cache.py` (md5 of pkl mtime + size)
- `defaults_hash` — hash of all 7 frozen `MappingProxyType` dicts in `tab_defaults.py`, computed once at import time

**`_DEFAULTS_HASH` in `tab_defaults.py`:**
```python
import hashlib

def _compute_defaults_hash() -> str:
    h = hashlib.md5()
    for d in (BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE, STACK, CITADEL):
        h.update(repr(sorted(d.items())).encode())
    return h.hexdigest()[:12]

_DEFAULTS_HASH = _compute_defaults_hash()
```

Using `repr(sorted(d.items()))` ensures deterministic ordering. The hash changes if any value in any frozen dict changes.

### Redis key format

```
l0:{fingerprint}:{tab}
```

Example: `l0:a3f8b2c1d4e5:bubble`

All L0 keys use a **7-day TTL**. Old entries from previous fingerprints expire naturally — no active cleanup needed.

### Boot sequence

```
Worker starts
  │
  ├─ Compute _L0_FINGERPRINT
  │
  ├─ Redis available?
  │    │
  │    ├─ YES: Check for all 7 l0:{fp}:{tab} keys
  │    │    │
  │    │    ├─ ALL PRESENT: Deserialize → pin in L0 dict → done (~100ms)
  │    │    │
  │    │    └─ ANY MISSING: Full prewarm → pin in L0 → write to Redis (7d TTL)
  │    │
  │    └─ NO: Full prewarm → pin in L0 memory only → set _l0_needs_flush = True
  │
  └─ Ready to serve
```

### Deferred flush

When Redis is unavailable at boot, L0 is populated in-memory but not persisted. A one-shot flush mechanism runs on the first figure request:

```python
_l0_needs_flush = False  # set True when Redis was down at boot

def _try_flush_l0():
    """One-shot: write in-memory L0 to Redis if it wasn't written at boot."""
    global _l0_needs_flush
    if not _l0_needs_flush:
        return
    try:
        # Write all pinned entries to Redis
        for tab, builder in _ALL_BUILDERS.items():
            for key, result in builder.pinned.items():
                _redis_set_l0(tab, key, result)
        _l0_needs_flush = False
    except Exception:
        pass  # Redis still down; next request will retry
```

Called inside each cached builder wrapper, before returning the result. Adds negligible overhead (one bool check per request when flush is not needed).

### Storage format

Plotly figures serialized via `fig.to_json()` / `plotly.io.from_json()` — same format as existing L2 cache. For builders that return `(fig, mc_result)` tuples, both components are stored as a JSON object `{"figure": ..., "mc_result": ...}`.

Total storage: ~2MB for all 7 tabs (one fingerprint set).

### What changes

**`btc_web/tab_defaults.py`:**
- Add `_compute_defaults_hash()` function
- Add `_DEFAULTS_HASH` module-level constant

**`btc_web/utils.py`:**
- Add `_L0_FINGERPRINT` computed from `_MODEL_FP` + `_DEFAULTS_HASH`
- Add `_l0_redis_key(tab)` helper
- Add `_load_l0_from_redis()` — checks Redis for all 7 tabs, returns dict of results or None
- Add `_store_l0_to_redis(tab, key, result)` — stores with 7-day TTL
- Modify `_make_cached_builder` wrapper to call `_try_flush_l0()` on first invocation
- Add `_ALL_BUILDERS` registry so deferred flush can iterate all builders

**`btc_web/app.py`:**
- Modify `_prewarm_caches()`:
  - First: try `_load_l0_from_redis()` — if all 7 tabs found, pin them and return early
  - Otherwise: compute as before, then pin results and write to Redis
  - If Redis unavailable: compute, pin in memory, set `_l0_needs_flush = True`

---

## Tests

### 1. Defaults hash changes when frozen dict changes

```python
def test_defaults_hash_changes():
    from tab_defaults import _compute_defaults_hash, BUBBLE
    h1 = _compute_defaults_hash()
    # Monkey-patch a value and recompute
    import types
    fake = types.MappingProxyType({**BUBBLE, "pt_alpha": 999})
    # (test helper that temporarily replaces BUBBLE)
    h2 = _compute_defaults_hash_with(fake)
    assert h1 != h2
```

### 2. L0 Redis round-trip

```python
def test_l0_redis_roundtrip():
    """Store a figure in Redis L0, load it back, verify it matches."""
    fig = go.Figure(data=[go.Scatter(x=[1,2], y=[3,4])])
    _store_l0_to_redis("bubble", fig)
    loaded = _load_l0_entry("bubble")
    assert loaded is not None
    assert len(loaded.data) == len(fig.data)
```

### 3. L0 skip prewarm when Redis has all entries

```python
def test_l0_skips_prewarm_when_cached(monkeypatch):
    """If Redis has all 7 L0 entries, prewarm should not compute figures."""
    # Pre-populate Redis with all 7 L0 keys
    # Monkeypatch figure builders to track calls
    # Call _prewarm_caches()
    # Assert builders were NOT called
```

### 4. Deferred flush fires when Redis was unavailable

```python
def test_deferred_flush():
    """L0 in memory + _l0_needs_flush=True → first request writes to Redis."""
    # Set _l0_needs_flush = True with pinned entries
    # Call a cached builder
    # Assert Redis now has the L0 keys
    # Assert _l0_needs_flush is False
```

### 5. Fallback to full prewarm when Redis is down

```python
def test_prewarm_fallback_no_redis():
    """When Redis is unavailable, prewarm runs fully and sets flush flag."""
    # Monkeypatch redis to raise ConnectionError
    # Call _prewarm_caches()
    # Assert L0 pinned dicts are populated
    # Assert _l0_needs_flush is True
```

---

## Not in scope

- Changing what figures are prewarmed (same 7 tabs as before)
- Changing L1 (LRU) or L2 (shared Redis) cache behavior
- MC overlay caching (separate system)
- Compression of stored figures (JSON is fine at ~2MB)
