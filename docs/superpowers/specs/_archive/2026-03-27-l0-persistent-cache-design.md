# L0 Persistent Cache — Design Spec

**Date:** 2026-03-27
**Branch:** `DefaultUpdateSanity`
**Goal:** Make the L0 (pinned) figure cache persistent across worker restarts via Redis, so prewarm only recomputes when the model or defaults actually change.

---

## Problem

`_prewarm_caches()` runs on every gunicorn worker boot (~15s), computing default figures for all chart tabs (8 entries: 2 bubble variants + heatmap + DCA + retire + supercharge + citadel). This happens even when nothing has changed — same model data, same defaults. On a 4-worker deploy, that's ~60s of redundant computation.

The L0 cache layer (`_pinned` dict in `_make_cached_builder`) was designed to hold prewarm results permanently in-process, but it clears on restart and **`_pin()` is never called** — prewarm currently populates L1 (`@lru_cache`) only. L0 is structurally present but functionally unused.

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

**Determinism constraint:** This works because all values in the frozen dicts are primitives and tuples — types with deterministic `repr()` output. If a `set` or `dict` value is ever added to a frozen dict, the hash may become non-deterministic across restarts. The `test_inner_collections_are_tuples` test in `test_defaults.py` enforces this constraint.

### Redis key format

```
l0:{fingerprint}:{prefix}:{params_hash}
```

Example: `l0:a3f8b2c1d4e5:bub:7f3a2b1c`

The `params_hash` is the same SHA-256 key used by L1/L2 — it distinguishes entries for the same tab with different params (e.g., bubble default vs bubble+PL overlay). All L0 keys use a **7-day TTL**. Old entries from previous fingerprints expire naturally.

**Prewarm produces 8 entries** (not 7): bubble default, bubble+PL, heatmap, DCA, retire, supercharge, citadel, plus the MC prewarm entries if Markov is available. The key format handles this naturally since each param set produces a unique `params_hash`.

### Boot sequence

```
Worker starts
  │
  ├─ Compute _L0_FINGERPRINT
  │
  ├─ Redis available?
  │    │
  │    ├─ YES: Check for all L0 keys matching l0:{fp}:*
  │    │    │
  │    │    ├─ ALL PRESENT: Deserialize → pin in L0 dict → done (~100ms)
  │    │    │
  │    │    └─ ANY MISSING: Full prewarm → pin in L0 via _pin() → write to Redis (7d TTL)
  │    │
  │    └─ NO: Full prewarm → pin in L0 via _pin() → set _l0_needs_flush = True
  │
  └─ Ready to serve
```

**Key detail:** Currently `_prewarm_caches()` calls `_get_*_fig(params)` which populates L1 (`@lru_cache`) but never calls `_pin()`. The implementation must change prewarm to explicitly call `builder.pin(key, result)` after each figure computation so results land in L0 (not just L1). The `_get_*_fig` wrappers return the figure but not the cache key, so prewarm must compute the key itself (via `_quantize_params` + JSON serialization, same as the cached wrapper).

**Multiple workers:** With 4 gunicorn workers booting simultaneously, all will check Redis and either all hit or all miss. On a miss, all 4 compute and write — redundant but harmless (identical results, last writer wins). A Redis lock could prevent this but adds complexity that isn't worth it for a 15s operation that only happens on model/defaults changes.

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

Total storage: ~2MB for all 8 entries (one fingerprint set).

### What changes

**`btc_web/tab_defaults.py`:**
- Add `_compute_defaults_hash()` function
- Add `_DEFAULTS_HASH` module-level constant

**`btc_web/utils.py`:**
- Add `_L0_FINGERPRINT` computed from `_MODEL_FP` + `_DEFAULTS_HASH`
- Add `_l0_redis_key(prefix, params_hash)` helper
- Add `_load_l0_from_redis()` — checks Redis for all L0 entries matching current fingerprint, returns dict or None
- Add `_store_l0_to_redis(prefix, params_hash, result)` — stores with 7-day TTL
- Modify `_make_cached_builder` wrapper to call `_try_flush_l0()` on first invocation
- Add `_ALL_BUILDERS` registry so deferred flush can iterate all builders

**`btc_web/app.py`:**
- Modify `_prewarm_caches()`:
  - First: try `_load_l0_from_redis()` — if all entries found, pin them via `builder.pin()` and return early
  - Otherwise: compute as before, **explicitly call `builder.pin(key, result)`** after each figure, write to Redis
  - If Redis unavailable: compute, pin in memory only, set `_l0_needs_flush = True`

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
    """If Redis has all L0 entries, prewarm should not compute figures."""
    # Pre-populate Redis with all L0 keys for current fingerprint
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

- Changing what figures are prewarmed (same 8 entries as before)
- Changing L1 (LRU) or L2 (shared Redis) cache behavior
- MC overlay caching (separate system)
- Compression of stored figures (JSON is fine at ~2MB)
