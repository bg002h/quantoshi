# L0 Persistent Cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the L0 (pinned) figure cache persistent across worker restarts via Redis, so prewarm only recomputes when the model or defaults actually change (~100ms boot instead of ~15s).

**Architecture:** Add a `_DEFAULTS_HASH` to `tab_defaults.py`, combine it with the existing `_MODEL_FP` to form a fingerprint. On boot, check Redis for L0 entries matching the fingerprint; if found, deserialize and pin — skipping all figure computation. If not found, compute as before but pin results and store in Redis with 7-day TTL. Deferred flush handles Redis-unavailable-at-boot.

**Tech Stack:** Python 3, Redis, `hashlib`, `plotly.io`, existing `cache.py` infrastructure

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `btc_web/tab_defaults.py` | Modify | Add `_DEFAULTS_HASH` constant |
| `btc_web/cache.py` | Modify | Add L0 Redis helpers (`_l0_key`, `get_l0`, `set_l0`, `load_all_l0`) + `_L0_FINGERPRINT` |
| `btc_web/utils.py` | Modify | Wire `_pin()` into `_get_*_fig` wrappers; add `_l0_needs_flush` + `_try_flush_l0()` |
| `btc_web/app.py` | Modify | Rewrite `_prewarm_caches()` to check Redis L0 first |
| `btc_web/test_defaults.py` | Modify | Add fingerprint + L0 tests |

---

## Task 1: Add `_DEFAULTS_HASH` to `tab_defaults.py`

**Files:**
- Modify: `btc_web/tab_defaults.py`
- Test: `btc_web/test_defaults.py`

- [ ] **Step 1: Write the failing test**

Append to `btc_web/test_defaults.py`:

```python
def test_defaults_hash_is_stable():
    """Hash is deterministic across calls."""
    from tab_defaults import _DEFAULTS_HASH, _compute_defaults_hash
    assert isinstance(_DEFAULTS_HASH, str)
    assert len(_DEFAULTS_HASH) == 12
    assert _compute_defaults_hash() == _DEFAULTS_HASH  # stable


def test_defaults_hash_changes_on_value_change():
    """Hash changes when a frozen dict value changes."""
    from tab_defaults import _compute_defaults_hash, BUBBLE
    import types
    original = _compute_defaults_hash()
    # Compute with a modified BUBBLE
    fake = types.MappingProxyType({**BUBBLE, "pt_alpha": 999.0})
    import hashlib
    h = hashlib.md5()
    for d in (fake,):  # just one dict is enough to prove it changes
        h.update(repr(sorted(d.items())).encode())
    partial = h.hexdigest()[:12]
    assert partial != original[:12]  # different input → different hash
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py::test_defaults_hash_is_stable -v`
Expected: ImportError for `_DEFAULTS_HASH`.

- [ ] **Step 3: Add `_compute_defaults_hash` and `_DEFAULTS_HASH` to `tab_defaults.py`**

Append after the `citadel_defaults()` function:

```python
# ── Defaults fingerprint (for L0 cache invalidation) ────────────────────────
import hashlib as _hashlib

def _compute_defaults_hash() -> str:
    """Hash all frozen dicts. Changes when any default value changes.

    Uses repr(sorted(items)) — deterministic for primitives and tuples.
    If a set or dict value is ever added, this may become non-deterministic.
    The test_inner_collections_are_tuples test guards against this.
    """
    h = _hashlib.md5()
    for d in (BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE, STACK, CITADEL):
        h.update(repr(sorted(d.items())).encode())
    return h.hexdigest()[:12]

_DEFAULTS_HASH = _compute_defaults_hash()
```

- [ ] **Step 4: Run tests**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py::test_defaults_hash_is_stable btc_web/test_defaults.py::test_defaults_hash_changes_on_value_change -v`
Expected: Both PASS.

- [ ] **Step 5: Commit**

```bash
git add btc_web/tab_defaults.py btc_web/test_defaults.py
git commit -m "feat: add _DEFAULTS_HASH for L0 cache fingerprinting"
```

---

## Task 2: Add L0 Redis Helpers to `cache.py`

**Files:**
- Modify: `btc_web/cache.py`
- Test: `btc_web/test_defaults.py`

- [ ] **Step 1: Write the failing test**

Append to `btc_web/test_defaults.py`:

```python
def test_l0_fingerprint_combines_model_and_defaults():
    """L0 fingerprint includes both model and defaults hashes."""
    from cache import _L0_FINGERPRINT, _MODEL_FP
    from tab_defaults import _DEFAULTS_HASH
    assert _MODEL_FP in _L0_FINGERPRINT or _DEFAULTS_HASH in _L0_FINGERPRINT
    assert len(_L0_FINGERPRINT) == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py::test_l0_fingerprint_combines_model_and_defaults -v`
Expected: ImportError for `_L0_FINGERPRINT`.

- [ ] **Step 3: Add L0 helpers to `cache.py`**

Append to `btc_web/cache.py` after the `redis_available()` function:

```python
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
```

- [ ] **Step 4: Run test**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py::test_l0_fingerprint_combines_model_and_defaults -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add btc_web/cache.py btc_web/test_defaults.py
git commit -m "feat: add L0 Redis helpers (get_l0, set_l0, scan_l0_keys, fingerprint)"
```

---

## Task 3: Wire `_pin()` into Prewarm + Add L0 Redis Persistence

**Files:**
- Modify: `btc_web/utils.py`
- Modify: `btc_web/app.py`

This is the core task. Prewarm must:
1. Compute the JSON key for each default figure (same key format as L1)
2. Check Redis for L0 entries → if all found, pin and skip
3. If any missing, compute figures, pin in L0, store in Redis

- [ ] **Step 1: Add `_prewarm_and_pin` helper to `utils.py`**

This helper computes a figure, pins it in L0, and optionally stores in Redis. Add after `_get_citadel_fig`:

```python
# ── L0 prewarm helpers ───────────────────────────────────────────────────────
import hashlib
import plotly.io as _pio

_l0_needs_flush = False  # set True when Redis was down at boot

# Map prefix → (cache_fn, get_fn)  for L0 operations
_L0_BUILDERS = {
    "bub":  (_cached_bubble_fig,  _get_bubble_fig),
    "hm":   (_cached_heatmap_fig, _get_heatmap_fig),
    "dca":  (_cached_dca_fig,     _get_dca_fig),
    "ret":  (_cached_retire_fig,  _get_retire_fig),
    "sc":   (_cached_supercharge_fig, _get_supercharge_fig),
    "cp":   (_cached_citadel_fig, _get_citadel_fig),
}


def _compute_cache_key(prefix: str, p: dict) -> str:
    """Compute the JSON cache key for a param dict, same as _get_*_fig would.

    NOTE: Only correct for prewarm defaults (no mc_* keys). For params with
    mc_* keys, the strip order differs from _get_mc_or_cached. This is fine
    because L0 only stores prewarm defaults.
    """
    from datetime import date as _date
    p_q = _quantize_params(p)
    # Strip mc_* params for non-MC entries (mirrors _get_mc_or_cached behavior)
    if not p_q.get("mc_enabled"):
        p_q = {k: v for k, v in p_q.items() if not k.startswith("mc_")}
    p_q.pop("mc_cached", None)
    p_q.pop("mc_free_tier", None)
    if prefix == "bub":
        p_q["_day"] = str(_date.today())
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
        for prefix, (cache_fn, _) in _L0_BUILDERS.items():
            for key, result in cache_fn.pinned.items():
                set_l0(prefix, hashlib.sha256(key.encode()).hexdigest()[:16],
                       _serialize_result(result))
        _l0_needs_flush = False
        logger.info("L0 deferred flush: wrote %d entries to Redis",
                    sum(len(c.pinned) for _, (c, _) in _L0_BUILDERS.items()))
    except Exception as e:
        logger.debug("L0 deferred flush failed: %s", e)
```

- [ ] **Step 2: Add `_try_flush_l0()` call to `_get_*_fig` wrappers (NOT inside `@lru_cache`)**

**IMPORTANT:** `_try_flush_l0()` must NOT be placed inside the `@lru_cache`-decorated `_cached()` function. Prewarm populates L1 via `_cached()`, so user requests with default params hit L1 directly — the function body never executes, and the flush would never fire.

Instead, add the call to each `_get_*_fig` wrapper function, which runs on every request BEFORE the L1 lookup:

```python
def _get_bubble_fig(p: dict):
    _try_flush_l0()  # one-shot deferred flush (negligible: one bool check)
    p = _quantize_params(p)
    p['_day'] = str(date.today())
    return _cached_bubble_fig(json.dumps(p, sort_keys=True, default=str))
```

Add the same `_try_flush_l0()` call as the first line of: `_get_bubble_fig`, `_get_dca_fig` (via `_get_mc_or_cached`), `_get_retire_fig`, `_get_supercharge_fig`, `_get_heatmap_fig`, `_get_citadel_fig`.

Simplest approach: add it to `_get_mc_or_cached` (covers DCA/Retire/SC) + `_get_bubble_fig` + `_get_heatmap_fig` + `_get_citadel_fig` — 4 call sites total.

Do NOT modify `_make_cached_builder` or the `_cached` inner function.

- [ ] **Step 3: Rewrite `_prewarm_caches()` in `app.py`**

Replace the current `_prewarm_caches()` with:

```python
def _prewarm_caches():
    """Pre-warm L0 caches. Tries Redis first; falls back to full compute."""
    import time as _time
    from tab_defaults import (bubble_defaults, heatmap_defaults, dca_defaults,
                              retire_defaults, supercharge_defaults, citadel_defaults)
    from utils import (_compute_cache_key, _serialize_result, _deserialize_result,
                       _L0_BUILDERS, _get_bubble_fig, _get_heatmap_fig,
                       _get_dca_fig, _get_retire_fig, _get_supercharge_fig,
                       _get_citadel_fig)
    import utils as _utils

    t0 = _time.time()

    # Build the list of (prefix, params_dict) to prewarm
    _entries = []

    bub = bubble_defaults()
    _entries.append(("bub", bub))

    bub_pl = bubble_defaults()
    bub_pl["active_models"] = ["bub", "pl"]
    _entries.append(("bub", bub_pl))

    _entries.append(("dca", dca_defaults()))
    _entries.append(("ret", retire_defaults()))

    sc = supercharge_defaults()
    sc["selected_qs"] = [q for q in [0.001, 0.10] if q in _app_ctx.DEFAULT_MODEL.fits]
    _entries.append(("sc", sc))

    cp = citadel_defaults()
    cp["selected_qs"] = [0.01, 0.10, 0.25]
    _entries.append(("cp", cp))

    hm = heatmap_defaults()
    hm["entry_q"] = _app_ctx._HM_ENTRY_Q_DEFAULT
    _entries.append(("hm", hm))

    # Compute cache keys
    keyed = [(pfx, p, _compute_cache_key(pfx, dict(p))) for pfx, p in _entries]

    # Try loading from Redis L0
    try:
        from cache import get_l0, set_l0, redis_available
        if redis_available():
            import hashlib
            hits = []
            for pfx, p, json_key in keyed:
                params_hash = hashlib.sha256(json_key.encode()).hexdigest()[:16]
                raw = get_l0(pfx, params_hash)
                if raw:
                    hits.append((pfx, json_key, raw))
                else:
                    break  # any miss → full recompute

            if len(hits) == len(keyed):
                # All found in Redis — pin in L0 and skip computation
                for pfx, json_key, raw in hits:
                    cache_fn = _L0_BUILDERS[pfx][0]
                    result = _deserialize_result(raw)
                    cache_fn.pin(json_key, result)
                logger.info("L0 warm from Redis: %d entries in %.1fs",
                            len(hits), _time.time() - t0)
                return
    except Exception as e:
        logger.debug("L0 Redis check failed: %s", e)

    # Redis miss or unavailable — full compute
    logger.info("L0 computing %d entries...", len(keyed))
    redis_ok = True
    for pfx, p, json_key in keyed:
        cache_fn = _L0_BUILDERS[pfx][0]
        get_fn = _L0_BUILDERS[pfx][1]
        result = get_fn(dict(p))  # compute figure (also populates L1)
        cache_fn.pin(json_key, result)  # pin in L0

        # Store in Redis L0
        try:
            from cache import set_l0, redis_available
            import hashlib
            if redis_ok and redis_available():
                params_hash = hashlib.sha256(json_key.encode()).hexdigest()[:16]
                set_l0(pfx, params_hash, _serialize_result(result))
            else:
                redis_ok = False
        except Exception:
            redis_ok = False

    if not redis_ok:
        _utils._l0_needs_flush = True
        logger.info("L0 compute done (Redis unavailable — deferred flush enabled)")
    else:
        logger.info("L0 compute + Redis store: %d entries in %.1fs",
                    len(keyed), _time.time() - t0)
```

- [ ] **Step 4: Run syntax check**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -c "from tab_defaults import _DEFAULTS_HASH; from cache import _L0_FINGERPRINT; print('OK:', _L0_FINGERPRINT)"`
Expected: `OK: <12-char hex>`

- [ ] **Step 5: Run full test suite**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py btc_web/test_web.py -v --timeout=180 -x -q 2>&1 | tail -20`
Expected: All existing tests pass + new tests pass.

- [ ] **Step 6: Commit**

```bash
git add btc_web/utils.py btc_web/app.py
git commit -m "feat: L0 persistent cache — Redis-backed prewarm with fingerprint invalidation"
```

---

## Task 4: Add L0 Integration Tests

**Files:**
- Modify: `btc_web/test_defaults.py`

- [ ] **Step 1: Add L0 round-trip test**

```python
def test_l0_serialize_deserialize_roundtrip():
    """Figure survives serialize → deserialize."""
    import plotly.graph_objects as go
    from utils import _serialize_result, _deserialize_result
    fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4], name="test")])
    raw = _serialize_result(fig)
    loaded = _deserialize_result(raw)
    assert len(loaded.data) == 1
    assert loaded.data[0].name == "test"


def test_l0_serialize_tuple_roundtrip():
    """(fig, mc_result) tuple survives roundtrip."""
    import plotly.graph_objects as go
    from utils import _serialize_result, _deserialize_result
    fig = go.Figure(data=[go.Scatter(x=[1], y=[2])])
    result = (fig, {"some": "data"})
    raw = _serialize_result(result)
    loaded = _deserialize_result(raw)
    assert isinstance(loaded, tuple)
    assert len(loaded[0].data) == 1
    assert loaded[1]["some"] == "data"


def test_compute_cache_key_deterministic():
    """Same params → same key."""
    from utils import _compute_cache_key
    from tab_defaults import bubble_defaults
    p1 = bubble_defaults()
    p2 = bubble_defaults()
    assert _compute_cache_key("bub", p1) == _compute_cache_key("bub", p2)


def test_compute_cache_key_changes_with_params():
    """Different params → different key."""
    from utils import _compute_cache_key
    from tab_defaults import bubble_defaults
    p1 = bubble_defaults()
    p2 = bubble_defaults()
    p2["pt_alpha"] = 0.99
    assert _compute_cache_key("bub", p1) != _compute_cache_key("bub", p2)
```

- [ ] **Step 2: Run tests**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py -k "l0_serialize or compute_cache_key" -v`
Expected: All 4 PASS.

- [ ] **Step 3: Commit**

```bash
git add btc_web/test_defaults.py
git commit -m "test: add L0 serialize/deserialize roundtrip + cache key tests"
```

---

## Task 5: Verify End-to-End + Manual Smoke Test

**Files:** (verification only, no code changes unless fixing bugs)

- [ ] **Step 1: Run full test suite**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py btc_web/test_web.py -v --timeout=180 -q 2>&1 | tail -10`
Expected: All tests pass (pre-existing failures excluded).

- [ ] **Step 2: Verify L0 populates Redis on first boot**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" TESTING=1 btc_venv/bin/python3 -c "
from btc_web.app import app
from cache import scan_l0_keys
keys = scan_l0_keys()
print(f'L0 keys in Redis: {len(keys)}')
for k in sorted(keys):
    print(f'  {k}')
"`
Expected: 7-8 L0 keys printed (one per prewarm entry).

- [ ] **Step 3: Verify second boot skips prewarm**

Run the same command again and check log output for "L0 warm from Redis" (fast path) vs "L0 computing" (slow path). Second run should show fast path.

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: resolve any L0 integration issues"
```

(Skip if no fixes needed.)
