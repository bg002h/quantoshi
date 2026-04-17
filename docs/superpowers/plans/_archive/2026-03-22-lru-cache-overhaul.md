# LRU Cache Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix cache key bugs and right-size the LRU figure cache for the expanded multi-model architecture (7 models: bub, qr, pl, lppl, exp, s2f, ef).

**Architecture:** Normalize list-valued cache key fields (`active_models`, `selected_qs`, `delays`) by sorting before serialization so ["pl","bub"] and ["bub","pl"] hit the same cache slot. Add `active_models` to the no-quantize exemption list. Increase maxsize from 32→64 per tab. Fix all prewarm param dicts to match the real callback shapes. Add cache hit-rate logging for observability.

**Tech Stack:** Python stdlib (`functools.lru_cache`, `json`, `logging`), pytest

---

## Problem Analysis

### Bug 1: Unsorted list keys cause cache misses

`active_models` is a list in the cache key. `json.dumps(..., sort_keys=True)` only sorts **dict keys**, not list values. So these produce different cache keys:

```python
{"active_models": ["bub", "pl"]}   # user clicks Bubble then PL
{"active_models": ["pl", "bub"]}   # user clicks PL then Bubble
```

Same problem applies to `selected_qs` (checklist click order) and `delays` (5-element list). Each unsorted permutation wastes a cache slot on an identical figure. Note: `delays` are already `sorted(set(...))` inside `build_supercharge_figure` (line 69 of `figures/supercharge.py`), so sorting in `_quantize_params` is safe and aligns cache keys with what the builder actually computes.

### Bug 2: `active_models` should be exempt from quantize

`active_models` contains string model keys like `["bub", "pl"]`. The `_quantize_params` list branch applies `_q3()` to each element — strings pass through safely (the `isinstance(x, float) and x != 0` guard skips them), but `active_models` should be explicitly exempt for clarity and to prevent any future breakage.

### Issue 3: maxsize=32 may be too small

With 7 models, `active_models` can have up to 2^7 = 128 sorted combinations. Combined with 3 palettes, 2 scale modes, and varying quantile selections, 32 slots per tab evicts useful entries quickly.

### Issue 4: Prewarm param dicts are incomplete

The existing `_prewarm_caches()` entries are missing fields that the real callbacks always pass:
- **Bubble** prewarm omits: `show_ucl`, `legend_pos`, `scanner_lines`
- **DCA** prewarm omits: `inflation`, `annotate`, `legend_pos`, `show_qr`, `show_mc`, `active_models`
- **Retire** prewarm omits: `legend_pos`, `show_qr`, `show_mc`, `active_models`
- **Supercharge** prewarm omits: `legend_pos`, `show_qr`, `show_mc`, `active_models`

Because the cache key is `json.dumps(p, sort_keys=True)`, any missing key means the prewarm produces a key that will **never match** a real callback request. The prewarm runs successfully but provides **zero cache benefit**.

Additionally, no heatmap prewarm exists at all.

### Issue 5: No observability

No logging of cache hit rates. When maxsize is exceeded or keys diverge, there's no way to diagnose it.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `btc_web/utils.py` | Modify | Normalize list keys, add `active_models` exemption, increase maxsize, add hit-rate logging |
| `btc_web/app.py` | Modify | Fix all prewarm param dicts to match real callbacks, add heatmap prewarm, add cache stats logging |
| `btc_web/test_web.py` | Modify | Add tests for list normalization, cache key stability, maxsize, quantize exemptions |

**Reference file (read-only):** `btc_web/callbacks/charts.py` — authoritative source for real callback param shapes.

---

### Task 1: Normalize list-valued cache keys in `_quantize_params`

**Files:**
- Modify: `btc_web/utils.py:30-44`
- Test: `btc_web/test_web.py`

**Context:** `_quantize_params()` processes the params dict before it becomes a cache key. Lists like `active_models`, `selected_qs`, `delays` are order-dependent in JSON serialization but order-independent for figure output. Sorting them ensures identical figures always produce identical cache keys.

- [ ] **Step 1: Write failing tests for list normalization**

Add to `TestQuantizeParams` class in `test_web.py`:

```python
def test_active_models_sorted(self):
    """active_models order should not affect cache key."""
    out1 = _quantize_params({"active_models": ["pl", "bub", "s2f"]})
    out2 = _quantize_params({"active_models": ["bub", "pl", "s2f"]})
    assert out1["active_models"] == out2["active_models"]
    assert out1["active_models"] == ["bub", "pl", "s2f"]

def test_selected_qs_sorted(self):
    """selected_qs order should not affect cache key."""
    out1 = _quantize_params({"selected_qs": [0.5, 0.1, 0.01]})
    out2 = _quantize_params({"selected_qs": [0.01, 0.1, 0.5]})
    assert out1["selected_qs"] == out2["selected_qs"]
    assert out1["selected_qs"] == [0.01, 0.1, 0.5]

def test_delays_sorted(self):
    """delays order should not affect cache key."""
    out1 = _quantize_params({"delays": [2.0, 0.0, 1.0]})
    out2 = _quantize_params({"delays": [0.0, 1.0, 2.0]})
    assert out1["delays"] == out2["delays"]
    assert out1["delays"] == [0.0, 1.0, 2.0]

def test_active_models_exempt_from_quantize(self):
    """active_models contains strings, must not be quantized."""
    out = _quantize_params({"active_models": ["bub", "pl"]})
    assert out["active_models"] == ["bub", "pl"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestQuantizeParams::test_active_models_sorted btc_web/test_web.py::TestQuantizeParams::test_delays_sorted -v -x`
Expected: FAIL — lists are not currently sorted.

- [ ] **Step 3: Add list normalization and `active_models` exemption**

In `btc_web/utils.py`, update the `_NO_QUANTIZE_KEYS` set and add a `_SORT_LIST_KEYS` set, then modify `_quantize_params`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestQuantizeParams -v`
Expected: All PASS including new tests.

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py -v`
Expected: All 435+ tests PASS.

- [ ] **Step 6: Commit**

```bash
git add btc_web/utils.py btc_web/test_web.py
git commit -m "fix: normalize list-valued cache keys for stable LRU hits

active_models/selected_qs/delays are now sorted before serialization
so permutation order doesn't waste cache slots. active_models added
to _NO_QUANTIZE_KEYS exemption."
```

---

### Task 2: Increase LRU maxsize from 32 to 64

**Files:**
- Modify: `btc_web/utils.py:46-62`
- Test: `btc_web/test_web.py`

**Context:** With 7 models and 3 palettes, the combinatorial space for `active_models` alone is 128 sorted subsets. maxsize=32 causes frequent eviction of useful entries. Doubling to 64 is conservative — memory impact is ~32 MB/worker (from ~16 MB) which is well within budget on the Hetzner VPS (the MC cache alone uses 834 MB).

- [ ] **Step 1: Write tests for the new maxsize**

Add a new test class in `test_web.py`:

```python
@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestCacheMaxsize:
    def test_bubble_cache_maxsize(self):
        from utils import _cached_bubble_fig
        assert _cached_bubble_fig.cache_info().maxsize == 64

    def test_heatmap_cache_maxsize(self):
        from utils import _cached_heatmap_fig
        assert _cached_heatmap_fig.cache_info().maxsize == 64

    def test_mc_heatmap_cache_maxsize(self):
        from utils import _cached_mc_heatmap_fig
        assert _cached_mc_heatmap_fig.cache_info().maxsize == 64
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCacheMaxsize -v`
Expected: FAIL — current maxsize is 32.

- [ ] **Step 3: Update maxsize in `_make_cached_builder`**

In `btc_web/utils.py`, change:

```python
def _make_cached_builder(builder_fn, maxsize=64):
```

Also update the comment:

```python
# ── LRU figure caches (maxsize=64 per tab, ~32 MB/worker) ─────────────────────
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCacheMaxsize -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add btc_web/utils.py btc_web/test_web.py
git commit -m "perf: increase LRU maxsize 32→64 for multi-model combinatorics

7 models × 3 palettes × scale modes means 32 slots evict too
aggressively. 64 doubles memory to ~32 MB/worker, well within budget."
```

---

### Task 3: Add cache hit-rate logging

**Files:**
- Modify: `btc_web/utils.py`
- Modify: `btc_web/app.py` (after prewarm)

**Context:** Without observability, cache sizing is guesswork. Log `cache_info()` stats so we can right-size maxsize based on real traffic patterns.

- [ ] **Step 1: Add a `_log_cache_stats()` helper in `utils.py`**

After the cache creation block (after line 62), add:

```python
_ALL_CACHES = {
    "bubble": _cached_bubble_fig,
    "heatmap": _cached_heatmap_fig,
    "dca": _cached_dca_fig,
    "retire": _cached_retire_fig,
    "supercharge": _cached_supercharge_fig,
    "mc_heatmap": _cached_mc_heatmap_fig,
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
```

- [ ] **Step 2: Call `_log_cache_stats()` after prewarm in `app.py`**

After the `_prewarm_caches()` call in `app.py`, add:

```python
from utils import _log_cache_stats
_log_cache_stats()
```

- [ ] **Step 3: Verify stats appear in dev server log**

Run: `DEV=1 bash run_web.sh` and check `/tmp/quantoshi_dev.log` for `cache/` lines.
Expected: 6 lines showing hits=0, misses=N, size=N (from prewarm).

- [ ] **Step 4: Commit**

```bash
git add btc_web/utils.py btc_web/app.py
git commit -m "ops: add LRU cache hit-rate logging for observability

_log_cache_stats() reports hits/misses/size/rate per tab cache.
Called after prewarm; can be called periodically for production
monitoring."
```

---

### Task 4: Fix all existing prewarm param dicts to match real callbacks

**Files:**
- Modify: `btc_web/app.py:194-266`
- Reference: `btc_web/callbacks/charts.py` (read-only — the authoritative param shapes)

**Context:** The existing prewarm entries omit fields that the real callbacks always include. Since the cache key is the full JSON-serialized param dict, any missing field means the prewarm key never matches a real request. This task fixes all existing prewarm entries before adding new ones.

**IMPORTANT:** Read `callbacks/charts.py` lines 64–92 (bubble), 416–448 (DCA), 515–535 (retire), 625–649 (supercharge) to get the exact param dict shapes. The dicts below are derived from those callbacks.

- [ ] **Step 1: Fix existing bubble prewarm**

Add the 3 missing fields to the existing bubble prewarm dict (lines 198–212 of `app.py`):

```python
_get_bubble_fig(dict(
    selected_qs = [0.5],
    shade=True, show_ols=False, show_ucl=False,       # ← added show_ucl
    show_data=True, show_today=True,
    show_legend=False, minor_grid=False,
    show_comp=True, show_sup=False,
    xscale="log", yscale="log",
    xmin=2012, xmax=yr_now + 4,
    ymin=0.01, ymax=1e7,
    n_future=3, pt_size=3, pt_alpha=0.3,
    stack=0, show_stack=False, use_lots=False, lots=[],
    legend_pos="outside",                              # ← added
    comp_color="#FFD700", comp_lw=2.0,
    sup_color="#888888", sup_lw=1.5,
    active_models=["bub"],
    palette="default",
    scanner_lines=[],                                  # ← added
))
```

- [ ] **Step 2: Fix existing DCA prewarm**

Add missing fields to match `callbacks/charts.py` lines 416–448:

```python
_get_dca_fig(dict(
    start_stack=0, use_lots=False,
    amount=100.0, freq="Monthly",
    inflation=0.0,                                     # ← added
    start_yr=yr_now, end_yr=yr_now + 10,
    disp_mode="btc", log_y=False,
    annotate=False,                                    # ← added
    show_today=False,
    show_legend=False,
    legend_pos="outside",                              # ← added
    minor_grid=False,
    selected_qs=[0.50], lots=[],
    sc_enabled=False, sc_loan_amount=0,
    sc_rate=_app_ctx.SC_DEFAULT_RATE,
    sc_loan_type="interest_only", sc_term_months=48.0,
    sc_repeats=0, sc_rollover=False,
    sc_entry_mode="live",
    sc_custom_price=float(_app_ctx.SC_DEFAULT_PRICE),
    sc_tax_rate=0.33, sc_live_price=None,
    show_qr=True,                                      # ← added (default: "qr" in model_show)
    show_mc=False,                                     # ← added
    active_models=[],                                  # ← added (empty: qr is a sentinel, not a model)
    palette="default",                                 # ← added
))
```

- [ ] **Step 3: Fix existing Retire prewarm**

Add missing fields to match `callbacks/charts.py` lines 515–535:

```python
_get_retire_fig(dict(
    start_stack=1.0, use_lots=False,
    wd_amount=5000.0, freq="Monthly",
    start_yr=2031, end_yr=2075,
    inflation=4.0, disp_mode="btc",
    log_y=True, annotate=True,
    show_legend=False,
    legend_pos="outside",                              # ← added
    minor_grid=True,
    selected_qs=[0.01, 0.10, 0.25],
    lots=[],
    show_qr=True,                                      # ← added
    show_mc=False,                                     # ← added
    active_models=[],                                  # ← added
    palette="default",                                 # ← added
))
```

- [ ] **Step 4: Fix existing Supercharge prewarm**

Add missing fields to match `callbacks/charts.py` lines 625–649:

```python
_get_supercharge_fig(dict(
    mode         = "a",
    start_stack  = 1.0,
    start_yr     = 2033,
    delays       = [0.0, 0.0, 0.0, 1.0, 2.0],
    freq         = "Annually",
    inflation    = 4.0,
    selected_qs  = [q for q in [0.001, 0.10] if q in _app_ctx.DEFAULT_MODEL.fits],
    chart_layout = 2,
    display_q    = _nearest_quantile(0.05, _app_ctx._ALL_QS),
    wd_amount    = 5000,
    end_yr       = 2075,
    disp_mode    = "usd",
    log_y        = True,
    annotate     = True,
    show_legend  = False,
    legend_pos   = "outside",                          # ← added
    minor_grid   = True,
    target_yr    = 2060,
    lots         = [],
    use_lots     = False,
    show_qr      = True,                               # ← added
    show_mc      = False,                              # ← added
    active_models = [],                                # ← added
    palette      = "default",                          # ← added
))
```

- [ ] **Step 5: Verify prewarm runs without error**

Run: `DEV=1 bash run_web.sh` and check log.
Expected: No errors, cache stats show size=1 per tab.

- [ ] **Step 6: Commit**

```bash
git add btc_web/app.py
git commit -m "fix: align prewarm param dicts with real callback shapes

Existing prewarm entries were missing fields (show_ucl, legend_pos,
scanner_lines, show_qr, show_mc, active_models, palette, etc.)
causing cache keys to never match real traffic. Now every prewarm
dict matches the exact shape passed by callbacks/charts.py."
```

---

### Task 5: Add multi-model bubble prewarm and heatmap prewarm

**Files:**
- Modify: `btc_web/app.py` (inside `_prewarm_caches()`)

**Context:** Most users view the bubble tab first then toggle PL/S2F overlays. The heatmap has no prewarm at all. Add entries for the most common model combinations.

- [ ] **Step 1: Add bubble+PL prewarm entry**

After the existing bubble prewarm (now fixed in Task 4), add:

```python
# Bubble with PL overlay (common toggle)
_get_bubble_fig(dict(
    selected_qs = [0.5],
    shade=True, show_ols=False, show_ucl=False,
    show_data=True, show_today=True,
    show_legend=False, minor_grid=False,
    show_comp=True, show_sup=False,
    xscale="log", yscale="log",
    xmin=2012, xmax=yr_now + 4,
    ymin=0.01, ymax=1e7,
    n_future=3, pt_size=3, pt_alpha=0.3,
    stack=0, show_stack=False, use_lots=False, lots=[],
    legend_pos="outside",
    comp_color="#FFD700", comp_lw=2.0,
    sup_color="#888888", sup_lw=1.5,
    active_models=["bub", "pl"],
    palette="default",
    scanner_lines=[],
))
```

- [ ] **Step 2: Add heatmap prewarm entries**

The heatmap callback (`callbacks/charts.py` lines 266–289) passes these keys: `entry_yr`, `entry_q`, `live_price`, `exit_yr_lo`, `exit_yr_hi`, `exit_qs`, `color_mode`, `b1`, `b2`, `c_lo`, `c_mid1`, `c_mid2`, `c_hi`, `n_disc`, `vfmt`, `cell_font_size`, `show_colorbar`, `stack`, `use_lots`, `lots`, `active_models`, `palette`. Plus `hm_model` is added via `dict(shared_params, hm_model=hm_model)` on line 321.

The heatmap model defaults are stored in `_app_ctx.M.CAGR_SEG_*` and `_app_ctx.M.CAGR_GRAD_STEPS`. Entry percentile default is `_app_ctx._HM_ENTRY_Q_DEFAULT`.

```python
# Heatmap (default: bubble model, current year entry)
_hm_q = _app_ctx._HM_ENTRY_Q_DEFAULT
_get_heatmap_fig(dict(
    entry_yr     = yr_now,
    entry_q      = _hm_q,
    live_price   = None,  # live_price is None when first loading (no ticker yet)
    exit_yr_lo   = yr_now,
    exit_yr_hi   = yr_now + 10,
    exit_qs      = [],  # default: no exit quantiles selected
    color_mode   = 0,
    b1           = float(_app_ctx.M.CAGR_SEG_B1),
    b2           = float(_app_ctx.M.CAGR_SEG_B2),
    c_lo         = _app_ctx.M.CAGR_SEG_C_LO,
    c_mid1       = _app_ctx.M.CAGR_SEG_C_MID1,
    c_mid2       = _app_ctx.M.CAGR_SEG_C_MID2,
    c_hi         = _app_ctx.M.CAGR_SEG_C_HI,
    n_disc       = _app_ctx.M.CAGR_GRAD_STEPS,
    vfmt         = "cagr",
    cell_font_size = 9,
    show_colorbar = False,
    stack        = 0,
    use_lots     = False,
    lots         = [],
    hm_model     = "bub",
    active_models = [],
    palette      = "default",
))
```

- [ ] **Step 3: Verify prewarm runs without error**

Run: `DEV=1 bash run_web.sh` and check cache stats.
Expected: Bubble shows size=2, heatmap shows size=1, others size=1.

- [ ] **Step 4: Commit**

```bash
git add btc_web/app.py
git commit -m "perf: add bubble+PL and heatmap prewarm entries

First PL overlay toggle and first heatmap load now instant."
```

---

### Task 6: Final integration test and cleanup

**Files:**
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Run the full test suite**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py -v`
Expected: All tests PASS with no regressions.

- [ ] **Step 2: Start dev server and manually test model toggles**

Run: `DEV=1 bash run_web.sh`
Manual checks:
1. Load bubble tab — should be instant (prewarmed)
2. Toggle PL overlay — should be instant (prewarmed)
3. Toggle S2F overlay — first load warm, second instant
4. Switch to heatmap — should be instant
5. Check `/tmp/quantoshi_dev.log` for `cache/` lines after a few interactions

- [ ] **Step 3: Verify cache stats show reasonable hit rates**

After manual testing, check logs for cache stats and verify prewarmed entries are actually being hit (hits > 0 for bubble and heatmap).

- [ ] **Step 4: Commit any final adjustments**

```bash
git add -A
git commit -m "test: verify LRU cache overhaul with integration tests"
```

---

## Summary of Changes

| Change | Impact | Risk |
|--------|--------|------|
| Sort list keys in `_quantize_params` | Eliminates permutation-based cache misses | Low — sorted order is deterministic; `delays` already sorted in builder |
| Add `active_models` to `_NO_QUANTIZE_KEYS` | Prevents potential issues on string model keys | None — strings already passed through |
| maxsize 32→64 | ~16 MB more memory per worker | Low — VPS has ample RAM |
| Cache stats logging | Enables data-driven maxsize tuning | None — info-level logging |
| Fix existing prewarm param shapes | Prewarm actually hits on real traffic (was broken before) | Low — just adding missing fields to match callbacks |
| Multi-model + heatmap prewarm | Faster first load for common views | Low — adds ~1s to startup |
