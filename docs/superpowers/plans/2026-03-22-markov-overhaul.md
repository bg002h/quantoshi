# Markov MC Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Markov MC simulation model-aware (6 models), redesign the cache (~828 MB), and make the entire cache free tier.

**Architecture:** Change `markov.py` functions to accept a `model` object instead of `qr_fits` dict. Route through `mc_overlay.py` via `_resolve_model()`. Regenerate cache with model dimension. Update `is_free_tier()` to check full cache membership.

**Tech Stack:** Python, Cython (recompile `.so`), numpy, pytest

**Spec:** `docs/superpowers/specs/2026-03-22-markov-overhaul-design.md`

---

### Task 1: Verify model interface on all quantized models

**Files:**
- Read: `archive/btc_app/btc_core.py`
- Test: `btc_web/test_web.py`

**Context:** All 6 quantized models (bub, qr, pl, lppl, exp, ef) must have `find_percentile(t, price)` and `interp_price(q, t)` methods. `_FitsBasedModel` (base for qr, pl) has them at lines 353–384. `_CompositeModel` (base for bub, ef) has them at lines 436–467. LPPL and Exp extend `_FitsBasedModel` pattern. This task verifies the contract holds for all 6.

- [ ] **Step 1: Write test verifying all quantized models have the required interface**

Add to `btc_web/test_web.py` after the existing model test classes:

```python
@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestModelMCInterface:
    """All quantized models must support find_percentile and interp_price for MC."""

    def test_all_quantized_have_find_percentile(self):
        import _app_ctx
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue
            assert hasattr(mdl, 'find_percentile'), f"{key} missing find_percentile"
            assert callable(mdl.find_percentile), f"{key}.find_percentile not callable"

    def test_all_quantized_have_interp_price(self):
        import _app_ctx
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue
            assert hasattr(mdl, 'interp_price'), f"{key} missing interp_price"
            assert callable(mdl.interp_price), f"{key}.interp_price not callable"

    def test_find_percentile_returns_float(self):
        import _app_ctx
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue
            pct = mdl.find_percentile(16.0, 60000.0)
            assert isinstance(pct, float), f"{key}.find_percentile returned {type(pct)}"
            assert 0.0 <= pct <= 1.0, f"{key} percentile {pct} out of range"

    def test_interp_price_returns_positive(self):
        import _app_ctx
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue
            price = mdl.interp_price(0.5, 16.0)
            assert isinstance(price, float), f"{key}.interp_price returned {type(price)}"
            assert price > 0, f"{key} price {price} not positive"
```

- [ ] **Step 2: Run tests**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestModelMCInterface -v`
Expected: All PASS. If any model is missing `find_percentile` or `interp_price`, add the method to `btc_core.py` following the `_FitsBasedModel` pattern before proceeding.

- [ ] **Step 3: Commit**

```bash
git add btc_web/test_web.py
git commit -m "test: verify all quantized models have MC interface (find_percentile, interp_price)"
```

---

### Task 2: Update markov.py — model-aware API

**Files:**
- Modify: `btc_web/markov.py:63-215`
- Recompile: `btc_web/markov.cpython-*.so`

**Context:** Three functions take `qr_fits` and must take `model` instead. `_interp_qr_price_safe` is removed (replaced by `model.interp_price`). The hot loop in `monte_carlo_prices` (lines 175–196) calls model methods directly.

- [ ] **Step 1: Update `_prices_to_percentiles` (line 63)**

Change from:
```python
def _prices_to_percentiles(prices, years, qr_fits):
    """Convert price series to percentile series (0–1) using QR model."""
    pcts = np.empty(len(prices))
    for i in range(len(prices)):
        t = max(float(years[i]), 0.5)
        pcts[i] = _find_lot_percentile(t, float(prices[i]), qr_fits)
    return pcts
```
To:
```python
def _prices_to_percentiles(prices, years, model):
    """Convert price series to percentile series (0–1) using a price model."""
    pcts = np.empty(len(prices))
    for i in range(len(prices)):
        t = max(float(years[i]), 0.5)
        pcts[i] = model.find_percentile(t, float(prices[i]))
    return pcts
```

Also update the import line 13 — remove `_find_lot_percentile` and `qr_price` (no longer needed):
```python
from btc_core import yr_to_t
```

- [ ] **Step 2: Update `build_transition_matrix` (line 79)**

Change `qr_fits` parameter to `model`:
```python
def build_transition_matrix(prices, years, model, n_bins=5,
                            window_start_yr=None, window_end_yr=None,
                            step_days=30):
```

And the call on line 117:
```python
    pct_series = _prices_to_percentiles(w_prices, w_years, model)
```

- [ ] **Step 3: Update `monte_carlo_prices` (line 142)**

Change `qr_fits, genesis` parameters to `model`:
```python
def monte_carlo_prices(trans_matrix, bin_edges, start_pctile, n_steps,
                       n_sims, model, start_t, dt):
```

Replace the price conversion in the inner loop (line 190):
```python
            price = model.interp_price(pctile, t)
```

Remove the `genesis` parameter — it's no longer needed (was only used by `_interp_qr_price_safe` which is removed).

- [ ] **Step 4: Remove `_interp_qr_price_safe` (lines 199–215)**

Delete the entire function. It's replaced by `model.interp_price()`.

- [ ] **Step 5: Syntax check**

Run: `btc_venv/bin/python3 -m py_compile btc_web/markov.py && echo "OK"`
Expected: OK

- [ ] **Step 6: Recompile Cython module**

Run: `cd btc_web && btc_venv/bin/cythonize -i markov.py && cd ..`
If Cython is not installed: `btc_venv/bin/pip install cython && cd btc_web && btc_venv/bin/cythonize -i markov.py && cd ..`

Verify: `btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from markov import build_transition_matrix; print('OK')"`

- [ ] **Step 7: Commit**

```bash
git add btc_web/markov.py btc_web/markov.c btc_web/markov.cpython-*.so
git commit -m "feat: model-aware Markov API — accept model object instead of qr_fits

build_transition_matrix, monte_carlo_prices, _prices_to_percentiles
now use model.find_percentile() and model.interp_price() instead of
hardcoded qr_fits. _interp_qr_price_safe removed."
```

---

### Task 3: Update mc_overlay.py — model routing

**Files:**
- Modify: `btc_web/mc_overlay.py`

**Context:** `_resolve_fits(p)` at line 58 always returns `M.qr_fits`. Replace with `_resolve_model(p)` returning actual model object. Update `_get_transition_matrix` cache key from `id(fits)` to model key string. Remove the `== "bub"` guard in `_try_cached`. Update all callers to pass model instead of fits.

- [ ] **Step 1: Replace `_resolve_fits` with `_resolve_model` (line 58)**

```python
def _resolve_model(p):
    """Resolve the price model for MC simulation from mc_model_src param."""
    key = p.get("mc_model_src", "bub")
    mdl = _app_ctx.PRICE_MODELS.get(key)
    if mdl is not None and mdl.quantized:
        return mdl
    return _app_ctx.DEFAULT_MODEL
```

- [ ] **Step 2: Update `_get_transition_matrix` cache key (line 240–270)**

Change `fits=None` parameter to `model=None`. Replace `fits_id = id(fits)` with `model_key = model.short_name if model else "bub"`. Update cache key tuple to use `model_key` instead of `fits_id`.

Update the call to `build_transition_matrix` to pass `model` instead of `fits`:
```python
trans, edges, _ = build_transition_matrix(
    m.price_prices, m.price_years, model, n_bins, ws, we, step_days)
```

- [ ] **Step 3: Update `_try_cached` — remove `== "bub"` guard (line 568–572)**

Change from:
```python
def _try_cached(p, mc_years, blocked):
    if not blocked and p.get("mc_model_src", "bub") == "bub":
        return try_precomputed_paths(p, mc_years)
    return None
```
To:
```python
def _try_cached(p, mc_years, blocked):
    if not blocked:
        return try_precomputed_paths(p, mc_years)
    return None
```

- [ ] **Step 4: Update `try_precomputed_paths` (line 318) to pass model key to cache**

Add `mc_model_src` to the cache lookup key passed to `get_cached_paths()`. The exact change depends on how `get_cached_paths` is updated in Task 4 — for now, pass `p.get("mc_model_src", "bub")` as a new first parameter.

- [ ] **Step 5: Update `_prepare_sim` (line 554) to accept `model` instead of `fits`**

`_prepare_sim` passes `fits` through to `_get_transition_matrix`. Change its parameter from `fits=None` to `model=None` and update the call:
```python
def _prepare_sim(m, p, n_bins, step_days, mc_window, mc_ts, n_sims, mc_t_start, mc_dt, model=None, snap_grid=0):
    ...
    trans, edges, _ = _get_transition_matrix(m, n_bins, step_days, mc_window, model=model)
```

- [ ] **Step 6: Update `_run_full_simulation` (line 588+) to use model**

Replace `fits = _resolve_fits(p)` with `model = _resolve_model(p)`. Pass `model` to `_prepare_sim` and `monte_carlo_prices` instead of `fits`.

Remove the `genesis` argument from the `monte_carlo_prices` call (it was removed from the signature in Task 2).

- [ ] **Step 7: Update all overlay functions that call `_resolve_fits`**

Search for all remaining `_resolve_fits` calls and replace with `_resolve_model`. Each overlay function (`_mc_dca_overlay`, `_mc_withdraw_overlay`, `_mc_heatmap_overlay`) should pass `model` through to `_run_full_simulation`.

- [ ] **Step 8: Syntax check**

Run: `btc_venv/bin/python3 -m py_compile btc_web/mc_overlay.py && echo "OK"`
Expected: OK

- [ ] **Step 9: Commit**

```bash
git add btc_web/mc_overlay.py
git commit -m "feat: route MC simulation through selected model

_resolve_model() replaces _resolve_fits(). Transition matrix cache
key uses model.short_name instead of id(fits). _try_cached guard
removed so all cached models can hit pre-computed cache."
```

---

### Task 4: Update mc_cache.py — multi-model cache

**Files:**
- Modify: `btc_web/mc_cache.py`
- Modify: `btc_web/_app_ctx.py`

**Context:** Cache constants at lines 51–53. `generate_cache` at line 74 hardcodes `M.qr_fits`. `get_cached_paths` at line 351 and `get_cached_overlay` at line 373 look up by `(start_yr, pct_bin, mc_years)` — must add model dimension.

- [ ] **Step 1: Update cache constants in `_app_ctx.py`**

Find the current MC cache constants (e.g., `MC_FREE_START_YRS`, `MC_FREE_YEARS`, `MC_FREE_ENTRY_Q`, etc.) and update to new values. Also update in `mc_cache.py` lines 51–53:

```python
CACHED_START_YRS = [2028, 2031, 2035]
ENTRY_PCT_BINS = [0.01, 0.10, 0.50]
MC_YEARS_OPTIONS = [40]
MC_SIMS = 200
```

- [ ] **Step 2: Update `generate_cache` (line 74) to accept and use model**

Change signature to `generate_cache(start_yr, m, model, progress_cb=None)`.

Replace hardcoded `_app_ctx.M.qr_fits` calls with `model`, and remove the `genesis` argument from `monte_carlo_prices` (removed in Task 2):
```python
trans, bin_edges, _ = build_transition_matrix(
    m.price_prices, m.price_years, model, ...)
price_paths, _ = monte_carlo_prices(
    trans, bin_edges, pct_bin, n_steps, MC_SIMS,
    model, t_start, MC_DT)
```

Update npz file naming to include model key:
```python
npz_path = cache_dir / f"paths_{model.short_name}_{start_yr}.npz"
overlay_path = cache_dir / f"overlays_{model.short_name}_{start_yr}.npz"
```

- [ ] **Step 3: Update `get_cached_paths` (line 351) — add model parameter**

Change signature to `get_cached_paths(model_key, start_yr, pct_bin, mc_years, max_sims=None)`.
Update internal lookup key to include `model_key`.

- [ ] **Step 4: Update `get_cached_overlay` (line 373) — add model parameter**

Change signature to `get_cached_overlay(model_key, start_yr, pct_bin, mc_years, wd, infl_pct, stack)`.
Update internal lookup key to include `model_key`.

- [ ] **Step 5: Replace `is_cached_year` (line 431) with `is_cached`**

```python
def is_cached(model_key, start_yr, entry_q, mc_years):
    """Check if a specific (model, start_yr, entry_q, mc_years) combo is pre-computed."""
    if start_yr not in CACHED_START_YRS:
        return False
    if mc_years not in MC_YEARS_OPTIONS:
        return False
    if not is_cache_aligned_q(entry_q):
        return False
    # Check model is one of the cached models
    return model_key in _CACHED_MODEL_KEYS

_CACHED_MODEL_KEYS = frozenset(["bub", "qr", "pl", "lppl", "exp", "ef"])
```

- [ ] **Step 6: Update `snap_to_bin` and `is_cache_aligned_q` for new bin set**

Current `snap_to_bin` (line 415) rounds to nearest 10% and `is_cache_aligned_q` (line 425) checks against 10% boundaries. With new bins `[0.01, 0.10, 0.50]`, these will break for 1% and most other values.

Replace with:
```python
_CACHE_BINS = [0.01, 0.10, 0.50]

def snap_to_bin(raw_pctile):
    """Snap to nearest cached entry percentile bin."""
    return min(_CACHE_BINS, key=lambda b: abs(raw_pctile - b))

def is_cache_aligned_q(entry_q):
    """Check if entry_q is close enough to a cached bin."""
    return any(abs(entry_q - b) < 0.005 for b in _CACHE_BINS)
```

- [ ] **Step 7: Update cache loading functions — concrete structure**

The in-memory cache structure changes to `{(model_key, start_yr): {"paths": {...}, "overlays": {...}}}`.

Update these functions:
- `_load_full_cache()` (line 319): iterate `_CACHED_MODEL_KEYS × CACHED_START_YRS`, load `paths_{model}_{yr}.npz`
- `load_startup_cache()` (line 218): iterate model keys, `_CACHE.setdefault((model_key, yr), {})`
- `_npz_fingerprint()` (line 272): iterate `_CACHED_MODEL_KEYS × CACHED_START_YRS` for file existence
- `get_cached_paths()` (already updated in Step 3): `_CACHE.get((model_key, start_yr))`
- `get_cached_overlay()` (already updated in Step 4): same pattern
- `_save_shm()` / `_try_load_shm()`: no changes needed (they serialize/deserialize `_CACHE` as-is)

- [ ] **Step 8: Syntax check**

Run: `btc_venv/bin/python3 -m py_compile btc_web/mc_cache.py && btc_venv/bin/python3 -m py_compile btc_web/_app_ctx.py && echo "OK"`
Expected: OK

- [ ] **Step 8: Commit**

```bash
git add btc_web/mc_cache.py btc_web/_app_ctx.py
git commit -m "feat: multi-model MC cache — 6 models × 3 start years × 3 entry bins

Cache key gains model dimension. generate_cache accepts model object.
Constants updated: start_yrs=[2028,2031,2035], entry_bins=[1%,10%,50%],
duration=40yr, sims=200."
```

---

### Task 5: Update btcpay.py — free tier = full cache

**Files:**
- Modify: `btc_web/btcpay.py`

**Context:** `is_free_tier` at line 65 checks a fixed set of combos. `compute_price` at line 56 has separate cached/live prices. `is_cached_request` at line 81 checks start year only. All need updating.

- [ ] **Step 1: Write failing test for new `is_free_tier` signature**

```python
def test_free_tier_with_model(self):
    """Free tier should accept model parameter and check cache membership."""
    # All 6 models at cached params should be free
    for model_key in ["bub", "qr", "pl", "lppl", "exp", "ef"]:
        assert btcpay.is_free_tier(model_key, 40, 2028, 0.10)
        assert btcpay.is_free_tier(model_key, 40, 2031, 0.50)
        assert btcpay.is_free_tier(model_key, 40, 2035, 0.01)
    # Non-cached params should NOT be free
    assert not btcpay.is_free_tier("bub", 20, 2028, 0.10)  # wrong duration
    assert not btcpay.is_free_tier("bub", 40, 2027, 0.10)  # wrong start year
    assert not btcpay.is_free_tier("bub", 40, 2028, 0.30)  # wrong entry bin
    assert not btcpay.is_free_tier("s2f", 40, 2028, 0.10)  # non-quantized model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestBtcpayPricing::test_free_tier_with_model -v -x`
Expected: FAIL — `is_free_tier` doesn't accept `model` parameter.

- [ ] **Step 3: Update `is_free_tier` (line 65)**

New signature and implementation:
```python
def is_free_tier(model_key: str, mc_years: int, start_yr: int, entry_q: float = 0,
                 mc_bins: int = MC_BINS, mc_sims: int = MC_FREE_SIMS,
                 mc_freq: str = MC_FREQ) -> bool:
    """Free tier: entire pre-computed cache is free."""
    from mc_cache import is_cached, is_cache_aligned_q
    if mc_bins != MC_BINS or mc_sims > MC_FREE_SIMS or mc_freq != MC_FREQ:
        return False
    return is_cached(model_key, start_yr, entry_q, mc_years)
```

- [ ] **Step 4: Remove `is_cached_request` (line 81) and simplify `compute_price` (line 56)**

Remove `is_cached_request` entirely. Update `compute_price` to remove the `is_cached` parameter — only live pricing remains:
```python
def compute_price(tab: str, mc_years: int) -> int:
    """Price in satoshis for a live MC simulation."""
    base = _PRICE_BASE.get(mc_years, (0, 2000))
    sats = base[1]  # always live price (index 1)
    if tab == "hm":
        sats = int(sats * 0.5)
    return sats
```

Update `_PRICE_BASE` to remove cached prices:
```python
_PRICE_BASE = {10: 500, 20: 1000, 30: 1500, 40: 2000}
```

- [ ] **Step 5: Update `create_invoice` (line 148) — remove `is_cached` parameter**

```python
def create_invoice(tab: str, mc_years: int, description: str = "") -> dict:
    sats = compute_price(tab, mc_years)
    ...
```

- [ ] **Step 6: Run test to verify it passes**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestBtcpayPricing -v`
Expected: PASS. Update any other failing btcpay tests for the new signatures.

- [ ] **Step 7: Commit**

```bash
git add btc_web/btcpay.py btc_web/test_web.py
git commit -m "feat: free tier = full cache, remove cached pricing tier

is_free_tier() gains model parameter, checks full cache membership.
compute_price/create_invoice simplified to live-only pricing.
is_cached_request removed."
```

---

### Task 6: Update callbacks — plumb model through all call sites

**Files:**
- Modify: `btc_web/callbacks/mc_helpers.py`
- Modify: `btc_web/callbacks/mc_payment.py`
- Modify: `btc_web/callbacks/mc_controls.py`
- Modify: `btc_web/api.py`

**Context:** `is_free_tier` is called at 5 locations, `is_cached_request` at 2, `create_invoice` at 2. All must pass `model_key` and remove `is_cached`.

- [ ] **Step 1: Update `_mc_payment_check` in `mc_helpers.py` (line 107)**

Add `mc_model_src` parameter. Pass to `is_free_tier`:
```python
if btcpay.is_free_tier(mc_model_src, mc_yrs, start_yr_, entry_q_,
                        mc_bins=bins_, mc_sims=sims_, mc_freq=freq_):
```

- [ ] **Step 2: Update `_mc_setup` in `mc_helpers.py` (line 187)**

Pass `mc_model_src` to `is_free_tier`:
```python
is_free = mc_ok and btcpay.is_free_tier(
    mc_p.get("mc_model_src", "bub"),
    mc_years_c, mc_start_yr_c, mc_entry_q_c,
    mc_bins=mc_bins_c, mc_sims=mc_sims_c, mc_freq=mc_freq_c)
```

- [ ] **Step 3: Update `_mc_payment_initiate` in `mc_payment.py` (lines 104, 112, 114)**

**Important:** The callback currently has no State input for the model dropdown. Add `State("{pfx}-mc-model-src", "value")` for each tab prefix to the callback decorator, extract `mc_model_src` per-tab (same pattern as `mc_years`/`start_yr`/`entry_q`), then:

```python
# Line 104: add model_key
model_key = mc_model_src or "bub"
if btcpay.is_free_tier(model_key, mc_years, start_yr, entry_q):
    ...

# Line 112: remove is_cached_request
# Line 114: remove is_cached from create_invoice
result = btcpay.create_invoice(tab, mc_years)
```

- [ ] **Step 4: Update `_calc_mc_cost` in `mc_controls.py` (line 356)**

**Important:** Trace the call chain to `_calc_mc_cost` to ensure `model_key` is available as a parameter. The callback that calls `_calc_mc_cost` must have access to the model dropdown value — add it as a State input if not already present. Then pass to `is_free_tier`:
```python
is_free = btcpay.is_free_tier(model_key, mc_years, start_yr, entry_q,
                               mc_bins=mc_bins, mc_sims=mc_sims, mc_freq=mc_freq)
```

- [ ] **Step 5: Update `_mc_create_invoice` in `api.py` (lines 173, 176, 179)**

```python
# Line 173: add model_key
if btcpay.is_free_tier(model_key, mc_years, start_yr, entry_q):
    ...
# Line 176: remove is_cached_request
# Line 179: remove is_cached from create_invoice
result = btcpay.create_invoice(tab, mc_years)
```

- [ ] **Step 6: Verify cost display matches invoice — trace the cost path**

Read the payment modal code in `mc_payment.py` and `mc_controls.py`. Verify that:
1. The cost shown in the UI (`_calc_mc_cost`) calls `compute_price(tab, mc_years)` (no `is_cached`)
2. The invoice created calls `create_invoice(tab, mc_years)` which internally calls `compute_price(tab, mc_years)`
3. Both paths produce the same amount

- [ ] **Step 7: Syntax check all modified files**

Run: `for f in btc_web/callbacks/mc_helpers.py btc_web/callbacks/mc_payment.py btc_web/callbacks/mc_controls.py btc_web/api.py; do btc_venv/bin/python3 -m py_compile $f || echo "FAIL: $f"; done && echo "OK"`
Expected: OK

- [ ] **Step 8: Commit**

```bash
git add btc_web/callbacks/mc_helpers.py btc_web/callbacks/mc_payment.py btc_web/callbacks/mc_controls.py btc_web/api.py
git commit -m "feat: plumb model_key through all MC payment/free-tier call sites

is_free_tier gains model parameter at all 5 call sites.
is_cached_request removed from mc_payment and api.
create_invoice simplified to remove is_cached parameter."
```

---

### Task 7: Update UI cached indicators

**Files:**
- Modify: `btc_web/layout/mc_controls.py`

**Context:** The start year dropdown, entry percentile dropdown, and duration dropdown show bold text for cached values. These indicators must update to reflect the new cache dimensions: start years 2028/2031/2035, entry bins 1%/10%/50%, duration 40yr only.

- [ ] **Step 1: Update cached start year indicators**

Find where `CACHED_START_YRS` is used to style the start year dropdown options (bold labels). Update to use the new `[2028, 2031, 2035]`.

- [ ] **Step 2: Update cached entry percentile indicators**

Find where `ENTRY_PCT_BINS` or `MC_FREE_ENTRY_Q` is used to style the entry percentile dropdown. Update to bold 1%, 10%, 50%.

- [ ] **Step 3: Update cached duration indicators**

Find where `MC_YEARS_OPTIONS` is used to style the duration dropdown. Update to bold 40yr only.

- [ ] **Step 4: Syntax check**

Run: `btc_venv/bin/python3 -m py_compile btc_web/layout/mc_controls.py && echo "OK"`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add btc_web/layout/mc_controls.py
git commit -m "ui: update cached indicators for new MC cache dimensions

Start years: 2028/2031/2035. Entry bins: 1%/10%/50%. Duration: 40yr."
```

---

### Task 8: Update app.py prewarm and startup

**Files:**
- Modify: `btc_web/app.py`
- Modify: `btc_web/mc_cache.py` (startup mini-cache)

**Context:** The MC figure prewarm and startup mini-cache need updating for the new cache dimensions and multi-model support.

- [ ] **Step 1: Update startup mini-cache in mc_cache.py**

The `_STARTUP_PATH_ENTRIES` and `_STARTUP_OVERLAY_ENTRIES` lists define what's loaded at startup. Update to cover free-tier defaults for all 6 models:

```python
_CACHED_MODEL_KEYS = ["bub", "qr", "pl", "lppl", "exp", "ef"]
_STARTUP_PATH_ENTRIES = [(mdl, 2028, 0.10, 40) for mdl in _CACHED_MODEL_KEYS]
_STARTUP_OVERLAY_ENTRIES = [(mdl, 2028, 0.10, 40, 5000, 4, 1.0) for mdl in _CACHED_MODEL_KEYS]
```

- [ ] **Step 2: Update MC figure prewarm in app.py**

If there's MC figure prewarming beyond the mini-cache, update the parameters to match new cache dimensions.

- [ ] **Step 3: Syntax check**

Run: `btc_venv/bin/python3 -m py_compile btc_web/app.py && btc_venv/bin/python3 -m py_compile btc_web/mc_cache.py && echo "OK"`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add btc_web/app.py btc_web/mc_cache.py
git commit -m "perf: update MC prewarm and startup mini-cache for 6-model cache"
```

---

### Task 9: Generate new multi-model cache

**Files:**
- Modify: `btc_web/mc_cache.py` (generation script)
- Output: `mc_cache/paths_{model}_{start_yr}.npz` and `mc_cache/overlays_{model}_{start_yr}.npz`

**Context:** Run cache generation for all 6 models × 3 start years. This produces the npz files that get deployed to production.

- [ ] **Step 1: Write a cache generation driver**

Create or update the cache generation entry point that iterates all 6 models and 3 start years:

```python
# Run from project root:
# PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -c "
# from mc_cache import generate_all_caches
# generate_all_caches()
# "
```

The function should:
1. Load ModelData
2. Initialize all 6 PRICE_MODELS (same as app.py startup)
3. For each model, for each start year, call `generate_cache(start_yr, M, model)`
4. Print progress and final size

- [ ] **Step 2: Run cache generation**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -c "from mc_cache import generate_all_caches; generate_all_caches()"`
Expected: ~18 npz files (6 models × 3 start years × paths + overlays), total ~828 MB uncompressed.

- [ ] **Step 3: Validate cache**

```python
# Verify all expected files exist and load correctly
from mc_cache import get_cached_paths, is_cached
for model in ["bub", "qr", "pl", "lppl", "exp", "ef"]:
    for yr in [2028, 2031, 2035]:
        for q in [0.01, 0.10, 0.50]:
            assert is_cached(model, yr, q, 40), f"Missing: {model}/{yr}/{q}"
            paths = get_cached_paths(model, yr, q, 40)
            assert paths is not None, f"Load failed: {model}/{yr}/{q}"
print("All cache entries validated")
```

- [ ] **Step 4: Commit**

```bash
git add btc_web/mc_cache.py
git commit -m "feat: multi-model cache generation for 6 models × 3 start years

Generates ~828 MB of pre-computed MC paths and overlays.
npz naming: paths_{model}_{start_yr}.npz"
```

Note: The npz data files themselves are NOT committed to git — they're deployed separately to the production server.

---

### Task 10: Full integration test

**Files:**
- Modify: `btc_web/test_web.py`

- [ ] **Step 1: Update existing MC tests for new API signatures**

Search `test_web.py` for all tests that call `build_transition_matrix`, `monte_carlo_prices`, `is_free_tier`, `is_cached_request`, `compute_price`, `create_invoice`. Update their arguments to match the new signatures.

Key test classes to update:
- `TestBtcpayPricing` — `compute_price` no longer takes `is_cached`, `is_free_tier` takes `model_key` first
- `TestCacheConstants` — `is_cached_year` replaced by `is_cached`
- Any MC-related callback smoke tests

- [ ] **Step 2: Write behavioral test — different models produce different MC results**

```python
@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestModelAwareMC:
    """Different models must produce different transition matrices and price paths."""

    def test_different_models_different_transition_matrices(self):
        import _app_ctx
        from markov import build_transition_matrix
        bub = _app_ctx.PRICE_MODELS["bub"]
        pl = _app_ctx.PRICE_MODELS["pl"]
        m = _app_ctx.M
        trans_bub, _, _ = build_transition_matrix(m.price_prices, m.price_years, bub)
        trans_pl, _, _ = build_transition_matrix(m.price_prices, m.price_years, pl)
        assert not np.allclose(trans_bub, trans_pl), "BM and PL should have different transition matrices"

    def test_different_models_different_median_prices(self):
        import _app_ctx
        from markov import build_transition_matrix, monte_carlo_prices
        bub = _app_ctx.PRICE_MODELS["bub"]
        pl = _app_ctx.PRICE_MODELS["pl"]
        m = _app_ctx.M
        t_start = 16.0
        for mdl, label in [(bub, "bub"), (pl, "pl")]:
            trans, edges, _ = build_transition_matrix(m.price_prices, m.price_years, mdl)
            paths, _ = monte_carlo_prices(trans, edges, 0.5, 12, 50, mdl, t_start, 1/12)
            assert paths.shape == (50, 12), f"{label} wrong shape"
        # Just verify both run without error — price distributions will differ
```

- [ ] **Step 3: Run the full test suite**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -v`
Expected: All non-pre-existing tests PASS.

- [ ] **Step 3: Start dev server and verify MC works**

Run: `lsof -ti :8050 | xargs kill -9 2>/dev/null; DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &`

Manual checks (if MC cache is available locally):
1. Open DCA tab, select Power Law model in MC model dropdown
2. Verify MC fan overlay renders with PL model
3. Switch to Bubble Model, verify different fan shape
4. Check free tier badge shows "Free" for cached combos
5. Check cost modal shows live pricing for non-cached combos

- [ ] **Step 4: Commit test updates**

```bash
git add btc_web/test_web.py
git commit -m "test: update MC tests for model-aware API and new cache dimensions"
```

---

### Task 11: Production deployment

**Files:**
- Deploy: npz cache files to production server
- Restart: quantoshi-cache.service + quantoshi.service

- [ ] **Step 1: Push code to GitHub**

```bash
git push origin master  # or merge MarkovOverhaul → master first
```

- [ ] **Step 2: Deploy cache files to production**

```bash
scp mc_cache/paths_*.npz mc_cache/overlays_*.npz root@89.167.70.45:/opt/quantoshi/mc_cache/
```

- [ ] **Step 3: Deploy code and restart services**

```bash
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && systemctl restart quantoshi-cache && systemctl restart quantoshi"
```

- [ ] **Step 4: Verify production**

```bash
ssh root@89.167.70.45 "systemctl status quantoshi && curl -s localhost:8050 | head -5"
```

Check that the MC cache loads correctly from `/dev/shm` by watching logs:
```bash
ssh root@89.167.70.45 "journalctl -u quantoshi -n 20 | grep -i cache"
```

- [ ] **Step 5: Recompile Cython on production**

```bash
ssh root@89.167.70.45 "cd /opt/quantoshi/btc_web && /opt/quantoshi/btc_venv/bin/cythonize -i markov.py && systemctl restart quantoshi"
```

---

## Summary of Changes

| Task | What | Risk |
|------|------|------|
| 1 | Verify model interface | None — read-only check |
| 2 | markov.py model-aware API + Cython recompile | Medium — core engine change |
| 3 | mc_overlay.py model routing | Medium — orchestration layer |
| 4 | mc_cache.py multi-model cache | Medium — storage layer |
| 5 | btcpay.py free tier = cache | Low — simplification |
| 6 | Callbacks plumbing | Low — parameter passing |
| 7 | UI cached indicators | Low — cosmetic |
| 8 | Prewarm + startup | Low — performance |
| 9 | Cache generation | Medium — data generation |
| 10 | Integration test | None — verification |
| 11 | Production deployment | Medium — production |
