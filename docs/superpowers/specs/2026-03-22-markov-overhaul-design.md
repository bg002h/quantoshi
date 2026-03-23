# Markov MC Overhaul — Design Spec

**Date:** 2026-03-22
**Branch:** MarkovOverhaul
**Goal:** Make the Markov Chain Monte Carlo simulation model-aware, update the percentile calculation to use each model's own methods, redesign the cache for 6 models, and make the entire cache free tier.

---

## 1. Markov Engine API (markov.py)

### Current
- `build_transition_matrix(prices, years, qr_fits, ...)` — hardcoded to raw QR fits
- `monte_carlo_prices(..., qr_fits, ...)` — uses `_interp_qr_price_safe(q, t, qr_fits)` in hot loop
- `_prices_to_percentiles(prices, years, qr_fits)` — calls `_find_lot_percentile(t, price, qr_fits)`

### New
All three functions accept a `model` object (any `_FitsBasedModel` subclass with `find_percentile(t, price)` and `interp_price(q, t)`) instead of `qr_fits`.

- `_prices_to_percentiles(prices, years, model)` calls `model.find_percentile(t, price)`
- `build_transition_matrix(prices, years, model, ...)` passes model through
- `monte_carlo_prices(..., model, ...)` calls `model.interp_price(q, t)` in hot loop
- `_interp_qr_price_safe()` removed, replaced by `model.interp_price()`

**Note:** The current `_interp_qr_price_safe` clamps `q` to `[0.001, 0.999]` and `t >= 0.5`. In the new design, `q` is implicitly bounded by bin sampling (always within `(0, 1)`), and `model.interp_price` handles boundary quantiles via nearest-neighbor fallback. No explicit clamping needed.

### Cython Impact
- Source `.py` changes required (markov.py is pure Python compiled via Cython, not `.pyx`)
- `.so` must be recompiled
- Hot loop overhead unchanged — both old and new call Python-level functions

### Models Available for MC
All quantized models (6 total):
| Key | Name | quantized |
|-----|------|-----------|
| bub | Bubble Model | True |
| qr | Quantile Regression | True |
| pl | Power Law | True |
| lppl | LPPL | True |
| exp | Exponential | True |
| ef | BM Empirical Floor | True |

S2F (`quantized=False`) is excluded.

---

## 2. Cache Architecture

### Current
Single-model: 5 start years x 9 entry bins x 4 durations x 800 sims = ~796 MB (Bubble only)

### New
Multi-model: 6 models x 3 start years x 3 entry bins x 1 duration x 200 sims = ~828 MB

| Parameter | Current | New |
|-----------|---------|-----|
| Models | bub only | bub, qr, pl, lppl, exp, ef |
| Start years | 2026, 2028, 2031, 2035, 2040 | 2028, 2031, 2035 |
| Entry bins | 10%-90% (9 bins) | 1%, 10%, 50% |
| Durations | 10yr, 20yr, 30yr, 40yr | 40yr only |
| Sims | 800 | 200 |
| Size | ~796 MB | ~828 MB |

### Cache Key Structure
- **Path key:** `(model_key, start_yr, entry_pct_bin, mc_years)`
- **Overlay key:** path key + `(wd_amount, inflation_pct, stack_btc)`

### Overlay Grid
Same wd/infl/stack grid as current (6 wd x 7 infl x 6 stack = 252 combos per path set), applied per-model. With 54 path sets (6 models x 3 start years x 3 entry bins) x 252 overlays = 13,608 overlay arrays. At 200 sims with fan percentile aggregation, overlay data is the majority of cache size (~800 MB of the ~828 MB total).

### npz File Naming
`paths_{model}_{start_yr}.npz` (e.g., `paths_pl_2028.npz`)
Replaces current `paths_{start_yr}.npz`.

### /dev/shm Fast Restart
Same approach: first load parses npz files (~10s for 6 models), serializes to `/dev/shm/quantoshi_mc.dat`, subsequent restarts load in ~1s. Fingerprint validated (npz mtime + size).

### Transition Matrix
Built per-model. Each model produces different percentile trajectories from the same historical price data, yielding different transition dynamics.

---

## 3. mc_overlay.py — Model Routing

### Current
`_resolve_fits(p)` always returns `_app_ctx.M.qr_fits` regardless of `mc_model_src`.

### New
`_resolve_model(p)` returns actual model object from `_app_ctx.PRICE_MODELS[p["mc_model_src"]]`. Falls back to `_app_ctx.DEFAULT_MODEL` if key missing or not quantized.

- All overlay functions pass model object instead of fits
- `_get_transition_matrix` cache key changes from `id(fits)` to `model.short_name` string (prevents cross-model cache collisions)
- `_try_cached` guard `== "bub"` removed — all cached models can hit pre-computed cache
- `try_precomputed_paths()` adds model key to cache lookup
- Client-side `mc_cached` path_key includes `mc_model_src` (already does)
- 3-level fallthrough unchanged: client cache -> server cache -> live simulation

---

## 4. Free Tier & Payment

### Current
- Free: 6 combos (3 start years x 2 durations), entry Q10%, Bubble only, <=100 sims
- Paid cached: 100-400 sats
- Paid live: 500-2000 sats

### New
- **Free: entire cache** (6 models x 3 start years x 3 entry bins x 40yr x 200 sims)
- **Paid: live simulations only** — same pricing as current live tier (500-2000 sats by duration), same price regardless of model
- `is_free_tier(model, mc_years, start_yr, entry_q, ...)` gains a `model` parameter; returns True if `(model, start_yr, entry_q, mc_years)` is in the cache
- Cached pricing tier removed entirely — `is_cached` parameter removed from `compute_price()` and `create_invoice()`
- `is_cached_request()` removed (dead code — free tier now gates all cached combos)

**Call sites requiring `model` parameter addition (6 total):**
1. `btcpay.py` — `_FREE_TIER_COMBOS` set construction
2. `callbacks/mc_helpers.py:107` — `_mc_payment_check()`
3. `callbacks/mc_helpers.py:187` — `_mc_setup()`
4. `callbacks/mc_payment.py:104` — `_mc_payment_initiate()`
5. `btcpay.py` or `api.py:173` — API validation
6. `callbacks/mc_controls.py:356` — control state

### BTCPay Cost Verification
Before triggering payment, verify the cost shown in the UI modal matches:
1. `markov.compute_cost()` output
2. The amount sent to `btcpay.create_invoice()`

Trace: `compute_cost()` -> callback -> modal display -> `create_invoice()` — all must agree.

---

## 5. UI Changes

Minimal — the model dropdown already exists in `_mc_controls(prefix)`.

1. **Model dropdown** — already works, backend now uses it (Sections 1+3)
2. **Start year dropdown** — update bold/cached indicators for 2028, 2031, 2035
3. **Entry percentile dropdown** — update bold/cached indicators for 1%, 10%, 50%
4. **Duration dropdown** — bold/cached indicator on 40yr only
5. **Free tier badge** — logic updates per Section 4, UI unchanged
6. **Cost modal** — verify displayed cost matches compute_cost() and BTCPay invoice

No new UI components required.

---

## 6. Cache Generation & Deployment

### Generation Script
1. Iterate all 6 models — build per-model transition matrix, run MC using that model's `find_percentile` / `price_at`
2. New npz naming: `paths_{model}_{start_yr}.npz`
3. Overlay generation: same wd/infl/stack grid, per-model
4. Generation time: ~1.5x current wall time (6x models, 0.25x sims)
5. Post-generation validation: verify each npz loads, total /dev/shm ~828 MB

### mc_cache.py Changes
- Key lookup functions gain `model` dimension
- `is_cached_year()` -> `is_cached(model, start_yr, entry_q, mc_years)`
- Startup mini-cache loads free-tier defaults for all 6 models

### Deployment
- systemd service (`quantoshi-cache.service`) — unchanged structure, loads more npz files
- `quantoshi.service` — unchanged, `After=quantoshi-cache.service`

---

## 7. Files Modified

| File | Changes |
|------|---------|
| `btc_web/markov.py` | Model-aware API: accept model object instead of qr_fits. Remove `_interp_qr_price_safe`. Recompile `.so`. |
| `btc_web/mc_overlay.py` | `_resolve_model()` replaces `_resolve_fits()`. Pass model to markov functions. Model key in transition matrix cache. |
| `btc_web/mc_cache.py` | Model dimension in cache keys, npz naming, `is_cached()` replaces `is_cached_year()`. Updated constants. `generate_cache()` updated to iterate all 6 models (currently hardcodes `qr_fits`). |
| `btc_web/btcpay.py` | `is_free_tier()` checks full cache membership. Remove cached pricing tier. Cost verification. |
| `btc_web/callbacks/mc_helpers.py` | Pass model key through `_mc_setup()` / `_build_mc_params()`. Add `model` to all `is_free_tier()` calls. |
| `btc_web/callbacks/mc_payment.py` | Remove `is_cached` from `create_invoice()` call. Always pass `is_cached=False` (free tier gates cached combos). Add `model` to `is_free_tier()` call. |
| `btc_web/layout/mc_controls.py` | Update cached indicators (start years, entry bins, durations). |
| `btc_web/app.py` | Update prewarm for new cache dimensions. |
| `btc_web/_app_ctx.py` | Update cache constants (start years, entry bins, free tier). |
| `btc_web/test_web.py` | Update MC tests for new API, cache constants, free tier logic. |
| `archive/btc_app/btc_core.py` | Verify all quantized models have `find_percentile()` and `interp_price()` methods. |
| `btc_web/api.py` | Add `model` to `is_free_tier()` call at line 173. |
