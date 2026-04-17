# Codebase Deduplication and Server Callback Reduction

**Date:** 2026-04-16
**Goal:** Reduce redundant code, consolidate repeated patterns, and minimize server callbacks (ALARA).

---

## Overview

A 4-agent audit of `btc_web/` identified three categories of waste:

1. **~25 server-side callbacks** doing trivial work that JavaScript can handle clientside (boolean toggles, static returns, dict lookups, outline switches)
2. **~400 lines of duplicated figure-building boilerplate** across 7 chart builder modules + layout
3. **~240 lines of duplicated engine logic** (tax brackets, Markov returns, account draining) plus utils/cache module overlap

The work is staged so each stage ships independently. Stage 1 directly reduces server load; Stages 2-3 reduce maintenance burden.

---

## Stage 1: Clientside Callback Conversions (~24 callbacks)

### Tier 1 -- Trivial returns (8 callbacks)

| Callback | File | Server work | JS equivalent |
|----------|------|-------------|---------------|
| `toggle_share_modal` | `nav.py` | `return not is_open` | `function(n1, n1m, n2, is_open) { return !is_open; }` (4 positional args: 3 Inputs + 1 State) |
| `_close_freq_modal` | `mc_controls.py` | `return False` | `function() { return false; }` |
| `_quant_proceed` | `mc_payment.py` | `return (False, trigger+1)` | `function(n, t) { return [false, (t||0)+1]; }` |
| `_quant_cancel` | `mc_payment.py` | `return False` | `function() { return false; }` |
| `_mc_payment_cancel` | `mc_payment.py` | `return (False, True, "")` | `function() { return [false, true, ""]; }` |
| `restore_my_lots` | `snapshot_cb.py` | `return None` | `function() { return null; }` |
| `clear_history` | `snapshot_cb.py` | `return []` | `function() { return []; }` |
| `delete_user_model` | `user_model.py` | Returns 9 static None/string values | Multi-return of constants |

### Tier 2 -- Simple logic (11 callbacks)

| Callback | File | Server work | Notes |
|----------|------|-------------|-------|
| `set_p1` / `set_p2` | `user_model.py` | Extract from store, `round()` + format | Reimplement `_fmt_price_display` in JS (3 magnitude branches) |
| 3x pill callbacks | `citadel_scenarios.py` | Toggle `outline` booleans | Embed preset keys from `citadel_presets`; use `dash_clientside.callback_context.triggered_id` |
| `apply_hm_palette` | `snapshot_cb.py` | Look up 4 colors from static dict | `HM_PRESET_PALETTES` is in `__skip_export__` -- NOT in generated JS. Embed the 4-preset dict as a JS literal via f-string at registration time. |
| 5x `_restore_mc` | `mc_controls.py` | Extract fields from cached dict | Parse `mc_cached.path_key` State object, extract 6 fields with fallback defaults, return `no_update` x6 if absent. Embed MC constants as JS literals; 5 factory registrations. |

### Tier 3 -- Medium complexity (6 callbacks)

| Callback | File | Server work | Notes |
|----------|------|-------------|-------|
| `_hm_pill_click` | `routing.py` | Determine clicked pill, set outlines | Dynamic pill list generated into JS at registration; `triggered_id` JS API |
| `_hm_pill_sync` | `routing.py` | Set outline booleans from model key | Embed `_HM_PILL_MODELS` + `_HM_LEGACY_MODEL_FALLBACK` at registration |
| 3x `_toggle_freq_unlock` | `mc_controls.py` | Tri-output toggle | Straightforward JS but 3 outputs each |
| `_hm_deep_link` | `routing.py` | Parse pathname, index lookup | Embed pill model list at registration |

### Not converting (server data required)

`update_scanner`, `toggle_scanner_row` (pattern-matching `ALL`), `preview_percentile`, `manage_lots`, `_save_or_cancel` (tax), all chart builders, all MC upload parsers, ticker price fetch, `open_model_info_lightbox` (pattern-matching `ALL`), `_lazy_load_*` callbacks.

### Implementation notes

- `no_update` -> `window.dash_clientside.no_update`
- `PreventUpdate` -> `throw window.dash_clientside.PreventUpdate`
- `allow_duplicate=True` + `prevent_initial_call=True` must be preserved on converted callbacks
- Tier 3 callbacks with dynamic data (pill lists, preset keys) must generate JS strings at registration time via f-string interpolation
- Test each conversion individually; run full test suite after each tier

---

## Stage 2: Figure Builder Deduplication (~400 lines)

### New helpers in `figures/common.py`

**`_parse_quantiles(p) -> list[float]`**
Replaces `sorted([float(q) for q in (p.get("selected_qs") or [])])`. Basic form only -- callers needing `model.fits` filtering, reverse sort, or custom defaults apply post-processing themselves. ~10 call sites across 7 modules; saves ~25 lines.

**`_format_final_value(vals, prices, disp_mode, show_usd_parens=True) -> tuple[array, str]`**
Replaces the `if disp_mode == "usd" ... else ...` display formatting block. The `show_usd_parens=False` path is used by `build_overlay_traces()` (scalar-quantized overlays intentionally omit USD parenthetical to avoid duplicating the same number on every legend line). ~5 call sites; saves ~60 lines.

**`_quantile_trace(ts, y, q, color, label, width, shape, **kw) -> go.Scatter`**
Replaces the `quantile_shade() + quantile_opacity() + go.Scatter()` combo. Callers pass `legendgroup`, `legendgrouptitle_text`, etc. via `**kw`. ~6 call sites; saves ~75 lines.

**`_empty_state_annotation(layout)`**
Replaces the identical "No models selected -- check Display Models" annotation block. 5 occurrences across 4 files (dca, retire, supercharge x2, residuals). Saves ~20 lines.

**`_today_line_shapes(t_today, y_lo, y_hi, color, glow=True, yref="y") -> list[dict]`**
Replaces glow+dash today-line construction. `bubble.py` uses `glow=True, yref="y"`; `residuals.py` uses `glow=False, yref="paper"`. Saves ~24 lines.

**`_apply_final_steps(fig, p, tab, recovery=False, hover_fmt=None, show_qr=True, show_mc=False)`**
Lower-level finalization helper called by both `_finalize_chart()` and bubble/residuals builders. Handles typography, date hover, config annotation, watermark. `_finalize_chart()` continues to exist as the higher-level wrapper adding legend positioning, MC premium, and tuple return. Saves ~20 lines.

### New helpers in `layout/common.py`

**`_use_lots_checklist(prefix) -> dcc.Checklist`**
Replaces 4 copy-pasted "Use Stack Tracker lots" checklists (bubble, heatmap, sim_tabs, citadel). Saves ~12 lines.

**`_two_freq_model_slot(family, slot, damping_label, **kw) -> html.Div`**
Unifies `_hybppl_model_slot` and `_eppl_model_slot`. Parameters: `family` (id prefix, e.g. "hybppl"/"eppl"), `slot` ("a"/"b"), `damping_label` ("damped"/"entropy damped"). ~160 lines of structural duplication collapse to ~80 + 2 thin wrappers. Similarly unify `_global_hybppl_modal` and `_global_eppl_modal` into `_global_two_freq_modal(family, title)`.

### Fix: `citadel_tax.py` imports

Import `_lbl` from `layout.common` and use `_STYLE_HINT` instead of redefining `_HINT` locally. ~4 lines.

### Not touching

Supercharge's 130-line overlay loop -- its delay-grid structure genuinely does not fit `build_overlay_traces()`'s `sim_fn(prices)` signature. Forcing it would create a worse abstraction.

### Investigation deferred

Dual-axis median trace pattern across dca/retire/supercharge (flagged by reviewer, not yet verified).

---

## Stage 3: Engine and Utility Cleanup (~240 lines)

### 3A. Tax bracket inflation helper (~40 lines saved)

Extract `_inflate_tax_context(sim_year, config) -> TaxContext` returning a dataclass with `ord_brackets`, `std_ded`, `ltcg_brackets`, `niit_threshold`. Called by `_score_sources`, `_max_draw_before_boundary`, and `_pay_tax_amount`. Does NOT pre-compute marginal rates -- site 3 (`_pay_tax_amount`) uses differential evaluation (`apply_progressive_brackets(agi)` vs `apply_progressive_brackets(agi+1)`), which is fundamentally different from the bracket-walking in sites 1 and 2.

Files: `engines/citadel_waterfall.py`, `engines/citadel_tax_integration.py`.

### 3B. Markov/lognormal return consolidation (~120 lines saved)

Extract `_apply_returns(state, wrapper_prefix, config, deterministic, rng, ppy)` from `citadel_step.py:148-274`. The 5-6 near-identical blocks (taxable/TD/TF x reserves/investments) become 3 calls with different prefixes.

Constraints:
- **Call order must be preserved** (taxable -> TD -> TF) to maintain RNG sequence determinism
- Regime attribute naming uses a mapping dict to handle inconsistency (`res_short_regime` vs `td_res_short_regime`)
- TD/TF bounds checks (`if i < len(new.td_reserves)`) must be preserved
- Add a deterministic before/after regression test

Rename `_lognormal_return` to `_lognormal_return_pct` and accept raw percentage values (push the `rate/100` division inside). No external callers exist.

### 3C. Account drain helper (~40 lines saved)

Extract `_drain_accounts(state, field_names, remaining) -> float` replacing the "remaining/draw/decrement" loop in:
- `citadel_tax_integration.py:52-78` (RMD computation)
- `citadel_tax_integration.py:153-164` (tax payment)
- `citadel_waterfall.py:342-404` (execute_draw, 3 wrapper branches)

### 3D. Redis/cache/utils consolidation

| Change | Files affected |
|--------|---------------|
| Remove `cache.py` pass-through `redis_available()` | `utils.py` (3 sites), `app.py` (2), `generate_citadel_cache.py` (2), `api.py` (1) -- update to `_app_ctx.redis_available()` |
| Remove `cache.py` copied `_REDIS`/`_HAS_REDIS`/`_MODEL_FP` aliases | `cache.py`, callers update to `_app_ctx.*` |
| Refactor `utils.py` inline serialization in `_make_cached_builder` to use `_serialize_result`/`_deserialize_result` | `utils.py` only |
| Move `_compute_sc_loan` from `_app_ctx.py` to `engines/sc_loan.py` (new file, distinct from `callbacks/sc_loan.py`) | `_app_ctx.py`, `figures/dca.py`, `callbacks/sc_loan.py`, `test_figures.py` (4 import sites in test methods) |
| Inline `_startup_heatmap_defaults` into `app.py` | `utils.py`, `app.py` |
| Inline `_log_cache_stats` into `app.py` | `utils.py`, `app.py` |
| Move `_fetch_sparkline_svg` to `callbacks/ticker.py` | `utils.py`, `callbacks/ticker.py` |

All Redis import changes must be atomic (single commit).

### 3E. Figure getter factory -- DROPPED

The 9 `_get_*_fig` functions have 4+ distinct patterns (plain, add-day, mc-aware, mc-forced). A factory saves ~10 lines and adds indirection. Not worth it.

### 3F. Minor engine cleanups

- `citadel_sim.py:39-56`: Loop over `["", "td_", "tf_"]` prefixes for regime initialization (15 lines -> 5)
- `tax_lots.py:78-118`: Extract `_make_lot_gain(btc_used, lot, sale_price, sale_dt)` to deduplicate whole-lot vs partial-lot branches
- `citadel_bands.py` vs `citadel_sim._aggregate_results`: Note the duplication but do NOT consolidate (bands module serves re-aggregation of multi-sim results)

---

## Staging and Risk

| Stage | Scope | Lines saved | Risk | Dependencies |
|-------|-------|-------------|------|-------------|
| 1 | Clientside callbacks | ~25 server round-trips eliminated | Low (each testable individually) | None |
| 2 | Figure deduplication | ~400 lines | Medium (shared helpers must not regress chart output) | None |
| 3 | Engine + utils cleanup | ~240 lines + cleaner imports | Higher (touches simulation math) | None |

Stages are independent -- each can ship on its own. Within each stage, work is ordered by risk (lowest first).

---

## Testing strategy

- **Stage 1**: Run full test suite after each tier. Visually verify each converted callback in the browser (clientside callbacks are not caught by Python-level tests).
- **Stage 2**: Run full test suite. Compare figure output before/after for each extracted helper (screenshot comparison or JSON diff of trace data).
- **Stage 3**: Run full test suite (1224 tests, including 92 tax-specific tests). Add deterministic regression test for Markov return consolidation. Run Citadel E2E tests.

---

## What this does NOT cover

- New features or UI changes
- Supercharge overlay loop consolidation (intentionally excluded -- bad abstraction)
- Figure getter factory (dropped -- minimal savings)
- `citadel_bands.py` / `citadel_sim._aggregate_results` consolidation (different use cases)
- Dual-axis median trace investigation (deferred)
