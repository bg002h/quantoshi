# Codebase Optimization — Design Spec

**Date:** 2026-03-28
**Branch:** `Optimize`
**Goal:** Reduce duplication, simplify structure, improve maintainability across the Quantoshi web app without changing user-visible behavior.

---

## Audit Summary

Full codebase audit identified 17 issues across 5 categories. Phase 1 (debug cleanup) already completed. Remaining work organized into 5 phases, each producing working software independently.

---

## Phase 2: Shared Constants Consolidation

**Risk:** Low. Moving constants to a single source doesn't change logic.

### 2a. `_q3` → `_app_ctx.py`

**Current:** `utils.py:23-29` and `figures/common.py:29-35` (identical copy named `_q3_trace`).

**Fix:** Move to `_app_ctx.py`. Both `utils.py` and `figures/common.py` import from `_app_ctx`. Delete both copies. No circular dependency risk — `_app_ctx` has no upstream imports.

### 2b. `FREQ_PPY` consolidation

**Current:** `_app_ctx.py:11`, `engines/citadel.py:13` (subset), `markov.py:17` (copy).

**Fix:**
- `engines/citadel.py`: import from `_app_ctx` and filter to supported freqs (`{k: v for k, v in FREQ_PPY.items() if k in ("Monthly", "Quarterly", "Annually")}`)
- `markov.py`: gitignored Cython source — document the duplication but can't modify

### 2c. `FREQ_LABEL` — new shared mapping

**Current:** `figures/supercharge.py:85` has a dict, `figures/dca.py:263` uses string slicing, inconsistent results.

**Fix:** Add `FREQ_LABEL = {"Daily": "/day", "Weekly": "/wk", "Monthly": "/mo", "Quarterly": "/qtr", "Annually": "/yr"}` to `_app_ctx.py`. Replace both usages.

### 2d. `_PATH_TO_TAB` consolidation

**Current:** `callbacks/nav.py:48-52` (Python dict), clientside JS string (nav.py:138), `snapshot.py` implicit.

**Fix:** Define in `_app_ctx.py`. Import in `nav.py`. Generate the JS map string from the Python dict in the clientside callback.

### 2e. `_THERMAL_STOPS` deduplication

**Current:** `figures/common.py:101-114` standalone list + `_app_ctx.py:34-39` inside `PALETTES["default"]["thermal_stops"]`.

**Fix:** Remove standalone `_THERMAL_STOPS` from `common.py`. Use `_app_ctx.PALETTES["default"]["thermal_stops"]` as the fallback.

---

## Phase 3: Figure Builder Deduplication

**Risk:** Medium. Changing figure builder internals must preserve exact chart output.

### 3a. Extract overlay model loop into `figures/common.py`

**Current:** 4 near-identical blocks in `dca.py`, `retire.py`, `supercharge.py`, `citadel.py`.

Each does:
1. Iterate `p.get("active_models", [])`
2. Check for `"u1"` → `UserModel.from_store_dict()`
3. Otherwise `PRICE_MODELS.get(model_key)`
4. Check `mdl.quantized` → per-quantile traces or single trajectory
5. Build `go.Scatter` with consistent styling

**Fix:** New function in `figures/common.py`:

```python
def build_overlay_traces(p, t_arr, sel_qs, stack, palette):
    """Build alternative model overlay traces. Used by all sim-tab figure builders."""
    traces = []
    for model_key in p.get("active_models", []):
        if model_key == "bub":
            continue
        mdl = _resolve_model(model_key, p)
        if not mdl:
            continue
        # ... unified quantized/non-quantized trace building
    return traces
```

Where `_resolve_model(key, p)` handles the `u1` special case centrally.

### 3b. Unify DCA/Retire overlay patterns

**Current:** `dca.py:206-256` and `retire.py:101-151` are nearly line-for-line identical (just different variable names).

**Fix:** Both call `build_overlay_traces()` from Phase 3a. The per-model simulation logic stays in each builder.

---

## Phase 4: Callback Boilerplate Reduction

**Risk:** Medium. Changing callback structure affects Dash's callback graph.

### 4a. MC-sandwich callback factory

**Current:** `callbacks/charts.py` has `update_dca` (~100 lines), `update_retire` (~100 lines), `update_supercharge` (~100 lines) following identical patterns:

```python
@callback(
    Output("{tab}-graph", "figure"),
    Output("{tab}-mc-results", "data"),
    Output("{tab}-mc-status", "children"),
    Output("{tab}-mc-rendered-key", "data"),
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Output("mc-save-tab", "data", allow_duplicate=True),
    Output("{tab}-mc-unblocked", "data"),
    Input("{tab}-run-btn" or controls...),
    # ~30 identical MC control inputs
    ...
)
def update_{tab}(...):
    mc_ok, is_free, mc_p, blocked = _mc_setup(tab, ...)
    fig, mc_result = _get_{tab}_fig(dict(..., **mc_p))
    return _mc_finalize(tab, fig, mc_result, ...)
```

**Fix:** Create `_register_sim_tab_callback(tab, defaults_dict, get_fig_fn, param_builder_fn)` that generates the callback with the correct outputs, inputs, and the 3-step MC sandwich pattern. Each tab provides only its unique param-building logic.

**Note:** This is the highest-impact dedup but also the riskiest. The heatmap and citadel callbacks are different enough to remain separate. Only DCA, Retire, and Supercharge are candidates.

---

## Phase 5: File Splitting

**Risk:** Low. Moving code between files with proper imports.

### 5a. `callbacks/nav.py` (773 lines) → 3 files

- `callbacks/routing.py` — tab routing, URL handling, FAQ accordion
- `callbacks/splash.py` — splash modal, easter eggs, journey tracking
- `callbacks/nav.py` — navbar drawer, palette sync, export, mobile toggles

### 5b. `mc_overlay.py` — split Citadel overlay

- `mc_overlay.py` — DCA/Retire/SC overlays, cache helpers, serialization
- `mc_citadel.py` — `_mc_citadel_overlay()` + `_CITADEL_MC_COLORS` + Celery integration

---

## Phase 6: UI Consistency & Polish

**Risk:** Low. Adding missing features, fixing inconsistencies.

### 6a. Citadel export button

Add `("citadel", "citadel-graph")` to `_EXPORT_TABS` in `callbacks/nav.py` so the Citadel tab has a working image export button.

### 6b. Consistent frequency labels

Use the new `FREQ_LABEL` from Phase 2c in all chart titles across DCA, Retire, Supercharge.

### 6c. Redis wrapper consolidation

Remove `redis_available()` from `cache.py` — callers should use `_app_ctx.redis_available()`. Keep `cache.py` focused on cache operations only.

---

## Execution Order

1. ~~Phase 1: Debug cleanup~~ ✅ DONE
2. Phase 2: Constants consolidation (safe, no logic changes)
3. Phase 3: Figure builder dedup (medium risk, highest LOC reduction)
4. Phase 5: File splitting (safe, can run in parallel with Phase 3)
5. Phase 4: Callback factory (depends on Phase 3 for overlay unification)
6. Phase 6: UI polish (last, depends on Phases 2-5)

---

## Not in Scope

- Changing any user-visible behavior or defaults
- Refactoring the Citadel engine (`engines/citadel.py`)
- Modifying `btc_core.py` model classes
- Changing snapshot encoding format
- Performance optimization (separate concern)
- Test refactoring (test_web.py is large but functional)
