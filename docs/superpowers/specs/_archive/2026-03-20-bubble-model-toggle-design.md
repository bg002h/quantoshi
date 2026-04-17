# Bubble Model Toggle + EF Composite/Support/Future Bubbles — Design Spec

## Summary

Two changes to the Bubble tab (tab 1):

1. **Bubble Model toggle**: Add `"bub"` as a default-checked entry in the `bub-model-show` Display Models checklist. When unchecked, all main BM traces (quantile lines, shading, composite, support, future bubbles) are hidden. Other elements (historical data scatter, OLS, UCL, today line, scanner overlays) remain unaffected.

2. **EF composite/support/future bubbles**: When the Empirical Floor model is enabled via `bub-model-show`, render its composite curve, support line, and N future bubbles — not just its quantile lines. Use the shared Bubble Model Config panel controls (`show_comp`, `show_sup`, `bub-n-future`).

## Scope

- Tab 1 (Bubble + QR Overlay) only.
- No changes to other tabs, heatmap, DCA, retire, or supercharger.

## Components

### 1. Layout (`layout/bubble.py`)

Add `"bub"` to the `bub-model-show` checklist options. It must appear first in the list and be checked by default (included in `value=`).

Currently `bub-model-show` is built dynamically from `_app_ctx.PRICE_MODELS`, excluding the default bubble model. Change: include the default bubble model as the first entry.

### 2. Figure Builder (`figures/bubble.py`)

#### Main BM traces become conditional

Currently the main BM's quantile lines (lines 82–95), shading (lines 63–80), composite (lines 192–203), support (lines 180–189), and future bubble composites are drawn unconditionally. Wrap all of these in `if "bub" in p.get("active_models", []):`.

The `model` variable (currently `_app_ctx.DEFAULT_MODEL`) is still used for axis config, tick computation, etc. — only the trace drawing becomes conditional.

Note: UCL, OLS, historical data scatter, today line, and scanner overlays are NOT gated by the `"bub"` toggle. They remain independent.

#### Overlay loop gains composite/support/future-bubble rendering

The existing overlay loop (lines 98–133) draws quantile lines for quantized models and a single trajectory for non-quantized models. Add a new branch: if the overlay model has composite data (check `hasattr(mdl, "comp_by_n")`), also draw:

- **Composite curve**: `mdl.comp_by_n[n]` where `n = p["n_future"]`, masked to `[t_lo, t_hi]`. Line color from model's own palette. Legend: `"{mdl.name} composite (N={n})  R²={mdl.bm_r2:.4f}"`.
- **Support line**: `mdl.support_plot`. Dashed line in model's support color. Legend: `"{mdl.name} support"`.
- **Future bubble composites**: Same as main BM's future-bubble logic but using `mdl.comp_by_n` and `mdl.n_future_max`.

All governed by the same params: `p["show_comp"]`, `p["show_sup"]`, `p["n_future"]`.

#### Color scheme

| Trace | BM Color | EF Color |
|-------|----------|----------|
| Composite | #FFD700 (gold) | #D4A017 (amber) |
| Support | #888888 (gray) | #8B6914 (dark amber) |
| Quantile lines | Thermal palette (blue→red) | Amber palette (already defined in EF model) |

EF colors come from the model's own color definitions. Future models inheriting `_CompositeModel` would use their own palettes automatically.

### 3. Callback (`callbacks/charts.py`)

The `update_bubble` callback passes `active_models = model_show or []` to the figure builder. No change needed — `"bub"` will flow through naturally when checked.

#### Auto-Y-range fallback

The `auto_bubble_yrange` callback currently unconditionally computes Y range from `_app_ctx.DEFAULT_MODEL`. When `"bub"` is not in `model_show`:

- If other quantized models are active (e.g., EF), use the first active quantized model's price range for auto-Y.
- If no quantized models are active, fall back to the default model's range (safe default to avoid empty Y axis).

### 4. EmpiricalFloorModel data access (`btc_core.py`)

The EF model inherits from `_CompositeModel`. Its composite/support data is stored in **private** attributes:
- `_comp_by_n` — list of composite arrays indexed by N future bubbles
- `_support_plot` — pre-computed support line array
- `_bm_r2` — R² fit quality
- `_n_future_max` — maximum N
- `_t_grid` — time grid matching composite/support arrays

**Required change**: Add public `@property` accessors to `_CompositeModel` (or directly on `EmpiricalFloorModel`) for `comp_by_n`, `support_plot`, `bm_r2`, `n_future_max`, and `t_grid`. The main `ModelData` class already exposes these as public attributes (`m.comp_by_n`, `m.support_bm`, etc.) — the properties make the interface consistent.

The figure builder uses `mdl.t_grid` (not `m.years_plot_bm`) for masking/plotting EF composites, since the time grids may differ between models.

### 5. Snapshot/Share

`bub-model-show` is already in `_SNAPSHOT_CONTROLS` and `_TAB_CONTROLS["bubble"]`.

**`_CHECKLIST_OPTIONS` update required**: In `snapshot.py`, the `_CHECKLIST_OPTIONS` dict has a `"bub-model-show"` entry listing valid bitmask values. Add `"bub"` as the first entry in this list.

**Backward compatibility**: Old share links (without `"bub"` in the encoded `bub-model-show` value) must still show the bubble model. The snapshot decode logic should inject `"bub"` into the decoded `bub-model-show` list when it is absent and the link format is `q3` or `q2`. This prevents old links from unexpectedly hiding the main model.

### 6. Cache

The bubble figure LRU cache key already includes `active_models` (via `model_show`). Adding `"bub"` to the list changes the key, so no stale-cache issues.

The `_prewarm_caches()` function in `app.py` should be updated to include `"bub"` in the default `active_models` for the bubble tab prewarm call.

## Files Modified

| File | Change |
|------|--------|
| `btc_web/layout/bubble.py` | Add `"bub"` to `bub-model-show` options + default value |
| `btc_web/figures/bubble.py` | Gate main BM traces; add composite/support/future to overlay loop |
| `btc_web/callbacks/charts.py` | Auto-Y fallback when BM unchecked |
| `archive/btc_app/btc_core.py` | Add `@property` accessors to `_CompositeModel` |
| `btc_web/snapshot.py` | Add `"bub"` to `_CHECKLIST_OPTIONS["bub-model-show"]`; backward compat |
| `btc_web/app.py` | Update `_prewarm_caches()` with `active_models=["bub"]` |

## Out of Scope

- Independent per-model controls for composite/support/future (deferred — "table with controls" if needed later)
- Changes to tabs 2–8
- New models or model registration changes
- EF model data generation (`build_ef_model.py`)
