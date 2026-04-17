# User-Defined Model — Design Spec

**Date:** 2026-03-27
**Branch:** `UserModel`
**Goal:** Let users click two points on the Tab 1 bubble chart to define a custom power law model. The model is fully quantized (parallel lines from empirical residual distribution), registered across all tabs, color-coded orange, and included in the ticker percentile cycle. Session-only — disappears on refresh.

---

## Interaction Flow

### Entry: Floating Action Button (FAB)

An orange pencil icon (✎) floats in the bottom-right corner of the bubble chart. Three behaviors depending on state:

| State | FAB tap action |
|-------|---------------|
| No model, not drawing | Enter draw mode (`placing_p1`) |
| Drawing (any phase) | Abort draw, return to `idle`, preserve previous model |
| Model exists, not drawing | Show menu: Redraw / Delete / Cancel |

### Draw Mode

When draw mode is active:
- FAB pulses orange (`draw-active` CSS class)
- Toast: "Tap two points to define your model"
- Plotly `dragmode` set to `False` (disables zoom/pan so taps register as clicks)
- Chart border glow (subtle orange outline)

### Point Placement (repeated for point 1 and point 2)

1. User taps chart → orange circle marker appears at tap location
2. Confirmation menu appears (overlaid on chart, bottom center): **Accept / Adjust / Cancel**
3. **Accept** → point locked, advance to next phase
4. **Adjust** → chart zooms 2× centered on point, user taps corrected position (marker moves), menu reappears. Can adjust repeatedly for progressive zoom precision.
5. **Cancel** → marker removed, still in same placement phase (can retry or tap FAB to abort entirely)

### Model Creation

After both points accepted:
1. Server callback computes slope, intercept, residual distribution, quantile shifts, R²
2. Model data written to `user-model-store`
3. Orange line drawn on chart (user's quantile at 3px, other quantiles at 1.5px)
4. FAB changes to ✎ with dot indicator (model exists)
5. Draw mode exits, zoom/pan restored
6. U1 appears in Display Models checklists across all tabs
7. U1 added to ticker percentile cycle

### Redraw / Delete (FAB tap when model exists)

Menu appears: **Redraw / Delete / Cancel**
- **Redraw** → clears existing model, enters draw mode
- **Delete** → clears `user-model-store`, removes U1 from all checklists and ticker
- **Cancel** → dismisses menu

---

## State Machine

```
                                ┌────────────────┐
                                │     idle       │
                                │  (no drawing)  │
                                └───┬───────┬────┘
                    FAB tap         │       │        FAB tap
                   (no model)       │       │       (has model)
                                    ▼       ▼
                            ┌──────────┐  ┌──────────────┐
                            │placing_p1│  │ showing_menu  │
                            └────┬─────┘  │Redraw/Del/Can│
                     chart tap   │        └──────┬───────┘
                                 ▼               │
                          ┌─────────────┐   Redraw → placing_p1
                          │confirming_p1│   Delete → idle (clear model)
                          │Acc/Adj/Can  │   Cancel → idle
                          └──┬──┬──┬────┘
                   Accept    │  │  │  Cancel
                             │  │  └──────→ placing_p1
                             │  │ Adjust
                             │  └──→ zoom 2× → placing_p1
                             ▼
                      ┌──────────┐
                      │placing_p2│
                      └────┬─────┘
               chart tap   │
                           ▼
                    ┌─────────────┐
                    │confirming_p2│
                    │Acc/Adj/Can  │
                    └──┬──┬──┬────┘
             Accept    │  │  │  Cancel
                       │  │  └──────→ placing_p2
                       │  │ Adjust
                       │  └──→ zoom 2× → placing_p2
                       ▼
                 ┌───────────┐
                 │   done    │──→ model created ──→ idle
                 └───────────┘

  ANY draw phase ──[FAB tap]──→ idle (abort, previous model preserved)
  ANY draw phase ──[tab switch]──→ idle (abort, previous model preserved)
  showing_menu ──[FAB tap]──→ idle (dismiss menu)
```

### State Store

`dcc.Store("draw-mode-store", storage_type="memory")`:

```python
{
    "phase": "idle",        # idle | placing_p1 | confirming_p1 |
                            # placing_p2 | confirming_p2 | done |
                            # showing_menu
    "point1": null,         # {"t": float, "price": float} or null
    "point2": null,         # {"t": float, "price": float} or null
    "pre_draw_zoom": null,  # saved dragmode + axis ranges for restore
}
```

---

## UserModel: Fully Quantized from Two Points

### Construction

Given two points `(t1, p1)` and `(t2, p2)`:

```python
log_t1, log_p1 = log10(t1), log10(p1)
log_t2, log_p2 = log10(t2), log10(p2)
slope = (log_p2 - log_p1) / (log_t2 - log_t1)
intercept = log_p1 - slope * log_t1
```

### Empirical Quantile Shifts (asymmetric)

1. Compute residuals against all historical data points:
   ```python
   residuals = [log10(price_i) - (intercept + slope * log10(t_i))
                for t_i, price_i in zip(price_years, price_prices)]
   ```

2. The user's line has some empirical quantile — the fraction of residuals ≤ 0:
   ```python
   own_quantile = sum(1 for r in residuals if r <= 0) / len(residuals)
   ```

3. For each standard quantile `q`, compute the shift as the q-th percentile of the residual distribution:
   ```python
   shifts = {q: np.percentile(residuals, q * 100) for q in quantiles}
   ```

4. Build the fits dict (same structure as `_FitsBasedModel`):
   ```python
   fits = {q: {"intercept": intercept + shifts[q], "slope": slope}
           for q in quantiles}
   ```

The asymmetry is captured naturally: bubble-era residuals push upper quantiles far above the median while lower quantiles are tighter.

### Class Definition

```python
class UserModel(_FitsBasedModel):
    name = "User Model"
    short_name = "u1"
    legend_name = "U1"
    dash_style = "solid"
    quantized = True

    def __init__(self, slope, intercept, shifts, quantiles, r2_per_quantile, own_quantile):
        self.fits = {q: {"intercept": intercept + shifts[q], "slope": slope}
                     for q in quantiles}
        self.quantiles = sorted(quantiles)
        self.r2_per_quantile = r2_per_quantile
        self.own_quantile = own_quantile  # what percentile the drawn line is
        self.colors = {q: "#e67e22" for q in quantiles}  # all orange

    @classmethod
    def from_points(cls, t1, p1, t2, p2, price_years, price_prices, quantiles):
        """Factory: two points + historical data → fully quantized model."""
        import numpy as np
        log_t1, log_p1 = np.log10(t1), np.log10(p1)
        log_t2, log_p2 = np.log10(t2), np.log10(p2)
        slope = (log_p2 - log_p1) / (log_t2 - log_t1)
        intercept = log_p1 - slope * log_t1
        # Residuals against historical data
        mask = price_years >= 0.5
        t_hist = price_years[mask]
        p_hist = price_prices[mask]
        residuals = np.log10(np.maximum(p_hist, 1e-10)) - (intercept + slope * np.log10(t_hist))
        own_quantile = float(np.mean(residuals <= 0))
        shifts = {q: float(np.percentile(residuals, q * 100)) for q in quantiles}
        # R² per quantile
        from btc_core import _compute_log_r2
        r2 = {}
        for q in quantiles:
            predicted = 10.0 ** (intercept + shifts[q] + slope * np.log10(t_hist))
            r2[q] = _compute_log_r2(p_hist, predicted)
        return cls(slope, intercept, shifts, quantiles, r2, own_quantile)
```

Inherits `price_at`, `interp_price`, `find_percentile` from `_FitsBasedModel`.

### R² Computation

Per-quantile R² computed at creation time against historical data:
```python
for q in quantiles:
    predicted = model.price_at(q, price_years)
    r2 = _compute_log_r2(price_prices, predicted)
    r2_per_quantile[q] = r2
```

Shown in legend: `U1 Q25% R²=0.87`.

---

## Storage & Cross-Tab Registration

### User Model Store

`dcc.Store("user-model-store", storage_type="memory", data=None)`:

```python
{
    "slope": float,
    "intercept": float,
    "shifts": {str(q): shift for q in quantiles},  # JSON keys must be strings
    "r2": {str(q): r2 for q in quantiles},
    "own_quantile": float,   # e.g. 0.73
    "point1": {"t": float, "price": float},
    "point2": {"t": float, "price": float},
}
```

Session-only (`memory`). Disappears on refresh.

### Reconstruction

Any callback that needs the UserModel reconstructs it from store data:

```python
def _reconstruct_user_model(store_data):
    if not store_data:
        return None
    return UserModel(
        slope=store_data["slope"],
        intercept=store_data["intercept"],
        shifts={float(q): v for q, v in store_data["shifts"].items()},
        quantiles=[float(q) for q in store_data["shifts"].keys()],
        r2_per_quantile={float(q): v for q, v in store_data["r2"].items()},
        own_quantile=store_data["own_quantile"],
    )
```

### Display Models Checklists

All tabs' `{prefix}-model-show` checklists dynamically gain a "U1" option. A clientside callback watches `user-model-store`:
- Store has data → inject `{"label": " U1", "value": "u1"}` into checklist options
- Store cleared → remove "u1" from options and value

### Ticker Cycling

The `update_price_ticker` callback checks `user-model-store` (added as `State`). When present, reconstructs UserModel, computes `find_percentile(t, price)`, and appends to the `model_data` list with color `#e67e22` (orange).

### Figure Builders

All figure builders (bubble, DCA, retire, supercharge, citadel, heatmap) already loop over `active_models` and handle quantized models. When `"u1"` is in `active_models`, the callback passes `user-model-store` data in the params dict. The figure builder reconstructs UserModel and draws it.

**Bubble chart special case:** The user's own quantile line is drawn at 3px width; other U1 quantile lines at standard 1.5px overlay width. This is identified by comparing the quantile to `own_quantile` from the store.

---

## Visual Design

### FAB Button

| State | Icon | Background | Border | Animation |
|-------|------|-----------|--------|-----------|
| Idle, no model | ✎ | dark translucent | subtle white | none |
| Draw mode active | ✎ | solid orange | white | pulse glow |
| Idle, has model | ✎· (dot) | dark translucent | subtle white | none |

### Model Line Appearance

- **Color:** `#e67e22` (orange) for all quantile lines
- **User's drawn quantile:** 3px solid (bold — "I drew this")
- **Other quantiles:** 1.5px solid (standard overlay width)
- **Legend group:** `U1` with thermal-style colors per quantile

### Confirmation Menu

Absolute-positioned overlay at bottom center of chart. Three buttons in a row:
- ✓ Accept (green)
- ↻ Adjust (orange)
- ✕ Cancel (gray)

Appears only during `confirming_p1` and `confirming_p2` phases. Hidden otherwise.

### Toast

Small text overlay at top center of chart during draw mode: "Tap two points to define your model." Fades after 3 seconds.

---

## Technical Notes

### Click Capture on Empty Chart Space

Plotly's `clickData` only fires when clicking on an existing trace (data point, line), not on empty chart background. To capture clicks anywhere during draw mode, the bubble figure builder adds an **invisible background scatter trace** when draw mode is active: a grid of transparent (`opacity=0`) points covering the full axis range at ~50×50 resolution. This ensures `clickData` fires for any tap location. The background trace is excluded from the legend and removed when draw mode exits.

### LRU Cache Interaction

The bubble figure cache (`@lru_cache`) keys on quantized params. When user model data (slope/intercept) is included in the params dict, different user models produce different cache keys naturally. When the user redraws, the new params produce a new key — no manual cache invalidation needed.

### Dynamic Checklist Options

Adding "U1" to existing `{prefix}-model-show` checklists dynamically: use a server-side callback that outputs both `options` and `value` atomically (not clientside) to avoid race conditions with snapshot restore. The callback watches `user-model-store` and re-renders options for each tab's checklist, preserving existing selections.

### Ticker Integration

U1 is appended to `model_data` **after** the main `_MODEL_CYCLE` loop in `update_price_ticker`, not added to the static `_MODEL_CYCLE` list. The callback reads `user-model-store` as `State`, reconstructs UserModel when present, calls `find_percentile(t, price)`, and appends `{"key": "u1", "label": "U1", "pct": pct, "color": "#e67e22"}` to the model_data list.

### State Machine: `showing_menu` Phase

`showing_menu` is explicitly included in the "abort on FAB tap" rule: `showing_menu --[FAB tap]--> idle`. It is not a draw phase — no markers or temporary traces exist during this phase.

---

## Files to Modify

| File | Action | Purpose |
|------|--------|---------|
| `archive/btc_app/btc_core.py` | Modify | Add `UserModel` class (extends `_FitsBasedModel`) |
| `btc_web/layout/bubble.py` | Modify | Add draw-mode stores, confirmation menu overlay, toast |
| `btc_web/layout/common.py` | Modify | Update `_chart_tab_layout_with_fab` with menu + toast elements |
| `btc_web/layout/__init__.py` | Modify | Add `user-model-store` and `draw-mode-store` |
| `btc_web/callbacks/user_model.py` | **Create** | All draw-mode callbacks: FAB click, clickData capture, accept/adjust/cancel, model construction |
| `btc_web/callbacks/charts.py` | Modify | Bubble callback: add `clickData` input, pass user model to figure builder |
| `btc_web/callbacks/ticker.py` | Modify | Add `user-model-store` as State, append U1 to cycling |
| `btc_web/figures/bubble.py` | Modify | Draw user's own quantile at 3px; handle U1 overlay |
| `btc_web/assets/style.css` | Modify | Menu overlay styles, toast animation, draw-mode chart border glow |
| `btc_web/test_defaults.py` | Modify | Tests for UserModel construction, residual shifts, R² |

---

## Tests

### 1. UserModel construction from two points

```python
def test_user_model_from_two_points():
    model = UserModel.from_points(t1=5.0, p1=1000, t2=15.0, p2=100000,
                                  price_years=M.price_years, price_prices=M.price_prices,
                                  quantiles=M.QR_QUANTILES)
    assert 0 < model.own_quantile < 1
    assert len(model.fits) == len(M.QR_QUANTILES)
    assert all("intercept" in f and "slope" in f for f in model.fits.values())
```

### 2. All quantile lines share the same slope

```python
def test_user_model_parallel_lines():
    model = UserModel.from_points(...)
    slopes = [f["slope"] for f in model.fits.values()]
    assert all(s == slopes[0] for s in slopes)
```

### 3. Own quantile matches empirical fraction

```python
def test_own_quantile_matches_empirical():
    model = UserModel.from_points(...)
    # Recompute: fraction of data below the drawn line
    residuals = [log10(p) - (model.fits[model.own_quantile]["intercept"]
                 ... ]  # (pseudocode)
    assert abs(model.own_quantile - expected) < 0.01
```

### 4. Serialization round-trip

```python
def test_user_model_store_roundtrip():
    model = UserModel.from_points(...)
    store_data = model.to_store_dict()
    reconstructed = _reconstruct_user_model(store_data)
    assert reconstructed.fits == model.fits
```

### 5. R² is reasonable

```python
def test_user_model_r2():
    model = UserModel.from_points(...)
    for q, r2 in model.r2_per_quantile.items():
        assert -1 < r2 < 1  # log R² can be negative for bad fits
```

### 6. State machine transitions

```python
def test_draw_mode_cancel_preserves_model():
    # Existing model in store → enter draw → cancel → model still there
```

---

## Not In Scope

- Multiple user models (U1, U2, ...) — data structures support it but UI is single model for v1
- Persistence across refresh (localStorage) — session-only for v1
- User model on heatmap as primary model (only as overlay via Display Models)
- Editing an existing model (must redraw from scratch)
- Sharing user models via snapshot links
