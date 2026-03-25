# Tab 5 Annotation Overhaul — Design Spec

**Date:** 2026-03-24
**Goal:** Make depletion and terminal annotations toggle with legend clicks, extend to all overlay models, and optimize for mobile.

---

## 1. Depletion Arrows Toggle with Legend (Approach B)

### Current
Depletion arrows are `layout.annotations` dicts — static, cannot be toggled by Plotly legend clicks. Only the primary model (BM) generates them.

### New
Keep arrows as `layout.annotations` (preserving real arrow styling: line + arrowhead, curved stems when close). Add a `plotly_legendclick` JS handler:

- Each annotation's `name` attribute (a recognized Plotly annotation property, not rendered visually) stores the legendgroup string (e.g., `"sc-bub"`, `"sc-pl"`)
- JS handler fires on `plotly_legendclick`. The event provides `curveNumber`; the handler looks up `gd.data[curveNumber].legendgroup` to get the legendgroup string, then loops through `gd.layout.annotations` and sets `visible=false/true` on annotations where `name` matches
- The handler returns `false` to prevent Plotly's default toggle, then manually toggles both trace visibility and annotation visibility together via `Plotly.update()`

**Note:** Custom keys (e.g., `_legendgroup`) are stripped by Plotly.js during its defaults-supply pipeline. The `name` attribute is a recognized annotation property preserved through rendering — use it instead.

### JS location
New file `btc_web/assets/sc_legend.js`. Binds to the `plotly_legendclick` event on the supercharger graph element (`#supercharge-graph .js-plotly-plot`).

---

## 2. Overlay Models Get Depletion + Terminal Annotations

### Depletion detection
The overlay model loop already computes `vals = np.maximum(start_stack - np.cumsum(...), 0.0)`. Add depletion detection:
```python
depl_mask = vals == 0.0
depl_t = float(ts_d[np.argmax(depl_mask)]) if depl_mask.any() else None
```

### `ov_results` tuple expansion
Current `ov_results[(d, q)]` stores `(ts_d, y_vals)`. Expand to `(ts_d, y_vals, depl_t, t_start_d, vals, prices)` to match the primary model's `results` structure. All unpacking sites in the overlay trace loops must be updated.

### Depletion arrows
- Added to the shared stagger pool (`deplete_annots` list)
- Arrow stem + arrowhead color: **delay color** (from `delay_colors[di]`) — visually connects to the band
- Arrow text color: **model trace color** (from `MODEL_TRACE_COLORS`) — distinguishes which model predicted depletion
- Arrow text format: `"{model_legend_name} ~{year}"` (e.g., `"PL ~2048"`)
- Each annotation's `name` attribute set to model legendgroup for JS toggle

### Terminal (endpoint) annotations
- Overlay model endpoints added to the existing `_pending_annots` list
- Colored with model trace color
- Go through `_resolve_edge_annotations` clustering (separate pool from depletion arrows)

---

## 3. Staggering

### Depletion arrows: shared pool
- All models' depletion arrows in one pool, sorted by x-position
- Expand `_ANNOT_STAGGER_Y` from 3 heights to 5: `[-20, -33, -46, -59, -72]` (13px apart, ~1 font-height). With 6 models x 3 delays = up to 18 arrows, 5 heights provide better spread before cycling
- `_stagger_depletion_annots` reassigns `ay` values from expanded `_ANNOT_STAGGER_Y`
- Arrow stems can be curved (`ay` pixel offsets) when annotations are close in x

### Terminal annotations: separate pool
- Existing `_resolve_edge_annotations` system handles overlap
- Clusters nearby labels, consolidates 4+ into merged text
- Each model's endpoints are distinct entries in the pool

---

## 4. Legend Labels Always Include Terminal Values

- All trace `name=` fields include the terminal value: `"BM Q1-10% -> $123,456"`
- On desktop: both on-chart annotations AND legend values visible (redundant but consistent)
- On mobile portrait: endpoint text traces skipped at figure-build time when `is_mobile=True`
- Mobile detection: `dcc.Store("viewport-width")` updated by a clientside callback on page load + window resize. Chart callbacks read this as a `State` input. Figure builder checks `width < 768` to set `is_mobile`
- **Rationale:** CSS cannot selectively hide Plotly SVG text traces (they have no distinguishing class). The `dcc.Store` approach is needed because the figure builder runs server-side and cannot read `window.innerWidth` directly

---

## 5. Arrow Styling

- Arrows remain as `layout.annotations` with `showarrow=True`
- `arrowhead=2` (filled triangle), `arrowsize=1`
- Stem is a line from text to `(x, y=0)` — real arrow, not just a marker
- `ax` / `ay` control stem curve and text offset
- Curved stems when multiple arrows are close in x-position (handled by stagger `ay` variation)

---

## 6. Files Modified

| File | Changes |
|------|---------|
| `btc_web/figures/supercharge.py` | Expand `ov_results` tuple. Add depletion detection + arrows to overlay model loop. Set `name` attribute on all depletion annotations. Add overlay terminal annotations to `_pending_annots`. Include terminal values in trace `name=` for all models. Skip edge annotations when `is_mobile`. |
| `btc_web/assets/sc_legend.js` | New file: `plotly_legendclick` handler. Looks up `gd.data[curveNumber].legendgroup`, matches against annotation `name`, toggles both trace visibility and annotation visibility via `Plotly.update()`. Returns `false` to prevent default. |
| `btc_web/figures/common.py` | Expand `_ANNOT_STAGGER_Y` to 5 heights. `_stagger_depletion_annots` preserves `name` key during re-sort. |
| `btc_web/_app_ctx.py` | Update `ANNOT_STAGGER_Y` constant to 5 values. |
| `btc_web/layout/common.py` | Add `dcc.Store("viewport-width")` to layout. Add clientside callback for viewport width detection. |
| `btc_web/callbacks/charts.py` | Add `State("viewport-width", "data")` to supercharger callback, pass `is_mobile` to figure builder. |

---

## 7. Constraints

- `yref="paper"` with `y=0` is the ONLY allowed paper-space usage (per CLAUDE.md)
- `xref="x"` always (data space)
- Endpoint text traces use data-space coords (never paper)
- User is colorblind: arrow text color (model trace color) + text prefix provide two independent identification channels
- Plotly.js strips unknown annotation keys — use recognized `name` attribute for legendgroup tagging
