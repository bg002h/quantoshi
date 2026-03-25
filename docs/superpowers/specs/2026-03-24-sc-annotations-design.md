# Tab 5 Annotation Overhaul — Design Spec

**Date:** 2026-03-24
**Goal:** Make depletion and terminal annotations toggle with legend clicks, extend to all overlay models, and optimize for mobile.

---

## 1. Depletion Arrows Toggle with Legend (Approach B)

### Current
Depletion arrows are `layout.annotations` dicts — static, cannot be toggled by Plotly legend clicks. Only the primary model (BM) generates them.

### New
Keep arrows as `layout.annotations` (preserving real arrow styling: line + arrowhead, curved stems when close). Add a `plotly_legendclick` JS handler:

- Each annotation gets a custom `_legendgroup` key matching its model's trace legendgroup (e.g., `"sc-bub"`, `"sc-pl"`)
- JS handler fires on `plotly_legendclick`, reads the toggled legendgroup, loops through `layout.annotations`, and sets `visible=false/true` on annotations with matching `_legendgroup`
- Plotly annotation dicts support arbitrary extra keys — they're ignored by the renderer but accessible to JS via `gd.layout.annotations[i]._legendgroup`

### JS location
New handler in `btc_web/assets/scanner.js` or a new `btc_web/assets/sc_legend.js`. Binds to the Plotly `plotly_legendclick` event on the supercharger graph element.

---

## 2. Overlay Models Get Depletion + Terminal Annotations

### Depletion detection
The overlay model loop already computes `vals = np.maximum(start_stack - np.cumsum(...), 0.0)`. Add depletion detection:
```python
depl_mask = vals == 0.0
depl_t = float(ts_d[np.argmax(depl_mask)]) if depl_mask.any() else None
```

### Depletion arrows
- Added to the shared stagger pool (`deplete_annots` list)
- Arrow stem + arrowhead color: **delay color** (from `delay_colors[di]`) — visually connects to the band
- Arrow text color: **model trace color** (from `MODEL_TRACE_COLORS`) — distinguishes which model predicted depletion
- Arrow text format: `"{model_legend_name} ~{year}"` (e.g., `"PL ~2048"`)
- Each annotation tagged with `_legendgroup` for JS toggle

### Terminal (endpoint) annotations
- Overlay model endpoints added to the existing `_pending_annots` list
- Colored with model trace color
- Go through `_resolve_edge_annotations` clustering (separate pool from depletion arrows)

---

## 3. Staggering

### Depletion arrows: shared pool
- All models' depletion arrows in one pool, sorted by x-position
- `_stagger_depletion_annots` reassigns `ay` values from `_ANNOT_STAGGER_Y` (3 heights: -20, -33, -46 pixels)
- Arrow stems can be curved (`ayref` pixel offsets) when annotations are close in x

### Terminal annotations: separate pool
- Existing `_resolve_edge_annotations` system handles overlap
- Clusters nearby labels, consolidates 4+ into merged text
- Each model's endpoints are distinct entries in the pool

---

## 4. Legend Labels Always Include Terminal Values

- All trace `name=` fields include the terminal value: `"BM Q1-10% -> $123,456"`
- On desktop: both on-chart annotations AND legend values visible (redundant but consistent)
- On mobile portrait: CSS/JS hides on-chart endpoint text traces (`max-width: 767px`), legend labels remain as the only place to see terminal values
- Uses same pattern as existing `d-md-none` / `matchMedia("(max-width: 767px)")` throughout the codebase

---

## 5. Arrow Styling

- Arrows remain as `layout.annotations` with `showarrow=True`
- `arrowhead=2` (filled triangle), `arrowsize=1`
- Stem is a line from text to `(x, y=0)` — real arrow, not just a marker
- `ax` / `ay` control stem curve and text offset
- Curved stems when multiple arrows are close in x-position (already handled by stagger `ay` variation)

---

## 6. Files Modified

| File | Changes |
|------|---------|
| `btc_web/figures/supercharge.py` | Add depletion detection + arrows to overlay model loop. Add `_legendgroup` key to all depletion annotations. Add overlay terminal annotations to `_pending_annots`. Include terminal values in trace `name=` for all models. |
| `btc_web/assets/sc_legend.js` | New file: `plotly_legendclick` handler that toggles annotation visibility by `_legendgroup`. |
| `btc_web/assets/style.css` | Mobile media query to hide endpoint text traces on portrait (`max-width: 767px`). |
| `btc_web/figures/common.py` | `_stagger_depletion_annots` may need minor update to preserve `_legendgroup` key during re-sort. |

---

## 7. Constraints

- `yref="paper"` with `y=0` is the ONLY allowed paper-space usage (per CLAUDE.md)
- `xref="x"` always (data space)
- Endpoint text traces use data-space coords (never paper)
- User is colorblind: arrow text color (model trace color) + text prefix provide two independent identification channels
