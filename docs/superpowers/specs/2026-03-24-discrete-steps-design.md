# Discrete Steps Toggle — Design Spec

**Date:** 2026-03-24
**Scope:** Tabs 3 (DCA), 4 (Retire), 5 (Supercharger) — chart settings toggle

## Goal

Add a "Discrete steps" checkbox that switches simulation traces from smooth linear interpolation to step-function rendering (`line_shape="hv"`). This is more honest at lower frequencies (Quarterly, Annually) where price is constant within each period.

## Approach

Use the existing `_chart_toggles()` helper in `layout/common.py`. Adding an option there automatically covers all three tabs with no new component IDs. The toggles checklist is already in `_SNAPSHOT_CONTROLS`, so share links work without changes.

## Changes

### 1. Layout — `btc_web/layout/common.py`

Add to `_chart_toggles()` options list:

```python
{"label": " Discrete steps", "value": "discrete"},
```

Position: after "Annotate final values", before "Show legend".

### 2. Callbacks — `btc_web/callbacks/charts.py`

In each of the 3 tab callbacks (`update_dca`, `update_retire`, `update_supercharge`), the `toggles` list is already unpacked into boolean flags. Add:

```python
discrete = "discrete" in toggles
```

Pass into the figure params dict:

```python
discrete = discrete,
```

### 3. Figure builders — `btc_web/figures/dca.py`, `retire.py`, `supercharge.py`

In each builder, read the flag:

```python
_line_shape = "hv" if p.get("discrete") else "linear"
```

Apply to every `go.Scatter(mode="lines")` trace's `line` dict:

```python
line=dict(color=col, width=2, shape=_line_shape)
```

This includes:
- Primary model quantile traces
- Shade band boundary traces (upper/lower)
- Overlay model traces
- Dual-Y median USD traces
- MC overlay traces (if present — `mc_overlay.py` traces)

Traces that are NOT affected:
- `mode="markers+text"` traces (endpoint annotations)
- `mode="markers"` traces (Mode B scatter points)
- `mode="lines+markers"` traces (Mode B line charts)

### 4. MC overlay traces — `btc_web/mc_overlay.py`

MC fan overlay traces also use `mode="lines"`. Pass `discrete` through the params dict (already forwarded via `p`) and apply `shape=_line_shape` to MC traces in `_mc_dca_overlay`, `_mc_retire_overlay`, `_mc_supercharge_overlay`.

### 5. Snapshot compatibility

No changes needed. The `{prefix}-toggles` checklist is already in `_SNAPSHOT_CONTROLS` with bitmask encoding via `_CHECKLIST_OPTIONS`. The new `"discrete"` value will be encoded in the bitmask automatically.

**Legacy links:** Old links that don't include `"discrete"` in the bitmask will decode to an empty value for that bit, so the checkbox defaults to unchecked — correct behavior (preserves current rendering).

### 6. LRU cache

The `discrete` flag flows through the params dict which is already part of the cache key (via `_quantize_params`). Boolean values pass through `_q3` unchanged. No cache changes needed.

### 7. Prewarm

Default is unchecked (off), matching current behavior. No prewarm changes needed.

## Defaults

| Control | Default |
|---------|---------|
| Discrete steps | unchecked (smooth linear interpolation) |

## Not in scope

- Bubble tab (tab 1) — projection lines, not simulation traces
- Heatmap tab (tab 2) — no line traces
- Spline smoothing option — not requested
- Per-tab discrete setting — one toggle covers all three tabs independently (each has its own toggles checklist)
