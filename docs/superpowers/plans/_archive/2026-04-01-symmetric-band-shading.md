# Symmetric Band Shading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current adjacent-pair shading with symmetric band shading. Colors are symmetric about Q50%, use the model's trace color, and allow up to 2 shaded regions (inner + outer) when 3+ quantile traces are present. Shading is per-model, user-togglable.

**Architecture:** 2 tasks: (1) New symmetric shading function in figures/bubble.py, (2) Symmetric quantile colors in the thermal palette. Clean, focused changes.

**Tech Stack:** Python 3.14, Plotly, Dash 4.0.0

**Test command:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -x -q --tb=short`

**Full suite:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -5`

---

## Design

### How symmetric banding works

Given sorted selected quantiles (e.g. `[0.01, 0.15, 0.50, 0.85, 0.99]`):

1. **Pair from outside in:** `(0.01, 0.99)` is the outer band, `(0.15, 0.85)` is the inner band. Q50% is the center line (unpaired if odd count).

2. **Up to 2 shaded regions:** outer band gets lighter opacity, inner band gets darker.
   - 2 quantiles → 1 band (outer only)
   - 3 quantiles → 1 band (outer, center is unpaired)
   - 4 quantiles → 2 bands (outer + inner)
   - 5 quantiles → 2 bands (outer + inner, center unpaired)

3. **Color from model:** Use `MODEL_TRACE_COLORS[model_key]` (e.g. black for BM, gold for QR). Same color for both bands, opacity differentiates.

4. **Opacity:** Outer band = 0.08, inner band = 0.15. (Lighter outer, darker inner — inner band is the "confidence core".)

### Symmetric quantile trace colors

Quantiles equidistant from 0.5 get the same color. The thermal palette already provides this symmetry visually (blue↔orange) but the mapping is asymmetric in practice. Instead of changing the palette, we simply use the same color for mirror pairs in the trace lines:
- Q1% and Q99% → same color
- Q15% and Q85% → same color
- Q50% → median color (gray)

This is done by mapping `q` to `min(q, 1-q)` before looking up the thermal color.

---

## File Structure

### Modified Files
| File | Change |
|------|--------|
| `btc_web/figures/bubble.py` | Replace adjacent-pair shading with `_build_symmetric_bands()` |
| `btc_web/figures/common.py` | Add `_symmetric_thermal_color()` helper |
| `btc_web/test_web.py` | Tests |

---

### Task 1: Symmetric band shading function

**Files:**
- Modify: `btc_web/figures/bubble.py`
- Modify: `btc_web/figures/common.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write failing tests**

```python
class TestSymmetricBandShading:
    def test_symmetric_bands_5_quantiles(self):
        """5 quantiles → 2 bands (outer + inner)."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.01, 0.15, 0.50, 0.85, 0.99]
        prices = {q: np.linspace(100 * (1 + q), 200 * (1 + q), 10) for q in qs}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#000000")
        # 2 bands × 2 traces each (lower boundary + fill) = 4 traces
        assert len(traces) == 4

    def test_symmetric_bands_3_quantiles(self):
        """3 quantiles → 1 band (outer only)."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.15, 0.50, 0.85]
        prices = {q: np.linspace(100, 200, 10) for q in qs}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#FF0000")
        assert len(traces) == 2  # 1 band = 2 traces

    def test_symmetric_bands_2_quantiles(self):
        """2 quantiles → 1 band."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.15, 0.85]
        prices = {q: np.linspace(100, 200, 10) for q in qs}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#000000")
        assert len(traces) == 2

    def test_symmetric_bands_1_quantile(self):
        """1 quantile → 0 bands (can't shade)."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.50]
        prices = {0.50: np.linspace(100, 200, 10)}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#000000")
        assert len(traces) == 0

    def test_symmetric_bands_outer_lighter_than_inner(self):
        """Outer band should have lower opacity than inner."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.01, 0.15, 0.50, 0.85, 0.99]
        prices = {q: np.linspace(100, 200, 10) for q in qs}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#000000")
        # traces[1] is outer fill, traces[3] is inner fill
        outer_fill = traces[1].fillcolor
        inner_fill = traces[3].fillcolor
        # Extract alpha values
        outer_alpha = float(outer_fill.split(",")[-1].rstrip(")"))
        inner_alpha = float(inner_fill.split(",")[-1].rstrip(")"))
        assert outer_alpha < inner_alpha


class TestSymmetricQuantileColors:
    def test_mirror_quantiles_same_color(self):
        """Q15% and Q85% should get the same color."""
        from figures.common import _symmetric_thermal_color
        c15 = _symmetric_thermal_color(0.15)
        c85 = _symmetric_thermal_color(0.85)
        assert c15 == c85

    def test_q50_gets_median_color(self):
        """Q50% should get the median (gray) color."""
        from figures.common import _symmetric_thermal_color
        c50 = _symmetric_thermal_color(0.50)
        assert c50 == "#bdbdbd"  # median gray from thermal palette
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestSymmetricBandShading -x -q --tb=short`

- [ ] **Step 3: Add `_symmetric_thermal_color()` to `btc_web/figures/common.py`**

Add after `_build_thermal_colors()`:

```python
def _symmetric_thermal_color(q: float, palette=None) -> str:
    """Map a quantile to a symmetric thermal color.

    Mirror quantiles about 0.5 so Q15% and Q85% get the same color.
    Uses min(q, 1-q) to look up the "distance from median" color.
    """
    mirror = min(q, 1.0 - q)
    return _thermal_color(mirror, palette)


def _build_symmetric_thermal_colors(quantiles: list, palette=None) -> dict:
    """Build {quantile: hex_color} dict with symmetric colors about Q50%."""
    return {q: _symmetric_thermal_color(q, palette) for q in quantiles}
```

- [ ] **Step 4: Add `_build_symmetric_bands()` to `btc_web/figures/bubble.py`**

Add near the top (after imports, before `build_bubble_figure`):

```python
def _build_symmetric_bands(sel_qs, price_cache, t_arr, model_color="#000000",
                            max_bands=2):
    """Build shaded band traces from symmetric quantile pairs.

    Pairs quantiles from outside in: (lowest, highest), (2nd lowest, 2nd highest).
    Up to max_bands bands. Outer band gets lighter opacity, inner gets darker.

    Args:
        sel_qs: sorted list of selected quantiles
        price_cache: {q: price_array} for each q in sel_qs
        t_arr: time array (x-axis values)
        model_color: hex color for fill (from MODEL_TRACE_COLORS)
        max_bands: max number of shaded regions (default 2)

    Returns list of go.Scatter traces (2 per band: lower boundary + fill).
    """
    if len(sel_qs) < 2:
        return []

    # Pair from outside in
    n = len(sel_qs)
    pairs = []
    for i in range(n // 2):
        lo_q = sel_qs[i]
        hi_q = sel_qs[n - 1 - i]
        if lo_q != hi_q and lo_q in price_cache and hi_q in price_cache:
            pairs.append((lo_q, hi_q))
    pairs = pairs[:max_bands]

    if not pairs:
        return []

    # Opacity: outer = lighter, inner = darker
    opacities = [0.08, 0.15] if len(pairs) >= 2 else [0.10]

    traces = []
    x = list(t_arr)
    for i, (lo_q, hi_q) in enumerate(pairs):
        alpha = opacities[i] if i < len(opacities) else opacities[-1]
        lo_p = price_cache[lo_q]
        hi_p = price_cache[hi_q]
        traces.append(go.Scatter(
            x=x, y=list(lo_p), mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        traces.append(go.Scatter(
            x=x, y=list(hi_p), mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=_hex_alpha(model_color, alpha),
            showlegend=False, hoverinfo="skip",
        ))

    return traces
```

- [ ] **Step 5: Replace existing shading in `build_bubble_figure()`**

Replace the current adjacent-pair shading block (around lines 85-102):

```python
        if p.get("shade") and len(sel_qs) >= 2:
            for j in range(len(sel_qs) - 1):
                if sel_qs[j] not in _price_cache or sel_qs[j+1] not in _price_cache:
                    continue
                lo_p = _price_cache[sel_qs[j]]
                hi_p = _price_cache[sel_qs[j+1]]
                col  = _thermal.get(sel_qs[j], model.colors.get(sel_qs[j], "#888888"))
                traces.append(go.Scatter(
                    x=list(t_arr), y=list(lo_p),
                    mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ))
                traces.append(go.Scatter(
                    x=list(t_arr), y=list(hi_p),
                    mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=_hex_alpha(col, _SHADE_ALPHA),
                    showlegend=False, hoverinfo="skip",
                ))
```

With:

```python
        if p.get("shade") and len(sel_qs) >= 2:
            _model_color = _app_ctx.MODEL_TRACE_COLORS.get("bub", "#000000")
            traces.extend(_build_symmetric_bands(
                sel_qs, _price_cache, t_arr, model_color=_model_color))
```

- [ ] **Step 6: Use symmetric colors for quantile trace lines**

Replace the `_thermal` color lookup with symmetric version. Change:
```python
    _thermal = _build_thermal_colors(sel_qs, palette)
```
to:
```python
    _thermal = _build_symmetric_thermal_colors(sel_qs, palette)
```

And similarly after the Q50% fallback:
```python
    if _fallback_q50:
        sel_qs = [0.5]
        _thermal = _build_symmetric_thermal_colors(sel_qs, palette)
```

Import `_build_symmetric_thermal_colors` from `figures.common`.

- [ ] **Step 7: Run test to verify it passes**

- [ ] **Step 8: Run full test suite**

- [ ] **Step 9: Commit**

```bash
git add btc_web/figures/bubble.py btc_web/figures/common.py btc_web/test_web.py
git commit -m "feat(bubble): symmetric band shading with model colors + symmetric quantile trace colors"
```

---

### Task 2: Apply symmetric shading to overlay models

**Files:**
- Modify: `btc_web/figures/bubble.py`
- Test: `btc_web/test_web.py`

Overlay models (PL, QR, LPPL, etc.) currently don't shade. When they're active with 2+ quantiles selected, add symmetric bands using each model's own trace color.

- [ ] **Step 1: Write test**

```python
class TestOverlayModelShading:
    def test_overlay_model_bands_use_model_color(self):
        """Overlay model bands should use that model's trace color."""
        from figures.bubble import build_bubble_figure
        import _app_ctx
        M = _app_ctx.M
        p = dict(selected_qs=[0.15, 0.50, 0.85], shade=True,
                 xscale="log", yscale="log",
                 xmin=2012, xmax=2030, ymin=0, ymax=7, n_future=3,
                 show_comp=False, show_ols=False, show_data=False,
                 show_today=False, pt_size=2, pt_alpha=0.3,
                 stack=0, show_stack=False, lots=[], use_lots=False,
                 show_legend=False, active_models=["bub", "pl"],
                 qs_mode=[])
        fig = build_bubble_figure(M, p)
        # Should have band traces for both BM (black) and PL (cyan)
        fill_traces = [t for t in fig.data if t.fill == "tonexty"]
        assert len(fill_traces) >= 2  # at least 1 per model
```

- [ ] **Step 2: Add overlay model shading**

In `build_bubble_figure()`, in the overlay model loop (around line 170-183), after the quantile traces are added, add:

```python
        if mdl.quantized and p.get("shade") and len(overlay_qs) >= 2:
            _overlay_prices = {}
            for q in overlay_qs:
                if q in mdl.fits:
                    _overlay_prices[q] = _round_trace_data(
                        mdl.price_at(q, t_arr) * (stack if stack > 0 else 1))
            _overlay_color = _app_ctx.MODEL_TRACE_COLORS.get(model_key, "#888888")
            traces.extend(_build_symmetric_bands(
                sorted(overlay_qs), _overlay_prices, t_arr,
                model_color=_overlay_color))
```

- [ ] **Step 3: Run tests — verify PASS**

- [ ] **Step 4: Run full suite**

- [ ] **Step 5: Commit**

```bash
git add btc_web/figures/bubble.py btc_web/test_web.py
git commit -m "feat(bubble): add symmetric band shading to overlay models"
```

---

### Task 3: Color-coded model labels in Display Models checklist

**Files:**
- Modify: `btc_web/layout/bubble.py`
- Test: `btc_web/test_web.py`

Replace plain text model labels with colored box + label. Each model name sits on top of a small rectangle filled with the model's shade color (from `MODEL_TRACE_COLORS`).

- [ ] **Step 1: Write test**

```python
class TestColorCodedModelLabels:
    def test_model_labels_have_color_swatch(self):
        """Display Models labels should have a colored box."""
        from layout.bubble import _bubble_controls
        import json
        layout_str = json.dumps(_bubble_controls().to_plotly_json())
        # Should contain inline style with backgroundColor for BM
        assert "backgroundColor" in layout_str
```

- [ ] **Step 2: Modify Display Models checklist in `btc_web/layout/bubble.py`**

Replace the current `dcc.Checklist(id="bub-model-show", ...)` with one that uses colored label spans:

```python
            _lbl("Display models"),
            dcc.Checklist(id="bub-model-show",
                          options=[
                              {"label": html.Span([
                                  html.Span(" ", style={
                                      "display": "inline-block", "width": "12px",
                                      "height": "12px", "borderRadius": "2px",
                                      "backgroundColor": _app_ctx.MODEL_TRACE_COLORS.get("bub", "#000"),
                                      "verticalAlign": "middle", "marginRight": "4px",
                                  }),
                                  "Bubble Model",
                              ]), "value": "bub"},
                          ] + [
                              {"label": html.Span([
                                  html.Span(" ", style={
                                      "display": "inline-block", "width": "12px",
                                      "height": "12px", "borderRadius": "2px",
                                      "backgroundColor": _app_ctx.MODEL_TRACE_COLORS.get(mdl.short_name, "#888"),
                                      "verticalAlign": "middle", "marginRight": "4px",
                                  }),
                                  mdl.name,
                              ]), "value": mdl.short_name}
                              for mdl in _app_ctx.PRICE_MODELS.values()
                              if mdl.short_name not in _app_ctx.MODEL_SENTINELS
                              and mdl.short_name != "bub"
                          ],
                          value=["bub"], inline=True,
                          inputStyle=_CB_MARGIN,
                          labelStyle={"marginRight": "12px", "fontSize": "11px"},
                          style={"marginBottom": "8px"}),
```

- [ ] **Step 3: Run tests — verify PASS**

- [ ] **Step 4: Run full suite**

- [ ] **Step 5: Commit**

```bash
git add btc_web/layout/bubble.py btc_web/test_web.py
git commit -m "feat(bubble): color-coded model swatches in Display Models checklist"
```

---

## Verification Checklist

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -10
```

## Notes

- Symmetric colors: `_symmetric_thermal_color(q)` maps `q` to `min(q, 1-q)` before thermal lookup. So Q15% and Q85% both map to 0.15 on the thermal scale → same blue-ish color.
- Band shading uses `MODEL_TRACE_COLORS` not thermal — BM=black, PL=cyan, QR=gold, etc.
- Max 2 bands per model (outer + inner). With 5 quantiles: (Q1%↔Q99%) outer, (Q15%↔Q85%) inner.
- Outer band opacity 0.08, inner 0.15 — lighter outer gives a gradient effect.
- The `shade` toggle (`bub-toggles`) controls whether bands appear — same UX as before.
- Existing `_SHADE_ALPHA` constant is no longer used (replaced by per-band opacity). Can be removed in cleanup.
