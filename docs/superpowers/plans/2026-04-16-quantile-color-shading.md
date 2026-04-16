# Quantile Color Shading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace opacity-based quantile trace fading with HSL lightness-tinted model-color variants so every quantile line reads as a shade of its model color, symmetric around Q50, lighter at extremes.

**Architecture:** New pure function `quantile_shade(base_hex, q)` in `colors.py` computes per-quantile hex colors via HSL lightness interpolation. Every figure builder site that currently uses `quantile_opacity(q)` + Plotly `opacity` switches to `quantile_shade(model_color, q)` with `opacity=1.0`. Band fills gain a `fill_mode` parameter to support both alpha (current) and opaque pastel variants.

**Tech Stack:** Python `colorsys` (stdlib), Plotly/Dash figure builders, existing `colors.py` SSOT.

**Spec:** `docs/superpowers/specs/2026-04-16-quantile-color-shading-design.md`

---

## File Manifest

### Modify
| Path | Change |
|---|---|
| `btc_web/colors.py` | Add `import colorsys`, `_hex_to_rgb`, `quantile_shade`, 3 constants, `BAND_FILL_MODE`, `BAND_PASTEL_ALPHA`. Update `__skip_export__`. |
| `btc_web/figures/bubble.py` | Replace 2 `quantile_opacity` + `opacity=` sites with `quantile_shade` |
| `btc_web/figures/common.py` | Replace 1 `quantile_opacity` + `opacity=` site in `build_overlay_traces`; extend `_build_symmetric_bands` with `fill_mode` |
| `btc_web/figures/dca.py` | Replace 2 `quantile_opacity` + `opacity=` sites (BM lines, SC lines) |
| `btc_web/figures/retire.py` | Replace 1 `quantile_opacity` + `opacity=` site |
| `btc_web/figures/supercharge.py` | Replace 2 `quantile_opacity` + `opacity=` sites (layout 1 lines, mode B lines) |
| `btc_web/figures/heatmap.py` | Replace 2 `quantile_opacity` uses (excursion band + CAGR trace) |
| `btc_web/test_colors_central.py` | Add 6 unit tests for `quantile_shade` |

---

## Task 0: Add `quantile_shade` + constants to `colors.py`

**Files:**
- Modify: `btc_web/colors.py`
- Modify: `btc_web/test_colors_central.py`

- [ ] **Step 1:** Write 6 failing tests in `btc_web/test_colors_central.py`.

Append at end of file:

```python
# ── quantile_shade tests ─────────────────────────────────────────────────

def test_quantile_shade_median_returns_base():
    from colors import quantile_shade
    assert quantile_shade("#C48209", 0.5) == "#c48209"


def test_quantile_shade_symmetric():
    from colors import quantile_shade
    assert quantile_shade("#C48209", 0.1) == quantile_shade("#C48209", 0.9)
    assert quantile_shade("#1B3352", 0.25) == quantile_shade("#1B3352", 0.75)


def test_quantile_shade_monotone_lightening():
    import colorsys
    from colors import quantile_shade
    base = "#C48209"
    qs = [0.5, 0.25, 0.10, 0.01]
    lightnesses = []
    for q in qs:
        h_str = quantile_shade(base, q)
        r, g, b = int(h_str[1:3], 16), int(h_str[3:5], 16), int(h_str[5:7], 16)
        _, l, _ = colorsys.rgb_to_hls(r/255, g/255, b/255)
        lightnesses.append(l)
    for i in range(len(lightnesses) - 1):
        assert lightnesses[i] < lightnesses[i+1], (
            f"L should increase away from Q50: {list(zip(qs, lightnesses))}"
        )


def test_quantile_shade_returns_valid_hex():
    from colors import quantile_shade
    result = quantile_shade("#FF0000", 0.01)
    assert result.startswith("#")
    assert len(result) == 7
    int(result[1:], 16)  # should not raise


def test_quantile_shade_does_not_exceed_cap():
    import colorsys
    from colors import quantile_shade
    result = quantile_shade("#000000", 0.001)
    r, g, b = int(result[1:3], 16), int(result[3:5], 16), int(result[5:7], 16)
    _, l, _ = colorsys.rgb_to_hls(r/255, g/255, b/255)
    assert l <= 0.97 + 1e-6


def test_quantile_shade_all_palettes():
    import colorsys
    from colors import quantile_shade, PALETTES
    for pal_name, pal in PALETTES.items():
        mc = pal.get("model_colors", {})
        for key in ("bub", "pl", "qr", "eppl", "hybppl", "lppl"):
            color = mc.get(key)
            if not color:
                continue
            base_hex = quantile_shade(color, 0.5)
            ext_hex = quantile_shade(color, 0.01)
            rb, gb, bb = int(base_hex[1:3],16), int(base_hex[3:5],16), int(base_hex[5:7],16)
            re, ge, be = int(ext_hex[1:3],16), int(ext_hex[3:5],16), int(ext_hex[5:7],16)
            _, lb, _ = colorsys.rgb_to_hls(rb/255, gb/255, bb/255)
            _, le, _ = colorsys.rgb_to_hls(re/255, ge/255, be/255)
            assert le > lb, f"{pal_name}/{key}: Q01 should be lighter than Q50"
```

- [ ] **Step 2:** Run tests to verify they fail.

```bash
btc_venv/bin/python3 -m pytest btc_web/test_colors_central.py -k "quantile_shade" -v
```

Expected: FAIL — `ImportError: cannot import name 'quantile_shade' from 'colors'`.

- [ ] **Step 3:** Add `import colorsys` and the implementation to `btc_web/colors.py`.

At the top of the file (with other imports), add:

```python
import colorsys
```

In Section 4 (Utility functions), after the existing `_hex_alpha` function, add:

```python
def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert #rrggbb hex to (r, g, b) integer tuple."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
```

In Section 5 (Appearance constants), after the `Q_OPACITY_DECAY` line, add:

```python
# ── Quantile shade formula parameters (function below in Section 4) ──
Q_SHADE_STRENGTH    = 0.70   # max fraction of the gap (L_target - L_base) used
Q_SHADE_EXPONENT    = 0.80   # concavity: < 1 = inner quantiles shift more
Q_SHADE_L_TARGET    = 0.92   # HSL lightness ceiling extremes approach
BAND_FILL_MODE      = "alpha"  # "alpha" = current translucent fills; "pastel" = opaque tinted
BAND_PASTEL_ALPHA   = 0.35   # alpha for pastel band fills (only used when BAND_FILL_MODE="pastel")
```

Back in Section 4 (after `_hex_to_rgb`), add:

```python
def quantile_shade(base_hex: str, q: float) -> str:
    """Return a lightened variant of base_hex based on distance from Q50.

    Q50 returns the base color unchanged. Extremes (Q01/Q99) approach
    Q_SHADE_L_TARGET. The curve is concave (exponent < 1) so inner
    quantiles (Q25/Q75) get a noticeable but modest shift while
    extremes get the dramatic change.
    """
    r, g, b = _hex_to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    d = abs(q - 0.5) / 0.5
    factor = d ** Q_SHADE_EXPONENT
    l_new = l + (Q_SHADE_L_TARGET - l) * factor * Q_SHADE_STRENGTH
    l_new = min(l_new, 0.97)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l_new, s)
    return f"#{int(r2*255+.5):02x}{int(g2*255+.5):02x}{int(b2*255+.5):02x}"
```

Update `__skip_export__` — add the new constants that shouldn't become CSS variables:

```python
__skip_export__ = frozenset({
    # ... existing entries ...
    "Q_SHADE_STRENGTH", "Q_SHADE_EXPONENT", "Q_SHADE_L_TARGET",
    "BAND_FILL_MODE", "BAND_PASTEL_ALPHA",
})
```

- [ ] **Step 4:** Run tests to verify they pass.

```bash
btc_venv/bin/python3 -m pytest btc_web/test_colors_central.py -k "quantile_shade" -v
```

Expected: 6 PASSED.

- [ ] **Step 5:** Commit.

```bash
git add btc_web/colors.py btc_web/test_colors_central.py
git commit -m "feat(colors): add quantile_shade HSL lightness helper + 6 tests"
```

---

## Task 1: Extend `_build_symmetric_bands` with `fill_mode`

**Files:**
- Modify: `btc_web/figures/common.py`

- [ ] **Step 1:** Modify `_build_symmetric_bands` in `btc_web/figures/common.py` (around line 807).

Change the function signature and body from:

```python
def _build_symmetric_bands(sel_qs, y_cache, x_arr, model_color=BLACK,
                            max_bands=2):
```

to:

```python
def _build_symmetric_bands(sel_qs, y_cache, x_arr, model_color=BLACK,
                            max_bands=2, fill_mode=None):
```

Inside the function, after `opacities = [0.08, 0.15] if len(pairs) >= 2 else [0.10]`, add:

```python
    if fill_mode is None:
        from colors import BAND_FILL_MODE
        fill_mode = BAND_FILL_MODE
```

Replace the existing fill color line inside the loop:

```python
            fillcolor=_hex_alpha(model_color, alpha),
```

with:

```python
            fillcolor=(_hex_alpha(quantile_shade(model_color, (lo_q + hi_q) / 2),
                                   BAND_PASTEL_ALPHA)
                       if fill_mode == "pastel"
                       else _hex_alpha(model_color, alpha)),
```

And add this import at the top of the function body (or at file-level):

```python
    from colors import quantile_shade, BAND_PASTEL_ALPHA
```

- [ ] **Step 2:** Verify smoke import.

```bash
cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "from figures.common import _build_symmetric_bands; print('OK')"
```

- [ ] **Step 3:** Commit.

```bash
git add btc_web/figures/common.py
git commit -m "feat(bands): add fill_mode param to _build_symmetric_bands (alpha/pastel)"
```

---

## Task 2: Replace `quantile_opacity` in `figures/bubble.py`

**Files:**
- Modify: `btc_web/figures/bubble.py`

- [ ] **Step 1:** Add `quantile_shade` to the imports.

Find the line importing `quantile_opacity` from `figures.common`:

```python
    quantile_opacity,
```

Add below it:

```python
from colors import quantile_shade
```

- [ ] **Step 2:** Replace BM quantile line opacity (around line 131–138).

Change:

```python
            _q_opacity = quantile_opacity(q)
            if _fallback_q50 and _default_mode:
                _q_opacity = _app_ctx.FALLBACK_Q50_OPACITY
            traces.append(go.Scatter(
                x=list(t_arr), y=list(prices),
                mode="lines", name=lbl,
                line=dict(color=_bub_color, width=_QR_LINE_WIDTH),
                opacity=_q_opacity,
            ))
```

to:

```python
            _shade = quantile_shade(_bub_color, q)
            _trace_opacity = 1.0
            if _fallback_q50 and _default_mode:
                _trace_opacity = _app_ctx.FALLBACK_Q50_OPACITY
            traces.append(go.Scatter(
                x=list(t_arr), y=list(prices),
                mode="lines", name=lbl,
                line=dict(color=_shade, width=_QR_LINE_WIDTH),
                opacity=_trace_opacity,
            ))
```

- [ ] **Step 3:** Replace overlay quantile line opacity (around line 191–197).

Change:

```python
                _q_opacity = quantile_opacity(q)
                _lw = 3.0 if (model_key == "u1" and hasattr(mdl, 'own_quantile') and abs(q - mdl.own_quantile) < 0.005) else _OVERLAY_LINE_WIDTH
                traces.append(go.Scatter(
                    x=list(t_arr), y=list(prices),
                    mode="lines", name=lbl,
                    line=dict(color=_ovl_color, width=_lw, dash=mdl.dash_style),
                    opacity=_q_opacity,
```

to:

```python
                _shade = quantile_shade(_ovl_color, q)
                _lw = 3.0 if (model_key == "u1" and hasattr(mdl, 'own_quantile') and abs(q - mdl.own_quantile) < 0.005) else _OVERLAY_LINE_WIDTH
                traces.append(go.Scatter(
                    x=list(t_arr), y=list(prices),
                    mode="lines", name=lbl,
                    line=dict(color=_shade, width=_lw, dash=mdl.dash_style),
```

(Remove the `opacity=_q_opacity` line entirely — Plotly defaults to 1.0.)

- [ ] **Step 4:** Commit.

```bash
git add btc_web/figures/bubble.py
git commit -m "feat(bubble): quantile traces use quantile_shade instead of opacity fade"
```

---

## Task 3: Replace `quantile_opacity` in `figures/common.py::build_overlay_traces`

**Files:**
- Modify: `btc_web/figures/common.py`

- [ ] **Step 1:** Add `quantile_shade` import.

Near the top of `figures/common.py`, where other `colors` imports are, add `quantile_shade`:

```python
from colors import (
    ...,
    quantile_shade,
)
```

(Or add it to the existing `from colors import (...)` block.)

- [ ] **Step 2:** Replace the overlay quantile line site (around line 1222–1228).

Change:

```python
                _q_opacity = quantile_opacity(q)
                _model_lines.append(go.Scatter(
                    x=list(ts), y=list(y_vals), mode="lines",
                    name=f"{mdl.legend_name} {_fmt_q_label(q, '')}  \u2192  {final_lbl}",
                    line=dict(color=_mdl_color, width=_OVERLAY_LINE_WIDTH,
                              dash=mdl.dash_style, shape=line_shape),
                    opacity=_q_opacity,
```

to:

```python
                _shade = quantile_shade(_mdl_color, q)
                _model_lines.append(go.Scatter(
                    x=list(ts), y=list(y_vals), mode="lines",
                    name=f"{mdl.legend_name} {_fmt_q_label(q, '')}  \u2192  {final_lbl}",
                    line=dict(color=_shade, width=_OVERLAY_LINE_WIDTH,
                              dash=mdl.dash_style, shape=line_shape),
```

(Remove `opacity=_q_opacity` line.)

- [ ] **Step 3:** Commit.

```bash
git add btc_web/figures/common.py
git commit -m "feat(overlays): quantile traces use quantile_shade in build_overlay_traces"
```

---

## Task 4: Replace `quantile_opacity` in `figures/dca.py`

**Files:**
- Modify: `btc_web/figures/dca.py`

- [ ] **Step 1:** Add import. Add `from colors import quantile_shade` near the other color imports.

- [ ] **Step 2:** Replace BM DCA line site (around line 220–224).

Change:

```python
            _q_opacity = quantile_opacity(q)
            _bm_line_traces.append(go.Scatter(
                x=list(ts), y=list(y_vals), mode="lines", name=lbl,
                line=dict(color=_bm_color, width=_QR_LINE_WIDTH, shape=_line_shape),
                opacity=_q_opacity,
            ))
```

to:

```python
            _shade = quantile_shade(_bm_color, q)
            _bm_line_traces.append(go.Scatter(
                x=list(ts), y=list(y_vals), mode="lines", name=lbl,
                line=dict(color=_shade, width=_QR_LINE_WIDTH, shape=_line_shape),
            ))
```

- [ ] **Step 3:** Replace SC (Stack-celerator) line site (around line 152–157).

Change:

```python
        _q_opacity = quantile_opacity(q)
        sc_traces.append(go.Scatter(
            x=list(ts), y=list(y_sc), mode="lines", name=lbl_sc,
            line=dict(color=_bm_color, width=_QR_LINE_WIDTH, dash="dash", shape=line_shape),
            opacity=_q_opacity,
        ))
```

to:

```python
        _shade = quantile_shade(_bm_color, q)
        sc_traces.append(go.Scatter(
            x=list(ts), y=list(y_sc), mode="lines", name=lbl_sc,
            line=dict(color=_shade, width=_QR_LINE_WIDTH, dash="dash", shape=line_shape),
        ))
```

- [ ] **Step 4:** Commit.

```bash
git add btc_web/figures/dca.py
git commit -m "feat(dca): quantile traces use quantile_shade instead of opacity fade"
```

---

## Task 5: Replace `quantile_opacity` in `figures/retire.py`

**Files:**
- Modify: `btc_web/figures/retire.py`

- [ ] **Step 1:** Add import. Add `from colors import quantile_shade` near the other color imports.

- [ ] **Step 2:** Replace BM retire line site (around line 82–87).

Change:

```python
            _q_opacity = quantile_opacity(q)
            lbl = f"{model.legend_name} {_fmt_q_label(q, '')}" + f"  \u2192  {final_lbl}"
            _bm_trace_traces.append(go.Scatter(
                x=list(ts), y=list(y_vals), mode="lines", name=lbl,
                line=dict(color=_bm_color, width=_QR_LINE_WIDTH, shape=_line_shape),
                opacity=_q_opacity,
            ))
```

to:

```python
            _shade = quantile_shade(_bm_color, q)
            lbl = f"{model.legend_name} {_fmt_q_label(q, '')}" + f"  \u2192  {final_lbl}"
            _bm_trace_traces.append(go.Scatter(
                x=list(ts), y=list(y_vals), mode="lines", name=lbl,
                line=dict(color=_shade, width=_QR_LINE_WIDTH, shape=_line_shape),
            ))
```

- [ ] **Step 3:** Commit.

```bash
git add btc_web/figures/retire.py
git commit -m "feat(retire): quantile traces use quantile_shade instead of opacity fade"
```

---

## Task 6: Replace `quantile_opacity` in `figures/supercharge.py`

**Files:**
- Modify: `btc_web/figures/supercharge.py`

- [ ] **Step 1:** Add import. Add `from colors import quantile_shade` near the other color imports.

- [ ] **Step 2:** Replace layout-1 line site (around line 204–212).

Change:

```python
                    _q_opacity = quantile_opacity(q)
                    traces.append(go.Scatter(
                        x=list(ts_d), y=list(y_vals), mode="lines",
                        name=_legend_name,
                        legendgroup=grp_model,
                        showlegend=_first_legend,
                        line=dict(color=_bm_color, width=_QR_LINE_WIDTH,
                                  dash=_DASH_STYLES[di % len(_DASH_STYLES)], shape=_line_shape),
                        opacity=_q_opacity,
                    ))
```

to:

```python
                    _shade = quantile_shade(_bm_color, q)
                    traces.append(go.Scatter(
                        x=list(ts_d), y=list(y_vals), mode="lines",
                        name=_legend_name,
                        legendgroup=grp_model,
                        showlegend=_first_legend,
                        line=dict(color=_shade, width=_QR_LINE_WIDTH,
                                  dash=_DASH_STYLES[di % len(_DASH_STYLES)], shape=_line_shape),
                    ))
```

- [ ] **Step 3:** Replace mode-B line site (around line 562–571).

Change:

```python
            _q_opacity = quantile_opacity(q)
            y_q   = [max_wd.get((d, q), 0) for d in delays]
            traces.append(go.Scatter(
                x=delays, y=y_q, mode="lines+markers",
                name=f"{model.legend_name} {q_range}",
                legendgroup=grp,
                showlegend=(qi == 0),
                line=dict(color=_bm_color, width=2),
                marker=dict(color=_bm_color, size=7),
                opacity=_q_opacity,
            ))
```

to:

```python
            _shade = quantile_shade(_bm_color, q)
            y_q   = [max_wd.get((d, q), 0) for d in delays]
            traces.append(go.Scatter(
                x=delays, y=y_q, mode="lines+markers",
                name=f"{model.legend_name} {q_range}",
                legendgroup=grp,
                showlegend=(qi == 0),
                line=dict(color=_shade, width=2),
                marker=dict(color=_shade, size=7),
            ))
```

- [ ] **Step 4:** Commit.

```bash
git add btc_web/figures/supercharge.py
git commit -m "feat(supercharge): quantile traces use quantile_shade instead of opacity fade"
```

---

## Task 7: Replace `quantile_opacity` in `figures/heatmap.py`

**Files:**
- Modify: `btc_web/figures/heatmap.py`

- [ ] **Step 1:** Add import. Add `from colors import quantile_shade` near the other color imports.

- [ ] **Step 2:** Replace CAGR excursion band + trace sites (around line 564–596).

Change:

```python
            _q_opacity = quantile_opacity(q)
```

(around line 564) to:

```python
            _shade = quantile_shade(color, q)
```

Then change the excursion band fill (around line 578):

```python
                fillcolor=_hex_alpha(color, 0.2 * _q_opacity),
```

to:

```python
                fillcolor=_hex_alpha(_shade, 0.2),
```

Then change the CAGR trace (around line 594–596):

```python
            traces.append(go.Scatter(
                x=years,
                y=cagrs,
                mode="lines",
                name=lbl,
                line=dict(color=color, width=2),
                opacity=_q_opacity,
```

to:

```python
            traces.append(go.Scatter(
                x=years,
                y=cagrs,
                mode="lines",
                name=lbl,
                line=dict(color=_shade, width=2),
```

(Remove `opacity=_q_opacity` line.)

- [ ] **Step 3:** Commit.

```bash
git add btc_web/figures/heatmap.py
git commit -m "feat(heatmap): quantile traces use quantile_shade instead of opacity fade"
```

---

## Task 8: Full test suite + dev server smoke test

- [ ] **Step 1:** Run the full non-E2E test suite.

```bash
btc_venv/bin/python3 -m pytest btc_web/ --ignore-glob='*_e2e.py' 2>&1 | tail -20
```

Expected: all new tests pass; no new regressions beyond existing ~32 pre-existing failures.

- [ ] **Step 2:** Start dev server and visually verify Tab 1.

```bash
lsof -ti :8050 2>/dev/null | xargs -r kill -9
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
curl -s http://localhost:8050/health | grep -oE '"status":"[^"]*"'
```

Open http://localhost:8050/1 in browser. Verify:
1. Q50 line matches model swatch in Display Models checklist.
2. Q25/Q75 are visibly lighter but same hue.
3. Q01/Q99 are clearly lighter — pastel, not washed out.
4. Two different models (BM + PL) have distinguishable Q25 lines.
5. CB-Brian palette: same behavior, no crashes.

- [ ] **Step 3:** Test band fill mode toggle.

Edit `btc_web/colors.py`, change `BAND_FILL_MODE = "alpha"` to `BAND_FILL_MODE = "pastel"`. Restart dev server. Verify band fills are opaque tinted instead of translucent alpha.

Revert to `"alpha"` (or leave as user preference).

- [ ] **Step 4:** Stop dev server and commit.

```bash
lsof -ti :8050 2>/dev/null | xargs -r kill -9
```

---

## Task 9: Deploy

- [ ] **Step 1:** Verify clean local state.

```bash
git status --short
```

- [ ] **Step 2:** Push and deploy.

```bash
git push origin master
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"
```

- [ ] **Step 3:** Verify prod `/health`.

```bash
sleep 25
curl -s https://quantoshi.xyz/health | python3 -m json.tool | grep status
```

---

## Self-review checklist

- **Spec coverage:** §3.1 → Task 0, §3.2 → Task 0, §3.3 → Tasks 0+1, §3.4 → Tasks 2–7, §3.5 → Task 0, §4.1 → Task 0, §4.2 → Task 8, §5 → Tasks 8+9.
- **No placeholders:** all code blocks are complete.
- **Type consistency:** `quantile_shade(base_hex: str, q: float) -> str` used consistently across Tasks 2–7; `fill_mode` param in Task 1 matches `BAND_FILL_MODE` from Task 0.
- **Call sites verified:** `grep quantile_opacity btc_web/figures/` finds 11 usage sites across 6 files — all covered by Tasks 2–7.
