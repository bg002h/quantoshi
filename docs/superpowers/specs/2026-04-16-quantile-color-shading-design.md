# Quantile Color Shading — Design Spec

## 1. Problem

Quantile traces currently use a single model hex color + Plotly `opacity` proportional to distance from Q50. On the cream chart background, low-opacity lines lose their color identity — Q10 traces from different models converge to similar pale tints, and extreme quantiles look "vastly different from model color". Past attempts to fix this via thermal gradients or per-quantile color assignments failed because they broke model-color association.

## 2. Goal

Every quantile trace should read as **a shade of the model color**, symmetric around Q50, with full Plotly opacity (1.0). The median line shows the base model color; extremes are progressively lighter tints of the same hue. Adjacent quantiles from different models remain distinguishable because their hues differ.

User requirements:
- **Lighter at extremes** (not darker). Q50 = base, Q01/Q99 lightest.
- **Strong variation** — ~45% lightness delta at Q01/Q99 relative to base.
- **Symmetric** — Q10 and Q90 produce the identical shade.
- **All 5 chart tabs** — Bubble, DCA, Retire, Supercharger, Citadel.
- **Both band-fill variants prepared** — (A) translucent alpha fills (current), and (B) opaque pastel fills using the same lightness curve — user picks at review time.

## 3. Architecture

### 3.1 New helper: `quantile_shade(base_hex, q) -> str`

Location: `btc_web/colors.py`, Section 5 (Appearance constants).

Pure function. Converts `base_hex` → HSL, computes lightened variant, returns new hex string.

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
    d = abs(q - 0.5) / 0.5  # 0.0 at median → 1.0 at extremes
    factor = d ** Q_SHADE_EXPONENT
    l_new = l + (Q_SHADE_L_TARGET - l) * factor * Q_SHADE_STRENGTH
    l_new = min(l_new, 0.97)  # hard cap to avoid near-white
    r2, g2, b2 = colorsys.hls_to_rgb(h, l_new, s)
    return f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"
```

### 3.2 New constants in `colors.py` Section 5

```python
Q_SHADE_STRENGTH    = 0.70   # max fraction of the gap (L_target - L_base) used
Q_SHADE_EXPONENT    = 0.80   # concavity: < 1 = inner quantiles shift more
Q_SHADE_L_TARGET    = 0.92   # HSL lightness ceiling extremes approach
```

Worked example for BM amber `#C48209`:
- Base HSL: H=38, S=0.88, L=0.40
- Q50: L=0.40 → `#C48209` (unchanged)
- Q25/Q75: d=0.5, factor=0.5^0.8=0.574, L=0.40 + (0.92-0.40)×0.574×0.70 = 0.61 → medium amber tint
- Q10/Q90: d=0.8, factor=0.8^0.8=0.837, L=0.40 + 0.52×0.837×0.70 = 0.70 → light amber
- Q01/Q99: d=0.98, factor=0.98^0.8=0.984, L=0.40 + 0.52×0.984×0.70 = 0.76 → pale amber

All still clearly amber. Visually distinct from PL navy `#1B3352` going through its own tinting curve.

### 3.3 Band fill variants

Both variants are implemented; a single constant `BAND_FILL_MODE` in `colors.py` selects between them. Default: `"alpha"` (current behavior). Alternative: `"pastel"`.

**Variant A — alpha (default, current):** `_build_symmetric_bands` continues using `fillcolor=rgba(base, 0.08/0.15)`. Lines sit on top at full opacity with lightness-shaded colors.

**Variant B — pastel:** `_build_symmetric_bands` uses `fillcolor=quantile_shade(base, mid_q)` where `mid_q = (lo_q + hi_q) / 2` is the midpoint quantile of the band pair, at a fixed alpha (`BAND_PASTEL_ALPHA = 0.35`). The fill color itself is a lightened variant of the model color rather than a transparent layer over the background.

`_build_symmetric_bands` gains an optional `fill_mode` parameter, defaulting to `BAND_FILL_MODE`.

### 3.4 Replacement in figure builders

Every site that currently does:
```python
opacity=quantile_opacity(q)
line=dict(color=model_color, ...)
```
becomes:
```python
line=dict(color=quantile_shade(model_color, q), ...)
# opacity omitted (defaults to 1.0)
```

**Call sites (5 tabs):**

| File | Builder function | Sites |
|---|---|---|
| `figures/bubble.py` | `build_bubble_figure` | BM quantile lines (~L137), overlay quantile lines (~L196), overlay band fills (~L209), scanner quantile lines (~L265) |
| `figures/common.py` | `build_overlay_traces` | Overlay quantile lines (~L1223), overlay band fills (~L1236) |
| `figures/dca.py` | `build_dca_figure` | (uses `build_overlay_traces`) |
| `figures/retire.py` | `build_retire_figure` | (uses `build_overlay_traces`) |
| `figures/supercharge.py` | `build_supercharge_figure` | (uses `build_overlay_traces`) |
| `figures/citadel.py` | `build_citadel_figure` | Quantile lines (grep `quantile_opacity`) |

### 3.5 What does NOT change

- `quantile_opacity()` — retained for the quantile-panel sidebar dots (layout/common.py `_q_options`). Not used for chart traces after this change.
- `_get_model_color()` — still returns the base color per model per palette. `quantile_shade` wraps it for per-quantile variants.
- Snapshot, routing, tab_defaults — no new controls.
- CB-Brian / CB-RG / CB-Full palettes — `quantile_shade` works on any base hex. Palette-specific colors are automatically tinted. No per-palette tuning needed.
- Heatmap — does not draw quantile traces (uses a heatmap grid). Unchanged.

## 4. Testing

### 4.1 Unit tests for `quantile_shade`

In `btc_web/test_colors_central.py` (existing test file for colors.py):

- `test_quantile_shade_median_returns_base` — `quantile_shade(hex, 0.5)` == `hex`.
- `test_quantile_shade_symmetric` — `quantile_shade(hex, 0.1)` == `quantile_shade(hex, 0.9)`.
- `test_quantile_shade_monotone_lightening` — L increases as |q-0.5| increases.
- `test_quantile_shade_returns_valid_hex` — 7-char string starting with `#`.
- `test_quantile_shade_does_not_exceed_cap` — L never exceeds 0.97.
- `test_quantile_shade_all_palettes` — iterate all 4 palettes × flagship 6 model colors, verify no crash and L(Q01) > L(Q50).

### 4.2 Visual verification

Start dev server, toggle each model on Tab 1, confirm:
1. Q50 line matches the model swatch color in the Display Models checklist.
2. Q25/Q75 are visibly lighter but recognizably the same hue.
3. Q01/Q99 are clearly lighter still — pastel-ish but not washed out.
4. Two different models' Q25 lines (e.g., BM amber and PL navy) are easily distinguishable.
5. Switching to CB-Brian palette: same behavior, no crashes, colors still distinct.
6. Band fills: check both `BAND_FILL_MODE="alpha"` and `"pastel"` in dev.

## 5. Rollout

- No new UI controls (no radio, no dropdown).
- `BAND_FILL_MODE` defaults to `"alpha"` (visual parity with current band fills; only line colors change). User tests, then we flip to `"pastel"` if preferred.
- Generated color artifacts (`_colors_generated.css/js`) — `Q_SHADE_*` constants are NOT exported to CSS/JS (chart-side only, Python computes). Add to `__skip_export__` if auto-export picks them up.
- Deploy: code change only, no pkl rebuild, no Redis flush needed (figure cache keys don't change — same params, just different visual output from the builder).

## 6. Non-goals

- Per-quantile color customization UI.
- Darkening variant (user chose "lighter at extremes").
- Heatmap quantile styling (no quantile traces there).
- Changing the quantile panel sidebar dot colors (those stay on `DIM_TEXT + quantile_opacity`).
