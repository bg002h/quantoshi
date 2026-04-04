# Unified Color System — Design Document

## Current State

The color system has 5+ independent sources, only ~40% respond to palette switching:

| Category | Source | Palette-aware? |
|---|---|---|
| BM quantile trace lines | Thermal palette | Yes |
| Quantile checkbox dots | `M.qr_colors` (baked at startup) | No |
| Band shading fill | `MODEL_TRACE_COLORS` | No |
| Overlay model trace lines | Per-model `_build_colors()` | No |
| Display Models swatches | `MODEL_TRACE_COLORS` | No |
| Non-quantized model line | `palette["non_quantized_model"]` | Yes |
| Heatmap colorscale defaults | `palette["hm_c_lo/mid1/mid2/hi"]` | Yes |
| Heatmap loss/exceptional text | `palette["hm_loss_text/hm_exceptional_text"]` | Yes |
| Heatmap named palettes (Forge, Thermal, etc.) | Hardcoded in callback | No |
| Today/delay/annot colors | Palette keys | Yes |
| OLS, UCL, U1, EF overlays | Hardcoded | No |

## Color Vocabulary

The app uses color for 4 distinct purposes:

1. **Model identity** — Which model generated this trace/band? (BM=black, PL=cyan, etc.)
2. **Quantile position** — Where in the distribution? (blue=low, gray=median, orange=high)
3. **Performance encoding** — How good is the return? (loss=red, exceptional=gold — heatmap)
4. **UI chrome** — Today line, annotations, lots, scanner

These should remain distinct — don't merge model identity colors with quantile position colors.

## Proposed Unified System

### Principle 1: One model = one color, everywhere

Each model gets a **single base color** from the palette, used consistently for:
- **Trace lines:** Base color at full opacity
- **Band shading:** Base color at 8%/15% opacity (symmetric bands)
- **Display Models swatch:** Base color as solid box
- **Heatmap pill bar active state:** Base color border

### Principle 2: Quantile colors are symmetric about Q50%

The thermal palette (blue→gray→orange) encodes distance from the median:
- Q1% and Q99% → same blue (both far from median)
- Q15% and Q85% → same lighter blue
- Q50% → neutral gray

This is already implemented via `_symmetric_thermal_color()`.

### Principle 3: Heatmap colors follow the global palette

The heatmap's named palettes (Forge, Thermal, etc.) are independent color schemes for the heatmap-specific color encoding (CAGR performance). These are NOT model colors — they encode return magnitude. But the default palette should match the global palette's feel:
- **Default palette:** Forge (dark→gold) — warm, matches BTC orange theme
- **cb-rg:** Ocean (navy→cyan) — cool, high contrast without red/green
- **cb-full:** Monochrome (gray) — pure luminance, no hue reliance

When the user switches the global palette, the heatmap default palette auto-switches to the matching scheme.

### Palette-aware `model_colors`

Move `MODEL_TRACE_COLORS` into `PALETTES`:

```python
PALETTES = {
    "default": {
        "model_colors": {
            "bub": "#000000",   # black
            "qr":  "#FFD700",   # gold
            "pl":  "#00E5FF",   # cyan
            "lppl":"#FF6D00",   # orange
            "exp": "#82B1FF",   # soft blue
            "ef":  "#FF80AB",   # pink
            "s2f": "#B0BEC5",   # blue-grey
        },
        "hm_default_palette": "forge",
        ...
    },
    "cb-rg": {
        "model_colors": {
            "bub": "#000000",
            "qr":  "#E69F00",   # amber
            "pl":  "#56B4E9",   # sky blue
            "lppl":"#CC79A7",   # muted pink
            "exp": "#0072B2",   # dark blue
            "ef":  "#D55E00",   # vermillion
            "s2f": "#999999",   # grey
        },
        "hm_default_palette": "ocean",
        ...
    },
    "cb-full": {
        "model_colors": {
            "bub": "#000000",
            "qr":  "#DDCC77",   # khaki
            "pl":  "#88CCEE",   # light cyan
            "lppl":"#CC6677",   # rose
            "exp": "#332288",   # indigo
            "ef":  "#AA4499",   # purple
            "s2f": "#999999",   # grey
        },
        "hm_default_palette": "mono",
        ...
    },
}
```

## Implementation Phases

### Phase A: Model colors palette-aware (4 tasks)
1. Add `model_colors` + `hm_default_palette` to each palette in `_app_ctx.py`
2. Add `_get_model_color(model_key, palette)` helper to `figures/common.py`
3. Update `figures/bubble.py` band shading to use `_get_model_color()`
4. Update `layout/bubble.py` model swatches — need to rebuild when palette changes (clientside callback or pass palette to layout)

### Phase B: Heatmap auto-palette (2 tasks)
5. Add callback: when global palette changes, set heatmap palette dropdown to `palette["hm_default_palette"]`
6. Update heatmap cell text colors to use `palette["hm_loss_text"]` and `palette["hm_exceptional_text"]` consistently

### Phase C: Overlay trace lines (2 tasks, larger)
7. Make per-model overlay trace lines use `model_colors` from palette (requires changing `_build_colors()` in btc_core.py or generating colors in figure builder)
8. Make quantile checkbox dots re-build when palette changes

### Phase D: Remaining hardcoded colors (optional)
9. OLS, UCL, U1, EF overlay colors → palette keys

## Colorblind-safe palette guidelines

Using Tol's qualitative palette for cb-rg and cb-full:
- Maximum 8 distinguishable colors
- High luminance contrast between adjacent colors
- No reliance on red/green discrimination
- cb-full uses purple/teal/rose — distinguishable by all 3 CVD types

## Heatmap named palettes (existing)

| Name | Lo→Hi | Feel |
|---|---|---|
| Forge | dark purple → gold | Warm, alchemical |
| Thermal | blue → red | Classic diverging |
| Bitcoin | dark → BTC orange | On-brand |
| Ocean | navy → cyan | Cool, accessible |
| Monochrome | gray range | Pure luminance |
| Custom | user picks 4 colors | Full control |

These stay independent — they're performance encoding, not model identity.

## Recommendation

Start with Phase A — makes model swatches + band shading palette-aware. Phase B (heatmap auto-palette) is a nice UX win. Phases C and D are deeper refactors for later.
