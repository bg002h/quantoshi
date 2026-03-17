# Colorblind-Friendly Chart Palettes — Design Spec

## Goal

Replace hardcoded chart colors with a user-selectable palette system supporting three modes: Default, Colorblind (red-green safe), and Colorblind (full CVD safe). Prioritize deuteranopia/protanopia coverage (~8% of males) with a secondary option for tritanopia.

## Constraints

- No new config panels — single navbar toggle applies globally
- Palette persists via localStorage and is included in snapshot/share links
- Dash patterns and line widths already provide shape coding — palettes add hue safety on top
- MC premium gold styling is independent and unaffected

---

## Palette Definitions

### Color Sources

CB-RG palette draws from the Wong (2011) colorblind-safe palette and IBM Design Language, both widely validated for deuteranopia/protanopia distinguishability. CB-Full uses a luminance-first approach with maximum L* separation between adjacent stops, validated against Brettel tritanopia simulation.

### Thermal Stops (quantile regression lines)

12 stops mapping quantile → hex color. Cold (low percentile) → neutral (median) → warm (high percentile).

| Quantile | Default | CB-RG | CB-Full |
|----------|---------|-------|---------|
| 0.001 (Q0.1%) | `#0d47a1` | `#0d47a1` | `#1a1a2e` (L*=12) |
| 0.01 (Q1%) | `#1565c0` | `#1565c0` | `#3d1f56` (L*=20) |
| 0.015 (Q1.5%) | `#1976d2` | `#1976d2` | `#6B3074` (L*=30) |
| 0.05 (Q5%) | `#42a5f5` | `#56B4E9` | `#995588` (L*=42) |
| 0.10 (Q10%) | `#80deea` | `#88CCEE` | `#BB7799` (L*=56) |
| 0.25 (Q25%) | `#b2dfdb` | `#AACCBB` | `#CCAAAA` (L*=72) |
| 0.50 (Q50%) | `#bdbdbd` | `#BBBBBB` | `#BBBBBB` (L*=77) |
| 0.75 (Q75%) | `#ffcc80` | `#E69F00` | `#88BBAA` (L*=72) |
| 0.90 (Q90%) | `#f7931a` | `#D55E00` | `#558899` (L*=56) |
| 0.95 (Q95%) | `#e65100` | `#CC6633` | `#336677` (L*=42) |
| 0.99 (Q99%) | `#c62828` | `#882255` | `#224466` (L*=30) |
| 0.999 (Q99.9%) | `#7f0000` | `#661155` | `#112244` (L*=12) |

**CB-RG rationale:** Lower quantiles stay blue (safe). Upper quantiles shift from red→vermillion→wine→purple, avoiding green entirely. Median remains neutral silver.

**CB-Full rationale:** Uses a luminance-driven approach. Adjacent stops are separated by L*>=10 to ensure distinguishability under any CVD type. The axis runs dark purple → light pink → silver → light teal → dark navy, relying on lightness contrast rather than hue. Validated: under Brettel tritanopia simulation, each adjacent pair maintains L*>=10 separation.

### Model Overlay Colors

| Element | Default | CB-RG | CB-Full |
|---------|---------|-------|---------|
| Non-quantized model (S2F) | `#8B4513` | `#CC79A7` (muted pink) | `#DDCC77` (sand) |
| Power Law colors | Algorithmic (current) | Same algorithm | Same algorithm |

Power Law is quantized — it uses its own per-quantile color dict (generated algorithmically in `btc_core.py`). No change needed since its colors are already distinct from thermal and use blue-purple hues.

### Delay Colors (HODL Supercharger)

| Delay | Default | CB-RG | CB-Full |
|-------|---------|-------|---------|
| 0 | `#00c853` | `#0072B2` (blue) | `#882255` (plum) |
| 1 | `#fdd835` | `#E69F00` (amber) | `#CC6677` (rose) |
| 2 | `#ff9100` | `#CC79A7` (pink) | `#DDCC77` (sand) |
| 3 | `#ff5252` | `#AA4499` (orchid) | `#117733` (forest) |
| 4 | `#b71c1c` | `#332288` (indigo) | `#332288` (indigo) |

**Note:** CB-RG delay 3 uses `#AA4499` (orchid) instead of `#882255` to avoid collision with the Q99% thermal stop which also uses `#882255` in CB-RG mode. The two contexts (quantile lines vs delay traces) can appear on the same chart in the Supercharger tab.

Annotation colors (darker versions for arrows/text) derived from each delay color by darkening ~15%.

### Delay Annotation Colors (darker)

| Delay | Default | CB-RG | CB-Full |
|-------|---------|-------|---------|
| 0 | `#00a844` | `#005B8E` | `#661144` |
| 1 | `#d4b12e` | `#B87E00` | `#AA4455` |
| 2 | `#e07d00` | `#AA6088` | `#BBAA55` |
| 3 | `#d44040` | `#883377` | `#0D5C28` |
| 4 | `#8f1616` | `#221166` | `#221166` |

### Heatmap Colorscale

| Element | Default | CB-RG | CB-Full |
|---------|---------|-------|---------|
| c_lo (min CAGR) | `#2166AC` (blue) | `#2166AC` (blue) | `#882255` (plum) |
| c_mid1 (break 1) | `#F7F7F7` (white) | `#F7F7F7` (white) | `#F7F7F7` (white) |
| c_mid2 (break 2) | `#FF8C00` (orange) | `#E69F00` (amber) | `#44AA99` (teal) |
| c_hi (max CAGR) | `#CC1100` (red) | `#882255` (wine) | `#004488` (navy) |
| Loss text | `#ff8a80` (soft red) | `#CC79A7` (pink) | `#CC6677` (rose) |
| Exceptional text | `#ffd700` (gold) | `#E69F00` (amber) | `#DDCC77` (sand) |

### Heatmap User-Customizable Color Interaction

The heatmap tab has 4 color picker controls (`hm-c-lo`, `hm-c-mid1`, `hm-c-mid2`, `hm-c-hi`). When the user switches palette:

- A callback updates the 4 color inputs to the new palette's heatmap defaults.
- If the user then manually adjusts a color picker, their custom value takes precedence until the next palette switch.
- This matches the existing "palette preset" behavior in the heatmap tab (the preset buttons already overwrite the color pickers).

### Today Line

| Element | Default | CB-RG | CB-Full |
|---------|---------|-------|---------|
| Today line | `#FF6600` | `#D55E00` (vermillion) | `#CC79A7` (pink) |

### Data Points (bubble chart scatter)

Data point color is loaded from `model_data.pkl` as `DATA_COLOR`. Currently gray (`#606060`). No change needed — gray is CVD-universal.

---

## Architecture

### Palette Registry (`_app_ctx.py`)

```python
PALETTES = {
    "default": {
        "thermal_stops": [(0.001, "#0d47a1"), (0.01, "#1565c0"), ...],
        "non_quantized_model": "#8B4513",
        "delay_colors": ["#00c853", "#fdd835", "#ff9100", "#ff5252", "#b71c1c"],
        "annot_colors": ["#00a844", "#d4b12e", "#e07d00", "#d44040", "#8f1616"],
        "today_line": "#FF6600",
        "hm_c_lo": "#2166AC",
        "hm_c_mid1": "#F7F7F7",
        "hm_c_mid2": "#FF8C00",
        "hm_c_hi": "#CC1100",
        "hm_loss_text": "#ff8a80",
        "hm_exceptional_text": "#ffd700",
    },
    "cb-rg": { ... },   # values from tables above
    "cb-full": { ... },  # values from tables above
}
PALETTE_LABELS = {
    "default": "Default",
    "cb-rg": "Colorblind (R-G)",
    "cb-full": "Colorblind (Full)",
}
```

### UI: Navbar Toggle

A small dropdown or button group in the navbar, near the share button. Icon: eye or palette icon. Three options matching `PALETTE_LABELS`. Selection stored in `dcc.Store(id="palette-store", storage_type="local", data="default")`.

### Data Flow

1. User selects palette → `palette-store` updates (localStorage persists)
2. Each chart callback reads `palette-store.data` as a `State` input
3. Callback passes `palette_key` into the `p` params dict
4. Figure builder calls `_get_palette(p)` → returns the palette dict
5. All color lookups read from the palette dict instead of module constants

### `figures/common.py` Changes

```python
def _get_palette(p):
    """Return active palette dict from params, defaulting to 'default'."""
    key = p.get("palette", "default")
    return _app_ctx.PALETTES.get(key, _app_ctx.PALETTES["default"])

def _thermal_color(q, palette=None):
    """Interpolate thermal color for quantile q from palette stops."""
    stops = (palette or _app_ctx.PALETTES["default"])["thermal_stops"]
    # ... existing interpolation logic using stops instead of _THERMAL_STOPS

def _build_thermal_colors(quantiles, palette=None):
    """Build {q: hex_color} dict for a list of quantiles."""
    return {q: _thermal_color(q, palette) for q in quantiles}
```

Module-level `_THERMAL_STOPS` constant remains as the default palette's thermal stops (backward compat for any direct callers). `_NON_QUANTIZED_MODEL_COLOR` becomes a function of palette.

### Startup Color Application (`app.py`)

The `_build_thermal_colors()` call at `app.py` line 151 runs once at module load time before any user session exists. It always uses the default palette to populate `M.qr_colors` (used for the layout quantile panel colored dots). This call remains unchanged.

Palette-aware thermal colors are built per-request inside each figure builder — the builder reads `p["palette"]` and passes the palette to `_build_thermal_colors()`. The startup call is purely for the static layout dot colors.

### Snapshot Integration

Add to `_SNAPSHOT_CONTROLS` in `snapshot.py`:
```python
("palette-store", "data"),
```

This encodes the palette choice in share links. On restore, the palette is applied. The user's own localStorage value takes precedence when no snapshot is active (standard Dash Store behavior).

**Type note:** `palette-store` holds a plain string (`"default"`, `"cb-rg"`, `"cb-full"`), not a checklist. It does NOT need a `_CHECKLIST_OPTIONS` entry — the existing snapshot encoder handles strings natively (same as other string-valued stores).

**Legacy link compatibility:** Existing share links (generated before this feature) won't contain a `palette-store` entry. The restore logic handles this gracefully: missing snapshot entries default to the component's initial value, which is `"default"`. No migration needed.

**Single-tab scope:** `palette-store` is a global (cross-tab) setting. Add it to every tab's set in `_TAB_CONTROLS` so that "Current tab only" share links still include the palette choice.

### Cache Impact

The LRU figure caches in `utils.py` key on `_quantize_params(p)`. Adding `palette` to `p` means each palette produces separate cache entries. With 3 palettes × 5 tabs × `maxsize=16`, this fits comfortably. The `palette` value is a string — `_quantize_params` only applies `_q3` rounding to floats, so strings pass through unchanged. No changes to `_quantize_params` needed.

### Prewarm Behavior

`_prewarm_caches()` in `app.py` does NOT include a `palette` key. This means only the default palette is prewarmed. The first request for `"cb-rg"` or `"cb-full"` on each tab will be a cache miss (slightly slower first load). This is acceptable — colorblind palettes are used by a minority of users, and prewarming 3× would triple startup time.

### Files Modified

| File | Change |
|------|--------|
| `_app_ctx.py` | Add `PALETTES` dict, `PALETTE_LABELS` |
| `figures/common.py` | Add `_get_palette(p)`; update `_thermal_color()`, `_build_thermal_colors()` to accept palette; today line from palette |
| `figures/bubble.py` | Read thermal colors from palette via `_build_thermal_colors(sel_qs, palette)` |
| `figures/dca.py` | Read non-quantized model color from palette |
| `figures/retire.py` | Same |
| `figures/supercharge.py` | Read delay/annot colors from palette |
| `figures/heatmap.py` | Read heatmap colorscale colors from palette |
| `layout/__init__.py` | Add palette toggle to navbar; add `dcc.Store("palette-store")`; add palette-change callback to update heatmap color inputs |
| `callbacks/charts.py` | Add `State("palette-store", "data")` to all 5 chart callbacks; pass into `p` |
| `callbacks/nav.py` | Add `_TAB_CONTROLS` entries for `palette-store` in all tabs |
| `snapshot.py` | Add `("palette-store", "data")` to `_SNAPSHOT_CONTROLS` |
| `app.py` | No change — startup `_build_thermal_colors()` stays default-only (see "Startup Color Application") |

### Files NOT Modified

| File | Why |
|------|-----|
| `mc_overlay.py` | MC traces use their own color logic, independent of palette |
| `mc_cache.py` | No color references |
| `btcpay.py`, `api.py` | No chart colors |
| `style.css` | CSS theme colors (navbar, body) are not chart colors |
| `btc_core.py` | Model color generation is algorithmic; Power Law colors stay as-is |

---

## Testing

- Existing 502 tests continue to pass (palette defaults to "default" when not in `p`)
- Add tests:
  - `_get_palette()` returns correct dict for each key; unknown key falls back to default
  - `_thermal_color()` interpolates correctly with non-default palette
  - Snapshot round-trip preserves palette choice
  - Each build function produces a figure without error for all 3 palettes
  - Heatmap color inputs update when palette changes
  - Legacy share links (no palette-store) restore with default palette
