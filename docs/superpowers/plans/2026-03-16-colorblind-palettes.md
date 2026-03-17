# Colorblind-Friendly Palettes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a three-tier user-selectable color palette system (Default / CB-RG / CB-Full) that makes all charts accessible to colorblind users.

**Architecture:** A `PALETTES` dict in `_app_ctx.py` stores all three palettes. A navbar toggle writes the selection to `dcc.Store("palette-store")` (localStorage). Each chart callback passes the palette key into the `p` params dict. Figure builders call `_get_palette(p)` to resolve colors. Snapshot links include the palette choice.

**Tech Stack:** Python 3, Plotly Dash 4.0, DBC 2.0.4, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-colorblind-palettes-design.md`

---

## File Structure

No new files. All changes are modifications to existing files:

| File | Responsibility |
|------|---------------|
| `btc_web/_app_ctx.py` | Palette data registry (PALETTES dict, PALETTE_LABELS) |
| `btc_web/figures/common.py` | `_get_palette(p)` helper; palette-aware `_thermal_color()`, `_build_thermal_colors()`, today line, non-quantized model color |
| `btc_web/figures/bubble.py` | Pass palette to thermal color builder and today line |
| `btc_web/figures/dca.py` | Read non-quantized model color from palette |
| `btc_web/figures/retire.py` | Same |
| `btc_web/figures/supercharge.py` | Read delay/annot/non-quantized colors from palette |
| `btc_web/figures/heatmap.py` | Read heatmap colorscale + cell text colors from palette |
| `btc_web/layout/__init__.py` | Navbar palette toggle + `dcc.Store("palette-store")` |
| `btc_web/callbacks/charts.py` | Add `State("palette-store", "data")` to 5 chart callbacks |
| `btc_web/callbacks/nav.py` | Add `"palette-store"` to all `_TAB_CONTROLS` sets |
| `btc_web/snapshot.py` | Add `("palette-store", "data")` to `_SNAPSHOT_CONTROLS` |
| `btc_web/test_web.py` | Add palette-related tests |

---

## Chunk 1: Palette Data + Core Helpers

### Task 1: Add PALETTES dict to _app_ctx.py

**Files:**
- Modify: `btc_web/_app_ctx.py`

- [ ] **Step 1: Add PALETTES and PALETTE_LABELS after BTC_ORANGE (line ~15)**

```python
# ── Color palettes (default + colorblind-safe alternatives) ──────────────
PALETTES = {
    "default": {
        "thermal_stops": [
            (0.001, "#0d47a1"), (0.01, "#1565c0"), (0.015, "#1976d2"),
            (0.05, "#42a5f5"), (0.10, "#80deea"), (0.25, "#b2dfdb"),
            (0.50, "#bdbdbd"), (0.75, "#ffcc80"), (0.90, "#f7931a"),
            (0.95, "#e65100"), (0.99, "#c62828"), (0.999, "#7f0000"),
        ],
        "non_quantized_model": "#8B4513",
        "delay_colors": ["#00c853", "#fdd835", "#ff9100", "#ff5252", "#b71c1c"],
        "annot_colors": ["#00a844", "#d4b12e", "#e07d00", "#d44040", "#8f1616"],
        "today_line": "#FF6600",
        "hm_c_lo": "#2166AC", "hm_c_mid1": "#F7F7F7",
        "hm_c_mid2": "#FF8C00", "hm_c_hi": "#CC1100",
        "hm_loss_text": "#ff8a80", "hm_exceptional_text": "#ffd700",
    },
    "cb-rg": {
        "thermal_stops": [
            (0.001, "#0d47a1"), (0.01, "#1565c0"), (0.015, "#1976d2"),
            (0.05, "#56B4E9"), (0.10, "#88CCEE"), (0.25, "#AACCBB"),
            (0.50, "#BBBBBB"), (0.75, "#E69F00"), (0.90, "#D55E00"),
            (0.95, "#CC6633"), (0.99, "#882255"), (0.999, "#661155"),
        ],
        "non_quantized_model": "#CC79A7",
        "delay_colors": ["#0072B2", "#E69F00", "#CC79A7", "#AA4499", "#332288"],
        "annot_colors": ["#005B8E", "#B87E00", "#AA6088", "#883377", "#221166"],
        "today_line": "#D55E00",
        "hm_c_lo": "#2166AC", "hm_c_mid1": "#F7F7F7",
        "hm_c_mid2": "#E69F00", "hm_c_hi": "#882255",
        "hm_loss_text": "#CC79A7", "hm_exceptional_text": "#E69F00",
    },
    "cb-full": {
        "thermal_stops": [
            (0.001, "#1a1a2e"), (0.01, "#3d1f56"), (0.015, "#6B3074"),
            (0.05, "#995588"), (0.10, "#BB7799"), (0.25, "#CCAAAA"),
            (0.50, "#BBBBBB"), (0.75, "#88BBAA"), (0.90, "#558899"),
            (0.95, "#336677"), (0.99, "#224466"), (0.999, "#112244"),
        ],
        "non_quantized_model": "#DDCC77",
        "delay_colors": ["#882255", "#CC6677", "#DDCC77", "#117733", "#332288"],
        "annot_colors": ["#661144", "#AA4455", "#BBAA55", "#0D5C28", "#221166"],
        "today_line": "#CC79A7",
        "hm_c_lo": "#882255", "hm_c_mid1": "#F7F7F7",
        "hm_c_mid2": "#44AA99", "hm_c_hi": "#004488",
        "hm_loss_text": "#CC6677", "hm_exceptional_text": "#DDCC77",
    },
}
PALETTE_LABELS = {
    "default": "Default",
    "cb-rg": "Colorblind (R-G)",
    "cb-full": "Colorblind (Full)",
}
```

- [ ] **Step 2: Verify it parses**

```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -c "from _app_ctx import PALETTES, PALETTE_LABELS; print(len(PALETTES), 'palettes OK')"
```

- [ ] **Step 3: Commit**

```bash
git add btc_web/_app_ctx.py
git commit -m "feat: add PALETTES registry with default, CB-RG, CB-Full palettes"
```

### Task 2: Add _get_palette helper and update thermal/color functions in figures/common.py

**Files:**
- Modify: `btc_web/figures/common.py`

- [ ] **Step 1: Add `_get_palette(p)` helper**

Add near the top of figures/common.py, after the existing constant definitions:

```python
def _get_palette(p):
    """Return active palette dict from params, defaulting to 'default'."""
    key = p.get("palette", "default")
    return _app_ctx.PALETTES.get(key, _app_ctx.PALETTES["default"])
```

- [ ] **Step 2: Update `_thermal_color()` to accept optional palette dict**

Current signature: `def _thermal_color(q: float) -> str`
New signature: `def _thermal_color(q: float, palette: dict | None = None) -> str`

Change the first line inside from:
```python
stops = _THERMAL_STOPS
```
to:
```python
stops = palette["thermal_stops"] if palette else _THERMAL_STOPS
```

The rest of the interpolation logic stays identical.

- [ ] **Step 3: Update `_build_thermal_colors()` to accept optional palette dict**

Current: `def _build_thermal_colors(quantiles: list) -> dict`
New: `def _build_thermal_colors(quantiles: list, palette: dict | None = None) -> dict`

Change body from:
```python
return {q: _thermal_color(q) for q in quantiles}
```
to:
```python
return {q: _thermal_color(q, palette) for q in quantiles}
```

- [ ] **Step 4: Verify existing tests still pass (backward compat — None defaults to _THERMAL_STOPS)**

```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py -q --tb=short --deselect btc_web/test_web.py::TestDCAMath::test_higher_quantile_less_btc 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/figures/common.py
git commit -m "feat: add _get_palette helper; _thermal_color/_build_thermal_colors accept palette"
```

### Task 3: Add palette tests

**Files:**
- Modify: `btc_web/test_web.py`

- [ ] **Step 1: Add palette test class**

Add after the existing `TestPhase5Polish` class (near end of test file):

```python
@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestPalettes:
    """Test palette registry and palette-aware color functions."""

    def test_get_palette_default(self):
        from figures.common import _get_palette
        pal = _get_palette({})
        assert pal is _app_ctx.PALETTES["default"]

    def test_get_palette_cb_rg(self):
        from figures.common import _get_palette
        pal = _get_palette({"palette": "cb-rg"})
        assert pal is _app_ctx.PALETTES["cb-rg"]

    def test_get_palette_unknown_falls_back(self):
        from figures.common import _get_palette
        pal = _get_palette({"palette": "nonexistent"})
        assert pal is _app_ctx.PALETTES["default"]

    def test_thermal_color_default_unchanged(self):
        from figures.common import _thermal_color
        # Median should be silver in default palette
        assert _thermal_color(0.50) == "#bdbdbd"

    def test_thermal_color_cb_rg_differs(self):
        from figures.common import _thermal_color
        pal = _app_ctx.PALETTES["cb-rg"]
        # Q90% should differ between default and CB-RG
        default_q90 = _thermal_color(0.90)
        cb_rg_q90 = _thermal_color(0.90, pal)
        assert default_q90 != cb_rg_q90

    def test_build_thermal_colors_with_palette(self):
        from figures.common import _build_thermal_colors
        pal = _app_ctx.PALETTES["cb-rg"]
        colors = _build_thermal_colors([0.50, 0.90], pal)
        assert len(colors) == 2
        assert colors[0.50] == "#BBBBBB"  # same across all palettes

    def test_all_palettes_have_required_keys(self):
        required = {"thermal_stops", "non_quantized_model", "delay_colors",
                    "annot_colors", "today_line", "hm_c_lo", "hm_c_mid1",
                    "hm_c_mid2", "hm_c_hi", "hm_loss_text", "hm_exceptional_text"}
        for name, pal in _app_ctx.PALETTES.items():
            missing = required - set(pal.keys())
            assert not missing, f"Palette {name!r} missing keys: {missing}"

    def test_all_palettes_thermal_stops_count(self):
        for name, pal in _app_ctx.PALETTES.items():
            assert len(pal["thermal_stops"]) == 12, f"{name} has {len(pal['thermal_stops'])} stops"

    def test_build_figures_all_palettes(self):
        """Each build function produces a figure for all 3 palettes."""
        from figures import (build_bubble_figure, build_heatmap_figure,
                             build_dca_figure, build_retire_figure,
                             build_supercharge_figure)
        for pal_key in _app_ctx.PALETTES:
            p_bub = dict(
                selected_qs=[0.5], shade=False, show_data=False,
                show_today=False, show_legend=False, minor_grid=False,
                show_comp=False, show_sup=False, xscale="log", yscale="log",
                xmin=2012, xmax=2030, ymin=1, ymax=1e6,
                n_future=1, pt_size=2, pt_alpha=0.2,
                stack=0, show_stack=False, use_lots=False, lots=[],
                comp_color="#FFD700", comp_lw=2.0, sup_color="#888", sup_lw=1.5,
                palette=pal_key,
            )
            fig = build_bubble_figure(M, p_bub)
            assert fig is not None, f"bubble failed for {pal_key}"

            p_dca = dict(
                start_stack=0, use_lots=False, amount=100, freq="Monthly",
                start_yr=2024, end_yr=2030, disp_mode="btc",
                log_y=False, show_today=False, show_legend=False,
                minor_grid=False, selected_qs=[0.5], lots=[],
                sc_enabled=False, sc_loan_amount=0, sc_rate=0.08,
                sc_loan_type="interest_only", sc_term_months=48,
                sc_repeats=0, sc_rollover=False, sc_entry_mode="live",
                sc_custom_price=100000, sc_tax_rate=0.33, sc_live_price=None,
                palette=pal_key,
            )
            fig, _ = build_dca_figure(M, p_dca)
            assert fig is not None, f"dca failed for {pal_key}"
```

- [ ] **Step 2: Run new tests to verify they pass**

```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestPalettes -v --tb=short 2>&1 | tail -20
```

- [ ] **Step 3: Commit**

```bash
git add btc_web/test_web.py
git commit -m "test: add palette registry and color function tests"
```

---

## Chunk 2: Wire Palettes Into Figure Builders

### Task 4: Update bubble.py to use palette

**Files:**
- Modify: `btc_web/figures/bubble.py`

- [ ] **Step 1: Pass palette to _build_thermal_colors**

Find the line (approx 77):
```python
_thermal = _build_thermal_colors(sel_qs)
```
Replace with:
```python
palette = _get_palette(p)
_thermal = _build_thermal_colors(sel_qs, palette)
```

Add `_get_palette` to the import from `figures.common`.

- [ ] **Step 2: Use palette today-line color**

Find the today-line shapes section (approx lines 224-233). Replace `_TODAY_LINE_COLOR` with:
```python
palette.get("today_line", _TODAY_LINE_COLOR)
```

Or simpler: at the top of the function, extract `today_color = palette.get("today_line", _TODAY_LINE_COLOR)` and use it in both shape dicts.

- [ ] **Step 3: Verify tests pass**

```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestBuildBubbleFigure -v --tb=short 2>&1 | tail -10
```

### Task 5: Update dca.py and retire.py to use palette

**Files:**
- Modify: `btc_web/figures/dca.py`
- Modify: `btc_web/figures/retire.py`

- [ ] **Step 1: In dca.py, read non-quantized color from palette**

Near the top of `build_dca_figure`, add:
```python
palette = _get_palette(p)
```
Add `_get_palette` to the import from `figures.common`.

Replace the `_NON_QUANTIZED_MODEL_COLOR` reference in the model overlay section (approx line 241) with:
```python
palette["non_quantized_model"]
```

- [ ] **Step 2: Same pattern in retire.py**

Add `palette = _get_palette(p)` near top of `build_retire_figure`.
Replace `_NON_QUANTIZED_MODEL_COLOR` (approx line 135) with `palette["non_quantized_model"]`.
Add `_get_palette` to imports.

- [ ] **Step 3: Verify tests pass**

```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestBuildDcaFigure btc_web/test_web.py::TestBuildRetireFigure -v --tb=short 2>&1 | tail -10
```

### Task 6: Update supercharge.py to use palette

**Files:**
- Modify: `btc_web/figures/supercharge.py`

- [ ] **Step 1: Read all colors from palette**

Near the top of `build_supercharge_figure`, add:
```python
palette = _get_palette(p)
delay_colors = palette["delay_colors"]
annot_colors = palette["annot_colors"]
```
Add `_get_palette` to imports.

Replace all references to `_DELAY_COLORS` with `delay_colors` and `_ANNOT_COLORS` with `annot_colors` within `build_supercharge_figure` and `_sc_mode_b`. Pass them through or extract at the right scope.

Replace `_NON_QUANTIZED_MODEL_COLOR` with `palette["non_quantized_model"]`.

Keep `_DELAY_COLORS`, `_ANNOT_COLORS` as module-level constants for backward compatibility (they become the default values).

- [ ] **Step 2: Verify tests pass**

```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestBuildSuperchargeFigure -v --tb=short 2>&1 | tail -10
```

### Task 7: Update heatmap.py to use palette

**Files:**
- Modify: `btc_web/figures/heatmap.py`

- [ ] **Step 1: Read heatmap colors from palette with fallback to model defaults**

In `build_heatmap_figure`, the colors are already read from `p` with model defaults:
```python
c_lo   = p.get("c_lo",   m.CAGR_SEG_C_LO)
```

The palette should provide the fallback instead of `m.CAGR_SEG_*` when no user-custom color is set. The callback will set `p["c_lo"]` etc. from the palette (see Task 9).

So heatmap.py only needs palette for the cell text colors. Near the top of `build_heatmap_figure` and `_heatmap_cell_annots`:

```python
palette = _get_palette(p)
```

In `_heatmap_cell_annots`, replace hardcoded loss/exceptional text colors:
- `"#ff8a80"` → `palette.get("hm_loss_text", "#ff8a80")`
- `"#ffd700"` → `palette.get("hm_exceptional_text", "#ffd700")`

Add `_get_palette` to imports.

- [ ] **Step 2: Verify tests pass**

```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestBuildHeatmapFigure -v --tb=short 2>&1 | tail -10
```

- [ ] **Step 3: Commit all figure builder changes**

```bash
git add btc_web/figures/
git commit -m "feat: figure builders read colors from palette via _get_palette(p)"
```

---

## Chunk 3: UI Toggle, Callbacks, and Snapshot Integration

### Task 8: Add palette-store and navbar toggle to layout/__init__.py

**Files:**
- Modify: `btc_web/layout/__init__.py`

- [ ] **Step 1: Add palette-store**

After the existing `dcc.Store` declarations (around line 73-75), add:
```python
dcc.Store(id="palette-store", storage_type="local", data="default"),
```

- [ ] **Step 2: Add palette toggle to navbar**

In the navbar right-side section (near the share button), add a small dropdown:
```python
dbc.Select(
    id="palette-select",
    options=[{"label": v, "value": k} for k, v in _app_ctx.PALETTE_LABELS.items()],
    value="default",
    size="sm",
    style={"width": "150px", "fontSize": "0.8rem"},
),
```

- [ ] **Step 3: Add callback to sync palette-select → palette-store**

Add a simple callback (can go in `callbacks/nav.py` or inline in layout):
```python
@callback(
    Output("palette-store", "data"),
    Input("palette-select", "value"),
)
def _update_palette_store(val):
    return val or "default"
```

And a reverse callback to sync palette-store → palette-select on page load (for localStorage restore):
```python
@callback(
    Output("palette-select", "value"),
    Input("palette-store", "data"),
)
def _sync_palette_select(stored):
    return stored or "default"
```

- [ ] **Step 4: Verify the app loads**

```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); sys.path.insert(0,'archive/btc_app'); import app; print('OK')"
```

### Task 9: Add palette-store as State to all 5 chart callbacks

**Files:**
- Modify: `btc_web/callbacks/charts.py`

- [ ] **Step 1: Add State("palette-store", "data") to each callback decorator**

For each of the 5 chart callbacks (`update_bubble`, `update_heatmap`, `update_dca`, `update_retire`, `update_supercharge`), add to the `@callback` decorator:
```python
State("palette-store", "data"),
```

Add a corresponding parameter to the function signature (e.g., `palette_key`).

Inside each callback, add `palette_key` to the `p` dict:
```python
p["palette"] = palette_key or "default"
```

- [ ] **Step 2: Add heatmap color inputs update on palette switch**

Add a callback that fires when `palette-store` changes and updates the 4 heatmap color inputs:

```python
@callback(
    Output("hm-c-lo", "value"),
    Output("hm-c-mid1", "value"),
    Output("hm-c-mid2", "value"),
    Output("hm-c-hi", "value"),
    Input("palette-store", "data"),
    prevent_initial_call=True,
)
def _update_hm_colors_on_palette(pal_key):
    pal = _app_ctx.PALETTES.get(pal_key, _app_ctx.PALETTES["default"])
    return pal["hm_c_lo"], pal["hm_c_mid1"], pal["hm_c_mid2"], pal["hm_c_hi"]
```

- [ ] **Step 3: Verify tests pass**

```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py -q --tb=short --deselect btc_web/test_web.py::TestDCAMath::test_higher_quantile_less_btc 2>&1 | tail -5
```

### Task 10: Add palette-store to snapshot and _TAB_CONTROLS

**Files:**
- Modify: `btc_web/snapshot.py`
- Modify: `btc_web/callbacks/nav.py`

- [ ] **Step 1: Add to _SNAPSHOT_CONTROLS**

Append to the `_SNAPSHOT_CONTROLS` list in `snapshot.py`:
```python
("palette-store", "data"),
```

This becomes index 98 (after the current 98 controls at indices 0–97).

- [ ] **Step 2: Add "palette-store" to all _TAB_CONTROLS sets in nav.py**

In `_TAB_CONTROLS`, add `"palette-store"` to every tab's set (since palette is a global setting that should be included in single-tab share links):

```python
for _tab_set in _TAB_CONTROLS.values():
    _tab_set.add("palette-store")
```

Or add it inline to each set definition.

- [ ] **Step 3: Add snapshot round-trip test**

In `test_web.py`, add to `TestPalettes`:

```python
def test_snapshot_preserves_palette(self):
    from snapshot import _encode_snapshot, _decode_snapshot, _SNAPSHOT_CONTROLS
    # Find palette-store index
    idx = next(i for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS)
               if cid == "palette-store")
    # Build a state dict with cb-rg palette
    state = [None] * len(_SNAPSHOT_CONTROLS)
    state[idx] = "cb-rg"
    encoded = _encode_snapshot(state)
    decoded = _decode_snapshot(encoded)
    assert decoded[idx] == "cb-rg"
```

- [ ] **Step 4: Verify full test suite passes**

```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py -q --tb=short --deselect btc_web/test_web.py::TestDCAMath::test_higher_quantile_less_btc 2>&1 | tail -5
```

- [ ] **Step 5: Commit all UI/callback/snapshot changes**

```bash
git add btc_web/layout/__init__.py btc_web/callbacks/charts.py btc_web/callbacks/nav.py btc_web/snapshot.py btc_web/test_web.py
git commit -m "feat: palette navbar toggle, chart callback integration, snapshot support"
```

---

## Chunk 4: Final Verification

### Task 11: Full integration test

- [ ] **Step 1: Run full test suite**

```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py -q --tb=short --deselect btc_web/test_web.py::TestDCAMath::test_higher_quantile_less_btc 2>&1 | tail -10
```

Expected: 510+ passed (502 existing + ~10 new palette tests).

- [ ] **Step 2: Verify app loads and serves**

```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); sys.path.insert(0,'archive/btc_app'); import app; print('Layout:', app.app.layout is not None); print('Palettes:', len(_app_ctx.PALETTES))"
```

- [ ] **Step 3: Final commit if any remaining changes**

```bash
git add -u && git status
git commit -m "feat: colorblind-friendly palette system complete — 3 tiers, navbar toggle, snapshot integration"
```

---

## Summary

| Task | What |
|------|------|
| 1 | PALETTES dict in `_app_ctx.py` (3 palettes × 12 color keys) |
| 2 | `_get_palette(p)` + palette-aware `_thermal_color`/`_build_thermal_colors` |
| 3 | Palette tests (registry, colors, figure builds) |
| 4 | `bubble.py` — thermal + today line from palette |
| 5 | `dca.py` + `retire.py` — non-quantized model color from palette |
| 6 | `supercharge.py` — delay/annot/non-quantized colors from palette |
| 7 | `heatmap.py` — cell text colors from palette |
| 8 | Layout: `palette-store` + navbar toggle |
| 9 | Callbacks: `State("palette-store")` on 5 chart callbacks + heatmap color sync |
| 10 | Snapshot: `_SNAPSHOT_CONTROLS` + `_TAB_CONTROLS` |
| 11 | Full integration test |
