# Color Centralization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize every color literal in `btc_web/` into a single `btc_web/colors.py` source-of-truth file, with auto-generated CSS and JS artifacts that all consumers reference instead of hex literals.

**Architecture:** Python-authoritative + generated artifacts (Architecture A from spec). `colors.py` is the only place hex literals appear. A generator script writes `assets/_colors_generated.css` (with `var(--qs-*)` definitions and `:root[data-palette="..."]` overrides for the 4 palettes) and `assets/_colors_generated.js` (with `window.QS_COLORS` / `QS_PALETTES`). An inline pre-paint script in `index_string` sets `documentElement.dataset.palette` from localStorage so first-paint already has the user's saved palette. A clientside callback handles post-load palette switches. A pytest lint test enforces the invariant by walking the codebase and rejecting any hex literal outside the source module.

**Tech Stack:** Python 3.12+, Dash 4.0.0, dash-bootstrap-components 2.0.4, Plotly 6.x, vanilla ES5 JS in `assets/`, plain CSS with custom properties (`var()`).

**Spec:** `docs/superpowers/specs/2026-04-10-color-centralization-design.md` (v2, approved 2026-04-10)

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `btc_web/colors.py` | **Create** | Single source of truth — every color constant + 4 palette dicts + ticker dict + Citadel overlay dict |
| `tools/generate_color_artifacts.py` | **Create** | CLI tool that reads `btc_web/colors.py` and emits `_colors_generated.css` + `_colors_generated.js` |
| `btc_web/assets/_colors_generated.css` | **Create (generated)** | CSS custom properties; loads first via underscore prefix |
| `btc_web/assets/_colors_generated.js` | **Create (generated)** | `window.QS_COLORS` + `QS_PALETTES` globals; loads first |
| `btc_web/test_colors_central.py` | **Create** | Pytest lint enforcing the centralization invariant |
| `btc_web/_app_ctx.py` | **Modify** | Replace inline `PALETTES`/`DECOMP_COLORS`/`DECOMP_SUM_COLOR`/`MODEL_TRACE_COLORS`/`BTC_ORANGE` with re-exports from `colors` |
| `btc_web/theme.py` | **Modify** | Re-export 5 constants from `colors`; preserve original names |
| `btc_web/callbacks/ticker.py` | **Modify** | Import `TICKER_MODEL_COLORS` from `colors` |
| `btc_web/figures/common.py` | **Modify** | Replace 13 hex literals with imports |
| `btc_web/figures/bubble.py` | **Modify** | Replace 9 hex literals (uses DECOMP_COLORS, etc.) |
| `btc_web/figures/dca.py` | **Modify** | (already mostly clean — sweep remaining hex) |
| `btc_web/figures/retire.py` | **Modify** | (already mostly clean) |
| `btc_web/figures/supercharge.py` | **Modify** | Replace 12 hex literals |
| `btc_web/figures/citadel.py` | **Modify** | Replace 12 hex literals |
| `btc_web/figures/heatmap.py` | **Modify** | Replace remaining hex |
| `btc_web/figures/residuals.py` | **Modify** | Replace remaining hex |
| `btc_web/layout/common.py` | **Modify** | Replace 10 hex literals |
| `btc_web/layout/bubble.py` | **Modify** | Replace 9 hex literals |
| `btc_web/layout/model_info.py` | **Modify** | Replace 7 hex literals |
| `btc_web/layout/__init__.py` | **Modify** | Replace 6 hex literals + add inline pre-paint script to `index_string` |
| `btc_web/callbacks/charts.py` | **Modify** | Replace 8 hex literals |
| `btc_web/callbacks/mc_controls.py` | **Modify** | Replace 8 hex literals |
| `btc_web/callbacks/snapshot_cb.py` | **Modify** | Replace 6 hex literals |
| `btc_web/callbacks/citadel_cb.py` | **Modify** | Replace 4 hex literals (Task 19 sweep) |
| `btc_web/callbacks/citadel_save_cb.py` | **Modify** | Replace rgba literals (Task 22 sweep) |
| `btc_web/callbacks/user_model.py` | **Modify** | Replace rgba literals (Task 22 sweep) |
| `btc_web/callbacks/mc_payment.py` | **Modify** | Replace 1 hex literal (Task 19 sweep) |
| `btc_web/callbacks/scanner.py` | **Modify** | Replace 1 hex literal (Task 19 sweep) |
| `btc_web/callbacks/nav.py` | **Modify** | Replace 1 hex literal + add new clientside callback for `data-palette` attribute |
| `btc_web/layout/citadel.py` | **Modify** | Replace 2 hex literals (Task 20 sweep) |
| `btc_web/layout/citadel_tax.py` | **Modify** | Replace 2 hex literals (Task 20 sweep) |
| `btc_web/layout/faq.py` | **Modify** | Replace 1 hex literal (Task 20 sweep) |
| `btc_web/layout/heatmap.py` | **Modify** | Replace 4 hex literals (Task 20 sweep) |
| `btc_web/layout/mc_controls.py` | **Modify** | Replace 2 hex literals (Task 20 sweep) |
| `btc_web/layout/stack.py` | **Modify** | Replace 4 hex literals (Task 20 sweep) |
| `btc_web/api.py` | **Modify** | Replace 30 SVG hex literals via f-string interpolation |
| `btc_web/mc_overlay.py` | **Modify** | Replace 6 Citadel overlay hex literals |
| `btc_web/utils.py` | **Modify** | Replace 1 hex literal (Task 21 sweep) |
| `btc_web/tasks.py` | **Modify** | Replace 1 hex literal (Task 21 sweep) |
| `btc_web/static_pages.py` | **Modify** | Replace 1 hex literal (Task 21 sweep) |
| `btc_web/tab_defaults.py` | **Modify** | Replace 4 hex literals (Task 21 sweep) |
| (15 Python files with rgba/rgb literals) | **Modify** | Migrate to `_hex_alpha(constant, alpha)` or named constants in colors.py (Task 22 sweep) |
| `btc_web/app.py` | **Modify** | Add DEV-only generator startup hook + modify `index_string` |
| `btc_web/assets/style.css` | **Modify** | Replace 48 hex literals with `var(--qs-*)` |
| `btc_web/assets/chart_responsive.js` | **Modify** | Read defaults from `window.QS_COLORS` (3 literals) |
| `btc_web/assets/plot_appearance.js` | **Modify** | Read defaults from `window.QS_COLORS` (3 literals) |

**Total:** ~30 files modified, 5 created. Estimated ~35 commits.

---

## Phase 1 — Foundation (no behavioral changes)

Phase 1 establishes `colors.py` as a parallel registry alongside `_app_ctx.py`. Nothing in the running app changes behavior — `_app_ctx.PALETTES` continues to serve the same dict; `colors.PALETTES` is identical and exists for the generator + downstream Phase 2 migration. Verification: a tiny test asserts `colors.PALETTES == _app_ctx.PALETTES`.

### Task 1: Create `btc_web/colors.py` source-of-truth file

**Files:**
- Create: `btc_web/colors.py`

- [ ] **Step 1: Write the file with the exact content below**

Create `btc_web/colors.py` with these contents (verbatim — every value matches the current `_app_ctx.py`, `theme.py`, `callbacks/ticker.py`, and `mc_overlay.py`):

```python
"""Single source of truth for every color in Quantoshi.

This module is the ONLY place hex color literals appear in the codebase.
A pytest lint test (test_colors_central.py) enforces this invariant.

Consumers:
  - Python: `from colors import BTC_ORANGE, PALETTES, ...`
  - CSS:    `var(--qs-btc-orange)` (from generated _colors_generated.css)
  - JS:     `window.QS_COLORS.btc_orange` (from generated _colors_generated.js)

Workflow when adding/changing a color:
  1. Edit a value here
  2. In DEV mode, restart the dev server — generator runs at startup
  3. CSS / JS artifacts are regenerated automatically
  4. The lint test verifies nothing leaked back into other files
  5. Commit colors.py + the regenerated _colors_generated.css/js together

Spec: docs/superpowers/specs/2026-04-10-color-centralization-design.md
"""

# ════════════════════════════════════════════════════════════════════
# SECTION 1 — Palette-invariant constants
# ════════════════════════════════════════════════════════════════════

# ── Brand identity ────────────────────────────────────────────────
BTC_ORANGE          = "#f7931a"   # Bitcoin canonical orange
QUANTOSHI_TITLE     = "#1A3060"   # navbar wordmark / chart titles (alias of TITLE_COLOR)
QUANTOSHI_NAVY      = "#0a1929"   # navbar background

# ── Status / semantic ──────────────────────────────────────────────
ERROR_RED           = "#ff5252"
WARNING_AMBER       = "#ffa726"
SUCCESS_GREEN       = "#4caf50"
INFO_BLUE           = "#1976d2"

# ── UI surfaces ───────────────────────────────────────────────────
MODAL_BG            = "#FFFFFF"
DRAWER_BG           = "#F5F5F5"
SECTION_CARD_BG     = "#FAFAFA"
FOCUS_RING          = "#1a6fa8"
LINK                = "#1a6fa8"

# ── Static SVG generation (api.py shareable badges) ───────────────
SVG_BADGE_BG        = "#1a3060"
SVG_BADGE_TEXT      = "#ffffff"

# ── Chart theme (palette-invariant — also re-exported from theme.py) ──
# These KEEP THEIR ORIGINAL NAMES for zero-breakage on existing importers.
PLOT_BG_COLOR       = "#FFFFFF"
TEXT_COLOR          = "#222222"
TITLE_COLOR         = "#1A3060"
SPINE_COLOR         = "#888888"
GRID_MAJOR_COLOR    = "#888888"
GRID_MINOR_COLOR    = "#B0B0B0"
FALLBACK_MODEL_GRAY = "#888888"
SCATTER_POINT       = "#2C3E50"

# ── Palette-invariant model trace fallback dict ────────────────────
# IMPORTANT: this dict is INTENTIONALLY DISTINCT from
# PALETTES["default"]["model_colors"] below. It is used by
# `_get_model_color()` as the dict-default when palette.get(...) is
# empty, and by some legacy code paths. Migrating these into a single
# dict would silently change visual colors. Kept as its own constant.
MODEL_TRACE_COLORS = {
    "bub":       "#DAA520",   # goldenrod — matches bubble composite
    "qr":        "#B0BEC5",   # blue-grey — muted
    "pl":        "#00E5FF",   # electric cyan
    "lppl":      "#FF6D00",   # deep orange
    "lp2":       "#FF9F40",
    "lp3":       "#FFD080",
    "lp4":       "#FFE0A0",
    "linppl":    "#00B8A0",   # teal
    "hybppl":    "#7B68EE",   # medium slate blue
    "hybppl_dd": "#B39DDB",   # lavender
    "exp":       "#CE93D8",
    "ef":        "#E8C860",
    "s2f":       "#FFD700",
    "hyb2l":     "#6A5ACD",
    "hyb2c":     "#20B2AA",
    "hyb2b":     "#DB7093",
    "hyb4d":     "#8B6914",
    "pca":       "#4B0082",
    "grdy":      "#228B22",
    "eppl":      "#D4760A",
    "gomp":      "#4682B4",
    "bpl":       "#CD853F",
}

# ── Ticker model colors (palette-invariant, distinct values) ──────
# These are the colors used by the navbar live-price ticker as it
# cycles through models. They have their own hue choices and are
# kept as a distinct constant.
TICKER_MODEL_COLORS = {
    "qr":         "#5dade2",   # sky blue
    "bub":        "#f39c12",   # amber/gold
    "pl":         "#2ecc71",   # green
    "lp3":        "#e74c3c",   # red — LPPL₃ (default config)
    "cfg_1d_1u":  "#7B68EE",   # medium slate blue — HybPPL (default config)
    "ecfg_1d_1u": "#D4760A",   # warm amber — EPPL (default config)
    "pca":        "#4B0082",   # indigo — PCA
    "grdy":       "#228B22",   # forest green — Greedy
    "ef":         "#1abc9c",   # teal
}

# ── Citadel asset overlay colors (palette-invariant) ──────────────
# Used by mc_overlay.py for the Citadel multi-asset MC overlay.
# Keys: total / btc_usd / cash / reserves_total / investments_total.
CITADEL_OVERLAY_COLORS = {
    "total":             "#000000",
    "btc_usd":           "#F7931A",
    "cash":              "#C0C0C0",
    "reserves_total":    "#4A90D9",
    "investments_total": "#27AE60",
}

# ════════════════════════════════════════════════════════════════════
# SECTION 2 — Per-palette dictionaries (palette-aware)
# ════════════════════════════════════════════════════════════════════

DEFAULT = {
    "model_colors": {
        "bub": "#C8960C", "qr": "#0055FF", "pl": "#00BB00",
        "lppl": "#EE0000", "lp2": "#FF6666", "lp3": "#FFAAAA", "lp4": "#FFCCCC",
        "linppl": "#00D4AA", "hybppl": "#9370DB", "hybppl_dd": "#B39DDB",
        "ef": "#FFE066", "exp": "#9933FF", "s2f": "#FF7700",
        "u1": "#333333",
        "hyb2l": "#6A5ACD", "hyb2c": "#20B2AA", "hyb2b": "#DB7093",
        "hyb4d": "#8B6914", "pca": "#4B0082",
        "grdy": "#228B22",
        "eppl": "#D4760A",
        "gomp": "#4682B4", "bpl": "#CD853F",
    },
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
    "decomp_colors": ["#E64A19", "#1976D2", "#388E3C", "#7B1FA2",
                      "#F57C00", "#00796B", "#5D4037"],
    "decomp_sum_color": "#000000",
}

CB_BRIAN = {
    "model_colors": {
        "bub": "#BF8C0A", "qr": "#556B2F", "pl": "#C635F5",
        "lppl": "#AD1457", "lp2": "#D81B60", "lp3": "#F06292", "lp4": "#F8BBD0",
        "linppl": "#006064", "hybppl": "#4527A0", "hybppl_dd": "#8E24AA",
        "ef": "#FFE082", "exp": "#E0E0E0", "s2f": "#777777",
        "u1": "#333333",
        "hyb2l": "#5B4AB0", "hyb2c": "#1A9A8F", "hyb2b": "#C4607A",
        "hyb4d": "#7A5B10", "pca": "#3A006F",
        "grdy": "#1B7A1B",
        "eppl": "#B86800",
        "gomp": "#3B6FA0", "bpl": "#B87333",
    },
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
    "decomp_colors": ["#D81B60", "#1E88E5", "#004D40", "#F4511E",
                      "#6A1B9A", "#00695C", "#3E2723"],
    "decomp_sum_color": "#000000",
}

CB_RG = {
    "model_colors": {
        "bub": "#F5793A", "qr": "#A8A8A8", "pl": "#0F2080",
        "lppl": "#85C0F9", "lp2": "#B0D8FF", "lp3": "#D4E9FF", "lp4": "#EAF4FF",
        "linppl": "#FFB000", "hybppl": "#D4A017", "hybppl_dd": "#ECC060",
        "ef": "#F5A060", "exp": "#BBBBBB", "s2f": "#F5C242",
        "hyb2l": "#7B68EE", "hyb2c": "#2E8B57", "hyb2b": "#CC6699",
        "hyb4d": "#8B7500", "pca": "#551A8B",
        "grdy": "#2E8B57",
        "eppl": "#CC8800",
        "gomp": "#4169E1", "bpl": "#CC7722",
        "u1": "#333333",
    },
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
    "decomp_colors": ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                      "#0072B2", "#D55E00", "#CC79A7"],
    "decomp_sum_color": "#000000",
}

CB_FULL = {
    "model_colors": {
        "bub": "#B8920C", "qr": "#606060", "pl": "#B0E0E6",
        "lppl": "#1A1A1A", "lp2": "#444444", "lp3": "#707070", "lp4": "#A0A0A0",
        "linppl": "#2A2A2A", "hybppl": "#505050", "hybppl_dd": "#989898",
        "ef": "#F0D870", "exp": "#909090", "s2f": "#FFE066",
        "u1": "#333333",
        "hyb2l": "#6060A0", "hyb2c": "#4A8A7A", "hyb2b": "#A06080",
        "hyb4d": "#7A7A50", "pca": "#4A4A70",
        "grdy": "#4A7A4A",
        "eppl": "#8A7030",
        "gomp": "#5B7FAA", "bpl": "#AA8844",
    },
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
    "decomp_colors": ["#000000", "#505050", "#808080", "#A0A0A0",
                      "#C0C0C0", "#6A6A6A", "#303030"],
    "decomp_sum_color": "#F5793A",
}

PALETTES = {
    "default":  DEFAULT,
    "cb-brian": CB_BRIAN,
    "cb-rg":    CB_RG,
    "cb-full":  CB_FULL,
}
PALETTE_KEYS = tuple(PALETTES.keys())

# ════════════════════════════════════════════════════════════════════
# SECTION 3 — Generation metadata
# ════════════════════════════════════════════════════════════════════

# Names that should NOT be emitted to generated CSS/JS artifacts.
# Default behavior: every module-level UPPER_CASE name with a string,
# dict, or list value gets emitted. Use this set to suppress.
__skip_export__ = frozenset({
    # Currently empty — every constant above is exposed to CSS/JS.
})
```

- [ ] **Step 2: Verify the file compiles and imports cleanly**

```bash
cd /scratch/code/bitcoinprojections/btc_web && ../btc_venv/bin/python3 -m py_compile colors.py && echo COMPILE_OK
cd /scratch/code/bitcoinprojections/btc_web && PYTHONPATH=".:../:../archive/btc_app" ../btc_venv/bin/python3 -c "
import colors
print('PALETTES keys:', list(colors.PALETTES.keys()))
print('BTC_ORANGE:', colors.BTC_ORANGE)
print('TICKER_MODEL_COLORS keys:', list(colors.TICKER_MODEL_COLORS.keys()))
print('CITADEL_OVERLAY_COLORS keys:', list(colors.CITADEL_OVERLAY_COLORS.keys()))
print('OK')
"
```

Expected:
```
COMPILE_OK
PALETTES keys: ['default', 'cb-brian', 'cb-rg', 'cb-full']
BTC_ORANGE: #f7931a
TICKER_MODEL_COLORS keys: ['qr', 'bub', 'pl', 'lp3', 'cfg_1d_1u', 'ecfg_1d_1u', 'pca', 'grdy', 'ef']
CITADEL_OVERLAY_COLORS keys: ['total', 'btc_usd', 'cash', 'reserves_total', 'investments_total']
OK
```

- [ ] **Step 3: Verify byte-for-byte parity with `_app_ctx.py:PALETTES`**

`colors.PALETTES` adds two NEW keys per palette (`decomp_colors` and `decomp_sum_color`) that don't yet exist in `_app_ctx.PALETTES`. So a top-level `==` would always fail. Compare key-by-key instead, excluding the new decomp keys, then verify the decomp values via the existing `DECOMP_COLORS` / `DECOMP_SUM_COLOR` top-level dicts in `_app_ctx.py`:

```bash
cd /scratch/code/bitcoinprojections/btc_web && PYTHONPATH=".:../:../archive/btc_app" ../btc_venv/bin/python3 -c "
import colors, _app_ctx
_NEW_KEYS = {'decomp_colors', 'decomp_sum_color'}
for pkey in ['default', 'cb-brian', 'cb-rg', 'cb-full']:
    cp = {k: v for k, v in colors.PALETTES[pkey].items() if k not in _NEW_KEYS}
    ap = _app_ctx.PALETTES[pkey]
    assert cp == ap, f'palette[{pkey}] mismatch on existing keys'
    assert colors.PALETTES[pkey]['decomp_colors'] == _app_ctx.DECOMP_COLORS[pkey], f'decomp_colors[{pkey}] drift'
    assert colors.PALETTES[pkey]['decomp_sum_color'] == _app_ctx.DECOMP_SUM_COLOR[pkey], f'decomp_sum_color[{pkey}] drift'
assert colors.MODEL_TRACE_COLORS == _app_ctx.MODEL_TRACE_COLORS, 'MODEL_TRACE_COLORS drift!'
assert colors.BTC_ORANGE == _app_ctx.BTC_ORANGE, 'BTC_ORANGE drift!'
print('PARITY OK')
"
```

Expected: `PARITY OK`. If any assert fails, fix the literal in colors.py to match `_app_ctx.py`.

- [ ] **Step 4: Verify ticker colors parity**

```bash
cd /scratch/code/bitcoinprojections/btc_web && PYTHONPATH=".:../:../archive/btc_app" ../btc_venv/bin/python3 -c "
import colors
from callbacks import ticker
assert colors.TICKER_MODEL_COLORS == ticker._MODEL_COLORS, 'ticker drift'
print('TICKER PARITY OK')
"
```

Expected: `TICKER PARITY OK`.

- [ ] **Step 5: Commit**

```bash
cd /scratch/code/bitcoinprojections
git add btc_web/colors.py
git commit -m "$(cat <<'EOF'
feat(colors): create btc_web/colors.py source-of-truth module

Phase 1, Task 1 of the color centralization plan. Creates the parallel
color registry containing every palette-invariant constant + the four
palette dicts (DEFAULT/CB_BRIAN/CB_RG/CB_FULL) + TICKER_MODEL_COLORS +
CITADEL_OVERLAY_COLORS.

Values are byte-identical to the existing _app_ctx.py:PALETTES,
DECOMP_COLORS, DECOMP_SUM_COLOR, MODEL_TRACE_COLORS, BTC_ORANGE,
callbacks/ticker.py:_MODEL_COLORS, and mc_overlay.py:_CITADEL_MC_COLORS.
Verified via parity test in this commit's verification.

DECOMP_COLORS and DECOMP_SUM_COLOR are now per-palette inside each
palette dict (decomp_colors / decomp_sum_color keys) instead of
top-level dicts keyed by palette name. Phase 2 migrations update the
consumers.

Spec: docs/superpowers/specs/2026-04-10-color-centralization-design.md
EOF
)"
```

---

### Task 2: Create `tools/generate_color_artifacts.py` and initial artifacts

**Files:**
- Create: `tools/generate_color_artifacts.py`
- Create: `btc_web/assets/_colors_generated.css`
- Create: `btc_web/assets/_colors_generated.js`

- [ ] **Step 1: Write the generator script**

Create `tools/generate_color_artifacts.py` with this exact content:

```python
#!/usr/bin/env python3
"""Generate _colors_generated.css and _colors_generated.js from btc_web/colors.py.

Usage:
    python tools/generate_color_artifacts.py            # write artifacts
    python tools/generate_color_artifacts.py --check    # exit 1 if drift
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Ensure btc_web is on the path so we can `import colors`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "btc_web"))
sys.path.insert(0, str(_PROJECT_ROOT))

import colors  # noqa: E402

CSS_PATH = _PROJECT_ROOT / "btc_web" / "assets" / "_colors_generated.css"
JS_PATH  = _PROJECT_ROOT / "btc_web" / "assets" / "_colors_generated.js"

# Names from colors.py that the generator should NOT emit.
SKIP = getattr(colors, "__skip_export__", frozenset())


def _kebab(name: str) -> str:
    """snake_case → kebab-case."""
    return name.replace("_", "-").lower()


def _flatten_palette_for_css(palette: dict, prefix: str = "qs") -> list[tuple[str, str]]:
    """Flatten one palette dict into a list of (--qs-...-key, hex) entries.

    Returns CSS variable name and value pairs for everything except
    `thermal_stops` (a list of (float, str) tuples — exposed only as
    JS arrays, NOT as individual CSS vars, since the float quantile
    values aren't meaningful selector keys). The isinstance(v, str)
    check inside the list branch silently skips tuples by design.
    """
    out: list[tuple[str, str]] = []
    for key, val in palette.items():
        kebab = _kebab(key)
        if isinstance(val, str):
            out.append((f"--{prefix}-{kebab}", val))
        elif isinstance(val, list):
            # delay_colors → --qs-delay-0, --qs-delay-1, ... (drop _colors suffix)
            stem = kebab.removesuffix("-colors") if kebab.endswith("-colors") else kebab
            for i, v in enumerate(val):
                if isinstance(v, str):
                    out.append((f"--{prefix}-{stem}-{i}", v))
        elif isinstance(val, dict):
            # model_colors → --qs-model-bub, --qs-model-qr, ...
            stem = kebab.removesuffix("-colors") if kebab.endswith("-colors") else kebab
            for k, v in val.items():
                if isinstance(v, str):
                    out.append((f"--{prefix}-{stem}-{_kebab(k)}", v))
        # tuple-of-tuple lists like thermal_stops are NOT exposed to CSS.
    return out


def _gather_top_level_constants() -> list[tuple[str, object]]:
    """Return [(name, value), ...] for every uppercase module-level
    constant in colors.py whose value is a str/dict/list and not in SKIP."""
    out = []
    for name in dir(colors):
        if not name.isupper():
            continue
        if name in SKIP:
            continue
        if name.startswith("_"):
            continue
        if name in ("PALETTE_KEYS",):
            continue
        val = getattr(colors, name)
        if isinstance(val, (str, dict, list)):
            out.append((name, val))
    return out


def _source_hash() -> str:
    """SHA256 of colors.py contents (first 16 hex chars)."""
    src = (_PROJECT_ROOT / "btc_web" / "colors.py").read_bytes()
    return hashlib.sha256(src).hexdigest()[:16]


def _generate_css() -> str:
    sha = _source_hash()
    lines = [
        f"/* AUTO-GENERATED by tools/generate_color_artifacts.py — DO NOT EDIT.",
        f"   Source: btc_web/colors.py  Source-SHA256: {sha} */",
        "",
        ":root {",
        "    /* ── Palette-invariant constants ── */",
    ]
    # Top-level scalar constants and dict-of-strings (e.g. CITADEL_OVERLAY_COLORS)
    for name, val in _gather_top_level_constants():
        if isinstance(val, str):
            lines.append(f"    --qs-{_kebab(name)}: {val};")
        elif isinstance(val, dict) and name not in ("DEFAULT", "CB_BRIAN", "CB_RG", "CB_FULL", "PALETTES"):
            # Flat dict-of-strings (TICKER_MODEL_COLORS, CITADEL_OVERLAY_COLORS, MODEL_TRACE_COLORS)
            stem = _kebab(name).removesuffix("-colors")
            for k, v in val.items():
                if isinstance(v, str):
                    lines.append(f"    --qs-{stem}-{_kebab(k)}: {v};")
    lines += [
        "",
        "    /* ── Default palette (active when no [data-palette] is set) ── */",
    ]
    for var_name, hex_val in _flatten_palette_for_css(colors.DEFAULT):
        lines.append(f"    {var_name}: {hex_val};")
    lines.append("}")
    lines.append("")
    # Per-palette overrides (skip default — it's already in :root)
    for pkey, pdict in colors.PALETTES.items():
        if pkey == "default":
            continue
        lines.append(f':root[data-palette="{pkey}"] {{')
        for var_name, hex_val in _flatten_palette_for_css(pdict):
            lines.append(f"    {var_name}: {hex_val};")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _js_repr(val) -> str:
    """JSON-ish JS literal for a Python value (str/list/tuple/dict)."""
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(_js_repr(v) for v in val) + "]"
    if isinstance(val, dict):
        items = ", ".join(f'"{k}": {_js_repr(v)}' for k, v in val.items())
        return "{" + items + "}"
    return str(val)


def _generate_js() -> str:
    sha = _source_hash()
    lines = [
        f"/* AUTO-GENERATED by tools/generate_color_artifacts.py — DO NOT EDIT.",
        f"   Source: btc_web/colors.py  Source-SHA256: {sha} */",
        "(function() {",
        "    'use strict';",
        "    window.QS_COLORS = {",
    ]
    for name, val in _gather_top_level_constants():
        if isinstance(val, str):
            lines.append(f"        {name.lower()}: {_js_repr(val)},")
        elif isinstance(val, dict) and name not in ("DEFAULT", "CB_BRIAN", "CB_RG", "CB_FULL", "PALETTES"):
            lines.append(f"        {name.lower()}: {_js_repr(val)},")
    lines.append("    };")
    lines.append("    window.QS_PALETTES = {")
    for pkey, pdict in colors.PALETTES.items():
        lines.append(f'        "{pkey}": {_js_repr(pdict)},')
    lines.append("    };")
    # Spec-required dedicated namespace for ticker colors
    lines.append(f"    window.QS_TICKER_COLORS = {_js_repr(colors.TICKER_MODEL_COLORS)};")
    lines.append("})();")
    return "\n".join(lines) + "\n"


def write_artifacts() -> tuple[Path, Path]:
    css_text = _generate_css()
    js_text = _generate_js()
    CSS_PATH.write_text(css_text)
    JS_PATH.write_text(js_text)
    return CSS_PATH, JS_PATH


def check_artifacts() -> bool:
    """Return True if on-disk matches generated. False (and print diff hint) if not."""
    css_text = _generate_css()
    js_text = _generate_js()
    css_match = CSS_PATH.exists() and CSS_PATH.read_text() == css_text
    js_match = JS_PATH.exists() and JS_PATH.read_text() == js_text
    if not css_match:
        print(f"DRIFT: {CSS_PATH} differs from generated output", file=sys.stderr)
    if not js_match:
        print(f"DRIFT: {JS_PATH} differs from generated output", file=sys.stderr)
    return css_match and js_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 if on-disk artifacts differ from generated output")
    args = parser.parse_args()
    if args.check:
        ok = check_artifacts()
        sys.exit(0 if ok else 1)
    css_path, js_path = write_artifacts()
    print(f"Wrote {css_path}")
    print(f"Wrote {js_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator and produce initial artifacts**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 tools/generate_color_artifacts.py
```

Expected output:
```
Wrote /scratch/code/bitcoinprojections/btc_web/assets/_colors_generated.css
Wrote /scratch/code/bitcoinprojections/btc_web/assets/_colors_generated.js
```

- [ ] **Step 3: Sanity-check the generated CSS file**

```bash
head -25 btc_web/assets/_colors_generated.css
```

Expected: header comment with `Source-SHA256:` line, `:root {` block with `--qs-btc-orange: #f7931a;`, `--qs-plot-bg-color: #FFFFFF;`, `--qs-model-bub: #C8960C;`, etc.

- [ ] **Step 4: Sanity-check the generated JS file**

```bash
head -10 btc_web/assets/_colors_generated.js
node --check btc_web/assets/_colors_generated.js && echo JS_OK
```

Expected: header comment, `window.QS_COLORS = { ... }`, `JS_OK` from node.

- [ ] **Step 5: Verify `--check` mode passes immediately after writing**

```bash
btc_venv/bin/python3 tools/generate_color_artifacts.py --check && echo CHECK_OK
```

Expected: `CHECK_OK` (exit 0, no drift output).

- [ ] **Step 6: Verify Dash asset load order — generated files sort first**

```bash
ls btc_web/assets/*.css btc_web/assets/*.js | head -10
```

Expected: `_colors_generated.css` and `_colors_generated.js` appear first alphabetically (underscore < letters in ASCII).

- [ ] **Step 7: Commit**

```bash
git add tools/generate_color_artifacts.py btc_web/assets/_colors_generated.css btc_web/assets/_colors_generated.js
git commit -m "$(cat <<'EOF'
feat(colors): generator script + initial CSS/JS artifacts

Phase 1, Task 2. Adds tools/generate_color_artifacts.py which reads
btc_web/colors.py and emits two artifacts:
- assets/_colors_generated.css with :root + :root[data-palette="..."]
  blocks defining --qs-* CSS custom properties for every constant
  and every palette
- assets/_colors_generated.js exposing window.QS_COLORS (palette-
  invariant constants) and window.QS_PALETTES (the 4 palette dicts
  including thermal_stops as nested arrays)

Generator features:
- --check mode for CI: exits non-zero if on-disk artifacts differ
  from what would be generated. Used to catch "edited colors.py
  but forgot to regenerate" footguns.
- Source-SHA256 header in both artifacts (no timestamp — keeps
  --check deterministic).
- Underscore-prefixed filenames so Dash's assets/* loader picks
  them up first alphabetically (loads BEFORE style.css and the
  consumer JS files).

Both artifacts are checked into git so production loads them
without depending on the generator running at runtime.
EOF
)"
```

---

### Task 3: Wire generator into DEV-mode app startup

**Files:**
- Modify: `btc_web/app.py`

- [ ] **Step 0: Create `tools/__init__.py`** so `from tools.generate_color_artifacts import ...` works as a package import.

```bash
test -f tools/__init__.py || (touch tools/__init__.py && echo CREATED)
```

- [ ] **Step 1: Read the current `app.py` to find a good insertion point**

```bash
grep -n "^import\|^from\|if __name__\|app = dash\|os.environ" btc_web/app.py | head -20
```

Identify where the app object is created. The generator hook should run BEFORE the layout is built but AFTER `import os` is available.

- [ ] **Step 2: Add the DEV-mode generator hook near the top of `app.py`**

First, find the precise insertion point. The hook must run AFTER `import os` but BEFORE the Dash app object is created. Use this command to find a stable anchor:

```bash
grep -n "^app = dash\|^app = Dash\|app = _dash" btc_web/app.py | head -3
```

Identify the line that creates the Dash app (e.g. `app = dash.Dash(...)`). The hook goes immediately ABOVE that line.

Use the Edit tool with `old_string` set to the line creating the Dash app object (verbatim), and `new_string` set to the hook block followed by the same line. Example:

If `app.py` has:
```python
app = dash.Dash(__name__, ...)
```
Then `old_string` = `app = dash.Dash(__name__, ...)` and `new_string` =
```python
# ── Color artifact regeneration (DEV mode only) ────────────────────────
# In dev mode, regenerate _colors_generated.css/js from colors.py on
# every startup so edits to colors.py propagate without manual steps.
# In production (gunicorn), the checked-in artifacts are used as-is to
# avoid race conditions between workers writing the same files.
if os.environ.get("DEV"):
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _proj_root = _Path(__file__).resolve().parent.parent
        _sys.path.insert(0, str(_proj_root))
        from tools.generate_color_artifacts import write_artifacts as _write_color_artifacts
        _write_color_artifacts()
    except Exception as _e:
        # Non-fatal — never block startup. Manual generator run is the fallback.
        print(f"[colors] DEV-mode color artifact regen skipped: {_e}")

app = dash.Dash(__name__, ...)
```

If `app.py` doesn't already `import os` at the top, add `import os` to the imports block via a separate Edit.

- [ ] **Step 3: Verify app.py compiles**

```bash
cd btc_web && ../btc_venv/bin/python3 -m py_compile app.py && echo COMPILE_OK
```

Expected: `COMPILE_OK`.

- [ ] **Step 4: Verify DEV-mode startup runs the generator**

```bash
lsof -ti :8050 | xargs -r kill -9; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
disown
sleep 8
curl -fsS http://localhost:8050/ -o /dev/null && echo SERVER_UP
btc_venv/bin/python3 tools/generate_color_artifacts.py --check && echo NO_DRIFT_AFTER_STARTUP
```

Expected: `SERVER_UP` and `NO_DRIFT_AFTER_STARTUP`. The startup hook regenerated the artifacts; they match the generator output.

- [ ] **Step 5: Commit**

```bash
git add btc_web/app.py tools/__init__.py
git commit -m "$(cat <<'EOF'
feat(colors): DEV-mode generator hook in app.py startup

Phase 1, Task 3. Adds a small startup hook to btc_web/app.py that
runs tools/generate_color_artifacts.write_artifacts() when DEV=1
is set. This way editing btc_web/colors.py and restarting the dev
server is enough to propagate changes — no manual generator step.

In production (gunicorn, no DEV), the hook is a no-op. The checked-
in artifacts are used as-is. Avoids the race condition risk of 5
gunicorn workers writing the same two files simultaneously.

Wrapped in try/except so a generator failure never blocks startup.
EOF
)"
```

---

## Phase 2 — Python migration

Phase 2 swaps consumers from inline literals / `_app_ctx.PALETTES` to imports from `colors`. Order: foundational re-exports first, then file-by-file consumer migration.

### Task 4: Migrate `_app_ctx.py` to re-export from `colors`

**Files:**
- Modify: `btc_web/_app_ctx.py`

- [ ] **Step 1: Read the relevant section of `_app_ctx.py`**

```bash
sed -n '26p;75p;87p;94p;120p;221p' btc_web/_app_ctx.py
```

Verify lines 26 (`BTC_ORANGE`), 75-84 (`DECOMP_COLORS`), 87-92 (`DECOMP_SUM_COLOR`), 94-117 (`MODEL_TRACE_COLORS`), 120-221 (`PALETTES`).

- [ ] **Step 2: Add a single top-of-file import block from `colors`**

At the top of `_app_ctx.py` (in the existing imports section, near `import math`), add:

```python
from colors import (
    BTC_ORANGE,
    MODEL_TRACE_COLORS,
    PALETTES,
)
```

This is the canonical top-of-file import — no mid-file imports. The four downstream replacements (steps 3-6) just delete the inline definitions; the names are already bound from this import.

- [ ] **Step 3: Delete the `BTC_ORANGE` inline definition**

Find the line `BTC_ORANGE = "#f7931a"` (around line 26) and delete it. The name is now provided by the top-of-file import.

- [ ] **Step 4: Replace `DECOMP_COLORS` definition with derived view**

Find lines 75-84 (the `DECOMP_COLORS = { ... }` block) and replace with:

```python
# DECOMP_COLORS — derived view from per-palette decomp_colors keys
DECOMP_COLORS = {pkey: PALETTES[pkey]["decomp_colors"] for pkey in PALETTES}
```

- [ ] **Step 5: Replace `DECOMP_SUM_COLOR` definition with derived view**

Find lines 87-92 (the `DECOMP_SUM_COLOR = { ... }` block) and replace with:

```python
# DECOMP_SUM_COLOR — derived view from per-palette decomp_sum_color keys
DECOMP_SUM_COLOR = {pkey: PALETTES[pkey]["decomp_sum_color"] for pkey in PALETTES}
```

- [ ] **Step 6: Delete the `MODEL_TRACE_COLORS` inline definition**

Find lines 94-117 (the `MODEL_TRACE_COLORS = { ... }` block) and delete it. The name is now provided by the top-of-file import.

- [ ] **Step 6b: Delete the inline `PALETTES = { ... }` block**

Find lines 120-221 (the entire `PALETTES = { ... }` block) and delete it. The name is now provided by the top-of-file import.

- [ ] **Step 7: Verify the file compiles and parity holds**

```bash
cd btc_web && ../btc_venv/bin/python3 -m py_compile _app_ctx.py && echo COMPILE_OK
PYTHONPATH=".:../:../archive/btc_app" ../btc_venv/bin/python3 -c "
import _app_ctx, colors
assert _app_ctx.PALETTES is colors.PALETTES
assert _app_ctx.MODEL_TRACE_COLORS is colors.MODEL_TRACE_COLORS
assert _app_ctx.BTC_ORANGE == colors.BTC_ORANGE
for k in ('default','cb-brian','cb-rg','cb-full'):
    assert _app_ctx.DECOMP_COLORS[k] == colors.PALETTES[k]['decomp_colors']
    assert _app_ctx.DECOMP_SUM_COLOR[k] == colors.PALETTES[k]['decomp_sum_color']
print('OK')
"
```

Expected: `COMPILE_OK` then `OK`.

- [ ] **Step 8: Run the figure regression suite**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/test_figures.py btc_web/test_callbacks.py btc_web/test_defaults.py -q --timeout=120 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add btc_web/_app_ctx.py
git commit -m "$(cat <<'EOF'
refactor(colors): _app_ctx re-exports BTC_ORANGE, PALETTES, etc. from colors

Phase 2, Task 4. _app_ctx.py is now a thin re-export layer:
- BTC_ORANGE          → from colors import BTC_ORANGE
- MODEL_TRACE_COLORS  → from colors import MODEL_TRACE_COLORS
- PALETTES            → from colors import PALETTES
- DECOMP_COLORS       → derived view {pkey: PALETTES[pkey]['decomp_colors']}
- DECOMP_SUM_COLOR    → derived view {pkey: PALETTES[pkey]['decomp_sum_color']}

Existing consumers (figures/, layouts/, callbacks/) keep working
unchanged via the re-exports. Subsequent tasks will migrate them to
import directly from colors.

Net delta: -160 lines from _app_ctx.py.
EOF
)"
```

---

### Task 5: Migrate `theme.py` to re-export from `colors`

**Files:**
- Modify: `btc_web/theme.py`

- [ ] **Step 1: Replace the entire file contents**

Use the Write tool to overwrite `btc_web/theme.py` with:

```python
"""Quantoshi chart theme — re-exports from colors.py for backward compat.

This module preserves the original constant names so existing importers
(`from theme import PLOT_BG_COLOR`) keep working without modification.
The actual values live in btc_web/colors.py.
"""
from colors import (
    PLOT_BG_COLOR,
    TEXT_COLOR,
    TITLE_COLOR,
    SPINE_COLOR,
    GRID_MAJOR_COLOR,
)

__all__ = [
    "PLOT_BG_COLOR",
    "TEXT_COLOR",
    "TITLE_COLOR",
    "SPINE_COLOR",
    "GRID_MAJOR_COLOR",
]
```

- [ ] **Step 2: Verify importers still work**

```bash
cd btc_web && PYTHONPATH=".:../:../archive/btc_app" ../btc_venv/bin/python3 -c "
import theme
print(theme.PLOT_BG_COLOR, theme.TEXT_COLOR, theme.TITLE_COLOR, theme.SPINE_COLOR, theme.GRID_MAJOR_COLOR)
"
```

Expected: `#FFFFFF #222222 #1A3060 #888888 #888888`

- [ ] **Step 3: Commit**

```bash
git add btc_web/theme.py
git commit -m "refactor(colors): theme.py re-exports from colors module

Phase 2, Task 5. Original constant names preserved (PLOT_BG_COLOR
etc.) for zero-breakage on existing importers. Values now live in
colors.py."
```

---

### Task 6: Migrate `callbacks/ticker.py` to import `TICKER_MODEL_COLORS` from `colors`

**Files:**
- Modify: `btc_web/callbacks/ticker.py`

- [ ] **Step 1: Find the existing `_MODEL_COLORS` block**

```bash
sed -n '14,24p' btc_web/callbacks/ticker.py
```

- [ ] **Step 2: Replace the inline dict with an import alias**

In `btc_web/callbacks/ticker.py`, replace the entire block:

```python
_MODEL_COLORS = {
    "qr":         "#5dade2",   # sky blue
    "bub":        "#f39c12",   # amber/gold
    "pl":         "#2ecc71",   # green
    "lp3":        "#e74c3c",   # red — LPPL₃ (default config)
    "cfg_1d_1u":  "#7B68EE",   # medium slate blue — HybPPL (default config)
    "ecfg_1d_1u": "#D4760A",   # warm amber — EPPL (default config)
    "pca":        "#4B0082",   # indigo — PCA
    "grdy":       "#228B22",   # forest green — Greedy
    "ef":         "#1abc9c",   # teal
}
```

with:

```python
from colors import TICKER_MODEL_COLORS as _MODEL_COLORS
```

- [ ] **Step 3: Verify**

```bash
cd btc_web && ../btc_venv/bin/python3 -m py_compile callbacks/ticker.py && echo OK
PYTHONPATH=".:../:../archive/btc_app" ../btc_venv/bin/python3 -c "
from callbacks.ticker import _MODEL_COLORS
import colors
assert _MODEL_COLORS is colors.TICKER_MODEL_COLORS
print('OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add btc_web/callbacks/ticker.py
git commit -m "refactor(colors): ticker imports TICKER_MODEL_COLORS from colors

Phase 2, Task 6. Local _MODEL_COLORS dict deleted; now an import
alias from colors.TICKER_MODEL_COLORS. Values unchanged."
```

---

### Tasks 7–17: File-by-file Python literal migration

For each file in this phase, the migration follows a uniform pattern:

1. **Grep the file** for hex literals: `grep -n '#[0-9a-fA-F]\{6\}' <file>`
2. **Add the import** at the top: `from colors import <CONSTANTS>` or `from _app_ctx import PALETTES` (already a re-export)
3. **For each literal**, decide whether it's a known palette-aware color (use `_get_palette(p)["..."]`) or a palette-invariant constant (import the named constant from `colors`). For one-off literals not yet in `colors.py`, ADD them to `colors.py` first (as a named constant), regenerate artifacts, then use them.
4. **Compile + run targeted tests** (`figures` tests for figure files, `callbacks` tests for callbacks)
5. **Commit** with a message naming the file and literal count

Each task below is one file. The implementer subagent for each task reads the file, identifies the literals, edits them, and verifies before committing.

### Task 7: Migrate `btc_web/figures/common.py`

**Files:**
- Modify: `btc_web/figures/common.py`

- [ ] **Step 1: Inventory the literals**

```bash
grep -n '#[0-9a-fA-F]\{6\}\|rgba(' btc_web/figures/common.py
```

Expected: ~13 hex literals + a few rgba/rgb. Likely candidates:
- Default fallback grays (`"#888888"`, `"#bdbdbd"`)
- Legend / annotation defaults
- Today line color
- Hover format colors

- [ ] **Step 2: Add the import**

Add at the top of the file (in the existing `from _app_ctx import ...` block or alongside other module-level imports):

```python
from colors import FALLBACK_MODEL_GRAY, BTC_ORANGE, GRID_MAJOR_COLOR, GRID_MINOR_COLOR
```

If any literal is NOT yet in colors.py and IS palette-invariant, ADD it to colors.py FIRST, regenerate artifacts (`python tools/generate_color_artifacts.py`), commit colors.py + artifacts as a separate sub-commit, then import.

- [ ] **Step 3: Replace each literal**

Use Edit tool, one literal at a time. Common substitutions:
- `"#888888"` → `FALLBACK_MODEL_GRAY` (in `_get_model_color` fallback)
- `"#888"` → `FALLBACK_MODEL_GRAY` (short hex form)
- `"#bdbdbd"` (palette default Q50 gray) → reference via palette dict, not direct import
- `BTC_ORANGE` literal `"#f7931a"` → `BTC_ORANGE` import

For palette-aware lookups already done via `_get_palette(p)`, leave the call unchanged — only the `.get(key, FALLBACK)` second argument needs to switch from `"#888888"` to `FALLBACK_MODEL_GRAY`.

- [ ] **Step 4: Verify**

```bash
cd btc_web && ../btc_venv/bin/python3 -m py_compile figures/common.py && echo OK
cd .. && btc_venv/bin/python3 -m pytest btc_web/test_figures.py -q --timeout=120 2>&1 | tail -5
```

Expected: `OK` and all tests pass.

- [ ] **Step 5: Confirm hex count is zero (or only allowlisted)**

```bash
grep -c '#[0-9a-fA-F]\{6\}' btc_web/figures/common.py
```

Expected: `0`. If non-zero, those are leftover literals that need either replacement OR an explicit allowlist comment `# qs-color-allow` (rare).

- [ ] **Step 6: Commit**

```bash
git add btc_web/figures/common.py
git commit -m "refactor(colors): figures/common.py imports from colors"
```

### Task 8: Migrate `btc_web/figures/bubble.py`

Same pattern as Task 7. Inventory:

```bash
grep -n '#[0-9a-fA-F]\{6\}' btc_web/figures/bubble.py
```

~9 literals expected (decomposition trace fallbacks, helper colors). Replace with imports from `colors` or palette lookups. Compile, test, commit.

- [ ] **Step 1**: Inventory literals
- [ ] **Step 2**: Add appropriate `from colors import ...`
- [ ] **Step 3**: Edit each literal in-place
- [ ] **Step 4**: `py_compile` + `pytest btc_web/test_figures.py`
- [ ] **Step 5**: Verify `grep -c '#[0-9a-fA-F]\{6\}' = 0`
- [ ] **Step 6**: `git commit -m "refactor(colors): figures/bubble.py imports from colors"`

### Task 9: Migrate `btc_web/figures/dca.py`

Same pattern. Most literals already removed in prior session work. Inventory + sweep.

### Task 10: Migrate `btc_web/figures/retire.py`

Same pattern. Most literals already removed. Inventory + sweep.

### Task 11: Migrate `btc_web/figures/supercharge.py`

Same pattern. ~12 literals remain (delay annotation defaults, dotted guide lines, fallback grays).

### Task 12: Migrate `btc_web/figures/citadel.py`

Same pattern. ~12 literals (asset overlay defaults, hardcoded text colors). Use `CITADEL_OVERLAY_COLORS` for the asset overlays.

### Task 13: Migrate `btc_web/figures/heatmap.py` and `btc_web/figures/residuals.py`

Same pattern. Heatmap colorscale fields use the per-palette `hm_c_*` keys (already in palettes). Residuals uses model colors. Two commits — one per file.

### Task 14: Migrate `btc_web/layout/common.py`

**Files:**
- Modify: `btc_web/layout/common.py`

~10 literals in style dicts (background colors, borders, focus rings). Add named constants to `colors.py` first if needed (e.g. `MODAL_BG`, `DRAWER_BG`, `SECTION_CARD_BG`, `FOCUS_RING`, `LINK`). Regenerate artifacts. Then import in layout/common.py and replace.

- [ ] **Step 1**: Inventory `grep -n '#[0-9a-fA-F]\{6\}' btc_web/layout/common.py`
- [ ] **Step 2**: Identify which literals don't yet have a name in `colors.py` and add them as new constants
- [ ] **Step 3**: Run `python tools/generate_color_artifacts.py` to regenerate
- [ ] **Step 4**: Commit `colors.py` + regenerated artifacts together
- [ ] **Step 5**: Edit `layout/common.py`: add `from colors import ...`, replace each literal
- [ ] **Step 6**: `py_compile` + `pytest btc_web/test_callbacks.py`
- [ ] **Step 7**: Verify `grep -c = 0`
- [ ] **Step 8**: Commit

### Task 15: Migrate `btc_web/layout/bubble.py`, `layout/model_info.py`, `layout/__init__.py`

Three files, three commits. Same pattern as Task 14. `layout/__init__.py` will get the inline pre-paint script in Task 18 — for now just sweep its hex literals.

### Task 16: Migrate `btc_web/callbacks/charts.py`, `callbacks/mc_controls.py`, `callbacks/snapshot_cb.py`

Three files, three commits. Same pattern.

### Task 17: Migrate `btc_web/api.py` (SVG generation)

**Files:**
- Modify: `btc_web/api.py`

**30 hex literals** (verified count) embedded in SVG template strings. `api.py` already uses Python f-strings with doubled braces (`{{ }}`) for embedded CSS — the implementer must preserve this escaping. Migration approach:

- [ ] **Step 1**: Inventory `grep -n '#[0-9a-fA-F]\{6\}' btc_web/api.py`
- [ ] **Step 2**: Add new SVG-specific constants to `colors.py` if any aren't already named (`SVG_BADGE_BG`, `SVG_BADGE_TEXT`, etc.). Regenerate artifacts. Commit colors.py + artifacts.
- [ ] **Step 3**: Add `from colors import ...` at top of api.py
- [ ] **Step 4**: Convert each SVG template literal to an f-string (or .format()) with the imported constant.

**Concrete example for an existing f-string with doubled braces** (api.py already does this for inline CSS):

Before:
```python
return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="80">
  <style>.lbl {{ font: 12px sans-serif; fill: #ffffff; }}</style>
  <rect width="100%" height="100%" fill="#1a3060"/>
  <text x="10" y="50" class="lbl">Bitcoin: {price}</text>
</svg>'''
```

After:
```python
return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="80">
  <style>.lbl {{ font: 12px sans-serif; fill: {SVG_BADGE_TEXT}; }}</style>
  <rect width="100%" height="100%" fill="{SVG_BADGE_BG}"/>
  <text x="10" y="50" class="lbl">Bitcoin: {price}</text>
</svg>'''
```

**Critical**: only the `#abc123` literals get replaced with `{CONSTANT_NAME}`. The pre-existing `{{ }}` doubled braces around `font: ...` stay as-is — they are escaped CSS rules inside an f-string and must remain literal `{` `}` for the rendered SVG.

For non-f-string templates that don't have `{{ }}` issues, simply convert to f-string and substitute. For `<text fill="#ffffff">` → `<text fill="{SVG_BADGE_TEXT}">`.
- [ ] **Step 5**: `py_compile`, run any api tests
- [ ] **Step 6**: Verify `grep -c = 0`
- [ ] **Step 7**: Commit

### Task 18 (was Task 17 in the original ordering): Migrate `btc_web/mc_overlay.py`

**Files:**
- Modify: `btc_web/mc_overlay.py`

- [ ] **Step 1**: Find the `_CITADEL_MC_COLORS` definition

```bash
sed -n '838,850p' btc_web/mc_overlay.py
```

- [ ] **Step 2**: Replace the inline dict with an import alias

Replace:

```python
_CITADEL_MC_COLORS = {
    "total":             "#000000",   # black
    "btc_usd":           "#F7931A",   # orange
    "cash":              "#C0C0C0",   # silver
    "reserves_total":    "#4A90D9",   # blue
    "investments_total": "#27AE60",   # green
}
```

with:

```python
from colors import CITADEL_OVERLAY_COLORS as _CITADEL_MC_COLORS
```

- [ ] **Step 3**: Find and fix the fallback at line 957 (`"#f7931a"`)

```bash
grep -n "#f7931a" btc_web/mc_overlay.py
```

Replace `"#f7931a"` with `BTC_ORANGE` after adding `from colors import BTC_ORANGE` at the top of the file.

- [ ] **Step 4**: Verify

```bash
cd btc_web && ../btc_venv/bin/python3 -m py_compile mc_overlay.py && echo OK
```

- [ ] **Step 5**: Commit

```bash
git add btc_web/mc_overlay.py
git commit -m "refactor(colors): mc_overlay.py imports CITADEL_OVERLAY_COLORS from colors"
```

---

### Task 19: Sweep remaining callback files

**Files:**
- Modify: `btc_web/callbacks/citadel_cb.py` (4 hex literals)
- Modify: `btc_web/callbacks/mc_payment.py` (1 hex literal)
- Modify: `btc_web/callbacks/scanner.py` (1 hex literal)
- Modify: `btc_web/callbacks/nav.py` (1 hex literal — separate from the data-palette callback added in Task 23)

These four files were missed by Tasks 16. They each have a small number of hex literals (mostly status colors, fallback grays, or one-off UI tints).

For each file in the list, perform the standard migration sequence:

- [ ] **Step 1**: `grep -n '#[0-9a-fA-F]\{6\}' btc_web/callbacks/<file>.py` to inventory
- [ ] **Step 2**: For any literal not yet in `colors.py`, add it as a named constant (e.g. `STATUS_OK_GREEN`, `MUTED_TEXT`, etc.). Regenerate artifacts via `btc_venv/bin/python3 tools/generate_color_artifacts.py`. Commit `colors.py` + the regenerated `_colors_generated.css`/`.js` first.
- [ ] **Step 3**: Add `from colors import <CONSTANTS>` at the top of the callback file
- [ ] **Step 4**: Replace each literal with its imported constant (one Edit call per literal)
- [ ] **Step 5**: `cd btc_web && ../btc_venv/bin/python3 -m py_compile callbacks/<file>.py && echo OK`
- [ ] **Step 6**: `grep -c '#[0-9a-fA-F]\{6\}' btc_web/callbacks/<file>.py` should print `0`
- [ ] **Step 7**: Run targeted tests if any reference this file: `btc_venv/bin/python3 -m pytest btc_web/test_callbacks.py -q`
- [ ] **Step 8**: Commit each file separately:
  ```bash
  git add btc_web/callbacks/<file>.py
  git commit -m "refactor(colors): callbacks/<file>.py imports from colors"
  ```

Repeat for all 4 files. 4 commits total (plus optional sub-commit for adding new constants to colors.py if needed).

---

### Task 20: Sweep remaining layout files

**Files:**
- Modify: `btc_web/layout/citadel.py` (2 hex literals)
- Modify: `btc_web/layout/citadel_tax.py` (2 hex literals)
- Modify: `btc_web/layout/faq.py` (1 hex literal)
- Modify: `btc_web/layout/heatmap.py` (4 hex literals)
- Modify: `btc_web/layout/mc_controls.py` (2 hex literals)
- Modify: `btc_web/layout/stack.py` (4 hex literals)

Same standard migration sequence as Task 19, applied to each layout file. Most literals are style-dict values (border colors, focus rings, badge backgrounds). Add new constants to `colors.py` as needed (one consolidated colors.py + artifacts sub-commit at the START of this task is fine — figure out the full set of new constants needed by grepping all 6 files first).

- [ ] **Step 1**: Inventory across all 6 files at once

```bash
for f in citadel.py citadel_tax.py faq.py heatmap.py mc_controls.py stack.py; do
    echo "--- $f ---"
    grep -n '#[0-9a-fA-F]\{6\}' btc_web/layout/$f
done
```

- [ ] **Step 2**: Decide on named constants for any literals not yet in colors.py. Add them in one batch, regenerate, and commit colors.py + artifacts.

```bash
btc_venv/bin/python3 tools/generate_color_artifacts.py
git add btc_web/colors.py btc_web/assets/_colors_generated.css btc_web/assets/_colors_generated.js
git commit -m "feat(colors): add layout-sweep constants for Task 20"
```

- [ ] **Steps 3–8**: For each of the 6 files, add imports, replace literals, py_compile, verify zero hex, commit individually. 6 commits total.

---

### Task 21: Sweep remaining root-level Python files

**Files:**
- Modify: `btc_web/utils.py` (1 hex literal)
- Modify: `btc_web/tasks.py` (1 hex literal)
- Modify: `btc_web/static_pages.py` (1 hex literal)
- Modify: `btc_web/tab_defaults.py` (4 hex literals — likely default UI/chart colors)

Same standard migration sequence as Task 19, one file per commit. 4 commits.

`tab_defaults.py` is special: its 4 literals are likely default values for plot appearance controls (`pt_color`, `grid_major_color`, etc.). Verify these match the constants in `colors.py` (e.g. `SCATTER_POINT`, `GRID_MAJOR_COLOR`) and import them rather than re-declaring.

```bash
grep -n '#[0-9a-fA-F]\{6\}' btc_web/utils.py btc_web/tasks.py btc_web/static_pages.py btc_web/tab_defaults.py
```

---

### Task 22: rgba/rgb string literal sweep

**Files (15 with rgba/rgb literals):**
- `btc_web/figures/bubble.py`
- `btc_web/figures/citadel.py`
- `btc_web/figures/common.py`
- `btc_web/figures/heatmap.py`
- `btc_web/figures/residuals.py`
- `btc_web/layout/bubble.py`
- `btc_web/layout/citadel.py`
- `btc_web/layout/common.py`
- `btc_web/layout/faq.py`
- `btc_web/layout/__init__.py`
- `btc_web/layout/mc_controls.py`
- `btc_web/layout/model_info.py`
- `btc_web/mc_overlay.py`
- `btc_web/callbacks/citadel_save_cb.py`
- `btc_web/callbacks/user_model.py`

Each contains literal `rgba(...)` or `rgb(...)` strings that the lint test (`test_no_rgba_literals_in_python` in Task 27) will reject.

**Pre-migration step: relocate `_hex_alpha` to `colors.py`**.

`_hex_alpha(hex_color, alpha)` currently lives in `figures/common.py:759`. After this task, **layout** files will need to call it, which would create an architectural smell (layouts importing from figures internals) and risks a circular import. Move it to `colors.py` first:

- [ ] **Step 0a**: Add the function to `colors.py`:

```python
def _hex_alpha(hex_color: str, alpha: float) -> str:
    """Convert a #rrggbb hex color to an rgba(...) string with the given alpha.

    Lives in colors.py so both Python figures AND Python layouts can call it
    without crossing the figures/layouts architectural boundary. Re-exported
    from figures/common.py for backward compatibility with existing callers.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
```

- [ ] **Step 0b**: In `figures/common.py`, replace the existing `_hex_alpha` definition with a re-export:

```python
from colors import _hex_alpha  # noqa: F401 — re-exported for backward compat
```

Verify `figures/common.py` compiles and tests still pass.

- [ ] **Step 0c**: Commit the relocation as a sub-commit at the START of this task:

```bash
git add btc_web/colors.py btc_web/figures/common.py
git commit -m "refactor(colors): move _hex_alpha into colors.py (re-export from figures/common.py)"
```

Note: `_hex_alpha` is a function (not a hex literal), so the lint test does NOT need to be updated. The function returns rgba strings at runtime, which AST inspection correctly skips because they're function returns, not source-literal Constant nodes.

**Migration approach for each literal**:

**Two replacement strategies:**

1. **`_hex_alpha(constant, alpha)`** — use this when the rgba is conceptually "named hex constant + alpha overlay". Example: `"rgba(100,100,100,0.35)"` → import `FALLBACK_MODEL_GRAY` from colors + use `_hex_alpha(FALLBACK_MODEL_GRAY, 0.35)`. The `_hex_alpha` helper already exists in `figures/common.py:759`. Function calls return runtime strings — the lint AST walker correctly skips them.

2. **Baked-alpha named constant in `colors.py`** — use this when the rgba string is reused multiple times or has a clear semantic name. Example: add `LOG_MINOR_GRID_FILL = "rgba(100,100,100,0.35)"` to colors.py as a string constant. The lint test allows colors.py itself to contain rgba string literals (it's the source of truth). Importers do `from colors import LOG_MINOR_GRID_FILL`.

**Per-file procedure:**

- [ ] **Step 1**: `grep -n 'rgba\?(' btc_web/figures/common.py` (etc.) to inventory
- [ ] **Step 2**: For each rgba literal, decide strategy 1 or 2. Strategy 1 if the alpha is a one-off; strategy 2 if the literal is reused or has a clear name.
- [ ] **Step 3**: For strategy 2, add the constants to `colors.py`. Regenerate artifacts. Commit colors.py + artifacts as a single sub-commit at the START of this task (handles all rgba constants for all 15 files in one go).
- [ ] **Step 4**: For each file, add imports + replace literals via Edit. Verify with:
  ```bash
  cd btc_web && ../btc_venv/bin/python3 -m py_compile <file> && echo OK
  ```
- [ ] **Step 5**: After each file, commit: `git commit -m "refactor(colors): <file> uses _hex_alpha and named constants"`

15 commits in this task (plus the single colors.py sub-commit at the start).

**Note**: this sweep is the most labor-intensive in the entire plan. The implementer subagent should be told to take its time and not rush — every rgba replacement is an opportunity for a typo'd alpha value or a wrong base color.

---

## Phase 3 — CSS migration

### Task 23: Add `data-palette` to `<html>` via inline script + clientside callback

**Files:**
- Modify: `btc_web/app.py` (or `btc_web/layout/__init__.py`, wherever `index_string` is set)
- Modify: `btc_web/callbacks/nav.py`

- [ ] **Step 1: Find where `index_string` is set**

```bash
grep -rn "index_string" btc_web/app.py btc_web/layout/ | head -5
```

- [ ] **Step 2: Add the inline pre-paint script to `index_string`**

Inside the `<head>` section of `index_string`, add this script BEFORE Dash's own scripts:

```html
<script>
  (function() {
    try {
      var raw = localStorage.getItem("palette-store");
      if (raw) {
        var key;
        try { key = JSON.parse(raw); } catch(e) { key = raw; }
        if (typeof key === "string") {
          document.documentElement.dataset.palette = key;
        }
      }
    } catch(e) {}
  })();
</script>
```

Note: Dash's `dcc.Store(storage_type="local")` may store the value as JSON-encoded or as a wrapping object. The try/catch around `JSON.parse` handles both cases. Pre-merge verification step in this task (Step 5) confirms the actual format.

- [ ] **Step 3: Add the clientside callback to `callbacks/nav.py`**

Find the existing palette dropdown handler in `callbacks/nav.py`. Add a new clientside callback (or extend the existing one with an additional Output) that updates `document.documentElement.dataset.palette` whenever `palette-store.data` changes:

```python
_app_ctx.app.clientside_callback(
    """
    function(palette_data) {
        if (palette_data && typeof palette_data === "string") {
            document.documentElement.dataset.palette = palette_data;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("palette-store", "data", allow_duplicate=True),
    Input("palette-store", "data"),
    prevent_initial_call=True,
)
```

**Why the no-update self-loop pattern**: Dash clientside callbacks must declare an Output. The side effect we want (setting `document.documentElement.dataset.palette`) doesn't have a natural Dash component to write to. Three options:

1. **Self-loop with `no_update`** (chosen): Output is `palette-store.data` itself with `allow_duplicate=True`, function always returns `no_update`. Side effect runs in the function body. Pro: zero new components, minimal diff. Con: looks odd to readers.
2. **Hidden sink div**: add an `html.Div(id="palette-dom-sink")` to the layout, output to its `data-dummy` attribute. Cleaner intent, but adds a layout element.
3. **Dash pattern-matching outputs to a `dcc.Store`**: similar to option 2 with more boilerplate.

Option 1 is the chosen pattern. The plan uses this approach because it's the smallest diff. Cross-reference: spec §Component 4 "Stage 2 reactive clientside callback".

- [ ] **Step 4: Verify**

```bash
cd btc_web && ../btc_venv/bin/python3 -m py_compile app.py callbacks/nav.py && echo OK
```

- [ ] **Step 5: Pre-merge browser verification of localStorage key format**

Start dev server, open Chrome DevTools, set palette to non-default via navbar, hard-refresh, inspect localStorage. Confirm:
- The exact key name (expected: `palette-store`)
- The exact value format (expected: a JSON-encoded string like `"cb-rg"` with quotes, OR a wrapping object)
- Adjust the inline script's parsing logic if the actual format differs

```bash
lsof -ti :8050 | xargs -r kill -9; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 & disown
sleep 8
btc_venv/bin/python3 <<'PY'
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_context().new_page()
    pg.goto("http://localhost:8050/", wait_until="networkidle", timeout=30000)
    time.sleep(2)
    raw = pg.evaluate('localStorage.getItem("palette-store")')
    print(f"localStorage palette-store: {raw!r}")
    # Set to cb-rg via navbar (or directly set localStorage and reload)
    pg.evaluate('localStorage.setItem("palette-store", JSON.stringify("cb-rg"))')
    pg.reload(wait_until="networkidle"); time.sleep(2)
    attr = pg.evaluate('document.documentElement.dataset.palette')
    print(f"documentElement.dataset.palette: {attr!r}")
    assert attr == "cb-rg", f"Expected cb-rg, got {attr!r}"
    print("OK — pre-paint script works")
    b.close()
PY
```

Expected: prints localStorage value, then `"OK — pre-paint script works"`.

- [ ] **Step 6: Commit**

```bash
git add btc_web/app.py btc_web/callbacks/nav.py
git commit -m "$(cat <<'EOF'
feat(colors): data-palette attribute on <html> for CSS palette switching

Phase 3, Task 23. Two changes:

1. Inline pre-paint script in app.py's index_string sets
   document.documentElement.dataset.palette synchronously from
   localStorage["palette-store"] BEFORE any Dash JS or first paint.
   Eliminates the flash-of-default-palette for users with a saved
   non-default palette.

2. Clientside callback in callbacks/nav.py updates
   document.documentElement.dataset.palette whenever palette-store
   changes (handles post-load palette dropdown switches).

Together these make CSS variable lookups via :root[data-palette="..."]
work correctly on first paint AND after subsequent palette switches.
The next commit migrates style.css to use var(--qs-*).
EOF
)"
```

---

### Task 24: Migrate `btc_web/assets/style.css` (MANDATORY visual regression)

**Files:**
- Modify: `btc_web/assets/style.css`

This is the largest single commit in the plan: **48 hex literal replacements + 99 rgba() literal replacements** (verified counts) + mandatory visual regression.

The 99 rgba literals contain 56 unique values, mostly black/white shadows with varying alpha (`rgba(0,0,0,0.3)` × 15 occurrences, etc.) plus a few brand-color overlays. Strategy: define **baked-alpha CSS variables** in `colors.py` (e.g. `SHADOW_DARK_30 = "rgba(0,0,0,0.30)"`) that the generator emits as `--qs-shadow-dark-30`, then replace each `rgba(...)` in style.css with `var(--qs-shadow-dark-30)`. The 56 unique rgba values become 56 new named constants. The plan accepts the scope expansion because the user's stated goal is "every color anywhere in the app".

**Preflight: ImageMagick required for visual regression diffing.**

```bash
command -v compare >/dev/null && command -v identify >/dev/null && command -v bc >/dev/null \
    && echo "PREFLIGHT_OK" \
    || { echo "Install with: sudo pacman -S imagemagick bc  (or apt: sudo apt install imagemagick bc)"; exit 1; }
```

- [ ] **Step 1: Take baseline screenshots of all 9 tabs in all 4 palettes**

```bash
mkdir -p /tmp/color_baselines
btc_venv/bin/python3 <<'PY'
from playwright.sync_api import sync_playwright
import time
PALETTES = ["default", "cb-brian", "cb-rg", "cb-full"]
TABS = list(range(1, 10))
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": 1600, "height": 1000})
    for pal in PALETTES:
        for tab in TABS:
            pg = ctx.new_page()
            pg.goto(f"http://localhost:8050/{tab}", wait_until="networkidle", timeout=30000)
            pg.evaluate(f'localStorage.setItem("palette-store", JSON.stringify("{pal}"))')
            pg.reload(wait_until="networkidle"); time.sleep(3)
            pg.screenshot(path=f"/tmp/color_baselines/{pal}_tab{tab}.png", full_page=False)
            pg.close()
    b.close()
print("36 baseline screenshots saved")
PY
ls /tmp/color_baselines/ | wc -l
```

Expected: `36`.

- [ ] **Step 2: Inventory style.css literals (hex AND rgba)**

```bash
echo "--- hex literals ---"
grep -nE '#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}\b' btc_web/assets/style.css | wc -l
echo "--- rgba literals ---"
grep -oE 'rgba?\([^)]*\)' btc_web/assets/style.css | wc -l
echo "--- unique rgba values ---"
grep -oE 'rgba?\([^)]*\)' btc_web/assets/style.css | sort -u
```

Expected: 48 hex lines, 99 rgba occurrences, 56 unique rgba values.

- [ ] **Step 3: Add any missing constants to `colors.py`**

Two passes:

**Pass A — hex literals**: for each hex literal in style.css that doesn't already have a named constant, ADD it (e.g. `BORDER_LIGHT`, `BORDER_DARK`, `HOVER_BG`).

**Pass B — rgba literals**: for each of the 56 UNIQUE rgba values, define a named constant. Naming convention: describe the *intent*, not the value. Examples:
```python
# Shadow / overlay alphas (palette-invariant)
SHADOW_DARK_10  = "rgba(0,0,0,0.1)"
SHADOW_DARK_20  = "rgba(0,0,0,0.2)"
SHADOW_DARK_25  = "rgba(0,0,0,0.25)"
SHADOW_DARK_30  = "rgba(0,0,0,0.3)"   # most common — 15 uses
SHADOW_DARK_50  = "rgba(0,0,0,0.5)"
OVERLAY_LIGHT_4 = "rgba(255,255,255,0.04)"
OVERLAY_LIGHT_90 = "rgba(255,255,255,0.9)"
BTC_ORANGE_30  = "rgba(247,147,26,0.3)"   # brand color with alpha
# … etc., one constant per unique rgba value
```

**Critical**: rgba constants in `colors.py` are STRING literals containing `rgba(...)`. The lint test allows colors.py to contain rgba string literals (it's the source of truth). Importers reference them via `from colors import SHADOW_DARK_30`.

The generator script (Task 2) emits these as CSS custom properties:
```css
:root {
    --qs-shadow-dark-10: rgba(0,0,0,0.1);
    --qs-shadow-dark-30: rgba(0,0,0,0.3);
    /* … */
}
```

Then style.css uses `box-shadow: 0 0 4px var(--qs-shadow-dark-30);` instead of literal rgba.

Regenerate artifacts and commit colors.py + artifacts as a separate sub-commit:

```bash
btc_venv/bin/python3 tools/generate_color_artifacts.py
git add btc_web/colors.py btc_web/assets/_colors_generated.css btc_web/assets/_colors_generated.js
git commit -m "feat(colors): add UI surface + shadow/overlay rgba constants for style.css migration"
```

- [ ] **Step 4: Replace each literal in style.css with `var(--qs-*)`**

For each hex literal AND each rgba literal in the inventory, replace it with the matching `var(--qs-...)` reference. Use `_colors_generated.css` as the reference for available variable names. Total: 48 hex + 99 rgba = 147 replacements.

Proceed in two passes for reviewability:
- Pass A: replace all 48 hex literals
- Pass B: replace all 99 rgba literals

After each pass, save the file and visually scan style.css for any remaining `#` or `rgba(` outside of `var(--...)` references:
```bash
grep -nE '#[0-9a-fA-F]{3,6}|rgba?\(' btc_web/assets/style.css
```
Expected after both passes: zero matches (except inside `var(...)` calls if any).

- [ ] **Step 5: Restart dev server and visual smoke test**

```bash
lsof -ti :8050 | xargs -r kill -9; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 & disown
sleep 8
curl -fsS http://localhost:8050/ -o /dev/null && echo SERVER_UP
```

- [ ] **Step 6: Take post-migration screenshots**

```bash
mkdir -p /tmp/color_after
btc_venv/bin/python3 <<'PY'
# (same script as Step 1 but writing to /tmp/color_after/)
PY
```

- [ ] **Step 7: Diff each pair, accept ≤1% pixel difference**

```bash
mkdir -p /tmp/color_diffs
total_pixels=0
fail_count=0
for f in /tmp/color_baselines/*.png; do
    name=$(basename $f)
    after="/tmp/color_after/$name"
    if [ ! -f "$after" ]; then
        echo "MISSING: $after"; continue
    fi
    diff_px=$(compare -metric AE "$f" "$after" "/tmp/color_diffs/$name" 2>&1 || true)
    total=$(identify -format "%w*%h" "$f" | bc)
    pct=$(echo "scale=4; $diff_px * 100 / $total" | bc)
    echo "$name: ${diff_px}px (${pct}%)"
    if (( $(echo "$pct > 1.0" | bc -l) )); then
        fail_count=$((fail_count+1))
    fi
done
echo "FAILED: $fail_count"
```

Expected: `FAILED: 0`. If any image is >1%, visually inspect the diff and confirm the change is intentional or fix the variable mapping.

- [ ] **Step 8: Commit**

```bash
git add btc_web/assets/style.css
git commit -m "$(cat <<'EOF'
refactor(colors): style.css uses var(--qs-*) for all 48 literals

Phase 3, Task 24. Replaces every hex literal in style.css with a CSS
custom property reference from _colors_generated.css. Palette
switching works automatically via :root[data-palette="..."] selectors
(set by the pre-paint script + clientside callback from the previous
commit).

Visual regression: 36 baselines (9 tabs × 4 palettes) compared
before/after. All within 1% pixel tolerance.
EOF
)"
```

---

## Phase 4 — JS migration

### Task 25: Migrate `btc_web/assets/chart_responsive.js`

**Files:**
- Modify: `btc_web/assets/chart_responsive.js`

- [ ] **Step 1: Inventory**

```bash
grep -n '#[0-9a-fA-F]\{6\}' btc_web/assets/chart_responsive.js
```

Expected: 3 literals in the `DEFAULTS` object (`grid_major_color: "#888888"`, `grid_minor_color: "#B0B0B0"`, `pt_color: "#2C3E50"`).

- [ ] **Step 2: Replace the DEFAULTS object with reads from `window.QS_COLORS`**

Edit the IIFE so that the DEFAULTS object pulls from `window.QS_COLORS` if available, falling back to hardcoded values if QS_COLORS isn't loaded (defensive — should never happen since `_colors_generated.js` loads first):

```javascript
var QS = window.QS_COLORS || {};
var DEFAULTS = {
    trace_width: 2.5,
    grid_major_width: 1.0,
    grid_major_color: QS.grid_major_color || "#888888",
    grid_minor_width: 0.8,
    grid_minor_color: QS.grid_minor_color || "#B0B0B0",
    pt_color: QS.scatter_point || "#2C3E50",
};
```

- [ ] **Step 3: Verify**

```bash
node --check btc_web/assets/chart_responsive.js && echo JS_OK
```

- [ ] **Step 4: Browser smoke test**

```bash
btc_venv/bin/python3 <<'PY'
from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_context().new_page()
    pg.goto("http://localhost:8050/1", wait_until="networkidle", timeout=30000)
    time.sleep(3)
    qs = pg.evaluate('window.QS_COLORS && window.QS_COLORS.grid_major_color')
    print(f"QS_COLORS.grid_major_color: {qs!r}")
    assert qs == "#888888", f"Expected #888888, got {qs!r}"
    print("OK")
    b.close()
PY
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/assets/chart_responsive.js
git commit -m "refactor(colors): chart_responsive.js reads defaults from window.QS_COLORS"
```

### Task 26: Migrate `btc_web/assets/plot_appearance.js`

Same pattern as Task 25. 3 literals in DEFAULTS object. Replace with `window.QS_COLORS` reads. Verify, commit.

---

## Phase 5 — Drift prevention

### Task 27: Create the lint test

**Files:**
- Create: `btc_web/test_colors_central.py`

- [ ] **Step 1: Write the test file**

Create `btc_web/test_colors_central.py` with this content:

```python
"""Lint test enforcing the color centralization invariant.

After the color centralization migration, no hex literal should appear
in btc_web/ except in:
  - btc_web/colors.py (the source of truth)
  - btc_web/assets/_colors_generated.css (generated artifact)
  - btc_web/assets/_colors_generated.js (generated artifact)
  - btc_web/test_*.py (test fixtures)
  - btc_web/assets/.deferred/*.js (easter-egg JS files, allowlisted)

Plus the generator script at tools/generate_color_artifacts.py and
this test file itself.

Spec: docs/superpowers/specs/2026-04-10-color-centralization-design.md
"""
import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BTC_WEB = _REPO_ROOT / "btc_web"
_TOOLS = _REPO_ROOT / "tools"

# Files that may legitimately contain hex literals
_ALLOWLIST = {
    _BTC_WEB / "colors.py",
    _BTC_WEB / "assets" / "_colors_generated.css",
    _BTC_WEB / "assets" / "_colors_generated.js",
    _BTC_WEB / "assets" / "bootstrap_flatly.min.css",  # vendor bundle (~331 hex literals)
    _TOOLS / "generate_color_artifacts.py",
    _BTC_WEB / "test_colors_central.py",
}

_ALLOWLIST_DIRS = {
    _BTC_WEB / "assets" / ".deferred",
    _BTC_WEB / "__pycache__",
}

# Vendor file patterns — any file matching is allowlisted regardless of path.
# Catches future minified vendor bundles dropped into assets/.
_VENDOR_PATTERNS = (
    re.compile(r"\.min\.css$"),
    re.compile(r"\.min\.js$"),
    re.compile(r"\.bundle\.css$"),
    re.compile(r"\.bundle\.js$"),
)

# Test fixtures with hardcoded color assertions are allowlisted as a class
_TEST_FILE_PATTERN = re.compile(r"^test_.*\.py$")

# Catches both #abcdef (6-digit) and #abc (3-digit) forms.
# Negative lookbehind/lookahead ensures we don't match a 6-digit form
# as the leading 3 chars of a longer string.
_HEX_PATTERN = re.compile(
    r"(?<![0-9a-fA-F#])#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})(?![0-9a-fA-F])"
)
_RGBA_PATTERN = re.compile(r'\brgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+(?:\s*,\s*[\d.]+)?\s*\)')


def _walk_btc_web():
    """Yield Path objects for every .py / .css / .js file in btc_web/
    that is NOT in the allowlist."""
    for path in _BTC_WEB.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in (".py", ".css", ".js"):
            continue
        if path in _ALLOWLIST:
            continue
        if any(parent in _ALLOWLIST_DIRS for parent in path.parents):
            continue
        if _TEST_FILE_PATTERN.match(path.name):
            continue
        if any(pat.search(path.name) for pat in _VENDOR_PATTERNS):
            continue
        yield path


def _strip_css_comments(src: str) -> str:
    """Remove /* ... */ comment blocks from CSS source."""
    return re.sub(r'/\*.*?\*/', '', src, flags=re.DOTALL)


def _strip_js_comments(src: str) -> str:
    """Remove /* ... */ and // ... comments from JS source.

    Strings are NOT stripped — hex literals legitimately live inside JS
    string defaults (e.g. plot_appearance.js DEFAULTS dict) and the lint
    must catch them.
    """
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.DOTALL)
    src = re.sub(r'//[^\n]*', '', src)
    return src


def _find_hex_literals_outside_string_constants(path: Path) -> list[tuple[int, str]]:
    """Find hex literals in a file, excluding allowed contexts.

    For .py: hex literals INSIDE string constants are still flagged
    (because that's where they live as Python source). However hex
    inside docstrings (which Python represents as Constant nodes
    immediately under FunctionDef/ClassDef/Module) is excluded.

    For .css/.js: comments are stripped before scanning.
    """
    src = path.read_text()
    if path.suffix == ".py":
        # For .py files, walk the AST and find Constant(value=str) nodes.
        # Skip any string constant whose parent is a docstring slot.
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return []
        hits = []
        # Build a set of (lineno, col) of docstring constants to exclude.
        doc_locations = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    doc_locations.add(node.body[0].value.lineno)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.lineno in doc_locations:
                    continue
                if _HEX_PATTERN.search(node.value):
                    for m in _HEX_PATTERN.finditer(node.value):
                        hits.append((node.lineno, m.group()))
        return hits
    elif path.suffix == ".js":
        cleaned = _strip_js_comments(src)
        hits = []
        for i, line in enumerate(cleaned.splitlines(), 1):
            for m in _HEX_PATTERN.finditer(line):
                hits.append((i, m.group()))
        return hits
    elif path.suffix == ".css":
        cleaned = _strip_css_comments(src)
        hits = []
        for i, line in enumerate(cleaned.splitlines(), 1):
            for m in _HEX_PATTERN.finditer(line):
                hits.append((i, m.group()))
        return hits
    return []


def _find_rgba_literals_in_python(path: Path) -> list[tuple[int, str]]:
    """Find rgba()/rgb() string literals in Python files via AST.

    The lint requires literal rgba(...) strings to be moved into colors.py
    OR converted to use _hex_alpha(constant, alpha) which produces the
    rgba() at runtime as a function return value, not as a source literal.

    AST inspection: walk every Constant(value=str) node and reject if its
    value matches the rgba/rgb pattern. This catches literal string forms
    only — function returns from _hex_alpha() are fine because they
    aren't string constants in the source.
    """
    if path.suffix != ".py":
        return []
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for m in _RGBA_PATTERN.finditer(node.value):
                hits.append((node.lineno, m.group()))
    return hits


def test_no_hex_literals_outside_colors_module():
    """No hex literal should appear outside colors.py + generated files."""
    leaks = []
    for path in _walk_btc_web():
        hits = _find_hex_literals_outside_string_constants(path)
        for lineno, hex_str in hits:
            leaks.append(f"{path.relative_to(_REPO_ROOT)}:{lineno} {hex_str}")
    assert not leaks, (
        "Hex literals found outside the centralized colors module:\n"
        + "\n".join(leaks)
        + "\n\nMove these to btc_web/colors.py and import."
    )


def test_no_rgba_literals_in_python():
    """No literal rgba(...) string in Python code. Use _hex_alpha(constant)
    or define a baked-alpha named constant in colors.py."""
    leaks = []
    for path in _walk_btc_web():
        if path.suffix != ".py":
            continue
        hits = _find_rgba_literals_in_python(path)
        for lineno, lit in hits:
            leaks.append(f"{path.relative_to(_REPO_ROOT)}:{lineno} {lit}")
    assert not leaks, (
        "Literal rgba()/rgb() strings found in Python source:\n"
        + "\n".join(leaks)
        + "\n\nReplace with _hex_alpha(named_constant, alpha) or add a "
        "baked-alpha named constant to btc_web/colors.py."
    )


def test_no_rgba_literals_in_css():
    """No literal rgba(...) in .css files. Use var(--qs-*) which references
    a baked-alpha named constant in colors.py."""
    leaks = []
    for path in _walk_btc_web():
        if path.suffix != ".css":
            continue
        cleaned = _strip_css_comments(path.read_text())
        for i, line in enumerate(cleaned.splitlines(), 1):
            for m in _RGBA_PATTERN.finditer(line):
                leaks.append(f"{path.relative_to(_REPO_ROOT)}:{i} {m.group()}")
    assert not leaks, (
        "Literal rgba()/rgb() values found in CSS:\n"
        + "\n".join(leaks)
        + "\n\nMove to btc_web/colors.py as a named constant and reference "
        "via var(--qs-...) from the generated _colors_generated.css."
    )


def test_generator_check_mode_passes():
    """Running tools/generate_color_artifacts.py --check should exit 0."""
    import subprocess
    result = subprocess.run(
        ["python", str(_TOOLS / "generate_color_artifacts.py"), "--check"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"generator --check failed (drift detected):\n{result.stderr}"
    )


def test_palette_key_parity():
    """Every palette must have an identical TOP-LEVEL key set AND
    identical inner model_colors key set."""
    import sys
    sys.path.insert(0, str(_BTC_WEB))
    import colors

    # Top-level parity
    key_sets = {pkey: set(pdict.keys()) for pkey, pdict in colors.PALETTES.items()}
    all_keys = set.union(*key_sets.values())
    top_divergences = {}
    for pkey, keys in key_sets.items():
        missing = all_keys - keys
        if missing:
            top_divergences[pkey] = sorted(missing)
    assert not top_divergences, (
        "Palette top-level key divergences:\n"
        + "\n".join(f"  {pkey}: missing {keys}" for pkey, keys in top_divergences.items())
    )

    # Inner model_colors parity — every palette must have the same set
    # of model keys. Catches drift like "lp4 added to default but not cb-rg".
    mc_sets = {pkey: set(pdict["model_colors"].keys())
               for pkey, pdict in colors.PALETTES.items()}
    all_models = set.union(*mc_sets.values())
    mc_divergences = {}
    for pkey, keys in mc_sets.items():
        missing = all_models - keys
        if missing:
            mc_divergences[pkey] = sorted(missing)
    assert not mc_divergences, (
        "Palette model_colors key divergences:\n"
        + "\n".join(f"  {pkey}: missing {keys}" for pkey, keys in mc_divergences.items())
    )


def test_css_var_consistency():
    """Every var(--qs-*) referenced in style.css must be defined in
    _colors_generated.css."""
    gen_css = (_BTC_WEB / "assets" / "_colors_generated.css").read_text()
    style_css = (_BTC_WEB / "assets" / "style.css").read_text()
    defined = set(re.findall(r'(--qs-[a-z0-9-]+):', gen_css))
    referenced = set(re.findall(r'var\((--qs-[a-z0-9-]+)\)', style_css))
    undefined = referenced - defined
    assert not undefined, (
        f"style.css references {len(undefined)} undefined CSS variables: "
        + ", ".join(sorted(undefined))
    )


def test_constant_export_coverage():
    """Every uppercase string/dict/list constant in colors.py must be
    either exported (matched by generator) or in __skip_export__."""
    import sys
    sys.path.insert(0, str(_BTC_WEB))
    import colors
    skip = getattr(colors, "__skip_export__", frozenset())
    src = (_BTC_WEB / "colors.py").read_text()
    tree = ast.parse(src)
    declared_uppercase = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    val = node.value
                    if isinstance(val, ast.Constant) and isinstance(val.value, str):
                        declared_uppercase.add(target.id)
                    elif isinstance(val, (ast.Dict, ast.List, ast.Tuple)):
                        declared_uppercase.add(target.id)
    # Every declared uppercase name should be either in skip or accessible from colors module
    missing = []
    for name in declared_uppercase:
        if name in skip:
            continue
        if not hasattr(colors, name):
            missing.append(name)
    assert not missing, f"Constants not accessible from colors module: {missing}"
```

- [ ] **Step 2: Run the test**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_colors_central.py -v --timeout=60 2>&1 | tail -30
```

Expected: All 6 tests pass (`test_no_hex_literals_outside_colors_module`, `test_no_rgba_literals_in_python`, `test_no_rgba_literals_in_css`, `test_generator_check_mode_passes`, `test_palette_key_parity`, `test_constant_export_coverage`). If any fail, surface the leaks (printed in the assertion message), fix them in the appropriate file (move literal to colors.py + import or var), regenerate artifacts if needed, re-run.

- [ ] **Step 3: Run the FULL pytest suite to confirm no regressions**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_*.py -q --timeout=120 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add btc_web/test_colors_central.py
git commit -m "$(cat <<'EOF'
test(colors): lint enforces single source of truth invariant

Phase 5, Task 27. New pytest test file with 6 checks:

1. test_no_hex_literals_outside_colors_module — walks btc_web/
   recursively, scans every .py/.css/.js for hex literals,
   excluding the allowlist (colors.py, generated artifacts,
   tests, .deferred easter-egg JS).

2. test_generator_check_mode_passes — runs the generator's
   --check mode and asserts on-disk artifacts match the
   generator's current output.

3. test_palette_key_parity — asserts all 4 palettes have
   identical key sets.

4. test_css_var_consistency — parses _colors_generated.css and
   asserts every var(--qs-*) referenced in style.css is defined.

5. test_constant_export_coverage — walks colors.py AST and
   asserts every uppercase string/dict/list constant is either
   accessible via the colors module OR in __skip_export__.

This locks the gate. Future drift gets caught at test time, not
in production.
EOF
)"
```

---

### Task 28: Final end-to-end review

**Files:** none modified.

- [ ] **Step 1: Run the entire pytest suite**

```bash
btc_venv/bin/python3 -m pytest btc_web/ -q --timeout=120 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 2: Run the canonical full app import smoke test**

```bash
cd btc_web && PYTHONPATH=".:../:../archive/btc_app" ../btc_venv/bin/python3 -c "import app; print('APP_IMPORT_OK')"
```

Expected: `APP_IMPORT_OK`.

- [ ] **Step 3: Manual smoke test on all 9 tabs**

Boot dev server, visit each tab, switch through all 4 palettes, verify charts render correctly with the right colors.

```bash
lsof -ti :8050 | xargs -r kill -9; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 & disown
sleep 8
curl -fsS http://localhost:8050/ -o /dev/null && echo SERVER_UP
```

Open Chrome to `http://localhost:8050/1` through `/9`, switch palette via navbar. Verify no visual regressions vs baseline screenshots from Task 24.

- [ ] **Step 4: Dispatch the full implementation reviewer**

Ask the orchestrator to dispatch the `feature-dev:code-reviewer` agent (or `general-purpose` agent with reviewer brief) with all the commits from Tasks 1–27 and the spec path. Reviewer verifies the entire migration matches the spec and flags any regressions or gaps.

- [ ] **Step 5: Address review findings**

Fix any issues the reviewer flags. Re-run the test suite. If clean, the migration is complete.

- [ ] **Step 6: No commit needed for this task**

Task 28 is verification only — no code changes unless the reviewer finds issues.

---

## Summary

| Phase | Tasks | Approx commits | Risk |
|---|---|---|---|
| 1 — Foundation | 1, 2, 3 | 3 | Low — parallel registry, no behavior change |
| 2 — Python migration (named files) | 4–18 | ~20 | Medium — touches many files; covered by regression tests |
| 2 — Python sweeps (missed files + rgba) | 19, 20, 21, 22 | ~30 (4+6+4+15+colors-sub-commits) | Medium — large surface; covered by lint test |
| 3 — CSS migration | 23, 24 | 2–3 (incl. colors.py sub-commit) + visual regression gate | Medium — large diff in style.css |
| 4 — JS migration | 25, 26 | 2 | Low — small diffs in 2 files |
| 5 — Drift prevention | 27 | 1 | Low — pure test |
| Final review | 28 | 0 | — |
| **Total** | **28 tasks** | **~60 commits** | — |

The plan is intentionally divided into many small commits (~60) so each subagent can complete a single file in one session and the orchestrator can review between commits. The four sweep tasks (19-22) catch every Python file with hex or rgba literals that wasn't named in Tasks 7-18. This ensures the lint test in Task 27 passes on first run.
