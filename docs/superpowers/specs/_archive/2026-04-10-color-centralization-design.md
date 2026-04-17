# Color Centralization — Single Source of Truth Design

**Status:** Reviewed & revised · 2026-04-10
**Architecture:** Python authoritative + generated CSS/JS artifacts (Approach A)
**Scope:** Every color used anywhere in `btc_web/`

> **Revision history**
> - v1: initial draft
> - v2 (this): incorporated review findings — added `DECOMP_COLORS`/`DECOMP_SUM_COLOR`, fixed `TICKER_MODEL_COLORS` actual values, resolved `today_line` palette-aware-only, corrected `MC_OVERLAY_COLORS` naming, fixed literal counts (style.css 68, api.py 39), made generator dev-only + CI `--check`, added inline pre-paint script for `data-palette`, made visual regression mandatory for CSS commit, switched `__all__` to opt-out, replaced timestamps with content hash, kept `MODEL_TRACE_COLORS` distinct from `DEFAULT["model_colors"]`, kept original `theme.py` constant names for zero-breakage.

---

## Problem

The Quantoshi codebase currently has **522 color literal lines** spread across **43 files** with at least **five** partially-overlapping sources of truth:

1. `btc_web/_app_ctx.py:PALETTES` — palette-aware registry (4 named palettes × ~30 keys each = 111 literals concentrated in one dict)
2. `btc_web/_app_ctx.py:DECOMP_COLORS` — palette-aware decomposition trace colors (4 palettes × 7 = 28 literals)
3. `btc_web/_app_ctx.py:DECOMP_SUM_COLOR` — palette-aware sum-trace color (4 entries)
4. `btc_web/_app_ctx.py:MODEL_TRACE_COLORS` — palette-INVARIANT fallback dict used by `_get_model_color` when a palette is missing entries. **Distinct values from `PALETTES["default"]["model_colors"]`** — see Component 1 note.
5. `btc_web/theme.py` — 5 chart theme constants (`PLOT_BG_COLOR`, `TEXT_COLOR`, `TITLE_COLOR`, `SPINE_COLOR`, `GRID_MAJOR_COLOR`)
6. `btc_web/callbacks/ticker.py:_MODEL_COLORS` — independent dict with its own values (different hues from `MODEL_TRACE_COLORS`) for ticker cycling

Plus hundreds of scattered ad-hoc literals in figure builders, layouts, callbacks, CSS, and JS asset files. Today's session uncovered three concrete drift bugs caused by this scatter (DCA thermal mismatch, EPPL family-color fallback to gray, supercharger thermal-per-quantile inconsistency).

**Per-file literal counts (verified by reviewer):**

| File | Hex literals |
|---|---|
| `_app_ctx.py` | 111 |
| `style.css` | 68 |
| `api.py` (SVG generation) | 39 |
| `test_models.py` | 21 |
| `assets/.deferred/wizard.js` | 28 |
| `figures/common.py` | 13 |
| `figures/citadel.py` | 12 |
| `callbacks/ticker.py` | 12 |
| `chart_responsive.js` | 3 |
| `plot_appearance.js` | 3 |
| (43 files total) | 522 |

A change to BTC orange today requires editing **at least four files** in unpredictable places, with no test catching mismatches. The user has explicitly requested a *robust* system: every color in one place, no drift.

## Goal

A single source-of-truth Python module from which **every** color reference in the app — Python, CSS, JS, SVG generation — is derived, with:

- **Zero hex literals outside the source module** (lint-enforced)
- **Automatic propagation** to CSS and JS via generated artifacts
- **Palette switching preserved** for the four existing colorblind-safe palettes
- **Discoverable**: a single grep target for "where does this color live?"
- **Reviewable diffs**: a single Pull Request can audit any color change

## Out of scope

- Brand redesign (no color VALUES change in this spec; this is centralization only)
- Adding/removing palette variants
- Adding computed color helpers (lighten/darken/contrast) — defer to a follow-up
- Test fixture color literals — kept as-is in test files via lint allowlist
- The MC overlay's per-regime colors (already a separate small dict in `mc_overlay.py`, will be migrated but no new structure)
- Anything in the standalone PyQt5 desktop app (`archive/btc_app/`) — out of project scope

## Architecture

Three layers, single direction of flow:

```
                  ┌──────────────────────┐
                  │   btc_web/colors.py  │   ← single source of truth
                  └──────────┬───────────┘
                             │  imports / generator reads
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         Python files    Generator    Generated artifacts
         (figures,       (tools/      (assets/
          layouts,        generate_   _colors_generated.css
          callbacks,      color_      _colors_generated.js)
          api.py)         artifacts.       │
              │           py)              │
              │                            ▼
              │                      CSS files (style.css)
              │                      JS files (chart_responsive.js,
              │                                 plot_appearance.js, ...)
              ▼                            │
         direct import                use var(--qs-*) /
                                      window.QS_COLORS
```

**Direction is one-way**: nothing reads colors back from CSS or JS into Python. The Python module is authoritative; everything else derives.

---

## Component 1 — `btc_web/colors.py` (source of truth)

A single Python module containing every color in the app, organized by semantic role.

### Structure

```python
"""Single source of truth for every color in Quantoshi.

This module is the ONLY place hex color literals appear in the codebase.
A pytest lint test (test_colors_central.py) enforces this.

Consumers:
  - Python: `from colors import BTC_ORANGE, PALETTES, ...`
  - CSS:    `var(--qs-btc-orange)` from generated _colors_generated.css
  - JS:     `window.QS_COLORS.btc_orange` from generated _colors_generated.js

Workflow:
  1. Edit a color value here
  2. Restart the dev server (or run `python tools/generate_color_artifacts.py`)
  3. CSS / JS artifacts are regenerated automatically
  4. The lint test verifies nothing leaked back into other files
"""

# ════════════════════════════════════════════════════════════════════
# SECTION 1 — Palette-invariant constants
# ════════════════════════════════════════════════════════════════════

# ── Brand identity ────────────────────────────────────────────────
BTC_ORANGE          = "#f7931a"   # Bitcoin canonical orange
QUANTOSHI_TITLE     = "#1A3060"   # navbar wordmark / chart titles
QUANTOSHI_NAVY      = "#0a1929"   # navbar background

# ── Status / semantic ──────────────────────────────────────────────
ERROR_RED           = "#ff5252"
WARNING_AMBER       = "#ffa726"
SUCCESS_GREEN       = "#4caf50"
INFO_BLUE           = "#1976d2"

# ── Chart theme (palette-invariant — also re-exported from theme.py) ──
# These KEEP THEIR ORIGINAL NAMES for zero-breakage on existing importers.
PLOT_BG_COLOR       = "#FFFFFF"   # was theme.PLOT_BG_COLOR
TEXT_COLOR          = "#222222"   # was theme.TEXT_COLOR
TITLE_COLOR         = "#1A3060"   # was theme.TITLE_COLOR (alias of QUANTOSHI_TITLE)
SPINE_COLOR         = "#888888"   # was theme.SPINE_COLOR
GRID_MAJOR_COLOR    = "#888888"   # was theme.GRID_MAJOR_COLOR
GRID_MINOR_COLOR    = "#B0B0B0"
FALLBACK_MODEL_GRAY = "#888888"   # `_get_model_color` fallback (== SPINE_COLOR but semantically distinct)
SCATTER_POINT       = "#2C3E50"   # default data point dark slate

# ── UI surfaces ─────────────────────────────────────────────────────
MODAL_BG            = "#FFFFFF"
DRAWER_BG           = "#F5F5F5"
SECTION_CARD_BG     = "#FAFAFA"
FOCUS_RING          = "#1a6fa8"
LINK                = "#1a6fa8"

# ── Static SVG generation (api.py shareable badges) ────────────────
SVG_BADGE_BG        = "#1a3060"
SVG_BADGE_TEXT      = "#ffffff"
# … etc. for every static SVG color (39 entries to be cataloged)

# ── Palette-invariant model trace fallback dict ────────────────────
# IMPORTANT: this dict is INTENTIONALLY DISTINCT from
# DEFAULT["model_colors"] below. It is used by `_get_model_color()` as
# the dict-default when `palette.get("model_colors", ...)` is empty,
# and by some legacy code paths. Migrating these into a single dict
# would silently change visual colors. Kept as its own constant.
MODEL_TRACE_COLORS = {
    "bub":  "#DAA520",  # goldenrod (matches bubble composite)
    "qr":   "#B0BEC5",
    "pl":   "#00E5FF",
    "lppl": "#FF6D00",
    # … 23 more keys verbatim from current _app_ctx.py:94-130
}

# ── Ticker model colors (palette-invariant, distinct values) ──────
# These are the colors used by the navbar live-price ticker as it
# cycles through models. They have their own hue choices (NOT shared
# with MODEL_TRACE_COLORS or PALETTES["default"]["model_colors"]) and
# are kept as a distinct constant.
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
# Misnamed in the prior draft as "MC regime colors" — they are NOT
# regime colors (regimes are colored by index from delay_colors).
CITADEL_OVERLAY_COLORS = {
    "total":              "#1a3060",
    "btc_usd":            "#f7931a",
    "cash":               "#4caf50",
    "reserves_total":     "#2196f3",
    "investments_total":  "#9c27b0",
    # exact values to be transcribed from mc_overlay.py:841-845 during Phase 1
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
        # ↑ EXACT values from _app_ctx.py:123-130 — note these differ
        # from MODEL_TRACE_COLORS above (e.g. bub=#C8960C here vs
        # bub=#DAA520 in MODEL_TRACE_COLORS). Do NOT unify silently.
    },
    "thermal_stops": [
        (0.001, "#0d47a1"), (0.01, "#1565c0"), (0.015, "#1976d2"),
        (0.05, "#42a5f5"),  (0.10, "#80deea"), (0.25, "#b2dfdb"),
        (0.50, "#bdbdbd"),  (0.75, "#ffcc80"), (0.90, "#f7931a"),
        (0.95, "#e65100"),  (0.99, "#c62828"), (0.999, "#7f0000"),
    ],
    "non_quantized_model": "#8B4513",
    "delay_colors":  ["#00c853", "#fdd835", "#ff9100", "#ff5252", "#b71c1c"],
    "annot_colors":  ["#00a844", "#d4b12e", "#e07d00", "#d44040", "#8f1616"],
    "today_line":    "#FF6600",      # palette-aware (varies across palettes)
    "hm_c_lo":       "#2166AC",
    "hm_c_mid1":     "#F7F7F7",
    "hm_c_mid2":     "#FF8C00",
    "hm_c_hi":       "#CC1100",
    "hm_loss_text":  "#ff8a80",
    "hm_exceptional_text": "#ffd700",
    # NEW: decomposition trace colors per palette (was DECOMP_COLORS top-level)
    "decomp_colors": ["#E64A19", "#1976D2", "#388E3C", "#7B1FA2",
                       "#F57C00", "#00796B", "#5D4037"],
    "decomp_sum_color": "#000000",   # was DECOMP_SUM_COLOR["default"]
}

CB_BRIAN = { ... }   # exact values copied from _app_ctx.py:146-220 — no semantic changes
CB_RG    = { ... }
CB_FULL  = { ... }

PALETTES = {
    "default":  DEFAULT,
    "cb-brian": CB_BRIAN,
    "cb-rg":    CB_RG,
    "cb-full":  CB_FULL,
}
PALETTE_KEYS = tuple(PALETTES.keys())

# ════════════════════════════════════════════════════════════════════
# SECTION 3 — Generation metadata (opt-out, not opt-in)
# ════════════════════════════════════════════════════════════════════

# Names that should NOT be emitted to generated CSS/JS artifacts.
# Default behavior: every module-level UPPER_CASE name with a string,
# dict, or list value gets emitted. Use this set to suppress.
__skip_export__ = frozenset({
    # Add any constants that are Python-only here.
    # Currently empty — every constant above is exposed to CSS/JS.
})
```

> **Note on opt-out vs opt-in**: the previous draft used `__all__` as an opt-in allowlist. Reviewer flagged that this is brittle — a developer adds a new constant, forgets `__all__`, generator silently skips it, runtime references fail. The opt-out pattern is safer: every uppercase constant is exposed by default; only known-Python-only constants need to be marked. The lint test asserts every uppercase string-or-dict-or-list constant is either exported or in `__skip_export__`.

> **`PALETTE_LABELS`** (i18n strings keyed by palette name) is NOT a color and stays in `_app_ctx.py`. Not moved to `colors.py`.

### Conventions

- **ALL_CAPS_SNAKE_CASE** for top-level constants
- **Lower-case dict keys** inside palette dicts (for compatibility with existing call sites)
- **Section dividers** with double-rule comments for visual scanning
- **Inline comments** explaining non-obvious choices
- **`__skip_export__`** is the opt-out set: every uppercase string/dict/list constant is exposed to CSS/JS by default; only Python-only constants need to be added here. The lint test enforces this.

---

## Component 2 — `tools/generate_color_artifacts.py` (generator script)

A standalone Python script that reads `btc_web/colors.py` and writes two artifacts:

1. `btc_web/assets/_colors_generated.css`
2. `btc_web/assets/_colors_generated.js`

### Script behavior

```
usage: python tools/generate_color_artifacts.py [--check]

  --check    Exit non-zero if generated files would differ from current
             on-disk versions (CI guard against forgetting to regenerate)
```

### Invocation policy

- **Dev mode (`DEV=1 bash run_web.sh`)**: app.py calls `generate()` at startup. Fast (~50ms). Picks up `colors.py` edits on hot-reload.
- **Production (gunicorn, 5 workers)**: NEVER calls the generator at runtime. Race-condition risk between workers writing the same files. Production reads the checked-in artifacts as-is.
- **CI**: runs `python tools/generate_color_artifacts.py --check` as a pre-merge gate. Fails if `colors.py` was edited without regenerating artifacts.
- **Local manual**: any developer can run `python tools/generate_color_artifacts.py` directly to regenerate before committing.

This policy is enforced in `app.py` startup with `if os.environ.get("DEV"): generate_color_artifacts.run()`.

### CSS output format

The header uses a content hash of `colors.py`, NOT a timestamp — timestamps would make every regeneration a non-trivial diff and defeat `--check`.

```css
/* AUTO-GENERATED by tools/generate_color_artifacts.py — DO NOT EDIT.
   Source: btc_web/colors.py  Source-SHA256: a3f4...e9c1 */

:root {
    /* ── Palette-invariant constants ── */
    --qs-btc-orange: #f7931a;
    --qs-quantoshi-title: #1A3060;
    --qs-error-red: #ff5252;
    --qs-warning-amber: #ffa726;
    --qs-success-green: #4caf50;
    --qs-info-blue: #1976d2;
    --qs-plot-bg: #FFFFFF;
    --qs-plot-text: #222222;
    --qs-grid-major: #888888;
    --qs-grid-minor: #B0B0B0;
    --qs-modal-bg: #FFFFFF;
    --qs-drawer-bg: #F5F5F5;
    --qs-focus-ring: #1a6fa8;
    --qs-link: #1a6fa8;
    /* … */

    /* ── Default palette (active when no [data-palette] is set) ── */
    --qs-model-bub: #DAA520;
    --qs-model-qr: #B0BEC5;
    --qs-model-lppl: #FF6D00;
    /* … one entry per (palette × key) ── */
    --qs-delay-0: #00c853;
    --qs-delay-1: #fdd835;
    /* … */
    --qs-hm-c-lo: #2166AC;
    --qs-hm-c-mid1: #F7F7F7;
    /* … */
}

:root[data-palette="cb-brian"] {
    --qs-model-bub: #BF8C0A;
    --qs-model-qr: #556B2F;
    /* … */
}

:root[data-palette="cb-rg"] {
    --qs-model-bub: #F5793A;
    /* … */
}

:root[data-palette="cb-full"] {
    --qs-model-bub: #B8920C;
    /* … */
}
```

### JS output format

```javascript
/* AUTO-GENERATED by tools/generate_color_artifacts.py — DO NOT EDIT.
   Source: btc_web/colors.py  Source-SHA256: a3f4...e9c1 */
(function() {
    'use strict';
    window.QS_COLORS = {
        btc_orange:      "#f7931a",
        quantoshi_title: "#1A3060",
        error_red:       "#ff5252",
        plot_bg:         "#FFFFFF",
        plot_text:       "#222222",
        grid_major:      "#888888",
        grid_minor:      "#B0B0B0",
        scatter_point:   "#2C3E50",
        fallback_model_gray: "#888888",
        /* … */
    };
    window.QS_PALETTES = {
        "default": {
            model_colors: { bub: "#DAA520", qr: "#B0BEC5", /* … */ },
            thermal_stops: [[0.001, "#0d47a1"], /* … */],
            delay_colors: ["#00c853", "#fdd835", /* … */],
            today_line: "#FF6600",
            hm_c_lo: "#2166AC",
            /* … */
        },
        "cb-brian": { /* … */ },
        "cb-rg":    { /* … */ },
        "cb-full":  { /* … */ },
    };
    window.QS_TICKER_COLORS = {
        qr: "#B0BEC5", bub: "#DAA520", /* … */
    };
})();
```

### Naming convention for CSS variables

| Python source | CSS variable |
|---|---|
| `BTC_ORANGE` | `--qs-btc-orange` |
| `PLOT_BG_COLOR` | `--qs-plot-bg-color` |
| `DEFAULT["model_colors"]["bub"]` | `--qs-model-bub` |
| `DEFAULT["model_colors"]["hybppl_dd"]` | `--qs-model-hybppl-dd` |
| `DEFAULT["delay_colors"][0]` | `--qs-delay-0` |
| `DEFAULT["thermal_stops"][3]` | (not exposed — JS-only; thermal is a sequence, not individual vars) |
| `DEFAULT["hm_c_lo"]` | `--qs-hm-c-lo` |
| `DEFAULT["decomp_colors"][2]` | `--qs-decomp-2` |
| `DEFAULT["decomp_sum_color"]` | `--qs-decomp-sum` |

Rule: snake_case → kebab-case (every `_` becomes `-` via `key.replace("_", "-")`), scalar dict values → flatten with `--qs-{section}-{key}`, list values get an index suffix `--qs-{section}-{i}`. Multi-underscore keys like `hybppl_dd` become `hybppl-dd` cleanly.

### Idempotency guarantee

Running the generator twice with the same input produces byte-identical output. The `--check` flag is used in CI to fail builds where someone edited `colors.py` but forgot to commit the regenerated artifacts.

---

## Component 3 — Generated artifact loading

### CSS

`assets/_colors_generated.css` is auto-loaded by Dash. The leading underscore forces it to sort first alphabetically (ASCII `_` = 0x5F, before letters), so it loads BEFORE `style.css`. This means `style.css` can use `var(--qs-*)` references.

### JS

`assets/_colors_generated.js` is auto-loaded by Dash. The leading underscore again sorts it before all other JS files (`chart_responsive.js`, `plot_appearance.js`, `inputs.js`, etc.), guaranteeing `window.QS_COLORS` is defined before any consumer reads it.

### Verification

The lint test asserts both generated files exist and have a recent regeneration timestamp matching the current `colors.py` file hash.

---

## Component 4 — Palette switching mechanism

### Today

1. User selects palette from navbar dropdown
2. Clientside callback writes `palette-store.data = "cb-rg"`
3. Python figure callbacks receive `palette` in their params dict
4. `_get_palette(p)` resolves to the right palette dict
5. Figure builders use that palette's colors
6. Charts re-render with new colors

CSS doesn't know about palettes today. Hardcoded literals in `style.css` are the same regardless of palette selection.

### After this spec

**Two-stage palette propagation** to prevent first-paint flicker:

#### Stage 1: Pre-paint inline script (MANDATORY)

A small inline script lives at the top of `app.py`'s `index_string`, executed BEFORE Dash renders anything. It reads `localStorage["palette-store"]` and sets `document.body.dataset.palette` immediately. CSS variables apply on first paint. No flicker for users who have a saved non-default palette.

```html
<head>
  <script>
    (function() {
      try {
        var raw = localStorage.getItem("palette-store");
        if (raw) {
          var key = JSON.parse(raw);
          if (typeof key === "string") {
            document.documentElement.dataset.palette = key;
          }
        }
      } catch(e) {}
    })();
  </script>
  <!-- rest of head -->
</head>
```

(Setting `documentElement` (the `<html>` element) instead of `body` because `<body>` may not yet exist when the script runs synchronously in `<head>`. CSS selectors use `:root[data-palette="..."]` instead of `body[data-palette="..."]`.)

#### Stage 2: Reactive clientside callback (existing flow + one new output)

A Dash clientside callback in `callbacks/nav.py` watches `palette-store.data` and:
1. Returns the palette key for the existing dropdown sync (current behavior)
2. **NEW**: also calls `document.documentElement.dataset.palette = key` so post-load palette switches propagate to CSS

The new callback is added alongside the existing palette dropdown handler in `callbacks/nav.py`.

### Rollover

On page load, Stage 1 sets the attribute synchronously before any paint. Stage 2's callback then takes over for any subsequent palette switches. Persists across reloads via the `dcc.Store(storage_type="local")` mechanism.

---

## Component 5 — Drift prevention (`btc_web/test_colors_central.py`)

A pytest test that walks the codebase and asserts the centralization invariant.

### What it checks

1. **Hex literal scan**: walk `btc_web/` recursively (excluding `__pycache__`, `assets/_colors_generated.*`, test files, allowlisted files). For each `.py`, `.js`, `.css` file, search for hex patterns matching `#[0-9a-fA-F]{6}` and `#[0-9a-fA-F]{3,4}`.
   - **Skips lines beginning with `#`** (Python comments) — shell-comment-style false positive avoidance
   - **Skips content inside `"""..."""` and `'''...'''` triple-quoted blocks** (docstrings, SVG templates) — uses a tokenize-based pass for `.py` files instead of naive line regex
   - **Skips `// ...` and `/* ... */`** in `.js` and `.css` files
   - **Allows hex in f-strings only when wrapped by an imported constant**: pattern `f".*{constant_name}.*"` is fine; pattern `f".*#abc123.*"` is rejected
   - **For api.py SVG generation**: triple-quoted SVG templates are converted to f-strings with imported constants in Phase 2; lint runs after migration
   - Asserts ZERO matches outside the allowlist after Phase 2 completes

2. **rgba/rgb literal scan**: same walk, search for `rgb(...)` and `rgba(...)`.
   - In `.css` files: literal `rgba(...)` is rejected; must use `color: var(--qs-...)` (CSS custom properties don't support alpha overlays cleanly, but the few cases are migrated to dedicated alpha-baked constants like `BAND_FILL_OUTER`)
   - In `.py` files: literal `rgba(...)` strings are rejected, but the result of `_hex_alpha(constant, alpha)` is fine because it's a function call, not a literal. This requires AST inspection (Python's `ast` module) — regex alone gives false positives. Lint walks the AST and rejects only `Constant(value="rgba(...)")` nodes whose parent is NOT a function call.
   - In `.js` files: literal `rgba(...)` is rejected; consumers should `_hex_alpha`-equivalent in JS or use a constant

3. **CSS var consistency**: parse `_colors_generated.css` (extract all `--qs-*` definitions) and assert every `var(--qs-*)` reference in any other `.css` file exists in the generated set. Catches `var(--qs-typo)` references that would silently fall back to inherit.

4. **Generator freshness**: re-run the generator in `--check` mode in a tmp dir and assert the on-disk artifacts byte-match.

5. **Palette key parity**: assert all 4 palettes have identical key sets.
   - **Known finding**: when this check runs for the first time during Phase 1, it may discover existing palette divergences (e.g. some palettes have a `u1` model_color that others don't). These are real bugs in the current codebase; Phase 1 documents them and either harmonizes the keys (preferred) or adds them to a `known_divergent_keys` set with a TODO.

6. **Constant export coverage**: walk `colors.py` AST, find all module-level `Name = "..."`, `Name = {...}`, or `Name = [...]` assignments where `Name` is uppercase. Assert each is either present in the generated CSS/JS OR is in `__skip_export__`. Catches the "added a constant, forgot to expose" footgun.

### Allowlist

Keep the allowlist tiny:

- `btc_web/colors.py` — the source itself
- `btc_web/assets/_colors_generated.css` and `_colors_generated.js`
- `btc_web/test_*.py` — test fixtures
- `btc_web/assets/.deferred/*.js` — easter-egg JS files (separate from main app)
- Specific lines in specific files marked with a `# qs-color-allow` comment for rare unavoidable cases (e.g. base64 PNG data that happens to contain `#xxxxxx` in escaped form)

The allowlist is a Python dict in `test_colors_central.py`, with one entry per file and either `True` (whole file) or a list of acceptable substrings. Adding to the allowlist requires a code review.

---

## Component 6 — Migration sequencing

Done as a series of small commits, each independently testable. Order matters: layers depend on the layer below.

### Phase 1 — Foundation (no behavioral changes)

1. **Create `btc_web/colors.py`** with all constants migrated from `_app_ctx.py:PALETTES`, `theme.py`, `callbacks/ticker.py:_MODEL_COLORS`, and a sweep of all hardcoded literals (we WRITE them all in colors.py but don't yet update consumers — colors.py becomes a parallel registry). Commit.
2. **Create `tools/generate_color_artifacts.py`** with full output for both CSS and JS. Run once to produce initial artifacts. Commit artifacts. Commit.
3. **Wire generator into app startup**: `app.py` calls `tools.generate_color_artifacts.generate()` on import. Verify generated files reproduce. Commit.

### Phase 2 — Python migration

4. **`_app_ctx.py`**: replace inline `PALETTES` dict with `from colors import PALETTES`. Re-export `BTC_ORANGE` from colors. Commit.
5. **`theme.py`**: rewrite as a thin re-export from `colors.py`. Original constant names preserved (`PLOT_BG_COLOR`, `TEXT_COLOR`, `TITLE_COLOR`, `SPINE_COLOR`, `GRID_MAJOR_COLOR`) for zero-breakage on existing importers. Commit.
6. **`callbacks/ticker.py`**: import `TICKER_MODEL_COLORS` from colors, delete local dict. Commit.
7. **`figures/`** modules: replace literals with imports. One file per commit (bubble, dca, retire, supercharge, citadel, heatmap, residuals, common). 8 commits.
8. **`layout/`** modules: replace literals with imports. One commit per major file. ~5 commits.
9. **`callbacks/`** modules: replace literals with imports. ~5 commits.
10. **`api.py`**: replace SVG color literals with constant imports + f-strings. 1 commit.
11. **`mc_overlay.py`**: migrate. 1 commit.

### Phase 3 — CSS migration

12. **Add `data-palette` to `<html>`** in two places:
    - **Inline pre-paint script** in `app.py`'s `index_string` (sets `document.documentElement.dataset.palette` synchronously from localStorage before any paint)
    - **Clientside callback** in `callbacks/nav.py` that updates `document.documentElement.dataset.palette` whenever `palette-store.data` changes (handles post-load palette switches)
    - **Pre-merge verification step**: open dev server in Chrome, store a non-default palette via the navbar dropdown, hard-refresh, open DevTools → Application → Local Storage and confirm the EXACT key name Dash uses (Dash 4.0.0 with `dcc.Store(id="palette-store", storage_type="local")` is expected to use the bare id `palette-store`, but verify before shipping). Update the inline script's `localStorage.getItem(...)` argument if Dash prefixes the key.
    1 commit.
13. **Migrate `style.css`**: replace each of the 68 hex literals with `var(--qs-*)`. 1 large commit. **MANDATORY**: visual regression baseline screenshots taken before this commit, diffed after, ≤1% pixel difference required for merge. Take screenshots of all 9 tabs in all 4 palettes (36 baselines) using Playwright `pg.screenshot()`. Diff with `imagemagick compare`.

### Phase 4 — JS migration

14. **Migrate `chart_responsive.js`** (3 hex literals): read defaults from `window.QS_COLORS`. 1 commit.
15. **Migrate `plot_appearance.js`** (3 hex literals): same. 1 commit.
16. **Other `assets/*.js`** files: GREP-VERIFIED ZERO HEX LITERALS in `inputs.js`, `streak.js`, `parallax.js`, `ambient.js`, `sc_legend.js`, `drawer.js`, `faq_lightbox.js`, `hm_swipe.js`, `scanner.js`, `tab_*.js`. **No migration needed** for these — they're already clean.
17. **`assets/.deferred/*.js`** (wizard.js, knighting.js, blockdrop.js): **permanently allowlisted** in the lint test. These are easter-egg JS files loaded conditionally; they have their own visual identity and don't need to participate in the centralized color system. 33 hex literals stay where they are.

### Phase 5 — Drift prevention

17. **Create `test_colors_central.py`** with the four checks above. Initial run should be GREEN if all migrations completed. If RED, surface the leaks and fix. Commit.
18. **Wire the test into the regular pytest run** (it should already run — confirm). Commit if any pytest config changes.

Total: ~30 commits over the migration. Each commit is a small, reviewable diff. After Phase 2 commits, the app is functionally identical with Python centralized. After Phase 3, CSS is centralized. After Phase 4, JS too. Phase 5 locks the gate.

---

## Backward compatibility policy

- `_app_ctx.PALETTES` continues to exist as a re-export from `colors.py` for the entire migration. Any external code (or tests) that does `from _app_ctx import PALETTES` keeps working.
- `_app_ctx.BTC_ORANGE` similarly re-exported.
- `theme.py` either re-exports from `colors.py` (zero-breakage) OR is deleted and importers updated. Reviewer decides.
- `callbacks/ticker.py:_MODEL_COLORS` becomes `from colors import TICKER_MODEL_COLORS as _MODEL_COLORS` for in-file backward compat.
- After the full migration completes and lint test passes, an OPTIONAL cleanup commit can remove the re-exports. Defer until proven safe.

---

## Testing strategy

### Existing tests

- The full pytest suite (~900+ tests) must still pass at every commit.
- E2E Playwright tests (`test_plot_appearance_e2e.py`, `test_tax_e2e.py`) verify visual behavior.
- Figure builder tests in `test_figures.py` verify trace colors — these may need updating if test fixtures asserted specific hex strings. Allowed to update test assertions to import from `colors`.

### New tests

- `test_colors_central.py` — 4 checks listed in Component 5. Required to pass.
- A small `test_color_generator.py` — feeds the generator a tiny mock module and asserts the CSS/JS output format is stable. Idempotency check. Not strictly required but cheap insurance.

### Manual verification

After each phase:
1. Boot dev server, smoke test all 9 tabs render without errors
2. Switch palettes via navbar dropdown, verify chart colors AND CSS variables update
3. Check DevTools `:root[data-palette="..."]` for `--qs-*` definitions
4. Hard-refresh in Chrome to bust browser cache, re-verify

### Visual regression — MANDATORY for Phase 3 step 13

Required for the `style.css` migration commit. Optional but recommended for other phases.

**Procedure:**
1. Before Phase 3 step 13 starts: take baseline screenshots of all 9 tabs in all 4 palettes (36 PNGs) via Playwright headless Chromium
2. Run the migration
3. Take post-migration screenshots
4. Diff each pair with `compare -metric AE` (ImageMagick)
5. Acceptance: ≤1% pixel difference per image (allows font anti-aliasing variation)
6. Any image >1% must be visually inspected and explained before merge

Stored in `tests/visual_baselines/` (not checked into git — too large; regenerated locally before each migration).

---

## Edge cases handled

1. **Browser caching of `_colors_generated.css`**: served via Dash's `assets/` route which has appropriate cache headers. Versioning via Dash's hash mechanism. No special handling needed.

2. **Generator running on a fresh checkout where artifacts don't exist**: app.py's startup hook generates them. Acceptable cold-start cost (~50 ms).

3. **`_hex_alpha(constant, alpha)` calls**: the resulting `rgba(...)` strings are runtime-derived, not source literals. The lint walks `*.py` files for `rgba(` patterns but allowlists strings that are computed (e.g. f-strings or function returns). Specifically: pattern is `rgba\(\d+,\d+,\d+,[\d.]+\)` as a literal — reject. As a function call — fine.

4. **Test fixtures with hardcoded color assertions**: tests are allowlisted as a class. Migration can opportunistically update them to import from colors, but it's not required for the lint to pass.

5. **CSS shorthand properties**: `border: 1px solid #888888` becomes `border: 1px solid var(--qs-grid-major)`. Standard CSS variable usage; works in all modern browsers.

6. **`data-palette` attribute timing**: on first page load, the body attribute may not be set yet when CSS first applies. CSS variables fall back to the `:root` defaults (default palette), which is the correct first-render behavior. The clientside callback updates the attribute within ~50 ms of layout, before any user interaction.

7. **Palette key not in `PALETTES`**: defensive code in `_get_palette(p)` already falls back to "default" — preserved as-is.

8. **`MODEL_TRACE_COLORS` top-level dict**: currently exists in `_app_ctx.py` as a palette-INVARIANT fallback dict with values that are **NOT** identical to `PALETTES["default"]["model_colors"]`. After migration, it stays as its own constant in `colors.py` with its existing values. NOT collapsed into the default palette dict. Reviewer caught a silent visual regression risk in the v1 draft of this spec.

9. **Citadel asset overlay colors** in `mc_overlay.py` (lines 841-845): currently hardcoded. Move to `colors.py` as `CITADEL_OVERLAY_COLORS`. NOT regime colors — they are per-asset overlay colors for the Citadel multi-asset MC view (`total`, `btc_usd`, `cash`, `reserves_total`, `investments_total`).

10. **3-digit hex (`#fff`) vs 6-digit (`#FFFFFF`)**: lint regex matches both. Generator always emits 6-digit form for consistency.

11. **Alpha-channel hex (`#FFFFFFAA`)**: 8-digit form is rare in this codebase. Lint regex matches `#[0-9a-fA-F]{6,8}`. Treated same as 6-digit.

12. **Palette dropdown has palette names — not colors**: those are i18n strings, not color literals. Out of scope for this spec.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Migration introduces visual regressions | Phased commits + manual smoke tests + optional pixel diff |
| Generated files get out of sync with `colors.py` in git | `--check` mode + CI guard + startup regeneration |
| CSS variable browser support | All modern browsers support `var()` since 2017; quantoshi.xyz target audience already uses modern browsers (Plotly 6.x requires modern JS). Documented in CLAUDE.md. |
| Generated JS file collides with another `window.QS_*` global | Namespaced under `window.QS_COLORS` / `QS_PALETTES` / `QS_TICKER_COLORS`. Grep confirms no existing `window.QS_*` namespace usage. |
| Backward compat break for external `_app_ctx.PALETTES` consumers | Re-exports preserved for entire migration; cleanup is optional. |
| Lint test false positives on legitimate hex strings (commit hashes, base64) | Allowlist supports per-line `# qs-color-allow` exceptions. |
| Performance: generation at startup adds time | Measured ~50ms for the full 100+ color set. Negligible vs. other startup costs (MC cache load takes ~7s). |
| `:root[data-palette]` selector specificity | `:root { --qs-model-bub: ... }` has specificity (0,0,1); `:root[data-palette="x"] { --qs-model-bub: ... }` has (0,1,1). The attributed selector wins, override works correctly. Setting `data-palette` on `<html>` (`documentElement`) instead of `<body>` lets the inline pre-paint script run from `<head>` before `<body>` exists. |
| Tests that mock `_app_ctx.PALETTES` | Mock target shifts to `colors.PALETTES`; identified during migration test pass. |
| Palette switching visible flicker on page load | Body attribute set in clientside callback BEFORE first paint by attaching to `dash-renderer-init` event or via inline `<script>` in `index_string`. |

---

## Files touched

Counts updated from reviewer-verified literal counts.

| Category | Files | LOC delta (rough) |
|---|---|---|
| New: `colors.py` | 1 | +280 |
| New: `tools/generate_color_artifacts.py` | 1 | +180 |
| New: `assets/_colors_generated.css` | 1 | +220 (generated) |
| New: `assets/_colors_generated.js` | 1 | +140 (generated) |
| New: `test_colors_central.py` | 1 | +220 |
| Modified: `_app_ctx.py` | 1 | -160, +10 |
| Modified: `theme.py` | 1 | -10, +5 (re-export from colors) |
| Modified: `callbacks/ticker.py` | 1 | -15, +5 |
| Modified: `figures/*.py` | 8 | ~-5/+5 each |
| Modified: `layout/*.py` | 6 | ~-3/+3 each |
| Modified: `callbacks/*.py` | 6 | ~-3/+3 each |
| Modified: `api.py` | 1 | -39/+39 |
| Modified: `mc_overlay.py` | 1 | -6/+6 |
| Modified: `app.py` | 1 | +20 (startup hook + index_string inline script) |
| Modified: `callbacks/nav.py` | 1 | +10 (new clientside callback for `data-palette`) |
| Modified: `assets/style.css` | 1 | -68/+68 |
| Modified: `assets/chart_responsive.js` | 1 | -3/+3 |
| Modified: `assets/plot_appearance.js` | 1 | -3/+3 |
| **TOTAL** | **~45 files** | **net ~+800 lines** |

Estimated commit count: ~35 (slightly over the original ~30 estimate to accommodate per-file figure/layout/callback migration commits).

---

## Approval checklist

This spec (v2) is ready for user approval. State:

- [x] User confirmed scope (Architecture A, all 6 layers)
- [x] Generator location decided (`tools/generate_color_artifacts.py`)
- [x] `theme.py` decision (re-export with original constant names)
- [x] Reviewer agent examined v1, all 4 critical issues + 11 important + 8 minor addressed in v2
- [x] Spec committed to `docs/superpowers/specs/`

After user approval, transition to writing-plans skill to produce the implementation plan.

---

## Open questions resolved by reviewer feedback

1. ✅ **Generator script location**: `tools/generate_color_artifacts.py` at project root.
2. ✅ **`theme.py` fate**: keep as re-export, original constant names preserved (`PLOT_BG_COLOR`, etc.) for zero-breakage on existing importers.
3. ✅ **Citadel overlay colors** (formerly mislabeled "MC regime colors"): move to `colors.py` as `CITADEL_OVERLAY_COLORS`.
4. ✅ **Single CSS file** (`_colors_generated.css`).
5. ✅ **`style.css` not split** — out of scope.
6. ✅ **`window.QS_COLORS`** uppercase global namespace.
7. ✅ **`__all__` → `__skip_export__`** opt-out pattern for safer constant exposure.
8. ✅ **`today_line`** is palette-aware only — no `TODAY_LINE_DEFAULT` top-level constant.
9. ✅ **`MODEL_TRACE_COLORS`** stays as its own constant — distinct from `PALETTES["default"]["model_colors"]`.
10. ✅ **`PALETTE_LABELS`** stays in `_app_ctx.py` (i18n strings, not colors).
11. ✅ **`TICKER_MODEL_COLORS`** keeps its current ticker-specific values, NOT unified with `MODEL_TRACE_COLORS`.
12. ✅ **`DECOMP_COLORS` / `DECOMP_SUM_COLOR`** moved into per-palette dicts as `decomp_colors` and `decomp_sum_color` keys.
13. ✅ **Generator runs in dev only**, with `--check` mode in CI.
14. ✅ **Pre-paint inline script** in `index_string` is mandatory.
15. ✅ **Visual regression** is mandatory for Phase 3 step 13 (style.css migration).
16. ✅ **JS scope** confirmed: only `chart_responsive.js` + `plot_appearance.js` need migration. `.deferred/*.js` allowlisted permanently.
17. ✅ **Content-hash header** in generated artifacts, not timestamps.
18. ✅ **Palette key parity** check known to surface existing divergences during Phase 1.

## Remaining open questions for user

- None. All reviewer concerns are resolved with sensible defaults documented above. User should review the resolved decisions and either approve or override any specific item.
