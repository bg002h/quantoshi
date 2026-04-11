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

# ── Additional palette-invariant chart constants ───────────────────
BLACK               = "#000000"   # default band/line color
WHITE               = "#ffffff"   # text on dark cells
DARK_TEXT           = "#111111"   # text on light cells
CLUSTER_MERGE_GRAY  = "#AAAAAA"   # merged overlapping edge annotations
LIGHT_GRAY          = "#CCCCCC"   # overlay fallback / faint lines
SPINE_COLOR_FALLBACK = "#999999"  # lighter border fallback (≈ #999)
THERMAL_NEUTRAL     = "#bdbdbd"   # thermal scale midpoint (Q50)
TODAY_LINE_COLOR    = "#FF6600"   # today vertical line default (matches palette today_line)
NON_QUANTIZED_MODEL_COLOR = "#8B4513"  # saddlebrown — single-trajectory models

# ── UI text hierarchy ────────────────────────────────────────────────
DIM_TEXT            = "#555555"   # secondary / hint text (layout sidebar, MC panels)
MUTED_TEXT          = "#666666"   # very muted text (tab hints, attributions)

# ── MC tier indicators ────────────────────────────────────────────────
MC_FREE_GREEN       = "#1a8f3c"   # free-tier label / cost indicator
MC_LIVE_AMBER       = "#c57600"   # live-tier label / stale-overlay warning

# ── MC premium figure styling ──────────────────────────────────────
MC_TITLE_COLOR      = "#996515"   # dark burnished gold — readable on light bg
MC_LEGEND_BORDER    = "#c9a227"   # legend border gold
KNIGHT_GOLD         = "#b8860b"   # dark goldenrod — knighthood UI / high-cost warning

# ── Surfaces / layout structure ──────────────────────────────────────
MODAL_DIVIDER_DARK  = "#444444"   # HR / 1px divider inside dark modals & config panels
PROGRESS_TRACK      = "#e0e0e0"   # progress-bar track background (e.g. CAGR loading bar)
CODE_BG             = "#f5f5f5"   # code block / monospace address background
BOOTSTRAP_LIGHT_BG  = "#f8f9fa"   # Bootstrap default light-surface background
BOOTSTRAP_BORDER    = "#dee2e6"   # Bootstrap default border/separator (table cells, cards)
BOOTSTRAP_TABLE_SELECT_BG     = "#cce5ff"   # Bootstrap table row selected background
BOOTSTRAP_TABLE_SELECT_BORDER = "#99caff"   # Bootstrap table row selected border
LIGHTBOX_BG         = "#1a1a2e"   # modal lightbox dark background (model-info images)
SPLASH_BRAND_DARK   = "#2c3e50"   # navbar color= prop + splash modal brand text
NEAR_BLACK          = "#1a1a1a"   # very dark near-black (MC badge bg, heatmap darkest stop)

# ── Static analysis pages (dark-theme pages served from api.py) ───────
# Used by /D (Residual FFT), /F (LPPL excess fits), /docs/* pages.
STATIC_PAGE_BG      = "#1a1a2e"   # page background (same as LIGHTBOX_BG, semantic alias)
STATIC_PAGE_TEXT    = "#cccccc"   # main body text on dark pages
STATIC_PAGE_ACCENT  = "#00d4ff"   # headings, links, table headers, border accents
STATIC_PAGE_SUBHEAD = "#8ecae6"   # secondary headings (h4-h6 in docs pages)
STATIC_PAGE_LINK    = "#FF9F40"   # anchor links on dark pages (warm orange)
STATIC_PAGE_LINK_HOVER = "#FFD080"  # hover state for anchor links
STATIC_PAGE_CODE_BG = "#16213e"   # code block / table header background
STATIC_PAGE_TH_BG   = "#2a3a5e"   # table header background in docs pages
STATIC_PAGE_DESC_BG = "#101a2e"   # .desc box background (info callouts)
STATIC_PAGE_DESC_TEXT = "#b8ccd8" # .desc box body text
STATIC_PAGE_FORMULA_BG = "#0e1624"  # .formula block background

# ── Error / validation surfaces ───────────────────────────────────────
DECOMP_ERROR_RED    = "#b71c1c"   # decomp warning / validation error text
ERROR_BG            = "#fff3f3"   # light-red error banner background
ERROR_BORDER        = "#f5c6cb"   # error banner border (Bootstrap danger-light)

# ── Table chrome ─────────────────────────────────────────────────────
TABLE_HEADER_BG     = "#f5f5f0"   # regime table header background (warm white)
TABLE_BORDER_LIGHT  = "#dddddd"   # light table / hr border (#ddd)
TABLE_BORDER_MID    = "#333333"   # medium table border (shares LOT_MARKER_OUTLINE value)
TABLE_BORDER_DARK   = "#555555"   # dark table header border (shares DIM_TEXT value)

# ── Model-panel summaries ──────────────────────────────────────────────
EPPL_SUMMARY_COLOR  = "#148C8C"   # EPPL current-model summary text (dark teal)

# ── Bubble chart specific ─────────────────────────────────────────
USER_MODEL_TRACE         = "#e67e22"   # user-drawn power-law line (pumpkin orange)
USER_MODEL_TICKER_ORANGE = "#e67e22"   # user model ticker entry color (same hue, distinct semantic)
UCL_LINE_COLOR      = "#ff6b6b"   # Unfairly Cheap Line
OLS_LINE_COLOR      = "#888888"   # OLS fit line (same value as FALLBACK_MODEL_GRAY)
SCAN_LINE_FALLBACK  = "#ffd93d"   # quantile scan line fallback (bright yellow)
LOT_MARKER_COLOR    = "#FFD700"   # stack lot dot (gold)
LOT_MARKER_OUTLINE  = "#333333"   # lot marker border

# ── Citadel chart specific ────────────────────────────────────────
CITADEL_SPENDING      = "#E74C3C"   # monthly spending line / depletion arrow (red)
CITADEL_BULLISH_QR    = "#8B4513"   # saddlebrown — bullish QR overlay line
CITADEL_BEARISH_QR    = "#228B22"   # forest green — bearish QR overlay line
CITADEL_SUCCESS_GREEN = "#27ae60"   # positive / success value in citadel panels

# ── Sparkline colors ──────────────────────────────────────────────
SPARKLINE_UP   = "#4cff88"   # navbar 24h sparkline up-trend (utils.py)
SPARKLINE_UP_2 = "#2ecc71"   # Celery task sparkline up-trend (tasks.py — emerald)
# Note: SPARKLINE_DOWN reuses UCL_LINE_COLOR = "#ff6b6b"

# ── Status colors (supplemental) ─────────────────────────────────
SILVER          = "#c0c0c0"   # silver — MC "NEW" badge text; same as CITADEL_OVERLAY_COLORS["cash"]
ERROR_RED_DARK  = "#cc0000"   # darker error red for payment failure (#c00 shorthand)

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

# ── Heatmap preset colormap schemes (c_lo, c_mid1, c_mid2, c_hi) ─────────────
# Used by the heatmap palette preset selector in snapshot_cb.py.
# Each tuple: (loss-color, neutral-mid1, mid2, gain-color).
HM_PRESET_PALETTES = {
    "finance": ("#b71c1c", "#ffffff", "#ffffff", "#1b5e20"),
    "forge":   ("#1b0a2e", "#2c2c3a", "#1b4332", "#ffd700"),
    "thermal": ("#0d47a1", "#80deea", "#e65100", "#c62828"),
    "bitcoin": ("#1a1a2e", "#2c3e50", "#f7931a", "#ffd700"),
    "ocean":   ("#0a1628", "#0d47a1", "#00838f", "#b2ebf2"),
    "mono":    ("#1a1a1a", "#555555", "#999999", "#e0e0e0"),
}

# ════════════════════════════════════════════════════════════════════
# SECTION 3 — Generation metadata
# ════════════════════════════════════════════════════════════════════

# Names that should NOT be emitted to generated CSS/JS artifacts.
# Default behavior: every module-level UPPER_CASE name with a string,
# dict, or list value gets emitted. Use this set to suppress.
# ── rgba-sourced named constants (used by _hex_alpha callers) ────────
# These are palette-invariant and cover the common rgba() patterns in the UI.
HEATMAP_GRADIENT_DEEP_PURPLE = "#1b0a2e"   # heatmap fallback colorscale dark stop (deep purple)
HEATMAP_GRADIENT_GOLD        = "#ffd700"   # heatmap fallback colorscale bright stop (gold)

LOG_MINOR_GRID_GRAY  = "#646464"   # rgba(100,100,100,*) — chart config annotation text, log-minor grid
WATERMARK_TEXT_COLOR = "#B4B4B4"   # rgba(180,180,180,*) — chart watermark URL text
TAX_DRAG_RED         = "#DC3232"   # rgba(220,50,50,*)  — cumulative taxes paid fill (citadel)
CTX_MENU_BG          = "#1E1E28"   # rgba(30,30,40,*)   — context menu / tooltip dark background
DANGER_HIGHLIGHT     = "#FF6464"   # rgba(255,100,100,*)— table row danger highlight (FAQ)
MC_AMBER             = "#DC7800"   # rgba(220,120,0,*)  — MC overlay amber bands / median line
MC_GHOST_GRAY        = "#969696"   # rgba(150,150,150,*)— MC ghost/reference fan bands
BADGE_GLOW_RED       = "#8B0000"   # rgba(139,0,0,*)    — "NEW" badge text-shadow glow (dark red)

__skip_export__ = frozenset({
    # Complex structured values — not useful as individual CSS variables.
    "HM_PRESET_PALETTES",
})


# ════════════════════════════════════════════════════════════════════
# SECTION 4 — Utility functions
# ════════════════════════════════════════════════════════════════════

def _hex_alpha(hex_color: str, alpha: float) -> str:
    """Convert a #rrggbb hex color to an rgba(...) string with the given alpha."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
