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

import colorsys

# ════════════════════════════════════════════════════════════════════
# SECTION 1 — Palette-invariant constants
# ════════════════════════════════════════════════════════════════════

# ── Brand identity ────────────────────────────────────────────────
BTC_ORANGE          = "#f7931a"   # Bitcoin canonical orange
QUANTOSHI_TITLE     = "#162D50"   # navbar wordmark / chart titles (alias of TITLE_COLOR)
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
FOCUS_RING          = "#2563AA"
LINK                = "#2563AA"

# ── Static SVG generation (api.py shareable badges) ───────────────
SVG_BADGE_BG        = "#1a3060"
SVG_BADGE_TEXT      = "#ffffff"

# ── Chart theme (palette-invariant — also re-exported from theme.py) ──
# These KEEP THEIR ORIGINAL NAMES for zero-breakage on existing importers.
PLOT_BG_COLOR       = "#FAF9F6"   # cooler ivory — editorial feel, not "default Plotly"
TEXT_COLOR          = "#1E1E1E"
TITLE_COLOR         = "#162D50"
SPINE_COLOR         = "#BCB9B0"   # warm-tuned to match ivory bg
GRID_MAJOR_COLOR    = "#E2E0DB"   # warm gray, recedes behind data (was #888888)
GRID_MINOR_COLOR    = "#F0EEED"   # barely visible, log-scale reference (was #B0B0B0)
FALLBACK_MODEL_GRAY = "#888888"
SCATTER_POINT       = "#1A1A2E"   # deep slate-navy ink; "newspaper ink on ivory paper"

# ── Additional palette-invariant chart constants ───────────────────
BLACK               = "#000000"   # default band/line color
WHITE               = "#ffffff"   # text on dark cells
DARK_TEXT           = "#111111"   # text on light cells
CLUSTER_MERGE_GRAY  = "#AAAAAA"   # merged overlapping edge annotations
LIGHT_GRAY          = "#CCCCCC"   # overlay fallback / faint lines
SPINE_COLOR_FALLBACK = "#999999"  # lighter border fallback (≈ #999)
THERMAL_NEUTRAL     = "#bdbdbd"   # thermal scale midpoint (Q50)
HM_DEFAULT_RED      = "#d32f2f"   # CAGR heatmap loss (rwg/rbg presets)
HM_DEFAULT_GREEN    = "#2e7d32"   # CAGR heatmap gain (rwg/rbg presets)
TODAY_LINE_COLOR    = "#F7931A"   # today vertical line default — unified with BTC orange
NON_QUANTIZED_MODEL_COLOR = "#8B4513"  # saddlebrown — single-trajectory models

# ── UI text hierarchy ────────────────────────────────────────────────
DIM_TEXT            = "#555555"   # secondary / hint text (layout sidebar, MC panels)
MUTED_TEXT          = "#666666"   # very muted text (tab hints, attributions)
MUTED_SUMMARY_TEXT  = "#9a9a9a"   # inline italic summaries next to Display Models entries

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

# ── App background / body palette ─────────────────────────────────────
BODY_BG             = "#f4f3f0"   # app body background (warm off-white)
BODY_TEXT           = "#212529"   # Bootstrap default body text color
SIDEBAR_GRADIENT_EDGE = "#e8e6e1"  # controls-col gradient edge (warm gray)
SIDEBAR_BG          = "#ebeef1"   # export-row-polished background (cool light gray)
SPLASH_MODAL_GRADIENT_END = "#f8f6f0"  # splash modal gradient end (warm white)
CTRL_CARD_HOVER_BORDER = "#c8ccd0"  # ctrl-card hover border color
BORDER_MUTED        = "#ccc"      # muted border (shorthand #ccc)

# ── Navbar colors ──────────────────────────────────────────────────────
NAV_DARK            = "#2c3e50"   # navbar primary background (alias of SPLASH_BRAND_DARK)
NAV_DARKER          = "#1a252f"   # navbar darkest gradient stop
NAV_LIGHTER         = "#34495e"   # navbar lightest gradient stop
LINK_HOVER          = "#154e78"   # link hover color (darker than LINK)

# ── Typography / UI text ──────────────────────────────────────────────
BODY_TEXT_SECONDARY = "#555"      # secondary UI text (shorthand, same as DIM_TEXT)

# ── Stone/parchment palette (blockdrop + knighting ceremony UI) ───────
STONE_LIGHT         = "#7a6a4f"   # blockdrop stone surface light
STONE_MID_1         = "#665535"   # blockdrop stone mid tone 1
STONE_MID_2         = "#55472a"   # blockdrop stone mid tone 2
STONE_MID_3         = "#483c22"   # blockdrop stone mid tone 3
STONE_DARK          = "#3a3018"   # blockdrop stone dark
STONE_DARKEST       = "#2e2510"   # blockdrop stone darkest shadow
PARCHMENT_TEXT      = "#3a2210"   # collapsed drawer CONFIG text (dark brown on parchment)
MC_GOLD_BORDER      = "#d4820a"   # MC premium chart gold border
MC_GOLD_ACCENT      = "#b8600a"   # MC premium chart gradient start (darker gold)
MC_GOLD_MID         = "#e8960e"   # MC premium chart gradient mid (bright gold)
ACCENT_BREATH_MID   = "#fbb034"   # accentBreath keyframe mid color (lighter orange)
PIN_LABEL_COLOR     = "#8b6336"   # drawer pin label text (medium brown)
DARK_AMBER_TEXT     = "#6b5300"   # model-panel paid badge text (very dark amber)

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
STATIC_PAGE_BODY_TEXT   = "#dddddd"  # body text on dark static pages (#ddd)
STATIC_PAGE_BORDER      = "#444444"  # table/hr borders on dark static pages (#444)
STATIC_PAGE_MUTED       = "#888888"  # muted/dim text on dark static pages (#888)

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
        # ── Flagship 6 — editorial warm/cool dichotomy ──────────────────
        "bub":    "#C48209",  # amber gold — darkened for 3:1 AA on ivory
        "pl":     "#1B3352",  # deep slate navy
        "qr":     "#9B2244",  # burgundy-crimson
        "eppl":   "#1F6B5C",  # forest teal
        "hybppl": "#A8431C",  # rust-orange — darker+redder to separate from BM amber
        "lppl":   "#7B3D9E",  # medium purple — lightened to separate from PL navy
        # ── LPPL family variants inherit master ─────────────────────────
        "lp2": "#7B3D9E", "lp3": "#7B3D9E", "lp4": "#7B3D9E",
        # ── HybPPL family variants inherit master ───────────────────────
        "linppl": "#A8431C", "hybppl_dd": "#A8431C",
        "hyb2l": "#A8431C", "hyb2c": "#A8431C",
        "hyb2b": "#A8431C", "hyb4d": "#A8431C",
        # ── Secondary models (muted, stay out of the way) ──────────────
        "pca":  "#4A2660",  # darker aubergine
        "grdy": "#163D4C",  # darker slate
        "ef":   "#6B5B95",  # muted violet
        "exp":  "#8A8A8A",  # neutral gray (display-only demo)
        "s2f":  "#9B7E46",  # antique brass
        "gomp": "#2D5F4E",  # pine
        "bpl":  "#7F4F2A",  # walnut
        "u1":   "#1A1A2E",  # matches scatter ink — user's own hand
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
    "today_line": "#F7931A",
    "hm_c_lo": "#2166AC", "hm_c_mid1": "#F7F7F7",
    "hm_c_mid2": "#FF8C00", "hm_c_hi": "#CC1100",
    "hm_loss_text": "#ff8a80", "hm_exceptional_text": "#ffd700",
    "decomp_colors": ["#E64A19", "#1976D2", "#388E3C", "#7B1FA2",
                      "#F57C00", "#00796B", "#5D4037"],
    "decomp_sum_color": "#000000",
}

CB_BRIAN = {
    "model_colors": {
        # ── Flagship 6 (hand-picked for deuteranomaly, 2026-04-11) ──
        # User-chosen for reliable separation on the CB-Brian profile.
        "bub":    "#D8BD65",  # muted gold
        "qr":     "#64B5F6",  # light blue
        "pl":     "#2204FD",  # electric blue
        "eppl":   "#BD3737",  # brick red
        "hybppl": "#FE6100",  # saturated orange
        "lppl":   "#FFB000",  # amber
        # ── LPPL family variants inherit the LPPL master color ──
        "lp2":    "#FFB000", "lp3": "#FFB000", "lp4": "#FFB000",
        # ── HybPPL family variants inherit the HybPPL master color ──
        # (hyb2l/c/b/4d and linppl are hidden from Display Models; hybppl_dd
        # is a DD-adjusted variant under the same master.)
        "hybppl_dd": "#FE6100",
        "linppl":    "#FE6100",
        "hyb2l":     "#FE6100", "hyb2c": "#FE6100",
        "hyb2b":     "#FE6100", "hyb4d": "#FE6100",
        # ── Remaining Display Models entries cycle the 6 flagship colors ──
        # Order: BM → QR → PL → EPPL → HybPPL → LPPL, then restart.
        "pca":  "#D8BD65",  # [BM]
        "grdy": "#64B5F6",  # [QR]
        "ef":   "#2204FD",  # [PL]
        "exp":  "#BD3737",  # [EPPL]
        "s2f":  "#FE6100",  # [HybPPL]
        "gomp": "#FFB000",  # [LPPL]
        "bpl":  "#D8BD65",  # [BM] — cycle restart
        "u1":   "#64B5F6",  # [QR]
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
# Flat 4-preset dict. Each preset is site-palette-invariant; the site-wide
# palette determines which preset is auto-selected as the default (see
# PALETTE_DEFAULT_HM_PRESET below).
HM_PRESET_PALETTES = {
    # Red → White → Green (classic finance, light neutral)
    "rwg":  (HM_DEFAULT_RED, WHITE, WHITE, HM_DEFAULT_GREEN),
    # Red → Black → Green (classic finance, dark neutral; same max hues as rwg)
    "rbg":  (HM_DEFAULT_RED, "#2c2c2c", "#2c2c2c", HM_DEFAULT_GREEN),
    # Blue → White → Orange (CVD-safe — deuteranomaly/protanopia/tritanopia)
    "bwo":  ("#0d47a1", "#ffffff", "#ffffff", "#e65100"),
    # Monochrome grayscale
    "mono": ("#1a1a1a", "#555555", "#999999", "#e0e0e0"),
}

# Default heatmap preset per site-wide palette. Defaults palette gets the
# classic red-green; CB palettes default to blue-orange which stays
# distinguishable on all deuteranomaly / protanopia / tritanopia profiles.
PALETTE_DEFAULT_HM_PRESET = {
    "default":  "rwg",
    "cb-brian": "bwo",
    "cb-rg":    "bwo",
    "cb-full":  "bwo",
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

# ════════════════════════════════════════════════════════════════════
# SECTION 3b — Baked-alpha rgba constants for style.css migration
# ════════════════════════════════════════════════════════════════════
# These named constants represent specific rgba() literals found in
# style.css. They are defined here as strings (not computed) so they
# are usable directly in both Python and as CSS custom property values.
# Naming convention: <BASE>_<ALPHA_PERCENT> where alpha is scaled 0-100.

# Black overlays (rgba(0,0,0,*))
BLACK_A0            = "rgba(0,0,0,0)"      # fully transparent (knighting overlay start)
BLACK_A6            = "rgba(0,0,0,0.06)"   # blockdrop grain texture
BLACK_A8            = "rgba(0,0,0,0.08)"   # box shadow / controls-col shadow
BLACK_A10           = "rgba(0,0,0,0.1)"    # pitting radial-gradient
BLACK_A12           = "rgba(0,0,0,0.12)"   # pitting + mineral vein overlay
BLACK_A15           = "rgba(0,0,0,0.15)"   # blockdrop cube highlight edge
BLACK_A20           = "rgba(0,0,0,0.2)"    # blockdrop shadow / fragment shadow
BLACK_A25           = "rgba(0,0,0,0.25)"   # blockdrop crack overlay
BLACK_A30           = "rgba(0,0,0,0.3)"    # splash modal shadow / copy toast shadow
BLACK_A35           = "rgba(0,0,0,0.35)"   # blockdrop crack-4 / pin label drop-shadow
BLACK_A40           = "rgba(0,0,0,0.4)"    # blockdrop-label text-shadow / mini-block shadow
BLACK_A50           = "rgba(0,0,0,0.5)"    # knight-text shadow
BLACK_A60           = "rgba(0,0,0,0.6)"    # knight overlay dim target / pricePulse shadow
BLACK_A70           = "rgba(0,0,0,0.7)"    # blockdrop-cube box-shadow / faq-overlay bg

# White overlays (rgba(255,255,255,*))
WHITE_A3            = "rgba(255,255,255,0.03)"   # navbar dot texture
WHITE_A4            = "rgba(255,255,255,0.04)"   # blockdrop surface grain
WHITE_A8            = "rgba(255,255,255,0.08)"   # blockdrop highlight gradient
WHITE_A15           = "rgba(255,255,255,0.15)"   # nav-link hover bg
WHITE_A40           = "rgba(255,255,255,0.4)"    # export-row btn border
WHITE_A60           = "rgba(255,255,255,0.6)"    # user-model-fab active border
WHITE_A75           = "rgba(255,255,255,0.75)"   # nav-link color (inactive)
WHITE_A90           = "rgba(255,255,255,0.9)"    # price flash end color

# BTC orange overlays (rgba(247,147,26,*) — #f7931a)
BTC_ORANGE_A12      = "rgba(247,147,26,0.12)"   # blockdrop-tx background
BTC_ORANGE_A30      = "rgba(247,147,26,0.3)"    # blockdrop-tx border / blockdrop-label glow
BTC_ORANGE_A40      = "rgba(247,147,26,0.4)"    # pricePulse keyframe / knight flash end
BTC_ORANGE_A50      = "rgba(247,147,26,0.5)"    # logo-glow hover filter
BTC_ORANGE_A60      = "rgba(247,147,26,0.6)"    # knight-staff drop-shadow / wizard sprite
BTC_ORANGE_A80      = "rgba(247,147,26,0.8)"    # pricePulse start
BTC_ORANGE_A90      = "rgba(247,147,26,0.9)"    # wizard toast background

# User-model orange overlays (rgba(230,126,34,*) — #e67e22)
USER_ORANGE_A50     = "rgba(230,126,34,0.5)"    # draw-mode-active box-shadow
USER_ORANGE_A60     = "rgba(230,126,34,0.6)"    # fab-pulse shadow peak
USER_ORANGE_A90     = "rgba(230,126,34,0.9)"    # user-model-fab hover background

# Accent / cyan overlays (rgba(0,212,255,*) — #00d4ff)
CYAN_A8             = "rgba(0,212,255,0.08)"    # scan-output background
CYAN_A10            = "rgba(0,212,255,0.1)"     # scan-results row hover
CYAN_A60            = "rgba(0,212,255,0.6)"     # price-ring-dot box-shadow
CYAN_A70            = "rgba(0,212,255,0.7)"     # price-ring-pulse border

# Green overlays (rgba(0,200,80,*) / rgba(26,143,60,*))
FLASH_GREEN_A90     = "rgba(0,200,80,0.9)"      # priceFlashGreen text-shadow strong
FLASH_GREEN_A40     = "rgba(0,200,80,0.4)"      # priceFlashGreen text-shadow fade
INPUT_ACCEPT_A60    = "rgba(26,143,60,0.6)"     # input-accepted keyframe border glow

# Red overlays (rgba(255,60,60,*))
FLASH_RED_A90       = "rgba(255,60,60,0.9)"     # priceFlashRed text-shadow strong
FLASH_RED_A40       = "rgba(255,60,60,0.4)"     # priceFlashRed text-shadow fade

# Gold overlays (rgba(255,215,0,*) — #ffd700)
GOLD_A50            = "rgba(255,215,0,0.5)"     # knight-text text-shadow

# Warm cream / parchment overlays
STONE_CREAM_A80     = "rgba(255,220,160,0.8)"   # blockdrop-sublabel / mini-sublabel color
MINERAL_WARM_A25    = "rgba(180,160,120,0.25)"  # blockdrop mineral vein light
MINERAL_WARM_A10    = "rgba(200,180,140,0.1)"   # blockdrop mineral vein faint
STONE_WARM_A15      = "rgba(120,100,70,0.15)"   # blockdrop mineral circle

# Parchment gold overlays (rgba(212,168,85,*))
PARCHMENT_GOLD_A50  = "rgba(212,168,85,0.5)"    # collapsed drawer text-shadow

# MC premium gold overlay (rgba(212,130,10,*))
MC_GOLD_A35         = "rgba(212,130,10,0.35)"   # mc-premium-chart box-shadow glow

# Paid badge overlay (rgba(184,134,11,*))
PAID_BADGE_A12      = "rgba(184,134,11,0.12)"   # model-panel-paid-badge background

# Ambient toast (rgba(44,62,80,*) — dark navbar blue)
AMBIENT_TOAST_A92   = "rgba(44,62,80,0.92)"     # ambient toast background
# MC chart gray overlay (rgba(64,64,64,*))
MC_OVERLAY_A65      = "rgba(64,64,64,0.65)"     # mc-chart-overlay gray

__skip_export__ = frozenset({
    # Complex structured values — not useful as individual CSS variables.
    "HM_PRESET_PALETTES",
    # Font stacks — exported via __appearance_export__, not as CSS vars.
    "FONT_SANS", "FONT_BRAND", "FONT_MONO", "FONT_CONDENSED",
    # UI font size strings — not color values, don't belong in QS_COLORS/CSS vars.
    "UI_FONT_XS", "UI_FONT_SM", "UI_FONT_MD", "UI_FONT_BASE",
    "UI_FONT_LG", "UI_FONT_XL", "UI_FONT_XXL", "UI_FONT_HEADING",
    # Chart margins — dict values, not useful as CSS vars.
    "CHART_MARGIN", "CHART_MARGIN_HM",
    # Quantile shade constants — chart-side only, not CSS/JS vars.
    "Q_SHADE_STRENGTH", "Q_SHADE_EXPONENT", "Q_SHADE_L_TARGET",
    "BAND_FILL_MODE", "BAND_PASTEL_ALPHA",
})


# ════════════════════════════════════════════════════════════════════
# SECTION 4 — Utility functions
# ════════════════════════════════════════════════════════════════════

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert #rrggbb hex to (r, g, b) integer tuple."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex_alpha(hex_color: str, alpha: float) -> str:
    """Convert a #rrggbb hex color to an rgba(...) string with the given alpha."""
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


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


# ════════════════════════════════════════════════════════════════════
# SECTION 5 — Appearance constants (fonts, sizes, widths, opacities)
# ════════════════════════════════════════════════════════════════════
# Single source of truth for all non-color visual appearance.  Python
# and JS (via __appearance_export__ → window.QS_APPEARANCE) read from
# here.  CSS font stacks remain hand-maintained in style.css :root.

# ── Font stacks ──────────────────────────────────────────────────────
# Python/JS authoritative; CSS uses --font-brand etc. in style.css.
FONT_SANS      = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
FONT_BRAND     = "'DM Serif Display', Georgia, serif"
FONT_MONO      = "'SF Mono', 'Cascadia Code', 'JetBrains Mono', 'Fira Code', Menlo, Consolas, monospace"
FONT_CONDENSED = "Inter, 'Arial Narrow', sans-serif"

# ── Chart font sizes (base / mobile) — int values for Plotly layout.font.size ──
CHART_FONT_TITLE      = 15
CHART_FONT_SUBTITLE   = 13
CHART_FONT_BODY       = 12
CHART_FONT_LEGEND     = 10
CHART_FONT_WATERMARK  = 9
CHART_FONT_ANNOT      = 11

# ── Chart font sizes (desktop / LG) — applied via _apply_sans_typography ──
CHART_FONT_TITLE_LG      = 19
CHART_FONT_BODY_LG       = 14
CHART_FONT_TICK_LG       = 12
CHART_FONT_LEGEND_LG     = 11
CHART_FONT_ANNOT_LG      = 12
CHART_FONT_WATERMARK_LG  = 10

# ── UI font sizes (CSS strings for layout/*.py inline styles) ────────
# NOTE: these are CSS px strings, NOT Plotly int sizes.
UI_FONT_XS      = "9px"
UI_FONT_SM      = "10px"
UI_FONT_MD      = "11px"
UI_FONT_BASE    = "12px"
UI_FONT_LG      = "13px"
UI_FONT_XL      = "14px"
UI_FONT_XXL     = "16px"
UI_FONT_HEADING = "18px"

# ── Trace / line widths ──────────────────────────────────────────────
TRACE_WIDTH             = 2.5       # QR quantile lines
TRACE_WIDTH_OVERLAY     = 2.0       # alt-model overlay lines (TRACE_WIDTH * 0.8)
TRACE_WIDTH_COMPOSITE   = 2.0       # bubble composite line
TRACE_WIDTH_SUPPORT     = 1.5       # bubble support line
TRACE_WIDTH_TODAY       = 2.0       # vertical "today" line
TODAY_GLOW_WIDTH        = 6         # glow halo around today line
GRID_MAJOR_WIDTH        = 1.0
GRID_MINOR_WIDTH        = 0.8
DESKTOP_TRACE_MULT      = 1.5       # JS chart_responsive.js multiplier
DESKTOP_GRID_MULT       = 1.5       # JS chart_responsive.js multiplier

# ── Point / marker sizes ────────────────────────────────────────────
PT_SIZE_DEFAULT         = 5         # default scatter point size
PT_ALPHA_DEFAULT        = 0.3       # default scatter point alpha
MARKER_SIZE_SM          = 6
MARKER_SIZE_MD          = 7
MARKER_SIZE_LOT         = 10
MARKER_SIZE_HIGHLIGHT   = 12

# ── Opacities ────────────────────────────────────────────────────────
SHADE_ALPHA             = 0.08      # fill between adjacent quantile lines
TODAY_LINE_OPACITY      = 0.85
TODAY_GLOW_OPACITY      = 0.12
WM_OPACITY              = 0.35      # watermark logo opacity
WM_SIZE_X               = 0.07      # watermark logo width  (fraction of paper)
WM_SIZE_Y               = 0.10      # watermark logo height (fraction of paper)
LEGEND_BG_OPACITY       = 0.92

# ── Chart trace opacities ──
SUPPORT_LINE_OPACITY  = 0.6    # support/composite line opacity
UCL_LINE_OPACITY      = 0.9    # Unfairly Cheap Line trace
OLS_LINE_OPACITY      = 0.8    # OLS regression line
RESIDUAL_LINE_OPACITY = 0.8    # residual overlay line

# ── MC overlay opacities ──
MC_BAND_OUTER_ALPHA   = 0.08   # MC 5-95% fan band fill
MC_BAND_INNER_ALPHA   = 0.15   # MC 25-75% fan band fill
MC_GHOST_OUTER_ALPHA  = 0.06   # MC ghost ref 5-95% band
MC_GHOST_INNER_ALPHA  = 0.12   # MC ghost ref 25-75% band
MC_GHOST_MEDIAN_ALPHA = 0.4    # MC ghost median line
MC_MEDIAN_ALPHA       = 0.9    # MC active median line
MC_LEGEND_BG_ALPHA    = 0.7    # MC legend repositioned bg

# ── Supercharge / Citadel band fills ──
SC_ENVELOPE_ALPHA         = 0.3   # supercharge model envelope fill
SC_OVERLAY_ENVELOPE_ALPHA = 0.12  # supercharge overlay envelope fill
CITADEL_BAND_OUTER_ALPHA  = 0.15  # citadel P5-P95 MC band
CITADEL_BAND_INNER_ALPHA  = 0.30  # citadel P25-P75 MC band
CITADEL_GHOST_LINE_ALPHA  = 0.5   # citadel no-tax ghost line

# ── Chart UI elements ──
LOG_MINOR_GRID_ALPHA  = 0.35   # log-scale minor grid line alpha
ANNOT_TEXT_ALPHA       = 0.8    # config annotation text
WATERMARK_TEXT_ALPHA   = 0.65   # watermark "quantoshi.xyz" text
TAX_LINE_ALPHA         = 0.6    # cumulative taxes paid line/fill
HM_ENTRY_HIGHLIGHT_ALPHA = 0.06 # heatmap entry-year column highlight

# ── Navbar / UI chrome ──
TICKER_TEXT_ALPHA      = 0.9    # price ticker text on dark navbar
NAV_DIM_ALPHA          = 0.45   # ticker mode toggle label
NAV_TAGLINE_ALPHA      = 0.4    # "Cooler than you think" tagline
NAV_TOGGLE_ALPHA       = 0.5    # nav toggle dots + onion link
NAV_HR_ALPHA           = 0.12   # mobile nav drawer divider
MOBILE_TICKER_ALPHA    = 0.95   # mobile price ticker text
FOOTER_TEXT_ALPHA      = 0.35   # site footer text

# ── UI component opacities ──
CTX_MENU_BG_ALPHA      = 0.95   # context menu / user-model menu bg
CTX_MENU_SHADOW_ALPHA  = 0.5    # context menu box-shadow
OVERLAY_DIM_ALPHA      = 0.35   # modal overlay dim background
UM_INPUT_BG_ALPHA      = 0.1    # user model input field background
UM_INPUT_BORDER_ALPHA  = 0.3    # user model input field border
BADGE_INSET_ALPHA      = 0.1    # badge inset highlight
BADGE_GLOW_ALPHA       = 0.6    # badge text-shadow glow
FAQ_DANGER_BG_ALPHA    = 0.1    # FAQ crash row highlight
OVERLAY_CARD_SHADOW_ALPHA = 0.18  # citadel load overlay card shadow

# ── Scanner ──
SCANNER_ROW_HIGHLIGHT_ALPHA = 0.18  # row click highlight bg
SCANNER_ROW_OUTLINE_ALPHA   = 0.4   # row click outline

# ── Quantile opacity formula parameters (function in figures/common.py) ──
Q_OPACITY_FLOOR         = 0.1
Q_OPACITY_RANGE         = 0.45
Q_OPACITY_DECAY         = 0.5

# ── Quantile shade formula parameters (quantile_shade in Section 4) ──
Q_SHADE_STRENGTH    = 0.70
Q_SHADE_EXPONENT    = 0.80
Q_SHADE_L_TARGET    = 0.92
BAND_FILL_MODE      = "alpha"   # "alpha" = current translucent; "pastel" = opaque tinted
BAND_PASTEL_ALPHA   = 0.35

# ── Chart margins ────────────────────────────────────────────────────
CHART_MARGIN    = dict(l=10, r=25, t=55, b=35, autoexpand=False)
CHART_MARGIN_HM = dict(l=40, r=8, t=40, b=32)

# ── Appearance export set ────────────────────────────────────────────
# Names exported to window.QS_APPEARANCE in _colors_generated.js.
# The generator reads this set independently from _gather_top_level_constants
# (which only handles str/dict/list for QS_COLORS).  Numeric (int/float) and
# font string constants listed here go to QS_APPEARANCE only.
__appearance_export__ = frozenset({
    "TRACE_WIDTH", "TRACE_WIDTH_OVERLAY", "GRID_MAJOR_WIDTH", "GRID_MINOR_WIDTH",
    "DESKTOP_TRACE_MULT", "DESKTOP_GRID_MULT",
    "PT_SIZE_DEFAULT", "PT_ALPHA_DEFAULT",
    "FONT_SANS", "FONT_BRAND", "FONT_MONO",
})
