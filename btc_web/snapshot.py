"""Snapshot / URL state encoding and decoding for share links."""

import json
import gzip
import base64
import logging
import os

import _app_ctx

log = logging.getLogger(__name__)

_SNAPSHOT_CONTROLS = [
    # ── Bubble tab ──
    ("bub-qs",            "value"),   # selected quantile lines
    ("bub-qs-mode",       "value"),   # quantile mode (default/advanced)
    ("bub-qs-adv",        "value"),   # advanced quantile selection
    ("bub-xscale",        "value"),   # x-axis scale (Log/Linear)
    ("bub-yscale",        "value"),   # y-axis scale (Log/Linear)
    ("bub-xrange",        "value"),   # x-axis year range [start, end]
    ("bub-yrange",        "value"),   # y-axis price range [lo, hi]
    ("bub-toggles",       "value"),   # shade/data/today/legend toggles
    ("bub-bubble-toggles","value"),   # bubble composite overlay toggles
    ("bub-n-future",      "value"),   # number of projected future bubbles
    ("bub-ptsize",        "value"),   # scatter point size (1–20)
    ("bub-ptalpha",       "value"),   # scatter point opacity (0–1)
    ("bub-stack",         "value"),   # starting BTC stack
    ("bub-show-stack",    "value"),   # show stack value in legend
    ("bub-use-lots",      "value"),   # use Stack Tracker lots for starting BTC
    ("scan-price",        "value"),   # scanner price input
    ("scan-date",         "value"),   # scanner date input
    ("scan-q",            "value"),   # scanner quantile input
    # ── Heatmap tab ──
    ("hm-entry-yr",       "value"),   # heatmap entry year
    ("hm-entry-q",        "value"),   # entry percentile (0.1–99.9%)
    ("hm-exit-range",     "value"),   # exit year range [start, end]
    ("hm-exit-qs",        "value"),   # exit quantile lines
    ("hm-mode",           "value"),   # colorscale mode (Segmented/DataScaled/Diverging)
    ("hm-b1",             "value"),   # segmented colorscale breakpoint 1 (CAGR %)
    ("hm-b2",             "value"),   # segmented colorscale breakpoint 2 (CAGR %)
    ("hm-c-lo",           "value"),   # color below breakpoint 1
    ("hm-c-mid1",         "value"),   # color at breakpoint 1
    ("hm-c-mid2",         "value"),   # color at breakpoint 2
    ("hm-c-hi",           "value"),   # color above breakpoint 2
    ("hm-grad",           "value"),   # gradient steps (cosmetic)
    ("hm-vfmt",           "value"),   # cell text format (cagr/price/both/stack/...)
    ("hm-cell-fs",        "value"),   # cell text font size
    ("hm-toggles",        "value"),   # heatmap display toggles
    ("hm-stack",          "value"),   # starting BTC stack
    ("hm-use-lots",       "value"),   # use Stack Tracker lots
    # ── DCA tab + Stack-celerator ──
    ("dca-stack",         "value"),   # starting BTC stack
    ("dca-use-lots",      "value"),   # use Stack Tracker lots
    ("dca-amount",        "value"),   # DCA amount per period ($)
    ("dca-freq",          "value"),   # DCA frequency (Daily/Weekly/Monthly/...)
    ("dca-freq-unlock",   "value"),   # custom frequency unlock
    ("dca-infl",          "value"),   # inflation rate (%)
    ("dca-yr-range",      "value"),   # simulation year range [start, end]
    ("dca-disp",          "value"),   # display mode (BTC/USD)
    ("dca-toggles",       "value"),   # log_y/dual_y/annotate/legend toggles
    ("dca-qs",            "value"),   # selected quantile lines
    ("dca-qs-mode",       "value"),   # quantile mode (default/advanced)
    ("dca-qs-adv",        "value"),   # advanced quantile selection
    ("dca-sc-enable",     "value"),   # Stack-celerator enable
    ("dca-sc-loan",       "value"),   # SC loan principal ($)
    ("dca-sc-rate",       "value"),   # SC annual interest rate (%)
    ("dca-sc-term",       "value"),   # SC loan term (months)
    ("dca-sc-type",       "value"),   # SC loan type (amortizing/interest_only)
    ("dca-sc-repeats",    "value"),   # SC loan repeat cycles (0=one-shot)
    ("dca-sc-entry-mode", "value"),   # SC entry price mode (live/model/custom)
    ("dca-sc-custom-price","value"),  # SC custom entry price ($)
    ("dca-sc-tax",        "value"),   # SC capital gains tax rate (%)
    ("dca-sc-rollover",   "value"),   # SC rollover (interest-only: defer repayment)
    # ── Retire tab ──
    ("ret-stack",         "value"),   # starting BTC stack
    ("ret-use-lots",      "value"),   # use Stack Tracker lots
    ("ret-wd",            "value"),   # withdrawal amount per period ($)
    ("ret-freq",          "value"),   # withdrawal frequency
    ("ret-freq-unlock",   "value"),   # custom frequency unlock
    ("ret-yr-range",      "value"),   # simulation year range [start, end]
    ("ret-infl",          "value"),   # inflation rate (%)
    ("ret-disp",          "value"),   # display mode (BTC/USD)
    ("ret-toggles",       "value"),   # log_y/dual_y/annotate/legend toggles
    ("ret-legend-pos",    "value"),   # legend position
    ("ret-qs",            "value"),   # selected quantile lines
    ("ret-qs-mode",       "value"),   # quantile mode (default/advanced)
    ("ret-qs-adv",        "value"),   # advanced quantile selection
    # ── Supercharger tab ──
    ("sc-stack",          "value"),   # starting BTC stack
    ("sc-use-lots",       "value"),   # use Stack Tracker lots
    ("sc-start-yr",       "value"),   # withdrawal start year
    ("sc-d0",             "value"),   # delay offset 0 (years before withdrawal)
    ("sc-d1",             "value"),   # delay offset 1
    ("sc-d2",             "value"),   # delay offset 2
    ("sc-d3",             "value"),   # delay offset 3
    ("sc-d4",             "value"),   # delay offset 4
    ("sc-freq",           "value"),   # withdrawal frequency
    ("sc-freq-unlock",    "value"),   # custom frequency unlock
    ("sc-infl",           "value"),   # inflation rate (%)
    ("sc-qs",             "value"),   # selected quantile lines
    ("sc-qs-mode",        "value"),   # quantile mode (default/advanced)
    ("sc-qs-adv",         "value"),   # advanced quantile selection
    ("sc-mode",           "value"),   # Mode A (fixed spending) / Mode B (fixed depletion)
    ("sc-wd",             "value"),   # Mode A: withdrawal amount per period ($)
    ("sc-end-yr",         "value"),   # Mode A: simulation end year
    ("sc-target-yr",      "value"),   # Mode B: target depletion year
    ("sc-disp",           "value"),   # display mode (BTC/USD)
    ("sc-toggles",        "value"),   # log_y/annotate/legend toggles
    ("sc-chart-layout",   "value"),   # chart layout (line/bands)
    ("sc-display-q",      "value"),   # single quantile display (line layout only)
    # ── Cross-tab settings ──
    ("bub-auto-y",        "value"),   # auto-fit Y axis to selected quantiles
    ("bub-legend-pos",    "value"),   # bubble legend position
    ("dca-legend-pos",    "value"),   # DCA legend position
    ("sc-legend-pos",     "value"),   # SC legend position
    ("main-tabs",         "active_tab"),  # active tab selection
    # ── Model display toggles ──
    ("dca-model-show",    "value"),   # QR/MC display toggle (DCA)
    ("ret-model-show",    "value"),   # QR/MC display toggle (Retire)
    ("sc-model-show",     "value"),   # QR/MC display toggle (SC)
    ("hm-model-show",     "value"),   # QR/MC display toggle (Heatmap)
    # ── MC model source ──
    ("dca-mc-model-src",  "value"),   # MC model source (DCA)
    ("ret-mc-model-src",  "value"),   # MC model source (Retire)
    ("sc-mc-model-src",   "value"),   # MC model source (SC)
    ("hm-mc-model-src",   "value"),   # MC model source (Heatmap)
    # ── Bubble overlay models ──
    ("bub-model-show",    "value"),   # PL/S2F overlay toggle (Bubble)
    # ── LPPL config panel (Bubble tab) ──
    # ── Defunct after display-models consolidation (2026-04-11) ──
    # The *-lppl-activate / *-hybppl-activate / *-eppl-activate tuples below
    # are retained for q3: link positional bit-index stability. The decoder
    # at line ~515 is positional; deleting these would silently corrupt all
    # pre-refactor share links. The components are rendered as hidden
    # placeholders by _serve_layout.
    ("bub-lppl-activate", "value"),   # master activate toggle ["yes"] or []
    ("lppl-n-freqs",      "value"),   # [1..4] checkbox multi-select
    ("lppl-weighted",     "value"),   # ["weighted"] or []
    ("lppl-no-13",        "value"),   # ["no13"] or []
    # ── Palette ──
    ("palette-store",     "data"),    # colorblind palette key
    # ── Heatmap model selector ──
    ("hm-active-model",   "data"),    # active heatmap model pill (bub/pl/s2f/mc)
    # ── MC controls (4 tabs x 9 controls) ────────────────────────────────
    # DCA MC
    ("dca-mc-enable",    "value"),
    ("dca-mc-start-yr",  "value"),
    ("dca-mc-entry-q",   "value"),
    ("dca-mc-years",     "value"),
    ("dca-mc-bins",      "value"),
    ("dca-mc-regime",    "value"),
    ("dca-mc-sims",      "value"),
    ("dca-mc-window",    "value"),
    ("dca-mc-advanced",  "value"),
    # Retire MC
    ("ret-mc-enable",    "value"),
    ("ret-mc-start-yr",  "value"),
    ("ret-mc-entry-q",   "value"),
    ("ret-mc-years",     "value"),
    ("ret-mc-bins",      "value"),
    ("ret-mc-regime",    "value"),
    ("ret-mc-sims",      "value"),
    ("ret-mc-window",    "value"),
    ("ret-mc-advanced",  "value"),
    # Heatmap MC
    ("hm-mc-enable",     "value"),
    ("hm-mc-start-yr",   "value"),
    ("hm-mc-entry-q",    "value"),
    ("hm-mc-years",      "value"),
    ("hm-mc-bins",       "value"),
    ("hm-mc-regime",     "value"),
    ("hm-mc-sims",       "value"),
    ("hm-mc-window",     "value"),
    ("hm-mc-advanced",   "value"),
    # Supercharger MC
    ("sc-mc-enable",     "value"),
    ("sc-mc-start-yr",   "value"),
    ("sc-mc-entry-q",    "value"),
    ("sc-mc-years",      "value"),
    ("sc-mc-bins",       "value"),
    ("sc-mc-regime",     "value"),
    ("sc-mc-sims",       "value"),
    ("sc-mc-window",     "value"),
    ("sc-mc-advanced",   "value"),
    # ── Heatmap palette ──────────────────────────────────────────────────
    ("hm-palette",       "value"),
    # ── Citadel Planner tab ──────────────────────────────────────────────
    ("cp-stack",            "value"),
    ("cp-use-lots",         "value"),
    ("cp-cash-init",        "value"),
    ("cp-cash-rate",        "value"),
    ("cp-res-short-init",   "value"),
    ("cp-res-short-rate",   "value"),
    ("cp-res-short-vol",    "value"),
    ("cp-res-med-init",     "value"),
    ("cp-res-med-rate",     "value"),
    ("cp-res-med-vol",      "value"),
    ("cp-res-long-init",    "value"),
    ("cp-res-long-rate",    "value"),
    ("cp-res-long-vol",     "value"),
    ("cp-inv-eq-init",      "value"),
    ("cp-inv-eq-rate",      "value"),
    ("cp-inv-eq-basis",     "value"),
    ("cp-inv-eq-vol",       "value"),
    ("cp-inv-bd-init",      "value"),
    ("cp-inv-bd-rate",      "value"),
    ("cp-inv-bd-basis",     "value"),
    ("cp-inv-bd-vol",       "value"),
    ("cp-spend",            "value"),
    ("cp-infl",             "value"),
    ("cp-spend-growth",     "value"),
    ("cp-high-q-enable",    "value"),
    ("cp-high-q-thresh",    "value"),
    ("cp-high-q-mode",      "value"),
    ("cp-high-q-rate",      "value"),
    ("cp-high-q-dur",       "value"),
    ("cp-high-q-split-cash","value"),
    ("cp-high-q-split-rs",  "value"),
    ("cp-high-q-split-rm",  "value"),
    ("cp-high-q-split-rl",  "value"),
    ("cp-high-q-split-eq",  "value"),
    ("cp-high-q-split-bd",  "value"),
    ("cp-low-q-enable",     "value"),
    ("cp-low-q-thresh",     "value"),
    ("cp-low-q-mode",       "value"),
    ("cp-low-q-rate",       "value"),
    ("cp-low-q-dur",        "value"),
    ("cp-low-q-split-cash", "value"),
    ("cp-low-q-split-rs",   "value"),
    ("cp-low-q-split-rm",   "value"),
    ("cp-low-q-split-rl",   "value"),
    ("cp-low-q-split-eq",   "value"),
    ("cp-low-q-split-bd",   "value"),
    ("cp-lump-cooldown",    "value"),
    ("cp-cash-floor",       "value"),
    ("cp-res-short-floor",  "value"),
    ("cp-res-med-floor",    "value"),
    ("cp-res-long-floor",   "value"),
    ("cp-cash-floor-growth","value"),
    ("cp-res-floor-growth", "value"),
    ("cp-scf-enable",       "value"),
    ("cp-scf-amount",       "value"),
    ("cp-scf-type",         "value"),
    ("cp-scf-rate",         "value"),
    ("cp-scf-term",         "value"),
    ("cp-scf-trigger",      "value"),
    ("cp-yr-range",         "value"),
    ("cp-freq",             "value"),
    ("cp-qs",               "value"),
    ("cp-model-src",        "value"),
    ("cp-disp",             "value"),
    ("cp-toggles",          "value"),
    ("cp-legend-pos",       "value"),
    ("cp-asset-model",      "value"),
    # ── Citadel MC controls ──
    ("cp-mc-enable",        "value"),
    ("cp-mc-start-yr",      "value"),
    ("cp-mc-entry-q",       "value"),
    ("cp-mc-years",         "value"),
    ("cp-mc-bins",          "value"),
    ("cp-mc-regime",        "value"),
    ("cp-mc-sims",          "value"),
    ("cp-mc-window",        "value"),
    ("cp-mc-advanced",      "value"),
    ("cp-mc-model-src",     "value"),
    # ── Citadel Tax controls ──
    ("cp-tax-toggle",       "value"),
    ("cp-tax-config",       "data"),
    ("cp-td-btc",           "value"),
    ("cp-td-cash",          "value"),
    ("cp-td-res-short",     "value"),
    ("cp-td-res-med",       "value"),
    ("cp-td-res-long",      "value"),
    ("cp-td-inv-eq",        "value"),
    ("cp-td-inv-bd",        "value"),
    ("cp-tf-btc",           "value"),
    ("cp-tf-cash",          "value"),
    ("cp-tf-res-short",     "value"),
    ("cp-tf-res-med",       "value"),
    ("cp-tf-res-long",      "value"),
    ("cp-tf-inv-eq",        "value"),
    ("cp-tf-inv-bd",        "value"),
    # ── Citadel Quick Scenario controls ──
    ("cp-scenario-wealth",    "data"),
    ("cp-scenario-regime",    "data"),
    ("cp-scenario-rules",     "data"),
    ("cp-scenario-start-yr",  "value"),
    ("cp-scenario-active",    "data"),
    # ── Phase 1 append-only additions (per-tab LPPL activate checkboxes) ──
    # Append-only to preserve bit-index ordering of earlier positions.
    # ── Defunct after display-models consolidation (2026-04-11) ──
    # The *-lppl-activate / *-hybppl-activate / *-eppl-activate tuples below
    # are retained for q3: link positional bit-index stability. The decoder
    # at line ~515 is positional; deleting these would silently corrupt all
    # pre-refactor share links. The components are rendered as hidden
    # placeholders by _serve_layout.
    ("dca-lppl-activate", "value"),
    ("ret-lppl-activate", "value"),
    ("sc-lppl-activate",  "value"),
    ("hm-lppl-activate",  "value"),   # placeholder for Phase 2
    # ── Component Decomposition (bubble tab) ──
    ("bub-decomp-model",      "value"),   # family dropdown
    ("bub-decomp-components", "value"),   # dynamic checklist — plain list
    ("bub-decomp-mode",       "value"),   # individual | reference | cumulative
    ("bub-decomp-show-formulas", "value"), # ["full", "selected"] checklist
    ("bub-view-mode",        "data"),    # price | cagr | resid pill
    # ── HybPPL config panel (all tabs) ──
    ("bub-hybppl-activate",  "value"),
    ("hybppl-cfg-a-nlog",    "value"),
    ("hybppl-cfg-a-ncal",    "value"),
    ("hybppl-cfg-a-log1d",   "value"),
    ("hybppl-cfg-a-log2d",   "value"),
    ("hybppl-cfg-a-cal1d",   "value"),
    ("hybppl-cfg-a-cal2d",   "value"),
    ("hybppl-cfg-b-enabled", "value"),
    ("hybppl-cfg-b-nlog",    "value"),
    ("hybppl-cfg-b-ncal",    "value"),
    ("hybppl-cfg-b-log1d",   "value"),
    ("hybppl-cfg-b-log2d",   "value"),
    ("hybppl-cfg-b-cal1d",   "value"),
    ("hybppl-cfg-b-cal2d",   "value"),
    ("dca-hybppl-activate",  "value"),
    ("ret-hybppl-activate",  "value"),
    ("sc-hybppl-activate",   "value"),
    ("hm-hybppl-activate",   "value"),
    # ── EPPL config panel (all tabs) ──
    ("bub-eppl-activate",    "value"),
    ("eppl-cfg-a-nlog",      "value"),
    ("eppl-cfg-a-ncal",      "value"),
    ("eppl-cfg-a-log1d",     "value"),
    ("eppl-cfg-a-log2d",     "value"),
    ("eppl-cfg-a-cal1d",     "value"),
    ("eppl-cfg-a-cal2d",     "value"),
    ("eppl-cfg-b-enabled",   "value"),
    ("eppl-cfg-b-nlog",      "value"),
    ("eppl-cfg-b-ncal",      "value"),
    ("eppl-cfg-b-log1d",     "value"),
    ("eppl-cfg-b-log2d",     "value"),
    ("eppl-cfg-b-cal1d",     "value"),
    ("eppl-cfg-b-cal2d",     "value"),
    ("dca-eppl-activate",    "value"),
    ("ret-eppl-activate",    "value"),
    ("sc-eppl-activate",     "value"),
    ("hm-eppl-activate",     "value"),
    # ── Custom Time Axis (Tab 1 only, appended 2026-04-13) ──
    # APPEND-ONLY: positional ordering must stay stable for q3: link compat.
    ("cta-active",         "value"),
    ("cta-scale",          "value"),
    ("cta-t0-cal",         "value"),
    ("cta-t0-cal-custom",  "date"),
    ("cta-t0-blk",         "value"),
    ("cta-t0-blk-custom",  "value"),
    ("cta-weighting",      "value"),
    ("cta-models",         "value"),
    # ── Residual QR sigma bands (Tab 1 only, appended 2026-04-15) ──
    ("bub-sigma-mode",     "value"),   # "constant" | "resqr"
    # ── Leverage Calculator tab ──
    ("lev-date",          "date"),
    ("lev-price",         "value"),
    ("lev-model",         "value"),
    ("lev-floor-q-store", "data"),
    ("lev-rb",            "value"),
    ("lev-rl",            "value"),
    ("lev-horizon",       "value"),
    ("lev-cagr",          "value"),
    ("lev-toggles",       "value"),
]

_SNAP_PREFIX_V4 = "q4:"   # current format (v4: sparse diff against fingerprint)
_SNAP_PREFIX    = "q3:"   # prior format (positional array w/ checklist bitmask)
_SNAP_PREFIX_V2 = "q2:"   # prior format (positional array, different control list)
_SNAP_PREFIX_V1 = "q1:"   # legacy format (dict-based)

# Why bitmask encoding: storing 17 quantile checkboxes as a list in JSON costs
# ~150 chars; a single bitmask integer costs ~5 chars. Across 20 checklist fields,
# this saves ~660 characters in share URLs — significant for link-sharing UX.
#
# All checklist component IDs → ordered list of their possible values.
# Encoded as bitmask integers in new links (bit i set ↔ opts[i] selected).
# Old q2 links store lists; the decoder handles both formats transparently
# via isinstance(val, int).
#
# IMPORTANT: hard-coded canonical quantile list, NOT `list(_app_ctx._ALL_QS)`.
# `_ALL_QS` is empty at snapshot.py module load time (it's populated later
# in app.py:443). Capturing it as `_QS_LIST = list(_app_ctx._ALL_QS)` would
# bind to an empty list — making every *-qs-adv / hm-exit-qs share link
# round-trip silently drop the user's advanced-quantile selection
# (bitmask-stripped to 0). Hard-coding the canonical list keeps the
# bitmask layout stable across deploys; the values match the QR_QUANTILES
# range filtered to (0.001, 0.999). If model_data.pkl ever changes the
# quantile grid, audit `test_snapshot.py::TestSnapshotDefaultsConsistency`
# for failures and update this list accordingly.
_QS_LIST = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
            0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
            0.95, 0.99, 0.999]

_CHECKLIST_OPTIONS = {
    # quantile band checklists (default mode: string band names)
    "bub-qs":             ["inner", "outer", "median"],
    "dca-qs":             ["inner", "outer", "median"],
    "ret-qs":             ["inner", "outer", "median"],
    "sc-qs":              ["inner", "outer", "median"],
    # heatmap exit quantiles (unchanged — still float values)
    "hm-exit-qs":         _QS_LIST,
    # quantile mode toggles (1 bit each — "advanced" or empty)
    "bub-qs-mode":        ["advanced"],
    "dca-qs-mode":        ["advanced"],
    "ret-qs-mode":        ["advanced"],
    "sc-qs-mode":         ["advanced"],
    # advanced quantile selections (same list as standard quantile checklists)
    "bub-qs-adv":         _QS_LIST,
    "dca-qs-adv":         _QS_LIST,
    "ret-qs-adv":         _QS_LIST,
    "sc-qs-adv":          _QS_LIST,
    # toggle/boolean checklists (string values)
    "bub-toggles":        ["shade", "show_ols", "show_data", "show_today", "show_legend", "minor_grid", "chart_zoom"],
    "bub-bubble-toggles": ["show_comp", "show_sup"],
    "bub-show-stack":     ["yes"],
    "bub-use-lots":       ["yes"],
    "hm-toggles":         ["colorbar", "chart_zoom"],
    "hm-use-lots":        ["yes"],
    "dca-use-lots":       ["yes"],
    "dca-freq-unlock":    ["yes"],
    "dca-toggles":        ["log_y", "annotate", "shade", "show_legend", "minor_grid", "chart_zoom", "discrete"],
    "dca-sc-enable":      ["yes"],
    "dca-sc-rollover":    ["yes"],
    "ret-use-lots":       ["yes"],
    "ret-freq-unlock":    ["yes"],
    "ret-toggles":        ["log_y", "annotate", "shade", "show_legend", "minor_grid", "chart_zoom", "discrete"],
    "sc-use-lots":        ["yes"],
    "sc-freq-unlock":     ["yes"],
    "sc-toggles":         ["annotate", "log_y", "shade", "show_legend", "minor_grid", "chart_zoom", "discrete"],
    "sc-chart-layout":    ["shade"],
    "bub-auto-y":         ["yes"],
    # NOTE: "bub" appended at position 32 (end) so the bubble model bit
    # round-trips correctly through bitmask encoding. Existing bit indices
    # 0-31 preserved → older share links decode unchanged. The widget value
    # emitted by these checklists is "bub" (not "qr"); snapshot_defaults
    # uses "bub" too — without this entry, encoded user state with "bub"
    # silently loses that bit. (Bug surfaced 2026-04-25 by /4 retire share
    # link round-trip showing missing BM model on chart.)
    "dca-model-show":     ["qr", "mc", "pl", "lppl", "lppl_w", "lp2", "lp2_w", "lp3", "lp3_w", "lp4", "lp4_w", "lp4_n13", "lp4_w_n13", "linppl", "hybppl", "hybppl_dd", "hyb2l", "hyb2c", "hyb2b", "hyb4d", "pca", "grdy", "eppl", "exp", "s2f", "ef", "gomp", "bpl", "u1", "plo", "sexp", "logi", "bub"],
    "ret-model-show":     ["qr", "mc", "pl", "lppl", "lppl_w", "lp2", "lp2_w", "lp3", "lp3_w", "lp4", "lp4_w", "lp4_n13", "lp4_w_n13", "linppl", "hybppl", "hybppl_dd", "hyb2l", "hyb2c", "hyb2b", "hyb4d", "pca", "grdy", "eppl", "exp", "s2f", "ef", "gomp", "bpl", "u1", "plo", "sexp", "logi", "bub"],
    "sc-model-show":      ["qr", "mc", "pl", "lppl", "lppl_w", "lp2", "lp2_w", "lp3", "lp3_w", "lp4", "lp4_w", "lp4_n13", "lp4_w_n13", "linppl", "hybppl", "hybppl_dd", "hyb2l", "hyb2c", "hyb2b", "hyb4d", "pca", "grdy", "eppl", "exp", "s2f", "ef", "gomp", "bpl", "u1", "plo", "sexp", "logi", "bub"],
    "hm-model-show":      ["qr", "mc", "pl", "lppl", "lppl_w", "lp2", "lp2_w", "lp3", "lp3_w", "lp4", "lp4_w", "lp4_n13", "lp4_w_n13", "linppl", "hybppl", "hybppl_dd", "hyb2l", "hyb2c", "hyb2b", "hyb4d", "pca", "grdy", "eppl", "exp", "s2f", "ef", "gomp", "bpl", "u1", "plo", "sexp", "logi", "bub"],
    "bub-model-show":     ["pl", "lppl", "lppl_w", "lp2", "lp2_w", "lp3", "lp3_w", "lp4", "lp4_w", "lp4_n13", "lp4_w_n13", "linppl", "hybppl", "hybppl_dd", "hyb2l", "hyb2c", "hyb2b", "hyb4d", "pca", "grdy", "eppl", "exp", "s2f", "ef", "bub", "qr", "gomp", "bpl", "plo", "sexp", "logi"],
    # Defunct after display-models consolidation (2026-04-11) — retained
    # for q3: link positional stability. See _SNAPSHOT_CONTROLS comment above.
    "bub-lppl-activate":  ["yes"],
    "dca-lppl-activate":  ["yes"],
    "ret-lppl-activate":  ["yes"],
    "sc-lppl-activate":   ["yes"],
    "hm-lppl-activate":   ["yes"],
    "lppl-n-freqs":       [1, 2, 3, 4],
    "lppl-weighted":      ["weighted"],
    "lppl-no-13":         ["no13"],
    "bub-hybppl-activate":  ["yes"],
    "dca-hybppl-activate":  ["yes"],
    "ret-hybppl-activate":  ["yes"],
    "sc-hybppl-activate":   ["yes"],
    "hm-hybppl-activate":   ["yes"],
    "hybppl-cfg-b-enabled": ["yes"],
    "bub-eppl-activate":    ["yes"],
    "dca-eppl-activate":    ["yes"],
    "ret-eppl-activate":    ["yes"],
    "sc-eppl-activate":     ["yes"],
    "hm-eppl-activate":     ["yes"],
    "eppl-cfg-b-enabled":   ["yes"],
    # MC enable/advanced checklists (1 bit each)
    "dca-mc-enable":    ["yes"],
    "dca-mc-advanced":  ["yes"],
    "ret-mc-enable":    ["yes"],
    "ret-mc-advanced":  ["yes"],
    "hm-mc-enable":     ["yes"],
    "hm-mc-advanced":   ["yes"],
    "sc-mc-enable":     ["yes"],
    "sc-mc-advanced":   ["yes"],
    # MC regime checklists (5 bits each — int values 0-4)
    "dca-mc-regime":    [0, 1, 2, 3, 4],
    "ret-mc-regime":    [0, 1, 2, 3, 4],
    "hm-mc-regime":     [0, 1, 2, 3, 4],
    "sc-mc-regime":     [0, 1, 2, 3, 4],
    # Citadel Planner checklists (cp-qs is a dropdown, not a checklist)
    "cp-toggles":       ["log_y", "annotate", "show_legend", "minor_grid", "chart_zoom"],
    "cp-use-lots":      ["yes"],
    "cp-scf-enable":    ["yes"],
    "cp-high-q-enable": ["yes"],
    "cp-low-q-enable":  ["yes"],
    # Citadel MC checklists
    "cp-mc-enable":     ["yes"],
    "cp-mc-advanced":   ["yes"],
    "cp-mc-regime":     [0, 1, 2, 3, 4],
    # NOTE: cp-tax-toggle is a dbc.Switch (bool), NOT a checklist.
    # It's encoded/decoded directly as a JSON bool — do not add it here.
    # ── Custom Time Axis (Tab 1, appended 2026-04-13) ──
    # cta-active: single-option checklist used as a toggle (1 bit).
    "cta-active":       ["yes"],
    # cta-models: order is LOAD-BEARING for bitmask encoding.
    # FREEZE THIS LIST. Reordering or removing entries breaks old share links.
    "cta-models":       ["pl", "qr", "bm_floor", "exp", "gomp"],
}


# ── Validation: every checklist ID in _CHECKLIST_OPTIONS must appear in _SNAPSHOT_CONTROLS
_snap_cids = {cid for cid, _ in _SNAPSHOT_CONTROLS}
_checklist_missing = set(_CHECKLIST_OPTIONS) - _snap_cids
assert not _checklist_missing, f"Checklist IDs not in _SNAPSHOT_CONTROLS: {_checklist_missing}"
del _snap_cids, _checklist_missing


def _list_to_mask(val, opts):
    """Encode a checklist value list as a bitmask integer."""
    if not val:
        return 0
    sel = set(val)
    return sum(1 << i for i, o in enumerate(opts) if o in sel)


def _mask_to_list(mask, opts):
    """Decode a bitmask integer back to a checklist value list."""
    return [opts[i] for i in range(len(opts)) if mask & (1 << i)]


def _encode_snapshot(state_dict, tab_filter=None):
    """v2: positional array — no key names, ~50% smaller than v1.

    All checklist fields (quantiles and toggles) are stored as bitmask
    integers for compactness.  Old links that stored lists are still decoded
    transparently.

    If tab_filter is a set of component IDs, only those controls (plus
    main-tabs) are encoded; all others become None and fall back to defaults
    on restore.
    """
    values = []
    for cid, prop in _SNAPSHOT_CONTROLS:
        val = state_dict.get(f"{cid}:{prop}")
        if tab_filter is not None and cid != "main-tabs" and cid not in tab_filter:
            val = None
        if val is not None and cid in _CHECKLIST_OPTIONS:
            val = _list_to_mask(val, _CHECKLIST_OPTIONS[cid])
        values.append(val)
    # ── Hybrid MC encoding: null-out MC controls for disabled tabs ────────
    _mc_prefixes = {"dca": "dca-mc-", "ret": "ret-mc-", "hm": "hm-mc-", "sc": "sc-mc-", "cp": "cp-mc-"}
    for _pfx_tab, _pfx_mc in _mc_prefixes.items():
        enable_idx = next(i for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS)
                          if cid == f"{_pfx_mc}enable")
        mc_on = values[enable_idx] not in (None, [], 0)
        if not mc_on:
            for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS):
                if cid.startswith(_pfx_mc) and cid != f"{_pfx_mc}model-src":
                    values[i] = None
    lots   = state_dict.get("_lots")
    payload = [values, lots]
    j = json.dumps(payload, separators=(',', ':'))
    return base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()


# ── Gzip-bomb guardrails ─────────────────────────────────────────────────────
# Attack: a 100KB URL hash decompressing to GBs of nulls OOMs a gunicorn
# worker. We cap both the input (base64 text) and the decompressed output
# to bounded sizes well above any legitimate snapshot (which runs ~2-4KB
# compressed, ~15KB decompressed for a full-scope share).
_SNAP_MAX_ENCODED = 32 * 1024       # 32 KB of base64 ≈ 24 KB raw gzip
_SNAP_MAX_DECOMPRESSED = 512 * 1024  # 512 KB JSON is plenty; normal <20 KB


def _safe_decompress(encoded):
    """Bounded gzip+base64 decode. Returns bytes or None on size-violation.

    Never raises on malformed input -- returns None for any failure path so
    the callers' `except Exception: return None` stays a belt-and-suspenders.
    """
    if not encoded or len(encoded) > _SNAP_MAX_ENCODED:
        return None
    try:
        raw = base64.urlsafe_b64decode(encoded)
    except Exception:
        return None
    # Stream-decode with a bounded read instead of gzip.decompress() which
    # will happily expand arbitrarily. Use GzipFile + .read(limit+1) and
    # reject if we hit the limit -- prevents any expansion ratio attack.
    import io as _io
    try:
        with gzip.GzipFile(fileobj=_io.BytesIO(raw)) as gf:
            out = gf.read(_SNAP_MAX_DECOMPRESSED + 1)
    except Exception:
        return None
    if len(out) > _SNAP_MAX_DECOMPRESSED:
        return None
    return out


def _decode_snapshot(encoded):
    """Decode v2 (positional array) snapshot.

    Checklist fields may be either a bitmask int (new links) or a list
    (old links) — both are handled transparently.
    """
    try:
        raw = _safe_decompress(encoded)
        if raw is None:
            return None
        payload = json.loads(raw)
        values, lots = payload
        # Forward/backward compat: pad or truncate to match current control count
        n_expected = len(_SNAPSHOT_CONTROLS)
        if len(values) < n_expected:
            log.info("Snapshot has %d controls, expected %d — padding with defaults",
                     len(values), n_expected)
            values.extend([None] * (n_expected - len(values)))
        elif len(values) > n_expected:
            log.info("Snapshot has %d controls, expected %d — truncating",
                     len(values), n_expected)
            values = values[:n_expected]
        state = {}
        for (cid, prop), val in zip(_SNAPSHOT_CONTROLS, values):
            if val is None:
                continue
            if cid in _CHECKLIST_OPTIONS and isinstance(val, int):
                val = _mask_to_list(val, _CHECKLIST_OPTIONS[cid])
            state[f"{cid}:{prop}"] = val
        if lots:
            state["_lots"] = lots
        return state
    except Exception:
        return None


def _decode_snapshot_v1(encoded):
    """Decode legacy v1 (dict-based) snapshot."""
    try:
        raw = _safe_decompress(encoded)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:
        return None


# ── q4: defaults registry ────────────────────────────────────────────────────
_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "snapshot_defaults_registry.json")
_registry_cache: dict | None = None


def _load_registry():
    """Cache the registry on first read. Returns {fp -> defaults_dict}.
    Resilient to a missing or malformed file - returns {} so q4: decode
    falls through to current SNAPSHOT_DEFAULTS."""
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache
    try:
        with open(_REGISTRY_PATH) as f:
            entries = json.load(f)
        _registry_cache = {e["fp"]: e["defaults"] for e in entries}
    except Exception:
        log.warning("q4: registry %s unavailable; falling back to current "
                    "SNAPSHOT_DEFAULTS for all decodes", _REGISTRY_PATH)
        _registry_cache = {}
    return _registry_cache


def _registry_lookup(fp):
    """Return historical defaults for fp, or None if fp not in registry."""
    return _load_registry().get(fp)


def _mc_null_out_diffs_v4(diffs):
    """Drop MC control diffs for tabs whose EFFECTIVE mc-enable state
    is disabled. Effective state = diff value if present (decoded from
    possible bitmask), else SNAPSHOT_DEFAULTS. Null-out fires only when
    effectively disabled - avoids clobbering Retire's user-changed
    downstream fields when ret-mc-enable defaults to ['yes']."""
    from snapshot_defaults import SNAPSHOT_DEFAULTS
    _mc_prefixes = ("dca-mc-", "ret-mc-", "hm-mc-", "sc-mc-", "cp-mc-")
    for pfx in _mc_prefixes:
        enable_cid = f"{pfx}enable"
        enable_idx = next(
            (i for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS)
             if cid == enable_cid),
            None,
        )
        if enable_idx is None:
            continue
        if str(enable_idx) in diffs:
            raw = diffs[str(enable_idx)]
            if enable_cid in _CHECKLIST_OPTIONS and isinstance(raw, int):
                eff = _mask_to_list(raw, _CHECKLIST_OPTIONS[enable_cid])
            else:
                eff = raw
        else:
            eff = SNAPSHOT_DEFAULTS.get(f"{enable_cid}:value")
        is_disabled = eff in (None, [], 0)
        if not is_disabled:
            continue
        for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS):
            if cid.startswith(pfx) and cid != f"{pfx}model-src":
                diffs.pop(str(i), None)


def _encode_snapshot_v4(state_dict, tab_filter=None):
    """v4: sparse diff against fingerprinted defaults.

    Returns "<fp>:<blob>". Caller prepends only "q4:" (NOT "q4:<fp>:" -
    the fp portion is already in the returned string).

    ALWAYS_ENCODE controls (dynamic-default fields) are emitted even
    when value matches the static placeholder, so link author's value
    survives day boundaries.
    """
    from snapshot_defaults import (SNAPSHOT_DEFAULTS, ALWAYS_ENCODE,
                                   _compute_snapshot_defaults_fingerprint)
    fp = _compute_snapshot_defaults_fingerprint()
    diffs = {}
    for i, (cid, prop) in enumerate(_SNAPSHOT_CONTROLS):
        if tab_filter is not None and cid != "main-tabs" and cid not in tab_filter:
            continue
        key = f"{cid}:{prop}"
        val = state_dict.get(key)
        default = SNAPSHOT_DEFAULTS.get(key)
        force = key in ALWAYS_ENCODE
        if not force:
            if val is None:
                continue
            if val == default:
                continue
        if val is not None and cid in _CHECKLIST_OPTIONS:
            val = _list_to_mask(val, _CHECKLIST_OPTIONS[cid])
        diffs[str(i)] = val
    _mc_null_out_diffs_v4(diffs)
    lots = state_dict.get("_lots")
    payload = [fp, diffs, lots]
    j = json.dumps(payload, separators=(',', ':'))
    blob = base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()
    return f"{fp}:{blob}"


def _decode_snapshot_v4(encoded_with_fp_prefix):
    """Decode v4 snapshot. Input form: '<8-char-fp>:<base64-blob>'
    (the 'q4:' prefix is stripped by the dispatcher).

    Returns state dict or None on failure / fingerprint tampering.
    Registry-evicted fingerprints fall back to current SNAPSHOT_DEFAULTS
    with a warning logged."""
    try:
        if ":" not in encoded_with_fp_prefix:
            return None
        fp_url, blob_b64 = encoded_with_fp_prefix.split(":", 1)
        if len(fp_url) != 8:
            return None
        raw = _safe_decompress(blob_b64)
        if raw is None:
            return None
        payload = json.loads(raw)
        if not (isinstance(payload, list) and len(payload) == 3):
            return None
        fp_payload, diffs, lots = payload
        if fp_url != fp_payload:
            return None
        if not isinstance(diffs, dict):
            return None
        from snapshot_defaults import SNAPSHOT_DEFAULTS
        historical = _registry_lookup(fp_payload)
        if historical is None:
            log.warning("q4: fingerprint %s not in registry; falling back "
                        "to current SNAPSHOT_DEFAULTS (some fields may "
                        "restore differently)", fp_payload)
            historical = SNAPSHOT_DEFAULTS
        state = {}
        for i, (cid, prop) in enumerate(_SNAPSHOT_CONTROLS):
            key = f"{cid}:{prop}"
            si = str(i)
            if si in diffs:
                v = diffs[si]
                if cid in _CHECKLIST_OPTIONS and isinstance(v, int):
                    v = _mask_to_list(v, _CHECKLIST_OPTIONS[cid])
                state[key] = v
            else:
                v = historical.get(key)
                if v is not None:
                    state[key] = v
        if lots:
            state["_lots"] = lots
        return state
    except Exception:
        return None
