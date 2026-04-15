"""Chart update callbacks — Bubble, Heatmap, DCA, Retire, Supercharge."""

import math

import dash
from dash import Input, Output, State, ctx, callback, html
import numpy as np
import pandas as pd

import _app_ctx
from colors import (
    LINK, FALLBACK_MODEL_GRAY, BLACK,
    MODEL_TRACE_COLORS, LOT_MARKER_COLOR, LOT_MARKER_OUTLINE,
    DECOMP_ERROR_RED, ERROR_BG, ERROR_BORDER,
    TRACE_WIDTH_COMPOSITE, TRACE_WIDTH_SUPPORT,
    UI_FONT_SM, UI_FONT_MD,
)


# ══════════════════════════════════════════════════════════════════════════════
# Display Models family summaries — single clientside multi-output callback.
# Computes LPPL/HybPPL/EPPL summary strings in JS and writes them directly to
# all 12 `{prefix}-{fam}-summary-inline` spans + the `display-model-summaries`
# Store (still read by heatmap status row + an apply_snapshot State).
# ══════════════════════════════════════════════════════════════════════════════
_app_ctx.app.clientside_callback(
    '''
    function(nFreqs, weighted, no13,
             aNlog, aNcal, aLog1d, aLog2d, aCal1d, aCal2d,
             bEn, bNlog, bNcal, bLog1d, bLog2d, bCal1d, bCal2d,
             eaNlog, eaNcal, eaLog1d, eaLog2d, eaCal1d, eaCal2d,
             ebEn, ebNlog, ebNcal, ebLog1d, ebLog2d, ebCal1d, ebCal2d) {
        var lppl;
        var ns = (nFreqs || []).slice().sort();
        if (!ns.length) {
            lppl = "(no flavor)";
        } else {
            var names = {1: "LPPL\u2081", 2: "LPPL\u2082", 3: "LPPL\u2083", 4: "LPPL\u2084"};
            lppl = ns.map(function(n){ return names[n] || ("LPPL" + n); }).join("+");
            if (weighted && weighted.indexOf("weighted") !== -1) lppl += " (w)";
            if (no13 && no13.indexOf("no13") !== -1) lppl += " (no \u03c9\u224813)";
        }
        function spec(n, d1, d2) {
            if (n === 0) return "0";
            if (n === 1) return "1" + (d1 || "d");
            return "2" + (d1 || "d") + (d2 || "d");
        }
        function fam(nlogA, ncalA, l1A, l2A, c1A, c2A, bE, nlogB, ncalB, l1B, l2B, c1B, c2B) {
            var t = spec(nlogA, l1A, l2A) + "+" + spec(ncalA, c1A, c2A);
            if (bE && bE.length > 0) {
                t += " / " + spec(nlogB, l1B, l2B) + "+" + spec(ncalB, c1B, c2B);
            }
            return t;
        }
        var hyb = fam(aNlog, aNcal, aLog1d, aLog2d, aCal1d, aCal2d,
                      bEn, bNlog, bNcal, bLog1d, bLog2d, bCal1d, bCal2d);
        var epp = fam(eaNlog, eaNcal, eaLog1d, eaLog2d, eaCal1d, eaCal2d,
                      ebEn, ebNlog, ebNcal, ebLog1d, ebLog2d, ebCal1d, ebCal2d);
        return [
            lppl, lppl, lppl, lppl,
            hyb, hyb, hyb, hyb,
            epp, epp, epp, epp,
            {lppl: lppl, hybppl: hyb, eppl: epp}
        ];
    }
    ''',
    Output("bub-lppl-summary-inline",   "children"),
    Output("dca-lppl-summary-inline",   "children"),
    Output("ret-lppl-summary-inline",   "children"),
    Output("sc-lppl-summary-inline",    "children"),
    Output("bub-hybppl-summary-inline", "children"),
    Output("dca-hybppl-summary-inline", "children"),
    Output("ret-hybppl-summary-inline", "children"),
    Output("sc-hybppl-summary-inline",  "children"),
    Output("bub-eppl-summary-inline",   "children"),
    Output("dca-eppl-summary-inline",   "children"),
    Output("ret-eppl-summary-inline",   "children"),
    Output("sc-eppl-summary-inline",    "children"),
    Output("display-model-summaries",   "data"),
    # LPPL (3)
    Input("lppl-n-freqs", "value"),
    Input("lppl-weighted", "value"),
    Input("lppl-no-13",   "value"),
    # HybPPL (13)
    Input("hybppl-cfg-a-nlog",    "value"),
    Input("hybppl-cfg-a-ncal",    "value"),
    Input("hybppl-cfg-a-log1d",   "value"),
    Input("hybppl-cfg-a-log2d",   "value"),
    Input("hybppl-cfg-a-cal1d",   "value"),
    Input("hybppl-cfg-a-cal2d",   "value"),
    Input("hybppl-cfg-b-enabled", "value"),
    Input("hybppl-cfg-b-nlog",    "value"),
    Input("hybppl-cfg-b-ncal",    "value"),
    Input("hybppl-cfg-b-log1d",   "value"),
    Input("hybppl-cfg-b-log2d",   "value"),
    Input("hybppl-cfg-b-cal1d",   "value"),
    Input("hybppl-cfg-b-cal2d",   "value"),
    # EPPL (13)
    Input("eppl-cfg-a-nlog",    "value"),
    Input("eppl-cfg-a-ncal",    "value"),
    Input("eppl-cfg-a-log1d",   "value"),
    Input("eppl-cfg-a-log2d",   "value"),
    Input("eppl-cfg-a-cal1d",   "value"),
    Input("eppl-cfg-a-cal2d",   "value"),
    Input("eppl-cfg-b-enabled", "value"),
    Input("eppl-cfg-b-nlog",    "value"),
    Input("eppl-cfg-b-ncal",    "value"),
    Input("eppl-cfg-b-log1d",   "value"),
    Input("eppl-cfg-b-log2d",   "value"),
    Input("eppl-cfg-b-cal1d",   "value"),
    Input("eppl-cfg-b-cal2d",   "value"),
)

# ── Heatmap status row: visibility + label (driven by hm-active-model) ──
_app_ctx.app.clientside_callback(
    r"""
    function(active) {
        var CONFIGURABLE = {"lppl": "LPPL", "hybppl": "HybPPL", "eppl": "\u{1FAE0} Entropy PPL"};
        if (!active || !(active in CONFIGURABLE)) {
            return [{display: "none"}, ""];
        }
        return [{display: "inline-flex", alignItems: "center",
                 gap: "4px", marginTop: "6px", fontSize: "11px"}, CONFIGURABLE[active]];
    }
    """,
    Output("hm-active-family-row", "style"),
    Output("hm-active-family-label", "children"),
    Input("hm-active-model", "data"),
)

# ── Heatmap status row: summary text (from display-model-summaries + hm-active-model) ──
_app_ctx.app.clientside_callback(
    """
    function(data, active) {
        if (!data || !active) return "";
        return data[active] || "";
    }
    """,
    Output("hm-active-family-summary-inline", "children"),
    Input("display-model-summaries", "data"),
    Input("hm-active-model", "data"),
)

# ── Heatmap gear → routes clicks to the correct modal ──
_app_ctx.app.clientside_callback(
    """
    function(n, active) {
        if (!n || !active) return [false, false, false];
        return [active === "lppl", active === "hybppl", active === "eppl"];
    }
    """,
    Output("lppl-config-modal",   "is_open", allow_duplicate=True),
    Output("hybppl-config-modal", "is_open", allow_duplicate=True),
    Output("eppl-config-modal",   "is_open", allow_duplicate=True),
    Input("hm-active-family-gear", "n_clicks"),
    State("hm-active-model", "data"),
    prevent_initial_call=True,
)


# Info-triangle click → close whichever modal was open (user navigates
# away to the Model Info tab via the href on the anchor, so keeping the
# modal up would obscure the destination). Each link writes is_open=false
# to its own modal; the href handles the actual navigation.
_app_ctx.app.clientside_callback(
    """function(n) { return n ? false : window.dash_clientside.no_update; }""",
    Output("bm-config-modal", "is_open", allow_duplicate=True),
    Input("bm-info-link", "n_clicks"),
    prevent_initial_call=True,
)
_app_ctx.app.clientside_callback(
    """function(n) { return n ? false : window.dash_clientside.no_update; }""",
    Output("lppl-config-modal", "is_open", allow_duplicate=True),
    Input("lppl-info-link", "n_clicks"),
    prevent_initial_call=True,
)
_app_ctx.app.clientside_callback(
    """function(n) { return n ? false : window.dash_clientside.no_update; }""",
    Output("hybppl-config-modal", "is_open", allow_duplicate=True),
    Input("hybppl-info-link", "n_clicks"),
    prevent_initial_call=True,
)
_app_ctx.app.clientside_callback(
    """function(n) { return n ? false : window.dash_clientside.no_update; }""",
    Output("eppl-config-modal", "is_open", allow_duplicate=True),
    Input("eppl-info-link", "n_clicks"),
    prevent_initial_call=True,
)


# BM modal: open via any tab's gear icon, close via close button.
_app_ctx.app.clientside_callback(
    """
    function(bub_n, dca_n, ret_n, sc_n, close_n, cur_open) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) return window.dash_clientside.no_update;
        var src = ctx.triggered[0].prop_id;
        if (src.indexOf('modal-close-btn') !== -1) return false;
        if (src.indexOf('-bm-gear') !== -1) return true;
        return window.dash_clientside.no_update;
    }
    """,
    Output("bm-config-modal", "is_open", allow_duplicate=True),
    Input("bub-bm-gear", "n_clicks"),
    Input("dca-bm-gear", "n_clicks"),
    Input("ret-bm-gear", "n_clicks"),
    Input("sc-bm-gear",  "n_clicks"),
    Input("bm-modal-close-btn", "n_clicks"),
    State("bm-config-modal", "is_open"),
    prevent_initial_call=True,
)

# LPPL modal: open via gear icon, close via close button.
_app_ctx.app.clientside_callback(
    """
    function(bub_n, dca_n, ret_n, sc_n, close_n, cur_open) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) return window.dash_clientside.no_update;
        var src = ctx.triggered[0].prop_id;
        if (src.indexOf('modal-close-btn') !== -1) return false;
        if (src.indexOf('-gear') !== -1) return true;
        return window.dash_clientside.no_update;
    }
    """,
    Output("lppl-config-modal", "is_open", allow_duplicate=True),
    Input("bub-lppl-gear", "n_clicks"),
    Input("dca-lppl-gear", "n_clicks"),
    Input("ret-lppl-gear", "n_clicks"),
    Input("sc-lppl-gear",  "n_clicks"),
    Input("lppl-modal-close-btn", "n_clicks"),
    State("lppl-config-modal", "is_open"),
    prevent_initial_call=True,
)


# ── HybPPL modal: open via gear icon, close via close button ─────────────
_app_ctx.app.clientside_callback(
    """
    function(bub_n, dca_n, ret_n, sc_n, close_n, cur_open) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) return window.dash_clientside.no_update;
        var src = ctx.triggered[0].prop_id;
        if (src.indexOf('modal-close-btn') !== -1) return false;
        if (src.indexOf('-gear') !== -1) return true;
        return window.dash_clientside.no_update;
    }
    """,
    Output("hybppl-config-modal", "is_open", allow_duplicate=True),
    Input("bub-hybppl-gear", "n_clicks"),
    Input("dca-hybppl-gear", "n_clicks"),
    Input("ret-hybppl-gear", "n_clicks"),
    Input("sc-hybppl-gear",  "n_clicks"),
    Input("hybppl-modal-close-btn", "n_clicks"),
    State("hybppl-config-modal", "is_open"),
    prevent_initial_call=True,
)

# Damping visibility toggles (Model A + Model B)
for _hs in ("a", "b"):
    _app_ctx.app.clientside_callback(
        """
        function(nlog) {
            return (nlog >= 1) ? {} : {display: 'none'};
        }
        """,
        Output(f"hybppl-cfg-{_hs}-log1d-wrap", "style"),
        Input(f"hybppl-cfg-{_hs}-nlog", "value"),
    )
    _app_ctx.app.clientside_callback(
        """
        function(nlog) {
            return (nlog >= 2) ? {} : {display: 'none'};
        }
        """,
        Output(f"hybppl-cfg-{_hs}-log2d-wrap", "style"),
        Input(f"hybppl-cfg-{_hs}-nlog", "value"),
    )
    _app_ctx.app.clientside_callback(
        """
        function(ncal) {
            return (ncal >= 1) ? {} : {display: 'none'};
        }
        """,
        Output(f"hybppl-cfg-{_hs}-cal1d-wrap", "style"),
        Input(f"hybppl-cfg-{_hs}-ncal", "value"),
    )
    _app_ctx.app.clientside_callback(
        """
        function(ncal) {
            return (ncal >= 2) ? {} : {display: 'none'};
        }
        """,
        Output(f"hybppl-cfg-{_hs}-cal2d-wrap", "style"),
        Input(f"hybppl-cfg-{_hs}-ncal", "value"),
    )

# Status text for each model slot (A and B)
for _hs in ("a", "b"):
    _app_ctx.app.clientside_callback(
        """
        function(nlog, ncal, log1d, log2d, cal1d, cal2d) {
            function spec(n, d1, d2) {
                if (n === 0) return "0";
                if (n === 1) return "1" + (d1 || "d");
                return "2" + (d1 || "d") + (d2 || "d");
            }
            var key = "cfg_" + spec(nlog, log1d, log2d) + "_" + spec(ncal, cal1d, cal2d);
            var info_links = {
                "cfg_1d_0":    "/8.4",
                "cfg_0_1u":    "/8.7",
                "cfg_1d_1u":   "/8.8",
                "cfg_1d_1d":   "/8.9",
                "cfg_2dd_1u":  "/8.10",
                "cfg_1d_2uu":  "/8.11",
                "cfg_2dd_2uu": "/8.12",
                "cfg_2dd_2dd": "/8.13",
            };
            var href = info_links[key] || "";
            var link_style = href ? {fontSize:"11px", marginLeft:"6px", color:"{lnk}"} : {display:"none"};
            return [key, href, link_style];
        }
        """.replace("{lnk}", LINK),
        Output(f"hybppl-cfg-{_hs}-status", "children"),
        Output(f"hybppl-cfg-{_hs}-info-link", "href"),
        Output(f"hybppl-cfg-{_hs}-info-link", "style"),
        Input(f"hybppl-cfg-{_hs}-nlog", "value"),
        Input(f"hybppl-cfg-{_hs}-ncal", "value"),
        Input(f"hybppl-cfg-{_hs}-log1d", "value"),
        Input(f"hybppl-cfg-{_hs}-log2d", "value"),
        Input(f"hybppl-cfg-{_hs}-cal1d", "value"),
        Input(f"hybppl-cfg-{_hs}-cal2d", "value"),
    )

# ══════════════════════════════════════════════════════════════════════════════
# EPPL config modal
# ══════════════════════════════════════════════════════════════════════════════

# EPPL modal: open via gear icon, close via close button.
_app_ctx.app.clientside_callback(
    """
    function(bub_n, dca_n, ret_n, sc_n, close_n, cur_open) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) return window.dash_clientside.no_update;
        var src = ctx.triggered[0].prop_id;
        if (src.indexOf('modal-close-btn') !== -1) return false;
        if (src.indexOf('-gear') !== -1) return true;
        return window.dash_clientside.no_update;
    }
    """,
    Output("eppl-config-modal", "is_open", allow_duplicate=True),
    Input("bub-eppl-gear", "n_clicks"),
    Input("dca-eppl-gear", "n_clicks"),
    Input("ret-eppl-gear", "n_clicks"),
    Input("sc-eppl-gear",  "n_clicks"),
    Input("eppl-modal-close-btn", "n_clicks"),
    State("eppl-config-modal", "is_open"),
    prevent_initial_call=True,
)

# Damping visibility toggles (Model A + Model B)
for _es in ("a", "b"):
    _app_ctx.app.clientside_callback(
        """
        function(nlog) {
            return (nlog >= 1) ? {} : {display: 'none'};
        }
        """,
        Output(f"eppl-cfg-{_es}-log1d-wrap", "style"),
        Input(f"eppl-cfg-{_es}-nlog", "value"),
    )
    _app_ctx.app.clientside_callback(
        """
        function(nlog) {
            return (nlog >= 2) ? {} : {display: 'none'};
        }
        """,
        Output(f"eppl-cfg-{_es}-log2d-wrap", "style"),
        Input(f"eppl-cfg-{_es}-nlog", "value"),
    )
    _app_ctx.app.clientside_callback(
        """
        function(ncal) {
            return (ncal >= 1) ? {} : {display: 'none'};
        }
        """,
        Output(f"eppl-cfg-{_es}-cal1d-wrap", "style"),
        Input(f"eppl-cfg-{_es}-ncal", "value"),
    )
    _app_ctx.app.clientside_callback(
        """
        function(ncal) {
            return (ncal >= 2) ? {} : {display: 'none'};
        }
        """,
        Output(f"eppl-cfg-{_es}-cal2d-wrap", "style"),
        Input(f"eppl-cfg-{_es}-ncal", "value"),
    )

# Status text for each EPPL model slot (A and B)
for _es in ("a", "b"):
    _app_ctx.app.clientside_callback(
        """
        function(nlog, ncal, log1d, log2d, cal1d, cal2d) {
            function spec(n, d1, d2) {
                if (n === 0) return "0";
                if (n === 1) return "1" + (d1 || "d");
                return "2" + (d1 || "d") + (d2 || "d");
            }
            var key = "ecfg_" + spec(nlog, log1d, log2d) + "_" + spec(ncal, cal1d, cal2d);
            var href = "";
            var link_style = {display:"none"};
            return [key, href, link_style];
        }
        """,
        Output(f"eppl-cfg-{_es}-status", "children"),
        Output(f"eppl-cfg-{_es}-info-link", "href"),
        Output(f"eppl-cfg-{_es}-info-link", "style"),
        Input(f"eppl-cfg-{_es}-nlog", "value"),
        Input(f"eppl-cfg-{_es}-ncal", "value"),
        Input(f"eppl-cfg-{_es}-log1d", "value"),
        Input(f"eppl-cfg-{_es}-log2d", "value"),
        Input(f"eppl-cfg-{_es}-cal1d", "value"),
        Input(f"eppl-cfg-{_es}-cal2d", "value"),
    )

from btc_core import yr_to_t, today_t, _find_lot_percentile
from tab_defaults import BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE
from layout.common import _bands_to_qs
from callbacks.coerce import _ci, _cf
from callbacks.mc_helpers import (_mc_setup, _mc_finalize, _mc_status,
                                  _strip_free_paths)
from mc_cache import (MC_BINS, MC_SIMS, MC_FREQ,
                      MC_DEFAULT_YEARS, MC_DEFAULT_START_YR, MC_DEFAULT_ENTRY_Q)
from utils import (_get_bubble_fig, _get_dca_fig, _get_retire_fig,
                   _get_supercharge_fig, _get_heatmap_fig, _get_mc_heatmap_fig,
                   _nearest_quantile)


def _resolve_hm_lppl_master(hm_model, lppl_n_freqs, lppl_weighted, lppl_no_13):
    """Translate 'lppl' master to specific flavor key for heatmap (single-select).

    Returns the flavor key the heatmap figure builder should use. For non-lppl
    models or when lppl flavor cannot be resolved, returns input unchanged.
    """
    if hm_model != "lppl":
        return hm_model
    _weighted = "weighted" in (lppl_weighted or [])
    _no_13 = "no13" in (lppl_no_13 or [])
    _n_list = (lppl_n_freqs or [3])
    _n = _n_list[0] if _n_list else 3
    if _n == 1:
        return "lppl_w" if _weighted else "lppl"
    if _n == 2:
        return "lp2_w" if _weighted else "lp2"
    if _n == 3 and not _no_13:
        return "lp3_w" if _weighted else "lp3"
    if _n == 4:
        if _no_13:
            return "lp4_w_n13" if _weighted else "lp4_n13"
        return "lp4_w" if _weighted else "lp4"
    return "lppl"  # fallback


def _resolve_lppl_master(model_show, lppl_n_freqs, lppl_weighted, lppl_no_13):
    """Translate the 'lppl' master in model_show into specific flavor key(s).

    Strips 'lppl' from the list and appends one flavor key per checked
    n_freqs entry, applying weighted and no_13 modifiers. When no master
    is present, returns the list unchanged. When master is present but
    no flavor is selected, the master is stripped with no replacement.
    """
    model_show = list(model_show or [])
    if "lppl" not in model_show:
        return model_show
    model_show = [v for v in model_show if v != "lppl"]
    _weighted = "weighted" in (lppl_weighted or [])
    _no_13 = "no13" in (lppl_no_13 or [])
    for n in (lppl_n_freqs or []):
        if n == 1:
            model_show.append("lppl_w" if _weighted else "lppl")
        elif n == 2:
            model_show.append("lp2_w" if _weighted else "lp2")
        elif n == 3 and not _no_13:
            model_show.append("lp3_w" if _weighted else "lp3")
        elif n == 4:
            if _no_13:
                model_show.append("lp4_w_n13" if _weighted else "lp4_n13")
            else:
                model_show.append("lp4_w" if _weighted else "lp4")
    return model_show


def _build_hybppl_config_key(nlog, ncal, log1d, log2d, cal1d, cal2d):
    """Build a cfg_* key from the modal's radio-item values."""
    def _spec(n, d1, d2):
        if n == 0:
            return "0"
        if n == 1:
            return f"1{d1 or 'd'}"
        return f"2{d1 or 'd'}{d2 or 'd'}"
    return f"cfg_{_spec(nlog, log1d, log2d)}_{_spec(ncal, cal1d, cal2d)}"


def _resolve_hybppl_master(model_show,
                            cfg_a_nlog, cfg_a_ncal,
                            cfg_a_log1d, cfg_a_log2d,
                            cfg_a_cal1d, cfg_a_cal2d,
                            cfg_b_enabled,
                            cfg_b_nlog, cfg_b_ncal,
                            cfg_b_log1d, cfg_b_log2d,
                            cfg_b_cal1d, cfg_b_cal2d):
    """Translate the 'hybppl' master in model_show into concrete cfg_* key(s).

    Strips 'hybppl' (and named variants) from the list and appends 1-2
    resolved cfg_* keys based on the modal config. When no master is
    present, returns the list unchanged.
    """
    model_show = list(model_show or [])
    _HYBPPL_NAMES = {"hybppl", "hybppl_dd", "hyb2l", "hyb2c", "hyb2b", "hyb4d"}
    if "hybppl" not in model_show:
        return model_show
    model_show = [v for v in model_show if v not in _HYBPPL_NAMES]
    # Model A
    key_a = _build_hybppl_config_key(
        cfg_a_nlog or 1, cfg_a_ncal or 1,
        cfg_a_log1d, cfg_a_log2d, cfg_a_cal1d, cfg_a_cal2d)
    if key_a in _app_ctx.PRICE_MODELS:
        model_show.append(key_a)
    # Model B (if enabled)
    if cfg_b_enabled and "yes" in (cfg_b_enabled or []):
        key_b = _build_hybppl_config_key(
            cfg_b_nlog or 0, cfg_b_ncal or 0,
            cfg_b_log1d, cfg_b_log2d, cfg_b_cal1d, cfg_b_cal2d)
        if key_b in _app_ctx.PRICE_MODELS and key_b != key_a:
            model_show.append(key_b)
    return model_show


def _resolve_hm_hybppl_master(hm_model,
                               cfg_a_nlog, cfg_a_ncal,
                               cfg_a_log1d, cfg_a_log2d,
                               cfg_a_cal1d, cfg_a_cal2d):
    """Translate 'hybppl' master to specific cfg_* key for heatmap (single-select).

    Returns the resolved model key. For non-hybppl models, returns unchanged.
    """
    if hm_model != "hybppl":
        return hm_model
    key = _build_hybppl_config_key(
        cfg_a_nlog or 1, cfg_a_ncal or 1,
        cfg_a_log1d, cfg_a_log2d, cfg_a_cal1d, cfg_a_cal2d)
    return key if key in _app_ctx.PRICE_MODELS else hm_model


def _build_eppl_config_key(nlog, ncal, log1d, log2d, cal1d, cal2d):
    """Build an ecfg_* key from the modal's radio-item values."""
    def _spec(n, d1, d2):
        if n == 0:
            return "0"
        if n == 1:
            return f"1{d1 or 'd'}"
        return f"2{d1 or 'd'}{d2 or 'd'}"
    return f"ecfg_{_spec(nlog, log1d, log2d)}_{_spec(ncal, cal1d, cal2d)}"


def _resolve_eppl_master(model_show,
                          cfg_a_nlog, cfg_a_ncal,
                          cfg_a_log1d, cfg_a_log2d,
                          cfg_a_cal1d, cfg_a_cal2d,
                          cfg_b_enabled,
                          cfg_b_nlog, cfg_b_ncal,
                          cfg_b_log1d, cfg_b_log2d,
                          cfg_b_cal1d, cfg_b_cal2d):
    """Translate the 'eppl' master in model_show into concrete ecfg_* key(s).

    Strips 'eppl' from the list and appends 1-2 resolved ecfg_* keys
    based on the modal config. When no master is present, returns the
    list unchanged.
    """
    model_show = list(model_show or [])
    if "eppl" not in model_show:
        return model_show
    model_show = [v for v in model_show if v != "eppl"]
    # Model A
    key_a = _build_eppl_config_key(
        cfg_a_nlog or 1, cfg_a_ncal or 1,
        cfg_a_log1d, cfg_a_log2d, cfg_a_cal1d, cfg_a_cal2d)
    if key_a in _app_ctx.PRICE_MODELS:
        model_show.append(key_a)
    # Model B (if enabled)
    if cfg_b_enabled and "yes" in (cfg_b_enabled or []):
        key_b = _build_eppl_config_key(
            cfg_b_nlog or 0, cfg_b_ncal or 0,
            cfg_b_log1d, cfg_b_log2d, cfg_b_cal1d, cfg_b_cal2d)
        if key_b in _app_ctx.PRICE_MODELS and key_b != key_a:
            model_show.append(key_b)
    return model_show


def _resolve_hm_eppl_master(hm_model,
                              cfg_a_nlog, cfg_a_ncal,
                              cfg_a_log1d, cfg_a_log2d,
                              cfg_a_cal1d, cfg_a_cal2d):
    """Translate 'eppl' master to specific ecfg_* key for heatmap (single-select).

    Returns the resolved model key. For non-eppl models, returns unchanged.
    """
    if hm_model != "eppl":
        return hm_model
    key = _build_eppl_config_key(
        cfg_a_nlog or 1, cfg_a_ncal or 1,
        cfg_a_log1d, cfg_a_log2d, cfg_a_cal1d, cfg_a_cal2d)
    return key if key in _app_ctx.PRICE_MODELS else hm_model


def _decomp_warning_banner(n_checked):
    """Inline banner shown when LPPL decomposition needs exactly 1 n_freqs."""
    return html.Div(
        html.Small(
            f"Pick exactly one LPPL variant in the LPPL config panel "
            f"to decompose (currently {n_checked} checked).",
            style={"color": DECOMP_ERROR_RED},
        ),
        style={"padding": "6px 8px", "backgroundColor": ERROR_BG,
                "border": f"1px solid {ERROR_BORDER}", "borderRadius": "4px",
                "fontSize": UI_FONT_MD, "marginTop": "6px"},
    )


def _r2_of_log_pred(log_pred, log_actual):
    """Compute R² of a log10 prediction array vs actual log10 prices."""
    residuals = log_actual - log_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((log_actual - np.mean(log_actual)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _component_label(model, name):
    """Return the component name as the checkbox label (clean)."""
    return f" {name}"


def update_decomp_options(family, n_freqs, weighted, no_13, hybppl_cfg=None, eppl_cfg=None):
    """Populate Component Decomposition checklist options + warning + visibility.

    Returns (options, warning_children, body_style). NEVER modifies
    bub-decomp-components.value (see prune_decomp_value_on_model_change).
    """
    if not family:
        return [], [], {"display": "none"}
    if family == "lppl" and len(n_freqs or []) != 1:
        return [], _decomp_warning_banner(len(n_freqs or [])), {"display": "block"}
    key = _resolve_decomp_model_key(family, n_freqs, weighted, no_13,
                                     hybppl_cfg=hybppl_cfg, eppl_cfg=eppl_cfg)
    if key is None:
        return [], _decomp_warning_banner(len(n_freqs or [])), {"display": "block"}
    model = _app_ctx.PRICE_MODELS.get(key)
    if model is None:
        return [], [], {"display": "none"}
    opts = [{"label": _component_label(model, name), "value": name}
            for name in model.component_names]
    return opts, [], {"display": "block"}


@callback(
    Output("bub-decomp-components", "options"),
    Output("bub-decomp-warning",    "children"),
    Output("bub-decomp-body",       "style"),
    Input("bub-decomp-model",  "value"),
    Input("lppl-n-freqs",      "value"),
    Input("lppl-weighted",     "value"),
    Input("lppl-no-13",        "value"),
    State("hybppl-cfg-a-nlog", "value"),
    State("hybppl-cfg-a-ncal", "value"),
    State("hybppl-cfg-a-log1d", "value"),
    State("hybppl-cfg-a-log2d", "value"),
    State("hybppl-cfg-a-cal1d", "value"),
    State("hybppl-cfg-a-cal2d", "value"),
    State("hybppl-cfg-b-nlog", "value"),
    State("hybppl-cfg-b-ncal", "value"),
    State("hybppl-cfg-b-log1d", "value"),
    State("hybppl-cfg-b-log2d", "value"),
    State("hybppl-cfg-b-cal1d", "value"),
    State("hybppl-cfg-b-cal2d", "value"),
    State("eppl-cfg-a-nlog", "value"),
    State("eppl-cfg-a-ncal", "value"),
    State("eppl-cfg-a-log1d", "value"),
    State("eppl-cfg-a-log2d", "value"),
    State("eppl-cfg-a-cal1d", "value"),
    State("eppl-cfg-a-cal2d", "value"),
    State("eppl-cfg-b-nlog", "value"),
    State("eppl-cfg-b-ncal", "value"),
    State("eppl-cfg-b-log1d", "value"),
    State("eppl-cfg-b-log2d", "value"),
    State("eppl-cfg-b-cal1d", "value"),
    State("eppl-cfg-b-cal2d", "value"),
    prevent_initial_call=False,
)
def _update_decomp_options_cb(family, n_freqs, weighted, no_13,
                               a_nlog, a_ncal, a_log1d, a_log2d, a_cal1d, a_cal2d,
                               b_nlog, b_ncal, b_log1d, b_log2d, b_cal1d, b_cal2d,
                               ea_nlog, ea_ncal, ea_log1d, ea_log2d, ea_cal1d, ea_cal2d,
                               eb_nlog, eb_ncal, eb_log1d, eb_log2d, eb_cal1d, eb_cal2d):
    hybppl_cfg = {
        "a_nlog": a_nlog, "a_ncal": a_ncal,
        "a_log1d": a_log1d, "a_log2d": a_log2d,
        "a_cal1d": a_cal1d, "a_cal2d": a_cal2d,
        "b_nlog": b_nlog, "b_ncal": b_ncal,
        "b_log1d": b_log1d, "b_log2d": b_log2d,
        "b_cal1d": b_cal1d, "b_cal2d": b_cal2d,
    }
    eppl_cfg = {
        "a_nlog": ea_nlog, "a_ncal": ea_ncal,
        "a_log1d": ea_log1d, "a_log2d": ea_log2d,
        "a_cal1d": ea_cal1d, "a_cal2d": ea_cal2d,
        "b_nlog": eb_nlog, "b_ncal": eb_ncal,
        "b_log1d": eb_log1d, "b_log2d": eb_log2d,
        "b_cal1d": eb_cal1d, "b_cal2d": eb_cal2d,
    }
    return update_decomp_options(family, n_freqs, weighted, no_13, hybppl_cfg=hybppl_cfg, eppl_cfg=eppl_cfg)


@callback(
    Output("bub-decomp-active-formula", "children"),
    Input("bub-decomp-model",          "value"),
    Input("bub-decomp-components",     "value"),
    Input("bub-decomp-show-formulas",  "value"),
    Input("lppl-n-freqs",      "value"),
    Input("lppl-weighted",     "value"),
    Input("lppl-no-13",        "value"),
    State("hybppl-cfg-a-nlog", "value"), State("hybppl-cfg-a-ncal", "value"),
    State("hybppl-cfg-a-log1d", "value"), State("hybppl-cfg-a-log2d", "value"),
    State("hybppl-cfg-a-cal1d", "value"), State("hybppl-cfg-a-cal2d", "value"),
    State("hybppl-cfg-b-nlog", "value"), State("hybppl-cfg-b-ncal", "value"),
    State("hybppl-cfg-b-log1d", "value"), State("hybppl-cfg-b-log2d", "value"),
    State("hybppl-cfg-b-cal1d", "value"), State("hybppl-cfg-b-cal2d", "value"),
    State("eppl-cfg-a-nlog", "value"), State("eppl-cfg-a-ncal", "value"),
    State("eppl-cfg-a-log1d", "value"), State("eppl-cfg-a-log2d", "value"),
    State("eppl-cfg-a-cal1d", "value"), State("eppl-cfg-a-cal2d", "value"),
    State("eppl-cfg-b-nlog", "value"), State("eppl-cfg-b-ncal", "value"),
    State("eppl-cfg-b-log1d", "value"), State("eppl-cfg-b-log2d", "value"),
    State("eppl-cfg-b-cal1d", "value"), State("eppl-cfg-b-cal2d", "value"),
    prevent_initial_call=False,
)
def _update_active_formula_cb(family, selected, show_toggles, n_freqs, weighted, no_13,
                               a_nlog, a_ncal, a_log1d, a_log2d, a_cal1d, a_cal2d,
                               b_nlog, b_ncal, b_log1d, b_log2d, b_cal1d, b_cal2d,
                               ea_nlog, ea_ncal, ea_log1d, ea_log2d, ea_cal1d, ea_cal2d,
                               eb_nlog, eb_ncal, eb_log1d, eb_log2d, eb_cal1d, eb_cal2d):
    """Display the formula for the currently-checked subset -- gated on toggle."""
    from dash import html
    if "selected" not in (show_toggles or []):
        return []
    hybppl_cfg = {
        "a_nlog": a_nlog, "a_ncal": a_ncal,
        "a_log1d": a_log1d, "a_log2d": a_log2d,
        "a_cal1d": a_cal1d, "a_cal2d": a_cal2d,
        "b_nlog": b_nlog, "b_ncal": b_ncal,
        "b_log1d": b_log1d, "b_log2d": b_log2d,
        "b_cal1d": b_cal1d, "b_cal2d": b_cal2d,
    }
    eppl_cfg = {
        "a_nlog": ea_nlog, "a_ncal": ea_ncal,
        "a_log1d": ea_log1d, "a_log2d": ea_log2d,
        "a_cal1d": ea_cal1d, "a_cal2d": ea_cal2d,
        "b_nlog": eb_nlog, "b_ncal": eb_ncal,
        "b_log1d": eb_log1d, "b_log2d": eb_log2d,
        "b_cal1d": eb_cal1d, "b_cal2d": eb_cal2d,
    }
    key = _resolve_decomp_model_key(family, n_freqs, weighted, no_13,
                                     hybppl_cfg=hybppl_cfg, eppl_cfg=eppl_cfg)
    if key is None:
        return []
    model = _app_ctx.PRICE_MODELS.get(key)
    if model is None:
        return []
    details = getattr(model, "component_details", {})
    if not details:
        return []
    selected = list(selected or [])
    canonical = [n for n in model.component_names if n in selected]
    header = html.Div(html.Strong("Selected:"),
                      style={"marginBottom": "3px", "color": LOT_MARKER_OUTLINE})
    if not canonical:
        return [header, html.Small("(no components selected)",
                                    style={"color": FALLBACK_MODEL_GRAY})]
    log_parts = []
    product_parts = []
    for name in canonical:
        d = details.get(name)
        if not d:
            continue
        formula_str = d[0]
        log_parts.append(formula_str)
        product_parts.append(f"10^({formula_str})")
    if not log_parts:
        return [header]
    log_str = " + ".join(log_parts)
    product_str = " \u00b7 ".join(product_parts)
    return [
        header,
        html.Div([html.Strong("log\u2081\u2080(price) = "), log_str],
                 style={"marginBottom": "3px"}),
        html.Div([html.Strong("price = "), product_str]),
    ]


@callback(
    Output("bub-decomp-formula", "children"),
    Input("bub-decomp-model",          "value"),
    Input("bub-decomp-show-formulas",  "value"),
    Input("lppl-n-freqs",      "value"),
    Input("lppl-weighted",     "value"),
    Input("lppl-no-13",        "value"),
    State("hybppl-cfg-a-nlog", "value"), State("hybppl-cfg-a-ncal", "value"),
    State("hybppl-cfg-a-log1d", "value"), State("hybppl-cfg-a-log2d", "value"),
    State("hybppl-cfg-a-cal1d", "value"), State("hybppl-cfg-a-cal2d", "value"),
    State("hybppl-cfg-b-nlog", "value"), State("hybppl-cfg-b-ncal", "value"),
    State("hybppl-cfg-b-log1d", "value"), State("hybppl-cfg-b-log2d", "value"),
    State("hybppl-cfg-b-cal1d", "value"), State("hybppl-cfg-b-cal2d", "value"),
    prevent_initial_call=False,
)
def _update_decomp_formula_cb(family, show_toggles, n_freqs, weighted, no_13,
                               a_nlog, a_ncal, a_log1d, a_log2d, a_cal1d, a_cal2d,
                               b_nlog, b_ncal, b_log1d, b_log2d, b_cal1d, b_cal2d):
    """Show the model's full formula — gated on toggle."""
    from dash import dcc, html
    if "full" not in (show_toggles or []):
        return []
    hybppl_cfg = {
        "a_nlog": a_nlog, "a_ncal": a_ncal,
        "a_log1d": a_log1d, "a_log2d": a_log2d,
        "a_cal1d": a_cal1d, "a_cal2d": a_cal2d,
        "b_nlog": b_nlog, "b_ncal": b_ncal,
        "b_log1d": b_log1d, "b_log2d": b_log2d,
        "b_cal1d": b_cal1d, "b_cal2d": b_cal2d,
    }
    key = _resolve_decomp_model_key(family, n_freqs, weighted, no_13,
                                     hybppl_cfg=hybppl_cfg)
    if key is None:
        return []
    model = _app_ctx.PRICE_MODELS.get(key)
    if model is None:
        return []
    latex = getattr(model, "formula_log10_latex", None)
    product = getattr(model, "formula_product_latex", None)
    if not latex:
        return []
    price_line = (rf"$$\text{{price}} = {product}$$" if product
                  else rf"$$\text{{price}} = 10^{{\,\log_{{10}}(\text{{price}})}}$$")
    return [
        html.Div(html.Strong("Full model:"),
                 style={"marginBottom": "3px", "color": LOT_MARKER_OUTLINE,
                        "fontSize": UI_FONT_MD}),
        dcc.Markdown(
            rf"""
$$\log_{{10}}(\text{{price}}) = {latex}$$

{price_line}
""",
            mathjax=True, className="small",
            style={"fontSize": UI_FONT_SM},
        ),
    ]


def _prune_decomp_value(family, options, current):
    """Keep only currently-valid values from the checklist."""
    if not family:
        return []
    valid = {o["value"] for o in (options or [])}
    return [v for v in (current or []) if v in valid]


@callback(
    Output("bub-decomp-components", "value", allow_duplicate=True),
    Input("bub-decomp-model",       "value"),
    State("bub-decomp-components", "options"),
    State("bub-decomp-components", "value"),
    prevent_initial_call=True,
)
def _prune_decomp_value_cb(family, opts, current):
    if ctx.triggered_id != "bub-decomp-model":
        raise dash.exceptions.PreventUpdate
    return _prune_decomp_value(family, opts, current)


def _resolve_decomp_model_key(family, lppl_n_freqs, lppl_weighted, lppl_no_13,
                               hybppl_cfg=None, eppl_cfg=None):
    """Translate (family, LPPL config) into a concrete model short_name.

    Returns None if family is empty OR if family is 'lppl' but exactly one
    n_freqs entry is not selected. Otherwise returns the model's short_name
    (e.g., 'bub', 'lp3_w', 'hybppl_dd').
    """
    if not family:
        return None
    # "hybppl" master (from Display Models consolidation) → resolve via
    # slot A of the HybPPL modal, matching the Display Models master → variant
    # routing behavior. Same for "eppl".
    if family == "hybppl" and hybppl_cfg:
        family = "hybppl_cfg_a"
    elif family == "eppl" and eppl_cfg:
        family = "eppl_cfg_a"
    # HybPPL config A/B -> resolve to cfg_* key
    if family in ("hybppl_cfg_a", "hybppl_cfg_b") and hybppl_cfg:
        slot = "a" if family.endswith("_a") else "b"
        nlog = hybppl_cfg.get(f"{slot}_nlog", 1)
        ncal = hybppl_cfg.get(f"{slot}_ncal", 1)
        log1d = hybppl_cfg.get(f"{slot}_log1d", "d")
        log2d = hybppl_cfg.get(f"{slot}_log2d", "d")
        cal1d = hybppl_cfg.get(f"{slot}_cal1d", "u")
        cal2d = hybppl_cfg.get(f"{slot}_cal2d", "u")
        key = _build_hybppl_config_key(nlog, ncal, log1d, log2d, cal1d, cal2d)
        return key if key in _app_ctx.PRICE_MODELS else None
    # EPPL config A/B -> resolve to ecfg_* key
    if family in ("eppl_cfg_a", "eppl_cfg_b") and eppl_cfg:
        slot = "a" if family.endswith("_a") else "b"
        nlog = eppl_cfg.get(f"{slot}_nlog", 1)
        ncal = eppl_cfg.get(f"{slot}_ncal", 1)
        log1d = eppl_cfg.get(f"{slot}_log1d", "d")
        log2d = eppl_cfg.get(f"{slot}_log2d", "d")
        cal1d = eppl_cfg.get(f"{slot}_cal1d", "u")
        cal2d = eppl_cfg.get(f"{slot}_cal2d", "u")
        key = _build_eppl_config_key(nlog, ncal, log1d, log2d, cal1d, cal2d)
        return key if key in _app_ctx.PRICE_MODELS else None
    if family != "lppl":
        return family
    if not lppl_n_freqs or len(lppl_n_freqs) != 1:
        return None
    n = lppl_n_freqs[0]
    weighted = "weighted" in (lppl_weighted or [])
    no13 = "no13" in (lppl_no_13 or [])
    if n == 1:
        return "lppl_w" if weighted else "lppl"
    if n == 2:
        return "lp2_w" if weighted else "lp2"
    if n == 3:
        return "lp3_w" if weighted else "lp3"
    if n == 4:
        if no13:
            return "lp4_w_n13" if weighted else "lp4_n13"
        return "lp4_w" if weighted else "lp4"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks — chart updates
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("bubble-graph", "figure"),
    Input("bubble-first-render", "data"),
    Input("bub-qs",            "value"),
    Input("bub-qs-adv",        "value"),
    Input("bub-toggles",       "value"),
    Input("bub-bubble-toggles","value"),
    Input("bub-xscale",        "value"),
    Input("bub-yscale",        "value"),
    Input("bub-xrange",        "value"),
    Input("bub-yrange",        "value"),
    Input("bub-n-future",      "value"),
    Input("bub-ptsize",        "value"),
    Input("bub-ptalpha",       "value"),
    Input("bub-stack",         "value"),
    Input("bub-show-stack",    "value"),
    Input("bub-use-lots",      "value"),
    Input("bub-legend-pos",    "value"),
    Input("bub-model-show",    "value"),
    Input("lppl-n-freqs",      "value"),
    Input("lppl-weighted",     "value"),
    Input("lppl-no-13",        "value"),
    Input("hybppl-cfg-a-nlog", "value"),
    Input("hybppl-cfg-a-ncal", "value"),
    Input("hybppl-cfg-a-log1d","value"),
    Input("hybppl-cfg-a-log2d","value"),
    Input("hybppl-cfg-a-cal1d","value"),
    Input("hybppl-cfg-a-cal2d","value"),
    Input("hybppl-cfg-b-enabled","value"),
    Input("hybppl-cfg-b-nlog", "value"),
    Input("hybppl-cfg-b-ncal", "value"),
    Input("hybppl-cfg-b-log1d","value"),
    Input("hybppl-cfg-b-log2d","value"),
    Input("hybppl-cfg-b-cal1d","value"),
    Input("hybppl-cfg-b-cal2d","value"),
    Input("eppl-cfg-a-nlog", "value"),
    Input("eppl-cfg-a-ncal", "value"),
    Input("eppl-cfg-a-log1d","value"),
    Input("eppl-cfg-a-log2d","value"),
    Input("eppl-cfg-a-cal1d","value"),
    Input("eppl-cfg-a-cal2d","value"),
    Input("eppl-cfg-b-enabled","value"),
    Input("eppl-cfg-b-nlog", "value"),
    Input("eppl-cfg-b-ncal", "value"),
    Input("eppl-cfg-b-log1d","value"),
    Input("eppl-cfg-b-log2d","value"),
    Input("eppl-cfg-b-cal1d","value"),
    Input("eppl-cfg-b-cal2d","value"),
    Input("bub-decomp-model",       "value"),
    Input("bub-decomp-components",  "value"),
    Input("bub-decomp-mode",        "value"),
    Input("effective-lots",    "data"),
    Input("palette-store",     "data"),
    Input("user-model-store",  "data"),
    # Custom Time Axis router: tick bump re-fires this callback; cta_active
    # State guards against clobbering the custom figure when active.
    Input("bub-redraw-tick",   "data"),
    State("cta-active",        "value"),
    State("bub-qs-mode",       "value"),
    State("scan-active-rows",  "data"),
    State("scan-q",            "value"),
    State("bub-sigma-mode",    "value"),
    prevent_initial_call=True,
)
def update_bubble(_first_render, sel_qs, adv_qs, toggles, bubble_toggles,
                  xscale, yscale, xrange, yrange,
                  n_future, ptsize, ptalpha, stack, show_stack, use_lots, legend_pos, model_show,
                  lppl_n_freqs, lppl_weighted, lppl_no_13,
                  hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d,
                  hyb_a_cal1d, hyb_a_cal2d,
                  hyb_b_enabled, hyb_b_nlog, hyb_b_ncal, hyb_b_log1d, hyb_b_log2d,
                  hyb_b_cal1d, hyb_b_cal2d,
                  ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d,
                  ep_a_cal1d, ep_a_cal2d,
                  ep_b_enabled, ep_b_nlog, ep_b_ncal, ep_b_log1d, ep_b_log2d,
                  ep_b_cal1d, ep_b_cal2d,
                  decomp_model, decomp_components, decomp_mode,
                  lots_data,
                  palette_key, user_model_store=None,
                  _redraw_tick=None, cta_active=None,
                  qs_mode=None, scan_active=None, scan_q_val=None,
                  sigma_mode=None):
    """Bubble + QR overlay chart callback -- coerce inputs, build figure."""
    # Custom Time Axis router: if cta-active is on, the Custom Time Axis
    # callback owns bubble-graph.figure. Refuse to overwrite.
    if cta_active and "yes" in cta_active:
        from dash.exceptions import PreventUpdate
        raise PreventUpdate
    toggles        = toggles or []
    bubble_toggles = bubble_toggles or []
    yrange         = yrange or [0, 7]
    xrange         = xrange or [2012, 2030]

    # The "lppl" entry in bub-model-show is a MASTER gate -- translate
    # to specific flavor key(s) via global LPPL config.
    model_show = _resolve_lppl_master(
        model_show, lppl_n_freqs, lppl_weighted, lppl_no_13)
    # The "hybppl" master gate -- translate to concrete cfg_* key(s).
    model_show = _resolve_hybppl_master(
        model_show,
        hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d, hyb_a_cal1d, hyb_a_cal2d,
        hyb_b_enabled, hyb_b_nlog, hyb_b_ncal, hyb_b_log1d, hyb_b_log2d,
        hyb_b_cal1d, hyb_b_cal2d)
    # The "eppl" master gate -- translate to concrete ecfg_* key(s).
    model_show = _resolve_eppl_master(
        model_show,
        ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d, ep_a_cal1d, ep_a_cal2d,
        ep_b_enabled, ep_b_nlog, ep_b_ncal, ep_b_log1d, ep_b_log2d,
        ep_b_cal1d, ep_b_cal2d)

    # Scanner lines
    scanner_lines = []
    if scan_active and scan_q_val is not None:
        q_frac = float(scan_q_val) / 100.0
        for model_key in (scan_active or []):
            scanner_lines.append({"model": model_key, "q": q_frac})

    if "advanced" in (qs_mode or []):
        _effective_qs = adv_qs or []
    else:
        # Default mode: expand band names to quantile floats
        _effective_qs = _bands_to_qs(sel_qs)
    fig = _get_bubble_fig(dict(
        selected_qs = _effective_qs,
        shade       = "shade"     in toggles,
        show_ols    = "show_ols"  in toggles,
        show_ucl    = "show_ucl"  in toggles,
        show_data   = "show_data"   in toggles,
        show_today  = "show_today"  in toggles,
        show_legend = "show_legend" in toggles,
        minor_grid  = "minor_grid" in toggles,
        show_comp   = "show_comp" in bubble_toggles,
        show_sup    = "show_sup"  in bubble_toggles,
        xscale      = xscale or BUBBLE["xscale"],
        yscale      = yscale or "log",
        xmin        = int(xrange[0]), xmax = int(xrange[1]),
        ymin        = 10 ** yrange[0], ymax = 10 ** yrange[1],
        n_future    = _ci(n_future, BUBBLE["n_future"]),
        pt_size     = _ci(ptsize, BUBBLE["pt_size"]),
        pt_alpha    = _cf(ptalpha, BUBBLE["pt_alpha"]),
        stack       = _cf(stack, BUBBLE["stack"]),
        show_stack  = bool(show_stack),
        use_lots    = bool(use_lots),
        lots        = lots_data or [],
        legend_pos  = legend_pos or "outside",
        comp_color  = LOT_MARKER_COLOR, comp_lw = TRACE_WIDTH_COMPOSITE,
        sup_color   = FALLBACK_MODEL_GRAY, sup_lw  = TRACE_WIDTH_SUPPORT,
        active_models = model_show or [],
        palette = palette_key or "default",
        scanner_lines = scanner_lines,
        user_model = user_model_store,
        qs_mode = qs_mode or [],
        decomp_model       = decomp_model or "",
        decomp_components  = list(decomp_components or []),
        decomp_mode        = decomp_mode or "individual",
        lppl_n_freqs       = list(lppl_n_freqs or []),
        lppl_weighted      = list(lppl_weighted or []),
        lppl_no_13         = list(lppl_no_13 or []),
        sigma_mode         = sigma_mode or "constant",
    ))
    if "chart_zoom" not in toggles:
        fig.update_layout(dragmode=False)
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)

    return fig


# ── Price/CAGR view pill bar ─────────────────────────────────────────────────

@callback(
    Output("bub-view-mode", "data"),
    Output("bub-price-wrap", "style"),
    Output("bub-cagr-wrap", "style"),
    Output("bub-resid-wrap", "style"),
    Output("bub-view-price", "outline"),
    Output("bub-view-cagr", "outline"),
    Output("bub-view-resid", "outline"),
    Output("bub-scale-controls", "style"),
    Output("bub-bubble-panel", "style"),
    Output("bub-cagr-fwd-wrap", "style"),
    Output("bub-xrange", "value", allow_duplicate=True),
    Input("bub-view-price", "n_clicks"),
    Input("bub-view-cagr", "n_clicks"),
    Input("bub-view-resid", "n_clicks"),
    State("bub-xrange", "value"),
    prevent_initial_call=True,
)
def toggle_bub_view(price_clicks, cagr_clicks, resid_clicks, cur_xrange):
    triggered = ctx.triggered_id
    _hide = {"display": "none"}
    _show_inline = {"display": "inline"}
    if triggered == "bub-view-cagr":
        xr = [2025, 2050] if cur_xrange == [2010, 2033] else dash.no_update
        return ("cagr", _hide, {}, _hide,
                True, False, True,
                _hide, _hide, _show_inline, xr)
    if triggered == "bub-view-resid":
        # Residuals: keep same x-range as price view, keep bubble panel visible
        xr = [2010, 2033] if cur_xrange == [2025, 2050] else dash.no_update
        return ("resid", _hide, _hide, {},
                True, True, False,
                {}, {}, _hide, xr)
    # Price
    xr = [2010, 2033] if cur_xrange == [2025, 2050] else dash.no_update
    return ("price", {}, _hide, _hide,
            False, True, True,
            {}, {}, _hide, xr)


# Sync view-mode wrappers + button outlines when bub-view-mode.data changes
# (e.g., from snapshot restore — button clicks set it directly in toggle_bub_view,
# but snapshot sets it via apply_snapshot without clicking buttons).
_app_ctx.app.clientside_callback(
    """
    function(mode) {
        var _h = {"display": "none"};
        if (mode === "cagr") {
            return [_h, {}, _h, true, false, true, _h, _h, {"display":"inline"}];
        }
        if (mode === "resid") {
            return [_h, _h, {}, true, true, false, {}, {}, _h];
        }
        /* price (default) */
        return [{}, _h, _h, false, true, true, {}, {}, _h];
    }
    """,
    Output("bub-price-wrap", "style", allow_duplicate=True),
    Output("bub-cagr-wrap", "style", allow_duplicate=True),
    Output("bub-resid-wrap", "style", allow_duplicate=True),
    Output("bub-view-price", "outline", allow_duplicate=True),
    Output("bub-view-cagr", "outline", allow_duplicate=True),
    Output("bub-view-resid", "outline", allow_duplicate=True),
    Output("bub-scale-controls", "style", allow_duplicate=True),
    Output("bub-bubble-panel", "style", allow_duplicate=True),
    Output("bub-cagr-fwd-wrap", "style", allow_duplicate=True),
    Input("bub-view-mode", "data"),
    prevent_initial_call=True,
)

# Hide "N future bubbles" slider in residuals view (doesn't apply to past data)
_app_ctx.app.clientside_callback(
    "function(mode) { return mode === 'resid' ? {display: 'none'} : {}; }",
    Output("bub-n-future-wrap", "style"),
    Input("bub-view-mode", "data"),
)

# Residuals view: cap X range slider max at (current year + 1). Price/CAGR
# views restore the full 2080 range for forward projection visibility.
_app_ctx.app.clientside_callback(
    """
    function(mode, cur_range) {
        var resid_max = (new Date()).getFullYear() + 1;
        var new_max = (mode === 'resid') ? resid_max : 2080;
        // Cap current value if it exceeds the new max
        var r = (cur_range || [2010, 2033]).slice();
        if (r[1] > new_max) r[1] = new_max;
        if (r[0] > new_max) r[0] = new_max - 1;
        return [new_max, r];
    }
    """,
    Output("bub-xrange", "max"),
    Output("bub-xrange", "value", allow_duplicate=True),
    Input("bub-view-mode", "data"),
    State("bub-xrange", "value"),
    prevent_initial_call='initial_duplicate',
)


# ── CAGR chart for tab 1 ────────────────────────────────────────────────────

@callback(
    Output("bub-cagr-graph", "figure"),
    Input("bub-view-mode", "data"),
    Input("bubble-first-render", "data"),
    Input("bub-qs", "value"),
    Input("bub-qs-adv", "value"),
    Input("bub-xrange", "value"),
    Input("bub-toggles", "value"),
    Input("bub-xscale", "value"),
    Input("bub-yscale", "value"),
    Input("bub-model-show", "value"),
    Input("bub-legend-pos", "value"),
    Input("bub-cagr-fwd-yrs", "value"),
    Input("palette-store", "data"),
    State("bub-qs-mode", "value"),
    prevent_initial_call=True,
)
def update_bub_cagr(view_mode, _first_render, sel_qs, adv_qs, xrange,
                    toggles, xscale, yscale, model_show, legend_pos,
                    fwd_yrs, palette_key, qs_mode):

    from utils import _get_cagr_fig

    toggles = toggles or []
    if "advanced" in (qs_mode or []):
        effective_qs = adv_qs or []
    else:
        effective_qs = _bands_to_qs(sel_qs) if sel_qs else [0.5]

    xrange = xrange or [2010, 2033]
    _fwd = max(1, int(fwd_yrs)) if fwd_yrs else 1
    p = dict(
        entry_q=50,
        exit_yr_lo=int(xrange[0]),
        exit_yr_hi=int(xrange[1]),
        fwd_years=_fwd,
        cagr_qs=effective_qs,
        cagr_models=model_show or ["bub"],
        palette=palette_key or "default",
        xscale=xscale or "log",
        yscale=yscale or "log",
        show_legend="show_legend" in toggles,
        minor_grid="minor_grid" in toggles,
        chart_zoom="chart_zoom" in toggles,
        legend_pos=legend_pos or "outside",
    )
    return _get_cagr_fig(p)


# ── Residuals chart for tab 1 ───────────────────────────────────────────────

@callback(
    Output("bub-resid-graph", "figure"),
    Input("bub-view-mode", "data"),
    Input("bub-xrange", "value"),
    Input("bub-toggles", "value"),
    Input("bub-xscale", "value"),
    Input("bub-model-show", "value"),
    Input("bub-bubble-toggles", "value"),
    Input("bub-n-future", "value"),
    Input("bub-legend-pos", "value"),
    Input("palette-store", "data"),
    Input("bub-decomp-model",       "value"),
    Input("bub-decomp-components",  "value"),
    Input("lppl-n-freqs",           "value"),
    Input("lppl-weighted",          "value"),
    Input("lppl-no-13",             "value"),
    State("user-model-store", "data"),
    prevent_initial_call=True,
)
def update_bub_resid(view_mode, xrange, toggles, xscale, model_show,
                     bub_toggles, n_future, legend_pos, palette_key,
                     decomp_model, decomp_components,
                     lppl_n_freqs, lppl_weighted, lppl_no_13,
                     user_model_store):
    from utils import _get_resid_fig
    toggles = toggles or []
    xrange = xrange or [2010, 2033]
    p = dict(
        xmin=int(xrange[0]), xmax=int(xrange[1]),
        active_models=sorted(model_show or []),
        bub_toggles=sorted(bub_toggles or []),
        n_future=int(n_future) if n_future is not None else 3,
        palette=palette_key or "default",
        xscale=xscale or "log",
        show_legend="show_legend" in toggles,
        show_today="show_today" in toggles,
        minor_grid="minor_grid" in toggles,
        chart_zoom="chart_zoom" in toggles,
        legend_pos=legend_pos or "outside",
        user_model=user_model_store,
        decomp_model=decomp_model or "",
        decomp_components=list(decomp_components or []),
        lppl_n_freqs=list(lppl_n_freqs or []),
        lppl_weighted=list(lppl_weighted or []),
        lppl_no_13=list(lppl_no_13 or []),
    )
    fig = _get_resid_fig(p)
    if "chart_zoom" not in toggles:
        fig.update_layout(dragmode=False)
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
    return fig


# auto_bubble_yrange — clientside for zero round-trip on mobile.
# Pre-computed envelope grids (AUTO_Y_GRID) are stored in a dcc.Store
# at layout time. The JS interpolates the grid at the current xrange.
_app_ctx.app.clientside_callback(
    """
    function(xrange, auto_y, yscale, model_show, grid, cur_yr) {
        var NU = window.dash_clientside.no_update;
        if (!auto_y || !auto_y.length || !xrange || !grid) return NU;

        var xmin = xrange[0], xmax = xrange[1];
        /* Convert calendar year to t (years since genesis) */
        var GENESIS_EPOCH_DAYS = grid.genesis_epoch_days;
        var GENESIS_MS = GENESIS_EPOCH_DAYS * 86400000;
        var MS_PER_YR = 365.25 * 86400000;
        var t_lo = Math.max(((new Date(xmin, 0, 1)).getTime() - GENESIS_MS) / MS_PER_YR, 0.1);
        var t_hi = ((new Date(xmax, 0, 1)).getTime() - GENESIS_MS) / MS_PER_YR;

        var t_grid = grid.t;
        var models = grid.models;
        var active = model_show || [];

        /* Linear interpolation helper */
        function interp(arr, t) {
            if (t <= t_grid[0]) return arr[0];
            if (t >= t_grid[t_grid.length - 1]) return arr[arr.length - 1];
            for (var i = 0; i < t_grid.length - 1; i++) {
                if (t_grid[i] <= t && t <= t_grid[i + 1]) {
                    var f = (t - t_grid[i]) / (t_grid[i + 1] - t_grid[i]);
                    return arr[i] + f * (arr[i + 1] - arr[i]);
                }
            }
            return arr[arr.length - 1];
        }

        /* Find base model */
        var base_key = (active.indexOf('bub') !== -1) ? 'bub' : null;
        if (!base_key) {
            for (var k = 0; k < active.length; k++) {
                if (models[active[k]]) { base_key = active[k]; break; }
            }
        }
        if (!base_key) base_key = 'bub';
        var base = models[base_key];
        if (!base) return NU;

        var y_lo_p = interp(base.lo, t_lo);
        var y_hi_p = interp(base.hi, t_hi);

        /* Extend with secondary models */
        for (var m = 0; m < active.length; m++) {
            var mk = active[m];
            var md = models[mk];
            if (!md || mk === base_key) continue;
            y_lo_p = Math.min(y_lo_p, interp(md.lo, t_lo));
            y_hi_p = Math.max(y_hi_p, interp(md.hi, t_hi));
        }

        /* Cap and round */
        var extreme = (active.indexOf('s2f') !== -1 || active.indexOf('exp') !== -1);
        var y_cap = extreme ? 20.0 : 9.0;
        var y_lo, y_hi;
        if ((yscale || 'log') === 'log') {
            y_lo = Math.floor(y_lo_p * 2) / 2;
            y_hi = Math.ceil(y_hi_p * 2) / 2;
            y_lo = Math.max(-1.5, Math.min(y_lo, 6.0));
            y_hi = Math.min(y_cap, Math.max(y_hi, 1.0));
        } else {
            y_lo = -2.0;
            y_hi = Math.ceil(y_hi_p * 2) / 2;
            y_hi = Math.min(y_cap, Math.max(y_hi, 1.0));
        }
        var new_lo = Math.round(y_lo * 10) / 10;
        var new_hi = Math.round(y_hi * 10) / 10;

        /* Skip if unchanged */
        if (cur_yr && cur_yr.length === 2
            && Math.round(cur_yr[0] * 10) / 10 === new_lo
            && Math.round(cur_yr[1] * 10) / 10 === new_hi) {
            return NU;
        }
        return [new_lo, new_hi];
    }
    """,
    Output("bub-yrange", "value", allow_duplicate=True),
    Input("bub-xrange",  "value"),
    Input("bub-auto-y",  "value"),
    Input("bub-yscale",  "value"),
    Input("bub-model-show", "value"),
    State("auto-y-grid", "data"),
    State("bub-yrange",  "value"),
    prevent_initial_call=True,
)


_app_ctx.app.clientside_callback(
    """
    function(auto_y) {
        return (auto_y && auto_y.length) ? {display: "none"} : {};
    }
    """,
    Output("bub-yrange-wrap", "style"),
    Input("bub-auto-y", "value"),
)

_YRANGE_BASIC = {-2: "1\u00a2", 0: "$1", 2: "$100", 4: "$10K", 6: "$1M", 9: "$1B"}
_YRANGE_EXT = {-2: "1\u00a2", 0: "$1", 2: "$100", 4: "$10K", 6: "$1M", 9: "$1B",
               12: "$1T", 15: "$1Q", 18: "$1Qi"}


@callback(
    Output("bub-yrange", "max"),
    Output("bub-yrange", "marks"),
    Input("bub-model-show", "value"),
    prevent_initial_call=True,
)
def update_yrange_slider_limits(model_show):
    """Extend Y range slider when S2F or Exponential are active."""
    if {"s2f", "exp"}.intersection(model_show or []):
        return 20, _YRANGE_EXT
    return 9, _YRANGE_BASIC


@callback(
    Output("heatmap-graph",  "figure"),
    Output("hm-mc-results",  "data"),
    Output("hm-mc-status",   "children"),
    Output("hm-mc-panel",    "style"),
    Output("hm-swipe-indicator", "style"),
    Output("hm-mc-rendered-key", "data"),
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Output("mc-save-tab", "data", allow_duplicate=True),
    Input("heatmap-first-render", "data"),
    Input("hm-active-model", "data"),
    Input("hm-entry-yr",  "value"),
    Input("hm-entry-q",   "value"),
    Input("hm-exit-range","value"),
    Input("hm-exit-qs",   "value"),
    Input("hm-mode",      "value"),
    Input("hm-b1",        "value"),
    Input("hm-b2",        "value"),
    Input("hm-c-lo",      "value"),
    Input("hm-c-mid1",    "value"),
    Input("hm-c-mid2",    "value"),
    Input("hm-c-hi",      "value"),
    Input("hm-grad",      "value"),
    Input("hm-vfmt",      "value"),
    Input("hm-cell-fs",   "value"),
    Input("hm-toggles",   "value"),
    Input("hm-stack",     "value"),
    Input("hm-use-lots",  "value"),
    Input("effective-lots","data"),
    Input("hm-mc-enable",  "value"),
    Input("hm-mc-amount",  "value"),
    Input("hm-mc-infl",    "value"),
    Input("hm-mc-bins",    "value"),
    Input("hm-mc-regime",  "value"),
    Input("hm-mc-sims",    "value"),
    Input("hm-mc-years",   "value"),
    Input("hm-mc-freq",    "value"),
    Input("hm-mc-window",  "value"),
    Input("hm-mc-start-yr", "value"),
    Input("hm-mc-entry-q",  "value"),
    Input("hm-mc-loaded",   "data"),
    Input("mc-pay-trigger", "data"),
    Input("hm-model-show",  "value"),
    Input("hm-mc-model-src", "value"),
    State("btc-price-store", "data"),
    State("hm-mc-results",  "data"),
    State("mc-pay-token",   "data"),
    State("hm-mc-rendered-key", "data"),
    Input("palette-store",      "data"),
    State("lppl-n-freqs",       "value"),
    State("lppl-weighted",      "value"),
    State("lppl-no-13",         "value"),
    State("hybppl-cfg-a-nlog",  "value"),
    State("hybppl-cfg-a-ncal",  "value"),
    State("hybppl-cfg-a-log1d", "value"),
    State("hybppl-cfg-a-log2d", "value"),
    State("hybppl-cfg-a-cal1d", "value"),
    State("hybppl-cfg-a-cal2d", "value"),
    State("eppl-cfg-a-nlog",  "value"),
    State("eppl-cfg-a-ncal",  "value"),
    State("eppl-cfg-a-log1d", "value"),
    State("eppl-cfg-a-log2d", "value"),
    State("eppl-cfg-a-cal1d", "value"),
    State("eppl-cfg-a-cal2d", "value"),
    prevent_initial_call=True,
)
def update_heatmap(_first_render, hm_model, entry_yr, entry_q, exit_range, exit_qs, mode,
                   b1, b2, c_lo, c_mid1, c_mid2, c_hi, grad,
                   vfmt, cell_fs, toggles, stack, use_lots, lots_data,
                   mc_enable, mc_amount, mc_infl, mc_bins, mc_regime, mc_sims, mc_years, mc_freq, mc_window,
                   mc_start_yr, mc_entry_q, _mc_loaded, _pay_trigger, model_show, mc_model_src,
                   live_price, mc_cached, pay_token, mc_auth, palette_key,
                   lppl_n_freqs=None, lppl_weighted=None, lppl_no_13=None,
                   hyb_a_nlog=None, hyb_a_ncal=None,
                   hyb_a_log1d=None, hyb_a_log2d=None,
                   hyb_a_cal1d=None, hyb_a_cal2d=None,
                   ep_a_nlog=None, ep_a_ncal=None,
                   ep_a_log1d=None, ep_a_log2d=None,
                   ep_a_cal1d=None, ep_a_cal2d=None):
    exit_range = exit_range or [entry_yr or 2025, (entry_yr or 2025) + 10]
    toggles    = toggles or []
    yr_now = pd.Timestamp.today().year
    hm_model = hm_model or "bub"
    # Legacy snapshot values (qr, exp, s2f, individual LPPL flavors) ->
    # surviving pill model. Keeps pill highlight and chart rendering
    # consistent when decoding old share links.
    from callbacks.routing import _HM_PILL_MODELS, _HM_LEGACY_MODEL_FALLBACK
    if hm_model not in _HM_PILL_MODELS and hm_model != "lppl" and hm_model != "hybppl" and hm_model != "eppl":
        hm_model = _HM_LEGACY_MODEL_FALLBACK.get(hm_model, hm_model)
    # Translate LPPL master to specific flavor via global config.
    # Only for the non-MC path; MC uses hm-mc-model-src separately.
    hm_model = _resolve_hm_lppl_master(
        hm_model, lppl_n_freqs, lppl_weighted, lppl_no_13)
    # Translate HybPPL master to specific cfg_* key.
    hm_model = _resolve_hm_hybppl_master(
        hm_model,
        hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d,
        hyb_a_cal1d, hyb_a_cal2d)
    # Translate EPPL master to specific ecfg_* key.
    hm_model = _resolve_hm_eppl_master(
        hm_model,
        ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d,
        ep_a_cal1d, ep_a_cal2d)

    # Only use live ticker price when entry_yr == current year AND the user
    # hasn't modified the entry percentile away from the ticker value.
    def _use_live(eyr_val, eq_val):
        if not live_price or _ci(eyr_val, yr_now) != yr_now:
            return None
        ticker_pct = _find_lot_percentile(today_t(_app_ctx.M.genesis), float(live_price), _app_ctx.M.qr_fits)
        if ticker_pct is None:
            return None
        ticker_q = round(ticker_pct * 100, 1)
        if abs(_cf(eq_val, 50) - ticker_q) > 0.05:
            return None  # user changed entry percentile
        return float(live_price)

    shared_params = dict(
        entry_yr     = _ci(entry_yr, yr_now),
        entry_q      = _cf(entry_q, 50),
        live_price   = _use_live(entry_yr, entry_q),
        exit_yr_lo   = int(exit_range[0]),
        exit_yr_hi   = int(exit_range[1]),
        exit_qs      = exit_qs or [],
        color_mode   = _ci(mode, HEATMAP["color_mode"]),
        b1           = _cf(b1, _app_ctx.M.CAGR_SEG_B1),
        b2           = _cf(b2, _app_ctx.M.CAGR_SEG_B2),
        c_lo         = c_lo   or _app_ctx.M.CAGR_SEG_C_LO,
        c_mid1       = c_mid1 or _app_ctx.M.CAGR_SEG_C_MID1,
        c_mid2       = c_mid2 or _app_ctx.M.CAGR_SEG_C_MID2,
        c_hi         = c_hi   or _app_ctx.M.CAGR_SEG_C_HI,
        n_disc       = _ci(grad, HEATMAP["n_disc"]),
        vfmt         = vfmt or HEATMAP["vfmt"],
        cell_font_size = _ci(cell_fs, HEATMAP["cell_font_size"]),
        show_colorbar = "colorbar" in toggles,
        stack        = _cf(stack, HEATMAP["stack"]),
        use_lots     = bool(use_lots),
        lots         = lots_data or [],
        active_models = [k for k in (model_show or []) if k not in _app_ctx.MODEL_SENTINELS],
        palette = palette_key or "default",
    )

    # MC heatmap via sandwich helper
    mc_enabled = bool(mc_enable) and _app_ctx._HAS_MARKOV
    mc_ok, is_free, mc_p, blocked = _mc_setup(
        "hm", mc_enable, mc_years, mc_start_yr, mc_entry_q,
        mc_bins, mc_sims, mc_freq, mc_window, mc_amount, mc_infl,
        mc_cached, _cf(live_price, 0), mc_regime, None, pay_token,
        mc_auth=mc_auth,
        stack=stack, amount_default=100, infl_default=0.0,
        start_yr_default=yr_now,
        mc_model_src=mc_model_src)

    # Heatmap-specific: cap MC training window at start year for historical sims
    if mc_ok and not mc_p.get("mc_stale"):
        mc_sy = mc_p.get("mc_start_yr", yr_now)
        if mc_sy < yr_now:
            win = mc_p.get("mc_window")
            if win and isinstance(win, list) and len(win) >= 2:
                mc_p["mc_window"] = [win[0], min(win[1], mc_sy)]

    mc_result = None
    if hm_model == "mc":
        # MC model selected via pill — render MC heatmap
        if mc_ok:
            mc_params = dict(shared_params, **mc_p,
                             live_price=_use_live(mc_p["mc_start_yr"], mc_p["mc_entry_q"]))
            fig, mc_result = _get_mc_heatmap_fig(mc_params)
        else:
            fig = _get_heatmap_fig(dict(shared_params))
    else:
        # QR or alternative model heatmap
        fig = _get_heatmap_fig(dict(shared_params, hm_model=hm_model))

    mc_result = _strip_free_paths(is_free, mc_result)
    store_val, status, show_modal = _mc_status(mc_result, mc_cached, mc_enabled)
    rendered_key = ({"years": _ci(mc_p["mc_years"], MC_DEFAULT_YEARS),
                     "start_yr": _ci(mc_p["mc_start_yr"], MC_DEFAULT_START_YR),
                     "entry_q": round(_cf(mc_p["mc_entry_q"], MC_DEFAULT_ENTRY_Q), 1),
                     "bins": int(mc_p.get("mc_bins", MC_BINS)),
                     "sims": int(mc_p.get("mc_sims", MC_SIMS)),
                     "freq": mc_p.get("mc_freq") or MC_FREQ}
                    if mc_ok else None)

    # Show/hide MC panel and swipe indicator
    model_show = model_show if model_show is not None else []
    mc_visible = mc_enabled and "mc" in model_show
    mc_panel_style = {} if mc_visible else {"display": "none"}
    indicator_style = {"display": "none"}

    if "chart_zoom" not in toggles:
        fig.update_layout(dragmode=False)

    return (fig, store_val, status, mc_panel_style, indicator_style,
            rendered_key,
            show_modal, "hm" if show_modal else dash.no_update)


# ── CAGR line chart (below heatmap) ─────────────────────────────────────────

@callback(
    Output("dca-graph", "figure"),
    Output("dca-mc-results", "data"),
    Output("dca-mc-status", "children"),
    Output("dca-mc-rendered-key", "data"),
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Output("mc-save-tab", "data", allow_duplicate=True),
    Output("dca-mc-unblocked", "data"),
    Output("dca-yr-range", "value", allow_duplicate=True),
    Input("dca-first-render", "data"),
    Input("dca-stack",    "value"),
    Input("dca-use-lots", "value"),
    Input("dca-amount",   "value"),
    Input("dca-freq",     "value"),
    Input("dca-infl",     "value"),
    Input("dca-yr-range", "value"),
    Input("dca-disp",     "value"),
    Input("dca-toggles",  "value"),
    Input("dca-legend-pos","value"),
    Input("dca-qs",       "value"),
    Input("dca-qs-adv",   "value"),
    Input("lppl-n-freqs", "value"),
    Input("lppl-weighted","value"),
    Input("lppl-no-13",   "value"),
    Input("hybppl-cfg-a-nlog", "value"),
    Input("hybppl-cfg-a-ncal", "value"),
    Input("hybppl-cfg-a-log1d","value"),
    Input("hybppl-cfg-a-log2d","value"),
    Input("hybppl-cfg-a-cal1d","value"),
    Input("hybppl-cfg-a-cal2d","value"),
    Input("hybppl-cfg-b-enabled","value"),
    Input("hybppl-cfg-b-nlog", "value"),
    Input("hybppl-cfg-b-ncal", "value"),
    Input("hybppl-cfg-b-log1d","value"),
    Input("hybppl-cfg-b-log2d","value"),
    Input("hybppl-cfg-b-cal1d","value"),
    Input("hybppl-cfg-b-cal2d","value"),
    Input("eppl-cfg-a-nlog", "value"),
    Input("eppl-cfg-a-ncal", "value"),
    Input("eppl-cfg-a-log1d","value"),
    Input("eppl-cfg-a-log2d","value"),
    Input("eppl-cfg-a-cal1d","value"),
    Input("eppl-cfg-a-cal2d","value"),
    Input("eppl-cfg-b-enabled","value"),
    Input("eppl-cfg-b-nlog", "value"),
    Input("eppl-cfg-b-ncal", "value"),
    Input("eppl-cfg-b-log1d","value"),
    Input("eppl-cfg-b-log2d","value"),
    Input("eppl-cfg-b-cal1d","value"),
    Input("eppl-cfg-b-cal2d","value"),
    Input("effective-lots","data"),
    Input("dca-sc-enable",  "value"),
    Input("dca-sc-loan",    "value"),
    Input("dca-sc-rate",    "value"),
    Input("dca-sc-term",    "value"),
    Input("dca-sc-type",         "value"),
    Input("dca-sc-repeats",      "value"),
    Input("dca-sc-entry-mode",   "value"),
    Input("dca-sc-custom-price", "value"),
    Input("dca-sc-tax",          "value"),
    Input("dca-sc-rollover",     "value"),
    Input("dca-mc-enable",  "value"),
    Input("dca-mc-bins",    "value"),
    Input("dca-mc-regime",  "value"),
    Input("dca-mc-sims",    "value"),
    Input("dca-mc-years",   "value"),
    Input("dca-mc-window",  "value"),
    Input("dca-mc-start-yr", "value"),
    Input("dca-mc-entry-q", "value"),
    Input("dca-mc-loaded",  "data"),
    Input("mc-pay-trigger", "data"),
    Input("dca-model-show", "value"),
    Input("dca-mc-model-src", "value"),
    State("btc-price-store","data"),
    State("dca-mc-results", "data"),
    State("mc-pay-token",   "data"),
    State("dca-mc-unblocked", "data"),
    State("dca-mc-rendered-key", "data"),
    Input("palette-store",      "data"),
    State("dca-qs-mode",        "value"),
    State("user-model-store",   "data"),
    prevent_initial_call=True,
)
def update_dca(_first_render, stack, use_lots, amount, freq, dca_infl, yr_range, disp, toggles, legend_pos, sel_qs, adv_qs,
               lppl_n_freqs, lppl_weighted, lppl_no_13,
               hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d,
               hyb_a_cal1d, hyb_a_cal2d,
               hyb_b_enabled, hyb_b_nlog, hyb_b_ncal, hyb_b_log1d, hyb_b_log2d,
               hyb_b_cal1d, hyb_b_cal2d,
               ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d,
               ep_a_cal1d, ep_a_cal2d,
               ep_b_enabled, ep_b_nlog, ep_b_ncal, ep_b_log1d, ep_b_log2d,
               ep_b_cal1d, ep_b_cal2d,
               lots_data,
               sc_enable, sc_loan, sc_rate, sc_term, sc_type, sc_repeats,
               sc_entry_mode, sc_custom_price, sc_tax, sc_rollover,
               mc_enable, mc_bins, mc_regime, mc_sims, mc_years, mc_window,
               mc_start_yr, mc_entry_q, _mc_loaded, _pay_trigger, model_show, mc_model_src,
               price_data, mc_cached, pay_token, mc_unblocked, mc_auth, palette_key,
               qs_mode=None, user_model_store=None):
    toggles    = toggles or []
    yr_range   = yr_range or [2024, 2034]
    live_price = _cf(price_data, 0)
    _advanced  = "advanced" in (qs_mode or [])
    _effective_qs = (adv_qs or []) if _advanced else (
        _bands_to_qs(sel_qs) if sel_qs and isinstance(sel_qs[0], str) else (sel_qs or []))
    mc_ok, is_free, mc_p, blocked = _mc_setup(
        "dca", mc_enable, mc_years, mc_start_yr, mc_entry_q,
        mc_bins, mc_sims, freq, mc_window, amount, dca_infl,
        mc_cached, live_price, mc_regime, mc_unblocked, pay_token,
        mc_auth=mc_auth,
        stack=stack, amount_default=DCA["amount"], infl_default=float(DCA["inflation"]), start_yr_default=2026,
        mc_model_src=mc_model_src)
    model_show = model_show if model_show is not None else []
    model_show = _resolve_lppl_master(
        model_show, lppl_n_freqs, lppl_weighted, lppl_no_13)
    model_show = _resolve_hybppl_master(
        model_show,
        hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d, hyb_a_cal1d, hyb_a_cal2d,
        hyb_b_enabled, hyb_b_nlog, hyb_b_ncal, hyb_b_log1d, hyb_b_log2d,
        hyb_b_cal1d, hyb_b_cal2d)
    model_show = _resolve_eppl_master(
        model_show,
        ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d, ep_a_cal1d, ep_a_cal2d,
        ep_b_enabled, ep_b_nlog, ep_b_ncal, ep_b_log1d, ep_b_log2d,
        ep_b_cal1d, ep_b_cal2d)
    fig, mc_result = _get_dca_fig(dict(
        start_stack    = _cf(stack, DCA["start_stack"]),
        use_lots       = bool(use_lots),
        amount         = _ci(amount, DCA["amount"], lo=0, hi=_app_ctx.MAX_USD),
        freq           = freq or "Monthly",
        inflation      = _cf(dca_infl, DCA["inflation"]),
        start_yr       = int(yr_range[0]),
        end_yr         = int(yr_range[1]),
        disp_mode      = disp or "btc",
        log_y          = "log_y"      in toggles,
        annotate       = "annotate"   in toggles,
        discrete       = "discrete"   in toggles,
        shade          = "shade"      in toggles,
        show_today     = "show_today" in toggles,
        show_legend    = "show_legend" in toggles,
        legend_pos     = legend_pos or "outside",
        minor_grid     = "minor_grid" in toggles,
        selected_qs    = _effective_qs,
        lots           = lots_data or [],
        sc_enabled     = bool(sc_enable),
        sc_loan_amount = _cf(sc_loan, 0),
        sc_rate        = _cf(sc_rate, DCA["sc_rate"]),
        sc_loan_type   = sc_type or "interest_only",
        sc_term_months = _cf(sc_term, DCA["sc_term_months"]),
        sc_repeats     = _ci(sc_repeats, 0),
        sc_live_price   = live_price,
        sc_entry_mode   = sc_entry_mode or "live",
        sc_custom_price = _cf(sc_custom_price, DCA["sc_custom_price"]),
        sc_tax_rate     = _cf(sc_tax, 33, lo=0, hi=100) / 100.0,
        sc_rollover     = bool(sc_rollover),
        show_qr        = "bub" in model_show,
        show_mc        = "mc" in model_show,
        active_models  = [k for k in model_show if k != "mc"],  # pass "bub" through for toggle
        palette = palette_key or "default",
        user_model = user_model_store,
        **mc_p,
    ))
    fig, store_val, status, rendered_key, show_modal, ub_val = _mc_finalize(
        "dca", fig, mc_result, mc_cached, mc_enable, mc_ok,
        is_free, blocked, mc_p["mc_years"], mc_p["mc_start_yr"],
        mc_p["mc_entry_q"], toggles, mc_stale=mc_p.get("mc_stale", False),
        mc_p=mc_p)
    yr_adjust = dash.no_update
    if mc_ok and mc_p.get("mc_start_yr"):
        mc_sy = int(mc_p["mc_start_yr"])
        if mc_sy < int(yr_range[0]):
            yr_adjust = [mc_sy, int(yr_range[1])]
    return (fig, store_val, status, rendered_key, show_modal,
            "dca" if show_modal else dash.no_update, ub_val, yr_adjust)


@callback(
    Output("retire-graph", "figure"),
    Output("ret-mc-results", "data"),
    Output("ret-mc-status", "children"),
    Output("ret-mc-rendered-key", "data"),
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Output("mc-save-tab", "data", allow_duplicate=True),
    Output("ret-mc-unblocked", "data"),
    Output("ret-yr-range", "value", allow_duplicate=True),
    Input("retire-first-render", "data"),
    Input("ret-stack",    "value"),
    Input("ret-use-lots", "value"),
    Input("ret-wd",       "value"),
    Input("ret-freq",     "value"),
    Input("ret-yr-range", "value"),
    Input("ret-infl",     "value"),
    Input("ret-disp",     "value"),
    Input("ret-toggles",  "value"),
    Input("ret-legend-pos","value"),
    Input("ret-qs",       "value"),
    Input("ret-qs-adv",   "value"),
    Input("lppl-n-freqs", "value"),
    Input("lppl-weighted","value"),
    Input("lppl-no-13",   "value"),
    Input("hybppl-cfg-a-nlog", "value"),
    Input("hybppl-cfg-a-ncal", "value"),
    Input("hybppl-cfg-a-log1d","value"),
    Input("hybppl-cfg-a-log2d","value"),
    Input("hybppl-cfg-a-cal1d","value"),
    Input("hybppl-cfg-a-cal2d","value"),
    Input("hybppl-cfg-b-enabled","value"),
    Input("hybppl-cfg-b-nlog", "value"),
    Input("hybppl-cfg-b-ncal", "value"),
    Input("hybppl-cfg-b-log1d","value"),
    Input("hybppl-cfg-b-log2d","value"),
    Input("hybppl-cfg-b-cal1d","value"),
    Input("hybppl-cfg-b-cal2d","value"),
    Input("eppl-cfg-a-nlog", "value"),
    Input("eppl-cfg-a-ncal", "value"),
    Input("eppl-cfg-a-log1d","value"),
    Input("eppl-cfg-a-log2d","value"),
    Input("eppl-cfg-a-cal1d","value"),
    Input("eppl-cfg-a-cal2d","value"),
    Input("eppl-cfg-b-enabled","value"),
    Input("eppl-cfg-b-nlog", "value"),
    Input("eppl-cfg-b-ncal", "value"),
    Input("eppl-cfg-b-log1d","value"),
    Input("eppl-cfg-b-log2d","value"),
    Input("eppl-cfg-b-cal1d","value"),
    Input("eppl-cfg-b-cal2d","value"),
    Input("effective-lots","data"),
    Input("ret-mc-enable",  "value"),
    Input("ret-mc-bins",    "value"),
    Input("ret-mc-regime",  "value"),
    Input("ret-mc-sims",    "value"),
    Input("ret-mc-years",   "value"),
    Input("ret-mc-window",  "value"),
    Input("ret-mc-start-yr", "value"),
    Input("ret-mc-entry-q",  "value"),
    Input("ret-mc-loaded",   "data"),
    Input("mc-pay-trigger", "data"),
    Input("ret-model-show", "value"),
    Input("ret-mc-model-src", "value"),
    State("btc-price-store","data"),
    State("ret-mc-results", "data"),
    State("mc-pay-token",   "data"),
    State("ret-mc-unblocked", "data"),
    State("ret-mc-rendered-key", "data"),
    Input("palette-store",      "data"),
    State("ret-qs-mode",        "value"),
    State("user-model-store",   "data"),
    prevent_initial_call=True,
)
def update_retire(_first_render, stack, use_lots, wd, freq, yr_range, infl, disp, toggles, legend_pos, sel_qs, adv_qs,
                  lppl_n_freqs, lppl_weighted, lppl_no_13,
                  hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d,
                  hyb_a_cal1d, hyb_a_cal2d,
                  hyb_b_enabled, hyb_b_nlog, hyb_b_ncal, hyb_b_log1d, hyb_b_log2d,
                  hyb_b_cal1d, hyb_b_cal2d,
                  ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d,
                  ep_a_cal1d, ep_a_cal2d,
                  ep_b_enabled, ep_b_nlog, ep_b_ncal, ep_b_log1d, ep_b_log2d,
                  ep_b_cal1d, ep_b_cal2d,
                  lots_data,
                  mc_enable, mc_bins, mc_regime, mc_sims, mc_years, mc_window,
                  mc_start_yr, mc_entry_q, _mc_loaded, _pay_trigger, model_show, mc_model_src,
                  price_data, mc_cached, pay_token, mc_unblocked, mc_auth, palette_key,
                  qs_mode=None, user_model_store=None):
    toggles  = toggles or []
    yr_range = yr_range or [RETIRE["start_yr"], RETIRE["end_yr"]]
    _advanced = "advanced" in (qs_mode or [])
    _effective_qs = (adv_qs or []) if _advanced else (
        _bands_to_qs(sel_qs) if sel_qs and isinstance(sel_qs[0], str) else (sel_qs or []))
    mc_ok, is_free, mc_p, blocked = _mc_setup(
        "ret", mc_enable, mc_years, mc_start_yr, mc_entry_q,
        mc_bins, mc_sims, freq, mc_window, wd, infl,
        mc_cached, _cf(price_data, 0), mc_regime, mc_unblocked, pay_token,
        mc_auth=mc_auth,
        stack=stack, amount_default=RETIRE["wd_amount"], infl_default=RETIRE["inflation"], start_yr_default=RETIRE["start_yr"],
        mc_model_src=mc_model_src)
    model_show = model_show if model_show is not None else []
    model_show = _resolve_lppl_master(
        model_show, lppl_n_freqs, lppl_weighted, lppl_no_13)
    model_show = _resolve_hybppl_master(
        model_show,
        hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d, hyb_a_cal1d, hyb_a_cal2d,
        hyb_b_enabled, hyb_b_nlog, hyb_b_ncal, hyb_b_log1d, hyb_b_log2d,
        hyb_b_cal1d, hyb_b_cal2d)
    model_show = _resolve_eppl_master(
        model_show,
        ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d, ep_a_cal1d, ep_a_cal2d,
        ep_b_enabled, ep_b_nlog, ep_b_ncal, ep_b_log1d, ep_b_log2d,
        ep_b_cal1d, ep_b_cal2d)
    fig, mc_result = _get_retire_fig(dict(
        start_stack  = _cf(stack, RETIRE["start_stack"]),
        use_lots     = bool(use_lots),
        wd_amount    = _ci(wd, RETIRE["wd_amount"], lo=0, hi=_app_ctx.MAX_USD),
        freq         = freq or "Monthly",
        start_yr     = int(yr_range[0]),
        end_yr       = int(yr_range[1]),
        inflation    = _cf(infl, RETIRE["inflation"]),
        disp_mode    = disp or "btc",
        log_y        = "log_y"     in toggles,
        annotate     = "annotate"  in toggles,
        discrete     = "discrete"  in toggles,
        shade        = "shade"     in toggles,
        show_legend  = "show_legend" in toggles,
        legend_pos   = legend_pos or "outside",
        minor_grid   = "minor_grid" in toggles,
        selected_qs  = _effective_qs,
        lots         = lots_data or [],
        show_qr      = "bub" in model_show,
        show_mc      = "mc" in model_show,
        active_models = [k for k in model_show if k != "mc"],  # pass "bub" through for toggle
        palette = palette_key or "default",
        user_model = user_model_store,
        **mc_p,
    ))
    fig, store_val, status, rendered_key, show_modal, ub_val = _mc_finalize(
        "ret", fig, mc_result, mc_cached, mc_enable, mc_ok,
        is_free, blocked, mc_p["mc_years"], mc_p["mc_start_yr"],
        mc_p["mc_entry_q"], toggles, mc_stale=mc_p.get("mc_stale", False),
        mc_p=mc_p)

    # Nudge year range slider if MC starts before visible range
    yr_adjust = dash.no_update
    if mc_ok and mc_p.get("mc_start_yr"):
        mc_sy = int(mc_p["mc_start_yr"])
        if mc_sy < int(yr_range[0]):
            yr_adjust = [mc_sy, int(yr_range[1])]

    return (fig, store_val, status, rendered_key, show_modal,
            "ret" if show_modal else dash.no_update, ub_val, yr_adjust)


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks — HODL Supercharger
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("supercharge-graph", "figure"),
    Output("sc-mc-results",     "data"),
    Output("sc-mc-status",      "children"),
    Output("sc-mc-rendered-key", "data"),
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Output("mc-save-tab", "data", allow_duplicate=True),
    Output("sc-mc-unblocked",   "data"),
    Output("sc-start-yr", "value", allow_duplicate=True),
    Input("supercharge-first-render", "data"),
    Input("sc-stack",        "value"),
    Input("sc-use-lots",     "value"),
    Input("sc-start-yr",     "value"),
    Input("sc-d0",           "value"),
    Input("sc-d1",           "value"),
    Input("sc-d2",           "value"),
    Input("sc-d3",           "value"),
    Input("sc-d4",           "value"),
    Input("sc-freq",         "value"),
    Input("sc-infl",         "value"),
    Input("sc-qs",           "value"),
    Input("sc-qs-adv",       "value"),
    Input("lppl-n-freqs",    "value"),
    Input("lppl-weighted",   "value"),
    Input("lppl-no-13",      "value"),
    Input("hybppl-cfg-a-nlog", "value"),
    Input("hybppl-cfg-a-ncal", "value"),
    Input("hybppl-cfg-a-log1d","value"),
    Input("hybppl-cfg-a-log2d","value"),
    Input("hybppl-cfg-a-cal1d","value"),
    Input("hybppl-cfg-a-cal2d","value"),
    Input("hybppl-cfg-b-enabled","value"),
    Input("hybppl-cfg-b-nlog", "value"),
    Input("hybppl-cfg-b-ncal", "value"),
    Input("hybppl-cfg-b-log1d","value"),
    Input("hybppl-cfg-b-log2d","value"),
    Input("hybppl-cfg-b-cal1d","value"),
    Input("hybppl-cfg-b-cal2d","value"),
    Input("eppl-cfg-a-nlog", "value"),
    Input("eppl-cfg-a-ncal", "value"),
    Input("eppl-cfg-a-log1d","value"),
    Input("eppl-cfg-a-log2d","value"),
    Input("eppl-cfg-a-cal1d","value"),
    Input("eppl-cfg-a-cal2d","value"),
    Input("eppl-cfg-b-enabled","value"),
    Input("eppl-cfg-b-nlog", "value"),
    Input("eppl-cfg-b-ncal", "value"),
    Input("eppl-cfg-b-log1d","value"),
    Input("eppl-cfg-b-log2d","value"),
    Input("eppl-cfg-b-cal1d","value"),
    Input("eppl-cfg-b-cal2d","value"),
    Input("sc-mode",         "value"),
    Input("sc-wd",           "value"),
    Input("sc-end-yr",       "value"),
    Input("sc-target-yr",    "value"),
    Input("sc-disp",         "value"),
    Input("sc-toggles",      "value"),
    Input("sc-legend-pos",   "value"),
    Input("sc-chart-layout", "value"),
    Input("sc-display-q",    "value"),
    Input("effective-lots",  "data"),
    Input("sc-mc-enable",    "value"),
    Input("sc-mc-bins",      "value"),
    Input("sc-mc-regime",    "value"),
    Input("sc-mc-sims",      "value"),
    Input("sc-mc-years",     "value"),
    Input("sc-mc-window",    "value"),
    Input("sc-mc-start-yr",  "value"),
    Input("sc-mc-entry-q",   "value"),
    Input("sc-mc-loaded",    "data"),
    Input("mc-pay-trigger", "data"),
    Input("sc-model-show",  "value"),
    Input("sc-mc-model-src", "value"),
    State("btc-price-store", "data"),
    State("sc-mc-results",   "data"),
    State("mc-pay-token",   "data"),
    State("sc-mc-unblocked", "data"),
    State("sc-mc-rendered-key", "data"),
    Input("palette-store",     "data"),
    State("sc-qs-mode",        "value"),
    State("viewport-width",    "data"),
    State("user-model-store",  "data"),
    prevent_initial_call=True,
)
def update_supercharge(_first_render, stack, use_lots, start_yr,
                       d0, d1, d2, d3, d4,
                       freq, infl, sel_qs, adv_qs,
                       lppl_n_freqs, lppl_weighted, lppl_no_13,
                       hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d,
                       hyb_a_cal1d, hyb_a_cal2d,
                       hyb_b_enabled, hyb_b_nlog, hyb_b_ncal, hyb_b_log1d, hyb_b_log2d,
                       hyb_b_cal1d, hyb_b_cal2d,
                       ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d,
                       ep_a_cal1d, ep_a_cal2d,
                       ep_b_enabled, ep_b_nlog, ep_b_ncal, ep_b_log1d, ep_b_log2d,
                       ep_b_cal1d, ep_b_cal2d,
                       mode,
                       wd, end_yr, target_yr, disp,
                       toggles, legend_pos, chart_layout, display_q, lots_data,
                       mc_enable, mc_bins, mc_regime, mc_sims, mc_years, mc_window,
                       mc_start_yr, mc_entry_q, _mc_loaded, _pay_trigger, model_show, mc_model_src,
                       price_data, mc_cached, pay_token, mc_unblocked, mc_auth, palette_key,
                       qs_mode=None, viewport_width=None, user_model_store=None):
    delays  = [float(x) for x in [d0, d1, d2, d3, d4] if x is not None]
    toggles = toggles or []
    yr_now  = pd.Timestamp.today().year
    _advanced = "advanced" in (qs_mode or [])
    _effective_qs = (adv_qs or []) if _advanced else (
        _bands_to_qs(sel_qs) if sel_qs and isinstance(sel_qs[0], str) else (sel_qs or []))
    mc_ok, is_free, mc_p, blocked = _mc_setup(
        "sc", mc_enable, mc_years, mc_start_yr, mc_entry_q,
        mc_bins, mc_sims, freq, mc_window, wd, infl,
        mc_cached, _cf(price_data, 0), mc_regime, mc_unblocked, pay_token,
        mc_auth=mc_auth,
        stack=stack, amount_default=5000, infl_default=4.0, start_yr_default=2031,
        mc_model_src=mc_model_src)
    # chart_layout is now a checklist list; legacy snapshots may send an int
    _cl = (2 if "shade" in (chart_layout or []) else 0) \
          if isinstance(chart_layout, list) \
          else (int(chart_layout) if chart_layout is not None else 2)
    model_show = model_show if model_show is not None else []
    model_show = _resolve_lppl_master(
        model_show, lppl_n_freqs, lppl_weighted, lppl_no_13)
    model_show = _resolve_hybppl_master(
        model_show,
        hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d, hyb_a_cal1d, hyb_a_cal2d,
        hyb_b_enabled, hyb_b_nlog, hyb_b_ncal, hyb_b_log1d, hyb_b_log2d,
        hyb_b_cal1d, hyb_b_cal2d)
    model_show = _resolve_eppl_master(
        model_show,
        ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d, ep_a_cal1d, ep_a_cal2d,
        ep_b_enabled, ep_b_nlog, ep_b_ncal, ep_b_log1d, ep_b_log2d,
        ep_b_cal1d, ep_b_cal2d)
    fig, mc_result = _get_supercharge_fig(dict(
        mode         = mode or "a",
        start_stack  = _cf(stack, SUPERCHARGE["start_stack"]),
        start_yr     = _ci(start_yr, SUPERCHARGE["start_yr"]),
        delays       = delays if delays else [0, 1, 2, 4, 8],
        freq         = freq or "Monthly",
        inflation    = _cf(infl, SUPERCHARGE["inflation"]),
        selected_qs  = _effective_qs,
        chart_layout = _cl,
        display_q    = _cf(display_q, _nearest_quantile(SUPERCHARGE["display_q"], _app_ctx._ALL_QS)),
        wd_amount    = _ci(wd, SUPERCHARGE["wd_amount"], lo=0, hi=_app_ctx.MAX_USD),
        end_yr       = _ci(end_yr, SUPERCHARGE["end_yr"]),
        disp_mode    = disp or "usd",
        log_y        = "log_y"      in toggles,
        annotate     = "annotate"   in toggles,
        discrete     = "discrete"   in toggles,
        show_legend  = "show_legend" in toggles,
        legend_pos   = legend_pos or "outside",
        minor_grid   = "minor_grid" in toggles,
        target_yr    = _ci(target_yr, SUPERCHARGE["target_yr"]),
        lots         = lots_data or [],
        use_lots     = bool(use_lots),
        show_qr      = "bub" in model_show,
        show_mc      = "mc" in model_show,
        active_models = [k for k in model_show if k != "mc"],  # pass "bub" through for toggle
        palette = palette_key or "default",
        is_mobile = (viewport_width or 1200) < 768,
        user_model = user_model_store,
        **mc_p,
    ))
    fig, store_val, status, rendered_key, show_modal, ub_val = _mc_finalize(
        "sc", fig, mc_result, mc_cached, mc_enable, mc_ok,
        is_free, blocked, mc_p["mc_years"], mc_p["mc_start_yr"],
        mc_p["mc_entry_q"], toggles,
        mc_stale=mc_p.get("mc_stale", False),
        mc_p=mc_p)
    yr_adjust = dash.no_update
    if mc_ok and mc_p.get("mc_start_yr"):
        mc_sy = int(mc_p["mc_start_yr"])
        if mc_sy < int(start_yr or 2033):
            yr_adjust = mc_sy
    return (fig, store_val, status, rendered_key, show_modal,
            "sc" if show_modal else dash.no_update, ub_val, yr_adjust)


# ── Model warning modals (S2F, Exponential) ──────────────────────────────────

@callback(
    Output("s2f-warn-dialog", "displayed"),
    Output("exp-warn-dialog", "displayed"),
    Output("u1-warn-dialog", "displayed"),
    Output("model-warn-dismissed", "data", allow_duplicate=True),
    Input("bub-model-show", "value"),
    Input("dca-model-show", "value"),
    Input("ret-model-show", "value"),
    Input("sc-model-show", "value"),
    State("model-warn-dismissed", "data"),
    prevent_initial_call=True,
)
def model_warnings(bub_models, dca_models, ret_models, sc_models, dismissed):
    dismissed = dismissed or {}
    all_active = set((bub_models or []) + (dca_models or []) + (ret_models or []) + (sc_models or []))
    show_s2f = "s2f" in all_active and not dismissed.get("s2f")
    show_exp = "exp" in all_active and not dismissed.get("exp")
    show_u1 = "u1" in all_active and not dismissed.get("u1")
    if show_s2f:
        dismissed["s2f"] = True
    if show_exp:
        dismissed["exp"] = True
    if show_u1:
        dismissed["u1"] = True
    if show_s2f or show_exp or show_u1:
        return show_s2f, show_exp, show_u1, dismissed
    return False, False, False, dash.no_update


# LP4 warning — fires EVERY time 4 is added to lppl-n-freqs (no dismissal)
_app_ctx.app.clientside_callback(
    """
    function(current, previous) {
        var cur = current || [];
        var prev = previous || [];
        // Fire only on transition: 4 not in prev but 4 in cur
        var had4 = prev.indexOf(4) !== -1;
        var has4 = cur.indexOf(4) !== -1;
        if (!had4 && has4) {
            return [true, cur];
        }
        return [window.dash_clientside.no_update, cur];
    }
    """,
    Output("lp4-warn-dialog", "displayed"),
    Output("lp4-warn-prev", "data"),
    Input("lppl-n-freqs", "value"),
    State("lp4-warn-prev", "data"),
    prevent_initial_call=True,
)


# ── Update Display Models swatches when palette changes ──────────────────────

@callback(
    Output("bub-model-show", "options"),
    Output("dca-model-show", "options", allow_duplicate=True),
    Output("ret-model-show", "options", allow_duplicate=True),
    Output("sc-model-show", "options", allow_duplicate=True),
    Input("palette-store", "data"),
    State("display-model-summaries", "data"),
    prevent_initial_call=True,
)
def update_model_swatches(palette_key, summaries):
    from layout.display_models import build_display_models_options
    pal = _app_ctx.PALETTES.get(palette_key or "default",
                                 _app_ctx.PALETTES["default"])
    mc = pal.get("model_colors", _app_ctx.MODEL_TRACE_COLORS)
    return (
        build_display_models_options(mc, "bub", include_bm_master=True, summaries=summaries),
        build_display_models_options(mc, "dca", summaries=summaries),
        build_display_models_options(mc, "ret", include_mc=True, summaries=summaries),
        build_display_models_options(mc, "sc",  summaries=summaries),
    )


# Heatmap pill swatches — update children (swatch + label) on palette change.
# Per-pill Output to avoid rebuilding the pill bar container, which would
# invalidate _hm_pill_click / _hm_pill_sync bindings.
from callbacks.routing import _HM_PILL_MODELS  # noqa: E402

_HM_PILL_LABELS = {
    "bub": "BM", "pl": "PL", "lppl": "LPPL",
    "linppl": "LinPPL", "hybppl": "HybPPL",
    "hyb2l": "H2L", "hyb2c": "H2C", "hyb2b": "H2B", "hyb4d": "H4D",
    "pca": "PCA", "grdy": "Grdy", "eppl": "EPPL",
    "gomp": "Gomp", "bpl": "BPL",
    "ef": "EF", "u1": "U\u2081", "mc": "MC",
}


def _hm_pill_label_html(key, mc):
    from dash import html
    return html.Span([
        html.Span(" ", style={
            "display": "inline-block", "width": "8px", "height": "8px",
            "borderRadius": "2px", "verticalAlign": "middle",
            "marginRight": "4px",
            "backgroundColor": mc.get(key, FALLBACK_MODEL_GRAY),
        }),
        _HM_PILL_LABELS.get(key, key),
    ])


@callback(
    *[Output(f"hm-pill-{k}", "children") for k in _HM_PILL_MODELS],
    Input("palette-store", "data"),
    prevent_initial_call=True,
)
def update_heatmap_pill_swatches(palette_key):
    pal = _app_ctx.PALETTES.get(palette_key or "default",
                                 _app_ctx.PALETTES["default"])
    mc = pal.get("model_colors", _app_ctx.MODEL_TRACE_COLORS)
    children = []
    for k in _HM_PILL_MODELS:
        if k == "mc":
            # MC has no swatch (warning color, text only)
            children.append("MC")
        else:
            children.append(_hm_pill_label_html(k, mc))
    return tuple(children)


# ── Quantile mode toggle (default ↔ advanced) — all tabs ─────────────────────

def _register_qs_mode_callbacks(prefix):
    """Register mode toggle + band limit callbacks for one tab's quantile panel."""

    @callback(
        Output(f"{prefix}-qs-default-wrap", "style"),
        Output(f"{prefix}-qs-advanced-wrap", "style"),
        Output(f"{prefix}-qs", "value", allow_duplicate=True),
        Output(f"{prefix}-qs-adv", "value", allow_duplicate=True),
        Input(f"{prefix}-qs-mode", "value"),
        State(f"{prefix}-qs", "value"),
        State(f"{prefix}-qs-adv", "value"),
        prevent_initial_call=True,
    )
    def toggle_mode(mode, default_vals, adv_vals):
        from layout.common import _DEFAULT_BANDS
        is_advanced = "advanced" in (mode or [])
        if is_advanced:
            # Expand band names to quantile floats for advanced checklist
            expanded = _bands_to_qs(default_vals) if default_vals else []
            return ({"display": "none"}, {}, dash.no_update, expanded)
        else:
            # Convert quantile floats back to band names
            adv_set = set(adv_vals or [])
            bands = []
            for b in _DEFAULT_BANDS:
                if any(q in adv_set for q in b["qs"]):
                    bands.append(b["value"])
            return ({}, {"display": "none"}, bands or ["median"], dash.no_update)



for _prefix in ("bub", "dca", "ret", "sc"):
    _register_qs_mode_callbacks(_prefix)
