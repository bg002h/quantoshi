"""Clientside callbacks for chart-related controls.

Registers ~30 clientside callbacks at import time (side effect). Covers:
  * Display Models family summaries — single multi-output callback that
    writes LPPL/HybPPL/EPPL summary strings to 12 tab-specific spans plus
    the `display-model-summaries` Store.
  * Heatmap status row visibility, label, summary text, gear routing.
  * Model-info-triangle links — close the active config modal.
  * BM / LPPL / HybPPL / EPPL modals — open via gear, close via close button.
  * HybPPL / EPPL damping visibility toggles + status text.

Extracted from ``callbacks/charts.py`` to keep the main chart-callback file
focused on the Python @callback functions.
"""

from dash import Input, Output, State
import _app_ctx
from colors import LINK  # noqa: F401 — referenced by the JS summary callback


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
