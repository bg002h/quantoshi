"""Tab routing, URL-based navigation, accordion deep-linking, pill bars."""

import dash
from dash import Input, Output, State, callback, ctx, no_update, ALL

import _app_ctx
from layout.faq import _FAQ


# ══════════════════════════════════════════════════════════════════════════════
# SC mode / display-q toggles
# ══════════════════════════════════════════════════════════════════════════════

_app_ctx.app.clientside_callback(
    "function(m) { return [m==='a', m==='b', m==='a']; }",
    Output("sc-mode-a-collapse", "is_open"),
    Output("sc-mode-b-collapse", "is_open"),
    Output("sc-depl-note-collapse", "is_open"),
    Input("sc-mode", "value"),
)

_app_ctx.app.clientside_callback(
    "function(v) { return !(v && v.indexOf('shade') !== -1); }",
    Output("sc-display-q-collapse", "is_open"),
    Input("sc-chart-layout", "value"),
)


# ══════════════════════════════════════════════════════════════════════════════
# Per-tab render triggers — fires the matching tab's store on each tab switch
# ══════════════════════════════════════════════════════════════════════════════

_app_ctx.app.clientside_callback(
    """
    function(tab, bub, hm, dca, ret, sc, cp) {
        var NU = window.dash_clientside.no_update;
        var out = [NU, NU, NU, NU, NU, NU];
        var map = {bubble:0, heatmap:1, dca:2, retire:3, supercharge:4, citadel:5};
        var idx = map[tab];
        if (idx !== undefined) {
            var cur = [bub, hm, dca, ret, sc, cp][idx];
            if (!cur) out[idx] = 1;
        }
        return out;
    }
    """,
    Output("bubble-first-render", "data", allow_duplicate=True),
    Output("heatmap-first-render", "data", allow_duplicate=True),
    Output("dca-first-render", "data", allow_duplicate=True),
    Output("retire-first-render", "data", allow_duplicate=True),
    Output("supercharge-first-render", "data", allow_duplicate=True),
    Output("citadel-first-render", "data", allow_duplicate=True),
    Input("main-tabs", "active_tab"),
    State("bubble-first-render", "data"),
    State("heatmap-first-render", "data"),
    State("dca-first-render", "data"),
    State("retire-first-render", "data"),
    State("supercharge-first-render", "data"),
    State("citadel-first-render", "data"),
    prevent_initial_call='initial_duplicate',
)


# ══════════════════════════════════════════════════════════════════════════════
# Path ↔ tab mappings and per-tab control IDs (used by snapshot)
# ══════════════════════════════════════════════════════════════════════════════

_PATH_TO_TAB = {
    "/1": "bubble", "/2": "heatmap", "/3": "dca",
    "/4": "retire",  "/5": "supercharge", "/6": "stack",
    "/7": "model_info", "/8": "faq", "/9": "citadel",
}
_TAB_TO_PATH = {v: k for k, v in _PATH_TO_TAB.items()}

# Component IDs that belong to each tab (for single-tab share links)
_TAB_CONTROLS = {
    "bubble":      {"bub-qs","bub-qs-mode","bub-qs-adv",
                    "bub-xscale","bub-yscale","bub-xrange","bub-yrange",
                    "bub-toggles","bub-bubble-toggles","bub-n-future","bub-ptsize",
                    "bub-ptalpha","bub-stack","bub-show-stack","bub-use-lots","bub-auto-y",
                    "bub-legend-pos","bub-model-show",
                    "scan-price","scan-date","scan-q"},
    "heatmap":     {"hm-entry-yr","hm-entry-q","hm-exit-range","hm-exit-qs","hm-mode",
                    "hm-b1","hm-b2","hm-c-lo","hm-c-mid1","hm-c-mid2","hm-c-hi",
                    "hm-grad","hm-vfmt","hm-cell-fs","hm-toggles","hm-stack","hm-use-lots",
                    "hm-model-show","hm-mc-model-src","hm-active-model"},
    "dca":         {"dca-stack","dca-use-lots","dca-amount","dca-freq","dca-freq-unlock",
                    "dca-infl","dca-yr-range",
                    "dca-disp","dca-toggles","dca-qs","dca-qs-mode","dca-qs-adv",
                    "dca-sc-enable","dca-sc-loan","dca-sc-rate","dca-sc-term",
                    "dca-sc-type","dca-sc-repeats",
                    "dca-sc-entry-mode","dca-sc-custom-price","dca-sc-tax",
                    "dca-sc-rollover","dca-legend-pos","dca-model-show","dca-mc-model-src"},
    "retire":      {"ret-stack","ret-use-lots","ret-wd","ret-freq","ret-freq-unlock",
                    "ret-yr-range",
                    "ret-infl","ret-disp","ret-toggles","ret-legend-pos","ret-qs","ret-qs-mode","ret-qs-adv",
                    "ret-model-show","ret-mc-model-src"},
    "supercharge": {"sc-stack","sc-use-lots","sc-start-yr","sc-d0","sc-d1","sc-d2",
                    "sc-d3","sc-d4","sc-freq","sc-freq-unlock","sc-infl","sc-qs","sc-qs-mode","sc-qs-adv",
                    "sc-mode","sc-wd",
                    "sc-end-yr","sc-target-yr","sc-disp","sc-toggles","sc-legend-pos",
                    "sc-chart-layout","sc-display-q","sc-model-show","sc-mc-model-src"},
    "stack":       set(),
    "model_info":  set(),
    "citadel":     {"cp-stack","cp-use-lots","cp-cash-init","cp-cash-rate",
                    "cp-res-short-init","cp-res-short-rate","cp-res-short-vol",
                    "cp-res-med-init","cp-res-med-rate","cp-res-med-vol",
                    "cp-res-long-init","cp-res-long-rate","cp-res-long-vol",
                    "cp-inv-eq-init","cp-inv-eq-rate","cp-inv-eq-basis","cp-inv-eq-vol",
                    "cp-inv-bd-init","cp-inv-bd-rate","cp-inv-bd-basis","cp-inv-bd-vol",
                    "cp-spend","cp-infl","cp-spend-growth",
                    "cp-high-q-enable","cp-high-q-thresh","cp-high-q-mode","cp-high-q-rate","cp-high-q-dur",
                    "cp-high-q-split-cash","cp-high-q-split-rs","cp-high-q-split-rm",
                    "cp-high-q-split-rl","cp-high-q-split-eq","cp-high-q-split-bd",
                    "cp-low-q-enable","cp-low-q-thresh","cp-low-q-mode","cp-low-q-rate","cp-low-q-dur",
                    "cp-low-q-split-cash","cp-low-q-split-rs","cp-low-q-split-rm",
                    "cp-low-q-split-rl","cp-low-q-split-eq","cp-low-q-split-bd",
                    "cp-lump-cooldown","cp-cash-floor",
                    "cp-res-short-floor","cp-res-med-floor","cp-res-long-floor",
                    "cp-scf-enable","cp-scf-amount","cp-scf-type","cp-scf-rate",
                    "cp-scf-term","cp-scf-trigger",
                    "cp-yr-range","cp-freq","cp-model-src","cp-qs","cp-disp",
                    "cp-toggles","cp-legend-pos","cp-asset-model",
                    "cp-cash-floor-growth","cp-res-floor-growth",
                    "cp-scenario-wealth","cp-scenario-regime","cp-scenario-rules",
                    "cp-scenario-start-yr","cp-scenario-active"},
    "faq":         set(),
}
# Palette is global -- add to every tab so single-tab share links include it
for _tab_set in _TAB_CONTROLS.values():
    _tab_set.add("palette-store")

# MC controls per tab (for single-tab share links)
_TAB_CONTROLS["dca"].update({
    "dca-mc-enable", "dca-mc-start-yr", "dca-mc-entry-q", "dca-mc-years",
    "dca-mc-bins", "dca-mc-regime", "dca-mc-sims", "dca-mc-window", "dca-mc-advanced",
})
_TAB_CONTROLS["retire"].update({
    "ret-mc-enable", "ret-mc-start-yr", "ret-mc-entry-q", "ret-mc-years",
    "ret-mc-bins", "ret-mc-regime", "ret-mc-sims", "ret-mc-window", "ret-mc-advanced",
})
_TAB_CONTROLS["heatmap"].update({
    "hm-mc-enable", "hm-mc-start-yr", "hm-mc-entry-q", "hm-mc-years",
    "hm-mc-bins", "hm-mc-regime", "hm-mc-sims", "hm-mc-window", "hm-mc-advanced",
    "hm-palette",
})
_TAB_CONTROLS["supercharge"].update({
    "sc-mc-enable", "sc-mc-start-yr", "sc-mc-entry-q", "sc-mc-years",
    "sc-mc-bins", "sc-mc-regime", "sc-mc-sims", "sc-mc-window", "sc-mc-advanced",
})
_TAB_CONTROLS["citadel"].update({
    "cp-mc-enable", "cp-mc-start-yr", "cp-mc-entry-q", "cp-mc-years",
    "cp-mc-bins", "cp-mc-regime", "cp-mc-sims", "cp-mc-window",
    "cp-mc-advanced", "cp-mc-model-src",
})
_TAB_CONTROLS["citadel"].update({
    "cp-tax-toggle", "cp-tax-config",
    "cp-td-btc", "cp-td-cash", "cp-td-res-short", "cp-td-res-med", "cp-td-res-long",
    "cp-td-inv-eq", "cp-td-inv-bd",
    "cp-tf-btc", "cp-tf-cash", "cp-tf-res-short", "cp-tf-res-med", "cp-tf-res-long",
    "cp-tf-inv-eq", "cp-tf-inv-bd",
})


# ══════════════════════════════════════════════════════════════════════════════
# Callback -- pathname-based tab routing (/1 ... /9)
# ══════════════════════════════════════════════════════════════════════════════

_app_ctx.app.clientside_callback(
    """
    function(pathname, splashOpen) {
        var NU = window.dash_clientside.no_update;
        var map = {"/1":"bubble","/2":"heatmap","/3":"dca",
                   "/4":"retire","/5":"supercharge","/6":"stack",
                   "/7":"model_info","/8":"faq","/9":"citadel"};
        /* While splash modal is open, defer the tab switch so chart
           callbacks don't fire into a container hidden behind the modal. */
        if (splashOpen) {
            window._pendingTabPath = pathname;
            return NU;
        }
        var p = window._pendingTabPath || pathname;
        if (window._pendingTabPath) {
            /* Splash just closed -- force Plotly resize after chart renders
               to handle rapid-dismiss edge case where layout hasn't settled. */
            setTimeout(function() {
                window.dispatchEvent(new Event("resize"));
            }, 1200);
        }
        window._pendingTabPath = null;
        if (p && /^\\/7\\.\\d+$/.test(p)) { return "model_info"; }
        if (p && /^\\/8\\.\\d+$/.test(p)) { return "faq"; }
        if (p === "/1.2") { return "bubble"; }
        if (p && /^\\/2\\.\\d+$/.test(p)) { return "heatmap"; }
        var tab = map[p];
        return tab ? tab : NU;
    }
    """,
    Output("main-tabs", "active_tab", allow_duplicate=True),
    Input("url", "pathname"),
    Input("splash-modal", "is_open"),
    prevent_initial_call="initial_duplicate",
)


# ══════════════════════════════════════════════════════════════════════════════
# Bubble sub-view deep-linking (/1.2 → Forward CAGR)
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("bub-view-mode", "data", allow_duplicate=True),
    Output("bub-price-wrap", "style", allow_duplicate=True),
    Output("bub-cagr-wrap", "style", allow_duplicate=True),
    Output("bub-view-price", "outline", allow_duplicate=True),
    Output("bub-view-cagr", "outline", allow_duplicate=True),
    Output("bub-scale-controls", "style", allow_duplicate=True),
    Output("bub-bubble-panel", "style", allow_duplicate=True),
    Output("bub-cagr-fwd-wrap", "style", allow_duplicate=True),
    Output("bub-xrange", "value", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call=True,
)
def deep_link_cagr(pathname):
    from dash import no_update
    if pathname != "/1.2":
        return (no_update,) * 9
    _hide = {"display": "none"}
    return ("cagr", _hide, {}, True, False, _hide, _hide,
            {"display": "inline"}, [2025, 2050])


# ══════════════════════════════════════════════════════════════════════════════
# Model Info accordion deep-linking (/7.N)
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_INFO_ITEMS = ["mi-bub", "mi-qr", "mi-pl", "mi-lppl", "mi-exp", "mi-s2f", "mi-mc", "mi-compare"]


@callback(
    Output("model-info-accordion", "active_item"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def open_model_info_item(pathname):
    """Open a specific Model Info accordion item when pathname is /7.N (1-indexed)."""
    if not pathname or not pathname.startswith("/7."):
        return no_update
    try:
        n = int(pathname[3:])
        if 1 <= n <= len(_MODEL_INFO_ITEMS):
            return _MODEL_INFO_ITEMS[n - 1]
    except (ValueError, IndexError):
        pass
    return no_update


@callback(
    Output("mi-lightbox", "is_open"),
    Output("mi-lightbox-img", "src"),
    Input({"type": "mi-img", "src": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def open_model_info_lightbox(n_clicks_list):
    """Open lightbox modal when a Model Info image is clicked."""
    if not any(n_clicks_list):
        return False, ""
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        return True, triggered["src"]
    return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# FAQ accordion deep-linking (/8.N)
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("faq-accordion", "active_item"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def open_faq_item(pathname):
    """Open a specific FAQ accordion item when pathname is /8.N (1-indexed)."""
    if not pathname or not pathname.startswith("/8."):
        return no_update
    try:
        n = int(pathname[3:])
        if 1 <= n <= len(_FAQ):
            return f"faq-{n - 1}"
    except (ValueError, IndexError):
        pass
    return no_update


# Scroll opened FAQ item to top of screen
_app_ctx.app.clientside_callback(
    """
    function(activeItem) {
        if (!activeItem) return window.dash_clientside.no_update;
        setTimeout(function() {
            var acc = document.getElementById('faq-accordion');
            if (!acc) return;
            var items = acc.querySelectorAll('.accordion-item');
            var idx = parseInt((activeItem || '').replace('faq-', ''), 10);
            if (!isNaN(idx) && items[idx]) {
                var y = items[idx].getBoundingClientRect().top + window.pageYOffset - 60;
                window.scrollTo({top: y, behavior: 'smooth'});
            }
        }, 300);
        return window.dash_clientside.no_update;
    }
    """,
    Output("faq-accordion", "className", allow_duplicate=True),
    Input("faq-accordion", "active_item"),
    prevent_initial_call='initial_duplicate',
)


# ══════════════════════════════════════════════════════════════════════════════
# Heatmap deep-linking (/2.N → select Nth pill, 1-indexed)
# Must match pill bar order in layout/heatmap.py: BM, then all models except bub/mc, then MC
_HM_PILL_MODELS = ["bub"] + [k for k in _app_ctx.PRICE_MODELS if k not in ("bub", "mc")]
if _app_ctx._HAS_MARKOV:
    _HM_PILL_MODELS.append("mc")
# Log the mapping for CLAUDE.md reference
# /2.1=bub, /2.2=qr, /2.3=pl, /2.4=lppl, /2.5=exp, /2.6=ef (if loaded), /2.7=s2f, /2.N+1=mc


@callback(
    Output("hm-active-model", "data", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call=True,
)
def _hm_deep_link(pathname):
    if not pathname or not pathname.startswith("/2."):
        return no_update
    try:
        n = int(pathname[3:])
        if 1 <= n <= len(_HM_PILL_MODELS):
            return _HM_PILL_MODELS[n - 1]
    except (ValueError, IndexError):
        pass
    return no_update


# Heatmap model pill bar -- click to select active model
# ══════════════════════════════════════════════════════════════════════════════

_HM_PILL_IDS = ["hm-pill-bub"] + [f"hm-pill-{k}" for k in _app_ctx.PRICE_MODELS if k != "bub"]
if _app_ctx._HAS_MARKOV:
    _HM_PILL_IDS.append("hm-pill-mc")


@callback(
    Output("hm-active-model", "data", allow_duplicate=True),
    *[Output(pid, "outline") for pid in _HM_PILL_IDS],
    *[Input(pid, "n_clicks") for pid in _HM_PILL_IDS],
    prevent_initial_call=True,
)
def _hm_pill_click(*args):
    trigger = ctx.triggered_id
    if not trigger:
        raise dash.exceptions.PreventUpdate
    model_key = trigger.replace("hm-pill-", "")
    outlines = [pid != trigger for pid in _HM_PILL_IDS]
    return (model_key, *outlines)


# Sync pill styles on snapshot restore / page load (hm-active-model store update)
@callback(
    *[Output(pid, "outline", allow_duplicate=True) for pid in _HM_PILL_IDS],
    Input("hm-active-model", "data"),
    prevent_initial_call=True,
)
def _hm_pill_sync(model_key):
    model_key = model_key or "bub"
    active_id = f"hm-pill-{model_key}"
    return tuple(pid != active_id for pid in _HM_PILL_IDS)
