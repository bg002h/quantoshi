"""Tab routing, URL-based navigation, accordion deep-linking, pill bars."""

import dash
from dash import Input, Output, State, callback, ctx, no_update, ALL

import _app_ctx
from layout.faq import _FAQ


def _norm(pathname: str | None) -> str | None:
    """Normalize pathname: treat '-' as '.' so /1-2-5-1 == /1.2.5.1."""
    if pathname and "-" in pathname:
        return pathname.replace("-", ".")
    return pathname


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


# After snapshot restore, force the active tab's chart to re-render.
# Mobile Safari/WebKit browsers sometimes swallow the Input-change events
# fired by apply_snapshot (~100 simultaneous allow_duplicate outputs),
# leaving the pre-injected figure stale. Incrementing first-render on
# the active tab guarantees the chart callback fires with the restored
# control values.
_app_ctx.app.clientside_callback(
    """
    function(state, tab, bub, hm, dca, ret, sc, cp) {
        var NU = window.dash_clientside.no_update;
        var out = [NU, NU, NU, NU, NU, NU];
        if (!state) return out;
        var map = {bubble:0, heatmap:1, dca:2, retire:3, supercharge:4, citadel:5};
        var idx = map[tab];
        if (idx === undefined) return out;
        var cur = [bub, hm, dca, ret, sc, cp][idx] || 0;
        out[idx] = cur + 1;
        return out;
    }
    """,
    Output("bubble-first-render", "data", allow_duplicate=True),
    Output("heatmap-first-render", "data", allow_duplicate=True),
    Output("dca-first-render", "data", allow_duplicate=True),
    Output("retire-first-render", "data", allow_duplicate=True),
    Output("supercharge-first-render", "data", allow_duplicate=True),
    Output("citadel-first-render", "data", allow_duplicate=True),
    Input("snapshot-state-store", "data"),
    State("main-tabs", "active_tab"),
    State("bubble-first-render", "data"),
    State("heatmap-first-render", "data"),
    State("dca-first-render", "data"),
    State("retire-first-render", "data"),
    State("supercharge-first-render", "data"),
    State("citadel-first-render", "data"),
    prevent_initial_call=True,
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
                    "bub-lppl-activate","lppl-n-freqs","lppl-weighted","lppl-no-13",
                    "scan-price","scan-date","scan-q"},
    "heatmap":     {"hm-entry-yr","hm-entry-q","hm-exit-range","hm-exit-qs","hm-mode",
                    "hm-b1","hm-b2","hm-c-lo","hm-c-mid1","hm-c-mid2","hm-c-hi",
                    "hm-grad","hm-vfmt","hm-cell-fs","hm-toggles","hm-stack","hm-use-lots",
                    "hm-model-show","hm-mc-model-src","hm-active-model",
                    "hm-lppl-activate","lppl-n-freqs","lppl-weighted","lppl-no-13"},
    "dca":         {"dca-stack","dca-use-lots","dca-amount","dca-freq","dca-freq-unlock",
                    "dca-infl","dca-yr-range",
                    "dca-disp","dca-toggles","dca-qs","dca-qs-mode","dca-qs-adv",
                    "dca-sc-enable","dca-sc-loan","dca-sc-rate","dca-sc-term",
                    "dca-sc-type","dca-sc-repeats",
                    "dca-sc-entry-mode","dca-sc-custom-price","dca-sc-tax",
                    "dca-sc-rollover","dca-legend-pos","dca-model-show","dca-mc-model-src",
                    "dca-lppl-activate","lppl-n-freqs","lppl-weighted","lppl-no-13"},
    "retire":      {"ret-stack","ret-use-lots","ret-wd","ret-freq","ret-freq-unlock",
                    "ret-yr-range",
                    "ret-infl","ret-disp","ret-toggles","ret-legend-pos","ret-qs","ret-qs-mode","ret-qs-adv",
                    "ret-model-show","ret-mc-model-src",
                    "ret-lppl-activate","lppl-n-freqs","lppl-weighted","lppl-no-13"},
    "supercharge": {"sc-stack","sc-use-lots","sc-start-yr","sc-d0","sc-d1","sc-d2",
                    "sc-d3","sc-d4","sc-freq","sc-freq-unlock","sc-infl","sc-qs","sc-qs-mode","sc-qs-adv",
                    "sc-mode","sc-wd",
                    "sc-end-yr","sc-target-yr","sc-disp","sc-toggles","sc-legend-pos",
                    "sc-chart-layout","sc-display-q","sc-model-show","sc-mc-model-src",
                    "sc-lppl-activate","lppl-n-freqs","lppl-weighted","lppl-no-13"},
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
        /* Normalize: treat '-' as '.' so /1-2-5-1 == /1.2.5.1 */
        if (pathname && pathname.indexOf('-') !== -1) {
            pathname = pathname.replace(/-/g, '.');
        }
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
        if (p && p.indexOf("/1.2") === 0) { return "bubble"; }
        if (p && p.indexOf("/1.3") === 0) { return "bubble"; }
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
# Bubble sub-view deep-linking (/1.2 → Forward CAGR, /1.3 → Residuals)
# Single callback to avoid "Duplicate callback outputs" when both fire on URL change.
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("bub-view-mode", "data", allow_duplicate=True),
    Output("bub-price-wrap", "style", allow_duplicate=True),
    Output("bub-cagr-wrap", "style", allow_duplicate=True),
    Output("bub-resid-wrap", "style", allow_duplicate=True),
    Output("bub-view-price", "outline", allow_duplicate=True),
    Output("bub-view-cagr", "outline", allow_duplicate=True),
    Output("bub-view-resid", "outline", allow_duplicate=True),
    Output("bub-scale-controls", "style", allow_duplicate=True),
    Output("bub-bubble-panel", "style", allow_duplicate=True),
    Output("bub-cagr-fwd-wrap", "style", allow_duplicate=True),
    Output("bub-xrange", "value", allow_duplicate=True),
    Output("bub-cagr-fwd-yrs", "value", allow_duplicate=True),
    Output("bub-cagr-hover-today", "data", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call=True,
)
def deep_link_bub_view(pathname):
    from dash import no_update
    pathname = _norm(pathname)
    if not pathname:
        return (no_update,) * 13
    _hide = {"display": "none"}

    if pathname.startswith("/1.3"):
        # Residuals view
        return ("resid", _hide, _hide, {},
                True, True, False,
                {}, {}, _hide,
                no_update, no_update, no_update)

    if pathname.startswith("/1.2"):
        # Forward CAGR view, parse /1.2.N.B
        _FWD_OPTIONS = [1, 2, 4, 10, 20, 30]
        fwd_yrs = no_update
        hover_today = no_update
        parts = pathname[1:].split(".")
        if len(parts) >= 3:
            try:
                n = int(parts[2])
                if 1 <= n <= len(_FWD_OPTIONS):
                    fwd_yrs = _FWD_OPTIONS[n - 1]
            except ValueError:
                pass
        if len(parts) >= 4:
            try:
                b = int(parts[3])
                if b == 1:
                    hover_today = True
            except ValueError:
                pass
        return ("cagr", _hide, {}, _hide,
                True, False, True,
                _hide, _hide, {"display": "inline"},
                [2025, 2050], fwd_yrs, hover_today)

    return (no_update,) * 13


# ══════════════════════════════════════════════════════════════════════════════
# CAGR progress bar — show on input change, hide on figure arrival
# ══════════════════════════════════════════════════════════════════════════════

_app_ctx.app.clientside_callback(
    """
    function(fwd_yrs, view_mode) {
        if (view_mode !== 'cagr') return window.dash_clientside.no_update;
        var wrap = document.getElementById('bub-cagr-progress-wrap');
        var bar = document.getElementById('bub-cagr-progress-bar');
        if (!wrap || !bar) return window.dash_clientside.no_update;
        // Stop any previous animation
        if (window._cagrProgressTimer) clearInterval(window._cagrProgressTimer);
        // Show overlay and reset bar
        wrap.style.display = 'block';
        bar.style.transition = 'none';
        bar.style.width = '0%';
        // Estimate duration: ~0.5s per forward year
        var yrs = parseInt(fwd_yrs) || 1;
        var estMs = Math.max(2000, yrs * 1500);
        var startTime = Date.now();
        window._cagrProgressTimer = setInterval(function() {
            var elapsed = Date.now() - startTime;
            // Ease out — slows down as it approaches 95%
            var pct = Math.min(95, 95 * (1 - Math.exp(-3 * elapsed / estMs)));
            bar.style.transition = 'width 0.2s linear';
            bar.style.width = pct + '%';
            if (pct >= 94.5) clearInterval(window._cagrProgressTimer);
        }, 100);
        return window.dash_clientside.no_update;
    }
    """,
    Output("bub-cagr-loading", "data"),
    Input("bub-cagr-fwd-yrs", "value"),
    Input("bub-view-mode", "data"),
    prevent_initial_call=True,
)

_app_ctx.app.clientside_callback(
    """
    function(fig) {
        if (window._cagrProgressTimer) {
            clearInterval(window._cagrProgressTimer);
            window._cagrProgressTimer = null;
        }
        var wrap = document.getElementById('bub-cagr-progress-wrap');
        var bar = document.getElementById('bub-cagr-progress-bar');
        if (wrap && bar) {
            bar.style.transition = 'width 0.2s linear';
            bar.style.width = '100%';
            setTimeout(function() { wrap.style.display = 'none'; }, 300);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("bub-cagr-loading", "data", allow_duplicate=True),
    Input("bub-cagr-graph", "figure"),
    prevent_initial_call=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# CAGR today-hover deep-link trigger (/1.2.N.1)
# ══════════════════════════════════════════════════════════════════════════════

_app_ctx.app.clientside_callback(
    """
    function(flag) {
        if (!flag) return window.dash_clientside.no_update;
        var attempts = 0;
        function tryHover() {
            var wrap = document.getElementById('bub-cagr-graph');
            var el = wrap && wrap.querySelector('.js-plotly-plot');
            if (!el || !el.data || el.data.length === 0) {
                if (++attempts < 50) { setTimeout(tryHover, 200); }
                return;
            }
            // Find the beacon dot trace (single-point marker, last one)
            for (var i = el.data.length - 1; i >= 0; i--) {
                var tr = el.data[i];
                if (tr.x && tr.x.length === 1 && tr.mode === 'markers') {
                    Plotly.Fx.hover(el, [{curveNumber: i, pointNumber: 0}]);
                    break;
                }
            }
        }
        setTimeout(tryHover, 1000);
        return false;
    }
    """,
    Output("bub-cagr-hover-today", "data", allow_duplicate=True),
    Input("bub-cagr-hover-today", "data"),
    prevent_initial_call=True,
)


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
    pathname = _norm(pathname)
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
    pathname = _norm(pathname)
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
# Standardized pill set (Phase 2). Tab 2 deep-link routes /2.N are renumbered;
# old URLs will land on different models — accepted per design decision.
# /2.1=bub, /2.2=pl, /2.3=lppl (master), /2.4=linppl, /2.5=hybppl,
# /2.6=ef (if loaded), /2.7=u1 (if loaded), /2.N+1=mc (if HAS_MARKOV)
_HM_PILL_MODELS = ["bub", "pl", "lppl", "linppl", "hybppl"]
if "ef" in _app_ctx.PRICE_MODELS:
    _HM_PILL_MODELS.append("ef")
if "u1" in _app_ctx.PRICE_MODELS:
    _HM_PILL_MODELS.append("u1")
if _app_ctx._HAS_MARKOV:
    _HM_PILL_MODELS.append("mc")

# Map removed pill IDs (Phase 1 share links may have these in hm-active-model)
# → surviving pill. Used as a graceful fallback when old snapshot decodes.
_HM_LEGACY_MODEL_FALLBACK = {
    "qr": "pl",        # QR was bands-only; PL is the closest match
    "lp2": "lppl", "lp3": "lppl", "lp4": "lppl",
    "lppl_w": "lppl", "lp2_w": "lppl", "lp3_w": "lppl", "lp4_w": "lppl",
    "lp4_n13": "lppl", "lp4_w_n13": "lppl",
    "exp": "bub",      # display-only demo
    "s2f": "bub",      # display-only demo
}


@callback(
    Output("hm-active-model", "data", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call=True,
)
def _hm_deep_link(pathname):
    pathname = _norm(pathname)
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

# Pill IDs match the Phase 2 standardized _HM_PILL_MODELS list above.
_HM_PILL_IDS = [f"hm-pill-{k}" for k in _HM_PILL_MODELS]


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
    # Legacy snapshot values (qr, lp2, exp, etc) → surviving pill
    if model_key not in _HM_PILL_MODELS:
        model_key = _HM_LEGACY_MODEL_FALLBACK.get(model_key, "bub")
    active_id = f"hm-pill-{model_key}"
    return tuple(pid != active_id for pid in _HM_PILL_IDS)
