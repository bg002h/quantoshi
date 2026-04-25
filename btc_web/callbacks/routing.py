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
    function(tab, bub, hm, dca, ret, sc, cp, lev) {
        var NU = window.dash_clientside.no_update;
        var out = [NU, NU, NU, NU, NU, NU, NU];
        var map = {bubble:0, heatmap:1, dca:2, retire:3, supercharge:4, citadel:5, leverage:6};
        var idx = map[tab];
        if (idx !== undefined) {
            var cur = [bub, hm, dca, ret, sc, cp, lev][idx];
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
    Output("leverage-first-render", "data", allow_duplicate=True),
    Input("main-tabs", "active_tab"),
    State("bubble-first-render", "data"),
    State("heatmap-first-render", "data"),
    State("dca-first-render", "data"),
    State("retire-first-render", "data"),
    State("supercharge-first-render", "data"),
    State("citadel-first-render", "data"),
    State("leverage-first-render", "data"),
    prevent_initial_call='initial_duplicate',
)


# ══════════════════════════════════════════════════════════════════════════════
# Post-snapshot chart re-render bump.
#
# After snapshot restore, force the active tab's chart to re-render. Mobile
# Safari/WebKit browsers sometimes swallow the Input-change events fired by
# apply_globals (~30 simultaneous allow_duplicate outputs), leaving the
# pre-injected figure stale. Incrementing first-render on the active tab
# guarantees the chart callback fires with the restored control values.
#
# INVARIANT: Input MUST remain `snapshot-state-store.data`. If you change it
# (e.g. to `main-tabs.active_tab`), `apply_tab_{active}` in snapshot_cb.py
# will fire BEFORE snapshot-state-store is written and read None as its
# State, silently writing nothing and leaving the active tab stuck at its
# pre-injected defaults. See spec
# docs/superpowers/specs/2026-04-24-drop-all-tabs-snapshot-design.md.
# ══════════════════════════════════════════════════════════════════════════════
_app_ctx.app.clientside_callback(
    """
    function(state, tab, bub, hm, dca, ret, sc, cp, lev) {
        var NU = window.dash_clientside.no_update;
        var out = [NU, NU, NU, NU, NU, NU, NU];
        if (!state) return out;
        if (window.__qsTrace) window.__qsTrace('state-store-written→first-render-bump', tab);
        var map = {bubble:0, heatmap:1, dca:2, retire:3, supercharge:4, citadel:5, leverage:6};
        var idx = map[tab];
        if (idx === undefined) return out;
        var cur = [bub, hm, dca, ret, sc, cp, lev][idx] || 0;
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
    Output("leverage-first-render", "data", allow_duplicate=True),
    Input("snapshot-state-store", "data"),
    State("main-tabs", "active_tab"),
    State("bubble-first-render", "data"),
    State("heatmap-first-render", "data"),
    State("dca-first-render", "data"),
    State("retire-first-render", "data"),
    State("supercharge-first-render", "data"),
    State("citadel-first-render", "data"),
    State("leverage-first-render", "data"),
    prevent_initial_call=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Path ↔ tab mappings and per-tab control IDs (used by snapshot)
# ══════════════════════════════════════════════════════════════════════════════

_PATH_TO_TAB = {
    "/1": "bubble", "/2": "heatmap", "/3": "dca",
    "/4": "retire",  "/5": "supercharge", "/6": "citadel",
    "/7": "leverage", "/8": "stack", "/9": "model_info", "/10": "faq",
    "/leverage": "leverage",
    "/faq": "faq", "/mi": "model_info",
}
_TAB_TO_PATH = {v: k for k, v in _PATH_TO_TAB.items()}

# Component IDs that belong to each tab (for single-tab share links)
_TAB_CONTROLS = {
    "bubble":      {"bub-qs","bub-qs-mode","bub-qs-adv",
                    "bub-xscale","bub-yscale","bub-xrange","bub-yrange",
                    "bub-toggles","bub-bubble-toggles","bub-n-future","bub-ptsize",
                    "bub-ptalpha","bub-stack","bub-show-stack","bub-use-lots","bub-auto-y",
                    "bub-legend-pos","bub-model-show",
                    "lppl-n-freqs","lppl-weighted","lppl-no-13",
                    "hybppl-cfg-a-nlog","hybppl-cfg-a-ncal",
                    "hybppl-cfg-a-log1d","hybppl-cfg-a-log2d",
                    "hybppl-cfg-a-cal1d","hybppl-cfg-a-cal2d",
                    "hybppl-cfg-b-enabled",
                    "hybppl-cfg-b-nlog","hybppl-cfg-b-ncal",
                    "hybppl-cfg-b-log1d","hybppl-cfg-b-log2d",
                    "hybppl-cfg-b-cal1d","hybppl-cfg-b-cal2d",
                    "eppl-cfg-a-nlog","eppl-cfg-a-ncal",
                    "eppl-cfg-a-log1d","eppl-cfg-a-log2d",
                    "eppl-cfg-a-cal1d","eppl-cfg-a-cal2d",
                    "eppl-cfg-b-enabled",
                    "eppl-cfg-b-nlog","eppl-cfg-b-ncal",
                    "eppl-cfg-b-log1d","eppl-cfg-b-log2d",
                    "eppl-cfg-b-cal1d","eppl-cfg-b-cal2d",
                    "scan-price","scan-date","scan-q",
                    "bub-decomp-model","bub-decomp-components","bub-decomp-mode",
                    "bub-decomp-show-formulas","bub-view-mode",
                    "bub-lppl-activate","bub-hybppl-activate","bub-eppl-activate",
                    # Custom Time Axis (Tab 1 only, added 2026-04-13)
                    "cta-active","cta-scale",
                    "cta-t0-cal","cta-t0-cal-custom",
                    "cta-t0-blk","cta-t0-blk-custom",
                    "cta-weighting","cta-models",
                    # Residual QR sigma bands (Tab 1 only, added 2026-04-15)
                    "bub-sigma-mode"},
    "heatmap":     {"hm-entry-yr","hm-entry-q","hm-exit-range","hm-exit-qs","hm-mode",
                    "hm-b1","hm-b2","hm-c-lo","hm-c-mid1","hm-c-mid2","hm-c-hi",
                    "hm-grad","hm-vfmt","hm-cell-fs","hm-toggles","hm-stack","hm-use-lots",
                    "hm-model-show","hm-mc-model-src","hm-active-model",
                    "lppl-n-freqs","lppl-weighted","lppl-no-13",
                    "hybppl-cfg-a-nlog","hybppl-cfg-a-ncal",
                    "hybppl-cfg-a-log1d","hybppl-cfg-a-log2d",
                    "hybppl-cfg-a-cal1d","hybppl-cfg-a-cal2d",
                    "hybppl-cfg-b-enabled",
                    "hybppl-cfg-b-nlog","hybppl-cfg-b-ncal",
                    "hybppl-cfg-b-log1d","hybppl-cfg-b-log2d",
                    "hybppl-cfg-b-cal1d","hybppl-cfg-b-cal2d",
                    "eppl-cfg-a-nlog","eppl-cfg-a-ncal",
                    "eppl-cfg-a-log1d","eppl-cfg-a-log2d",
                    "eppl-cfg-a-cal1d","eppl-cfg-a-cal2d",
                    "eppl-cfg-b-enabled",
                    "eppl-cfg-b-nlog","eppl-cfg-b-ncal",
                    "eppl-cfg-b-log1d","eppl-cfg-b-log2d",
                    "eppl-cfg-b-cal1d","eppl-cfg-b-cal2d",
                    "hm-lppl-activate","hm-hybppl-activate","hm-eppl-activate"},
    "dca":         {"dca-stack","dca-use-lots","dca-amount","dca-freq","dca-freq-unlock",
                    "dca-infl","dca-yr-range",
                    "dca-disp","dca-toggles","dca-qs","dca-qs-mode","dca-qs-adv",
                    "dca-sc-enable","dca-sc-loan","dca-sc-rate","dca-sc-term",
                    "dca-sc-type","dca-sc-repeats",
                    "dca-sc-entry-mode","dca-sc-custom-price","dca-sc-tax",
                    "dca-sc-rollover","dca-legend-pos","dca-model-show","dca-mc-model-src",
                    "lppl-n-freqs","lppl-weighted","lppl-no-13",
                    "hybppl-cfg-a-nlog","hybppl-cfg-a-ncal",
                    "hybppl-cfg-a-log1d","hybppl-cfg-a-log2d",
                    "hybppl-cfg-a-cal1d","hybppl-cfg-a-cal2d",
                    "hybppl-cfg-b-enabled",
                    "hybppl-cfg-b-nlog","hybppl-cfg-b-ncal",
                    "hybppl-cfg-b-log1d","hybppl-cfg-b-log2d",
                    "hybppl-cfg-b-cal1d","hybppl-cfg-b-cal2d",
                    "eppl-cfg-a-nlog","eppl-cfg-a-ncal",
                    "eppl-cfg-a-log1d","eppl-cfg-a-log2d",
                    "eppl-cfg-a-cal1d","eppl-cfg-a-cal2d",
                    "eppl-cfg-b-enabled",
                    "eppl-cfg-b-nlog","eppl-cfg-b-ncal",
                    "eppl-cfg-b-log1d","eppl-cfg-b-log2d",
                    "eppl-cfg-b-cal1d","eppl-cfg-b-cal2d",
                    "dca-lppl-activate","dca-hybppl-activate","dca-eppl-activate"},
    "retire":      {"ret-stack","ret-use-lots","ret-wd","ret-freq","ret-freq-unlock",
                    "ret-yr-range",
                    "ret-infl","ret-disp","ret-toggles","ret-legend-pos","ret-qs","ret-qs-mode","ret-qs-adv",
                    "ret-model-show","ret-mc-model-src",
                    "lppl-n-freqs","lppl-weighted","lppl-no-13",
                    "hybppl-cfg-a-nlog","hybppl-cfg-a-ncal",
                    "hybppl-cfg-a-log1d","hybppl-cfg-a-log2d",
                    "hybppl-cfg-a-cal1d","hybppl-cfg-a-cal2d",
                    "hybppl-cfg-b-enabled",
                    "hybppl-cfg-b-nlog","hybppl-cfg-b-ncal",
                    "hybppl-cfg-b-log1d","hybppl-cfg-b-log2d",
                    "hybppl-cfg-b-cal1d","hybppl-cfg-b-cal2d",
                    "eppl-cfg-a-nlog","eppl-cfg-a-ncal",
                    "eppl-cfg-a-log1d","eppl-cfg-a-log2d",
                    "eppl-cfg-a-cal1d","eppl-cfg-a-cal2d",
                    "eppl-cfg-b-enabled",
                    "eppl-cfg-b-nlog","eppl-cfg-b-ncal",
                    "eppl-cfg-b-log1d","eppl-cfg-b-log2d",
                    "eppl-cfg-b-cal1d","eppl-cfg-b-cal2d",
                    "ret-lppl-activate","ret-hybppl-activate","ret-eppl-activate"},
    "supercharge": {"sc-stack","sc-use-lots","sc-start-yr","sc-d0","sc-d1","sc-d2",
                    "sc-d3","sc-d4","sc-freq","sc-freq-unlock","sc-infl","sc-qs","sc-qs-mode","sc-qs-adv",
                    "sc-mode","sc-wd",
                    "sc-end-yr","sc-target-yr","sc-disp","sc-toggles","sc-legend-pos",
                    "sc-chart-layout","sc-display-q","sc-model-show","sc-mc-model-src",
                    "lppl-n-freqs","lppl-weighted","lppl-no-13",
                    "hybppl-cfg-a-nlog","hybppl-cfg-a-ncal",
                    "hybppl-cfg-a-log1d","hybppl-cfg-a-log2d",
                    "hybppl-cfg-a-cal1d","hybppl-cfg-a-cal2d",
                    "hybppl-cfg-b-enabled",
                    "hybppl-cfg-b-nlog","hybppl-cfg-b-ncal",
                    "hybppl-cfg-b-log1d","hybppl-cfg-b-log2d",
                    "hybppl-cfg-b-cal1d","hybppl-cfg-b-cal2d",
                    "eppl-cfg-a-nlog","eppl-cfg-a-ncal",
                    "eppl-cfg-a-log1d","eppl-cfg-a-log2d",
                    "eppl-cfg-a-cal1d","eppl-cfg-a-cal2d",
                    "eppl-cfg-b-enabled",
                    "eppl-cfg-b-nlog","eppl-cfg-b-ncal",
                    "eppl-cfg-b-log1d","eppl-cfg-b-log2d",
                    "eppl-cfg-b-cal1d","eppl-cfg-b-cal2d",
                    "sc-lppl-activate","sc-hybppl-activate","sc-eppl-activate"},
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
    "leverage":    {"lev-date", "lev-price", "lev-model", "lev-floor-q-store",
                    "lev-rb", "lev-rl", "lev-horizon", "lev-cagr", "lev-toggles"},
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
    function(pathname) {
        var NU = window.dash_clientside.no_update;
        var map = {"/1":"bubble","/2":"heatmap","/3":"dca",
                   "/4":"retire","/5":"supercharge","/6":"citadel",
                   "/7":"leverage","/8":"stack","/9":"model_info","/10":"faq",
                   "/leverage":"leverage",
                   "/faq":"faq","/mi":"model_info"};
        /* Normalize: treat '-' as '.' so /1-2-5-1 == /1.2.5.1 */
        if (pathname && pathname.indexOf('-') !== -1) {
            pathname = pathname.replace(/-/g, '.');
        }
        /* Dedupe: Dash's dcc.Location may re-fire this callback on
           hash/search updates even when pathname hasn't changed; skip
           those to avoid clobbering the user's tab-click active_tab. */
        if (window._routingLastPath === pathname) {
            return NU;
        }
        window._routingLastPath = pathname;
        var p = pathname;
        if (p && /^\\/9\\.\\d+$/.test(p)) { return "model_info"; }
        if (p && /^\\/10\\.\\d+$/.test(p)) { return "faq"; }
        if (p && p.indexOf("/faq.") === 0) { return "faq"; }
        if (p && p.indexOf("/mi.") === 0) { return "model_info"; }
        if (p && p.indexOf("/1.2") === 0) { return "bubble"; }
        if (p && p.indexOf("/1.3") === 0) { return "bubble"; }
        if (p && /^\\/2\\.\\d+$/.test(p)) { return "heatmap"; }
        var tab = map[p];
        return tab ? tab : NU;
    }
    """,
    Output("main-tabs", "active_tab", allow_duplicate=True),
    Input("url", "pathname"),
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
# Model Info accordion deep-linking (/9.N)
# ══════════════════════════════════════════════════════════════════════════════

# Must match the actual accordion ORDER in layout/model_info.
# Single source of truth is layout/model_info/__init__.py::_MODEL_INFO_ITEM_IDS —
# mirrored here verbatim. Keep in sync when adding new accordion items.
_MODEL_INFO_ITEMS = [
    "mi-bub", "mi-qr", "mi-pl", "mi-lppl", "mi-lp2",
    "mi-lppl-weighting", "mi-linppl", "mi-hybppl", "mi-hybppl-dd",
    "mi-hyb2l", "mi-hyb2c", "mi-hyb2b", "mi-hyb4d", "mi-pca",
    "mi-grdy", "mi-eppl", "mi-exp", "mi-gomp", "mi-bpl",
    "mi-plo", "mi-sexp", "mi-logi",
    "mi-s2f", "mi-mc", "mi-ef", "mi-u1",
    "mi-compare", "mi-regimes", "mi-citadel",
]


@callback(
    Output("auto-y-grid", "data"),
    Input("main-tabs", "active_tab"),
    State("auto-y-grid", "data"),
    prevent_initial_call=True,
)
def _lazy_load_auto_y_grid(tab, current):
    """Populate auto-Y grid on first bubble visit (saves ~162KB from layout JSON)."""
    if tab != "bubble" or current is not None:
        return no_update
    return _app_ctx.AUTO_Y_GRID


# ══════════════════════════════════════════════════════════════════════════════
# Universal tab lazy-load: initial_tab is eager-rendered in layout;
# every OTHER tab has a "{tab_id}-lazy" Div placeholder that gets populated
# on first user visit. Keeps the initial /N layout JSON small (~70-80% of
# the pre-lazy ~602KB was non-active tab content).
# ══════════════════════════════════════════════════════════════════════════════

# Module-level caches per tab — _<tab>_tab() is ~5-10ms to build; cache so
# subsequent visits in same worker skip the rebuild.
_TAB_CONTENT_CACHE: dict[str, object] = {}


def _build_tab_content(tab_id):
    if tab_id in _TAB_CONTENT_CACHE:
        return _TAB_CONTENT_CACHE[tab_id]
    if tab_id == "bubble":
        from layout.bubble import _bubble_tab
        content = _bubble_tab()
    elif tab_id == "heatmap":
        from layout.heatmap import _heatmap_tab
        content = _heatmap_tab()
    elif tab_id == "dca":
        from layout.sim_tabs import _dca_tab
        content = _dca_tab()
    elif tab_id == "retire":
        from layout.sim_tabs import _retire_tab
        content = _retire_tab()
    elif tab_id == "supercharge":
        from layout.supercharge import _supercharge_tab
        content = _supercharge_tab()
    elif tab_id == "citadel":
        from layout.citadel import _citadel_tab
        content = _citadel_tab()
    elif tab_id == "leverage":
        from layout.leverage import _leverage_tab
        content = _leverage_tab()
    elif tab_id == "stack":
        from layout.stack import _stack_tracker_tab
        content = _stack_tracker_tab()
    elif tab_id == "model_info":
        from layout.model_info import _model_info_tab
        content = _model_info_tab().children
    elif tab_id == "faq":
        from layout.faq import _faq_tab
        content = _faq_tab()
    else:
        return None
    _TAB_CONTENT_CACHE[tab_id] = content
    return content


def _register_lazy_tab(tab_id):
    """Register a @callback that populates {tab_id}-lazy on first active visit.

    `allow_duplicate=True` is needed because `_prefetch_non_active_tabs`
    below also targets {tab_id}-lazy.children. The two never race: this
    callback fires on `main-tabs.active_tab` change; the prefetch callback
    fires once on `prefetch-interval.n_intervals` change.
    """
    @callback(
        Output(f"{tab_id}-lazy", "children", allow_duplicate=True),
        Input("main-tabs", "active_tab"),
        State(f"{tab_id}-lazy", "children"),
        prevent_initial_call=True,
    )
    def _lazy_load(tab, current, _tid=tab_id):  # default-arg captures tab_id
        if tab != _tid:
            return no_update
        # Skip if already loaded — re-rendering would clobber user state.
        if not _is_loading_placeholder(current):
            return no_update
        content = _build_tab_content(_tid)
        return content if content is not None else no_update


def _is_loading_placeholder(children):
    """True if children is the initial 'Loading...' placeholder Div."""
    if children is None:
        return True
    if isinstance(children, str):
        return children == "Loading..."
    if isinstance(children, dict):
        inner = (children.get("props") or {}).get("children")
        return _is_loading_placeholder(inner)
    if isinstance(children, list) and len(children) == 1:
        return _is_loading_placeholder(children[0])
    return False


for _tid in ("bubble", "heatmap", "dca", "retire", "supercharge",
             "citadel", "leverage", "stack", "model_info", "faq"):
    _register_lazy_tab(_tid)


# ── Staggered background prefetch ──────────────────────────────────────
# One server callback per tab, each triggered by its own `{tab_id}-prefetch-iv`
# Interval (staggered in layout). Active tab is skipped (already populated at
# layout time). Each callback writes only its own lazy-div — no multi-output
# payload, so React reconciles one small chunk at a time with idle gaps
# between for user interaction. Guards: only load when current content is
# still the "Loading..." placeholder (idempotent with click-triggered lazy
# load).
def _register_prefetch(tab_id):
    @callback(
        Output(f"{tab_id}-lazy", "children", allow_duplicate=True),
        Input(f"{tab_id}-prefetch-iv", "n_intervals"),
        Input("prefetch-ready", "data"),
        State(f"{tab_id}-lazy", "children"),
        State("main-tabs", "active_tab"),
        prevent_initial_call=True,
    )
    def _pf(n, ready, current, active, _tid=tab_id):
        if not n or not ready or _tid == active:
            return no_update
        if not _is_loading_placeholder(current):
            return no_update  # user already clicked through
        content = _build_tab_content(_tid)
        return content if content is not None else no_update


for _tid in ("bubble", "heatmap", "dca", "retire", "supercharge",
             "citadel", "leverage", "stack", "model_info", "faq"):
    _register_prefetch(_tid)


# After a chart-tab lazy-load injects DOM, bump first-render so the chart
# callback fires with the newly-mounted graph element present. Guarded on
# active_tab so background-prefetch populating non-active tabs does NOT
# fire their chart callbacks (ALARA — charts only compute when the user
# actually views that tab).
def _register_first_render_bump(tab_id):
    _app_ctx.app.clientside_callback(
        f"""
        function(children, active, cur) {{
            var NU = window.dash_clientside.no_update;
            if (!children) return NU;
            if (typeof children === 'string' && children === 'Loading...') return NU;
            if (active !== '{tab_id}') return NU;  // skip prefetch fires
            if (cur && cur > 0) return NU;         // already rendered
            return (cur || 0) + 1;
        }}
        """,
        Output(f"{tab_id}-first-render", "data", allow_duplicate=True),
        Input(f"{tab_id}-lazy", "children"),
        Input("main-tabs", "active_tab"),
        State(f"{tab_id}-first-render", "data"),
        prevent_initial_call=True,
    )


for _tid in ("bubble", "heatmap", "dca", "retire", "supercharge",
             "citadel", "leverage"):
    _register_first_render_bump(_tid)


def _mi_item_for_pathname(pathname):
    """Extract accordion item_id from /9.N or /mi.N pathname, else None."""
    pathname = _norm(pathname)
    if not pathname:
        return None
    if pathname.startswith("/9."):
        suffix = pathname[3:]
    elif pathname.startswith("/mi."):
        suffix = pathname[4:]
    else:
        return None
    try:
        n = int(suffix)
        if 1 <= n <= len(_MODEL_INFO_ITEMS):
            return _MODEL_INFO_ITEMS[n - 1]
    except (ValueError, IndexError):
        pass
    return None


# Model Info lazy-loading is handled by the universal `_register_lazy_tab`
# loop above (module-level _TAB_CONTENT_CACHE provides the ~70ms-to-cache
# benefit the old _MI_CACHED_CHILDREN provided).


# Accordion item opening (for /mi.N and /9.N deep links) is handled
# entirely by the clientside `_mi_spa_open` callback below. A parallel
# server-side callback targeting the same output triggers
# `DuplicateCallback` on Dash < 4 even with allow_duplicate=True, so
# clientside-only is the safer choice.


# Clientside SPA-nav fallback (kept as belt-and-suspenders):
import json as _json
_mi_items_json = _json.dumps(_MODEL_INFO_ITEMS)

# Two triggers: (1) url.pathname change (SPA nav while MI already loaded),
# (2) model_info-lazy.children change (lazy-load just completed, accordion
# just rendered, need to open the target item that the pathname indicates).
_app_ctx.app.clientside_callback(
    f"""
    function(pathname, _children) {{
        var NU = window.dash_clientside.no_update;
        if (!pathname) return NU;
        var m = pathname.match(/^\\/(?:mi|8)\\.(\\d+)$/);
        if (!m) return NU;
        var n = parseInt(m[1], 10);
        var items = {_mi_items_json};
        if (n < 1 || n > items.length) return NU;
        return items[n - 1];
    }}
    """,
    Output("model-info-accordion", "active_item", allow_duplicate=True),
    Input("url", "pathname"),
    Input("model_info-lazy", "children"),
    prevent_initial_call=True,
)


# Scroll the opened accordion item into view after it expands.
_app_ctx.app.clientside_callback(
    """
    function(active_item) {
        if (!active_item) return window.dash_clientside.no_update;
        /* Retry scroll with increasing delays — on initial page load
           the accordion may take longer to render than on tab switch. */
        function tryScroll(attempts) {
            /* Find the open accordion button by checking all buttons in
               the model-info-accordion for a non-collapsed state or
               matching aria-controls / sibling id. */
            var btns = document.querySelectorAll(
                '#model-info-accordion .accordion-button');
            var target = null;
            for (var i = 0; i < btns.length; i++) {
                var ctrl = btns[i].getAttribute('aria-controls') || '';
                var sib = btns[i].parentElement &&
                          btns[i].parentElement.nextElementSibling;
                var sibId = sib ? (sib.id || '') : '';
                if (ctrl === active_item || ctrl.indexOf(active_item) !== -1 ||
                    sibId === active_item || sibId.indexOf(active_item) !== -1) {
                    target = btns[i]; break;
                }
            }
            /* Fallback: find the non-collapsed button */
            if (!target) {
                for (var j = 0; j < btns.length; j++) {
                    if (!btns[j].classList.contains('collapsed')) {
                        target = btns[j]; break;
                    }
                }
            }
            if (target && target.getBoundingClientRect().height > 0) {
                var rect = target.getBoundingClientRect();
                var offset = window.pageYOffset + rect.top - 60;
                window.scrollTo({top: Math.max(0, offset), behavior: 'smooth'});
            } else if (attempts > 0) {
                setTimeout(function() { tryScroll(attempts - 1); }, 500);
            }
        }
        /* Long initial delay: on fresh page load the tab content
           may not be laid out yet, especially on mobile. */
        setTimeout(function() { tryScroll(5); }, 1000);
        return window.dash_clientside.no_update;
    }
    """,
    Output("model-info-accordion", "className"),
    Input("model-info-accordion", "active_item"),
    prevent_initial_call=True,
)


# Clientside: open lightbox modal when a Model Info image is clicked.
# Moved from server — the callback read ctx.triggered_id and returned two
# Output values with zero other server state, so the round-trip was pure
# plumbing. Now runs entirely in the browser.
_app_ctx.app.clientside_callback(
    """
    function(n_clicks_list) {
        var NU = window.dash_clientside.no_update;
        if (!n_clicks_list || !n_clicks_list.some(function(n){ return n; }))
            return [false, ""];
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) return [false, ""];
        var prop_id = ctx.triggered[0].prop_id;
        var brace_open = prop_id.indexOf('{');
        var brace_close = prop_id.lastIndexOf('}');
        if (brace_open < 0 || brace_close < 0) return [false, ""];
        try {
            var id_obj = JSON.parse(prop_id.slice(brace_open, brace_close + 1));
            return [true, id_obj.src || ""];
        } catch (e) { return [false, ""]; }
    }
    """,
    Output("mi-lightbox", "is_open"),
    Output("mi-lightbox-img", "src"),
    Input({"type": "mi-img", "src": ALL}, "n_clicks"),
    prevent_initial_call=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# FAQ accordion deep-linking (/10.N)
# FAQ lazy-loading handled by the universal `_register_lazy_tab` loop above.
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("faq-accordion", "active_item"),
    Input("faq-lazy", "children"),
    State("url", "pathname"),
    prevent_initial_call=True,
)
def open_faq_item(children, pathname):
    """Open a specific FAQ accordion item after lazy load: /10.N (1-indexed)."""
    pathname = _norm(pathname)
    if not pathname:
        return no_update
    n_str = None
    if pathname.startswith("/10."):
        n_str = pathname[4:]
    elif pathname.startswith("/faq."):
        n_str = pathname[5:]
    if n_str is None:
        return no_update
    try:
        n = int(n_str)
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
# Source of truth lives in layout/heatmap.py so the rendered pill ids and
# the callback Input/Output set stay in sync. Divergence causes Dash to
# abort _hm_pill_click with a nonexistent-object ReferenceError on click.
from layout.heatmap import _hm_pill_models as _hm_pill_models_fn  # noqa: E402
_HM_PILL_MODELS = _hm_pill_models_fn()

# Map removed pill IDs (Phase 1 share links may have these in hm-active-model)
# → surviving pill. Used as a graceful fallback when old snapshot decodes.
_HM_LEGACY_MODEL_FALLBACK = {
    "qr": "pl",        # QR was bands-only; PL is the closest match
    "lp2": "lppl", "lp3": "lppl", "lp4": "lppl",
    "lppl_w": "lppl", "lp2_w": "lppl", "lp3_w": "lppl", "lp4_w": "lppl",
    "lp4_n13": "lppl", "lp4_w_n13": "lppl",
    "linppl": "lppl",  # removed from heatmap pills 2026-04-11 — fall back to LPPL
    "hybppl_dd": "hybppl", "hyb2l": "hybppl", "hyb2c": "hybppl",
    "hyb2b": "hybppl", "hyb4d": "hybppl",
    "exp": "bub",      # display-only demo
    "s2f": "bub",      # display-only demo
}


# Heatmap model pill bar -- click to select active model
# ══════════════════════════════════════════════════════════════════════════════

# Pill IDs match the Phase 2 standardized _HM_PILL_MODELS list above.
_HM_PILL_IDS = [f"hm-pill-{k}" for k in _HM_PILL_MODELS]

import json as _json
_pill_models_json = _json.dumps(_HM_PILL_MODELS)
_pill_ids_json = _json.dumps(_HM_PILL_IDS)
_legacy_json = _json.dumps(_HM_LEGACY_MODEL_FALLBACK)

# _hm_deep_link
_app_ctx.app.clientside_callback(
    f"""function(pathname) {{
        var nu = window.dash_clientside.no_update;
        var models = {_pill_models_json};
        if (!pathname) return nu;
        pathname = pathname.replace(/\\/+$/, '') || '/';
        if (!pathname.startsWith('/2.')) return nu;
        try {{
            var n = parseInt(pathname.substring(3));
            if (n >= 1 && n <= models.length) return models[n - 1];
        }} catch(e) {{}}
        return nu;
    }}""",
    Output("hm-active-model", "data", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call=True,
)

# _hm_pill_click
_app_ctx.app.clientside_callback(
    f"""function() {{
        var pill_ids = {_pill_ids_json};
        var tid = dash_clientside.callback_context.triggered_id;
        if (!tid) throw window.dash_clientside.PreventUpdate;
        var model_key = tid.replace('hm-pill-', '');
        var outlines = pill_ids.map(function(pid) {{ return pid !== tid; }});
        return [model_key].concat(outlines);
    }}""",
    Output("hm-active-model", "data", allow_duplicate=True),
    *[Output(pid, "outline", allow_duplicate=True) for pid in _HM_PILL_IDS],
    *[Input(pid, "n_clicks") for pid in _HM_PILL_IDS],
    prevent_initial_call=True,
)

# _hm_pill_sync
_app_ctx.app.clientside_callback(
    f"""function(model_key) {{
        var pill_ids = {_pill_ids_json};
        var legacy = {_legacy_json};
        model_key = model_key || 'bub';
        var models = {_pill_models_json};
        if (models.indexOf(model_key) === -1) {{
            model_key = legacy[model_key] || 'bub';
        }}
        var active_id = 'hm-pill-' + model_key;
        return pill_ids.map(function(pid) {{ return pid !== active_id; }});
    }}""",
    *[Output(pid, "outline", allow_duplicate=True) for pid in _HM_PILL_IDS],
    Input("hm-active-model", "data"),
    prevent_initial_call=True,
)
