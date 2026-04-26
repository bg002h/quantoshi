"""MC UI control callbacks — toggles, regime options, year sync, cost display."""

import json

import dash
from dash import html, Input, Output, State, callback

import _app_ctx
from colors import (
    FALLBACK_MODEL_GRAY, DIM_TEXT, KNIGHT_GOLD,
    MC_FREE_GREEN, MC_LIVE_AMBER,
    UI_FONT_SM, UI_FONT_LG,
)
from callbacks.coerce import _ci, _cf
from figures import FREQ_PPY
from layout import (_bold_opts, _MC_CACHED_START_YRS, _MC_CACHED_YEARS,
                    _MC_CACHED_ENTRY_QS, _MC_PRICE_LIVE,
                    _MC_ENTRY_Q_OPTIONS, _MC_ENTRY_Q_OPTIONS_ADV,
                    _regime_options)
from mc_cache import (MC_YEARS_OPTIONS, MC_BINS, MC_SIMS, MC_FREQ,
                      MC_DEFAULT_YEARS, MC_DEFAULT_START_YR, MC_DEFAULT_ENTRY_Q,
                      MC_FREE_SIMS, CACHED_START_YRS, ENTRY_PCT_BINS,
                      _CACHED_MODEL_KEYS)
import btcpay


_app_ctx.app.clientside_callback(
    "function(v) { return (v && v.length) ? {} : {display:'none'}; }",
    Output("dca-sc-body", "style"),
    Input("dca-sc-enable", "value"),
)
for _mc_tog in ("dca", "ret", "hm", "sc", "cp", "bub"):
    _app_ctx.app.clientside_callback(
        "function(v) { return (v && v.length) ? {} : {display:'none'}; }",
        Output(f"{_mc_tog}-mc-body", "style"),
        Input(f"{_mc_tog}-mc-enable", "value"),
    )

# MC engine toggle → inject/remove MC from Display Models options + value
# (excludes "hm" because the heatmap uses a pill bar instead of a checklist)
for _mc_auto in ("dca", "ret", "sc", "bub"):
    _app_ctx.app.clientside_callback(
        """
        function(mc_enable, cur_opts, cur_models) {
            var opts = (cur_opts || []).slice();
            var models = (cur_models || []).slice();
            var mc_opt = {label: " MC Simulation", value: "mc"};
            var has_mc_opt = opts.some(function(o) { return o.value === "mc"; });
            if (mc_enable && mc_enable.length) {
                // Activating: add MC option if missing, check it
                if (!has_mc_opt) opts.push(mc_opt);
                if (models.indexOf("mc") === -1) models.push("mc");
            } else {
                // Deactivating: remove MC option and uncheck it
                opts = opts.filter(function(o) { return o.value !== "mc"; });
                models = models.filter(function(v) { return v !== "mc"; });
            }
            return [opts, models];
        }
        """,
        Output(f"{_mc_auto}-model-show", "options", allow_duplicate=True),
        Output(f"{_mc_auto}-model-show", "value", allow_duplicate=True),
        Input(f"{_mc_auto}-mc-enable", "value"),
        State(f"{_mc_auto}-model-show", "options"),
        State(f"{_mc_auto}-model-show", "value"),
        prevent_initial_call='initial_duplicate',
    )

# ── Pre-computed options (embedded as JS constants for clientside callbacks) ──
# _MC_ENTRY_Q_OPTIONS / _ADV contain html.Span labels → serialize via
# to_plotly_json(). _regime_options(n) returns plain-string labels for n in
# 2..10 (bins slider range).
def _opts_to_jsonable(opts):
    out = []
    for o in opts:
        label = o["label"]
        if hasattr(label, "to_plotly_json"):
            label = label.to_plotly_json()
        out.append({"label": label, "value": o["value"]})
    return out

_MC_ENTRY_Q_OPTIONS_JSON = json.dumps(_opts_to_jsonable(_MC_ENTRY_Q_OPTIONS))
_MC_ENTRY_Q_OPTIONS_ADV_JSON = json.dumps(_opts_to_jsonable(_MC_ENTRY_Q_OPTIONS_ADV))
_REGIME_OPTIONS_JSON = json.dumps(
    {str(n): _opts_to_jsonable(_regime_options(n)) for n in range(2, 11)}
)

_MC_ADV_JS = """
function(val) {
    var OPTS = %s;
    var OPTS_ADV = %s;
    var active = !!(val && val.length);
    return [active ? {} : {display:'none'}, active ? OPTS_ADV : OPTS];
}
""" % (_MC_ENTRY_Q_OPTIONS_JSON, _MC_ENTRY_Q_OPTIONS_ADV_JSON)

for _mc_adv in ("dca", "ret", "hm", "sc", "cp", "bub"):
    _app_ctx.app.clientside_callback(
        _MC_ADV_JS,
        Output(f"{_mc_adv}-mc-adv-body", "style"),
        Output(f"{_mc_adv}-mc-entry-q", "options"),
        Input(f"{_mc_adv}-mc-advanced", "value"),
    )

_REGIME_JS = """
function(n_bins) {
    var TABLE = %s;
    var n = parseInt(n_bins);
    if (isNaN(n) || n < 2 || n > 10) n = 5;
    var opts = TABLE[String(n)] || TABLE["5"];
    var values = [];
    for (var i = 0; i < n; i++) values.push(i);
    return [opts, values];
}
""" % _REGIME_OPTIONS_JSON

for _mc_reg in ("dca", "ret", "hm", "sc", "cp", "bub"):
    _app_ctx.app.clientside_callback(
        _REGIME_JS,
        Output(f"{_mc_reg}-mc-regime", "options"),
        Output(f"{_mc_reg}-mc-regime", "value"),
        Input(f"{_mc_reg}-mc-bins", "value"),
        prevent_initial_call=True,
    )

# ── Frequency unlock toggle + modal ──────────────────────────────────────────
for _fp in ("dca", "ret", "sc"):
    _app_ctx.app.clientside_callback(
        """function(unlock, cur_freq) {
            if (unlock && unlock.length) {
                return [false, cur_freq, true];
            }
            return [true, 'Monthly', false];
        }""",
        Output(f"{_fp}-freq", "disabled"),
        Output(f"{_fp}-freq", "value", allow_duplicate=True),
        Output("freq-warning-modal", "is_open", allow_duplicate=True),
        Input(f"{_fp}-freq-unlock", "value"),
        State(f"{_fp}-freq", "value"),
        prevent_initial_call=True,
    )

_app_ctx.app.clientside_callback(
    "function(n) { return false; }",
    Output("freq-warning-modal", "is_open", allow_duplicate=True),
    Input("freq-warning-ok", "n_clicks"),
    prevent_initial_call=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# MC ↔ Year Range sync — auto-extend year range to include MC horizon
# ══════════════════════════════════════════════════════════════════════════════
# DCA/Ret RangeSlider extension is handled by the clientside _MC_EXTEND_YR_JS
# callback registered further below. SC uses a separate single-end slider:
_MC_SC_YR_SYNC_JS = """
function(mc_start_yr, mc_years, mc_enable, end_yr) {
    var noUpdate = window.dash_clientside.no_update;
    if (!mc_enable || !mc_enable.length) return [noUpdate, noUpdate];
    var syr = parseInt(mc_start_yr); if (isNaN(syr)) syr = 2031;
    var yrs = parseInt(mc_years);    if (isNaN(yrs)) yrs = 10;
    var mc_end = syr + yrs;
    var ey = parseInt(end_yr); if (isNaN(ey)) ey = 2075;
    if (ey >= mc_end) return [noUpdate, noUpdate];
    return [mc_end, Math.max(mc_end, 2100)];
}
"""
_app_ctx.app.clientside_callback(
    _MC_SC_YR_SYNC_JS,
    Output("sc-end-yr", "value", allow_duplicate=True),
    Output("sc-end-yr", "max", allow_duplicate=True),
    Input("sc-mc-start-yr", "value"),
    Input("sc-mc-years", "value"),
    Input("sc-mc-enable", "value"),
    State("sc-end-yr", "value"),
    prevent_initial_call='initial_duplicate',
)


# MC match indicator — show whether chart reflects current MC settings
# Returns: [match_text, match_style, overlay_style, wrap_class, badge_style,
#           restore_btn_style]
_MC_MATCH_JS_TPL = """
function(mc_enable, mc_years, mc_start_yr, mc_entry_q, mc_model_src, rendered_key) {{
    var hide = {{display: "none"}};
    var base = {{fontSize: "10px", fontWeight: "600", textAlign: "center", marginTop: "4px"}};
    var noPremium = "{base_cls}";
    var premium = "{base_cls}{sep}mc-premium-chart";
    if (!mc_enable || !mc_enable.length) return ["", hide, hide, noPremium, hide, hide];
    if (!rendered_key) return [
        "\u26a0 Chart does not include MC overlay",
        Object.assign({{}}, base, {{color: "{live_clr}"}}),
        {{}},
        noPremium,
        hide,
        hide
    ];
    var yrs = parseInt(mc_years) || 40;
    var syr = parseInt(mc_start_yr) || 2028;
    var eq  = parseInt(mc_entry_q) || 10;
    var msrc = mc_model_src || "bub";
    var match = (yrs === rendered_key.years &&
                 syr === rendered_key.start_yr &&
                 eq === rendered_key.entry_q);
    if (match) {{
        return [
            "\u2713 Chart reflects current MC settings",
            Object.assign({{}}, base, {{color: "{free_clr}"}}),
            hide,
            premium,
            {{}},
            hide
        ];
    }}
    return [
        "\u26a0 MC settings changed \u2014 chart is stale",
        Object.assign({{}}, base, {{color: "{live_clr}"}}),
        {{}},
        noPremium,
        hide,
        {{}}
    ];
}}
"""
for _mc_m in ("dca", "ret", "hm", "sc", "cp", "bub"):
    _wrap_id = {"dca": "dca-chart-wrap", "ret": "ret-chart-wrap",
                "hm": "hm-mc-panel", "sc": "sc-chart-wrap",
                "cp": "cp-chart-wrap",
                "bub": "bubble-graph-chart-wrap"}[_mc_m]
    _base_cls = ""
    _sep = " " if _base_cls else ""
    _app_ctx.app.clientside_callback(
        _MC_MATCH_JS_TPL.format(
            base_cls=_base_cls, sep=_sep,
            free_clr=MC_FREE_GREEN, live_clr=MC_LIVE_AMBER,
        ),
        Output(f"{_mc_m}-mc-match", "children"),
        Output(f"{_mc_m}-mc-match", "style"),
        Output(f"{_mc_m}-mc-overlay", "style"),
        Output(_wrap_id, "className"),
        Output(f"{_mc_m}-mc-badge", "style"),
        Output(f"{_mc_m}-mc-restore-btn", "style"),
        Input(f"{_mc_m}-mc-enable", "value"),
        Input(f"{_mc_m}-mc-years", "value"),
        Input(f"{_mc_m}-mc-start-yr", "value"),
        Input(f"{_mc_m}-mc-entry-q", "value"),
        Input(f"{_mc_m}-mc-model-src", "value"),
        Input(f"{_mc_m}-mc-rendered-key", "data"),
    )

# MC restore button — revert controls to last cached simulation settings
_RESTORE_MC_JS = f"""function(n_clicks, mc_cached) {{
    var nu = window.dash_clientside.no_update;
    if (!mc_cached || !mc_cached.path_key) return [nu, nu, nu, nu, nu, nu];
    var pk = mc_cached.path_key;
    return [
        pk.mc_years    !== undefined ? pk.mc_years    : {MC_DEFAULT_YEARS},
        pk.mc_start_yr !== undefined ? pk.mc_start_yr : {MC_DEFAULT_START_YR},
        pk.mc_entry_q  !== undefined ? pk.mc_entry_q  : {MC_DEFAULT_ENTRY_Q},
        pk.mc_bins     !== undefined ? pk.mc_bins     : {MC_BINS},
        pk.mc_sims     !== undefined ? pk.mc_sims     : {MC_SIMS},
        pk.mc_window   !== undefined ? pk.mc_window   : null
    ];
}}"""

for _rpfx in ("hm", "dca", "ret", "sc", "cp", "bub"):
    _app_ctx.app.clientside_callback(
        _RESTORE_MC_JS,
        Output(f"{_rpfx}-mc-years", "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-start-yr", "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-entry-q", "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-bins", "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-sims", "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-window", "value", allow_duplicate=True),
        Input(f"{_rpfx}-mc-restore-btn", "n_clicks"),
        State(f"{_rpfx}-mc-results", "data"),
        prevent_initial_call=True,
    )


def _restore_mc(n_clicks, mc_cached):
    """Plain helper kept for tests + __init__ re-export. The live callback is
    a clientside port (above)."""
    if not mc_cached or not mc_cached.get("path_key"):
        return [dash.no_update] * 6
    pk = mc_cached["path_key"]
    return (
        pk.get("mc_years", MC_DEFAULT_YEARS),
        pk.get("mc_start_yr", MC_DEFAULT_START_YR),
        pk.get("mc_entry_q", MC_DEFAULT_ENTRY_Q),
        pk.get("mc_bins", MC_BINS),
        pk.get("mc_sims", MC_SIMS),
        pk.get("mc_window"),
    )


# MC horizon → auto-extend year range slider (DCA + Retire)
_MC_EXTEND_YR_JS = """
function(mc_enable, mc_years, mc_start_yr, yr_range, slider_max) {
    if (!mc_enable || !mc_enable.length)
        return [window.dash_clientside.no_update, window.dash_clientside.no_update];
    var yrs = parseInt(mc_years) || 10;
    var syr = parseInt(mc_start_yr) || 2026;
    var need = syr + yrs;
    var cur = yr_range ? yr_range.slice() : [syr, need];
    var mx = slider_max || cur[1];
    var changed = false;
    if (cur[1] < need) { cur[1] = need; changed = true; }
    if (mx < need) { mx = need; changed = true; }
    if (!changed)
        return [window.dash_clientside.no_update, window.dash_clientside.no_update];
    return [cur, mx];
}
"""
for _ext_pfx in ("dca", "ret"):
    _app_ctx.app.clientside_callback(
        _MC_EXTEND_YR_JS,
        Output(f"{_ext_pfx}-yr-range", "value", allow_duplicate=True),
        Output(f"{_ext_pfx}-yr-range", "max", allow_duplicate=True),
        Input(f"{_ext_pfx}-mc-enable", "value"),
        Input(f"{_ext_pfx}-mc-years", "value"),
        Input(f"{_ext_pfx}-mc-start-yr", "value"),
        State(f"{_ext_pfx}-yr-range", "value"),
        State(f"{_ext_pfx}-yr-range", "max"),
        prevent_initial_call=True,
    )

# SC match callback now included in the loop above

# PPY display (steps/year) — clientside for instant feedback
_PPY_JS = """
function(freq) {
    var m = {Daily:"365/yr", Weekly:"52/yr", Monthly:"12/yr", Quarterly:"4/yr", Annually:"1/yr"};
    return m[freq] || "12/yr";
}
"""
_app_ctx.app.clientside_callback(_PPY_JS, Output("hm-mc-ppy","value"),  Input("hm-mc-freq","value"))

# Placeholder callback for hm-swipe-scroll-dummy (swipe container removed; pills replaced it)
_app_ctx.app.clientside_callback(
    "function(panelStyle) { return ''; }",
    Output("hm-swipe-scroll-dummy", "children"),
    Input("hm-mc-panel", "style"),
    prevent_initial_call=True,
)


# ── Dynamic years limit based on sims × freq (cap at 250K datapoints) ────────
_MC_MAX_DATAPOINTS = 50_000_000
def _mc_years_options(sims, freq):
    """Return filtered years dropdown options based on sims × freq cap."""
    ppy = FREQ_PPY.get(freq or "Monthly", 12)
    sims = _ci(sims, 800)
    max_steps = _MC_MAX_DATAPOINTS // sims
    max_years = max_steps // ppy if ppy > 0 else 50
    valid = [y for y in MC_YEARS_OPTIONS if y <= max_years]
    if not valid:
        return [{"label": "1 yr", "value": 1}]
    return _bold_opts(valid, lambda v: f"{v} yr", _MC_CACHED_YEARS)

# HM keeps mc-freq (not consolidated)
@callback(
    Output("hm-mc-years", "options"),
    Output("hm-mc-years", "value"),
    Input("hm-mc-sims", "value"),
    Input("hm-mc-freq", "value"),
    State("hm-mc-years", "value"),
    prevent_initial_call=True,
)
def _update_hm_mc_years_opts(sims, freq, cur_years):
    opts = _mc_years_options(sims, freq)
    max_avail = opts[-1]["value"]
    val = cur_years if (cur_years and cur_years <= max_avail) else max_avail
    return opts, val

# DCA/Ret/SC use shared freq (consolidated)
for _mc_pfx in ("dca", "ret", "sc", "cp"):
    @callback(
        Output(f"{_mc_pfx}-mc-years", "options"),
        Output(f"{_mc_pfx}-mc-years", "value"),
        Input(f"{_mc_pfx}-mc-sims", "value"),
        Input(f"{_mc_pfx}-freq", "value"),
        State(f"{_mc_pfx}-mc-years", "value"),
        prevent_initial_call=True,
    )
    def _update_mc_years_opts(sims, freq, cur_years, _pfx=_mc_pfx):
        opts = _mc_years_options(sims, freq)
        max_avail = opts[-1]["value"]
        val = cur_years if (cur_years and cur_years <= max_avail) else max_avail
        return opts, val



_MC_BASE_SIMS = 800  # pricing baseline — costs scale linearly from this
_MC_BASE_PPY  = 12   # pricing baseline — Monthly
_MC_BASE_BINS = 5    # cache uses 5×5 transition matrix

# ── MC cost display — clientside ──────────────────────────────────────────────
# Pure arithmetic + static-dict lookups: pre-computed constants live in the
# static asset btc_web/assets/_mc_cost_consts.js (loaded once per page as
# `window.QS_MC_COST_CONSTS`), and the JS template below reproduces
# _calc_mc_cost + _mc_cost_display to eliminate the 5-tab × 9-Input chatty
# server callback that used to fire on every slider nudge.
#
# Regenerate btc_web/assets/_mc_cost_consts.js whenever any of these sources
# drift: FREQ_PPY (_app_ctx), _MC_PRICE_LIVE (layout/mc_controls.py),
# MC_BINS/MC_FREE_SIMS/MC_FREQ/MC_DEFAULT_ENTRY_Q/_CACHED_MODEL_KEYS/
# CACHED_START_YRS/MC_YEARS_OPTIONS/ENTRY_PCT_BINS (mc_cache), or the
# styling constants imported from colors.py below.
_MC_COST_JS_TPL = """
function(mc_enable, mc_freq, mc_years, mc_bins, mc_sims, mc_window,
         mc_start_yr, mc_entry_q, mc_model_src) {
    var C = window.QS_MC_COST_CONSTS;
    var TAB = "%s";

    function span(text, style) {
        return {namespace: "dash_html_components", type: "Span",
                props: {children: text, style: style || {}}};
    }
    function div(children, style) {
        return {namespace: "dash_html_components", type: "Div",
                props: {children: children, style: style || {}}};
    }
    function bold(text) {
        return {namespace: "dash_html_components", type: "B",
                props: {children: text}};
    }
    function fmtInt(n) {
        return n.toString().replace(/\\B(?=(\\d{3})+(?!\\d))/g, ",");
    }
    function ci(v, dflt) {
        if (v === null || v === undefined || v === "") return dflt;
        var n = parseInt(v); return isNaN(n) ? dflt : n;
    }
    function cf(v, dflt) {
        if (v === null || v === undefined || v === "") return dflt;
        var n = parseFloat(v); return isNaN(n) ? dflt : n;
    }

    var mc_years_c = ci(mc_years, 40);
    var start_yr_c = ci(mc_start_yr, 2028);
    var entry_q_c  = Math.round(cf(mc_entry_q, 50) * 10) / 10;
    var sims_c     = ci(mc_sims, C.BASE_SIMS);
    var bins_c     = ci(mc_bins, C.BASE_BINS);
    var freq_c     = mc_freq || "Monthly";
    var ppy        = C.FREQ_PPY[freq_c] != null ? C.FREQ_PPY[freq_c] : C.BASE_PPY;
    var model_key  = mc_model_src || "bub";

    // ── btcpay.is_free_tier port ──────────────────────────────────────────
    var is_free = false;
    if (bins_c === C.MC_BINS && sims_c <= C.MC_FREE_SIMS && freq_c === C.MC_FREQ) {
        // is_cached(model_key, start_yr, entry_q, mc_years)
        if (C.CACHED_MODEL_KEYS.indexOf(model_key) !== -1 &&
            C.CACHED_START_YRS.indexOf(start_yr_c) !== -1 &&
            C.MC_YEARS_OPTIONS.indexOf(mc_years_c) !== -1) {
            // is_cache_aligned_q
            var raw = (entry_q_c || C.MC_DEFAULT_ENTRY_Q) / 100.0;
            var best = C.ENTRY_PCT_BINS[0];
            var bestD = Math.abs(raw - best);
            for (var i = 1; i < C.ENTRY_PCT_BINS.length; i++) {
                var d = Math.abs(raw - C.ENTRY_PCT_BINS[i]);
                if (d < bestD) { best = C.ENTRY_PCT_BINS[i]; bestD = d; }
            }
            if (Math.abs(raw - best) < 0.005) is_free = true;
        }
    }

    if (is_free) {
        var children = [
            div([
                span("Free tier", {fontWeight: "bold", color: C.MC_FREE_GREEN}),
                span(" \u2022 " + mc_years_c + " yr simulation", {color: C.DIM_TEXT}),
            ]),
            div("Pre-computed \u2022 instant",
                {color: C.FALLBACK_MODEL_GRAY, fontSize: C.UI_FONT_SM}),
            div(bold("Cost: Free \u2713"),
                {marginTop: "2px", color: C.MC_FREE_GREEN}),
        ];
        return [children, 0];
    }

    // ── Live tier pricing ────────────────────────────────────────────────
    var scale = (sims_c / C.BASE_SIMS) * (ppy / C.BASE_PPY) *
                ((bins_c * bins_c) / (C.BASE_BINS * C.BASE_BINS));
    var base_price = C.PRICE_LIVE[String(mc_years_c)];
    if (base_price == null) base_price = 2000;
    var tier_label = "Live";
    var tier_color = C.MC_LIVE_AMBER;
    var time_scale = scale * (mc_years_c / 10);
    var lo = Math.max(1, Math.round(1 * time_scale));
    var hi = Math.max(1, Math.round(3 * time_scale));
    var tier_note = (lo < hi)
        ? "Computed on demand \u2022 ~" + lo + "\u2013" + hi + "s"
        : "Computed on demand \u2022 ~" + lo + "s";
    var price = Math.trunc(base_price * scale);
    if (TAB === "hm") price = Math.trunc(price * 0.5);

    var costRowChildren = [bold("Cost: " + fmtInt(price) + " sats")];
    if (price <= 400) {
        costRowChildren.push(span("  \u26a1", {fontSize: C.UI_FONT_LG}));
    } else {
        costRowChildren.push("");
    }

    var children2 = [
        div([
            span(tier_label, {fontWeight: "bold", color: tier_color}),
            span(" \u2022 " + mc_years_c + " yr simulation", {color: C.DIM_TEXT}),
        ]),
        div(tier_note, {color: C.FALLBACK_MODEL_GRAY, fontSize: C.UI_FONT_SM}),
        div(costRowChildren, {marginTop: "2px"}),
    ];

    if (price > 10000) {
        children2.push(div(
            "\u26a0 Most users are unlikely to benefit from simulations " +
            "this expensive. Consider using cached (bold) settings.",
            {fontSize: C.UI_FONT_SM, color: C.KNIGHT_GOLD, marginTop: "4px",
             fontStyle: "italic", lineHeight: "1.3"}));
    }

    return [children2, price];
}
"""

for _cost_pfx in ("hm", "dca", "ret", "sc", "cp", "bub"):
    # hm + bub use {prefix}-mc-freq (no shared freq dropdown on those tabs);
    # dca/ret/sc/cp share the tab's main {prefix}-freq dropdown.
    _freq_id = (f"{_cost_pfx}-mc-freq" if _cost_pfx in ("hm", "bub")
                else f"{_cost_pfx}-freq")
    _app_ctx.app.clientside_callback(
        _MC_COST_JS_TPL % (_cost_pfx,),
        Output(f"{_cost_pfx}-mc-cost", "children"),
        Output(f"{_cost_pfx}-mc-price-val", "data"),
        Input(f"{_cost_pfx}-mc-enable",   "value"),
        Input(_freq_id,                    "value"),
        Input(f"{_cost_pfx}-mc-years",    "value"),
        Input(f"{_cost_pfx}-mc-bins",     "value"),
        Input(f"{_cost_pfx}-mc-sims",     "value"),
        Input(f"{_cost_pfx}-mc-window",   "value"),
        Input(f"{_cost_pfx}-mc-start-yr", "value"),
        Input(f"{_cost_pfx}-mc-entry-q", "value"),
        Input(f"{_cost_pfx}-mc-model-src", "value"),
        prevent_initial_call=True,
    )
