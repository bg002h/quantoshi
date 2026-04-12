"""MC UI control callbacks — toggles, regime options, year sync, cost display."""

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
                      MC_DEFAULT_YEARS, MC_DEFAULT_START_YR, MC_DEFAULT_ENTRY_Q)
import btcpay


_app_ctx.app.clientside_callback(
    "function(v) { return (v && v.length) ? {} : {display:'none'}; }",
    Output("dca-sc-body", "style"),
    Input("dca-sc-enable", "value"),
)
for _mc_tog in ("dca", "ret", "hm", "sc", "cp"):
    _app_ctx.app.clientside_callback(
        "function(v) { return (v && v.length) ? {} : {display:'none'}; }",
        Output(f"{_mc_tog}-mc-body", "style"),
        Input(f"{_mc_tog}-mc-enable", "value"),
    )

# MC engine toggle → inject/remove MC from Display Models options + value
for _mc_auto in ("dca", "ret", "sc"):
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

for _mc_adv in ("dca", "ret", "hm", "sc", "cp"):
    @callback(
        Output(f"{_mc_adv}-mc-adv-body", "style"),
        Output(f"{_mc_adv}-mc-entry-q", "options"),
        Input(f"{_mc_adv}-mc-advanced", "value"),
    )
    def _toggle_mc_adv(val):
        style = {} if val else {"display": "none"}
        opts = _MC_ENTRY_Q_OPTIONS_ADV if val else _MC_ENTRY_Q_OPTIONS
        return style, opts

for _mc_reg in ("dca", "ret", "hm", "sc", "cp"):
    @callback(
        Output(f"{_mc_reg}-mc-regime", "options"),
        Output(f"{_mc_reg}-mc-regime", "value"),
        Input(f"{_mc_reg}-mc-bins", "value"),
        prevent_initial_call=True,
    )
    def _update_regime_opts(n_bins):
        n = _ci(n_bins, 5)
        return _regime_options(n), list(range(n))

# ── Frequency unlock toggle + modal ──────────────────────────────────────────
for _fp in ("dca", "ret", "sc"):
    @callback(
        Output(f"{_fp}-freq", "disabled"),
        Output(f"{_fp}-freq", "value", allow_duplicate=True),
        Output("freq-warning-modal", "is_open", allow_duplicate=True),
        Input(f"{_fp}-freq-unlock", "value"),
        State(f"{_fp}-freq", "value"),
        prevent_initial_call=True,
    )
    def _toggle_freq_unlock(unlock, cur_freq, _pfx=_fp):
        if unlock:
            return False, cur_freq, True   # enable dropdown, keep value, show modal
        return True, "Monthly", False      # disable, reset to Monthly, hide modal

@callback(
    Output("freq-warning-modal", "is_open", allow_duplicate=True),
    Input("freq-warning-ok", "n_clicks"),
    prevent_initial_call=True,
)
def _close_freq_modal(n):
    return False

# ══════════════════════════════════════════════════════════════════════════════
# MC ↔ Year Range sync — auto-extend year range to include MC horizon
# ══════════════════════════════════════════════════════════════════════════════

for _pfx in ("dca", "ret"):
    @callback(
        Output(f"{_pfx}-yr-range", "value", allow_duplicate=True),
        Output(f"{_pfx}-yr-range", "max", allow_duplicate=True),
        Input(f"{_pfx}-mc-start-yr", "value"),
        Input(f"{_pfx}-mc-years", "value"),
        Input(f"{_pfx}-mc-enable", "value"),
        State(f"{_pfx}-yr-range", "value"),
        prevent_initial_call='initial_duplicate',
    )
    def _mc_yr_sync(mc_start_yr, mc_years, mc_enable, yr_range):
        """Extend RangeSlider to include MC horizon."""
        if not mc_enable:
            return dash.no_update, dash.no_update
        mc_end = _ci(mc_start_yr, 2031) + _ci(mc_years, 10)
        yr_range = yr_range or [2024, 2034]
        if yr_range[1] >= mc_end:
            return dash.no_update, dash.no_update
        return [yr_range[0], mc_end], mc_end

# SC uses separate start/end sliders — sync end-yr slider
@callback(
    Output("sc-end-yr", "value", allow_duplicate=True),
    Output("sc-end-yr", "max", allow_duplicate=True),
    Input("sc-mc-start-yr", "value"),
    Input("sc-mc-years", "value"),
    Input("sc-mc-enable", "value"),
    State("sc-end-yr", "value"),
    prevent_initial_call='initial_duplicate',
)
def _mc_sc_yr_sync(mc_start_yr, mc_years, mc_enable, end_yr):
    """Extend SC end-year slider to include MC horizon."""
    if not mc_enable:
        return dash.no_update, dash.no_update
    mc_end = _ci(mc_start_yr, 2031) + _ci(mc_years, 10)
    end_yr = _ci(end_yr, 2075)
    if end_yr >= mc_end:
        return dash.no_update, dash.no_update
    return mc_end, max(mc_end, 2100)


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
for _mc_m in ("dca", "ret", "hm", "sc", "cp"):
    _wrap_id = {"dca": "dca-chart-wrap", "ret": "ret-chart-wrap",
                "hm": "hm-mc-panel", "sc": "sc-chart-wrap",
                "cp": "cp-chart-wrap"}[_mc_m]
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
for _rpfx in ("hm", "dca", "ret", "sc", "cp"):
    @callback(
        Output(f"{_rpfx}-mc-years",    "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-start-yr", "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-entry-q",  "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-bins",     "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-sims",     "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-window",   "value", allow_duplicate=True),
        Input(f"{_rpfx}-mc-restore-btn", "n_clicks"),
        State(f"{_rpfx}-mc-results",     "data"),
        prevent_initial_call=True,
    )
    def _restore_mc(n_clicks, mc_cached):
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

def _calc_mc_cost(mc_years, start_yr, entry_q=50, mc_sims=200, mc_freq="Monthly",
                  mc_bins=5, tab="dca", model_key="bub"):
    """Calculate MC simulation cost and tier info.

    Returns (price_sats: int, is_free: bool, is_cached: bool,
             tier_label: str, tier_color: str, tier_note: str,
             mc_years_c: int) where *mc_years_c* is the coerced value.
    """
    mc_years = _ci(mc_years, 40)
    start_yr = _ci(start_yr, 2028)
    entry_q  = round(_cf(entry_q, 50), 1)
    mc_sims  = _ci(mc_sims, _MC_BASE_SIMS)
    mc_bins  = _ci(mc_bins, _MC_BASE_BINS)
    mc_ppy   = FREQ_PPY.get(mc_freq or "Monthly", _MC_BASE_PPY)

    is_free = btcpay.is_free_tier(model_key, mc_years, start_yr, entry_q,
                                  mc_bins=mc_bins, mc_sims=mc_sims,
                                  mc_freq=mc_freq)

    if is_free:
        price = 0
        is_cached = True
        tier_label = "Cached"
        tier_color = MC_FREE_GREEN
        tier_note = "Pre-computed \u2022 instant"
    else:
        # Scale factor relative to baseline (200 sims, Monthly, 5×5 matrix)
        scale = ((mc_sims / _MC_BASE_SIMS) * (mc_ppy / _MC_BASE_PPY)
                 * (mc_bins ** 2 / _MC_BASE_BINS ** 2))
        base_price = _MC_PRICE_LIVE.get(mc_years, 2000)
        tier_label = "Live"
        tier_color = MC_LIVE_AMBER
        time_scale = scale * (mc_years / 10)
        lo, hi = max(1, round(1 * time_scale)), max(1, round(3 * time_scale))
        tier_note = (f"Computed on demand \u2022 ~{lo}\u2013{hi}s" if lo < hi
                     else f"Computed on demand \u2022 ~{lo}s")
        price = int(base_price * scale)
        if tab == "hm":
            price = int(price * 0.5)
        is_cached = False

    return price, is_free, is_cached, tier_label, tier_color, tier_note, mc_years


def _mc_cost_display(mc_years, start_yr, entry_q=50, mc_sims=200, mc_freq="Monthly",
                     mc_bins=5, tab="dca", model_key="bub"):
    """Return cost display elements showing cached vs live pricing."""
    price, is_free, is_cached, tier_label, tier_color, tier_note, mc_years_c = \
        _calc_mc_cost(mc_years, start_yr, entry_q, mc_sims, mc_freq, mc_bins, tab, model_key=model_key)

    if is_free:
        return ([
            html.Div([
                html.Span("Free tier", style={"fontWeight": "bold", "color": MC_FREE_GREEN}),
                html.Span(f" \u2022 {mc_years_c} yr simulation", style={"color": DIM_TEXT}),
            ]),
            html.Div(tier_note, style={"color": FALLBACK_MODEL_GRAY, "fontSize": UI_FONT_SM}),
            html.Div(html.B("Cost: Free \u2713"),
                     style={"marginTop": "2px", "color": MC_FREE_GREEN}),
        ], 0)

    children = [
        html.Div([
            html.Span(f"{tier_label}", style={"fontWeight": "bold", "color": tier_color}),
            html.Span(f" \u2022 {mc_years_c} yr simulation", style={"color": DIM_TEXT}),
        ]),
        html.Div(tier_note, style={"color": FALLBACK_MODEL_GRAY, "fontSize": UI_FONT_SM}),
        html.Div([
            html.B(f"Cost: {price:,} sats"),
            html.Span("  \u26a1", style={"fontSize": UI_FONT_LG}) if price <= 400 else "",
        ], style={"marginTop": "2px"}),
    ]

    if price > 10_000:
        children.append(html.Div(
            "\u26a0 Most users are unlikely to benefit from simulations "
            "this expensive. Consider using cached (bold) settings.",
            style={"fontSize": UI_FONT_SM, "color": KNIGHT_GOLD, "marginTop": "4px",
                   "fontStyle": "italic", "lineHeight": "1.3"}))

    return children, price


for _cost_pfx in ("hm", "dca", "ret", "sc", "cp"):
    _freq_id = f"{_cost_pfx}-mc-freq" if _cost_pfx == "hm" else f"{_cost_pfx}-freq"
    @callback(
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
    def _update_mc_cost(mc_enable, mc_freq, mc_years, mc_bins, mc_sims, mc_window,
                        mc_start_yr, mc_entry_q, mc_model_src, _tab=_cost_pfx):
        children, price = _mc_cost_display(mc_years, mc_start_yr, entry_q=mc_entry_q,
                                           mc_sims=mc_sims, mc_freq=mc_freq,
                                           mc_bins=mc_bins, tab=_tab,
                                           model_key=mc_model_src or "bub")
        return children, price
