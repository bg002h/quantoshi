"""Citadel Planner chart callback — 7-output MC sandwich pattern."""

import logging

import dash
from dash import Input, Output, State, ctx, callback, html, dcc

import _app_ctx

logger = logging.getLogger(__name__)
from callbacks.coerce import _ci, _cf
from tab_defaults import CITADEL
from callbacks.mc_helpers import _mc_setup, _mc_finalize
from utils import _get_citadel_fig

# ── SCF body visibility toggle ──────────────────────────────────────────────
_app_ctx.app.clientside_callback(
    "function(v) { return (v && v.length) ? {} : {display:'none'}; }",
    Output("cp-scf-body", "style"),
    Input("cp-scf-enable", "value"),
)


# ── Legend show/hide all ──────────────────────────────────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        var gd = document.getElementById('citadel-graph');
        if (gd) {
            var plot = gd.querySelector('.js-plotly-plot') || gd;
            if (plot && plot.data) {
                var vis = plot.data.map(function() { return true; });
                Plotly.restyle(plot, {visible: vis});
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("cp-legend-all", "n_clicks"),
    Input("cp-legend-all", "n_clicks"),
    prevent_initial_call=True,
)

_app_ctx.app.clientside_callback(
    """
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        var gd = document.getElementById('citadel-graph');
        if (gd) {
            var plot = gd.querySelector('.js-plotly-plot') || gd;
            if (plot && plot.data) {
                var vis = plot.data.map(function() { return 'legendonly'; });
                Plotly.restyle(plot, {visible: vis});
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("cp-legend-none", "n_clicks"),
    Input("cp-legend-none", "n_clicks"),
    prevent_initial_call=True,
)

# ── Dollar Asset Returns model info ──────────────────────────────────────────
@callback(
    Output("cp-asset-model-info", "children"),
    Output("cp-asset-model-info", "style"),
    Input("cp-asset-model", "value"),
    prevent_initial_call=True,
)
def show_asset_model_info(model):
    _style_visible = {"display": "block", "marginTop": "6px", "fontSize": "11px",
                      "color": "#aaa", "lineHeight": "1.4"}
    if model == "markov":
        return html.Div([
            html.Span("Historical Regimes", style={"fontWeight": "600", "color": "#e67e22"}),
            html.Span(" \u2014 ignores your input rates. Each asset class transitions "
                      "between bull/bear/neutral regimes based on historical data:"),
            html.Ul([
                html.Li("Equities: S&P 500 monthly returns"),
                html.Li("Bonds: AGG Bond ETF monthly returns"),
                html.Li("Treasuries: yield-to-total-return (short/med/long duration)"),
            ], style={"marginTop": "4px", "marginBottom": "0", "paddingLeft": "18px"}),
            html.Small("Regime transitions are independent of BTC price paths.",
                       style={"color": "#888", "fontStyle": "italic"}),
        ]), _style_visible
    return "", {"display": "none"}


@callback(
    Output("citadel-graph", "figure"),
    Output("cp-mc-results", "data"),
    Output("cp-mc-status", "children"),
    Output("cp-mc-rendered-key", "data"),
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Output("mc-save-tab", "data", allow_duplicate=True),
    Output("cp-mc-unblocked", "data"),
    Output("cp-yr-range", "value", allow_duplicate=True),
    Input("citadel-first-render", "data"),
    Input("cp-run-btn",          "n_clicks"),
    Input("mc-pay-trigger",      "data"),
    Input("cp-mc-loaded",        "data"),
    # ── Everything else is State (read on run, don't trigger) ──
    # Assets
    State("cp-stack",            "value"),
    State("cp-use-lots",         "value"),
    State("cp-cash-init",        "value"),
    State("cp-cash-rate",        "value"),
    State("cp-res-short-init",   "value"),
    State("cp-res-short-rate",   "value"),
    State("cp-res-short-vol",    "value"),
    State("cp-res-med-init",     "value"),
    State("cp-res-med-rate",     "value"),
    State("cp-res-med-vol",      "value"),
    State("cp-res-long-init",    "value"),
    State("cp-res-long-rate",    "value"),
    State("cp-res-long-vol",     "value"),
    State("cp-inv-eq-init",      "value"),
    State("cp-inv-eq-rate",      "value"),
    State("cp-inv-eq-vol",       "value"),
    State("cp-inv-bd-init",      "value"),
    State("cp-inv-bd-rate",      "value"),
    State("cp-inv-bd-vol",       "value"),
    # Spending
    State("cp-spend",            "value"),
    State("cp-infl",             "value"),
    State("cp-spend-growth",     "value"),
    # Rules: enable toggles
    State("cp-high-q-enable",    "value"),
    State("cp-low-q-enable",     "value"),
    # Rules: high-Q
    State("cp-high-q-thresh",    "value"),
    State("cp-high-q-mode",      "value"),
    State("cp-high-q-rate",      "value"),
    State("cp-high-q-dur",       "value"),
    State("cp-high-q-split-cash","value"),
    State("cp-high-q-split-rs",  "value"),
    State("cp-high-q-split-rm",  "value"),
    State("cp-high-q-split-rl",  "value"),
    State("cp-high-q-split-eq",  "value"),
    State("cp-high-q-split-bd",  "value"),
    # Rules: low-Q
    State("cp-low-q-thresh",     "value"),
    State("cp-low-q-mode",       "value"),
    State("cp-low-q-rate",       "value"),
    State("cp-low-q-dur",        "value"),
    State("cp-low-q-split-cash", "value"),
    State("cp-low-q-split-rs",   "value"),
    State("cp-low-q-split-rm",   "value"),
    State("cp-low-q-split-rl",   "value"),
    State("cp-low-q-split-eq",   "value"),
    State("cp-low-q-split-bd",   "value"),
    # Rules: cooldown, floors
    State("cp-lump-cooldown",    "value"),
    State("cp-cash-floor",       "value"),
    State("cp-res-short-floor",  "value"),
    State("cp-res-med-floor",    "value"),
    State("cp-res-long-floor",   "value"),
    State("cp-cash-floor-growth","value"),
    State("cp-res-floor-growth", "value"),
    # Rules: SCF
    State("cp-scf-enable",       "value"),
    State("cp-scf-amount",       "value"),
    State("cp-scf-type",         "value"),
    State("cp-scf-rate",         "value"),
    State("cp-scf-term",         "value"),
    State("cp-scf-trigger",      "value"),
    # Simulation
    State("cp-yr-range",         "value"),
    State("cp-freq",             "value"),
    State("cp-model-src",        "value"),
    State("cp-asset-model",      "value"),
    State("cp-qs",               "value"),
    State("cp-disp",             "value"),
    State("cp-toggles",          "value"),
    State("cp-legend-pos",       "value"),
    # Global
    State("effective-lots",      "data"),
    # MC controls
    State("cp-mc-enable",        "value"),
    State("cp-mc-bins",          "value"),
    State("cp-mc-regime",        "value"),
    State("cp-mc-sims",          "value"),
    State("cp-mc-years",         "value"),
    State("cp-mc-window",        "value"),
    State("cp-mc-start-yr",      "value"),
    State("cp-mc-entry-q",       "value"),
    State("cp-mc-model-src",     "value"),
    State("palette-store",       "data"),
    State("user-model-store",    "data"),
    # ── MC States ──
    State("btc-price-store",     "data"),
    State("cp-mc-results",       "data"),
    State("mc-pay-token",        "data"),
    State("cp-mc-unblocked",     "data"),
    State("cp-mc-rendered-key",  "data"),
    prevent_initial_call=True,
)
def update_citadel(
    _first_render, run_clicks, _pay_trigger, _mc_loaded,
    # Assets
    stack, use_lots,
    cash_init, cash_rate,
    res_short_init, res_short_rate, res_short_vol,
    res_med_init, res_med_rate, res_med_vol,
    res_long_init, res_long_rate, res_long_vol,
    inv_eq_init, inv_eq_rate, inv_eq_vol,
    inv_bd_init, inv_bd_rate, inv_bd_vol,
    # Spending
    spend, infl, spend_growth,
    # Enable toggles
    high_q_enable, low_q_enable,
    # High-Q
    high_q_thresh, high_q_mode, high_q_rate, high_q_dur,
    high_q_split_cash, high_q_split_rs, high_q_split_rm,
    high_q_split_rl, high_q_split_eq, high_q_split_bd,
    # Low-Q
    low_q_thresh, low_q_mode, low_q_rate, low_q_dur,
    low_q_split_cash, low_q_split_rs, low_q_split_rm,
    low_q_split_rl, low_q_split_eq, low_q_split_bd,
    # Cooldown, floors
    lump_cooldown,
    cash_floor, res_short_floor, res_med_floor, res_long_floor,
    cash_floor_growth, res_floor_growth,
    # SCF
    scf_enable, scf_amount, scf_type, scf_rate, scf_term, scf_trigger,
    # Simulation
    yr_range, freq, model_src, asset_model, sel_qs, disp, toggles, legend_pos,
    # Global
    lots_data,
    # MC controls
    mc_enable, mc_bins, mc_regime, mc_sims, mc_years, mc_window,
    mc_start_yr, mc_entry_q, mc_model_src,
    palette_key, user_model_store,
    # MC states
    price_data, mc_cached, pay_token, mc_unblocked, mc_auth,
):
    """Citadel Planner chart callback."""
    logger.debug("[CP-CB] triggered_id=%s, mc_enable=%s, run_clicks=%s", ctx.triggered_id, mc_enable, run_clicks)

    # Only run simulation when Run button clicked, payment trigger fires,
    # or tab becomes active (for loading cached default chart)
    if ctx.triggered_id not in ("cp-run-btn", "mc-pay-trigger", "cp-mc-loaded",
                                 "citadel-first-render", None):
        raise dash.exceptions.PreventUpdate

    # Before first click (or on tab switch): load cached default
    # mc-pay-trigger and cp-mc-loaded bypass — they should always run
    if (not run_clicks and ctx.triggered_id not in ("mc-pay-trigger", "cp-mc-loaded")) \
       or ctx.triggered_id == "citadel-first-render":
        import plotly.graph_objects as go
        import plotly.io as pio
        try:
            from cache import get_citadel_cached
            # Default: bubble model, Q25%
            cached = get_citadel_cached("default:bub:q0.25")
            if cached and cached.get("figure"):
                fig = pio.from_json(cached["figure"])
                return (fig, dash.no_update, dash.no_update, dash.no_update,
                        dash.no_update, dash.no_update, dash.no_update, dash.no_update)
        except Exception:
            pass
        # Fallback: compute default simulation live
        from tab_defaults import citadel_defaults
        _dp = citadel_defaults()
        _dp["selected_qs"] = [0.25]
        _dp["disp_mode"] = CITADEL["disp_mode"]
        _dp["log_y"] = True
        _dp["annotate"] = True
        _dp["show_legend"] = True
        _dp["minor_grid"] = True
        _dp["legend_pos"] = CITADEL["legend_pos"]
        _dp["palette"] = "default"
        _dp["active_models"] = ["bub"]
        fig, _ = _get_citadel_fig(_dp)
        return (fig, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update)

    toggles = toggles or []
    yr_range = yr_range or [2031, 2075]

    # 1. MC setup
    mc_ok, is_free, mc_p, blocked = _mc_setup(
        "cp", mc_enable, mc_years, mc_start_yr, mc_entry_q,
        mc_bins, mc_sims, freq, mc_window, spend, infl,
        mc_cached, _cf(price_data, 0), mc_regime, mc_unblocked, pay_token,
        mc_auth=mc_auth,
        stack=stack, amount_default=5000, infl_default=4.0, start_yr_default=2031,
        mc_model_src=mc_model_src)

    # 2. Build figure (merge MC params)
    fig, mc_result = _get_citadel_fig(dict(
        start_stack     = _cf(stack, CITADEL["start_stack"]),
        use_lots        = bool(use_lots),
        lots            = lots_data or [],
        # Cash
        cash_initial    = _cf(cash_init, CITADEL["cash_initial"], lo=0),
        cash_rate       = _cf(cash_rate, CITADEL["cash_rate"], lo=0),
        # Reserves
        res_short_init  = _cf(res_short_init, CITADEL["res_short_init"], lo=0),
        res_short_rate  = _cf(res_short_rate, CITADEL["res_short_rate"], lo=0),
        res_short_vol   = _cf(res_short_vol, CITADEL["res_short_vol"], lo=0),
        res_med_init    = _cf(res_med_init, CITADEL["res_med_init"], lo=0),
        res_med_rate    = _cf(res_med_rate, CITADEL["res_med_rate"], lo=0),
        res_med_vol     = _cf(res_med_vol, CITADEL["res_med_vol"], lo=0),
        res_long_init   = _cf(res_long_init, CITADEL["res_long_init"], lo=0),
        res_long_rate   = _cf(res_long_rate, CITADEL["res_long_rate"], lo=0),
        res_long_vol    = _cf(res_long_vol, CITADEL["res_long_vol"], lo=0),
        # Investments
        inv_eq_init     = _cf(inv_eq_init, CITADEL["inv_eq_init"], lo=0),
        inv_eq_rate     = _cf(inv_eq_rate, CITADEL["inv_eq_rate"], lo=0),
        inv_eq_vol      = _cf(inv_eq_vol, CITADEL["inv_eq_vol"], lo=0),
        inv_bd_init     = _cf(inv_bd_init, CITADEL["inv_bd_init"], lo=0),
        inv_bd_rate     = _cf(inv_bd_rate, CITADEL["inv_bd_rate"], lo=0),
        inv_bd_vol      = _cf(inv_bd_vol, CITADEL["inv_bd_vol"], lo=0),
        # Spending
        monthly_spend   = _cf(spend, CITADEL["monthly_spend"], lo=0),
        inflation       = _cf(infl, CITADEL["inflation"], lo=0, hi=100),
        spend_growth    = _cf(spend_growth, CITADEL["spend_growth"], lo=0, hi=100),
        # High-Q trigger (disabled → threshold 100 = never triggers)
        high_q_trigger  = _cf(high_q_thresh, CITADEL["high_q_trigger"], lo=1, hi=99) if high_q_enable else 100,
        high_q_mode     = high_q_mode or CITADEL["high_q_mode"],
        high_q_rate     = _cf(high_q_rate, CITADEL["high_q_rate"], lo=0.1, hi=100),
        high_q_dur      = _ci(high_q_dur, CITADEL["high_q_dur"], lo=1, hi=120),
        high_q_split_cash = _cf(high_q_split_cash, CITADEL["high_q_split_cash"], lo=0, hi=100),
        high_q_split_rs = _cf(high_q_split_rs, CITADEL["high_q_split_rs"], lo=0, hi=100),
        high_q_split_rm = _cf(high_q_split_rm, CITADEL["high_q_split_rm"], lo=0, hi=100),
        high_q_split_rl = _cf(high_q_split_rl, CITADEL["high_q_split_rl"], lo=0, hi=100),
        high_q_split_eq = _cf(high_q_split_eq, CITADEL["high_q_split_eq"], lo=0, hi=100),
        high_q_split_bd = _cf(high_q_split_bd, CITADEL["high_q_split_bd"], lo=0, hi=100),
        # Low-Q trigger
        # Low-Q trigger (disabled → threshold 0 = never triggers)
        low_q_trigger   = _cf(low_q_thresh, CITADEL["low_q_trigger"], lo=1, hi=99) if low_q_enable else 0,
        low_q_mode      = low_q_mode or CITADEL["low_q_mode"],
        low_q_rate      = _cf(low_q_rate, CITADEL["low_q_rate"], lo=0.1, hi=100),
        low_q_dur       = _ci(low_q_dur, CITADEL["low_q_dur"], lo=1, hi=120),
        low_q_split_cash = _cf(low_q_split_cash, CITADEL["low_q_split_cash"], lo=0, hi=100),
        low_q_split_rs  = _cf(low_q_split_rs, CITADEL["low_q_split_rs"], lo=0, hi=100),
        low_q_split_rm  = _cf(low_q_split_rm, CITADEL["low_q_split_rm"], lo=0, hi=100),
        low_q_split_rl  = _cf(low_q_split_rl, CITADEL["low_q_split_rl"], lo=0, hi=100),
        low_q_split_eq  = _cf(low_q_split_eq, CITADEL["low_q_split_eq"], lo=0, hi=100),
        low_q_split_bd  = _cf(low_q_split_bd, CITADEL["low_q_split_bd"], lo=0, hi=100),
        # Cooldown, floors
        lump_cooldown   = _ci(lump_cooldown, CITADEL["lump_cooldown"], lo=1),
        cash_floor      = _cf(cash_floor, CITADEL["cash_floor"], lo=0),
        res_short_floor = _cf(res_short_floor, CITADEL["res_short_floor"], lo=0),
        res_med_floor   = _cf(res_med_floor, CITADEL["res_med_floor"], lo=0),
        res_long_floor  = _cf(res_long_floor, CITADEL["res_long_floor"], lo=0),
        cash_floor_growth = _cf(cash_floor_growth, CITADEL["cash_floor_growth"], lo=0, hi=50),
        reserve_floor_growth = _cf(res_floor_growth, CITADEL["reserve_floor_growth"], lo=0, hi=50),
        # SCF
        scf_enabled     = "yes" in (scf_enable or []),
        scf_amount      = _cf(scf_amount, CITADEL["scf_amount"], lo=0),
        scf_type        = scf_type or CITADEL["scf_type"],
        scf_rate        = _cf(scf_rate, CITADEL["scf_rate"], lo=0),
        scf_term        = _ci(scf_term, CITADEL["scf_term"], lo=1),
        scf_repay_trigger = _cf(scf_trigger, CITADEL["scf_repay_trigger"], lo=0.1),
        # Simulation
        start_yr        = int(yr_range[0]),
        end_yr          = int(yr_range[1]),
        freq            = freq or CITADEL["freq"],
        price_model     = model_src or CITADEL["price_model"],
        asset_return_model = asset_model or CITADEL["asset_return_model"],
        n_sims          = 1,
        selected_qs     = [float(sel_qs)] if sel_qs is not None else list(CITADEL["selected_qs"]),
        disp_mode       = disp or CITADEL["disp_mode"],
        # Chart toggles
        log_y           = "log_y"      in toggles,
        annotate        = "annotate"   in toggles,
        show_legend     = "show_legend" in toggles,
        legend_pos      = legend_pos or CITADEL["legend_pos"],
        minor_grid      = "minor_grid" in toggles,
        palette         = palette_key or CITADEL["palette"],
        user_model      = user_model_store,
        **mc_p,
    ))

    # 3. MC finalize
    fig, store_val, status, rendered_key, show_modal, ub_val = _mc_finalize(
        "cp", fig, mc_result, mc_cached, mc_enable, mc_ok,
        is_free, blocked, mc_p["mc_years"], mc_p["mc_start_yr"],
        mc_p["mc_entry_q"], toggles, mc_stale=mc_p.get("mc_stale", False),
        mc_p=mc_p)

    # Update title with MC info (the figure was built before MC overlay ran)
    logger.debug("[CP-CB] title update: mc_ok=%s, mc_result type=%s, is_dict=%s, pending=%s",
                 mc_ok, type(mc_result).__name__, isinstance(mc_result, dict),
                 mc_result.get('_pending') if isinstance(mc_result, dict) else 'N/A')
    if mc_ok and mc_result and isinstance(mc_result, dict) and not mc_result.get("_pending"):
        mc_eq = mc_p.get("mc_entry_q", "")
        actual_sims = mc_result.get("n_sims", mc_p.get("mc_sims", "?"))
        old_title = fig.layout.title.text if fig.layout.title else ""
        if "MC entry" not in (old_title or ""):
            fig.update_layout(title_text=f"{old_title}  \u00b7  MC entry Q{mc_eq}% ({actual_sims} sims)")

    # Check if MC result is a Celery pending marker — enable polling
    if mc_result and isinstance(mc_result, dict) and mc_result.get("_pending"):
        task_id = mc_result.get("_celery_task_id")
        store_val = {"_celery_task_id": task_id, "_pending": True}
        status = html.Span("MC simulation computing in background...",
                           style={"color": "#b8860b", "fontSize": "12px"})
        return (fig, store_val, status, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update)

    # Nudge year range slider if MC starts before visible range
    yr_adjust = dash.no_update
    if mc_ok and mc_p.get("mc_start_yr"):
        mc_sy = int(mc_p["mc_start_yr"])
        if mc_sy < int(yr_range[0]):
            yr_adjust = [mc_sy, int(yr_range[1])]

    return (fig, store_val, status, rendered_key, show_modal,
            "cp" if show_modal else dash.no_update, ub_val, yr_adjust)


# ── Celery polling: enable/disable interval based on pending state ────────
@callback(
    Output("cp-celery-poll", "disabled"),
    Input("cp-mc-results", "data"),
    prevent_initial_call=True,
)
def _toggle_celery_poll(mc_cached):
    """Enable polling when Celery task is pending, disable when done."""
    if mc_cached and isinstance(mc_cached, dict) and mc_cached.get("_pending"):
        return False  # enable polling
    return True  # disable polling


# ── Celery task polling — check if background MC sim is done ──────────────
@callback(
    Output("cp-mc-results", "data", allow_duplicate=True),
    Output("cp-mc-status", "children", allow_duplicate=True),
    Input("cp-celery-poll", "n_intervals"),
    State("cp-mc-results", "data"),
    prevent_initial_call=True,
)
def _check_celery_task(n_intervals, mc_cached):
    """Periodically check if Celery MC task has completed."""
    if not mc_cached or not isinstance(mc_cached, dict):
        raise dash.exceptions.PreventUpdate
    task_id = mc_cached.get("_celery_task_id")
    if not task_id:
        raise dash.exceptions.PreventUpdate

    try:
        from celery_app import celery_app
        result = celery_app.AsyncResult(task_id)
        if result.ready():
            sim_result = result.get(timeout=5)
            return sim_result, html.Span(
                "\u2705 MC simulation complete — click Run to render fan bands",
                style={"color": "#27ae60", "fontSize": "12px"})
        elif result.failed():
            return {"_failed": True}, html.Span(
                "\u274c MC simulation failed",
                style={"color": "#e74c3c", "fontSize": "12px"})
    except Exception:
        pass
    raise dash.exceptions.PreventUpdate
