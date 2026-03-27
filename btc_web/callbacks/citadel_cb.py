"""Citadel Planner chart callback — 7-output MC sandwich pattern."""

import dash
from dash import Input, Output, State, ctx, callback

import _app_ctx
from callbacks.coerce import _ci, _cf
from callbacks.mc_helpers import _mc_setup, _mc_finalize
from utils import _get_citadel_fig

# ── SCF body visibility toggle ──────────────────────────────────────────────
_app_ctx.app.clientside_callback(
    "function(v) { return (v && v.length) ? {} : {display:'none'}; }",
    Output("cp-scf-body", "style"),
    Input("cp-scf-enable", "value"),
)


@callback(
    Output("citadel-graph", "figure"),
    Output("cp-mc-results", "data"),
    Output("cp-mc-status", "children"),
    Output("cp-mc-rendered-key", "data"),
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Output("mc-save-tab", "data", allow_duplicate=True),
    Output("cp-mc-unblocked", "data"),
    Input("main-tabs",           "active_tab"),
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
    # ── MC States ──
    State("btc-price-store",     "data"),
    State("cp-mc-results",       "data"),
    State("mc-pay-token",        "data"),
    State("cp-mc-unblocked",     "data"),
    State("cp-mc-rendered-key",  "data"),
    prevent_initial_call='initial_duplicate',
)
def update_citadel(
    active_tab, run_clicks, _pay_trigger, _mc_loaded,
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
    palette_key,
    # MC states
    price_data, mc_cached, pay_token, mc_unblocked, mc_auth,
):
    """Citadel Planner chart callback."""
    # Skip if another tab is active (tab switch away, or initial load on different tab)
    if active_tab != "citadel" and ctx.triggered_id in ("main-tabs", None):
        raise dash.exceptions.PreventUpdate

    # Only run simulation when Run button clicked, payment trigger fires,
    # or tab becomes active (for loading cached default chart)
    if ctx.triggered_id not in ("cp-run-btn", "mc-pay-trigger", "cp-mc-loaded",
                                 "main-tabs", None):
        raise dash.exceptions.PreventUpdate

    # Before first click (or on tab switch without click): load cached default
    if not run_clicks or ctx.triggered_id == "main-tabs":
        import plotly.graph_objects as go
        import plotly.io as pio
        try:
            from cache import get_citadel_cached
            # Default: bubble model, Q25%
            cached = get_citadel_cached("default:bub:q0.25")
            if cached and cached.get("figure"):
                fig = pio.from_json(cached["figure"])
                return (fig, dash.no_update, dash.no_update, dash.no_update,
                        dash.no_update, dash.no_update, dash.no_update)
        except Exception:
            pass
        # Fallback: show instructions
        fig = go.Figure()
        fig.add_annotation(
            text="Configure your settings, then click<br><b>Run Simulation</b>",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=18, color="#888"),
        )
        fig.update_layout(
            template="plotly_white",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            height=500, margin=dict(t=40, b=40),
        )
        return (fig, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update)

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
        start_stack     = _cf(stack, 1.0),
        use_lots        = bool(use_lots),
        lots            = lots_data or [],
        # Cash
        cash_initial    = _cf(cash_init, 50000, lo=0),
        cash_rate       = _cf(cash_rate, 4.0, lo=0),
        # Reserves
        res_short_init  = _cf(res_short_init, 50000, lo=0),
        res_short_rate  = _cf(res_short_rate, 5.0, lo=0),
        res_short_vol   = _cf(res_short_vol, 2.0, lo=0),
        res_med_init    = _cf(res_med_init, 100000, lo=0),
        res_med_rate    = _cf(res_med_rate, 4.5, lo=0),
        res_med_vol     = _cf(res_med_vol, 8.0, lo=0),
        res_long_init   = _cf(res_long_init, 50000, lo=0),
        res_long_rate   = _cf(res_long_rate, 4.0, lo=0),
        res_long_vol    = _cf(res_long_vol, 15.0, lo=0),
        # Investments
        inv_eq_init     = _cf(inv_eq_init, 200000, lo=0),
        inv_eq_rate     = _cf(inv_eq_rate, 10.0, lo=0),
        inv_eq_vol      = _cf(inv_eq_vol, 16.0, lo=0),
        inv_bd_init     = _cf(inv_bd_init, 100000, lo=0),
        inv_bd_rate     = _cf(inv_bd_rate, 5.0, lo=0),
        inv_bd_vol      = _cf(inv_bd_vol, 7.0, lo=0),
        # Spending
        monthly_spend   = _cf(spend, 5000, lo=0),
        inflation       = _cf(infl, 4.0, lo=0, hi=100),
        spend_growth    = _cf(spend_growth, 0.0, lo=0, hi=100),
        # High-Q trigger
        high_q_trigger  = _cf(high_q_thresh, 95, lo=1, hi=99),
        high_q_mode     = high_q_mode or "gradual",
        high_q_rate     = _cf(high_q_rate, 2.0, lo=0.1, hi=100),
        high_q_dur      = _ci(high_q_dur, 6, lo=1, hi=120),
        high_q_split_cash = _cf(high_q_split_cash, 20, lo=0, hi=100),
        high_q_split_rs = _cf(high_q_split_rs, 20, lo=0, hi=100),
        high_q_split_rm = _cf(high_q_split_rm, 20, lo=0, hi=100),
        high_q_split_rl = _cf(high_q_split_rl, 10, lo=0, hi=100),
        high_q_split_eq = _cf(high_q_split_eq, 20, lo=0, hi=100),
        high_q_split_bd = _cf(high_q_split_bd, 10, lo=0, hi=100),
        # Low-Q trigger
        low_q_trigger   = _cf(low_q_thresh, 5, lo=1, hi=99),
        low_q_mode      = low_q_mode or "lump",
        low_q_rate      = _cf(low_q_rate, 10.0, lo=0.1, hi=100),
        low_q_dur       = _ci(low_q_dur, 1, lo=1, hi=120),
        low_q_split_cash = _cf(low_q_split_cash, 10, lo=0, hi=100),
        low_q_split_rs  = _cf(low_q_split_rs, 10, lo=0, hi=100),
        low_q_split_rm  = _cf(low_q_split_rm, 10, lo=0, hi=100),
        low_q_split_rl  = _cf(low_q_split_rl, 10, lo=0, hi=100),
        low_q_split_eq  = _cf(low_q_split_eq, 40, lo=0, hi=100),
        low_q_split_bd  = _cf(low_q_split_bd, 20, lo=0, hi=100),
        # Cooldown, floors
        lump_cooldown   = _ci(lump_cooldown, 12, lo=1),
        cash_floor      = _cf(cash_floor, 0, lo=0),
        res_short_floor = _cf(res_short_floor, 0, lo=0),
        res_med_floor   = _cf(res_med_floor, 0, lo=0),
        res_long_floor  = _cf(res_long_floor, 0, lo=0),
        cash_floor_growth = _cf(cash_floor_growth, 0, lo=0, hi=50),
        reserve_floor_growth = _cf(res_floor_growth, 0, lo=0, hi=50),
        # SCF
        scf_enabled     = "yes" in (scf_enable or []),
        scf_amount      = _cf(scf_amount, 0, lo=0),
        scf_type        = scf_type or "term",
        scf_rate        = _cf(scf_rate, 8.0, lo=0),
        scf_term        = _ci(scf_term, 60, lo=1),
        scf_repay_trigger = _cf(scf_trigger, 1.0, lo=0.1),
        # Simulation
        start_yr        = int(yr_range[0]),
        end_yr          = int(yr_range[1]),
        freq            = freq or "Monthly",
        price_model     = model_src or "bub",
        asset_return_model = asset_model or "lognormal",
        n_sims          = 1,
        selected_qs     = [float(sel_qs)] if sel_qs is not None else [0.25],
        disp_mode       = disp or "usd_per_asset",
        # Chart toggles
        log_y           = "log_y"      in toggles,
        annotate        = "annotate"   in toggles,
        show_legend     = "show_legend" in toggles,
        legend_pos      = legend_pos or "bottom-right",
        minor_grid      = "minor_grid" in toggles,
        palette         = palette_key or "default",
        **mc_p,
    ))

    # 3. MC finalize
    fig, store_val, status, rendered_key, show_modal, ub_val = _mc_finalize(
        "cp", fig, mc_result, mc_cached, mc_enable, mc_ok,
        is_free, blocked, mc_p["mc_years"], mc_p["mc_start_yr"],
        mc_p["mc_entry_q"], toggles, mc_stale=mc_p.get("mc_stale", False),
        mc_p=mc_p)

    return (fig, store_val, status, rendered_key, show_modal,
            "cp" if show_modal else dash.no_update, ub_val)
