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
    # ── Assets panel ──
    Input("cp-stack",            "value"),
    Input("cp-use-lots",         "value"),
    Input("cp-cash-init",        "value"),
    Input("cp-cash-rate",        "value"),
    Input("cp-res-short-init",   "value"),
    Input("cp-res-short-rate",   "value"),
    Input("cp-res-short-vol",    "value"),
    Input("cp-res-med-init",     "value"),
    Input("cp-res-med-rate",     "value"),
    Input("cp-res-med-vol",      "value"),
    Input("cp-res-long-init",    "value"),
    Input("cp-res-long-rate",    "value"),
    Input("cp-res-long-vol",     "value"),
    Input("cp-inv-eq-init",      "value"),
    Input("cp-inv-eq-rate",      "value"),
    Input("cp-inv-eq-vol",       "value"),
    Input("cp-inv-bd-init",      "value"),
    Input("cp-inv-bd-rate",      "value"),
    Input("cp-inv-bd-vol",       "value"),
    # ── Spending panel ──
    Input("cp-spend",            "value"),
    Input("cp-infl",             "value"),
    Input("cp-spend-growth",     "value"),
    # ── Rules panel: high-Q trigger ──
    Input("cp-high-q-thresh",    "value"),
    Input("cp-high-q-mode",      "value"),
    Input("cp-high-q-rate",      "value"),
    Input("cp-high-q-dur",       "value"),
    Input("cp-high-q-split-cash","value"),
    Input("cp-high-q-split-rs",  "value"),
    Input("cp-high-q-split-rm",  "value"),
    Input("cp-high-q-split-rl",  "value"),
    Input("cp-high-q-split-eq",  "value"),
    Input("cp-high-q-split-bd",  "value"),
    # ── Rules panel: low-Q trigger ──
    Input("cp-low-q-thresh",     "value"),
    Input("cp-low-q-mode",       "value"),
    Input("cp-low-q-rate",       "value"),
    Input("cp-low-q-dur",        "value"),
    Input("cp-low-q-split-cash", "value"),
    Input("cp-low-q-split-rs",   "value"),
    Input("cp-low-q-split-rm",   "value"),
    Input("cp-low-q-split-rl",   "value"),
    Input("cp-low-q-split-eq",   "value"),
    Input("cp-low-q-split-bd",   "value"),
    # ── Rules panel: cooldown, floors ──
    Input("cp-lump-cooldown",    "value"),
    Input("cp-cash-floor",       "value"),
    Input("cp-res-short-floor",  "value"),
    Input("cp-res-med-floor",    "value"),
    Input("cp-res-long-floor",   "value"),
    # ── Rules panel: SCF ──
    Input("cp-scf-enable",       "value"),
    Input("cp-scf-amount",       "value"),
    Input("cp-scf-type",         "value"),
    Input("cp-scf-rate",         "value"),
    Input("cp-scf-term",         "value"),
    Input("cp-scf-trigger",      "value"),
    # ── Simulation panel ──
    Input("cp-yr-range",         "value"),
    Input("cp-freq",             "value"),
    Input("cp-model-src",        "value"),
    Input("cp-qs",               "value"),
    Input("cp-disp",             "value"),
    Input("cp-toggles",          "value"),
    Input("cp-legend-pos",       "value"),
    # ── Global ──
    Input("effective-lots",      "data"),
    # ── MC controls ──
    Input("cp-mc-enable",        "value"),
    Input("cp-mc-bins",          "value"),
    Input("cp-mc-regime",        "value"),
    Input("cp-mc-sims",          "value"),
    Input("cp-mc-years",         "value"),
    Input("cp-mc-window",        "value"),
    Input("cp-mc-start-yr",      "value"),
    Input("cp-mc-entry-q",       "value"),
    Input("cp-mc-loaded",        "data"),
    Input("mc-pay-trigger",      "data"),
    Input("cp-mc-model-src",     "value"),
    Input("palette-store",       "data"),
    # ── MC States ──
    State("btc-price-store",     "data"),
    State("cp-mc-results",       "data"),
    State("mc-pay-token",        "data"),
    State("cp-mc-unblocked",     "data"),
    State("cp-mc-rendered-key",  "data"),
    prevent_initial_call='initial_duplicate',
)
def update_citadel(
    active_tab,
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
    # SCF
    scf_enable, scf_amount, scf_type, scf_rate, scf_term, scf_trigger,
    # Simulation
    yr_range, freq, model_src, sel_qs, disp, toggles, legend_pos,
    # Global
    lots_data,
    # MC controls
    mc_enable, mc_bins, mc_regime, mc_sims, mc_years, mc_window,
    mc_start_yr, mc_entry_q, _mc_loaded, _pay_trigger, mc_model_src,
    palette_key,
    # MC states
    price_data, mc_cached, pay_token, mc_unblocked, mc_auth,
):
    """Citadel Planner chart callback — 7-output MC sandwich."""
    if ctx.triggered_id == "main-tabs" and active_tab != "citadel":
        raise dash.exceptions.PreventUpdate

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
