"""Chart update callbacks — Bubble, Heatmap, DCA, Retire, Supercharge."""

import math

import dash
from dash import Input, Output, State, ctx, callback
import pandas as pd

import _app_ctx
from btc_core import yr_to_t, today_t, _find_lot_percentile
from tab_defaults import BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE
from callbacks.coerce import _ci, _cf
from callbacks.mc_helpers import (_mc_setup, _mc_finalize, _mc_status,
                                  _strip_free_paths)
from mc_cache import (MC_BINS, MC_SIMS, MC_FREQ,
                      MC_DEFAULT_YEARS, MC_DEFAULT_START_YR, MC_DEFAULT_ENTRY_Q)
from utils import (_get_bubble_fig, _get_dca_fig, _get_retire_fig,
                   _get_supercharge_fig, _get_heatmap_fig, _get_mc_heatmap_fig,
                   _nearest_quantile)


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks — chart updates
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("bubble-graph", "figure", allow_duplicate=True),
    Input("bub-qs",            "value"),
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
    Input("effective-lots",    "data"),
    Input("palette-store",     "data"),
    Input("draw-mode-store",   "data"),
    State("scan-active-rows",  "data"),
    State("scan-q",            "value"),
    State("user-model-store",  "data"),
    prevent_initial_call="initial_duplicate",
)
def update_bubble(sel_qs, toggles, bubble_toggles,
                  xscale, yscale, xrange, yrange,
                  n_future, ptsize, ptalpha, stack, show_stack, use_lots, legend_pos, model_show, lots_data,
                  palette_key, draw_state=None,
                  scan_active=None, scan_q_val=None, user_model_store=None):
    """Bubble + QR overlay chart callback — coerce inputs, build figure."""
    toggles        = toggles or []
    bubble_toggles = bubble_toggles or []
    yrange         = yrange or [0, 7]
    xrange         = xrange or [2012, 2030]

    # Scanner lines
    scanner_lines = []
    if scan_active and scan_q_val is not None:
        q_frac = float(scan_q_val) / 100.0
        for model_key in (scan_active or []):
            scanner_lines.append({"model": model_key, "q": q_frac})

    fig = _get_bubble_fig(dict(
        selected_qs = sel_qs or [],
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
        comp_color  = "#FFD700", comp_lw = 2.0,
        sup_color   = "#888888", sup_lw  = 1.5,
        active_models = model_show or [],
        palette = palette_key or "default",
        scanner_lines = scanner_lines,
        draw_mode  = (draw_state or {}).get("phase", "idle") if (draw_state or {}).get("phase", "idle") != "idle" else None,
        draw_point1 = (draw_state or {}).get("point1"),
        draw_point2 = (draw_state or {}).get("point2"),
        user_model = user_model_store,
    ))
    draw_active = (draw_state or {}).get("phase", "idle") not in ("idle", "showing_menu")
    if "chart_zoom" not in toggles or draw_active:
        fig.update_layout(dragmode=False)

    # Adjust zoom: 2x centered on the point being refined (compounds on successive adjusts)
    if (draw_state or {}).get("_adjust_zoom"):
        import math
        phase = (draw_state or {}).get("phase", "idle")
        pt = None
        if phase == "placing_p1" and (draw_state or {}).get("point1"):
            pt = draw_state["point1"]
        elif phase == "placing_p2" and (draw_state or {}).get("point2"):
            pt = draw_state["point2"]
        if pt:
            cx, cy = pt["t"], pt["price"]
            is_log_x = (xscale or "log") == "log"
            # Use saved zoom level if available (compounds), else start from slider range
            zoom_level = (draw_state or {}).get("_zoom_level", 0) + 1
            zoom_factor = 2 ** zoom_level  # 2x, 4x, 8x, ...
            # Base range from slider inputs
            x_lo, x_hi = float(xrange[0]), float(xrange[1])
            y_lo, y_hi = float(yrange[0]), float(yrange[1])  # log10 exponents
            if is_log_x:
                log_cx = math.log10(max(cx, 0.01))
                log_xlo = math.log10(max(x_lo, 0.01))
                log_xhi = math.log10(max(x_hi, 0.02))
                x_half = (log_xhi - log_xlo) / (2 * zoom_factor)
                fig.update_xaxes(range=[log_cx - x_half, log_cx + x_half])
            else:
                x_half = (x_hi - x_lo) / (2 * zoom_factor)
                fig.update_xaxes(range=[cx - x_half, cx + x_half])
            log_cy = math.log10(max(cy, 1e-10))
            y_half = (y_hi - y_lo) / (2 * zoom_factor)
            fig.update_yaxes(range=[log_cy - y_half, log_cy + y_half])

    return fig


@callback(
    Output("bub-yrange", "value", allow_duplicate=True),
    Input("bub-xrange",  "value"),
    Input("bub-auto-y",  "value"),
    Input("bub-yscale",  "value"),
    Input("bub-model-show", "value"),
    State("bub-qs",      "value"),
    prevent_initial_call=True,
)
def auto_bubble_yrange(xrange, auto_y, yscale, model_show, sel_qs):
    """Auto-fit bubble Y range to selected quantiles at current X range."""
    if not auto_y or not xrange:
        raise dash.exceptions.PreventUpdate
    xmin, xmax = int(xrange[0]), int(xrange[1])
    t_lo = max(yr_to_t(xmin, _app_ctx.M.genesis), 0.1)
    t_hi = yr_to_t(xmax, _app_ctx.M.genesis)

    # Base Y range from BM (if active) or first active quantized model
    if "bub" in (model_show or []):
        base_mdl = _app_ctx.DEFAULT_MODEL
    else:
        base_mdl = None
        for key in (model_show or []):
            mdl = _app_ctx.PRICE_MODELS.get(key)
            if mdl and mdl.quantized:
                base_mdl = mdl
                break
        if base_mdl is None:
            base_mdl = _app_ctx.DEFAULT_MODEL  # safe fallback

    qs = sorted([float(q) for q in (sel_qs or []) if float(q) in base_mdl.fits])
    if not qs:
        qs = sorted(base_mdl.fits.keys())
    p_lo = float(base_mdl.price_at(qs[0], t_lo))
    p_hi = float(base_mdl.price_at(qs[-1], t_hi))
    # Include secondary models (PL, S2F) in Y range if active
    for key in (model_show or []):
        mdl = _app_ctx.PRICE_MODELS.get(key)
        if mdl is None or mdl is _app_ctx.DEFAULT_MODEL:
            continue
        if mdl.quantized:
            mdl_qs = [q for q in qs if q in mdl.fits]
            if mdl_qs:
                p_lo = min(p_lo, float(mdl.price_at(mdl_qs[0], t_lo)))
                p_hi = max(p_hi, float(mdl.price_at(mdl_qs[-1], t_hi)))
        else:
            p_mid = float(mdl.price_at(0.5, t_hi))
            p_hi = max(p_hi, p_mid)
    # Cap Y at $100M unless extreme models (S2F, Exponential) are active
    _extreme = {"s2f", "exp"}
    y_cap = 20.0 if _extreme.intersection(model_show or []) else 9.0
    if (yscale or "log") == "log":
        y_lo = math.floor(math.log10(max(p_lo, 1e-10)) * 2) / 2 - 0.5
        y_hi = math.ceil( math.log10(max(p_hi, 1e-10)) * 2) / 2 + 0.5
        y_lo = max(-2.0, min(y_lo, 6.0))
        y_hi = min(y_cap, max(y_hi, 1.0))
    else:  # linear — floor near zero, ceiling at highest quantile + 10% headroom
        y_lo = -2.0
        y_hi = math.ceil(math.log10(max(p_hi * 1.1, 1e-10)) * 2) / 2
        y_hi = min(y_cap, max(y_hi, 1.0))
    return [round(y_lo, 1), round(y_hi, 1)]


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
    Input("main-tabs",    "active_tab"),
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
    prevent_initial_call=True,
)
def update_heatmap(active_tab, hm_model, entry_yr, entry_q, exit_range, exit_qs, mode,
                   b1, b2, c_lo, c_mid1, c_mid2, c_hi, grad,
                   vfmt, cell_fs, toggles, stack, use_lots, lots_data,
                   mc_enable, mc_amount, mc_infl, mc_bins, mc_regime, mc_sims, mc_years, mc_freq, mc_window,
                   mc_start_yr, mc_entry_q, _mc_loaded, _pay_trigger, model_show, mc_model_src,
                   live_price, mc_cached, pay_token, mc_auth, palette_key):
    if ctx.triggered_id == "main-tabs" and active_tab != "heatmap":
        raise dash.exceptions.PreventUpdate
    exit_range = exit_range or [entry_yr or 2025, (entry_yr or 2025) + 10]
    toggles    = toggles or []
    yr_now = pd.Timestamp.today().year
    hm_model = hm_model or "bub"

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


@callback(
    Output("dca-graph", "figure"),
    Output("dca-mc-results", "data"),
    Output("dca-mc-status", "children"),
    Output("dca-mc-rendered-key", "data"),
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Output("mc-save-tab", "data", allow_duplicate=True),
    Output("dca-mc-unblocked", "data"),
    Input("main-tabs",    "active_tab"),
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
    State("user-model-store",   "data"),
    prevent_initial_call=True,
)
def update_dca(active_tab, stack, use_lots, amount, freq, dca_infl, yr_range, disp, toggles, legend_pos, sel_qs, lots_data,
               sc_enable, sc_loan, sc_rate, sc_term, sc_type, sc_repeats,
               sc_entry_mode, sc_custom_price, sc_tax, sc_rollover,
               mc_enable, mc_bins, mc_regime, mc_sims, mc_years, mc_window,
               mc_start_yr, mc_entry_q, _mc_loaded, _pay_trigger, model_show, mc_model_src,
               price_data, mc_cached, pay_token, mc_unblocked, mc_auth, palette_key,
               user_model_store=None):
    if ctx.triggered_id == "main-tabs" and active_tab != "dca":
        raise dash.exceptions.PreventUpdate
    toggles    = toggles or []
    yr_range   = yr_range or [2024, 2034]
    live_price = _cf(price_data, 0)
    mc_ok, is_free, mc_p, blocked = _mc_setup(
        "dca", mc_enable, mc_years, mc_start_yr, mc_entry_q,
        mc_bins, mc_sims, freq, mc_window, amount, dca_infl,
        mc_cached, live_price, mc_regime, mc_unblocked, pay_token,
        mc_auth=mc_auth,
        stack=stack, amount_default=DCA["amount"], infl_default=float(DCA["inflation"]), start_yr_default=2026,
        mc_model_src=mc_model_src)
    model_show = model_show if model_show is not None else []
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
        show_today     = "show_today" in toggles,
        show_legend    = "show_legend" in toggles,
        legend_pos     = legend_pos or "outside",
        minor_grid     = "minor_grid" in toggles,
        selected_qs    = sel_qs or [],
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
        active_models  = [k for k in model_show if k not in _app_ctx.MODEL_SENTINELS],
        palette = palette_key or "default",
        user_model = user_model_store,
        **mc_p,
    ))
    fig, store_val, status, rendered_key, show_modal, ub_val = _mc_finalize(
        "dca", fig, mc_result, mc_cached, mc_enable, mc_ok,
        is_free, blocked, mc_p["mc_years"], mc_p["mc_start_yr"],
        mc_p["mc_entry_q"], toggles, mc_stale=mc_p.get("mc_stale", False),
        mc_p=mc_p)
    return (fig, store_val, status, rendered_key, show_modal,
            "dca" if show_modal else dash.no_update, ub_val)


@callback(
    Output("retire-graph", "figure"),
    Output("ret-mc-results", "data"),
    Output("ret-mc-status", "children"),
    Output("ret-mc-rendered-key", "data"),
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Output("mc-save-tab", "data", allow_duplicate=True),
    Output("ret-mc-unblocked", "data"),
    Input("main-tabs",    "active_tab"),
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
    State("user-model-store",   "data"),
    prevent_initial_call=True,
)
def update_retire(active_tab, stack, use_lots, wd, freq, yr_range, infl, disp, toggles, legend_pos, sel_qs, lots_data,
                  mc_enable, mc_bins, mc_regime, mc_sims, mc_years, mc_window,
                  mc_start_yr, mc_entry_q, _mc_loaded, _pay_trigger, model_show, mc_model_src,
                  price_data, mc_cached, pay_token, mc_unblocked, mc_auth, palette_key,
                  user_model_store=None):
    if ctx.triggered_id == "main-tabs" and active_tab != "retire":
        raise dash.exceptions.PreventUpdate
    toggles  = toggles or []
    yr_range = yr_range or [RETIRE["start_yr"], RETIRE["end_yr"]]
    mc_ok, is_free, mc_p, blocked = _mc_setup(
        "ret", mc_enable, mc_years, mc_start_yr, mc_entry_q,
        mc_bins, mc_sims, freq, mc_window, wd, infl,
        mc_cached, _cf(price_data, 0), mc_regime, mc_unblocked, pay_token,
        mc_auth=mc_auth,
        stack=stack, amount_default=RETIRE["wd_amount"], infl_default=RETIRE["inflation"], start_yr_default=RETIRE["start_yr"],
        mc_model_src=mc_model_src)
    model_show = model_show if model_show is not None else []
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
        show_legend  = "show_legend" in toggles,
        legend_pos   = legend_pos or "outside",
        minor_grid   = "minor_grid" in toggles,
        selected_qs  = sel_qs or [],
        lots         = lots_data or [],
        show_qr      = "bub" in model_show,
        show_mc      = "mc" in model_show,
        active_models = [k for k in model_show if k not in _app_ctx.MODEL_SENTINELS],
        palette = palette_key or "default",
        user_model = user_model_store,
        **mc_p,
    ))
    fig, store_val, status, rendered_key, show_modal, ub_val = _mc_finalize(
        "ret", fig, mc_result, mc_cached, mc_enable, mc_ok,
        is_free, blocked, mc_p["mc_years"], mc_p["mc_start_yr"],
        mc_p["mc_entry_q"], toggles, mc_stale=mc_p.get("mc_stale", False),
        mc_p=mc_p)
    return (fig, store_val, status, rendered_key, show_modal,
            "ret" if show_modal else dash.no_update, ub_val)


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
    Input("main-tabs",       "active_tab"),
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
    State("viewport-width",    "data"),
    State("user-model-store",  "data"),
    prevent_initial_call=True,
)
def update_supercharge(active_tab, stack, use_lots, start_yr,
                       d0, d1, d2, d3, d4,
                       freq, infl, sel_qs, mode,
                       wd, end_yr, target_yr, disp,
                       toggles, legend_pos, chart_layout, display_q, lots_data,
                       mc_enable, mc_bins, mc_regime, mc_sims, mc_years, mc_window,
                       mc_start_yr, mc_entry_q, _mc_loaded, _pay_trigger, model_show, mc_model_src,
                       price_data, mc_cached, pay_token, mc_unblocked, mc_auth, palette_key,
                       viewport_width, user_model_store=None):
    if ctx.triggered_id == "main-tabs" and active_tab != "supercharge":
        raise dash.exceptions.PreventUpdate
    delays  = [float(x) for x in [d0, d1, d2, d3, d4] if x is not None]
    toggles = toggles or []
    yr_now  = pd.Timestamp.today().year
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
    fig, mc_result = _get_supercharge_fig(dict(
        mode         = mode or "a",
        start_stack  = _cf(stack, SUPERCHARGE["start_stack"]),
        start_yr     = _ci(start_yr, SUPERCHARGE["start_yr"]),
        delays       = delays if delays else [0, 1, 2, 4, 8],
        freq         = freq or "Monthly",
        inflation    = _cf(infl, SUPERCHARGE["inflation"]),
        selected_qs  = sel_qs or [],
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
        active_models = [k for k in model_show if k not in _app_ctx.MODEL_SENTINELS],
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
    return (fig, store_val, status, rendered_key, show_modal,
            "sc" if show_modal else dash.no_update, ub_val)
