"""Callbacks: chart updates, UI interactions, share/snapshot, tab routing."""
#
# Sections:
#   Imports ........................ ~1-35
#   MC parameter helpers ........... ~36-90
#   Chart update callbacks ......... ~91-500
#     Bubble ....................... ~95
#     Heatmap ...................... ~200
#     DCA .......................... ~330
#     Retire ....................... ~400 (approx)
#     Supercharge .................. ~450 (approx)
#   MC match indicator ............. ~500-560
#   MC datapoint cap ............... ~570-610
#   Quantile panel collapse ........ ~610-660
#   MC cost display ................ ~660-730
#   MC payment flow ................ ~730-1000
#   SC loan info & toggles ......... ~1000-1370
#   Stack Tracker (lots CRUD) ...... ~1370-1590
#   MC simulation save/load ........ ~1590-1760
#   Live price ticker .............. ~1750-1775
#   Image export ................... ~1777-1830
#   Tab routing & navigation ....... ~1830-2250
#   Snapshot / share ............... ~2250-end

import json
import base64
import logging
import math
import os

logger = logging.getLogger(__name__)

import dash
from dash import dcc, html, Input, Output, State, ctx, callback, no_update
import dash_bootstrap_components as dbc
import pandas as pd

import _app_ctx
from btc_core import (_find_lot_percentile, fmt_price, yr_to_t, today_t,
                      leo_weighted_entry, qr_price)
from figures import FREQ_PPY
from utils import (_get_bubble_fig, _get_dca_fig, _get_retire_fig,
                   _get_supercharge_fig, _get_heatmap_fig, _get_mc_heatmap_fig,
                   _nearest_quantile, _fetch_btc_price)
from flask import request as flask_request
from snapshot import (_SNAPSHOT_CONTROLS, _CHECKLIST_OPTIONS,
                      _SNAP_PREFIX, _SNAP_PREFIX_V1, _SNAP_PREFIX_V2,
                      _encode_snapshot, _decode_snapshot, _decode_snapshot_v1,
                      _list_to_mask, _mask_to_list)
from layout import (_SPLASH_QUOTES, _SPLASH_QUOTES_JS,
                    _MC_CACHED_START_YRS, _MC_PRICE_CACHED, _MC_PRICE_LIVE,
                    _FAQ, _bold_opts, _MC_CACHED_YEARS,
                    _MC_CACHED_ENTRY_QS,
                    _MC_ENTRY_Q_OPTIONS, _MC_ENTRY_Q_OPTIONS_ADV,
                    _regime_options)
from mc_cache import (MC_YEARS_OPTIONS, MC_BINS, MC_SIMS, MC_FREQ,
                      MC_DEFAULT_YEARS, MC_DEFAULT_ENTRY_Q, MC_DEFAULT_START_YR)
import btcpay

def _format_lots_for_table(lots: list[dict]) -> list[dict]:
    """Format lot dicts for the Stack Tracker DataTable display."""
    return [
        {**l,
         "total_paid": fmt_price(l["btc"] * l["price"]),
         "pct_q":      f"Q{l['pct_q']*100:.2f}%"}
        for l in lots
    ]


# ── Type coercion helpers ───────────────────────────────────────────────────
# Use `is not None` instead of `or` to handle 0 as a valid value.

def _ci(val: object, default: int, lo: int | None = None, hi: int | None = None) -> int:
    """Coerce to int. Returns default when val is None."""
    v = int(float(val)) if val is not None else default
    if lo is not None and v < lo:
        v = lo
    if hi is not None and v > hi:
        v = hi
    return v


def _cf(val: object, default: float, lo: float | None = None, hi: float | None = None) -> float:
    """Coerce to float. Returns default when val is None."""
    v = float(val) if val is not None else default
    if lo is not None and v < lo:
        v = lo
    if hi is not None and v > hi:
        v = hi
    return v


def _coerce_mc(mc_years, mc_start_yr, mc_entry_q, mc_bins, mc_sims, mc_freq,
               start_yr_default=MC_DEFAULT_START_YR):
    """Coerce all MC control values to clean ints/floats in one call."""
    return (_ci(mc_years, MC_DEFAULT_YEARS),
            _ci(mc_start_yr, start_yr_default),
            _cf(mc_entry_q, MC_DEFAULT_ENTRY_Q),
            _ci(mc_bins, MC_BINS),
            _ci(mc_sims, MC_SIMS),
            mc_freq or MC_FREQ)


# ══════════════════════════════════════════════════════════════════════════════
# MC parameter helper — single assembly point for all 4 MC-enabled tabs
# ══════════════════════════════════════════════════════════════════════════════

def _strip_free_paths(is_free: bool, mc_result: dict | None) -> dict | None:
    """Drop price_paths from MC result for free-tier users (saves memory)."""
    if is_free and mc_result and isinstance(mc_result, dict):
        return {k: v for k, v in mc_result.items() if k != "price_paths"}
    return mc_result



def _build_mc_params(*, mc_enable, mc_amount, mc_infl, mc_bins, mc_sims,
                     mc_years, mc_freq, mc_window, mc_start_yr, mc_entry_q,
                     mc_cached, mc_live_price, mc_regime=None,
                     amount_default=100, infl_default=4.0,
                     mc_start_stack=None):
    """Assemble MC simulation parameters from pre-coerced callback inputs.

    All MC control values (mc_bins, mc_sims, mc_years, mc_freq, mc_start_yr,
    mc_entry_q) must be pre-coerced by the caller.  Only mc_amount, mc_infl,
    and mc_start_stack are coerced here (tab-specific defaults).

    ``mc_regime`` is the list of *checked* (allowed) bin indices from the
    regime filter checklist.  Blocked bins = all bins minus allowed bins.
    If all bins are unchecked (empty list), treat as all-allowed to avoid
    a degenerate simulation with no valid transitions.
    """
    # Compute blocked bins from regime checklist (checked = allowed).
    # Guard: at least 1 bin must remain — if all unchecked, treat as all allowed.
    if mc_regime is not None and 0 < len(mc_regime) < mc_bins:
        blocked = sorted(set(range(mc_bins)) - set(mc_regime))
    else:
        blocked = []
    d = dict(
        mc_enabled    = bool(mc_enable),
        mc_amount     = _ci(mc_amount, amount_default, lo=0, hi=_app_ctx.MAX_USD),
        mc_infl       = _cf(mc_infl, infl_default),
        mc_bins       = mc_bins,
        mc_sims       = mc_sims,
        mc_years      = mc_years,
        mc_freq       = mc_freq,
        mc_window     = mc_window,
        mc_live_price = mc_live_price,
        mc_start_yr   = mc_start_yr,
        mc_entry_q    = mc_entry_q,
        mc_cached     = mc_cached,
        mc_blocked_bins = blocked,
    )
    if mc_start_stack is not None:
        v = round(_cf(mc_start_stack, 1.0, lo=0.0, hi=1000.0), 2)
        d["mc_start_stack"] = v
    return d


def _mc_payment_check(tab, mc_years, start_yr, entry_q, pay_token,
                      mc_bins=MC_BINS, mc_sims=MC_SIMS, mc_freq=MC_FREQ,
                      mc_auth=None):
    """Check if MC simulation is authorized (free tier, rendered_key, or paid).

    All MC control values must be pre-coerced by the caller.

    Authorization passes if ANY of these hold:
      1. Free tier (covered combo + default simulator settings)
      2. Previously authorized — rendered_key persists across visual-only
         changes (quantiles, display mode, log scale, year range, etc.)
      3. DEV/no-BTCPay + Run Simulation button was just clicked
      4. Valid payment token (production)
    """
    mc_yrs    = mc_years
    start_yr_ = start_yr
    entry_q_  = entry_q
    bins_     = mc_bins
    sims_     = mc_sims
    freq_     = mc_freq

    # 1. Free tier — always auto-approve
    if btcpay.is_free_tier(mc_yrs, start_yr_, entry_q_,
                           mc_bins=bins_, mc_sims=sims_, mc_freq=freq_):
        return True

    # 2. Previously authorized — rendered_key persists across visual-only
    #    changes.  The simulation already ran (paid or triggered); changing
    #    QR visual params should NOT require re-payment.
    if mc_auth and isinstance(mc_auth, dict):
        if (int(mc_auth.get("years", 0)) == mc_yrs
                and int(mc_auth.get("start_yr", 0)) == start_yr_
                and abs(float(mc_auth.get("entry_q", -1)) - entry_q_) < 0.05
                and int(mc_auth.get("bins", 0)) == bins_
                and int(mc_auth.get("sims", 0)) <= sims_
                and (mc_auth.get("freq") or MC_FREQ) == freq_):
            return True

    # 3. DEV / no BTCPay — require Run Simulation button click
    if not _app_ctx._HAS_BTCPAY or os.environ.get("DEV") == "1":
        return ctx.triggered_id == "mc-pay-trigger"

    # 4. Production — verify payment token
    if not pay_token:
        logger.info("MC payment required: %s %dyr start=%d q=%g (no token)",
                     tab, mc_yrs, start_yr_, entry_q_)
        return False
    valid = btcpay.verify_payment_token(
        pay_token.get("payment_token", ""),
        pay_token.get("invoice_id", ""),
        tab, mc_yrs,
    )
    if not valid:
        logger.warning("MC payment token invalid: %s %dyr", tab, mc_yrs)
    return valid


# ── MC callback sandwich ──────────────────────────────────────────────────────
# All MC-enabled chart callbacks (HM/DCA/Ret/SC) follow this 3-step pattern:
#   1. _mc_setup()    → payment auth, free-tier check, build MC params, ghost match
#   2. figure builder → builds QR traces + MC overlay, returns (fig, mc_result)
#   3. _mc_finalize() → strip free paths, rendered key, status badge, zoom sync
# This avoids duplicating ~30 lines of MC orchestration in each callback.
def _mc_setup(tab: str, mc_enable, mc_years, mc_start_yr, mc_entry_q,
              mc_bins, mc_sims, mc_freq, mc_window, mc_amount, mc_infl,
              mc_cached, live_price: float, mc_regime, mc_unblocked, pay_token,
              mc_auth=None,
              stack=None, amount_default: int = 100, infl_default: float = 0.0,
              start_yr_default: int = MC_DEFAULT_START_YR
              ) -> tuple[bool, bool, dict, tuple]:
    """Shared MC setup: coerce → payment check → free tier → build params → ghost."""
    mc_years_c, mc_start_yr_c, mc_entry_q_c, mc_bins_c, mc_sims_c, mc_freq_c = \
        _coerce_mc(mc_years, mc_start_yr, mc_entry_q, mc_bins, mc_sims, mc_freq,
                   start_yr_default)

    mc_ok = bool(mc_enable) and _mc_payment_check(
        tab, mc_years_c, mc_start_yr_c, mc_entry_q_c, pay_token,
        mc_bins=mc_bins_c, mc_sims=mc_sims_c, mc_freq=mc_freq_c,
        mc_auth=mc_auth)

    # ── Stale mode: keep cached overlay visible when MC params change ──
    # When payment check fails but cached MC data exists, override model
    # params with cached path_key values so the overlay function finds a
    # match and renders from cached data.  The existing mc-match clientside
    # indicator detects the param mismatch and shows "chart is stale".
    mc_stale = False
    _spk = None
    if not mc_ok and bool(mc_enable) and mc_cached and isinstance(mc_cached, dict):
        _spk = mc_cached.get("path_key")
        if _spk and _spk.get("tab", "") == tab:
            mc_years_c    = int(_spk.get("mc_years", MC_DEFAULT_YEARS))
            mc_start_yr_c = int(_spk.get("mc_start_yr", start_yr_default))
            mc_entry_q_c  = float(_spk.get("mc_entry_q", MC_DEFAULT_ENTRY_Q))
            mc_bins_c     = int(_spk.get("mc_bins", MC_BINS))
            mc_sims_c     = int(_spk.get("mc_sims", MC_SIMS))
            mc_freq_c     = _spk.get("mc_freq", MC_FREQ)
            mc_ok = True
            mc_stale = True
        else:
            _spk = None

    is_free = mc_ok and btcpay.is_free_tier(
        mc_years_c, mc_start_yr_c, mc_entry_q_c,
        mc_bins=mc_bins_c, mc_sims=mc_sims_c, mc_freq=mc_freq_c)
    mc_p = _build_mc_params(
        mc_enable=mc_ok, mc_amount=mc_amount, mc_infl=mc_infl,
        mc_bins=mc_bins_c, mc_sims=mc_sims_c, mc_years=mc_years_c,
        mc_freq=mc_freq_c,
        mc_window=_spk.get("mc_window") if mc_stale else mc_window,
        mc_start_yr=mc_start_yr_c, mc_entry_q=mc_entry_q_c,
        mc_cached=mc_cached, mc_live_price=live_price,
        mc_regime=None if mc_stale else mc_regime,
        amount_default=amount_default, infl_default=infl_default,
        mc_start_stack=stack,
    )
    if mc_stale:
        mc_p["mc_stale"] = True
        mc_p["mc_blocked_bins"] = list(_spk.get("mc_blocked_bins", []))
    if is_free:
        mc_p["mc_cached"] = None
        mc_p["mc_free_tier"] = True
    blocked = mc_p.get("mc_blocked_bins", [])
    if mc_ok and blocked:
        ghost = _ghost_match(mc_unblocked, mc_p, tab)
        if ghost:
            mc_p["mc_ghost_fan"] = ghost
    return mc_ok, is_free, mc_p, blocked


def _mc_finalize(tab: str, fig, mc_result, mc_cached, mc_enable, mc_ok: bool,
                 is_free: bool, blocked, mc_years, mc_start_yr, mc_entry_q,
                 toggles: list, has_rendered_key: bool = True,
                 mc_stale: bool = False, mc_p: dict | None = None) -> tuple:
    """Shared MC finalize: strip → rendered key → status → unblocked → zoom."""
    mc_result = _strip_free_paths(is_free, mc_result)

    if has_rendered_key:
        rendered_key = ({"years": _ci(mc_years, MC_DEFAULT_YEARS),
                         "start_yr": _ci(mc_start_yr, MC_DEFAULT_START_YR),
                         "entry_q": round(_cf(mc_entry_q, MC_DEFAULT_ENTRY_Q), 1),
                         "bins": int(mc_p.get("mc_bins", MC_BINS)) if mc_p else MC_BINS,
                         "sims": int(mc_p.get("mc_sims", MC_SIMS)) if mc_p else MC_SIMS,
                         "freq": (mc_p.get("mc_freq") or MC_FREQ) if mc_p else MC_FREQ}
                        if mc_ok else None)
    else:
        rendered_key = None
    store_val, status, show_modal = _mc_status(mc_result, mc_cached, mc_enable)
    ub_val = (dash.no_update if mc_stale
              else _unblocked_val(mc_ok, blocked, mc_result, mc_cached))
    if "chart_zoom" not in (toggles or []):
        fig.update_layout(dragmode=False)
    return fig, store_val, status, rendered_key, show_modal, ub_val


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks — chart updates
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("bubble-graph", "figure"),
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
    Input("effective-lots",    "data"),
)
def update_bubble(sel_qs, toggles, bubble_toggles,
                  xscale, yscale, xrange, yrange,
                  n_future, ptsize, ptalpha, stack, show_stack, use_lots, legend_pos, lots_data):
    """Bubble + QR overlay chart callback — coerce inputs, build figure."""
    toggles        = toggles or []
    bubble_toggles = bubble_toggles or []
    yrange         = yrange or [0, 7]
    xrange         = xrange or [2012, 2030]

    fig = _get_bubble_fig(dict(
        selected_qs = sel_qs or [],
        shade       = "shade"     in toggles,
        show_ols    = "show_ols"  in toggles,
        show_data   = "show_data"   in toggles,
        show_today  = "show_today"  in toggles,
        show_legend = "show_legend" in toggles,
        minor_grid  = "minor_grid" in toggles,
        show_comp   = "show_comp" in bubble_toggles,
        show_sup    = "show_sup"  in bubble_toggles,
        xscale      = xscale or "linear",
        yscale      = yscale or "log",
        xmin        = int(xrange[0]), xmax = int(xrange[1]),
        ymin        = 10 ** yrange[0], ymax = 10 ** yrange[1],
        n_future    = _ci(n_future, 0),
        pt_size     = _ci(ptsize, 3),
        pt_alpha    = _cf(ptalpha, 0.6),
        stack       = _cf(stack, 0),
        show_stack  = bool(show_stack),
        use_lots    = bool(use_lots),
        lots        = lots_data or [],
        legend_pos  = legend_pos or "outside",
        comp_color  = "#FFD700", comp_lw = 2.0,
        sup_color   = "#888888", sup_lw  = 1.5,
    ))
    if "chart_zoom" not in toggles:
        fig.update_layout(dragmode=False)
    return fig


@callback(
    Output("bub-yrange", "value", allow_duplicate=True),
    Input("bub-xrange",  "value"),
    Input("bub-auto-y",  "value"),
    Input("bub-yscale",  "value"),
    State("bub-qs",      "value"),
    prevent_initial_call=True,
)
def auto_bubble_yrange(xrange, auto_y, yscale, sel_qs):
    """Auto-fit bubble Y range to selected quantiles at current X range."""
    if not auto_y or not xrange:
        raise dash.exceptions.PreventUpdate
    xmin, xmax = int(xrange[0]), int(xrange[1])
    qs = sorted([float(q) for q in (sel_qs or []) if float(q) in _app_ctx.M.qr_fits])
    if not qs:
        qs = sorted(_app_ctx.M.qr_fits.keys())
    t_lo = max(yr_to_t(xmin, _app_ctx.M.genesis), 0.1)
    t_hi = yr_to_t(xmax, _app_ctx.M.genesis)
    p_lo = float(qr_price(qs[0],  t_lo, _app_ctx.M.qr_fits))
    p_hi = float(qr_price(qs[-1], t_hi, _app_ctx.M.qr_fits))
    if (yscale or "log") == "log":
        y_lo = math.floor(math.log10(max(p_lo, 1e-10)) * 2) / 2 - 0.5
        y_hi = math.ceil( math.log10(max(p_hi, 1e-10)) * 2) / 2 + 0.5
        y_lo = max(-2.0, min(y_lo, 6.0))
        y_hi = min(8.0,  max(y_hi, 1.0))
    else:  # linear — floor near zero, ceiling at highest quantile + 10% headroom
        y_lo = -2.0
        y_hi = math.ceil(math.log10(max(p_hi * 1.1, 1e-10)) * 2) / 2
        y_hi = min(8.0, max(y_hi, 1.0))
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


def _mc_status(mc_result, mc_cached, mc_enable):
    """Common MC result → (store_val, status, show_modal) for all tab callbacks."""
    store_val = mc_result if mc_result else dash.no_update
    if mc_result:
        status = f"Saved: {mc_result['created'][:19]}Z"
    elif mc_cached and bool(mc_enable):
        status = f"Using saved: {mc_cached.get('created', '?')[:19]}Z"
    else:
        status = ""
    return store_val, status, bool(mc_result)


def _ghost_match(unblocked, mc_p, tab):
    """Return unblocked cache data if its path_key matches current params
    (ignoring mc_blocked_bins), otherwise None."""
    if not unblocked or not isinstance(unblocked, dict):
        return None
    cpk = unblocked.get("path_key")
    if not cpk or cpk.get("tab") != tab:
        return None
    for k in ("mc_bins", "mc_sims", "mc_years", "mc_freq",
              "mc_window", "mc_start_yr", "mc_entry_q"):
        if cpk.get(k) != mc_p.get(k):
            return None
    return unblocked


def _unblocked_val(mc_ok, blocked, mc_result, mc_cached):
    """Compute value for unblocked-cache store output.

    Save effective result when no bins blocked; no_update otherwise.
    """
    if not mc_ok or blocked:
        return dash.no_update
    effective = mc_result or mc_cached
    if effective and isinstance(effective, dict) and "fan_btc" in effective:
        return effective
    return dash.no_update


@callback(
    Output("heatmap-graph",  "figure"),
    Output("hm-mc-graph",    "figure"),
    Output("hm-mc-results",  "data"),
    Output("hm-mc-status",   "children"),
    Output("hm-mc-panel",    "style"),
    Output("hm-swipe-indicator", "style"),
    Output("hm-mc-rendered-key", "data"),
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Output("mc-save-tab", "data", allow_duplicate=True),
    Input("main-tabs",    "active_tab"),
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
    State("btc-price-store", "data"),
    State("hm-mc-results",  "data"),
    State("mc-pay-token",   "data"),
    State("hm-mc-rendered-key", "data"),
    prevent_initial_call=True,
)
def update_heatmap(active_tab, entry_yr, entry_q, exit_range, exit_qs, mode,
                   b1, b2, c_lo, c_mid1, c_mid2, c_hi, grad,
                   vfmt, cell_fs, toggles, stack, use_lots, lots_data,
                   mc_enable, mc_amount, mc_infl, mc_bins, mc_regime, mc_sims, mc_years, mc_freq, mc_window,
                   mc_start_yr, mc_entry_q, _mc_loaded, _pay_trigger, model_show,
                   live_price, mc_cached, pay_token, mc_auth):
    if ctx.triggered_id == "main-tabs" and active_tab != "heatmap":
        raise dash.exceptions.PreventUpdate
    exit_range = exit_range or [entry_yr or 2025, (entry_yr or 2025) + 10]
    toggles    = toggles or []
    yr_now = pd.Timestamp.today().year

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
        color_mode   = _ci(mode, 0),
        b1           = _cf(b1, _app_ctx.M.CAGR_SEG_B1),
        b2           = _cf(b2, _app_ctx.M.CAGR_SEG_B2),
        c_lo         = c_lo   or _app_ctx.M.CAGR_SEG_C_LO,
        c_mid1       = c_mid1 or _app_ctx.M.CAGR_SEG_C_MID1,
        c_mid2       = c_mid2 or _app_ctx.M.CAGR_SEG_C_MID2,
        c_hi         = c_hi   or _app_ctx.M.CAGR_SEG_C_HI,
        n_disc       = _ci(grad, _app_ctx.M.CAGR_GRAD_STEPS),
        vfmt         = vfmt or "cagr",
        cell_font_size = _ci(cell_fs, 9),
        show_colorbar = "colorbar" in toggles,
        stack        = _cf(stack, 0),
        use_lots     = bool(use_lots),
        lots         = lots_data or [],
    )

    # QR heatmap (always)
    qr_fig = _get_heatmap_fig(dict(shared_params))

    # MC heatmap via sandwich helper
    mc_enabled = bool(mc_enable) and _app_ctx._HAS_MARKOV
    mc_ok, is_free, mc_p, blocked = _mc_setup(
        "hm", mc_enable, mc_years, mc_start_yr, mc_entry_q,
        mc_bins, mc_sims, mc_freq, mc_window, mc_amount, mc_infl,
        mc_cached, _cf(live_price, 0), mc_regime, None, pay_token,
        mc_auth=mc_auth,
        stack=stack, amount_default=100, infl_default=0.0,
        start_yr_default=yr_now)

    # Heatmap-specific: cap MC training window at start year for historical sims
    if mc_ok and not mc_p.get("mc_stale"):
        mc_sy = mc_p.get("mc_start_yr", yr_now)
        if mc_sy < yr_now:
            win = mc_p.get("mc_window")
            if win and isinstance(win, list) and len(win) >= 2:
                mc_p["mc_window"] = [win[0], min(win[1], mc_sy)]

    if mc_ok:
        mc_params = dict(shared_params, **mc_p,
                         live_price=_use_live(mc_p["mc_start_yr"], mc_p["mc_entry_q"]))
        mc_fig, mc_result = _get_mc_heatmap_fig(mc_params)
    else:
        mc_fig = dash.no_update
        mc_result = None

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
    model_show = model_show or ["qr", "mc"]
    mc_visible = mc_enabled and "mc" in model_show
    mc_panel_style = {} if mc_visible else {"display": "none"}
    indicator_style = ({"fontSize": "0.85rem", "color": "#6c757d", "userSelect": "none"}
                       if mc_visible
                       else {"display": "none"})

    if "chart_zoom" not in toggles:
        qr_fig.update_layout(dragmode=False)
        if mc_fig is not dash.no_update:
            mc_fig.update_layout(dragmode=False)

    return (qr_fig, mc_fig, store_val, status, mc_panel_style, indicator_style,
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
    State("btc-price-store","data"),
    State("dca-mc-results", "data"),
    State("mc-pay-token",   "data"),
    State("dca-mc-unblocked", "data"),
    State("dca-mc-rendered-key", "data"),
    prevent_initial_call=True,
)
def update_dca(active_tab, stack, use_lots, amount, freq, dca_infl, yr_range, disp, toggles, legend_pos, sel_qs, lots_data,
               sc_enable, sc_loan, sc_rate, sc_term, sc_type, sc_repeats,
               sc_entry_mode, sc_custom_price, sc_tax, sc_rollover,
               mc_enable, mc_bins, mc_regime, mc_sims, mc_years, mc_window,
               mc_start_yr, mc_entry_q, _mc_loaded, _pay_trigger, model_show,
               price_data, mc_cached, pay_token, mc_unblocked, mc_auth):
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
        stack=stack, amount_default=100, infl_default=0.0, start_yr_default=2026)
    model_show = model_show or ["qr", "mc"]
    fig, mc_result = _get_dca_fig(dict(
        start_stack    = _cf(stack, 0),
        use_lots       = bool(use_lots),
        amount         = _ci(amount, 100, lo=0, hi=_app_ctx.MAX_USD),
        freq           = freq or "Monthly",
        inflation      = _cf(dca_infl, 0),
        start_yr       = int(yr_range[0]),
        end_yr         = int(yr_range[1]),
        disp_mode      = disp or "btc",
        log_y          = "log_y"      in toggles,
        annotate       = "annotate"   in toggles,
        show_today     = "show_today" in toggles,
        show_legend    = "show_legend" in toggles,
        legend_pos     = legend_pos or "outside",
        minor_grid     = "minor_grid" in toggles,
        selected_qs    = sel_qs or [],
        lots           = lots_data or [],
        sc_enabled     = bool(sc_enable),
        sc_loan_amount = _cf(sc_loan, 0),
        sc_rate        = _cf(sc_rate, _app_ctx.SC_DEFAULT_RATE),
        sc_loan_type   = sc_type or "interest_only",
        sc_term_months = _cf(sc_term, 12),
        sc_repeats     = _ci(sc_repeats, 0),
        sc_live_price   = live_price,
        sc_entry_mode   = sc_entry_mode or "live",
        sc_custom_price = _cf(sc_custom_price, _app_ctx.SC_DEFAULT_PRICE),
        sc_tax_rate     = _cf(sc_tax, 33, lo=0, hi=100) / 100.0,
        sc_rollover     = bool(sc_rollover),
        show_qr        = "qr" in model_show,
        show_mc        = "mc" in model_show,
        **mc_p,
    ))
    fig, store_val, status, rendered_key, show_modal, ub_val = _mc_finalize(
        "dca", fig, mc_result, mc_cached, mc_enable, mc_ok,
        is_free, blocked, mc_p["mc_years"], mc_p["mc_start_yr"],
        mc_p["mc_entry_q"], toggles, mc_stale=mc_p.get("mc_stale", False),
        mc_p=mc_p)
    return (fig, store_val, status, rendered_key, show_modal,
            "dca" if show_modal else dash.no_update, ub_val)


@callback(Output("dca-sc-body","style"), Input("dca-sc-enable","value"))
def _toggle_dca_sc_body(val):
    return {} if val else {"display": "none"}

for _mc_tog in ("dca", "ret", "hm", "sc"):
    @callback(Output(f"{_mc_tog}-mc-body","style"), Input(f"{_mc_tog}-mc-enable","value"))
    def _toggle_mc_body(val):
        return {} if val else {"display": "none"}

for _mc_adv in ("dca", "ret", "hm", "sc"):
    @callback(
        Output(f"{_mc_adv}-mc-adv-body", "style"),
        Output(f"{_mc_adv}-mc-entry-q", "options"),
        Input(f"{_mc_adv}-mc-advanced", "value"),
    )
    def _toggle_mc_adv(val):
        style = {} if val else {"display": "none"}
        opts = _MC_ENTRY_Q_OPTIONS_ADV if val else _MC_ENTRY_Q_OPTIONS
        return style, opts

for _mc_reg in ("dca", "ret", "hm", "sc"):
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
function(mc_enable, mc_years, mc_start_yr, mc_entry_q, rendered_key) {{
    var hide = {{display: "none"}};
    var base = {{fontSize: "10px", fontWeight: "600", textAlign: "center", marginTop: "4px"}};
    var noPremium = "{base_cls}";
    var premium = "{base_cls}{sep}mc-premium-chart";
    if (!mc_enable || !mc_enable.length) return ["", hide, hide, noPremium, hide, hide];
    if (!rendered_key) return [
        "\u26a0 Chart does not include MC overlay",
        Object.assign({{}}, base, {{color: "#c57600"}}),
        {{}},
        noPremium,
        hide,
        hide
    ];
    var yrs = parseInt(mc_years) || 10;
    var syr = parseInt(mc_start_yr) || 2026;
    var eq  = parseInt(mc_entry_q) || 50;
    if (yrs === rendered_key.years && syr === rendered_key.start_yr && eq === rendered_key.entry_q) {{
        return [
            "\u2713 Chart reflects current MC settings",
            Object.assign({{}}, base, {{color: "#1a8f3c"}}),
            hide,
            premium,
            {{}},
            hide
        ];
    }}
    return [
        "\u26a0 MC settings changed \u2014 chart is stale",
        Object.assign({{}}, base, {{color: "#c57600"}}),
        {{}},
        noPremium,
        hide,
        {{}}
    ];
}}
"""
for _mc_m in ("dca", "ret", "hm"):
    _wrap_id = {"dca": "dca-chart-wrap", "ret": "ret-chart-wrap", "hm": "hm-mc-panel"}[_mc_m]
    _base_cls = "hm-swipe-panel" if _mc_m == "hm" else ""
    _sep = " " if _base_cls else ""
    _app_ctx.app.clientside_callback(
        _MC_MATCH_JS_TPL.format(base_cls=_base_cls, sep=_sep),
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
        Input(f"{_mc_m}-mc-rendered-key", "data"),
    )

# MC restore button — revert controls to last cached simulation settings
for _rpfx in ("hm", "dca", "ret", "sc"):
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

# SC tab has no MC overlay
_app_ctx.app.clientside_callback(
    "function() { return ['', {}]; }",
    Output("sc-mc-match", "children"),
    Output("sc-mc-match", "style"),
    Input("sc-mc-enable", "value"),
)

# PPY display (steps/year) — clientside for instant feedback
_PPY_JS = """
function(freq) {
    var m = {Daily:"365/yr", Weekly:"52/yr", Monthly:"12/yr", Quarterly:"4/yr", Annually:"1/yr"};
    return m[freq] || "12/yr";
}
"""
_app_ctx.app.clientside_callback(_PPY_JS, Output("hm-mc-ppy","value"),  Input("hm-mc-freq","value"))

# Auto-scroll heatmap swipe container to MC panel when it becomes visible
_app_ctx.app.clientside_callback(
    """
    function(panelStyle) {
        var isHidden = panelStyle && panelStyle.display === "none";
        if (!isHidden) {
            var wrap = document.getElementById("hm-swipe-wrap");
            if (wrap) {
                setTimeout(function() {
                    wrap.scrollTo({ left: wrap.scrollWidth, behavior: "smooth" });
                }, 200);
            }
        }
        return "";
    }
    """,
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
for _mc_pfx in ("dca", "ret", "sc"):
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

def _calc_mc_cost(mc_years, start_yr, entry_q=50, mc_sims=800, mc_freq="Monthly",
                  mc_bins=5, tab="dca"):
    """Calculate MC simulation cost and tier info.

    Returns (price_sats: int, is_free: bool, is_cached: bool,
             tier_label: str, tier_color: str, tier_note: str,
             mc_years_c: int) where *mc_years_c* is the coerced value.
    """
    mc_years = _ci(mc_years, 10)
    start_yr = _ci(start_yr, 2026)
    entry_q  = round(_cf(entry_q, 50), 1)
    mc_sims  = _ci(mc_sims, _MC_BASE_SIMS)
    mc_bins  = _ci(mc_bins, _MC_BASE_BINS)
    mc_ppy   = FREQ_PPY.get(mc_freq or "Monthly", _MC_BASE_PPY)
    is_cached = (start_yr in _MC_CACHED_START_YRS
                 and mc_bins == _MC_BASE_BINS
                 and mc_sims <= _MC_BASE_SIMS
                 and (mc_freq or "Monthly") == "Monthly"
                 and entry_q in _MC_CACHED_ENTRY_QS
                 and mc_years in _MC_CACHED_YEARS)

    # Scale factor relative to baseline (800 sims, Monthly, 5×5 matrix)
    scale = ((mc_sims / _MC_BASE_SIMS) * (mc_ppy / _MC_BASE_PPY)
             * (mc_bins ** 2 / _MC_BASE_BINS ** 2))

    if is_cached:
        base_price = _MC_PRICE_CACHED.get(mc_years, 100)
        tier_label = "Cached"
        tier_color = "#1a8f3c"
        tier_note = "Pre-computed \u2022 instant"
    else:
        base_price = _MC_PRICE_LIVE.get(mc_years, 500)
        tier_label = "Live"
        tier_color = "#c57600"
        time_scale = scale * (mc_years / 10)  # baseline 1-3s calibrated for 10yr
        lo, hi = max(1, round(1 * time_scale)), max(1, round(3 * time_scale))
        tier_note = (f"Computed on demand \u2022 ~{lo}\u2013{hi}s" if lo < hi
                     else f"Computed on demand \u2022 ~{lo}s")

    price = int(base_price * scale)

    # Heatmap gets 50% discount
    if tab == "hm":
        price = int(price * 0.5)

    is_free = btcpay.is_free_tier(mc_years, start_yr, entry_q,
                                  mc_bins=mc_bins, mc_sims=mc_sims,
                                  mc_freq=mc_freq)
    if is_free:
        price = 0

    return price, is_free, is_cached, tier_label, tier_color, tier_note, mc_years


def _mc_cost_display(mc_years, start_yr, entry_q=50, mc_sims=800, mc_freq="Monthly",
                     mc_bins=5, tab="dca"):
    """Return cost display elements showing cached vs live pricing."""
    price, is_free, is_cached, tier_label, tier_color, tier_note, mc_years_c = \
        _calc_mc_cost(mc_years, start_yr, entry_q, mc_sims, mc_freq, mc_bins, tab)

    if is_free:
        return ([
            html.Div([
                html.Span("Free tier", style={"fontWeight": "bold", "color": "#1a8f3c"}),
                html.Span(f" \u2022 {mc_years_c} yr simulation", style={"color": "#555"}),
            ]),
            html.Div(tier_note, style={"color": "#888", "fontSize": "10px"}),
            html.Div(html.B("Cost: Free \u2713"),
                     style={"marginTop": "2px", "color": "#1a8f3c"}),
        ], 0)

    children = [
        html.Div([
            html.Span(f"{tier_label}", style={"fontWeight": "bold", "color": tier_color}),
            html.Span(f" \u2022 {mc_years_c} yr simulation", style={"color": "#555"}),
        ]),
        html.Div(tier_note, style={"color": "#888", "fontSize": "10px"}),
        html.Div([
            html.B(f"Cost: {price:,} sats"),
            html.Span("  \u26a1", style={"fontSize": "13px"}) if price <= 400 else "",
        ], style={"marginTop": "2px"}),
    ]

    if price > 10_000:
        children.append(html.Div(
            "\u26a0 Most users are unlikely to benefit from simulations "
            "this expensive. Consider using cached (bold) settings.",
            style={"fontSize": "10px", "color": "#b8860b", "marginTop": "4px",
                   "fontStyle": "italic", "lineHeight": "1.3"}))

    return children, price


for _cost_pfx in ("hm", "dca", "ret", "sc"):
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
        prevent_initial_call=True,
    )
    def _update_mc_cost(mc_enable, mc_freq, mc_years, mc_bins, mc_sims, mc_window,
                        mc_start_yr, mc_entry_q, _tab=_cost_pfx):
        children, price = _mc_cost_display(mc_years, mc_start_yr, entry_q=mc_entry_q,
                                           mc_sims=mc_sims, mc_freq=mc_freq,
                                           mc_bins=mc_bins, tab=_tab)
        return children, price


# ══════════════════════════════════════════════════════════════════════════════
# MC payment callbacks (BTCPay integration)
# ══════════════════════════════════════════════════════════════════════════════

# Map button IDs → tab short names used by btcpay/api
_MC_BTN_TO_TAB = {
    "dca-mc-run-btn": "dca",
    "ret-mc-run-btn": "ret",
    "hm-mc-run-btn":  "hm",
    "sc-mc-run-btn":  "sc",
}

_MC_QUANT_THRESHOLD = 50_000  # sats — trigger quant warning modal

@callback(
    Output("mc-pay-modal",   "is_open", allow_duplicate=True),
    Output("mc-pay-invoice", "data",    allow_duplicate=True),
    Output("mc-pay-info",    "children"),
    Output("mc-pay-status",  "children",   allow_duplicate=True),
    Output("mc-pay-poll",    "disabled",   allow_duplicate=True),
    Output("mc-pay-poll",    "n_intervals", allow_duplicate=True),
    Output("mc-pay-trigger", "data",    allow_duplicate=True),
    Output("mc-quant-modal", "is_open",    allow_duplicate=True),
    Output("mc-quant-cost-info", "children", allow_duplicate=True),
    Output("mc-quant-onchain-note", "children", allow_duplicate=True),
    *(Output(f"{pfx}-mc-run-status", "children", allow_duplicate=True)
      for pfx in ("dca", "ret", "hm", "sc")),
    Input("dca-mc-run-btn", "n_clicks"),
    Input("ret-mc-run-btn", "n_clicks"),
    Input("hm-mc-run-btn",  "n_clicks"),
    Input("sc-mc-run-btn",  "n_clicks"),
    State("dca-mc-years", "value"), State("dca-mc-start-yr", "value"),
    State("dca-mc-entry-q", "value"),
    State("ret-mc-years", "value"), State("ret-mc-start-yr", "value"),
    State("ret-mc-entry-q", "value"),
    State("hm-mc-years", "value"), State("hm-mc-start-yr", "value"),
    State("hm-mc-entry-q", "value"),
    State("sc-mc-years", "value"), State("sc-mc-start-yr", "value"),
    State("sc-mc-entry-q", "value"),
    State("mc-pay-trigger", "data"),
    *(State(f"{pfx}-mc-price-val", "data") for pfx in ("dca", "ret", "hm", "sc")),
    prevent_initial_call=True,
)
def _mc_payment_initiate(*args):
    """Handle Run Simulation button clicks — check free tier or create invoice."""
    # Determine which button was clicked
    triggered = ctx.triggered_id
    if triggered not in _MC_BTN_TO_TAB:
        raise dash.exceptions.PreventUpdate
    tab = _MC_BTN_TO_TAB[triggered]

    # Extract the relevant tab's MC params from states
    tab_idx = list(_MC_BTN_TO_TAB.keys()).index(triggered)
    # Layout: 4 Inputs, then (years, start_yr, entry_q) × 4 tabs, trigger, 4 prices
    state_base = 4  # skip the 4 button Inputs
    mc_years  = _ci(args[state_base + tab_idx * 3],     MC_DEFAULT_YEARS)
    start_yr  = _ci(args[state_base + tab_idx * 3 + 1], MC_DEFAULT_START_YR)
    entry_q   = _cf(args[state_base + tab_idx * 3 + 2], MC_DEFAULT_ENTRY_Q)
    cur_trigger = args[state_base + 12] or 0  # after 4×3 tab states
    # Price stores: 4 values after trigger
    price_vals = args[state_base + 13 : state_base + 17]
    tab_price = _ci(price_vals[tab_idx], 0)

    # Default outputs: modal closed, no change to per-tab status
    no_tab_status = [dash.no_update] * 4
    # Helper: base return with quant modal closed
    def _ret(*vals):
        """Insert quant-modal defaults (closed, no update) into return tuple."""
        # Original 7 outputs + 3 quant outputs + 4 tab statuses
        return vals[:7] + (False, dash.no_update, "") + vals[7:]

    # Quant-tier warning (>50k sats) — fires before payment/free checks
    if tab_price > _MC_QUANT_THRESHOLD:
        onchain_note = ("(Bitcoin on-chain payment may be necessary)"
                        if tab_price > 250_000 else "")
        return (dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                True, f"Estimated cost: {tab_price:,} sats", onchain_note,
                *no_tab_status)

    # If BTCPay not configured or DEV mode, just increment trigger (free mode)
    if not _app_ctx._HAS_BTCPAY or os.environ.get("DEV") == "1":
        return _ret(False, dash.no_update, dash.no_update,
                    dash.no_update, True, 0, cur_trigger + 1,
                    *no_tab_status)

    # Free tier check (bins/sims/freq not available here; uses defaults)
    if btcpay.is_free_tier(mc_years, start_yr, entry_q):
        tab_statuses = list(no_tab_status)
        tab_statuses[tab_idx] = html.Span("Free tier", style={"color": "#1a8f3c"})
        return _ret(False, dash.no_update, dash.no_update,
                    dash.no_update, True, 0, cur_trigger + 1,
                    *tab_statuses)

    # Create invoice directly via btcpay module
    is_cached = btcpay.is_cached_request(start_yr)
    try:
        result = btcpay.create_invoice(tab, mc_years, is_cached)
    except Exception:
        tab_statuses = list(no_tab_status)
        tab_statuses[tab_idx] = html.Span(
            "Payment service unavailable", style={"color": "#c00"})
        return _ret(False, dash.no_update, "",
                    "", True, 0, dash.no_update, *tab_statuses)

    # Open payment modal — clientside callback handles iframe vs QR display
    invoice_data = {
        "invoice_id": result["invoice_id"],
        "tab": tab,
        "mc_years": mc_years,
        "checkout_url": result.get("checkout_url", ""),
        "amount_sats": result.get("amount_sats", 0),
    }
    info = f"Cost: {invoice_data['amount_sats']} sats \u2022 {mc_years}yr MC simulation"

    return _ret(True, invoice_data, info,
                "Waiting for payment...", False, 0,
                dash.no_update, *no_tab_status)


# ── Quant warning modal — proceed / cancel ────────────────────────────────────

@callback(
    Output("mc-quant-modal", "is_open", allow_duplicate=True),
    Output("mc-pay-trigger", "data", allow_duplicate=True),
    Input("mc-quant-proceed", "n_clicks"),
    State("mc-pay-trigger", "data"),
    prevent_initial_call=True,
)
def _quant_proceed(n, cur_trigger):
    """User confirmed expensive simulation — increment trigger to run it."""
    return False, (cur_trigger or 0) + 1

@callback(
    Output("mc-quant-modal", "is_open", allow_duplicate=True),
    Input("mc-quant-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def _quant_cancel(n):
    return False


# ── Clientside polling: check invoice status every 3s ─────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(n, invoice_data, cur_trigger) {
        if (!invoice_data || !invoice_data.invoice_id) {
            return [window.dash_clientside.no_update, true,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }
        var inv = invoice_data;
        var url = "/api/mc/invoice/" + inv.invoice_id
                  + "?tab=" + inv.tab + "&mc_years=" + inv.mc_years;

        return fetch(url)
            .then(function(r) { return r.json(); })
            .then(function(d) {
                if (d.paid) {
                    // Payment confirmed — store token, close modal, trigger MC
                    var token = {
                        payment_token: d.payment_token,
                        invoice_id: inv.invoice_id,
                        tab: inv.tab,
                        mc_years: inv.mc_years
                    };
                    return [token, true, false,
                            "Payment confirmed!",
                            (cur_trigger || 0) + 1];
                }
                if (d.status === "Expired" || d.status === "Invalid") {
                    return [window.dash_clientside.no_update, true, false,
                            "Invoice " + d.status.toLowerCase() + ". Please try again.",
                            window.dash_clientside.no_update];
                }
                // Still waiting
                return [window.dash_clientside.no_update,
                        window.dash_clientside.no_update,
                        window.dash_clientside.no_update,
                        "Waiting for payment...",
                        window.dash_clientside.no_update];
            })
            .catch(function(e) {
                return [window.dash_clientside.no_update,
                        window.dash_clientside.no_update,
                        window.dash_clientside.no_update,
                        "Checking payment...",
                        window.dash_clientside.no_update];
            });
    }
    """,
    Output("mc-pay-token",   "data"),
    Output("mc-pay-poll",    "disabled"),
    Output("mc-pay-modal",   "is_open"),
    Output("mc-pay-status",  "children"),
    Output("mc-pay-trigger", "data"),
    Input("mc-pay-poll",     "n_intervals"),
    State("mc-pay-invoice",  "data"),
    State("mc-pay-trigger",  "data"),
    prevent_initial_call=True,
)


# ── Cancel payment modal ─────────────────────────────────────────────────────
@callback(
    Output("mc-pay-modal", "is_open",  allow_duplicate=True),
    Output("mc-pay-poll",  "disabled", allow_duplicate=True),
    Output("mc-pay-status","children", allow_duplicate=True),
    Input("mc-pay-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def _mc_payment_cancel(n):
    return False, True, ""


# ── Clientside: detect onion vs clearnet and set up payment display ──────────
_app_ctx.app.clientside_callback(
    """
    function(invoice_data) {
        var nu = window.dash_clientside.no_update;
        if (!invoice_data || !invoice_data.invoice_id) {
            return [{"display":"none"}, "about:blank", {"display":"none"}, null];
        }
        var isOnion = window.location.hostname.endsWith('.onion');
        if (isOnion) {
            return [{}, invoice_data.checkout_url, {"display":"none"}, null];
        }
        // Clearnet: hide iframe, fetch payment methods, show QR + text
        return fetch("/api/mc/invoice/" + invoice_data.invoice_id + "/payment")
            .then(function(r) { return r.json(); })
            .then(function(d) {
                return [{"display":"none"}, "about:blank", {}, d.methods || []];
            })
            .catch(function() {
                return [{"display":"none"}, "about:blank", {},
                        [{method:"error", destination:"Could not load payment methods",
                          amount:"", qr_svg:""}]];
            });
    }
    """,
    Output("mc-pay-iframe-wrap", "style"),
    Output("mc-pay-iframe",      "src"),
    Output("mc-pay-details",     "style"),
    Output("mc-pay-methods",     "data"),
    Input("mc-pay-invoice",      "data"),
    prevent_initial_call=True,
)


# ── Clientside: render active payment method (QR + dest + amount) ────────────
_app_ctx.app.clientside_callback(
    """
    function(methods, ln_clicks, chain_clicks) {
        var nu = window.dash_clientside.no_update;
        if (!methods || !methods.length) return ["", "", ""];
        var ctx = window.dash_clientside.callback_context;
        var triggered = (ctx && ctx.triggered && ctx.triggered.length)
                        ? ctx.triggered[0].prop_id : "";
        var wantLn = triggered.indexOf("chain-btn") === -1;
        var m;
        if (wantLn) {
            m = methods.find(function(x) { return x.method && x.method.indexOf("LN") !== -1; });
        } else {
            m = methods.find(function(x) { return x.method && x.method.indexOf("LN") === -1; });
        }
        if (!m) m = methods[0];
        var qr = m.qr_svg || "";
        var dest = m.destination || "";
        var amt = parseFloat(m.amount || "0");
        var info;
        if (wantLn) {
            var sats = Math.round(amt * 1e8);
            info = sats.toLocaleString() + " sats";
        } else {
            info = amt + " BTC";
        }
        return [qr, dest, info];
    }
    """,
    Output("mc-pay-qr",          "src"),
    Output("mc-pay-dest",        "children"),
    Output("mc-pay-amount-info", "children"),
    Input("mc-pay-methods",      "data"),
    Input("mc-pay-ln-btn",       "n_clicks"),
    Input("mc-pay-chain-btn",    "n_clicks"),
    prevent_initial_call=True,
)


# ── Clientside: toggle LN/on-chain button active state ──────────────────────
_app_ctx.app.clientside_callback(
    """
    function(ln_clicks, chain_clicks) {
        var ctx = window.dash_clientside.callback_context;
        var triggered = (ctx && ctx.triggered && ctx.triggered.length)
                        ? ctx.triggered[0].prop_id : "";
        var isChain = triggered.indexOf("chain-btn") !== -1;
        return [!isChain, !isChain, isChain, isChain];
    }
    """,
    Output("mc-pay-ln-btn",    "active"),
    Output("mc-pay-chain-btn", "outline"),
    Output("mc-pay-chain-btn", "active"),
    Output("mc-pay-ln-btn",    "outline"),
    Input("mc-pay-ln-btn",     "n_clicks"),
    Input("mc-pay-chain-btn",  "n_clicks"),
    prevent_initial_call=True,
)


# ── Clientside: copy payment destination to clipboard ────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(n) {
        var el = document.getElementById("mc-pay-dest");
        if (el && el.textContent) {
            navigator.clipboard.writeText(el.textContent);
        }
        return "";
    }
    """,
    Output("mc-pay-status", "children", allow_duplicate=True),
    Input("mc-pay-copy-btn", "n_clicks"),
    prevent_initial_call=True,
)


# ── Saylor Mode: first-time quote toast ──────────────────────────────────────
_SAYLOR_QUOTES = [
    "There is no second best.",
    "If you\u2019ve got a billion-dollar problem, you need a trillion-dollar solution.",
    "Buy bitcoin. Then go figure out what you sold.",
    "You don\u2019t need a Plan B. You just need more Bitcoin.",
    "Buy Bitcoin and wait. Not complicated.",
    "The best time to buy bitcoin was yesterday. The second best time is today.",
    "There is no top, because fiat has no bottom.",
    "I decided to put all my eggs in one basket, and that basket is Bitcoin.",
]
_SAYLOR_QUOTES_JS = json.dumps(_SAYLOR_QUOTES)

_app_ctx.app.clientside_callback(
    f"""
    function(val) {{
        var NU = window.dash_clientside.no_update;
        if (!val || !val.length) return NU;
        var WK = "wizard-flags";
        var isDev = (location.hostname !== "quantoshi.xyz" &&
                     !location.hostname.endsWith(".onion"));
        try {{
            var f = JSON.parse(localStorage.getItem(WK)) || {{}};
            var now = Date.now();
            var day = 24 * 3600 * 1000;
            if (f.saylor_toast_ts && (now - f.saylor_toast_ts < day) && !isDev) return NU;
            f.saylor_toast_ts = now;
            localStorage.setItem(WK, JSON.stringify(f));
        }} catch(e) {{ return NU; }}
        var quotes = {_SAYLOR_QUOTES_JS};
        var q = quotes[Math.floor(Math.random() * quotes.length)];
        var el = document.createElement("div");
        el.className = "ambient-toast";
        el.textContent = "\\u201c" + q + "\\u201d \\u2014 Michael Saylor";
        document.body.appendChild(el);
        setTimeout(function() {{
            if (el.parentNode) el.parentNode.removeChild(el);
        }}, 6500);
        return NU;
    }}
    """,
    Output("dca-sc-body", "style", allow_duplicate=True),
    Input("dca-sc-enable", "value"),
    prevent_initial_call="initial_duplicate",
)


@callback(Output("dca-sc-custom-price-row","style"), Input("dca-sc-entry-mode","value"))
def _toggle_custom_price_row(mode):
    return {} if mode == "custom" else {"display": "none"}


@callback(Output("dca-sc-rollover-row","style"), Input("dca-sc-type","value"))
def _toggle_rollover_row(loan_type):
    return {} if (loan_type or "interest_only") == "interest_only" else {"display": "none"}


@callback(
    Output("dca-sc-info","children"),
    Input("dca-amount",     "value"),
    Input("dca-freq",       "value"),
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
    State("btc-price-store","data"),
)
def update_sc_info(amount, freq, enabled, sc_loan, rate, term, loan_type, repeats,
                   entry_mode, custom_price, tax, rollover, price_data):
    if not enabled:
        return ""
    from _app_ctx import _compute_sc_loan

    ppy          = FREQ_PPY.get(freq or "Monthly", 12)
    amount       = _ci(amount, 100, lo=0, hi=_app_ctx.MAX_USD)
    principal    = _cf(sc_loan, 0)
    rate         = _cf(rate, 13.0)
    term         = _cf(term, 12)
    loan_type    = loan_type or "interest_only"
    sc_rollover  = bool(rollover) and loan_type == "interest_only"
    n_repeats    = _ci(repeats, 0)
    n_cycles     = 1 + n_repeats
    r            = rate / 100.0 / ppy
    term_periods = max(1, round(term * ppy / 12))
    entry_mode   = entry_mode or "live"
    tax_rate     = _cf(tax, 33, lo=0, hi=100) / 100.0
    live         = _cf(price_data, 0)

    principal, pmt, capped = _compute_sc_loan(principal, amount, r, term_periods, loan_type)

    if loan_type == "amortizing":
        total_interest = pmt * term_periods - principal
        type_lbl = "Amortizing"
    else:
        total_interest = pmt * term_periods
        type_lbl = "Interest-only (rollover)" if sc_rollover else "Interest-only"

    reduced = amount - pmt

    # Entry price for display
    if entry_mode == "live":
        ep = live
        ep_lbl = f"Live ticker ({fmt_price(live)})" if live > 0 else "Live ticker"
    elif entry_mode == "custom":
        ep = _cf(custom_price, _app_ctx.SC_DEFAULT_PRICE)
        ep_lbl = f"Custom ({fmt_price(ep)})"
    else:
        ep = 0.0
        ep_lbl = "Model price"

    lump_btc  = principal / ep if ep > 0 else None
    active_mo = n_cycles * int(term)

    # Tax cost line — tax applies only to the capital gain (sell_price − cost_basis),
    # not to the full sale proceeds.  Actual BTC sold depends on future price.
    if loan_type == "interest_only":
        basis_str = fmt_price(ep) if ep > 0 else "model price at cycle start"
        if sc_rollover:
            tax_lbl = (f"Tax @{tax_rate*100:.4g}%: on gain at simulation end "
                       f"(cost basis: {basis_str})")
        elif tax_rate > 0:
            tax_lbl = (f"Tax @{tax_rate*100:.4g}%: on gain at each cycle-end repayment "
                       f"(cost basis: {basis_str})")
        else:
            tax_lbl = None
    else:  # amortizing
        tax_lbl = f"Tax @{tax_rate*100:.4g}%: N/A — principal repaid in fiat (no BTC sold)"

    loan_lbl = fmt_price(principal)
    if capped:
        loan_lbl += f"  (capped — max for {fmt_price(amount)}/period DCA)"
    lines = [
        f"Loan: {loan_lbl}  \u00b7  {type_lbl}",
        f"Entry: {ep_lbl}",
        f"Payment: {fmt_price(pmt)}/period  \u2192  DCA: {fmt_price(reduced)}/period",
        f"Total interest/cycle: {fmt_price(total_interest)}  (over {int(term)} mo)",
    ]
    if lump_btc:
        cycle_note = "first cycle only @ entry price" if sc_rollover else "each cycle @ entry price"
        lines.append(f"Buys \u2248 {lump_btc:.5f} BTC  ({cycle_note})")
    if tax_lbl:
        lines.append(tax_lbl)
    lines.append(f"Cycles: {n_cycles} total  \u00b7  Loan active {active_mo} mo")
    return [html.Div(l) for l in lines]


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
    State("btc-price-store","data"),
    State("ret-mc-results", "data"),
    State("mc-pay-token",   "data"),
    State("ret-mc-unblocked", "data"),
    State("ret-mc-rendered-key", "data"),
    prevent_initial_call=True,
)
def update_retire(active_tab, stack, use_lots, wd, freq, yr_range, infl, disp, toggles, legend_pos, sel_qs, lots_data,
                  mc_enable, mc_bins, mc_regime, mc_sims, mc_years, mc_window,
                  mc_start_yr, mc_entry_q, _mc_loaded, _pay_trigger, model_show,
                  price_data, mc_cached, pay_token, mc_unblocked, mc_auth):
    if ctx.triggered_id == "main-tabs" and active_tab != "retire":
        raise dash.exceptions.PreventUpdate
    toggles  = toggles or []
    yr_range = yr_range or [2025, 2045]
    mc_ok, is_free, mc_p, blocked = _mc_setup(
        "ret", mc_enable, mc_years, mc_start_yr, mc_entry_q,
        mc_bins, mc_sims, freq, mc_window, wd, infl,
        mc_cached, _cf(price_data, 0), mc_regime, mc_unblocked, pay_token,
        mc_auth=mc_auth,
        stack=stack, amount_default=5000, infl_default=4.0, start_yr_default=2031)
    model_show = model_show or ["qr", "mc"]
    fig, mc_result = _get_retire_fig(dict(
        start_stack  = _cf(stack, 1.0),
        use_lots     = bool(use_lots),
        wd_amount    = _ci(wd, 5000, lo=0, hi=_app_ctx.MAX_USD),
        freq         = freq or "Monthly",
        start_yr     = int(yr_range[0]),
        end_yr       = int(yr_range[1]),
        inflation    = _cf(infl, 0),
        disp_mode    = disp or "btc",
        log_y        = "log_y"     in toggles,
        annotate     = "annotate"  in toggles,
        show_legend  = "show_legend" in toggles,
        legend_pos   = legend_pos or "outside",
        minor_grid   = "minor_grid" in toggles,
        selected_qs  = sel_qs or [],
        lots         = lots_data or [],
        show_qr      = "qr" in model_show,
        show_mc      = "mc" in model_show,
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
    State("btc-price-store", "data"),
    State("sc-mc-results",   "data"),
    State("mc-pay-token",   "data"),
    State("sc-mc-unblocked", "data"),
    State("sc-mc-rendered-key", "data"),
    prevent_initial_call=True,
)
def update_supercharge(active_tab, stack, use_lots, start_yr,
                       d0, d1, d2, d3, d4,
                       freq, infl, sel_qs, mode,
                       wd, end_yr, target_yr, disp,
                       toggles, legend_pos, chart_layout, display_q, lots_data,
                       mc_enable, mc_bins, mc_regime, mc_sims, mc_years, mc_window,
                       mc_start_yr, mc_entry_q, _mc_loaded, _pay_trigger, model_show,
                       price_data, mc_cached, pay_token, mc_unblocked, mc_auth):
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
        stack=stack, amount_default=5000, infl_default=4.0, start_yr_default=2031)
    # chart_layout is now a checklist list; legacy snapshots may send an int
    _cl = (2 if "shade" in (chart_layout or []) else 0) \
          if isinstance(chart_layout, list) \
          else (int(chart_layout) if chart_layout is not None else 2)
    model_show = model_show or ["qr", "mc"]
    fig, mc_result = _get_supercharge_fig(dict(
        mode         = mode or "a",
        start_stack  = _cf(stack, 1.0),
        start_yr     = _ci(start_yr, yr_now),
        delays       = delays if delays else [0, 1, 2, 4, 8],
        freq         = freq or "Monthly",
        inflation    = _cf(infl, 4.0),
        selected_qs  = sel_qs or [],
        chart_layout = _cl,
        display_q    = _cf(display_q, _nearest_quantile(0.5, _app_ctx._ALL_QS)),
        wd_amount    = _ci(wd, 5000, lo=0, hi=_app_ctx.MAX_USD),
        end_yr       = _ci(end_yr, 2075),
        disp_mode    = disp or "usd",
        log_y        = "log_y"      in toggles,
        annotate     = "annotate"   in toggles,
        show_legend  = "show_legend" in toggles,
        legend_pos   = legend_pos or "outside",
        minor_grid   = "minor_grid" in toggles,
        target_yr    = _ci(target_yr, 2060),
        lots         = lots_data or [],
        use_lots     = bool(use_lots),
        show_qr      = "qr" in model_show,
        show_mc      = "mc" in model_show,
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


@callback(
    Output("sc-mode-a-collapse", "is_open"),
    Output("sc-mode-b-collapse", "is_open"),
    Output("sc-depl-note-collapse", "is_open"),
    Input("sc-mode", "value"),
)
def toggle_sc_mode(mode):
    return mode == "a", mode == "b", mode == "a"


@callback(
    Output("sc-display-q-collapse", "is_open"),
    Input("sc-chart-layout", "value"),
)
def toggle_sc_display_q(layout):
    return "shade" not in (layout or [])


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks — Stack Tracker
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("lot-pct-preview", "children"),
    Input("lot-date",  "value"),
    Input("lot-price", "value"),
)
def preview_percentile(date_str, price):
    if not date_str or not price or float(price) <= 0:
        return ""
    try:
        t   = (pd.Timestamp(date_str) - _app_ctx.M.genesis).days / 365.25
        pct = _find_lot_percentile(t, float(price), _app_ctx.M.qr_fits)
        return f"Q{pct*100:.2f}%"
    except Exception:
        return ""


@callback(
    Output("lots-store",        "data"),
    Output("lots-table",        "data"),
    Output("lots-table",        "selected_rows"),
    Output("lots-summary",      "children"),
    Output("lots-import-status","children"),
    Input("lot-add-btn",        "n_clicks"),
    Input("lot-del-btn",        "n_clicks"),
    Input("lot-clear-btn",      "n_clicks"),
    Input("lots-import-upload", "contents"),
    State("lot-date",           "value"),
    State("lot-btc",            "value"),
    State("lot-price",          "value"),
    State("lot-notes",          "value"),
    State("lots-table",         "selected_rows"),
    State("lots-store",         "data"),
    prevent_initial_call=True,
)
def manage_lots(add_n, del_n, clear_n, import_contents,
                date_str, btc_amt, price_val, notes,
                selected_rows, lots_data):
    """CRUD for Stack Tracker lots (add/delete/clear/import JSON)."""
    triggered = ctx.triggered_id
    lots = list(lots_data or [])
    import_status = dash.no_update

    if triggered == "lot-add-btn":
        if not date_str or not btc_amt or not price_val:
            raise dash.exceptions.PreventUpdate
        try:
            btc   = float(btc_amt)
            price = float(price_val)
            if btc <= 0 or price <= 0:
                raise ValueError
            t     = (pd.Timestamp(date_str) - _app_ctx.M.genesis).days / 365.25
            pct_q = _find_lot_percentile(t, price, _app_ctx.M.qr_fits)
            lots.append({
                "date":  date_str,
                "btc":   round(btc, 8),
                "price": round(price, 2),
                "pct_q": round(pct_q, 6),
                "notes": (notes or "").strip(),
            })
            lots.sort(key=lambda l: l["date"])
        except Exception:
            raise dash.exceptions.PreventUpdate

    elif triggered == "lot-del-btn":
        if selected_rows:
            lots = [l for i, l in enumerate(lots) if i not in selected_rows]

    elif triggered == "lot-clear-btn":
        lots = []

    elif triggered == "lots-import-upload" and import_contents:
        try:
            _hdr, b64 = import_contents.split(",", 1)
            raw  = base64.b64decode(b64).decode("utf-8")
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("expected a JSON array")
            if len(data) > 5000:
                raise ValueError(f"too many lots ({len(data)}), max 5000")
            # recompute pct_q in case file came from a different model version
            parsed = []
            for row in data:
                if not isinstance(row, dict):
                    raise ValueError("each lot must be a JSON object")
                if not isinstance(row.get("date"), str):
                    raise ValueError("lot missing or invalid 'date'")
                if not isinstance(row.get("btc"), (int, float)):
                    raise ValueError("lot missing or invalid 'btc'")
                if not isinstance(row.get("price"), (int, float)):
                    raise ValueError("lot missing or invalid 'price'")
                t     = (pd.Timestamp(row["date"]) - _app_ctx.M.genesis).days / 365.25
                pct_q = _find_lot_percentile(t, float(row["price"]), _app_ctx.M.qr_fits)
                parsed.append({
                    "date":  row["date"],
                    "btc":   round(float(row["btc"]), 8),
                    "price": round(float(row["price"]), 2),
                    "pct_q": round(pct_q, 6),
                    "notes": str(row.get("notes", "")).strip()[:200],
                })
            parsed.sort(key=lambda l: l["date"])
            lots = parsed
            import_status = f"Imported {len(lots)} lot(s) ✓"
            logger.info("Lot import: %d lots", len(lots))
        except Exception as e:
            import_status = f"Import failed: {e}"
            logger.warning("Lot import failed: %s", e)
            raise dash.exceptions.PreventUpdate

    table_data = _format_lots_for_table(lots)
    return lots, table_data, [], _lots_summary(lots), import_status


@callback(
    Output("lots-table",   "data",    allow_duplicate=True),
    Output("lots-summary", "children", allow_duplicate=True),
    Input("lots-store",    "data"),
    prevent_initial_call=True,
)
def sync_table_on_load(lots_data):
    lots = lots_data or []
    table_data = _format_lots_for_table(lots)
    return table_data, _lots_summary(lots)


def _lots_summary(lots):
    """Generate summary text for lot display (total BTC, avg cost, P&L)."""
    if not lots:
        return "No lots."
    total_btc  = sum(l["btc"] for l in lots)
    total_paid = sum(l["btc"] * l["price"] for l in lots)
    avg_price  = total_paid / total_btc if total_btc else 0
    avg_pct    = sum(l["pct_q"] * l["btc"] for l in lots) / total_btc if total_btc else 0
    return (f"{len(lots)} lot(s)  |  {total_btc:.8g} BTC  |  "
            f"Avg {fmt_price(avg_price)}/BTC  |  "
            f"Total paid {fmt_price(total_paid)}  |  "
            f"Avg Q{avg_pct*100:.2f}%")


# ── Stack Tracker: clientside JSON export (data never leaves the browser) ─────
_app_ctx.app.clientside_callback(
    """
    function(n_clicks, lots_data) {
        if (!n_clicks) return window.dash_clientside.no_update;
        var data = lots_data || [];
        var json = JSON.stringify(data, null, 2);
        var blob = new Blob([json], {type: 'application/json'});
        var url  = URL.createObjectURL(blob);
        var a    = document.createElement('a');
        a.href     = url;
        a.download = 'btc_lots.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return window.dash_clientside.no_update;
    }
    """,
    Output("lots-export-dummy", "data"),
    Input("lots-export-btn",    "n_clicks"),
    State("lots-store",         "data"),
    prevent_initial_call=True,
)


# ── MC simulation save (clientside JSON download) ────────────────────────────
_MC_FILENAME_JS = """
    function _mcFilename(tab, mc_data) {
        var pk = mc_data.path_key || {};
        var ok = mc_data.overlay_key || {};
        var parts = ['mc', tab];
        if (pk.mc_start_yr || pk.start_yr)
            parts.push('yr' + (pk.mc_start_yr || pk.start_yr));
        if (pk.mc_years) parts.push(pk.mc_years + 'y');
        var eq = pk.mc_entry_q || pk.entry_q;
        if (eq) parts.push('q' + Math.round(eq));
        if (pk.mc_bins) parts.push(pk.mc_bins + 'b');
        if (pk.mc_sims) parts.push(pk.mc_sims + 's');
        if (pk.mc_freq) parts.push(pk.mc_freq.charAt(0).toLowerCase());
        if (ok.mc_amount) parts.push('$' + ok.mc_amount);
        if (ok.mc_infl) parts.push(ok.mc_infl + 'pctInfl');
        if (ok.start_stack) parts.push(ok.start_stack + 'btc');
        var blocked = pk.mc_blocked_bins || [];
        if (blocked.length) parts.push('no' + blocked.join('_'));
        return parts.join('_') + '.json';
    }
"""

for _mc_prefix in ("dca", "ret", "hm", "sc"):
    _app_ctx.app.clientside_callback(
        """
        function(n_clicks, mc_data) {
            """ + _MC_FILENAME_JS + """
            if (!n_clicks || !mc_data) return window.dash_clientside.no_update;
            var json = JSON.stringify(mc_data, null, 2);
            var blob = new Blob([json], {type: 'application/json'});
            var url  = URL.createObjectURL(blob);
            var a    = document.createElement('a');
            a.href     = url;
            a.download = _mcFilename('""" + _mc_prefix + """', mc_data);
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            return window.dash_clientside.no_update;
        }
        """,
        Output(f"{_mc_prefix}-mc-dl-dummy", "data"),
        Input(f"{_mc_prefix}-mc-dl-btn",    "n_clicks"),
        State(f"{_mc_prefix}-mc-results",   "data"),
        prevent_initial_call=True,
    )


# ── MC save-prompt modal: dismiss closes modal ──────────────────────────────
@callback(
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Input("mc-save-modal-dismiss", "n_clicks"),
    prevent_initial_call=True,
)
def _mc_modal_dismiss(n):
    return False


# ── MC save-prompt modal: save triggers download then closes ────────────────
_app_ctx.app.clientside_callback(
    """
    function(n_clicks, tab, dca_data, ret_data, hm_data, sc_data) {
        """ + _MC_FILENAME_JS + """
        if (!n_clicks) return [window.dash_clientside.no_update, true];
        var map = {dca: dca_data, ret: ret_data, hm: hm_data, sc: sc_data};
        var mc_data = map[tab];
        if (!mc_data) return [window.dash_clientside.no_update, false];
        var json = JSON.stringify(mc_data, null, 2);
        var blob = new Blob([json], {type: 'application/json'});
        var url  = URL.createObjectURL(blob);
        var a    = document.createElement('a');
        a.href     = url;
        a.download = _mcFilename(tab, mc_data);
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return [window.dash_clientside.no_update, false];
    }
    """,
    Output("mc-save-modal-dummy", "data"),
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Input("mc-save-modal-dl", "n_clicks"),
    State("mc-save-tab", "data"),
    State("dca-mc-results", "data"),
    State("ret-mc-results", "data"),
    State("hm-mc-results", "data"),
    State("sc-mc-results", "data"),
    prevent_initial_call=True,
)


# ── MC simulation load (upload JSON → store + set UI controls) ───────────────

_TAB_LABELS = {"dca": "DCA (tab 3)", "ret": "Retire (tab 4)",
               "hm": "Heatmap (tab 2)", "sc": "Supercharger (tab 5)"}

def _parse_mc_upload(contents, expected_tab=None):
    """Decode uploaded MC JSON, return (data, error_msg).

    If expected_tab is given, reject files saved from a different tab.
    Validates metadata.app == "Quantoshi" when metadata is present.
    """
    if not contents:
        return None, None
    _, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    data = json.loads(raw)
    # Reject files with metadata from a different app
    meta = data.get("metadata")
    if meta and meta.get("app") and meta["app"] != "Quantoshi":
        return None, f"Not a Quantoshi file (app: {meta['app']})."
    if "path_key" not in data and "params" not in data:
        return None, "Invalid MC file (missing path data)."
    file_tab = data.get("tab")
    if expected_tab and file_tab and file_tab != expected_tab:
        src = _TAB_LABELS.get(file_tab, file_tab)
        dst = _TAB_LABELS.get(expected_tab, expected_tab)
        return None, f"Wrong tab: this is a {src} simulation. Upload it on {dst}."
    return data, None


def _extract_mc_key_val(data, key, default=None):
    """Get a value from loaded data's path_key or overlay_key."""
    pk = data.get("path_key", {})
    ok = data.get("overlay_key", {})
    return pk.get(key, ok.get(key, default))


# ── MC upload callback factory ───────────────────────────────────────────────
# Each entry: (output_suffix, data_key_or_fallback_tuple, default, cast_int)
_MC_UPLOAD_FIELDS = {
    "dca": [
        ("years",    "mc_years",                None, False),
        ("start-yr", "mc_start_yr",             None, False),
        ("entry-q",  "mc_entry_q",              50,   True),
        ("window",   "mc_window",               None, False),
    ],
    "ret": [
        ("years",    "mc_years",                None, False),
        ("start-yr", "mc_start_yr",             None, False),
        ("entry-q",  "mc_entry_q",              50,   True),
        ("window",   "mc_window",               None, False),
    ],
    "hm": [
        ("years",    "mc_years",                None, False),
        ("start-yr", "mc_start_yr",             None, False),
        ("entry-q",  "mc_entry_q",              50,   True),
        ("window",   "mc_window",               None, False),
    ],
    "sc": [
        ("years",    "mc_years",                None, False),
        ("start-yr", "mc_start_yr",             None, False),
        ("entry-q",  "mc_entry_q",              50,   True),
        ("window",   "mc_window",               None, False),
    ],
}


def _register_mc_upload(prefix, fields):
    """Register an MC file upload callback for a given tab prefix."""
    outputs = [
        Output(f"{prefix}-mc-upload", "contents"),
        Output(f"{prefix}-mc-results", "data", allow_duplicate=True),
        Output(f"{prefix}-mc-upload-status", "children"),
        Output(f"{prefix}-mc-loaded", "data", allow_duplicate=True),
        Output(f"{prefix}-mc-enable", "value", allow_duplicate=True),
    ] + [
        Output(f"{prefix}-mc-{suffix}", "value", allow_duplicate=True)
        for suffix, _, _, _ in fields
    ]
    n_total = len(outputs)

    @callback(
        *outputs,
        Input(f"{prefix}-mc-upload", "contents"),
        State(f"{prefix}-mc-loaded", "data"),
        prevent_initial_call=True,
    )
    def _load_mc(contents, loaded_count):
        if not contents:
            raise dash.exceptions.PreventUpdate
        nu = dash.no_update
        err_tuple = (dash.no_update,) * n_total
        try:
            data, err = _parse_mc_upload(contents, expected_tab=prefix)
            if err:
                return err_tuple[:1] + (err,) + err_tuple[2:]
            extra = []
            for _, data_key, default, cast_int in fields:
                val = _extract_mc_key_val(data, data_key, default if default is not None else nu)
                if cast_int and val is not nu:
                    val = int(val)
                extra.append(val)
            return (None,  # clear upload so re-uploading same file works
                    data,
                    f"Loaded: {data.get('created', '?')[:19]}Z",
                    (loaded_count or 0) + 1,
                    ["yes"],
                    *extra)
        except Exception as e:
            return err_tuple[:1] + (f"Error: {e}",) + err_tuple[2:]


for _prefix, _fields in _MC_UPLOAD_FIELDS.items():
    _register_mc_upload(_prefix, _fields)


# ══════════════════════════════════════════════════════════════════════════════
# Callback — live BTC price ticker (Binance, refreshes every 5 min)
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("price-ticker",        "children"),
    Output("price-ticker-mobile", "children"),
    Output("btc-price-store",     "data"),
    Output("hm-entry-q",          "value", allow_duplicate=True),
    Output("hm-mc-entry-q",       "value", allow_duplicate=True),
    Output("dca-mc-entry-q",      "value", allow_duplicate=True),
    Input("price-interval", "n_intervals"),
    prevent_initial_call="initial_duplicate",
)
def update_price_ticker(_):
    price = _fetch_btc_price()
    if price is None:
        return "₿ —", "₿ —", no_update, no_update, no_update, no_update
    pct = _find_lot_percentile(today_t(_app_ctx.M.genesis), price, _app_ctx.M.qr_fits)
    pct_str = f"Q{pct*100:.1f}%" if pct is not None else "—"
    pct_val = round(pct * 100, 1) if pct is not None else no_update
    # Snap to nearest 10% for cache-aligned dropdowns (hm-mc, dca-mc)
    snapped_pct = max(10, min(90, round(pct * 10) * 10)) if pct is not None else no_update
    txt = f"₿ {fmt_price(price)}  ·  {pct_str}"
    txt_m = f"₿{fmt_price(price)}·{pct_str}"
    return txt, txt_m, price, pct_val, snapped_pct, snapped_pct


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks — image export (client-side, no kaleido/Chrome needed)
# Uses Plotly.downloadImage() which renders in the browser.
# ══════════════════════════════════════════════════════════════════════════════

_EXPORT_TABS = [
    ("bubble",  "bubble-graph"),
    ("heatmap", "heatmap-graph"),
    ("dca",     "dca-graph"),
    ("retire",  "retire-graph"),
    ("supercharge", "supercharge-graph"),
]

for _tab_id, _graph_id in _EXPORT_TABS:
    _app_ctx.app.clientside_callback(
        f"""
        function(n_clicks, fmt, fname, scale, figure, wmStore) {{
            if (!n_clicks) return window.dash_clientside.no_update;
            if (!figure)   return window.dash_clientside.no_update;
            var s = scale || 2;
            var fig = JSON.parse(JSON.stringify(figure));
            if (wmStore && fig.layout && fig.layout.images) {{
                var wmB64 = wmStore[String(s)];
                if (wmB64) {{
                    for (var i = 0; i < fig.layout.images.length; i++) {{
                        if (fig.layout.images[i].source &&
                            fig.layout.images[i].source.indexOf('data:image/png;base64,') === 0) {{
                            fig.layout.images[i].source = wmB64;
                            break;
                        }}
                    }}
                }}
            }}
            if ((fmt || 'png') === 'html') {{
                var fn = (fname || '{_tab_id}') + '.html';
                var html = '<!DOCTYPE html>\\n<html><head>'
                    + '<meta charset="utf-8">'
                    + '<title>' + fn + ' — Quantoshi</title>'
                    + '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"><\\/script>'
                    + '<style>body{{margin:0;background:#1a1a2e}}'
                    + '#chart{{width:100vw;height:100vh}}</style>'
                    + '</head><body>'
                    + '<div id="chart"></div><script>'
                    + 'Plotly.newPlot("chart",'
                    + JSON.stringify(fig.data) + ','
                    + JSON.stringify(fig.layout) + ','
                    + '{{responsive:true}});'
                    + '<\\/script></body></html>';
                var blob = new Blob([html], {{type: 'text/html'}});
                var a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = fn;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(a.href);
                return window.dash_clientside.no_update;
            }}
            Plotly.downloadImage(fig, {{
                format:   fmt   || 'png',
                width:    1920,
                height:   1080,
                scale:    s,
                filename: fname || '{_tab_id}'
            }});
            return window.dash_clientside.no_update;
        }}
        """,
        Output(f"{_tab_id}-dl-dummy", "data"),
        Input(f"{_tab_id}-export-btn", "n_clicks"),
        State(f"{_tab_id}-fmt",        "value"),
        State(f"{_tab_id}-fname",      "value"),
        State(f"{_tab_id}-scale",      "value"),
        State(f"{_tab_id}-graph",      "figure"),
        State("wm-b64-store",          "data"),
        prevent_initial_call=True,
    )



# ══════════════════════════════════════════════════════════════════════════════
# Callback — pathname-based tab routing (/1 … /6)
# ══════════════════════════════════════════════════════════════════════════════

_PATH_TO_TAB = {
    "/1": "bubble", "/2": "heatmap", "/3": "dca",
    "/4": "retire",  "/5": "supercharge", "/6": "stack", "/7": "faq",
}
_TAB_TO_PATH = {v: k for k, v in _PATH_TO_TAB.items()}

# Component IDs that belong to each tab (for single-tab share links)
_TAB_CONTROLS = {
    "bubble":      {"bub-qs","bub-xscale","bub-yscale","bub-xrange","bub-yrange",
                    "bub-toggles","bub-bubble-toggles","bub-n-future","bub-ptsize",
                    "bub-ptalpha","bub-stack","bub-show-stack","bub-use-lots","bub-auto-y",
                    "bub-legend-pos"},
    "heatmap":     {"hm-entry-yr","hm-entry-q","hm-exit-range","hm-exit-qs","hm-mode",
                    "hm-b1","hm-b2","hm-c-lo","hm-c-mid1","hm-c-mid2","hm-c-hi",
                    "hm-grad","hm-vfmt","hm-cell-fs","hm-toggles","hm-stack","hm-use-lots",
                    "hm-model-show"},
    "dca":         {"dca-stack","dca-use-lots","dca-amount","dca-freq","dca-freq-unlock",
                    "dca-infl","dca-yr-range",
                    "dca-disp","dca-toggles","dca-qs",
                    "dca-sc-enable","dca-sc-loan","dca-sc-rate","dca-sc-term",
                    "dca-sc-type","dca-sc-repeats",
                    "dca-sc-entry-mode","dca-sc-custom-price","dca-sc-tax",
                    "dca-sc-rollover","dca-legend-pos","dca-model-show"},
    "retire":      {"ret-stack","ret-use-lots","ret-wd","ret-freq","ret-freq-unlock",
                    "ret-yr-range",
                    "ret-infl","ret-disp","ret-toggles","ret-legend-pos","ret-qs",
                    "ret-model-show"},
    "supercharge": {"sc-stack","sc-use-lots","sc-start-yr","sc-d0","sc-d1","sc-d2",
                    "sc-d3","sc-d4","sc-freq","sc-freq-unlock","sc-infl","sc-qs",
                    "sc-mode","sc-wd",
                    "sc-end-yr","sc-target-yr","sc-disp","sc-toggles","sc-legend-pos",
                    "sc-chart-layout","sc-display-q","sc-model-show"},
    "stack":       set(),
    "faq":         set(),
}

_app_ctx.app.clientside_callback(
    """
    function(pathname, splashOpen) {
        var NU = window.dash_clientside.no_update;
        var map = {"/1":"bubble","/2":"heatmap","/3":"dca",
                   "/4":"retire","/5":"supercharge","/6":"stack","/7":"faq"};
        /* While splash modal is open, defer the tab switch so chart
           callbacks don't fire into a container hidden behind the modal. */
        if (splashOpen) {
            window._pendingTabPath = pathname;
            return NU;
        }
        var p = window._pendingTabPath || pathname;
        if (window._pendingTabPath) {
            /* Splash just closed — force Plotly resize after chart renders
               to handle rapid-dismiss edge case where layout hasn't settled. */
            setTimeout(function() {
                window.dispatchEvent(new Event("resize"));
            }, 1200);
        }
        window._pendingTabPath = null;
        if (p && /^\\/7\\.\\d+$/.test(p)) { return "faq"; }
        var tab = map[p];
        return tab ? tab : NU;
    }
    """,
    Output("main-tabs", "active_tab", allow_duplicate=True),
    Input("url", "pathname"),
    Input("splash-modal", "is_open"),
    prevent_initial_call="initial_duplicate",
)

# ── Journey tracker: update milestones in localStorage on every page load ─────
_app_ctx.app.clientside_callback(
    """
    function(pathname, journey, price) {
        var NU = window.dash_clientside.no_update;
        var now = Date.now();
        var THIRTY_SIX_H = 36 * 3600 * 1000;
        var ONE_DAY = 24 * 3600 * 1000;
        if (!journey || !journey.first_ts) {
            /* First ever visit */
            return {
                first_ts: now, first_price: price || null,
                visits: 1, tabs_seen: [],
                streak_days: 1, last_visit_ts: now, streak_unlocked: false
            };
        }
        /* Returning visitor — update visits + streak */
        journey.visits = (journey.visits || 0) + 1;
        if (!journey.first_price && price) { journey.first_price = price; }
        var gap = now - (journey.last_visit_ts || 0);
        if (gap >= ONE_DAY / 2 && gap <= THIRTY_SIX_H) {
            journey.streak_days = (journey.streak_days || 1) + 1;
            journey.last_visit_ts = now;
            if (journey.streak_days >= 7) { journey.streak_unlocked = true; }
        } else if (gap > THIRTY_SIX_H) {
            journey.streak_days = 1;
            journey.last_visit_ts = now;
        }
        return journey;
    }
    """,
    Output("journey-store", "data"),
    Input("url", "pathname"),
    State("journey-store", "data"),
    State("btc-price-store", "data"),
    prevent_initial_call=False,
)

# ── Journey: track tab visits ────────────────────────────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(tab, journey) {
        if (!tab || !journey) return window.dash_clientside.no_update;
        var seen = journey.tabs_seen || [];
        if (seen.indexOf(tab) === -1) {
            seen.push(tab);
            journey.tabs_seen = seen;
            return journey;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("journey-store", "data", allow_duplicate=True),
    Input("main-tabs", "active_tab"),
    State("journey-store", "data"),
    prevent_initial_call="initial_duplicate",
)

# ── Easter egg: 6 clicks on logo → genesis quote in splash modal ─────────────
_JOURNEY_BODY = """
        var journey = null;
        try { journey = JSON.parse(localStorage.getItem("journey-store")); } catch(e) {}
        var jText = "";
        var jStyle = {"display":"none"};
        var _jnow = Date.now();
        if (journey && journey.first_ts) {
            var days = Math.floor((_jnow - journey.first_ts) / 86400000);
            var parts = [];
            if (days >= 1) {
                parts.push("\\u2615 Day " + days + " of your Bitcoin journey");
            } else {
                parts.push("\\u2615 Welcome to the rabbit hole");
            }
            if (journey.first_price) {
                var cp = null;
                try {
                    var el = document.getElementById("price-ticker");
                    if (el) {
                        var m = (el.textContent || "").match(/\\$([\\.\\d]+)(K|M)?/);
                        if (m) {
                            cp = parseFloat(m[1]);
                            if (m[2] === "K") cp *= 1000;
                            else if (m[2] === "M") cp *= 1000000;
                        }
                    }
                } catch(e) {}
                if (cp && cp > 0) {
                    var pct = ((cp - journey.first_price) / journey.first_price * 100);
                    var sign = pct >= 0 ? "+" : "";
                    parts.push("\\u20bf was $" + Math.round(journey.first_price).toLocaleString()
                               + " when you first visited (" + sign + pct.toFixed(1) + "%)");
                } else {
                    parts.push("\\u20bf was $" + Math.round(journey.first_price).toLocaleString()
                               + " when you first visited");
                }
            }
            if (journey.visits && journey.visits > 1) {
                parts.push("Visit #" + journey.visits);
            }
            var tabCount = (journey.tabs_seen || []).length;
            if (tabCount > 0 && tabCount < 7) {
                parts.push(tabCount + "/7 tabs explored");
            } else if (tabCount >= 7) {
                parts.push("\\u2b50 All 7 tabs explored!");
            }
            /* Prepend noble title if knighted */
            try {
                var wf = JSON.parse(localStorage.getItem("wizard-flags")) || {};
                if (wf.noble_title) {
                    parts.unshift("\\u2694\\ufe0f " + wf.noble_title);
                }
            } catch(e) {}
            if (parts.length > 0) {
                jText = parts.join("  \\u00b7  ");
                jStyle = {"display":"block", "textAlign":"center", "fontSize":"12px",
                          "color":"#888", "marginTop":"16px", "lineHeight":"1.7"};
            }
        }
"""

# ── Splash quote: show if 6+ hours since last visit (regular quotes only) ─────
_app_ctx.app.clientside_callback(
    """
    function(ts_store) {
        var SIX_HOURS = 6 * 3600 * 1000;
        var now = Date.now();
        var last = ts_store ? parseInt(ts_store) : 0;
        if (now - last >= SIX_HOURS) {
            var quotes = """ + json.dumps([list(q) for q in _SPLASH_QUOTES]) + """;
            /* Deterministic pseudo-random shuffle using epoch as seed */
            var seed = Math.floor(now / (6 * 3600 * 1000));
            // Mulberry32: fast deterministic PRNG (no crypto needed, just shuffling quotes)
            function mulberry32(a) { return function() {
                a |= 0; a = a + 0x6D2B79F5 | 0;
                var t = Math.imul(a ^ a >>> 15, 1 | a);
                t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
                return ((t ^ t >>> 14) >>> 0) / 4294967296;
            }}
            var rng = mulberry32(seed);
            for (var i = quotes.length - 1; i > 0; i--) {
                var j = Math.floor(rng() * (i + 1));
                var tmp = quotes[i]; quotes[i] = quotes[j]; quotes[j] = tmp;
            }
            var idx = 0;
            var q = quotes[idx];
            """ + _JOURNEY_BODY + """
            /* Hide onion knight button during regular splash */
            var _kw2 = document.getElementById("onion-knight-wrap");
            if (_kw2) _kw2.style.display = "none";
            return [true, now.toString(), '"' + q[0] + '"', "\\u2014 " + q[1],
                    {"display":"none"}, jText, jStyle];
        }
        return [false, window.dash_clientside.no_update,
                window.dash_clientside.no_update, window.dash_clientside.no_update,
                window.dash_clientside.no_update,
                window.dash_clientside.no_update, window.dash_clientside.no_update];
    }
    """,
    Output("splash-modal", "is_open"),
    Output("splash-ts-store", "data"),
    Output("splash-quote-text", "children"),
    Output("splash-quote-attr", "children"),
    Output("splash-next-wrap", "style"),
    Output("journey-stats", "children"),
    Output("journey-stats", "style"),
    Input("splash-ts-store", "data"),
    prevent_initial_call=False,
)

# Dismiss splash
_app_ctx.app.clientside_callback(
    """
    function(n) {
        if (n) { return false; }
        return window.dash_clientside.no_update;
    }
    """,
    Output("splash-modal", "is_open", allow_duplicate=True),
    Input("splash-dismiss", "n_clicks"),
    prevent_initial_call="initial_duplicate",
)


_EGG_JS = """
    function(n) {
        var NU = window.dash_clientside.no_update;
        if (!n) return [NU, NU, NU, NU, NU, NU];
        window._eggClicks = (window._eggClicks || 0) + 1;
        clearTimeout(window._eggTimer);
        if (window._eggClicks >= 6) {
            window._eggClicks = 0;
            window._splashIdx = 0;
            """ + _JOURNEY_BODY + """
            /* Show Accept Knighthood button on .onion (or dev) if not already knighted.
               setTimeout: React re-renders modal children when is_open flips,
               so DOM manipulation must happen after React settles. */
            setTimeout(function() {
                var _wfE = {};
                try { _wfE = JSON.parse(localStorage.getItem("wizard-flags")) || {}; } catch(e) {}
                var _kw = document.getElementById("onion-knight-wrap");
                if (_kw) {
                    var _isOnion = location.hostname.endsWith(".onion");
                    var _isDevE = !_isOnion && location.hostname !== "quantoshi.xyz";
                    _kw.style.display = ((_isOnion || _isDevE) && !_wfE.knighted) ? "block" : "none";
                }
                var _rk = document.getElementById("replay-knight-wrap");
                if (_rk) _rk.style.display = _wfE.knighted ? "block" : "none";
            }, 200);
            return [true,
                    "\\u201cThe Times 03/Jan/2009 Chancellor on brink of second bailout for banks.\\u201d",
                    "\\u2014 Bitcoin Genesis Block",
                    {"display":"inline"}, jText, jStyle];
        }
        window._eggTimer = setTimeout(function() { window._eggClicks = 0; }, 3000);
        return [NU, NU, NU, NU, NU, NU];
    }
"""
_app_ctx.app.clientside_callback(
    _EGG_JS,
    Output("splash-modal", "is_open", allow_duplicate=True),
    Output("splash-quote-text", "children", allow_duplicate=True),
    Output("splash-quote-attr", "children", allow_duplicate=True),
    Output("splash-next-wrap", "style", allow_duplicate=True),
    Output("journey-stats", "children", allow_duplicate=True),
    Output("journey-stats", "style", allow_duplicate=True),
    Input("logo-easter-egg", "n_clicks"),
    prevent_initial_call=True,
)
_app_ctx.app.clientside_callback(
    _EGG_JS,
    Output("splash-modal", "is_open", allow_duplicate=True),
    Output("splash-quote-text", "children", allow_duplicate=True),
    Output("splash-quote-attr", "children", allow_duplicate=True),
    Output("splash-next-wrap", "style", allow_duplicate=True),
    Output("journey-stats", "children", allow_duplicate=True),
    Output("journey-stats", "style", allow_duplicate=True),
    Input("logo-easter-egg-mobile", "n_clicks"),
    prevent_initial_call=True,
)

# Next quote button — cycle through all quotes (genesis first, then regular)
_app_ctx.app.clientside_callback(
    """
    function(n) {
        if (!n) return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        var quotes = """ + _SPLASH_QUOTES_JS + """;
        window._splashIdx = ((window._splashIdx || 0) + 1) % quotes.length;
        var q = quotes[window._splashIdx];
        return ['"' + q[0] + '"', "\\u2014 " + q[1]];
    }
    """,
    Output("splash-quote-text", "children", allow_duplicate=True),
    Output("splash-quote-attr", "children", allow_duplicate=True),
    Input("splash-next", "n_clicks"),
    prevent_initial_call="initial_duplicate",
)

# ── Onion knighting: Accept Knighthood button → close splash + play ceremony ──
_app_ctx.app.clientside_callback(
    """
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        /* Hide the button immediately */
        var kw = document.getElementById("onion-knight-wrap");
        if (kw) kw.style.display = "none";
        /* Play the onion ceremony after modal closes */
        setTimeout(function() {
            if (window._playOnionKnighting) window._playOnionKnighting();
        }, 400);
        return false;
    }
    """,
    Output("splash-modal", "is_open", allow_duplicate=True),
    Input("onion-knight-btn", "n_clicks"),
    prevent_initial_call="initial_duplicate",
)


# ── Replay knighting from easter egg panel ────────────────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        if (window._replayKnighting) {
            setTimeout(function() { window._replayKnighting(); }, 400);
        }
        return false;
    }
    """,
    Output("splash-modal", "is_open", allow_duplicate=True),
    Input("replay-knight-link", "n_clicks"),
    prevent_initial_call=True,
)

# ── LT-8c: Welcome message for returning knights ─────────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(ts) {
        try {
            var flags = JSON.parse(localStorage.getItem("wizard-flags") || "{}");
            if (flags.noble_title) {
                return "Welcome back, " + flags.noble_title + ". The quest continues.";
            }
        } catch(e) {}
        return "";
    }
    """,
    Output("knight-welcome", "children"),
    Input("splash-ts-store", "data"),
)


# ── Mobile nav drawer: auto-collapse after 3s, toggle on tap ──────────────────
_app_ctx.app.clientside_callback(
    """
    function(n) {
        var drawer = document.getElementById("mobile-nav-drawer");
        var toggle = document.getElementById("mobile-nav-toggle");
        if (!drawer || !toggle) return window.dash_clientside.no_update;

        if (!window._navDrawerInit) {
            window._navDrawerInit = true;
            // Auto-collapse after 3 seconds
            setTimeout(function() {
                if (!window._navDrawerManual) {
                    drawer.classList.add("collapsed");
                    toggle.classList.add("visible");
                }
            }, 3000);
        }
        // On tap: toggle open/closed
        if (n) {
            window._navDrawerManual = true;
            var isCollapsed = drawer.classList.contains("collapsed");
            if (isCollapsed) {
                drawer.classList.remove("collapsed");
                toggle.classList.remove("visible");
                // Re-collapse after 4s
                setTimeout(function() {
                    drawer.classList.add("collapsed");
                    toggle.classList.add("visible");
                }, 4000);
            } else {
                drawer.classList.add("collapsed");
                toggle.classList.add("visible");
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("mobile-nav-toggle", "className"),
    Input("mobile-nav-toggle", "n_clicks"),
    prevent_initial_call=False,
)

# ── Desktop nav drawer: auto-collapse after 4s, toggle on tap ─────────────────
_app_ctx.app.clientside_callback(
    """
    function(n) {
        var drawer = document.getElementById("desktop-nav-drawer");
        var toggle = document.getElementById("desktop-nav-toggle");
        if (!drawer || !toggle) return window.dash_clientside.no_update;

        if (!window._deskDrawerInit) {
            window._deskDrawerInit = true;
            setTimeout(function() {
                if (!window._deskDrawerManual) {
                    drawer.classList.add("collapsed");
                    toggle.classList.add("visible");
                }
            }, 4000);
        }
        if (n) {
            window._deskDrawerManual = true;
            var isCollapsed = drawer.classList.contains("collapsed");
            if (isCollapsed) {
                drawer.classList.remove("collapsed");
                toggle.classList.remove("visible");
                setTimeout(function() {
                    drawer.classList.add("collapsed");
                    toggle.classList.add("visible");
                }, 5000);
            } else {
                drawer.classList.add("collapsed");
                toggle.classList.add("visible");
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("desktop-nav-toggle", "className"),
    Input("desktop-nav-toggle", "n_clicks"),
    prevent_initial_call=False,
)

# ── Price ticker pulse + green/red flash ──────────────────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(children) {
        ["price-ticker", "price-ticker-mobile"].forEach(function(id) {
            var el = document.getElementById(id);
            if (!el) return;
            /* Parse current price from text */
            var m = (el.textContent || "").match(/\\$([\\d.]+)(K|M)?/);
            var newPrice = null;
            if (m) {
                newPrice = parseFloat(m[1]);
                if (m[2] === "K") newPrice *= 1000;
                else if (m[2] === "M") newPrice *= 1000000;
            }
            /* Compare with stored previous price */
            var prev = parseFloat(el.getAttribute("data-prev-price"));
            if (newPrice) el.setAttribute("data-prev-price", newPrice);

            el.classList.remove("price-pulse", "price-flash-green", "price-flash-red");
            void el.offsetWidth;
            el.classList.add("price-pulse");
            if (newPrice && prev && newPrice !== prev) {
                el.classList.add(newPrice > prev ? "price-flash-green" : "price-flash-red");
            }
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output("price-ticker", "className", allow_duplicate=True),
    Input("price-ticker", "children"),
    prevent_initial_call="initial_duplicate",
)


@callback(
    Output("faq-accordion", "active_item"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def open_faq_item(pathname):
    """Open a specific FAQ accordion item when pathname is /7.N (1-indexed)."""
    if not pathname or not pathname.startswith("/7."):
        return no_update
    try:
        n = int(pathname[3:])
        if 1 <= n <= len(_FAQ):
            return f"faq-{n - 1}"
    except (ValueError, IndexError):
        pass
    return no_update


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks — Snapshot / Share
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("share-modal", "is_open"),
    Input("share-btn",          "n_clicks"),
    Input("share-btn-mobile",   "n_clicks"),
    Input("share-modal-close",  "n_clicks"),
    State("share-modal",        "is_open"),
    prevent_initial_call=True,
)
def toggle_share_modal(n1, n1m, n2, is_open):
    return not is_open


@callback(
    *[Output(cid, prop) for cid, prop in _SNAPSHOT_CONTROLS],
    Output("snapshot-lots",     "data"),
    Output("loaded-hash-store", "data"),
    Input("url", "hash"),
    prevent_initial_call=False,
)
def restore_from_url(hash_str):
    """Decode snapshot hash from URL → restore all controls + lots."""
    n_outs = len(_SNAPSHOT_CONTROLS) + 2
    blank  = [no_update] * n_outs
    if not hash_str:
        return blank
    h = hash_str.lstrip("#")
    if h.startswith(_SNAP_PREFIX):
        state = _decode_snapshot(h[len(_SNAP_PREFIX):])
    elif h.startswith(_SNAP_PREFIX_V2):
        # v2 links have different positional layout — decode may mismatch
        state = _decode_snapshot(h[len(_SNAP_PREFIX_V2):])
    elif h.startswith(_SNAP_PREFIX_V1):
        state = _decode_snapshot_v1(h[len(_SNAP_PREFIX_V1):])
    else:
        return blank
    if not state:
        logger.warning("Snapshot decode failed for hash: %s…", hash_str[:20])
        return blank
    logger.info("Snapshot restored: %d controls, lots=%s",
                sum(1 for k in state if k != "_lots"), "yes" if "_lots" in state else "no")
    results = [state.get(f"{cid}:{prop}", no_update)
               for cid, prop in _SNAPSHOT_CONTROLS]
    results.append(state.get("_lots", None))  # snapshot-lots
    results.append(hash_str)                  # loaded-hash-store
    return results


@callback(
    Output("share-url-display", "value"),
    Output("link-history",      "data"),
    Input("share-copy-btn",    "n_clicks"),
    Input("loaded-hash-store", "data"),
    State("share-scope",        "value"),
    State("share-include-lots", "value"),
    State("lots-store",         "data"),
    *[State(cid, prop) for cid, prop in _SNAPSHOT_CONTROLS],
    State("link-history",       "data"),
    prevent_initial_call=True,
)
def manage_snapshot(n_btn, loaded_hash, share_scope, include_lots, lots_data, *rest):
    *ctrl_vals, history = rest
    history  = list(history or [])
    existing = {h["hash"] for h in history}
    triggered = ctx.triggered_id

    if triggered == "share-copy-btn":
        state = {f"{cid}:{prop}": val
                 for (cid, prop), val in zip(_SNAPSHOT_CONTROLS, ctrl_vals)}
        if include_lots and lots_data:
            state["_lots"] = lots_data
        active_tab = state.get("main-tabs:active_tab") or "bubble"
        tab_path   = _TAB_TO_PATH.get(active_tab, "/1")
        scope      = share_scope or "all"
        tab_filter = _TAB_CONTROLS.get(active_tab) if scope == "tab" else None
        encoded    = _encode_snapshot(state, tab_filter=tab_filter)
        base_url   = flask_request.host_url.rstrip("/")
        full_url   = f"{base_url}{tab_path}#{_SNAP_PREFIX}{encoded}"
        _add_snapshot_entry(history, existing, encoded, full_url,
                            bool(include_lots and lots_data), scope, active_tab)
        return full_url, history

    if triggered == "loaded-hash-store" and loaded_hash:
        h = loaded_hash.lstrip("#")
        if h.startswith(_SNAP_PREFIX):
            encoded = h[len(_SNAP_PREFIX):]
            state   = _decode_snapshot(encoded)
            prefix  = _SNAP_PREFIX
        elif h.startswith(_SNAP_PREFIX_V2):
            encoded = h[len(_SNAP_PREFIX_V2):]
            state   = _decode_snapshot(encoded)
            prefix  = _SNAP_PREFIX_V2
        elif h.startswith(_SNAP_PREFIX_V1):
            encoded = h[len(_SNAP_PREFIX_V1):]
            state   = _decode_snapshot_v1(encoded)
            prefix  = _SNAP_PREFIX_V1
        else:
            return no_update, no_update
        if not state:
            return no_update, no_update
        active_tab = state.get("main-tabs:active_tab") or "bubble"
        tab_path   = _TAB_TO_PATH.get(active_tab, "/1")
        base_url   = flask_request.host_url.rstrip("/")
        full_url   = f"{base_url}{tab_path}#{prefix}{encoded}"
        if _add_snapshot_entry(history, existing, encoded, full_url,
                               "_lots" in state, "unknown", active_tab):
            return no_update, history

    return no_update, no_update


def _add_snapshot_entry(history, existing, encoded, full_url,
                        includes_lots, scope, tab):
    """Append a snapshot entry to history if not already present.

    Mutates history in-place and returns True if an entry was added.
    """
    if encoded in existing:
        return False
    history.insert(0, {
        "hash": encoded, "url": full_url,
        "ts": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "includes_lots": includes_lots,
        "scope": scope,
        "tab": tab,
    })
    history[:] = history[:50]
    return True


@callback(
    Output("effective-lots", "data"),
    Input("lots-store",    "data"),
    Input("snapshot-lots", "data"),
)
def update_effective_lots(local_lots, snapshot_lots):
    return snapshot_lots if snapshot_lots is not None else (local_lots or [])


@callback(
    Output("snapshot-lots-banner", "children"),
    Input("snapshot-lots", "data"),
)
def update_snapshot_banner(snapshot_lots):
    if not snapshot_lots:
        return []
    n = len(snapshot_lots)
    return dbc.Alert([
        html.Span(f"Showing {n} lot(s) from a shared link.  "),
        dbc.Button("Restore my lots", id="restore-lots-btn",
                   color="link", size="sm", className="p-0 ms-1 align-baseline"),
    ], color="info", className="py-1 px-3 mb-2 d-flex align-items-center")


@callback(
    Output("snapshot-lots", "data", allow_duplicate=True),
    Input("restore-lots-btn", "n_clicks"),
    prevent_initial_call=True,
)
def restore_my_lots(_):
    return None


@callback(
    Output("link-history-display", "children"),
    Input("link-history", "data"),
)
def render_link_history(history):
    if not history:
        return html.Small("No links yet.", className="text-muted")
    items = []
    for entry in history:
        badge = (dbc.Badge("lots", color="info", pill=True, className="me-1")
                 if entry.get("includes_lots") else None)
        items.append(dbc.ListGroupItem([
            html.Div([
                html.Small(entry.get("ts", ""), className="text-muted me-2"),
                badge,
            ], className="mb-1"),
            dbc.InputGroup([
                dbc.Input(value=entry.get("url", ""), readonly=True, size="sm",
                          style={"fontFamily":"monospace","fontSize":"11px"}),
            ], size="sm"),
            html.Div(
                html.A("↩ Restore this configuration",
                       href=("#" + entry["url"].split("#", 1)[1])
                            if "#" in entry.get("url", "")
                            else f"#{_SNAP_PREFIX}{entry['hash']}",
                       className="small"),
                className="mt-1",
            ),
        ], className="py-2"))
    return dbc.ListGroup(items, flush=True,
                         style={"maxHeight":"300px","overflowY":"auto"})


@callback(
    Output("link-history", "data", allow_duplicate=True),
    Input("clear-history-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_history(_):
    return []

