"""MC parameter helpers — coercion, setup, finalize, ghost matching."""

import logging
import os

import dash
from dash import ctx

import _app_ctx
import btcpay
from callbacks.coerce import _ci, _cf
from mc_cache import (MC_YEARS_OPTIONS, MC_BINS, MC_SIMS, MC_FREQ,
                      MC_DEFAULT_YEARS, MC_DEFAULT_ENTRY_Q, MC_DEFAULT_START_YR)

logger = logging.getLogger(__name__)


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
                     mc_start_stack=None, mc_model_src=None):
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
        mc_model_src  = mc_model_src or "bub",
    )
    if mc_start_stack is not None:
        v = round(_cf(mc_start_stack, 1.0, lo=0.0, hi=1000.0), 2)
        d["mc_start_stack"] = v
    return d


def _mc_payment_check(tab, mc_years, start_yr, entry_q, pay_token,
                      mc_bins=MC_BINS, mc_sims=MC_SIMS, mc_freq=MC_FREQ,
                      mc_auth=None, mc_model_src="bub"):
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
    if btcpay.is_free_tier(mc_model_src or "bub", mc_yrs, start_yr_, entry_q_,
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
              start_yr_default: int = MC_DEFAULT_START_YR,
              mc_model_src=None,
              ) -> tuple[bool, bool, dict, tuple]:
    """Shared MC setup: coerce → payment check → free tier → build params → ghost."""
    mc_years_c, mc_start_yr_c, mc_entry_q_c, mc_bins_c, mc_sims_c, mc_freq_c = \
        _coerce_mc(mc_years, mc_start_yr, mc_entry_q, mc_bins, mc_sims, mc_freq,
                   start_yr_default)

    mc_ok = bool(mc_enable) and _mc_payment_check(
        tab, mc_years_c, mc_start_yr_c, mc_entry_q_c, pay_token,
        mc_bins=mc_bins_c, mc_sims=mc_sims_c, mc_freq=mc_freq_c,
        mc_auth=mc_auth, mc_model_src=mc_model_src or "bub")

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

    _model_src = _spk.get("mc_model_src", "bub") if mc_stale else (mc_model_src or "bub")
    is_free = mc_ok and btcpay.is_free_tier(
        _model_src, mc_years_c, mc_start_yr_c, mc_entry_q_c,
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
        mc_model_src=_spk.get("mc_model_src", "bub") if mc_stale else mc_model_src,
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
