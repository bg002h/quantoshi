"""Citadel Planner — simulation driver and result aggregation."""
from __future__ import annotations

import numpy as np

from .citadel_types import (
    CitadelState, SimConfig, SimResult, PriceModel, FREQ_PPY, _SATOSHI,
)
from .citadel_step import step, _get_btc_price

__all__ = [
    "simulate", "validate_config",
    "_initial_state", "_snapshot_state", "_aggregate_results", "_compute_n_periods",
]


def _initial_state(config: SimConfig, model: "PriceModel | None" = None) -> CitadelState:
    """Create initial state from config."""
    from btc_core import yr_to_t
    t0 = yr_to_t(config.start_yr)
    btc_price = 0.0
    if model and config.selected_qs:
        q = config.selected_qs[len(config.selected_qs) // 2]
        btc_price = float(model.price_at(q, max(t0, 0.5)))
    inv_initials = [ib["initial"] for ib in config.invest_bins]
    # Cost basis: user-specified if provided, else same as initial value (no prior gains)
    if config.invest_cost_basis_initial is not None:
        inv_basis = list(config.invest_cost_basis_initial)
    else:
        inv_basis = list(inv_initials)
    state = CitadelState(
        t=t0, btc_stack=config.start_stack, btc_price=btc_price,
        btc_cost_basis=btc_price, cash=config.cash_initial,
        reserves=[rb["initial"] for rb in config.reserve_bins],
        investments=list(inv_initials),
        invest_cost_basis=inv_basis,
    )

    # Seed taxable + TD + TF wrapper regimes from config
    _REGIME_ATTRS = ("equity_regime", "bond_regime",
                     "res_short_regime", "res_med_regime", "res_long_regime")
    for prefix in ("", "td_", "tf_"):
        for attr in _REGIME_ATTRS:
            setattr(state, f"{prefix}{attr}", getattr(config, f"initial_{attr}"))

    if config.scf_enabled and config.scf_amount > 0 and btc_price > 0:
        btc_bought = config.scf_amount / btc_price
        state.btc_stack += btc_bought
        state.scf_outstanding = config.scf_amount
        state.scf_active = True
        total_btc = config.start_stack + btc_bought
        if total_btc > 0:
            state.btc_cost_basis = (config.start_stack * btc_price + config.scf_amount) / total_btc

    # Always seed lots (even when tax_enabled=False) — forward-compatible
    from .tax_lots import seed_lots, TaxLot
    start_date = f"{config.start_yr}-01-01"
    state.tax_lots = seed_lots(
        [], start_stack=config.start_stack, start_price=btc_price,
        start_date=start_date,
    )
    # SCF initial purchase gets its own lot
    if config.scf_enabled and config.scf_amount > 0 and btc_price > 0:
        scf_btc = config.scf_amount / btc_price
        state.tax_lots.append(TaxLot(
            date=start_date, btc=scf_btc,
            cost_basis=btc_price, source="scf",
        ))

    # Tax-specific initialization (accumulator, wrappers)
    if config.tax_enabled:
        from .tax import TaxYearAccumulator

        # TD/TF wrapper balances from config
        state.td_btc_stack = config.td_btc_stack
        state.td_cash = config.td_cash_initial
        state.td_reserves = [rb["initial"] for rb in config.td_reserve_bins]
        state.td_investments = [ib["initial"] for ib in config.td_invest_bins]

        state.tf_btc_stack = config.tf_btc_stack
        state.tf_cash = config.tf_cash_initial
        state.tf_reserves = [rb["initial"] for rb in config.tf_reserve_bins]
        state.tf_investments = [ib["initial"] for ib in config.tf_invest_bins]

        state.tax_year_accum = TaxYearAccumulator()
        state.loss_carryforward = 0.0
        state.total_taxes_paid = 0.0

    return state


def _compute_n_periods(config: SimConfig) -> int:
    ppy = FREQ_PPY[config.freq]
    return int((config.end_yr - config.start_yr) * ppy)


def _snapshot_state(state: CitadelState, tax_enabled: bool = False) -> dict:
    """Capture scalar values from state for history recording."""
    snap = {
        "btc_stack": state.btc_stack,
        "btc_price": state.btc_price,
        "cash": state.cash,
        "reserves": list(state.reserves),
        "investments": list(state.investments),
        "total_usd": (state.btc_stack * state.btc_price + state.cash
                      + sum(state.reserves) + sum(state.investments)),
        "period_spend": state.period_spend,
        "rebal_event": state.rebal_event,
    }
    if tax_enabled:
        td_total = (state.td_btc_stack * max(state.btc_price, 0) + state.td_cash
                    + sum(state.td_reserves) + sum(state.td_investments))
        tf_total = (state.tf_btc_stack * max(state.btc_price, 0) + state.tf_cash
                    + sum(state.tf_reserves) + sum(state.tf_investments))
        taxable = snap["total_usd"]
        snap["td_total"] = td_total
        snap["tf_total"] = tf_total
        snap["taxable_total"] = taxable
        snap["total_taxes_paid"] = state.total_taxes_paid
        # Include wrapper totals in the grand total
        snap["total_usd"] += td_total + tf_total
    return snap


def _aggregate_results(all_histories: list[list[dict]], config: SimConfig,
                       time_axis: np.ndarray,
                       sim_annual_taxes: list | None = None) -> SimResult:
    """Aggregate per-sim histories into SimResult with median/percentile bands."""
    n_sims = len(all_histories)
    n_periods = len(time_axis)
    n_res = len(config.reserve_bins)
    n_inv = len(config.invest_bins)

    btc_h = np.zeros((n_sims, n_periods))
    btc_p = np.zeros((n_sims, n_periods))
    cash_b = np.zeros((n_sims, n_periods))
    res_b = np.zeros((n_sims, n_periods, n_res))
    inv_b = np.zeros((n_sims, n_periods, n_inv))
    total = np.zeros((n_sims, n_periods))
    cum_spend = np.zeros((n_sims, n_periods))
    depl = []
    rebal_events = []

    tax_en = config.tax_enabled
    taxes_paid_arr = np.zeros((n_sims, n_periods)) if tax_en else None
    td_total_arr = np.zeros((n_sims, n_periods)) if tax_en else None
    tf_total_arr = np.zeros((n_sims, n_periods)) if tax_en else None
    taxable_total_arr = np.zeros((n_sims, n_periods)) if tax_en else None
    all_annual_taxes = [] if tax_en else None

    for s, history in enumerate(all_histories):
        sim_events = []
        running_spend = 0.0
        depl_period = None
        for p, snap in enumerate(history):
            btc_h[s, p] = snap["btc_stack"]
            btc_p[s, p] = snap["btc_price"]
            cash_b[s, p] = snap["cash"]
            for i, rv in enumerate(snap["reserves"]):
                res_b[s, p, i] = rv
            for i, iv in enumerate(snap["investments"]):
                inv_b[s, p, i] = iv
            total[s, p] = snap["total_usd"]
            running_spend += snap["period_spend"]
            cum_spend[s, p] = running_spend
            if snap["rebal_event"]:
                sim_events.append({"period": p, **snap["rebal_event"]})
            if depl_period is None and total[s, p] <= 0:
                depl_period = p
            if tax_en:
                taxes_paid_arr[s, p] = snap.get("total_taxes_paid", 0.0)
                td_total_arr[s, p] = snap.get("td_total", 0.0)
                tf_total_arr[s, p] = snap.get("tf_total", 0.0)
                taxable_total_arr[s, p] = snap.get("taxable_total", 0.0)
        depl.append(depl_period)
        rebal_events.append(sim_events)
        if tax_en and sim_annual_taxes is not None:
            all_annual_taxes.append(sim_annual_taxes[s] if s < len(sim_annual_taxes) else [])

    # Compute median and percentiles across sims
    _zero = np.zeros((n_sims, n_periods)) if n_sims > 0 else np.zeros((1, n_periods))
    _btc_usd = btc_h * btc_p
    _res_total = res_b.sum(axis=2)
    _inv_total = inv_b.sum(axis=2)
    _td = td_total_arr if td_total_arr is not None else np.zeros((n_sims, n_periods))
    _tf = tf_total_arr if tf_total_arr is not None else np.zeros((n_sims, n_periods))
    _tax = taxes_paid_arr if taxes_paid_arr is not None else np.zeros((n_sims, n_periods))

    # Build series dict for consistent iteration
    series = {
        "total": total, "btc_stack": btc_h, "btc_usd": _btc_usd,
        "cash": cash_b, "reserves_total": _res_total,
        "investments_total": _inv_total,
        "td_total": _td, "tf_total": _tf,
        "cumulative_spend": cum_spend, "taxes_paid": _tax,
    }

    # Median across all 10 numeric series
    median = {k: np.median(v, axis=0) for k, v in series.items()}
    # Depletion: fraction of sims depleted at each step (not a percentile)
    median["depletion"] = (total <= 0).astype(np.float64).mean(axis=0)

    # Percentiles (7 levels) across all 10 numeric series
    depletion_frac = median["depletion"]  # same for all percentile levels
    percentiles = {}
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        pct_dict = {k: np.percentile(v, pct, axis=0) for k, v in series.items()}
        pct_dict["depletion"] = depletion_frac  # fraction, not percentile
        percentiles[pct] = pct_dict

    return SimResult(
        time_axis=time_axis, btc_holdings=btc_h, btc_prices=btc_p,
        cash_balances=cash_b, reserve_balances=res_b, invest_balances=inv_b,
        total_usd=total, cumulative_spend=cum_spend,
        depletion_period=depl, rebal_events=rebal_events,
        taxes_paid=taxes_paid_arr, annual_taxes=all_annual_taxes,
        td_total=td_total_arr, tf_total=tf_total_arr,
        taxable_total=taxable_total_arr,
        median=median, percentiles=percentiles,
    )


def simulate(config: SimConfig, model: PriceModel,
             rng_seed: int = 42,
             price_paths: np.ndarray | None = None) -> SimResult:
    """Run simulations, aggregate results.

    - price_paths=None, n_sims=1: deterministic mode using median selected quantile
    - price_paths provided: MC mode — one sim per row in price_paths array
      shape (n_sims, n_periods). Each sim gets a unique RNG seed for
      dollar-asset volatility (rng_seed + sim_id).
    """
    from copy import deepcopy
    config = deepcopy(config)  # avoid mutating caller's config
    validate_config(config)
    ppy = FREQ_PPY[config.freq]
    n_periods = _compute_n_periods(config)

    # Determine sim count
    if price_paths is not None:
        n_sims = price_paths.shape[0]
        if price_paths.shape[1] < n_periods:
            raise ValueError(
                f"price_paths has {price_paths.shape[1]} steps, need {n_periods}")
        # Override config.n_sims so step() knows we're in MC mode
        # (affects is_deterministic check for dollar-asset volatility)
        config.n_sims = n_sims
    else:
        n_sims = 1  # deterministic

    # Build time axis
    from btc_core import yr_to_t
    t0 = yr_to_t(config.start_yr)
    dt = 1.0 / ppy
    time_axis = np.array([t0 + i * dt for i in range(n_periods)])

    all_histories = []
    sim_annual_taxes = [] if config.tax_enabled else None
    for sim_id in range(n_sims):
        # Each sim gets unique RNG for dollar-asset volatility
        rng = np.random.default_rng(rng_seed + sim_id)
        state = _initial_state(config, model=model)
        history = []
        for period_idx in range(n_periods):
            if price_paths is not None:
                btc_price = float(price_paths[sim_id, period_idx])
            else:
                q = config.selected_qs[len(config.selected_qs) // 2] if config.selected_qs else 0.5
                btc_price = _get_btc_price(time_axis[period_idx], config, model, rng,
                                           sim_mode="deterministic", q=q)
            new_state = step(state, config, btc_price, rng, model=model)
            history.append(_snapshot_state(new_state, tax_enabled=config.tax_enabled))
            state = new_state
        all_histories.append(history)
        if config.tax_enabled:
            sim_annual_taxes.append(list(state.annual_tax_history))

    return _aggregate_results(all_histories, config, time_axis,
                              sim_annual_taxes=sim_annual_taxes)


def validate_config(config: SimConfig) -> None:
    """Raise ValueError with descriptive message on invalid config."""
    # Date range
    if config.start_yr >= config.end_yr:
        raise ValueError(f"start_yr ({config.start_yr}) must be < end_yr ({config.end_yr})")
    # Frequency
    if config.freq not in FREQ_PPY:
        raise ValueError(f"freq must be one of {list(FREQ_PPY)}, got '{config.freq}'")
    # Non-negative balances
    for name, val in [("cash_initial", config.cash_initial),
                      ("monthly_spend", config.monthly_spend)]:
        if val < 0:
            raise ValueError(f"{name} must be non-negative, got {val}")
    for i, rb in enumerate(config.reserve_bins):
        if rb["initial"] < 0:
            raise ValueError(f"reserve_bins[{i}].initial must be non-negative")
    for i, ib in enumerate(config.invest_bins):
        if ib["initial"] < 0:
            raise ValueError(f"invest_bins[{i}].initial must be non-negative")
    # Trigger thresholds
    if config.high_q_trigger <= config.low_q_trigger:
        raise ValueError(
            f"high_q_trigger ({config.high_q_trigger}) must be > "
            f"low_q_trigger ({config.low_q_trigger})")
    if (config.high_q_trigger - config.low_q_trigger) < 0.05:
        raise ValueError(
            "high_q_trigger and low_q_trigger must be at least "
            "5 percentile points apart")
    # Split validation
    for name, action in [("high_q_action", config.high_q_action),
                         ("low_q_action", config.low_q_action)]:
        split = action.get("split", {})
        total = sum(split.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"{name}.split must sum to 1.0, got {total:.4f}")
    # Floors non-negative
    if config.cash_floor < 0:
        raise ValueError("cash_floor must be non-negative")
    for i, f in enumerate(config.reserve_floors):
        if f < 0:
            raise ValueError(f"reserve_floors[{i}] must be non-negative")
    # n_sims
    if config.n_sims < 1:
        raise ValueError(f"n_sims must be >= 1, got {config.n_sims}")
    # SCF validation
    if config.scf_enabled:
        if config.scf_type == "term" and config.scf_term <= 0:
            raise ValueError("scf_term must be > 0 for term loans")
        if config.scf_type == "perpetual" and config.scf_repay_trigger <= 0:
            raise ValueError("scf_repay_trigger must be > 0 for perpetual loans")
