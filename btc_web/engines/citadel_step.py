"""Citadel Planner — simulation step (one-period heartbeat)."""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from .citadel_types import CitadelState, SimConfig, PriceModel, FREQ_PPY, _SATOSHI
from .citadel_waterfall import _spending_waterfall
from .citadel_floors import _enforce_floors
from .citadel_rebalancing import _evaluate_rebalancing
from .citadel_tax_integration import (
    _compute_rmd, _quarterly_estimated_payment, _year_boundary_tax,
)
from .citadel_transactions import _sell_btc_tracked

__all__ = [
    "step", "_get_btc_price", "_lognormal_return", "_markov_return",
    "_scf_payment_amount", "_scf_check_repay",
]


def _get_btc_price(t: float, config: SimConfig, model: PriceModel,
                   rng: np.random.Generator,
                   sim_mode: str = "deterministic",
                   q: float = 0.50,
                   transition_matrix=None) -> float:
    """Get BTC price for time t.
    - deterministic: model.price_at(q, t)
    - stochastic: Markov transition draw (future implementation)
    """
    if sim_mode == "deterministic":
        return float(model.price_at(q, t))
    raise NotImplementedError("MC BTC pricing requires Markov engine integration")




def _lognormal_return(annual_rate: float, annual_vol: float, ppy: int,
                      deterministic: bool = False,
                      rng: np.random.Generator | None = None) -> float:
    """One-period return using lognormal model. Always > -1.0."""
    if deterministic:
        return (1 + annual_rate) ** (1.0 / ppy) - 1.0
    if annual_vol <= 0:
        return (1 + max(annual_rate, -0.99)) ** (1.0 / ppy) - 1.0
    annual_rate = max(annual_rate, -0.99)  # guard: rate > -100%
    sigma_ln = math.sqrt(math.log(1 + (annual_vol / (1 + annual_rate)) ** 2))
    mu_ln = math.log(1 + annual_rate) - sigma_ln ** 2 / 2
    period_mu = mu_ln / ppy
    period_sigma = sigma_ln / math.sqrt(ppy)
    return math.exp(rng.normal(period_mu, period_sigma)) - 1.0


def _markov_return(matrix: dict, current_regime: int,
                   rng: np.random.Generator) -> tuple[float, int]:
    """Sample one-period return from a Markov transition matrix.

    Args:
        matrix: dict with "trans" (n_bins x n_bins), "bin_means", "bin_vols"
        current_regime: current bin index
        rng: random number generator

    Returns:
        (monthly_return, new_regime) — the sampled return and the new regime bin
    """
    trans = matrix["trans"]
    n_bins = trans.shape[0]
    current_regime = max(0, min(current_regime, n_bins - 1))

    # Sample next regime from transition probabilities
    probs = trans[current_regime]
    new_regime = int(rng.choice(n_bins, p=probs))

    # Sample return from the new regime's distribution
    mean = matrix["bin_means"][new_regime]
    vol = matrix["bin_vols"][new_regime]
    if vol > 0:
        ret = rng.normal(mean, vol)
    else:
        ret = mean

    return float(ret), new_regime


def _scf_payment_amount(config: SimConfig, ppy: int) -> float:
    """Calculate monthly loan payment. Returns monthly $ amount."""
    if not config.scf_enabled or config.scf_amount <= 0:
        return 0.0
    monthly_rate = (config.scf_rate / 100) / 12
    if config.scf_type == "perpetual":
        return config.scf_amount * monthly_rate
    n = config.scf_term
    if monthly_rate == 0:
        return config.scf_amount / n
    return config.scf_amount * monthly_rate / (1 - (1 + monthly_rate) ** -n)


def _scf_check_repay(state: CitadelState, config: SimConfig,
                     btc_annual_return: float) -> None:
    """For perpetual loans: check if BTC return has fallen below threshold.
    If so, sell BTC to repay outstanding principal."""
    if not state.scf_active or config.scf_type != "perpetual":
        return
    threshold = (config.scf_rate / 100) * config.scf_repay_trigger
    if btc_annual_return <= threshold:
        if state.btc_price > 0 and state.btc_stack > 0:
            btc_needed = state.scf_outstanding / state.btc_price
            result = _sell_btc_tracked(state, config, btc_needed)
            repaid = result.btc_sold * state.btc_price
            state.scf_outstanding -= repaid
        if state.scf_outstanding <= 0.01:
            state.scf_outstanding = 0
            state.scf_active = False


def step(state: CitadelState, config: SimConfig,
         btc_price_new: float, rng: np.random.Generator,
         model: "PriceModel | None" = None) -> CitadelState:
    """Advance simulation by one period. Returns new state (does not mutate input)."""
    new = deepcopy(state)
    new.period += 1
    ppy = FREQ_PPY[config.freq]
    dt = 1.0 / ppy
    t_before = new.t  # save pre-increment time for quantile lookup
    new.t += dt

    # Set sim_date for this period (used by tracking helpers)
    _date_years = new.period / ppy
    _date_yr = config.start_yr + int(_date_years)
    _date_mo = min(max(1, int((_date_years % 1) * 12) + 1), 12)
    new.sim_date = f"{_date_yr}-{_date_mo:02d}-15"

    deterministic = (config.n_sims == 1)

    # 1. Update BTC price
    new.btc_price = btc_price_new

    # 2. Dollar-asset returns
    use_markov = (config.asset_return_model == "markov"
                  and config.asset_matrices is not None
                  and config.n_sims > 1)

    # Cash: always deterministic (zero volatility)
    new.cash *= (1 + config.cash_rate / 100) ** (1.0 / ppy)

    if use_markov:
        # Markov regime-based returns for reserves and investments
        am = config.asset_matrices
        _res_keys = ["tres_short", "tres_med", "tres_long"]
        _inv_keys = ["equity", "bond"]
        _regime_attrs = ["res_short_regime", "res_med_regime", "res_long_regime"]
        _inv_regime_attrs = ["equity_regime", "bond_regime"]

        for i, (mkey, rattr) in enumerate(zip(_res_keys, _regime_attrs)):
            if mkey in am:
                ret, new_regime = _markov_return(am[mkey], getattr(new, rattr), rng)
                setattr(new, rattr, new_regime)
                new.reserves[i] *= (1 + ret)
            else:
                # Fallback to lognormal
                rb = config.reserve_bins[i]
                r = _lognormal_return(rb["rate"] / 100, rb["volatility"] / 100, ppy,
                                      deterministic=deterministic, rng=rng)
                new.reserves[i] *= (1 + r)

        for i, (mkey, rattr) in enumerate(zip(_inv_keys, _inv_regime_attrs)):
            if mkey in am:
                ret, new_regime = _markov_return(am[mkey], getattr(new, rattr), rng)
                setattr(new, rattr, new_regime)
                new.investments[i] *= (1 + ret)
            else:
                ib = config.invest_bins[i]
                r = _lognormal_return(ib["return_rate"] / 100, ib["volatility"] / 100, ppy,
                                      deterministic=deterministic, rng=rng)
                new.investments[i] *= (1 + r)
    else:
        # Lognormal returns (user-input rates/volatility)
        for i, rb in enumerate(config.reserve_bins):
            r = _lognormal_return(rb["rate"] / 100, rb["volatility"] / 100, ppy,
                                  deterministic=deterministic, rng=rng)
            new.reserves[i] *= (1 + r)
        for i, ib in enumerate(config.invest_bins):
            r = _lognormal_return(ib["return_rate"] / 100, ib["volatility"] / 100, ppy,
                                  deterministic=deterministic, rng=rng)
            new.investments[i] *= (1 + r)

    # 2b. TD/TF wrapper growth (same rates as taxable wrapper)
    if config.tax_enabled:
        cash_growth = (1 + config.cash_rate / 100) ** (1.0 / ppy)
        new.td_cash *= cash_growth
        new.tf_cash *= cash_growth
        for i, rb in enumerate(config.reserve_bins):
            r = _lognormal_return(rb["rate"] / 100, rb["volatility"] / 100, ppy,
                                  deterministic=deterministic, rng=rng)
            if i < len(new.td_reserves):
                new.td_reserves[i] *= (1 + r)
            r2 = _lognormal_return(rb["rate"] / 100, rb["volatility"] / 100, ppy,
                                   deterministic=deterministic, rng=rng)
            if i < len(new.tf_reserves):
                new.tf_reserves[i] *= (1 + r2)
        for i, ib in enumerate(config.invest_bins):
            r = _lognormal_return(ib["return_rate"] / 100, ib["volatility"] / 100, ppy,
                                  deterministic=deterministic, rng=rng)
            if i < len(new.td_investments):
                new.td_investments[i] *= (1 + r)
            r2 = _lognormal_return(ib["return_rate"] / 100, ib["volatility"] / 100, ppy,
                                   deterministic=deterministic, rng=rng)
            if i < len(new.tf_investments):
                new.tf_investments[i] *= (1 + r2)

    # 2c. Accumulate taxable-wrapper interest income per period
    # Use PRE-GROWTH balances to avoid double-counting (the growth IS the interest)
    if config.tax_enabled and new.tax_year_accum is not None:
        _period_frac = 1.0 / ppy
        _cash_growth_factor = (1 + config.cash_rate / 100) ** _period_frac
        _pre_growth_cash = new.cash / _cash_growth_factor if _cash_growth_factor > 0 else new.cash
        new.tax_year_accum.interest_income += _pre_growth_cash * (config.cash_rate / 100) * _period_frac
        # Treasury interest → ordinary income (federal) but state-exempt
        for _ri, _rb in enumerate(config.reserve_bins):
            _res_rate = _rb["rate"] / 100
            _res_growth = (1 + _res_rate) ** _period_frac
            _pre_growth_res = new.reserves[_ri] / _res_growth if _res_growth > 0 else new.reserves[_ri]
            new.tax_year_accum.treasury_interest += _pre_growth_res * _res_rate * _period_frac

    # 3. Compute BTC quantile from price via model
    #    Use t_before (pre-increment) since the price was generated for that time
    if model is not None:
        btc_quantile = model.quantile_at(new.btc_price, t_before)
    else:
        btc_quantile = 0.5

    # 4. Evaluate rebalancing triggers
    _evaluate_rebalancing(new, config, btc_quantile)

    # 5. Spending
    years_elapsed = new.period / ppy
    combined_rate = (config.inflation + config.spend_growth) / 100
    period_spend = config.monthly_spend * (1 + combined_rate) ** years_elapsed
    if new.scf_active:
        # Retire term loan when term expires
        if config.scf_type == "term":
            # Term is in months; convert period count to months
            months_elapsed = new.period * (12 / ppy)
            if months_elapsed >= config.scf_term:
                new.scf_active = False
                new.scf_outstanding = 0.0
        if new.scf_active:
            period_spend += _scf_payment_amount(config, ppy)
    period_spend *= (12 / ppy)  # scale monthly base to period frequency
    new.period_spend = period_spend

    new.spending_shortfall = _spending_waterfall(new, config, period_spend, model=model)

    # 6. Enforce floor rules (AFTER spending, so floors replenish drawdowns)
    _enforce_floors(new, config, model=model)

    # 7. SCF perpetual loan repayment check
    if new.scf_active and config.scf_type == "perpetual":
        years_elapsed_scf = new.period / ppy
        if years_elapsed_scf > 0 and new.btc_cost_basis > 0:
            btc_annual_return = (new.btc_price / new.btc_cost_basis) ** (
                1 / years_elapsed_scf) - 1
        else:
            btc_annual_return = 0.0
        _scf_check_repay(new, config, btc_annual_return)

    # 8. Tax: quarterly estimated payments + year-end true-up
    if config.tax_enabled:
        sim_year = config.start_yr + int(years_elapsed)
        is_year_end = new.period > 0 and new.period % ppy == 0

        # Quarterly estimated payments (Q1-Q3) — only for freq >= Quarterly
        if ppy >= 4 and not is_year_end:
            qtr_periods = ppy // 4  # periods per quarter
            if new.period % qtr_periods == 0:
                quarter = (new.period % ppy) // qtr_periods  # 1, 2, or 3
                if 1 <= quarter <= 3:
                    _quarterly_estimated_payment(new, config, quarter, sim_year)
                    _enforce_floors(new, config, model=model)

        # Year-end: RMD + Q4 true-up
        if is_year_end:
            # RMD first (adds ordinary income)
            rmd = _compute_rmd(new, config, sim_year)
            if rmd > 0 and new.tax_year_accum is not None:
                new.tax_year_accum.tax_deferred_withdrawals += rmd
                new.tax_year_accum.rmd_required = rmd
                new.tax_year_accum.rmd_taken = rmd
            # Q4 true-up: actual tax minus quarterly payments already made
            _year_boundary_tax(new, config, sim_year, ppy)
            _enforce_floors(new, config, model=model)

    # Clamp sub-satoshi BTC to zero (1 sat = 10^-8 BTC is the smallest unit)
    if 0 < new.btc_stack < _SATOSHI:
        new.btc_stack = 0.0

    return new
