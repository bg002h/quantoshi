"""Citadel Planner — tax integration: RMDs, estimated payments, year-end true-up."""
from __future__ import annotations

from .citadel_types import CitadelState, SimConfig, FREQ_PPY
from .citadel_transactions import _sell_investments_tracked

__all__ = [
    "_get_state_rate", "_rmd_start_age", "_compute_rmd",
    "_pay_tax_amount", "_quarterly_estimated_payment", "_year_boundary_tax",
]


def _get_state_rate(config: SimConfig) -> float:
    """Resolve effective state tax rate (percentage)."""
    if config.state_rate_override is not None:
        return config.state_rate_override
    from .tax_data import STATE_TAX_RATES
    return STATE_TAX_RATES.get(config.state_code, 0.0)


def _rmd_start_age(birth_year: int) -> int:
    """RMD start age based on birth year (SECURE 2.0 Act)."""
    if birth_year <= 1959:
        return 73
    return 75


def _compute_rmd(state: CitadelState, config: SimConfig, sim_year: int) -> float:
    """Compute Required Minimum Distribution for the year. Returns $ amount.
    Withdraws from TD accounts and records as ordinary income."""
    if config.birth_year is None:
        return 0.0
    age = sim_year - config.birth_year
    start_age = _rmd_start_age(config.birth_year)
    if age < start_age:
        return 0.0

    from .tax_data import RMD_FACTORS
    factor = RMD_FACTORS.get(age)
    if factor is None or factor <= 0:
        return 0.0

    # Total TD balance
    td_total = (state.td_btc_stack * state.btc_price + state.td_cash
                + sum(state.td_reserves) + sum(state.td_investments))
    if td_total <= 0:
        return 0.0

    rmd_amount = td_total / factor

    # Withdraw from TD: cash -> reserves -> investments -> BTC
    remaining = rmd_amount
    draw = min(state.td_cash, remaining)
    state.td_cash -= draw
    remaining -= draw

    for i in range(len(state.td_reserves)):
        if remaining <= 0:
            break
        draw = min(state.td_reserves[i], remaining)
        state.td_reserves[i] -= draw
        remaining -= draw

    for i in reversed(range(len(state.td_investments))):
        if remaining <= 0:
            break
        draw = min(state.td_investments[i], remaining)
        state.td_investments[i] -= draw
        remaining -= draw

    if remaining > 0 and state.td_btc_stack > 0 and state.btc_price > 0:
        btc_val = state.td_btc_stack * state.btc_price
        if btc_val >= remaining:
            state.td_btc_stack -= remaining / state.btc_price
            remaining = 0.0
        else:
            state.td_btc_stack = 0.0
            remaining -= btc_val

    actual_rmd = rmd_amount - remaining
    # RMD becomes taxable cash in the taxable wrapper
    state.cash += actual_rmd
    return actual_rmd


def _pay_tax_amount(state: CitadelState, config: SimConfig,
                    amount: float, sim_year: int,
                    tax_result: dict | None = None) -> None:
    """Draw *amount* from taxable wrapper to pay a tax bill.

    Sources are tried in order: cash → reserves → investments (with LTCG
    gross-up) → tax-deferred (with ordinary-income gross-up).  Mutates
    *state* in place.
    """
    if amount <= 0:
        return

    tax_remaining = amount

    # 1. Cash — no gross-up (paying cash is not a taxable event)
    draw = min(state.cash, tax_remaining)
    state.cash -= draw
    tax_remaining -= draw

    # 2. Reserves — no gross-up (principal withdrawal)
    for i in range(len(state.reserves)):
        if tax_remaining <= 0:
            break
        draw = min(state.reserves[i], tax_remaining)
        state.reserves[i] -= draw
        tax_remaining -= draw

    # 3. Investments — gross-up for LTCG on the gain portion
    if tax_remaining > 0:
        # Estimate marginal LTCG rate (15% typical, 20% + 3.8% NIIT at top)
        agi = tax_result.get("agi", 0) if tax_result else 0
        from .tax_data import NIIT_THRESHOLD
        niit_applies = agi > NIIT_THRESHOLD.get(config.filing_status, 200_000)
        ltcg_rate = 0.15 + (0.038 if niit_applies else 0) + _get_state_rate(config) / 100
        for i in reversed(range(len(state.investments))):
            if tax_remaining <= 0:
                break
            current = state.investments[i]
            if current <= 0:
                continue
            basis_frac = state.invest_cost_basis[i] / current if current > 0 else 0
            gain_frac = 1.0 - basis_frac
            effective_rate = ltcg_rate * gain_frac
            gross = tax_remaining / max(1.0 - effective_rate, 0.1)
            sell_amount = min(current, gross)
            drawn, gain = _sell_investments_tracked(state, config, i, sell_amount)
            tax_on_sale = max(gain, 0) * ltcg_rate
            tax_remaining -= (drawn - tax_on_sale)

    # 4. TD withdrawal — gross-up for ordinary income at actual marginal rate
    if tax_remaining > 0:
        from .tax import apply_progressive_brackets, _inflate_brackets
        from .tax_data import FEDERAL_BRACKETS_TCJA, FEDERAL_BRACKETS_SUNSET
        _agi = tax_result.get("agi", 0) if tax_result else 0
        _yrs = max(sim_year - 2025, 0)
        if config.tcja_sunset:
            _brk = _inflate_brackets(FEDERAL_BRACKETS_SUNSET[config.filing_status], _yrs, config.inflation / 100)
        else:
            _brk = _inflate_brackets(FEDERAL_BRACKETS_TCJA[config.filing_status], _yrs, config.inflation / 100)
        # Marginal rate = rate on the next dollar above current AGI
        _tax_at_agi = apply_progressive_brackets(_agi, _brk)
        _tax_at_agi_plus = apply_progressive_brackets(_agi + 1, _brk)
        _marginal_fed = _tax_at_agi_plus - _tax_at_agi
        ordinary_rate = _marginal_fed + _get_state_rate(config) / 100
        gross = tax_remaining / max(1.0 - ordinary_rate, 0.1)
        td_avail = state.td_cash + sum(state.td_reserves) + sum(state.td_investments)
        td_draw = min(gross, td_avail)
        if td_draw > 0:
            rem = td_draw
            d = min(state.td_cash, rem); state.td_cash -= d; rem -= d
            for j in range(len(state.td_reserves)):
                if rem <= 0: break
                d = min(state.td_reserves[j], rem); state.td_reserves[j] -= d; rem -= d
            for j in reversed(range(len(state.td_investments))):
                if rem <= 0: break
                d = min(state.td_investments[j], rem); state.td_investments[j] -= d; rem -= d
            actual = td_draw - rem
            tax_on_td = actual * ordinary_rate
            tax_remaining -= (actual - tax_on_td)


def _quarterly_estimated_payment(state: CitadelState, config: SimConfig,
                                  quarter: int, sim_year: int) -> None:
    """Pay estimated quarterly tax (Q1-Q3). Annualizes YTD income.

    quarter: 1, 2, or 3 (Q4 is handled by _year_boundary_tax as the true-up).
    Mutates state: draws payment from accounts, updates quarterly_tax_paid_ytd.
    """
    from .tax import compute_annual_tax, TaxYearAccumulator
    from copy import copy

    if state.tax_year_accum is None or quarter < 1 or quarter > 3:
        return

    ytd_fraction = quarter / 4.0  # 0.25, 0.50, 0.75

    # Annualize YTD income by scaling up
    ann = copy(state.tax_year_accum)
    scale = 1.0 / ytd_fraction
    ann.tax_deferred_withdrawals *= scale
    ann.interest_income *= scale
    ann.treasury_interest *= scale
    ann.other_income *= scale
    ann.st_capital_gains *= scale
    ann.st_capital_losses *= scale
    ann.lt_capital_gains *= scale
    ann.lt_capital_losses *= scale
    ann.roth_withdrawals *= scale
    ann.loss_carryforward = state.loss_carryforward  # not scaled

    state_rate = _get_state_rate(config)
    projected = compute_annual_tax(
        ann,
        filing_status=config.filing_status,
        tcja_sunset=config.tcja_sunset,
        sim_year=sim_year,
        inflation_rate=config.inflation / 100,
        state_rate=state_rate,
    )

    cumulative_target = projected["total"] * (quarter / 4.0)
    payment = max(cumulative_target - state.quarterly_tax_paid_ytd, 0)

    if payment > 0:
        _pay_tax_amount(state, config, payment, sim_year)
        state.quarterly_tax_paid_ytd += payment
        state.total_taxes_paid += payment


def _year_boundary_tax(state: CitadelState, config: SimConfig,
                       sim_year: int, ppy: int) -> None:
    """Compute and pay annual tax at year boundary. Mutates state."""
    from .tax import compute_annual_tax, TaxYearAccumulator

    if state.tax_year_accum is None:
        return

    # Add other income
    years_elapsed = max(sim_year - config.start_yr, 0)
    other_inc = config.other_income * (1 + config.other_income_growth / 100) ** years_elapsed
    state.tax_year_accum.other_income += other_inc

    # Interest income is now accumulated per-period in step(), not here.

    # Set loss carryforward
    state.tax_year_accum.loss_carryforward = state.loss_carryforward

    state_rate = _get_state_rate(config)
    tax_result = compute_annual_tax(
        state.tax_year_accum,
        filing_status=config.filing_status,
        tcja_sunset=config.tcja_sunset,
        sim_year=sim_year,
        inflation_rate=config.inflation / 100,
        state_rate=state_rate,
    )

    tax_owed_full = tax_result["total"]

    # Q4 true-up: subtract quarterly estimated payments already made
    q4_payment = tax_owed_full - state.quarterly_tax_paid_ytd
    if q4_payment > 0:
        _pay_tax_amount(state, config, q4_payment, sim_year, tax_result=tax_result)

    # total_taxes_paid: quarterly payments already added their amounts,
    # so only add the Q4 true-up portion (may be negative for overpayment credit)
    state.total_taxes_paid += max(q4_payment, 0)
    if q4_payment < 0:
        state.total_taxes_paid += q4_payment  # negative → reduces total

    state.loss_carryforward = tax_result["loss_carryforward"]
    state.annual_tax_history.append(tax_result)

    # Reset accumulator for next year
    state.tax_year_accum = TaxYearAccumulator()
    state.quarterly_tax_paid_ytd = 0.0
