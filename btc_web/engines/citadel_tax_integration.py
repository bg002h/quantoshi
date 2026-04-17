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
                    tax_result: dict | None = None) -> tuple[float, float]:
    """Draw *amount* from taxable wrapper to pay a tax bill.

    Sources are tried in order: cash → reserves → investments → tax-deferred.
    Mutates *state* in place.

    Returns ``(paid, unpaid)`` — dollars actually paid and dollars that
    could not be covered from any source (portfolio exhausted). Callers
    must credit only ``paid`` to ``state.total_taxes_paid`` and surface
    ``unpaid`` as a spending shortfall.

    **Historical note**: the previous implementation attempted a gross-up
    (sell $gross = amount / (1 - effective_rate)) to pre-account for the
    LTCG/ordinary-income tax the sale itself would trigger. That logic
    had three bugs:
      1. Selling $gross from investments and crediting only (drawn -
         tax_on_sale) to the tax bill silently DROPPED tax_on_sale worth
         of portfolio value (never deposited anywhere).
      2. The same gain was already accumulated in
         tax_year_accum.lt_capital_gains, so compute_annual_tax would
         tax it AGAIN at year-end — double-counting.
      3. The LTCG rate was hard-coded at 15% (ignoring the 20% bracket
         and state tax), so the gross-up under-sold investments at high
         AGI.
    The architecturally cleaner fix: no gross-up here. Sell exactly the
    amount needed; the sale's realised gain is accumulated naturally and
    flows through ``compute_annual_tax`` at the next year boundary. This
    lets the simulation's normal annual-tax pipeline own all the rate
    math (inflated brackets, NIIT stacking, state tax, etc.) instead of
    duplicating an approximation here.
    """
    if amount <= 0:
        return (0.0, 0.0)

    tax_remaining = amount

    # 1. Cash — no tax consequence (principal).
    draw = min(state.cash, tax_remaining)
    state.cash -= draw
    tax_remaining -= draw

    # 2. Reserves — no tax consequence (principal; interest was accrued
    #    separately into tax_year_accum during step()).
    for i in range(len(state.reserves)):
        if tax_remaining <= 0:
            break
        draw = min(state.reserves[i], tax_remaining)
        state.reserves[i] -= draw
        tax_remaining -= draw

    # 3. Investments — record realised gain into accumulator (future tax
    #    via compute_annual_tax); credit the FULL sale proceeds to the
    #    current bill. No gross-up — the annual pipeline owns rate math.
    if tax_remaining > 0:
        for i in reversed(range(len(state.investments))):
            if tax_remaining <= 0:
                break
            current = state.investments[i]
            if current <= 0:
                continue
            sell_amount = min(current, tax_remaining)
            drawn, _gain = _sell_investments_tracked(state, config, i, sell_amount)
            tax_remaining -= drawn

    # 4. TD withdrawal — add to tax_deferred_withdrawals (ordinary-income
    #    bucket; taxed at year-end). Same no-gross-up principle.
    if tax_remaining > 0:
        td_avail = state.td_cash + sum(state.td_reserves) + sum(state.td_investments)
        td_draw = min(tax_remaining, td_avail)
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
            # TD withdrawals are ordinary income — feed the accumulator.
            if state.tax_year_accum is not None:
                state.tax_year_accum.tax_deferred_withdrawals += actual
            tax_remaining -= actual

    # 5. BTC — last-resort source. Sells from taxable lots (FIFO/LIFO via
    #    _sell_btc_tracked). Realised gain flows to tax_year_accum; next
    #    year's compute_annual_tax will include it.
    if tax_remaining > 0 and state.btc_stack > 0 and state.btc_price > 0:
        from .citadel_transactions import _sell_btc_tracked
        btc_value = state.btc_stack * state.btc_price
        sell_usd = min(btc_value, tax_remaining)
        sell_btc = sell_usd / state.btc_price
        _sell_btc_tracked(state, config, sell_btc)
        tax_remaining -= sell_usd

    paid = amount - max(tax_remaining, 0.0)
    unpaid = max(tax_remaining, 0.0)
    return (paid, unpaid)


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
    # Character-preserved carryforwards (§1212(b)); not scaled by quarter.
    ann.st_carryforward = state.st_carryforward
    ann.lt_carryforward = state.lt_carryforward
    ann.loss_carryforward = state.loss_carryforward  # legacy sum, for estimator only

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
        paid, unpaid = _pay_tax_amount(state, config, payment, sim_year)
        state.quarterly_tax_paid_ytd += paid
        state.total_taxes_paid += paid
        if unpaid > 0:
            # Portfolio exhausted before the estimated bill could be
            # covered. Surface as a spending shortfall; the year-boundary
            # true-up will reconcile in case the shortfall goes away.
            state.spending_shortfall += unpaid


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

    # Set loss carryforwards (ST + LT, §1212(b) character-preserved)
    state.tax_year_accum.st_carryforward = state.st_carryforward
    state.tax_year_accum.lt_carryforward = state.lt_carryforward
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
        paid, unpaid = _pay_tax_amount(state, config, q4_payment, sim_year,
                                        tax_result=tax_result)
        state.total_taxes_paid += paid
        if unpaid > 0:
            state.spending_shortfall += unpaid
    elif q4_payment < 0:
        # Overpayment credit. Credit the refund to cash AND decrement the
        # running total so the accumulator stays consistent with actual
        # cash outflow. Without the cash credit, earlier code decremented
        # total_taxes_paid for a refund that never materialized in state.cash.
        state.cash += -q4_payment
        state.total_taxes_paid += q4_payment  # negative → reduces total

    # Preserve character of the carryforward across years (§1212(b)).
    state.st_carryforward = tax_result.get("st_carryforward", 0.0)
    state.lt_carryforward = tax_result.get(
        "lt_carryforward", tax_result["loss_carryforward"]
    )
    state.loss_carryforward = state.st_carryforward + state.lt_carryforward
    state.annual_tax_history.append(tax_result)

    # Reset accumulator for next year
    state.tax_year_accum = TaxYearAccumulator()
    state.quarterly_tax_paid_ytd = 0.0
