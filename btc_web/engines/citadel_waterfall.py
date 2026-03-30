"""Citadel Planner — cost-ranked dynamic spending waterfall."""
from __future__ import annotations

from .citadel_types import (
    _WithdrawalSource, CitadelState, SimConfig, PriceModel, FREQ_PPY,
)
from .citadel_transactions import _sell_btc_tracked, _sell_investments_tracked

__all__ = [
    "_build_source_list", "_score_sources", "_rank_sources",
    "_max_draw_before_boundary", "_execute_draw", "_spending_waterfall",
]


def _build_source_list(state: "CitadelState", config: "SimConfig",
                       model: "PriceModel | None" = None) -> list[_WithdrawalSource]:
    """Enumerate all available withdrawal sources from current state."""
    sources = []
    ppy = FREQ_PPY.get(config.freq, 12)

    # Compute BTC opportunity cost horizon and growth
    _btc_growth = config.invest_bins[0]["return_rate"] / 100 if config.invest_bins else 0.10
    if model is not None and state.btc_price > 0:
        try:
            _q = config.selected_qs[len(config.selected_qs) // 2] if config.selected_qs else 0.25
            _p_now = float(model.price_at(_q, max(state.t, 0.5)))
            _p_fwd = float(model.price_at(_q, max(state.t + 10, 0.5)))
            if _p_now > 0:
                _btc_growth = (_p_fwd / _p_now) - 1
        except Exception:
            pass

    # Treasury horizon: remaining lifetime
    if config.birth_year:
        _age = config.start_yr + int(state.period / ppy) - config.birth_year
    else:
        _age = 0
    _tres_horizon = max(min(90 - _age, 40), 1)

    # TD horizon: RMD factor reduces effective compounding horizon
    # Before RMD age: ramps down as forced distributions approach
    # At/after RMD age: IRS actuarial factor IS the expected remaining years
    _td_horizon = 15  # default when no birth_year
    if config.birth_year:
        from .tax_data import RMD_FACTORS
        from .citadel import _rmd_start_age
        _rmd_start = _rmd_start_age(config.birth_year)
        if _age >= _rmd_start:
            _td_horizon = max(int(RMD_FACTORS.get(_age, 1.0)), 1)
        else:
            _td_horizon = min(15, max(_rmd_start - _age, 1))

    # BTC aggregate gain fraction
    _btc_gain_frac = 0.0
    if state.btc_stack > 0 and state.btc_price > 0:
        btc_value = state.btc_stack * state.btc_price
        lot_basis_total = sum(l.btc * l.cost_basis for l in state.tax_lots) if state.tax_lots else 0
        if btc_value > 0:
            _btc_gain_frac = max(1.0 - lot_basis_total / btc_value, 0.0)

    # --- Taxable sources ---
    if state.cash > 0.01:
        sources.append(_WithdrawalSource(
            key="cash", wrapper="taxable", asset_type="cash", index=0,
            available=state.cash, growth_rate=config.cash_rate / 100,
            horizon=15, gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=False, bracket_type="none",
        ))
    for i, rb in enumerate(config.reserve_bins):
        bal = state.reserves[i] if i < len(state.reserves) else 0
        if bal > 0.01:
            sources.append(_WithdrawalSource(
                key=f"reserve_{i}", wrapper="taxable", asset_type="reserve", index=i,
                available=bal, growth_rate=rb["rate"] / 100,
                horizon=_tres_horizon, gain_fraction=0.0, is_roth=False,
                is_bracket_sensitive=False, bracket_type="none",
            ))
    for i, ib in enumerate(config.invest_bins):
        bal = state.investments[i] if i < len(state.investments) else 0
        if bal > 0.01:
            basis = state.invest_cost_basis[i] if i < len(state.invest_cost_basis) else bal
            gf = max(1.0 - basis / bal, 0.0) if bal > 0 else 0.0
            sources.append(_WithdrawalSource(
                key=f"invest_{i}", wrapper="taxable", asset_type="invest", index=i,
                available=bal, growth_rate=ib.get("return_rate", ib.get("rate", 5.0)) / 100,
                horizon=15, gain_fraction=gf, is_roth=False,
                is_bracket_sensitive=True, bracket_type="ltcg",
            ))
    if state.btc_stack * max(state.btc_price, 0) > 0.01:
        sources.append(_WithdrawalSource(
            key="btc", wrapper="taxable", asset_type="btc", index=0,
            available=state.btc_stack * state.btc_price,
            growth_rate=_btc_growth if isinstance(_btc_growth, float) else 0.10,
            horizon=10, gain_fraction=_btc_gain_frac, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ltcg",
        ))

    # --- TD sources (tax_enabled only) ---
    if config.tax_enabled:
        if state.td_cash > 0.01:
            sources.append(_WithdrawalSource(
                key="td_cash", wrapper="td", asset_type="cash", index=0,
                available=state.td_cash, growth_rate=config.cash_rate / 100,
                horizon=_td_horizon, gain_fraction=0.0, is_roth=False,
                is_bracket_sensitive=True, bracket_type="ordinary",
            ))
        for i, rb in enumerate(config.reserve_bins):
            bal = state.td_reserves[i] if i < len(state.td_reserves) else 0
            if bal > 0.01:
                sources.append(_WithdrawalSource(
                    key=f"td_reserve_{i}", wrapper="td", asset_type="reserve", index=i,
                    available=bal, growth_rate=rb["rate"] / 100,
                    horizon=_td_horizon, gain_fraction=0.0, is_roth=False,
                    is_bracket_sensitive=True, bracket_type="ordinary",
                ))
        for i, ib in enumerate(config.invest_bins):
            bal = state.td_investments[i] if i < len(state.td_investments) else 0
            if bal > 0.01:
                sources.append(_WithdrawalSource(
                    key=f"td_invest_{i}", wrapper="td", asset_type="invest", index=i,
                    available=bal, growth_rate=ib.get("return_rate", ib.get("rate", 5.0)) / 100,
                    horizon=_td_horizon, gain_fraction=0.0, is_roth=False,
                    is_bracket_sensitive=True, bracket_type="ordinary",
                ))
        if state.td_btc_stack * max(state.btc_price, 0) > 0.01:
            sources.append(_WithdrawalSource(
                key="td_btc", wrapper="td", asset_type="btc", index=0,
                available=state.td_btc_stack * state.btc_price,
                growth_rate=_btc_growth if isinstance(_btc_growth, float) else 0.10,
                horizon=min(_td_horizon, 10), gain_fraction=0.0, is_roth=False,
                is_bracket_sensitive=True, bracket_type="ordinary",
            ))

    # --- TF (Roth) sources (tax_enabled only) ---
    if config.tax_enabled:
        tf_cash_res = state.tf_cash + sum(state.tf_reserves)
        if tf_cash_res > 0.01:
            sources.append(_WithdrawalSource(
                key="tf_cash_res", wrapper="tf", asset_type="cash", index=0,
                available=tf_cash_res, growth_rate=config.cash_rate / 100,
                horizon=15, gain_fraction=0.0, is_roth=True,
                is_bracket_sensitive=False, bracket_type="none",
            ))
        tf_inv = sum(state.tf_investments)
        if tf_inv > 0.01:
            avg_rate = sum(ib.get("return_rate", 5.0) for ib in config.invest_bins) / max(len(config.invest_bins), 1)
            sources.append(_WithdrawalSource(
                key="tf_invest", wrapper="tf", asset_type="invest", index=0,
                available=tf_inv, growth_rate=avg_rate / 100,
                horizon=15, gain_fraction=0.0, is_roth=True,
                is_bracket_sensitive=False, bracket_type="none",
            ))
        if state.tf_btc_stack * max(state.btc_price, 0) > 0.01:
            sources.append(_WithdrawalSource(
                key="tf_btc", wrapper="tf", asset_type="btc", index=0,
                available=state.tf_btc_stack * state.btc_price,
                growth_rate=_btc_growth if isinstance(_btc_growth, float) else 0.10,
                horizon=10, gain_fraction=0.0, is_roth=True,
                is_bracket_sensitive=False, bracket_type="none",
            ))

    return sources


def _score_sources(sources: list[_WithdrawalSource], state: CitadelState,
                   config: SimConfig, model: "PriceModel | None" = None) -> None:
    """Compute cost-per-dollar for each source. Mutates source.cost in place."""
    from .tax import _inflate_brackets
    from .tax_data import (FEDERAL_BRACKETS_TCJA, FEDERAL_BRACKETS_SUNSET,
                           LTCG_BRACKETS, NIIT_RATE, NIIT_THRESHOLD,
                           STANDARD_DEDUCTION_TCJA, STANDARD_DEDUCTION_SUNSET)
    from .citadel import _get_state_rate

    state_rate = _get_state_rate(config) / 100  # as fraction

    # Current bracket position from accumulator
    _years_from_base = 0
    _ordinary_ytd = 0.0
    _ltcg_ytd = 0.0
    _magi = 0.0
    if state.tax_year_accum is not None:
        a = state.tax_year_accum
        _ordinary_ytd = (a.tax_deferred_withdrawals + a.interest_income
                         + a.treasury_interest + a.other_income)
        _ltcg_ytd = max(a.lt_capital_gains - a.lt_capital_losses, 0)
        _magi = _ordinary_ytd + _ltcg_ytd + max(a.st_capital_gains - a.st_capital_losses, 0)

    ppy = FREQ_PPY.get(config.freq, 12)
    sim_year = config.start_yr + int(state.period / ppy)
    _years_from_base = max(sim_year - 2025, 0)
    infl = config.inflation / 100

    # Inflate brackets
    if config.tcja_sunset:
        _ord_brackets = _inflate_brackets(FEDERAL_BRACKETS_SUNSET[config.filing_status], _years_from_base, infl)
        _std_ded = STANDARD_DEDUCTION_SUNSET[config.filing_status] * (1 + infl) ** _years_from_base
    else:
        _ord_brackets = _inflate_brackets(FEDERAL_BRACKETS_TCJA[config.filing_status], _years_from_base, infl)
        _std_ded = STANDARD_DEDUCTION_TCJA[config.filing_status] * (1 + infl) ** _years_from_base
    _ltcg_brackets = _inflate_brackets(LTCG_BRACKETS[config.filing_status], _years_from_base, infl)
    _niit_threshold = NIIT_THRESHOLD[config.filing_status]  # NOT inflation-indexed

    # Marginal ordinary rate at current YTD position
    _ord_taxable = max(_ordinary_ytd - _std_ded, 0)
    if _ordinary_ytd < _std_ded:
        _marginal_ord = 0.0  # still within standard deduction — no tax
    else:
        _marginal_ord = 0.10  # default (first bracket)
        for upper, rate in _ord_brackets:
            if _ord_taxable < upper:
                _marginal_ord = rate
                break

    # LTCG rate at stacked position (ordinary + LTCG)
    _stacked = _ord_taxable + _ltcg_ytd
    _marginal_ltcg = 0.15  # default
    for upper, rate in _ltcg_brackets:
        if _stacked < upper:
            _marginal_ltcg = rate
            break

    # NIIT applies?
    _niit = NIIT_RATE if _magi > _niit_threshold else 0.0

    for s in sources:
        # Tax cost per dollar
        if s.wrapper == "tf":
            tax_cost = 0.0  # Roth — no tax
        elif s.wrapper == "td":
            tax_cost = _marginal_ord + state_rate
        elif s.asset_type in ("invest", "btc"):
            tax_cost = (_marginal_ltcg + _niit + state_rate) * s.gain_fraction
        else:
            tax_cost = 0.0  # taxable cash/reserves — principal

        # Opportunity cost
        if s.wrapper == "td":
            # TD grows gross, taxed on withdrawal → reduce by (1 - marginal_rate)
            opp = ((1 + s.growth_rate) ** s.horizon - 1) * (1 - _marginal_ord)
        elif s.wrapper == "taxable" and s.asset_type == "reserve":
            # Taxable treasury: after-tax interest compounding
            # Treasury interest is state-exempt (US law) — only federal tax on coupons
            after_tax_rate = s.growth_rate * (1 - _marginal_ord)
            opp = (1 + max(after_tax_rate, 0)) ** s.horizon - 1
        else:
            opp = (1 + s.growth_rate) ** s.horizon - 1

        s.cost = tax_cost + max(opp, 0.0)


def _rank_sources(sources: list[_WithdrawalSource]) -> list[_WithdrawalSource]:
    """Sort all sources by cost ascending. The cost function already accounts
    for tax-free compounding (Roth has zero tax but full opportunity cost),
    so Roth naturally ranks expensive — no need to force it last."""
    return sorted(sources, key=lambda s: s.cost)


def _max_draw_before_boundary(state: CitadelState, config: SimConfig,
                               source: _WithdrawalSource) -> float:
    """Max dollars drawable before crossing a tax bracket boundary.
    Returns float("inf") for non-bracket-sensitive sources.
    """
    if not source.is_bracket_sensitive:
        return float("inf")

    from .tax import _inflate_brackets
    from .tax_data import (FEDERAL_BRACKETS_TCJA, FEDERAL_BRACKETS_SUNSET,
                           LTCG_BRACKETS, NIIT_THRESHOLD,
                           STANDARD_DEDUCTION_TCJA, STANDARD_DEDUCTION_SUNSET)

    ppy = FREQ_PPY.get(config.freq, 12)
    sim_year = config.start_yr + int(state.period / ppy)
    yrs = max(sim_year - 2025, 0)
    infl = config.inflation / 100

    if config.tcja_sunset:
        ord_brackets = _inflate_brackets(FEDERAL_BRACKETS_SUNSET[config.filing_status], yrs, infl)
        std_ded = STANDARD_DEDUCTION_SUNSET[config.filing_status] * (1 + infl) ** yrs
    else:
        ord_brackets = _inflate_brackets(FEDERAL_BRACKETS_TCJA[config.filing_status], yrs, infl)
        std_ded = STANDARD_DEDUCTION_TCJA[config.filing_status] * (1 + infl) ** yrs

    # Current positions from accumulator
    ordinary_ytd = 0.0
    ltcg_ytd = 0.0
    magi = 0.0
    if state.tax_year_accum is not None:
        a = state.tax_year_accum
        ordinary_ytd = (a.tax_deferred_withdrawals + a.interest_income
                        + a.treasury_interest + a.other_income)
        ltcg_ytd = max(a.lt_capital_gains - a.lt_capital_losses, 0)
        stcg_ytd = max(a.st_capital_gains - a.st_capital_losses, 0)
        magi = ordinary_ytd + ltcg_ytd + stcg_ytd

    distances = []

    if source.bracket_type == "ordinary":
        # Distance to next ordinary bracket (in gross income space)
        if ordinary_ytd < std_ded:
            # Still within standard deduction — distance includes remaining cushion
            # plus the first bracket's full width
            remaining_ded = std_ded - ordinary_ytd
            distances.append(remaining_ded + ord_brackets[0][0])
        else:
            ord_taxable = ordinary_ytd - std_ded
            for upper, _rate in ord_brackets:
                if ord_taxable < upper - 0.01:  # skip brackets within float rounding
                    distances.append(upper - ord_taxable)
                    break

        # NIIT threshold (MAGI-based, NOT inflated)
        niit_thresh = NIIT_THRESHOLD[config.filing_status]
        if magi < niit_thresh:
            distances.append(niit_thresh - magi)

    elif source.bracket_type == "ltcg":
        # LTCG brackets stacked on ordinary taxable income
        ord_taxable = max(ordinary_ytd - std_ded, 0)
        stacked = ord_taxable + ltcg_ytd
        ltcg_brackets = _inflate_brackets(LTCG_BRACKETS[config.filing_status], yrs, infl)
        for upper, _rate in ltcg_brackets:
            if stacked < upper - 0.01:  # skip brackets within float rounding
                gain_distance = upper - stacked  # distance in gain-space
                # Convert to sale-space: if gain_fraction=0.5, need to sell $2 to generate $1 gain
                gf = max(source.gain_fraction, 0.01)  # avoid div-by-zero
                distances.append(gain_distance / gf)
                break

        # NIIT threshold (MAGI-based). For LTCG sources, the MAGI increase per
        # dollar sold = gain_fraction (only the gain portion increases MAGI)
        niit_thresh = NIIT_THRESHOLD[config.filing_status]
        if magi < niit_thresh:
            magi_distance = niit_thresh - magi
            gf = max(source.gain_fraction, 0.01)
            distances.append(magi_distance / gf)

    if not distances:
        return float("inf")  # in top bracket, no boundary ahead
    return max(min(distances), 0.0)


def _execute_draw(state: CitadelState, config: SimConfig,
                  source: _WithdrawalSource, amount: float) -> None:
    """Execute a withdrawal from the specified source. Mutates state."""
    if amount <= 0:
        return

    if source.wrapper == "taxable":
        if source.asset_type == "cash":
            state.cash -= min(amount, state.cash)
        elif source.asset_type == "reserve":
            state.reserves[source.index] -= min(amount, state.reserves[source.index])
        elif source.asset_type == "invest":
            _sell_investments_tracked(state, config, source.index, amount)
        elif source.asset_type == "btc":
            if state.btc_price > 0:
                btc_to_sell = amount / state.btc_price
                _sell_btc_tracked(state, config, btc_to_sell)

    elif source.wrapper == "td":
        remaining = amount
        if source.asset_type == "cash":
            d = min(state.td_cash, remaining); state.td_cash -= d; remaining -= d
        elif source.asset_type == "reserve":
            d = min(state.td_reserves[source.index], remaining)
            state.td_reserves[source.index] -= d; remaining -= d
        elif source.asset_type == "invest":
            d = min(state.td_investments[source.index], remaining)
            state.td_investments[source.index] -= d; remaining -= d
        elif source.asset_type == "btc":
            if state.btc_price > 0 and state.td_btc_stack > 0:
                btc_val = state.td_btc_stack * state.btc_price
                d = min(btc_val, remaining)
                state.td_btc_stack -= d / state.btc_price
                remaining -= d
        actual = amount - remaining
        if state.tax_year_accum is not None and actual > 0:
            state.tax_year_accum.tax_deferred_withdrawals += actual

    elif source.wrapper == "tf":
        remaining = amount
        if source.asset_type == "cash":
            # TF cash + reserves combined source — draw cash first, then reserves
            d = min(state.tf_cash, remaining); state.tf_cash -= d; remaining -= d
            for i in range(len(state.tf_reserves)):
                if remaining <= 0:
                    break
                d = min(state.tf_reserves[i], remaining)
                state.tf_reserves[i] -= d; remaining -= d
        elif source.asset_type == "invest":
            for i in reversed(range(len(state.tf_investments))):
                if remaining <= 0:
                    break
                d = min(state.tf_investments[i], remaining)
                state.tf_investments[i] -= d; remaining -= d
        elif source.asset_type == "btc":
            if state.btc_price > 0 and state.tf_btc_stack > 0:
                btc_val = state.tf_btc_stack * state.btc_price
                d = min(btc_val, remaining)
                state.tf_btc_stack -= d / state.btc_price
                remaining -= d
        actual = amount - remaining
        if state.tax_year_accum is not None and actual > 0:
            state.tax_year_accum.roth_withdrawals += actual


def _spending_waterfall(state: CitadelState, config: SimConfig,
                        amount: float,
                        model: "PriceModel | None" = None) -> float:
    """Draw `amount` from accounts using cost-ranked dynamic ordering.
    Returns unmet shortfall. Mutates state in place.

    Computes tax cost + opportunity cost for each available source,
    draws from cheapest first, re-ranks at bracket boundaries.
    Roth sources always rank after all non-Roth.
    """
    remaining = amount
    if remaining <= 0:
        return 0.0

    sources = _build_source_list(state, config, model)
    if not sources:
        return remaining

    while remaining > 0.01 and sources:
        _score_sources(sources, state, config, model)
        ranked = _rank_sources(sources)

        drew_something = False
        for best in ranked:
            if best.available < 0.01:
                continue

            max_draw = _max_draw_before_boundary(state, config, best)
            if max_draw < 0.01:
                continue

            draw = min(remaining, best.available, max_draw)
            _execute_draw(state, config, best, draw)
            remaining -= draw

            best.available -= draw
            drew_something = True
            break  # re-rank with updated state

        if not drew_something:
            # All sources were skipped on bracket boundaries, but money
            # remains. A shortfall is infinitely more expensive than any
            # tax bracket — draw from cheapest available, ignoring caps.
            for best in ranked:
                if best.available < 0.01:
                    continue
                draw = min(remaining, best.available)
                _execute_draw(state, config, best, draw)
                remaining -= draw
                best.available -= draw
                drew_something = True
                break

            if not drew_something:
                break  # truly exhausted

        sources = [s for s in sources if s.available > 0.01]

    return max(remaining, 0.0)
