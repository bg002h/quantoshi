"""Citadel Planner — cost-ranked dynamic spending waterfall."""
from __future__ import annotations

from dataclasses import dataclass

from .citadel_types import (
    _WithdrawalSource, CitadelState, SimConfig, PriceModel, FREQ_PPY,
)
from .citadel_transactions import _sell_btc_tracked, _sell_investments_tracked

__all__ = [
    "_build_source_list", "_score_sources", "_rank_sources",
    "_max_draw_before_boundary", "_execute_draw", "_spending_waterfall",
    "TaxContext", "_inflate_tax_context",
]


@dataclass(frozen=True)
class TaxContext:
    """Pre-inflated federal + state tax brackets for a single simulation year."""
    ord_brackets: list
    std_ded: float
    ltcg_brackets: list
    niit_threshold: float
    sim_year: int
    infl: float


def _inflate_tax_context(config: "SimConfig", sim_year: int) -> TaxContext:
    """Build a TaxContext with federal brackets inflated to `sim_year`.

    Centralizes the inflation math that was previously duplicated in
    _score_sources, _max_draw_before_boundary, and _pay_tax_amount.
    """
    from .tax import _inflate_brackets
    from .tax_data import (
        FEDERAL_BRACKETS_TCJA, FEDERAL_BRACKETS_SUNSET,
        LTCG_BRACKETS, NIIT_THRESHOLD,
        STANDARD_DEDUCTION_TCJA, STANDARD_DEDUCTION_SUNSET,
    )
    yrs = max(sim_year - 2025, 0)
    infl = config.inflation / 100

    if config.tcja_sunset:
        ord_brackets = _inflate_brackets(FEDERAL_BRACKETS_SUNSET[config.filing_status], yrs, infl)
        std_ded = STANDARD_DEDUCTION_SUNSET[config.filing_status] * (1 + infl) ** yrs
    else:
        ord_brackets = _inflate_brackets(FEDERAL_BRACKETS_TCJA[config.filing_status], yrs, infl)
        std_ded = STANDARD_DEDUCTION_TCJA[config.filing_status] * (1 + infl) ** yrs

    ltcg_brackets = _inflate_brackets(LTCG_BRACKETS[config.filing_status], yrs, infl)
    niit_threshold = NIIT_THRESHOLD[config.filing_status]  # NOT inflation-indexed
    return TaxContext(ord_brackets, std_ded, ltcg_brackets, niit_threshold,
                      sim_year, infl)


_REGIME_ATTR_MAP = {
    "res_short": "res_short_regime",
    "res_med":   "res_med_regime",
    "res_long":  "res_long_regime",
    "equity":    "equity_regime",
    "bond":      "bond_regime",
}


def _expected_annual_rate(config: "SimConfig", state: "CitadelState",
                          asset_key: str, fallback_pct: float) -> float:
    """Return the expected ANNUAL rate for a source, regime-aware.

    When ``config.asset_return_model == "markov"`` and the simulation
    state carries a regime index for ``asset_key``, return the current
    regime's conditional mean return annualised to per-year. Otherwise
    fall back to the user-entered Fixed Rates value (``fallback_pct`` is
    the PERCENTAGE value — e.g. 10.0 for 10%/yr).

    Fixes the Citadel-MC scoring inconsistency: previously the waterfall
    always used fixed rates even when the simulation was running
    regime-conditional Markov returns, so opportunity-cost scoring
    ignored the whole point of the "Historical Regimes" mode.
    """
    fallback = fallback_pct / 100.0
    if (config.asset_return_model != "markov"
            or config.asset_matrices is None
            or asset_key not in (config.asset_matrices or {})):
        return fallback
    rattr = _REGIME_ATTR_MAP.get(asset_key)
    if rattr is None:
        return fallback
    regime = getattr(state, rattr, 0)
    am = config.asset_matrices[asset_key]
    bin_means = am.get("bin_means")
    if bin_means is None or regime >= len(bin_means):
        return fallback
    ppy = FREQ_PPY.get(config.freq, 12)
    per_period = float(bin_means[regime])
    # Compound per-period return to annual. Per-period returns from the
    # asset_matrices are already at the sim's frequency (monthly by default).
    try:
        return (1.0 + per_period) ** ppy - 1.0
    except (OverflowError, ValueError):
        return fallback


def _build_source_list(state: "CitadelState", config: "SimConfig",
                       model: "PriceModel | None" = None) -> list[_WithdrawalSource]:
    """Enumerate all available withdrawal sources from current state."""
    sources = []
    ppy = FREQ_PPY.get(config.freq, 12)

    # Common horizon (C4): remaining sim years. Replaces hardcoded 15y/10y
    # per-source horizons that distorted ranking.
    _cur_year = config.start_yr + int(state.period / ppy)
    _sim_horizon = max(config.end_yr - _cur_year, 1)

    # Compute BTC opportunity cost growth rate.
    #
    # Design intent (2026-04-17): BTC-preservation is NOT an unconditional
    # invariant — it's an emergent property of comparing the model's
    # predicted BTC rate of return against the other assets' rates. If the
    # model expects BTC to outgrow cash/bonds/equities, BTC ranks as the
    # most expensive source to sell (high opportunity cost) and is drawn
    # last. Conversely, if the model predicts weak BTC growth, the user
    # is better off selling BTC first and preserving higher-yielding assets.
    #
    # The forward window for estimating BTC's annualised growth matches the
    # sim horizon so the comparison is apples-to-apples with other sources
    # scored over the same future window.
    _btc_growth = config.invest_bins[0]["return_rate"] / 100 if config.invest_bins else 0.10
    if model is not None and state.btc_price > 0:
        try:
            _q = config.selected_qs[len(config.selected_qs) // 2] if config.selected_qs else 0.25
            _p_now = float(model.price_at(_q, max(state.t, 0.5)))
            _p_fwd = float(model.price_at(_q, max(state.t + _sim_horizon, 0.5)))
            if _p_now > 0 and _p_fwd > 0:
                _btc_growth = (_p_fwd / _p_now) ** (1.0 / _sim_horizon) - 1.0
        except Exception:
            pass

    # Treasury horizon: remaining lifetime (age-clamped) if birth_year given;
    # otherwise fall back to sim horizon. Treasury bins remain "near the end
    # of life" assets — the clamp ensures we don't compound past age 90.
    if config.birth_year:
        _age = _cur_year - config.birth_year
        _tres_horizon = max(min(90 - _age, _sim_horizon), 1)
    else:
        _age = 0
        _tres_horizon = _sim_horizon

    # TD horizon: RMD factor reduces effective compounding horizon.
    # Before RMD age: ramps down as forced distributions approach (capped
    # at 15y to reflect tax-law uncertainty beyond that).
    # At/after RMD age: IRS actuarial factor IS the expected remaining years.
    _td_horizon = min(_sim_horizon, 15)  # default when no birth_year
    if config.birth_year:
        from .tax_data import RMD_FACTORS
        from .citadel_tax_integration import _rmd_start_age
        _rmd_start = _rmd_start_age(config.birth_year)
        if _age >= _rmd_start:
            _td_horizon = max(int(RMD_FACTORS.get(_age, 1.0)), 1)
        else:
            _td_horizon = min(15, _sim_horizon, max(_rmd_start - _age, 1))

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
            horizon=_sim_horizon, gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=False, bracket_type="none",
        ))
    # Reserve bins map positionally to res_short / res_med / res_long asset keys
    # in asset_matrices when the Markov model is active.
    _reserve_keys = ["res_short", "res_med", "res_long"]
    for i, rb in enumerate(config.reserve_bins):
        bal = state.reserves[i] if i < len(state.reserves) else 0
        if bal > 0.01:
            asset_key = _reserve_keys[i] if i < len(_reserve_keys) else ""
            rate = _expected_annual_rate(config, state, asset_key, rb["rate"])
            sources.append(_WithdrawalSource(
                key=f"reserve_{i}", wrapper="taxable", asset_type="reserve", index=i,
                available=bal, growth_rate=rate,
                horizon=_tres_horizon, gain_fraction=0.0, is_roth=False,
                is_bracket_sensitive=False, bracket_type="none",
            ))
    # Invest bins: index 0 = equity, index 1 = bond (by position — the same
    # convention citadel_step.py uses when it advances Markov regimes).
    _invest_keys = ["equity", "bond"]
    for i, ib in enumerate(config.invest_bins):
        bal = state.investments[i] if i < len(state.investments) else 0
        if bal > 0.01:
            basis = state.invest_cost_basis[i] if i < len(state.invest_cost_basis) else bal
            gf = max(1.0 - basis / bal, 0.0) if bal > 0 else 0.0
            asset_key = _invest_keys[i] if i < len(_invest_keys) else ""
            fallback_pct = ib.get("return_rate", ib.get("rate", 5.0))
            rate = _expected_annual_rate(config, state, asset_key, fallback_pct)
            sources.append(_WithdrawalSource(
                key=f"invest_{i}", wrapper="taxable", asset_type="invest", index=i,
                available=bal, growth_rate=rate,
                horizon=_sim_horizon, gain_fraction=gf, is_roth=False,
                is_bracket_sensitive=True, bracket_type="ltcg",
            ))
    if state.btc_stack * max(state.btc_price, 0) > 0.01:
        sources.append(_WithdrawalSource(
            key="btc", wrapper="taxable", asset_type="btc", index=0,
            available=state.btc_stack * state.btc_price,
            growth_rate=_btc_growth if isinstance(_btc_growth, float) else 0.10,
            horizon=_sim_horizon, gain_fraction=_btc_gain_frac, is_roth=False,
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
                asset_key = _reserve_keys[i] if i < len(_reserve_keys) else ""
                rate = _expected_annual_rate(config, state, asset_key, rb["rate"])
                sources.append(_WithdrawalSource(
                    key=f"td_reserve_{i}", wrapper="td", asset_type="reserve", index=i,
                    available=bal, growth_rate=rate,
                    horizon=_td_horizon, gain_fraction=0.0, is_roth=False,
                    is_bracket_sensitive=True, bracket_type="ordinary",
                ))
        for i, ib in enumerate(config.invest_bins):
            bal = state.td_investments[i] if i < len(state.td_investments) else 0
            if bal > 0.01:
                asset_key = _invest_keys[i] if i < len(_invest_keys) else ""
                fallback_pct = ib.get("return_rate", ib.get("rate", 5.0))
                rate = _expected_annual_rate(config, state, asset_key, fallback_pct)
                sources.append(_WithdrawalSource(
                    key=f"td_invest_{i}", wrapper="td", asset_type="invest", index=i,
                    available=bal, growth_rate=rate,
                    horizon=_td_horizon, gain_fraction=0.0, is_roth=False,
                    is_bracket_sensitive=True, bracket_type="ordinary",
                ))
        if state.td_btc_stack * max(state.btc_price, 0) > 0.01:
            sources.append(_WithdrawalSource(
                key="td_btc", wrapper="td", asset_type="btc", index=0,
                available=state.td_btc_stack * state.btc_price,
                growth_rate=_btc_growth if isinstance(_btc_growth, float) else 0.10,
                horizon=_td_horizon, gain_fraction=0.0, is_roth=False,
                is_bracket_sensitive=True, bracket_type="ordinary",
            ))

    # --- TF (Roth) sources (tax_enabled only) ---
    if config.tax_enabled:
        tf_cash_res = state.tf_cash + sum(state.tf_reserves)
        if tf_cash_res > 0.01:
            sources.append(_WithdrawalSource(
                key="tf_cash_res", wrapper="tf", asset_type="cash", index=0,
                available=tf_cash_res, growth_rate=config.cash_rate / 100,
                horizon=_sim_horizon, gain_fraction=0.0, is_roth=True,
                is_bracket_sensitive=False, bracket_type="none",
            ))
        tf_inv = sum(state.tf_investments)
        if tf_inv > 0.01:
            # Average across invest bins; each bin's rate is regime-aware when
            # asset_return_model == "markov".
            if config.invest_bins:
                rates = []
                for i, ib in enumerate(config.invest_bins):
                    asset_key = _invest_keys[i] if i < len(_invest_keys) else ""
                    fallback_pct = ib.get("return_rate", 5.0)
                    rates.append(_expected_annual_rate(
                        config, state, asset_key, fallback_pct))
                avg_rate = sum(rates) / len(rates)
            else:
                avg_rate = 0.05
            sources.append(_WithdrawalSource(
                key="tf_invest", wrapper="tf", asset_type="invest", index=0,
                available=tf_inv, growth_rate=avg_rate,
                horizon=_sim_horizon, gain_fraction=0.0, is_roth=True,
                is_bracket_sensitive=False, bracket_type="none",
            ))
        if state.tf_btc_stack * max(state.btc_price, 0) > 0.01:
            sources.append(_WithdrawalSource(
                key="tf_btc", wrapper="tf", asset_type="btc", index=0,
                available=state.tf_btc_stack * state.btc_price,
                growth_rate=_btc_growth if isinstance(_btc_growth, float) else 0.10,
                horizon=_sim_horizon, gain_fraction=0.0, is_roth=True,
                is_bracket_sensitive=False, bracket_type="none",
            ))

    return sources


def _score_sources(sources: list[_WithdrawalSource], state: CitadelState,
                   config: SimConfig, model: "PriceModel | None" = None) -> None:
    """Compute cost-per-dollar for each source. Mutates source.cost in place."""
    from .tax_data import NIIT_RATE
    from .citadel_tax_integration import _get_state_rate

    state_rate = _get_state_rate(config) / 100  # as fraction

    # Current bracket position from accumulator
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
    tc = _inflate_tax_context(config, sim_year)
    _ord_brackets = tc.ord_brackets
    _std_ded = tc.std_ded
    _ltcg_brackets = tc.ltcg_brackets
    _niit_threshold = tc.niit_threshold

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

    # Common discount rate (C1): PV-discount future opportunity cost at the
    # user's cash rate — our risk-free "do-nothing" alternative. Without
    # discounting, a long-horizon compounded total return (e.g. 1.10^30 - 1
    # ≈ 16.4) completely swamps one-time tax costs (e.g. 0.19) regardless of
    # growth magnitude, so the ranking degenerates to "whatever grows
    # fastest, skip." PV-discounting ties the opp cost to the nominal-$
    # today, so sources with growth rates at or below cash rate contribute
    # ~0 opp cost and tax starts to matter again.
    _discount = max(config.cash_rate / 100, 0.001)

    for s in sources:
        # Tax cost per dollar drawn
        if s.wrapper == "tf":
            tax_cost = 0.0  # Roth — no tax
        elif s.wrapper == "td":
            # Ordinary income tax (federal + state) on full withdrawal,
            # paid at year-end from taxable wrappers.
            tax_cost = _marginal_ord + state_rate
        elif s.asset_type in ("invest", "btc"):
            # LTCG on gain portion only (approximation: uses current
            # gain_fraction as proxy for future gain-fraction at sale).
            tax_cost = (_marginal_ltcg + _niit + state_rate) * s.gain_fraction
        else:
            tax_cost = 0.0  # taxable cash/reserves — principal only

        # PV-discounted opportunity cost per $1 drawn (C1 + C3).
        #
        # Rationale: "If I *don't* draw this $1 today, what is its PV of
        # future spending (in today's dollars)?" For growth > discount,
        # pv_opp is positive (cost of drawing now). For growth < discount,
        # pv_opp is negative, which correctly flags the source as beneficial
        # to draw *first* — the asset is decaying vs the risk-free baseline.
        # We do NOT clamp at 0: below-discount sources need to rank cheaper
        # than zero-cost sources (cash at discount rate), so the waterfall
        # drains them preferentially.
        horizon = s.horizon
        discount_factor = (1 + _discount) ** horizon
        if s.wrapper == "td":
            # TD: gross growth at rate r, but eventually withdrawn and taxed
            # at the (future) marginal ordinary rate. Net future spending
            # per $1 left-untouched = (1-t_future)(1+r)^h, approximated with
            # current marginal rates.
            combined_rate = _marginal_ord + state_rate
            after_tax_future = (1 - combined_rate) * (1 + s.growth_rate) ** horizon
            pv_opp = after_tax_future / discount_factor - (1 - combined_rate)
        elif s.wrapper == "taxable" and s.asset_type == "reserve":
            # Taxable treasury: interest taxed annually at ordinary rate,
            # state-exempt. Effective compounding = r(1-t_ord).
            after_tax_rate = s.growth_rate * (1 - _marginal_ord)
            pv_opp = (1 + after_tax_rate) ** horizon / discount_factor - 1
        else:
            # Cash, taxable investments, BTC, and Roth all compound at
            # gross r with tax-on-sale captured separately in tax_cost.
            pv_opp = (1 + s.growth_rate) ** horizon / discount_factor - 1

        s.cost = tax_cost + pv_opp


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

    from .tax_data import NIIT_THRESHOLD

    ppy = FREQ_PPY.get(config.freq, 12)
    sim_year = config.start_yr + int(state.period / ppy)
    tc = _inflate_tax_context(config, sim_year)
    ord_brackets = tc.ord_brackets
    std_ded = tc.std_ded

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
        # LTCG brackets stacked on ordinary taxable income.
        #
        # C5 fix: when gain_fraction is genuinely ~0 (no unrealised gain),
        # selling this asset does NOT increase LTCG — the bracket boundary
        # is effectively infinite. Previously the code clamped gain_fraction
        # to 0.01, inventing a fake bracket distance that over-capped draws
        # against zero-gain assets. Now we only append a distance when
        # gain_fraction actually contributes to taxable gain.
        if source.gain_fraction > 1e-6:
            ord_taxable = max(ordinary_ytd - std_ded, 0)
            stacked = ord_taxable + ltcg_ytd
            ltcg_brackets = tc.ltcg_brackets
            for upper, _rate in ltcg_brackets:
                if stacked < upper - 0.01:  # skip brackets within float rounding
                    gain_distance = upper - stacked  # distance in gain-space
                    # Convert to sale-space: if gain_fraction=0.5, need to sell $2 to generate $1 gain
                    distances.append(gain_distance / source.gain_fraction)
                    break

            # NIIT threshold (MAGI-based). For LTCG sources, MAGI increase per
            # dollar sold = gain_fraction (only the gain portion increases MAGI)
            niit_thresh = NIIT_THRESHOLD[config.filing_status]
            if magi < niit_thresh:
                magi_distance = niit_thresh - magi
                distances.append(magi_distance / source.gain_fraction)

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
