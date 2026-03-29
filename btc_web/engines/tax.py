"""Tax computation engine — brackets, LTCG stacking, NIIT, capital loss netting.

Implements IRS Section 1(h) netting, progressive bracket computation,
and a full annual tax pipeline for the Citadel Planner simulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .tax_data import (
    FEDERAL_BRACKETS_TCJA,
    FEDERAL_BRACKETS_SUNSET,
    LTCG_BRACKETS,
    NIIT_RATE,
    NIIT_THRESHOLD,
    STANDARD_DEDUCTION_TCJA,
    STANDARD_DEDUCTION_SUNSET,
)


# ---------------------------------------------------------------------------
# Bracket helpers
# ---------------------------------------------------------------------------

def _inflate_brackets(
    brackets: list[tuple[float, float]], years: int, rate: float
) -> list[tuple[float, float]]:
    """Inflate bracket thresholds by (1+rate)^years.  inf stays inf."""
    factor = (1 + rate) ** years
    return [
        (thresh * factor if not math.isinf(thresh) else float("inf"), mrate)
        for thresh, mrate in brackets
    ]


def apply_progressive_brackets(
    taxable_income: float, brackets: list[tuple[float, float]]
) -> float:
    """Compute tax on *taxable_income* using progressive *brackets*.

    *brackets* is a sorted list of ``(upper_threshold, marginal_rate)``.
    The last bracket's threshold is ``inf``.
    """
    if taxable_income <= 0:
        return 0.0
    tax = 0.0
    prev = 0.0
    for upper, rate in brackets:
        if taxable_income <= prev:
            break
        span = min(taxable_income, upper) - prev
        tax += span * rate
        prev = upper
    return tax


# ---------------------------------------------------------------------------
# LTCG with stacking
# ---------------------------------------------------------------------------

def compute_ltcg_tax(
    taxable_ltcg: float,
    stacking_base: float,
    filing_status: str,
) -> float:
    """LTCG tax using the stacking rule (LTCG brackets start where ordinary
    income left off).

    ``stacking_base`` is the ordinary taxable income that has already
    "filled up" the lower LTCG brackets.
    """
    if taxable_ltcg <= 0:
        return 0.0
    brk = LTCG_BRACKETS[filing_status]
    total = apply_progressive_brackets(stacking_base + taxable_ltcg, brk)
    base = apply_progressive_brackets(stacking_base, brk)
    return total - base


# ---------------------------------------------------------------------------
# Capital-gain netting (IRS §1(h))
# ---------------------------------------------------------------------------

@dataclass
class CapitalGainResult:
    """Result of IRS capital-gain netting."""
    net_st: float = 0.0
    net_lt: float = 0.0
    loss_deduction: float = 0.0      # up to $3,000 against ordinary income
    new_carryforward: float = 0.0    # excess loss carried to next year


def net_capital_gains(
    st_gains: float,
    st_losses: float,
    lt_gains: float,
    lt_losses: float,
    carryforward: float,
) -> CapitalGainResult:
    """IRS Section 1(h) netting of short-term and long-term capital gains.

    *carryforward* is prior-year loss carryforward (applied as LT loss,
    v1 simplification).
    """
    net_st = st_gains - st_losses
    net_lt = lt_gains - lt_losses - carryforward

    # Cross-category offset: if one is negative and the other positive
    if net_st < 0 and net_lt > 0:
        combined = net_st + net_lt
        if combined >= 0:
            net_lt = combined
            net_st = 0.0
        else:
            net_lt = 0.0
            net_st = combined
    elif net_lt < 0 and net_st > 0:
        combined = net_st + net_lt
        if combined >= 0:
            net_st = combined
            net_lt = 0.0
        else:
            net_st = 0.0
            net_lt = combined

    # Remaining net loss
    total_net = net_st + net_lt
    if total_net < 0:
        loss_deduction = min(-total_net, 3_000.0)
        new_carry = -total_net - loss_deduction
        # Zero out the gain fields (all loss has been accounted for)
        net_st = 0.0
        net_lt = 0.0
    else:
        loss_deduction = 0.0
        new_carry = 0.0

    return CapitalGainResult(
        net_st=net_st,
        net_lt=net_lt,
        loss_deduction=loss_deduction,
        new_carryforward=new_carry,
    )


# ---------------------------------------------------------------------------
# NIIT
# ---------------------------------------------------------------------------

def compute_niit(
    magi: float, nii: float, filing_status: str
) -> float:
    """3.8 % Net Investment Income Tax on lesser of NII or MAGI above threshold."""
    threshold = NIIT_THRESHOLD[filing_status]
    excess = magi - threshold
    if excess <= 0:
        return 0.0
    return NIIT_RATE * min(nii, excess)


# ---------------------------------------------------------------------------
# Annual accumulator
# ---------------------------------------------------------------------------

@dataclass
class TaxYearAccumulator:
    """Collects all taxable events for one simulation year."""
    tax_deferred_withdrawals: float = 0.0
    interest_income: float = 0.0       # taxable-wrapper only
    other_income: float = 0.0
    st_capital_gains: float = 0.0
    st_capital_losses: float = 0.0
    lt_capital_gains: float = 0.0
    lt_capital_losses: float = 0.0
    loss_carryforward: float = 0.0
    roth_withdrawals: float = 0.0
    rmd_required: float = 0.0
    rmd_taken: float = 0.0


# ---------------------------------------------------------------------------
# Full annual tax pipeline
# ---------------------------------------------------------------------------

def compute_annual_tax(
    accum: TaxYearAccumulator,
    filing_status: str,
    tcja_sunset: bool,
    sim_year: int,
    inflation_rate: float,
    state_rate: float,
) -> dict:
    """Compute federal + state tax for one simulation year.

    Returns a dict with all intermediate and final values.
    *state_rate* is a percentage (e.g. 13.30 for California).
    Brackets and standard deduction are inflation-indexed from the 2025 base.
    """

    # --- 0. Pick regime and inflate from 2025 base ---
    years_from_base = max(sim_year - 2025, 0)

    if tcja_sunset:
        ord_brackets_base = FEDERAL_BRACKETS_SUNSET[filing_status]
        std_ded_base = STANDARD_DEDUCTION_SUNSET[filing_status]
    else:
        ord_brackets_base = FEDERAL_BRACKETS_TCJA[filing_status]
        std_ded_base = STANDARD_DEDUCTION_TCJA[filing_status]

    ord_brackets = _inflate_brackets(ord_brackets_base, years_from_base, inflation_rate)
    ltcg_brackets = _inflate_brackets(LTCG_BRACKETS[filing_status], years_from_base, inflation_rate)
    std_ded = std_ded_base * (1 + inflation_rate) ** years_from_base
    niit_threshold = NIIT_THRESHOLD[filing_status]  # NOT inflation-indexed per IRS

    # --- 1. Capital loss netting ---
    cap = net_capital_gains(
        st_gains=accum.st_capital_gains,
        st_losses=accum.st_capital_losses,
        lt_gains=accum.lt_capital_gains,
        lt_losses=accum.lt_capital_losses,
        carryforward=accum.loss_carryforward,
    )

    # --- 2. AGI ---
    agi = (
        accum.tax_deferred_withdrawals
        + accum.interest_income
        + accum.other_income
        + max(cap.net_st, 0)
        + max(cap.net_lt, 0)
        - cap.loss_deduction
    )

    # --- 3. MAGI (same as AGI for v1) ---
    magi = agi

    # --- 4. Standard deduction split ---
    ordinary_gross = agi - max(cap.net_lt, 0)
    ordinary_taxable = max(ordinary_gross - std_ded, 0)
    remaining_ded = max(std_ded - ordinary_gross, 0)
    taxable_ltcg = max(cap.net_lt - remaining_ded, 0)

    # --- 5. Federal ordinary tax ---
    federal_ordinary = apply_progressive_brackets(ordinary_taxable, ord_brackets)

    # --- 6. Federal LTCG tax (stacking) ---
    federal_ltcg = compute_ltcg_tax(
        taxable_ltcg, stacking_base=ordinary_taxable, filing_status=filing_status
    )
    # Use inflated LTCG brackets for stacking
    if years_from_base > 0:
        total_ltcg = apply_progressive_brackets(
            ordinary_taxable + taxable_ltcg, ltcg_brackets
        )
        base_ltcg = apply_progressive_brackets(ordinary_taxable, ltcg_brackets)
        federal_ltcg = total_ltcg - base_ltcg

    # --- 7. NIIT ---
    nii = max(cap.net_st, 0) + max(cap.net_lt, 0) + accum.interest_income
    niit = NIIT_RATE * min(nii, max(magi - niit_threshold, 0)) if magi > niit_threshold else 0.0

    # --- 8. State tax (flat rate on AGI, v1 simplification) ---
    state_tax = (state_rate / 100.0) * max(agi, 0)

    # --- 9. Totals ---
    total = federal_ordinary + federal_ltcg + niit + state_tax
    effective_rate = total / agi if agi > 0 else 0.0

    return {
        "year": sim_year,
        "ordinary_income": ordinary_gross,
        "st_gains": cap.net_st,
        "lt_gains": cap.net_lt,
        "agi": agi,
        "standard_deduction": std_ded,
        "federal_ordinary": federal_ordinary,
        "federal_ltcg": federal_ltcg,
        "niit": niit,
        "state": state_tax,
        "total": total,
        "effective_rate": effective_rate,
        "loss_carryforward": cap.new_carryforward,
        "net_cap_loss_deduction": cap.loss_deduction,
    }
