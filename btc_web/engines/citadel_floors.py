"""Citadel Planner — floor enforcement and account distribution."""
from __future__ import annotations

from .citadel_types import CitadelState, SimConfig, FREQ_PPY
from .citadel_waterfall import _spending_waterfall
from .citadel_transactions import _sell_investments_tracked

__all__ = ["_enforce_floors", "_distribute_to_accounts", "_source_from_accounts"]


def _enforce_floors(state: CitadelState, config: SimConfig,
                    model: "PriceModel | None" = None) -> None:
    """Replenish accounts below their floor minimums.

    For cash floor: uses _spending_waterfall to draw from the cheapest
    source with full tax accounting, lot tracking, and bracket awareness.
    For reserve floors: redistributes among taxable dollar accounts only
    (never sells BTC or touches TD/TF for reserve replenishment).
    """
    ppy = FREQ_PPY.get(config.freq, 12)
    years_elapsed = state.period / ppy
    cash_floor_eff = config.cash_floor * (1 + config.cash_floor_growth / 100) ** years_elapsed
    res_floor_growth = (1 + config.reserve_floor_growth / 100) ** years_elapsed

    # --- Cash floor: delegate to _spending_waterfall ---
    if cash_floor_eff > 0:
        deficit = cash_floor_eff - state.cash
        if deficit > 0:
            # Temporarily zero cash so the waterfall doesn't draw from it
            saved_cash = state.cash
            state.cash = 0.0
            shortfall = _spending_waterfall(state, config, deficit, model=model)
            drawn = deficit - shortfall
            state.cash += saved_cash + drawn

    # --- Reserve floors: redistribute among taxable dollar accounts only ---
    for i, floor in enumerate(config.reserve_floors):
        eff = floor * res_floor_growth
        if eff <= 0:
            continue
        deficit = eff - state.reserves[i]
        if deficit <= 0:
            continue

        sources = []
        for j in reversed(range(len(state.investments))):
            sources.append(("inv", j))
        for j in reversed(range(len(state.reserves))):
            if j != i:
                sources.append(("res", j))
        sources.append(("cash", 0))

        drawn_total = 0.0
        for src_type, src_idx in sources:
            if deficit <= 0:
                break
            if src_type == "inv":
                draw_want = min(state.investments[src_idx], deficit)
                draw, _gain = _sell_investments_tracked(state, config, src_idx, draw_want)
            elif src_type == "res":
                draw = min(state.reserves[src_idx], deficit)
                state.reserves[src_idx] -= draw
            elif src_type == "cash":
                draw = min(state.cash, deficit)
                state.cash -= draw
            else:
                draw = 0
            deficit -= draw
            drawn_total += draw

        state.reserves[i] += drawn_total


def _distribute_to_accounts(state: CitadelState, amount: float, split: dict) -> None:
    """Distribute `amount` to accounts according to `split` fractions."""
    state.cash += amount * split.get("cash", 0)
    state.reserves[0] += amount * split.get("res_short", 0)
    state.reserves[1] += amount * split.get("res_med", 0)
    state.reserves[2] += amount * split.get("res_long", 0)
    state.investments[0] += amount * split.get("inv_eq", 0)
    state.investments[1] += amount * split.get("inv_bd", 0)

def _source_from_accounts(state: CitadelState, amount: float, split: dict,
                          config: "SimConfig | None" = None) -> float:
    """Draw `amount` from accounts according to `split` fractions.
    Returns actual amount sourced (may be less if total insufficient).
    When one account can't cover its share, shortfall is redistributed
    proportionally to remaining accounts with nonzero allocation.
    If `config` is provided, respects floor rules — never draws an
    account below its floor minimum."""
    def _get_floor(acct):
        if config is None:
            return 0.0
        if acct == "cash":
            return config.cash_floor
        if acct.startswith("res_"):
            idx = int(acct[-1])
            return config.reserve_floors[idx] if idx < len(config.reserve_floors) else 0.0
        return 0.0  # investments have no floors

    def _get_balance(acct):
        if acct == "cash":
            return state.cash
        if acct.startswith("res_"):
            return state.reserves[int(acct[-1])]
        if acct.startswith("inv_"):
            return state.investments[int(acct[-1])]
        return 0.0

    def _debit(acct, amt):
        if acct == "cash":
            state.cash -= amt
        elif acct.startswith("res_"):
            state.reserves[int(acct[-1])] -= amt
        elif acct.startswith("inv_"):
            idx = int(acct[-1])
            if config is not None:
                _sell_investments_tracked(state, config, idx, amt)
            else:
                state.investments[idx] -= amt

    accounts = [
        ("cash", split.get("cash", 0)),
        ("res_0", split.get("res_short", 0)),
        ("res_1", split.get("res_med", 0)),
        ("res_2", split.get("res_long", 0)),
        ("inv_0", split.get("inv_eq", 0)),
        ("inv_1", split.get("inv_bd", 0)),
    ]

    remaining = amount
    total_sourced = 0.0
    active = [(a, f) for a, f in accounts if f > 0]

    # Iteratively source, redistributing shortfalls
    while remaining > 0.01 and active:
        frac_sum = sum(f for _, f in active)
        if frac_sum <= 0:
            break
        next_active = []
        shortfall = 0.0
        for acct, frac in active:
            want = remaining * (frac / frac_sum)
            bal = _get_balance(acct)
            floor = _get_floor(acct)
            avail = max(bal - floor, 0.0)  # respect floor
            got = min(avail, want)
            _debit(acct, got)
            total_sourced += got
            if got < want - 0.01:
                shortfall += want - got
            else:
                next_active.append((acct, frac))
        remaining = shortfall
        active = next_active

    return total_sourced
