"""Citadel Planner simulation engine — pure Python + NumPy, zero Dash deps."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# Intentionally limited to Monthly/Quarterly/Annually — Daily/Weekly excluded
# for v1 performance (Daily = 14,600 steps over 40yr). See spec section
# "Performance Notes". Diverges from _app_ctx.FREQ_PPY which includes all 5.
from _app_ctx import FREQ_PPY as _ALL_FREQ_PPY
FREQ_PPY = {k: v for k, v in _ALL_FREQ_PPY.items() if k in ("Monthly", "Quarterly", "Annually")}
_SATOSHI = 1e-8  # smallest BTC unit — anything below this is zero


@dataclass
class SimConfig:
    """Simulation configuration for the Citadel Planner engine.

    Field defaults here serve as the engine's 'unset sentinel' — what you get
    if a field isn't explicitly passed. UI defaults live in tab_defaults.CITADEL.
    Where they differ intentionally, a comment explains why.
    """
    # BTC
    price_model: str = "bub"
    start_stack: float = 1.0
    selected_qs: list[float] = field(default_factory=lambda: [0.01, 0.10, 0.25])  # INTENTIONAL: engine default is [0.1%, 10%, 25%]; UI default is (25%,)

    # Cash
    cash_initial: float = 50_000.0
    cash_rate: float = 4.0  # annual %

    # Reserves: list of {initial, rate, volatility}
    reserve_bins: list[dict] = field(default_factory=lambda: [
        {"label": "Short (T-Bills)", "initial": 50_000.0, "rate": 5.0, "volatility": 2.0},
        {"label": "Medium (T-Notes)", "initial": 100_000.0, "rate": 4.5, "volatility": 8.0},
        {"label": "Long (T-Bonds)", "initial": 50_000.0, "rate": 4.0, "volatility": 15.0},
    ])

    # Investments: list of {initial, return_rate, volatility}
    invest_bins: list[dict] = field(default_factory=lambda: [
        {"label": "Equities", "initial": 200_000.0, "return_rate": 10.0, "volatility": 16.0},
        {"label": "Bonds", "initial": 100_000.0, "return_rate": 5.0, "volatility": 7.0},
    ])
    # Cost basis for taxable investments (what user originally paid).
    # None = use initial value (no prior gains). List of floats, one per invest_bin.
    invest_cost_basis_initial: list[float] | None = None

    # Spending
    monthly_spend: float = 5_000.0
    inflation: float = 4.0       # annual %
    spend_growth: float = 0.0    # annual % above inflation

    # Rebalancing — defaults are conservative (triggers rarely fire)
    # User must deliberately configure aggressive thresholds
    high_q_trigger: float = 0.95   # only fire at extreme overvaluation
    high_q_action: dict = field(default_factory=lambda: {
        "mode": "gradual", "rate": 2.0, "duration": 6,
        "split": {"cash": 0.20, "res_short": 0.20, "res_med": 0.20,
                  "res_long": 0.10, "inv_eq": 0.20, "inv_bd": 0.10},
    })
    low_q_trigger: float = 0.05    # only fire at extreme undervaluation
    low_q_action: dict = field(default_factory=lambda: {
        "mode": "lump", "rate": 10.0, "duration": 1,
        "split": {"cash": 0.10, "res_short": 0.10, "res_med": 0.10,
                  "res_long": 0.10, "inv_eq": 0.40, "inv_bd": 0.20},
    })
    lump_cooldown: int = 12  # periods

    # Floor rules
    cash_floor: float = 0.0  # INTENTIONAL: engine sentinel is 0.0; UI default is 50000 via CITADEL["cash_floor"]
    cash_floor_growth: float = 0.0     # annual % increase in cash floor
    reserve_floors: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    reserve_floor_growth: float = 0.0  # annual % increase in all reserve floors

    # Saylor Citadel Fortifier
    scf_enabled: bool = False
    scf_amount: float = 0.0
    scf_type: str = "term"       # "term" or "perpetual"
    scf_rate: float = 8.0        # annual %
    scf_term: int = 60           # months (term loan only)
    scf_repay_trigger: float = 1.0  # N multiplier (perpetual only)

    # Simulation
    start_yr: int = 2031
    end_yr: int = 2075
    freq: str = "Monthly"
    n_sims: int = 1  # INTENTIONAL: engine default is 1 (deterministic); UI default not exposed (always 1)
    # Tax configuration
    tax_enabled: bool = False
    filing_status: str = "single"           # "single" or "mfj"
    state_code: str = "TX"                  # default: no income tax
    state_rate_override: float | None = None
    tcja_sunset: bool = False
    birth_year: int | None = None           # for RMD (None = disabled)
    cost_basis_method: str = "fifo"
    other_income: float = 0.0               # external income (wages, SS, etc.)
    other_income_growth: float = 0.0        # annual growth rate

    # Tax-Deferred wrapper
    td_btc_stack: float = 0.0
    td_cash_initial: float = 0.0
    td_reserve_bins: list[dict] = field(default_factory=lambda: [
        {"label": "Short", "initial": 0.0},
        {"label": "Medium", "initial": 0.0},
        {"label": "Long", "initial": 0.0},
    ])
    td_invest_bins: list[dict] = field(default_factory=lambda: [
        {"label": "Equities", "initial": 0.0},
        {"label": "Bonds", "initial": 0.0},
    ])

    # Tax-Free (Roth) wrapper
    tf_btc_stack: float = 0.0
    tf_cash_initial: float = 0.0
    tf_reserve_bins: list[dict] = field(default_factory=lambda: [
        {"label": "Short", "initial": 0.0},
        {"label": "Medium", "initial": 0.0},
        {"label": "Long", "initial": 0.0},
    ])
    tf_invest_bins: list[dict] = field(default_factory=lambda: [
        {"label": "Equities", "initial": 0.0},
        {"label": "Bonds", "initial": 0.0},
    ])

    # Asset return model: "lognormal" (user-input rates) or "markov" (historical regimes)
    asset_return_model: str = "lognormal"
    # Transition matrices for Markov mode (loaded from data/asset_matrices.py)
    # Keys: "equity", "bond", "tres_short", "tres_med", "tres_long"
    # Each value: dict with "trans", "bin_edges", "bin_means", "bin_vols"
    asset_matrices: dict | None = None

    @classmethod
    def default(cls) -> SimConfig:
        return cls()


@dataclass
class CitadelState:
    """Mutable state passed forward each simulation step."""
    t: float = 0.0
    period: int = 0
    btc_stack: float = 0.0
    btc_price: float = 0.0
    btc_cost_basis: float = 0.0
    cash: float = 0.0
    reserves: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    investments: list[float] = field(default_factory=lambda: [0.0, 0.0])
    # Rebalancing
    rebal_cooldown: int = 0
    grad_active: bool = False
    grad_remaining: int = 0
    grad_rate: float = 0.0
    grad_direction: str = ""
    grad_split: dict = field(default_factory=dict)
    # Saylor Fortifier
    scf_outstanding: float = 0.0
    scf_active: bool = False
    # Asset regime states (for Markov return model)
    # Indices into the transition matrix bins for each asset class
    equity_regime: int = 2     # middle bin = neutral
    bond_regime: int = 2
    res_short_regime: int = 2
    res_med_regime: int = 2
    res_long_regime: int = 2
    # Tracking
    period_spend: float = 0.0
    spending_shortfall: float = 0.0
    rebal_event: dict | None = None

    # Investment cost basis (taxable wrapper) — tracks original purchase value
    # so capital gains = current_value - cost_basis on sale
    invest_cost_basis: list[float] = field(default_factory=lambda: [0.0, 0.0])

    # Tax state (only used when config.tax_enabled)
    tax_lots: list = field(default_factory=list)         # list[TaxLot]
    tax_year_accum: object | None = None                 # TaxYearAccumulator
    loss_carryforward: float = 0.0
    total_taxes_paid: float = 0.0
    annual_tax_history: list = field(default_factory=list)  # list[dict]
    quarterly_tax_paid_ytd: float = 0.0       # cumulative estimated payments this tax year
    sim_date: str = ""                        # ISO date string, set each period in step()

    # Tax-Deferred wrapper balances
    td_btc_stack: float = 0.0
    td_cash: float = 0.0
    td_reserves: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    td_investments: list[float] = field(default_factory=lambda: [0.0, 0.0])

    # Tax-Free (Roth) wrapper balances
    tf_btc_stack: float = 0.0
    tf_cash: float = 0.0
    tf_reserves: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    tf_investments: list[float] = field(default_factory=lambda: [0.0, 0.0])


@runtime_checkable
class PriceModel(Protocol):
    """Protocol for BTC price models. The callback layer wraps
    _app_ctx.PRICE_MODELS entries to satisfy this interface."""
    fits: dict
    genesis: float

    def price_at(self, q: float, t: float) -> float: ...
    def quantile_at(self, price: float, t: float) -> float: ...


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


def _apply_spending_waterfall(state: CitadelState, amount: float) -> float:
    """Draw `amount` from accounts in waterfall order. Returns unmet shortfall.
    Mutates state in place. Order: Cash -> Reserves (short->med->long) ->
    Investments (bonds->equities) -> BTC (emergency liquidation)."""
    remaining = amount
    if remaining <= 0:
        return 0.0

    # 1. Cash
    draw = min(state.cash, remaining)
    state.cash -= draw
    remaining -= draw
    if remaining <= 0:
        return 0.0

    # 2. Reserves: short -> medium -> long
    for i in range(len(state.reserves)):
        draw = min(state.reserves[i], remaining)
        state.reserves[i] -= draw
        remaining -= draw
        if remaining <= 0:
            return 0.0

    # 3. Investments: bonds (last index) -> equities (first index)
    for i in reversed(range(len(state.investments))):
        draw = min(state.investments[i], remaining)
        state.investments[i] -= draw
        remaining -= draw
        if remaining <= 0:
            return 0.0

    # 4. BTC (emergency liquidation)
    if state.btc_stack > 0 and state.btc_price > 0:
        btc_value = state.btc_stack * state.btc_price
        if btc_value >= remaining:
            state.btc_stack -= remaining / state.btc_price
            remaining = 0.0
        else:
            state.btc_stack = 0.0
            remaining -= btc_value

    return max(remaining, 0.0)


def _enforce_floors(state: CitadelState, config: SimConfig) -> None:
    """Replenish accounts below their floor minimums.
    Draw order for replenishment sources (reverse priority):
    1. Investment Bonds (index 1)
    2. Investment Equities (index 0)
    3. Reserve Long (index 2)
    4. Reserve Medium (index 1)
    5. Reserve Short (index 0)
    6. Cash (only for reserve replenishment, not self-replenishment)
    7. BTC (last resort, for CASH FLOOR ONLY — ensures the retirement
       spending account stays funded even when all dollar assets are depleted)
    Reserve floors do NOT sell BTC — they only redistribute among dollar accounts."""
    # Compute time-adjusted floors (grow annually by configured %)
    ppy = FREQ_PPY.get(config.freq, 12)
    years_elapsed = state.period / ppy
    cash_floor_eff = config.cash_floor * (1 + config.cash_floor_growth / 100) ** years_elapsed
    res_floor_growth = (1 + config.reserve_floor_growth / 100) ** years_elapsed

    accounts_to_check = []
    if cash_floor_eff > 0:
        accounts_to_check.append(("cash", cash_floor_eff))
    for i, floor in enumerate(config.reserve_floors):
        eff = floor * res_floor_growth
        if eff > 0:
            accounts_to_check.append((f"reserve_{i}", eff))

    for acct_key, floor in accounts_to_check:
        if acct_key == "cash":
            current = state.cash
        else:
            idx = int(acct_key.split("_")[1])
            current = state.reserves[idx]

        deficit = floor - current
        if deficit <= 0:
            continue

        sources = []
        for i in reversed(range(len(state.investments))):
            sources.append(("inv", i))
        for i in reversed(range(len(state.reserves))):
            if acct_key != f"reserve_{i}":
                sources.append(("res", i))
        if acct_key != "cash":
            sources.append(("cash", 0))

        for src_type, src_idx in sources:
            if deficit <= 0:
                break
            if src_type == "inv":
                draw = min(state.investments[src_idx], deficit)
                state.investments[src_idx] -= draw
            elif src_type == "res":
                draw = min(state.reserves[src_idx], deficit)
                state.reserves[src_idx] -= draw
            elif src_type == "cash":
                draw = min(state.cash, deficit)
                state.cash -= draw
            else:
                draw = 0
            deficit -= draw

        # BTC last resort — only for cash floor
        if deficit > 0 and acct_key == "cash":
            if state.btc_stack > 0 and state.btc_price > 0:
                btc_needed = deficit / state.btc_price
                btc_sold = min(state.btc_stack, btc_needed)
                draw = btc_sold * state.btc_price
                state.btc_stack -= btc_sold
                deficit -= draw

        replenished = (floor - current) - deficit
        if acct_key == "cash":
            state.cash += replenished
        else:
            idx = int(acct_key.split("_")[1])
            state.reserves[idx] += replenished


_SPLIT_KEYS = ["cash", "res_short", "res_med", "res_long", "inv_eq", "inv_bd"]

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
            state.investments[int(acct[-1])] -= amt

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

def _execute_sell_btc(state: CitadelState, rate_pct: float, split: dict) -> dict:
    """Sell rate_pct% of BTC stack, distribute proceeds via split."""
    btc_to_sell = state.btc_stack * (rate_pct / 100.0)
    if btc_to_sell <= 0 or state.btc_price <= 0:
        return {}
    proceeds = btc_to_sell * state.btc_price
    state.btc_stack -= btc_to_sell
    _distribute_to_accounts(state, proceeds, split)
    return {"action": "sell_btc", "btc_sold": btc_to_sell, "proceeds": proceeds}

def _execute_buy_btc(state: CitadelState, rate_pct: float, split: dict,
                     config: "SimConfig | None" = None) -> dict:
    """Source funds from accounts via split, buy BTC.
    Respects floor rules if config provided — won't draw accounts below floors."""
    total_dollar = state.cash + sum(state.reserves) + sum(state.investments)
    target = total_dollar * (rate_pct / 100.0)
    if target <= 0 or state.btc_price <= 0:
        return {}
    sourced = _source_from_accounts(state, target, split, config=config)
    if sourced <= 0:
        return {}
    btc_bought = sourced / state.btc_price
    state.btc_stack += btc_bought
    return {"action": "buy_btc", "btc_bought": btc_bought, "cost": sourced}

def _evaluate_rebalancing(state: CitadelState, config: SimConfig,
                          btc_quantile: float) -> None:
    """Evaluate and execute rebalancing triggers. Mutates state."""
    state.rebal_event = None
    if state.rebal_cooldown > 0:
        state.rebal_cooldown -= 1
    # If gradual is active, continue it (ignoring new triggers)
    if state.grad_active:
        if state.grad_remaining > 0:
            if state.grad_direction == "sell_btc":
                evt = _execute_sell_btc(state, state.grad_rate, state.grad_split)
            else:
                evt = _execute_buy_btc(state, state.grad_rate, state.grad_split, config=config)
            state.grad_remaining -= 1
            if evt:
                evt["type"] = "gradual_continue"
                state.rebal_event = evt
        if state.grad_remaining <= 0:
            state.grad_active = False
        return
    # Check high-Q trigger
    if btc_quantile >= config.high_q_trigger:
        action = config.high_q_action
        split = action.get("split", {})
        if action["mode"] == "lump" and state.rebal_cooldown <= 0:
            evt = _execute_sell_btc(state, action["rate"], split)
            if evt:
                evt["type"] = "lump_sell"
                state.rebal_event = evt
                state.rebal_cooldown = config.lump_cooldown
        elif action["mode"] == "gradual":
            state.grad_active = True
            state.grad_remaining = action.get("duration", 6)
            state.grad_rate = action["rate"]
            state.grad_direction = "sell_btc"
            state.grad_split = split
            evt = _execute_sell_btc(state, state.grad_rate, split)
            state.grad_remaining -= 1
            if evt:
                evt["type"] = "gradual_start"
                state.rebal_event = evt
        return
    # Check low-Q trigger
    if btc_quantile <= config.low_q_trigger:
        action = config.low_q_action
        split = action.get("split", {})
        if action["mode"] == "lump" and state.rebal_cooldown <= 0:
            evt = _execute_buy_btc(state, action["rate"], split, config=config)
            if evt:
                evt["type"] = "lump_buy"
                state.rebal_event = evt
                state.rebal_cooldown = config.lump_cooldown
        elif action["mode"] == "gradual":
            state.grad_active = True
            state.grad_remaining = action.get("duration", 6)
            state.grad_rate = action["rate"]
            state.grad_direction = "buy_btc"
            state.grad_split = split
            evt = _execute_buy_btc(state, state.grad_rate, split, config=config)
            state.grad_remaining -= 1
            if evt:
                evt["type"] = "gradual_start"
                state.rebal_event = evt


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
            btc_sold = min(state.btc_stack, btc_needed)
            repaid = btc_sold * state.btc_price
            state.btc_stack -= btc_sold
            state.scf_outstanding -= repaid
        if state.scf_outstanding <= 0.01:
            state.scf_outstanding = 0
            state.scf_active = False


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


def _tax_aware_waterfall(state: CitadelState, config: SimConfig,
                         amount: float, sim_date: str,
                         model: "PriceModel | None" = None) -> float:
    """Draw `amount` from accounts in growth-aware tax-efficient order.
    Returns unmet shortfall. Mutates state in place.

    Order:
    1. Taxable principal (cash, reserves) — no tax event
    2. Tax-Deferred bracket-filling (12% bracket, inflation-adjusted)
    3-4. Taxable investments vs BTC — DYNAMIC: if BTC forward growth > equity
         return, sell investments first (protect BTC). Otherwise sell BTC first.
    5. Tax-Deferred remaining
    6. Roth cash/reserves/investments
    7. Roth BTC (absolute last — tax-free compounding on highest-growth asset)
    """
    from .tax_lots import sell_lots

    remaining = amount
    if remaining <= 0:
        return 0.0

    # --- 1. Taxable principal: cash, then reserves (no tax event) ---
    draw = min(state.cash, remaining)
    state.cash -= draw
    remaining -= draw
    if remaining <= 0:
        return 0.0

    for i in range(len(state.reserves)):
        draw = min(state.reserves[i], remaining)
        state.reserves[i] -= draw
        remaining -= draw
        if remaining <= 0:
            return 0.0

    # --- 2. Tax-Deferred bracket-filling (stay in low ordinary brackets) ---
    # Draw enough from TD to fill up to the 12% bracket top (inflation-adjusted).
    # Pro-rate to the period frequency so monthly sims don't over-fill.
    from .tax_data import FEDERAL_BRACKETS_TCJA
    _bracket_12_top = FEDERAL_BRACKETS_TCJA[config.filing_status][1][0]  # 2nd bracket upper
    _years_elapsed = state.period / FREQ_PPY[config.freq]
    _bracket_inflated = _bracket_12_top * (1 + config.inflation / 100) ** _years_elapsed
    # Subtract ordinary income already accumulated this year
    _already_ordinary = 0.0
    if state.tax_year_accum is not None:
        _already_ordinary = (state.tax_year_accum.tax_deferred_withdrawals
                             + state.tax_year_accum.interest_income
                             + state.tax_year_accum.treasury_interest
                             + state.tax_year_accum.other_income)
    _annual_room = max(_bracket_inflated - _already_ordinary, 0.0)
    _ppy = FREQ_PPY[config.freq]
    td_bracket_fill = min(remaining, _annual_room / _ppy)  # pro-rate to period
    td_avail = (state.td_cash + sum(state.td_reserves) + sum(state.td_investments)
                + state.td_btc_stack * max(state.btc_price, 0))
    td_draw = min(td_bracket_fill, td_avail)
    if td_draw > 0:
        td_remaining = td_draw
        d = min(state.td_cash, td_remaining)
        state.td_cash -= d
        td_remaining -= d
        for i in range(len(state.td_reserves)):
            if td_remaining <= 0:
                break
            d = min(state.td_reserves[i], td_remaining)
            state.td_reserves[i] -= d
            td_remaining -= d
        for i in reversed(range(len(state.td_investments))):
            if td_remaining <= 0:
                break
            d = min(state.td_investments[i], td_remaining)
            state.td_investments[i] -= d
            td_remaining -= d
        if td_remaining > 0 and state.td_btc_stack > 0 and state.btc_price > 0:
            btc_val = state.td_btc_stack * state.btc_price
            d = min(btc_val, td_remaining)
            state.td_btc_stack -= d / state.btc_price
            td_remaining -= d
        actual_td = td_draw - td_remaining
        state.cash += actual_td  # flows into taxable cash
        if state.tax_year_accum is not None:
            state.tax_year_accum.tax_deferred_withdrawals += actual_td
        remaining -= actual_td
        if remaining <= 0:
            return 0.0

    # --- 3-4. Taxable investments vs BTC: growth-aware ordering ---
    # Compare BTC forward growth to equity return rate. If BTC growth is high,
    # sell investments first (protect BTC). If low, sell BTC first.
    _equity_rate = config.invest_bins[0]["return_rate"] / 100 if config.invest_bins else 0.10
    _btc_fwd_growth = _equity_rate  # default: same as equities
    if model is not None and state.btc_price > 0:
        try:
            _t_now = state.t
            _q = config.selected_qs[len(config.selected_qs) // 2] if config.selected_qs else 0.25
            _price_now = float(model.price_at(_q, max(_t_now, 0.5)))
            _price_next = float(model.price_at(_q, max(_t_now + 1, 0.5)))
            if _price_now > 0:
                _btc_fwd_growth = (_price_next / _price_now) - 1
        except Exception:
            pass  # fall back to equity rate

    _sell_investments_first = (_btc_fwd_growth > _equity_rate)

    def _sell_taxable_investments():
        nonlocal remaining
        for i in reversed(range(len(state.investments))):
            draw = min(state.investments[i], remaining)
            if draw > 0:
                current = state.investments[i]
                if current > 0:
                    fraction = draw / current
                    basis_sold = state.invest_cost_basis[i] * fraction
                    gain = draw - basis_sold
                    state.invest_cost_basis[i] -= basis_sold
                else:
                    gain = draw
                if state.tax_year_accum is not None:
                    if gain >= 0:
                        state.tax_year_accum.lt_capital_gains += gain
                    else:
                        state.tax_year_accum.lt_capital_losses += abs(gain)
            state.investments[i] -= draw
            remaining -= draw
            if remaining <= 0:
                return True
        return False

    def _sell_taxable_btc():
        nonlocal remaining
        if remaining > 0 and state.btc_price > 0 and state.tax_lots:
            btc_to_sell = remaining / state.btc_price
            btc_to_sell = min(btc_to_sell, state.btc_stack)
            if btc_to_sell > _SATOSHI:
                result = sell_lots(
                    state.tax_lots, btc_to_sell, state.btc_price,
                    sim_date, method=config.cost_basis_method,
                )
                state.tax_lots = result.remaining_lots
                state.btc_stack -= result.btc_sold
                proceeds = result.btc_sold * state.btc_price
                remaining -= proceeds
                if state.tax_year_accum is not None:
                    for g in result.gains:
                        if g.is_long_term:
                            if g.gain >= 0:
                                state.tax_year_accum.lt_capital_gains += g.gain
                            else:
                                state.tax_year_accum.lt_capital_losses += abs(g.gain)
                        else:
                            if g.gain >= 0:
                                state.tax_year_accum.st_capital_gains += g.gain
                            else:
                                state.tax_year_accum.st_capital_losses += abs(g.gain)
        return remaining <= 0

    if _sell_investments_first:
        # BTC growth > equity growth: protect BTC, sell investments first
        if _sell_taxable_investments():
            return 0.0
        if _sell_taxable_btc():
            return 0.0
    else:
        # BTC growth <= equity growth: sell BTC first (lower opportunity cost)
        if _sell_taxable_btc():
            return 0.0
        if _sell_taxable_investments():
            return 0.0

    if remaining <= 0:
        return 0.0

    # --- 5. Tax-Deferred remaining ---
    td_avail2 = (state.td_cash + sum(state.td_reserves) + sum(state.td_investments)
                 + state.td_btc_stack * max(state.btc_price, 0))
    if td_avail2 > 0:
        td_remaining = min(remaining, td_avail2)
        d = min(state.td_cash, td_remaining)
        state.td_cash -= d
        td_remaining -= d
        for i in range(len(state.td_reserves)):
            if td_remaining <= 0:
                break
            d = min(state.td_reserves[i], td_remaining)
            state.td_reserves[i] -= d
            td_remaining -= d
        for i in reversed(range(len(state.td_investments))):
            if td_remaining <= 0:
                break
            d = min(state.td_investments[i], td_remaining)
            state.td_investments[i] -= d
            td_remaining -= d
        if td_remaining > 0 and state.td_btc_stack > 0 and state.btc_price > 0:
            btc_val = state.td_btc_stack * state.btc_price
            d = min(btc_val, td_remaining)
            state.td_btc_stack -= d / state.btc_price
            td_remaining -= d
        actual_td2 = min(remaining, td_avail2) - td_remaining
        if state.tax_year_accum is not None:
            state.tax_year_accum.tax_deferred_withdrawals += actual_td2
        remaining -= actual_td2

    if remaining <= 0:
        return 0.0

    # --- 6. (BTC short-term lots already handled in step 4 — sell_lots uses FIFO) ---

    # --- 7. Roth cash/reserves/investments (no tax impact) ---
    _roth_drawn = 0.0
    draw = min(state.tf_cash, remaining)
    state.tf_cash -= draw
    _roth_drawn += draw
    remaining -= draw
    if remaining <= 0:
        if state.tax_year_accum is not None:
            state.tax_year_accum.roth_withdrawals += _roth_drawn
        return 0.0

    for i in range(len(state.tf_reserves)):
        draw = min(state.tf_reserves[i], remaining)
        state.tf_reserves[i] -= draw
        _roth_drawn += draw
        remaining -= draw
        if remaining <= 0:
            if state.tax_year_accum is not None:
                state.tax_year_accum.roth_withdrawals += _roth_drawn
            return 0.0

    for i in reversed(range(len(state.tf_investments))):
        draw = min(state.tf_investments[i], remaining)
        state.tf_investments[i] -= draw
        _roth_drawn += draw
        remaining -= draw
        if remaining <= 0:
            if state.tax_year_accum is not None:
                state.tax_year_accum.roth_withdrawals += _roth_drawn
            return 0.0

    # --- 8. Roth BTC (absolute last, no tax) ---
    if state.tf_btc_stack > 0 and state.btc_price > 0:
        btc_val = state.tf_btc_stack * state.btc_price
        if btc_val >= remaining:
            _roth_drawn += remaining
            state.tf_btc_stack -= remaining / state.btc_price
            remaining = 0.0
        else:
            _roth_drawn += btc_val
            state.tf_btc_stack = 0.0
            remaining -= btc_val

    if state.tax_year_accum is not None:
        state.tax_year_accum.roth_withdrawals += _roth_drawn

    return max(remaining, 0.0)


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
            gain_frac = 1.0 - basis_frac  # fraction of each dollar that is gain
            effective_rate = ltcg_rate * gain_frac  # tax per dollar sold
            gross = tax_remaining / max(1.0 - effective_rate, 0.1)  # gross-up
            draw = min(current, gross)
            fraction = draw / current
            basis_removed = state.invest_cost_basis[i] * fraction
            state.invest_cost_basis[i] -= basis_removed
            state.investments[i] -= draw
            net_received = draw  # proceeds from sale
            tax_on_sale = (draw - basis_removed) * ltcg_rate
            tax_remaining -= (net_received - tax_on_sale)  # net after tax-on-sale

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


def step(state: CitadelState, config: SimConfig,
         btc_price_new: float, rng: np.random.Generator,
         model: "PriceModel | None" = None) -> CitadelState:
    """Advance simulation by one period. Returns new state (does not mutate input)."""
    from copy import deepcopy
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

    is_deterministic = (config.n_sims == 1)

    # 1. Update BTC price
    new.btc_price = btc_price_new

    # 2. Dollar-asset returns
    use_markov = (config.asset_return_model == "markov"
                  and config.asset_matrices is not None
                  and not is_deterministic)

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
                                      deterministic=is_deterministic, rng=rng)
                new.reserves[i] *= (1 + r)

        for i, (mkey, rattr) in enumerate(zip(_inv_keys, _inv_regime_attrs)):
            if mkey in am:
                ret, new_regime = _markov_return(am[mkey], getattr(new, rattr), rng)
                setattr(new, rattr, new_regime)
                new.investments[i] *= (1 + ret)
            else:
                ib = config.invest_bins[i]
                r = _lognormal_return(ib["return_rate"] / 100, ib["volatility"] / 100, ppy,
                                      deterministic=is_deterministic, rng=rng)
                new.investments[i] *= (1 + r)
    else:
        # Lognormal returns (user-input rates/volatility)
        for i, rb in enumerate(config.reserve_bins):
            r = _lognormal_return(rb["rate"] / 100, rb["volatility"] / 100, ppy,
                                  deterministic=is_deterministic, rng=rng)
            new.reserves[i] *= (1 + r)
        for i, ib in enumerate(config.invest_bins):
            r = _lognormal_return(ib["return_rate"] / 100, ib["volatility"] / 100, ppy,
                                  deterministic=is_deterministic, rng=rng)
            new.investments[i] *= (1 + r)

    # 2b. TD/TF wrapper growth (same rates as taxable wrapper)
    if config.tax_enabled:
        cash_growth = (1 + config.cash_rate / 100) ** (1.0 / ppy)
        new.td_cash *= cash_growth
        new.tf_cash *= cash_growth
        for i, rb in enumerate(config.reserve_bins):
            r = _lognormal_return(rb["rate"] / 100, rb["volatility"] / 100, ppy,
                                  deterministic=is_deterministic, rng=rng)
            if i < len(new.td_reserves):
                new.td_reserves[i] *= (1 + r)
            r2 = _lognormal_return(rb["rate"] / 100, rb["volatility"] / 100, ppy,
                                   deterministic=is_deterministic, rng=rng)
            if i < len(new.tf_reserves):
                new.tf_reserves[i] *= (1 + r2)
        for i, ib in enumerate(config.invest_bins):
            r = _lognormal_return(ib["return_rate"] / 100, ib["volatility"] / 100, ppy,
                                  deterministic=is_deterministic, rng=rng)
            if i < len(new.td_investments):
                new.td_investments[i] *= (1 + r)
            r2 = _lognormal_return(ib["return_rate"] / 100, ib["volatility"] / 100, ppy,
                                   deterministic=is_deterministic, rng=rng)
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

    if config.tax_enabled:
        # Tax-aware spending waterfall
        sim_year = config.start_yr + int(years_elapsed)
        sim_date = f"{sim_year}-{min(max(1, int((years_elapsed % 1) * 12) + 1), 12):02d}-15"
        new.spending_shortfall = _tax_aware_waterfall(new, config, period_spend, sim_date, model=model)
    else:
        new.spending_shortfall = _apply_spending_waterfall(new, period_spend)

    # 6. Enforce floor rules (AFTER spending, so floors replenish drawdowns)
    _enforce_floors(new, config)

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
                    _enforce_floors(new, config)

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
            _enforce_floors(new, config)

    # Clamp sub-satoshi BTC to zero (1 sat = 10^-8 BTC is the smallest unit)
    if 0 < new.btc_stack < _SATOSHI:
        new.btc_stack = 0.0

    return new


@dataclass
class SimResult:
    """Serializable simulation output."""
    time_axis: np.ndarray          # (n_periods,)
    btc_holdings: np.ndarray       # (n_sims, n_periods)
    btc_prices: np.ndarray         # (n_sims, n_periods)
    cash_balances: np.ndarray      # (n_sims, n_periods)
    reserve_balances: np.ndarray   # (n_sims, n_periods, n_reserve_bins)
    invest_balances: np.ndarray    # (n_sims, n_periods, n_invest_bins)
    total_usd: np.ndarray          # (n_sims, n_periods)
    cumulative_spend: np.ndarray   # (n_sims, n_periods)
    depletion_period: list[int | None]
    rebal_events: list[list[dict]]
    # Aggregated
    median: dict                   # {asset_class: ndarray}
    percentiles: dict              # {pct: {asset_class: ndarray}}
    # Tax arrays (only populated when tax_enabled)
    taxes_paid: np.ndarray | None = None       # (n_sims, n_periods) cumulative
    annual_taxes: list | None = None           # list[list[dict]] per sim per year
    td_total: np.ndarray | None = None         # (n_sims, n_periods)
    tf_total: np.ndarray | None = None         # (n_sims, n_periods)
    taxable_total: np.ndarray | None = None    # (n_sims, n_periods)

    def to_dict(self) -> dict:
        """Serialize for JSON transport. ndarrays -> lists."""
        d = {}
        for key in ["time_axis", "btc_holdings", "btc_prices", "cash_balances",
                     "reserve_balances", "invest_balances", "total_usd", "cumulative_spend"]:
            val = getattr(self, key)
            d[key] = val.tolist() if isinstance(val, np.ndarray) else val
        d["depletion_period"] = self.depletion_period
        d["rebal_events"] = self.rebal_events
        d["median"] = {k: v.tolist() for k, v in self.median.items()}
        d["percentiles"] = {
            str(p): {k: v.tolist() for k, v in assets.items()}
            for p, assets in self.percentiles.items()
        }
        # Tax fields (optional)
        for key in ["taxes_paid", "td_total", "tf_total", "taxable_total"]:
            val = getattr(self, key, None)
            d[key] = val.tolist() if isinstance(val, np.ndarray) else None
        d["annual_taxes"] = self.annual_taxes
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SimResult:
        """Deserialize from JSON."""
        arrays = {}
        for key in ["time_axis", "btc_holdings", "btc_prices", "cash_balances",
                     "reserve_balances", "invest_balances", "total_usd", "cumulative_spend"]:
            arrays[key] = np.array(d[key])
        median = {k: np.array(v) for k, v in d["median"].items()}
        percentiles = {
            int(p): {k: np.array(v) for k, v in assets.items()}
            for p, assets in d["percentiles"].items()
        }
        # Tax fields (optional)
        tax_kw = {}
        for key in ["taxes_paid", "td_total", "tf_total", "taxable_total"]:
            val = d.get(key)
            tax_kw[key] = np.array(val) if val is not None else None
        tax_kw["annual_taxes"] = d.get("annual_taxes")
        return cls(**arrays,
                   depletion_period=d["depletion_period"],
                   rebal_events=d["rebal_events"],
                   median=median, percentiles=percentiles,
                   **tax_kw)


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
    asset_keys = {
        "btc_usd": btc_h * btc_p,
        "cash": cash_b,
        "reserves_total": res_b.sum(axis=2),
        "investments_total": inv_b.sum(axis=2),
        "total": total,
    }
    median = {k: np.median(v, axis=0) for k, v in asset_keys.items()}
    percentiles = {}
    for pct in [5, 25, 75, 95]:
        percentiles[pct] = {k: np.percentile(v, pct, axis=0) for k, v in asset_keys.items()}

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
