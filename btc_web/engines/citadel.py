"""Citadel Planner simulation engine — pure Python + NumPy, zero Dash deps."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# Intentionally limited to Monthly/Quarterly/Annually — Daily/Weekly excluded
# for v1 performance (Daily = 14,600 steps over 40yr). See spec section
# "Performance Notes". Diverges from _app_ctx.FREQ_PPY which includes all 5.
FREQ_PPY = {"Monthly": 12, "Quarterly": 4, "Annually": 1}


@dataclass
class SimConfig:
    """All user inputs, frozen for a simulation run."""
    # BTC
    price_model: str = "bub"
    start_stack: float = 1.0
    selected_qs: list[float] = field(default_factory=lambda: [0.01, 0.10, 0.25])

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

    # Spending
    monthly_spend: float = 5_000.0
    inflation: float = 4.0       # annual %
    spend_growth: float = 0.0    # annual % above inflation

    # Rebalancing
    high_q_trigger: float = 0.80
    high_q_action: dict = field(default_factory=lambda: {
        "mode": "gradual", "rate": 2.0, "duration": 6,
        "split": {"cash": 0.20, "res_short": 0.20, "res_med": 0.20,
                  "res_long": 0.10, "inv_eq": 0.20, "inv_bd": 0.10},
    })
    low_q_trigger: float = 0.20
    low_q_action: dict = field(default_factory=lambda: {
        "mode": "lump", "rate": 10.0, "duration": 1,
        "split": {"cash": 0.10, "res_short": 0.10, "res_med": 0.10,
                  "res_long": 0.10, "inv_eq": 0.40, "inv_bd": 0.20},
    })
    lump_cooldown: int = 12  # periods

    # Floor rules
    cash_floor: float = 0.0
    reserve_floors: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

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
    n_sims: int = 1
    tax_rate: float = 0.0  # placeholder

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
    # Tracking
    period_spend: float = 0.0
    spending_shortfall: float = 0.0
    rebal_event: dict | None = None


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
    BTC is NEVER sold for floors."""
    accounts_to_check = []
    if config.cash_floor > 0:
        accounts_to_check.append(("cash", config.cash_floor))
    for i, floor in enumerate(config.reserve_floors):
        if floor > 0:
            accounts_to_check.append((f"reserve_{i}", floor))

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

def _source_from_accounts(state: CitadelState, amount: float, split: dict) -> float:
    """Draw `amount` from accounts according to `split` fractions.
    Returns actual amount sourced (may be less if total insufficient).
    When one account can't cover its share, shortfall is redistributed
    proportionally to remaining accounts with nonzero allocation."""
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
            avail = _get_balance(acct)
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

def _execute_buy_btc(state: CitadelState, rate_pct: float, split: dict) -> dict:
    """Source funds from accounts via split, buy BTC."""
    total_dollar = state.cash + sum(state.reserves) + sum(state.investments)
    target = total_dollar * (rate_pct / 100.0)
    if target <= 0 or state.btc_price <= 0:
        return {}
    sourced = _source_from_accounts(state, target, split)
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
                evt = _execute_buy_btc(state, state.grad_rate, state.grad_split)
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
            evt = _execute_buy_btc(state, action["rate"], split)
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
            evt = _execute_buy_btc(state, state.grad_rate, split)
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
    if annual_rate <= 0 or annual_vol <= 0:
        return (1 + annual_rate) ** (1.0 / ppy) - 1.0
    sigma_ln = math.sqrt(math.log(1 + (annual_vol / (1 + annual_rate)) ** 2))
    mu_ln = math.log(1 + annual_rate) - sigma_ln ** 2 / 2
    period_mu = mu_ln / ppy
    period_sigma = sigma_ln / math.sqrt(ppy)
    return math.exp(rng.normal(period_mu, period_sigma)) - 1.0


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
    state = CitadelState(
        t=t0, btc_stack=config.start_stack, btc_price=btc_price,
        btc_cost_basis=btc_price, cash=config.cash_initial,
        reserves=[rb["initial"] for rb in config.reserve_bins],
        investments=[ib["initial"] for ib in config.invest_bins],
    )
    if config.scf_enabled and config.scf_amount > 0 and btc_price > 0:
        btc_bought = config.scf_amount / btc_price
        state.btc_stack += btc_bought
        state.scf_outstanding = config.scf_amount
        state.scf_active = True
        total_btc = config.start_stack + btc_bought
        if total_btc > 0:
            state.btc_cost_basis = (config.start_stack * btc_price + config.scf_amount) / total_btc
    return state


def step(state: CitadelState, config: SimConfig,
         btc_price_new: float, rng: np.random.Generator,
         model: "PriceModel | None" = None) -> CitadelState:
    """Advance simulation by one period. Returns new state (does not mutate input)."""
    from copy import deepcopy
    new = deepcopy(state)
    new.period += 1
    ppy = FREQ_PPY[config.freq]
    dt = 1.0 / ppy
    new.t += dt
    is_deterministic = (config.n_sims == 1)

    # 1. Update BTC price
    new.btc_price = btc_price_new

    # 2. Dollar-asset returns
    new.cash *= (1 + config.cash_rate / 100) ** (1.0 / ppy)
    for i, rb in enumerate(config.reserve_bins):
        r = _lognormal_return(rb["rate"] / 100, rb["volatility"] / 100, ppy,
                              deterministic=is_deterministic, rng=rng)
        new.reserves[i] *= (1 + r)
    for i, ib in enumerate(config.invest_bins):
        r = _lognormal_return(ib["return_rate"] / 100, ib["volatility"] / 100, ppy,
                              deterministic=is_deterministic, rng=rng)
        new.investments[i] *= (1 + r)

    # 3. Compute BTC quantile from price via model
    if model is not None:
        btc_quantile = model.quantile_at(new.btc_price, new.t)
    else:
        btc_quantile = 0.5

    # 4. Evaluate rebalancing triggers
    _evaluate_rebalancing(new, config, btc_quantile)

    # 5. Spending
    years_elapsed = new.period / ppy
    combined_rate = (config.inflation + config.spend_growth) / 100
    period_spend = config.monthly_spend * (1 + combined_rate) ** years_elapsed
    if new.scf_active:
        period_spend += _scf_payment_amount(config, ppy)
    period_spend *= (12 / ppy)  # scale monthly base to period frequency
    new.period_spend = period_spend
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
        return cls(**arrays,
                   depletion_period=d["depletion_period"],
                   rebal_events=d["rebal_events"],
                   median=median, percentiles=percentiles)


def _compute_n_periods(config: SimConfig) -> int:
    ppy = FREQ_PPY[config.freq]
    return int((config.end_yr - config.start_yr) * ppy)


def _snapshot_state(state: CitadelState) -> dict:
    """Capture scalar values from state for history recording."""
    return {
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


def _aggregate_results(all_histories: list[list[dict]], config: SimConfig,
                       time_axis: np.ndarray) -> SimResult:
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
        depl.append(depl_period)
        rebal_events.append(sim_events)

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
        median=median, percentiles=percentiles,
    )


def simulate(config: SimConfig, model: PriceModel,
             rng_seed: int = 42) -> SimResult:
    """Run n_sims simulations, aggregate results.
    - For n_sims=1 (deterministic): uses median selected quantile for BTC price.
    - For n_sims>1 (MC): raises NotImplementedError in v1 (Markov integration TBD).
    """
    validate_config(config)
    ppy = FREQ_PPY[config.freq]
    n_periods = _compute_n_periods(config)
    rng = np.random.default_rng(rng_seed)

    # Build time axis
    from btc_core import yr_to_t
    t0 = yr_to_t(config.start_yr)
    dt = 1.0 / ppy
    time_axis = np.array([t0 + i * dt for i in range(n_periods)])

    all_histories = []
    for sim_id in range(config.n_sims):
        state = _initial_state(config, model=model)
        history = []
        for period_idx in range(n_periods):
            t = time_axis[period_idx]
            # Get BTC price for this period
            if config.n_sims == 1:
                q = config.selected_qs[len(config.selected_qs) // 2] if config.selected_qs else 0.5
                btc_price = _get_btc_price(t, config, model, rng,
                                           sim_mode="deterministic", q=q)
            else:
                raise NotImplementedError("MC mode requires Markov engine integration")
            new_state = step(state, config, btc_price, rng, model=model)
            history.append(_snapshot_state(new_state))
            state = new_state
        all_histories.append(history)

    return _aggregate_results(all_histories, config, time_axis)


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
