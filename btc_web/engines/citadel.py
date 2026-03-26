"""Citadel Planner simulation engine — pure Python + NumPy, zero Dash deps."""
from __future__ import annotations

from dataclasses import dataclass, field

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
