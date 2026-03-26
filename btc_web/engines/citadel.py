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
