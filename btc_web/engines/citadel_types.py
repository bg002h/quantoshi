"""Citadel Planner — data types and configuration."""
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

__all__ = [
    "FREQ_PPY",
    "_SATOSHI",
    "_WithdrawalSource",
    "SimConfig",
    "CitadelState",
    "PriceModel",
    "SimResult",
]


@dataclass
class _WithdrawalSource:
    """Represents one drawable account for the cost-ranked waterfall."""
    key: str              # e.g., "cash", "reserve_0", "invest_1", "btc", "td_cash", "tf_btc"
    wrapper: str          # "taxable", "td", or "tf"
    asset_type: str       # "cash", "reserve", "invest", "btc"
    index: int            # bin index (0-2 for reserves, 0-1 for investments, 0 for cash/btc)
    available: float      # current dollar balance available to draw
    growth_rate: float    # annual growth rate for opportunity cost
    horizon: int          # opportunity cost horizon in years
    gain_fraction: float  # for investments/BTC: 1 - (basis/value). 0 for cash/reserves
    is_roth: bool         # True for TF sources — forced last
    is_bracket_sensitive: bool  # True if draw affects tax bracket position
    bracket_type: str     # "ordinary", "ltcg", or "none"
    cost: float = 0.0     # computed by _score_sources


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

    # Starting regime bins for Markov model (0=bearish, 2=neutral, 4=bullish)
    # Used by _initial_state() to seed CitadelState regime fields.
    # Macro regime presets (Bear/Neutral/Bull) set all five to the same bin.
    initial_equity_regime: int = 2
    initial_bond_regime: int = 2
    initial_res_short_regime: int = 2
    initial_res_med_regime: int = 2
    initial_res_long_regime: int = 2

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
    # TD wrapper regime states (independent from taxable regimes)
    td_equity_regime: int = 2
    td_bond_regime: int = 2
    td_res_short_regime: int = 2
    td_res_med_regime: int = 2
    td_res_long_regime: int = 2
    # TF wrapper regime states (independent from taxable regimes)
    tf_equity_regime: int = 2
    tf_bond_regime: int = 2
    tf_res_short_regime: int = 2
    tf_res_med_regime: int = 2
    tf_res_long_regime: int = 2
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
    # §1212(b) character-preserved carryforwards. `loss_carryforward` stays
    # as the sum (read-only for legacy callers); writes should target the
    # two fields directly.
    st_carryforward: float = 0.0
    lt_carryforward: float = 0.0
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
