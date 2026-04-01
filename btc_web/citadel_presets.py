"""Citadel Planner — Quick Scenario preset definitions.

All preset values in one place. Edit values here, not in logic modules.
"""
from __future__ import annotations

from engines.citadel_types import SimConfig
from data.asset_matrices import load_asset_matrices

_CACHED_MATRICES: dict | None = None

def _get_asset_matrices() -> dict:
    """Load asset matrices once, cache for reuse."""
    global _CACHED_MATRICES
    if _CACHED_MATRICES is None:
        _CACHED_MATRICES = load_asset_matrices(n_bins=5)
    return _CACHED_MATRICES

__all__ = [
    "WEALTH_LEVELS", "MACRO_REGIMES", "RULE_SETS",
    "BTC_MODELS", "BTC_ENTRY_QS", "START_YEARS", "TAX_STATUSES",
    "SIMS_PER_SCENARIO", "END_YEAR", "FREQ",
    "build_config", "preset_control_values",
]

# ── Cache dimensions ─────────────────────────────────────────────────────────

BTC_MODELS = ["bub", "qr", "pl", "lppl", "ef"]
BTC_ENTRY_QS = [1, 10, 50]       # percentile (0-100 scale)
START_YEARS = [2028, 2035]
TAX_STATUSES = ["single", "mfj"]
SIMS_PER_SCENARIO = 800
END_YEAR = 2075
FREQ = "Monthly"

# ── Wealth levels ────────────────────────────────────────────────────────────

_DEFAULT_ALLOCATION = {
    "cash": 10, "res_short": 10, "res_med": 10, "res_long": 10,
    "inv_eq": 40, "inv_bd": 20,
}

WEALTH_LEVELS = {
    "starter": {
        "label": "Starter Citadel",
        "dollar_assets": 500_000, "btc": 0.5,
        "monthly_spend": 5_000, "spend_growth": 1.0, "inflation": 4.0,
        "allocation": dict(_DEFAULT_ALLOCATION),
    },
    "full": {
        "label": "Full Citadel",
        "dollar_assets": 2_500_000, "btc": 2.5,
        "monthly_spend": 25_000, "spend_growth": 2.0, "inflation": 4.0,
        "allocation": dict(_DEFAULT_ALLOCATION),
    },
    "bitcoin": {
        "label": "Bitcoin Citadel",
        "dollar_assets": 2_500_000, "btc": 12.5,
        "monthly_spend": 50_000, "spend_growth": 4.0, "inflation": 4.0,
        "allocation": dict(_DEFAULT_ALLOCATION),
    },
}

# ── Macro regimes ────────────────────────────────────────────────────────────

MACRO_REGIMES = {
    "bear":    {"label": "Bear",    "bin": 0},
    "neutral": {"label": "Neutral", "bin": 2},
    "bull":    {"label": "Bull",    "bin": 4},
}

# ── Rule sets ────────────────────────────────────────────────────────────────

RULE_SETS = {
    "no_rebal": {
        "label": "No Rebal",
        "cash_floor": 0.0,
        "high_q_trigger": 0.99,
        "low_q_trigger": 0.01,
    },
    "cautious": {
        "label": "Cautious",
        "cash_floor": 50_000.0,
        "high_q_trigger": 0.90,
        "low_q_trigger": 0.10,
    },
    "aggressive": {
        "label": "Aggressive",
        "cash_floor": 100_000.0,
        "high_q_trigger": 0.75,
        "low_q_trigger": 0.25,
    },
}


# ── Control value mapper ─────────────────────────────────────────────────────

def preset_control_values(wealth: str, regime: str, rules: str,
                          start_year: int) -> dict[str, object]:
    """Map preset selections to {component_id: value} for Citadel controls."""
    wl = WEALTH_LEVELS[wealth]
    rs = RULE_SETS[rules]
    alloc = wl["allocation"]
    da = wl["dollar_assets"]

    return {
        "cp-stack": wl["btc"],
        "cp-cash-init": da * alloc["cash"] / 100,
        "cp-res-short-init": da * alloc["res_short"] / 100,
        "cp-res-med-init": da * alloc["res_med"] / 100,
        "cp-res-long-init": da * alloc["res_long"] / 100,
        "cp-inv-eq-init": da * alloc["inv_eq"] / 100,
        "cp-inv-bd-init": da * alloc["inv_bd"] / 100,
        "cp-spend": wl["monthly_spend"],
        "cp-infl": wl["inflation"],
        "cp-spend-growth": wl["spend_growth"],
        "cp-cash-floor": rs["cash_floor"],
        "cp-high-q-thresh": int(rs["high_q_trigger"] * 100),
        "cp-low-q-thresh": int(rs["low_q_trigger"] * 100),
        "cp-yr-range": [start_year, END_YEAR],
        "cp-asset-model": "markov",
    }


# ── Config builder ───────────────────────────────────────────────────────────

def build_config(
    wealth: str,
    regime: str,
    rules: str,
    start_year: int,
    tax_status: str,
) -> SimConfig:
    """Build a SimConfig from preset selections."""
    wl = WEALTH_LEVELS[wealth]
    mr = MACRO_REGIMES[regime]
    rs = RULE_SETS[rules]
    alloc = wl["allocation"]
    da = wl["dollar_assets"]

    cfg = SimConfig(
        start_stack=wl["btc"],
        cash_initial=da * alloc["cash"] / 100,
        cash_rate=4.0,
        reserve_bins=[
            {"label": "Short (T-Bills)", "initial": da * alloc["res_short"] / 100,
             "rate": 5.0, "volatility": 2.0},
            {"label": "Medium (T-Notes)", "initial": da * alloc["res_med"] / 100,
             "rate": 4.5, "volatility": 8.0},
            {"label": "Long (T-Bonds)", "initial": da * alloc["res_long"] / 100,
             "rate": 4.0, "volatility": 15.0},
        ],
        invest_bins=[
            {"label": "Equities", "initial": da * alloc["inv_eq"] / 100,
             "return_rate": 10.0, "volatility": 16.0},
            {"label": "Bonds", "initial": da * alloc["inv_bd"] / 100,
             "return_rate": 5.0, "volatility": 7.0},
        ],
        monthly_spend=wl["monthly_spend"],
        spend_growth=wl["spend_growth"],
        inflation=wl["inflation"],
        start_yr=start_year,
        end_yr=END_YEAR,
        freq=FREQ,
        asset_return_model="markov",
        initial_equity_regime=mr["bin"],
        initial_bond_regime=mr["bin"],
        initial_res_short_regime=mr["bin"],
        initial_res_med_regime=mr["bin"],
        initial_res_long_regime=mr["bin"],
        cash_floor=rs["cash_floor"],
        high_q_trigger=rs["high_q_trigger"],
        low_q_trigger=rs["low_q_trigger"],
        tax_enabled=True,
        filing_status=tax_status,
        state_code="TX",
        asset_matrices=_get_asset_matrices(),
    )
    return cfg
