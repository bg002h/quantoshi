"""Diagnostic tests for Citadel Planner — traces every feature through the full
engine pipeline and prints a state log so we can see exactly what happens."""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
for _p in (str(_ROOT), str(_ROOT / "btc_web"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math
import numpy as np
import pytest
from copy import deepcopy
from engines.citadel import (
    SimConfig, CitadelState, FREQ_PPY, PriceModel,
    validate_config, _enforce_floors,
    _evaluate_rebalancing, _lognormal_return, _scf_payment_amount,
    _scf_check_repay, _initial_state, step, simulate,
    SimResult, _compute_n_periods, _get_btc_price,
)


# ── Controllable mock model ──────────────────────────────────────────────────

class DiagPriceModel:
    """Mock where we control the quantile returned for any price."""
    def __init__(self, base_price=100000.0, quantile_schedule=None):
        self.fits = {q/100: None for q in range(1, 100)}
        self.genesis = 0.0
        self.base_price = base_price
        # quantile_schedule: list of (t_threshold, quantile_to_return)
        # When quantile_at is called, returns the quantile for the
        # matching time window. This lets us control when triggers fire.
        self._q_schedule = quantile_schedule or []
        self._fixed_q = 0.50

    def price_at(self, q: float, t: float) -> float:
        return self.base_price * (1 + t * 0.01)  # slow growth

    def quantile_at(self, price: float, t: float) -> float:
        for t_thresh, q_val in reversed(self._q_schedule):
            if t >= t_thresh:
                return q_val
        return self._fixed_q


def _log_state(label, s):
    """Print one-line state summary."""
    total = s.btc_stack * s.btc_price + s.cash + sum(s.reserves) + sum(s.investments)
    print(f"  [{label:>20}] BTC={s.btc_stack:.4f}@${s.btc_price:.0f} "
          f"Cash=${s.cash:.0f} Res=[${s.reserves[0]:.0f},${s.reserves[1]:.0f},${s.reserves[2]:.0f}] "
          f"Inv=[${s.investments[0]:.0f},${s.investments[1]:.0f}] "
          f"Total=${total:.0f} Spend=${s.period_spend:.0f} "
          f"Shortfall=${s.spending_shortfall:.0f} "
          f"Rebal={s.rebal_event is not None} "
          f"Cooldown={s.rebal_cooldown} GradActive={s.grad_active}")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 1: Basic spending waterfall — does cash decrease each period?
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagSpendingWorks:
    def test_cash_decreases_over_time(self):
        """With $50K cash and $5K/mo spending, cash should decrease."""
        print("\n=== DIAG 1: Basic spending waterfall ===")
        cfg = SimConfig.default()
        cfg.n_sims = 1
        cfg.monthly_spend = 5000
        cfg.inflation = 0
        cfg.spend_growth = 0
        cfg.start_yr = 2031
        cfg.end_yr = 2032  # 12 months
        cfg.cash_initial = 50000
        cfg.cash_rate = 0  # no interest to simplify
        # Zero out other accounts to isolate cash spending
        cfg.reserve_bins = [{"label": "S", "initial": 0, "rate": 0, "volatility": 0},
                           {"label": "M", "initial": 0, "rate": 0, "volatility": 0},
                           {"label": "L", "initial": 0, "rate": 0, "volatility": 0}]
        cfg.invest_bins = [{"label": "Eq", "initial": 0, "return_rate": 0, "volatility": 0},
                          {"label": "Bd", "initial": 0, "return_rate": 0, "volatility": 0}]
        cfg.high_q_trigger = 0.99  # effectively disable
        cfg.low_q_trigger = 0.01

        model = DiagPriceModel(base_price=50000)
        state = _initial_state(cfg, model=model)
        _log_state("initial", state)
        rng = np.random.default_rng(42)

        for i in range(12):
            price = model.price_at(0.5, state.t + 1/12)
            state = step(state, cfg, price, rng, model=model)
            _log_state(f"month {i+1}", state)

        # After 12 months of $5K/mo spending, cash should be near $0
        # (12 * 5000 = 60000 > 50000, so should hit BTC emergency)
        print(f"\n  Final: cash=${state.cash:.2f}, btc={state.btc_stack:.6f}")
        assert state.cash < 50000, "Cash should decrease from spending"
        # After month 10 ($50K spent), should start hitting reserves/BTC
        assert state.spending_shortfall == 0 or state.btc_stack < 1.0, \
            "Should have drawn from BTC after cash depleted"


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2: Floor enforcement — does cash get replenished?
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagFloors:
    def test_cash_floor_replenished_from_investments(self):
        """Set cash floor = $10K. Start with cash=$5K, investments=$100K.
        Floor should pull from investments to bring cash to $10K."""
        print("\n=== DIAG 2: Cash floor enforcement ===")
        cfg = SimConfig.default()
        cfg.n_sims = 1
        cfg.monthly_spend = 0  # no spending to isolate floor effect
        cfg.cash_initial = 5000
        cfg.cash_floor = 10000
        cfg.cash_rate = 0
        cfg.reserve_bins = [{"label": "S", "initial": 0, "rate": 0, "volatility": 0},
                           {"label": "M", "initial": 0, "rate": 0, "volatility": 0},
                           {"label": "L", "initial": 0, "rate": 0, "volatility": 0}]
        cfg.invest_bins = [{"label": "Eq", "initial": 50000, "return_rate": 0, "volatility": 0},
                          {"label": "Bd", "initial": 50000, "return_rate": 0, "volatility": 0}]
        cfg.high_q_trigger = 0.99
        cfg.low_q_trigger = 0.01
        cfg.start_yr = 2031
        cfg.end_yr = 2032

        model = DiagPriceModel()
        state = _initial_state(cfg, model=model)
        _log_state("initial", state)
        rng = np.random.default_rng(42)

        price = model.price_at(0.5, state.t + 1/12)
        state = step(state, cfg, price, rng, model=model)
        _log_state("after step 1", state)

        assert state.cash >= 10000, f"Cash floor not enforced: cash=${state.cash}"
        total_inv = sum(state.investments)
        assert total_inv < 100000, f"Investments should decrease: ${total_inv}"
        print(f"  PASS: cash=${state.cash:.0f} (floor=$10K), inv={total_inv:.0f}")

    def test_reserve_floor_replenished(self):
        """Reserve short floor = $20K, start with $5K. Should replenish."""
        print("\n=== DIAG 2b: Reserve floor enforcement ===")
        cfg = SimConfig.default()
        cfg.n_sims = 1
        cfg.monthly_spend = 0
        cfg.cash_initial = 0
        cfg.cash_floor = 0
        cfg.cash_rate = 0
        cfg.reserve_bins = [{"label": "S", "initial": 5000, "rate": 0, "volatility": 0},
                           {"label": "M", "initial": 0, "rate": 0, "volatility": 0},
                           {"label": "L", "initial": 0, "rate": 0, "volatility": 0}]
        cfg.reserve_floors = [20000, 0, 0]
        cfg.invest_bins = [{"label": "Eq", "initial": 50000, "return_rate": 0, "volatility": 0},
                          {"label": "Bd", "initial": 50000, "return_rate": 0, "volatility": 0}]
        cfg.high_q_trigger = 0.99
        cfg.low_q_trigger = 0.01
        cfg.start_yr = 2031
        cfg.end_yr = 2032

        model = DiagPriceModel()
        state = _initial_state(cfg, model=model)
        _log_state("initial", state)
        rng = np.random.default_rng(42)

        price = model.price_at(0.5, state.t + 1/12)
        state = step(state, cfg, price, rng, model=model)
        _log_state("after step 1", state)

        assert state.reserves[0] >= 20000, \
            f"Reserve short floor not enforced: ${state.reserves[0]}"


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 3: Rebalancing triggers — do they fire when quantile crosses?
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagRebalancing:
    def test_high_q_trigger_sells_btc(self):
        """Force quantile to 0.90 (above 0.80 threshold). Should sell BTC."""
        print("\n=== DIAG 3: High-Q trigger ===")
        cfg = SimConfig.default()
        cfg.n_sims = 1
        cfg.monthly_spend = 0
        cfg.cash_initial = 0
        cfg.cash_rate = 0
        cfg.high_q_trigger = 0.80
        cfg.high_q_action = {"mode": "lump", "rate": 50.0, "duration": 1,
            "split": {"cash": 1.0, "res_short": 0, "res_med": 0,
                      "res_long": 0, "inv_eq": 0, "inv_bd": 0}}
        cfg.low_q_trigger = 0.01
        cfg.reserve_bins = [{"label": "S", "initial": 0, "rate": 0, "volatility": 0},
                           {"label": "M", "initial": 0, "rate": 0, "volatility": 0},
                           {"label": "L", "initial": 0, "rate": 0, "volatility": 0}]
        cfg.invest_bins = [{"label": "Eq", "initial": 0, "return_rate": 0, "volatility": 0},
                          {"label": "Bd", "initial": 0, "return_rate": 0, "volatility": 0}]
        cfg.start_yr = 2031
        cfg.end_yr = 2032

        # Model that always returns quantile 0.90
        model = DiagPriceModel(base_price=100000)
        model._fixed_q = 0.90

        state = _initial_state(cfg, model=model)
        _log_state("initial", state)
        rng = np.random.default_rng(42)

        price = model.price_at(0.5, state.t + 1/12)
        state = step(state, cfg, price, rng, model=model)
        _log_state("after step 1", state)

        assert state.btc_stack < 1.0, f"BTC should have been sold: {state.btc_stack}"
        assert state.cash > 0, f"Cash should have received proceeds: ${state.cash}"
        assert state.rebal_event is not None, "Rebal event should be logged"
        print(f"  PASS: BTC={state.btc_stack:.4f}, Cash=${state.cash:.0f}, "
              f"Event={state.rebal_event}")

    def test_low_q_trigger_buys_btc(self):
        """Force quantile to 0.10 (below 0.20 threshold). Should buy BTC."""
        print("\n=== DIAG 3b: Low-Q trigger ===")
        cfg = SimConfig.default()
        cfg.n_sims = 1
        cfg.monthly_spend = 0
        cfg.start_stack = 0.5
        cfg.cash_initial = 100000
        cfg.cash_rate = 0
        cfg.high_q_trigger = 0.99
        cfg.low_q_trigger = 0.20
        cfg.low_q_action = {"mode": "lump", "rate": 20.0, "duration": 1,
            "split": {"cash": 1.0, "res_short": 0, "res_med": 0,
                      "res_long": 0, "inv_eq": 0, "inv_bd": 0}}
        cfg.reserve_bins = [{"label": "S", "initial": 0, "rate": 0, "volatility": 0},
                           {"label": "M", "initial": 0, "rate": 0, "volatility": 0},
                           {"label": "L", "initial": 0, "rate": 0, "volatility": 0}]
        cfg.invest_bins = [{"label": "Eq", "initial": 0, "return_rate": 0, "volatility": 0},
                          {"label": "Bd", "initial": 0, "return_rate": 0, "volatility": 0}]
        cfg.start_yr = 2031
        cfg.end_yr = 2032

        model = DiagPriceModel(base_price=20000)
        model._fixed_q = 0.10  # below threshold

        state = _initial_state(cfg, model=model)
        _log_state("initial", state)
        rng = np.random.default_rng(42)

        price = model.price_at(0.5, state.t + 1/12)
        state = step(state, cfg, price, rng, model=model)
        _log_state("after step 1", state)

        assert state.btc_stack > 0.5, f"BTC should have been bought: {state.btc_stack}"
        assert state.cash < 100000, f"Cash should have decreased: ${state.cash}"


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 4: Full simulation trace — 12 months with all features
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagFullTrace:
    def test_12_month_trace_with_floors_and_spending(self):
        """Run 12 months, print every state, verify spending + floors."""
        print("\n=== DIAG 4: 12-month full trace ===")
        cfg = SimConfig.default()
        cfg.n_sims = 1
        cfg.start_yr = 2031
        cfg.end_yr = 2032
        cfg.monthly_spend = 8000
        cfg.inflation = 0
        cfg.spend_growth = 0
        cfg.cash_initial = 30000
        cfg.cash_floor = 10000  # Should replenish cash to $10K each month
        cfg.cash_rate = 0
        cfg.reserve_bins = [
            {"label": "S", "initial": 20000, "rate": 0, "volatility": 0},
            {"label": "M", "initial": 20000, "rate": 0, "volatility": 0},
            {"label": "L", "initial": 0, "rate": 0, "volatility": 0},
        ]
        cfg.reserve_floors = [0, 0, 0]
        cfg.invest_bins = [
            {"label": "Eq", "initial": 50000, "return_rate": 0, "volatility": 0},
            {"label": "Bd", "initial": 50000, "return_rate": 0, "volatility": 0},
        ]
        cfg.high_q_trigger = 0.99
        cfg.low_q_trigger = 0.01

        model = DiagPriceModel(base_price=50000)
        model._fixed_q = 0.50
        state = _initial_state(cfg, model=model)
        _log_state("initial", state)
        rng = np.random.default_rng(42)

        for i in range(12):
            price = model.price_at(0.5, state.t + 1/12)
            state = step(state, cfg, price, rng, model=model)
            _log_state(f"month {i+1}", state)

        # After 12 months: $96K spent. Started with $170K in dollar accounts.
        # Cash floor should have been maintained at $10K until funds run out.
        total_dollar = state.cash + sum(state.reserves) + sum(state.investments)
        print(f"\n  Final dollar assets: ${total_dollar:.0f}")
        print(f"  Final BTC: {state.btc_stack:.6f}")
        # We should have spent from cash, then replenished from investments/reserves
        assert total_dollar < 170000, "Dollar assets should decrease from spending"


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 5: simulate() produces valid output
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagSimulate:
    def test_simulate_returns_decreasing_portfolio(self):
        """With $5K/mo spending and modest assets, portfolio should decline."""
        print("\n=== DIAG 5: simulate() output validation ===")
        cfg = SimConfig.default()
        cfg.n_sims = 1
        cfg.start_yr = 2031
        cfg.end_yr = 2035  # 4 years
        cfg.monthly_spend = 5000
        cfg.high_q_trigger = 0.99
        cfg.low_q_trigger = 0.01

        model = DiagPriceModel(base_price=80000)
        model._fixed_q = 0.50
        result = simulate(cfg, model)

        print(f"  time_axis length: {len(result.time_axis)}")
        print(f"  total_usd shape: {result.total_usd.shape}")
        print(f"  First 5 totals: {result.total_usd[0, :5].tolist()}")
        print(f"  Last 5 totals: {result.total_usd[0, -5:].tolist()}")
        print(f"  Depletion: {result.depletion_period}")
        print(f"  Rebal events: {len(result.rebal_events[0])}")

        # Portfolio should be smaller at end than beginning (spending > returns)
        assert result.total_usd[0, -1] < result.total_usd[0, 0], \
            "Portfolio should decline with constant spending"


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 6: _build_sim_config round-trip from callback params
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagConfigMapping:
    def test_callback_params_reach_engine(self):
        """Verify that callback param keys map correctly to SimConfig."""
        print("\n=== DIAG 6: Callback → SimConfig mapping ===")
        sys.path.insert(0, str(Path(__file__).parent))
        from figures.citadel import _build_sim_config

        p = {
            "start_stack": 2.5,
            "cash_initial": 75000,
            "cash_rate": 3.0,
            "res_short_init": 10000, "res_short_rate": 4.0, "res_short_vol": 1.0,
            "res_med_init": 20000, "res_med_rate": 3.5, "res_med_vol": 5.0,
            "res_long_init": 30000, "res_long_rate": 3.0, "res_long_vol": 10.0,
            "inv_eq_init": 100000, "inv_eq_rate": 8.0, "inv_eq_vol": 15.0,
            "inv_bd_init": 50000, "inv_bd_rate": 4.0, "inv_bd_vol": 5.0,
            "monthly_spend": 7500,
            "inflation": 3.0,
            "spend_growth": 1.0,
            "high_q_trigger": 75,  # UI sends percentage
            "high_q_mode": "lump",
            "high_q_rate": 15.0,
            "high_q_dur": 1,
            "high_q_split_cash": 30, "high_q_split_rs": 20, "high_q_split_rm": 20,
            "high_q_split_rl": 10, "high_q_split_eq": 10, "high_q_split_bd": 10,
            "low_q_trigger": 25,
            "low_q_mode": "gradual",
            "low_q_rate": 5.0,
            "low_q_dur": 6,
            "low_q_split_cash": 20, "low_q_split_rs": 20, "low_q_split_rm": 20,
            "low_q_split_rl": 10, "low_q_split_eq": 20, "low_q_split_bd": 10,
            "lump_cooldown": 6,
            "cash_floor": 15000,
            "res_short_floor": 5000,
            "res_med_floor": 0,
            "res_long_floor": 0,
            "scf_enabled": True,
            "scf_amount": 50000,
            "scf_type": "perpetual",
            "scf_rate": 7.0,
            "scf_term": 48,
            "scf_repay_trigger": 1.5,
            "start_yr": 2033,
            "end_yr": 2070,
            "freq": "Monthly",
            "price_model": "bub",
            "selected_qs": [0.10, 0.50],
            "n_sims": 1,
        }
        cfg = _build_sim_config(p)

        # Verify every field made it through
        assert cfg.start_stack == 2.5, f"start_stack: {cfg.start_stack}"
        assert cfg.cash_initial == 75000, f"cash_initial: {cfg.cash_initial}"
        assert cfg.cash_rate == 3.0, f"cash_rate: {cfg.cash_rate}"
        assert cfg.monthly_spend == 7500, f"monthly_spend: {cfg.monthly_spend}"
        assert cfg.inflation == 3.0, f"inflation: {cfg.inflation}"
        assert cfg.spend_growth == 1.0, f"spend_growth: {cfg.spend_growth}"
        # Triggers — should be converted from percentage to fraction
        assert abs(cfg.high_q_trigger - 0.75) < 0.001, f"high_q_trigger: {cfg.high_q_trigger}"
        assert abs(cfg.low_q_trigger - 0.25) < 0.001, f"low_q_trigger: {cfg.low_q_trigger}"
        assert cfg.high_q_action["mode"] == "lump"
        assert cfg.low_q_action["mode"] == "gradual"
        # Splits — should be fractions
        assert abs(cfg.high_q_action["split"]["cash"] - 0.30) < 0.001
        assert abs(cfg.low_q_action["split"]["inv_eq"] - 0.20) < 0.001
        # Floors
        assert cfg.cash_floor == 15000, f"cash_floor: {cfg.cash_floor}"
        assert cfg.reserve_floors == [5000, 0, 0], f"reserve_floors: {cfg.reserve_floors}"
        # SCF
        assert cfg.scf_enabled is True
        assert cfg.scf_amount == 50000
        assert cfg.scf_type == "perpetual"
        assert cfg.scf_repay_trigger == 1.5
        # n_sims
        assert cfg.n_sims == 1
        print("  PASS: All callback params map correctly to SimConfig")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
