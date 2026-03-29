import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
for _p in (str(_ROOT), str(_ROOT / "btc_web"), str(_ROOT / "archive" / "btc_app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math

import numpy as np
import pytest
from engines.citadel import SimConfig, CitadelState, FREQ_PPY, validate_config, _spending_waterfall, _enforce_floors, _get_btc_price, _evaluate_rebalancing, _lognormal_return, _initial_state, step, _scf_payment_amount, _scf_check_repay, SimResult, simulate, _compute_n_periods


class MockPriceModel:
    """Simple mock: price = 1000 * quantile * time."""
    def __init__(self):
        self.fits = {0.01: None, 0.10: None, 0.25: None, 0.50: None,
                     0.75: None, 0.90: None, 0.99: None}
        self.genesis = 14822.375

    def price_at(self, q: float, t: float) -> float:
        return 1000.0 * q * max(t, 0.1)

    def quantile_at(self, price: float, t: float) -> float:
        q = price / (1000.0 * max(t, 0.1))
        return max(0.001, min(q, 0.999))

def _mock_model_data():
    return MockPriceModel()


class TestSimConfig:
    def test_default_config(self):
        cfg = SimConfig.default()
        assert cfg.price_model == "bub"
        assert cfg.start_stack == 1.0
        assert cfg.cash_initial == 50000.0
        assert cfg.monthly_spend == 5000.0
        assert len(cfg.reserve_bins) == 3
        assert len(cfg.invest_bins) == 2
        assert cfg.high_q_trigger > cfg.low_q_trigger
        assert cfg.n_sims == 1

    def test_freq_ppy(self):
        assert FREQ_PPY["Monthly"] == 12
        assert FREQ_PPY["Quarterly"] == 4
        assert FREQ_PPY["Annually"] == 1


class TestConfigValidation:
    def test_valid_default_passes(self):
        validate_config(SimConfig.default())  # should not raise

    def test_inverted_triggers_rejected(self):
        cfg = SimConfig.default()
        cfg.high_q_trigger = 0.20
        cfg.low_q_trigger = 0.80
        with pytest.raises(ValueError, match="high_q_trigger"):
            validate_config(cfg)

    def test_triggers_too_close_rejected(self):
        cfg = SimConfig.default()
        cfg.high_q_trigger = 0.52
        cfg.low_q_trigger = 0.50
        with pytest.raises(ValueError, match="5 percentile"):
            validate_config(cfg)

    def test_split_not_summing_to_one(self):
        cfg = SimConfig.default()
        cfg.high_q_action["split"] = {"cash": 0.5, "res_short": 0.5,
            "res_med": 0.5, "res_long": 0.0, "inv_eq": 0.0, "inv_bd": 0.0}
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_config(cfg)

    def test_negative_initial_balance(self):
        cfg = SimConfig.default()
        cfg.cash_initial = -100
        with pytest.raises(ValueError, match="non-negative"):
            validate_config(cfg)

    def test_invalid_freq(self):
        cfg = SimConfig.default()
        cfg.freq = "Daily"
        with pytest.raises(ValueError, match="freq"):
            validate_config(cfg)

    def test_bad_date_range(self):
        cfg = SimConfig.default()
        cfg.start_yr = 2080
        cfg.end_yr = 2030
        with pytest.raises(ValueError, match="start_yr"):
            validate_config(cfg)

    def test_scf_term_zero_rejected(self):
        cfg = SimConfig.default()
        cfg.scf_enabled = True
        cfg.scf_type = "term"
        cfg.scf_term = 0
        with pytest.raises(ValueError, match="scf_term"):
            validate_config(cfg)


class TestSpendingWaterfall:
    def _make_state(self, cash=10000, reserves=None, investments=None, btc=1.0, price=50000):
        s = CitadelState(sim_date="2035-01-15")
        s.cash = cash
        s.reserves = reserves or [5000.0, 5000.0, 5000.0]
        s.investments = investments or [10000.0, 10000.0]
        s.invest_cost_basis = [v for v in s.investments]
        s.btc_stack = btc
        s.btc_price = price
        return s

    def test_cash_covers_all(self):
        s = self._make_state(cash=10000)
        shortfall = _spending_waterfall(s, SimConfig(), 5000)
        assert shortfall == 0.0
        assert s.cash == 5000.0

    def test_cash_depleted_draws_reserves(self):
        s = self._make_state(cash=2000)
        shortfall = _spending_waterfall(s, SimConfig(), 5000)
        assert shortfall == 0.0
        assert s.cash == 0.0
        assert s.reserves[0] == 2000.0  # short lost 3000

    def test_full_waterfall_to_btc(self):
        s = self._make_state(cash=100, reserves=[100, 100, 100],
                             investments=[100, 100], btc=1.0, price=50000)
        shortfall = _spending_waterfall(s, SimConfig(), 1000)
        assert shortfall == 0.0
        assert s.cash == 0.0
        assert all(r == 0.0 for r in s.reserves)
        # Growth-aware ordering (no model → BTC growth == equity → BTC sold first)
        # Remaining 600 after cash(100)+reserves(300) drawn from BTC (600/50000 = 0.012)
        assert abs(s.btc_stack - (1.0 - 600 / 50000)) < 1e-10
        # Investments untouched since BTC covered the shortfall
        assert s.investments == [100, 100]

    def test_total_depletion(self):
        s = self._make_state(cash=100, reserves=[0, 0, 0],
                             investments=[0, 0], btc=0.001, price=50000)
        shortfall = _spending_waterfall(s, SimConfig(), 1000)
        assert shortfall > 0  # can't cover full spend
        assert s.btc_stack == 0.0
        assert s.cash == 0.0

    def test_zero_spend(self):
        s = self._make_state(cash=10000)
        shortfall = _spending_waterfall(s, SimConfig(), 0)
        assert shortfall == 0.0
        assert s.cash == 10000.0


class TestFloorEnforcement:
    def test_cash_below_floor_replenished(self):
        s = CitadelState()
        s.cash = 1000
        s.reserves = [5000.0, 5000.0, 5000.0]
        s.investments = [10000.0, 10000.0]
        cfg = SimConfig.default()
        cfg.cash_floor = 5000
        cfg.reserve_floors = [0, 0, 0]
        _enforce_floors(s, cfg)
        assert s.cash >= 5000.0
        # Drawn from investments (bonds first)
        assert s.investments[1] < 10000.0

    def test_reserve_below_floor_replenished(self):
        s = CitadelState()
        s.cash = 50000
        s.reserves = [100.0, 5000.0, 5000.0]
        s.investments = [10000.0, 10000.0]
        cfg = SimConfig.default()
        cfg.cash_floor = 0
        cfg.reserve_floors = [5000, 0, 0]
        _enforce_floors(s, cfg)
        assert s.reserves[0] >= 5000.0

    def test_insufficient_funds_partial_fill(self):
        s = CitadelState()
        s.cash = 100
        s.reserves = [100.0, 100.0, 100.0]
        s.investments = [100.0, 100.0]
        cfg = SimConfig.default()
        cfg.cash_floor = 99999
        cfg.reserve_floors = [0, 0, 0]
        _enforce_floors(s, cfg)
        assert s.cash > 100  # got some replenishment
        assert s.btc_stack == 0.0  # BTC not touched

    def test_no_floors_no_change(self):
        s = CitadelState()
        s.cash = 1000
        s.reserves = [500.0, 500.0, 500.0]
        s.investments = [500.0, 500.0]
        cfg = SimConfig.default()
        cfg.cash_floor = 0
        cfg.reserve_floors = [0, 0, 0]
        _enforce_floors(s, cfg)
        assert s.cash == 1000


class TestBTCPricing:
    def test_get_btc_price_deterministic(self):
        model = _mock_model_data()
        rng = np.random.default_rng(42)
        cfg = SimConfig.default()
        cfg.n_sims = 1
        cfg.selected_qs = [0.50]
        price = _get_btc_price(t=10.0, config=cfg, model=model,
                               rng=rng, sim_mode="deterministic", q=0.50)
        assert price == model.price_at(0.50, 10.0)

    def test_price_to_quantile_roundtrip(self):
        model = _mock_model_data()
        t = 15.0
        for q in [0.10, 0.50, 0.90]:
            price = model.price_at(q, t)
            q_back = model.quantile_at(price, t)
            assert abs(q_back - q) < 0.01


class TestRebalancing:
    def test_high_q_lump_sells_btc(self):
        s = CitadelState()
        s.btc_stack = 10.0
        s.btc_price = 100000.0
        s.cash = 50000.0
        s.reserves = [10000.0, 10000.0, 10000.0]
        s.investments = [50000.0, 50000.0]
        cfg = SimConfig.default()
        cfg.high_q_trigger = 0.80  # explicit for test
        cfg.high_q_action = {"mode": "lump", "rate": 10.0, "duration": 1,
            "split": {"cash": 1.0, "res_short": 0.0, "res_med": 0.0,
                      "res_long": 0.0, "inv_eq": 0.0, "inv_bd": 0.0}}
        _evaluate_rebalancing(s, cfg, btc_quantile=0.90)
        assert s.btc_stack < 10.0
        assert s.cash > 50000.0
        assert s.rebal_cooldown == cfg.lump_cooldown

    def test_low_q_lump_buys_btc(self):
        s = CitadelState()
        s.btc_stack = 1.0
        s.btc_price = 20000.0
        s.cash = 50000.0
        s.reserves = [10000.0, 10000.0, 10000.0]
        s.investments = [50000.0, 50000.0]
        cfg = SimConfig.default()
        cfg.low_q_trigger = 0.20  # explicit for test
        cfg.low_q_action = {"mode": "lump", "rate": 10.0, "duration": 1,
            "split": {"cash": 0.5, "res_short": 0.0, "res_med": 0.0,
                      "res_long": 0.0, "inv_eq": 0.5, "inv_bd": 0.0}}
        _evaluate_rebalancing(s, cfg, btc_quantile=0.10)
        assert s.btc_stack > 1.0
        assert s.cash < 50000.0

    def test_cooldown_prevents_lump(self):
        s = CitadelState()
        s.btc_stack = 10.0
        s.btc_price = 100000.0
        s.rebal_cooldown = 5
        s.cash = 50000.0
        s.reserves = [10000.0, 10000.0, 10000.0]
        s.investments = [50000.0, 50000.0]
        cfg = SimConfig.default()
        cfg.high_q_trigger = 0.80
        cfg.high_q_action = {"mode": "lump", "rate": 10.0, "duration": 1,
            "split": {"cash": 1.0, "res_short": 0.0, "res_med": 0.0,
                      "res_long": 0.0, "inv_eq": 0.0, "inv_bd": 0.0}}
        _evaluate_rebalancing(s, cfg, btc_quantile=0.90)
        assert s.btc_stack == 10.0  # nothing sold

    def test_gradual_starts_and_continues(self):
        s = CitadelState()
        s.btc_stack = 10.0
        s.btc_price = 100000.0
        s.cash = 0.0
        s.reserves = [0.0, 0.0, 0.0]
        s.investments = [0.0, 0.0]
        cfg = SimConfig.default()
        cfg.high_q_trigger = 0.80
        cfg.high_q_action = {"mode": "gradual", "rate": 5.0, "duration": 3,
            "split": {"cash": 1.0, "res_short": 0.0, "res_med": 0.0,
                      "res_long": 0.0, "inv_eq": 0.0, "inv_bd": 0.0}}
        _evaluate_rebalancing(s, cfg, btc_quantile=0.90)
        assert s.grad_active
        assert s.grad_remaining == 2
        btc_after_1 = s.btc_stack
        assert btc_after_1 < 10.0
        _evaluate_rebalancing(s, cfg, btc_quantile=0.50)
        assert s.grad_remaining == 1
        assert s.btc_stack < btc_after_1

    def test_gradual_blocks_new_trigger(self):
        s = CitadelState()
        s.btc_stack = 10.0
        s.btc_price = 100000.0
        s.grad_active = True
        s.grad_remaining = 5
        s.grad_rate = 2.0
        s.grad_direction = "sell_btc"
        s.grad_split = {"cash": 1.0, "res_short": 0.0, "res_med": 0.0,
                        "res_long": 0.0, "inv_eq": 0.0, "inv_bd": 0.0}
        s.cash = 0.0
        s.reserves = [0.0, 0.0, 0.0]
        s.investments = [0.0, 0.0]
        cfg = SimConfig.default()
        _evaluate_rebalancing(s, cfg, btc_quantile=0.10)
        assert s.grad_direction == "sell_btc"
        assert s.grad_remaining == 4

    def test_no_trigger_in_neutral_zone(self):
        s = CitadelState()
        s.btc_stack = 10.0
        s.btc_price = 50000.0
        s.cash = 50000.0
        s.reserves = [10000.0, 10000.0, 10000.0]
        s.investments = [50000.0, 50000.0]
        cfg = SimConfig.default()
        _evaluate_rebalancing(s, cfg, btc_quantile=0.50)
        assert s.rebal_event is None


class TestLognormalReturns:
    def test_deterministic_return(self):
        r = _lognormal_return(0.10, 0.16, 12, deterministic=True)
        expected = (1 + 0.10) ** (1/12) - 1
        assert abs(r - expected) < 1e-10

    def test_stochastic_mode_has_variance(self):
        rng = np.random.default_rng(42)
        returns = [_lognormal_return(0.10, 0.16, 12, deterministic=False, rng=rng)
                   for _ in range(100)]
        assert max(returns) != min(returns)

    def test_returns_always_above_minus_one(self):
        rng = np.random.default_rng(42)
        for _ in range(10000):
            r = _lognormal_return(0.05, 0.30, 12, deterministic=False, rng=rng)
            assert r > -1.0


class TestStepFunction:
    def test_basic_step_advances_period(self):
        cfg = SimConfig.default()
        cfg.n_sims = 1
        s = _initial_state(cfg)
        rng = np.random.default_rng(42)
        s2 = step(s, cfg, btc_price_new=50000.0, rng=rng)
        assert s2.period == 1
        assert s2.btc_price == 50000.0

    def test_cash_earns_interest(self):
        cfg = SimConfig.default()
        cfg.n_sims = 1
        cfg.cash_rate = 12.0
        cfg.monthly_spend = 0
        cfg.cash_initial = 10000
        s = _initial_state(cfg)
        rng = np.random.default_rng(42)
        s2 = step(s, cfg, btc_price_new=50000.0, rng=rng)
        expected = 10000 * (1 + 0.12) ** (1/12)
        assert abs(s2.cash - expected) < 0.01

    def test_spending_reduces_cash(self):
        cfg = SimConfig.default()
        cfg.n_sims = 1
        cfg.monthly_spend = 1000
        cfg.inflation = 0
        cfg.spend_growth = 0
        cfg.cash_initial = 100000
        s = _initial_state(cfg)
        rng = np.random.default_rng(42)
        s2 = step(s, cfg, btc_price_new=50000.0, rng=rng)
        assert s2.cash < 100000


class TestSaylorFortifier:
    def test_scf_init_buys_btc(self):
        cfg = SimConfig.default()
        cfg.scf_enabled = True
        cfg.scf_amount = 100000
        cfg.scf_type = "term"
        cfg.scf_rate = 8.0
        cfg.scf_term = 60
        s = _initial_state(cfg, model=_mock_model_data())
        assert s.scf_active
        assert s.scf_outstanding == 100000
        assert s.btc_stack > cfg.start_stack

    def test_term_loan_monthly_payment(self):
        cfg = SimConfig.default()
        cfg.scf_enabled = True
        cfg.scf_amount = 120000
        cfg.scf_rate = 12.0
        cfg.scf_type = "term"
        cfg.scf_term = 12
        pmt = _scf_payment_amount(cfg, 12)
        assert 10000 < pmt < 11000

    def test_perpetual_interest_only(self):
        cfg = SimConfig.default()
        cfg.scf_enabled = True
        cfg.scf_amount = 100000
        cfg.scf_rate = 12.0
        cfg.scf_type = "perpetual"
        pmt = _scf_payment_amount(cfg, 12)
        assert abs(pmt - 1000) < 1

    def test_perpetual_repay_trigger(self):
        cfg = SimConfig.default()
        cfg.scf_enabled = True
        cfg.scf_type = "perpetual"
        cfg.scf_rate = 8.0
        cfg.scf_repay_trigger = 1.0
        s = CitadelState()
        s.scf_active = True
        s.scf_outstanding = 100000
        s.btc_stack = 5.0
        s.btc_price = 50000
        _scf_check_repay(s, cfg, btc_annual_return=0.05)
        assert s.scf_outstanding == 0
        assert s.btc_stack < 5.0

    def test_scf_disabled_no_effect(self):
        cfg = SimConfig.default()
        cfg.scf_enabled = False
        s = _initial_state(cfg)
        assert not s.scf_active
        assert s.scf_outstanding == 0


class TestSimulate:
    def test_single_sim_returns_result(self):
        cfg = SimConfig.default()
        cfg.n_sims = 1
        cfg.start_yr = 2031
        cfg.end_yr = 2035
        result = simulate(cfg, _mock_model_data())
        assert result.time_axis is not None
        assert len(result.time_axis) == 48  # 4 years * 12 months
        assert result.btc_holdings.shape == (1, 48)
        assert result.total_usd.shape == (1, 48)
        assert len(result.depletion_period) == 1

    def test_result_serialization_roundtrip(self):
        cfg = SimConfig.default()
        cfg.n_sims = 1
        cfg.start_yr = 2031
        cfg.end_yr = 2033
        result = simulate(cfg, _mock_model_data())
        d = result.to_dict()
        assert isinstance(d, dict)
        result2 = SimResult.from_dict(d)
        assert np.allclose(result.total_usd, result2.total_usd)
        assert result.depletion_period == result2.depletion_period

    def test_n_periods_calculation(self):
        cfg = SimConfig.default()
        cfg.start_yr = 2031
        cfg.end_yr = 2075
        cfg.freq = "Monthly"
        assert _compute_n_periods(cfg) == 528  # 44 * 12


class TestSimulateMultiSim:
    def test_price_paths_produces_multi_sim(self):
        """simulate() with price_paths should produce n_sims results."""
        cfg = SimConfig.default()
        cfg.start_yr = 2031
        cfg.end_yr = 2033  # 24 months
        model = _mock_model_data()
        price_paths = np.array([[50000 + i * 100 + j * 10 for j in range(24)]
                                for i in range(5)])
        result = simulate(cfg, model, price_paths=price_paths)
        assert result.btc_holdings.shape == (5, 24)
        assert result.total_usd.shape == (5, 24)
        assert len(result.depletion_period) == 5
        assert "total" in result.median
        assert "btc_usd" in result.median
        assert result.median["total"].shape == (24,)

    def test_fan_band_spread(self):
        """MC with varying paths should produce nonzero percentile spread."""
        cfg = SimConfig.default()
        cfg.start_yr = 2031
        cfg.end_yr = 2032  # 12 months
        model = _mock_model_data()
        # 10 paths with increasing price levels
        paths = np.array([[30000 + i * 20000 + j * 100 for j in range(12)]
                          for i in range(10)])
        result = simulate(cfg, model, price_paths=paths)
        p5 = result.percentiles[5]["total"]
        p95 = result.percentiles[95]["total"]
        assert np.any(p95 > p5), "Fan bands should have nonzero spread"

    def test_price_paths_too_short_raises(self):
        """price_paths with fewer steps than needed should raise."""
        cfg = SimConfig.default()
        cfg.start_yr = 2031
        cfg.end_yr = 2033  # needs 24 steps
        model = _mock_model_data()
        short_paths = np.array([[50000] * 10])  # only 10 steps
        with pytest.raises(ValueError, match="price_paths"):
            simulate(cfg, model, price_paths=short_paths)


class TestAdapter:
    def test_submit_returns_result(self):
        from engines.adapter import submit_simulation
        cfg = SimConfig.default()
        cfg.n_sims = 1
        cfg.start_yr = 2031
        cfg.end_yr = 2033
        result = submit_simulation(cfg, _mock_model_data())
        assert result.time_axis is not None
        assert result.total_usd.shape[0] == 1


class TestCitadelSnapshot:
    def test_roundtrip(self):
        from snapshot import _encode_snapshot, _decode_snapshot
        state = {"cp-stack:value": 2.5, "cp-spend:value": 8000,
                 "cp-qs:value": 0.25}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["cp-stack:value"] == 2.5
        assert decoded["cp-spend:value"] == 8000
        assert decoded["cp-qs:value"] == 0.25
