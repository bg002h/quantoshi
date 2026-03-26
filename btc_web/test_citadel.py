import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
for _p in (str(_ROOT), str(_ROOT / "btc_web"), str(_ROOT / "archive" / "btc_app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest
from engines.citadel import SimConfig, CitadelState, FREQ_PPY, validate_config, _apply_spending_waterfall, _enforce_floors, _get_btc_price, _evaluate_rebalancing


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
        s = CitadelState()
        s.cash = cash
        s.reserves = reserves or [5000.0, 5000.0, 5000.0]
        s.investments = investments or [10000.0, 10000.0]
        s.btc_stack = btc
        s.btc_price = price
        return s

    def test_cash_covers_all(self):
        s = self._make_state(cash=10000)
        shortfall = _apply_spending_waterfall(s, 5000)
        assert shortfall == 0.0
        assert s.cash == 5000.0

    def test_cash_depleted_draws_reserves(self):
        s = self._make_state(cash=2000)
        shortfall = _apply_spending_waterfall(s, 5000)
        assert shortfall == 0.0
        assert s.cash == 0.0
        assert s.reserves[0] == 2000.0  # short lost 3000

    def test_full_waterfall_to_btc(self):
        s = self._make_state(cash=100, reserves=[100, 100, 100],
                             investments=[100, 100], btc=1.0, price=50000)
        shortfall = _apply_spending_waterfall(s, 1000)
        assert shortfall == 0.0
        assert s.cash == 0.0
        assert all(r == 0.0 for r in s.reserves)
        assert all(inv == 0.0 for inv in s.investments)
        # Remaining 400 drawn from BTC (400/50000 = 0.008 BTC)
        assert abs(s.btc_stack - (1.0 - 400 / 50000)) < 1e-10

    def test_total_depletion(self):
        s = self._make_state(cash=100, reserves=[0, 0, 0],
                             investments=[0, 0], btc=0.001, price=50000)
        shortfall = _apply_spending_waterfall(s, 1000)
        assert shortfall > 0  # can't cover full spend
        assert s.btc_stack == 0.0
        assert s.cash == 0.0

    def test_zero_spend(self):
        s = self._make_state(cash=10000)
        shortfall = _apply_spending_waterfall(s, 0)
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
