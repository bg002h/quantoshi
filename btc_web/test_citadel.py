import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
for _p in (str(_ROOT), str(_ROOT / "btc_web"), str(_ROOT / "archive" / "btc_app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
from engines.citadel import SimConfig, CitadelState, FREQ_PPY


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
