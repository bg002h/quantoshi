# Citadel Planner (Tab 9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Tab 9 "Citadel Planner" — a multi-asset retirement simulator with BTC, cash, treasuries, investments, rebalancing rules, spending waterfall, and leveraged BTC accumulation.

**Architecture:** Bottom-up build: (1) pure-Python simulation engine with full test coverage, (2) Dash layout with tabbed sub-panels, (3) figure builder for multi-line chart, (4) callback wiring with MC payment integration, (5) snapshot/routing integration. The engine has zero Dash dependencies and communicates via serializable dataclasses. BTC pricing is bridged via a `PriceModel` protocol object passed into `simulate()` — the callback layer constructs this from `_app_ctx.PRICE_MODELS`.

**Key integration note:** The figure builder `build_citadel_figure(m, p)` calls the simulation engine internally (SimConfig assembled from params dict `p`). This means the LRU cache works identically to other tabs — the cache key is the quantized params dict.

**Naming convention:** Both high_q_action and low_q_action use `"split"` as the dict key for distribution weights (not `"source_split"` for low). The spec should be updated to match.

**Tab numbering:** The app currently has 8 tabs (/1=bubble through /8=faq). Tab 9 at `/9` is the Citadel Planner. Note: CLAUDE.md shows /7=model_info, /8=faq (correct).

**Tech Stack:** Python 3.12+, NumPy, Plotly, Dash 4.0.0, DBC 2.0.4

**Spec:** `docs/superpowers/specs/2026-03-26-citadel-planner-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `btc_web/engines/__init__.py` | Engine package marker |
| `btc_web/engines/citadel.py` | Simulation engine: SimConfig, CitadelState, SimResult, step(), simulate() |
| `btc_web/engines/adapter.py` | Thin adapter: submit_simulation() — in-process for v1, Celery-ready interface |
| `btc_web/layout/citadel.py` | Tab 9 layout: tabbed sub-panels (Assets/Spending/Rules/Simulation), all controls |
| `btc_web/figures/citadel.py` | Chart builder: build_citadel_figure() — multi-line portfolio chart |
| `btc_web/callbacks/citadel_cb.py` | Tab 9 callback: update_citadel() — params assembly, engine call, chart render |
| `btc_web/test_citadel.py` | All Tab 9 tests: engine unit tests, integration tests, snapshot tests |

### Modified Files
| File | Change |
|------|--------|
| `btc_web/layout/__init__.py` | Import `_citadel_tab`, register tab, add MC stores for "cp" prefix |
| `btc_web/callbacks/__init__.py` | Import and re-export `citadel_cb` |
| `btc_web/callbacks/nav.py` | Add `/9` routing, `_TAB_CONTROLS["citadel"]` |
| `btc_web/snapshot.py` | Append ~62 entries to `_SNAPSHOT_CONTROLS`, add `_CHECKLIST_OPTIONS` |
| `btc_web/utils.py` | Add `_cached_citadel_fig`, `_get_citadel_fig` |
| `btc_web/app.py` | Add `/9` to cache-control paths, add prewarm call |

---

### Task 1: Engine — SimConfig and CitadelState dataclasses

**Files:**
- Create: `btc_web/engines/__init__.py`
- Create: `btc_web/engines/citadel.py`
- Create: `btc_web/test_citadel.py`

- [ ] **Step 1: Create engine package init**

```python
# btc_web/engines/__init__.py
"""Simulation engines — pure Python, no Dash dependencies."""
```

- [ ] **Step 2: Write failing test for SimConfig creation**

```python
# btc_web/test_citadel.py
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_citadel.py::TestSimConfig -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engines.citadel'`

- [ ] **Step 4: Implement SimConfig and CitadelState**

```python
# btc_web/engines/citadel.py
"""Citadel Planner simulation engine — pure Python + NumPy, zero Dash deps."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_citadel.py::TestSimConfig -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add btc_web/engines/__init__.py btc_web/engines/citadel.py btc_web/test_citadel.py
git commit -m "feat(citadel): add SimConfig and CitadelState dataclasses"
```

---

### Task 2: Engine — Config validation

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Modify: `btc_web/test_citadel.py`

- [ ] **Step 1: Write failing tests for validation**

```python
# Add to btc_web/test_citadel.py
from engines.citadel import validate_config


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_citadel.py::TestConfigValidation -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement validate_config**

```python
# Add to btc_web/engines/citadel.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_citadel.py::TestConfigValidation -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_citadel.py
git commit -m "feat(citadel): add config validation with full edge case coverage"
```

---

### Task 3: Engine — Spending waterfall

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Modify: `btc_web/test_citadel.py`

- [ ] **Step 1: Write failing tests for spending waterfall**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_citadel.py::TestSpendingWaterfall -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement spending waterfall**

```python
# Add to btc_web/engines/citadel.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_citadel.py::TestSpendingWaterfall -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_citadel.py
git commit -m "feat(citadel): implement spending waterfall with BTC last-resort"
```

---

### Task 4: Engine — Floor enforcement

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Modify: `btc_web/test_citadel.py`

- [ ] **Step 1: Write failing tests for floor enforcement**

```python
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
        # Can't fully meet floor, but all available funds go to cash
        assert s.cash > 100  # got some replenishment
        # BTC not touched
        assert s.btc_stack == 0.0  # was already 0

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_citadel.py::TestFloorEnforcement -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement floor enforcement**

```python
# Add to btc_web/engines/citadel.py

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
    # Collect all (account_ref, floor, current_balance) tuples in priority order
    # Cash first, then reserves short->med->long
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

        # Draw from sources in reverse priority order
        # Skip the account being replenished
        sources = []
        # Investments: bonds (1) then equities (0)
        for i in reversed(range(len(state.investments))):
            sources.append(("inv", i))
        # Reserves: long (2) then med (1) then short (0)
        for i in reversed(range(len(state.reserves))):
            if acct_key != f"reserve_{i}":
                sources.append(("res", i))
        # Cash (only if we're replenishing a reserve, not cash itself)
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

        # Apply replenishment
        replenished = (floor - current) - deficit
        if acct_key == "cash":
            state.cash += replenished
        else:
            idx = int(acct_key.split("_")[1])
            state.reserves[idx] += replenished
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_citadel.py::TestFloorEnforcement -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_citadel.py
git commit -m "feat(citadel): implement account floor enforcement"
```

---

### Task 4b: Engine — BTC pricing bridge and quantile inversion

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Modify: `btc_web/test_citadel.py`

The engine needs to: (a) get BTC prices for each period, and (b) invert price→quantile for rebalancing triggers. Both require access to the QR model fits without importing Dash modules.

**Solution:** Define a `PriceModel` protocol that the engine accepts. The callback layer constructs it from `_app_ctx.PRICE_MODELS`. For tests, a mock implementation is used.

- [ ] **Step 1: Write failing tests**

```python
from engines.citadel import PriceModel, _price_to_quantile, _get_btc_price


class MockPriceModel:
    """Simple mock: price = 1000 * quantile * time."""
    def __init__(self):
        self.fits = {0.01: None, 0.10: None, 0.25: None, 0.50: None,
                     0.75: None, 0.90: None, 0.99: None}
        self.genesis = 14822.375  # 2009-07-25 as days

    def price_at(self, q: float, t: float) -> float:
        return 1000.0 * q * max(t, 0.1)

    def quantile_at(self, price: float, t: float) -> float:
        """Inverse: given price and time, return quantile [0,1]."""
        q = price / (1000.0 * max(t, 0.1))
        return max(0.001, min(q, 0.999))


def _mock_model_data():
    return MockPriceModel()


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
```

- [ ] **Step 2: Implement PriceModel protocol and pricing functions**

```python
# Add to btc_web/engines/citadel.py
from typing import Protocol, runtime_checkable


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
    # MC mode: draw from transition matrix
    # (v1: not implemented, will use markov.sample_path)
    raise NotImplementedError("MC BTC pricing requires Markov engine integration")
```

The real `_app_ctx.DEFAULT_MODEL` already has `price_at(q, t)`. For `quantile_at(price, t)`, the callback layer will wrap `_find_lot_percentile` or build a simple bisection search over the quantile curves.

- [ ] **Step 3: Run tests, verify pass, commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_citadel.py
git commit -m "feat(citadel): add PriceModel protocol and BTC pricing bridge"
```

---

### Task 5: Engine — Rebalancing logic

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Modify: `btc_web/test_citadel.py`

- [ ] **Step 1: Write failing tests for rebalancing**

```python
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
        _evaluate_rebalancing(s, cfg, btc_quantile=0.90)  # above 0.80
        assert s.btc_stack < 10.0  # sold some BTC
        assert s.cash > 50000.0    # proceeds went to cash
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
        _evaluate_rebalancing(s, cfg, btc_quantile=0.10)  # below 0.20
        assert s.btc_stack > 1.0   # bought BTC
        assert s.cash < 50000.0    # cash drawn

    def test_cooldown_prevents_lump(self):
        s = CitadelState()
        s.btc_stack = 10.0
        s.btc_price = 100000.0
        s.rebal_cooldown = 5  # still cooling down
        s.cash = 50000.0
        s.reserves = [10000.0, 10000.0, 10000.0]
        s.investments = [50000.0, 50000.0]
        cfg = SimConfig.default()
        cfg.high_q_action = {"mode": "lump", "rate": 10.0, "duration": 1,
            "split": {"cash": 1.0, "res_short": 0.0, "res_med": 0.0,
                      "res_long": 0.0, "inv_eq": 0.0, "inv_bd": 0.0}}
        _evaluate_rebalancing(s, cfg, btc_quantile=0.90)
        assert s.btc_stack == 10.0  # nothing sold — on cooldown

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
        # First call: starts gradual
        _evaluate_rebalancing(s, cfg, btc_quantile=0.90)
        assert s.grad_active
        assert s.grad_remaining == 2  # 3 total, 1 executed
        btc_after_1 = s.btc_stack
        assert btc_after_1 < 10.0

        # Second call: continues (even though quantile may have changed)
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
        # Low trigger fires but gradual is active — should be ignored
        _evaluate_rebalancing(s, cfg, btc_quantile=0.10)
        assert s.grad_direction == "sell_btc"  # not overridden
        assert s.grad_remaining == 4  # continued, not reset

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
```

- [ ] **Step 2: Implement `_evaluate_rebalancing(state, config, btc_quantile)`**

```python
# Add to btc_web/engines/citadel.py

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
    Returns actual amount sourced (may be less if accounts insufficient)."""
    total_sourced = 0.0
    targets = [
        ("cash", split.get("cash", 0)),
        ("res_0", split.get("res_short", 0)),
        ("res_1", split.get("res_med", 0)),
        ("res_2", split.get("res_long", 0)),
        ("inv_0", split.get("inv_eq", 0)),
        ("inv_1", split.get("inv_bd", 0)),
    ]
    for acct, frac in targets:
        want = amount * frac
        if want <= 0:
            continue
        if acct == "cash":
            got = min(state.cash, want)
            state.cash -= got
        elif acct.startswith("res_"):
            i = int(acct[-1])
            got = min(state.reserves[i], want)
            state.reserves[i] -= got
        elif acct.startswith("inv_"):
            i = int(acct[-1])
            got = min(state.investments[i], want)
            state.investments[i] -= got
        else:
            got = 0
        total_sourced += got
    return total_sourced


def _execute_sell_btc(state: CitadelState, rate_pct: float, split: dict) -> dict:
    """Sell rate_pct% of BTC stack, distribute proceeds via split."""
    btc_to_sell = state.btc_stack * (rate_pct / 100.0)
    if btc_to_sell <= 0 or state.btc_price <= 0:
        return {}
    proceeds = btc_to_sell * state.btc_price
    state.btc_stack -= btc_to_sell
    _distribute_to_accounts(state, proceeds, split)
    return {"action": "sell_btc", "btc_sold": btc_to_sell,
            "proceeds": proceeds}


def _execute_buy_btc(state: CitadelState, rate_pct: float,
                     split: dict) -> dict:
    """Source funds from accounts via split, buy BTC."""
    # Calculate target dollar amount based on rate% of total dollar assets
    total_dollar = state.cash + sum(state.reserves) + sum(state.investments)
    target = total_dollar * (rate_pct / 100.0)
    if target <= 0 or state.btc_price <= 0:
        return {}
    sourced = _source_from_accounts(state, target, split)
    btc_bought = sourced / state.btc_price
    state.btc_stack += btc_bought
    return {"action": "buy_btc", "btc_bought": btc_bought,
            "cost": sourced}


def _evaluate_rebalancing(state: CitadelState, config: SimConfig,
                          btc_quantile: float) -> None:
    """Evaluate and execute rebalancing triggers. Mutates state."""
    state.rebal_event = None

    # Decrement cooldown
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
```

- [ ] **Step 3: Run tests, verify pass**

- [ ] **Step 4: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_citadel.py
git commit -m "feat(citadel): implement rebalancing triggers with gradual/lump actions"
```

---

### Task 6: Engine — Lognormal returns and step function

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Modify: `btc_web/test_citadel.py`

- [ ] **Step 1: Write failing tests**

```python
class TestLognormalReturns:
    def test_deterministic_return(self):
        """Deterministic mode uses compound return, no randomness."""
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
        cfg.cash_rate = 12.0  # 12% annual = ~1% monthly
        cfg.monthly_spend = 0  # no spending
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
        # Cash should be ~100000 + interest - 1000
        assert s2.cash < 100000
```

- [ ] **Step 2: Implement `_lognormal_return`, `_initial_state`, `step`**

```python
# Add to btc_web/engines/citadel.py

def _lognormal_return(annual_rate: float, annual_vol: float, ppy: int,
                      deterministic: bool = False,
                      rng: np.random.Generator | None = None) -> float:
    """One-period return using lognormal model. Always > -1.0.
    - annual_rate: expected annual return as decimal (e.g., 0.10 for 10%)
    - annual_vol: annual volatility as decimal (e.g., 0.16 for 16%)
    - ppy: periods per year
    - deterministic: if True, return compound expected rate (no randomness)
    """
    if deterministic:
        return (1 + annual_rate) ** (1.0 / ppy) - 1.0
    if annual_rate <= 0 or annual_vol <= 0:
        # Fallback for edge cases: simple compound return
        return (1 + annual_rate) ** (1.0 / ppy) - 1.0
    sigma_ln = math.sqrt(math.log(1 + (annual_vol / (1 + annual_rate)) ** 2))
    mu_ln = math.log(1 + annual_rate) - sigma_ln ** 2 / 2
    period_mu = mu_ln / ppy
    period_sigma = sigma_ln / math.sqrt(ppy)
    return math.exp(rng.normal(period_mu, period_sigma)) - 1.0


def _initial_state(config: SimConfig, model: PriceModel | None = None) -> CitadelState:
    """Create initial state from config."""
    from btc_core import yr_to_t
    t0 = yr_to_t(config.start_yr, 14822.375)  # genesis days
    # Get initial BTC price
    btc_price = 0.0
    if model and config.selected_qs:
        q = config.selected_qs[len(config.selected_qs) // 2]  # median quantile
        btc_price = float(model.price_at(q, max(t0, 0.5)))

    state = CitadelState(
        t=t0,
        btc_stack=config.start_stack,
        btc_price=btc_price,
        btc_cost_basis=btc_price,
        cash=config.cash_initial,
        reserves=[rb["initial"] for rb in config.reserve_bins],
        investments=[ib["initial"] for ib in config.invest_bins],
    )
    # Initialize SCF if enabled
    if config.scf_enabled and config.scf_amount > 0 and btc_price > 0:
        btc_bought = config.scf_amount / btc_price
        state.btc_stack += btc_bought
        state.scf_outstanding = config.scf_amount
        state.scf_active = True
        # Update cost basis (weighted average)
        total_btc = config.start_stack + btc_bought
        if total_btc > 0:
            state.btc_cost_basis = (
                (config.start_stack * btc_price + config.scf_amount) / total_btc
            )
    return state
```

- [ ] **Step 3: Run tests, verify pass**

- [ ] **Step 4: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_citadel.py
git commit -m "feat(citadel): implement step function with lognormal returns"
```

---

### Task 7: Engine — Saylor Citadel Fortifier

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Modify: `btc_web/test_citadel.py`

- [ ] **Step 1: Write failing tests for SCF**

```python
class TestSaylorFortifier:
    def test_scf_init_buys_btc(self):
        cfg = SimConfig.default()
        cfg.scf_enabled = True
        cfg.scf_amount = 100000
        cfg.scf_type = "term"
        cfg.scf_rate = 8.0
        cfg.scf_term = 60
        s = _initial_state(cfg)
        # SCF should have bought BTC with the loan proceeds
        assert s.scf_active
        assert s.scf_outstanding == 100000
        assert s.btc_stack > cfg.start_stack  # got extra BTC

    def test_term_loan_monthly_payment(self):
        cfg = SimConfig.default()
        cfg.scf_enabled = True
        cfg.scf_amount = 120000
        cfg.scf_rate = 12.0  # 12% annual = 1% monthly
        cfg.scf_type = "term"
        cfg.scf_term = 12
        ppy = 12
        pmt = _scf_payment_amount(cfg, ppy)
        # Amortizing: ~$10,661/mo for $120K at 1%/mo for 12 months
        assert 10000 < pmt < 11000

    def test_perpetual_interest_only(self):
        cfg = SimConfig.default()
        cfg.scf_enabled = True
        cfg.scf_amount = 100000
        cfg.scf_rate = 12.0  # 1% monthly
        cfg.scf_type = "perpetual"
        ppy = 12
        pmt = _scf_payment_amount(cfg, ppy)
        # Interest only: 100000 * 0.12 / 12 = $1000/mo
        assert abs(pmt - 1000) < 1

    def test_perpetual_repay_trigger(self):
        cfg = SimConfig.default()
        cfg.scf_enabled = True
        cfg.scf_type = "perpetual"
        cfg.scf_rate = 8.0
        cfg.scf_repay_trigger = 1.0  # repay when BTC return <= 8%
        s = CitadelState()
        s.scf_active = True
        s.scf_outstanding = 100000
        s.btc_stack = 5.0
        s.btc_price = 50000
        # BTC return = 5% (below 8% threshold) → trigger fires
        _scf_check_repay(s, cfg, btc_annual_return=0.05)
        assert s.scf_outstanding == 0  # repaid
        assert s.btc_stack < 5.0  # sold BTC to repay

    def test_scf_disabled_no_effect(self):
        cfg = SimConfig.default()
        cfg.scf_enabled = False
        s = _initial_state(cfg)
        assert not s.scf_active
        assert s.scf_outstanding == 0
```

- [ ] **Step 2: Implement SCF functions**

```python
# Add to btc_web/engines/citadel.py
import math


def _scf_payment_amount(config: SimConfig, ppy: int) -> float:
    """Calculate monthly loan payment. Returns monthly $ amount."""
    if not config.scf_enabled or config.scf_amount <= 0:
        return 0.0
    monthly_rate = (config.scf_rate / 100) / 12

    if config.scf_type == "perpetual":
        # Interest only
        return config.scf_amount * monthly_rate

    # Term loan: amortizing
    n = config.scf_term  # total months
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
        # Sell BTC to repay
        if state.btc_price > 0 and state.btc_stack > 0:
            btc_needed = state.scf_outstanding / state.btc_price
            btc_sold = min(state.btc_stack, btc_needed)
            repaid = btc_sold * state.btc_price
            state.btc_stack -= btc_sold
            state.scf_outstanding -= repaid
        if state.scf_outstanding <= 0.01:  # float tolerance
            state.scf_outstanding = 0
            state.scf_active = False
```

- [ ] **Step 3: Run tests, verify pass**

- [ ] **Step 4: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_citadel.py
git commit -m "feat(citadel): implement Saylor Citadel Fortifier loan mechanics"
```

---

### Task 8: Engine — simulate() runner and SimResult

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Modify: `btc_web/test_citadel.py`

- [ ] **Step 1: Write failing tests**

```python
class TestSimulate:
    def test_single_sim_returns_result(self):
        cfg = SimConfig.default()
        cfg.n_sims = 1
        cfg.start_yr = 2031
        cfg.end_yr = 2035  # short for speed
        result = simulate(cfg, _mock_model_data())
        assert result.time_axis is not None
        assert len(result.time_axis) == 48  # 4 years * 12 months
        assert result.btc_holdings.shape == (1, 48)
        assert result.total_usd.shape == (1, 48)
        assert result.depletion_period[0] is None or isinstance(result.depletion_period[0], int)

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
```

- [ ] **Step 2: Implement `simulate()`, `SimResult.to_dict()`, `SimResult.from_dict()`, `_aggregate_results()`**

The `simulate()` function:
1. Validates config
2. Pre-computes spending schedule
3. For each sim: initializes state, loops through periods calling `step()`
4. Aggregates all sim histories into SimResult with median/percentile bands

For BTC prices in deterministic mode (n_sims=1), use the quantile regression model directly: `model.price_at(q, t)`. For the test, create a `_mock_model_data()` helper that returns a simple model object.

- [ ] **Step 3: Run tests, verify pass**

- [ ] **Step 4: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_citadel.py
git commit -m "feat(citadel): implement simulate() runner and SimResult serialization"
```

---

### Task 9: Engine — Adapter interface

**Files:**
- Create: `btc_web/engines/adapter.py`
- Modify: `btc_web/test_citadel.py`

- [ ] **Step 1: Write test**

```python
class TestAdapter:
    def test_submit_returns_result(self):
        from engines.adapter import submit_simulation
        cfg = SimConfig.default()
        cfg.start_yr = 2031
        cfg.end_yr = 2033
        result = submit_simulation(cfg, _mock_model_data())
        assert result.time_axis is not None
```

- [ ] **Step 2: Implement adapter**

```python
# btc_web/engines/adapter.py
"""Simulation submission adapter. v1: in-process. v2: Celery task."""
from engines.citadel import SimConfig, SimResult, simulate


def submit_simulation(config: SimConfig, model_data,
                      rng_seed: int = 42) -> SimResult:
    """Run simulation in-process. Returns SimResult directly.
    v2: replace with celery_app.send_task() returning job_id."""
    return simulate(config, model_data, rng_seed=rng_seed)
```

- [ ] **Step 3: Run test, verify pass, commit**

```bash
git add btc_web/engines/adapter.py btc_web/test_citadel.py
git commit -m "feat(citadel): add simulation adapter interface (in-process v1)"
```

---

### Task 10: Layout — Tab 9 with tabbed sub-panels

**Files:**
- Create: `btc_web/layout/citadel.py`

- [ ] **Step 1: Implement the full layout module**

Create `btc_web/layout/citadel.py` with:
- `_citadel_assets_panel()` — BTC stack, cash, reserves grid, investments grid
- `_citadel_spending_panel()` — monthly spend, inflation, growth
- `_citadel_rules_panel()` — high/low Q triggers, floor rules, SCF section
- `_citadel_sim_panel()` — year range, freq, quantiles, MC, chart toggles
- `_citadel_controls()` — wraps above in inner `dbc.Tabs`
- `_citadel_tab()` — wraps in `_chart_tab_layout`

All component IDs use `cp-` prefix. Use `_section_card`, `_lbl`, `dbc.Input`, `dcc.Checklist`, `dcc.RadioItems` from existing patterns.

For the SCF section, use `html.Div(style={"display":"none"})` toggled by callback — NOT `dbc.Collapse`.

- [ ] **Step 2: Verify it imports without error**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -c "from layout.citadel import _citadel_tab; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add btc_web/layout/citadel.py
git commit -m "feat(citadel): add Tab 9 layout with tabbed sub-panels"
```

---

### Task 11: Figure builder — multi-line portfolio chart

**Files:**
- Create: `btc_web/figures/citadel.py`

- [ ] **Step 1: Implement `build_citadel_figure(m, p)`**

Follow the `figures/retire.py` pattern. Build traces for:
- Total Portfolio (white, thick)
- BTC Holdings in USD (orange)
- Cash (silver, dashed)
- Reserve Fund total (blue)
- Investments total (green)
- Monthly Spending level (red, dotted)

Include depletion annotation, endpoint annotations, and MC overlay integration point.

Return type: `tuple[go.Figure, dict | None]`

- [ ] **Step 2: Write test**

```python
class TestCitadelFigure:
    def test_builds_with_defaults(self):
        from figures.citadel import build_citadel_figure
        import _app_ctx
        m = _app_ctx.M
        p = {... default params ...}
        fig, mc = build_citadel_figure(m, p)
        assert len(fig.data) >= 5  # at least 5 asset lines
        assert mc is None
```

- [ ] **Step 3: Run test, verify pass, commit**

```bash
git add btc_web/figures/citadel.py btc_web/test_citadel.py
git commit -m "feat(citadel): add multi-line portfolio chart builder"
```

---

### Task 12: Cache and utils integration

**Files:**
- Modify: `btc_web/utils.py`

- [ ] **Step 1: Add cached figure builder**

Add to `utils.py`:
```python
from figures.citadel import build_citadel_figure
_cached_citadel_fig = _make_cached_builder(build_citadel_figure)
# Add to _ALL_CACHES dict
def _get_citadel_fig(p: dict):
    return _get_mc_or_cached(p, build_citadel_figure, _cached_citadel_fig)
```

- [ ] **Step 2: Commit**

```bash
git add btc_web/utils.py
git commit -m "feat(citadel): register LRU figure cache"
```

---

### Task 13: Callback — update_citadel

**Files:**
- Create: `btc_web/callbacks/citadel_cb.py`

- [ ] **Step 1: Implement `update_citadel` callback**

Follow `update_retire` pattern exactly:
- `@callback` with Output for graph + MC stores
- `Input` for all `cp-*` controls + MC controls + `main-tabs`
- Tab guard: `if ctx.triggered_id == "main-tabs" and active_tab != "citadel": PreventUpdate`
- Assemble params dict from inputs, call `_get_citadel_fig(params)`
- MC finalization via `_mc_finalize("cp", ...)`

- [ ] **Step 2: Add callback smoke test**

```python
class TestCitadelCallback:
    def test_returns_figure(self):
        # Same pattern as existing TestUpdateRetireCallback
        ...
```

- [ ] **Step 3: Commit**

```bash
git add btc_web/callbacks/citadel_cb.py btc_web/test_citadel.py
git commit -m "feat(citadel): add update_citadel callback with MC integration"
```

---

### Task 14: Tab registration and routing

**Files:**
- Modify: `btc_web/layout/__init__.py`
- Modify: `btc_web/callbacks/__init__.py`
- Modify: `btc_web/callbacks/nav.py`
- Modify: `btc_web/app.py`

- [ ] **Step 1: Register tab in layout**

In `layout/__init__.py`:
- Add import: `from layout.citadel import _citadel_tab`
- Add MC stores for "cp" prefix (add "cp" to the store generation loops)
- Add `dbc.Tab(_citadel_tab(), label="\U0001f3f0 Citadel Planner", tab_id="citadel")` to `dbc.Tabs`

- [ ] **Step 2: Register callback imports**

In `callbacks/__init__.py`:
- Add: `import callbacks.citadel_cb`

- [ ] **Step 3: Add routing**

In `callbacks/nav.py`:
- Add `"/9": "citadel"` to `_PATH_TO_TAB`
- Add `_TAB_CONTROLS["citadel"]` with all `cp-*` component IDs

- [ ] **Step 4: Add prewarm and cache-control path**

In `app.py`:
- Add `/9` to the cache-control path check
- Add `_get_citadel_fig({...default params...})` to `_prewarm_caches()`

- [ ] **Step 5: Verify app starts**

Run: `DEV=1 bash run_web.sh` — verify no import errors, Tab 9 visible, chart renders.

- [ ] **Step 6: Commit**

```bash
git add btc_web/layout/__init__.py btc_web/callbacks/__init__.py \
    btc_web/callbacks/nav.py btc_web/app.py
git commit -m "feat(citadel): register Tab 9 — routing, stores, prewarm"
```

---

### Task 15: Snapshot integration

**Files:**
- Modify: `btc_web/snapshot.py`

- [ ] **Step 1: Append ~62 entries to `_SNAPSHOT_CONTROLS`**

Add all `cp-*` component IDs as `(component_id, "value")` tuples after the current last entry.

- [ ] **Step 2: Add `_CHECKLIST_OPTIONS` entries**

```python
"cp-qs":         _QS_LIST,
"cp-toggles":    ["log_y", "annotate", "show_legend", "minor_grid", "chart_zoom"],
"cp-use-lots":   ["yes"],
"cp-scf-enable": ["yes"],
```

- [ ] **Step 3: Add MC snapshot controls**

Add `cp-mc-enable`, `cp-mc-start-yr`, `cp-mc-entry-q`, `cp-mc-years`, `cp-mc-bins`, `cp-mc-regime`, `cp-mc-sims`, `cp-mc-window`, `cp-mc-advanced` to both `_SNAPSHOT_CONTROLS` and `_CHECKLIST_OPTIONS`.

- [ ] **Step 4: Write snapshot roundtrip test**

```python
class TestCitadelSnapshot:
    def test_roundtrip(self):
        state = {"cp-stack:value": 2.5, "cp-spend:value": 8000,
                 "cp-qs:value": [0.01, 0.50]}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["cp-stack:value"] == 2.5
        assert decoded["cp-spend:value"] == 8000
```

- [ ] **Step 5: Run full test suite**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_citadel.py btc_web/test_web.py -q --tb=short`
Verify: no new failures from snapshot integration. The `TestNoDuplicateCallbackOutputs` test should still pass.

- [ ] **Step 6: Commit**

```bash
git add btc_web/snapshot.py btc_web/test_citadel.py
git commit -m "feat(citadel): add ~62 snapshot controls + MC encoding"
```

---

### Task 16: SCF visibility toggle callback

**Files:**
- Modify: `btc_web/callbacks/citadel_cb.py`

- [ ] **Step 1: Add clientside callback to toggle SCF body visibility**

```python
_app_ctx.app.clientside_callback(
    "function(v) { return (v && v.length) ? {} : {display:'none'}; }",
    Output("cp-scf-body", "style"),
    Input("cp-scf-enable", "value"),
)
```

Also add callbacks to toggle conditional fields (duration/cooldown based on mode, term/trigger based on SCF type).

- [ ] **Step 2: Commit**

```bash
git add btc_web/callbacks/citadel_cb.py
git commit -m "feat(citadel): add SCF and conditional field visibility toggles"
```

---

### Task 17: Integration test — full app startup

**Files:**
- Modify: `btc_web/test_citadel.py`

- [ ] **Step 1: Write integration tests**

```python
class TestCitadelIntegration:
    def test_app_imports_cleanly(self):
        """Verify the full import chain works."""
        import os; os.environ['TESTING'] = '1'
        from btc_web.app import app
        assert app is not None

    def test_tab_registered(self):
        from callbacks.nav import _PATH_TO_TAB, _TAB_CONTROLS
        assert "/9" in _PATH_TO_TAB
        assert _PATH_TO_TAB["/9"] == "citadel"
        assert "citadel" in _TAB_CONTROLS

    def test_no_duplicate_outputs(self):
        """Reuses existing TestNoDuplicateCallbackOutputs pattern."""
        # Already covered by test_web.py — verify it still passes
        pass
```

- [ ] **Step 2: Run full test suite**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_citadel.py btc_web/test_web.py -q --tb=short -k "not test_usd_mode_equals_btc_times_price"`

- [ ] **Step 3: Commit**

```bash
git add btc_web/test_citadel.py
git commit -m "test(citadel): add integration tests for full app startup"
```

---

### Task 18: Final verification and push

- [ ] **Step 1: Start dev server and manually verify**

```bash
DEV=1 bash run_web.sh
```

Open `http://localhost:8050/9` — verify:
- Tab 9 loads with all 4 sub-panels
- Default chart renders with multi-line output
- Sub-tab switching works
- SCF toggle shows/hides controls
- Share link encodes/decodes Tab 9 state

- [ ] **Step 2: Run complete test suite**

```bash
PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_citadel.py btc_web/test_web.py -v --tb=short
```

- [ ] **Step 3: Push to Plan9 branch**

```bash
git push origin Plan9
```

- [ ] **Step 4: Deploy Plan9 branch to production (no merge)**

```bash
ssh root@89.167.70.45 "cd /opt/quantoshi && git fetch origin && git checkout Plan9 && systemctl restart quantoshi"
```
