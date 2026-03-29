# Tax Accounting Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 7 tax accounting gaps (untracked BTC sales/purchases, missing investment gain recording), merge the two spending waterfalls, and always lot-track BTC.

**Architecture:** Add `sim_date` to `CitadelState`, always seed lots in `_initial_state`. Create 3 tracking helpers (`_sell_btc_tracked`, `_buy_btc_tracked`, `_sell_investments_tracked`) that centralize lot/basis/accumulator logic. Replace raw mutations at 10 call sites with helper calls. Delete `_apply_spending_waterfall`, merge into unified `_spending_waterfall`.

**Tech Stack:** Python 3.14, dataclasses, numpy

**Spec:** `docs/superpowers/specs/2026-03-29-tax-accounting-fixes-design.md`

**Test command:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short`

---

### Task 1: Add `sim_date` to CitadelState + always seed lots

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write tests**

Add to `TestQuarterlyTaxPayments` class (end of test_web.py) or create a new class `TestTaxAccountingHelpers`:

```python
class TestTaxAccountingHelpers:
    """Tests for the 3 tracking helpers and related infrastructure."""

    def test_state_has_sim_date(self):
        from engines.citadel import CitadelState
        s = CitadelState()
        assert hasattr(s, "sim_date")
        assert s.sim_date == ""

    def test_lots_seeded_when_tax_off(self):
        """Lots should be created even when tax_enabled=False."""
        from engines.citadel import SimConfig, _initial_state
        cfg = SimConfig(start_stack=2.0, start_yr=2031, end_yr=2035,
                        cash_initial=100_000, selected_qs=[0.25],
                        tax_enabled=False)
        state = _initial_state(cfg, model=_test_model())
        assert len(state.tax_lots) == 1
        assert state.tax_lots[0].btc == 2.0
        assert state.tax_year_accum is None  # NOT created when tax off

    def test_scf_purchase_creates_lot(self):
        """SCF initial BTC purchase must create a separate lot."""
        from engines.citadel import SimConfig, _initial_state
        cfg = SimConfig(start_stack=1.0, start_yr=2031, end_yr=2035,
                        cash_initial=100_000, selected_qs=[0.25],
                        tax_enabled=False,
                        scf_enabled=True, scf_amount=50_000)
        state = _initial_state(cfg, model=_test_model())
        assert len(state.tax_lots) == 2  # start_stack + SCF
        assert state.tax_lots[0].source == "initial"
        assert state.tax_lots[1].source == "scf"
        total_lot_btc = sum(l.btc for l in state.tax_lots)
        assert abs(total_lot_btc - state.btc_stack) < 1e-8
```

- [ ] **Step 2: Run tests — expect failures**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestTaxAccountingHelpers -v
```

Expected: `test_state_has_sim_date` fails (no `sim_date` field), `test_lots_seeded_when_tax_off` fails (lots empty), `test_scf_purchase_creates_lot` fails.

- [ ] **Step 3: Add `sim_date` to CitadelState**

In `btc_web/engines/citadel.py`, add after `quarterly_tax_paid_ytd` (line ~183):

```python
    sim_date: str = ""                        # ISO date string, set each period in step()
```

- [ ] **Step 4: Move lot seeding outside `if config.tax_enabled:` block + add SCF lot**

In `_initial_state()`, restructure the lot seeding. Replace the current `if config.tax_enabled:` block (lines ~628-654) with:

```python
    # Always seed lots (even when tax_enabled=False) — forward-compatible
    from .tax_lots import seed_lots, TaxLot
    start_date = f"{config.start_yr}-01-01"
    state.tax_lots = seed_lots(
        [], start_stack=config.start_stack, start_price=btc_price,
        start_date=start_date,
    )
    # SCF initial purchase gets its own lot
    if config.scf_enabled and config.scf_amount > 0 and btc_price > 0:
        scf_btc = config.scf_amount / btc_price
        state.tax_lots.append(TaxLot(
            date=start_date, btc=scf_btc,
            cost_basis=btc_price, source="scf",
        ))

    # Tax-specific initialization (accumulator, wrappers)
    if config.tax_enabled:
        from .tax import TaxYearAccumulator

        # TD/TF wrapper balances from config
        state.td_btc_stack = config.td_btc_stack
        state.td_cash = config.td_cash_initial
        state.td_reserves = [rb["initial"] for rb in config.td_reserve_bins]
        state.td_investments = [ib["initial"] for ib in config.td_invest_bins]

        state.tf_btc_stack = config.tf_btc_stack
        state.tf_cash = config.tf_cash_initial
        state.tf_reserves = [rb["initial"] for rb in config.tf_reserve_bins]
        state.tf_investments = [ib["initial"] for ib in config.tf_invest_bins]

        state.tax_year_accum = TaxYearAccumulator()
        state.loss_carryforward = 0.0
        state.total_taxes_paid = 0.0
```

Note: the `from .tax_lots import seed_lots, TaxLot` moves outside the `if config.tax_enabled:` block. The `from .tax import TaxYearAccumulator` stays inside it.

- [ ] **Step 5: Set `sim_date` in `step()`**

In the `step()` function, after the time advance (`new.period += 1`, around line ~1175) and before any function that might need `sim_date`, add:

```python
    # Set sim_date for this period (used by tracking helpers)
    years_elapsed_for_date = new.period / ppy
    _sim_yr = config.start_yr + int(years_elapsed_for_date)
    _sim_mo = min(max(1, int((years_elapsed_for_date % 1) * 12) + 1), 12)
    new.sim_date = f"{_sim_yr}-{_sim_mo:02d}-15"
```

This should go early in `step()`, before the spending waterfall call.

- [ ] **Step 6: Run tests**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestTaxAccountingHelpers -v
```

Expected: All 3 PASS.

- [ ] **Step 7: Run full suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -10
```

Expected: All tests PASS (some existing tests may need `_test_model()` adjustments if `_initial_state` import changed — investigate any failures).

- [ ] **Step 8: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "feat(citadel): add sim_date to state, always seed lots incl. SCF"
```

---

### Task 2: Create 3 tracking helpers + unit tests

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write tests for `_sell_btc_tracked`**

Add to `TestTaxAccountingHelpers`:

```python
    def test_sell_btc_tracked_records_gains_tax_on(self):
        """With tax on, selling BTC records capital gains in accumulator."""
        from engines.citadel import CitadelState, SimConfig, _sell_btc_tracked
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=2.0, btc_price=100_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=2.0,
                             cost_basis=50_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, cost_basis_method="fifo")
        result = _sell_btc_tracked(state, cfg, 1.0)
        assert result.btc_sold == pytest.approx(1.0)
        assert state.btc_stack == pytest.approx(1.0)
        assert len(state.tax_lots) == 1
        assert state.tax_lots[0].btc == pytest.approx(1.0)
        # Gain: sold at 100k, basis 50k = 50k gain, long-term (>365 days)
        assert state.tax_year_accum.lt_capital_gains == pytest.approx(50_000)

    def test_sell_btc_tracked_no_gains_tax_off(self):
        """With tax off (accum=None), BTC still sold but no gains recorded."""
        from engines.citadel import CitadelState, SimConfig, _sell_btc_tracked
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=2.0, btc_price=100_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=2.0,
                             cost_basis=50_000, source="initial")],
            tax_year_accum=None,  # tax off
        )
        cfg = SimConfig(tax_enabled=False, cost_basis_method="fifo")
        result = _sell_btc_tracked(state, cfg, 1.0)
        assert result.btc_sold == pytest.approx(1.0)
        assert state.btc_stack == pytest.approx(1.0)

    def test_sell_btc_tracked_empty_lots_fallback(self):
        """With no lots, raw stack decrement as fallback."""
        from engines.citadel import CitadelState, SimConfig, _sell_btc_tracked
        state = CitadelState(btc_stack=3.0, btc_price=50_000, sim_date="2035-01-15")
        cfg = SimConfig(cost_basis_method="fifo")
        result = _sell_btc_tracked(state, cfg, 1.0)
        assert result.btc_sold == pytest.approx(1.0)
        assert state.btc_stack == pytest.approx(2.0)
```

- [ ] **Step 2: Write tests for `_buy_btc_tracked`**

```python
    def test_buy_btc_tracked_creates_lot(self):
        """Buying BTC creates a lot with correct date/basis/source."""
        from engines.citadel import CitadelState, SimConfig, _buy_btc_tracked
        state = CitadelState(btc_stack=1.0, btc_price=80_000, sim_date="2033-03-15")
        cfg = SimConfig()
        _buy_btc_tracked(state, cfg, 0.5, source="rebal_buy")
        assert state.btc_stack == pytest.approx(1.5)
        assert len(state.tax_lots) == 1
        lot = state.tax_lots[0]
        assert lot.btc == pytest.approx(0.5)
        assert lot.cost_basis == 80_000
        assert lot.date == "2033-03-15"
        assert lot.source == "rebal_buy"
```

- [ ] **Step 3: Write tests for `_sell_investments_tracked`**

```python
    def test_sell_investments_tracked_records_ltcg(self):
        """Investment sale records LTCG in accumulator."""
        from engines.citadel import CitadelState, SimConfig, _sell_investments_tracked
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            investments=[200_000, 100_000],
            invest_cost_basis=[100_000, 80_000],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True)
        drawn, gain = _sell_investments_tracked(state, cfg, 0, 50_000)
        assert drawn == pytest.approx(50_000)
        # Equities: basis fraction = 100k/200k = 50%, gain = 50k - 25k = 25k
        assert gain == pytest.approx(25_000)
        assert state.tax_year_accum.lt_capital_gains == pytest.approx(25_000)
        assert state.investments[0] == pytest.approx(150_000)
        assert state.invest_cost_basis[0] == pytest.approx(75_000)

    def test_sell_investments_tracked_noop_tax_off(self):
        """Investment sale updates balances but skips accumulator when tax off."""
        from engines.citadel import CitadelState, SimConfig, _sell_investments_tracked
        state = CitadelState(
            investments=[200_000, 0],
            invest_cost_basis=[100_000, 0],
            tax_year_accum=None,
        )
        cfg = SimConfig(tax_enabled=False)
        drawn, gain = _sell_investments_tracked(state, cfg, 0, 50_000)
        assert drawn == pytest.approx(50_000)
        assert state.investments[0] == pytest.approx(150_000)
```

- [ ] **Step 4: Run tests — expect import failures**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestTaxAccountingHelpers -v
```

Expected: FAIL with `ImportError: cannot import name '_sell_btc_tracked'`

- [ ] **Step 5: Implement `_sell_btc_tracked`**

Add to `btc_web/engines/citadel.py`, after the `_SATOSHI` constant and before `_apply_spending_waterfall`. Import `SaleResult` at the top of the function:

```python
def _sell_btc_tracked(state: CitadelState, config: SimConfig,
                      btc_to_sell: float) -> "SaleResult":
    """Sell BTC with lot tracking + accumulator update. Returns SaleResult.

    If lots exist, uses sell_lots() for proper cost basis tracking.
    If lots are empty (defensive), does raw stack decrement.
    Accumulator gains are only recorded when state.tax_year_accum is not None.
    """
    from .tax_lots import sell_lots, SaleResult, LotGain

    if state.tax_lots:
        result = sell_lots(
            state.tax_lots, btc_to_sell, state.btc_price,
            state.sim_date, method=config.cost_basis_method,
        )
        state.btc_stack -= result.btc_sold
        state.tax_lots = result.remaining_lots
        if state.tax_year_accum is not None:
            for g in result.gains:
                if g.is_long_term:
                    if g.gain >= 0:
                        state.tax_year_accum.lt_capital_gains += g.gain
                    else:
                        state.tax_year_accum.lt_capital_losses += abs(g.gain)
                else:
                    if g.gain >= 0:
                        state.tax_year_accum.st_capital_gains += g.gain
                    else:
                        state.tax_year_accum.st_capital_losses += abs(g.gain)
        return result
    else:
        # Defensive: no lots available — raw decrement
        btc_sold = min(btc_to_sell, state.btc_stack)
        state.btc_stack -= btc_sold
        return SaleResult(btc_sold=btc_sold, gains=[], remaining_lots=[])
```

- [ ] **Step 6: Implement `_buy_btc_tracked`**

```python
def _buy_btc_tracked(state: CitadelState, config: SimConfig,
                     btc_bought: float, source: str = "rebal_buy") -> None:
    """Buy BTC and create a tax lot for cost basis tracking."""
    from .tax_lots import TaxLot
    state.btc_stack += btc_bought
    state.tax_lots.append(TaxLot(
        date=state.sim_date, btc=btc_bought,
        cost_basis=state.btc_price, source=source,
    ))
```

- [ ] **Step 7: Implement `_sell_investments_tracked`**

```python
def _sell_investments_tracked(state: CitadelState, config: SimConfig,
                              bin_index: int, amount: float) -> tuple[float, float]:
    """Sell from investment bin with cost basis tracking + accumulator update.

    Returns (amount_drawn, gain). Gain is positive for profit, negative for loss.
    Accumulator update only when state.tax_year_accum is not None.
    """
    current = state.investments[bin_index]
    if current <= 0 or amount <= 0:
        return (0.0, 0.0)
    draw = min(current, amount)
    fraction = draw / current
    basis_sold = state.invest_cost_basis[bin_index] * fraction
    state.invest_cost_basis[bin_index] -= basis_sold
    state.investments[bin_index] -= draw
    gain = draw - basis_sold
    if state.tax_year_accum is not None:
        if gain >= 0:
            state.tax_year_accum.lt_capital_gains += gain
        else:
            state.tax_year_accum.lt_capital_losses += abs(gain)
    return (draw, gain)
```

- [ ] **Step 8: Run tests**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestTaxAccountingHelpers -v
```

Expected: All PASS.

- [ ] **Step 9: Run full suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -10
```

Expected: All tests PASS.

- [ ] **Step 10: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "feat(citadel): add _sell_btc_tracked, _buy_btc_tracked, _sell_investments_tracked helpers"
```

---

### Task 3: Update `_enforce_floors` call sites (bugs 1 + 7)

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write regression tests**

Add to `TestTaxAccountingHelpers`:

```python
    def test_floor_enforcement_btc_sale_lot_tracked(self):
        """Bug 1: BTC sold to replenish cash floor must be lot-tracked."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
            btc_stack=2.0, btc_price=50_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=2.0,
                             cost_basis=30_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
            invest_cost_basis=[0, 0],
        )
        cfg = SimConfig(cash_floor=20_000, cost_basis_method="fifo")
        _enforce_floors(state, cfg)
        assert state.cash >= 20_000 - 1
        assert state.btc_stack < 2.0
        # Capital gain should be recorded (sold at 50k, basis 30k)
        assert state.tax_year_accum.lt_capital_gains > 0

    def test_floor_enforcement_investment_sale_tracks_basis(self):
        """Bug 7: Investment sold for floor must update cost basis."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[100_000, 50_000],
            invest_cost_basis=[60_000, 30_000],
            btc_stack=0, btc_price=50_000, sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(cash_floor=30_000, cost_basis_method="fifo")
        _enforce_floors(state, cfg)
        assert state.cash >= 30_000 - 1
        # Cost basis should have been reduced proportionally
        assert state.invest_cost_basis[1] < 30_000 or state.invest_cost_basis[0] < 60_000
        # LTCG should be recorded
        assert state.tax_year_accum.lt_capital_gains > 0
```

- [ ] **Step 2: Run tests — expect failures**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestTaxAccountingHelpers::test_floor_enforcement_btc_sale_lot_tracked btc_web/test_web.py::TestTaxAccountingHelpers::test_floor_enforcement_investment_sale_tracks_basis -v
```

Expected: FAIL (gains not recorded because `_enforce_floors` doesn't use helpers yet).

- [ ] **Step 3: Update `_enforce_floors` investment sales**

In the investment source block (lines ~316-318), replace:

```python
            if src_type == "inv":
                draw = min(state.investments[src_idx], deficit)
                state.investments[src_idx] -= draw
```

With:

```python
            if src_type == "inv":
                draw_want = min(state.investments[src_idx], deficit)
                draw, _gain = _sell_investments_tracked(state, config, src_idx, draw_want)
```

- [ ] **Step 4: Update `_enforce_floors` BTC sale**

In the BTC last resort block (lines ~329-336), replace:

```python
        if deficit > 0 and acct_key == "cash":
            if state.btc_stack > 0 and state.btc_price > 0:
                btc_needed = deficit / state.btc_price
                btc_sold = min(state.btc_stack, btc_needed)
                draw = btc_sold * state.btc_price
                state.btc_stack -= btc_sold
                deficit -= draw
```

With:

```python
        if deficit > 0 and acct_key == "cash":
            if state.btc_stack > 0 and state.btc_price > 0:
                btc_needed = deficit / state.btc_price
                result = _sell_btc_tracked(state, config, btc_needed)
                draw = result.btc_sold * state.btc_price
                deficit -= draw
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestTaxAccountingHelpers -v
```

Expected: All PASS.

- [ ] **Step 6: Run full suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -10
```

Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "fix(citadel): lot-track BTC sales + basis-track investment sales in _enforce_floors"
```

---

### Task 4: Update rebalancing + SCF call sites (bugs 2, 3, 5, 6)

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write regression tests**

Add to `TestTaxAccountingHelpers`:

```python
    def test_rebalancing_sell_btc_lot_tracked(self):
        """Bug 2: Rebalancing BTC sell must be lot-tracked."""
        from engines.citadel import CitadelState, SimConfig, _execute_sell_btc
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=10.0, btc_price=50_000, sim_date="2035-06-15",
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
            tax_lots=[TaxLot(date="2031-01-01", btc=10.0,
                             cost_basis=20_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(cost_basis_method="fifo")
        evt = _execute_sell_btc(state, cfg, rate_pct=10.0, split={"cash": 1.0})
        assert evt["btc_sold"] == pytest.approx(1.0)
        assert state.btc_stack == pytest.approx(9.0)
        # Gain: 1 BTC sold at 50k, basis 20k = 30k gain
        assert state.tax_year_accum.lt_capital_gains == pytest.approx(30_000)

    def test_rebalancing_buy_btc_creates_lot(self):
        """Bug 3: Rebalancing BTC buy must create a tax lot."""
        from engines.citadel import CitadelState, SimConfig, _execute_buy_btc
        state = CitadelState(
            btc_stack=1.0, btc_price=50_000, sim_date="2033-03-15",
            cash=100_000, reserves=[0, 0, 0], investments=[0, 0],
        )
        cfg = SimConfig(cash_floor=0)
        evt = _execute_buy_btc(state, cfg, rate_pct=10.0, split={"cash": 1.0})
        assert evt["action"] == "buy_btc"
        assert state.btc_stack > 1.0
        # Should have a lot for the purchased BTC
        new_lots = [l for l in state.tax_lots if l.source == "rebal_buy"]
        assert len(new_lots) == 1
        assert new_lots[0].cost_basis == 50_000

    def test_scf_repay_btc_sale_lot_tracked(self):
        """Bug 6: SCF perpetual loan repayment must lot-track BTC sale."""
        from engines.citadel import CitadelState, SimConfig, _scf_check_repay
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=5.0, btc_price=50_000, sim_date="2040-01-15",
            scf_outstanding=100_000, scf_active=True,
            tax_lots=[TaxLot(date="2031-01-01", btc=5.0,
                             cost_basis=30_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(scf_enabled=True, scf_type="perpetual",
                        scf_rate=8.0, scf_repay_trigger=1.0,
                        cost_basis_method="fifo")
        # btc_annual_return = 0 → below threshold → triggers repayment
        _scf_check_repay(state, cfg, btc_annual_return=0.0)
        assert state.btc_stack < 5.0
        # Gain should be recorded
        assert state.tax_year_accum.lt_capital_gains > 0

    def test_lot_inventory_matches_stack_after_operations(self):
        """Lot sum must match btc_stack after sell/buy operations."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _sell_btc_tracked, _buy_btc_tracked)
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=5.0, btc_price=60_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=5.0,
                             cost_basis=30_000, source="initial")],
        )
        cfg = SimConfig(cost_basis_method="fifo")
        _sell_btc_tracked(state, cfg, 2.0)
        _buy_btc_tracked(state, cfg, 1.0, source="rebal_buy")
        lot_sum = sum(l.btc for l in state.tax_lots)
        assert abs(lot_sum - state.btc_stack) < 1e-8, \
            f"Lot sum {lot_sum} != btc_stack {state.btc_stack}"
```

- [ ] **Step 2: Run tests — expect failures for bugs 2, 3, 6**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestTaxAccountingHelpers -v -k "rebalancing or scf_repay or lot_inventory"
```

- [ ] **Step 3: Update `_execute_sell_btc` — add `config` parameter, use `_sell_btc_tracked`**

Replace the function (lines ~429-437):

```python
def _execute_sell_btc(state: CitadelState, config: SimConfig,
                      rate_pct: float, split: dict) -> dict:
    """Sell rate_pct% of BTC stack, distribute proceeds via split."""
    btc_to_sell = state.btc_stack * (rate_pct / 100.0)
    if btc_to_sell <= 0 or state.btc_price <= 0:
        return {}
    result = _sell_btc_tracked(state, config, btc_to_sell)
    if result.btc_sold <= 0:
        return {}
    proceeds = result.btc_sold * state.btc_price
    _distribute_to_accounts(state, proceeds, split)
    return {"action": "sell_btc", "btc_sold": result.btc_sold, "proceeds": proceeds}
```

- [ ] **Step 4: Update `_execute_buy_btc` — make `config` required, use `_buy_btc_tracked`**

Replace the function (lines ~439-452):

```python
def _execute_buy_btc(state: CitadelState, config: SimConfig,
                     rate_pct: float, split: dict) -> dict:
    """Source funds from accounts via split, buy BTC.
    Respects floor rules — won't draw accounts below floors."""
    total_dollar = state.cash + sum(state.reserves) + sum(state.investments)
    target = total_dollar * (rate_pct / 100.0)
    if target <= 0 or state.btc_price <= 0:
        return {}
    sourced = _source_from_accounts(state, target, split, config=config)
    if sourced <= 0:
        return {}
    btc_bought = sourced / state.btc_price
    _buy_btc_tracked(state, config, btc_bought, source="rebal_buy")
    return {"action": "buy_btc", "btc_bought": btc_bought, "cost": sourced}
```

- [ ] **Step 5: Update all `_execute_sell_btc` callers in `_evaluate_rebalancing`**

Every call to `_execute_sell_btc(state, ...)` needs `config` as the second arg. Find and replace all 3 occurrences:

Line ~464: `_execute_sell_btc(state, state.grad_rate, state.grad_split)` → `_execute_sell_btc(state, config, state.grad_rate, state.grad_split)`

Line ~479: `_execute_sell_btc(state, action["rate"], split)` → `_execute_sell_btc(state, config, action["rate"], split)`

Line ~490: `_execute_sell_btc(state, state.grad_rate, split)` → `_execute_sell_btc(state, config, state.grad_rate, split)`

- [ ] **Step 6: Update `_scf_check_repay` to use `_sell_btc_tracked`**

Replace the BTC sale block (lines ~587-592):

```python
        if state.btc_price > 0 and state.btc_stack > 0:
            btc_needed = state.scf_outstanding / state.btc_price
            result = _sell_btc_tracked(state, config, btc_needed)
            repaid = result.btc_sold * state.btc_price
            state.scf_outstanding -= repaid
```

The function signature needs `config` added. Change from:
```python
def _scf_check_repay(state: CitadelState, config: SimConfig,
                     btc_annual_return: float) -> None:
```
This already has `config` — good, no signature change needed.

- [ ] **Step 7: Run tests**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestTaxAccountingHelpers -v
```

Expected: All PASS.

- [ ] **Step 8: Run full suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -10
```

Expected: All PASS. Existing rebalancing tests (TestBtcThresholdRules, TestBtcSaleDistribution, etc.) should still pass with the new signatures.

- [ ] **Step 9: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "fix(citadel): lot-track rebalancing + SCF BTC sales/purchases"
```

---

### Task 5: Update `_pay_tax_amount` + merge waterfalls (bugs 4 + waterfall merge)

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write regression test for bug 4**

Add to `TestTaxAccountingHelpers`:

```python
    def test_pay_tax_investment_sale_recorded(self):
        """Bug 4: Investment gains during tax payment must be in accumulator."""
        from engines.citadel import CitadelState, SimConfig, _pay_tax_amount
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[200_000, 0],
            invest_cost_basis=[100_000, 0],  # 50% unrealized gain
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            tax_year_accum=TaxYearAccumulator(),
            sim_date="2035-06-15",
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX")
        _pay_tax_amount(state, cfg, amount=50_000, sim_year=2035)
        # Investment sale to pay tax should record the LTCG
        assert state.tax_year_accum.lt_capital_gains > 0
```

- [ ] **Step 2: Write waterfall merge test**

```python
    def test_merged_waterfall_tax_off_same_behavior(self):
        """Merged waterfall with tax_enabled=False matches old non-tax behavior."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=1.0, start_yr=2031, end_yr=2035,
            freq="Annually", monthly_spend=5000,
            cash_initial=50_000, selected_qs=[0.25],
            tax_enabled=False,
            reserve_bins=[
                {"label": "Short", "initial": 20_000, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 20_000, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 50_000, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        r = simulate(cfg, _test_model())
        assert r.total_usd.shape[1] > 0
        assert r.taxes_paid is None  # tax off
        # Should not crash, should produce reasonable totals
        assert r.total_usd[0, -1] >= 0

    def test_tax_off_still_zero_tax(self):
        """Critical regression: tax_enabled=False must produce zero tax."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=5.0, start_yr=2031, end_yr=2035,
            freq="Monthly", monthly_spend=5000,
            cash_initial=100_000, selected_qs=[0.25],
            tax_enabled=False,
        )
        r = simulate(cfg, _test_model())
        assert r.taxes_paid is None
        assert r.annual_taxes is None
```

- [ ] **Step 3: Update `_pay_tax_amount` investment block to use `_sell_investments_tracked`**

In `_pay_tax_amount`, replace the investment sale loop (lines ~1020-1037). The gross-up logic stays but the actual sale goes through the helper. Replace:

```python
        for i in reversed(range(len(state.investments))):
            if tax_remaining <= 0:
                break
            current = state.investments[i]
            if current <= 0:
                continue
            basis_frac = state.invest_cost_basis[i] / current if current > 0 else 0
            gain_frac = 1.0 - basis_frac
            effective_rate = ltcg_rate * gain_frac
            gross = tax_remaining / max(1.0 - effective_rate, 0.1)
            draw = min(current, gross)
            fraction = draw / current
            basis_removed = state.invest_cost_basis[i] * fraction
            state.invest_cost_basis[i] -= basis_removed
            state.investments[i] -= draw
            net_received = draw
            tax_on_sale = (draw - basis_removed) * ltcg_rate
            tax_remaining -= (net_received - tax_on_sale)
```

With:

```python
        for i in reversed(range(len(state.investments))):
            if tax_remaining <= 0:
                break
            current = state.investments[i]
            if current <= 0:
                continue
            basis_frac = state.invest_cost_basis[i] / current if current > 0 else 0
            gain_frac = 1.0 - basis_frac
            effective_rate = ltcg_rate * gain_frac
            gross = tax_remaining / max(1.0 - effective_rate, 0.1)
            sell_amount = min(current, gross)
            drawn, gain = _sell_investments_tracked(state, config, i, sell_amount)
            tax_on_sale = max(gain, 0) * ltcg_rate
            tax_remaining -= (drawn - tax_on_sale)
```

- [ ] **Step 4: Rename `_tax_aware_waterfall` to `_spending_waterfall` and merge**

Rename the function:

```python
def _spending_waterfall(state: CitadelState, config: SimConfig,
                        amount: float,
                        model: "PriceModel | None" = None) -> float:
```

Remove the `sim_date` parameter — read from `state.sim_date` instead. Update the internal reference where `sim_date` was used (in the `_sell_taxable_btc` closure).

Wrap the TD/TF steps (steps 2, 5, 6, 7, 8) in `if config.tax_enabled:` guards.

Replace the inline `_sell_taxable_investments()` closure with calls to `_sell_investments_tracked`.

Replace the inline `_sell_taxable_btc()` closure with calls to `_sell_btc_tracked`.

- [ ] **Step 5: Delete `_apply_spending_waterfall`**

Remove the entire function (lines ~223-264).

- [ ] **Step 6: Update `step()` to use the merged waterfall**

Replace:

```python
    if config.tax_enabled:
        sim_year = config.start_yr + int(years_elapsed)
        sim_date = f"{sim_year}-{min(max(1, int((years_elapsed % 1) * 12) + 1), 12):02d}-15"
        new.spending_shortfall = _tax_aware_waterfall(new, config, period_spend, sim_date, model=model)
    else:
        new.spending_shortfall = _apply_spending_waterfall(new, period_spend)
```

With:

```python
    new.spending_shortfall = _spending_waterfall(new, config, period_spend, model=model)
```

The `sim_date` computation was already moved to earlier in step() (Task 1, Step 5).

- [ ] **Step 7: Run tests**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -15
```

Expected: All PASS. This is the highest-risk task — investigate any failures carefully.

- [ ] **Step 8: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "fix(citadel): record investment gains in _pay_tax_amount, merge waterfalls into _spending_waterfall"
```

---

### Task 6: Model Info note + gradual rebalancing test + final verification

**Files:**
- Modify: `btc_web/layout/model_info.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Add Model Info note**

Read `btc_web/layout/model_info.py` to find the existing accordion items. Add a note about investment gain classification. The exact location depends on the file structure — look for where Citadel-related notes are or add to a general "Assumptions" section. The text:

"Investment gains in the Citadel Planner are classified as long-term capital gains. Individual equity and bond lot tracking is not modeled."

- [ ] **Step 2: Write gradual rebalancing multi-period test**

Add to `TestTaxAccountingHelpers`:

```python
    def test_gradual_rebalancing_consumes_lots_across_periods(self):
        """Gradual sell over multiple periods correctly consumes lots."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _evaluate_rebalancing)
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=10.0, btc_price=50_000, sim_date="2035-06-15",
            cash=100_000, reserves=[0, 0, 0], investments=[0, 0],
            tax_lots=[TaxLot(date="2031-01-01", btc=10.0,
                             cost_basis=20_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(
            cost_basis_method="fifo",
            high_q_trigger=0.90,
            high_q_action={"mode": "gradual", "rate": 5.0, "duration": 3,
                           "split": {"cash": 1.0}},
        )
        initial_btc = state.btc_stack
        # Trigger gradual + 2 continuations = 3 sells
        for i in range(3):
            _evaluate_rebalancing(state, cfg, btc_quantile=0.95)
        assert state.btc_stack < initial_btc
        # Lots should be consumed consistently
        lot_sum = sum(l.btc for l in state.tax_lots)
        assert abs(lot_sum - state.btc_stack) < 1e-8
        # Gains should be accumulated
        assert state.tax_year_accum.lt_capital_gains > 0
```

- [ ] **Step 3: Run full test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 4: Verify imports**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c \
  "from engines.citadel import _sell_btc_tracked, _buy_btc_tracked, _sell_investments_tracked, _spending_waterfall; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Verify `_apply_spending_waterfall` is gone**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c \
  "from engines.citadel import _apply_spending_waterfall" 2>&1
```

Expected: `ImportError` (function deleted).

- [ ] **Step 6: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/layout/model_info.py btc_web/test_web.py
git commit -m "feat(citadel): Model Info investment gains note + gradual rebalancing lot test"
```
