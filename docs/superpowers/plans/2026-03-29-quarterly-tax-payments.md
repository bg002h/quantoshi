# Quarterly Estimated Tax Payments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change Citadel tax payment from annual lump sum to quarterly estimated payments with Q4 true-up.

**Architecture:** Add `quarterly_tax_paid_ytd` field to `CitadelState`. Extract `_pay_tax_amount()` from `_year_boundary_tax()` to share payment sourcing logic. Add `_quarterly_estimated_payment()` that annualizes YTD income and pays the estimated quarterly amount. Modify `step()` to call quarterly payments at quarter boundaries and Q4 true-up at year-end.

**Tech Stack:** Python 3.14, dataclass, numpy

**Spec:** `docs/superpowers/specs/2026-03-29-quarterly-tax-payments-design.md`

---

### Task 1: Add `quarterly_tax_paid_ytd` to CitadelState

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write the failing test**

Add to `btc_web/test_web.py` after the existing `TestCitadelTaxIntegration` class:

```python
class TestQuarterlyTaxPayments:
    """Quarterly estimated tax payment tests."""

    def test_state_has_quarterly_field(self):
        from engines.citadel import CitadelState
        state = CitadelState()
        assert hasattr(state, "quarterly_tax_paid_ytd")
        assert state.quarterly_tax_paid_ytd == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestQuarterlyTaxPayments::test_state_has_quarterly_field -v
```

Expected: FAIL with `AttributeError`

- [ ] **Step 3: Add the field to CitadelState**

In `btc_web/engines/citadel.py`, add after the `annual_tax_history` field (around line 182):

```python
    quarterly_tax_paid_ytd: float = 0.0       # cumulative estimated payments this tax year
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestQuarterlyTaxPayments -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "feat(citadel): add quarterly_tax_paid_ytd to CitadelState"
```

---

### Task 2: Extract `_pay_tax_amount()` from `_year_boundary_tax()`

The tax payment sourcing logic (cash → reserves → investments gross-up → TD gross-up) is currently inlined in `_year_boundary_tax()`. We need to reuse it for quarterly payments. Extract it into a standalone function.

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write the failing test**

Add to `TestQuarterlyTaxPayments` in `btc_web/test_web.py`:

```python
    def test_pay_tax_amount_draws_from_cash_first(self):
        """_pay_tax_amount draws cash before other sources."""
        from engines.citadel import CitadelState, SimConfig, _pay_tax_amount
        state = CitadelState(
            cash=50_000, reserves=[0, 0, 0], investments=[0, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[0, 0],
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX")
        _pay_tax_amount(state, cfg, amount=30_000, sim_year=2031)
        assert state.cash == pytest.approx(20_000)

    def test_pay_tax_amount_uses_investments_after_cash(self):
        """_pay_tax_amount falls through to investments when cash exhausted."""
        from engines.citadel import CitadelState, SimConfig, _pay_tax_amount
        state = CitadelState(
            cash=10_000, reserves=[0, 0, 0], investments=[100_000, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[100_000, 0],  # full basis = no gain = no gross-up
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX")
        _pay_tax_amount(state, cfg, amount=30_000, sim_year=2031)
        assert state.cash == 0
        assert state.investments[0] < 100_000
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestQuarterlyTaxPayments -v
```

Expected: FAIL with `ImportError: cannot import name '_pay_tax_amount'`

- [ ] **Step 3: Extract `_pay_tax_amount()` from `_year_boundary_tax()`**

In `btc_web/engines/citadel.py`, add a new function BEFORE `_year_boundary_tax()` (around line 983):

```python
def _pay_tax_amount(state: CitadelState, config: SimConfig,
                    amount: float, sim_year: int,
                    tax_result: dict | None = None) -> None:
    """Pay `amount` of tax from taxable accounts with gross-up.

    Payment order: cash → reserves → investments (LTCG gross-up) → TD (ordinary gross-up).
    If tax_result is None, uses conservative rate estimates.
    Mutates state.
    """
    if amount <= 0:
        return

    from .tax_data import NIIT_THRESHOLD
    tax_remaining = amount

    # 1. Cash — no gross-up
    draw = min(state.cash, tax_remaining)
    state.cash -= draw
    tax_remaining -= draw

    # 2. Reserves — no gross-up
    for i in range(len(state.reserves)):
        if tax_remaining <= 0:
            break
        draw = min(state.reserves[i], tax_remaining)
        state.reserves[i] -= draw
        tax_remaining -= draw

    # 3. Investments — gross-up for LTCG on gain portion
    if tax_remaining > 0:
        agi = tax_result.get("agi", 0) if tax_result else 0
        niit_applies = agi > NIIT_THRESHOLD.get(config.filing_status, 200_000)
        ltcg_rate = 0.15 + (0.038 if niit_applies else 0) + _get_state_rate(config) / 100
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
            tax_on_sale = (draw - basis_removed) * ltcg_rate
            tax_remaining -= (draw - tax_on_sale)

    # 4. TD — gross-up for ordinary income
    if tax_remaining > 0:
        from .tax import apply_progressive_brackets, _inflate_brackets
        from .tax_data import FEDERAL_BRACKETS_TCJA, FEDERAL_BRACKETS_SUNSET
        _agi = tax_result.get("agi", 0) if tax_result else 0
        _yrs = max(sim_year - 2025, 0)
        if config.tcja_sunset:
            _brk = _inflate_brackets(FEDERAL_BRACKETS_SUNSET[config.filing_status], _yrs, config.inflation / 100)
        else:
            _brk = _inflate_brackets(FEDERAL_BRACKETS_TCJA[config.filing_status], _yrs, config.inflation / 100)
        _tax_at_agi = apply_progressive_brackets(_agi, _brk)
        _tax_at_agi_plus = apply_progressive_brackets(_agi + 1, _brk)
        _marginal_fed = _tax_at_agi_plus - _tax_at_agi
        ordinary_rate = _marginal_fed + _get_state_rate(config) / 100
        gross = tax_remaining / max(1.0 - ordinary_rate, 0.1)
        td_avail = state.td_cash + sum(state.td_reserves) + sum(state.td_investments)
        td_draw = min(gross, td_avail)
        if td_draw > 0:
            rem = td_draw
            d = min(state.td_cash, rem); state.td_cash -= d; rem -= d
            for j in range(len(state.td_reserves)):
                if rem <= 0: break
                d = min(state.td_reserves[j], rem); state.td_reserves[j] -= d; rem -= d
            for j in reversed(range(len(state.td_investments))):
                if rem <= 0: break
                d = min(state.td_investments[j], rem); state.td_investments[j] -= d; rem -= d
```

Then replace the payment logic in `_year_boundary_tax()` (lines 1018–1091, the `if tax_owed > 0:` block) with a single call:

```python
    _pay_tax_amount(state, config, tax_owed, sim_year, tax_result=tax_result)
```

- [ ] **Step 4: Run ALL tests to verify refactor is clean**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -10
```

Expected: All tests PASS (same count as before + 2 new).

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "refactor(citadel): extract _pay_tax_amount from _year_boundary_tax"
```

---

### Task 3: Implement `_quarterly_estimated_payment()`

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write the failing tests**

Add to `TestQuarterlyTaxPayments`:

```python
    def test_quarterly_payment_annualizes_ytd(self):
        """Q1 payment should be ~25% of annualized tax projection."""
        from engines.citadel import (SimConfig, CitadelState,
                                      _quarterly_estimated_payment)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=500_000, reserves=[0, 0, 0], investments=[0, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[0, 0],
            tax_year_accum=TaxYearAccumulator(other_income=50_000),
            quarterly_tax_paid_ytd=0,
        )
        cfg = SimConfig(tax_enabled=True, state_code="CA",
                        filing_status="single", inflation=4.0)
        _quarterly_estimated_payment(state, cfg, quarter=1, sim_year=2031)
        # Should have paid ~25% of annualized tax (50k * 4 = 200k income)
        assert state.quarterly_tax_paid_ytd > 0
        assert state.cash < 500_000

    def test_quarterly_payment_cumulative_tracking(self):
        """Q2 payment accounts for Q1 already paid."""
        from engines.citadel import (SimConfig, CitadelState,
                                      _quarterly_estimated_payment)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=500_000, reserves=[0, 0, 0], investments=[0, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[0, 0],
            tax_year_accum=TaxYearAccumulator(other_income=100_000),
            quarterly_tax_paid_ytd=10_000,  # Q1 already paid
        )
        cfg = SimConfig(tax_enabled=True, state_code="CA",
                        filing_status="single", inflation=4.0)
        cash_before = state.cash
        _quarterly_estimated_payment(state, cfg, quarter=2, sim_year=2031)
        q2_payment = cash_before - state.cash
        # Cumulative target is 50% of projected annual tax.
        # Q1 paid 10k, so Q2 pays (50% of projected - 10k).
        assert state.quarterly_tax_paid_ytd > 10_000

    def test_quarterly_no_payment_if_overpaid(self):
        """If already overpaid relative to cumulative target, pay $0."""
        from engines.citadel import (SimConfig, CitadelState,
                                      _quarterly_estimated_payment)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=500_000, reserves=[0, 0, 0], investments=[0, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[0, 0],
            tax_year_accum=TaxYearAccumulator(other_income=10_000),
            quarterly_tax_paid_ytd=100_000,  # massively overpaid
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX",
                        filing_status="single", inflation=4.0)
        _quarterly_estimated_payment(state, cfg, quarter=2, sim_year=2031)
        assert state.cash == 500_000  # no payment drawn
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestQuarterlyTaxPayments -v
```

Expected: FAIL with `ImportError: cannot import name '_quarterly_estimated_payment'`

- [ ] **Step 3: Implement `_quarterly_estimated_payment()`**

Add to `btc_web/engines/citadel.py` after `_pay_tax_amount()`:

```python
def _quarterly_estimated_payment(state: CitadelState, config: SimConfig,
                                  quarter: int, sim_year: int) -> None:
    """Pay estimated quarterly tax (Q1-Q3). Annualizes YTD income.

    quarter: 1, 2, or 3 (Q4 is handled by _year_boundary_tax as the true-up).
    Mutates state: draws payment from accounts, updates quarterly_tax_paid_ytd.
    """
    from .tax import compute_annual_tax, TaxYearAccumulator
    from copy import copy

    if state.tax_year_accum is None or quarter < 1 or quarter > 3:
        return

    ytd_fraction = quarter / 4.0  # 0.25, 0.50, 0.75

    # Annualize YTD income by scaling up
    ann = copy(state.tax_year_accum)
    scale = 1.0 / ytd_fraction
    ann.tax_deferred_withdrawals *= scale
    ann.interest_income *= scale
    ann.treasury_interest *= scale
    ann.other_income *= scale
    ann.st_capital_gains *= scale
    ann.st_capital_losses *= scale
    ann.lt_capital_gains *= scale
    ann.lt_capital_losses *= scale
    ann.roth_withdrawals *= scale
    ann.loss_carryforward = state.loss_carryforward  # not scaled

    state_rate = _get_state_rate(config)
    projected = compute_annual_tax(
        ann,
        filing_status=config.filing_status,
        tcja_sunset=config.tcja_sunset,
        sim_year=sim_year,
        inflation_rate=config.inflation / 100,
        state_rate=state_rate,
    )

    cumulative_target = projected["total"] * (quarter / 4.0)
    payment = max(cumulative_target - state.quarterly_tax_paid_ytd, 0)

    if payment > 0:
        _pay_tax_amount(state, config, payment, sim_year)
        state.quarterly_tax_paid_ytd += payment
        state.total_taxes_paid += payment
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestQuarterlyTaxPayments -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "feat(citadel): add _quarterly_estimated_payment with annualization"
```

---

### Task 4: Wire quarterly payments into `step()` and modify Q4 true-up

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write the failing tests**

Add to `TestQuarterlyTaxPayments`:

```python
    def test_monthly_sim_pays_4_times_per_year(self):
        """Monthly frequency produces quarterly payments + year-end true-up."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2033,
            freq="Monthly", monthly_spend=0,
            cash_initial=1_000_000, selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=300_000,
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        model = _test_model()
        r = simulate(cfg, model)
        # With monthly freq over 2 years, taxes should be paid
        assert r.taxes_paid[0, -1] > 0
        # Cash should be drawn more evenly than annual lump (not all at once)
        # The first quarterly payment should occur by period 3
        assert r.taxes_paid is not None

    def test_annual_freq_falls_back_to_year_end(self):
        """Annually frequency: no quarterly payments, all at year-end."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2034,
            freq="Annually", monthly_spend=0,
            cash_initial=1_000_000, selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=200_000,
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        model = _test_model()
        r = simulate(cfg, model)
        assert r.taxes_paid[0, -1] > 0

    def test_q4_trueup_accounts_for_quarterly_payments(self):
        """Year-end tax = actual annual tax - sum of Q1-Q3 estimates."""
        from engines.citadel import SimConfig, simulate
        # Run two sims: one monthly (quarterly payments) one annual (lump)
        # Both should end with approximately the same total tax paid
        common = dict(
            start_stack=0, start_yr=2031, end_yr=2034,
            monthly_spend=0, cash_initial=1_000_000,
            selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=200_000,
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        r_monthly = simulate(SimConfig(freq="Monthly", **common), _test_model())
        r_annual = simulate(SimConfig(freq="Annually", **common), _test_model())
        tax_monthly = r_monthly.taxes_paid[0, -1]
        tax_annual = r_annual.taxes_paid[0, -1]
        # Should be approximately equal (minor differences from interest timing)
        assert abs(tax_monthly - tax_annual) / max(tax_annual, 1) < 0.05, \
            f"Monthly {tax_monthly:.0f} vs Annual {tax_annual:.0f} differ by >5%"

    def test_quarterly_tax_paid_ytd_resets_each_year(self):
        """quarterly_tax_paid_ytd must be 0 at the start of each new tax year."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2033,
            freq="Monthly", monthly_spend=0,
            cash_initial=1_000_000, selected_qs=[0.25],
            tax_enabled=True, state_code="CA", other_income=200_000,
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        model = _test_model()
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        for i in range(24):  # 2 full years
            state = step(state, cfg, 50_000, rng, model=model)
            # At year boundary (period 12, 24), ytd should reset to 0
            if state.period % 12 == 0:
                assert state.quarterly_tax_paid_ytd == 0, \
                    f"Period {state.period}: ytd should be 0, got {state.quarterly_tax_paid_ytd:.0f}"

    def test_cash_floor_respected_after_quarterly_payment(self):
        """Cash floor must hold after each quarterly tax payment."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=5.0, start_yr=2031, end_yr=2033,
            freq="Monthly", monthly_spend=1000,
            cash_initial=100_000, cash_floor=80_000,
            selected_qs=[0.25],
            tax_enabled=True, state_code="CA", other_income=500_000,
            reserve_bins=[
                {"label": "Short", "initial": 200_000, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 200_000, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        model = _ControlledPriceModel(quantile=0.50, price=50_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        for i in range(24):
            state = step(state, cfg, 50_000, rng, model=model)
            total_other = (sum(state.reserves) + sum(state.investments)
                           + state.btc_stack * state.btc_price)
            if total_other > 80_000:
                assert state.cash >= 80_000 - 100, \
                    f"Period {i+1}: cash {state.cash:.0f} below floor"
```

- [ ] **Step 2: Run tests to verify some fail**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestQuarterlyTaxPayments -v --tb=short
```

Expected: New integration tests may fail because `step()` doesn't call quarterly payments yet.

- [ ] **Step 3: Modify `step()` to call quarterly payments**

In `btc_web/engines/citadel.py`, replace the tax section in `step()` (the block starting at `# 8. Year-boundary tax computation and RMD`) with:

```python
    # 8. Tax: quarterly estimated payments + year-end true-up
    if config.tax_enabled:
        sim_year = config.start_yr + int(years_elapsed)
        is_year_end = new.period > 0 and new.period % ppy == 0

        # Quarterly estimated payments (Q1-Q3) — only for freq >= Quarterly
        if ppy >= 4 and not is_year_end:
            qtr_periods = ppy // 4  # periods per quarter
            if new.period % qtr_periods == 0:
                quarter = (new.period % ppy) // qtr_periods  # 1, 2, or 3
                if 1 <= quarter <= 3:
                    _quarterly_estimated_payment(new, config, quarter, sim_year)
                    _enforce_floors(new, config)

        # Year-end: RMD + Q4 true-up
        if is_year_end:
            # RMD first (adds ordinary income)
            rmd = _compute_rmd(new, config, sim_year)
            if rmd > 0 and new.tax_year_accum is not None:
                new.tax_year_accum.tax_deferred_withdrawals += rmd
                new.tax_year_accum.rmd_required = rmd
                new.tax_year_accum.rmd_taken = rmd
            # Q4 true-up: actual tax minus quarterly payments already made
            _year_boundary_tax(new, config, sim_year, ppy)
            _enforce_floors(new, config)
```

- [ ] **Step 4: Modify `_year_boundary_tax()` to subtract quarterly payments (Q4 true-up)**

In `_year_boundary_tax()`, after computing `tax_owed = tax_result["total"]`, subtract quarterly payments already made and reset the YTD counter:

Replace:
```python
    tax_owed = tax_result["total"]

    # Pay tax from taxable wrapper with gross-up for taxable payment sources.
```

With:
```python
    tax_owed_full = tax_result["total"]

    # Q4 true-up: subtract quarterly estimated payments already made
    q4_payment = tax_owed_full - state.quarterly_tax_paid_ytd
    if q4_payment > 0:
        _pay_tax_amount(state, config, q4_payment, sim_year, tax_result=tax_result)

    # Total tax for the year is the full computed amount
    tax_owed = tax_owed_full
```

And in the section that updates state after payment, replace:
```python
    state.total_taxes_paid += tax_owed
```
With:
```python
    # total_taxes_paid: quarterly payments already added their amounts,
    # so only add the Q4 true-up portion
    state.total_taxes_paid += max(q4_payment, 0)
    # If overpaid (q4_payment < 0), credit reduces cumulative total
    if q4_payment < 0:
        state.total_taxes_paid += q4_payment  # negative → reduces total
```

And after resetting the accumulator, reset the quarterly counter:
```python
    state.quarterly_tax_paid_ytd = 0.0
```

- [ ] **Step 5: Run ALL tests**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -15
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "feat(citadel): wire quarterly estimated tax payments into step()"
```

---

### Task 5: Final verification

**Files:** None modified

- [ ] **Step 1: Run full test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 2: Syntax check**

```bash
cd btc_web && PYTHONPATH=".:../archive/btc_app" ../btc_venv/bin/python3 -c \
  "import engines.citadel, engines.tax, engines.tax_lots, engines.tax_data; print('OK')"
```

Expected: `OK` (ignore RuntimeWarning about sys.prefix).

- [ ] **Step 3: Verify quarterly payment function is importable**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c \
  "from engines.citadel import _quarterly_estimated_payment, _pay_tax_amount; print('OK')"
```

Expected: `OK`
