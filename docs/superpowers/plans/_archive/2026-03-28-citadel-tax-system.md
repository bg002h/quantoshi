# Citadel Tax System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a comprehensive US federal + state tax simulation layer to the Citadel Planner, with lot-level BTC cost basis tracking, three account wrappers, growth-aware withdrawal ordering, and a full-screen tax configuration modal.

**Architecture:** Bottom-up build in 12 tasks: (1-3) pure-Python tax modules with full test coverage (no Dash dependencies), (4) integrate tax into the Citadel engine, (5) tab defaults + config plumbing, (6) tax modal layout, (7) tax modal callbacks, (8) callback wiring, (9) figure builder ghost traces + annotations, (10) snapshot integration, (11) tax summary panel + comparison toggle, (12) cache key expansion + final integration. Each task produces a working, testable commit.

**Tech Stack:** Python 3.14, NumPy, Plotly, Dash 4.0.0, DBC 2.0.4

**Spec:** `docs/superpowers/specs/2026-03-28-citadel-tax-system-design.md`

**Test file:** `btc_web/test_web.py` — all tests go here, following existing class-based patterns. Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py -v -k "Tax"`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `btc_web/engines/tax_data.py` | Static tax data: federal brackets (TCJA + sunset), LTCG brackets, state rates, RMD factors, standard deductions |
| `btc_web/engines/tax_lots.py` | TaxLot dataclass, FIFO/LIFO sell logic, gain classification (ST/LT), partial lot splitting |
| `btc_web/engines/tax.py` | TaxYearAccumulator, annual tax computation (brackets, NIIT, state, loss netting), growth-aware withdrawal ordering |
| `btc_web/layout/citadel_tax.py` | Full-screen tax configuration modal + master toggle widget |
| `btc_web/callbacks/citadel_tax_cb.py` | Tax modal callbacks: open/close, state dropdown auto-fill, save config |

### Modified Files
| File | Change |
|------|--------|
| `btc_web/engines/citadel.py` | Add tax fields to SimConfig/CitadelState, tax branch in `step()`, tax-aware waterfall |
| `btc_web/figures/citadel.py` | Wire tax config into `_build_sim_config()`, ghost traces, tax-drag annotation |
| `btc_web/layout/citadel.py` | Insert master toggle + modal into `_sim_panel()` |
| `btc_web/callbacks/citadel_cb.py` | Add tax State inputs, pass tax config to figure builder |
| `btc_web/tab_defaults.py` | Add tax defaults to CITADEL dict |
| `btc_web/snapshot.py` | Add tax controls to `_SNAPSHOT_CONTROLS` and `_CHECKLIST_OPTIONS` |
| `btc_web/callbacks/routing.py` | Add tax component IDs to `_TAB_CONTROLS["citadel"]` |
| `btc_web/test_web.py` | All new tests (tax engine unit tests, integration, UI) |

---

### Task 1: Tax Data Module — Brackets, State Rates, RMD Factors

**Files:**
- Create: `btc_web/engines/tax_data.py`
- Modify: `btc_web/test_web.py`

This module is pure static data with zero dependencies. All other tax modules import from it.

- [ ] **Step 1: Write failing test for federal bracket lookup**

Add to `btc_web/test_web.py`:
```python
class TestTaxData:
    def test_federal_brackets_single_tcja(self):
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        single = FEDERAL_BRACKETS_TCJA["single"]
        assert single[0] == (11_925, 0.10)
        assert single[-1][1] == 0.37
        assert len(single) == 7

    def test_federal_brackets_mfj_tcja(self):
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        mfj = FEDERAL_BRACKETS_TCJA["mfj"]
        assert mfj[0] == (23_850, 0.10)
        assert mfj[-1][1] == 0.37

    def test_federal_brackets_sunset(self):
        from engines.tax_data import FEDERAL_BRACKETS_SUNSET
        single = FEDERAL_BRACKETS_SUNSET["single"]
        assert single[-1][1] == 0.396
        assert len(single) == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/test_web.py -v -k "TestTaxData" --no-header -q`
Expected: FAIL (import error)

- [ ] **Step 3: Write tax_data.py — federal brackets**

Create `btc_web/engines/tax_data.py`:
```python
"""Static US federal + state tax data for the Citadel tax simulation.

All bracket thresholds are 2025 base values. The engine inflation-indexes
them forward from 2025 using the user's configured inflation rate.
"""
from __future__ import annotations

# ---------- Federal ordinary income brackets (threshold, rate) ----------
# Threshold = upper limit of that bracket. Rates are marginal.
FEDERAL_BRACKETS_TCJA: dict[str, list[tuple[float, float]]] = {
    "single": [
        (11_925, 0.10), (48_475, 0.12), (103_350, 0.22),
        (197_300, 0.24), (252_525, 0.32), (591_975, 0.35),
        (float("inf"), 0.37),
    ],
    "mfj": [
        (23_850, 0.10), (96_950, 0.12), (206_700, 0.22),
        (394_600, 0.24), (505_050, 0.32), (731_200, 0.35),
        (float("inf"), 0.37),
    ],
}

FEDERAL_BRACKETS_SUNSET: dict[str, list[tuple[float, float]]] = {
    "single": [
        (11_925, 0.10), (48_475, 0.15), (103_350, 0.25),
        (197_300, 0.28), (252_525, 0.33), (471_475, 0.35),
        (float("inf"), 0.396),
    ],
    "mfj": [
        (23_850, 0.10), (96_950, 0.15), (206_700, 0.25),
        (394_600, 0.28), (505_050, 0.33), (565_175, 0.35),
        (float("inf"), 0.396),
    ],
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/test_web.py -v -k "TestTaxData" --no-header -q`
Expected: 3 passed

- [ ] **Step 5: Write failing tests for LTCG brackets, standard deductions, NIIT thresholds**

```python
    def test_ltcg_brackets_single(self):
        from engines.tax_data import LTCG_BRACKETS
        single = LTCG_BRACKETS["single"]
        assert single[0] == (48_350, 0.00)
        assert single[1] == (533_400, 0.15)
        assert single[2] == (float("inf"), 0.20)

    def test_standard_deduction_tcja(self):
        from engines.tax_data import STANDARD_DEDUCTION_TCJA, STANDARD_DEDUCTION_SUNSET
        assert STANDARD_DEDUCTION_TCJA["single"] == 15_000
        assert STANDARD_DEDUCTION_TCJA["mfj"] == 30_000
        assert STANDARD_DEDUCTION_SUNSET["single"] == 8_300
        assert STANDARD_DEDUCTION_SUNSET["mfj"] == 16_600

    def test_niit_thresholds(self):
        from engines.tax_data import NIIT_RATE, NIIT_THRESHOLD
        assert NIIT_RATE == 0.038
        assert NIIT_THRESHOLD["single"] == 200_000
        assert NIIT_THRESHOLD["mfj"] == 250_000
```

- [ ] **Step 6: Implement LTCG brackets, standard deductions, NIIT constants**

Add to `btc_web/engines/tax_data.py`:
```python
# ---------- Long-term capital gains brackets ----------
LTCG_BRACKETS: dict[str, list[tuple[float, float]]] = {
    "single": [
        (48_350, 0.00), (533_400, 0.15), (float("inf"), 0.20),
    ],
    "mfj": [
        (96_700, 0.00), (600_050, 0.15), (float("inf"), 0.20),
    ],
}

# ---------- Standard deduction (2025 base) ----------
STANDARD_DEDUCTION_TCJA: dict[str, float] = {"single": 15_000, "mfj": 30_000}
STANDARD_DEDUCTION_SUNSET: dict[str, float] = {"single": 8_300, "mfj": 16_600}

# ---------- NIIT (Net Investment Income Tax) ----------
NIIT_RATE: float = 0.038
NIIT_THRESHOLD: dict[str, float] = {"single": 200_000, "mfj": 250_000}
```

- [ ] **Step 7: Run tests, verify pass**

- [ ] **Step 8: Write failing tests for state tax lookup**

```python
    def test_state_tax_no_income_tax(self):
        from engines.tax_data import STATE_TAX_RATES
        for st in ("AK", "FL", "NV", "NH", "SD", "TN", "TX", "WA", "WY"):
            assert STATE_TAX_RATES[st] == 0.0, f"{st} should be 0"

    def test_state_tax_california(self):
        from engines.tax_data import STATE_TAX_RATES
        assert STATE_TAX_RATES["CA"] == 13.30

    def test_state_tax_count(self):
        from engines.tax_data import STATE_TAX_RATES
        assert len(STATE_TAX_RATES) == 51  # 50 states + DC
```

- [ ] **Step 9: Implement STATE_TAX_RATES dict**

Add the full 51-entry dict to `tax_data.py` (all 50 states + DC with 2025 top marginal rates). See spec Section 2.4 and the state rate research.

- [ ] **Step 10: Write failing tests for RMD factors**

```python
    def test_rmd_factors(self):
        from engines.tax_data import RMD_FACTORS
        assert RMD_FACTORS[73] == 26.5
        assert RMD_FACTORS[75] == 24.6
        assert RMD_FACTORS[80] == 20.2
        assert RMD_FACTORS[90] == 12.2
        assert 72 in RMD_FACTORS
        assert 120 in RMD_FACTORS
```

- [ ] **Step 11: Implement RMD_FACTORS dict**

Add IRS Uniform Lifetime Table factors (ages 72-120) to `tax_data.py`.

- [ ] **Step 12: Run all TestTaxData tests, verify pass**

- [ ] **Step 13: Commit**

```bash
git add btc_web/engines/tax_data.py btc_web/test_web.py
git commit -m "feat(tax): add static tax data module — brackets, state rates, RMD factors"
```

---

### Task 2: Tax Lots Module — FIFO/LIFO, Gain Classification

**Files:**
- Create: `btc_web/engines/tax_lots.py`
- Modify: `btc_web/test_web.py`

Pure Python, depends only on `dataclasses` and `datetime`. No Dash or engine dependencies.

- [ ] **Step 1: Write failing test for TaxLot creation**

```python
class TestTaxLots:
    def test_create_lot(self):
        from engines.tax_lots import TaxLot
        lot = TaxLot(date="2024-01-15", btc=0.5, cost_basis=42_000.0, source="initial")
        assert lot.btc == 0.5
        assert lot.cost_basis == 42_000.0
```

- [ ] **Step 2: Run test, verify fail**

- [ ] **Step 3: Implement TaxLot dataclass**

Create `btc_web/engines/tax_lots.py`:
```python
"""Lot-level BTC tracking for capital gains tax computation."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date as Date

@dataclass
class TaxLot:
    date: str          # ISO format YYYY-MM-DD
    btc: float         # BTC amount in this lot
    cost_basis: float  # USD per BTC at acquisition
    source: str        # "initial", "rebal_buy", "scf", "low_q"
```

- [ ] **Step 4: Run test, verify pass**

- [ ] **Step 5: Write failing tests for sell_lots (FIFO)**

```python
    def test_sell_fifo_single_lot(self):
        from engines.tax_lots import TaxLot, sell_lots
        lots = [TaxLot("2023-01-01", 1.0, 20_000.0, "initial")]
        result = sell_lots(lots, btc_to_sell=0.5, sale_price=50_000.0,
                          sale_date="2025-06-01", method="fifo")
        assert result.btc_sold == 0.5
        assert len(result.gains) == 1
        g = result.gains[0]
        assert g.btc == 0.5
        assert g.proceeds == 25_000.0  # 0.5 * 50k
        assert g.cost == 10_000.0      # 0.5 * 20k
        assert g.gain == 15_000.0
        assert g.is_long_term is True  # > 365 days
        assert len(result.remaining_lots) == 1
        assert result.remaining_lots[0].btc == 0.5  # partial lot

    def test_sell_fifo_multiple_lots(self):
        from engines.tax_lots import TaxLot, sell_lots
        lots = [
            TaxLot("2023-01-01", 0.3, 20_000.0, "initial"),
            TaxLot("2025-03-01", 0.7, 80_000.0, "rebal_buy"),
        ]
        result = sell_lots(lots, btc_to_sell=0.5, sale_price=100_000.0,
                          sale_date="2025-06-01", method="fifo")
        assert result.btc_sold == 0.5
        assert len(result.gains) == 2
        # First lot fully consumed (0.3 BTC, long-term)
        assert result.gains[0].btc == 0.3
        assert result.gains[0].is_long_term is True
        # Second lot partially consumed (0.2 BTC, short-term < 365 days)
        assert result.gains[1].btc == pytest.approx(0.2)
        assert result.gains[1].is_long_term is False
        assert len(result.remaining_lots) == 1
        assert result.remaining_lots[0].btc == pytest.approx(0.5)

    def test_sell_lifo(self):
        from engines.tax_lots import TaxLot, sell_lots
        lots = [
            TaxLot("2023-01-01", 0.5, 20_000.0, "initial"),
            TaxLot("2025-05-01", 0.5, 80_000.0, "rebal_buy"),
        ]
        result = sell_lots(lots, btc_to_sell=0.3, sale_price=100_000.0,
                          sale_date="2025-06-01", method="lifo")
        # LIFO sells newest first
        assert result.gains[0].cost_basis == 80_000.0
        assert result.gains[0].is_long_term is False

    def test_sell_loss(self):
        from engines.tax_lots import TaxLot, sell_lots
        lots = [TaxLot("2024-01-01", 1.0, 100_000.0, "initial")]
        result = sell_lots(lots, btc_to_sell=0.5, sale_price=50_000.0,
                          sale_date="2025-06-01", method="fifo")
        assert result.gains[0].gain == -25_000.0  # loss
```

- [ ] **Step 6: Implement sell_lots with SaleResult and LotGain dataclasses**

```python
@dataclass
class LotGain:
    btc: float
    cost_basis: float    # per-BTC basis of the lot
    sale_price: float    # per-BTC sale price
    proceeds: float      # btc * sale_price
    cost: float          # btc * cost_basis
    gain: float          # proceeds - cost
    is_long_term: bool   # held >= 365 days
    holding_days: int

@dataclass
class SaleResult:
    btc_sold: float
    gains: list[LotGain]
    remaining_lots: list[TaxLot]

def sell_lots(lots: list[TaxLot], btc_to_sell: float, sale_price: float,
              sale_date: str, method: str = "fifo") -> SaleResult:
    """Sell BTC from lots using FIFO or LIFO. Returns gains and remaining lots."""
    ordered = sorted(lots, key=lambda l: l.date, reverse=(method == "lifo"))
    remaining = []
    gains = []
    to_sell = btc_to_sell
    for lot in ordered:
        if to_sell <= 0:
            remaining.append(TaxLot(lot.date, lot.btc, lot.cost_basis, lot.source))
            continue
        sell_from_lot = min(lot.btc, to_sell)
        days = (Date.fromisoformat(sale_date) - Date.fromisoformat(lot.date)).days
        proceeds = sell_from_lot * sale_price
        cost = sell_from_lot * lot.cost_basis
        gains.append(LotGain(
            btc=sell_from_lot, cost_basis=lot.cost_basis, sale_price=sale_price,
            proceeds=proceeds, cost=cost, gain=proceeds - cost,
            is_long_term=(days >= 365), holding_days=days,
        ))
        leftover = lot.btc - sell_from_lot
        if leftover > 1e-8:
            remaining.append(TaxLot(lot.date, leftover, lot.cost_basis, lot.source))
        to_sell -= sell_from_lot
    # Re-sort remaining by date ascending (canonical order)
    remaining.sort(key=lambda l: l.date)
    return SaleResult(btc_sold=btc_to_sell - max(to_sell, 0), gains=gains,
                      remaining_lots=remaining)
```

- [ ] **Step 7: Run tests, verify pass**

- [ ] **Step 8: Write failing test for seed_lots_from_stack_tracker**

```python
    def test_seed_from_stack_tracker(self):
        from engines.tax_lots import seed_lots
        st_lots = [
            {"date": "2023-06-15", "btc": 0.5, "price": 30_000},
            {"date": "2024-01-10", "btc": 0.3, "price": 45_000},
        ]
        tax_lots = seed_lots(st_lots)
        assert len(tax_lots) == 2
        assert tax_lots[0].date == "2023-06-15"
        assert tax_lots[0].cost_basis == 30_000
        assert tax_lots[1].source == "initial"

    def test_seed_manual_entry(self):
        from engines.tax_lots import seed_lots
        tax_lots = seed_lots([], start_stack=1.0, start_price=60_000.0,
                             start_date="2031-01-01")
        assert len(tax_lots) == 1
        assert tax_lots[0].btc == 1.0
        assert tax_lots[0].cost_basis == 60_000.0
```

- [ ] **Step 9: Implement seed_lots**

```python
def seed_lots(stack_tracker_lots: list[dict], *, start_stack: float = 0.0,
              start_price: float = 0.0, start_date: str = "") -> list[TaxLot]:
    """Create TaxLots from Stack Tracker lots or manual entry."""
    if stack_tracker_lots:
        return [TaxLot(date=l["date"], btc=float(l["btc"]),
                       cost_basis=float(l["price"]), source="initial")
                for l in stack_tracker_lots if float(l.get("btc", 0)) > 0]
    if start_stack > 0 and start_date:
        return [TaxLot(date=start_date, btc=start_stack,
                       cost_basis=start_price, source="initial")]
    return []
```

- [ ] **Step 10: Run tests, verify pass**

- [ ] **Step 11: Commit**

```bash
git add btc_web/engines/tax_lots.py btc_web/test_web.py
git commit -m "feat(tax): add lot-level BTC tracking — FIFO/LIFO, gain classification"
```

---

### Task 3: Tax Computation Module — Brackets, NIIT, Loss Netting, Withdrawal Ordering

**Files:**
- Create: `btc_web/engines/tax.py`
- Modify: `btc_web/test_web.py`

Depends on `tax_data.py`. Pure Python, no Dash dependencies.

- [ ] **Step 1: Write failing tests for progressive bracket computation**

```python
class TestTaxComputation:
    def test_apply_brackets_10pct_only(self):
        from engines.tax import apply_progressive_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        # $10,000 income, single — all in 10% bracket
        tax = apply_progressive_brackets(10_000, FEDERAL_BRACKETS_TCJA["single"])
        assert tax == pytest.approx(1_000.0)

    def test_apply_brackets_two_brackets(self):
        from engines.tax import apply_progressive_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        # $30,000 single: 10% on $11,925 + 12% on $18,075
        tax = apply_progressive_brackets(30_000, FEDERAL_BRACKETS_TCJA["single"])
        expected = 11_925 * 0.10 + (30_000 - 11_925) * 0.12
        assert tax == pytest.approx(expected)

    def test_apply_brackets_top_bracket(self):
        from engines.tax import apply_progressive_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        # $1M single — hits 37% bracket
        tax = apply_progressive_brackets(1_000_000, FEDERAL_BRACKETS_TCJA["single"])
        assert tax > 300_000  # rough sanity check

    def test_apply_brackets_zero(self):
        from engines.tax import apply_progressive_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        assert apply_progressive_brackets(0, FEDERAL_BRACKETS_TCJA["single"]) == 0.0
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement apply_progressive_brackets**

Create `btc_web/engines/tax.py`:
```python
"""Tax computation engine: brackets, NIIT, loss netting, annual tax calc."""
from __future__ import annotations
from dataclasses import dataclass, field

def apply_progressive_brackets(taxable_income: float,
                                brackets: list[tuple[float, float]]) -> float:
    """Compute tax using progressive brackets. Each bracket is (threshold, rate)."""
    if taxable_income <= 0:
        return 0.0
    tax = 0.0
    prev_threshold = 0.0
    for threshold, rate in brackets:
        taxable_in_bracket = min(taxable_income, threshold) - prev_threshold
        if taxable_in_bracket > 0:
            tax += taxable_in_bracket * rate
        if taxable_income <= threshold:
            break
        prev_threshold = threshold
    return tax
```

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Write failing tests for LTCG stacking**

```python
    def test_ltcg_stacking_zero_ordinary(self):
        from engines.tax import compute_ltcg_tax
        # $50k LTCG, $0 ordinary: first $48,350 at 0%, rest at 15%
        tax = compute_ltcg_tax(50_000, stacking_base=0, filing_status="single")
        expected = (50_000 - 48_350) * 0.15
        assert tax == pytest.approx(expected)

    def test_ltcg_stacking_high_ordinary(self):
        from engines.tax import compute_ltcg_tax
        # $100k LTCG, $80k ordinary: 0% bracket fully consumed by ordinary
        tax = compute_ltcg_tax(100_000, stacking_base=80_000, filing_status="single")
        # All $100k in the 15% bracket (80k+100k=180k < 533,400 threshold)
        assert tax == pytest.approx(100_000 * 0.15)
```

- [ ] **Step 6: Implement compute_ltcg_tax**

```python
def compute_ltcg_tax(taxable_ltcg: float, stacking_base: float,
                     filing_status: str) -> float:
    """Compute LTCG tax with stacking: brackets start above ordinary_taxable."""
    from engines.tax_data import LTCG_BRACKETS
    if taxable_ltcg <= 0:
        return 0.0
    brackets = LTCG_BRACKETS[filing_status]
    tax = 0.0
    prev_threshold = 0.0
    for threshold, rate in brackets:
        # Adjust bracket for stacking: effective bracket space available
        bracket_start = max(prev_threshold, stacking_base)
        bracket_end = threshold
        if bracket_start >= bracket_end:
            prev_threshold = threshold
            continue
        space_in_bracket = bracket_end - bracket_start
        ltcg_remaining = taxable_ltcg - (bracket_start - stacking_base)
        if ltcg_remaining <= 0:
            break
        taxable_in_bracket = min(ltcg_remaining, space_in_bracket)
        # Recalculate more simply:
        prev_threshold = threshold
    # Simpler approach: treat stacking_base + ltcg as total, subtract tax on stacking_base
    total = stacking_base + taxable_ltcg
    tax_total = apply_progressive_brackets(total, brackets)
    tax_base = apply_progressive_brackets(stacking_base, brackets)
    return tax_total - tax_base
```

- [ ] **Step 7: Run tests, verify pass**

- [ ] **Step 8: Write failing tests for capital loss netting**

```python
    def test_loss_netting_st_loss_offsets_lt_gain(self):
        from engines.tax import net_capital_gains
        result = net_capital_gains(
            st_gains=1_000, st_losses=5_000,
            lt_gains=10_000, lt_losses=0, carryforward=0)
        # Net ST = -4000, Net LT = 10000
        # Cross-category: net_lt reduced by 4000 = 6000
        assert result.net_lt == 6_000
        assert result.net_st == 0
        assert result.loss_deduction == 0  # no remaining loss
        assert result.new_carryforward == 0

    def test_loss_netting_excess_carries_forward(self):
        from engines.tax import net_capital_gains
        result = net_capital_gains(
            st_gains=0, st_losses=10_000,
            lt_gains=0, lt_losses=0, carryforward=0)
        # Net ST = -10000, Net LT = 0 → combined loss = 10000
        # $3000 deduction, $7000 carries forward
        assert result.loss_deduction == 3_000
        assert result.new_carryforward == 7_000

    def test_loss_netting_with_carryforward(self):
        from engines.tax import net_capital_gains
        result = net_capital_gains(
            st_gains=5_000, st_losses=0,
            lt_gains=0, lt_losses=0, carryforward=8_000)
        # Carryforward applied as LT loss: net_lt = -8000
        # Cross-category: net_st 5000 reduced by 3000 = 2000, remaining loss = 3000
        # Wait — carryforward offsets gains: net_st=5000, carryforward as LT: net_lt=-8000
        # Cross: combined = 5000 + (-8000) = -3000
        assert result.loss_deduction == 3_000
        assert result.new_carryforward == 0
```

- [ ] **Step 9: Implement net_capital_gains**

```python
@dataclass
class CapitalGainResult:
    net_st: float          # Net short-term (positive = gain, 0 if cross-netted)
    net_lt: float          # Net long-term (positive = gain, 0 if cross-netted)
    loss_deduction: float  # Up to $3,000 against ordinary income
    new_carryforward: float

def net_capital_gains(st_gains: float, st_losses: float,
                      lt_gains: float, lt_losses: float,
                      carryforward: float) -> CapitalGainResult:
    """Apply IRS Section 1(h) capital loss netting rules."""
    net_st = st_gains - st_losses
    net_lt = lt_gains - lt_losses - carryforward  # carryforward treated as LT loss (v1)

    # Cross-category offset
    if net_st < 0 and net_lt > 0:
        combined = net_st + net_lt
        net_st = min(combined, 0.0)
        net_lt = max(combined, 0.0)
    elif net_lt < 0 and net_st > 0:
        combined = net_st + net_lt
        net_st = max(combined, 0.0)
        net_lt = min(combined, 0.0)

    total_net_loss = abs(min(net_st, 0.0)) + abs(min(net_lt, 0.0))
    loss_deduction = min(total_net_loss, 3_000.0)
    new_carryforward = max(total_net_loss - 3_000.0, 0.0)

    return CapitalGainResult(
        net_st=max(net_st, 0.0), net_lt=max(net_lt, 0.0),
        loss_deduction=loss_deduction, new_carryforward=new_carryforward,
    )
```

- [ ] **Step 10: Run tests, verify pass**

- [ ] **Step 11: Write failing tests for NIIT computation**

```python
    def test_niit_below_threshold(self):
        from engines.tax import compute_niit
        # MAGI $150k single — below $200k threshold
        assert compute_niit(magi=150_000, nii=50_000, filing_status="single") == 0.0

    def test_niit_above_threshold(self):
        from engines.tax import compute_niit
        # MAGI $300k, NII $80k, single
        # 3.8% * min(80k, 300k - 200k) = 3.8% * 80k = 3040
        assert compute_niit(300_000, 80_000, "single") == pytest.approx(3_040.0)

    def test_niit_lesser_of_rule(self):
        from engines.tax import compute_niit
        # MAGI $220k, NII $50k, single
        # 3.8% * min(50k, 220k - 200k) = 3.8% * 20k = 760
        assert compute_niit(220_000, 50_000, "single") == pytest.approx(760.0)
```

- [ ] **Step 12: Implement compute_niit**

```python
def compute_niit(magi: float, nii: float, filing_status: str) -> float:
    from engines.tax_data import NIIT_RATE, NIIT_THRESHOLD
    threshold = NIIT_THRESHOLD[filing_status]
    excess_magi = max(magi - threshold, 0.0)
    return NIIT_RATE * min(nii, excess_magi)
```

- [ ] **Step 13: Run tests, verify pass**

- [ ] **Step 14: Write failing test for compute_annual_tax (full pipeline)**

```python
    def test_annual_tax_simple_case(self):
        from engines.tax import TaxYearAccumulator, compute_annual_tax
        accum = TaxYearAccumulator(
            tax_deferred_withdrawals=60_000,
            interest_income=5_000,
            other_income=0,
            lt_capital_gains=45_000,
        )
        result = compute_annual_tax(accum, filing_status="single",
                                     tcja_sunset=False, sim_year=2031,
                                     inflation_rate=0.04, state_rate=0.0)
        assert result["total"] > 0
        assert result["federal_ordinary"] > 0
        assert result["federal_ltcg"] >= 0
        assert result["niit"] == 0  # AGI under $200k
        assert result["effective_rate"] > 0
```

- [ ] **Step 15: Implement TaxYearAccumulator and compute_annual_tax**

Implement the full `TaxYearAccumulator` dataclass and `compute_annual_tax()` function following spec Section 5.1 and 5.2 exactly. This is the core tax computation: loss netting → AGI → standard deduction allocation → ordinary tax → LTCG stacking → NIIT → state tax → total.

- [ ] **Step 16: Run tests, verify pass**

- [ ] **Step 17: Write failing test for inflation-indexed brackets**

```python
    def test_brackets_inflation_indexed(self):
        from engines.tax import _inflate_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        base = FEDERAL_BRACKETS_TCJA["single"]
        inflated = _inflate_brackets(base, years=10, rate=0.04)
        # After 10 years at 4%, first threshold ~= 11925 * 1.04^10 ≈ 17648
        assert inflated[0][0] == pytest.approx(11_925 * 1.04**10, rel=0.01)
        assert inflated[0][1] == 0.10  # rates don't change
```

- [ ] **Step 18: Implement _inflate_brackets helper**

```python
def _inflate_brackets(brackets: list[tuple[float, float]], years: float,
                      rate: float) -> list[tuple[float, float]]:
    factor = (1 + rate) ** years
    return [(t * factor if t != float("inf") else float("inf"), r)
            for t, r in brackets]
```

- [ ] **Step 19: Run all TestTaxComputation tests, verify pass**

- [ ] **Step 20: Commit**

```bash
git add btc_web/engines/tax.py btc_web/test_web.py
git commit -m "feat(tax): add tax computation engine — brackets, NIIT, loss netting"
```

---

### Task 4: Integrate Tax Into Citadel Engine

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Modify: `btc_web/test_web.py`

This is the core integration: add tax fields to SimConfig/CitadelState, add the tax branch in `step()`, implement tax-aware spending waterfall with growth-aware ordering.

- [ ] **Step 1: Write failing test for new SimConfig tax fields**

```python
class TestCitadelTaxIntegration:
    def test_sim_config_has_tax_fields(self):
        from engines.citadel import SimConfig
        cfg = SimConfig(tax_enabled=True, filing_status="single",
                        state_code="CA", birth_year=1985,
                        cost_basis_method="fifo")
        assert cfg.tax_enabled is True
        assert cfg.state_code == "CA"
        assert cfg.td_btc_stack == 0.0
        assert cfg.tf_btc_stack == 0.0
```

- [ ] **Step 2: Add tax fields to SimConfig**

Add all fields from spec Section 7.1 to the `SimConfig` dataclass in `btc_web/engines/citadel.py` (after the existing `tax_rate` field at line 88). Replace the `tax_rate` placeholder with the full tax config fields.

- [ ] **Step 3: Run test, verify pass**

- [ ] **Step 4: Write failing test for new CitadelState tax fields**

```python
    def test_citadel_state_has_tax_fields(self):
        from engines.citadel import CitadelState
        state = CitadelState()
        assert hasattr(state, "tax_lots")
        assert hasattr(state, "td_btc_stack")
        assert hasattr(state, "tf_btc_stack")
        assert hasattr(state, "td_cash")
        assert hasattr(state, "total_taxes_paid")
        assert state.total_taxes_paid == 0.0
```

- [ ] **Step 5: Add tax fields to CitadelState**

Add all fields from spec Section 7.2 to `CitadelState`.

- [ ] **Step 6: Run test, verify pass**

- [ ] **Step 7: Write failing test for SimResult tax extensions**

```python
    def test_sim_result_has_tax_fields(self):
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=1.0, start_yr=2031, end_yr=2033,
                        freq="Annually", monthly_spend=5000,
                        cash_initial=200_000, selected_qs=[0.25],
                        tax_enabled=True, filing_status="single")
        result = simulate(cfg, M)
        assert hasattr(result, "taxes_paid")
        assert hasattr(result, "annual_taxes")
        assert hasattr(result, "td_total")
        assert hasattr(result, "tf_total")
        assert hasattr(result, "taxable_total")
        assert len(result.annual_taxes) >= 1
```

- [ ] **Step 8: Add SimResult tax fields**

Add these fields to the `SimResult` dataclass (or its constructor) per spec Section 7.4:
- `taxes_paid: np.ndarray` — cumulative taxes paid per period
- `annual_taxes: list[dict]` — per-year tax breakdown dicts
- `td_total: np.ndarray` — Tax-Deferred balance per period
- `tf_total: np.ndarray` — Tax-Free balance per period
- `taxable_total: np.ndarray` — Taxable balance per period
- `tax_shortfall: float` — unpaid tax when all accounts depleted

- [ ] **Step 9: Write failing test for tax-enabled simulation end-to-end**

```python
    def test_tax_enabled_reduces_terminal_wealth(self):
        """Tax-on sim should have lower terminal portfolio than tax-off."""
        from engines.citadel import SimConfig, simulate
        base = dict(start_stack=1.0, start_yr=2031, end_yr=2035,
                    freq="Monthly", monthly_spend=5000,
                    cash_initial=200_000, selected_qs=[0.25])
        cfg_off = SimConfig(**base, tax_enabled=False)
        cfg_on = SimConfig(**base, tax_enabled=True, filing_status="single",
                           state_code="CA")
        # Need a mock model or the real model
        # Use the loaded model M from the test module top
        result_off = simulate(cfg_off, M)
        result_on = simulate(cfg_on, M)
        # Tax-on terminal total should be <= tax-off
        final_off = result_off.total[0, -1]
        final_on = result_on.total[0, -1]
        assert final_on <= final_off
```

- [ ] **Step 10: Implement the tax branch in step()**

Add the tax-aware code path to `step()` following spec Section 7.3. When `config.tax_enabled`:
1. Grow all three wrapper balances
2. Accumulate taxable-wrapper interest to `tax_year_accum`
3. Inflate `other_income`: `accum.other_income = config.other_income * (1 + growth)^years`
4. Force RMD if applicable (age-based: born 1951-1959 → age 73, born 1960+ → age 75)
5. Evaluate rebalancing (BTC sales record lot gains)
6. Apply growth-aware tax-optimized waterfall across all 3 wrappers
7. Enforce floors (taxable wrapper only)
8. At year boundary: compute annual tax via `compute_annual_tax()`, pay from taxable wrapper using gross-up formula `gross_amount = net_tax_owed / (1 - marginal_rate)`
9. Reset accumulator, carry forward losses
10. Check total depletion across ALL three wrappers

Key helper functions to implement:
- `_tax_aware_waterfall()` — growth-aware ordering from spec Section 6.2
- `_btc_fwd_growth()` — forward-looking annual growth rate from the model
- `_compute_opportunity_cost()` — `tax_rate + growth_rate * shelter_multiplier`
- `_pay_annual_tax()` — 5-step payment cascade: taxable cash → reserves → investments → TD withdrawal → BTC sale, each using gross-up formula
- `_rmd_start_age(birth_year)` — returns 73 for born 1951-1959, 75 for born 1960+

- [ ] **Step 11: Run integration test, verify pass**

- [ ] **Step 12: Write test for tax payment cascade (gross-up)**

```python
    def test_tax_payment_gross_up(self):
        """Tax payment on investments should use gross-up to cover tax-on-tax."""
        from engines.citadel import SimConfig, simulate
        # Set up: only taxable investments, force high income to trigger taxes
        cfg = SimConfig(
            start_stack=0.0, start_yr=2031, end_yr=2032,
            freq="Annually", monthly_spend=0,
            cash_initial=0, selected_qs=[0.25],
            invest_bins=[{"label": "Equities", "initial": 1_000_000,
                          "return_rate": 10, "volatility": 0},
                         {"label": "Bonds", "initial": 0,
                          "return_rate": 0, "volatility": 0}],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=200_000,  # pushes into high bracket
        )
        result = simulate(cfg, M)
        # Should have paid taxes > 0
        assert result.annual_taxes[0]["total"] > 0

    def test_tax_off_preserves_existing_behavior(self):
        """When tax_enabled=False, behavior is identical to current engine."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=1.0, start_yr=2031, end_yr=2033,
                        freq="Monthly", monthly_spend=5000,
                        cash_initial=200_000, selected_qs=[0.25],
                        tax_enabled=False)
        result = simulate(cfg, M)
        # No tax fields populated
        assert not hasattr(result, "annual_taxes") or len(result.annual_taxes) == 0
```

- [ ] **Step 13: Write test for RMD age 75 boundary (SECURE 2.0)**

```python
    def test_rmd_age_75_for_born_1960(self):
        """Born 1960+: RMD starts at age 75, not 73."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0.0, start_yr=2035, end_yr=2037,
            freq="Annually", monthly_spend=0, cash_initial=0,
            selected_qs=[0.25], tax_enabled=True,
            birth_year=1960, filing_status="single",
            td_cash_initial=500_000,
        )
        result = simulate(cfg, M)
        # Born 1960, age 75 = 2035 — RMD should start in first year
        assert result.annual_taxes[0]["ordinary_income"] > 0

    def test_rmd_not_yet_for_born_1960_at_age_73(self):
        """Born 1960: no RMD at age 73 (2033) — must wait until 75."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0.0, start_yr=2033, end_yr=2034,
            freq="Annually", monthly_spend=0, cash_initial=0,
            selected_qs=[0.25], tax_enabled=True,
            birth_year=1960, filing_status="single",
            td_cash_initial=500_000,
        )
        result = simulate(cfg, M)
        # At age 73 (2033), no RMD for born-1960 — ordinary income should be 0
        if result.annual_taxes:
            assert result.annual_taxes[0]["ordinary_income"] == 0

    def test_other_income_inflates(self):
        """Other income should grow by other_income_growth rate."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0.0, start_yr=2031, end_yr=2035,
            freq="Annually", monthly_spend=0, cash_initial=100_000,
            selected_qs=[0.25], tax_enabled=True,
            other_income=50_000, other_income_growth=4.0,
            filing_status="single",
        )
        result = simulate(cfg, M)
        # Year 2035 = 4 years of growth: 50k * 1.04^4 ≈ 58,493
        yr_2035 = [t for t in result.annual_taxes if t["year"] == 2035]
        if yr_2035:
            assert yr_2035[0]["ordinary_income"] > 50_000
```

- [ ] **Step 14: Write test for lot creation on BTC purchase during rebalancing**

```python
    def test_rebalancing_buy_creates_lot(self):
        """Low-Q rebalancing buy should create a new TaxLot."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0.5, start_yr=2031, end_yr=2032,
            freq="Monthly", monthly_spend=0, cash_initial=500_000,
            selected_qs=[0.25], tax_enabled=True,
            low_q_trigger=0.50,  # trigger easily for test
            low_q_action={"mode": "lump", "rate": 5.0, "duration": 1,
                          "split": {"cash": 1.0, "res_short": 0, "res_med": 0,
                                    "res_long": 0, "inv_eq": 0, "inv_bd": 0}},
        )
        result = simulate(cfg, M)
        # Should have lots created from rebalancing buys
        final_state = result.states[-1]
        assert len(final_state.tax_lots) >= 1
```

- [ ] **Step 15: Run test, iterate on implementation**

- [ ] **Step 16: Write test for RMD forced withdrawal**

```python
    def test_rmd_forced_at_correct_age(self):
        """RMD should force TD withdrawal starting at age 73 (born 1958)."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0.0, start_yr=2031, end_yr=2033,
            freq="Annually", monthly_spend=0, cash_initial=0,
            selected_qs=[0.25], tax_enabled=True,
            birth_year=1958, filing_status="single",
            td_cash_initial=500_000,
        )
        result = simulate(cfg, M)
        # Born 1958: RMD starts at age 73 → year 2031
        tax_years = [t for t in result.annual_taxes if t.get("year") == 2031]
        if tax_years:
            assert tax_years[0]["ordinary_income"] > 0
```

- [ ] **Step 17: Run test, iterate**

- [ ] **Step 18: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "feat(tax): integrate tax computation into Citadel engine — wrappers, lots, waterfall"
```

---

### Task 5: Tab Defaults + Config Plumbing

**Files:**
- Modify: `btc_web/tab_defaults.py`
- Modify: `btc_web/figures/citadel.py`
- Modify: `btc_web/test_web.py`

Wire the tax config from UI defaults through `_build_sim_config()` to the engine.

- [ ] **Step 1: Write failing test for tax defaults in CITADEL dict**

```python
class TestTaxDefaults:
    def test_citadel_has_tax_defaults(self):
        from tab_defaults import CITADEL
        assert CITADEL["tax_enabled"] is False
        assert CITADEL["filing_status"] == "single"
        assert CITADEL["state_code"] == "TX"
        assert CITADEL["td_btc"] == 0.0
        assert CITADEL["tf_btc"] == 0.0
        assert CITADEL["cost_basis_method"] == "fifo"
```

- [ ] **Step 2: Add tax defaults to CITADEL dict in tab_defaults.py**

Add all keys from spec Section 9.2 to the `CITADEL` MappingProxyType in `btc_web/tab_defaults.py`.

- [ ] **Step 3: Run test, verify pass**

- [ ] **Step 4: Write failing test for _build_sim_config tax field passthrough**

```python
    def test_build_sim_config_passes_tax_fields(self):
        from figures.citadel import _build_sim_config
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["tax_enabled"] = True
        p["filing_status"] = "mfj"
        p["state_code"] = "CA"
        p["birth_year"] = 1985
        cfg = _build_sim_config(p)
        assert cfg.tax_enabled is True
        assert cfg.filing_status == "mfj"
        assert cfg.state_code == "CA"
        assert cfg.birth_year == 1985
```

- [ ] **Step 5: Update _build_sim_config in figures/citadel.py**

Replace the hardcoded `tax_rate=0.0` (line 185) with the full set of tax fields extracted from params dict `p`.

- [ ] **Step 6: Run test, verify pass**

- [ ] **Step 7: Commit**

```bash
git add btc_web/tab_defaults.py btc_web/figures/citadel.py btc_web/test_web.py
git commit -m "feat(tax): add tax defaults and wire config through figure builder"
```

---

### Task 6: Tax Modal Layout

**Files:**
- Create: `btc_web/layout/citadel_tax.py`
- Modify: `btc_web/layout/citadel.py`
- Modify: `btc_web/test_web.py`

Build the full-screen modal and master toggle. No callbacks yet — just the DOM.

- [ ] **Step 1: Write failing test for tax toggle existence**

```python
class TestTaxLayout:
    def test_tax_toggle_exists(self):
        """The master tax toggle should be in the Citadel layout."""
        import app as _app
        from layout.citadel_tax import tax_toggle_widget
        widget = tax_toggle_widget()
        # Should contain a Switch with id cp-tax-toggle
        assert widget is not None
```

- [ ] **Step 2: Create citadel_tax.py with tax_toggle_widget and tax_modal**

Create `btc_web/layout/citadel_tax.py` with:
- `tax_toggle_widget()` — returns a `dbc.Card` containing a `dbc.Switch(id="cp-tax-toggle")` and a `dbc.Button(id="cp-tax-config-btn")` for opening the modal
- `tax_config_modal()` — returns a `dbc.Modal(id="cp-tax-modal", fullscreen=True)` with three sections:
  - Section A: Filing & Rates (filing status radio, state dropdown, birth year, other income, TCJA toggle, cost basis method)
  - Section B: Account Wrappers (3 cards for Taxable/TD/TF, TD and TF have BTC + cash + reserves + investments inputs)
  - Section C: Tax Rate Reference (collapsible read-only bracket display)
- `dcc.Store(id="cp-tax-config", storage_type="memory")` for persisting tax settings

Component IDs follow the `cp-tax-*` prefix pattern: `cp-tax-toggle`, `cp-tax-config-btn`, `cp-tax-modal`, `cp-tax-filing`, `cp-tax-state`, `cp-tax-state-rate`, `cp-tax-birth-year`, `cp-tax-other-income`, `cp-tax-other-income-growth`, `cp-tax-tcja`, `cp-tax-basis-method`, `cp-td-btc`, `cp-td-cash`, `cp-td-res-short`, `cp-td-res-med`, `cp-td-res-long`, `cp-td-inv-eq`, `cp-td-inv-bd`, `cp-tf-btc`, `cp-tf-cash`, `cp-tf-res-short`, `cp-tf-res-med`, `cp-tf-res-long`, `cp-tf-inv-eq`, `cp-tf-inv-bd`.

- [ ] **Step 3: Run test, verify pass**

- [ ] **Step 4: Insert toggle and modal into _sim_panel()**

Modify `btc_web/layout/citadel.py` to import and insert `tax_toggle_widget()` and `tax_config_modal()` into `_sim_panel()`, above the existing "BTC Price Scenario" section.

- [ ] **Step 5: Syntax-check the web app**

Run: `cd btc_web && PYTHONPATH=".:../archive/btc_app" ../btc_venv/bin/python3 -c "import layout, figures, callbacks, cache, engines.adapter, engines.citadel, data.asset_matrices; print('OK')"`

- [ ] **Step 6: Commit**

```bash
git add btc_web/layout/citadel_tax.py btc_web/layout/citadel.py btc_web/test_web.py
git commit -m "feat(tax): add full-screen tax configuration modal and master toggle"
```

---

### Task 7: Tax Modal Callbacks

**Files:**
- Create: `btc_web/callbacks/citadel_tax_cb.py`
- Modify: `btc_web/callbacks/__init__.py`
- Modify: `btc_web/test_web.py`

Wire up modal open/close, state dropdown auto-fill, and save config.

- [ ] **Step 1: Write failing test for state dropdown auto-fill**

```python
class TestTaxCallbacks:
    def test_state_dropdown_fills_rate(self):
        from callbacks.citadel_tax_cb import _state_to_rate
        assert _state_to_rate("CA") == 13.30
        assert _state_to_rate("TX") == 0.0
        assert _state_to_rate("NY") == 10.90
```

- [ ] **Step 2: Create citadel_tax_cb.py with callbacks**

Create `btc_web/callbacks/citadel_tax_cb.py` with:
- Clientside callback: toggle "Configure" button visibility based on `cp-tax-toggle` value
- Server callback: `_update_state_rate()` — when state dropdown changes, auto-fill the rate input
- Server callback: `_open_tax_modal()` — opens modal on button click
- Server callback: `_save_tax_config()` — collects all modal inputs, writes to `cp-tax-config` Store, closes modal
- Clientside callback: update Run button label based on `cp-tax-toggle` ("Run Simulation" vs "Run Simulation (with Tax)")

- [ ] **Step 3: Import callbacks in __init__.py**

Add `import callbacks.citadel_tax_cb` to `btc_web/callbacks/__init__.py`.

- [ ] **Step 4: Run test, verify pass**

- [ ] **Step 5: Syntax-check the web app**

- [ ] **Step 6: Commit**

```bash
git add btc_web/callbacks/citadel_tax_cb.py btc_web/callbacks/__init__.py btc_web/test_web.py
git commit -m "feat(tax): add tax modal callbacks — state dropdown, save config"
```

---

### Task 8: Wire Tax Config Into Citadel Callback

**Files:**
- Modify: `btc_web/callbacks/citadel_cb.py`
- Modify: `btc_web/test_web.py`

Add the tax toggle and tax config as State inputs to the main Citadel callback.

- [ ] **Step 1: Add State inputs to update_citadel callback**

Add these State parameters to the `update_citadel` callback signature in `btc_web/callbacks/citadel_cb.py`:
- `State("cp-tax-toggle", "value")`
- `State("cp-tax-config", "data")`

In the callback body, unpack the tax config dict and pass fields into the params dict `p` for `_get_citadel_fig()`.

- [ ] **Step 2: Write test for tax config passthrough in callback**

```python
    def test_citadel_callback_passes_tax_config(self):
        """Tax config from modal should reach the figure builder."""
        # This is a smoke test — verify the callback doesn't crash
        # when tax_toggle is on and tax_config is populated
        from callbacks.citadel_cb import update_citadel
        # ... construct args matching the callback signature with tax fields
```

- [ ] **Step 3: Run tests, verify pass**

- [ ] **Step 4: Syntax-check the web app**

- [ ] **Step 5: Commit**

```bash
git add btc_web/callbacks/citadel_cb.py btc_web/test_web.py
git commit -m "feat(tax): wire tax config into Citadel callback"
```

---

### Task 9: Figure Builder — Ghost Traces, Tax Annotations, Summary Panel

**Files:**
- Modify: `btc_web/figures/citadel.py`
- Modify: `btc_web/test_web.py`

When tax is enabled: run a parallel tax-off sim for comparison, add ghost traces, tax-drag annotation, and per-year tax summary.

- [ ] **Step 1: Write failing test for comparison traces**

```python
class TestTaxFigures:
    def test_tax_on_produces_ghost_traces(self):
        from figures.citadel import build_citadel_figure
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["tax_enabled"] = True
        p["filing_status"] = "single"
        p["state_code"] = "CA"
        fig, extra = build_citadel_figure(M, p)
        trace_names = [t.name for t in fig.data if t.name]
        # Should have dashed "no-tax" ghost trace
        assert any("(no tax)" in n for n in trace_names)

    def test_tax_off_no_ghost_traces(self):
        from figures.citadel import build_citadel_figure
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        fig, extra = build_citadel_figure(M, p)
        trace_names = [t.name for t in fig.data if t.name]
        assert not any("(no tax)" in n for n in trace_names)
```

- [ ] **Step 2: Implement comparison traces in build_citadel_figure**

When `p.get("tax_enabled")`:
1. Build tax-on SimConfig and run simulation
2. Build tax-off SimConfig (same params, `tax_enabled=False`) and run simulation
3. Add main traces from tax-on result (solid lines)
4. Add ghost traces from tax-off result (dashed, 50% opacity, name suffixed " (no tax)")
5. Add "Taxes Paid" cumulative trace (red area, secondary y-axis) if `result.taxes_paid`
6. Add tax-drag annotation at the final period
7. Append chart title with `(Federal + {State} Tax)`
8. Store `result.annual_taxes` in the extra return dict for the summary panel

- [ ] **Step 3: Run tests, verify pass**

- [ ] **Step 4: Write test for tax summary data returned**

```python
    def test_tax_summary_data_returned(self):
        from figures.citadel import build_citadel_figure
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["tax_enabled"] = True
        p["start_yr"] = 2031
        p["end_yr"] = 2033
        fig, extra = build_citadel_figure(M, p)
        assert "annual_taxes" in extra
        assert len(extra["annual_taxes"]) >= 1
        first_year = extra["annual_taxes"][0]
        assert "total" in first_year
        assert "effective_rate" in first_year
```

- [ ] **Step 5: Run test, verify pass**

- [ ] **Step 6: Commit**

```bash
git add btc_web/figures/citadel.py btc_web/test_web.py
git commit -m "feat(tax): add ghost traces, tax-drag annotation, tax summary to Citadel chart"
```

---

### Task 10: Snapshot Integration + Final Wiring

**Files:**
- Modify: `btc_web/snapshot.py`
- Modify: `btc_web/callbacks/routing.py`
- Modify: `btc_web/test_web.py`

Add tax controls to snapshot system so share links preserve tax settings.

- [ ] **Step 1: Write failing test for snapshot round-trip**

```python
class TestTaxSnapshot:
    def test_tax_controls_in_snapshot(self):
        from snapshot import _SNAPSHOT_CONTROLS
        tax_ids = [c[0] for c in _SNAPSHOT_CONTROLS if "tax" in c[0]]
        assert "cp-tax-toggle" in [c[0] for c in _SNAPSHOT_CONTROLS]
        assert "cp-tax-config" in [c[0] for c in _SNAPSHOT_CONTROLS]
```

- [ ] **Step 2: Add ALL tax controls to _SNAPSHOT_CONTROLS in snapshot.py**

Append to the Citadel section of `_SNAPSHOT_CONTROLS`:
```python
("cp-tax-toggle", "value"),
("cp-tax-config", "data"),
("cp-td-btc", "value"), ("cp-td-cash", "value"),
("cp-td-res-short", "value"), ("cp-td-res-med", "value"), ("cp-td-res-long", "value"),
("cp-td-inv-eq", "value"), ("cp-td-inv-bd", "value"),
("cp-tf-btc", "value"), ("cp-tf-cash", "value"),
("cp-tf-res-short", "value"), ("cp-tf-res-med", "value"), ("cp-tf-res-long", "value"),
("cp-tf-inv-eq", "value"), ("cp-tf-inv-bd", "value"),
```

Add to `_CHECKLIST_OPTIONS`:
```python
"cp-tax-toggle": ["yes"],
```

- [ ] **Step 3: Add ALL tax IDs to _TAB_CONTROLS in routing.py**

Add `"cp-tax-toggle"`, `"cp-tax-config"`, and all 14 `"cp-td-*"` / `"cp-tf-*"` component IDs to the `"citadel"` set in `_TAB_CONTROLS`.

- [ ] **Step 4: Run snapshot validation assertion**

Run: `cd btc_web && PYTHONPATH=".:../archive/btc_app" ../btc_venv/bin/python3 -c "import snapshot; print('Snapshot OK')"`

This will trigger the assertion at lines 316-318 that verifies all `_CHECKLIST_OPTIONS` keys exist in `_SNAPSHOT_CONTROLS`.

- [ ] **Step 5: Run snapshot-related tests**

Run: `cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/test_web.py -v -k "TaxSnapshot" --no-header -q`

- [ ] **Step 6: Commit**

```bash
git add btc_web/snapshot.py btc_web/callbacks/routing.py btc_web/test_web.py
git commit -m "feat(tax): add tax controls to snapshot system and tab routing"
```

---

### Task 11: Tax Summary Panel + Comparison Toggle

**Files:**
- Modify: `btc_web/layout/citadel_tax.py`
- Modify: `btc_web/layout/citadel.py`
- Modify: `btc_web/callbacks/citadel_cb.py`
- Modify: `btc_web/test_web.py`

Build the year-by-year tax summary table below the chart and the "Show tax comparison" checkbox.

- [ ] **Step 1: Add "Show tax comparison" checkbox to chart toggles**

In `btc_web/layout/citadel.py`, add a `"tax_compare"` option to `cp-toggles` checklist (or add a separate `cp-tax-compare` checklist visible only when tax is on).

- [ ] **Step 2: Add tax summary panel layout**

In `btc_web/layout/citadel_tax.py`, create `tax_summary_panel()` returning a `dbc.Collapse(id="cp-tax-summary")` containing a `dash_table.DataTable(id="cp-tax-summary-table")` or `dbc.Table`. Columns: Year, Ordinary Inc, ST Gains, LT Gains, Loss Ded., Federal Tax, NIIT, State Tax, Total Tax, Eff. Rate, Carryforward.

- [ ] **Step 3: Insert tax summary panel below the Citadel chart**

In `btc_web/layout/citadel.py`, add the summary panel below the `dcc.Graph(id="citadel-graph")`.

- [ ] **Step 4: Wire summary panel data in callback**

In `btc_web/callbacks/citadel_cb.py`, add `Output("cp-tax-summary-table", "data")` and `Output("cp-tax-summary", "is_open")` to the callback. Populate from `extra["annual_taxes"]` when tax is on. Collapse is closed when tax is off.

- [ ] **Step 5: Wire "Show tax comparison" toggle**

Pass the comparison toggle value through to the figure builder. In `build_citadel_figure`, only add ghost traces when the toggle is on.

- [ ] **Step 6: Write tests**

```python
class TestTaxSummaryPanel:
    def test_summary_panel_exists_in_layout(self):
        from layout.citadel_tax import tax_summary_panel
        panel = tax_summary_panel()
        assert panel is not None
```

- [ ] **Step 7: Run tests, syntax-check**

- [ ] **Step 8: Commit**

```bash
git add btc_web/layout/citadel_tax.py btc_web/layout/citadel.py btc_web/callbacks/citadel_cb.py btc_web/test_web.py
git commit -m "feat(tax): add tax summary panel and comparison toggle"
```

---

### Task 12: Cache Key Expansion + Final Integration

**Files:**
- Modify: `btc_web/cache.py`
- Modify: `btc_web/utils.py`
- Modify: `btc_web/test_web.py`

Ensure tax params are included in cache keys so tax-on and tax-off results don't collide.

- [ ] **Step 1: Verify cache key includes tax params**

The `_quantize_params(p)` function in `btc_web/utils.py` already quantizes all float params via `_q3`. Tax params (filing_status, state_code, etc.) are strings/bools that pass through unchanged. Verify that the cache key differentiates `tax_enabled=True` from `tax_enabled=False` by running:

```python
    def test_cache_key_differs_with_tax(self):
        from utils import _quantize_params
        from tab_defaults import citadel_defaults
        p_off = citadel_defaults()
        p_on = citadel_defaults()
        p_on["tax_enabled"] = True
        assert _quantize_params(p_off) != _quantize_params(p_on)
```

- [ ] **Step 2: If test fails, add tax_enabled to cache key differentiation**

The `_quantize_params` function should already handle this since `tax_enabled` is a key in the params dict. If not, ensure it's included.

- [ ] **Step 3: Verify L0 prewarm skips tax (tax_enabled=False by default)**

The L0 prewarm in `cache.py` uses `citadel_defaults()` which has `tax_enabled=False`. No changes needed — just verify.

- [ ] **Step 4: Run full test suite**

Run: `cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --no-header -q --timeout=120`

- [ ] **Step 5: Final syntax check + dev server smoke test**

Run: `cd btc_web && PYTHONPATH=".:../archive/btc_app" ../btc_venv/bin/python3 -c "import layout, figures, callbacks, cache, engines.adapter, engines.citadel, engines.tax, engines.tax_lots, engines.tax_data, data.asset_matrices; print('OK')"`

Then: `DEV=1 bash run_web.sh` — navigate to `/9`, verify:
- Tax toggle appears in Simulation sub-tab
- Toggle ON → "Configure Tax Settings" button appears, Run button says "(with Tax)"
- Modal opens fullscreen with Filing, Wrappers, Reference sections
- State dropdown auto-fills rate
- Run simulation with tax ON → ghost traces, tax-drag annotation, summary panel
- Toggle OFF → no ghost traces, no summary panel

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "feat(tax): cache key expansion + final integration — Citadel tax system complete"
```
