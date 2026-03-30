# Citadel Engine Split — Design Spec

**Goal:** Split `btc_web/engines/citadel.py` (1885 lines, 31 functions, 5 classes) into 8 focused modules under `engines/`, each with a single responsibility and under 450 lines. A re-export facade preserves all existing imports — zero changes outside `engines/`.

**Motivation:** The file has 6+ responsibilities making it hard to navigate, review, and extend. The dynamic waterfall alone added ~400 lines. Next feature will push it past 2000.

---

## Module Breakdown

### `citadel_types.py` (~280 lines)

All data definitions. No logic.

**Contains:**
- `FREQ_PPY` (imported from `_app_ctx`, filtered to Monthly/Quarterly/Annually)
- `_SATOSHI = 1e-8`
- `SimConfig` dataclass (40+ fields)
- `CitadelState` dataclass
- `PriceModel` protocol
- `SimResult` dataclass (with `to_dict()` and `from_dict()`)
- `_WithdrawalSource` dataclass

**Imports:** `_app_ctx.FREQ_PPY`, `numpy`, `dataclasses`, `typing`

**Depends on:** nothing internal

---

### `citadel_transactions.py` (~80 lines)

Shared BTC and investment transaction helpers with cost basis tracking.

**Contains:**
- `_sell_btc_tracked(state, config, btc_to_sell)` → `SaleResult`
- `_buy_btc_tracked(state, config, btc_bought, source)`
- `_sell_investments_tracked(state, config, bin_index, amount)` → `(drawn, gain)`

**Imports:** `citadel_types` (CitadelState, SimConfig)

**Lazy imports:** `.tax_lots` (TaxLot, sell_lots) — only when tax_enabled

**Depends on:** `citadel_types`

---

### `citadel_waterfall.py` (~450 lines)

Cost-ranked dynamic spending waterfall. The 6 helper functions plus the main loop.

**Contains:**
- `_build_source_list(state, config, model)` → list of `_WithdrawalSource`
- `_score_sources(sources, state, config, model)` — mutates `source.cost`
- `_rank_sources(sources)` → sorted list by cost ascending
- `_max_draw_before_boundary(state, config, source)` → max draw before bracket change
- `_execute_draw(state, config, source, amount)` — dispatches to transaction helpers
- `_spending_waterfall(state, config, amount, model)` → shortfall

**Imports:** `citadel_types` (_WithdrawalSource, CitadelState, SimConfig, FREQ_PPY), `citadel_transactions`

**Lazy imports:** `.tax` (_inflate_brackets), `.tax_data` (brackets, NIIT, standard deductions)

**Depends on:** `citadel_types`, `citadel_transactions`

---

### `citadel_floors.py` (~150 lines)

Floor enforcement — maintains minimum balances by drawing from the waterfall.

**Contains:**
- `_enforce_floors(state, config, model)` — cash floor delegates to `_spending_waterfall`; reserve floors redistribute among taxable dollar accounts
- `_distribute_to_accounts(state, amount, split)` — distributes cash to accounts by split fractions
- `_source_from_accounts(state, config, amount)` — draws from taxable accounts for BTC purchases

**Imports:** `citadel_types`, `citadel_waterfall` (_spending_waterfall), `citadel_transactions` (_sell_investments_tracked)

**Depends on:** `citadel_types`, `citadel_waterfall`, `citadel_transactions`

---

### `citadel_rebalancing.py` (~100 lines)

Threshold-based BTC buy/sell triggers.

**Contains:**
- `_evaluate_rebalancing(state, config, btc_quantile)` — checks high/low quantile triggers, fires lump or gradual actions
- `_execute_sell_btc(state, config, btc_to_sell)` — sells BTC and distributes proceeds
- `_execute_buy_btc(state, config, usd_amount)` — sources USD from accounts and buys BTC

**Imports:** `citadel_types`, `citadel_transactions` (_sell_btc_tracked, _buy_btc_tracked), `citadel_floors` (_distribute_to_accounts, _source_from_accounts)

**Depends on:** `citadel_types`, `citadel_transactions`, `citadel_floors`

---

### `citadel_tax_integration.py` (~220 lines)

Tax computation integration — RMDs, estimated payments, year-end true-up, tax payment execution.

**Contains:**
- `_get_state_rate(config)` → effective state tax rate
- `_rmd_start_age(birth_year)` → 73 or 75
- `_compute_rmd(state, config, sim_year)` → RMD amount (withdraws from TD, adds to cash)
- `_pay_tax_amount(state, config, amount, sim_year, tax_result)` — draws from taxable accounts to pay tax bill
- `_quarterly_estimated_payment(state, config, quarter, sim_year)` — Q1-Q3 estimated payments
- `_year_boundary_tax(state, config, sim_year, ppy)` — year-end RMD + Q4 true-up

**Imports:** `citadel_types`, `citadel_transactions` (_sell_investments_tracked)

**Lazy imports:** `.tax` (TaxYearAccumulator, compute_annual_tax), `.tax_data` (STATE_TAX_RATES, RMD_FACTORS, brackets)

**Depends on:** `citadel_types`, `citadel_transactions`

---

### `citadel_step.py` (~280 lines)

The simulation heartbeat — advances one period.

**Contains:**
- `step(state, config, btc_price, rng, model)` → new CitadelState — the central orchestrator that calls waterfall, floors, rebalancing, tax, SCF in order
- `_get_btc_price(t, config, model, rng, sim_mode, q)` — resolves BTC price for a time step
- `_lognormal_return(rate, vol, rng)` — stochastic dollar-asset return
- `_markov_return(state, config, rng)` — Markov regime-based returns
- `_scf_payment_amount(state, config)` — SCF loan payment calculation
- `_scf_check_repay(state, config, btc_annual_return)` — perpetual loan repayment trigger

**Imports:** `citadel_types`, `citadel_waterfall` (_spending_waterfall), `citadel_floors` (_enforce_floors), `citadel_rebalancing` (_evaluate_rebalancing), `citadel_tax_integration` (_compute_rmd, _quarterly_estimated_payment, _year_boundary_tax), `citadel_transactions` (_sell_btc_tracked)

**Depends on:** `citadel_types`, `citadel_waterfall`, `citadel_floors`, `citadel_rebalancing`, `citadel_tax_integration`, `citadel_transactions`

---

### `citadel_sim.py` (~250 lines)

Simulation driver — loops over `step()`, aggregates results.

**Contains:**
- `simulate(config, model, rng_seed, price_paths)` → SimResult
- `validate_config(config)` — raises ValueError on invalid config
- `_initial_state(config, model)` → CitadelState
- `_snapshot_state(state, tax_enabled)` → dict
- `_aggregate_results(all_histories, config, time_axis, sim_annual_taxes)` → SimResult
- `_compute_n_periods(config)` → int

**Imports:** `citadel_types` (SimConfig, CitadelState, SimResult, PriceModel, FREQ_PPY), `citadel_step` (step, _get_btc_price)

**Lazy imports:** `btc_core` (yr_to_t), `.tax` (TaxYearAccumulator), `.tax_lots` (seed_lots)

**Depends on:** `citadel_types`, `citadel_step`

---

### `citadel.py` (facade, ~40 lines)

Re-exports all public names. Preserves every existing import.

```python
"""Citadel Planner simulation engine — public API facade.

All implementation lives in citadel_*.py submodules. This file
re-exports the public interface so external imports don't change.
"""
from .citadel_types import *
from .citadel_transactions import *
from .citadel_waterfall import *
from .citadel_floors import *
from .citadel_rebalancing import *
from .citadel_tax_integration import *
from .citadel_step import *
from .citadel_sim import *
```

---

## Dependency Graph

```
citadel_types              (no internal deps)
    ↑
citadel_transactions       (types)
    ↑
citadel_waterfall          (types, transactions)
citadel_tax_integration    (types, transactions)
    ↑
citadel_floors             (types, waterfall, transactions)
citadel_rebalancing        (types, transactions, floors)
    ↑
citadel_step               (types, waterfall, floors, rebalancing, tax_integration, transactions)
    ↑
citadel_sim                (types, step)
    ↑
citadel.py facade          (re-exports all)
```

All arrows point up. No circular dependencies.

---

## Constraints

- **Zero changes outside `engines/`** — the facade preserves all existing imports (`from engines.citadel import SimConfig, simulate, step`, etc.)
- **Lazy tax imports preserved** — tax modules are imported inside function bodies, not at module top level, keeping the engine core decoupled
- **Test files unchanged** — they import through the facade. Test file splitting is a separate task.
- **`FREQ_PPY`** stays imported from `_app_ctx` (not hardcoded), lives in `citadel_types.py`
- **`_SATOSHI`** constant lives in `citadel_types.py`

## Verification

- Full test suite passes with zero changes to test files: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short`
- Expected: 902 passed, 2 pre-existing failures, 5 skipped
- Syntax check: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "from engines.citadel import SimConfig, simulate, step, _spending_waterfall, _enforce_floors, _WithdrawalSource, _build_source_list, _score_sources, _rank_sources, _execute_draw, _sell_btc_tracked, SimResult, CitadelState, PriceModel, FREQ_PPY; print('OK')"`
