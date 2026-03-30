# Citadel Engine Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `btc_web/engines/citadel.py` (1885 lines) into 8 focused modules + a re-export facade, with zero changes to files outside `engines/`.

**Architecture:** Extract functions by responsibility into `citadel_*.py` modules. Each module defines `__all__` for wildcard re-export. The existing `citadel.py` becomes a thin facade that re-exports everything, preserving all external imports unchanged.

**Tech Stack:** Python 3.14, dataclasses, numpy

**Spec:** `docs/superpowers/specs/2026-03-30-citadel-engine-split-design.md`

**Test command:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short`

**Import check:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "from engines.citadel import SimConfig, simulate, step, _spending_waterfall, _enforce_floors, _WithdrawalSource, _build_source_list, _score_sources, _rank_sources, _execute_draw, _sell_btc_tracked, SimResult, CitadelState, PriceModel, FREQ_PPY; print('OK')"`

---

### Task 1: Extract `citadel_types.py`

**Files:**
- Create: `btc_web/engines/citadel_types.py`
- Modify: `btc_web/engines/citadel.py`

- [ ] **Step 1: Create `citadel_types.py`**

Extract from `citadel.py` lines 1–15 (imports, FREQ_PPY, _SATOSHI), 19–32 (_WithdrawalSource), 36–154 (SimConfig), 158–213 (CitadelState), 217–224 (PriceModel), 1595–1660 (SimResult). Copy verbatim — do not modify any logic.

Add at the top of the new file:

```python
"""Citadel Planner — data types and configuration."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from _app_ctx import FREQ_PPY as _ALL_FREQ_PPY
FREQ_PPY = {k: v for k, v in _ALL_FREQ_PPY.items() if k in ("Monthly", "Quarterly", "Annually")}
_SATOSHI = 1e-8  # smallest BTC unit — anything below this is zero
```

Then paste the class definitions in order: `_WithdrawalSource`, `SimConfig`, `CitadelState`, `PriceModel`, `SimResult`.

Add `__all__` at the top after imports:

```python
__all__ = [
    "FREQ_PPY", "_SATOSHI",
    "_WithdrawalSource", "SimConfig", "CitadelState", "PriceModel", "SimResult",
]
```

- [ ] **Step 2: Update `citadel.py` — replace extracted code with import**

Remove the extracted class definitions and constants from `citadel.py`. Replace the top section (lines 1–15, classes at 19–224, and SimResult at 1595–1660) with:

```python
"""Citadel Planner simulation engine — public API facade."""
from __future__ import annotations

import math
from copy import copy, deepcopy
from dataclasses import dataclass, field

import numpy as np

from .citadel_types import *
```

Keep all function definitions in `citadel.py` for now — later tasks will extract them.

- [ ] **Step 3: Run import check**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "from engines.citadel import SimConfig, CitadelState, SimResult, PriceModel, _WithdrawalSource, FREQ_PPY, _SATOSHI; print('OK')"
```

- [ ] **Step 4: Run full test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5
```

Expected: 902 passed, 2 pre-existing failures, 5 skipped.

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel_types.py btc_web/engines/citadel.py
git commit -m "refactor(citadel): extract citadel_types.py — data definitions"
```

---

### Task 2: Extract `citadel_transactions.py`

**Files:**
- Create: `btc_web/engines/citadel_transactions.py`
- Modify: `btc_web/engines/citadel.py`

- [ ] **Step 1: Create `citadel_transactions.py`**

Extract from `citadel.py`: `_sell_btc_tracked` (lines 633–669), `_buy_btc_tracked` (672–682), `_sell_investments_tracked` (685–706).

```python
"""Citadel Planner — BTC and investment transaction helpers with cost basis tracking."""
from __future__ import annotations

from .citadel_types import CitadelState, SimConfig, _SATOSHI

__all__ = ["_sell_btc_tracked", "_buy_btc_tracked", "_sell_investments_tracked"]
```

Then paste the three functions verbatim. They contain lazy imports to `.tax_lots` — leave those as-is.

- [ ] **Step 2: Update `citadel.py`**

Remove the three functions from `citadel.py`. Add to the imports at top:

```python
from .citadel_transactions import *
```

- [ ] **Step 3: Run import check**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "from engines.citadel import _sell_btc_tracked, _buy_btc_tracked, _sell_investments_tracked; print('OK')"
```

- [ ] **Step 4: Run full test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel_transactions.py btc_web/engines/citadel.py
git commit -m "refactor(citadel): extract citadel_transactions.py — BTC/investment helpers"
```

---

### Task 3: Extract `citadel_waterfall.py`

**Files:**
- Create: `btc_web/engines/citadel_waterfall.py`
- Modify: `btc_web/engines/citadel.py`

- [ ] **Step 1: Create `citadel_waterfall.py`**

Extract from `citadel.py`: `_build_source_list` (243–389), `_score_sources` (392–474), `_rank_sources` (477–481), `_max_draw_before_boundary` (484–565), `_execute_draw` (568–630), `_spending_waterfall` (1170–1228).

```python
"""Citadel Planner — cost-ranked dynamic spending waterfall."""
from __future__ import annotations

from .citadel_types import (
    _WithdrawalSource, CitadelState, SimConfig, PriceModel, FREQ_PPY,
)
from .citadel_transactions import _sell_btc_tracked, _sell_investments_tracked

__all__ = [
    "_build_source_list", "_score_sources", "_rank_sources",
    "_max_draw_before_boundary", "_execute_draw", "_spending_waterfall",
]
```

Then paste the six functions verbatim. `_build_source_list` has a call to `_rmd_start_age` — this needs to be a lazy import since `_rmd_start_age` will live in `citadel_tax_integration.py` (extracted in Task 6). Replace the bare call with:

```python
from .citadel_tax_integration import _rmd_start_age
```

Add this as a lazy import inside `_build_source_list` where `_rmd_start_age` is called (inside the `if config.birth_year:` block around the `_td_horizon` computation).

`_score_sources` calls `_get_state_rate(config)` which will live in `citadel_tax_integration.py`. Add a lazy import inside `_score_sources`:

```python
from .citadel_tax_integration import _get_state_rate
```

Place this at the top of the `_score_sources` function body, alongside the existing lazy imports from `.tax` and `.tax_data`.

- [ ] **Step 2: Update `citadel.py`**

Remove the six functions. Add import:

```python
from .citadel_waterfall import *
```

- [ ] **Step 3: Run import check**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "from engines.citadel import _build_source_list, _score_sources, _rank_sources, _max_draw_before_boundary, _execute_draw, _spending_waterfall; print('OK')"
```

- [ ] **Step 4: Run full test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel_waterfall.py btc_web/engines/citadel.py
git commit -m "refactor(citadel): extract citadel_waterfall.py — cost-ranked spending"
```

---

### Task 4: Extract `citadel_floors.py`

**Files:**
- Create: `btc_web/engines/citadel_floors.py`
- Modify: `btc_web/engines/citadel.py`

- [ ] **Step 1: Create `citadel_floors.py`**

Extract from `citadel.py`: `_enforce_floors` (709–769), `_distribute_to_accounts` (774–781), `_source_from_accounts` (783–857).

**Delete `_SPLIT_KEYS`** (line 772 in original — dead code, not referenced anywhere).

```python
"""Citadel Planner — floor enforcement and account distribution."""
from __future__ import annotations

from .citadel_types import CitadelState, SimConfig, FREQ_PPY
from .citadel_waterfall import _spending_waterfall
from .citadel_transactions import _sell_investments_tracked

__all__ = ["_enforce_floors", "_distribute_to_accounts", "_source_from_accounts"]
```

Paste the three functions verbatim (excluding `_SPLIT_KEYS`).

- [ ] **Step 2: Update `citadel.py`**

Remove the three functions and `_SPLIT_KEYS`. Add import:

```python
from .citadel_floors import *
```

- [ ] **Step 3: Run import check**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "from engines.citadel import _enforce_floors, _distribute_to_accounts, _source_from_accounts; print('OK')"
```

- [ ] **Step 4: Run full test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel_floors.py btc_web/engines/citadel.py
git commit -m "refactor(citadel): extract citadel_floors.py — floor enforcement"
```

---

### Task 5: Extract `citadel_rebalancing.py`

**Files:**
- Create: `btc_web/engines/citadel_rebalancing.py`
- Modify: `btc_web/engines/citadel.py`

- [ ] **Step 1: Create `citadel_rebalancing.py`**

Extract from `citadel.py`: `_execute_sell_btc` (859–870), `_execute_buy_btc` (872–885), `_evaluate_rebalancing` (887–949).

```python
"""Citadel Planner — threshold-based BTC rebalancing triggers."""
from __future__ import annotations

from .citadel_types import CitadelState, SimConfig
from .citadel_transactions import _sell_btc_tracked, _buy_btc_tracked
from .citadel_floors import _distribute_to_accounts, _source_from_accounts

__all__ = ["_evaluate_rebalancing", "_execute_sell_btc", "_execute_buy_btc"]
```

Paste the three functions verbatim.

- [ ] **Step 2: Update `citadel.py`**

Remove the three functions. Add import:

```python
from .citadel_rebalancing import *
```

- [ ] **Step 3: Run full test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add btc_web/engines/citadel_rebalancing.py btc_web/engines/citadel.py
git commit -m "refactor(citadel): extract citadel_rebalancing.py — BTC threshold triggers"
```

---

### Task 6: Extract `citadel_tax_integration.py`

**Files:**
- Create: `btc_web/engines/citadel_tax_integration.py`
- Modify: `btc_web/engines/citadel.py`

- [ ] **Step 1: Create `citadel_tax_integration.py`**

Extract from `citadel.py`: `_get_state_rate` (1097–1102), `_rmd_start_age` (1105–1109), `_compute_rmd` (1112–1167), `_pay_tax_amount` (1231–1309), `_quarterly_estimated_payment` (1312–1357), `_year_boundary_tax` (1360–1406).

```python
"""Citadel Planner — tax integration: RMDs, estimated payments, year-end true-up."""
from __future__ import annotations

from .citadel_types import CitadelState, SimConfig, FREQ_PPY
from .citadel_transactions import _sell_investments_tracked

__all__ = [
    "_get_state_rate", "_rmd_start_age", "_compute_rmd",
    "_pay_tax_amount", "_quarterly_estimated_payment", "_year_boundary_tax",
]
```

Paste the six functions verbatim. They contain lazy imports to `.tax` and `.tax_data` — leave those as-is. No circular dependency issues — `_year_boundary_tax` does not call `step()`.

- [ ] **Step 2: Update `citadel.py`**

Remove the six functions. Add import:

```python
from .citadel_tax_integration import *
```

- [ ] **Step 3: Run import check**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "from engines.citadel import _get_state_rate, _rmd_start_age, _compute_rmd, _pay_tax_amount, _quarterly_estimated_payment, _year_boundary_tax; print('OK')"
```

- [ ] **Step 4: Run full test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel_tax_integration.py btc_web/engines/citadel.py
git commit -m "refactor(citadel): extract citadel_tax_integration.py — RMDs, tax payments"
```

---

### Task 7: Extract `citadel_step.py`

**Files:**
- Create: `btc_web/engines/citadel_step.py`
- Modify: `btc_web/engines/citadel.py`

- [ ] **Step 1: Create `citadel_step.py`**

Extract from `citadel.py`: `_get_btc_price` (227–238), `_lognormal_return` (952–965), `_markov_return` (968–996), `_scf_payment_amount` (999–1009), `_scf_check_repay` (1012–1027), `step` (1409–1591).

```python
"""Citadel Planner — simulation step (one-period heartbeat)."""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from .citadel_types import CitadelState, SimConfig, PriceModel, FREQ_PPY, _SATOSHI
from .citadel_waterfall import _spending_waterfall
from .citadel_floors import _enforce_floors
from .citadel_rebalancing import _evaluate_rebalancing
from .citadel_tax_integration import (
    _compute_rmd, _quarterly_estimated_payment, _year_boundary_tax,
)
from .citadel_transactions import _sell_btc_tracked

__all__ = [
    "step", "_get_btc_price", "_lognormal_return", "_markov_return",
    "_scf_payment_amount", "_scf_check_repay",
]
```

Paste the six functions verbatim.

- [ ] **Step 2: Update `citadel.py`**

Remove the six functions. Add import:

```python
from .citadel_step import *
```

- [ ] **Step 3: Run full test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add btc_web/engines/citadel_step.py btc_web/engines/citadel.py
git commit -m "refactor(citadel): extract citadel_step.py — simulation heartbeat"
```

---

### Task 8: Extract `citadel_sim.py` and finalize facade

**Files:**
- Create: `btc_web/engines/citadel_sim.py`
- Modify: `btc_web/engines/citadel.py` (becomes facade)

- [ ] **Step 1: Create `citadel_sim.py`**

Extract from `citadel.py`: `_initial_state` (1030–1094), `_compute_n_periods` (1663–1665), `_snapshot_state` (1668–1693), `_aggregate_results` (1696–1773), `simulate` (1776–1832), `validate_config` (1835–1885).

```python
"""Citadel Planner — simulation driver and result aggregation."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .citadel_types import (
    CitadelState, SimConfig, SimResult, PriceModel, FREQ_PPY, _SATOSHI,
)
from .citadel_step import step, _get_btc_price

__all__ = [
    "simulate", "validate_config",
    "_initial_state", "_snapshot_state", "_aggregate_results", "_compute_n_periods",
]
```

Paste the six functions verbatim. They contain lazy imports to `btc_core`, `.tax`, `.tax_lots` — leave as-is.

- [ ] **Step 2: Replace `citadel.py` with facade**

Replace the entire remaining content of `citadel.py` with:

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

- [ ] **Step 3: Run full import check**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "from engines.citadel import SimConfig, simulate, step, _spending_waterfall, _enforce_floors, _WithdrawalSource, _build_source_list, _score_sources, _rank_sources, _execute_draw, _sell_btc_tracked, SimResult, CitadelState, PriceModel, FREQ_PPY, _SATOSHI, _initial_state, _evaluate_rebalancing, _lognormal_return, _pay_tax_amount, validate_config, _compute_n_periods, _get_btc_price, _buy_btc_tracked, _sell_investments_tracked, _execute_sell_btc, _execute_buy_btc, _enforce_floors, _distribute_to_accounts, _source_from_accounts, _scf_payment_amount, _scf_check_repay; print('OK')"
```

- [ ] **Step 4: Run full test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5
```

Expected: 902 passed, 2 pre-existing failures, 5 skipped.

- [ ] **Step 5: Verify no changes outside `engines/`**

```bash
git diff --name-only HEAD | grep -v 'btc_web/engines/'
```

Expected: no output (only engines/ files changed).

- [ ] **Step 6: Verify file sizes**

```bash
wc -l btc_web/engines/citadel*.py | sort -n
```

Expected: facade ~10 lines, every module under 450 lines.

- [ ] **Step 7: Commit**

```bash
git add btc_web/engines/citadel_sim.py btc_web/engines/citadel.py
git commit -m "refactor(citadel): extract citadel_sim.py + finalize facade

engines/citadel.py is now a re-export facade. All implementation
lives in 8 focused citadel_*.py modules, each under 450 lines."
```

---

### Task 9: End-to-end verification

Verify the complete split works with both tax-enabled and non-tax simulations.

- [ ] **Step 1: Verify tax-enabled simulation works**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "
from engines.citadel import SimConfig, simulate
cfg = SimConfig(tax_enabled=True, start_yr=2031, end_yr=2033, freq='Annually',
                monthly_spend=5000, selected_qs=[0.25], start_stack=1.0,
                cash_initial=100000, filing_status='single', state_code='TX',
                td_cash_initial=50000)

class M:
    def __init__(self):
        import pandas as pd
        self.fits = {0.25: {'slope': 5.0, 'intercept': 2.0}}
        self.genesis = pd.Timestamp('2009-07-25')
    def price_at(self, q, t): return 50000.0 * (1 + t/100)
    def quantile_at(self, price, t): return 0.5

r = simulate(cfg, M())
print(f'OK — {r.total_usd.shape[1]} periods, final=\${r.total_usd[0,-1]:,.0f}')
"
```

- [ ] **Step 2: Verify non-tax simulation works**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "
from engines.citadel import SimConfig, simulate
cfg = SimConfig(tax_enabled=False, start_yr=2031, end_yr=2035, freq='Annually',
                monthly_spend=5000, selected_qs=[0.25], start_stack=1.0,
                cash_initial=100000)

class M:
    def __init__(self):
        import pandas as pd
        self.fits = {0.25: {'slope': 5.0, 'intercept': 2.0}}
        self.genesis = pd.Timestamp('2009-07-25')
    def price_at(self, q, t): return 50000.0 * (1 + t/100)
    def quantile_at(self, price, t): return 0.5

r = simulate(cfg, M())
print(f'OK — {r.total_usd.shape[1]} periods, final=\${r.total_usd[0,-1]:,.0f}')
"
```

- [ ] **Step 3: Verify lazy imports resolve correctly**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "
# Verify the two lazy cross-module imports work:
# 1. _score_sources -> _get_state_rate (waterfall -> tax_integration)
# 2. _build_source_list -> _rmd_start_age (waterfall -> tax_integration)
from engines.citadel import (CitadelState, SimConfig, _build_source_list,
                              _score_sources)
from engines.tax import TaxYearAccumulator
state = CitadelState(
    cash=50000, td_cash=50000, td_reserves=[0,0,0], td_investments=[0,0],
    sim_date='2035-06-15', tax_year_accum=TaxYearAccumulator(),
)
cfg = SimConfig(tax_enabled=True, birth_year=1985, start_yr=2035,
                state_code='TX', filing_status='single', inflation=4.0,
                reserve_bins=[
                    {'label': 'S', 'initial': 0, 'rate': 5.0, 'volatility': 0},
                    {'label': 'M', 'initial': 0, 'rate': 4.5, 'volatility': 0},
                    {'label': 'L', 'initial': 0, 'rate': 4.0, 'volatility': 0},
                ],
                invest_bins=[
                    {'label': 'Eq', 'initial': 0, 'return_rate': 10.0, 'volatility': 0},
                    {'label': 'Bd', 'initial': 0, 'return_rate': 5.0, 'volatility': 0},
                ])
sources = _build_source_list(state, cfg, model=None)
_score_sources(sources, state, cfg, model=None)
print(f'OK — {len(sources)} sources scored')
"
```

- [ ] **Step 4: Run full test suite one final time**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5
```

Expected: 902 passed, 2 pre-existing failures, 5 skipped.
