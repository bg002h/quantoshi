# Unified Citadel MC — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Citadel engine to run N simulation paths with stochastic Markov returns on ALL asset classes (equities, bonds, treasuries) across all three wrappers (taxable, tax-deferred, tax-free), and aggregate results into 7-percentile bands across 11 output series.

**Architecture:** 7 tasks, bottom-up: (1) Add initial regime fields to SimConfig, (2) Wire initial regimes through `_initial_state()` + add TD/TF regime fields, (3) Replace the `is_deterministic` guard, (4) Apply Markov returns to TD/TF wrappers with independent regime tracking, (5) Band aggregation module, (6) Verify dev bypass for MC payment, (7) Integration tests. Each task produces a working, testable commit.

**Tech Stack:** Python 3.14, dataclasses, numpy

**Spec:** `docs/superpowers/specs/2026-03-31-unified-citadel-mc-design.md`

**Test command:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -x -q --tb=short`

**Full suite:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `btc_web/engines/citadel_bands.py` | Band aggregation: 7 percentiles × 11 series from SimResult |

### Modified Files
| File | Change |
|------|--------|
| `btc_web/engines/citadel_types.py` | Add `initial_*_regime` fields to SimConfig; add `td_*_regime` and `tf_*_regime` fields to CitadelState |
| `btc_web/engines/citadel_sim.py` | Wire initial regimes through `_initial_state()`; expand `_aggregate_results()` to 7 percentiles and 11 series |
| `btc_web/engines/citadel_step.py` | Replace `is_deterministic` guard with `n_sims > 1`; add Markov path for TD/TF wrappers |
| `btc_web/engines/citadel.py` | Re-export `citadel_bands` |
| `btc_web/test_web.py` | All new tests (added to existing test file) |

---

### Task 1: Add initial regime fields to SimConfig

**Files:**
- Modify: `btc_web/engines/citadel_types.py:156-160`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write failing test for initial regime fields**

```python
class TestInitialRegimeConfig:
    def test_default_initial_regimes_are_neutral(self):
        from engines.citadel_types import SimConfig
        cfg = SimConfig()
        assert cfg.initial_equity_regime == 2
        assert cfg.initial_bond_regime == 2
        assert cfg.initial_res_short_regime == 2
        assert cfg.initial_res_med_regime == 2
        assert cfg.initial_res_long_regime == 2

    def test_initial_regimes_are_configurable(self):
        from engines.citadel_types import SimConfig
        cfg = SimConfig(
            initial_equity_regime=0,
            initial_bond_regime=4,
            initial_res_short_regime=1,
            initial_res_med_regime=3,
            initial_res_long_regime=0,
        )
        assert cfg.initial_equity_regime == 0
        assert cfg.initial_bond_regime == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestInitialRegimeConfig -x -q --tb=short`
Expected: FAIL with "unexpected keyword argument 'initial_equity_regime'"

- [ ] **Step 3: Add initial regime fields to SimConfig**

In `btc_web/engines/citadel_types.py`, add after the `asset_matrices` field (after line 160):

```python
    # Starting regime bins for Markov model (0=bearish, 2=neutral, 4=bullish)
    # Used by _initial_state() to seed CitadelState regime fields.
    # Macro regime presets (Bear/Neutral/Bull) set all five to the same bin.
    initial_equity_regime: int = 2
    initial_bond_regime: int = 2
    initial_res_short_regime: int = 2
    initial_res_med_regime: int = 2
    initial_res_long_regime: int = 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestInitialRegimeConfig -x -q --tb=short`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5`
Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add btc_web/engines/citadel_types.py btc_web/test_web.py
git commit -m "feat(citadel): add initial_*_regime fields to SimConfig"
```

---

### Task 2: Add TD/TF regime fields to CitadelState and wire _initial_state()

**Files:**
- Modify: `btc_web/engines/citadel_types.py:190-194`
- Modify: `btc_web/engines/citadel_sim.py:17-81`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write failing tests**

```python
class TestInitialRegimeWiring:
    def test_td_tf_regime_fields_exist(self):
        from engines.citadel_types import CitadelState
        state = CitadelState()
        assert state.td_equity_regime == 2
        assert state.td_bond_regime == 2
        assert state.td_res_short_regime == 2
        assert state.td_res_med_regime == 2
        assert state.td_res_long_regime == 2
        assert state.tf_equity_regime == 2
        assert state.tf_bond_regime == 2
        assert state.tf_res_short_regime == 2
        assert state.tf_res_med_regime == 2
        assert state.tf_res_long_regime == 2

    def test_initial_state_uses_config_regimes(self):
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import _initial_state
        cfg = SimConfig()
        cfg.initial_equity_regime = 0
        cfg.initial_bond_regime = 4
        cfg.initial_res_short_regime = 1
        cfg.initial_res_med_regime = 3
        cfg.initial_res_long_regime = 0
        state = _initial_state(cfg, model=None)
        assert state.equity_regime == 0
        assert state.bond_regime == 4
        assert state.res_short_regime == 1
        assert state.res_med_regime == 3
        assert state.res_long_regime == 0

    def test_initial_state_seeds_td_tf_regimes_unconditionally(self):
        """TD/TF regimes seeded from config regardless of tax_enabled."""
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import _initial_state
        cfg = SimConfig()
        cfg.tax_enabled = False  # TD/TF regimes still seeded
        cfg.initial_equity_regime = 4
        cfg.initial_bond_regime = 0
        state = _initial_state(cfg, model=None)
        assert state.td_equity_regime == 4
        assert state.td_bond_regime == 0
        assert state.tf_equity_regime == 4
        assert state.tf_bond_regime == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestInitialRegimeWiring -x -q --tb=short`
Expected: FAIL — `td_equity_regime` not a field on CitadelState

- [ ] **Step 3: Add TD/TF regime fields to CitadelState**

In `btc_web/engines/citadel_types.py`, add after `res_long_regime` (after line 194):

```python
    # TD wrapper regime states (independent from taxable regimes)
    td_equity_regime: int = 2
    td_bond_regime: int = 2
    td_res_short_regime: int = 2
    td_res_med_regime: int = 2
    td_res_long_regime: int = 2
    # TF wrapper regime states (independent from taxable regimes)
    tf_equity_regime: int = 2
    tf_bond_regime: int = 2
    tf_res_short_regime: int = 2
    tf_res_med_regime: int = 2
    tf_res_long_regime: int = 2
```

- [ ] **Step 4: Wire _initial_state() to read config regimes**

In `btc_web/engines/citadel_sim.py`, in `_initial_state()`, add after CitadelState construction (after line ~37, before SCF block):

```python
    # Seed taxable-wrapper regimes from config
    state.equity_regime = config.initial_equity_regime
    state.bond_regime = config.initial_bond_regime
    state.res_short_regime = config.initial_res_short_regime
    state.res_med_regime = config.initial_res_med_regime
    state.res_long_regime = config.initial_res_long_regime
```

Also add (unconditionally, NOT inside `if config.tax_enabled:` — TD/TF regime fields exist on CitadelState regardless of tax mode, and Markov may apply to TD/TF assets even without the tax layer):

```python
    # Seed TD/TF wrapper regimes from config (same starting point as taxable)
    state.td_equity_regime = config.initial_equity_regime
    state.td_bond_regime = config.initial_bond_regime
    state.td_res_short_regime = config.initial_res_short_regime
    state.td_res_med_regime = config.initial_res_med_regime
    state.td_res_long_regime = config.initial_res_long_regime
    state.tf_equity_regime = config.initial_equity_regime
    state.tf_bond_regime = config.initial_bond_regime
    state.tf_res_short_regime = config.initial_res_short_regime
    state.tf_res_med_regime = config.initial_res_med_regime
    state.tf_res_long_regime = config.initial_res_long_regime
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestInitialRegimeWiring -x -q --tb=short`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5`

- [ ] **Step 7: Commit**

```bash
git add btc_web/engines/citadel_types.py btc_web/engines/citadel_sim.py btc_web/test_web.py
git commit -m "feat(citadel): add TD/TF regime fields, wire _initial_state() to config regimes"
```

---

### Task 3: Replace is_deterministic guard with n_sims > 1 check

**Files:**
- Modify: `btc_web/engines/citadel_step.py:135,141-143`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write failing test**

```python
class TestMarkovGuard:
    def _make_markov_config(self, n_sims=10):
        import numpy as np
        from engines.citadel_types import SimConfig
        matrices = {}
        for key in ("equity", "bond", "tres_short", "tres_med", "tres_long"):
            n_bins = 5
            trans = np.ones((n_bins, n_bins)) / n_bins
            bin_means = np.array([-0.02, -0.005, 0.005, 0.01, 0.02])
            bin_vols = np.array([0.01, 0.005, 0.003, 0.005, 0.01])
            matrices[key] = {"trans": trans, "bin_means": bin_means, "bin_vols": bin_vols}
        cfg = SimConfig()
        cfg.asset_return_model = "markov"
        cfg.asset_matrices = matrices
        cfg.n_sims = n_sims
        return cfg

    def test_markov_fires_when_n_sims_gt_1(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=10)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=None)
        for _ in range(20):
            state = step(state, cfg, 100_000.0, rng, model=None)
        regimes = [state.equity_regime, state.bond_regime,
                   state.res_short_regime, state.res_med_regime, state.res_long_regime]
        assert any(r != 2 for r in regimes), "After 20 Markov steps, at least one regime should change"

    def test_markov_does_not_fire_when_n_sims_1(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=1)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=None)
        state = step(state, cfg, 100_000.0, rng, model=None)
        assert state.equity_regime == 2
        assert state.bond_regime == 2
```

- [ ] **Step 2: Run test to verify behavior**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestMarkovGuard -x -q --tb=short`
Expected: `test_markov_fires_when_n_sims_gt_1` FAILS (old guard blocks Markov even with n_sims>1 when is_deterministic check is different)

- [ ] **Step 3: Replace the guard**

In `btc_web/engines/citadel_step.py`:

1. Replace line 135 (`is_deterministic = (config.n_sims == 1)`) with:
```python
    deterministic = (config.n_sims == 1)
```

2. Replace lines 141-143 with:
```python
    use_markov = (config.asset_return_model == "markov"
                  and config.asset_matrices is not None
                  and config.n_sims > 1)
```

3. Replace all remaining `is_deterministic` references with `deterministic` (same variable, just renamed). **Note:** This includes references inside the TD/TF block (lines ~196-209) which Task 4 will replace entirely. That's fine — Task 3 renames them, Task 4 replaces the whole block.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestMarkovGuard -x -q --tb=short`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5`

- [ ] **Step 6: Commit**

```bash
git add btc_web/engines/citadel_step.py btc_web/test_web.py
git commit -m "feat(citadel): replace is_deterministic guard with n_sims > 1 for Markov activation"
```

---

### Task 4: Apply Markov returns to TD/TF wrappers

**Files:**
- Modify: `btc_web/engines/citadel_step.py:189-211`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write failing tests**

```python
class TestTdTfMarkovReturns:
    def _make_markov_config(self, n_sims=10):
        import numpy as np
        from engines.citadel_types import SimConfig
        matrices = {}
        for key in ("equity", "bond", "tres_short", "tres_med", "tres_long"):
            n_bins = 5
            trans = np.ones((n_bins, n_bins)) / n_bins
            bin_means = np.array([-0.02, -0.005, 0.005, 0.01, 0.02])
            bin_vols = np.array([0.01, 0.005, 0.003, 0.005, 0.01])
            matrices[key] = {"trans": trans, "bin_means": bin_means, "bin_vols": bin_vols}
        cfg = SimConfig()
        cfg.asset_return_model = "markov"
        cfg.asset_matrices = matrices
        cfg.n_sims = n_sims
        cfg.tax_enabled = True
        return cfg

    def test_td_regimes_evolve_under_markov(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=10)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=None)
        for _ in range(30):
            state = step(state, cfg, 100_000.0, rng, model=None)
        td_regimes = [state.td_equity_regime, state.td_bond_regime,
                      state.td_res_short_regime, state.td_res_med_regime,
                      state.td_res_long_regime]
        assert any(r != 2 for r in td_regimes), "TD regimes should evolve under Markov"

    def test_tf_regimes_evolve_under_markov(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=10)
        rng = np.random.default_rng(99)
        state = _initial_state(cfg, model=None)
        for _ in range(30):
            state = step(state, cfg, 100_000.0, rng, model=None)
        tf_regimes = [state.tf_equity_regime, state.tf_bond_regime]
        assert any(r != 2 for r in tf_regimes), "TF regimes should evolve under Markov"

    def test_td_tf_use_lognormal_when_n_sims_1(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=1)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=None)
        state = step(state, cfg, 100_000.0, rng, model=None)
        assert state.td_equity_regime == 2, "TD regimes unchanged when n_sims=1"
        assert state.tf_equity_regime == 2, "TF regimes unchanged when n_sims=1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestTdTfMarkovReturns -x -q --tb=short`
Expected: FAIL — TD/TF regimes stay at 2 because current code uses lognormal only

- [ ] **Step 3: Replace TD/TF growth block with Markov-aware version**

In `btc_web/engines/citadel_step.py`, replace the TD/TF wrapper growth section (the `# 2b. TD/TF wrapper growth` block) with the following Markov-aware version. This mirrors the taxable wrapper pattern but tracks `td_*_regime` / `tf_*_regime` independently:

```python
    # 2b. TD/TF wrapper growth (same return model as taxable wrapper)
    # Guard: only runs when tax_enabled=True. When tax_enabled=False,
    # TD/TF balances are zero (not initialized by _initial_state()),
    # so growing them has no effect. The regime fields ARE seeded
    # unconditionally (Task 2), but that's harmless — they just sit unused.
    if config.tax_enabled:
        cash_growth = (1 + config.cash_rate / 100) ** (1.0 / ppy)
        new.td_cash *= cash_growth
        new.tf_cash *= cash_growth

        if use_markov:
            am = config.asset_matrices
            _res_keys = ["tres_short", "tres_med", "tres_long"]
            _inv_keys = ["equity", "bond"]

            # TD reserves
            for i, mkey in enumerate(_res_keys):
                rattr = f"td_res_{'short' if i == 0 else 'med' if i == 1 else 'long'}_regime"
                if mkey in am:
                    ret, nr = _markov_return(am[mkey], getattr(new, rattr), rng)
                    setattr(new, rattr, nr)
                    if i < len(new.td_reserves):
                        new.td_reserves[i] *= (1 + ret)
                elif i < len(new.td_reserves):
                    rb = config.reserve_bins[i]
                    r = _lognormal_return(rb["rate"] / 100, rb["volatility"] / 100, ppy,
                                          deterministic=deterministic, rng=rng)
                    new.td_reserves[i] *= (1 + r)

            # TF reserves
            for i, mkey in enumerate(_res_keys):
                rattr = f"tf_res_{'short' if i == 0 else 'med' if i == 1 else 'long'}_regime"
                if mkey in am:
                    ret, nr = _markov_return(am[mkey], getattr(new, rattr), rng)
                    setattr(new, rattr, nr)
                    if i < len(new.tf_reserves):
                        new.tf_reserves[i] *= (1 + ret)
                elif i < len(new.tf_reserves):
                    rb = config.reserve_bins[i]
                    r = _lognormal_return(rb["rate"] / 100, rb["volatility"] / 100, ppy,
                                          deterministic=deterministic, rng=rng)
                    new.tf_reserves[i] *= (1 + r)

            # TD investments
            for i, mkey in enumerate(_inv_keys):
                rattr = f"td_{'equity' if i == 0 else 'bond'}_regime"
                if mkey in am:
                    ret, nr = _markov_return(am[mkey], getattr(new, rattr), rng)
                    setattr(new, rattr, nr)
                    if i < len(new.td_investments):
                        new.td_investments[i] *= (1 + ret)
                elif i < len(new.td_investments):
                    ib = config.invest_bins[i]
                    r = _lognormal_return(ib["return_rate"] / 100, ib["volatility"] / 100, ppy,
                                          deterministic=deterministic, rng=rng)
                    new.td_investments[i] *= (1 + r)

            # TF investments
            for i, mkey in enumerate(_inv_keys):
                rattr = f"tf_{'equity' if i == 0 else 'bond'}_regime"
                if mkey in am:
                    ret, nr = _markov_return(am[mkey], getattr(new, rattr), rng)
                    setattr(new, rattr, nr)
                    if i < len(new.tf_investments):
                        new.tf_investments[i] *= (1 + ret)
                elif i < len(new.tf_investments):
                    ib = config.invest_bins[i]
                    r = _lognormal_return(ib["return_rate"] / 100, ib["volatility"] / 100, ppy,
                                          deterministic=deterministic, rng=rng)
                    new.tf_investments[i] *= (1 + r)
        else:
            # Lognormal returns for TD/TF (original behavior)
            for i, rb in enumerate(config.reserve_bins):
                r = _lognormal_return(rb["rate"] / 100, rb["volatility"] / 100, ppy,
                                      deterministic=deterministic, rng=rng)
                if i < len(new.td_reserves):
                    new.td_reserves[i] *= (1 + r)
                r2 = _lognormal_return(rb["rate"] / 100, rb["volatility"] / 100, ppy,
                                       deterministic=deterministic, rng=rng)
                if i < len(new.tf_reserves):
                    new.tf_reserves[i] *= (1 + r2)
            for i, ib in enumerate(config.invest_bins):
                r = _lognormal_return(ib["return_rate"] / 100, ib["volatility"] / 100, ppy,
                                      deterministic=deterministic, rng=rng)
                if i < len(new.td_investments):
                    new.td_investments[i] *= (1 + r)
                r2 = _lognormal_return(ib["return_rate"] / 100, ib["volatility"] / 100, ppy,
                                       deterministic=deterministic, rng=rng)
                if i < len(new.tf_investments):
                    new.tf_investments[i] *= (1 + r2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestTdTfMarkovReturns -x -q --tb=short`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5`

- [ ] **Step 6: Commit**

```bash
git add btc_web/engines/citadel_step.py btc_web/test_web.py
git commit -m "feat(citadel): apply Markov returns to TD/TF wrappers with independent regime tracking"
```

---

### Task 5: Band aggregation module

**Files:**
- Create: `btc_web/engines/citadel_bands.py`
- Modify: `btc_web/engines/citadel_sim.py`
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write failing tests**

```python
class TestBandAggregation:
    def test_compute_bands_returns_7_percentiles(self):
        from engines.citadel_bands import compute_bands, BAND_PERCENTILES
        assert BAND_PERCENTILES == (5, 10, 25, 50, 75, 90, 95)

    def test_compute_bands_returns_11_series(self):
        from engines.citadel_bands import BAND_SERIES
        assert len(BAND_SERIES) == 11
        assert "total" in BAND_SERIES
        assert "btc_stack" in BAND_SERIES
        assert "td_total" in BAND_SERIES
        assert "tf_total" in BAND_SERIES
        assert "depletion" in BAND_SERIES

    def test_band_ordering(self):
        """P5 <= P25 <= P50 <= P75 <= P95 for total portfolio."""
        import numpy as np
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands
        cfg = SimConfig()
        cfg.start_yr = 2031; cfg.end_yr = 2032
        paths = np.array([[20000 + i * 30000 + j * 100 for j in range(12)]
                          for i in range(20)])
        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)
        for t in range(12):
            vals = [bands[p]["total"][t] for p in [5, 25, 50, 75, 95]]
            for k in range(len(vals) - 1):
                assert vals[k] <= vals[k + 1] + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestBandAggregation -x -q --tb=short`
Expected: FAIL — `citadel_bands` module doesn't exist

- [ ] **Step 3: Create `btc_web/engines/citadel_bands.py`**

```python
"""Citadel Planner — percentile band aggregation from simulation results.

Computes 7 percentile levels (P5/P10/P25/P50/P75/P90/P95) across 11 output
series from a multi-sim SimResult.
"""
from __future__ import annotations

import numpy as np

from .citadel_types import SimResult

__all__ = ["compute_bands", "BAND_PERCENTILES", "BAND_SERIES"]

BAND_PERCENTILES = (5, 10, 25, 50, 75, 90, 95)

BAND_SERIES = (
    "total",              # Total portfolio (USD)
    "btc_stack",          # BTC stack (native BTC units)
    "btc_usd",            # BTC holdings (USD)
    "cash",               # Cash (USD)
    "reserves_total",     # Reserves total (USD)
    "investments_total",  # Investments total (USD)
    "td_total",           # Tax-deferred total (USD)
    "tf_total",           # Tax-free total (USD)
    "cumulative_spend",   # Cumulative spending (USD)
    "taxes_paid",         # Cumulative taxes paid (USD)
    "depletion",          # Depletion fraction (0-1 per step)
)


def compute_bands(result: SimResult) -> dict[int, dict[str, np.ndarray]]:
    """Compute percentile bands from a SimResult.

    Returns dict keyed by percentile (5..95), each value a dict of
    series_name -> ndarray of shape (n_periods,).
    """
    n_sims = result.total_usd.shape[0]
    n_periods = result.total_usd.shape[1]
    _zero = np.zeros((n_sims, n_periods))

    series_data = {
        "total": result.total_usd,
        "btc_stack": result.btc_holdings,
        "btc_usd": result.btc_holdings * result.btc_prices,
        "cash": result.cash_balances,
        "reserves_total": result.reserve_balances.sum(axis=2),
        "investments_total": result.invest_balances.sum(axis=2),
        "td_total": result.td_total if result.td_total is not None else _zero,
        "tf_total": result.tf_total if result.tf_total is not None else _zero,
        "cumulative_spend": result.cumulative_spend,
        "taxes_paid": result.taxes_paid if result.taxes_paid is not None else _zero,
    }

    depletion_frac = (result.total_usd <= 0).astype(np.float64).mean(axis=0)

    bands: dict[int, dict[str, np.ndarray]] = {}
    for pct in BAND_PERCENTILES:
        band = {key: np.percentile(data, pct, axis=0)
                for key, data in series_data.items()}
        band["depletion"] = depletion_frac
        bands[pct] = band
    return bands
```

- [ ] **Step 4: Add re-export to facade**

In `btc_web/engines/citadel.py`, add:

```python
from .citadel_bands import compute_bands, BAND_PERCENTILES, BAND_SERIES  # noqa: F401
```

- [ ] **Step 5: Expand _aggregate_results() in citadel_sim.py**

Update the percentile and median computation in `_aggregate_results()` to use all 7 percentile levels, include all 11 series, and expand `median` to match. Replace the existing percentile/median block with:

```python
    _zero = np.zeros((n_sims, n_periods)) if n_sims > 0 else np.zeros((1, n_periods))
    _btc_usd = btc_h * btc_p
    _res_total = res_b.sum(axis=2)
    _inv_total = inv_b.sum(axis=2)
    _td = td_total_arr if td_total_arr is not None else np.zeros((n_sims, n_periods))
    _tf = tf_total_arr if tf_total_arr is not None else np.zeros((n_sims, n_periods))
    _tax = taxes_paid_arr if taxes_paid_arr is not None else np.zeros((n_sims, n_periods))

    # Build series dict for consistent iteration
    series = {
        "total": total, "btc_stack": btc_h, "btc_usd": _btc_usd,
        "cash": cash_b, "reserves_total": _res_total,
        "investments_total": _inv_total,
        "td_total": _td, "tf_total": _tf,
        "cumulative_spend": cum_spend, "taxes_paid": _tax,
    }

    # Median across all 10 numeric series
    median = {k: np.median(v, axis=0) for k, v in series.items()}
    # Depletion: fraction of sims depleted at each step (not a percentile)
    median["depletion"] = (total <= 0).astype(np.float64).mean(axis=0)

    # Percentiles (7 levels) across all 10 numeric series
    depletion_frac = median["depletion"]  # same for all percentile levels
    percentiles = {}
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        pct_dict = {k: np.percentile(v, pct, axis=0) for k, v in series.items()}
        pct_dict["depletion"] = depletion_frac  # fraction, not percentile
        percentiles[pct] = pct_dict
```

**Note on `depletion`:** Depletion is stored as the fraction of sims depleted at each time step — the same value in every percentile bucket. It is not a percentile itself. Phase 3 will render it as a single line or annotation, not as a band. This is semantically correct: "30% of sims are depleted by year 15" is a single number, not a spread.

- [ ] **Step 6: Run test to verify it passes**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestBandAggregation -x -q --tb=short`
Expected: PASS

- [ ] **Step 7: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5`
Note: Existing tests that check `result.percentiles` keys `{5, 25, 75, 95}` will still pass since those keys are preserved in the expanded set.

- [ ] **Step 8: Commit**

```bash
git add btc_web/engines/citadel_bands.py btc_web/engines/citadel_sim.py btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "feat(citadel): add citadel_bands.py, expand aggregation to 7 percentiles x 11 series"
```

---

### Task 6: Verify dev bypass for MC payment

**Files:**
- Test: `btc_web/test_web.py`

No code changes needed — the dev bypass already exists in `btc_web/callbacks/mc_payment.py`. Add a test to document it.

- [ ] **Step 1: Write test**

```python
class TestDevBypass:
    def test_dev_bypass_exists_in_mc_payment(self):
        import inspect
        from callbacks import mc_payment
        source = inspect.getsource(mc_payment)
        assert "DEV" in source, "mc_payment should check DEV env var for bypass"
```

- [ ] **Step 2: Run test**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestDevBypass -x -q --tb=short`

- [ ] **Step 3: Commit**

```bash
git add btc_web/test_web.py
git commit -m "test(citadel): document DEV=1 payment bypass"
```

---

### Task 7: Integration tests — full multi-asset MC

**Files:**
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write integration tests**

```python
class TestUnifiedMcIntegration:
    def _make_matrices(self):
        import numpy as np
        matrices = {}
        for key in ("equity", "bond", "tres_short", "tres_med", "tres_long"):
            n_bins = 5
            trans = np.full((n_bins, n_bins), 0.05)
            np.fill_diagonal(trans, 0.80)
            trans /= trans.sum(axis=1, keepdims=True)
            bin_means = np.array([-0.03, -0.01, 0.005, 0.015, 0.03])
            bin_vols = np.array([0.015, 0.008, 0.005, 0.008, 0.015])
            matrices[key] = {"trans": trans, "bin_means": bin_means, "bin_vols": bin_vols}
        return matrices

    def test_full_mc_20_sims_produces_spread(self):
        import numpy as np
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import simulate
        cfg = SimConfig()
        cfg.start_yr = 2031; cfg.end_yr = 2033
        cfg.asset_return_model = "markov"
        cfg.asset_matrices = self._make_matrices()
        cfg.initial_equity_regime = 4  # Bull
        cfg.initial_bond_regime = 0    # Bear
        rng = np.random.default_rng(123)
        base = np.linspace(50000, 150000, 24)
        paths = np.array([base * (1 + rng.normal(0, 0.1, 24)) for _ in range(20)])
        result = simulate(cfg, model=None, price_paths=paths)
        assert result.total_usd.shape == (20, 24)
        assert set(result.percentiles.keys()) == {5, 10, 25, 50, 75, 90, 95}
        p5 = result.percentiles[5]["total"]
        p95 = result.percentiles[95]["total"]
        assert np.any(p95 > p5 + 1.0), "MC should produce nonzero spread"

    def test_deterministic_unchanged(self):
        """n_sims=1 with a single price path: all percentiles identical."""
        import numpy as np
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import simulate
        cfg = SimConfig()
        cfg.start_yr = 2031; cfg.end_yr = 2033; cfg.n_sims = 1
        # Must provide price_paths (model=None can't generate deterministic prices)
        paths = np.array([[80000 + j * 200 for j in range(24)]])  # 1 sim, 24 months
        result = simulate(cfg, model=None, price_paths=paths)
        assert result.total_usd.shape[0] == 1
        for key in ["total", "btc_usd", "cash"]:
            np.testing.assert_array_almost_equal(
                result.percentiles[5][key], result.percentiles[95][key],
                err_msg=f"Deterministic: P5 should equal P95 for {key}")

    def test_bands_match_standalone_compute(self):
        import numpy as np
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands
        cfg = SimConfig()
        cfg.start_yr = 2031; cfg.end_yr = 2032
        paths = np.array([[50000 + i * 10000 + j * 100 for j in range(12)]
                          for i in range(10)])
        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)
        for pct in [5, 50, 95]:
            np.testing.assert_array_almost_equal(
                bands[pct]["total"], result.percentiles[pct]["total"])
```

- [ ] **Step 2: Run integration tests**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestUnifiedMcIntegration -x -q --tb=short`

- [ ] **Step 3: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5`

- [ ] **Step 4: Commit**

```bash
git add btc_web/test_web.py
git commit -m "test(citadel): integration tests for unified multi-asset MC Phase 1"
```

---

## Verification Checklist

After all 7 tasks, run:

```bash
# Full test suite
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -10

# Import check
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "
from engines.citadel import SimConfig, CitadelState, simulate
from engines.citadel_bands import compute_bands, BAND_PERCENTILES, BAND_SERIES
cfg = SimConfig()
print(f'initial_equity_regime: {cfg.initial_equity_regime}')
state = CitadelState()
print(f'td_equity_regime: {state.td_equity_regime}')
print(f'tf_equity_regime: {state.tf_equity_regime}')
print(f'BAND_PERCENTILES: {BAND_PERCENTILES}')
print(f'BAND_SERIES count: {len(BAND_SERIES)}')
print('Phase 1 OK')
"
```
