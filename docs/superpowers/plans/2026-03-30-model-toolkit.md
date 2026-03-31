# Model Toolkit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor notebook computation into `tools/model_toolkit/`, replacing exec-based build scripts with proper Python module imports.

**Architecture:** Extract computation from `sp_stripped.ipynb` cells into 8 focused modules under `tools/model_toolkit/`. Both `build_bm_model.py` and `build_ef_model.py` become thin recipes that import and compose these modules. `fit_sigma.py` is absorbed into `bands.py`.

**Tech Stack:** Python 3.14, numpy, scipy, statsmodels, pandas

**Spec:** `docs/superpowers/specs/2026-03-30-model-toolkit-design.md`

**Branch:** `SPReplace`

**Source of truth:** `sp_stripped.ipynb` (3 cells of computation-only code). Read cell source with:

```bash
btc_venv/bin/python3 -c "
import json
with open('sp_stripped.ipynb') as f:
    nb = json.load(f)
src = ''.join(nb['cells'][CELL]['source'])
for i, line in enumerate(src.split('\n')[START-1:END], START):
    print(f'{i:4d}: {line}')
"
```

**Test command:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --ignore=btc_web/test_tax_e2e.py --tb=short`

**Baseline:** 889 passed, 0 failed, 5 skipped

---

## ERRATA — Read Before Implementing

These corrections override specific details in the tasks below. The notebook source (`sp_stripped.ipynb`) is the ground truth — when in doubt, read the actual code.

### E1: `bubble_shape` uses `slope_sup` as a closure variable (CRITICAL)

In the notebook, `bubble_shape()` references `slope_sup` (the support line slope) as a global/closure variable at Cell 0 lines ~396, 408, 419. When extracted to `bubble_shape.py`, this global won't exist.

**Fix:** Add `slope_sup` as a parameter to `bubble_shape`:

```python
def bubble_shape(t, t_rise, r, t_plateau, t_decay, d, plat_pow=0.0, slope_sup=0.0):
```

Every caller must pass `support.slope` (or `support.B`). This affects:
- `fitting.py`: all `bubble_shape()` calls inside `fit_bubble` and `fit_sequential`
- `composite.py`: all `bubble_shape()` calls inside `_total_bubble` and `build_comp_by_n`
- Any test code that calls `bubble_shape` directly

### E2: `fit_manual_bubble` globals not enumerated (CRITICAL)

The notebook's `fit_manual_bubble` (Cell 0 lines 507-635) closes over many globals beyond `slope_sup`. These must ALL be threaded through `config` or as explicit parameters:

- `slope_sup` — via `bubble_shape` calls and directly for K_peak computation
- `years_fit` — use `price_data.years` instead
- `FIT_CONTEXT_YR = 1.0` — add to `DEFAULT_CONFIG`
- `FIT_RISE_LOOKBACK_YR = 0.75` — add to `DEFAULT_CONFIG`
- `PLATEAU_PARALLEL_SUPPORT = True` — already in `DEFAULT_CONFIG`
- `PLAT_POW_RANGE = 8.0` — add to `DEFAULT_CONFIG`
- `DE_MAXITER = 2000` — already in `DEFAULT_CONFIG`
- `DE_POPSIZE = 18` — **NOT 30** (see E5)
- `genesis` — use `pd.Timestamp(genesis_date)` from config or hardcode "2009-07-25"

The implementer must read Cell 0 lines 507-635 line by line and identify every global reference.

### E3: DE seed uses original peak index, not magnitude rank (CRITICAL)

The notebook's fitting loop (Cell 0 ~line 620) passes `orig_idx` to `fit_manual_bubble`, where `orig_idx` is the peak's position in the **discovery order** (order of `BUBBLE_YEARS`), NOT the magnitude-sorted rank.

**Fix:** `fit_sequential` must track the original peak index and pass it as the seed index. Do NOT use the magnitude-sorted position.

### E4: `N_MAJOR = 5`, not 2 (IMPORTANT)

The notebook uses `N_MAJOR = 5` (Cell 0 line 128). With 5 bubble years and `N_MAJOR=5`, ALL bubbles are classified as major. The plan's `classify(fitted, n_major=2)` is WRONG and would fundamentally change the model.

**Fix:** Use `n_major=5` in build scripts and tests. Also implement `MAX_MAJOR_BUBBLES` and `MAX_MINOR_BUBBLES` caps from Cell 0 lines 658-663 in the `classify` function.

### E5: `DE_POPSIZE = 18`, not 30 (IMPORTANT)

Cell 0 line 122: `DE_POPSIZE = 18`. The plan's `DEFAULT_CONFIG` says 30. Wrong default = different optimization results.

**Fix:** `"de_popsize": 18` in `DEFAULT_CONFIG`.

### E6: `price_dates/years/prices` source data (IMPORTANT)

Cell 2 exports price data from the Cell 1 `df` which includes ALL dates with `years >= 1.0` (no `fit_min_date` filter applied to the export data). The plan's `build_bm_pkl_dict` uses `price_data.df` (which IS filtered by `fit_min_date`).

**Fix:** Use `price_data.df_full` filtered to `years >= 1.0` for the price export keys, matching the original Cell 2 behavior. Or add a separate `df_export` field to `PriceData` that applies only the `years >= 1.0` filter.

### E7: `composite_at_grid` is linear USD, sigma fitting needs log10 (IMPORTANT)

`comp.composite_at_grid` is in linear USD. `fit_sigma.py` converts to log10 before interpolation. The `fit_asymmetric_sigma` implementation must apply `np.log10()` internally, matching `fit_sigma.py` line 33.

### E8: `comp_by_n` must return Python lists, not numpy arrays (MINOR)

Both BM Cell 2 and EF `build_ef_model.py` call `.tolist()` on comp_by_n arrays before writing to pkl. `build_comp_by_n` should return `[array.tolist() for array in results]`.

### Errata verification

After each task, the implementer (or reviewer) must verify:
- [ ] E1: `bubble_shape` accepts `slope_sup` parameter; all callers pass it
- [ ] E2: No `NameError` on any global when running the full pipeline
- [ ] E3: DE seeds match notebook's original-index pattern
- [ ] E4: `n_major=5` in build scripts; `classify` supports MAX caps
- [ ] E5: `de_popsize=18` in DEFAULT_CONFIG
- [ ] E6: Price export uses full date range (years >= 1.0, no fit_min_date filter)
- [ ] E7: Sigma fitting applies log10 to composite before interpolation
- [ ] E8: comp_by_n entries are Python lists

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `tools/model_toolkit/__init__.py` | Create | Package marker |
| `tools/model_toolkit/data.py` | Create | Load CSV, compute time columns |
| `tools/model_toolkit/support.py` | Create | Fit/define power-law support line |
| `tools/model_toolkit/bubble_shape.py` | Create | Parametric bubble function |
| `tools/model_toolkit/fitting.py` | Create | Peak finding + sequential DE fitting + classify |
| `tools/model_toolkit/prediction.py` | Create | Future bubble extrapolation |
| `tools/model_toolkit/composite.py` | Create | Composite model, R2, comp_by_n |
| `tools/model_toolkit/bands.py` | Create | QR channels + asymmetric gaussian sigma |
| `tools/model_toolkit/export.py` | Create | Assemble pkl dicts, write |
| `tools/build_bm_model.py` | Rewrite | Import from model_toolkit |
| `tools/build_ef_model.py` | Rewrite | Import from model_toolkit |
| `tools/verify_pkl.py` | Modify | Add EF key list + --type flag |
| `update_prices.py` | Modify | Remove fit_sigma call |
| `.gitignore` | Modify | Add debris/ |
| `tools/fit_sigma.py` | Delete | Absorbed into bands.py |

---

### Task 1: Package skeleton + `data.py` + `bubble_shape.py`

**Files:**
- Create: `tools/model_toolkit/__init__.py`
- Create: `tools/model_toolkit/data.py`
- Create: `tools/model_toolkit/bubble_shape.py`

Foundation modules with no internal dependencies.

- [ ] **Step 1: Create package and `__init__.py`**

```bash
mkdir -p tools/model_toolkit
```

Write `tools/model_toolkit/__init__.py`:
```python
"""Model toolkit -- reusable components for building Bitcoin price models."""
```

- [ ] **Step 2: Create `data.py`**

Read Cell 0 lines 271-322 (data loading) and Cell 1 lines 174-209 (QR data loading). Merge into a single standardized loader.

The module must:
1. Read CSV with case-insensitive column detection (Cell 1 pattern)
2. Compute `years = (date - genesis).days / 365.25`
3. Compute `log_years = log10(years)`, `log_price = log10(price)`
4. Filter to `years >= 1.0` and `date >= fit_min_date`
5. Return `PriceData` dataclass with both DataFrame and array views

```python
# tools/model_toolkit/data.py
"""Load and prepare Bitcoin price data."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class PriceData:
    df: pd.DataFrame          # filtered: date, price, years, log_years, log_price
    df_full: pd.DataFrame     # unfiltered (for price history export)
    years: np.ndarray
    log_years: np.ndarray
    prices: np.ndarray
    log_prices: np.ndarray
    dates: list

def load_prices(csv_path, genesis_date="2009-07-25", fit_min_date="2010-07-17"):
    genesis = pd.Timestamp(genesis_date)
    fit_min = pd.Timestamp(fit_min_date)
    df = pd.read_csv(csv_path)
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    price_col = next((c for c in df.columns if "price" in c.lower()), df.columns[1])
    df = df.rename(columns={date_col: "date", price_col: "price"})
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = df["price"].astype(float)
    df = df.sort_values("date").reset_index(drop=True)
    df["years"] = (df["date"] - genesis).dt.days / 365.25
    df["log_years"] = np.log10(df["years"].clip(lower=1e-10))
    df["log_price"] = np.log10(df["price"])
    df_full = df.copy()
    mask = (df["years"] >= 1.0) & (df["date"] >= fit_min)
    df = df[mask].reset_index(drop=True)
    return PriceData(
        df=df, df_full=df_full,
        years=df["years"].values, log_years=df["log_years"].values,
        prices=df["price"].values, log_prices=df["log_price"].values,
        dates=df["date"].dt.strftime("%Y-%m-%d").tolist(),
    )
```

- [ ] **Step 3: Create `bubble_shape.py`**

Extract `bubble_shape()` verbatim from Cell 0 lines 357-423. Read those lines, then create the module with the function copied exactly. No math changes.

```python
# tools/model_toolkit/bubble_shape.py
"""Parametric bubble shape function -- pure math, no state."""
from __future__ import annotations
import numpy as np

# Paste bubble_shape() verbatim from Cell 0 lines 357-423.
```

- [ ] **Step 4: Test**

```bash
btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'tools')
from model_toolkit.data import load_prices
from model_toolkit.bubble_shape import bubble_shape
import numpy as np
p = load_prices('BitcoinPricesDaily.csv')
print(f'Filtered: {len(p.df)} rows, Full: {len(p.df_full)} rows')
assert len(p.df) > 5000
t = np.linspace(1, 20, 100)
result = bubble_shape(t, 5.0, 3.0, 6.0, 7.0, 2.0, 0.0)
assert result.shape == (100,) and np.all(np.isfinite(result))
print('OK')
"
```

- [ ] **Step 5: Commit**

```bash
git add tools/model_toolkit/
git commit -m "feat(toolkit): add data.py + bubble_shape.py -- foundation modules"
```

---

### Task 2: `support.py`

**Files:**
- Create: `tools/model_toolkit/support.py`

- [ ] **Step 1: Create `support.py`**

Extract from Cell 0 lines 325-354 (STEP 1: FIT SUPPORT LINE). Read those lines to understand the fitting procedure: OLS -> bottom percentile filter -> QuantReg on subset.

```python
# tools/model_toolkit/support.py
"""Fit or define a power-law support line."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import linregress
import statsmodels.api as sm
from .data import PriceData

@dataclass
class SupportLine:
    intercept: float
    slope: float
    A: float                  # 10^intercept
    B: float                  # slope alias
    log_excess: np.ndarray    # log_price - log_support at data points
    log_support: np.ndarray   # support at data points (log10)

def fit_support(price_data: PriceData, percentile=0.20, quantile=0.50) -> SupportLine:
    """Fit support: OLS -> bottom percentile -> QuantReg."""
    log_t, log_p = price_data.log_years, price_data.log_prices
    slope_ols, intercept_ols, _, _, _ = linregress(log_t, log_p)
    residuals = log_p - (intercept_ols + slope_ols * log_t)
    threshold = np.percentile(residuals, percentile * 100)
    support_mask = residuals <= threshold
    X_support = sm.add_constant(log_t[support_mask])
    res_sup = sm.QuantReg(log_p[support_mask], X_support).fit(q=quantile)
    intercept_sup, slope_sup = res_sup.params[0], res_sup.params[1]
    log_support = intercept_sup + slope_sup * log_t
    log_excess = log_p - log_support
    return SupportLine(intercept=intercept_sup, slope=slope_sup,
                       A=10**intercept_sup, B=slope_sup,
                       log_excess=log_excess, log_support=log_support)

def fixed_support(intercept: float, slope: float, price_data: PriceData) -> SupportLine:
    """Use hardcoded support line (e.g. Empirical Floor)."""
    log_t, log_p = price_data.log_years, price_data.log_prices
    log_support = intercept + slope * log_t
    log_excess = log_p - log_support
    return SupportLine(intercept=intercept, slope=slope,
                       A=10**intercept, B=slope,
                       log_excess=log_excess, log_support=log_support)
```

NOTE: Verify against Cell 0 lines 325-354. The notebook uses `SUPPORT_PERCENTILE=0.20` and `SUPPORT_QUANTILE=0.50` -- these become the function parameter defaults.

- [ ] **Step 2: Test**

```bash
btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'tools')
from model_toolkit.data import load_prices
from model_toolkit.support import fit_support, fixed_support
p = load_prices('BitcoinPricesDaily.csv')
sup = fit_support(p)
print(f'slope={sup.slope:.4f}, intercept={sup.intercept:.4f}')
assert 4.5 < sup.slope < 6.0
sup_ef = fixed_support(-1.6306, 5.2480, p)
print(f'EF slope={sup_ef.slope:.4f}')
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add tools/model_toolkit/support.py
git commit -m "feat(toolkit): add support.py -- power-law support line fitting"
```

---

### Task 3: `fitting.py`

**Files:**
- Create: `tools/model_toolkit/fitting.py`

The largest module. Combines peak finding (Cell 0 STEP 2, lines 424-482), bubble fitting via DE (STEP 3, lines 483-635), and classification (STEP 4, lines 636-708).

- [ ] **Step 1: Read Cell 0 lines 424-710**

```bash
btc_venv/bin/python3 -c "
import json
with open('sp_stripped.ipynb') as f:
    nb = json.load(f)
src = ''.join(nb['cells'][0]['source'])
for i, line in enumerate(src.split('\n')[423:710], 424):
    print(f'{i:4d}: {line}')
"
```

Also read the config constants (Cell 0 lines 44-170) for DE parameters.

- [ ] **Step 2: Create `fitting.py`**

Extract the three stages into functions. The module structure:

```python
# tools/model_toolkit/fitting.py
"""Peak finding, sequential bubble fitting, classification."""
from __future__ import annotations
import numpy as np
from scipy.optimize import differential_evolution
from .bubble_shape import bubble_shape
from .data import PriceData
from .support import SupportLine

DEFAULT_CONFIG = {
    "bubble_years": [2011, 2013, 2017, 2021, 2025],
    "bubble_year_window": 0.75,
    "de_maxiter": 2000,
    "de_popsize": 30,
    "seed_base": 42,
    "plateau_parallel_support": True,
    # DE bounds, constraints -- extract from Cell 0 lines 83-125
}

def find_peaks(log_excess, years, bubble_years, window=0.75):
    """Cell 0 STEP 2 (lines 424-482): locate peaks in log-excess."""
    ...

def fit_bubble(log_excess, years, peak, support, config, seed_idx):
    """Cell 0 STEP 3 (lines 507-635): fit one bubble via DE.
    seed = config['seed_base'] + seed_idx for determinism."""
    ...

def fit_sequential(price_data, support, config=None):
    """Fit largest peak, subtract, repeat. Returns list of param dicts."""
    ...

def classify(fitted_bubbles, n_major=2):
    """Cell 0 STEP 4 (lines 636-708): split by K into major/minor."""
    ...
```

The implementer must read the notebook source and fill in each function body verbatim. Key details:
- `fit_bubble` wraps `fit_manual_bubble` (Cell 0 lines 507-635)
- 5D optimization when `plateau_parallel_support=True` (plat_pow=0), 6D otherwise
- `fit_sequential` iterates in descending magnitude order, each bubble gets `seed=42+idx`
- `classify` sorts by K descending, takes top n_major, re-sorts both groups chronologically

- [ ] **Step 3: Test** (takes ~60-120s for DE fitting)

```bash
btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'tools')
from model_toolkit.data import load_prices
from model_toolkit.support import fit_support
from model_toolkit.fitting import find_peaks, fit_sequential, classify
p = load_prices('BitcoinPricesDaily.csv')
sup = fit_support(p)
peaks = find_peaks(sup.log_excess, p.years, [2011,2013,2017,2021,2025])
print(f'{len(peaks)} peaks')
assert len(peaks) == 5
fitted = fit_sequential(p, sup)
print(f'{len(fitted)} bubbles fitted')
major, minor = classify(fitted, n_major=2)
print(f'{len(major)} major + {len(minor)} minor')
assert len(major) == 2
print('OK')
" 2>&1
```

- [ ] **Step 4: Commit**

```bash
git add tools/model_toolkit/fitting.py
git commit -m "feat(toolkit): add fitting.py -- peak finding + sequential DE fitting"
```

---

### Task 4: `prediction.py`

**Files:**
- Create: `tools/model_toolkit/prediction.py`

- [ ] **Step 1: Read Cell 0 lines 800-955**

The `predict_future_bubbles()` function. 155 lines of extrapolation logic.

- [ ] **Step 2: Create `prediction.py`**

Copy `predict_future_bubbles()` verbatim from Cell 0, wrap in module interface:

```python
# tools/model_toolkit/prediction.py
"""Extrapolate or average fitted bubble parameters to predict future bubbles."""
from __future__ import annotations
import numpy as np

def predict_future(fitted_major, fitted_minor, n_major=3, n_minor=1, mode="extrap"):
    """Predict future bubbles.
    mode="extrap": linear trend extrapolation per parameter.
    mode="avg": average last N, replicate with mean spacing.
    Returns (future_major, future_minor).
    """
    # Internal: call _predict(fitted_major, n_major, mode) then
    #           _predict(fitted_minor, n_minor, mode)
    ...

def _predict(bubbles, n_predict, mode):
    """Core prediction logic from Cell 0 lines 800-955."""
    ...
```

Copy the function body from the notebook. All internal constants (extrap_weights, PLAT_POW_RANGE, interval params) stay as local variables inside `_predict`.

- [ ] **Step 3: Test**

```bash
btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'tools')
from model_toolkit.data import load_prices
from model_toolkit.support import fit_support
from model_toolkit.fitting import fit_sequential, classify
from model_toolkit.prediction import predict_future
p = load_prices('BitcoinPricesDaily.csv')
sup = fit_support(p)
fitted = fit_sequential(p, sup)
major, minor = classify(fitted, n_major=2)
f_maj, f_min = predict_future(major, minor, n_major=3, n_minor=1)
print(f'Future: {len(f_maj)} major + {len(f_min)} minor')
assert len(f_maj) == 3 and len(f_min) == 1
print('OK')
" 2>&1
```

- [ ] **Step 4: Commit**

```bash
git add tools/model_toolkit/prediction.py
git commit -m "feat(toolkit): add prediction.py -- future bubble extrapolation"
```

---

### Task 5: `composite.py`

**Files:**
- Create: `tools/model_toolkit/composite.py`

- [ ] **Step 1: Read Cell 0 lines 709-791 (STEP 5) and Cell 2 lines 1-36 (comp_by_n)**

- [ ] **Step 2: Create `composite.py`**

Contains `_total_bubble` (from `bm_total_bubble`, Cell 0 lines 729-762), `build_composite`, and `build_comp_by_n` (from Cell 2 lines 1-36).

```python
# tools/model_toolkit/composite.py
"""Composite model: support + bubbles, R2, comp_by_n."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .bubble_shape import bubble_shape

PLOT_GRID_POINTS = 3000
PLOT_YEARS_MIN = 1.0
PLOT_YEARS_MAX = 72.0

@dataclass
class CompositeResult:
    composite_at_grid: np.ndarray
    composite_at_data: np.ndarray
    r2: float
    r2_support: float
    hist_K_max: float
    total_plot: np.ndarray
    t_grid: np.ndarray
    log_support_grid: np.ndarray
    support_grid: np.ndarray

def _total_bubble(t, fitted_bubbles, budget=None):
    """Capped sum of bubble contributions. From Cell 0 lines 729-762."""
    ...

def build_composite(support, fitted, price_data, t_grid=None, cap_overlap=True):
    """Composite on t_grid + at data points. R2 in log space."""
    if t_grid is None:
        t_grid = np.linspace(PLOT_YEARS_MIN, PLOT_YEARS_MAX, PLOT_GRID_POINTS)
    log_t_grid = np.log10(t_grid)
    log_support_grid = support.intercept + support.slope * log_t_grid
    support_grid = 10 ** log_support_grid
    hist_K_max = float(np.max(np.maximum(0.0, support.log_excess)))
    budget_grid = np.full(len(t_grid), hist_K_max) if cap_overlap else None
    total_plot = _total_bubble(t_grid, fitted, budget_grid)
    composite_grid = 10 ** (log_support_grid + total_plot)
    budget_data = np.full(len(price_data.years), hist_K_max) if cap_overlap else None
    total_data = _total_bubble(price_data.years, fitted, budget_data)
    composite_data = support.log_support + total_data
    ss_tot = np.sum((price_data.log_prices - np.mean(price_data.log_prices))**2)
    r2 = 1.0 - np.sum((price_data.log_prices - composite_data)**2) / ss_tot
    r2_sup = 1.0 - np.sum((price_data.log_prices - support.log_support)**2) / ss_tot
    return CompositeResult(
        composite_at_grid=composite_grid, composite_at_data=composite_data,
        r2=r2, r2_support=r2_sup, hist_K_max=hist_K_max,
        total_plot=total_plot, t_grid=t_grid,
        log_support_grid=log_support_grid, support_grid=support_grid)

def build_comp_by_n(support, fitted, future, t_grid, hist_K_max, total_plot, cap_overlap=True):
    """Precompute composite for N=0..max future bubbles. From Cell 2 lines 1-36."""
    ...
```

- [ ] **Step 3: Test**

```bash
btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'tools')
from model_toolkit.data import load_prices
from model_toolkit.support import fit_support
from model_toolkit.fitting import fit_sequential, classify
from model_toolkit.prediction import predict_future
from model_toolkit.composite import build_composite, build_comp_by_n
p = load_prices('BitcoinPricesDaily.csv')
sup = fit_support(p)
fitted = fit_sequential(p, sup)
major, minor = classify(fitted, n_major=2)
f_maj, f_min = predict_future(major, minor)
comp = build_composite(sup, fitted, p)
print(f'R2={comp.r2:.4f}, R2_sup={comp.r2_support:.4f}')
assert comp.r2 > 0.98
all_future = sorted(f_maj + f_min, key=lambda b: b['t_rise'])
cbn = build_comp_by_n(sup, fitted, all_future, comp.t_grid, comp.hist_K_max, comp.total_plot)
print(f'comp_by_n: {len(cbn)} arrays')
assert len(cbn) == len(all_future) + 1
print('OK')
" 2>&1
```

- [ ] **Step 4: Commit**

```bash
git add tools/model_toolkit/composite.py
git commit -m "feat(toolkit): add composite.py -- composite model + R2 + comp_by_n"
```

---

### Task 6: `bands.py`

**Files:**
- Create: `tools/model_toolkit/bands.py`

- [ ] **Step 1: Read Cell 1 lines 212-241 (QR fitting) and `tools/fit_sigma.py`**

- [ ] **Step 2: Create `bands.py`**

Two functions: `fit_qr_channels` (from Cell 1) and `fit_asymmetric_sigma` (absorbed from `fit_sigma.py`).

```python
# tools/model_toolkit/bands.py
"""Uncertainty bands: QR channels + asymmetric gaussian sigma."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import statsmodels.api as sm
from .data import PriceData

BM_QUANTILES = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
    0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
    0.99, 0.999, 0.9999, 0.99999]  # 27

EF_QUANTILES = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
    0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999]  # 21

@dataclass
class QRResult:
    fits: dict  # {q: {"intercept": f, "slope": f, "r2": f}}
    ols_intercept: float
    ols_slope: float

@dataclass
class SigmaParams:
    sigma0_up: float
    alpha_up: float
    sigma0_down: float
    alpha_down: float

def fit_qr_channels(price_data: PriceData, quantiles=None) -> QRResult:
    """From Cell 1 lines 212-241."""
    if quantiles is None: quantiles = BM_QUANTILES
    log_t, log_p = price_data.log_years, price_data.log_prices
    ols_slope, ols_intercept, _, _, _ = linregress(log_t, log_p)
    X = sm.add_constant(log_t)
    fits = {}
    for q in quantiles:
        res = sm.QuantReg(log_p, X).fit(q=q)
        pred = res.params[0] + res.params[1] * log_t
        ss_res = np.sum((log_p - pred)**2)
        ss_tot = np.sum((log_p - np.mean(log_p))**2)
        fits[q] = {"intercept": res.params[0], "slope": res.params[1], "r2": 1 - ss_res/ss_tot}
    return QRResult(fits=fits, ols_intercept=ols_intercept, ols_slope=ols_slope)

def fit_asymmetric_sigma(prices_log, composite_grid, t_grid, t_data,
                          n_bins=20, min_pts=10) -> SigmaParams:
    """Absorbed from tools/fit_sigma.py.
    1. Interpolate composite from t_grid to t_data
    2. residuals = prices_log - composite_at_data
    3. Bin by time, compute sigma upper/lower per bin
    4. Fit sigma(t) = sigma0 * t^(-alpha)
    """
    # Copy logic from fit_sigma.py lines 28-83
    ...
```

The implementer must read `tools/fit_sigma.py` and copy the binning + power-law fitting logic into `fit_asymmetric_sigma`. Key params: `n_bins=20`, `min_pts=10`.

- [ ] **Step 3: Test**

```bash
btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'tools')
from model_toolkit.data import load_prices
from model_toolkit.support import fit_support
from model_toolkit.fitting import fit_sequential
from model_toolkit.composite import build_composite
from model_toolkit.bands import fit_qr_channels, fit_asymmetric_sigma, BM_QUANTILES
p = load_prices('BitcoinPricesDaily.csv')
sup = fit_support(p)
fitted = fit_sequential(p, sup)
comp = build_composite(sup, fitted, p)
qr = fit_qr_channels(p, BM_QUANTILES)
print(f'QR: {len(qr.fits)} quantiles, OLS slope={qr.ols_slope:.4f}')
assert len(qr.fits) == 27
sigma = fit_asymmetric_sigma(p.log_prices, comp.composite_at_grid, comp.t_grid, p.years)
print(f'sigma0_up={sigma.sigma0_up:.4f}, alpha_up={sigma.alpha_up:.4f}')
assert sigma.sigma0_up > 0
print('OK')
" 2>&1
```

- [ ] **Step 4: Commit**

```bash
git add tools/model_toolkit/bands.py
git commit -m "feat(toolkit): add bands.py -- QR channels + asymmetric sigma"
```

---

### Task 7: `export.py`

**Files:**
- Create: `tools/model_toolkit/export.py`

- [ ] **Step 1: Create `export.py`**

Two pkl dict builders (BM and EF) with exact key names for backward compatibility.

```python
# tools/model_toolkit/export.py
"""Assemble and write model pkl files."""
from __future__ import annotations
import os
import pickle

def build_bm_pkl_dict(price_data, support, composite, comp_by_n, qr, sigma,
                       genesis_date="2009-07-25"):
    """17 keys. String keys for qr_fits. float() wrappers on scalars."""
    return {
        "qr_fits": {str(k): dict(v) for k, v in qr.fits.items()},
        "QR_QUANTILES": list(qr.fits.keys()),
        "ols_intercept": float(qr.ols_intercept),
        "ols_slope": float(qr.ols_slope),
        "GENESIS_DATE": genesis_date,
        "years_plot_bm": list(composite.t_grid),
        "support_plot_bm": list(composite.support_grid),
        "bm_comp_by_n": comp_by_n,
        "bm_r2_comp": float(composite.r2),
        "bm_n_future_max": len(comp_by_n) - 1,
        "bm_sigma0_up": float(sigma.sigma0_up),
        "bm_sigma0_down": float(sigma.sigma0_down),
        "bm_alpha_up": float(sigma.alpha_up),
        "bm_alpha_down": float(sigma.alpha_down),
        "price_dates": price_data.dates,
        "price_years": price_data.df["years"].tolist(),
        "price_prices": price_data.df["price"].tolist(),
    }

def build_ef_pkl_dict(support, composite, comp_by_n, sigma, fitted,
                       price_years, price_prices, quantiles, genesis_date="2009-07-25"):
    """EF key names (different convention from BM)."""
    fitted_params = []
    for b in sorted(fitted, key=lambda b: b["t_rise"]):
        fitted_params.append({k: b.get(k, 0.0) for k in
            ["t_rise","r","t_plateau","t_decay","d","K","plat_pow","dur_rise","dur_plateau"]})
    return {
        "ef_support_slope": support.slope,
        "ef_support_intercept": support.intercept,
        "genesis": genesis_date,
        "years_plot": composite.t_grid.tolist(),
        "support_plot": composite.support_grid.tolist(),
        "comp_by_n": comp_by_n,
        "bm_r2": float(composite.r2),
        "n_future_max": len(comp_by_n) - 1,
        "sigma0_up": float(sigma.sigma0_up),
        "sigma0_down": float(sigma.sigma0_down),
        "alpha_up": float(sigma.alpha_up),
        "alpha_down": float(sigma.alpha_down),
        "price_years": price_years,
        "price_prices": price_prices,
        "QR_QUANTILES": list(quantiles),
        "fitted_bubbles": fitted_params,
    }

def write_pkl(data, path, protocol=4):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=protocol)
    print(f"Wrote {path}  ({os.path.getsize(path)//1024} KB, {len(data)} keys)")
```

NOTE: `price_dates/years/prices` in BM dict comes from `price_data.df` (filtered, years >= 1.0) and `price_data.dates` -- verify this matches the original Cell 2 export.

- [ ] **Step 2: Commit**

```bash
git add tools/model_toolkit/export.py
git commit -m "feat(toolkit): add export.py -- BM/EF pkl dict assembly"
```

---

### Task 8: Rewrite `build_bm_model.py` + verify

**Files:**
- Rewrite: `tools/build_bm_model.py`

- [ ] **Step 1: Save reference**

```bash
cp archive/btc_app/model_data.pkl /tmp/model_data_reference.pkl
```

- [ ] **Step 2: Rewrite**

```python
#!/usr/bin/env python3
"""Build model_data.pkl -- BM model via model_toolkit."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from model_toolkit.data import load_prices
from model_toolkit.support import fit_support
from model_toolkit.fitting import fit_sequential, classify
from model_toolkit.prediction import predict_future
from model_toolkit.composite import build_composite, build_comp_by_n
from model_toolkit.bands import fit_qr_channels, fit_asymmetric_sigma, BM_QUANTILES
from model_toolkit.export import build_bm_pkl_dict, write_pkl

def main():
    prices = load_prices("BitcoinPricesDaily.csv")
    sup = fit_support(prices)
    fitted = fit_sequential(prices, sup)
    major, minor = classify(fitted, n_major=2)
    f_maj, f_min = predict_future(major, minor, n_major=3, n_minor=1)
    all_future = sorted(f_maj + f_min, key=lambda b: b["t_rise"])
    comp = build_composite(sup, fitted, prices)
    cbn = build_comp_by_n(sup, fitted, all_future, comp.t_grid, comp.hist_K_max, comp.total_plot)
    qr = fit_qr_channels(prices, BM_QUANTILES)
    sigma = fit_asymmetric_sigma(prices.log_prices, comp.composite_at_grid, comp.t_grid, prices.years)
    write_pkl(build_bm_pkl_dict(prices, sup, comp, cbn, qr, sigma),
              os.path.join(ROOT, "archive", "btc_app", "model_data.pkl"))

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run and verify**

```bash
btc_venv/bin/python3 tools/build_bm_model.py
btc_venv/bin/python3 tools/verify_pkl.py /tmp/model_data_reference.pkl archive/btc_app/model_data.pkl
```

All 17 keys should be OK (machine-epsilon tolerance on fitted values).

- [ ] **Step 4: Restore and commit**

```bash
cp /tmp/model_data_reference.pkl archive/btc_app/model_data.pkl
git add tools/build_bm_model.py
git commit -m "refactor: rewrite build_bm_model.py -- import from model_toolkit"
```

---

### Task 9: Rewrite `build_ef_model.py` + update `verify_pkl.py`

**Files:**
- Rewrite: `tools/build_ef_model.py`
- Modify: `tools/verify_pkl.py`

- [ ] **Step 1: Save EF reference**

```bash
cp btc_app/model_data_ef.pkl /tmp/model_data_ef_reference.pkl
```

- [ ] **Step 2: Add EF keys to verify_pkl.py**

Add `EF_KEYS` list and `--type` argument:

```python
EF_KEYS = [
    "ef_support_slope", "ef_support_intercept", "genesis",
    "years_plot", "support_plot", "comp_by_n", "bm_r2", "n_future_max",
    "sigma0_up", "sigma0_down", "alpha_up", "alpha_down",
    "price_years", "price_prices", "QR_QUANTILES", "fitted_bubbles",
]
```

In `main()`, add argparse for `--type bm|ef` that selects `MODEL_KEYS` or `EF_KEYS`.

- [ ] **Step 3: Rewrite `build_ef_model.py`**

```python
#!/usr/bin/env python3
"""Build model_data_ef.pkl -- Empirical Floor model via model_toolkit."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from model_toolkit.data import load_prices
from model_toolkit.support import fixed_support
from model_toolkit.fitting import fit_sequential, classify
from model_toolkit.prediction import predict_future
from model_toolkit.composite import build_composite, build_comp_by_n
from model_toolkit.bands import fit_asymmetric_sigma, EF_QUANTILES
from model_toolkit.export import build_ef_pkl_dict, write_pkl

EF_SLOPE = 5.248017
EF_INTERCEPT = -1.630623

def main():
    prices = load_prices("BitcoinPricesDaily.csv")
    sup = fixed_support(EF_INTERCEPT, EF_SLOPE, prices)
    fitted = fit_sequential(prices, sup)
    major, minor = classify(fitted, n_major=2)
    f_maj, f_min = predict_future(major, minor, n_major=3, n_minor=1)
    all_future = sorted(f_maj + f_min, key=lambda b: b["t_rise"])
    comp = build_composite(sup, fitted, prices)
    cbn = build_comp_by_n(sup, fitted, all_future, comp.t_grid, comp.hist_K_max, comp.total_plot)
    sigma = fit_asymmetric_sigma(prices.log_prices, comp.composite_at_grid, comp.t_grid, prices.years)
    write_pkl(build_ef_pkl_dict(sup, comp, cbn, sigma, fitted,
                                 prices.years.tolist(), prices.prices.tolist(), EF_QUANTILES),
              os.path.join(ROOT, "btc_app", "model_data_ef.pkl"))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run and verify EF**

```bash
btc_venv/bin/python3 tools/build_ef_model.py
btc_venv/bin/python3 tools/verify_pkl.py --type ef /tmp/model_data_ef_reference.pkl btc_app/model_data_ef.pkl
```

- [ ] **Step 5: Restore and commit**

```bash
cp /tmp/model_data_ef_reference.pkl btc_app/model_data_ef.pkl
git add tools/build_ef_model.py tools/verify_pkl.py
git commit -m "refactor: rewrite build_ef_model.py + add EF verification"
```

---

### Task 10: Cleanup

**Files:**
- Delete: `tools/fit_sigma.py`
- Modify: `.gitignore`
- Modify: `update_prices.py`

- [ ] **Step 1: Delete fit_sigma.py**

```bash
git rm tools/fit_sigma.py
```

- [ ] **Step 2: Create debris/ and move files**

```bash
mkdir -p debris
git mv sp_stripped.ipynb debris/ 2>/dev/null || mv sp_stripped.ipynb debris/
ls SP.ipynb.2026-03-20_1059.bak 2>/dev/null && mv SP.ipynb.2026-03-20_1059.bak debris/
echo "debris/" >> .gitignore
```

- [ ] **Step 3: Update update_prices.py**

Remove the `fit_sigma.py` call from `run_model_build()` -- sigma is now computed inside `build_bm_model.py`. Just call `build_bm_model.py`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "cleanup: delete fit_sigma.py, move debris, simplify update_prices.py"
```

---

### Task 11: End-to-end verification

- [ ] **Step 1: BM pipeline**

```bash
cp archive/btc_app/model_data.pkl /tmp/model_data_reference.pkl
btc_venv/bin/python3 tools/build_bm_model.py
btc_venv/bin/python3 tools/verify_pkl.py /tmp/model_data_reference.pkl archive/btc_app/model_data.pkl
```

- [ ] **Step 2: EF pipeline**

```bash
cp btc_app/model_data_ef.pkl /tmp/model_data_ef_reference.pkl
btc_venv/bin/python3 tools/build_ef_model.py
btc_venv/bin/python3 tools/verify_pkl.py --type ef /tmp/model_data_ef_reference.pkl btc_app/model_data_ef.pkl
```

- [ ] **Step 3: Web app test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --ignore=btc_web/test_tax_e2e.py --tb=short 2>&1 | tail -5
```

Expected: 889 passed, 0 failed, 5 skipped.

- [ ] **Step 4: Figures render**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "
import os; os.environ['DEV'] = '1'
import app
from _app_ctx import M
from figures.bubble import build_bubble_figure
from figures.heatmap import build_heatmap_figure
from tab_defaults import bubble_defaults, heatmap_defaults
fig1 = build_bubble_figure(M, bubble_defaults())
fig2 = build_heatmap_figure(M, heatmap_defaults())
print(f'Bubble: {len(fig1.data)} traces, Heatmap: {len(fig2.data)} traces')
print('OK')
" 2>&1 | grep -v '^\[MC'
```

- [ ] **Step 5: Restore reference pkls**

```bash
cp /tmp/model_data_reference.pkl archive/btc_app/model_data.pkl
cp /tmp/model_data_ef_reference.pkl btc_app/model_data_ef.pkl 2>/dev/null || true
```
