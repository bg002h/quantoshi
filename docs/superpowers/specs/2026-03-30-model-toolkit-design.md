# Model Toolkit — Design Spec

**Goal:** Refactor notebook computation into `tools/model_toolkit/`, a library of independent capabilities that model builders compose. Both `build_bm_model.py` and `build_ef_model.py` become thin recipes that import from the toolkit. `fit_sigma.py` is absorbed into `bands.py`. No more exec-ing notebook cells.

**Motivation:** Both build scripts currently exec notebook source code — BM reads `sp_stripped.ipynb` cells, EF extracts/patches SP.ipynb Cell 0 via string matching. Both are fragile. The computation has clear independent stages that map naturally to focused modules. Refactoring into proper Python makes the code testable, importable, and maintainable.

---

## Package Structure

```
tools/model_toolkit/
    __init__.py          — package marker
    data.py              — load CSV, compute time columns, filter
    support.py           — fit or define power-law support line
    bubble_shape.py      — parametric bubble function (pure math)
    fitting.py           — peak finding + sequential residual fitting
    prediction.py        — extrapolate/average bubble params for future
    composite.py         — assemble composite model, R2, comp_by_n
    bands.py             — QR channels + asymmetric gaussian sigma
    export.py            — assemble pkl dict, write with protocol=4
```

---

## Module Specifications

### `data.py` (~60 lines)

Load and prepare Bitcoin price data.

```python
def load_prices(csv_path, genesis_date="2009-07-25", fit_min_date="2010-07-17"):
    """Load CSV, compute years since genesis, log columns, filter.

    Returns DataFrame with columns: date, price, years, log_years, log_price
    Also returns raw unfiltered df for price history export.
    """
```

**Inputs:** CSV path, genesis date, min fit date
**Outputs:** filtered DataFrame (for fitting), full DataFrame (for price export)
**Dependencies:** pandas, numpy

---

### `support.py` (~80 lines)

Fit or define the power-law support line.

```python
def fit_support(df, percentile=0.20, quantile=0.50):
    """Fit support line: OLS -> bottom percentile filter -> QuantReg.

    Returns SupportLine(intercept, slope, log_excess)
    """

def fixed_support(intercept, slope, df):
    """Use hardcoded support line (e.g. Empirical Floor).

    Returns SupportLine(intercept, slope, log_excess)
    """
```

**SupportLine:** namedtuple or dataclass with `intercept`, `slope`, `log_excess` (residuals above support for all data points), `A` (10^intercept), `B` (slope alias).

**Dependencies:** numpy, scipy.stats.linregress, statsmodels.QuantReg

---

### `bubble_shape.py` (~70 lines)

The parametric bubble function. Pure math, no state.

```python
def bubble_shape(t, t_rise, r, t_plateau, t_decay, d, plat_pow=0.0):
    """Compute log10(excess above support) for a single bubble.

    Three phases: exponential rise, plateau, exponential decay.
    Returns ndarray of log-excess values.
    """
```

Copied verbatim from Cell 0 lines 357-424. No changes to the math.

**Dependencies:** numpy

---

### `fitting.py` (~250 lines)

Peak finding and sequential residual fitting.

```python
def find_peaks(log_excess, years, bubble_years, window=0.75):
    """Locate peaks in log-excess residuals.

    Returns list of peak dicts sorted by magnitude (largest first).
    """

def fit_bubble(log_excess, years, peak, support, config):
    """Fit a single bubble via differential_evolution.

    Returns fitted bubble params dict.
    """

def fit_sequential(df, support, bubble_years, config):
    """Sequential residual fitting: fit largest peak, subtract, repeat.

    Returns (fitted_bubbles, residual) where fitted_bubbles is a list of
    param dicts and residual is the remaining log-excess after all fits.
    """

def classify(fitted_bubbles, n_major=2):
    """Classify into major/minor by peak magnitude.

    Returns (major, minor) lists, each sorted chronologically.
    """
```

**Config:** a dict or namespace carrying DE optimizer settings (maxiter, popsize, seed base), search windows, plateau/decay constraints. Extracted from the ~80 lines of constants at the top of Cell 0.

**Dependencies:** numpy, scipy.optimize.differential_evolution, bubble_shape

---

### `prediction.py` (~120 lines)

Extrapolate or average fitted bubble parameters to predict future bubbles.

```python
def predict_future(fitted_major, fitted_minor, n_major=3, n_minor=1, mode="extrap"):
    """Predict future bubbles by extrapolating parameter trends.

    Returns (future_major, future_minor) lists of param dicts.
    """
```

**Dependencies:** numpy

---

### `composite.py` (~100 lines)

Build composite model from support + fitted + predicted bubbles.

```python
def build_composite(support, fitted, future, t_grid, cap_overlap=True):
    """Composite price = support + sum(bubble contributions).

    Returns CompositeResult(composite_prices, r2, r2_support,
                            hist_K_max, total_plot)
    """

def build_comp_by_n(support, fitted, future, t_grid, cap_overlap=True):
    """Precompute composite for N=0..max future bubbles.

    Returns list of price arrays, one per future-bubble count.
    """
```

**Dependencies:** numpy, bubble_shape

---

### `bands.py` (~150 lines)

Uncertainty band fitting. Two methods: quantile regression channels and asymmetric gaussian sigma.

```python
def fit_qr_channels(df, quantiles):
    """Fit quantile regression at each quantile level.

    Returns dict {q: {intercept, slope, r2}} and OLS (intercept, slope).
    """

def fit_asymmetric_sigma(prices_log, composite_log, years,
                         n_windows=12, min_window_pts=50):
    """Fit asymmetric shrinking gaussian to residuals.

    Splits residuals into upper/lower, fits sigma(t) = sigma0 * t^(-alpha)
    in rolling windows.

    Returns SigmaParams(sigma0_up, alpha_up, sigma0_down, alpha_down).
    """
```

`fit_asymmetric_sigma` absorbs the logic currently in `tools/fit_sigma.py`. The standalone script is deleted.

**Dependencies:** numpy, scipy.optimize.curve_fit, statsmodels.QuantReg, scipy.stats.linregress

---

### `export.py` (~50 lines)

Assemble and write the lean pkl.

```python
def build_bm_pkl_dict(df_full, support, composite, comp_by_n, qr, sigma, genesis_date):
    """Assemble BM model pkl dict (17 keys)."""

def build_ef_pkl_dict(support, composite, comp_by_n, sigma, fitted,
                      price_years, price_prices, quantiles, genesis_date):
    """Assemble EF model pkl dict."""

def write_pkl(data, path, protocol=4):
    """Write pkl file."""
```

**Dependencies:** standard library only

---

## Refactored Build Scripts

### `tools/build_bm_model.py` (~50 lines)

```python
from model_toolkit import data, support, fitting, prediction, composite, bands, export

df, df_full = data.load_prices("BitcoinPricesDaily.csv")
sup = support.fit_support(df)
fitted = fitting.fit_sequential(df, sup, BUBBLE_YEARS, config)
major, minor = fitting.classify(fitted)
future_major = prediction.predict_future(major, minor)
comp = composite.build_composite(sup, fitted, future_major, t_grid)
cbn = composite.build_comp_by_n(sup, fitted, future_major, t_grid)
qr = bands.fit_qr_channels(df, QR_QUANTILES)
sigma = bands.fit_asymmetric_sigma(df, comp)
export.write_pkl(export.build_bm_pkl_dict(...), "archive/btc_app/model_data.pkl")
```

### `tools/build_ef_model.py` (~50 lines)

```python
from model_toolkit import data, support, fitting, prediction, composite, bands, export

df, df_full = data.load_prices("BitcoinPricesDaily.csv")
sup = support.fixed(intercept=EF_INTERCEPT, slope=EF_SLOPE, df=df)
fitted = fitting.fit_sequential(df, sup, BUBBLE_YEARS, config)
major, minor = fitting.classify(fitted)
future_major = prediction.predict_future(major, minor)
comp = composite.build_composite(sup, fitted, future_major, t_grid)
cbn = composite.build_comp_by_n(sup, fitted, future_major, t_grid)
sigma = bands.fit_asymmetric_sigma(df, comp)
export.write_pkl(export.build_ef_pkl_dict(...), "btc_app/model_data_ef.pkl")
```

The only differences: `support.fit_support()` vs `support.fixed()`, BM includes QR channels, different pkl dict structure.

---

## Files Deleted

- `tools/fit_sigma.py` — absorbed into `model_toolkit/bands.py`

## Files Moved to `debris/`

- `sp_stripped.ipynb`
- `SP.ipynb.2026-03-20_1059.bak` (if exists)

## Files Modified

- `tools/build_bm_model.py` — rewritten to import from model_toolkit
- `tools/build_ef_model.py` — rewritten to import from model_toolkit
- `update_prices.py` — remove `fit_sigma.py` call (sigma now part of build)
- `.gitignore` — add `debris/`

---

## Configuration

The notebook's ~80 lines of constants (bubble years, search windows, DE optimizer settings, N_MAJOR, N_PREDICT, etc.) live as module-level defaults in the relevant modules:
- `fitting.py`: DE_MAXITER, DE_POPSIZE, BUBBLE_YEARS, BUBBLE_YEAR_WINDOW, N_MAJOR
- `prediction.py`: N_PREDICT_MAJOR, N_PREDICT_MINOR, prediction mode
- `composite.py`: CAP_COMPOSITE_OVERLAP, PLOT_GRID_POINTS, PLOT_YEARS_MIN/MAX
- `bands.py`: QR_QUANTILES, sigma window settings

Build scripts can override any default by passing explicit arguments.

---

## Verification

- `tools/verify_pkl.py` compares new build output against reference pkl
- BM: all 17 model keys value-identical (machine-epsilon tolerance for DE-fitted floats)
- EF: same approach with EF reference pkl
- Full test suite must pass: 889 passed, 0 failed, 5 skipped

---

## Constraints

- **No matplotlib dependency** in the toolkit — computation only
- **Deterministic**: all DE calls use `seed=42+idx` pattern
- **No notebook dependency**: build scripts import Python modules, not exec cells
- **Backward compatible**: pkl output structure unchanged for both BM and EF
