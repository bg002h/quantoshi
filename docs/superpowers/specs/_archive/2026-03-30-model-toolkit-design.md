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

### `data.py` (~80 lines)

Load and prepare Bitcoin price data. Standardizes the two different loading patterns (Cell 0's array-based and Cell 1's DataFrame-based) into a single interface.

```python
@dataclass
class PriceData:
    """Standardized price data for all toolkit consumers."""
    df: pd.DataFrame          # filtered: date, price, years, log_years, log_price
    df_full: pd.DataFrame     # unfiltered (for price history export)
    years: np.ndarray         # years since genesis (filtered)
    log_years: np.ndarray     # log10(years) (filtered)
    prices: np.ndarray        # USD prices (filtered)
    log_prices: np.ndarray    # log10(prices) (filtered)
    dates: list               # date strings (filtered)

def load_prices(csv_path, genesis_date="2009-07-25", fit_min_date="2010-07-17"):
    """Load CSV, compute years since genesis, log columns, filter.

    Returns PriceData with both DataFrame and array views.
    Handles CSV column detection (case-insensitive 'date'/'price').
    """
```

**Inputs:** CSV path, genesis date, min fit date
**Outputs:** `PriceData` dataclass
**Dependencies:** pandas, numpy

---

### `support.py` (~80 lines)

Fit or define the power-law support line.

```python
@dataclass
class SupportLine:
    intercept: float          # log10 intercept
    slope: float              # power-law exponent
    A: float                  # 10^intercept (price coefficient)
    B: float                  # slope alias
    log_excess: np.ndarray    # residuals above support at data points
    log_support: np.ndarray   # support values at data points

def fit_support(price_data, percentile=0.20, quantile=0.50):
    """Fit support line: OLS -> bottom percentile filter -> QuantReg.

    Returns SupportLine.
    """

def fixed_support(intercept, slope, price_data):
    """Use hardcoded support line (e.g. Empirical Floor).

    Returns SupportLine with same structure as fit_support.
    """
```

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

def fit_bubble(log_excess, years, peak, support, config, seed_idx):
    """Fit a single bubble via differential_evolution.

    seed_idx: integer index for deterministic seeding (seed=42+seed_idx).
    This is the magnitude-sorted fitting order (0=largest peak first).

    Returns fitted bubble params dict.
    """

def fit_sequential(price_data, support, bubble_years, config):
    """Sequential residual fitting: fit largest peak, subtract, repeat.

    Fits in descending magnitude order. Each bubble gets seed=42+idx
    where idx is its position in the magnitude-sorted sequence.

    Returns (fitted_bubbles, residual) where fitted_bubbles is a list of
    param dicts and residual is the remaining log-excess after all fits.
    """

def classify(fitted_bubbles, n_major=2):
    """Classify into major/minor by peak magnitude.

    Returns (major, minor) lists, each sorted chronologically.
    """
```

**Config:** a dict or namespace carrying DE optimizer settings (maxiter, popsize, seed base=42), search windows, plateau/decay constraints. Extracted from the ~80 lines of constants at the top of Cell 0.

**Dependencies:** numpy, scipy.optimize.differential_evolution, bubble_shape

---

### `prediction.py` (~155 lines)

Extrapolate or average fitted bubble parameters to predict future bubbles.

```python
def predict_future(fitted_major, fitted_minor, n_major=3, n_minor=1, mode="extrap"):
    """Predict future bubbles by extrapolating parameter trends.

    mode="extrap": fit linear trend to each parameter across historical
    bubbles, extrapolate forward. Handles per-parameter log vs linear
    space, weight matrices, interval trends, plat_pow clipping.

    mode="avg": average last N bubbles' parameters, replicate with
    mean spacing. Simpler, no trend.

    Returns (future_major, future_minor) lists of param dicts.
    """
```

Internal implementation preserves all notebook logic verbatim: weighted extrapolation, log-linear parameter handling, interval trend computation, PLAT_POW_RANGE clipping, derived parameters (dur_rise=K/r, K_end from dur_plateau). These are internal constants, not exposed in the public interface. Future cleanup can expose knobs as needed.

**Dependencies:** numpy

---

### `composite.py` (~130 lines)

Build composite model from support + fitted + predicted bubbles. Contains `bm_total_bubble` (the capped sum of bubble contributions) as an internal helper.

```python
@dataclass
class CompositeResult:
    composite_at_grid: np.ndarray   # USD prices on t_grid (3000 pts)
    composite_at_data: np.ndarray   # log composite at data timestamps (for R2)
    r2: float                       # composite R2
    r2_support: float               # support-only R2
    hist_K_max: float               # max historical log-excess (budget cap)
    total_plot: np.ndarray          # raw log-excess total on t_grid

def build_composite(support, fitted, price_data, t_grid, cap_overlap=True):
    """Composite price = support + sum(bubble contributions).

    Computes composite both on the dense t_grid (for plotting/export)
    and at actual data timestamps (for R2 computation).

    Returns CompositeResult.
    """

def build_comp_by_n(support, fitted, future, t_grid, hist_K_max,
                     total_plot, cap_overlap=True):
    """Precompute composite for N=0..max future bubbles.

    Returns list of price arrays, one per future-bubble count.
    """
```

The internal `_total_bubble(t, fitted_bubbles, budget)` function computes the capped sum of bubble contributions, used by both `build_composite` and `build_comp_by_n`.

**Dependencies:** numpy, bubble_shape

---

### `bands.py` (~180 lines)

Uncertainty band fitting. Two methods: quantile regression channels and asymmetric gaussian sigma.

```python
@dataclass
class QRResult:
    fits: dict                    # {q: {intercept, slope, r2}}
    ols_intercept: float
    ols_slope: float

@dataclass
class SigmaParams:
    sigma0_up: float
    alpha_up: float
    sigma0_down: float
    alpha_down: float

BM_QUANTILES = [
    0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
    0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
    0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999,
]  # 27 quantiles — used by BM build

EF_QUANTILES = [
    0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
    0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999,
]  # 21 quantiles — used by EF build

def fit_qr_channels(df, quantiles=BM_QUANTILES):
    """Fit quantile regression at each quantile level.

    Returns QRResult.
    """

def fit_asymmetric_sigma(prices_log, composite_grid, t_grid, t_data,
                          n_bins=20, min_pts=10):
    """Fit asymmetric shrinking gaussian to residuals.

    Interpolates composite from t_grid to t_data via np.interp,
    then splits residuals into upper/lower, computes sigma in
    rolling bins, fits sigma(t) = sigma0 * t^(-alpha).

    Returns SigmaParams.
    """
```

`fit_asymmetric_sigma` absorbs the logic currently in `tools/fit_sigma.py`. The standalone script is deleted. Parameters match the actual `fit_sigma.py` implementation: `n_bins=20`, `min_pts=10`.

**Dependencies:** numpy, scipy.optimize.curve_fit, statsmodels.QuantReg, scipy.stats.linregress

---

### `export.py` (~70 lines)

Assemble and write pkl files. Preserves exact key names for backward compatibility.

```python
def build_bm_pkl_dict(price_data, support, composite, comp_by_n,
                       qr, sigma, genesis_date="2009-07-25"):
    """Assemble BM model pkl dict (17 keys).

    Keys: qr_fits (str keys), QR_QUANTILES, ols_intercept, ols_slope,
    GENESIS_DATE, years_plot_bm, support_plot_bm, bm_comp_by_n,
    bm_r2_comp, bm_n_future_max, bm_sigma0_up, bm_sigma0_down,
    bm_alpha_up, bm_alpha_down, price_dates, price_years, price_prices.

    Float scalars wrapped in float() to avoid numpy.float64 in pkl.
    """

def build_ef_pkl_dict(support, composite, comp_by_n, sigma, fitted,
                       price_years, price_prices, quantiles,
                       genesis_date="2009-07-25"):
    """Assemble EF model pkl dict.

    Key names follow EF convention (different from BM):
    ef_support_slope, ef_support_intercept, genesis, years_plot,
    support_plot, comp_by_n, bm_r2, n_future_max, sigma0_up,
    sigma0_down, alpha_up, alpha_down, price_years, price_prices,
    QR_QUANTILES, fitted_bubbles.
    """

def write_pkl(data, path, protocol=4):
    """Write pkl file."""
```

**Dependencies:** standard library only

---

## Refactored Build Scripts

### `tools/build_bm_model.py` (~50 lines)

```python
from model_toolkit import data, support, fitting, prediction, composite, bands, export

prices = data.load_prices("BitcoinPricesDaily.csv")
sup = support.fit_support(prices)
fitted = fitting.fit_sequential(prices, sup, BUBBLE_YEARS, config)
major, minor = fitting.classify(fitted)
future_maj, future_min = prediction.predict_future(major, minor)
comp = composite.build_composite(sup, fitted, prices, t_grid)
cbn = composite.build_comp_by_n(sup, fitted, future_maj + future_min,
                                 t_grid, comp.hist_K_max, comp.total_plot)
qr = bands.fit_qr_channels(prices.df, bands.BM_QUANTILES)
sigma = bands.fit_asymmetric_sigma(prices.log_prices, comp.composite_at_grid,
                                    t_grid, prices.years)
export.write_pkl(export.build_bm_pkl_dict(prices, sup, comp, cbn, qr, sigma),
                 "archive/btc_app/model_data.pkl")
```

### `tools/build_ef_model.py` (~50 lines)

```python
from model_toolkit import data, support, fitting, prediction, composite, bands, export

prices = data.load_prices("BitcoinPricesDaily.csv")
sup = support.fixed(intercept=EF_INTERCEPT, slope=EF_SLOPE, price_data=prices)
fitted = fitting.fit_sequential(prices, sup, BUBBLE_YEARS, config)
major, minor = fitting.classify(fitted)
future_maj, future_min = prediction.predict_future(major, minor)
comp = composite.build_composite(sup, fitted, prices, t_grid)
cbn = composite.build_comp_by_n(sup, fitted, future_maj + future_min,
                                 t_grid, comp.hist_K_max, comp.total_plot)
sigma = bands.fit_asymmetric_sigma(prices.log_prices, comp.composite_at_grid,
                                    t_grid, prices.years)
export.write_pkl(export.build_ef_pkl_dict(sup, comp, cbn, sigma, fitted, ...),
                 "btc_app/model_data_ef.pkl")
```

The only differences: `support.fit_support()` vs `support.fixed()`, BM includes QR channels, different pkl dict structure and key naming.

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
- `tools/verify_pkl.py` — add EF_KEYS list for EF pkl verification

---

## Configuration

The notebook's ~80 lines of constants (bubble years, search windows, DE optimizer settings, N_MAJOR, N_PREDICT, etc.) live as module-level defaults in the relevant modules:
- `fitting.py`: DE_MAXITER, DE_POPSIZE, BUBBLE_YEARS, BUBBLE_YEAR_WINDOW, N_MAJOR, seed base (42)
- `prediction.py`: N_PREDICT_MAJOR, N_PREDICT_MINOR, prediction mode, internal: extrap weights, interval params, PLAT_POW_RANGE
- `composite.py`: CAP_COMPOSITE_OVERLAP, PLOT_GRID_POINTS=3000, PLOT_YEARS_MIN=1.0, PLOT_YEARS_MAX=72.0
- `bands.py`: BM_QUANTILES (27), EF_QUANTILES (21), sigma bin settings (n_bins=20, min_pts=10)

Build scripts can override any default by passing explicit arguments.

---

## Verification

- `tools/verify_pkl.py` compares new build output against reference pkl
- BM: all 17 model keys value-identical (machine-epsilon tolerance for DE-fitted floats)
- EF: same approach with EF reference pkl, using EF-specific key list
- Full web app test suite must pass: 889 passed, 0 failed, 5 skipped
- Note: `today_years` variable in Cell 0 is date-dependent. Verification must run same day as reference, or this variable must be excluded from comparison. The toolkit should compute `today_years` from the CSV's last date rather than system clock.

---

## Constraints

- **No matplotlib dependency** in the toolkit — computation only
- **Deterministic**: all DE calls use `seed=42+idx` where idx is magnitude-sorted fitting order
- **No notebook dependency**: build scripts import Python modules, not exec cells
- **Backward compatible**: pkl output structure unchanged for both BM and EF (exact key names preserved)
- **Lean pkl only**: BM produces 17-key pkl (visual config lives in `btc_web/theme.py`). Web app's `btc_core.py` has `.get()` fallbacks for missing visual keys.
