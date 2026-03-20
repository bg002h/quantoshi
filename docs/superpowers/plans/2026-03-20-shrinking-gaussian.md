# Asymmetric Shrinking Gaussian Bands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace BubbleModel's QR-based bands and EmpiricalFloorModel's constant-σ bands with asymmetric shrinking Gaussian bands around each model's bubble composite, fixing internal consistency and capturing volatility compression.

**Architecture:** A new `_CompositeModel` base class provides `price_at`/`interp_price`/`find_percentile` using composite interpolation + `σ(t) = σ₀ × t^(-α)` (separate σ for above/below median). Both `BubbleModel` and `EmpiricalFloorModel` extend it. A standalone `tools/fit_sigma.py` computes the 4 σ parameters from historical residuals and writes them to each pkl.

**Tech Stack:** Python, NumPy, SciPy (curve_fit), pickle

**Spec:** `docs/superpowers/specs/2026-03-20-shrinking-gaussian-design.md`

**Branch:** `ShrinkingGaussian`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `tools/fit_sigma.py` | Create | Fit σ₀_up, α_up, σ₀_down, α_down from residuals, write to pkl |
| `archive/btc_app/btc_core.py` | Modify | Add `_CompositeModel` base class, convert `BubbleModel` and `EmpiricalFloorModel` |
| `btc_web/test_web.py` | Modify | Update BM and EF tests for new band behavior |
| `btc_app/model_data.pkl` | Modify (via tool) | Add 4 σ fields |
| `btc_app/model_data_ef.pkl` | Modify (via tool) | Replace `sigma` with 4 σ fields |

---

### Task 1: Create `tools/fit_sigma.py`

**Files:**
- Create: `tools/fit_sigma.py`

- [ ] **Step 1: Write the script**

The script loads a pkl, computes asymmetric shrinking σ from residuals, and writes the parameters back.

```python
#!/usr/bin/env python3
"""
Fit asymmetric shrinking sigma parameters for a bubble model pkl.

Computes σ_up(t) = σ₀_up × t^(-α_up) and σ_down(t) = σ₀_down × t^(-α_down)
from windowed residuals of log10(price) vs log10(composite).

Usage:
    btc_venv/bin/python3 tools/fit_sigma.py --pkl btc_app/model_data.pkl --type bm
    btc_venv/bin/python3 tools/fit_sigma.py --pkl btc_app/model_data_ef.pkl --type ef
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _fit_sigma(pkl_path, model_type):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    # Load historical data
    if model_type == "bm":
        prices = np.asarray(d["price_prices"], float)
        years = np.asarray(d["price_years"], float)
        comp_grid_y = np.asarray(d["years_plot_bm"], float)
        comp_grid_p = np.asarray(d["bm_comp_by_n"][-1], float)
    else:  # ef
        prices = np.asarray(d["price_prices"], float)
        years = np.asarray(d["price_years"], float)
        comp_grid_y = np.asarray(d["years_plot"], float)
        comp_grid_p = np.asarray(d["comp_by_n"][-1], float)

    log_p = np.log10(prices)
    log_comp = np.interp(years,
                         comp_grid_y,
                         np.log10(np.maximum(comp_grid_p, 1e-10)))
    residuals = log_p - log_comp

    # Windowed fit in 20 log-time bins
    log_t = np.log10(years)
    n_bins = 20
    bin_edges = np.linspace(log_t.min(), log_t.max(), n_bins + 1)

    t_centers, sigma_up_bins, sigma_down_bins = [], [], []
    for b in range(n_bins):
        mask = (log_t >= bin_edges[b]) & (log_t < bin_edges[b + 1])
        if mask.sum() < 10:
            continue
        r = residuals[mask]
        t_centers.append(10 ** ((bin_edges[b] + bin_edges[b + 1]) / 2))
        r_up = r[r >= 0]
        r_down = r[r < 0]
        sigma_up_bins.append(np.std(r_up) if len(r_up) > 3 else np.nan)
        sigma_down_bins.append(np.std(np.abs(r_down)) if len(r_down) > 3 else np.nan)

    t_centers = np.array(t_centers)
    sigma_up_bins = np.array(sigma_up_bins)
    sigma_down_bins = np.array(sigma_down_bins)

    def sigma_model(t, sigma0, alpha):
        return sigma0 * t ** (-alpha)

    # Fit upside
    valid = ~np.isnan(sigma_up_bins) & (sigma_up_bins > 0)
    popt_up, _ = curve_fit(sigma_model, t_centers[valid], sigma_up_bins[valid],
                           p0=[0.5, 0.3], bounds=([0.01, -1], [5.0, 3.0]))

    # Fit downside
    valid = ~np.isnan(sigma_down_bins) & (sigma_down_bins > 0)
    popt_down, _ = curve_fit(sigma_model, t_centers[valid], sigma_down_bins[valid],
                             p0=[0.3, 0.3], bounds=([0.01, -1], [5.0, 3.0]))

    sigma0_up, alpha_up = float(popt_up[0]), float(popt_up[1])
    sigma0_down, alpha_down = float(popt_down[0]), float(popt_down[1])

    # Write to pkl
    prefix = "bm_" if model_type == "bm" else ""
    d[f"{prefix}sigma0_up"] = sigma0_up
    d[f"{prefix}alpha_up"] = alpha_up
    d[f"{prefix}sigma0_down"] = sigma0_down
    d[f"{prefix}alpha_down"] = alpha_down

    # Remove old constant sigma if present (EF)
    d.pop("sigma", None)

    with open(pkl_path, "wb") as f:
        pickle.dump(d, f)

    print(f"σ₀_up={sigma0_up:.4f}  α_up={alpha_up:.4f}")
    print(f"σ₀_down={sigma0_down:.4f}  α_down={alpha_down:.4f}")
    print(f"Asymmetry ratio (α_down/α_up): {alpha_down/alpha_up:.2f}")
    print(f"Written to {pkl_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--type", required=True, choices=["bm", "ef"])
    args = parser.parse_args()
    _fit_sigma(args.pkl, args.type)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run for both pkls**

```bash
btc_venv/bin/python3 tools/fit_sigma.py --pkl btc_app/model_data.pkl --type bm
btc_venv/bin/python3 tools/fit_sigma.py --pkl btc_app/model_data_ef.pkl --type ef
```

Expected: σ parameters printed, pkls updated.

- [ ] **Step 3: Verify pkl contents**

```bash
btc_venv/bin/python3 -c "
import pickle
for name in ['btc_app/model_data.pkl', 'btc_app/model_data_ef.pkl']:
    with open(name, 'rb') as f:
        d = pickle.load(f)
    keys = [k for k in d.keys() if 'sigma' in k or 'alpha' in k]
    print(f'{name}: {keys}')
    for k in keys:
        print(f'  {k} = {d[k]:.4f}')
"
```

- [ ] **Step 4: Commit**

```bash
git add tools/fit_sigma.py btc_app/model_data.pkl btc_app/model_data_ef.pkl archive/btc_app/model_data.pkl
git commit -m "feat: add fit_sigma.py tool and compute σ parameters for both pkls"
```

---

### Task 2: Add `_CompositeModel` base class to `btc_core.py`

**Files:**
- Modify: `archive/btc_app/btc_core.py`

- [ ] **Step 1: Write tests for _CompositeModel behavior**

Add to `btc_web/test_web.py`:

```python
class TestCompositeModelBands(unittest.TestCase):
    """Test asymmetric shrinking Gaussian band behavior."""

    def test_bands_narrow_over_time(self):
        """σ(t) decreases with increasing t."""
        from btc_core import BubbleModel
        import _app_ctx
        model = _app_ctx.PRICE_MODELS["bub"]
        # Q10 band width should be narrower at t=30 than t=5
        p10_early = float(model.price_at(0.1, 5.0))
        p50_early = float(model.price_at(0.5, 5.0))
        p10_late = float(model.price_at(0.1, 30.0))
        p50_late = float(model.price_at(0.5, 30.0))
        ratio_early = p50_early / p10_early
        ratio_late = p50_late / p10_late
        self.assertGreater(ratio_early, ratio_late)

    def test_asymmetric_bands(self):
        """Downside band should be narrower than upside at late times."""
        import _app_ctx
        model = _app_ctx.PRICE_MODELS["bub"]
        t = 30.0
        p50 = np.log10(float(model.price_at(0.5, t)))
        p10 = np.log10(float(model.price_at(0.1, t)))
        p90 = np.log10(float(model.price_at(0.9, t)))
        down_width = p50 - p10
        up_width = p90 - p50
        self.assertGreater(up_width, down_width)

    def test_quantile_ordering_preserved(self):
        """Q1 < Q10 < Q50 < Q90 < Q99 at all times."""
        import _app_ctx
        model = _app_ctx.PRICE_MODELS["bub"]
        for t in [3.0, 10.0, 30.0, 50.0]:
            prices = [float(model.price_at(q, t))
                      for q in [0.01, 0.1, 0.5, 0.9, 0.99]]
            for i in range(len(prices) - 1):
                self.assertLess(prices[i], prices[i + 1],
                    f"Ordering violated at t={t}: Q{[0.01,0.1,0.5,0.9,0.99][i]}")

    def test_q1_never_exceeds_composite(self):
        """Q1% must never exceed Q50% — the bug this change fixes."""
        import _app_ctx
        model = _app_ctx.PRICE_MODELS["bub"]
        for t in [10, 20, 30, 40, 50, 60]:
            p1 = float(model.price_at(0.01, t))
            p50 = float(model.price_at(0.5, t))
            self.assertLess(p1, p50,
                f"Q1% ({p1:.0f}) >= Q50% ({p50:.0f}) at t={t}")
```

- [ ] **Step 2: Add `_CompositeModel` base class**

Insert after `_FitsBasedModel` class (around line 346) in `archive/btc_app/btc_core.py`:

```python
class _CompositeModel:
    """Base for models with a shaped composite median and asymmetric shrinking
    Gaussian bands.

    Subclasses must set:
        self._t_grid    — time grid array (years since genesis)
        self._log_comp  — log10(composite) on the grid
        self._sigma0_up, self._alpha_up     — upside σ(t) = σ₀ × t^(-α)
        self._sigma0_down, self._alpha_down — downside σ(t)
        self.fits       — {q: {"z": z_value}} for all quantiles
        self.quantiles  — sorted list of quantile keys
        self.colors     — {q: "#hex"} color dict
    """
    quantized = True

    def _composite_log10(self, t):
        """Interpolate composite curve in log10 space at arbitrary t."""
        t = np.asarray(t, float)
        return np.interp(t, self._t_grid, self._log_comp)

    def _sigma_at(self, t, q):
        """Compute σ at time t for quantile q (asymmetric, shrinking)."""
        t = np.maximum(np.asarray(t, float), 0.5)
        if q >= 0.5:
            return self._sigma0_up * t ** (-self._alpha_up)
        else:
            return self._sigma0_down * t ** (-self._alpha_down)

    def price_at(self, q, t):
        """Price at quantile q, time t (years since genesis)."""
        t_arr = np.asarray(t, float)
        log_median = self._composite_log10(t_arr)
        z = norm.ppf(q)
        sigma = self._sigma_at(t_arr, q)
        return 10.0 ** (log_median + z * sigma)

    def interp_price(self, q, t):
        """Log-space interpolated price for arbitrary quantile."""
        if q in self.fits:
            return float(self.price_at(q, t))
        sorted_qs = self.quantiles
        lo = max((qq for qq in sorted_qs if qq <= q), default=sorted_qs[0])
        hi = min((qq for qq in sorted_qs if qq >= q), default=sorted_qs[-1])
        if lo == hi:
            return float(self.price_at(lo, t))
        frac = (q - lo) / (hi - lo)
        p_lo = np.log10(float(self.price_at(lo, t)))
        p_hi = np.log10(float(self.price_at(hi, t)))
        return 10.0 ** (p_lo + frac * (p_hi - p_lo))

    def find_percentile(self, t, price):
        """Reverse lookup: time + price → quantile."""
        sorted_qs = self.quantiles
        if not sorted_qs:
            return 0.5
        t_safe = max(float(t), 0.5)
        log_p = np.log10(max(float(price), 1e-10))
        log_ps = [np.log10(max(float(self.price_at(q, t_safe)), 1e-10))
                  for q in sorted_qs]
        if log_p <= log_ps[0]:
            return sorted_qs[0]
        if log_p >= log_ps[-1]:
            return sorted_qs[-1]
        for i in range(len(sorted_qs) - 1):
            if log_ps[i] <= log_p <= log_ps[i + 1]:
                frac = (log_p - log_ps[i]) / (log_ps[i + 1] - log_ps[i] + 1e-30)
                return sorted_qs[i] + frac * (sorted_qs[i + 1] - sorted_qs[i])
        return sorted_qs[-1]

    def _init_bands(self, sigma0_up, alpha_up, sigma0_down, alpha_down, quantiles):
        """Initialize σ parameters and build fits dict."""
        self._sigma0_up = sigma0_up
        self._alpha_up = alpha_up
        self._sigma0_down = sigma0_down
        self._alpha_down = alpha_down
        self.fits = {}
        for q in quantiles:
            self.fits[q] = {"z": float(norm.ppf(q))}
        self.quantiles = sorted(self.fits.keys())
```

- [ ] **Step 3: Commit base class (tests will fail until Task 3)**

```bash
git add archive/btc_app/btc_core.py btc_web/test_web.py
git commit -m "feat: add _CompositeModel base class and shrinking band tests"
```

---

### Task 3: Convert `BubbleModel` to `_CompositeModel`

**Files:**
- Modify: `archive/btc_app/btc_core.py` (BubbleModel class, ~line 348)

- [ ] **Step 1: Rewrite BubbleModel**

Replace the current `BubbleModel(_FitsBasedModel)` with:

```python
class BubbleModel(_CompositeModel):
    """Bubble model with asymmetric shrinking Gaussian bands around composite."""
    name = "Bubble Model"
    short_name = "bub"
    dash_style = "solid"

    def __init__(self, md):
        # Composite curve (max future bubbles)
        self._t_grid = np.asarray(md.years_plot_bm, float)
        comp = md.bm_comp_by_n[-1]
        self._log_comp = np.log10(np.maximum(np.asarray(comp, float), 1e-10))

        # Shrinking σ parameters (from pkl, fitted by tools/fit_sigma.py)
        sigma0_up = getattr(md, 'bm_sigma0_up', 0.085)
        alpha_up = getattr(md, 'bm_alpha_up', 0.132)
        sigma0_down = getattr(md, 'bm_sigma0_down', 0.075)
        alpha_down = getattr(md, 'bm_alpha_down', 0.218)

        self._init_bands(sigma0_up, alpha_up, sigma0_down, alpha_down,
                         md.QR_QUANTILES)

        # Use thermal colors (set by app.py after construction)
        self.colors = dict(md.qr_colors)
```

Note: `ModelData.__init__` in `btc_core.py` needs to load the new σ fields from the pkl. Find where `ModelData` reads pkl keys and add:

```python
self.bm_sigma0_up = d.get("bm_sigma0_up", 0.085)
self.bm_alpha_up = d.get("bm_alpha_up", 0.132)
self.bm_sigma0_down = d.get("bm_sigma0_down", 0.075)
self.bm_alpha_down = d.get("bm_alpha_down", 0.218)
```

The defaults are fallbacks so old pkls without σ fields still work.

- [ ] **Step 2: Run tests**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -k "CompositeModelBands or BubbleModel" -v
```

Expected: All band tests pass (narrowing, asymmetry, ordering, Q1<Q50).

- [ ] **Step 3: Commit**

```bash
git add archive/btc_app/btc_core.py
git commit -m "feat: convert BubbleModel to _CompositeModel with shrinking σ bands"
```

---

### Task 4: Convert `EmpiricalFloorModel` to `_CompositeModel`

**Files:**
- Modify: `archive/btc_app/btc_core.py` (EmpiricalFloorModel class)

- [ ] **Step 1: Rewrite EmpiricalFloorModel**

Replace the current standalone class with:

```python
class EmpiricalFloorModel(_CompositeModel):
    """BM Empirical Floor with asymmetric shrinking Gaussian bands."""
    name = "BM Empirical Floor"
    short_name = "ef"
    dash_style = "dashdot"

    def __init__(self, pkl_path):
        import pickle
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)

        self._slope = d["ef_support_slope"]
        self._intercept = d["ef_support_intercept"]
        self._t_grid = np.asarray(d["years_plot"], float)
        self._support_plot = np.asarray(d["support_plot"], float)
        self._comp_by_n = d["comp_by_n"]
        self._bm_r2 = d["bm_r2"]
        self._n_future_max = d["n_future_max"]

        comp = self._comp_by_n[-1]
        self._log_comp = np.log10(np.maximum(np.asarray(comp, float), 1e-10))

        # Shrinking σ parameters
        sigma0_up = d.get("sigma0_up", 0.093)
        alpha_up = d.get("alpha_up", 0.297)
        sigma0_down = d.get("sigma0_down", 0.085)
        alpha_down = d.get("alpha_down", 0.295)

        quantiles = d.get("QR_QUANTILES", [
            0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
            0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999])

        self._init_bands(sigma0_up, alpha_up, sigma0_down, alpha_down, quantiles)
        self._build_colors()

    def _build_colors(self):
        """Amber/warm palette."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(139 + 100 * frac)
            g = int(105 + 87 * frac)
            b = int(20 + 44 * frac)
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"
```

- [ ] **Step 2: Run EF tests**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -k "EmpiricalFloor" -v
```

Expected: All 7 EF tests pass.

- [ ] **Step 3: Commit**

```bash
git add archive/btc_app/btc_core.py
git commit -m "feat: convert EmpiricalFloorModel to _CompositeModel with shrinking σ bands"
```

---

### Task 5: Update `tools/build_ef_model.py` for new σ format

**Files:**
- Modify: `tools/build_ef_model.py`

- [ ] **Step 1: Replace constant σ export with placeholder**

In `build_ef_model.py`, the pkl export currently writes `"sigma": sigma`. Remove this — the σ parameters are now fitted by `tools/fit_sigma.py` as a separate step.

Remove or replace:
```python
"sigma": sigma,
```
With a comment noting that σ is fitted by `tools/fit_sigma.py`.

- [ ] **Step 2: Update the build workflow**

After running `build_ef_model.py`, run `fit_sigma.py`:

```bash
btc_venv/bin/python3 tools/build_ef_model.py
btc_venv/bin/python3 tools/fit_sigma.py --pkl btc_app/model_data_ef.pkl --type ef
```

- [ ] **Step 3: Commit**

```bash
git add tools/build_ef_model.py
git commit -m "feat: remove constant sigma from build_ef_model, use fit_sigma.py instead"
```

---

### Task 6: Run full test suite and verify

**Files:**
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Run all tests**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -30
```

Expected: All tests pass including new shrinking band tests.

- [ ] **Step 2: Test locally**

```bash
DEV=1 bash run_web.sh
```

Verify:
- Bubble tab: quantile bands follow composite shape (wavy, not straight)
- Bands narrow visibly between early and late years
- Q1% never crosses above Q50%
- Heatmap, DCA, Retire, Supercharger tabs all work with both BM and EF

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: test and visual verification fixes for shrinking Gaussian bands"
```

---

### Task 7: Deploy

- [ ] **Step 1: Push branch**

```bash
git push origin ShrinkingGaussian
```

- [ ] **Step 2: User reviews on branch before merging to master**

The user will review the branch, test on production, and decide whether to merge.
