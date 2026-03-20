# BM Empirical Floor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new "BM Empirical Floor" price model to Quantoshi with a steeper support line (slope=5.3106) anchored to two observed bear-market lows, producing faster bubble convergence projections across all tabs.

**Architecture:** A standalone script (`tools/build_ef_model.py`) generates `model_data_ef.pkl` containing the EF bubble composite and residual σ. A new `EmpiricalFloorModel` class in `btc_core.py` (following the `LPPLModel` pattern — shaped median + Gaussian z-shifted bands) loads this pkl. The model is conditionally registered in `app.py` and auto-discovered by the UI via `PRICE_MODELS` iteration.

**Tech Stack:** Python, NumPy, SciPy (differential_evolution), statsmodels, pandas, pickle

**Spec:** `docs/superpowers/specs/2026-03-20-empirical-floor-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `tools/build_ef_model.py` | Create | Generate `model_data_ef.pkl` — support line, bubble fitting, composite, σ |
| `archive/btc_app/btc_core.py` | Modify | Add `EmpiricalFloorModel` class (~80 lines) |
| `btc_web/app.py` | Modify | Conditional registration (~5 lines) |
| `btc_web/layout/model_info.py` | Modify | Add accordion item (~30 lines) |
| `btc_web/layout/faq.py` | Modify | Add FAQ entry (~40 lines) |
| `docs/architecture.md` | Modify | Add EF to model table + "how to add a model" section |
| `docs/user_manual.md` | Modify | Mention EF in models section |
| `btc_web/test_web.py` | Modify | Add EF model tests |
| `btc_app/model_data_ef.pkl` | Create (generated) | EF model data |

---

### Task 1: Create `tools/build_ef_model.py`

**Files:**
- Create: `tools/build_ef_model.py`
- Output: `btc_app/model_data_ef.pkl`

This is the most complex task. The script must extract the bubble fitting pipeline from notebook cell 0 (steps 2–6) and run it with the hardcoded EF support line.

- [ ] **Step 1: Write the script**

The script must:
1. Load `BitcoinPricesDaily.csv`, compute years since genesis (2009-07-25)
2. Use hardcoded EF support: `slope=5.3106`, `intercept=-1.6246`
3. Compute `log_excess = log10(price) - (intercept + slope * log10(t))` for all data
4. Locate bubble peaks for `BUBBLE_YEARS = [2011, 2013, 2017, 2021, 2025]` within ±0.75yr windows
5. Fit `bubble_shape()` to each peak sequentially (largest first, residual subtraction) using `scipy.optimize.differential_evolution`
6. Classify major/minor bubbles
7. Predict future bubbles (extrapolate trend in parameters)
8. Build composite curves for N=1..`n_future_max` future bubbles
9. Compute σ = std(log10(actual) - log10(composite)) on historical data
10. Export pkl dict with keys:
    - `ef_support_slope` (5.3106)
    - `ef_support_intercept` (-1.6246)
    - `genesis` ("2009-07-25")
    - `years_plot` (time grid, same range as notebook's `years_plot_bm`)
    - `support_plot` (support line on grid, USD)
    - `comp_by_n` (list of composite curves, USD, for N=1..max)
    - `bm_r2` (composite R²)
    - `n_future_max`
    - `sigma` (residual std in log10 space)
    - `price_years`, `price_prices` (historical arrays)
    - `QR_QUANTILES` (list of quantiles for Gaussian bands)
    - `fitted_bubbles` (list of fitted bubble param dicts)

**Implementation approach:** Extract the core fitting functions (`bubble_shape`, `fit_manual_bubble`, `bm_total_bubble`, `predict_future_bubbles`) from `SP.ipynb.2026-03-20_1059.bak` cell 0. The key difference: instead of fitting the support line from data (STEP 1), use hardcoded values.

The script should read BUBBLE_YEARS, BUBBLE_YEAR_WINDOW, FIT_CONTEXT_YR, FIT_RISE_LOOKBACK_YR, N_PREDICT_MAJOR, MAX_MAJOR_BUBBLES, MAJOR_THRESHOLD_FRAC, CAP_COMPOSITE_OVERLAP, and other config from `SP.ipynb` cell 0 automatically (same approach as `tools/sweep_support.py`), but override the support line.

```bash
btc_venv/bin/python3 tools/build_ef_model.py [--out btc_app/model_data_ef.pkl]
```

- [ ] **Step 2: Run the script and verify output**

```bash
btc_venv/bin/python3 tools/build_ef_model.py
btc_venv/bin/python3 -c "
import pickle
with open('btc_app/model_data_ef.pkl', 'rb') as f:
    d = pickle.load(f)
print('Keys:', sorted(d.keys()))
print('R²:', d['bm_r2'])
print('sigma:', d['sigma'])
print('n_future_max:', d['n_future_max'])
print('comp_by_n shapes:', [c.shape for c in d['comp_by_n']])
"
```

Expected: R² ≈ 0.9932, sigma ≈ 0.30, comp_by_n with 3+ entries.

- [ ] **Step 3: Commit**

```bash
git add tools/build_ef_model.py btc_app/model_data_ef.pkl
git commit -m "feat: add build_ef_model.py script and generate initial EF pkl"
```

---

### Task 2: Add `EmpiricalFloorModel` class to `btc_core.py`

**Files:**
- Modify: `archive/btc_app/btc_core.py` (after `LPPLModel` class, around line 504)
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write tests for EmpiricalFloorModel**

Add to `btc_web/test_web.py`:

```python
class TestEmpiricalFloorModel(unittest.TestCase):
    """Tests for EmpiricalFloorModel."""

    @classmethod
    def setUpClass(cls):
        from btc_core import EmpiricalFloorModel
        import os
        pkl = os.path.join(os.path.dirname(__file__), "..", "btc_app", "model_data_ef.pkl")
        if not os.path.exists(pkl):
            raise unittest.SkipTest("model_data_ef.pkl not found")
        cls.model = EmpiricalFloorModel(pkl)

    def test_protocol_fields(self):
        """Model has all required PriceModel fields."""
        self.assertEqual(self.model.name, "BM Empirical Floor")
        self.assertEqual(self.model.short_name, "ef")
        self.assertTrue(self.model.quantized)
        self.assertIsInstance(self.model.quantiles, list)
        self.assertGreater(len(self.model.quantiles), 10)
        self.assertIsInstance(self.model.colors, dict)
        self.assertIsInstance(self.model.fits, dict)
        self.assertIn(0.5, self.model.fits)

    def test_price_at_scalar(self):
        """price_at returns a positive float for scalar t."""
        p = self.model.price_at(0.5, 10.0)
        self.assertGreater(float(p), 0)

    def test_price_at_array(self):
        """price_at returns array for array t."""
        import numpy as np
        t = np.array([5.0, 10.0, 15.0])
        prices = self.model.price_at(0.5, t)
        self.assertEqual(len(prices), 3)
        self.assertTrue(all(p > 0 for p in prices))

    def test_quantile_ordering(self):
        """Higher quantiles produce higher prices at same t."""
        p10 = float(self.model.price_at(0.1, 10.0))
        p50 = float(self.model.price_at(0.5, 10.0))
        p90 = float(self.model.price_at(0.9, 10.0))
        self.assertLess(p10, p50)
        self.assertLess(p50, p90)

    def test_interp_price(self):
        """interp_price works for arbitrary quantile."""
        p = self.model.interp_price(0.37, 10.0)
        self.assertGreater(p, 0)

    def test_find_percentile(self):
        """find_percentile round-trips with price_at."""
        t = 12.0
        p50 = float(self.model.price_at(0.5, t))
        q = self.model.find_percentile(t, p50)
        self.assertAlmostEqual(q, 0.5, places=1)

    def test_dash_style(self):
        self.assertEqual(self.model.dash_style, "dashdot")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestEmpiricalFloorModel -v
```

Expected: ImportError — `EmpiricalFloorModel` doesn't exist yet.

- [ ] **Step 3: Implement EmpiricalFloorModel**

Add to `archive/btc_app/btc_core.py` after the `LPPLModel` class (after line ~503). Follow the `LPPLModel` pattern exactly:

```python
class EmpiricalFloorModel:
    """BM Empirical Floor — steeper support line through observed bear-market lows.

    Uses a bubble composite (support + fitted bubble shapes) as the median
    curve. Quantile bands are generated by Gaussian z-shift of the composite,
    like LPPLModel. The steeper support (slope ~5.31) produces faster bubble
    convergence — the "end of the 4-year cycle" model.

    Anchor points: 2010-10-05 ($0.06) and 2026-02-09 ($70,339), chosen to
    maximize KS temporal uniformity of below-line data points.
    """
    name = "BM Empirical Floor"
    short_name = "ef"
    dash_style = "dashdot"
    quantized = True

    def __init__(self, pkl_path):
        import pickle
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)

        self._slope = d["ef_support_slope"]
        self._intercept = d["ef_support_intercept"]
        self._years_plot = d["years_plot"]
        self._support_plot = d["support_plot"]       # USD on grid
        self._comp_by_n = d["comp_by_n"]             # list of USD arrays
        self._sigma = d["sigma"]
        self._bm_r2 = d["bm_r2"]
        self._n_future_max = d["n_future_max"]

        # Default composite: use n_future_max
        self._log_comp = np.log10(np.maximum(self._comp_by_n[-1], 1e-10))
        self._t_grid = d["years_plot"]

        # Build quantile fits (z-shifted, like LPPL)
        quantiles = d.get("QR_QUANTILES", [
            0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
            0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999])
        self.fits = {}
        for q in quantiles:
            z = norm.ppf(q)
            self.fits[q] = {"z_shift": z * self._sigma}
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    def _composite_log10(self, t):
        """Interpolate composite curve in log10 space at arbitrary t values."""
        t = np.asarray(t, float)
        return np.interp(t, self._t_grid, self._log_comp)

    def price_at(self, q, t):
        """Price at quantile q, time t (years since genesis)."""
        t_arr = np.asarray(t, float)
        log_median = self._composite_log10(t_arr)
        shift = self.fits[q]["z_shift"]
        return 10.0 ** (log_median + shift)

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

    def _build_colors(self):
        """Amber/warm palette — visually distinct from other models."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(139 + 100 * frac)    # 139 → 239 (dark amber → bright gold)
            g = int(105 + 87 * frac)     # 105 → 192
            b = int(20 + 44 * frac)      #  20 →  64
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"
```

Also add `EmpiricalFloorModel` to the import in `btc_core.py`'s `__all__` or ensure it's importable.

- [ ] **Step 4: Run tests to verify they pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestEmpiricalFloorModel -v
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add archive/btc_app/btc_core.py btc_web/test_web.py
git commit -m "feat: add EmpiricalFloorModel class with tests"
```

---

### Task 3: Register EF model in `app.py`

**Files:**
- Modify: `btc_web/app.py:35` (add import)
- Modify: `btc_web/app.py:148` (add registration, after S2F)

- [ ] **Step 1: Add import**

In `btc_web/app.py` line 35, add `EmpiricalFloorModel` to the import:

```python
from btc_core import load_model_data, BubbleModel, PowerLawModel, LPPLModel, ExponentialModel, S2FModel, EmpiricalFloorModel
```

- [ ] **Step 2: Add conditional registration**

After line 148 (`_app_ctx.PRICE_MODELS["s2f"] = ...`), add:

```python
# ── Empirical Floor (conditional — only if pkl exists) ────────────────
_ef_pkl = Path(__file__).parent.parent / "btc_app" / "model_data_ef.pkl"
if _ef_pkl.exists():
    _app_ctx.PRICE_MODELS["ef"] = EmpiricalFloorModel(str(_ef_pkl))
```

Add `from pathlib import Path` at top if not already imported.

- [ ] **Step 3: Test locally**

```bash
DEV=1 bash run_web.sh &
# Check that the app starts without errors
# Check that the EF model appears in Display Models checklists
kill %1
```

- [ ] **Step 4: Commit**

```bash
git add btc_web/app.py
git commit -m "feat: register EmpiricalFloorModel conditionally in app.py"
```

---

### Task 4: Add Model Info accordion item

**Files:**
- Modify: `btc_web/layout/model_info.py`

- [ ] **Step 1: Add accordion item**

Add a new `dbc.AccordionItem` in the accordion list in `_model_info_tab()`. Place it after the existing Bubble Model item. Content:

```python
dbc.AccordionItem([
    html.H6("Overview"),
    html.P(
        "The Empirical Floor uses a power law support line drawn through "
        "two observed bear-market lows: 2010-10-05 ($0.06) and "
        "2026-02-09 ($70,339). These anchor points were chosen to maximize "
        "the temporal uniformity of below-line data points (KS = 0.247), "
        "ensuring the support is equally relevant across all eras of "
        "Bitcoin\u2019s history rather than being an artifact of one crash."
    ),
    html.H6("Parameters"),
    html.P([
        "Support slope: 5.3106 (vs 5.13 standard). ",
        "Support intercept: \u22121.6246. ",
        "R\u00b2 with bubble fitting: 0.9932. ",
        "Quantile bands: Gaussian z-shifted from the bubble composite median."
    ]),
    html.H6("Convergence Narrative"),
    html.P(
        "The steeper support line means bubble amplitudes decay faster. "
        "Predicted future bubbles converge on the support rapidly \u2014 "
        "implying that the classic 4-year halving-driven boom/bust cycle "
        "is approaching its end, with Bitcoin transitioning to a more "
        "mature, lower-volatility asset. See the FAQ for the full "
        "derivation."
    ),
], title="BM Empirical Floor", item_id="mi-ef"),
```

- [ ] **Step 2: Syntax check**

```bash
btc_venv/bin/python3 -m py_compile btc_web/layout/model_info.py && echo "OK"
```

- [ ] **Step 3: Commit**

```bash
git add btc_web/layout/model_info.py
git commit -m "feat: add BM Empirical Floor to Model Info tab"
```

---

### Task 5: Add FAQ entry

**Files:**
- Modify: `btc_web/layout/faq.py`
- Create: `btc_web/assets/support_4way_loglog.jpg` (copy from repo root)

- [ ] **Step 1: Copy comparison chart to assets**

```bash
cp support_4way_loglog.jpg btc_web/assets/support_4way_loglog.jpg
```

- [ ] **Step 2: Add FAQ entry**

Add new entry to `_FAQ` list in `btc_web/layout/faq.py`, after the regime analysis entry:

```python
{
    "q": "What is the BM Empirical Floor model?",
    "a": html.Span([
        "The BM Empirical Floor is an alternate bubble model with a steeper "
        "power law support line, anchored to two observed bear-market "
        "floor prices:",
        html.Br(), html.Br(),
        html.Strong("Anchor 1: "),
        "October 5, 2010 ($0.06) \u2014 the end of Bitcoin\u2019s initial "
        "flat-price run, the earliest observable floor.",
        html.Br(),
        html.Strong("Anchor 2: "),
        "February 9, 2026 ($70,339) \u2014 selected from several candidates "
        "to maximize the temporal uniformity of below-line data points.",
        html.Br(), html.Br(),
        "The second anchor was chosen using the Kolmogorov-Smirnov (KS) "
        "statistic, which measures how evenly the below-line points are "
        "distributed across time. A good support line should have a "
        "consistent fraction of prices below it in every era, not just "
        "during one or two crashes. The standard bubble model support "
        "(KS = 0.581) clusters its below-line points in 2\u20133 bear "
        "markets; the Empirical Floor (KS = 0.247) distributes them "
        "across 8 of 10 time bins.",
        html.Br(), html.Br(),
        html.Img(src="/assets/support_4way_loglog.jpg",
                 style={"width": "100%", "maxWidth": "800px",
                        "borderRadius": "8px",
                        "marginTop": "4px", "marginBottom": "8px"}),
        html.Br(),
        html.Strong("End of the 4-year cycle: "),
        "The steeper support (slope 5.31 vs 5.13) means each successive "
        "bubble sits lower above the floor. When bubble shapes are fitted "
        "and extrapolated, future bubbles converge on the support line "
        "much faster than in the standard model \u2014 implying that the "
        "classic halving-driven boom/bust cycle is approaching its end. "
        "Bitcoin would transition from a volatile, cycle-driven asset to "
        "one with steadily diminishing oscillations around a steep but "
        "smooth power law growth path.",
    ]),
},
```

- [ ] **Step 3: Syntax check**

```bash
btc_venv/bin/python3 -m py_compile btc_web/layout/faq.py && echo "OK"
```

- [ ] **Step 4: Commit**

```bash
git add btc_web/layout/faq.py btc_web/assets/support_4way_loglog.jpg
git commit -m "feat: add BM Empirical Floor FAQ entry with comparison chart"
```

---

### Task 6: Update documentation

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/user_manual.md`

- [ ] **Step 1: Update architecture.md**

In section 3 (Tab Architecture), the price models table: add EF row:

```markdown
- **BM Empirical Floor** (`"ef"`) — steeper support (slope 5.31) with Gaussian composite bands, loaded from `model_data_ef.pkl`
```

Add new section "How to Add a New Price Model" before the appendices:

```markdown
## N. Adding a New Price Model

1. Implement the `PriceModel` protocol in `archive/btc_app/btc_core.py`:
   - Required fields: `name`, `short_name`, `quantized`, `quantiles`, `colors`, `fits`, `dash_style`
   - Required methods: `price_at(q, t)`, `interp_price(q, t)`, `find_percentile(t, price)`
   - For composite-median models (shaped curves): follow `LPPLModel` / `EmpiricalFloorModel` pattern
   - For log-linear models (straight lines in log-log): extend `_FitsBasedModel`
2. Register in `btc_web/app.py` inside the "register price models" block
3. The UI auto-discovers via `PRICE_MODELS` iteration — no layout changes needed
4. Add accordion item to `btc_web/layout/model_info.py`
5. Add entry to `btc_web/layout/faq.py` if the model warrants user-facing explanation
6. Update `docs/architecture.md` and `docs/user_manual.md`
7. Add tests to `btc_web/test_web.py`
```

- [ ] **Step 2: Update user_manual.md**

In the price models section, add:

```markdown
- **BM Empirical Floor** — An alternate bubble model with a steeper support
  line anchored to observed bear-market lows. Projects faster bubble
  convergence, suggesting the end of the 4-year halving cycle. Available
  on all tabs when enabled.
```

- [ ] **Step 3: Commit**

```bash
git add docs/architecture.md docs/user_manual.md
git commit -m "docs: add EF model and 'how to add a model' to architecture and user manual"
```

---

### Task 7: Deploy

- [ ] **Step 1: Run full test suite**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests pass including new `TestEmpiricalFloorModel`.

- [ ] **Step 2: Test locally**

```bash
DEV=1 bash run_web.sh
```

Verify in browser:
- EF appears in bubble tab Display Models checklist
- EF appears in heatmap pill bar
- EF appears in DCA/Retire/Supercharger Display Models
- Selecting EF produces different projections than standard BM
- Model Info tab has EF accordion item
- FAQ has EF entry with comparison chart

- [ ] **Step 3: Commit any fixes, push, deploy**

```bash
git push origin master
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && systemctl restart quantoshi"
```

Verify production: `https://quantoshi.xyz` shows EF model option on all tabs.
