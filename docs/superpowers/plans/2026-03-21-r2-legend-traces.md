# R² on All Legend Traces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **IMPORTANT:** After each task is implemented, dispatch a code-reviewer agent before proceeding to the next task.

**Goal:** Display per-quantile R² in the legend label of every model trace on the bubble chart where computable.

**Architecture:** Add `_compute_log_r2()` helper and `compute_model_r2()` standalone function to `btc_core.py`. Call from `app.py` at startup for all models. Append R² to legend labels in `figures/bubble.py`.

**Tech Stack:** Python, numpy, Plotly

**Spec:** `docs/superpowers/specs/2026-03-21-r2-legend-traces-design.md`

---

### Task 1: Add R² computation infrastructure

**Files:**
- Modify: `archive/btc_app/btc_core.py`
- Modify: `btc_web/app.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write failing test**

```python
class TestModelR2:
    """All registered models get r2_per_quantile after startup."""

    def test_bubble_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("bub")
        assert hasattr(mdl, "r2_per_quantile")
        assert isinstance(mdl.r2_per_quantile, dict)
        assert len(mdl.r2_per_quantile) > 0
        # R² should be between 0 and 1 for reasonable fits
        for q, r2 in mdl.r2_per_quantile.items():
            assert 0 < r2 <= 1.0, f"BM Q{q}: R²={r2}"

    def test_pl_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("pl")
        assert hasattr(mdl, "r2_per_quantile")
        assert len(mdl.r2_per_quantile) > 0
        # PL: all quantiles share same R² (same slope, shifted intercept)
        vals = list(mdl.r2_per_quantile.values())
        assert all(0 < v <= 1.0 for v in vals)

    def test_ef_model_has_r2(self):
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert hasattr(ef, "r2_per_quantile")
        assert len(ef.r2_per_quantile) > 0

    def test_s2f_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("s2f")
        assert hasattr(mdl, "r2_per_quantile")
        assert 0.5 in mdl.r2_per_quantile

    def test_lppl_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("lppl")
        assert hasattr(mdl, "r2_per_quantile")
        assert len(mdl.r2_per_quantile) > 0

    def test_exp_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("exp")
        assert hasattr(mdl, "r2_per_quantile")
        assert len(mdl.r2_per_quantile) > 0

    def test_qr_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("qr")
        assert hasattr(mdl, "r2_per_quantile")
        assert len(mdl.r2_per_quantile) > 0

    def test_ols_r2_on_model_data(self):
        assert hasattr(M, "ols_r2")
        assert isinstance(M.ols_r2, float)
        assert 0.9 < M.ols_r2 <= 1.0  # OLS should be a good fit
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestModelR2 -v`
Expected: FAIL — `r2_per_quantile` not found on any model.

- [ ] **Step 3: Add `_compute_log_r2` and `compute_model_r2` to btc_core.py**

Add these at module level in `archive/btc_app/btc_core.py` (after the imports, before the model classes):

```python
def _compute_log_r2(actual_prices, predicted_prices):
    """R² in log10 space. Returns float or None if degenerate."""
    log_a = np.log10(np.maximum(np.asarray(actual_prices, float), 1e-10))
    log_p = np.log10(np.maximum(np.asarray(predicted_prices, float), 1e-10))
    ss_res = np.sum((log_a - log_p) ** 2)
    ss_tot = np.sum((log_a - np.mean(log_a)) ** 2)
    if ss_tot == 0:
        return None
    return float(1.0 - ss_res / ss_tot)


def compute_model_r2(mdl, price_years, price_prices):
    """Compute per-quantile R² for any model with price_at() and quantiles."""
    mdl.r2_per_quantile = {}
    mask = price_years >= 1.0  # skip very early data
    t = price_years[mask]
    actual = price_prices[mask]
    if hasattr(mdl, 'quantiles') and mdl.quantiles:
        for q in mdl.quantiles:
            try:
                predicted = np.asarray(mdl.price_at(q, t), float)
                r2 = _compute_log_r2(actual, predicted)
                if r2 is not None:
                    mdl.r2_per_quantile[q] = r2
            except Exception:
                pass
    elif hasattr(mdl, 'price_at'):
        # Non-quantized (S2F): single trajectory
        try:
            predicted = np.asarray(mdl.price_at(0.5, t), float)
            r2 = _compute_log_r2(actual, predicted)
            if r2 is not None:
                mdl.r2_per_quantile[0.5] = r2
        except Exception:
            pass
```

- [ ] **Step 4: Call from app.py after model registration**

In `btc_web/app.py`, after all models are registered in `_app_ctx.PRICE_MODELS` (after the EF conditional block, around line 153), add:

```python
# ── compute per-quantile R² for all models ───────────────────────────────
from btc_core import compute_model_r2, _compute_log_r2
for _mdl in _app_ctx.PRICE_MODELS.values():
    compute_model_r2(_mdl, M.price_years, M.price_prices)

# OLS R²
_ols_pred = 10 ** (M.ols_intercept + M.ols_slope * np.log10(
    np.maximum(M.price_years[M.price_years >= 1.0], 0.1)))
M.ols_r2 = _compute_log_r2(M.price_prices[M.price_years >= 1.0], _ols_pred)
```

Note: `np` is already imported in app.py. If not, add `import numpy as np`.

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestModelR2 -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add archive/btc_app/btc_core.py btc_web/app.py btc_web/test_web.py
git commit -m "feat: compute per-quantile R² for all models at startup"
```

- [ ] **Step 7: Code review**

Dispatch `superpowers:code-reviewer` agent.

---

### Task 2: Add R² to legend labels in bubble figure

**Files:**
- Modify: `btc_web/figures/bubble.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write failing test**

```python
class TestR2InLegend:
    """Legend labels include R² where available."""

    _BASE = dict(
        selected_qs=[0.5] if 0.5 in _app_ctx.DEFAULT_MODEL.fits else [0.10],
        shade=False, show_ols=True, show_ucl=True,
        show_data=False, show_today=False,
        show_legend=True, minor_grid=False,
        show_comp=True, show_sup=True,
        xscale="log", yscale="log",
        xmin=2012, xmax=2030,
        ymin=0.01, ymax=1e7,
        n_future=3, pt_size=3, pt_alpha=0.3,
        stack=0, show_stack=False, use_lots=False, lots=[],
        comp_color="#FFD700", comp_lw=2.0,
        sup_color="#888888", sup_lw=1.5,
        palette="default",
    )

    def test_bm_quantile_has_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub"]))
        q_traces = [t for t in fig.data if t.name and "Q" in t.name
                    and "%" in t.name and "R\u00b2" in t.name
                    and not getattr(t, "legendgroup", None)]
        assert len(q_traces) > 0, "BM quantile lines should show R²"

    def test_overlay_quantile_has_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub", "pl"]))
        pl_traces = [t for t in fig.data if t.name and "Power Law" in t.name
                     and "R\u00b2" in t.name]
        assert len(pl_traces) > 0, "PL overlay lines should show R²"

    def test_ols_has_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub"]))
        ols_traces = [t for t in fig.data if t.name and t.name.startswith("OLS")]
        assert len(ols_traces) > 0
        assert "R\u00b2" in ols_traces[0].name

    def test_s2f_has_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["s2f"]))
        s2f_traces = [t for t in fig.data if t.name and "Stock-to-Flow" in t.name]
        assert len(s2f_traces) > 0
        assert "R\u00b2" in s2f_traces[0].name

    def test_support_no_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub"]))
        sup_traces = [t for t in fig.data if t.name and "support" in t.name]
        for t in sup_traces:
            assert "R\u00b2" not in t.name, f"Support should not have R²: {t.name}"

    def test_ucl_no_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub"]))
        ucl_traces = [t for t in fig.data if t.name and "Unfairly Cheap" in t.name]
        for t in ucl_traces:
            assert "R\u00b2" not in t.name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestR2InLegend -v`
Expected: `test_bm_quantile_has_r2`, `test_overlay_quantile_has_r2`, `test_ols_has_r2`, `test_s2f_has_r2` FAIL.

- [ ] **Step 3: Add R² helper function to bubble.py**

Add at module level in `btc_web/figures/bubble.py` (near the top, after imports):

```python
def _r2_suffix(mdl, q):
    """Return ' R²=X.XXXX' suffix if R² available for model at quantile q, else ''."""
    r2 = getattr(mdl, 'r2_per_quantile', {}).get(q)
    if r2 is not None:
        return f"  R\u00b2={r2:.4f}"
    return ""
```

- [ ] **Step 4: Add R² to main BM quantile line labels**

In the quantile lines loop (inside `if bub_active:`), find:
```python
        lbl = _fmt_q_label(q)
```
Change to:
```python
        lbl = _fmt_q_label(q) + _r2_suffix(model, q)
```

Where `model = _app_ctx.DEFAULT_MODEL` (already defined at top of function).

- [ ] **Step 5: Add R² to overlay quantile line labels**

In the overlay loop, find:
```python
                lbl = f"{mdl.name} {_fmt_q_label(q, '')}"
```
Change to:
```python
                lbl = f"{mdl.name} {_fmt_q_label(q, '')}" + _r2_suffix(mdl, q)
```

- [ ] **Step 6: Add R² to non-quantized overlay labels (S2F)**

In the non-quantized overlay branch, find:
```python
            lbl = mdl.name
```
Change to:
```python
            lbl = mdl.name + _r2_suffix(mdl, 0.5)
```

- [ ] **Step 7: Add R² to OLS label**

Find the OLS trace creation:
```python
        mode="lines", name="OLS",
```
Change to:
```python
        mode="lines", name=f"OLS  R\u00b2={m.ols_r2:.4f}" if hasattr(m, 'ols_r2') and m.ols_r2 else "OLS",
```

- [ ] **Step 8: Run tests**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestR2InLegend btc_web/test_web.py::TestBuildBubbleFigure btc_web/test_web.py::TestBubbleModelGating -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add btc_web/figures/bubble.py btc_web/test_web.py
git commit -m "feat: display R² in legend labels for all model traces"
```

- [ ] **Step 10: Code review**

Dispatch `superpowers:code-reviewer` agent.

---

### Task 3: Manual verification

- [ ] **Step 1: Start dev server**

```bash
DEV=1 bash run_web.sh
```

- [ ] **Step 2: Verify R² on BM quantile lines**

Select Q5%, Q50%, Q95% → each legend entry shows R² value (e.g., `Q50%  R²=0.9512`)

- [ ] **Step 3: Verify R² on overlay models**

Enable PL, EF, LPPL, Exp, S2F in Display Models → all overlay traces show R²

- [ ] **Step 4: Verify R² on OLS**

Enable OLS toggle → legend shows `OLS  R²=0.95XX`

- [ ] **Step 5: Verify no R² on support/UCL**

Enable support + UCL → labels remain `"Bubble support"` and `"Unfairly Cheap Line"` (no R²)

- [ ] **Step 6: Verify composite R² unchanged**

Composite traces still show `R²=X.XXXX` from the composite fit (not per-quantile R²)

- [ ] **Step 7: Final code review**

Dispatch `superpowers:requesting-code-review` to review ALL changes against the spec.
