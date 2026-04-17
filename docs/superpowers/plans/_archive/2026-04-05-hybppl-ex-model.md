# HybPPL_excess Model Integration — Implementation Plan

> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote HybPPL-on-excess into a first-class Quantoshi model (`hybppl_ex`, "HybPPL (excess)") — daily-refit, pulls BM support dynamically, registered in PRICE_MODELS.

**Architecture:** New `HybPPLExcessModel` class in `btc_core.py` inheriting from `LPPLModel`. Reads `A_sup`/`B_sup` from `ModelData` at instantiation; stores 8 oscillation params as class constants written by `tools/fit_hybppl_excess.py --update`. BM support intercept/slope flows from `CompositeResult` → `tools/model_toolkit/export.py` → `model_data.pkl` → `ModelData`.

**Spec:** `docs/superpowers/specs/2026-04-05-hybppl-ex-model.md`

---

## Pre-flight

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -q
grep -n "support_intercept\|support_slope" btc_core.py tools/model_toolkit/export.py tools/model_toolkit/composite.py
```

---

## Task 1: Expose BM support on `CompositeResult` + export to pkl

**Files:** `tools/model_toolkit/composite.py`, `tools/model_toolkit/export.py`

- [ ] **Step 1:** Add `support_intercept`/`support_slope` fields to `CompositeResult` dataclass. Populate in `build_composite()` from `support.intercept` and `support.slope`.

- [ ] **Step 2:** In `tools/model_toolkit/export.py`, export two new keys to pkl:
```python
"bm_support_intercept": composite.support_intercept,
"bm_support_slope": composite.support_slope,
```

- [ ] **Step 3: Rebuild pkl**
```bash
btc_venv/bin/python3 tools/build_bm_model.py
btc_venv/bin/python3 tools/fit_sigma.py --pkl archive/btc_app/model_data.pkl --type bm
```

- [ ] **Step 4: Verify keys present** — smoke-test reading the pkl.

- [ ] **Step 5: Commit**
```bash
git add tools/model_toolkit/composite.py tools/model_toolkit/export.py archive/btc_app/model_data.pkl
git commit -m "feat(bm): export bm_support_intercept/slope to pkl

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `ModelData` reads new keys

**Files:** `btc_core.py`

- [ ] **Step 1:** Add two attributes to `ModelData.__init__` after `self.support_bm = ...`:
```python
self.support_intercept = float(d.get("bm_support_intercept", -1.5594))
self.support_slope = float(d.get("bm_support_slope", 5.1248))
```
Fallback values preserve backward compat with old pkls.

- [ ] **Step 2: Verify via quick import test.**

- [ ] **Step 3: Commit**
```bash
git add btc_core.py
git commit -m "feat(model): ModelData.support_intercept/support_slope attributes

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Create `HybPPLExcessModel` class

**Files:** `btc_core.py`

- [ ] **Step 1: Insert new class AFTER `HybPPLModel`.**

Required:
- Inherits from `LPPLModel`
- `name = "HybPPL (excess)"`, `short_name = "hybppl_ex"`, `legend_name = "HybPPL (ex)"`
- 8 class constants: `_a0, _C1, _W_log, _PHI1, _D, _C2, _W_cal, _PHI2` (initial values from /F: a0=0.3499, C1=0.6421, W_log=7.4808, PHI1=1.4272, D=0.6607, C2=0.2315, W_cal=1.7489, PHI2=-2.1002)
- `__init__(price_years, price_prices, quantiles, a_sup=None, b_sup=None)` stores `self._A_sup`, `self._B_sup` before `super().__init__()`
- `_lppl_log10(t)` returns `A_sup + B_sup*log10(t) + a0 + damped(log-time) + undamped(calendar)`

- [ ] **Step 2: Syntax + smoke test**
```bash
btc_venv/bin/python3 -m py_compile btc_core.py && echo OK
```

- [ ] **Step 3: Commit**
```bash
git add btc_core.py
git commit -m "feat(model): HybPPLExcessModel class

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Create `tools/fit_hybppl_excess.py`

**Files:** `tools/fit_hybppl_excess.py` (NEW)

- [ ] **Step 1:** Copy `tools/fit_hybppl.py` as a template; replace with excess-fit logic.

Model: `excess = a0 + C1*t^(-D)*cos(W_log*ln(t)+PHI1) + C2*cos(W_cal*t+PHI2)`, where `excess = log_price - (A_sup + B_sup*log10(t))` and A_sup/B_sup come from `fit_support()`.

DE bounds: a0=[-1,2], C1=[0.01,3], W_log=[2,40], PHI1=[-π,π], D=[0.01,2], C2=[0,2], W_cal=[0.5,10], PHI2=[-π,π]. seed=42, maxiter=2000.

`--update` rewrites 8 class constants via regex (match `fit_hybppl.py` pattern). Order: `_a0, _C1, _W_log, _PHI1, _D, _C2, _W_cal, _PHI2`.

- [ ] **Step 2: Run without update**
```bash
btc_venv/bin/python3 tools/fit_hybppl_excess.py 2>&1 | tail -15
```
Expected: `R^2 ≈ 0.699, sigma ≈ 0.162`.

- [ ] **Step 3: Run with --update**
```bash
btc_venv/bin/python3 tools/fit_hybppl_excess.py --update 2>&1 | tail -15
```

- [ ] **Step 4: Commit** (NOT the .bak file — it's a runtime backup, not a deliverable)
```bash
git add tools/fit_hybppl_excess.py btc_core.py
git commit -m "feat(fit): tools/fit_hybppl_excess.py with --update

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Register in `PRICE_MODELS` (`btc_web/app.py`)

- [ ] **Step 1:** Import `HybPPLExcessModel` near existing `HybPPLModel` import.

- [ ] **Step 2:** After `_app_ctx.PRICE_MODELS["hybppl"] = HybPPLModel(...)` add:
```python
_app_ctx.PRICE_MODELS["hybppl_ex"] = HybPPLExcessModel(
    M.price_years, M.price_prices, M.QR_QUANTILES,
    a_sup=M.support_intercept, b_sup=M.support_slope,
)
```

- [ ] **Step 3: Boot test**
```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
grep -iE "traceback|error" /tmp/quantoshi_dev.log | head -5 || echo clean
lsof -ti :8050 | xargs -r kill -9 2>/dev/null
```

- [ ] **Step 4: Commit**
```bash
git add btc_web/app.py
git commit -m "feat: register hybppl_ex in PRICE_MODELS

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Color, label, snapshot entries

**Files:** `btc_web/_app_ctx.py`, `btc_web/figures/common.py`, `btc_web/snapshot.py`

- [ ] **Step 1:** Add `"hybppl_ex": "#9B8AFF"` to `MODEL_TRACE_COLORS` AND `PALETTES["default"]["model_colors"]`.

- [ ] **Step 2:** Add `"hybppl_ex": "HybPPL (ex)"` to `_MODEL_LABELS` in `btc_web/figures/common.py` (line ~378).

- [ ] **Step 3: Append `"hybppl_ex"` to END of 5 bitmask lists** in `btc_web/snapshot.py::_CHECKLIST_OPTIONS`:
- `bub-model-show`
- `dca-model-show`
- `ret-model-show`
- `sc-model-show`
- `hm-model-show`

APPEND-ONLY. Do not reorder existing entries.

- [ ] **Step 4: Boot test**
```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
grep -iE "traceback" /tmp/quantoshi_dev.log || echo clean
lsof -ti :8050 | xargs -r kill -9 2>/dev/null
```

- [ ] **Step 5: Commit**
```bash
git add btc_web/_app_ctx.py btc_web/figures/common.py btc_web/snapshot.py
git commit -m "feat: color + label + snapshot entries for hybppl_ex

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Model Info accordion

**Files:** `btc_web/layout/model_info.py`

- [ ] **Step 1:** Locate existing `item_id="mi-hybppl"`.

- [ ] **Step 2:** Insert new `dbc.AccordionItem(..., title="HybPPL (excess)", item_id="mi-hybppl-ex")` AFTER the HybPPL block. Include:
- Formula (LaTeX markdown with mathjax)
- Motivation paragraph (decoupled trend from oscillation)
- Brief coefficient note (A_sup/B_sup dynamic, 8 oscillation params)
- Comparison to HybPPL₂ with reference to /F
- Refit cadence line ("Refitted daily via tools/fit_hybppl_excess.py")

- [ ] **Step 3: Boot + verify**
```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
curl -sS http://localhost:8050/_dash-layout | grep -c "mi-hybppl-ex"
lsof -ti :8050 | xargs -r kill -9 2>/dev/null
```

- [ ] **Step 4: Commit**
```bash
git add btc_web/layout/model_info.py
git commit -m "feat: Model Info accordion for hybppl_ex

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Daily refit pipeline (`update_prices.py`)

- [ ] **Step 1:** Append HybPPL_excess refit block after the HybPPL block, using the same pattern as other LPPL-family refits.

- [ ] **Step 2: Commit**
```bash
git add update_prices.py
git commit -m "feat(ops): add HybPPL_excess to daily refit pipeline

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Unit tests

**Files:** `btc_web/test_web.py`

- [ ] **Step 1:** Add `TestHybPPLExcessModel` class with 4 tests:
1. `test_instantiates_with_support_params` — stores A_sup/B_sup, short_name correct.
2. `test_lppl_log10_returns_finite` — baseline at t=1,5,10,16 is finite+positive.
3. `test_included_in_price_models` — `"hybppl_ex" in _app_ctx.PRICE_MODELS`.
4. `test_support_matches_model_data` — `_A_sup`/`_B_sup` match `_app_ctx.M.support_intercept`/`support_slope` to 1e-6.

- [ ] **Step 2: Run new tests**
```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestHybPPLExcessModel -v
```

- [ ] **Step 3: Run full suite**
```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -q
# Expected: 857 passed, 5 skipped
```

- [ ] **Step 4: Commit**
```bash
git add btc_web/test_web.py
git commit -m "test: HybPPLExcessModel unit tests

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: End-to-end verification (Playwright)

- [ ] **Step 1:** Start DEV server. Navigate to /1. Verify "HybPPL (ex)" in Display Models with purple swatch. Click to activate. Verify new trace renders on the bubble chart.

- [ ] **Step 2:** Navigate to /7 (Model Info). Expand "HybPPL (excess)" accordion. Verify formula + content renders.

- [ ] **Step 3:** Kill server. (No commit.)

---

## Post-implementation

- [ ] All 857 tests pass.
- [ ] App boots clean in DEV mode.
- [ ] Bubble tab shows HybPPL (ex) in Display Models.
- [ ] Selecting it draws a chart trace.
- [ ] Model Info has mi-hybppl-ex section.
- [ ] Push + deploy to production.
