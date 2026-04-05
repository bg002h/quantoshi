# Component Decomposition Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-model additive component decomposition overlay on Tab 1 (Bubble + QR Overlay). Users pick a model from a family dropdown, see its additive terms as a dynamic checklist, and each checked component plots as its own dotted trace. A "Σ Sum of selected" entry adds a bold solid trace summing the selected components.

**Architecture:** New `component_names` class attribute + `components(t) → dict[str, ndarray]` method added to 14 decomposable model classes in `btc_core.py`. Web layer gains a family dropdown + dynamic checklist + two callbacks (options populator, value pruner) + chart trace builder. Snapshot sharing supports the new controls as plain-JSON entries (dynamic option set, no bitmask).

**Tech Stack:** Python (NumPy, Dash 4.0.0, Plotly 6.6.0), existing `btc_web/` layout + callbacks + figures packages.

**Spec:** `docs/superpowers/specs/2026-04-05-component-decomposition-design.md`

---

## Pre-flight

Run these once before starting:

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -q 2>&1 | tail -3
# Expected: 857 passed, 5 skipped

grep -n "class LPPLModel\|class LinPPLModel\|class HybPPLModel\|class HybPPLExcessModel\|class BubbleModel\|class EmpiricalFloorModel" btc_core.py
# Expected: 9 matches (LPPL family 1/2/3/4, W variants, N13 variants, LinPPL, HybPPL, HybPPLExcess, BubbleModel, EmpiricalFloorModel)
```

---

## Task 1: LPPL family components

**Files:**
- Modify: `btc_core.py` (LPPLModel, LPPL2Model, LPPL3Model, LPPL4Model classes)
- Test: `btc_web/test_web.py`

LPPL formula decomposition by literal `+` split (outside trigonometric args):

- **LPPLModel** (3 components): `A + B·log₁₀(t) + C·t^(-D)·cos(W·ln(t)+φ)`
- **LPPL2Model** (4 components): adds `+ C2·cos(W2·ln(t)+φ2)` (undamped)
- **LPPL3Model** (5 components): adds `+ C3·cos(W3·ln(t)+φ3)` (undamped)
- **LPPL4Model** (6 components): adds `+ C4·cos(W4·ln(t)+φ4)` (undamped)

Weighted variants (`LPPLModelW`, `LPPL2ModelW`, `LPPL3ModelW`, `LPPL4ModelW`) and N13 variants (`LPPL4ModelN13`, `LPPL4ModelWN13`) inherit `components()` automatically — they override class constants (`_C`, `_W`, etc.) but not `_lppl_log10`.

- [ ] **Step 1: Write the failing test**

Add to `btc_web/test_web.py`, placed right before `# Section: MC model interface verification`:

```python
class TestLPPLComponentDecomposition:
    """LPPL family: sum(components(t)) == _lppl_log10(t) to 1e-10."""

    T_TEST = np.array([1.0, 5.0, 10.0, 16.0, 30.0, 50.0])

    def _assert_invariant(self, model):
        comps = model.components(self.T_TEST)
        assert set(comps.keys()) == set(model.component_names), (
            f"{type(model).__name__}: components() keys != component_names")
        total = sum(comps.values())
        expected = model._lppl_log10(self.T_TEST)
        np.testing.assert_allclose(
            total, expected, rtol=0, atol=1e-10,
            err_msg=f"{type(model).__name__}: sum(components) != _lppl_log10")

    def test_lppl_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["lppl"])

    def test_lppl2_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["lp2"])

    def test_lppl3_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["lp3"])

    def test_lppl4_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["lp4"])

    def test_lppl_weighted_variants_inherit(self):
        """All 4 weighted variants reuse their base class's components()."""
        import _app_ctx
        for key in ("lppl_w", "lp2_w", "lp3_w", "lp4_w"):
            self._assert_invariant(_app_ctx.PRICE_MODELS[key])

    def test_lppl4_n13_variants_inherit(self):
        import _app_ctx
        for key in ("lp4_n13", "lp4_w_n13"):
            self._assert_invariant(_app_ctx.PRICE_MODELS[key])

    def test_lppl_component_count(self):
        import _app_ctx
        assert len(_app_ctx.PRICE_MODELS["lppl"].component_names) == 3
        assert len(_app_ctx.PRICE_MODELS["lp2"].component_names) == 4
        assert len(_app_ctx.PRICE_MODELS["lp3"].component_names) == 5
        assert len(_app_ctx.PRICE_MODELS["lp4"].component_names) == 6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestLPPLComponentDecomposition -v 2>&1 | tail -20
```
Expected: FAIL with `AttributeError: 'LPPLModel' object has no attribute 'components'`

- [ ] **Step 3: Implement in btc_core.py**

Add to `LPPLModel` class (right after `_lppl_log10` method, around line 647):

```python
    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9_log)",
    ]

    def components(self, t):
        """Additive terms in log10 space. sum(values) == _lppl_log10(t)."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":            np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)": self._B * np.log10(t_safe),
            "damped osc (\u03c9_log)":  self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
        }
```

Add to `LPPL2Model` class (after its `_lppl_log10`, around line 735):

```python
    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9\u2081)",
        "undamped osc (\u03c9\u2082)",
    ]

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":              np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":  self._B * np.log10(t_safe),
            "damped osc (\u03c9\u2081)":  self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
            "undamped osc (\u03c9\u2082)": self._C2 * np.cos(
                self._W2 * np.log(t_safe) + self._PHI2),
        }
```

Add to `LPPL3Model` class (after its `_lppl_log10`, around line 776):

```python
    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9\u2081)",
        "undamped osc (\u03c9\u2082)",
        "undamped osc (\u03c9\u2083)",
    ]

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":    self._B * np.log10(t_safe),
            "damped osc (\u03c9\u2081)":    self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
            "undamped osc (\u03c9\u2082)":  self._C2 * np.cos(
                self._W2 * np.log(t_safe) + self._PHI2),
            "undamped osc (\u03c9\u2083)":  self._C3 * np.cos(
                self._W3 * np.log(t_safe) + self._PHI3),
        }
```

Add to `LPPL4Model` class (after its `_lppl_log10`, around line 873):

```python
    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9\u2081)",
        "undamped osc (\u03c9\u2082)",
        "undamped osc (\u03c9\u2083)",
        "undamped osc (\u03c9\u2084)",
    ]

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":    self._B * np.log10(t_safe),
            "damped osc (\u03c9\u2081)":    self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
            "undamped osc (\u03c9\u2082)":  self._C2 * np.cos(
                self._W2 * np.log(t_safe) + self._PHI2),
            "undamped osc (\u03c9\u2083)":  self._C3 * np.cos(
                self._W3 * np.log(t_safe) + self._PHI3),
            "undamped osc (\u03c9\u2084)":  self._C4 * np.cos(
                self._W4 * np.log(t_safe) + self._PHI4),
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestLPPLComponentDecomposition -v 2>&1 | tail -12
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add btc_core.py btc_web/test_web.py
git commit -m "$(cat <<'EOF'
feat(decomp): components() + component_names on LPPL family

LPPLModel (3 comps), LPPL2Model (4), LPPL3Model (5), LPPL4Model (6).
Weighted (LPPLW/LPPL2W/LPPL3W/LPPL4W) and N13 (LPPL4N13/LPPL4WN13)
variants inherit automatically — they override constants, not
_lppl_log10.

Invariant sum(components(t)) == _lppl_log10(t) verified to 1e-10
across all 10 LPPL variants.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: LinPPL components

**Files:**
- Modify: `btc_core.py` (LinPPLModel class)
- Test: `btc_web/test_web.py`

LinPPL uses `cos(W_cal · t + φ)` (calendar-time argument) instead of LPPL's `cos(W · ln(t) + φ)`. 3 components.

- [ ] **Step 1: Write the failing test**

Append to `TestLPPLComponentDecomposition` class in `btc_web/test_web.py`:

```python
    def test_linppl_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["linppl"])

    def test_linppl_component_count(self):
        import _app_ctx
        assert len(_app_ctx.PRICE_MODELS["linppl"].component_names) == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestLPPLComponentDecomposition::test_linppl_invariant -v 2>&1 | tail -10
```
Expected: FAIL — LinPPLModel inherits LPPLModel's `components()` which uses `W · ln(t)`, but LinPPL's `_lppl_log10` uses `W · t`, so the invariant fails.

- [ ] **Step 3: Implement in btc_core.py**

Add to `LinPPLModel` class (after its `_lppl_log10`, around line 1062):

```python
    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9_cal\u00b7t)",
    ]

    def components(self, t):
        """Calendar-time oscillation (cos(W_cal * t + phi)), not log-time."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                    np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":        self._B * np.log10(t_safe),
            "damped osc (\u03c9_cal\u00b7t)":   self._C * t_safe ** (-self._D) * np.cos(
                self._W * t_safe + self._PHI),
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestLPPLComponentDecomposition::test_linppl_invariant btc_web/test_web.py::TestLPPLComponentDecomposition::test_linppl_component_count -v 2>&1 | tail -10
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add btc_core.py btc_web/test_web.py
git commit -m "$(cat <<'EOF'
feat(decomp): components() on LinPPLModel

LinPPL uses cos(W_cal * t + phi) (calendar time) instead of LPPL's
cos(W * ln(t) + phi). 3 components. Invariant verified.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: HybPPL components

**Files:**
- Modify: `btc_core.py` (HybPPLModel class)
- Test: `btc_web/test_web.py`

HybPPL combines a log-time damped osc with a calendar-time undamped osc. 4 components.

- [ ] **Step 1: Write the failing test**

Append to `TestLPPLComponentDecomposition` class:

```python
    def test_hybppl_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["hybppl"])

    def test_hybppl_component_count(self):
        import _app_ctx
        assert len(_app_ctx.PRICE_MODELS["hybppl"].component_names) == 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestLPPLComponentDecomposition::test_hybppl_invariant -v 2>&1 | tail -10
```
Expected: FAIL — HybPPL inherits LPPL's 3-component decomposition but its `_lppl_log10` has 4 additive terms.

- [ ] **Step 3: Implement in btc_core.py**

Add to `HybPPLModel` class (after its `_lppl_log10`, around line 984):

```python
    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped log osc (\u03c9_log)",
        "undamped cal osc (\u03c9_cal)",
    ]

    def components(self, t):
        """Hybrid: log-periodic damped + linear-periodic undamped."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                        np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":            self._B * np.log10(t_safe),
            "damped log osc (\u03c9_log)":          self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
            "undamped cal osc (\u03c9_cal)":        self._C2 * np.cos(
                self._W2 * t_safe + self._PHI2),
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestLPPLComponentDecomposition::test_hybppl_invariant btc_web/test_web.py::TestLPPLComponentDecomposition::test_hybppl_component_count -v 2>&1 | tail -10
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add btc_core.py btc_web/test_web.py
git commit -m "$(cat <<'EOF'
feat(decomp): components() on HybPPLModel

4 components: A + B*log10(t) + damped_log_osc + undamped_cal_osc.
Invariant verified.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: HybPPLExcess components

**Files:**
- Modify: `btc_core.py` (HybPPLExcessModel class)
- Test: `btc_web/test_web.py`

HybPPLExcess has a separate support line (A_sup + B_sup·log₁₀(t)), a constant offset a₀, and two oscillators. 5 components.

- [ ] **Step 1: Write the failing test**

Append to `TestLPPLComponentDecomposition` class:

```python
    def test_hybppl_ex_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["hybppl_ex"])

    def test_hybppl_ex_component_count(self):
        import _app_ctx
        assert len(_app_ctx.PRICE_MODELS["hybppl_ex"].component_names) == 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestLPPLComponentDecomposition::test_hybppl_ex_invariant -v 2>&1 | tail -10
```
Expected: FAIL (inherits LPPL's 3-term decomposition but has 5 terms).

- [ ] **Step 3: Implement in btc_core.py**

Add to `HybPPLExcessModel` class (after its `_lppl_log10`, around line 1028):

```python
    component_names = [
        "A_sup",
        "B_sup\u00b7log\u2081\u2080(t)",
        "a\u2080",
        "damped log osc (\u03c9_log)",
        "undamped cal osc (\u03c9_cal)",
    ]

    def components(self, t):
        """BM support + constant + damped log-periodic + undamped calendar."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A_sup":                              np.full_like(t_safe, self._A_sup),
            "B_sup\u00b7log\u2081\u2080(t)":      self._B_sup * np.log10(t_safe),
            "a\u2080":                            np.full_like(t_safe, self._a0),
            "damped log osc (\u03c9_log)":        self._C1 * t_safe ** (-self._D) * np.cos(
                self._W_log * np.log(t_safe) + self._PHI1),
            "undamped cal osc (\u03c9_cal)":      self._C2 * np.cos(
                self._W_cal * t_safe + self._PHI2),
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestLPPLComponentDecomposition::test_hybppl_ex_invariant btc_web/test_web.py::TestLPPLComponentDecomposition::test_hybppl_ex_component_count -v 2>&1 | tail -10
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add btc_core.py btc_web/test_web.py
git commit -m "$(cat <<'EOF'
feat(decomp): components() on HybPPLExcessModel

5 components: A_sup + B_sup*log10(t) + a0 + damped_log + undamped_cal.
Invariant verified.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Composite models (BubbleModel + EmpiricalFloorModel) components

**Files:**
- Modify: `btc_core.py` (BubbleModel + EmpiricalFloorModel classes, and shared `_CompositeModel` base)
- Test: `btc_web/test_web.py`

Composite models (BM, EF) don't use `_lppl_log10` — they use `_composite_log10` (interpolating a pre-computed grid from the pkl). Decomposition splits into `support + bubbles` where `bubbles = log_composite - log_support`.

For BM: grid comes from `ModelData.support_bm` (USD) on `ModelData.years_plot_bm`. Need to add `self._log_support` to `BubbleModel.__init__`.
For EF: already has `self._support_plot` (USD) on `self._t_grid`. Just compute `self._log_support` in `__init__`.

- [ ] **Step 1: Write the failing test**

Append new class to `btc_web/test_web.py` (right after `TestLPPLComponentDecomposition`):

```python
class TestCompositeComponentDecomposition:
    """BM / EF: sum(components(t)) == _composite_log10(t) to 1e-10."""

    T_TEST = np.array([1.0, 5.0, 10.0, 16.0, 30.0, 50.0])

    def _assert_composite_invariant(self, model):
        comps = model.components(self.T_TEST)
        assert set(comps.keys()) == set(model.component_names)
        total = sum(comps.values())
        expected = model._composite_log10(self.T_TEST)
        np.testing.assert_allclose(
            total, expected, rtol=0, atol=1e-10,
            err_msg=f"{type(model).__name__}: sum(components) != _composite_log10")

    def test_bm_invariant(self):
        import _app_ctx
        self._assert_composite_invariant(_app_ctx.PRICE_MODELS["bub"])

    def test_bm_component_count(self):
        import _app_ctx
        assert _app_ctx.PRICE_MODELS["bub"].component_names == ["support", "bubbles"]

    def test_ef_invariant(self):
        import _app_ctx
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded (model_data_ef.pkl absent)")
        self._assert_composite_invariant(ef)

    def test_ef_component_count(self):
        import _app_ctx
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert ef.component_names == ["support", "bubbles"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCompositeComponentDecomposition -v 2>&1 | tail -15
```
Expected: FAIL — `AttributeError: 'BubbleModel' object has no attribute 'components'`.

- [ ] **Step 3: Implement in btc_core.py — shared method on `_CompositeModel`**

Add `component_names` + `components()` to the `_CompositeModel` base class (after the `_init_bands` method, around line 498):

```python
    component_names = ["support", "bubbles"]

    def components(self, t):
        """Composite decomposition: support + bubbles (both in log10 space).

        sum(components(t)) == self._composite_log10(t).
        """
        t = np.asarray(t, float)
        log_support = np.interp(t, self._t_grid, self._log_support)
        log_composite = self._composite_log10(t)
        return {
            "support": log_support,
            "bubbles": log_composite - log_support,
        }
```

Then ensure both subclasses populate `self._log_support`:

In `BubbleModel.__init__` (around line 528 — right after `self._log_comp` line, before `self._init_bands(...)`):

```python
        # Support line (log10 USD) for component decomposition
        self._log_support = np.log10(np.maximum(
            np.asarray(md.support_bm, float), 1e-10))
```

In `EmpiricalFloorModel.__init__` (around line 1167 — right after `self._log_comp` line):

```python
        # Support line (log10 USD) for component decomposition
        self._log_support = np.log10(np.maximum(self._support_plot, 1e-10))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCompositeComponentDecomposition -v 2>&1 | tail -12
```
Expected: `4 passed` (or 2 passed + 2 skipped if EF pkl absent)

- [ ] **Step 5: Commit**

```bash
git add btc_core.py btc_web/test_web.py
git commit -m "$(cat <<'EOF'
feat(decomp): components() on composite models (BM + EF)

_CompositeModel base class now has 2-component decomposition:
support + bubbles. BubbleModel + EmpiricalFloorModel populate
self._log_support in __init__ from their respective pkl grids.

Invariant sum(components) == _composite_log10(t) verified.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: DECOMP registry + palette in _app_ctx.py

**Files:**
- Modify: `btc_web/_app_ctx.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write the failing test**

Add new class in `btc_web/test_web.py` (after `TestCompositeComponentDecomposition`):

```python
class TestDecompRegistry:
    def test_families_keys(self):
        import _app_ctx
        expected = {"bub", "ef", "lppl", "linppl", "hybppl", "hybppl_ex"}
        assert set(_app_ctx.DECOMP_FAMILIES.keys()) == expected

    def test_families_labels(self):
        import _app_ctx
        assert _app_ctx.DECOMP_FAMILIES["bub"] == "BM"
        assert _app_ctx.DECOMP_FAMILIES["lppl"] == "LPPL (family)"
        assert _app_ctx.DECOMP_FAMILIES["hybppl_ex"] == "HybPPL (ex)"

    def test_palette_has_all_four_schemes(self):
        import _app_ctx
        assert set(_app_ctx.DECOMP_COLORS.keys()) == {"default", "cb-brian", "cb-rg", "cb-full"}
        for key, colors in _app_ctx.DECOMP_COLORS.items():
            assert len(colors) == 7, f"{key} palette has {len(colors)} colors, expected 7"
            for c in colors:
                assert c.startswith("#") and len(c) == 7

    def test_sum_color_has_all_four_schemes(self):
        import _app_ctx
        assert set(_app_ctx.DECOMP_SUM_COLOR.keys()) == {"default", "cb-brian", "cb-rg", "cb-full"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestDecompRegistry -v 2>&1 | tail -10
```
Expected: FAIL — `AttributeError: module '_app_ctx' has no attribute 'DECOMP_FAMILIES'`

- [ ] **Step 3: Implement in _app_ctx.py**

Add after the `MODEL_TRACE_COLORS` dict (around line 58, before `# ── Color palettes`):

```python
# Component decomposition — family dropdown options and trace palette.
# The "lppl" family is resolved at render time via the LPPL config panel.
DECOMP_FAMILIES = {
    "bub":       "BM",
    "ef":        "EF",
    "lppl":      "LPPL (family)",
    "linppl":    "LinPPL",
    "hybppl":    "HybPPL",
    "hybppl_ex": "HybPPL (ex)",
}

# 7-color decomposition palette per color scheme (cycles if model has >7 comps)
DECOMP_COLORS = {
    "default":  ["#E64A19", "#1976D2", "#388E3C", "#7B1FA2",
                 "#F57C00", "#00796B", "#5D4037"],
    "cb-brian": ["#D81B60", "#1E88E5", "#004D40", "#F4511E",
                 "#6A1B9A", "#00695C", "#3E2723"],
    "cb-rg":    ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                 "#0072B2", "#D55E00", "#CC79A7"],
    "cb-full":  ["#000000", "#505050", "#808080", "#A0A0A0",
                 "#C0C0C0", "#6A6A6A", "#303030"],
}

# Dedicated sum-trace color per palette (distinct from individual components)
DECOMP_SUM_COLOR = {
    "default":  "#000000",
    "cb-brian": "#000000",
    "cb-rg":    "#000000",
    "cb-full":  "#F5793A",
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestDecompRegistry -v 2>&1 | tail -10
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add btc_web/_app_ctx.py btc_web/test_web.py
git commit -m "$(cat <<'EOF'
feat(decomp): DECOMP_FAMILIES registry + trace color palettes

6 decomposable families (bub, ef, lppl, linppl, hybppl, hybppl_ex).
7-color trace palette per palette key + dedicated DECOMP_SUM_COLOR.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: _resolve_decomp_model_key helper

**Files:**
- Modify: `btc_web/callbacks/charts.py`
- Test: `btc_web/test_web.py`

This helper takes the family dropdown value + LPPL config state and returns either:
- a concrete model key (e.g., `"bub"`, `"lp3_w"`, `"hybppl_ex"`) if resolvable, OR
- `None` if family is `"lppl"` and `n_freqs` count != 1.

- [ ] **Step 1: Write the failing test**

Add new class in `btc_web/test_web.py` (after `TestDecompRegistry`):

```python
class TestResolveDecompModelKey:
    def test_non_lppl_families_pass_through(self):
        from callbacks.charts import _resolve_decomp_model_key
        assert _resolve_decomp_model_key("bub", [3], [], []) == "bub"
        assert _resolve_decomp_model_key("hybppl_ex", [3], [], []) == "hybppl_ex"
        assert _resolve_decomp_model_key("linppl", [], [], []) == "linppl"
        assert _resolve_decomp_model_key("hybppl", [1, 2], [], []) == "hybppl"
        assert _resolve_decomp_model_key("ef", [3], [], []) == "ef"

    def test_empty_family_returns_none(self):
        from callbacks.charts import _resolve_decomp_model_key
        assert _resolve_decomp_model_key("", [3], [], []) is None
        assert _resolve_decomp_model_key(None, [3], [], []) is None

    def test_lppl_single_nfreq_resolves(self):
        from callbacks.charts import _resolve_decomp_model_key
        assert _resolve_decomp_model_key("lppl", [1], [], []) == "lppl"
        assert _resolve_decomp_model_key("lppl", [2], [], []) == "lp2"
        assert _resolve_decomp_model_key("lppl", [3], [], []) == "lp3"
        assert _resolve_decomp_model_key("lppl", [4], [], []) == "lp4"

    def test_lppl_weighted_modifier(self):
        from callbacks.charts import _resolve_decomp_model_key
        assert _resolve_decomp_model_key("lppl", [1], ["weighted"], []) == "lppl_w"
        assert _resolve_decomp_model_key("lppl", [3], ["weighted"], []) == "lp3_w"
        assert _resolve_decomp_model_key("lppl", [4], ["weighted"], []) == "lp4_w"

    def test_lppl_no13_modifier(self):
        from callbacks.charts import _resolve_decomp_model_key
        assert _resolve_decomp_model_key("lppl", [4], [], ["no13"]) == "lp4_n13"
        assert _resolve_decomp_model_key("lppl", [4], ["weighted"], ["no13"]) == "lp4_w_n13"

    def test_lppl_zero_or_multi_returns_none(self):
        from callbacks.charts import _resolve_decomp_model_key
        assert _resolve_decomp_model_key("lppl", [], [], []) is None
        assert _resolve_decomp_model_key("lppl", [1, 2], [], []) is None
        assert _resolve_decomp_model_key("lppl", [1, 2, 3, 4], [], []) is None
        assert _resolve_decomp_model_key("lppl", None, [], []) is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestResolveDecompModelKey -v 2>&1 | tail -10
```
Expected: FAIL — `ImportError: cannot import name '_resolve_decomp_model_key'`

- [ ] **Step 3: Implement in callbacks/charts.py**

Add right after `_resolve_lppl_master` function (around line 240):

```python
def _resolve_decomp_model_key(family, lppl_n_freqs, lppl_weighted, lppl_no_13):
    """Translate (family, LPPL config) into a concrete model short_name.

    Returns None if family is empty OR if family is 'lppl' but exactly one
    n_freqs entry is not selected. Otherwise returns the model's short_name
    (e.g., 'bub', 'lp3_w', 'hybppl_ex').
    """
    if not family:
        return None
    if family != "lppl":
        return family
    # LPPL family: require exactly one n_freqs
    if not lppl_n_freqs or len(lppl_n_freqs) != 1:
        return None
    n = lppl_n_freqs[0]
    weighted = "weighted" in (lppl_weighted or [])
    no13 = "no13" in (lppl_no_13 or [])
    if n == 1:
        return "lppl_w" if weighted else "lppl"
    if n == 2:
        return "lp2_w" if weighted else "lp2"
    if n == 3:
        return "lp3_w" if weighted else "lp3"
    if n == 4:
        if no13:
            return "lp4_w_n13" if weighted else "lp4_n13"
        return "lp4_w" if weighted else "lp4"
    return None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestResolveDecompModelKey -v 2>&1 | tail -15
```
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/test_web.py
git commit -m "$(cat <<'EOF'
feat(decomp): _resolve_decomp_model_key helper

Translates family dropdown + LPPL config → concrete model short_name.
Returns None when family=='lppl' but n_freqs count != 1, triggering
the LPPL reminder banner in the UI.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Layout — icon + Component Decomposition section card

**Files:**
- Modify: `btc_web/layout/common.py` (add 🧬 icon)
- Modify: `btc_web/layout/bubble.py` (add new section card)

- [ ] **Step 1: Add icon**

In `btc_web/layout/common.py`, add `"Component Decomposition"` to `_SECTION_ICONS` dict. Insert right after the `"Bubble Model"` line (in the Model-config panels group):

```python
    "Component Decomposition": "\U0001F9EC",   # 🧬
```

- [ ] **Step 2: Add section card to bubble tab layout**

In `btc_web/layout/bubble.py`, insert a new section card right AFTER the `_lppl_config_panel("bub"),` call (currently line 144). The new section goes here so it appears directly below the LPPL Models config (the LPPL family dropdown entry resolves via those controls):

```python
        _section_card("Component Decomposition",
            _lbl("Model"),
            dcc.Dropdown(
                id="bub-decomp-model",
                options=[{"label": "(none)", "value": ""}] +
                        [{"label": label, "value": key}
                         for key, label in _app_ctx.DECOMP_FAMILIES.items()],
                value="", clearable=False,
            ),
            html.Div(id="bub-decomp-body", style=_STYLE_HIDDEN, children=[
                dcc.Checklist(
                    id="bub-decomp-components",
                    options=[], value=[],
                    labelStyle={"display": "block", "fontSize": "11px"},
                    inputStyle=_CB_MARGIN,
                ),
            ]),
            html.Div(id="bub-decomp-warning", children=[]),
        ),
```

- [ ] **Step 3: Syntax check + boot**

```bash
btc_venv/bin/python3 -m py_compile btc_web/layout/common.py btc_web/layout/bubble.py && echo OK
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 14
grep -iE "traceback|error " /tmp/quantoshi_dev.log | head -5 || echo clean
curl -sS http://localhost:8050/ -o /dev/null -w "HTTP %{http_code}\n"
lsof -ti :8050 | xargs -r kill -9 2>/dev/null
```
Expected: `OK`, `clean`, `HTTP 200`

- [ ] **Step 4: Commit**

```bash
git add btc_web/layout/common.py btc_web/layout/bubble.py
git commit -m "$(cat <<'EOF'
feat(decomp): Component Decomposition section card on bubble tab

New _section_card 🧬 placed after LPPL Models config. Contains:
model family dropdown, dynamic component checklist (initially hidden),
and warning banner slot for the LPPL '!=1 variant' case.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: update_decomp_options callback

**Files:**
- Modify: `btc_web/callbacks/charts.py`
- Test: `btc_web/test_web.py`

Populates the component checklist options + warning + body visibility when model dropdown OR LPPL config changes. NEVER touches `.value` (that's Callback B's job).

- [ ] **Step 1: Write the failing test**

Add to `btc_web/test_web.py` (after `TestResolveDecompModelKey`):

```python
class TestUpdateDecompOptions:
    def test_empty_family_hides_body(self):
        from callbacks.charts import update_decomp_options
        opts, warning, style = update_decomp_options("", [3], [], [])
        assert opts == []
        assert warning == []
        assert style == {"display": "none"}

    def test_bm_shows_2_components_plus_sum(self):
        from callbacks.charts import update_decomp_options
        opts, warning, style = update_decomp_options("bub", [3], [], [])
        assert style == {"display": "block"}
        assert warning == []
        values = [o["value"] for o in opts]
        assert values == ["support", "bubbles", "__sum__"]

    def test_hybppl_ex_shows_5_components_plus_sum(self):
        from callbacks.charts import update_decomp_options
        opts, warning, style = update_decomp_options("hybppl_ex", [3], [], [])
        assert len(opts) == 6  # 5 components + sum
        assert opts[-1]["value"] == "__sum__"

    def test_lppl_single_nfreq_shows_components(self):
        from callbacks.charts import update_decomp_options
        opts, warning, style = update_decomp_options("lppl", [3], [], [])
        assert style == {"display": "block"}
        assert warning == []
        # LPPL3 has 5 components + sum
        assert len(opts) == 6

    def test_lppl_zero_nfreq_shows_warning(self):
        from callbacks.charts import update_decomp_options
        opts, warning, style = update_decomp_options("lppl", [], [], [])
        assert opts == []
        assert style == {"display": "block"}
        assert warning != []  # has banner

    def test_lppl_multi_nfreq_shows_warning(self):
        from callbacks.charts import update_decomp_options
        opts, warning, style = update_decomp_options("lppl", [1, 2, 3], [], [])
        assert opts == []
        assert style == {"display": "block"}
        assert warning != []

    def test_lppl_weighted_modifier_resolves(self):
        from callbacks.charts import update_decomp_options
        opts, _, _ = update_decomp_options("lppl", [3], ["weighted"], [])
        # LPPL3 weighted = 5 components + sum
        assert len(opts) == 6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestUpdateDecompOptions -v 2>&1 | tail -10
```
Expected: FAIL — `ImportError: cannot import name 'update_decomp_options'`

- [ ] **Step 3: Implement in callbacks/charts.py**

Add after `_resolve_decomp_model_key` (written in Task 7):

```python
def _decomp_warning_banner(n_checked):
    """Inline banner shown when LPPL decomposition needs exactly 1 n_freqs."""
    return html.Div(
        html.Small(
            f"Pick exactly one LPPL variant in the LPPL config panel "
            f"to decompose (currently {n_checked} checked).",
            style={"color": "#b71c1c"},
        ),
        style={"padding": "6px 8px", "backgroundColor": "#fff3f3",
                "border": "1px solid #f5c6cb", "borderRadius": "4px",
                "fontSize": "11px", "marginTop": "6px"},
    )


def update_decomp_options(family, n_freqs, weighted, no_13):
    """Populate Component Decomposition checklist options + warning + visibility.

    Returns (options, warning_children, body_style). NEVER modifies
    bub-decomp-components.value (see prune_decomp_value_on_model_change).
    """
    if not family:
        return [], [], {"display": "none"}
    if family == "lppl" and len(n_freqs or []) != 1:
        return [], _decomp_warning_banner(len(n_freqs or [])), {"display": "block"}
    key = _resolve_decomp_model_key(family, n_freqs, weighted, no_13)
    if key is None:
        return [], _decomp_warning_banner(len(n_freqs or [])), {"display": "block"}
    model = _app_ctx.PRICE_MODELS.get(key)
    if model is None:
        return [], [], {"display": "none"}
    opts = [{"label": f" {name}", "value": name} for name in model.component_names]
    opts.append({"label": " \u03a3 Sum of selected", "value": "__sum__"})
    return opts, [], {"display": "block"}
```

Add the Dash `@callback` registration at module scope (right after the function, so tests can call the plain function):

```python
@callback(
    Output("bub-decomp-components", "options"),
    Output("bub-decomp-warning",    "children"),
    Output("bub-decomp-body",       "style"),
    Input("bub-decomp-model",  "value"),
    Input("lppl-n-freqs",      "value"),
    Input("lppl-weighted",     "value"),
    Input("lppl-no-13",        "value"),
    prevent_initial_call=False,
)
def _update_decomp_options_cb(family, n_freqs, weighted, no_13):
    return update_decomp_options(family, n_freqs, weighted, no_13)
```

Ensure `html` is imported in charts.py; if not, add it:

```python
from dash import html  # verify this import exists near top of file
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestUpdateDecompOptions -v 2>&1 | tail -15
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/test_web.py
git commit -m "$(cat <<'EOF'
feat(decomp): update_decomp_options callback

Populates component checklist options + LPPL-variant warning banner
+ body visibility. Separates option-update from value-update (Task 10)
to avoid snapshot-restore race.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: prune_decomp_value_on_model_change callback

**Files:**
- Modify: `btc_web/callbacks/charts.py`
- Test: `btc_web/test_web.py`

Fires only when user changes the model dropdown (not on snapshot restore). Prunes `bub-decomp-components.value` to entries still valid in new options.

- [ ] **Step 1: Write the failing test**

Add to `btc_web/test_web.py` (after `TestUpdateDecompOptions`):

```python
class TestPruneDecompValue:
    def test_empty_family_clears_value(self):
        from callbacks.charts import _prune_decomp_value
        assert _prune_decomp_value("", [{"value": "a"}], ["a"]) == []

    def test_prune_preserves_valid_values(self):
        from callbacks.charts import _prune_decomp_value
        opts = [{"value": "a"}, {"value": "b"}, {"value": "__sum__"}]
        assert _prune_decomp_value("bub", opts, ["a", "__sum__"]) == ["a", "__sum__"]

    def test_prune_drops_invalid_values(self):
        from callbacks.charts import _prune_decomp_value
        opts = [{"value": "support"}, {"value": "bubbles"}, {"value": "__sum__"}]
        # User previously had "damped osc" checked (from LPPL), now on BM
        assert _prune_decomp_value("bub", opts, ["damped osc", "support"]) == ["support"]

    def test_prune_empty_current(self):
        from callbacks.charts import _prune_decomp_value
        opts = [{"value": "a"}]
        assert _prune_decomp_value("bub", opts, []) == []
        assert _prune_decomp_value("bub", opts, None) == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestPruneDecompValue -v 2>&1 | tail -10
```
Expected: FAIL — `ImportError: cannot import name '_prune_decomp_value'`

- [ ] **Step 3: Implement in callbacks/charts.py**

Add after `update_decomp_options` (written in Task 9):

```python
def _prune_decomp_value(family, options, current):
    """Keep only currently-valid values from the checklist."""
    if not family:
        return []
    valid = {o["value"] for o in (options or [])}
    return [v for v in (current or []) if v in valid]


@callback(
    Output("bub-decomp-components", "value", allow_duplicate=True),
    Input("bub-decomp-model",       "value"),
    State("bub-decomp-components", "options"),
    State("bub-decomp-components", "value"),
    prevent_initial_call=True,
)
def _prune_decomp_value_cb(family, opts, current):
    # Only fire when bub-decomp-model was the trigger (user change),
    # never on snapshot restore.
    if ctx.triggered_id != "bub-decomp-model":
        raise dash.exceptions.PreventUpdate
    return _prune_decomp_value(family, opts, current)
```

Ensure `ctx` and `dash` are imported at the top of charts.py. Verify with:
```bash
grep -n "^from dash import\|^import dash" btc_web/callbacks/charts.py | head -5
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestPruneDecompValue -v 2>&1 | tail -10
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/test_web.py
git commit -m "$(cat <<'EOF'
feat(decomp): prune_decomp_value callback on model dropdown change

Guard via ctx.triggered_id ensures snapshot restoration of
bub-decomp-components.value is not clobbered.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Extend update_bubble chart callback + figure builder with decomposition

**Files:**
- Modify: `btc_web/callbacks/charts.py` (add decomp Inputs to update_bubble + params pass-through)
- Modify: `btc_web/figures/bubble.py` (trace builder + call site)
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write the failing test**

Add to `btc_web/test_web.py` (after `TestPruneDecompValue`):

```python
class TestDecompositionTraces:
    """Verify build_bubble_figure renders decomposition traces when active."""

    def _base_p(self, **overrides):
        """Minimal params dict matching what update_bubble builds."""
        p = dict(
            selected_qs=[0.5], shade=True, show_ols=False, show_ucl=False,
            show_data=False, show_today=False, show_legend=False,
            minor_grid=False, show_comp=False, show_sup=False,
            xscale="log", yscale="log", xmin=2015, xmax=2030,
            ymin=1, ymax=100000, n_future=0, pt_size=3, pt_alpha=0.3,
            stack=0, show_stack=False, use_lots=False,
            lots=[], legend_pos="outside", comp_color="#FFD700",
            comp_lw=2.0, sup_color="#888888", sup_lw=1.5,
            active_models=[], palette="default", scanner_lines=[],
            user_model=None, qs_mode=[],
            decomp_model="", decomp_components=[],
            lppl_n_freqs=[], lppl_weighted=[], lppl_no_13=[],
        )
        p.update(overrides)
        return p

    def test_no_model_no_extra_traces(self):
        import _app_ctx
        from figures.bubble import build_bubble_figure
        fig = build_bubble_figure(_app_ctx.M, self._base_p())
        decomp_traces = [t for t in fig.data
                         if getattr(t, 'name', None) and " | " in t.name]
        assert decomp_traces == []

    def test_decomp_adds_component_traces(self):
        import _app_ctx
        from figures.bubble import build_bubble_figure
        fig = build_bubble_figure(_app_ctx.M, self._base_p(
            decomp_model="bub", decomp_components=["support", "bubbles"]))
        trace_names = [t.name for t in fig.data if getattr(t, 'name', None)]
        bm_decomp = [n for n in trace_names if n.startswith("BM | ")]
        assert len(bm_decomp) == 2  # support + bubbles

    def test_decomp_sum_trace_appears(self):
        import _app_ctx
        from figures.bubble import build_bubble_figure
        fig = build_bubble_figure(_app_ctx.M, self._base_p(
            decomp_model="bub",
            decomp_components=["support", "bubbles", "__sum__"]))
        trace_names = [t.name for t in fig.data if getattr(t, 'name', None)]
        sum_traces = [n for n in trace_names if " | \u03a3 (" in n]
        assert len(sum_traces) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestDecompositionTraces -v 2>&1 | tail -15
```
Expected: FAIL — `KeyError: 'decomp_model'` or similar.

- [ ] **Step 3: Implement in figures/bubble.py**

`build_bubble_figure` builds a `traces = []` list, appends `go.Scatter(...)` entries, and finally constructs `fig = go.Figure(data=traces, layout=go.Layout(**layout))` at line 448. The existing `t_arr = np.linspace(max(t_lo, 0.1), t_hi, _INTERP_POINTS)` (line 59) is the plot grid. x-values are NUMERIC years-since-genesis (not dates) — e.g., `x=list(m.years_plot_bm[mask])`.

Add a helper function at the END of `btc_web/figures/bubble.py` (after the `return fig` of `build_bubble_figure`):

```python
def _add_decomposition_traces(traces, t_arr, m, p):
    """Append component decomposition traces + optional Σ sum trace.

    traces: list being built by build_bubble_figure().
    t_arr: numpy array of years-since-genesis (plot grid).
    m: ModelData object (unused here but kept for symmetry).
    p: params dict with decomp_model, decomp_components, lppl_*, palette.
    """
    import _app_ctx
    from callbacks.charts import _resolve_decomp_model_key

    family = p.get("decomp_model", "") or ""
    selected = list(p.get("decomp_components", []) or [])
    if not family or not selected:
        return

    key = _resolve_decomp_model_key(
        family,
        p.get("lppl_n_freqs", []),
        p.get("lppl_weighted", []),
        p.get("lppl_no_13", []),
    )
    if key is None:
        return
    model = _app_ctx.PRICE_MODELS.get(key)
    if model is None:
        return

    palette = p.get("palette", "default")
    colors = _app_ctx.DECOMP_COLORS.get(
        palette, _app_ctx.DECOMP_COLORS["default"])
    sum_color = _app_ctx.DECOMP_SUM_COLOR.get(
        palette, _app_ctx.DECOMP_SUM_COLOR["default"])

    comps = model.components(t_arr)
    names = [s for s in selected if s != "__sum__" and s in comps]

    x_list = list(t_arr)
    for i, name in enumerate(names):
        log_vals = comps[name]
        y_usd = list(10.0 ** log_vals)
        traces.append(go.Scatter(
            x=x_list, y=y_usd, mode="lines",
            line=dict(dash="dot", width=1.5,
                       color=colors[i % len(colors)]),
            name=f"{model.legend_name} | {name}",
            hovertemplate="%{y:$,.0f}<extra></extra>",
        ))

    if "__sum__" in selected and names:
        sum_log = comps[names[0]].copy()
        for n in names[1:]:
            sum_log = sum_log + comps[n]
        y_sum = list(10.0 ** sum_log)
        traces.append(go.Scatter(
            x=x_list, y=y_sum, mode="lines",
            line=dict(dash="solid", width=3, color=sum_color),
            name=f"{model.legend_name} | \u03a3 ({len(names)} components)",
            hovertemplate="%{y:$,.0f}<extra></extra>",
        ))
```

Then add the call site inside `build_bubble_figure()` — right BEFORE the final `fig = go.Figure(data=traces, ...)` line (around line 448):

```python
    # ── component decomposition traces (dotted individual + solid Σ sum) ─────
    _add_decomposition_traces(traces, t_arr, m, p)

    fig = go.Figure(data=traces, layout=go.Layout(**layout))
```

Verify the call site with:
```bash
grep -n "fig = go.Figure" btc_web/figures/bubble.py
```

- [ ] **Step 4: Extend update_bubble in callbacks/charts.py**

In the `@callback` decorator for `update_bubble` (lines 247-275), insert two new Inputs immediately after `Input("lppl-no-13", "value"),`:

```python
    Input("bub-decomp-model",       "value"),
    Input("bub-decomp-components",  "value"),
```

Update the `update_bubble` function signature (lines 277-282). Add two new positional params after `lppl_no_13,`:

OLD:
```python
def update_bubble(_first_render, sel_qs, adv_qs, toggles, bubble_toggles,
                  xscale, yscale, xrange, yrange,
                  n_future, ptsize, ptalpha, stack, show_stack, use_lots, legend_pos, model_show,
                  lppl_n_freqs, lppl_weighted, lppl_no_13, lots_data,
                  palette_key, user_model_store=None,
                  qs_mode=None, scan_active=None, scan_q_val=None):
```

NEW:
```python
def update_bubble(_first_render, sel_qs, adv_qs, toggles, bubble_toggles,
                  xscale, yscale, xrange, yrange,
                  n_future, ptsize, ptalpha, stack, show_stack, use_lots, legend_pos, model_show,
                  lppl_n_freqs, lppl_weighted, lppl_no_13,
                  decomp_model, decomp_components,
                  lots_data,
                  palette_key, user_model_store=None,
                  qs_mode=None, scan_active=None, scan_q_val=None):
```

Inside the function, inside the `dict(...)` passed to `_get_bubble_fig(...)` (starts line 306, ends line 336), add FIVE new keys right before the closing `))`:

```python
        qs_mode = qs_mode or [],
        decomp_model       = decomp_model or "",
        decomp_components  = list(decomp_components or []),
        lppl_n_freqs       = list(lppl_n_freqs or []),
        lppl_weighted      = list(lppl_weighted or []),
        lppl_no_13         = list(lppl_no_13 or []),
    ))
```

(The first line `qs_mode = qs_mode or [],` is existing; the 5 below are new.)

- [ ] **Step 5: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestDecompositionTraces -v 2>&1 | tail -15
```
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/figures/bubble.py btc_web/test_web.py
git commit -m "$(cat <<'EOF'
feat(decomp): component traces on bubble chart

Extended update_bubble callback with 2 new Inputs
(bub-decomp-model, bub-decomp-components) + LPPL config pass-through.
Added _add_decomposition_traces() helper that renders:
- each checked component as dotted 1.5px trace in palette color
- Σ Sum of selected as solid 3px DECOMP_SUM_COLOR trace

Trace names: "{legend_name} | {component_name}" and
"{legend_name} | Σ ({n} components)".

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Extend auto_bubble_yrange with decomposition Inputs

**Files:**
- Modify: `btc_web/callbacks/charts.py`
- Test: `btc_web/test_web.py`

When decomposition is active and auto-y is on, the computed Y-range must include component min/max values. Otherwise constants like `A = -1.15` (horizontal line at $0.07) push the range off the chart.

- [ ] **Step 1: Write the failing test**

Add to `btc_web/test_web.py` (after `TestDecompositionTraces`):

```python
class TestAutoYWithDecomposition:
    def test_decomp_a_constant_expands_yrange_low(self):
        """Checking LPPL's 'A (constant)' (log10 ~= -1.15) should drop
        Y-range low-end to at least -1.0."""
        from callbacks.charts import auto_bubble_yrange
        yr = auto_bubble_yrange(
            [2015, 2025], ["yes"], "log", ["bub"],
            "lppl", ["A (constant)"], [1], [], [],
            [0.5],
        )
        assert yr[0] <= -1.0, f"Y-low {yr[0]} should extend to include A constant"

    def test_no_decomp_normal_yrange(self):
        """Without decomp, Y-range is derived from quantiles only."""
        from callbacks.charts import auto_bubble_yrange
        yr = auto_bubble_yrange(
            [2020, 2030], ["yes"], "log", ["bub"],
            "", [], [3], [], [],
            [0.5],
        )
        assert yr[0] >= -1.5 and yr[1] <= 9.0
        assert yr[1] > yr[0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestAutoYWithDecomposition -v 2>&1 | tail -10
```
Expected: FAIL — either TypeError (new params not accepted) or AssertionError (Y-range not extended).

- [ ] **Step 3: Implement in callbacks/charts.py**

Update the `@callback` decorator for `auto_bubble_yrange` (lines 489-497). Add 5 new Inputs right before `State("bub-qs", "value"),`:

OLD:
```python
@callback(
    Output("bub-yrange", "value", allow_duplicate=True),
    Input("bub-xrange",  "value"),
    Input("bub-auto-y",  "value"),
    Input("bub-yscale",  "value"),
    Input("bub-model-show", "value"),
    State("bub-qs",      "value"),
    prevent_initial_call=True,
)
def auto_bubble_yrange(xrange, auto_y, yscale, model_show, sel_qs):
```

NEW:
```python
@callback(
    Output("bub-yrange", "value", allow_duplicate=True),
    Input("bub-xrange",  "value"),
    Input("bub-auto-y",  "value"),
    Input("bub-yscale",  "value"),
    Input("bub-model-show", "value"),
    Input("bub-decomp-model",       "value"),
    Input("bub-decomp-components",  "value"),
    Input("lppl-n-freqs",           "value"),
    Input("lppl-weighted",          "value"),
    Input("lppl-no-13",             "value"),
    State("bub-qs",      "value"),
    prevent_initial_call=True,
)
def auto_bubble_yrange(xrange, auto_y, yscale, model_show,
                       decomp_model, decomp_components,
                       lppl_n_freqs, lppl_weighted, lppl_no_13,
                       sel_qs):
```

The test (Step 1) uses positional args matching this new signature order.

- [ ] **Step 3b: Update 2 existing tests that call the old signature**

In `btc_web/test_web.py`, 3 existing call sites of `auto_bubble_yrange` need the 5 new params added. Find them with:
```bash
grep -n "auto_bubble_yrange(" btc_web/test_web.py
```

Apply these 3 edits:

**Edit 1** — around line 3093 (`test_no_auto_prevents_update`):
OLD:
```python
                auto_bubble_yrange(
                    xrange=[2015, 2030], auto_y=[], yscale="log",
                    model_show=[], sel_qs=[0.5],
                )
```
NEW:
```python
                auto_bubble_yrange(
                    xrange=[2015, 2030], auto_y=[], yscale="log",
                    model_show=[],
                    decomp_model="", decomp_components=[],
                    lppl_n_freqs=[], lppl_weighted=[], lppl_no_13=[],
                    sel_qs=[0.5],
                )
```

**Edit 2** — around line 3101 (`test_returns_yrange`):
OLD:
```python
            result = auto_bubble_yrange(
                xrange=[2015, 2030], auto_y=["yes"], yscale="log",
                model_show=[], sel_qs=[0.5],
            )
```
NEW:
```python
            result = auto_bubble_yrange(
                xrange=[2015, 2030], auto_y=["yes"], yscale="log",
                model_show=[],
                decomp_model="", decomp_components=[],
                lppl_n_freqs=[], lppl_weighted=[], lppl_no_13=[],
                sel_qs=[0.5],
            )
```

**Edit 3** — around line 4729 (`test_auto_y_no_bub_uses_fallback`):
OLD:
```python
            result = auto_bubble_yrange([2012, 2030], ["yes"], "log", [], [0.5])
```
NEW:
```python
            result = auto_bubble_yrange(
                [2012, 2030], ["yes"], "log", [],
                "", [], [], [], [],
                [0.5])
```

Inside the function, right before the final `return [round(y_lo, 1), round(y_hi, 1)]`, add decomposition extension:

```python
    # Extend Y-range to cover active decomposition components
    key = _resolve_decomp_model_key(
        decomp_model or "", lppl_n_freqs, lppl_weighted, lppl_no_13)
    if key and decomp_components:
        comp_model = _app_ctx.PRICE_MODELS.get(key)
        if comp_model is not None:
            t_decomp = np.linspace(max(t_lo, 0.1), t_hi, 100)
            comps = comp_model.components(t_decomp)
            names = [s for s in decomp_components if s != "__sum__" and s in comps]
            for name in names:
                log_vals = comps[name]
                y_lo = min(y_lo, float(np.floor(np.min(log_vals) * 2) / 2))
                y_hi = max(y_hi, float(np.ceil(np.max(log_vals) * 2) / 2))
            # Sum trace range
            if "__sum__" in decomp_components and names:
                sum_log = sum(comps[n] for n in names)
                y_lo = min(y_lo, float(np.floor(np.min(sum_log) * 2) / 2))
                y_hi = max(y_hi, float(np.ceil(np.max(sum_log) * 2) / 2))
            # Re-clamp to absolute limits
            y_lo = max(-1.5, min(y_lo, 6.0))
            y_hi = min(y_cap, max(y_hi, 1.0))

    return [round(y_lo, 1), round(y_hi, 1)]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestAutoYWithDecomposition -v 2>&1 | tail -10
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/test_web.py
git commit -m "$(cat <<'EOF'
feat(decomp): auto_bubble_yrange includes component min/max

When decomposition is active, Y auto-fit extends to cover component
values (otherwise 'A (constant)' at ~$0.07 falls off the chart).
5 new Inputs added to callback.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Snapshot entries + TAB_CONTROLS

**Files:**
- Modify: `btc_web/snapshot.py`
- Modify: `btc_web/callbacks/routing.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write the failing test**

Add to `btc_web/test_web.py` (after `TestAutoYWithDecomposition`):

```python
class TestDecompSnapshot:
    def test_decomp_fields_in_snapshot_controls(self):
        from snapshot import _SNAPSHOT_CONTROLS
        cids = {cid for cid, _ in _SNAPSHOT_CONTROLS}
        assert "bub-decomp-model" in cids
        assert "bub-decomp-components" in cids

    def test_decomp_fields_in_bubble_tab_controls(self):
        from callbacks.routing import _TAB_CONTROLS
        assert "bub-decomp-model" in _TAB_CONTROLS["bubble"]
        assert "bub-decomp-components" in _TAB_CONTROLS["bubble"]

    def test_decomp_not_bitmask_encoded(self):
        """Dynamic option set — decomp-components stored as plain list."""
        from snapshot import _CHECKLIST_OPTIONS
        assert "bub-decomp-components" not in _CHECKLIST_OPTIONS

    def test_decomp_roundtrip_encode_decode(self):
        from snapshot import _encode_snapshot, _decode_snapshot, _SNAPSHOT_CONTROLS
        state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
        state["bub-decomp-model:value"] = "hybppl_ex"
        state["bub-decomp-components:value"] = ["A_sup", "a\u2080", "__sum__"]
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["bub-decomp-model:value"] == "hybppl_ex"
        assert decoded["bub-decomp-components:value"] == [
            "A_sup", "a\u2080", "__sum__"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestDecompSnapshot -v 2>&1 | tail -10
```
Expected: FAIL — `bub-decomp-model not in cids`.

- [ ] **Step 3: Append to _SNAPSHOT_CONTROLS in snapshot.py**

Open `btc_web/snapshot.py`, find the end of `_SNAPSHOT_CONTROLS` list (the last entry). Append these two entries AT THE END of the list:

```python
    # ── Component Decomposition (bubble tab) ──
    ("bub-decomp-model",      "value"),   # family dropdown
    ("bub-decomp-components", "value"),   # dynamic checklist — plain list
```

Append-only rule — do not reorder existing entries (preserves bitmask positions in old share links).

- [ ] **Step 4: Add to _TAB_CONTROLS["bubble"] in callbacks/routing.py**

Open `btc_web/callbacks/routing.py`, find `_TAB_CONTROLS["bubble"]` set. Add `"bub-decomp-model"` and `"bub-decomp-components"` entries at the end of that set definition:

```python
    "bubble":      {"bub-qs","bub-qs-mode","bub-qs-adv",
                    ...  # existing entries unchanged
                    "scan-price","scan-date","scan-q",
                    "bub-decomp-model","bub-decomp-components"},
```

- [ ] **Step 5: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestDecompSnapshot -v 2>&1 | tail -10
```
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add btc_web/snapshot.py btc_web/callbacks/routing.py btc_web/test_web.py
git commit -m "$(cat <<'EOF'
feat(decomp): snapshot + share-link support

Append bub-decomp-model / bub-decomp-components to _SNAPSHOT_CONTROLS
(end of list — preserves bitmask positions of existing entries).
Add both to _TAB_CONTROLS['bubble'] for 'Current tab only' share links.

decomp-components is NOT bitmask-encoded (dynamic options set) —
stored as plain list[str].

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Full test suite + boot smoke

- [ ] **Step 1: Run complete test suite**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -q 2>&1 | tail -5
```
Expected: `886 passed, 5 skipped` (857 baseline + ~29 new tests)

- [ ] **Step 2: DEV server boot check**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 14
grep -iE "traceback|error " /tmp/quantoshi_dev.log | head -10 || echo clean
curl -sS http://localhost:8050/ -o /dev/null -w "HTTP %{http_code}\n"
curl -sS http://localhost:8050/1 -o /dev/null -w "/1 HTTP %{http_code}\n"
```
Expected: `clean`, `HTTP 200`, `/1 HTTP 200`

- [ ] **Step 3: Kill server**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null
```

- [ ] **Step 4: Skip commit**

No file changes; nothing to commit.

---

## Task 15: E2E Playwright verification

- [ ] **Step 1: Start DEV server**

```bash
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 14
curl -sS http://localhost:8050/ -o /dev/null -w "HTTP %{http_code}\n"
```

- [ ] **Step 2: Navigate to /1 via browser_navigate**

Via MCP Playwright:
```
browser_navigate → http://localhost:8050/1
```

- [ ] **Step 3: Verify decomp UI present**

Via MCP Playwright browser_evaluate:
```javascript
() => {
  const section = Array.from(document.querySelectorAll('.ctrl-section-header'))
    .find(h => h.textContent.includes('Component Decomposition'));
  const dropdown = document.querySelector('#bub-decomp-model');
  return {
    section_found: !!section,
    section_text: section?.textContent.trim(),
    has_icon: section?.textContent.includes('🧬'),
    dropdown_found: !!dropdown,
    dropdown_value: dropdown?.querySelector('.Select-value-label')?.textContent
  };
}
```
Expected: `section_found: true`, `has_icon: true`, `dropdown_found: true`.

- [ ] **Step 4: Pick HybPPL (ex) and verify 6 options appear**

Via MCP Playwright browser_evaluate — select dropdown option via the Dash state:
```javascript
() => {
  const sel = document.querySelector('#bub-decomp-model .Select-arrow-zone');
  if (sel) sel.click();
  return true;
}
```
Then click option "HybPPL (ex)". Verify checklist has 6 entries (5 components + Σ Sum).

- [ ] **Step 5: Kill server**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null
```

- [ ] **Step 6: No commit** (E2E only).

---

## Task 16: Deploy to production

- [ ] **Step 1: Push to origin**

```bash
git push origin master 2>&1 | tail -5
```

- [ ] **Step 2: Pull on prod + restart**

```bash
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi" 2>&1 | tail -10
```

- [ ] **Step 3: Verify production health**

```bash
sleep 18 && curl -sS https://quantoshi.xyz/ -o /dev/null -w "HTTP %{http_code}\n"
```
Expected: `HTTP 200`

- [ ] **Step 4: Smoke-test decomp on prod**

Via MCP Playwright:
```
browser_navigate → https://quantoshi.xyz/1
```
Verify Component Decomposition section appears. Pick HybPPL (ex), check 2 components, confirm chart shows 2 dotted traces.

---

## Post-implementation

- [ ] All 886 tests pass.
- [ ] App boots clean in DEV mode.
- [ ] Bubble tab (/1) has Component Decomposition section with 🧬 icon.
- [ ] Selecting a model populates checklist dynamically.
- [ ] Checking components adds dotted traces; Σ adds a solid trace.
- [ ] LPPL family warning banner appears when n_freqs≠1.
- [ ] Share link round-trips decomp state correctly.
- [ ] Deployed to https://quantoshi.xyz.
