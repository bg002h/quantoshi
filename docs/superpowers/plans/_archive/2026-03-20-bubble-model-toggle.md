# Bubble Model Toggle + EF Composite/Support/Future Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **IMPORTANT:** After each task is implemented, dispatch a code-reviewer agent before proceeding to the next task.

**Goal:** Allow toggling the main Bubble Model on/off in the Display Models checklist, and render EF's composite/support/future-bubble traces when EF is enabled.

**Architecture:** Add `"bub"` to `bub-model-show` (checked by default), gate main BM trace drawing on its presence, and extend the overlay loop to draw composite/support/future for any `_CompositeModel`. Add `@property` accessors to `_CompositeModel` so the figure builder can read EF's private data.

**Tech Stack:** Dash 4, Plotly, Python, numpy

**Spec:** `docs/superpowers/specs/2026-03-20-bubble-model-toggle-design.md`

---

### Task 1: Add `@property` accessors to `_CompositeModel`

**Files:**
- Modify: `archive/btc_app/btc_core.py:370-444` (`_CompositeModel` class)
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write failing test**

Add to `btc_web/test_web.py` (near existing model tests):

```python
class TestCompositeModelAccessors:
    """EmpiricalFloorModel exposes composite data via public properties."""

    def test_ef_has_comp_by_n(self):
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert hasattr(ef, "comp_by_n")
        assert isinstance(ef.comp_by_n, list)
        assert len(ef.comp_by_n) > 0

    def test_ef_has_support_plot(self):
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert hasattr(ef, "support_plot")
        assert len(ef.support_plot) > 0

    def test_ef_has_t_grid(self):
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert hasattr(ef, "t_grid")
        assert len(ef.t_grid) == len(ef.support_plot)

    def test_ef_has_bm_r2(self):
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert isinstance(ef.bm_r2, float)
        assert 0 < ef.bm_r2 < 1

    def test_ef_has_n_future_max(self):
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert isinstance(ef.n_future_max, int)
        assert ef.n_future_max >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCompositeModelAccessors -v`
Expected: FAIL — `AttributeError` because `comp_by_n` etc. are private (`_comp_by_n`).

- [ ] **Step 3: Add `@property` accessors to `_CompositeModel`**

In `archive/btc_app/btc_core.py`, add these properties inside `_CompositeModel` (after `_init_bands` method, before the class ends):

```python
    @property
    def comp_by_n(self):
        return self._comp_by_n

    @property
    def support_plot(self):
        return self._support_plot

    @property
    def t_grid(self):
        return self._t_grid

    @property
    def bm_r2(self):
        return self._bm_r2

    @property
    def n_future_max(self):
        return self._n_future_max
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCompositeModelAccessors -v`
Expected: PASS (or SKIP if EF pkl not present locally)

- [ ] **Step 5: Commit**

```bash
git add archive/btc_app/btc_core.py btc_web/test_web.py
git commit -m "feat: add public property accessors to _CompositeModel"
```

- [ ] **Step 6: Code review**

Dispatch `superpowers:code-reviewer` agent to review the changes against the spec.

---

### Task 2: Add `"bub"` to `bub-model-show` checklist

**Files:**
- Modify: `btc_web/layout/bubble.py:66-74`
- Modify: `btc_web/snapshot.py:171` (`_CHECKLIST_OPTIONS`)
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write failing test**

```python
class TestBubbleModelToggle:
    """bub-model-show includes 'bub' checked by default."""

    def test_bub_in_model_show_options(self):
        """The bubble model appears in Display Models checklist."""
        from layout.bubble import _bubble_controls
        controls = _bubble_controls()
        # Find the bub-model-show checklist
        def find_checklist(component):
            if hasattr(component, 'id') and component.id == 'bub-model-show':
                return component
            if hasattr(component, 'children'):
                kids = component.children
                if isinstance(kids, list):
                    for c in kids:
                        r = find_checklist(c)
                        if r: return r
                elif kids:
                    return find_checklist(kids)
            return None
        cl = find_checklist(controls)
        assert cl is not None
        option_values = [o["value"] for o in cl.options]
        assert "bub" in option_values
        assert option_values[0] == "bub"  # first in list
        assert "bub" in cl.value  # checked by default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestBubbleModelToggle -v`
Expected: FAIL — `"bub"` not in options (currently filtered out).

- [ ] **Step 3: Update layout**

In `btc_web/layout/bubble.py`, change lines 66-74 from:

```python
_lbl("Overlay models"),
dcc.Checklist(id="bub-model-show",
              options=[{"label": f" {mdl.name}", "value": mdl.short_name}
                       for mdl in _app_ctx.PRICE_MODELS.values()
                       if mdl.short_name != "bub"],
              value=[], inline=True,
```

To:

```python
_lbl("Display models"),
dcc.Checklist(id="bub-model-show",
              options=[{"label": f" {mdl.name}", "value": mdl.short_name}
                       for mdl in _app_ctx.PRICE_MODELS.values()
                       if mdl.short_name not in _app_ctx.MODEL_SENTINELS],
              value=["bub"], inline=True,
```

This removes the `!= "bub"` filter so "Bubble Model" appears, and defaults it to checked. The `MODEL_SENTINELS` filter keeps `"qr"` and `"mc"` out (they're not standalone display models).

- [ ] **Step 4: Update `_CHECKLIST_OPTIONS` in snapshot.py**

In `btc_web/snapshot.py` line 171, change:

```python
"bub-model-show":     ["pl", "lppl", "exp", "s2f", "ef"],
```

To:

```python
"bub-model-show":     ["pl", "lppl", "exp", "s2f", "ef", "bub"],
```

**Important:** `"bub"` goes at the END to avoid shifting existing bitmask positions. Old links that encoded PL/S2F/EF will decode correctly. The backward-compat injection (Step 5) adds `"bub"` when absent.

- [ ] **Step 5: Add backward compatibility for old snapshot links**

In `btc_web/snapshot.py`, find the `_decode_snapshot` function. After the checklist bitmask decoding, add logic: if the decoded `bub-model-show` value is a list and `"bub"` is not in it, inject it. This ensures old links (which never encoded `"bub"`) still show the bubble model.

In `_decode_snapshot()`, after the `for (cid, prop), val in zip(...)` loop builds the `state` dict (around line 245), before `return state`, add:

```python
        # Backward compat: old links didn't have "bub" in bub-model-show
        _bms_key = "bub-model-show:value"
        if _bms_key in state and isinstance(state[_bms_key], list):
            if "bub" not in state[_bms_key]:
                state[_bms_key] = ["bub"] + state[_bms_key]
```

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestBubbleModelToggle -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add btc_web/layout/bubble.py btc_web/snapshot.py btc_web/test_web.py
git commit -m "feat: add 'bub' to Display Models checklist, default checked"
```

- [ ] **Step 8: Code review**

Dispatch `superpowers:code-reviewer` agent.

---

### Task 3: Gate main BM traces on `"bub"` in `active_models`

**Files:**
- Modify: `btc_web/figures/bubble.py:52-203`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write failing test**

```python
class TestBubbleModelGating:
    """Main BM traces are conditional on 'bub' in active_models."""

    _BASE = dict(
        selected_qs=[0.5] if 0.5 in _app_ctx.DEFAULT_MODEL.fits else [0.10],
        shade=True, show_ols=False, show_ucl=False,
        show_data=False, show_today=False,
        show_legend=False, minor_grid=False,
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

    def test_bub_active_draws_traces(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub"]))
        names = [t.name for t in fig.data if t.name]
        assert any("Bubble composite" in n for n in names)
        assert any("Bubble support" in n for n in names)

    def test_bub_inactive_hides_traces(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=[]))
        # No traces should have no legendgroup (BM traces lack legendgroup;
        # overlay traces always set legendgroup=mdl.short_name)
        bm_traces = [t for t in fig.data if t.name
                     and not getattr(t, "legendgroup", None)
                     and t.name not in ("Price data", "Lots")]
        assert len(bm_traces) == 0, f"BM traces should be hidden, found: {[t.name for t in bm_traces]}"

    def test_bub_inactive_preserves_data_scatter(self):
        """Data scatter, OLS, UCL, today line survive when BM is off."""
        fig = build_bubble_figure(M, dict(self._BASE,
            active_models=[], show_data=True, show_today=True,
            show_ols=True, show_ucl=True))
        names = [t.name for t in fig.data if t.name]
        assert any("Price data" in n for n in names)

    def test_bub_inactive_still_has_axis_config(self):
        """Even with BM hidden, chart should render without error."""
        fig = build_bubble_figure(M, dict(self._BASE, active_models=[]))
        assert isinstance(fig, go.Figure)
        assert fig.layout.xaxis.type in ("log", "linear", "-")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestBubbleModelGating -v`
Expected: `test_bub_inactive_hides_traces` FAILS — BM traces drawn regardless of `active_models`.

- [ ] **Step 3: Wrap main BM trace drawing in conditional**

In `btc_web/figures/bubble.py`, wrap lines 52-95 (price cache, shading, quantile lines) and 180-203 (support, composite) in `if "bub" in p.get("active_models", []):`.

The key change: move the existing quantile/shade/composite/support blocks inside a conditional:

```python
    # ── Main Bubble Model traces (conditional on "bub" in active_models) ──
    bub_active = "bub" in p.get("active_models", ["bub"])

    sel_qs = sorted([float(q) for q in (p.get("selected_qs") or [])])
    _thermal = _build_thermal_colors(sel_qs, palette)

    if bub_active:
        # Pre-compute prices for all selected quantiles
        _price_cache = {}
        for q in sel_qs:
            if q in model.fits:
                _price_cache[q] = _round_trace_data(model.price_at(q, t_arr) * (stack if stack > 0 else 1))

        if p.get("shade") and len(sel_qs) >= 2:
            # ... existing shading code (unchanged) ...

        # ── quantile lines ────────────────────────────────────
        for q in sel_qs:
            # ... existing quantile line code (unchanged) ...
```

And similarly wrap the support (lines 180-189) and composite (lines 191-203) blocks:

```python
    if bub_active:
        # ── bubble support ────────────────────────────────
        if p.get("show_sup"):
            # ... existing support code (unchanged) ...

        # ── bubble composite ──────────────────────────────
        if p.get("show_comp"):
            # ... existing composite code (unchanged) ...
```

**Important:** `sel_qs` and `_thermal` must remain outside the conditional — they're needed by the overlay loop for non-BM models.

**Default `active_models`:** Use `p.get("active_models", ["bub"])` (not `[]`) so callers that omit the param (including existing tests) still get BM traces by default. The callback always passes `active_models` explicitly, so this default only affects direct `build_bubble_figure` callers.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestBubbleModelGating btc_web/test_web.py::TestBuildBubbleFigure -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add btc_web/figures/bubble.py btc_web/test_web.py
git commit -m "feat: gate main BM traces on 'bub' in active_models"
```

- [ ] **Step 6: Code review**

Dispatch `superpowers:code-reviewer` agent.

---

### Task 4: Render EF composite/support/future in overlay loop

**Files:**
- Modify: `btc_web/figures/bubble.py:97-133` (overlay loop)
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write failing test**

```python
class TestEFCompositeOverlay:
    """EF overlay renders composite/support/future when enabled."""

    _BASE = dict(
        selected_qs=[0.5] if 0.5 in _app_ctx.DEFAULT_MODEL.fits else [0.10],
        shade=False, show_ols=False, show_ucl=False,
        show_data=False, show_today=False,
        show_legend=False, minor_grid=False,
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

    def test_ef_overlay_draws_composite(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["ef"]))
        names = [t.name for t in fig.data if t.name]
        assert any("Empirical Floor" in n and "composite" in n for n in names)

    def test_ef_overlay_draws_support(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["ef"]))
        names = [t.name for t in fig.data if t.name]
        assert any("Empirical Floor" in n and "support" in n for n in names)

    def test_ef_composite_uses_own_color(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["ef"]))
        comp_traces = [t for t in fig.data if t.name and "Empirical Floor" in t.name and "composite" in t.name]
        assert len(comp_traces) > 0
        assert comp_traces[0].line.color == "#D4A017"  # EF amber

    def test_ef_no_composite_when_show_comp_off(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["ef"], show_comp=False))
        names = [t.name for t in fig.data if t.name]
        assert not any("composite" in n for n in names)

    def test_both_bub_and_ef_composite(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub", "ef"]))
        names = [t.name for t in fig.data if t.name]
        assert any("Bubble composite" in n for n in names)
        assert any("Empirical Floor" in n and "composite" in n for n in names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestEFCompositeOverlay -v`
Expected: FAIL — no EF composite/support traces in overlay.

- [ ] **Step 3: Extend overlay loop with composite/support/future rendering**

In `btc_web/figures/bubble.py`, inside the `for model_key in p.get("active_models", []):` loop, after the existing quantized/non-quantized branches, add a new block for composite models:

```python
        # ── composite/support/future for _CompositeModel overlays ─────
        if hasattr(mdl, "comp_by_n") and hasattr(mdl, "t_grid"):
            _EF_COMP_COLOR = "#D4A017"
            _EF_SUP_COLOR  = "#8B6914"
            mdl_t = np.asarray(mdl.t_grid)
            mdl_mask = (mdl_t >= t_lo) & (mdl_t <= t_hi)

            if p.get("show_sup") and hasattr(mdl, "support_plot"):
                sup_y = np.asarray(mdl.support_plot)[mdl_mask] * (stack if stack > 0 else 1)
                traces.append(go.Scatter(
                    x=list(mdl_t[mdl_mask]), y=list(sup_y),
                    mode="lines", name=f"{mdl.name} support",
                    line=dict(color=_EF_SUP_COLOR, dash="dash", width=1.5),
                    opacity=0.9,
                    legendgroup=mdl.short_name,
                ))

            if p.get("show_comp"):
                n = int(p.get("n_future", 0))
                n = min(n, len(mdl.comp_by_n) - 1)
                comp_y = np.asarray(mdl.comp_by_n[n])[mdl_mask] * (stack if stack > 0 else 1)
                traces.append(go.Scatter(
                    x=list(mdl_t[mdl_mask]), y=list(comp_y),
                    mode="lines",
                    name=f"{mdl.name} composite (N={n})  R\u00b2={mdl.bm_r2:.4f}",
                    line=dict(color=_EF_COMP_COLOR, width=2.0),
                    legendgroup=mdl.short_name,
                ))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestEFCompositeOverlay btc_web/test_web.py::TestMultiModelBubbleFigure -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add btc_web/figures/bubble.py btc_web/test_web.py
git commit -m "feat: render EF composite/support/future in bubble overlay"
```

- [ ] **Step 6: Code review**

Dispatch `superpowers:code-reviewer` agent.

---

### Task 5: Fix auto-Y-range and prewarm cache

**Files:**
- Modify: `btc_web/callbacks/charts.py:98-132` (`auto_bubble_yrange`)
- Modify: `btc_web/app.py:187-199` (`_prewarm_caches`)
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write failing test**

```python
class TestAutoYWithBubToggle:
    """Auto-Y-range respects bub toggle."""

    def test_auto_y_no_bub_uses_fallback(self):
        """When BM is unchecked, auto-Y should not crash."""
        from callbacks.charts import auto_bubble_yrange
        # model_show=[] means BM is off, no overlays
        try:
            result = auto_bubble_yrange([2012, 2030], ["yes"], "log", [], [0.5])
        except Exception:
            pytest.fail("auto_bubble_yrange should not crash when bub is off")
        assert isinstance(result, list)
        assert len(result) == 2
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestAutoYWithBubToggle -v`
Expected: May pass (current code always uses DEFAULT_MODEL) — but verify it runs correctly.

- [ ] **Step 3: Update auto-Y-range callback**

In `btc_web/callbacks/charts.py`, modify `auto_bubble_yrange` to check if `"bub"` is in `model_show`:

```python
    # Base Y range from BM (if active) or first active quantized model
    if "bub" in (model_show or []):
        qs = sorted([float(q) for q in (sel_qs or []) if float(q) in _app_ctx.DEFAULT_MODEL.fits])
        if not qs:
            qs = sorted(_app_ctx.DEFAULT_MODEL.fits.keys())
        p_lo = float(_app_ctx.DEFAULT_MODEL.price_at(qs[0], t_lo))
        p_hi = float(_app_ctx.DEFAULT_MODEL.price_at(qs[-1], t_hi))
    else:
        # Fallback: find first active quantized model, or use DEFAULT_MODEL
        p_lo, p_hi = None, None
        for key in (model_show or []):
            mdl = _app_ctx.PRICE_MODELS.get(key)
            if mdl and mdl.quantized:
                mdl_qs = sorted([float(q) for q in (sel_qs or []) if float(q) in mdl.fits])
                if not mdl_qs:
                    mdl_qs = sorted(mdl.fits.keys())
                p_lo = float(mdl.price_at(mdl_qs[0], t_lo))
                p_hi = float(mdl.price_at(mdl_qs[-1], t_hi))
                break
        if p_lo is None:
            # No active model — use DEFAULT_MODEL as safe fallback
            qs = sorted(_app_ctx.DEFAULT_MODEL.fits.keys())
            p_lo = float(_app_ctx.DEFAULT_MODEL.price_at(qs[0], t_lo))
            p_hi = float(_app_ctx.DEFAULT_MODEL.price_at(qs[-1], t_hi))
```

- [ ] **Step 4: Update `_prewarm_caches`**

In `btc_web/app.py`, add `active_models=["bub"]` to the bubble prewarm call:

```python
    _get_bubble_fig(dict(
        selected_qs = [],
        shade=True, show_ols=False, show_data=True, show_today=True,
        show_legend=False, minor_grid=False,
        show_comp=True, show_sup=False,
        xscale="log", yscale="log",
        xmin=2012, xmax=yr_now + 4,
        ymin=0.01, ymax=1e7,
        n_future=3, pt_size=3, pt_alpha=0.3,
        stack=0, show_stack=False, use_lots=False, lots=[],
        comp_color="#FFD700", comp_lw=2.0,
        sup_color="#888888", sup_lw=1.5,
        active_models=["bub"],
        palette="default",
    ))
```

- [ ] **Step 5: Run full test suite**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --timeout=120`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/app.py btc_web/test_web.py
git commit -m "fix: auto-Y fallback when BM unchecked, add active_models to prewarm"
```

- [ ] **Step 7: Code review**

Dispatch `superpowers:code-reviewer` agent.

---

### Task 6: Manual verification and final commit

**Files:** None (testing only)

- [ ] **Step 1: Start dev server**

```bash
DEV=1 bash run_web.sh
```

- [ ] **Step 2: Verify bubble model toggle**

1. Load tab 1 — BM quantile lines, composite, support should render (default)
2. Uncheck "Bubble Model" in Display Models — all BM traces disappear
3. Data scatter, OLS, UCL, today line remain visible
4. Re-check "Bubble Model" — traces reappear

- [ ] **Step 3: Verify EF composite/support**

1. Check "BM Empirical Floor" in Display Models
2. With show_comp on → EF composite appears in amber (#D4A017)
3. With show_sup on → EF support appears in dark amber (#8B6914)
4. Adjust N future bubbles slider → EF composite changes shape
5. Both BM and EF active → two composites, two supports visible with distinct colors

- [ ] **Step 4: Verify snapshot round-trip**

1. Enable EF, disable BM
2. Click Share → generate link
3. Open link in new tab → BM should be off, EF should be on with composite/support

- [ ] **Step 5: Final code review**

Dispatch `superpowers:requesting-code-review` to review ALL changes against the spec.
