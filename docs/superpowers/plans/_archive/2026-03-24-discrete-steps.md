# Discrete Steps Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Discrete steps" checkbox to tabs 3-5 that switches simulation traces from smooth linear interpolation to step-function rendering (`line_shape="hv"`).

**Architecture:** Single toggle value added to existing `_chart_toggles()` checklist. Flag flows through callbacks → params dict → figure builders. Each `go.Scatter(mode="lines")` trace gets `shape=_line_shape` in its `line` dict. Snapshot bitmask updated for share link fidelity.

**Tech Stack:** Python (Plotly/Dash), `line_shape` Plotly property

**Spec:** `docs/superpowers/specs/2026-03-24-discrete-steps-design.md`

---

### Task 1: Add toggle option + snapshot bitmask

**Files:**
- Modify: `btc_web/layout/common.py:115`
- Modify: `btc_web/snapshot.py:156,161,164`

- [ ] **Step 1: Add "Discrete steps" to `_chart_toggles()` options**

In `btc_web/layout/common.py` line 115, add after the `"annotate"` entry:

```python
    opts = [{"label": " Log Y", "value": "log_y"},
            {"label": " Annotate final values", "value": "annotate"},
            {"label": " Discrete steps", "value": "discrete"},
            {"label": " Show legend", "value": "show_legend"},
            {"label": html.Span(" Minor grid", className="minor-grid-opt"),
             "value": "minor_grid"},
            {"label": " Enable chart zoom", "value": "chart_zoom"}]
```

- [ ] **Step 2: Update `_CHECKLIST_OPTIONS` in snapshot.py**

Append `"discrete"` at end of each toggles list (preserves existing bit positions):

```python
"dca-toggles":        ["log_y", "annotate", "show_legend", "minor_grid", "chart_zoom", "discrete"],
"ret-toggles":        ["log_y", "annotate", "show_legend", "minor_grid", "chart_zoom", "discrete"],
"sc-toggles":         ["annotate", "log_y", "show_legend", "minor_grid", "chart_zoom", "discrete"],
```

- [ ] **Step 3: Syntax check and commit**

```bash
btc_venv/bin/python3 -m py_compile btc_web/layout/common.py && btc_venv/bin/python3 -m py_compile btc_web/snapshot.py && echo "OK"
git add btc_web/layout/common.py btc_web/snapshot.py
git commit -m "feat: add discrete steps toggle to chart settings + snapshot bitmask"
```

---

### Task 2: Wire flag through callbacks

**Files:**
- Modify: `btc_web/callbacks/charts.py` (3 locations — DCA ~line 426, Retire ~line 525, SC ~line 641)

- [ ] **Step 1: Add `discrete` flag to DCA callback params dict**

After `annotate = "annotate" in toggles` (~line 426), the params dict already unpacks toggles. Add:

```python
        discrete     = "discrete"   in toggles,
```

Place it after the `minor_grid` line in the DCA params dict.

- [ ] **Step 2: Add `discrete` flag to Retire callback params dict**

Same pattern — after the `minor_grid` line in the Retire params dict (~line 528):

```python
        discrete     = "discrete"   in toggles,
```

- [ ] **Step 3: Add `discrete` flag to Supercharger callback params dict**

Same pattern — after the `minor_grid` line in the SC params dict (~line 644):

```python
        discrete     = "discrete"   in toggles,
```

- [ ] **Step 4: Syntax check and commit**

```bash
btc_venv/bin/python3 -m py_compile btc_web/callbacks/charts.py && echo "OK"
git add btc_web/callbacks/charts.py
git commit -m "feat: wire discrete flag through DCA/Retire/SC callbacks"
```

---

### Task 3: Apply `line_shape` in DCA figure builder

**Files:**
- Modify: `btc_web/figures/dca.py` (lines 143-145, 198-200, 222-225, 240-243)

- [ ] **Step 1: Compute `_line_shape` at top of `build_dca_figure`**

Near the top of the function, after reading other flags from `p`:

```python
_line_shape = "hv" if p.get("discrete") else "linear"
```

- [ ] **Step 2: Apply to primary model traces**

Line 200 — add `shape=_line_shape`:

```python
line=dict(color=col, width=_QR_LINE_WIDTH, shape=_line_shape),
```

- [ ] **Step 3: Apply to overlay model traces (quantized)**

Line 225:

```python
line=dict(color=col, width=_OVERLAY_LINE_WIDTH, dash=mdl.dash_style, shape=_line_shape),
```

- [ ] **Step 4: Apply to overlay model traces (non-quantized)**

Line 243:

```python
line=dict(color=palette["non_quantized_model"], width=_OVERLAY_LINE_WIDTH, dash=mdl.dash_style, shape=_line_shape),
```

- [ ] **Step 5: Pass `_line_shape` to `_dca_sc_overlay` and apply there**

Update the function signature at line 29:

```python
def _dca_sc_overlay(m, p, ts, sel_qs, start_stack, all_prices, disp_mode, ppy, thermal=None, line_shape="linear"):
```

At line 145, apply it:

```python
line=dict(color=col, width=_QR_LINE_WIDTH, dash="dash", shape=line_shape),
```

At the call site (search for `_dca_sc_overlay(` in `build_dca_figure`), pass `line_shape=_line_shape`.

- [ ] **Step 6: Syntax check and commit**

```bash
btc_venv/bin/python3 -m py_compile btc_web/figures/dca.py && echo "OK"
git add btc_web/figures/dca.py
git commit -m "feat: apply discrete line_shape to DCA traces + SC overlay"
```

---

### Task 4: Apply `line_shape` in Retire figure builder

**Files:**
- Modify: `btc_web/figures/retire.py` (lines 77-79, 117-120, 135-138)

- [ ] **Step 1: Compute `_line_shape` at top of `build_retire_figure`**

```python
_line_shape = "hv" if p.get("discrete") else "linear"
```

- [ ] **Step 2: Apply to primary model traces**

Line 79:

```python
line=dict(color=col, width=_QR_LINE_WIDTH, shape=_line_shape),
```

- [ ] **Step 3: Apply to overlay model traces (quantized + non-quantized)**

Lines 120 and 138 — same pattern as DCA, add `shape=_line_shape` to each `line` dict.

- [ ] **Step 4: Syntax check and commit**

```bash
btc_venv/bin/python3 -m py_compile btc_web/figures/retire.py && echo "OK"
git add btc_web/figures/retire.py
git commit -m "feat: apply discrete line_shape to Retire traces"
```

---

### Task 5: Apply `line_shape` in Supercharger figure builder (Mode A only)

**Files:**
- Modify: `btc_web/figures/supercharge.py` (multiple `go.Scatter` calls in Mode A branch)

- [ ] **Step 1: Compute `_line_shape` inside Mode A branch only**

After `if mode == "a":` and the variable setup, add:

```python
_line_shape = "hv" if p.get("discrete") else "linear"
```

This must be inside the Mode A block. Mode B traces use `x=delays` (not time-series) and must NOT get `shape="hv"`.

- [ ] **Step 2: Apply to all primary model `mode="lines"` traces**

Add `shape=_line_shape` to every `line=dict(...)` in the primary model traces:
- Layout 0 traces (~line 177)
- Layout 1 traces (~line 201)
- Layout 2 shade band boundary traces (~lines 233, 238)
- Layout 2 individual quantile traces (~line 253)

- [ ] **Step 3: Apply to overlay model traces**

Add `shape=_line_shape` to:
- Overlay shade band boundary traces (~lines 319, 324)
- Overlay quantile traces (~line 337)
- Overlay individual line traces (~line 360)

- [ ] **Step 4: Syntax check and commit**

```bash
btc_venv/bin/python3 -m py_compile btc_web/figures/supercharge.py && echo "OK"
git add btc_web/figures/supercharge.py
git commit -m "feat: apply discrete line_shape to Supercharger traces (Mode A only)"
```

---

### Task 6: Apply `line_shape` to MC overlay traces

**Files:**
- Modify: `btc_web/mc_overlay.py` (~line 441, `_mc_build_traces`)

- [ ] **Step 1: Add `line_shape` parameter to `_mc_build_traces`**

```python
def _mc_build_traces(mc_ts, fan, extra_label="", show_median=True,
                     show_final_values=False, fan_usd=None,
                     hide_5_95_legend=False, bands=None,
                     suppress_legend=False, line_shape="linear"):
```

- [ ] **Step 2: Apply to fan boundary and median traces**

Line 468:
```python
mode="lines", line=dict(width=0, shape=line_shape), showlegend=False, hoverinfo="skip",
```

Line 472:
```python
mode="lines", line=dict(width=0, shape=line_shape), fill="tonexty",
```

Line 487:
```python
line=dict(color=med_color, width=med_width, dash=med_dash, shape=line_shape),
```

- [ ] **Step 3: Pass `line_shape` from each tab's MC overlay caller**

Search for calls to `_mc_build_traces` in `mc_overlay.py`. Each tab's `_mc_*_overlay` function calls it. Add `line_shape=p.get("discrete") and "hv" or "linear"` (or compute once from `p` and pass through).

The simplest approach: in each `_mc_*_overlay` function, compute `_ls = "hv" if p.get("discrete") else "linear"` and pass `line_shape=_ls` to `_mc_build_traces`.

- [ ] **Step 4: Syntax check and commit**

```bash
btc_venv/bin/python3 -m py_compile btc_web/mc_overlay.py && echo "OK"
git add btc_web/mc_overlay.py
git commit -m "feat: apply discrete line_shape to MC overlay traces"
```

---

### Task 7: Run tests + deploy

- [ ] **Step 1: Syntax check all modified files**

```bash
for f in btc_web/layout/common.py btc_web/snapshot.py btc_web/callbacks/charts.py btc_web/figures/dca.py btc_web/figures/retire.py btc_web/figures/supercharge.py btc_web/mc_overlay.py; do
    btc_venv/bin/python3 -m py_compile $f || echo "FAIL: $f"
done && echo "ALL OK"
```

- [ ] **Step 2: Run test suite**

```bash
PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --timeout=120
```

Verify no new failures beyond pre-existing ones (19 known failures).

- [ ] **Step 3: Deploy**

```bash
git push origin master
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && systemctl restart quantoshi"
```

---

## Summary

| Task | What | Risk |
|------|------|------|
| 1 | Toggle option + snapshot bitmask | None — additive |
| 2 | Wire flag through 3 callbacks | None — adding a key to existing dicts |
| 3 | DCA figure builder + SC overlay helper | Low — adding `shape` to `line` dicts |
| 4 | Retire figure builder | Low — same pattern as DCA |
| 5 | Supercharger figure builder (Mode A) | Low — must guard Mode B |
| 6 | MC overlay traces | Medium — `_mc_build_traces` has many callers |
| 7 | Tests + deploy | None — verification |
