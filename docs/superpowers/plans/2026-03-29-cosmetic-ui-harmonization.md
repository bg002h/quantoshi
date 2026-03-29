# Cosmetic UI Harmonization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standardize labels, checkbox margins, and quantile hints across all 9 web app tabs for visual consistency.

**Architecture:** Add shared constants (`_CB_MARGIN`, `_INFL_LABEL`, `_Q_HINT_BASE`) to `layout/common.py`, then sweep all layout modules to reference them. Pure cosmetic — no callback, structural, or behavioral changes.

**Tech Stack:** Plotly Dash 4.0.0, dash-bootstrap-components 2.0.4, Python 3.14

**Spec:** `docs/superpowers/specs/2026-03-29-cosmetic-ui-harmonization-design.md`

---

### Task 1: Baseline — verify all tests pass

**Files:** None modified

- [ ] **Step 1: Run the full test suite**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -20
```

Expected: All ~650+ tests PASS. If any fail, stop and investigate before proceeding.

---

### Task 2: Add constants to `layout/common.py` and update its own margins

**Files:**
- Modify: `btc_web/layout/common.py`

- [ ] **Step 1: Add the three new constants after line 16** (after `_STYLE_ADDR_CODE`)

```python
_CB_MARGIN  = {"marginRight": "4px"}
_INFL_LABEL = "Inflation rate (0\u2013100% / yr)"
_Q_HINT_BASE = "Lower quantiles = more conservative price paths."
```

- [ ] **Step 2: Update `_q_panel()` (~line 40)**

Replace:
```python
inputStyle={"marginRight":"4px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 3: Update `_chart_toggles()` (~line 124)**

Replace:
```python
inputStyle={"marginRight": "5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 4: Update `_model_show_checklist()` (~line 298)**

Replace:
```python
inputStyle={"marginRight": "4px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 5: Update `_shared_settings_card()` first occurrence (~line 314)**

Replace:
```python
inputStyle={"marginRight": "5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 6: Update `_shared_settings_card()` second occurrence (~line 338)**

Replace:
```python
inputStyle={"marginRight": "4px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 7: Run tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add btc_web/layout/common.py
git commit -m "refactor(layout): add _CB_MARGIN, _INFL_LABEL, _Q_HINT_BASE constants to common.py"
```

---

### Task 3: Update `layout/sim_tabs.py` — DCA + Retire labels, margins, hints

**Files:**
- Modify: `btc_web/layout/sim_tabs.py`

- [ ] **Step 1: Add `_CB_MARGIN`, `_Q_HINT_BASE` to imports**

In the import block (lines 10-16), add `_CB_MARGIN` and `_Q_HINT_BASE` to the existing `from layout.common import (...)` statement.

- [ ] **Step 2: Update DCA amount label (~line 117)**

Replace:
```python
"Per-period amount ($)"
```
With:
```python
"Purchase amount ($)"
```

- [ ] **Step 3: Update DCA MC amount label (~line 120)**

Replace:
```python
"DCA amount per period ($)"
```
With:
```python
"Purchase amount per period ($)"
```

- [ ] **Step 4: Update Retire withdrawal label (~line 140)**

Replace:
```python
"Withdrawal/period ($)"
```
With:
```python
"Withdrawal amount ($)"
```

- [ ] **Step 5: Update Retire MC withdrawal label (~line 143)**

Replace:
```python
"Withdrawal per period ($)"
```
With:
```python
"Withdrawal amount per period ($)"
```

- [ ] **Step 6: Update margin at ~line 33**

Replace:
```python
inputStyle={"marginRight":"4px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 7: Update margin at ~line 58**

Replace:
```python
inputStyle={"marginRight":"5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 8: Update margin at ~line 87**

Replace:
```python
inputStyle={"marginRight":"5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 9: Update DCA quantile hint (~line 115)**

Replace:
```python
"Price path drives sat accumulation — lower quantile = lower price = more sats/period."
```
With:
```python
_Q_HINT_BASE + " Lower prices mean more sats per period."
```

- [ ] **Step 10: Update Retire quantile hint (~line 138)**

Replace:
```python
"Lower quantile = lower price = faster depletion — worst-case planning."
```
With:
```python
_Q_HINT_BASE + " Lower prices mean faster depletion."
```

- [ ] **Step 11: Run tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 12: Commit**

```bash
git add btc_web/layout/sim_tabs.py
git commit -m "style(layout): harmonize DCA+Retire labels, margins, hints"
```

---

### Task 4: Update `layout/supercharge.py` — labels, margins, hints

**Files:**
- Modify: `btc_web/layout/supercharge.py`

- [ ] **Step 1: Add `_CB_MARGIN`, `_Q_HINT_BASE` to imports**

Add to the existing `from layout.common import (...)` block (lines 11-15).

- [ ] **Step 2: Update withdrawal label (~line 73)**

Replace:
```python
"Withdrawal/period ($)"
```
With:
```python
"Withdrawal amount ($)"
```

- [ ] **Step 3: Update MC withdrawal label (~line 91)**

Replace:
```python
"Withdrawal per period ($)"
```
With:
```python
"Withdrawal amount per period ($)"
```

- [ ] **Step 4: Update margin at ~line 35**

Replace:
```python
inputStyle={"marginRight":"4px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 5: Update margin at ~line 43**

Replace:
```python
inputStyle={"marginRight":"5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 6: Update margin at ~line 102**

Replace:
```python
inputStyle={"marginRight":"5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 7: Update quantile hint (~lines 27-29)**

Replace the existing hint text (two lines: "Select quantiles to follow." + "Lower quantile = earlier depletion..."):
```python
"Select quantiles to follow."
```
With:
```python
_Q_HINT_BASE
```

And replace:
```python
"Lower quantile = earlier depletion — use multiple quantiles to see the range."
```
With:
```python
"Lower prices mean earlier depletion."
```

- [ ] **Step 8: Run tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 9: Commit**

```bash
git add btc_web/layout/supercharge.py
git commit -m "style(layout): harmonize Supercharge labels, margins, hints"
```

---

### Task 5: Update `layout/citadel.py` — labels, margins

**Files:**
- Modify: `btc_web/layout/citadel.py`

- [ ] **Step 1: Add `_CB_MARGIN`, `_INFL_LABEL` to imports**

Add to the existing `from layout.common import (...)` block (lines 9-13).

- [ ] **Step 2: Update spending label (~line 96)**

Replace:
```python
"Spending amount ($ / month)"
```
With:
```python
"Monthly spending ($)"
```

- [ ] **Step 3: Update inflation label (~line 98)**

Replace:
```python
"Inflation rate (% / yr)"
```
With:
```python
_INFL_LABEL
```

- [ ] **Step 4: Update MC spending label (~line 276)**

Replace:
```python
"Spending per period ($)"
```
With:
```python
"Monthly spending ($)"
```

- [ ] **Step 5: Update margin at ~line 25** (5px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight": "5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 6: Update margin at ~line 115** (4px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight": "4px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 7: Update margin at ~line 125** (4px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight": "4px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 8: Update margin at ~line 189** (5px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight": "5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 9: Update margin at ~line 200** (4px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight": "4px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 10: Run tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 11: Commit**

```bash
git add btc_web/layout/citadel.py
git commit -m "style(layout): harmonize Citadel labels and margins"
```

---

### Task 6: Update `layout/bubble.py` — margins

**Files:**
- Modify: `btc_web/layout/bubble.py`

- [ ] **Step 1: Add `_CB_MARGIN` to imports**

Add to the existing `from layout.common import (...)` block.

- [ ] **Step 2: Update margin at ~line 41** (3px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight":"3px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 3: Update margin at ~line 65** (5px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight":"5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 4: Update margin at ~line 75** (4px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight": "4px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 5: Update margin at ~line 86** (5px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight":"5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 6: Update margin at ~line 139** (4px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight":"4px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 7: Update margin at ~line 143** (5px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight":"5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 8: Run tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 9: Commit**

```bash
git add btc_web/layout/bubble.py
git commit -m "style(layout): harmonize Bubble tab margins"
```

---

### Task 7: Update `layout/heatmap.py` — margins, hints

**Files:**
- Modify: `btc_web/layout/heatmap.py`

- [ ] **Step 1: Add `_CB_MARGIN`, `_Q_HINT_BASE` to imports**

Add to the existing `from layout.common import (...)` block.

- [ ] **Step 2: Update margin at ~line 29** (4px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight":"4px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 3: Update margin at ~line 45** (5px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight": "5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 4: Update margin at ~line 63** (5px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight":"5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 5: Update margin at ~line 116** (5px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight":"5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 6: Update quantile hint at ~line 23**

Replace:
```python
"Select quantiles to follow."
```
With:
```python
_Q_HINT_BASE
```

- [ ] **Step 7: Update exit quantile hint at ~line 25**

Replace:
```python
"Select exit quantiles for CAGR projection columns."
```
With:
```python
"Select quantiles for CAGR projection columns."
```

- [ ] **Step 8: Run tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 9: Commit**

```bash
git add btc_web/layout/heatmap.py
git commit -m "style(layout): harmonize Heatmap margins and hints"
```

---

### Task 8: Update `layout/mc_controls.py` — margins

**Files:**
- Modify: `btc_web/layout/mc_controls.py`

- [ ] **Step 1: Add `_CB_MARGIN` to imports**

Add to the existing imports from `layout.common` (or add a new import line if none exists).

- [ ] **Step 2: Update margin at ~line 133** (5px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight": "5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 3: Update margin at ~line 137** (5px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight": "5px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 4: Update margin at ~line 196** (4px → `_CB_MARGIN`)

Replace:
```python
inputStyle={"marginRight": "4px"}
```
With:
```python
inputStyle=_CB_MARGIN
```

- [ ] **Step 5: Run tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add btc_web/layout/mc_controls.py
git commit -m "style(layout): harmonize MC controls margins"
```

---

### Task 9: Final verification

**Files:** None modified

- [ ] **Step 1: Syntax check the web app**

```bash
cd btc_web && PYTHONPATH=".:../archive/btc_app" ../btc_venv/bin/python3 -c \
  "import layout, figures, callbacks, cache, engines.adapter, engines.citadel, engines.tax, engines.tax_lots, engines.tax_data, data.asset_matrices; print('OK')"
```

Expected: `OK`

- [ ] **Step 2: Run full test suite**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 3: Grep for any remaining hardcoded marginRight in layout modules**

```bash
grep -rn 'marginRight.*[345]px' btc_web/layout/ | grep -v '__pycache__'
```

Expected: No hits except non-checkbox margins (e.g., `labelStyle` in `_model_show_checklist` which uses `"marginRight": "12px"`, or the context menu `"marginRight": "8px"` in common.py, or icon spans with `"marginRight": "3px"`).

- [ ] **Step 4: Verify no remaining old label strings**

```bash
grep -rn 'Per-period amount\|Withdrawal/period\|Spending amount.*month\|Inflation rate.*% / yr' btc_web/layout/ | grep -v '__pycache__'
```

Expected: Zero hits (all old labels replaced). The grep for `"Inflation rate"` should only match the `_INFL_LABEL` constant definition in `common.py`.
