# Default Update Sanity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate ~23 default value divergences across the app by consolidating into a single `MappingProxyType`-based source of truth per tab, with immutability protection and drift-detection tests.

**Architecture:** One new file `btc_web/tab_defaults.py` holds frozen `MappingProxyType` dicts per tab (static values) plus `_defaults()` functions (dynamic values like `yr_now`). All consumers — layout builders, callback fallbacks, figure builder fallbacks, prewarm, and the Citadel cache generator — import from this file instead of hardcoding. Tests enforce immutability, tuple-only inner values, and layout-vs-defaults consistency.

**Tech Stack:** Python 3, `types.MappingProxyType`, `pytest`, Plotly Dash 4.0.0

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `btc_web/tab_defaults.py` | **Create** | Canonical defaults for all 7+1 tabs (Bubble, Heatmap, DCA, Retire, Supercharge, Stack, Citadel, plus MC controls template) |
| `btc_web/layout/bubble.py` | Modify | Replace hardcoded `value=` props with `BUBBLE["key"]` references |
| `btc_web/layout/heatmap.py` | Modify | Replace hardcoded `value=` props with `HEATMAP["key"]` references |
| `btc_web/layout/sim_tabs.py` | Modify | Replace hardcoded `value=` props for DCA + Retire with `DCA["key"]` / `RETIRE["key"]` |
| `btc_web/layout/supercharge.py` | Modify | Replace hardcoded `value=` props with `SUPERCHARGE["key"]` |
| `btc_web/layout/stack.py` | Modify | Replace hardcoded `value=` props with `STACK["key"]` |
| `btc_web/layout/citadel.py` | Modify | Replace hardcoded `value=` props with `CITADEL["key"]` |
| `btc_web/callbacks/charts.py` | Modify | Replace `_cf(val, HARDCODED)` with `_cf(val, DEFAULTS["key"])` |
| `btc_web/callbacks/citadel_cb.py` | Modify | Replace `_cf(val, HARDCODED)` with `_cf(val, CITADEL["key"])` |
| `btc_web/figures/bubble.py` | Modify | Replace `p.get("key", HARDCODED)` with `p.get("key", BUBBLE["key"])` |
| `btc_web/figures/dca.py` | Modify | Replace `p.get("key", HARDCODED)` with `p.get("key", DCA["key"])` |
| `btc_web/figures/retire.py` | Modify | Replace `p.get("key", HARDCODED)` with `p.get("key", RETIRE["key"])` |
| `btc_web/figures/supercharge.py` | Modify | Replace `p.get("key", HARDCODED)` with `p.get("key", SUPERCHARGE["key"])` |
| `btc_web/figures/citadel.py` | Modify | Replace `p.get("key", HARDCODED)` with `p.get("key", CITADEL["key"])` |
| `btc_web/figures/common.py` | Modify | Replace `_finalize_chart()` fallbacks (`legend_pos`, `show_legend`) — shared by DCA/Retire/SC/Citadel |
| `btc_web/app.py` | Modify | Replace `_prewarm_caches()` inline dicts with `_defaults()` calls |
| `btc_web/generate_citadel_cache.py` | Modify | Replace inline `p = {...}` with `citadel_defaults()` |
| `btc_web/engines/citadel.py` | Modify (light) | Document intentional divergences between `SimConfig` field defaults and `CITADEL` UI defaults |
| `btc_web/test_defaults.py` | **Create** | All 6 test categories from the spec |

---

## Confirmed Divergences to Fix

These are verified from the current codebase. **Layout values are the source of truth** (what the user sees on page load).

| # | Tab | Param | Layout (truth) | Callback | Figure | Prewarm | CacheGen | Fix |
|---|-----|-------|----------------|----------|--------|---------|----------|-----|
| 1 | Bubble | `pt_alpha` | 0.3 | **0.6** | **0.6** | 0.3 | — | CB+Fig→0.3 |
| 2 | Bubble | `show_sup` | True | — | — | **False** | — | Prewarm→True |
| 3 | Bubble | `legend_pos` | "top-left" | — | **"outside"** | **"outside"** | — | Fig+PW→"top-left" |
| 4 | Heatmap | `show_colorbar` | True | — | — | **False** | — | PW→True |
| 5 | Heatmap | `exit_yr_hi` | yr_now+15 | — | — | **yr_now+10** | — | PW→yr_now+15 |
| 6 | Heatmap | colors | "forge" palette | — | — | **M.CAGR_SEG_*** | — | PW→"forge" |
| 7 | DCA | `annotate` | True | — | — | **False** | — | PW→True |
| 8 | DCA | `legend_pos` | "bottom-right" | — | — | **"outside"** | — | PW→"bottom-right" |
| 9 | DCA | `sc_term_months` | 12 | 12 | 12 | **48** | — | PW→12 |
| 10 | Retire | `inflation` | 4% | **0%** | **0%** | 4% | — | CB+Fig→4% |
| 11 | Retire | `legend_pos` | "bottom-right" | — | — | **"outside"** | — | PW→"bottom-right" |
| 12 | Retire | `yr_range` | [2031,2075] | **[2025,2045]** | — | [2031,2075] | — | CB→[2031,2075] |
| 13 | SC | `freq` | "Monthly" | "Monthly" | "Monthly" | **"Annually"** | — | PW→"Monthly" |
| 14 | SC | `legend_pos` | "top-left" | — | **"outside"** | **"outside"** | — | Fig+PW→"top-left" |
| 15 | SC | `display_q` | Q5% | — | **Q50%** | Q5% | — | Fig→Q5% |
| 16 | Citadel | `high_q_trigger` | 95 | 95 | **80** | **80** | 95 | Fig+PW→95 |
| 17 | Citadel | `low_q_trigger` | 5 | 5 | **20** | **20** | 5 | Fig+PW→5 |
| 18 | Citadel | `cash_floor` | 50000 | **0** | — | **0** | 50000 | CB+PW→50000 |
| 19 | Bubble | `n_future` | 3 | **0** | — | 3 | — | CB→3 |
| 20 | Bubble | `xscale` | "log" | **"linear"** | — | "log" | — | CB→"log" |
| 21 | Citadel | `show_legend` | True | — | — | **False** | True | PW→True |
| 22 | Citadel | `log_y` | True | — | — | True | **False** | CacheGen→True |
| 23 | Citadel | `low_q_split_*` | 20/20/20/10/20/10 | **10/10/10/10/40/20** | **10/10/10/10/40/20** | — | 10/10/10/10/40/20 | CB+Fig+CG→layout |

---

## Task 1: Create `btc_web/tab_defaults.py` — Static Defaults

**Files:**
- Create: `btc_web/tab_defaults.py`
- Test: `btc_web/test_defaults.py`

- [ ] **Step 1: Write the immutability test**

```python
# btc_web/test_defaults.py
"""Tests for tab_defaults.py — single source of truth for all tab defaults."""
import pytest


def test_defaults_are_immutable():
    """MappingProxyType prevents mutation."""
    from tab_defaults import BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE, STACK, CITADEL
    for name, defaults in [("BUBBLE", BUBBLE), ("HEATMAP", HEATMAP),
                           ("DCA", DCA), ("RETIRE", RETIRE),
                           ("SUPERCHARGE", SUPERCHARGE), ("STACK", STACK),
                           ("CITADEL", CITADEL)]:
        with pytest.raises(TypeError, match="does not support item assignment"):
            defaults["new_key"] = "bad"


def test_inner_collections_are_tuples():
    """Inner values must be tuples/frozensets, not lists/sets."""
    from tab_defaults import BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE, STACK, CITADEL
    for name, defaults in [("BUBBLE", BUBBLE), ("HEATMAP", HEATMAP),
                           ("DCA", DCA), ("RETIRE", RETIRE),
                           ("SUPERCHARGE", SUPERCHARGE), ("STACK", STACK),
                           ("CITADEL", CITADEL)]:
        for key, val in defaults.items():
            assert not isinstance(val, list), f"{name}[{key!r}] is a list, should be tuple"
            assert not isinstance(val, set), f"{name}[{key!r}] is a set, should be frozenset"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py -v`
Expected: ImportError — `tab_defaults` module does not exist yet.

- [ ] **Step 3: Create `tab_defaults.py` with all static defaults**

Create `btc_web/tab_defaults.py`. **Layout values are the source of truth.** Every `MappingProxyType` uses tuples (not lists) for inner collections.

The file must contain:
- Module docstring with usage constraints (no `json.dumps()` on proxies, always use `dict(DEFAULTS)` or `_defaults()`, inner values are tuples)
- `BUBBLE` — all Bubble tab static defaults (see extracted data below)
- `HEATMAP` — all Heatmap tab static defaults
- `DCA` — all DCA tab static defaults
- `RETIRE` — all Retire tab static defaults
- `SUPERCHARGE` — all Supercharge tab static defaults
- `STACK` — Stack Tracker static defaults
- `CITADEL` — Citadel Planner static defaults

**Key values to get right (these fix divergences):**

```python
# BUBBLE
"pt_alpha": 0.3,          # NOT 0.6 (callbacks/figures had wrong fallback)
"show_sup": True,          # NOT False (prewarm was wrong)
"legend_pos": "top-left",  # NOT "outside" (prewarm/figures had wrong fallback)

# HEATMAP
"show_colorbar": True,     # NOT False (prewarm was wrong)
# exit_yr_hi is dynamic — handled in heatmap_defaults()

# DCA
"annotate": True,          # NOT False (prewarm was wrong)
"legend_pos": "bottom-right",  # NOT "outside"
"sc_term_months": 12,      # NOT 48 (prewarm was wrong)

# RETIRE
"inflation": 4.0,          # NOT 0 (callbacks/figures had wrong fallback)
"legend_pos": "bottom-right",  # NOT "outside"
"start_yr": 2031,          # NOT yr_now (callback was wrong)

# SUPERCHARGE
"legend_pos": "top-left",  # NOT "outside"
"display_q": 0.05,         # NOT 0.5 (figure builder had wrong fallback)

# CITADEL
"high_q_trigger": 95,      # NOT 80 (figure/prewarm were wrong)
"low_q_trigger": 5,        # NOT 20 (figure/prewarm were wrong)
"cash_floor": 50000,       # NOT 0 (callback/prewarm were wrong)
```

**Complete BUBBLE dict:**
```python
BUBBLE = MappingProxyType({
    # Quantiles
    "selected_qs": (0.5,),
    # Axes & range
    "xscale": "log",
    "yscale": "log",
    "auto_y": ("yes",),
    "ymin": 1.0, "ymax": 1e7,  # actual prices (10**0, 10**7) — figure builder reads p["ymin"] directly
    # Display toggles
    "shade": True, "show_data": True, "show_today": True,
    "show_legend": False, "minor_grid": False,
    "show_ols": False, "show_ucl": False,
    # Bubble model
    "show_comp": True, "show_sup": True,
    "n_future": 3,
    # Data point appearance
    "pt_size": 3, "pt_alpha": 0.3,
    # Stack
    "stack": 0, "show_stack": False, "use_lots": False,
    # Legend
    "legend_pos": "top-left",
    # Composite/support colors
    "comp_color": "#FFD700", "comp_lw": 2.0,
    "sup_color": "#888888", "sup_lw": 1.5,
    # Model display
    "active_models": ("bub",),
    # Palette
    "palette": "default",
    # Scanner
    "scanner_lines": (),
})
```

**Complete HEATMAP dict:**
```python
HEATMAP = MappingProxyType({
    # exit_qs, entry_yr, entry_q, exit_yr_lo, exit_yr_hi are dynamic
    "exit_qs": (),
    "color_mode": 0,
    "b1": 0, "b2": 20,
    "hm_palette": "forge",
    "c_lo": "#1b0a2e", "c_mid1": "#2c2c3a",
    "c_mid2": "#1b4332", "c_hi": "#ffd700",
    "n_disc": 32,
    "vfmt": "cagr",
    "cell_font_size": 9,
    "show_colorbar": True,
    "stack": 0, "use_lots": False,
    "hm_model": "bub",
    "active_models": (),
    "palette": "default",
})
```

**Complete DCA dict:**
```python
DCA = MappingProxyType({
    "start_stack": 0, "use_lots": False,
    "amount": 100, "freq": "Monthly", "inflation": 0.0,
    "selected_qs": (0.5,),
    "disp_mode": "btc",
    "annotate": True, "show_today": False,
    "show_legend": False, "minor_grid": False,
    "log_y": False,
    "legend_pos": "bottom-right",
    "active_models": (),
    "palette": "default",
    # Stack-celerator
    "sc_enabled": False, "sc_loan_amount": 1200,
    "sc_rate": 13.0, "sc_loan_type": "interest_only",
    "sc_term_months": 12, "sc_repeats": 0, "sc_rollover": False,
    "sc_entry_mode": "live", "sc_custom_price": 80000.0,
    "sc_tax_rate": 0.33,
    # Show model toggles
    "show_qr": True, "show_mc": False,
})
```

**Complete RETIRE dict:**
```python
RETIRE = MappingProxyType({
    "start_stack": 1.0, "use_lots": False,
    "wd_amount": 5000, "freq": "Monthly",
    "inflation": 4.0,
    "selected_qs": (0.01, 0.10, 0.25),
    "start_yr": 2031, "end_yr": 2075,
    "disp_mode": "btc",
    "annotate": True, "log_y": True,
    "show_legend": False, "minor_grid": True,
    "legend_pos": "bottom-right",
    "active_models": (),
    "palette": "default",
    "show_qr": True, "show_mc": False,
})
```

**Complete SUPERCHARGE dict:**
```python
SUPERCHARGE = MappingProxyType({
    "mode": "a", "start_stack": 1.0, "use_lots": False,
    "start_yr": 2033,
    "delays": (0.0, 0.0, 0.0, 1.0, 2.0),
    "freq": "Monthly", "inflation": 4.0,
    "selected_qs": (0.001, 0.10),  # filtered against model.fits at runtime
    "chart_layout": 2,  # shade bands on
    "display_q": 0.05,
    "wd_amount": 5000, "end_yr": 2075,
    "disp_mode": "usd",
    "annotate": True, "log_y": True,
    "show_legend": False, "minor_grid": True,
    "legend_pos": "top-left",
    "target_yr": 2060,
    "active_models": (),
    "palette": "default",
    "show_qr": True, "show_mc": False,
})
```

**Complete STACK dict:**
```python
STACK = MappingProxyType({
    "lot_btc": 0.01,
    "lot_price": 69420,
    "lot_notes": "",
})
```

**Complete CITADEL dict:**
```python
CITADEL = MappingProxyType({
    "start_stack": 1.0, "use_lots": False,
    # Cash
    "cash_initial": 50000, "cash_rate": 4.0,
    # Reserves
    "res_short_init": 50000, "res_short_rate": 5.0, "res_short_vol": 2.0,
    "res_med_init": 100000, "res_med_rate": 4.5, "res_med_vol": 8.0,
    "res_long_init": 50000, "res_long_rate": 4.0, "res_long_vol": 15.0,
    # Investments
    "inv_eq_init": 200000, "inv_eq_rate": 10.0, "inv_eq_vol": 16.0,
    "inv_bd_init": 100000, "inv_bd_rate": 5.0, "inv_bd_vol": 7.0,
    # Spending
    "monthly_spend": 5000, "inflation": 4.0, "spend_growth": 0.0,
    # High-Q rebalancing
    "high_q_trigger": 95, "high_q_mode": "gradual", "high_q_rate": 2.0, "high_q_dur": 6,
    "high_q_split_cash": 20, "high_q_split_rs": 20, "high_q_split_rm": 20,
    "high_q_split_rl": 10, "high_q_split_eq": 20, "high_q_split_bd": 10,
    # Low-Q rebalancing (layout uses shared _trigger_section template → same splits as high-q)
    "low_q_trigger": 5, "low_q_mode": "lump", "low_q_rate": 10.0, "low_q_dur": 1,
    "low_q_split_cash": 20, "low_q_split_rs": 20, "low_q_split_rm": 20,
    "low_q_split_rl": 10, "low_q_split_eq": 20, "low_q_split_bd": 10,
    # Cooldown
    "lump_cooldown": 12,
    # Floors
    "cash_floor": 50000, "res_short_floor": 0, "res_med_floor": 0, "res_long_floor": 0,
    "cash_floor_growth": 0, "reserve_floor_growth": 0,
    # SCF
    "scf_enabled": False, "scf_amount": 100000, "scf_type": "term",
    "scf_rate": 8.0, "scf_term": 60, "scf_repay_trigger": 1.0,
    # Simulation
    "start_yr": 2031, "end_yr": 2075, "freq": "Monthly",
    "price_model": "bub", "asset_return_model": "lognormal",
    "selected_qs": (0.25,),
    "disp_mode": "usd_per_asset",
    # Chart
    "annotate": True, "log_y": True, "show_legend": True, "minor_grid": True,
    "legend_pos": "bottom-right",
    "palette": "default",
})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add btc_web/tab_defaults.py btc_web/test_defaults.py
git commit -m "feat: add tab_defaults.py with immutable MappingProxyType defaults"
```

---

## Task 2: Add Dynamic Defaults Functions

**Files:**
- Modify: `btc_web/tab_defaults.py`
- Test: `btc_web/test_defaults.py`

- [ ] **Step 1: Write the test for dynamic defaults**

Append to `test_defaults.py`:

```python
def test_bubble_defaults_returns_mutable_dict():
    from tab_defaults import bubble_defaults
    d = bubble_defaults()
    assert isinstance(d, dict)
    # Must be mutable
    d["new_key"] = "ok"
    # Must have dynamic xrange
    assert "xmin" in d and "xmax" in d
    # selected_qs must be a list (mutable copy from tuple)
    assert isinstance(d["selected_qs"], list)


def test_heatmap_defaults_returns_mutable_dict():
    from tab_defaults import heatmap_defaults
    d = heatmap_defaults()
    assert isinstance(d, dict)
    assert "entry_yr" in d
    assert "exit_yr_lo" in d
    assert "exit_yr_hi" in d


def test_dca_defaults_returns_mutable_dict():
    from tab_defaults import dca_defaults
    d = dca_defaults()
    assert isinstance(d, dict)
    assert "start_yr" in d and "end_yr" in d
    assert isinstance(d["selected_qs"], list)


def test_retire_defaults_has_correct_static_yr():
    """Retire defaults use fixed years (2031-2075), not dynamic."""
    from tab_defaults import retire_defaults
    d = retire_defaults()
    assert d["start_yr"] == 2031
    assert d["end_yr"] == 2075
    assert isinstance(d["selected_qs"], list)


def test_supercharge_defaults_has_list_delays():
    from tab_defaults import supercharge_defaults
    d = supercharge_defaults()
    assert isinstance(d["delays"], list)
    assert isinstance(d["selected_qs"], list)


def test_citadel_defaults_returns_mutable_dict():
    from tab_defaults import citadel_defaults
    d = citadel_defaults()
    assert isinstance(d, dict)
    assert d["high_q_trigger"] == 95
    assert d["cash_floor"] == 50000
    assert isinstance(d["selected_qs"], list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py -v`
Expected: ImportError for `bubble_defaults` etc.

- [ ] **Step 3: Add `_defaults()` functions to `tab_defaults.py`**

Add these functions after the frozen dicts:

```python
import pandas as pd

def bubble_defaults() -> dict:
    """Mutable dict with dynamic values resolved."""
    yr_now = pd.Timestamp.today().year
    d = dict(BUBBLE)
    d["xmin"] = 2012
    d["xmax"] = yr_now + 4
    d["selected_qs"] = list(BUBBLE["selected_qs"])
    d["active_models"] = list(BUBBLE["active_models"])
    d["scanner_lines"] = list(BUBBLE["scanner_lines"])
    d["auto_y"] = list(BUBBLE["auto_y"])
    d["lots"] = []
    return d

def heatmap_defaults() -> dict:
    yr_now = pd.Timestamp.today().year
    d = dict(HEATMAP)
    d["entry_yr"] = yr_now
    d["entry_q"] = 50.0  # safe fallback; caller should override with live ticker value
    d["exit_yr_lo"] = yr_now
    d["exit_yr_hi"] = yr_now + 15
    d["exit_qs"] = list(HEATMAP["exit_qs"])
    d["active_models"] = list(HEATMAP["active_models"])
    d["lots"] = []
    return d

def dca_defaults() -> dict:
    yr_now = pd.Timestamp.today().year
    d = dict(DCA)
    d["start_yr"] = yr_now
    d["end_yr"] = yr_now + 10
    d["selected_qs"] = list(DCA["selected_qs"])
    d["active_models"] = list(DCA["active_models"])
    d["lots"] = []
    return d

def retire_defaults() -> dict:
    d = dict(RETIRE)
    d["selected_qs"] = list(RETIRE["selected_qs"])
    d["active_models"] = list(RETIRE["active_models"])
    d["lots"] = []
    return d

def supercharge_defaults() -> dict:
    d = dict(SUPERCHARGE)
    d["delays"] = list(SUPERCHARGE["delays"])
    d["selected_qs"] = list(SUPERCHARGE["selected_qs"])
    d["active_models"] = list(SUPERCHARGE["active_models"])
    d["lots"] = []
    return d

def citadel_defaults() -> dict:
    d = dict(CITADEL)
    d["selected_qs"] = list(CITADEL["selected_qs"])
    d["lots"] = []
    return d
```

**Important:** `selected_qs` for Supercharge uses `(0.001, 0.10)` in the frozen dict but the actual layout filters these against the model's available quantiles. The `_defaults()` function returns them as-is; the caller (layout/prewarm) filters against `model.fits` if needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add btc_web/tab_defaults.py btc_web/test_defaults.py
git commit -m "feat: add dynamic _defaults() functions for all tabs"
```

---

## Task 3: Wire Bubble Tab — Layout + Callbacks + Figure Builder

**Files:**
- Modify: `btc_web/layout/bubble.py`
- Modify: `btc_web/callbacks/charts.py` (bubble callback section)
- Modify: `btc_web/figures/bubble.py`

- [ ] **Step 1: Wire layout/bubble.py**

Add import at top:
```python
from tab_defaults import BUBBLE
```

Replace every hardcoded `value=` prop with `BUBBLE["key"]` reference. Key replacements:

| Component ID | Old `value=` | New `value=` |
|-------------|-------------|-------------|
| `bub-ptsize` | `3` | `BUBBLE["pt_size"]` |
| `bub-ptalpha` | `0.3` | `BUBBLE["pt_alpha"]` |
| `bub-n-future` | `3` | `BUBBLE["n_future"]` |
| `bub-xscale` | `"log"` | `BUBBLE["xscale"]` |
| `bub-yscale` | `"log"` | `BUBBLE["yscale"]` |
| `bub-stack` | `0` | `BUBBLE["stack"]` |
| `bub-legend-pos` | `"top-left"` | `BUBBLE["legend_pos"]` |

For checklist `bub-toggles`, the current `value=["shade","show_data","show_today"]` should derive from BUBBLE keys. The simplest approach: keep the layout value as-is (since it matches BUBBLE) but reference BUBBLE for documentation. The important thing is callback/figure fallbacks match.

- [ ] **Step 2: Wire callbacks/charts.py — bubble fallbacks**

Add import:
```python
from tab_defaults import BUBBLE
```

Replace these specific hardcoded fallbacks:
- Line 75: `xscale or "linear"` → `xscale or BUBBLE["xscale"]` — **fixes divergence #20**
- Line 79: `_ci(n_future, 0)` → `_ci(n_future, BUBBLE["n_future"])` — **fixes divergence #19**
- Line 80: `_ci(ptsize, 3)` → `_ci(ptsize, BUBBLE["pt_size"])`
- Line 81: `_cf(ptalpha, 0.6)` → `_cf(ptalpha, BUBBLE["pt_alpha"])` — **fixes divergence #1**
- Line 82: `_cf(stack, 0)` → `_cf(stack, BUBBLE["stack"])`

- [ ] **Step 3: Wire figures/bubble.py — p.get() fallbacks**

Add import:
```python
from tab_defaults import BUBBLE
```

Replace these specific hardcoded fallbacks:
- `p.get("pt_alpha", 0.6)` → `p.get("pt_alpha", BUBBLE["pt_alpha"])` — **fixes divergence #1**
- `p.get("pt_size", 3)` → `p.get("pt_size", BUBBLE["pt_size"])`
- `p.get("n_future", 0)` → `p.get("n_future", BUBBLE["n_future"])`
- `p.get("show_legend", True)` → `p.get("show_legend", BUBBLE["show_legend"])`
- `p.get("legend_pos", "outside")` → `p.get("legend_pos", BUBBLE["legend_pos"])` — **fixes divergence #3**
- `p.get("yscale", "log")` → `p.get("yscale", BUBBLE["yscale"])`
- `p.get("xscale", "linear")` → `p.get("xscale", BUBBLE["xscale"])`
- `p.get("sup_color", "#888888")` → `p.get("sup_color", BUBBLE["sup_color"])`
- `p.get("sup_lw", 1.5)` → `p.get("sup_lw", BUBBLE["sup_lw"])`
- `p.get("comp_color", "#FFD700")` → `p.get("comp_color", BUBBLE["comp_color"])`
- `p.get("comp_lw", 2.0)` → `p.get("comp_lw", BUBBLE["comp_lw"])`

- [ ] **Step 4: Run syntax check**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -c "from tab_defaults import BUBBLE; from figures.bubble import build_bubble_figure; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Run existing tests**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -k "bubble" -v --timeout=60`
Expected: All bubble-related tests pass.

- [ ] **Step 6: Commit**

```bash
git add btc_web/layout/bubble.py btc_web/callbacks/charts.py btc_web/figures/bubble.py
git commit -m "refactor: wire Bubble tab to tab_defaults (fixes pt_alpha, legend_pos, show_sup divergences)"
```

---

## Task 4: Wire Heatmap Tab — Layout + Callbacks + Figure Builder

**Files:**
- Modify: `btc_web/layout/heatmap.py`
- Modify: `btc_web/callbacks/charts.py` (heatmap callback section)
- Modify: `btc_web/figures/heatmap.py`

- [ ] **Step 1: Wire layout/heatmap.py**

Add import:
```python
from tab_defaults import HEATMAP
```

Replace hardcoded values:
| Component ID | Old | New |
|-------------|-----|-----|
| `hm-mode` | `0` | `HEATMAP["color_mode"]` |
| `hm-b1` | `0` | `HEATMAP["b1"]` |
| `hm-b2` | `20` | `HEATMAP["b2"]` |
| `hm-grad` | `32` | `HEATMAP["n_disc"]` |
| `hm-vfmt` | `"cagr"` | `HEATMAP["vfmt"]` |
| `hm-cell-fs` | `9` | `HEATMAP["cell_font_size"]` |
| `hm-stack` | `0` | `HEATMAP["stack"]` |

- [ ] **Step 2: Wire callbacks/charts.py — heatmap fallbacks**

Replace:
- `_ci(mode, 0)` → `_ci(mode, HEATMAP["color_mode"])`
- `_cf(b1, _app_ctx.M.CAGR_SEG_B1)` → leave as-is (these are model-derived, not tab defaults)
- `_cf(entry_q, 50)` → `_cf(entry_q, 50)` — leave as-is (dynamic from ticker)
- `_ci(grad, _app_ctx.M.CAGR_GRAD_STEPS)` → `_ci(grad, HEATMAP["n_disc"])`
- `_ci(cell_fs, 9)` → `_ci(cell_fs, HEATMAP["cell_font_size"])`
- `_cf(stack, 0)` → `_cf(stack, HEATMAP["stack"])`

Add `from tab_defaults import HEATMAP` to imports.

- [ ] **Step 3: Wire figures/heatmap.py (if it has hardcoded fallbacks)**

Check `btc_web/figures/heatmap.py` for `p.get()` with hardcoded defaults and replace with `HEATMAP["key"]` references.

- [ ] **Step 4: Run syntax check + tests**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -k "heatmap" -v --timeout=60`
Expected: All heatmap tests pass.

- [ ] **Step 5: Commit**

```bash
git add btc_web/layout/heatmap.py btc_web/callbacks/charts.py btc_web/figures/heatmap.py
git commit -m "refactor: wire Heatmap tab to tab_defaults (fixes show_colorbar, exit_yr_hi divergences)"
```

---

## Task 5: Wire DCA Tab — Layout + Callbacks + Figure Builder

**Files:**
- Modify: `btc_web/layout/sim_tabs.py` (DCA section)
- Modify: `btc_web/callbacks/charts.py` (DCA callback section)
- Modify: `btc_web/figures/dca.py`

- [ ] **Step 1: Wire layout/sim_tabs.py — DCA controls**

Add import:
```python
from tab_defaults import DCA
```

Replace hardcoded DCA `value=` props:
| Component ID | Old | New |
|-------------|-----|-----|
| `dca-amount` | `100` | `DCA["amount"]` |
| `dca-freq` | `"Monthly"` | `DCA["freq"]` |
| `dca-infl` | `0` | `DCA["inflation"]` |
| `dca-stack` | `0` | `DCA["start_stack"]` |
| `dca-legend-pos` | `"bottom-right"` | `DCA["legend_pos"]` |

- [ ] **Step 2: Wire callbacks/charts.py — DCA fallbacks**

Replace:
- `_ci(amount, 100, ...)` → `_ci(amount, DCA["amount"], ...)`
- `_cf(dca_infl, 0)` → `_cf(dca_infl, DCA["inflation"])`
- `_cf(stack, 0)` → `_cf(stack, DCA["start_stack"])`
- `_cf(sc_term, 12)` → `_cf(sc_term, DCA["sc_term_months"])`
- `_cf(sc_rate, _app_ctx.SC_DEFAULT_RATE)` → `_cf(sc_rate, DCA["sc_rate"])`
- `_cf(sc_tax, 33, ...)` → `_cf(sc_tax, DCA["sc_tax_rate"] * 100, ...)`

Note: `sc_tax_rate` in DCA defaults is stored as 0.33 (fraction), matching the figure builder convention (`p.get("sc_tax_rate", 0.33)`). The callback converts from UI percentage: `_cf(sc_tax, 33, ...) / 100.0`. Keep the defaults at 0.33 (fraction) — the callback does the division.

- [ ] **Step 3: Wire figures/common.py — shared `_finalize_chart()` fallbacks**

`_finalize_chart()` in `figures/common.py` (line ~600) serves DCA, Retire, Supercharge, and Citadel. It has:
- `p.get("show_legend", True)` → leave as-is (True is a safe fallback; each tab's callback always provides this)
- `p.get("legend_pos", "outside")` → leave as-is (each tab's callback always provides this explicitly)

These fallbacks are only hit if a callback fails to supply the key. Since all callbacks do supply them, no change is strictly needed. But if you want belt-and-suspenders, you could import a default — the trade-off is that `_finalize_chart` serves multiple tabs with different legend defaults. **Decision: leave `figures/common.py` unchanged** — the per-tab fallbacks in callbacks and figure builders are the ones that matter.

- [ ] **Step 4: Wire figures/dca.py — p.get() fallbacks**

Add import: `from tab_defaults import DCA`

Replace:
- `p.get("amount", 100)` → `p.get("amount", DCA["amount"])`
- `p.get("inflation", 0)` → `p.get("inflation", DCA["inflation"])` — **fixes divergence #10 for DCA**
- `p.get("sc_term_months", 12)` → `p.get("sc_term_months", DCA["sc_term_months"])`
- `p.get("sc_rate", 13.0)` → `p.get("sc_rate", DCA["sc_rate"])`
- `p.get("sc_tax_rate", 0.33)` → `p.get("sc_tax_rate", DCA["sc_tax_rate"])`
- `p.get("sc_custom_price", 0)` → `p.get("sc_custom_price", DCA["sc_custom_price"])`

- [ ] **Step 5: Run tests**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -k "dca" -v --timeout=60`
Expected: All DCA tests pass.

- [ ] **Step 6: Commit**

```bash
git add btc_web/layout/sim_tabs.py btc_web/callbacks/charts.py btc_web/figures/dca.py
git commit -m "refactor: wire DCA tab to tab_defaults (fixes annotate, legend_pos, sc_term divergences)"
```

---

## Task 6: Wire Retire Tab — Layout + Callbacks + Figure Builder

**Files:**
- Modify: `btc_web/layout/sim_tabs.py` (Retire section)
- Modify: `btc_web/callbacks/charts.py` (Retire callback section)
- Modify: `btc_web/figures/retire.py`

- [ ] **Step 1: Wire layout/sim_tabs.py — Retire controls**

Add import (if not already): `from tab_defaults import RETIRE`

Replace hardcoded Retire `value=` props:
| Component ID | Old | New |
|-------------|-----|-----|
| `ret-stack` | `1.0` | `RETIRE["start_stack"]` |
| `ret-wd` | `5000` | `RETIRE["wd_amount"]` |
| `ret-freq` | `"Monthly"` | `RETIRE["freq"]` |
| `ret-infl` | `4` | `RETIRE["inflation"]` |
| `ret-legend-pos` | `"bottom-right"` | `RETIRE["legend_pos"]` |

- [ ] **Step 2: Wire callbacks/charts.py — Retire fallbacks**

Replace:
- `_cf(stack, 1.0)` → `_cf(stack, RETIRE["start_stack"])`
- `_ci(wd, 5000, ...)` → `_ci(wd, RETIRE["wd_amount"], ...)`
- `_cf(infl, 0)` → `_cf(infl, RETIRE["inflation"])` — **fixes divergence #10**
- `yr_range or [2025, 2045]` → `yr_range or [RETIRE["start_yr"], RETIRE["end_yr"]]` — **fixes divergence #12**

- [ ] **Step 3: Wire figures/retire.py — p.get() fallbacks**

Add import: `from tab_defaults import RETIRE`

Replace:
- `p.get("wd_amount", 5000)` → `p.get("wd_amount", RETIRE["wd_amount"])`
- `p.get("inflation", 0)` → `p.get("inflation", RETIRE["inflation"])` — **fixes divergence #10**

- [ ] **Step 4: Run tests**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -k "retire" -v --timeout=60`
Expected: All retire tests pass.

- [ ] **Step 5: Commit**

```bash
git add btc_web/layout/sim_tabs.py btc_web/callbacks/charts.py btc_web/figures/retire.py
git commit -m "refactor: wire Retire tab to tab_defaults (fixes inflation fallback, legend_pos divergences)"
```

---

## Task 7: Wire Supercharge Tab — Layout + Callbacks + Figure Builder

**Files:**
- Modify: `btc_web/layout/supercharge.py`
- Modify: `btc_web/callbacks/charts.py` (Supercharge callback section)
- Modify: `btc_web/figures/supercharge.py`

- [ ] **Step 1: Wire layout/supercharge.py**

Add import: `from tab_defaults import SUPERCHARGE`

Replace hardcoded `value=` props:
| Component ID | Old | New |
|-------------|-----|-----|
| `sc-stack` | `1.0` | `SUPERCHARGE["start_stack"]` |
| `sc-infl` | `4` | `SUPERCHARGE["inflation"]` |
| `sc-mode` | `"a"` | `SUPERCHARGE["mode"]` |
| `sc-start-yr` | `2033` | `SUPERCHARGE["start_yr"]` |
| `sc-wd` | `5000` (via `SC_DEFAULT_WD`) | `SUPERCHARGE["wd_amount"]` |
| `sc-end-yr` | `2075` | `SUPERCHARGE["end_yr"]` |
| `sc-target-yr` | `2060` | `SUPERCHARGE["target_yr"]` |
| `sc-legend-pos` | `"top-left"` | `SUPERCHARGE["legend_pos"]` |

Note: SC `wd_amount` in layout is set via `_app_ctx.SC_DEFAULT_WD` which is 5000. The prewarm also uses 5000. These match — no divergence here.

- [ ] **Step 2: Wire callbacks/charts.py — Supercharge fallbacks**

Replace:
- `_cf(stack, 1.0)` → `_cf(stack, SUPERCHARGE["start_stack"])`
- `_cf(infl, 4.0)` → `_cf(infl, SUPERCHARGE["inflation"])`
- `_ci(start_yr, yr_now)` → `_ci(start_yr, SUPERCHARGE["start_yr"])`
- `_cf(display_q, ...)` → `_cf(display_q, SUPERCHARGE["display_q"])`
- `_ci(wd, 5000, ...)` → `_ci(wd, SUPERCHARGE["wd_amount"], ...)`
- `_ci(end_yr, 2075)` → `_ci(end_yr, SUPERCHARGE["end_yr"])`
- `_ci(target_yr, 2060)` → `_ci(target_yr, SUPERCHARGE["target_yr"])`

- [ ] **Step 3: Wire figures/supercharge.py — p.get() fallbacks**

Add import: `from tab_defaults import SUPERCHARGE`

Replace:
- `p.get("display_q", 0.5)` → `p.get("display_q", SUPERCHARGE["display_q"])` — **fixes divergence #15**
- `p.get("inflation", 4)` → `p.get("inflation", SUPERCHARGE["inflation"])`
- `p.get("mode", "a")` → `p.get("mode", SUPERCHARGE["mode"])`
- `p.get("start_yr", ...)` → `p.get("start_yr", SUPERCHARGE["start_yr"])`
- `p.get("end_yr", 2075)` → `p.get("end_yr", SUPERCHARGE["end_yr"])`
- `p.get("wd_amount", 5000)` → `p.get("wd_amount", SUPERCHARGE["wd_amount"])`
- `p.get("show_legend", True)` → `p.get("show_legend", SUPERCHARGE["show_legend"])`
- `p.get("delays") or [0, 1, 2, 4, 8]` → `p.get("delays") or list(SUPERCHARGE["delays"])`
- `p.get("target_yr", 2060)` → `p.get("target_yr", SUPERCHARGE["target_yr"])`

- [ ] **Step 4: Run tests**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -k "supercharge" -v --timeout=60`
Expected: All supercharge tests pass.

- [ ] **Step 5: Commit**

```bash
git add btc_web/layout/supercharge.py btc_web/callbacks/charts.py btc_web/figures/supercharge.py
git commit -m "refactor: wire Supercharge tab to tab_defaults (fixes display_q, legend_pos divergences)"
```

---

## Task 8: Wire Citadel Tab — Layout + Callbacks + Figure Builder + Cache Generator

**Files:**
- Modify: `btc_web/layout/citadel.py`
- Modify: `btc_web/callbacks/citadel_cb.py`
- Modify: `btc_web/figures/citadel.py`
- Modify: `btc_web/generate_citadel_cache.py`

- [ ] **Step 1: Wire layout/citadel.py**

Add import: `from tab_defaults import CITADEL`

Replace all hardcoded `value=` props in Citadel layout with `CITADEL["key"]` references. There are ~50+ controls. Key ones:

| Component ID | Old | New |
|-------------|-----|-----|
| `cp-stack` | `1.0` | `CITADEL["start_stack"]` |
| `cp-cash-init` | `50000` | `CITADEL["cash_initial"]` |
| `cp-high-q-thresh` | `95` | `CITADEL["high_q_trigger"]` |
| `cp-low-q-thresh` | `5` | `CITADEL["low_q_trigger"]` |
| `cp-cash-floor` | `50000` | `CITADEL["cash_floor"]` |
| (all others) | hardcoded | `CITADEL["key"]` |

- [ ] **Step 2: Wire callbacks/citadel_cb.py — _cf() fallbacks**

Add import: `from tab_defaults import CITADEL`

Replace all `_cf(val, HARDCODED)` with `_cf(val, CITADEL["key"])`. **Key divergence fixes:**
- `_cf(cash_floor, 0, ...)` → `_cf(cash_floor, CITADEL["cash_floor"], ...)` — **fixes divergence #18**
- All other ~40 `_cf()` calls should reference CITADEL.

- [ ] **Step 3: Wire figures/citadel.py — p.get() fallbacks**

Add import: `from tab_defaults import CITADEL`

Replace all `p.get("key", HARDCODED)` with `p.get("key", CITADEL["key"])`. **Key divergence fixes:**
- `p.get("high_q_trigger", 80)` → `p.get("high_q_trigger", CITADEL["high_q_trigger"])` — **fixes divergence #16**
- `p.get("low_q_trigger", 20)` → `p.get("low_q_trigger", CITADEL["low_q_trigger"])` — **fixes divergence #17**
- All ~40 other `p.get()` calls.

- [ ] **Step 4: Wire generate_citadel_cache.py**

Add import: `from tab_defaults import citadel_defaults`

Replace the entire inline `p = {...}` dict (lines 83–118) with:
```python
p = citadel_defaults()
p["price_model"] = model_key
p["selected_qs"] = [q]
p["n_sims"] = 1
```

This eliminates 35 lines of duplicated defaults.

- [ ] **Step 5: Run tests**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -k "citadel" -v --timeout=60`
Expected: All citadel tests pass.

- [ ] **Step 6: Commit**

```bash
git add btc_web/layout/citadel.py btc_web/callbacks/citadel_cb.py btc_web/figures/citadel.py btc_web/generate_citadel_cache.py
git commit -m "refactor: wire Citadel tab to tab_defaults (fixes high/low_q_trigger, cash_floor divergences)"
```

---

## Task 9: Wire Stack Tracker + Prewarm

**Files:**
- Modify: `btc_web/layout/stack.py`
- Modify: `btc_web/app.py` (`_prewarm_caches()`)

- [ ] **Step 1: Wire layout/stack.py**

Add import: `from tab_defaults import STACK`

Replace:
- `lot-price` value `69420` → `STACK["lot_price"]`
- `lot-btc` value `0.01` → `STACK["lot_btc"]`

- [ ] **Step 2: Wire app.py — `_prewarm_caches()`**

Add imports at top of `_prewarm_caches()`:
```python
from tab_defaults import (bubble_defaults, heatmap_defaults, dca_defaults,
                          retire_defaults, supercharge_defaults, citadel_defaults)
```

Replace each inline dict with `_defaults()` call:

**Bubble prewarm (lines 199–216):** Replace with:
```python
_get_bubble_fig(bubble_defaults())
```

**Bubble+PL prewarm (lines 219–236):** Replace with:
```python
bub_pl = bubble_defaults()
bub_pl["active_models"] = ["bub", "pl"]
_get_bubble_fig(bub_pl)
```

**DCA prewarm (lines 239–258):** Replace with:
```python
_get_dca_fig(dca_defaults())
```

**Retire prewarm (lines 261–273):** Replace with:
```python
_get_retire_fig(retire_defaults())
```

**Supercharge prewarm (lines 276–301):** Replace with:
```python
sc = supercharge_defaults()
sc["selected_qs"] = [q for q in [0.001, 0.10] if q in _app_ctx.DEFAULT_MODEL.fits]
_get_supercharge_fig(sc)
```

**Citadel prewarm (lines 304–329):** Replace with:
```python
cp = citadel_defaults()
cp["selected_qs"] = [0.01, 0.10, 0.25]
_get_citadel_fig(cp)
```

**Heatmap prewarm (lines 333–357):** Replace with:
```python
hm = heatmap_defaults()
hm["entry_q"] = _app_ctx._HM_ENTRY_Q_DEFAULT
_get_heatmap_fig(hm)
```

**This fixes divergences #2, #3, #4, #5, #6, #7, #8, #9, #11, #14** — every prewarm cache miss caused by mismatched defaults.

- [ ] **Step 3: Run full test suite**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --timeout=120`
Expected: All 435+ tests pass.

- [ ] **Step 4: Commit**

```bash
git add btc_web/layout/stack.py btc_web/app.py
git commit -m "refactor: wire Stack tab + prewarm to tab_defaults (fixes all cache miss divergences)"
```

---

## Task 10: Align Engine Defaults + Document Intentional Differences

**Files:**
- Modify: `btc_web/engines/citadel.py` (comments only)

- [ ] **Step 1: Review SimConfig field defaults vs CITADEL dict**

Read `btc_web/engines/citadel.py` `SimConfig` dataclass. Compare each field default to `CITADEL` dict. Document in comments any intentional differences:

```python
@dataclass
class SimConfig:
    start_stack: float = 1.0  # matches CITADEL["start_stack"]
    cash_initial: float = 50_000  # matches CITADEL["cash_initial"]
    # ... etc ...
    n_sims: int = 1  # INTENTIONAL: engine default is 1 (deterministic), UI default varies
    cash_floor: float = 0.0  # INTENTIONAL: engine sentinel; UI default is 50000 via CITADEL["cash_floor"]
```

- [ ] **Step 2: Add docstring noting relationship to tab_defaults**

Add to SimConfig docstring:
```python
"""Simulation configuration for the Citadel Planner engine.

Field defaults here serve as the engine's 'unset sentinel' — what you get
if a field isn't explicitly passed. UI defaults live in tab_defaults.CITADEL.
Where they differ intentionally, a comment explains why.
"""
```

- [ ] **Step 3: Commit**

```bash
git add btc_web/engines/citadel.py
git commit -m "docs: document SimConfig vs CITADEL defaults relationship"
```

---

## Task 11: Add Drift Detection + Figure Builder Smoke Tests

**Files:**
- Modify: `btc_web/test_defaults.py`

- [ ] **Step 1: Write figure builder smoke tests**

Append to `test_defaults.py`:

```python
import os
os.environ.setdefault("TESTING", "1")


def test_bubble_defaults_produce_valid_figure():
    from tab_defaults import bubble_defaults
    from btc_web.app import app  # noqa: F401 — triggers model load
    import _app_ctx
    from figures.bubble import build_bubble_figure
    fig = build_bubble_figure(_app_ctx.M, bubble_defaults())
    assert len(fig.data) > 0


def test_heatmap_defaults_produce_valid_figure():
    from tab_defaults import heatmap_defaults
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from figures.heatmap import build_heatmap_figure
    d = heatmap_defaults()
    d["entry_q"] = 50.0
    fig = build_heatmap_figure(_app_ctx.M, d)
    assert len(fig.data) > 0


def test_dca_defaults_produce_valid_figure():
    from tab_defaults import dca_defaults
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from figures.dca import build_dca_figure
    fig = build_dca_figure(_app_ctx.M, dca_defaults())
    assert len(fig.data) > 0


def test_retire_defaults_produce_valid_figure():
    from tab_defaults import retire_defaults
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from figures.retire import build_retire_figure
    fig = build_retire_figure(_app_ctx.M, retire_defaults())
    assert len(fig.data) > 0


def test_supercharge_defaults_produce_valid_figure():
    from tab_defaults import supercharge_defaults
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from figures.supercharge import build_supercharge_figure
    d = supercharge_defaults()
    d["selected_qs"] = [q for q in [0.001, 0.10] if q in _app_ctx.DEFAULT_MODEL.fits]
    fig = build_supercharge_figure(_app_ctx.M, d)
    assert len(fig.data) > 0


def test_citadel_defaults_produce_valid_figure():
    from tab_defaults import citadel_defaults
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from figures.citadel import build_citadel_figure
    d = citadel_defaults()
    d["selected_qs"] = [0.25]
    fig_result = build_citadel_figure(_app_ctx.M, d)
    # build_citadel_figure returns (fig, mc_result) tuple
    fig = fig_result[0] if isinstance(fig_result, tuple) else fig_result
    assert len(fig.data) > 0
```

- [ ] **Step 2: Run smoke tests**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py::test_bubble_defaults_produce_valid_figure btc_web/test_defaults.py::test_heatmap_defaults_produce_valid_figure btc_web/test_defaults.py::test_dca_defaults_produce_valid_figure btc_web/test_defaults.py::test_retire_defaults_produce_valid_figure btc_web/test_defaults.py::test_supercharge_defaults_produce_valid_figure btc_web/test_defaults.py::test_citadel_defaults_produce_valid_figure -v --timeout=120`
Expected: All 6 tests PASS.

- [ ] **Step 3: Write drift detection test (layout tree walk)**

Append to `test_defaults.py`:

```python
def test_layout_values_match_defaults():
    """Walk the layout component tree, check value= props match defaults.

    This is the key anti-drift test. It catches future divergences between
    layout components and tab_defaults by programmatically inspecting the
    rendered layout.
    """
    from tab_defaults import BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE, STACK, CITADEL
    from btc_web.app import app
    from dash import dcc, html
    import dash_bootstrap_components as dbc

    # Map component ID prefix → frozen defaults dict
    # Key name mapping: component ID suffix → defaults key
    prefix_map = {
        "bub-": BUBBLE, "hm-": HEATMAP, "dca-": DCA,
        "ret-": RETIRE, "sc-": SUPERCHARGE, "cp-": CITADEL,
        "lot-": STACK,
    }

    # Some component IDs don't map directly to defaults keys.
    # This maps component_id → defaults_key for non-obvious mappings.
    # IMPORTANT: when adding new controls, check if the ID suffix matches
    # the defaults key. If not, add a mapping here.
    id_to_key = {
        # Bubble
        "bub-ptsize": "pt_size",
        "bub-ptalpha": "pt_alpha",
        "bub-n-future": "n_future",
        # Heatmap
        "hm-mode": "color_mode",
        "hm-grad": "n_disc",
        "hm-cell-fs": "cell_font_size",
        # DCA
        "dca-infl": "inflation",
        "dca-sc-loan": "sc_loan_amount",
        "dca-sc-term": "sc_term_months",
        "dca-sc-rate": "sc_rate",
        # NOTE: dca-sc-tax is in skip_ids — layout value is int % (33), defaults is fraction (0.33)
        "dca-sc-custom-price": "sc_custom_price",
        "dca-sc-repeats": "sc_repeats",
        # Retire
        "ret-wd": "wd_amount",
        "ret-infl": "inflation",
        # Supercharge
        "sc-infl": "inflation",
        "sc-wd": "wd_amount",
        "sc-start-yr": "start_yr",
        "sc-end-yr": "end_yr",
        "sc-target-yr": "target_yr",
        # Citadel
        "cp-cash-init": "cash_initial",
        "cp-cash-rate": "cash_rate",
        "cp-spend": "monthly_spend",
        "cp-infl": "inflation",
        "cp-spend-growth": "spend_growth",
        "cp-high-q-thresh": "high_q_trigger",
        "cp-high-q-rate": "high_q_rate",
        "cp-high-q-dur": "high_q_dur",
        "cp-low-q-thresh": "low_q_trigger",
        "cp-low-q-rate": "low_q_rate",
        "cp-low-q-dur": "low_q_dur",
        "cp-scf-amount": "scf_amount",
        "cp-scf-rate": "scf_rate",
        "cp-scf-term": "scf_term",
        "cp-scf-trigger": "scf_repay_trigger",
        "cp-cash-floor": "cash_floor",
        "cp-res-floor-growth": "reserve_floor_growth",
        # Stack
        "lot-price": "lot_price",
        "lot-btc": "lot_btc",
    }

    # Components to skip (dynamic defaults, not in frozen dict)
    skip_ids = {
        "bub-xrange", "bub-yrange", "bub-auto-y",
        "bub-toggles", "bub-bubble-toggles", "bub-qs",
        "bub-model-show", "bub-show-stack", "bub-use-lots",
        "scan-price", "scan-date", "scan-q",
        "hm-entry-yr", "hm-entry-q", "hm-exit-range", "hm-exit-qs",
        "hm-use-lots", "hm-palette", "hm-toggles", "hm-model-show",
        "hm-c-lo", "hm-c-mid1", "hm-c-mid2", "hm-c-hi",
        "dca-yr-range", "dca-qs", "dca-toggles", "dca-model-show",
        "dca-use-lots", "dca-sc-enable", "dca-sc-rollover",
        "dca-sc-tax",  # layout=int% (33), defaults=fraction (0.33) — unit conversion in callback
        "ret-yr-range", "ret-qs", "ret-toggles", "ret-model-show", "ret-use-lots",
        "sc-qs", "sc-toggles", "sc-model-show", "sc-use-lots",
        "sc-chart-layout", "sc-display-q",
        "cp-qs", "cp-toggles", "cp-use-lots", "cp-yr-range",
        "cp-model-src", "cp-asset-model", "cp-disp",
        "cp-scf-enable", "cp-legend-pos",
        "lot-date", "lot-notes",
    }

    # Also skip MC controls (dynamic, per-tab prefix)
    def is_mc_control(cid):
        return "-mc-" in cid

    mismatches = []

    def walk(component):
        cid = getattr(component, 'id', None)
        if cid and isinstance(cid, str) and not is_mc_control(cid) and cid not in skip_ids:
            val = getattr(component, 'value', '__MISSING__')
            if val != '__MISSING__':
                for prefix, defaults in prefix_map.items():
                    if cid.startswith(prefix):
                        # Derive key name
                        raw_key = cid[len(prefix):].replace("-", "_")
                        key = id_to_key.get(cid, raw_key)
                        if key in defaults:
                            expected = defaults[key]
                            if val != expected:
                                mismatches.append(
                                    f"{cid}: layout={val!r}, defaults={expected!r}")
                        break

        # Recurse into children
        children = getattr(component, 'children', None)
        if isinstance(children, (list, tuple)):
            for child in children:
                if child is not None:
                    walk(child)
        elif children is not None:
            walk(children)

    walk(app.layout)
    assert mismatches == [], "Layout/defaults mismatches:\n  " + "\n  ".join(mismatches)
```

**Note:** The `skip_ids` set is deliberately large — it covers dynamic defaults (year ranges, quantile selectors, toggles stored as checklists) and controls whose values depend on runtime state (live ticker). The test catches **scalar value= props** like amounts, rates, positions — the most common source of divergence. The skip set should be reviewed when adding new controls.

- [ ] **Step 4: Run drift detection test**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py::test_layout_values_match_defaults -v --timeout=120`
Expected: PASS (after all layout wiring is done).

If it fails, the mismatches output tells you exactly which component IDs have diverged and what the expected value should be. Fix the layout or defaults accordingly.

- [ ] **Step 5: Commit**

```bash
git add btc_web/test_defaults.py
git commit -m "test: add figure builder smoke tests + layout drift detection"
```

---

## Task 12: Run Full Test Suite + Fix Regressions

**Files:** (any file that needs fixing)

- [ ] **Step 1: Run full test suite**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py btc_web/test_defaults.py -v --timeout=180`
Expected: All tests pass (435+ existing + ~16 new).

- [ ] **Step 2: Fix any failures**

If any tests fail due to the refactoring:
- Check if the test was relying on a wrong default (e.g., testing that `pt_alpha` == 0.6). Update the test to use the correct value.
- Check if an import path is wrong.
- Check if a key name mismatch between layout and defaults (e.g., `pt_alpha` vs `ptalpha`).

**Known test_web.py updates needed:**
- Lines ~4179, ~4315, ~4432: `sc_term_months=48.0` → `sc_term_months=12` (prewarm was using wrong value; tests copied it)
- Any test asserting `pt_alpha=0.6` → update to `0.3`
- Any test asserting retire `inflation=0` → update to `4.0`

- [ ] **Step 3: Run the app locally to verify**

Run: `cd /scratch/code/bitcoinprojections && DEV=1 bash run_web.sh`
Manually verify:
- Each tab loads with correct default values
- Bubble chart shows correct alpha/legend position
- Citadel cash floor defaults to $50,000
- Heatmap exit year range extends to yr_now+15

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: resolve test regressions from defaults consolidation"
```

---

## Summary of Divergence Fixes by Task

| Divergence | Fixed In |
|-----------|----------|
| #1 Bubble `pt_alpha` (0.6→0.3) | Task 3 |
| #2 Bubble `show_sup` (False→True) | Task 9 |
| #3 Bubble `legend_pos` ("outside"→"top-left") | Task 3 + Task 9 |
| #4 Heatmap `show_colorbar` (False→True) | Task 9 |
| #5 Heatmap `exit_yr_hi` (+10→+15) | Task 9 |
| #6 Heatmap colors (model→forge) | Task 9 |
| #7 DCA `annotate` (False→True) | Task 9 |
| #8 DCA `legend_pos` ("outside"→"bottom-right") | Task 5 + Task 9 |
| #9 DCA `sc_term_months` (48→12) | Task 5 + Task 9 |
| #10 Retire `inflation` (0%→4%) | Task 6 |
| #11 Retire `legend_pos` ("outside"→"bottom-right") | Task 6 + Task 9 |
| #12 Retire `yr_range` start (yr_now→2031) | Task 6 |
| #13 SC `freq` ("Annually"→"Monthly" in prewarm) | Task 9 |
| #14 SC `legend_pos` ("outside"→"top-left") | Task 7 + Task 9 |
| #15 SC `display_q` (Q50%→Q5%) | Task 7 |
| #16 Citadel `high_q_trigger` (80→95) | Task 8 |
| #17 Citadel `low_q_trigger` (20→5) | Task 8 |
| #18 Citadel `cash_floor` (0→50000) | Task 8 |
| #19 Bubble `n_future` (0→3 in callback) | Task 3 |
| #20 Bubble `xscale` ("linear"→"log" in callback) | Task 3 |
| #21 Citadel `show_legend` (False→True in prewarm) | Task 9 |
| #22 Citadel `log_y` (False→True in cache gen) | Task 8 |
| #23 Citadel `low_q_split_*` (callback/fig→layout values) | Task 8 |
