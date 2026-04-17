# Default Update Sanity — Design Spec (v2, post-review)

**Date:** 2026-03-27
**Branch:** `DefaultUpdateSanity`
**Goal:** Eliminate default value divergences across the app by consolidating into a single source of truth per tab, with immutability protection and tests that catch drift.

---

## Problem

Default values for each tab are independently hardcoded in up to 6 locations:
1. Layout files (`value=` props)
2. Callback files (`_cf(val, DEFAULT)` fallbacks)
3. Figure builders (`p.get("key", DEFAULT)` fallbacks)
4. Prewarm function (`_prewarm_caches()` in `app.py`)
5. Citadel cache generator (`generate_citadel_cache.py`)
6. Engine dataclass defaults (`engines/citadel.py` SimConfig field defaults)

**~18 divergences found** across 7 chart tabs (Bubble, Heatmap, DCA, Retire, Supercharger, Citadel, plus minor Stack Tracker):
- **~10 cause cache misses** on every default page load (prewarm caches wrong params)
- **~6 cause wrong fallback behavior** when user clears a field
- **~2 are internal inconsistencies** between prewarm and cache generator

### Confirmed divergences (verified against code)

| # | Tab | Parameter | Layout | Callback fallback | Prewarm | Figure builder | Impact |
|---|-----|-----------|--------|-------------------|---------|----------------|--------|
| 1 | Bubble | `pt_alpha` | 0.3 | 0.6 | 0.3 | 0.6 | Wrong fallback |
| 2 | Bubble | `show_sup` | True | — | False | — | Cache miss |
| 3 | Bubble | `legend_pos` | "top-left" | "outside" | "outside" | — | Cache miss |
| 4 | Heatmap | `show_colorbar` | True | — | False | — | Cache miss |
| 5 | Heatmap | `exit_yr_hi` | yr_now+15 | — | yr_now+10 | — | Cache miss |
| 6 | Heatmap | colors | forge palette | M.CAGR_SEG_* | M.CAGR_SEG_* | — | Cache miss |
| 7 | DCA | `annotate` | True | — | False | — | Cache miss |
| 8 | DCA | `legend_pos` | "bottom-right" | "outside" | "outside" | — | Cache miss |
| 9 | DCA | `sc_term_months` | 12 | 12 | 48 | 12 | Cache miss |
| 10 | Retire | `inflation` | 4% | 0% | 4% | 0% | Wrong fallback |
| 11 | Retire | `legend_pos` | "bottom-right" | "outside" | "outside" | — | Cache miss |
| 12 | Retire | `yr_range` | [2031,2075] | [2025,2045] | [2031,2075] | — | Wrong fallback |
| 13 | SC | `freq` | "Monthly" | "Monthly" | "Annually" | "Monthly" | Cache miss |
| 14 | SC | `legend_pos` | "top-left" | "outside" | "outside" | — | Cache miss |
| 15 | SC | `display_q` | Q5% | Q50% | Q5% | Q50% | Wrong fallback |
| 16 | Citadel | `high_q_trigger` | 95 | 95 | 80 | 80 | Cache miss + wrong fallback |
| 17 | Citadel | `low_q_trigger` | 5 | 5 | 20 | 20 | Cache miss + wrong fallback |
| 18 | Citadel | `cash_floor` | 50000 | 0 | 0 | 0 | Cache miss + wrong fallback |

---

## Solution

### New file: `btc_web/tab_defaults.py`

One `MappingProxyType` dict per tab. `MappingProxyType` is a read-only dict view — `BUBBLE["x"] = 1` raises `TypeError`. Works with `.get()`, `**unpacking`, `dict(DEFAULTS, **overrides)`, iteration.

**Inner values use tuples, not lists** to prevent mutation of nested structures. `BUBBLE["selected_qs"]` is `()` not `[]`. Callers that need a list do `list(BUBBLE["selected_qs"])`.

```python
from types import MappingProxyType

BUBBLE = MappingProxyType({
    "selected_qs": (),           # tuple, not list
    "shade": True,
    "show_data": True,
    "show_today": True,
    "show_legend": False,
    "show_ols": False,
    "show_comp": True,
    "show_sup": True,
    "xscale": "log",
    "yscale": "log",
    "n_future": 3,
    "pt_size": 3,
    "pt_alpha": 0.3,
    "stack": 0,
    "legend_pos": "top-left",
    "palette": "default",
    ...
})
```

**Dynamic defaults** (year ranges, current year) resolved at call time:
```python
def bubble_defaults() -> dict:
    """Returns a mutable dict with dynamic values resolved."""
    yr_now = pd.Timestamp.today().year
    d = dict(BUBBLE)  # mutable copy from frozen source
    d["xrange"] = [2012, yr_now + 4]
    d["selected_qs"] = list(BUBBLE["selected_qs"])  # tuple → list
    return d
```

**IMPORTANT constraints (document in module docstring):**
- Never pass a `MappingProxyType` directly to `json.dumps()` — it raises `TypeError`. Always use `dict(DEFAULTS)` or the `_defaults()` functions which return regular dicts.
- Never pass a `MappingProxyType` directly to a figure builder. Use the `_defaults()` function or `dict(DEFAULTS, **overrides)`.
- All inner collection values must be tuples/frozensets, not lists/sets.
- Dynamic values (`_ALL_QS`, `yr_now`) must NOT be resolved at import time — only inside `_defaults()` functions called at runtime.

### Engine dataclass alignment

`engines/citadel.py` `SimConfig` field defaults (e.g., `cash_floor: float = 0.0`) serve as the engine's "unset sentinel" — they're what you get if a field isn't explicitly passed. These should match `CITADEL` defaults where possible. Where they differ intentionally (e.g., engine default `n_sims=1` is a runtime sentinel, not a UI default), document why.

### Consumer changes

**Layout:**
```python
from tab_defaults import BUBBLE
# Before: dbc.Input(id="bub-ptalpha", value=0.3)
# After:  dbc.Input(id="bub-ptalpha", value=BUBBLE["pt_alpha"])
```

**Callback fallback:**
```python
from tab_defaults import BUBBLE
# Before: _cf(ptalpha, 0.6)  ← WRONG
# After:  _cf(ptalpha, BUBBLE["pt_alpha"])
```

**Figure builder fallback:**
```python
from tab_defaults import BUBBLE
# Before: p.get("pt_alpha", 0.6)  ← WRONG
# After:  p.get("pt_alpha", BUBBLE["pt_alpha"])
```

**Prewarm:**
```python
from tab_defaults import bubble_defaults
# Before: _get_bubble_fig(dict(selected_qs=[], shade=True, ...30 lines...))
# After:  _get_bubble_fig(bubble_defaults())
```

---

## Tests

### 1. Runtime smoke: figure builders accept their defaults (6 tests)

```python
def test_bubble_defaults_produce_valid_figure():
    fig = build_bubble_figure(M, bubble_defaults())
    assert len(fig.data) > 0
```

One per chart tab. Catches missing keys, wrong types.

### 2. Immutability: MappingProxyType prevents mutation (1 test)

```python
def test_defaults_are_immutable():
    for defaults in [BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE, CITADEL]:
        with pytest.raises(TypeError):
            defaults["new_key"] = "bad"
```

### 3. Inner values are tuples, not lists (1 test)

```python
def test_inner_collections_are_tuples():
    for defaults in [BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE, CITADEL]:
        for key, val in defaults.items():
            assert not isinstance(val, list), f"{key} is a list, should be tuple"
            assert not isinstance(val, set), f"{key} is a set, should be frozenset"
```

### 4. Drift detection: layout values match defaults (real implementation)

```python
def test_layout_values_match_defaults():
    """Walk the layout component tree, extract value= props for known
    component IDs, compare to TAB_DEFAULTS."""
    import os; os.environ['TESTING'] = '1'
    from btc_web.app import app
    from dash import dcc
    import dash_bootstrap_components as dbc

    # Map component ID prefix → defaults dict
    prefix_map = {"bub-": BUBBLE, "hm-": HEATMAP, "dca-": DCA,
                  "ret-": RETIRE, "sc-": SUPERCHARGE, "cp-": CITADEL}

    # Walk layout tree, find components, check values
    mismatches = []
    def walk(component):
        if hasattr(component, 'id') and hasattr(component, 'value'):
            cid = component.id
            for prefix, defaults in prefix_map.items():
                if cid.startswith(prefix):
                    # Extract the param name from the component ID
                    param = cid[len(prefix):]
                    if param in defaults and component.value != defaults[param]:
                        mismatches.append((cid, component.value, defaults[param]))
        if hasattr(component, 'children'):
            children = component.children
            if isinstance(children, list):
                for child in children:
                    walk(child)
            elif children is not None:
                walk(children)

    walk(app.layout)
    assert mismatches == [], f"Layout/defaults mismatches: {mismatches}"
```

This is the key test that prevents future drift. It programmatically checks the actual rendered layout against `TAB_DEFAULTS`.

### 5. App startup: catches import-time errors (existing)

```python
def test_app_imports_cleanly():
    from btc_web.app import app
    assert app is not None
```

---

## Implementation Phases

### Phase 0: Automated audit script

Write a script that extracts every `value=` prop from layout components and every `_cf()`/`_ci()` fallback from callbacks, then diffs them. Establishes the ground truth before refactoring.

### Phase 1: Create `tab_defaults.py`

Write canonical defaults for all 7 chart tabs. **Layout values are the source of truth** (that's what the user sees). Use tuples for inner collections. Add `_defaults()` functions for dynamic values.

### Phase 2: Wire layout builders

Replace hardcoded `value=` props with `DEFAULTS["key"]` references. One tab at a time. Test after each tab.

### Phase 3: Wire callback fallbacks

Replace `_cf(val, HARDCODED)` and `p.get("key", HARDCODED)` with `DEFAULTS["key"]`. Fixes wrong-fallback bugs.

### Phase 4: Wire prewarm + cache generator

Replace 150+ lines of hardcoded prewarm params with `_defaults()` calls. Fixes all cache miss bugs. Wire Citadel cache generator similarly.

### Phase 5: Align engine defaults

Review `SimConfig` field defaults against `CITADEL` defaults. Align where appropriate, document intentional differences.

### Phase 6: Tests

Add all 5 test categories. The drift detection test (test 4) is the most important — it prevents the problem from recurring.

---

## Not In Scope

- Changing any actual default values (this refactor preserves current layout defaults exactly)
- Adding new parameters
- Changing the params dict pattern used by figure builders
- Static type checking / mypy integration
- Refactoring how `_cf()`/`_ci()` work (only changing what values they reference)
