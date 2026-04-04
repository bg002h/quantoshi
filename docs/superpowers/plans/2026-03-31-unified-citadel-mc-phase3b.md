# Unified Citadel MC — Phase 3b Polish Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish the Quick Scenarios feature with auto-fill, confirmation dialog, stale indicator, and dynamic model/tax lookup.

**Architecture:** 4 tasks: (1) Auto-fill controls from preset, (2) Confirmation dialog, (3) Stale indicator, (4) Wire active model + tax status into lookup.

**Tech Stack:** Python 3.14, Dash 4.0.0, DBC 2.0.4

**Test command:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -x -q --tb=short`

**Full suite:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -5`

---

## File Structure

### Modified Files
| File | Change |
|------|--------|
| `btc_web/citadel_presets.py` | Add `preset_control_values()` mapping presets → control ID/value pairs |
| `btc_web/callbacks/citadel_scenarios.py` | Auto-fill callback, confirmation dialog logic, stale detection, model/tax lookup |
| `btc_web/layout/citadel.py` | Add confirmation modal + stale badge |
| `btc_web/test_web.py` | All new tests |

---

### Task 1: Add `preset_control_values()` + auto-fill callback

**Files:**
- Modify: `btc_web/citadel_presets.py`
- Modify: `btc_web/callbacks/citadel_scenarios.py`
- Test: `btc_web/test_web.py`

Adds a function that maps (wealth, regime, rules, start_year) to a dict of `{component_id: value}` pairs matching the Citadel control IDs. Then a callback that writes these values to the controls when a preset is selected.

- [ ] **Step 1: Write failing tests**

```python
class TestPresetControlValues:
    def test_preset_control_values_returns_dict(self):
        from citadel_presets import preset_control_values
        vals = preset_control_values("starter", "neutral", "no_rebal", 2035)
        assert isinstance(vals, dict)
        assert "cp-stack" in vals
        assert "cp-spend" in vals
        assert "cp-cash-init" in vals

    def test_preset_control_values_starter(self):
        from citadel_presets import preset_control_values
        vals = preset_control_values("starter", "neutral", "no_rebal", 2035)
        assert vals["cp-stack"] == 0.5
        assert vals["cp-spend"] == 5000
        assert vals["cp-cash-init"] == 50000  # 10% of 500k
        assert vals["cp-infl"] == 4.0
        assert vals["cp-spend-growth"] == 1.0

    def test_preset_control_values_bitcoin_bull_aggressive(self):
        from citadel_presets import preset_control_values
        vals = preset_control_values("bitcoin", "bull", "aggressive", 2028)
        assert vals["cp-stack"] == 12.5
        assert vals["cp-spend"] == 50000
        assert vals["cp-cash-floor"] == 100000
        assert vals["cp-yr-range"] == [2028, 2075]

    def test_preset_control_values_rules_no_rebal(self):
        from citadel_presets import preset_control_values
        vals = preset_control_values("starter", "neutral", "no_rebal", 2035)
        # no_rebal: triggers effectively disabled
        assert vals["cp-high-q-thresh"] == 99
        assert vals["cp-low-q-thresh"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestPresetControlValues -x -q --tb=short`

- [ ] **Step 3: Add `preset_control_values()` to `btc_web/citadel_presets.py`**

```python
def preset_control_values(wealth: str, regime: str, rules: str,
                          start_year: int) -> dict[str, object]:
    """Map preset selections to {component_id: value} for Citadel controls.

    Returns a dict where keys are Dash component IDs and values are
    what those controls should be set to.
    """
    wl = WEALTH_LEVELS[wealth]
    rs = RULE_SETS[rules]
    alloc = wl["allocation"]
    da = wl["dollar_assets"]

    return {
        # Assets
        "cp-stack": wl["btc"],
        "cp-cash-init": da * alloc["cash"] / 100,
        "cp-res-short-init": da * alloc["res_short"] / 100,
        "cp-res-med-init": da * alloc["res_med"] / 100,
        "cp-res-long-init": da * alloc["res_long"] / 100,
        "cp-inv-eq-init": da * alloc["inv_eq"] / 100,
        "cp-inv-bd-init": da * alloc["inv_bd"] / 100,
        # Spending
        "cp-spend": wl["monthly_spend"],
        "cp-infl": wl["inflation"],
        "cp-spend-growth": wl["spend_growth"],
        # Rules
        "cp-cash-floor": rs["cash_floor"],
        "cp-high-q-thresh": int(rs["high_q_trigger"] * 100),
        "cp-low-q-thresh": int(rs["low_q_trigger"] * 100),
        # Simulation
        "cp-yr-range": [start_year, END_YEAR],
        "cp-asset-model": "markov",
    }
```

- [ ] **Step 4: Add auto-fill callback to `citadel_scenarios.py`**

Add a callback that fires when `cp-scenario-active` changes (set by the band loading callback) and writes preset values to all controls:

```python
@callback(
    [Output("cp-stack", "value", allow_duplicate=True),
     Output("cp-cash-init", "value", allow_duplicate=True),
     Output("cp-res-short-init", "value", allow_duplicate=True),
     Output("cp-res-med-init", "value", allow_duplicate=True),
     Output("cp-res-long-init", "value", allow_duplicate=True),
     Output("cp-inv-eq-init", "value", allow_duplicate=True),
     Output("cp-inv-bd-init", "value", allow_duplicate=True),
     Output("cp-spend", "value", allow_duplicate=True),
     Output("cp-infl", "value", allow_duplicate=True),
     Output("cp-spend-growth", "value", allow_duplicate=True),
     Output("cp-cash-floor", "value", allow_duplicate=True),
     Output("cp-high-q-thresh", "value", allow_duplicate=True),
     Output("cp-low-q-thresh", "value", allow_duplicate=True),
     Output("cp-yr-range", "value", allow_duplicate=True),
     Output("cp-asset-model", "value", allow_duplicate=True)],
    Input("cp-scenario-active", "data"),
    State("cp-scenario-wealth", "data"),
    State("cp-scenario-regime", "data"),
    State("cp-scenario-rules", "data"),
    State("cp-scenario-start-yr", "value"),
    prevent_initial_call=True,
)
def auto_fill_controls(active_key, wealth, regime, rules, start_yr):
    """Fill Citadel controls with preset values when a scenario loads."""
    if not active_key or not wealth:
        return [no_update] * 15
    from citadel_presets import preset_control_values
    vals = preset_control_values(wealth, regime, rules, int(start_yr))
    return [
        vals["cp-stack"], vals["cp-cash-init"],
        vals["cp-res-short-init"], vals["cp-res-med-init"], vals["cp-res-long-init"],
        vals["cp-inv-eq-init"], vals["cp-inv-bd-init"],
        vals["cp-spend"], vals["cp-infl"], vals["cp-spend-growth"],
        vals["cp-cash-floor"],
        vals["cp-high-q-thresh"], vals["cp-low-q-thresh"],
        vals["cp-yr-range"], vals["cp-asset-model"],
    ]
```

- [ ] **Step 5: Run test to verify it passes**

- [ ] **Step 6: Run full test suite**

- [ ] **Step 7: Commit**

```bash
git add btc_web/citadel_presets.py btc_web/callbacks/citadel_scenarios.py btc_web/test_web.py
git commit -m "feat(citadel): auto-fill controls from Quick Scenario presets"
```

---

### Task 2: Wire active model + tax status into band lookup

**Files:**
- Modify: `btc_web/callbacks/citadel_scenarios.py`
- Test: `btc_web/test_web.py`

Read the active BTC model from `cp-model-src` and filing status from `cp-tax-config` store.

- [ ] **Step 1: Write failing tests**

```python
class TestScenarioDynamicLookup:
    def test_snap_entry_q_values(self):
        from callbacks.citadel_scenarios import _snap_entry_q
        assert _snap_entry_q(0.01) == 1
        assert _snap_entry_q(0.10) == 10
        assert _snap_entry_q(0.50) == 50
```

- [ ] **Step 2: Modify `load_scenario_bands` in `citadel_scenarios.py`**

Add `State("cp-model-src", "value")` and `State("cp-tax-config", "data")` to the callback, then use them:

```python
@callback(
    Output("cp-scenario-bands", "data"),
    Output("cp-scenario-active", "data"),
    Input("cp-scenario-wealth", "data"),
    Input("cp-scenario-regime", "data"),
    Input("cp-scenario-rules", "data"),
    Input("cp-scenario-start-yr", "value"),
    State("cp-qs", "value"),
    State("cp-model-src", "value"),
    State("cp-tax-config", "data"),
    prevent_initial_call=True,
)
def load_scenario_bands(wealth, regime, rules, start_yr, quantile,
                        model_src, tax_config):
    """Look up pre-computed bands for the selected scenario."""
    if not all([wealth, regime, rules, start_yr]):
        return no_update, no_update

    from citadel_band_cache import lookup_entry

    # Use active model, fallback to "bub"
    model_key = model_src or "bub"
    entry_q = _snap_entry_q(float(quantile or 0.25))

    # Use filing status from tax config, fallback to "single"
    tax_status = "single"
    if isinstance(tax_config, dict):
        tax_status = tax_config.get("filing_status", "single")

    bands = lookup_entry(model_key, entry_q, regime, wealth, rules,
                         int(start_yr), tax_status)
    if bands is None:
        return None, None

    serialized = {}
    for pct, series_dict in bands.items():
        serialized[str(pct)] = {k: v.tolist() for k, v in series_dict.items()}

    active_key = f"{model_key}_q{entry_q}_{regime}_{wealth}_{rules}_{start_yr}_{tax_status}"
    return serialized, active_key
```

- [ ] **Step 3: Run full test suite**

- [ ] **Step 4: Commit**

```bash
git add btc_web/callbacks/citadel_scenarios.py btc_web/test_web.py
git commit -m "feat(citadel): wire active model + tax status into band lookup"
```

---

### Task 3: Stale indicator

**Files:**
- Modify: `btc_web/layout/citadel.py`
- Modify: `btc_web/callbacks/citadel_scenarios.py`
- Test: `btc_web/test_web.py`

Add a small "Stale" badge next to the Quick Scenarios header that appears when the user modifies any preset-controlled input after loading bands.

- [ ] **Step 1: Add stale badge to layout**

In `btc_web/layout/citadel.py`, modify the Quick Scenarios header:

```python
html.Div([
    html.Span("Quick Scenarios (Free)"),
    html.Span(" — stale", id="cp-scenario-stale",
              className="text-warning small",
              style={"display": "none"}),
], className="ctrl-section-header"),
```

- [ ] **Step 2: Add stale detection callback**

In `citadel_scenarios.py`, add a callback that compares current control values against preset values:

```python
@callback(
    Output("cp-scenario-stale", "style"),
    Input("cp-stack", "value"),
    Input("cp-spend", "value"),
    Input("cp-cash-init", "value"),
    Input("cp-infl", "value"),
    Input("cp-cash-floor", "value"),
    State("cp-scenario-active", "data"),
    State("cp-scenario-wealth", "data"),
    State("cp-scenario-regime", "data"),
    State("cp-scenario-rules", "data"),
    State("cp-scenario-start-yr", "value"),
    prevent_initial_call=True,
)
def detect_stale(stack, spend, cash_init, infl, cash_floor,
                 active_key, wealth, regime, rules, start_yr):
    """Show stale indicator when controls differ from loaded preset."""
    if not active_key or not wealth:
        return {"display": "none"}
    from citadel_presets import preset_control_values
    vals = preset_control_values(wealth, regime, rules, int(start_yr or 2035))
    # Check key controls
    is_stale = (
        float(stack or 0) != vals["cp-stack"] or
        float(spend or 0) != vals["cp-spend"] or
        float(cash_init or 0) != vals["cp-cash-init"] or
        float(infl or 0) != vals["cp-infl"] or
        float(cash_floor or 0) != vals["cp-cash-floor"]
    )
    return {} if is_stale else {"display": "none"}
```

- [ ] **Step 3: Run full test suite**

- [ ] **Step 4: Commit**

```bash
git add btc_web/layout/citadel.py btc_web/callbacks/citadel_scenarios.py btc_web/test_web.py
git commit -m "feat(citadel): add stale indicator for Quick Scenarios"
```

---

### Task 4: Confirmation dialog (deferred)

The confirmation dialog (spec lines 211-216) compares current values against the preset, shows a list of changes, and requires Apply/Cancel. This is complex UI with a multi-step flow:
1. User clicks preset pill → check if values differ
2. If differ → show modal with change list
3. Apply → fill controls + load bands
4. Cancel → revert pill selection

This requires reworking the pill click → band load flow into a two-step process with a modal gate. **Recommended: defer to a future session** when the core auto-fill + stale indicator are proven in production.

---

## Verification Checklist

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -10
```
