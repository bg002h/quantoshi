# Unified Citadel MC — Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Quick Scenarios UI panel to the Citadel Planner tab, enabling users to select preset wealth/regime/rule combos and instantly display pre-computed percentile band overlays on the chart.

**Architecture:** 5 tasks: (1) Quick Scenarios layout panel with pill buttons + dropdown, (2) Preset apply callback that fills controls and loads cached bands, (3) Band trace rendering in the Citadel figure builder, (4) Snapshot/routing registration for new controls, (5) Integration tests. Bands load from `citadel_band_cache.lookup_entry()` (Phase 2) and render as shaded `go.Scatter` fills.

**Tech Stack:** Python 3.14, Dash 4.0.0, DBC 2.0.4, Plotly

**Spec:** `docs/superpowers/specs/2026-03-31-unified-citadel-mc-design.md` (UI Design section)

**Test command:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -x -q --tb=short`

**Full suite:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -5`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `btc_web/callbacks/citadel_scenarios.py` | Quick Scenarios callback: reads pill/dropdown state, looks up cached bands, writes to band store |

### Modified Files
| File | Change |
|------|--------|
| `btc_web/layout/citadel.py` | Add Quick Scenarios panel above `dbc.Tabs` in `_citadel_controls()` |
| `btc_web/figures/citadel.py` | Add band trace rendering from `cp-scenario-bands` store |
| `btc_web/callbacks/citadel_cb.py` | Read `cp-scenario-bands` store as State, pass to figure builder |
| `btc_web/snapshot.py` | Register new scenario controls in `_SNAPSHOT_CONTROLS` |
| `btc_web/callbacks/routing.py` | Register new controls in `_TAB_CONTROLS["citadel"]` |
| `btc_web/test_web.py` | All new tests |

---

### Task 1: Quick Scenarios layout panel

**Files:**
- Modify: `btc_web/layout/citadel.py`
- Test: `btc_web/test_web.py`

Adds a collapsible "Quick Scenarios" card above the Citadel sub-tabs with three rows of pill buttons (Wealth, Regime, Rules) + a start year dropdown + stores.

- [ ] **Step 1: Write failing tests**

```python
class TestCitadelQuickScenariosLayout:
    def test_scenario_stores_exist(self):
        """Verify scenario-related stores are in the layout."""
        from layout.citadel import _citadel_controls
        import json
        layout = _citadel_controls()
        layout_str = json.dumps(layout.to_plotly_json())
        assert "cp-scenario-wealth" in layout_str
        assert "cp-scenario-regime" in layout_str
        assert "cp-scenario-rules" in layout_str
        assert "cp-scenario-start-yr" in layout_str
        assert "cp-scenario-bands" in layout_str

    def test_scenario_pill_buttons_exist(self):
        """Verify pill button IDs are present."""
        from layout.citadel import _citadel_controls
        import json
        layout = _citadel_controls()
        layout_str = json.dumps(layout.to_plotly_json())
        for wl in ["starter", "full", "bitcoin"]:
            assert f"cp-pill-{wl}" in layout_str
        for reg in ["bear", "neutral", "bull"]:
            assert f"cp-pill-{reg}" in layout_str
        for rs in ["no_rebal", "cautious", "aggressive"]:
            assert f"cp-pill-{rs}" in layout_str
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelQuickScenariosLayout -x -q --tb=short`
Expected: FAIL — stores don't exist yet

- [ ] **Step 3: Add Quick Scenarios panel to `_citadel_controls()`**

In `btc_web/layout/citadel.py`, add the following imports at the top (after existing imports):

```python
from citadel_presets import WEALTH_LEVELS, MACRO_REGIMES, RULE_SETS, START_YEARS
```

Then in `_citadel_controls()`, insert the Quick Scenarios panel **between** the `cp-load-overlay` div and the `dbc.Tabs(...)` block. Add the following code just before `dbc.Tabs([`:

```python
        # ── Quick Scenarios ──────────────────────────────────────────────
        _ctrl_card(
            html.Div("Quick Scenarios (Free)", className="ctrl-section-header"),
            # Wealth row
            html.Div([
                html.Small("Wealth", className="text-muted me-2",
                           style={"minWidth": "50px"}),
                dbc.ButtonGroup([
                    dbc.Button(WEALTH_LEVELS[k]["label"].split()[0], id=f"cp-pill-{k}",
                               outline=(k != "starter"), color="primary", size="sm")
                    for k in WEALTH_LEVELS
                ], size="sm"),
            ], className="d-flex align-items-center mb-1"),
            # Regime row
            html.Div([
                html.Small("Regime", className="text-muted me-2",
                           style={"minWidth": "50px"}),
                dbc.ButtonGroup([
                    dbc.Button(MACRO_REGIMES[k]["label"], id=f"cp-pill-{k}",
                               outline=(k != "neutral"), color="primary", size="sm")
                    for k in MACRO_REGIMES
                ], size="sm"),
            ], className="d-flex align-items-center mb-1"),
            # Rules row
            html.Div([
                html.Small("Rules", className="text-muted me-2",
                           style={"minWidth": "50px"}),
                dbc.ButtonGroup([
                    dbc.Button(RULE_SETS[k]["label"], id=f"cp-pill-{k}",
                               outline=(k != "no_rebal"), color="primary", size="sm")
                    for k in RULE_SETS
                ], size="sm"),
            ], className="d-flex align-items-center mb-1"),
            # Start year dropdown
            html.Div([
                html.Small("Start", className="text-muted me-2",
                           style={"minWidth": "50px"}),
                dcc.Dropdown(id="cp-scenario-start-yr",
                    options=[{"label": f"{'**' if y in START_YEARS else ''}{y}",
                              "value": y}
                             for y in range(2025, 2041)],
                    value=2035, clearable=False,
                    style={"width": "100px", "fontSize": "13px"}),
            ], className="d-flex align-items-center mb-1"),
            html.Small("800 simulations per scenario",
                       style={"color": "#888", "fontSize": "10px",
                              "display": "block", "marginTop": "4px"}),
        ),
        # Scenario stores
        dcc.Store(id="cp-scenario-wealth", data="starter"),
        dcc.Store(id="cp-scenario-regime", data="neutral"),
        dcc.Store(id="cp-scenario-rules", data="no_rebal"),
        dcc.Store(id="cp-scenario-bands", storage_type="memory"),
        dcc.Store(id="cp-scenario-active", data=None),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelQuickScenariosLayout -x -q --tb=short`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -5`

- [ ] **Step 6: Commit**

```bash
git add btc_web/layout/citadel.py btc_web/test_web.py
git commit -m "feat(citadel): add Quick Scenarios panel layout with pill buttons + stores"
```

---

### Task 2: Preset apply callback

**Files:**
- Create: `btc_web/callbacks/citadel_scenarios.py`
- Test: `btc_web/test_web.py`

When a pill button is clicked, update the corresponding store and look up cached bands. The callback reads all three stores + start year dropdown, calls `citadel_band_cache.lookup_entry()`, and writes the result to `cp-scenario-bands`.

- [ ] **Step 1: Write failing tests**

```python
class TestCitadelScenarioCallback:
    def test_scenario_lookup_returns_bands_for_valid_combo(self, tmp_path):
        """Verify lookup returns bands when cache exists."""
        import numpy as np
        from citadel_band_cache import store_entry, lookup_entry
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 480
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.ones(n_periods, dtype=np.float32) * pct
                          for s in BAND_SERIES}
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)
        result = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)
        assert result is not None
        assert 50 in result
        assert "total" in result[50]
        assert len(result[50]["total"]) == n_periods

    def test_scenario_lookup_returns_none_for_missing(self, tmp_path):
        from citadel_band_cache import lookup_entry
        result = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2099, "single", cache_dir=tmp_path)
        assert result is None
```

- [ ] **Step 2: Run tests — should PASS (uses existing cache module)**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelScenarioCallback -x -q --tb=short`

- [ ] **Step 3: Create `btc_web/callbacks/citadel_scenarios.py`**

```python
"""Citadel Quick Scenarios — pill button callbacks and band loading."""
from __future__ import annotations

from dash import callback, no_update, Input, Output, State
import _app_ctx

from citadel_presets import WEALTH_LEVELS, MACRO_REGIMES, RULE_SETS, START_YEARS


# ── Pill button click → update store + toggle outline ────────────────────────

def _pill_click_cb(group_name, keys):
    """Register a callback that updates a store and toggles pill outlines."""
    outputs = [Output(f"cp-scenario-{group_name}", "data")]
    outputs += [Output(f"cp-pill-{k}", "outline") for k in keys]
    inputs = [Input(f"cp-pill-{k}", "n_clicks") for k in keys]

    @callback(outputs, inputs, prevent_initial_call=True)
    def _on_pill_click(*n_clicks_args):
        from dash import ctx
        triggered = ctx.triggered_id
        if not triggered:
            return [no_update] * (1 + len(keys))
        # Extract the key from "cp-pill-{key}"
        selected = triggered.replace("cp-pill-", "")
        outlines = [k != selected for k in keys]
        return [selected] + outlines

    return _on_pill_click


_pill_click_cb("wealth", list(WEALTH_LEVELS.keys()))
_pill_click_cb("regime", list(MACRO_REGIMES.keys()))
_pill_click_cb("rules", list(RULE_SETS.keys()))


# ── Load cached bands when any scenario dimension changes ────────────────────

@callback(
    Output("cp-scenario-bands", "data"),
    Output("cp-scenario-active", "data"),
    Input("cp-scenario-wealth", "data"),
    Input("cp-scenario-regime", "data"),
    Input("cp-scenario-rules", "data"),
    Input("cp-scenario-start-yr", "value"),
    State("cp-qs", "value"),
    prevent_initial_call=True,
)
def load_scenario_bands(wealth, regime, rules, start_yr, quantile):
    """Look up pre-computed bands for the selected scenario."""
    if not all([wealth, regime, rules, start_yr]):
        return no_update, no_update

    from citadel_band_cache import lookup_entry
    from citadel_presets import BTC_MODELS

    # Determine which BTC model is active
    model_key = "bub"  # default

    # Determine entry quantile bin (snap to cached bins: 1, 10, 50)
    q_pct = int(round(float(quantile or 0.25) * 100))
    entry_q = min([1, 10, 50], key=lambda b: abs(q_pct - b))

    # Try to find cached bands
    bands = lookup_entry(model_key, entry_q, regime, wealth, rules,
                         int(start_yr), "single")
    if bands is None:
        return None, None

    # Serialize bands for dcc.Store (convert numpy arrays to lists)
    serialized = {}
    for pct, series_dict in bands.items():
        serialized[str(pct)] = {k: v.tolist() for k, v in series_dict.items()}

    active_key = f"{model_key}_q{entry_q}_{regime}_{wealth}_{rules}_{start_yr}_single"
    return serialized, active_key
```

- [ ] **Step 4: Register the callback module**

In `btc_web/callbacks/__init__.py` (or wherever callbacks are imported), add:

```python
from callbacks import citadel_scenarios  # noqa: F401
```

Check how other callback modules are registered — likely in `app.py` or `callbacks/__init__.py`.

- [ ] **Step 5: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -5`

- [ ] **Step 6: Commit**

```bash
git add btc_web/callbacks/citadel_scenarios.py btc_web/test_web.py
git commit -m "feat(citadel): add Quick Scenarios callbacks for pill buttons + band loading"
```

---

### Task 3: Band trace rendering in Citadel figure builder

**Files:**
- Modify: `btc_web/figures/citadel.py`
- Modify: `btc_web/callbacks/citadel_cb.py`
- Test: `btc_web/test_web.py`

Adds band rendering (P25-P75 dark fill, P5-P95 light fill) to the Citadel chart when `cp-scenario-bands` store has data.

- [ ] **Step 1: Write failing tests**

```python
class TestCitadelBandRendering:
    def test_build_band_traces_returns_traces(self):
        """Verify band trace builder produces scatter traces."""
        import numpy as np
        from figures.citadel import _build_band_traces
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 24
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.linspace(1000, 2000, n_periods).tolist()
                          for s in BAND_SERIES}
        time_axis = np.linspace(22, 24, n_periods).tolist()
        traces = _build_band_traces(bands, time_axis, series_key="total",
                                     color="#000000")
        assert len(traces) == 4  # P5-P95 lower/upper + P25-P75 lower/upper
        # All should be go.Scatter
        import plotly.graph_objects as go
        for t in traces:
            assert isinstance(t, go.Scatter)

    def test_build_band_traces_empty_bands(self):
        from figures.citadel import _build_band_traces
        traces = _build_band_traces(None, [], series_key="total", color="#000")
        assert traces == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelBandRendering -x -q --tb=short`
Expected: FAIL — `_build_band_traces` doesn't exist

- [ ] **Step 3: Add `_build_band_traces()` to `btc_web/figures/citadel.py`**

Add this function near the top of the file (after imports):

```python
def _build_band_traces(bands, time_axis, series_key="total", color="#000000",
                       name_prefix="MC spread"):
    """Build shaded band traces from percentile band data.

    Returns 4 traces: P5-P95 (light fill) lower/upper + P25-P75 (dark fill) lower/upper.
    """
    if not bands or not time_axis:
        return []

    import plotly.graph_objects as go

    def _hex_alpha(hex_color, alpha):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"

    traces = []
    # Keys may be string (from JSON store) or int
    def _get(pct):
        return bands.get(pct) or bands.get(str(pct)) or {}

    p5 = _get(5).get(series_key, [])
    p25 = _get(25).get(series_key, [])
    p75 = _get(75).get(series_key, [])
    p95 = _get(95).get(series_key, [])

    if not p5 or not p95:
        return []

    x = list(time_axis)

    # P5-P95 band (light fill, opacity 0.15)
    traces.append(go.Scatter(
        x=x, y=list(p5), mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    traces.append(go.Scatter(
        x=x, y=list(p95), mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=_hex_alpha(color, 0.15),
        name=f"{name_prefix} (P5\u2013P95)",
        legendgroup="mc-bands",
    ))
    # P25-P75 band (dark fill, opacity 0.30)
    traces.append(go.Scatter(
        x=x, y=list(p25), mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    traces.append(go.Scatter(
        x=x, y=list(p75), mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=_hex_alpha(color, 0.30),
        name=f"{name_prefix} (P25\u2013P75)",
        legendgroup="mc-bands",
    ))

    return traces
```

- [ ] **Step 4: Wire bands into `build_citadel_figure()`**

In `btc_web/figures/citadel.py`, in `build_citadel_figure()`, after the deterministic traces are built but before the MC overlay section, add:

```python
    # Quick Scenario bands from cached presets
    scenario_bands = p.get("scenario_bands")
    if scenario_bands:
        _band_series = "total" if disp_mode == "usd_total" else "btc_stack" if disp_mode == "btc" else "total"
        _band_color = _C_TOTAL if disp_mode != "btc" else _C_BTC
        band_traces = _build_band_traces(
            scenario_bands, time_axis.tolist(),
            series_key=_band_series, color=_band_color)
        traces.extend(band_traces)
```

Where `time_axis` is the ndarray already computed in the function. Find the right variable name by reading the existing code.

- [ ] **Step 5: Add `scenario_bands` to params in `citadel_cb.py`**

In `btc_web/callbacks/citadel_cb.py`, in `update_citadel()`:
1. Add `State("cp-scenario-bands", "data")` to the callback's State list
2. Add `scenario_bands=scenario_bands_data` to the params dict `p` that gets passed to the figure builder

- [ ] **Step 6: Run test to verify it passes**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelBandRendering -x -q --tb=short`
Expected: PASS

- [ ] **Step 7: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -5`

- [ ] **Step 8: Commit**

```bash
git add btc_web/figures/citadel.py btc_web/callbacks/citadel_cb.py btc_web/test_web.py
git commit -m "feat(citadel): add band trace rendering from cached Quick Scenarios"
```

---

### Task 4: Snapshot and routing registration

**Files:**
- Modify: `btc_web/snapshot.py`
- Modify: `btc_web/callbacks/routing.py`
- Test: `btc_web/test_web.py`

Register the 5 new scenario controls so they're included in share links and tab routing.

- [ ] **Step 1: Write failing tests**

```python
class TestCitadelScenarioSnapshot:
    def test_scenario_controls_in_snapshot(self):
        from snapshot import _SNAPSHOT_CONTROLS
        ids = {c[0] for c in _SNAPSHOT_CONTROLS}
        assert "cp-scenario-wealth" in ids
        assert "cp-scenario-regime" in ids
        assert "cp-scenario-rules" in ids
        assert "cp-scenario-start-yr" in ids

    def test_scenario_controls_in_tab_controls(self):
        from callbacks.routing import _TAB_CONTROLS
        citadel_ids = _TAB_CONTROLS["citadel"]
        assert "cp-scenario-wealth" in citadel_ids
        assert "cp-scenario-regime" in citadel_ids
        assert "cp-scenario-rules" in citadel_ids
        assert "cp-scenario-start-yr" in citadel_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelScenarioSnapshot -x -q --tb=short`

- [ ] **Step 3: Add to `_SNAPSHOT_CONTROLS` in `btc_web/snapshot.py`**

Find the citadel section of `_SNAPSHOT_CONTROLS` (around line 168). Add at the end of the citadel controls:

```python
    ("cp-scenario-wealth",    "data"),
    ("cp-scenario-regime",    "data"),
    ("cp-scenario-rules",     "data"),
    ("cp-scenario-start-yr",  "value"),
```

- [ ] **Step 4: Add to `_TAB_CONTROLS["citadel"]` in `btc_web/callbacks/routing.py`**

Find the citadel section where `_TAB_CONTROLS["citadel"]` is built. Add the new IDs to the set:

```python
    "cp-scenario-wealth", "cp-scenario-regime", "cp-scenario-rules",
    "cp-scenario-start-yr",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelScenarioSnapshot -x -q --tb=short`

- [ ] **Step 6: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -5`

- [ ] **Step 7: Commit**

```bash
git add btc_web/snapshot.py btc_web/callbacks/routing.py btc_web/test_web.py
git commit -m "feat(citadel): register Quick Scenario controls in snapshot + routing"
```

---

### Task 5: Integration tests

**Files:**
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write integration tests**

```python
class TestCitadelQuickScenariosIntegration:
    def test_full_scenario_pipeline(self, tmp_path):
        """End-to-end: store bands → lookup → build traces."""
        import numpy as np
        from citadel_band_cache import store_entry, lookup_entry
        from figures.citadel import _build_band_traces
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES

        n_periods = 24
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.linspace(100 * pct, 200 * pct, n_periods).astype(np.float32)
                          for s in BAND_SERIES}
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)

        loaded = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)
        assert loaded is not None

        # Serialize like the callback does
        serialized = {}
        for pct, series_dict in loaded.items():
            serialized[str(pct)] = {k: v.tolist() for k, v in series_dict.items()}

        time_axis = np.linspace(26, 28, n_periods).tolist()
        traces = _build_band_traces(serialized, time_axis,
                                     series_key="total", color="#000000")
        assert len(traces) == 4
        # P5 lower bound should be less than P95 upper bound
        assert traces[0].y[0] < traces[1].y[0]

    def test_all_preset_combos_produce_valid_configs(self):
        """Every preset combo builds a valid SimConfig."""
        from citadel_presets import (WEALTH_LEVELS, MACRO_REGIMES, RULE_SETS,
                                     START_YEARS, TAX_STATUSES, build_config)
        from engines.citadel_sim import validate_config
        for wealth in WEALTH_LEVELS:
            for regime in MACRO_REGIMES:
                for rules in RULE_SETS:
                    cfg = build_config(wealth, regime, rules, 2035, "single")
                    validate_config(cfg)  # raises on invalid
```

- [ ] **Step 2: Run integration tests**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelQuickScenariosIntegration -x -q --tb=short`

- [ ] **Step 3: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -5`

- [ ] **Step 4: Commit**

```bash
git add btc_web/test_web.py
git commit -m "test(citadel): integration tests for Quick Scenarios pipeline"
```

---

## Verification Checklist

After all 5 tasks, run:

```bash
# Full test suite
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -10

# Import check
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "
from layout.citadel import _citadel_controls
from callbacks.citadel_scenarios import load_scenario_bands
from figures.citadel import _build_band_traces
from citadel_band_cache import lookup_entry
from citadel_presets import WEALTH_LEVELS, MACRO_REGIMES, RULE_SETS
print(f'Wealth levels: {list(WEALTH_LEVELS.keys())}')
print(f'Regimes: {list(MACRO_REGIMES.keys())}')
print(f'Rules: {list(RULE_SETS.keys())}')
print('Phase 3 OK')
"

# Syntax check
cd btc_web && PYTHONPATH=".:../archive/btc_app" ../btc_venv/bin/python3 -c \
  "import layout, figures, callbacks, cache, engines.adapter, engines.citadel, engines.tax, engines.tax_lots, engines.tax_data, data.asset_matrices; print('OK')"
```
