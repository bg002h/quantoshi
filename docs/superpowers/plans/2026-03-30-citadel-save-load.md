# Citadel Save/Load Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add save/load for Citadel Planner scenarios — controls, simulation results, and Plotly figure for instant interactive restore.

**Architecture:** Save button collects controls + sim results + figure via server callback, clientside JS triggers download. Load button (dcc.Upload) parses JSON, restores controls via multi-output callback, injects figure directly into graph. Follows existing MC upload pattern. Also fixes 4 missing snapshot controls.

**Tech Stack:** Dash 4, dbc, Plotly, clientside JS

**Spec:** `docs/superpowers/specs/2026-03-30-citadel-save-load-design.md`

**Test command:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --ignore=btc_web/test_tax_e2e.py --tb=short`

**Baseline:** 889 passed, 0 failed, 5 skipped

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `btc_web/snapshot.py` | Modify | Add 4 missing controls + checklist options |
| `btc_web/layout/citadel.py` | Modify | Add Save/Load buttons + dcc.Upload + dcc.Store |
| `btc_web/callbacks/citadel_save_cb.py` | Create | Save prep callback + load/restore callback + clientside download |
| `btc_web/callbacks/__init__.py` | Modify | Import citadel_save_cb |

---

### Task 1: Fix missing snapshot controls

**Files:**
- Modify: `btc_web/snapshot.py`

- [ ] **Step 1: Read snapshot.py to find where to add controls**

```bash
grep -n 'cp-high-q\|cp-low-q\|cp-inv-eq-basis\|cp-inv-bd-basis' btc_web/snapshot.py
```

Also read the `_SNAPSHOT_CONTROLS` list and `_TAB_CONTROLS["citadel"]` to find insertion points:

```bash
grep -n 'cp-high-q-thresh\|cp-low-q-thresh\|cp-inv-eq-rate\|cp-inv-bd-rate' btc_web/snapshot.py
```

- [ ] **Step 2: Add 4 missing controls to `_SNAPSHOT_CONTROLS`**

Add these tuples near their related controls:
- `("cp-high-q-enable", "value")` — near `cp-high-q-thresh`
- `("cp-low-q-enable", "value")` — near `cp-low-q-thresh`
- `("cp-inv-eq-basis", "value")` — near `cp-inv-eq-rate`
- `("cp-inv-bd-basis", "value")` — near `cp-inv-bd-rate`

- [ ] **Step 3: Add to `_TAB_CONTROLS["citadel"]`**

Add the 4 component IDs to the citadel set.

- [ ] **Step 4: Add checklist entries to `_CHECKLIST_OPTIONS`**

```python
"cp-high-q-enable": ["yes"],
"cp-low-q-enable": ["yes"],
```

(`cp-inv-eq-basis` and `cp-inv-bd-basis` are numbers, not checklists — no entry needed.)

- [ ] **Step 5: Run test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --ignore=btc_web/test_tax_e2e.py --tb=short 2>&1 | tail -5
```

Expected: 889 passed, 0 failed, 5 skipped.

- [ ] **Step 6: Commit**

```bash
git add btc_web/snapshot.py
git commit -m "fix: add 4 missing Citadel controls to snapshot system"
```

---

### Task 2: Add Save/Load UI to Citadel layout

**Files:**
- Modify: `btc_web/layout/citadel.py`

- [ ] **Step 1: Read the layout around the Run button**

```bash
grep -n -B5 -A15 'cp-run-btn' btc_web/layout/citadel.py
```

- [ ] **Step 2: Add Save/Load buttons and supporting components**

After the Run button (`cp-run-btn`), add:

```python
dbc.Row([
    dbc.Col(
        dbc.Button("↓ Save Scenario", id="cp-save-btn",
                   color="secondary", outline=True, size="sm",
                   className="w-100", disabled=True),
        width=6,
    ),
    dbc.Col(
        dcc.Upload(
            id="cp-scenario-upload",
            children=dbc.Button("↑ Load", color="secondary",
                                outline=True, size="sm",
                                className="w-100"),
            accept=".json",
        ),
        width=6,
    ),
], className="mb-2 gx-2"),
html.Div(id="cp-load-status", className="text-muted small mb-2"),
```

Also add these stores (can go at the top of the controls div or with existing stores):

```python
dcc.Store(id="cp-save-prep", storage_type="memory"),
```

- [ ] **Step 3: Enable Save button when results exist**

Add a clientside callback that enables the Save button when the citadel graph has data:

```python
app.clientside_callback(
    "function(fig) { return !(fig && fig.data && fig.data.length > 0); }",
    Output("cp-save-btn", "disabled"),
    Input("citadel-graph", "figure"),
)
```

This goes in the new `citadel_save_cb.py` (Task 3), but note the dependency: the `cp-save-btn` must exist in the layout first.

- [ ] **Step 4: Verify layout loads**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "
import os; os.environ['DEV'] = '1'
from layout.citadel import _citadel_controls
print('Layout OK')
"
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/layout/citadel.py
git commit -m "feat: add Save/Load buttons to Citadel Planner layout"
```

---

### Task 3: Create `citadel_save_cb.py` — save flow

**Files:**
- Create: `btc_web/callbacks/citadel_save_cb.py`
- Modify: `btc_web/callbacks/__init__.py`

The save flow is hybrid: server callback collects all data into `cp-save-prep`, clientside callback triggers the download.

- [ ] **Step 1: Read the existing MC clientside download for reference**

```bash
sed -n '72,103p' btc_web/callbacks/mc_upload.py
```

- [ ] **Step 2: Read the Citadel callback State inputs to know which controls to capture**

```bash
sed -n '105,205p' btc_web/callbacks/citadel_cb.py
```

- [ ] **Step 3: Create `citadel_save_cb.py` with the save prep callback**

The server callback reads all cp-* controls via State, plus simulation results, figure, and tax data. It builds the scenario dict and writes to `cp-save-prep`.

```python
# btc_web/callbacks/citadel_save_cb.py
"""Citadel Planner scenario save/load callbacks."""
from __future__ import annotations

import base64
import datetime
import json

import dash
from dash import callback, Input, Output, State, clientside_callback, no_update
import dash_bootstrap_components as dbc

import _app_ctx

# All cp-* controls from _SNAPSHOT_CONTROLS (citadel subset)
# Import the list to stay in sync
from snapshot import _SNAPSHOT_CONTROLS, _TAB_CONTROLS


def _citadel_control_ids():
    """Return list of (cid, prop) for all Citadel controls in snapshot system."""
    cp_ids = _TAB_CONTROLS.get("citadel", set())
    return [(cid, prop) for cid, prop in _SNAPSHOT_CONTROLS if cid in cp_ids]


_CP_CONTROLS = _citadel_control_ids()


def _abbreviate(val, suffix=""):
    """Abbreviate large numbers: 100000 -> '100k', 1000000 -> '1M'."""
    if val is None:
        return ""
    v = float(val)
    if v >= 1_000_000:
        return f"{v/1_000_000:.0f}M{suffix}"
    if v >= 1_000:
        return f"{v/1_000:.0f}k{suffix}"
    return f"{v:.0f}{suffix}"


def _build_filename(controls):
    """Generate descriptive filename from control values."""
    def _get(key, default=""):
        return controls.get(f"{key}:value", default)

    yr = _get("cp-yr-range", [2031, 2075])
    start, end = (yr[0], yr[1]) if isinstance(yr, list) and len(yr) == 2 else (2031, 2075)
    freq = str(_get("cp-freq", "Monthly")).lower()
    qs = _get("cp-qs", [0.25])
    q_str = f"Q{int(qs[0]*100)}" if isinstance(qs, list) and qs else "Q50"
    stack = _get("cp-stack", 1)
    cash = _abbreviate(_get("cp-cash-init", 0))
    spend = _abbreviate(_get("cp-spend", 0))

    parts = [f"citadel_{start}-{end}_{freq}_{q_str}"]
    if stack:
        parts.append(f"stack-{stack}btc")
    if cash:
        parts.append(f"cash-{cash}")
    if spend:
        parts.append(f"spend-{spend}")

    # Tax
    tax_on = _get("cp-tax-toggle", [])
    if tax_on and "on" in (tax_on if isinstance(tax_on, list) else [tax_on]):
        tax_cfg = controls.get("cp-tax-config:data", {})
        if isinstance(tax_cfg, dict):
            st = tax_cfg.get("state_code", "")
            if st:
                parts.append(f"tax-{st}")

    # MC
    mc_on = _get("cp-mc-enable", [])
    if mc_on and "on" in (mc_on if isinstance(mc_on, list) else [mc_on]):
        sims = _get("cp-mc-sims", "")
        if sims:
            parts.append(f"mc-{sims}s")

    date_str = datetime.date.today().isoformat()
    parts.append(date_str)
    return "_".join(parts) + ".json"


# ── Save prep: server callback collects all data ─────────────────────────────
# Build the State() list dynamically from _CP_CONTROLS
_save_states = [State(cid, prop) for cid, prop in _CP_CONTROLS]

@callback(
    Output("cp-save-prep", "data"),
    Input("cp-save-btn", "n_clicks"),
    State("citadel-graph", "figure"),
    State("cp-mc-results", "data"),
    State("cp-tax-config", "data"),
    State("cp-tax-annual-data", "data"),
    *_save_states,
    prevent_initial_call=True,
)
def _save_prep(n_clicks, figure, mc_results, tax_config, annual_taxes, *control_vals):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    # Build controls dict in snapshot format
    controls = {}
    for (cid, prop), val in zip(_CP_CONTROLS, control_vals):
        controls[f"{cid}:{prop}"] = val

    # Build sim_data — strip per-sim arrays for MC
    sim_data = mc_results
    if sim_data and isinstance(sim_data, dict):
        n_sims = sim_data.get("n_sims", 1)
        if n_sims is None:
            # Try to infer from array shapes
            ta = sim_data.get("total_usd")
            n_sims = len(ta) if isinstance(ta, list) and ta and isinstance(ta[0], list) else 1
        if n_sims > 1:
            # Strip per-sim arrays, keep aggregated data
            keep_keys = {"time_axis", "percentiles", "median", "depletion_period",
                         "annual_taxes", "rebal_events", "n_sims", "cumulative_spend",
                         "path_key", "overlay_key", "created", "tab", "metadata"}
            sim_data = {k: v for k, v in sim_data.items() if k in keep_keys}

    scenario = {
        "type": "quantoshi_scenario",
        "version": 1,
        "tab": "cp",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "controls": controls,
        "tax_config": tax_config,
        "figure": figure,
        "sim_data": sim_data,
        "annual_taxes": annual_taxes,
    }

    filename = _build_filename(controls)
    return {"filename": filename, "data": scenario}


# ── Clientside download: triggered by cp-save-prep ───────────────────────────
clientside_callback(
    """
    function(prep) {
        if (!prep || !prep.data) return window.dash_clientside.no_update;
        var json = JSON.stringify(prep.data, null, 2);
        var blob = new Blob([json], {type: 'application/json'});
        var url  = URL.createObjectURL(blob);
        var a    = document.createElement('a');
        a.href     = url;
        a.download = prep.filename || 'citadel_scenario.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return window.dash_clientside.no_update;
    }
    """,
    Output("cp-save-prep", "data", allow_duplicate=True),
    Input("cp-save-prep", "data"),
    prevent_initial_call=True,
)
```

NOTE: The `clientside_callback` must be called on the `app` object, not as a standalone. Check how existing clientside callbacks are registered (e.g. in `_app_ctx.py` or the layout module) and follow that pattern. If `_app_ctx.app` is the Dash app instance, use `_app_ctx.app.clientside_callback(...)`.

- [ ] **Step 4: Add import to `callbacks/__init__.py`**

```python
import callbacks.citadel_save_cb  # noqa: F401
```

- [ ] **Step 5: Test save flow**

Start dev server and manually test:
```bash
DEV=1 bash run_web.sh
```
Navigate to Citadel tab, run a simulation, click Save. Verify JSON file downloads with correct filename and contents.

If unable to start dev server, at minimum verify the module imports:
```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "
import os; os.environ['DEV'] = '1'
import app
print('App loaded with save callbacks OK')
"
```

- [ ] **Step 6: Commit**

```bash
git add btc_web/callbacks/citadel_save_cb.py btc_web/callbacks/__init__.py
git commit -m "feat: Citadel save callback — server prep + clientside download"
```

---

### Task 4: Add load callback to `citadel_save_cb.py`

**Files:**
- Modify: `btc_web/callbacks/citadel_save_cb.py`

- [ ] **Step 1: Read the existing MC upload pattern for reference**

```bash
sed -n '174,217p' btc_web/callbacks/mc_upload.py
```

- [ ] **Step 2: Add the load callback**

The load callback parses the uploaded JSON, validates it, then outputs:
- All cp-* control values (multi-output with `allow_duplicate=True`)
- `cp-mc-results.data` (sim results)
- `cp-tax-config.data` (tax config)
- `cp-tax-annual-data.data` (annual taxes)
- `citadel-graph.figure` (direct figure injection — no re-run)
- `cp-load-status` (status message)

```python
# ── Load: parse uploaded JSON, restore controls + inject figure ──────────────
_load_outputs = [
    Output("cp-scenario-upload", "contents"),      # clear upload
    Output("cp-load-status", "children"),           # status message
    Output("citadel-graph", "figure", allow_duplicate=True),  # direct figure inject
    Output("cp-mc-results", "data", allow_duplicate=True),
    Output("cp-tax-config", "data", allow_duplicate=True),
    Output("cp-tax-annual-data", "data", allow_duplicate=True),
] + [
    Output(cid, prop, allow_duplicate=True) for cid, prop in _CP_CONTROLS
]

@callback(
    *_load_outputs,
    Input("cp-scenario-upload", "contents"),
    prevent_initial_call=True,
)
def _load_scenario(contents):
    if not contents:
        raise dash.exceptions.PreventUpdate

    n_outputs = 6 + len(_CP_CONTROLS)
    err = lambda msg: (None, msg) + (no_update,) * (n_outputs - 2)

    try:
        # Parse base64-encoded upload
        content_type, content_string = contents.split(",", 1)
        raw = base64.b64decode(content_string)
        if len(raw) > 2_000_000:  # 2MB cap
            return err("File too large (max 2MB)")
        data = json.loads(raw)
    except Exception as e:
        return err(f"Parse error: {e}")

    # Validate
    if data.get("type") != "quantoshi_scenario":
        return err("Not a Quantoshi scenario file")
    if data.get("tab") != "cp":
        return err(f"Wrong tab: expected 'cp', got '{data.get('tab')}'")
    if data.get("version", 0) > 1:
        return err(f"Unsupported version: {data.get('version')}")

    controls = data.get("controls", {})
    figure = data.get("figure", no_update)
    sim_data = data.get("sim_data", no_update)
    tax_config = data.get("tax_config", no_update)
    annual_taxes = data.get("annual_taxes", no_update)

    # Build control outputs in _CP_CONTROLS order
    # Whitelist: only restore known control IDs
    known_keys = {f"{cid}:{prop}" for cid, prop in _CP_CONTROLS}
    control_vals = []
    for cid, prop in _CP_CONTROLS:
        key = f"{cid}:{prop}"
        if key in controls and key in known_keys:
            control_vals.append(controls[key])
        else:
            control_vals.append(no_update)

    created = data.get("created", "unknown")[:19]
    status = f"Loaded scenario from {created}"

    return (
        None,           # clear upload
        status,         # status message
        figure,         # direct figure injection
        sim_data,       # mc results
        tax_config,     # tax config
        annual_taxes,   # annual taxes
        *control_vals,  # all cp-* controls
    )
```

- [ ] **Step 3: Also add the Save button enable/disable clientside callback**

```python
# Enable Save button when graph has data
_app_ctx.app.clientside_callback(
    "function(fig) { return !(fig && fig.data && fig.data.length > 0); }",
    Output("cp-save-btn", "disabled"),
    Input("citadel-graph", "figure"),
)
```

- [ ] **Step 4: Test load flow**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "
import os; os.environ['DEV'] = '1'
import app
from callbacks.citadel_save_cb import _CP_CONTROLS, _build_filename
print(f'Citadel controls: {len(_CP_CONTROLS)}')
print(f'Sample filename: {_build_filename({\"cp-yr-range:value\": [2031, 2075], \"cp-freq:value\": \"Monthly\", \"cp-qs:value\": [0.25], \"cp-stack:value\": 1.0, \"cp-cash-init:value\": 100000, \"cp-spend:value\": 5000})}')
print('OK')
"
```

- [ ] **Step 5: Run full test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --ignore=btc_web/test_tax_e2e.py --tb=short 2>&1 | tail -5
```

Expected: 889 passed, 0 failed, 5 skipped.

- [ ] **Step 6: Commit**

```bash
git add btc_web/callbacks/citadel_save_cb.py
git commit -m "feat: Citadel load callback — parse JSON, restore controls, inject figure"
```

---

### Task 5: End-to-end testing

- [ ] **Step 1: Start dev server and test manually**

```bash
DEV=1 bash run_web.sh &
sleep 5
```

Test sequence:
1. Navigate to Citadel tab (localhost:8050/9)
2. Configure a scenario (change stack, spending, enable tax)
3. Click Run Simulation
4. Verify Save button becomes enabled
5. Click Save — verify JSON downloads with descriptive filename
6. Inspect the JSON — verify controls, figure, sim_data, tax_config present
7. Change some controls (different stack amount)
8. Click Load, select the saved file
9. Verify controls restore to saved values
10. Verify chart appears immediately (no "Computing..." spinner)

- [ ] **Step 2: Test with MC (if available)**

If Markov module available, enable MC, run, save, verify per-sim arrays are stripped.

- [ ] **Step 3: Run test suite one final time**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --ignore=btc_web/test_tax_e2e.py --tb=short 2>&1 | tail -5
```

- [ ] **Step 4: Commit any fixes**

```bash
git add -A btc_web/
git commit -m "fix: citadel save/load E2E fixes"
```
