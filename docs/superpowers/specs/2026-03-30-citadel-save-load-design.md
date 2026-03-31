# Citadel Save/Load — Design Spec

**Goal:** Add save/load for Citadel Planner scenarios. A saved file captures all controls, tax config, and simulation results. Loading restores everything instantly — controls populate, figure re-renders from saved data without re-running the simulation.

**Motivation:** The Citadel Planner has 92+ controls across 4 sub-tabs. Users build complex retirement scenarios (physician with $2M portfolio, specific tax settings, rebalancing triggers) that take time to configure and may take minutes to simulate with MC. There's no way to save and revisit a scenario.

---

## Save File Format

Single JSON file, ~50-200KB:

```json
{
  "type": "citadel_scenario",
  "version": 1,
  "app": "Quantoshi",
  "created": "2026-03-30T16:34:00Z",
  "controls": {
    "cp-stack:value": 10.0,
    "cp-cash-init:value": 100000,
    "cp-spend:value": 5000,
    "cp-yr-range:value": [2031, 2075],
    "cp-freq:value": "Monthly",
    "cp-qs:value": [0.25],
    "cp-tax-toggle:value": ["on"],
    "cp-mc-enable:value": ["on"],
    "cp-mc-sims:value": 1000,
    ...all cp-* controls in "{cid}:{prop}" snapshot format...
  },
  "tax_config": {
    "filing_status": "married",
    "state_code": "TX",
    "birth_year": 1975,
    ...full cp-tax-config store data...
  },
  "sim_result": {
    "time_axis": [...],
    "btc_holdings": [[...]],
    "total_usd": [[...]],
    "cash_balances": [[...]],
    ...SimResult.to_dict() output...
  },
  "annual_taxes": [...]
}
```

Key format uses `"{cid}:{prop}"` matching the snapshot system for trivial restore.

---

## Filename

Auto-generated, detailed:

```
citadel_{start_yr}-{end_yr}_{freq}_{quantile}_stack-{btc}_cash-{cash}_spend-{spend}{_tax-{state}-{filing}}{_mc-{sims}s}_{date}.json
```

Examples:
- `citadel_2031-2075_monthly_Q25_stack-1btc_cash-100k_spend-5k_2026-03-30.json`
- `citadel_2031-2075_monthly_Q25_stack-10btc_cash-500k_spend-8k_tax-TX-married_mc-1000s_2026-03-30.json`

Values abbreviated: cash/spend in `k` or `M` for readability.

---

## UI

Two buttons below "Run Simulation" in the Simulation sub-tab:

```
[▶ Run Simulation]          (existing, dbc.Button color="warning")
[↓ Save Scenario] [↑ Load]  (new, dbc.Button outline=True color="secondary" size="sm")
```

- **Save** is enabled only when simulation results exist (the graph has data)
- **Load** wraps a `dcc.Upload` component

---

## Save Flow

1. User clicks "Save Scenario"
2. Clientside JS callback:
   - Reads all `cp-*` control values from the DOM
   - Reads `cp-tax-config.data`, `cp-mc-results.data`, annual taxes from stores
   - Builds JSON dict with controls, tax_config, sim_result, annual_taxes
   - Generates filename from control values
   - Creates Blob and triggers browser download
3. No server round-trip needed

---

## Load Flow

1. User clicks "Load" and selects a `.json` file
2. Server callback (`citadel_save_cb.py`):
   - Parses JSON from upload
   - Validates `type == "citadel_scenario"` and `version`
   - Whitelists control IDs against known `cp-*` set (drops unknown keys)
   - Caps file size (reject > 500KB)
3. Writes control values to `cp-scenario-store` (new memory store)
4. A separate callback watches `cp-scenario-store` and outputs to all `cp-*` controls using `allow_duplicate=True`
5. Writes `sim_result` to `cp-mc-results.data` and `annual_taxes` to the tax store
6. The existing figure callback detects the new data and re-renders the chart from the saved simulation results — no re-run needed

---

## Auto-Save Modal (Paid MC Only)

The existing `mc-save-modal` behavior stays for paid MC simulations. Change: the saved file now uses the citadel_scenario format (controls + results) instead of MC-results-only format. This means:

- The clientside download callback for `mc-save-modal-dl` (in `mc_upload.py`) gets a special case for `mc-save-tab.data == "cp"`: it saves the full scenario format instead of bare MC results
- Other tabs (DCA, Retire, Heatmap, SC) continue using the existing MC-only format

---

## Missing Snapshot Controls (Bug Fix)

These controls are `State` inputs to the Citadel callback but missing from `_SNAPSHOT_CONTROLS` in `snapshot.py`:

- `cp-high-q-enable` (checklist)
- `cp-low-q-enable` (checklist)
- `cp-inv-eq-basis` (number)
- `cp-inv-bd-basis` (number)

Fix: add them to `_SNAPSHOT_CONTROLS` and `_TAB_CONTROLS["citadel"]`. The save/load feature inherits the fix. This also fixes snapshot share links silently dropping these values.

---

## Files

| File | Action | Purpose |
|------|--------|---------|
| `btc_web/callbacks/citadel_save_cb.py` | Create | Save (clientside) + load (server) callbacks |
| `btc_web/layout/citadel.py` | Modify | Add Save/Load buttons + `dcc.Upload` + `dcc.Store("cp-scenario-store")` |
| `btc_web/callbacks/citadel_cb.py` | Modify | Accept `cp-scenario-store` trigger for figure re-render |
| `btc_web/callbacks/mc_upload.py` | Modify | Special-case Citadel in MC save modal to use scenario format |
| `btc_web/snapshot.py` | Modify | Add 4 missing controls |
| `btc_web/callbacks/__init__.py` | Modify | Import citadel_save_cb |

---

## Constraints

- **No server-side storage** — files download to/upload from user's device only
- **Privacy** — scenario files contain financial planning data; never sent to server beyond the upload parse
- **Backward compatible** — existing MC save/load for other tabs unchanged
- **Version field** — enables future format changes without breaking old files
