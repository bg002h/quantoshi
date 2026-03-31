# Citadel Save/Load — Design Spec

**Goal:** Add save/load for Citadel Planner scenarios. A saved file captures all controls, tax config, and simulation results. Loading restores controls and injects saved results — the figure re-renders from data without re-running the simulation.

**Motivation:** The Citadel Planner has 92+ controls across 4 sub-tabs. Users build complex retirement scenarios (physician with $2M portfolio, specific tax settings, rebalancing triggers) that take time to configure and may take minutes to simulate with MC. There's no way to save and revisit a scenario.

---

## Save File Format

Single JSON file. Deterministic scenarios ~50-300KB, MC scenarios ~50-100KB (aggregated data only).

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

**MC size handling:** For MC simulations (1000+ sims), `SimResult.to_dict()` produces ~84MB of per-sim arrays. The save flow must save only the **aggregated** data (median, percentile bands — the arrays the figure actually renders), not per-sim raw arrays. This keeps MC saves under ~100KB. The `to_dict()` method already includes percentile data; the save flow strips the full per-sim arrays (`btc_holdings`, `cash_balances`, etc. where shape is `(n_sims, n_periods)` with n_sims > 1).

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

Hybrid approach: server callback collects controls, clientside callback does the download.

1. User clicks "Save Scenario"
2. Server callback:
   - Reads all `cp-*` control values via `State()` inputs
   - Reads `cp-tax-config.data`, `cp-mc-results.data`, `cp-tax-annual-data.data`
   - For MC results: strips per-sim arrays, keeps only aggregated percentile data
   - Builds the scenario dict
   - Generates filename from control values
   - Writes to `cp-save-prep` store (new memory store) as `{"filename": "...", "data": {...}}`
3. Clientside callback watches `cp-save-prep`:
   - Creates Blob from `data`
   - Triggers browser download with `filename`

---

## Load Flow

Follows the existing MC upload pattern used by DCA/Retire/Heatmap/SC tabs.

1. User clicks "Load" and selects a `.json` file
2. Server callback (`citadel_save_cb.py`):
   - Parses JSON from upload (base64 decode)
   - Validates `type == "citadel_scenario"` and `version`
   - Whitelists control IDs against known `cp-*` set (drops unknown keys)
   - Validates value types (numbers stay numbers, lists stay lists)
   - File size sanity check (reject > 2MB)
3. Outputs:
   - All `cp-*` control values restored via multi-output callback with `allow_duplicate=True` (~96 outputs)
   - `cp-tax-config.data` restored from `tax_config`
   - `cp-mc-results.data` restored from `sim_result`
   - `cp-tax-annual-data.data` restored from `annual_taxes`
   - `cp-mc-loaded.data` **incremented** — this is the existing trigger that fires the figure callback
4. The main Citadel figure callback fires (triggered by `cp-mc-loaded`):
   - Detects `cp-mc-results` already contains valid data
   - **Fast-path**: renders figure from saved results without re-running simulation
   - The callback already has this fast-path for MC uploads — it checks if results match current params

---

## Auto-Save Modal (Paid MC)

Deferred to v2. The dedicated Save button handles all cases including post-MC. The existing MC modal continues saving MC-only data in the current format. Users who want the full scenario use the Save button.

---

## Missing Snapshot Controls (Bug Fix)

These controls are `State` inputs to the Citadel callback but missing from `_SNAPSHOT_CONTROLS` in `snapshot.py`:

- `cp-high-q-enable` (checklist — add to `_CHECKLIST_OPTIONS`)
- `cp-low-q-enable` (checklist — add to `_CHECKLIST_OPTIONS`)
- `cp-inv-eq-basis` (number)
- `cp-inv-bd-basis` (number)

Fix: add to `_SNAPSHOT_CONTROLS`, `_TAB_CONTROLS["citadel"]`, and `_CHECKLIST_OPTIONS` (for the two checklists). The save/load feature inherits the fix. This also fixes snapshot share links silently dropping these values.

---

## Files

| File | Action | Purpose |
|------|--------|---------|
| `btc_web/callbacks/citadel_save_cb.py` | Create | Save (server prep + clientside download) + load (server parse + restore) |
| `btc_web/layout/citadel.py` | Modify | Add Save/Load buttons, `dcc.Upload`, `dcc.Store("cp-save-prep")` |
| `btc_web/callbacks/citadel_cb.py` | Modify | Fast-path for loaded scenarios (skip simulation when results already present) |
| `btc_web/snapshot.py` | Modify | Add 4 missing controls + `_CHECKLIST_OPTIONS` entries |
| `btc_web/callbacks/__init__.py` | Modify | Import citadel_save_cb |

---

## Constraints

- **No server-side storage** — files download to/upload from user's device only
- **Privacy** — scenario files contain financial planning data; never sent to server beyond the upload parse
- **Backward compatible** — existing MC save/load for other tabs unchanged
- **MC size cap** — per-sim arrays stripped from MC saves; only aggregated data saved
- **Version field** — enables future format changes. Migration strategy: version N reader applies transforms sequentially (v1→v2→...→current)
