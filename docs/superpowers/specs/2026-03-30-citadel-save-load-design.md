# Citadel Save/Load — Design Spec (v2)

**Goal:** Add save/load for Citadel Planner scenarios. Extends the existing MC save/upload pattern to Citadel, adds control persistence, and includes Plotly figure JSON for instant interactive viewing on load.

**Motivation:** Citadel MC simulations take ~80 seconds (engine cost, not sampling cost). Users need to save results without re-running. The existing MC save/upload system works for DCA/Retire/SC/Heatmap but Citadel has no upload capability. Additionally, no tab saves control settings — users must use share URLs for that.

---

## Unified Save Strategy

| Tab | Simulation data saved | Controls | Figure JSON | Reuse on load |
|-----|----------------------|----------|-------------|---------------|
| DCA/Retire/SC/Heatmap | price_paths (full, ~2.4MB) | Yes (new) | Yes (new) | Re-run overlay with different params (ms) |
| Citadel (deterministic) | Full SimResult (~290KB) | Yes | Yes | Change display mode without re-run |
| Citadel (MC) | Percentile bands only (~155KB gzip) | Yes | Yes | View saved chart; re-run for changes |

**Why Citadel MC doesn't save price_paths:** The engine (waterfall, tax, floors, rebalancing) takes ~80s for 1000 sims. Cached paths save only the Markov sampling (~1s), not the engine run. Saving 48MB of paths to save 1 second isn't worthwhile.

**Why other tabs DO save price_paths:** Their overlay math (price x amount) is milliseconds. Cached paths let users tweak DCA amount, withdrawal rate, etc. and get new results without paying again.

---

## Save File Format

Single JSON file. All tabs use the same structure:

```json
{
  "type": "quantoshi_scenario",
  "version": 1,
  "tab": "cp",
  "created": "2026-03-30T16:34:00Z",
  "controls": {
    "cp-stack:value": 10.0,
    "cp-cash-init:value": 100000,
    ...all tab controls in "{cid}:{prop}" snapshot format...
  },
  "figure": { ...Plotly figure JSON... },
  "sim_data": { ...tab-specific simulation results... }
}
```

- `type`: always `"quantoshi_scenario"` (validates on load)
- `tab`: which tab produced this (`"cp"`, `"dca"`, `"ret"`, `"sc"`, `"hm"`)
- `controls`: all controls for this tab in snapshot key format
- `figure`: full Plotly figure dict for instant interactive restore. Note: saved figures may become visually inconsistent after app updates (new trace styles, layout changes). The `version` field flags this — future version bumps can trigger re-render from `sim_data` instead of using the stale figure.
- `sim_data`: tab-specific — MC results dict for DCA/Retire/SC/HM; SimResult dict for Citadel

**Citadel sim_data details:**
- Deterministic (`n_sims == 1`): full `SimResult.to_dict()` output — all keys kept
- MC (`n_sims > 1`): `SimResult.to_dict()` with per-sim arrays stripped. **Keep:** `time_axis`, `percentiles`, `depletion_period`, `annual_taxes`, `n_sims`. **Drop:** any top-level value that is a 2D array/list with first dimension == n_sims (`btc_holdings`, `btc_prices`, `cash_balances`, `reserve_balances`, `invest_balances`, `total_usd`, `td_total`, `tf_total`, `taxable_total`)
- Detection: automatic from `n_sims` field in the dict

**Version migration:** Loader rejects files with `version` > supported. Future format changes bump `version` and add sequential migration transforms (v1→v2→...→current).

**Other tabs sim_data:** The existing MC result dict (with `price_paths`, `fan_btc`, `fan_usd`, `path_key`, `overlay_key`). Same as what they already save — just now wrapped in the unified format.

---

## Filename

Auto-generated per tab:

- Citadel: `citadel_{start_yr}-{end_yr}_{freq}_{quantile}_stack-{btc}_cash-{cash}_spend-{spend}{_tax-{state}}{_mc-{sims}s}_{date}.json`
- DCA: `dca_{start_yr}-{end_yr}_{amount}_{freq}{_mc-{sims}s}_{date}.json`
- Retire: `retire_{start_yr}-{end_yr}_{withdrawal}_{freq}{_mc}_{date}.json`
- SC/HM: similar patterns from their key controls

---

## UI — Citadel Tab

Two new elements below "Run Simulation" in the Simulation sub-tab:

```
[Run Simulation]
[Save Scenario] [Load Scenario]
```

- **Save**: `dbc.Button`, outline, secondary, small. Enabled when simulation results exist.
- **Load**: `dcc.Upload` styled as a button. Same styling.

For other tabs: MC upload already exists. Add a "Save" button next to the existing MC upload component. This is a v2 enhancement — Citadel first.

---

## Save Flow

Hybrid: server collects data, clientside does the download.

1. User clicks "Save Scenario"
2. Server callback reads all `cp-*` controls via `State()`, plus `cp-mc-results.data`, `cp-tax-config.data`, `cp-tax-annual-data.data`
3. Builds scenario dict: controls (snapshot format), sim_data (SimResult, stripped if MC), figure (from `citadel-graph.figure`)
4. For MC results: strips per-sim arrays when `n_sims > 1`
5. Generates filename from control values
6. Writes to `cp-save-prep` store as `{"filename": "...", "data": {...}}`
7. Clientside callback watches `cp-save-prep`, creates Blob, triggers download

---

## Load Flow

Follows the proven MC upload pattern from DCA/Retire/SC/HM.

1. User clicks "Load" and selects `.json` file
2. Server callback (`citadel_save_cb.py`):
   - Parses JSON (base64 decode from `dcc.Upload`)
   - Validates `type == "quantoshi_scenario"` and `tab == "cp"`
   - Whitelists control IDs (drop unknown keys)
   - Validates value types
3. Outputs:
   - Restore all `cp-*` controls via multi-output with `allow_duplicate=True`
   - Write `sim_data` to `cp-mc-results.data`
   - Write `annual_taxes` to `cp-tax-annual-data.data`
   - Write `tax_config` to `cp-tax-config.data`
   - Write saved `figure` directly to `citadel-graph.figure` — **bypasses the figure callback entirely**
4. Instant interactive chart appears. No simulation re-run, no background process, no spinner.

**Why direct figure injection (not cp-mc-loaded trigger):** The figure callback uses `background=True` with DiskcacheManager. Triggering it would spawn a background process, show "Computing..." spinner, disable the Run button, and re-run the full simulation — defeating the purpose of save/load. Writing the figure directly as a callback `Output` avoids all of this.

**Implementation note:** The load callback restores ~96 controls simultaneously. Audit for cascading callback triggers — if any callbacks have these controls as `Input` and have side effects (clearing stores, resetting state), the load could be partially undone. Use `prevent_initial_call=True` and test thoroughly.

---

## Auto-Save Modal (Paid MC)

Existing `mc-save-modal` continues to appear after paid MC runs. For Citadel (`mc-save-tab == "cp"`), the clientside download now saves the unified scenario format (controls + sim_data + figure) instead of bare MC results. This requires the server to prepare the scenario dict and write to a store before the modal appears.

**v1 simplification:** The MC modal saves in the existing MC-only format. The new Save button saves the full scenario format. Users who want controls saved use the Save button after the modal dismisses. MC modal integration is v2.

---

## Missing Snapshot Controls (Bug Fix)

Add to `_SNAPSHOT_CONTROLS`, `_TAB_CONTROLS["citadel"]`, and `_CHECKLIST_OPTIONS`:

- `cp-high-q-enable` (checklist)
- `cp-low-q-enable` (checklist)
- `cp-inv-eq-basis` (number)
- `cp-inv-bd-basis` (number)

This fixes share URLs silently dropping these values.

---

## Files

| File | Action | Purpose |
|------|--------|---------|
| `btc_web/callbacks/citadel_save_cb.py` | Create | Save (server prep + clientside download) + load (server parse + restore) |
| `btc_web/layout/citadel.py` | Modify | Add Save/Load buttons, `dcc.Upload`, `dcc.Store("cp-save-prep")` |
| `btc_web/callbacks/citadel_cb.py` | Modify | Fast-path for loaded scenarios |
| `btc_web/snapshot.py` | Modify | Add 4 missing controls |
| `btc_web/callbacks/__init__.py` | Modify | Import citadel_save_cb |

---

## Constraints

- **No server-side storage** — files to/from user's device only
- **Privacy** — financial data never persisted server-side
- **Citadel MC files stay small** — percentile bands only (~155KB gzip), no per-sim arrays
- **Other tabs unchanged for v1** — unified save comes to DCA/Retire/SC/HM in v2
- **MC modal unchanged for v1** — saves MC-only format; scenario save via button
- **Version field** — enables future format migration
