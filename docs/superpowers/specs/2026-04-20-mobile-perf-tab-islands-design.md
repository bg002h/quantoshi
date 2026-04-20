# Mobile Performance — Tab Islands Tightening

**Date:** 2026-04-20
**Status:** Design
**Target:** Quantoshi Dash web app (`btc_web/`), Dash 4.0.0

## Goal

Eliminate mobile control-interaction lag on tabs 1–6 by tightening Dash's island model: no redundant server round-trips, no cross-tab Input crosstalk, no full figure rebuilds when only display properties change.

## Scope

Five independently-deployable batches. Each ships as one commit to master and gets tested on prod; roll back the batch if regression. No profiling infrastructure — we act on known bottlenecks (validated with the `dash-callback-reviewer` subagent).

Out of scope: initial page load optimization, chart-callback computation internals, slider-debounce on dropdowns/checklists (those aren't the pain point).

## Architecture principles (applied across all batches)

1. **Island boundary:** a tab's callbacks only read/write components belonging to that tab. Globals (palette, lots) flow in as `State`, never `Input`.
2. **No server round-trips for display-only changes:** color, legend visibility, text formatting → clientside `Patch()` on `figure` prop.
3. **Commit-store pattern:** clientside callback watches fine-grained controls (slider, MC inputs), debounces, writes committed value to a single `{scope}-commit` memory `Store`. Chart callback reads only the commit store.
4. **In-flight guard:** clientside state tracks whether the last emitted commit has been rendered by Plotly; new commits hold until acknowledged. Lets us drop debounce to 100ms without queue buildup on slow mobile CPUs.

---

## Batch 1 — Slider debounce (100ms + in-flight guard)

**Problem:** slider mouseup/debounce was reverted when Dash was pinned to 4.0.0 (commits `82dcfaa` → `2557863`). Every drag pixel currently fires a server callback; on mobile this compounds into visible lag.

**Fix:**
- For each chart-driving slider on tabs 1–6 (~12 sliders: `bub-xrange`, `bub-yrange`, `hm-entry-yr`, `hm-exit-range`, `hm-b1`, `hm-b2`, `dca-yr`, `ret-yr`, `sc-yr`, `cp-yr-range`, `bub-ptsize`, `bub-ptalpha`), add a `{id}-commit` memory Store. (`hm-entry-q` is a `dbc.Input(type=number)`, not a slider — use built-in `debounce=True` on that component instead of the commit-store pattern.)
- One clientside callback per slider: Input=`{id}.value`, State=`{tab}-render-done` counter, Output=`{id}-commit.data`. Logic: 100ms debounce; if `render-done` < last-emitted-id, hold.
- `{tab}-render-done` counter: clientside callback on `{tab}-graph.figure` (triggered when Plotly re-renders) increments the counter.
- Chart callbacks change `Input("slider-id", "value")` → `Input("slider-id-commit", "data")`.

**Risk:** medium. Affects every chart tab. Must preserve existing slider `value` state for snapshot encode/decode (snapshot reads `.value`, not `-commit`, so unaffected).

**Rollback signal:** any chart fails to update when slider dragged; `-commit` store never fires.

---

## Batch 2 — Per-tab MC commit Store

**Problem:** `update_dca` (+ retire/sc/heatmap/citadel) list ~15 `{tab}-mc-*` controls as individual `Input`s. Not a real perf issue (Dash skips dispatch for missing components per the reviewer), but noisy DEV errors and breaks the island model.

**Fix:**
- Each of the 5 MC-enabled tabs gets one `{tab}-mc-commit` memory Store.
- One clientside callback per tab: all `{tab}-mc-*` controls as Input, 100ms debounce + in-flight guard (same as Batch 1), Output=`{tab}-mc-commit.data` (dict of all MC values).
- Chart callback: demote all `{tab}-mc-*` from `Input` → `State`; add `Input("{tab}-mc-commit", "data")` as new trigger.
- Removes 75 individual Input slots; 5 commit-store Inputs in their place.

**Risk:** low. Mechanical substitution. One aggregator per tab.

---

## Batch 3 — Eliminate palette/lots bridge

**Problem:** the bridge clientside callback in `_clientside.py` (added by the `dash-tab-islands` branch) watches `palette-store` and `effective-lots` as Inputs and bumps the active tab's `first-render`. Fires on initial Store hydration (redundant rebuild) and on snapshot restore (potential race).

**Fix (reviewer's recommendation):**
- Delete the bridge clientside callback.
- Palette dropdown click callbacks: add `{active-tab}-first-render` bump as additional Output (clientside).
- Lots write callbacks (lots import, lots edit, snapshot restore): add `{active-tab}-first-render` bump as additional Output.
- Active tab determined clientside from `main-tabs.active_tab` State.

**Risk:** low. Reviewer's simpler alternative. Eliminates hydration heuristic entirely.

---

## Batch 4 — Heatmap color/text Patch

**Problem:** `update_heatmap` rebuilds the full figure when any Input changes, including color-only ones (`hm-b1`, `hm-b2`, `hm-palette`, `hm-c-lo`, `hm-c-mid1`, `hm-c-mid2`, `hm-c-hi`, `hm-grad`, `hm-mode`) and text-only ones (`hm-vfmt`, `hm-cell-fs`).

**Fix:**
- Demote all color-only and text-only Inputs from `update_heatmap` to `State`.
- New clientside callback: Input=color-only controls, Output=`heatmap-graph.figure` with `allow_duplicate=True, prevent_initial_call=True`. Uses `Patch()` to update `figure.data[0].colorscale` + `zmin`/`zmax` (needed for diverging mode per reviewer).
- Second clientside callback for text-only: patches `figure.data[0].texttemplate` and `textfont.size`.
- Verify `update_heatmap` routes through L1 cache via `_quantize_params` (reviewer flagged as unconfirmed — audit and fix if not).

**Risk:** low. Surgical. Matrix-dependent Inputs (entry_yr, entry_q, exit_range, stack, use_lots, active_model) stay on `update_heatmap`; batch 1 slider debounce handles their drag behavior.

---

## Batch 5 — Citadel palette Patch (minimal)

**Problem:** Citadel is already State-heavy (reviewer confirmed all display-only controls are already `State`). Only one mobile win remains: switching palette post-sim currently requires clicking "▶ Run Simulation" again.

**Fix:**
- `build_citadel_figure` emits a trace→model-key map (e.g., `customdata` or a Store written alongside `cp-mc-results`).
- Single clientside callback: Input=`palette-store.data`, State=trace→model map + `citadel-graph.figure` existence check. Output=`citadel-graph.figure` via `Patch()` updating `data[i].line.color` + `fillcolor` per trace.

**Risk:** very low. One clientside callback. Non-citadel tabs unaffected.

---

## Testing

- No new test requirements beyond the existing pytest suite (`btc_web/test_*.py`). Running it before each deploy guards against regressions in snapshot, cache-key alignment, and callback wiring tests.
- Manual prod verification is the primary gate: user tests on real mobile after each deploy.
- Rollback on any batch = revert the batch's commit, deploy, Redis flush.

## Deployment flow per batch

```
implement → run pytest → git commit → git push → ssh prod:
  cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi
→ user tests on mobile → approve or revert
```

## Non-goals

- No Dash version upgrade.
- No chart-builder algorithmic changes.
- No profiling instrumentation (user ruled out Task 0 for budget reasons).
- No page-load optimization (splash, bundle size, preview images).

## Open questions

None. All validated with `dash-callback-reviewer`.
