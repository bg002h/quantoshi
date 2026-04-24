# Single Chart Redraw per Snapshot Restore — Design

**Date:** 2026-04-24
**Status:** Approved to implement
**Depends on:** `docs/superpowers/specs/2026-04-24-drop-all-tabs-snapshot-design.md` (shipped commit 7e929cd)
**Supersedes:** `memory/parked_single_redraw_brainstorm.md`

## Goal

On share-link restore, the active tab's chart callback should render **once** with the fully-restored control state, instead of 2–3 times as it does today. Measured impact: ~800 ms off the ~3 s perceived lag on prod cold-cache (~60% of user-visible restore delay).

## Motivation

Profiled on prod after `redis-cli FLUSHDB` (Firefox desktop, 1400 × 900):

| Event | Time from nav start |
|---|---|
| DOM interactive | 908 ms |
| First chart callback fires | 1.56 s |
| First chart figure committed (75 KB Plotly payload) | 2.77 s |
| Figure settles after 2nd chart fire | ~3.5 s |
| Figure settles after 3rd chart fire | ~4 s |

Three chart-callback invocations per restore × ~400 ms server compute each = ~1.2 s of redundant work on the critical path. The final figure is the only one the user sees; the first two are thrown away when the next Input batch overwrites them.

## Non-goals

- Not reducing the total number of Dash callback invocations (eager writers still fire).
- Not refactoring the chart callback's Input list or parameter shapes.
- Not changing the background prefetch behavior.
- Not touching the cache-key pipeline or figure builders.
- Not optimizing network / payload / Plotly render cost — those are separate and much smaller wins.

## User-visible behavior

### What changes
- Restore lag drops from ~3–4 s to ~2–2.5 s on the active tab's first render.
- No user-facing UI changes. No spinner, no banner, no visible "Restoring…" state.

### What stays the same
- The pre-injected figure from L1 cache is visible immediately on layout build (unchanged).
- If restore fails (decode error, empty hash), behavior is identical to today.
- Chart callback semantics under user interaction (clicks, typing, sliders) are unchanged.

## Architecture

### New component

- `dcc.Store(id="snapshot-pending", storage_type="memory", data=False)` in root layout (`btc_web/layout/__init__.py`, near the other snapshot-related stores).

### Writers of `snapshot-pending`

| Callback | Sets | When |
|---|---|---|
| `restore_from_url` (server) | `True` | A valid share-hash is decoded. Added to its existing output list. |
| `apply_tab_{tab}` (server, 7 instances) | `False` | Same output batch as the tab's control writes. Only the active tab's `apply_tab_{tab}` fires per restore. |
| `_release_snapshot_pending_safety_timer` (new clientside) | `False` | 3000 ms after `snapshot-pending` flips to `True`, unconditionally. |

### Readers

All 9 figure-writing chart callbacks — the 5 main tab callbacks in `charts/__init__.py` (`update_bubble`, `update_heatmap`, `update_dca`, `update_retire`, `update_supercharge`), the 2 bubble sub-view callbacks also in `charts/__init__.py` (`update_bub_cagr`, `update_bub_resid`), plus `update_citadel` (`citadel_cb.py`) and `update_leverage` (`leverage_cb.py`) — add `State("snapshot-pending","data")` and early-return `no_update` when it is `True`.

### Why this works

Cost inventory during a share-link restore:

1. `restore_from_url` writes `snapshot-state-store` → `snapshot-pending=True`.
2. `apply_globals` fires on `snapshot-state-store` change → writes 31 global controls including `palette-store`. Chart callback fires (`palette-store` is an Input). Reads `snapshot-pending=True` as State → early-returns `no_update`. **Fire skipped.**
3. Safety bump (`routing.py:79-110`) fires on `snapshot-state-store` → bumps `{active}-first-render`. Chart callback fires (`first-render` is an Input). Reads `snapshot-pending=True` → early-returns. **Fire skipped.** `apply_tab_{active}` also fires on `first-render` change.
4. `apply_tab_{active}` writes ~30 tab controls + `snapshot-pending=False` in the same output batch. Chart callback fires on the control changes, reads `snapshot-pending=False` → renders. **Meaningful fire (the one the user sees).**

Net: 1 render instead of 3.

### Non-chart-tab restore

Restores to `/8` (stack), `/9` (model_info), `/10` (faq) do not fire any chart callback — no gate necessary, but `snapshot-pending=True` would stay set forever without a release path (no `apply_tab_{tab}` for those). The 3000 ms safety timer clears it unconditionally, covering this case plus any unanticipated broken path.

### Failure modes

| Scenario | Outcome |
|---|---|
| `restore_from_url` decodes a bad hash | Returns `no_update` for state store AND `snapshot-pending`. Gate stays `False`. No effect. |
| User pastes a share link mid-session | Same flow as initial load. Gate fires, releases via `apply_tab_{active}`. |
| Server slow (cold cache, >3 s compute) | Safety timer fires mid-compute, gate releases. The first chart-callback fire *after* release renders — likely with partial state (only globals applied, tab controls not yet). `apply_tab_{active}`'s subsequent write triggers a second render with full state. **Regresses to current 2-render behavior on this worst-case path.** Acceptable: still ≤ today. |
| `apply_tab_{active}` never fires (lazy-load race, missing first-render bump) | Safety timer rescues after 3 s. |

## Key invariants (locked by tests)

1. **`snapshot-pending` must NOT enter any chart callback's `params` dict.** Read the State, early-return `no_update` for `figure` **before** building `params`. Violation causes prewarm cache to miss every cold load. Test asserts no chart callback's `_q3`-quantized params dict contains the key `snapshot_pending` (or any key aliased from it).
2. **`restore_from_url` must use `prevent_initial_call='initial_duplicate'` after gaining an `allow_duplicate=True` Output.** Currently `prevent_initial_call=False`. Switching is safe — `'initial_duplicate'` still fires on initial load. Test asserts the parameter value.
3. **All `snapshot-pending` writers use `allow_duplicate=True`.** 1 restore + 7 apply_tab + 1 safety timer = 9 writers. Test walks `app.callback_map` and asserts every callback that outputs to `snapshot-pending.data` has `allow_duplicate=True`.
4. **Safety timer ≥ 3000 ms.** Shorter values collapse to pre-gate behavior on cold-cache citadel/supercharge. Test extracts the timer value from the clientside callback source string and asserts `>= 3000`.

## Components to change

| File | Change |
|---|---|
| `btc_web/layout/__init__.py` | Add `dcc.Store(id="snapshot-pending", storage_type="memory", data=False)` alongside existing snapshot stores. |
| `btc_web/callbacks/snapshot_cb.py` | `restore_from_url`: add `Output("snapshot-pending","data",allow_duplicate=True)`, switch `prevent_initial_call` to `'initial_duplicate'`. On successful decode return `True`; on empty hash or decode failure return **`no_update`** (not `False`, to avoid spurious writes that would churn the safety timer). Factory `_make_apply_tab_callback`: add `Output("snapshot-pending","data",allow_duplicate=True)` as the LAST output. When state is None/empty, return `[no_update] * (len(_ctrls) + 1)` — the gate output must also be `no_update`, never `False`, so non-restore first-render bumps don't accidentally clear the gate. On populated state, append `False` to release the gate alongside tab control writes. |
| `btc_web/callbacks/snapshot_cb.py` (append at end) | Register a clientside callback with a declared `Output("snapshot-pending","data",allow_duplicate=True)`. `Input("snapshot-pending","data")` fires on every change. On flip to `True`, stash a 3000 ms `setTimeout` on `window._snapshotTimer` that calls `window.dash_clientside.set_props("snapshot-pending", {data: false})` and returns `no_update`. On flip to `False`, `clearTimeout(window._snapshotTimer)` and return `no_update`. `prevent_initial_call=True`. This declared Output is not strictly needed (the timer uses `set_props`) but it satisfies invariant #3's "every writer has allow_duplicate" static check. |
| `btc_web/callbacks/charts/__init__.py` | All 7 figure-writing chart callbacks registered here: `update_bubble` (→`bubble-graph.figure`), `update_heatmap` (→`heatmap-graph.figure`), `update_dca` (→`dca-graph.figure`), `update_retire` (→`retire-graph.figure`), `update_supercharge` (→`supercharge-graph.figure`), `update_bub_cagr` (→`bub-cagr-graph.figure`), `update_bub_resid` (→`bub-resid-graph.figure`). Add trailing `State("snapshot-pending","data")` Input. In each function body, as the **very first statement, BEFORE any existing PreventUpdate / hydration / cta-active guards**, check the state value and `return no_update` (or tuple of `no_update` for multi-output callbacks). Gate must win over all other early-return logic so restore always settles deterministically. |
| `btc_web/callbacks/citadel_cb.py` | `update_citadel`: same pattern as above. |
| `btc_web/callbacks/leverage_cb.py` | `update_leverage`: same pattern. |

## Tests

### New tests in `btc_web/test_snapshot.py`

- `test_snapshot_pending_in_layout`: assert the Store id exists in the rendered layout.
- `test_snapshot_pending_writers_have_allow_duplicate`: walk `app.callback_map`; every callback with Output `snapshot-pending.data` must have `@` (allow_duplicate marker) in its key.
- `test_restore_from_url_uses_initial_duplicate`: introspect `restore_from_url` via Dash's callback map; assert its `prevent_initial_call` setting is `'initial_duplicate'`.
- `test_apply_tab_outputs_include_snapshot_pending`: each of the 7 `apply_tab_*` callbacks' Outputs includes `snapshot-pending.data`.
- `test_apply_tab_releases_gate`: invoke `apply_tab_bubble(1, {"bub-xscale:value":"Lin"})` and confirm the last return value is `False`.
- `test_apply_tab_is_still_noop_when_state_none`: existing test, verify still passes (state None → all `no_update`, including gate position).

### New tests in `btc_web/test_callbacks.py`

- `test_chart_callbacks_have_snapshot_pending_state`: for each of the 5 chart callbacks in `charts/__init__.py`, introspect the registered callback's State list and assert `snapshot-pending.data` is present.
- `test_citadel_and_leverage_chart_callbacks_have_snapshot_pending_state`: same for `update_citadel`, `update_leverage`.
- `test_chart_callback_short_circuits_when_pending`: invoke each of the 9 figure-writing chart callbacks with `snapshot_pending=True` via a mock ctx; assert output is `no_update`.
- `test_snapshot_pending_not_in_params_dict`: invoke `update_bubble` with a crafted state that would set the gate to False; check the params dict passed to `build_bubble_figure` does NOT contain `snapshot_pending`. Protects cache-key alignment.

### New test in `btc_web/test_web.py` or a new file

- `test_safety_timer_is_at_least_3000ms`: read the clientside callback source string from the registered clientside callback list; `assert "3000" in src` (or parse the numeric literal). Guards the cold-cache regression case.

### Preserve
- All existing snapshot-restore tests (`test_valid_roundtrip`, partition integrity, etc.).
- The 6 post-drop-all-tabs architecture tests.

### Additional negative tests

- `test_apply_globals_does_not_output_snapshot_pending`: walk the registered callback for `apply_globals` and assert its Output list does NOT include `snapshot-pending.data`. Guards the race where a future editor moves the release into `apply_globals` — which would clear the gate before `apply_tab_{active}` writes tab controls and break the single-redraw invariant.

## Rollout

- Single feature branch, single commit cluster, single PR.
- Run `dash-callback-reviewer` on the diff before push (user-requested hard gate).
- Deploy to prod via `git push origin master && ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"`.
- Re-profile on prod immediately post-deploy: navigate to the same test share link, measure time from nav to first chart figure commit. Target: ≤ 2.5 s on desktop Firefox (was ~2.77 s), AND no 2nd/3rd chart figure payload in the resource timeline (currently 75 KB figure appears 3 times).
- Monitor prod logs for 24 h: no new "nonexistent object" or "allow_duplicate" errors; no gunicorn crashes.
- If profile shows regression or the safety timer firing routinely (via optional telemetry console.log), rollback is `git revert` — no data migration, no cache rebuild.

## Out of scope

- Pre-warming Redis cache with common share-link configs.
- Deferring background prefetch further past the restore window.
- Batching `apply_globals` + `apply_tab_{active}` into a single callback invocation (would simplify but requires restructuring the dispatch and removing the safety bump pattern).
- Server-side chart-compute speedups.
- Any UI feedback during the remaining ~2.5 s of restore (deferred to a future spec if the post-deploy profile shows it's still needed).
