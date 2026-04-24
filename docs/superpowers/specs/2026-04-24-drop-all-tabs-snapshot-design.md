# Drop "All tabs" Snapshot Share Scope — Design

**Date:** 2026-04-24
**Status:** Approved to implement
**Pre-refactor reference commit:** `34a19a5` (see `memory/reference_alltabs_share_revert_point.md` for rollback path)

## Goal

Retire the "All tabs" share-scope option. Share links always encode the active tab's controls plus globals. Replace the 8-callback restore architecture (1 `apply_snapshot` + 7 stage-2 `_apply_lazy`) with a thinner `apply_globals` + 7 `apply_tab_{tab}` pattern that eliminates the per-tab relay stores.

## Motivation

- Default scope was already "Current tab only"; typical usage is single-tab.
- Current eager/lazy partition in `_SNAPSHOT_CONTROLS` + 7 relay stores + 7 stage-2 callbacks exist to support the "All tabs" scope and to avoid "nonexistent object" errors for lazy-mounted tabs. Dropping "All tabs" doesn't remove the second concern, but it enables a simpler mechanism to handle it.
- Thinning the restore flow unblocks the follow-up "single chart-callback redraw per snapshot restore" goal: with just two writer callbacks per restore (globals + one per-tab), the redraw-count problem becomes tractable.

## Non-goals

- Not changing the hash encoding format (`q3:` stays).
- Not changing the background prefetch behavior (`routing.py:642–657`).
- Not changing the chart callback signatures or the cache-key pipeline.
- Not tackling the single-redraw goal. That is deferred to a follow-up spec.

## User-visible behavior

### What changes
- Share modal: the **"Scope: All tabs / Current tab only" RadioItems is removed.** Modal shows only the "Include Stack Tracker lots" checkbox + Generate-link button.
- New share links (`q3:…`) encode globals + the active tab's controls only. Payload is uniformly small.

### What stays the same
- Hash prefix remains `q3:`; no version bump.
- `q1:` / `q2:` / `q3:` legacy decoders all continue to work.
- Link history UI unchanged (the `scope` field was stored but never displayed; dropped from new entries, ignored on read).

### Legacy "All tabs" links — actually better than "partial restore"
An old share link that encoded controls for all tabs will restore **fully across tab visits within the session**, not silently drop non-active-tab keys. Mechanism: `restore_from_url` writes the decoded state into `snapshot-state-store`, which persists for the session. When the user visits a different tab later (either by clicking or via background prefetch), that tab's `{tab}-first-render` bumps, its new `apply_tab_{tab}` callback fires and reads `snapshot-state-store` as State — finds the old tab's values and writes them.

Constraint: if the user navigates to `/3` and never visits `/2`, the heatmap values in the link never apply. Acceptable — no worse than today.

## Architecture

### Current restore flow (today)

```
url.hash → restore_from_url → snapshot-state-store
            ↓ (fires apply_snapshot)
         apply_snapshot
            ├─ writes ~50 eager controls directly (globals + always-mounted)
            └─ writes the full state dict to 7 per-tab relay stores
                    ↓ (each fires its own stage-2)
                 7 × _apply_lazy (Input: {tab}-first-render, State: relay store)
                    └─ writes that tab's ~20 per-tab controls
```

### Proposed restore flow

```
url.hash → restore_from_url → snapshot-state-store
            ├─ apply_globals       (Input: snapshot-state-store.data)
            │     └─ writes 31 global controls (always-mounted):
            │          main-tabs, palette-store,
            │          lppl-* (3), hybppl-cfg-{a,b}-* (14), eppl-cfg-{a,b}-* (14),
            │          snapshot-lots (special-case via _lots key).
            │
            └─ 7 × apply_tab_{tab}  (Input: {tab}-first-render.data,
                                      State: snapshot-state-store.data)
                  └─ writes 30–60 per-tab controls (bub-*, scan-*, cta-* for bubble;
                     hm-* for heatmap; dca-*, ret-*, sc-*, cp-*, lev-* for the rest).
```

Callback count: **1 apply_globals + 7 apply_tab_{tab} = 8 callbacks**, same total as today but with 7 relay-store components deleted from layout and no eager/lazy partition in source.

### Key invariant (enforced by the spec)

> **The clientside first-render-bump callback at `routing.py:79–110` MUST keep `Input("snapshot-state-store","data")` as its trigger.**
>
> Rationale: `apply_tab_{active}` reads `snapshot-state-store` as State. On initial load the bump is what causes `apply_tab_{active}` to fire. If the bump were retriggered by something earlier than `snapshot-state-store`'s write (e.g. `active_tab` change), `apply_tab_{active}` would read None and write nothing. Keeping the Input on `snapshot-state-store` guarantees Dash's write-before-read ordering holds.
>
> Tests assert this binding to prevent regression.

### Control partition (concrete)

Derived from parsing `_SNAPSHOT_CONTROLS` in `snapshot.py` against the lazy prefix set `(bub-, scan-, cta-, hm-, dca-, ret-, sc-, cp-, lev-)`:

**Globals (31) — go to `apply_globals`:**
- `main-tabs.active_tab`
- `palette-store.data`
- `lppl-n-freqs.value`, `lppl-weighted.value`, `lppl-no-13.value`
- `hybppl-cfg-a-*.value` (7 controls)
- `hybppl-cfg-b-*.value` (7 controls, including `b-enabled`)
- `eppl-cfg-a-*.value` (7 controls)
- `eppl-cfg-b-*.value` (7 controls, including `b-enabled`)
- `snapshot-lots.data` (via the `_lots` key, same as today)

**Per-tab (279) — split by prefix into 7 `apply_tab_{tab}` callbacks:**
- `bubble`: `bub-*`, `scan-*`, `cta-*`
- `heatmap`: `hm-*`
- `dca`: `dca-*`
- `retire`: `ret-*`
- `supercharge`: `sc-*`
- `citadel`: `cp-*`
- `leverage`: `lev-*`

The `lppl-*` / `hybppl-*` / `eppl-*` controls are NOT duplicated in per-tab lists — they're globals only. This eliminates the double-write risk the reviewer flagged.

## Components to change

| File | Change |
|---|---|
| `btc_web/layout/__init__.py` | Remove `share-scope` RadioItems from share modal (lines ~679–732). Remove the 7 `dcc.Store(id="snapshot-apply-{tab}", …)` declarations. |
| `btc_web/snapshot.py` | Remove `tab_filter=None` branch in `_encode_snapshot` — always apply `tab_filter=<active-tab-cids>`. Signature stays for backward compatibility with tests; add a deprecation comment. `_decode_snapshot` / `_decode_snapshot_v1` unchanged. |
| `btc_web/callbacks/snapshot_cb.py` | Delete `_EAGER_CONTROLS`, `_TAB_LAZY_CONTROLS`, `_ALL_LAZY_PREFIXES`, `_BUBBLE_LAZY_CONTROLS`, `_LAZY_TAB_SPECS`, `_N_RELAY_STORES`, `_make_lazy_relay_callback` and its 7 registrations. Rewrite `apply_snapshot` → `apply_globals` (smaller Output list). Add `_make_apply_tab_callback(tab_id, prefix_tuple)` factory that registers one `apply_tab_{tab}` per tab. Remove `State("share-scope","value")` from the encode callback. Drop `scope` from `_add_snapshot_entry`. |
| `btc_web/callbacks/splash.py` | No change needed — `prefetch-ready` gating shipped yesterday is compatible. |
| `btc_web/callbacks/routing.py` | No logic change. Add a regression-guard comment near line 79–110: "Input MUST be snapshot-state-store per spec 2026-04-24-drop-all-tabs-snapshot-design.md." |
| `btc_web/test_snapshot.py` | Delete `test_tab_filter`, `test_tab_filter_encodes_only_matching`, `test_each_tab_filter_roundtrips`, `test_single_tab_shorter_than_all`. Add tests per "Tests" section below. |
| `btc_web/test_palette_roundtrip.py` | Sweep for scope references; update if any assertions target the removed `share-scope` component. |

## Data flow details

1. User visits `host/N#q3:<payload>`.
2. `_serve_layout` builds the layout with `active_tab=<tab-for-N>` and pre-injects the active tab's figure from cache. `{active}-first-render` is pre-set to `1` in the store.
3. Dash fires `restore_from_url(hash)` (has `prevent_initial_call=False`). It decodes the hash and writes `snapshot-state-store`.
4. `apply_globals` fires on `snapshot-state-store`. Writes the 31 global controls in one output batch.
5. The clientside bump at `routing.py:79–110` fires on `snapshot-state-store` change. Increments `{active}-first-render` from `1` → `2`.
6. `apply_tab_{active}` fires on `{active}-first-render` change. Reads `snapshot-state-store` as State (populated by step 3, Dash ordering guarantee). Writes ~30 per-tab controls for the active tab in one output batch.
7. Chart callback fires on the Input changes from steps 4 + 5 + 6. (Still fires 2–3 times; single-redraw optimization is deferred.)

For **non-active tabs** visited later in the session (click or background prefetch):
- User clicks tab N or prefetch materializes it.
- `{N}-first-render` bumps.
- `apply_tab_{N}` fires, reads `snapshot-state-store` (still populated from step 3), writes N's controls if they exist in the state.
- On legacy "All tabs" links, this is how non-active-tab values get applied late. On modern "active-tab-only" links, the state dict only has the original active tab's keys, so apply_tab_{N} is a silent no-op for N≠active.

## Error handling

- `restore_from_url` on decode failure: logs warning, returns `no_update` (unchanged).
- `apply_globals` / `apply_tab_{tab}` when `snapshot-state-store` is `None` (no snapshot in session): returns all `no_update`. Normal page load, no snapshot in URL.
- `apply_tab_{tab}` when state is populated but contains no keys for this tab: returns all `no_update`. Covers legacy all-tabs links on non-visited tabs and new active-tab-only links on non-active tabs.

## Tests

### Delete
- `test_snapshot.py::test_tab_filter`
- `test_snapshot.py::test_tab_filter_encodes_only_matching`
- `test_snapshot.py::test_each_tab_filter_roundtrips`
- `test_snapshot.py::test_single_tab_shorter_than_all`
- Any test that imports `_EAGER_CONTROLS`, `_TAB_LAZY_CONTROLS`, or `_BUBBLE_LAZY_CONTROLS`.
- Any test that references the `snapshot-apply-*` relay store IDs.

### Add
- `test_encode_always_active_tab_only`: calling `_encode_snapshot(state)` without tab_filter encodes only active tab's controls + globals (via the encode callback's new default). Assert other-tab keys are null.
- `test_apply_globals_writes_only_global_cids`: Fake state dict with mixed keys; assert `apply_globals` outputs cover only the 31 global cids, never a per-tab cid.
- `test_apply_tab_is_noop_when_state_none`: Set `snapshot-state-store` to None; invoke `apply_tab_bubble`; assert all outputs are `no_update`.
- `test_apply_tab_partial_restore_on_legacy_payload`: Simulate a q3: state dict containing keys for bubble AND heatmap. Invoke `apply_tab_heatmap` with `hm-first-render` change; assert heatmap keys get written. Documents the cross-tab-on-visit feature.
- `test_first_render_bump_input_is_snapshot_state_store`: Static assertion on `routing.py` source/callback graph that the post-snapshot first-render bump callback has `snapshot-state-store` as its Input (guards the spec invariant).
- `test_no_double_write_partition`: Programmatically verify that the union of global-cids and per-tab-cids equals the original `_SNAPSHOT_CONTROLS` cids, and the intersection is empty. Protects against config-modal controls drifting into per-tab lists.

### Preserve
- All v1/v2/v3 decode-roundtrip tests.
- All checklist-bitmask tests.
- All MC-controls snapshot tests.

## Rollout

- Single PR / single commit cluster. No feature flag.
- Deploy to prod immediately after tests pass and dash-callback-reviewer signs off on the diff.
- Legacy share links continue to work (partial-then-full restore across visits). No user-facing migration notice required.
- Monitor prod logs for ~24h for any "nonexistent object" errors that would indicate a missed control in the partition.

## Out of scope

- Collapsing the 2–3 chart-callback redraws per restore into one. Deferred to follow-up spec `single-redraw-per-snapshot` (see `memory/parked_single_redraw_brainstorm.md`).
- Changing the hash encoding or adding a `q4:` version.
- Removing any of the v1/v2 legacy decoders.
- Simplifying `_CHECKLIST_OPTIONS` bitmask infrastructure.
