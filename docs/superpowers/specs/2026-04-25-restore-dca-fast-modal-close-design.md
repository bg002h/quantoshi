# Phase 2 — `/3` (DCA) Fast Modal Close — Design Spec

**Date:** 2026-04-25
**Status:** Approved for implementation plan
**Builds on:** Phase 1 (commit `b4895ec`, deployed to prod)

## Goal

Extend the bubble fast-modal-close pattern to `/3` (DCA) share links. Pre-build the DCA figure inside `restore_from_url`, deliver it via an always-mounted Store + clientside `set_props` relay, write `active-chart-committed` so the existing direct modal-close listener fires within ~3-5 s instead of falling back to the 7 s timer.

This is one of six per-tab ships planned in user-suggested order: **3 → 4 → 5 → 7 → 2 → 6.** Each tab ships and verifies in dev + prod independently. No speculative scaffolding for the other five.

## Non-goals

- Other tabs (4 retire, 5 supercharge, 7 leverage, 2 heatmap, 6 citadel) — separate ships.
- Saylor-live fast path — needs a server-side BTC price source not currently available inside `restore_from_url`. Falls back to 7 s timer; revisit later if SC-live `/3` shares prove common.
- MC-enabled fast path — paid feature; falls back to 7 s timer. Same fallback strategy as bubble's CTA-active case.
- Refactor of Phase 1's silent-pass tests (`test_no_unguarded_duplicate_outputs`, `test_restore_from_url_uses_intermediate_store`).
- Restructuring `_POST_RESTORE_TRIGGERS` to module scope — keep the inline-in-function-body pattern from Phase 1 for consistency.

## Architecture

### Five changes

#### 1. New always-mounted Store

`btc_web/layout/__init__.py`, alongside `restore-bubble-fig`:
```python
dcc.Store(id="restore-dca-fig", storage_type="memory", data=None),
```

#### 2. New helper `_build_dca_figure_from_state(state)` in `btc_web/restore_builder.py`

Mirrors `update_dca`'s param construction (line 958 of `callbacks/charts/__init__.py`). Reads ~50 dca-* and shared (lppl-*, hybppl-*, eppl-*) state values from the decoded snapshot dict, resolves master keys via the existing `_resolve_*_master` helpers, then calls `_get_dca_fig(...)` from `utils.py`.

**Returns `None`** (caller falls back to existing chart-callback path) when:
- `dca-mc-enable=True` — MC fallback. MC is paid + complex; not worth replicating in builder.
- `dca-sc-enable=True AND dca-sc-entry-mode=="live"` — Saylor-live needs `sc_live_price` from `btc-price-store`, which is a clientside-only Store. Builder has no live price source.

**Critical:** the builder must NOT attempt to read `btc-price-store` from the state dict. There is no such key in snapshots. The `sc_live_price` arg passed to `_get_dca_fig` is `0` (unused on non-live SC paths and on non-SC paths).

**State dict keys to read** (mirror update_dca's signature line-for-line):
- All `dca-*` Inputs/States: `dca-stack`, `dca-use-lots`, `dca-amount`, `dca-freq`, `dca-infl`, `dca-yr-range`, `dca-disp`, `dca-toggles`, `dca-legend-pos`, `dca-qs`, `dca-qs-adv`, `dca-qs-mode` (note: this is a State at line 953, easy to miss — affects which quantile-resolution path the builder takes), `dca-sc-*` (10 keys), `dca-mc-enable` (gate), `dca-model-show`.
- Shared model-config States: `lppl-n-freqs`, `lppl-weighted`, `lppl-no-13`, `hybppl-cfg-{a,b}-*` (14 keys), `eppl-cfg-{a,b}-*` (14 keys), `palette-store`, `user-model-store`.
- Snapshot `_lots` and `_user_model` (or equivalent).

**MC params construction** (the param-dict shape `_get_mc_or_cached` expects):

`_get_mc_or_cached` (`utils.py:149-175`) does `p.pop("mc_cached", None)` and `p.pop("mc_free_tier", False)` (both have pop defaults — safe if absent). Then `mc_active = always_mc or (p.get("mc_enabled") and _app_ctx._HAS_MARKOV)`. When the builder runs, `dca-mc-enable=False` is the gate precondition, so `mc_active=False` and `_get_mc_or_cached` strips ALL `mc_*` keys from p before quantizing the cache key. **The builder therefore only needs `mc_enabled=False` in the params dict** — no other `mc_*` keys, no `_mc_setup` invocation, no `mc_p` stub. Pass `mc_enabled=False` (Python keyword to dict key — set explicitly: `dict(..., mc_enabled=False, ...)`).

**Return shape:** `_get_dca_fig` always returns `(fig, mc_result)` per its type annotation (`figures/dca.py:163`: `tuple[go.Figure, dict | None]`). Builder unpacks: `fig, _ = _get_dca_fig(params)` and returns `fig`.

**Error handling:** wrap body in `try/except`. Any exception logs a warning and returns `None`. Same pattern as bubble at `snapshot_cb.py:91-94`.

#### 3. `restore_from_url` extended

`btc_web/callbacks/snapshot_cb.py:44-100`:
- Add 6th Output (last position): `Output("restore-dca-fig", "data", allow_duplicate=True)`. Function now returns 6-tuple.
- **Early-exit returns at lines 62 (`if not hash_str`) and 67 (`if not state`) must also become 6-tuples** — currently they return 5 `no_update`s; spec change requires 6.
- New branch: when `active_tab == "dca"`, call `_build_dca_figure_from_state(state)`. If non-None, return figure to position 5 (the new Output, last position) AND write `active-chart-committed = hash_str` (position 4 — unchanged) so the modal-close listener fires.
- Add `[trace] restore-dca-build BUILT Xms` instrumentation around the builder call (mirrors `[trace] restore-direct-build BUILT` for bubble at line 98).

**Tuple positions (canonical):**
| Position | Output |
|---|---|
| 0 | `snapshot-state-store.data` |
| 1 | `loaded-hash-store.data` |
| 2 | `snapshot-pending.data` |
| 3 | `restore-bubble-fig.data` (Phase 1) |
| 4 | `active-chart-committed.data` |
| 5 | `restore-dca-fig.data` (NEW — Phase 2) |

#### 4. New clientside relay

`btc_web/callbacks/snapshot_cb.py`, immediately after the existing `restore-bubble-fig` relay block (around line 841). Full registration (verbatim from the bubble pattern, with `bubble`→`dca`):
```python
_app_ctx.app.clientside_callback(
    """
    function(fig) {
        var NU = window.dash_clientside.no_update;
        if (fig == null) return NU;
        try { window.dash_clientside.set_props('dca-graph', {figure: fig}); }
        catch (e) { console.warn('restore-dca-fig: set_props failed', e); }
        if (window.__qsTrace) window.__qsTrace('restore-dca-fig delivered');
        return null;  // self-clear
    }
    """,
    Output("restore-dca-fig", "data", allow_duplicate=True),
    Input("restore-dca-fig", "data"),
    prevent_initial_call=True,
)
```

**Critical:** the Output MUST be `restore-dca-fig.data` (the always-mounted Store), NOT `dca-graph.figure` directly. `dca-graph` is inside `dca-lazy` and absent on `/1`/`/2`/`/4`/`/5`/`/6`/`/7` initial loads — using it as a registered Output would re-introduce the Phase 1 dispatch-drop bug. The whole point of the relay is that `set_props` is the path that bypasses the registered-Output existence check.

Pattern identical to bubble: self-clear via `return null`, guard `if (fig == null) return NU`, try-catch insurance for `plotly/dash#2897`, `prevent_initial_call=True`.

#### 5. `update_dca` post-restore short-circuit

`btc_web/callbacks/charts/__init__.py:871-1054`:
- Add 2 new States to the decorator: `State("active-chart-committed", "data")` + `State("loaded-hash-store", "data")`.
- Define `_POST_RESTORE_TRIGGERS_DCA` inside `update_dca` body (function-local, mirroring Phase 1's `_POST_RESTORE_TRIGGERS` at line ~190 of `update_bubble`):

```python
_POST_RESTORE_TRIGGERS_DCA = {
    "dca-first-render", "dca-stack", "dca-use-lots", "dca-amount",
    "dca-freq", "dca-infl", "dca-yr-range", "dca-disp", "dca-toggles",
    "dca-legend-pos", "dca-qs", "dca-qs-adv",
    "lppl-n-freqs", "lppl-weighted", "lppl-no-13",
    "hybppl-commit-trigger", "eppl-commit-trigger",
    "dca-sc-enable", "dca-sc-loan", "dca-sc-rate", "dca-sc-term",
    "dca-sc-type", "dca-sc-repeats", "dca-sc-entry-mode",
    "dca-sc-custom-price", "dca-sc-tax", "dca-sc-rollover",
    "dca-mc-enable", "dca-mc-bins", "dca-mc-regime", "dca-mc-sims",
    "dca-mc-years", "dca-mc-window", "dca-mc-start-yr", "dca-mc-entry-q",
    "dca-model-show", "dca-mc-model-src",
    # NOTE: `dca-mc-loaded` deliberately excluded so MC async completion
    # can rebuild the chart after the post-restore window.
    # Tuple-size invariant: keep `(dash.no_update,) * 8` aligned with
    # update_dca's 8 Outputs (figure, mc-results, mc-status, mc-rendered-key,
    # mc-save-modal, mc-save-tab, mc-unblocked, yr-range). If a 9th Output
    # is added, this guard's return tuple must grow too.
}
```
**37 entries** — every Input from the @callback decorator (lines 880-946 of `callbacks/charts/__init__.py`) EXCEPT `dca-mc-loaded`. Counting: 12 `dca-*` core + 5 shared model (`lppl-*`, `*-commit-trigger`) + 10 `dca-sc-*` + 8 `dca-mc-*` + 2 `dca-model-*` = 37.

- Add early-return guard immediately after the existing `snapshot_pending` gate (around line 978). The new States `active_chart_committed` and `loaded_hash` are appended to the function signature AFTER all existing positional args (specifically: after `snapshot_pending=False` at the end of the signature, mirroring `update_bubble`'s pattern at `charts/__init__.py:75-160`). Both default to `None`:
```python
def update_dca(_first_render, ..., snapshot_pending=False,
               active_chart_committed=None, loaded_hash=None):
    ...
    if snapshot_pending:
        return (dash.no_update,) * 8
    # Phase 2: post-restore short-circuit — see _POST_RESTORE_TRIGGERS_DCA
    _trg = ctx.triggered_id
    if active_chart_committed and active_chart_committed == loaded_hash \
            and _trg in _POST_RESTORE_TRIGGERS_DCA:
        return (dash.no_update,) * 8
```
The 8-tuple matches `update_dca`'s 8 Outputs.

**`dca-build-count` Store** (used only by E2E test 9 to deterministically detect phantom rebuilds):
- Declared in `btc_web/layout/__init__.py` alongside `restore-dca-fig`:
```python
dcc.Store(id="dca-build-count", storage_type="memory", data=0),
```
- Incremented via a **pure clientside callback** that watches `dca-graph.figure`. No server-side change to `update_dca`. Place near the other clientside callbacks in `callbacks/snapshot_cb.py`:
```python
_app_ctx.app.clientside_callback(
    """function(fig, cur) { return (cur || 0) + 1; }""",
    Output("dca-build-count", "data", allow_duplicate=True),
    Input("dca-graph", "figure"),
    State("dca-build-count", "data"),
    prevent_initial_call=True,
)
```
- Counter increments any time `dca-graph.figure` mutates, regardless of source (relay `set_props`, server `update_dca` return, or any future writer). For a `/3` share link with a working post-restore guard, expected count is **exactly 1** (single delivery via the relay). If the guard fails and the cascade rebuilds, count becomes 2+.
- E2E test 9 reads the Store via `page.evaluate("() => window._dashprivate_layout && (function find(c){if(!c)return;if(c.props && c.props.id==='dca-build-count')return c.props.data;var ch=c.props && c.props.children;if(Array.isArray(ch))for(var i=0;i<ch.length;i++){var r=find(ch[i]);if(r!==undefined)return r;}else return find(ch);})(window._dashprivate_layout)")`. (Or simpler: use Dash's built-in store API — see test code in plan.)
- This approach keeps `update_dca`'s 8-Output signature unchanged. The post-restore guard returns `(dash.no_update,) * 8` (NOT 9).

The existing **clear-on-user-input clientside listener** (`snapshot_cb.py:863-889`) already nulls `active-chart-committed` on first DOM interaction — same gate handles `/1` AND `/3`.

### Data flow

#### `/3` non-MC, non-Saylor-live share (the fast path)

1. `restore_from_url` decodes hash, calls `_build_dca_figure_from_state(state)` → figure.
2. Returns 6 outputs in one HTTP response: `snapshot-state-store=state`, `loaded-hash-store=hash`, `snapshot-pending=True`, `restore-bubble-fig=no_update`, `active-chart-committed=hash`, `restore-dca-fig=fig`.
3. Browser receives response. `set_props('dca-graph', {figure: fig})` from clientside relay → Plotly renders DCA chart.
4. Modal-close listener fires on `active-chart-committed` → modal closes (~3-5 s).
5. `apply_globals` + `apply_tab_dca` cascade with real state → snapshot-pending flips False.
6. Cascade fires `update_dca` via dca-* widget Input changes. Post-restore guard: `active_chart_committed == loaded_hash AND _trg in _POST_RESTORE_TRIGGERS_DCA` → return `(no_update,) * 8`. Phantom rebuild suppressed.
7. User clicks anywhere → clear-on-input listener nulls `active-chart-committed` → gate cleared, steady-state edits proceed normally.

#### `/3` MC-enabled or Saylor-live share (fallback path)

1. `restore_from_url` decodes, builder returns None.
2. Returns: state, hash, True, no_update (bubble), no_update (committed), no_update (dca).
3. snapshot-state-store written → cascade proceeds → apply_tab_dca writes dca-* widgets → snapshot-pending=False → update_dca builds figure normally (with MC overlay if MC params valid).
4. Modal closes via 7 s timer fallback (no `active-chart-committed` to fire fast path).

## Error handling

| Scenario | Behavior |
|---|---|
| Builder raises exception | try/except logs warning, returns None, falls through to cascade path |
| Builder returns None for any reason | Cascade path (snapshot-state-store + apply_globals + apply_tab_dca + update_dca) handles restore |
| `set_props` throws (lazy mount of `dca-graph`) | try/catch in clientside relay logs warning; figure not delivered. Cascade still proceeds. Modal closes via 7 s timer. |
| `_get_dca_fig` returns figure with `data=[]` (model unavailable) | Builder returns the empty figure. User sees an empty chart for ~1s until cascade rebuilds with proper params. Acceptable degradation. |

## Testing plan

### Unit tests (extend `btc_web/test_restore_builder.py`)

| # | Name | Asserts |
|---|---|---|
| 1 | `test_dca_basic_returns_figure` | minimal state (`main-tabs:active_tab=dca`), returns Plotly figure with at least one trace whose `.name` matches `r"Q\d"` (quantile pattern) |
| 2 | `test_dca_mc_enabled_returns_none` | state with `dca-mc-enable=["yes"]` (decoded list, not bitmask integer), returns None |
| 3 | `test_dca_sc_live_returns_none` | state with `dca-sc-enable=["yes"]` + `dca-sc-entry-mode="live"`, returns None AND no exception. Builder must NOT attempt to read `btc-price-store` from state. |
| 4 | `test_dca_sc_custom_returns_figure` | state with `dca-sc-enable=["yes"]` + `dca-sc-entry-mode="custom"` + `dca-sc-custom-price=50000`, returns figure |
| 5 | `test_dca_with_lots` | state with `_lots=[...]` + `dca-use-lots=["yes"]`, returns figure (lots respected — lots-resolution mirrors bubble) |

### E2E tests (`btc_web/test_restore_phase2_dca_e2e.py` — new file)

| # | Name | Asserts |
|---|---|---|
| 6 | `test_dca_share_fast_modal_close` | load `/3#q4:...dca-amount=999`. Modal closes <5 s. `dca-amount` widget reads `999`. `dca-graph` non-empty. |
| 7 | `test_dca_mc_share_falls_back` | load `/3#q4:...dca-mc-enable=yes`. Wait for `dca-graph` figure to mutate from initial empty value. Assert elapsed `<12 s` (no false-positive on cache hit). Modal eventually closes. |
| 8 | `test_dca_sc_live_falls_back` | same pattern as #7 for Saylor-live. |
| 9 | `test_dca_no_phantom_rebuild` | NEW MECHANISM — `dca-build-count` Store + clientside increment on `dca-graph.figure` change (architecture section). Load `/3#q4:...dca-amount=999`. Wait until figure renders (count >= 1). Wait additional 1.5 s for any post-restore cascade. Read Store via `page.evaluate`. Assert `count == 1` (single delivery via relay; guard suppressed all cascade rebuilds). If `count >= 2`, the post-restore guard failed. |
| 10 | `test_dca_yr_range_restored` | load `/3#q4:...dca-yr-range=[2030,2040]`. Assert post-restore the dca-yr-range slider value is exactly `[2030, 2040]`. Verifies no spurious yr-range adjustment writes. |
| 11 | regression: `/1` bubble fast-path | reuse the existing `test_bubble_share_still_restores` from `test_restore_phase1_e2e.py` — DON'T duplicate. |

### Cold-cache timing probe

- Add `[trace] restore-dca-build BUILT Xms` server-side instrumentation.
- Manual probe before commit: `systemctl restart quantoshi` then load representative `/3` share with cold L1 cache. Confirm BUILT `<500 ms` median across 3-5 reps for varying yr-range/SC settings. (Higher than bubble's 300 ms threshold because DCA simulates years of paths.)
- If consistently `>500 ms`: revisit option (a) — may need to fall back to existing path for SC-enabled too, or accept slower modal close.

## Verification gates (in plan order)

1. Unit tests pass.
2. E2E tests pass against running dev server.
3. Full unit suite still passes (no regression).
4. Dev `?trace=1` Playwright probe shows:
   - `[trace] restore_from_url prefix=q4: controls=N` for `/3` load
   - `[trace] restore-dca-build BUILT <500> ms` for non-MC non-SC-live shares
   - NO `[trace] dca-fig BUILT` lines (post-restore guard suppresses cascade)
5. Single commit (architect-recommended atomicity for this kind of multi-file change).
6. Prod deploy + prod E2E probe — same checks against `quantoshi.xyz/3`.

## Files touched

| File | Change |
|---|---|
| `btc_web/layout/__init__.py` | +1 line: new `restore-dca-fig` Store |
| `btc_web/restore_builder.py` | +~150 lines: `_build_dca_figure_from_state` |
| `btc_web/callbacks/snapshot_cb.py` | ~10 line edits in `restore_from_url` (add 6th Output, dca branch); +~15 line clientside relay |
| `btc_web/callbacks/charts/__init__.py` | +2 States in `update_dca` decorator; +~38 line set + 3 line guard inside function body; +`dca-build-count` Store increment for the no-phantom-rebuild test |
| `btc_web/test_restore_builder.py` | +~80 lines: 5 DCA unit tests |
| `btc_web/test_restore_phase2_dca_e2e.py` | new file ~150 lines: 5 E2E tests |
| `memory/restore_callback_architecture.md` | +Phase 2 (DCA) entry after Phase 1 section. Must contain: (a) which paths now use the fast path (`/1`, `/3`); (b) Store name (`restore-dca-fig`); (c) builder fallback conditions (MC enabled, Saylor-live); (d) `dca-build-count` test instrumentation pattern; (e) link to spec + commit hash; (f) measured prod latency for `/3`. |
| `docs/architecture.md` | Update the existing "Restore performance architecture" section: change "Citadel + other non-bubble share-link tabs fall back to the existing callback cascade" to list `/3` separately (now fast). Add 2-3 sentence description of the per-tab Store relay pattern as the canonical extension mechanism. |

## Risks

1. **Cold-cache build time** could exceed 500 ms threshold. Probe before commit. If so, narrow option (a) to also include SC-enabled (similar to MC).
2. **`dca-build-count` Store + clientside increment** is a new test infrastructure pattern. Phase 1 didn't need it; Phase 2 uses it because journal-grep is flaky under gunicorn 5-worker prod. Risk: the increment itself could silently break (e.g., if added to wrong code path). Mitigate with a tiny unit test that asserts the increment fires on a real build path but not on snapshot-pending or post-restore short-circuit.
3. **MC bookkeeping divergence** if a future change to `update_dca`'s 8 Outputs (e.g., adding a 9th) breaks the `(no_update,) * 8` guard. Mitigate with a comment near `_POST_RESTORE_TRIGGERS_DCA` flagging the tuple-size invariant.
