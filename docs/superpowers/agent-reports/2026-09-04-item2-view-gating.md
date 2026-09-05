# Item 2 — Tab-1 hidden-view figure gating

Agent report, 2026-09-04. Scope: gate `update_bub_cagr` / `update_bub_resid` /
`update_bub_pctile` on `bub-view-mode`, matching the gate `update_bub_occ`
already ships. New test file `btc_web/test_bub_view_gating.py`.

## 1. Progress-bar analysis (does the gate strand `bub-cagr-progress-*`?)

**Conclusion: no. The bar cannot be left stuck, and the gate makes it strictly
less likely, not more.**

The bar is a pure clientside pair in `btc_web/callbacks/routing.py`, under the
header `# CAGR progress bar — show on input change, hide on figure arrival`
(routing.py:467):

- **SHOW** — routing.py:471-501. Inputs are `bub-cagr-fwd-yrs.value`
  (routing.py:499) and `bub-view-mode.data` (routing.py:500). Its first
  statement is `if (view_mode !== 'cagr') return ...no_update;`
  (routing.py:473), so **the bar is never shown while the CAGR view is
  hidden** — i.e. never shown by any of the runs the new gate skips.
- **HIDE** — routing.py:503-525. Its only Input is `bub-cagr-graph.figure`
  (routing.py:522). It clears `window._cagrProgressTimer`, drives the bar to
  100% and hides `bub-cagr-progress-wrap` after 300 ms. **The hide trigger is
  the figure's arrival and nothing else** — no skipped server run was carrying
  any other hide responsibility (the `bub-cagr-loading` Store, layout/common.py:662,
  is written by both clientside callbacks but read by nothing:
  `grep -rn 'bub-cagr-loading'` finds only these two Outputs and the Store
  itself).

The switch into CAGR still builds, because `bub-view-mode.data` is an **Input**
of `update_bub_cagr` (charts/__init__.py:678) and the new gate sits *after* the
snapshot gate but *before* the build (charts/__init__.py:704). So on a switch
to CAGR the same store write fans out to both callbacks: SHOW paints the bar,
`update_bub_cagr` passes the gate and returns a figure, HIDE fires. Every
writer of `bub-view-mode.data` is a callback Output, so all three switch paths
re-trigger the build:

| writer | file:line |
|---|---|
| pill bar `toggle_bub_view` | charts/__init__.py:542 |
| deep link `deep_link_bub_view` (`/1.2`…`/1.5`) | routing.py:379 |
| snapshot restore `apply_tab_bubble` | snapshot_cb.py:266-268 (dynamic `Output(cid, prop, allow_duplicate=True)` over `_SNAPSHOT_CONTROLS`, which lists `("bub-view-mode","data")` at snapshot.py:308) |

The one theoretical residue is "figure identical to the one already in the
Graph → does the client re-dispatch the HIDE callback?". Checked in the shipped
renderer rather than assumed: `dash_renderer.dev.js:82912-82920`
(`executedCallbacks` observer) builds `requestedCallbacks` from
`keys(props)` of the response via `getCallbacksByInput`, with **no deep-equality
check** against the previous value — any prop present in a callback response
re-triggers its observers, identical value or not. `no_update` outputs are
dropped server-side and so are absent from `props`. Therefore:

- gated run while hidden → no `figure` key → HIDE does not fire → but SHOW
  never fired either (view_mode !== 'cagr'), so nothing to strand;
- run on the switch into CAGR → `figure` key always present → HIDE always fires.

Net effect of the gate on this subsystem: before, a hidden-view rebuild kept
the graph's figure in sync while invisible, so switching to CAGR could return a
byte-identical figure; after, the figure is more often genuinely different.
Either way HIDE fires. **No change required to the progress bar.**

## 2. Diff

```diff
diff --git i/btc_web/callbacks/charts/__init__.py w/btc_web/callbacks/charts/__init__.py
index ef5ffb2..810c2e2 100644
--- i/btc_web/callbacks/charts/__init__.py
+++ w/btc_web/callbacks/charts/__init__.py
@@ -699,6 +699,10 @@ def update_bub_cagr(view_mode, _first_render, sel_qs, adv_qs, xrange,
     # Snapshot gate — see spec 2026-04-24-single-redraw-per-snapshot.
     if snapshot_pending:
         return dash.no_update
+    # The graph is hidden in every other mode; don't rebuild it on each
+    # x-range / model tick while the user is looking at something else.
+    if view_mode != "cagr":
+        return dash.no_update
     from utils import _get_cagr_fig
 
     toggles = toggles or []
@@ -761,6 +765,10 @@ def update_bub_resid(view_mode, xrange, toggles, xscale, model_show,
     # Snapshot gate — see spec 2026-04-24-single-redraw-per-snapshot.
     if snapshot_pending:
         return dash.no_update
+    # The graph is hidden in every other mode; don't rebuild it on each
+    # x-range / model tick while the user is looking at something else.
+    if view_mode != "resid":
+        return dash.no_update
     from utils import _get_resid_fig
     toggles = toggles or []
     xrange = xrange or [2010, 2033]
@@ -810,6 +818,10 @@ def update_bub_pctile(view_mode, xrange, toggles, xscale, model_show,
     # Snapshot gate — see spec 2026-04-24-single-redraw-per-snapshot.
     if snapshot_pending:
         return dash.no_update
+    # The graph is hidden in every other mode; don't rebuild it on each
+    # x-range / model tick while the user is looking at something else.
+    if view_mode != "percentile":
+        return dash.no_update
     from utils import _get_pctile_fig
     toggles = toggles or []
     xrange = xrange or [2010, 2033]
```

The comment wording is copied verbatim from the shipped `update_bub_occ` gate
(charts/__init__.py:873-876) so the four callbacks read identically. Placement
is after the `snapshot_pending` early-return in all four.

## 3. Tests

New file `btc_web/test_bub_view_gating.py` (13 tests):

- `test_snapshot_pending_still_wins` ×4 — `snapshot_pending=True` with the
  callback's *own* mode still returns `dash.no_update` (the pre-existing gate
  stays first).
- `test_other_view_modes_return_no_update` ×4 — each callback called with each
  of the other four modes returns `dash.no_update` (identity check,
  `is dash.no_update`).
- `test_own_view_mode_builds_a_figure` ×4 — own mode returns a `go.Figure`
  with `len(fig.data) > 0`.
- `test_all_four_view_callbacks_take_view_mode_as_input` — looks up
  `bub-{cagr,resid,pctile,occ}-graph.figure` in
  `{**dash._callback.GLOBAL_CALLBACK_MAP, **_app_ctx.app.callback_map}` and
  asserts `("bub-view-mode","data")` is among each callback's `inputs`. This is
  the regression guard: demote that Input to a State and the gate would strand
  a stale figure in a view the user switches into.

TDD order was honoured. Run **before** the implementation (occupancy row
already green, proving the test shape matches shipped behaviour):

```
FAILED btc_web/test_bub_view_gating.py::TestHiddenViewsSkipTheBuild::test_other_view_modes_return_no_update[cagr]
FAILED btc_web/test_bub_view_gating.py::TestHiddenViewsSkipTheBuild::test_other_view_modes_return_no_update[resid]
FAILED btc_web/test_bub_view_gating.py::TestHiddenViewsSkipTheBuild::test_other_view_modes_return_no_update[percentile]
3 failed, 10 passed in 0.40s
```

Failure reason was the intended one — a `Figure(...)` came back where
`dash.no_update` was expected — not an import or arity error.

Runs after the implementation:

```
btc_web/test_bub_view_gating.py                        13 passed in 0.31s
test_occupancy + test_bub_deep_links + test_callbacks + test_figures
                                       1 failed, 427 passed, 5 skipped in 17.65s
full suite (btc_venv/bin/python3 -m pytest -q)
                                    1 failed, 2889 passed, 10 skipped, 17 warnings in 24.56s
```

The single failure in both runs is the known pre-existing
`test_callbacks.py::TestBTCPayPricing::test_free_tier_all_models`
(`is_free_tier('lppl', 40, 2028, 10)` is False) — unrelated to this change and
red before it.

## 4. Uncertain / worth a second pair of eyes

1. **First-render build of the CAGR figure is now skipped in Price view.**
   `update_bub_cagr` also takes `Input("bubble-first-render","data")`
   (charts/__init__.py:679). Previously a plain `/1` load pre-built the CAGR
   figure in the background, so the first pill click found it warm; now the
   first click pays the build (which is exactly the CPU the item is reclaiming,
   and is what the progress bar exists for). `_get_cagr_fig` is still L1/L2
   cached, so it is one build per param set per worker, not per click. Flagging
   it as an intentional latency-for-CPU trade rather than a defect.
2. **Not exercised in a browser.** The analysis in §1 is source- plus
   renderer-level; I did not drive a live dev server through
   price→CAGR→price→CAGR with and without input changes. If the controller
   wants belt-and-braces, that click sequence with the progress overlay visible
   is the one to watch.
3. **`bub-view-mode` is snapshot-encoded sparsely** (default `'price'`,
   snapshot_defaults.py:79), so a share link for the Price view omits the field
   and no build is triggered for any sub-view — correct, but it means a
   restored link into a *non-default* view relies on `apply_tab_bubble` writing
   the field, which the §1 table covers.
