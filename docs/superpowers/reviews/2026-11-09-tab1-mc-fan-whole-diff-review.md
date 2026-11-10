Critical: 0 · Important: 1 · Minor: 2 · Nit: 4

# Whole-diff review: Tab 1 MC fan, example paths, sims=200 default, auto-widen

Reviewer: independent (did not write the code). Scope: the uncommitted working-tree diff
(`phase1.diff`, 1569 lines), read in full, plus the surrounding code it calls. No source or
test file was modified. Dev server on :8050 (started 22:02, after the last edit) was used
read-only through a browser.

Verdict: **one blocking finding (Important #1)**, a missing case in the new auto-widen
callback. Everything else in the diff held up under inspection and under runtime probes.

---

## Important

### I-1. Auto-widen fires in the historical-only views and pushes `bub-xrange` above the slider's `max`

- Severity: Important (a missing case in new behaviour that produces an out-of-contract state)
- Where: `btc_web/layout/bubble.py:83-100` (`MC_AUTOWIDEN_JS`), registration
  `btc_web/callbacks/mc_controls.py:199-206`. Conflicts with the cap callback at
  `btc_web/callbacks/charts/__init__.py:665-682`.
- What is wrong: The callback widens whenever the user ticks `bub-mc-enable`, in every Tab-1
  view. The MC panel (`bub-mc-exclude-wrap`) is hidden only by Time Machine, not by the view
  pills. In Residuals / Percentile / Occupancy (`historical=True` in `bub_views.py`) the
  app deliberately caps the X slider at next year (`max` = 2027 today) and clamps its value
  (`charts/__init__.py:669-673`). The new callback writes `[lo, start + 10]` regardless, so
  the slider `value` exceeds its own `max`. MC cannot be drawn in those views (it is added only to
  `bubble-graph`), so the widen buys nothing there and distorts the chart the user is looking at.
- Reproduction (confirmed, run against the dev server on :8050):
  1. Open `http://localhost:8050/1.3` (Residuals view), wait for hydration.
  2. Before: `#bub-xrange [aria-valuenow]` -> 2010 and 2027, `aria-valuemax` = "2027";
     `#bub-resid-graph` `layout.xaxis.range` = `[-0.02777, 1.24148]`; `#bub-mc-enable` checkbox is
     visible.
  3. Click the MC "Activate" checkbox, wait 5 s.
  4. After: `aria-valuenow` -> 2010 and **2038** with `aria-valuemax` still "2027";
     residuals `xaxis.range` -> `[-0.02777, 1.45390]` (the chart now extends into blank future space).
  The same mechanism applies to `/1.4` and `/1.5` (same `historical_modes_js()` gate). Before this
  diff, ticking MC in these views left the range alone.
- Likely same class, unconfirmed: with Custom Time Axis active, `update_bubble` raises
  `PreventUpdate` (CTA owns the figure), yet the widen still fires, so the CTA chart's range
  grows although no MC is drawn.
- Why it is not already caught: `test_mc_autowiden.py` pins the State list to exactly
  `bub-xrange`, `bub-mc-start-yr` (`test_reads_range_and_start_year_as_state`) and the node
  harness has no view-mode case; the E2E test runs only in the default Price view.
- Suggested fix: add `State("bub-view-mode", "data")` (and optionally `State("cta-active", "value")`)
  to the registration; in `MC_AUTOWIDEN_JS` return `no_update` when the mode is historical (reuse
  `bub_views.historical_modes_js()`) or when CTA is on; add a node-harness case per mode and update
  the pinned State list in `TestRegistration`. Also clamp `want` to the live slider max if you prefer
  a view-agnostic guard.

---

## Minor

### M-1. The "live (paid) run" arm of the MC block in `update_bubble` is unreachable, and its new comments/logic describe behaviour that cannot occur
- Where: `btc_web/callbacks/charts/__init__.py:484-485` (`elif mc_result and isinstance(mc_result, dict)`),
  `:510-515` (comment: "A live run really was simulated under the filter, so its fan is the result and
  stays amber"), `:515` (`fan_is_reference = bool(blocked) and is_free`).
- Evidence: `btc_web/figures/bubble.py:830` is `return fig, None`, the only tuple return of
  `build_bubble_figure`, so `mc_result` is always `None` on Tab 1. Runtime probe
  (`update_bubble` via the test helper, cached scenario bub/2028/q10/40y):
  `mc_sims=3200` -> `fan: [] lines: 0` (nothing is drawn, not even the fan).
  This is pre-existing (the old code had the same dead `elif`), but the diff's intent statement and
  new comments rely on a live-run fan. No test can exercise it.
- Impact: none today (sims > 200 is the paid tier and draws nothing before and after the diff). Risk
  is a future reader or maintainer trusting the "live stays amber" claim.
- Fix: either soften the comment ("only reachable once the Tab-1 builder returns an `mc_result`"), or
  file a follow-up for the paid path.

### M-2. Unrelated modified binary in the working tree: `model_data_ef.pkl`
- `git status` shows `M model_data_ef.pkl` (mtime 2026-11-07 12:27, the daily-pipeline rebuild; last
  committed 2026-04-27). It is not part of this change. Commit with explicit paths (per
  `feedback_git_add_scoped`), not `git add -A`, or it ships inside the feature commit.

---

## Nit

- N-1. `test_mc_autowiden.py::TestRegistration::test_asset_listens_in_the_capture_phase`: name claims a
  capture-phase assertion but only checks the substrings `#bub-mc-enable` / `__qsMcEnableClickAt`. The
  capture flag is asserted by `test_listener_stamps_only_clicks_inside_the_mc_enable_checklist`
  (`capture: True`), so there is no coverage gap, just a misleading name.
- N-2. A browser tab left open across the deploy keeps the old cached `chart_responsive.js` (no
  `qs_keep_width` support). It receives the new hairline traces and restyles them to 3.75 px until
  reload. Transient; the stale-deps hazard already documented in CLAUDE.md.
- N-3. The click timestamp lingers up to 3 s if a click on `#bub-mc-enable` does not change the value
  (nothing consumes it); a programmatic write inside that window would widen. Practically unreachable.
- N-4. Unconfirmed (legend is off by default): with the legend turned on, the "MC paths" swatch uses
  the first path's colour, `rgba(48,48,48,0.04)` at 200 paths, so it is nearly invisible.

---

## Checked and found sound

Evidence in parentheses is something I ran or read.

- **Dash registration** (`mc_controls.py:199-206`): exactly one Input, `allow_duplicate=True`,
  `prevent_initial_call=True` (never False), States/Input ids all exist in the layout, `no_update`
  paths cover non-list `mc`, short/None `xrange`, null/NaN start year, a stale or missing click stamp
  (node harness cases in `test_mc_autowiden.py`; 35 passed with `test_chart_responsive.py` and
  `test_snapshot_defaults.py`, `-n0 -p no:randomly`). `%`-formatting of the JS template is clean (no
  stray `%`).
- **Programmatic writes**: the guard is the consumed click stamp, not the value; `set_props` enable
  leaves the range alone (E2E per the brief; logic re-read). Label-click vs input-click ordering means
  the stamp is always set before React's change and never after the callback consumes it.
- **Shared-figure cloning**: `fig = _go.Figure(fig)` precedes both helpers on the only path that adds
  traces. Mutation probe: I replaced `go.Figure(fig)` with identity via a throwaway pytest plugin
  (scratchpad, not in the repo); `test_repeat_render_does_not_accumulate_mc_traces` failed
  (`assert 40 == 8` lines), so that test does detect a missing clone. The other two failures in that
  mutation were collateral (same shared object).
- **Fan vs lines consistency** in `update_bubble`: fan is the percentile of all (truncated) paths; lines
  are ranked/trimmed. Both get the same `keep` column slice, the same `t_axis`, and the same stack
  multiplier. Ranking runs on full-length paths before truncation. Probes:
  xrange 2010-2020 (MC starts 2028) -> 2 points per trace, no error; 2010-2080 -> 481; `mc_sims` of
  0/None -> 100/200 lines; 1 line with a regime filter -> gray "MC ref" fan + 1 line.
- **Falsy-zero / stale-client tolerance**: `mc_sims` 0/None coerce to defaults, no crash; non-list
  `mc_enable` already gated by `isinstance`. (Non-numeric strings still raise in `_ci`, pre-existing
  and unreachable from a number input.)
- **Old share links**: `HEAD`'s fingerprint is `f10ac563` (recomputed with sims=8) and is in the
  registry with sims 8. Decoding a synthetic v4 payload with `{bub-mc-enable: 1}` gives
  `bub-mc-sims = 8` for `f10ac563` and `b98ab5f7`, and 200 for the new `72f9cada`.
- **Free-tier fit**: `MC_FREE_SIMS = 200`, so the new default (200) stays free; the cache lookup no
  longer passes `max_sims`, so the fan sees all 200 paths.
- **Prod runtime imports**: no new third-party import on the `btc_web/**` runtime path; the removed
  `plotly.colors.sample_colorscale` import is no longer referenced; nothing requires Python 3.14.
  `node` is test-only and its tests skip cleanly when absent.
- **`chart_responsive.js`**: `keepsWidth` guards both `needsApply` and `applySettings`, so a leading
  hairline trace cannot cause a restyle on every 500 ms poll (covered by
  `test_a_leading_keep_width_trace_does_not_cause_a_restyle_every_poll`).
- **Fan construction**: band edges get an explicit transparent colour, the reference style gets
  legend names, the shared `_MC_BANDS` / `_GHOST_BANDS` are not mutated (tested), date hover is applied
  to a scratch figure only.
- **Generated colour artifacts** are in sync: `sha256(colors.py)[:16]` = `4a9074bd2943a18d` matches the
  `Source-SHA256` header in `_colors_generated.css`; `test_colors_central.py` 13 passed.
- **Browser re-check on :8050 (Price view)**: ticking MC gave 208 traces (200 example paths, the fan
  named "MC 5-95%", "MC 25-75%", "MC median", plus the chart's own), no console error mentioning the new callback, X range 2010-2038.
- **Test sensitivity**: new `TestUpdateBubbleCallbackMC` tests run (7 passed, none skipped; cache is
  loaded here). They would skip, not falsely pass, on a machine without the MC cache.
