# Review r2 — Tab-1 Raw MC implementation plan after fold (`c8784fc7`)

**Counts: 0 Critical · 3 Important · 6 Minor · 3 Nit**

Scope: (1) is each round-1 finding resolved in the plan text, (2) did the fold add a wrong
citation, an inconsistent name between tasks, a test that cannot fail or pass, or a Dash 4.0
shape error. No fresh audit; the spec is not re-reviewed. Nothing in the repo was modified.

## Machine-checked (working tree at `c8784fc7`)

- `_resolve_mc_model_src` is in `callbacks/charts/_resolvers.py:229` and passes unknown keys
  through. `_TAB_CONTROLS` is at `callbacks/routing.py:141`. `_POST_RESTORE_TRIGGERS` is in
  `update_bubble` (`callbacks/charts/__init__.py:238`). `apply_zoom_lock` is
  `figures/common.py:494`. `_add_mc_fan(label_suffix=)` and `_add_mc_spaghetti(n_display=)`
  exist in `figures/bubble.py`. `qs_keep_width` is an existing meta flag. `_MODEL_FP` and
  `M.price_dates` / `M.price_prices` exist. scipy, numpy and pandas are in
  `btc_web/requirements.txt`. `dash_nocache.js` exists.
- `_mc_payment_initiate` reads `model_srcs[tab_idx]` at `callbacks/mc_payment.py:103-104`, so
  the guard position is valid.
- `_MC_MATCH_JS_TPL`, `MC_AUTOWIDEN_JS` (5 params, 1 Input + 4 States) and `MC_REGIME_PATHS_JS`
  (3 params, 1 Input + 2 States) are where the plan says. `_MC_MATCH_JS_TPL` is `.format()`-ed
  and its return order is `[text, style, overlay, wrapClass, badge, restoreBtn]`, matching the
  plan's neutral vector.
- `_SNAPSHOT_CONTROLS` is 340 long with `bub-mc-paths` at 339. The snapshot apply callbacks are
  generated from `_TAB_CONTROLS`, so adding the ids to `routing.py` is sufficient.
- Default `bub-xrange` is `[2010, 2033]` and the Tab-1 MC start year defaults to 2028. So the
  Task 8 distinguisher (Raw `max(2033, 2036)` vs model `max(2033, 2038)`) is real.
- `restore_builder._build_bubble_figure_from_state` already punts to the update_bubble cascade
  when `bub-mc-enable` is on, so a Raw share link renders through `update_bubble`.
- `_HALVING_KNOWN_DATES` is a tuple of strings and `_HALVING_EPOCHS` is a list of
  `(Timestamp, is_estimated)`. Rule is `2024-04-20 + DateOffset(years=4k)`, cut at year 2080.

## Round-1 findings: disposition

| r1 | Resolved in plan text? | Note |
|---|---|---|
| I-1 badge outside `bub-mc-body` | **Partly** | Interfaces now put the class on `bub-mc-panel`, but Task 5 Step 3 still writes `bub-mc-body.className`. See I-A. |
| I-2 shape tests not scheduled | **Partly** | Task 5/6 files are listed, but the `len(result)==8` sites are wrong (see I-B) and Task 7 is not swept (see I-C). |
| I-3 `raw_mc` imports `figures.common` | Yes | Own halving dates, parity test, `test_import_light`. |
| I-4 seed on raw `checked` | Yes | `seed_for` uses `effective_checked`; test pins it. |
| I-5 no filter-on timing | Yes | `test_performance_filtered` added. See M-5 on the bound. |
| I-6 auto-tick stale on restore | Yes | Both trackers updated every call, fresh stamp decides, restore test added. Stale remnants in M-3. |
| M-1 resolver file | Yes | `_resolvers.py`, regression pin only. |
| M-2 `_TAB_CONTROLS` file | Yes | `callbacks/routing.py` ~141. |
| M-3 window default | Yes | Static `[2010, 2026]`, same as `bub-mc-window`. |
| M-4 no-Markov placeholders | **Partly** | 9 ids added; `bub-mc-panel` (new in the I-1 fold) is not. Folded into I-A. |
| M-5 zoom lock | Yes | Step 3 + `test_raw_zoom_lock_applied`. |
| M-6 payment guard position | Yes | After `model_srcs[tab_idx]`, with the >50 000 case. |
| M-7 price source | **Partly** | `update_bubble` path specified; `regime_options` and the layout's initial options are not. See M-1. |
| M-8 spec items without tests | Yes | Hint, labels, status items all have asserts. |
| M-9 regime-options callback | Yes | `prevent_initial_call`, tuple conversion, exp-failure labels. |
| M-10 vacuous E2E | Yes | Upper-bound assert, verified distinguishable. |
| M-11 stale clients | Yes (accepted) | Recorded in Review Focus 4. |
| N-1 `include_raw_src` | Yes | Threaded through `_mc_controls`. |
| N-2 edges/states on result | Yes | `edges`, `states` fields. |
| N-3 line-count constants | Yes | In `raw_mc.py` (but see M-4). |
| N-4 φ at step midpoint | Yes | F-22 decisions. |
| N-5 remaining-span σ | Text yes, test no | See N-2. |
| N-6 seed test comment | Yes | "unfiltered only". |
| N-7 illustrative figures | Yes | Plan header. |
| N-8 `.format()` braces | Yes | Task 6 note. See N-3 for the `%` constants. |

## Important

### I-A — Task 5 Step 3 still targets `bub-mc-body`; the I-1 fix exists only in the Interfaces paragraph
- **Where:** Task 5 Step 3 (plan line 229) says the clientside callback writes
  `Output("bub-mc-body","className")`. The Interfaces paragraph (line 202) says the mode class
  goes on `bub-mc-panel`, an ancestor of the header and body.
- **Failure:** an engineer executing Step 3 puts the class on `bub-mc-body`. The `⚡ Paid` badge
  is in the card header, not a descendant, so it stays visible in Raw mode, which is exactly
  r1 I-1. No unit test catches this:
  - `test_paid_badge_hidden_in_raw_mode` checks only structure (class present, descendant of
    `bub-mc-panel`).
  - `test_raw_mode_class_toggle_js` checks only the returned string.
  - The Task 8 E2E is the first gate that fails.
- **Same fold, same gap:** `bub-mc-panel` is not in the no-Markov placeholder branch, and the
  plan never says to emit it only when `include_raw_src` is set. `_mc_controls` is shared by six
  tabs, so a literal id would duplicate on five of them. A prefixed id would collide with the
  existing `hm-mc-panel` (`layout/heatmap.py:293`). With no Markov engine the clientside
  Output then points at a nonexistent component, the same failure r1 M-4 named for the 9 ids.
- **Fix:** change Step 3 to `Output("bub-mc-panel","className")`. State that the id exists only
  when `include_raw_src` is true, in both the real and the placeholder branches. Add
  `bub-mc-panel` to `test_placeholders_without_markov`. Add one test that reads the registered
  clientside callback's Output and asserts it is `bub-mc-panel.className` and that the
  `⚡ Paid` span is inside that component.

### I-B — Task 5 sends the engineer to change four `len(result) == 8` asserts that must stay 8
- **Where:** Task 5 Files (plan line 197): "the 8 `len(result) == 8` asserts (lines ~1142, 1201,
  1230, 1297, 1499, 1534, 1617, 1650) → 9".
- **Failure:** only 1142, 1201, 1230 (`TestUpdateBubbleCallback`) and 1297
  (`TestUpdateBubbleCallbackMC`) are `update_bubble`. Lines 1499, 1534, 1617 and 1650 are in
  `TestUpdateHeatmapCallback`, `TestUpdateDcaCallback`, `TestUpdateRetireCallback` and
  `TestUpdateSuperchargeCallback`. Those callbacks keep 8 Outputs. Editing them to 9 turns four
  green tests red. The r1 report said "8 sites, all broken by the 9th Output"; that was wrong,
  and the plan inherited it.
- **Fix:** list 1142, 1201, 1230 and 1297 only, and say the other four are untouched.

### I-C — Task 7 breaks an existing test it does not schedule, and Step 4 does not run it
- **Where:** Task 7 Files and Step 4 (plan lines 264, 281).
- **Failure:** `test_mc_paths_toggle.py::TestSnapshotRegistry::test_appended_at_the_absolute_tail`
  asserts `len(_SNAPSHOT_CONTROLS) == 340`. Appending 8 fields makes it 348 and the test goes
  red. Step 4 runs `test_snapshot*.py` and `test_restore_builder.py`, and
  `test_mc_paths_toggle.py` matches neither glob. The red test surfaces only at Task 8 Step 3,
  after the Task 7 commit. This is the r1 I-2 pattern, applied to the one task the fold did not
  sweep.
- **Fix:** add `test_mc_paths_toggle.py` to Task 7 Files. Change the assert to
  `[339] == bub-mc-paths` and `len == 348`, or drop the length pin in favour of the new file's
  tail test. Add the file to Step 4's run list.

## Minor

- **M-1 `regime_options` has no price source (M-7 residual).** Task 3 declares
  `regime_options(bins, window, trend, asof=None)` with `lru_cache`, in a module forbidden to
  import `_app_ctx`. It is called from two places that do not hold prices: the layout's initial
  options and the server options callback. `pd.Series` is unhashable, so the cache cannot key
  on it, and the plan names no data fingerprint. Fix: give raw_mc a small registered price
  source (`set_prices(series, data_fp)` called once at app start, or `regime_options(..., data_fp)`
  with a provider), and use the same object as `simulate`. Otherwise bin labels and the edges
  `simulate` uses can come from different series.
- **M-2 "passes untouched" survived.** Task 5 `test_model_mode_unchanged` still says the
  `TestUpdateBubbleCallbackMC` suite "passes untouched", but its `len(result)` assert (line 1297)
  changes. r1 I-2 asked for the rewording. Fix: "passes, with the output-count assertions
  updated in this commit".
- **M-3 Stale I-6 remnants in Task 6.** The Interfaces line says the decision is "applied to
  whichever checklist triggered", and the shape-test paragraph adds "a stubbed
  `dash_clientside.callback_context`". Both contradict the Note (decide by fresh stamp, never by
  `triggered[0]`). A reader following the Interfaces line rebuilds the r1 I-6 defect. Fix: reword
  the Interfaces line to "applied to each checklist whose click stamp is fresh", and drop the
  stub.
- **M-4 Task 4 Files incomplete.** `RAW_LINES_MIN/MAX/DEFAULT` live in `raw_mc.py` (r1 N-3), but
  Task 4's Files lists only `figures/bubble.py`, `colors.py` and the test. The home of
  `select_line_paths` is not stated. Fix: add `btc_web/raw_mc.py` to Task 4 Files and name the
  module for `select_line_paths`.
- **M-5 Wall-clock asserts under xdist.** The fold set Task 2 `test_performance` to < 0.2 s
  (r1 had 1.0 s) and added < 0.2 s filtered tests. The suite runs `-n logical` (24 workers on
  16 cores). Two contended workers can exceed 0.2 s on an honest vectorised implementation. Fix:
  take the best of 3 runs, mark the tests so they can run serially, and keep the 0.2 s figure as
  a printed target with a looser hard bound (e.g. 0.6 s).
- **M-6 Dead model controls remain in Raw mode.** The model-only list (plan line 202) omits the
  "Advanced simulator options" checkbox (`bub-mc-advanced`) and the Frequency / Periods-per-year
  controls in `bub-mc-adv-body` (Tab 1 does not share them). In Raw mode the checkbox stays
  visible and reveals a Frequency dropdown that does nothing. Not introduced by the fold; found
  while checking I-A. Fix: add both to the `qs-mc-model-only` list.

## Nit

- **N-1** Task 1 `test_halvings_match_figures_common`: `HALVINGS[1:]` holds Timestamps and
  `_HALVING_KNOWN_DATES` holds strings, and `_HALVING_EPOCHS` holds tuples. The comment is
  shorthand; the test must convert through `pd.Timestamp` and filter on `is_estimated`.
- **N-2** The remaining-span σ midpoint for the current cycle (r1 N-5) is stated in F-22
  decisions but still untested. `asof = today = 2024-04-20` is a halving, so remaining span and
  whole cycle coincide. Add one mid-cycle case to `test_sigma_held_constant_per_cycle`.
- **N-3** `MC_AUTOWIDEN_JS` and `MC_REGIME_PATHS_JS` are `%`-formatted (not `.format()`), so any
  literal `%` in the new JS must be doubled. The plan's N-8 note covers only `_MC_MATCH_JS_TPL`.
