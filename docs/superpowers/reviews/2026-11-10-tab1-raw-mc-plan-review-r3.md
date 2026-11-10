# Review r3 — Tab-1 Raw MC implementation plan after fold 2 (`ca3c2e2f`)

**Counts: 0 Critical · 1 Important · 4 Minor · 3 Nit**

Scope: (1) is each round-2 finding resolved in the plan text, (2) did the fold add a wrong
citation, an inconsistent name between tasks' Interfaces blocks, a test that cannot fail or
cannot pass, or a Dash 4.0 shape error. No fresh audit; the spec is not re-reviewed. Nothing in
the repo was modified (one read-only fingerprint computation with bytecode writing off;
`git status` unchanged: only the pre-existing `model_data_ef.pkl`).

## Machine-checked (working tree at `ca3c2e2f`)

- `test_callbacks.py`: the `update_bubble` shape asserts are at 1142, 1201, 1230
  (`TestUpdateBubbleCallback`) and 1297 (`TestUpdateBubbleCallbackMC`); 1499, 1534, 1617, 1650
  are heatmap / DCA / retire / supercharge. The plan's four-and-four split is correct.
- `test_mc_paths_toggle.py::TestSnapshotRegistry::test_appended_at_the_absolute_tail` asserts
  `len(_SNAPSHOT_CONTROLS) == 340` (line 52); no other test pins the length (others are
  index-addressed at 334-339, which do not move).
- `outer Div` in `_mc_controls` is `html.Div(style={"position":"relative"})` with no className,
  and the `⚡ Paid` badge sits in `header_right` of the `_section_card` inside it, so a class on
  that Div is an ancestor of both header and `bub-mc-body`. `bub-mc-panel` is unused anywhere;
  `hm-mc-panel` exists (`layout/heatmap.py:293`).
- `_mc_controls` no-Markov branch returns `html.Div(style=_STYLE_HIDDEN, children=_ph)` (line
  211); `bub-mc-model-src` is already a placeholder there (line 210). `Frequency` / `Periods per
  year` live in `bub-mc-adv-body` and Tab 1 shares neither. `_mc_model_src_options()` has one
  caller.
- `MC_AUTOWIDEN_JS(mc, xrange, startYr, mode, cta)` is 1 Input + 4 States, registered in
  `callbacks/mc_controls.py:144-155`; `hi = max(xr[1], min(xmax, start + 10))`,
  `lo = max(xmin, currentYear)`. `MC_REGIME_PATHS_JS(checked, bins, paths)` is registered at
  `callbacks/mc_controls.py:162-169` with a primary (non-`allow_duplicate`) Output and
  `prevent_initial_call=True`; the Dash 4.0 bug note (`plot_appearance.py:22-28`) is specific to
  `allow_duplicate` + several Inputs, so two Inputs on this callback is consistent with it.
  `_MC_MATCH_JS_TPL` already takes `mc_model_src` as its 5th Input; no new Input needed.
- `update_bubble` Outputs are 8 (`callbacks/charts/__init__.py:81-88`); every Output except
  `bub-mc-status.children` is `allow_duplicate`; the two short-circuit returns are at lines 229
  and 259. `bub-mc-model-src` is already an Input and in `_POST_RESTORE_TRIGGERS`.
- `test_bub_moving_avg.py::test_positionally_aligned_with_the_signature` and
  `test_mc_paths_toggle.py::test_is_the_last_input_and_maps_onto_mc_paths` require
  `len(decorator args) == len(params)` and pin `mc_paths` as the last param: the plan's "last 8
  Inputs" rewrite covers the second; the first is satisfied if params are appended too (N-2).
- `_mc_payment_initiate`: `model_srcs[tab_idx]` at `callbacks/mc_payment.py:103-104`; the
  `> _MC_QUANT_THRESHOLD` branch follows at ~137, so the guard position is valid.
- `_HALVING_KNOWN_DATES` (4 strings) and `_HALVING_EPOCHS` (cut at `year > 2080`) in
  `figures/common.py:165-184`; the rewritten `test_halvings_match_figures_common` can pass.
- Current fingerprint is `04aa50d6`; it is the last of 19 registry entries, so one more entry
  (20) fits under the cap without evicting.
- `pytest.ini`: `addopts = -n logical --ignore-glob=...`; no `--dist` option anywhere.

## Round-2 findings: disposition

| r2 | Resolved in plan text? | Note |
|---|---|---|
| I-A Step 3 targets `bub-mc-body` | **Yes**, one residual | Step 3 now `Output("bub-mc-panel","className")`; Files: literal id, emitted only with `include_raw_src`, also in the no-Markov branch; `test_mode_toggle_targets_panel` added. The r2-requested placeholder assertion for `bub-mc-panel` was not added (M-1). |
| I-B four `len == 8` sites must stay 8 | Yes | Plan lists 1142 / 1201 / 1230 / 1297 only and says the other four stay 8. |
| I-C Task 7 breaks `test_mc_paths_toggle.py` | Yes (Files gap, M-3) | Paragraph + Step 4 run list added, exact test named, 340 to 348 stated. |
| M-1 `regime_options` price source | **Partly** | `set_prices` / `get_prices` introduced but half-owned; see I-1. |
| M-2 "passes untouched" | Yes | Reworded. |
| M-3 stale I-6 remnants | Yes | Interfaces line and stubbed `callback_context` both gone. |
| M-4 Task 4 Files | Yes | `raw_mc.py` and `select_line_paths` module named. |
| M-5 wall-clock under xdist | **Partly** | Best-of-3 and 0.6 s bound are in; the `xdist_group` marker is inert (M-2). |
| M-6 dead model controls | Yes | `bub-mc-advanced` and the Frequency / Periods-per-year controls added. |
| N-1 halving test conversion | Yes | |
| N-2 remaining-span sigma untested | Yes | Mid-cycle case in `test_sigma_held_constant_per_cycle`. |
| N-3 `%` constants | Yes | Task 6 note covers both `%`-formatted constants. |

## Important

### I-1 — `set_prices` / `get_prices` has no owner for its production call site, no test, and no unset behaviour
- **Where:** Task 1 Interfaces (plan line 62), Task 3 Interfaces (line 144), Task 5 Step 3
  (line 231).
- **Failure:**
  1. Task 1 promises "a test that `get_prices()` returns what was set", but Task 1's Step 1 list
     (lines 67-87) has no such test and Step 3 never mentions the registry. Task 1 says "see
     Task 3"; Task 3 says "Task 1 adds"; neither gives the unset behaviour.
  2. Task 3 says "`app.py` calls `set_prices` once at startup from `_app_ctx.M` with
     `_app_ctx._MODEL_FP`", but `btc_web/app.py` is in no task's Files (Task 3 lists only
     `raw_mc.py` and the test file; Task 5 lists five other files). Under scoped `git add` the
     edit is never staged.
  3. No test pins that startup registers the series. Every Task 3 / Task 5 test registers the
     frozen CSV itself, so all unit tests pass with the production wiring absent; the first gate
     that can fail is the Task 8 E2E, after Tasks 5-7 are committed. If instead `get_prices()`
     raises when unset, `_bubble_controls()` (which builds the initial `bub-mc-raw-regime`
     options on every layout build and is called directly by `test_axes_presets.py`,
     `test_timemachine_layout.py`, `test_mc_paths_toggle.py`) goes red at Task 5 with no file in
     the plan to fix.
  4. Two sources for one fact: Step 3 builds `RawRequest.data_fp` from `_app_ctx._MODEL_FP` but
     takes the series from `get_prices()`, which carries its own `data_fp`. They agree in prod
     and differ in tests (`"frozen"` vs the real value), so the seed and the `regime_options`
     cache key disagree. "`simulate` with `raw_mc.get_prices()`" also reads as passing a tuple
     where `simulate(req, prices: pd.Series)` takes a Series.
  5. Tests that call `set_prices(frozen)` change a process-global that the layout and later
     tests read, with no restore fixture.
- **Fix:** Add `btc_web/app.py` to Task 3 Files (or Task 5), call
  `raw_mc.set_prices(pd.Series(M.price_prices, index=pd.to_datetime(M.price_dates)), _app_ctx._MODEL_FP)`
  right after `_app_ctx.M = M` (`app.py:343`). Add to Task 1 `test_set_get_prices_roundtrip` and
  `test_get_prices_unset` (define: `(None, "")`; `regime_options` then returns plain "Bin 1…n"
  labels; the Raw branch shows a status reason, no exception). Add to Task 5
  `test_startup_registers_prices` (`import app`; `get_prices()[1] == _app_ctx._MODEL_FP` and the
  series length equals `len(M.price_dates)`). In the Raw branch take both series and fp from one
  `get_prices()` call (`prices, fp = get_prices()`). Give the tests that override the registry a
  fixture that restores it.

## Minor

- **M-1 `bub-mc-panel` not in the placeholder test (r2 I-A residual).** Task 5
  `test_placeholders_without_markov` (line 213) asserts "all 9 new ids"; `bub-mc-panel` is a
  tenth and the Files line says it must also be present in the placeholder branch. Without the
  assertion a no-Markov build points the clientside Output at a nonexistent component and
  nothing fails (`test_no_orphan_callbacks` walks the Markov layout). Fix: "the 9 ids plus
  `bub-mc-panel`", and assert the panel is absent on the other five tabs in that mode too.
- **M-2 `xdist_group("raw_perf")` does nothing.** Task 2 line 129 and Task 3 line 165 rely on
  it to keep the timing tests off contended workers. `xdist_group` only acts under
  `--dist loadgroup`; `pytest.ini` has `-n logical` only. The best-of-3 / 0.6 s bound is the real
  protection. Fix: drop the marker claim, or add `--dist loadgroup` to `pytest.ini` and Task 2
  Files, or say Step 4 runs those two tests with `-n0`.
- **M-3 Files lists omit edited test files, and commits are scoped.** Task 6 Files omits
  `test_mc_autowiden.py` (named only in the trailing paragraph); Task 7 Files omits
  `test_mc_paths_toggle.py` (named only in the unnumbered paragraph). A Files-driven `git add`
  leaves them unstaged and the next commit starts red. Task 5 lists its test files correctly.
  Fix: add both to the respective Files.
- **M-4 The `load_months` / `cycle_table` cache has no home or safe key.** Task 5 Step 3
  ("cached per `data_fp` — M-7") and `test_prices_built_once` (line 222) assume a cache that no
  Interfaces block defines. Keyed on `data_fp` alone it collides across the frozen tests
  (asof 2024-04-20, 2024-09-30, 2026-11-08 under one fp) and returns the wrong months. The test
  text still describes the pre-registry design ("price series built once"). Fix: define the
  cache in `raw_mc` keyed `(data_fp, asof)`; re-word the test to "`load_months` called once per
  `(data_fp, asof)` across two renders".

## Nit

- **N-1** Task 7's "Also update `test_mc_paths_toggle.py`…" (line 283) is an unnumbered
  paragraph between checkbox steps, so a checklist-driven executor can skip it. Make it Step 3b.
- **N-2** Task 5 Step 3 does not name the eight new `update_bubble` parameters or say they are
  appended in the same order with defaults. `test_positionally_aligned_with_the_signature`
  enforces equal lengths, so a missed param fails only at Step 4. Name them.
- **N-3** Task 6 adds `assets/mc_raw_regime_paths.js` but no test mirrors the existing
  `test_asset_and_callback_share_the_click_stamp` for it (`#bub-mc-raw-regime`,
  `__qsMcRawRegimeClickAt` present in both the asset and `MC_REGIME_PATHS_JS`). Add one assert
  to `test_raw_mc_guards.py`.
