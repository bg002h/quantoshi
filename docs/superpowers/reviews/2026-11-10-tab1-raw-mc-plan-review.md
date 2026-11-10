# Review — Tab-1 Raw MC implementation plan (draft, `be018406`)

**Counts: 0 Critical · 6 Important · 11 Minor · 8 Nit**

Scope: `docs/superpowers/plans/2026-11-10-tab1-raw-mc.md` against spec rev 6 and F-22. The
question is whether executing the plan exactly would violate the spec, break existing behaviour,
or leave the engineer stuck on a wrong file, id, signature or test target. The spec's design is
not re-reviewed.

## Machine-checked (all against the working tree at `be018406`)

- `tools/raw_mc_cycle_stats.py` (data to 2026-11-08): months `[27, 44, 46, 47, 31]`, full
  `[T,T,T,T,F]`, complete `[F,T,T,T,F]`. Default fit: n 5.1182, a 2.4603, b 0.4861, α 1.1787,
  β −0.1786, γ 0.6570, δ −0.0866. Window 2013: n 5.932, b 0.717, "window". Window 2015 falls back
  to "all history" with n 5.12. Window 2017 has 1 complete cycle. All of the plan's pinned
  targets fall within abs 0.005. The tightest is b (0.0039), which passes. The CLI prints
  `fit on 4 full cycles: power n = 5.12`. Row 2012-11 has φ = −0.0099, so the clip test is
  meaningful.
- `_SNAPSHOT_CONTROLS`: len 340, `bub-mc-paths` is at index 339, so 340–347 is correct. The
  current defaults fingerprint is `04aa50d6` and is already the last registry entry.
  `_CHECKLIST_OPTIONS` holds `bub-mc-paths` but not `bub-mc-regime`, which is consistent with
  the "treat like" rule. `bub-mc-window` has the static default `[2010, 2026]` and is not in
  `ALWAYS_ENCODE`.
- `update_bubble` has 8 Outputs, and both early returns are `(dash.no_update,) * 8`. The last
  Input is `bub-mc-paths`. `_mc_setup`, `_mc_finalize` and `get_cached_paths` sit on the paths
  the plan describes. `_add_mc_fan(label_suffix=)` exists, and the median is named
  `"MC median" + suffix`. `_add_mc_spaghetti` uses stride `n // n_display`, which is 1 when
  `n_display == len`.
- The `bub-mc-paths.value` writer is single and primary (`prevent_initial_call=True`, Input
  `bub-mc-regime`). `MC_AUTOWIDEN_JS` has 1 Input and 4 States. `dash_clientside.callback_context.triggered`
  exists in the Dash 4.0 renderer bundle.
- `_resolve_mc_model_src("raw_cycle"|"raw_iid", …)` already returns its input unchanged.
- The future halvings (`_HALVING_EPOCHS`) all fall on April 20. A monthly step grid on a fixed day
  of the month therefore gives exactly 48 steps per whole future cycle, so the plan's
  `assert n == 48` holds.

## Important

### I-1 — The "Paid" badge sits outside `bub-mc-body`, so the plan's class toggle cannot hide it
- **Where:** Task 5 Interfaces. The clientside callback writes `bub-mc-body.className`; the CSS
  is `.qs-mc-raw-mode .qs-mc-model-only {display:none}`; "badge" is in the model-only list.
- **Failure:** `_mc_controls` passes the `⚡ Paid` span (`.model-panel-paid-badge`) as
  `header_right` of `_section_card`. That renders it in the card header
  (`layout/common.py:278-281`), a sibling of `bub-mc-body`, not a descendant. The descendant
  selector never matches, so Raw mode still shows "Paid". The spec lists the "Paid" badge as
  Hidden. The E2E in Task 8 checks the Run button, cost panel and Saved card, but not this badge.
- **Fix:** put the mode class on an ancestor of both header and body. Either give the outer
  `position: relative` Div in `_mc_controls` an id (e.g. `bub-mc-panel`) and target it, or add a
  second Output. Add the badge to the Task 8 E2E "not visible" list.

### I-2 — Existing tests that pin the old shapes are not scheduled for update, and the plan says the suite passes "untouched"
- **Where:** Task 5 (`test_model_mode_unchanged`: "existing TestUpdateBubbleCallbackMC suite
  passes untouched"; Step 4 runs `test_callbacks.py`) and Task 6 ("Update
  `test_mc_regime_paths.py::TestRegistration`"; Step 4 runs `test_mc_autowiden.py`).
- **Failure:** each of these goes red when the plan is executed as written.
  - `test_callbacks.py` asserts `len(result) == 8` at lines 1142, 1201, 1230, 1297, 1499,
    1534, 1617 and 1650. That is 8 sites, all broken by the 9th Output.
  - `test_mc_paths_toggle.py::TestCallbackWiring::test_is_the_last_input_and_maps_onto_mc_paths`
    asserts `seq[-1] == ("Input", "bub-mc-paths")` and `params[-1] == "mc_paths"`. It is broken
    by appending 8 Inputs.
  - `test_mc_regime_paths.py`:
    - `_entry()` filters on `inputs == ["bub-mc-regime"]`, so every TestRegistration test
      fails, not only the single-Input one.
    - The **TestDecision** node harness calls `fn(c.checked, c.bins, c.paths)`. Under the new
      5-arg signature `(checked, rawChecked, bins, rawBins, paths)`, bins lands in `rawChecked`.
    - The harness defines no `dash_clientside.callback_context`, so a `triggered[0]` branch
      throws a TypeError. All ~20 decision tests break.
  - `test_mc_autowiden.py::TestRegistration::test_reads_range_start_year_view_and_cta_as_state`
    pins exactly 4 States, so it breaks on the new `bub-mc-model-src` State.
- **Fix:** list these files under Task 5 and Task 6 "Modify". Say each one changes: 8→9;
  last Input → `bub-mc-extremes`/`extremes`, with `bub-mc-paths` still followed by the 8 new
  Inputs in order; the harness passes 5 args and mocks
  `window.dash_clientside.callback_context`; the State list gains `bub-mc-model-src`. Reword
  "untouched" to "passes, with the shape assertions updated in this commit".

### I-3 — `raw_mc` importing `figures.common` breaks the CLI and the module's light-import contract
- **Where:** Task 1 Interfaces (`future_halvings` "from `figures.common._HALVING_EPOCHS`"),
  Step 3 ("add `btc_web` to `sys.path`"), Step 4 (run the script). Also the Architecture line
  ("A numpy/pandas/scipy module").
- **Failure:** `figures.common` imports `btc_core` (repo root) and `_app_ctx`. With only
  `btc_web` on the path, the import fails with `ModuleNotFoundError: No module named 'btc_core'`
  (checked), so Step 4 cannot pass. With the repo root added as well, importing the module still
  loads `_app_ctx` and the MC cache (`[MC-CACHE] Loaded from /dev/shm`). That takes about 7 s
  from `.npz` when shm is cold, which is a lot for a CLI that prints a table and for a
  supposedly numpy-only module.
- **Fix:** keep `raw_mc` free of app imports.
  - Move `_HALVING_KNOWN_DATES` and the epoch builder into a dependency-free module (e.g.
    `btc_web/halvings.py`).
  - Have `figures.common` re-export them, and add a test that both see the same epochs.
  - Or duplicate the dates in `raw_mc` and add a test that they equal `figures.common`'s.

### I-4 — Seeding on the raw `checked` tuple contradicts the plan's own no-filter test
- **Where:** Task 2 `seed_for(req)` ("every field except `p0`"), together with Task 3
  `test_no_effective_filter_variants` ("`()`, `(0..4)`, `(7, 9)` … paths identical to
  unfiltered").
- **Failure:** `()`, `(0,1,2,3,4)` and `(7,9)` all mean "no filter", but their canonical JSON
  differs. The seed differs too, so the paths are not identical and the pinned test fails if the
  Interface is implemented literally.
- **Second failure:** Dash checklists can return values in click order (see the `ma_sel` sort
  comment in `update_bubble`). So `[1,0]` and `[0,1]` draw different fans for the same filter.
- **Fix:** seed on `effective_checked(checked, bins)`, with `None` for no filter and the indices
  sorted otherwise. State this in the `seed_for` Interface.

### I-5 — The spec's performance target ("with the filter on") is never tested
- **Where:** Task 2 `test_performance` runs unfiltered with a 1.0 s dev bound. Task 3 has no
  timing test, and its Step 3 describes "rejection per path per step … ≤ 20 tries".
- **Failure:** the spec target is "3200 paths × 480 months, with the filter on, well under
  200 ms on prod (rejection loops only over the rejected subset)". A per-path Python rejection
  loop passes every listed test and can still cost seconds per request on a 5-worker live
  callback. The 1.0 s dev bound already allows 5× the target on a faster machine.
- **Fix:**
  - In Task 3, add a filtered perf test with a two-bin selection, both variants, 3200 × 40 y.
  - Bound it near the target (e.g. ≤ 0.3 s dev) and print the timing.
  - In Step 3, require rejection vectorised over the rejected subset (mask, redraw, repeat).

### I-6 — The merged auto-tick callback updates only the `triggered[0]` tracker
- **Where:** Task 6 note ("branch on `callback_context.triggered[0].prop_id`, keep separate
  trackers").
- **Failure:** a share-link restore (`apply_tab_bubble`) writes `bub-mc-regime` and
  `bub-mc-raw-regime` in the same batch, so both are triggered in one call.
  - If only `triggered[0]`'s tracker (`__qsRegimeK` / `__qsRawRegimeK`) is updated, the other
    checklist's restored filter is never remembered.
  - Take a link restored with that filter on and paths deliberately off. The user's next click
    on that checklist reads as a no-filter→filter transition and ticks paths.
  - That breaks the guarantee pinned by
    `test_a_restored_filter_is_remembered_so_the_next_click_is_not_a_transition` (review I-1,
    2026-11-10). It breaks for model mode too, if the batch order ever puts the Raw id first.
- **Fix:** both values arrive as Inputs on every call. Update **both** trackers on every call.
  Decide only for the checklist whose click stamp is fresh, not by `triggered[0]`. Add a node case
  that has both triggered with no stamp, followed by a click on each.

## Minor

- **M-1 Wrong file for the resolver.** Task 5 Files says `callbacks/mc_helpers.py
  (_resolve_mc_model_src …)`, but the function lives in `callbacks/charts/_resolvers.py`. It
  already passes `raw_*` through (checked), so `test_resolver_passes_raw_through` will not fail
  at Step 2. Fix: drop the edit, and keep the test as a regression pin.
- **M-2 `_TAB_CONTROLS` lives in `callbacks/routing.py:141`, not `snapshot.py`.** There is no
  separate "bubble reset id set". Fix: Task 7 Files should say "`callbacks/routing.py`
  `_TAB_CONTROLS["bubble"]`" and rename the test accordingly.
- **M-3 `bub-mc-raw-window` default.** `test_defaults` expects `[2010, <current year>]`, but
  Step 3 says to follow `bub-mc-window`, whose default is the static literal `[2010, 2026]`.
  The test goes red in 2027, or the fingerprint changes every year. Fix: pin `[2010, 2026]`.
- **M-4 No-Markov placeholder branch.** `_mc_controls` builds hidden placeholders when
  `_HAS_MARKOV` is False (precedent: `test_mc_paths_toggle.py:237-245`). The 9 new ids are not
  added there. Without them, `update_bubble` references nonexistent Inputs and an Output, and
  Tab 1 stops rendering. Fix: add placeholders and extend that test.
- **M-5 Raw early return skips `apply_zoom_lock`.** The model path applies it twice: once
  directly, once via `_mc_finalize(axes=False)`. The cached figure is a shared object with
  two-way writes, so a Raw return without it shows the zoom state of the last render. Fix: Step 3
  should call `apply_zoom_lock(fig, "chart_zoom" in toggles)` on the Raw figure.
- **M-6 Payment guard placement.** In `_mc_payment_initiate`, the quant-modal branch
  (`tab_price > 50 000`) runs before the free-tier and invoice branches. The cost JS prices any
  `raw_*` as uncached and paid. Fix: place the Raw guard right after `model_srcs[tab_idx]` is
  read, and add a test case with `bub-mc-price-val` > 50 000, asserting `mc-quant-modal` stays
  closed.
- **M-7 Price source unspecified.** The plan never says where `simulate(req, prices)` gets its
  `prices` inside `update_bubble`. Reading the CSV per request goes against ALARA. Fix: build the
  series once from `_app_ctx.M.price_dates` / `price_prices`, or a module cache keyed on
  `_MODEL_FP`, and cache `load_months` / `cycle_table` per `data_fp`.
- **M-8 Spec items with no task or test.**
  - The "Raw starts today at the current price" hint where start year and entry percentile are
    hidden.
  - Raw-specific labels: no "Markov transition matrix", no "≤ 200 is free".
  - Status-line items: the current cycle's pace against the curve, the cycle-mode caveat, and
    how often the ±0.04 tolerance widened.

  Fix: add asserts to `test_status_line_contents`, plus a layout test for the labels and hint.
- **M-9 Raw regime-options server callback.**
  - It needs `prevent_initial_call=True`, with initial options set in the layout. Otherwise every
    Tab-1 load makes a server round trip, against ALARA and the renderer main-thread budget.
  - Dash sends `window` as a list, so it must become a tuple before the `lru_cache` call.
  - Behaviour when the exp fit fails is undefined.
- **M-10 Vacuous E2E.** "Ticking MC in Raw mode sets the `bub-xrange` lower bound to the current
  year" already holds in model mode, where `lo = max(XRANGE_MIN, thisYear)`. Fix: assert the
  upper bound instead: `max(right, thisYear + 10)`, against model mode's
  `max(right, startYr + 10)`.
- **M-11 Stale clients after deploy.** The 9th Output changes `update_bubble`'s output key, so a
  browser holding the pre-deploy `/_dash-dependencies` gets a 500 on every Tab-1 redraw until it
  reloads. Review Focus 4's "no 500" is therefore not true for `update_bubble`. The spec mandates
  the Output, so this is recorded only. Optionally, write `bub-mc-raw-status` with
  `dash.set_props` and keep the output key.

## Nit

- **N-1** `layout/bubble.py` cannot "pass `include_raw=True`" to `_mc_model_src_options`. It
  only calls `_mc_controls`. Add a `_mc_controls(…, include_raw_src=False)` kwarg and thread it
  through.
- **N-2** `RawResult` has no bin edges or per-path state, but Task 3's `_closed_range` and
  `_median_state_at_40y` need them. Add `edges` and `final_state` (or `states`) fields so the test
  does not re-derive D.
- **N-3** `RAW_LINES_MIN/MAX/DEFAULT` are counts, not appearance constants. Put them in
  `raw_mc` or the layout, not `colors.py`.
- **N-4** Future φ is taken at `d_k` (step start), but history φ is at the month midpoint, a
  ~0.01 systematic offset. Consider `d_k + 15 d`, matching μ. E and test (a) adapt
  automatically.
- **N-5** Step 3 does not restate that the current cycle's σ uses its *remaining* span's
  midpoint, and the frozen start (on a halving) cannot detect the difference.
- **N-6** `test_seed_ignores_start_price` ("exactly doubled") holds only unfiltered: D₀ depends
  on `P_today`. Say so in the comment.
- **N-7** F-22 N-2 is "settled" by declaring the spec's figures illustrative. The figures
  themselves (1.57 %, 0.07, the "r4" attribution) remain wrong in the spec. That is acceptable,
  but note it in F-22's closure.
- **N-8** `_MC_MATCH_JS_TPL` is `.format()`-ed, so any new JS braces must be doubled.

## F-22 disposition

| Item | Settled in plan? |
|---|---|
| M-1 step → cycle, φ point | Yes ("F-22 decisions", step dates). See N-4. |
| M-2 redraw never crosses a halving | Yes (block keeps slot and length) |
| M-3 test (a) reads generator objects | Yes (`schedule`, `z_shift` on `RawResult`) |
| N-1 E/n, n = 48, short horizons | Yes (n = 48 holds; checked against `_HALVING_EPOCHS`) |
| N-2 figure labels, short-block pool | Pool: yes. Labels: by declaration (N-7) |
| N-3 D₀ partial-month μ | Yes |
| N-4 thin-cycle standardisation | Yes (< 12 months → trend μ/σ; None → pooled) |
