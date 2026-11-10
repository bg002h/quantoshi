# Review: Tab-1 regime filter draws the fan from the best-matching 50 paths (eeba0381)

Reviewer: independent (author != reviewer). Scope: `git diff 8ef11c4c..eeba0381`
(`callbacks/charts/__init__.py`, `callbacks/mc_controls.py`, `figures/bubble.py`,
`layout/mc_controls.py`, `test_callbacks.py`, `test_mc_regime_subset.py` new).
Method: own worktrees at `eeba0381` and `8ef11c4c` (since removed), venv
`btc_venv`, tests with `-n0 -o addopts=""`, mutations only in the worktree copy.

**Verdict: 0 Critical / 1 Important / 3 Minor / 4 Nit.** The behaviour is correct
for every case the brief names (subset fan + lines for cached scenarios, label,
note, and no change for the other cases). The one Important is a test whose title
claims a property it cannot detect losing.

## Findings

### I-1 (Important) — `test_fewer_sims_than_the_subset_draws_the_best_of_it` cannot fail for the property it names

`btc_web/test_mc_regime_subset.py:125`. The test asserts only `len(self._lines(fig)) == 8`.
The "best first" trim it is named for lives at `callbacks/charts/__init__.py:536-537`
(`line_paths = line_paths[:display_sims]` after ranking). Deleting it leaves every
test green:

```
$ mutate j_no_trim_to_display_sims   # replace the 2-line trim with `pass`
$ pytest btc_web/test_mc_regime_subset.py btc_web/test_callbacks.py btc_web/test_mc_paths_toggle.py
j_no_trim_to_display_sims -> 1 failed, 280 passed, 5 skipped
    FAILED btc_web/test_callbacks.py::TestBTCPayPricing::test_free_tier_all_models   # the known local-only failure
```

Without the trim, `_add_mc_spaghetti` stride-samples the 50 (rows 0,6,12,…) instead
of drawing the 8 best. That breaks the documented guarantee in
`mc_overlay.filter_paths_by_regime` ("With sims=1 the user gets the SINGLE path that
spent the least time in the blocked regimes") and it was never tested before
either (the removed gray test also asserted only `len == 8`).
No other test file references `filter_paths_by_regime` (grep: only
`test_cache_key_alignment`, `test_snapshot`, `test_callbacks`, `test_mc_regime_subset`
mention `mc_regime`).
Fix: in that test compare the drawn lines with `self._ranked([2,3,4])[:8]` (same
`_mc_path_steps` + 2-sig-fig tolerance as `test_example_lines_come_from_the_same_subset`),
and add a `mc_sims=1` case that must match `ranked[0]`.

### M-1 (Minor) — two retained branches have no test
Both survive deletion (`pytest` on the three files above: only the known failure):

* `elif is_free: fan_is_reference = True` (`charts/__init__.py:528`, `regime_model is None`):
  mutation `f_no_model_arm_removed` -> `1 failed, 280 passed` (known failure only).
  The commit deleted the two tests that used to pin the gray reference and added none.
  Practically unreachable (all five cached models are registered, see Verified OK).
* `elif ranked is not None: line_paths = ranked` (paid/live re-rank, `:533`):
  mutation `g_paid_ranked_arm_removed` -> same. Dead on Tab 1 today (F-15: mc_result is
  always None), so it can only be reached by patching `_get_mc_bubble_fig`, which is
  what my probe did (below).

Suggested: one test each, using `monkeypatch` of `utils._get_mc_bubble_fig` +
`btcpay.is_free_tier` for the paid arm, and `mc_cache.get_cached_paths` +
`is_free_tier` + `mc_model_src="bogus"` for the no-model arm.

### M-2 (Minor) — ranking now runs on every Tab-1 render with a filter, including the default paths-off state, uncached
Before: ranking ran only `if blocked and show_paths` (example lines are opt-in, so the
default state never paid for it). Now `ranked` is computed whenever `is_free` (the fan needs it).
`filter_paths_by_regime` is a Python loop of 481 steps x 27 `price_at` calls:

```
bub         steps=481 0.68s 0.67s 0.67s
qr          steps=481 0.04s 0.04s 0.04s
pl          steps=481 0.04s 0.04s 0.04s
ecfg_1d_1u  steps=481 0.10s 0.11s 0.12s
ef          steps=481 0.54s 0.56s 0.51s
```

~0.67 s of gunicorn-worker CPU for the default model on every `update_bubble`
(any Tab-1 control change) while a regime box is unchecked. The result depends only on
`(mc_model_src, start_yr, entry_q, years, blocked, bins)`; an `lru_cache` keyed on that
tuple (storing the row order, not the array) would make it free after the first call.
Matches the project's ALARA preference; not a correctness issue.

### M-3 (Minor) — the Tab-1 note describes the cached subset in states where the server is not doing that
The note shows whenever `0 < checked < bins`, but the cached subset is only drawn when
`is_free`. `is_free_tier` is False for bins != 5 and sims > 200 (both live in the same
Advanced body as the note):

```
bins5 sims8 :  True
bins6 sims8 :  False
bins5 sims400: False
```

There the note still says "the fan and paths use the 50 of 200 cached paths…" while
the chart draws nothing (F-15) or asks for payment. The generic note it replaced
ended "Paid live MC: paths cannot leave selected regimes"; the Tab-1 note dropped that
clause. Not a "wrong show/hide" (the server does filter), but the sentence is false
in those states. Suggest appending "(free tier: 5x5, up to 200 sims)" or hiding the
note when the JS can see `bins != 5`.

### N-1 (Nit) — JS and server disagree only when `bub-mc-bins` is null
72-case grid (12 checklist values x 6 bins values) of `MC_REGIME_NOTE_JS` run under node
against `_build_mc_params(...)["mc_blocked_bins"]` (bins coerced with `_ci(b, MC_BINS)`
exactly like `_coerce_mc`):

```
cases: 72 mismatches: 6
([0], None, 'server_filters=True', 'js_shows=False')   ... (all six have bins=None)
```

Server defaults null bins to 5; JS hides. Unreachable in the UI (`bub-mc-bins` is
`clearable=False`). All 66 other cases (out-of-range indices, duplicates, string "5",
empty, null checklist) agree.

### N-2 (Nit) — stale docstring
`figures/bubble.py:184`: `_add_mc_fan` says `reference=True` is "used when a regime filter
is active on a cached scenario". Now it is only the no-model fallback.

### N-3 (Nit) — "200" is hard-coded in the note, computed in the legend
Note: `"…50 of 200 cached paths…"` (`layout/mc_controls.py:109-110`); legend:
`paths.shape[0]`. Equal today (all 45 cached `paths_*.npz` arrays have 200 rows).

### N-4 (Nit) — a stale checklist value can produce a "best 50 of 200" that is not ranked
The bins->options callback deliberately does not reset the checklist value
(`callbacks/mc_controls.py:~124-133`). Set bins=7, leave only boxes 5 and 6 checked, set bins back to 5:

```
stale value [5,6] after bins 7->5: blocked = [0, 1, 2, 3, 4]
ranked order == original order (all ties): True
```

The UI shows no box ticked, the server treats every regime as blocked, all paths tie,
and the fan is the first 50 cached paths labelled "best 50 of 200". Exotic (needs a detour
through a paid bin count) and the pre-existing comment accepts stale values; before this
commit the same state drew the gray reference fan.

## Observations

**The subset fan is a weak, short-lived signal.** Near-term it moves in the right
direction (2028 start, entry Q10, 40 yr, model `bub`; median over the 50 best-ranked
paths vs all 200):

```
year   all-200     allow{Bargain,Cheap}   allow{Pricey,Bubble}   lo/hi
2030   233,508       220,815                252,444              0.87
2035   467,391       425,227                487,824              0.87
2040 1,412,989     1,402,326              1,391,419              1.01
2050 4,822,496     5,259,028              4,998,316              1.05
```

By 2040 the two opposite filters are indistinguishable and by 2050 they invert,
because the ranking scores the whole 40-year path while the chart (default range)
shows the first ~5-12 years. Max |log| shift of the median over a 2010-2040 view is
~10% for any of the four filters I tried; the 5-95 band width ratio stays 0.83-0.99.
Ranking itself is sound for all five cached models (`ranked[:50] != paths[:50]`, selected
rows spread over 2..199). The label and note disclose "best 50 of 200"; they do not say
the ranking horizon is the full run. Not rated as a finding per the brief.

## Verified OK

1. **Callback branch logic.** Probe `update_bubble` on `8ef11c4c` vs `eeba0381`
   (same kwargs; paid case via patched `utils._get_mc_bubble_fig` returning
   `(fig, {"price_paths": all200})`, `btcpay.is_free_tier -> False`,
   `mc_helpers._mc_payment_check -> True`; no-model case via `mc_model_src="bogus"` with
   patched `mc_cache.get_cached_paths`/`is_free_tier`). Trace names, fill colours, legend
   flags, line counts and md5 of rounded y-data of every MC trace:

   ```
   A_free_nofilter_paths8 / A2 (paths200)      IDENTICAL
   B_free_filter_paths8 / B2 (200) / C (off)   DIFFERENT  (by design)
   D_free_allunchecked / D2_free_allchecked    IDENTICAL
   E_paid_filter_paths8 / E2 (200) / F (off)   IDENTICAL  (amber, no suffix, lines re-ranked; F: no ranking)
   G_paid_nofilter_paths8                      IDENTICAL
   H_free_filter_nomodel / H2 (paths off)      IDENTICAL  (gray 'MC ref', fan = all 200)
   I_free_filter_stack2.5                      DIFFERENT  (by design)
   ```
   New free+filter output: `MC 5–95% · best 50 of 200`, `MC 25–75% · …`, `MC median · …`,
   fills `rgba(220,120,0,…)` (amber), 8 lines (sims=8) / 50 lines (sims=200) / 0 lines (paths off).
   Order: `filter_paths_by_regime(paths, t_axis, …)` at `:519` runs on the full 481-step
   paths; the right-edge `keep` trim and `* stack_f` follow (`:538-560`). Mutation
   `e_rank_after_trim` (rank `paths[:, :145]`) -> `1 failed` (`test_filter_draws_the_fan_from_the_best_matching_subset`).
2. **Label suffix.** `_add_mc_fan` over the four (reference, suffix) combinations:
   ```
   reference=False suffix=' · S' -> MC 5–95% · S | MC 25–75% · S | MC median · S
   reference=True  suffix=' · S' -> MC ref 5–95% · S | MC ref 25–75% · S | MC ref median · S
   reference=True  suffix=''     -> MC ref 5–95% | MC ref 25–75% | MC ref median
   reference=False suffix=''     -> MC 5–95% | MC 25–75% | MC median
   ```
   Each legend entry once. The `endswith("median")` patch only reaches the median trace
   (band names end in `%`+suffix; the median label is hard-coded in `_mc_build_traces`,
   not derived from the bands), so it cannot double-apply.
3. **Note JS vs server, and no new POST.** See N-1 for the 72-case grid. The callback is
   registered clientside (`TestNoteCallback::test_is_clientside`; exactly one entry owns
   `bub-mc-regime-note.style`), no `allow_duplicate`/`prevent_initial_call`, so it fires on
   load and on share-link restore. `bub-mc-bins`, `bub-mc-regime` and the note div exist in the
   no-Markov placeholder (`layout/mc_controls.py:141-144`, covered by
   `test_placeholder_exists_when_mc_is_unavailable`). Live dev server (`:8050`, commit
   `eeba0381`), page `/`: `_dash-layout` contains `bub-mc-regime-note`; the DOM node exists with
   `style="display: none;"`; the console (7 errors) names `sc-mc-years`, `cp-mc-years`,
   `ret-mc-years` (State, other tabs) and the F-20 `citadel-graph.figure` 500s; no entry
   mentions any `bub-mc-*` id or the note.
4. **Layout.** `_bub_regime_note() if prefix == "bub" else html.Small(...)` sits in a list
   literal; `TestNoteLayout` (all five other prefixes keep the generic note and get no
   `*-mc-regime-note`; Tab 1 loses "percentiles unchanged"). Import check:
   ```
   $ TESTING=1 PYTHONPATH="btc_web:." btc_venv/bin/python3 -c "import app, engines.adapter; print('import ok')"
   import ok
   ```
   (`figures.bubble` is imported lazily inside `_bub_regime_note()` at layout-build time.)
5. **Name consumers.** `grep` of `btc_web tools scripts` (`*.py`, `*.js`, excluding `build/` and
   minified bundles) for `MC median`, `MC ref`, `MC 5`, `MC 25`, `MC paths`: only
   `mc_overlay.py` (producers), `figures/bubble.py` (producer), colors.py comments and tests.
   `restore_builder.py`, `snapshot_cb.py`, `timemachine*.py` and every project JS asset have no match.
   `figures/common.py::_find_mc_median_trace` selects by `line.dash == "dot"`, not by name.
   Restore re-fires `update_bubble` (restore_builder: "punt to cascade"), the only caller of
   `_add_mc_fan`.
6. **Test honesty.**
   ```
   $ pytest btc_web/test_mc_regime_subset.py --collect-only -q   -> 26 tests collected
   $ pytest btc_web/test_callbacks.py --collect-only -q | grep -c TestUpdateBubbleCallbackMC::   -> 12  (14 on 8ef11c4c minus the 2 deleted)
   ```
   `TestCallback` collects exactly its 8 own tests (the `setattr(..., None)` loop works);
   the parent's 12 still run from `test_callbacks.py`. Mutations (new test file only):
   ```
   a_fan_all200            -> 2 failed (…fan_from_the_best_matching_subset, …filter_moves_the_fan)
   b_lines_all200          -> 1 failed (…example_lines_come_from_the_same_subset)
   b2_lines_unranked50     -> 1 failed (same)
   b3_lines_ranked_all     -> 1 failed (same)
   c_js_k_lt_n_only        -> 2 failed (TestNoteDecision none_checked / missing_values)
   c2_js_drop_k_gt_0       -> 2 failed (same)
   d_suffix_missing        -> 3 failed
   d2_median_unsuffixed    -> 4 failed
   e_rank_after_trim       -> 1 failed
   h_fan_amber_to_ref_when_subset (callback passes reference=True with the suffix) -> 3 failed
   i_subset_size_10        -> 1 failed (test_subset_size_is_fifty)
   ```
   Suites: `test_mc_regime_subset.py` + `test_callbacks.py` -> `1 failed, 253 passed, 5 skipped`
   (the known `TestBTCPayPricing::test_free_tier_all_models`); guard files
   (`test_no_orphan_callbacks`, `test_infrastructure`, `test_cache_key_alignment`,
   `test_mc_paths_toggle`, `test_mc_autowiden`, `test_checklist_option_coverage`,
   `test_snapshot_defaults`, `test_restore_builder`, `test_noop_bumps`, `test_modal_commit`)
   -> `157 passed`.
7. **Cached-model preconditions.** `_CACHED_MODEL_KEYS = bub, ecfg_1d_1u, ef, pl, qr`, all registered
   in `PRICE_MODELS` with 27/27/21/27/27 quantiles (`filter_paths_by_regime` early-returns the
   input only for < 2), so the `regime_model is None` and "ranking is a no-op" cases are not reachable.

## Not covered

* Browser interaction with the live checklist (ticking boxes, bins change) was not re-driven;
  I relied on the controller's dev/prod run and on the node/grid check of the decision function.
* Share-link round-trip of a filtered Tab-1 state, mobile legend width with the longer
  names, and Time Machine interplay were not exercised (no code path in the diff touches
  them; consumers grepped, not run).
* Paid/live Tab-1 MC (F-15) is dead code today; the paid arm was checked only through a patched builder.
* Full non-E2E suite not re-run (controller reports 1 failed / 3225 passed on this commit);
  E2E suites not run.
* The Model Info "Regime Filter" text (`layout/model_info/_items.py:1968`) still describes only
  the paid column-zeroing behaviour; pre-existing and untouched by this commit.
