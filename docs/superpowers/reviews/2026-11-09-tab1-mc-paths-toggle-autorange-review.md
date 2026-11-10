# Review: Tab-1 MC paths toggle + auto-range from current year (997d3f33..afb60c42)
Verdict: 0 Critical / 0 Important / 2 Minor / 3 Nit

Reviewer: independent (author != reviewer). Range reviewed exactly `997d3f33..afb60c42`
(commits `198ffc9b`, `a6ad3015`, `afb60c42`). All experiments ran either read-only against
the working tree or in scratch copies built with `git archive afb60c42` under
`/tmp/claude-1000/-scratch-code-bitcoinprojections/3a79ac35-3364-4e4c-9013-baf596cc6ebf/scratchpad/review/{repo,repo2}`
(symlinked `mc_cache`/`citadel_band_cache`). No source or test file in the repo was left
modified by me (see "Working-tree state" at the end).

## Findings

### M-1 — "A stale client sends nothing for this Input" is false: Dash answers 500, so the `mc_paths=None` default is unreachable and its test justifies a case that cannot occur
- Where: `btc_web/callbacks/charts/__init__.py:189-190` (comment), `:222` (`mc_paths=None`),
  `:482-483` (comment "a stale client sends nothing for this Input"),
  `btc_web/test_callbacks.py:1461` (`test_omitting_the_argument_means_off`, docstring "A browser still holding the previous callback signature sends no value for the new Input at all").
- What: the author's rationale for appending the Input LAST and defaulting `mc_paths=None` is that a browser holding the previous callback signature still lines up and the new argument reads as off. Dash does not behave that way. The server maps the request's `inputs + state` list through `inputs_state_indices` (`dash/dash.py:1519`, `_prepare_grouping`); a request one Input short raises `IndexError` and the response is HTTP 500. The callback body never runs, so `mc_paths` is never `None` for that reason. Position in the decorator is irrelevant (the flat list is inputs-then-states, the new Input shifts every State). The observable behaviour is the already-documented stale-`/_dash-dependencies` hazard (CLAUDE.md "Stale /_dash-dependencies between deploys"), identical for any added Input; this is not new damage. The defect is that the comments and the test assert a safety net that does not exist. `test_omitting_the_argument_means_off` only checks `inspect.signature(...).default is None`, i.e. a property of the Python signature, not of what a stale client causes.
- Evidence: scratch probe POSTing to `/_dash-update-component` with the full 8-output list, 42 State, and 42 vs 41 Inputs:
  ```
  cd .../review/repo2 && python3 .../exp2.py
  outputs parsed: 8
  inputs dropped=0: status=200
    File ".../dash/dash.py", line 1519, in _prepare_callback
  IndexError: list index out of range
  inputs dropped=1: status=500
  ```
  (An earlier probe with a truncated `outputs` list failed at `dash.py:1522` instead, the outputs grouping; the complete-body run above isolates the Input count as the only difference.)
- Suggested fix: reword the three comments to "a stale client is rejected by Dash until it reloads (same as any new Input)"; drop or rename `test_omitting_the_argument_means_off`, keeping the `off=None` parametrisation of `test_paths_off_draws_the_fan_alone` (which does test `None` handling inside the body).

### M-2 — The predating-link rule never fires for an unregistered (evicted) fingerprint; those links silently lose their paths
- Where: `btc_web/snapshot.py:920-923` (unregistered fp -> `historical = SNAPSHOT_DEFAULTS`) interacting with `_fill_predating_fields` (`:690`, which skips any key already in `state`).
- What: for an fp absent from the registry the decoder fills every unset key from the CURRENT `SNAPSHOT_DEFAULTS`, which now contains `bub-mc-paths:value = []`. The key is therefore present in `state`, so the rule (which keys on absence) is skipped and an old MC-on link restores fan-only. Today this affects no shipped link: I computed the defaults fingerprint at every commit that touched `snapshot.py`/`snapshot_defaults.py` since 2026-04-24 and all 19 distinct fingerprints are in the registry (output below). It is latent: the registry is capped at 20 (`tools/update_defaults_registry.py` CAP) and now holds 19; after a few more default changes the Tab-1-MC-bearing fps (first one `206b715a`, the 7th entry) start being evicted, and from then the rule cannot rescue them. Those links already restore under new defaults for every other changed field, so this is the same class as the existing eviction hazard, but this diff adds a field whose absence matters.
- Evidence:
  ```
  python3 fpwalk.py   # fingerprint at every relevant commit vs registry
  028d77e5 REG ... faf00e93 REG ... 31c83534 REG ... (19 lines, all REG, 0 UNREG, 0 ERR) ... 72f9cada REG ... 04aa50d6 REG
  python3 difftest.py   # unregistered fp "deadbeef", bub-mc-enable=["yes"]
  MC on, paths not on: v4/deadbeef/['yes'] []
  ```
- Suggested fix: acceptable to leave (record as a follow-up owned by whichever phase next touches the registry cap). If wanted: when the fp is unregistered, build `historical` from `SNAPSHOT_DEFAULTS` minus the keys of `_PREDATING_LINK_VALUES`, so the rule can still apply (trade-off: an unregistered NEWER fp with a deliberate "off" would then flip to on, so decide knowingly).

### N-1 — `MC_AUTOWIDEN_JS` type guard on `xrange` elements is not pinned by any test
- Where: `btc_web/layout/bubble.py:104` (`typeof xrange[0] !== "number" || typeof xrange[1] !== "number"`).
- What: deleting the line leaves all tests green (surviving mutant), because no malformed-values case passes a length-2 array of non-numbers. Reachable only if the slider hydrates with strings, which Dash does not do, so harmless today.
- Evidence (clean scratch copy of `afb60c42`): `M12 JS typeof guard dropped: 131 passed, 5 warnings`.
- Suggested fix: add `(ON, ["a", "b"], 2028)` to `test_malformed_values_are_a_no_op`, or drop the guard.

### N-2 — E2E asserts the browser's current year equals the Python host's current year
- Where: `btc_web/test_mc_autowiden_e2e.py:88` (`assert got[0] == datetime.date.today().year`).
- What: the JS uses the browser's local clock (`new Date(now).getFullYear()`), the test the host's; they can differ for a few hours around 31 Dec / 1 Jan when the Playwright browser and host timezones differ. Not a defect in the shipped JS (the node unit test fixes the clock mid-year).
- Evidence: reading the two sources; the unit tests use `_ms(year)` = 15 June 12:00 UTC to avoid exactly this.
- Suggested fix: read the year from the page (`page.evaluate("new Date().getFullYear()")`).

### N-3 — Two `TestLayout` tests fail instead of skipping when the Markov engine is unavailable
- Where: `btc_web/test_mc_paths_toggle.py:218` (`test_one_checklist_on_tab_1_default_off`) and `:227` (`test_sits_inside_the_mc_panel_body`).
- What: with `_app_ctx._HAS_MARKOV = False` the layout returns the hidden placeholder, whose `dcc.Checklist` has no `options` and whose `bub-mc-body` does not exist. Dev has the engine, so they pass here; a checkout without a built `markov` extension would show two red tests unrelated to the code under test.
- Evidence (scratch, `monkeypatch.setattr(_app_ctx, "_HAS_MARKOV", False)`):
  ```
  NOMARKOV test_one_checklist_on_tab_1_default_off AttributeError 'Checklist' object has no attribute 'options'
  NOMARKOV test_sits_inside_the_mc_panel_body AssertionError assert 0 == 1
  ```
- Suggested fix: `pytest.skip` when `not _app_ctx._HAS_MARKOV` in those two tests.

## Checked and found sound

**1. Share-link backward compatibility**
- Control list: `_SNAPSHOT_CONTROLS[:339]` at `997d3f33` equals `[:339]` at `afb60c42`; only new tail is `('bub-mc-paths','value')`. Registry entry `04aa50d6` differs from `72f9cada` only by the added `bub-mc-paths:value: []`; and equals current `SNAPSHOT_DEFAULTS`.
  ```
  len old/new 339 340 | prefix identical: True | new tail: [('bub-mc-paths', 'value')]
  registry n = 19 | 04aa50d6 vs 72f9cada: added ['bub-mc-paths:value'] removed [] changed []
  current fp 04aa50d6 | registry entry == SNAPSHOT_DEFAULTS: True
  ```
- Differential decode, old code (`997d3f33`, `git archive` copy) vs new code, 344 synthetic links (all 18 pre-existing fingerprints x MC unset/on/off x 6 random index-diff sets each, a `deadbeef` unregistered fp, positional links of length 206/300/334/336/339/340): every decoded key other than `bub-mc-paths:value` is identical. The key differs in exactly 113 cases and in every one the link had `"yes"` in `bub-mc-enable:value`, no value for the key in the old decode, and now decodes to `["yes"]`. The only other flagged rows were harness artefacts: `pos/340/0` (a new-format link with an explicit 0, which old code truncated away), and the `deadbeef` rows (current-defaults fallback, see M-2).
- Rule cannot override a deliberate off: it skips any key already in `state`. A post-change v4 link always carries the key (the registered default of its own fp, `historical.get(key)` is `[]`, not `None`), a post-change positional link carries a bitmask int (0 decodes to `[]`, present). Confirmed by `manage_snapshot` end-to-end in a scratch test (real `manage_snapshot` encode, real prefix decode): `RT {'on': (['yes'], ['yes']), 'off': (['yes'], [])}`; and mutation M5 (rule overrides existing) is killed by 3 tests.
- Unregistered fp: falls back to current `SNAPSHOT_DEFAULTS` (see M-2); no already-shipped fp is unregistered.
- Restore path to `update_bubble`: `restore_from_url` -> `_decode_snapshot_by_prefix` -> state; `apply_tab_bubble` has 66 Outputs with `bub-mc-paths` last (index 65). Old MC-on link sends `['yes']`, old MC-off link `no_update`, new MC-on/paths-off link `[]`. `restore_builder._build_bubble_figure_from_state` returns `None` when `bub-mc-enable` is ticked (verified), so MC links always take the callback path that reads the new Input. `"bub-mc-paths"` is in `_POST_RESTORE_TRIGGERS` (MC-off fast-restore links do not rebuild when the cascade writes it; the gate clears on mousedown via `snapshot_cb.py` ~1162).
- `_decode_snapshot_v1` (q1) is not patched; q1 predates every `bub-mc-*` control so it cannot carry the gate.

**2. Dash wiring**
- Runtime introspection of the live registration (`test_no_orphan_callbacks._all_callbacks`): 42 Inputs + 42 States = 84 params; `inputs_state_indices` maps the last parameter `mc_paths` to `bub-mc-paths.value`, and `tm_toggle`/`asof_slider` to the two Time Machine Inputs before it (printed from `flat[idx[i]]`).
- Only `callbacks/__init__.py` re-exports `update_bubble`; no other caller outside tests (`grep -rn update_bubble` over `btc_web tools scripts tests`). `_mc_controls(...)` callers (sim_tabs, citadel, heatmap, supercharge, bubble) keep the default `show_paths_toggle=False`; `test_other_tabs_do_not_get_one` covers it.
- `bub-mc-paths` is in `_TAB_CONTROLS["bubble"]` (M9 killed), the non-Markov placeholder branch (M11 killed), the layout inside `bub-mc-body` within `bub-mc-exclude-wrap` (so Time Machine hides it; TM force-off clears `bub-mc-enable` only, harmless).
- Cache-key claim holds: `mc_paths` is read only after `_get_mc_bubble_fig`/`_get_bubble_fig` return, never enters the params dict, `tab_defaults.py`, prewarm or any cache key; traces go on `_go.Figure(fig)` clone. On->off->on->off->None->on sequence against the shared cached figure: `SEQ [(['yes'],8,15),([],0,7),(['yes'],8,15),([],0,7),(None,0,7),(['yes'],8,15)]` (line count and trace count are stable, no leakage).
- `if blocked and show_paths` only skips the ranking work when no lines are drawn; fan percentiles and the `fan_is_reference` decision are unchanged (original block read side by side).

**3. Auto-range JS**
- Read in full. Outputs are within `[XRANGE_MIN, XRANGE_MAX]`: `hi = max(xr[1], min(2080, start+10))` so `hi <= 2080` for any start; `lo = max(2010, year)` and is used only when `lo < hi`, else the original `xr[0]`; inverted/empty result is impossible from a valid slider value; exact no-op returns `NU`. Fires only on a fresh click stamp, Price view (null = default), CTA off, MC ticked (not unticked), numeric slider and start year. Stamp is consumed on every invocation.
- Other writers/readers of `bub-xrange`: single `Input` (`bub-mc-enable`), `State` for the rest, `allow_duplicate` + `prevent_initial_call=True` as the Dash 4.0 note requires (asserted by `test_mc_autowiden.py`); view-mode cap, CTA and Time Machine are guarded or hidden; Auto-Y and the "-1y" preset visibility react to the write as to any user edit.
- Mutations (clean scratch copy, JS unit + listener tests): xmin clamp removed -> killed; inverted guard removed -> killed; no-op guard removed -> killed; right edge pulled in -> killed (2); CTA guard removed -> killed; view-mode guard removed -> killed (5); `isFinite` removed -> killed; click-age guard removed -> killed; startYr null guard removed -> killed. Survivor: typeof guard (N-1).

**4. Tests that cannot fail** (temporary-edit mutation, one mutation at a time, `PYTHONDONTWRITEBYTECODE=1`, 131 tests across the five files; M1-M11, M13-M16 ran in the live tree, M12 and M17-M23 re-run in a clean scratch copy)
```
M1  show_paths=True                      6 failed (test_paths_off_draws_the_fan_alone x4, paths_on_adds..., regime_filter_reference)
M2  show_paths=bool(mc_paths)            2 failed (off="yes", off=["no"])
M3  line_paths ungated                   6 failed
M4  regime ranking ungated               1 failed (test_paths_off_with_a_regime_filter_still_draws_the_gray_reference)
M5  predating rule overrides existing    3 failed
M6  predating rule ignores MC gate       1 failed (old_v4_link_without_mc_is_left_alone)
M7  positional decode skips rule         1 failed (old_positional_link_with_mc_on_keeps_its_paths)
M8  _CHECKLIST_OPTIONS entry removed     2 failed (checklist_options_registered, encoded_as_a_bitmask)
M9  _TAB_CONTROLS entry removed          2 failed
M10 _POST_RESTORE_TRIGGERS entry removed 1 failed (test_in_post_restore_triggers)
M11 non-Markov placeholder removed       1 failed (test_placeholder_exists_when_mc_is_unavailable)
M12 JS typeof guard removed              131 passed  <- survives (N-1)
M13..M23 (JS guards, default flipped to on, spaghetti drawn unconditionally) each killed by >= 1 test
```
(In the live tree M10-M13 also showed two `TestUpdateBubbleCallbackMC` fan failures that are not attributable to the mutations: they appeared when the controller's uncommitted edits to `figures/bubble.py`/`test_callbacks.py` landed mid-run. I excluded them and re-ran the remainder in the clean copy, where baseline is `131 passed`.) Also checked the tests actually exercise what they claim: `_lines` is non-empty in the paths-on test (8), the regime-reference test asserts the gray "MC ref median" trace, the bitmask test asserts the diff value `1`.

## Not checked
- Browser behaviour of the new checkbox and of the E2E restore suites (`test_restore_*_e2e`, Playwright); only the node unit harness and the author's reported E2E pass were relied on.
- The full non-E2E suite (author-verified: 3197 passed); I ran only the five files the diff touches plus scratch tests.
- Multi-worker/gunicorn and prod (Redis, L0/L2) behaviour; the MC engine and cache; fan appearance.
- `cbecb30e` and anything after `afb60c42` (outside the range).
- Timezone behaviour of the JS beyond reading it (node tests pin the clock to mid-June UTC).

## Working-tree state at hand-back
`git status --short` shows only `model_data_ef.pkl` (pre-existing) and `docs/superpowers/followups.md` (the controller's own edit; I never touched it). All six files I mutated (`callbacks/charts/__init__.py`, `snapshot.py`, `layout/bubble.py`, `layout/mc_controls.py`, `callbacks/routing.py`, `snapshot_defaults.py`) are byte-identical to `afb60c42` (`git diff --quiet afb60c42 -- <file>` for each). One mutation (the CTA-guard line in `layout/bubble.py`) was left behind when I stopped my in-tree driver at the controller's request; it was reverted by path with `git checkout -- btc_web/layout/bubble.py`. No `git checkout -- .`, `stash`, `restore .` or `reset` was used. Nothing committed or pushed; dev server on :8050 untouched.
