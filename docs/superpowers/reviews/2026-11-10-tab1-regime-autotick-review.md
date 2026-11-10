# Review: Tab-1 regime filter auto-ticks "Show individual price paths" (`fa68f071..f4cd75ae`)

**Verdict: 0 Critical / 2 Important / 3 Minor / 2 Nit.** The click-stamp gate, the
never-untick rule and the restore/set_props isolation all hold in a real browser, and
every unit/E2E mutant I tried is killed except one that is behaviourally equivalent.
Two things do not hold: the `was` tracker goes stale when the **bins** dropdown changes
(I-1), and a tick click costs **two** `update_bubble` POSTs instead of one (I-2); I-2
has a verified one-word fix.

Reviewed: `1306308b` (test-only fold + docstring) and `f4cd75ae` (the auto-tick).
Done in a private worktree on ports 8062-8066 (not the shared tree, not :8050), now
removed. Firefox via Playwright from `btc_venv`.

---

## Important

### I-1. `window.__qsRegimeFilterOn` is not "updated on every change" -- a bins change leaves it stale, in both directions

`btc_web/layout/mc_controls.py:125-126` (`var was = window.__qsRegimeFilterOn === true;
window.__qsRegimeFilterOn = on;`); the comment at `:112-115` promises the tracker is
"updated on EVERY run". The callback's only Input is `bub-mc-regime.value`. The bins
dropdown (`callbacks/mc_controls.py:117-135`, `_REGIME_JS`) rewrites only
`bub-mc-regime.options`, never `.value`, so the callback does not run when bins
changes -- but the server's notion of "filter on" (`0 < len(value) < bins`,
`mc_helpers.py:59`) does change with bins. The tracker therefore describes the last
*regime value*, judged against the *old* bins.

**(a) A genuine first filter click does not tick.** Real UI actions, Firefox, shipped code:

```
bins 5 -> 7        regs [T,T,T,T,T,F,F]   tracker {on: None}   (server filters: bins 5,6 blocked)
click box 5        paths True             tracker {on: True}   (k=6 < 7, ticks -- fine)
bins 7 -> 6        regs [T,T,T,T,T,T]     tracker {on: True}   paths True   (server: k=6, n=6 => NO filter)
untick paths manually
uncheck box 3      paths -> False         tracker {on: True}   <- all six boxes were checked; this click
                                                                 applies the filter and must tick
```
(`/tmp/.../scratchpad/e2e_bins.py S1`, output pasted above verbatim; the control --
same click on a fresh tracker -- returns `["yes"]` under node.) Node, same inputs:

```
S1.2 bins=6 click uncheck box3 -> k=5<6 (genuine first filter click)   "NU"
S1.ctl fresh tracker, same click (expected tick)                       ["yes"]
```

**(b) The opposite direction -- tick on a click that is not a transition, overriding
the "manual untick" rule.** Raising bins from the default leaves the default value
`[0..4]` in place, so the server filters (bin 5 blocked) and `#bub-mc-regime-note` is
visible, yet the tracker never saw it:

```
bins 5 -> 6   regs [T,T,T,T,T,F]  tracker {on: None}  paths False  note visible: True
click box 0   paths -> True       (filter was already on; the rule says no tick)
```
If the user had ticked paths, then unticked them while this filter was active, the
next regime click re-ticks them -- exactly what the transition rule exists to prevent.
A share link that carries bins>5 with the default regime value restores into the same
state (the callback does not fire because the value is unchanged).

Impact: the feature's own rule ("only on no-filter -> filter, never override an untick")
is violated whenever bins is touched; low frequency (advanced panel), but reachable
from the UI. No test covers a bins change (`test_mc_regime_paths.py` and the E2E file
never touch bins).

**Fix (verified under node, one Input kept):** store the previous *checked count* and
judge it against the *current* bins, seeding the first run with the default count:

```js
var pk = (window.__qsRegimeK === undefined) ? 5 : window.__qsRegimeK;  // 5 = len(default regime value)
var was = (n > 0 && pk > 0 && pk < n);
window.__qsRegimeK = k;
```
```
S1.2 bins 7->6 stale, genuine first filter click (want tick)       ["yes"]
S2 bins 5->6 by user, first regime click (want NU)                 "NU"
default flow: bins 5, uncheck one (want tick)                      ["yes"]
then narrow again, paths unticked by user (want NU)                "NU"
then clear (want NU)                                               "NU"
then re-apply (want tick)                                          ["yes"]
```
The harness in `test_mc_regime_paths.py` resets `__qsRegimeFilterOn` by name, so it
needs the new variable name and two bins-change cases (`bins=7` then `bins=6`; `bins=5`
then `bins=6`), plus one E2E that changes the bins dropdown.

### I-2. A tick click costs two `update_bubble` POSTs instead of one (`allow_duplicate=True` defeats Dash's input blocking)

`btc_web/callbacks/mc_controls.py:165` (`Output("bub-mc-paths", "value", allow_duplicate=True)`).
The brief says the feature must not add a server round-trip. The clientside callback
itself adds none, but the paths write re-triggers `update_bubble`
(`callbacks/charts/__init__.py:191`, Input `bub-mc-paths`) and, because the output id
is `bub-mc-paths.value@<hash>`, the renderer does not hold `update_bubble` back until
the clientside callback has written -- so the regime change fires its own POST with
`paths=[]` first, and the paths change fires a second:

```
label-text click (box4): paths=True  POSTs=2
     (5.22, '..bubble-graph.figure@...', ['bub-mc-regime.value'])
     (5.22, '..bubble-graph.figure@...', ['bub-mc-paths.value'])
second regime click (box3), no tick expected: paths=False  POSTs=1
clear: re-check box3 / box4: POSTs=1 each
```
Three tick rounds repeated the pair (`e2e_race.py`); the final figure was correct every
time (58 traces with paths, 8 without; no stale-response overwrite observed). The first
POST is wasted work: a full Tab-1 build including the ~0.7 s regime ranking (F-21).
Same pattern as `MC_AUTOWIDEN_JS`, so it is inherited rather than new.

**Verified fix:** `bub-mc-paths.value` has no primary writer (the restore's
`apply_tab_bubble` outputs are all `allow_duplicate`), so drop `allow_duplicate=True`
from this one Output. In my copy:

```
round 0: after tick click paths=True traces=58 responses=[(8.83, ['bub-mc-regime.value', 'bub-mc-paths.value'])]
round 1: ... responses=[(8.59, ['bub-mc-regime.value', 'bub-mc-paths.value'])]
round 2: ... responses=[(8.46, ['bub-mc-regime.value', 'bub-mc-paths.value'])]
```
One POST carrying both props. With that change: `test_mc_regime_paths.py`,
`test_no_orphan_callbacks.py`, `test_snapshot.py`, `test_mc_paths_toggle.py`,
`test_callbacks.py`, `test_mc_autowiden.py` = 390 passed, 1 failed (the known
`TestBTCPayPricing::test_free_tier_all_models`); the 4 E2E tests pass; sabotage script
15/15. NOT re-run with the change: the restore E2E files (`test_restore_*_e2e.py`) --
run them before adopting, since the restore also writes `bub-mc-paths.value`.
Keep `prevent_initial_call=True` and the single Input.

---

## Minor

- **M-1. E2E never exercises a label-text click or a keyboard toggle**
  (`test_mc_regime_paths_e2e.py`, every action is `locator(_REGIME).nth(i).click()` on
  the `<input>`). Both work -- checked by hand: label click at (40, 8) ticked paths;
  Space on the focused box ticked paths (`e2e_posts.py`) -- so this is a coverage gap,
  not a defect. One E2E using `#bub-mc-regime label` would pin the `closest()` claim.
- **M-2. Capture-phase is guarded by source text only.** With the asset flipped to
  `}, false)` the unit asset test fails (`test_asset_listens_in_the_capture_phase...`),
  but the E2E tick test still passes on Firefox. Acceptable; the text test is the only
  pin and it holds.
- **M-3. Server and JS read `bins=None` differently.** Server: `_ci(None, 5)` -> 5, so
  `0 < k < 5` filters; JS: `parseInt(null)` is NaN -> off. Unreachable from the UI
  (`bub-mc-bins` is `clearable=False`, options 5-10; placeholder `dbc.Input(value=5)`
  when Markov is absent is hidden); only a hand-edited link could produce it.

## Nit

- N-1. `MC_REGIME_CLICK_MS = 3000` duplicates `MC_AUTOWIDEN_CLICK_MS = 3000`
  (`layout/bubble.py:92`); two copies of one tuning constant.
- N-2. In `mc_stale` mode (`mc_helpers.py:198`) the server ignores the regime
  (`mc_regime=None if mc_stale`) while the JS still ticks paths. Harmless (stale overlay
  is already flagged), noted for completeness.

---

## Verified OK

1. **Registration.** Exactly one callback writes `bub-mc-paths.value` as a plain
   registration: `Output("bub-mc-paths.value@71e1...")`, 1 Input (`bub-mc-regime`),
   2 State, `prevent_initial_call=True`, clientside. The only other writers are the
   restore's global `apply_tab_bubble` (also `allow_duplicate`, Input `bubble-first-render`).
   ```
   $ python regcheck3.py   # app._callback_list + dash._callback.GLOBAL_CALLBACK_LIST
   bub-mc-paths.value  -> bub-mc-paths.value@71e1126a...  | client: True | nin 1 ['bub-mc-regime'] | state 2 | pic True
   bub-mc-paths.value  -> ..bub-qs.value@36d1fa26...      | client: False | nin 1 ['bubble-first-render']
   ```
2. **Guards still fail as designed** (my copy, HEAD): `scripts/sabotage_callback_guards.py`
   -> `15/15 checks behaved as expected`.
3. **Real share-link restore does not tick.** Links built with `_encode_snapshot_v4`
   (MC on, regime `[0,1,2]`):
   ```
   [filter on, paths OFF] after restore: regs=[T,T,T,F,F] paths=False tracker={'on': True}
   [filter on, paths OFF] after first user regime click: paths=False   (already filtering: no tick)
   [filter on, paths ON ] after restore: paths=True
   ```
   (`e2e_restore.py`; no console errors.)
4. **Load paths.** `/`, `/1`, `/2`, `/6`, `/9`, `/10` in dev mode: the "nonexistent object"
   panel lists only `cp/dca/ret/sc-mc-years` and `bub-qs` (all present before this
   change); none of `bub-mc-regime`, `bub-mc-paths`, `bub-mc-bins` (`e2e_loads2.py`).
   No tracebacks in the dev log. With `_app_ctx._HAS_MARKOV=False` the layout still
   contains all three ids:
   ```
   HAS_MARKOV True  {'bub-mc-regime': True, 'bub-mc-bins': True, 'bub-mc-paths': True}
   HAS_MARKOV False {'bub-mc-regime': True, 'bub-mc-bins': True, 'bub-mc-paths': True}
   ```
   Time Machine only writes `bub-mc-enable` (`callbacks/timemachine.py:230`), never regime
   or paths.
5. **Server/JS agreement** (apart from I-1/M-3): `checked=None` and `[]` -> both off;
   all-checked -> both off; `bins` as a string `"5"` -> `parseInt` OK (unit test); stale
   indices >= bins: both use `len(value)`, so they agree.
6. **Click stamp.** `closest('#bub-mc-regime')` fires for label-text clicks and keyboard
   Space (hand-checked, see M-1). The regime checklist has no `disabled` state anywhere
   in `layout/mc_controls.py`. The 3000 ms window is generous: the callback is clientside
   and nothing else outputs `bub-mc-regime`/`bub-mc-bins`/`bub-mc-paths` as a primary,
   so Dash does not queue it behind a server round-trip.
7. **Mutation results, unit file `test_mc_regime_paths.py`** (`-n0`, my copy):
   | mutant | result |
   |---|---|
   | (a) click-stamp check removed | 4 failed (`no_click`, `stale_click`, `restored_filter_remembered`, `stamp_is_consumed`) |
   | (b) `was` check removed | 2 failed (`narrowing_further...`, `restored_filter_remembered`) |
   | (c) "already ticked" check removed | 1 failed (`already_on_is_left_alone`) |
   | (d) untick on clearing (fresh click) | 2 failed (`clearing_the_filter_never_unticks`, `clearing_then_reapplying`) |
   | (d2) untick on clearing (any run) | 3 failed |
   | (e) tracker never updated | 2 failed |
   | asset bubble-phase | 1 failed (capture-phase text test) |
8. **Mutation results, E2E file** (4 mutant servers + 1 clean, same tests):
   clean 4 passed; (a) fails `test_a_programmatic_regime_write_leaves_paths_alone`;
   (b) fails `test_a_manual_untick_survives_the_next_regime_click`; (d) fails
   `test_clearing_the_filter_does_not_untick_paths`; (c) **survives** -- expected: with
   a single-option checklist, returning `["yes"]` over `["yes"]` is a no-op, so only the
   unit file can see it. No E2E test is vacuous: each starts by proving the callback
   fires (a tick, or a `set_props` that visibly changes the regime).
9. **Fold `1306308b`.** `test_fewer_sims_than_the_subset_draws_the_best_of_it` in
   `test_mc_regime_subset.py` (26 tests per run):
   | mutant (callbacks/charts/__init__.py unless noted) | result |
   |---|---|
   | best-first trim deleted (`pass`) | FAILED (only that test) |
   | trim takes the worst 8 (`[-display_sims:]`) | FAILED |
   | trim takes every 5th path | FAILED |
   | trim keeps best 8 in reverse order | FAILED |
   | ranking reversed (`mc_overlay.py`, `argsort(-n)`) | subset file alone: 26 passed (its oracle calls the same function); `test_callbacks.py::TestFilterPathsByRegime` fails 2 -- the ranking is pinned there |
10. **Baseline tests in the worktree at HEAD:** `test_mc_regime_paths.py` +
    `test_mc_regime_subset.py` -> 47 passed. The `bubble.py` docstring change matches
    the code (`elif is_free: fan_is_reference = True` only when no model).

## Not covered

- The restore E2E suites (`test_restore_*_e2e.py`) were not run, with or without the
  I-2 change.
- No deploy-shape run (gunicorn, Redis, nginx `burst=200` under the extra POST).
- Mobile/touch behaviour of the click stamp; WebKit/Chromium (Firefox only).
- Out-of-order arrival of the two I-2 responses was not provoked (3 rounds, all fine);
  the final-figure guarantee rests on Dash's own response handling.
- The full non-E2E suite was not re-run; the controller's 3246 passed / 1 known fail
  stands. `test_free_tier_all_models` is the known local-only BTCPay failure.
- Scratch scripts (in the session scratchpad, not the repo): `e2e_bins.py`, `e2e_posts.py`,
  `e2e_race.py`, `e2e_restore.py`, `e2e_loads2.py`, `seq.js`, `fix.js`, `mutate.py`, `mut2.py`.
