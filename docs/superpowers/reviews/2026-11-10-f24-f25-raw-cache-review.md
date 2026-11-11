# Review: F-24 / F-25 / Raw redraw cache (`9589f564..cad7e6e2`)

**Counts: 0 Critical · 1 Important · 2 Minor · 2 Nit**

Scope: only `git diff 9589f564..cad7e6e2`. No source file edited. Nothing in this
diff gives a wrong final result or breaks a share link. One finding (I-1) adds a
wasted server round-trip on the most common user action the diff targets.

What I checked and found sound:
- **F-24.** `_CHECKLIST_OPTIONS` is consumed only by `_list_to_mask` /
  `_mask_to_list` (snapshot.py:644-955). Widening only appends bits. No fingerprint
  or fixed-width packing depends on the table length. `bub-mc-regime` is not
  bitmasked (it encodes as a plain list), so it never had the truncation.
- **Cache key.** `RawRequest` is a frozen dataclass, and every input `simulate`
  reads is a field of it: the seed (`seed_for`) is a pure function of the request,
  and the series is pinned by `data_fp` plus `_CACHE_CLEARERS`. `today` (normalised)
  and `p0` are in the key, and `diagnostics` is always False here. `set_prices` runs
  once per process (app.py:76). So a stale hit is not possible in production.
- **No consumer writes to the cached arrays.** `_add_mc_fan` (np.percentile),
  `select_line_paths` (argsort on a column), `_add_mc_spaghetti` and
  `_add_mc_extremes` (fancy-index copies) only read. The stack multiply makes a
  new array. Read-only flags would raise on any future in-place write.
- **Threading.** The lock guards every dict access. Shared values are read-only.
- **F-25 tracker.** The hash never reaches the server, so a restore always arrives
  as callback outputs (`apply_tab_*` on `*-first-render`, all `allow_duplicate`)
  or as `set_props`. Either way the tick callback fires, and `__qsBinsPrev` learns
  the restored count before any user raise. Lazy tabs mount with layout defaults
  (5 = `MC_BINS_DEFAULT`). Lowering bins then raising behaves as designed. The tick
  write does not stamp `__qsMcRegimeClickAt`, so `MC_REGIME_PATHS_JS` sees k == n
  ("no filter") and does not tick paths. That is correct.
- **Tests.** The new tests make real assertions. The re-hits in
  `test_raw_start_price_falls_back_to_last_close` still discriminate: a wrong
  fallback p0 changes the key and so misses.

---

## Important

### I-1. A user bin raise costs two chart POSTs, and the first one computes the stale filtered scenario

`btc_web/callbacks/mc_controls.py:186` (`Output(_regime_id, "value", allow_duplicate=True)`).

The chart callbacks on Tabs 1-5 take `*-mc-regime.value` (and `bub-mc-raw-regime.value`)
as an **Input** (update_bubble, update_dca, update_retire, update_heatmap,
update_supercharge). The tick's output id is `…-regime.value@<hash>`. As in the
earlier review's I-2 (`2026-11-10-tab1-regime-autotick-review.md`), the renderer
therefore does not hold the chart callback back until the clientside tick resolves.
A bins raise fires the chart once with (new bins, **old** regime) and then a second
time when the tick lands.

Measured on a DEV server at `cad7e6e2` (port 8061, Firefox). Tab 1 Raw cycle, user
picks 8 in `#bub-mc-raw-bins`, all 5 bins ticked:

```
POSTs: 2
  ['bub-mc-raw-bins.value']      <- bins 8, regime [0..4]: a FILTERED run (3 of 8 bins excluded)
  ['bub-mc-raw-regime.value']    <- bins 8, regime [0..7]
```

The final figure is correct. But the default state is "all ticked", so this happens
on almost every user raise, which is exactly the path F-25 targets. The first POST
is a full Tab-1 build of a scenario the user never asked for:
- on Raw, a filtered simulation (the dearer branch) that also takes one of the two
  redraw-cache slots, evicting a useful entry;
- on the Tab-1 model source, the ~0.7 s regime ranking (F-21);
- on Tabs 2-5 with MC on, an MC computation for the stale filter.

This conflicts with the ALARA standing preference.

**Fix (verified):** drop `allow_duplicate=True` from this one Output. No primary
writer of any `*-mc-regime.value` exists: every other writer (the six `apply_tab_*`
restore callbacks) is `allow_duplicate`, so the tick can be the primary. It keeps
one Input and `prevent_initial_call=True`, so the Dash 4.0 rule is unaffected.

I emulated the fix without editing source: a Playwright route stripped the `@hash`
from those two clientside callbacks in `/_dash-dependencies`. Result:

```
POSTs: 1 [['bub-mc-raw-bins.value', 'bub-mc-raw-regime.value']]
ticked: all 8;  fan drawn
set_props bins=10 after set_props regime [0,1,2] -> regime stays [0,1,2] (not ticked)
```

Also update the comment at mc_controls.py:169-173 ("hence allow_duplicate").
`test_mc_regime_bins.py::TestF25Wiring::test_shape` asserts the `@` hash and needs
to flip.

Not re-run with the change: `scripts/sabotage_callback_guards.py`,
`test_no_orphan_callbacks.py`, and the `test_restore_*_e2e.py` files. The restores
write these regime values, so run all three before adopting.

---

## Minor

### M-1. float32 paths times a large stack overflow to `inf` and drop points from the extreme line

`btc_web/callbacks/charts/_raw.py:115` (`_frozen(res.paths, np.float32)`) together
with `_raw.py:177` (`paths = paths * stack_f`).

The multiply stays float32. That holds under NumPy 2 (NEP 50) and under 1.x
value-based casting. Measured worst-case path: Raw, trend "none", 40 y, window
2010-2026, 3200 sims, p0 1e5 gives a max of **1.1e32**. That leaves only ~3e6x of
headroom to float32's 3.4e38.

With "Show stack" on and a stack of about 3 million BTC or more (no `max` on
`bub-stack`), the top paths become `inf`. `_sig_round_list` passes `inf` through
and Plotly serialises it as `null`, so "Highest end" and some example lines get
gaps. Before this diff it was float64 and could not overflow. The fan percentiles
are unaffected unless the stack is far larger. This needs absurd input, but
someone typing 21,000,000 BTC is plausible.

**Fix:** do the stack multiply in float64:
`paths = np.multiply(paths, stack_f, dtype=np.float64)`. That is a transient per
request; the cache stays float32.

Otherwise float32 changes no visible number. The relative error is about 6e-8,
far below the 3-significant-figure path rounding and the hover format.

### M-2. `test_price_change_clears_the_cache` never changes the price

`btc_web/test_raw_mc_callback.py:760-764` only asserts that `clear_cache` is
registered in `_CACHE_CLEARERS`. It never calls `raw_mc.set_prices`, and it never
checks that the cache empties. The name claims more than the test checks.

**Fix:** after `_run()`, call `raw_mc.set_prices(*raw_mc.get_prices())` and
`assert not _raw._cache`.

---

## Nit

- **N-1.** `_raw.py:110-125`: the lock is released while `simulate` runs, so two
  threads missing on the same request both simulate. This is harmless (the
  results are identical, and gunicorn uses sync workers). A note would stop a
  future reader from "fixing" it.
- **N-2.** `assets/mc_bins_tick.js`: a stamp is consumed only when the bins
  callback fires. Clicking the bins dropdown without changing its value leaves a
  stamp live for 3 s. If the MC "restore" button (`_RESTORE_MC_JS`, which writes
  `*-mc-bins`) raises bins inside that window, the tick treats it as a user raise.
  Only when all old bins were ticked, so the result is the same as a user raise.
  Cosmetic.
