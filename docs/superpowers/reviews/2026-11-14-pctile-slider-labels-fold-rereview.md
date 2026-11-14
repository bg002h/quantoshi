C/I/M/N = 0/0/1/1

# Fold re-review: pctile slider and labels whole-diff fold (a41501ff)

- Scope: `git show a41501ff -- btc_web/`. This is the fold of
  `2026-11-14-pctile-slider-labels-whole-diff-review.md` (I-1, I-2, M-1..M-3,
  N-1, N-2). The rest of the feature was not re-audited.
- Reviewer: independent (Opus), 2026-11-14. No code was edited.
- The controller had already machine-verified these, so they were not re-run:
  the suite, the red-on-unfixed check for the new tests, and the sabotage
  guards.

## Per-finding verdicts

### I-1 is fixed
- **Counting.** `hidden` = the forecast turns in `(t_last, t_lo)`, taken from
  `turns(fit, t_last, t_lo)`. Both ends are excluded.
- **Boundary at `t_lo`.** `turns()` is inclusive on both ends
  (`t0 <= tk <= t1`). A turn at exactly `t_lo` therefore appears in the
  visible list `turns(fit, t_lo, t_end)` and is excluded from `hidden` by the
  strict `<`, so it is never counted twice. Both calls compute `tk` with the
  same expression for the same `k`, so float equality behaves consistently.
- **Boundary at `t_last`.** A turn at exactly `t_last` is classed as past
  (`r[0] <= t_last`), and `hidden` excludes it too. This is consistent.
- **Cycle index.** `seen` starts at the per-kind hidden count and is
  incremented before use. So `j` is the turn's true 1-based index after the
  in-sample anchor, which is the same semantics as before. `anchor` is taken
  from `cyc["t"][0]..t_last`, which does not depend on `t_lo`. The `j = 0`
  branch with no anchor is unaffected.
- **Labels.** `idx` uses a separate `vis` counter, so the labels go on the
  first *visible* turn of each kind. The label text names the kind and date
  only (it never says "next"), so this choice is honest.
- **Run beyond the new test.** Forecast-turn customdata at xmin 2026, 2027,
  2028, 2029, 2031, 2035 and 2045 (linear and log x) is identical to xmin
  2010 for every shared turn. Each window shows 2 labels.

### I-2 is fixed
- **Mechanism.** `_MIGRATION_SENTINELS` maps the labels key to `{757574d3}`.
  The registry confirms that `757574d3` is the only entry that has
  `bub-pctile-fit` and the horizon but no labels key. Every older entry has
  none of the three.
- **Unregistered-fp branch.** The key is popped when
  `fp_payload in pre_fps`, or when the oldest entry lacks it.
  - The `pre_fps` test does not depend on `oldest`. So with an empty or
    unloadable registry (`oldest is None`), a `757574d3` link now still
    migrates. Before, nothing was popped.
  - Any other unregistered fp with `oldest is None` keeps today's default,
    as it did before.
  - The `_PREDATING_LINK_VALUES` loop is unchanged: it still runs only when
    `oldest` is present.
- **Registered `757574d3`.** The branch is not reached, and the entry lacks
  the key anyway.
- **New test.** It leaves only the current fp in the registry, so the oldest
  entry *has* the key. The test therefore exercises the `pre_fps` path
  specifically. The operator link restores `[2010, y+21]`.

### M-1 is fixed
The test now places `xmax` from the fit: the year after the first forecast
turn. That turn sits more than 0.85 of the axis from the 2010 start, so the
test does not depend on the date (until the 2060s). It asserts
unconditionally that exactly one label sits at that turn and that it anchors
`right`. The `mid` check (2010–2080, all labels `center`) holds on linear and
log axes.

### M-2 is fixed, with one new Minor and one new Nit (below)
- **Shadowing.** `from datetime import date` shadows nothing. There is no
  other `date` name in the module, and no parameter or local is called `date`.
- **Cap value.** The value matches what the cap writes:
  `(new Date()).getFullYear() + 1` versus `date.today().year + 1`.
  - The cap only lowers `r[1]` (and moves `r[0]` to `new_max - 1` if `r[0]`
    is past the cap).
  - So any state arriving from Residuals or Occupancy with a future end has
    `r[1] == next year` exactly.
  - Around New Year, browser and server years can differ for a few hours. The
    check then just misses and falls back to `no_update`, the old behaviour.
    This is benign.
- **Ordering.** `toggle_bub_view` writes the mode and the xrange in one
  response. The cap callback is triggered by the mode and reads the xrange
  State after that response, so it sees `[r0, 2033]`. In Percentile mode
  (max 2080) it does not cap.

### M-3 is fixed
The in-sample trace is skipped when `t_lo > t_last`. Nothing else indexes
`front[0]`. The forecast trace still starts at `t_last`, entering from the
left edge. The new test asserts that the trace is gone, that the
extrapolation is present, and that 2 labels show.

### N-1 and N-2 are fixed
- **N-1.** The formula-mirroring test was removed, and its coverage moved to
  the M-1 test.
- **N-2.** `fp._pc.is_weak` is the function `_cycle_result` calls
  (`percentile.py:263`). The `lru_cache` is cleared before and after (in a
  `finally`), so the patch takes effect and does not leak.

## New findings

### M-1 (Minor): the M-2 fold clobbers a user-chosen end equal to next year
- **Location.** `callbacks/charts/__init__.py:684-689`.
- **What happens.** The condition `cur_xrange[1] == next year` cannot tell a
  capped value from a deliberate one. A user who sets the end to next year
  gets it rewritten to 2033 in three cases:
  - in Price or CAGR, then clicking Percentile;
  - in Percentile, by re-clicking the already-active Percentile pill;
  - in Percentile, by going to Price and back.
- **Docstring.** The function's docstring still promises that "a user-chosen
  range is never clobbered", which is now false.
- **Impact.** Low. The value is a single slider drag away, and no wrong
  number is shown. Only the range changes.
- **Fix.**
  - Add `State("bub-view-mode", "data")` and restore only when the
    *previous* mode is in `XCAPPED_MODES`. A plain State adds no Input, so
    it does not trigger the stale-deps issue.
  - Alternatively, keep the current behaviour and amend the docstring to
    name the exception.

### N-1 (Nit): the literal 2033 in the M-2 branch will go stale
- **Location.** `callbacks/charts/__init__.py:689`.
- **What happens.**
  - From 2032 on, `next year >= 2033`, so the restore no longer works.
  - From 2033 on, it writes an end *below* the cap: `[r0, 2033]` with
    `r0 <= next year`. That can be a past end (no forecast at all) or an
    inverted range.
- **Context.** The same literal already appears in the neighbouring
  CAGR-swap lines, so this matches existing debt.
- **Fix.** Take the end from `tab_defaults.BUBBLE`, or use
  `max(2033, next year + 1)`.

## Out of scope, for the controller (not counted)
`_PREDATING_LINK_VALUES["bub-pctile-fit:value"]` has the eviction hole that
I-2 just closed for the sentinel.
- **When it opens.** Once `0eac47fc` (the last fp before the fit) is evicted,
  the oldest retained entry carries `bub-pctile-fit`. The pop is then skipped.
- **Effect.** A pre-fit Percentile link would get `fit` ON from today's
  defaults: "a forecast its sender never saw".
- **Not from this fold.** This existed before the fold. A `pre_fps`-style
  set (all 18 pre-fit fps, or "any fp not at or after `757574d3`") would
  close it. Worth filing as a follow-up with an owning phase.
