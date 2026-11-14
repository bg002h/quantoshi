C/I/M/N = 0/1/4/4

# Review — `2026-11-14-pctile-cycle-slider-labels-design.md` (commit 0ccd8840)

Question: can a builder implement this note correctly as written; does any
decision give a wrong result, break old share links, or break the callback graph?

Answer: yes, with one Important fix to D7's trigger. The rest of the note
holds up against the code.

## What I checked and found sound

- **D4, v4 links.** `_decode_snapshot_v4` fills every omitted field from the
  link's *registered* fingerprint defaults (`historical.get(key)`). Once the
  new fingerprint is registered with `bub-pctile-fit-labels:value = ['labels']`,
  every new link decodes *with* the key, even though the sparse diff leaves it
  out at its default. A `757574d3` link decodes *without* it (that registry
  entry has no such key). Registry check: the registry holds 20 entries,
  `757574d3` is the newest (index 19), and the current fingerprint is
  `757574d3`. Appending the new one drops `028d77e5`, not `757574d3`.
- **D4, positional links.** Positional links pad short arrays with `None`, so
  they lack the key. `manage_snapshot` encodes only v4 (snapshot_cb.py:331), so
  no new positional links are made. Any positional link also predates the
  cycle fit, so `_fill_predating_fields` sets `bub-pctile-fit` to `[]` and D7
  does nothing for it.
- **Restore paths.** Both live restore entry points (snapshot_cb.py:69, :340)
  go through `_decode_snapshot_by_prefix`. The two decoders that matter
  (`_decode_snapshot`, `_decode_snapshot_v4`) both return through
  `_fill_predating_fields`. `restore_builder` consumes the decoded state and
  never precomputes the Percentile figure. So D7 belongs inside, or right after,
  `_fill_predating_fields`. Putting it there covers every path; v1 links predate
  the feature.
- **The new id has to go in `_TAB_CONTROLS["bubble"]`.** If it is left out,
  snapshot_cb.py:213 sorts it into `_GLOBAL_CONTROLS`, which still works but is
  not the intended bucket. D4 lists this step.
- **D3.** The hidden `bub-pctile-fit-horizon` stays in the layout and stays a
  per-tab restore Output and a share `State`, so nothing becomes orphaned.
  Removing it as an `Input` of `update_bub_pctile` leaves no dangling reference.
- **D1/D2 against `build_percentile_figure` / `_cycle_overlay`.**
  - Ticks: `_year_ticks(xmin, xmax)` produces `yr_to_t(xmax)`, which is
    bit-identical to `t_edge`, so the `<= t_edge` filter keeps the 2080 tick.
    Step is 5 over 70 years.
  - `turns(fit, t_lo, t_end)` already limits markers and labels to the axis.
  - `np.interp` beyond the grid would clamp silently, but D2's 2081 grid end
    covers the slider's domain.
  - Timing error bars past the axis are clipped by the plot area, as they are
    today.
  - Changing the params schema (`cycle_horizon` removed, `cycle_labels` added)
    changes every cache key, so skipping FLUSHDB serves nothing stale.
- **D5, log axis.** Labels keep using `ax_x()`, which is log10 on a log axis,
  so they are placed correctly.

## Findings

### I-1 (Important): D7's trigger fails for links whose fingerprint has been evicted
- **Location:** D7 ("every link made after it carries that key, per D4's
  reasoning"); snapshot.py:960-973.
- **Failure scenario:** after about 19 more defaults changes, `757574d3` is
  dropped from the 20-entry registry. A Percentile link with fit on at that
  fingerprint then takes the unregistered branch, where
  `historical = dict(SNAPSHOT_DEFAULTS)`. That branch only removes
  `_PREDATING_LINK_VALUES` keys that the oldest entry lacks, so the link gets
  today's `bub-pctile-fit-labels` value. D7 sees the key, does not fire, and
  the link restores with its forecast cut back to its old slider end (the
  operator's link: Jan 2027 instead of about 2047). That is exactly the
  silent breakage D7 exists to prevent. The note's premise ("absence means
  older") does not hold on this path, which the codebase already handles for
  other fields.
- **Fix:** give D7's trigger key the same treatment in the unregistered
  branch. For example, add `MIGRATION_SENTINELS = ("bub-pctile-fit-labels:value",)`
  and include it in the pop loop at :971 (`for k in (*_PREDATING_LINK_VALUES,
  *MIGRATION_SENTINELS)`). Add a test: an unregistered-fp link whose oldest
  registry entry lacks the labels key still triggers D7. Also correct the
  sentence in D4/D7.

### M-1 (Minor): `max(e, …)` does not reproduce the sender's forecast when `e` was past `today + h`
- **Location:** D7, the migration formula.
- **Failure scenario:** prod drew the axis to `today + h` whatever `e` was,
  as long as `e > current year`. A link with `h = 10` and `e = 2060` showed
  the forecast to about 2036. D7 restores 2060, which is 24 years longer than
  the sender saw. Links at the default end (2033 < current year + 11) come
  out exact, so only links whose slider was moved are affected.
- **Fix:** decide explicitly. To match "keep their forecast length", set the
  end to exactly `current_year + h + 1`. If keeping `max` is intended, say so
  and drop the claim that the length is kept.

### M-2 (Minor): D7 can write a slider value above `XRANGE_MAX`
- **Location:** D7.
- **Failure scenario:** from 2050 on, `current_year + 31` exceeds 2080. The
  RangeSlider gets a value outside its `max`, and the figure asks for
  `t_edge` beyond the D2 grid, where `np.interp` clamps the bands flat.
- **Fix:** `min(…, XRANGE_MAX)`, using a snapshot-local constant or a test
  that pins it equal to `layout.bubble.XRANGE_MAX` (same layering rule as D2).

### M-3 (Minor): the 3-line label overflows the plot top and bottom
- **Location:** D5, placement.
- **Failure scenario:** the y headroom is `[-12, 112]`, sized for today's
  2-line label centred at `yshift = 24`, whose top sits about 40 px above the
  marker. With `yanchor="bottom"` and the same `yshift = 24`, a 3-line box
  (about 48 px tall) reaches about 72 px above the marker. A Q90+ peak on a
  ~450 px chart, or any peak on the 55vw mobile chart, puts the box over the
  title and subtitle. Troughs below Q10 run into the x tick labels the same
  way.
- **Fix:** specify `yshift` of about 6–8 px with the new anchors, and either
  enlarge the headroom or test that the label stays inside the plot area for
  a Q95 peak.

### M-4 (Minor): labels near the right edge spill off the axis
- **Location:** D5 / "Review focus" bullet 1.
- **Failure scenario:** turns are already limited to `t_end`, but a
  centre-anchored box for a turn just before `t_edge` hangs about half its
  width (~80 px for the "range … – …" line) past the axis into the legend or
  margin. This is likely at a slider end of 2027: the next trough (about
  Dec 2026 – Jan 2027) sits right at the edge. That is exactly the case the
  Review-focus bullet raises, and the note gives no rule for it.
- **Fix:** use `xanchor="right"` when the turn's `ax_x(x)` falls in the last
  ~15% of the axis span (the mirror rule on the left). This is one
  conditional, not collision logic.

### N-1 (Nit): D1 states the rule twice, differently
D1 writes `yr_to_t(xmax) > today_t`, then says "today's rule is `>=`, keep it".
Write `>=` once.

### N-2 (Nit): the hidden horizon dropdown needs literal options
D3 deletes `pctile_cycle.HORIZONS`, but `layout/common.py:572-574` builds the
dropdown's options from it. Keep the options as `[10, 20, 30]` written out, so
a restored `30` is not sanitised away by a dropdown with no options.

### N-3 (Nit): the two restore tests in Plan step 3 test the wrong layer
A decoded `757574d3` link or positional link does not *contain*
`bub-pctile-fit-labels`; the value `['labels']` comes from the layout default
because the apply callback leaves the field untouched. Word the tests as "key
absent from the decoded state, and the layout default is `['labels']`".

### N-4 (Nit): a D4 sentence contradicts itself
"`_TAB_CONTROLS` / deep link (routing.py) need no new outputs" conflicts with
the earlier "`_TAB_CONTROLS["bubble"]`" step. Reword it to: deep link needs no
new outputs; `_TAB_CONTROLS["bubble"]` gains the id. Also have D7 skip states
missing `bub-xrange` or with a non-list value.
