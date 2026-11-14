C/I/M/N = 0/2/3/2

# Whole-diff review — pctile cycle fit: slider-driven extrapolation + labels switch

- Diff: `git diff f0cf95c5..3af61574` (implementation commit `3af61574`)
- Note: `docs/superpowers/specs/2026-11-14-pctile-cycle-slider-labels-design.md` (D1–D8)
- Prior review: `docs/superpowers/reviews/2026-11-14-pctile-slider-labels-note-review.md`
- Reviewer: independent (Opus), 2026-11-14. No code edited.

Question: does this diff ship a wrong result, break old share links, break the
Dash callback graph, or regress another Tab-1 view?

Short answer: the callback graph, the other Tab-1 views and today's links are
fine. Two Important findings. (1) D8 makes a pre-existing turn-index bug
reachable: wrong timing ranges, error bars and range prices whenever the
slider's *left* handle is past today. (2) The I-1 fold does not survive the
eviction it was written for: once `757574d3` itself is evicted, the operator's
link loses its forecast again. Both are reproduced below.

## Verified clean (ran code where possible)

- **D8 cap.** `XCAPPED_MODES = {resid, occupancy}`, and the N-future hide
  still uses all of `HISTORICAL_MODES`. Percentile → Residuals/Occupancy:
  the cap callback re-caps the value to next year (same JS as before).
  Price → Percentile: inherits 2033. CAGR swap: `toggle_bub_view` only
  swaps the `[2025,2050]` ↔ `[2010,2033]` defaults, and that logic is
  unchanged. Deep link `/1.4` writes no xrange (`extras` = NU). CTA
  (`custom_time.py`) writes only min/marks/value and needs no change.
- **Fit off / weak with xmax 2080.** The axis is capped at today
  (`fitoff2080`, `bmweak2080`: xr end 17.31 = today). Other models'
  percentile lines use `t_data` masked by `t_hi = min(xmax, today)`, so they
  never extend past today.
- **Edge windows.** No crash for `[2026,2027]`, `[2027,2028]`,
  `[2030,2060]`, log 2080, censor, `log`/`calw` bases, or labels off.
  A turn near the right edge anchors `right`. A window with only one kind of
  forecast turn yields 1 label. Turns beyond `xmax` are never labelled
  (`turns(fit, t_lo, t_end)`).
- **Customdata alignment.** Forecast marker cd is 7 wide after
  `_add_date_hover` prepends the date:
  `[date, kind, $P, a, b, $A, $B]`. The template indices
  `[1],[0],[2],[3],[4],[5],[6]` read correctly. The label reads the
  pre-prepend row `row[2..5]` = a, b, $A, $B, which is also correct.
  `_logfan` → `(qs, lf)` → `fan_prices(lf, qs, …)` uses the correct
  argument order.
- **Log-axis xanchor.** `frac` is computed in `ax_x` space (log10), the
  same space as the log `xaxis.range`, so it is consistent.
- **`cycle_labels` default.** The builder defaults it to `True`. Fit-off
  `_cycle_params` pins it to `True`, so there is one cache key. Prewarm does
  not build the pctile figure. `_get_pctile_fig` keys on the full param
  dict. No leftover users of `cycle_horizon`, `HORIZONS`, `DEFAULT_HORIZON`
  or `_LINEAR_LABELS` remain outside tests.
- **D7 decode paths.**
  - Both `_decode_snapshot` (q2/q3 positional) and `_decode_snapshot_v4`
    return through `_fill_predating_fields`, so both run D7.
  - The migration is idempotent if decoded twice.
  - A new link with labels `[]` is encoded as mask 0, so the key is present
    and D7 is skipped (test `off[_LAB] == []`).
  - A new link with labels at the default omits the key from the diff and
    gets it back from the `7ac5a01e` defaults, so D7 is skipped.
  - A "current tab only" link from another tab has no view-mode, so it
    defaults to price and D7 is skipped.
  - A non-percentile old link is untouched.
  - `_XRANGE_MAX == layout.bubble.XRANGE_MAX` is pinned by a test.
  - The operator's link restores `[2010, 2047]`.
- **Callback graph.** `update_bub_pctile` swaps the horizon Input for the
  labels Input, which sits in the same `bub-pctile-ctl-wrap` container.
  `_PCTILE_SUB_CONTROLS` includes labels, so toggling it with the fit off is
  a no-op. Index 352 is appended at the tail, and `_TAB_CONTROLS` and
  `_CHECKLIST_OPTIONS` are updated.

## Findings

### I-1 (Important): D8 exposes a turn-index miscount — wrong timing ranges, bars and range prices when the left handle is past today
- **Location:** `btc_web/figures/percentile.py:432-469`. `tturns =
  _pc.turns(fit, t_lo, t_end)`, then `seen[k]` counts only the *visible*
  forecast turns, and that count is passed as the cycle index `j` to
  `_pc.turn_timing(params, anchor, j, …)`.
- **Why now:** before D8, the Percentile slider was capped at next year, so
  `xmin ≤ 2026` and every forecast turn (all `> t_last ≥ t_lo`) was visible.
  The count was then right. D8 uncaps the slider, so `xmin` up to 2079 is
  reachable by dragging the left handle. It is also reachable by arriving from
  a user-adjusted CAGR range (only the CAGR *default* is swapped back), or
  from a share link.
- **Reproduced:** fit on QR, cal, linear, xmax 2060. Timing of the Jun 2030
  trough and Apr 2032 peak:

  | xmin | Jun 2030 trough range | Apr 2032 peak range |
  |---|---|---|
  | 2010 / 2026 | Mar 2030 – Oct 2030 (correct) | Dec 2031 – Jul 2032 (correct) |
  | 2027 | **Sep 2026 – Jun 2030** | Dec 2031 – Jul 2032 |
  | 2030 | **Sep 2026 – Jun 2030** | **Jun 2028 – Apr 2032** |

  The wrong ranges reach the hover, the error bars, and the label's
  `range … / $A – $B` lines. For example, xmin 2030 labels the trough
  `range Sep 2026 – Jun 2030 | $79k – $227k`. These are confidently wrong
  numbers on screen.
- **Fix:** derive the cycle index from all forecast turns, not the visible
  ones. Compute `fturns = _pc.turns(fit, t_last, t_end)`, assign `idx` and
  `timing` over that list, then keep only rows with `x >= t_lo` (or count
  the forecast turns in `(t_last, t_lo)` per kind and start `seen` there).
  Decide whether the "next peak/trough" labels go on the first *visible*
  turn of each kind (as now) or only on true `j == 0`. Add a test:
  timing/customdata for a given turn is identical at xmin 2010 and xmin
  2030.

### I-2 (Important): the I-1 fold does not survive eviction of `757574d3` — the operator's link loses its forecast exactly in the scenario I-1 described
- **Location:** `btc_web/snapshot.py:1014-1019`. In the unregistered-fp
  branch, `_MIGRATION_SENTINELS` keys are popped only `if k not in oldest`.
- **Failure scenario:** the registry order is `… 0eac47fc, 757574d3 (fit,
  no labels), 7ac5a01e (labels)`. `757574d3` is the *last* pre-change fp.
  When it is evicted (after 19 more registrations), every remaining entry
  carries the labels key, including the oldest. A `757574d3` link then gets
  the key from `SNAPSHOT_DEFAULTS`, D7 does not fire, and the link restores
  `[2010, 2027]`. That is the silent breakage I-1 named. The fold only helps
  links at fps *older* than `757574d3`. Those predate the cycle fit
  (fit → absent), so D7 is inert for them anyway. In practice the fold
  protects nothing.
- **Reproduced:** with `_registry_cache` replaced by `{7ac5a01e + 19
  fakes}`, decoding the operator's link gives
  `bub-xrange [2010, 2027]` with the labels key present. With today's
  registry it gives `[2010, 2047]`.
- **Mitigation present:** `test_evicted_fingerprint_still_migrates`
  (deadbeef) will go red when that registry state arrives. The tripwire
  exists, but the guarantee the note claims ("D7 migration must survive
  fingerprint eviction") is not met.
- **Fix (pick one):**
  - (a) Pin pre-change fingerprints such as `757574d3` against eviction in
    `tools/update_defaults_registry.py` (and `_load_registry` is unaffected).
    This is the simplest and exact.
  - (b) Record the introducing fp per sentinel, e.g.
    `_MIGRATION_SENTINELS = {"bub-pctile-fit-labels:value": "7ac5a01e"}`,
    and pop when the oldest entry lacks the key **or the oldest entry is
    the introducing fp**. An unregistered fp is older than the oldest
    retained entry, so it is pre-change. This stays exact until `7ac5a01e`
    itself is evicted. Pair it with (a) or accept the ambiguity after that.

  Add a test that evicts `757574d3` (patch `_registry_cache` as above) and
  asserts `[2010, y+21]`.

### M-1 (Minor): time-bomb test — `test_label_near_slider_end_anchors_right` will go red in ~2 weeks, and its anchor assert is guarded
- **Location:** `btc_web/test_pctile_cycle.py`, `test_label_near_slider_end_anchors_right`.
- **Failure scenario:** it builds at `xmax = today.year + 1` and asserts
  `labs` is non-empty. It passes today only because the next QR trough
  (Nov 26 2026) sits in the ~6-week window. Once the daily data passes that
  turn (~late Nov 2026), the window (last close → 1 Jan 2027) contains no
  forecast turn, so `assert labs` fails. Through 2027 (window → 1 Jan 2028;
  next peak Sep 2028) it stays red, and the full suite with it. Separately,
  the `if frac > 0.85:` guard means the xanchor assert can pass vacuously.
- **Fix:** pick the window from the fit, e.g. set `xmax` to the year after
  the first forecast turn, or monkeypatch `_pc.turns` to return a turn at
  97% of the axis. Assert unconditionally that the right-edge label anchors
  `right`.

### M-2 (Minor): leaving and returning via Residuals/Occupancy silently shrinks the forecast to ~weeks
- **Location:** `callbacks/charts/__init__.py:738-753` (cap re-applies,
  correctly) plus D1.
- **Scenario:** Percentile at 2033 → Residuals (value capped to 2027) →
  Percentile. The slider stays 2027, so the forecast region runs from the
  last close to 1 Jan 2027. In Dec 2026 that is a few days, with no turn and
  no label. Before this diff the Percentile view drew today + 20 regardless.
  This is consistent with the note (the slider drives it), but the result
  looks like "no forecast" with no hint.
- **Fix (optional):** when entering Percentile from a capped view with
  `r[1] <= next year`, restore `[r[0], 2033]` (mirroring the CAGR default
  swap), or accept and document.

### M-3 (Minor): `xmin > today` leaves a stray one-point in-sample trace and labels the first *visible* turn as "next"
- **Location:** `figures/percentile.py:410-412`. With `t_lo > t_last`,
  `ins = [t_last]`, a one-point "cycle fit" trace left of the axis, still in
  the legend. The labels go on the first visible peak/trough, not the actual
  next ones (see I-1 for the index part).
- **Fix:** skip the in-sample trace when `t_lo > t_last`. Fold this in with
  I-1.

### N-1 (Nit): `test_label_at_right_edge_anchors_right` re-implements the production formula
It recomputes `frac` with the same 0.15/0.85 thresholds on the default
window, where both labels are `center`. It cannot catch a wrong threshold
in a way the builder would not share. M-1's fix covers the edge case.

### N-2 (Nit): `test_fit_off_and_weak_stay_capped_at_today` relies on BM being weak with today's data
If a refit makes BM non-weak, the test fails for a reason unrelated to the
cap. Monkeypatch `weak` instead (the same pattern as `beats_flat` elsewhere
in the file).
