# Fold re-review: percentile cycle fit, `a503e491` (pre-deploy gate)

**Scope:** `git show a503e491` only. It answers `2026-11-10-pctile-cycle-whole-diff-review.md` (0C/1I/6M/4N). The feature itself was not re-audited.
**Reviewer:** independent and read-only, except for this file. Scratch scripts are in `/tmp/claude-1000/-scratch-code-bitcoinprojections/6c13708a-e257-46cf-bdef-a2a8a773488f/scratchpad/fr/`.

**Counts: 0 Critical / 0 Important / 1 Minor / 4 Nit**

**Verdict:** The fold resolves I-1 and every Minor it took on, and it introduces no blocking defect. It is clear to deploy. The new Minor is a rounding-precision issue in `_money()`: the magnitude shown is right, but some values keep only one significant figure.

## Checks run

| Check | Result |
|---|---|
| `check.py`: 200 figures. 10 picker models × 3 elements (cal/calw/log) × censor on/off × log/lin axis × horizon 10/30, resqr. For every trace whose hovertemplate reads `customdata[n]`, each row was checked: length > max n; row[0] is a date (`_add_date_hover` prepend happened); indices with `:.0f` are numbers; indices without a format are `$…` strings that match `^\$[\d.,]+[kMBT]?$`. Forecast turns: "or later" appears exactly when the bar reaches the cap, and no bar runs past `t_last + _TIMING_CAP`. | **0 mismatches** of any kind. No `$nan`/`$inf`. Three `$0.00` strings (LPPL/log only, N-a). |
| `grid_and_ctx.py`, part 1: `_grid_best(chunk=50)` vs one pass (chunk=10⁶) on the real percentile series. 10 models × 3 elements × censor × {daily, 400 nodes; weekly, 200 nodes}. | **120/120 same index** |
| `grid_and_ctx.py`, part 2: `update_bub_pctile` driven through a real Dash 4.0.0 request (`app.server.test_client()` POST `/_dash-update-component`, real `ctx`), 7 `changedPropIds` cases. | All correct. Fit off with a sub-control only (one or two of them) gives `"response":{}`, i.e. no update; Dash 4 returns 200, not 204. Sub-control + `bub-view-mode` (either order), fit on, the `bub-pctile-fit` toggle in the batch, and an empty trigger list each return a figure. |
| `lowclamp.py`: 60 model × element × censor × horizon builds, every forecast turn. | 739 rows. **0** hit the lower clamp and **0** hit the upper ("or later"). Both clamps are purely protective on today's data. |
| `phone.py`: kaleido at 380 × 420 px. | The default subtitle fits. The longest one clips its last character (N-b). |
| JS consumers of the cycle customdata | None (`grep customdata btc_web/assets/*.js`), so no client code depends on the old numeric columns. |

## Whole-diff finding → resolved?

| Finding | Resolved? | Evidence |
|---|---|---|
| **I-1**: hover prices in SI "milli" ($0.36 → "$362m") | **yes** | No `.3s` is left in any template. The wave rows are `[date, flo, fhi, llo, lhi, $fit, $lo, $hi]`, and the template reads `customdata[5..7]` unformatted and `[1..4]` with `:.0f`. Fitted turns are `[date, kind, $]` and forecast turns `[date, kind, $, from, to]`; both match their templates. 200-figure sweep: 0 misalignments. The default 2010 row shows `$0.0x`/`$0.3x`, not `m`. Hover and labels now share one formatter. |
| **M-1**: unhashable basis → 500 | **yes** | `isinstance(fit_basis, str)` precedes the dict lookup. The model check (list membership) and horizon check (tuple membership) compare with `==`, so they cannot raise. A test covers list, dict and int. |
| **M-2**: timing bound past pandas 2262 | **yes** | `b` is clamped to `t_last + 100` (year ~2126) and labelled "or later". `a` is clamped below to the first close. `a` cannot overflow upward: `turn_timing` already returns `min(a, t_e) ≤ t_e ≤ t_edge`. Covered by the monkeypatched ±900 yr test. Never triggered on real data (0/739). |
| **M-3**: optimistic-check wording | **yes** | The box text branches on `honest` ("…even against today's fan"). The sentence now reads "an optimistic check: the gap overstates the wave's edge". The QR fallback without the grid reads "the Time Machine as-of fans are unavailable here", and its legend still says "(not checked out-of-sample)", which is accurate. |
| **M-4**: at-limit legend shows the grid-edge period | **yes** | HybPPL cal today: `at_limit=True`, R² 0.151. The legend reads "HybPPL cycle fit (period at search limit)". `at_limit` implies weak, so no forecast legend needs the same change. |
| **M-5**: censored honest hindcast says "no look-ahead" | **yes** | When `censor` is on, "(Each fit withheld 2019–20, a window picked after seeing it misfit, so this flatters the wave a little.)" is appended. |
| **M-6**: 76 MB transient scan | **yes** | Chunks of 50. Taking the first chunk with strictly smaller SSE keeps `argmin`'s first-minimum tie rule. A NaN chunk is now skipped instead of winning, which is an improvement. `_grid_best` has a single caller. 120/120 real-series indices match. |
| **N-1**: stale docstring / misplaced constant | **yes** | Today's calendar R²: BM 0.044, HybPPL 0.151, EPPL 0.126, PCA 0.072, Greedy 0.078, EF 0.066, all below 0.20. LPPL is 0.509 and forecasts. The docstring matches, and `DEFAULT_HORIZON` now sits beside `HORIZONS`. |
| **N-2**: early return checks only the first trigger | **yes** | `set(ctx.triggered_prop_ids.values()) <= _PCTILE_SUB_CONTROLS`. Behaviour was confirmed in a real Dash context (table above). A pattern-matching id (a dict value) would make `set()` raise, but the `except` turns that into a redraw, which is the safe default. |
| **N-3**: noise-level toggles | n/a | Kept by design, as the commit message states. |
| **N-4**: subtitle overflows on phones | **partly** | The default "QR · calendar 2–8 yr · …" (59 chars) fits at 380 px. The longest, HybPPL · calendar 0.75–16 yr (67 chars), still clips "forecas". See N-b. |

## New findings

### Minor

**M-new-1: `_money()` keeps one significant figure for $1–$9.99 and $1k–$9.99k, and the fold extends this from labels to every hover price.**
- `_money` formats `$v:,.0f` for v ≥ 1, and `$v/1e3:,.0f`k for v ≥ 1e3. Before the fold, hover used 3 significant figures (`.3s`). Labels have always used `_money`.
- Measured on the default view: **815 of 5,712** hover price strings (14 %) carry one significant figure (`prec.py`).
- Rounding examples: $2.53 → "$3", $1.49 → "$1", $1,499 → "$1k", $1,500 → "$2k", $9.6 → "$10". Rollover: $999,600 → "$1,000k" (should be "$1.00M"). Below $0.01: $0.004 → "$0.00".
- Not Important. The order of magnitude is always right and the prices are prefixed "≈", so nobody is misled about scale. This is coarseness, unlike I-1's wrong unit.
- **Fix:** use three significant figures within each suffix and carry the rollover to the next suffix. Pin it with a test: 2.53 → "$2.53", 1,499 → "$1.50k", 999,600 → "$1.00M", 0.004 → "$0.004". The label regex `_PRICE_LABEL` still matches.

### Nit

- **N-a:** LPPL with the log element shows `≈ $0.00–$0.06` on 3 early-2010 rows, because a sub-cent Q0 likely-low rounds to zero. Non-default combination; the M-new-1 fix also fixes this.
- **N-b:** The longest subtitle (HybPPL + calendar 0.75–16 yr) still clips one character at 380 px. The default fits. Fix: shorten the element label in the subtitle (e.g. "cal 0.75–16 yr") or drop "bands: fit / likely".
- **N-c:** The lower timing clamp (first close) has no "or earlier" marker, unlike the upper clamp's "or later". It never fires today (0/739 rows), and a pre-2010 bound for a future turn would be self-evidently degenerate.
- **N-d:** `test_hover_prices_are_preformatted_not_si` would still pass if the date prepend were lost, because a 7-element row's `[5:8]` slice also yields `$` strings. Add `assert len(early) == 8` and a date match on `early[0]`. The sweep shows the prepend does happen today.

## Not re-checked
- The full pytest suite, Py 3.12 compile and the live Firefox journeys. The controller already ran these (brief), and nothing here contradicts them.
- Prod RAM and timing.
