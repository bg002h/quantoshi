# Tab-1 Raw MC spec — adversarial review, round 2 (2026-11-10)

**Spec:** `docs/superpowers/specs/2026-11-10-tab1-raw-mc-design.md` at a14cf974 (revision 2)
**Counts: 0 Critical · 4 Important · 7 Minor · 2 Nit**

Read-only. Code checked: `callbacks/charts/__init__.py` (update_bubble), `layout/mc_controls.py`,
`figures/bubble.py`, `figures/common.py`, `snapshot.py`, `snapshot_defaults.py`, `tab_defaults.py`,
`test_cache_key_alignment.py`. Statistics come from scripts in
`/tmp/claude-1000/-scratch-code-bitcoinprojections/3a79ac35-3364-4e4c-9013-baf596cc6ebf/scratchpad/r2/`
(`base.py`, `fit.py`, `sim.py`, `bias.py`, `bias2.py`, `regime.py`, `payload.py`). They use
`BitcoinPricesDaily.csv` with the last close on 2026-11-08, month-end closes, the incomplete Nov-2026
month dropped, 195 months from 2010-08, cycles split at the halving dates (genesis epoch = cycle 1),
start price $74,406 at t = 17.30.

## Round-1 findings

| R1 | Finding | Status in rev 2 |
|---|---|---|
| I-1 | Raw sims/bins leak into model mode | **Resolved.** Raw has its own `bub-mc-raw-*` controls. |
| I-2 | Regime filter does the opposite of its labels; mostly noise | **Resolved in direction**: it now filters on the path's own state. The new mechanism has two defects of its own (I-B, I-C below). The R1 request to raise the 12-month floor was not taken (see M-3). |
| I-3 | Thin-phase fallback counted overlapping starts | **Resolved.** The rule is now ≥ 2 distinct cycles. Filtering is now rejection after the draw, so bins no longer thin the candidate pool. On the real data every block start has 4–5 distinct cycles at the default window, so the fallback works and rarely fires. The caveat about replaying ~3–4 cycles per phase is still missing (M-4). |
| I-4 | Window nudge does not stop absurd fans; growth not stated | **Resolved.** The trend curves bound the median, and Trend = None states the pace. For the cycle variant, see I-A. |
| I-5 | Click-stamp cannot see a Dash 4 Dropdown selection | **Resolved by removal.** With separate controls there is no switch-time nudge or reset, so no dropdown stamp is needed. The remaining stamps are on checklists (regime auto-tick) and on `bub-mc-enable`, which work today. |
| M-1 | Partial trailing month | Resolved |
| M-2 | Seed stability | Resolved (blake2b; daily re-draw stated) |
| M-3 | Time axis starts 1 Jan | Resolved |
| M-4 | Premium chrome via match indicator | Resolved |
| M-5 | Callback wiring not stated | Resolved (Inputs last, `_POST_RESTORE_TRIGGERS`, tuple growth). Snapshot indices 340–347 are correct: the tail is `bub-mc-paths` at 339. |
| M-6 | Block phase alignment | Resolved (±2 months) |
| M-7 | Window nudge persists into model mode | Resolved by removal |
| M-8 | Model-mode labels in Raw | Resolved |
| N-1 | Stamp pattern naming | Moot |
| N-2 | Lines change on pan | Resolved |

---

## Important

### I-A. Raw · halving cycle does not follow the fitted trend: its median runs ×1.7 above it at 10 years and ×3–5 at 40 years

**Where:** "Generating a path", Raw · halving cycle. The Why section ("its mean and its standard
deviation follow curves fitted across halving cycles"). The status line ("Power law n = …").

**Failure:** `r = μ(t) + σ(t)·z` keeps the fitted median only if the drawn `z` averages zero over
time. That holds in iid mode: the median matches the curve to 3 % (`sim.py`). In cycle mode the
phase-matched blocks do not average to zero. The expected sum of `z` over one simulated 48-month
future cycle is **+1.30** (`bias.py`). The cause has three parts:

- Future cycles are 48 months long (`_HALVING_EPOCHS` = anchor + 4 y). Historical cycles were
  43–47 months.
- A block that starts at months-since-halving 45–47 runs into the next cycle's post-halving bull
  months. Those months are then counted a second time at the start of the simulated cycle.
- The partial cycles (genesis epoch, which only has data for months 19–46, and the current cycle,
  months 1–30) are standardised by their own partial means.

Terminal p5 / p50 / p95 with Power law, compared with the curve-only median:

| Fit set | Variant | 10 y | 40 y | curve median 10 y / 40 y |
|---|---|---|---|---|
| cycles 2–4 | iid | 0.10 M / **1.03 M** / 11.8 M | 2.6 M / **75 M** / 2.9 B | 1.06 M / 80 M |
| cycles 2–4 | cycle | 0.19 M / **1.78 M** / 15.5 M | 8.4 M / **267 M** / 10.6 B | 1.06 M / 80 M |
| all 4 | cycle | 0.10 M / **1.45 M** / 19 M | 1.9 M / **175 M** / 21 B | 0.78 M / 35 M |
| cycles 2–4, exp | cycle | 0.13 M / 0.63 M / 2.8 M | 0.33 M / 1.87 M / 11 M | 0.45 M / 1.09 M |

The status line would say "Power law n = 5.8" while the fan's median grows markedly faster. That is
a defect in what the tool claims. A sceptic who compares the two variants will see that "cycle"
adds growth, not just rhythm.

**Fix:** Define the phase as the **fraction of the cycle elapsed** (months since halving ÷ that
cycle's length; future cycles 48 months) and draw blocks only from **full** cycles. In `bias2.py`
this brings the expected z-sum per future cycle to **−0.02**, or +0.15 if blocks must also stay
inside one cycle. A blunter alternative is to subtract the cycle-average of the phase-conditional
mean of `z`, which keeps the rhythm and removes the drift. Pin a test that the cycle-mode median at
10 and 40 years is within ±25 % of the curve-only median, for both trends.

### I-B. The Raw regime note promises "never", but forced draws put 4–13 % (iid) and 31–64 % (cycle) of months in unchecked bins

**Where:** Regime filter (Raw): "Unchecked bins are never visited … up to 20 tries; after that the
last draw is kept". The regime note "Raw: paths never enter unchecked momentum bins." Forced-draw
reporting above 1 % (controller's call).

**Failure:** Today's state is **−1.27 σ**, which is in the coldest of the 5 historical equal-count
bins (edges −2.50 / −1.14 / −0.35 / +0.08 / +0.66 / +2.30 σ). Measured with the cycles 2–4 power
fit and 2000 paths × 480 months (`regime.py`):

| Checked bins | iid: months in unchecked bins | cycle: months in unchecked bins |
|---|---|---|
| {0} | 12.6 % | 64.0 % |
| {1} | 4.3 % | 30.7 % |
| {2} | 6.3 % | 63.2 % |
| {4} | 8.4 % | 37.8 % |
| {0,1} / {3,4} | 2.8 % / 2.9 % | 19.2 % / 17.5 % |
| {1,2,3} | 0.7 % | 3.4 % |

The state moves only by one month's replacement in a 12-month sum, so a path that starts in an
excluded bin cannot reach a checked bin in one draw. The first months are then forced whatever is
drawn. In cycle mode a whole 3-month block must keep **all three** months in checked bins, which
fails most of the time. The "keep the last draw" rule then hands the path a random draw, which is
the unfiltered behaviour. For cycle mode with one bin checked, the filter is mostly not applied
while the note says "never". This is "a refusal that does not refuse".

**Fix:**
- Keep the **best** of the 20 tries (the one closest to a checked bin), not the last.
- In cycle mode, accept a block up to its first violating month and redraw from there.
- Word the note from the measured share ("paths are steered away from unchecked bins; N % of
  months still fall in them"), or pin a ceiling (e.g. ≤ 2 % occupancy) as a test across single-bin
  selections in both variants.
- State the start-state rule: today's state may sit in an unchecked bin.

### I-C. A momentum-state filter compounds without bound, so filtered fans leave every plausible range while the status line still quotes the trend

**Where:** Regime filter, state definition: "trailing 12-month log return minus the trend's
expected 12-month return, divided by the trend's 12-month std" (controller's call). Anti-goal "A fan
whose growth assumption is not stated on screen".

**Failure:** The state is a *rate* of deviation, not a *level*. Holding a path in a bin holds its
yearly excess return in that band for the whole horizon, so the deviation from trend grows linearly
in log price. iid, 40-year medians (`regime.py`):

| Checked | Median at 40 y |
|---|---|
| coldest only | **$0.44** |
| bin 1 | $8.6 k |
| bin 3 | $7.1 B |
| hottest only | **$4.9·10¹⁴** |
| {0,1} | $190 |
| {3,4} | $5·10¹² |

In model mode the same checklist bounds the price to bands **around a model level**, so filtered
fans stay near the model. Users will read the Raw labels the same way. The status line meanwhile
says "Power law n = 5.8, vol ∝ t^−0.48", which describes none of these fans.

**Fix (pick one):**
- (a) Define the state as the path's **level** deviation from the trend curve: cumulative log
  residual over trend ÷ its cumulative std. Bin it from history the same way. This is bounded,
  mirrors model-mode semantics, and still means "how far above or below normal for its era".
- (b) Keep momentum, but whenever the filter is active the status line must state the filtered
  fan's realised median pace (e.g. "Filtered fan grows −14 %/yr vs trend +16 %/yr"). Pin this with
  a test.

### I-D. Which cycles are fitted is under-determined, and the spec's quoted numbers match neither reading

**Where:** Trend curves: "at least 3/4 of its months are inside the Raw window"; "otherwise on all
four full cycles"; "least squares on `g_c` vs `t_c`"; the status example "Power law n = 5.8, vol ∝
t^−0.48, fitted on 4 cycles"; the Measured table.

**Failure:** Price data starts 2010-07, so the genesis cycle has **28 of its ~47 calendar months**
(60 %). The two readings give different defaults:

- **Calendar months:** cycle 1 is never "in the window", even at the default all-history window.
  The default fit uses cycles 2–4: **n = 5.82, vol exponent b = 0.70**.
- **Data months:** cycle 1 counts, and the fit uses 4 cycles: **n = 5.14** (unweighted LS of
  annual log growth on n/t, rms 0.09), **b = 0.48**.

The spec's example pairs n = 5.8 (the 3-cycle value) with b = 0.48 (the 4-cycle value) and says "4
cycles". No implementation can produce it. The two readings differ materially:

| Reading | 40 y median | 40 y p5 | vol in 2050 |
|---|---|---|---|
| calendar (cycles 2–4) | $80 M | $2.6 M | 30 % |
| data (all 4) | $35 M | $0.39 M | 41 % |

"t = cycle midpoint" is also ambiguous for cycle 1. The calendar midpoint is t ≈ 1.4; the mean of
its data months is 2.18. The table does not reproduce exactly either: cycle 1 growth +868 %/yr
here vs +924 %; g·t for cycle 4 6.23 vs 6.5; partial cycle 1.36 vs 0.9; power rms 0.09 vs 0.18.
A test author cannot pin the default fit.

**Fix:**
- Define "a cycle's months" as its months **with price data**.
- Define t_c as the mean t of those months.
- State the fitting objective explicitly (unweighted LS of g on n/t, or n = mean(g·t) = 5.73).
- State whether the default fit is 3 or 4 cycles.
- Regenerate the table and the status example with a committed script, and pin the default
  numbers in a test.

---

## Minor

- **M-1. Units of n** (Trend curves: "`g_c = μ_c` (mean monthly log return)", `n / t`). Fitted on
  monthly μ, n ≈ 0.43–0.49. The status line and table quote the annual value 5–6. **Fix:** state
  μ(t) = n / (12 t) per month, n in annual log units, and the same for σ (annual a·t^−b ÷ √12).
- **M-2. Lines payload** ("≤ 202 lines ≈ ≤ 25 KB gzip"). Measured with the existing
  `_add_mc_spaghetti` treatment (every other step, 2 s.f., 480 months): **20 lines = 8 KB, 202
  lines = 68 KB** gzip, about 2.7× the claim. That matters under the ALARA preference. **Fix:**
  correct the figure, or cap Lines shown lower (e.g. 100 ≈ 34 KB), or stride further for large
  counts.
- **M-3. 12-month floor kept** (status line, "fewer than 12 eligible months", controller's call).
  R1 asked to raise it. With Trend = None, a 12-month pool resampled for 40 years is noise. With a
  trend, 12 months is the whole shape pool, and in cycle mode the phase matching is empty. R1's
  "warn when paths < ~100" for the 1–3200 sims input is also absent. **Fix:** set a floor of 36
  months (or one full cycle) and add the sims < 100 warning.
- **M-4. Cycle-mode caveat missing.** R1 I-3 asked the spec to say that cycle mode replays only
  about 3–4 past cycles per phase (measured: 4–5 distinct cycles per block start at the default
  window, 3 when the partial cycles are excluded). **Fix:** add it to the variant's caveat.
- **M-5. Months-since-halving for the halving month and for cycle 1.** "0, 1, 2, …; from genesis
  for cycle 1". It is not stated whether the month containing the halving is msh 0 of the new cycle
  or the last month of the old one. Cycle 1's data begins at msh ~18. This interacts with I-A.
  **Fix:** define the phase by day count (or by fraction, per I-A) and say how the halving month is
  assigned.
- **M-6. Raw Years options unstated.** The model dropdown is 10/20/30/40 with "bold = cached"
  styling, which would be misleading in Raw. `_HALVING_EPOCHS` stops at 2080; a horizon past ~54
  years would run months-since-halving past any history and force the whole-pool fallback.
  **Fix:** list the Raw Years options (≤ 40, or cap at 2080) and drop the cached bolding.
- **M-7. Historical bin statistic near the window start.** The 12-month trailing state for the
  first 11 eligible months needs pre-window returns. It is not stated whether those are used or
  whether the months are dropped. For short windows this changes the bin edges. **Fix:** state it
  (recommended: use pre-window data for the trailing sum, so eligibility is unchanged).

## Nit

- **N-1.** "Lines shown … evenly spaced ranks from the 2nd to the 98th percentile" needs a new
  selector. Today's `_add_mc_spaghetti` subsamples by stride. Say so, so the builder does not reuse
  stride sampling.
- **N-2.** The Measured table's "growth × t (log)" column is natural-log annual growth × t. Label
  the units.
