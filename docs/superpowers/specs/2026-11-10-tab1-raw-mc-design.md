# Tab 1 "Raw" Monte Carlo — design

Status: revision 6, 2026-11-10 (folds reviews r2–r5). Rewritten after spec review round 1
(`docs/superpowers/reviews/2026-11-10-tab1-raw-mc-spec-review.md`, 0C/5I/8M/2N)
and a second brainstorm with the operator that added the trend-adjusted
mechanism. Everything here is operator-approved; items marked
**(controller's call)** were proposed by the controller and accepted by the
operator as a batch ("I agree with your recommendations / guesses"), so a
reviewer may still challenge them on soundness.

## Why

Every MC option today simulates transitions between percentile bands of a
chosen price model. The operator wants an option built from Bitcoin's own
price history — its swings, momentum and halving-cycle rhythm — whose first
question from a viewer is "does it reflect Bitcoin's momentum and 4-year
cycle?".

Pure replay of history cannot be the whole answer: any resample of a window
inherits that window's average growth and compounds it (2015–now ≈ +59 %/yr →
median ≈ $10¹³ after 40 years, far above the chart's $1 B y cap). So Raw
extrapolates **the per-cycle histogram of monthly returns** — its mean and its
standard deviation follow curves fitted across halving cycles — and each path
is a bootstrap walk through those extrapolated histograms. A **Trend**
selector lets a sceptic pick the curve shape, or none (honest replay).

### Measured (`tools/raw_mc_cycle_stats.py`, data to 2026-11-08)

Per halving cycle, using this spec's definitions (Data and Trend curves
below; t = mean t of the cycle's data months, years since 2009-07-25;
growth = 12 × mean monthly log return, shown as %/yr; "g × t" in natural-log
annual units; vol = √12 × std of monthly log returns):

| Cycle | data months | t | growth/yr | g × t | vol |
|---|---|---|---|---|---|
| 1: 2010-08 → 2012-10 | 27 | 2.14 | +895 % | 4.92 | 160 % |
| 2: 2012-11 → 2016-06 | 44 | 5.10 | +205 % | 5.69 | 127 % |
| 3: 2016-07 → 2020-04 | 46 | 8.85 | +98 % | 6.02 | 82 % |
| 4: 2020-05 → 2024-03 | 47 | 12.72 | +70 % | 6.74 | 69 % |
| 5: 2024-04 → 2026-10 (partial) | 31 | 15.97 | +4 % | 0.57 | 44 % |

**Default fit (all history, 4 full cycles):** power law n = 5.12 (rms
0.110), vol = 2.46 · t^−0.49; exponential growth exp(1.18 − 0.179 t)
(rms 0.142), vol exp(0.66 − 0.087 t). Implied growth / vol: 2030 power
+28 %/yr, 56 % · exp +8 %/yr, 31 %; 2050 power +13 %/yr, 40 % · exp +0 %/yr,
6 %. Window from 2013 (3 full cycles): n = 5.93, vol ∝ t^−0.72. These are
pinned in tests from the script's functions.

- n ≈ 5–6 is the power-law slope, found without fitting a price curve.
- Shape: each cycle's histogram, standardised by its own mean and std, is
  statistically indistinguishable from the others (KS p = 0.70–0.89, ad
  hoc), so one pooled shape can be reused.
- Momentum: monthly autocorrelation +0.24 (since 2010) / +0.11 (since 2015);
  variance ratio 1.4–1.56 at 6–12 months; vol clustering +0.46 / +0.38.

## Goals

1. Two new entries in Tab 1's **Model source** dropdown
   (`bub-mc-model-src`): **Raw · independent** (`raw_iid`) and **Raw ·
   halving cycle** (`raw_cycle`). No other tab gets them.
2. **Free for everyone**, computed live per request; no cache build, no
   payment prompt, whatever the Raw settings.
3. A **Trend** selector: **Power law** (default), **Exponential decay**,
   **None (honest replay)**.
4. **Raw has its own controls** (sims, years, window, bins, regime
   checklist, trend), styled and placed like the model ones, so Raw settings
   never alter model mode (review I-1).
5. Example lines: **Lines shown** (5–200, default 20) spread across end
   prices, plus **Show extremes** (highest and lowest-ending paths).
6. Regime filter on the path's own price level relative to the trend (review
   I-2).

## Non-goals

- Any tab other than Tab 1.
- Changing model-based MC on Tab 1 (behaviour, look, controls, defaults, the
  "best 50 of 200" subset).
- Skipping a middle bubble in the window.
- Daily steps, GARCH, smoothed (kernel) sampling of the histogram.
- A paid tier or precomputed cache for Raw.

## Anti-goals

- A fan whose growth assumption is not stated on screen.
- Example lines that hide the bands.
- Restyling the approved MC look (amber fan, dark-neutral lines,
  `MC_PATH_ALPHA_BUDGET / n`), `_MC_BANDS` / `_GHOST_BANDS`, or any CB
  palette. The operator is deuteranomalous: never distinguish lines by
  red/green hue.
- Any Raw state reaching BTCPay, the free-tier check, the MC cache, the
  payment modal or the premium/match chrome.

## The simulation

### Data

Month-end closes from the full price CSV → monthly log returns `r`. The
**incomplete trailing month is dropped** (review M-1). Each month carries:

- `t`: years since 2009-07-25 (the project's time origin) at the month's
  midpoint;
- **cycle**: the halving interval its month-end falls in (halvings:
  `figures/common.py::_HALVING_KNOWN_DATES`; the genesis block 2009-01-03 is
  epoch 0, so months before 2012-11-28 belong to cycle 1). A month belongs
  to the cycle containing its **month-end date**, so the month containing a
  halving belongs to the new cycle only if the halving precedes its month-end
  (review r2 M-5; `tools/raw_mc_cycle_stats.py` is the reference);
- **cycle fraction** `φ ≈ [0, 1)` (a month can read −0.01 when its month-end and midpoint straddle a halving; clip to [0, 1) — review r4/r5 N-1): days elapsed since the cycle's start
  epoch at the month's midpoint ÷ that cycle's length in days (future
  cycles: `_HALVING_EPOCHS`) (review r2 I-A);
- its cycle's sample mean `μ_c` and std `σ_c` (monthly), and its **shape**
  `z = (r − μ_c) / σ_c`.

### Trend curves (fitted once per request; cheap)

A cycle's months are its **in-window months with price data** (review r3
M-2). A cycle is **full** when it ends at a known halving and has **≥ 24**
such months (cycle 1 has 27, so all four completed cycles are full at the
default window). A full cycle is also **complete** when those months span
cycle fraction φ < 0.05 to φ > 0.95; only complete cycles feed cycle-mode
blocks (cycles 2–4 today; cycle 1's data cover only φ 0.41–0.97 — review r3
I-1). The current cycle is never full. **Fit set**: the full cycles in the
window if there are ≥ 3, otherwise every full cycle of the whole history.
The status line says which. (Review r2 I-D.) The reference implementation
of all of this, including the fit-set rule, is
`tools/raw_mc_cycle_stats.py` (`cycle_table`, `fit_set`, `fits`); tests pin
its outputs.

Per full cycle c, in **annual natural-log units** (review r2 M-1):
`t_c` = mean t of its months, `g_c = 12 · μ_c`, `s_c = √12 · σ_c`. The
simulation uses monthly `μ(t) = g(t) / 12` and `σ(t) = s(t) / √12`.

| Trend | growth g(t) | volatility s(t) |
|---|---|---|
| Power law | `n / t`, unweighted LS: `n = Σ(g_c/t_c) / Σ(1/t_c²)` | `a · t^−b` (linear LS of `log s_c` on `log t_c`) |
| Exponential decay | `exp(α + β t)`, nonlinear least squares directly on `g_c` (`scipy.optimize.curve_fit`) | `exp(γ + δ t)`, same method on `s_c` |
| None | — | — |

Exponential curves are fitted **directly in growth space, never via
`log g_c`**, so a full cycle with negative growth (possible once the
2024–2028 cycle completes) simply pulls the curve down instead of breaking
the fit (operator decision, 2026-11-10). No fallback between trends: if
`curve_fit` fails to converge (or its start guess is undefined), no fan is
drawn and the status line says "Exponential trend could not be fitted to
these cycles" (review r3 N-2). The start guess must not take the log of a
non-positive mean. `scipy` is already in `btc_web/requirements.txt`.

### Generating a path

Paths start **today** at the current price — the live ticker price when
available (`btc-price-store`), else the last daily close — at today's
fractional `t`, and step monthly for the Raw **Years** horizon (review M-3:
the Raw time axis starts today, not at 1 January).

For each simulated month at time `t`, with **Trend ≠ None**:

> `r = μ(t) + σ(t) · z`

`μ(t)` is evaluated at each month's own `t`. `σ` is **held constant within
each future cycle** at the trend's value at that cycle's midpoint `t`
(halving to halving; the current cycle uses its remaining span's
midpoint). Without this, vol decaying inside a cycle weights the bull
phase's large `z` over the bear phase's and biases cycle mode upward
(review r4 I-2).

where `z` is a **plain bootstrap** draw (uniform over eligible months, which
reproduces the histogram's frequencies) of a historical shape:

- **Raw · independent**: one month's `z`, independently each month.
- **Raw · halving cycle**: the future is walked in **3-month blocks**; each
  block copies three consecutive historical `z` values from a **complete
  cycle** whose starting cycle fraction φ is within **±0.04** (≈ 2 months)
  of the future block's own φ. φ is **linear, not circular**: a block near
  φ = 1 matches only late-cycle history (review r3 N-1). Matching by
  fraction of cycle elapsed and drawing only from complete cycles keeps the
  median on the trend (review r1 M-6, r2 I-A, r3 I-1). If those candidates
  come from **fewer than 2 distinct cycles**, the tolerance widens to ±0.08,
  then to all complete-cycle blocks; the status line reports how often
  (review r1 I-3). Cycle mode needs **≥ 2 complete cycles in the window**;
  otherwise no fan is drawn and the status line says so (review r3 M-3). The
  caveat says cycle mode replays only the complete past cycles in the window
  (review r2 M-4).
- **Block grid and debias (review r5 I-1).** The future block grid
  **restarts at each projected halving**: blocks run in 3s from the
  halving, and the last block before the next halving may be 1–2 months.
  The current cycle's grid runs from today to the next halving the same
  way. Because the ±0.04 phase windows weight historical blocks unevenly,
  the expected `z` over a whole future cycle is not exactly 0 (+0.41 per
  cycle at the fixed start below). So per request, cycle mode computes
  `E` = Σ over one whole cycle's grid blocks of the pool-mean block
  `z`-sum, and draws `z − E/48` (about 0.009 per month, imperceptible as
  shape). Trend = None stays raw.
- **Tests (review r3 I-2, r4 I-1, r5 M-1/M-2, N-4).** Every statistical
  Raw test runs on **frozen inputs**: price data truncated at a fixed start
  on a halving (2024-04-20 as "today", start price the 2024-04-20 close),
  injected, so no test depends on the real date or on growing data.
  - (a) Deterministic: after the debias, the expected `z`-sum over each
    whole future cycle's grid is **|E′| ≤ 1e-9**.
  - (b) Statistical: at the month ending on each subsequent halving
    **≤ the horizon** (2028, 2032, … 2064 for 40 years), the median log
    price of 3200 paths is within **±0.25 × the cross-sectional std of log
    price at that date** of the trend-median path, for both variants and
    both fitted trends (prototype worst 0.07 across 20 seeds).
  - (c) The generator's `σ` is identical across all months of each future
    cycle, steps only at projected halvings, equals the trend's value at
    that cycle's midpoint `t`, and the current cycle uses its remaining
    span's midpoint.
  - The regime-filter tests (below) use the same frozen inputs.
  - Mid-cycle horizons and the partial first cycle of a real "today" are
    not tested: sitting above or below trend there is the rhythm the
    variant exists to show.
- **Block integrity:** a block's three months come from **one** complete
  cycle (its `z` values share that cycle's standardisation); a block may
  not straddle a halving in the history it copies (review r4 M).

With **Trend = None** the same draws use raw `r` instead of `z` (honest
replay). The status line states **the simulated fan's own median pace**,
computed from the paths over the chosen horizon, beside the window's
actual pace, e.g. "Fan median grows +120 %/yr over 40 years (history
2010–2026 grew +137 %/yr); the fan compounds replayed history." The two
differ in cycle mode, because phase matching re-weights the months (review
r3 I-4; example figures from the r4 prototype, illustrative only).

**Eligible** months: inside the Raw window, and (cycle mode) all three
months of a block inside the window.

### Regime filter (Raw) — level relative to trend (operator decision, r2 I-C)

- **State** of a path at a month: how far its **price level** sits above or
  below the trend, in standard deviations (review r3 M-1 pins the formula):
  - historical: `D_k = log P_k − Σ_{j≤k} μ(t_j) − C`, summing from the
    first eligible month, with the constant `C` chosen so the **mean of
    historical `D` over eligible months is 0**;
  - simulated: `D_{k+1} = D_k + r_{k+1} − μ(t_{k+1})`, starting from
    `D_0` = the last month-end's historical `D` + log(P_today / P_last
    month-end) (the same anchored curve continued; review r4/r5 N-2);
  - state = `D / S`, `S` = std of historical `D` over eligible months.
  - With Trend = None, μ is the window's constant mean monthly log return.
- **Bins**: `bub-mc-raw-bins` equal-count bins of the historical state over
  eligible months, coldest first. The **outer bins are closed** at the
  historical minimum and maximum state (review r3 I-3): a draw that would
  take the state beyond them counts as entering an unchecked region and is
  steered like any other. Labels show each bin's range, e.g. "−1.5σ to
  −0.8σ vs trend".
- **Steering**: a draw (a month; in cycle mode the block is accepted month by
  month up to its first violating month and redrawn from there) that would
  put the state into an unchecked bin is rejected and redrawn, up to 20
  tries; then the **best** try (closest to a checked bin) is kept. Today's
  state may itself sit in an unchecked bin; paths are steered toward checked
  bins from there (review r2 I-B).
- A test pins, for every single-bin and two-bin selection in both variants
  at 5 bins, the share of simulated months in unchecked bins **≤ 5 %**
  excluding the first 12 months (prototype: ≤ 1.57 %, frozen inputs), and the 40-year
  median state within the checked bins' closed range ± 0.5 (bounded,
  review r2 I-C, r3 I-3), for all three trends.
- The note states the measured share: "Raw: paths are steered away from
  unchecked bins (N % of months still fall in them)."
- Same server rule as model mode: applies only when 0 < checked < bins.
- The auto-tick of "Show individual price paths" on a real filter click
  applies to `bub-mc-raw-regime` too (own click stamp, same pattern as
  `assets/mc_regime_paths.js`; one Input per clientside callback).

### Determinism

The RNG is seeded from a **stable digest** (`hashlib.blake2b` of canonical
JSON, never Python `hash()`, which is salted per worker — review M-2) of:
variant, trend, window, bins, checked bins, sims, years, today's date and the
price-data fingerprint. Lines shown, extremes, x-range and the paths switch
do **not** enter the seed, so toggling them never redraws the fan. The fan
re-draws daily and scales with the start price; a share link reproduces the
same fan on the same day.

Performance target: 3200 paths × 480 months, with the filter on, in well
under 200 ms on prod (vectorised across paths; rejection loops only over the
rejected subset).

## Display

### Fan and lines

- Fan: `_add_mc_fan` over all paths, legend suffix naming variant, trend and
  window, e.g. "MC median · raw cycle, power law, 2010–2026".
- **Lines shown** (`bub-mc-lines`, number 5–200, step 1, default 20): paths
  ranked by price at the last plotted step (the visible right edge — so the
  sample changes when the x-range is panned, by design; review N-2), taken at
  evenly spaced ranks from the 2nd to the 98th percentile by a **new
  rank selector** (not `_add_mc_spaghetti`'s stride subsampling, which is
  bypassed by passing exactly the chosen paths; review r2 N-1). If lines ≥
  sims, all are drawn.
- **Show extremes** (`bub-mc-extremes`, checklist `["yes"]`, default off):
  the top- and bottom-ranked paths, dashed and slightly bolder in the same
  dark neutral (new constants in `colors.py`), legend "Highest end" /
  "Lowest end", no hover.
- Both sit directly under **Show individual price paths** (`bub-mc-paths`,
  shared, still the master switch, default off) and only show in Raw mode.
- Lines keep today's payload treatment (`_add_mc_spaghetti`: every other
  step, 2 significant figures), trimmed at the visible right edge. Measured
  gzip: 200 lines × 12 visible years ≈ 23 KB; 202 lines × 40 visible years
  ≈ 68 KB; 20 lines × 40 years ≈ 8 KB (review r2 M-2). Accepted; the default
  (20 lines, auto-ranged view) is ≈ 3 KB.

### Raw controls (shown only in Raw mode; model controls hidden)

| Control | id | Default |
|---|---|---|
| Simulations (1–3200, free) | `bub-mc-raw-sims` | 1000 **(controller's call)** |
| Years (10 / 20 / 30 / 40; no "cached" bolding — review r2 M-6) | `bub-mc-raw-years` | 40 |
| History window | `bub-mc-raw-window` | all history (2010 – current year) |
| Trend | `bub-mc-raw-trend` | `power` (`power` / `exp` / `none`) |
| Trend-deviation bins | `bub-mc-raw-bins` | 5 |
| Trend-deviation regime checklist | `bub-mc-raw-regime` | all checked |
| Lines shown | `bub-mc-lines` | 20 |
| Show extremes | `bub-mc-extremes` | `[]` |

Labels are Raw-specific (review M-8): no "Markov transition matrix", no
"≤ 200 is free". Start year and entry percentile are hidden in Raw with the
hint "Raw starts today at the current price". Hidden vs disabled and exact
layout are the builder's call; positions mirror the model controls.

### Raw status line

A new `bub-mc-raw-status` div directly under the model-source dropdown, Raw
only. It always states the trend used and its fitted numbers and which
cycles were fitted (e.g. "Power law n = 5.12, vol ∝ t^−0.49, fitted on 4
cycles"), the current cycle's pace against the curve, the independent-mode
caveat ("No memory: understates multi-month swings"), the cycle-mode caveat,
cycle fallbacks, the filter's measured share, a warning when sims < 100
("percentiles are rough below 100 simulations"), and any reason no fan is
drawn (fewer than **36** eligible months — review r2 M-3). `bub-mc-status`
is not used for Raw.

### Paid and premium chrome in Raw

Hidden: Run MC Simulation button, cost panel, "Paid" badge, Saved
Simulation card (save / download / upload / restore). The match indicator
must show **no premium class, no badge, no restore button and no mismatch
warning** beside a Raw fan (review M-4). Tests pin that no Raw state
(including a Run click from a stale client) reaches `btcpay.create_invoice`
or opens `bub-mc-save-modal`.

### Auto-range

The real-click MC auto-range uses the current year as the MC start in Raw
mode.

## Share links

- `raw_iid` / `raw_cycle` are new values of `bub-mc-model-src`.
- Append at the absolute tail of `_SNAPSHOT_CONTROLS`, in this order:
  `bub-mc-raw-sims`, `bub-mc-raw-years`, `bub-mc-raw-window`,
  `bub-mc-raw-trend`, `bub-mc-raw-bins`, `bub-mc-raw-regime`,
  `bub-mc-lines`, `bub-mc-extremes` (indices 340–347). Add defaults; treat
  `bub-mc-raw-regime` / `bub-mc-extremes` like `bub-mc-regime` /
  `bub-mc-paths` in `_CHECKLIST_OPTIONS`; treat `bub-mc-raw-window` like
  `bub-mc-window` for its current-year default; add all to
  `_TAB_CONTROLS["bubble"]`; register old and new defaults fingerprints
  (`tools/update_defaults_registry.py`).
- Old links decode with the defaults; model-mode links are unaffected.

## Constraints

- TDD; full non-E2E suite green; Firefox E2E for real-click behaviour
  (regime auto-tick in Raw, auto-range in Raw).
- New update_bubble Inputs are appended **last** (stale-client rule, as for
  `bub-mc-paths`) and added to `_POST_RESTORE_TRIGGERS`; a new
  `bub-mc-raw-status` Output grows the return tuple, and every
  `(dash.no_update,) * N` must grow with it (review M-5).
- Clientside callbacks: one Input when `allow_duplicate` +
  `prevent_initial_call` (Dash 4.0 drops multi-Input ones).
- Colours and widths in `colors.py`; `_SNAPSHOT_CONTROLS` append-only;
  runtime imports in `btc_web/requirements.txt` (numpy/pandas only
  expected).
- MC traces are added on a clone after the cached figure; Raw params should
  not enter the bubble cache key (`test_cache_key_alignment.py`).

## Left to the builder

Module layout (a numpy-only `btc_web/raw_mc.py` is the obvious home); hidden
vs disabled for model-only controls; exact status, caveat and label wording;
extremes' width/opacity; test structure.
