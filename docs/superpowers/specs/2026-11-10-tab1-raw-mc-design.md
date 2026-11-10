# Tab 1 "Raw" Monte Carlo — design

Status: revision 2, 2026-11-10. Rewritten after spec review round 1
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

### Measured (ad hoc, `BitcoinPricesDaily.csv`, 2026-11-10)

Per halving cycle (t = cycle midpoint, years since 2009-07-25; monthly log
returns; vol annualised):

| Cycle | t | growth/yr | growth × t (log) | monthly vol |
|---|---|---|---|---|
| 2010-07 → 2012-11 | 2.2 | +924 % | 5.0 | 157 % |
| 2012-11 → 2016-07 | 5.2 | +201 % | 5.7 | 127 % |
| 2016-07 → 2020-05 | 8.9 | +97 % | 6.0 | 82 % |
| 2020-05 → 2024-04 | 12.8 | +66 % | 6.5 | 69 % |
| 2024-04 → now (partial) | 16.0 | +6 % | 0.9 | 43 % |

- Growth: `g = n/t` fits the four full cycles best (rms 0.18 vs exponential
  0.21, linear 0.31); n ≈ 6 — the power-law slope, found without fitting
  price.
- Vol: power law `σ ∝ t^−0.48` (rms 0.08) or exponential (rms 0.06); they
  diverge far out (2050: 41 % vs 7 %).
- Shape: each cycle's histogram, standardised by its own mean and std, is
  statistically indistinguishable from the others (KS p = 0.70–0.89), so one
  pooled shape can be reused.
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
6. Regime filter on the path's own momentum relative to the trend (review
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
- **cycle**: the halving interval it falls in (halvings:
  `figures/common.py::_HALVING_KNOWN_DATES`; the genesis block 2009-01-03 is
  epoch 0, so months before 2012-11-28 belong to cycle 1);
- **months since halving** (0, 1, 2, …; from genesis for cycle 1);
- its cycle's sample mean `μ_c` and std `σ_c`, and its **shape**
  `z = (r − μ_c) / σ_c`. The partial current cycle uses its own sample mean
  and std.

### Trend curves (fitted once per request; cheap)

A cycle is **in the window** when at least 3/4 of its months are inside the
Raw window **(controller's call)**. The current cycle never counts as full.
If the window holds **≥ 3 full cycles**, the curves are fitted on those;
otherwise on all four full cycles. The status line says which.

Per full cycle c: `g_c = μ_c` (mean monthly log return), `s_c = σ_c`.

| Trend | growth μ(t) | volatility σ(t) |
|---|---|---|
| Power law | `n / t` (one parameter, least squares on `g_c` vs `t_c`) | `a · t^−b` (linear fit of `log s_c` on `log t_c`) |
| Exponential decay | `exp(α + β t)` (linear fit of `log g_c` on `t_c`) | `exp(γ + δ t)` |
| None | — | — |

(Fitting form for exponential growth requires `g_c > 0` for every fitted
cycle; true for all four today. If a future data refresh breaks that, the
exponential option falls back to power law with a status-line note
**(controller's call)**.)

### Generating a path

Paths start **today** at the current price — the live ticker price when
available (`btc-price-store`), else the last daily close — at today's
fractional `t`, and step monthly for the Raw **Years** horizon (review M-3:
the Raw time axis starts today, not at 1 January).

For each simulated month at time `t`, with **Trend ≠ None**:

> `r = μ(t) + σ(t) · z`

where `z` is a **plain bootstrap** draw (uniform over eligible months, which
reproduces the histogram's frequencies) of a historical shape:

- **Raw · independent**: one month's `z`, independently each month.
- **Raw · halving cycle**: the future is walked in **3-month blocks**; each
  block copies three consecutive historical `z` values whose starting
  months-since-halving is within **±2 months** of the future block's own
  months-since-halving (future halvings from `_HALVING_EPOCHS`) (review
  M-6). If those candidates come from **fewer than 2 distinct cycles**, the
  block widens to the 6-month phase bucket, then to the whole eligible pool;
  the status line reports how often (review I-3).

With **Trend = None** the same draws use raw `r` instead of `z` (honest
replay), and the status line states the window's compounded pace, e.g.
"History 2015–2026 grew +59 %/yr on average; the fan compounds that."

**Eligible** months: inside the Raw window, and (cycle mode) all three
months of a block inside the window.

### Regime filter (Raw)

- **State** of a path at a month: its trailing 12-month log return minus the
  trend's expected 12-month return, divided by the trend's 12-month std,
  i.e. "how far above or below normal for its era" **(controller's call)**.
  With Trend = None: the plain trailing 12-month log return. The first 12
  simulated months use real history for the months before today.
- **Bins**: `bub-mc-raw-bins` equal-count bins of the same statistic
  computed over eligible historical months, coldest first. Checklist labels
  show each bin's range (e.g. "−1.2σ to −0.4σ"; plain % for Trend = None).
- **Unchecked bins are never visited**: a draw (a month, or a block checked
  month by month) that would put the path's state into an unchecked bin is
  rejected and redrawn, up to 20 tries; after that the last draw is kept and
  counted. The status line reports the share of forced draws when it
  exceeds 1 % **(controller's call)**.
- Same server rule as model mode: applies only when 0 < checked < bins.
- The auto-tick of "Show individual price paths" on a real filter click
  applies to `bub-mc-raw-regime` too (own click stamp, same pattern as
  `assets/mc_regime_paths.js`; one Input per clientside callback).
- The Tab-1 regime note gets Raw wording: "Raw: paths never enter unchecked
  momentum bins."

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
  evenly spaced ranks from the 2nd to the 98th percentile. If lines ≥ sims,
  all are drawn.
- **Show extremes** (`bub-mc-extremes`, checklist `["yes"]`, default off):
  the top- and bottom-ranked paths, dashed and slightly bolder in the same
  dark neutral (new constants in `colors.py`), legend "Highest end" /
  "Lowest end", no hover.
- Both sit directly under **Show individual price paths** (`bub-mc-paths`,
  shared, still the master switch, default off) and only show in Raw mode.
- Lines keep today's payload treatment (`_add_mc_spaghetti`: every other
  step, 2 significant figures). ≤ 202 lines ≈ ≤ 25 KB gzip on top of the
  ≈ 38 KB fan.

### Raw controls (shown only in Raw mode; model controls hidden)

| Control | id | Default |
|---|---|---|
| Simulations (1–3200, free) | `bub-mc-raw-sims` | 1000 **(controller's call)** |
| Years | `bub-mc-raw-years` | 40 |
| History window | `bub-mc-raw-window` | all history (2010 – current year) |
| Trend | `bub-mc-raw-trend` | `power` (`power` / `exp` / `none`) |
| Momentum bins | `bub-mc-raw-bins` | 5 |
| Momentum regime checklist | `bub-mc-raw-regime` | all checked |
| Lines shown | `bub-mc-lines` | 20 |
| Show extremes | `bub-mc-extremes` | `[]` |

Labels are Raw-specific (review M-8): no "Markov transition matrix", no
"≤ 200 is free". Start year and entry percentile are hidden in Raw with the
hint "Raw starts today at the current price". Hidden vs disabled and exact
layout are the builder's call; positions mirror the model controls.

### Raw status line

A new `bub-mc-raw-status` div directly under the model-source dropdown, Raw
only. It always states the trend used and its fitted numbers and which
cycles were fitted (e.g. "Power law n = 5.8, vol ∝ t^−0.48, fitted on 4
cycles"), the current cycle's pace against the curve, the independent-mode
caveat ("No memory: understates multi-month swings"), cycle fallbacks,
forced draws, and any reason no fan is drawn (fewer than 12 eligible
months **(controller's call)**). `bub-mc-status` is not used for Raw.

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
