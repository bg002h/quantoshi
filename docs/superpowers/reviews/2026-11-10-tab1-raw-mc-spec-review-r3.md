# Tab-1 Raw MC spec — adversarial review, round 3 (2026-11-10)

**Spec:** `docs/superpowers/specs/2026-11-10-tab1-raw-mc-design.md` at f88889d6 (revision 3)
**Counts: 0 Critical · 4 Important · 3 Minor · 2 Nit**

Scope: (1) whether each round-2 finding is resolved; (2) defects introduced by rev 3 (0b67508a plus
the rev-3 fold). Operator decisions were not re-litigated. Read-only.

**Machine checks.**
- `tools/raw_mc_cycle_stats.py` (default, `--from 2013`) reproduces every number in the Measured
  table, the default fit (n = 5.12, rms 0.110, vol 2.46·t^−0.49, exp(1.18 − 0.179 t), rms 0.142,
  vol exp(0.66 − 0.087 t)), the 2030/2050 implied rates and the from-2013 fit (n = 5.93,
  t^−0.72). The spec quotes the script exactly.
- Prototypes live in
  `/tmp/claude-1000/-scratch-code-bitcoinprojections/3a79ac35-3364-4e4c-9013-baf596cc6ebf/scratchpad/r3/`:
  - `sim3.py`: rev-3 simulator. Imports the script's `monthly_returns`, `cycle_table` and `fits`.
    It implements φ = days since cycle start at the month midpoint ÷ cycle length, with future
    cycles from 2024-04-20 + 4k years. Blocks come from full cycles at ±0.04, widening to ±0.08
    and then the whole pool on fewer than 2 distinct cycles. The level state D accumulates
    r − μ(t) from the first window month and is normalised by the historical std of D. Steering
    takes best-of-20, and cycle blocks are accepted month by month.
  - `dbg.py`, `fix.py`, `fix2.py`: drift diagnosis and fixes.
  - `regime3.py`: all 15 selections × both variants × three trends.
  - `closed.py`: the closed-bin fix.
  - `pace.py`: the pace under Trend = None.
- Start conditions: start $74,406, t = 17.30, 480 months. 4000–6000 paths for medians, 1000 for
  the filter tests.

## Round-2 findings

| R2 | Finding | Status in rev 3 |
|---|---|---|
| I-A | Cycle-mode median runs above trend | **Not resolved.** Rev 3 makes cycle 1 "full", which brings the drift back and makes it larger (I-1). The pinned test is also ill-posed (I-2). |
| I-B | "Never" filter is mostly not applied | **Resolved.** With the level state, best-of-20 and month-by-month block acceptance, the share of months in unchecked bins (first 12 excluded) is 0.0–0.8 % for all 15 selections × 2 variants × 3 trends. The ≤ 5 % target is achievable. The start-state rule and the measured-share note were added. |
| I-C | Momentum state compounds without bound | **Resolved in direction.** The state is now a level. However, the outer bins are open-ended, so the bounded-median target fails (I-3). |
| I-D | Fit set under-determined | **Resolved.** Data months, t_c and the objective are defined, the default is 4 cycles, and the script reproduces the numbers. Residue in M-2. |
| M-1 | Units of n | Resolved |
| M-2 | Lines payload | Resolved (figures consistent with r2; the default ≈ 3 KB is plausible for the 10-year auto-widen span) |
| M-3 | 12-month floor | Resolved (36 months, sims < 100 warning). For a cycle-mode gap, see M-3 below. |
| M-4 | Cycle-mode caveat | Resolved |
| M-5 | Halving-month assignment | Resolved (month-end rule, same as the script) |
| M-6 | Raw Years options | Resolved (10/20/30/40, no bolding, ≤ 2080) |
| M-7 | Pre-window lookback | Resolved |
| N-1 | Rank selector | Resolved |
| N-2 | g × t units | Resolved |

---

## Important

### I-1. Counting cycle 1 as "full" puts the cycle-mode drift back: E[Σz] = +2.1 per future cycle, median ×2.4 at 10 y and ×14 at 40 y (power)

**Where:** Trend curves ("A cycle is **full** when … ≥ 24 data months (cycle 1 has 27, so all four
completed cycles are full today)"). Generating a path, Raw · halving cycle ("from a **full cycle**
… measured residual drift −0.02σ per cycle").

**Failure:** The −0.02 figure that the spec cites comes from r2 `bias2.py` with the genesis and
current cycles **excluded** (`exclude=(0, 4)`). With cycle 1 included, that same script gives
+0.76. Rev 3's definition of "full" admits cycle 1. Its data cover only φ = 0.41–0.97, because
price data start in 2010-08. So at φ < 0.41 (the post-halving bull phase) the candidates come from
3 cycles, and at φ ≥ 0.41 from 4. Each month of cycles 2–4 in the bull phase therefore gets 4/3 the
weight of the same cycle's bear-phase months. The z values average zero per cycle only under
uniform weights. Under the φ weights, cycles 1–4 average +0.075 / +0.069 / +0.046 / +0.012
(`dbg.py`).

| | E[Σz] per 48 future months | median ÷ trend median, 10 y | 40 y |
|---|---|---|---|
| rev 3 as written, power | **+2.11** | **×2.4** (+0.50 fan-sd) | **×14** (+0.92 fan-sd) |
| rev 3 as written, exp | | ×1.6 (+0.49) | ×2.1 (+0.73) |
| blocks from cycles 2–4 only, power | −0.01 | ×1.63 (+0.33, see I-2) | ×1.24 (+0.04–0.09) |

This is the r2 I-A defect again, and larger. The status line says "Power law n = 5.12" while the
cycle fan grows far faster.

**Fix:** Separate the two meanings of "full".
- *Fitting* may keep the ≥ 24-data-month rule.
- *Block sampling* must use only cycles whose data cover the whole phase range (for example, data
  at φ ≤ 0.05 and at φ ≥ 0.95; today that is cycles 2–4).
- Alternatively, weight the candidates at each φ so that every contributing cycle carries equal
  total weight, or subtract each candidate set's mean z.

Pin E[Σz per future cycle], computed deterministically from the candidate sets, within ±0.1.

### I-2. The pinned cycle-mode median test cannot pass as written, even with a drift-free pool

**Where:** "A test pins the cycle-mode median within ±0.25σ (log) of the trend's median path at 10
and 40 years."

**Failure:**
- σ is undefined. It could be the fan's log std, the monthly σ(t), or annual vol.
- A cycle-mode median *should* differ from the trend at a horizon that is not a whole number of
  cycles from today. That difference is the rhythm the variant exists to show. Today φ = 0.65, and
  10 years ahead φ ≈ 0.14 (early bull). With the drift fixed (I-1, pool 2–4), `fix2.py` measures
  the deviation in fan-sd as follows:

| Trend | 4 y | 8 y | **10 y** | 16 y | 32 y | 40 y |
|---|---|---|---|---|---|---|
| power | +0.02 | +0.03 | **+0.31** | +0.03 | +0.02 | +0.04 |
| exp | +0.11 | +0.16 | **+0.36** | +0.21 | +0.26 | +0.26 |

Even a correct implementation fails the test at 10 y. Under the exponential trend, the decaying
σ(t) weights each cycle's early bull above its later bear, so exp stays at +0.26 at 40 y.

**Fix:**
- Define σ as the fan's cross-sectional std of log price at that horizon.
- Pin the median at whole-cycle horizons from today (4, 8, …, 40 years; 40 is aligned, 10 is not):
  ±0.1 fan-sd for power, ±0.3 for exp.
- Add the deterministic E[Σz] check from I-1, which is free of simulation noise.

### I-3. The bounded-median target fails for every selection that includes the coldest or hottest bin, because the outer bins are open-ended

**Where:** Regime filter: "The state is bounded: it reverts like the model-mode bands do". The test
"40-year median within the range of the checked bins' historical levels". Bins: "equal-count bins
of the historical state".

**Failure:** Nothing in the generator reverts. D evolves as D + σ(t)·z, a random walk. It is
bounded only by the filter's rejections. The coldest bin is (−∞, e₁] and the hottest is
[e₄, +∞). A path held in either bin is rejected only when it crosses the inner edge, so it drifts
outward without limit. Power trend, 40-year median state against the checked bins' historical
range (`regime3.py`):

| Checked | iid | cycle | historical range |
|---|---|---|---|
| {0} | −3.57σ ($5.1 M) | −2.60σ | −1.48…−0.81 |
| {4} | +3.78σ ($717 M) | **+6.57σ ($4.7 B)** | +0.94…+3.52 |
| {0,1} / {0,3} / {0,4} | −3.1 / −3.6 / −3.6 | −1.9 / −2.3 / −2.6 | — |
| {2,4} / {3,4} | (pass) / +3.12 | **+5.67 / +5.89** | up to +3.52 |

That is 10 of 15 selections failing in at least one variant with the power trend, and similar
counts for exp and none. With Trend = None, {4} cycle reaches 12.3σ. The note's claim that the
state "reverts like the model-mode bands do" is false. The middle bins (1–3) pass.

**Fix:** Close the outer bins at the historical min/max state, treating anything beyond as
unchecked, so steering keeps paths inside the historical range. In `closed.py`, every tested
extreme selection then passes. Unchecked share is ≤ 0.01 %, and the 40-year medians are −1.16σ
to +2.25σ. Example: {4} cycle median $257 M instead of $4.7 B. Delete "it reverts like the
model-mode bands do", or say the bounding comes from the filter.

### I-4. Trend = None in cycle mode compounds +220 %/yr while the status line quotes the window's +137 %/yr

**Where:** "With **Trend = None** the same draws use raw `r` … the status line states the window's
compounded pace … 'the fan compounds that.'" Combined with rev 3's "blocks from a **full
cycle**".

**Failure:** Under Trend = None, cycle mode draws raw returns only from full cycles, so it leaves
out the weak current cycle (+4 %/yr), and it uses the φ weights from I-1. At the default window,
the pool's weighted pace is **+220 %/yr**, against the window pace of **+137 %/yr** (`pace.py`).
The status line's growth statement does not describe the fan. That breaks the anti-goal "a fan
whose growth assumption is not stated on screen" and is a defect in what the tool claims.

**Fix:** Compute the stated pace from the pool the variant actually samples (the candidate weights
over the horizon), or make Trend = None cycle mode draw from every cycle in the window. Pin a test
that the stated pace matches the fan's median log growth within a few %/yr in both variants.

---

## Minor

- **M-1. Level-state anchor: bin membership is anchor-invariant, but labels are not, and the spec
  contradicts itself.** The spec says "anchor and normalisation are pinned by tests, exact formula
  is the builder's" — a test cannot pin what the spec leaves open. Shifting the anchor shifts the
  historical D, today's D and the equal-count edges by the same constant, so binning and steering do
  not change. That holds only if the simulated D continues the same anchored curve:
  D_{k+1} = D_k + r − μ(t_k), with D_0 = today's historical D. The user-visible labels ("−1.2σ to
  −0.4σ below trend") do move with the anchor. With the first-window-month anchor, the power edges
  are −1.48 / −0.79 / −0.33 / +0.25 / +0.93 / +3.52, and under Trend = None the coldest bin is
  −0.10…+1.20σ, so all of history reads as "above trend". **Fix:** pin the anchor so the mean
  historical D over the window is 0, state the continuation formula above, and normalise by the
  std over eligible months.
- **M-2. Window-rule statistics and fallback are not in the reference script.** For a cycle that
  is partly inside the window, the spec's "t_c = mean t of its data months" can be read as all its
  data months. The script uses in-window months: from 2013, cycle 2 has 42 months and t = 5.18,
  which is how it gets n = 5.93. The "< 3 full cycles → fit on all full cycles" fallback is not
  implemented by the script's functions, which the spec says the tests pin from. `--from 2015`
  prints a 2-cycle fit with rms 0.000 and an `OptimizeWarning`. **Fix:** say "in-window data
  months", and add the fit-set rule to the script (e.g. a `fit_set(tab_window, tab_all)`).
- **M-3. Narrow windows starve cycle mode silently.** The 36-month floor counts all eligible
  months, but cycle mode samples only full-cycle months. Example: a 2023–2026 window has 46
  eligible months but only 15 full-cycle months (cycle 4, φ 0.67–0.97), so every future block
  replays late 2023–24 through the whole-pool fallback. A window starting after 2024-04 leaves the
  block pool empty. **Fix:** apply the floor to the cycle-mode block pool (for example, at least one
  phase-covering cycle), else refuse with a status reason.

## Nit

- **N-1.** The φ tolerance is not stated to wrap around. A future block at φ = 0.98 cannot match
  historical φ = 0.01–0.02. Say whether φ is circular. Linear matching is fine if it is stated.
- **N-2.** "No fallback between trends" leaves a failed `curve_fit` with no defined behaviour.
  The script's `p0 = log(mean g)` is NaN once the mean fitted growth is ≤ 0, and convergence is
  not guaranteed. State the failure behaviour, e.g. "no fan, status reason".
