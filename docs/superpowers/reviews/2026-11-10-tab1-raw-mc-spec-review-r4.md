# Tab-1 Raw MC spec — adversarial review, round 4 (2026-11-10)

**Spec:** `docs/superpowers/specs/2026-11-10-tab1-raw-mc-design.md` at 3cb42b28 (revision 4); tool at 1301ac68
**Counts: 0 Critical · 2 Important · 4 Minor · 2 Nit**

Scope: (1) whether each round-3 finding is resolved; (2) defects introduced by rev 4. Operator
decisions were not re-litigated. Read-only (nothing in the repo modified).

**Machine checks.**
- `tools/raw_mc_cycle_stats.py`, default / `--from 2013` / `--from 2015`. The spec quotes the
  script exactly: the Measured table, n = 5.12 (rms 0.110), vol 2.46·t^−0.49, exp(1.18 − 0.179 t)
  (rms 0.142), vol exp(0.66 − 0.087 t), the 2030/2050 rates, and from 2013 n = 5.93 with
  t^−0.72 on 3 in-window cycles (cycle 2 has 42 months, t = 5.18). `complete` is set for cycles
  2–4 only. `--from 2015` gives "fit set: all history", with cycle 2 at 18 months and not full,
  and the 2-cycle rms-0.000 fit from r3 is gone.
- Prototypes live in
  `/tmp/claude-1000/-scratch-code-bitcoinprojections/3a79ac35-3364-4e4c-9013-baf596cc6ebf/scratchpad/r4/`:
  - `sim4.py`: the rev-4 simulator. It uses the script's `monthly_returns` (including φ),
    `cycle_table`, `fit_set` and `fits`. Blocks come from complete cycles only. The level state
    D has mean 0 over eligible months. The outer bins are closed at the historical min/max state.
    Steering is best-of-20 and month by month. `STRICT=1` requires all three block months to lie
    in the same complete cycle (see M-2). `TODAY=` sets the start date.
  - `zsum.py`: deterministic E[Σz] per future cycle.
  - `med.py`: median against the trend at the projected halvings, 3200 paths, seeds 1–3.
  - `remainder.py`: E[Σz] from a start phase to the next halving.
  - `regime4.py`: 15 selections × 2 variants × 3 trends, 1000 paths.
  - `pace4.py`: the Trend = None pace.
  - `pwsigma.py`: the fix for I-2.
- Pinned targets:
  - **Regime (≤ 5 % unchecked, 40-y median state within the closed range ± 0.5): achievable.**
    All 90 cases pass. The unchecked share is 0.00–1.27 % (worst: none · cycle · {4}). Every
    40-y median lies within ±0.5 of the checked hull, and also of the union of the checked bins.
  - **(a) |E[Σz]| ≤ 0.5 per future cycle: passes today** (whole cycles −0.105, or +0.078 with
    STRICT; current partial +0.19 / +0.37; last partial −0.29). It fails on other start dates
    (I-1).
  - **(b) median within ±0.25 fan-sd at the halvings:** power passes (iid ≤ 0.05; cycle
    ≤ 0.09, or ≤ 0.16 with STRICT). **Exp · cycle fails (I-2).** All of (b) is date-dependent
    (I-1).
  - **Trend = None pace:** the status line now uses the fan's own pace, so the claim holds by
    construction. The example numbers are stale (M-1).

## Round-3 findings

| R3 | Finding | Status in rev 4 |
|---|---|---|
| I-1 | Cycle 1 counted as full brings the drift back | **Resolved.** Blocks come only from complete cycles (2–4). E[Σz] per whole future cycle is −0.105 (+0.078 STRICT), down from +2.11. |
| I-2 | Cycle-mode median test ill-posed | **Not resolved.** σ is now defined (fan cross-sectional std), but the horizons are calendar halvings rather than whole cycles from the start, so both tests depend on the date (I-1). The ±0.25 bound fails for exp (I-2). |
| I-3 | Open outer bins → unbounded median | **Resolved.** With closed outer bins, all 90 cases pass. The "reverts like the model-mode bands" claim is deleted. |
| I-4 | Trend = None status pace ≠ fan pace | **Resolved** in mechanism (the pace is computed from the paths). The example text is stale (M-1). |
| M-1 | Level-state anchor | Resolved (mean-zero C, continuation, S). Index slip in N-2. |
| M-2 | In-window months + fit-set rule in script | Resolved (`fit_set`; `--from 2015` falls back correctly). |
| M-3 | Narrow windows starve cycle mode | Resolved (cycle mode needs ≥ 2 complete cycles, else no fan and a reason). |
| N-1 | φ circular? | Resolved (linear). |
| N-2 | `curve_fit` failure | Resolved in spec. The reference script still contradicts it (M-3). |

---

## Important

### I-1. Both drift tests are anchored to calendar halvings and the live date: they pass today only because today's φ ≈ 0.65 sits where the remainder z-sum is ≈ 0, and they go red without any code change once φ passes about 0.1 (late 2028)

**Where:** Generating a path, "Drift tests": (a) "for each future cycle … |value| ≤ 0.5"; (b) "at
whole-cycle horizons (the projected halvings 2028, 2032, 2036, …)". Paths start "today" at
today's t and φ.

**Failure:** A projected halving is a whole number of cycles from *the previous halving*, not
from the start. The first halving horizon covers the rest of the current cycle, φ₀ → 1. Over that
stretch the expected z-sum is the rhythm itself, which is what r3 I-2 said must not be tested.
The deterministic remainder E[Σz from φ₀ to 1] (`remainder.py`) is:

| φ₀ | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | **0.65 (today)** | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E[Σz] | −0.18 | −2.49 | −4.66 | −6.23 | −6.29 | −3.54 | −0.37 | **+0.41** | −0.04 | −0.21 | +0.37 |

Results with `TODAY=` set to other dates (power, cycle mode, 3200 paths):

| start | φ₀ | (a) current / last partial cycle | (b) dev at 1st halving … 2064, fan-sd |
|---|---|---|---|
| 2025-08-10 | 0.34 | **−6.37 / +6.27** | **−1.59 … −0.49** |
| 2029-06-10 | 0.30 | **−5.93 / +6.02** | **−1.24 … −0.30** |
| 2030-02-10 | 0.46 | **−5.39 / +5.28** | **−1.67 … −0.41** |

The simulator does what the design intends here: starting near a peak, the fan sits below
"today × trend" permanently. It is the test reference that is wrong. Because the spec seeds and
starts from today, and the price data grow daily, a test written as specified starts failing on
its own. "For each future cycle" also leaves undefined whether the partial first and last cycles
count.

**Fix:**
- Run both drift tests from a **fixed start at φ₀ = 0**: a synthetic start date on a halving,
  with frozen data and a frozen "today" injected into the generator.
- Use horizons of whole cycles from that start.
- Restrict (a) to whole future cycles.
- Alternatively, keep today's start but compare (b) against the trend median plus the
  deterministic σ-weighted remainder offset. Tests must not depend on the wall-clock date.

### I-2. Exp-trend cycle mode fails the ±0.25 fan-sd target, and test (a) cannot see why

**Where:** Drift tests (b) "±0.25 × the cross-sectional std … both variants" with the exp
trend; test (a) sums unweighted z.

**Failure:** `med.py`, exp · cycle, at the 2032–2064 halvings:

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| block start in a complete cycle | +0.23 | **+0.26** | **+0.28** |
| STRICT (all 3 months in the cycle) | **+0.32** | **+0.34** | **+0.36** |

The values rise steadily with the horizon. This is the r3 I-2 mechanism: exp vol decays at
δ = −0.087/yr, so within each future cycle σ(t) is about 1.4× larger in the early bull (z > 0)
than in the late bear (z < 0). Σσ(t)z therefore stays positive even though Σz ≈ 0. Test (a)
passes (−0.105 or +0.078) while the drift exists. R3 measured +0.26 and proposed ±0.3 for exp;
rev 4 tightened the bound to ±0.25 for both trends. Whether the test passes now depends on the
seed, or it fails outright under the STRICT reading. Power passes (≤ 0.16).

**Fix (preferred, verified in `pwsigma.py`):** evaluate σ **piecewise-constant per future cycle**
(at the cycle's mean t) in cycle mode, so that Σσz = σ_c·Σz. Exp · cycle then measures
−0.05…+0.11 at every halving for both readings and seeds 1–3. Make test (a) σ-weighted:
E[Σ σ(t_k) z_k] / mean σ. The fallback is to loosen exp to ±0.4 and document the bias, but then
the median sits visibly above the stated trend.

---

## Minor

- **M-1. The Trend = None example status line is stale and mislabelled.** "+220 %/yr" was r3's
  figure for a pool that included cycle 1. With rev 4's pool (cycles 2–4), the cycle-mode fan
  median grows **+135 %/yr at 10 y and +116 %/yr at 40 y**; iid gives +135 / +137 (`pace4.py`).
  "+137 %/yr" is the **2010–2026** pace. 2015–2026 is +59 %/yr, as the Why section says. The
  pace also depends on the horizon. **Fix:** correct the example, and state the pace as the
  median log growth over the Raw **Years** horizon.
- **M-2. Block membership is ambiguous.** "Three consecutive historical z values from a complete
  cycle" could mean the start month only, or all three months. Blocks starting at φ > 0.94
  straddle into the next (possibly incomplete) cycle. The two readings give different whole-cycle
  E[Σz] (−0.105 vs +0.078) and exp drift (+0.23–0.28 vs +0.32–0.36). **Fix:** say which. All
  three months in the same complete cycle is cleaner; the late-φ blocks then rely on ±0.04
  matches near φ ≈ 0.94.
- **M-3. The reference script contradicts the N-2 sentence that tests pin from it.** `fits()`
  still uses `p0=(np.log(g.mean()), …)` and `np.log(s.mean())`, and does not catch the
  `curve_fit` RuntimeError. The spec says "the start guess must not take the log of a
  non-positive mean" and that "tests pin its outputs". **Fix:** in the script, use a safe p0
  (e.g. `log(max(mean g, small))` or a fixed guess) and handle failure as "no fit".
- **M-4. The bound in test (a) is loose.** At ±0.5 per cycle, a defect contributing +0.5 z per
  cycle (about 0.07–0.08 log per cycle under power vol, ≈ ×1.8 at 40 y) passes. The measured
  whole-cycle values are −0.105 / +0.078, so **±0.2** is achievable (±0.1 is not for the
  start-month reading). Once I-1 is fixed, tighten to ±0.2 over whole cycles.

## Nit

- **N-1.** The spec says φ ∈ [0, 1), but the script gives φ = −0.01 for 2012-11. The month-end
  rule puts that month in cycle 2, while its midpoint (the 15th) precedes the 11-28 halving.
  This is harmless. Say "≈ [0, 1)" or clip.
- **N-2.** The simulated recursion `D_{k+1} = D_k + r_k − μ(t_k)` mixes indices; it should be
  `r_{k+1} − μ(t_{k+1})`. "Today's historical D" is undefined mid-month, because historical D
  exists only at month-ends. Add `+ log(P_today / P_last month-end)`, or define the start at the
  last month-end.
