# Tab-1 Raw MC spec — adversarial review, round 5 (2026-11-10)

**Spec:** `docs/superpowers/specs/2026-11-10-tab1-raw-mc-design.md` at 85a45d13 (revision 5); tool at ee0b936d
**Counts: 0 Critical · 1 Important · 2 Minor · 5 Nit**

Scope: (1) whether each round-4 finding is resolved; (2) defects introduced by rev 5. Operator
decisions were not re-litigated. Read-only (nothing in the repo modified).

**Machine checks.**
- `tools/raw_mc_cycle_stats.py` (default) still prints exactly what the spec quotes: the
  Measured table, n = 5.12 (rms 0.110), vol 2.46·t^−0.49, exp(1.18 − 0.179 t) (rms 0.142),
  vol exp(0.66 − 0.087 t), and cycles 2–4 complete. The p0 guard does not change any fit.
- Prototypes live in
  `/tmp/claude-1000/-scratch-code-bitcoinprojections/3a79ac35-3364-4e4c-9013-baf596cc6ebf/scratchpad/r5/`:
  - `sim5.py`: rev 5 exactly as specified, built on r4's `sim4.py`.
    - σ is held constant within each future cycle at the cycle-midpoint `t`; the current
      cycle uses the midpoint of its remaining span. `SIG=cont` switches this off.
    - All three months of a block lie in one complete cycle, with no straddling (`STRICT=1`).
    - The start is fixed at `TODAY=2024-04-20`. With `TRUNC=1`, data are truncated at the
      start: 164 months, last month-end 2024-03, S0 = $64,814, same fits.
  - `zsum5.py`: test (a).
  - `decomp5.py`: per-block and per-cycle weights.
  - `med5.py`, `med5b.py`: test (b), 3200 paths, seeds 1–5 and 1–20. `DEBIAS=1` applies the
    proposed fix.
  - `regime5.py`: 90 cases, 1000 paths.
  - `pace5.py`: Trend = None pace.
- Setup sanity: the first simulated month has φ = 0.01, and months 0–47 are exactly future
  cycle 5. All 160 block starts find candidates at ±0.04 from all 3 complete cycles (no
  widening). The last future block (φ 0.948) has 6 strict candidates.
- Pinned targets at the fixed 2024-04-20 start:
  - **(a) |σ-weighted E[Σz]| ≤ 0.2 per whole cycle: FAILS. It is +0.413 in every cycle,
    for both power and exp** (I-1).
  - **(b) median within ±0.25 fan-sd at each halving 2028…2064: passes, narrowly.** Over 20
    seeds the worst case is power · cycle 0.239 and exp · cycle 0.203, and iid stays ≤ 0.04.
    The 20-seed mean deviation still rises steadily: power +0.08 → +0.20 at 2064, exp +0.08 →
    +0.16. This is the same bias as in (a).
  - **Regime (≤ 5 % unchecked excl. first 12 months; 40-y median state within checked range
    ± 0.5): 90/90 pass,** both from today (full data) and from the fixed start. The worst
    unchecked share is 1.57 % (none · cycle · {4}, today) and 0.65 % from the fixed start.
    The 40-y median lies inside the union of the checked bins in every case.

## Round-4 findings

| R4 | Finding | Status in rev 5 |
|---|---|---|
| I-1 | Drift tests anchored to the live date and calendar halvings | **Resolved** in mechanism: fixed halving start, frozen inputs, whole cycles only. But see I-1 below: the pinned (a) bound fails at that start. |
| I-2 | Exp · cycle median drift; (a) blind to σ weighting | **Resolved** for the σ mechanism. With σ constant within each cycle, exp · cycle has no σ-induced bias, and (a) is now σ-weighted. The residual drift is the z-weight bias of I-1, which shows in both trends. |
| M-1 | Trend = None example stale | **Resolved.** "+116 %/yr over 40 y, history 2010–2026 +137 %/yr" matches the default all-history window. Under strict blocks it is now +120 (N-3). |
| M-2 | Block membership ambiguous | **Resolved.** All three months come from one complete cycle, with no halving straddle. |
| M-3 | Script contradicts the start-guess and failure rule | **Partly resolved.** p0 is guarded with `max(mean, 1e-3)`. The `curve_fit` RuntimeError is still not caught (N-5). |
| M-4 | Bound in (a) loose (±0.5) | **Resolved as worded** (±0.2), but the bound is unachievable at the new fixed start (I-1). |
| N-1 | φ = −0.01 vs "φ ∈ [0, 1)" | **Not resolved.** Line 108 is unchanged, though the commit says "2N" (N-1). |
| N-2 | D recursion index slip; "today's historical D" mid-month | **Not resolved.** Line 218 is unchanged (N-2). |

---

## Important

### I-1. Test (a)'s pinned bound |·| ≤ 0.2 is unachievable at the pinned 2024-04-20 start: the expected z-sum is +0.413 per whole cycle, and the bias behind it pushes (b) to within 0.01 of its bound

**Where:** Generating a path, "Drift tests" (a): "|value| ≤ **0.2** per cycle". Spec line 164
also claims "≤ 0.11 in the prototype".

**Failure:** `zsum5.py` gives **+0.413 in every future cycle 5–14, power and exp alike**. Because
σ is constant within each cycle, the σ-weighted sum reduces exactly to the plain Σz, so the
trend does not matter. Relaxing block integrity to the start month only gives +0.288, which
also fails.

r4's ±0.2 came from a grid aligned to *today* (φ₀ ≈ 0.65), which gave +0.078. Moving the test
to a halving start moves the 3-month block grid to φ = 0.01, 0.07, …, 0.95, and the pool means
there do not cancel.

`decomp5.py` shows the cause. Every historical month is reachable, but the ±0.04 windows give
months unequal draw weights. Each cycle's own z sums to 0, but the *weighted* sums are cycle 2
−0.02, cycle 3 −0.26 and cycle 4 **+0.70**. That is a real upward drift of 0.41 z per cycle.
(b) sees it: the 20-seed mean deviation climbs +0.08, +0.10, … **+0.20** (power) by 2064,
which is 80 % of the ±0.25 bound at the 40-y maximum horizon. The worst seed is 0.239.

The production grid also differs from the tested grid. From a real "today", future blocks run
on a grid offset by today's φ and straddle future halvings, so a passing fixed-start test does
not certify the whole cycles users see. For example, r4's today-grid gave +0.078 where the
fixed grid gives +0.413.

**Fix (verified, `med5b.py DEBIAS=1`):**
1. **Restart the future block grid at each projected halving.** The current cycle's last block
   may be shortened. Every whole future cycle then uses the same 16-block grid as the
   fixed-start test.
2. **Debias cycle-mode z deterministically per request.** Compute E = Σ over that grid's blocks
   of the pool-mean block z-sum, and draw `z − E/48` in cycle mode (here 0.0086 per month,
   imperceptible as shape). Trend = None stays raw.

With both changes, (a) is exactly 0 by construction. (b) becomes ≤ 0.07 worst across 20 seeds,
with means of +0.01 or less at every halving for power and exp.

Then replace the "≤ 0.11" claim with the measured figure. Keep (a) as a guard that the debias
is applied: assert |E| ≤ 0.2 on the *debiased* pools, or better ≤ 1e-9.

The fallback is to loosen (a) to ±0.5 and (b) to ±0.3 and document the drift, but that
leaves the median visibly above the stated trend late in the 40-y fan.

---

## Minor

- **M-1. The σ-weighting in (a) is now vacuous, and nothing checks the σ hold directly.**
  (a) multiplies and divides by "that cycle's σ", which is the same constant, and it takes σ
  from the spec formula rather than from the generator. An implementation that kept
  continuous σ(t) would still pass (a); only (b) would catch it, statistically, for exp.
  **Fix:** drop the weighting from (a) and state plainly that it is Σz. Add a unit test that
  the generator's σ is identical across all months of each future cycle, steps only at
  projected halvings, and equals the trend's value at the cycle-midpoint t, with the current
  cycle using its remaining-span midpoint.
- **M-2. The regime test is still pinned to the live date.** The r4 I-1 rationale (no test may
  depend on wall-clock date or growing data) applies equally to the regime test, and rev 5
  pins only the drift tests. It passes 90/90 from both today and 2024-04-20, but the bin
  edges and today's state move daily: today's state is −0.84 (power) now versus +0.15 at the
  fixed start. **Fix:** run the regime test on the same frozen inputs and fixed start.

## Nit

- **N-1.** (r4 N-1, carried.) Spec line 108 still says `φ ∈ [0, 1)`, but the script gives
  −0.01 for 2012-11. Say "≈ [0, 1)" or clip.
- **N-2.** (r4 N-2, carried.) Line 218 `D_{k+1} = D_k + r_k − μ(t_k)` should use `k+1`, and
  "today's historical D" still needs `+ log(P_today / P_last month-end)`.
- **N-3. Stale prototype figures.** Line 236 says "prototype: 0–0.8 %"; it is now 0–1.57 %.
  Line 164 says "≤ 0.11"; it is 0.20 worst for exp at the fixed start (see I-1). Line 203's
  "+116 %/yr" is +120 under strict blocks. Update the figures, or drop them as
  "illustrative".
- **N-4. The test inputs are under-defined.**
  - "Frozen price data": say "truncated at the start date". Results for (a) are the same
    either way, but S0, the iid pool and D_today differ.
  - "Each subsequent halving inside the horizon": the 2064-04-20 halving falls exactly at the
    40-y end. Say "≤ horizon", and evaluate at the month ending on the halving, not the
    nearest mid-month.
- **N-5.** The reference script still raises an uncaught `RuntimeError` when `curve_fit` fails.
  The spec says failure → no fan → status line. Catch it and print "no fit".
