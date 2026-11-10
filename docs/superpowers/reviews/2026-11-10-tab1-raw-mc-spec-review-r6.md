# Tab-1 Raw MC spec — adversarial review, round 6 (2026-11-10)

**Spec:** `docs/superpowers/specs/2026-11-10-tab1-raw-mc-design.md` at db940f19 (revision 6); tool at 141143b2
**Counts: 0 Critical · 0 Important · 3 Minor · 4 Nit**

Scope: (1) whether each round-5 finding is resolved; (2) defects introduced by rev 6 (halving-aligned
block grid with a short last block, debias `z − E/48`, frozen-input tests, σ-hold test, D recursion).
Operator decisions were not re-litigated. Read-only (nothing in the repo modified).

**Machine checks.**
- `tools/raw_mc_cycle_stats.py` (default) still prints what the spec quotes: n = 5.12 (rms 0.110),
  vol 2.46·t^−0.49, exp(1.18 − 0.179 t) (rms 0.142), vol exp(0.66 − 0.087 t). The `fits()` call
  now catches `RuntimeError` and prints "no fit".
- Prototypes are in
  `/tmp/claude-1000/-scratch-code-bitcoinprojections/3a79ac35-3364-4e4c-9013-baf596cc6ebf/scratchpad/r6/`.
  - `sim6.py` implements rev 6 as written, built on r5's `sim5.py`:
    - The block grid restarts at each projected halving; a short last block copies L ≤ 2 months.
    - In cycle mode the generator draws `z − E/48`, where E is taken from the first whole future
      cycle's grid.
    - Simulated step k spans [today + k mo, today + k+1 mo]. Cycle, φ (clipped to [0, 1)) and the
      σ hold are taken at the step midpoint.
    - With `TRUNC=1` the data are frozen and truncated at 2024-04-20 (S0 = $64,814).
  - `grid6.py`, `scan6.py`: grid shape, E and E′ per cycle.
  - `med6.py`: test (b).
  - `regime6.py`: the regime test, 90 cases × 3200 paths.
  - `offgrid6.py`: drift over whole cycles from a start that is not on a halving.
- **Is E well-defined?** Yes, for whole future cycles.
  - `_HALVING_EPOCHS` projects halvings on exact 4-year anniversaries of 2024-04-20, and steps are
    month-anniversaries of today. So **every whole future cycle has exactly 48 steps (16 × 3) and
    an identical φ grid.**
  - This was checked on 51 start dates: 45 dates 9 days apart from 2026-11-10, plus 2028-02-29 and
    dates on either side of the 2028 halving. Each one gives 48 steps per whole cycle, the same E
    in every whole cycle, and max |E′| = 3.1e-15.
  - A 1–2-month block appears only at the end of the current partial cycle, or where the horizon
    cuts the last cycle. So `E/48` is exact under the current epochs (but see N-1).
  - In the partial cycles, E′ ranges from −6 to +6. That is the untested cycle-position rhythm,
    as intended.
- **(a) |E′| ≤ 1e-9: passes** at the frozen start, at 3e-15 in all 10 whole cycles.
  - E = +0.340 here, against r5's +0.413 using a `+15 d` midpoint. Across the 51 start dates E
    ranges from −0.33 to +0.41, depending on the step's day offset from the 20th (M-1).
- **(b) passes, 20 seeds × 3200 paths, frozen start, halvings 2028…2064.** Each step ending on a
  halving was asserted to match the halving date exactly.

  | Trend · variant | Worst | Mean deviation per halving |
  |---|---|---|
  | power · iid | 0.073 | −0.01 to −0.02 |
  | power · cycle | 0.055 | 0.00 to +0.01 |
  | exp · iid | 0.080 | −0.01 to −0.02 |
  | exp · cycle | 0.063 | 0.00 to +0.01 |

  Without the debias, cycle mode drifts again: +0.05 → +0.17 for power and +0.05 → +0.12 for exp.
  - From an off-halving start (2026-11-25, full data), the debias is still effective. Drift over
    9 whole cycles is −0.04 for power and −0.03 for exp, against −0.16 and −0.12 without it.
- **Regime test (frozen inputs, debias, new grid): 90/90 pass.**
  - The worst unchecked share is **0.54 %** (none · cycle · {2}).
  - Every 40-y median lies inside the checked range ± 0.5, and also inside the union of the
    checked bins.

## Round-5 findings

| R5 | Finding | Status in rev 6 |
|---|---|---|
| I-1 | (a) ≤ 0.2 unachievable at the fixed start; z-weight drift | **Resolved.** The grid restarts at halvings and the debias is applied per request. (a) is 3e-15 and (b) has a worst case of 0.080 with means ≤ 0.02. Grid conventions: M-1, M-2. |
| M-1 | σ weighting in (a) vacuous; σ hold untested | **Resolved.** (a) is now the plain z-sum, and (c) tests the σ hold directly. The step-assignment rule (c) relies on is unpinned (M-1). |
| M-2 | Regime test pinned to the live date | **Resolved.** The regime test now uses the same frozen inputs (passes 90/90 above). |
| N-1 | φ = −0.01 vs "[0, 1)" | **Resolved** (clip). |
| N-2 | D recursion index; D_0 mid-month | **Resolved** as worded. The partial month's μ is still missing (N-3). |
| N-3 | Stale prototype figures | **Partly resolved.** The 1.57 % figure is labelled "frozen inputs" but was measured at the live date, and 0.07 is now 0.080 (N-2). |
| N-4 | Test inputs under-defined | **Resolved.** Data are truncated at the start, the start price is the 2024-04-20 close, the horizon condition is "≤ the horizon", and evaluation is at the month ending on the halving. |
| N-5 | Uncaught `curve_fit` RuntimeError | **Resolved** (141143b2). |

---

## Minor

- **M-1. The rule that assigns a simulated step to a cycle and φ is unpinned, yet E, the grid
  restart and test (c) all depend on it.**
  - **Where:** "Block grid and debias" says blocks "run in 3s from the halving". Test (c) says σ
    "steps only at projected halvings".
  - **Failure:** for any today not on the 20th, a projected halving falls inside a monthly step,
    so "from the halving" does not say which step opens the new cycle. φ per step is also left
    open (step start, midpoint, or `+15 d`), and E is sensitive to that choice:
    - +0.340 with the true step midpoint, +0.413 with r5's `+15 d`;
    - −0.33 to +0.41 across start days.

    Two conforming implementations can therefore draw different fans. Test (c) also cannot state
    which step σ changes at.
  - **Fix:** pin one rule: "each simulated step belongs to the cycle, and has the φ, of its
    midpoint (step start + half its length)". Use it for the grid restart, the σ hold, CAND and
    E. Mark the "+0.41 / 0.009" figures as convention-dependent and illustrative; the step
    midpoint gives +0.34 / 0.0071.
- **M-2. Steering's mid-block redraw is not reconciled with the grid restart.**
  - **Where:** the regime filter's steering rule, "the block is accepted month by month up to its
    first violating month and redrawn from there", read against "the future block grid restarts
    at each projected halving".
  - **Failure:** a redrawn block starts off-grid. If it runs a full 3 months, it can cross a
    projected halving and copy late-cycle z into the new cycle's φ ≈ 0. The restart then holds
    only for unsteered paths.
  - **Fix:** state that a block, whether new or redrawn, ends at the end of its grid block, so
    never past a halving. The next grid block starts fresh. `sim6.py` does this and still passes
    90/90.
- **M-3. Test (a) is tautological unless it reads the generator's own schedule and shape array.**
  - **Failure:** E′ = E − 48·E/48 holds by algebra. A test that recomputes E and E′ from the
    spec's formula passes even when the generator skips the debias or uses a different grid. This
    is the same class of defect as r5 M-1.
  - **Fix:** compute E′ from the generator's actual block-start schedule, candidate pools and
    shifted `z` array, exposed for the test. Optionally, also assert that the generator's grid
    has 16 blocks of 3 per whole cycle.

## Nit

- **N-1. Two ways to make E's definition robust.**
  - `E/48` hard-codes 48 months per cycle. That is exact under `_HALVING_EPOCHS` (verified
    above), but it breaks silently if the epochs are ever projected from block height. Write
    `E/n`, with n the number of steps in that cycle's grid, or assert n = 48.
  - "One whole cycle's grid" can be absent from a short horizon. Define E on the first projected
    future cycle's grid, whatever the horizon.
- **N-2. Figures and attributions.**
  - Regime: "prototype: ≤ 1.57 %, frozen inputs". 1.57 % was the live-date figure. Frozen inputs
    give 0.65 % (r5) and 0.54 % (r6).
  - (b): "prototype worst 0.07". It is 0.080 (exp · iid) in r6.
  - Trend = None: "+120 %/yr … figures from the r4 prototype". The figure is from r5.
  - A short block's candidate pool is not stated. Say it draws from 3-month-valid starts and
    copies the first L months, or from L-month-valid starts.
- **N-3. D_0 omits the partial month's trend.** `D_0 = D_lastME + log(P_today/P_lastME)` adds the
  partial month's return but not its `−μ·(elapsed fraction)`. The offset is at most about one
  month of trend (≈ 0.02 log for power in 2026, ≈ 0.03σ of state). Subtract it, or say the
  omission is accepted.
- **N-4. (Pre-existing, outside rev-6 scope; filed for visibility.)** A cycle with one in-window
  month has `σ_c` = NaN (sample std, ddof = 1). For example, in May 2028 cycle 6 holds only the
  2028-04 month. Its z is then NaN and enters the iid pool, so iid paths go NaN for about a month
  after each halving. **Fix:** exclude cycles with fewer than 2 in-window months from the z pool,
  or pin a minimum count.
