# Review r2: Percentile cycle fit — design spec (fold)

**Spec:** `docs/superpowers/specs/2026-11-10-percentile-cycle-fit-design.md` at `b37404b5` (the r1 fold).
**Previous:** `docs/superpowers/reviews/2026-11-10-pctile-cycle-spec-review-r1.md` (2C / 4I / 10M / 3N).
**Scope:** the fold only (`git diff 8e42e1f0 b37404b5 -- docs/superpowers/specs/`). Settled items (operator table, one element per model, refit band + outer band, M-1 eviction) are not re-raised.
**Reviewer:** independent (author != reviewer). Read-only. Experiments are scratch scripts in
`/tmp/claude-1000/-scratch-code-bitcoinprojections/6c13708a-e257-46cf-bdef-a2a8a773488f/scratchpad/r2/`.
They call `btc_web/pctile_cycle.py` (`fit_cycle`, `bootstrap_params`, `likely_band`, `hindcast`, `is_weak`),
`figures.percentile._percentile_series` and `timemachine.asof_qr` without modification.

**Counts: 0 Critical / 4 Important / 6 Minor / 2 Nit**

Implementation commits `0ed8fca1` and `f5ab25f1` landed while this review ran. They are out of scope.
However, `likely_band`, `hindcast` and `is_weak` in HEAD are the same functions measured here, and
`_cycle_result` (`figures/percentile.py:204-226`) drops censored days before the hindcast. So r2-I-1,
I-3 and I-4 also apply to the build as committed.

The fold reproduces r1's numbers: QR honest hindcast 53 %, 21.8 vs 23.3 pp; fit-band coverage 31 %.
The headline fixes hold up:
- The likely band is calibrated on average. Honest out-of-sample coverage is 77–81 %.
- The as-of QR hindcast is implementable as written.
- The predating rule is correct on every decode path.

What remains is in the gate and the hindcast plumbing around those fixes.

## r1 finding → resolved?

| r1 | resolved? | note |
|---|---|---|
| C-1 look-ahead hindcast | **partly** | The QR as-of hindcast is sound and implementable. The censor leak named in C-1's last paragraph is not folded (r2-I-3). Optimistic non-QR win rates are still shown and decide the forecast (r2-I-2). |
| C-2 band read as a range | **yes** | Likely band: in-sample 82 % (QR); honest out-of-sample mean 77 % (2-yr) and 81 % (4-yr). The construction is ragged (r2-I-4). |
| I-1 old links show fit on | **yes** | Correct for v4 registered fingerprints, v4 any scope, and q3 positional links. Notation nit in r2-M-1. |
| I-2 weak fits forecast | **yes** for drawing; **partly** for wording | Weak fits are no longer extrapolated. The period wording for at-limit and log fits is not specified (r2-M-4). The new hindcast arm is unstable (r2-I-1). |
| I-3 EPPL name clash | **yes** | Masters are used, and their `legend_name`s are distinct. This change moved LPPL across the gate (r2-I-2). |
| I-4 label overprint | **yes** at defaults (log, N = 1); **partly** on linear | r2-M-2 |
| M-1 … M-10, N-1 … N-3 | mostly yes | M-10 censored hindcast → r2-I-3; M-7/M-8/M-9 adopted |

---

## I-1 — The weak gate's hindcast arm decides the default QR forecast on a margin that daily data shifts across zero

**Evidence.** `r2/gate_stability.py`, QR honest hindcast (as-of fan), the spec's rule "weak if mean RMSE wave ≥ flat":

```
data truncated   0 d: n=17 win 53% wave 21.8 flat 23.3 margin +1.5 -> weak 0
data truncated  90 d: n=17 win 53% wave 22.5 flat 23.4 margin +0.9 -> weak 0
data truncated 135 d: n=17 win 47% wave 22.8 flat 23.0 margin +0.1 -> weak 0
data truncated 165 d: n=17 win 47% wave 23.1 flat 23.1 margin -0.0 -> weak 1
-- data fixed, cutoff grid shifted earlier by δ --
δ=0.00 margin +1.5 weak 0 | δ=0.30 +0.5 weak 0 | δ=0.40 +0.1 weak 0 | δ=0.45 -0.0 weak 1
```

**Why.** The cutoffs are `arange(last-10, last-2, 0.5)`, anchored to `last`. Every daily price
update moves them by one day. The default model's edge is at most 1.5 pp on about 4 independent
windows, so the binary verdict crosses zero within a half-year of window slide. On the days it does,
the default view loses its forecast, axis extension and labels after a routine price update, with no
change to the model. The feature is going to production for the operator to test, and there a
forecast that disappears with no change to the model will look like a bug.

**Fix.**
1. Do not let a noise-level margin suppress the forecast. Two options:
   - make the hindcast arm one-sided with a tolerance: weak only if wave RMSE ≥ flat + τ, with τ about 2 pp;
   - make the arm informational only: the summary says "no out-of-sample edge", while R² and `at_limit`
     still decide whether the forecast is drawn.
2. Remove the phase jitter. Monthly cutoffs (`every = 1/12`, n ≈ 97) cut the drift over the same
   0.4-yr slide to +1.6 → +0.8 pp. Calendar-anchored cutoffs would also work.
3. Pin it with a test: the QR default is not weak for data truncated by 0…180 days.

## I-2 — Optimistic hindcasts are reported as win rates and decide the forecast; with the masters picker, LPPL forecasts at "88 %" beside QR's honest 53 %

**Evidence.** `r2/weak_matrix.py`, resqr, `cal` element, censor off:

```
qr   R2 0.557 | opt win 76% w 21.4 f 27.0 | HONEST win 53% w 21.8 f 23.3 | weak 0
pl   R2 0.454 | opt win 59% w 27.1 f 28.0                              | weak 0
spl  R2 0.459 | opt win 65% w 25.2 f 27.5                              | weak 0
lppl R2 0.509 | opt win 88% w 21.7 f 31.5                              | weak 0
eppl R2 0.126 | opt win 76% w 30.0 f 31.4                              | weak 1 (R² only)
```

The Statistical-judgment bullet says LPPL and EPPL have no element that beats flat out of sample.
That was measured on `lp3` and `ecfg_1d_1u` (`tools/pctile_cycle_basis_explore.py:67`). The I-3 fold
swapped in the masters, and for the masters the claim is false. Master `lppl` (1-freq) keeps a strong
3.6-yr cycle and passes the gate.

**Why.** For QR the leak inflates the margin from 1.5 to 5.6 pp (+4.1) and the win rate from 53 % to 76 %.
- PL's optimistic margin is 0.9 pp and SatPL's is 2.3 pp. Both are smaller than QR's leak, so both
  plausibly fail an honest test, yet the spec draws their forecasts.
- LPPL's optimistic 88 % sits in the same summary slot as QR's honest 53 %. A user who switches the
  picker will compare the two and conclude LPPL's cycle forecast is the better one. That is the ordering
  the evidence cannot support.
- r1 C-1(a) asked for no win rate on models without an as-of fan.

The one-sided arm can only add weakness, so a leaky number there is lenient rather than harmful in
itself. The defect is what passes the gate and how it is reported.

**Fix.**
- (a) No win share for non-honest hindcasts. Show the RMSE pair only, labelled "optimistic (today's fan)".
- (b) State the policy for models that have only an optimistic hindcast. Either treat them as weak
  ("no out-of-sample check for ‹model› — no forecast"), or draw the forecast with its legend entry
  saying "not checked out-of-sample".
- (c) Correct the judgment bullet for the masters: LPPL R² 0.51, optimistic 88 %; EPPL weak by R² only.
- Optional: the Time Machine grid has SatPL as-of frames (`timemachine.asof_spl`, parametric σ). An
  honest SatPL hindcast may therefore be possible at σ-mode `constant`. Unverified.

## I-3 — `censor` still leaks hindsight into the "honest" QR hindcast (r1 C-1, last paragraph, not folded)

**Evidence.** `r2/censor_hindcast.py`, QR as-of fan, 17 cutoffs. Six of the 17 scoring windows overlap the window:

```
none      : win 53% wave 21.8 flat 23.3      (margin 1.5)
fit+score : win 59% wave 18.6 flat 24.0      (margin 5.4)  <- spec: censor drops days from the series
fit-only  : win 53% wave 20.8 flat 23.4      (margin 2.6)
```

PL's optimistic win rate goes from 59 % to 82 % with censor on (`weak_matrix.py`).

**Why.** The 2019–20 window was chosen after seeing that the wave misfits there. Dropping it from
*scoring* removes exactly the days the wave got wrong. That more than triples QR's edge, and the
result is still presented as honest. This is the anti-goal "a skill number with look-ahead presented as
out-of-sample". The fold handled censor only in the R² wording (M-10).

**Fix.** `censor` removes the window from *fitting* only: the central fit, the bootstrap and each
hindcast fit. Hindcast scoring and both coverage figures always use every day, and the gate uses that
hindcast. Say so under Math → Hindcast.

## I-4 — The likely band as specified is ragged: one-week steps of up to 12–15 pp, which the hover then reports as a price range

**Evidence.** `r2/likely_fcst.py`, forecast region, weekly grid, B = 150, one residual path per replicate (spec):

```
qr: weekly steps median 1.75 pp, p95 4.21, max 12.46; weeks with a >5 pp step: 28 of 1565
pl: weekly steps median 1.63 pp, p95 4.34, max 15.45; weeks with a >5 pp step: 47 of 1565
(fit band, for comparison: max 1.4 pp)
```

The steps sit at block boundaries. Every replicate's path changes block in the same week (multiples of
52 from the grid start), and a pointwise 10/90 from 150 draws has about 3 pp of sampling SE.
`r2/likely_smooth.py` (coverage unchanged at 81–83 %):

```
K=1  synchronized (spec)  max step 13.07, p99 4.72
K=1  random block phase   max step  8.20
K=8  random block phase   max step  3.15, p99 2.14
K=20 random block phase   max step  2.17
```

**Why.** The band edge and the hover's "price at likely lo/hi" jump between adjacent weeks for no
reason. Readers take a notch as structure, for example "the range narrows in 2031". C-2 itself is
otherwise sound. `r2/likely_band.py`:
- in-sample coverage: 82 % for QR, 82 % for PL, 81 % for LPPL;
- honest QR out-of-sample coverage at the as-of fan: mean 77 % (2-yr) and 81 % (4-yr), per cutoff
  24–100 %. Calibrated on average, not window by window.

**Fix.** Use K ≥ 8 residual paths per replicate, each with a random block phase (start offset in
[0, 52)), under a fixed seed. That is about 3 M floats once per fit, which is negligible.

---

## Minor

**M-1 — `_PREDATING_LINK_VALUES["bub-pctile-fit:value"] = []` is not the dict's shape.**
Entries are `(value, requires)` tuples, unpacked by `for key, (value, requires) in …` (`snapshot.py:708-722` at `b37404b5`).
- A literal `[]` raises ValueError inside both decoders' `try`. Their `except Exception: return None`
  then makes every q3 and q4 link fail to decode.
- `state.get(None)` never fires, so `requires=None` also needs its own branch.

Write `([], None)` plus an explicit `requires is None` branch. HEAD does exactly this, and the existing
decode tests would catch the literal form.

Checked and sound:
- v4 fills every absent key from the link's registered fingerprint (`snapshot.py:949-960`), whatever the
  share scope. A link made after the feature exists therefore always carries the key.
- An explicit off encodes as mask 0 and decodes to `[]`, which counts as present.
- Only v4 is emitted (`snapshot_cb.py:331`), so every q3 link predates the field.

**M-2 — Linear-axis N = 3 labels collide.** Same-kind label pitch on a linear axis is P / span × plot
width. With t_lo 0.94 and today 17.3, the span is 26.4 / 36.4 / 46.4 yr for H = 10 / 20 / 30. That gives:
- 108 / 78 / 62 px on an 800 px plot;
- 34 / 25 / 19 px on a ~250 px phone plot.

A two-line label ("Aug 2028 / Q80 · $310k") is about 70 px wide, so labels overlap at H = 30 on desktop
and at every horizon on phones. Use N = 1 on both axes, or N = 3 only for linear with H ≤ 20.

**M-3 — Info div placement and clearing.**
- The div must sit inside `bub-pctile-wrap` (`layout/common.py:689`). `update_bub_pctile` returns
  `no_update` outside Percentile mode (`charts/__init__.py:904-905`), so an outside div would keep
  stale text on screen.
- The clientside callback must return empty for a null figure (the graph mounts empty inside
  `bubble-lazy`) and when `meta.cycle_info` is absent (fit off).

Checked and sound:
- `charts/__init__.py:881` is the only writer of `bub-pctile-graph.figure`.
- Restore goes through it: `apply_tab_bubble` writes the controls and `snapshot-pending=False` in one
  batch (`snapshot_cb.py:264-295`).
- `layout.meta` survives the cache's JSON round-trip (verified with `pio`).

**M-4 — Summary wording for at-limit and log fits.**
- At-limit fits are now weak, but Goal 4 still prints a period: `hybppl`/`cal` would show "2.00 yr",
  which is the grid edge.
- For `log`, `CycleFit.period` is the scale ratio λ = e^(1/f) (`pctile_cycle.py:90-93`). QR log's
  "2.42" means t_next = 2.42·t, not 2.42 yr.

Specify "period at search limit (≤ 2 yr / ≥ 8 yr)" and "turn ratio λ".

**M-5 — Cost and fallback of the honest hindcast are unspecified.**
- `timemachine._load()` takes 0.59 s and retains 18.9 MB per worker (dev).
- On top of that there is one as-of percentile series per distinct frame: 17 cutoffs now, about 97 if
  I-1's monthly cutoffs are adopted.
- The fold dropped the "< 1.5 s cold" budget and the cache location.

Specify:
- Cache the as-of series per (frame, data fingerprint), shared across element and censor.
- Fallback when `tm.available()` is false or `asof_qr` returns None: use the optimistic label.
  `hindcast()` skips None cutoffs, so if every cutoff is None it returns n = 0 and the gate's hindcast
  arm silently disappears.

**M-6 — The footnote no longer says the price range assumes the model's fan.** The pre-fold footnote
said the band excludes model-form uncertainty, and the folded list does not carry that over. The
likely-band price hover at +20–30 yr is a dollar range that holds only if today's fan holds for
30 years. Add one clause.

## Nit

**N-1 — The predating rule is unconditional.** It also turns the fit off for the recipient of an old
*Price*-view link who later clicks Percentile, while a fresh visitor gets the fit on. Gating on
`("bub-view-mode:data", "percentile")` (`bub-view-mode` is snapshotted, `snapshot.py:308`) avoids this.
Either choice is acceptable; say which.

**N-2 — Links age out of the forecast.** Take a link with x-range [2010, 2027], opened after
2027-01-01. The cap only lowers values, so it stays at 2027, and `yr_to_t(2027) < today` hides the
forecast the sender saw.

## Not checked

Browser rendering, E2E, prod timing, non-resqr σ-modes for the weak matrix, and whether the SatPL
as-of frames match the live σ at `constant`.
