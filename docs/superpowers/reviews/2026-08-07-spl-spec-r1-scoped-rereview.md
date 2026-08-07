# `spl` spec — R1 scoped re-review (opus)

**Date**: 2026-08-07
**Artifact reviewed**: `docs/superpowers/specs/2026-08-07-saturating-power-law-design.md` @ `cddd92d`
**Prior review**: `2026-08-07-spl-spec-r0-architect-review.md` @ `9849626` (NOT READY, 4C/7I)
**Folds under review**: `git diff 9849626..cddd92d` — five folds
**Reviewer**: independent opus agent (author ≠ reviewer)

Scope: verdict each R0 finding; audit the rebuilt §3 as new work; find defects
introduced by the folds. Build gate and `tools/analyze_spl.py` declared
machine-verified up front.

Persisted verbatim, unedited, **before** any fold. Do not amend this file.

---

```
VERDICT: still not ready — but scoped: §§1–2 and §§4–10 are ready to plan.
         §3 alone blocks (1 Critical, 5 Important). The registration surface,
         which was the bulk of R0's findings, is fully folded.
```

## R0 FINDINGS

**CRITICAL 1 (ΔAIC band mislabel) — ADDRESSED.** `analyze_spl.py:298-302` now computes `best*(1 + k/n)` for k∈{1,2}, each labelled. Reproduces: ΔAIC≤1 → t₀∈{28.4, 35}; ΔAIC≤2 → {25, 28.4, 35, 40}; `pl` outside both. Spec §3.3 table matches exactly. "$19T floor / 30× range" deleted and explicitly retracted (§3.3 L188-192).

**CRITICAL 2 (resqr allow-list) — ADDRESSED.** §4.1 rewrites the step-19 rationale, decides non-flagship, names both mirrored registries for later promotion, documents the σ-mode silent no-op as out-of-scope, and drops the rating High→Medium (§9 row agrees). Verified `RESQR_FLAGSHIP_MODELS` = 15 entries excluding logi/plo/sexp (`tools/build_bm_model.py:55-60`); `EXPECTED_FLAGSHIP_MODELS` mirrors it (`btc_web/test_resqr_build.py:29-34`). §6 row +C records the deliberate no-change.

**CRITICAL 3 (autocorrelation) — PARTIAL.** §3.2 exists, reports DW/ρ at 1/30/365 d, retracts the scalar n_eff correctly (the mean-of-a-stationary-series objection is right), and adds two corrections. The *framing* is fixed. The *inference* is not — both corrections have defects that reverse §3.2's headline. See NEW-C1/I1–I4.

**CRITICAL 4 (mi-spl positional) — ADDRESSED.** §6.2 is a dedicated subsection with the correct SSOT hierarchy. Verified: `btc_web/layout/model_info/__init__.py:60` is SSOT (so stated at `callbacks/routing.py:535`), `routing.py:537` is the mirror, both lists 29 entries ending `mi-citadel`, and `routing.py:730` is `_MODEL_INFO_ITEMS[n - 1]`. So `mi-spl` = item 30, and `/mi.23`–`/mi.29` are preserved. Step 22 adds the three-way sync test. §9 risk row present.

**IMPORTANT 1 (§4 bounds ≠ §3's bounds) — ADDRESSED.** `analyze_spl.py:251-278` fits under exactly §4's bounds; `T0_LO=1.0` so the previously unsearched t₀∈[1, 16.86) region is now covered; bound activity reported as `none (interior)`; §3.1 states it.

**IMPORTANT 2 (derived-L constraint mechanism) — ADDRESSED.** §4 specifies "penalty returning `1e6 + 1e3·(distance outside)`"; `analyze_spl.py:253-258` implements precisely that.

**IMPORTANT 3 ("bounded below, not above" contradiction) — ADDRESSED.** Heading replaced; §3.3 states steep-below / flat-above. (A milder new self-contradiction was introduced — see NEW-M4.)

**IMPORTANT 4 (percentages vs own asymptote) — ADDRESSED.** §3.4 measures against the fitted `pl` (`analyze_spl.py:304-312`); +1.21/−0.80/−3.42/−13.43/−48.75/−85.39 all reproduce; the pivot is named; retraction explicit.

**IMPORTANT 5 (heatmap pill overflow) — ADDRESSED.** §6.3; `layout/heatmap.py:229-231` quoted verbatim and verified; `style.css` confirmed to contain zero pill/btn-group rules.

**IMPORTANT 6 ($124,642 is a bound value) — ADDRESSED.** §3.5 reports only RMSE 0.4467 / ΔBIC +4832 (both reproduce), states lower-bound-by-construction, drops the dollar figure, softens "log-time or nothing".

**IMPORTANT 7 (fit_spl.py must use the class mask) — ADDRESSED.** `t >= T_MIN` (1.0) stated in §4, applied at `analyze_spl.py:66`, repeated in §4.1 step 21, and in §5's `__init__`.

**CHECKLIST GAPS — ADDRESSED**, with two nits:
- steps 2/3/4/5/6/8/9/10/22/23-25 all have explicit §6 rows. `dash_style = "longdash"` verified safe — `test_models.py:812` whitelists it and duplicates already ship (`_simple.py:367`, `:420`).
- +A / +B / +C all present (`api.py:154-156`, `scanner.py:22-25` verified).
- NIT: §6's table has no row for step 11 or step 26 while claiming "every step below is explicit". 11 is a documented no-op; 26 is covered by §5's `short_name` comment + §9. Cosmetic, but R0 counted both as covered previously.
- NIT NOT FOLDED: R0's `CLAUDE.md:262-276` third doc registry appears nowhere in the spec.

---

## NEW FINDINGS IN THE REBUILT §3

**CRITICAL — the GLS result reverses under Prais–Winsten, and §3.2's one positive result rests on it.**
Measured, same data, same ρ, sole difference = retaining observation 1 at weight √(1−ρ²)=0.0587:

| | pl exponent | move vs OLS |
|---|---|---|
| OLS | 5.0736 | — |
| iterated Cochrane–Orcutt (= `gls_refit`) | 4.7782 | **−5.82%** |
| iterated Prais–Winsten | 5.0987 | **+0.49%** |

spl PW-analog: β = **5.2367**, t₀ = 17.68 (CO gave 5.1218 / 18.53). Consequences:
- §3.2 L115 "`pl` exponent −5.82%" and L135 "the OLS/GLS gap of 5.8% is real" are artifacts of discarding one observation out of 5792.
- The central rhetorical argument — L127-131/L151, "*which* estimate refused to move: pl −5.82% vs spl β +0.61%" — **reverses**: under PW, pl moves +0.49% and spl's β moves +2.9%, i.e. spl's β moves ~6× *more*. The "a model fitting noise would take that freedom; it declines to" inference does not survive.
- L152-153's app-wide warning ("the published ≈5.07 would read ≈4.78 under an AR(1) correction") would propagate a wrong number to the rest of the app.

Root cause: ρ̂ = 0.9983 is the near-unit-root regime, where feasible AR(1) GLS is ill-conditioned — the level information collapses onto the observation CO throws away. §3.2 L102-104 calls near-differencing "a much harder test"; it is an *ill-conditioned* one. **Fix:** report PW (or both), state the near-unit-root caveat, and either drop the asymmetry argument or rebuild it on a correction stable to this choice.

**IMPORTANT — `block_bootstrap` systematically excludes the recent data that identifies t₀.**
`rng.integers(0, n-blk)` (`analyze_spl.py:172`) yields starts 0…4330 for blk=1461, n=5792 (`high` is exclusive → start `n−blk` never drawn). Measured inclusion counts: interior max 1461; **final index 5791 = 0, never sampled**; t=16.70 yr = 57 (rel 0.039); t=16.07 yr = 289 (rel 0.198); t=15.27 yr = 579 (rel 0.396). MBB edge effect + off-by-one downweights the last ~1.5 yr by 2.5×–∞. This biases t₀ upward toward `pl` — 50% of replicates land at t₀ > 1000 yr. **Fix:** circular block bootstrap (wrap mod n), or at minimum `rng.integers(0, n-blk+1)`; then re-derive §3.2/§3.3's bootstrap rows.

**IMPORTANT — it is a *pairs* bootstrap on a fixed design; a residual bootstrap is the right tool.**
Answering the brief directly: **sorting does not invalidate it.** The estimator is least squares on (t, price) pairs and is permutation-invariant — measured |sorted − unsorted| exponent = **4.4×10⁻¹⁵** (summation-order noise). But that is exactly the problem: nothing in the replicate fit uses observation order, so the *block* structure does no work on the error process. It only randomizes the *design* — which stretches of t appear and how often — and t here is a fixed exhaustive grid, not a random sample. The correct instrument for propagating an autocorrelated error process into regression-parameter uncertainty on a fixed design is a **residual block bootstrap** (resample residual blocks, add back to fitted values, refit). **Fix:** switch to it, or relabel the current output as sub-sample/design sensitivity, not a confidence interval.

**IMPORTANT — 4-year blocks are defensible in scale but not at this n: only 4 blocks per replicate.**
nb = ceil(5792/1461) = **4**. MBB consistency requires n/ℓ → ∞. The ACF supports the *scale* (ρ(30d)=0.90 ⇒ correlation time ~1/(1−ρ) ≈ 1.5 yr; ρ(365d)=−0.22 supports a cycle multiple), but 5–95% quantiles built from 4-unit draws have very low resolution. **Fix:** show block-length sensitivity (1/2/4 yr), or label the interval indicative.

**IMPORTANT — "its own interval is ±10%" (§3.2 L134-135) understates the measured interval.**
Measured 5–95%: pl exponent 4.670–5.786 = **−8.0% / +14.0%**; spl β 4.807–7.871 = **−5.6% / +54.6%**. **Fix:** quote both intervals.

**IMPORTANT — §3.2's table is not reproducible by running the tool, and §3.2 and §3.3 quote two different runs of the same quantity.**
`block_bootstrap` default is `n_boot=200` (`analyze_spl.py:158`) and `main()` uses the default; §3.2 L105 says "400 resamples". Measured:

| | pl med | spl β med | spl β 5–95 | t₀ med | t₀ 5–95 |
|---|---|---|---|---|---|
| n_boot=400 | **5.088** | **5.177** | **4.824–7.868** | **794.8** | **11.630–11910.3** |
| n_boot=200 (shipped) | 5.076 | 5.158 | 4.807–7.871 | 778.9 | 11.859–11761.0 |

§3.2's table is the 400 run; **§3.3's bootstrap row ("11.9 – 11,761") is the 200 run.** So the spec states two different t₀ intervals (11.6–11,910 vs 11.9–11,761) for one quantity, and the header's "re-run the tool" reproduces §3.3, not §3.2. Downstream: "43×" = 794.8/18.53; the shipped tool gives 42×. **Fix:** set `n_boot=400` in `main()` (or restate §3.2 at 200) and derive both rows from one run.

**IMPORTANT — §3.2's `spl L` bootstrap row is not produced by the tool at all.** `block_bootstrap` returns only `(expo, betas, t0s)` (`analyze_spl.py:183`); no L distribution is computed or printed. "bootstrap median 3.7×10¹¹, 5–95% 1 – 1.8×10¹⁴" is **UNVERIFIED** and contradicts the preamble's "every number here is generated by `tools/analyze_spl.py`". **Fix:** compute it or delete the row.

**MINOR — no convergence reporting in `gls_refit`.** Measured it *does* converge (linear CO at iteration 8/30, nonlinear CO at 4/30), so the numbers stand — but a silent fall-through past 30 iterations would print a non-converged iterate as a result. Add a flag.

**MINOR — `gls_refit`'s spl "OLS" fit is a second, independent optimizer.** Nelder-Mead from hardcoded seed `[-1.178, 5.091, log10(28.31)]` (`:140`), not the DE fit of §3.1 (they agree: 28.31 vs 28.314). Both this seed and `block_bootstrap`'s `seed_th` are pinned to a past snapshot and go stale as the CSV grows; on a flat ridge a stale seed makes Nelder-Mead stall rather than converge, which itself inflates the reported t₀ spread.

**MINOR — "Identical to six decimal places" (§3.2 L72) is false.** ρ = 0.998127 vs 0.998126 differ *in* the sixth decimal. Identical to five.

**MINOR — "removes 0.00007% of the autocorrelation"** is Δρ/ρ, not a share of autocorrelation removed. True as written; the phrasing invites the stronger reading.

### Overshoot in the other direction (the brief's last bullet)

**MINOR (NEW-M4) — "Corrected, the data do not constrain the ceiling at all" (§3.3 L190) is an overcorrection that contradicts the next paragraph** ("SSE rises steeply below t₀≈25 (+3.33 at t₀=20)"). At face value ΔSSE +3.33 on SSE_best 502.24 is ΔAIC ≈ +38 for t₀=20 — a large, real degradation. The defensible statement is the one the spec already makes two sentences later: the lower edge is a genuine feature, the upper edge is threshold placement. **Drop "at all".**

**MINOR — §3.3's bootstrap row marks `pl` "inside".** `pl` is t₀→∞ and the 95th percentile is finite (11,761 yr), so `pl` is strictly *outside*. Substantively defensible (50% of replicates exceed 1000 yr, where spl is numerically `pl`) but it is not computed by the tool and the table presents it as if it were.

---

## NEW DEFECTS FROM THE FOLDS

- **§7 item 3 (line 448) "Both criteria at both sample sizes"** — stale from the retracted n_eff era; only n=5792 exists now. This is a card-content instruction, so it ships. Fix: "both criteria at n=5792, with the independence caveat".
- **§7 item 2 (line 446) "The ceiling in particular spans ~1000x"** — ~1000× is the **t₀** spread (991.7×). §3.2's own L row spans 1 → 1.8×10¹⁴ ($T). Off by 11 orders against the spec's own table.
- **§3.2 t₀ interval (11.6–11,910) vs §3.3 t₀ interval (11.9–11,761)** — same quantity, two runs. See NEW-I5.
- **`tools/spl_spec_build_gate.py:53-59`** — gate 3 tests `spl − pl = −log10(1+(t/t₀)^β)`, labelled "§3.4 identity" in the comment and "§3.3 identity" in the print. **The spec no longer states that identity anywhere** (pre-fold §3.3 had it; the rebuilt §3.4 measures against the fitted `pl` instead). The gate guards a claim the spec doesn't make and cites two different sections for it.
- **§3.4's "+0.029" log-log quadratic** reproduces (I measure +0.02857) but `main()` no longer computes it — the §3 rebuild dropped it. Another number outside the "everything is generated" guarantee.
- **§6 step table drops rows 11 and 26** (both were covered pre-fold). Cosmetic; 11 is a no-op, 26 is covered in §5/§9.
- **`docs/architecture.md:339-351`** (§6 row +A) — the registry table actually starts at `:342` and runs past `:355`. Inherited from R0, not introduced, but still wrong.
- **Reproducibility**: n=5792 requires 22 **uncommitted** rows in `BitcoinPricesDaily.csv` (HEAD 5780 lines, worktree 5802). A clean checkout of `cddd92d` yields n=5770 and reproduces none of §3. Worth one line in the preamble.
- **NIT** — §4 writes the β bound as "(0.01, 20]"; `differential_evolution` bounds are closed on both ends.

---

## STATISTICAL METHOD AUDIT

- **`gls_refit()` : flawed as used.** The implementation is correct — proper Cochrane–Orcutt for the linear branch, and the nonlinear branch quasi-differences the *residual* rather than the regressor, which is the legitimate generalization for a nonlinear mean function. It converges (8 and 4 iterations). **But dropping the first observation is decisive here, not immaterial:** at ρ̂=0.998 the CO and PW slopes differ by 0.32, and PW says the exponent barely moves. Every §3.2 conclusion drawn from OLS→GLS *movement* is an artifact of that choice, in the regime where FGLS is known to be unreliable.
- **`block_bootstrap()` : flawed.** Sorting is a harmless no-op (4×10⁻¹⁵) — but that fact exposes that the block structure does nothing to the error process. It is a pairs bootstrap on a fixed exhaustive design (should be a residual block bootstrap); an off-by-one plus the MBB edge effect drives the final observation's inclusion weight to zero and biases t₀ toward `pl`; 4-year blocks give only 4 blocks per replicate; it doesn't produce the L distribution the spec quotes; and `main()` runs 200 while §3.2 reports 400.
- **The verbal claims:**
  - "spl removes 0.039% of the variance and 0.00007% of the autocorrelation" — **SUPPORTED**, reproduces exactly (0.0394% / +0.00007%, Δρ = −7.03e-07). The strongest claim in §3.
  - "the two corrections disagree by 43×" — arithmetically consistent with §3.2's own 400-run numbers (42× from the shipped tool), but **both sides are compromised**: the GLS side flips under PW, the bootstrap side is edge-biased. Overstated as "the sharpest fact in this section". The *conclusion* is independently carried by §3.3's profile, which needs no resampling — rest it there.
  - "the exponent is the thing this analysis establishes" — **PARTIALLY supported**. Point estimates do cluster near 5 and `cddd92d` correctly concedes the six are not independent, but the supporting asymmetry argument reverses under PW and the "±10%" interval understates the measured spl β range (−5.6%/+54.6%).
  - "when two corrections disagree that violently, that parameter is assumed not measured" — **conclusion sound, evidence weak**. Re-base it on the flat SSE ridge in §3.3.

---

## CLEARED

- Every number in §3.1, §3.3's profile table, §3.3's two ΔAIC rows, §3.4's deviation table, and §3.5 reproduces **exactly** against the tool.
- ΔAIC criterion arithmetic is now correct (`k/n`, not `2/n`).
- §4's bounds and the derived-L penalty match `analyze_spl.py:251-278`; the fit is interior on all three.
- `T_MIN = 1.0` mask is consistent across tool, §4, §4.1 step 21, and §5's constructor.
- Code citations verified correct: `build_bm_model.py:55-60` (15 flagships, no logi/plo/sexp) and `:96`; `refit_all_ppl.py:59`; `fit_shrinking_sigma.py:219` and `:248`; `_helpers.py:99-101`; `test_resqr_build.py:29-34`; `routing.py:535/537/730`; `layout/model_info/__init__.py:60`; `heatmap.py:229-231` (verbatim); `scanner.py:22-25`; `api.py:154-156`; `update_prices.py:142`.
- `style.css` genuinely has no pill/btn-group rule — §6.3's "verified" is true.
- `dash_style = "longdash"` is valid and duplicates are already normal.
- Build gate passes: RMSE 0.294470 matches §3.1; logaddexp finite; identity holds to 2.5e-15.
- Both R0 Phase-2 findings folded: §8 specifies the per-request instance (backed by §5's optional constructor args) and the params-dict + `tab_defaults` mirror.
- R0's minors folded: `updatemode` de-escalated, RMSE rounded to 4 dp with rationale, `_HM_PILL_LABELS["spl"] = "SatPL"` chosen, quadratic caveat retained.

**Shortest path to green:** §3 needs one more pass — add Prais–Winsten (or drop the movement-based argument), fix the bootstrap's start range and switch to residual blocks, unify `n_boot`, and repair §7's two card-bound claims. Nothing outside §3 blocks.
