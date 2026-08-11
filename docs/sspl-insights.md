# sSPL (support-phase Saturating Power Law) — trial insights

**Status:** exploratory trial, 2026-08-08. Never committed to the repo as code;
scripts lived in the session scratchpad (`spl_support.py`, `spl_two_cluster_fig.py`,
`spl_floor_v3_fig.py`) and are regenerable. The summary-level findings are in
`docs/superpowers/followups.md` **F-9**; this file is the *undocumented* detail —
mostly the methodological traps that cost time and the numbers behind the
conclusion. Data window: prices through **2026-08-06** (n=5,865), origin
2009-07-25.

sSPL = fit `spl` (`price = L/(1+(t/t₀)^−β)`) to **support-phase prices only**,
omitting bubble-phase days (not interpolating them). Irregularly spaced by design.

---

## Bottom line

**The support floor does not saturate. It is a power law, and it looks like one
more emphatically the cleaner you make the data.** On support-only data `spl`
collapses onto `pl`: β ≈ the pl slope, L pins at whatever cap is imposed
(i.e. the data want L → ∞), and the likelihood-ratio test goes *negative*. This
is the opposite of the full-history fit, where `spl` rejects a pure power law —
so the apparent saturation lives entirely in the bubble phases, not the floor.

---

## The methodological traps (the valuable part)

### 1. Per-segment slope is meaningless — the collapsed lever arm
Slope = Δlog₁₀(p) / Δlog₁₀(t). Support segments are short in **log-t**, so the
denominator collapses and any price wiggle produces a wild "exponent." Measured
individual-segment exponents: **3.54, 5.72, −3.96** — the −3.96 is *negative*,
which no power law has. A later segment spanning 0.013 decades turns an ordinary
6-month price move into a slope of ±6. **Never compare per-segment slopes.**
Corollary artifact: an early run found a "+12.5 slope-units/decade trend,
p=0.015" that looked like significant acceleration and was pure log-t
compression — nothing.

### 2. The right test is a per-segment LEVEL offset against a pooled fit
Fit one law to the pooled support data, then check each segment's **mean
residual**. Level offsets preserve the lever arm; slopes destroy it. Segment
residuals vs the pooled fit: `+0.10, +0.00, −0.067, +0.33*, +0.072`
(*=2019 sliver, unreliable). **Non-monotone, and the most recent is positive** —
no saturation signature (which would be a monotone decline).

### 3. Starting a series at its own minimum guarantees a positive slope
This is selection-on-the-outcome and is statistically invalid. A start-date
sweep across the 2026 bottom swung the individual exponent from **+10.01
(June 25 start) to −2.79 (July 13 start)** with no change in data. Note the
max wasn't even at the 6/30 low — so "start at the bottom to maximise the
positive slope" doesn't even achieve its goal. **What stayed stable across every
start date: pooled exponent 4.93–4.99, pooled RMSE 0.081–0.083, and the recent
floor's offset above the early extrapolation at 1.46–1.48×.** Trust the pooled
columns, never the per-cluster slope.

### 4. Phase classification needs slope AND level, not slope alone
Plateau is *parallel* to support (same slope ≈ 5.12) but offset **above** it
(+0.39 to +0.68 log₁₀ = 2.5–4.8× the line). Classifying by slope alone puts
plateau at slope≈0 and finds ~0.1% of it — wrong. Correct rule:
`support = |slope − 5.1215| ≤ 0.5 AND offset ≤ 0.02`. Phase split (plateau
excluded from the fit): **support 1,167 (19.9%), plateau 743 (12.7%),
exp-up 2,024 (34.5%), exp-down 1,931 (32.9%)**.

### 5. The selection is *nearly* circular — but real information survives
The BM support line is a pure straight line in log-log (verified to 5.6e-17),
and support slope 5.1215 ≈ pl slope 5.0618 ≈ spl β. So "select days on the
support line" ≈ "select days on the power law." **But** observed support-phase
prices scatter around the line: sd **0.115** log₁₀, spanning 0.75×–3.40×, with
**35.5% below** the line. So real data survives selection; the result isn't
purely manufactured. The clean way to break the circularity is a **drawdown**
criterion (days within X% of a recent low), which references no straight line —
and it *agreed* (see below).

### 6. Two selection rules got mixed
Final datasets combined BM-composite-phase (early clusters) with
drawdown-from-Feb-2026-low (recent cluster) because BM classified **zero** 2026
days as support (the drought). They agreed — combining *improved* RMSE — which
is itself evidence. The fully clean version would use one criterion throughout.

---

## The numbers

| dataset | n | log-t dec | pl slope | RMSE | spl β | L | LRT (crit 2.706) |
|---|---|---|---|---|---|---|---|
| all price history | 5,865 | 1.24 | 5.0618 | 0.2940 | 5.1033 | $14.8T | **+13.59 REJECTS** |
| BM support only | 1,167 | 1.15 | 5.0640 | 0.1142 | 5.0642 | cap (→∞) | −0.48 |
| + Feb-2026 drawdown | 1,346 | 1.24 | 5.1077 | 0.1103 | 5.1081 | cap | −1.20 |
| − 2017-04…2025-12 | 1,095 | 1.24 | 5.0449 | 0.0991 | 5.0452 | cap | −1.80 |
| − pre-2011-06 | 1,026 | 0.86 | 5.1153 | 0.0918 | 5.1158 | cap | −1.67 |
| clusters 1+2, mid-Jun c3 | 900 | 0.86 | 4.9734 | 0.0821 | 4.9737 | cap | −0.75 |

**Two facts hold across every floor variant (seven of them):**
- **L pins at the cap.** Remove the cap and β → the pl slope exactly and LRT → −0.000. The floor doesn't fail to find a ceiling — it drives L → ∞.
- **LRT is negative.** A nested model can't fit worse at a true optimum, so negative LRT is the signature that the optimiser wanted `pl` and the bound stopped it short. Report it, don't clip it to zero.

**The noise-floor inversion — the strongest single argument.** RMSE falls
0.2940 → 0.1142 → 0.1103 → 0.0991 → 0.0918 as the floor data get cleaner, and
the LRT gets *more* negative (+13.59 → −0.48 → −1.20 → −1.80 → −1.67). If
saturation were real and merely buried in bubble noise, cleaning the data would
**sharpen** it. It inverts instead. We looked with a 2.57–3.2× tighter
instrument and the signal went the other way.

---

## The extrapolation test (out-of-sample)

Fit the floor on **2011–2017 only** (clusters 1+2, slope ≈ 4.875), extrapolate
**nine years**, predict today's floor ≈ **$44k**. Actual 2026 floor is
**1.47–1.68× higher** (mean residual +0.17 to +0.23, sd ~0.01–0.05 across the
recent cluster). **Saturation predicts the opposite sign** — a bending floor
lands *below* a long extrapolation; this lands well above. Adding the recent
cluster also *raises* the exponent (4.875 → 4.97–5.12): the floor is steepening
slightly, not rolling over.

---

## sSPL as a point prediction (with the right caveat)

sSPL fitted to support-only data predicts **$61,268** for 2026-08-06 vs actual
**$64,294** (1.05×, +0.18σ; 1σ band ×/÷1.30 ≈ $47k–$80k). That's a genuine
~3-year out-of-sample extrapolation (support data ended 2023-06-19) landing
within 5%. **Caveat that must ride with any such number:** sSPL estimates the
**floor** (lower envelope), not the expected price — 65% of support-phase days
sit *above* the line even during support phases. It is a floor, not a central
forecast. (Because L pins at the cap, sSPL and the plain support power law give
the same number to 0.1% — the ceiling parameter contributes nothing.)

---

## The clusters, and the drought

Five raw BM-support segments, but only three are usable:

| # | dates | n | usable? |
|---|---|---|---|
| 1 | 2010-07 … 2010-09 | 69 | marginal (illiquid, $0.05–0.30) |
| 2 | 2011-12 … 2012-11 | 337 | yes |
| 3 | 2015-09 … 2017-02 | 510 | yes |
| 4 | 2019-05 … 2019-06 | 35 | **no** — 0.004 decades, drop it |
| 5 | 2022-11 … 2023-06 | 216 | yes (last BM-support days) |
| (drawdown) | 2026-02 … 2026-08 | 179 | hand-defined; BM saw none |

**The drought:** the last BM-support day was **2023-06-19**, then **3.13 years
with none** through 2026. Only **251 of 1,167** support days are post-2018. This
is why the 2026 floor had to be drawdown-defined, and it's the empirical hook
for F-9 (time-evolving BM): support phases are becoming rare, so a static
phase decomposition increasingly mislabels.

---

## Reproduction recipe

- Phases from `model_data.pkl`: take `bm_comp_by_n[-1]`, compute its log-log
  **slope** via `np.gradient(log10(comp), log10(t_grid))` and its **offset**
  from the support line `−1.5549 + 5.1215·log10(t)`. Map observed days onto the
  grid with `searchsorted(years_plot_bm, price_years)`.
- Fit: reuse `tools/analyze_spl.py::fit_spl` (L penalised, not boxed;
  bounds t₀∈[1,100], β∈[1,12] is ample, cap $1000T/21e6). Same optimiser as the
  committed model — don't write a second one.
- Profile curve (the identification picture): SSE with log₁₀L held fixed on a
  grid; on support data it descends monotonically to the bound and never turns
  back up (= L→∞).

---

## If this is ever revisited

The one well-posed follow-up (also noted in F-9): fit the three usable clusters
with a **common exponent + free intercepts** vs **free exponents**, and compare
on a boundary-corrected LRT. If free exponents don't win, the floor is one power
law and only its *level* drifts — a far smaller claim than a saturating or
time-evolving model, and the honest one given everything above.
