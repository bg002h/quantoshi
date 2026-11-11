# Percentile cycle fit — design

**Date:** 2026-11-10 · **Status:** r1 folded (review `docs/superpowers/reviews/2026-11-10-pctile-cycle-spec-review-r1.md`) · **Tab:** 1 (Percentile pill)

## Intent

The operator made a one-off study figure (`tools/render_percentile_sinusoid_artifacts.py`,
`docs/artifacts/percentile-sinusoid-calendar-censored-20yr.png`) that fits
`y = c + A·cos(2πf·t + φ)` to the percentile of the BTC close within the QR
fan, extrapolates it 20 years, and labels each fitted peak/trough with its
date, percentile and the price the fan puts there. They want that concept —
*a fitted and extrapolated path through percentile-space, translated back into
price, for the selected model* — as a live feature, done better than the
static figure, with an honest statistical judgment, deployed to production for
testing.

Operator decisions (2026-11-10):

| question | answer |
|---|---|
| placement | **Upgrade the Percentile pill** (fit + extrapolation behind a checkbox; existing history lines stay) |
| uncertainty | **Widening band** from refitting on resampled data, fanning out as period/phase uncertainty compounds |
| models | **One at a time, picker** (default QR) |
| reviews | Opus for every review subagent |
| the fit | "a **single calendar-time sinusoid** to the percentile (or whatever element of the appropriate basis set for the model is)" |
| tight models | "sinusoid may not be best for tight models like BM … there may be an element of a basis set that does work better" |

So the fitted form is exactly **one basis element**, `c + A·cos(2πf·g(t) + φ)`,
fitted to the picked model's own percentile series (the level in that model's
quantile basis at which each daily close sits, at the view's σ-mode). `g` is
the element's time coordinate from the repo's oscillator dictionary
(GreedyModel `_BASIS`): calendar (`g = t`) or log-time (`g = ln t`). Each model
is fit against its own fan; nothing is pooled.

## Statistical judgment (measured 2026-11-10 on the live `model_data.pkl`)

Sources: `tools/pctile_cycle_explore.py`, `tools/pctile_cycle_basis_explore.py`,
and the r1 reviewer's scratch checks (reproduced the explore numbers exactly).

* **Skill is modest, and the naive hindcast overstates it.** Scoring a
  rolling-origin hindcast against *today's* fan leaks the future into the
  percentile series (the fan saw the later data). With QR's **as-of fans**
  (Time Machine grid, fitted only on data to each frame), the 2-yr hindcast:
  wave beats flat in **53 %** of 17 overlapping windows (≈ 4 independent),
  RMSE **21.8 vs 23.3 pp** — versus a leaky 76 % / 21.4 vs 27.0. At 4 yr: 69 %,
  22.4 vs 24.3. Peak timing is mostly robust (2019/2021/2023 cutoffs move < 1
  month). Verdict: **a real but small edge for QR; worth showing only with the
  honest number next to it.**
* **Daily points overstate the evidence.** Residual lag-1 autocorrelation 0.996
  (n_eff ≈ 12); residual ACF ≈ 0 at 365 d. Hence 1-yr block resampling.
* **The refit band is narrow by nature.** It is uncertainty in *where the wave
  is*; the actual percentile sat inside it only **31 %** of days (QR). Read as a
  range for the percentile or price it would mislead. Hence a second, outer
  **predictive** band (below) and explicit labels.
* **Basis element by model** (2-yr hindcast vs flat against today's fan):
  calendar 2–8 yr is best in- *and* out-of-sample for QR / PL / SatPL. Damped
  and log-time elements gain in-sample and lose out-of-sample. For BM and the
  oscillator models (LPPL, HybPPL, EPPL, PCA, Greedy) and EF, **no single
  element beats a flat forecast out-of-sample**: their fans already carry the
  cycle. Best in-sample for BM: calendar ~1.85 yr (R² 0.12 vs 0.04). The
  operator's hunch is right in-sample and wrong out-of-sample; the view lets
  them try the elements and *shows the out-of-sample verdict*.
* **Amplitude does not decay in percentile units** (`docs/artifacts/README.md`),
  so no damping element is offered.
* **Clamped series.** At resqr, PL/SatPL percentiles pin to the fan's edge on
  ~16 % of days (flat resqr tails); least squares on a clamped series biases
  c and A. Reported when > 5 %.

## Goals

1. With **Cycle fit** on, for one picked model: actual percentile history, the
   fitted element in-sample, its extrapolation to a chosen horizon, an inner
   **fit band** (where the wave is) and an outer **likely band** (where the
   percentile is likely to be).
2. Turn markers on the central wave; forecast turns carry 10–90 % timing bars
   and hover with date, percentile and fan price; the **next** forecast turns get
   text labels (density rule below).
3. Hover on the wave: date, fitted percentile + price, fit band, likely band
   with its price range.
4. A summary under the chart: model, element, period (10–90 % range), amplitude,
   R², hindcast (honest where possible, labelled where not), band coverage.
5. A weak fit is said plainly and **not extrapolated**.
6. State survives share links; links made before the feature restore with it off.

## Non-goals

Price-view overlay; several models at once; multi-term or damped elements; U₁;
MC interaction; deep-link segments for the fit settings; changing the
Percentile view when Cycle fit is off.

## Anti-goals

* A forecast with no visible uncertainty, or a band labelled as if it were a
  range for the percentile/price when it is not.
* A skill number with look-ahead presented as out-of-sample.
* Non-deterministic bands (seeds fixed); uncached server cost (one fit per
  model × σ × censor × element × data version per worker).
* Shifting any existing share-link index; changing what an old link shows.

## Design

### Controls (span `bub-pctile-ctl-wrap`, Percentile mode only)

| id | component | options | default |
|---|---|---|---|
| `bub-pctile-fit` | `dcc.Checklist`, inline-block | `fit` "Cycle fit", `censor` "skip 2019–20" | `["fit"]` |
| `bub-pctile-fit-model` | `dcc.Dropdown` | picker list below | `"qr"` |
| `bub-pctile-fit-basis` | `dcc.Dropdown` | `cal` "calendar 2–8 yr", `calw` "calendar 0.75–16 yr", `log` "log-time" | `"cal"` |
| `bub-pctile-fit-horizon` | `dcc.Dropdown` | 10 / 20 / 30 yr | `20` |

**Picker list:** `qr, bub, pl, spl, lppl, hybppl, eppl, pca, grdy, ef` — the
**same keys `bub-model-show` uses** (masters, not the ticker's config variants),
filtered at layout time to registered quantized models; labels and every
on-chart name use the model's `legend_name`, so a picked model and its line in
the model list share one name and one series (r1 I-3).

**Callback coercion (share links can carry anything):** model not in the picker
list or not registered → `qr`; basis not in the table → `cal`; horizon not in
{10, 20, 30} → 20; checklist `None` → `[]`.

**Old links (r1 I-1):** `_PREDATING_LINK_VALUES["bub-pctile-fit:value"] = []`,
unconditional — `_fill_predating_fields` gains `requires=None`. A link that
carries no value for the field predates it, and restores with the fit off,
exactly as its sender saw it.

**Registry:** pinning the new defaults fingerprint evicts the oldest entry
(`faf00e93`, live ~1 day in 2026-04); accepted (r1 M-1).

The pill row gets `flexWrap: "wrap"` so the controls drop to a second line on
a phone.

### Math (`btc_web/pctile_cycle.py`, numpy + scipy only)

* **Series:** the picked model's percentile of every daily close with t ≥ 1 yr,
  full record (never the x-window); `censor` drops 2019-04-15 – 2020-02-15.
* **Element fit:** least squares of `c + a·cos(2πf·g) + b·sin(2πf·g)`; f scanned
  on a 400-point grid over the element's range (cal 1/8–1/2, calw 1/16–1/0.75
  cycles/yr; log ω = 2πf ∈ [1, 30]) and refined by bounded scalar minimisation
  within one grid step. `at_limit` = refined f within one grid step of either
  end (the "period" is then an artefact of the window).
* **Bootstrap:** circular moving-block on the weekly residuals, block 52, B = 150,
  fixed seed; each replicate refit with the same element (200-point grid).
* **Fit band:** pointwise 10/90 of replicate curves.
* **Likely band (r1 C-2):** pointwise 10/90 of *replicate curve + one
  block-resampled residual path* (weekly, circular 52-week blocks, its own fixed
  seed), clipped to [0, 100]. Computed once on a fixed weekly grid from the
  series start to last + 30 yr, so zoom and horizon never re-draw it.
  **Coverage** = share of actual daily percentiles inside each band (in-sample),
  reported in the summary.
* **Turns:** analytic in g (peak 2πf·g + φ = 2πk; trough π + 2πk). **Timing by
  cycle index (r1 M-2):** the j-th same-kind turn after the last observation in
  each replicate; 10/90 → bar, `array`/`arrayminus` clamped ≥ 0.
* **Hindcast:** cutoffs every 0.5 yr from last−10 to last−2, ≥ 5 yr history,
  weekly fit, 2-yr horizon, flat = in-sample mean. **For QR it is honest:** the
  percentile series at each cutoff is recomputed against the Time Machine as-of
  QR fan of the last frame ≤ cutoff (QR is σ-invariant). Other models are scored
  against today's fan and labelled **"optimistic (today's fan)"**. Report n,
  "overlapping", win share, mean RMSE wave vs flat.
* **Weak** = R² < 0.20, **or** hindcast mean RMSE wave ≥ flat, **or** `at_limit`.
* **Percentile → price:** sorted (monotone-rearranged) log10 fan, linear interp
  in quantile, clipped to the outer quantiles.

### Figure (Cycle fit on)

* **x-range:** forecast shown iff the fit is not weak and
  `yr_to_t(xmax) ≥ today_t` (the capped default 2027 qualifies; 2026 = 1 Jan
  does not). Then axis range and ticks run to today + horizon while the data
  mask stays capped at today. No callback writes `bub-xrange`.
* **Traces:** likely band (model colour, lightest), fit band (model colour,
  light), other active models' lines dimmed, picked model's actual percentile
  (grey), wave in-sample (solid) and forecast (dashed), turn markers (forecast
  ones with timing bars). Legend: "likely range (10–90 %)", "fit uncertainty
  (10–90 % of refits)".
* **Weak:** in-sample wave only (dotted, model colour), fit band in-sample,
  no forecast segment, no forecast turns, no labels, no axis extension, plus a
  short in-chart warning ("weak cycle in ‹model›'s percentile — no forecast").
* **Labels (r1 I-4):** text labels on the next N peaks and N troughs after the
  last observation, N = 1 on a log x-axis and 3 on a linear one; all other turns
  carry hover only. Annotation x is log10(t) on a log axis.
* **Hover:** wave rows (after the shared date prepend at `customdata[0]`):
  fit band lo/hi, likely band lo/hi, price at fit, price at likely lo/hi.
* **Summary + footnote:** written into `layout.meta.cycle_info` (list of lines)
  and shown by a clientside callback in a wrapping `html.Div`
  (`bub-pctile-info`) under the graph (r1 M-9), so every writer of the figure
  updates it. The title's subtitle carries a short caveat for PNG exports.
  Wording: R² reads "R² (2019–20 excluded)" when censored (M-10); the band
  footnote says the fit band is narrowest at turns and timing lives in the bars
  (M-3); clamped share shown when > 5 % (M-4).
* **Payload:** weekly grid for all new traces (r1 M-7).
* Colours are `colors.py` constants; the weak box is not the rich-zone amber
  (N-2).

### Wiring

* `update_bub_pctile` gains four Inputs and passes `cycle_fit`, `cycle_censor`,
  `cycle_model`, `cycle_basis`, `cycle_horizon` (coerced as above) in `p`.
* `VIEW_MODES["percentile"].ctl = "bub-pctile-ctl-wrap"`: `STYLE_OUTPUT_IDS` grows
  to 15 with the new id **between** `bub-cagr-fwd-wrap` and `bub-occ-ctl-wrap`
  in all three emitters; `deep_link_bub_view`'s no-update tuple grows 20 → 21;
  docstring/test counts follow (r1 N-1).
* Snapshot: append `bub-pctile-fit`, `bub-pctile-fit-model`,
  `bub-pctile-fit-basis`, `bub-pctile-fit-horizon` at indices 348–351;
  `_CHECKLIST_OPTIONS["bub-pctile-fit"] = ["fit", "censor"]`; defaults in
  `SNAPSHOT_DEFAULTS`; ids in `_TAB_CONTROLS["bubble"]`; fingerprint pinned
  before and after.

## Left to the builder

Typography, offsets, opacity constants, test granularity, exact info-div styling.
