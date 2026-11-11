# Percentile cycle fit — design

**Date:** 2026-11-10 · **Status:** draft for review · **Tab:** 1 (Percentile pill)

## Intent

The operator made a one-off study figure (`tools/render_percentile_sinusoid_artifacts.py`,
`docs/artifacts/percentile-sinusoid-calendar-censored-20yr.png`) that fits
`y = c + A·cos(2πf·t + φ)` to the percentile of the BTC close within the QR
fan, extrapolates it 20 years, and labels each fitted peak/trough with its
date, percentile and the price the fan puts there. They want that concept —
*a fitted and extrapolated path through percentile-space, translated back into
price, for the selected model* — as a live feature, done better than the
static figure. They asked for statistical judgment on whether the idea is any
good, and asked that it go to production for testing.

Operator decisions (2026-11-10, verbatim choices):

| question | answer |
|---|---|
| placement | **Upgrade the Percentile pill** (fit + extrapolation behind a checkbox; existing history lines stay) |
| uncertainty | **Widening band** — central line plus a band from refitting on resampled data that fans out as period/phase uncertainty compounds |
| models | **One at a time, picker** — dropdown chooses the model (default QR) |
| reviews | Opus for every review subagent |

## Statistical judgment (measured 2026-11-10 on the live `model_data.pkl`)

Numbers below are from a scratch run against the current data (QR fan,
`resqr` σ-mode, full record t ≥ 1 yr, period searched over 2–8 yr). They are
the evidence for the design, not values to hard-code.

* **The wave is real enough to show.** Rolling-origin hindcast — fit on data up
  to a cutoff, score the next H years, cutoffs every 6 months over the last 10
  years: the wave beats a flat (in-sample mean) forecast in **68 % / 76 % / 92 %**
  of cutoffs for H = 1 / 2 / 4 yr, RMSE **20.0 / 21.4 / 22.8 pp** against
  **26.4 / 27.0 / 28.3 pp** flat. Peak timing is stable: cutoffs at 2017-06,
  2019-06, 2021-06 predicted the 2021 peak at 2021-08, 2021-11, 2021-08.
  Modest skill, not nothing — and the feature should *show* this number.
* **Daily points overstate the evidence ~460×.** Residual lag-1
  autocorrelation is 0.996 (n_eff ≈ 12 by the AR(1) formula). The residual
  ACF reaches ~0 at 365 days (0.84 / 0.54 / 0.24 / 0.005 at 30 / 90 / 180 / 365 d).
  So any uncertainty estimate that treats days as independent is wrong by an
  order of magnitude; a **moving-block bootstrap with 1-year blocks** is
  justified by that ACF.
* **Band size.** 1-yr-block bootstrap (200 refits): period 3.57 yr, sd 0.067
  (10–90 %: 3.49–3.66). Pointwise 10–90 % band width ≈ 14 pp today, 24 pp at
  +5 yr, 41 pp at +10 yr. Block length 0.5 / 2 yr changes these by < 20 %.
* **Model choice matters a lot.** R² of the wave per model: QR 0.557, PL 0.454,
  SatPL 0.459, **BM 0.044**. The Bubble Model already contains the cycle, so its
  percentile has almost none left. The view must say so instead of drawing a
  confident-looking sinusoid through noise.
* **Amplitude does not decay in percentile units** (established in
  `docs/artifacts/README.md`, every parameterisation tried). So no damping
  option: the constant-amplitude wave is the fitted model, and the band — not a
  damping term — carries the growing uncertainty.
* **What the band does not include:** model-form uncertainty (is a single
  sinusoid right at all?) and the day-to-day scatter of the percentile around
  the wave (hindcast RMSE ~21 pp). The figure must say this in words.

## Goals

1. On the Percentile view, with **Cycle fit** on, draw for one picked model:
   its actual percentile history, the fitted wave in-sample, the wave
   extrapolated to a chosen horizon, and a 10–90 % bootstrap band around it.
2. Mark every peak and trough of the central wave; in the forecast region give
   each one a label `Mon YYYY · Qnn · $price` and a horizontal bar showing the
   10–90 % range of that turn's date across the bootstrap refits.
3. Hover anywhere on the wave: date, fitted percentile, band, and the price
   the model's fan puts at the fitted percentile and at both band edges.
4. A one-line fit summary on the chart: model, period (with its 10–90 %
   range), amplitude, R², and the 2-year hindcast win rate vs. flat.
5. A clear warning when the fit is weak (R² < 0.20).
6. State survives share links (snapshot), and old links still decode.

## Non-goals

* Drawing the implied price path on the Price view (operator chose the
  Percentile pill only).
* Fitting several models at once, log-time or two-term fits, damping, U₁ as a
  fit model, MC interaction, new deep-link segments for the fit settings.
* Changing the Percentile view's existing behaviour when Cycle fit is off.

## Anti-goals (failure even if it "works")

* An extrapolation that looks like a forecast with no visible uncertainty, or a
  band that understates it (e.g. an i.i.d. residual bootstrap — n_eff ≈ 12).
* A different band on every reload or on different workers (non-deterministic
  resampling). Seeds are fixed.
* Server cost out of line with ALARA: the fit must be cached so a worker pays
  for each (model, σ-mode, censor, data version) at most once.
* Shifting any existing share-link field index.

## Design

### Controls (one span, shown only in Percentile mode)

`bub_views.VIEW_MODES["percentile"].ctl` becomes `"bub-pctile-ctl-wrap"`. The
span holds:

| id | component | options | default |
|---|---|---|---|
| `bub-pctile-fit` | `dcc.Checklist`, inline | `fit` "Cycle fit", `censor` "skip 2019–20" | `["fit"]` |
| `bub-pctile-fit-model` | `dcc.Dropdown`, not clearable | the cycle-model list below | `"qr"` |
| `bub-pctile-fit-horizon` | `dcc.Dropdown`, not clearable | 10 / 20 / 30 yr | `20` |

**Cycle on by default** — the operator wants to test this, and the pill is
already opt-in. Old share links carry no value for the new fields and restore at
the layout default (fit on); this is a new capability, nothing on screen has to
be preserved (same reasoning as Raw MC's fields).

**`censor`** withholds 2019-04-15 – 2020-02-15 from the fit (and from hindcast
scoring), reproducing the operator's study figure. Off by default: dropping a
window because it misfits is a judgment call, and the band should show the
misfit rather than hide it.

**Cycle-model list** (fixed, filtered at layout time to models present in
`_app_ctx.PRICE_MODELS` and `quantized`): the navbar ticker's cycle
`qr, bub, pl, spl, lp3, cfg_1d_1u, ecfg_1d_1u, pca, grdy, ef`, labelled with
the ticker's labels (`LPPL₃`, `HybPPL`, `EPPL`) or the model's `legend_name`.

The pill-bar row gets `flexWrap: "wrap"` so the extra controls drop to a second
line on a phone instead of overflowing.

### Math (`btc_web/pctile_cycle.py`, import-light: numpy + scipy only)

* **Input series.** The picked model's percentile of every daily close with
  t ≥ 1 yr (log-log anchor), from the existing `_percentile_series` at the
  view's σ-mode, over the **full record** — never the visible x-window, so the
  slider cannot change the fit.
* **Fit.** Least squares of `c + a·cos(2πf t) + b·sin(2πf t)`; for fixed f this
  is linear, so f is scanned on a 400-point grid over periods 2–8 yr and refined
  by a bounded scalar minimisation within one grid step of the best node.
  A = √(a²+b²), φ = atan2(−b, a). R² against the series mean. The 2–8 yr window
  is an a-priori halving-cycle band (stated on the chart's footnote), which also
  stops a bootstrap replicate from locking onto a sub-year wiggle.
* **Bootstrap.** Circular moving-block bootstrap of the central fit's residuals
  on the weekly subsample (every 7th day), block = 1 yr (52 points),
  **B = 150**, fixed seed. Each replicate y* = fit + resampled residuals is
  refit with the same procedure (200-point grid). Keep each replicate's
  (c, A, f, φ).
* **Band.** Pointwise 10th / 90th percentile of the B replicate curves on the
  plotting grid, clipped to [0, 100].
* **Turns.** Peaks and troughs of the central wave are analytic:
  2πf t + φ = 2πk (peak) or π + 2πk (trough). For each, every replicate's
  nearest same-kind turn (round to the nearest k) gives a date; the 10–90 %
  range of those dates is the turn's timing bar.
* **Hindcast.** Cutoffs every 0.5 yr from (last − 10 yr) to (last − 2 yr);
  each cutoff with ≥ 5 yr of history is fit on the weekly subsample of data up to
  it and scored on the next 2 years against the flat forecast (in-sample mean).
  Report: number of cutoffs, mean RMSE of each, share of cutoffs where the wave
  wins. Censored days are excluded from fitting and scoring alike.
* **Percentile → price.** At each date, the model's fan in log10 price at every
  quantile (`price_at(q, t, sigma_mode=…)`), **sorted** (monotone rearrangement,
  Chernozhukov–Fernández-Val–Galichon 2010 — same as the study script), then
  linear interpolation in quantile, clipped to the outer quantiles. Forward
  direction only.

Cache: one `functools.lru_cache` in `figures/percentile.py` keyed on
(model key, σ-mode, censor, `_app_ctx._MODEL_FP`) holding the series, central
fit, replicate parameters and hindcast. Budget: < 1.5 s cold, measured.

### Figure (Percentile view, Cycle fit on)

* **x-range.** Left edge unchanged. Right edge: if the slider's right handle is
  at or past the current calendar year, today + horizon; otherwise the handle
  (the user zoomed into the past — no forecast region shown). The slider and its
  historical-mode cap are not touched; no callback writes `bub-xrange`.
* **Traces, back to front:** band (model colour, low alpha, legend "10–90 %
  band"); other active models' percentile lines exactly as today but at
  reduced opacity; the picked model's actual percentile (neutral grey, thin,
  legend "‹model› percentile (actual)"); the wave in-sample (model colour,
  solid) and forecast (same colour, dashed); turn markers (▲ peak / ▼ trough).
  Forecast turns carry `error_x` timing bars and a text label; in-sample turns
  carry hover only (clutter).
* **Forecast region:** light grey rectangle from today to the right edge.
* **Summary line** (paper-anchored, top-left):
  `QR · period 3.57 yr (3.49–3.66) · amplitude 30 pp · R² 0.56 · beats flat in 76 % of 2-yr hindcasts`.
* **Footnote** (paper-anchored, bottom): band = 10–90 % of 150 block-bootstrap
  refits (1-yr blocks, periods 2–8 yr searched); it shows uncertainty in where
  the wave is, not the ±N pp the percentile typically strays from it (N = the
  hindcast wave RMSE); descriptive fit, not a forecast.
* **Weak fit (R² < 0.20):** the summary line turns into a warning in the rich
  amber colour: "weak cycle (R² 0.04) — this model already explains most of the
  cycle; the wave is not meaningful", and forecast labels are omitted.
* **Log x-axis:** annotation x positions are log10(t) (Plotly annotations take
  log coordinates on a log axis); shapes and traces take data coordinates.
* All new colours are constants in `colors.py` (no hex literals elsewhere).

### Wiring

* `update_bub_pctile` gains three Inputs (the controls) and passes
  `cycle_fit`, `cycle_censor`, `cycle_model`, `cycle_horizon` in `p`; they flow
  through the existing `_get_pctile_fig` L1/L2 cache key.
* The three callbacks that emit `mode_styles` (`toggle_bub_view`, the clientside
  sync, `routing.deep_link_bub_view`) gain `Output("bub-pctile-ctl-wrap",
  "style")` in `bub_views.STYLE_OUTPUT_IDS` order (15 values).
* Snapshot: append the three fields at the **absolute tail** of
  `_SNAPSHOT_CONTROLS` (indices 348–350), `bub-pctile-fit` in
  `_CHECKLIST_OPTIONS` (`["fit", "censor"]`), defaults in
  `SNAPSHOT_DEFAULTS`, ids in `_TAB_CONTROLS["bubble"]`, defaults fingerprint
  pinned before and after via `tools/update_defaults_registry.py`.

## Left to the builder

Exact label typography and offsets, the in-sample turn hover text, opacity
values (as `colors.py` constants), how the summary wraps on narrow screens, and
test granularity.
