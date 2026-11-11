# Review r1: Percentile cycle fit — design spec

**Spec:** `docs/superpowers/specs/2026-11-10-percentile-cycle-fit-design.md` (as of `8e42e1f0`)
**Reviewer:** independent (author != reviewer). Read-only against the repo; all experiments are
scratch scripts under
`/tmp/claude-1000/-scratch-code-bitcoinprojections/6c13708a-e257-46cf-bdef-a2a8a773488f/scratchpad/`
that reuse `fit()` / `curve()` from `tools/pctile_cycle_explore.py` unchanged (same estimator as
the spec's numbers).

**Counts: 2 Critical / 4 Important / 10 Minor / 3 Nit**

The spec's measured numbers reproduce exactly (R², periods, ACF, band widths, hindcast RMSEs).
The two Criticals are about what those numbers are made to *claim* on the chart: the hindcast
win rate is inflated by look-ahead, and the band, as labelled and hovered, will be read as a
range for the percentile (and price) that it covers less than a third of the time.

---

## C-1 — The on-chart hindcast win rate has look-ahead bias: 76 % falls to 53 % with an as-of fan

**Evidence.** The percentile series being hindcast is computed against the model's fan fitted on
**all** data through today (`_percentile_series(mdl, …)` uses the live `PRICE_MODELS` object). So a
"fit up to 2017, score 2017–2019" test scores a series whose 2017–2019 values were defined by a fan
that saw 2021. For QR this can be removed: the Time Machine grid holds QR fits made on data up to
each frame (`btc_web/timemachine.py:131` `asof_qr`, built by `tools/build_timemachine_grid.py` on
truncated data; 142 frames, 2012-04-01 → 2026-07-01, ~monthly). QR is σ-invariant
(`resqr == constant`, verified), so the as-of fan is an exact like-for-like.
Scratch `honest_hindcast.py`, same cutoffs/fit/score as the explore script, the percentile
recomputed against `asof_qr(last frame <= cutoff)`:

```
H=1.0: n=19 LEAKY wave 20.0 flat 26.4 win 68% | HONEST wave 19.1 flat 22.4 win 63%
H=2.0: n=17 LEAKY wave 21.4 flat 27.0 win 76% | HONEST wave 21.8 flat 23.3 win 53%
H=4.0: n=13 LEAKY wave 22.8 flat 28.3 win 92% | HONEST wave 22.4 flat 24.3 win 69%
```

Peak timing is mostly robust to the fix (2019-06 / 2021-06 / 2023-06 cutoffs move by < 1 month),
except the 2017-06 cutoff, whose next peak moves from 2021-08 (spec) to **2020-10** with the as-of fan.

**Why it matters.** Goal 4 puts "beats flat in 76 % of 2-yr hindcasts" on the chart as the evidence
of skill. Without look-ahead it is a coin flip (53 %, n = 17 overlapping windows, roughly 4
independent ones). The RMSE edge survives but it is small (21.8 vs 23.3 pp). The spec's "the wave is
real enough to show" verdict rests on the leaky figure. Every non-QR model has the same leak, and
also through its resqr bands, which are fitted on full-sample residuals. The `censor` option makes
it worse, because the withheld window was chosen *after* seeing that it misfits.

**Fix.** Choose one and write it into the spec:
(a) For QR, compute the hindcast against `timemachine.asof_qr(frame)` per cutoff (about 17 × 20 ms
extra, cached with the fit). For models with no as-of fan at the view's σ-mode, show no win rate
("no out-of-sample hindcast for this model").
(b) Drop the win rate from the summary line. Keep the RMSE pair only, labelled
"hindcast against today's fan (optimistic)".

Either way, the Statistical-judgment section should quote the honest numbers, so later readers do
not inherit 76 %.

## C-2 — The "10–90 % band" will be read as where the percentile (and price) will be; it covers the actual series 31 % of the time

**Evidence.** Scratch `band_checks.py`: the spec's procedure (1-yr circular block bootstrap of
weekly residuals, B = 150, seed 0, 200-pt grid), with the pointwise 10/90 band evaluated in-sample:

```
qr : share of actual daily percentile inside the 10-90 band = 31%  (mean band width 16.5 pp, resid sd 19.2 pp)
pl : 36%      bub: 27%
```

Out of sample it is no better. In `calib.py`, the band built at each cutoff is scored against the
next 4 years of actual percentile: 26 / 54 / 48 / 32 / 37 / 63 / 58 % for cutoffs 2016.5 … 2022.5.
Even as a band for *where the wave is*, it looks anti-conservative. The cutoff band's 10–90 %
period interval excluded the later full-data period (3.57 yr) in 3 of 7 cutoffs (expected ≈ 1.4):
2016.5 gave 2.85–3.18, 2019.5 gave 3.61–3.96, and 2020.5 gave 3.60–4.12. The evidence is weak
(n = 7, correlated), but it points the same way. n_eff ≈ 12–16 is small-sample territory for
bootstrap percentile intervals.

**Why it matters.** The spec's anti-goal is "a band that understates it". The band's only
disclosure is the footnote. The two places a reader actually takes numbers from contradict that
footnote:
- the legend reads "10–90 % band", the standard wording for an 80 % interval;
- Goal 3 hover gives "the price the model's fan puts at … both band edges". That is a dollar range
  for a future date, and the realised percentile falls outside it about 70 % of the time.

The operator chose the refit band, and that choice stands. The labelling and the band-edge price
hover are the spec author's, and they make the display misleading.

**Fix (minimum).**
1. Name the band for what it is, in the legend and in hover: "fit uncertainty (10–90 % of refits)".
2. Do not present band-edge prices as a price range. In hover, show the price at the fitted
   percentile, plus "percentile typically strays ±N pp (hindcast RMSE) → $a–$b".
3. State the coverage in the footnote: "the actual percentile sat inside this band X % of the time".
   Compute X at runtime from the in-sample series.

**Fix (better, still within the operator's "band from refitting on resampled data").** Add an outer,
lighter *predictive* envelope. For each replicate, take its curve plus one block-resampled residual
path, then take the pointwise 10/90 of that sum. This also widens with horizon, and its in-sample
coverage is ≈ 80 % by construction. Keep the refit band as the inner band.

## I-1 — Old Percentile-view share links restore with the cycle overlay on, the sender's lines dimmed, and the x-axis stretched to 2046

**Evidence.** For a registered fingerprint, the v4 decoder fills only keys present in that fp's
registry entry (`btc_web/snapshot.py:949-960`). Older entries have no `bub-pctile-fit:value`, so the
key is absent and `apply_tab_bubble` leaves the control at its layout default `["fit"]`
(`callbacks/snapshot_cb.py:285`, `state.get(..., no_update)`). Positional (q3) links behave the same.
The spec justifies this as "nothing on screen has to be preserved (same reasoning as Raw MC)". That
reasoning does not carry over. Raw MC was a new mode with nothing on screen to preserve. This feature
changes an existing view: the sender's model lines are dimmed, a forecast is added, and the x-axis
right edge moves from today to today + 20 yr. On a linear x-axis that squeezes the shared history
into about 44 % of the width. `_PREDATING_LINK_VALUES` (`snapshot.py:698-711`) exists for exactly
this case: "Fields whose DEFAULT differs from what the app did before the field existed". The
`bub-mc-paths` precedent was treated as a real bug.

**Why it matters.** A link shared before deploy no longer shows what the sender saw.

**Fix.** Add `"bub-pctile-fit:value": ([], …)` to `_PREDATING_LINK_VALUES`. Today
`_fill_predating_fields` (`:714-722`) supports only a `"yes"`-ticked checklist gate. Extend it to
accept `requires=None` (unconditional), or a `(key, value)` gate such as
`("bub-view-mode:data", "percentile")`. Add tests mirroring `test_mc_paths_toggle.py`'s predating
cases: v4 registered fp, positional, and a new link with an explicit off. Note that the M-2 latent
hole in the 2026-11-09 review (an unregistered fp falls back to current defaults, so the rule cannot
fire) applies here too.

## I-2 — 7 of the 10 picker models are "weak", and the spec still draws a confident forecast wave, turn markers and timing bars for them

**Evidence.** Scratch `models_check.py`, resqr, full record:

```
qr .557  pl .454  spl .459 | bub .044  lp3 .121  cfg_1d_1u .151  ecfg_1d_1u .155  pca .072  grdy .078  ef .066
periods: cfg_1d_1u 2.00, ecfg_1d_1u 2.00 (lower grid bound), ef 7.89 (upper bound)
```

For BM, bootstrap periods span 2.27–6.72 yr (10–90). The spec's own judgment says "The view must say
so **instead of drawing** a confident-looking sinusoid through noise". The Figure section, however,
only swaps the summary line for a warning and omits forecast *labels*. The dashed forecast wave, its
turn markers and its `error_x` bars are all still drawn. Three of the weak fits also sit at a grid
boundary, so the reported "period" there is an artefact of the search window rather than an estimate.

**Why it matters.** The default picker is fine (QR), but most picker choices produce exactly the
display the spec says must not happen.

**Fix.** When R² < 0.20, draw the in-sample wave only, faint, with no forecast segment, no turn
markers and no timing bars. Keep the band if wanted. When the refined period is within one grid step
of 2 or 8 yr, report "period at search limit (≤ 2 yr / ≥ 8 yr)" instead of a number, and treat the
fit as weak.

## I-3 — The picker's "EPPL" is a different series from the EPPL line already on the chart, under the same name

**Evidence.** Scratch `master_vs_cfg.py`, resqr:

```
hybppl 'HybPPL'  vs cfg_1d_1u  '1D_1U'  | max |diff| 0.0 pp
eppl   'EPPL'    vs ecfg_1d_1u '1D_1U'  | max |diff| 69.7 pp, mean 18.3 pp
lppl   'LPPL'    vs lp3        'LPPL₃'  | max |diff| 75.8 pp, mean 17.3 pp
```

`bub-model-show` offers the master `eppl` (`snapshot.py:537`), and the Percentile view passes masters
through as-is (`callbacks/charts/__init__.py:907-909`). The picker uses the ticker's `ecfg_1d_1u`,
labelled "EPPL" (`callbacks/ticker.py:16-20`). With EPPL ticked in the model list and "EPPL" picked,
the chart shows a dimmed "EPPL" line and a grey "EPPL percentile (actual)" up to 70 pp apart. The
cfg variants' `legend_name` is "1D_1U" for both HybPPL and EPPL. A builder who uses `legend_name` for
the summary line or the "‹model› percentile (actual)" legend, as the spec suggests, prints "1D_1U".

**Why it matters.** Two different curves carry the same model name on one chart.

**Fix.** Either use the view's own keys (`eppl`, `hybppl`, and `lp3` / `lppl` as wanted) in the
picker, or label the config variants distinctly (e.g. "EPPL 1D_1U"). Specify that the picker label,
not `legend_name`, is used in every on-chart string.

## I-4 — Forecast turn labels collide at the defaults (log x, 20 yr) on every screen size

**Evidence.** The default xscale is `log` (`snapshot_defaults.py:98`). With t_lo ≈ 0.94 (the 2010.5
floor, `figures/percentile.py:94-96`), today t ≈ 17.3 and right edge t ≈ 37.3, the 20-yr forecast
takes log10(37.3/17.3) / 1.599 ≈ **21 %** of the plot width. P = 3.57 yr gives about 11 turns in
that strip. Same-kind turns sit Δlog = 0.078 → 0.042 apart:
- an 800 px desktop plot gives about 39 → 21 px between labels;
- a phone plot (~250 px; `style.css:1189-1190` sets the graph to 55vw, min 280 px) gives about 12 → 7 px.

A one-line "Aug 2028 · Q80 · $310k" label is about 130 px wide. Even on a linear axis, same-kind
spacing is about 78 px on desktop.

**Why it matters.** At the default settings the forecast labels (Goal 2) are an unreadable overprint.

**Fix.** Add a label-density rule: text-label only the next peak and next trough (or the first N
turns whose label pitch is ≥ ~120 px at the current axis type), and give the rest markers with
hover only. Alternatively, use 3-line labels (date / Qnn / $) that alternate above and below. Pin
the rule with a test on the log-axis default.

---

## Minor

**M-1 — The "after" registry pin is the first eviction ever.** `snapshot_defaults_registry.json`
already holds 20 entries (CAP = 20, `tools/update_defaults_registry.py:22,42-45`). The new
fingerprint evicts `faf00e93` (2026-04-24). Links from that fp would fall back to current defaults,
and 7 fields differ: `dca-mc-enable, dca-model-show, hm-c-mid1, hm-c-mid2, hm-palette, sc-mc-enable,
sc-model-show`. That fp was live for about one day, so the impact is small. Decide knowingly: raise
CAP in a separate tooling commit, or accept the eviction in the spec.

**M-2 — Turn timing by "nearest same-kind turn" truncates at ±P/2.** For QR/PL it agrees with
cycle-index matching up to about 25 yr. At a 30-yr horizon, PL's 10–90 width (2.0 yr) already exceeds
P/2 = 1.79, so nearest-k caps it. For BM, nearest-k gives 3.2–4.2 yr against 3.8–21 yr by cycle index.
Use "the j-th peak after the last observation" in each replicate. Omit the bar when its width > P/2.
Clamp `array` / `arrayminus` to ≥ 0: the central turn can fall outside its replicates' 10–90 range
when the distribution is skewed.

**M-3 — The band pinches at every central turn by construction.** cos is flat at an extremum, so
phase spread barely moves the value there. Measured QR widths: +10 yr 41.7 pp, +20 yr 35.3 pp (narrower).
A reader will read a narrow band at a forecast peak as confidence in the peak. Say so in the footnote
("narrowest at turns; turn timing is the bars"), or drop the "widening" wording.

**M-4 — PL/SatPL at resqr are clamped series.** About 16 % of days sit at the fan floor (all of
2015–16 at 0.001 %), and parts of 2021 sit at 99.999 %. The resqr fan tails are flat: PL Q0.001 % ==
Q1 %, e.g. $372 / $372 at 2015.5 against an actual $256. Least squares on a censored series biases c
and A, and the bootstrap resamples the 2-yr plateau as noise. At minimum, show the clamped share in
the summary when it exceeds 5 %.

**M-5 — Invalid share-link values.** Specify a fallback (QR, 20 yr) for these cases: a
`bub-pctile-fit-model` that is not in `PRICE_MODELS`, not quantized, `u1`, or `None` (for example
`ef` on a host without `model_data_ef.pkl`), and a horizon outside {10, 20, 30}.

**M-6 — Data window vs axis window.** Today `t_hi = min(xmax, today)` drives the data mask, the axis
range and the ticks (`figures/percentile.py:100-160`; `_year_ticks(p["xmin"], p["xmax"])` is filtered
to [t_lo, t_hi]). The spec must state that the axis range and the tick generation run to
today + horizon while the data mask stays capped. The trigger "handle ≥ current calendar year" also
fires for handle = 2026, which means 1 Jan 2026, before today. Prefer `yr_to_t(xmax) >= today_t`;
the capped default of 2027 still triggers.

**M-7 — Payload (ALARA).** Specify the plotting grid: weekly, or about 1 000 points over the window,
in float32. Daily values over 37 yr for wave, forecast, two band edges and three hover-price customdata
columns come to roughly 13 k points × 6 arrays. Dash base64-encodes numpy arrays, so rounding does
not save bytes.

**M-8 — Hover plumbing.** `_add_date_hover` prepends the date to an existing customdata row, so
customdata[0] becomes the date and the trace's own values move to [1:] (`figures/common.py:904-906,
941-945`). It also gives every trace without a hovertemplate the view's default
"percentile %{y}%" template. The builder's hover templates must index from customdata[1]. The band
edge traces should be `hoverinfo="skip"`, with the band shown in the wave's hover.

**M-9 — Narrow screens.** The summary (about 95 chars) and the footnote (about 230 chars) are Plotly
annotations, which never wrap. On a 280 px phone chart they either clip or take most of the plot
height, and the footnote carries the caveats that C-2 depends on. Put them in an `html.Div` under
the graph (one extra Output, which wraps naturally), or at least make the legend and hover text carry
the caveat (C-2 fix 1–2).

**M-10 — Hindcast and R² reporting.** "76 % of 17 cutoffs" counts 2-yr windows 0.5 yr apart, which
is about 4 independent trials. Show n and the word "overlapping". With `censor` on, R² is computed on
the retained days only, so it rises when the misfit is removed. Label it "R² (2019–20 excluded)".

## Nit

**N-1 — Hard-coded counts the builder must update.** `routing.py:427` returns `(NU,) * 20`; it must
become 21, or every non-Tab-1 URL change returns the wrong arity. The docstrings say 14 / 15
(`bub_views.py:52,90,103`, `charts/__init__.py:670,686-693`, `routing.py:417`). The tests pin 14 / 20
(`test_bub_view_modes.py:85,97,105,216`, `test_bub_deep_links.py:20,36,39`). The new
`bub-pctile-ctl-wrap` goes *between* `bub-cagr-fwd-wrap` and `bub-occ-ctl-wrap` in all three
emitters (CTL_IDS order follows VIEW_MODES order), not at the end.

**N-2 — Weak-fit wording and colour.** "This model already explains most of the cycle" is plausible
for BM/HybPPL/EPPL/LPPL₃, but it is an unverified causal claim for EF, whose period is pinned at the
7.89 yr edge. "Weak cycle in this model's percentile" is safer. "Rich amber" is `PCTILE_RICH_COLOR`,
which is also the rich-zone shading, so the two meanings get conflated.

**N-3 — Layout collisions.** A top-left summary collides with `legend_pos` top-left
(`_MC_LEGEND_POS`). Forecast labels can land on the bottom-right watermark. A `dcc.Checklist` inside
an inline span needs `display: inline-block`, as the existing ctl Dropdowns use.

---

## Checked and found sound

- **Snapshot indices.** `len(_SNAPSHOT_CONTROLS) == 348`, tail `bub-mc-extremes`, so the new fields
  land at 348–350. `apply_tab_bubble` outputs derive from the `bub-` prefix
  (`snapshot_cb.py:227-232`), so no hand wiring is needed. The checklist bitmask for `["fit","censor"]`
  round-trips: `_mask_to_list(0)` gives `[]`, which is present, not absent.
- **Output arity.** Adding `ctl` to the percentile row makes `STYLE_OUTPUT_IDS` 15 long, in the order
  wraps / pills / panels / (cagr, pctile, occ). The clientside table regenerates from `VIEW_MODES`.
- **Cache.** `_get_pctile_fig` keys on the whole `p` plus `_day` (`utils.py:233-237`). L2 Redis keys
  include `_MODEL_FP` (`cache.py:36`). `_MODEL_FP` exists (`_app_ctx.py:161`).
- **X-range vs the historical cap.** The cap writes `[2010, 2027]` on entering the view
  (`charts/__init__.py:733-750`). 2027 ≥ 2026, so the forecast shows by default. No callback writes
  `bub-xrange`. The σ-mode default is `resqr` (`snapshot_defaults.py:91`).
- **Reproduction.** Every spec number reproduces from `tools/pctile_cycle_explore.py`, including the
  persistence baseline it prints but the spec omits (last-value RMSE 26.1 / 35.1 / 35.8; the wave
  beats it too).
- **Central line and timing.** The central line lies inside its own band on 100 % of forecast days
  (QR, PL, BM, 30 yr). Nearest-k and cycle-index turn timing agree for QR up to 30 yr.
- **Cold cost.** 1.0–1.2 s on dev (QR: series 0.01 s, central 0.49 s, bootstrap ×150 0.60 s,
  hindcast 0.10 s; PCA series 0.22 s), within the 1.5 s budget on dev. Prod will be slower.
  Vectorising the grid scan is the obvious lever.

## Not checked

Browser rendering, E2E, and prod behaviour. The `db93853c` implementation commit that appeared during
this review is out of scope (spec only). I did not check whether resqr's flat tails (M-4) are intended
in `btc_core`.
