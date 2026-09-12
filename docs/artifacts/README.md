# docs/artifacts

Durable figures from analyses worth keeping. Unlike the throwaway plots in a
scratch directory, everything here is committed **and regenerable**: each
figure has a generator under `tools/` that recomputes its inputs from
`model_data.pkl`, so a price update or a model refit is one command away from
a refreshed artifact.

Note `.gitignore` excludes `qr_*.png` (and several other generated-plot
patterns) at the repo root — hence the `percentile-…` naming here.

## Percentile sinusoid study

| file | what it shows |
|---|---|
| `percentile-sinusoid-fits.png` | three panels, full record, 2010-07-26 – 2026-09-03 |
| `percentile-sinusoid-fits-extrapolated.png` | three panels, fits carried 10 years past the data, cropped to 2020 on |
| `percentile-sinusoid-calendar-only-10yr.png` | the calendar fit alone on one wide panel — whole record **and** its 10-year extension in a single plot |
| `percentile-sinusoid-calendar-only-20yr.png` | the same, extended 20 years |
| `percentile-sinusoid-calendar-censored-10yr.png` | as above, but the 2019 S2F-era window withheld **from the fit** |
| `percentile-sinusoid-calendar-censored-20yr.png` | the same, extended 20 years |
| `percentile-amplitude-reconciliation.png` | why a narrowing fan and a flat amplitude are the same fact |
| `…-pl.png` (7 files) | every figure above, re-read off the **Power Law** fan |

The single-panel pair is the one to read for dates and levels off one fit; the
three-panel pair is for comparing the forms against each other. All four come
from the same run of the same generator, so they cannot disagree.

The single-panel figures also carry a **half-amplitude twin** (orange): same
offset, frequency and phase, `A/2`. Because it shares the phase its extrema
fall on the *same dates* as the full fit, so its labels are drawn INSIDE the
envelope — peaks below the marker, troughs above — which is the inverse of the
full fit's convention and is what keeps the two label sets apart. It answers
"what if the cycle were half as wide?" without moving any turning point.

```bash
btc_venv/bin/python3 tools/render_percentile_sinusoid_artifacts.py
```

Three panels each, fitted to the **percentile of the BTC price within the QR
model's quantile fan** — not to price:

| fit | form | R² |
|---|---|---|
| Calendar-time | `c + A·cos(2πf·t + φ)`, period **3.571 yr** | 0.554 |
| Log-time | `c + A·cos(ω·ln t + φ)`, **ω = 7.226** (λ = 2.386) | 0.264 |
| Both terms | the two summed | 0.699 |

Each figure carries two kinds of label:

* **coloured** — peaks and troughs of the *fitted curve*: date, fitted
  percentile, and the price the QR fan puts at that percentile on that date;
* **dark** — major highs and lows of the *actual BTC price*: date, the real
  close, and the percentile that close sat at.

### The fold, and the valid way out of it

Every price on these charts is "what the fan puts at this percentile on this
date", and past **2028-09-07** the raw QR fan stops being able to support that
phrase: the channels cross, so Q65 can sit above Q80 and a percentile label is
no longer a ranking. By 2080 the raw fan is fully inverted — a Q10→Q90 width of
**−0.034 dex**.

The figures are drawn from a **monotone-rearranged** fan (Chernozhukov,
Fernández-Val & Galichon 2010, *Quantile and Probability Curves Without
Crossing*, Econometrica 78(3)). The true conditional quantile function is
monotone in τ by definition, so any crossing in a fitted fan is estimation
error, and sorting the fitted values at each date gives an estimator **weakly
closer to the truth in every Lᵖ norm**. It is a theorem, not a repair.

What makes it legitimate where `np.maximum.accumulate` is not: sorting returns
the *same multiset* of fitted values, reassigned to quantile levels in
increasing order. A running maximum discards values and duplicates others,
inventing a fan the fit never produced. Rearrangement invents nothing.

Measured on this fan, it is free inside the record and decisive outside it:

| | 2015 | today | 2032 | 2046 | 2080 |
|---|---|---|---|---|---|
| bands moved by sorting | 0 of 27 | 0 of 27 | 9 of 27 | 19 of 27 | 26 of 27 |
| Q10–Q90 width, raw | 0.864 | 0.462 | — | 0.192 | **−0.034** |
| Q10–Q90 width, sorted | 0.864 | 0.462 | — | 0.220 | 0.176 |

The percentile series moves by at most **0.09 pp** (correlation 1.000000, zero
days moved by more than 1 pp) and the sinusoid fit is identical to four
decimals. So rearrangement costs nothing and removes the fold outright.

**It fixes the order, not the confidence.** The rearranged fan still narrows to
0.22 dex by 2046 — a 1.7× spread between Q10 and Q90, tighter than any period
on record. That over-confidence is QR extrapolating independent slopes, and no
amount of sorting touches it; the footer says so on every figure.

`--raw` renders without rearrangement, which is what produced the daggered
figures this replaced. `fan_folds()` stays in the generator as the assertion
that the rearrangement worked, rather than being deleted as dead code.

The model-level alternative, not implemented: fit the fan in a location-scale
form `Q_τ(t) = μ(t) + σ(t)·z_τ` (He 1997), which cannot cross at any date *by
construction* while still letting σ vary with time. That is the principled fix
— rearrangement is the post-hoc one — and it would need a refit rather than a
render change.

### QR or PL? The crossing and the narrowing are the same feature

`tools/render_percentile_sinusoid_artifacts.py pl` regenerates the whole set
against the Power Law fan, which never folds. Before switching to it, the
reason PL's bands are parallel is worth knowing, because it is also the reason
not to switch.

**PL does not fit its bands.** It fits ONE line by OLS, takes the standard
deviation of the residuals, and puts band *q* at `intercept + z_q·σ` with the
same slope (`btc_core/_simple.py::PowerLawModel.__init__`). Every band is that
one line shifted vertically, so the fan is parallel by construction — and its
width is frozen for all time:

| Q10→Q90 width, dex | 2011 | 2015 | 2019 | 2023 | today | 2046 |
|---|---|---|---|---|---|---|
| PL | 0.753 | 0.753 | 0.753 | 0.753 | 0.753 | 0.753 |
| QR | 1.331 | 0.864 | 0.671 | 0.547 | 0.462 | 0.192 |

QR fits each quantile separately, so each gets its own slope and the fan is
free to change width. **That freedom is exactly what lets the channels cross.**
You cannot have a fan that narrows and a fan that can never fold — not from
these two models.

Which assumption is false is measurable. Residual sd by era: **0.379, 0.350,
0.235, 0.150 dex** (2010-14, 14-18, 18-22, 22-27). The spread really has
compressed, by 2.5×. So QR's crossing is the cost of tracking something true,
while PL's non-crossing is the cost of freezing something false — and PL's
error is *inside* the record, where the data is, whereas QR's folding begins in
2028, out in the extrapolation where nothing is measured.

What it costs to switch, on the censored fit: period 3.573 → 3.596 yr (+0.6 %),
phase −126.7° → −120.1° (≈ 23 days), amplitude 34.1 → 32.2 pp, and **R² 0.642 →
0.571**. Every conclusion survives; the fit is simply worse. The two percentile
series correlate 0.975.

For completeness `sigma_mode="resqr"` (time-varying σ, which would in principle
give PL a narrowing fan that still cannot cross) is worse than both: 1,030 of
5,884 days pin to the rails and R² falls to 0.552.

**QR stays the default.** PL is generated so the difference can be looked at
rather than argued about.

### Why a narrowing fan and a non-decaying amplitude are not a contradiction

They look incompatible and are not, because **percentile is a normalised
coordinate**. It measures where price sits *inside* the fan, not how far it is
in dollars. If the fan narrows at the rate the price deviations narrow, the
ratio is flat and the percentile swing is flat with it.

Measured era by era — price spread is the sd of log10(close) minus the QR
median line; fan sd is the Q10–Q90 width converted to a Gaussian σ:

| era | price sd (dex) | fan sd (dex) | ratio | mean \|pct − 50\| |
|---|---|---|---|---|
| 2010–2014 | 0.378 | 0.432 | 0.875 | 22.1 |
| 2014–2018 | 0.353 | 0.314 | 1.124 | 32.6 |
| 2018–2022 | 0.234 | 0.248 | 0.942 | 24.5 |
| 2022–2027 | 0.150 | 0.201 | 0.747 | 20.8 |

The price spread shrinks **2.53×** and the fan shrinks **2.16×** — nearly
together. The ratio drifts down only 15 % across sixteen years and is not even
monotone (the 2014–18 era is the highest of the four). The last column says the
same thing directly: the mean percentile excursion has no trend.

The cancellation is close to tautological, which is the real answer. The QR fan
is fitted **to those very deviations**, so of course its width tracks them. An
amplitude measured in percentile units can only trend if the *shape* of the
residual distribution changes; a change in its *scale* divides out.

**Ask the question in dollars and the decay appears**, as it should. The same
sinusoid fitted to the log-price deviation instead of the percentile:

| target | exp-to-floor τ | amplitude across the record |
|---|---|---|
| percentile (pp), all data | 4 × 10¹⁵ yr | 30.26 → 30.26 (−0.0 %) |
| log-price deviation (dex), all data | **41.1 yr** | 0.336 → 0.227 (**−32.5 %**) |
| log-price deviation, 2019 withheld | 57.6 yr | 0.351 → 0.266 (−24.4 %) |

So "the amplitude does not decay" is a statement about the percentile series
and is correctly scoped, but it is easy to misread as "volatility is not
falling" — which is false. The narrowing fan **is** the falling volatility.

One caveat on the dollar-space decay: it is not robust either. Withholding the
2019 window cuts it from −32.5 % to −24.4 %, and the free-sign power law flips
from D = +0.021 (decay) to D = −0.020 (growth). Part of what reads as decay is
the 2019–20 excursion inflating the early-record amplitude's opposite end.

`percentile-amplitude-reconciliation.png` shows this in three panels sharing
one time axis, because the claim is a ratio and a ratio needs its numerator and
denominator shown separately: **A** the price's deviation from the QR median
with the fan drawn over it in the same units (both close, 3.04× and 2.63× on
the rolling endpoints), **B** the ratio of the two (mean 0.85, wandering
0.20–1.34 with no trend), **C** the percentile series the sinusoid was actually
fitted to — which is panel B, and so has no decay to find.

**Rescaling the percentile by a constant does not recover the decay**, because
R² and the fitted damping exponent are scale-invariant. Multiplying the whole
series by 1/2.53 gives amplitude 30.261 → 11.961 pp and leaves everything else
byte-identical: R² 0.554061 either way, `t^−D` still picks D = −0.1701, exp-to-
floor still runs τ → 10¹⁵ yr. It is a change of units, and a change of units
cannot create a time trend.

What does work is a **time-varying** rescale — multiplying the excursion by the
fan width *at each date*. That gives τ = 33.6 yr and D = +0.047, a decay. But
multiplying the percentile back by the fan width is just un-normalising it: the
result is panel A, in dollars, which is where the decay lived all along.

### What the study found

* The calendar period lands at **3.57–3.61 yr** under every variant tried —
  persistently *shorter* than the 4-year halving interval.
* The log-frequency **ω ≈ 7.23** was fitted blind to the percentile series and
  falls inside the cluster this repo's own LPPL models find against log price
  (`_lppl.py`: 7.126 / 7.343 / 7.503). Two different targets, two different
  procedures, one frequency band.
* A power spectrum puts the dominant peak at **3.580 yr**, agreeing with the
  independently fitted 3.571 yr to 0.25 %. It stands **5.95×** above an AR(1)
  red-noise background — significant against an *a-priori* 3–5 yr band
  (p = 0.005), not significant if the whole spectrum is trawled.
* **Amplitude is not decaying.** Exponential-to-floor, exponential-to-zero,
  power-law `t^−D`, and pinned asymptotes at 0 / 12.5 / 25 / 37.5 all either
  refuse to decay (τ → 10⁶ yr, collapsing onto the constant-amplitude fit) or
  degenerate into a 2010-12 transient. Given a free sign, `t^−D` chooses
  **D = −0.170** — mild *growth*. Re-run against the PL fan it is the same
  answer: exponential-to-floor picks τ = 679 yr (a 2.3 % decline across the
  whole record, worth ΔR² = 0.0001) on all data and τ → 10¹⁵ yr once the 2019
  window is withheld, while the free-sign power law picks D = −0.119 / −0.155
  — growth again. No parameterisation of either model supports a decaying
  amplitude **in percentile units**; see the section above for why that is
  compatible with a fan that narrows 2.16×, and what happens when the same
  question is asked in dollars.
* The fits systematically **turn ~90 days late** and undershoot both rails:
  the 2021 peak was actually Q99.8 against a fitted Q79.3.

Fits are descriptive (R² 0.55–0.70). The extrapolation is drawn because it was
asked for, not because a sinusoid predicts ten years.
