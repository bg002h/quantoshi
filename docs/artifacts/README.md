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

### The 2019 censored variant

`2019-04-15 – 2020-02-15` (307 days, 5.2 % of the record) is withheld from the
FIT only — the data stays on the chart, shaded, so the omission is visible.
The window brackets the excursion that followed PlanB's stock-to-flow article
(2019-03-22).

| | all data | window withheld |
|---|---|---|
| period | 3.5706 yr | **3.5733 yr** (+0.08 %) |
| phase | −127.85° | **−126.69°** (−0.91 %) |
| amplitude | 30.26 pp | 34.11 pp (+12.7 %) |
| offset | 49.04 | 46.92 (−4.3 %) |
| R² | 0.554 | **0.642** (+15.8 %) |

**The cycle does not depend on that window.** Dropping 5 % of the record buys
a 16 % better fit while moving the period by 0.08 % and the phase by 0.9 %;
the excursion is absorbed entirely by amplitude and offset.

Re-asking the decay question on the censored fit is the sharper version of it,
since a large late excursion is exactly what could prop an amplitude up and hide
a decay. It does the opposite. Exponential-to-floor and exponential-to-zero both
still run τ → 10¹⁶ yr and collapse onto the constant-amplitude fit (ΔR² = 0.0000),
and the free-sign power law `A(t) = A₀·t^−D` picks **D = −0.207** — *growth*,
and steeper than the −0.170 it picks on all the data. Envelope 22.6 pp (2010) →
40.7 pp (2026). Withholding the window makes the amplitude grow faster, not
slower.

On whether the article *caused* the excursion: the window's mean residual of
+33.1 pp is the largest of the 19 non-overlapping windows in the record (next:
+20.4), and a model trained only on pre-publication data under-predicts it by
+36.6 pp against +3.4 pp for all 6.5 years after. But causation is not testable
here — n = 1, no counterfactual, the window was chosen by eye, and S2F's own
thesis is about the May-2020 halving, so "the model moved the price" and
"halving anticipation moved the price" predict identical timing. The numbers
measure anomaly, never cause.

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
  **D = −0.170** — mild *growth*.
* The fits systematically **turn ~90 days late** and undershoot both rails:
  the 2021 peak was actually Q99.8 against a fitted Q79.3.

Fits are descriptive (R² 0.55–0.70). The extrapolation is drawn because it was
asked for, not because a sinusoid predicts ten years.
