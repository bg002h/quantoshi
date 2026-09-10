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
| `percentile-sinusoid-fits.png` | full record, 2010-07-26 – 2026-09-03 |
| `percentile-sinusoid-fits-extrapolated.png` | same fits carried 10 years past the data, cropped to 2020 on |

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
