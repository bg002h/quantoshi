# LPPL Model Family

Quantoshi implements 10 Log-Periodic Power Law (LPPL) variants as first-class
price models. This document explains the family, the parameter-fitting
methodology, and which variants are physically meaningful.

## Model variants

All defined in [`btc_core.py`](../btc_core.py), registered in `_app_ctx.PRICE_MODELS`:

| Class | Key | Freqs | Params | Weighting | Notes |
|-------|-----|-------|--------|-----------|-------|
| `LPPLModel` | `lppl` | 1 | 6 | Unweighted | Damped primary oscillation |
| `LPPLModelW` | `lppl_w` | 1 | 6 | Log-time (1/t) | Weighted counterpart |
| `LPPL2Model` | `lp2` | 2 | 9 | Unweighted | Primary damped, secondary undamped |
| `LPPL2ModelW` | `lp2_w` | 2 | 9 | Log-time | |
| `LPPL3Model` | `lp3` | 3 | 12 | Unweighted | **Recommended default** |
| `LPPL3ModelW` | `lp3_w` | 3 | 12 | Log-time | |
| `LPPL4Model` | `lp4` | 4 | 15 | Unweighted | ⚠ Likely overfit — see below |
| `LPPL4ModelW` | `lp4_w` | 4 | 15 | Log-time | ⚠ Likely overfit |
| `LPPL4ModelN13` | `lp4_n13` | 4 | 15 | Unweighted | Excludes ω ∈ [11.5, 14.5] |
| `LPPL4ModelWN13` | `lp4_w_n13` | 4 | 15 | Log-time | Excludes the ω=13 band |

## Model formulas

**Single-frequency (LPPL):**
$$\log_{10}(\text{price}) = A + B\log_{10}(t) + C \cdot t^{-D}\cos(\omega_1 \ln t + \varphi_1)$$

**Two-frequency (LPPL₂):**
$$... + C_1 t^{-D}\cos(\omega_1 \ln t + \varphi_1) + C_2\cos(\omega_2 \ln t + \varphi_2)$$

**Three-frequency (LPPL₃):**
$$... + C_3\cos(\omega_3 \ln t + \varphi_3)$$

**Four-frequency (LPPL₄):**
$$... + C_4\cos(\omega_4 \ln t + \varphi_4)$$

The primary oscillation is damped by $t^{-D}$. All secondary oscillations are
undamped (constant amplitude).

## The three robust frequencies

Across all weighting schemes (unweighted/weighted) and all constraints
(with/without ω=13 excluded, different model orders), exactly **three**
frequencies consistently appear:

- **ω ≈ 7** — primary halving cycle (damped)
- **ω ≈ 9** — non-harmonic secondary (ratio ≈ 1.36)
- **ω ≈ 21** — either 3×W₁ harmonic or distinct structural mode

LPPL₃ captures all three simultaneously. This is the physically most honest
fit — it's the smallest model that represents all genuine log-periodic
structure in Bitcoin's price history without inventing oscillations that
aren't there.

## Why LPPL₄ is probably overfit

LPPL₄'s fourth frequency is **not stable** across fitting constraints:

- **Unweighted fit**: ω ≈ 13.3
- **Weighted fit**: ω ≈ 17.1
- **Excluding [11.5, 14.5]**: ω ≈ 17.5

Each of these is explainable as an **intermodulation product** of the three
stable frequencies:

- ω ≈ 13 ≈ W₂ − W₁ = 20.9 − 7.4 = 13.5 (difference frequency)
- ω ≈ 17 ≈ W₁ + W₃ = 7.1 + 9.9 = 17.0 (sum frequency)

Forcing the optimizer out of the [11.5, 14.5] band just pushes it into the
next available intermod pocket. This is characteristic of overfitting: the
model has enough degrees of freedom to fit noise by combining its genuine
oscillations in arbitrary ways.

**LPPL₃ is the recommended default.** LPPL₄ is available in the config panel
for comparison, but a warning dialog fires every time it's enabled.

## Weighting and the 2017-2020 market regime shift

Bitcoin's raw daily data is **uniform in calendar time but non-uniform in
log-time**: at t=16 (year 2025) we have ~8× the log-time sample density of
t=2 (year 2011). Standard least-squares over-weights the recent market era.

**Applying 1/t weighting reveals:**

- **Damping D drops 30-47%** across all LPPL variants — the narrative that
  "Bitcoin bubbles are shrinking" is partly an artifact of the 2020-2024
  cycle being over-weighted
- **LPPL₂'s secondary frequency flips entirely** — ω=21 unweighted
  (recent-era dominant) vs ω=9 weighted (older-era dominant)

Bitcoin's market structure changed materially around 2017 (retail boom, CME
futures) and 2020-2021 (institutional adoption, ETFs). The 2010-2019 era
and the 2020-2025 era have different dominant oscillations. LPPL₃ is rich
enough to capture both simultaneously, which is why its frequencies are
stable under both weightings.

## Fitting pipeline

All LPPL fits use scipy `differential_evolution` with `seed=42` and
`workers=1` (multiprocessing serialization fails on local closures).

**Unweighted variants** are refitted daily via
[`update_prices.py`](../update_prices.py):

- [`tools/fit_lppl.py`](../tools/fit_lppl.py) — LPPL₁
- [`tools/fit_lppl2.py`](../tools/fit_lppl2.py) — LPPL₂
- [`tools/fit_lppl3.py`](../tools/fit_lppl3.py) — LPPL₃
- [`tools/fit_lppl4.py`](../tools/fit_lppl4.py) — LPPL₄ (with optional `--no-13`)

**Weighted and ω=13-excluded variants** are refitted monthly on the
production server via a systemd timer running
[`tools/fit_lppl_variants.py`](../tools/fit_lppl_variants.py). Fits all 6
variants in ~15-25 minutes, then flushes Redis and restarts quantoshi.

## UI: LPPL Models config panel

Tab 1 has a dedicated "LPPL Models" card between Bubble Model and Display
Models sections:

- **Number of frequencies**: Checkboxes 1/2/3/4, multi-select, default [3]
- **Log-time weighted fits**: Global toggle (swaps regular ↔ weighted
  variants for every checked order)
- **Exclude ω≈13 intermod**: Toggle that disables LPPL₃ (its ω≈10 is close
  to the excluded band) and swaps LPPL₄ for LPPL₄ₙ₁₃
- **LP4 warning**: A confirm dialog fires every single time LPPL₄ is
  toggled on, linking to this documentation

## Model Info accordion reference

The following accordion items in the Model Info tab document the LPPL family:

- `/7.4` Log-Periodic Power Law (LPPL)
- `/7.5` LPPL₂ (Two-Frequency)
- `/7.6` LPPL Weighting & Regime Shifts — most comprehensive discussion
