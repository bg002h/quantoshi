# Quantoshi Research Summary — April 2026

## Investigations Conducted

### Models Registered (11 new)
1. **Gompertz** (gomp) — S-curve saturation, R²=0.933, K=$77K carrying capacity
2. **Broken Power Law** (bpl) — two-segment, breakpoint 2016-04, slope increased 4.92→5.32
3. **HybPPL +2L** (hyb2l) — +2nd log-periodic, 13p, R²=0.991
4. **HybPPL +2C** (hyb2c) — +2nd calendar (sub-halving T≈1.9yr), 12p, R²=0.991
5. **HybPPL +2B** (hyb2b) — +both 2nd harmonics, 16p, R²=0.993
6. **HybPPL 4D** (hyb4d) — all damped, 18p, R²=0.992 (educational counterexample)
7. **PCA** (pca) — 6 PCs from HybPPL basis, 7p, R²=0.993
8. **Greedy Select v2** (grdy) — entropy-damped sparse selection, 7p, R²=0.994, best BIC at 7p
9. **Entropy PPL** (eppl) — Shannon entropy damping, 16p, R²=0.993
10. **HybPPL Config** — 36 pre-fitted configurations (all log/cal/damping combos)
11. **EPPL Config** — 36 entropy-damped configurations

### Decomposition Methods (4)
1. **PCA** — 30 components from 6 models → 6 PCs explain 99.97%
2. **EMD** — 8 IMFs, 3 IMFs give R²=0.992 with zero assumptions
3. **DMD** — eigenmodes: ~4yr (stable), ~2yr (decaying)
4. **Sparse Component Selection** — greedy BIC minimization from 467 functions

### Basis Function Research
- **Sinusoidal vs non-sinusoidal**: triangle/flat-top beat cosine by ~1,000 BIC
- **Mixed shapes** (27p): R²=0.997 — flat-top dominates (10/25 terms)
- **FFT basis**: needs k=24 to match curated k=6 — pre-fitted components are near-optimal
- **Polynomial basis**: useless for oscillatory structure (R²=0.964 max)
- **Synthetic dictionary** (224 functions): curated HybPPL basis wins

### Damping Envelope Research
- **Entropy -w·t·ln(w·t)**: best damping function, beats t^(-D) on BIC
- **(w·t²)·ln(w·t)**: 2nd best, peaks later
- **(w·t^0.5)·ln(w·t)**: peaks too early, worst
- **Entropy dominates greedy selection**: 16/24 terms entropy-damped, 0 power-law-damped

### Bubble Analysis
- **Shape evolution**: duration widening (p=0.012), amplitude shrinking (p=0.016)
- **Triangle → trapezoid transition**: early bubbles sharp, later ones plateau
- **Evolving shape model**: tanh(k(t)·sin) with k increasing over time
- **Two-channel model**: major (herd) + minor (whale) — minor lags by 13 months

### Genesis Date Investigation
- **521-date sweep** (2007-2016): best R²=0.963 at July 12, 2009
- **7 independent statistical tests** converge on June-August 2009
- **WLS (log-uniform)**: optimal shifts to May 2007 — genesis is weighting-dependent
- **Floor-to-peaks**: all percentiles follow β≈5.02-5.14

### Regime Analysis
- **Box-Cox λ=0.012**: log-log is the correct form
- **Rolling regression**: slope oscillates but mean-reverts to ~5.0
- **Bai-Perron**: 6 breaks, all at bubble peaks/troughs — cycle boundaries, not regime shifts
- **Conclusion**: single power law regime

---

## Novel Findings (April 2026)

### 1. Time-Varying Frequency
ω₁ = -0.056 ≈ 0 — **the log-periodic frequency is constant**, not drifting. Validates fixed-ω assumption.

### 2. Heteroscedastic Volatility
σ(t) = 0.077·t^(-0.19) — volatility shrinks 1.5× from early to current era. Windowed σ: 0.150 (2010-13) → 0.092 (2021-26).

### 3. Cross-Validation
All models overfit (train R²≈0.99, test R²≈0.64). **EPPL 1d+1u is the most consistent generalizer** across 5 cutoff dates — entropy envelope prevents projecting dead oscillations forward.

### 4. Symbolic Regression
**PySR independently discovers the power law**: log₁₀(price) = 2.174·ln(t) - 1.097 = 5.01·log₁₀(t) - 1.10. Genesis date is an input assumption, not discovered.

### 5. Wavelet Scalogram
**Halving cycle (T≈3.6yr) is the overwhelmingly dominant signal** across all scales. Early era power is 2× late era — oscillations fading. Log-periodic invisible in wavelets because it's a chirp in calendar time.

### 6. Asymmetric Oscillations
**Falls damp 3.5× faster than rises** (D_rise=0.56, D_fall=2.0). FOMO builds slowly, panic resolves quickly. BIC improves by 120.

### 7. Quantile-Specific Oscillations
**Upper quantiles oscillate 4× larger** than floor (C_log: Q5%=0.07, Q95%=0.27). Floor slope steeper (β=5.41) than peaks (β=4.87) — volatility compression confirmed.

### 8. Fourier Extrapolation
Top 20 modes project $724K (2027), $85K (2028), $677K (2030). Not reliable — no damping — but shows cyclical structure.

### 9. Multi-Scale Decomposition
**Power law is self-similar**: β≈5.08 at daily, 5.09 weekly, 5.12 monthly, 5.18 quarterly, 5.34 yearly. R²>0.96 at every resolution.

---

## Key Conclusions

1. **Two oscillatory structures**: log-periodic (ω≈7.4, fractal) + calendar halving (~3.6yr, human behavior) — confirmed by PCA, EMD, DMD, wavelet, and sparse selection.

2. **The speculative era ended ~2019**: Shannon entropy damping peaks at 2013 and goes to zero by 2019. The log-periodic signal is extinct.

3. **Entropy damping is the preferred envelope**: dominates greedy selection (16/24 terms), beats power-law on BIC, and generalizes better out-of-sample.

4. **Bubble shapes mature**: duration widens, amplitude shrinks, triangles → trapezoids. Consistent with growing participant base.

5. **The power law is robust**: single regime (Box-Cox λ≈0), self-similar across scales, slope stable across 30+ model variants (β≈5.0-5.1).

6. **All models overfit recent data**: test R²≈0.64 regardless of model complexity. Post-2023 ETF-driven dynamics don't follow historical oscillatory patterns.
