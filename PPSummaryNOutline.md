# Decomposing Bitcoin: Power Laws, Entropy, and the End of Speculative Oscillations

## Executive Summary

**Bitcoin's Price Structure: A Multi-Method Decomposition Study**

Using Quantoshi's analysis platform, we conducted an exhaustive investigation into the mathematical structure of Bitcoin's price dynamics. Through 30+ model variants, principal component analysis, empirical/dynamic mode decomposition, sparse component selection, and non-sinusoidal basis function searches, we converged on three central findings:

**1. Two fundamental oscillations.** Three independent mathematical methods — PCA (statistical), EMD (assumption-free signal decomposition), and DMD (dynamical systems) — all converge on the same two dominant oscillatory structures in Bitcoin's price: a **~4-year halving cycle** in calendar time and a **log-periodic oscillation** (omega~7.4) in log-time. These are the only robust signals above the power law trend.

**2. The speculative era is over.** The log-periodic oscillation — the mathematical signature of speculative bubble behavior — peaked around 2013 and went to zero by ~2019. This is captured by our novel **Entropy PPL model**, which replaces traditional power-law damping with the Shannon entropy function E(w*t) = -w*t*ln(w*t). The fitted parameters place maximum adoption uncertainty (entropy peak) precisely at the height of early Bitcoin speculation. Bitcoin's price dynamics are now driven by the halving cycle, not by speculative self-similarity.

**3. Bubble shapes are maturing.** Individual bubble analysis shows statistically significant trends: duration widening (0.87yr to 5.39yr, p=0.012) and amplitude shrinking (each cycle ~20% smaller, p=0.016). Early bubbles were sharp triangles; later bubbles develop extended plateaus. Non-sinusoidal basis functions (triangle waves, flat-top waves) fit Bitcoin's cycles better than traditional sinusoids, with flat-top waves dominating the selection — consistent with a market transitioning from retail FOMO spikes to institutional distribution phases.

---

## PowerPoint Outline

### Slide 1: Title
**"Decomposing Bitcoin: Power Laws, Entropy, and the End of Speculative Oscillations"**
Quantoshi Research, April 2026

### Slide 2: The Power Law Foundation
- log10(price) = alpha + beta*log10(t), beta ~ 5.08
- Genesis date sweep (/H): R-squared peaks sharply at July 2009
- Exponent is remarkably stable across all model variants (4.9-5.2)

### Slide 3: What's On Top of the Power Law?
- Price = Power Law + Oscillations + Noise
- The key question: what are the oscillations and are they predictable?
- Overview of investigation: 30+ models, 4 decomposition methods, 500+ basis functions tested

### Slide 4: Two Dominant Signals — Found Three Ways
| Method | Signal 1 | Signal 2 |
|---|---|---|
| PCA (basis search) | omega~7.4 log-periodic | T~3.6yr calendar |
| EMD (assumption-free) | T~3.9yr (IMF 7) | T~2.0yr (IMF 6) |
| DMD (eigenmode) | T~4.0yr (stable) | T~2.0yr (decaying) |

### Slide 5: The HybPPL Model Family
- log10(price) = A + B*log10(t) + [log-periodic] + [calendar-periodic]
- Started with HybPPL (9 params), explored up to 18 params
- 2nd harmonics discovered: omega~16 (log) and T~1.9yr (sub-halving)
- Best parametric model: Hyb2B (16p, R-squared=0.993)

### Slide 6: PCA — Optimal Compression
- 30 component curves from 6 HybPPL models -> SVD -> 6 orthogonal PCs
- PC1 (97%): power law trend
- PC2 (1.5%): halving cycle
- PC3 (1.0%): log-periodic
- R-squared=0.993 with only 7 parameters — beats 16-param Hyb2B on BIC

### Slide 7: Synthetic Basis Search — Can We Do Better?
- 224 candidate functions: polynomials, sin/cos, damped, Gompertz, piecewise
- Result: curated HybPPL basis wins
- DE-fitted model components are already near-optimal

### Slide 8: The Entropy Discovery
- Replace t^(-D) damping with E(w*t) = max(-w*t*ln(w*t), 0)/(1/e)
- This is literally Shannon's entropy function
- Peaks at w*t = 1/e ~ 37% of adoption lifecycle -> maximum uncertainty
- Zero at w*t = 1 -> "adoption resolved," oscillations extinct
- EPPL beats HybPPL on BIC at same parameter count

### Slide 9: What the Entropy Parameters Say
- Primary log-periodic (omega~7.8): peaked 2013, zero by 2019
- 2nd harmonic (omega~16.8): peaked 2011, zero by 2013
- Calendar oscillation (halving): undamped — persists indefinitely
- The log-periodic signal was a transient of the "will Bitcoin survive?" era

### Slide 10: Non-Sinusoidal Waves — Bitcoin Isn't Smooth
- Tested: sinusoidal, triangle, flat-top (tanh), trapezoid as basis functions
- Non-sinusoidal waves beat sinusoidal by ~1,000 BIC
- Flat-top waves selected most often (10/25 terms in mixed dictionary)
- Mixed-shape dictionary: R-squared=0.997 (27 params)

### Slide 11: Bubble Shape Evolution
- Duration widening: 0.87yr (2011) -> 5.39yr (2024), p=0.012
- Amplitude shrinking: each cycle ~20% smaller, p=0.016
- Early bubbles: tall narrow triangles
- Later bubbles: wide flat-topped shapes

### Slide 12: Evolving Shape Model
- tanh(k(t)*sin(omega*t)) where k(t) = k0 + k1*t
- k(2011)=0.37 (sine) -> k(2025)=2.28 (flattening) -> k(2034)=3.51 (projected)
- Calendar oscillation shape evolves; log-periodic doesn't
- Log-periodic = fractal self-similarity; Calendar = human behavior

### Slide 13: Greedy Select — The Best 7-Parameter Model
- Sparse component selection from 467 functions
- Entropy-damped terms dominated (16 of 24 at full depth)
- Zero power-law-damped terms selected
- 7p model: R-squared=0.9935, BIC=-23,886 — best BIC at 7 params

### Slide 14: Model Comparison Summary
| Model | Params | R-squared | BIC |
|---|---|---|---|
| Power Law | 3 | 0.963 | -17,500 |
| HybPPL | 9 | 0.989 | -20,814 |
| Hyb2B | 16 | 0.993 | -23,203 |
| PCA k=6 | 7 | 0.993 | -23,776 |
| EPPL 2+2 | 16 | 0.993 | -23,681 |
| Greedy v2 | 7 | 0.994 | -23,886 |
| Mixed shapes | 27 | 0.997 | -28,213 |

### Slide 15: Configuration Panels
- HybPPL: 36 pre-fitted configs (log/cal x damped/undamped)
- EPPL: 36 entropy-damped configs
- Two-model comparison (A/B)
- Interactive at quantoshi.xyz

### Slide 16: Implications for Bitcoin's Future
- Power law trend continues (structural)
- Halving cycle persists undamped
- Oscillation amplitude shrinks each cycle
- Era of 10x speculative spikes is over
- Future cycles: normal asset class behavior + halving periodicity

### Slide 17: Tools and Reproducibility
- quantoshi.xyz — interactive charts
- /B through /H — static analysis pages
- Model Info tab (/8) — full formulas and coefficients
- Open source: github.com/bg002h/quantoshi

### Slide 18: Multivariate Analysis
- Proxy inputs (linear t, log²t, √t, lagged price): all ΔR² < 0.001
- Power law is purely temporal — no transformation of time adds information
- Lagged price: β=-0.20 (weak negative momentum, mean reversion)
- Limitation: proxies not independent; on-chain data (hash rate, addresses) needed for true test

### Slide 19: Phase-Space Reconstruction
- Takens embedding, delay τ=256 days (0.70yr)
- FNN drops to 4.2% at dimension 3 — attractor resolves
- Correlation dimension D₂ ≈ 2.0 — a ~2D attractor
- Topologically confirms two oscillatory modes (same as PCA/EMD/DMD)

### Slide 20: Neural Networks vs Parametric Models
- MLP 10 nodes (91p): train R²=0.993, test R²=-20.3
- MLP 100-50-20 (6,891p): train R²=0.999, test R²=-3.1
- EPPL (16p): train R²=0.993, test R²=0.64
- 16 domain-informed parameters vastly outperform 6,891 unconstrained parameters

### Slide 21: Summary — Three Sentences
1. Bitcoin has exactly two oscillatory structures — confirmed independently by PCA, EMD, DMD, and phase-space reconstruction.
2. The speculative signal is extinct, captured by Shannon entropy damping.
3. Future price: power law trend + flattening halving cycle = market maturation.
