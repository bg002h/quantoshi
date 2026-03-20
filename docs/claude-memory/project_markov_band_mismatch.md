---
name: Markov MC uses QR bands, not shrinking Gaussian
description: MC simulation still uses old QR straight-line fits for bin classification, not the new shrinking Gaussian bands — known architectural mismatch requiring Cython recompile + cache rebuild to fix
type: project
---

After the ShrinkingGaussian change, the Markov MC simulation still uses `M.qr_fits` (QR straight-line fits with intercept/slope) for regime bin classification. The new shrinking Gaussian bands (`_CompositeModel.price_at`) are only used by chart projections (DCA/retire/heatmap/bubble).

**Why:** The Cython `markov.py` module directly reads `fits["intercept"]` and `fits["slope"]`. The transition matrix was trained on QR-based bin boundaries. Switching bins requires retraining + cache rebuild (~834 MB).

**How to apply:** To make MC use shrinking Gaussian bands:
1. Modify Cython `markov.py` to accept a `price_at` callable instead of a fits dict
2. Retrain transition matrices with new bin definitions
3. Rebuild the entire MC cache
4. Recompile the Cython module

**Broader goal: make Markov code model-agnostic.** Currently the Cython module
is tightly coupled to the QR fits format — it hardcodes `fits["intercept"]` and
`fits["slope"]` to compute bin boundaries. The right fix is to decouple:
- Markov should accept a `price_at(q, t) → float` callable for bin edge computation
  instead of a fits dict with a specific internal format
- This lets any PriceModel (QR, composite, shrinking Gaussian, future models)
  drive MC simulation without Cython changes
- The transition matrix training and the simulation inference should both use
  the same `price_at` callable for consistency
- The MC cache would need to be keyed by model (not just parameters) since
  different models produce different bin boundaries → different transition matrices
- Performance consideration: `price_at` via numpy interp is ~10x slower than
  direct intercept+slope math. May need to pre-compute bin edges at all needed
  time points and pass as arrays instead of calling per-step

This is a separate project — do not attempt as part of other model changes.
