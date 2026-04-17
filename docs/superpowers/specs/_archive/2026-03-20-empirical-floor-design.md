# BM Empirical Floor Model — Design Spec

## Summary

Add a new price model ("BM Empirical Floor") to Quantoshi. It uses a two-point
power law support line (slope=5.3106, intercept=-1.6246) drawn through the
2010-10-05 price floor and the 2026-02-09 local minimum ($70,339). Bubble
shapes are fitted above this steeper support. The composite (support + bubbles)
serves as the median; Gaussian-shifted copies create quantile bands. The model
produces distinct projections on all tabs (heatmap, DCA, retire, supercharger)
reflecting faster bubble convergence — the "end of the 4-year cycle" narrative.

## Motivation

The standard Bubble Model (pct=20, q=0.50) produces a support line with
slope ~5.13. Its below-line points cluster in just 2–3 bear-market eras
(KS stat 0.581 — far from the uniform ideal of 0). We searched for a support
line whose below-line points are distributed more evenly across time.

The first anchor point (2010-10-05, $0.06) is the end of Bitcoin's initial
flat-price run — the earliest observable floor. Several candidates were tested
for the second anchor: the Dec 2022 bear bottom ($16,902, KS=0.580 — no
improvement), the Feb 2026 local minimum at actual price ($64,114, KS=0.403),
and the Feb 2026 date at $70,339 (KS=0.247 — best temporal distribution).
The $70,339 value was chosen because it produces the most uniform distribution
of below-line points across 10 equal time bins, meaning the support line is
equally relevant in early, middle, and recent Bitcoin history rather than
being an artifact of one or two crash eras.

The resulting Empirical Floor has slope ~5.31 — steeper than the standard
model, which means:

- Bubbles sit lower above the floor (smaller log-excess)
- The decay trend in bubble amplitudes is steeper
- Future predicted bubbles converge on the support faster
- This implies diminishing volatility and the approaching "end" of the
  classic 4-year halving-driven boom/bust cycle

Statistical comparison (Q% = fraction of historical data below the line):

| Support Line | Slope | Below | KS stat | Bin std |
|---|---|---|---|---|
| Standard BM (pct=20, q=0.50) | 5.125 | 10.0% | 0.581 | 0.187 |
| Empirical Floor (two-point) | 5.311 | 22.9% | 0.247 | 0.267 |

The EF line has dramatically better temporal distribution (KS 0.247 vs 0.581)
— below-line points appear across most eras instead of clustering in 2–3 bear
markets. R² with bubble fitting: 0.9932 (essentially tied with standard BM's
0.9921).

## Architecture

### Data generation: `tools/build_ef_model.py`

Standalone script (no notebook dependency). Generates `btc_app/model_data_ef.pkl`.

**Pipeline:**
1. Load `BitcoinPricesDaily.csv`
2. Use hardcoded EF support: slope=5.3106, intercept=-1.6246, genesis=2009-07-25
3. Compute log-excess above support for all price data
4. Locate bubble peaks (same BUBBLE_YEARS as notebook: [2011, 2013, 2017, 2021, 2025])
5. Fit bubble shapes sequentially (largest first, residual subtraction)
6. Build composite curves for N=1..n_future_max future bubbles
7. Compute σ = std of (log10(actual_price) - log10(composite)) residuals
8. Export pkl containing:
   - `ef_support_slope`, `ef_support_intercept`
   - `years_plot_bm` (x-axis grid)
   - `support_bm` (support line values on grid)
   - `comp_by_n` (composite curves for N future bubbles)
   - `bm_r2` (composite R²)
   - `n_future_max`
   - `sigma` (residual std for Gaussian bands)
   - `price_years`, `price_prices` (historical data)
   - Fitted bubble parameters (for decomposition plots)

**Usage:**
```bash
btc_venv/bin/python3 tools/build_ef_model.py [--out btc_app/model_data_ef.pkl]
```

### Model class: `EmpiricalFloorModel` in `btc_core.py`

Not a `_FitsBasedModel` subclass (those assume log-linear fits per quantile).
Instead, similar pattern to `LPPLModel`:

```python
class EmpiricalFloorModel:
    name = "BM Empirical Floor"
    short_name = "ef"
    dash_style = "dashdot"
    quantized = True  # has quantile bands → works on all tabs

    def __init__(self, pkl_path):
        # Load ef pkl, build quantile bands from composite + sigma
        # Generate fits dict: {q: {"composite_curve", "z_shift"}}
        # Build color ramp (amber/orange tones)

    def price_at(self, q, t):
        # Interpolate composite curve at time t, shift by z_q * sigma
        # Return 10^(log10(composite(t)) + z_shift)

    def interp_price(self, q, t):
        # Same as price_at for arbitrary quantiles

    def find_percentile(self, t, price):
        # Inverse: given price and time, find which quantile
```

Key difference from `_FitsBasedModel`: `price_at` interpolates a shaped curve
(not a straight line in log-log), so bands follow the bubble humps.

### Model registration: `app.py`

```python
ef_pkl = Path(__file__).parent.parent / "btc_app" / "model_data_ef.pkl"
if ef_pkl.exists():
    _app_ctx.PRICE_MODELS["ef"] = EmpiricalFloorModel(str(ef_pkl))
```

Conditional load — model appears only if pkl exists. Clean removal = delete pkl.

### UI integration (auto-discovered)

Existing `PRICE_MODELS` iteration handles:
- **Bubble tab:** `bub-model-show` checklist gains "BM Empirical Floor" toggle
- **Heatmap tab:** pill bar gains "BM Empirical Floor" button
- **DCA/Retire/Supercharger:** `{prefix}-model-show` checklist gains EF option
- **Snapshot/Share:** automatically included (PRICE_MODELS keys are snapshot-safe)

No layout changes needed if the auto-discovery pattern is correctly implemented
for the existing models. Verify that all model iteration points handle arbitrary
models, not just hardcoded "bub"/"pl"/"s2f" checks.

### Figure builders

The figure builders (`figures/bubble.py`, `figures/heatmap.py`, etc.) call
`model.price_at(q, t)` to get prices. The EF model's `price_at` returns
composite-based prices instead of log-linear ones. This should work
transparently as long as `price_at` returns the same types (scalar or ndarray).

**Bubble chart specifics:** The EF composite curve and future bubble predictions
need to be drawn as overlay traces, similar to how PL/S2F are drawn. The
bubble builder may need to handle models that have `comp_by_n` (composite
curves) vs models that are just quantile lines.

### Website content

**Model Info tab (tab 7) — new accordion item:**
- Title: "BM Empirical Floor"
- Brief methodology: two anchor points (2010-10-05 $0.06, 2026-02-09 $70,339),
  slope 5.31, R² 0.9932
- Convergence argument: steeper support → bubbles converge faster → end of
  4-year cycle
- Link to FAQ for full analysis

**FAQ (tab 8) — new entry:**
- Title: "What is the BM Empirical Floor model?"
- Full derivation story: support line sweeps, KS temporal uniformity analysis,
  comparison table, what "end of 4-year cycle" means (diminishing bubble
  amplitude, mature asset behavior)
- Include comparison chart (support_4way_loglog.jpg or similar)

### Documentation updates

- `docs/architecture.md`: Add EF to model table, document `model_data_ef.pkl`,
  add "How to add a new price model" checklist section
- `docs/user_manual.md`: Mention EF in models section
- `CLAUDE.md`: Add EF model to relevant sections

### Color ramp

Amber/warm tones to distinguish from the blue bubble model and suggest
"convergence warmth":
- Low quantiles: dark amber (#8B6914)
- Mid quantiles: warm orange (#D4881A)
- High quantiles: bright gold (#F0C040)

### Deletion path

To remove the model in the future:
1. Delete `btc_app/model_data_ef.pkl`
2. Remove `EmpiricalFloorModel` class from `btc_core.py`
3. Remove import + registration from `app.py`
4. Remove Model Info and FAQ entries
5. The conditional `if ef_pkl.exists()` in registration means step 1 alone
   disables the model without code changes.

## Out of scope

- MC (Monte Carlo) overlay for the EF model — future work if needed
- Modifying the standard Bubble Model's parameters
- Changing the genesis date
