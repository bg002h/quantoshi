# R² on All Legend Traces (Tab 1) — Design Spec

## Summary

Display per-quantile R² values in the legend label of every model trace on the bubble chart where a goodness-of-fit metric is computable. Computed at app startup from historical price data.

## Scope

- Tab 1 (Bubble + QR Overlay) only.
- All model classes in `btc_core.py`.
- Legend labels in `figures/bubble.py`.

## R² Computation

### Formula

Standard coefficient of determination in log-space:

```
R² = 1 - Σ(log10(actual_i) - log10(predicted_i))² / Σ(log10(actual_i) - mean(log10(actual)))²
```

Where `predicted_i = model.price_at(q, t_i)` for each historical data point `(t_i, price_i)`.

### Shared helper

Add to `btc_core.py`:

```python
def _compute_log_r2(actual_prices, predicted_prices):
    """R² in log10 space. Returns float or None if degenerate."""
    log_a = np.log10(np.maximum(actual_prices, 1e-10))
    log_p = np.log10(np.maximum(predicted_prices, 1e-10))
    ss_res = np.sum((log_a - log_p) ** 2)
    ss_tot = np.sum((log_a - np.mean(log_a)) ** 2)
    if ss_tot == 0:
        return None
    return float(1.0 - ss_res / ss_tot)
```

### Per-model computation

Each model stores `r2_per_quantile: dict[float, float]` — mapping quantile → R² value.

Implemented as a **standalone function** (not a method) since the model classes have no common base:

```python
def compute_model_r2(mdl, price_years, price_prices):
    """Compute per-quantile R² for any model with price_at() and quantiles."""
    mdl.r2_per_quantile = {}
    mask = price_years >= 1.0  # skip very early data
    t = price_years[mask]
    actual = price_prices[mask]
    if hasattr(mdl, 'quantiles') and mdl.quantiles:
        for q in mdl.quantiles:
            predicted = mdl.price_at(q, t)
            r2 = _compute_log_r2(actual, predicted)
            if r2 is not None:
                mdl.r2_per_quantile[q] = r2
    elif hasattr(mdl, 'price_at'):
        # Non-quantized (S2F): single trajectory at q=0.5
        predicted = mdl.price_at(0.5, t)
        r2 = _compute_log_r2(actual, predicted)
        if r2 is not None:
            mdl.r2_per_quantile[0.5] = r2
```

This handles all model classes uniformly:
- `_FitsBasedModel` subclasses: QR, PL (have `quantiles`)
- `_CompositeModel` subclasses: BM, EF (have `quantiles`)
- Standalone classes: LPPL, Exp (have `quantiles`)
- Non-quantized: S2F (no `quantiles`, single trajectory)

### When it runs

Called from `app.py` after model registration, passing `(m.price_years, m.price_prices)`:

```python
for mdl in _app_ctx.PRICE_MODELS.values():
    compute_model_r2(mdl, m.price_years, m.price_prices)
```

BubbleModel and EmpiricalFloorModel already have `bm_r2` (composite R²) — that stays unchanged. The new `r2_per_quantile` is separate (per-quantile, against raw price data).

## Legend Labels

### Format

Append `  R²=X.XXXX` (4 decimal places, Unicode ²) to any trace name where R² is available.

### Trace-by-trace changes (`figures/bubble.py`)

| Trace | Current label | New label | R² source |
|-------|--------------|-----------|-----------|
| BM quantile lines | `Q50%` | `Q50%  R²=0.9512` | `model.r2_per_quantile[q]` |
| BM composite | `Bubble composite (N=3)  R²=0.9847` | unchanged | `m.bm_r2` (already shown) |
| BM support | `Bubble support` | unchanged | N/A |
| Overlay quantile lines | `Power Law Q50%` | `Power Law Q50%  R²=0.9501` | `mdl.r2_per_quantile[q]` |
| Overlay composite (EF) | `BM Empirical Floor composite (N=3)  R²=0.9834` | unchanged | `mdl.bm_r2` (already shown) |
| Overlay support (EF) | `BM Empirical Floor support` | unchanged | N/A |
| Non-quantized overlay (S2F) | `Stock-to-Flow` | `Stock-to-Flow  R²=0.8721` | `mdl.r2_per_quantile[0.5]` |
| OLS line | `OLS` | `OLS  R²=0.9503` | Compute from `m.ols_intercept`, `m.ols_slope` against price data |
| UCL | `Unfairly Cheap Line` | unchanged | Reference line, not fitted |
| Scanner lines | `{Model} Q38.8%` | unchanged | Arbitrary quantile — skip R² (not precomputed) |
| Data scatter | `Price data` | unchanged | Not a model |
| Shade fills | (no legend) | unchanged | N/A |

### OLS R² (special case)

OLS is not a registered PriceModel — it's drawn directly from `m.ols_intercept` and `m.ols_slope`. Compute OLS R² once at startup and store on ModelData (`m.ols_r2`), or compute inline in the figure builder. Since it's a single computation, inline is fine:

```python
predicted = 10 ** (m.ols_intercept + m.ols_slope * np.log10(price_years))
ols_r2 = _compute_log_r2(price_prices, predicted)
```

Store as `m.ols_r2` computed in `app.py` after model loading.

## Files Modified

| File | Change |
|------|--------|
| `archive/btc_app/btc_core.py` | Add `_compute_log_r2()` helper; add `compute_r2()` method + `r2_per_quantile` to `_FitsBasedModel`, `_CompositeModel`, `S2FModel` |
| `btc_web/figures/bubble.py` | Append R² to legend labels for: BM quantile lines, overlay quantile lines, non-quantized overlays, OLS line, scanner lines |
| `btc_web/app.py` | Call `mdl.compute_r2()` for all models after registration; compute `m.ols_r2` |

## What doesn't get R²

- Support lines (geometric construction, not a regression)
- UCL (fixed reference line, not a fit)
- Data scatter points
- Shade fills (no legend entry)
- Composite curves (already show R² from the composite fit)

## Performance

~100 vectorized numpy operations on ~5,500 elements each = ~15-30ms total at startup. Negligible vs. existing startup costs (Markov cache, figure prewarm).

## Out of Scope

- R² on tabs 2-8
- Caching R² in pkl files
- Changing R² decimal precision (4 dp matches existing composite labels)
