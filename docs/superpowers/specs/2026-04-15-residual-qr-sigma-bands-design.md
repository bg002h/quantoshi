# Residual QR σ Bands (Tab 1) — Design Spec

**Status:** ✅ Brainstorming complete. All 6 sections approved, 4 rounds of Opus code review, 30+ issues caught and fixed before implementation.
**Spec date:** 2026-04-15
**Next step:** `superpowers:writing-plans` to generate the implementation plan.

---

## Problem statement

Quantoshi's Tab 1 (Bubble chart) currently renders quantile bands for parametric price models (BubbleModel, PowerLaw, HybPPL family, EntropyPPL, LinPPL, Exp, PCA, Greedy, Gompertz, BPL, and LPPL family) via `median(t) ± z(q) · σ_const`. These bands are **homoscedastic**: band width is constant in log-price space across all t, despite the empirical residual variance being both non-monotonic in t AND asymmetric between upside/downside residuals.

A prior attempt to replace constant σ with a parametric `σ(t) = σ₀·t^(-α)` was reverted (commit `b449539`) because the power-law form forced σ near-zero at t ≈ 16 yr, causing the current Bitcoin price to register as Q99%. Empirical windowed σ is non-monotonic (0.150 → 0.109 → 0.139 → 0.092 across 4-year windows from 2010 to 2026) and no rigid parametric form captures this.

## Solution summary

Replace the shrinking-σ infrastructure with **direct residual quantile regression** — a piecewise-linear-in-log10(t) QR fit on each model's residuals, per target quantile, with knots at t=3/6/9/12 years. User-facing as a **2-option radio** in the Display Models panel on Tab 1: `Constant σ` (legacy) or `Residual quantile` (new). Scope is **Tab 1 only** (Bubble chart + Model Scanner). Other tabs always use Constant σ.

## Locked decisions (from brainstorming 2026-04-15)

### UX scope
- Tab 1 only. Display Models panel on Tab 1 gains a new "σ mode" section with a horizontal radio.
- **Scope warning:** static disclaimer below the radio reads *"σ mode affects Tab 1 only (Bubble chart + Model Scanner). Other tabs always use Constant σ."*
- **LPPL caveat:** second static disclaimer reads *"LPPL family and models without fitted residuals always use Constant σ."*
- **Default mode:** `"constant"` (backward compatible — no visual change until user explicitly flips).
- **Share link persistence:** yes (`bub-sigma-mode` in `_SNAPSHOT_CONTROLS` + `_TAB_CONTROLS["bubble"]`).
- **localStorage persistence:** no. Reload = reset to default. Matches existing app behavior.

### Model scope
**In scope (14 parametric models gain residual-QR bands):** `bub`, `pl`, `hybppl`, `hyb2l`, `hyb2c`, `hyb2b`, `hyb4d`, `eppl`, `linppl`, `exp`, `pca`, `grdy`, `gomp`, `bpl`.

**Explicitly excluded:**
- LPPL family (10 classes: `lppl`, `lp2`, `lp3`, `lp4`, `lppl_w`, `lp2_w`, `lp3_w`, `lp4_w`, `lp4_n13`, `lp4_w_n13`) — residuals carry a 3.92-yr halving-cycle periodicity (32% of spectral power) that piecewise-linear basis cannot represent. Coverage is still nominal but bands would be systematically wrong in shape.
- `u1` (User Model) — two-point anchored, session-only, no residuals.
- `s2f` (Stock-to-Flow) — different x-axis semantics.
- `ef` (Empirical Floor) — fixed-anchor support line, not a fit.
- `mc` / `markov` — paid feature, different code path entirely.

### Fit methodology
- **Basis:** piecewise-linear in log10(t) with knots at `(3.0, 6.0, 9.0, 12.0)` years.
  Basis vector: `[1, log10(t), relu(log10(t)-log10(3)), relu(log10(t)-log10(6)), relu(log10(t)-log10(9)), relu(log10(t)-log10(12))]` — 6 columns.
- **Quantiles:** `(0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99)` — 8 total. Q50% uses each model's own median (not fitted via residual QR).
- **Minimum samples:** 500 (else `ValueError`).
- **Residual computation:** `r(t) = log10(price) − model_median_log10(t)`, filtering `t > 1.0` yr.
- **Validation:** 80/20 random holdout (seed=42) at build time. Out-of-sample coverage must be within ±5pp of nominal for `Q05, Q25, Q75, Q95` (tails Q01/Q99 excluded from the tolerance check due to small-N noise).
- **Monotonicity enforcement:** post-sort across quantiles at query time (`np.sort(offsets, axis=1)`) as a belt-and-suspenders correction for numerical wiggle in the interior. Raw crossing fraction is logged as a diagnostic at build time; warning if >5%.
- **Extrapolation control:** query-time clipping `t_clipped = np.minimum(t, knots[-1])` ensures the basis is evaluated at t=12 for all t≥12. Bands produce a constant-log-offset plateau past the last knot — no slope drift, no crossings by construction. **The fit-time `coefs[-1] = 0.0` freeze considered in earlier drafts is NOT used** — the query-time clip is necessary and sufficient.

### Fit-failure policy
- **Policy A (per-model skip, build continues):** `ValueError` from `fit_residual_qr_pwl` — too few samples, NaN residuals, QuantReg solver non-convergence on any individual quantile. The failing model's key is absent from `resqr_coefs`; it falls back to constant σ at runtime.
- **Policy B (global abort):** triggered by any of the following:
  1. **Non-finite fitted coefficients** from an otherwise-successful `QuantReg.fit` (explicit post-fit `np.isfinite(coefs).all()` check in `fit_residual_qr_pwl`, raises `RuntimeError`).
  2. **BubbleModel (`bub`) failure** — the flagship. Cannot ship a pkl with broken BM bands.
  3. **More than 50% of in-scope models failing** fit or coverage assertion — signals a methodology-wide regression.
- **Relaxed from earlier draft:** single-model coverage failures (other than `bub`) no longer trigger global abort. Rationale: daily cron could otherwise flip one borderline model and kill the entire update chain for weeks. Instead, the failing model is demoted to Policy A (skip), surfaces via `/health`'s `resqr_bands.model_count < 14`, and triggers the staleness alert after 3 days.

## Architecture

```
BUILD TIME (tools/build_bm_model.py, runs nightly via daily_update.sh)
  │
  ├─ [existing] fit QR, BM composite, per-model coefficient arrays
  ├─ [existing] write core keys to model_data.pkl
  │
  ├─ [NEW] for each model_key in RESQR_MODELS (14 models):
  │     1. Instantiate model class from raw fit arrays (NO pkl read — Option C)
  │     2. Compute median: log10_median = model._model_log10(price_years)
  │     3. residuals = log10(price) − log10_median
  │     4. Pre-flight: np.isfinite(residuals).all()
  │     5. sorted_qs, coef_matrix, coverage, raw_crossings =
  │            fit_and_validate(t, residuals, model_key)
  │     6. Policy A: on ValueError → log warning, skip model
  │        Policy B: on RuntimeError → propagate, check global abort conditions
  │
  ├─ [NEW] m['resqr_coefs'][model_key] = (sorted_qs, coef_matrix)
  │        m['resqr_models'] = frozenset(resqr_coefs.keys())
  │        m['resqr_build_ts'] = ISO-8601 UTC timestamp
  │        m['resqr_knots'] = (3.0, 6.0, 9.0, 12.0)
  │        m['resqr_quantiles'] = (0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99)
  │
  ├─ [NEW] Write diagnostics to model_data_resqr_diagnostics.json (NOT pkl):
  │        {model_key: {in_sample_coverage, oos_coverage, pinball_loss,
  │                     raw_crossing_fraction, sample_count_used, oos_warnings}}
  │
  ├─ [existing] if Policy B abort:
  │        RuntimeError propagates → build_bm_model.py exits non-zero
  │        → daily_update.sh notify_failure → systemd-cat + notify-send
  │        → prod pkl stays on last-good → /health reports stale build age
  │        → quantoshi-health probes /health → fires popup on 3-day staleness
  │
  └─ [existing] git add model_data.pkl model_data_resqr_diagnostics.json
     [existing] commit, push, deploy

WORKER STARTUP (app.py, after _app_ctx.load_model_data() completes)
  │
  ├─ Read m['resqr_coefs'], m['resqr_knots'], m['resqr_quantiles']
  ├─ Schema validation: knots == expected, quantiles == expected
  ├─ For each model in PRICE_MODELS:
  │     If model_key in resqr_coefs and shape validates:
  │         model._resqr = (sorted_qs, coef_matrix)  # tuple binding
  │     Else:
  │         model._resqr remains None → fallback to constant σ
  │
  ├─ _app_ctx._HAS_RESQR = any(getattr(m, '_resqr', None) is not None
  │                             for m in PRICE_MODELS.values())
  └─ [Critical: this happens in app.py, NOT in _app_ctx.py body, because
     PRICE_MODELS is empty at _app_ctx.py import time]

REQUEST TIME (figure callback + scanner callback on Tab 1)
  │
  ├─ Dash writes bub-sigma-mode.value → read as State by update_bubble callback
  ├─ sigma_mode = bub_sigma_mode or "constant"  # defensive default
  ├─ Figure builder calls model.price_at(q, t, sigma_mode=sigma_mode)
  ├─ _ShrinkingBandsMixin.price_at:
  │     if sigma_mode == "resqr" and self._resqr is not None:
  │         offsets = eval_resqr_offsets(t, *self._resqr)
  │         interpolate across q if needed
  │         return 10 ** (log_median + offset(q, t))
  │     else:
  │         # existing constant-σ path
  │         return 10 ** (log_median + z(q) · self._sigma)
  │
  └─ Scanner callback passes sigma_mode to find_percentile, which threads it
     to interp_price, which threads it to price_at.
```

## Storage schema (model_data.pkl, additive only)

```python
# ADDITIVE — existing keys untouched
m['resqr_build_ts']  = '2026-04-15T03:47:22+00:00'   # ISO-8601 UTC
m['resqr_knots']     = (3.0, 6.0, 9.0, 12.0)
m['resqr_quantiles'] = (0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99)
m['resqr_models']    = frozenset({'bub', 'pl', 'hybppl', 'hyb2l', 'hyb2c',
                                    'hyb2b', 'hyb4d', 'eppl', 'linppl', 'exp',
                                    'pca', 'grdy', 'gomp', 'bpl'})
m['resqr_coefs']     = {
    'bub': (sorted_qs_1D_array, coef_matrix_float64),  # shape (8,), (8, 6)
    'pl':  (sorted_qs_1D_array, coef_matrix_float64),
    ...  # 14 entries max, fewer if some failed Policy A
}
```

**Coverage/diagnostic data lives in `model_data_resqr_diagnostics.json`** next to the pkl, NOT in the pkl itself. Rationale from review: reduces pkl bloat, keeps diagnostics human-readable, separable concerns.

**Naming consistency:** runtime attribute is `model._resqr = (sorted_qs, coef_matrix)` — same shape as the pkl entry. Never mixes with stale names like `_resqr_mat`.

## UI specification

### New "σ mode" section in `layout/display_models.py`

Gated to `prefix == "bub"` only. Other Display Models panels (DCA/Retire/Supercharger) do not get this section.

```python
# Appended to display_models_panel(prefix="bub", ...) after the model checklist,
# before the legend-position dropdown
if prefix == "bub":
    children.append(html.Div([
        _lbl("σ mode"),
        dcc.RadioItems(
            id="bub-sigma-mode",
            options=[
                {"label": " Constant σ",       "value": "constant"},
                {"label": " Residual quantile", "value": "resqr",
                 "disabled": not getattr(_app_ctx, "_HAS_RESQR", False)},
            ],
            value="constant",
            inline=True,
            labelStyle={"display": "inline-block", "marginRight": "12px"},
            inputStyle={"marginRight": "4px"},
        ),
        html.Small([
            "σ mode affects ", html.B("Tab 1 only"),
            " (Bubble chart + Model Scanner). ",
            "Other tabs always use Constant σ."
        ], style={"color": DIM_TEXT, "fontSize": UI_FONT_SM,
                  "display": "block", "marginTop": "4px"}),
        html.Small(
            "LPPL family and models without fitted residuals always use Constant σ.",
            style={"color": DIM_TEXT, "fontSize": UI_FONT_SM, "display": "block"},
        ),
    ], className="mt-2"))
```

### Scanner section header (Tab 1)

In `layout/bubble.py`, the `_section_card("Model Scanner", ...)` call becomes:

```python
_section_card(
    html.Span(id="bub-scanner-header", children="Model Scanner · Constant σ"),
    ...
)
```

Clientside callback registered in `callbacks/scanner.py` (the scanner is a Tab 1 feature):

```python
_app_ctx.app.clientside_callback(
    """
    function(mode) {
        var label = (mode === "resqr") ? "Residual quantile" : "Constant σ";
        return "Model Scanner · " + label;
    }
    """,
    Output("bub-scanner-header", "children"),
    Input("bub-sigma-mode", "value"),
    # prevent_initial_call=False is the default; do NOT set 'initial_duplicate'
)
```

### FAQ text replacement

In `layout/faq.py`, the existing `"2. Heteroscedastic volatility (now implemented): ..."` paragraph is replaced with:

```
2. Heteroscedastic volatility: Residual analysis shows non-monotonic
windowed σ — 0.150 (2010-13) → 0.109 (2013-17) → 0.139 (2017-21) →
0.092 (2021-26). A prior attempt using σ(t)=σ₀·t^(-α) produced bands
that collapsed at current t and was reverted. The current
implementation fits a residual quantile regression directly on each
model's residuals with a piecewise-linear log-t basis (knots at
3/6/9/12 yr). The Display Models panel on Tab 1 has a σ mode radio
that toggles between legacy Constant σ and the new Residual quantile
bands. The Model Scanner on Tab 1 reflects the same choice.
LPPL family keeps Constant σ because LPPL residuals carry a
3.92-yr halving-cycle periodicity the piecewise basis cannot
represent (though calibration is nominal either way). Other tabs
(DCA, Retire, Supercharger, Citadel) continue to render parametric
bands with Constant σ; the toggle is scoped to Tab 1 only.
```

## Callback + signature plumbing

### `callbacks/charts.py::update_bubble`
Gains one new State:
```python
State("bub-sigma-mode", "value"),  # appended to existing State list
```
Function signature gains a new keyword arg `sigma_mode` (positioned after `cta_active` to match the existing pattern). Reads as `sigma_mode = sigma_mode or "constant"` defensively.

The param dict passed to `_get_bubble_fig(m, p)` gains:
```python
p["sigma_mode"] = sigma_mode
```

### `figures/bubble.py::build_bubble_figure(m, p)`
Reads `sigma_mode = p.get("sigma_mode", "constant")` and passes as kwarg into every `model.price_at(q, t, sigma_mode=sigma_mode)` call.

### `callbacks/scanner.py::update_scanner`
Gains `State("bub-sigma-mode", "value")` and threads it into every `model.find_percentile(t, price, sigma_mode=sigma_mode)` call.

### `btc_core.py::_ShrinkingBandsMixin`
```python
def price_at(self, q, t, sigma_mode="constant"):
    t_arr = np.asarray(t, float)
    log_median = self._model_log10(t_arr)
    if sigma_mode == "resqr" and getattr(self, '_resqr', None) is not None:
        sorted_qs, coef_matrix = self._resqr
        offsets = eval_resqr_offsets(t_arr, sorted_qs, coef_matrix)  # (n_t, n_q)
        # Linear interpolation across q (matches existing interp_price pattern)
        q_idx_f = np.interp(q, sorted_qs, np.arange(len(sorted_qs)))
        q_idx_lo = int(np.floor(q_idx_f))
        q_idx_hi = min(q_idx_lo + 1, len(sorted_qs) - 1)
        frac = q_idx_f - q_idx_lo
        offset = (1 - frac) * offsets[:, q_idx_lo] + frac * offsets[:, q_idx_hi]
        return 10.0 ** (log_median + offset)
    # Constant-σ fallback (existing behavior)
    z = self.fits[q]["z"]
    sigma = self._sigma_at(t_arr, q)
    return 10.0 ** (log_median + z * sigma)

def interp_price(self, q, t, sigma_mode="constant"):
    return float(self.price_at(q, t, sigma_mode=sigma_mode))

def find_percentile(self, t, price, sigma_mode="constant"):
    # Existing interpolation logic unchanged; all internal price_at calls
    # now pass sigma_mode through.
    ...
```

### `tab_defaults.py::BUBBLE`
Gains `"sigma_mode": "constant"` so prewarm cache key aligns with runtime key. (Existing cache-key-alignment invariant from CLAUDE.md.)

### `btc_web/_app_ctx.py`
No change at module import time. `PRICE_MODELS` is populated later by `app.py`.

### `btc_web/app.py`
After `load_model_data()` and `PRICE_MODELS` population, adds:
```python
# Bind resqr coefficients to models; compute _HAS_RESQR flag
_HAS_RESQR = False
resqr_coefs = _model_data.get('resqr_coefs', {})
expected_knots = (3.0, 6.0, 9.0, 12.0)
expected_qs    = (0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99)
if (resqr_coefs
        and _model_data.get('resqr_knots')    == expected_knots
        and _model_data.get('resqr_quantiles') == expected_qs):
    for model_key, model in _app_ctx.PRICE_MODELS.items():
        entry = resqr_coefs.get(model_key)
        if entry is None:
            continue
        try:
            sorted_qs, coef_matrix = entry
            assert coef_matrix.shape == (len(sorted_qs), 6)
            assert sorted_qs.shape == (len(sorted_qs),)
            assert np.isfinite(coef_matrix).all()
            model._resqr = (sorted_qs, coef_matrix)
        except Exception as exc:
            _LOG.warning("resqr coefs invalid for %s: %s", model_key, exc)
    _HAS_RESQR = any(
        getattr(m, '_resqr', None) is not None
        for m in _app_ctx.PRICE_MODELS.values()
    )
    _LOG.info("resqr bands: %d/%d models loaded",
              sum(1 for m in _app_ctx.PRICE_MODELS.values()
                  if getattr(m, '_resqr', None) is not None),
              len(_app_ctx.PRICE_MODELS))
else:
    _LOG.warning("resqr bands not loaded: pkl schema missing or mismatched")

_app_ctx._HAS_RESQR = _HAS_RESQR
```

### `btc_web/callbacks/snapshot_cb.py::apply_snapshot`
After the positional decode results assembly, coerce `bub-sigma-mode` when `_HAS_RESQR=False`:
```python
# Defensive: when the server can't actually render resqr bands, coerce
# legacy share links with sigma_mode=resqr to constant so the radio and
# the chart agree. Uses getattr to tolerate import-order edge cases.
if not getattr(_app_ctx, "_HAS_RESQR", False):
    # Find the index of bub-sigma-mode in the positional result list and
    # coerce. Positional ID lookup happens once at module import.
    for i, (cid, prop) in enumerate(_SNAPSHOT_CONTROLS):
        if cid == "bub-sigma-mode" and prop == "value":
            if results[i] == "resqr":
                results[i] = "constant"
            break
```

Coerce location is **`apply_snapshot` (callbacks/snapshot_cb.py)**, NOT `_decode_snapshot` (snapshot.py). Keeps the pure codec decoupled from runtime capability flags.

## Fit engine — `tools/model_toolkit/resqr_bands.py`

```python
"""Residual quantile regression bands with piecewise-linear log-t basis."""
from __future__ import annotations
import logging
import numpy as np
import statsmodels.api as sm

_LOG = logging.getLogger("resqr_bands")

DEFAULT_KNOTS     = (3.0, 6.0, 9.0, 12.0)
DEFAULT_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99)
MIN_SAMPLES       = 500
OOS_TOLERANCE     = 0.05  # ±5pp on Q05/Q25/Q75/Q95
INTERIOR_QS       = (0.05, 0.25, 0.75, 0.95)  # tail Q01/Q99 excluded from assertion


def _basis(t: np.ndarray, knots=DEFAULT_KNOTS) -> np.ndarray:
    """Piecewise-linear-in-log10(t) basis with hinges at knot positions."""
    log_t = np.log10(np.maximum(np.asarray(t, float), 1e-6))
    cols = [np.ones_like(log_t), log_t]
    for k in knots:
        cols.append(np.maximum(log_t - np.log10(k), 0.0))
    return np.column_stack(cols)


def fit_residual_qr_pwl(
    t: np.ndarray,
    residuals: np.ndarray,
    quantiles: tuple = DEFAULT_QUANTILES,
    knots: tuple = DEFAULT_KNOTS,
):
    """Fit PWL QR on residuals per quantile. Raises on any hard failure.

    Returns (sorted_qs: np.ndarray, coef_matrix: (n_q, n_basis) np.ndarray,
             in_sample_coverage: (n_q,) np.ndarray, raw_crossing_frac: float).
    """
    t = np.asarray(t, float)
    residuals = np.asarray(residuals, float)

    # Policy A: pre-flight
    if not np.isfinite(residuals).all():
        n_nans = int((~np.isfinite(residuals)).sum())
        raise ValueError(f"residuals contain {n_nans} non-finite values")

    mask = t > 1.0
    t_fit, r_fit = t[mask], residuals[mask]
    if len(t_fit) < MIN_SAMPLES:
        raise ValueError(f"only {len(t_fit)} samples (need ≥{MIN_SAMPLES})")

    sorted_qs = np.array(sorted(quantiles), dtype=np.float64)
    X = _basis(t_fit, knots)
    n_basis = X.shape[1]
    coef_matrix = np.zeros((len(sorted_qs), n_basis), dtype=np.float64)
    coverage = np.zeros(len(sorted_qs), dtype=np.float64)

    for i, q in enumerate(sorted_qs):
        try:
            res = sm.QuantReg(r_fit, X).fit(q=float(q), max_iter=10000)
        except Exception as exc:
            # Policy A — wrap the solver failure into a ValueError so callers
            # can skip this model cleanly.
            raise ValueError(f"QuantReg failed at q={q}: {exc}") from exc
        coefs = np.asarray(res.params, dtype=np.float64)
        # Policy B — non-finite coefs from a "successful" fit are data corruption
        if not np.isfinite(coefs).all():
            raise RuntimeError(f"q={q} returned non-finite coefs: {coefs}")
        coef_matrix[i] = coefs
        pred = X @ coefs
        coverage[i] = float((r_fit <= pred).mean())

    # Raw crossing diagnostic (pre-sort) — informational only
    all_preds = X @ coef_matrix.T  # (n_t, n_q)
    diffs = np.diff(all_preds, axis=1)
    raw_crossing_frac = float((diffs < 0).sum() / max(diffs.size, 1))

    return sorted_qs, coef_matrix, coverage, raw_crossing_frac


def eval_resqr_offsets(
    t: np.ndarray,
    sorted_qs: np.ndarray,
    coef_matrix: np.ndarray,
    knots: tuple = DEFAULT_KNOTS,
) -> np.ndarray:
    """Return (n_t, n_q) matrix of log10-offsets from median, q-monotone.

    Query-time clipping at the last knot produces a constant-log-offset
    plateau for t > knots[-1] — no extrapolation drift, no crossings
    past the last knot by construction.
    """
    t_arr = np.asarray(t, float)
    t_clipped = np.minimum(t_arr, float(knots[-1]))
    X = _basis(t_clipped, knots)
    offsets = X @ coef_matrix.T                      # (n_t, n_q)
    offsets = np.sort(offsets, axis=1)               # monotone safety net
    return offsets


def fit_and_validate(
    t: np.ndarray,
    residuals: np.ndarray,
    model_key: str,
    quantiles: tuple = DEFAULT_QUANTILES,
    knots: tuple = DEFAULT_KNOTS,
    oos_tolerance: float = OOS_TOLERANCE,
) -> dict:
    """80/20 random holdout fit + validate. Returns diagnostic dict.

    Raises ValueError for Policy A (per-model skip).
    Raises RuntimeError for Policy B (global abort triggers checked externally).
    """
    t = np.asarray(t, float)
    residuals = np.asarray(residuals, float)
    rng = np.random.default_rng(42)
    n = len(t)
    if n < MIN_SAMPLES * 2:  # need enough for a holdout
        raise ValueError(f"only {n} total samples (need ≥{2*MIN_SAMPLES} for holdout)")

    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    sorted_qs, coef_matrix, _, raw_crossings = fit_residual_qr_pwl(
        t[train_idx], residuals[train_idx], quantiles, knots)

    X_te = _basis(t[test_idx], knots)
    pred_te = X_te @ coef_matrix.T
    pred_te = np.sort(pred_te, axis=1)
    oos_cov = (residuals[test_idx][:, None] <= pred_te).mean(axis=0)

    # Hard assert on interior quantiles (policy A: skip)
    for i, q in enumerate(sorted_qs):
        if q in INTERIOR_QS:
            err = abs(oos_cov[i] - q)
            if err > oos_tolerance:
                raise ValueError(
                    f"{model_key} q={q}: OOS coverage {oos_cov[i]:.3f} "
                    f"deviates from nominal by {err:.3f} (>{oos_tolerance})")

    if raw_crossings > 0.05:
        _LOG.warning("%s raw quantile crossings: %.1f%% (will be fixed "
                     "by post-sort)", model_key, raw_crossings * 100)

    # Refit on full data for the stored coefficients
    sorted_qs, coef_matrix, in_cov, _ = fit_residual_qr_pwl(
        t, residuals, quantiles, knots)

    return {
        "model_key": model_key,
        "sorted_qs": sorted_qs,
        "coef_matrix": coef_matrix,
        "in_sample_coverage": in_cov,
        "oos_coverage": oos_cov,
        "raw_crossing_frac": raw_crossings,
        "n_samples": int(n),
    }
```

## Build orchestration — `tools/build_bm_model.py` additions

Build script imports `btc_core` directly and instantiates each in-scope model class from raw fit arrays (Option C: no pkl round-trip). `_model_log10` methods are pure and use only instance state set at `__init__` — confirmed for the 14 in-scope models.

```python
# After all existing model fits complete, before writing pkl
from tools.model_toolkit.resqr_bands import (
    fit_and_validate, DEFAULT_KNOTS, DEFAULT_QUANTILES,
)

RESQR_MODELS = frozenset({
    'bub', 'pl', 'hybppl', 'hyb2l', 'hyb2c', 'hyb2b', 'hyb4d',
    'eppl', 'linppl', 'exp', 'pca', 'grdy', 'gomp', 'bpl',
})

resqr_coefs = {}
diagnostics = {}
failures = []

for model_key in RESQR_MODELS:
    try:
        model = _instantiate_model(model_key, t_all, log_p_all, ...)
        log_median = model._model_log10(t_all)
        residuals = log_p_all - log_median
        result = fit_and_validate(t_all, residuals, model_key)
        resqr_coefs[model_key] = (result["sorted_qs"], result["coef_matrix"])
        diagnostics[model_key] = {
            "in_sample_coverage": result["in_sample_coverage"].tolist(),
            "oos_coverage": result["oos_coverage"].tolist(),
            "raw_crossing_frac": result["raw_crossing_frac"],
            "n_samples": result["n_samples"],
        }
    except ValueError as exc:
        _LOG.warning("resqr skip %s: %s", model_key, exc)
        failures.append((model_key, "A", str(exc)))
    except RuntimeError as exc:
        _LOG.error("resqr Policy B failure on %s: %s", model_key, exc)
        failures.append((model_key, "B", str(exc)))
        if model_key == "bub" or len([f for f in failures if f[1] == "B"]) > len(RESQR_MODELS) // 2:
            raise RuntimeError(
                f"resqr global abort: {model_key} failed with Policy B "
                f"({'BubbleModel flagship' if model_key == 'bub' else '>50% failure rate'})"
            )

m['resqr_coefs']     = resqr_coefs
m['resqr_models']    = frozenset(resqr_coefs.keys())
m['resqr_build_ts']  = datetime.now(timezone.utc).isoformat(timespec="seconds")
m['resqr_knots']     = DEFAULT_KNOTS
m['resqr_quantiles'] = DEFAULT_QUANTILES

# External diagnostics file (NOT in pkl)
with open(DIAGNOSTICS_PATH, 'w') as f:
    json.dump({
        "build_ts": m['resqr_build_ts'],
        "knots": list(DEFAULT_KNOTS),
        "quantiles": list(DEFAULT_QUANTILES),
        "models": diagnostics,
        "failures": failures,
    }, f, indent=2)
```

## Error handling

### Build-time failures
| Case | Trigger | Action |
|---|---|---|
| Policy A: per-model skip | `ValueError`: too few samples, NaN residuals, solver fail, OOS coverage >±5pp on Q05/25/75/95 | Log warning, skip model, build continues |
| Policy B: global abort | `RuntimeError`: NaN coefs from successful fit | Check: is this `bub`? Is >50% already failed? If yes, propagate; build aborts, pkl unchanged |
| Policy B: bub failure | Any Policy A or B error on model `bub` | Immediate global abort — cannot ship broken flagship |

### Runtime fallbacks
- `_HAS_RESQR = False` at worker startup → radio's "Residual quantile" option is `disabled=True`, tooltip explains
- Individual model missing `_resqr` attribute → falls back to constant σ (per-model)
- Legacy share link without `bub-sigma-mode` entry → `None` from decoder → layout default `"constant"` takes effect
- Legacy share link with `bub-sigma-mode="resqr"` on server with `_HAS_RESQR=False` → coerced to `"constant"` in `apply_snapshot`

### Alert delivery path
```
BUILD ABORT on dev laptop (daily_update.sh)
  │
  ├─ systemd-cat journal entry        ← immediate, dev-local
  ├─ notify-send desktop toast        ← immediate, dev-local
  │
  └─ Prod still runs yesterday's pkl
       │
       ├─ /health exposes model_build_age_hours (pkl mtime)
       │  and model_build_stale_72h flag
       │
       └─ quantoshi-health timer (dev, when awake)
            ├─ Check A: grep /tmp/quantoshi-daily-update.log for FAILURE:
            │           (dev-local; fires when laptop is awake)
            └─ Check B: HTTPS probe prod /health → stale_72h flag
                        (works regardless of dev laptop state)
                        → Qt fullscreen popup via --popup flow
```

### Health check additions (app.py `/health` route)
```python
import os
pkl_path = _ROOT / "model_data.pkl"
build_age_hours = None
if pkl_path.exists():
    build_age_hours = (time.time() - pkl_path.stat().st_mtime) / 3600

result["model_build_age_hours"] = round(build_age_hours, 1) if build_age_hours else None
result["model_build_stale_72h"] = (build_age_hours or 0) > 72
result["resqr_bands"] = {
    "loaded": getattr(_app_ctx, "_HAS_RESQR", False),
    "model_count": sum(1 for m in _app_ctx.PRICE_MODELS.values()
                       if getattr(m, '_resqr', None) is not None),
    "build_ts": _model_data.get('resqr_build_ts'),
}
```

### `quantoshi-health` script additions
```bash
# After existing HTTP/SSH/systemd checks
health_body=$(curl -s --max-time "$TIMEOUT" "$URL" 2>/dev/null)
stale=$(echo "$health_body" | python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print('true' if d.get('model_build_stale_72h') else 'false')
except Exception:
    print('unknown')
")
if [[ "$stale" == "true" ]]; then
    errors+=("model_data.pkl on prod is >72h stale (daily update may be broken)")
fi

# Dev-local log check (new)
log_file="/tmp/quantoshi-daily-update.log"
if [[ -f "$log_file" ]]; then
    if tail -50 "$log_file" | grep -qE "FAILURE:|coverage assertion|build_bm_model.py failed"; then
        errors+=("Recent daily update log shows a build failure — see $log_file")
    fi
fi

# Stale-lock sweep (new)
find /tmp/quantoshi-health-alert.lock -mmin +60 -delete 2>/dev/null || true
```

### Rollback invariant (NORMATIVE)

> **Any runtime rollback (`btc_core.py`, `_app_ctx.py`, layout, callbacks) is performed via GitHub PR merge on a workstation.**
>
> **Never hand-edit `/scratch/code/bitcoinprojections/btc_core.py` on the dev laptop that runs `daily_update.sh`.** The cron would auto-commit the revert and the next deploy would find a stale `btc_core.py` on master.
>
> **`tools/model_toolkit/resqr_bands.py` and `tools/build_bm_model.py`'s resqr-writing code are strictly additive and never reverted as part of a runtime rollback.** They continue producing a pkl with the resqr keys even during a runtime regression, so forward re-deploy can pick up the keys without a manual rebuild.
>
> **Emergency escape:** `touch /tmp/quantoshi-update.disable` pauses the daily update job. `daily_update.sh` honors the lockfile at its start and exits 0 with a log line.

## Testing

### Test files (5 total)

**`btc_web/test_resqr_bands.py` — pure unit (~18 cases, <3 sec)**
- Basis construction: shape, zero-before-knot, linear-after-knot, NaN-safe clamp
- Fit math: recovers known PWL slopes, NaN pre-flight → ValueError, NaN coefs → RuntimeError, <500 samples → ValueError
- Eval math: monotone sort fixes crossings, clip at last knot, interior PWL correctness
- Coverage/crossing diagnostics

**`btc_web/test_resqr_runtime.py` — runtime branching (~12 cases, ~5 sec)**
- `price_at` constant vs resqr mode dispatch
- Interpolation across quantiles
- `find_percentile` coupled with mode
- Thread safety (concurrent calls don't cross-contaminate)
- Far-future flatline past last knot

**`btc_web/test_resqr_snapshot.py` — snapshot roundtrip (~8 cases, <1 sec)**
- `bub-sigma-mode` registered in `_SNAPSHOT_CONTROLS` + `_TAB_CONTROLS["bubble"]`
- Not in `_CHECKLIST_OPTIONS`
- Roundtrip "constant" and "resqr"
- Coerce when `_HAS_RESQR=False`
- Legacy link without entry defaults to constant
- Positional stability

**`btc_web/test_resqr_build.py` — build orchestration (~6 cases, ~5 sec)**
- Writes all expected pkl keys
- Global abort on bub failure
- Global abort on >50% failure rate
- Skips LPPL family
- Writes diagnostics JSON
- Deterministic with seed

**`btc_web/test_resqr_e2e.py` — Playwright + Firefox (~6 cases, ~60 sec)**
- Radio renders, toggle changes band width, scanner header updates
- LPPL unchanged across modes, share link restores resqr
- Disabled state when pkl lacks keys

**Runtime budget:** ~14 sec non-E2E, ~60 sec E2E (gated behind `--ignore-glob='*_e2e.py'` convention)

## Files touched

### Create
- `tools/model_toolkit/resqr_bands.py` — fit engine
- `btc_web/test_resqr_bands.py`
- `btc_web/test_resqr_runtime.py`
- `btc_web/test_resqr_snapshot.py`
- `btc_web/test_resqr_build.py`
- `btc_web/test_resqr_e2e.py`

### Modify
- `tools/build_bm_model.py` — orchestrate resqr fit, write pkl keys, write diagnostics JSON
- `btc_core.py` — `_ShrinkingBandsMixin.price_at/interp_price/find_percentile` gain `sigma_mode` kwarg
- `btc_web/app.py` — compute `_HAS_RESQR`, bind `_resqr` to models, extend `/health` route
- `btc_web/layout/display_models.py` — new "σ mode" section on Tab 1 only
- `btc_web/layout/bubble.py` — Scanner section title becomes `bub-scanner-header` span
- `btc_web/layout/faq.py` — replace the "(now implemented)" text
- `btc_web/callbacks/charts.py::update_bubble` — gain `State("bub-sigma-mode", "value")`
- `btc_web/callbacks/scanner.py` — gain `State("bub-sigma-mode", "value")`, thread into `find_percentile`
- `btc_web/figures/bubble.py::build_bubble_figure` — read `p["sigma_mode"]`, pass to `price_at` calls
- `btc_web/callbacks/snapshot_cb.py::apply_snapshot` — coerce when `_HAS_RESQR=False`
- `btc_web/snapshot.py` — append `("bub-sigma-mode", "value")` to `_SNAPSHOT_CONTROLS`
- `btc_web/callbacks/routing.py::_TAB_CONTROLS["bubble"]` — add `"bub-sigma-mode"`
- `btc_web/tab_defaults.py::BUBBLE` — add `"sigma_mode": "constant"` for cache alignment
- `daily_update.sh` — honor `/tmp/quantoshi-update.disable` lockfile
- `btc-web.service` — no change needed (this feature doesn't touch systemd)
- `~/bin/quantoshi/quantoshi-health` — add stale-pkl check + log grep + stale-lock sweep

### Implementation sequence (for writing-plans)
1. Prerequisite: re-read `sigma_bakeoff_report.md` + `sigma_bakeoff_knots_report.md`
2. `tools/model_toolkit/resqr_bands.py` + `test_resqr_bands.py` (TDD)
3. `btc_core.py::_ShrinkingBandsMixin` signature change + `test_resqr_runtime.py`
4. `tools/build_bm_model.py` additions + `test_resqr_build.py`
5. Run `build_bm_model.py` locally, produce fresh `model_data.pkl` + `model_data_resqr_diagnostics.json`
6. `btc_web/snapshot.py` + `btc_web/callbacks/routing.py` + `tab_defaults.py`
7. `btc_web/app.py::_HAS_RESQR` binding + `/health` route extension
8. `btc_web/layout/display_models.py` + `layout/bubble.py` + `layout/faq.py` UI additions
9. `btc_web/callbacks/charts.py::update_bubble` + `callbacks/scanner.py` + `figures/bubble.py`
10. `btc_web/callbacks/snapshot_cb.py::apply_snapshot` coerce
11. `test_resqr_snapshot.py` + integration-style tests
12. `daily_update.sh` lockfile honor + `~/bin/quantoshi/quantoshi-health` additions
13. Full test suite run, commit, push, deploy

## Not in v1 (deferred)

- **Custom Time Axis panel σ bands** — the Tab 1 Custom Time Axis feature currently shows 4 central fit lines without quantile bands. Adding resqr bands there would require live-refitting per slider tick. Deferred; the σ-mode toggle explicitly does NOT apply to that panel.
- **Per-model knot placement** — the knot bake-off showed a single global `(3, 6, 9, 12)` works for all in-scope models. Per-model knots are out of scope for v1.
- **Regression baseline fixture** — per-coefficient snapshot file (`test_resqr_baseline.py`) to catch silent drift on scipy/statsmodels version bumps. Deferred to post-ship.
- **Dynamic label updates** on Display Models checklist showing `· const σ` suffix per model — deferred, replaced with static disclaimers.
- **Per-tab σ-mode selectors** on DCA/Retire/Supercharger — deferred as a future v2 if users ask.
