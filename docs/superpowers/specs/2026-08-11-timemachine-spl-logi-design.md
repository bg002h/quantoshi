# Time Machine — spl (saturating power law) + logi (logistic) as-of

**Written** 2026-08-11 · **Branch** `master` · **Status** design (autonomous build, user-approved through deploy)

## Goal

Add **spl** (SaturatingPowerLawModel) and **logi** (LogisticSCurveModel) as
selectable Time Machine models, with bands fit on data ≤ the as-of date D,
animating as D sweeps — like BM, EPPL, and QR already do.

## Key facts (from the understand workflow)

Both `spl` and `logi` subclass **`_ShrinkingBandsMixin`** (same as
`EPPLConfigModel`), are **`quantized=True`**, band via **constant σ = std(as-of
residuals)** (the shrinking σ₀/α are computed but inert — `_sigma_at` returns the
constant `self._sigma`), and draw through the **generic overlay loop +
`_build_symmetric_bands`** (`figures/bubble.py:300-365`). Their 3-parameter median
is fit **offline** by `differential_evolution` (never in `__init__`) and is
**fast** (~0.3–1 s/fit — QR-like, not EPPL-slow).

So the as-of pattern mirrors **EPPL exactly**: fit the 3 median params + σ on data
≤ D, store params-only, and rebuild a **real** model object per request via
override kwargs. (A `SimpleNamespace` shim like BM's is insufficient — the overlay
loop needs the full model interface: `price_at`, `fits`, `quantized`,
`legend_name`, `r2`.)

## Design decisions

1. **Class overrides** (`btc_core/_simple.py`), copying `EPPLConfigModel`
   (`_hybppl_eppl.py:843-847`):
   - **spl** already has median overrides `log10_L/t0/beta` (shadow class attrs).
     **Add keyword-only `sigma_override=None`.** When set: `self._sigma =
     sigma_override`; `self.fits = {q: {"z": norm.ppf(q)} for q in quantiles}`;
     `self.quantiles = sorted(...)`; `self._build_colors()`; **skip**
     `_init_shrinking_bands`. Price arrays unused in that branch.
   - **logi** has NO overrides. **Add keyword-only `K=None, r=None, t0=None,
     sigma_override=None`.** When median overrides are set, assign them as
     **instance** attrs `self._K/_r/_t0` (never touch the class attrs — `_K/_r/_t0`
     are shared with GompertzModel; instance shadowing protects gomp). Same
     `sigma_override` skip-band-fit branch as spl.
   - Gotcha: spl constructor kwargs are `log10_L/t0/beta` (no leading underscore)
     vs class attrs `_log10_L/_t0/_beta`.

2. **Fit** (`tools/timemachine/fit_spl_asof.py`, `fit_logi_asof.py`): reuse
   `fit_bm_asof._truncate(prices, ymax)` (shared as-of window), DE-fit the 3 median
   params on the truncated data by **reusing the live fitters' own routines** for
   fidelity (spl: `tools/analyze_spl.py::fit_spl` seeds 0,1,2 + `tools/fit_spl.py`
   polish; logi: `tools/fit_logistic.py` DE + polish), compute residuals over the
   truncated window → `sigma = float(std(resid))`, `r2`. Store **params-only**:
   spl `{"params": {"log10_L","t0","beta"}, "sigma": f, "r2": f}`; logi
   `{"params": {"K","r","t0"}, "sigma": f, "r2": f}`.

3. **Grid** (`tools/build_timemachine_grid.py`): add `models["spl"]` / `["logi"]`
   series. Generalize the incremental `add_qr_to_grid` into one
   `add_series_to_grid(grid_path, kinds, workers)` (DRY across qr/spl/logi); CLI
   `--add-models spl,logi`. Also thread `include_spl/include_logi` through
   `build_grid` for full rebuilds. Params-only (no downsampling). Parallel over
   `--workers` (≤24). Failed fits → `null`.

4. **Loader** (`btc_web/timemachine.py`): `asof_spl(idx)` / `asof_logi(idx)` build a
   fresh real model from the frame's params + `sigma_override` (dummy price arrays,
   like `asof_eppl`). None for missing/failed frame. Never touch `PRICE_MODELS`.

5. **Resolution** (`figures/bubble.py::_asof_resolve`): add `"spl"` and `"logi"`
   branches after `"qr"`. Default path unchanged.

6. **Eligibility** (`callbacks/timemachine.py`): `_TM_ELIGIBLE += ("spl","logi")`.

## Out of scope

resqr-as-of; block-space median re-derivation (logi/spl medians are calendar-only);
other families; new snapshot controls (spl/logi are already valid `bub-model-show`
values; TM toggle + date already encode).

## Expected diagnostic behavior (not bugs)

- **spl**: ceiling `L` is unidentifiable (pins near the $1000T cap across cutoffs;
  see `docs/sspl-insights.md`), so the as-of ceiling will jump wildly frame-to-frame.
- **logi**: saturates below current BTC price (intentionally poor `_DEPRIORITIZED`
  model). Watching either fail to hold up is the point of a Time Machine.

## Testing

- `fit_{spl,logi}_asof` return the expected keys; a frame reproduces the live
  fitter on the same truncated data (params within tolerance).
- Class overrides: constructing with `sigma_override` skips the band fit, yields
  `fits`/`quantiles`, and does **not** mutate class attrs (gomp/spl unaffected).
- `asof_{spl,logi}` return a real model with the right `short_name`, never mutate
  `PRICE_MODELS`, None for a null frame / missing series.
- `_asof_resolve` routes "spl"/"logi" to the as-of loaders in as-of mode; default
  path intact. `_TM_ELIGIBLE == {bub,eppl,qr,spl,logi}`.
- Grid: `add_series_to_grid` injects spl+logi, leaves bub/ecfg/qr untouched.
- Bubble: as-of spl/logi fan differs from live; `asof_date=None` byte-identical.
- `check_model_registration.py` green for spl+logi; full non-E2E suite green.
