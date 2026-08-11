# Time Machine — QR (quantile regression) + bands as-of D

**Written** 2026-08-11 · **Branch** `master` · **Status** design (autonomous build, user-approved through deploy)

## Goal

Add **Quantile Regression (`qr`)** as a third selectable Time Machine model
(alongside BM and the EPPL family), with its quantile **bands** fit on data ≤ the
as-of date D, animating as D sweeps — exactly like the EPPL family already does.

## Key architectural fact

**QR is an overlay model, structurally identical to an EPPL `ecfg_*` config**, not
a primary like BM:
- It is a `_FitsBasedModel` (`quantized = True`) whose bands are the per-quantile
  log-linear channels `log10(price) = intercept + slope·log10(t)`.
- On Tab 1 it is drawn by the **overlay loop** (`figures/bubble.py:296+`) via
  `_asof_resolve(model_key, p)`, then `_build_symmetric_bands` shades between the
  selected quantiles (`bubble.py:357`). EPPL rides this exact path.

Therefore the *only* runtime change needed to make QR animate as-of is to teach
`_asof_resolve` to return an as-of QR object for `model_key == "qr"`. Band drawing,
shading, the realized-price reveal, the single-model constraint, MC/CTA exclusion,
and snapshot all already generalize.

## Design decisions

1. **Fit** — `tools/timemachine/fit_qr_asof.py::fit_qr_asof(prices, ymax)`:
   truncate `prices` to rows with `years <= ymax` (reuse `fit_bm_asof._truncate`),
   then `model_toolkit.bands.fit_qr_channels(trunc)` with the **default
   `BM_QUANTILES`** (27) — the same set the live `qr` model uses
   (`M.QR_QUANTILES`, verified n=27). Return params-only
   `{"fits": {"<q>": {"intercept", "slope", "r2"}}}`. Tiny (~27×3 floats/frame);
   no downsampling.

2. **Grid** — add a `"qr"` model series (`models["qr"] = [qr_frame | None, ...]`).
   Two entry points in `tools/build_timemachine_grid.py`:
   - `build_grid(..., include_qr=True)` for full rebuilds (adds `("qr", …)` jobs
     to the same `ProcessPoolExecutor`, so QR fitting parallelizes across the
     caller's core count — run with `--workers os.cpu_count()`, ≤24).
   - `add_qr_to_grid(grid_path, workers)` — **ALARA incremental mode**: load the
     EXISTING grid, fit QR for its exact `frames` (via `_frame_ymax`), inject
     `models["qr"]`, re-save. Avoids redoing the expensive BM/EPPL fits. Frame
     alignment is exact because interior/edge as-of horizons depend only on data
     ≤ D, which is immutable. CLI: `--add-qr`.
   - Failed fits stored as `null` (try/except per job), never dropped — mirrors
     BM/ecfg.

3. **Loader** — `btc_web/timemachine.py::asof_qr(frame_idx)`: return `None` for a
   missing/failed frame; else build a **fresh** `QuantileRegressionModel` from a
   `SimpleNamespace` shim carrying `qr_fits` (float-keyed) + `qr_colors` (from the
   live `_app_ctx.M`, or `{}` pre-startup). Never touches `PRICE_MODELS` — mirrors
   `asof_bm` / `asof_eppl`.

4. **Resolution** — `figures/bubble.py::_asof_resolve`: add
   `if asof_idx is not None and model_key == "qr": return tm.asof_qr(asof_idx)`.
   Non-as-of and all other keys unchanged (byte-identical default path).

5. **Eligibility** — `callbacks/timemachine.py::_TM_ELIGIBLE = ("bub","eppl","qr")`
   so the single-model constraint keeps `qr` selectable in Time Machine mode.

6. **Bands = QR's own channels** fit on data ≤ D (honest as-of; consistent with
   the v1 design's "each model bands via its own native mechanism"). No resqr.

## Out of scope

resqr-as-of; QR as a BM-style *primary* (it stays an overlay); new snapshot
controls (QR is already a valid `bub-model-show` value; TM toggle + date already
encode); other model families.

## Testing

- `fit_qr_asof` returns the expected keys; a frame's Q50 fit reproduces
  `fit_qr_channels` on the same truncated data.
- `asof_qr` returns a `QuantileRegressionModel` with 27 float-keyed fits and
  **never mutates `PRICE_MODELS`** (identity/params unchanged after a call);
  returns `None` for a `null` frame.
- `_asof_resolve("qr", {asof_date: i})` routes to `asof_qr`; with no `asof_date`
  it returns the live `PRICE_MODELS["qr"]` (default path intact).
- Grid post-build: `models["qr"]` present, same length as `frames`, ≥1 non-null.
- `check_model_registration.py` stays green; full non-E2E suite green.
