# HybPPL_excess Model Integration — Design Spec

**Date:** 2026-04-05
**Scope:** Promote the HybPPL-on-excess experimental fit (currently on /F) into a first-class Quantoshi price model, alongside LPPL/LinPPL/HybPPL.
**Out of scope:** LPPL_excess, LPPL2_excess, LinPPL_excess — only HybPPL_excess graduates (it had the best R² on /F).

## Goal

Turn `HybPPL_excess` (oscillator-only fit on log-price minus BM support) into a daily-refit Quantoshi model key-named `hybppl_ex` that:

- Shows up in the `PRICE_MODELS` dict and renders on the Bubble tab and other chart tabs just like LPPL/LinPPL/HybPPL.
- Gets its 8 oscillation parameters refit daily via `update_prices.py`.
- Pulls the BM support line (A_sup, B_sup) **dynamically from `ModelData`** at instantiation, so the trend stays in sync with the live BM fit rather than being hard-coded.
- Gets documented on the Model Info tab and rendered with a distinct color.

## User stories

- **As a user comparing models on the Bubble tab**, I want to select "HybPPL (excess)" alongside HybPPL to see the difference a detrended oscillator makes.
- **As a model developer**, I want `hybppl_ex`'s trend to always match the current BM support, not drift from it.
- **As an ops person**, I want `hybppl_ex` to refit automatically with the daily price-update pipeline, just like the other LPPL-family models.

## Constraints

- **Snapshot backward-compatibility.** Adding a new model to `PRICE_MODELS` auto-adds it to most tabs' Display Models checklists. The bit-index ordering in `_CHECKLIST_OPTIONS` for `*-model-show` must stay stable — new value entries append, don't re-order.
- **Parameter count: 8 oscillation params** (a0, C1, W_log, PHI1, D, C2, W_cal, PHI2). The BM support intercept and slope are NOT stored with the model — they come from `ModelData` at runtime.
- **Daily refit required.** Parameters need a `--update` flag in the fitter script that rewrites the class constants in `btc_core.py`, matching the existing `fit_hybppl.py` pattern.
- **Short model name: `hybppl_ex`.** Display name: "HybPPL (excess)". Display abbreviation in legends: "HybPPL (ex)".

## Architecture

### Layer 1 — New model class (`btc_core.py`)

**`HybPPLExcessModel(LPPLModel)`** — new class, inherits from LPPLModel for quantile shift / color palette machinery.

```python
class HybPPLExcessModel(LPPLModel):
    """HybPPL oscillators fit to BM-excess (log_price - BM_support).

    Model: log10(price) = A_sup + B_sup*log10(t)
                        + a0
                        + C1*t^(-D)*cos(W_log*ln(t) + PHI1)
                        + C2*cos(W_cal*t + PHI2)

    A_sup and B_sup are pulled from ModelData at instantiation (dynamic
    trend, stays in sync with live BM fit). The 8 oscillation params
    (a0, C1, W_log, PHI1, D, C2, W_cal, PHI2) are refit daily via
    tools/fit_hybppl_excess.py --update.
    """
    name = "HybPPL (excess)"
    short_name = "hybppl_ex"
    legend_name = "HybPPL (ex)"
    dash_style = "dashdot"

    # Fitted oscillation parameters (written by fit_hybppl_excess.py --update)
    _a0   =  0.349900
    _C1   =  0.642100
    _W    =  7.480800   # log-time angular freq (named _W for LPPLModel compat)
    _PHI  =  1.427200
    _D    =  0.660700
    _C2   =  0.231500
    _W_cal= 1.748900    # calendar angular freq (rad/yr)
    _PHI2 = -2.100200

    def __init__(self, price_years, price_prices, quantiles,
                 a_sup=None, b_sup=None):
        """a_sup, b_sup pulled from ModelData.A_sup/B_sup at construction time."""
        self._A_sup = float(a_sup) if a_sup is not None else 0.0
        self._B_sup = float(b_sup) if b_sup is not None else 0.0
        super().__init__(price_years, price_prices, quantiles)

    def _lppl_log10(self, t):
        """BM support + constant + damped log-periodic + undamped calendar."""
        t_arr = np.asarray(t, float)
        t_safe = np.maximum(t_arr, 0.1)
        log_t = np.log10(t_safe)
        support = self._A_sup + self._B_sup * log_t
        damped = self._C1 * t_safe ** (-self._D) * np.cos(
            self._W * np.log(t_safe) + self._PHI)
        undamped = self._C2 * np.cos(self._W_cal * t_safe + self._PHI2)
        return support + self._a0 + damped + undamped
```

**Notes on the base-class interface:** `LPPLModel.__init__` calls `self._lppl_log10(t)` to compute the baseline curve, then derives `sigma` from residuals against `log_price`. Because we override `_lppl_log10`, that works transparently — the baseline curve IS `support + a0 + damped + undamped` and residuals are `log_price - baseline`.

### Layer 2 — BM support wiring (`ModelData` / `app.py`)

`ModelData` already carries the BM support line shape (`support_bm`). We need to expose the **intercept + slope** as attributes.

Check whether `ModelData` already has these. If not, add:
- `ModelData.support_intercept` = A_sup
- `ModelData.support_slope` = B_sup

These come from `model_data.pkl`'s `bm_support_slope` / `bm_support_intercept` keys (or equivalent — check `tools/model_toolkit/export.py`).

**In `app.py`** where models are instantiated:
```python
_app_ctx.PRICE_MODELS["hybppl_ex"] = HybPPLExcessModel(
    M.price_years, M.price_prices, M.QR_QUANTILES,
    a_sup=M.support_intercept, b_sup=M.support_slope,
)
```

### Layer 3 — Fitter script (`tools/fit_hybppl_excess.py`)

Extract the HybPPL-on-excess logic from `tools/fit_linppl_hybppl_excess.py` into a dedicated script with `--update` flag. Pattern matches `tools/fit_hybppl.py`:

- Load prices, compute BM support via `fit_support()`, form `excess = log_price - support`.
- Fit 8 oscillation params via `differential_evolution` (same bounds as the /F generator).
- Print fit summary.
- If `--update`: rewrite the 8 class-level constants in `btc_core.py::HybPPLExcessModel` via regex patterns (identical mechanism to `fit_hybppl.py`).

### Layer 4 — Daily refit integration (`update_prices.py`)

Append a new step after the existing HybPPL refit block:

```python
# HybPPL_excess: refit on BM-excess signal
print("\nRefitting HybPPL_excess parameters …")
hyb_ex_script = REPO_ROOT / "tools" / "fit_hybppl_excess.py"
res = subprocess.run([sys.executable, str(hyb_ex_script), "--update"],
                     capture_output=True, text=True)
if res.returncode != 0:
    print("HybPPL_excess FIT FAILED — stderr (last 3000 chars):")
    print(res.stderr[-3000:])
    print("WARNING: HybPPL_excess fit failed. Continuing with existing parameters.")
else:
    print(res.stdout.strip().split("\n")[-1])
```

### Layer 5 — Display / palette / labels

**`_app_ctx.py` `MODEL_TRACE_COLORS`:** add `"hybppl_ex": "#9B8AFF"` (matches /D swatch).

Also add to each palette in `_app_ctx.PALETTES` (default, cb-brian, cb-rg, cb-full) — use `#9B8AFF` for default and pick analogous light-purple tones for colorblind variants.

**`figures/common.py` `_MODEL_LABELS`:** add `"hybppl_ex": "HybPPL (ex)"`.

**Model Info tab (`layout/model_info.py`):** add a new accordion item `mi-hybppl-ex` right after `mi-hybppl`. Covers:
- Formula (BM support + constant + damped log-periodic + undamped calendar).
- Coefficient table (9 values: A_sup, B_sup, a0, C1, W_log, PHI1, D, C2, W_cal, PHI2).
- Motivation: detrended decomposition. Cite the /F comparison (R²=0.699 vs HybPPL joint fit ≈ same on raw log_price, but ~cleaner residual structure).
- Refit cadence note: "parameters drift daily via update_prices.py".

### Layer 6 — Auto-integration via existing iteration

Many tabs build their Display Models checklist by iterating `_app_ctx.PRICE_MODELS.values()`. `hybppl_ex` will auto-appear wherever LinPPL/HybPPL appear:

- DCA/Retire/SC/Heatmap: included automatically via `_model_show_checklist` (when `standardized=False`) or via the shared LPPL master on the Bubble tab.
- Wait — on Bubble/DCA/Retire/SC when `standardized=True`, the helper injects a single "LPPL" master and drops all LPPL family variants including hybppl. **Decision:** `hybppl_ex` is NOT an LPPL family member (it's a hybrid with calendar-time) — keep it as a separate entry, not hidden by the standardized filter. The standardized filter checks `_LPPL_FAM = {"lppl", "lp2", "lp3", "lp4"} | LPPL_FAMILY_HIDDEN_FROM_BUBBLE` — hybppl_ex is NOT in either set, so it renders as its own entry. Same way HybPPL/LinPPL render today.

So no layout changes needed — just the new model key.

### Layer 7 — Snapshot

Adding `hybppl_ex` as a new Display Models value auto-appears in the model-show checkboxes. Bit-index compat: `_CHECKLIST_OPTIONS["*-model-show"]` currently lists:
`["pl", "lppl", "exp", "s2f", "ef", "bub", "qr"]` (7 values).

**Decision point:** do we also list `hybppl_ex` (and `linppl`, `hybppl`) in the bitmask options? Currently they aren't listed — meaning old snapshot links don't encode their selection state via bitmask. This is a pre-existing pattern — leave it. Adding `hybppl_ex` to the checklist won't corrupt decoding because:
- Encoding path: `_list_to_mask` only encodes values IN opts list, silently drops unknown.
- Decoding path: `_mask_to_list` only decodes bits FOR opts list values.

So old snapshots never had hybppl_ex encoded and won't after the change. Simple append to the bitmask list:

```python
"bub-model-show": ["pl", "lppl", "exp", "s2f", "ef", "bub", "qr", "hybppl_ex"],
```

(Adding it at the END preserves all existing bit positions.)

Same for `dca-model-show`, `ret-model-show`, `sc-model-show`, `hm-model-show`.

## Testing

### Unit tests (pytest)

- `test_hybppl_ex_model_instantiates` — given ModelData fixture, construct model; verify `_A_sup`, `_B_sup` copied; verify `_lppl_log10(t)` returns finite values at t=1, 10, 50.
- `test_hybppl_ex_matches_composite_at_fit` — at a sample t=10 year, value should equal `A_sup + B_sup*1 + a0 + damped + undamped`. Verify bit-by-bit.
- `test_fit_hybppl_excess_update_writes_correctly` — mock btc_core.py, run fitter with `--update`, verify class constants updated with valid new values (positive C1, etc).
- `test_hybppl_ex_in_price_models` — after app init, verify `"hybppl_ex" in _app_ctx.PRICE_MODELS`.

### Manual verification

- DEV server boots without duplicate-ID errors.
- Bubble tab Display Models shows HybPPL (ex) entry with swatch.
- Selecting it draws a trace on the bubble chart.
- Model Info accordion has the new section.

### Regression

- All 853 existing tests still pass.
- `update_prices.py` (dry-run or abbreviated) successfully invokes the new fitter.

## File list

**Created:**
- `tools/fit_hybppl_excess.py` — dedicated fitter with `--update` flag.

**Modified:**
- `btc_core.py` — new `HybPPLExcessModel` class (after `HybPPLModel`).
- `app.py` — add PRICE_MODELS registration (after hybppl entry), pass `a_sup`/`b_sup` from ModelData.
- `btc_core.py` (ModelData) — add `support_intercept`/`support_slope` attributes if not present.
- `tools/model_toolkit/export.py` — ensure `bm_support_intercept`/`bm_support_slope` exported if not already (**verify**).
- `update_prices.py` — append HybPPL_excess refit step.
- `_app_ctx.py` — add `"hybppl_ex": "#9B8AFF"` to `MODEL_TRACE_COLORS` + palettes.
- `btc_web/figures/common.py` — add `"hybppl_ex": "HybPPL (ex)"` to `_MODEL_LABELS`.
- `btc_web/layout/model_info.py` — new accordion item `mi-hybppl-ex`.
- `btc_web/snapshot.py` — append `"hybppl_ex"` to `_CHECKLIST_OPTIONS["*-model-show"]` lists (all 5).
- `btc_web/test_web.py` — new unit tests.
- `CLAUDE.md` — document the new model.

**No deletions.**

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Snapshot bit-index break | `hybppl_ex` appended to end of each `*-model-show` options list. Verified via existing backward-compat tests. |
| BM support mismatch (A_sup/B_sup change between fit and runtime) | Both fitter and model read from same `fit_support()` pipeline. On rebuild, pkl and class constants update together via update_prices.py. |
| `ModelData.support_intercept` doesn't exist | Layer 2 adds it. If add is needed, it's a 2-line change. |
| DE fit non-convergence / worse params than /F values | Seed=42 + same bounds as /F generator → reproducible. On rare DE failure, `update_prices.py` logs warning and keeps prior params (existing pattern from HybPPL/LinPPL). |
| Model renders wildly on Display Models due to B_sup being 0 if a_sup/b_sup not passed | Fallback to 0 means no trend, so curve shows just the oscillations around 0 — visually obvious failure mode, not a crash. Test catches this. |

## Success criteria

- `hybppl_ex` appears in Display Models checklist on Bubble tab with swatch and label "HybPPL (ex)".
- Selecting it draws a trace on the chart matching the BM support + oscillation shape.
- `update_prices.py` refit completes and writes new params into `btc_core.py::HybPPLExcessModel`.
- Model Info accordion has documentation section.
- All 853+ existing tests pass + 4 new tests pass.
- Old snapshot links still decode correctly (no bit-index shifts).
