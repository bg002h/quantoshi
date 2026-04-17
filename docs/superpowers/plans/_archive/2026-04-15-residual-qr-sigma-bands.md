# Residual QR σ Bands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline batch execution). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Tab 1 "σ mode" radio toggling between legacy Constant σ and new Residual quantile bands for 15 parametric price models (LPPL family excluded).

**Architecture:** Offline fit in `tools/build_bm_model.py`, coefficients stored in `model_data.pkl`, runtime dispatch via `price_at(q, t, sigma_mode=...)` kwarg threading through three parallel `_ShrinkingBandsMixin` / `_FitsBasedModel` / `_CompositeModel` implementations.

**Tech Stack:** Dash 4.0.0, statsmodels QuantReg, scipy, numpy, pandas. Python 3.14.3 dev / 3.12.3 prod.

**Spec:** `docs/superpowers/specs/2026-04-15-residual-qr-sigma-bands-design.md`

---

## File Manifest

### Create
| Path | Responsibility |
|---|---|
| `tools/model_toolkit/resqr_bands.py` | `_basis`, `fit_residual_qr_pwl`, `eval_resqr_offsets`, `fit_and_validate` |
| `btc_web/test_resqr_bands.py` | Pure unit tests for the fit engine |
| `btc_web/test_resqr_runtime.py` | `price_at` mode dispatch, interp, far-future flatline |
| `btc_web/test_resqr_snapshot.py` | Snapshot roundtrip + coerce |
| `btc_web/test_resqr_build.py` | Build orchestration, skip LPPL, abort conditions |

### Modify
| Path | Change |
|---|---|
| `tools/build_bm_model.py` | Add `_median_dispatch`, resqr fit loop, write pkl + diagnostics |
| `btc_core.py` | Three parallel `price_at` sites gain `sigma_mode` kwarg + `_model_log10` shims |
| `btc_web/app.py` | Compute `_HAS_RESQR` after model load, extend `/health` route |
| `btc_web/snapshot.py` | Append `("bub-sigma-mode", "value")` to `_SNAPSHOT_CONTROLS` |
| `btc_web/callbacks/routing.py` | Add `"bub-sigma-mode"` to `_TAB_CONTROLS["bubble"]` |
| `btc_web/callbacks/snapshot_cb.py::apply_snapshot` | Coerce `"resqr"` → `"constant"` when `_HAS_RESQR=False` |
| `btc_web/tab_defaults.py` | Add `"sigma_mode": "constant"` to `BUBBLE` for cache alignment |
| `btc_web/layout/display_models.py` | New σ mode section gated to `prefix=="bub"` |
| `btc_web/layout/bubble.py` | Scanner section title becomes `bub-scanner-header` span |
| `btc_web/layout/faq.py` | Replace "(now implemented)" paragraph |
| `btc_web/callbacks/charts.py::update_bubble` | Gain `State("bub-sigma-mode","value")` |
| `btc_web/callbacks/scanner.py` | Gain `State("bub-sigma-mode","value")` |
| `btc_web/figures/bubble.py::build_bubble_figure` | Read `p["sigma_mode"]`, pass to `price_at` calls |
| `daily_update.sh` | Honor `/tmp/quantoshi-update.disable` lockfile |

---

## Task 0: Prerequisite verification

- [ ] **Step 1:** Confirm class hierarchy in `btc_core.py`

Run Grep for `def price_at\|class _ShrinkingBandsMixin\|class _CompositeModel\|class _FitsBasedModel\|class BubbleModel\|class PowerLawModel` in `btc_core.py`.

Expected: `_ShrinkingBandsMixin.price_at` near line 116, `_FitsBasedModel.price_at` near 460, `_CompositeModel.price_at` near 541, `BubbleModel(_CompositeModel)` at 637, `PowerLawModel(_FitsBasedModel)` at 672.

- [ ] **Step 2:** Verify existing pkl has required keys for `_median_dispatch`

Use the existing `_app_ctx.M` load path at worker startup to inspect: needed keys are `qr_fits`, `ols_intercept`, `ols_slope`, `bm_support_slope`, `bm_support_intercept`, `price_years`, `price_prices`. All already present per earlier exploration.

---

## Task 1: `tools/model_toolkit/resqr_bands.py` — fit engine

**Files:**
- Create: `tools/model_toolkit/resqr_bands.py`
- Create: `btc_web/test_resqr_bands.py`

- [ ] **Step 1:** Write the full module per spec §4. Four functions:
  - `_basis(t, knots)` — PWL log-t design matrix (6 columns)
  - `fit_residual_qr_pwl(t, residuals, quantiles, knots)` — per-quantile QR fit, returns `(sorted_qs, coef_matrix, coverage, raw_crossing_frac)`
  - `eval_resqr_offsets(t, sorted_qs, coef_matrix, knots)` — query-time evaluation with clip-at-last-knot and monotone sort
  - `fit_and_validate(t, residuals, model_key, ...)` — 80/20 random holdout, OOS coverage assertion, returns diagnostic dict

- [ ] **Step 2:** Write ~16 unit tests in `btc_web/test_resqr_bands.py`. Coverage per spec §6 test file A.

- [ ] **Step 3:** Run: `btc_venv/bin/python3 -m pytest btc_web/test_resqr_bands.py -v`

- [ ] **Step 4:** Commit.

---

## Task 2: `btc_core.py` — three parallel patches + `_model_log10` shims

**Files:**
- Modify: `btc_core.py` (lines ~90, ~460, ~541)
- Create: `btc_web/test_resqr_runtime.py`

- [ ] **Step 1:** Add module-level `_resqr_price_at(model, q, t_arr, log_median)` helper just above `_ShrinkingBandsMixin`.

- [ ] **Step 2:** Patch `_ShrinkingBandsMixin.price_at`, `interp_price`, `find_percentile` with `sigma_mode="constant"` kwarg.

- [ ] **Step 3:** Add `_FitsBasedModel._model_log10(t)` using `self.fits[0.5]` (or nearest). Patch `price_at`, `interp_price`, `find_percentile`.

- [ ] **Step 4:** Add `_CompositeModel._model_log10(t)` using `self.md.bm_support_slope * log10(t) + self.md.bm_support_intercept`. Patch `price_at`, `interp_price`, `find_percentile`.

- [ ] **Step 5:** Write `btc_web/test_resqr_runtime.py` (~10 tests per spec §6 test file B).

- [ ] **Step 6:** Run tests + smoke import `app`.

- [ ] **Step 7:** Commit.

---

## Task 3: `tools/build_bm_model.py` — `_median_dispatch` + resqr fit loop

**Files:**
- Modify: `tools/build_bm_model.py`
- Create: `btc_web/test_resqr_build.py`

- [ ] **Step 1:** Add `_median_dispatch(model_key, m, t)` function per spec §4 build orchestration block. Branches for each of 15 in-scope models: `bub, pl, hybppl, hybppl_dd, hyb2l, hyb2c, hyb2b, hyb4d, eppl, linppl, exp, pca, grdy, gomp, bpl`. For each branch, transcribe the class's `_model_log10` formula verbatim from `btc_core.py`.

- [ ] **Step 2:** After all existing fits complete, add the resqr fit loop. Policy A (skip on ValueError) vs Policy B (abort on RuntimeError + `bub` failure or >50% failures).

- [ ] **Step 3:** Write new pkl keys: `resqr_coefs`, `resqr_models`, `resqr_build_ts`, `resqr_knots`, `resqr_quantiles`. Write `model_data_resqr_diagnostics.json` next to the pkl.

- [ ] **Step 4:** Write `btc_web/test_resqr_build.py` (~6 tests per spec §6 test file D).

- [ ] **Step 5:** Run the build locally:
```
btc_venv/bin/python3 tools/build_bm_model.py
```
Expected: exits 0, pkl grows by ~5-6 KB, diagnostics JSON created.

- [ ] **Step 6:** Verify new pkl keys via the existing `_app_ctx.load_model_data` loader.

- [ ] **Step 7:** Run build tests + commit.

---

## Task 4: `btc_web/app.py` — `_HAS_RESQR` binding + `/health` route

- [ ] **Step 1:** After `load_model_data()` and `PRICE_MODELS` population, add the binding loop that attaches `model._resqr` and computes `_HAS_RESQR`. Assign to `_app_ctx._HAS_RESQR`.

- [ ] **Step 2:** Extend `/health` route with `model_build_age_hours`, `model_build_stale_72h`, `resqr_bands` dict.

- [ ] **Step 3:** Smoke import + commit.

---

## Task 5: Snapshot + routing + tab_defaults registration

- [ ] **Step 1:** Append `("bub-sigma-mode", "value")` to `_SNAPSHOT_CONTROLS` in `snapshot.py`.
- [ ] **Step 2:** Add `"bub-sigma-mode"` to `_TAB_CONTROLS["bubble"]` in `routing.py`.
- [ ] **Step 3:** Add `"sigma_mode": "constant"` to `BUBBLE` dict in `tab_defaults.py`.
- [ ] **Step 4:** Write `btc_web/test_resqr_snapshot.py` (~8 tests per spec §6 test file C).
- [ ] **Step 5:** Run tests + commit.

---

## Task 6: `snapshot_cb.py::apply_snapshot` coerce

- [ ] **Step 1:** After the `results` list is assembled, before returning, add:
```python
import _app_ctx
if not getattr(_app_ctx, "_HAS_RESQR", False):
    for i, (cid, prop) in enumerate(_SNAPSHOT_CONTROLS):
        if cid == "bub-sigma-mode" and prop == "value":
            if results[i] == "resqr":
                results[i] = "constant"
            break
```

- [ ] **Step 2:** Smoke import + commit.

---

## Task 7: Layout — Display Models σ section + Scanner header + FAQ

- [ ] **Step 1:** `layout/display_models.py` — add σ mode section gated to `prefix=="bub"`, per spec §3. Includes radio + two static disclaimers.

- [ ] **Step 2:** `layout/bubble.py` — find `_section_card("Model Scanner", ...)` and replace the title arg with `html.Span(id="bub-scanner-header", children="Model Scanner · Constant σ")`.

- [ ] **Step 3:** `layout/faq.py` — find the existing `"2. Heteroscedastic volatility (now implemented)..."` paragraph and replace with the new text per spec §3.

- [ ] **Step 4:** Smoke import + commit.

---

## Task 8: Callback wiring — `update_bubble`, `scanner`, `figures/bubble.py`

- [ ] **Step 1:** `callbacks/charts.py::update_bubble` — add `State("bub-sigma-mode", "value")` to the decorator, rename kwarg in signature, read with `or "constant"` fallback, thread into `p["sigma_mode"] = sigma_mode`.

- [ ] **Step 2:** `figures/bubble.py::build_bubble_figure` — read `sigma_mode = p.get("sigma_mode", "constant")`, pass as kwarg to every `model.price_at(q, t, sigma_mode=sigma_mode)` call. Grep for `price_at` in the file first to find all call sites.

- [ ] **Step 3:** `callbacks/scanner.py` — add `State("bub-sigma-mode", "value")`, thread into every `model.find_percentile(t, price, sigma_mode=sigma_mode)` call.

- [ ] **Step 4:** Add clientside callback in `callbacks/scanner.py` updating `bub-scanner-header` text:
```python
_app_ctx.app.clientside_callback(
    """function(mode) {
        var label = (mode === "resqr") ? "Residual quantile" : "Constant σ";
        return "Model Scanner · " + label;
    }""",
    Output("bub-scanner-header", "children"),
    Input("bub-sigma-mode", "value"),
)
```

- [ ] **Step 5:** Smoke import + commit.

---

## Task 9: Full test run + dev server smoke test

- [ ] **Step 1:** Full non-E2E test suite
```
btc_venv/bin/python3 -m pytest btc_web/ --ignore-glob='*_e2e.py' 2>&1 | tail -20
```
Expected: all new tests pass; no regressions beyond existing ~32 pre-existing failures (unrelated to this feature per CLAUDE.md).

- [ ] **Step 2:** Start dev server, check `/health`, probe layout.
```
lsof -ti :8050 2>/dev/null | xargs -r kill -9
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 6
curl -s http://localhost:8050/health | python3 -m json.tool | grep -E "resqr|stale"
```
Expected: `"loaded": true`, `"model_count": 14-15`, `"model_build_stale_72h": false`.

- [ ] **Step 3:** Verify layout contains new control IDs.
```
curl -s http://localhost:8050/_dash-layout | grep -oE "bub-sigma-mode|bub-scanner-header" | sort -u
```
Expected: both IDs present.

- [ ] **Step 4:** Stop dev server.

---

## Task 10: Ops plumbing — `daily_update.sh` + `quantoshi-health`

- [ ] **Step 1:** `daily_update.sh` — add at the top of the script (after `set -eo pipefail`):
```bash
if [[ -f /tmp/quantoshi-update.disable ]]; then
    echo "$(date): daily update disabled via /tmp/quantoshi-update.disable — exiting"
    exit 0
fi
```

- [ ] **Step 2:** `~/bin/quantoshi/quantoshi-health` — add three checks after the existing HTTP check:
  - Parse `/health` JSON, check `model_build_stale_72h`
  - Grep `/tmp/quantoshi-daily-update.log` tail for FAILURE markers
  - `find /tmp/quantoshi-health-alert.lock -mmin +60 -delete` at script start

- [ ] **Step 3:** Commit `daily_update.sh` changes. `quantoshi-health` lives in `~/bin/` and is not in the repo — update in place without commit.

---

## Task 11: Deploy

- [ ] **Step 1:** Verify clean local state: `git status --short`
- [ ] **Step 2:** Push: `git push origin master`
- [ ] **Step 3:** Deploy to prod:
```
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi && systemctl is-active quantoshi"
```
- [ ] **Step 4:** Verify prod `/health`:
```
curl -s https://quantoshi.xyz/health | python3 -m json.tool | grep -E "resqr|stale"
```
Expected: `loaded: true`, `model_count: 14-15`.

- [ ] **Step 5:** Verify prod layout has new controls:
```
curl -s https://quantoshi.xyz/_dash-layout | grep -c "bub-sigma-mode"
```
Expected: ≥1.

---

## Task 12: Deferred post-deploy items (for morning review)

Not blocking the overnight deploy:
- `test_resqr_e2e.py` — 6 Playwright cases (activate radio, scanner header, share link roundtrip, screenshot diff)
- `test_resqr_baseline.py` — per-coefficient snapshot for silent-drift detection
- `~/bin/quantoshi/quantoshi-health` manual verification via `quantoshi-health --verbose`

---

## Self-review checklist

- **Spec coverage:** all six spec sections map to tasks. §1 → Task 4/5, §2 → Task 3, §3 → Task 7, §4 → Task 1/3, §5 → Task 3/4/6/10, §6 → Task 1/2/3/5.
- **No placeholders:** grep'd; clean.
- **Type consistency:** `_resqr` attribute naming consistent; `sigma_mode` kwarg signature consistent across three parallel sites; `RESQR_MODELS` frozenset matches spec.
- **File paths:** all real existing paths except the 5 new test files and `tools/model_toolkit/resqr_bands.py`.
