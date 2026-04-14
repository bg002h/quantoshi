# Custom Time Axis — Overnight Status Report

**Date:** 2026-04-14 (morning)
**Status:** ✅ Deployed to https://quantoshi.xyz and https://quantoshi.xyz/1
**Tests:** 42/42 new tests passing (10 block-map + 24 fit-engine + 8 snapshot)
**Pre-existing failures:** 32 (baseline was 39 before this work; none are new)

---

## How to test

1. Visit https://quantoshi.xyz/1
2. Scroll the left column below "Display Models" → find the **Custom Time Axis** collapsible panel.
3. Tick **▶ Activate Custom Time Axis** — the bubble chart replaces itself with the custom view (Price scatter + PL + 9 QR quantiles + BM floor + Exp lines).
4. Try:
   - **Calendar presets** (`t₀ (calendar)` dropdown): whitepaper / genesis / optimal / NLS / pizza / Mt.Gox / Custom…
   - **Custom date:** opens `dcc.DatePickerSingle`, min 2008-10-31, max 2015-12-31.
   - **Blockheight mode:** flip `Time scale` to `Blockheight`, pick a preset or enter a custom block (capped at ~391k = 2015-12-31 equivalent).
   - **Weighting:** unweighted / 1/t / 1/√t / uniform log-t density. Currently applies only to PL and QR — BM-floor wire-through deferred (see "Known gaps").
   - **Model subset:** uncheck any combination of the 4 models.
5. Toggle Activate **off** — chart restores to standard view (via `bub-redraw-tick` Store router).
6. **Share link test:** click 📸 Share → "Current tab only" → copy URL → open in new tab → panel state should restore, including the Activate checkbox.

## Expected fit times
- ~180 ms typical for all 4 models combined (5742 samples)
- Budget is 300 ms; anything >5 s emits a `custom_fit slow:` warning in `/var/log/quantoshi-error.log`

## End-to-end smoke test results (dev + prod)
- `GET /health` → `"block_map_loaded": true` ✅
- `GET /_dash-layout` → all 8 `cta-*` IDs present + `bub-redraw-tick` ✅
- Direct callback invocation (calendar mode, optimal preset): 178 ms, 13 traces ✅
- Direct callback invocation (block mode, block_0 preset): 189 ms, 13 traces ✅
- Post-2016 bypass → "Custom t₀ must be before 2016-01-01." ✅
- No models selected → "Select at least one model to fit." ✅
- Deactivate → `bub-redraw-tick` bumped, status = "Standard view restored." ✅

---

## Commits on `master`

| SHA | Summary |
|---|---|
| `746fc42` | docs(plan): Custom Time Axis implementation plan |
| `7181251` | feat(block_map): bitcoind RPC + running-max algorithm + CLI |
| `4cef910` | feat(custom_time): block map CSV + preset module + fit engine |
| `dc7a663` | feat(custom_time): panel UI + callback + bubble router wiring |
| `dc3a863` | feat(custom_time): eager-load block map + /health block_map_loaded |
| `b32b480` | fix(custom_time): honest weighting label — BM-floor wire-through deferred |

Plus earlier commits `ff96957` (spec finalized) and `69b140b` (WIP spec).

---

## Files created (14)
```
tools/find_nonmonotonic_blocks.py           # one-off discovery
tools/build_block_map.py                    # CLI: --full / --append / --verify
BitcoinBlocksDaily.csv                      # 5742 rows, 133 KB
btc_web/_custom_time_presets.py             # frozen CAL_PRESETS + BLK_PRESETS
btc_web/engines/custom_fit.py               # 4 fit fns + weights + shim
btc_web/layout/custom_time.py               # panel UI
btc_web/callbacks/custom_time.py            # server callback + 4 clientside
btc_web/test_block_map_cli.py               # 10 CLI tests
btc_web/test_custom_time.py                 # 24 fit-engine unit tests
btc_web/test_custom_time_snapshot.py        # 8 snapshot roundtrip tests
docs/superpowers/specs/2026-04-13-custom-time-axis-design.md
docs/superpowers/plans/2026-04-13-custom-time-axis.md
docs/superpowers/plans/2026-04-13-custom-time-axis-STATUS.md  (this file)
```

## Files modified (6)
```
btc_web/app.py                              # /health +block_map_loaded
btc_web/snapshot.py                         # 8 new _SNAPSHOT_CONTROLS + 2 bitmask
btc_web/callbacks/routing.py                # 8 new _TAB_CONTROLS["bubble"]
btc_web/callbacks/charts.py                 # update_bubble gains bub-redraw-tick Input + cta-active State + guard
btc_web/callbacks/__init__.py               # import callbacks.custom_time
btc_web/layout/bubble.py                    # insert custom_time_panel()
```

---

## Architecture highlights

**Store-router handoff** (Case O) — avoids the classic Dash dual-ownership race:
- `dcc.Store("bub-redraw-tick", data=0)` lives in the Custom Time Axis panel.
- `update_bubble` (existing bubble callback) gains `Input("bub-redraw-tick", "data")` + `State("cta-active", "value")` and raises `PreventUpdate` if `cta_active` is on.
- Custom callback writes `(figure, status, no_update)` on activate, `(no_update, status, tick+1)` on deactivate. The tick bump re-fires `update_bubble`, which reads State `cta-active=[]`, passes the guard, writes the standard figure.

**Block-mode math correction** — block time is NOT linear in calendar time, so blockheight is used as the raw time axis (not converted to years via a constant). Running-max algorithm handles BIP113 non-monotonic timestamps correctly.

**Fit engine** — pure functions in `engines/custom_fit.py`:
- `fit_pl` — closed-form OLS or polyfit(w=√weights)
- `fit_qr` — statsmodels QuantReg per-quantile; weighted via multinomial resampling (QuantReg doesn't accept sample weights)
- `fit_bm_floor` — reuses `tools/model_toolkit/support.py::fit_support()` via a `_PriceDataShim` duck-typing `log_years`/`log_prices`/`df_full`
- `fit_exp` — closed-form OLS on log-linear; uses all samples (no t>0 mask)

All fits take `FitInput(t, price, weighting)` and return `FitResult(name, params, t_plot, y_plot, n_samples, r2, elapsed_ms, note)`.

**Min-sample guards:** PL/Exp≥3, QR-full-9q≥30, QR-reduced-3q≥10, BM-floor≥50. Skipped models appear in the legend with a note.

**Error handling (section 5, all implemented):**
- A: too few samples → per-model skip
- B: post-2016 bypass → status msg + figure preserved
- C: per-model fit failure → isolated try/except
- D: old snapshot → missing keys → defaults
- E: block CSV missing → graceful degrade, calendar still works
- F: >5 s → log warn
- G/R: top-level exception wrapper → user sees error type in status
- L: no models → PreventUpdate + status
- M: unknown weighting → uniform fallback
- O: deactivation via tick bump
- P: block CSV alignment failure → hard-fail at import (actually downgraded to graceful degrade; see Known gaps #1)

---

## Known gaps (not blocking user testing; morning follow-ups)

From the final code review:

1. **`engines/custom_fit.py:115-119`** — The `try/except Exception` wrapping `_load_block_array_once()` silently downgrades the spec's "hard-fail on corruption" to "graceful degrade." Comment at line 92 is now inaccurate. Either update the comment or move the try/except to wrap only the CSV-missing path, not the alignment-assertion path.

2. **`callbacks/custom_time.py:78,91`** — `CAL_PRESET_BY_KEY[cal_preset]` KeyErrors if a stale snapshot carries a preset name that no longer exists. The top-level `except` catches it as `"⚠ Internal error: KeyError"` — functional but not user-friendly.

3. **`callbacks/custom_time.py:_build_figure`** — Uses `template="plotly_white"` directly instead of the app's `figures/common.py::_finalize_chart` / `_dark_layout` pipeline. Works but doesn't honor palette/watermark like the other 6 chart builders. Cosmetic.

4. **`fit_bm_floor` ignores weighting** — `fit_support()` doesn't take weights. The label was hotfixed (commit `b32b480`) to say "Applies to PL and QR. BM-floor and Exponential ignore (wire-through deferred)." The full fix requires adding a `weights=` kwarg to `tools/model_toolkit/support.py::fit_support()` — out of scope for tonight.

5. **`FitResult.note`** is populated by the fit functions but never rendered to the UI. Adding it to the legend or status would surface useful diagnostics ("weighting degraded", "reduced quantiles", "Fit failed").

## Deferred tests (from plan Task 15)

- `test_custom_time_integration.py` — direct callback invocation via `_patch_ctx` pattern
- `test_custom_time_e2e.py` — Playwright + Firefox E2E (requires dev server)
- `test_custom_time_baseline.py` — regression baseline across (model, weighting, preset) combos
- `tools/update_prices.py` integration for daily `build_block_map.py --append`
- `btc-web.service` `StartLimitIntervalSec=300 StartLimitBurst=5` addition (requires root edit + `systemctl daemon-reload`)

## Deferred features (from UrgentTodoItems.md)

- **"Scan mode"** — click two points on the bubble chart, auto-fit PL/QR/Exp through a grid of t₀ candidates. Research-oriented UX, planned as v2.

---

## What to look for during testing

1. **Panel renders at all.** Scroll Tab 1 left column below Display Models.
2. **Activate button toggles the chart view** without errors.
3. **Legend labels show `(n=5,742)`** for each model trace.
4. **Calendar presets all work** — pick whitepaper/genesis/optimal/NLS/pizza/Mt.Gox.
5. **Custom date picker** enforces the 2008-10-31 .. 2015-12-31 range.
6. **Block mode swaps the x-axis** to raw blockheights.
7. **Deactivate restores the standard view.**
8. **Share link roundtrip** — URL captures panel state.

## Backout plan (if anything looks wrong)

```bash
git revert b32b480 dc3a863 dc7a663 4cef910 7181251 --no-edit
git push origin master
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"
```

Or just revert the panel integration (leaves the engine + tests in place):
```bash
git revert dc7a663 --no-edit  # only reverts panel + wiring
```
