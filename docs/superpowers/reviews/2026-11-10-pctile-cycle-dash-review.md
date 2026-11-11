# Percentile cycle fit: Dash footgun review

- **Scope:** `git diff fe96f4e7 afaffa98 -- btc_web/callbacks/ btc_web/layout/ btc_web/bub_views.py btc_web/snapshot.py btc_web/snapshot_defaults.py` (6 files, +120/-5). I also read `snapshot_defaults_registry.json`, `figures/percentile.py`, `utils.py`, `snapshot_cb.py`, `restore_builder.py`, `app.py` and `assets/dash_nocache.js` where a question depended on them.
- **Reviewer:** dash-callback-reviewer (read-only; this file is the only write).
- **Counts: 0 Critical, 0 Important, 4 Minor, 3 Nit.**

## Machine checks run during this review

These are in addition to the controller's suite and sabotage runs.

| Check | Result |
|---|---|
| The fit-off figure from the new builder (via `_cycle_params([] / ['censor'] / None, 'eppl', 'log', 30)`) vs the `fe96f4e7` builder, compared as `to_plotly_json()`. Three param sets: qr/log/legend+today, qr+bub+eppl+s2f/linear/minor_grid, and no models. | **9/9 byte-identical** |
| `_SNAPSHOT_CONTROLS` | 352 entries. 348–351 are the four new fields in spec order, and 347 is still `bub-mc-extremes`. |
| `STYLE_OUTPUT_IDS` | 15 entries. The tail is `cagr-fwd-wrap, pctile-ctl-wrap, occ-ctl-wrap`, matching the Output order in all three callbacks. `mode_styles('percentile')` shows only the pctile ctl. |
| Defaults registry diff | Evicts only the oldest entry (`faf00e93`), adds `757574d3`, and the 19 shared entries are unchanged. The current fp is registered with `bub-pctile-fit:value = ['fit']`. |
| Old-link decode, view-mode=percentile | A registered old fp (`0eac47fc`) restores `bub-pctile-fit = []`. The evicted fp (`faf00e93`) restores **`['fit']`** (finding M1). |
| Cold build of `build_percentile_figure` (dev box) | Fit off: 0.15 s, 258 KB. Fit on (qr/cal): **2.76 s cold, 0.28 s warm, 579 KB**. qr/log: 2.05 s. eppl/calw: 2.15 s. gunicorn `--timeout 120`. |

## Your specific questions

1. **allow_duplicate / prevent_initial_call.** Clean.
   - `toggle_bub_view` (`charts/__init__.py:643`) writes `bub-pctile-ctl-wrap.style` as the single non-duplicate writer, like `bub-occ-ctl-wrap`. Its only allow_duplicate output is `bub-xrange`, and it has `prevent_initial_call=True`.
   - The clientside sync (`:703`) and `deep_link_bub_view` (`routing.py:385`) mark every output allow_duplicate and use `prevent_initial_call=True`.
   - The new info clientside callback (`:902`) has no allow_duplicate output and is the sole writer of `bub-pctile-info.children/style`, so its default initial call is legal.
   - `apply_tab_bubble` picks the four new controls up by the `bub-` prefix, as allow_duplicate outputs with `prevent_initial_call=True`.

2. **"Nonexistent object" under lazy load / prefetch.** No new exposure.
   - Every new id (`bub-pctile-ctl-wrap`, `bub-pctile-fit*`, `bub-pctile-info`) is built in `_chart_tab_layout_with_fab` (`layout/common.py:557`, `:740`). That is the same `bubble-lazy` subtree as `bub-pctile-graph`, `bub-xrange`, `bub-view-mode` and `bub-occ-ctl-wrap`.
   - `_chart_tab_layout_with_fab` has exactly one caller (`layout/bubble.py:661`), so the ids cannot be duplicated.
   - `update_bub_pctile`'s existence profile is unchanged. The new Inputs are present exactly when its existing `bub-` Inputs are, and its only non-Tab-1 deps are still `palette-store`, `user-model-store` and `snapshot-pending`.
   - The info callback's only Input is `bub-pctile-graph.figure`, so it cannot be dispatched while Tab 1 is unmounted.
   - On prefetch mount, `update_bub_pctile` is `prevent_initial_call=True`, so mounting adds no server round trip. Only the free clientside info callback fires.

3. **Falsy-zero.** None. `_cycle_params` (`charts/__init__.py:882`) uses membership tests (`in _pc.HORIZONS` etc.), not `or`. Horizon values are 10/20/30, so 0 is never valid. `legend_pos or "outside"` and `sigma_mode or "constant"` are pre-existing string fallbacks.

4. **Tab-defaults, snapshot-defaults and cache-key alignment.** Aligned.
   - There is no pctile prewarm or L0 pin (no `pctile` in `app.py` or `tab_defaults.py`), so no prewarm key can drift.
   - Runtime key: `_get_pctile_fig` hashes the whole `p` plus `_day`. The new keys are bool/str/int, which `_quantize_params` passes through unchanged (`utils.py:33`), so the `int` horizon is not floated.
   - Defaults agree in all three places: `SNAPSHOT_DEFAULTS` (`qr`/`cal`/`20`/`['fit']`), the `_cycle_params` fallbacks (`"qr"`, `DEFAULT_BASIS="cal"`, `DEFAULT_HORIZON=20`), and the layout initial values, which read `SNAPSHOT_DEFAULTS`.
   - The L2 Redis round trip keeps `layout.meta.cycle_info` (`pio.from_json` keeps `meta`).

5. **JS tab map drift.** N/A. No tab is added or renamed, and `/1.4` was already routed. No `assets/*.js` references the new ids. The view tables reach the browser through `styles_table_json()`, which is generated.

6. **Stale `/_dash-dependencies`.** Reloads are covered; open tabs are not (M2).
   - Covered: `dash_nocache.js` forces `no-store`. `_DEPLOY_FP = md5(repr(app._callback_list))` (`app.py:564`) changes because both the Inputs and the multi-output keys changed, so the ETag busts.
   - Not covered: a tab left open across the deploy.

7. **Snapshot restore path.** Holds.
   - `_TAB_CONTROLS["bubble"]` lists the four ids (`routing.py:166`), so "Current tab only" links carry them.
   - `apply_tab_bubble` derives them by the `bub-` prefix (`snapshot_cb.py:216-232`). It writes them in the same batch as `snapshot-pending=False`, so `update_bub_pctile` (State `snapshot-pending`) fires once with the restored values. The gate is intact.
   - `_CHECKLIST_OPTIONS["bub-pctile-fit"] = ["fit", "censor"]` is a new key, so its position in the dict is not load-bearing; nothing iterates it in order.
   - Predating links: q2/q3 positional links and v4 links under a still-registered old fp restore the fit OFF. A v4 link under a new fp gets `['fit']` from its own registry entry.
   - The one gap is evicted fps (M1).

8. **Percentile view unchanged when the checklist is empty.** Yes for the figure: byte-identical to `fe96f4e7` in 9/9 cases. Two UI-level differences remain:
   - The dropdowns stay live while the fit is off (M3).
   - The pill row now wraps in every view (N3).

## Findings

### Minor

**[M1] `btc_web/snapshot.py:959-964`, plus `_fill_predating_fields` at `:728`: links under an evicted fingerprint restore with the cycle fit ON**
- **Rule:** 6 (snapshot coverage), predating-link semantics.
- **Issue:**
  - When a v4 link's fp is not in the registry, the decode falls back to the *current* `SNAPSHOT_DEFAULTS`, which now include `bub-pctile-fit:value = ['fit']`.
  - `_fill_predating_fields` then sees the key as present and skips it. An old Percentile link therefore restores with a forecast its sender never saw, which is exactly what `_PREDATING_LINK_VALUES` exists to prevent.
  - Verified: `faf00e93`, evicted by this very change, decodes to `['fit']`.
  - The comment at `:715-717` says only that post-field links always carry a value. The converse, that pre-field links always lack one, does not hold under eviction.
  - Not Important: evicted fps are documented to "restore differently" and a warning is logged. This is the same lossy contract every default change has, and `bub-mc-paths` already has the same gap.
- **Fix:** In the fallback branch, drop predating keys that the oldest retained entry lacks. An evicted fp is older than every retained one, so it predates any such field. The registry is oldest-first: append at the tail, evict at the head, as confirmed by this diff.
  ```python
  if historical is None:
      log.warning(...)
      historical = dict(SNAPSHOT_DEFAULTS)
      oldest = next(iter(_load_registry().values()), None)
      if oldest is not None:
          for k in _PREDATING_LINK_VALUES:
              if k not in oldest:
                  historical.pop(k, None)
  ```
  Pin it with a test that decodes a link under a fake unregistered fp and asserts `bub-pctile-fit:value == []`.

**[M2] `charts/__init__.py:643-664`, `:915-931`; `routing.py:385-410`: tabs open across the deploy lose the Tab-1 view pills and the Percentile chart until reload**
- **Rule:** stale `/_dash-dependencies`.
- **Issue:**
  - `toggle_bub_view` and `deep_link_bub_view` gained an Output, so their multi-output key string changes. An old in-memory graph posts the old key, and the server's `callback_map` lookup fails with a 500, so the pills go dead.
  - `update_bub_pctile` gained four Inputs, so old clients send 8+2 args. `validate_and_group_input_args` raises IndexError and the response is a 500 on every trigger.
  - The repo's mitigation (no-store fetch plus the `_DEPLOY_FP` ETag) only takes effect on reload. Nothing in `assets/` detects a deploy or reloads an open tab (searched for `location.reload` and `DEPLOY_FP`).
  - Every signature-changing deploy has this property, and it could not be avoided here because the Output list itself had to grow.
- **Fix:** No code change is required for this feature. Deploy as usual. If it matters later, return `_DEPLOY_FP` as a response header on `/_dash-update-component` and have a small asset reload the page when it differs from the value seen at load.

**[M3] `charts/__init__.py:882-897`, `:933-967`: with the fit unticked, the model/basis/horizon dropdowns still trigger a server rebuild of an identical figure under a new cache key**
- **Rule:** 8-adjacent (ALARA; wasted callbacks).
- **Issue:**
  - All four controls show whenever the Percentile view is active. `_cycle_params` passes `cycle_model`, `cycle_basis`, `cycle_horizon` and `cycle_censor` into `p` even when `cycle_fit` is False.
  - Each dropdown change while the fit is off therefore costs a POST, an L1 miss (new key), a rebuild (about 0.15 s) and about 258 KB on the wire, for a figure that is byte-identical (verified above).
- **Fix:**
  1. In `_cycle_params`, normalise the four sub-keys to their defaults when `"fit"` is not ticked, so the key is stable and L1 hits.
  2. In `update_bub_pctile`, `return dash.no_update` when `set(ctx.triggered_prop_ids.values()) <= {"bub-pctile-fit-model", "bub-pctile-fit-basis", "bub-pctile-fit-horizon"}` and `"fit"` is not in `fit_val`.
  3. Optionally, grey the dropdowns out clientside while the fit is off.

**[M4] `figures/percentile.py` (outside the diff scope, flagged because the fit defaults ON): Percentile payload and cold cost roughly double for every default visit**
- **Rule:** ALARA (performance, not correctness).
- **Issue:**
  - With `['fit']` as the new-visit default, the Percentile figure grows from 258 KB to 579 KB.
  - The first fit-on build on each gunicorn worker pays about 2–3 s on the dev box (likely more on the VPS). That cost recurs per (model, basis, censor, sigma) combination after each daily `model_data.pkl` rebuild, because `_cycle_result`'s lru key includes `_MODEL_FP`.
  - There is no prewarm. This is far under the 120 s timeout, so it is not a failure.
- **Fix:** Optional. Warm `_cycle_result("qr", "constant", False, "cal", fp)` in `_prewarm_caches()` (a single call per worker). Trim the band grid if the wire size matters, e.g. drop band points beyond `today + max(HORIZONS)` that the axis never shows. Leave as is if the operator accepts the cost.

### Nit

**[N1] Stale counts in comments.** `bub_views.py:52, 91, 104, 150`, `charts/__init__.py:671, 688, 694` still say "14" view-state values; it is now 15. `routing.py:420` says "the first 15 outputs are the mode plus mode_styles(mode)"; it is now 16. **Fix:** say "the `len(STYLE_OUTPUT_IDS)` values" so the next view does not re-stale them.

**[N2] `layout/common.py:552-555` vs `charts/__init__.py:892-893` vs `figures/percentile.py:_cycle_model`: three slightly different validity filters for the model picker.**
- The layout requires `quantized`.
- `_cycle_params` requires only registration.
- The builder requires `quantized` and non-empty `quantiles`.
- On a host without `model_data_ef.pkl`, an `ef` share link writes a value that is not among the dropdown's options. The dropdown shows blank while the chart silently draws QR (via `_cycle_params`).
- Harmless today, since all 10 keys pass every filter on this host. **Fix:** one `pctile_cycle.available_models()` helper used by all three, with `_cycle_params` mapping unavailable keys to `"qr"` (as now).

**[N3] `layout/common.py:650-651`: `flexWrap: "wrap"` applies to the pill row in all five views,** not only Percentile. On narrow viewports the Price, CAGR, Residuals and Occupancy pill rows now wrap instead of overflowing. This is probably the intended improvement. Noted only because "Percentile view unchanged" does not cover the other views' layout.

## Checklist rules not covered above

- Rule 2 (number inputs): N/A. The new controls are a Checklist and Dropdowns with fixed options.
- Rule 7 (SC body toggle): N/A.
- Rule 8 (palette storm): N/A. `palette-store` was already an Input of `update_bub_pctile`, and no palette callback was touched.

Reviewed 6 changed files (10 callback/layout/snapshot changes); 7 findings (0 critical, 0 important, 4 minor, 3 nit).
