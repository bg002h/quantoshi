# Percentile Cycle Fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On Tab 1's Percentile pill, fit one calendar-time sinusoid to the picked model's percentile series, extrapolate it with a 1-yr block-bootstrap band, label the turns with fan-mapped prices, and report a live hindcast score.

**Architecture:** Pure math in a new import-light module `btc_web/pctile_cycle.py`; an `lru_cache`d orchestrator plus trace/annotation building in `btc_web/figures/percentile.py`; three new Tab-1 controls in a `bub-pctile-ctl-wrap` span driven by `bub_views.VIEW_MODES`; three snapshot fields appended at the tail.

**Tech Stack:** Python 3.12 (prod) / 3.14 (dev), numpy, scipy (`minimize_scalar`), Plotly, Dash 4.

**Spec:** `docs/superpowers/specs/2026-11-10-percentile-cycle-fit-design.md`

## Global Constraints

- Fitted form is exactly `c + A·cos(2πf·t + φ)`, t in calendar years since 2009-07-25, fitted to the picked model's own percentile series (0–100) at the view's σ-mode, over the **full record with t ≥ 1**.
- Period search band 2–8 yr; central grid 400 nodes, bootstrap grid 200 nodes, bounded refine within one grid step.
- Bootstrap: circular moving-block on residuals of the weekly (every 7th point) subsample, block 52 points, B = 150, seed fixed (`20090725`).
- Band = pointwise 10th/90th percentile of replicate curves, clipped to [0, 100].
- Hindcast: horizon 2 yr, cutoffs every 0.5 yr from last−10 to last−2, ≥ 5 yr history per cutoff, weekly fit, flat = in-sample mean.
- Censor window 2019-04-15 – 2020-02-15 (removed before fitting and scoring).
- Weak fit threshold R² < 0.20.
- Percentile→price: sorted (monotone-rearranged) log10 fan, linear interp in quantile, clipped to outer quantiles.
- `_SNAPSHOT_CONTROLS` is append-only; new fields at indices 348, 349, 350.
- No hex literals outside `btc_web/colors.py` (`test_colors_central.py`); regenerate color artifacts after editing colors.py.
- No callback writes `bub-xrange`.
- Tests import `from app import app` / `conftest`, never `btc_web.app`.

## Review Focus

1. **Picker on a model with no cycle (BM, R² ≈ 0.04)** — a reader expects a warning, not a confident wave with price labels. Pinned in Task 2 (`test_weak_fit_warns_and_drops_labels`).
2. **Slider zoomed into the past (e.g. 2014–2018)** — expect no forecast region and no 2046 axis. Pinned in Task 2 (`test_zoomed_past_has_no_forecast_region`).
3. **Log x-axis** — annotations must sit on their turns (log10 x), not vanish. Pinned in Task 2 (`test_log_axis_annotations_use_log10`).
4. **Cycle fit off** — the view must be exactly today's figure. Pinned in Task 2 (`test_fit_off_matches_legacy_traces`).
5. **Old share link on the Percentile view** — decodes, new fields absent → layout defaults; existing indices unchanged. Pinned in Task 3 (`test_new_fields_appended_at_the_absolute_tail`, `test_q4_link_without_new_fields_decodes`).

---

### Task 1: Math module `pctile_cycle.py`

**Files:**
- Create: `btc_web/pctile_cycle.py`
- Test: `btc_web/test_pctile_cycle.py` (class `TestMath`)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class CycleFit: c: float; A: float; f: float; phi: float; r2: float; n: int` with property `period -> float` (1/f).
  - `fit_cycle(t: np.ndarray, y: np.ndarray, *, pmin=2.0, pmax=8.0, ngrid=400) -> CycleFit`
  - `wave(c, A, f, phi, x) -> np.ndarray` — broadcasts: scalars → shape `x.shape`; arrays of shape (B,) → (B, len(x)).
  - `bootstrap_params(t, y, fit: CycleFit, *, B=150, block=52, step=7, seed=20090725, ngrid=200) -> np.ndarray` shape (B, 4) columns `c, A, f, phi`.
  - `band(params: np.ndarray, x: np.ndarray, lo=10.0, hi=90.0) -> tuple[np.ndarray, np.ndarray]`
  - `turns(fit: CycleFit, t0: float, t1: float) -> list[tuple[float, float, str]]` — (t, value, "peak"|"trough"), sorted by t, analytic.
  - `turn_timing(params: np.ndarray, t_e: float, kind: str, lo=10.0, hi=90.0) -> tuple[float, float]`
  - `hindcast(t, y, *, horizon=2.0, every=0.5, lookback=10.0, min_span=5.0, step=7) -> dict` keys `n, rmse_wave, rmse_flat, win_rate` (win_rate in [0,1]; `n == 0` → the other three are `None`).
  - `fan_prices(logfan: np.ndarray, qs: np.ndarray, pct: np.ndarray) -> np.ndarray` — logfan (n_q, n_t) log10 USD, pct (n_t,) in 0–100 → USD (n_t,); sorts each column first.
  - Constants: `PMIN = 2.0`, `PMAX = 8.0`, `B_DEFAULT = 150`, `BLOCK_WEEKS = 52`, `SEED = 20090725`, `WEAK_R2 = 0.20`, `CENSOR_WINDOW = ("2019-04-15", "2020-02-15")`, `CYCLE_MODEL_KEYS = ("qr","bub","pl","spl","lp3","cfg_1d_1u","ecfg_1d_1u","pca","grdy","ef")`, `CYCLE_MODEL_LABELS = {"lp3": "LPPL₃", "cfg_1d_1u": "HybPPL", "ecfg_1d_1u": "EPPL"}`.

- [ ] **Step 1: Write failing tests** in `TestMath`:
  - `test_recovers_known_sinusoid`: t = 1..17 daily, y = 50 + 30·cos(2π t/3.6 + 1.0) + N(0, 3) seeded → period within 0.02, A within 1.5, c within 1.0, phase wrapped difference < 0.05, r2 > 0.95.
  - `test_period_search_is_bounded`: y = cos(2π t/0.5) (0.5-yr wave) → `2.0 <= fit.period <= 8.0`.
  - `test_wave_broadcasts`: scalars → shape (len x,); params (5,4) columns → (5, len x).
  - `test_bootstrap_deterministic_and_shaped`: two calls same seed → identical array; shape (150, 4).
  - `test_band_widens_into_the_future`: synthetic AR-noise sinusoid; band width at t_last+10 > width at t_last; all values in [0, 100]; lo <= hi.
  - `test_turns_are_extrema_of_the_wave`: for each turn, wave at t±0.01 is ≤ (peak) / ≥ (trough) the turn value; values equal c±A; kinds alternate.
  - `test_turn_timing_brackets_central_turn`: lo <= t_e <= hi for a central fit's own turns.
  - `test_hindcast_pure_sinusoid_wins`: noiseless sinusoid 16 yr → win_rate == 1.0, rmse_wave < 1e-3·rmse_flat... (use `< 0.05 * rmse_flat`).
  - `test_hindcast_too_short_returns_none`: 4 yr of data → n == 0 and rmse fields None.
  - `test_fan_prices_monotone_rearranged_and_clipped`: logfan with one crossed pair → result equals interp on sorted column; pct 0 → lowest sorted level, pct 100 → highest.
- [ ] **Step 2:** `btc_venv/bin/python3 -m pytest btc_web/test_pctile_cycle.py -q` → FAIL (module missing).
- [ ] **Step 3: Implement.** Frequency scan via per-node 3×3 normal equations vectorised over the grid (cos/sin matrices `(ngrid, n)`), then `scipy.optimize.minimize_scalar(method="bounded")` over `[f*−df, f*+df]` ∩ grid range, final coefficients by `np.linalg.lstsq`. `A = hypot(a, b)`, `phi = atan2(−b, a)`. Bootstrap: `rng = np.random.default_rng(seed)`; for each replicate draw `ceil(n/block)` starts, take `(start + arange(block)) % n`, truncate to n. Turns: peaks at `t = (2πk − φ)/(2πf)`, troughs at `(π + 2πk − φ)/(2πf)` for integer k covering [t0, t1]. Turn timing: per replicate `k* = round((2π f_b t_e + φ_b − offset)/(2π))` with offset 0 (peak) / π (trough), `t_b = (2πk* + offset − φ_b)/(2π f_b)`; percentiles of `t_b`. `fan_prices`: `np.sort(logfan, axis=0)`, per column `np.interp(clip(pct/100, qs[0], qs[-1]), qs, col)`, return `10**`.
- [ ] **Step 4:** same pytest command → PASS.
- [ ] **Step 5:** Commit `feat(pctile-cycle): sinusoid fit, block bootstrap, turns, hindcast, fan map`.

### Task 2: Figure — cycle overlay on the Percentile view

**Files:**
- Modify: `btc_web/figures/percentile.py` (add `_cycle_result`, `_cycle_overlay`; branch in `build_percentile_figure`)
- Modify: `btc_web/colors.py` (constants below), then run `btc_venv/bin/python3 tools/generate_color_artifacts.py`
- Test: `btc_web/test_pctile_cycle.py` (class `TestFigure`)

**Interfaces:**
- Consumes: everything Task 1 produces.
- Produces:
  - p keys read by the builder: `cycle_fit: bool` (default False when absent), `cycle_censor: bool`, `cycle_model: str` (default "qr"), `cycle_horizon: int` (default 20).
  - `_cycle_result(model_key: str, sigma_mode: str, censor: bool, fp: str) -> dict | None` (`lru_cache(maxsize=16)`), keys: `t`, `pct` (full-record series, censored days removed), `fit: CycleFit`, `params: np.ndarray`, `hind: dict`, `model_key`. `None` if the model is missing, not quantized, or has no fan.
  - colors.py: `PCTILE_FIT_DATA_COLOR = "#9A9A9A"`, `PCTILE_FIT_BAND_ALPHA = 0.18`, `PCTILE_FIT_CONTEXT_OPACITY = 0.35`, `PCTILE_FORECAST_SHADE = "#000000"`, `PCTILE_FORECAST_SHADE_ALPHA = 0.045`. Weak-fit warning uses existing `PCTILE_RICH_COLOR`.

- [ ] **Step 1: Write failing tests** in `TestFigure` (helper `_p(**kw)` as in `test_occupancy.py`, plus `cycle_fit=True, cycle_censor=False, cycle_model="qr", cycle_horizon=20`):
  - `test_fit_off_matches_legacy_traces`: `cycle_fit=False` → trace names/count equal to the figure built from a p without any `cycle_*` keys.
  - `test_fit_on_adds_band_wave_actual_and_turns`: names include "QR percentile (actual)", a wave trace containing "cycle fit", a band trace "10–90% band"; forecast wave trace has `line.dash == "dash"`.
  - `test_xrange_extends_to_horizon`: xmax = current year + 1 → `layout.xaxis.range[1] ≈ today_t + 20` (± 0.02); horizon 10 → +10.
  - `test_zoomed_past_has_no_forecast_region`: xmin 2014, xmax 2018 → range[1] ≤ yr_to_t(2018)+1e-6, no shape with `fillcolor` of the forecast shade, no forecast annotations.
  - `test_forecast_turn_labels_have_prices`: at least 4 annotations whose text matches `r"Q\d+ · \$[\d.,]+[kMB]?"`; each forecast turn marker trace has `error_x.array` and `error_x.arrayminus` all ≥ 0.
  - `test_summary_line_reports_period_r2_and_hindcast`: one annotation text contains "period", "R²", "hindcast".
  - `test_weak_fit_warns_and_drops_labels`: `cycle_model="bub"` → an annotation containing "weak cycle"; zero price-label annotations.
  - `test_log_axis_annotations_use_log10`: `xscale="log"` → every turn label annotation's x < 2 (log10 of t ≤ 100), and equals log10 of the matching marker x within 1e-6.
  - `test_wave_hover_carries_band_and_prices`: wave trace customdata rows have 6 entries after `_apply_final_steps` (date + 5 numbers) and `hovertemplate` references `customdata[5]`.
  - `test_fit_model_not_active_still_drawn`: `active_models=["bub"]`, `cycle_model="qr"` → "QR percentile (actual)" present and BM's line has opacity `PCTILE_FIT_CONTEXT_OPACITY`.
  - `test_censor_changes_fit_and_is_cached`: `_cycle_result(..., censor=True)` differs in `fit.r2` from `censor=False`; a second call returns the same object (`is`).
  - `test_cycle_result_is_fast_enough`: `_cycle_result.cache_clear()`; cold call on QR completes in < 3 s (generous CI bound; spec budget 1.5 s measured separately).
- [ ] **Step 2:** pytest `-k TestFigure` → FAIL.
- [ ] **Step 3: Implement.**
  - `_cycle_result`: series via `_percentile_series(mdl, t, px, sigma_mode)` on `M.price_years >= 1`; if censor, drop rows with t in `[yr_to_t(2019-04-15), yr_to_t(2020-02-15)]` (convert dates with the existing genesis helpers); `fit_cycle` on the daily series; `bootstrap_params` + `hindcast` on the same arrays.
  - Plot grid: weekly from the series' first t to the right edge.
  - Right edge: `cycle_fit` and `xmax >= date.today().year` → `td + horizon`; else existing `min(yr_to_t(xmax), td)`.
  - Labels: forecast turns only (t > td) and only when `fit.r2 >= WEAK_R2`; text `f"{Mon YYYY}<br>Q{v:.0f} · {money}"` with money formatting like the study's `money()` (`$1.23M`, `$275k`, `$2.10B`). Price = `fan_prices` at the turn date. Annotation `x = log10(t)` on log axis else t; `yref="y"`, peaks `yshift` up / troughs down.
  - Summary annotation paper (0.01, 0.99); footnote paper (0.5, −0.2) small font; weak-fit summary replaced by the warning text in `PCTILE_RICH_COLOR`.
  - Wave hover: customdata rows `[band_lo, band_hi, price, price_lo, price_hi]`; template `"<b>%{fullData.name}</b><br>%{customdata[0]}<br>fit Q%{y:.0f} (band Q%{customdata[1]:.0f}–Q%{customdata[2]:.0f})<br>≈ $%{customdata[3]:.3s} ($%{customdata[4]:.3s}–$%{customdata[5]:.3s})<extra></extra>"`.
  - Band: two traces (`hi` then `lo` with `fill="tonexty"`), `hoverinfo="skip"`, `showlegend` only on the second.
  - Title when on: `"Price percentile within model fan + cycle fit"`.
- [ ] **Step 4:** pytest file → PASS; `btc_venv/bin/python3 -m pytest btc_web/test_colors_central.py -q` → PASS.
- [ ] **Step 5:** Commit `feat(pctile-cycle): cycle overlay on the Percentile view`.

### Task 3: Controls, wiring, snapshot

**Files:**
- Modify: `btc_web/bub_views.py` (percentile `ctl="bub-pctile-ctl-wrap"`; docstrings "14" → derived count)
- Modify: `btc_web/layout/common.py` (span + `flexWrap: "wrap"` on the pill row)
- Modify: `btc_web/callbacks/charts/__init__.py` (`toggle_bub_view` + clientside sync: add `Output("bub-pctile-ctl-wrap","style")` after `bub-occ-ctl-wrap`... in `STYLE_OUTPUT_IDS` order; `update_bub_pctile` + 3 Inputs)
- Modify: `btc_web/callbacks/routing.py` (`deep_link_bub_view` Output in the same position; NU tuple length; `_TAB_CONTROLS["bubble"]` += 3 ids)
- Modify: `btc_web/snapshot.py` (append 3 tail entries; `_CHECKLIST_OPTIONS["bub-pctile-fit"] = ["fit", "censor"]`)
- Modify: `btc_web/snapshot_defaults.py` (`'bub-pctile-fit:value': ['fit']`, `'bub-pctile-fit-model:value': 'qr'`, `'bub-pctile-fit-horizon:value': 20`) — run `tools/update_defaults_registry.py` before and after.
- Modify tests that pin the old shape: `test_bub_view_modes.py`, `test_occupancy.py`, `test_raw_mc_snapshot.py` / others found by the suite.
- Test: `btc_web/test_pctile_cycle.py` (classes `TestWiring`, `TestSnapshot`)

**Interfaces:**
- Consumes: Task 2's p keys.
- Produces: component ids `bub-pctile-ctl-wrap`, `bub-pctile-fit`, `bub-pctile-fit-model`, `bub-pctile-fit-horizon`.

- [ ] **Step 1: Write failing tests:**
  - `TestWiring.test_percentile_mode_shows_ctl_span`: `mode_styles("percentile")` has `{"display": "inline"}` at `STYLE_OUTPUT_IDS.index("bub-pctile-ctl-wrap")`; every other mode hides it.
  - `TestWiring.test_layout_has_controls_with_defaults`: walk the layout; ids exist; checklist value `["fit"]`, model value `"qr"`, horizon value `20`; model options ⊆ `CYCLE_MODEL_KEYS` and include "qr".
  - `TestWiring.test_update_callback_passes_cycle_params`: call `update_bub_pctile` (patch `utils._get_pctile_fig` to capture p) with fit `["fit","censor"]`, model "pl", horizon 30 → p has `cycle_fit True, cycle_censor True, cycle_model "pl", cycle_horizon 30`; with checklist `[]` → `cycle_fit False`; horizon `None` → 20.
  - `TestSnapshot.test_new_fields_appended_at_the_absolute_tail`: `_SNAPSHOT_CONTROLS[348:351] == [("bub-pctile-fit","value"),("bub-pctile-fit-model","value"),("bub-pctile-fit-horizon","value")]` and `len == 351`.
  - `TestSnapshot.test_round_trip`: encode `{"bub-view-mode:data": "percentile", "bub-pctile-fit:value": ["fit","censor"], "bub-pctile-fit-model:value": "pl", "bub-pctile-fit-horizon:value": 30}` → decode equal.
  - `TestSnapshot.test_q4_link_without_new_fields_decodes`: an encoding that omits them decodes without them (absent → layout default).
  - `TestSnapshot.test_defaults_and_tab_controls`: SNAPSHOT_DEFAULTS values as above; ids ⊆ `_TAB_CONTROLS["bubble"]`; `_CHECKLIST_OPTIONS["bub-pctile-fit"] == ["fit","censor"]`.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement** per Files. Model options: `[{"label": CYCLE_MODEL_LABELS.get(k, PRICE_MODELS[k].legend_name), "value": k} for k in CYCLE_MODEL_KEYS if k in PRICE_MODELS and PRICE_MODELS[k].quantized]`. Checklist `inline=True`, `inputStyle={"marginRight": "3px"}`, `labelStyle={"marginLeft": "8px", "fontSize": UI_FONT_BASE}`. Dropdown widths 100px / 80px, same style dict as the occupancy dropdowns. Callback coercion: `cycle_fit = "fit" in (fit_val or [])`, `cycle_censor = "censor" in (fit_val or [])`, `cycle_model = model if model in CYCLE_MODEL_KEYS else "qr"`, `cycle_horizon = int(h) if h in (10, 20, 30) else 20`.
- [ ] **Step 4:** `btc_venv/bin/python3 -m pytest btc_web/ -q --ignore-glob='*_e2e.py' -x -p no:randomly 2>&1 | tail` → only the known `test_free_tier_all_models` failure; fix every other red.
- [ ] **Step 5:** `btc_venv/bin/python3 scripts/sabotage_callback_guards.py` → all guards still fail under sabotage. Commit `feat(pctile-cycle): controls, wiring and share-link fields`.

### Task 4: Live check, docs, ship

- [ ] **Step 1:** Restart DEV server (`lsof -ti :8050 | xargs kill -9; DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &`), open `/1.4` in Playwright (Firefox) at 1400×900 and 390×844; screenshot; switch picker to BM and to PL; toggle "skip 2019–20"; check console for errors and the dev log for tracebacks.
- [ ] **Step 2:** CLAUDE.md Tab-1 row: one sentence on the cycle fit; `docs/superpowers/followups.md` gets any parked items with an owning phase.
- [ ] **Step 3:** Whole-diff Opus review → persist commit → fold commit (gate output in message) → re-review until 0C/0I.
- [ ] **Step 4:** `git status` clean except the unrelated `model_data_ef.pkl`; push; deploy (`git pull && systemctl restart quantoshi` — no pkl change, no FLUSHDB needed, but figure L2 keys include the new p keys so stale entries cannot collide); `curl -s -o /dev/null -w "%{http_code}" https://quantoshi.xyz/1.4` → 200; prod error log clean.
