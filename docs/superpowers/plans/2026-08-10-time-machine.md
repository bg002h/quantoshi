# Time Machine — as-of-date model view — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Tab-1 "Time Machine" mode that shows BM or an EPPL model fit on data through a chosen past date D, projected forward, with the realized price after D drawn on top; a slider (▶ play) sweeps D through a precomputed as-of-date grid.

**Architecture:** A one-time, dev-built **params-only grid** (`gzipped JSON`, git-tracked) stores each model's *cold* fit at ~140 as-of dates. At runtime, when `asof_date` (a frame index) is set in the bubble params dict, model resolution is redirected **per-request** to a frozen as-of model object built from grid params — never mutating the shared `PRICE_MODELS` singleton. Bubble figure draws that frozen fan plus the realized price.

**Tech Stack:** Python 3.14 (dev) / 3.12 (prod), NumPy, SciPy (`differential_evolution`, `least_squares`), Plotly Dash 4, pytest. No new runtime deps (must stay within `btc_web/requirements.txt`).

## Global Constraints

- **Design authority:** `docs/superpowers/specs/2026-08-10-time-machine-design.md`. Where this plan and the spec disagree, stop and ask.
- **v1 scope:** BM + the 36 `ecfg_*` EPPL variants + flagship `eppl`. NOT LPPL/HybPPL/PCA/Greedy. One animated model at a time (Model A); EPPL Model B rides free.
- **Grid dates:** quarterly from the left-edge date through 2015-Q4, then monthly from 2016-01 to the latest full month. Right edge = the live deployed model (not a grid frame).
- **Cold every frame** — no warm-chaining. Fits must be *converged* (Task 0 pins `maxiter`); fixed `seed=42`.
- **Bands as-of D:** each model's shrinking/constant-σ bands from residuals of data ≤ D. NOT resqr.
- **Per-request only:** never mutate `_app_ctx.PRICE_MODELS`. Mirror the `u1` escape hatch (`figures/common.py:1262`).
- **No `matplotlib`/`kaleido` imports under `btc_web/`** (prod startup breaks — dev-only libs). The grid builder lives under `tools/` and may use them; the runtime loader under `btc_web/` may not.
- **Grid storage = gzipped JSON, NOT pickle.** The grid is params-only (floats/ints/strings/small lists), so it serializes as pure JSON. Use `gzip`+`json` — never `pickle`/`np.save(allow_pickle=True)`/`dtype=object`. This keeps zero arbitrary-code-execution surface on load even though the file is first-party and git-tracked. (Distinct from `model_data.pkl`, which is a pre-existing pickle; do not add a second one.)
- **`git add` scoped to named paths** — never `-A`/`.`.
- Model short-names are permanent share-link keys; do not rename.

---

# PHASE 0 — Spike (resolve the four measured unknowns)

### Task 0: Build-parameter spike

**Files:**
- Create: `tools/timemachine/spike.py` (scratch measurement; committed for reproducibility)
- Create: `tools/timemachine/__init__.py` (empty package marker)

**Interfaces:**
- Produces: four constants consumed by Task 1/2/3 — `LEFT_EDGE_DATE` (str `YYYY-MM-DD`), `EPPL_MAXITER` (int), `FLAGSHIP_POLICY` (`"own_frame"` | `"map_2dd2uu"`), and a measured BM per-frame cost (informational).

- [ ] **Step 1: Convergence sweep for `maxiter`.** For `ecfg_2dd_2uu` (16p, the hardest) on full data, run `differential_evolution` at `maxiter ∈ {150, 300, 600, 1200}`, `tol=1e-8`, `seed=42`, `workers=1`; record params and r². Pick the smallest `maxiter` where max |Δparam| between it and the next step < 1e-3 and Δr² < 1e-5. Record as `EPPL_MAXITER`.

```python
# tools/timemachine/spike.py (excerpt)
import os, sys, time, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "tools"), ROOT]
from model_toolkit.data import load_prices
from scipy.optimize import differential_evolution
from fit_all_eppl_configs import build_model_fn

pr = load_prices("BitcoinPricesDaily.csv")
t = pr.df_full["years"].values; lp = pr.df_full["log_price"].values
m = t >= 1.0; tf, lpf = t[m], lp[m]
func, pn, bounds = build_model_fn(2, 2, ["d", "d"], ["u", "u"])
prev = None
for mit in (150, 300, 600, 1200):
    r = differential_evolution(lambda p: np.sum((lpf - func(tf, *p))**2),
                               bounds, maxiter=mit, tol=1e-8, seed=42, workers=1)
    r2 = 1 - np.sum((lpf - func(tf, *r.x))**2) / np.sum((lpf - lpf.mean())**2)
    d = None if prev is None else float(np.max(np.abs(r.x - prev)))
    print(f"maxiter={mit:5d} r2={r2:.6f} max|dparam|={d}")
    prev = r.x
```

- [ ] **Step 2: Run it.** `Run: btc_venv/bin/python3 tools/timemachine/spike.py` — record the chosen `EPPL_MAXITER`.

- [ ] **Step 3: Left-edge detection.** Cold-fit flagship-shape `ecfg_2dd_2uu` at as-of windows `ymax ∈ {2.5, 3.0, 3.5, 4.0}` years (≈ 2012-01 … 2013-07); the left edge is the earliest window whose r² ≥ 0.95 AND whose oscillator amplitudes `C_*` are all < their upper bound − 0.05 (not rail-pinned). Record the calendar date as `LEFT_EDGE_DATE`.

- [ ] **Step 4: BM per-frame cost + flagship policy.** Time one BM as-of build (Task 3's approach) on a truncated window; record ms. Decide `FLAGSHIP_POLICY`: if the flagship `eppl` class params differ from a fresh `ecfg_2dd_2uu` cold fit by max |Δparam| > 1e-2, set `"own_frame"` (fit it separately); else `"map_2dd2uu"`.

- [ ] **Step 5: Write findings + commit.** Write the four constants into a module docstring/`CONSTANTS` dict at the top of `tools/timemachine/spike.py`. `git add tools/timemachine/spike.py tools/timemachine/__init__.py && git commit`.

---

# PHASE 1 — Grid builder + storage (offline, dev)

### Task 1: Frame-date schedule

**Files:**
- Create: `tools/timemachine/frames.py`
- Test: `btc_web/test_timemachine_frames.py`

**Interfaces:**
- Produces: `frame_dates(left_edge: str, last_full_month: str) -> list[str]` — sorted, unique `YYYY-MM-DD`, quarterly (Jan/Apr/Jul/Oct 1st) through 2015-Q4, then monthly (1st) from 2016-01 to `last_full_month`.

- [ ] **Step 1: Write the failing test.**

```python
# btc_web/test_timemachine_frames.py
from tools.timemachine.frames import frame_dates

def test_quarterly_then_monthly_boundary():
    d = frame_dates("2013-01-01", "2026-07-01")
    assert d == sorted(set(d))                      # sorted, unique
    assert "2013-01-01" in d and "2013-04-01" in d  # quarterly early
    assert "2015-10-01" in d and "2015-11-01" not in d
    assert "2016-01-01" in d and "2016-02-01" in d  # monthly from 2016
    assert d[0] == "2013-01-01" and d[-1] == "2026-07-01"
```

- [ ] **Step 2: Run — expect fail** (`ModuleNotFoundError`). `Run: btc_venv/bin/python3 -m pytest btc_web/test_timemachine_frames.py -v`

- [ ] **Step 3: Implement `frame_dates`.**

```python
# tools/timemachine/frames.py
import pandas as pd

def frame_dates(left_edge: str, last_full_month: str) -> list[str]:
    le = pd.Timestamp(left_edge); lm = pd.Timestamp(last_full_month)
    q = pd.date_range(le, "2015-10-01", freq="QS")            # quarter starts
    m = pd.date_range(max(pd.Timestamp("2016-01-01"), le), lm, freq="MS")
    out = sorted({d.strftime("%Y-%m-%d") for d in list(q) + list(m)})
    return out
```

- [ ] **Step 4: Run — expect pass.**
- [ ] **Step 5: Commit** `tools/timemachine/frames.py btc_web/test_timemachine_frames.py`.

### Task 2: EPPL family as-of fitter

**Files:**
- Create: `tools/timemachine/fit_eppl_asof.py`
- Test: `btc_web/test_timemachine_eppl_fit.py`

**Interfaces:**
- Consumes: `fit_all_eppl_configs.build_model_fn`, `all_configs`; `EPPL_MAXITER` (Task 0).
- Produces: `fit_config_asof(cfg, t, lp, ymax, maxiter) -> dict` returning `{"params": {name: float}, "sigma": f, "r2": f, "n_log": i, "n_cal": i, "log_damps": [...], "cal_damps": [...]}`. **`sigma` is the constant band σ** = `float(np.std(residuals of data ≤ ymax))`. This is exactly what the runtime uses: `_ShrinkingBandsMixin._sigma_at` (btc_core/_base.py:38-40) returns the constant `self._sigma`, NOT the shrinking σ₀/α that `_init_shrinking_bands` fits (shrinking was reverted — "too narrow at late times"). So store ONE σ, not four params. Confirmed: no `_sigma_at` override exists on any EPPL class.

- [ ] **Step 1: Failing test — cold fit at right edge ≈ published config params.**

```python
# btc_web/test_timemachine_eppl_fit.py
import numpy as np
from model_toolkit.data import load_prices
from tools.timemachine.fit_eppl_asof import fit_config_asof

def test_rightedge_recovers_published_ecfg_1d_1u():
    pr = load_prices("BitcoinPricesDaily.csv")
    t = pr.df_full["years"].values; lp = pr.df_full["log_price"].values
    r = fit_config_asof((1, 1, ["d"], ["u"]), t, lp, ymax=t.max(), maxiter=300)
    assert r["r2"] > 0.97
    assert r["sigma"] > 0                       # constant band σ = std(residuals)
    assert r["params"]["B"] > 3                 # trend slope in a sane range
    assert r["n_log"] == 1 and r["n_cal"] == 1
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement `fit_config_asof`** — cold DE (`maxiter`, `tol=1e-8`, `seed=42`, `workers=1`) on `t≤ymax & t≥1`; then `sigma = float(np.std(lp_fit - model_fn(t_fit, *res.x)))` — the constant σ `_sigma_at` uses (matches `self._sigma = float(np.std(residuals))` set at btc_core/_base.py:32). Return the dict. Do NOT compute/store the shrinking σ₀/α — they are dead for the constant path.
- [ ] **Step 4: Run — expect pass.**
- [ ] **Step 5: Commit.**

### Task 3: BM as-of builder

**Files:**
- Create: `tools/timemachine/fit_bm_asof.py`
- Test: `btc_web/test_timemachine_bm_fit.py`

**Interfaces:**
- Consumes: the pipeline `tools/build_bm_model.py:264-294` uses — `fit_support`, `fit_sequential`, `classify`, `predict_future`, `build_composite`, `build_comp_by_n`, `fit_asymmetric_sigma` from `model_toolkit.*`, plus `load_prices` + the `PriceData` dataclass from `model_toolkit.data`. **`load_prices` has NO cutoff param** (`tools/model_toolkit/data.py:20`) — truncate the `PriceData` yourself.
- Produces: `fit_bm_asof(prices, ymax) -> dict` where `prices` is a full `PriceData` (from `load_prices`) and `ymax` is the as-of horizon in `years`. Truncates `prices` to `years ≤ ymax`, runs the pipeline, returns `{"comp_by_n": [[...]...], "t_grid": [...], "support_slope": f, "support_intercept": f, "sigma0_up": f, "alpha_up": f, "sigma0_down": f, "alpha_down": f, "bm_r2": f}`. BM legitimately uses the **4 ASYMMETRIC shrinking σ params** (it is a `_CompositeModel`; `_sigma_at` = `σ0·t^(−α)`, btc_core/_base.py:198-202) — unlike EPPL's constant σ, so all four are kept.

- [ ] **Step 1: Failing test — right-edge composite ≈ shipped `model_data.pkl`.**

```python
# btc_web/test_timemachine_bm_fit.py
import pickle
from tools.model_toolkit.data import load_prices
from tools.timemachine.fit_bm_asof import fit_bm_asof

def test_rightedge_bm_matches_shipped_within_tol():
    pr = load_prices("BitcoinPricesDaily.csv")
    r = fit_bm_asof(pr, ymax=float(pr.df["years"].max()))
    d = pickle.load(open("model_data.pkl", "rb"))
    assert abs(r["support_slope"] - d["bm_support_slope"]) < 0.05  # most stable BM invariant
    assert r["bm_r2"] > 0.9
    assert len(r["comp_by_n"]) >= 1 and len(r["t_grid"]) > 0
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement `fit_bm_asof`** — at the module top insert the repo `tools/` dir on `sys.path` (as Task 2 did) so `from model_toolkit... import ...` resolves. Add a `_truncate(prices, ymax) -> PriceData` helper that masks `df`/`df_full` to `years ≤ ymax` and rebuilds the arrays exactly as `load_prices` does (`tools/model_toolkit/data.py:103-108`). Then run the `build_bm_model.py:264-294` sequence on the truncated `PriceData`: `fit_support → fit_sequential → classify(n_major=5) → predict_future(t_last_data=trunc.years[-1], n_major=3, n_minor=1) → build_composite → build_comp_by_n → fit_asymmetric_sigma(np.log10(trunc.df_full["price"]), cbn[-1], comp.t_grid, trunc.df_full["years"])`. Assemble the return dict. Keep the calls identical to the shipped build so the right edge matches.
- **Early-frame robustness** (deferred Task-0 concern C2): a very early `ymax` may give `classify(n_major=5)` too few bubbles. Do NOT special-case it here — Task 4's grid build adds the continuity/degeneracy log. If the right-edge test passes, this task is done.
- [ ] **Step 4: Run — expect pass.**
- [ ] **Step 5: Commit.**

### Task 4: Grid assembly + parallel driver + storage

**Files:**
- Create: `tools/build_timemachine_grid.py`
- Test: `btc_web/test_timemachine_grid_build.py`

**Interfaces:**
- Consumes: Tasks 0–3.
- Produces: `timemachine_grid.json.gz` at repo root — `gzip`-compressed JSON of `{"frames": [YYYY-MM-DD...], "models": {model_key: [frame_record, ...]}}` where `model_key ∈ {"bub", "ecfg_...", "eppl"}`. Pure JSON (no pickle, no `dtype=object`). Keep total < 5 MB compressed (BM composites may be downsampled to ≤512 points per frame if raw JSON exceeds budget; gzip typically makes downsampling unnecessary).

- [ ] **Step 1: Failing test — a tiny 2-frame build round-trips.**

```python
# btc_web/test_timemachine_grid_build.py
import gzip, json
from tools.build_timemachine_grid import build_grid

def test_two_frame_build(tmp_path):
    out = tmp_path / "g.json.gz"
    build_grid(frames=["2016-01-01", "2016-02-01"],
               configs=[(1, 1, ["d"], ["u"])], include_bm=True,
               out_path=str(out), maxiter=150, workers=1)
    with gzip.open(out, "rt") as f:
        g = json.load(f)
    assert g["frames"] == ["2016-01-01", "2016-02-01"]
    assert "ecfg_1d_1u" in g["models"] and "bub" in g["models"]
    assert len(g["models"]["ecfg_1d_1u"]) == 2
    assert "params" in g["models"]["ecfg_1d_1u"][0]
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement `build_grid`** — for each config, cold-fit each frame **independently (cold — no warm reuse)**; parallelize (config,frame) tasks over a `ProcessPoolExecutor(max_workers=…)` capped at `min(nproc-2, 22)`. BM frames via Task 3. Assemble the plain-dict object graph and write with `with gzip.open(out_path, "wt") as f: json.dump(obj, f)`. Log dropped/failed frames explicitly (no silent caps).
- [ ] **Step 4: Run — expect pass.** Then a `--full` mode runs the real grid (documented, not run in CI).
- [ ] **Step 5: Commit** the builder + test (NOT the multi-MB grid yet — `timemachine_grid.json.gz` lands in Task 5's integration commit once the loader consumes it).

### Task 5: Runtime grid loader

**Files:**
- Create: `btc_web/timemachine.py`
- Test: `btc_web/test_timemachine_loader.py`

**Interfaces:**
- Produces:
  - `frames() -> list[str]` (cached, gzipped-JSON loaded once per worker)
  - `n_frames() -> int`
  - `asof_eppl(config_key: str, frame_idx: int) -> EPPLConfigModel` (per-request object built from grid params — Task 6 override)
  - `asof_bm(frame_idx: int) -> object` exposing `comp_by_n`, `t_grid`, `bm_r2`, support coefs, σ for the bubble BM-primary path
  - `available() -> bool` (False if `timemachine_grid.json.gz` missing → mode disabled)
- **Must NOT import matplotlib/kaleido. Must NOT mutate `_app_ctx.PRICE_MODELS`. Load with `gzip`+`json` — never pickle.**

- [ ] **Step 1: Failing test — loader returns per-request objects, singleton untouched.**

```python
# btc_web/test_timemachine_loader.py
import btc_web.timemachine as tm
from btc_web import _app_ctx

def test_asof_is_per_request_and_pure():
    if not tm.available():
        import pytest; pytest.skip("grid npz not built in this env")
    before = _app_ctx.PRICE_MODELS["ecfg_1d_1u"]._params.copy()
    a = tm.asof_eppl("ecfg_1d_1u", 0)
    b = tm.asof_eppl("ecfg_1d_1u", 0)
    assert a is not b                                   # fresh object each call
    assert _app_ctx.PRICE_MODELS["ecfg_1d_1u"]._params == before  # no mutation
```

- [ ] **Step 2: Run — expect skip or fail.**
- [ ] **Step 3: Implement loader** — `@lru_cache` a `gzip.open`+`json.load` of the grid; `available()` guards on file existence; `asof_eppl` constructs a fresh `EPPLConfigModel` via the Task-6 override each call.
- [ ] **Step 4: Run — expect pass/skip.**
- [ ] **Step 5:** Build the real grid (`btc_venv/bin/python3 tools/build_timemachine_grid.py --full`), then **commit** `btc_web/timemachine.py`, the test, AND `timemachine_grid.json.gz`.

---

# PHASE 2 — Model overrides + runtime substitution

### Task 6: Per-request param overrides on in-scope model classes

**Files:**
- Modify: `btc_core/_hybppl_eppl.py:815` (`EPPLConfigModel.__init__` only — the flagship `EntropyPPLModel` is NOT gridded; `"eppl"` in `bub-model-show` is always resolved to an `ecfg_*` key by `_resolve_eppl_master` before the bubble chart draws, so the flagship is never drawn on Tab 1)
- Test: `btc_web/test_timemachine_overrides.py`

**Interfaces:**
- Produces: `EPPLConfigModel(config_key, price_years, price_prices, quantiles, *, cfg_override=None, sigma_override=None)`. When `cfg_override` is a full cfg dict (`params`/`n_log`/`n_cal`/`log_damps`/`cal_damps`/`r2`), it bypasses `_EPPL_CONFIG_PARAMS`. When `sigma_override` (a float) is given, set `self._sigma = sigma_override` (the constant σ `_sigma_at` uses) and skip the residual-based band fit entirely. Overrides **shadow**, never mutate, class/global state.

- [ ] **Step 1: Failing test — override shadows, global dict untouched, σ applied.**

```python
# btc_web/test_timemachine_overrides.py
import numpy as np
from btc_core._hybppl_eppl import EPPLConfigModel, _EPPL_CONFIG_PARAMS

def test_cfg_override_bypasses_global_dict():
    base = _EPPL_CONFIG_PARAMS["ecfg_1d_1u"]
    ov = {**base, "params": {**base["params"], "B": base["params"]["B"] + 0.5}}
    t = np.linspace(1, 16, 500); p = 10.0 ** (0.0 + 5.0 * np.log10(t))
    m = EPPLConfigModel("ecfg_1d_1u", t, p, [0.5],
                        cfg_override=ov, sigma_override=0.13)
    assert m._params["B"] == base["params"]["B"] + 0.5
    assert m._sigma == 0.13
    assert _EPPL_CONFIG_PARAMS["ecfg_1d_1u"]["params"]["B"] == base["params"]["B"]
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement overrides** — add the two keyword params; when `cfg_override` present use it (deep-copied so the global is never aliased) in place of `_EPPL_CONFIG_PARAMS[config_key]`; when `sigma_override` present set `self._sigma = sigma_override` and skip `_init_shrinking_bands` (still build `self.fits`/`self.quantiles`/colors). Deep-copy any dict read.
- [ ] **Step 4: Run — expect pass.**
- [ ] **Step 5: Commit.**

### Task 7: Bubble-figure substitution + realized-price split

**Files:**
- Modify: `btc_web/figures/bubble.py` (overlay resolution `:214-225`, BM primary `:393-403`, realized price `:149-152`)
- Test: `btc_web/test_timemachine_bubble.py`

**Interfaces:**
- Consumes: `p["asof_date"]` — an int frame index, or `None`. `btc_web.timemachine` loader.
- Behaviour when `asof_date is not None`:
  - overlay `ecfg_*`/`eppl` keys resolve via `timemachine.asof_eppl(key, idx)` instead of `PRICE_MODELS.get` (`:225`);
  - the BM primary composite/support/σ come from `timemachine.asof_bm(idx)` instead of `m` (`:393`);
  - realized price (`:149`) is drawn as two segments: solid up to the frame date, faded/dashed after (the "how did it do" reveal).

- [ ] **Step 1: Failing test — as-of fan differs from live; no singleton mutation.**

```python
# btc_web/test_timemachine_bubble.py
import btc_web.timemachine as tm
from btc_web.figures.bubble import build_bubble_figure
from btc_core import load_model_data

def test_asof_changes_the_eppl_fan():
    if not tm.available():
        import pytest; pytest.skip("grid not built")
    m = load_model_data("model_data.pkl")
    base = dict(selected_qs=[0.5], active_models=["ecfg_1d_1u"], sigma_mode="constant",
                xscale="log", yscale="log", xmin=2012, xmax=2030, show_data=True)
    live, _ = build_bubble_figure(m, {**base, "asof_date": None})
    past, _ = build_bubble_figure(m, {**base, "asof_date": 0})   # earliest frame
    # the ecfg trace y-values must differ between live and earliest-frame fits
    def yvals(fig):
        return next(tr.y for tr in fig.data if "ECFG" in (tr.name or "").upper()
                    or "1D_1U" in (tr.name or "").upper())
    assert list(yvals(live)) != list(yvals(past))
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement the three hooks** behind `if p.get("asof_date") is not None:`. Keep the non-as-of path byte-for-byte unchanged (guard-only edits). Add a helper `_asof_resolve(model_key, p)` local to bubble.py.
- [ ] **Step 4: Run — expect pass.** Also run the full bubble figure test file to confirm no regression: `pytest btc_web/test_figures.py -v`.
- [ ] **Step 5: Commit.**

### Task 8: Params-dict wiring + cache-key alignment

**Files:**
- Modify: `btc_web/callbacks/charts/__init__.py` (both `dict(...)` calls, `:351` and `:394`), `btc_web/tab_defaults.py` (`_BUBBLE_RAW` + `bubble_defaults()`), `btc_web/app.py` (`_prewarm_caches` bubble key)
- Test: `btc_web/test_cache_key_alignment.py` (existing — must stay green), `btc_web/test_timemachine_defaults.py`

**Interfaces:**
- Produces: `asof_date` key present in both figure-builder dicts, defaulting to `None`, and in `bubble_defaults()`/prewarm.

- [ ] **Step 1: Failing test — default present & None.**

```python
# btc_web/test_timemachine_defaults.py
from tab_defaults import bubble_defaults
def test_asof_default_none():
    assert bubble_defaults().get("asof_date", "MISSING") is None
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Add `asof_date = _ci(asof_frame, None)`** (a new callback input, wired in Task 10) to both `dict(...)` blocks; add `"asof_date": None` to `bubble_defaults()` and the prewarm dict. Keep the two dicts structurally identical (the AST walker requires it).
- [ ] **Step 4: Run** `pytest btc_web/test_timemachine_defaults.py btc_web/test_cache_key_alignment.py -v` — both pass.
- [ ] **Step 5: Commit.**

---

# PHASE 3 — UI, callbacks, snapshot, interactions

### Task 9: Layout — Time Machine controls

**Files:**
- Modify: `btc_web/layout/bubble.py` (`_bubble_controls()` at `:187` — add the control block), `btc_web/tab_defaults.py` (`sd(...)` defaults for the new IDs)
- Test: `btc_web/test_timemachine_layout.py`

**Interfaces:**
- Produces components: `bub-timemachine-toggle` (dcc.Checklist, option `"on"`, default `[]`), `bub-asof-slider` (dcc.Slider, min 0, max `n_frames-1`, step 1, default `n_frames-1`), `bub-asof-play` (button), `bub-asof-interval` (dcc.Interval, disabled), `bub-asof-label` (Div, shows the active date), all inside a `bub-timemachine-body` Div hidden by default.

- [ ] **Step 1: Failing test — IDs exist.**

```python
# btc_web/test_timemachine_layout.py
from layout.bubble import _bubble_controls
def test_timemachine_ids_present():
    ids = _collect_ids(_bubble_controls())   # helper: walk .children for .id
    for i in ("bub-timemachine-toggle", "bub-asof-slider", "bub-asof-play",
              "bub-asof-interval", "bub-asof-label", "bub-timemachine-body"):
        assert i in ids
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Add the control block** (slider marks label only the year at quarter/decade boundaries to avoid clutter; body wrapped in `html.Div(style={"display":"none"})` — NOT `dbc.Collapse`, per project gotcha).
- [ ] **Step 4: Run — expect pass.**
- [ ] **Step 5: Commit.**

### Task 10: Callbacks — reveal, slider→param, play, single-model + MC/CTA exclusion

**Files:**
- Modify: `btc_web/callbacks/custom_time.py` (or a new `btc_web/callbacks/timemachine.py`), `btc_web/callbacks/charts/__init__.py` (add `Input("bub-asof-slider","value")`, `Input("bub-timemachine-toggle","value")` to `update_bubble`; compute `asof_frame`)
- Test: `btc_web/test_timemachine_callbacks.py`

**Interfaces:**
- Consumes: the Task-9 components.
- Behaviour: toggle reveals body (clientside style toggle); slider value → `asof_frame` passed to `update_bubble` (None when toggle off); ▶ toggles `bub-asof-interval.disabled`, each tick advances the slider by 1, stopping at max (clientside); when toggle on, hide the non-selected Display-Models options and hide the MC + CTA control blocks (clientside style).

- [ ] **Step 1: Failing test — asof_frame is None when toggle off, index when on.** Use a thin pure helper `_asof_frame(toggle, slider_val)` so it is unit-testable without Dash.

```python
# btc_web/test_timemachine_callbacks.py
from callbacks.timemachine import _asof_frame
def test_asof_frame_gate():
    assert _asof_frame([], 5) is None
    assert _asof_frame(["on"], 5) == 5
    assert _asof_frame(["on"], None) is None
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement** `_asof_frame` + the reveal/play/exclusion clientside callbacks + wire the two new Inputs into `update_bubble`, computing `asof_frame = _asof_frame(tm_toggle, asof_slider)` and threading it into both `dict(...)` calls (Task 8's `asof_date`). Use `prevent_initial_call` per project rules; clientside for pure visibility (ALARA).
- [ ] **Step 4: Run — expect pass.** Manually verify in `DEV=1` that dragging the slider redraws and ▶ animates.
- [ ] **Step 5: Commit.**

### Task 11: Snapshot / share-link encoding

**Files:**
- Modify: `btc_web/snapshot.py` (`_SNAPSHOT_CONTROLS`, `_TAB_CONTROLS["bubble"]`, `_CHECKLIST_OPTIONS`)
- Test: `btc_web/test_timemachine_snapshot.py`

**Interfaces:**
- Produces: `("bub-timemachine-toggle","value")` appended to `_SNAPSHOT_CONTROLS` + as an **append-only** entry in `_CHECKLIST_OPTIONS` (`["on"]`); `("bub-asof-slider","value")` appended (plain int). Both added to `_TAB_CONTROLS["bubble"]`.

- [ ] **Step 1: Failing test — round-trip.**

```python
# btc_web/test_timemachine_snapshot.py
from snapshot import _encode_snapshot_v4, _decode_snapshot_v4
def test_timemachine_roundtrip():
    st = {"bub-timemachine-toggle": ["on"], "bub-asof-slider": 42}
    dec = _decode_snapshot_v4(_encode_snapshot_v4(st))
    assert dec["bub-timemachine-toggle"] == ["on"]
    assert dec["bub-asof-slider"] == 42
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Append the entries** (checklist ID at END of `_CHECKLIST_OPTIONS`; controls at END of the bubble section — positional append, never reorder). Register today's fingerprint if defaults changed (`tools/update_defaults_registry.py`).
- [ ] **Step 4: Run** `pytest btc_web/test_timemachine_snapshot.py btc_web/test_snapshot.py -v`.
- [ ] **Step 5: Commit.**

### Task 12: Registration linter + caption + full-suite gate

**Files:**
- Modify: `btc_web/figures/bubble.py` (add the σ-band caption when `asof_date` set), any doc touch-ups
- Test: run the full suite

- [ ] **Step 1:** Add the caption *"Time Machine — model fit through {date}; bands use σ from data available then; price after {date} is what actually happened."* as an annotation when `asof_date is not None`.
- [ ] **Step 2:** `Run: btc_venv/bin/python3 tools/check_model_registration.py` — expect all ✓/KNOWN (no new holes).
- [ ] **Step 3:** `Run: btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'` — expect the two known pre-existing failures only (`test_free_tier_all_models`, `test_no_hex_literals_outside_colors_module`), everything else green.
- [ ] **Step 4:** Manual `DEV=1` smoke: toggle Time Machine, drag to 2017, confirm the fan freezes and realized price shows past it; ▶ animates; snapshot link restores the date.
- [ ] **Step 5: Commit.**

---

## Self-review notes (author)

- **Spec coverage:** grid (T1–5), cold-every-frame (T2/4), bands-not-resqr (T2/6/7), substitution+no-mutation (T5/6/7), UI (T9/10), single-model + MC/CTA exclusion (T10), snapshot (T11), realized price + caption (T7/12), left edge (T0). ✅
- **Deferred (Phase 2+ of the product):** resqr-as-of, LPPL/HybPPL families, arbitrary D, user ceiling, MC/tax integration — not tasked here.
- **Type consistency:** `asof_date` = int frame index everywhere (params dict, callbacks); `frame_idx` in loader; slider value is that index; snapshot stores the int. `cfg_override`/`band_override` names consistent T5↔T6.
- **Build-gate reminder:** every code block above compiles against the real modules named; the implementer runs each test to red before green. Phase 0 must complete before Phase 1 (it supplies `EPPL_MAXITER`, `LEFT_EDGE_DATE`, `FLAGSHIP_POLICY`).
