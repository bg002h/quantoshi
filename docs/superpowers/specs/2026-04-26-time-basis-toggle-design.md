# Time-Basis Toggle: Block-Axis vs Calendar-Axis Canonical Models

**Date**: 2026-04-26
**Branch**: `time-basis-toggle`
**Status**: Design — pending implementation plan

---

## 1. Goal

A/B-test **block-indexed** vs **calendar-indexed** model fits as the canonical site-wide time axis for the Quantoshi BTC projection web app. The motivation is purely statistical: do parametric fits — especially the LPPL / HybPPL / EPPL family — explain BTC price more cleanly when the training axis `t` is measured in blocks since the origin block, vs years since the origin date?

Output (chart x-axes, share-link UI fields, year-based simulator state) **always remains calendar**. The toggle changes only what axis the model was trained on. Block-axis fitting reads real observed block heights for the training data and uses a deterministic protocol-target rate (144 blocks/day) for forward projection display.

---

## 2. Modeling decisions

| Decision | Choice | Rationale |
|---|---|---|
| Time origin | Same date in both axes — calendar `2009-07-25` ↔ block height at `2009-07-25` | Isolate the variable: only the axis changes, not the training window |
| Future block→date map | 144 blocks/day, fixed, deterministic | Determinism. Halvings land at exact protocol heights. **Cumulative drift** ~1.5 weeks total at 50-yr horizon (real rate ≈ 1.001× target × 50 yr) — noise vs price uncertainty |
| Past data | Real observed block heights from `BitcoinBlocksDaily.csv` | No proxy needed — the chain is the chain |
| LPPL log-osc terms | Phase-shift invariant under unit swap; refit re-finds equivalent solutions | `cos(ω·ln t + φ)` ⇒ `ln(α·t) = ln(α) + ln(t)` ⇒ `φ' = φ + ω·ln(α)` |
| HybPPL/LinPPL/EPPL **calendar-osc** terms | **Bounds rescale** from `rad/yr` to `rad/block`. Halving prior becomes `2π/(4 × 52596)` rad/block. **Treat amplitude collapse as a finding, not a bug.** | These terms are physically motivated by the calendar-derived halving cycle. Block-mode tests whether that motivation is empirically supported. |
| Damping `t^(-D)` | D refits freely on each axis; no rescaling expected to give same value | D is dimensionless but `t^(-D)` envelope is non-equivalent at t=15 (yr) vs t=789,000 (blocks). Different fitted D is expected. |
| S2F, EF | **Axis-exempt** — calendar-native in both site modes | S2F's `from_flow` is per-year. EF uses yearly bins. Refitting in blocks is not meaningful. **Block site mode loads the calendar-built `model_data_ef.pkl` unchanged.** |
| Bubble composite shape | `bubble_shape.py` rise/plat/decay phases refit in block coordinates as `t_rise/t_plat/t_decay/t_end`; **shape semantics** (parabolic top, log decay) are unit-invariant by construction (functions of `t/t_phase`, not absolute t). Phase durations themselves are free parameters in either mode. | Shape is normalized; only durations differ between axes. |
| UI model registry | **Axis-agnostic.** `_app_ctx.PRICE_MODELS`, `_HM_PILL_MODELS_BASE`, model display checklists are identical in both modes — same models, same labels, same pill bar. Only the underlying fitted parameters differ. | UI is a presentation layer over whichever pkl was loaded. |
| Toggle granularity | **Build-time only**; one axis canonical at a time; switching = rebuild + restart | Runtime per-request switching rejected: snapshot/cache complexity, no user-facing benefit for an admin A/B experiment |
| Output axis | Chart x-axis always calendar | Hard invariant. Display layer always converts t → calendar dates. |

---

## 3. Architecture

### 3.1 Configuration

**`quantoshi.toml`** — new admin preferences file at repo root, parsed via stdlib `tomllib` (Python 3.11+).

```toml
# quantoshi.toml — site-wide admin preferences
time_basis = "calendar"  # "calendar" | "block"

# Block origin pinned for reproducibility. Looked up once from
# BitcoinBlocksDaily.csv at the row where date == 2009-07-25.
# DO NOT derive at runtime — must match what build pipeline produced.
block_origin = 17448  # placeholder; resolved during Phase 1

# Forward block-rate constant. 144 blocks/day × 365.25 days/year.
# Used only to project future block heights from calendar dates
# (training uses real observed heights from BitcoinBlocksDaily.csv).
blocks_per_year = 52596
```

The file is **checked into git**. Admin flips a value, runs the rebuild pipeline, restarts. No env-var override (deliberate — keeps the canonical axis unambiguous in deployed state).

### 3.2 New module: `btc_web/time_basis.py`

Single source of truth for axis-derived constants and conversions.

```python
TIME_BASIS: Literal["calendar", "block"]
T_ORIGIN_DATE: date              # 2009-07-25
T_ORIGIN_BLOCK: int              # from quantoshi.toml
T_LABEL: str                     # "years" | "blocks"
T_PER_YEAR: float                # 1.0 | 52596.0
T_MIN: float                     # 1.0 | 52596.0 — replaces hardcoded `>= 1.0` masks

def to_t(date_or_block) -> float: ...      # date (cal) | int block (block)
def t_to_calendar(t: float) -> date: ...   # display: t → calendar date
def calendar_to_t(d: date) -> float: ...   # UI input: year → t

# Eager-loaded on import; graceful fallback to "calendar" if file missing.
```

The module is imported by `btc_core/`, `tools/build_*.py`, `btc_web/_app_ctx.py`, and `btc_web/cache.py`. The CTA (`engines/custom_fit.py`) **does not import `TIME_BASIS`** — its scale toggle remains per-fit user-controlled.

### 3.3 Cache fingerprint

`btc_web/cache.py::_fp()` extends to include `TIME_BASIS`. Calendar and block caches never collide. Worker startup invalidates anything keyed under the wrong axis.

### 3.4 Snapshot fingerprint

The 8-char registry fingerprint in `snapshot_defaults_registry.json` extends to include `TIME_BASIS`. Old `q3:` links computed under a different axis fail decode with a clear user-facing warning ("link from a different model axis"). Fields encoded in share links (year, percentile, BTC stack, etc.) are axis-independent at the UI layer; the fingerprint guard is defensive.

**Phase split:** Phase 1 reserves the `TIME_BASIS` slot in the fingerprint computation (so calendar-mode default keeps the same fingerprint it has today). Phase 3 enforces strict cross-axis decode rejection (so a link generated under block-mode cannot silently restore under calendar-mode after an admin axis flip). The split lets Phase 1 ship without producing user-visible link breakage.

### 3.4.1 QR cache key compatibility

`_quantize_params(p)` exempts `selected_qs` and `exit_qs` from `_q3()` rounding because those keys must match `qr_fits` exactly. Block-mode regenerates `qr_fits` with the same quantile-string keys (`"0.01"`, `"0.05"`, …, `"0.99"`); the keying is axis-invariant. **No QR keying changes needed.** Defensive test in Phase 1 verifies that block-mode `qr_fits` keys are byte-identical to calendar-mode keys.

### 3.5 Pkl schema (Phase 1 widens)

`model_data.pkl` gains top-level fields:

```python
{
  "time_basis": "calendar" | "block",
  "t_label": "years" | "blocks",
  "t_per_year": 1.0 | 52596.0,
  "t_origin": "2009-07-25" | 17448,
  # ... existing fields, with `years` alias retained in calendar mode
  #     for back-compat with consumers we haven't updated yet
}
```

Calendar pkl files retain the `years` field as an alias for `t` so older code paths keep working through Phase 2. **Block pkl files do NOT carry a `years` alias** — block-mode is a new code path; consumers must use `t` directly. Code that reads `years` from a block pkl is a bug to be caught in Phase 2.

### 3.6 Filename convention

Block artifacts use a `_block` suffix:
- `model_data.pkl` (calendar canonical) | `model_data_block.pkl` (block canonical)
- `model_data_ef.pkl` — **calendar only** (EF is axis-exempt per §2; block site mode loads this same file unchanged)
- `mc_cache/` (calendar) | `mc_cache_block/` (block)
- `citadel_band_cache/` (calendar) | `citadel_band_cache_block/` (block)

The runtime loads whichever set matches `TIME_BASIS`. The other set, if present on disk, is ignored. Phase 2 produces the block artifacts as parallel files without the runtime touching them.

### 3.7 Simulator boundary (HARD non-goal)

**Citadel, the MC year-stepping logic, and the tax/RMD/inflation loops remain calendar-native in both site modes.** UI controls (`cp-start-yr`, `cp-end-yr`, `dca-yr-min/max`, `ret-yr-min/max`, `sc-start-yr`, etc.) stay calendar. The simulator advances its own state in calendar years and converts each annual evaluation point to `t` at the boundary where it calls `model.predict(t)`. This conversion goes through `time_basis.calendar_to_t()`.

This is **load-bearing**: tax brackets are inflation-indexed per calendar year; RMD start age depends on birth year; halvings (in Citadel's regime model) are calendar-anchored events at well-known dates. Trying to make the simulator block-native would entangle US tax law with block height, which is absurd. The block-axis experiment is about whether **price models** explain price more cleanly when trained on blocks; it is not about making the entire site block-native.

#### 3.7.1 Markov transition matrix under block mode

The Markov MC engine (`btc_web/markov.py`) trains its transition matrix from observed (price, t)-percentile-bin transitions over a sliding window. Variables `window_years`, `MIN_WINDOW_YEARS`, `step_days=30` are calendar-flavored. **The transition matrix stays calendar — it is rebuilt against `_app_ctx.PRICE_MODELS["bub"]` regardless of axis.**

But there's a subtle dependency: when `bub` is block-trained, the percentile bins themselves are *evaluated* using the block-trained model's `_q(t)`, with `t` obtained via `calendar_to_t(date)` for each calendar tick of the training window. So the bins are axis-aware (use the active model), but the *training window* and *step size* are calendar. This produces a transition matrix that is conceptually "what does the block-trained bub model say about how percentiles transition over calendar weeks?" — which is what the simulator needs.

Phase 4 task: rebuild the Markov transition matrix once after switching to block-mode. Phase 4 spec includes an MC parity test: identical scenario, block vs calendar bub model, terminal-distribution KS test should show fit-quality differences only.

---

## 4. Five-phase staged refactor

Each phase is one PR. Each is shippable on its own with `time_basis = "calendar"` (no behavior change). The site is never broken in between.

### Phase 1 — Plumbing (~6 files, low risk)

- Create `quantoshi.toml` with `time_basis = "calendar"`.
- Create `btc_web/time_basis.py` with constants + conversion helpers.
- Resolve `T_ORIGIN_BLOCK` (look up block at 2009-07-25 in `BitcoinBlocksDaily.csv`, pin in TOML).
- Extend `cache.py::_fp()` to include `TIME_BASIS`.
- **Reserve** the `TIME_BASIS` slot in the snapshot fingerprint computation (does not change the calendar-mode hash; defensive only).
- Widen `model_data.pkl` schema to carry `time_basis`, `t_label`, `t_per_year`, `t_origin`.
- Test: `engines/custom_fit.py` does not import `TIME_BASIS` (CTA stays per-fit).

**Default `calendar`. No fits change. No charts change.**

### Phase 2 — Parameterize the build pipeline (~25 files, highest risk in T_MIN sweep)

- `T_MIN` sweep across `btc_core/_simple.py`, `_lppl.py`, `_hybppl_eppl.py`, `_basis.py`, `_base.py`. Replace every `price_years >= 1.0` (and `1.0/365.25`) with axis-aware `t >= T_MIN`. ~12 classes affected.
- Public API rename: `price_years` → `t` on every model `__init__` and `_q(t)` call site. Update docstrings.
- Bound rescaling: `tools/fit_*.py` for HybPPL/LinPPL/EPPL — calendar-osc `W_cal` bounds + halving prior rescaled when `--time-basis=block`.
- `tools/model_toolkit/{data,fitting,composite,bands,export,prediction}.py`: parameterize over `t_label` + `t_per_year`. Replace hardcoded `/365.25`, `GENESIS`, `date_rise/plat/decay/end` calendar derivations with axis-aware equivalents.
- `tools/build_bm_model.py`, `tools/build_ef_model.py`, `tools/refit_all_ppl.py`, all `tools/fit_*.py`: accept `--time-basis={calendar,block}`.
- Build `model_data_block.pkl` as a **parallel artifact** — site does not load it yet. (No `model_data_ef_block.pkl`; EF is axis-exempt per §2.)
- Compute and report **R², AIC, OOS-RMSE, AND calendar-oscillator amplitude (C/B ratio)** for each model on each axis. Save comparison report to `docs/superpowers/specs/time_basis_phase2_results.md`.
- S2F + EF flagged axis-exempt — refit only in calendar; block-mode loads calendar version of these.

**Decision point.** If block-axis does not improve fit quality on the parametric models that benefit from it (LPPL log-osc, EPPL, PCA basis), and the calendar-osc collapse hypothesis is either confirmed or refuted, **stop here**. Sunk cost is just the refactor (which leaves the codebase cleaner regardless). Phases 3–5 are skipped.

### Phase 3 — Runtime axis loader (~15 files)

- `btc_web/_app_ctx.py`: load `model_data_block.pkl` when `TIME_BASIS == "block"`.
- `btc_web/snapshot.py` + `snapshot_defaults_registry.json`: **enforce** the snapshot fingerprint guard (Phase 1 reserved the slot; Phase 3 makes cross-axis decode reject with a user-facing warning).
- `btc_web/figures/common.py`: `_resolve_model`, edge annotations, MC overlay integration — verify all use `t` agnostically and convert via `t_to_calendar` on output.
- `btc_web/mc_overlay.py`: cached MC paths are calendar-trained; ensure block-mode site doesn't try to overlay them on a block-trained model without re-binning. (This is the "MC stays calendar" half of §3.7.)
- `update_prices.py`: rebuilds **whichever pkls already exist on disk**. If `model_data.pkl` is present, rebuild it. If `model_data_block.pkl` is present, rebuild it too. This means an admin who's done Phase 2 (calendar canonical, block parallel) gets both refreshed daily; an admin who only ever runs calendar gets calendar; an admin who's flipped to block gets block. **Manual catch-up rule:** when an admin first creates the alternate axis pkl (e.g., switching from calendar-only to running both), they must run the build pipeline once explicitly; the cron does not auto-create.
- Test parameterization: parameterize the ~1456-test suite over both axes for the touched modules. **Highest risk** — `test_cache_key_alignment.py` and figure-builder tests need `TIME_BASIS` fixtures.
- Admin flips `time_basis = "block"` in dev, full E2E pass.

### Phase 4 — Heavy caches (~3h MC + ~3h Citadel)

- Block-mode MC cache built from block-trained Bubble model.
- Citadel band cache rebuilt against block-trained model. **Citadel sim itself stays calendar (§3.7)** — only the band-edge price evaluations use the block model.
- Verify: same Citadel scenario produces ~equivalent calendar-projected results in both modes. Differences should be attributable to fit quality only.
- rsync to prod paths under `mc_cache_block/` and `citadel_band_cache_block/`.

### Phase 5 — Validation + decision (operational only, no code)

- Side-by-side comparison: chart-by-chart visual diff, fit-metric report, MC distribution comparison.
- Decide whether `time_basis = "block"` becomes the prod default.
- Either way, the toggle stays — the experiment is reproducible.

---

## 5. Data flow examples

### Calendar mode (default)

```
User enters "2031" in dca-yr-min
→ calendar_to_t(2031-01-01) = 21.43 (years since 2009-07-25)
→ model.predict(t=21.43) = log10_price
→ chart plots (2031-01-01, price)
```

### Block mode

```
User enters "2031" in dca-yr-min
→ calendar_to_t(2031-01-01) = 1,127,003 (block height projection from 144/day)
→ model.predict(t=1,127,003) = log10_price
→ chart plots (2031-01-01, price)        ← x-axis still calendar
```

### Citadel mode (axis-independent)

```
Citadel sim loop: for cal_year in range(start_yr, end_yr+1):
    t = calendar_to_t(date(cal_year, 1, 1))    # axis conversion at boundary
    price = model.predict(t)                    # block or calendar — model knows
    apply_tax_brackets(cal_year)                # ALWAYS calendar
    apply_inflation(cal_year - last_year)       # ALWAYS calendar
    advance_state(cal_year)                     # state is calendar-keyed
```

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| `BitcoinBlocksDaily.csv` has gaps before 2009-07-25 | Phase 1 sanity check; backfill with `tools/build_block_map.py --append` if needed |
| Hidden calendar bounds in model fit drivers | Phase 2 bound-rescaling pass; review every `tools/fit_*.py` for hardcoded year-scale priors |
| `T_MIN` sweep misses a hardcoded threshold | Test fixture parameterized over both axes catches silent regressions on extreme-quantile data |
| Block-mode model_toolkit produces malformed pkl (e.g., `date_rise` fields meaningless in block-time) | Phase 1 pkl schema widening; `date_rise/plat/decay/end` get block-equivalents (`t_rise`, `t_plat`, `t_decay`, `t_end`) |
| Cython `markov.so` calendar-tied | Markov stays calendar (§3.7); no `markov.so` rebuild needed for this work |
| Phase 2 surface larger than estimated | Time-box Phase 2; the shippable Phase 1 plumbing has standalone value if Phase 2 stalls |
| Snapshot links from before Phase 1 | Fingerprint guard refuses cross-axis decode with clear warning, falls back to defaults — same pattern as existing `q3:` legacy handling |

---

## 7. Non-goals (explicit)

- Runtime per-request axis switching. Build-time only.
- Block-native Citadel simulator. Citadel stays calendar in both site modes.
- Block-native MC year-stepping. MC scenarios stay calendar; the toggle affects only the underlying model the MC paths are sampled against.
- Block-native tax / RMD / inflation. US tax law is calendar-anchored.
- Re-optimizing the time origin in block space. We use the same date in both axes for variable isolation. Re-optimization is a follow-up if block-axis wins under (a).
- Refitting S2F or EF in block mode. Both are calendar-native by construction.
- A user-facing UI control for the toggle. Admin-only via TOML.

---

## 8. Open items resolved during Phase 1

- Look up exact block height at `2009-07-25` from `BitcoinBlocksDaily.csv`. Pin in `quantoshi.toml`.
- Confirm `BitcoinBlocksDaily.csv` covers full price history (no gaps).
- Decide `T_PER_YEAR` precise value: 144 × 365.25 = 52,596 (exact protocol). Pin in `quantoshi.toml`.
- Decide pkl schema field names: `t`, `t_label`, `t_per_year`, `t_origin` (proposed).
