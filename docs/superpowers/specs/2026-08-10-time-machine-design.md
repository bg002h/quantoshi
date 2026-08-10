# Time Machine — as-of-date model view (v1 design)

**Written** 2026-08-10 · **Branch** `time-basis-toggle-phase2b` · **Status** design, pending user review

## Goal

Let a user pick a **past date D**, see the BM or an Entropy-PPL (EPPL) model **as it
would have been fit using only data through D**, projected forward — with the
**realized price after D drawn on top** so they can judge how the old projection
held up. A slider drags D through time (▶ play) so the fan visibly re-forms as
data accumulates.

Two purposes, weighted equally:
- **Backtest** ("how did it do?") — freeze the model at D, overlay what actually happened.
- **Animation** ("watch it evolve") — sweep D and watch the model learn.

## What it is / is not

- It adds **no new parameters** to any model. It is a *data-window* knob wrapped
  around models we already ship. (Contrast `spl`, which added one unidentifiable
  parameter — the identifiability trap does not apply here; there is nothing new
  to identify.)
- It is **not** F-9 (an evolving-support-line model). It is a meta-view.

## Scope (v1)

| In | Out (Phase 2+) |
|---|---|
| **BM** and the **EPPL family** (36 `ecfg_*` config variants + flagship `eppl`) | LPPL / HybPPL / PCA / Greedy families |
| Quantile **bands** as-of D | resqr bands as-of D (v1 uses shrinking/constant-σ) |
| **One** animated model at a time (Model A) | Multiple simultaneous animated models |
| EPPL Model B (config comparison) rides along free (see §7) | User-settable ceiling / other model knobs |
| Grid snapped to discrete precomputed dates | Arbitrary continuous D |
| Realized price overlay | MC / tax / Citadel integration |

EPPL is the star because it is the only in-scope family with rich, visibly
*evolving* structure (log-periodic + calendar oscillators). BM is the stable
reference.

## Architecture — fit-source substitution

As-of mode is a **modifier on the existing Tab-1 overlay pipeline**, not a bespoke
panel. For a chosen D:

- The model keys already in `bub-model-show` are resolved against the **grid's
  as-of-D fit** instead of the full-data `_app_ctx.PRICE_MODELS` singleton.
- Every active key flows through the *identical* trace-building path
  (`figures/common.py::build_overlay_traces` / `_resolve_model`), so BM, EPPL
  Model A, and (free) EPPL Model B are all frozen at D by one code path.

**Hard rail:** the as-of fit **resolves per-request; it never mutates
`PRICE_MODELS`.** That singleton is shared across every request in every gunicorn
worker — mutating it is a cross-user data race. `UserModel`/`u1` already has
exactly this per-request escape hatch (`_resolve_model`, `figures/common.py`); the
grid lookup mirrors it: build a lightweight per-request model object (or bind
grid params onto a throwaway instance) keyed by (model, D).

## The grid

### Definition
- **Frames:** **quarterly** from the first clean-fit date (§10) through **2015-Q4**,
  then **monthly** from **2016-01** to present. ~140 frames. Coarse where early
  data is sparse, fine where it matters.
- **Right edge = the live deployed model.** Dragging D fully right shows the
  already-computed `PRICE_MODELS` fit; only the *interior* frames come from the grid.
- **Immutable history.** The fit as-of 2019-01 depends only on data ≤ 2019-01,
  which never changes. So the grid is **backfilled once** and then only **extended**
  by one frame per month at the leading edge.

### Cold every frame (honesty decision)
Each frame is fit **cold** — from scratch, independently — so the fan is the true,
uncontaminated historical fit with **zero carryover** between frames. Warm-chaining
(seeding each month from the previous month's params) was rejected: its smoothness
is partly the optimizer clinging to last month, which could **lag a real regime
change or hide a real jump** — the same "false stability" failure `spl` taught us
to avoid.

- Cold fits **must be converged**, not merely fast: an under-converged cold fit
  jitters frame-to-frame from optimizer randomness (the mirror image of the warm
  trap). Pin `differential_evolution(maxiter=…)` at the build spike (where params
  stop moving); use a **fixed seed** for reproducibility.

### Cost (measured, dev, single-threaded, relaxed `maxiter=200, tol=1e-6`)

| fit | params | time |
|---|---|---|
| Cold DE, default `ecfg_1d_1u` | 9p | 8.3 s |
| Cold DE, heavy `ecfg_2dd_2uu` (flagship shape) | 16p | 14.9 s |

r² held 0.977–0.989 across windows, so chart-resolution precision is cheap.
Build = ~140 frames × 37 fits ≈ **5,180 cold fits**. The fits are independent →
config/frame-level parallelism across the **24-core dev box** (≈22 workers, the
`rebuild_caches.sh` pattern) brings the one-time build to **well under an hour** at
relaxed precision (a few hours if we converge harder — still a one-time dev job).
Monthly extend = 37 cold fits ≈ a few minutes. **Nothing recurring on prod.**

### Storage
Store fit **params** per (model, frame) — mean params + the 4 shrinking-band
params (σ₀/α up/down) + r² — and **reconstruct curves at runtime** (`_model_log10`
is vectorized, sub-ms). ~140 × 37 × ~20 floats < **1 MB as `.npz`** → small enough
to **git-track** alongside `model_data.pkl` (no rsync infra). A build script does
the once-backfill + monthly append.

## Bands as-of D

Use each model's own **shrinking-/constant-σ bands computed from residuals of data
≤ D — NOT resqr.** Rationale:
1. EPPL already bands this way natively (`_ShrinkingBandsMixin`); zero new work for the star.
2. resqr is a separate spline layer that would need re-fitting per frame and is unstable on short early windows.
3. Honest — bands use only residuals available at D.
4. BM and EPPL stay on one band mechanism.

**Caveat:** at D = today, BM's time-machine bands (σ) won't exactly match BM's live
**resqr** bands on the normal chart. A caption covers it: *"Time Machine bands use
σ from data available at each date."* resqr-as-of is a documented Phase-2 upgrade.

**BM rider:** BM's as-of mean reuses the fast, non-DE `model_toolkit` bubble build
on truncated data. Per-frame cost to be measured at the plan spike (cheap relative
to EPPL).

## UI — a Tab-1 mode called "Time Machine"

- A checkbox — **"🕰️ Time Machine"** — in the Tab-1 controls reveals a **date
  slider + ▶ play** (U₁'s "checkbox reveals a panel" pattern; clientside style toggle).
- The slider **snaps to grid frames** (quarterly early, monthly recent); ▶ steps
  frame-by-frame. Right end = live model.
- **Single-model selection:** entering the mode constrains the pick to **one**
  master — BM or one EPPL master — and **hides** the other Display-Models options
  while active. (Display Models is normally multi-select; the mode needs exactly
  one Model A.)
- **Realized price** from D → today is always drawn over the frozen fan — the
  "how did it do" judge.
- A caption shows the active as-of date and the σ-band note.

## Model B (EPPL config comparison)

The EPPL config panel (`layout/common.py::_global_eppl_modal`) already has **Model A**
(always on) and an opt-in **Model B (comparison)** — a second `ecfg_*` variant
overlaid to compare configs (`_resolvers.py::_resolve_eppl_master`). Under the
substitution architecture, **Model B is just another key in `model_show`**, so if
it is enabled it is frozen at D and **animates exactly like A, for free**. We do
**not** special-case it out (excluding it would be the extra work, and B off-by-default
means most users never see it). Constraint honored: **if B appears, it animates like A.**

## Interactions

- **Snapshot / share-links:** Time-Machine on/off **and** the date encode into share
  links (control state). Append to `_SNAPSHOT_CONTROLS` + `_TAB_CONTROLS["bubble"]`.
- **MC overlay:** mutually exclusive — MC (spaghetti) controls hidden while Time
  Machine is on (both are heavy Tab-1 modes).
- **Custom Time Axis:** mutually exclusive — CTA also re-fits; hidden while Time
  Machine is on.

## Left edge (first fittable date)

The slider starts at the first date the flagship EPPL **cold-fits cleanly** (sane
r² and stable oscillator params). Earlier Bitcoin history (2010–2011) lacks the
data for the log-periodic structure. Pinned at build time by fitting earlier and
earlier until the fit degrades, then stopping one frame back (expected ~2012–13).

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Warm-start fakes stability | **Cold every frame** (decided). |
| Under-converged cold → frame jitter | Converge `maxiter` (pin at spike); fixed seed. |
| Singleton mutation → cross-user race | Per-request resolution; mirror `u1` escape hatch. |
| resqr/σ band mismatch at right edge | Caption; resqr-as-of deferred to Phase 2. |
| Early frames degenerate | Left-edge gate (§10). |
| Backfill cost creep | One-time dev build, parallel on 24 cores; params-only storage. |

## Testing

- Grid builder: shapes/keys/monotone dates; params reproduce a known frame; a
  frame's `_model_log10` matches an independent recompute.
- Substitution: as-of resolution returns per-request objects and **never** mutates
  `PRICE_MODELS` (assert singleton identity/params unchanged after a request).
- Bands: as-of-D residual σ uses only data ≤ D.
- UI/callbacks: single-model constraint; MC/CTA hidden while on; snapshot round-trip
  of the date + toggle; realized-price trace present.
- Registration linter (`check_model_registration.py`) stays green.

## Open items for the plan spike (not blocking design)

1. BM per-frame as-of build cost.
2. The converged `maxiter` for cold fits (where params stabilize).
3. The exact left-edge date.
4. Whether the flagship `eppl` gets its own frame or maps to `ecfg_2dd_2uu`.

## Non-goals (explicit)

resqr-as-of; LPPL/HybPPL/PCA/Greedy families; arbitrary continuous D; user-settable
ceiling; MC/tax/Citadel integration; multiple simultaneous animated models. All
Phase 2+.
