# R0 Adversarial Design Review — Time-Basis Toggle, Remaining Phases 2b.ii → 2e

**Date**: 2026-07-10
**Reviewer**: fable (adversarial, read-only)
**Source spec**: `docs/superpowers/specs/2026-04-26-time-basis-toggle-design.md`
**Decisions log**: `docs/superpowers/plans/2026-04-26-decisions-log.md`
**Charge**: Hard R0 gate before any implementation plan for phases 2b.ii (family refits) → 2c (runtime loader) → 2d (heavy caches) → 2e (flip + deploy).

---

## Verdict

**NOT ship-ready.** The remaining design has two structural holes that are missing architecture, not implementation details: (1) the runtime calendar→block conversion is mathematically inconsistent with the training axis (uniform 144/day vs. real chain history — measured error ~5% *today*, non-monotonic sign over history), and (2) the PPL model-parameter storage architecture (regex-updated class attributes in `btc_core` source, single copy) is incompatible with the spec's parallel-artifact scheme and with D11's "reversible at the TOML level" claim. Additionally, the spec's quantitative premise about block-rate drift is wrong by ~30×. The spec needs amendments before an implementation plan for 2b.ii–2e is written.

---

## Critical (would-ship-broken)

### C1 — CONFIRMED: Uniform-rate `calendar_to_t` is inconsistent with the real-block training axis, for past dates AND today

Training uses **real observed heights**: `tools/model_toolkit/data.py:88` — `df["years"] = (df["blockheight"] - T_ORIGIN_BLOCK).astype(float)`. Runtime converts **every** date at the uniform protocol rate: `btc_web/time_basis.py:102-104`:

```python
days = (d - T_ORIGIN_DATE).days
years = days / 365.25
return years if TIME_BASIS == "calendar" else years * T_PER_YEAR
```

Measured against `BitcoinBlocksDaily.csv` (origin 20188):

| Date | Real offset | Protocol projection | Error |
|---|---|---|---|
| 2010-07-17 (fit_min!) | 48,591 | 51,408 | **−5.5%** |
| 2011-02-09 | 86,977 | 81,216 | **+7.1%** |
| 2026-06-03 (≈today) | 932,078 | 886,608 | **+5.1%** |

The error is non-monotonic (early chain slower than target, post-2010 chain ~4–5% faster), so no constant correction exists. The block QR fit in the actual `model_data_block.pkl` has slope **5.176** (verified by loading the pkl: intercept −25.84, `price_years` 48,591→925,661, confirming real offsets). So evaluating "today" at 886,608 instead of 932,078 understates the model price by 1.0513^5.176 ≈ **×1.30 (~23% low)** — every day, forever. The model is systematically evaluated at t-values displaced from its training support.

Concrete failure surfaces (all currently calendar-only, but even after a naive `×T_PER_YEAR` fix in Phase 2c they'd inherit this error): live ticker current-quantile (`callbacks/ticker.py:55`, `callbacks/charts/__init__.py:1019`, `utils.py:427`), scanner date solving (`callbacks/scanner.py:50-52`), heatmap entry-t (`figures/heatmap.py:215,249,284`), today-lines (`figures/bubble.py:436`), and historical model-curve alignment (curve evaluated at protocol-t vs. history scattered at real-t → visible weeks-months displacement around 2011, plus spurious residual structure tracking hashrate history).

**Spec premise is factually wrong**: §2 claims "real rate ≈ 1.001× target … cumulative drift ~1.5 weeks total at 50-yr horizon." Measured real/protocol ratio is **1.043–1.051×** (~10–15 days of drift *per year*). Off by ~30×.

**Fix direction**: a piecewise map in `time_basis.py` — real CSV lookup for dates ≤ last CSV row, protocol-rate extrapolation **anchored at the last real block** (not the 2009 origin) for future dates. Nothing currently gives the runtime this map outside `engines/custom_fit.py` (verified: only `custom_fit.py` and `callbacks/custom_time.py` read `BitcoinBlocksDaily.csv` in `btc_web/`); `custom_fit._BLOCKS` (date-aligned, read-only, health-checked) is reusable prior art, but the seam belongs in `time_basis` to avoid a CTA dependency. Note: the CSV **starts at 2010-07-17** (row 1 = 68779), so 2009-07-25→2010-07-16 needs interpolation between 20188 and 68779 — moot for display (history starts at fit_min = 2010-07-17) but the map must not crash there.

### C2 — CONFIRMED: The entire runtime boundary uses calendar-only `btc_core.yr_to_t`/`today_t`; spec §3.7 describes a conversion that does not exist

Spec §3.7: "This conversion goes through `time_basis.calendar_to_t()`." Reality: ~50 call sites in 14+ files use `btc_core._helpers.yr_to_t` (line 160, pure calendar) and `today_t` (line 168), then pass the result straight into `model.price_at(q, t)`:

- `engines/citadel_sim.py:19-20, 252-253` (`t0 = yr_to_t(config.start_yr)`); worse, Citadel's `state.t` advances **in year units** (`engines/citadel_step.py:137-139`: `dt = 1.0/ppy; new.t += dt`) and is fed directly to `model.price_at` (`citadel_step.py:34`) — the conversion point the spec mandates has no hook.
- `figures/common.py:349-351` (`_build_time_array`: `ts` is simultaneously the x-axis array and the model-eval array), `:469,473` (`_year_ticks`)
- `figures/bubble.py:107-108,436`, `figures/heatmap.py:215,249,284,403,557-587`, `figures/residuals.py:42-43,163`
- `mc_overlay.py:365-366,719,1075-1098,1255,1440`, `mc_cache.py:221-240`, `markov.py:13`
- `callbacks/charts/__init__.py:466-467,1019`, `callbacks/scanner.py:8,50-52`, `callbacks/ticker.py:55`, `utils.py:427`
- Cache generators: `tools/generate_citadel_bands.py:87-101`, `generate_mc_cache.py:56` (Phase 2d inputs)

Failure scenario after a naive Phase 2c flip: `price_at(0.5, t≈16.9)` against the block model → log10(price) = −25.84 + 5.176·log10(16.9) ≈ −19.5 → price ≈ 10⁻¹⁹·⁵ ≈ $0. Every simulator, figure, and readout collapses. The spec's Phase 3 file list (~15 files) omits scanner, ticker, utils, heatmap, residuals, markov, mc_cache, and both citadel engine files. `time_basis.py:130,143` acknowledges "Phase 2c will consolidate," but the spec must enumerate the surface and decide the mechanism (consolidate `btc_core.yr_to_t` to be axis-aware vs. per-site swap), and the mechanism must ride on C1's piecewise map or the ~5% error is baked into every boundary.

### C3 — CONFIRMED: PPL parameters are single-copy class attributes regex-rewritten in `btc_core` source — the parallel-artifact scheme cannot hold them, and the monthly refit timer will clobber a block-mode site

`tools/fit_linppl.py:104-146` (and siblings) update fitted params by regex **in place** in `btc_core/_lppl.py` / `_hybppl_eppl.py` class bodies (`btc_core/_lppl.py:34-39`: `_A = -1.153754 …`). The runtime instantiates from these attrs plus the pkl's history (`app.py:304-331`). Consequences:

1. **No parallel block artifact is possible** for LPPL/LinPPL/HybPPL/EPPL/config-families (72 config variants via `_HYBPPL_CONFIG_PARAMS`/`_EPPL_CONFIG_PARAMS`) under spec §3.6's "parallel files" scheme. Refitting on block-t overwrites the calendar params in git-tracked source.
2. **D11's reversibility claim is false**: "reversible at the quantoshi.toml level — flip back and rebuild" — flipping back also requires reverting `btc_core` source params.
3. **After 2e**, if `model_data_block.pkl` is loaded while class attrs are calendar-fit (or vice versa after the monthly timer runs), predictions are garbage: calendar A=−1.15/B=5.08 at t=932,078 → log10(price) ≈ 29 → $10²⁹.
4. `tools/refit_all_ppl.py` (runs monthly via `quantoshi-ppl-refit.timer`) has **zero** time-basis support (verified by grep) and shells out to fit tools whose data load defaults to calendar (`fit_linppl.py:41` — `load_prices("BitcoinPricesDaily.csv")`, no `time_basis` arg). Post-flip, the 1st-of-month refit silently reverts the site's models to calendar-axis params.

**Fix direction**: 2b.ii must first decide param storage — move fitted params into the pkl (cleanest; makes the artifact scheme coherent), or axis-suffixed attr blocks — and thread `--time-basis` through `refit_all_ppl.py` + every `fit_*.py`, and gate/disable the systemd timers until they're axis-aware. This is unbudgeted architectural work the spec doesn't mention.

### C4 — CONFIRMED: Fit-tool bounds are calendar-hardcoded far beyond `W_cal` — block-mode fits cannot even represent the solution

`tools/fit_linppl.py:56-63`: `A ∈ (−3.0, 1.0)`, `B ∈ (3, 7)`, `W_cal ∈ (0.5, 10.0)`, plus `mask = t >= 1.0` at line 45 (T_MIN sweep never touched `tools/fit_*.py`). The block-axis intercept is ≈ **−25.8** (measured from the block pkl's QR fit; A shifts by −B·log10(52596) ≈ −4.72·B). Differential evolution with A capped at −3.0 **cannot reach the block-axis optimum at all** — you get boundary-pinned garbage, not a "finding." `W ∈ (0.5,10) rad/block` means oscillation periods of 0.6–12.6 *blocks* against ~144-block daily sampling (Nyquist ≈ 0.022 rad/block) — pure aliasing. Same pattern in `fit_hybppl.py:55` and, plausibly, all ~20 `fit_*.py` (PLAUSIBLE for the ones not read; the two read are conclusive). The spec's §2/§6 rescaling note covers only W bounds + halving prior; the A-shift, the `t >= 1.0` masks, and C/D envelope scale interplay are unaddressed. The 2b.ii plan must specify the full bound-transformation rule per parameter, per tool.

---

## Important

### I1 — CONFIRMED: Cache fingerprint hardcodes `model_data.pkl`
`_app_ctx.py:154-158` fingerprints `"model_data.pkl"` mtime/size regardless of axis. Post-flip: daily calendar rebuilds churn the fingerprint (needless cache flushes) while block-pkl rebuilds *don't* invalidate anything. Must select the pkl per `TIME_BASIS`. Not in the spec's Phase 2c list.

### I2 — CONFIRMED: Ops pipeline rebuilds only calendar artifacts; block pkl is already stale in-tree
`update_prices.py:141-167,265` rebuilds `model_data.pkl` + `model_data_ef.pkl` only; the spec's "rebuild whichever pkls exist" is unimplemented. Live evidence: working tree has `model_data.pkl` modified (June 3 data) while `model_data_block.pkl` last block = 925,661 ≈ **2026-05-20** — 2 weeks stale relative to its sibling. Post-2e the cron would refresh the *inactive* pkl daily and leave the canonical one frozen. Also: `quantoshi-cache-rebuild` timers and `generate_mc_cache.py`/`generate_citadel_bands.py` (Phase 2d) are axis-blind. The bitcoind deploy dependency is real but already managed on this branch — `daily_update.sh:85-95` aborts (3 retries + notify) if `build_block_map.py --append` fails, and enforces price/block row parity (`:105-115`).

### I3 — CONFIRMED: Branch topology is a trap; do not merge `time-basis-toggle-phase2b` wholesale
Fork point 645db69 (2026-04-27). Since then **both** branches accumulated independent daily-update commits touching the same binaries (CSVs/pkls) and **divergent `daily_update.sh` rewrites** — phase2b: `030ef07`/`23a870b`/`520ed02`; main (`time-basis-toggle`): `bae74a5`/`327460c`/`f3661f0`/`32a7936`/`a0b3c4d` — plus main-only features (halving lines `908fdb7`, projection-table monthly rebuild, "prevent prod block-map decay"). A merge/rebase guarantees binary conflicts and semantic conflicts in ops scripts. **Recommendation**: cut the new feature branch from `time-basis-toggle` HEAD (`28a1274`), cherry-pick only the Phase 2b.i code commits (`771aaa6`, `ed593a0`, `fac6b0f`, `ba8a56f`, `8b53e0b`, `21b2423` + plan docs), rebuild `model_data_block.pkl` fresh from current CSVs, and reconcile `daily_update.sh` semantics explicitly (main's version is presumably what prod runs). Note the phase2b `daily_update.sh` work (SETTLE_LAG, partial-update guard) may contain fixes main lacks — diff them deliberately, don't let either silently win.

### I4 — CONFIRMED: Cross-axis snapshot rejection has no mechanism to distinguish "old defaults" from "other axis"
`snapshot_defaults.py:391-396` hashes `TIME_BASIS` into the fp (Phase 1 slot, done), but decode fallback (`snapshot.py:679-699`) resolves *any* historical fp from the registry and silently restores. After an axis flip, calendar-era links decode via the registry path with **no** "different model axis" warning — the registry entries carry no axis tag. Spec §3.4's promised warning needs a design: tag registry entries with axis, or embed axis in the link prefix.

### I5 — CONFIRMED: Chart-x-coupled calendar literals in user-facing interactive paths (Phase 2c must own these; spec omits them)
- `callbacks/user_model.py:9,39-41,185-186`: `_GENESIS_YR = 2009.56`, click `t_val = pt["x"]` assumed years, `t1 = p1y − _GENESIS_YR` then `UserModel.from_points` fits that year-t against `M.price_years` — which in a block pkl holds **block offsets** → slope/intercept nonsense. Also `btc_core/_simple.py:572` `mask = price_years >= 0.5` becomes a no-op in block mode (silently *includes* sub-year history the calendar mode excluded — a distribution change, not a crash).
- `callbacks/scanner.py:50-52`: `brentq(f, 0.5, 72.0)` (years bracket) then `genesis + t*365.25 days`; clientside JS at `:284-295` re-derives dates from `t * MS_PER_YR`.
These are all consequences of the codebase invariant the spec never states explicitly: **chart x-values ARE t-in-years**, not datetimes (confirmed by the `_add_date_hover` 0.3–120 heuristic at `figures/common.py:789` and `_year_ticks` at `:456-475`). Phase 2c must declare "chart-x stays calendar-years" as the contract and convert to block-t only inside model evaluation.

### I6 — D11 / A/B gate: skipping it is a documented decision but an undocumented risk in the spec
CONFIRMED: no comparison artifact exists (`docs/superpowers/specs/time_basis_phase2_results.md` absent); the user's supporting research (`qstar_*`, `blocksweep`, etc., master commit `25eecff`) is not on this branch and was never distilled into a committed report. The spec §4 still says "Decision point … stop here" — spec and decisions log now contradict each other. **Recommendation: resurrect the report as a hard gate before 2e** (not 2b) — once both pkls coexist it's cheap, and it doubles as the regression check that catches C4-type degenerate block fits before they hit prod. Flipping the public default with zero committed quantitative evidence is exactly the kind of thing this review process exists to stop.

### I7 — Spec text now wrong or moot (drift since 2026-04-27)
- §3.1/§8: "block_origin … looked up from `BitcoinBlocksDaily.csv` at date == 2009-07-25" — **impossible**; CSV starts 2010-07-17. Actual source: bitcoind RPC (D1). §6's "gaps before 2009-07-25" risk mischaracterizes total absence of pre-2010-07-17 coverage.
- §2 drift estimate wrong (see C1).
- §3.1 "No env-var override (deliberate)" — contradicted by `QS_TIME_BASIS` in `time_basis.py:60-73` (added in 2b.i, `771aaa6`, for build tooling). Legitimate, but it weakens the "deployed state unambiguous" invariant: a stray env var in a service unit flips a worker's axis. Spec should be amended to bless it as build-tool-only and require it never appears in service environments.
- §3.7/§3.7.1 describe boundary conversions (`calendar_to_t`) as existing; they don't (C2).
- New main-branch feature the spec doesn't know: halving lines (`908fdb7`) estimate **future halvings by calendar cadence anchored 2024-04-20** (`_HALVING_KNOWN_DATES`/`_HALVING_ANCHOR` in main's `figures/common.py:159-184`) — under block mode these should derive from 210000·k through the block→date map. Small, but it's a brand-new calendar literal added after the T_MIN sweep.
- §2 halving prior "2π/(4 × 52596)" = 2π/210384 is conceptually wrong: the halving period in block space is **exactly 210000 blocks** → 2π/210000 (2.99199e-5 vs 2.98653e-5). Numerically small over current history (~0.05 rad phase), but the spec's framing ("cal-osc terms are calendar-motivated; expect collapse") has the physics backwards — halvings are *block-native by protocol*; block mode should make these terms sharper if the halving hypothesis is real. The 2b.ii plan should center the prior at 2π/210000 and treat *strengthening* as the expected outcome to test.

---

## Minor / Nit

- **CONFIRMED** `_custom_time_presets.py:6` docstring: "len(CAL_PRESETS) == 6, len(BLK_PRESETS) == 5" — actual 7/6; `test_custom_time.py:308-309` asserts 7/6. Stale docstring only.
- **CONFIRMED** BLK_PRESETS label errors (`_custom_time_presets.py:24-31`): `block_3300 "≈ 2009-07-25"` contradicts the RPC-verified 20188 by ~17k blocks (block 3300 ≈ late Jan 2009); `block_67700 "≈ Pizza Day"` — the pizza transaction is block 57043; `block_32000 "≈ first dollar trade"` (Oct 2009 ≈ block ~25k; 32000 ≈ Dec 2009). Only `block_107165` (dollar parity, matches CSV exactly) and `block_70000` (Mt. Gox, CSV 68779) are close. User-facing dropdown mislabels; CTA-only, doesn't affect canonical models.
- **CONFIRMED harmless** t-floors at block scale (t ≥ 48,591): `max(t,0.5)` (`_base.py:70,149,245`, `_helpers.py:202`), `np.maximum(t,0.1)` (`_lppl.py` ×5, `_hybppl_eppl.py` ×5, `_simple.py:392`). No-ops for in-range block-t. **Caveat**: if C2's wrong-unit t (years ~16) leaks through, these floors *mask* the bug into plausible-looking output instead of crashing — consider a debug assert `t >= T_MIN/2` in `price_at` during Phase 2c.
- σ(t)=σ₀·t^(−α) (`_helpers.py:47-89`): log-log regression is unit-covariant; refit per axis is sound; evaluation fine *given* consistent t. Only breaks via C2.
- `today_year()` (`_helpers.py:174-175`) uses 365.25 on day-of-year — pre-existing calendar quirk, axis-irrelevant.

---

## Scope recommendation: calendar-display vs. block-native display

**Keep the calendar-display invariant (finish the design as spec'd), with one cheap addition.** Reasons:

1. **The invariant is not why the design has bugs.** C1/C2 are conversion-layer defects; a block-valued x-axis wouldn't fix them (Citadel/tax/RMD still needs calendar, so the conversion layer must exist regardless — spec §3.7's tax-law argument is genuinely load-bearing and correct).
2. **Cost of breaking it is large and enumerable**: `_year_ticks` (`figures/common.py:456`), `_add_date_hover`'s 0.3–120 heuristic (`:789` — silently drops all hover at x~10⁵–10⁶), `bub-xrange` slider + snapshot year-field semantics (`snapshot.py:20` — q4 link payloads would change meaning → registry churn + link breakage), scanner clientside JS (`scanner.py:284-295`), user-model click mapping (`user_model.py:39-41`), every E2E restore test, plus x arrays in all figure builders and MC overlays. That's a second project of comparable size to Phase 2c, with user-visible regression risk and worse legibility (nobody thinks in block heights for retirement dates).
3. **The real benefit the user wants — models honest to chain time — is fully delivered by block training + a correct piecewise conversion map.** "Convert the entire codebase to block height" is satisfied where it statistically matters (the fit axis); display units are presentation.
4. **Cheap compromise**: a secondary top x-axis (or hover line) showing block height via the real map — cosmetic, no snapshot/slider semantic change, satisfies "block-native" visibility. Do this after 2e if desired.

**And: resurrect the A/B comparison report as a gate before 2e** (see I6). It is the only committed evidence that the flip improves anything, and it doubles as the acceptance test for 2b.ii's refits.

---

## Verified-OK (checked; don't chase)

- **Cache fingerprint includes axis**: `cache.py:19,36,107,119,134` — figure, citadel, and L0 keys all carry `TIME_BASIS`. Matches spec §3.3.
- **Snapshot fp Phase-1 slot**: `snapshot_defaults.py:396` hashes `TIME_BASIS` first. Done as spec'd (enforcement is Phase 2c work, see I4).
- **Block training data path is sound**: `data.py:70-91` — inner join on date, real heights, `T_MIN = T_PER_YEAR` preserves "skip first year" semantics; price/block CSVs row-aligned (5802/5802) with a daily-update invariant guard.
- **`model_data_block.pkl` schema**: carries `time_basis/t_label/t_per_year/t_origin` (= block/blocks/52596/20188), real block offsets in `price_years`, **no** `years` alias — exactly per spec §3.5. Sidecar meta JSON matches.
- **`build_bm_model.py`** `--time-basis` flag with pre-argparse `QS_TIME_BASIS` env setup and `_block` filename suffixing (`:19-37,233-250`) — 2b.i delivered as planned.
- **QR cache keying** (§3.4.1): block pkl `qr_fits` keyed identically (float quantiles) — axis-invariant as claimed.
- **CTA isolation**: `time_basis.py:10-11` docstring + `custom_fit.py` — no `TIME_BASIS` import; the CTA block prototype (row-aligned `_BLOCKS`, read-only, `/health` reporting, r-range normalization for block-scale t) is solid and is the best in-repo prior art for C1's piecewise map.
- **T_MIN sweep in `btc_core`**: `_simple/_lppl/_hybppl_eppl/_basis/_helpers` all use `from time_basis import T_MIN` with `price_years >= T_MIN` masks (the one intentional exception, `_simple.py:572`'s 0.5, is flagged in I5).
- **bitcoind daily dependency**: already handled on this branch — `daily_update.sh` aborts before deploy on block-map append failure; not a new deploy risk introduced by 2e (the *rebuild* gap, I2, is the real ops risk).
- **Markov `.so` concern from spec §6**: transition matrix is built at cache-generation time against the active model (`mc_cache.py:120`); no compiled-artifact rebuild needed. The conversion call sites are covered by C2.
