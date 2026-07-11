# Time-Basis (Block-Height) Conversion — Status & Handoff

**Date**: 2026-07-10
**Status**: **PAUSED** — to be folded into a larger codebase overhaul (user decision, 2026-07-10).
**Do NOT** start implementation from this doc alone; it is a resumption brief, not a plan. The design is **NOT ship-ready** (4 Criticals below).

## Companion docs (read these)
- **Design spec** (2026-04-26): `docs/superpowers/specs/2026-04-26-time-basis-toggle-design.md`
- **Phase-1 decisions log** (D1–D12, incl. D11 "blocks win" pivot): `docs/superpowers/plans/2026-04-26-decisions-log.md`
- **R0 adversarial review** (2026-07-10, full detail on C1–C4 + I1–I7): `docs/superpowers/reviews/2026-07-10-time-basis-spec-r0-adversarial-review.md`
- Phase plans: `docs/superpowers/plans/2026-04-26-time-basis-toggle-phase1.md`, `2026-04-27-time-basis-toggle-phase2a.md`, `2026-04-27-time-basis-toggle-phase2b-i.md`

---

## Where the effort stands

Architecture: **block-TRAINED models, calendar DISPLAY** (charts/sliders/pickers/sims stay calendar; only the axis models are *fit* on changes). Admin build-time toggle via `quantoshi.toml` (`time_basis`), plus `QS_TIME_BASIS` env override for build tooling. No user-facing UI toggle.

| Phase | Status | Summary |
|---|---|---|
| 1 — Plumbing | ✅ done, deployed | `quantoshi.toml` (`block_origin=20188`, `blocks_per_year=52596`), `btc_web/time_basis.py`, cache + snapshot fingerprints salted with `TIME_BASIS`, pkl schema widened + `_meta.json` sidecars |
| 2a — Parameterize build | ✅ done | `T_MIN` sweep across `btc_core` (`>=1.0` → `>=T_MIN`), `--time-basis` on build scripts, calendar rebuild byte-identical (D12) |
| 2b.i — Block pkl scaffold | ✅ done (on `time-basis-toggle-phase2b`) | `model_data_block.pkl` + meta exist; BM/QR/sigma/OLS fit cleanly on block-`t`; **LPPL/HybPPL/EPPL/PCA/Greedy predictions in it are garbage** (calendar-fit class attrs at block magnitudes) |
| 2b.ii — Family refits | ❌ not started | Refit calendar-oscillator families on block-`t` — blocked by C3/C4 below |
| 2c — Runtime axis loader | ❌ not started | Load block pkl at runtime; enforce cross-axis snapshot rejection — blocked by C1/C2 below |
| 2d — Heavy caches | ❌ **OUT OF SCOPE** (MC + Citadel) | Per 2026-07-10 scope cut |
| 2e — Flip default + deploy | ❌ not started / deferred | Requires the out-of-scope surfaces handled + A/B report gate |

Terminology note: on the log-log fit `log10(price)=A+m·log10(t)`, the origin anchor is **t=1**, never t=0 (`log10(0)=-∞`). Holds in block space too (origin block → t=1).

---

## Decisions made this session (2026-07-10)

1. **Scope = calendar display** (models block-trained, charts stay calendar). Reviewer-confirmed this fully delivers "models honest to chain time"; tax/inflation/RMD calendar-coupling is load-bearing.
2. **Out of scope for this effort (fold into a later, larger cycle): Markov/MC, the Citadel tax system, and the entire Citadel tab.** These are the most entangled surfaces (C2's year-unit `citadel_step`, `mc_overlay` sprawl, Phase-2d 6-hour caches).
3. **Branch base**: cut the new dedicated feature branch **fresh from `time-basis-toggle` HEAD** and **cherry-pick only the 2b.i code commits** (`771aaa6`, `ed593a0`, `fac6b0f`, `ba8a56f`, `8b53e0b`, `21b2423` + plan docs); rebuild `model_data_block.pkl` fresh. Do **not** merge `time-basis-toggle-phase2b` wholesale — it forked 5 weeks ago with divergent `daily_update.sh` + binary CSV/pkl churn (I3).
4. **Cycle boundary**: **do not flip prod to block-default in this effort.** The switch will ride in with the larger overhaul. (Paused before any implementation.)

---

## The 4 Criticals that MUST be fixed before any block-default flip
(Full detail + file:line + failure scenarios in the R0 review doc.)

- **C1 — Conversion inconsistency.** Runtime `time_basis.calendar_to_t` projects blocks at a uniform 144/day, but models are trained on *real observed* block heights. Same historical date → different `t` at train vs. display → model price **~23% low today, forever**; non-monotonic (~±5% over history). Spec's "1.5-week drift" claim is wrong ~30×. **Fix**: piecewise map in `time_basis.py` — real CSV lookup for past dates, protocol rate anchored at the *last real block* for future. Prior art: `custom_fit._BLOCKS` (row-aligned, health-checked). CSV starts 2010-07-17 (block 68779); handle the 20188→68779 gap without crashing.
- **C2 — Missing boundary conversion.** Spec §3.7 claims the sim/figure boundary converts via `time_basis.calendar_to_t()`; it doesn't. ~50 call sites in 14+ files still use calendar-only `btc_core.yr_to_t`/`today_t` → naive flip gives `$0`. In-scope surfaces to convert: `figures/common.py` (`_build_time_array`, `_year_ticks`), `figures/bubble.py`, `figures/heatmap.py`, `figures/residuals.py`, `callbacks/charts`, `callbacks/scanner.py` (+ its clientside JS), `callbacks/ticker.py`, `utils.py`, `callbacks/user_model.py`. (Out-of-scope: `citadel_*`, `mc_*`, `markov`.) Must ride on C1's map. Contract to declare: **chart x-values ARE calendar-years; block-`t` only inside model evaluation.**
- **C3 — Param storage.** PPL params are single-copy class attributes regex-rewritten in `btc_core` source, not per-artifact → the "parallel pkl" scheme can't hold them; D11's "reversible at TOML level" is false; and the **monthly `refit_all_ppl` timer silently reverts** a block site to calendar params ($10²⁹). **Fix**: move fitted params into the pkl (cleanest) or axis-suffixed attr blocks; thread `--time-basis` through `refit_all_ppl.py` + every `fit_*.py`; gate the systemd timers until axis-aware.
- **C4 — Fit bounds calendar-hardcoded.** `tools/fit_*.py` bounds (`A∈(−3,1)` vs block intercept ≈ −25.8; `W∈(0.5,10) rad/block` past Nyquist; `mask t>=1.0`) mean DE **can't reach** the block optimum. **Fix**: full per-parameter, per-tool bound-transformation rule.

### Key Importants
- **I6 — A/B evidence gate.** D11 skipped the R²/AIC/OOS-RMSE + cal-osc-amplitude comparison report. **Resurrect it as a hard gate before the eventual flip** (2e). Cheap once both pkls coexist; doubles as the C4 regression check.
- **I7 — halving-prior physics is backwards.** Halvings are *block-native* (exactly 210,000 blocks) → the cal-osc terms should get **sharper** in block mode, not collapse. Center the prior at `2π/210000` and test for *strengthening*. Also: main's halving-lines feature (`908fdb7`) estimates future halvings by calendar cadence — should derive from `210000·k` via the block→date map.
- **I2 — ops staleness**: `update_prices.py`/`daily_update.sh` rebuild only calendar pkls; block pkl already ~2 weeks stale in-tree.
- **I1** cache fingerprint hardcodes `model_data.pkl`; **I4** cross-axis snapshot links have no axis tag to warn on; **I5** residual calendar literals (`_GENESIS_YR=2009.56`, scanner JS, UserModel `>=0.5` mask).

---

## Recon seam inventory (condensed — otherwise lived only in the session transcript)

Categories: **[B]** already block-aware · **[C]** calendar-only · **[H]** hybrid/done.

- **[H]** Training-`t`: `tools/model_toolkit/data.py` — calendar `days/365.25`; block `blockheight−20188` (real heights), col still named `years`.
- **[H]** Runtime helpers: `btc_core/_helpers.py` `yr_to_t`/`today_t` (calendar-only) vs `time_basis.py` axis-aware — **duplicated**; figures/citadel/mc import the calendar-only one (→ C2).
- **[B]** CTA engine `engines/custom_fit.py` + `callbacks/custom_time.py` — real prior art for block mode (block x-axis prototype, row-aligned `_BLOCKS`, corruption guard). Deliberately does NOT import `TIME_BASIS` (per-fit user toggle only).
- **[B]** Block↔date map: `BitcoinBlocksDaily.csv` (`date,blockheight`, from 2010-07-17), built by `tools/build_block_map.py` (bitcoind RPC); equal-length with prices (parity guard); `test_block_map_cli.py`.
- **[C→convert]** Figure x-axes (all in-scope tabs): `figures/common.py`, `bubble.py`, `heatmap.py`, `residuals.py` — plot `x=t` numerically with calendar-year ticks; `_add_date_hover` assumes `t∈[0.3,120]`.
- **[C, inherently]** Heatmap year inputs + CAGR (per-annum); DCA/Retire/SC inflation `(1+infl)^(ts−t_start)` reuses the same `ts` as model-`t` → needs **dual-track** time (t_model for price, years for inflation/labels).
- **[C, out of scope]** Citadel/tax/RMD (`engines/citadel_*`, `tax*`) — stay calendar by design (§3.7).
- **[C, out of scope]** MC/Markov (`mc_cache.py`, `mc_overlay.py`, `markov.py`) — stay calendar; matrix rebuilt against active model.
- **[exempt]** S2F, EF — calendar-native by construction; block site loads calendar EF pkl unchanged.

---

## Recommended shape for the larger-overhaul cycle
1. Amend the spec to fix C1–C4 + I-items (this doc lists them; the review has the detail). Fable-review the amendment to 0C/0I before planning.
2. Branch fresh from main + cherry-pick 2b.i (decision 3 above).
3. Sequence: C3/C4 (param storage + fit bounds) → 2b.ii family refits → C1 map + C2 boundary on in-scope surfaces → 2c runtime loader → **A/B report gate (I6)** → then decide the flip, incl. how the (then-in-scope) Citadel/MC/tax surfaces are handled.
4. Validate block mode in dev via `QS_TIME_BASIS=block` before any prod flip.
