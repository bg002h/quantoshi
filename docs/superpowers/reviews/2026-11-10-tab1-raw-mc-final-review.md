# Tab-1 Raw MC — final whole-branch review

- **Range:** `d1608fb5..91003ad6` (11 commits, 34 files, +4733 / −193)
- **Reviewer:** independent (Opus), read-only on the worktree
- **Date:** 2026-11-10
- **Authorities:** spec rev 6 (`docs/superpowers/specs/2026-11-10-tab1-raw-mc-design.md`); plan (`docs/superpowers/plans/2026-11-10-tab1-raw-mc.md`), including its Global Constraints and Review Focus.

## How this review was done

The review package was read in four passes:
1. `raw_mc.py` in full.
2. The callback branch: `charts/__init__.py`, `_raw.py`, `coerce.py`, `mc_payment.py` and `app.py`.
3. Layout, CSS, JS and `callbacks/mc_controls.py`.
4. Share links: `snapshot.py`, `snapshot_defaults.py`, `routing.py` and the follow-ups.

Test files were read only for the specific questions below. Machine-verified claims were not re-derived (suite, E2E 8/8, sabotage 15/15, registration, fingerprints).

Focused probes run for this review (scripts in the session scratchpad):
- **Raw share-link restore, end to end in Firefox** against the worktree dev server on :8050. The link was a `q4:` v4 link carrying `raw_cycle`, `exp`, bins 8, regime `[0,1,2]`, paths and extremes on, lines 7 and sims 300. The panel came up in `qs-mc-raw-mode` and the 8 regime boxes were restored as `[T,T,T,F,F,F,F,F]`. The legend read "MC … · raw cycle, exponential decay, 2010–2026", with "MC paths", "Highest end" and "Lowest end". The status line had all its parts, including "steered away … (0.1 % of months still fall in them)". `restore_builder.py:99` sends MC-enabled links to the `update_bubble` cascade, so Raw links render correctly. No task owned this journey; it works.
- **Per-request memory and CPU** of `simulate` at 3200 sims × 40 years, with real prices and today's date:

| Run | Peak transient memory | Kept in `RawResult` | CPU, quiet host (load 0.9) |
|---|---|---|---|
| iid, unfiltered | 62 MB | 12 MB | ≈ 30 ms |
| iid, filtered | 74 MB | 25 MB | 79–122 ms |
| cycle, unfiltered | 62 MB | 12 MB | ≈ 31 ms |
| cycle, filtered | 86 MB | 37 MB | 113–126 ms |

  The memory figures were measured with tracemalloc. The CPU figures used a separate run without it.
- **Python 3.12 grammar:** `ast.parse(feature_version=(3, 12))` was run over all 28 changed `.py` files. All are clean.
- **Commit hygiene:** all 11 commits carry both trailers. No `*.pkl`, `.playwright-mcp/` or `.superpowers/sdd` files are in the diff.

## Strengths

- **The Raw/model boundary is clean.** `raw_mode` is decided above `_mc_setup` (`charts/__init__.py:381`), so Raw never reaches `_mc_payment_check`, `is_free_tier` or `mc_cache`. With MC off, `bool([])` short-circuits `_mc_setup` before any payment call. The Raw return tuple leaves every MC store as `no_update`. `_mc_payment_initiate` stops a Raw Run click before the quant modal or the invoice. The match JS goes neutral, and the cost panel is clientside and CSS-hidden.
- **Model mode is effectively unchanged.** The only model-path changes are the `raw_mode` if/else, the 9th tuple element `""`, and `qs-mc-model-only` wrapper Divs that have no effect on display outside Raw.
- **Determinism is sound.** blake2b over canonical JSON; `today` is normalised; `effective_checked` makes click order and no-op selections seed-neutral; `p0` is excluded from the seed.
- **The single price registry.** `set_prices`, `months_for` and `_CACHE_CLEARERS` give simulate, the labels and the layout one shared series. The layout also works under gunicorn `--preload`: registration happens in the master, and the workers inherit it copy-on-write.
- **The steering implementation is careful.** It batches the 19 redraws per violating row, keeps "first accepted, else closest", and draws candidates for the same grid slot at the same offset, so a redraw never straddles a historical halving.
- **Share links are well specified.** The fields are appended at indices 340–347, and the 10-wide bitmask for `bub-mc-raw-regime` avoids the F-24 trap. The static `[2010, 2026]` window default is faithful: a link made in a later year encodes its own window explicitly.
- **The status line covers every item the spec lists:** trend and fit numbers, fit set (window / all history), current-cycle pace against the curve, both caveats, widened-tolerance counts, the measured unchecked share, the sims < 100 warning, and every refusal reason.

## Issues

### Critical
None.

### Important

**I-1. Raw depends on pandas ≥ 2.2. Requirements are unpinned and prod's pandas version is unverified. On an older pandas every page load fails.**
- **Where:** `btc_web/raw_mc.py:95` (`s.resample("ME")`); `btc_web/layout/mc_controls.py:92` (`except RuntimeError` only); `btc_web/requirements.txt:6` (`pandas`, no version).
- **What:** the `"ME"` alias appeared in pandas 2.2 (January 2024). Older pandas raises `ValueError: Invalid frequency: ME`. This is the first use of `"ME"` anywhere in the codebase. Dev runs pandas 3.0.1, so the suite cannot catch it.
- **Failure path:** `_raw_controls_block` calls `raw_regime_options(5, …)` while building the layout. That calls `months_for`, then `load_months`, which raises `ValueError`. The `except RuntimeError` does not catch it, so the Tab-1 layout and with it the whole page fail to build. That is an outage of every page, not only Raw.
- **Why it blocks:** the code assumes something about prod that nobody has checked, and if the assumption is wrong the failure is site-wide. Prod runs Python 3.12.3 in a separately installed venv.
- **Fix (both parts are cheap):**
  1. Confirm the prod version before deploying (`ssh … "/opt/quantoshi/btc_venv/bin/python3 -c 'import pandas; print(pandas.__version__)'"`) and pin `pandas>=2.2` in `btc_web/requirements.txt`.
  2. Make the layout path fail soft: in `raw_regime_options`, catch `Exception`, log it, and fall back to the plain "Bin n" labels. Then a Raw-only fault can never take down the layout. The same reasoning applies to `add_raw_overlay` (see M-7).

### Minor

**M-1. The spec does not yet describe the shipped steering, the near-n line fallback or the I-1 comment.** Files: `docs/superpowers/specs/…-design.md:212` and the Display section; `btc_web/raw_mc.py:707-712`.
- Spec l.212 still says "a block's three months come from one complete cycle". The l.288 rank rule has no clause for `n_lines` close to `sims`.
- The in-code comment cites `task-3-report.md`, which is gitignored and will not exist in the repo.
- **Fix:** amend the spec (text in the ledger triage below). Point the comment and the `simulate` docstring at the spec, not at the report.

**M-2. Production computes and keeps test-only diagnostics (ALARA).** File: `btc_web/raw_mc.py:649, 695, 723-735`.
- Filtered runs build `states` (sims × 481 float64, plus a transposed copy). Filtered cycle runs also build `sources` (int64, plus a transposed copy).
- Measured: 37 MB kept in `RawResult` and 86 MB peak per 3200-sim filtered cycle request. Five workers serving concurrently would peak at about 430 MB.
- Production reads only `ok`, `status`, `paths` and `t_axis`.
- **Fix:** add `RawRequest.diagnostics: bool = False` (not part of the seed). When it is false, return `states=None` and `sources=None` and skip the transposed copies. The steering still needs the per-step row of `states`, but not the full history unless diagnostics are requested. Tests pass `True`. This saves about 25–37 MB of peak per filtered request.

**M-3. Raw re-simulates on every `update_bubble` trigger (ALARA).** File: `btc_web/callbacks/charts/_raw.py:103`.
- Triggers include point size, legend, palette and x-range pans.
- Cost is about 30 ms of CPU unfiltered and up to about 126 ms filtered (quiet host).
- The recommendation is in the ledger triage under "Task 5 ALARA".

**M-4. Raising bins leaves the regime value at `[0..4]`, which silently filters the new top bins.**
- Files: `btc_web/callbacks/mc_controls.py:153-165`; `btc_web/raw_mc.py:330`.
- Model mode does the same by design.
- See the ledger triage.

**M-5. "Current cycle: no complete months yet" is wrong when the window ends before the current cycle.** File: `btc_web/raw_mc.py:742, 756`.
- Example: window 2013–2023 gives `cur = win[win.cycle == cyc_now]`, which is empty, so the status prints that line although 31 months exist outside the window.
- **Fix:** distinguish the two cases, e.g. "Current cycle is outside the history window".

**M-6. Some labels and controls make less sense in Raw mode.** File: `btc_web/layout/mc_controls.py:405`.
- The dropdown label "Model source (quantile bands)" stays as is in Raw.
- "Lines shown" and "Show extremes" are visible and active-looking while "Show individual price paths" is off, but they do nothing until it is ticked.
- **Fix:** add a Raw-only alternate label ("Source"), and hide `_raw_lines_block` unless paths are on (clientside, one Input).

**M-7. Nothing catches an exception from `simulate` in the callback.** File: `btc_web/callbacks/charts/_raw.py:103`.
- Any internal assert (`raw_mc.py:428, 686`) or `ValueError` becomes an `update_bubble` 500, and the whole Tab-1 chart stops updating, not only the Raw overlay.
- **Fix:** wrap the call in `try/except Exception`, log it, and return the fast-path figure with the status "Raw simulation failed".

**M-8. The `_add_mc_extremes` docstring overclaims.** File: `btc_web/figures/bubble.py:232`.
- It says "never double an example line". That is false when lines ≥ sims, and for the near-n fallback.
- This is the Task 4 ledger item; see the triage.

**M-9. Docs drift.**
- `CLAUDE.md` does not mention Raw in the Tab-1 row, the MC section or the test-file list. It also does not cover `raw_mc.py` or `callbacks/charts/_raw.py` in the file table.
- MEMORY's "currently `bub-mc-paths`, index 339" becomes stale at merge.

**M-10. Typing a sims value above 3200 is coerced to 1000, not clamped to 3200.** File: `btc_web/layout/mc_controls.py` (`bub-mc-raw-sims`, `max=3200`).
- The HTML5 `max` makes the browser send `null`, so `_ci_or(None)` gives 1000. Review Focus 5's "5000 → 3200" holds only on the server path.
- This is the documented CLAUDE.md gotcha, and the label shows the range.
- **Fix (optional):** drop `max` and let the server clamp.

## Ledger triage

| Ledger item | Ruling | Reason |
|---|---|---|
| T1: `standardise` 11/12-month boundary untested | FINE AS IS | A single `>=` on a constant, and the thin-cycle test covers the borrowed branch. |
| T1: CSV loader duplicated (tool vs `raw_mc_testing`) | FINE AS IS | Dev and test only, never on the prod path. |
| T1: `test_cycle_table_annual_units` pins 2-dp figures | FINE AS IS | Frozen `asof` makes them stable, and they are the spec's measured table. |
| T1: `test_cli_matches_module` name overclaims | FINE AS IS | Nit. |
| T1: `months_for` returns a shared mutable frame | FINE AS IS | I checked every caller: none mutates it (`win` is a slice; `standardise` builds new series). |
| T1: `fit_trend` catches `(RuntimeError, ValueError)` | FINE AS IS | Both mean "could not fit", which the spec maps to "no fan". |
| T1: 2 unreachable Nits | FINE AS IS | Unreachable. |
| T2: `today` not normalised | FINE AS IS (closed) | Done in T3 (`seed_for` and `simulate` normalise). |
| T2: φ ±0.04/±0.08 widening and short blocks not pinned | FIX SOON | Synthetic-input test; the branches are live only on narrow windows (the probe showed 0 of 160 widened at default). Owner: next Raw touch. |
| T2: other minors/nits | FINE AS IS | Nothing material. |
| **T3: Ruling I-1 fallback (per-run integrity)** | **FINE AS IS for the code; MUST FIX the spec before merge** | Sound, for four reasons. (1) Each run copies consecutive months of a candidate valid for that slot's length, so it stays inside one complete cycle and never straddles a historical halving: the stated purpose of l.212. (2) Phase matching is kept, because redraws come from the same slot's candidate set at the same offset. (3) `z` is standardised per cycle, so mixing runs from two cycles is the same exchangeability iid already relies on. (4) The splice only loses 3-month momentum on months the filter is already overriding. The same-cycle alternative failed the binding 5 % bound in 21/45 cases (max 9.32 %). The spec amendment is below. |
| T3 follow-up: spec l.212 amendment | **MUST FIX before merge** | The spec is the binding authority and currently contradicts shipped behaviour; it is a docs-only commit. |
| T3: perf test skips (2,) and (1,3) | FINE AS IS | Measured here: (1,3) takes 90–126 ms CPU, below the 0.6 s bound. |
| T3: `RawResult.sources` ~12 MB, must not be cached | FINE AS IS (verified) / FIX SOON | `_raw.py:105` `del res`; nothing caches it. It is still computed in production: M-2, owner Raw follow-up. |
| T3: docstring omits splice; comment cites gitignored report | **MUST FIX before merge** (same commit as the amendment) | Dangling reference; one-line edit. |
| T3: 0.2 s filtered target unverified on a quiet host | FINE AS IS (closed on dev) | Quiet host (load 0.9): filtered ≤ 126 ms CPU, unfiltered ≈ 30 ms. Prod CPU is unmeasured; check the `[trace] bubble-fig BUILT (raw)` timings after deploy. |
| T4: `_add_mc_extremes` docstring "never double" false | FIX SOON | Trivial wording fix (M-8). Can go in the same doc commit. |
| T4: near-n fallback biased low when n−k is odd | FINE AS IS | One rank. |
| T4: near-n fallback not in spec | **MUST FIX before merge** (same amendment commit) | One sentence in Display. |
| T4: tie-stability test weak; no `MC_EXTREME_ALPHA > MC_PATH_ALPHA` test | FINE AS IS | Constants sit side by side in `colors.py`. |
| **T5: bins-raise silently filters new top bins** | **FIX SOON** (owner: one fix for model and Raw) | Visible (5 of 8 boxes ticked, and the status says "steered away"), and model mode behaves the same by design, so it does not block. It still surprises a reasonable user. **Fix:** a clientside callback on bins, gated on a fresh user click stamp on the bins dropdown (the user-click-only pattern, so share-link restore is untouched). If the previous value covered every previous bin, write `range(newBins)`. Write nothing otherwise. |
| **T5: ALARA re-simulate on every trigger** | **FIX SOON** (operator's call; not blocking) | See the recommendation after this table. |
| T5: bool price edge | FINE AS IS | `btc-price-store` is written only by the server ticker; a hostile client gains nothing. |
| T5: mid-module import in `app.py` | FINE AS IS | Needed for ordering; commented. |
| T5: no standalone model-mode-unchanged test | FINE AS IS | The model-path diff is one if/else plus a `""` tuple element; the existing `TestUpdateBubbleCallbackMC` covers it. |
| Ruling: bins-raise not T6 scope | FINE AS IS | Correct; tracked above. |
| **T6: count tracker (plan-mandated), bins 8→5 then a click re-ticks paths** | **FINE AS IS** | Defensible: the bins change really did turn the filter off, so the next click really is a no-filter → filter transition. The result is visible and the user can untick it. |
| T6: payment guard value-based, not `tab=="bub"` | FINE AS IS | Fail-safe: Raw values exist only on Tab 1. |
| T6: guard-order test uses `inspect.getsource` | FINE AS IS | Brittle but red-on-change. |
| T6: 4 Nits; autowiden uses thisYear+10, not raw years | FINE AS IS | Matches the spec ("current year as the MC start"). |
| T7: pre-existing 5-wide regime bitmasks, Tabs 2–6 (F-24) | Out of scope (confirmed) | The diff touches no Tab 2–6 callback or snapshot table. F-24 is filed with an owning phase. |
| Ruling: 10-wide bitmask for raw-regime | FINE AS IS | Lossless and append-only. |
| T7: raw-window static `[2010, 2026]` default drifts in 2027 | FINE AS IS | Faithful: later-year links encode their window explicitly. Only pre-Raw links get `[2010, 2026]`. |
| T7: registry at 20 cap (next change evicts `faf00e93`) | FINE AS IS | Pre-existing policy. Note it in the next defaults change. |
| T7: q3 padding test all-None; tests import `_find` across modules | FINE AS IS | Test style only. |
| T7: at merge, update MEMORY tail note to `bub-mc-extremes`, 347 | **MUST at merge** (housekeeping) | A stale index note invites a wrong append. Tests would catch it, but the note is what is read first. |
| T8: legend checked by name, not `showlegend` | FINE AS IS | `_add_mc_fan` and `_add_mc_extremes` set `showlegend` explicitly today. |
| T8: CSS-hide mutation evidence incidental | FINE AS IS (closed in r1) | The r1 fold added RED mutation evidence (M-2); the re-review is clean. |
| T8: hard-coded 2035 guard | FIX SOON | Fails loudly in 2035 (not vacuous). Derive the year from `this_year`. |
| T8: `_choose_src` retry diagnosis unproven; Escape to focused element | FINE AS IS | E2E passed 2× in a row. |

### Spec amendment (replaces spec l.212–214; add the line rule to Display)

> - **Block integrity:** an unsteered block copies `L` consecutive months
>   (`L` = its grid length) of one complete cycle and never straddles a
>   halving in the history it copies (review r4 M). With the regime filter
>   on, a block is accepted month by month; a redraw from violating month `j`
>   replaces months `j … L−1` with months `j … L−1` of another candidate for
>   the **same grid slot** (same phase window, same offset). A steered block
>   is therefore a splice of up to `L` contiguous runs. Each run copies
>   consecutive months of one complete cycle and never straddles a
>   historical halving, but two runs may come from different cycles.
>   Restricting the redraw to the prefix's cycle was measured to break the
>   5 % steering bound (21 of 45 cycle-mode cases, max 9.32 %), so per-run
>   integrity is the binding rule.

Display, after "If lines ≥ sims, all are drawn":

> When the 2nd–98th-percentile rank window holds fewer distinct ranks than
> the lines requested, the lines are that many consecutive ranks centred on
> the median, so no path is drawn twice.

### Task 5 ALARA: recommendation

Add a **per-worker paths-only LRU of size 2** in `callbacks/charts/_raw.py`.

- **Key:** `(seed_for(req), req.p0)`. Including `p0` keeps filtered runs exact, because `D_0` depends on `p0`.
- **Value:** `(t_axis, paths.astype(np.float32), tuple(status))`. Store no `RawResult`.
- **Where it sits:** check the cache before `simulate`, then trim, scale and draw from the cached arrays as now.

RAM cost:
- Default request (1000 sims × 481 steps × 4 B): ≈ 1.9 MB per entry, so ≈ 3.9 MB per worker and ≈ 19 MB across 5 workers.
- Worst case (3200 sims): ≈ 6.2 MB per entry, so 12.3 MB per worker and ≈ 62 MB across 5 workers.

Float32 is ample, because lines are rounded to 2 significant figures and the fan percentiles survive 7 digits.

Benefit: cosmetic triggers (point size, legend, palette, pans, and toggling lines or extremes, which are seed-neutral by design) cost about 1 ms instead of 30–126 ms. Requests round-robin across workers, so each worker misses once per new seed.

If the operator prefers zero added RAM, the alternative is M-2 alone: it cuts transient memory by 25–37 MB per filtered request and leaves CPU as it is. Recommend doing M-2 in any case, and the LRU only if Raw sees real traffic.

## Declined to judge

- **Prod CPU timing:** prod hardware is not measurable from here (dev quiet-host numbers are above).
- **Prod pandas/numpy/scipy versions:** reading them needs prod access, which is outside this review's read-only scope. This is why I-1 is a verify-or-fix.
- **Statistical soundness of the spec's model choices** (power-law growth, per-cycle σ, the ±0.04 phase tolerance): these are operator-approved spec decisions, and the tests pin the spec's bounds.
- **Visual quality of the fan and lines:** the E2E checks structure, not appearance. Judging appearance needs the operator's eyes (deuteranomaly-aware); no colour is distinguished by hue.
- **Mobile layout of the longer Raw panel:** not exercised; no task owned it.
- **Behaviour against a stale `/_dash-dependencies` client:** accepted in plan Review Focus 4. No test is possible without two deploy states.
- **Time Machine plus Raw** (the fan starts today while the chart is as-of a past date): the same as model-mode MC today, and the spec says nothing about it.

## Assessment

**Ready to merge? With fixes.**

**Reasoning:** There are no Criticals.

The one Important (I-1) is an unverified prod assumption whose failure would be site-wide. Raw needs pandas ≥ 2.2, and the Tab-1 layout calls into it while catching only `RuntimeError`. Pinning `pandas>=2.2`, confirming the prod version and making `raw_regime_options` fail soft closes it in minutes.

The other must-dos before merge are documentation only:
- the spec amendment for per-run block integrity and the near-n line rule (text above);
- re-pointing the comment that cites the gitignored task report;
- at merge, the MEMORY tail-index update.

The cross-task interfaces line up: `simulate` → `RawResult` → `add_raw_overlay` → figure helpers, the status contents, the snapshot tail order, and the Input order matching the snapshot order. The global constraints hold across the whole diff, and the user journey works end to end, including Raw share-link restore, which I verified in Firefox. The ALARA items (M-2, M-3) and the bins-raise surprise (M-4) are real but neither wrong nor unsafe, and can follow as FIX SOON.
