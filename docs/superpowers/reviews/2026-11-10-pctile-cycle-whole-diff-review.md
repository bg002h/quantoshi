# Whole-diff review: Percentile cycle fit (pre-deploy)

**Scope:** `git diff fe96f4e7 HEAD -- btc_web/` plus the spec `docs/superpowers/specs/2026-11-10-percentile-cycle-fit-design.md`.
**HEAD reviewed:** `5b096849`. HEAD moved while this review ran, from `afaffa98` (the r2 fold) through `3abe3688`, `a35ffa00` and `5b096849`. The Dash-review fold (`a35ffa00`) is therefore in scope; its own findings are not re-raised.
**Reviewer:** independent, read-only except for this file. Scratch scripts are in `/tmp/claude-1000/-scratch-code-bitcoinprojections/6c13708a-e257-46cf-bdef-a2a8a773488f/scratchpad/wd/`.

**Counts: 0 Critical / 1 Important / 6 Minor / 4 Nit**

**Verdict:** Not ready to deploy as is, because of one Important. On the default view, hovering the wave or a fitted turn in 2010–11 shows sub-dollar prices with an SI "milli" suffix: $0.36 reads as "$362m". The fix is a few lines. Everything else the question asks about holds:
- No crash on any supported input.
- The Percentile view is byte-identical to the base when the fit is off.
- Share links behave as specified.
- Worker cost is about 1 s per cold key.
- All four r2 Importants are resolved in the code.

## Verified independently (beyond the controller's suite and sabotage runs)

| Check | Result |
|---|---|
| Sweep (`sweep.py`): 10 picker models × 2 σ × 3 elements × 2 censor = 120 `_cycle_result` keys. Each was built at lin/log × horizon 10/20/30, with active models qr+pl+bub+picked. | **0 exceptions, 0 NaN/inf** in any trace `y`/customdata, annotation or info line, and every error bar ≥ 0. Cold time per key was 0.66–2.44 s, with 10 sweeps running at once on a loaded box. |
| Fit-off figure vs the `fe96f4e7` builder (`legacy.py`). The base `percentile.py` was exec'd in-process and compared as `to_plotly_json()` over 8 param sets × 2 fit-off variants: log/lin, past zoom, constant σ, legend pos, no models, minor grid, u1 without a store. | **16/16 byte-identical** |
| Weak / forecast matrix, resqr `cal` | QR 0.557 (honest, 52/97 windows, 21.5 vs 23.2), PL 0.454, SatPL 0.459 and LPPL 0.509 forecast. BM, EPPL, PCA, Greedy and EF are weak by R². HybPPL `cal` is weak at the search limit. HybPPL `calw` has R² 0.205 and forecasts (N-3). |
| Firefox against a DEV server on :8061 at `5b096849` (`e2e.py`, `e2e_restore.py`) | `/1.4` shows the controls and the summary. Picking PL gives "Not checked out-of-sample". Unticking the fit hides the summary. The Price pill hides the controls. An old `0eac47fc` Percentile link restores with the fit off and only the BM trace. Four new-fingerprint restores were tried: fit off with PL/log/30, fit off alone, fit on with PL/log/30, and fit+censor. Each renders the right traces. |
| HTTP 500s seen during the live run | Only `citadel-graph` (`MissingLongCallbackManagerError`, because diskcache is not installed locally). Pre-existing and unrelated. |
| Python 3.12.13 (`uv run --python 3.12`) | All 9 changed runtime modules compile. |
| Registry diff | `faf00e93` evicted and `757574d3` added. The other 19 entries are unchanged, and no older entry carries `bub-pctile-fit`. `faf00e93` (2026-04-24) is the only fingerprint ever evicted, and it predates the Percentile pill (`a8a21f76`, 2026-08-22). |
| Likely band smoothness, forecast region (`steps.py`) | Maximum weekly step: QR 2.8, PL 3.9, SatPL 3.1, LPPL 3.4, HybPPL `calw` 5.0 pp. The fit band's maximum is 1.2–1.7 pp. |
| Cold-fit memory (`mem.py`, tracemalloc) | Peak 76 MB transient, from the central fit's grid scan on the daily series. Retained 0.2–0.9 MB per key, and `_qr_asof_weekly` holds 97 entries. |

## r2 Important → resolved in code?

| r2 | Resolved? | Evidence |
|---|---|---|
| **I-1**: a noise-level hindcast margin decides whether the default forecast is drawn | **yes** | `pctile_cycle.is_weak` is `r2 < WEAK_R2 or at_limit` only. `hindcast(every=1/12)` gives 97 monthly cutoffs (sweep: `n=97` everywhere). `test_default_qr_stays_a_forecast_as_data_moves` pins trims of 0–180 days. The margin still moves (−0.13 → +1.68 pp over the last 39 weeks, `margin.py`), but it now toggles only the informational ⚠ box (N-3), never the forecast. |
| **I-2**: optimistic hindcasts reported as win rates beside QR's honest one | **yes** (wording residue in M-3) | Non-QR summaries carry no "x of n" (live LPPL and PL text, and `test_non_qr_is_not_checked_out_of_sample_and_shows_no_win_share`). The forecast legend reads "‹model› cycle extrapolation (not checked out-of-sample)" (live restore: `'PL cycle extrapolation (not checked out-of-sample)'`). The spec bullet is corrected (spec lines 62–65). |
| **I-3**: censor leaks hindsight into the honest hindcast | **yes** | `_cycle_result` (`figures/percentile.py:220-231`) applies `fit_mask` to the central fit and the bootstrap. `hindcast(fit_mask=…)` masks `m_in` only, while `m_out` scores every day. `cover()` and `clamped` use the unmasked `tw/yw/pct`. Sweep: with censor on, QR keeps `n=97` and full-series coverage. Tests: `test_censor_mask_fits_without_but_scores_with`, `test_censor_changes_fit_and_is_cached`. |
| **I-4**: ragged likely band (12–15 pp steps) | **yes** | `likely_band(paths=8)` uses `_block_path`, which draws a random phase in [0, 52) under its own seed. The maximum forecast-region weekly step is now 2.5–3.9 pp (see above), and QR coverage is 83 %. `test_likely_band_is_smooth_week_to_week`. |

---

## Important

### I-1: The hover renders sub-dollar fan prices with the SI "milli" suffix ($0.36 → "$362m")

**Evidence**
- `figures/percentile.py:394`, `:397`, `:437` and `:459` format every hover price as `$%{customdata[n]:.3s}`. That is d3 SI notation, so a value below 1 gets the milli prefix `m`.
- d3-format run in node (`d3-format` 3.1.2, the same `s` semantics Plotly uses):
  ```
  0.1943 -> $194m    0.7871 -> $787m    0.0502 -> $50.2m    2.534 -> $2.53    83030 -> $83.0k
  ```
- The live app in Firefox (`e2e.py`, `Plotly.Fx.hover` on the default `/1.4` view):
  ```
  QR cycle fit (3.57 yr) Aug 29, 2010  fit Q78 ≈ $362m  wave Q69–Q85 (fit uncertainty)  likely Q52–Q100 ≈ $86.7m–$3.70
  ```
- Extent (`si.py`), default view: 49 of 852 in-sample wave points plus the first fitted-peak marker (Oct 2010, $0.79 → "$787m") show this. It holds for every forecasting model (PL 43, LPPL 62, SatPL 44). On the default log x-axis, mid-2010 to Feb 2011 covers **13.9 % of the plot width**.
- This is new code. No other `.3s` exists in `btc_web/`, and the rest of the app uses `$%{y:,.0f}`.

**Why it matters.** On the default view the hover reports a price nine orders of magnitude too high. "$362m" reads as $362 million. Correct numbers in the wrong unit are still a misleading statistical display.

**Fix.** Format prices server-side with the existing `_money()`, which already handles values below $1 and k/M/B/T, and put the strings in customdata. The turn labels already use `_money`, so hover and labels would then agree. Alternatively, use a non-SI d3 format such as `$%{customdata[5]:,.3r}`. Pin it with a test that a 2010 wave row's hover price contains no `m` suffix.

## Minor

### M-1: `_cycle_params` raises on an unhashable basis from a crafted link
`callbacks/charts/__init__.py:900`: `fit_basis in _pc.BASES` tests membership in a dict. When the fit is on, a share link that carries a list or dict for `bub-pctile-fit-basis` raises `TypeError` (`head_checks.py`: `basis ['cal'] -> EXC TypeError cannot use 'list' as a dict key`), and `update_bub_pctile` returns a 500. The model and horizon checks are tuple/list membership and are safe. The spec's coercion rule says "share links can carry anything". Reachable only by a hand-crafted link.
**Fix:** `fit_basis if isinstance(fit_basis, str) and fit_basis in _pc.BASES else …`, and add a crafted-type case to `test_update_callback_passes_coerced_cycle_params`.

### M-2: A widely spread log-element bootstrap can push a timing bound past pandas' year-2262 limit and crash the figure
- `figures/percentile.py:120` (`_mon_yr`) feeds `turn_timing`'s 10/90 bounds (`:455`) to `pd.Timedelta`. For the log element, a replicate with low `f` puts its next turn at `t_anchor·e^(1/f)`, which can be thousands of years away.
- `logtiming.py` / `logrisk.py`:
  - QR/PL/SatPL log replicates span the whole search range (`f` 0.159–4.77).
  - 2.0–4.0 % of replicates already land beyond 250 yr. The crash needs more than 10 %, so the 90th percentile stays at 31–59 yr today.
  - Forcing 15 % of QR log replicates to `f = 0.2`, inside the searched range, gives `build EXC OutOfBoundsTimedelta` from `_mon_yr`.
- Latent: only the non-default log element is affected, the margin is about 2.5×, and the seed is fixed. But the residuals change daily.

**Fix:** clamp each timing bound to the axis edge (or to `t_last + 100`) before formatting, or return "beyond 2100" from `_mon_yr` for t past a cap. Also show the clamp in the bar.

### M-3: Wording around the optimistic (non-QR) hindcast contradicts itself or is ambiguous
- **Contradiction.** `:487`: when `edge is False`, the chart box says "⚠ no **out-of-sample** edge over a flat line" for every model. For a non-QR model that sits beside a legend entry reading "(not checked out-of-sample)". The live LPPL/log render shows both at once.
- **Ambiguous bound.** `:317`: "Against today's fan, which has seen the later data (an upper bound), the wave's mean miss is 21.5 vs 31.6 pp". The phrase "an upper bound" can be read as bounding the misses. The leak actually inflates the wave's *edge*: QR's margin is 5.6 leaky vs 1.5 honest per r2, and the flat line's miss moves the other way.
- **Wrong fallback text.** If the Time Machine grid is missing, QR falls back to this same sentence, which then says "only QR has as-of fans" about QR itself.

**Fix:**
- Non-honest box text: "⚠ no edge over a flat line, even against today's fan".
- Hindcast sentence: "…an optimistic check (the gap below overstates the wave's edge)".
- Branch the fallback wording when `model_key == "qr"`.

### M-4: The at-limit weak fit still shows its grid-edge period in the legend
`:398`: `per_txt` ignores `at_limit`. The live HybPPL `cal` legend reads `'HybPPL cycle fit (2.00 yr)'`, while the summary correctly says "period at the search limit (not an estimate)". This is r2 M-4, resolved in the summary but not in the legend.
**Fix:** when `fit.at_limit`, make the legend suffix "(period at search limit)".

### M-5: With censor on, QR's hindcast line still reads "no look-ahead", but the fits withheld a window chosen in hindsight
- `:309`: QR's honest margin is +1.68 pp with censor off and +2.76 pp with it on (20.58 vs 23.34, sweep). Scoring uses every day, as r2 I-3 asked.
- The extra pp comes from fitting without the 2019–20 window, which was picked after seeing the misfit (r2 I-3, "Why").
- The R² label mentions the exclusion. The hindcast line does not.

**Fix:** when `cyc["censor"]`, append "(each fit withheld 2019–20)" to the hindcast line.

### M-6: A cold fit allocates about 76 MB transiently (ALARA)
- `pctile_cycle.py:139`: `_grid_best` builds `W`, `C`, `S` and three products for 400 frequencies × 5,951 daily points, about 19 MB each, for the central fit only (`figures/percentile.py:229`). The bootstrap and hindcast run on the weekly subsample.
- Measured: tracemalloc peak 76 MB, `ru_maxrss` +63 MB.
- The memory is freed afterwards, but a few workers computing cold keys at once spike RAM on the VPS.

**Fix (result unchanged):** scan the grid in chunks of about 50 frequencies, or fit the central wave on the weekly series the way every other fit is done. The second option shifts the numbers slightly and needs the spec's figures re-measured.

## Nit

- **N-1: Stale module docstring.** `pctile_cycle.py:18-22` still says "for BM and the oscillator models no single element beats a flat forecast out-of-sample — which is why `is_weak` gates on the hindcast". After the r2 fold both halves are false: `lppl` master forecasts, and `is_weak` ignores the hindcast. Separately, `available_models` sits between `HORIZONS` and `DEFAULT_HORIZON` (`:62-74`).
- **N-2: The fit-off early return checks only the first trigger.** `charts/__init__.py:960` tests `ctx.triggered_id`. The Dash review suggested testing all triggers: `set(ctx.triggered_prop_ids.values()) <= _PCTILE_SUB_CONTROLS`. All four live restores render, so this is not a defect today. It would become one if a batch ever listed a sub-control first.
- **N-3: Noise-level binaries on screen.**
  - The default view's ⚠ "no out-of-sample edge" box was on 39 weeks ago (margin −0.13) and is off now (+1.68), so it will toggle on a months timescale (`margin.py`).
  - HybPPL `calw` has R² 0.205 against a 0.20 gate, so its forecast will come and go with daily data.
  - Both follow the settled rules. This is noted only so the operator is not surprised in prod.
- **N-4: The PNG subtitle overflows on phones.** The roughly 110-character subtitle (`:520`) is cut at 380 px ("· oute…", `phone_log.png`). Plotly titles do not wrap.

## Not re-raised
- Settled operator decisions.
- Payload growth (Dash review M4, accepted).
- Tabs left open across the deploy (Dash review M2).
- Pill-row wrap in every view (Dash review N3).
- Registry eviction. `a35ffa00` now also drops predating keys for unregistered fingerprints.

## Not checked
Prod timing and RAM headroom on the VPS, phone-browser rendering of the control row, the E2E suites, and non-resqr σ-modes beyond the sweep's crash/NaN checks.
