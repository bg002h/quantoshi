# Custom Time Axis (Tab 1) — Design Spec

**Status:** ⏸ PAUSED DURING BRAINSTORMING — section 1 approved, sections 2-6 not yet started.
**Resume:** Say "Resume Custom Time Axis brainstorming — read the spec file." Start at section 2 (Panel layout / UX).

---

## Problem statement

Allow the user to redefine the time axis on Tab 1 of the Quantoshi web app. The user picks:

- **Time scale** — calendar time (years since t0) OR bitcoin blockheight (blocks since block0)
- **t=0 anchor** — preset or custom date/block
- **Fit weighting** — applies to log-log models only (PL, QR, BM-floor); Exponential ignores
- **Model subset** — a reduced set of fast-fitting models

Fits run server-side at ~300 ms budget per config change. No caching. Custom Time Axis only affects Tab 1's plot — all other tabs continue to use the default time axis (years since 2009-07-25).

## Locked decisions (from brainstorming 2026-04-13)

### UX scope
- Tab-1 only. New collapsible panel below Display Models.
- Warning banner on the panel: *"Custom Time Axis only affects this tab. All other tabs continue to use the default time axis (years since 2009-07-25)."*
- Panel title: **"Custom Time Axis"**.
- Chart behavior: **swap view** — "Activate Custom Time Axis" toggle replaces the standard bubble chart with a custom view. Standard traces hide, custom-fit traces appear. Deactivation restores the standard view.

### Model subset
Exactly 4 models, all live-refit per config change:
- **Power Law (PL)** — closed-form OLS on log-log. ~1 ms.
- **Quantile Regression (QR)** — linprog, 9 quantiles. ~50–200 ms.
- **BM floor only** — `fit_support()` output. No bubble composite. No future bubble extrapolation. ~100-200 ms.
- **Exponential (Exp)** — closed-form on log-linear. ~1 ms. Included because some t=0 choices make PL look worse than Exp.

Total budget per request: ~300 ms.

### t=0 preset list
Both calendar and block presets for the same underlying moments:

**Calendar anchors:**
- Bitcoin whitepaper — **2008-10-31**
- Genesis block — 2009-01-03
- Current optimal (from change_origin analysis) — **2009-07-25** (default)
- New Liberty Standard first $/BTC quote — 2009-10-05
- Bitcoin Pizza Day — 2010-05-22
- Mt. Gox launch — 2010-07-17
- **Custom...** — opens `dcc.DatePickerSingle`

**Block anchors** (same moments, as blockheight):
- Block 0 — genesis
- Block ~3300 — 2009-07-25 equivalent
- Block ~32000 — first dollar trade
- Block ~67700 — pizza day
- Block ~70000 — Mt. Gox launch
- **Custom...** — opens a numeric input

### Negative-t handling
- Drop samples with t ≤ 0 for log-log models (PL, QR, BM-floor).
- Keep full series for Exponential (handles any t).
- Per-model sample count shown in legend: e.g. `"PL (n=4,521)"`.
- Show a notice in the panel header: *"Fitting on N samples from {t0_date} onward ({dropped} samples before t=0 excluded)"*.

### Weighting dropdown
Applies to log-log models only (PL, QR, BM-floor). Exp is N/A (greyed label: "N/A for Exponential").

1. **Unweighted** (default, matches current behavior)
2. **1/t**
3. **1/√t**
4. **Uniform log-t density** — `w_i ∝ 1/local_density(log(t_i))` via kernel density estimate

### Block map data source
- **Storage:** parallel file `BitcoinBlocksDaily.csv` with columns `date,blockheight`. One row per calendar day, blockheight = block at 00:00 UTC (or last block before midnight). Forward-fill days with zero blocks (common in 2009). ~6,000 rows, ~150 KB total.
- **Generation:** one-shot `tools/build_block_map.py` on dev, queries local bitcoind RPC (greenfield — no existing RPC code in repo). Committed to the repo as a static file. Prod has no bitcoind; pulls the CSV via `git pull`. `update_prices.py` gains a dev-only splice step to append new rows alongside price updates.

---

## Section 1 — Architecture & data flow (✅ APPROVED)

**Co-owning `bubble-graph.figure`.** The existing `update_bubble` callback in `btc_web/callbacks/charts.py:953` has ~50 Inputs and re-fires on every bubble control change. A second writer would get clobbered every slider nudge. Fix:
- Add `dcc.Store("bub-custom-active")`.
- Guard the existing `update_bubble` with `if custom_active: raise PreventUpdate`.
- New Custom Time Axis callback writes `bubble-graph.figure` with `allow_duplicate=True, prevent_initial_call=True`.
- Router-by-state handoff, no DOM swap.

**BM-floor = `tools/model_toolkit/support.py:22 fit_support()`** — NOT a stripped-down BubbleModel. Build a fresh `PriceData` via `tools/model_toolkit/data.py:19 load_prices(csv_path, genesis_date=user_t0)` (already parametric in genesis). Call `fit_support(price_data, percentile=0.20)`. Zero new math.

**PL / Exp / QR math extraction.** `btc_core.py:684 PowerLawModel.__init__` hard-masks `price_years >= 1.0` which drops everything for late t0 choices. New module uses the inner math only:
- PL: `scipy.stats.linregress` on `(log10(t[t>0]), log10(price))`
- Exp: `linregress` on `(t, log10(price))` (linear t, no log)
- QR: inner loop of `fit_qr_from_csv` (`btc_core.py:256`) — `statsmodels.QuantReg` on `log10(t)`, `log10(price)`
- No class constructors reused.

**Prod has no bitcoind.** `BitcoinBlocksDaily.csv` committed to repo. Dev generates via `tools/build_block_map.py`. Prod `git pull`s. No runtime RPC dependency, no RAM cost beyond ~150 KB.

**Snapshot.** Use `dcc.Dropdown` for single-select (time scale, t0 preset, weighting). Use `dcc.Checklist` only for multi-select (model subset). Register the checklist in `_CHECKLIST_OPTIONS` in `snapshot.py` for bitmask encoding. Add 8 control IDs to `_SNAPSHOT_CONTROLS` and `_TAB_CONTROLS["bubble"]`. `test_snapshot.py:241-246` catches any omissions automatically.

**New files:**
- `btc_web/engines/custom_fit.py` — 4 pure fit functions
- `btc_web/layout/custom_time.py` — panel UI
- `btc_web/callbacks/custom_time.py` — server callback + Store-based guard
- `tools/build_block_map.py` — one-off bitcoind RPC → CSV (dev only)
- `BitcoinBlocksDaily.csv` — committed data file
- `btc_web/test_custom_time.py` — unit tests
- `btc_web/test_custom_time_e2e.py` — Playwright E2E

**Reusable code pointers** (use, don't re-implement):

| Need | Use |
|---|---|
| Re-fit support line on custom t0 | `tools/model_toolkit/support.py:22` `fit_support(PriceData, percentile, quantile)` |
| Build fresh `PriceData` with custom genesis | `tools/model_toolkit/data.py:19` `load_prices(csv_path, genesis_date=...)` |
| QR fit math template | `btc_core.py:256-276` `fit_qr_from_csv` (extract inner loop) |
| PL fit math template | `btc_core.py:687-705` `PowerLawModel.__init__` (drop `>=1.0` mask) |
| Exp fit math template | `btc_core.py:2213-2225` `ExponentialModel.__init__` |
| In-memory price series | `_app_ctx.M.price_years`, `_app_ctx.M.price_prices` (set at `btc_web/app.py:55`) |
| Date helpers | `btc_core.py:180` `yr_to_t(cal_year, genesis=...)`, `btc_core.py:188` `today_t(genesis=...)` |
| Snapshot reverse-decoder pattern | `btc_web/snapshot.py:12` `_SNAPSHOT_CONTROLS`, `_CHECKLIST_OPTIONS` |
| Tab control set test guard | `btc_web/test_snapshot.py:241-246` |

---

## ⏸ PAUSED HERE

## Remaining sections to brainstorm

### Section 2 — Panel layout / UX
What the user sees and clicks. Control order, labels, warning banner styling, Activate toggle placement, where per-model sample counts appear, how "Custom..." opens the date picker / block input.

### Section 3 — Fit engine (`engines/custom_fit.py`)
The 4 fit function signatures. Weight computation. Return shape (params + trace arrays for plotting). How weights are applied in QR (statsmodels `QuantReg(weights=...)` vs manual). How "uniform log-t density" is computed (KDE, histogram, bandwidth choice).

### Section 4 — Block map pipeline
`tools/build_block_map.py` implementation sketch — bitcoind RPC flow, how to find "first block of day" efficiently, forward-fill policy, how `update_prices.py` splices append rows.

### Section 5 — Error handling
- bitcoind down during build (dev only)
- User picks custom date after data's last row → all t > 0 but data beyond
- User picks a t0 that leaves < 10 samples
- QR fails to converge
- Snapshot restore of old link without custom controls

### Section 6 — Testing
- Unit tests for `custom_fit.py` (each fit function, each weighting scheme)
- Integration test for the Store router (active/inactive switching)
- E2E test for the full panel (Playwright)
- Snapshot roundtrip test for the 8 new controls

---

## Not in v1 (deferred)

- **"Scan mode"** — user clicks two points on the bubble chart, panel auto-fits PL/QR/Exp through a grid of t=0 candidates between them, highlights best R². Tracked in `UrgentTodoItems.md` under "Deferred feature ideas".
- LPPL / HybPPL / EPPL support in the custom panel — all too slow for live refit.
- Monte Carlo overlay in the custom view — paid feature, out of scope.
- S2F — doesn't depend on t=0 in the same way (its own x axis).
