# Custom Time Axis (Tab 1) — Design Spec

**Status:** ✅ COMPLETE — all sections approved 2026-04-13.
**Next step:** transition to `superpowers:writing-plans` to generate implementation plan.

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
- New `dcc.Store("bub-redraw-tick", data=0)` — a monotonic counter used as a trigger by the existing bubble callback without adding user-state coupling.
- **Existing `update_bubble`** gains one new Input (`bub-redraw-tick.data`) and one new State (`cta-active.value`). Guard body: `if cta_active and len(cta_active): raise PreventUpdate` — prevents the standard figure from clobbering the custom figure when Custom Time Axis is active.
- **New Custom Time Axis callback** writes `bubble-graph.figure` with `allow_duplicate=True, prevent_initial_call=True`, plus `cta-status.children`, plus `bub-redraw-tick.data`. On activate: `(custom_fig, status, no_update)`. On deactivate: `(no_update, "Standard view restored.", tick+1)` — the tick bump re-fires `update_bubble`, which sees `cta_active` is falsy and writes the standard figure. Clean handoff in both directions.
- Router-by-state handoff, no DOM swap, no duplicated figure-building code.

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

## Section 2 — Panel layout / UX (✅ APPROVED)

**Panel slot.** Append below `display_models_panel("bub", ...)` at `btc_web/layout/bubble.py:78`, before the Model Component Decomposition card. Wrap in `_section_card("Custom Time Axis", ..., no_hover=True)` (`no_hover=True` required for panels containing dropdowns — iOS Safari clip bug, `common.py:240-244`). Nearest sibling pattern: Model Scanner card at `bubble.py:137-163`.

**Body visibility.** `html.Div(id="cta-body", style=_STYLE_HIDDEN)` toggled via clientside callback from `cta-active`. NOT `dbc.Collapse` — per CLAUDE.md, dropdowns inside Collapse don't render full height. Matches `dca-sc-body` / MC body pattern at `mc_controls.py:29`.

**Warning.** Single-line `html.Small` in `DIM_TEXT` + `UI_FONT_SM`, mirroring User Model hint at `bubble.py:191-192`:
*"⚠ Custom Time Axis changes only affect this tab. Other tabs stay on the default axis (years since 2009-07-25)."*

**Row 1 — Activate toggle.** `dcc.Checklist(id="cta-active", options=[{"label": "Activate Custom Time Axis", "value": "yes"}], value=[])` styled as a pseudo-button. Single bit for snapshot, keyboard nav native. Precedent: `bub-auto-y` (`bubble.py:50-54`), `dca-sc-enable`, `cp-tax-toggle`. Drives `cta-body` visibility AND the section-1 Store router.

**Row 2 — Time scale.** `dcc.RadioItems(id="cta-scale")` horizontal: Calendar (default) / Blockheight.

**Row 3 — t₀ anchor.** Two wrapper divs (`cta-t0-cal-wrap`, `cta-t0-blk-wrap`), visibility gated by `cta-scale` via clientside callback (mirrors `bub-yrange-wrap` ↔ `bub-auto-y` at `bubble.py:56`).
- **Calendar:** `dcc.Dropdown(id="cta-t0-cal")` with 6 presets + "Custom…". Below: `html.Div(id="cta-t0-cal-custom-wrap", style=_STYLE_HIDDEN)` containing `dcc.DatePickerSingle(id="cta-t0-cal-custom", min_date_allowed="2008-10-31", max_date_allowed=<today>, display_format="YYYY-MM-DD", initial_visible_month=<preset>)`. Revealed when `cta-t0-cal == "custom"` (mirrors `hm-mode == "custom"` at `sim_tabs.py:66,75`).
- **Block:** `dcc.Dropdown(id="cta-t0-blk")` with 5 presets + "Custom…". Below: `html.Div(id="cta-t0-blk-custom-wrap", style=_STYLE_HIDDEN)` containing `dbc.Input(id="cta-t0-blk-custom", type="number", min=0, step=1, max=<latest_block>, debounce=True)`. Min/step per CLAUDE.md HTML5-number footgun rules.

**Row 4 — Weighting.** `dcc.Dropdown(id="cta-weighting")` single-select: Unweighted (default) / 1/t / 1/√t / Uniform log-t density. Info icon via `_INFO_ICON` constant (`layout/common.py:26`, the 📐 constant). Below: `html.Small("Applies to PL, QR, BM-floor. Exponential ignores.", style=_MUTED_STYLE)`.

**Row 5 — Model subset.** `dcc.Checklist(id="cta-models")` with 4 checkboxes all on by default: Power Law, Quantile Regression, BM floor, Exponential.

**Footer — Fit status.** `html.Div(id="cta-status")` in body, `UI_FONT_SM` + `DIM_TEXT`, matching Model Component Decomposition formula divs at `bubble.py:106-112`.
- Inactive: *"Press Activate to fit."*
- Active + fitting: *"⏳ Fitting…"* (set clientside on `cta-active` flip)
- Active + done: *"N samples fit from {t0_date} onward ({dropped} before t=0 excluded). Xms."*
- Spinner flow: clientside write on button flip → server callback overwrites with final status string when fitting completes. Two-write sequence, no extra machinery.

**Legend labels.** Trace `name` gets a suffix `f" (n={n:,})"`. Fall back to `f" (n={n//1000}k)"` if > 9,999.

**Snapshot registration.**
- `_CHECKLIST_OPTIONS["cta-active"] = ["yes"]` (1 bit)
- `_CHECKLIST_OPTIONS["cta-models"] = ["pl", "qr", "bm_floor", "exp"]` (4 bits) — order **load-bearing for bitmask encoding; freeze this list**. Add a comment in `snapshot.py` next to the registration mirroring the defunct-LPPL warning at `snapshot.py:403-404`. Reordering or removing entries breaks old share links.
- 8 new IDs in `_SNAPSHOT_CONTROLS`
- 8 new IDs in `_TAB_CONTROLS["bubble"]` at `routing.py:147`
- Dropped optimization: "only contribute `cta-t0-*-custom` when parent dropdown == custom" — not implementable without a 4-line encoder change at `snapshot.py:485-502` and the win is only ~10-20 URL chars. Both custom fields always serialize; empty-string when the parent is on a preset.

**Control IDs (8 total):** `cta-active`, `cta-scale`, `cta-t0-cal`, `cta-t0-cal-custom`, `cta-t0-blk`, `cta-t0-blk-custom`, `cta-weighting`, `cta-models`.

---

## Section 3 — Fit engine `btc_web/engines/custom_fit.py` (✅ APPROVED)

### Cached-array architecture
Module import loads price + block arrays ONCE per worker:
```python
_DATES, _PRICES = _load_price_arrays_once()   # pd.DatetimeIndex, np.ndarray[float]
_BLOCKS = _load_block_array_once()            # np.ndarray[int], index-aligned with _DATES
```
Never re-read at request time. Per-request work is `numpy.subtract` on the cached arrays (~0.5 ms).

### `build_fit_input(scale, t0, weighting)`
```python
if scale == "calendar":
    t_raw = (_DATES - pd.Timestamp(t0)).days.values / 365.25     # years
else:  # "block"
    t_raw = (_BLOCKS - int(t0)).astype(float)                    # raw block offset
return FitInput(t=t_raw, price=_PRICES, weighting=weighting)
```
**No `_BLOCKS_PER_YEAR` constant.** Block mode uses raw block offsets as the time axis — the whole point of block mode is that block intervals are NOT linear in calendar time.

### Data classes
```python
@dataclass(frozen=True)
class FitInput:
    t: np.ndarray         # user-chosen scale (years or block-units)
    price: np.ndarray
    weighting: str        # "none" | "inv_t" | "inv_sqrt_t" | "log_density"

@dataclass(frozen=True)
class FitResult:
    name: str                          # "PL" / "QR" / "BM floor" / "Exp"
    params: dict                       # {slope, intercept} etc
    t_plot: np.ndarray                 # log-spaced for PL/QR/BM, linear for Exp
    y_plot: np.ndarray | dict          # array, or {q: y_arr} for QR
    n_samples: int                     # after t>0 filter (Exp sees all)
    r2: float                          # NaN for QR
    elapsed_ms: float
    note: str | None                   # "Skipped: insufficient samples", "Weighting degraded", etc
```

### Four public functions
- **`fit_pl(fi)`** — mask t>0, `linregress` (unweighted) or `np.polyfit(..., w=np.sqrt(weights))` (weighted). Weighted R² computed manually: `1 - Σw·resid² / Σw·(y-ȳ_w)²`. Inline comment citing numpy polyfit docs (loss is `Σ w[j]² · r[j]²`).
- **`fit_qr(fi, quantiles)`** — mask t>0. **No `QuantReg.fit(weights=...)` — doesn't exist in any statsmodels version.** Weighted modes use **multinomial resampling** (draw 5×N samples with `p ∝ w`). Quantile semantics survive because the empirical CDF of the resampled set matches the weighted CDF in expectation. Document as approximate. Full 9 quantiles for n≥30; reduced `(0.25, 0.5, 0.75)` for 10≤n<30; skipped for n<10.
- **`fit_bm_floor(fi)`** — reuses `tools/model_toolkit/support.py:22 fit_support()` via a `_PriceDataShim` (duck-types `log_years`, `log_prices`, `df_full["log_years"]`, `df_full["log_price"]`). **Shim writes the column verbatim as `log_years`** — misleading name in block mode (where it holds `log10(block_offset)`) but `support.py:38-39` reads that exact name. 1-line comment in the shim notes this. Skip for n<50. Lower the upstream `years >= 1.0` threshold to `years >= 1/365.25` (one day) to salvage late-t0 fits.
- **`fit_exp(fi)`** — uses ALL samples (no mask), `linregress(t, log10(price))`. Ignores `weighting`.

### Shared weight computer
```python
def _compute_weights(t_positive, mode):
    n = len(t_positive)
    if mode == "none" or n < 30:
        return np.ones(n), (n < 30 and mode != "none")   # (weights, degraded_flag)
    if mode == "inv_t":      return 1.0 / t_positive, False
    if mode == "inv_sqrt_t": return 1.0 / np.sqrt(t_positive), False
    if mode == "log_density":
        kde = scipy.stats.gaussian_kde(np.log10(t_positive))  # Scott's rule
        return 1.0 / np.maximum(kde(np.log10(t_positive)), 1e-9), False
```
Weights normalized to mean=1.0. KDE degrades to unweighted for n<30; `degraded=True` surfaces in `FitResult.note`.

### Min-sample guards per model
| Model | Min n | Below |
|---|---|---|
| PL | 3 | return None |
| QR (full 9q) | 30 | reduced to 3q |
| QR (reduced 3q) | 10 | return None |
| BM-floor | 50 | return None |
| Exp | 3 | return None |

### Block-mode semantics (consequences of no `_BLOCKS_PER_YEAR`)
- Log-log models operate on `log10(t)` regardless of scale. Calendar: log-years (0..1.2). Block: log-blocks (0..5.9). **Slope is dimensionless; intercept differs between modes by a large constant offset (≈ log10(blocks/year)).** Expected. Consumers do not compare fit params across modes. The visual bubble/support composite should look qualitatively similar between modes for the same data. Regression test will spot-check.
- Exp `b` exponent has units of "log-price per year" in calendar mode vs "log-price per block" in block mode. Expected.
- Weighting mode `log_density` is scale-invariant (KDE bandwidth uses empirical std, translation-invariant under `log10(k·x) = log10(x) + log10(k)`). `1/t` and `1/√t` rescale magnitudes but preserve monotone ordering. No correction needed.
- **Chart x-axis in block mode** shows raw blockheight (e.g. 200,000 / 400,000 / 600,000). `layout.xaxis.title` → "Blockheight since block {t0_block}".
- **Forward-filled duplicate-t:** days in 2009 where no new blocks were mined share the same `blockheight` across multiple calendar-day price samples. `linregress` / `polyfit` / `QuantReg` handle duplicates fine (well-posed). Effective sample size in 2009 drops from ~150 calendar-days to ~30-50 unique blockheights — **intentional**: this is how block-event density naturally down-weights the early volatile low-block-count period. Document; do not "fix".

### X-range control (deferred from section 2 UX)
**Custom Time Axis callback ignores `bub-xrange` entirely.** The existing slider assumes calendar years and would pass nonsense in block mode. Custom callback always fits over the full data range. Users zoom via Plotly's built-in drag-zoom on the rendered chart. No parallel block-range slider. One less control on the panel.

### Performance budget (per request)
| Step | Time |
|---|---|
| Cached-array rebuild (`np.subtract`) | ~0.5 ms |
| `fit_pl` | ~2 ms |
| `fit_qr` (n=5000, 9q, weighted resample) | ~150-250 ms |
| `fit_bm_floor` (shim + `fit_support`) | ~30-60 ms |
| `fit_exp` | ~1 ms |
| `_compute_weights` (log_density KDE) | ~5 ms |
| **Total** | **~190-320 ms** |

Within the 300 ms budget. No CSV re-read.

### Concurrency
All fit functions are pure on local arrays. `statsmodels.QuantReg`, `scipy.stats.linregress`, `scipy.stats.gaussian_kde` build per-call objects. Safe for concurrent gunicorn workers.

---

## Section 4 — Block map pipeline (✅ APPROVED)

### `tools/build_block_map.py` — three CLI modes (dev only)
```bash
python3 tools/build_block_map.py --full     # one-time build, ~5-10 min
python3 tools/build_block_map.py --append   # append missing rows, ~1 sec
python3 tools/build_block_map.py --verify   # sanity check, ~1 sec
```

### RPC connection (stdlib `http.client` + JSON-RPC, no external dep)
Auth resolution order:
1. `BITCOIN_RPC_URL` env var (explicit override) — `http://user:pass@host:port/`
2. `~/.bitcoin/.cookie` (modern default, mainnet)
3. Parse `~/.bitcoin/bitcoin.conf` for `rpcuser`/`rpcpassword` (legacy)

Testnet/signet/regtest cookies: `~/.bitcoin/testnet3/.cookie` etc. Default mainnet.

Three RPC methods: `getblockcount`, `getblockhash(h)`, `getblockheader(hash)`. JSON-RPC batching used throughout (Bitcoin Core accepts a JSON array of requests).

### Row semantics
Each row = blockheight of the chain-tip block at `midnight_utc(D+1)`. Uses **`time`** (wall-clock nTime), not `mediantime` (BIP113 MTP). Forward-fill for zero-block days is automatic (running-max logic).

### Algorithm — `--full` (running-max time table, safe against non-monotonic timestamps)
**Why not binary search by timestamp:** block `time` is not monotonic. BIP113 only requires `time` to exceed the median of the previous 11 blocks, so two consecutive blocks can have `time` values in either order (by up to ~2 hours). Binary searching `time` directly can return the wrong block.

1. `getblockcount()` → current tip H.
2. Estimate earliest needed height: `h_start = max(0, H - (today - first_price_date).days × 150)`. Overshoot intentional.
3. Walk heights sequentially from `h_start` to H. For each batch of ~1000 heights:
   - `getblockhash(h)` batched → list of hashes
   - `getblockheader(hash)` batched → list of headers with `time`
4. Build aligned arrays `heights[]` and `times[]`.
5. Compute `running_max_time[h] = max(times[0..h])` — monotonic by construction.
6. For each date D in `BitcoinPricesDaily.csv`, binary search `running_max_time` for the highest h where `running_max_time[h] < midnight_utc(D+1)`. Record that `h` as the blockheight for date D.
7. Write to temp file as ISO `YYYY-MM-DD,blockheight`. Atomic rename at end.

Total ~815 HTTP round trips (batched at 1000), ~5-10 min on localhost.

### Algorithm — `--append` (fills any gap, not just future rows)
1. Load both CSVs. Compute `missing = set(price_dates) - set(block_dates)`.
2. Sort missing dates, find required height range. Binary-search-safe version of the running-max approach for just those rows (~20 RPC calls).
3. Append rows, atomic rename.

Fills gaps from manual edits, source switches, past-date inserts. Not just "future rows since last entry".

### Algorithm — `--verify`
Sample first 5 rows + last 5 rows + 10 random rows. Re-run the running-max lookup for each and compare. Fail loudly on mismatch. Boundary bugs are most likely to escape uniform sampling.

### `update_prices.py` integration
```python
subprocess.run([sys.executable, str(ROOT / "tools" / "build_block_map.py"), "--append"],
               check=False)
```
`check=False` so bitcoind being down doesn't abort the price update. Log a warning on non-zero exit. Dev-only in practice (prod doesn't run `update_prices.py` — prod gets data via `git pull`).

### Output file: `BitcoinBlocksDaily.csv`
```
date,blockheight
2010-07-17,67000
2010-07-18,67148
...
2026-04-13,895217
```
- **Location:** repo root (alongside `BitcoinPricesDaily.csv`)
- **Format:** ISO `YYYY-MM-DD` dates (sortable, unambiguous). Price CSV uses `M/D/YY`; the alignment join in section 3's `_load_block_array_once()` MUST use explicit `pd.to_datetime(..., format=...)` on both sides and verify row-for-row alignment. Fail loudly at worker startup if alignment breaks.
- **Size:** ~5,743 rows × ~20 bytes = ~115 KB
- **Header line:** `date,blockheight`. No comment line (matches price CSV convention).
- **Committed to git.** Prod pulls as static file. Read-only at runtime.

### Testing
- Factor `_rpc(method, params)` as a single function. Tests monkeypatch it.
- **Fixture:** 100 blocks spanning a known non-monotonic timestamp pair cherry-picked from early 2010. Locks in the running-max algorithm's correctness against real Bitcoin history.
- **Regression test:** assert specific known boundaries (e.g. blockheight at `2010-07-17 23:59:59 UTC` from a known-good source) to catch any future re-refactor that reintroduces the binary-search bug.

### Failure modes
- **bitcoind unreachable** — exit 2 with: `"bitcoind unreachable at <url>. Start it with 'bitcoind -daemon' or set BITCOIN_RPC_URL."`
- **Partial `--full` interrupted** — temp file + atomic rename protects existing CSV.
- **Chain reorg shifting old blockheights** — `--verify` catches it; reorgs beyond 6-block depth are essentially impossible for old dates, check is paranoia.
- **Cookie file not readable** — fall through to `bitcoin.conf` parse; if both fail, actionable error message pointing to `BITCOIN_RPC_URL` env var.
- **`--append` race with next-day `update_prices.py`** — append is idempotent against gap-fill logic; worst case an extra run fills the same rows with identical values.

### Files touched
- **New:** `tools/build_block_map.py` (~250 lines incl. tests)
- **New:** `BitcoinBlocksDaily.csv` (committed data file)
- **Modified:** `update_prices.py` (+3 lines: subprocess call + log + comment)

---

## Section 5 — Error handling (✅ APPROVED)

### Global constraint: custom dates restricted to before 2016-01-01
- `DatePickerSingle(cta-t0-cal-custom, min_date_allowed="2008-10-31", max_date_allowed="2015-12-31", ...)` — widget cap.
- `cta-t0-blk-custom` capped at `_BLOCK_CAP = _lookup_block_for_date("2015-12-31")` (~block 391,000), looked up from `BitcoinBlocksDaily.csv` at module import. If `_BLOCKS is None`, `_BLOCK_CAP = None`; layout hides the block "Custom…" option.
- **Callback-level guard** always checks: Dash's react-dates accepts keyboard-typed post-cap dates that bypass the widget. Redundant defense required.
- **Preset drift guard:** unit test iterates `CAL_PRESETS` / `BLK_PRESETS` and asserts every calendar preset `< date(2016,1,1)` and every block preset `< _BLOCK_CAP`.
- **Preset constants home:** new `btc_web/_custom_time_presets.py` with two frozen lists. Imported by `layout/custom_time.py` and `test_custom_time.py`. Single source of truth.

### Case-by-case

**A. Too few samples after t>0 filter.** Per-model min-sample guards in `custom_fit.py`: PL/Exp≥3, QR≥10 (reduced 3q) / ≥30 (full 9q), BM-floor≥50. Skipped models appear in legend with muted `"PL — skipped (n=2, need ≥3)"`. Status div summarizes. No error toast.

**B. Custom date after data's last row.** Prevented by the 2016-01-01 cap. Callback still guards against snapshot URL bypass. Returns `no_update` for figure (preserves last good chart) + red-tinted status *"⚠ t₀ after available data. Pick an earlier date."*

**C. QR / BM-floor / PL fit failure.** `try/except (LinAlgError, ValueError, scipy.stats.FitError)` around each fit call. Per-model independent. Failing model returns `FitResult(note="Fit failed: <short msg>")`. Legend: `"QR — fit failed"`. `logger.warning("custom_fit qr failed: %s", exc)`. Other models render normally. **`_compute_weights` has its own try/except** — KDE failures fall back to uniform with `degraded=True, note="weighting failed: <msg>; using uniform"`, surfaced in the FitResult note.

**D. Snapshot restore of old link without Custom Time Axis controls.** Decoder returns `None` for missing keys; `_CHECKLIST_OPTIONS` bitmask None-default pattern kicks in; `cta-active = []` (inactive), all other controls at dropdown/input defaults. Forward-compatible by construction. Old links keep working.

**E. `BitcoinBlocksDaily.csv` missing at worker startup (deployment race).** `_load_block_array_once()` wraps in try/except(FileNotFoundError). Logs error, sets `_BLOCKS = None`. Module import succeeds. At request time: if `scale == "block" and _BLOCKS is None`, callback returns `no_update` for figure + status *"⚠ Block mode unavailable: BitcoinBlocksDaily.csv missing. Calendar mode still works."* Calendar mode unaffected. Graceful degradation.
- **`/health` JSON route** (`app.py:104-135` — already JSON) gains `block_map_loaded: bool` field. `scripts/quantoshi-health` flags `false` post-deploy.

**F. Callback > 5 seconds.** Wall-clock timer at top of callback. `logger.warning("custom_fit slow: %dms, params=%s", elapsed, params_dict)` if `elapsed > 5000`. No user throttling; gunicorn's 120s timeout handles runaway (`run_web.sh:69`, `btc-web.service:14`). Budget is 300 ms typical; anything > 1 sec is a bug to diagnose from logs.

**G. Spinner-never-clears race + top-level error wrapper.** Entire callback body wrapped in `try/except Exception as e: logger.error("custom_fit crash: %s", e, exc_info=True); return dash.no_update, f"⚠ Internal error: {type(e).__name__}", dash.no_update`. Unexpected crashes surface in the UI AND the error log. Prevents Dash silent-failure mode (CLAUDE.md `BackgroundCallbackError` section). User keeps their last good chart; status div shows the error type.

**H. Active-flip races with model-subset change.** Accept the brief flicker; Dash's callback queue serializes per-output writes. No special handling.

**I. "Custom…" picked but no value entered / value mid-edit (merged with Q).** Top of callback:
```python
if scale == "calendar" and cta_t0_cal == "custom" and cta_t0_cal_custom is None:
    raise PreventUpdate
if scale == "block" and cta_t0_blk == "custom" and cta_t0_blk_custom is None:
    raise PreventUpdate
```
`dcc.DatePickerSingle` emits `None` while user is typing/clearing — `PreventUpdate` avoids crashing on `pd.Timestamp(None)`. Status hint: *"Enter a date to fit."* / *"Enter a block number to fit."*

**J. QR weighted-resample duplicates.** QuantReg LP solver is well-posed on ties. No action.

**K. Structured log lines prefixed `custom_fit:`.** Format: `custom_fit <level>: <message>: <detail> at t0=<t0> weighting=<mode> scale=<scale>`. Grep-friendly for 27-day prod logs.

**L. All 4 models deselected in `cta-models`.** Top of callback: `if not cta_models: raise PreventUpdate`. Status: *"Select at least one model to fit."* Previous chart preserved via `no_update`.

**M. Unrecognized `cta-weighting` from future-version snapshot.** `_compute_weights` else-branch catches unknown strings → fall back to `"none"` with `note="unknown weighting '{mode}'; using uniform"`. No `KeyError`. Direction: old server, new snapshot URL from an upgraded client — silent fallback is the only option since the old server can't honor modes it doesn't know; note surfaces in status div.

**N. Mid-fit tab switch.** No special handling. Figure auto-resizes via existing `tab_dblclick.js` escape hatch when user returns to Tab 1. Documented so future debug sessions don't chase ghosts.

**O. Custom↔Standard deactivation transition.** Section 1 router uses a new `dcc.Store("bub-redraw-tick", data=0)`.
- **Custom callback outputs:** `[bubble-graph.figure, cta-status, bub-redraw-tick.data]`.
  - **On activate** (`cta-active` flips `[]` → `["yes"]`): `(custom_fig, status_text, dash.no_update)`.
  - **On deactivate** (`["yes"]` → `[]`): `(dash.no_update, "Standard view restored.", tick+1)` — figure handled by the tick bump, status updated.
  - **On error / PreventUpdate cases above:** `(dash.no_update, status_text, dash.no_update)`.
- **`update_bubble` gains one new Input:** `Input("bub-redraw-tick", "data")`. Its existing State guard (`if custom_active: raise PreventUpdate`) continues to protect the active-direction case. When the tick increments, `update_bubble` re-fires, sees `custom_active` is falsy in State, writes the standard figure. Clean handoff in both directions, zero duplicated figure-building code.

**P. Block CSV alignment failure at worker startup (DATA CORRUPTION).** Distinct from case E (file missing). `_load_block_array_once()` asserts:
```python
assert len(blocks) == len(_DATES), "block/price row count mismatch"
assert (blocks["date"].values == _DATES.values).all(), "block/price dates not aligned row-for-row"
```
Misalignment is data corruption → **hard-fail at worker startup**. Refuse to load. Log clear error. Gunicorn worker crashes. systemd restarts. Restart loop bounded by new policy:
```
# btc-web.service additions:
StartLimitIntervalSec=300
StartLimitBurst=5
```
After 5 failed restarts in 5 min, systemd gives up and marks the unit `failed`. `quantoshi-health` catches the `failed` state via its existing systemd probe AND the `/health` `block_map_loaded: false` flag.

### Failure-mode decision table

| Case | File status | User input | Worker behavior |
|---|---|---|---|
| E | missing | any | graceful degrade (block mode disabled, calendar works, `/health` reports false) |
| P | misaligned | any | **hard-fail** at worker startup (data corruption, systemd gives up after 5 restarts) |
| B | file OK | post-2016 date (snapshot bypass) | PreventUpdate + status message + `no_update` figure |
| L | file OK | no models selected | PreventUpdate + status message |
| I/Q | file OK | Custom… picked, no value | PreventUpdate + status hint |
| others | file OK | various | per-case handling |

### Callback-body check order

1. **Module readiness:** `_app_ctx.M is not None` (import-order safety), `_BLOCKS is not None` if scale=="block".
2. **Input validity:** cap bypass, Custom-but-None, no models selected, unknown weighting.
3. **Sample-count guards:** per-model min-n.
4. **Fit:** per-model independent try/except.
5. **Return:** `(figure, status, tick)` tuple.

All wrapped in top-level `try/except Exception` (case G).

---

## Section 6 — Testing (✅ APPROVED)

### Test files

**A. `btc_web/test_custom_time.py` — pure-function unit tests (~35 cases, <2 sec)**
Imports `conftest.M` for real price data. Monkeypatches `_BLOCKS` to a synthetic 200-row array for block-mode tests — no dependency on `BitcoinBlocksDaily.csv` existing.
- `_compute_weights`: 5 cases (each mode, n<30 fallback, KDE failure, unknown mode, mean=1 normalization split 4 ways)
- `fit_pl`: 5 cases (synthetic recovery, n<3 None, weighted vs unweighted params differ, t=0 drop, weighted R² uses weighted mean)
- `fit_qr`: 5 cases (9-quantile recovery, 10≤n<30 reduced 3q, n<10 None, weighted resample shifts, flat-series failure)
- `fit_bm_floor`: 5 cases (matches `fit_support()` ≤1e-6 on `M.price_prices[:200]` slice, n<50 None, block-mode shim column preserved, weighted shifts down, lowered 1/365.25 threshold)
- `fit_exp`: 3 cases
- `build_fit_input`: 4 cases
- Preset drift guard: 4 cases (all calendar <2016, all block <`_BLOCK_CAP`, tuples immutable, exact counts 6/5 frozen)
- Duplicate-t regression: 1 case (50 rows, 20 identical t-values, all 4 fits return finite params)
- Slow-callback warning: 1 case (monkeypatch `time.perf_counter`, assert `custom_fit slow:` log line)
- Exception wrapper parameterized over 6 raise points: `fit_pl`/`fit_qr`/`fit_bm_floor`/`fit_exp`/`_compute_weights`/`build_fit_input`
- Static lint: 1 case (`grep -L "except Exception" callbacks/custom_time.py` fails the test if wrapper removed)

**B. `btc_web/test_custom_time_integration.py` — direct callback invocation (~17 cases, <3 sec)**
NOT `dash_duo` (Dash 4.0.0 has no in-process headless harness). Mirrors existing `conftest.py:101 _CallbackCtx` + `_patch_ctx()` pattern: imports callback functions directly, invokes with canned Input/State values.
- Activate → `(custom_fig, status, no_update)`.
- **Deactivate (Case O double-assert)** — custom callback returns `(no_update, "Standard view restored.", tick+1)` AND `update_bubble` invoked with incremented tick + `custom_active=False` State produces the standard figure.
- State-vs-Input guard: `update_bubble` does NOT fire on `cta-active` flip alone.
- No-models `PreventUpdate`.
- Each weighting mode produces different params.
- Each of 6 calendar presets produces a valid fit.
- Custom picker None → `PreventUpdate`.
- Post-2016 bypass → status + `no_update` figure.
- Block mode toggle shows/hides wraps.
- `_BLOCKS = None` monkeypatch → block mode degrades, calendar works.
- Top-level exception wrapper: monkeypatch `fit_pl` to raise → status shows `"⚠ Internal error: ValueError"`, figure preserved.

**C. `btc_web/test_custom_time_snapshot.py` — snapshot roundtrip (~9 cases, <1 sec)**
All 8 control IDs present in `_SNAPSHOT_CONTROLS` + `_TAB_CONTROLS["bubble"]` (leverages existing `test_snapshot.py:241-246` collection-time guard). `cta-active` bitmask, `cta-models` 16-combo roundtrip, future-version snapshot unknown-weighting forward-compat, missing-fields defaults, `tab_filter="bubble"` inclusion, URL length sanity, `cta-t0-cal-custom` always serializes, **bitmask-order freeze** (literal equality assertion on `_CHECKLIST_OPTIONS["cta-models"]`).

**D. `btc_web/test_custom_time_e2e.py` — Playwright + Firefox (~8 cases, ~60 sec)**
Mirrors `test_plot_appearance_e2e.py` (screenshot-diff) and `test_scanner_e2e.py` (activate + status-text). All 8 original cases.
**Share-link roundtrip:** encode URL in Python via `_encode_snapshot(...)`, then `page.goto(BASE_URL + "/1#q3:" + encoded)`. Do NOT click Share button. Document the pattern.

**E. `btc_web/test_block_map_cli.py` — `build_block_map.py` tests (~7 cases, ~2 sec, all with mocked `_rpc`)**
Running-max on cherry-picked non-monotonic 2010 pair, atomic temp-file + rename, `--append` fills gap, `--verify` catches corruption, auth resolution order, alignment hard-fail assertion, `quantoshi-health` asserts `block_map_loaded: false` triggers non-zero exit.

**Prerequisite for (E1):** one-off script `tools/find_nonmonotonic_blocks.py` walks `getblockheader` for blocks 30000–80000 and prints the first non-monotonic pair. Output hardcoded into test fixture. Dev-only, run once against local bitcoind, results committed.

### Regression baseline (`btc_web/test_custom_time_baseline.py`)
Python dict, not JSON:
```python
BASELINE = {
    ("pl", "none", "whitepaper"): {"slope": 5.73, "intercept": -38.24, "r2": 0.9834},
    ...  # 4 models × 4 weightings × 6 presets = 96 entries
}
```
Test iterates `BASELINE` and asserts current fits match within `pytest.approx(rel=1e-4)`. Rationale: diffs review well in PRs; no parser layer; direct import.

### Test running
- Unit (A+C+E) ~7 sec on every commit via standard pytest.
- Integration (B) ~3 sec in standard collection.
- E2E (D) ~60 sec behind `--ignore-glob='*_e2e.py'`. Manual pre-deploy.
- **Total new: ~10 sec in main suite, ~60 sec in E2E.**

### Deferred / manual
- **Case P systemd `StartLimitBurst`** — manual ops verification via `systemctl status quantoshi`.
- **Live bitcoind RPC** — always mocked; real verification is `--verify`.
- Cross-browser E2E beyond Firefox — matches existing convention.
- Scan mode tests — deferred with the feature.

**Total test count: ~75.**

---

## Implementation sequencing (for writing-plans skill)

1. **Prerequisite discovery** — run `tools/find_nonmonotonic_blocks.py` against local bitcoind to identify the fixture pair. Commit output.
2. **`tools/build_block_map.py`** + tests (E). Generate `BitcoinBlocksDaily.csv`. Commit the CSV.
3. **`btc_web/_custom_time_presets.py`** — constants only.
4. **`btc_web/engines/custom_fit.py`** + tests (A) via TDD.
5. **`btc_web/layout/custom_time.py`** — panel UI.
6. **`btc_web/callbacks/custom_time.py`** — server callback + Store router.
7. **Snapshot + routing** — register 8 IDs in `snapshot.py` and `routing.py` atomically.
8. **Modify `btc_web/callbacks/charts.py` `update_bubble`** — add `bub-redraw-tick` Input + `State("cta-active", "value")` + `PreventUpdate` guard.
9. **Integrate panel into `layout/bubble.py`** — insert below display_models panel.
10. **Extend `/health` route** + `scripts/quantoshi-health`.
11. **Modify `btc-web.service`** — add `StartLimitIntervalSec=300 StartLimitBurst=5`.
12. **Integration tests (B)** after callbacks wire up.
13. **Baseline (`test_custom_time_baseline.py`)** — run fits, record, commit.
14. **Snapshot tests (C)** post-registration.
15. **E2E tests (D)** post-full-integration.
16. **Extend `update_prices.py`** — dev-only block map append step.
17. **Deploy** — push, pull, flush Redis, restart quantoshi.
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
