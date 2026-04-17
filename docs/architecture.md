# Quantoshi Architecture Guide

Developer-facing reference for the Quantoshi web app codebase. Covers system
design, module responsibilities, model math, key subsystems, and internal
patterns.

---

## 1. System Overview

Quantoshi is a Bitcoin price projection toolkit with three components:

```
BitcoinPricesDaily.csv
        │
        ▼
  ┌─────────────┐    model_data.pkl    ┌──────────────┐
  │  SP.ipynb    │ ──────────────────►  │  btc_web/    │  (Plotly Dash, 9 tabs)
  │  (notebook)  │                      └──────────────┘
  │              │    model_data.pkl    ┌──────────────┐
  │  Cell 0: BM  │ ──────────────────►  │  btc_app/    │  (PyQt5 desktop, 5 tabs)
  │  Cell 1: QR  │                      └──────────────┘
  │  Cell 3: pkl │
  └─────────────┘
```

**Data flow**: Daily CSV prices feed the notebook. Cell 0 fits the bubble model,
Cell 1 runs quantile regression at each percentile, Cell 3 serializes everything
into `btc_app/model_data.pkl`. Both the web app and desktop app load this pkl at
startup.

**Runtime**: pkl load → Dash app init → figure builders generate Plotly
charts on demand → browser renders interactive graphs. All user state lives
in browser `localStorage` — nothing is stored server-side.

---

## 2. Web App Module Map

### Import chain

```
app.py
  ├── populates _app_ctx (M, app, server, flags)
  ├── imports cache.py        (three-layer figure cache + Redis)
  ├── imports tab_defaults.py (per-tab default param dicts)
  ├── imports utils.py        (float quantization, price fetching)
  ├── imports snapshot.py     (state encoding/decoding)
  ├── imports layout.py       (all tab layouts)
  ├── imports callbacks.py    (all callback registrations)
  └── imports api.py          (REST API routes)

figures.py
  ├── imports mc_overlay.py  (MC simulation + trace builders)
  │     ├── imports mc_cache.py   (pre-computed cache)
  │     ├── imports btc_core.py   (ModelData, qr_price)
  │     └── imports markov        (Cython engine, optional)
  └── imports _app_ctx.py    (shared constants)

engines/adapter.py
  └── imports engines/citadel.py  (Citadel simulation engine)
```

### Module responsibilities

| Module | Purpose | Key exports |
|--------|---------|-------------|
| `app.py` | Orchestrator: app creation, model load, Flask routes, cache prewarm | `app`, `server` |
| `_app_ctx.py` | Shared state and constants (models, palettes, flags) | `M`, `app`, `FREQ_PPY`, `PALETTES`, `PRICE_MODELS`, `_compute_sc_loan()` |
| `cache.py` | Three-layer figure cache (L0 pinned, L1 LRU, L2 Redis) | `get_figure()`, `invalidate()` |
| `tab_defaults.py` | Per-tab default parameter dicts, used by prewarm and coercion | `TAB_DEFAULTS` |
| `utils.py` | Float quantization, price fetching | `_q3()`, `_fetch_btc_price()` |
| `snapshot.py` | Snapshot encoding/decoding, bitmask helpers | `_encode_snapshot()`, `_decode_snapshot()`, `_SNAPSHOT_CONTROLS` |
| `layout/` | Layout package (13 modules): `__init__` (navbar, modal, stores), `bubble`, `heatmap` (pill bar), `sim_tabs` (DCA+Retire), `supercharge`, `stack`, `faq`, `common` (shared helpers), `mc_controls`, `splash`, `model_info`, `citadel` (+1 TBD) | `main_layout()` |
| `callbacks/` | Callbacks package (17 modules): `__init__`, `charts`, `nav` (tab routing, pill clicks), `ticker`, `snapshot_cb`, `mc_controls`, `mc_helpers`, `mc_payment`, `mc_upload`, `lots`, `coerce` (`_ci()`/`_cf()`), `sc_loan`, `routing`, `splash`, `user_model`, `citadel_cb`, `scanner` | `update_bubble()`, `update_heatmap()`, etc. |
| `figures/` | Figures package (8 modules): `__init__`, `common` (palette, watermark, annotations, MC overlay), `bubble` (+ PL/S2F overlays), `heatmap`, `dca`, `retire`, `supercharge`, `citadel` | `build_bubble_figure()`, `build_heatmap_figure()`, etc. |
| `engines/adapter.py` | Simulation engine adapter — routes to QR, MC, or Citadel engine | — |
| `engines/citadel.py` | Citadel planning simulation engine | `SimConfig`, `CitadelState`, `simulate()` |
| `engines/tax.py` | Annual tax computation (brackets, NIIT, loss netting) | `compute_annual_tax()`, `TaxYearAccumulator` |
| `engines/tax_lots.py` | Lot-level BTC tracking for capital gains | `TaxLot`, `sell_lots()`, `seed_lots()` |
| `engines/tax_data.py` | Static US tax data (brackets, state rates, RMD factors) | `FEDERAL_BRACKETS_TCJA`, `STATE_TAX_RATES`, `RMD_FACTORS` |
| `mc_overlay.py` | MC simulation, caching, fan band traces, regime filters | `_mc_dca_overlay()`, `_mc_retire_overlay()`, etc. |
| `mc_cache.py` | Pre-computed MC cache generation/loading/lookup | `load_caches()`, `get_cached_paths()`, `get_cached_overlay()` |
| `load_shm_cache.py` | Shared memory cache loading | — |
| `api.py` | REST API endpoints | `register_routes()` |
| `btcpay.py` | BTCPay Server payment integration | Invoice lifecycle |
| `btc_core.py` | ModelData class, QR pricing math, lot percentiles | `ModelData`, `qr_price()`, `yr_to_t()` |
| `test_web.py` | Tests: utilities, builders, snapshots, callbacks, btcpay, regime filters, tax (~650+ passing) | — |
| `test_tax_e2e.py` | Playwright E2E smoke tests for tax UI (15 tests, requires dev server + Firefox) | — |

---

## 3. Tab Architecture

### 9 tabs

| # | Tab | ID | Chart builder | MC overlay | Key controls |
|---|-----|----|---------------|------------|--------------|
| 1 | Bubble + QR Overlay | `bubble` | `build_bubble_figure()` | None | Quantiles, axes, N future bubbles, stack |
| 2 | CAGR Heatmap | `heatmap` | `build_heatmap_figure()` | `_mc_heatmap_overlay()` | Entry yr/percentile, color mode, multi-model pill bar (Bubble/PL/S2F/MC) |
| 3 | BTC Accumulator | `dca` | `build_dca_figure()` | `_mc_dca_overlay()` | DCA amount, freq, Stack-celerator |
| 4 | BTC RetireMentator | `retire` | `build_retire_figure()` | `_mc_retire_overlay()` | Withdrawal, inflation, depletion arrows |
| 5 | HODL Supercharger | `supercharge` | `build_supercharge_figure()` | `_mc_supercharge_overlay()` | Mode A/B, delays, depletion bands |
| 6 | Stack Tracker | `stack` | None (DataTable) | None | Lot CRUD, import/export |
| 7 | Citadel Planner | `citadel` | `build_citadel_figure()` | None | Citadel simulation engine, multi-scenario planning |
| 8 | Model Info | `model_info` | None | None | Accordion, deep-linkable (`/8.N`) |
| 9 | FAQ | `faq` | None | None | Accordion, 20 entries, deep-linkable (`/9.N`) |

### Control panel structure (tabs 2–5)

Each MC-enabled tab follows a consistent layout pattern:

```
┌─ Tab Hints ────────────────────────────────────┐
│  Collapsible "How to use this tab" bullets      │
├─ Shared Model Settings ────────────────────────┤
│  Stack (BTC), Use lots, Amount*, Freq†, Infl   │
├─ Quantile Regression Model ────────────────────┤
│  "Select quantiles to follow"                   │
│  Quantile checklist grid                        │
├─ Monte Carlo Simulation ──────────────────────-┤
│  Activate, Start yr, Entry Q, Years, Bins      │
│  ▶ Advanced: sims, window, regime filter       │
│  [Run Simulation] [Save] [Load]                │
├─ Chart Settings ───────────────────────────────┤
│  Display Models [✓QR] [✓MC]                    │
│  Year range, Display mode, Toggles, Legend pos │
└────────────────────────────────────────────────┘
```

*Amount: DCA/Ret only (SC withdrawal stays in Plan section)
†Freq: locked to Monthly by default; unlock checkbox + warning modal

**Shared settings**: Stack, amount, frequency, and inflation are shared between
QR and MC on the same tab. HM is an exception — only stack is shared; HM retains
its own mc-amount, mc-freq, mc-infl since QR heatmap doesn't use those parameters.

### Price models & Display Models

Seven or more price models registered at startup in `_app_ctx.PRICE_MODELS`:
- **Bubble Model** (`"bub"`) — default, loaded from `model_data.pkl`
- **Power Law** (`"pl"`) — OLS fit to log-log data
- **S2F (Stock-to-Flow)** (`"s2f"`) — alternative parameterization
- **BM Empirical Floor** (`"ef"`) — steeper support (slope 5.31) with Gaussian composite bands, loaded from `model_data_ef.pkl` (conditional — only if pkl exists)
- **Quantile Regression** (`"qr"`) — direct QR power law channels (standalone display model)
- **LPPL** (`"lppl"`) — Log-Periodic Power Law model
- **Exponential** (`"exp"`) — exponential trend fit
- **U₁** (`"u1"`) — additional alternative parameterization

Per-tab model display:
- **Bubble tab**: `bub-model-show` checklist toggles PL + S2F overlays on the bubble chart.
- **Heatmap tab**: pill bar carousel (`hm-active-model` Store) — one model active at a time. `hm-model-show` checklist exists in layout for snapshot compat but is hidden, replaced by the pill bar. Pill buttons are built dynamically from `PRICE_MODELS` + optional MC.
- **DCA / Retire / Supercharger**: `{prefix}-model-show` checklist showing QR, MC (if available), PL, S2F.

When `_HAS_MARKOV` is `False`, "MC Simulation" is hidden from all Display Models checklists.

---

## 4. Quantile Regression Model

### What it does

Quantile regression (QR) fits a power law to Bitcoin's historical price data at
each percentile level. Unlike OLS (which fits the mean), QR fits arbitrary
quantiles — Q10% captures the 10th percentile price path, Q50% the median, etc.

### Math

All fitting happens in log-log space:

```
log10(price) = intercept + slope * log10(t)
```

where `t = (date - genesis).days / 365.25` (years since the optimal time
origin, July 25, 2009 — the statistically optimal start date for the power law fit).

Inverting:

```
price(q, t) = 10^(intercept_q + slope_q * log10(t))
```

This is a straight line in log-log space, which appears as a power law curve in
linear space.

### Data structures

```python
qr_fits = {
    0.001: {"intercept": float, "slope": float, "r2": float},
    0.01:  {"intercept": float, "slope": float, "r2": float},
    ...
    0.999: {"intercept": float, "slope": float, "r2": float},
}
```

Each key is a quantile (0.001 = Q0.1%, 0.50 = Q50%, etc.). The fitting uses
`statsmodels.QuantReg` on log-transformed data via `fit_qr_from_csv()`.

### Interpolation for arbitrary percentiles

`_interp_qr_price(q, t, qr_fits)` in `figures/common.py` handles non-standard
percentiles (e.g., Q7.5%) by interpolating in log-price space between the two
nearest fitted quantiles.

### How quantiles appear on charts

Each quantile produces a price channel line. Lower quantiles (Q1%, Q5%) represent
pessimistic scenarios; higher quantiles (Q85%, Q99%) represent optimistic ones.
Colors are assigned per-quantile from `ModelData.qr_colors`. When "shade" is
enabled, the area between adjacent selected quantiles is filled with translucent
color at `_SHADE_ALPHA = 0.08`.

---

## 5. Bubble Model

### Composite construction

Cell 0 of the notebook fits a parameterized bubble shape to each historical
Bitcoin bubble. The model identifies bubble peaks, fits amplitude/width/skewness
parameters, and constructs a composite curve.

Key arrays in `ModelData`:
- `years_plot_bm` — x-axis (calendar years) for the bubble model
- `support_bm` — long-term support line (bubble floor)
- `comp_by_n` — list of composite curves for N=1..`n_future_max` future bubbles
- `bm_r2` — bubble model R-squared

### N future bubbles

The "N future bubbles" control extrapolates the bubble pattern forward. Each
value of N adds one more projected bubble cycle. `comp_by_n[n-1]` gives the
composite curve assuming `n` future bubbles.

---

## 6. Markov MC Engine

### Overview

The Monte Carlo (MC) simulation uses a Markov chain model trained on historical
Bitcoin price transitions to generate forward price paths.

### Transition matrix

`build_transition_matrix(prices, n_bins, step_days, window)` (in the Cython
`markov` module):

1. Discretizes log-prices into `n_bins` bins (default 5: Bargain, Cheap, Fair,
   Pricey, Bubble)
2. For each consecutive `step_days` interval, records the bin-to-bin transition
3. Normalizes rows to get transition probabilities

The training window defaults to 2010–present (`MC_WINDOW_START = 2010`).

### Forward simulation

`monte_carlo_prices(trans, bin_edges, start_bin, n_steps, n_sims)`:

1. Starts all `n_sims` paths in `start_bin`
2. At each step, samples the next bin from the transition probability row
3. Converts bin indices back to log-prices (uniform within bin)
4. Returns `(n_sims, n_steps)` price array

### Regime filter

Blocked bins zero out their columns in the transition matrix via
`_apply_bin_mask(trans, blocked_bins)`. This removes certain price regimes from
the simulation (e.g., blocking the "Bubble" bin prevents extreme bull scenarios).
Ghost overlay compares filtered vs unfiltered results.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MC_BINS` | 5 | Price regime bins |
| `MC_SIMS` | 800 | Simulations per path |
| `MC_FREQ` | Monthly | Default frequency |
| `MC_STEP_DAYS` | 30 | Price data sampling interval |
| `MC_WINDOW_START` | 2010 | Training window start |

---

## 7. MC Overlay Pipeline

### 3-level cache fallthrough

```
1. Client-side cache (browser dcc.Store)
   │ miss
   ▼
2. Pre-computed server cache (npz files / /dev/shm pickle)
   │ miss
   ▼
3. Live Markov simulation (Cython engine)
```

Level 1 avoids server round-trips for repeated parameter combinations. Level 2
provides near-instant results for common parameter grids (~834 MB cache). Level 3
runs the full simulation when no cache hit (requires `markov` module).

### Cache keys

Two separate key types control cache behavior:

- **`_mc_path_key(p, tab)`**: Identifies the expensive MC price path simulation.
  Components: `mc_start_yr`, `mc_entry_q`, `mc_years`, `n_bins`, `n_sims`,
  `mc_freq`, `mc_window`, `mc_blocked_bins`. Changing frequency triggers a
  full re-simulation (expensive).

- **`_mc_overlay_key(p, tab, start_stack)`**: Identifies the post-simulation
  overlay (DCA accumulation, withdrawal depletion, etc.). Adds `amount`,
  `inflation`, `start_stack` to the path key. These are cheap to recompute.

### Fan percentiles

`FAN_PCTS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95)` — six percentile bands
computed across all simulated paths. Traces are built by `_mc_build_traces()`
with filled regions between bands.

### 5 overlay functions

| Function | Tab | Description |
|----------|-----|-------------|
| `_mc_dca_overlay()` | DCA | Simulates periodic BTC purchases across MC paths |
| `_mc_retire_overlay()` | Retire | Simulates withdrawals with inflation across MC paths |
| `_mc_supercharge_overlay()` (mode A) | SC | Depletion curves across MC paths |
| `_mc_supercharge_overlay()` (mode B) | SC | Binary search for max withdrawal |
| `_mc_heatmap_overlay()` | Heatmap | MC-simulated CAGR percentiles |

### Pre-computed cache structure

```
mc_cache/
    paths_YYYY.npz       — price paths per (entry_pct_bin, mc_years)
    overlays_YYYY.npz    — fan bands per (entry_pct_bin, mc_years, wd, infl, stack)
```

Cached start years: 2026, 2028, 2031, 2035, 2040. Entry percentile bins: 10%
increments (0.1–0.9). Duration options: 10, 20, 30, 40 years.

### Fast restart via /dev/shm

After the first full npz load (~7s), the entire cache is pickled to
`/dev/shm/quantoshi_mc.pkl` (~834 MB). Subsequent worker restarts load in ~0.7s.
A fingerprint (npz mtime + total size) validates freshness.

### Chart finalization (`_finalize_chart()` in figures/common.py)

All chart builders (DCA, Retire, SC Modes A/B) share a common finalization
sequence extracted into `_finalize_chart(traces, layout, p, tab, mc_result,
mc_premium)`:

1. Apply legend position from `p["legend_pos"]`
2. Apply sans typography (`_apply_sans_typography`)
3. Create `go.Figure`
4. Apply MC premium styling (if enabled and `mc_premium=True`)
5. Apply config annotation (`_apply_config_annotation`)
6. Apply watermark (position opposite to legend)
7. Return `(fig, mc_result)`

### Free tier

| Parameter | Free | Paid |
|-----------|------|------|
| Simulations | 100 | 800 |
| Start years | 2028, 2031 | Any cached |
| Entry percentile | 10% | Any |
| Duration | 10, 20 yr | 10–40 yr |

---

## 8. Chart Annotation System

### Edge text traces

All endpoint annotations use `_edge_text_trace()` — a `go.Scatter(mode=
"markers+text")` placed at the last data point of a trace. This avoids
paper-space `xref`/`yref` annotations which drift on resize/zoom.

Depletion annotations (arrows to y=0 with `yref="paper"`) are the sole
exception — they point to the plot bottom which is always correct.

### Overlap resolution

`_resolve_edge_annotations(pending, log_y)` prevents overlapping labels:

1. **Collect**: each builder gathers pending annotations as dicts with
   `x_arr`, `y_arr`, `label`, `short_label`, `color`, `y_last`
2. **Sort**: by `y_last` ascending (log-space aware)
3. **Cluster**: group consecutive items within `_OVERLAP_LOG = 0.12`
   log-decades (or `_OVERLAP_FRAC = 0.06` of linear range)
4. **Resolve**:
   - **1 item**: rank-based position (lower half → `"bottom left"`,
     upper half → `"top left"`) to prevent visual crossing
   - **2–3 items**: spread within cluster (bottom/middle/top)
   - **4+ items**: consolidate into single merged label using
     `short_label` values joined with ` · `, dot markers at each position

### Short label format

`_fmt_short(btc, usd)` → `B0.32/$1.23M` — compact BTC/USD format used
in consolidated annotations. USD uses K/M/B suffixes.

### Annotate toggle

"Annotate final values" checkbox (in Chart Settings) controls edge text
trace visibility. Depletion arrows always display regardless of this toggle.
DCA, Retire, and SC tabs all support this toggle.

---

## 9. Snapshot / Share System

### Control inventory

~206 `_SNAPSHOT_CONTROLS` entries — `(component_id, property)` tuples covering
all UI controls across 9 tabs (Model Info has no snapshot controls).

### Encoding pipeline

```
Control states → JSON array (~206 elements) → gzip → base64 urlsafe → URL hash
```

URL format: `host/N#q3:ENCODED` where `N` is the tab path (1–9).

### Versioning

| Prefix | Format | Status |
|--------|--------|--------|
| `q3:` | Positional array, shared settings controls | Current |
| `q2:` | Positional array, pre-shared-settings | Decoded (positions may mismatch) |
| `q1:` | Dict-based | Decoded (legacy) |

### Bitmask encoding

28 checklist fields use bitmask encoding for compact URLs:
- 5 quantile checklists: up to 17-bit each (17 possible quantiles)
- 23 toggle checklists: 1–7 bits each

`_list_to_mask(val, opts)` encodes, `_mask_to_list(mask, opts)` decodes. Old
links stored plain lists — decoder handles both via `isinstance(val, int)`.

### Tab-scoped snapshots

`_TAB_CONTROLS` maps each `tab_id` to its set of component IDs.
`_encode_snapshot(state_dict, tab_filter=controls)` encodes only the active tab's
controls as non-null; others default to `null` and fall back to defaults on
restore. This produces much shorter URLs for single-tab shares.

---

## 10. Callback Architecture

### Per-tab render trigger pattern

The app uses a "first-render trigger" pattern for lazy tab rendering:

1. Each chart tab has a `dcc.Store("{tab}-first-render")` initialized to 0
2. A single clientside callback watches `main-tabs.active_tab` and increments
   the matching tab's store
3. Chart callbacks use `Input("{tab}-first-render", "data")` instead of
   `Input("main-tabs", "active_tab")`
4. All chart callbacks have `prevent_initial_call=True` — they ONLY fire when
   their trigger increments
5. Result: switching tabs fires exactly 1 callback (the active tab), not 6

**URL-based initial tab + pre-injected figures**: The layout is a function
(`_serve_layout`) that reads `flask.request.path` to determine the initial
`active_tab`. It also pre-builds ALL tab figures from the L1 LRU cache
(populated by prewarm) and injects them directly into the initial HTML. All
`{tab}-first-render` stores start at `1`. Visiting `/9` builds the layout with
`active_tab="citadel"` — the bubble callback never fires. Switching tabs
requires **zero server round-trips** — figures are already present in the DOM;
callbacks only fire when the user changes a control.

**`tab_resize.js`**: Forces `Plotly.Plots.resize()` when a tab becomes
visible. Required because hidden tabs render at zero/wrong size when the
browser has not painted them yet.

**`tab_dblclick.js`**: Double-clicking a tab header increments its
`{tab}-first-render` store, triggering a full figure reload from the server.
Escape hatch for stale-looking figures.

**Background callbacks (Citadel)**: The Citadel "Run Simulation" button uses
Dash's `background=True` with `DiskcacheManager`. The simulation runs in a
separate process and does not block gunicorn workers. Button shows
"⏳ Computing..." during long MC runs. Requires `diskcache`, `psutil`, `dill`,
`multiprocess` in the environment.

### `prevent_initial_call` settings

| Callback | Setting | Why |
|---|---|---|
| All chart callbacks (bubble, heatmap, DCA, retire, SC, citadel) | `True` | Only fire via first-render trigger on tab visit |
| MC body toggles (5x) | N/A (clientside) | Zero server round-trips |
| SC mode/display toggles | N/A (clientside) | Same |
| Price ticker | `'initial_duplicate'` | Must fire once on load to populate price |

Constraint: `allow_duplicate=True` on outputs is incompatible with
`prevent_initial_call=False` (crashes gunicorn). The first-render trigger
pattern solves this — callbacks use `prevent_initial_call=True` and get
triggered by the clientside store instead.

### Callback inventory

| Type | Count | Description |
|------|-------|-------------|
| Server callbacks | ~20 | Tab updates, ticker, share modal, MC controls |
| Clientside callbacks | ~30 | Tab routing, zoom toggle, UI visibility, first-render triggers |
| Loop-created callbacks | ~16 | MC toggle, advanced toggle, regime opts, freq unlock |

### Clientside callback pattern

Trivial visibility toggles should be clientside callbacks (no server
round-trip):

```python
_app_ctx.app.clientside_callback(
    "function(v) { return (v && v.length) ? {} : {display:'none'}; }",
    Output("component-id", "style"),
    Input("toggle-id", "value"),
)
```

### Type coercion helpers

`_ci(val, default, lo, hi)` and `_cf(val, default, lo, hi)` coerce callback
inputs to `int`/`float`. They use `is not None` (not `or`) so that `0` is
treated as a valid value. Optional `lo`/`hi` clamp the result. All numeric
coercion sites in callbacks use these helpers.

### Tab update pattern

The four chart-with-MC tabs (DCA, Retire, SC, Heatmap) follow a shared update
pattern:

1. Guard: if tab not active → `PreventUpdate`
2. Coerce inputs via `_ci()`/`_cf()`, set toggle/range defaults
3. MC setup: `_mc_setup()` → payment check, free tier, build params, ghost match
4. Build tab-specific params dict: map raw inputs to figure builder kwargs
5. Call figure builder (returns `(fig, mc_result)`)
6. MC finalize: `_mc_finalize()` → strip paths, rendered key, status, zoom
7. Return: figure + 4–5 ancillary outputs (mc status, result store, etc.)

### `_mc_setup()` and `_mc_finalize()` (callbacks.py)

These two helpers extract the shared MC boilerplate from DCA/Retire/SC
callbacks (steps 3–8 and 11–15 of the original 16-step pattern):

- **`_mc_setup(tab, ...)`** → `(mc_ok, is_free, mc_p, blocked)` — wraps
  `_mc_payment_check()`, `_is_free_tier()`, `_build_mc_params()`, free tier
  cache override, and ghost match.

- **`_mc_finalize(tab, fig, ...)`** → `(fig, store_val, status, rendered_key,
  show_modal, ub_val)` — wraps `_strip_free_paths()`, rendered key
  construction, `_mc_status()`, `_unblocked_val()`, and chart zoom toggle.

Heatmap still uses inline MC handling (its dual-panel pattern differs
significantly from the other three tabs).

### Shared settings flow

DCA/Ret/SC: shared controls (`{prefix}-stack`, `{prefix}-amount`,
`{prefix}-freq`, `{prefix}-infl`) feed both QR and MC models. The callback
passes these values to `_build_mc_params()` as `mc_amount`, `mc_freq`, etc.

HM: only `hm-stack` is shared. HM retains independent `hm-mc-amount`,
`hm-mc-freq`, `hm-mc-infl` controls.

### Frequency lock UX

Frequency is locked to Monthly by default via a disabled dropdown.
`{prefix}-freq-unlock` checkbox enables editing. On unlock, a shared
`freq-warning-modal` explains that changing frequency affects MC simulation
cost. On uncheck, frequency resets to Monthly.

### `_build_mc_params()` (callbacks.py)

Centralized MC parameter assembly for all 4 tabs. Takes raw MC control values
and returns a standardized dict consumed by MC overlay functions. Called
internally by `_mc_setup()` — tab callbacks don't call it directly.

### Clientside callbacks

30 clientside callbacks handle fast UI interactions without server round-trips:
- Tab routing (`/1`–`/9` → tab switch)
- Zoom toggle (dragmode enable/disable)
- MC control visibility
- SC mode A/B panel switching
- Model Info deep-linking (`/8.N`)
- FAQ deep-linking (`/9.N`)

---

## 11. LRU Figure Cache

### Architecture

`@lru_cache(maxsize=8)` per tab (bubble, heatmap, DCA, retire, supercharge)
in `utils.py`. Cache key is a frozen tuple of all quantized params.

### Float quantization

`_q3(x)` rounds floats to 3 significant figures for cache-friendly keys.
Scales naturally across BTC's price range ($0.06 → $0.06, $95,437 → $95,400).

`_quantize_params(p)` applies `_q3` to all float params but **exempts
`selected_qs` and `exit_qs`** (must match `qr_fits` keys exactly).

### Cache warming

`_prewarm_caches()` runs at worker startup, pre-building figures for default
parameters across all tabs. Bubble cache key includes `date.today()` for
natural daily TTL.

**Cache key alignment**: The `*_defaults()` functions in `tab_defaults.py` must
include ALL keys that callbacks add to the params dict (including `show_qr`,
`show_mc`, `palette`, `user_model`, `sc_live_price`). This ensures the prewarm
cache key matches the runtime callback cache key, yielding an L1 cache hit on
first tab visit. A mismatch means the prewarmed entry sits unused while the
first real request rebuilds the figure from scratch.

### Three-layer figure cache (`cache.py`)

Figure caching has three layers with fallthrough:

```
L0: Pinned cache — always-warm entries for default params (never evicted)
    │ miss
    ▼
L1: LRU cache — @lru_cache(maxsize=8) per tab, in-process (fast)
    │ miss
    ▼
L2: Redis cache — cross-worker shared cache, serialized figures
```

`cache.py` provides `get_figure()` and `invalidate()`. Tab defaults live in
`tab_defaults.py` (`TAB_DEFAULTS` dict) and are consumed by both the prewarm
logic and the coercion helpers in `callbacks/coerce.py`.

---

## 11b. User Model Feature

The User Model feature allows users to supply custom price model parameters
(intercept, slope, or full QR fits) to overlay a personalized price path on
charts. `callbacks/user_model.py` handles upload, validation, and storage.
User model state is kept in a `dcc.Store` (memory) and wired into the figure
builders via `_app_ctx`. The `layout/model_info.py` module exposes the upload
UI.

---

## 11c. Citadel Simulation Engine

The Citadel Planner (tab 9) uses a dedicated simulation engine in
`engines/citadel.py`. It models multi-asset retirement with BTC + cash +
treasuries + equities + bonds. `engines/adapter.py` provides a unified
interface. `figures/citadel.py` builds the chart; `callbacks/citadel_cb.py`
wires the Dash callbacks.

### Tax System (opt-in)

Master toggle `cp-tax-toggle` activates US federal + state tax simulation.
When off, the engine runs unmodified (zero tax drag).

**Three account wrappers** — each can hold BTC, cash, reserves, investments:
- **Taxable**: BTC sales → lot-level capital gains (ST/LT via `sell_lots()`).
  Investment sales → LT gains with dynamic cost basis tracking. Cash/reserve
  withdrawals → no tax. Interest → ordinary income. Treasury interest →
  state-exempt.
- **Tax-Deferred** (Traditional IRA/401k): All withdrawals → ordinary income.
  Subject to RMDs (age 73 for born 1951-1959, 75 for born 1960+).
- **Tax-Free** (Roth): All withdrawals → tax-free. No RMDs.

**Annual tax pipeline** (at year boundary in `step()`):
1. IRS §1(h) loss netting → AGI → standard deduction split → ordinary brackets
   → LTCG stacking → NIIT (3.8%) → state tax → total
2. Tax paid from taxable wrapper via gross-up formula
3. Brackets inflation-indexed from 2025 base. NIIT threshold NOT indexed.
4. TCJA sunset toggle reverts to 39.6% top rate + lower standard deduction.

**Growth-aware withdrawal ordering**: Engine consults BTC price model's
forward-looking growth rate each period. High BTC growth → avoid selling BTC.
Low BTC growth (late decades) → BTC moves earlier in withdrawal order. Roth BTC
is always last (tax-free compounding on highest-growth asset).

**Modules**: `engines/tax.py` (computation), `engines/tax_lots.py` (lot tracking),
`engines/tax_data.py` (static data), `layout/citadel_tax.py` (full-screen modal),
`callbacks/citadel_tax_cb.py` (modal callbacks + summary table builder).

### MC engine performance

Two optimizations that drop first-render cost significantly on MC
Citadel runs; remember these when editing the inner loop.

**In-place state mutation in `citadel_step.step()`** (`engines/citadel_step.py`).
The step function previously ran `deepcopy(state)` at the top of every
period — ~95% of `step()`'s cost for 40-year Monthly × 1000-sim runs
(~480k copies, each growing as `tax_lots` accumulated). The copy was
defensive; `simulate()` snapshots scalars via `_snapshot_state()` and
never reads the prior state again, so it's safe to mutate the passed-in
state in place and return the same object. The snapshot helper
explicitly `list()`-copies `reserves` / `investments`; `rebal_event`
is reassigned to fresh dicts each trigger (`citadel_rebalancing.py`).
**Do not** reintroduce a deepcopy here without a real correctness bug
— it masks a 3–10× slowdown.

**Pre-built quantile grid in `_ModelAdapter.prebuild_grid()`**
(`figures/citadel.py`). The adapter serves `quantile_at(price, t)` via
a `(n_quantiles, n_t)` price grid. Before the sim loop runs we
vectorize-evaluate the model once across the sim's full monthly time
axis, populating the internal per-t cache in one shot. This moves
~14 400 price evaluations (30 quantiles × 480 periods for a 40-yr
Monthly sim) out of the 1000-sim critical path. MC Citadel first-render
is ~200–800 ms faster with no behavioural change.

---

## 12. Live Price Ticker

- `dcc.Interval(id="price-interval", interval=20*60*1000)` fires every 20 min
- Primary: Binance (`api.binance.com/api/v3/ticker/price?symbol=BTCUSDT`)
- Fallback: CoinGecko (for US geo-blocked users)
- Outputs: navbar `price-ticker` div, `btc-price-store`, heatmap `hm-entry-q`
- Display: `₿ $X` · `QY.Y%` (current quantile) + 24h sparkline SVG (CoinGecko)
- Sats/$ toggle: switches ticker between USD price and sats-per-dollar display
- Heatmap uses `live_price` as entry price when `entry_yr == current_year`
- Binance is geo-blocked in the US (HTTP 451) but works from the Hetzner server

---

## 13. Layout Patterns

### Style constants — centralized in `colors.py`

**All visual appearance constants live in `btc_web/colors.py` Section 5** — fonts, font sizes, trace widths, point sizes, opacities, margins. Nothing else defines visual constants (enforced by `test_colors_central.py`).

| Category | Constants | Count | Notes |
|----------|-----------|-------|-------|
| Font stacks | `FONT_BRAND`, `FONT_SANS`, `FONT_MONO`, `FONT_CONDENSED` | 4 | DM Serif Display + Inter via Google Fonts |
| Chart font sizes | `CHART_FONT_TITLE` through `CHART_FONT_WATERMARK_LG` | 12 | Int values for Plotly (base + desktop tiers) |
| UI font sizes | `UI_FONT_XS` through `UI_FONT_HEADING` | 8 | CSS px strings for layout inline styles |
| Trace widths | `TRACE_WIDTH` through `DESKTOP_GRID_MULT` | 10 | Includes desktop multipliers |
| Point/marker sizes | `PT_SIZE_DEFAULT` through `MARKER_SIZE_HIGHLIGHT` | 6 | |
| Opacities | `SHADE_ALPHA` through `SCANNER_ROW_OUTLINE_ALPHA` | ~35 | Every `_hex_alpha()` and `opacity=` uses a named constant |
| Quantile opacity | `Q_OPACITY_FLOOR`, `Q_OPACITY_RANGE`, `Q_OPACITY_DECAY` | 3 | Formula in `figures/common.py::quantile_opacity()` |
| Chart margins | `CHART_MARGIN`, `CHART_MARGIN_HM` | 2 | Dicts |
| Watermark | `WM_OPACITY`, `WM_SIZE_X`, `WM_SIZE_Y` | 3 | |

**Generated artifacts** (`tools/generate_color_artifacts.py`):
- `_colors_generated.css` — CSS custom properties `var(--qs-*)`
- `_colors_generated.js` — `window.QS_COLORS` + `window.QS_PALETTES` + `window.QS_APPEARANCE`
- Export control: `__skip_export__` excludes non-color values from CSS; `__appearance_export__` defines JS `QS_APPEARANCE` subset

Layout modules import directly: `from colors import UI_FONT_MD, FONT_BRAND, SUPPORT_LINE_OPACITY, ...`

Module-level style dicts in `layout/common.py`:

| Constant | Value | Used for |
|----------|-------|----------|
| `_STYLE_HIDDEN` | `{"display": "none"}` | Hidden containers, placeholder controls |
| `_STYLE_HINT` | `{"color": DIM_TEXT, ...}` | Hint/instruction text below controls |
| `_STYLE_GRAPH_H` | `{"height": "78vh"}` | Chart graph containers |
| `_GEAR_STYLE` | `{cursor, fontSize, opacity}` | In-checklist gear icons for model config modals |
| `_MUTED_STYLE` | `{color: MUTED_SUMMARY_TEXT, italic}` | Inline summary spans in Display Models |

### Shared helpers

| Helper | Purpose |
|--------|---------|
| `_section_card(title, *children)` | Titled card with consistent styling |
| `_ctrl_card(*children)` | Untitled compact card |
| `_lbl(text)` | Small bold label |
| `_row(*cols)` | Horizontal `dbc.Row` with auto columns |
| `_q_panel(id, default)` | Quantile checklist in a QR Model section card |
| `_shared_settings_card(prefix, ...)` | Stack + amount + freq + inflation panel |
| `_model_show_checklist(prefix)` | Display Models [QR] [MC] checklist |
| `_mc_controls(prefix, ...)` | MC simulation control panel |
| `_year_range_slider(prefix, ...)` | Dual-handle year range slider |
| `_legend_pos_dropdown(prefix, default)` | Legend position selector |
| `_export_row(prefix)` | Chart download buttons + mobile hint |
| `_chart_tab_layout(controls_fn, graph_id, ...)` | Standard 2-column chart tab layout |
| `_tab_hints(tab_id)` | Collapsible "How to use" section |

### Tab hints

6 tabs have hint bullets (set via `_TAB_HINTS` dict). Each MC-enabled tab's
second bullet reads "Configure your Quantile Regression model or Markov
Simulation" and the last bullet references "using the chart configuration tab
below."

### Mobile layout

On `max-width: 767px`, columns stack vertically (controls below chart). The
`dcc.Graph` height is overridden in CSS (`55vw !important`). A "↓ Scroll down
to configure" hint appears via `_export_row()` (hidden on ≥768px).

---

## 13b. Appearance & Brand System

### Typography

Google Fonts loaded via `<link>` in `index_string <head>`:
- **DM Serif Display** — brand name "Quantoshi" + chart titles (left-aligned)
- **Inter** — all UI text, axis labels, legend, form controls

Font stacks defined in `colors.py`:
- `FONT_BRAND` = `"'DM Serif Display', Georgia, serif"`
- `FONT_SANS` = `"Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"`
- `FONT_MONO` = `"'JetBrains Mono', 'Fira Code', ..."`

CSS custom properties (`style.css :root`): `--font-ui`, `--font-brand`, `--font-mono`. These are hand-maintained in CSS and must match `colors.py` if changed.

### 4 site-wide palettes

| Key | Name | Audience |
|-----|------|----------|
| `default` | Default | Full color vision |
| `cb-brian` | Deuteranomaly | Red-green deficient (Brian's profile) |
| `cb-rg` | Colorblind R-G | Classic CB-safe (Okabe-Ito style) |
| `cb-full` | Colorblind Full | Near-monochromatic, luminance-only |

Palette selector appears at the bottom of every chart tab's controls (`_palette_selector(tab_key)`). Per-tab `dbc.Select(id=f"palette-select-{tab_key}")` syncs bidirectionally with `palette-store` via clientside callbacks in `nav.py`. Forward callbacks include a `State("palette-store", "data")` guard to prevent circular callback storms.

### Default palette design

Chart background: `#FAF9F6` (ivory). Grid: warm-tuned `#E2E0DB`/`#F0EEED`. Scatter data: `#1A1A2E` deep ink, size 5, alpha 0.3.

Flagship 6 model trace colors (deltaE-optimized warm/cool dichotomy):

| Model | Color | Hex | Family |
|-------|-------|-----|--------|
| BM | Amber | `#C48209` | Warm |
| PL | Navy | `#1B3352` | Cool |
| QR | Burgundy | `#9B2244` | Warm |
| EPPL | Teal | `#1F6B5C` | Cool |
| HybPPL | Rust | `#A8431C` | Warm |
| LPPL | Purple | `#7B3D9E` | Cool |

Family variants inherit their master color. Non-flagship models use a muted secondary palette.

### Heatmap CAGR presets

4 presets: `rwg` (Red->White->Green), `rbg` (Red->Black->Green), `bwo` (Blue->White->Orange), `mono` (Grayscale). Default palette auto-selects: `default`->rwg, CB palettes->bwo. Auto-select fires on palette switch via clientside callback.

### Display Models config panel

Shared component `layout/display_models.py::display_models_panel(prefix, ...)` used by tabs 1/3/4/5. 4 model families have gear icons opening global config modals: BM (Bubble Model settings), LPPL (n_freqs/weighted/no13), HybPPL (slot A/B frequency/damping), EPPL (slot A/B). Heatmap (tab 2) has a pill bar + status row instead. 15 defunct `*-activate` placeholder checklists kept in `_serve_layout` for snapshot positional stability.

### Quantile rendering

Quantile traces use model-centric coloring (NOT thermal gradient — thermal pipeline removed 2026-04-12):
- Trace lines: model color at `quantile_opacity(q)` — full at median, fading at extremes
- Band fills: model color at `SHADE_ALPHA` (0.08 outer) / 0.15 (inner) via `_build_symmetric_bands`
- Quantile panel dots: `DIM_TEXT` at `quantile_opacity(q)` alpha

---

## 14. Payment Flow

### BTCPay Server integration

`btcpay.py` integrates with BTCPay Server's Greenfield API for payment-gated MC
simulations.

### Flow

1. **Free tier check**: `_is_free_tier()` in `callbacks.py` checks if requested
   parameters fall within free limits
2. **Token check**: If not free, check for valid HMAC token (daily expiry)
3. **Invoice creation**: BTCPay Greenfield API creates a Lightning invoice
4. **Polling**: Client polls for payment confirmation
5. **Token generation**: On payment, server generates HMAC token
   (`hmac.new(secret, date_str, sha256)`) valid for 24 hours
6. **Authorization**: Token stored client-side, sent with subsequent MC requests

### Token structure

```python
token = hmac.new(BTCPAY_SECRET, today_str.encode(), hashlib.sha256).hexdigest()
```

Daily expiry — token is valid only for the calendar day it was generated. No
server-side session state; token is self-validating.

---

## 15. Deployment

### Production stack

```
nginx (HTTPS, Let's Encrypt)
  └── reverse proxy → 127.0.0.1:8050
        └── gunicorn (5 workers, 120s timeout, --preload)
              └── Dash app (Plotly + Flask)

systemd services:
  quantoshi.service       — main app (gunicorn)
  quantoshi-cache.service — oneshot, pre-loads MC cache to /dev/shm at boot
```

### Server

- **Host**: Hetzner VPS, IP `89.167.70.45`
- **Clearnet**: https://quantoshi.xyz
- **Tor**: `u5dprelc4ti7xoczb5sbtye6qidlji2l6psmkx35anvxgjyqrkmu32ad.onion`
- **Log retention**: 27 days (nginx + gunicorn, daily rotation)

### Deploy process

```bash
git push origin master
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && systemctl restart quantoshi"
```

### nginx JS caching

`/_dash-component-suites/` URLs contain version hashes (immutable assets).
nginx caches them for 7 days:

```nginx
location /_dash-component-suites/ {
    add_header Cache-Control "public, max-age=604800, immutable";
}
```

Plotly.js is 4.7 MB (gzipped ~1.5 MB) — fetched once, then served from
browser cache. `/_dash-layout` and `/_dash-dependencies` are explicitly set
to `no-cache` so callback-graph changes are always detected.

### Redis socket pre-check

`redis_available()` in `_app_ctx.py` probes the Redis socket with a 0.2s
timeout before attempting a full `ping`. Without this, a missing Redis
instance causes an 8.6s TCP connection timeout that blocks gunicorn worker
startup.

### Security headers

Set via `@server.after_request` in `app.py`:
- Content-Security-Policy
- Referrer-Policy: `no-referrer`
- X-Frame-Options: `DENY`
- Onion-Location (for Tor discovery)
- Cache-Control: `no-cache` on `/_dash-layout` and `/_dash-dependencies`

---

## 16. Testing

### Test suite

428 tests in `btc_web/test_web.py` (3,345 lines), organized as
`unittest.TestCase` classes.

### Test categories

| Category | Coverage |
|----------|----------|
| Utilities | `_q3()` quantization, `_quantize_params()`, `_nearest_quantile()` |
| Figure builders | All 6 `build_*_figure()` functions with various param combos |
| MC cache | Cache generation, loading, lookup, bin snapping |
| Snapshots | Encode/decode round-trips, bitmask encoding, v1/v2/v3 compat, edge cases |
| Financial math | `_compute_sc_loan()`, DCA accumulation, tax treatment |
| Callback smoke tests | Each major callback with representative inputs |
| BTCPay | Pricing tiers, HMAC token generation/verification |
| API validation | Rate limiting, input sanitization |
| Price cache | Fetching, TTL, circuit breaker |
| Regime filters | Bin masking, ghost overlay, fuzz testing |

### Running tests

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v
```

---

## 17. Developer Tools

### `tools/sweep_support.py` — Support Line Parameter Sweep

2D grid search over `SUPPORT_PERCENTILE` × `SUPPORT_QUANTILE` to find the
combination that maximises the bubble model composite R². Extracts the core
fitting logic from `SP.ipynb` cell 0 (support line → peak detection → bubble
fitting → R²) without running the full notebook.

```bash
btc_venv/bin/python3 tools/sweep_support.py [--pct-lo 5] [--pct-hi 50] \
    [--pct-step 5] [--q-lo 0.1] [--q-hi 0.9] [--q-step 0.1] \
    [--out sweep_support.jpg]
```

Reads `BUBBLE_YEARS`, `FIT_MIN_DATE`, and other config from `SP.ipynb` cell 0
automatically. Outputs a 2-panel heatmap (R² composite + support slope) and
prints the top 10 parameter combinations. Run after changing `BUBBLE_YEARS`,
`FIT_MIN_DATE`, or genesis date to re-optimise the support line.

---

## 18. Adding a New Price Model

Checklist for adding a new model to Quantoshi:

1. **Implement** the `PriceModel` protocol in `archive/btc_app/btc_core.py`:
   - Required fields: `name`, `short_name`, `quantized`, `quantiles`, `colors`, `fits`, `dash_style`
   - Required methods: `price_at(q, t)`, `interp_price(q, t)`, `find_percentile(t, price)`
   - `fits` dict must contain keys for all quantiles — figure builders check `q in model.fits`
   - Composite-median models (shaped curves): follow `LPPLModel` / `EmpiricalFloorModel` pattern
   - Log-linear models (straight lines in log-log): extend `_FitsBasedModel`
2. **Register** in `btc_web/app.py` inside the "register price models" block
3. **Update `btc_web/snapshot.py`** — add the model's `short_name` to `_CHECKLIST_OPTIONS` for all `*-model-show` and `bub-model-show` keys (~lines 164–168). Without this, snapshot/share links cannot encode the model selection. Old links decode safely (missing bits default to unselected).
4. **Update `btc_web/test_web.py`** — the `PRICE_MODELS.keys()` assertion uses a hardcoded set. Use `issubset()` or add the new key. Also add model-specific test class.
5. **UI auto-discovers** via `PRICE_MODELS` iteration in `_model_show_checklist()` (`layout/common.py`) and heatmap pill bar (`layout/heatmap.py`) — no layout changes needed.
6. Add accordion item to `btc_web/layout/model_info.py`
7. Add FAQ entry if warranted in `btc_web/layout/faq.py`
8. Update `docs/architecture.md` and `docs/user_manual.md`

---

## Appendix A: ModelData Fields

```python
class ModelData:
    qr_fits: dict[float, dict]     # {quantile → {"intercept", "slope", "r2"}}
    QR_QUANTILES: list[float]      # all fitted quantiles (0.001–0.999)
    ols_intercept: float           # OLS regression intercept
    ols_slope: float               # OLS regression slope
    genesis: pd.Timestamp          # "2009-07-25"
    years_plot_bm: ndarray         # bubble model x-axis (years)
    support_bm: ndarray            # bubble support line
    comp_by_n: list[ndarray]       # composites for N=1..n_future_max
    bm_r2: float                   # bubble model R-squared
    n_future_max: int              # max future bubble count
    price_dates: list[str]         # daily dates
    price_years: ndarray           # daily years-since-genesis
    price_prices: ndarray          # daily prices (USD)
    qr_colors: dict[float, str]    # hex color per quantile
    qr_linestyles: dict            # line style per quantile
    # Visual config: PLOT_BG_COLOR, TEXT_COLOR, TITLE_COLOR, etc.
    # Heatmap config: CAGR_SEG_*, CAGR_GRAD_STEPS, TABLE_YEARS, etc.
```

## Appendix B: Key Constants

### `_app_ctx.py`

| Constant | Value | Purpose |
|----------|-------|---------|
| `FREQ_PPY` | `{Daily:365, Weekly:52, Monthly:12, Quarterly:4, Annually:1}` | Periods per year |
| `FREQ_STEP_DAYS` | `{Daily:1, Weekly:7, Monthly:30, Quarterly:91, Annually:365}` | MC step size |
| `MAX_USD` | `4,294,967,295` | uint32 clamp for dollar inputs |
| `SC_DEFAULT_RATE` | `13.0` | Stack-celerator default interest rate (%) |
| `SC_DEFAULT_PRICE` | `80,000` | Stack-celerator default entry price ($) |

Note: `BTC_ORANGE` moved to `colors.py`. `FONT_LEGEND` removed (now `CHART_FONT_LEGEND` in `colors.py`).

### `colors.py` Section 5 — Appearance constants (single source of truth)

All rendering constants now live in `btc_web/colors.py` Section 5. The old `figures/common.py` private constants (`_QR_LINE_WIDTH`, `_SHADE_ALPHA`, etc.) are backward-compat aliases that import from `colors.py`.

| Constant | Value | Purpose |
|----------|-------|---------|
| `TRACE_WIDTH` | `2.5` | Primary quantile trace line width |
| `TRACE_WIDTH_OVERLAY` | `2.0` | Alt-model overlay line width |
| `TRACE_WIDTH_COMPOSITE` | `2.0` | Bubble composite trace |
| `TRACE_WIDTH_SUPPORT` | `1.5` | Bubble support trace |
| `SHADE_ALPHA` | `0.08` | Outer quantile band fill alpha |
| `WM_OPACITY` | `0.35` | Watermark opacity |
| `LEGEND_BG_OPACITY` | `0.92` | Chart legend background alpha |
| `PT_SIZE_DEFAULT` | `5` | Scatter data point size |
| `PT_ALPHA_DEFAULT` | `0.3` | Scatter data point alpha |
| `CHART_FONT_TITLE` | `15` | Chart title font size (mobile) |
| `CHART_FONT_TITLE_LG` | `19` | Chart title font size (desktop) |

See `colors.py` for the full list (~150 constants across 5 sections).

### `mc_cache.py` free tier

| Constant | Value |
|----------|-------|
| `MC_FREE_SIMS` | 100 |
| `MC_FREE_START_YRS` | [2028, 2031] |
| `MC_FREE_ENTRY_Q` | 10 |
| `MC_FREE_YEARS` | [10, 20] |
