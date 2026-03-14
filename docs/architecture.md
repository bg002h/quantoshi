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
  │  SP.ipynb    │ ──────────────────►  │  btc_web/    │  (Plotly Dash, 7 tabs)
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
  ├── imports utils.py       (LRU caches, price fetching)
  ├── imports snapshot.py    (state encoding/decoding)
  ├── imports layout.py      (all tab layouts)
  ├── imports callbacks.py   (all callback registrations)
  └── imports api.py         (REST API routes)

figures.py
  ├── imports mc_overlay.py  (MC simulation + trace builders)
  │     ├── imports mc_cache.py   (pre-computed cache)
  │     ├── imports btc_core.py   (ModelData, qr_price)
  │     └── imports markov        (Cython engine, optional)
  └── imports _app_ctx.py    (shared constants)
```

### Module responsibilities

| Module | Lines | Purpose | Key exports |
|--------|-------|---------|-------------|
| `app.py` | 340 | Orchestrator: app creation, model load, Flask routes, cache prewarm | `app`, `server` |
| `_app_ctx.py` | 56 | Shared state and constants | `M`, `app`, `FREQ_PPY`, `BTC_ORANGE`, `_compute_sc_loan()` |
| `utils.py` | 188 | Float quantization, 6 LRU figure caches, price fetching | `_q3()`, `_get_*_fig()`, `_fetch_btc_price()` |
| `snapshot.py` | 234 | Snapshot encoding/decoding, bitmask helpers | `_encode_snapshot()`, `_decode_snapshot()`, `_SNAPSHOT_CONTROLS` |
| `layout.py` | 2,256 | All 7 tab layouts, shared settings panels, splash modal, navbar | `main_layout()`, `_STYLE_*` constants |
| `callbacks.py` | 2,950 | ~20 server callbacks + ~30 clientside JS callbacks, `_ci()`/`_cf()` coercion | `update_bubble()`, `update_heatmap()`, `update_dca()`, etc. |
| `figures.py` | 2,148 | 6 Plotly chart builders, annotation overlap resolver, `_finalize_chart()` | `build_bubble_figure()`, `build_heatmap_figure()`, etc. |
| `mc_overlay.py` | 919 | MC simulation, caching, fan band traces, regime filters | `_mc_dca_overlay()`, `_mc_retire_overlay()`, etc. |
| `mc_cache.py` | 431 | Pre-computed MC cache generation/loading/lookup | `load_caches()`, `get_cached_paths()`, `get_cached_overlay()` |
| `api.py` | 204 | REST API endpoints | `register_routes()` |
| `btcpay.py` | 260 | BTCPay Server payment integration | Invoice lifecycle |
| `btc_core.py` | 267 | ModelData class, QR pricing math, lot percentiles | `ModelData`, `qr_price()`, `yr_to_t()` |
| `test_web.py` | 3,345 | 428 tests: utilities, builders, snapshots, callbacks, btcpay, regime filters | — |

**Total web app**: ~10,000 lines across 12 modules + 3,345 lines of tests.

---

## 3. Tab Architecture

### 7 tabs

| # | Tab | ID | Chart builder | MC overlay | Key controls |
|---|-----|----|---------------|------------|--------------|
| 1 | Bubble + QR Overlay | `bubble` | `build_bubble_figure()` | None | Quantiles, axes, N future bubbles, stack |
| 2 | CAGR Heatmap | `heatmap` | `build_heatmap_figure()` | `_mc_heatmap_overlay()` | Entry yr/percentile, color mode, gradient |
| 3 | BTC Accumulator | `dca` | `build_dca_figure()` | `_mc_dca_overlay()` | DCA amount, freq, Stack-celerator |
| 4 | BTC RetireMentator | `retire` | `build_retire_figure()` | `_mc_retire_overlay()` | Withdrawal, inflation, depletion arrows |
| 5 | HODL Supercharger | `supercharge` | `build_supercharge_figure()` | `_mc_supercharge_overlay()` | Mode A/B, delays, depletion bands |
| 6 | Stack Tracker | `stack` | None (DataTable) | None | Lot CRUD, import/export |
| 7 | FAQ | `faq` | None | None | Accordion, deep-linkable |

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

### Display Models toggle

Each MC-enabled tab has a `{prefix}-model-show` checklist in Chart Settings
with options `["qr", "mc"]` (both checked by default when MC is enabled).
This controls which model's traces are visible on the chart. When MC is off,
QR is always shown.

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

where `t = (date - genesis).days / 365.25` (years since the Genesis Block,
January 3, 2009).

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

`_interp_qr_price(q, t, qr_fits)` in `figures.py` handles non-standard
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

### Chart finalization (`_finalize_chart()` in figures.py)

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

90 `_SNAPSHOT_CONTROLS` entries — `(component_id, property)` tuples covering
all UI controls across 7 tabs.

### Encoding pipeline

```
Control states → JSON array (90 elements) → gzip → base64 urlsafe → URL hash
```

URL format: `host/N#q3:ENCODED` where `N` is the tab path (1–7).

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

### Callback inventory

| Type | Count | Description |
|------|-------|-------------|
| Server callbacks | ~20 | Tab updates, ticker, share modal, MC controls |
| Clientside callbacks | ~30 | Tab routing, zoom toggle, UI visibility |
| Loop-created callbacks | ~16 | MC toggle, advanced toggle, regime opts, freq unlock |

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
- Tab routing (`/1`–`/7` → tab switch)
- Zoom toggle (dragmode enable/disable)
- MC control visibility
- SC mode A/B panel switching
- FAQ deep-linking (`/7.N`)

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

---

## 12. Live Price Ticker

- `dcc.Interval(id="price-interval", interval=20*60*1000)` fires every 20 min
- Primary: Binance (`api.binance.com/api/v3/ticker/price?symbol=BTCUSDT`)
- Fallback: CoinGecko (for US geo-blocked users)
- Outputs: navbar `price-ticker` div, `btc-price-store`, heatmap `hm-entry-q`
- Heatmap uses `live_price` as entry price when `entry_yr == current_year`
- Binance is geo-blocked in the US (HTTP 451) but works from the Hetzner server

---

## 13. Layout Patterns

### Style constants

Module-level constants in `layout.py` for repeated inline styles:

| Constant | Value | Used for |
|----------|-------|----------|
| `_STYLE_HIDDEN` | `{"display": "none"}` | Hidden containers, placeholder controls |
| `_STYLE_HINT` | `{"color": "#888", ...}` | Hint/instruction text below controls |
| `_STYLE_GRAPH_H` | `{"height": "78vh"}` | Chart graph containers |
| `_STYLE_COLOR_H` | `{"height": "28px"}` | Color picker inputs |
| `_STYLE_ADDR_CELL` | `{...nowrap, verticalAlign}` | FAQ address table cells |
| `_STYLE_ADDR_CODE` | `{...break-all, 11px}` | FAQ address code blocks |

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

## Appendix A: ModelData Fields

```python
class ModelData:
    qr_fits: dict[float, dict]     # {quantile → {"intercept", "slope", "r2"}}
    QR_QUANTILES: list[float]      # all fitted quantiles (0.001–0.999)
    ols_intercept: float           # OLS regression intercept
    ols_slope: float               # OLS regression slope
    genesis: pd.Timestamp          # "2009-01-03"
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
| `BTC_ORANGE` | `#f7931a` | Bitcoin brand color |

### `figures.py` rendering constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `_QR_LINE_WIDTH` | `1.8` | Quantile trace line width |
| `_SHADE_ALPHA` | `0.08` | Fill opacity between quantile bands |
| `_WM_OPACITY` | `0.55` | Watermark opacity |
| `_OVERLAP_LOG` | `0.12` | Annotation overlap threshold (log-decades) |
| `_OVERLAP_FRAC` | `0.06` | Annotation overlap threshold (linear fraction) |
| `_CONSOLIDATE_THRESHOLD` | `4` | Annotations per cluster before merging |
| `_BISECT_ITERS` | `60` | Binary search iterations (SC Mode B) |

### `mc_cache.py` free tier

| Constant | Value |
|----------|-------|
| `MC_FREE_SIMS` | 100 |
| `MC_FREE_START_YRS` | [2028, 2031] |
| `MC_FREE_ENTRY_Q` | 10 |
| `MC_FREE_YEARS` | [10, 20] |
