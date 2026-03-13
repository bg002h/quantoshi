# Quantoshi Architecture Guide

Developer-facing reference for the Quantoshi codebase. Covers system design,
module responsibilities, model math, and key subsystems.

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

**Runtime**: pkl load &rarr; Dash app init &rarr; figure builders generate Plotly
charts on demand &rarr; browser renders interactive graphs. All user state lives
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
| `snapshot.py` | 218 | Snapshot encoding/decoding, bitmask helpers | `_encode_snapshot()`, `_decode_snapshot()`, `_SNAPSHOT_CONTROLS` |
| `layout.py` | 2155 | All 7 tab layouts, splash modal, navbar | `main_layout()` |
| `callbacks.py` | 2864 | ~20 server callbacks + ~30 clientside JS callbacks | `update_bubble()`, `update_heatmap()`, `update_dca()`, etc. |
| `figures.py` | 1910 | 6 Plotly chart builders + styling helpers | `build_bubble_figure()`, `build_heatmap_figure()`, etc. |
| `mc_overlay.py` | 919 | MC simulation, caching, fan band traces, regime filters | `_mc_dca_overlay()`, `_mc_retire_overlay()`, etc. |
| `mc_cache.py` | 431 | Pre-computed MC cache generation/loading/lookup | `load_caches()`, `get_cached_paths()`, `get_cached_overlay()` |
| `api.py` | 199 | REST API endpoints | `register_routes()` |
| `btcpay.py` | 260 | BTCPay Server payment integration | invoice lifecycle |
| `btc_core.py` | 267 | ModelData class, QR pricing math, lot percentiles | `ModelData`, `qr_price()`, `yr_to_t()` |

**Total web app**: ~13,200 lines across 12 modules.

---

## 3. Quantile Regression Model

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

## 4. Bubble Model

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

### Envelope and decomposition

The notebook also generates envelope (upper bound) and decomposition
(individual bubble contributions) charts, exported as static images
(`bm_envelope.*`, `bm_decomposition.*`).

---

## 5. Markov MC Engine

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

## 6. MC Overlay Pipeline

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
  `mc_freq`, `mc_window`, `mc_blocked_bins`.

- **`_mc_overlay_key(p, tab, start_stack)`**: Identifies the post-simulation
  overlay (DCA accumulation, withdrawal depletion, etc.). Adds `amount`,
  `inflation`, `start_stack` to the path key.

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

### Free tier

| Parameter | Free | Paid |
|-----------|------|------|
| Simulations | 100 | 800 |
| Start years | 2028, 2031 | Any cached |
| Entry percentile | 10% | Any |
| Duration | 10, 20 yr | 10–40 yr |

---

## 7. Payment Flow

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

## 8. Snapshot / Share System

### Control inventory

95 `_SNAPSHOT_CONTROLS` entries — `(component_id, property)` tuples covering all
UI controls across 7 tabs. Each entry maps to a Dash component whose state is
captured for sharing.

### Encoding pipeline

```
Control states → JSON array (95 elements) → gzip → base64 urlsafe → URL hash
```

URL format: `host/N#q2:ENCODED` where `N` is the tab path (1–7).

### Bitmask encoding

20 checklist fields use bitmask encoding for compact URLs:
- 5 quantile checklists: up to 17-bit each (17 possible quantiles)
- 15 toggle checklists: 1–7 bits each

`_list_to_mask(val, opts)` encodes, `_mask_to_list(mask, opts)` decodes. Old v1
links (`#q1:...`) stored plain lists — decoder handles both via
`isinstance(val, int)`.

### Tab-scoped snapshots

`_TAB_CONTROLS` maps each `tab_id` to its set of component IDs.
`_encode_snapshot(state_dict, tab_filter=controls)` encodes only the active tab's
controls as non-null; others default to `null` and fall back to defaults on
restore. This produces much shorter URLs for single-tab shares.

---

## 9. Deployment

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
- **Tor**: Hidden service via `tor@default`
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

## 10. Testing

### Test suite

418 tests in `btc_web/test_web.py`, organized as `unittest.TestCase` classes.

### Test categories

| Category | Coverage |
|----------|----------|
| Utilities | `_q3()` quantization, `_quantize_params()`, `_nearest_quantile()` |
| Figure builders | All 6 `build_*_figure()` functions with various param combos |
| MC cache | Cache generation, loading, lookup, bin snapping |
| Snapshots | Encode/decode round-trips, bitmask encoding, v1 compat, edge cases |
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

### Adding tests

Create a new `class TestXxx(unittest.TestCase)` in `test_web.py`. Follow
existing patterns — most tests construct a params dict and call a builder
function or helper directly.

---

## Appendix: ModelData Fields

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
