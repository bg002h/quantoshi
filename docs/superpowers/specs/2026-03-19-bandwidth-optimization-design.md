# Bandwidth Optimization Plan

**Date:** 2026-03-19
**Status:** Approved

## Problem

Quantoshi's Dash web app transmits full Plotly JSON figures on every control interaction. A single bubble chart render is 1.8 MB raw JSON. A typical 20-interaction session consumes 30+ MB. This prevents:
- Making MC simulation free (bandwidth cost too high)
- Supporting low-speed / Tor connections
- Handling many simultaneous users on a 2-core Hetzner VPS
- Operating without bot scraping driving up costs

## Verified Data

Measured with real figure builders using default parameters:

| Chart | Raw JSON | Gzipped | Traces | Points |
|-------|----------|---------|--------|--------|
| **Bubble (9 quantiles)** | **2,048 KB** | **956 KB** | 35 | 52,296 |
| Heatmap (9q x 33yr) | 63 KB | 22 KB | 1 | 11 |
| DCA (7q, 14yr monthly) | 106 KB | 30 KB | 14 | 1,862 |
| Retire (7q, 44yr monthly) | 121 KB | 23 KB | 14 | 3,374 |
| Supercharger (2q, 42yr annual) | 31 KB | 18 KB | 3 | ~130 |

**The bubble chart is the only bandwidth problem.** At 2 MB raw (956 KB gzipped), it has 52,296 data points across 35 traces — driven by 1,500-point QR interpolation curves with glow shadows. All other charts are already under 130 KB raw / 30 KB gzipped.

Synthetic test with reduced bubble (400 pts instead of 1500): **493 KB raw → 76 KB gzipped** (96% reduction from original).

Key insight: gzip alone saves only 53% on the bubble chart (floating point numbers produce unique strings that compress poorly). **Reducing interpolation points is the multiplier** — fewer unique floats means dramatically better gzip ratios.

---

## Tier 1: Data Diet (implement now)

**Effort:** Days. **UX change:** None. **Bandwidth reduction:** ~95%.

### 1a. Reduce interpolation points (bubble chart)

**File:** `btc_web/figures/common.py:43-44`

```python
# Before
_INTERP_POINTS = 1500

# After
_INTERP_POINTS = 400
```

**Rationale:** A 1920px screen has ~1200 usable chart pixels. Mobile has ~400px. 400 interpolation points is more than sufficient for visual fidelity at any screen width.

**Note:** `_INTERP_POINTS` only affects the bubble chart QR curves. DCA/retire/supercharger generate time series via `_build_time_array()` using `np.arange(t_start, t_end, dt)` — their point count is determined by time span and frequency (monthly = ~120 pts for 10 years, daily = ~18,000 for 50 years). Those charts need separate analysis (see 1a-ii below).

**Keep `_MAX_SCATTER_PTS = 1200`** — scatter data is only ~30 KB raw (1200 x/y pairs). Not the bandwidth problem, and reducing it risks visible gaps in sparse early Bitcoin data (2009-2012).

**Impact on bubble chart:** Raw JSON drops from 1,814 KB to ~493 KB (73% reduction). Combined with gzip, drops to ~76 KB.

**Note:** DCA/retire/supercharger are already small (30-106 KB raw, 18-30 KB gzipped). They use simulation time arrays (~120-530 monthly steps), not `_INTERP_POINTS`. No changes needed for these charts — gzip alone is sufficient.

### 1b. Enable gzip compression on nginx

**File:** Production nginx config on Hetzner VPS (`/etc/nginx/sites-available/quantoshi` or similar)

Add to the `server` or `http` block:
```nginx
gzip on;
gzip_vary on;
gzip_proxied any;
gzip_comp_level 6;
gzip_min_length 256;
gzip_types
    application/json
    application/javascript
    text/css
    text/html
    text/plain;
```

**Impact:** 61% reduction on raw JSON, 84-96% when combined with point reduction. Also compresses the Plotly.js bundle (~3.5 MB → ~900 KB gzipped) — a free win on initial page load.

### 1c. Round trace floats (significant figures, not fixed decimals)

In figure builders, round x/y data to reduce unique float strings (improves gzip ratio). Use significant-figure rounding, not fixed decimal places — BTC prices span $0.06 to $10M+ on log scale, so fixed `round(v, 2)` would cause visible errors at low values.

Use the existing `_q3()` function from `utils.py` (rounds to 3 significant figures) or similar:

```python
# In each figure builder, before returning traces:
x = [_q3(v) for v in x_values]
y = [_q3(v) for v in y_values]
```

Apply to: `figures/bubble.py`, `figures/dca.py`, `figures/retire.py`, `figures/supercharge.py`, `figures/common.py` (interpolation helper).

**Impact:** Further 10-20% gzip improvement on already-reduced data.

### 1d. Verify LRU cache prewarm still works

**File:** `btc_web/app.py` (`_prewarm_caches()`)

The prewarm function calls figure builders with default params — it does not reference point counts directly. Figures will simply be smaller. Verify prewarm completes without error after changes.

### Tier 1 Expected Result

| Chart | Before (raw) | Before (gzip) | After (reduced pts + gzip) |
|-------|-------------|---------------|---------------------------|
| Bubble | 2,048 KB | 956 KB | **~76 KB** |
| Heatmap | 63 KB | 22 KB | 22 KB (already small) |
| DCA | 106 KB | 30 KB | 30 KB (already small) |
| Retire | 121 KB | 23 KB | 23 KB (already small) |
| Supercharger | 31 KB | 18 KB | 18 KB (already small) |

**Session total:** Bubble-heavy session ~20 MB → ~1 MB. Non-bubble tabs already efficient with gzip alone.

---

## Tier 2: Free MC + Rate Limiting (implement after Tier 1)

**Effort:** 1-2 weeks. **UX change:** MC unlocked for all users (cached scenarios). **Enables:** Free-tier MC.

### 2a. Reduce MC simulations 800 → 200

**Files:** `btc_web/mc_cache.py` (MC_SIMS constant), cache rebuild required.

- Cache RAM drops from ~834 MB → ~210 MB
- Cache build time drops proportionally (~75% faster)
- P5-P95 bands: still statistically solid (10+ paths per tail)
- P1%/P99%: **highly unstable** — only 2 paths define each extreme. Consider dropping P1% from the fan display at 200 sims, or increasing to 250 sims for 2-3 paths per extreme
- P25-P75: barely affected (50+ paths per quartile)

### 2b. Make cached MC scenarios free

**Files:** `btc_web/callbacks/mc_payment.py`, `btc_web/callbacks/mc_controls.py`

- Remove payment gate for pre-computed cache hits
- Keep paywall for custom (non-cached) simulations that require real-time compute
- `_HAS_MARKOV` flag stays — controls whether the markov module is available at all
- New flag: `_MC_FREE_CACHED = True` — allows free access to pre-computed scenarios

### 2c. nginx rate limiting

**File:** Production nginx config

```nginx
# Rate limit zone: 120 requests/min (2/sec) per IP — scoped to callback endpoint only
limit_req_zone $binary_remote_addr zone=dash:10m rate=120r/m;

server {
    # Only rate-limit the Dash callback endpoint (where figure JSON lives)
    location /_dash-update-component {
        limit_req zone=dash burst=30 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:8050;
    }

    # Static assets, layout, dependencies — no rate limit
    location / {
        proxy_pass http://127.0.0.1:8050;
    }
}
```

**Note:** Dash fires a callback on every slider change. Dragging a range slider can fire 10+ callbacks in seconds. 120r/m with burst=30 allows rapid real use while blocking sustained scraping. Start generous, tighten based on access logs. Scoped to `/_dash-update-component` only — static assets and page loads are unlimited.

### 2d. Application-level rate limiting

**File:** New middleware in `btc_web/app.py` or separate `btc_web/ratelimit.py`

- Token bucket per IP with exponential backoff
- Exempt static assets and initial page load
- Target `/_dash-update-component` endpoint (where figure JSON lives)
- Return `429 Too Many Requests` with `Retry-After` header
- Log rate-limited IPs for monitoring

### 2e. Rebuild MC cache with 200 sims

After deploying 2a, rebuild cache (locally on desktop, scp to server — ~10 min build time with 24 cores vs 8 hours on VPS).

### Tier 2 Expected Result

- MC overlays add ~30-50 KB per chart (gzipped)
- Full MC-enabled session: ~2-3 MB total
- All ~45,000 pre-computed scenarios available to all users
- Custom simulations remain paywalled
- Scrapers blocked at 120 req/min sustained (callback endpoint only)

---

## Tier 3: Lightweight Rendering (SAVE FOR FUTURE DEVELOPMENT — DO NOT IMPLEMENT NOW)

**Effort:** Months. **UX change:** Significant (static charts by default, interactive on demand).

### 3a. Server-side image rendering

Replace Plotly JSON with server-rendered WebP images via kaleido:

```python
img_bytes = fig.to_image(format="webp", width=1200, height=700, scale=2)
```

A WebP chart image is ~50-100 KB vs ~76 KB gzipped JSON (similar size, but eliminates client-side Plotly.js rendering overhead and the 3+ MB Plotly.js bundle download).

### 3b. "Go Interactive" toggle

Default: static WebP image (fast, lightweight, works everywhere including Tor).

User clicks "Go Interactive" button → loads full Plotly.js bundle + JSON figure on demand. Enables:
- Hover tooltips with exact values
- Zoom/pan
- Plotly toolbar (download, autoscale, etc.)

Only users who need interactivity pay the bandwidth cost. Most users just viewing charts never load Plotly.js at all.

**Implementation:** Each `dcc.Graph` replaced with a custom component:
- Renders `<img>` tag by default (server-sent WebP)
- On toggle, dynamically loads Plotly.js and replaces img with interactive figure
- Plotly.js loaded once per session (cached), not per chart

### 3c. Proof-of-work gate

Require client to solve a small hashcash puzzle before serving chart data:
- ~100ms on a real browser (imperceptible to user)
- Expensive for bot farms running thousands of requests
- No payment friction, no captcha UI
- Implemented as a middleware: server issues challenge, client returns solution in header

### 3d. Replace Plotly.js with lightweight renderer

The nuclear option. Plotly.js is 3+ MB minified. Replace with:
- Canvas2D or WebGL renderer (~50 KB)
- Server sends binary data arrays (MessagePack or protobuf, ~5-20 KB per chart)
- Client renders locally

This eliminates the Plotly.js download entirely and reduces per-chart payloads to binary data. But it means rewriting all 5 chart builders and losing the Plotly ecosystem (export, annotations, etc.).

### Tier 3 Expected Result

| Metric | Current | After Tier 1+2 | After Tier 3 |
|--------|---------|----------------|--------------|
| Initial page load | ~5 MB | ~4 MB | ~500 KB |
| Per-chart interaction | 1.8 MB | ~76 KB | ~50 KB (image) or ~10 KB (binary) |
| 20-interaction session | 30+ MB | ~2 MB | ~1 MB |
| Plotly.js bundle | 3+ MB | 3+ MB | 0 (or on-demand) |

---

## Implementation Order

```
Tier 1 (now) → Tier 2 (after Tier 1 deployed) → Tier 3 (future, if needed)
```

Tier 1 alone delivers 95% of the bandwidth reduction. Tier 2 unlocks the business goal (free MC). Tier 3 is insurance if traffic grows beyond what Tiers 1+2 can handle on the current VPS.

## Risk

- **Tier 1:** Very low. Reducing points is invisible to users. Gzip is standard.
- **Tier 2:** Low-medium. Rate limiting needs tuning to avoid blocking real users. MC quality slightly reduced at 200 sims.
- **Tier 3:** High. Major refactor, new rendering pipeline, potential regressions.
