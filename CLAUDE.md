# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**Quantoshi** — Bitcoin price projection tools. Three components share the same underlying quantile regression bubble model:

1. **`SP.ipynb`** — Jupyter notebook with bubble model + quantile regression analysis, chart generation, and PowerPoint export.
2. **`btc_web/`** — Plotly Dash web app. Live at [quantoshi.xyz](https://quantoshi.xyz) and `u5dprelc4ti7xoczb5sbtye6qidlji2l6psmkx35anvxgjyqrkmu32ad.onion`.
3. **`archive/btc_app/btc_projections.py`** — Standalone PyQt5 GUI app (5 interactive tabs) distributed as a Linux AppImage. **On back burner** — moved to `archive/` during simplification.

The notebook generates `archive/btc_app/model_data.pkl`, which both the web app and the standalone app load at runtime.

**Optimal time origin:** All models use `2009-07-25` as their time origin — the statistically optimal start date for the power law fit, confirmed independently by multiple researchers. Three statistical tests (Durbin-Watson, out-of-sample RMSE, slope stability) converge on this date across 546 candidates. Blockchain data shows no economic transactions at this time; the first dollar-denominated transactions appear late 2009. The date marks where Bitcoin's price behavior begins to follow a power law, not a specific economic event. Distinct from the Bitcoin genesis block (2009-01-03).

---

## Workflow

**Never auto-deploy.** After making changes, stop at committing locally. Do NOT push to GitHub or SSH-deploy to the production server unless explicitly asked. The user will say "deploy to production" when ready to ship.

**Local test environment:** `DEV=1 bash run_web.sh` (hot-reload, single user) or `bash run_web.sh` (gunicorn). The local btc-web systemd service has been stopped; start the app manually when needed.

---

## Commands

### Run the notebook
```bash
~/.local/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 SP.ipynb
```
`jupyter` is installed via **pipx** (`~/.local/bin/jupyter`), not in `btc_venv`. The 600s timeout is required — cell 1 generates many charts and takes well over 2 minutes.

### Run the standalone app directly (for testing)
```bash
cd /scratch/code/bitcoinprojections/archive/btc_app
/scratch/code/bitcoinprojections/btc_venv/bin/python3 btc_projections.py
```

### Syntax check the app
```bash
/scratch/code/bitcoinprojections/btc_venv/bin/python3 -m py_compile \
    archive/btc_app/btc_projections.py && echo "OK"
```

### Build the AppImage
```bash
cd /scratch/code/bitcoinprojections/btc_app
bash build_appimage.sh          # uses up to 18 CPUs
JOBS=8 bash build_appimage.sh   # override CPU count
```
Output: `archive/btc_app/Quantoshi-x86_64.AppImage` (~140 MB)

### Update Bitcoin price data
```bash
python3 update_prices.py            # dry-run to preview
python3 update_prices.py --dry-run  # (same — add flag explicitly)
python3 update_prices.py            # live: appends CSV + re-runs notebook
```
- Fetches daily closes from Binance (primary) or CoinGecko (fallback if geo-blocked)
- Intentionally skips the **8 most recent days** (settling period — data may be revised)
- Appends new rows to `BitcoinPricesDaily.csv` then re-executes `SP.ipynb`
- Prints a preview table of new rows; review before deploying

### Full rebuild after notebook changes
```bash
# 1. Execute notebook (regenerates model_data.pkl)
~/.local/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 SP.ipynb
# 2. Build AppImage
cd /scratch/code/bitcoinprojections/archive/btc_app && bash build_appimage.sh
```

### Run the web app locally
```bash
bash run_web.sh           # gunicorn, 5 workers, port 8050
DEV=1 bash run_web.sh     # Dash dev server with hot-reload (single user, skips prewarm)
PORT=8080 bash run_web.sh # custom port
```

### Syntax-check the web app
```bash
cd btc_web && PYTHONPATH=".:../archive/btc_app" ../btc_venv/bin/python3 -c \
  "import layout, figures, callbacks, cache, engines.adapter, engines.citadel, data.asset_matrices; print('OK')"
```

### Deploy to production (Hetzner VPS)
```bash
git push origin master
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"
```
**After price data update** (new model_data.pkl), also regenerate Citadel cache:
```bash
ssh root@89.167.70.45 "PYTHONPATH=/opt/quantoshi:/opt/quantoshi/btc_app:/opt/quantoshi/archive/btc_app:/opt/quantoshi/btc_web \
    btc_venv/bin/python3 btc_web/generate_citadel_cache.py"
```
Redis flush clears stale cache entries from old model data. New fingerprinted entries are built fresh on first request + Citadel cache regeneration.

---

## Notebook Architecture (`SP.ipynb`)

| Cell | Lines | Purpose |
|------|-------|---------|
| 0 | ~1390 | Bubble model — **do not modify** except `t_max` (projection horizon, currently 72 ≈ year 2081) |
| 1 | ~1112 | QR config & chart generation — primary work cell |
| 2 | ~293  | PowerPoint export (`bitcoin_projections.pptx`) |
| 3 | ~93   | Export cell — writes `archive/btc_app/model_data.pkl` |
| 4 | 0     | Empty |
| 5 | ~396  | Interactive bubble+QR overlay (`_launch_interactive()`) |
| 6 | ~316  | Interactive CAGR heatmap (`_launch_heatmap()`) |
| 7 | 0     | Empty |

**Cell 1 key functions:** `_draw_channels()`, `_draw_ols()`, `_draw_data()`, `_draw_today()`, `_price_yaxis()`, `_save_show()`. Quantile colors/linestyles are built dynamically from `QR_QUANTILES`.

### Editing notebook cells
Never edit the notebook JSON directly. Use a Python patch script:

```python
import json
with open("SP.ipynb") as f:
    nb = json.load(f)
raw = nb['cells'][N]['source']
src = ''.join(raw) if isinstance(raw, list) else raw
assert src.count(old) == 1, f"found {src.count(old)} times"
src = src.replace(old, new)
nb['cells'][N]['source'] = src
with open("SP.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
```

Write scripts to `/tmp/` and run with `python3 /tmp/script.py`.

**Encoding gotcha:** Notebook JSON stores some Unicode as literal escape sequences (e.g. `\u2500` as 6 characters). Use `\\u2500` in Python replacement strings to match them.

---

## Standalone App Architecture (`archive/btc_app/btc_projections.py`) — ON BACK BURNER

~3900 lines, structured as a set of tab classes managed by `MainWindow`.

### Tab classes
| Class | Tab | Key controls |
|-------|-----|-------------|
| `BubbleTab` | 1 | Bubble + QR overlay; axes scaling, quantile rows, bubble extrapolation |
| `HeatmapTab` | 2 | CAGR heatmap; entry/exit year/quantile, color modes, cell text modes |
| `DCATab` | 3 | Dollar-cost averaging simulation |
| `RetireTab` | 4 | Retirement withdrawal simulation |
| `StackTrackerTab` | 5 | Bitcoin lot tracking; emits `lots_changed` → updates all stack spinboxes |

### Key shared components
- **`FontPicker(QWidget)`** — family edit + size spinbox + "…" QFontDialog button. Emits `font_changed(str)` and `size_changed(int)`. Has `set_family()`/`set_size()` (no signal emit).
- **`ColorButton(QWidget)`** — color swatch that opens QColorDialog. Emits `color_changed(str)`.
- **`ModelData`** — dataclass loaded from `model_data.pkl`; carries all precomputed QR fits, bubble composites, and config constants.

### Cross-tab synchronization (MainWindow)
- **Font family** sync: `_font_role_map` (dict: role → list of FontPickers across tabs) + `_on_font_changed()`. Roles: `title`, `axis_t`, `ticks`, `legend`.
- **Font size** sync: `_syncing_font_sizes` guard + `_on_font_size_changed()`.
- **Minor ticks** sync: `_syncing_minor_ticks` guard + `_on_minor_ticks_changed()`.
- **"All tabs" font button**: Each tab emits `all_fonts_applied(str, int)` → `_apply_font_to_all_tabs()`.
- **Quantile state** sync: `q_state_changed(list)` → `_on_q_state_changed()` (shares QR row state across Bubble/DCA/Retire).

### Settings persistence
- Saved/loaded as JSON at `~/.config/btc-projections/ui_settings.json`.
- Each tab implements `_collect_settings() → dict` and `_apply_settings(dict)`.
- Font size keys use `_sz` suffix (e.g. `font_title_sz`), defaulting to match original hardcoded values: title=11, axis=10, ticks=10, ticks_minor=6, legend=7.

### Modifying btc_projections.py
Use string-replacement patch scripts (same `/tmp/` approach as notebook). Key rules:
- Each OLD pattern must appear **exactly once** (verify with `src.count(old)`), or use `replace_all=True` only when all occurrences should change identically.
- Patches must be ordered so later patterns match the already-transformed source.
- `DCATab` and `RetireTab` often share identical patterns and can use `replace_all=True`.
- `BubbleTab` uses separate `addWidget()` lines in its font all-row; `DCATab`/`RetireTab` use semicolon style.
- Font family variables are extracted **before** `self.fig.clear()` in DCA/Retire redraw (opposite of Bubble).

### Heatmap cell text modes
`mm` array = `exit_price / entry_price`. Modes: `cagr`, `price`, `both`, `stack` (CAGR + portfolio), `port_only`, `mult_only` (×), `cagr_mult`, `mult_port`, `none`.

---

## Web App Architecture (`btc_web/`)

### Files
| File | Purpose |
|------|---------|
| `app.py` | Dash app entry point — model loading, Flask config, cache prewarm |
| `_app_ctx.py` | Shared application context — `_q3`, `FREQ_LABEL`, `FREQ_PPY`, singleton flags (`_HAS_MARKOV`, `_HAS_CELERY`, `_HAS_REDIS`), `_MODEL_FP`, `redis_available()`, `redis_client()`, dynamic state (models, palettes) |
| `snapshot.py` | Snapshot/URL state encoding and decoding for share links |
| `api.py` | API route handlers |
| `utils.py` | Shared utilities (cache, figure builders, price fetching) |
| `btcpay.py` | BTCPay payment integration (Lightning/on-chain) |
| `mc_overlay.py` | Monte Carlo overlay integration + transition matrix caching |
| `mc_cache.py` | MC cache configuration and helpers |
| `tab_defaults.py` | Single source of truth for all tab defaults (`MappingProxyType` frozen dicts) |
| `cache.py` | L0 pinned + L2 Redis-backed figure cache (fingerprint invalidation) |
| `celery_app.py` | Celery application factory |
| `tasks.py` | Celery background tasks |
| `engines/` | `adapter.py` (Celery-or-in-process fallback), `citadel.py` (Citadel simulation engine) |
| `data/` | `asset_matrices.py`, `fetch_historical.py`, historical CSV data files |
| `load_shm_cache.py` | Shared memory cache loading |
| `test_web.py` | Comprehensive test suite |
| `layout/` | Layout package — tab controls, navbar, main assembly (13 modules incl. `citadel`, `model_info`) |
| `callbacks/` | Callbacks package — all Dash callbacks (17 modules incl. `routing`, `splash`, `user_model`, `citadel_cb`, `scanner`) |
| `figures/` | Figures package — chart builders + shared helpers (8 modules incl. `citadel`) |
| `assets/style.css` | Light theme (FLATLY) overrides + mobile layout |
| `assets/quantoshi_logo.png` | Master logo (575×360, 250KB — not directly served) |
| `assets/quantoshi_favicon.png` | Favicon (48×48, 3KB) |
| `assets/quantoshi_logo_nav.png` | Navbar image (128×80, 15KB) |
| `assets/quantoshi_logo_wm.png` | Chart watermark (100×63, 10KB) |
| `requirements.txt` | Python dependencies |

### Layout structure
`dbc.Navbar` (logo + "Quantoshi" in Palatino + live price ticker with 24h sparkline SVG + sats/$ toggle + "Stay dark, Anon →" + 🧅 Tor link + palette dropdown + 📸 Share button with "▲ Cooler than you think") → `dbc.Tabs` (9 tabs):

| Tab | ID | Key controls |
|-----|----|-------------|
| Bubble + QR Overlay | `bubble` | Quantiles, axes scale/range, bubble composite, N future bubbles |
| CAGR Heatmap | `heatmap` | Entry/exit year+quantile, color modes (Segmented/DataScaled/Diverging), multi-model pill bar carousel |
| BTC Accumulator | `dca` | DCA amount/frequency, year range, display mode, Stack-celerator |
| BTC RetireMentator | `retire` | Withdrawal amount, inflation rate, year range |
| HODL Supercharger | `supercharge` | Mode A (fixed spending → depletion date) or Mode B (fixed depletion → max spending); 5 delay offsets, 2 chart layouts |
| Stack Tracker | `stack` | Lot management (add/delete/import/export JSON) |
| Model Info | `model_info` | Accordion with per-model details. Deep-link: `/7.N` opens item N |
| FAQ | `faq` | Static accordion — `_FAQ` list in `layout/faq.py`. Deep-link: `/8.N` opens item N. Answers: plain strings or Dash components. Link color: `#1a6fa8` via `.accordion a` in style.css. |
| Citadel Planner | `citadel` | Sub-tabs: Assets / Spending / Rules / Simulation. "▶ Run Simulation" button. Trigger enable checkboxes, Historical Regimes asset growth mode, Show All/Hide All legend toggles |

### Tab defaults
All defaults are canonical in `btc_web/tab_defaults.py` (`MappingProxyType` frozen dicts) — do not hardcode elsewhere. `_prewarm_caches()` must stay in sync.

| Tab | Notable defaults |
|-----|-----------------|
| Bubble | Q50% selected, X scale=Log, N future bubbles=3, shade+show_data+show_today on (legend off). Pt size=3, Alpha=0.3. Auto Y checkbox (default on) rescales Y to fit selected quantiles at xmin/xmax. When Stack (BTC) is enabled, each quantile legend label gains `→ $X` showing the final USD stack value at the right x-range edge. |
| Heatmap | Entry year=current year, entry percentile=live BTC percentile (free numeric input 0–100%, NOT dropdown), exit years allow past. Entry price=live ticker when entry_yr==current year. Break1=0%, Break2=20%, Gradient Steps=32. |
| DCA | Default quantile Q50% only. dual_y+show_legend on. BTC-mode trace labels include final USD value in parentheses. Dual-y "USD Value (median)" always shows median USD across selected quantiles. **Stack-celerator** ("Enter Saylor Mode" checkbox): borrows `dca-sc-loan` $, buys BTC lump sum upfront, reduces DCA by the loan payment. Dashed overlay traces per quantile. Stack-celeration factor (median SC / median DCA) shown in chart title + legend. Controls: `dca-sc-type` (interest_only/amortizing), `dca-sc-rate`, `dca-sc-term`, `dca-sc-repeats` (0=one loan, N=N extra cycles back-to-back), `dca-sc-entry-mode` (live/model/custom for cycle 0; cycles 1+ always use model price), `dca-sc-custom-price`, `dca-sc-tax` (capital gains % on BTC sold to repay; default 33%). **Loan cap**: `principal` is silently capped at `max_principal = amount*(1-(1+r)^-n)/r` (amortizing) or `amount/r` (interest-only) when r>0, ensuring pmt ≤ DCA amount always. Info panel notes when cap is applied. **Tax applies only to interest-only**: BTC sold at cycle end to repay principal; tax applies **only to the capital gain** (sell_price − cost_basis), not full proceeds. Correct formula: `gain_per_btc = max(price - ep, 0.0); net_per_btc = price - tax_rate * gain_per_btc; sc_stack -= principal / net_per_btc`. If selling at a loss (`price ≤ ep`) no tax is owed. `ep` is the buy price at cycle start (tracked through the loop; rollover keeps first-cycle ep throughout). Amortizing repays principal in fiat — no BTC sold, tax has no effect. `outstanding` balance tracked per-period; deducted tax-adjusted at cycle end (interest-only) or post-loop (incomplete final cycle). **Rollover** (`dca-sc-rollover` checklist, interest-only only): repeat cycles skip BTC purchase (new loan pays off old, net zero BTC movement); cycle-end BTC sale skipped; single final repayment by post-loop deduction at simulation end price (with tax). Without rollover: each cycle independently buys BTC at start and sells at end. Rollover row hidden for amortizing. `dbc.Collapse` must NOT be used for SC body — use `html.Div(style={"display":"none"})` toggled via callback. |
| Retire | Default quantiles Q1%+Q10%+Q25%. year slider min=2024, default range 2031–2075, inflation=4%, log_y+dual_y+annotate on. Dual-y median same approach as DCA. |
| HODL Supercharger | Mode A, stack=1.0 BTC, delays=[0,0,0,1,2] yr, start_yr=2033, Monthly, inflation=4%, wd=$5,000/mo, end_yr=2075, USD display, annotate+log_y+show_legend on. Quantiles: Q0.1%+Q10% only. `sc-chart-layout` is a `dcc.Checklist` with single option "shade"; default `["shade"]` (bands on = layout 2). Display-q dropdown hidden when bands on. Depletion annotations: `_DELAY_COLORS` for traces/shading, `_ANNOT_COLORS` for arrow+text, staggered at 3 heights to avoid overlap. |
| Stack Tracker | default lot Price=$69,420 |
| Citadel Planner | start_yr=2031, end_yr=2075, Monthly, Q25% only. Sub-tabs for assets/spending/rules/simulation. |

### State and privacy
- Lot data lives in **browser `localStorage`** only — `dcc.Store(id='lots-store', storage_type='local')`.
- Nothing written server-side. Export via clientside JS blob download. Import via `dcc.Upload` + server-side base64 decode.
- Chart callbacks use `effective-lots` store (routes to snapshot lots or localStorage lots).

### Snapshot / Share feature
- `📸 Share` button → modal → **Scope** radio ("All tabs" / "Current tab only") → **Generate link** encodes control states + optional lots as gzip+base64 in URL hash. **Default scope: "Current tab only"** (shorter link; user can switch to "All tabs" for full cross-tab fidelity).
- URL format: `host/N#q3:...` where N is the tab path (`/1`–`/9`), so tab routing fires independently of hash decode.
- **All tabs** scope: encodes all ~206 controls (full fidelity). **Current tab only** scope: encodes only the active tab's controls via `tab_filter` — non-matching controls encode as `null` and fall back to defaults on restore (much shorter link).
- `_SNAPSHOT_CONTROLS` — list of ~206 `(component_id, property)` tuples. Format: `#q3:...` current; `#q2:...` and `#q1:...` legacy (still decoded).
- **Checklist bitmask encoding**: All 20 checklist fields (5 quantile + 15 toggle/boolean) are stored as bitmask integers in new links via `_CHECKLIST_OPTIONS` dict (component ID → ordered list of possible values) + `_list_to_mask(val, opts)` / `_mask_to_list(mask, opts)` helpers. Quantile fields: 17-bit each, saves ~435 JSON chars. Toggle fields: saves ~224 JSON chars. **Old `q2` links stored plain lists** — decoder handles both via `isinstance(val, int)` check. No version bump. Encoding uses `urlsafe_b64encode/decode`. Color fields (4 hex strings) are intentionally NOT bitmask-encoded (only ~14–20 URL chars savings, not worth complexity).
- `_TAB_CONTROLS` — dict mapping each `tab_id` → set of component IDs belonging to that tab. `_TAB_TO_PATH` — reverse of `_PATH_TO_TAB`.
- `_encode_snapshot(state_dict, tab_filter=None)` — pass `tab_filter=_TAB_CONTROLS[tab_id]` for single-tab links.
- `restore_from_url` callback (`prevent_initial_call=False`) decodes hash on page load → restores all controls.
- Snapshot lots override localStorage; "Restore my lots" button reverts.
- `link-history` store (localStorage) — deduplicates, up to 50 entries; each entry records `scope` and `tab`.
- Key stores: `snapshot-lots` (memory), `effective-lots` (memory), `link-history` (local), `loaded-hash-store` (memory).

### URL tab routing
- Visiting `/1`–`/9` navigates directly to a tab (clientside callback on `url.pathname`).
- Map: `/1`=bubble, `/2`=heatmap, `/3`=dca, `/4`=retire, `/5`=supercharge, `/6`=stack, `/7`=model_info, `/8`=faq, `/9`=citadel.
- `/7.N` opens Model Info accordion item N; `/8.N` opens FAQ item N (both 1-indexed in URL, 0-indexed internally).
- Routing logic lives in `callbacks/routing.py` (split from old `nav.py`). Uses `allow_duplicate=True` + `prevent_initial_call='initial_duplicate'`. **Never use `prevent_initial_call=False` with `allow_duplicate=True`** — Dash raises an error that crashes gunicorn (exit code 3).

### Live price ticker
- `dcc.Interval(id="price-interval", interval=20*60*1000)` fires every 20 min (5 × 4 min intervals).
- `update_price_ticker` callback fetches Binance (`api.binance.com/api/v3/ticker/price?symbol=BTCUSDT`), CoinGecko fallback. Outputs to `price-ticker` div (navbar), `btc-price-store` (memory Store), and `hm-entry-q` (keeps heatmap entry quantile in sync with ticker on every refresh).
- Ticker displays: `₿ $X` · `QY.Y%` (current quantile percentile) + 24h sparkline SVG (from CoinGecko). Mode toggle switches between USD and sats/$ display. Multi-model percentile cycling on tap: QR → BM → PL → LPPL → Exp → EF → U₁ (skips S2F — non-quantized). Each model's percentile is color-coded.
- `_startup_heatmap_defaults()` fetches price at module load → sets heatmap entry percentile default.
- `_interp_qr_price(q, t, qr_fits)` in `figures/common.py` — log-space interpolation between adjacent QR fits for arbitrary quantile (e.g. Q7.5%).
- Heatmap uses `live_price` from `btc-price-store` as entry price when `entry_yr == current_year`; falls back to model interpolation for historical entry years.
- **Binance is geo-blocked in the US** (HTTP 451) but works fine from the Hetzner server (Germany).

### Chart builders (`figures/`)

Split into a package with one module per chart type + shared helpers:

| Module | Chart |
|--------|-------|
| `figures/common.py` | Shared helpers: `_get_palette`, `_thermal_color`, `_build_thermal_colors`, `_dark_layout`, `_sim_layout`, `_finalize_chart`, `_apply_watermark`, `build_overlay_traces()`, `_resolve_model()`, edge annotations, MC overlay integration |
| `figures/bubble.py` | `build_bubble_figure(m, p)` — Bubble model + QR channels + overlay models (PL, S2F) |
| `figures/heatmap.py` | `build_heatmap_figure(m, p)` — CAGR heatmap (go.Heatmap) |
| `figures/dca.py` | `build_dca_figure(m, p)` — DCA accumulation simulation |
| `figures/retire.py` | `build_retire_figure(m, p)` — Retirement withdrawal simulation |
| `figures/supercharge.py` | `build_supercharge_figure(m, p)` — HODL Supercharger |
| `figures/citadel.py` | `build_citadel_figure(m, p)` — Citadel Planner |

### Price models & Display Models

Seven+ price models registered at startup in `_app_ctx.PRICE_MODELS`:
- **Bubble Model** (`"bub"`) — default, loaded from `model_data.pkl`
- **Quantile Regression** (`"qr"`) — standalone QR model
- **Power Law** (`"pl"`) — OLS fit to log-log data
- **LPPL** (`"lppl"`) — Log-Periodic Power Law
- **Exponential** (`"exp"`) — exponential fit
- **Empirical Floor** (`"ef"`) — conditional on `model_data_ef.pkl` existing
- **S2F (Stock-to-Flow)** (`"s2f"`) — alternative parameterization
- **U₁ (User Model)** (`"u1"`) — session-only, click-to-draw power law from two user-defined points (see below)

Per-tab model display:
- **Bubble tab**: `bub-model-show` checklist toggles overlay models on the bubble chart.
- **Heatmap tab**: pill bar carousel (`hm-active-model` Store) — one model active at a time; `hm-model-show` checklist exists in layout (for snapshot compat) but is hidden, replaced by the pill bar.
- **DCA / Retire / Supercharger / Citadel**: `{prefix}-model-show` checklist showing available models.

### Monte Carlo / Markov simulation

- MC features are **paid** (Lightning/on-chain via BTCPay). Controlled by `_app_ctx._HAS_MARKOV` flag.
- When `_HAS_MARKOV` is `False`, all MC controls are hidden via placeholder `dcc.Checklist` elements and "MC Simulation" is hidden from Display Models checklists.
- Pre-computed cache: ~45,000 scenarios covering different entry percentiles, time horizons, withdrawal amounts, inflation, stack sizes (~834 MB RAM at startup).
- MC controls appear on DCA, Retire, Heatmap, and Supercharger tabs.
- Layout: `layout/mc_controls.py` (reusable MC control panel). Callbacks: `callbacks/mc_controls.py`, `callbacks/mc_helpers.py`, `callbacks/mc_payment.py`, `callbacks/mc_upload.py`.

### Heatmap pill bar carousel

Multi-model heatmap switching via pill buttons (added in `a94a987`):
- Built dynamically from `_app_ctx.PRICE_MODELS` + optional MC in `layout/heatmap.py` `_hm_pill_bar()`.
- Pill buttons: Bubble Model (solid blue), Power Law, S2F, Monte Carlo (warning orange, if Markov available).
- One pill active at a time, stored in `hm-active-model` Store.
- Callback `_hm_pill_click()` updates active model; `_hm_pill_sync()` syncs pill outlines on snapshot restore / page load.

### User Model (U₁)
- Click-to-draw power law from two user-defined points (P1, P2) on the bubble chart.
- Context menu sets P1/P2 coordinates (year + price). Power law fitted through both points, then empirical residual quantization generates full quantile fan.
- Session-only (`user-model-store`, memory Store) — not persisted across page reloads.
- `callbacks/user_model.py`: draw/delete callbacks + injection of `u1` option into all `{prefix}-model-show` checklists.
- When drawn, `u1` is auto-selected in `bub-model-show`.

### Citadel Planner (tab 9)
- Multi-asset retirement simulation with BTC + cash + bonds + equities + real estate.
- Four sub-tabs: **Assets** (BTC stack, cash, reserves, investments), **Spending** (monthly amount, inflation, growth), **Rules** (rebalancing triggers, floor rules, Saylor Fortifier), **Simulation** (quantiles, asset growth mode, MC controls, chart toggles).
- Asset growth modes: "Fixed rates" or "Historical Regimes" (Markov-based).
- **"▶ Run Simulation"** button triggers computation (via Celery if available, in-process fallback via `engines/adapter.py`).
- Engine: `engines/citadel.py`. Figure builder: `figures/citadel.py`. Layout: `layout/citadel.py`. Callbacks: `callbacks/citadel_cb.py`.
- Historical data in `data/`: equity, bond, treasury CSV files + `asset_matrices.py` for correlation/return matrices.

### Colorblind palette system

Three-tier palette (Default / CB-RG / CB-Full) stored in `_app_ctx.PALETTES`. Navbar dropdown writes to `dcc.Store("palette-store", storage_type="local")`. Each chart callback passes `palette` key in the `p` params dict. Figure builders call `_get_palette(p)` to resolve colors. Palette choice is included in snapshot/share links via `_SNAPSHOT_CONTROLS`.

**Watermark**: `_LOGO_B64` (base64-encoded logo loaded at module startup) and `_apply_watermark(fig)` add the Quantoshi logo (bottom-right, 55% opacity, `sizex=0.07 sizey=0.12`) plus `"quantoshi.xyz"` text annotation to all exported figures. Called in all 6 chart builders before return.

Heatmap colorscale: all three modes use `_dense_colorscale()` — 256-point `rgb()` colorscale for browser compatibility. Diverging mode centers at 0% CAGR. The "Gradient steps" UI control is cosmetic (no longer affects rendering).

Heatmap chart title format: `Entry: {year}  {price}  ·  Q{percentile}%` — price first, then quantile, matching the navbar ticker format.

### Callback performance

**Per-tab render triggers**: Each chart tab has a `dcc.Store("{tab}-first-render")` initialized to 0. A single clientside callback watches `main-tabs.active_tab` and increments the matching tab's store. Chart callbacks use `Input("{tab}-first-render", "data")` instead of `Input("main-tabs", "active_tab")`, with `prevent_initial_call=True` — they ONLY fire when their trigger increments. Result: switching tabs fires exactly 1 chart callback (the active tab), not all 6.

**URL-based initial tab**: The layout is a function (`_serve_layout`) that reads `flask.request.path` to determine the initial `active_tab`. Visiting `/9` builds the layout with `active_tab="citadel"` — the bubble callback never fires. No wasted computation for tabs the user didn't request.

**Pre-injected figures / zero-callback tab switching**: `_serve_layout` pre-builds ALL tab figures from the L1 LRU cache (which prewarm populated) and injects them directly into the initial HTML. All `{tab}-first-render` stores start at `1`, not `0`. Switching tabs requires zero server round-trips — figures are already present in the DOM. Callbacks only fire when the user changes a control.

**`tab_resize.js`**: Calls `Plotly.Plots.resize()` when a tab becomes visible. Required because hidden tabs render at zero/wrong size when the browser hasn't painted them yet.

**`tab_dblclick.js`**: Double-clicking a tab header increments its `{tab}-first-render` store, triggering a full figure reload from server. Escape hatch if a figure looks stale.

**Background callbacks (Citadel)**: The Citadel "Run Simulation" button uses `background=True` with Dash's `DiskcacheManager`. The simulation runs in a separate process and does not block gunicorn workers. The button shows "⏳ Computing..." during long MC runs. Requires `diskcache`, `psutil`, `dill`, `multiprocess` in the environment.

**Redis socket pre-check**: At startup, `redis_available()` probes the Redis socket with a 0.2s timeout before attempting a full `ping`. Without this, a missing Redis instance causes an 8.6s connection timeout that blocks gunicorn worker startup.

**`prevent_initial_call` settings**:

| Callback | Setting | Why |
|---|---|---|
| All chart callbacks (bubble, heatmap, DCA, retire, SC, citadel) | `True` | Only fire via first-render trigger on tab visit |
| MC body toggles (5x) | N/A (clientside) | Zero server round-trips |
| SC mode/display toggles | N/A (clientside) | Same |
| Price ticker | `'initial_duplicate'` | Must fire once on load to populate price |

Constraint: `allow_duplicate=True` on outputs is incompatible with `prevent_initial_call=False` (crashes gunicorn). The first-render trigger pattern solves this — callbacks use `prevent_initial_call=True` and get triggered by the clientside store instead.

**Clientside callback pattern**: Trivial visibility toggles should be clientside callbacks (no server round-trip):
```python
_app_ctx.app.clientside_callback(
    "function(v) { return (v && v.length) ? {} : {display:'none'}; }",
    Output("component-id", "style"),
    Input("toggle-id", "value"),
)
```

### Figure cache (three layers)
- **L0 (pinned)**: Redis-backed persistent cache for default params. Fingerprint = `md5(model_fp + defaults_hash)`. 7-day TTL. Survives restarts. Defined in `cache.py`.
- **L1 (LRU)**: `@lru_cache(maxsize=8)` per tab (bubble, heatmap, DCA, retire, supercharge, citadel). Per-worker, in-memory.
- **L2 (Redis)**: Shared across all workers. No TTL — Redis LRU eviction handles pressure. Fingerprint invalidation on model data change.
- If Redis is unavailable, falls back to L1-only.
- `_q3(x)` (in `_app_ctx.py`): rounds floats to 3 significant figures for cache-friendly keys.
- `_quantize_params(p)`: applies `_q3` to all float params. **Exempts `selected_qs` and `exit_qs`** (must match `qr_fits` keys exactly).
- `_ALL_QS` filtered to Q0.1%–Q99.9% (extreme quantiles break `_q3` due to float rounding).
- Bubble cache key includes `date.today()` for natural daily TTL.
- `_prewarm_caches()` runs at worker startup (skipped in `DEV=1` mode). **Must be updated when tab defaults change** — canonical defaults in `tab_defaults.py`.
- **Cache key alignment**: The `*_defaults()` functions in `tab_defaults.py` must include ALL keys that callbacks add to the params dict (including `show_qr`, `show_mc`, `palette`, `user_model`, `sc_live_price`). This ensures the prewarm cache key matches the runtime callback cache key, yielding an L1 cache hit on first tab visit.

### Known gotchas

**`dbc.Input(type="number")` step/min validation**: HTML5 number inputs send `null` (Python `None`) when the typed value doesn't satisfy `value = min + n × step`. With `min=1, step=10`, the valid series is 1, 11, 21, ... — so common values like 100, 200, 1000 silently become `None` and callbacks fall back to defaults, appearing to do nothing.
- Rule: `min` must itself be a valid step value (i.e. `(min - base) % step == 0` where base=0 unless min is the anchor). Simplest safe choices: `step=1` for integer dollar amounts; `min=0` for BTC amounts with `step=0.001`; align `min` to be a multiple of `step` for decimal inputs.
- Current state: `dca-amount`, `ret-wd`, `sc-wd` use `step=1`; `hm-entry-q` uses `min=0.1, step=0.1`; `ret-infl`/`sc-infl` use `min=0, max=100, step=0.5`.
- **Also**: `max` is enforced the same way — values above `max` send `null`. Always keep `max` in sync with actual valid range.
- **Labels**: bounded inputs show their range/step in the label text, e.g. "Pt size (1–20)", "Inflation rate (0–100% / yr)".

**Falsy-zero in callbacks**: `float(x or default)` substitutes `default` when `x=0` because 0 is falsy. For any numeric input where 0 is a valid value (inflation rate, interest rate, etc.), use `float(x) if x is not None else default`. Affected fields fixed: `dca-sc-rate`, `sc-infl`. `ret-infl` uses `float(infl or 0)` which is safe since its fallback is also 0.

**Frequency options**: All frequency dropdowns (dca-freq, ret-freq, sc-freq, cp-freq) offer Daily/Weekly/Monthly/Quarterly/Annually. `FREQ_PPY` in `_app_ctx.py` maps these to 365/52/12/4/1. `FREQ_LABEL` maps to "/day"/"/wk"/"/mo"/"/qtr"/"/yr".

**Mobile portrait layout**: On small screens (`max-width: 767px`) columns stack vertically (controls below chart). The `dcc.Graph` inline `style="height:78vh"` must be overridden in CSS or it leaves a large blank gap above the controls. Fix in `style.css`: `[id$="-graph"] { height: 55vw !important; min-height: 280px !important; }` alongside the same rule on `.js-plotly-plot`. A mobile-only `↓ Scroll down to configure` hint is appended inside `_export_row()` (hidden on ≥768px via `d-md-none`), covering all 5 chart tabs automatically.

**Nginx JS caching**: `/_dash-component-suites/` URLs contain version hashes (immutable assets). nginx caches them for 7 days with `Cache-Control: public, max-age=604800, immutable`. Plotly.js is 4.7 MB (gzipped ~1.5 MB) — cached after the first load, not re-fetched on subsequent visits or deploys.

**Stale `/_dash-dependencies` between deploys**: Old browsers cache Dash's callback signature map. If the callback graph changes (new outputs added), cached clients send requests with old output-key hashes → server returns 500 → Dash marks those output components as errored → user interactions silently do nothing.
- Fix (already in place): `@server.after_request` hook sets `Cache-Control: no-cache` on `/_dash-layout` and `/_dash-dependencies`. Defined immediately after `server = app.server`.

**Versions**: Dash 4.0.0, DBC 2.0.4, React 18 (bundled with Dash 4).

### Production server
- **VPS**: Hetzner, IP `89.167.70.45`, SSH as `root`
- **App path**: `/opt/quantoshi/` (git clone of this repo)
- **Service**: `quantoshi.service` (systemd, gunicorn binds `127.0.0.1:8050`, 5 workers)
- **nginx**: reverse proxy with HTTPS via Let's Encrypt
- **Tor**: `tor@default`, hidden service at `/var/lib/tor/quantoshi/`
- **gunicorn** must be installed separately: `pip install gunicorn` (not in requirements.txt)
- **Log retention**: 27 days — `/etc/logrotate.d/nginx` and `/etc/logrotate.d/quantoshi` both set `rotate 27` with daily rotation. Covers nginx logs and gunicorn's `/var/log/quantoshi-access.log` + `/var/log/quantoshi-error.log`.

---

## Key Files

| File | Purpose |
|------|---------|
| `BitcoinPricesDaily.csv` | Daily BTC price data (read by notebook + web app + desktop app) |
| `archive/btc_app/model_data.pkl` | Precomputed QR fits + bubble composites (regenerated by Cell 3) |
| `archive/btc_app/btc_projections.spec` | PyInstaller spec — bundles pkl + CSV; excludes tkinter/jupyter/torch; **do not add `unittest` to excludes** (scipy dep) |
| `archive/btc_app/btc_projections.desktop` | Desktop entry for AppImage |
| `quantoshi_logo.png` | Master logo file |
| `run_web.sh` | Web app startup script (gunicorn or DEV mode) |
| `btc-web.service` | systemd unit template for local installs |
