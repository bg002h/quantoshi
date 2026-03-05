# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**Quantoshi** — Bitcoin price projection tools. Three components share the same underlying quantile regression bubble model:

1. **`SP.ipynb`** — Jupyter notebook with bubble model + quantile regression analysis, chart generation, and PowerPoint export.
2. **`btc_web/`** — Plotly Dash web app. Live at [quantoshi.xyz](https://quantoshi.xyz) and `u5dprelc4ti7xoczb5sbtye6qidlji2l6psmkx35anvxgjyqrkmu32ad.onion`.
3. **`btc_app/btc_projections.py`** — Standalone PyQt5 GUI app (5 interactive tabs) distributed as a Linux AppImage.

The notebook generates `btc_app/model_data.pkl`, which both the web app and the standalone app load at runtime.

---

## Workflow

**Never auto-deploy.** After making changes, stop at committing locally. Do NOT push to GitHub or SSH-deploy to the production server unless explicitly asked. The user will say "deploy to production" when ready to ship.

**Local test environment:** `DEV=1 bash run_web.sh` (hot-reload, single user) or `bash run_web.sh` (gunicorn). The local btc-web systemd service has been stopped; start the app manually when needed.

---

## Commands

### Run the notebook
```bash
/scratch/code/bitcoinprojections/btc_venv/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=120 SP.ipynb
```

### Run the standalone app directly (for testing)
```bash
cd /scratch/code/bitcoinprojections/btc_app
/scratch/code/bitcoinprojections/btc_venv/bin/python3 btc_projections.py
```

### Syntax check the app
```bash
/scratch/code/bitcoinprojections/btc_venv/bin/python3 -m py_compile \
    btc_app/btc_projections.py && echo "OK"
```

### Build the AppImage
```bash
cd /scratch/code/bitcoinprojections/btc_app
bash build_appimage.sh          # uses up to 18 CPUs
JOBS=8 bash build_appimage.sh   # override CPU count
```
Output: `btc_app/Quantoshi-x86_64.AppImage` (~140 MB)

### Full rebuild after notebook changes
```bash
# 1. Execute notebook (regenerates model_data.pkl)
/scratch/code/bitcoinprojections/btc_venv/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=120 SP.ipynb
# 2. Build AppImage
cd /scratch/code/bitcoinprojections/btc_app && bash build_appimage.sh
```

### Run the web app locally
```bash
bash run_web.sh           # gunicorn, 4 workers, port 8050
DEV=1 bash run_web.sh     # Dash dev server with hot-reload (single user)
PORT=8080 bash run_web.sh # custom port
```

### Syntax-check the web app
```bash
/scratch/code/bitcoinprojections/btc_venv/bin/python3 -m py_compile \
    btc_web/app.py btc_web/figures.py && echo "OK"
```

### Deploy to production (Hetzner VPS)
```bash
git push origin master
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && systemctl restart quantoshi"
```

---

## Notebook Architecture (`SP.ipynb`)

| Cell | Lines | Purpose |
|------|-------|---------|
| 0 | ~1390 | Bubble model — **do not modify** |
| 1 | ~1112 | QR config & chart generation — primary work cell |
| 2 | ~293  | PowerPoint export (`bitcoin_projections.pptx`) |
| 3 | ~93   | Export cell — writes `btc_app/model_data.pkl` |
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

## Standalone App Architecture (`btc_app/btc_projections.py`)

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
| `app.py` | Dash app — layout, all callbacks, snapshot/share logic |
| `figures.py` | Plotly chart builders — one function per tab |
| `assets/style.css` | Light theme (FLATLY) overrides + mobile layout |
| `assets/quantoshi_logo.png` | Logo — served as navbar image and favicon |
| `requirements.txt` | Python dependencies |

### Layout structure
`dbc.Navbar` (logo + "Quantoshi" in Palatino + "Stay dark, Anon →" + 🧅 Tor link + 📸 Share button with "▲ Cooler than you think") → `dbc.Tabs` (7 tabs):

| Tab | ID | Key controls |
|-----|----|-------------|
| Bubble + QR Overlay | `bubble` | Quantiles, axes scale/range, bubble composite, N future bubbles |
| CAGR Heatmap | `heatmap` | Entry/exit year+quantile, color modes (Segmented/DataScaled/Diverging) |
| BTC Accumulator | `dca` | DCA amount/frequency, year range, display mode, Stack-cellerator |
| BTC RetireMentator | `retire` | Withdrawal amount, inflation rate, year range |
| HODL Supercharger | `supercharge` | Mode A (fixed spending → depletion date) or Mode B (fixed depletion → max spending); 5 delay offsets, 2 chart layouts (single-quantile line or quantile bands) |
| Stack Tracker | `stack` | Lot management (add/delete/import/export JSON) |
| FAQ | `faq` | Static accordion — add entries to `_FAQ` list in app.py. 14 entries: Share, quantile regression, price prediction disclaimer, appearance, crossing projection lines, Power Law, bubble model, why I made this, podcast (porkopolis.io), direct tab linking (/1-/7), open source (BSD-2/GitHub/AppImage), data privacy (localStorage/27-day logs/onion), tip addresses, contact. Answers: plain strings or Dash components (`html.Span`/`html.A`/`html.Table`). Link color: `#1a6fa8` via `.accordion a` in style.css. |

### Tab defaults
| Tab | Notable defaults |
|-----|-----------------|
| Bubble | Q5% only, X scale=Log, N future bubbles=3, shade+show_data+show_today on (legend off). Pt size=2, Alpha=0.2. Panel order: scales, toggles, bubble, quantiles, pt size/alpha, stack, use lots. X range slider min=2010 (marks from '10), default value [2012, yr_now+4]. Auto Y checkbox (default on) rescales Y to fit selected quantiles at xmin/xmax. When Stack (BTC) is enabled, each quantile legend label gains `→ $X` showing the final USD stack value at the right x-range edge. |
| Heatmap | Entry year=current year, entry percentile=live BTC percentile (free numeric input 0–100%, NOT dropdown), exit years allow past. Entry price=live ticker when entry_yr==current year. Break1=0%, Break2=20%, Gradient Steps=32. |
| DCA | Default quantile Q50% only. dual_y+show_legend on. BTC-mode trace labels include final USD value in parentheses. Dual-y "USD Value (median)" always shows median USD across selected quantiles. **Stack-cellerator** ("Enter Saylor Mode" checkbox): borrows `dca-sc-loan` $, buys BTC lump sum upfront, reduces DCA by the loan payment. Dashed overlay traces per quantile. Stack-celeration factor (median SC / median DCA) shown in chart title + legend. Controls: `dca-sc-type` (interest_only/amortizing), `dca-sc-rate`, `dca-sc-term`, `dca-sc-repeats` (0=one loan, N=N extra cycles back-to-back), `dca-sc-entry-mode` (live/model/custom for cycle 0; cycles 1+ always use model price), `dca-sc-custom-price`, `dca-sc-tax` (capital gains % on BTC sold to repay; default 33%). **Loan cap**: `principal` is silently capped at `max_principal = amount*(1-(1+r)^-n)/r` (amortizing) or `amount/r` (interest-only) when r>0, ensuring pmt ≤ DCA amount always. Info panel notes when cap is applied. **Tax applies only to interest-only**: BTC sold at cycle end to repay principal; tax-adjusted sell = `principal/(price*(1-tax_rate))`. Amortizing repays principal in fiat — no BTC sold, tax has no effect. `outstanding` balance tracked per-period; deducted tax-adjusted at cycle end (interest-only) or post-loop (incomplete final cycle). **Rollover** (`dca-sc-rollover` checklist, interest-only only): repeat cycles skip BTC purchase (new loan pays off old, net zero BTC movement); cycle-end BTC sale skipped; single final repayment by post-loop deduction at simulation end price (with tax). Without rollover: each cycle independently buys BTC at start and sells at end. Rollover row hidden for amortizing. `dbc.Collapse` must NOT be used for SC body — use `html.Div(style={"display":"none"})` toggled via callback. Snapshot: 78 controls total. |
| Retire | year slider min=2024, default range 2031–2075, inflation=4%, log_y+dual_y+annotate on. Dual-y median same approach as DCA. |
| HODL Supercharger | Mode A, stack=1.0 BTC, delays=[0,1,2,4,8] yr, Monthly, inflation=4%, wd=$5000/mo, end_yr=2075, USD display, annotate+log_y+show_legend on. `sc-chart-layout` is a `dcc.Checklist` with single option "shade"; default `["shade"]` (bands on = layout 2). Display-q dropdown hidden when bands on. Median final value shown in legend for both layouts. |
| Stack Tracker | default lot Price=$69,420 |

### State and privacy
- Lot data lives in **browser `localStorage`** only — `dcc.Store(id='lots-store', storage_type='local')`.
- Nothing written server-side. Export via clientside JS blob download. Import via `dcc.Upload` + server-side base64 decode.
- Chart callbacks use `effective-lots` store (routes to snapshot lots or localStorage lots).

### Snapshot / Share feature
- `📸 Share` button → modal → **Scope** radio ("All tabs" / "Current tab only") → **Generate link** encodes control states + optional lots as gzip+base64 in URL hash.
- URL format: `host/N#q2:...` where N is the tab path (`/1`–`/7`), so tab routing fires independently of hash decode.
- **All tabs** scope: encodes all 77 controls (full fidelity). **Current tab only** scope: encodes only the active tab's controls via `tab_filter` — non-matching controls encode as `null` and fall back to defaults on restore (much shorter link).
- `_SNAPSHOT_CONTROLS` — list of 77 `(component_id, property)` tuples. Format: `#q2:...` current; `#q1:...` legacy (still decoded).
- **Checklist bitmask encoding**: All 20 checklist fields (5 quantile + 15 toggle/boolean) are stored as bitmask integers in new links via `_CHECKLIST_OPTIONS` dict (component ID → ordered list of possible values) + `_list_to_mask(val, opts)` / `_mask_to_list(mask, opts)` helpers. Quantile fields: 17-bit each, saves ~435 JSON chars. Toggle fields: saves ~224 JSON chars. **Old `q2` links stored plain lists** — decoder handles both via `isinstance(val, int)` check. No version bump. Encoding uses `urlsafe_b64encode/decode`. Color fields (4 hex strings) are intentionally NOT bitmask-encoded (only ~14–20 URL chars savings, not worth complexity).
- `_TAB_CONTROLS` — dict mapping each `tab_id` → set of component IDs belonging to that tab. `_TAB_TO_PATH` — reverse of `_PATH_TO_TAB`.
- `_encode_snapshot(state_dict, tab_filter=None)` — pass `tab_filter=_TAB_CONTROLS[tab_id]` for single-tab links.
- `restore_from_url` callback (`prevent_initial_call=False`) decodes hash on page load → restores all controls.
- Snapshot lots override localStorage; "Restore my lots" button reverts.
- `link-history` store (localStorage) — deduplicates, up to 50 entries; each entry records `scope` and `tab`.
- Key stores: `snapshot-lots` (memory), `effective-lots` (memory), `link-history` (local), `loaded-hash-store` (memory).

### URL tab routing
- Visiting `/1`–`/7` navigates directly to a tab (clientside callback on `url.pathname`).
- Map: `/1`=bubble, `/2`=heatmap, `/3`=dca, `/4`=retire, `/5`=supercharge, `/6`=stack, `/7`=faq.
- `/7.N` (e.g. `/7.3`) navigates to the FAQ tab AND opens question N (1-indexed). Clientside callback matches `/7.N` regex → "faq"; separate `open_faq_item` Python callback sets `faq-accordion.active_item` to `faq-{N-1}` (0-indexed). Accordion has `id="faq-accordion"`.
- Uses `allow_duplicate=True` + `prevent_initial_call='initial_duplicate'`. **Never use `prevent_initial_call=False` with `allow_duplicate=True`** — Dash raises an error that crashes gunicorn (exit code 3).

### Live price ticker
- `dcc.Interval(id="price-interval", interval=20*60*1000)` fires every 20 min.
- `update_price_ticker` callback fetches Binance (`api.binance.com/api/v3/ticker/price?symbol=BTCUSDT`), CoinGecko fallback. Outputs to `price-ticker` div (navbar), `btc-price-store` (memory Store), and `hm-entry-q` (keeps heatmap entry quantile in sync with ticker on every refresh).
- `_startup_heatmap_defaults()` fetches price at module load → sets heatmap entry percentile default.
- `_interp_qr_price(q, t, qr_fits)` in `figures.py` — log-space interpolation between adjacent QR fits for arbitrary quantile (e.g. Q7.5%).
- Heatmap uses `live_price` from `btc-price-store` as entry price when `entry_yr == current_year`; falls back to model interpolation for historical entry years.
- **Binance is geo-blocked in the US** (HTTP 451) but works fine from the Hetzner server (Germany).

### Chart builders (`figures.py`)
| Function | Chart |
|----------|-------|
| `build_bubble_figure(m, p)` | Bubble model + QR channels |
| `build_heatmap_figure(m, p)` | CAGR heatmap (go.Heatmap) |
| `build_dca_figure(m, p)` | DCA accumulation simulation |
| `build_retire_figure(m, p)` | Retirement withdrawal simulation |
| `build_supercharge_figure(m, p)` | HODL Supercharger — depletion curves (Mode A) or max-withdrawal bar (Mode B) per delay scenario |

Module-level constants in `figures.py`: `_DELAY_COLORS = ['#636EFA','#EF553B','#00CC96','#AB63FA','#FFA15A']`, `_DASH_STYLES = ['solid','dash','dot','dashdot','longdash']`.

**Watermark**: `_LOGO_B64` (base64-encoded logo loaded at module startup) and `_apply_watermark(fig)` add the Quantoshi logo (bottom-right, 55% opacity, `sizex=0.07 sizey=0.12`) plus `"quantoshi.xyz"` text annotation to all exported figures. Called in all 5 chart builders before return.

Heatmap colorscale: all three modes use `_dense_colorscale()` — 256-point `rgb()` colorscale for browser compatibility. Diverging mode centers at 0% CAGR. The "Gradient steps" UI control is cosmetic (no longer affects rendering).

Heatmap chart title format: `Entry: {year}  {price}  ·  Q{percentile}%` — price first, then quantile, matching the navbar ticker format.

### Known gotchas

**`dbc.Input(type="number")` step/min validation**: HTML5 number inputs send `null` (Python `None`) when the typed value doesn't satisfy `value = min + n × step`. With `min=1, step=10`, the valid series is 1, 11, 21, ... — so common values like 100, 200, 1000 silently become `None` and callbacks fall back to defaults, appearing to do nothing.
- Rule: `min` must itself be a valid step value (i.e. `(min - base) % step == 0` where base=0 unless min is the anchor). Simplest safe choices: `step=1` for integer dollar amounts; `min=0` for BTC amounts with `step=0.001`; align `min` to be a multiple of `step` for decimal inputs.
- Current state: `dca-amount`, `ret-wd`, `sc-wd` use `step=1`; `hm-entry-q` uses `min=0.1, step=0.1`; `ret-infl`/`sc-infl` use `min=0, max=100, step=0.5`.
- **Also**: `max` is enforced the same way — values above `max` send `null`. Always keep `max` in sync with actual valid range.
- **Labels**: bounded inputs show their range/step in the label text, e.g. "Pt size (1–20)", "Inflation rate (0–100% / yr)".

**Falsy-zero in callbacks**: `float(x or default)` substitutes `default` when `x=0` because 0 is falsy. For any numeric input where 0 is a valid value (inflation rate, interest rate, etc.), use `float(x) if x is not None else default`. Affected fields fixed: `dca-sc-rate`, `sc-infl`. `ret-infl` uses `float(infl or 0)` which is safe since its fallback is also 0.

**Frequency options**: All three frequency dropdowns (dca-freq, ret-freq, sc-freq) offer Daily/Weekly/Monthly/Quarterly/Annually. `FREQ_PPY` in figures.py maps these to 365/52/12/4/1. `freq_label` maps to "/day"/"/wk"/"/mo"/"/qtr"/"/yr".

**Mobile portrait layout**: On small screens (`max-width: 767px`) columns stack vertically (controls below chart). The `dcc.Graph` inline `style="height:78vh"` must be overridden in CSS or it leaves a large blank gap above the controls. Fix in `style.css`: `[id$="-graph"] { height: 55vw !important; min-height: 280px !important; }` alongside the same rule on `.js-plotly-plot`. A mobile-only `↓ Scroll down to configure` hint is appended inside `_export_row()` (hidden on ≥768px via `d-md-none`), covering all 5 chart tabs automatically.

**Stale `/_dash-dependencies` between deploys**: Old browsers cache Dash's callback signature map. If the callback graph changes (new outputs added), cached clients send requests with old output-key hashes → server returns 500 → Dash marks those output components as errored → user interactions silently do nothing.
- Fix (already in place): `@server.after_request` hook sets `Cache-Control: no-cache` on `/_dash-layout` and `/_dash-dependencies`. Defined immediately after `server = app.server`.

**Versions**: Dash 4.0.0, DBC 2.0.4, React 18 (bundled with Dash 4).

### Production server
- **VPS**: Hetzner, IP `89.167.70.45`, SSH as `root`
- **App path**: `/opt/quantoshi/` (git clone of this repo)
- **Service**: `quantoshi.service` (systemd, gunicorn binds `127.0.0.1:8050`, 4 workers)
- **nginx**: reverse proxy with HTTPS via Let's Encrypt
- **Tor**: `tor@default`, hidden service at `/var/lib/tor/quantoshi/`
- **gunicorn** must be installed separately: `pip install gunicorn` (not in requirements.txt)
- **Log retention**: 27 days — `/etc/logrotate.d/nginx` and `/etc/logrotate.d/quantoshi` both set `rotate 27` with daily rotation. Covers nginx logs and gunicorn's `/var/log/quantoshi-access.log` + `/var/log/quantoshi-error.log`.

---

## Key Files

| File | Purpose |
|------|---------|
| `BitcoinPricesDaily.csv` | Daily BTC price data (read by notebook + web app + desktop app) |
| `btc_app/model_data.pkl` | Precomputed QR fits + bubble composites (regenerated by Cell 3) |
| `btc_app/btc_projections.spec` | PyInstaller spec — bundles pkl + CSV; excludes tkinter/jupyter/torch; **do not add `unittest` to excludes** (scipy dep) |
| `btc_app/btc_projections.desktop` | Desktop entry for AppImage |
| `quantoshi_logo.png` | Master logo file |
| `run_web.sh` | Web app startup script (gunicorn or DEV mode) |
| `btc-web.service` | systemd unit template for local installs |
