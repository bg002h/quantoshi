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
| BTC Accumulator | `dca` | DCA amount/frequency, year range, display mode |
| BTC RetireMentator | `retire` | Withdrawal amount, inflation rate, year range |
| HODL Supercharger | `supercharge` | Mode A (fixed spending → depletion date) or Mode B (fixed depletion → max spending); 5 delay offsets, 2 chart layouts (single-quantile line or quantile bands) |
| Stack Tracker | `stack` | Lot management (add/delete/import/export JSON) |
| FAQ | `faq` | Static accordion — add entries to `_FAQ` list in app.py. 14 entries: Share, quantile regression, price prediction disclaimer, appearance, crossing projection lines, Power Law, bubble model, why I made this, podcast (porkopolis.io), direct tab linking (/1-/7), open source (BSD-2/GitHub/AppImage), data privacy (localStorage/27-day logs/onion), tip addresses, contact. Answers: plain strings or Dash components (`html.Span`/`html.A`/`html.Table`). Link color: `#1a6fa8` via `.accordion a` in style.css. |

### Tab defaults
| Tab | Notable defaults |
|-----|-----------------|
| Bubble | Q5% only, X scale=Log, N future bubbles=3, shade+show_data+show_today on (legend off). Panel order: scales, toggles, bubble, quantiles, pt size/alpha, stack, use lots. X range slider min=2010 (marks from '10), default value [2012, yr_now+4]. Auto Y checkbox (default on) rescales Y to fit selected quantiles at xmin/xmax. |
| Heatmap | Entry year=current year, entry percentile=live BTC percentile (free numeric input 0–100%, NOT dropdown), exit years allow past. Entry price=live ticker when entry_yr==current year. Break1=0%, Break2=20%, Gradient Steps=32. |
| DCA | dual_y+show_legend on |
| Retire | year slider min=2024, default range 2031–2075, inflation=4%, log_y+dual_y+annotate on |
| HODL Supercharger | Mode A, stack=1.0 BTC, delays=[0,1,2,4,8] yr, Monthly, inflation=4%, wd=$5000/mo, end_yr=2075, USD display, annotate+log_y+show_legend on. `sc-chart-layout` is a `dcc.Checklist` with single option "shade"; default `["shade"]` (bands on = layout 2). Display-q dropdown hidden when bands on. Median final value shown in legend for both layouts. |
| Stack Tracker | default lot Price=$69,420 |

### State and privacy
- Lot data lives in **browser `localStorage`** only — `dcc.Store(id='lots-store', storage_type='local')`.
- Nothing written server-side. Export via clientside JS blob download. Import via `dcc.Upload` + server-side base64 decode.
- Chart callbacks use `effective-lots` store (routes to snapshot lots or localStorage lots).

### Snapshot / Share feature
- `📸 Share` button → modal → **Generate link** encodes all 67 control states + optional lots as gzip+base64 in URL hash (`#q2:...` current format; `#q1:...` legacy format still decoded for backward compat).
- `_SNAPSHOT_CONTROLS` in `app.py` — list of 67 `(component_id, property)` tuples defining what gets captured (47 original + 19 `sc-*` entries for HODL Supercharger + 1 `bub-auto-y`, inserted before `main-tabs` entry — backward compatible).
- `restore_from_url` callback (`prevent_initial_call=False`) decodes hash on page load → restores all controls.
- Snapshot lots override localStorage; "Restore my lots" button reverts.
- `link-history` store (localStorage) — deduplicates, up to 50 entries.
- Key stores: `snapshot-lots` (memory), `effective-lots` (memory), `link-history` (local), `loaded-hash-store` (memory).

### URL tab routing
- Visiting `/1`–`/7` navigates directly to a tab (clientside callback on `url.pathname`).
- Map: `/1`=bubble, `/2`=heatmap, `/3`=dca, `/4`=retire, `/5`=supercharge, `/6`=stack, `/7`=faq.
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
