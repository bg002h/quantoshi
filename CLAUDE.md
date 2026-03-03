# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Bitcoin price projection system with two components:
1. **`SP.ipynb`** — Jupyter notebook with bubble model + quantile regression analysis, chart generation, and PowerPoint export.
2. **`btc_app/btc_projections.py`** — Standalone PyQt5 GUI app (5 interactive tabs) distributed as a Linux AppImage.

The notebook generates `btc_app/model_data.pkl`, which the standalone app loads at runtime.

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

## Key Files

| File | Purpose |
|------|---------|
| `BitcoinPricesDaily.csv` | Daily BTC price data (read by notebook + app) |
| `btc_app/model_data.pkl` | Precomputed QR fits + bubble composites (regenerated by Cell 3) |
| `btc_app/btc_projections.spec` | PyInstaller spec — bundles pkl + CSV; excludes tkinter/jupyter/torch; **do not add `unittest` to excludes** (scipy dep) |
| `btc_app/btc_projections.desktop` | Desktop entry for AppImage |
