# Web App Simplification — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the btc_web monolithic files (callbacks.py 3241 LOC, layout.py 2320 LOC, figures.py 2407 LOC) into ~25 focused modules of 100–400 lines each, extract duplicated logic into shared helpers, and delete dead code — all while keeping behavior identical and tests passing.

**Architecture:** Split each monolith into a Python package (directory with `__init__.py`) that re-exports the public API. Shared helpers (quantile line builder, model overlay builder, MC setup/finalize sandwich) are extracted into `common.py` within each package. Tests update their imports but don't change assertions.

**Tech Stack:** Python 3, Plotly Dash 4.0, DBC 2.0.4, pytest

**Branch:** `simplify-web` (created from `master`)

---

## File Structure

After refactoring, the btc_web directory will look like:

```
btc_web/
├── app.py                    (entry point — modified imports only)
├── _app_ctx.py               (unchanged)
├── utils.py                  (unchanged)
├── snapshot.py               (unchanged)
├── api.py                    (unchanged)
├── btcpay.py                 (unchanged)
├── mc_overlay.py             (unchanged)
├── mc_cache.py               (unchanged)
├── test_web.py               (updated imports only)
│
├── layout/
│   ├── __init__.py           (re-exports build_layout + constants used by callbacks)
│   ├── common.py             (shared helpers: _q_panel, _section_card, _ctrl_card, _row, _lbl,
│   │                          _export_row, _chart_toggles, _btc_usd_dropdown, _legend_pos_dropdown,
│   │                          _chart_tab_layout, _year_range_slider, _freq_dropdown,
│   │                          _freq_warning_modal, _model_show_checklist, _shared_settings_card,
│   │                          _tab_hints, _q_options, style constants)
│   ├── bubble.py             (_bubble_controls, _bubble_tab)
│   ├── heatmap.py            (_heatmap_controls, _heatmap_tab)
│   ├── sim_tabs.py           (_accum_withdraw_controls, _dca_controls, _stackcelerator_controls,
│   │                          _retire_controls, _dca_tab, _retire_tab)
│   ├── supercharge.py        (_supercharge_controls, _supercharge_tab)
│   ├── stack.py              (_stack_tracker_tab)
│   ├── faq.py                (_FAQ list, _faq_tab)
│   ├── splash.py             (_SPLASH_QUOTES, _splash_quote_index, _SPLASH_QUOTES_JS, splash modal)
│   └── mc_controls.py        (_bold_opts, _regime_options, _mc_controls,
│                              _MC_CACHED_START_YRS, _MC_CACHED_YEARS, _MC_CACHED_ENTRY_QS,
│                              _MC_PRICE_CACHED, _MC_PRICE_LIVE, _MC_ENTRY_Q_OPTIONS,
│                              _MC_ENTRY_Q_OPTIONS_ADV, _MC_REGIME_OPTIONS_5)
│
├── callbacks/
│   ├── __init__.py           (register_all_callbacks — imports all submodules to trigger @callback)
│   ├── coerce.py             (_ci, _cf, _format_lots_for_table — shared across all callback modules)
│   ├── mc_helpers.py         (_coerce_mc, _build_mc_params, _mc_payment_check, _mc_setup,
│   │                          _mc_finalize, _mc_status, _ghost_match, _unblocked_val,
│   │                          _strip_free_paths)
│   ├── charts.py             (update_bubble, auto_bubble_yrange, update_heatmap, update_dca,
│   │                          update_retire, update_supercharge — the 6 main chart callbacks)
│   ├── mc_controls.py        (_toggle_dca_sc_body, MC body/advanced toggles, regime options,
│   │                          year sync, freq unlock, cost calculation, years options,
│   │                          _restore_mc — loop-generated MC UI callbacks)
│   ├── mc_payment.py         (_mc_payment_initiate, _quant_proceed, _quant_cancel,
│   │                          _mc_payment_cancel, _mc_modal_dismiss)
│   ├── mc_upload.py          (_parse_mc_upload, _extract_mc_key_val, _register_mc_upload,
│   │                          _TAB_LABELS, _MC_UPLOAD_FIELDS, mc download clientside callbacks)
│   ├── sc_loan.py            (_toggle_custom_price_row, _toggle_rollover_row, update_sc_info,
│   │                          _SAYLOR_QUOTES, _SAYLOR_QUOTES_JS)
│   ├── lots.py               (preview_percentile, manage_lots, sync_table_on_load, _lots_summary)
│   ├── nav.py                (toggle_sc_mode, toggle_sc_display_q, open_faq_item,
│   │                          toggle_share_modal, tab routing clientside callbacks,
│   │                          _PATH_TO_TAB, _TAB_CONTROLS, _TAB_TO_PATH)
│   ├── snapshot_cb.py        (restore_from_url, manage_snapshot, update_effective_lots,
│   │                          update_snapshot_banner, restore_my_lots, render_link_history,
│   │                          clear_history, apply_hm_palette, generate_share_qr,
│   │                          _add_snapshot_entry, _COLORSCALES)
│   └── ticker.py             (update_price_ticker)
│
├── figures/
│   ├── __init__.py           (re-exports all build_* functions + FREQ_PPY + _FREQ_STEP_DAYS
│   │                          + _LOGO_B64_ALL + _apply_watermark + _build_thermal_colors
│   │                          + _price_tickvals)
│   ├── common.py             (constants, _apply_sans_typography, _thermal_color,
│   │                          _build_thermal_colors, _add_glow_trace, _fmt_q_label,
│   │                          _error_figure, _apply_log_y, _stagger_depletion_annots,
│   │                          _build_freq_config, _build_time_array, _get_starting_stack,
│   │                          _sim_layout, _apply_mc_overlay, _dark_layout, _year_ticks,
│   │                          _build_qr_config_text, _build_mc_config_text,
│   │                          _apply_config_annotation, _apply_mc_premium,
│   │                          _apply_watermark, _finalize_chart,
│   │                          _lerp_hex, _dense_colorscale, _hex_alpha,
│   │                          _build_quantile_traces, _add_model_overlays,
│   │                          _clip_mc_traces, _post_mc_overlay, _find_mc_median_trace,
│   │                          _mc_median_annot, _fmt_short, _edge_text_trace,
│   │                          _resolve_edge_annotations, FREQ_PPY, _FREQ_STEP_DAYS,
│   │                          _LOGO_B64, _LOGO_B64_ALL)
│   ├── bubble.py             (build_bubble_figure)
│   ├── heatmap.py            (_seg_colorscale, _heatmap_colorscale, _heatmap_cell_annots,
│   │                          build_heatmap_figure, build_mc_heatmap_figure)
│   ├── dca.py                (_dca_sc_overlay, build_dca_figure)
│   ├── retire.py             (build_retire_figure)
│   └── supercharge.py        (_sc_mode_b, build_supercharge_figure,
│                              _DELAY_COLORS, _ANNOT_COLORS, _DASH_STYLES)
│
└── assets/                   (unchanged)
    ├── style.css
    ├── drawer.js
    └── *.png
```

Top-level repo change:
```
btc_app/  →  archive/btc_app/     (moved, not deleted)
```

---

## Chunk 1: Setup and Archive

### Task 1: Create branch and archive desktop app

**Files:**
- Move: `btc_app/` → `archive/btc_app/`

- [ ] **Step 1: Create the simplify-web branch**

```bash
cd /Users/bcg/Desktop/btc_test/quantoshi
git checkout -b simplify-web
```

- [ ] **Step 2: Move btc_app to archive**

```bash
mkdir -p archive
git mv btc_app archive/btc_app
```

- [ ] **Step 3: Commit**

```bash
git add archive/btc_app
git commit -m "Move btc_app/ to archive/ — desktop app on back burner"
```

---

## Chunk 2: Split layout.py into layout/ package

### Task 2: Create layout/common.py with shared helpers

**Files:**
- Create: `btc_web/layout/__init__.py`
- Create: `btc_web/layout/common.py`

- [ ] **Step 1: Create layout package directory**

```bash
mkdir -p btc_web/layout
```

- [ ] **Step 2: Create layout/common.py**

Extract from `layout.py` lines 17–29 (imports), 31–41 (style constants), 43–148 (all shared helpers), 159–265 (dropdown/model helpers), 268–328 (shared settings/stack), 373–385 (_tab_hints). Also extract `_SECTION_ICONS` (lines 78–89), `_LEGEND_POS_OPTIONS` (130–136), `_TAB_HINTS` (333–371).

This file contains every helper function used by 2+ tab builders:
- `_q_options`, `_q_panel`, `_ctrl_card`, `_section_card`, `_row`, `_lbl`, `_export_row`
- `_chart_toggles`, `_btc_usd_dropdown`, `_legend_pos_dropdown`
- `_chart_tab_layout`, `_year_range_slider`, `_freq_dropdown`
- `_freq_warning_modal`, `_model_show_checklist`
- `_shared_settings_card`, `_tab_hints`
- Style constants: `_STYLE_HIDDEN`, `_STYLE_HINT`, `_STYLE_GRAPH_H`, `_STYLE_COLOR_H`, etc.
- `_SECTION_ICONS`, `_LEGEND_POS_OPTIONS`, `_TAB_HINTS`, `_BTC_ORANGE`

Imports needed:
```python
from dash import dcc, html
import dash_bootstrap_components as dbc
import _app_ctx
from snapshot import _SNAPSHOT_CONTROLS
```

- [ ] **Step 3: Verify layout/common.py parses**

```bash
cd /Users/bcg/Desktop/btc_test/quantoshi
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from layout.common import _section_card; print('OK')"
```

### Task 3: Create layout/mc_controls.py

**Files:**
- Create: `btc_web/layout/mc_controls.py`

- [ ] **Step 1: Create layout/mc_controls.py**

Extract from `layout.py`:
- `_bold_opts` (lines 674–681)
- `_regime_options` (lines 699–702)
- `_mc_controls` (lines 706–915) — the big MC control builder
- Module-level MC constants: `_MC_PRICE_CACHED`, `_MC_PRICE_LIVE`, `_MC_START_YR_OPTIONS`, `_MC_ENTRY_Q_OPTIONS`, `_MC_ENTRY_Q_OPTIONS_ADV`, `_MC_YEARS_OPTIONS`, `_MC_WD_OPTIONS`, `_MC_INFL_OPTIONS`, `_MC_REGIME_OPTIONS_5`, `_MC_CACHED_START_YRS`, `_MC_CACHED_YEARS`, `_MC_CACHED_ENTRY_QS`

Imports needed:
```python
from dash import dcc, html
import dash_bootstrap_components as dbc
import _app_ctx
from layout.common import _section_card, _ctrl_card, _row, _lbl, _STYLE_HIDDEN
from mc_cache import (CACHED_START_YRS, WD_AMOUNTS, ENTRY_PCT_BINS,
                      MC_YEARS_OPTIONS, INFL_OPTIONS)
from mc_overlay import bin_regime_labels
```

- [ ] **Step 2: Verify it parses**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from layout.mc_controls import _mc_controls; print('OK')"
```

### Task 4: Create layout/bubble.py, layout/heatmap.py

**Files:**
- Create: `btc_web/layout/bubble.py`
- Create: `btc_web/layout/heatmap.py`

- [ ] **Step 1: Create layout/bubble.py**

Extract `_bubble_controls` (lines 392–491) and `_bubble_tab` (lines 494–495).

Imports from `layout.common`: `_q_panel`, `_section_card`, `_ctrl_card`, `_row`, `_lbl`, `_export_row`, `_legend_pos_dropdown`, `_chart_tab_layout`, `_tab_hints`, `_STYLE_HIDDEN`.
Imports from `layout.mc_controls`: `_model_show_checklist` (if bubble uses it — check: bubble does NOT use _model_show_checklist, it only uses _q_panel and custom controls).

- [ ] **Step 2: Create layout/heatmap.py**

Extract `_heatmap_controls` (lines 502–604) and `_heatmap_tab` (lines 607–673).

Imports from `layout.common`: `_section_card`, `_ctrl_card`, `_row`, `_lbl`, `_export_row`, `_q_panel`, `_model_show_checklist`, `_tab_hints`, `_STYLE_HIDDEN`, `_STYLE_HINT`, `_STYLE_COLOR_H`.
Imports from `layout.mc_controls`: `_mc_controls`.

Note: `_model_show_checklist` is defined at layout.py line 248, which falls in the common.py extraction range (lines 43–265). Import it from `layout.common`, NOT `layout.mc_controls`.

- [ ] **Step 3: Verify both parse**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from layout.bubble import _bubble_tab; from layout.heatmap import _heatmap_tab; print('OK')"
```

### Task 5: Create layout/sim_tabs.py (DCA + Retire)

**Files:**
- Create: `btc_web/layout/sim_tabs.py`

- [ ] **Step 1: Create layout/sim_tabs.py**

Extract:
- `_accum_withdraw_controls` (lines 922–952) — shared builder for DCA & Retire
- `_dca_controls` (lines 955–970)
- `_stackcelerator_controls` (lines 973–1028)
- `_retire_controls` (lines 1039–1053)
- `_dca_tab` (lines 1031–1032)
- `_retire_tab` (lines 1056–1057)

Imports from `layout.common`: `_section_card`, `_ctrl_card`, `_row`, `_lbl`, `_q_panel`, `_chart_toggles`, `_btc_usd_dropdown`, `_legend_pos_dropdown`, `_model_show_checklist`, `_chart_tab_layout`, `_year_range_slider`, `_shared_settings_card`, `_tab_hints`, `_STYLE_HIDDEN`, `_STYLE_HINT`.
Imports from `layout.mc_controls`: `_mc_controls`.

- [ ] **Step 2: Verify it parses**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from layout.sim_tabs import _dca_tab, _retire_tab; print('OK')"
```

### Task 6: Create layout/supercharge.py, layout/stack.py

**Files:**
- Create: `btc_web/layout/supercharge.py`
- Create: `btc_web/layout/stack.py`

- [ ] **Step 1: Create layout/supercharge.py**

Extract `_supercharge_controls` (lines 1064–1157) and `_supercharge_tab` (lines 1160–1161).

- [ ] **Step 2: Create layout/stack.py**

Extract `_stack_tracker_tab` (lines 1492–1567).

- [ ] **Step 3: Verify both parse**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from layout.supercharge import _supercharge_tab; from layout.stack import _stack_tracker_tab; print('OK')"
```

### Task 7: Create layout/faq.py, layout/splash.py

**Files:**
- Create: `btc_web/layout/faq.py`
- Create: `btc_web/layout/splash.py`

- [ ] **Step 1: Create layout/faq.py**

Extract `_FAQ` list (lines 1168–1455) and `_faq_tab` (lines 1462–1484).

- [ ] **Step 2: Create layout/splash.py**

Extract `_SPLASH_QUOTES` (lines 1593–1902), `_splash_quote_index` (lines 1904–1913), `_SPLASH_IDX`, `_SPLASH_Q`, `_SPLASH_A`, `_GENESIS_QUOTE`, `_SPLASH_QUOTES_JS` (lines 1915–1928), and the splash modal builder.

- [ ] **Step 3: Verify both parse**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from layout.faq import _FAQ, _faq_tab; from layout.splash import _SPLASH_QUOTES, _SPLASH_QUOTES_JS; print('OK')"
```

### Task 8: Create layout/__init__.py and retire old layout.py

**Files:**
- Create: `btc_web/layout/__init__.py`
- Delete: `btc_web/layout.py` (replaced by package)

- [ ] **Step 1: Create layout/__init__.py**

Re-export everything that callbacks.py and app.py import from layout:

```python
"""Layout package — re-exports public API for backward compatibility."""

from layout.common import (
    _STYLE_HIDDEN, _STYLE_HINT, _STYLE_GRAPH_H, _STYLE_COLOR_H,
    _freq_warning_modal,
)
# Note: _STYLE_ADDR_CELL and _STYLE_ADDR_CODE are only used within
# layout/faq.py — no need to re-export them here.
from layout.mc_controls import (
    _bold_opts, _regime_options,
    _MC_CACHED_START_YRS, _MC_CACHED_YEARS, _MC_CACHED_ENTRY_QS,
    _MC_PRICE_CACHED, _MC_PRICE_LIVE,
    _MC_ENTRY_Q_OPTIONS, _MC_ENTRY_Q_OPTIONS_ADV,
)
from layout.faq import _FAQ
from layout.splash import _SPLASH_QUOTES, _SPLASH_QUOTES_JS
from layout.bubble import _bubble_tab
from layout.heatmap import _heatmap_tab
from layout.sim_tabs import _dca_tab, _retire_tab
from layout.supercharge import _supercharge_tab
from layout.stack import _stack_tracker_tab
from layout.faq import _faq_tab
from layout.splash import build_splash_modal


def build_layout(M):
    """Assemble the full app layout. Called from app.py."""
    # Move the main layout assembly (old layout.py lines 1930–2319) here,
    # importing tab builders from submodules.
    ...
```

The `build_layout` function (or equivalent — whatever the main assembly at lines 1930–2319 does) lives here. It calls each `_*_tab()` function and wires up the navbar, stores, intervals, and modals.

- [ ] **Step 2: Delete the old monolithic layout.py**

```bash
git rm btc_web/layout.py
```

- [ ] **Step 3: Verify the full layout package loads**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from layout import _SPLASH_QUOTES, _FAQ, _bold_opts; print('OK')"
```

- [ ] **Step 4: Commit the layout split**

```bash
git add btc_web/layout/
git rm btc_web/layout.py
git commit -m "Split layout.py (2320 LOC) into layout/ package (9 modules)"
```

---

## Chunk 3: Split figures.py into figures/ package

### Task 9: Create figures/common.py with shared helpers + new extracted helpers

**Files:**
- Create: `btc_web/figures/__init__.py`
- Create: `btc_web/figures/common.py`

- [ ] **Step 1: Create figures package directory**

```bash
mkdir -p btc_web/figures
```

- [ ] **Step 2: Create figures/common.py**

Extract from `figures.py`:
- All imports (lines 19–45)
- All module-level constants (lines 48–85, 100–113, 173–174, 315–348)
- All shared helpers (lines 88–573, 576–608, 920–923, 1367–1635)
- `FREQ_PPY`, `_FREQ_STEP_DAYS` re-exports (lines 1842–1843)
- Logo loading code (lines 315–335)

**NEW extracted helper — `_build_quantile_traces()`:**

Extract the duplicated quantile-line-building loop (appears in bubble:684–699, dca:1665–1688, retire:1874–1896) into a shared function:

```python
def _build_quantile_traces(
    traces, model, sel_qs, t_arr, y_fn, label_fn,
    *, colors=None, glow=True, line_width=_QR_LINE_WIDTH,
):
    """Build glow + scatter traces for each quantile.

    Args:
        traces: list to append to (modified in place)
        model: price model with .fits and .colors
        sel_qs: sorted list of quantile floats
        t_arr: x-axis time array
        y_fn: callable(q, prices) -> y_values array
        label_fn: callable(q, y_vals) -> legend label string
        colors: optional dict q->hex_color (e.g. thermal); falls back to model.colors
        glow: whether to add glow shadow traces
        line_width: trace line width
    """
    for q in sel_qs:
        if q not in model.fits:
            continue
        prices = model.price_at(q, np.maximum(t_arr, 0.5))
        y_vals = y_fn(q, prices)
        lbl = label_fn(q, y_vals)
        col = (colors or {}).get(q, model.colors.get(q, "#888888"))
        if glow:
            _add_glow_trace(traces, t_arr, y_vals, col)
        traces.append(go.Scatter(
            x=list(t_arr), y=list(y_vals),
            mode="lines", name=lbl,
            line=dict(color=col, width=line_width),
        ))
```

**NEW extracted helper — `_add_model_overlays()`:**

Extract the duplicated model overlay loop (appears in bubble:702–735, dca:1691–1732, retire:1915–1956) into a shared function:

```python
def _add_model_overlays(
    traces, p, sel_qs, t_arr, y_fn, label_fn,
    *, line_width=_QR_LINE_WIDTH * 0.8,
):
    """Add alternative model overlay traces (Power Law, S2F, etc.).

    Args:
        traces: list to append to
        p: params dict (reads p["active_models"])
        sel_qs: selected quantiles
        t_arr: x-axis time array
        y_fn: callable(model, q, prices) -> y_values for quantized models
        label_fn: callable(model_name, q_or_none, y_vals) -> legend label
        line_width: trace width
    """
    for model_key in p.get("active_models", []):
        mdl = _app_ctx.PRICE_MODELS.get(model_key)
        if not mdl:
            continue
        ts_clamped = np.maximum(t_arr, 0.5)
        if mdl.quantized:
            for q in sel_qs:
                if q not in mdl.fits:
                    continue
                prices = mdl.price_at(q, ts_clamped)
                y_vals = y_fn(mdl, q, prices)
                lbl = label_fn(mdl.name, q, y_vals)
                col = mdl.colors.get(q, "#888888")
                traces.append(go.Scatter(
                    x=list(t_arr), y=list(y_vals), mode="lines", name=lbl,
                    line=dict(color=col, width=line_width, dash=mdl.dash_style),
                    legendgroup=mdl.short_name,
                    legendgrouptitle_text=mdl.name,
                ))
        else:
            prices = mdl.price_at(0.5, ts_clamped)
            y_vals = y_fn(mdl, 0.5, prices)
            lbl = label_fn(mdl.name, None, y_vals)
            traces.append(go.Scatter(
                x=list(t_arr), y=list(y_vals), mode="lines", name=lbl,
                line=dict(color="#8B4513", width=line_width, dash=mdl.dash_style),
                legendgroup=mdl.short_name,
            ))
```

- [ ] **Step 3: Include all used helpers**

Include `_price_tickvals` (lines 576–579, used by `build_bubble_figure` at line 850) and `_apply_mc_xlabel` (lines 466–471, used by `build_mc_heatmap_figure` at line 1238) in common.py. Both are NOT dead code.

- [ ] **Step 4: Verify common.py parses**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from figures.common import _build_quantile_traces, _add_model_overlays, _finalize_chart; print('OK')"
```

### Task 10: Create figures/bubble.py

**Files:**
- Create: `btc_web/figures/bubble.py`

- [ ] **Step 1: Create figures/bubble.py**

Extract `build_bubble_figure` (lines 646–917). Refactor to use `_build_quantile_traces()` and `_add_model_overlays()` from common.py, replacing the inline loops at lines 684–735.

The bubble-specific `y_fn` is simply `lambda q, prices: prices * (stack if stack > 0 else 1)`.
The bubble-specific `label_fn` appends `→ $X` when stack > 0.
Pass `colors=_build_thermal_colors(sel_qs)` for thermal palette.

Imports:
```python
import numpy as np
import plotly.graph_objects as go
from btc_core import yr_to_t, today_t, fmt_price
import _app_ctx
from figures.common import (
    _build_thermal_colors, _add_glow_trace, _fmt_q_label,
    _build_quantile_traces, _add_model_overlays,
    _year_ticks, _dark_layout, _apply_sans_typography,
    _apply_config_annotation, _apply_watermark, _SANS_FONT,
    _QR_LINE_WIDTH, _TODAY_LINE_COLOR, _TODAY_LINE_WIDTH,
    _TODAY_LINE_OPACITY, _TODAY_GLOW_WIDTH, _TODAY_GLOW_OPACITY,
    _FONT_TITLE, _SHADE_ALPHA, _GLOW_ALPHA, _GLOW_WIDTH,
)
```

- [ ] **Step 2: Verify it parses**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from figures.bubble import build_bubble_figure; print('OK')"
```

### Task 11: Create figures/heatmap.py

**Files:**
- Create: `btc_web/figures/heatmap.py`

- [ ] **Step 1: Create figures/heatmap.py**

Extract:
- `_seg_colorscale` (lines 611–641)
- `_heatmap_colorscale` (lines 929–952)
- `_heatmap_cell_annots` (lines 955–1031)
- `build_heatmap_figure` (lines 1034–1150)
- `build_mc_heatmap_figure` (lines 1153–1240)

Imports from common: `_fmt_q_label`, `_lerp_hex`, `_dense_colorscale`, `_apply_config_annotation`, `_apply_mc_premium`, `_apply_mc_xlabel` (used by `build_mc_heatmap_figure` at line 1238), `_apply_watermark`, `_SANS_FONT`, `_FONT_TITLE`, `_HM_TEXT_THRESHOLD`, `_COLORSCALE_STEPS`.

- [ ] **Step 2: Verify it parses**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from figures.heatmap import build_heatmap_figure, build_mc_heatmap_figure; print('OK')"
```

### Task 12: Create figures/dca.py, figures/retire.py, figures/supercharge.py

**Files:**
- Create: `btc_web/figures/dca.py`
- Create: `btc_web/figures/retire.py`
- Create: `btc_web/figures/supercharge.py`

- [ ] **Step 1: Create figures/dca.py**

Extract:
- `_dca_sc_overlay` (lines 1245–1364)
- `build_dca_figure` (lines 1640–1838)

Refactor `build_dca_figure` to use `_build_quantile_traces()` and `_add_model_overlays()` from common. The DCA-specific `y_fn` computes cumulative BTC and stores into `all_btc_vals`/`all_usd_vals` dicts as a side effect — so the refactored version may need a thin wrapper that accumulates state, or keep the loop inline if the side effects make extraction awkward. Use judgment: if the side effects (populating dicts) make `_build_quantile_traces` not cleanly applicable, keep the loop but extract just the trace-creation part.

- [ ] **Step 2: Create figures/retire.py**

Extract `build_retire_figure` (lines 1848–2007). Same refactoring approach as DCA.

- [ ] **Step 3: Create figures/supercharge.py**

Extract:
- `_sc_mode_b` (lines 2316+)
- `build_supercharge_figure` (lines 2017–2313)
- Constants: `_DELAY_COLORS`, `_ANNOT_COLORS`, `_DASH_STYLES` (lines 2012–2014)

- [ ] **Step 4: Verify all parse**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from figures.dca import build_dca_figure; from figures.retire import build_retire_figure; from figures.supercharge import build_supercharge_figure; print('OK')"
```

### Task 13: Create figures/__init__.py and retire old figures.py

**Files:**
- Create: `btc_web/figures/__init__.py`
- Delete: `btc_web/figures.py`

- [ ] **Step 1: Create figures/__init__.py**

Re-export the public API that other modules import:

```python
"""Figures package — re-exports public API."""

from figures.common import (
    FREQ_PPY, _FREQ_STEP_DAYS, _LOGO_B64_ALL, _apply_watermark,
    _build_qr_config_text, _build_mc_config_text, _apply_config_annotation,
    _build_thermal_colors, _price_tickvals,
)
from figures.bubble import build_bubble_figure
from figures.heatmap import build_heatmap_figure, build_mc_heatmap_figure
from figures.dca import build_dca_figure
from figures.retire import build_retire_figure
from figures.supercharge import build_supercharge_figure
```

Note: `_build_thermal_colors` is imported by app.py (line 150). `_price_tickvals` is used by `build_bubble_figure`. Add any other symbols that grep reveals are imported externally from `figures`.

- [ ] **Step 2: Delete the old monolithic figures.py**

```bash
git rm btc_web/figures.py
```

- [ ] **Step 3: Verify the full figures package loads**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from figures import build_bubble_figure, build_dca_figure, FREQ_PPY; print('OK')"
```

- [ ] **Step 4: Commit the figures split**

```bash
git add btc_web/figures/
git rm btc_web/figures.py
git commit -m "Split figures.py (2407 LOC) into figures/ package (7 modules), extract shared quantile/model helpers"
```

---

## Chunk 4: Split callbacks.py into callbacks/ package

### Task 14: Create callbacks/coerce.py and callbacks/mc_helpers.py

**Files:**
- Create: `btc_web/callbacks/__init__.py`
- Create: `btc_web/callbacks/coerce.py`
- Create: `btc_web/callbacks/mc_helpers.py`

- [ ] **Step 1: Create callbacks package directory**

```bash
mkdir -p btc_web/callbacks
```

- [ ] **Step 2: Create callbacks/coerce.py**

Extract from `callbacks.py`:
- `_format_lots_for_table` (line 60)
- `_ci` (line 73) — coerce to int
- `_cf` (line 83) — coerce to float

These are zero-dependency helpers used everywhere.

```python
"""Coercion helpers shared across all callback modules."""

def _ci(val, default=0, lo=None, hi=None):
    ...

def _cf(val, default=0.0, lo=None, hi=None):
    ...

def _format_lots_for_table(lots, model=None):
    ...
```

- [ ] **Step 3: Create callbacks/mc_helpers.py**

Extract from `callbacks.py` lines 93–460:
- `_coerce_mc` (line 93)
- `_strip_free_paths` (line 108)
- `_build_mc_params` (line 116)
- `_mc_payment_check` (line 160)
- `_mc_setup` (line 223)
- `_mc_finalize` (line 291)
- `_mc_status` (line 422)
- `_ghost_match` (line 434)
- `_unblocked_val` (line 449)

Imports:
```python
import dash
from callbacks.coerce import _ci, _cf
import _app_ctx
import btcpay
from mc_cache import (MC_DEFAULT_YEARS, MC_DEFAULT_ENTRY_Q, MC_DEFAULT_START_YR,
                      MC_YEARS_OPTIONS, MC_BINS, MC_SIMS, MC_FREQ)
```

- [ ] **Step 4: Verify both parse**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from callbacks.coerce import _ci, _cf; from callbacks.mc_helpers import _mc_setup; print('OK')"
```

### Task 15: Create callbacks/charts.py

**Files:**
- Create: `btc_web/callbacks/charts.py`

- [ ] **Step 1: Create callbacks/charts.py**

Extract the 6 main chart callbacks:
- `update_bubble` (line 320)
- `auto_bubble_yrange` (line 379)
- `update_heatmap` (line 462)
- `update_dca` (line 619)
- `update_retire` (line 1632)
- `update_supercharge` (line 1721)

Also extract clientside callbacks: `_MC_MATCH_JS_TPL` (line 831), `_MC_EXTEND_YR_JS` (line 915), `_PPY_JS` (line 954), `_MC_MAX_DATAPOINTS` (line 985).

Imports:
```python
import dash
from dash import Input, Output, State, ctx, callback, no_update
from callbacks.coerce import _ci, _cf
from callbacks.mc_helpers import _mc_setup, _mc_finalize, _mc_status
from utils import (_get_bubble_fig, _get_dca_fig, _get_retire_fig,
                   _get_supercharge_fig, _get_heatmap_fig, _get_mc_heatmap_fig)
import _app_ctx
```

- [ ] **Step 2: Verify it parses**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from callbacks.charts import update_bubble; print('OK')"
```

### Task 16: Create callbacks/mc_controls.py and callbacks/mc_payment.py

**Files:**
- Create: `btc_web/callbacks/mc_controls.py`
- Create: `btc_web/callbacks/mc_payment.py`

- [ ] **Step 1: Create callbacks/mc_controls.py**

Extract from `callbacks.py` lines 729–1165:
- `_toggle_dca_sc_body` (line 729) — DCA SC body toggle (NOT in sc_loan.py — it's at line 729, not in the 1481–1629 range)
- All loop-generated MC toggle callbacks (body, advanced, regime, freq, year sync)
- `_restore_mc` loop-generated callbacks (lines 901–924) — these are created via `for _rpfx in (...)` loop; bring the entire loop
- `_mc_years_options` (line 986)
- `_update_hm_mc_years_opts` (line 1006)
- `_update_mc_years_opts` loop (line 1022)
- `_calc_mc_cost` (line 1036)
- `_mc_cost_display` (line 1090)
- `_update_mc_cost` loop (line 1144)

- [ ] **Step 2: Create callbacks/mc_payment.py**

Extract from `callbacks.py` lines 1166–1364:
- `_mc_payment_initiate` (line 1166)
- `_quant_proceed` (line 1274)
- `_quant_cancel` (line 1285)
- `_mc_payment_cancel` (line 1357)

Also include `_MC_BTN_TO_TAB` (line 1157), `_MC_QUANT_THRESHOLD` (line 1164).

- [ ] **Step 3: Verify both parse**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); import callbacks.mc_controls; import callbacks.mc_payment; print('OK')"
```

### Task 17: Create callbacks/mc_upload.py, callbacks/sc_loan.py

**Files:**
- Create: `btc_web/callbacks/mc_upload.py`
- Create: `btc_web/callbacks/sc_loan.py`

- [ ] **Step 1: Create callbacks/mc_upload.py**

Extract:
- `_parse_mc_upload` (line 2115)
- `_extract_mc_key_val` (line 2140)
- `_register_mc_upload` factory + loop (lines 2177–2222)
- `_TAB_LABELS` (line 2112)
- `_MC_UPLOAD_FIELDS` (line 2149)
- MC download clientside callbacks (lines 2040–2063)
- `_mc_modal_dismiss` (line 2067)

- [ ] **Step 2: Create callbacks/sc_loan.py**

Extract from `callbacks.py` lines 1481–1629:
- `_SAYLOR_QUOTES` (line 1481)
- `_SAYLOR_QUOTES_JS` (line 1491)
- `_toggle_custom_price_row` (line 1527)
- `_toggle_rollover_row` (line 1532)
- `update_sc_info` (line 1553)

- [ ] **Step 3: Verify both parse**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); import callbacks.mc_upload; import callbacks.sc_loan; print('OK')"
```

### Task 18: Create callbacks/lots.py, callbacks/nav.py, callbacks/ticker.py

**Files:**
- Create: `btc_web/callbacks/lots.py`
- Create: `btc_web/callbacks/nav.py`
- Create: `btc_web/callbacks/ticker.py`

- [ ] **Step 1: Create callbacks/lots.py**

Extract from `callbacks.py` lines 1853–1971:
- `preview_percentile` (line 1853)
- `manage_lots` (line 1869)
- `sync_table_on_load` (line 1965)
- `_lots_summary` helper (line 1977)

- [ ] **Step 2: Create callbacks/nav.py**

Extract:
- `_PATH_TO_TAB` (line 2352) — tab path mapping constant
- `_TAB_CONTROLS` (lines 2353–2385) — dict mapping tab_id → set of component IDs
- `_TAB_TO_PATH` (line 2386) — reverse of `_PATH_TO_TAB`
- `toggle_sc_mode` (line 1831)
- `toggle_sc_display_q` (line 1841)
- `open_faq_item` (line 2850)
- `toggle_share_modal` (line 2872)
- Tab routing clientside callbacks (lines 2388+)

**Important:** `_TAB_CONTROLS` and `_TAB_TO_PATH` are at lines 2352–2386, between the MC upload section and the FAQ/nav section. They must be explicitly included here — they are NOT in the snapshot_cb extraction range (2884+).

- [ ] **Step 3: Create callbacks/ticker.py**

Extract `update_price_ticker` (line 2230).

- [ ] **Step 4: Verify all parse**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); import callbacks.lots; import callbacks.nav; import callbacks.ticker; print('OK')"
```

### Task 19: Create callbacks/snapshot_cb.py

**Files:**
- Create: `btc_web/callbacks/snapshot_cb.py`

- [ ] **Step 1: Create callbacks/snapshot_cb.py**

Extract from `callbacks.py` lines 2884–3240:
- `restore_from_url` (line 2884)
- `manage_snapshot` (line 2919)
- `update_effective_lots` (line 3001)
- `update_snapshot_banner` (line 3010)
- `restore_my_lots` (line 3025)
- `render_link_history` (line 3034)
- `clear_history` (line 3067)
- `apply_hm_palette` (line 3196)
- `generate_share_qr` (line 3211)
- `_add_snapshot_entry` helper (line 2982)
- `_COLORSCALES` constant (if present)

**Extract `_decode_snapshot_by_prefix()` helper** to deduplicate the prefix detection logic (lines 2898–2904 and 2955–2966):

```python
def _decode_snapshot_by_prefix(h):
    """Decode a snapshot hash, detecting version automatically.

    Returns (state_dict, prefix, encoded_str) or (None, None, None).
    """
    if h.startswith(_SNAP_PREFIX):
        return _decode_snapshot(h[len(_SNAP_PREFIX):]), _SNAP_PREFIX, h[len(_SNAP_PREFIX):]
    if h.startswith(_SNAP_PREFIX_V2):
        return _decode_snapshot(h[len(_SNAP_PREFIX_V2):]), _SNAP_PREFIX_V2, h[len(_SNAP_PREFIX_V2):]
    if h.startswith(_SNAP_PREFIX_V1):
        return _decode_snapshot_v1(h[len(_SNAP_PREFIX_V1):]), _SNAP_PREFIX_V1, h[len(_SNAP_PREFIX_V1):]
    return None, None, None
```

Use this in both `restore_from_url` and `manage_snapshot`.

- [ ] **Step 2: Verify it parses**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); import callbacks.snapshot_cb; print('OK')"
```

### Task 20: Create callbacks/__init__.py and retire old callbacks.py

**Files:**
- Create: `btc_web/callbacks/__init__.py`
- Delete: `btc_web/callbacks.py`

- [ ] **Step 1: Create callbacks/__init__.py**

Dash callbacks are registered via decorators at import time. The `__init__.py` just needs to import every submodule so the decorators fire:

```python
"""Callbacks package — importing this module registers all Dash callbacks."""

from callbacks import coerce  # noqa: F401 — helpers only, no callbacks
from callbacks import mc_helpers  # noqa: F401 — helpers only
from callbacks import charts  # noqa: F401
from callbacks import mc_controls  # noqa: F401
from callbacks import mc_payment  # noqa: F401
from callbacks import mc_upload  # noqa: F401
from callbacks import sc_loan  # noqa: F401
from callbacks import lots  # noqa: F401
from callbacks import nav  # noqa: F401
from callbacks import snapshot_cb  # noqa: F401
from callbacks import ticker  # noqa: F401

# Re-export symbols that test_web.py imports from "callbacks"
from callbacks.coerce import _ci, _cf, _format_lots_for_table
from callbacks.mc_helpers import (_build_mc_params, _mc_setup, _mc_finalize,
                                  _mc_payment_check, _ghost_match, _unblocked_val)
from callbacks.mc_upload import _parse_mc_upload, _extract_mc_key_val, _MC_UPLOAD_FIELDS
from callbacks.mc_controls import _mc_years_options, _toggle_dca_sc_body
from callbacks.lots import _lots_summary, manage_lots, preview_percentile
from callbacks.charts import (update_bubble, update_heatmap, update_dca,
                              update_retire, update_supercharge, auto_bubble_yrange)
from callbacks.snapshot_cb import restore_from_url, update_effective_lots
from callbacks.nav import _TAB_CONTROLS, _TAB_TO_PATH, toggle_sc_mode
from callbacks.sc_loan import update_sc_info
```

- [ ] **Step 2: Delete the old monolithic callbacks.py**

```bash
git rm btc_web/callbacks.py
```

- [ ] **Step 3: Verify the full callbacks package loads**

```bash
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); from callbacks import update_bubble, _lots_summary, _parse_mc_upload; print('OK')"
```

- [ ] **Step 4: Commit the callbacks split**

```bash
git add btc_web/callbacks/
git rm btc_web/callbacks.py
git commit -m "Split callbacks.py (3241 LOC) into callbacks/ package (12 modules)"
```

---

## Chunk 5: Update app.py and tests

### Task 21: Update app.py imports

**Files:**
- Modify: `btc_web/app.py`

- [ ] **Step 1: Update app.py**

`app.py` imports from `layout`, `figures`, and `callbacks`. Since `__init__.py` files re-export the public API, most imports should work unchanged. Verify and fix any that break.

Key import patterns in app.py:
```python
from layout import build_layout, ...
from figures import _LOGO_B64_ALL
import callbacks  # triggers callback registration
```

The `import callbacks` line is critical — it triggers all `@callback` decorators.

- [ ] **Step 2: Verify app.py loads without errors**

```bash
cd /Users/bcg/Desktop/btc_test/quantoshi
btc_venv/bin/python3 -c "import sys; sys.path.insert(0,'btc_web'); import app; print('OK')"
```

### Task 22: Update test_web.py imports

**Files:**
- Modify: `btc_web/test_web.py`

- [ ] **Step 1: Audit test imports**

The test file imports from these btc_web modules (lines 141–167):
- `from utils import _q3, _quantize_params, _nearest_quantile` — **unchanged**
- `from snapshot import ...` — **unchanged**
- `from callbacks import _parse_mc_upload, _extract_mc_key_val, _lots_summary, ...` — works via callbacks/__init__.py re-exports
- `from layout import _SPLASH_QUOTES, _SPLASH_QUOTES_JS, ...` — works via layout/__init__.py re-exports
- `from figures import _apply_watermark, build_bubble_figure, ...` — works via figures/__init__.py re-exports
- `from mc_cache import ...` — **unchanged**
- `from mc_overlay import ...` — **unchanged**

If `__init__.py` re-exports are complete, **no test changes needed**. If any imports fail, add the missing re-exports to the relevant `__init__.py`.

- [ ] **Step 2: Run the test suite**

```bash
cd /Users/bcg/Desktop/btc_test/quantoshi
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | head -100
```

- [ ] **Step 3: Fix any failing imports or tests**

Iterate: fix → re-run until green. Common issues:
- Missing re-export in `__init__.py`
- Circular import (break by lazy import or moving shared constants)
- `_TAB_CONTROLS` / `_TAB_TO_PATH` not re-exported from callbacks

- [ ] **Step 4: Run tests/test_figures.py too**

```bash
btc_venv/bin/python3 tests/test_figures.py
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/app.py btc_web/test_web.py
git commit -m "Update app.py and test imports for new package structure"
```

---

## Chunk 6: Dead code cleanup and final verification

### Task 23: Delete confirmed dead code

**Files:**
- Modify: `btc_web/figures/common.py` (or wherever these ended up)
- Modify: `btc_web/layout/common.py`

- [ ] **Step 1: Verify no dead code in figures**

`_price_tickvals` (line 576) IS used by `build_bubble_figure` (line 850) — keep it.
`_apply_mc_xlabel` (line 466) IS used by `build_mc_heatmap_figure` (line 1238) — keep it.

No dead functions to remove from figures. Run a grep to confirm no other orphans:

```bash
cd /Users/bcg/Desktop/btc_test/quantoshi
grep -rn 'def _' btc_web/figures/common.py | while read line; do
  fn=$(echo "$line" | sed 's/.*def \(_[a-zA-Z_]*\).*/\1/')
  count=$(grep -rn "$fn" btc_web/figures/ | grep -v "^.*:.*def " | wc -l)
  if [ "$count" -eq 0 ]; then echo "UNUSED: $fn"; fi
done
```

- [ ] **Step 2: Remove dead functions from layout**

- `_freq_dropdown` (old lines 219–225) — defined but never called (frequency is inlined in `_shared_settings_card`)
- `_stack_control_card` (old lines 315–328) — defined but never called

Verify with grep before deleting:
```bash
grep -rn '_freq_dropdown\|_stack_control_card' btc_web/layout/
```

- [ ] **Step 3: Run tests again**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add -u btc_web/
git commit -m "Remove dead code: _freq_dropdown, _stack_control_card"
```

### Task 24: Syntax check the web app

- [ ] **Step 1: Compile-check all new modules**

```bash
cd /Users/bcg/Desktop/btc_test/quantoshi
btc_venv/bin/python3 -m py_compile btc_web/app.py && \
btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'btc_web')
import layout, figures, callbacks
print('All packages load OK')
"
```

- [ ] **Step 2: Run the web app locally and verify it serves**

```bash
cd /Users/bcg/Desktop/btc_test/quantoshi
DEV=1 timeout 10 bash run_web.sh 2>&1 || true
```

Check that it starts without import errors (it will timeout after 10s, which is expected).

- [ ] **Step 3: Run full test suite one final time**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v 2>&1 | tail -30
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git status
git commit -m "Web app simplification complete: 3 monoliths → 3 packages (28 focused modules)"
```

---

## Summary

| Before | After | Change |
|--------|-------|--------|
| `callbacks.py` (3241 LOC) | `callbacks/` (12 modules, ~200-400 LOC each) | Split by domain |
| `layout.py` (2320 LOC) | `layout/` (9 modules, ~100-350 LOC each) | Split by tab + data |
| `figures.py` (2407 LOC) | `figures/` (7 modules, ~150-400 LOC each) | Split by chart + shared helpers |
| `btc_app/` in root | `archive/btc_app/` | Segmented |
| 2 dead functions (`_freq_dropdown`, `_stack_control_card`) | Deleted | Cleanup |
| Duplicated quantile loop (3×) | `_build_quantile_traces()` | DRY |
| Duplicated model overlay (3×) | `_add_model_overlays()` | DRY |
| Duplicated snapshot decode (2×) | `_decode_snapshot_by_prefix()` | DRY |
