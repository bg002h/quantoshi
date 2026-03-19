# Bandwidth Tier 1: Data Diet — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce bubble chart bandwidth from 2,048 KB raw (956 KB gzipped) to ~60 KB gzipped by reducing interpolation points, removing glow shadows, and rounding trace floats.

**Architecture:** Three surgical changes to the figure pipeline — all in `btc_web/figures/`. No callback changes, no layout changes, no client-side changes. Gzip is already enabled on nginx (`/etc/nginx/nginx.conf`), so reduced data automatically compresses better.

**Tech Stack:** Python, Plotly, Dash

**Spec:** `docs/superpowers/specs/2026-03-19-bandwidth-optimization-design.md` (Tier 1 only)

**Pre-existing conditions:**
- nginx gzip already enabled for `application/json` — verified working on production
- Rate limiting already configured on `/_dash-update-component` (burst=10)
- LRU cache (`@lru_cache(maxsize=8)`) on all figure builders

---

### Task 1: Reduce QR interpolation points

**Files:**
- Modify: `btc_web/figures/common.py:43`

- [ ] **Step 1: Change `_INTERP_POINTS` from 1500 to 400**

Use the Edit tool to replace:
```python
_INTERP_POINTS    = 1500     # sample points for QR interpolation curves
```
with:
```python
_INTERP_POINTS    = 400      # sample points for QR interpolation curves (400 > max screen px)
```

- [ ] **Step 2: Verify bubble figure still builds**

Run:
```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -c "
from btc_core import load_model_data, BubbleModel
import _app_ctx
m = load_model_data()
_app_ctx.M = m
_app_ctx.DEFAULT_MODEL = BubbleModel(m)
_app_ctx.PRICE_MODELS['bub'] = _app_ctx.DEFAULT_MODEL
from figures.bubble import build_bubble_figure
fig = build_bubble_figure(m, {
    'palette':'default','selected_qs':[0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99],
    'log_y':True,'log_x':False,'show_legend':True,'show_data':True,'show_today':True,
    'shade':True,'xmin':2012,'xmax':2030,'auto_y':True,'ymin':1,'ymax':1e6,
    'n_future':3,'pt_size':2,'pt_alpha':0.2,'active_models':[],'mc_enabled':False,
    'mc_fan':None,'mc_ghost_fan':None,'stack_btc':0,'use_lots':False,'lots':[],
})
import json, gzip
d = json.dumps(fig.to_dict()).encode()
print(f'Traces: {len(fig.data)}')
print(f'Raw: {len(d)//1024} KB')
print(f'Gzip: {len(gzip.compress(d, 6))//1024} KB')
print('OK')
"
```
Expected: Raw should be ~600-700 KB (down from 2,048), gzip ~200-300 KB. Print `OK`.

- [ ] **Step 3: Run existing tests**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py -x -q --tb=short 2>&1 | tail -5`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add btc_web/figures/common.py
git commit -m "Reduce _INTERP_POINTS from 1500 to 400 for bandwidth savings"
```

---

### Task 2: Remove QR glow shadow traces (all charts)

**Files:**
- Modify: `btc_web/figures/bubble.py` — remove `_add_glow_trace` call (line ~88) and import
- Modify: `btc_web/figures/dca.py` — remove `_add_glow_trace` call (line ~197) and import
- Modify: `btc_web/figures/retire.py` — remove `_add_glow_trace` call (line ~76) and import
- Modify: `btc_web/figures/supercharge.py` — remove `_add_glow_trace` calls (lines ~137, ~159) and import
- Modify: `btc_web/figures/common.py` — remove `_add_glow_trace` function definition (lines ~129-139) and related constants (`_GLOW_WIDTH`, `_GLOW_ALPHA`)

Glow traces add a wider, semi-transparent duplicate of each QR line for a subtle "neon wire" effect. They add no information and cost 7-9 extra traces per chart.

**Important:** Keep the today-line glow in bubble.py (lines ~229-233) — that uses inline `go.Scatter`, not `_add_glow_trace`.

- [ ] **Step 1: Remove `_add_glow_trace` calls from all 4 figure builders**

For each file, find and delete the `_add_glow_trace(...)` call line, and remove `_add_glow_trace` from its import line. The files are:
- `bubble.py`: 1 call site (in `for q in sel_qs:` loop)
- `dca.py`: 1 call site
- `retire.py`: 1 call site
- `supercharge.py`: 2 call sites

- [ ] **Step 2: Remove `_add_glow_trace` definition from `common.py`**

Delete the function definition (lines ~129-139) and any glow-specific constants (`_GLOW_WIDTH`, `_GLOW_ALPHA`) that are no longer referenced.

- [ ] **Step 3: Verify all 4 figure builders still work**

Run:
```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -c "
import ast
for f in ['btc_web/figures/bubble.py','btc_web/figures/dca.py','btc_web/figures/retire.py','btc_web/figures/supercharge.py','btc_web/figures/common.py']:
    with open(f) as fh: ast.parse(fh.read())
    print(f'{f}: OK')
"
```
Expected: All 5 files pass AST check.

- [ ] **Step 4: Run existing tests**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py -x -q --tb=short 2>&1 | tail -5`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add btc_web/figures/bubble.py btc_web/figures/dca.py btc_web/figures/retire.py btc_web/figures/supercharge.py btc_web/figures/common.py
git commit -m "Remove QR glow shadow traces from all charts (saves 7-9 traces per chart)"
```

---

### Task 3: Round trace floats to 3 significant figures (all charts)

**Files:**
- Modify: `btc_web/figures/common.py` — add `_round_trace_data` helper
- Modify: `btc_web/figures/bubble.py` — round at `_price_cache` and `t_arr` level
- Modify: `btc_web/figures/dca.py` — round simulation output arrays
- Modify: `btc_web/figures/retire.py` — round simulation output arrays
- Modify: `btc_web/figures/supercharge.py` — round simulation output arrays

The goal is to round x/y float arrays to 3 significant figures across all charts, reducing unique float strings for better gzip compression. Measured savings: Bubble -91%, DCA -40%, Retire -21%.

Use the existing `_q3()` from `utils.py` — no new rounding function needed. Visually indistinguishable on log-scale charts (3 sig figs covers $0.06 to $10M+ with sub-pixel precision).

- [ ] **Step 1: Add `_round_trace_data` helper to `common.py`**

Add a convenience function that rounds a list/array to 3 sig figs:

```python
from utils import _q3

def _round_trace_data(arr):
    """Round array to 3 sig figs for bandwidth savings. Passes through 0/None/NaN."""
    return [_q3(v) if v and v == v else v for v in arr]
```

Export it so all figure builders can import it.

- [ ] **Step 2: Apply rounding in `bubble.py`**

Round `t_arr` once after creation, and round `_price_cache[q]` values after computation. This ensures both shading fill traces and QR line traces use the same rounded data (critical for gzip deduplication).

```python
t_arr = _round_trace_data(t_arr)
```

```python
_price_cache[q] = _round_trace_data(_price_cache[q])
```

Also round alt-model overlay trace data (PL, S2F) in the `for model_key in active_models:` loop.

**Do NOT round:** historical scatter data (real prices), bubble composite/support lines.

- [ ] **Step 3: Apply rounding in `dca.py`, `retire.py`, `supercharge.py`**

In each simulation figure builder, round the time array and simulation output arrays before passing to `go.Scatter`. The pattern is the same: find where x/y data is passed to trace creation and wrap with `_round_trace_data()`.

Also round alt-model overlay traces in these files.

- [ ] **Step 4: Measure combined impact**

Run:
```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -c "
import json, gzip, os
os.environ['MPLBACKEND'] = 'Agg'
from btc_core import load_model_data, BubbleModel, PowerLawModel, S2FModel, ExponentialModel, OptGenesisPLModel
import _app_ctx
m = load_model_data()
_app_ctx.M = m
_app_ctx.DEFAULT_MODEL = BubbleModel(m)
_app_ctx.PRICE_MODELS['bub'] = _app_ctx.DEFAULT_MODEL
_app_ctx.PRICE_MODELS['pl'] = PowerLawModel(m.ols_intercept, m.ols_slope, m.price_years, m.price_prices, m.genesis, m.QR_QUANTILES)
_app_ctx.PRICE_MODELS['ogpl'] = OptGenesisPLModel(m.price_years, m.price_prices, m.genesis, m.QR_QUANTILES)
_app_ctx.PRICE_MODELS['s2f'] = S2FModel(m.price_years, m.price_prices, m.genesis)
_app_ctx.PRICE_MODELS['exp'] = ExponentialModel(m.price_years, m.price_prices, m.QR_QUANTILES)

from figures.bubble import build_bubble_figure
from figures.dca import build_dca_figure
from figures.retire import build_retire_figure

def sz(result, name):
    fig = result[0] if isinstance(result, tuple) else result
    d = json.dumps(fig.to_dict()).encode()
    gz = len(gzip.compress(d, 6))
    print(f'{name:40s} {len(d)//1024:>5} KB raw  {gz//1024:>4} KB gz  {len(fig.data):>2} traces')

sz(build_bubble_figure(m, {'palette':'default','selected_qs':[0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99],'log_y':True,'log_x':False,'show_legend':True,'show_data':True,'show_today':True,'shade':True,'xmin':2012,'xmax':2030,'auto_y':True,'ymin':1,'ymax':1e6,'n_future':3,'pt_size':2,'pt_alpha':0.2,'active_models':['pl','s2f'],'mc_enabled':False,'mc_fan':None,'mc_ghost_fan':None,'stack_btc':0,'use_lots':False,'lots':[]}), 'Bubble (9q + PL + S2F)')
sz(build_dca_figure(m, {'palette':'default','selected_qs':[0.01,0.10,0.25,0.50,0.75,0.90,0.99],'amount':500,'freq':'Monthly','yr_start':2026,'yr_end':2040,'display':'btc','log_y':True,'dual_y':True,'show_legend':True,'active_models':['pl','s2f'],'mc_enabled':False,'mc_fan':None,'mc_ghost_fan':None,'sc_enabled':True,'sc_type':'amortizing','sc_loan':50000,'sc_rate':5,'sc_term':60,'sc_repeats':0,'sc_rollover':[],'sc_entry_mode':'live','sc_custom_price':85000,'sc_tax':33,'stack_btc':0,'use_lots':False,'lots':[]}), 'DCA (7q + SC + PL + S2F)')
sz(build_retire_figure(m, {'palette':'default','selected_qs':[0.01,0.10,0.25,0.50,0.75,0.90,0.99],'wd':5000,'freq':'Monthly','infl':4.0,'yr_start':2031,'yr_end':2075,'log_y':True,'dual_y':True,'annotate':True,'show_legend':True,'active_models':['pl','s2f'],'mc_enabled':False,'mc_fan':None,'mc_ghost_fan':None,'stack_btc':1.0,'use_lots':False,'lots':[]}), 'Retire (7q + PL + S2F)')
print('OK')
"
```
Expected targets (gzipped):
- Bubble: ~108 KB (was 1,225 KB — 91% reduction)
- DCA: ~24 KB (was 40 KB — 40% reduction)
- Retire: ~22 KB (was 28 KB — 21% reduction)

- [ ] **Step 5: Run existing tests**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py -x -q --tb=short 2>&1 | tail -5`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add btc_web/figures/common.py btc_web/figures/bubble.py btc_web/figures/dca.py btc_web/figures/retire.py btc_web/figures/supercharge.py
git commit -m "Round trace floats to 3 sig figs across all charts for gzip savings"
```

- [ ] **Step 3: Measure the combined impact**

Run:
```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -c "
from btc_core import load_model_data, BubbleModel
import _app_ctx
m = load_model_data()
_app_ctx.M = m
_app_ctx.DEFAULT_MODEL = BubbleModel(m)
_app_ctx.PRICE_MODELS['bub'] = _app_ctx.DEFAULT_MODEL
from figures.bubble import build_bubble_figure
fig = build_bubble_figure(m, {
    'palette':'default','selected_qs':[0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99],
    'log_y':True,'log_x':False,'show_legend':True,'show_data':True,'show_today':True,
    'shade':True,'xmin':2012,'xmax':2030,'auto_y':True,'ymin':1,'ymax':1e6,
    'n_future':3,'pt_size':2,'pt_alpha':0.2,'active_models':[],'mc_enabled':False,
    'mc_fan':None,'mc_ghost_fan':None,'stack_btc':0,'use_lots':False,'lots':[],
})
import json, gzip
d = json.dumps(fig.to_dict()).encode()
gz = len(gzip.compress(d, 6))
print(f'Traces: {len(fig.data)}')
print(f'Raw: {len(d)//1024} KB')
print(f'Gzip: {gz//1024} KB')
print(f'Reduction from original 956 KB: {(1 - gz/978944)*100:.0f}%')
print('OK')
"
```
Expected: Gzip should be ~50-70 KB (down from original 956 KB).

- [ ] **Step 4: Run existing tests**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py -x -q --tb=short 2>&1 | tail -5`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add btc_web/figures/common.py btc_web/figures/bubble.py
git commit -m "Round QR trace floats to 3 sig figs for better gzip compression"
```

---

### Task 4: Verify prewarm and update spec

- [ ] **Step 1: Verify prewarm builds smaller figures**

Run:
```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -c "
from btc_core import load_model_data, BubbleModel, PowerLawModel, OptGenesisPLModel, S2FModel, ExponentialModel
import _app_ctx
m = load_model_data()
_app_ctx.M = m
_app_ctx.DEFAULT_MODEL = BubbleModel(m)
_app_ctx.PRICE_MODELS['bub'] = _app_ctx.DEFAULT_MODEL
_app_ctx.PRICE_MODELS['pl'] = PowerLawModel(m.ols_intercept, m.ols_slope, m.price_years, m.price_prices, m.genesis, m.QR_QUANTILES)
_app_ctx.PRICE_MODELS['ogpl'] = OptGenesisPLModel(m.price_years, m.price_prices, m.genesis, m.QR_QUANTILES)
_app_ctx.PRICE_MODELS['s2f'] = S2FModel(m.price_years, m.price_prices, m.genesis)
_app_ctx.PRICE_MODELS['exp'] = ExponentialModel(m.price_years, m.price_prices, m.QR_QUANTILES)

import json, gzip
from figures.bubble import build_bubble_figure
from figures.heatmap import build_heatmap_figure
from figures.dca import build_dca_figure
from figures.retire import build_retire_figure

def sz(result, name):
    fig = result[0] if isinstance(result, tuple) else result
    d = json.dumps(fig.to_dict()).encode()
    gz = len(gzip.compress(d, 6))
    print(f'{name:30s} {len(d)//1024:>5} KB raw  {gz//1024:>4} KB gz  {len(fig.data):>2} traces')

sz(build_bubble_figure(m, {'palette':'default','selected_qs':[0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99],'log_y':True,'log_x':False,'show_legend':True,'show_data':True,'show_today':True,'shade':True,'xmin':2012,'xmax':2030,'auto_y':True,'ymin':1,'ymax':1e6,'n_future':3,'pt_size':2,'pt_alpha':0.2,'active_models':[],'mc_enabled':False,'mc_fan':None,'mc_ghost_fan':None,'stack_btc':0,'use_lots':False,'lots':[]}), 'Bubble (9q)')
sz(build_heatmap_figure(m, {'palette':'default','entry_yr':2026,'entry_q':50.0,'entry_price':85000,'exit_yr_min':2027,'exit_yr_max':2060,'exit_qs':[0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99],'color_mode':0,'break1':0,'break2':20,'grad_steps':32,'cell_text':'cagr','cell_fs':9,'hm_stk':0,'active_model':'bub','mc_enabled':False,'mc_fan':None}), 'Heatmap (9q x 33yr)')
sz(build_dca_figure(m, {'palette':'default','selected_qs':[0.50],'amount':500,'freq':'Monthly','yr_start':2026,'yr_end':2040,'display':'btc','log_y':True,'dual_y':True,'show_legend':True,'active_models':[],'mc_enabled':False,'mc_fan':None,'mc_ghost_fan':None,'sc_enabled':False,'sc_type':'amortizing','sc_loan':0,'sc_rate':5,'sc_term':60,'sc_repeats':0,'sc_rollover':[],'sc_entry_mode':'live','sc_custom_price':85000,'sc_tax':33,'stack_btc':0,'use_lots':False,'lots':[]}), 'DCA (1q, 14yr)')
sz(build_retire_figure(m, {'palette':'default','selected_qs':[0.01,0.10,0.25],'wd':5000,'freq':'Monthly','infl':4.0,'yr_start':2031,'yr_end':2075,'log_y':True,'dual_y':True,'annotate':True,'show_legend':True,'active_models':[],'mc_enabled':False,'mc_fan':None,'mc_ghost_fan':None,'stack_btc':1.0,'use_lots':False,'lots':[]}), 'Retire (3q, 44yr)')

print('\\nAll figures built OK')
"
```
Expected: Bubble should be dramatically smaller. Other charts unchanged. No errors.

- [ ] **Step 2: Update spec with final measured numbers**

Edit `docs/superpowers/specs/2026-03-19-bandwidth-optimization-design.md` — update the Tier 1 results table with actual post-implementation measurements.

- [ ] **Step 3: Commit spec update**

```bash
git add docs/superpowers/specs/2026-03-19-bandwidth-optimization-design.md
git commit -m "Update bandwidth spec with measured Tier 1 results"
```

---

### Task 5: Final smoke test

- [ ] **Step 1: Run full test suite**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py -x -q --tb=short 2>&1 | tail -10`
Expected: All tests pass.

- [ ] **Step 2: AST check all modified files**

Run:
```bash
btc_venv/bin/python3 -c "
import ast
for f in ['btc_web/figures/common.py', 'btc_web/figures/bubble.py']:
    with open(f) as fh: ast.parse(fh.read())
    print(f'{f}: OK')
"
```
Expected: Both files OK.

---
