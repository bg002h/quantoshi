# SP.ipynb Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract model computation from SP.ipynb into a standalone build script, move visual config to `btc_web/theme.py`, slim the pkl to model-only keys.

**Architecture:** Three phases. Phase 1 creates `sp_stripped.ipynb` (computation-only notebook producing lean pkl). Phase 2 creates `btc_web/theme.py` and migrates figure builders away from model-object visual keys. Phase 3 creates `tools/build_bm_model.py` (standalone script producing identical pkl).

**Tech Stack:** Python 3.14, numpy, scipy, statsmodels, pandas

**Spec:** `docs/superpowers/specs/2026-03-30-sp-extraction-design.md`

**Branch:** `SPFix`

**Test command:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --ignore=btc_web/test_tax_e2e.py --tb=short`

**Expected baseline:** 889 passed, 0 failed, 5 skipped

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `sp_stripped.ipynb` | Create | Minimal 3-cell notebook -- computation only, lean pkl |
| `tools/verify_pkl.py` | Create | Key-by-key + sha256 pkl comparison script |
| `btc_web/theme.py` | Create | Visual constants for figure builders |
| `btc_web/figures/common.py` | Modify | Import from theme; drop `m` from `_base_layout`, `_error_figure` |
| `btc_web/figures/heatmap.py` | Modify | Import from theme; update `_error_figure` calls |
| `btc_web/figures/bubble.py` | Modify | Import from theme; update `_base_layout` call |
| `btc_web/figures/supercharge.py` | Modify | Import from theme; update `_base_layout`/`_error_figure` calls |
| `btc_web/figures/dca.py` | Modify | Update `_error_figure` calls (drop `m`) |
| `btc_web/figures/retire.py` | Modify | Update `_error_figure` calls if present (drop `m`) |
| `btc_web/figures/citadel.py` | Modify | Update `_error_figure` calls (drop `m`) |
| `archive/btc_app/btc_core.py` | Modify | `.get()` fallbacks for removed visual keys |
| `tools/build_bm_model.py` | Create | Standalone model build script |
| `update_prices.py` | Modify | Call build script instead of notebook |
| `CLAUDE.md` | Modify | Update notebook/rebuild sections |

---

### Task 1: Create verification script

**Files:**
- Create: `tools/verify_pkl.py`

This script is used throughout all phases to compare pkl files.

- [ ] **Step 1: Create `tools/verify_pkl.py`**

```python
#!/usr/bin/env python3
"""Compare two model_data.pkl files -- key-by-key values + sha256 hashes.

NOTE: Uses pickle.load for trusted, locally-generated model data files only.
"""
import hashlib
import struct
import sys
import pickle

import numpy as np

MODEL_KEYS = [
    "qr_fits", "QR_QUANTILES", "ols_intercept", "ols_slope", "GENESIS_DATE",
    "years_plot_bm", "support_plot_bm", "bm_comp_by_n", "bm_r2_comp",
    "bm_n_future_max", "bm_sigma0_up", "bm_sigma0_down", "bm_alpha_up",
    "bm_alpha_down", "price_dates", "price_years", "price_prices",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def floats_identical(a, b):
    return struct.pack("d", a) == struct.pack("d", b)


def compare_values(a, b, path=""):
    """Recursively compare two values. Return list of mismatch descriptions."""
    mismatches = []
    if type(a) != type(b):
        mismatches.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
        return mismatches
    if isinstance(a, dict):
        for k in sorted(set(list(a.keys()) + list(b.keys()))):
            if k not in a:
                mismatches.append(f"{path}[{k!r}]: missing in first")
            elif k not in b:
                mismatches.append(f"{path}[{k!r}]: missing in second")
            else:
                mismatches.extend(compare_values(a[k], b[k], f"{path}[{k!r}]"))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            mismatches.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                mismatches.extend(compare_values(x, y, f"{path}[{i}]"))
    elif isinstance(a, float):
        if not floats_identical(a, b):
            mismatches.append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            diff = np.where(a != b)
            mismatches.append(f"{path}: arrays differ at {len(diff[0])} positions")
    else:
        if a != b:
            mismatches.append(f"{path}: {a!r} vs {b!r}")
    return mismatches


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <reference.pkl> <candidate.pkl>")
        sys.exit(1)

    ref_path, cand_path = sys.argv[1], sys.argv[2]

    # SHA256 comparison
    ref_hash = sha256(ref_path)
    cand_hash = sha256(cand_path)
    print(f"SHA256 reference:  {ref_hash}")
    print(f"SHA256 candidate:  {cand_hash}")
    print(f"SHA256 match:      {'YES' if ref_hash == cand_hash else 'NO'}")
    print()

    # Key-by-key comparison
    ref = load_pkl(ref_path)
    cand = load_pkl(cand_path)

    all_ok = True
    for key in MODEL_KEYS:
        if key not in ref:
            print(f"  {key:25s}  SKIP (not in reference)")
            continue
        if key not in cand:
            print(f"  {key:25s}  MISSING in candidate")
            all_ok = False
            continue
        mismatches = compare_values(ref[key], cand[key], key)
        if mismatches:
            print(f"  {key:25s}  FAIL")
            for m in mismatches[:3]:
                print(f"    {m}")
            all_ok = False
        else:
            print(f"  {key:25s}  OK")

    extra = set(cand.keys()) - set(ref.keys()) - set(MODEL_KEYS)
    if extra:
        print(f"\nExtra keys in candidate: {sorted(extra)}")

    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the verification script (self-comparison)**

```bash
btc_venv/bin/python3 tools/verify_pkl.py archive/btc_app/model_data.pkl archive/btc_app/model_data.pkl
```

Expected: SHA256 match YES, all 17 keys OK, Overall PASS.

- [ ] **Step 3: Commit**

```bash
git add tools/verify_pkl.py
git commit -m "feat: add pkl verification script -- sha256 + key-by-key comparison"
```

---

### Task 2: Create `sp_stripped.ipynb` -- Phase 1

**Files:**
- Create: `sp_stripped.ipynb`

Create a minimal notebook by extracting computation-only code from SP.ipynb cells.

- [ ] **Step 1: Confirm line ranges**

```bash
btc_venv/bin/python3 -c "
import json
with open('SP.ipynb') as f:
    nb = json.load(f)
src0 = ''.join(nb['cells'][0]['source'])
for i, line in enumerate(src0.split('\n'), 1):
    if 'plt.subplots' in line or 'plt.figure' in line:
        print(f'Cell 0 first plot: line {i}: {line.strip()}')
        break
src1 = ''.join(nb['cells'][1]['source'])
for i, line in enumerate(src1.split('\n'), 1):
    if 'plt.subplots' in line or 'plt.figure' in line:
        print(f'Cell 1 first plot: line {i}: {line.strip()}')
        break
src3 = ''.join(nb['cells'][3]['source'])
print(f'Cell 3 line 1: {src3.split(chr(10))[0]}')
"
```

Expected: Cell 0 first plot ~line 1072, Cell 1 first plot ~line 329.

- [ ] **Step 2: Build stripped notebook via script**

Write to `/tmp/build_stripped.py` and run:

```python
#!/usr/bin/env python3
"""Build sp_stripped.ipynb from SP.ipynb -- computation only, lean pkl."""
import json

with open("SP.ipynb") as f:
    nb = json.load(f)

def get_source(cell_idx):
    s = nb["cells"][cell_idx]["source"]
    return "".join(s) if isinstance(s, list) else s

# Cell 0: keep everything before first plot
src0 = get_source(0)
lines0 = src0.split("\n")
first_plot = None
for i, line in enumerate(lines0):
    if "plt.subplots" in line or "plt.figure" in line:
        first_plot = i
        break
assert first_plot is not None, "Could not find first plot in Cell 0"
cell0 = "\n".join(lines0[:first_plot])

# Cell 1: keep everything before first plot
src1 = get_source(1)
lines1 = src1.split("\n")
first_plot_1 = None
for i, line in enumerate(lines1):
    if "plt.subplots" in line or "plt.figure" in line:
        first_plot_1 = i
        break
assert first_plot_1 is not None, "Could not find first plot in Cell 1"
cell1 = "\n".join(lines1[:first_plot_1])

# Cell 2 (was Cell 3): keep comp_by_n computation, rewrite dict
src3 = get_source(3)
lines3 = src3.split("\n")
model_start = None
for i, line in enumerate(lines3):
    if line.strip().startswith("_model") and "=" in line and "{" in line:
        model_start = i
        break
pre_dict = "\n".join(lines3[:model_start]) if model_start else "\n".join(lines3[:37])

cell2 = pre_dict + '''

_model = {
    "qr_fits":         {str(k): dict(v) for k, v in qr_fits.items()},
    "QR_QUANTILES":    list(QR_QUANTILES),
    "ols_intercept":   float(ols_intercept),
    "ols_slope":       float(ols_slope),
    "GENESIS_DATE":    str(GENESIS_DATE.date()),
    "years_plot_bm":   years_plot_bm.tolist(),
    "support_plot_bm": support_plot_bm.tolist(),
    "bm_comp_by_n":    [c.tolist() for c in _comp_by_n],
    "bm_r2_comp":      float(bm_r2_comp),
    "bm_n_future_max": _max_n,
    "price_dates":     df["date"].dt.strftime("%Y-%m-%d").tolist(),
    "price_years":     df["years"].tolist(),
    "price_prices":    df["price"].tolist(),
}

# NOTE: Export path logic must match original Cell 3.
# The implementer should copy the original Cell 3 export path
# logic verbatim rather than rewriting it. The path below is
# a template -- verify against the actual Cell 3 before using.
import pickle as _pkl, os as _os
_export_dir = "archive/btc_app"
if not _os.path.isdir(_export_dir):
    _export_dir = "btc_app"
if not _os.path.isdir(_export_dir):
    _os.makedirs(_export_dir)
_out = _os.path.join(_export_dir, "model_data.pkl")
with open(_out, "wb") as _f:
    _pkl.dump(_model, _f, protocol=4)
print(f"Wrote {_out}  ({_os.path.getsize(_out)/1024:.0f} KB, {len(_model)} keys)")
'''

def make_cell(source):
    return {
        "cell_type": "code", "execution_count": None,
        "id": "", "metadata": {}, "outputs": [], "source": source,
    }

stripped = {
    "cells": [make_cell(cell0), make_cell(cell1), make_cell(cell2)],
    "metadata": nb["metadata"],
    "nbformat": nb["nbformat"],
    "nbformat_minor": nb["nbformat_minor"],
}

with open("sp_stripped.ipynb", "w") as f:
    json.dump(stripped, f, indent=1)

for i, cell in enumerate(stripped["cells"]):
    n = cell["source"].count("\n") + 1
    print(f"Cell {i}: {n} lines")
```

```bash
btc_venv/bin/python3 /tmp/build_stripped.py
```

- [ ] **Step 3: Save reference pkl and run stripped notebook**

```bash
cp archive/btc_app/model_data.pkl /tmp/model_data_reference.pkl
~/.local/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 sp_stripped.ipynb
```

- [ ] **Step 4: Verify lean pkl against reference**

```bash
btc_venv/bin/python3 tools/verify_pkl.py /tmp/model_data_reference.pkl archive/btc_app/model_data.pkl
```

Expected: SHA256 NO (fewer keys). 13 non-sigma keys OK. 4 sigma keys MISSING (from `fit_sigma.py`). Fix any non-sigma failures before proceeding.

- [ ] **Step 5: Restore reference pkl**

```bash
cp /tmp/model_data_reference.pkl archive/btc_app/model_data.pkl
```

- [ ] **Step 6: Commit**

```bash
git add sp_stripped.ipynb
git commit -m "feat: add sp_stripped.ipynb -- computation-only notebook for lean pkl"
```

---

### Task 3: Empirical visual key testing -- Phase 2a

**Files:** None -- diagnostic step only.

- [ ] **Step 1: Test which visual keys affect web app figure output**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "
from btc_core import load_model_data
M = load_model_data()
VISUAL_KEYS = [
    'PLOT_BG_COLOR', 'TEXT_COLOR', 'TITLE_COLOR', 'SPINE_COLOR',
    'GRID_MAJOR_COLOR', 'GRID_MINOR_COLOR', 'DATA_COLOR',
    'DATA_PT_SIZE', 'DATA_PT_SIZE_ZOOM',
    'CAGR_SEG_B1', 'CAGR_SEG_B2',
    'CAGR_SEG_C_LO', 'CAGR_SEG_C_MID1', 'CAGR_SEG_C_MID2', 'CAGR_SEG_C_HI',
    'CAGR_GRAD_STEPS', 'CAGR_HEATMAP_FONTSIZE',
    'ZOOM_YEAR_LO', 'ZOOM_YEAR_HI', 'ZOOM_PRICE_LO', 'ZOOM_PRICE_HI',
    'TABLE_YEARS', 'qr_colors', 'QR_LINESTYLES',
]
from figures.bubble import build_bubble_figure
from figures.heatmap import build_heatmap_figure
from figures.dca import build_dca_figure
from figures.retire import build_retire_figure
from figures.supercharge import build_supercharge_figure
from figures.citadel import build_citadel_figure
from tab_defaults import (bubble_defaults, heatmap_defaults, dca_defaults,
                           retire_defaults, supercharge_defaults, citadel_defaults)
import plotly.io as pio
builders = [
    ('bubble', build_bubble_figure, bubble_defaults()),
    ('heatmap', build_heatmap_figure, heatmap_defaults()),
    ('dca', build_dca_figure, dca_defaults()),
    ('retire', build_retire_figure, retire_defaults()),
    ('supercharge', build_supercharge_figure, supercharge_defaults()),
    ('citadel', build_citadel_figure, citadel_defaults()),
]
baselines = {}
for name, builder, params in builders:
    try:
        fig = builder(M, params)
        baselines[name] = pio.to_json(fig)
    except Exception as e:
        baselines[name] = f'ERROR: {e}'
for key in VISUAL_KEYS:
    if not hasattr(M, key):
        print(f'{key:30s}  NOT ON MODEL')
        continue
    original = getattr(M, key)
    delattr(M, key)
    affected = []
    for name, builder, params in builders:
        try:
            fig = builder(M, params)
            result = pio.to_json(fig)
            if result != baselines[name]:
                affected.append(name)
        except Exception as e:
            affected.append(f'{name}(CRASH)')
    setattr(M, key, original)
    if affected:
        print(f'{key:30s}  LIVE -- affects: {chr(44).join(affected)}')
    else:
        print(f'{key:30s}  DEAD')
"
```

- [ ] **Step 2: Record results**

Save output. LIVE keys go in theme.py. DEAD keys are dropped entirely.

---

### Task 4: Create `btc_web/theme.py` -- Phase 2b

**Files:**
- Create: `btc_web/theme.py`

- [ ] **Step 1: Create theme.py with live keys only**

Based on Task 3 results. Expected (adjust per empirical findings):

```python
"""Quantoshi chart theme -- visual constants for figure builders.

These were previously bundled in model_data.pkl and accessed via the
model object (m.PLOT_BG_COLOR, etc.).  Now they live here so the pkl
only carries model data.
"""

# -- Chart colors ---------------------------------------------------------------
PLOT_BG_COLOR    = "#FFFFFF"
TEXT_COLOR       = "#222222"
TITLE_COLOR      = "#1A3060"
SPINE_COLOR      = "#888888"
GRID_MAJOR_COLOR = "#BBBBBB"

# -- CAGR heatmap defaults ------------------------------------------------------
CAGR_SEG_B1     = 5.0
CAGR_SEG_B2     = 16.0
CAGR_SEG_C_LO   = "#2166AC"
CAGR_SEG_C_MID1 = "#F7F7F7"
CAGR_SEG_C_MID2 = "#FF8C00"
CAGR_SEG_C_HI   = "#CC1100"
```

- [ ] **Step 2: Commit**

```bash
git add btc_web/theme.py
git commit -m "feat: add theme.py -- visual constants extracted from pkl"
```

---

### Task 5: Migrate figure builders to theme.py -- Phase 2c

**Files:**
- Modify: `btc_web/figures/common.py`
- Modify: `btc_web/figures/heatmap.py`
- Modify: `btc_web/figures/bubble.py`
- Modify: `btc_web/figures/supercharge.py`
- Modify: `btc_web/figures/dca.py`
- Modify: `btc_web/figures/citadel.py`

**Signature changes:**
- `_base_layout(m, title, xlabel, ylabel)` → `_base_layout(title, xlabel, ylabel)` — `m` removed, only used visual keys
- `_error_figure(m, title)` → `_error_figure(title)` — `m` removed, only used visual keys
- `_sim_layout(m, p, ...)` — keeps `m` (needs `m.genesis`), but its `_base_layout` call drops `m`

- [ ] **Step 1: Update `figures/common.py`**

Add `import theme` at top.

`_error_figure`: replace `m.PLOT_BG_COLOR` → `theme.PLOT_BG_COLOR`, `m.TEXT_COLOR` → `theme.TEXT_COLOR`. Remove `m` from signature: `def _error_figure(title):`.

`_base_layout`: replace all `m.*` visual keys with `theme.*`. Remove `m` from signature: `def _base_layout(title, xlabel, ylabel, **kwargs):`.

`_sim_layout`: update call from `_base_layout(m, title=..., xlabel=..., ylabel=...)` to `_base_layout(title=..., xlabel=..., ylabel=...)`. Keep `m` in `_sim_layout` signature (still needs `m.genesis`).

- [ ] **Step 2: Update `figures/heatmap.py`**

Add `import theme`. Replace all `m.TEXT_COLOR`, `m.PLOT_BG_COLOR`, `m.SPINE_COLOR`, `m.TITLE_COLOR`, `m.GRID_MAJOR_COLOR`, `m.CAGR_SEG_*`, `m.CAGR_SEG_B1/B2` with `theme.*`. Update all `_error_figure(m, ...)` calls to `_error_figure(...)` (5 call sites).

- [ ] **Step 3: Update `figures/bubble.py`**

Add `import theme`. Replace `m.PLOT_BG_COLOR`, `m.SPINE_COLOR`, `m.TEXT_COLOR` in legend annotation. Update `_base_layout(m, ...)` call to `_base_layout(...)`.

- [ ] **Step 4: Update `figures/supercharge.py`**

Add `import theme`. Replace `m.PLOT_BG_COLOR`, `m.TEXT_COLOR` in early-return layout. Update any `_error_figure(m, ...)` calls to `_error_figure(...)`.

- [ ] **Step 4b: Update `figures/dca.py`**

Update `_error_figure(m, ...)` calls to `_error_figure(...)`. (dca.py imports `_error_figure` but may not use visual keys directly — check.)

- [ ] **Step 4c: Update `figures/citadel.py`**

Update `_error_figure(m, ...)` calls to `_error_figure(...)` (3 call sites).

- [ ] **Step 5: Syntax check**

```bash
cd btc_web && PYTHONPATH=".:../archive/btc_app" ../btc_venv/bin/python3 -c \
  "import theme, figures.common, figures.heatmap, figures.bubble, figures.supercharge; print('OK')"
```

- [ ] **Step 6: Run test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --ignore=btc_web/test_tax_e2e.py --tb=short 2>&1 | tail -5
```

Expected: 889 passed, 0 failed, 5 skipped.

- [ ] **Step 7: Verify figures match baselines**

Re-run Task 3's empirical test comparing new output (theme.py) against original baselines. All 5 charts must produce identical plotly JSON.

- [ ] **Step 8: Commit**

```bash
git add btc_web/figures/common.py btc_web/figures/heatmap.py btc_web/figures/bubble.py btc_web/figures/supercharge.py
git commit -m "refactor: migrate figure builders from m.VISUAL_KEY to theme.VISUAL_KEY"
```

---

### Task 6: Patch `btc_core.py` for lean pkl -- Phase 2d

**Files:**
- Modify: `archive/btc_app/btc_core.py`

- [ ] **Step 1: Read `ModelData.__init__`**

```bash
sed -n '200,240p' archive/btc_app/btc_core.py
```

- [ ] **Step 2: Add `.get()` fallbacks for visual keys**

Replace hard reads with fallback versions:

```python
_VIS_STR = {
    "PLOT_BG_COLOR": "#FFFFFF", "TEXT_COLOR": "#222222",
    "TITLE_COLOR": "#1A3060", "SPINE_COLOR": "#888888",
    "GRID_MAJOR_COLOR": "#BBBBBB", "GRID_MINOR_COLOR": "#E8E8E8",
    "DATA_COLOR": "#606060",
    "CAGR_SEG_C_LO": "#2166AC", "CAGR_SEG_C_MID1": "#F7F7F7",
    "CAGR_SEG_C_MID2": "#FF8C00", "CAGR_SEG_C_HI": "#CC1100",
}
_VIS_INT = {
    "DATA_PT_SIZE": 16, "DATA_PT_SIZE_ZOOM": 32,
    "ZOOM_YEAR_LO": 2025, "ZOOM_YEAR_HI": 2038,
    "CAGR_GRAD_STEPS": 24, "CAGR_HEATMAP_FONTSIZE": 6,
}
_VIS_FLOAT = {
    "ZOOM_PRICE_LO": 40000.0, "ZOOM_PRICE_HI": 1750000.0,
    "CAGR_SEG_B1": 5.0, "CAGR_SEG_B2": 16.0,
}
for key, default in _VIS_STR.items():
    setattr(self, key, d.get(key, default))
for key, default in _VIS_INT.items():
    setattr(self, key, int(d.get(key, default)))
for key, default in _VIS_FLOAT.items():
    setattr(self, key, float(d.get(key, default)))
```

Patch `qr_colors`:
```python
self.qr_colors = {float(k): v for k, v in d["qr_colors"].items()} if "qr_colors" in d else {}
```

Patch `QR_LINESTYLES`:
```python
raw_ls = d.get("QR_LINESTYLES", {})
```

- [ ] **Step 3: Run test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --ignore=btc_web/test_tax_e2e.py --tb=short 2>&1 | tail -5
```

Expected: 889 passed, 0 failed, 5 skipped.

- [ ] **Step 4: Test with lean pkl**

```bash
cp archive/btc_app/model_data.pkl /tmp/model_data_reference.pkl
~/.local/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 sp_stripped.ipynb
cd btc_web && PYTHONPATH=".:../archive/btc_app" ../btc_venv/bin/python3 -c \
  "import layout, figures, callbacks, cache, engines.adapter, engines.citadel; print('OK')"
cp /tmp/model_data_reference.pkl archive/btc_app/model_data.pkl
```

- [ ] **Step 5: Commit**

```bash
git add archive/btc_app/btc_core.py
git commit -m "fix: add .get() fallbacks for removed visual keys in ModelData"
```

---

### Task 7: Create `tools/build_bm_model.py` -- Phase 3

**Files:**
- Create: `tools/build_bm_model.py`

Standalone build script. Extracts computation cells from `sp_stripped.ipynb` and runs them in sequence -- same code, no notebook kernel needed.

NOTE: This script uses Python's `exec()` to run notebook cell source code in a shared namespace. This is intentional -- it guarantees identical computation to the notebook by running the exact same code. The input (`sp_stripped.ipynb`) is a trusted, locally-generated file, not user input. This pattern matches the existing `tools/build_ef_model.py`.

- [ ] **Step 1: Create `tools/build_bm_model.py`**

```python
#!/usr/bin/env python3
"""Build model_data.pkl from BitcoinPricesDaily.csv -- standalone, no notebook.

Extracts computation code from sp_stripped.ipynb cells 0, 1, 2 and
runs them in sequence in a shared namespace.  Produces the same lean
pkl as running the stripped notebook via jupyter nbconvert.

NOTE: Uses exec() on trusted local notebook source to guarantee
identical computation.  This matches the build_ef_model.py pattern.

Usage:
    btc_venv/bin/python3 tools/build_bm_model.py [--out PATH]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_cell(nb, idx):
    """Return source code of notebook cell as a single string."""
    s = nb["cells"][idx]["source"]
    return "".join(s) if isinstance(s, list) else s


def main():
    parser = argparse.ArgumentParser(description="Build model_data.pkl")
    parser.add_argument("--out", default=None,
                        help="Output pkl path (default: btc_app/model_data.pkl)")
    parser.add_argument("--notebook",
                        default=os.path.join(ROOT, "sp_stripped.ipynb"),
                        help="Source notebook")
    args = parser.parse_args()

    with open(args.notebook) as f:
        nb = json.load(f)

    # Set non-interactive backend before Cell 0 imports matplotlib.
    # Required for headless servers (production VPS).
    import matplotlib
    matplotlib.use("Agg")

    os.chdir(ROOT)
    ns = {"__name__": "__main__"}

    for i, label in enumerate(["Bubble model", "QR/OLS fitting", "Export"]):
        print(f"Cell {i}: {label}...")
        code = compile(extract_cell(nb, i), f"{args.notebook}:cell{i}", "exec")
        # exec() on trusted local notebook cell source -- see module docstring
        exec(code, ns)  # noqa: S102

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test build script vs stripped notebook**

```bash
cp archive/btc_app/model_data.pkl /tmp/model_data_reference.pkl

# Build via script
btc_venv/bin/python3 tools/build_bm_model.py
cp archive/btc_app/model_data.pkl /tmp/model_data_script.pkl

# Build via notebook
cp /tmp/model_data_reference.pkl archive/btc_app/model_data.pkl
~/.local/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 sp_stripped.ipynb
cp archive/btc_app/model_data.pkl /tmp/model_data_notebook.pkl

# Compare
btc_venv/bin/python3 tools/verify_pkl.py /tmp/model_data_notebook.pkl /tmp/model_data_script.pkl

# Restore
cp /tmp/model_data_reference.pkl archive/btc_app/model_data.pkl
```

Expected: SHA256 match (same code). Key-by-key all OK. Report results either way.

- [ ] **Step 3: Commit**

```bash
git add tools/build_bm_model.py
git commit -m "feat: add build_bm_model.py -- standalone model builder"
```

---

### Task 8: Update `update_prices.py` -- Phase 3b

**Files:**
- Modify: `update_prices.py`

- [ ] **Step 1: Read current `update_prices.py`**

```bash
cat update_prices.py
```

Find the `jupyter nbconvert` invocation.

- [ ] **Step 2: Replace notebook call with build script + fit_sigma**

Replace the notebook execution line with:

```python
subprocess.run([sys.executable, "tools/build_bm_model.py"], check=True)
subprocess.run([sys.executable, "tools/fit_sigma.py",
                "--pkl", "archive/btc_app/model_data.pkl", "--type", "bm"], check=True)
```

If `fit_sigma.py` was already called after the notebook, keep that and just replace the notebook line.

- [ ] **Step 3: Test dry-run**

```bash
btc_venv/bin/python3 update_prices.py --dry-run
```

- [ ] **Step 4: Commit**

```bash
git add update_prices.py
git commit -m "refactor: update_prices.py uses build_bm_model.py instead of notebook"
```

- [ ] **Step 5: Update CLAUDE.md**

Update the "Run the notebook" section to document the new build command:
```
btc_venv/bin/python3 tools/build_bm_model.py
btc_venv/bin/python3 tools/fit_sigma.py --pkl archive/btc_app/model_data.pkl --type bm
```

Update the "Full rebuild after notebook changes" section similarly. Note that `SP.ipynb` is now for exploration only — the model is built via `tools/build_bm_model.py`.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md — build_bm_model.py replaces notebook execution"
```

---

### Task 9: End-to-end verification

- [ ] **Step 1: Full pipeline**

```bash
btc_venv/bin/python3 tools/build_bm_model.py
btc_venv/bin/python3 tools/fit_sigma.py --pkl archive/btc_app/model_data.pkl --type bm
btc_venv/bin/python3 tools/verify_pkl.py /tmp/model_data_reference.pkl archive/btc_app/model_data.pkl
```

Expected: All 17 keys OK.

- [ ] **Step 2: Test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --ignore=btc_web/test_tax_e2e.py --tb=short 2>&1 | tail -5
```

Expected: 889 passed, 0 failed, 5 skipped.

- [ ] **Step 3: Verify figures render**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "
from btc_core import load_model_data
from figures.bubble import build_bubble_figure
from figures.heatmap import build_heatmap_figure
from tab_defaults import bubble_defaults, heatmap_defaults
M = load_model_data()
fig1 = build_bubble_figure(M, bubble_defaults())
fig2 = build_heatmap_figure(M, heatmap_defaults())
print(f'Bubble: {len(fig1.data)} traces')
print(f'Heatmap: {len(fig2.data)} traces')
print('OK')
"
```

- [ ] **Step 4: Restore reference pkl**

```bash
cp /tmp/model_data_reference.pkl archive/btc_app/model_data.pkl
```
