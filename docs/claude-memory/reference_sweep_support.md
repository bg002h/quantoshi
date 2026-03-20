---
name: Support parameter sweep tool
description: tools/sweep_support.py sweeps SUPPORT_PERCENTILE × SUPPORT_QUANTILE to find optimal bubble model R²
type: reference
---

`tools/sweep_support.py` — 2D grid sweep over bubble model support line parameters.

**Usage:**
```bash
btc_venv/bin/python3 tools/sweep_support.py [--pct-lo 5] [--pct-hi 50] [--pct-step 5] [--q-lo 0.1] [--q-hi 0.9] [--q-step 0.1] [--out sweep_support.jpg]
```

Reads config (BUBBLE_YEARS, FIT_MIN_DATE, etc.) from SP.ipynb cell 0 automatically. Outputs a 2-panel heatmap (R² composite + support slope) and prints top 10 results.

**How to apply:** Run after changing BUBBLE_YEARS, FIT_MIN_DATE, or genesis date to re-optimize the support line parameters.
