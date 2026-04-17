# Economic Genesis Update — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the Quantoshi codebase to use 2009-07-25 as the "economic genesis" origin and document the rationale everywhere it appears.

**Architecture:** The genesis date flows from SP.ipynb → model_data.pkl → web app / desktop app. SP.ipynb and btc_core.py already have the correct date value. This plan fixes stale labels/comments, re-runs the notebook to regenerate the pkl, updates user-facing content (Model Info, FAQ, docs, CLAUDE.md), and rebuilds the MC cache.

**Tech Stack:** Python, Jupyter, Dash/Plotly, JSON notebook patching

**Spec:** `docs/superpowers/specs/2026-03-19-economic-genesis-design.md`

---

### Task 1: Fix stale axis labels in SP.ipynb Cell 0

**Files:**
- Create: `/tmp/patch_cell0_labels.py`
- Modify: `SP.ipynb` (Cell 0)

- [ ] **Step 1: Write the patch script**

```python
# /tmp/patch_cell0_labels.py
import json

with open("SP.ipynb") as f:
    nb = json.load(f)

src = ''.join(nb['cells'][0]['source'])

# Fix comment (line ~52) — remove stale "→ ~2010" suffix too
old = "# Bitcoin had no real market price in its first year (genesis 2009-01-09 \u2192 ~2010)."
new = "# Bitcoin had no real market price in its first year (economic genesis 2009-07-25)."
assert src.count(old) == 1, f"Comment: found {src.count(old)}"
src = src.replace(old, new)

# Fix 5 xlabel instances
old = "Years since Bitcoin genesis (Jan 2009)"
new = "Years since economic genesis (Jul 2009)"
count = src.count(old)
assert count == 5, f"Cell 0 xlabels: expected 5, found {count}"
src = src.replace(old, new)

nb['cells'][0]['source'] = src
with open("SP.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"Cell 0: fixed comment + {count} xlabels")
```

- [ ] **Step 2: Run the patch**

Run: `btc_venv/bin/python3 /tmp/patch_cell0_labels.py`
Expected: `Cell 0: fixed comment + 5 xlabels`

- [ ] **Step 3: Verify**

Run: `btc_venv/bin/python3 -c "import json; src=''.join(json.load(open('SP.ipynb'))['cells'][0]['source']); assert 'Jan 2009' not in src; assert 'genesis 2009-01-09' not in src; assert '\u2192 ~2010' not in src; print('OK')"`
Expected: `OK`

---

### Task 2: Fix stale axis labels in SP.ipynb Cell 1

**Files:**
- Create: `/tmp/patch_cell1_labels.py`
- Modify: `SP.ipynb` (Cell 1)

- [ ] **Step 1: Write the patch script**

```python
# /tmp/patch_cell1_labels.py
import json

with open("SP.ipynb") as f:
    nb = json.load(f)

src = ''.join(nb['cells'][1]['source'])

# Fix 2 instances of "Jan 2009" xlabels
old1 = "Years since Bitcoin genesis (Jan 2009)"
new1 = "Years since economic genesis (Jul 2009)"
count1 = src.count(old1)
assert count1 == 2, f"Cell 1 Jan xlabels: expected 2, found {count1}"
src = src.replace(old1, new1)

# Fix 3 instances with en-dash dates (stored as literal \u2013 in notebook JSON)
old2 = "Years since Bitcoin genesis (2009\u201301\u201303)"
new2 = "Years since economic genesis (2009\u201307\u201325)"
count2 = src.count(old2)
assert count2 == 3, f"Cell 1 en-dash xlabels: expected 3, found {count2}"
src = src.replace(old2, new2)

nb['cells'][1]['source'] = src
with open("SP.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"Cell 1: fixed {count1} Jan xlabels + {count2} en-dash xlabels")
```

- [ ] **Step 2: Run the patch**

Run: `btc_venv/bin/python3 /tmp/patch_cell1_labels.py`
Expected: `Cell 1: fixed 2 Jan xlabels + 3 en-dash xlabels`

- [ ] **Step 3: Verify**

Run: `btc_venv/bin/python3 -c "import json; src=''.join(json.load(open('SP.ipynb'))['cells'][1]['source']); assert 'Jan 2009' not in src; assert '01\u201303' not in src; print('OK')"`
Expected: `OK`

---

### Task 3: Fix btc_core.py OptGenesisPL docstring

**Files:**
- Modify: `archive/btc_app/btc_core.py:406-410`

- [ ] **Step 1: Edit the docstring**

Use the Edit tool to replace:
```python
    """Power law with optimal genesis date for highest R².

    Sweeps genesis dates to find the one that maximizes OLS R² in log-log
    space. Best fit: 2009-07-25 (197 days after actual genesis). Gaussian
    quantile bands like PowerLawModel.
    """
```
with:
```python
    """Power law with optimal genesis date for highest R².

    Sweeps genesis dates to find the one that maximizes OLS R² in log-log
    space. Best fit: 2009-07-25 (= economic genesis, offset 0 days). Gaussian
    quantile bands like PowerLawModel.
    """
```

- [ ] **Step 2: Verify**

Run: `btc_venv/bin/python3 -m py_compile archive/btc_app/btc_core.py && echo "OK"`
Expected: `OK`

---

### Task 4: Commit label fixes

- [ ] **Step 1: Commit**

```bash
git add SP.ipynb archive/btc_app/btc_core.py
git commit -m "Update stale genesis labels to economic genesis (2009-07-25)"
```

---

### Task 5: Rerun SP.ipynb to regenerate model_data.pkl

**Files:**
- Modify: `SP.ipynb` (executed in place)
- Output: `archive/btc_app/model_data.pkl`

- [ ] **Step 1: Back up current pkl**

Run: `cp archive/btc_app/model_data.pkl archive/btc_app/model_data.pkl.bak`

- [ ] **Step 2: Execute notebook**

Run:
```bash
~/.local/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 SP.ipynb
```
Expected: Notebook executes successfully, all cells complete.

- [ ] **Step 3: Verify pkl was regenerated**

Run:
```bash
btc_venv/bin/python3 -c "
import pickle
with open('archive/btc_app/model_data.pkl', 'rb') as f:
    d = pickle.load(f)
print('GENESIS_DATE:', d['GENESIS_DATE'])
print('ols_slope:', round(d['ols_slope'], 4))
print('ols_intercept:', round(d['ols_intercept'], 4))
print('bm_r2_comp:', round(d['bm_r2_comp'], 6))
"
```
Expected: `GENESIS_DATE: 2009-07-25`, coefficients printed for verification in Task 8.

- [ ] **Step 4: Commit**

```bash
git add SP.ipynb archive/btc_app/model_data.pkl
git commit -m "Regenerate model_data.pkl with economic genesis 2009-07-25"
```

---

### Task 6: Update Model Info page — intro text

**Files:**
- Modify: `btc_web/layout/model_info.py:30-33`

- [ ] **Step 1: Edit intro text**

Use the Edit tool to replace:
```python
                        "All models operate in log\u2081\u2080 space where t is years since the "
                        "Bitcoin genesis block (2009-07-25). This page documents the mathematics, "
                        "fitted coefficients, and methodology behind each.",
```
with:
```python
                        "All models operate in log\u2081\u2080 space where t is years since the "
                        "Bitcoin economic genesis (2009-07-25) \u2014 the empirically optimal origin "
                        "confirmed by three independent statistical tests. This page documents "
                        "the mathematics, fitted coefficients, and methodology behind each.",
```

- [ ] **Step 2: Syntax check**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -c "import layout.model_info; print('OK')"`
Expected: `OK`

---

### Task 7: Update Model Info page — OptGenesisPL section

**Files:**
- Modify: `btc_web/layout/model_info.py:116-163`

- [ ] **Step 1: Replace formula block (lines ~119-127)**

Use the Edit tool to replace (exact old string from model_info.py lines 119-127):
```python
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = (\alpha + z_q \cdot \sigma) + \beta \cdot \log_{10}(t - t_{\text{offset}})$$

Solved for price:

$$\text{price}(q,\, t) = 10^{\,\alpha + z_q \sigma} \cdot (t - t_{\text{offset}})^{\,\beta}$$

where $t_{\text{offset}} = 197/365.25 \approx 0.539$ years shifts the effective genesis from 2009-07-25 to **2009-07-25**.
                            """, mathjax=True, className="mb-3"),
```
with:

```python
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = (\alpha + z_q \cdot \sigma) + \beta \cdot \log_{10}(t)$$

Solved for price:

$$\text{price}(q,\, t) = 10^{\,\alpha + z_q \sigma} \cdot t^{\,\beta}$$

With economic genesis (2009-07-25) as the model origin, the optimal genesis offset is zero — this model reduces to the standard Power Law, confirming that the chosen origin is statistically optimal.
                            """, mathjax=True, className="mb-3"),
```

- [ ] **Step 2: Replace method paragraph (lines ~130-137)**

Use the Edit tool to replace:
```python
                            html.P(
                                "The standard Power Law model uses the Bitcoin genesis block "
                                "(2009-07-25) as t=0, but no trading occurred until months later. "
                                "This model sweeps 4,000 candidate genesis dates and selects the one "
                                "that maximizes OLS R\u00b2 in log-log space. The optimal date is "
                                "2009-07-25 \u2014 close to when Bitcoin first had a real exchange rate. "
                                "Gaussian quantile bands are computed identically to PL."
                            ),
```
with:
```python
                            html.P(
                                "This model validates the economic genesis choice. It sweeps 546 "
                                "candidate genesis dates (2009-01-01 through 2010-06-30) and selects "
                                "the one that maximizes OLS R\u00b2 in log-log space. The optimal date is "
                                "2009-07-25 \u2014 confirming that the economic genesis used by all models "
                                "is already the statistically optimal origin. Three independent tests "
                                "(Durbin-Watson, out-of-sample RMSE, slope stability) corroborate this."
                            ),
```

- [ ] **Step 3: Replace coefficient table (line ~141)**

Use the Edit tool to replace:
```python
                                ("Optimal genesis", "2009-07-25 (+203 days)"),
```
with:
```python
                                ("Optimal genesis", "2009-07-25 (= economic genesis)"),
```

Note: after notebook re-run, verify the α, β, σ, R² values match the standard PL. If they do, update them to match exactly.

- [ ] **Step 4: Replace comparison section (lines ~148-163)**

Use the Edit tool to replace:
```python
                            html.H6("Comparison to standard PL"),
                            html.Ul([
                                html.Li(
                                    "R\u00b2 improves from 0.961 to 0.963 \u2014 a modest but "
                                    "statistically meaningful improvement."
                                ),
                                html.Li(
                                    "Slope drops from 5.69 to 5.08 because the effective time range "
                                    "is compressed by ~0.56 years."
                                ),
                                html.Li(
                                    "The optimal date aligns with Bitcoin\u2019s first real market "
                                    "activity, suggesting the power law describes price discovery, "
                                    "not just block creation."
                                ),
                            ]),
```
with:
```python
                            html.H6("Relationship to standard PL"),
                            html.P(
                                "With the economic genesis as model origin, the optimal genesis search "
                                "converges on the same date \u2014 confirming that the chosen origin is "
                                "statistically optimal. The PL and Optimal Genesis PL models produce "
                                "identical fits."
                            ),
```

- [ ] **Step 5: Syntax check**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -c "import layout.model_info; print('OK')"`
Expected: `OK`

---

### Task 8: Verify displayed coefficients match pkl

**Files:**
- Modify: `btc_web/layout/model_info.py` (if coefficients differ)

- [ ] **Step 1: Extract coefficients from pkl**

Run:
```bash
btc_venv/bin/python3 -c "
import pickle, numpy as np
with open('archive/btc_app/model_data.pkl', 'rb') as f:
    d = pickle.load(f)
print('=== From pkl ===')
print(f\"OLS intercept: {d['ols_intercept']:.6f}\")
print(f\"OLS slope: {d['ols_slope']:.6f}\")
print(f\"bm_r2_comp: {d['bm_r2_comp']:.6f}\")
"
```

- [ ] **Step 2: Compare with displayed values in model_info.py**

Current displayed values:
- PL: α = −1.175443, β = 5.084045, σ ≈ 0.302
- OGPL: α = −1.5308, β = 5.0840, σ ≈ 0.284, R² = 0.96303

If OGPL values now match PL values (expected since offset=0), update the OGPL coefficient table to match.

- [ ] **Step 3: Update any mismatched coefficients**

Use the Edit tool to fix any values that don't match the pkl output.

- [ ] **Step 4: Commit model info changes**

```bash
git add btc_web/layout/model_info.py
git commit -m "Update Model Info page for economic genesis terminology"
```

---

### Task 9: Add FAQ entry

**Files:**
- Modify: `btc_web/layout/faq.py:48-50`

- [ ] **Step 1: Insert FAQ entry**

Use the Edit tool to insert after line 48 (after the price prediction FAQ entry's closing `},`):

```python
    {
        "q": "Why does the model start from July 2009, not the genesis block?",
        "a": (
            "Bitcoin\u2019s genesis block was mined January 3, 2009, but the network had "
            "no real market price for its first ~6 months. The model uses July 25, "
            "2009 \u2014 when Bitcoin first had real exchange activity \u2014 as its time "
            "origin. We call this the \u201ceconomic genesis.\u201d Three independent "
            "statistical tests (measuring prediction accuracy, residual randomness, "
            "and model consistency over time) all converge on this date as optimal "
            "across 546 candidates tested."
        ),
    },
```

**Important:** This inserts as entry index 3 (0-indexed), which shifts all subsequent FAQ item_ids. The FAQ uses positional item_ids (`faq-0`, `faq-1`, ...) so any direct links (`/7.N` for N≥4) will now point to different questions. This is acceptable since `/7.N` links are not widely shared.

- [ ] **Step 2: Syntax check**

Run: `PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -c "import layout.faq; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add btc_web/layout/faq.py
git commit -m "Add economic genesis FAQ entry"
```

---

### Task 10: Update architecture docs

**Files:**
- Modify: `docs/architecture.md:155-156`

- [ ] **Step 1: Edit genesis reference**

Use the Edit tool to replace:
```
where `t = (date - genesis).days / 365.25` (years since the Genesis Block,
January 3, 2009).
```
with:
```
where `t = (date - genesis).days / 365.25` (years since economic genesis,
July 25, 2009 — when Bitcoin first had real exchange activity).
```

- [ ] **Step 2: Commit**

```bash
git add docs/architecture.md
git commit -m "Update architecture.md genesis reference to economic genesis"
```

---

### Task 11: Update user manual glossary

**Files:**
- Modify: `docs/user_manual.md:444`

- [ ] **Step 1: Edit glossary entry**

Use the Edit tool to replace:
```
| **Genesis block** | Bitcoin's first block, mined January 3, 2009. All time calculations reference this date |
```
with:
```
| **Genesis block** | Bitcoin's first block, mined January 3, 2009 |
| **Economic genesis** | July 25, 2009 — the empirically optimal model origin when Bitcoin first had real exchange activity. All time calculations reference this date |
```

- [ ] **Step 2: Commit**

```bash
git add docs/user_manual.md
git commit -m "Add economic genesis glossary entry to user manual"
```

---

### Task 12: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (Repository Overview section, after line ~13)

- [ ] **Step 1: Add economic genesis note**

Use the Edit tool to insert after the line `The notebook generates \`archive/btc_app/model_data.pkl\`, which both the web app and the standalone app load at runtime.`:

```markdown

**Economic genesis:** All models use `2009-07-25` as their time origin — the "economic genesis" when Bitcoin first had real exchange activity. This date was selected by three independent statistical tests (Durbin-Watson, out-of-sample RMSE, slope stability) across 546 candidates. It is distinct from the Bitcoin genesis block (2009-01-03).
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "Add economic genesis definition to CLAUDE.md"
```

---

### Task 13: Local smoke test

Dash is a single-page app — curl won't find server-rendered text. Test by importing layout modules directly.

- [ ] **Step 1: Verify layout modules render without error**

Run:
```bash
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -c "
from layout.model_info import _model_info_tab
from layout.faq import _faq_tab
tab = _model_info_tab()
print('Model info OK')
faq = _faq_tab()
print('FAQ OK')
"
```
Expected: `Model info OK` then `FAQ OK` (no import errors or exceptions).

- [ ] **Step 2: Verify no hardcoded stale dates in dynamic files**

Run:
```bash
grep -rn "January 3, 2009\|2009-01-03\|2009-01-09\|Jan 2009" \
    btc_web/callbacks/ btc_web/figures/ btc_web/layout/model_info.py \
    btc_web/layout/faq.py btc_web/mc_overlay.py btc_web/mc_cache.py || echo "Clean"
```
Expected: `Clean` (no stale hardcoded dates in app code). Hits in splash.py or nav.py (genesis block quotes) are expected and correct — those refer to the actual Bitcoin genesis block, not the model origin.

---

### Task 14: Rebuild MC cache (production — defer to deploy time)

This task happens on the production server during deployment. Do NOT run locally.

- [ ] **Step 1: Research exact cache rebuild command**

Run: `grep -n "def.*build\|def.*cache\|def.*warm" btc_web/mc_cache.py | head -10`
Identify the correct entry point function name.

- [ ] **Step 2: Document the command for deployment**

The deploy sequence will be:
```bash
git push origin master
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && systemctl stop quantoshi"
ssh root@89.167.70.45 "cd /opt/quantoshi && btc_venv/bin/python3 -c 'from btc_web.mc_cache import <entry_point>; <entry_point>()'"
ssh root@89.167.70.45 "systemctl start quantoshi"
```

- [ ] **Step 3: Inform user**

Print: "MC cache rebuild must happen on production server at deploy time. Command documented above."

---
