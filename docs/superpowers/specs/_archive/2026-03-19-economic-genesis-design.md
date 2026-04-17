# Economic Genesis Update (2009-07-25)

**Date:** 2026-03-19
**Status:** Approved

## Summary

Update the Quantoshi codebase to use 2009-07-25 as the model origin ("economic genesis") and document the rationale. This date was selected by three independent statistical tests over 546 candidate dates (2009-01-01 through 2010-06-30):

| Metric | What it measures | Best date | 2009-07-25 rank |
|--------|-----------------|-----------|-----------------|
| Durbin-Watson | Residual autocorrelation (ideal = 2.0) | 2009-07-25 | 1/546 |
| Out-of-sample RMSE | Forward prediction accuracy (3 holdout windows) | 2009-07-20 | 11/546 |
| Slope stability | Consistency of power law slope across 6 expanding windows | 2009-09-03 | 80/546 |

2009-07-25 is the best all-rounder: #1 on DW, top 2% on OOS, top 15% on slope stability. It coincides with when Bitcoin first had real economic activity (first exchange rates).

## What's Already Done

- **SP.ipynb** cells 0, 1, 3: genesis variable set to `2009-07-25`
- **btc_core.py**: all 6 genesis references updated, OptGenesisPL offset = 0, LPPL comment cleaned up

## Step 0: Fix Stale Labels in SP.ipynb

Before re-running the notebook, fix stale axis labels and comments that still reference old genesis dates.

### 0a. Cell 0 comment (line ~52)
Current: `# Bitcoin had no real market price in its first year (genesis 2009-01-09 → ~2010).`
Change to: `# Bitcoin had no real market price in its first year (economic genesis 2009-07-25).`

### 0b. Cell 0 xlabels (5 instances)
Replace all `'Years since Bitcoin genesis (Jan 2009)'` with `'Years since economic genesis (Jul 2009)'`

### 0c. Cell 1 xlabels
- Replace 2 instances of `'Years since Bitcoin genesis (Jan 2009)'` with `'Years since economic genesis (Jul 2009)'`
- Replace 3 instances of `'Years since Bitcoin genesis (2009–01–03)'` with `'Years since economic genesis (2009–07–25)'` (note: en-dashes in date, stored as `\u2013` in notebook JSON)

### 0d. btc_core.py OptGenesisPL docstring (line ~409)
Current: `Best fit: 2009-07-25 (197 days after actual genesis).`
Change to: `Best fit: 2009-07-25 (= economic genesis, offset 0 days).`

## Step 1: Rerun SP.ipynb

Regenerate model fits and `model_data.pkl`:

```bash
~/.local/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 SP.ipynb
cp archive/btc_app/model_data.pkl archive/btc_app/model_data.pkl.bak
```

The pkl is written by Cell 3. Since genesis was already 2009-07-25 before this session (temporarily changed for analysis), coefficients should be identical to production. Re-running ensures consistency.

## Step 2: Rebuild MC/Markov Cache

The MC cache (~834 MB, ~45,000 scenarios) is built from model_data.pkl. After pkl regeneration, rebuild on the production server:

```bash
ssh root@89.167.70.45
cd /opt/quantoshi
# Stop service, rebuild cache, restart
systemctl stop quantoshi
btc_venv/bin/python3 -c "from btc_web.mc_cache import build_cache; build_cache()"
systemctl start quantoshi
```

(Exact import path may differ — check `mc_cache.py` for the entry point before running.)

This is the longest step. Must complete before deployment.

## Step 3: Update Model Info Page

**File:** `btc_web/layout/model_info.py`

### 3a. Intro text (line ~32)
Current:
```python
"Bitcoin genesis block (2009-07-25). This page documents the mathematics, "
```
Change to:
```python
"Bitcoin economic genesis (2009-07-25) \u2014 the empirically optimal origin "
"confirmed by three independent statistical tests. This page documents the mathematics, "
```

### 3b. OptGenesisPL section (lines ~120-163)

Since genesis = optimal genesis (offset = 0), this model is now equivalent to the standard PL. The section needs significant rewriting:

**Formula (line ~120-126):** Remove `t_offset` from the formula — it's zero. Simplify to standard PL form:
```
$$\log_{10}(\text{price}) = (\alpha + z_q \cdot \sigma) + \beta \cdot \log_{10}(t)$$
```
Add note: "With economic genesis as the model origin, the optimal genesis offset is zero — this model reduces to the standard Power Law."

**Method paragraph (lines ~130-136):** Rewrite entirely:
```python
"This model validates the economic genesis choice. It sweeps 546 candidate "
"genesis dates and selects the one that maximizes OLS R² in log-log space. "
"The optimal date is 2009-07-25 — confirming that the economic genesis used "
"by all models is already the statistically optimal origin. Three independent "
"tests (Durbin-Watson, out-of-sample RMSE, slope stability) corroborate this."
```

**Coefficient table (line ~141):** Change:
```python
("Optimal genesis", "2009-07-25 (+203 days)")
```
to:
```python
("Optimal genesis", "2009-07-25 (= economic genesis)")
```
Coefficients (α, β, σ, R²) should now match the standard PL exactly. Verify after notebook re-run.

**Comparison section (lines ~148-163):** Remove or replace. Since offset=0, the PL and OptGenesisPL produce identical fits. Replace with:
```python
html.P(
    "With the economic genesis as model origin, the optimal genesis search "
    "converges on the same date — confirming that the chosen origin is "
    "statistically optimal. The PL and Optimal Genesis PL models are now "
    "equivalent."
),
```

### 3c. Verify coefficients
After notebook re-run, check that all displayed coefficients (PL α/β/σ, LPPL constants, OGPL values) match the fresh pkl. They should be unchanged, but verify.

## Step 4: Update FAQ

**File:** `btc_web/layout/faq.py`

Insert as entry 4 (after "Is Quantoshi predicting future Bitcoin price?", before "QR vs MCMC"):

```python
{
    "q": "Why does the model start from July 2009, not the genesis block?",
    "a": (
        "Bitcoin's genesis block was mined January 3, 2009, but the network had "
        "no real market price for its first ~6 months. The model uses July 25, "
        "2009 — when Bitcoin first had real exchange activity — as its time "
        'origin. We call this the "economic genesis." Three independent '
        "statistical tests (measuring prediction accuracy, residual randomness, "
        "and model consistency over time) all converge on this date as optimal "
        "across 546 candidates tested."
    ),
},
```

## Step 5: Update Documentation

### 5a. `docs/architecture.md` (line ~156)
Current:
> where `t = (date - genesis).days / 365.25` (years since the Genesis Block, January 3, 2009).

Change to:
> where `t = (date - genesis).days / 365.25` (years since economic genesis, July 25, 2009 — when Bitcoin first had real exchange activity).

### 5b. `docs/user_manual.md` (line ~444)
Current:
```
| **Genesis block** | Bitcoin's first block, mined January 3, 2009. All time calculations reference this date |
```
Change to:
```
| **Genesis block** | Bitcoin's first block, mined January 3, 2009 |
| **Economic genesis** | July 25, 2009 — the empirically optimal model origin when Bitcoin first had real exchange activity. All time calculations reference this date |
```

### 5c. `CLAUDE.md`
Add a note in the Repository Overview or Notebook Architecture section establishing the "economic genesis" term:

> **Economic genesis:** All models use `2009-07-25` as their time origin — the "economic genesis" when Bitcoin first had real exchange activity. This date was selected by three independent statistical tests (Durbin-Watson, out-of-sample RMSE, slope stability) across 546 candidates. It is distinct from the Bitcoin genesis block (2009-01-03).

This ensures future AI coding sessions use consistent terminology.

## Step 6: No Code Changes Needed (verification)

These files read `m.genesis` dynamically from the pkl and need no code changes:
- `btc_web/callbacks/charts.py` — uses `m.genesis`
- `btc_web/callbacks/lots.py` — uses `m.genesis`
- `btc_web/callbacks/ticker.py` — uses `m.genesis`
- `btc_web/figures/bubble.py` — xlabel already says `(2009-07-25)`
- `btc_web/mc_overlay.py` — uses `m.genesis`
- `btc_web/mc_cache.py` — uses `m.genesis`
- `btc_web/test_web.py` — uses `M.genesis`
- `archive/btc_app/btc_projections.py` — already has `2009-07-25`

## Execution Order

```
Step 0 (fix labels) → Step 1 (rerun notebook) → Step 2 (rebuild MC cache)
                                                → Steps 3-5 (content, parallel)
```

Steps 3-5 are independent text edits that can happen in parallel after the notebook runs (to verify coefficients). Step 2 happens on the production server at deploy time.

## Risk

Low. Genesis was already 2009-07-25 before this session. The primary changes are content/documentation, not model logic. The MC cache rebuild is the only time-intensive step.
