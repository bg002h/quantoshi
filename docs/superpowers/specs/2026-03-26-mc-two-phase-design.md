# MC Integration Two-Phase Design

**Date:** 2026-03-26
**Branch:** `MCCitadelIntegration`

---

## Phase 1: BTC Markov Integration (COMPLETE)

BTC price paths generated via existing Cython Markov engine, fed into Citadel engine's `simulate()`. Dollar assets use user-input lognormal volatility.

### Status: Deployed

- `simulate()` accepts `price_paths` parameter — DONE
- `_mc_citadel_overlay()` in mc_overlay.py — DONE
- Figure builder wiring — DONE
- 7-output callback with `_mc_setup`/`_mc_finalize` — DONE
- MC control callbacks registered for "cp" prefix — DONE
- Snapshot/routing MC entries — DONE
- Performance: 200 sims x 168 steps = 11.8s (quantile grid cache) — DONE
- Payment wiring (BTCPay) — DONE

### What works now
- User enables MC → Markov engine generates BTC price paths → Citadel engine runs full simulation per path → fan bands for all 5 asset classes
- Rebalancing triggers fire based on BTC quantile from Markov paths
- Dollar-asset volatility uses per-sim RNG (lognormal, user-input rates)
- Payment: free tier = 1 deterministic sim, paid = N stochastic sims

### What's missing
- Dollar-asset returns are independent lognormal — no regime structure, no correlation with BTC
- Interest rates are static user inputs, not stochastic

---

## Phase 2: Stock/Bond/Treasury Transition Matrices

### Goal

Replace the simple lognormal model for dollar assets with Markov chain regime-based returns derived from historical data. This makes the multi-asset simulation realistic: equity bear/bull regimes, bond yield curve regimes, correlation structure between BTC and traditional assets.

### Data Sources

| Asset Class | Data Source | History | Granularity |
|------------|-------------|---------|-------------|
| US Equities | S&P 500 total return (SPY ETF or ^GSPC) | 1990-present | Monthly |
| US Bonds | Bloomberg US Aggregate Bond Index (AGG ETF or similar) | 2003-present | Monthly |
| T-Bills (short) | 3-month Treasury yield (^IRX or FRED DGS3MO) | 1990-present | Monthly |
| T-Notes (medium) | 5-year Treasury yield (FRED DGS5) | 1990-present | Monthly |
| T-Bonds (long) | 30-year Treasury yield (FRED DGS30) | 1990-present | Monthly |

### Architecture

**Approach A: Independent per-asset transition matrices**
Build a separate transition matrix for each asset class. Each matrix captures regime switching for that asset independently. Simple, reuses existing Cython Markov infrastructure.

- Pro: Reuses existing `build_transition_matrix` and `monte_carlo_prices` patterns
- Con: No correlation between assets — equity crash + bond rally correlation not captured

**Approach B: Joint transition matrix (correlated regimes)**
Build a single joint state space: (BTC_regime, equity_regime, bond_regime). Transition matrix captures regime co-movement. When BTC crashes, equities may also crash while bonds rally.

- Pro: Captures correlation structure — more realistic
- Con: State space explosion (5 BTC bins x 5 equity bins x 5 bond bins = 125 states). Requires longer history for reliable estimation. Much more complex.

**Recommended: Approach A for v2, with correlation overlay for v3.**

Phase 2 builds independent transition matrices per asset class. The Citadel engine's step function draws returns from each asset's matrix independently. This is achievable now and significantly better than static lognormal. Correlation can be added later via copula or joint-state model.

### Implementation Plan

#### 2a. Historical data pipeline

Create `btc_web/data/` directory with:
- `fetch_historical.py` — download historical returns from Yahoo Finance / FRED
- `equity_returns.csv` — S&P 500 monthly total returns
- `bond_returns.csv` — aggregate bond monthly returns
- `treasury_yields.csv` — 3mo/5yr/30yr monthly yields

For Treasury bins, convert yields to total returns using duration approximation:
```
monthly_return ≈ yield/12 - duration × Δyield
```

#### 2b. Asset transition matrix builder

Extend the existing `build_transition_matrix()` pattern or create a new `build_asset_transition_matrix()` in `btc_web/engines/`:
- Input: monthly return series
- Output: (n_bins x n_bins) transition matrix + bin edges (return percentiles)
- Same bin/regime/window parameters as BTC matrix

#### 2c. Engine integration

Modify `engines/citadel.py` step function:
- When MC mode: instead of `_lognormal_return()`, draw from asset-specific transition matrices
- Each asset class gets its own matrix and current state (regime bin)
- `CitadelState` gains `equity_regime: int`, `bond_regime: int`, `res_short_regime: int`, etc.
- Returns are sampled from the transition matrix bin for each step

#### 2d. UI controls

Add to the Citadel Planner Rules or Simulation sub-tab:
- "Asset return model" toggle: Lognormal (current) vs Markov (historical regimes)
- When Markov: show regime bin visualization, historical window selector
- Reserve bin returns switch from user-input to Treasury yield model

### Not in Phase 2 scope

- Cross-asset correlation (v3 — joint transition matrix)
- Real-time data updates (manual CSV refresh for now)
- Stochastic yield curve model (v3)
- Celery/Redis async (v3)
