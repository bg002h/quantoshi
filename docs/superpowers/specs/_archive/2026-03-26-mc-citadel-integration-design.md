# MC Citadel Integration — Design Spec

**Date:** 2026-03-26
**Scope:** Wire Markov chain Monte Carlo simulation into the Citadel Planner (Tab 9) — BTC price paths, fan band charts, BTCPay payment integration.
**Branch:** `MCCitadelIntegration`

---

## Goal

Enable Monte Carlo simulation on Tab 9 so users can run N stochastic simulations (default 200) showing fan band percentile charts across all asset classes. Free tier: 1 deterministic sim. Paid tier: N sims via existing BTCPay Lightning/on-chain payment flow. The MC integration reuses the existing Markov engine for BTC price paths and runs the full Citadel step loop (rebalancing, floors, spending, SCF) for each path.

---

## Architecture

### Why not a new Cython function

The existing MC tabs (DCA, Retire, SC) use Cython functions (`mc_retire()`, `mc_dca()`) for the inner simulation loop. These are simple: for each price path, compute cumulative BTC balance after withdrawals/deposits. The Citadel engine is fundamentally more complex — it has:

- Multi-asset returns with lognormal volatility
- Rebalancing triggers that modify BTC holdings based on quantile position
- Account floor enforcement that can sell BTC
- Spending waterfall across 7+ accounts
- SCF loan payments and repayment triggers

Writing this in Cython would duplicate the entire engine and create a maintenance nightmare. Instead:

### Approach: Pre-generated price paths + Python engine loop

1. **Generate BTC price paths** via `monte_carlo_prices()` (existing Markov Cython function) — produces `(n_sims, n_steps)` array of BTC prices
2. **For each sim**, feed the price path into the Citadel `simulate()` function — the step loop runs in Python with full rebalancing/floor/spending logic
3. **Aggregate** all sim results into fan band percentiles via `compute_fan_percentiles()`
4. **Render** fan bands on the chart using the existing `_mc_build_traces()` pattern

### Performance estimate

- 200 sims x 528 steps (44yr monthly) = 105,600 step() calls
- Each step: lognormal sampling (5 bins), spending waterfall, conditional rebalancing, floor checks = ~200-500us in CPython
- Total estimate: 20-50 seconds for 200 sims (conservative)
- The Markov price path generation is sub-second (Cython)
- **Benchmark required:** Implementation must include a timing test. If > 30s, optimize by vectorizing non-BTC returns across sims (use numpy broadcasting instead of per-sim rng calls)
- **Optimization path:** The step function's hot loop can be restructured to operate on arrays of shape (n_sims,) instead of scalars. This requires no architectural change — just array operations instead of scalar ones in the inner loop. Deferred unless benchmarks require it.

---

## Changes

### 1. Engine: `btc_web/engines/citadel.py`

**Modify `simulate()` to accept optional pre-generated price paths:**

```python
def simulate(config: SimConfig, model: PriceModel,
             rng_seed: int = 42,
             price_paths: np.ndarray | None = None) -> SimResult:
```

When `price_paths` is provided (shape `(n_sims, n_periods)`):
- The sim loop count is `price_paths.shape[0]` (overrides `config.n_sims`)
- Skip `_get_btc_price()` call — use `price_paths[sim_id, period_idx]` directly
- `config.n_sims` on the SimConfig is NOT mutated — the loop count comes from the price_paths array shape
- `validate_config()` runs before the loop with the original config (n_sims may be 1 or >1)
- For dollar-asset volatility: each sim gets a unique RNG derived from `rng_seed + sim_id` for reproducible stochastic returns across runs

When `price_paths` is None (deterministic mode):
- Existing behavior: uses `model.price_at(q, t)` for single quantile path
- `config.n_sims` must be 1

This is the minimal change to support MC — the step function, rebalancing, floors, spending, and SCF all work unchanged.

**Important:** The `_aggregate_results()` function already computes `np.median()` and `np.percentile()` across axis=0 (sims dimension). It was designed for multi-sim from the start but only tested with n_sims=1. The MC integration exercises this code path for the first time — tests must verify shapes and percentile correctness for n_sims=200.

### 2. MC overlay: `btc_web/mc_overlay.py`

**Add `_mc_citadel_overlay()` function** following the existing `_mc_withdraw_overlay()` pattern.

The function:
1. Extracts MC params via `_mc_setup_vars(p)`
2. Builds MC timeline via `_build_mc_timeline(p, m, mc_years, mc_dt)`
3. 3-level cache fallthrough:
   - Client-side cache (`mc_cached` store)
   - Pre-computed server cache (not applicable for v2 Citadel — too many config combinations to pre-compute)
   - Live simulation
4. For live simulation:
   - Calls `build_transition_matrix()` and `monte_carlo_prices()` to get BTC price paths
   - Passes `price_paths` into `engines.citadel.simulate(config, model, price_paths=price_paths)`
   - Extracts per-asset fan bands from `SimResult.median` and `SimResult.percentiles`
5. Builds fan band traces for each asset class (Total, BTC USD, Cash, Reserves, Investments)
6. Returns `(traces, annots, result_dict)`

**Key difference from existing overlays:** The existing overlays produce fan bands for a single metric (BTC balance or USD value). The Citadel overlay produces fan bands for **5 asset classes** simultaneously. Each gets its own set of percentile bands with distinct colors matching the deterministic trace colors.

### 3. Figure builder: `btc_web/figures/citadel.py`

**Add MC overlay integration:**

```python
if _HAS_MARKOV and p.get("mc_enabled"):
    mc_traces, mc_result = _mc_citadel_overlay(m, p, config, model)
    traces.extend(mc_traces)
```

Remove the `mc_requested` fallback that currently forces `n_sims=1`. When MC is enabled and the Markov module is available, run the full MC simulation.

Return `(fig, mc_result)` instead of `(fig, None)`.

### 4. Callback: `btc_web/callbacks/citadel_cb.py`

**Upgrade to the 7-output MC callback pattern:**

```python
@callback(
    Output("citadel-graph", "figure"),
    Output("cp-mc-results", "data"),
    Output("cp-mc-status", "children"),
    Output("cp-mc-rendered-key", "data"),
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Output("mc-save-tab", "data", allow_duplicate=True),
    Output("cp-mc-unblocked", "data"),
    # ... all existing Inputs ...
    # Add MC Inputs:
    Input("cp-mc-enable", "value"),
    Input("cp-mc-bins", "value"),
    Input("cp-mc-regime", "value"),
    Input("cp-mc-sims", "value"),
    Input("cp-mc-years", "value"),
    Input("cp-mc-window", "value"),
    Input("cp-mc-start-yr", "value"),
    Input("cp-mc-entry-q", "value"),
    Input("cp-mc-loaded", "data"),
    Input("mc-pay-trigger", "data"),
    Input("cp-mc-model-src", "value"),
    # Add MC States:
    State("btc-price-store", "data"),
    State("cp-mc-results", "data"),
    State("mc-pay-token", "data"),
    State("cp-mc-unblocked", "data"),
    State("cp-mc-rendered-key", "data"),
    prevent_initial_call=True,
)
def update_citadel(...):
    # MC setup
    mc_ok, is_free, mc_p, blocked = _mc_setup("cp", mc_enable, ...)
    # Build figure (with MC params merged)
    fig, mc_result = _get_citadel_fig(dict(..., **mc_p))
    # MC finalize
    fig, store_val, status, rendered_key, show_modal, ub_val = _mc_finalize(
        "cp", fig, mc_result, ...)
    return (fig, store_val, status, rendered_key, show_modal,
            "cp" if show_modal else dash.no_update, ub_val)
```

**CRITICAL:** The callback now has `allow_duplicate=True` on `mc-save-modal.is_open` and `mc-save-tab.data` — these are shared outputs across all MC-enabled tabs. The `apply_snapshot` callback also outputs to MC stores with `allow_duplicate=True`. Must ensure no duplicate-output violations.

### 5. Payment: `btc_web/callbacks/mc_payment.py`

**Add Citadel to `_MC_BTN_TO_TAB`:**

```python
_MC_BTN_TO_TAB["cp-mc-run-btn"] = "cp"
```

**Update `_mc_payment_initiate` callback signature:**

The callback at `mc_payment.py:31-59` uses positional args with 4 Input buttons and `State` groups per tab. Adding a 5th tab (cp) requires:
1. Add `Input("cp-mc-run-btn", "n_clicks")` to the button inputs
2. Add `State("cp-mc-years", "value")`, `State("cp-mc-start-yr", "value")`, `State("cp-mc-entry-q", "value")` to the state groups
3. Add `State("cp-mc-model-src", "value")` and `State("cp-mc-price-val", "data")` to model/price states
4. Add `Output("cp-mc-run-status", "children")` to outputs
5. Update the positional index arithmetic (`state_base + tab_idx * 3`, etc.) for the 5th tab

The payment flow works identically to existing tabs — the user clicks Run Simulation, BTCPay invoice is created, Lightning payment is made, token is stored, chart callback re-fires with valid token.

**Frequency alignment:** The MC engine generates price paths with a step size determined by `mc_dt`. The Citadel engine steps at `config.freq` intervals. These MUST match. The `_mc_controls("cp", ..., shared_controls={"freq"})` should suppress the MC frequency dropdown so the tab's `cp-freq` is used for both MC path generation and engine stepping.

### 6. Snapshot: `btc_web/snapshot.py`

MC control IDs are NOT yet in `_SNAPSHOT_CONTROLS` — v1 only added the simulation/display controls. The following 10 entries must be appended to `_SNAPSHOT_CONTROLS`:

```python
# Citadel MC controls (append after existing cp-* entries)
("cp-mc-enable",    "value"),
("cp-mc-start-yr",  "value"),
("cp-mc-entry-q",   "value"),
("cp-mc-years",     "value"),
("cp-mc-bins",      "value"),
("cp-mc-regime",    "value"),
("cp-mc-sims",      "value"),
("cp-mc-window",    "value"),
("cp-mc-advanced",  "value"),
("cp-mc-model-src", "value"),
```

Add to `_CHECKLIST_OPTIONS`:
```python
"cp-mc-enable":   ["yes"],
"cp-mc-advanced": ["yes"],
"cp-mc-regime":   [0, 1, 2, 3, 4],
```

Add `cp-mc-model-src` to `_TAB_CONTROLS["citadel"]` alongside the other 9 MC control IDs.

**CRITICAL: Atomic deployment.** The `_CHECKLIST_OPTIONS` assertion at module load checks all keys exist in `_SNAPSHOT_CONTROLS`. Both changes must be in the same commit.

### 7. Nav: `btc_web/callbacks/nav.py`

**Add MC control IDs to `_TAB_CONTROLS["citadel"]`:**

```python
_TAB_CONTROLS["citadel"].update({
    "cp-mc-enable", "cp-mc-start-yr", "cp-mc-entry-q", "cp-mc-years",
    "cp-mc-bins", "cp-mc-regime", "cp-mc-sims", "cp-mc-window",
    "cp-mc-advanced",
})
```

---

## Fan Band Chart Design

### Multi-asset fan bands

Each asset class gets its own fan band set, color-matched to its deterministic trace:

| Asset | Base Color | Fan Alpha |
|-------|-----------|-----------|
| Total Portfolio | Black (#000) | 0.08 / 0.12 / 0.18 |
| BTC Holdings USD | Orange (#F7931A) | 0.08 / 0.12 / 0.18 |
| Cash | Silver (#C0C0C0) | 0.06 / 0.10 / 0.15 |
| Reserves | Blue (#4A90D9) | 0.06 / 0.10 / 0.15 |
| Investments | Green (#27AE60) | 0.06 / 0.10 / 0.15 |

Fan bands: 5th-95th (lightest), 25th-75th (medium), median (solid line).

The spending trace does NOT get fan bands (spending is deterministic in v2).

### Building fan band traces

The existing `_mc_build_traces()` in `mc_overlay.py:441` uses a fixed orange color scheme and builds traces for a single series. For the Citadel overlay with 5 asset classes, we will **NOT reuse `_mc_build_traces()`**. Instead, `_mc_citadel_overlay()` builds traces directly using `go.Scatter(fill="tonexty")` pairs, parameterized with per-asset colors. This avoids modifying the shared helper function.

For each asset class, the overlay builds:
1. Lower 5th percentile line (invisible, width=0)
2. Upper 95th percentile fill to previous (lightest alpha)
3. Lower 25th percentile line (invisible)
4. Upper 75th percentile fill to previous (medium alpha)
5. Median line (solid, matching base color)

Total MC traces: 5 assets x 5 traces = 25 traces. With 6 deterministic + 1 spending = 32 traces total. This is within Plotly's comfortable rendering range.

### Data format mapping

`SimResult.percentiles` is `{pct_int: {asset_key: ndarray}}` (e.g., `{5: {"total": array, "cash": array, ...}}`). The overlay code reshapes this into per-asset fan dicts:
```python
for asset_key in ["total", "btc_usd", "cash", "reserves_total", "investments_total"]:
    fan = {pct/100: result.percentiles[pct][asset_key] for pct in [5, 25, 75, 95]}
    fan[0.50] = result.median[asset_key]
    # Build traces for this asset...
```

### Legend

Each asset's fan band group shares a legendgroup so clicking the legend toggles all bands + median for that asset.

---

## v2 Feature Additions (beyond MC)

### Feature: Model display checklist for Citadel

Add `cp-model-show` checklist (matching existing tabs) so users can toggle:
- QR (deterministic quantile path) — on by default
- MC (Markov fan bands) — on when MC enabled

This follows the existing `{prefix}-model-show` pattern on DCA/Retire/SC tabs.

### Feature: MC status badge

Show MC computation status (computing/ready/payment required) below the chart, matching the existing `{prefix}-mc-status` pattern.

---

## Not in Scope (v2)

- Pre-computed MC cache for Citadel (too many config combinations)
- Equity/bond return triggers (v3)
- BTC-to-equity ratio trigger (v3)
- Spending windows (v3)
- Adaptive spending cuts (v3)
- Stochastic interest rates (v3)
- Celery/Redis async (v3)
- App-wide save/load (Sub-project B)

---

## Testing

| Test | What |
|------|------|
| `simulate()` with price_paths | Verify n_sims > 1 produces correct shapes |
| Fan band percentiles | Verify 5/25/50/75/95 computed correctly |
| MC callback smoke test | Verify 7-output callback returns valid figure |
| Payment button wiring | Verify `cp-mc-run-btn` appears in `_MC_BTN_TO_TAB` |
| Snapshot MC controls | Verify MC control IDs roundtrip through encode/decode |
| No duplicate outputs | Existing guard test still passes |
| Deterministic unchanged | Verify n_sims=1 still works identically |
