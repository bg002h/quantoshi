# MC Citadel Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Markov chain Monte Carlo simulation into the Citadel Planner so users can run N stochastic simulations with fan band percentile charts and BTCPay payment.

**Architecture:** Generate BTC price paths via existing Cython Markov engine (`monte_carlo_prices()`), feed each path into the Python Citadel `simulate()` function, aggregate into fan bands. Reuse the existing `_mc_setup`/`_mc_finalize` callback pattern and BTCPay payment flow.

**Tech Stack:** Python 3.12+, NumPy, Plotly, Dash 4.0.0, Cython Markov module

**Spec:** `docs/superpowers/specs/2026-03-26-mc-citadel-integration-design.md`

---

## File Structure

### Modified Files
| File | Change |
|------|--------|
| `btc_web/engines/citadel.py` | Add `price_paths` parameter to `simulate()`, handle multi-sim loop |
| `btc_web/mc_overlay.py` | Add `_mc_citadel_overlay()` function |
| `btc_web/figures/citadel.py` | Wire MC overlay into figure builder |
| `btc_web/callbacks/citadel_cb.py` | Upgrade to 7-output MC callback with `_mc_setup`/`_mc_finalize` |
| `btc_web/callbacks/mc_payment.py` | Add `cp` to `_MC_BTN_TO_TAB` and payment callback |
| `btc_web/snapshot.py` | Add 10 MC control entries to `_SNAPSHOT_CONTROLS` + `_CHECKLIST_OPTIONS` |
| `btc_web/callbacks/nav.py` | Add MC control IDs to `_TAB_CONTROLS["citadel"]` |
| `btc_web/test_citadel.py` | Add MC simulation tests |

---

### Task 1: Engine — accept price_paths in simulate()

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Modify: `btc_web/test_citadel.py`

- [ ] **Step 1: Write failing test**

```python
class TestSimulateMultiSim:
    def test_price_paths_produces_multi_sim(self):
        """simulate() with price_paths should produce n_sims results."""
        cfg = SimConfig.default()
        cfg.start_yr = 2031
        cfg.end_yr = 2033  # 24 months
        model = _mock_model_data()
        # Generate fake price paths: 5 sims x 24 periods
        price_paths = np.array([[50000 + i*100 + j*10 for j in range(24)]
                                for i in range(5)])
        result = simulate(cfg, model, price_paths=price_paths)
        assert result.btc_holdings.shape == (5, 24)
        assert result.total_usd.shape == (5, 24)
        assert len(result.depletion_period) == 5
        # Median should exist for each asset class
        assert "total" in result.median
        assert "btc_usd" in result.median
        assert result.median["total"].shape == (24,)

    def test_price_paths_different_from_deterministic(self):
        """MC results should differ from deterministic when paths vary."""
        cfg = SimConfig.default()
        cfg.start_yr = 2031
        cfg.end_yr = 2032  # 12 months
        model = _mock_model_data()
        # Deterministic
        result_det = simulate(cfg, model)
        # MC with varying paths
        base = result_det.btc_prices[0]  # use deterministic prices as base
        paths = np.array([base * (0.8 + 0.1*i) for i in range(5)])  # 5 scaled paths
        result_mc = simulate(cfg, model, price_paths=paths)
        # Percentile spread should be nonzero
        p5 = result_mc.percentiles[5]["total"]
        p95 = result_mc.percentiles[95]["total"]
        assert np.any(p95 > p5), "Fan bands should have nonzero spread"
```

- [ ] **Step 2: Implement price_paths support in simulate()**

Modify `simulate()` in `engines/citadel.py`:

```python
def simulate(config: SimConfig, model: PriceModel,
             rng_seed: int = 42,
             price_paths: np.ndarray | None = None) -> SimResult:
    validate_config(config)
    ppy = FREQ_PPY[config.freq]
    n_periods = _compute_n_periods(config)

    # Determine sim count
    if price_paths is not None:
        n_sims = price_paths.shape[0]
        assert price_paths.shape[1] >= n_periods, \
            f"price_paths has {price_paths.shape[1]} steps, need {n_periods}"
    else:
        n_sims = 1  # deterministic mode

    from btc_core import yr_to_t
    t0 = yr_to_t(config.start_yr)
    dt = 1.0 / ppy
    time_axis = np.array([t0 + i * dt for i in range(n_periods)])

    all_histories = []
    for sim_id in range(n_sims):
        # Each sim gets a unique RNG for dollar-asset volatility
        rng = np.random.default_rng(rng_seed + sim_id)
        state = _initial_state(config, model=model)
        history = []
        for period_idx in range(n_periods):
            if price_paths is not None:
                btc_price = float(price_paths[sim_id, period_idx])
            else:
                q = config.selected_qs[len(config.selected_qs)//2] if config.selected_qs else 0.5
                btc_price = _get_btc_price(time_axis[period_idx], config, model, rng,
                                           sim_mode="deterministic", q=q)
            new_state = step(state, config, btc_price, rng, model=model)
            history.append(_snapshot_state(new_state))
            state = new_state
        all_histories.append(history)

    return _aggregate_results(all_histories, config, time_axis)
```

- [ ] **Step 3: Run tests, verify pass**

Run: `PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_citadel.py -v -k "multi_sim or price_paths"`

- [ ] **Step 4: Benchmark 200 sims x 528 steps**

```python
import time
cfg = SimConfig.default()
cfg.start_yr = 2031
cfg.end_yr = 2075  # 528 steps
paths = np.full((200, 528), 100000.0)  # constant price for timing
t0 = time.time()
result = simulate(cfg, _mock_model_data(), price_paths=paths)
elapsed = time.time() - t0
print(f"200 sims x 528 steps: {elapsed:.1f}s")
# If > 30s, flag for optimization
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_citadel.py
git commit -m "feat(citadel-mc): accept price_paths in simulate() for MC mode"
```

---

### Task 2: MC overlay function

**Files:**
- Modify: `btc_web/mc_overlay.py`

- [ ] **Step 1: Add `_mc_citadel_overlay()` function**

This function follows the `_mc_withdraw_overlay()` pattern but:
- Runs the full Citadel engine for each BTC price path
- Produces fan bands for 5 asset classes (not just 1)
- Does NOT use `_mc_build_traces()` — builds per-asset colored bands directly

```python
def _mc_citadel_overlay(m, p, citadel_config, citadel_model):
    """MC overlay for Citadel Planner — runs full engine simulation per path.

    Returns (traces, result_dict) where traces is a list of Plotly fan band
    traces for Total/BTC/Cash/Reserves/Investments.
    """
    if not _HAS_MARKOV:
        return [], None

    mc_enabled = p.get("mc_enabled")
    if not mc_enabled:
        return [], None

    # Extract MC params
    mc_years = p.get("mc_years", 40)
    mc_start_yr = p.get("mc_start_yr", 2031)
    mc_entry_q = p.get("mc_entry_q", 10)
    mc_bins = p.get("mc_bins", 5)
    mc_sims = p.get("mc_sims", 200)
    mc_window = p.get("mc_window", [2010, 2025])
    mc_regime = p.get("mc_regime", list(range(5)))
    mc_model_src = p.get("mc_model_src", "bub")

    # Build transition matrix and generate BTC price paths
    from btc_core import yr_to_t
    ppy = citadel_config.freq_ppy if hasattr(citadel_config, 'freq_ppy') else 12
    n_periods = int((citadel_config.end_yr - citadel_config.start_yr) * ppy)
    t_start = yr_to_t(citadel_config.start_yr)
    dt = 1.0 / ppy
    step_days = int(365.25 / ppy)

    # Use markov engine
    from markov import build_transition_matrix, monte_carlo_prices
    trans, bin_edges, info = build_transition_matrix(
        ..., mc_bins, mc_window[0], mc_window[1], step_days)

    start_pctile = mc_entry_q / 100.0
    price_paths, states = monte_carlo_prices(
        trans, bin_edges, start_pctile, n_periods, mc_sims,
        mc_model_src, t_start, dt)

    # Run full Citadel simulation with these price paths
    from engines.citadel import simulate
    result = simulate(citadel_config, citadel_model, price_paths=price_paths)

    # Build fan band traces per asset class
    traces = []
    _ASSET_COLORS = {
        "total": "#000000",
        "btc_usd": "#F7931A",
        "cash": "#C0C0C0",
        "reserves_total": "#4A90D9",
        "investments_total": "#27AE60",
    }
    _FAN_ALPHAS = [(5, 95, 0.06), (25, 75, 0.12)]  # (lo_pct, hi_pct, alpha)

    ts = result.time_axis
    for asset_key, base_color in _ASSET_COLORS.items():
        # Median line
        med = result.median[asset_key]
        traces.append(go.Scatter(
            x=list(ts), y=list(med), mode="lines",
            name=f"MC {asset_key} median",
            line=dict(color=base_color, width=1.5, dash="dot"),
            legendgroup=f"mc_{asset_key}",
            showlegend=False,
        ))
        # Fan bands
        for lo_pct, hi_pct, alpha in _FAN_ALPHAS:
            lo = result.percentiles[lo_pct][asset_key]
            hi = result.percentiles[hi_pct][asset_key]
            # Lower bound (invisible)
            traces.append(go.Scatter(
                x=list(ts), y=list(lo), mode="lines",
                line=dict(width=0), showlegend=False,
                legendgroup=f"mc_{asset_key}",
            ))
            # Upper bound with fill
            fill_color = _hex_to_rgba(base_color, alpha)
            traces.append(go.Scatter(
                x=list(ts), y=list(hi), mode="lines",
                fill="tonexty", fillcolor=fill_color,
                line=dict(width=0), showlegend=False,
                legendgroup=f"mc_{asset_key}",
            ))

    # Build result dict for client-side caching
    result_dict = result.to_dict()

    return traces, result_dict
```

Note: The actual implementation will need to read `mc_overlay.py` carefully to use the correct `build_transition_matrix` arguments (they depend on the model's price data). The pseudocode above shows the structure.

- [ ] **Step 2: Commit**

```bash
git add btc_web/mc_overlay.py
git commit -m "feat(citadel-mc): add _mc_citadel_overlay with per-asset fan bands"
```

---

### Task 3: Figure builder — wire MC overlay

**Files:**
- Modify: `btc_web/figures/citadel.py`

- [ ] **Step 1: Replace MC fallback with real MC integration**

Remove the `mc_requested` / "coming soon" fallback. Add:

```python
from figures.common import _HAS_MARKOV

# After building deterministic traces...
mc_result = None
if _HAS_MARKOV and p.get("mc_enabled"):
    from mc_overlay import _mc_citadel_overlay
    mc_traces, mc_result = _mc_citadel_overlay(m, p, config, model)
    traces.extend(mc_traces)

return _finalize_chart(traces, layout, p, "cp", mc_result)
```

- [ ] **Step 2: Commit**

```bash
git add btc_web/figures/citadel.py
git commit -m "feat(citadel-mc): wire MC overlay into figure builder"
```

---

### Task 4: Callback — upgrade to 7-output MC pattern

**Files:**
- Modify: `btc_web/callbacks/citadel_cb.py`

- [ ] **Step 1: Rewrite callback with full MC sandwich pattern**

Follow the exact `update_retire` callback pattern from `callbacks/charts.py`. The callback needs:
- 7 Outputs (figure + 6 MC stores)
- All existing cp-* Inputs
- New MC Inputs: cp-mc-enable, cp-mc-bins, cp-mc-regime, cp-mc-sims, cp-mc-years, cp-mc-window, cp-mc-start-yr, cp-mc-entry-q, cp-mc-loaded, mc-pay-trigger, cp-mc-model-src
- New MC States: btc-price-store, cp-mc-results, mc-pay-token, cp-mc-unblocked, cp-mc-rendered-key
- `_mc_setup()` → figure builder → `_mc_finalize()` sandwich

- [ ] **Step 2: Commit**

```bash
git add btc_web/callbacks/citadel_cb.py
git commit -m "feat(citadel-mc): upgrade callback to 7-output MC pattern"
```

---

### Task 5: Payment + snapshot + routing wiring

**Files:**
- Modify: `btc_web/callbacks/mc_payment.py`
- Modify: `btc_web/snapshot.py`
- Modify: `btc_web/callbacks/nav.py`

- [ ] **Step 1: Add cp to payment callback**

Add `"cp-mc-run-btn": "cp"` to `_MC_BTN_TO_TAB`. Extend `_mc_payment_initiate` callback with 5th tab inputs/outputs/states.

- [ ] **Step 2: Add MC snapshot controls**

Append 10 MC control entries to `_SNAPSHOT_CONTROLS`. Add 3 entries to `_CHECKLIST_OPTIONS` (cp-mc-enable, cp-mc-advanced, cp-mc-regime). **Must be atomic with _SNAPSHOT_CONTROLS.**

- [ ] **Step 3: Add MC IDs to _TAB_CONTROLS**

```python
_TAB_CONTROLS["citadel"].update({
    "cp-mc-enable", "cp-mc-start-yr", "cp-mc-entry-q", "cp-mc-years",
    "cp-mc-bins", "cp-mc-regime", "cp-mc-sims", "cp-mc-window",
    "cp-mc-advanced", "cp-mc-model-src",
})
```

- [ ] **Step 4: Run full test suite**

```bash
PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_citadel.py btc_web/test_web.py -q --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/callbacks/mc_payment.py btc_web/snapshot.py btc_web/callbacks/nav.py
git commit -m "feat(citadel-mc): wire payment, snapshot, and routing for MC controls"
```

---

### Task 6: Integration test and deploy

- [ ] **Step 1: Run all tests**
- [ ] **Step 2: Start dev server and verify MC controls render**
- [ ] **Step 3: Push to MCCitadelIntegration branch**
- [ ] **Step 4: Deploy branch to production**

```bash
git push origin MCCitadelIntegration
ssh root@89.167.70.45 "cd /opt/quantoshi && git fetch origin && git checkout MCCitadelIntegration && systemctl restart quantoshi"
```
