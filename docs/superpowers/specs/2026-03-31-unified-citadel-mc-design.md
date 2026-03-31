# Unified Multi-Asset Citadel MC — Design Spec

## Goal

Transform the Citadel Planner from a deterministic single-path simulator with a bolted-on BTC-only MC overlay into a unified multi-asset Monte Carlo engine where all asset classes (BTC, equities, bonds, treasuries, cash) vary stochastically across simulation paths.

## Background

**Current architecture:**
- Free "Run Simulation" → deterministic, n_sims=1. BTC follows a QR quantile path. Dollar assets use user-input fixed rates. Volatility inputs are silently ignored.
- Paid "Run MC Simulation" → pre-computed BTC-only Markov paths rendered as overlay. Dollar assets use fixed rates or Markov regimes, but only when `is_deterministic=False` (MC mode). The MC overlay is a separate layer, not integrated with the engine.
- The `is_deterministic` guard in `citadel_step.py` prevents Markov dollar asset returns from running in deterministic mode.
- Dollar asset Markov starting regimes are hardcoded to bin 2 (neutral) in `CitadelState`.

**Problems:**
- Dollar asset volatility inputs do nothing in free mode.
- "Historical Regimes" dropdown has no effect in deterministic mode.
- MC overlay is BTC-only — portfolio-level spread isn't captured.
- No way to see how dollar asset variation affects retirement outcomes.

## Architecture

### Simulation Modes

1. **Deterministic** (free, "Run Simulation" button) — Engine runs once (n_sims=1). BTC follows QR quantile path. Dollar assets use user-input fixed rates, zero volatility. Unchanged from today.

2. **Quick Scenarios** (free, cached presets) — Pre-computed percentile bands loaded from shared memory. Displayed as band overlay on the deterministic chart. No engine run needed. Instant load.

3. **Custom MC** (paid, "Run MC Simulation" button) — Engine runs N times with user's full configuration. Each path: Markov BTC + Markov dollar assets + full waterfall/tax/rebalancing. Returns percentile bands + optional full sim download.

### Engine Changes

Remove the `and not is_deterministic` guard from `citadel_step.py` (line 141-143). The Markov vs lognormal decision is based purely on `config.asset_return_model` and `n_sims`. When n_sims > 1, each path gets independent Markov draws for all assets. When n_sims=1, fixed rates are used (deterministic mode unchanged).

Starting regimes in `CitadelState` (`equity_regime`, `bond_regime`, `res_short_regime`, `res_med_regime`, `res_long_regime`) are initialized from the macro_regime preset (Bear=bin 0, Neutral=bin 2, Bull=bin 4) for cached scenarios. For custom MC runs, users can set them individually or via macro preset.

### Sim Output & Band Aggregation

After running N simulation paths, aggregate to percentile bands (P5/P10/P25/P50/P75/P90/P95) across paths for each output series:

- Total portfolio (USD)
- BTC stack (BTC, native units)
- BTC holdings (USD)
- Cash (USD)
- Reserves (USD)
- Investments (USD)
- Cumulative spending (USD)
- Depletion flag per step

Aggregation happens post-simulation, before caching or charting. All sims run 40 years from start_year; UI truncates to 10/20/30/40 year views by slicing the stored bands.

## Cache Architecture

### Cache Key

`(btc_entry_q, macro_regime, wealth_level, rule_set, start_year)`

### Dimensions

| Dimension | Values | Count |
|-----------|--------|-------|
| btc_entry_q | 1%, 10%, 50% | 3 |
| macro_regime | Bear, Neutral, Bull | 3 |
| wealth_level | Starter, Full, Bitcoin | 3 |
| rule_set | No Rebal, Cautious, Aggressive | 3 |
| start_year | 2028, 2031, 2035 | 3 |

**Total: 3 × 3 × 3 × 3 × 3 = 243 cached scenarios.**

### Storage

Each scenario stores 7 percentiles × 480 time steps × ~8 output series = ~27k floats ≈ 210KB per entry. Total cache: ~50MB. Fits comfortably in shared memory (`/dev/shm`).

Bands only — no full per-path sim data in cache. Deterministic run provides full single-path detail for free users.

### Pricing Tags

Each cache entry tagged as `free`, `discounted`, or untagged. Free entries served instantly. Discounted entries require half-price payment. Specific tag assignments TBD based on simulation results.

## Preset Configuration

All preset values live in a single module: `btc_web/citadel_presets.py`. Easy to edit without touching logic.

### Wealth Levels

| Level | Label | Dollar Assets | BTC | Spending | Spend Growth | Inflation |
|-------|-------|-------------|-----|----------|-------------|-----------|
| starter | Starter Citadel | $500k | 0.5 BTC | $5k/mo | 1%/yr | 4%/yr |
| full | Full Citadel | $2.5M | 2.5 BTC | $25k/mo | 2%/yr | 4%/yr |
| bitcoin | Bitcoin Citadel | $2.5M | 12.5 BTC | $50k/mo | 4%/yr | 4%/yr |

### Dollar Asset Allocation

Same default percentages across all wealth levels, separately editable per level:

| Asset | Default % |
|-------|----------|
| Cash | 10% |
| Reserves — Short (T-bills) | 10% |
| Reserves — Medium (T-notes) | 10% |
| Reserves — Long (T-bonds) | 10% |
| Investments — Equities | 40% |
| Investments — Bonds | 20% |

### Macro Regimes

| Regime | Label | Starting Bin |
|--------|-------|-------------|
| bear | Bear | 0 |
| neutral | Neutral | 2 |
| bull | Bull | 4 |

Regime definitions (what each bin means in terms of returns/vol/transition matrix for each dollar asset class) are TBD — to be determined when simulation results are reviewed.

### Rule Sets

| Rule Set | Label | Cash Floor | Rebal Trigger |
|----------|-------|-----------|---------------|
| no_rebal | No Rebal | TBD | None |
| cautious | Cautious | TBD | TBD |
| aggressive | Aggressive | TBD | TBD |

Cash floor levels and rebalancing triggers TBD based on simulation results.

### Other Preset Constants

```
START_YEARS = [2028, 2031, 2035]
BTC_ENTRY_QS = [1, 10, 50]   # percentile
```

## UI Design

### Quick Scenarios Panel

Positioned at the top of the Citadel controls column, above the existing sub-tabs. Three rows of pill buttons:

```
Quick Scenarios (Free)
─────────────────────
Wealth:  [Starter] [Full] [Bitcoin]
Regime:  [Bear] [Neutral] [Bull]
Rules:   [No Rebal] [Cautious] [Aggressive]
```

Selecting any combination loads cached bands instantly. Existing controls below auto-fill to match the preset. Subtle label: "Customize below (MC requires payment)" bridges to the full controls.

### Chart: Band Rendering

Bands are the hero visual — the primary way users understand portfolio uncertainty.

**In "USD (per asset class)" mode with MC enabled:**
- Solid lines — deterministic path per asset (BTC, cash, reserves, investments, total)
- Shaded bands — P25–P75 (dark fill, opacity 0.30) and P5–P95 (light fill, opacity 0.15) around each asset's solid line
- Total portfolio band is most prominent (widest line, front layer)
- Individual asset bands are softer/thinner, behind the total
- Same color as their deterministic trace but at lower opacity — naturally groups bands with traces

**In "BTC Holdings" mode with MC bands:**
- Solid line = deterministic BTC stack (native units)
- Bands show how stochastic price paths cause different sell/hold decisions → different remaining stacks

**Legend:**
- Deterministic traces get normal legend entries
- Bands get a single grouped entry: "MC spread (P5–P95)" with a gradient swatch, toggleable to hide all bands at once
- No per-percentile legend clutter

**Two distinct toggleable trace groups:**
1. Deterministic — the single QR quantile path (always visible by default)
2. MC Bands — percentile fans from the unified simulation (toggled on/off)

### Colorblind Safety

Uses the existing three-tier palette system (`_get_palette`). Band opacity approach inherently works for colorblind users — bands share their trace's color at reduced opacity.

## Pricing & Payment

### Three Tiers

| Tier | What | Price | Data |
|------|------|-------|------|
| Free | Quick Scenarios (cached presets) | $0 | Bands only |
| Discounted | Extended cached presets (TBD) | 0.5× base | Bands only |
| Custom | User-configured, on demand | 1× base | Bands + full sim download |

Base price scaled linearly from other tabs' MC pricing, accounting for higher compute cost (~800 paths × full engine).

### Dev Bypass

When `DEV=1`, all MC runs are free. Payment check returns `True` unconditionally. Applies to both cached lookup and on-demand computation.

## Phasing

### Phase 1: Engine — Unified Multi-Asset MC
- Remove `is_deterministic` guard on Markov dollar assets
- Engine runs N sims with stochastic returns on all assets
- Percentile band aggregation from sim results
- Dev bypass for payment
- Tests: verify bands match expected distributions, deterministic mode unchanged

### Phase 2: Presets & Cache
- `citadel_presets.py` module with all preset definitions
- Cache generation script for multi-asset scenarios
- Shared memory storage/loading for band data
- 243 cached scenarios
- Tests: cache key lookup, band data integrity

### Phase 3: UI — Quick Scenarios Panel + Band Rendering
- Pill button selectors (Wealth, Regime, Rules)
- Cached band loading on preset selection
- Band rendering in figure builder (opacity fills, grouped legend toggle)
- BTC stack display mode with bands
- Auto-fill existing controls from preset values

### Phase 4: Payment Integration
- Pricing tiers (free/discounted/custom)
- BTCPay flow for custom MC runs
- Discounted tier for extended cached entries
- Full sim download for paid users

Each phase deployable independently. Phase 1 is invisible to users (engine-only). Phase 2 adds cache infrastructure. Phase 3 is the user-facing feature. Phase 4 monetizes custom runs.
