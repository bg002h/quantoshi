# Unified Multi-Asset Citadel MC — Design Spec

## Goal

Transform the Citadel Planner from a deterministic single-path simulator with a bolted-on BTC-only MC overlay into a unified multi-asset Monte Carlo engine where all asset classes (BTC, equities, bonds, treasuries, cash) vary stochastically across simulation paths.

## Background

**Current architecture:**
- Free "Run Simulation" → deterministic, n_sims=1. BTC follows a QR quantile path. Dollar assets use user-input fixed rates. Volatility inputs are silently ignored.
- Paid "Run MC Simulation" → pre-computed BTC-only Markov paths rendered as overlay. Dollar assets use fixed rates or Markov regimes, but only when `is_deterministic=False` (MC mode). The MC overlay is a separate layer, not integrated with the engine.
- The `is_deterministic` guard in `citadel_step.py` prevents Markov dollar asset returns from running in deterministic mode.
- Dollar asset Markov starting regimes are hardcoded to bin 2 (neutral) in `CitadelState`.
- BTC stochastic price paths are generated externally by the Cython `markov` module (in `mc_overlay.py`, gated by `_HAS_MARKOV`) and passed into `simulate()` as `price_paths` array.

**Problems:**
- Dollar asset volatility inputs do nothing in free mode.
- "Historical Regimes" dropdown has no effect in deterministic mode.
- MC overlay is BTC-only — portfolio-level spread isn't captured.
- No way to see how dollar asset variation affects retirement outcomes.

## Architecture

### Simulation Modes

1. **Deterministic** (free, "Run Simulation" button) — Engine runs once (n_sims=1). BTC follows QR quantile path. Dollar assets use user-input fixed rates, zero volatility. Unchanged from today.

2. **Quick Scenarios** (free, cached presets) — Pre-computed percentile bands loaded from shared memory. Displayed as band overlay on the deterministic chart. No engine run needed. Instant load. Bands computed from 800 simulation paths.

3. **Custom MC** (paid, "Run MC Simulation" button) — Engine runs N times with user's full configuration. Each path: Markov BTC + Markov dollar assets + full waterfall/tax/rebalancing. Returns percentile bands. Full sim download available for paid users only; free cached scenarios are not available for download.

### BTC Price Path Source

BTC stochastic price paths remain an external input to `simulate()` — they are not generated inside the engine.

- **Quick Scenarios (free):** BTC paths are pre-baked into the cached bands during cache generation. Cache generation runs on a machine with the Markov module installed. Free users never need the Markov module at runtime — they load pre-computed bands.
- **Custom MC (paid):** BTC paths generated at runtime by the existing Cython Markov module (`_HAS_MARKOV` required on prod). The engine receives BTC price paths externally (same as today), plus runs Markov dollar asset returns internally.

### Engine Changes

**Replace** (not remove) the `is_deterministic` guard in `citadel_step.py` (line 141-143). New logic:

```python
use_markov = (config.asset_return_model == "markov"
              and config.asset_matrices is not None
              and config.n_sims > 1)
```

This preserves deterministic behavior exactly as-is. Markov fires only during MC runs (n_sims > 1). Phase 1 engine changes are not reachable from the UI until Phase 3 wires the new flow.

**Required API change:** Add starting regime fields to `SimConfig`:

```python
initial_equity_regime: int = 2
initial_bond_regime: int = 2
initial_res_short_regime: int = 2
initial_res_med_regime: int = 2
initial_res_long_regime: int = 2
```

`_initial_state()` reads these from config instead of using hardcoded bin 2. The macro_regime preset (Bear=0, Neutral=2, Bull=4) sets all five to the same bin. Custom MC can set them individually.

**TD/TF wrappers:** Tax-deferred and tax-free wrappers get Markov returns same as taxable wrapper when `use_markov` is True. All wrappers use consistent return models to avoid unrealistic divergence between account types holding the same asset class.

### Sim Output & Band Aggregation

After running N simulation paths, aggregate to percentile bands (P5/P10/P25/P50/P75/P90/P95) across paths for the following 11 output series:

**Taxable wrapper:**
1. Total portfolio (USD)
2. BTC stack (BTC, native units)
3. BTC holdings (USD)
4. Cash (USD)
5. Reserves total (USD)
6. Investments total (USD)

**Tax-advantaged wrappers:**
7. Tax-deferred total (USD)
8. Tax-free total (USD)

**Cross-cutting:**
9. Cumulative spending (USD)
10. Cumulative taxes paid (USD)
11. Depletion flag (boolean per step)

Aggregation happens post-simulation, before caching or charting. All sims run 40 years from start_year at Monthly frequency (480 time steps). UI truncates to 10/20/30/40 year views by slicing the stored bands.

## Cache Architecture

### Cache Key

`(btc_model, btc_entry_q, macro_regime, wealth_level, rule_set, start_year, tax_status)`

### Dimensions

| Dimension | Values | Count |
|-----------|--------|-------|
| btc_model | bub, qr, pl, lppl, ef | 5 |
| btc_entry_q | 1%, 10%, 50% | 3 |
| macro_regime | Bear, Neutral, Bull | 3 |
| wealth_level | Starter, Full, Bitcoin | 3 |
| rule_set | No Rebal, Cautious, Aggressive | 3 |
| start_year | 2028, 2035 | 2 |
| tax_status | Single, MFJ | 2 |

**Total: 5 × 3 × 3 × 3 × 3 × 2 × 2 = 1,620 cached scenarios.** Each computed from 800 simulation paths.

### Storage

Each scenario stores 7 percentiles × 480 time steps × 11 output series = ~37k floats ≈ 290KB per entry. Total cache: 1,620 × 290KB ≈ **470MB**. Fits in shared memory alongside the existing 834MB BTC-only MC cache.

Cached scenarios always use Monthly frequency. Other frequencies available only for paid custom runs.

Bands only — no full per-path sim data in cache. Deterministic run provides full single-path detail for free users. Free cached scenarios are not available for download.

### Cache Generation

Generated locally on a multi-core machine (~20 Intel 13700 cores). Estimated time: 1,620 scenarios × 800 paths × ~80ms per path ≈ 28.8 hours single-core, ~1.5 hours parallelized across 20 cores.

### Pricing Tags

Each cache entry tagged as `free`, `discounted`, or untagged. Free entries served instantly. Discounted entries require half-price payment. Discounted tier tag assignments deferred to post-Phase 3. Phase 4 implementation will define which cache entries are free vs discounted based on simulation review.

## Preset Configuration

All preset values live in a single module: `btc_web/citadel_presets.py`. Designed for easy editing without touching logic.

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

### TD/TF Default Balances

Default to zero for all wealth levels. Adjustable per level when needed.

### Tax Configuration (Cached Presets)

- Filing status: Single or MFJ (cached dimension)
- State tax: None (no state tax in cached scenarios; state tax available for paid custom MC)
- TCJA sunset: default (current law)

### Macro Regimes

| Regime | Label | Starting Bin |
|--------|-------|-------------|
| bear | Bear | 0 |
| neutral | Neutral | 2 |
| bull | Bull | 4 |

Regime definitions (returns/vol/transition matrix implications for each dollar asset class) TBD — to be determined when simulation results are reviewed. Starting bin sets all 5 dollar asset regime fields to the same value.

### Rule Sets

| Rule Set | Label | Cash Floor | Rebal Trigger |
|----------|-------|-----------|---------------|
| no_rebal | No Rebal | TBD | None |
| cautious | Cautious | TBD | TBD |
| aggressive | Aggressive | TBD | TBD |

Cash floor levels and rebalancing triggers TBD based on simulation results.

### Other Preset Constants

```
BTC_MODELS = ["bub", "qr", "pl", "lppl", "ef"]
START_YEARS = [2028, 2035]
BTC_ENTRY_QS = [1, 10, 50]   # percentile
SIMS_PER_SCENARIO = 800
```

## UI Design

### Quick Scenarios Panel

Positioned at the top of the Citadel controls column, above the existing sub-tabs. Three rows of pill buttons plus a start year dropdown:

```
Quick Scenarios (Free)
─────────────────────
Wealth:  [Starter] [Full] [Bitcoin]
Regime:  [Bear] [Neutral] [Bull]
Rules:   [No Rebal] [Cautious] [Aggressive]
Start:   [2035 ▾]  (cached years bolded)
```

Default start year: 2035. Dropdown shows a range of years; cached entries (2028, 2035) are bolded (same pattern as `cp-mc-entry-q`).

Subtle label below pills: "Customize below (MC requires payment)".

Note below bands: "800 simulations" to communicate statistical basis.

### Preset Auto-Fill Behavior

When a user selects a Quick Scenario preset:
1. Compare each preset-defined control value against the current control value.
2. If the current value equals the default OR equals the preset value → fill silently.
3. If the current value differs from both the default and the preset value → show confirmation dialog listing the controls that will change (e.g., "Spending: $10k → $25k, BTC Stack: 1.0 → 2.5") with Apply / Cancel.
4. On Apply → fill controls, load cached bands.
5. On Cancel → nothing changes.

After preset load, if the user modifies a preset-defined control, show a "stale" indicator on the bands (same pattern as existing MC match/stale indicator) noting that the displayed bands reflect the preset, not the current configuration.

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
| Free | Quick Scenarios (cached presets) | $0 | Bands display only, no download |
| Discounted | Extended cached presets (TBD) | 0.5× base | Bands display only, no download |
| Custom | User-configured, on demand | 1× base | Bands + full sim download |

Base price scaled linearly from other tabs' MC pricing, accounting for higher compute cost (~800 paths × full engine).

Full sim download format and serving mechanism (background task + file generation) to be specified in Phase 4 design.

### Dev Bypass

When `DEV=1`, all MC runs are free. Payment check returns `True` unconditionally. Applies to both cached lookup and on-demand computation.

## Known Limitations

- **Cross-asset correlation not modeled.** Dollar asset Markov chains are independent. Equity can be in a bull regime while bonds are in a bear regime at the same step. Historical data shows these correlations are real and time-varying (especially post-2022). This is a simplifying assumption for v1; correlated regime models may be explored in future versions.
- **LPPL model has hardcoded parameters.** The LPPL fit was done once via differential evolution. Only residual sigma recomputes from new data. The curve shape will drift over time. Converting LPPL to the model_toolkit build pipeline is tracked in the backlog (#23).

## Phasing

### Phase 1: Engine — Unified Multi-Asset MC
- Replace `is_deterministic` guard with `n_sims > 1` check
- Add starting regime fields to `SimConfig`, wire through `_initial_state()`
- Apply Markov returns to TD/TF wrappers (same as taxable)
- Engine runs N sims with stochastic returns on all assets
- Percentile band aggregation from sim results (11 series)
- Dev bypass for payment
- Phase 1 engine changes are not reachable from the UI until Phase 3 wires the new flow
- Tests: verify bands match expected distributions, deterministic mode unchanged, TD/TF wrappers get Markov returns

### Phase 2: Presets & Cache
- `citadel_presets.py` module with all preset definitions (easy to edit)
- Cache generation script for multi-asset scenarios (runs locally, ~20 cores)
- Shared memory storage/loading for band data
- 1,620 cached scenarios (~470MB)
- Tests: cache key lookup, band data integrity, correct series count

### Phase 3: UI — Quick Scenarios Panel + Band Rendering
- Pill button selectors (Wealth, Regime, Rules) + start year dropdown (cached years bolded)
- Preset auto-fill with confirmation dialog when overwriting user customizations
- Stale indicator when user modifies preset-defined controls after loading bands
- Cached band loading on preset selection
- Band rendering in figure builder (opacity fills, grouped legend toggle)
- BTC stack display mode with bands
- "800 simulations" label on Quick Scenarios panel

### Phase 4: Payment Integration
- Pricing tiers (free/discounted/custom)
- BTCPay flow for custom MC runs
- Discounted tier for extended cached entries (tag assignments post-Phase 3 review)
- Full sim download for paid users (format and serving mechanism TBD)
- Free cached scenarios not available for download

Each phase deployable independently. Phase 1 is invisible to users (engine-only, not reachable from UI). Phase 2 adds cache infrastructure. Phase 3 is the user-facing feature. Phase 4 monetizes custom runs.
