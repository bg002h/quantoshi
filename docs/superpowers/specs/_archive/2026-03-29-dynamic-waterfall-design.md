# Dynamic Cost-Ranked Spending Waterfall

**Date:** 2026-03-29
**Scope:** Replace the fixed-sequence spending waterfall with a dynamic cost-ranked waterfall that draws from the cheapest source first, re-ranking at tax bracket boundaries.
**Depends on:** Tax accounting fixes (completed — tracking helpers, merged waterfall, always lot-tracked).

---

## Problem

The current `_spending_waterfall` uses a fixed sequence: taxable cash → TD bracket-fill → taxable investments/BTC (growth-aware) → TD remaining → Roth. This ordering is suboptimal because:

1. It always sells taxable BTC before drawing from TD, even when BTC's projected growth far exceeds TD asset growth — the opportunity cost of selling BTC dwarfs the tax savings of LTCG vs ordinary rates.
2. It doesn't account for the compounding horizon of different asset classes (treasuries held to death vs BTC's diminishing power-law growth).
3. The fixed ordering doesn't adapt to the retiree's stage of life — early retirement should aggressively protect BTC, late retirement should freely sell BTC when its growth has slowed below other assets.

## Solution

Replace the fixed sequence with a **cost-per-dollar ranking** that computes the total cost (tax + opportunity cost) of withdrawing from each source and draws from the cheapest first, re-ranking at every tax bracket boundary.

## Cost Function

Each withdrawal source is scored:

```
cost(source) = tax_cost_per_dollar(source) + opportunity_cost(source)
```

Lower cost = sell first.

### Tax cost per dollar

Depends on the source type, current bracket position, and gain fraction:

| Source | Tax cost formula |
|---|---|
| Taxable cash | `0` |
| Taxable reserves (treasuries) | `0` (principal withdrawal) |
| TD assets (any) | `marginal_ordinary_rate + state_rate` |
| Taxable investments | `(base_LTCG_rate + NIIT_if_applicable + state_rate) × gain_fraction` |
| Taxable BTC | `(base_LTCG_rate + NIIT_if_applicable + state_rate) × gain_fraction` |
| TF (Roth) assets | `0` (but forced last — see Roth rule below) |

Where:
- `marginal_ordinary_rate` = the federal rate on the next dollar of ordinary income, from inflated bracket tables at the current YTD ordinary income position
- `base_LTCG_rate` = 0%, 15%, or 20% from LTCG brackets stacked above ordinary taxable income
- `NIIT_if_applicable` = 3.8% if current MAGI > NIIT threshold ($200k single, $250k MFJ). **NIIT thresholds are NOT inflation-indexed.**
- `state_rate` = flat state tax rate from `tax_data.STATE_TAX_RATES`
- `gain_fraction` = `1 - (cost_basis / current_value)` for investments (proportional basis), or lot-weighted average for BTC

### Opportunity cost per dollar

Compounded growth forgone over an asset-specific horizon:

| Asset class | Horizon | Formula |
|---|---|---|
| BTC | 10 years | `(model.price_at(q, t+10) / model.price_at(q, t)) - 1` — computed from the price model each period |
| Treasuries (reserves) | `max(min(90 - age, 40), 1)` years; age=0 if birth_year unknown. Clamped to min 1yr for ages 90+ | Taxable: `((1 + rate × (1 - marginal_ord_rate))^horizon - 1)`. TD: `((1 + rate)^horizon - 1) × (1 - marginal_ord_rate)` |
| Equities / Bonds | 15 years | Taxable: `((1 + rate)^15 - 1)`. TD: `((1 + rate)^15 - 1) × (1 - marginal_ord_rate)` |
| Cash | 15 years | Same as equities formula using cash_rate |

**Horizon rationale:**
- BTC 10 years: twice the historical 5-year break-even period. Historically, no one has lost money holding Bitcoin for 5 years; we conservatively project opportunity cost over twice this horizon.
- Treasuries: government-guaranteed returns held to remaining lifetime. Capped at 40 years.
- Equities/Bonds: 15 years (long enough to capture compounding, not so long as to dominate).

**TD opportunity cost adjustment:** TD assets grow at the gross (pre-tax) rate, but future withdrawals are taxed as ordinary income. The opportunity cost is reduced by `× (1 - marginal_ord_rate)` to reflect this. The marginal rate used is the current period's rate (a simplification — future rates may differ in a retirement drawdown).

**BTC growth from model, not assumed:** The engine calls `model.price_at(q, t)` and `model.price_at(q, t+10)` using the median selected quantile. This naturally captures the power law's diminishing returns — BTC growth is higher in early years (2031: ~660% 10yr) and lower in late years (2065: ~141% 10yr). When the model is unavailable, BTC falls back to the equity return rate.

## Roth-Last Rule

All Roth (TF) sources always rank after ALL non-Roth sources (taxable + TD), regardless of cost. Within Roth, sources are sorted by cost (Roth BTC last). This preserves tax-free compounding for as long as possible — standard financial planning advice.

**Model Info note:** "Roth (tax-free) account withdrawals are always deferred until all taxable and tax-deferred sources are exhausted. This preserves the unique benefit of tax-free compounding. Within Roth, the same cost-ranking applies."

## Source Data Structure

```python
@dataclass
class _WithdrawalSource:
    key: str              # e.g., "cash", "reserve_0", "invest_1", "btc", "td_cash", "tf_btc"
    wrapper: str          # "taxable", "td", or "tf"
    asset_type: str       # "cash", "reserve", "invest", "btc"
    index: int            # bin index (0-2 for reserves, 0-1 for investments, 0 for cash/btc)
    available: float      # current dollar balance available to draw
    growth_rate: float    # annual growth rate for opportunity cost
    horizon: int          # opportunity cost horizon in years
    gain_fraction: float  # for investments/BTC: 1 - (basis/value). 0 for cash/reserves/TD
    is_roth: bool         # True for TF sources — forced last
    is_bracket_sensitive: bool  # True if draw affects tax bracket position
    bracket_type: str     # "ordinary", "ltcg", or "none"
    cost: float = 0.0     # computed by _score_sources
```

## Known Simplifications

1. **Reserve principal withdrawal tax cost is 0.** The tax drag on interest income is captured in the after-tax opportunity cost formula, not as a direct tax cost.

2. **TD NIIT indirect cost not modeled.** TD withdrawals increase MAGI, which can push NII above the NIIT threshold. This indirect 3.8% cost is not included in TD's tax cost score. The NIIT boundary cap provides a safety net (draws are capped before crossing the threshold), but the ranking may be slightly suboptimal in the $180k-$200k MAGI zone for single filers. Accepted simplification for v1.

3. **TD opportunity cost uses current marginal rate** (not future projected rate). In a retirement drawdown, future rates may be lower. This slightly over-penalizes TD withdrawals early in retirement.

4. **NIIT marginal approximation at the cliff.** The cost function uses a binary 0%/3.8% flag. The true marginal cost of the dollar that crosses the NIIT threshold is higher (retroactive 3.8% on all accumulated NII). The boundary cap is the primary defense.

## The Re-Ranking Loop

The waterfall is a loop that draws from the cheapest source, re-ranks when bracket boundaries are crossed:

```python
while remaining > 0.01 and sources:
    _score_sources(sources, state, config, model)  # recompute costs from current state
    ranked = _rank_sources(sources)                 # non-Roth by cost, then Roth by cost

    drew_something = False
    for best in ranked:
        if best.available < 0.01:
            continue

        # Cap at next relevant boundary
        max_draw = _max_draw_before_boundary(state, config, best)
        if max_draw < 0.01:
            continue  # at boundary — skip, next source may be cheaper post-boundary

        draw = min(remaining, best.available, max_draw)
        _execute_draw(state, config, best, draw)  # updates state synchronously
        remaining -= draw
        drew_something = True
        break  # re-rank from the top with updated state

    if not drew_something:
        break  # all sources at boundaries or exhausted

    sources = [s for s in sources if s.available > 0.01]

return max(remaining, 0.0)
```

### Key properties:

1. **Re-ranks after every draw** — each withdrawal changes the bracket position, gain fractions, and available balances. The next draw uses updated costs.
2. **Skips sources at boundaries** — when the cheapest source is at a bracket boundary (`max_draw < 0.01`), tries the next source. This prevents infinite loops and handles the zero-distance edge case.
3. **`drew_something` guard** — if no source can draw anything (all at boundaries or exhausted), the loop exits with the remaining shortfall.
4. **Synchronous state updates** — `_execute_draw` calls the tracking helpers (`_sell_btc_tracked`, `_sell_investments_tracked`, etc.) which update lots, cost basis, and the accumulator immediately. The next `_score_sources` call sees the updated state.

## Bracket Boundary Tracking

`_max_draw_before_boundary(state, config, source)` returns the maximum dollars drawable before crossing a tax boundary. Three boundary types:

### 1. Ordinary income brackets (TD sources)

Distance from current YTD ordinary income to the next federal bracket threshold. Uses inflated bracket tables. Ordinary income includes: TD withdrawals + interest + treasury interest + other income.

```python
_already_ordinary = (accum.tax_deferred_withdrawals + accum.interest_income
                     + accum.treasury_interest + accum.other_income)
next_bracket = first bracket threshold > _already_ordinary
distance = next_bracket - _already_ordinary
return max(distance / ppy, 0)  # pro-rate to period if needed? No — the accumulator tracks annual.
```

Actually, the accumulator tracks YTD, not pro-rated. The distance is in annual dollars.

### 2. LTCG brackets (taxable BTC, taxable investments)

LTCG brackets **stack on top of ordinary taxable income** (IRS §1(h) stacking rule). The effective LTCG bracket starts where ordinary income ends.

```python
ordinary_taxable = max(ordinary_gross - standard_deduction, 0)
stacked_base = ordinary_taxable
current_ltcg_position = stacked_base + accum.lt_capital_gains
next_ltcg_threshold = first LTCG bracket threshold > current_ltcg_position
distance = next_ltcg_threshold - current_ltcg_position
```

**Critical coupling:** TD withdrawals increase `ordinary_taxable`, which shifts `stacked_base` upward, pushing LTCG into higher brackets. After every TD draw, LTCG bracket distances must be recomputed. The re-ranking loop handles this naturally since `_score_sources` recomputes everything from the updated accumulator.

### 3. NIIT threshold ($200k single / $250k MFJ — NOT inflation-indexed)

NIIT is a cliff, not a bracket. When MAGI crosses the threshold, 3.8% applies to the lesser of (NII, MAGI - threshold). This is a discontinuity.

```python
magi = agi_from_accumulator  # same as AGI in v1
niit_threshold = NIIT_THRESHOLD[filing_status]  # raw, not inflated
distance = max(niit_threshold - magi, 0)
```

When MAGI is within `remaining` of the threshold, any draw that pushes MAGI over triggers NIIT on accumulated NII. `_max_draw_before_boundary` returns the distance to the NIIT threshold as a candidate cap (the min of all applicable boundaries).

**What pushes MAGI up:** TD withdrawals (ordinary income), capital gains (from BTC/investment sales), interest income. Cash/reserve principal withdrawals do NOT increase MAGI.

## Source List

~14 entries enumerated from state:

| # | Source | Wrapper | Bracket-sensitive | Growth rate |
|---|---|---|---|---|
| 1 | Cash | Taxable | No | `config.cash_rate / 100` |
| 2 | Reserve Short | Taxable | No | `config.reserve_bins[0]["rate"] / 100` |
| 3 | Reserve Medium | Taxable | No | `config.reserve_bins[1]["rate"] / 100` |
| 4 | Reserve Long | Taxable | No | `config.reserve_bins[2]["rate"] / 100` |
| 5 | Equities | Taxable | LTCG + NIIT | `config.invest_bins[0]["return_rate"] / 100` |
| 6 | Bonds | Taxable | LTCG + NIIT | `config.invest_bins[1]["return_rate"] / 100` |
| 7 | BTC | Taxable | LTCG + NIIT | Model-computed 10yr |
| 8 | TD Cash | TD | Ordinary + NIIT (indirect) | `config.cash_rate / 100` |
| 9 | TD Reserves | TD | Ordinary + NIIT (indirect) | Weighted avg of reserve rates |
| 10 | TD Investments | TD | Ordinary + NIIT (indirect) | Weighted avg of invest rates |
| 11 | TD BTC | TD | Ordinary + NIIT (indirect) | Model-computed 10yr |
| 12 | TF Cash + Reserves | TF (Roth) | No | — |
| 13 | TF Investments | TF (Roth) | No | — |
| 14 | TF BTC | TF (Roth) | No | — |

Roth sources are scored but always rank after all non-Roth. Within Roth, cost-ranked (TF BTC last due to highest opportunity cost).

## Non-Tax Mode

When `config.tax_enabled = False`: TD/TF sources don't exist (all zeros). The ranking degenerates to taxable sources only, scored by opportunity cost alone (all tax costs = 0). Cash ranks first (lowest growth), BTC ranks last (highest growth). This matches the current non-tax behavior but with growth-aware ordering.

## `_execute_draw` Dispatch

```python
def _execute_draw(state, config, source, amount):
    if source.type == "cash":        state.cash -= amount
    elif source.type == "reserve":   state.reserves[source.index] -= amount
    elif source.type == "invest":    _sell_investments_tracked(state, config, source.index, amount)
    elif source.type == "btc":       _sell_btc_tracked(state, config, amount / state.btc_price)
    elif source.type == "td_cash":   state.td_cash -= amount; record_ordinary(amount)
    elif source.type == "td_res":    state.td_reserves[i] -= amount; record_ordinary(amount)
    elif source.type == "td_inv":    state.td_investments[i] -= amount; record_ordinary(amount)
    elif source.type == "td_btc":    state.td_btc_stack -= amount/price; record_ordinary(amount)
    elif source.type == "tf_*":      similar, record_roth(amount)
```

All draws use the existing tracking helpers where applicable. State is updated synchronously.

## Model Info Tab Notes

Add three notes:

1. **Opportunity cost horizons:** "The Citadel Planner computes withdrawal cost as immediate tax plus forgone compounding. Bitcoin uses a 10-year horizon (twice the historical 5-year break-even). Equities and bonds use 15 years. Treasuries use the holder's remaining lifetime (capped at 40 years). These horizons determine how aggressively each asset is protected from withdrawal."

2. **Roth-last policy:** "Roth (tax-free) account withdrawals are always deferred until all taxable and tax-deferred sources are exhausted, preserving the benefit of tax-free compounding."

3. **Investment gains classification:** (Already added in prior spec.) "Investment gains in the Citadel Planner are classified as long-term capital gains. Individual equity and bond lot tracking is not modeled."

## Performance

Per period: ~14 sources × ~7 bracket boundaries = ~98 iterations worst case. Each iteration is arithmetic (no I/O, no complex data structures). Over 528 periods (44yr monthly): ~52k iterations total. Well under 1ms per period. No performance concern.

## Files Touched

| File | Change |
|---|---|
| `engines/citadel.py` | Replace `_spending_waterfall` with cost-ranked loop, add `_score_sources`, `_rank_sources`, `_max_draw_before_boundary`, `_execute_draw`, `_build_source_list` |
| `engines/tax_data.py` | No changes (brackets, NIIT thresholds already present) |
| `engines/tax.py` | No changes |
| `layout/model_info.py` | Add 2 new notes (horizons, Roth-last) |
| `test_web.py` | Tests for cost function, ranking, bracket transitions, BTC protection in early/late retirement, Roth-last, non-tax mode, high-spender bracket crossing, NIIT cliff |

## What This Does NOT Change

- Tax computation pipeline (`compute_annual_tax` unchanged)
- Quarterly estimated payments (unchanged)
- Floor enforcement (unchanged — runs after waterfall)
- Rebalancing triggers (unchanged)
- `_pay_tax_amount` (unchanged — has its own fixed sourcing)
- Any UI controls or layout (except Model Info notes)

## Tests

1. Cost function: taxable cash scores 0 tax + low opportunity
2. Cost function: BTC scores high opportunity in 2035, lower in 2065
3. Cost function: TD scores marginal ordinary rate
4. Cost function: NIIT adds 3.8% above threshold
5. Ranking: in 2035, BTC ranks last among non-Roth (highest cost)
6. Ranking: in 2065, BTC moves to mid-pack (growth slowed)
7. Ranking: Roth always after non-Roth regardless of cost
8. Bracket transition: TD draw shifts LTCG stack base
9. Bracket transition: $500k monthly spend crosses multiple brackets, ranking changes
10. NIIT cliff: draw capped at NIIT threshold boundary
11. Zero bracket distance: loop skips source, doesn't hang
12. All sources exhausted: returns correct shortfall
13. Non-tax mode: ranks by opportunity cost only, no crash
14. Model failure: fallback to equity rate for BTC opportunity cost
15. Gain fraction updates: partial BTC sale changes gain fraction for next iteration
16. Treasury horizon: uses remaining lifetime (max(min(90-age, 40), 1))
17. Late retirement crossover: BTC becomes cheaper to sell than treasuries
18. Treasury horizon age 92: horizon clamps to 1 (not negative)
19. Negative BTC model growth: BTC ranks first/cheapest when model projects price decline
20. Reserve tax cost note: principal withdrawal = 0 tax, interest drag in opportunity cost
