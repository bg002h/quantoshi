# Tax Accounting Fixes

**Date:** 2026-03-29
**Scope:** Fix 7 tax accounting gaps, merge waterfalls, always lot-track BTC.
**Prerequisite for:** Dynamic waterfall ordering (separate spec).

---

## Problem

Seven code paths sell/buy BTC or investments without proper tax accounting:

1. **Floor enforcement BTC sales** (`_enforce_floors`, line ~332) — raw `btc_stack -= btc_sold`, no `sell_lots()`, no capital gains recorded.
2. **Rebalancing BTC sales** (`_execute_sell_btc`, line ~429) — raw stack reduction, no lot tracking, no gain recorded.
3. **Rebalancing BTC purchases** (`_execute_buy_btc`, line ~450) — no `TaxLot` created, so future sales of this BTC have no cost basis.
4. **Investment sales during tax payment** (`_pay_tax_amount`, lines ~1041-1058) — LTCG computed for gross-up but never recorded in `TaxYearAccumulator`. Gains are invisible to subsequent tax computation.
5. **SCF initial BTC purchase** (`_initial_state`, lines ~619-626) — Saylor Fortifier buys BTC at startup but `seed_lots()` only covers `start_stack`. The SCF BTC has no lot, so future sales undercount.
6. **SCF perpetual loan repayment BTC sale** (`_scf_check_repay`, lines ~589-592) — raw `btc_stack -= btc_sold`, same bug class as #1 and #2.
7. **Floor enforcement investment sales** (`_enforce_floors`, lines ~317-318) — `investments[i] -= draw` without adjusting `invest_cost_basis`. Cost basis drifts from actual value.

Additionally, two separate waterfall functions (`_apply_spending_waterfall` and `_tax_aware_waterfall`) duplicate logic and will diverge as the dynamic waterfall is added next. They should be merged.

## Design Decisions

- **Always lot-track BTC** even when `tax_enabled=False`. One code path, forward-compatible with future tax model changes. Lots are metadata — they don't trigger tax computation.
- **Investment gains always classified as long-term.** In a retirement planner spanning decades, positions are virtually always held >1 year. Note added to Model Info tab.
- **Merge the two waterfalls** into one unified function. Non-tax mode skips TD/TF steps and accumulator writes, but still uses growth-aware ordering and lot tracking.

## New State Field

Add `sim_date: str = ""` to `CitadelState`. Set once per period in `step()` before any function that needs it. Eliminates threading `sim_date` through function parameters.

## Always Seed Lots

`_initial_state` moves the `seed_lots()` call outside the `if config.tax_enabled:` block. Lots are always created for the starting BTC stack. If SCF is enabled, a second lot is created for the SCF BTC purchase (source="scf") immediately after the SCF stack addition. The `TaxYearAccumulator` is still only created when `tax_enabled=True`.

## Three Tracking Helpers

Placed near the top of `citadel.py`, before functions that use them.

### `_sell_btc_tracked(state, config, btc_to_sell)`

- If `state.tax_lots` is non-empty: calls `sell_lots(state.tax_lots, btc_to_sell, state.btc_price, state.sim_date, method=config.cost_basis_method)`. Updates `state.btc_stack -= result.btc_sold` and `state.tax_lots = result.remaining_lots`. If `state.tax_year_accum is not None`: records each gain as ST or LT in the accumulator.
- If `state.tax_lots` is empty (defensive): raw `state.btc_stack -= min(btc_to_sell, state.btc_stack)`.
- Returns the `SaleResult` (or a minimal equivalent for the empty-lots path) so callers can access `btc_sold`. **Callers that need a dollar amount** (e.g., `_enforce_floors` for deficit reduction) must use `result.btc_sold * state.btc_price`, not the requested amount, since `sell_lots()` may sell fewer BTC than requested if lots are exhausted.

### `_buy_btc_tracked(state, config, btc_bought, source="rebal_buy")`

- Creates `TaxLot(date=state.sim_date, btc=btc_bought, cost_basis=state.btc_price, source=source)`.
- Appends to `state.tax_lots`.
- Updates `state.btc_stack += btc_bought`.

### `_sell_investments_tracked(state, config, bin_index, amount)`

- Computes proportional cost basis: `fraction = amount / current`, `basis_sold = invest_cost_basis[i] * fraction`.
- Updates `state.investments[i] -= amount`, `state.invest_cost_basis[i] -= basis_sold`.
- Computes `gain = amount - basis_sold`.
- If `state.tax_year_accum is not None`: records gain as `lt_capital_gains` or `lt_capital_losses`.
- Returns `(amount_drawn, gain)`.

## No Taxes When Toggle Is Off

Guaranteed by three independent guards:

1. `state.tax_year_accum` is `None` when `tax_enabled=False` → helpers skip all accumulator writes.
2. `_quarterly_estimated_payment` and `_year_boundary_tax` only execute when `config.tax_enabled=True`.
3. `taxes_paid` result array is `None` → UI shows nothing.

Lots are maintained regardless (metadata only), but no tax events are recorded or computed.

## Call Sites Updated

| Function | Current behavior | New behavior |
|---|---|---|
| `_enforce_floors` (BTC sale for cash floor) | `state.btc_stack -= btc_sold` | `_sell_btc_tracked(state, config, btc_sold)`; use `result.btc_sold * price` for deficit |
| `_enforce_floors` (investment sales for floor) | `state.investments[i] -= draw` | `_sell_investments_tracked(state, config, i, draw)` |
| `_execute_sell_btc` (rebalancing) | `state.btc_stack -= btc_to_sell` | `_sell_btc_tracked(state, config, btc_to_sell)` |
| `_execute_buy_btc` (rebalancing) | `state.btc_stack += btc_bought` | `_buy_btc_tracked(state, config, btc_bought, source="rebal_buy")` |
| `_scf_check_repay` (SCF loan repayment) | `state.btc_stack -= btc_sold` | `_sell_btc_tracked(state, config, btc_sold)` |
| `_initial_state` (SCF purchase) | `state.btc_stack += btc_bought`, no lot | `_buy_btc_tracked(state, config, btc_bought, source="scf")` (or inline lot creation since `sim_date` may not be set yet) |
| `_pay_tax_amount` (investment sales) | Computes gain for gross-up, not recorded | Calls `_sell_investments_tracked`, gain auto-recorded |
| `_tax_aware_waterfall` (investment sales) | Inline basis tracking + accumulator | Calls `_sell_investments_tracked` |
| `_tax_aware_waterfall` (BTC sales) | Calls `sell_lots` inline | Calls `_sell_btc_tracked` |
| `_apply_spending_waterfall` (BTC sales) | `state.btc_stack -= btc_sold` | **Function removed** — merged into unified waterfall |

## Waterfall Merge

`_apply_spending_waterfall` is deleted. `_tax_aware_waterfall` becomes the sole waterfall, renamed to `_spending_waterfall`.

When `config.tax_enabled=False`:
- TD/TF steps are skipped (no TD bracket-fill, no TD remaining, no TF draw)
- Accumulator updates still flow through helpers (they no-op when `tax_year_accum is None`)
- Growth-aware BTC vs investments ordering still applies (benefits non-tax mode too)
- `model` parameter stays optional (falls back to equity rate when `None`)

`step()` changes from:
```python
if config.tax_enabled:
    shortfall = _tax_aware_waterfall(new, config, period_spend, sim_date, model=model)
else:
    shortfall = _apply_spending_waterfall(new, period_spend)
```
To:
```python
shortfall = _spending_waterfall(new, config, period_spend, model=model)
```

The `sim_date` parameter is dropped — the function reads `state.sim_date` directly.

## `_execute_sell_btc` / `_execute_buy_btc` Parameter Changes

Both gain `config` as a required parameter. `_execute_sell_btc` currently takes `(state, rate_pct, split)` — changes to `(state, config, rate_pct, split)`. `_execute_buy_btc` already takes `config` optionally — becomes required.

All callers (`_evaluate_rebalancing`) updated to pass `config`.

## Model Info Tab Note

Add to the Model Info accordion: "Investment gains in the Citadel Planner are classified as long-term capital gains. Individual equity and bond lot tracking is not modeled."

## Files Touched

| File | Change |
|---|---|
| `engines/citadel.py` | Add `sim_date` to state, add 3 helpers, update 10 call sites, merge waterfalls, always seed lots (incl. SCF), update `_execute_sell/buy_btc` and `_scf_check_repay` signatures |
| `engines/tax_lots.py` | No changes |
| `layout/model_info.py` | Add investment gains classification note |
| `test_web.py` | Tests for each helper, regression tests for floor/rebalancing lot tracking, waterfall merge, tax-off-still-works |

## What This Does NOT Change

- Waterfall ordering (fixed sequence stays — dynamic ordering is the next spec)
- `_enforce_floors` source order (still investments → reserves → cash → BTC for cash floor)
- Tax computation pipeline (`compute_annual_tax` unchanged)
- Quarterly payment logic (unchanged)
- Any UI controls or layout

## Tests

### Helper unit tests
1. `_sell_btc_tracked` creates gains in accumulator when tax on
2. `_sell_btc_tracked` works with empty lots (defensive path)
3. `_sell_btc_tracked` does not record gains when `tax_year_accum is None`
4. `_buy_btc_tracked` creates a lot with correct date/basis/source
5. `_sell_investments_tracked` records LTCG in accumulator
6. `_sell_investments_tracked` no-ops accumulator when tax off

### Regression tests (each bug fixed)
7. Floor enforcement BTC sale now lot-tracked
8. Floor enforcement investment sale now updates cost basis
9. Rebalancing BTC sell now lot-tracked
10. Rebalancing BTC buy creates lot
11. SCF initial purchase creates lot
12. SCF perpetual repayment BTC sale lot-tracked
13. Investment sale during tax payment now recorded in accumulator

### Integration / safety tests
14. Merged waterfall produces same results as old non-tax waterfall (for tax-off scenarios)
15. Tax-off simulation still pays zero tax (critical regression)
16. Lots are seeded even when tax is off
17. Lot inventory sum matches `btc_stack` after sell operations (consistency check)
18. Gradual rebalancing multi-period correctly consumes lots across successive sells
