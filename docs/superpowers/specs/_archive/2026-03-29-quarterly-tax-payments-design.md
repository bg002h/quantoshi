# Quarterly Estimated Tax Payments

**Date:** 2026-03-29
**Scope:** Change tax payment from annual lump sum to quarterly estimated payments with Q4 true-up.

---

## Problem

Tax is computed and paid as a single lump sum at year-end (`period % ppy == 0`). This creates unrealistically large cash drawdowns and makes the cash floor harder to maintain. Real US taxpayers pay estimated taxes quarterly.

## Design

### Payment schedule

Tax is paid in 4 quarterly installments per year:

- **Q1-Q3**: Annualize YTD income from `TaxYearAccumulator`, run `compute_annual_tax()` on the annualized projection, pay 25% of the projected annual tax minus cumulative payments already made YTD. If the computed payment is negative (overpaid), skip — credit carries to Q4.
- **Q4 (year-end)**: Compute actual annual tax from the full year's `TaxYearAccumulator`, subtract Q1-Q3 payments already made, pay the remainder as the true-up. If negative (overpaid), credit reduces `total_taxes_paid`. Then reset the accumulator and `quarterly_tax_paid_ytd` for the new year.

### Quarter boundary detection

For a frequency with `ppy` periods per year, quarter boundaries occur at periods where `period % (ppy // 4) == 0`, excluding period 0. Concretely:

| Frequency | ppy | Quarter every N periods | Quarters at periods |
|-----------|-----|------------------------|---------------------|
| Monthly | 12 | 3 | 3, 6, 9, 12 |
| Weekly | 52 | 13 | 13, 26, 39, 52 |
| Daily | 365 | 91 | 91, 182, 273, 365* |
| Quarterly | 4 | 1 | 1, 2, 3, 4 |
| Annually | 1 | N/A | 1 (year-end only, current behavior) |

For Annually (`ppy == 1`), there's only one period per year, so fall back to year-end-only payment (existing behavior).

### State changes

Add to `CitadelState`:
```python
quarterly_tax_paid_ytd: float = 0.0
```
Tracks cumulative estimated payments made in the current tax year. Reset to 0 at year-end after Q4 true-up.

### Q1-Q3 estimated payment logic

At quarter `q` (1-indexed, q=1,2,3):

```
ytd_fraction = q / 4                          # 0.25, 0.50, 0.75
annualized_accum = copy of TaxYearAccumulator with all income fields / ytd_fraction
projected_tax = compute_annual_tax(annualized_accum, ...).total
cumulative_target = projected_tax * (q / 4)   # 25%, 50%, 75% of projected annual
payment = max(cumulative_target - quarterly_tax_paid_ytd, 0)
```

This ensures cumulative payments track 25%, 50%, 75% of the projected total. If Q2's projection is lower than Q1's (e.g., a Q1 capital gain wasn't repeated), the cumulative target may be below what's already been paid — payment is $0 (no refund until Q4).

### Q4 true-up logic

At year-end (existing `period % ppy == 0` check):
1. Compute actual annual tax via `compute_annual_tax()` (unchanged)
2. `q4_payment = actual_tax - quarterly_tax_paid_ytd`
3. If `q4_payment > 0`: pay via existing payment sourcing (cash → reserves → investments → TD)
4. If `q4_payment <= 0`: overpayment credit (reduce `total_taxes_paid`, no cash movement)
5. Update `loss_carryforward`, append to `annual_tax_history`
6. Reset `quarterly_tax_paid_ytd = 0` and `tax_year_accum` for new year
7. Re-enforce floors

### Payment sourcing

Each quarterly payment uses the same payment sourcing as the current `_year_boundary_tax()`: cash → reserves → investments (gross-up for LTCG) → TD (gross-up for ordinary income). Cash floor re-enforcement runs after each payment.

### No penalties

Underpayment penalties are not modeled. Quarterly payments are for cash flow timing only.

## What doesn't change

- `TaxYearAccumulator` accumulates all year — not reset at quarters
- `compute_annual_tax()` function signature and logic unchanged
- Loss carryforward computed at year-end only
- `annual_tax_history` records one entry per year (not per quarter)
- RMD computation stays at year-end

## Files touched

| File | Change |
|------|--------|
| `engines/citadel.py` | Add `quarterly_tax_paid_ytd` to `CitadelState`, extract `_quarterly_estimated_payment()`, modify step() to call it at quarter boundaries, modify year-end to do Q4 true-up |
| `engines/tax.py` | No changes |
| `test_web.py` | Tests for quarterly timing, true-up, coarse frequency fallback, floor enforcement after each payment |

## Tests

1. Quarterly payments sum to approximately the annual tax (within rounding)
2. Q4 true-up corrects over/underpayment from Q1-Q3 estimates
3. Cash floor respected after each quarterly payment (regression for the bug just fixed)
4. Annual frequency falls back to year-end-only payment
5. `quarterly_tax_paid_ytd` resets to 0 each new year
6. Overpayment at Q4 credits rather than draws
