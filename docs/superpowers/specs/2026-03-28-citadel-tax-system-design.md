# Citadel Planner Tax System Design

**Date:** 2026-03-28
**Status:** Draft
**Depends on:** `2026-03-26-citadel-planner-design.md` (existing Citadel engine)

---

## 1. Overview

Add a comprehensive US federal + state tax simulation layer to the Citadel Planner (tab 9). The system models realistic tax drag on retirement spending by tracking lot-level cost basis, distinguishing short-term vs long-term capital gains, applying progressive federal brackets, NIIT, and state taxes, and optimizing withdrawal ordering across three account wrapper types.

**Design principle:** Tax is an opt-in overlay. A master toggle separates the tax-aware simulation from the current tax-free mode. When off, the existing engine runs unmodified. When on, the full tax system activates via a full-screen modal for configuration.

### Agreed Decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Account wrappers | Three: Taxable, Tax-Deferred (Traditional IRA/401k), Tax-Free (Roth) |
| 2 | Filing status | Single + Married Filing Jointly |
| 3 | AMT | Skip for v1; include NIIT (3.8% surtax) |
| 4 | Cost basis tracking | Lot-level, FIFO default + LIFO option |
| 5 | Withdrawal ordering | Tax-optimized automatic (engine picks cheapest source) |
| 6 | State tax | Dropdown with auto-filled rate lookup |
| 7 | TCJA sunset | Toggle: "Current law" vs "Scheduled sunset" (post-2025 reversion) |
| 8 | UI | Full-screen modal, master toggle, visual indicator, with/without tax comparison |

---

## 2. Tax Law Model

### 2.1 Federal Income Tax Brackets (2025 Current Law)

Stored as tuples `(threshold, rate)` for progressive computation.

**Single:**

| Rate | Taxable Income |
|------|---------------|
| 10% | $0 -- $11,925 |
| 12% | $11,926 -- $48,475 |
| 22% | $48,476 -- $103,350 |
| 24% | $103,351 -- $197,300 |
| 32% | $197,301 -- $252,525 |
| 35% | $252,526 -- $591,975 |
| 37% | $591,976+ |

**Married Filing Jointly:**

| Rate | Taxable Income |
|------|---------------|
| 10% | $0 -- $23,850 |
| 12% | $23,851 -- $96,950 |
| 22% | $96,951 -- $206,700 |
| 24% | $206,701 -- $394,600 |
| 32% | $394,601 -- $505,050 |
| 35% | $505,051 -- $731,200 |
| 37% | $731,201+ |

**TCJA Sunset (post-2025 reversion):** Top rate reverts to 39.6%, bracket thresholds shift. Standard deduction reverts from $15,000/$30,000 to approximately $8,300/$16,600 (2025 inflation-adjusted). The engine stores both bracket sets and both standard deduction values in `tax_data.py`, selecting based on the toggle + simulation year.

**Pre-TCJA Brackets (2025-adjusted estimates for sunset scenario):**

| Rate | Single | MFJ |
|------|--------|-----|
| 10% | $0 -- $11,925 | $0 -- $23,850 |
| 15% | $11,926 -- $48,475 | $23,851 -- $96,950 |
| 25% | $48,476 -- $103,350 | $96,951 -- $206,700 |
| 28% | $103,351 -- $197,300 | $206,701 -- $394,600 |
| 33% | $197,301 -- $252,525 | $394,601 -- $505,050 |
| 35% | $252,526 -- $471,475 | $505,051 -- $565,175 |
| 39.6% | $471,476+ | $565,176+ |

Note: Exact sunset thresholds will be confirmed from IRS guidance if/when TCJA actually expires. These are CPI-adjusted projections from 2017 base values.

### 2.2 Long-Term Capital Gains Brackets

LTCG (assets held >= 1 year) use separate preferential rates:

| Rate | Single | MFJ |
|------|--------|-----|
| 0% | $0 -- $48,350 | $0 -- $96,700 |
| 15% | $48,351 -- $533,400 | $96,701 -- $600,050 |
| 20% | $533,401+ | $600,051+ |

Short-term capital gains (held < 1 year) are taxed as ordinary income using the brackets above.

### 2.3 Net Investment Income Tax (NIIT)

- **Rate:** 3.8% surtax (not inflation-indexed)
- **Threshold:** $200,000 (Single), $250,000 (MFJ)
- **Applies to:** lesser of (a) net investment income or (b) MAGI exceeding threshold
- **Net investment income includes:** capital gains (ST and LT), interest, dividends, rental income
- **Does NOT include:** Traditional IRA/401k distributions (but those DO increase MAGI)

**Effective top combined rates:**
- Long-term: 20% + 3.8% = **23.8%**
- Short-term: 37% + 3.8% = **40.8%** (current law)

### 2.4 State Tax

A lookup table mapping state abbreviation to top marginal rate (51 entries: 50 states + DC). Special handling for Washington state (no income tax but 7%/9.9% capital gains tax).

Applied as a flat rate on all taxable income (simplification -- most users care about the top marginal rate). User can override the rate after selecting a state.

### 2.5 Tax-Advantaged Account Rules

**Tax-Deferred (Traditional IRA/401k):**
- Withdrawals taxed as **ordinary income** (not capital gains)
- Subject to ordinary income brackets
- Does NOT generate NIIT-eligible investment income (but increases MAGI)
- No lot tracking needed (no capital gains concept)
- RMDs starting at age 73 (increases to 75 in 2033 under SECURE 2.0)

**Tax-Free (Roth IRA/401k):**
- Qualified withdrawals are **completely tax-free**
- No impact on taxable income, MAGI, or NIIT
- No RMDs (Roth IRA; Roth 401k also exempt post-SECURE 2.0)
- No lot tracking needed

**Taxable (Brokerage):**
- Capital gains on BTC sales: ST vs LT based on lot holding period
- Capital gains on investment sales: simplified as LT (assumed held > 1 year in a long-horizon sim)
- Interest income (cash, reserves) taxed as ordinary income
- All proceeds count toward NIIT

### 2.6 Bracket Inflation Indexing

Federal brackets are inflation-indexed annually. The engine applies the user's configured inflation rate to adjust bracket thresholds forward from the 2025 base year. This prevents bracket creep from distorting multi-decade simulations.

---

## 3. Account Wrapper Architecture

### 3.1 Wrapper Model

Each existing asset bin is assigned to one of three wrappers:

```
TaxableWrapper
  ├── BTC lots (with per-lot date, amount, cost basis)
  ├── Cash account
  ├── Reserve bins (Short, Medium, Long)
  └── Investment bins (Equities, Bonds)

TaxDeferredWrapper
  ├── BTC stack (no lot tracking -- gains are irrelevant inside TD wrapper)
  ├── Cash account
  ├── Reserve bins
  └── Investment bins

TaxFreeWrapper
  ├── BTC stack (no lot tracking -- gains are irrelevant inside TF wrapper)
  ├── Cash account
  ├── Reserve bins
  └── Investment bins
```

**BTC in retirement accounts:** Self-directed IRAs (e.g., iTrustCapital, Choice) allow holding BTC directly. The simulation supports BTC in all three wrappers. Tax treatment:

- **Taxable BTC:** Lot-level cost basis tracking, ST/LT capital gains on sale.
- **Tax-Deferred BTC:** No capital gains on internal sales. Withdrawals (including BTC converted to cash) are taxed as **ordinary income** regardless of holding period. No lot tracking needed.
- **Tax-Free (Roth) BTC:** No capital gains. Qualified withdrawals are **completely tax-free**. No lot tracking needed. This is the most powerful place to hold BTC for long-term appreciation.

**Key implication for withdrawal ordering:** BTC in a Roth grows tax-free forever. The engine should strongly prefer selling taxable BTC (with capital gains tax) over Roth BTC (zero tax but high opportunity cost). See Section 6.

### 3.2 Wrapper Defaults

| Wrapper | Initial Assets |
|---------|---------------|
| Taxable | All current Citadel defaults (BTC + Cash + Reserves + Investments) |
| Tax-Deferred | $0 (user configures if they have a Traditional IRA/401k) |
| Tax-Free | $0 (user configures if they have a Roth) |

When tax mode is OFF, the engine uses a single implicit "Taxable" wrapper with all assets (identical to current behavior).

### 3.3 Required Minimum Distributions (RMDs)

Applicable to Tax-Deferred wrapper only. RMD start age is **birth-year-based** (not simulation-year-based):
- Born 1951--1959: RMDs begin at age 73
- Born 1960 or later: RMDs begin at age 75 (SECURE 2.0)

Rules:
- RMD amount = prior year-end balance / IRS distribution period factor
- Distribution factors stored as a lookup table (ages 72--120, from IRS Uniform Lifetime Table)
- Example factors: age 73 = 26.5, age 75 = 24.6, age 80 = 20.2, age 85 = 16.0, age 90 = 12.2
- RMD withdrawal is **forced** at the start of each year, taxed as ordinary income
- If the user does not specify a birth year, RMDs are disabled
- Penalty for missed RMD: 25% of shortfall (not modeled in v1 -- engine always forces the distribution)

---

## 4. Lot-Level BTC Tracking

### 4.1 Lot Data Structure

Extends the existing Stack Tracker lot format:

```python
@dataclass
class TaxLot:
    date: str          # ISO date of acquisition (YYYY-MM-DD)
    btc: float         # BTC amount
    cost_basis: float  # USD cost per BTC at acquisition
    source: str        # "initial", "rebal_buy", "scf", "low_q"
```

During simulation, lots are maintained as a sorted list (by date for FIFO, reverse for LIFO).

### 4.2 Lot Operations

**On BTC purchase** (rebalancing buy, low-Q trigger, SCF):
- Create a new `TaxLot` with current date and price as cost basis

**On BTC sale** (spending waterfall, rebalancing sell, high-Q trigger, SCF repay, floor enforcement):
- Select lots via FIFO (default) or LIFO
- For each lot consumed:
  - Compute holding period: `sale_date - lot.date`
  - If >= 365 days: long-term gain
  - If < 365 days: short-term gain
  - Gain = `(sale_price - lot.cost_basis) * btc_sold_from_lot`
  - If gain < 0: capital loss (offsets gains, up to $3,000/yr against ordinary income, remainder carries forward)
- Partially consumed lots are split (remaining BTC stays in the lot list)

### 4.3 Seeding Lots at Simulation Start

**If "Use Stack Tracker lots" is enabled:** Import lots from `effective-lots` store. Each lot provides date, BTC amount, and purchase price (cost basis). This is the primary integration with tab 6.

**If manual BTC entry:** Create a single lot dated `sim_start_date` with cost basis = BTC price at simulation start (current behavior, but now wrapped as a proper lot).

**SCF loan purchase:** Creates an additional lot at simulation start.

---

## 5. Annual Tax Computation

### 5.1 Tax Year Accumulator

The engine maintains a per-year accumulator that resets each January:

```python
@dataclass
class TaxYearAccumulator:
    # --- Gross income components (before deductions) ---
    # Each component is tracked separately for correct MAGI / NIIT / bracket computation.
    tax_deferred_withdrawals: float = 0.0  # Trad IRA/401k distributions (ordinary income)
    interest_income: float = 0.0           # Taxable-wrapper cash + reserve interest (ordinary income + NIIT)
    other_income: float = 0.0              # External: wages, SS, pension (from SimConfig, inflation-adj)

    # Capital gains/losses -- tracked separately by category for IRS netting rules
    st_capital_gains: float = 0.0          # Short-term gains (taxable wrapper)
    st_capital_losses: float = 0.0         # Short-term losses (positive number)
    lt_capital_gains: float = 0.0          # Long-term gains (taxable wrapper)
    lt_capital_losses: float = 0.0         # Long-term losses (positive number)
    loss_carryforward: float = 0.0         # Net capital loss carried from prior years

    # Roth (tracked for reporting, not taxed)
    roth_withdrawals: float = 0.0          # Not taxed, does not affect MAGI or NIIT

    # RMD tracking
    rmd_required: float = 0.0             # Required minimum for this year
    rmd_taken: float = 0.0                # Amount already withdrawn toward RMD
```

**Key definitions derived from the accumulator:**

```python
# 1. Capital loss netting (IRS Section 1(h) order):

# 1a. Net within each category
net_st = st_capital_gains - st_capital_losses
net_lt = lt_capital_gains - lt_capital_losses

# 1b. Apply prior-year loss carryforward (v1 simplification: treat as LT loss)
net_lt -= loss_carryforward

# 1c. Cross-category offset: if one is net negative, reduce the other
if net_st < 0 and net_lt > 0:
    combined = net_st + net_lt
    net_st = min(combined, 0)
    net_lt = max(combined, 0)
elif net_lt < 0 and net_st > 0:
    combined = net_st + net_lt
    net_st = max(combined, 0)
    net_lt = min(combined, 0)

# 1d. Remaining net capital loss: up to $3,000 deduction against ordinary income
total_net_loss = abs(min(net_st, 0)) + abs(min(net_lt, 0))
net_cap_loss_deduction = min(total_net_loss, 3000)
# Excess carries forward to next year
new_loss_carryforward = max(total_net_loss - 3000, 0)

# 2. AGI (Adjusted Gross Income):
AGI = (tax_deferred_withdrawals + interest_income + other_income
       + max(net_st, 0) + max(net_lt, 0) - net_cap_loss_deduction)

# 3. MAGI (for NIIT threshold -- same as AGI for most filers):
MAGI = AGI

# 4. Taxable income (standard deduction allocation):
standard_deduction = get_standard_deduction(filing_status, tcja_sunset, sim_year)
ordinary_gross = AGI - max(net_lt, 0)  # AGI minus LTCG = ordinary component
ordinary_taxable = max(ordinary_gross - standard_deduction, 0)
# Standard deduction first reduces ordinary income; excess can reduce LTCG:
remaining_deduction = max(standard_deduction - ordinary_gross, 0)
taxable_ltcg = max(max(net_lt, 0) - remaining_deduction, 0)

# 5. Net investment income (for NIIT):
NII = max(net_st, 0) + max(net_lt, 0) + interest_income
# Note: TD withdrawals are NOT NII (but they ARE in MAGI)
```

### 5.2 Tax Computation (Annual)

At each year boundary (or end of simulation), compute taxes owed. Uses the derived quantities from the accumulator (Section 5.1).

**Step 1: Capital loss netting.** Apply IRS Section 1(h) netting order (see accumulator definitions above). Produces `net_st`, `net_lt`, `net_cap_loss_deduction`, and updated `loss_carryforward`.

**Step 2: AGI and MAGI.**
```
AGI = tax_deferred_withdrawals + interest_income + other_income
      + max(net_st, 0) + max(net_lt, 0) - net_cap_loss_deduction
MAGI = AGI  (same for most filers; no AMT adjustments in v1)
```

**Step 3: Standard deduction and taxable income.**
- Standard deduction 2025: $15,000 (Single), $30,000 (MFJ). Inflation-indexed annually.
- Under TCJA sunset: reverts to ~$8,300/$16,600 (inflation-adjusted).
- Deduction first reduces ordinary income; any excess reduces LTCG:
```
ordinary_taxable = max(AGI - max(net_lt, 0) - standard_deduction, 0)
remaining_deduction = max(standard_deduction - (AGI - max(net_lt, 0)), 0)
taxable_ltcg = max(net_lt - remaining_deduction, 0)
```

**Step 4: Federal tax on ordinary income.** Apply progressive brackets to `ordinary_taxable`.

**Step 5: Federal tax on LTCG.** Apply 0%/15%/20% brackets to `taxable_ltcg` with **stacking rule** -- LTCG brackets start where ordinary income left off:
```python
# LTCG stacking: 0%/15%/20% brackets start above ordinary_taxable
ltcg_base = ordinary_taxable  # from Step 3, post-standard-deduction
federal_ltcg = apply_ltcg_brackets(taxable_ltcg, stacking_base=ltcg_base, filing_status)
# Example: if ordinary_taxable=$80k (Single), the 0% LTCG bracket ($0-$48,350)
# is already "filled", so the first ~$453k of LTCG is taxed at 15%.
```

**Step 6: NIIT.**
```
NII = max(net_st, 0) + max(net_lt, 0) + interest_income
niit_threshold = 200_000 if single else 250_000  # NOT inflation-indexed
niit = 0.038 * min(NII, max(MAGI - niit_threshold, 0))
```

**Step 7: State tax.** `state_rate * AGI`. Uses the looked-up or overridden rate. Roth withdrawals are excluded (they are not in AGI). v1 simplification: applies the flat top marginal rate to full AGI without a state-level standard deduction (state deductions vary wildly and are a minor factor compared to the rate itself; consistent with the "flat top rate" simplification documented in Section 2.4).

**Step 8: Total tax liability.** `federal_ordinary + federal_ltcg + niit + state`

### 5.3 Tax Payment Timing

Taxes are computed annually but **paid from the Taxable wrapper** (cash first, then reserves, then investments). Payment waterfall:

1. **Taxable cash** (no additional tax event)
2. **Taxable reserves** (no additional tax event -- principal withdrawal)
3. **Taxable investments** (generates LTCG on the sold portion)
4. **Tax-Deferred withdrawal** (generates ordinary income)
5. **BTC sale** (generates ST or LT capital gains depending on lot)

When steps 3--5 generate additional taxable income, use a **gross-up formula** to compute the exact amount needed in one pass (no iteration):
```
gross_amount = net_tax_owed / (1 - marginal_rate)
```
Where `marginal_rate` is the effective rate on the next dollar withdrawn from that source (ordinary bracket rate for TD, LTCG rate + NIIT for investments/BTC). This avoids infinite recursion while correctly modeling the tax-on-tax effect.

If all sources are exhausted, the remaining unpaid tax is recorded as `tax_shortfall` (simulation is effectively over -- the user cannot meet obligations). Depletion detection considers all three wrappers combined.

This models the real-world constraint that you must pay taxes from somewhere, and that payment itself can generate taxable events.

---

## 6. Tax-Optimized Withdrawal Ordering

### 6.1 Optimization Strategy

The engine uses a **growth-aware greedy heuristic** that balances three factors each period:

1. **Tax cost** — the marginal tax rate on withdrawing from this source now
2. **Growth potential** — the projected future return of this asset (higher growth = higher opportunity cost to sell)
3. **Tax shelter value** — Roth > Tax-Deferred > Taxable (tax-free compounding is most valuable on the highest-growth asset)

**BTC growth is model-derived and time-varying.** The bubble/power-law model projects declining annual growth over time (e.g., ~40%/yr in 2031, ~10%/yr in 2050, ~5%/yr in 2065). The engine computes BTC's **forward-looking annualized growth rate** at each period:

```python
# Forward-looking BTC growth from the price model
btc_price_now = model.price_at(t)
btc_price_next = model.price_at(t + 1)  # 1 year ahead
btc_fwd_growth = (btc_price_next / btc_price_now) - 1
```

This rate determines where BTC sits in the withdrawal order relative to other assets. When `btc_fwd_growth >> equity_return`, BTC is extremely expensive to sell (high opportunity cost). When `btc_fwd_growth ≈ equity_return`, BTC and equities are roughly interchangeable.

**Opportunity cost per dollar withdrawn:**
```
cost(source) = tax_rate(source) + growth_rate(source) * shelter_multiplier(source)
```
Where `shelter_multiplier` = 1.0 for Taxable, ~1.3 for Tax-Deferred (tax deferral value), ~1.5 for Roth (tax-free forever). The engine sorts sources by ascending cost and draws from cheapest first.

### 6.2 Simplified Implementation

Rather than full dynamic programming, use the growth-aware greedy heuristic each period. The **base ordering** (which shifts dynamically based on BTC growth rate and accumulated income):

**When BTC growth is HIGH (e.g., early decades, >15%/yr):**
1. **Taxable principal** (cash, reserve principal -- no gain, low growth)
2. **Tax-Deferred cash/reserves/investments** up to 12% bracket (bracket-filling)
3. **Taxable investments** (LT gains, ~10% growth)
4. **Tax-Deferred remaining** (higher brackets, but lower-growth assets)
5. **Taxable BTC short-term lots** (if forced -- ST rates, but avoid selling high-growth BTC)
6. **Taxable BTC long-term lots** (high growth makes this expensive despite lower tax rate)
7. **Roth cash/reserves/investments** (tax-free, moderate growth)
8. **Roth BTC** (absolute last -- tax-free compounding on highest-growth asset)

**When BTC growth is LOW (e.g., late decades, <8%/yr):**
1. **Taxable principal** (cash, reserves)
2. **Tax-Deferred bracket-filling** (up to 12%)
3. **Taxable BTC long-term lots** (LT gains, but growth is now comparable to equities)
4. **Taxable investments** (may have higher growth than late-stage BTC)
5. **Tax-Deferred remaining**
6. **Taxable BTC short-term lots**
7. **Roth cash/reserves/investments**
8. **Roth BTC** (still last -- tax-free shelter is always most valuable)

The key insight: **BTC's position in the withdrawal order is not fixed.** In early years the model says "never sell BTC if you can avoid it." In later years, BTC growth converges toward equity-like returns and selling BTC becomes less costly. The engine re-evaluates every period.

The bracket-filling threshold for Tax-Deferred withdrawals ensures low brackets aren't "wasted" -- this is the most impactful optimization for multi-decade simulations.

### 6.3 RMD Interaction

If Tax-Deferred RMD has not been met for the year, RMD draws are forced before any other Tax-Deferred withdrawals. RMDs count toward spending (they reduce the amount needed from other sources).

---

## 7. Engine Changes

### 7.1 New Config Fields

```python
# Added to SimConfig
tax_enabled: bool = False
filing_status: str = "single"            # "single" or "mfj"
state_code: str = "TX"                   # State abbreviation (TX = no tax default)
state_rate_override: float | None = None # Override auto rate
tcja_sunset: bool = False                # If True, revert to pre-TCJA after 2025
birth_year: int | None = None            # For RMD computation (None = no RMDs)
cost_basis_method: str = "fifo"          # "fifo" or "lifo"
other_income: float = 0.0               # Annual non-simulation income (wages, SS, etc.)
other_income_growth: float = 0.0         # Annual growth rate for other_income (default: 0 = real dollars;
                                         # set equal to inflation for nominal dollars)

# Tax-Deferred wrapper initial balances
td_btc_stack: float = 0.0             # BTC held in Traditional IRA (self-directed)
td_cash_initial: float = 0.0
td_reserve_bins: list[dict] = field(default_factory=lambda: [
    {"label": "Short", "initial": 0.0},
    {"label": "Medium", "initial": 0.0},
    {"label": "Long", "initial": 0.0},
])
td_invest_bins: list[dict] = field(default_factory=lambda: [
    {"label": "Equities", "initial": 0.0},
    {"label": "Bonds", "initial": 0.0},
])

# Tax-Free (Roth) wrapper initial balances
tf_btc_stack: float = 0.0             # BTC held in Roth IRA (self-directed)
tf_cash_initial: float = 0.0
tf_reserve_bins: list[dict] = field(default_factory=lambda: [
    {"label": "Short", "initial": 0.0},
    {"label": "Medium", "initial": 0.0},
    {"label": "Long", "initial": 0.0},
])
tf_invest_bins: list[dict] = field(default_factory=lambda: [
    {"label": "Equities", "initial": 0.0},
    {"label": "Bonds", "initial": 0.0},
])
```

Note: TD/TF wrappers share the same growth rates/volatility as the Taxable wrapper (configured in the existing Assets sub-tab). Only initial balances differ per wrapper.

### 7.2 New State Fields

```python
# Added to CitadelState
tax_lots: list[TaxLot] = field(default_factory=list)
tax_year_accum: TaxYearAccumulator = field(default_factory=TaxYearAccumulator)
loss_carryforward: float = 0.0
total_taxes_paid: float = 0.0
annual_tax_history: list[dict] = field(default_factory=list)  # Per-year tax breakdown

# Wrapper balances (when tax_enabled)
td_btc_stack: float = 0.0      # Tax-Deferred BTC (no lot tracking)
td_cash: float = 0.0           # Tax-Deferred cash
td_reserves: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
td_investments: list[float] = field(default_factory=lambda: [0.0, 0.0])
tf_btc_stack: float = 0.0      # Tax-Free BTC (no lot tracking)
tf_cash: float = 0.0           # Tax-Free cash
tf_reserves: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
tf_investments: list[float] = field(default_factory=lambda: [0.0, 0.0])
```

### 7.3 Modified Engine Flow

The `step()` method gains a tax branch:

```
if tax_enabled:
    1. Grow all three wrapper balances (same return models, same rates)
    2. Accumulate TAXABLE-WRAPPER-ONLY interest income to tax_year_accum.interest_income
       (TD/TF interest grows tax-deferred/tax-free -- not a current taxable event)
    3. Inflate other_income: accum.other_income = config.other_income * (1 + growth)^years
    4. Check/force RMD if applicable (age-based, adds to accum.tax_deferred_withdrawals)
    5. Evaluate rebalancing (BTC sales create lot gain/loss entries in accumulator)
    6. Apply tax-optimized spending waterfall (across all 3 wrappers, Section 6.2 order)
    7. Enforce floors (within taxable wrapper only)
    8. At year boundary: compute annual tax (Section 5.2), pay from taxable wrapper (Section 5.3)
    9. Reset tax_year_accum (except loss_carryforward), record annual_tax_history entry
    10. Check total depletion across ALL three wrappers (not just taxable)
else:
    (existing logic unchanged)
```

### 7.4 SimResult Extensions

```python
# New fields in SimResult (or parallel arrays)
taxes_paid: list[float]          # Cumulative taxes paid per period
annual_taxes: list[dict]         # Per-year breakdown (see below)
td_total: list[float]            # Tax-Deferred balance per period
tf_total: list[float]            # Tax-Free balance per period
taxable_total: list[float]       # Taxable balance per period
tax_shortfall: float = 0.0       # Unpaid tax when all accounts depleted
```

Each entry in `annual_taxes`:
```python
{
    "year": int,
    # Income components (for Tax Summary Panel display)
    "ordinary_income": float,     # TD withdrawals + interest + other_income
    "st_gains": float,            # Net short-term gains
    "lt_gains": float,            # Net long-term gains
    "agi": float,
    "standard_deduction": float,
    # Tax amounts
    "federal_ordinary": float,    # Tax on ordinary income
    "federal_ltcg": float,        # Tax on long-term gains
    "niit": float,
    "state": float,
    "total": float,
    "effective_rate": float,      # total / AGI (or 0 if AGI <= 0)
    # Loss tracking
    "loss_carryforward": float,   # Carried to next year
}
```

---

## 8. UI Design

### 8.1 Master Toggle

Located in the Citadel **Simulation** sub-tab, prominently placed above the "Run Simulation" button:

```
┌─────────────────────────────────────────┐
│  💰 Taxation   [OFF ⬤───── ON]          │
│  ┌──────────────────────────────────┐   │
│  │  ⚙️  Configure Tax Settings...   │   │  ← Opens full-screen modal
│  └──────────────────────────────────┘   │
│                                         │
│  [▶ Run Simulation]                     │
└─────────────────────────────────────────┘
```

- Toggle is a `dbc.Switch` with clear ON/OFF labeling
- "Configure Tax Settings" button only visible when toggle is ON
- Visual state indicator: when tax is ON, the Run button label changes to "Run Simulation (with Tax)"

### 8.2 Full-Screen Tax Configuration Modal

Opens via `dbc.Modal(fullscreen=True)`. Three sections organized as vertical cards:

#### Section A: Filing & Rates

```
┌─ Filing & Location ──────────────────────────────────────────┐
│                                                               │
│  Filing Status    ◉ Single    ○ Married Filing Jointly        │
│                                                               │
│  State            [California ▾]    Rate: [13.30] %           │
│                                     (auto-filled, editable)   │
│                                                               │
│  Birth Year       [1985    ]   (for RMDs -- leave blank to    │
│                                 skip RMD modeling)            │
│                                                               │
│  Other Annual     [$0      ]   (wages, SS, pension --         │
│  Income                         increases MAGI/bracket)       │
│  Income Growth    [0       ] % (annual growth; 0 = real $)    │
│                                                               │
│  Tax Law          ◉ Current law (TCJA)                        │
│                   ○ Scheduled sunset (post-2025 reversion)    │
│                                                               │
│  Cost Basis       ◉ FIFO (sell oldest first)                  │
│  Method           ○ LIFO (sell newest first)                  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

#### Section B: Account Wrappers

Three cards side-by-side (or stacked on mobile):

```
┌─ Taxable Account ────────┐  ┌─ Tax-Deferred (Trad. IRA) ─┐  ┌─ Tax-Free (Roth) ─────────┐
│                           │  │                              │  │                             │
│  (Uses existing Citadel   │  │  BTC Stack   [0.0      ]   │  │  BTC Stack   [0.0      ]   │
│   asset configuration     │  │  Cash        [$0        ]   │  │  Cash        [$0        ]   │
│   from Assets sub-tab)    │  │  Reserves                    │  │  Reserves                   │
│                           │  │    Short     [$0        ]   │  │    Short     [$0        ]   │
│  BTC: from Assets tab     │  │    Medium    [$0        ]   │  │    Medium    [$0        ]   │
│  Cash: from Assets tab    │  │    Long      [$0        ]   │  │    Long      [$0        ]   │
│  Reserves: from Assets    │  │  Investments                 │  │  Investments                │
│  Investments: from Assets │  │    Equities  [$0        ]   │  │    Equities  [$0        ]   │
│                           │  │    Bonds     [$0        ]   │  │    Bonds     [$0        ]   │
│  Growth rates: same as    │  │                              │  │                             │
│  Assets tab config        │  │  Growth rates: same as       │  │  Growth rates: same as      │
│                           │  │  Taxable wrapper             │  │  Taxable wrapper            │
└───────────────────────────┘  └──────────────────────────────┘  └─────────────────────────────┘
```

**Key design choices:**
- The Taxable wrapper's balances are the *existing* Assets sub-tab values. The modal only shows Tax-Deferred and Tax-Free initial balances (defaulting to $0). This means existing non-tax users see zero UI change in the Assets tab.
- **BTC in retirement accounts:** TD and TF wrappers each have a BTC Stack field (self-directed IRA). No lot tracking -- gains inside retirement accounts are not taxable events. TD BTC withdrawals are taxed as ordinary income; TF BTC withdrawals are tax-free.

#### Section C: Tax Rate Reference (read-only)

A collapsible "Reference: Current Tax Brackets" panel showing:
- Ordinary income brackets for selected filing status
- LTCG brackets
- NIIT threshold
- Standard deduction
- State rate

This is informational only -- the engine uses these internally. Shown so the user understands what's being applied.

#### Modal Footer

```
┌──────────────────────────────────────────────────────────────┐
│  [Cancel]                              [Save Tax Settings]   │
└──────────────────────────────────────────────────────────────┘
```

"Save" persists to a `dcc.Store("cp-tax-config", storage_type="memory")`. Settings survive within the session but reset on page reload (consistent with Citadel's existing non-persistent design).

### 8.3 Visual Indicator on Chart

When tax mode is ON:

- **Chart title** appends: `(Federal + {State} Tax)`
- **Subtitle annotation** shows effective tax rate: `Eff. rate: {X}% | Taxes paid: ${Y}`
- **"Taxes Paid" trace** (red area, secondary y-axis) shows cumulative tax drag
- **Color distinction:** Tax-mode charts use a slightly different background tint or border to make screenshots unambiguous

### 8.4 With/Without Tax Comparison

When tax mode is ON, the chart includes:

- **Dashed "no-tax" ghost traces** for Total Portfolio and BTC Holdings, showing what the balances would be without tax drag
- **Annotation** showing the tax drag delta at simulation end: `Tax drag: -${X} ({Y}%)`
- These ghost traces can be toggled via a "Show tax comparison" checkbox in chart toggles

### 8.5 Tax Summary Panel

Below the chart (or in a collapsible accordion), a year-by-year tax breakdown table:

| Year | Ordinary Inc | ST Gains | LT Gains | Loss Ded. | Federal Tax | NIIT | State Tax | Total Tax | Eff. Rate | Carryforward |
|------|-------------|----------|----------|-----------|-------------|------|-----------|-----------|-----------|-------------|
| 2031 | $60,000 | $0 | $45,000 | $0 | $12,400 | $0 | $5,980 | $18,380 | 17.5% | $0 |
| 2032 | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Note: ST/LT Gains columns show **net** values (can be negative when losses exceed gains). "Loss Ded." shows the $3,000 capital loss deduction applied against ordinary income. "Carryforward" shows unused losses carried to the next year.

---

## 9. Snapshot / Share Integration

### 9.1 New Snapshot Controls

Add to `_SNAPSHOT_CONTROLS`:
- `("cp-tax-enabled", "value")` -- the master toggle
- `("cp-tax-config", "data")` -- the full tax config dict (filing status, state, birth year, etc.)
- `("cp-td-*", "value")` -- Tax-Deferred initial balances (6 fields)
- `("cp-tf-*", "value")` -- Tax-Free initial balances (6 fields)

Tax config is a single JSON-serialized store, keeping the snapshot control count manageable.

`_TAB_CONTROLS["citadel"]` must include `"cp-tax-config"`, `"cp-tax-enabled"`, and all `"cp-td-*"` / `"cp-tf-*"` component IDs for single-tab snapshot filtering to work correctly.

### 9.2 Tab Defaults Extension

Add to `tab_defaults.py` CITADEL dict:
```python
"tax_enabled": False,
"filing_status": "single",
"state_code": "TX",
"state_rate_override": None,
"tcja_sunset": False,
"birth_year": None,
"cost_basis_method": "fifo",
"other_income": 0, "other_income_growth": 0,
"td_btc": 0.0, "td_cash": 0, "td_res_short": 0, "td_res_med": 0, "td_res_long": 0,
"td_inv_eq": 0, "td_inv_bd": 0,
"tf_btc": 0.0, "tf_cash": 0, "tf_res_short": 0, "tf_res_med": 0, "tf_res_long": 0,
"tf_inv_eq": 0, "tf_inv_bd": 0,
```

---

## 10. Tax Module Organization

### 10.1 New Files

| File | Purpose |
|------|---------|
| `btc_web/engines/tax.py` | Tax computation engine: brackets, NIIT, state rates, annual tax calc |
| `btc_web/engines/tax_lots.py` | Lot-level tracking: TaxLot dataclass, FIFO/LIFO sell, gain computation |
| `btc_web/engines/tax_data.py` | Static data: federal brackets (current + sunset), LTCG brackets, state rate table, RMD factors, standard deductions |
| `btc_web/layout/citadel_tax.py` | Tax modal layout + master toggle |
| `btc_web/callbacks/citadel_tax_cb.py` | Tax modal callbacks (open/close, state dropdown, save config) |

### 10.2 Modified Files

| File | Changes |
|------|---------|
| `btc_web/engines/citadel.py` | Add wrapper fields to SimConfig/CitadelState, tax branch in `step()`, tax-aware waterfall |
| `btc_web/figures/citadel.py` | Ghost traces, tax summary trace, tax-drag annotation, `_build_sim_config()` tax fields |
| `btc_web/layout/citadel.py` | Insert master toggle in Simulation sub-tab, include tax modal |
| `btc_web/callbacks/citadel_cb.py` | Pass tax config to figure builder, new State inputs |
| `btc_web/tab_defaults.py` | Add tax defaults to CITADEL dict |
| `btc_web/snapshot.py` | Add tax controls to `_SNAPSHOT_CONTROLS` |
| `btc_web/cache.py` | Ensure tax params are part of cache key when tax is enabled |

---

## 11. Performance Considerations

- **Lot tracking overhead:** Each BTC transaction creates/modifies lots. For a 44-year monthly simulation with regular rebalancing, expect ~500-1000 lot operations. This is trivial computationally.
- **Annual tax computation:** One tax calc per year boundary (up to 44 years). Each involves bracket lookups + a few multiplications. Negligible.
- **Three-wrapper accounting:** Triples the asset bin count (7 → 21 bins). Growth computation scales linearly. Still fast.
- **Cache key expansion:** Tax config adds ~10 params to the cache key. Cache miss rate increases for tax-enabled runs (many more parameter combinations), but L1 LRU still works. L0 prewarm only covers tax-OFF defaults.
- **Comparison traces:** Running two sims (tax-on + tax-off) doubles compute for comparison mode. Since each sim is <100ms, this is acceptable.

---

## 12. Testing Strategy

### 12.1 Unit Tests (tax engine)

- Bracket computation: verify tax on known income amounts against IRS tax tables
- NIIT: threshold behavior, lesser-of rule
- LTCG stacking: verify 0% bracket fills after ordinary income
- Lot FIFO/LIFO: correct lot selection, partial lot splitting, ST vs LT classification
- Loss harvesting: $3,000 offset, carryforward
- RMD: correct factors by age, forced withdrawal
- State tax: lookup table correctness
- TCJA sunset: bracket shift at correct year

### 12.2 Integration Tests (engine)

- Full simulation with tax ON vs OFF: verify tax reduces terminal wealth
- Withdrawal ordering: verify taxable drawn before Roth (Roth preserved as last resort)
- RMD forcing: verify distributions happen at correct age
- Year-boundary tax payment: verify taxes deducted from correct accounts
- Lot seeding from Stack Tracker: verify imported lots produce correct cost basis

### 12.3 UI Tests

- Modal open/close
- State dropdown populates rate
- Filing status toggle updates reference brackets
- Master toggle enables/disables Configure button
- Snapshot round-trip with tax config
- Comparison traces appear/disappear with toggle

---

## 13. Future Extensions (Not In Scope v1)

- **AMT:** Full parallel tax computation. Add when "other deductions" input is available.
- **Tax-loss harvesting optimizer:** Proactively sell losing lots to offset gains.
- **Roth conversion ladder:** Model strategic Traditional → Roth conversions.
- **State-specific capital gains:** WA's 7%/9.9% capital gains tax, state-level LTCG exemptions.
- **Estate/inheritance tax:** Federal estate tax exemption ($13.99M in 2025).
- **Quarterly estimated tax payments:** Model cash flow impact of quarterly payments vs annual.
- **Medicare IRMAA:** Income-based Medicare premium surcharges.
- **Social Security taxation:** Up to 85% of SS benefits taxable above certain thresholds.
- **Charitable giving strategies:** Donor-advised funds, appreciated BTC donation (avoid capital gains).
