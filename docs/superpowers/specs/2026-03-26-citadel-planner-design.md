# Citadel Planner (Tab 9) — Design Spec

**Date:** 2026-03-26
**Scope:** New tab — multi-asset retirement simulator with BTC, cash, treasuries, equities/bonds, rebalancing rules, and leveraged BTC accumulation.
**Phase:** Sub-project A (Tab 9). Sub-project B (app-wide save/load system) designed in parallel, implemented separately.

---

## Goal

Add a Tab 9 "Citadel Planner" that simulates retirement across a multi-asset portfolio: Bitcoin (priced via QR/MC models), a cash account, a US Treasury reserve fund (3 maturity bins), and an investment account (equities + bonds). The user configures spending, allocation, rebalancing rules (BTC quantile triggers + account floor rules), and an optional leveraged BTC accumulation feature ("Saylor Citadel Fortifier"). The simulator runs a step-by-step loop where BTC transitions, dollar-asset returns, rebalancing, and spending all interact each period.

---

## Asset Classes

### 1. Bitcoin Stack
- Initial BTC holdings (manual entry or from Stack Tracker lots)
- Priced via the selected price model (Bubble Model default, also PL, S2F)
- For deterministic mode (1 sim): price follows selected quantile path
- For MC mode (N sims): price transitions drawn from Markov chain of selected model
- BTC is the last-resort spending source — only liquidated when all dollar accounts are depleted, or strategically via rebalancing rules

### 2. Cash Account
- Initial balance ($)
- Annual interest rate (%)
- Volatility: 0 (deterministic growth)
- First in the spending waterfall — covers expenses before any other account is touched

### 3. Reserve Fund — US Treasury Obligations
3 maturity bins for v1 (architected for 4-5 bins in future versions):

| Bin | Label | Maturity | Description |
|-----|-------|----------|-------------|
| Short | T-Bills | ≤ 1 year | Near-cash, lowest yield, lowest volatility |
| Medium | T-Notes | 2–10 years | Moderate yield and volatility |
| Long | T-Bonds | 10–30 years | Highest yield, highest volatility |

Each bin has:
- Allocation: initial dollar amount ($)
- Expected return (% annual) — labeled "Expected return" not "Interest rate" since this is total return (coupon + price change)
- Volatility (% annual)

v1: User-input rates per bin. Architecture supports stochastic rate simulation (Markov chain on yield curve) for a future version.

### 4. Investment Account
2 bins for v1 (architected for 3+ bins in future versions):

| Bin | Label | Description |
|-----|-------|-------------|
| Equities | Stocks | Higher expected return, higher volatility |
| Bonds | Bonds | Lower expected return, lower volatility |

Each bin has:
- Allocation: initial dollar amount ($)
- Expected return (% annual)
- Volatility (% annual)

### Return Modeling
- **Cash:** Deterministic compound interest. `balance * (1 + rate)^(1/ppy)` per period.
- **Reserves and Investments:** Lognormal returns to prevent negative balances:
  ```
  # annual_rate and annual_vol are decimals (e.g., 0.10 for 10%)
  # Guard: if annual_rate <= 0, use simple normal with floor at -1
  sigma_ln = sqrt(log(1 + (annual_vol / (1 + annual_rate))^2))
  mu_ln = log(1 + annual_rate) - sigma_ln^2 / 2
  # Scale to period: mean scales by 1/ppy, vol scales by 1/sqrt(ppy)
  period_mu = mu_ln / ppy
  period_sigma = sigma_ln / sqrt(ppy)
  period_return = exp(N(period_mu, period_sigma)) - 1
  # Result is always > -1.0 (balance cannot go negative)
  ```
- **Deterministic mode (n_sims=1):** All dollar assets use expected returns only, no volatility draws. Everything is deterministic.
- **MC mode (n_sims>1):** Dollar assets use lognormal draws with specified volatility.

---

## Spending

### Monthly Spending
- Base amount ($/month)
- Inflation rate (% / year)
- Spending growth above inflation (% / year) — e.g., lifestyle creep or planned increases
- Combined formula: `period_spend(t) = monthly_spend * (1 + (inflation + spend_growth) / 100)^years_elapsed * (12 / ppy)`
  - `monthly_spend` is the base dollar amount per month
  - `inflation` and `spend_growth` are annual percentages; divided by 100 for the formula
  - `years_elapsed = period_index / ppy` where `ppy` = periods per year
  - `12 / ppy` scales the monthly amount to the simulation frequency (e.g., x3 for quarterly)
- Pre-computed schedule before simulation loop (deterministic, independent of portfolio state in v1)

### v2 planned extensions:
- Start-after-N-years and last-for-Y-years windows for the growth rate
- Multiple spending phases (e.g., high spend years 1-10, lower after)

### v3 planned extensions:
- Adaptive spending cuts in response to bad market/BTC returns

### Spending Waterfall
When monthly expenses are due, accounts are drawn in this priority order:
```
Cash -> Reserves (Short -> Medium -> Long) -> Investments (Bonds -> Equities) -> BTC (emergency liquidation)
```

BTC is the **last resort**. The rebalancing rules are the *strategic* mechanism for moving BTC value into dollar accounts. If all dollar accounts are depleted and no rebalancing has fired, BTC is sold to cover expenses. Depletion occurs when ALL assets (including BTC) reach zero.

Partial-period handling: returns are applied first, then spending is withdrawn. If an account is drained mid-waterfall, the remainder cascades to the next account.

---

## Rebalancing Rules

### BTC Quantile Triggers (v1)
Two independent triggers based on the current BTC price's quantile position:

**High-quantile trigger** — "BTC is overvalued, take profits":
- Threshold: BTC quantile >= X% (e.g., 80%)
- Action: sell BTC, distribute proceeds to dollar accounts

**Low-quantile trigger** — "BTC is cheap, accumulate":
- Threshold: BTC quantile <= Y% (e.g., 20%)
- Action: sell from dollar accounts (reverse waterfall: Investments -> Reserves -> Cash), buy BTC

**Config validation:** `high_q_trigger > low_q_trigger` enforced, minimum 5 percentile point gap.

### Action Modes
Each trigger has one of two action modes:

**Gradual** — sell/buy X% of source per period for N periods:
- Rate: % per period (e.g., 1%/month)
- Duration: number of periods (e.g., 12 months)
- If a new trigger fires while a gradual action is in progress, the new trigger is **ignored** (v1 simplicity). The in-progress gradual must complete first.

**Lump** — sell/buy X% of source immediately:
- Rate: % (e.g., 10% of BTC stack)
- Cooldown: minimum periods between lump fires (v1: 12 periods = 1 year for monthly sims)
- Cannot fire again until cooldown expires

### Proceeds Distribution
When selling BTC (high trigger), proceeds are distributed by user-defined split:
- Split is a dict with 6 keys: `{"cash": %, "res_short": %, "res_med": %, "res_long": %, "inv_eq": %, "inv_bd": %}`
- All values are percentages (0-100) in the UI; callback divides by 100 before passing to engine
- Engine stores splits as fractions (0.0-1.0), must sum to 1.0 (validated in `validate_config`)
- UI: 6 number inputs with labels, constrained to non-negative, with real-time sum display
- Component IDs: `cp-high-q-split-cash`, `cp-high-q-split-rs`, `cp-high-q-split-rm`, `cp-high-q-split-rl`, `cp-high-q-split-eq`, `cp-high-q-split-bd`

When buying BTC (low trigger), funds are sourced by user-defined split:
- Source split: same schema as proceeds split — `{"cash": %, "res_short": %, ...}`
- Component IDs: `cp-low-q-split-cash`, `cp-low-q-split-rs`, `cp-low-q-split-rm`, `cp-low-q-split-rl`, `cp-low-q-split-eq`, `cp-low-q-split-bd`
- Each source account is drawn proportionally; if a source account has insufficient funds, the shortfall is redistributed proportionally to remaining sources with nonzero allocation
- If total available across ALL sources is less than the intended BTC buy amount, buy only what is available (partial execution)

### Account Floor Rules
Minimum balance maintenance, evaluated every period **before** rebalancing triggers:

- **Cash floor:** maintain minimum $X in cash
- **Reserve bin floors:** maintain minimum $X in each of Short/Medium/Long

When an account falls below its floor, the deficit is replenished from other dollar accounts. Draw order for replenishment:
1. Investment Bonds (lowest priority dollar asset)
2. Investment Equities
3. Reserve Long (longest maturity first)
4. Reserve Medium
5. Reserve Short
6. Cash (only if a reserve bin needs replenishing; cash never replenishes itself)

Each source is drawn up to the amount needed, cascading to the next if insufficient. If multiple accounts are below their floors simultaneously, they are replenished in waterfall priority order (cash first, then reserves short->med->long). BTC is NOT sold to maintain floors — floors only redistribute among dollar accounts. If total dollar assets are insufficient to meet all floors, each account gets its proportional share of available funds.

### Evaluation Order Per Period
```
1. Update BTC price (model transition)
2. Simulate dollar-asset returns
3. Enforce floor rules (replenish accounts below minimums)
4. Compute BTC quantile from current price
5. Evaluate rebalancing triggers -> execute if fired
6. Apply spending via waterfall
7. Check depletion (all assets including BTC <= 0)
```

### v2 planned trigger extensions:
- Equity/bond return triggers (trailing return thresholds)
- BTC-to-equity ratio trigger (relative valuation)
- User-configurable lump cooldown duration
- Gradual override rules (allow new trigger to cancel in-progress gradual)

---

## Saylor Citadel Fortifier

Leveraged BTC accumulation feature — borrow money to buy BTC, service the loan from the spending budget or a dedicated allocation.

### Loan Types

**Term Loan:**
- Fixed duration (months)
- Amortizing or interest-only (radio selection)
- Interest rate (% annual)
- Loan amount ($)
- Regular payments deducted from spending budget (same pattern as Stack-celerator on tab 3)

**Perpetual Loan:**
- Interest-only, no fixed term
- Interest rate (% annual)
- Interest payments deducted from spending budget each period
- Principal repayment trigger: fires when trailing annualized BTC return over the past `lookback` periods falls at or below `scf_rate * scf_repay_trigger`. Specifically:
  ```
  btc_annual_return = (btc_price_now / btc_price_lookback_periods_ago)^(ppy / lookback) - 1
  if btc_annual_return <= (scf_rate / 100) * scf_repay_trigger:
      sell BTC to repay outstanding principal
  ```
  - `lookback` = 12 periods for monthly frequency (1 year trailing window)
  - `scf_repay_trigger` = N multiplier (user-configurable, default 1.0 = repay when BTC return <= loan rate)
  - If insufficient BTC to cover principal: sell all BTC, remaining loan becomes debt (negative scf_outstanding tracked but not enforced further in v1)

### Mechanics
- Loan proceeds immediately buy BTC at current price, increasing BTC holdings
- Loan payments are added to the spending amount each period (increasing waterfall draw)
- For term loans: payments follow standard amortization or interest-only schedule
- For perpetual loans: interest-only payments until trigger fires, then principal repaid from BTC sale
- Tax placeholder: `tax_rate = 0.0` in config (not applied in v1, field exists for forward compatibility)

### UI
- Section within the Rules sub-tab, hidden/shown via `html.Div(style={"display":"none"})` toggle (NOT `dbc.Collapse` — Dash unmounts children in Collapse, breaking component state)
- Enable checkbox: "Enable Saylor Citadel Fortifier"
- Controls: loan amount, loan type (Term/Perpetual radio), interest rate, term (for term loans), repayment trigger N (for perpetual loans)

---

## Simulation Engine

### Architecture
**Module:** `btc_web/engines/citadel.py` — pure Python + NumPy, zero Dash dependencies.
**Adapter:** `btc_web/engines/adapter.py` — thin wrapper; v1 runs in-process, v2 swaps to Celery task submission.

The engine is designed so that the v1-to-Celery transition requires only replacing the adapter's `run_in_process()` with `celery_app.send_task()`. The engine code, UI, and caching all stay unchanged.

### SimConfig (serializable dict)
```python
@dataclass
class SimConfig:
    # BTC
    price_model: str           # "bub", "pl", "s2f"
    start_stack: float         # initial BTC
    selected_qs: list[float]   # quantiles for deterministic mode

    # Cash
    cash_initial: float        # starting $
    cash_rate: float           # annual interest %

    # Reserves (per bin)
    reserve_bins: list[dict]   # [{initial, rate, volatility}] x 3

    # Investments (per bin)
    invest_bins: list[dict]    # [{initial, return_rate, volatility}] x 2

    # Spending
    monthly_spend: float       # $/month
    inflation: float           # annual %
    spend_growth: float        # annual % above inflation

    # Rebalancing — high quantile trigger
    high_q_trigger: float      # quantile threshold (e.g., 0.80)
    high_q_action: dict        # {mode, rate, duration, split}
    # Rebalancing — low quantile trigger
    low_q_trigger: float       # quantile threshold (e.g., 0.20)
    low_q_action: dict         # {mode, rate, duration, source_split}
    lump_cooldown: int         # global cooldown in periods (applies to both high and low lump triggers)

    # Floor rules
    cash_floor: float          # minimum cash balance ($)
    reserve_floors: list[float]  # per-bin minimums [$, $, $]

    # Saylor Citadel Fortifier
    scf_enabled: bool
    scf_amount: float          # loan amount ($)
    scf_type: str              # "term" or "perpetual"
    scf_rate: float            # annual interest %
    scf_term: int              # months (term loan only)
    scf_repay_trigger: float   # N multiplier (perpetual only)

    # Simulation
    start_yr: int
    end_yr: int
    freq: str                  # "Monthly", "Quarterly", "Annually"
    n_sims: int                # 1 = free, >1 = paid
    tax_rate: float = 0.0      # placeholder for v2
```

### CitadelState (per-step mutable state)
```python
@dataclass
class CitadelState:
    t: float                   # current time (years since genesis)
    period: int                # current period index
    btc_stack: float           # BTC holdings
    btc_price: float           # current BTC price
    btc_cost_basis: float      # average cost basis (for future tax calc)
    cash: float                # cash account balance ($)
    reserves: list[float]      # per-bin balances [$, $, $]
    investments: list[float]   # per-bin balances [$, $]
    # Rebalancing state
    rebal_cooldown: int        # global cooldown: periods remaining until any lump can fire
    grad_active: bool          # is a gradual rebalance in progress?
    grad_remaining: int        # periods remaining in active gradual rebalance
    grad_rate: float           # % per period for active gradual
    grad_direction: str        # "sell_btc" or "buy_btc"
    grad_split: dict           # proceeds/source distribution (same schema as config splits)
    # Saylor Fortifier state
    scf_outstanding: float     # remaining loan principal
    scf_active: bool           # is loan currently active
    # Tracking
    period_spend: float        # actual spending this period
    spending_shortfall: float  # unmet spending (waterfall exhausted)
    rebal_event: dict | None   # rebalance action taken this period, if any
```

### SimResult (serializable output)
```python
@dataclass
class SimResult:
    time_axis: np.ndarray          # (n_periods,)
    # Per-sim histories: shape (n_sims, n_periods)
    btc_holdings: np.ndarray
    btc_prices: np.ndarray
    cash_balances: np.ndarray
    reserve_balances: np.ndarray   # (n_sims, n_periods, n_reserve_bins)
    invest_balances: np.ndarray    # (n_sims, n_periods, n_invest_bins)
    total_usd: np.ndarray          # total portfolio in USD
    cumulative_spend: np.ndarray   # running total of spending
    depletion_period: list[int | None]  # per sim: first period all assets = 0
    rebal_events: list[list[dict]] # per sim: list of rebalance event logs
    # Aggregated across sims (for chart rendering)
    median: dict                   # {asset_class: ndarray} median paths
    percentiles: dict              # {pct: {asset_class: ndarray}}

    def to_dict(self) -> dict:
        """Serialize for JSON/cache transport. ndarrays converted to lists."""
        ...

    @classmethod
    def from_dict(cls, d: dict) -> "SimResult":
        """Deserialize from JSON/cache."""
        ...
```

### Step Function
```python
def step(state: CitadelState, config: SimConfig,
         btc_price_new: float, rng: np.random.Generator) -> CitadelState:
    """Advance simulation by one period."""
    new = copy(state)
    new.period += 1
    ppy = FREQ_PPY[config.freq]

    # 1. Update BTC price
    new.btc_price = btc_price_new

    # 2. Dollar-asset returns
    #    Cash: deterministic compound
    new.cash *= (1 + config.cash_rate / 100) ** (1 / ppy)
    #    Reserves: lognormal draw per bin (deterministic if n_sims=1)
    for i, rb in enumerate(config.reserve_bins):
        if config.n_sims == 1:
            new.reserves[i] *= (1 + rb["rate"] / 100) ** (1 / ppy)
        else:
            new.reserves[i] *= _lognormal_return(
                rb["rate"] / 100, rb["volatility"] / 100, ppy, rng)
    #    Investments: same pattern
    for i, ib in enumerate(config.invest_bins):
        if config.n_sims == 1:
            new.investments[i] *= (1 + ib["return_rate"] / 100) ** (1 / ppy)
        else:
            new.investments[i] *= _lognormal_return(
                ib["return_rate"] / 100, ib["volatility"] / 100, ppy, rng)

    # 3. Enforce floor rules
    _enforce_floors(new, config)

    # 4. Compute BTC quantile
    btc_quantile = _price_to_quantile(new.btc_price, new.t, config.price_model)

    # 5. Evaluate rebalancing triggers
    _evaluate_rebalancing(new, config, btc_quantile)

    # 6. Spending
    years_elapsed = new.period / ppy
    combined_rate = (config.inflation + config.spend_growth) / 100
    period_spend = config.monthly_spend * (1 + combined_rate) ** years_elapsed
    #    Add Fortifier loan payments if active
    #    _scf_payment returns a MONTHLY amount (caller scales for frequency)
    if new.scf_active:
        period_spend += _scf_payment(new, config, ppy)
    #    Scale for frequency (monthly base, adjust for quarterly/annually)
    period_spend *= (12 / ppy)
    new.period_spend = period_spend
    new.spending_shortfall = _apply_spending_waterfall(new, period_spend)

    # 7. Depletion check
    total = (new.btc_stack * new.btc_price + new.cash
             + sum(new.reserves) + sum(new.investments))
    if total <= 0:
        # Mark depletion
        pass

    return new
```

### Top-Level Runner
```python
def simulate(config: SimConfig, model_data, rng_seed: int = 42) -> SimResult:
    """Run n_sims simulations, aggregate results."""
    validate_config(config)
    rng = np.random.default_rng(rng_seed)
    n_periods = _compute_n_periods(config)

    all_results = []
    for sim_id in range(config.n_sims):
        state = _initial_state(config)
        history = []
        for period in range(n_periods):
            btc_price = _get_btc_price(state.t, config, model_data, rng)
            state = step(state, config, btc_price, rng)
            history.append(_snapshot_state(state))
        all_results.append(history)

    return _aggregate_results(all_results, config)
```

Generator variant for future animated playback:
```python
def _run_single(config, model_data, rng):
    """Generator yielding state after each step."""
    state = _initial_state(config)
    n_periods = _compute_n_periods(config)
    for period in range(n_periods):
        btc_price = _get_btc_price(state.t, config, model_data, rng)
        state = step(state, config, btc_price, rng)
        yield state
```

### Adapter Interface
```python
# btc_web/engines/adapter.py

def submit_simulation(config: SimConfig, model_data) -> SimResult:
    """v1: run in-process. v2: submit to Celery, return job_id."""
    return simulate(config, model_data)

# v2 additions:
# def submit_async(config) -> str:  # returns job_id
# def poll_result(job_id) -> SimResult | None:  # returns result or None
```

### Input Coercion (callback layer)
All numeric inputs from the UI must use the falsy-zero-safe pattern:
```python
# WRONG: float(x or default)  — treats 0 as falsy, substitutes default
# RIGHT: float(x) if x is not None else default
```
This is critical for fields where 0 is a valid input: `cash_rate`, `inflation`, `spend_growth`, `cash_floor`, all reserve/investment floors.

### Config Validation
```python
def validate_config(config: SimConfig) -> None:
    """Raise ValueError with descriptive message on invalid config."""
    # Non-negative initial balances
    # high_q_trigger > low_q_trigger with >= 5pp gap
    # Splits sum to 1.0 (within float tolerance)
    # Date range valid (start < end)
    # Freq in {"Monthly", "Quarterly", "Annually"}
    # n_sims >= 1
    # Reserve/invest bins have required keys
    # Floors non-negative
    # SCF: term > 0 if type="term", repay_trigger > 0 if type="perpetual"
```

### BTC Quantile Inversion
Reuse existing logic from `btc_core._find_lot_percentile` and `figures/common.py:_interp_qr_price`. Extract into a standalone function:

```python
def _price_to_quantile(price: float, t: float, model_key: str) -> float:
    """Invert the quantile regression: given price and time, return quantile [0, 1].
    Interpolates between adjacent QR fit curves in log space.
    Clamps to [0.001, 0.999] at extremes."""
```

### Performance Notes
- v1: Pure Python loop, scalar state. 1 sim x 480 steps = sub-second.
- Paid tier (200 sims): expect 5-15 seconds depending on config complexity.
- Future optimization: vectorize across sims (state arrays of shape (n_sims,) instead of scalars). The step function logic is kept simple enough for this refactor.
- Frequency limited to Monthly/Quarterly/Annually for v1 (Daily creates 14,600 steps over 40 years).

---

## UI Layout

### Tab Registration
- Tab label: "🏰 Citadel Planner"
- Tab ID: `"citadel"`
- URL path: `/9`
- Component ID prefix: `cp`

### Control Panel: Tabbed Sub-Panels
Inner `dbc.Tabs` within the left control column (width=3), with 4 sub-tabs:

#### Sub-tab 1: Assets
- **BTC Stack** section: `cp-stack` (number input), `cp-use-lots` (checkbox)
- **Cash Account** section: `cp-cash-init` ($ input), `cp-cash-rate` (% input)
- **Reserve Fund** section: 3-row grid, one row per bin:
  | | Initial ($) | Return (%) | Volatility (%) |
  |---|---|---|---|
  | Short (<=1yr) | `cp-res-short-init` | `cp-res-short-rate` | `cp-res-short-vol` |
  | Medium (2-10yr) | `cp-res-med-init` | `cp-res-med-rate` | `cp-res-med-vol` |
  | Long (10-30yr) | `cp-res-long-init` | `cp-res-long-rate` | `cp-res-long-vol` |
- **Investments** section: 2-row grid:
  | | Initial ($) | Return (%) | Volatility (%) |
  |---|---|---|---|
  | Equities | `cp-inv-eq-init` | `cp-inv-eq-rate` | `cp-inv-eq-vol` |
  | Bonds | `cp-inv-bd-init` | `cp-inv-bd-rate` | `cp-inv-bd-vol` |

#### Sub-tab 2: Spending
- Monthly spending: `cp-spend` ($ input)
- Inflation rate: `cp-infl` (% input, step=0.5)
- Spending growth above inflation: `cp-spend-growth` (% input, step=0.5)

#### Sub-tab 3: Rules
- **High-Quantile Trigger** section:
  - Threshold: `cp-high-q-thresh` (% input, default 80)
  - Mode: `cp-high-q-mode` (radio: Gradual / Lump)
  - Rate: `cp-high-q-rate` (% input)
  - Duration: `cp-high-q-dur` (periods, shown only in Gradual mode)
  - Proceeds split: 6 inputs summing to 100% (Cash, Short, Med, Long, Equities, Bonds)
- **Low-Quantile Trigger** section: same structure as high-Q, labels changed to "source split"
- **Global Lump Cooldown**: `cp-lump-cooldown` (periods input, default 12, shown when either trigger uses Lump mode) — shared between high and low triggers to prevent whipsawing
- **Floor Rules** section:
  - `cp-cash-floor` ($ input, default 0)
  - `cp-res-short-floor`, `cp-res-med-floor`, `cp-res-long-floor` ($ inputs, default 0)
- **Saylor Citadel Fortifier** section (collapsible):
  - Enable: `cp-scf-enable` (checkbox: "Enable Saylor Citadel Fortifier")
  - Amount: `cp-scf-amount` ($ input)
  - Type: `cp-scf-type` (radio: Term / Perpetual)
  - Rate: `cp-scf-rate` (% input)
  - Term: `cp-scf-term` (months, shown only for Term type)
  - Repayment trigger: `cp-scf-trigger` (N multiplier, shown only for Perpetual type)

#### Sub-tab 4: Simulation
- Year range: `cp-yr-range` (range slider, min=2025, max=2080)
- Frequency: `cp-freq` (dropdown: Monthly / Quarterly / Annually)
- Price model: `cp-model-src` (dropdown, shared pattern)
- Quantiles: `cp-qs` (checklist, shared `_q_panel`)
- MC toggle: standard `_mc_controls("cp")` — n_sims, regime, etc.
- Display: `cp-disp` (dropdown: USD Total / USD Per-Asset / BTC Holdings)
- Chart toggles: `cp-toggles` (Log Y, Annotate, Show legend, Minor grid, Zoom)
- Legend position: `cp-legend-pos` (dropdown)

### Default Values
| Control | Default | Rationale |
|---------|---------|-----------|
| BTC stack | 1.0 | Same as retire tab |
| Cash initial | $50,000 | 1 year emergency fund |
| Cash rate | 4.0% | Current HYSA rates |
| Reserve Short | $50,000, 5.0%, 2.0% | T-Bill level |
| Reserve Medium | $100,000, 4.5%, 8.0% | T-Note level |
| Reserve Long | $50,000, 4.0%, 15.0% | T-Bond level |
| Equities | $200,000, 10.0%, 16.0% | S&P 500 long-term average |
| Bonds | $100,000, 5.0%, 7.0% | Aggregate bond index |
| Monthly spend | $5,000 | |
| Inflation | 4.0% | Consistent with other tabs |
| Spend growth | 0.0% | Conservative default |
| High-Q trigger | 80%, Gradual, 2%/mo, 6 months | |
| Low-Q trigger | 20%, Lump, 10%, 12-period cooldown | |
| All floors | $0 | Off by default |
| SCF | Disabled | Off by default |
| Year range | 2031-2075 | Same as retire tab |
| Frequency | Monthly | |
| Quantiles | Q1%, Q10%, Q25% | Same as retire tab |

---

## Chart Output

### v1: Multi-Line Chart
One line per asset class, following existing `_sim_layout` pattern:

| Line | Color | Description |
|------|-------|-------------|
| Total Portfolio (USD) | White (#e0e0e0), width=2.5 | Sum of all assets in USD |
| BTC Holdings (USD) | BTC Orange (#f7931a), width=2 | btc_stack x btc_price |
| Cash | Silver (#bdc3c7), width=1.5, dashed | Cash balance |
| Reserve Fund | Blue (#3498db), width=1.5 | Sum across reserve bins |
| Investments | Green (#2ecc71), width=1.5 | Sum across investment bins |
| Monthly Spending | Red (#e74c3c), width=1, dotted | Inflation+growth adjusted spending |

### Depletion Annotation
Arrow to y=0 with year label when total portfolio (including BTC) hits zero. Same `_depl_annot` pattern as supercharger tab.

### MC Mode (n_sims > 1)
Each asset line gets fan bands: median solid line + shaded 5th-95th and 25th-75th percentile regions. Same visual pattern as existing MC overlays on tabs 3-5.

### Endpoint Annotations
Final value labels at right edge using `_edge_text_trace` pattern. Show total portfolio value and BTC holdings.

### v2/v3 planned chart extensions:
- User-configurable viewports (stacked area, multi-line, single line with hover)
- Animated time-slider playback (drag slider to advance simulation frame by frame)
- Per-bin breakdown views (individual reserve/investment bins)
- Rebalancing event markers (vertical lines at rebalance periods)

---

## Snapshot / Share Integration

### New `_SNAPSHOT_CONTROLS` Entries
~62 new entries appended after index 136 (current end), plus MC controls. Component IDs use `cp-` prefix. The authoritative list is the `_TAB_CONTROLS["citadel"]` set below.

The existing backward-compat mechanism handles this: old links pad with `None` (Tab 9 controls at defaults), new links on old code truncate (Tab 9 controls lost, other tabs work).

### New `_CHECKLIST_OPTIONS` Entries
```python
"cp-qs":         _QS_LIST,          # quantile checklist
"cp-toggles":    ["log_y", "annotate", "show_legend", "minor_grid", "chart_zoom"],
"cp-use-lots":   ["yes"],
"cp-scf-enable": ["yes"],
```

### Hybrid MC Encoding
Same pattern as existing tabs: when MC is disabled on Tab 9, MC controls encode as `null` in the snapshot.

### `_TAB_CONTROLS` Entry
```python
_TAB_CONTROLS["citadel"] = {
    "cp-stack", "cp-use-lots", "cp-cash-init", "cp-cash-rate",
    "cp-res-short-init", "cp-res-short-rate", "cp-res-short-vol",
    "cp-res-med-init", "cp-res-med-rate", "cp-res-med-vol",
    "cp-res-long-init", "cp-res-long-rate", "cp-res-long-vol",
    "cp-inv-eq-init", "cp-inv-eq-rate", "cp-inv-eq-vol",
    "cp-inv-bd-init", "cp-inv-bd-rate", "cp-inv-bd-vol",
    "cp-spend", "cp-infl", "cp-spend-growth",
    "cp-high-q-thresh", "cp-high-q-mode", "cp-high-q-rate", "cp-high-q-dur",
    "cp-high-q-split-cash", "cp-high-q-split-rs", "cp-high-q-split-rm",
    "cp-high-q-split-rl", "cp-high-q-split-eq", "cp-high-q-split-bd",
    "cp-low-q-thresh", "cp-low-q-mode", "cp-low-q-rate", "cp-low-q-dur",
    "cp-low-q-split-cash", "cp-low-q-split-rs", "cp-low-q-split-rm",
    "cp-low-q-split-rl", "cp-low-q-split-eq", "cp-low-q-split-bd",
    "cp-lump-cooldown",
    "cp-cash-floor", "cp-res-short-floor", "cp-res-med-floor", "cp-res-long-floor",
    "cp-scf-enable", "cp-scf-amount", "cp-scf-type", "cp-scf-rate",
    "cp-scf-term", "cp-scf-trigger",
    "cp-yr-range", "cp-freq", "cp-qs", "cp-toggles",
    "cp-model-src", "cp-disp", "cp-legend-pos",
    # MC controls added when MC is implemented
}
```

### URL Routing
```python
_PATH_TO_TAB["/9"] = "citadel"
```

---

## File Structure

### New Files
| File | Purpose |
|------|---------|
| `btc_web/engines/__init__.py` | Engine package init |
| `btc_web/engines/citadel.py` | Simulation engine (pure Python + NumPy) |
| `btc_web/engines/adapter.py` | Submission adapter (in-process v1, Celery v2) |
| `btc_web/layout/citadel.py` | Tab 9 layout — tabbed sub-panels, all controls |
| `btc_web/figures/citadel.py` | Chart builder — multi-line portfolio chart |
| `btc_web/callbacks/citadel_cb.py` | Tab 9 callback — assembles params, calls engine, builds chart |

### Modified Files
| File | Change |
|------|--------|
| `btc_web/layout/__init__.py` | Import `_citadel_tab`, register in `dbc.Tabs`, add `dcc.Store` for Tab 9 |
| `btc_web/callbacks/__init__.py` | Import `citadel_cb` |
| `btc_web/snapshot.py` | Append ~62 entries to `_SNAPSHOT_CONTROLS`, add `_CHECKLIST_OPTIONS` |
| `btc_web/callbacks/nav.py` | Add `/9` -> `citadel` routing, `_TAB_CONTROLS["citadel"]` |
| `btc_web/utils.py` | Add `_cached_citadel_fig`, `_get_citadel_fig` |
| `btc_web/app.py` | Add prewarm call for Tab 9 defaults |

### Testing (separate implementation phase)
| Test Area | What to Test |
|-----------|-------------|
| Engine unit tests | `step()` with known inputs, verify state transitions |
| Config validation | Invalid configs rejected with clear messages |
| Spending waterfall | Correct draw order, partial drains, cascading |
| Rebalancing logic | Trigger evaluation, gradual/lump execution, cooldown |
| Floor enforcement | Replenishment logic, no BTC liquidation for floors |
| Depletion detection | All-asset-zero correctly identified |
| Fortifier mechanics | Term/perpetual payment schedules, repayment trigger |
| SimResult serialization | `to_dict()` / `from_dict()` roundtrip |
| Snapshot integration | New controls roundtrip through encode/decode |
| Callback smoke tests | Callback returns valid figure with default params |
| Duplicate output guard | Existing `TestNoDuplicateCallbackOutputs` catches regressions |

---

## MC / Payment Integration

### Free Tier
- `n_sims = 1`: deterministic simulation, runs in-process, instant result
- Uses expected returns for all dollar assets (no volatility)
- BTC price follows selected quantile path

### Paid Tier
- `n_sims > 1`: stochastic simulation with MC fan bands
- Uses existing BTCPay Lightning/on-chain payment flow
- **Architecture difference from tabs 3-5:** The existing MC tabs use a pre-computed Markov cache for BTC prices, then overlay fan bands on top of the deterministic figure. Tab 9 cannot use this pattern because rebalancing modifies BTC holdings (and thus the portfolio) based on price — the BTC price path and portfolio state are coupled. Instead:
  - Tab 9's engine runs the *entire* simulation (BTC prices + dollar assets + rebalancing + spending) N times internally
  - BTC price transitions use the same Markov transition matrix as the existing MC system, drawn via `markov.sample_path()` or equivalent
  - The engine produces aggregated fan bands (median + percentiles) across all sims
  - The callback uses `_mc_setup` for payment validation and `_mc_finalize` for result caching/UI, but the actual simulation is delegated to `engines/citadel.simulate()` rather than `mc_overlay._mc_withdraw_overlay()`
- Results cached in `dcc.Store("cp-mc-results")`
- MC controls: standard `_mc_controls("cp")` panel in Simulation sub-tab

### Cache Strategy
- Deterministic (1 sim): LRU cache via `_cached_citadel_fig` (same as other tabs)
- MC results: stored in memory Store, survives within session
- Future: shared-memory cache for popular configurations (same pattern as existing MC cache system)

---

## Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| Old link (137 entries) on new code (192+) | Pad with `None` -> Tab 9 at defaults |
| New link (192+ entries) on old code (137) | Truncate -> Tab 9 controls lost, tabs 1-8 work |
| New link, Tab 9 scope, MC disabled | MC controls encode as `null` -> minimal URL growth |

---

## Sub-Project B: App-Wide Save/Load (designed together, implemented separately)

### Scope
Full save/load system for the entire app — all 9 tabs' settings + output data.

### File Format
- JSON-based, versioned
- Version 0.1 (development), version 1.0 (stable, backward-compat guaranteed after 1.0)
- Includes: all control values across all tabs, lots data, MC results references, Tab 9 config
- Excludes: cached figures (too large), session-only state

### Ctrl+S Override
- Clientside JS intercepts `Ctrl+S` / `Cmd+S`
- Triggers Dash callback that assembles state -> generates JSON -> browser download

### Backward Compatibility
- Reader must handle all versions >= 1.0
- Unknown fields ignored (forward compat)
- Missing fields filled with defaults (backward compat)
- Version field in file header: `{"quantoshi_version": "0.1", ...}`

### Implementation: separate spec + plan after Tab 9 ships.

---

## Not In Scope (v1)

- Stochastic interest rate simulation (Markov chain on yield curve)
- Equity/bond return triggers and ratio triggers
- More than 3 reserve bins or 2 investment bins
- Adaptive spending cuts
- Animated time-slider playback
- User-configurable chart viewports (stacked area, hover breakdown)
- Daily frequency
- Tax calculation on BTC sales
- Celery/Redis async worker
- App-wide save/load file system (Sub-project B)
- Multiple spending phases
- Gradual rebalance override/cancellation rules
