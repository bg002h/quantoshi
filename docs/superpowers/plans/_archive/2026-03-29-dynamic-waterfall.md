# Dynamic Cost-Ranked Waterfall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed-sequence spending waterfall with a dynamic cost-ranked waterfall that draws from the cheapest source first, re-ranking at tax bracket boundaries.

**Architecture:** Define a `_WithdrawalSource` dataclass representing each drawable account. Build the source list from state, score each source with `tax_cost + opportunity_cost`, rank (non-Roth by cost, then Roth by cost), draw from cheapest with bracket-boundary caps, re-rank after each draw. Six new helper functions replace the existing `_spending_waterfall` body.

**Tech Stack:** Python 3.14, dataclasses, numpy

**Spec:** `docs/superpowers/specs/2026-03-29-dynamic-waterfall-design.md`

**Test command:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short`

---

### Task 1: `_WithdrawalSource` dataclass + `_build_source_list`

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write tests**

Add a new test class `TestDynamicWaterfall` at the end of `btc_web/test_web.py`:

```python
class TestDynamicWaterfall:
    """Tests for the dynamic cost-ranked spending waterfall."""

    def test_build_source_list_taxable_only(self):
        """Non-tax mode produces only taxable sources."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list)
        state = CitadelState(
            cash=10_000, reserves=[20_000, 30_000, 5_000],
            investments=[100_000, 50_000], invest_cost_basis=[60_000, 30_000],
            btc_stack=1.0, btc_price=50_000, sim_date="2035-06-15",
        )
        cfg = SimConfig(tax_enabled=False, cash_rate=4.0,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        # 7 taxable sources: cash + 3 reserves + 2 investments + BTC
        assert len(sources) == 7
        assert all(not s.is_roth for s in sources)
        # Check available balances
        cash_src = [s for s in sources if s.key == "cash"][0]
        assert cash_src.available == pytest.approx(10_000)
        btc_src = [s for s in sources if s.key == "btc"][0]
        assert btc_src.available == pytest.approx(50_000)

    def test_build_source_list_with_tax(self):
        """Tax mode produces taxable + TD + TF sources."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=10_000, reserves=[0, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=1.0, btc_price=50_000, sim_date="2035-06-15",
            td_cash=20_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            td_btc_stack=0.5,
            tf_cash=10_000, tf_reserves=[0, 0, 0], tf_investments=[0, 0],
            tf_btc_stack=0.3,
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, cash_rate=4.0,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        # Should have taxable + TD + TF sources
        wrappers = set(s.wrapper for s in sources)
        assert "taxable" in wrappers
        assert "td" in wrappers
        assert "tf" in wrappers
        # TF sources should be marked is_roth
        tf_sources = [s for s in sources if s.wrapper == "tf"]
        assert all(s.is_roth for s in tf_sources)
        # Only include sources with available > 0
        assert all(s.available > 0.01 for s in sources)

    def test_source_gain_fraction(self):
        """Gain fraction computed correctly for investments and BTC."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list)
        from engines.tax_lots import TaxLot
        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[200_000, 0], invest_cost_basis=[100_000, 0],
            btc_stack=2.0, btc_price=80_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=2.0,
                             cost_basis=30_000, source="initial")],
        )
        cfg = SimConfig(tax_enabled=False,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        eq_src = [s for s in sources if s.key == "invest_0"][0]
        # Gain fraction: 1 - (100k / 200k) = 0.5
        assert eq_src.gain_fraction == pytest.approx(0.5)
        btc_src = [s for s in sources if s.key == "btc"][0]
        # BTC gain fraction: 1 - (30k / 80k) = 0.625
        assert btc_src.gain_fraction == pytest.approx(0.625)
```

- [ ] **Step 2: Implement `_WithdrawalSource` dataclass**

Add to `btc_web/engines/citadel.py`, after the `_SATOSHI` constant and before the tracking helpers:

```python
@dataclass
class _WithdrawalSource:
    """Represents one drawable account for the cost-ranked waterfall."""
    key: str              # e.g., "cash", "reserve_0", "invest_1", "btc", "td_cash", "tf_btc"
    wrapper: str          # "taxable", "td", or "tf"
    asset_type: str       # "cash", "reserve", "invest", "btc"
    index: int            # bin index (0-2 for reserves, 0-1 for investments, 0 for cash/btc)
    available: float      # current dollar balance available to draw
    growth_rate: float    # annual growth rate for opportunity cost
    horizon: int          # opportunity cost horizon in years
    gain_fraction: float  # for investments/BTC: 1 - (basis/value). 0 for cash/reserves
    is_roth: bool         # True for TF sources — forced last
    is_bracket_sensitive: bool  # True if draw affects tax bracket position
    bracket_type: str     # "ordinary", "ltcg", or "none"
    cost: float = 0.0     # computed by _score_sources
```

- [ ] **Step 3: Implement `_build_source_list`**

```python
def _build_source_list(state: CitadelState, config: SimConfig,
                       model: "PriceModel | None" = None) -> list[_WithdrawalSource]:
    """Enumerate all available withdrawal sources from current state."""
    sources = []
    ppy = FREQ_PPY.get(config.freq, 12)

    # Compute BTC opportunity cost horizon and growth
    _btc_growth = config.invest_bins[0]["return_rate"] / 100 if config.invest_bins else 0.10
    if model is not None and state.btc_price > 0:
        try:
            _q = config.selected_qs[len(config.selected_qs) // 2] if config.selected_qs else 0.25
            _p_now = float(model.price_at(_q, max(state.t, 0.5)))
            _p_fwd = float(model.price_at(_q, max(state.t + 10, 0.5)))
            if _p_now > 0:
                _btc_growth = (_p_fwd / _p_now) - 1
        except Exception:
            pass

    # Treasury horizon: remaining lifetime
    if config.birth_year:
        _age = config.start_yr + int(state.period / ppy) - config.birth_year
    else:
        _age = 0
    _tres_horizon = max(min(90 - _age, 40), 1)

    # BTC gain fraction (lot-weighted average)
    _btc_gain_frac = 0.0
    if state.btc_stack > 0 and state.btc_price > 0:
        btc_value = state.btc_stack * state.btc_price
        lot_basis_total = sum(l.btc * l.cost_basis for l in state.tax_lots) if state.tax_lots else 0
        if btc_value > 0:
            _btc_gain_frac = max(1.0 - lot_basis_total / btc_value, 0.0)

    # --- Taxable sources ---
    if state.cash > 0.01:
        sources.append(_WithdrawalSource(
            key="cash", wrapper="taxable", asset_type="cash", index=0,
            available=state.cash, growth_rate=config.cash_rate / 100,
            horizon=15, gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=False, bracket_type="none",
        ))
    for i, rb in enumerate(config.reserve_bins):
        bal = state.reserves[i] if i < len(state.reserves) else 0
        if bal > 0.01:
            sources.append(_WithdrawalSource(
                key=f"reserve_{i}", wrapper="taxable", asset_type="reserve", index=i,
                available=bal, growth_rate=rb["rate"] / 100,
                horizon=_tres_horizon, gain_fraction=0.0, is_roth=False,
                is_bracket_sensitive=False, bracket_type="none",
            ))
    for i, ib in enumerate(config.invest_bins):
        bal = state.investments[i] if i < len(state.investments) else 0
        if bal > 0.01:
            basis = state.invest_cost_basis[i] if i < len(state.invest_cost_basis) else bal
            gf = max(1.0 - basis / bal, 0.0) if bal > 0 else 0.0
            sources.append(_WithdrawalSource(
                key=f"invest_{i}", wrapper="taxable", asset_type="invest", index=i,
                available=bal, growth_rate=ib.get("return_rate", ib.get("rate", 5.0)) / 100,
                horizon=15, gain_fraction=gf, is_roth=False,
                is_bracket_sensitive=True, bracket_type="ltcg",
            ))
    if state.btc_stack > 0.01 and state.btc_price > 0:
        sources.append(_WithdrawalSource(
            key="btc", wrapper="taxable", asset_type="btc", index=0,
            available=state.btc_stack * state.btc_price,
            growth_rate=_btc_growth if isinstance(_btc_growth, float) else 0.10,
            horizon=10, gain_fraction=_btc_gain_frac, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ltcg",
        ))

    # --- TD sources (tax_enabled only) ---
    if config.tax_enabled:
        if state.td_cash > 0.01:
            sources.append(_WithdrawalSource(
                key="td_cash", wrapper="td", asset_type="cash", index=0,
                available=state.td_cash, growth_rate=config.cash_rate / 100,
                horizon=15, gain_fraction=0.0, is_roth=False,
                is_bracket_sensitive=True, bracket_type="ordinary",
            ))
        for i, rb in enumerate(config.reserve_bins):
            bal = state.td_reserves[i] if i < len(state.td_reserves) else 0
            if bal > 0.01:
                sources.append(_WithdrawalSource(
                    key=f"td_reserve_{i}", wrapper="td", asset_type="reserve", index=i,
                    available=bal, growth_rate=rb["rate"] / 100,
                    horizon=_tres_horizon, gain_fraction=0.0, is_roth=False,
                    is_bracket_sensitive=True, bracket_type="ordinary",
                ))
        for i, ib in enumerate(config.invest_bins):
            bal = state.td_investments[i] if i < len(state.td_investments) else 0
            if bal > 0.01:
                sources.append(_WithdrawalSource(
                    key=f"td_invest_{i}", wrapper="td", asset_type="invest", index=i,
                    available=bal, growth_rate=ib.get("return_rate", ib.get("rate", 5.0)) / 100,
                    horizon=15, gain_fraction=0.0, is_roth=False,
                    is_bracket_sensitive=True, bracket_type="ordinary",
                ))
        if state.td_btc_stack > 0.01 and state.btc_price > 0:
            sources.append(_WithdrawalSource(
                key="td_btc", wrapper="td", asset_type="btc", index=0,
                available=state.td_btc_stack * state.btc_price,
                growth_rate=_btc_growth if isinstance(_btc_growth, float) else 0.10,
                horizon=10, gain_fraction=0.0, is_roth=False,
                is_bracket_sensitive=True, bracket_type="ordinary",
            ))

    # --- TF (Roth) sources (tax_enabled only) ---
    if config.tax_enabled:
        tf_cash_res = state.tf_cash + sum(state.tf_reserves)
        if tf_cash_res > 0.01:
            sources.append(_WithdrawalSource(
                key="tf_cash_res", wrapper="tf", asset_type="cash", index=0,
                available=tf_cash_res, growth_rate=config.cash_rate / 100,
                horizon=15, gain_fraction=0.0, is_roth=True,
                is_bracket_sensitive=False, bracket_type="none",
            ))
        tf_inv = sum(state.tf_investments)
        if tf_inv > 0.01:
            avg_rate = sum(ib.get("return_rate", 5.0) for ib in config.invest_bins) / max(len(config.invest_bins), 1)
            sources.append(_WithdrawalSource(
                key="tf_invest", wrapper="tf", asset_type="invest", index=0,
                available=tf_inv, growth_rate=avg_rate / 100,
                horizon=15, gain_fraction=0.0, is_roth=True,
                is_bracket_sensitive=False, bracket_type="none",
            ))
        if state.tf_btc_stack > 0.01 and state.btc_price > 0:
            sources.append(_WithdrawalSource(
                key="tf_btc", wrapper="tf", asset_type="btc", index=0,
                available=state.tf_btc_stack * state.btc_price,
                growth_rate=_btc_growth if isinstance(_btc_growth, float) else 0.10,
                horizon=10, gain_fraction=0.0, is_roth=True,
                is_bracket_sensitive=False, bracket_type="none",
            ))

    return sources
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestDynamicWaterfall -v
```

- [ ] **Step 5: Run full suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -15
```

- [ ] **Step 6: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "feat(citadel): add _WithdrawalSource dataclass and _build_source_list"
```

---

### Task 2: `_score_sources` (cost function)

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write tests**

Add to `TestDynamicWaterfall`:

```python
    def test_score_taxable_cash_zero_tax(self):
        """Taxable cash has zero tax cost, only opportunity cost."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        state = CitadelState(
            cash=50_000, reserves=[0, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=0, btc_price=50_000, sim_date="2035-06-15",
        )
        cfg = SimConfig(tax_enabled=False, cash_rate=4.0,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        _score_sources(sources, state, cfg, model=None)
        cash = [s for s in sources if s.key == "cash"][0]
        # Tax cost = 0, opportunity = (1.04)^15 - 1 ≈ 0.80
        assert cash.cost == pytest.approx((1.04 ** 15) - 1, rel=0.01)

    def test_score_td_ordinary_rate(self):
        """TD source tax cost = marginal ordinary rate + state rate."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=50_000, reserves=[0, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=0, btc_price=50_000, sim_date="2035-06-15",
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX", cash_rate=4.0,
                        filing_status="single", inflation=4.0,
                        start_yr=2031,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        _score_sources(sources, state, cfg, model=None)
        td = [s for s in sources if s.key == "td_cash"][0]
        # At $0 YTD ordinary income, marginal rate = 10% (first bracket)
        # TX = 0% state. Tax cost = 0.10. Opportunity = (1.04^15 - 1) × (1-0.10)
        assert td.cost > 0.10  # tax cost alone
        cash_src = [s for s in sources if s.key == "cash"][0]
        assert td.cost > cash_src.cost  # TD more expensive than cash (tax + opp)

    def test_score_niit_above_threshold(self):
        """NIIT adds 3.8% to capital gains sources when MAGI > threshold."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[200_000, 0], invest_cost_basis=[100_000, 0],
            btc_stack=0, btc_price=50_000, sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(other_income=250_000),  # above NIIT
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX", filing_status="single",
                        inflation=4.0, start_yr=2031,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        _score_sources(sources, state, cfg, model=None)
        eq = [s for s in sources if s.key == "invest_0"][0]
        # LTCG rate (15%) + NIIT (3.8%) + state (0%) = 18.8% × gain_fraction (0.5) = 9.4% tax
        # Plus opportunity cost
        assert eq.cost > 0.094  # tax component alone

    def test_score_btc_high_early_low_late(self):
        """BTC opportunity cost is higher in 2035 than 2065."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources,
                                      _WithdrawalSource)
        from engines.tax_lots import TaxLot
        from btc_core import yr_to_t

        def _btc_cost_at_year(yr):
            t = yr_to_t(yr)
            state = CitadelState(
                cash=0, reserves=[0, 0, 0],
                investments=[0, 0], invest_cost_basis=[0, 0],
                btc_stack=1.0, btc_price=50_000, t=t, sim_date=f"{yr}-06-15",
                tax_lots=[TaxLot(date="2031-01-01", btc=1.0,
                                 cost_basis=10_000, source="initial")],
            )
            cfg = SimConfig(tax_enabled=False, start_yr=yr,
                            reserve_bins=[
                                {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                                {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                                {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                            ],
                            invest_bins=[
                                {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                                {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                            ])
            sources = _build_source_list(state, cfg, model=_test_model())
            _score_sources(sources, state, cfg, model=_test_model())
            btc = [s for s in sources if s.key == "btc"][0]
            return btc.cost

        cost_2035 = _btc_cost_at_year(2035)
        cost_2065 = _btc_cost_at_year(2065)
        assert cost_2035 > cost_2065, "BTC cost should decrease as growth slows"
```

- [ ] **Step 2: Implement `_score_sources`**

Add to `btc_web/engines/citadel.py`:

```python
def _score_sources(sources: list[_WithdrawalSource], state: CitadelState,
                   config: SimConfig, model: "PriceModel | None" = None) -> None:
    """Compute cost-per-dollar for each source. Mutates source.cost in place."""
    from .tax import _inflate_brackets
    from .tax_data import (FEDERAL_BRACKETS_TCJA, FEDERAL_BRACKETS_SUNSET,
                           LTCG_BRACKETS, NIIT_RATE, NIIT_THRESHOLD,
                           STANDARD_DEDUCTION_TCJA, STANDARD_DEDUCTION_SUNSET)

    state_rate = _get_state_rate(config) / 100  # as fraction

    # Current bracket position from accumulator
    _years_from_base = 0
    _ordinary_ytd = 0.0
    _ltcg_ytd = 0.0
    _magi = 0.0
    if state.tax_year_accum is not None:
        a = state.tax_year_accum
        _ordinary_ytd = (a.tax_deferred_withdrawals + a.interest_income
                         + a.treasury_interest + a.other_income)
        _ltcg_ytd = max(a.lt_capital_gains - a.lt_capital_losses, 0)
        _magi = _ordinary_ytd + _ltcg_ytd + max(a.st_capital_gains - a.st_capital_losses, 0)

    ppy = FREQ_PPY.get(config.freq, 12)
    sim_year = config.start_yr + int(state.period / ppy)
    _years_from_base = max(sim_year - 2025, 0)
    infl = config.inflation / 100

    # Inflate brackets
    if config.tcja_sunset:
        _ord_brackets = _inflate_brackets(FEDERAL_BRACKETS_SUNSET[config.filing_status], _years_from_base, infl)
        _std_ded = STANDARD_DEDUCTION_SUNSET[config.filing_status] * (1 + infl) ** _years_from_base
    else:
        _ord_brackets = _inflate_brackets(FEDERAL_BRACKETS_TCJA[config.filing_status], _years_from_base, infl)
        _std_ded = STANDARD_DEDUCTION_TCJA[config.filing_status] * (1 + infl) ** _years_from_base
    _ltcg_brackets = _inflate_brackets(LTCG_BRACKETS[config.filing_status], _years_from_base, infl)
    _niit_threshold = NIIT_THRESHOLD[config.filing_status]  # NOT inflation-indexed

    # Marginal ordinary rate at current YTD position
    _ord_taxable = max(_ordinary_ytd - _std_ded, 0)
    _marginal_ord = 0.10  # default
    for upper, rate in _ord_brackets:
        if _ord_taxable < upper:
            _marginal_ord = rate
            break

    # LTCG rate at stacked position (ordinary + LTCG)
    _stacked = _ord_taxable + _ltcg_ytd
    _marginal_ltcg = 0.15  # default
    for upper, rate in _ltcg_brackets:
        if _stacked < upper:
            _marginal_ltcg = rate
            break

    # NIIT applies?
    _niit = NIIT_RATE if _magi > _niit_threshold else 0.0

    for s in sources:
        # Tax cost per dollar
        if s.wrapper == "tf":
            tax_cost = 0.0  # Roth — no tax
        elif s.wrapper == "td":
            tax_cost = _marginal_ord + state_rate
        elif s.asset_type in ("invest", "btc"):
            tax_cost = (_marginal_ltcg + _niit + state_rate) * s.gain_fraction
        else:
            tax_cost = 0.0  # taxable cash/reserves — principal

        # Opportunity cost
        if s.wrapper == "td":
            # TD grows gross, taxed on withdrawal → reduce by (1 - marginal_rate)
            opp = ((1 + s.growth_rate) ** s.horizon - 1) * (1 - _marginal_ord)
        elif s.wrapper == "taxable" and s.asset_type == "reserve":
            # Taxable treasury: after-tax interest compounding
            # Treasury interest is state-exempt (US law) — only federal tax on coupons
            after_tax_rate = s.growth_rate * (1 - _marginal_ord)
            opp = (1 + max(after_tax_rate, 0)) ** s.horizon - 1
        else:
            opp = (1 + s.growth_rate) ** s.horizon - 1

        s.cost = tax_cost + max(opp, 0.0)
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestDynamicWaterfall -v
```

- [ ] **Step 4: Run full suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -15
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "feat(citadel): add _score_sources cost function with tax + opportunity cost"
```

---

### Task 3: `_max_draw_before_boundary` + `_rank_sources`

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write tests**

Add to `TestDynamicWaterfall`:

```python
    def test_rank_roth_always_last(self):
        """Roth sources rank after all non-Roth regardless of cost."""
        from engines.citadel import _WithdrawalSource, _rank_sources
        sources = [
            _WithdrawalSource(key="btc", wrapper="taxable", asset_type="btc",
                              index=0, available=50_000, growth_rate=0.5,
                              horizon=10, gain_fraction=0.9, is_roth=False,
                              is_bracket_sensitive=True, bracket_type="ltcg", cost=5.0),
            _WithdrawalSource(key="tf_cash_res", wrapper="tf", asset_type="cash",
                              index=0, available=10_000, growth_rate=0.04,
                              horizon=15, gain_fraction=0.0, is_roth=True,
                              is_bracket_sensitive=False, bracket_type="none", cost=0.01),
        ]
        ranked = _rank_sources(sources)
        # TF cash has lower cost (0.01) but must rank after taxable BTC (5.0)
        assert ranked[0].key == "btc"
        assert ranked[1].key == "tf_cash_res"

    def test_max_draw_ordinary_bracket(self):
        """Distance to next ordinary bracket computed correctly."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _max_draw_before_boundary)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(other_income=10_000),
        )
        cfg = SimConfig(tax_enabled=True, filing_status="single",
                        inflation=4.0, start_yr=2031, freq="Monthly")
        td_source = _WithdrawalSource(
            key="td_cash", wrapper="td", asset_type="cash", index=0,
            available=100_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ordinary",
        )
        max_draw = _max_draw_before_boundary(state, cfg, td_source)
        # 10k ordinary income, first bracket top ~11,925 × inflation^10 ≈ ~17,651
        # Distance ≈ 7,651 (approximate due to inflation)
        assert max_draw > 0
        assert max_draw < 100_000  # capped at bracket boundary

    def test_max_draw_niit_cliff(self):
        """Draw capped at NIIT threshold when MAGI is below it."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _max_draw_before_boundary)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(other_income=190_000),
        )
        cfg = SimConfig(tax_enabled=True, filing_status="single",
                        inflation=4.0, start_yr=2031, freq="Monthly")
        td_source = _WithdrawalSource(
            key="td_cash", wrapper="td", asset_type="cash", index=0,
            available=100_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ordinary",
        )
        max_draw = _max_draw_before_boundary(state, cfg, td_source)
        # MAGI at 190k, NIIT threshold 200k (NOT inflated) → distance = 10k
        # Should be capped at 10k (or less if ordinary bracket is closer)
        assert max_draw <= 10_001

    def test_zero_bracket_distance_skips(self):
        """When at exact bracket boundary, max_draw returns ~0."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _max_draw_before_boundary)
        from engines.tax import TaxYearAccumulator
        # Set income exactly at an inflated bracket boundary
        from engines.tax import _inflate_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        brackets = _inflate_brackets(FEDERAL_BRACKETS_TCJA["single"], 10, 0.04)
        boundary = brackets[0][0]  # first bracket top, inflated
        state = CitadelState(
            sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(other_income=boundary),
        )
        cfg = SimConfig(tax_enabled=True, filing_status="single",
                        inflation=4.0, start_yr=2031, freq="Monthly")
        td_source = _WithdrawalSource(
            key="td_cash", wrapper="td", asset_type="cash", index=0,
            available=100_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ordinary",
        )
        max_draw = _max_draw_before_boundary(state, cfg, td_source)
        # At exact boundary → distance to NEXT bracket should be > 0
        # (we're AT the first bracket top, next bracket starts there)
        # Actually, _already_ordinary = boundary, which sits exactly at brackets[0][0]
        # The next threshold is brackets[1][0]. Distance = brackets[1][0] - boundary > 0
        assert max_draw > 0
```

- [ ] **Step 2: Implement `_rank_sources`**

```python
def _rank_sources(sources: list[_WithdrawalSource]) -> list[_WithdrawalSource]:
    """Sort sources: non-Roth by cost ascending, then Roth by cost ascending."""
    non_roth = sorted([s for s in sources if not s.is_roth], key=lambda s: s.cost)
    roth = sorted([s for s in sources if s.is_roth], key=lambda s: s.cost)
    return non_roth + roth
```

- [ ] **Step 3: Implement `_max_draw_before_boundary`**

```python
def _max_draw_before_boundary(state: CitadelState, config: SimConfig,
                               source: _WithdrawalSource) -> float:
    """Max dollars drawable before crossing a tax bracket boundary.
    Returns float("inf") for non-bracket-sensitive sources.
    """
    if not source.is_bracket_sensitive:
        return float("inf")

    from .tax import _inflate_brackets
    from .tax_data import (FEDERAL_BRACKETS_TCJA, FEDERAL_BRACKETS_SUNSET,
                           LTCG_BRACKETS, NIIT_THRESHOLD,
                           STANDARD_DEDUCTION_TCJA, STANDARD_DEDUCTION_SUNSET)

    ppy = FREQ_PPY.get(config.freq, 12)
    sim_year = config.start_yr + int(state.period / ppy)
    yrs = max(sim_year - 2025, 0)
    infl = config.inflation / 100

    if config.tcja_sunset:
        ord_brackets = _inflate_brackets(FEDERAL_BRACKETS_SUNSET[config.filing_status], yrs, infl)
        std_ded = STANDARD_DEDUCTION_SUNSET[config.filing_status] * (1 + infl) ** yrs
    else:
        ord_brackets = _inflate_brackets(FEDERAL_BRACKETS_TCJA[config.filing_status], yrs, infl)
        std_ded = STANDARD_DEDUCTION_TCJA[config.filing_status] * (1 + infl) ** yrs

    # Current positions from accumulator
    ordinary_ytd = 0.0
    ltcg_ytd = 0.0
    magi = 0.0
    if state.tax_year_accum is not None:
        a = state.tax_year_accum
        ordinary_ytd = (a.tax_deferred_withdrawals + a.interest_income
                        + a.treasury_interest + a.other_income)
        ltcg_ytd = max(a.lt_capital_gains - a.lt_capital_losses, 0)
        stcg_ytd = max(a.st_capital_gains - a.st_capital_losses, 0)
        magi = ordinary_ytd + ltcg_ytd + stcg_ytd

    distances = []

    if source.bracket_type == "ordinary":
        # Distance to next ordinary bracket
        ord_taxable = max(ordinary_ytd - std_ded, 0)
        for upper, _rate in ord_brackets:
            if ord_taxable < upper:
                distances.append(upper - ord_taxable)
                break

        # NIIT threshold (MAGI-based, NOT inflated)
        niit_thresh = NIIT_THRESHOLD[config.filing_status]
        if magi < niit_thresh:
            distances.append(niit_thresh - magi)

    elif source.bracket_type == "ltcg":
        # LTCG brackets stacked on ordinary taxable income
        ord_taxable = max(ordinary_ytd - std_ded, 0)
        stacked = ord_taxable + ltcg_ytd
        ltcg_brackets = _inflate_brackets(LTCG_BRACKETS[config.filing_status], yrs, infl)
        for upper, _rate in ltcg_brackets:
            if stacked < upper:
                gain_distance = upper - stacked  # distance in gain-space
                # Convert to sale-space: if gain_fraction=0.5, need to sell $2 to generate $1 gain
                gf = max(source.gain_fraction, 0.01)  # avoid div-by-zero
                distances.append(gain_distance / gf)
                break

        # NIIT threshold (MAGI-based). For LTCG sources, the MAGI increase per
        # dollar sold = gain_fraction (only the gain portion increases MAGI)
        niit_thresh = NIIT_THRESHOLD[config.filing_status]
        if magi < niit_thresh:
            magi_distance = niit_thresh - magi
            gf = max(source.gain_fraction, 0.01)
            distances.append(magi_distance / gf)

    if not distances:
        return float("inf")  # in top bracket, no boundary ahead
    return max(min(distances), 0.0)
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestDynamicWaterfall -v
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "feat(citadel): add _rank_sources and _max_draw_before_boundary"
```

---

### Task 4: `_execute_draw` dispatch

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write test**

```python
    def test_execute_draw_td_records_ordinary(self):
        """Drawing from TD records ordinary income in accumulator."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _execute_draw)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            td_cash=50_000, sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True)
        source = _WithdrawalSource(
            key="td_cash", wrapper="td", asset_type="cash", index=0,
            available=50_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ordinary",
        )
        _execute_draw(state, cfg, source, 10_000)
        assert state.td_cash == pytest.approx(40_000)
        assert state.tax_year_accum.tax_deferred_withdrawals == pytest.approx(10_000)

    def test_execute_draw_btc_uses_sell_tracked(self):
        """Drawing BTC uses _sell_btc_tracked for lot tracking."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _execute_draw)
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=2.0, btc_price=50_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=2.0,
                             cost_basis=30_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, cost_basis_method="fifo")
        source = _WithdrawalSource(
            key="btc", wrapper="taxable", asset_type="btc", index=0,
            available=100_000, growth_rate=0.5, horizon=10,
            gain_fraction=0.7, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ltcg",
        )
        _execute_draw(state, cfg, source, 25_000)  # sell $25k worth
        assert state.btc_stack < 2.0
        assert state.tax_year_accum.lt_capital_gains > 0

    def test_execute_draw_roth_records_roth(self):
        """Drawing from Roth records roth_withdrawals, no tax."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _execute_draw)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            tf_cash=30_000, tf_reserves=[10_000, 0, 0],
            sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True)
        source = _WithdrawalSource(
            key="tf_cash_res", wrapper="tf", asset_type="cash", index=0,
            available=40_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=True,
            is_bracket_sensitive=False, bracket_type="none",
        )
        _execute_draw(state, cfg, source, 35_000)
        assert state.tf_cash == 0  # drained cash first
        assert state.tf_reserves[0] == pytest.approx(5_000)  # then reserves
        assert state.tax_year_accum.roth_withdrawals == pytest.approx(35_000)
```

- [ ] **Step 2: Implement `_execute_draw`**

```python
def _execute_draw(state: CitadelState, config: SimConfig,
                  source: _WithdrawalSource, amount: float) -> None:
    """Execute a withdrawal from the specified source. Mutates state."""
    if amount <= 0:
        return

    if source.wrapper == "taxable":
        if source.asset_type == "cash":
            state.cash -= min(amount, state.cash)
        elif source.asset_type == "reserve":
            state.reserves[source.index] -= min(amount, state.reserves[source.index])
        elif source.asset_type == "invest":
            _sell_investments_tracked(state, config, source.index, amount)
        elif source.asset_type == "btc":
            if state.btc_price > 0:
                btc_to_sell = amount / state.btc_price
                _sell_btc_tracked(state, config, btc_to_sell)

    elif source.wrapper == "td":
        remaining = amount
        if source.asset_type == "cash":
            d = min(state.td_cash, remaining); state.td_cash -= d; remaining -= d
        elif source.asset_type == "reserve":
            d = min(state.td_reserves[source.index], remaining)
            state.td_reserves[source.index] -= d; remaining -= d
        elif source.asset_type == "invest":
            d = min(state.td_investments[source.index], remaining)
            state.td_investments[source.index] -= d; remaining -= d
        elif source.asset_type == "btc":
            if state.btc_price > 0 and state.td_btc_stack > 0:
                btc_val = state.td_btc_stack * state.btc_price
                d = min(btc_val, remaining)
                state.td_btc_stack -= d / state.btc_price
                remaining -= d
        actual = amount - remaining
        if state.tax_year_accum is not None and actual > 0:
            state.tax_year_accum.tax_deferred_withdrawals += actual

    elif source.wrapper == "tf":
        remaining = amount
        if source.asset_type == "cash":
            # TF cash + reserves combined source — draw cash first, then reserves
            d = min(state.tf_cash, remaining); state.tf_cash -= d; remaining -= d
            for i in range(len(state.tf_reserves)):
                if remaining <= 0: break
                d = min(state.tf_reserves[i], remaining)
                state.tf_reserves[i] -= d; remaining -= d
        elif source.asset_type == "invest":
            for i in reversed(range(len(state.tf_investments))):
                if remaining <= 0: break
                d = min(state.tf_investments[i], remaining)
                state.tf_investments[i] -= d; remaining -= d
        elif source.asset_type == "btc":
            if state.btc_price > 0 and state.tf_btc_stack > 0:
                btc_val = state.tf_btc_stack * state.btc_price
                d = min(btc_val, remaining)
                state.tf_btc_stack -= d / state.btc_price
                remaining -= d
        actual = amount - remaining
        if state.tax_year_accum is not None and actual > 0:
            state.tax_year_accum.roth_withdrawals += actual
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestDynamicWaterfall -v
```

- [ ] **Step 4: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "feat(citadel): add _execute_draw dispatch for all source types"
```

---

### Task 5: Replace `_spending_waterfall` with cost-ranked loop

This is the integration task. Replace the body of `_spending_waterfall` with the re-ranking loop that uses all the helpers from Tasks 1-4.

**Files:**
- Modify: `btc_web/engines/citadel.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write integration tests**

Add to `TestDynamicWaterfall`:

```python
    def test_full_waterfall_btc_protected_early(self):
        """In 2035, BTC should be among the last non-Roth assets sold."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=1.0, start_yr=2035, end_yr=2037,
            freq="Annually", monthly_spend=30_000,
            cash_initial=50_000, cash_rate=4.0,
            selected_qs=[0.25], tax_enabled=True, state_code="TX",
            td_cash_initial=100_000,
            reserve_bins=[
                {"label": "S", "initial": 20_000, "rate": 5.0, "volatility": 0},
                {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Eq", "initial": 50_000, "return_rate": 10.0, "volatility": 0},
                {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
            ],
        )
        model = _test_model()
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        initial_btc = state.btc_stack
        # Run 2 annual steps — with $360k annual spend, cash+reserves+TD+equities
        # should be drawn before BTC (BTC has highest opportunity cost)
        for _ in range(2):
            state = step(state, cfg, model.price_at(0.25, state.t + 1), rng, model=model)
        # BTC should be preserved (or minimally touched)
        assert state.btc_stack >= initial_btc * 0.9, \
            f"BTC should be mostly preserved in early retirement, got {state.btc_stack:.3f}"

    def test_full_waterfall_roth_last(self):
        """Roth is never touched while other sources remain."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=0, start_yr=2035, end_yr=2037,
            freq="Annually", monthly_spend=5000,
            cash_initial=50_000, selected_qs=[0.25],
            tax_enabled=True, state_code="TX",
            td_cash_initial=50_000,
            tf_cash_initial=50_000,
            reserve_bins=[
                {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
            ],
        )
        model = _test_model()
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        for _ in range(2):
            state = step(state, cfg, 50_000, rng, model=model)
        # Taxable cash + TD cash should cover spending. Roth untouched.
        assert state.tf_cash == pytest.approx(50_000, abs=1000)

    def test_full_waterfall_non_tax_mode(self):
        """Non-tax mode still works with the dynamic waterfall."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=1.0, start_yr=2031, end_yr=2035,
            freq="Annually", monthly_spend=5000,
            cash_initial=50_000, selected_qs=[0.25],
            tax_enabled=False,
            reserve_bins=[
                {"label": "S", "initial": 20_000, "rate": 0, "volatility": 0},
                {"label": "M", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "L", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Eq", "initial": 50_000, "return_rate": 0, "volatility": 0},
                {"label": "Bd", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        r = simulate(cfg, _test_model())
        assert r.total_usd.shape[1] > 0
        assert r.taxes_paid is None
        assert r.total_usd[0, -1] >= 0

    def test_full_waterfall_high_spender_crosses_brackets(self):
        """$500k monthly spend should cross multiple brackets without hanging."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=0, start_yr=2035, end_yr=2036,
            freq="Monthly", monthly_spend=500_000,
            cash_initial=0, selected_qs=[0.25],
            tax_enabled=True, state_code="CA",
            td_cash_initial=10_000_000,
            reserve_bins=[
                {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
            ],
        )
        model = _test_model()
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        # Run 1 month — should not hang and should draw from TD
        state = step(state, cfg, 50_000, rng, model=model)
        assert state.td_cash < 10_000_000
        assert state.spending_shortfall == 0

    def test_full_waterfall_shortfall_when_all_depleted(self):
        """Returns shortfall when all sources exhausted."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=0, start_yr=2035, end_yr=2036,
            freq="Annually", monthly_spend=100_000,
            cash_initial=1_000, selected_qs=[0.25],
            tax_enabled=False,
            reserve_bins=[
                {"label": "S", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "M", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "L", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Eq", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bd", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        model = _test_model()
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        state = step(state, cfg, 50_000, rng, model=model)
        assert state.spending_shortfall > 0
```

- [ ] **Step 2: Replace `_spending_waterfall` body**

Read the current `_spending_waterfall` function in `btc_web/engines/citadel.py` (starts around line ~778). Replace the ENTIRE function body (keep the signature and docstring) with the cost-ranked loop:

```python
def _spending_waterfall(state: CitadelState, config: SimConfig,
                        amount: float,
                        model: "PriceModel | None" = None) -> float:
    """Draw `amount` from accounts using cost-ranked dynamic ordering.
    Returns unmet shortfall. Mutates state in place.

    Computes tax cost + opportunity cost for each available source,
    draws from cheapest first, re-ranks at bracket boundaries.
    Roth sources always rank after all non-Roth.
    """
    remaining = amount
    if remaining <= 0:
        return 0.0

    sources = _build_source_list(state, config, model)
    if not sources:
        return remaining

    while remaining > 0.01 and sources:
        _score_sources(sources, state, config, model)
        ranked = _rank_sources(sources)

        drew_something = False
        for best in ranked:
            if best.available < 0.01:
                continue

            max_draw = _max_draw_before_boundary(state, config, best)
            if max_draw < 0.01:
                continue

            draw = min(remaining, best.available, max_draw)
            _execute_draw(state, config, best, draw)
            remaining -= draw

            # Update source availability
            best.available -= draw
            # BTC/investment gain fractions may have changed — rebuild on next score
            drew_something = True
            break  # re-rank with updated state

        if not drew_something:
            break

        sources = [s for s in sources if s.available > 0.01]

    return max(remaining, 0.0)
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestDynamicWaterfall -v --tb=short
```

- [ ] **Step 4: Run full suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -15
```

Investigate any failures — the full suite includes existing tests that exercise the waterfall through `simulate()` and `step()`. Some existing test expectations may need updating since the withdrawal ordering changed.

- [ ] **Step 5: Fix any failing existing tests**

Existing tests that assert specific withdrawal ordering or terminal balances may fail because the dynamic waterfall draws from different sources than the old fixed sequence. Update expectations to match the new cost-ranked behavior. Key tests to check:
- `TestTaxSimComparative` in `test_web.py` — may see different tax totals
- `TestCashFloorEnforcement` in `test_web.py` — floor behavior should be unchanged (runs after waterfall)
- `TestWithdrawalOrderTaxAdvantaged` in `test_web.py` — ordering assertions will likely change
- **`test_citadel.py`** — 5 direct calls to `_spending_waterfall` (lines ~118, 124, 132, 145, 152) with ordering assertions based on the old fixed sequence. Tests like `test_full_waterfall_to_btc` and `test_cash_depleted_draws_reserves` expect specific reserve draw order (by index) that will change under cost-ranked ordering (reserves drawn by ascending opportunity cost, not by index). Update assertions to match new behavior.

- [ ] **Step 6: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/test_web.py
git commit -m "feat(citadel): replace fixed waterfall with dynamic cost-ranked loop"
```

---

### Task 6: Model Info notes + remaining tests + final verification

**Files:**
- Modify: `btc_web/layout/model_info.py`
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Add Model Info notes**

Read `btc_web/layout/model_info.py`. Add two notes to the Citadel Planner section:

1. **Opportunity cost horizons:** "The Citadel Planner computes withdrawal cost as immediate tax plus forgone compounding. Bitcoin uses a 10-year horizon (twice the historical 5-year break-even). Equities and bonds use 15 years. Treasuries use the holder's remaining lifetime (capped at 40 years). These horizons determine how aggressively each asset is protected from withdrawal."

2. **Roth-last policy:** "Roth (tax-free) account withdrawals are always deferred until all taxable and tax-deferred sources are exhausted, preserving the benefit of tax-free compounding."

- [ ] **Step 2: Add remaining edge case tests**

Add to `TestDynamicWaterfall`:

```python
    def test_btc_midpack_in_2065(self):
        """Spec test 6: In 2065 BTC moves to mid-pack ranking (growth slowed)."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        from engines.tax_lots import TaxLot
        from engines.tax import TaxYearAccumulator
        from btc_core import yr_to_t
        t = yr_to_t(2065)
        state = CitadelState(
            cash=50_000, reserves=[50_000, 0, 0],
            investments=[100_000, 0], invest_cost_basis=[50_000, 0],
            btc_stack=1.0, btc_price=50_000, t=t, sim_date="2065-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=1.0,
                             cost_basis=10_000, source="initial")],
            td_cash=50_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX", start_yr=2065,
                        filing_status="single", inflation=4.0,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=_test_model())
        _score_sources(sources, state, cfg, model=_test_model())
        non_roth = [s for s in sources if not s.is_roth]
        ranked = sorted(non_roth, key=lambda s: s.cost)
        btc = [s for s in ranked if s.key == "btc"][0]
        btc_rank = ranked.index(btc)
        # BTC should NOT be last (it was in 2035) — should be mid-pack
        assert btc_rank < len(ranked) - 1, "BTC should be mid-pack in 2065"

    def test_td_draw_shifts_ltcg_stack_base(self):
        """Spec test 8: TD draw increases ordinary income → shifts LTCG bracket base."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _execute_draw,
                                      _max_draw_before_boundary)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            td_cash=500_000, sim_date="2035-06-15",
            investments=[200_000, 0], invest_cost_basis=[100_000, 0],
            btc_stack=0, btc_price=50_000,
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX",
                        filing_status="single", inflation=4.0, start_yr=2031,
                        freq="Monthly",
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        inv_src = _WithdrawalSource(
            key="invest_0", wrapper="taxable", asset_type="invest", index=0,
            available=200_000, growth_rate=0.10, horizon=15,
            gain_fraction=0.5, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ltcg",
        )
        # LTCG boundary before TD draw
        before = _max_draw_before_boundary(state, cfg, inv_src)
        # Draw $100k from TD → adds $100k ordinary income
        td_src = _WithdrawalSource(
            key="td_cash", wrapper="td", asset_type="cash", index=0,
            available=500_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ordinary",
        )
        _execute_draw(state, cfg, td_src, 100_000)
        # LTCG boundary after TD draw — should be smaller (base shifted up)
        after = _max_draw_before_boundary(state, cfg, inv_src)
        assert after < before, "TD draw should shift LTCG stack base, reducing boundary distance"

    def test_gain_fraction_updates_after_partial_sale(self):
        """Spec test 15: Partial BTC sale changes gain fraction for next scoring."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources,
                                      _sell_btc_tracked)
        from engines.tax_lots import TaxLot
        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=2.0, btc_price=100_000, sim_date="2035-06-15",
            tax_lots=[
                TaxLot(date="2031-01-01", btc=1.0, cost_basis=10_000, source="initial"),
                TaxLot(date="2034-01-01", btc=1.0, cost_basis=90_000, source="rebal_buy"),
            ],
        )
        cfg = SimConfig(tax_enabled=False, cost_basis_method="fifo",
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        # Before sale: avg basis = (10k + 90k) / (2 * 100k) = 50%, gain_frac = 50%
        sources_before = _build_source_list(state, cfg, model=None)
        btc_before = [s for s in sources_before if s.key == "btc"][0]
        # Sell the cheap lot (FIFO sells the 10k-basis lot first)
        _sell_btc_tracked(state, cfg, 1.0)
        # After sale: only the 90k-basis lot remains, gain_frac = 1 - 90k/100k = 10%
        sources_after = _build_source_list(state, cfg, model=None)
        btc_after = [s for s in sources_after if s.key == "btc"][0]
        assert btc_after.gain_fraction < btc_before.gain_fraction

    def test_late_retirement_crossover(self):
        """Spec test 17: BTC becomes cheaper to sell than treasuries in late retirement."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        from engines.tax_lots import TaxLot
        from btc_core import yr_to_t
        t = yr_to_t(2070)
        state = CitadelState(
            cash=0, reserves=[100_000, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=1.0, btc_price=50_000, t=t, sim_date="2070-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=1.0,
                             cost_basis=10_000, source="initial")],
        )
        cfg = SimConfig(tax_enabled=False, start_yr=2070, birth_year=1990,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=_test_model())
        _score_sources(sources, state, cfg, model=_test_model())
        btc = [s for s in sources if s.key == "btc"][0]
        tres = [s for s in sources if s.key == "reserve_0"][0]
        # In 2070, age 80, treasury horizon = min(90-80, 40) = 10 years
        # BTC 10yr growth has slowed. Treasury at 5% over 10yr = 63%
        # BTC should be cheaper or comparable to treasury
        assert btc.cost < tres.cost * 1.5, \
            f"BTC ({btc.cost:.2f}) should be comparable or cheaper than treasury ({tres.cost:.2f}) in 2070"

    def test_negative_btc_growth_ranks_first(self):
        """When model returns negative 10yr growth, BTC is cheapest to sell."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        from engines.tax_lots import TaxLot

        class _DeclineModel:
            def __init__(self):
                import pandas as pd
                self.fits = {0.25: {"slope": 5.0, "intercept": 2.0}}
                self.genesis = pd.Timestamp("2009-07-25")
            def price_at(self, q, t):
                return max(50_000 * (1 - t / 200), 100)  # declining
            def quantile_at(self, price, t):
                return 0.5

        state = CitadelState(
            cash=50_000, reserves=[0, 0, 0],
            investments=[100_000, 0], invest_cost_basis=[50_000, 0],
            btc_stack=1.0, btc_price=50_000, t=50, sim_date="2060-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=1.0,
                             cost_basis=10_000, source="initial")],
        )
        cfg = SimConfig(tax_enabled=False, start_yr=2060,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=_DeclineModel())
        _score_sources(sources, state, cfg, model=_DeclineModel())
        btc = [s for s in sources if s.key == "btc"][0]
        cash = [s for s in sources if s.key == "cash"][0]
        # Negative growth → negative opportunity cost → BTC cheaper than cash
        assert btc.cost < cash.cost

    def test_treasury_horizon_age_92(self):
        """Treasury horizon clamps to 1 for ages 90+."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list)
        state = CitadelState(
            cash=0, reserves=[50_000, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=0, btc_price=50_000, sim_date="2035-06-15",
            period=0,
        )
        cfg = SimConfig(tax_enabled=False, birth_year=1943, start_yr=2035,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        res = [s for s in sources if s.key == "reserve_0"][0]
        # Age 92, horizon = max(min(90-92, 40), 1) = max(-2, 1) = 1
        assert res.horizon == 1

    def test_model_failure_fallback(self):
        """When model.price_at throws, BTC falls back to equity rate."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list)
        from engines.tax_lots import TaxLot

        class _BrokenModel:
            def __init__(self):
                import pandas as pd
                self.fits = {0.25: {"slope": 5.0, "intercept": 2.0}}
                self.genesis = pd.Timestamp("2009-07-25")
            def price_at(self, q, t):
                raise ValueError("model broken")
            def quantile_at(self, price, t):
                return 0.5

        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=1.0, btc_price=50_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=1.0,
                             cost_basis=10_000, source="initial")],
        )
        cfg = SimConfig(tax_enabled=False,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=_BrokenModel())
        btc = [s for s in sources if s.key == "btc"][0]
        # Should fall back to equity rate (10%)
        assert btc.growth_rate == pytest.approx(0.10)
```

- [ ] **Step 3: Run full suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ -v --tb=short 2>&1 | tail -25
```

- [ ] **Step 4: Verify imports**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c \
  "from engines.citadel import (_WithdrawalSource, _build_source_list, _score_sources, _rank_sources, _max_draw_before_boundary, _execute_draw); print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add btc_web/engines/citadel.py btc_web/layout/model_info.py btc_web/test_web.py
git commit -m "feat(citadel): Model Info notes, edge case tests, final verification"
```
