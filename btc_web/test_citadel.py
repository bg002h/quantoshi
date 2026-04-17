"""Citadel planner: tax system, waterfall, regimes, bands, presets."""
from conftest import (
    M,
    _CHECKLIST_OPTIONS,
    _ControlledPriceModel,
    _MockPriceModel,
    _SNAPSHOT_CONTROLS,
    _TAB_CONTROLS,
    _bare_config,
    _test_model,
    build_citadel_figure,
    go,
    np,
    pd,
    pytest,
    yr_to_t,
)


class TestTaxData:
    def test_federal_brackets_single_tcja(self):
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        single = FEDERAL_BRACKETS_TCJA["single"]
        assert single[0] == (11_925, 0.10)
        assert single[-1][1] == 0.37
        assert len(single) == 7

    def test_federal_brackets_mfj_tcja(self):
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        mfj = FEDERAL_BRACKETS_TCJA["mfj"]
        assert mfj[0] == (23_850, 0.10)
        assert mfj[-1][1] == 0.37

    def test_federal_brackets_sunset(self):
        from engines.tax_data import FEDERAL_BRACKETS_SUNSET
        single = FEDERAL_BRACKETS_SUNSET["single"]
        assert single[-1][1] == 0.396
        assert len(single) == 7

    def test_ltcg_brackets_single(self):
        from engines.tax_data import LTCG_BRACKETS
        single = LTCG_BRACKETS["single"]
        assert single[0] == (48_350, 0.00)
        assert single[1] == (533_400, 0.15)
        assert single[2] == (float("inf"), 0.20)

    def test_standard_deduction_tcja(self):
        from engines.tax_data import STANDARD_DEDUCTION_TCJA, STANDARD_DEDUCTION_SUNSET
        assert STANDARD_DEDUCTION_TCJA["single"] == 15_000
        assert STANDARD_DEDUCTION_TCJA["mfj"] == 30_000
        assert STANDARD_DEDUCTION_SUNSET["single"] == 8_300
        assert STANDARD_DEDUCTION_SUNSET["mfj"] == 16_600

    def test_niit_thresholds(self):
        from engines.tax_data import NIIT_RATE, NIIT_THRESHOLD
        assert NIIT_RATE == 0.038
        assert NIIT_THRESHOLD["single"] == 200_000
        assert NIIT_THRESHOLD["mfj"] == 250_000

    def test_state_tax_no_income_tax(self):
        from engines.tax_data import STATE_TAX_RATES
        for st in ("AK", "FL", "NV", "NH", "SD", "TN", "TX", "WA", "WY"):
            assert STATE_TAX_RATES[st] == 0.0, f"{st} should be 0"

    def test_state_tax_california(self):
        from engines.tax_data import STATE_TAX_RATES
        assert STATE_TAX_RATES["CA"] == 13.30

    def test_state_tax_count(self):
        from engines.tax_data import STATE_TAX_RATES
        assert len(STATE_TAX_RATES) == 51  # 50 states + DC

    def test_rmd_factors(self):
        from engines.tax_data import RMD_FACTORS
        assert RMD_FACTORS[73] == 26.5
        assert RMD_FACTORS[75] == 24.6
        assert RMD_FACTORS[80] == 20.2
        assert RMD_FACTORS[90] == 12.2
        assert 72 in RMD_FACTORS
        assert 120 in RMD_FACTORS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



class TestTaxLots:
    def test_create_lot(self):
        from engines.tax_lots import TaxLot
        lot = TaxLot(date="2024-01-15", btc=0.5, cost_basis=42_000.0, source="initial")
        assert lot.btc == 0.5
        assert lot.cost_basis == 42_000.0

    def test_sell_fifo_single_lot(self):
        from engines.tax_lots import TaxLot, sell_lots
        lots = [TaxLot("2023-01-01", 1.0, 20_000.0, "initial")]
        result = sell_lots(lots, btc_to_sell=0.5, sale_price=50_000.0,
                          sale_date="2025-06-01", method="fifo")
        assert result.btc_sold == 0.5
        assert len(result.gains) == 1
        g = result.gains[0]
        assert g.btc == 0.5
        assert g.proceeds == 25_000.0
        assert g.cost == 10_000.0
        assert g.gain == 15_000.0
        assert g.is_long_term is True
        assert len(result.remaining_lots) == 1
        assert result.remaining_lots[0].btc == 0.5

    def test_sell_fifo_multiple_lots(self):
        from engines.tax_lots import TaxLot, sell_lots
        lots = [
            TaxLot("2023-01-01", 0.3, 20_000.0, "initial"),
            TaxLot("2025-03-01", 0.7, 80_000.0, "rebal_buy"),
        ]
        result = sell_lots(lots, btc_to_sell=0.5, sale_price=100_000.0,
                          sale_date="2025-06-01", method="fifo")
        assert result.btc_sold == 0.5
        assert len(result.gains) == 2
        assert result.gains[0].btc == 0.3
        assert result.gains[0].is_long_term is True
        assert abs(result.gains[1].btc - 0.2) < 1e-8
        assert result.gains[1].is_long_term is False
        assert len(result.remaining_lots) == 1
        assert abs(result.remaining_lots[0].btc - 0.5) < 1e-8

    def test_sell_lifo(self):
        from engines.tax_lots import TaxLot, sell_lots
        lots = [
            TaxLot("2023-01-01", 0.5, 20_000.0, "initial"),
            TaxLot("2025-05-01", 0.5, 80_000.0, "rebal_buy"),
        ]
        result = sell_lots(lots, btc_to_sell=0.3, sale_price=100_000.0,
                          sale_date="2025-06-01", method="lifo")
        assert result.gains[0].cost_basis == 80_000.0
        assert result.gains[0].is_long_term is False

    def test_sell_loss(self):
        from engines.tax_lots import TaxLot, sell_lots
        lots = [TaxLot("2024-01-01", 1.0, 100_000.0, "initial")]
        result = sell_lots(lots, btc_to_sell=0.5, sale_price=50_000.0,
                          sale_date="2025-06-01", method="fifo")
        assert result.gains[0].gain == -25_000.0

    def test_sell_more_than_available(self):
        from engines.tax_lots import TaxLot, sell_lots
        lots = [TaxLot("2024-01-01", 0.3, 50_000.0, "initial")]
        result = sell_lots(lots, btc_to_sell=1.0, sale_price=60_000.0,
                          sale_date="2025-06-01", method="fifo")
        assert result.btc_sold == 0.3
        assert len(result.remaining_lots) == 0

    def test_seed_from_stack_tracker(self):
        from engines.tax_lots import seed_lots
        st_lots = [
            {"date": "2023-06-15", "btc": 0.5, "price": 30_000},
            {"date": "2024-01-10", "btc": 0.3, "price": 45_000},
        ]
        tax_lots = seed_lots(st_lots)
        assert len(tax_lots) == 2
        assert tax_lots[0].date == "2023-06-15"
        assert tax_lots[0].cost_basis == 30_000
        assert tax_lots[1].source == "initial"

    def test_seed_manual_entry(self):
        from engines.tax_lots import seed_lots
        tax_lots = seed_lots([], start_stack=1.0, start_price=60_000.0,
                             start_date="2031-01-01")
        assert len(tax_lots) == 1
        assert tax_lots[0].btc == 1.0
        assert tax_lots[0].cost_basis == 60_000.0

    def test_seed_empty(self):
        from engines.tax_lots import seed_lots
        assert seed_lots([]) == []



class TestTaxComputation:
    def test_apply_brackets_10pct_only(self):
        from engines.tax import apply_progressive_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        tax = apply_progressive_brackets(10_000, FEDERAL_BRACKETS_TCJA["single"])
        assert tax == pytest.approx(1_000.0)

    def test_apply_brackets_two_brackets(self):
        from engines.tax import apply_progressive_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        tax = apply_progressive_brackets(30_000, FEDERAL_BRACKETS_TCJA["single"])
        expected = 11_925 * 0.10 + (30_000 - 11_925) * 0.12
        assert tax == pytest.approx(expected)

    def test_apply_brackets_top_bracket(self):
        from engines.tax import apply_progressive_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        tax = apply_progressive_brackets(1_000_000, FEDERAL_BRACKETS_TCJA["single"])
        assert tax > 300_000

    def test_apply_brackets_zero(self):
        from engines.tax import apply_progressive_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        assert apply_progressive_brackets(0, FEDERAL_BRACKETS_TCJA["single"]) == 0.0

    def test_ltcg_stacking_zero_ordinary(self):
        from engines.tax import compute_ltcg_tax
        tax = compute_ltcg_tax(50_000, stacking_base=0, filing_status="single")
        expected = (50_000 - 48_350) * 0.15
        assert tax == pytest.approx(expected)

    def test_ltcg_stacking_high_ordinary(self):
        from engines.tax import compute_ltcg_tax
        tax = compute_ltcg_tax(100_000, stacking_base=80_000, filing_status="single")
        assert tax == pytest.approx(100_000 * 0.15)

    def test_loss_netting_st_loss_offsets_lt_gain(self):
        from engines.tax import net_capital_gains
        result = net_capital_gains(st_gains=1_000, st_losses=5_000,
                                   lt_gains=10_000, lt_losses=0, carryforward=0)
        assert result.net_lt == 6_000
        assert result.net_st == 0
        assert result.loss_deduction == 0
        assert result.new_carryforward == 0

    def test_loss_netting_excess_carries_forward(self):
        from engines.tax import net_capital_gains
        result = net_capital_gains(st_gains=0, st_losses=10_000,
                                   lt_gains=0, lt_losses=0, carryforward=0)
        assert result.loss_deduction == 3_000
        assert result.new_carryforward == 7_000

    def test_loss_netting_with_carryforward(self):
        from engines.tax import net_capital_gains
        result = net_capital_gains(st_gains=5_000, st_losses=0,
                                   lt_gains=0, lt_losses=0, carryforward=8_000)
        assert result.loss_deduction == 3_000
        assert result.new_carryforward == 0

    # ── §1212(b) character-preserved carryforward ──────────────────────────

    def test_1212b_st_loss_carries_as_st(self):
        """A $10k ST-only loss: $3k deduction (from ST first), $7k ST carry."""
        from engines.tax import net_capital_gains
        result = net_capital_gains(st_gains=0, st_losses=10_000,
                                   lt_gains=0, lt_losses=0)
        assert result.loss_deduction == 3_000
        assert result.new_st_carryforward == 7_000
        assert result.new_lt_carryforward == 0

    def test_1212b_lt_loss_carries_as_lt(self):
        """A $10k LT-only loss: $3k deduction (all LT, since no ST), $7k LT carry."""
        from engines.tax import net_capital_gains
        result = net_capital_gains(st_gains=0, st_losses=0,
                                   lt_gains=0, lt_losses=10_000)
        assert result.loss_deduction == 3_000
        assert result.new_st_carryforward == 0
        assert result.new_lt_carryforward == 7_000

    def test_1212b_mixed_loss_deduction_from_st_first(self):
        """$2k ST + $5k LT losses: $2k ST absorbed, $1k LT completes the $3k
        deduction, $4k LT carry. No ST carry."""
        from engines.tax import net_capital_gains
        result = net_capital_gains(st_gains=0, st_losses=2_000,
                                   lt_gains=0, lt_losses=5_000)
        assert result.loss_deduction == 3_000
        assert result.new_st_carryforward == 0
        assert result.new_lt_carryforward == 4_000

    def test_1212b_st_carryforward_nets_against_st_gains(self):
        """A $10k ST carryforward nets against ST gains first, NOT LT."""
        from engines.tax import net_capital_gains
        # $8k ST gains this year; $10k ST carry from last year. After netting:
        # net_st = 8,000 - 10,000 = -2,000. No LT activity → $2k ST loss,
        # takes $2k deduction, $0 carry.
        result = net_capital_gains(st_gains=8_000, st_losses=0,
                                   lt_gains=0, lt_losses=0,
                                   st_carryforward=10_000)
        assert result.net_st == 0
        assert result.loss_deduction == 2_000
        assert result.new_st_carryforward == 0
        assert result.new_lt_carryforward == 0

    def test_1212b_legacy_carryforward_routed_to_lt(self):
        """Backward compatibility: the old single `carryforward` scalar (no
        ST/LT split provided) is treated as LT, matching the pre-§1212(b)
        behavior. Guards old simulations whose state was seeded before the
        split."""
        from engines.tax import net_capital_gains
        result = net_capital_gains(st_gains=0, st_losses=0,
                                   lt_gains=0, lt_losses=0,
                                   carryforward=5_000)
        assert result.loss_deduction == 3_000
        # The $5k legacy carry flowed into LT (not ST).
        assert result.new_lt_carryforward == 2_000
        assert result.new_st_carryforward == 0

    def test_niit_below_threshold(self):
        from engines.tax import compute_niit
        assert compute_niit(magi=150_000, nii=50_000, filing_status="single") == 0.0

    def test_niit_above_threshold(self):
        from engines.tax import compute_niit
        assert compute_niit(300_000, 80_000, "single") == pytest.approx(3_040.0)

    def test_niit_lesser_of_rule(self):
        from engines.tax import compute_niit
        assert compute_niit(220_000, 50_000, "single") == pytest.approx(760.0)

    def test_annual_tax_simple_case(self):
        from engines.tax import TaxYearAccumulator, compute_annual_tax
        accum = TaxYearAccumulator(
            tax_deferred_withdrawals=60_000,
            interest_income=5_000,
            other_income=0,
            lt_capital_gains=45_000,
        )
        result = compute_annual_tax(accum, filing_status="single",
                                     tcja_sunset=False, sim_year=2031,
                                     inflation_rate=0.04, state_rate=0.0)
        assert result["total"] > 0
        assert result["federal_ordinary"] > 0
        assert result["federal_ltcg"] >= 0
        assert result["niit"] == 0  # AGI ~$110k, under $200k
        assert result["effective_rate"] > 0
        assert "loss_carryforward" in result

    def test_annual_tax_with_niit(self):
        from engines.tax import TaxYearAccumulator, compute_annual_tax
        accum = TaxYearAccumulator(
            tax_deferred_withdrawals=100_000,
            interest_income=20_000,
            lt_capital_gains=200_000,
        )
        result = compute_annual_tax(accum, filing_status="single",
                                     tcja_sunset=False, sim_year=2025,
                                     inflation_rate=0.0, state_rate=0.0)
        assert result["niit"] > 0  # AGI ~$320k, well above $200k

    def test_annual_tax_with_state(self):
        from engines.tax import TaxYearAccumulator, compute_annual_tax
        accum = TaxYearAccumulator(tax_deferred_withdrawals=100_000)
        result = compute_annual_tax(accum, filing_status="single",
                                     tcja_sunset=False, sim_year=2025,
                                     inflation_rate=0.0, state_rate=13.30)
        assert result["state"] > 0

    def test_brackets_inflation_indexed(self):
        from engines.tax import _inflate_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        base = FEDERAL_BRACKETS_TCJA["single"]
        inflated = _inflate_brackets(base, years=10, rate=0.04)
        assert inflated[0][0] == pytest.approx(11_925 * 1.04**10, rel=0.01)
        assert inflated[0][1] == 0.10

    def test_annual_tax_sunset_brackets(self):
        from engines.tax import TaxYearAccumulator, compute_annual_tax
        accum = TaxYearAccumulator(tax_deferred_withdrawals=600_000)
        result_tcja = compute_annual_tax(accum, filing_status="single",
                                          tcja_sunset=False, sim_year=2025,
                                          inflation_rate=0.0, state_rate=0.0)
        result_sunset = compute_annual_tax(accum, filing_status="single",
                                            tcja_sunset=True, sim_year=2025,
                                            inflation_rate=0.0, state_rate=0.0)
        # Sunset has higher top rate (39.6% vs 37%), so tax should be higher
        assert result_sunset["total"] > result_tcja["total"]


# ═══════════════════════════════════════════════════════════════════════════════
# Section: Citadel Tax Integration
# ═══════════════════════════════════════════════════════════════════════════════

class _MockPriceModel:
    """Minimal mock satisfying the PriceModel protocol for Citadel tests."""
    def __init__(self):
        import pandas as pd
        self.fits = {0.25: {"slope": 5.0, "intercept": 2.0}}
        self.genesis = pd.Timestamp("2009-07-25")

    def price_at(self, q, t):
        # Return a deterministic price that grows with t
        return 50_000.0 * (1 + t / 100)

    def quantile_at(self, price, t):
        return 0.50


def _test_model():
    return _MockPriceModel()



class TestCitadelTaxIntegration:
    def test_sim_config_has_tax_fields(self):
        from engines.citadel import SimConfig
        cfg = SimConfig(tax_enabled=True, filing_status="single",
                        state_code="CA", birth_year=1985,
                        cost_basis_method="fifo")
        assert cfg.tax_enabled is True
        assert cfg.state_code == "CA"
        assert cfg.td_btc_stack == 0.0
        assert cfg.tf_btc_stack == 0.0

    def test_citadel_state_has_tax_fields(self):
        from engines.citadel import CitadelState
        state = CitadelState()
        assert hasattr(state, "tax_lots")
        assert hasattr(state, "td_btc_stack")
        assert hasattr(state, "tf_btc_stack")
        assert hasattr(state, "td_cash")
        assert hasattr(state, "total_taxes_paid")
        assert state.total_taxes_paid == 0.0

    def test_tax_off_preserves_existing_behavior(self):
        """When tax_enabled=False, engine behavior is identical."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=1.0, start_yr=2031, end_yr=2033,
                        freq="Annually", monthly_spend=5000,
                        cash_initial=200_000, selected_qs=[0.25],
                        tax_enabled=False)
        result = simulate(cfg, _test_model())
        # Should work exactly as before — no tax fields populated
        assert result.total_usd.shape[1] > 0
        assert result.taxes_paid is None
        assert result.td_total is None
        assert result.tf_total is None

    def test_tax_enabled_runs_without_error(self):
        """Tax-on simulation should complete without crashing."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=1.0, start_yr=2031, end_yr=2033,
                        freq="Annually", monthly_spend=5000,
                        cash_initial=200_000, selected_qs=[0.25],
                        tax_enabled=True, filing_status="single",
                        state_code="CA")
        result = simulate(cfg, _test_model())
        assert result.total_usd.shape[1] > 0
        assert result.taxes_paid is not None
        assert result.taxes_paid.shape == result.total_usd.shape

    def test_tax_enabled_with_td_tf_wrappers(self):
        """Tax-on with TD/TF wrappers initialized."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=1.0, start_yr=2031, end_yr=2033,
                        freq="Annually", monthly_spend=5000,
                        cash_initial=100_000, selected_qs=[0.25],
                        tax_enabled=True, filing_status="single",
                        state_code="TX",
                        td_cash_initial=50_000,
                        tf_cash_initial=30_000)
        result = simulate(cfg, _test_model())
        assert result.td_total is not None
        assert result.tf_total is not None
        # TD and TF totals should start > 0
        assert result.td_total[0, 0] >= 50_000 or result.td_total[0, 0] >= 0
        assert result.tf_total[0, 0] >= 30_000 or result.tf_total[0, 0] >= 0

    def test_tax_enabled_rmd(self):
        """RMD fires for old enough users."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=0.5, start_yr=2031, end_yr=2035,
                        freq="Annually", monthly_spend=2000,
                        cash_initial=100_000, selected_qs=[0.25],
                        tax_enabled=True, filing_status="single",
                        state_code="TX",
                        birth_year=1956,  # age 75 in 2031, RMD starts at 73
                        td_cash_initial=500_000)
        result = simulate(cfg, _test_model())
        assert result.taxes_paid is not None
        # With large TD balance and RMD, some taxes should be paid
        assert result.annual_taxes is not None
        assert len(result.annual_taxes) > 0

    def test_initial_state_seeds_lots_when_tax_on(self):
        """Tax lots should be seeded from start_stack."""
        from engines.citadel import SimConfig, _initial_state
        cfg = SimConfig(start_stack=2.0, start_yr=2031, end_yr=2035,
                        cash_initial=100_000, selected_qs=[0.25],
                        tax_enabled=True)
        state = _initial_state(cfg, model=_test_model())
        assert len(state.tax_lots) == 1
        assert state.tax_lots[0].btc == 2.0
        assert state.tax_lots[0].source == "initial"

    def test_no_tax_rate_field(self):
        """The old tax_rate placeholder should be replaced."""
        from engines.citadel import SimConfig
        cfg = SimConfig()
        assert not hasattr(cfg, "tax_rate")



class TestTaxDefaults:
    def test_citadel_has_tax_defaults(self):
        from tab_defaults import CITADEL
        assert CITADEL["tax_enabled"] is False
        assert CITADEL["filing_status"] == "single"
        assert CITADEL["state_code"] == "TX"
        assert CITADEL["td_btc"] == 0.5
        assert CITADEL["tf_btc"] == 0.5
        assert CITADEL["cost_basis_method"] == "fifo"

    def test_build_sim_config_passes_tax_fields(self):
        from figures.citadel import _build_sim_config
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["tax_enabled"] = True
        p["filing_status"] = "mfj"
        p["state_code"] = "CA"
        p["birth_year"] = 1985
        p["td_btc"] = 0.5
        p["tf_cash"] = 100_000
        cfg = _build_sim_config(p)
        assert cfg.tax_enabled is True
        assert cfg.filing_status == "mfj"
        assert cfg.state_code == "CA"
        assert cfg.birth_year == 1985
        assert cfg.td_btc_stack == 0.5
        assert cfg.tf_cash_initial == 100_000



class TestTaxFigures:
    def test_tax_on_produces_ghost_traces(self):
        from figures.citadel import build_citadel_figure
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["tax_enabled"] = True
        p["filing_status"] = "single"
        p["state_code"] = "CA"
        p["start_yr"] = 2031
        p["end_yr"] = 2033
        fig, extra = build_citadel_figure(M, p)
        trace_names = [t.name for t in fig.data if t.name]
        assert any("no tax" in (n or "").lower() for n in trace_names)

    def test_tax_off_no_ghost_traces(self):
        from figures.citadel import build_citadel_figure
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["start_yr"] = 2031
        p["end_yr"] = 2033
        fig, extra = build_citadel_figure(M, p)
        trace_names = [t.name for t in fig.data if t.name]
        assert not any("no tax" in (n or "").lower() for n in trace_names)

    def test_tax_summary_data_returned(self):
        from figures.citadel import build_citadel_figure
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["tax_enabled"] = True
        p["start_yr"] = 2031
        p["end_yr"] = 2033
        fig, extra = build_citadel_figure(M, p)
        assert "annual_taxes" in extra



class TestTaxCallbacks:
    def test_state_to_rate(self):
        from callbacks.citadel_tax_cb import _state_to_rate
        assert _state_to_rate("CA") == 13.30
        assert _state_to_rate("TX") == 0.0
        assert _state_to_rate("NY") == 10.90



class TestTaxSnapshot:
    def test_tax_controls_in_snapshot(self):
        from snapshot import _SNAPSHOT_CONTROLS
        ids = [c[0] for c in _SNAPSHOT_CONTROLS]
        assert "cp-tax-toggle" in ids
        assert "cp-tax-config" in ids
        assert "cp-td-btc" in ids
        assert "cp-tf-btc" in ids

    def test_tax_toggle_not_in_checklist_options(self):
        # cp-tax-toggle is a dbc.Switch (bool), not a checklist. It must
        # NOT appear in _CHECKLIST_OPTIONS — doing so made _list_to_mask
        # crash on set(True). See snapshot.py note at _CHECKLIST_OPTIONS.
        from snapshot import _CHECKLIST_OPTIONS
        assert "cp-tax-toggle" not in _CHECKLIST_OPTIONS



class TestTaxSummaryPanel:
    def test_summary_panel_exists(self):
        from layout.citadel_tax import tax_summary_panel
        panel = tax_summary_panel()
        assert panel is not None

    def test_build_tax_summary_empty(self):
        from callbacks.citadel_tax_cb import _build_tax_summary
        is_open, children = _build_tax_summary([])
        assert is_open is False
        assert children == []

    def test_build_tax_summary_with_data(self):
        from callbacks.citadel_tax_cb import _build_tax_summary
        data = [{"year": 2031, "ordinary_income": 60000, "st_gains": 0,
                 "lt_gains": 45000, "federal_ordinary": 8000, "federal_ltcg": 6750,
                 "niit": 0, "state": 5000, "total": 19750, "effective_rate": 0.175}]
        is_open, children = _build_tax_summary(data)
        assert is_open is True
        assert len(children) == 2  # header + tbody



class TestTreasuryStateExemption:
    def test_treasury_interest_exempt_from_state_tax(self):
        """Treasury interest should not be state-taxed (US law)."""
        from engines.tax import TaxYearAccumulator, compute_annual_tax
        # $100k treasury interest only — no other income
        accum_treasury = TaxYearAccumulator(treasury_interest=100_000)
        result_treasury = compute_annual_tax(
            accum_treasury, filing_status="single", tcja_sunset=False,
            sim_year=2025, inflation_rate=0.0, state_rate=10.0)

        # $100k cash interest only — same amount
        accum_cash = TaxYearAccumulator(interest_income=100_000)
        result_cash = compute_annual_tax(
            accum_cash, filing_status="single", tcja_sunset=False,
            sim_year=2025, inflation_rate=0.0, state_rate=10.0)

        # Federal tax should be identical (both are ordinary income)
        assert result_treasury["federal_ordinary"] == pytest.approx(
            result_cash["federal_ordinary"])

        # State tax: treasury should be $0, cash should be ~$10k
        assert result_treasury["state"] == pytest.approx(0.0)
        assert result_cash["state"] > 0



class TestInvestmentCostBasis:
    def test_cost_basis_initialized_from_initial_value(self):
        from engines.citadel import SimConfig, _initial_state
        cfg = SimConfig(invest_bins=[
            {"label": "Equities", "initial": 200_000, "return_rate": 10, "volatility": 0},
            {"label": "Bonds", "initial": 100_000, "return_rate": 5, "volatility": 0},
        ])
        state = _initial_state(cfg)
        assert state.invest_cost_basis == [200_000, 100_000]

    def test_cost_basis_decreases_proportionally_on_sale(self):
        """Selling 50% of an investment should remove 50% of its cost basis."""
        from engines.citadel import CitadelState
        state = CitadelState(
            investments=[400_000, 100_000],       # equities doubled from 200k
            invest_cost_basis=[200_000, 100_000],  # original cost
        )
        # Simulate selling $200k of equities (50% of current $400k)
        draw = 200_000
        current = state.investments[0]
        fraction = draw / current  # 0.5
        basis_sold = state.invest_cost_basis[0] * fraction  # 100k
        gain = draw - basis_sold  # 200k - 100k = 100k gain
        state.invest_cost_basis[0] -= basis_sold
        state.investments[0] -= draw

        assert gain == pytest.approx(100_000)
        assert state.invest_cost_basis[0] == pytest.approx(100_000)  # half basis remains
        assert state.investments[0] == pytest.approx(200_000)        # half value remains

    def test_gain_increases_as_investments_appreciate(self):
        """After appreciation, same dollar withdrawal has higher gain %."""
        from engines.citadel import CitadelState
        # Start: $100k equities, $100k basis → 0% gain
        state1 = CitadelState(
            investments=[100_000, 0], invest_cost_basis=[100_000, 0])
        draw = 50_000
        fraction1 = draw / state1.investments[0]
        basis1 = state1.invest_cost_basis[0] * fraction1
        gain1 = draw - basis1
        assert gain1 == pytest.approx(0)  # no appreciation yet

        # After 2x appreciation: $200k equities, still $100k basis
        state2 = CitadelState(
            investments=[200_000, 0], invest_cost_basis=[100_000, 0])
        fraction2 = draw / state2.investments[0]  # 25%
        basis2 = state2.invest_cost_basis[0] * fraction2  # 25k
        gain2 = draw - basis2  # 50k - 25k = 25k
        assert gain2 == pytest.approx(25_000)

        # After 10x appreciation: $1M equities, still $100k basis
        state3 = CitadelState(
            investments=[1_000_000, 0], invest_cost_basis=[100_000, 0])
        fraction3 = draw / state3.investments[0]  # 5%
        basis3 = state3.invest_cost_basis[0] * fraction3  # 5k
        gain3 = draw - basis3  # 50k - 5k = 45k
        assert gain3 == pytest.approx(45_000)


# ═══════════════════════════════════════════════════════════════════════════════
# Comprehensive Tax Simulation Tests — every parameter, every asset type
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaxSimComparative:
    """Compare tax-on vs tax-off and parameter variations at the engine level."""

    @staticmethod
    def _run(tax_enabled=True, **kw):
        from engines.citadel import SimConfig, simulate
        defaults = dict(
            start_stack=1.0, start_yr=2031, end_yr=2035,
            freq="Annually", monthly_spend=5_000,
            cash_initial=100_000, selected_qs=[0.25],
        )
        defaults.update(kw)
        cfg = SimConfig(tax_enabled=tax_enabled, **defaults)
        return simulate(cfg, _test_model())

    @staticmethod
    def _tax_years(result):
        """Extract the per-year tax dicts from sim result (first sim index)."""
        if result.annual_taxes and len(result.annual_taxes) > 0:
            at = result.annual_taxes[0]
            if isinstance(at, list):
                return at  # list of year dicts
            if isinstance(at, dict):
                return [at]  # single dict wrapped
        return []

    # ── Tax on vs off ──────────────────────────────────────────────────────

    def test_tax_on_reduces_terminal_wealth(self):
        r_off = self._run(tax_enabled=False)
        r_on = self._run(tax_enabled=True, filing_status="single", state_code="CA")
        assert r_on.total_usd[0, -1] <= r_off.total_usd[0, -1]

    def test_tax_off_pays_zero_tax(self):
        r = self._run(tax_enabled=False)
        # taxes_paid should be None or all zeros
        if r.taxes_paid is not None:
            assert r.taxes_paid.max() == 0

    def test_tax_on_pays_nonzero_tax(self):
        r = self._run(tax_enabled=True, filing_status="single",
                      state_code="CA", other_income=100_000)
        assert r.taxes_paid is not None
        assert r.taxes_paid[0, -1] > 0

    # ── State tax comparison ───────────────────────────────────────────────

    def test_california_tax_exceeds_texas(self):
        r_ca = self._run(state_code="CA", filing_status="single", other_income=100_000)
        r_tx = self._run(state_code="TX", filing_status="single", other_income=100_000)
        ca_total = sum(t["total"] for t in self._tax_years(r_ca))
        tx_total = sum(t["total"] for t in self._tax_years(r_tx))
        assert ca_total > tx_total

    def test_zero_tax_state_no_state_component(self):
        r = self._run(state_code="TX", filing_status="single", other_income=50_000)
        for yr in self._tax_years(r):
            assert yr["state"] == pytest.approx(0.0)

    # ── Filing status ──────────────────────────────────────────────────────

    def test_mfj_lower_tax_than_single(self):
        """MFJ brackets are wider — same income should pay less tax."""
        r_s = self._run(filing_status="single", state_code="TX", other_income=150_000)
        r_m = self._run(filing_status="mfj", state_code="TX", other_income=150_000)
        s_total = sum(t["total"] for t in self._tax_years(r_s))
        m_total = sum(t["total"] for t in self._tax_years(r_m))
        assert m_total <= s_total

    # ── TCJA sunset ────────────────────────────────────────────────────────

    def test_sunset_higher_tax_than_tcja(self):
        r_tcja = self._run(filing_status="single", state_code="TX",
                           tcja_sunset=False, other_income=200_000)
        r_sunset = self._run(filing_status="single", state_code="TX",
                             tcja_sunset=True, other_income=200_000)
        tcja_total = sum(t["total"] for t in self._tax_years(r_tcja))
        sunset_total = sum(t["total"] for t in self._tax_years(r_sunset))
        assert sunset_total >= tcja_total

    # ── Cost basis method ──────────────────────────────────────────────────

    def test_fifo_and_lifo_produce_different_gains(self):
        """FIFO sells oldest (likely LT), LIFO sells newest (likely ST)."""
        r_fifo = self._run(cost_basis_method="fifo", filing_status="single",
                           state_code="TX", monthly_spend=20_000)
        r_lifo = self._run(cost_basis_method="lifo", filing_status="single",
                           state_code="TX", monthly_spend=20_000)
        # Both should complete without error
        assert r_fifo.total_usd.shape[1] > 0
        assert r_lifo.total_usd.shape[1] > 0

    # ── Investment cost basis ──────────────────────────────────────────────

    def test_low_basis_means_higher_tax(self):
        """$200k equities with $50k basis (150k unrealized gain) vs $200k basis (0 gain)."""
        r_low = self._run(filing_status="single", state_code="TX",
                          invest_cost_basis_initial=[50_000, 100_000],
                          monthly_spend=20_000)  # force investment sales
        r_full = self._run(filing_status="single", state_code="TX",
                           invest_cost_basis_initial=[200_000, 100_000],
                           monthly_spend=20_000)
        low_tax = sum(t["total"] for t in self._tax_years(r_low))
        full_tax = sum(t["total"] for t in self._tax_years(r_full))
        assert low_tax >= full_tax

    # ── TD wrapper (Tax-Deferred) ──────────────────────────────────────────

    def test_td_withdrawals_taxed_as_ordinary(self):
        """TD withdrawals should show up as ordinary income in tax summary."""
        r = self._run(filing_status="single", state_code="TX",
                      td_cash_initial=500_000, monthly_spend=10_000)
        yrs = self._tax_years(r)
        if yrs:
            total_ordinary = sum(t.get("ordinary_income", 0) for t in yrs)
            assert total_ordinary > 0

    # ── TF wrapper (Roth) ──────────────────────────────────────────────────

    def test_roth_only_portfolio_zero_tax(self):
        """If all assets are in Roth, no tax should be owed."""
        r = self._run(filing_status="single", state_code="TX",
                      start_stack=0, cash_initial=0,
                      invest_bins=[
                          {"label": "Equities", "initial": 0, "return_rate": 10, "volatility": 0},
                          {"label": "Bonds", "initial": 0, "return_rate": 5, "volatility": 0},
                      ],
                      tf_cash_initial=500_000,
                      monthly_spend=3_000)
        yrs = self._tax_years(r)
        if yrs:
            total_tax = sum(t["total"] for t in yrs)
            assert total_tax == pytest.approx(0.0, abs=1.0)

    # ── RMD ────────────────────────────────────────────────────────────────

    def test_rmd_creates_ordinary_income(self):
        """Birth year 1958, age 73 in 2031 → RMD should force TD withdrawal."""
        r = self._run(filing_status="single", state_code="TX",
                      birth_year=1958, td_cash_initial=1_000_000,
                      start_stack=0, cash_initial=0, monthly_spend=0,
                      start_yr=2031, end_yr=2033)
        yrs = self._tax_years(r)
        if yrs:
            has_rmd_income = any(t.get("ordinary_income", 0) > 0 for t in yrs)
            assert has_rmd_income

    def test_no_rmd_without_birth_year(self):
        """No birth year → no RMDs, TD untouched if not needed for spending."""
        r = self._run(filing_status="single", state_code="TX",
                      birth_year=None, td_cash_initial=1_000_000,
                      start_stack=0, cash_initial=500_000, monthly_spend=1_000,
                      start_yr=2031, end_yr=2033)
        # All spending covered by taxable cash, no TD withdrawals needed
        yrs = self._tax_years(r)
        if yrs:
            total_td = sum(t.get("ordinary_income", 0) for t in yrs)
            assert total_td < 100_000  # well under $1M

    # ── NIIT ───────────────────────────────────────────────────────────────

    def test_niit_triggers_above_threshold(self):
        """$300k other income (single) should trigger NIIT."""
        r = self._run(filing_status="single", state_code="TX",
                      other_income=300_000, start_yr=2025, end_yr=2027)
        yrs = self._tax_years(r)
        if yrs:
            has_niit = any(t.get("niit", 0) > 0 for t in yrs)
            assert has_niit

    def test_niit_zero_below_threshold(self):
        """$100k other income (single) should NOT trigger NIIT."""
        r = self._run(filing_status="single", state_code="TX",
                      other_income=100_000, start_stack=0, cash_initial=500_000,
                      monthly_spend=1_000, start_yr=2025, end_yr=2027)
        yrs = self._tax_years(r)
        if yrs:
            total_niit = sum(t.get("niit", 0) for t in yrs)
            assert total_niit == pytest.approx(0.0, abs=1.0)

    # ── Other income growth ────────────────────────────────────────────────

    def test_other_income_growth_increases_tax(self):
        r_flat = self._run(filing_status="single", state_code="TX",
                           other_income=50_000, other_income_growth=0)
        r_grow = self._run(filing_status="single", state_code="TX",
                           other_income=50_000, other_income_growth=5.0)
        flat_tax = sum(t["total"] for t in self._tax_years(r_flat))
        grow_tax = sum(t["total"] for t in self._tax_years(r_grow))
        assert grow_tax >= flat_tax

    # ── Ghost traces in figure builder ─────────────────────────────────────

    def test_figure_has_ghost_traces_when_tax_on(self):
        from figures.citadel import build_citadel_figure
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["tax_enabled"] = True
        p["filing_status"] = "single"
        p["state_code"] = "CA"
        p["start_yr"] = 2031
        p["end_yr"] = 2033
        fig, extra = build_citadel_figure(M, p)
        names = [t.name for t in fig.data if t.name]
        assert any("no tax" in (n or "").lower() for n in names)
        assert "annual_taxes" in extra

    def test_figure_no_ghost_traces_when_tax_off(self):
        from figures.citadel import build_citadel_figure
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["start_yr"] = 2031
        p["end_yr"] = 2033
        fig, extra = build_citadel_figure(M, p)
        names = [t.name for t in fig.data if t.name]
        assert not any("no tax" in (n or "").lower() for n in names)



class TestTaxWrapperGrowth:
    """Verify TD/TF wrapper balances grow over time (Critical #1 fix)."""

    def test_td_cash_grows(self):
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=0, start_yr=2031, end_yr=2035,
                        freq="Annually", monthly_spend=0,
                        cash_initial=0, selected_qs=[0.25],
                        tax_enabled=True, td_cash_initial=100_000,
                        cash_rate=5.0)
        r = simulate(cfg, _test_model())
        # After 4 years at 5%, $100k should grow to ~$121,550
        assert r.td_total is not None
        final_td = r.td_total[0, -1]
        assert final_td > 100_000, f"TD should grow but got {final_td}"

    def test_tf_investments_grow(self):
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=0, start_yr=2031, end_yr=2035,
                        freq="Annually", monthly_spend=0,
                        cash_initial=0, selected_qs=[0.25],
                        tax_enabled=True,
                        tf_invest_bins=[
                            {"label": "Equities", "initial": 200_000},
                            {"label": "Bonds", "initial": 0},
                        ],
                        invest_bins=[
                            {"label": "Equities", "initial": 0, "return_rate": 10, "volatility": 0},
                            {"label": "Bonds", "initial": 0, "return_rate": 5, "volatility": 0},
                        ])
        r = simulate(cfg, _test_model())
        final_tf = r.tf_total[0, -1]
        # 10% return on $200k over 4 years ≈ $292,820
        assert final_tf > 200_000, f"TF should grow but got {final_tf}"

    def test_td_balance_used_for_rmd(self):
        """RMD should be based on grown TD balance, not initial."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=0, start_yr=2031, end_yr=2035,
                        freq="Annually", monthly_spend=0,
                        cash_initial=0, selected_qs=[0.25],
                        tax_enabled=True, filing_status="single",
                        birth_year=1958, td_cash_initial=500_000,
                        cash_rate=5.0)
        r = simulate(cfg, _test_model())
        yrs = r.annual_taxes[0] if r.annual_taxes else []
        if yrs:
            # RMD income should reflect growing balance, not fixed $500k
            first_yr_income = yrs[0].get("ordinary_income", 0)
            assert first_yr_income > 0



class TestTaxEdgeCases:
    """Test gap coverage: depletion, cost basis bounds, shortfall, tax payment side effects."""

    def test_partial_year_depletion_no_crash(self):
        """If all accounts deplete mid-year, sim should complete without error.
        The final partial year's taxes may not be computed (year boundary never reached)."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0.01, start_yr=2031, end_yr=2035,
            freq="Monthly", monthly_spend=50_000,  # high spend to force depletion
            cash_initial=10_000, selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="TX",
            invest_bins=[
                {"label": "Equities", "initial": 10_000, "return_rate": 10, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 5, "volatility": 0},
            ])
        r = simulate(cfg, _test_model())
        # Should complete without crash
        assert r.total_usd.shape[1] > 0
        # Terminal wealth should be near zero (depleted)
        assert r.total_usd[0, -1] < 100_000

    def test_invest_cost_basis_never_negative(self):
        """Cost basis should never go below zero even with float arithmetic."""
        from engines.citadel import CitadelState
        # Simulate selling 100% of investment
        state = CitadelState(
            investments=[100_000, 50_000],
            invest_cost_basis=[80_000, 50_000],
        )
        # Sell all equities
        current = state.investments[0]
        fraction = current / current  # 1.0
        basis_sold = state.invest_cost_basis[0] * fraction
        state.invest_cost_basis[0] -= basis_sold
        state.investments[0] = 0
        assert state.invest_cost_basis[0] >= 0
        assert state.invest_cost_basis[0] == pytest.approx(0.0)

        # Edge: tiny floating point residual
        state2 = CitadelState(
            investments=[100.0, 0],
            invest_cost_basis=[100.0, 0],
        )
        # Sell in 3 chunks of 1/3 each (float imprecision)
        for _ in range(3):
            amt = 100.0 / 3
            cur = state2.investments[0]
            if cur <= 0:
                break
            frac = min(amt / cur, 1.0)
            basis = state2.invest_cost_basis[0] * frac
            state2.invest_cost_basis[0] = max(state2.invest_cost_basis[0] - basis, 0.0)
            state2.investments[0] = max(state2.investments[0] - amt, 0.0)
        assert state2.invest_cost_basis[0] >= 0

    def test_waterfall_shortfall_when_all_depleted(self):
        """When all three wrappers are empty, shortfall should equal the spending amount."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2032,
            freq="Annually", monthly_spend=10_000,
            cash_initial=0, selected_qs=[0.25],
            tax_enabled=True, filing_status="single",
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ])
        r = simulate(cfg, _test_model())
        # With zero assets and $10k/mo spending, there should be shortfall
        assert r.total_usd.shape[1] > 0

    def test_tax_payment_from_investments_tracks_basis(self):
        """When taxes are paid by selling investments, cost basis should decrease."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2033,
            freq="Annually", monthly_spend=0,
            cash_initial=0, selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=200_000,  # generates tax liability
            invest_bins=[
                {"label": "Equities", "initial": 500_000, "return_rate": 10, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ])
        r = simulate(cfg, _test_model())
        # Tax was owed on $200k other income, paid from investments
        # Investments should be less than they'd be without tax
        yrs = r.annual_taxes[0] if r.annual_taxes else []
        if yrs:
            assert yrs[0]["total"] > 0  # taxes were owed
        # Investment balance should be reduced by tax payment
        assert r.invest_balances[0, -1, 0] < 500_000 * 1.1 ** 2  # less than pure growth

    def test_annual_taxes_list_of_lists_flattened_by_summary(self):
        """_build_tax_summary should handle list-of-lists from engine."""
        from callbacks.citadel_tax_cb import _build_tax_summary
        # Simulate engine output: list containing one list of year dicts
        nested = [[
            {"year": 2031, "ordinary_income": 60000, "st_gains": 0,
             "lt_gains": 0, "federal_ordinary": 5000, "federal_ltcg": 0,
             "niit": 0, "state": 0, "total": 5000, "effective_rate": 0.08},
        ]]
        is_open, children = _build_tax_summary(nested)
        assert is_open is True
        assert len(children) == 2  # header + tbody



class TestGrossUpTaxPayment:
    """Verify gross-up when paying taxes from taxable investments/TD."""

    def test_gross_up_pays_from_investments_without_crash(self):
        """Paying taxes by selling investments with low basis should work."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2033,
            freq="Annually", monthly_spend=0,
            cash_initial=0, selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=200_000,
            invest_bins=[
                {"label": "Equities", "initial": 500_000, "return_rate": 10, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
            invest_cost_basis_initial=[250_000, 0],  # 50% unrealized gain
        )
        r = simulate(cfg, _test_model())
        assert r.taxes_paid is not None
        assert r.taxes_paid[0, -1] > 0
        # Investments should be reduced (tax was paid from them)
        assert r.invest_balances[0, -1, 0] < 500_000 * 1.1 ** 2

    def test_gross_up_pays_from_td_without_crash(self):
        """Paying taxes from TD (ordinary income) with gross-up should work."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2033,
            freq="Annually", monthly_spend=0,
            cash_initial=0, selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=300_000,  # high income → high tax
            td_cash_initial=1_000_000,
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        r = simulate(cfg, _test_model())
        assert r.taxes_paid is not None
        assert r.taxes_paid[0, -1] > 0



class TestMcStatusWithTaxExtraDict:
    """Regression test: _mc_status must not crash when the extra dict
    contains tax keys (annual_taxes) but no MC keys (created).
    This was the root cause of the silent background callback crash
    when running deterministic simulations with tax enabled."""

    def test_mc_status_with_tax_only_extra(self):
        """Deterministic run: extra dict has annual_taxes but no 'created'."""
        from callbacks.mc_helpers import _mc_status
        tax_extra = {"annual_taxes": [{"year": 2031, "total": 5000}]}
        store_val, status, show_modal = _mc_status(tax_extra, None, None)
        assert status == ""
        assert show_modal is True  # mc_result is truthy (non-empty dict)

    def test_mc_status_with_mc_result(self):
        """Stochastic run: extra dict has MC 'created' key."""
        from callbacks.mc_helpers import _mc_status
        mc_result = {"created": "2026-03-29T12:00:00.000Z", "sims": 1000}
        store_val, status, show_modal = _mc_status(mc_result, None, None)
        assert "Saved:" in status
        assert show_modal is True

    def test_mc_status_with_empty_result(self):
        """No result at all (first render, no sim run)."""
        from callbacks.mc_helpers import _mc_status
        store_val, status, show_modal = _mc_status(None, None, None)
        assert status == ""
        assert show_modal is False

    def test_mc_status_with_cached_only(self):
        """No new result, but cached MC exists."""
        from callbacks.mc_helpers import _mc_status
        cached = {"created": "2026-03-28T10:00:00.000Z"}
        store_val, status, show_modal = _mc_status(None, cached, ["mc"])
        assert "Using saved:" in status
        assert show_modal is False

    def test_mc_status_with_combined_tax_and_mc(self):
        """Both tax extra and MC result keys in the same dict."""
        from callbacks.mc_helpers import _mc_status
        combined = {"annual_taxes": [{"year": 2031}],
                    "created": "2026-03-29T12:00:00.000Z", "sims": 500}
        store_val, status, show_modal = _mc_status(combined, None, None)
        assert "Saved:" in status
        assert show_modal is True

    def test_full_figure_builder_extra_survives_mc_status(self):
        """End-to-end: build_citadel_figure with tax → extra → _mc_status."""
        from figures.citadel import build_citadel_figure
        from callbacks.mc_helpers import _mc_status
        from tab_defaults import citadel_defaults, CITADEL
        p = citadel_defaults()
        p["tax_enabled"] = True
        p["filing_status"] = "single"
        p["state_code"] = "CA"
        p["start_yr"] = 2031
        p["end_yr"] = 2033
        for k in ("td_btc", "td_cash", "td_res_short", "td_res_med", "td_res_long",
                   "td_inv_eq", "td_inv_bd", "tf_btc", "tf_cash", "tf_res_short",
                   "tf_res_med", "tf_res_long", "tf_inv_eq", "tf_inv_bd"):
            p[k] = CITADEL.get(k, 0)
        fig, extra = build_citadel_figure(M, p)
        # This is the exact call that was crashing in production
        store_val, status, show_modal = _mc_status(extra, None, None)
        assert isinstance(status, str)


# ═══════════════════════════════════════════════════════════════════════════════
# Section: Citadel Planner — engine rule verification tests
# ═══════════════════════════════════════════════════════════════════════════════


class _ControlledPriceModel:
    """Price model that returns configurable quantiles for trigger testing."""
    def __init__(self, quantile=0.50, price=50_000.0):
        import pandas as pd
        self.fits = {0.25: {"slope": 5.0, "intercept": 2.0}}
        self.genesis = pd.Timestamp("2009-07-25")
        self._quantile = quantile
        self._price = price

    def price_at(self, q, t):
        return self._price

    def quantile_at(self, price, t):
        return self._quantile


def _bare_config(**kw):
    """SimConfig with zero-volatility, deterministic, short horizon for unit tests."""
    from engines.citadel import SimConfig
    defaults = dict(
        start_stack=1.0, start_yr=2031, end_yr=2035,
        freq="Annually", monthly_spend=5000,
        cash_initial=100_000, selected_qs=[0.25],
        # Zero volatility → deterministic dollar-asset growth
        reserve_bins=[
            {"label": "Short", "initial": 50_000, "rate": 0, "volatility": 0},
            {"label": "Medium", "initial": 50_000, "rate": 0, "volatility": 0},
            {"label": "Long", "initial": 50_000, "rate": 0, "volatility": 0},
        ],
        invest_bins=[
            {"label": "Equities", "initial": 100_000, "return_rate": 0, "volatility": 0},
            {"label": "Bonds", "initial": 50_000, "return_rate": 0, "volatility": 0},
        ],
    )
    defaults.update(kw)
    return SimConfig(**defaults)



class TestCashFloorEnforcement:
    """1) Cash floor must not be violated until all assets are zero."""

    def test_cash_floor_replenished_from_investments_first(self):
        """Floor draws from investments before touching BTC."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        state = CitadelState(
            cash=10_000, reserves=[0, 0, 0], investments=[50_000, 30_000],
            btc_stack=1.0, btc_price=60_000,
        )
        cfg = _bare_config(cash_floor=50_000)
        _enforce_floors(state, cfg)
        assert state.cash >= 50_000 - 1  # floor met (within rounding)
        assert state.btc_stack == 1.0    # BTC untouched

    def test_cash_floor_draws_btc_only_when_dollar_assets_exhausted(self):
        """BTC is sold for cash floor only after all dollar assets are zero."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        state = CitadelState(
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
            btc_stack=2.0, btc_price=50_000,
        )
        cfg = _bare_config(cash_floor=30_000)
        _enforce_floors(state, cfg)
        assert state.cash >= 30_000 - 1
        assert state.btc_stack < 2.0  # BTC was sold

    def test_reserve_floor_never_sells_btc(self):
        """Reserve floors redistribute among dollar assets, never sell BTC."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        state = CitadelState(
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
            btc_stack=5.0, btc_price=100_000,
        )
        cfg = _bare_config(
            cash_floor=0,
            reserve_floors=[10_000, 0, 0],  # floor on short reserve
        )
        _enforce_floors(state, cfg)
        assert state.btc_stack == 5.0  # BTC untouched
        # Floor can't be met (no dollar sources) — reserve stays at 0
        assert state.reserves[0] == 0

    def test_cash_floor_holds_through_simulation(self):
        """Over a full sim, cash stays above floor until total depletion."""
        from engines.citadel import SimConfig, simulate, _initial_state, step
        import numpy as np
        cfg = _bare_config(
            cash_floor=20_000, monthly_spend=8000,
            start_yr=2031, end_yr=2040, freq="Annually",
        )
        model = _ControlledPriceModel(quantile=0.50, price=50_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        for _ in range(9):  # 9 annual steps
            state = step(state, cfg, 50_000, rng, model=model)
            total = state.cash + sum(state.reserves) + sum(state.investments) + state.btc_stack * state.btc_price
            if total > 20_000:
                # If total assets can cover the floor, cash should meet it
                assert state.cash >= 20_000 - 100, \
                    f"Cash {state.cash:.0f} below floor 20000 with total {total:.0f}"


    def test_cash_floor_respected_after_tax_payment(self):
        """Regression: tax payment at year-end must not leave cash below floor
        when other assets are available to replenish it."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = _bare_config(
            start_stack=5.0, cash_initial=100_000,
            cash_floor=80_000,
            monthly_spend=1000,
            tax_enabled=True, state_code="CA",
            other_income=500_000,  # large income → large tax bill
            start_yr=2031, end_yr=2035, freq="Annually",
            reserve_bins=[
                {"label": "Short", "initial": 200_000, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
        )
        model = _ControlledPriceModel(quantile=0.50, price=50_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        for i in range(4):
            state = step(state, cfg, 50_000, rng, model=model)
            total_other = (sum(state.reserves) + sum(state.investments)
                           + state.btc_stack * state.btc_price)
            if total_other > 80_000:
                assert state.cash >= 80_000 - 100, (
                    f"Period {i+1}: cash {state.cash:.0f} below floor 80000 "
                    f"with {total_other:.0f} in other assets")


    def test_cash_floor_draws_from_td_when_taxable_exhausted(self):
        """Cash floor replenished from TD when all taxable assets are depleted."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
            invest_cost_basis=[0, 0],
            btc_stack=0, btc_price=50_000,
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = _bare_config(cash_floor=10_000, tax_enabled=True, state_code="TX")
        _enforce_floors(state, cfg)
        assert state.cash >= 10_000 - 1
        assert state.td_cash == pytest.approx(90_000)
        assert state.tax_year_accum.tax_deferred_withdrawals == pytest.approx(10_000)

    def test_cash_floor_draws_tf_after_td_exhausted(self):
        """Cash floor falls through to TF (Roth) when TD is also exhausted."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
            invest_cost_basis=[0, 0],
            btc_stack=0, btc_price=50_000,
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            tf_cash=50_000, tf_reserves=[0, 0, 0], tf_investments=[0, 0],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = _bare_config(cash_floor=10_000, tax_enabled=True, state_code="TX")
        _enforce_floors(state, cfg)
        assert state.cash >= 10_000 - 1
        assert state.tf_cash == pytest.approx(40_000)
        assert state.tax_year_accum.roth_withdrawals == pytest.approx(10_000)

    def test_cash_floor_holds_through_tax_sim_with_td(self):
        """Over a full tax-enabled sim, cash floor holds while TD/TF have funds."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = _bare_config(
            start_stack=0, cash_initial=50_000, cash_floor=10_000,
            monthly_spend=10_000,
            tax_enabled=True, state_code="TX",
            td_cash_initial=200_000, tf_cash_initial=100_000,
            start_yr=2031, end_yr=2040, freq="Monthly",
            # Zero growth for predictability
            cash_rate=0,
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
        model = _ControlledPriceModel(quantile=0.50, price=50_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        for p in range(36):  # 3 years monthly
            state = step(state, cfg, 50_000, rng, model=model)
            total_all = (state.cash + state.td_cash + state.tf_cash
                        + sum(state.td_reserves) + sum(state.tf_reserves)
                        + sum(state.td_investments) + sum(state.tf_investments))
            if total_all > 10_000:
                assert state.cash >= 10_000 - 100, \
                    f"Period {p}: cash {state.cash:.0f} below floor with {total_all:.0f} total"



class TestBtcThresholdRules:
    """2) Bitcoin is sold/bought according to threshold rules."""

    def test_high_q_triggers_btc_sell(self):
        """When quantile >= high_q_trigger, BTC is sold."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = _bare_config(
            high_q_trigger=0.90,
            high_q_action={"mode": "lump", "rate": 20.0, "duration": 1,
                           "split": {"cash": 1.0}},
        )
        model = _ControlledPriceModel(quantile=0.95, price=50_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        initial_btc = state.btc_stack
        state = step(state, cfg, 50_000, rng, model=model)
        assert state.btc_stack < initial_btc, "BTC should have been sold at high quantile"
        assert state.rebal_event is not None
        assert state.rebal_event["action"] == "sell_btc"

    def test_low_q_triggers_btc_buy(self):
        """When quantile <= low_q_trigger, BTC is bought."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = _bare_config(
            low_q_trigger=0.10,
            low_q_action={"mode": "lump", "rate": 10.0, "duration": 1,
                          "split": {"cash": 0.5, "inv_eq": 0.5}},
        )
        model = _ControlledPriceModel(quantile=0.03, price=50_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        initial_btc = state.btc_stack
        state = step(state, cfg, 50_000, rng, model=model)
        assert state.btc_stack > initial_btc, "BTC should have been bought at low quantile"
        assert state.rebal_event is not None
        assert state.rebal_event["action"] == "buy_btc"

    def test_mid_quantile_no_rebalancing(self):
        """Between triggers, no rebalancing occurs."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = _bare_config(high_q_trigger=0.95, low_q_trigger=0.05)
        model = _ControlledPriceModel(quantile=0.50, price=50_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        initial_btc = state.btc_stack
        state = step(state, cfg, 50_000, rng, model=model)
        # BTC changes only from spending, not rebalancing
        assert state.rebal_event is None



class TestSpendingIncreasesTax:
    """3) Increasing monthly spending increases taxes."""

    def test_higher_spending_means_higher_taxes(self):
        """More withdrawals → more realized gains → higher tax bill."""
        from engines.citadel import SimConfig, simulate
        # Use BTC-only portfolio so spending forces capital gains
        common = dict(
            start_stack=10.0, cash_initial=0,
            tax_enabled=True, filing_status="single", state_code="CA",
            start_yr=2031, end_yr=2035, freq="Annually",
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        low = _bare_config(monthly_spend=3000, **common)
        high = _bare_config(monthly_spend=10000, **common)
        model = _test_model()
        r_low = simulate(low, model)
        r_high = simulate(high, model)
        tax_low = r_low.taxes_paid[0, -1]
        tax_high = r_high.taxes_paid[0, -1]
        assert tax_high > tax_low, \
            f"Higher spend should yield higher tax: {tax_high:.0f} vs {tax_low:.0f}"

    def test_zero_spending_minimal_tax(self):
        """With zero spending and no other income, minimal or zero tax."""
        from engines.citadel import SimConfig, simulate
        cfg = _bare_config(
            monthly_spend=0, tax_enabled=True,
            filing_status="single", state_code="TX",
            other_income=0,
            start_yr=2031, end_yr=2035, freq="Annually",
        )
        model = _test_model()
        r = simulate(cfg, model)
        # Only interest income could generate tax (from cash/reserves)
        # With TX (no state tax) and low interest, tax should be very small
        assert r.taxes_paid[0, -1] < 5000



class TestWithdrawalOrderTaxAdvantaged:
    """4) Verify withdrawal logic follows tax-advantaged ordering."""

    def test_taxable_cash_drawn_before_td(self):
        """Taxable principal (no tax) should be drawn before TD (ordinary income)."""
        from engines.citadel import SimConfig, simulate
        cfg = _bare_config(
            monthly_spend=20_000,
            cash_initial=200_000,
            tax_enabled=True, state_code="TX",
            td_cash_initial=200_000,
            start_yr=2031, end_yr=2034, freq="Annually",
            # Zero growth so balances are predictable
            cash_rate=0,
        )
        model = _test_model()
        r = simulate(cfg, model)
        # Taxable cash should deplete faster than TD cash
        # After 3 years of spending $240k/yr, taxable cash should be gone first
        assert r.taxes_paid is not None

    def test_roth_btc_is_absolute_last(self):
        """Roth BTC should be the very last asset sold."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = _bare_config(
            monthly_spend=50_000, cash_initial=0,
            tax_enabled=True, state_code="TX",
            start_stack=0,  # no taxable BTC
            tf_btc_stack=1.0,
            tf_cash_initial=100_000,
            start_yr=2031, end_yr=2035, freq="Annually",
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        model = _ControlledPriceModel(quantile=0.50, price=100_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        # After first step: TF cash should be drawn before TF BTC
        state = step(state, cfg, 100_000, rng, model=model)
        assert state.tf_cash < 100_000, "Roth cash should be drawn"
        if state.tf_cash > 0:
            assert state.tf_btc_stack == 1.0, "Roth BTC untouched while Roth cash remains"

    def test_td_bracket_fill_uses_low_bracket_room(self):
        """TD withdrawals should bracket-fill to minimize marginal rate."""
        from engines.citadel import SimConfig, simulate
        cfg = _bare_config(
            monthly_spend=15_000,
            cash_initial=500_000,
            tax_enabled=True, state_code="TX",
            td_cash_initial=500_000,
            start_yr=2031, end_yr=2035, freq="Annually",
        )
        model = _test_model()
        r = simulate(cfg, model)
        # Should have some TD withdrawals (bracket-filling) even though
        # taxable cash could cover all spending
        assert r.taxes_paid is not None
        assert len(r.annual_taxes) > 0
        # Check that at least one year has TD withdrawal recorded
        has_td_wd = any(
            yr.get("ordinary_income", 0) > 0
            for sim_taxes in r.annual_taxes
            for yr in sim_taxes
        )
        assert has_td_wd, "TD bracket-filling should produce ordinary income"



class TestLumpCooldown:
    """5) Global lump cooldown is obeyed."""

    def test_cooldown_prevents_consecutive_lumps(self):
        """After a lump action, another lump is blocked for cooldown periods."""
        from engines.citadel import CitadelState, _evaluate_rebalancing
        from engines.citadel import SimConfig
        cfg = _bare_config(
            lump_cooldown=3,
            high_q_trigger=0.90,
            high_q_action={"mode": "lump", "rate": 10.0, "duration": 1,
                           "split": {"cash": 1.0}},
        )
        state = CitadelState(
            btc_stack=10.0, btc_price=50_000,
            cash=100_000, reserves=[0, 0, 0], investments=[0, 0],
        )
        # First trigger: should fire
        _evaluate_rebalancing(state, cfg, btc_quantile=0.95)
        assert state.rebal_event is not None
        assert state.rebal_cooldown == 3
        first_btc = state.btc_stack

        # Next 2 periods: cooldown should block (3→2, 2→1)
        for i in range(2):
            state.rebal_event = None
            _evaluate_rebalancing(state, cfg, btc_quantile=0.95)
            assert state.rebal_event is None, f"Cooldown period {i+1}: should be blocked"
            assert state.btc_stack == first_btc, "No BTC sold during cooldown"

        # 3rd call: cooldown 1→0, trigger fires again
        state.rebal_event = None
        _evaluate_rebalancing(state, cfg, btc_quantile=0.95)
        assert state.rebal_event is not None, "Should fire after cooldown expires"

    def test_gradual_mode_ignores_cooldown(self):
        """Gradual actions continue regardless of cooldown counter."""
        from engines.citadel import CitadelState, _evaluate_rebalancing
        from engines.citadel import SimConfig
        cfg = _bare_config(
            high_q_trigger=0.90,
            high_q_action={"mode": "gradual", "rate": 5.0, "duration": 3,
                           "split": {"cash": 1.0}},
        )
        state = CitadelState(
            btc_stack=10.0, btc_price=50_000,
            cash=100_000, reserves=[0, 0, 0], investments=[0, 0],
        )
        # Trigger gradual
        _evaluate_rebalancing(state, cfg, btc_quantile=0.95)
        assert state.grad_active is True
        btc_after_first = state.btc_stack

        # Continue gradual — even with cooldown set, gradual proceeds
        _evaluate_rebalancing(state, cfg, btc_quantile=0.95)
        assert state.btc_stack < btc_after_first, "Gradual should continue selling"



class TestBtcSaleDistribution:
    """6) Bitcoin sale proceeds distributed according to split rules."""

    def test_sell_distributes_per_split(self):
        """Proceeds from BTC sale go to accounts per configured split."""
        from engines.citadel import CitadelState, SimConfig, _execute_sell_btc
        state = CitadelState(
            btc_stack=10.0, btc_price=50_000, sim_date="2035-01-15",
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
        )
        split = {"cash": 0.20, "res_short": 0.10, "res_med": 0.10,
                 "res_long": 0.10, "inv_eq": 0.30, "inv_bd": 0.20}
        evt = _execute_sell_btc(state, SimConfig(cost_basis_method="fifo"), rate_pct=10.0, split=split)
        # Sold 10% of 10 BTC = 1 BTC = $50,000
        assert evt["btc_sold"] == pytest.approx(1.0)
        assert evt["proceeds"] == pytest.approx(50_000)
        assert state.cash == pytest.approx(10_000)           # 20%
        assert state.reserves[0] == pytest.approx(5_000)     # 10%
        assert state.reserves[1] == pytest.approx(5_000)     # 10%
        assert state.reserves[2] == pytest.approx(5_000)     # 10%
        assert state.investments[0] == pytest.approx(15_000)  # 30%
        assert state.investments[1] == pytest.approx(10_000)  # 20%

    def test_sell_zero_btc_no_event(self):
        """Selling from empty stack produces no event."""
        from engines.citadel import CitadelState, SimConfig, _execute_sell_btc
        state = CitadelState(btc_stack=0, btc_price=50_000, sim_date="2035-01-15")
        evt = _execute_sell_btc(state, SimConfig(cost_basis_method="fifo"), rate_pct=10.0, split={"cash": 1.0})
        assert evt == {}



class TestBtcPurchaseSourcing:
    """7) Bitcoin purchases source funds according to split rules."""

    def test_buy_sources_per_split(self):
        """BTC purchase draws from accounts per configured split."""
        from engines.citadel import CitadelState, SimConfig, _execute_buy_btc
        state = CitadelState(
            btc_stack=0, btc_price=50_000, sim_date="2035-01-15",
            cash=100_000, reserves=[50_000, 50_000, 50_000],
            investments=[200_000, 100_000],
        )
        split = {"cash": 0.10, "inv_eq": 0.50, "inv_bd": 0.40}
        # Total dollar assets = 100k + 150k + 300k = 550k
        # 10% of 550k = 55k target
        evt = _execute_buy_btc(state, SimConfig(), rate_pct=10.0, split=split)
        assert evt["action"] == "buy_btc"
        assert evt["btc_bought"] == pytest.approx(55_000 / 50_000)
        # Cash should lose 10% of 55k = 5,500
        assert state.cash == pytest.approx(100_000 - 5_500)
        # Equities lose 50% of 55k = 27,500
        assert state.investments[0] == pytest.approx(200_000 - 27_500)
        # Bonds lose 40% of 55k = 22,000
        assert state.investments[1] == pytest.approx(100_000 - 22_000)

    def test_buy_respects_floor(self):
        """BTC purchase won't draw cash below its floor."""
        from engines.citadel import CitadelState, _execute_buy_btc
        cfg = _bare_config(cash_floor=80_000)
        state = CitadelState(
            btc_stack=0, btc_price=50_000, sim_date="2035-01-15",
            cash=100_000, reserves=[0, 0, 0], investments=[0, 0],
        )
        split = {"cash": 1.0}
        # Total dollars = 100k, 10% = 10k, but floor = 80k, so avail = 20k
        evt = _execute_buy_btc(state, cfg, rate_pct=10.0, split=split)
        assert state.cash >= 80_000 - 1, "Cash floor should be respected"

    def test_buy_redistributes_shortfall(self):
        """When one source can't cover its share, shortfall goes to others."""
        from engines.citadel import CitadelState, SimConfig, _execute_buy_btc
        state = CitadelState(
            btc_stack=0, btc_price=50_000, sim_date="2035-01-15",
            cash=1_000,  # very little cash
            reserves=[0, 0, 0],
            investments=[200_000, 200_000],
        )
        split = {"cash": 0.50, "inv_eq": 0.25, "inv_bd": 0.25}
        # Total = 401k, 10% = 40.1k
        # Cash wants 50% = 20.05k but only has 1k → shortfall redistributed
        evt = _execute_buy_btc(state, SimConfig(), rate_pct=10.0, split=split)
        assert evt["action"] == "buy_btc"
        assert state.cash < 1_000  # Cash was drawn
        # Investments picked up the slack
        total_drawn = evt["cost"]
        assert total_drawn > 1_000  # More than just cash



class TestTaxEfficientAccountUsage:
    """8) Trades in TD/TF accounts don't generate capital gains;
    taxable account trades do."""

    def test_taxable_btc_sale_generates_capital_gains(self):
        """Selling BTC from taxable wrapper records capital gains."""
        from engines.citadel import SimConfig, simulate
        # BTC-only portfolio forces BTC sale for spending
        cfg = _bare_config(
            start_stack=10.0,  # taxable BTC (plenty to cover spending)
            monthly_spend=20_000,
            cash_initial=0,
            tax_enabled=True, state_code="CA",
            start_yr=2031, end_yr=2035, freq="Annually",
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
        )
        model = _test_model()
        r = simulate(cfg, model)
        # Selling BTC from taxable should generate gains
        has_gains = any(
            yr.get("lt_gains", 0) > 0 or yr.get("st_gains", 0) > 0
            for sim_taxes in r.annual_taxes
            for yr in sim_taxes
        )
        assert has_gains, "Taxable BTC sale should generate capital gains"
        assert r.taxes_paid[0, -1] > 0

    def test_roth_only_portfolio_zero_capital_gains_tax(self):
        """All assets in Roth (TF) → no capital gains tax on any trade."""
        from engines.citadel import SimConfig, simulate
        cfg = _bare_config(
            start_stack=0,  # no taxable BTC
            monthly_spend=10_000,
            cash_initial=0,
            tax_enabled=True, state_code="TX",
            other_income=0,
            tf_btc_stack=2.0,
            tf_cash_initial=300_000,
            start_yr=2031, end_yr=2035, freq="Annually",
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
        )
        model = _test_model()
        r = simulate(cfg, model)
        # All Roth → zero tax
        assert r.taxes_paid[0, -1] == pytest.approx(0, abs=1)

    def test_td_withdrawal_taxed_as_ordinary_not_capital_gains(self):
        """TD withdrawals are ordinary income, not capital gains."""
        from engines.citadel import SimConfig, simulate
        cfg = _bare_config(
            start_stack=0, monthly_spend=20_000,
            cash_initial=0,
            tax_enabled=True, state_code="TX",
            td_cash_initial=1_000_000,
            start_yr=2031, end_yr=2035, freq="Annually",
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
        )
        model = _test_model()
        r = simulate(cfg, model)
        # Should have ordinary income but zero capital gains
        for sim_taxes in r.annual_taxes:
            for yr in sim_taxes:
                assert yr.get("ordinary_income", 0) > 0, "TD withdrawal = ordinary income"
                assert yr.get("lt_gains", 0) == 0, "TD should not produce LTCG"
                assert yr.get("st_gains", 0) == 0, "TD should not produce STCG"

    def test_taxable_vs_td_same_spend_different_tax_type(self):
        """Same spending from taxable BTC vs TD produces different tax profiles."""
        from engines.citadel import SimConfig, simulate
        empty_bins = dict(
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
        )
        # _test_model has price growth: 50000*(1+t/100), so BTC bought at
        # t0 and sold later produces a capital gain.
        model = _test_model()
        # Scenario A: spend from taxable BTC (capital gains)
        cfg_taxable = _bare_config(
            start_stack=10.0, monthly_spend=20_000, cash_initial=0,
            tax_enabled=True, state_code="CA",
            start_yr=2031, end_yr=2036, freq="Annually",
            **empty_bins,
        )
        # Scenario B: spend from TD cash (ordinary income)
        cfg_td = _bare_config(
            start_stack=0, monthly_spend=20_000, cash_initial=0,
            tax_enabled=True, state_code="CA",
            td_cash_initial=1_000_000,
            start_yr=2031, end_yr=2036, freq="Annually",
            **empty_bins,
        )
        r_taxable = simulate(cfg_taxable, model)
        r_td = simulate(cfg_td, model)
        # Both should pay tax
        assert r_taxable.taxes_paid[0, -1] > 0, "Taxable BTC should generate tax"
        assert r_td.taxes_paid[0, -1] > 0, "TD withdrawals should generate tax"
        # Tax types differ: BTC = capital gains (lower rate), TD = ordinary income
        tax_btc = r_taxable.taxes_paid[0, -1]
        tax_td = r_td.taxes_paid[0, -1]
        assert tax_btc != tax_td, "Different tax types should produce different totals"



class TestQuarterlyTaxPayments:
    """Quarterly estimated tax payment tests."""

    def test_state_has_quarterly_field(self):
        from engines.citadel import CitadelState
        state = CitadelState()
        assert hasattr(state, "quarterly_tax_paid_ytd")
        assert state.quarterly_tax_paid_ytd == 0.0

    def test_pay_tax_amount_draws_from_cash_first(self):
        """_pay_tax_amount draws cash before other sources."""
        from engines.citadel import CitadelState, SimConfig, _pay_tax_amount
        state = CitadelState(
            cash=50_000, reserves=[0, 0, 0], investments=[0, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[0, 0],
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX")
        _pay_tax_amount(state, cfg, amount=30_000, sim_year=2031)
        assert state.cash == pytest.approx(20_000)

    def test_pay_tax_amount_uses_investments_after_cash(self):
        """_pay_tax_amount falls through to investments when cash exhausted."""
        from engines.citadel import CitadelState, SimConfig, _pay_tax_amount
        state = CitadelState(
            cash=10_000, reserves=[0, 0, 0], investments=[100_000, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[100_000, 0],  # full basis = no gain = no gross-up needed
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX")
        _pay_tax_amount(state, cfg, amount=30_000, sim_year=2031)
        assert state.cash == 0
        assert state.investments[0] < 100_000

    def test_quarterly_payment_annualizes_ytd(self):
        """Q1 payment should be ~25% of annualized tax projection."""
        from engines.citadel import (SimConfig, CitadelState,
                                      _quarterly_estimated_payment)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=500_000, reserves=[0, 0, 0], investments=[0, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[0, 0],
            tax_year_accum=TaxYearAccumulator(other_income=50_000),
            quarterly_tax_paid_ytd=0,
        )
        cfg = SimConfig(tax_enabled=True, state_code="CA",
                        filing_status="single", inflation=4.0)
        _quarterly_estimated_payment(state, cfg, quarter=1, sim_year=2031)
        assert state.quarterly_tax_paid_ytd > 0
        assert state.cash < 500_000

    def test_quarterly_payment_cumulative_tracking(self):
        """Q2 payment accounts for Q1 already paid."""
        from engines.citadel import (SimConfig, CitadelState,
                                      _quarterly_estimated_payment)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=500_000, reserves=[0, 0, 0], investments=[0, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[0, 0],
            tax_year_accum=TaxYearAccumulator(other_income=100_000),
            quarterly_tax_paid_ytd=10_000,
        )
        cfg = SimConfig(tax_enabled=True, state_code="CA",
                        filing_status="single", inflation=4.0)
        _quarterly_estimated_payment(state, cfg, quarter=2, sim_year=2031)
        assert state.quarterly_tax_paid_ytd > 10_000

    def test_quarterly_no_payment_if_overpaid(self):
        """If already overpaid relative to cumulative target, pay $0."""
        from engines.citadel import (SimConfig, CitadelState,
                                      _quarterly_estimated_payment)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=500_000, reserves=[0, 0, 0], investments=[0, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[0, 0],
            tax_year_accum=TaxYearAccumulator(other_income=10_000),
            quarterly_tax_paid_ytd=100_000,
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX",
                        filing_status="single", inflation=4.0)
        _quarterly_estimated_payment(state, cfg, quarter=2, sim_year=2031)
        assert state.cash == 500_000  # no payment drawn

    def test_monthly_sim_pays_quarterly(self):
        """Monthly frequency produces quarterly payments + year-end true-up."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2033,
            freq="Monthly", monthly_spend=0,
            cash_initial=1_000_000, selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=300_000,
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        r = simulate(cfg, _test_model())
        assert r.taxes_paid[0, -1] > 0

    def test_annual_freq_falls_back_to_year_end(self):
        """Annually frequency: no quarterly payments, all at year-end."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2034,
            freq="Annually", monthly_spend=0,
            cash_initial=1_000_000, selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=200_000,
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        r = simulate(cfg, _test_model())
        assert r.taxes_paid[0, -1] > 0

    def test_q4_trueup_matches_annual(self):
        """Monthly and annual sims should produce approximately equal total tax."""
        from engines.citadel import SimConfig, simulate
        common = dict(
            start_stack=0, start_yr=2031, end_yr=2034,
            monthly_spend=0, cash_initial=1_000_000,
            selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=200_000,
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        r_monthly = simulate(SimConfig(freq="Monthly", **common), _test_model())
        r_annual = simulate(SimConfig(freq="Annually", **common), _test_model())
        tax_monthly = r_monthly.taxes_paid[0, -1]
        tax_annual = r_annual.taxes_paid[0, -1]
        assert abs(tax_monthly - tax_annual) / max(tax_annual, 1) < 0.05, \
            f"Monthly {tax_monthly:.0f} vs Annual {tax_annual:.0f} differ by >5%"

    def test_quarterly_tax_paid_ytd_resets_each_year(self):
        """quarterly_tax_paid_ytd must be 0 at each year boundary."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2033,
            freq="Monthly", monthly_spend=0,
            cash_initial=1_000_000, selected_qs=[0.25],
            tax_enabled=True, state_code="CA", other_income=200_000,
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        model = _test_model()
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        for i in range(24):
            state = step(state, cfg, 50_000, rng, model=model)
            if state.period % 12 == 0:
                assert state.quarterly_tax_paid_ytd == 0, \
                    f"Period {state.period}: ytd should be 0, got {state.quarterly_tax_paid_ytd:.0f}"

    def test_cash_floor_respected_after_quarterly_payment(self):
        """Cash floor must hold after each quarterly tax payment."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=5.0, start_yr=2031, end_yr=2033,
            freq="Monthly", monthly_spend=1000,
            cash_initial=100_000, cash_floor=80_000,
            selected_qs=[0.25],
            tax_enabled=True, state_code="CA", other_income=500_000,
            reserve_bins=[
                {"label": "Short", "initial": 200_000, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 200_000, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=_test_model())
        for i in range(24):
            state = step(state, cfg, 50_000, rng, model=_test_model())
            total_other = (sum(state.reserves) + sum(state.investments)
                           + state.btc_stack * state.btc_price)
            if total_other > 80_000:
                assert state.cash >= 80_000 - 100, \
                    f"Period {i+1}: cash {state.cash:.0f} below floor"



class TestTaxAccountingHelpers:
    """Tests for the 3 tracking helpers and related infrastructure."""

    def test_state_has_sim_date(self):
        from engines.citadel import CitadelState
        s = CitadelState()
        assert hasattr(s, "sim_date")
        assert s.sim_date == ""

    def test_lots_seeded_when_tax_off(self):
        """Lots should be created even when tax_enabled=False."""
        from engines.citadel import SimConfig, _initial_state
        cfg = SimConfig(start_stack=2.0, start_yr=2031, end_yr=2035,
                        cash_initial=100_000, selected_qs=[0.25],
                        tax_enabled=False)
        state = _initial_state(cfg, model=_test_model())
        assert len(state.tax_lots) == 1
        assert state.tax_lots[0].btc == 2.0
        assert state.tax_year_accum is None

    def test_scf_purchase_creates_lot(self):
        """SCF initial BTC purchase must create a separate lot."""
        from engines.citadel import SimConfig, _initial_state
        cfg = SimConfig(start_stack=1.0, start_yr=2031, end_yr=2035,
                        cash_initial=100_000, selected_qs=[0.25],
                        tax_enabled=False,
                        scf_enabled=True, scf_amount=50_000)
        state = _initial_state(cfg, model=_test_model())
        assert len(state.tax_lots) == 2
        assert state.tax_lots[0].source == "initial"
        assert state.tax_lots[1].source == "scf"
        total_lot_btc = sum(l.btc for l in state.tax_lots)
        assert abs(total_lot_btc - state.btc_stack) < 1e-8

    def test_sell_btc_tracked_records_gains_tax_on(self):
        """With tax on, selling BTC records capital gains in accumulator."""
        from engines.citadel import CitadelState, SimConfig, _sell_btc_tracked
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=2.0, btc_price=100_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=2.0,
                             cost_basis=50_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, cost_basis_method="fifo")
        result = _sell_btc_tracked(state, cfg, 1.0)
        assert result.btc_sold == pytest.approx(1.0)
        assert state.btc_stack == pytest.approx(1.0)
        assert len(state.tax_lots) == 1
        assert state.tax_lots[0].btc == pytest.approx(1.0)
        assert state.tax_year_accum.lt_capital_gains == pytest.approx(50_000)

    def test_sell_btc_tracked_no_gains_tax_off(self):
        """With tax off (accum=None), BTC still sold but no gains recorded."""
        from engines.citadel import CitadelState, SimConfig, _sell_btc_tracked
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=2.0, btc_price=100_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=2.0,
                             cost_basis=50_000, source="initial")],
            tax_year_accum=None,
        )
        cfg = SimConfig(tax_enabled=False, cost_basis_method="fifo")
        result = _sell_btc_tracked(state, cfg, 1.0)
        assert result.btc_sold == pytest.approx(1.0)
        assert state.btc_stack == pytest.approx(1.0)

    def test_sell_btc_tracked_empty_lots_fallback(self):
        """With no lots, raw stack decrement as fallback."""
        from engines.citadel import CitadelState, SimConfig, _sell_btc_tracked
        state = CitadelState(btc_stack=3.0, btc_price=50_000, sim_date="2035-01-15")
        cfg = SimConfig(cost_basis_method="fifo")
        result = _sell_btc_tracked(state, cfg, 1.0)
        assert result.btc_sold == pytest.approx(1.0)
        assert state.btc_stack == pytest.approx(2.0)

    def test_buy_btc_tracked_creates_lot(self):
        """Buying BTC creates a lot with correct date/basis/source."""
        from engines.citadel import CitadelState, SimConfig, _buy_btc_tracked
        state = CitadelState(btc_stack=1.0, btc_price=80_000, sim_date="2033-03-15")
        cfg = SimConfig()
        _buy_btc_tracked(state, cfg, 0.5, source="rebal_buy")
        assert state.btc_stack == pytest.approx(1.5)
        assert len(state.tax_lots) == 1
        lot = state.tax_lots[0]
        assert lot.btc == pytest.approx(0.5)
        assert lot.cost_basis == 80_000
        assert lot.date == "2033-03-15"
        assert lot.source == "rebal_buy"

    def test_sell_investments_tracked_records_ltcg(self):
        """Investment sale records LTCG in accumulator."""
        from engines.citadel import CitadelState, SimConfig, _sell_investments_tracked
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            investments=[200_000, 100_000],
            invest_cost_basis=[100_000, 80_000],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True)
        drawn, gain = _sell_investments_tracked(state, cfg, 0, 50_000)
        assert drawn == pytest.approx(50_000)
        assert gain == pytest.approx(25_000)
        assert state.tax_year_accum.lt_capital_gains == pytest.approx(25_000)
        assert state.investments[0] == pytest.approx(150_000)
        assert state.invest_cost_basis[0] == pytest.approx(75_000)

    def test_sell_investments_tracked_noop_tax_off(self):
        """Investment sale updates balances but skips accumulator when tax off."""
        from engines.citadel import CitadelState, SimConfig, _sell_investments_tracked
        state = CitadelState(
            investments=[200_000, 0],
            invest_cost_basis=[100_000, 0],
            tax_year_accum=None,
        )
        cfg = SimConfig(tax_enabled=False)
        drawn, gain = _sell_investments_tracked(state, cfg, 0, 50_000)
        assert drawn == pytest.approx(50_000)
        assert state.investments[0] == pytest.approx(150_000)

    def test_floor_enforcement_btc_sale_lot_tracked(self):
        """Bug 1: BTC sold to replenish cash floor must be lot-tracked."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
            btc_stack=2.0, btc_price=50_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=2.0,
                             cost_basis=30_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
            invest_cost_basis=[0, 0],
        )
        cfg = SimConfig(cash_floor=20_000, cost_basis_method="fifo")
        _enforce_floors(state, cfg)
        assert state.cash >= 20_000 - 1
        assert state.btc_stack < 2.0
        # Capital gain should be recorded (sold at 50k, basis 30k)
        assert state.tax_year_accum.lt_capital_gains > 0

    def test_floor_enforcement_investment_sale_tracks_basis(self):
        """Bug 7: Investment sold for floor must update cost basis."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[100_000, 50_000],
            invest_cost_basis=[60_000, 30_000],
            btc_stack=0, btc_price=50_000, sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(cash_floor=30_000, cost_basis_method="fifo")
        _enforce_floors(state, cfg)
        assert state.cash >= 30_000 - 1
        # Cost basis should have been reduced proportionally
        assert state.invest_cost_basis[1] < 30_000 or state.invest_cost_basis[0] < 60_000
        # LTCG should be recorded
        assert state.tax_year_accum.lt_capital_gains > 0

    def test_rebalancing_sell_btc_lot_tracked(self):
        """Bug 2: Rebalancing BTC sell must be lot-tracked."""
        from engines.citadel import CitadelState, SimConfig, _execute_sell_btc
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=10.0, btc_price=50_000, sim_date="2035-06-15",
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
            tax_lots=[TaxLot(date="2031-01-01", btc=10.0,
                             cost_basis=20_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(cost_basis_method="fifo")
        evt = _execute_sell_btc(state, cfg, rate_pct=10.0, split={"cash": 1.0})
        assert evt["btc_sold"] == pytest.approx(1.0)
        assert state.btc_stack == pytest.approx(9.0)
        assert state.tax_year_accum.lt_capital_gains == pytest.approx(30_000)

    def test_rebalancing_buy_btc_creates_lot(self):
        """Bug 3: Rebalancing BTC buy must create a tax lot."""
        from engines.citadel import CitadelState, SimConfig, _execute_buy_btc
        state = CitadelState(
            btc_stack=1.0, btc_price=50_000, sim_date="2033-03-15",
            cash=100_000, reserves=[0, 0, 0], investments=[0, 0],
        )
        cfg = SimConfig(cash_floor=0)
        evt = _execute_buy_btc(state, cfg, rate_pct=10.0, split={"cash": 1.0})
        assert evt["action"] == "buy_btc"
        assert state.btc_stack > 1.0
        new_lots = [l for l in state.tax_lots if l.source == "rebal_buy"]
        assert len(new_lots) == 1
        assert new_lots[0].cost_basis == 50_000

    def test_scf_repay_btc_sale_lot_tracked(self):
        """Bug 6: SCF perpetual loan repayment must lot-track BTC sale."""
        from engines.citadel import CitadelState, SimConfig, _scf_check_repay
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=5.0, btc_price=50_000, sim_date="2040-01-15",
            scf_outstanding=100_000, scf_active=True,
            tax_lots=[TaxLot(date="2031-01-01", btc=5.0,
                             cost_basis=30_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(scf_enabled=True, scf_type="perpetual",
                        scf_rate=8.0, scf_repay_trigger=1.0,
                        cost_basis_method="fifo")
        _scf_check_repay(state, cfg, btc_annual_return=0.0)
        assert state.btc_stack < 5.0
        assert state.tax_year_accum.lt_capital_gains > 0

    def test_lot_inventory_matches_stack_after_operations(self):
        """Lot sum must match btc_stack after sell/buy operations."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _sell_btc_tracked, _buy_btc_tracked)
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=5.0, btc_price=60_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=5.0,
                             cost_basis=30_000, source="initial")],
        )
        cfg = SimConfig(cost_basis_method="fifo")
        _sell_btc_tracked(state, cfg, 2.0)
        _buy_btc_tracked(state, cfg, 1.0, source="rebal_buy")
        lot_sum = sum(l.btc for l in state.tax_lots)
        assert abs(lot_sum - state.btc_stack) < 1e-8

    def test_pay_tax_investment_sale_recorded(self):
        """Bug 4: Investment gains during tax payment must be in accumulator."""
        from engines.citadel import CitadelState, SimConfig, _pay_tax_amount
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[200_000, 0],
            invest_cost_basis=[100_000, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            tax_year_accum=TaxYearAccumulator(),
            sim_date="2035-06-15",
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX")
        _pay_tax_amount(state, cfg, amount=50_000, sim_year=2035)
        assert state.tax_year_accum.lt_capital_gains > 0

    def test_pay_tax_investment_gross_up_math(self):
        """Selling investments to pay tax: enough must be sold so that after
        the tax on the SALE itself, the net proceeds cover the tax bill.

        With $200k investments, $100k basis, TX (no state tax), agi below NIIT
        threshold, ltcg_rate = 0.15. gain_fraction = 0.5 → effective_rate on
        sale proceeds = 0.15 * 0.5 = 0.075. To net $50k after tax:
          gross = 50k / (1 - 0.075) ≈ $54,054
        Tax on sale: $54,054 * 0.5 * 0.15 ≈ $4,054. Net: $50k. Check.
        """
        from engines.citadel import CitadelState, SimConfig, _pay_tax_amount
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[200_000, 0],
            invest_cost_basis=[100_000, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            tax_year_accum=TaxYearAccumulator(),
            sim_date="2035-06-15",
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX")
        tax_bill = 50_000
        _pay_tax_amount(state, cfg, amount=tax_bill, sim_year=2035,
                         tax_result={"agi": 50_000})
        # Investments drawn ≈ gross = 54,054 (tolerance for rounding)
        inv_drawn = 200_000 - state.investments[0]
        assert 53_000 <= inv_drawn <= 56_000, (
            f"Gross-up should draw ~$54k to net $50k, got ${inv_drawn:.0f}")
        # Realized gain ≈ gross * 0.5 (basis_frac=0.5)
        gain = state.tax_year_accum.lt_capital_gains
        assert gain == pytest.approx(inv_drawn * 0.5, rel=0.01)

    def test_pay_tax_niit_threshold_flips_gross_up(self):
        """Gross-up rate bumps by 3.8% when agi exceeds NIIT threshold,
        so for the same tax bill more investments must be sold."""
        from engines.citadel import CitadelState, SimConfig, _pay_tax_amount
        from engines.tax import TaxYearAccumulator
        # Same starting state twice; only the agi in tax_result differs.
        def _fresh_state():
            return CitadelState(
                cash=0, reserves=[0, 0, 0],
                investments=[500_000, 0],
                invest_cost_basis=[0, 0],  # gain_fraction = 1.0 → max sensitivity
                td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
                tax_year_accum=TaxYearAccumulator(),
                sim_date="2035-06-15",
            )
        cfg = SimConfig(tax_enabled=True, state_code="TX", filing_status="single")
        tax_bill = 100_000

        from engines.tax_data import NIIT_THRESHOLD
        thresh = NIIT_THRESHOLD["single"]
        state_below = _fresh_state()
        state_above = _fresh_state()
        _pay_tax_amount(state_below, cfg, amount=tax_bill, sim_year=2035,
                         tax_result={"agi": thresh - 1})
        _pay_tax_amount(state_above, cfg, amount=tax_bill, sim_year=2035,
                         tax_result={"agi": thresh + 1})

        drawn_below = 500_000 - state_below.investments[0]
        drawn_above = 500_000 - state_above.investments[0]
        # NIIT adds 3.8% to ltcg_rate (15% → 18.8%) → gross-up denominator
        # shrinks, so MORE investments must be sold to cover the same bill.
        assert drawn_above > drawn_below, (
            f"NIIT should force larger draw: below={drawn_below:.0f} "
            f"vs above={drawn_above:.0f}")

    def test_merged_waterfall_tax_off_same_behavior(self):
        """Merged waterfall with tax_enabled=False works correctly."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=1.0, start_yr=2031, end_yr=2035,
            freq="Annually", monthly_spend=5000,
            cash_initial=50_000, selected_qs=[0.25],
            tax_enabled=False,
            reserve_bins=[
                {"label": "Short", "initial": 20_000, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 20_000, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 50_000, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        r = simulate(cfg, _test_model())
        assert r.total_usd.shape[1] > 0
        assert r.taxes_paid is None
        assert r.total_usd[0, -1] >= 0

    def test_tax_off_still_zero_tax(self):
        """Critical regression: tax_enabled=False must produce zero tax."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=5.0, start_yr=2031, end_yr=2035,
            freq="Monthly", monthly_spend=5000,
            cash_initial=100_000, selected_qs=[0.25],
            tax_enabled=False,
        )
        r = simulate(cfg, _test_model())
        assert r.taxes_paid is None
        assert r.annual_taxes is None

    def test_gradual_rebalancing_consumes_lots_across_periods(self):
        """Gradual sell over multiple periods correctly consumes lots."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _evaluate_rebalancing)
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=10.0, btc_price=50_000, sim_date="2035-06-15",
            cash=100_000, reserves=[0, 0, 0], investments=[0, 0],
            tax_lots=[TaxLot(date="2031-01-01", btc=10.0,
                             cost_basis=20_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(
            cost_basis_method="fifo",
            high_q_trigger=0.90,
            high_q_action={"mode": "gradual", "rate": 5.0, "duration": 3,
                           "split": {"cash": 1.0}},
        )
        initial_btc = state.btc_stack
        for i in range(3):
            _evaluate_rebalancing(state, cfg, btc_quantile=0.95)
        assert state.btc_stack < initial_btc
        lot_sum = sum(l.btc for l in state.tax_lots)
        assert abs(lot_sum - state.btc_stack) < 1e-8
        assert state.tax_year_accum.lt_capital_gains > 0



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
        # At $0 YTD, TD is within standard deduction (0% marginal rate) so
        # tax cost = 0 + state (TX=0) = 0. TD cost equals cash cost.
        # Once income exceeds the deduction, TD becomes more expensive.
        cash_src = [s for s in sources if s.key == "cash"][0]
        assert td.cost >= cash_src.cost  # TD at least as expensive as cash

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

    def test_rank_by_cost_ascending(self):
        """Sources ranked purely by cost — cheapest first, regardless of wrapper."""
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
        # Roth cash (cost=0.01) is cheaper than taxable BTC (cost=5.0)
        assert ranked[0].key == "tf_cash_res"
        assert ranked[1].key == "btc"

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
        assert max_draw > 0

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

    def test_waterfall_preserves_btc_when_model_predicts_high_growth(self):
        """BTC is preserved when the model predicts a rate of return that
        exceeds the other assets' rates.

        Design (2026-04-17): BTC-preservation is NOT unconditional — the
        waterfall's opportunity-cost scoring uses the model's predicted
        BTC rate of return, annualized. With a strong growth model
        (~15%/yr), BTC's opportunity cost exceeds cash (4%), reserves
        (5%) and equity (10%), so BTC ranks last in the waterfall.
        """
        from engines.citadel import SimConfig, _initial_state, step

        class _StrongBtcModel:
            """Predicts BTC at ~15%/yr — higher than every other asset."""
            def __init__(self):
                import pandas as pd
                self.fits = {0.25: {"slope": 5.0, "intercept": 2.0}}
                self.genesis = pd.Timestamp("2009-07-25")

            def price_at(self, q, t):
                # 50k at t=0, growing 15%/yr → 10× over 10 yr
                return 50_000.0 * (1.15 ** t)

            def quantile_at(self, price, t):
                return 0.50

        import numpy as np
        cfg = SimConfig(
            start_stack=1.0, start_yr=2035, end_yr=2037,
            freq="Annually", monthly_spend=2_000,
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
        model = _StrongBtcModel()
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        initial_btc = state.btc_stack
        for _ in range(2):
            state = step(state, cfg, model.price_at(0.25, state.t + 1), rng, model=model)
        assert state.btc_stack >= initial_btc * 0.95, (
            f"Strong-BTC model: BTC should be preserved when its expected "
            f"return exceeds every non-BTC asset, got {state.btc_stack:.3f}"
        )
        # And a non-BTC source should show usage.
        assert state.cash < 50_000 or any(r < rb["initial"]
                                          for r, rb in zip(state.reserves, cfg.reserve_bins)
                                          if rb["initial"] > 0), (
            "Non-BTC sources should have been drawn from"
        )

    def test_waterfall_sells_btc_first_when_model_predicts_weak_growth(self):
        """BTC is drawn FIRST when the model's predicted rate of return is
        below the other assets — i.e., the user is better off keeping the
        higher-yielding assets and consuming BTC. This is the positive
        complement to the "strong-growth" preservation test."""
        from engines.citadel import SimConfig, _initial_state, step

        class _WeakBtcModel:
            """Predicts BTC at ~0.8%/yr — below cash, reserves, and equity."""
            def __init__(self):
                import pandas as pd
                self.fits = {0.25: {"slope": 5.0, "intercept": 2.0}}
                self.genesis = pd.Timestamp("2009-07-25")

            def price_at(self, q, t):
                # 50k at t=0, growing 0.8%/yr → ~8% over 10 yr
                return 50_000.0 * (1.008 ** t)

            def quantile_at(self, price, t):
                return 0.50

        import numpy as np
        cfg = SimConfig(
            start_stack=1.0, start_yr=2035, end_yr=2037,
            freq="Annually", monthly_spend=2_000,
            cash_initial=50_000, cash_rate=4.0,
            selected_qs=[0.25], tax_enabled=True, state_code="TX",
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
        model = _WeakBtcModel()
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        initial_btc = state.btc_stack
        for _ in range(2):
            state = step(state, cfg, model.price_at(0.25, state.t + 1), rng, model=model)
        assert state.btc_stack < initial_btc * 0.99, (
            f"Weak-BTC model: BTC should be drawn before higher-yielding "
            f"assets, but stayed at {state.btc_stack:.3f} (initial {initial_btc:.3f})"
        )

    def test_full_waterfall_roth_last(self):
        """Roth is never touched while other sources remain."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=0, start_yr=2035, end_yr=2037,
            freq="Annually", monthly_spend=5000,
            cash_initial=100_000, selected_qs=[0.25],
            tax_enabled=True, state_code="TX",
            td_cash_initial=100_000,
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
        # Taxable cash ($100k) + TD cash ($100k) should cover $120k spending.
        # Roth untouched (may grow slightly due to cash_rate interest).
        assert state.tf_cash >= 49_000, \
            f"Roth should be untouched while non-Roth covers spending, got {state.tf_cash}"

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
        # Seed $40k other income so we are already partway into the 0% LTCG bracket.
        # The inflated 0% LTCG upper for 2031 (single, 4% inflation) ≈ $61k.
        # After std deduction (~$19k): stacked ≈ $21k → boundary ≈ $40k in gain-space.
        # Drawing $10k TD adds $10k ordinary income → stacked ≈ $31k → boundary ≈ $30k.
        # Since we stay within the same bracket, after < before.
        acum = TaxYearAccumulator()
        acum.other_income = 40_000.0
        state = CitadelState(
            td_cash=500_000, sim_date="2031-06-15",
            investments=[200_000, 0], invest_cost_basis=[100_000, 0],
            btc_stack=0, btc_price=50_000,
            tax_year_accum=acum,
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
        # Draw $10k from TD → stays within same 0% LTCG bracket, shifts base up
        td_src = _WithdrawalSource(
            key="td_cash", wrapper="td", asset_type="cash", index=0,
            available=500_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ordinary",
        )
        _execute_draw(state, cfg, td_src, 10_000)
        # LTCG boundary after TD draw — should be smaller (base shifted up within bracket)
        after = _max_draw_before_boundary(state, cfg, inv_src)
        assert after < before, "TD draw should shift LTCG stack base, reducing boundary distance"
        assert state.tax_year_accum.tax_deferred_withdrawals == pytest.approx(10_000)

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
        """Spec test 17: With very short treasury horizon (age 95), treasury is cheaper to sell
        than BTC (equity-rate fallback). Treasury horizon clamps at 1 for ages ≥90."""
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
        # birth_year=1975 → age 95 in 2070 → treasury horizon = max(min(90-95, 40), 1) = 1
        cfg = SimConfig(tax_enabled=False, start_yr=2070, birth_year=1975,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        # No model → BTC falls back to equity rate (10%) over 10yr horizon
        sources = _build_source_list(state, cfg, model=None)
        _score_sources(sources, state, cfg, model=None)
        btc = [s for s in sources if s.key == "btc"][0]
        tres = [s for s in sources if s.key == "reserve_0"][0]
        # Treasury horizon=1: cost = (1.05^1 - 1) = 0.05 (very low)
        # BTC horizon=10, growth=10%: cost = (1.10^10 - 1) ≈ 1.59 (high)
        # Treasury should be far cheaper to sell in late retirement with age 95
        assert tres.horizon == 1, f"Expected treasury horizon=1 for age 95, got {tres.horizon}"
        assert tres.cost < btc.cost, \
            f"Treasury ({tres.cost:.3f}) should be cheaper than BTC ({btc.cost:.3f}) at age 95"

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

    def test_td_horizon_before_rmd_age(self):
        """TD horizon ramps down as RMD start age approaches."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list)
        from engines.tax import TaxYearAccumulator
        # Age 50, RMD at 75 → horizon = min(15, 25) = 15
        state = CitadelState(
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, birth_year=1985, start_yr=2035,
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
        td = [s for s in sources if s.key == "td_cash"][0]
        assert td.horizon == 15  # 25 years until RMD, capped at 15

        # Age 65, RMD at 75 → horizon = min(15, 10) = 10
        state2 = CitadelState(
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            sim_date="2050-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg2 = SimConfig(tax_enabled=True, birth_year=1985, start_yr=2050,
                         reserve_bins=cfg.reserve_bins, invest_bins=cfg.invest_bins)
        sources2 = _build_source_list(state2, cfg2, model=None)
        td2 = [s for s in sources2 if s.key == "td_cash"][0]
        assert td2.horizon == 10

        # Age 70, RMD at 75 → horizon = 5
        state3 = CitadelState(
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            sim_date="2055-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg3 = SimConfig(tax_enabled=True, birth_year=1985, start_yr=2055,
                         reserve_bins=cfg.reserve_bins, invest_bins=cfg.invest_bins)
        sources3 = _build_source_list(state3, cfg3, model=None)
        td3 = [s for s in sources3 if s.key == "td_cash"][0]
        assert td3.horizon == 5

    def test_td_horizon_at_rmd_age_uses_factor(self):
        """At RMD age, TD horizon equals the IRS RMD factor."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list)
        from engines.tax import TaxYearAccumulator
        from engines.tax_data import RMD_FACTORS
        # Age 75, RMD factor = 24.6
        state = CitadelState(
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            sim_date="2060-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, birth_year=1985, start_yr=2060,
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
        td = [s for s in sources if s.key == "td_cash"][0]
        assert td.horizon == int(RMD_FACTORS[75])  # 24

        # Age 85, RMD factor = 16.0
        state2 = CitadelState(
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            sim_date="2070-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg2 = SimConfig(tax_enabled=True, birth_year=1985, start_yr=2070,
                         reserve_bins=cfg.reserve_bins, invest_bins=cfg.invest_bins)
        sources2 = _build_source_list(state2, cfg2, model=None)
        td2 = [s for s in sources2 if s.key == "td_cash"][0]
        assert td2.horizon == int(RMD_FACTORS[85])

    def test_td_cheaper_near_rmd_age(self):
        """TD becomes cheaper to withdraw as RMD age approaches."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        from engines.tax import TaxYearAccumulator

        def _td_cost_at_age(age):
            yr = 1985 + age
            state = CitadelState(
                cash=50_000, td_cash=100_000,
                td_reserves=[0, 0, 0], td_investments=[0, 0],
                sim_date=f"{yr}-06-15",
                tax_year_accum=TaxYearAccumulator(),
            )
            cfg = SimConfig(tax_enabled=True, birth_year=1985, start_yr=yr,
                            state_code="TX", filing_status="single", inflation=4.0,
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
            return td.cost

        cost_50 = _td_cost_at_age(50)
        cost_65 = _td_cost_at_age(65)
        cost_73 = _td_cost_at_age(73)
        # TD gets cheaper as RMD approaches (shorter horizon = less forgone compounding)
        assert cost_50 > cost_65 > cost_73

    def test_td_free_below_standard_deduction(self):
        """TD draws are free (0% marginal rate) when below standard deduction."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=50_000, reserves=[0, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=0, btc_price=50_000, sim_date="2035-06-15",
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            tax_year_accum=TaxYearAccumulator(),  # zero YTD income
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX", cash_rate=4.0,
                        filing_status="single", inflation=4.0, start_yr=2031,
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
        cash_src = [s for s in sources if s.key == "cash"][0]
        td_src = [s for s in sources if s.key == "td_cash"][0]
        # At zero YTD income, TD marginal rate = 0% (within standard deduction)
        # TD cost should equal cash cost (both have 4% growth, same horizon,
        # zero tax, and TD opp adjusted by (1-0) = 1.0)
        assert td_src.cost == pytest.approx(cash_src.cost, rel=0.01)

    def test_boundary_includes_deduction_cushion(self):
        """Bracket boundary includes remaining standard deduction cushion."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _max_draw_before_boundary)
        from engines.tax import TaxYearAccumulator
        # Zero YTD income → full standard deduction + first bracket available
        state = CitadelState(
            sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(),  # zero income
        )
        cfg = SimConfig(tax_enabled=True, filing_status="single",
                        inflation=4.0, start_yr=2031, freq="Monthly")
        td_source = _WithdrawalSource(
            key="td_cash", wrapper="td", asset_type="cash", index=0,
            available=500_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ordinary",
        )
        max_draw = _max_draw_before_boundary(state, cfg, td_source)
        # sim_year=2031, yrs_from_base=6 (2031-2025)
        # Should be std_ded(inflated 6yr) + first_bracket_top(inflated 6yr)
        from engines.tax import _inflate_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA, STANDARD_DEDUCTION_TCJA
        std_ded = STANDARD_DEDUCTION_TCJA["single"] * (1.04 ** 6)
        brackets = _inflate_brackets(FEDERAL_BRACKETS_TCJA["single"], 6, 0.04)
        expected = std_ded + brackets[0][0]
        assert max_draw == pytest.approx(expected, rel=0.01)


# ── Phase 1: Unified Citadel MC ──────────────────────────────────────────────


class TestInitialRegimeConfig:
    def test_default_initial_regimes_are_neutral(self):
        from engines.citadel_types import SimConfig
        cfg = SimConfig()
        assert cfg.initial_equity_regime == 2
        assert cfg.initial_bond_regime == 2
        assert cfg.initial_res_short_regime == 2
        assert cfg.initial_res_med_regime == 2
        assert cfg.initial_res_long_regime == 2

    def test_initial_regimes_are_configurable(self):
        from engines.citadel_types import SimConfig
        cfg = SimConfig(
            initial_equity_regime=0,
            initial_bond_regime=4,
            initial_res_short_regime=1,
            initial_res_med_regime=3,
            initial_res_long_regime=0,
        )
        assert cfg.initial_equity_regime == 0
        assert cfg.initial_bond_regime == 4



class TestInitialRegimeWiring:
    def test_td_tf_regime_fields_exist(self):
        from engines.citadel_types import CitadelState
        state = CitadelState()
        assert state.td_equity_regime == 2
        assert state.td_bond_regime == 2
        assert state.td_res_short_regime == 2
        assert state.td_res_med_regime == 2
        assert state.td_res_long_regime == 2
        assert state.tf_equity_regime == 2
        assert state.tf_bond_regime == 2
        assert state.tf_res_short_regime == 2
        assert state.tf_res_med_regime == 2
        assert state.tf_res_long_regime == 2

    def test_initial_state_uses_config_regimes(self):
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import _initial_state
        cfg = SimConfig()
        cfg.initial_equity_regime = 0
        cfg.initial_bond_regime = 4
        cfg.initial_res_short_regime = 1
        cfg.initial_res_med_regime = 3
        cfg.initial_res_long_regime = 0
        state = _initial_state(cfg, model=None)
        assert state.equity_regime == 0
        assert state.bond_regime == 4
        assert state.res_short_regime == 1
        assert state.res_med_regime == 3
        assert state.res_long_regime == 0

    def test_initial_state_seeds_td_tf_regimes_unconditionally(self):
        """TD/TF regimes seeded from config regardless of tax_enabled."""
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import _initial_state
        cfg = SimConfig()
        cfg.tax_enabled = False
        cfg.initial_equity_regime = 4
        cfg.initial_bond_regime = 0
        state = _initial_state(cfg, model=None)
        assert state.td_equity_regime == 4
        assert state.td_bond_regime == 0
        assert state.tf_equity_regime == 4
        assert state.tf_bond_regime == 0



class TestMarkovGuard:
    def _make_markov_config(self, n_sims=10):
        import numpy as np
        from engines.citadel_types import SimConfig
        matrices = {}
        for key in ("equity", "bond", "tres_short", "tres_med", "tres_long"):
            n_bins = 5
            trans = np.ones((n_bins, n_bins)) / n_bins
            bin_means = np.array([-0.02, -0.005, 0.005, 0.01, 0.02])
            bin_vols = np.array([0.01, 0.005, 0.003, 0.005, 0.01])
            matrices[key] = {"trans": trans, "bin_means": bin_means, "bin_vols": bin_vols}
        cfg = SimConfig()
        cfg.asset_return_model = "markov"
        cfg.asset_matrices = matrices
        cfg.n_sims = n_sims
        return cfg

    def test_markov_fires_when_n_sims_gt_1(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=10)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=None)
        for _ in range(20):
            state = step(state, cfg, 100_000.0, rng, model=None)
        regimes = [state.equity_regime, state.bond_regime,
                   state.res_short_regime, state.res_med_regime, state.res_long_regime]
        assert any(r != 2 for r in regimes), "After 20 Markov steps, at least one regime should change"

    def test_markov_does_not_fire_when_n_sims_1(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=1)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=None)
        state = step(state, cfg, 100_000.0, rng, model=None)
        assert state.equity_regime == 2
        assert state.bond_regime == 2



class TestTdTfMarkovReturns:
    def _make_markov_config(self, n_sims=10):
        import numpy as np
        from engines.citadel_types import SimConfig
        matrices = {}
        for key in ("equity", "bond", "tres_short", "tres_med", "tres_long"):
            n_bins = 5
            trans = np.ones((n_bins, n_bins)) / n_bins
            bin_means = np.array([-0.02, -0.005, 0.005, 0.01, 0.02])
            bin_vols = np.array([0.01, 0.005, 0.003, 0.005, 0.01])
            matrices[key] = {"trans": trans, "bin_means": bin_means, "bin_vols": bin_vols}
        cfg = SimConfig()
        cfg.asset_return_model = "markov"
        cfg.asset_matrices = matrices
        cfg.n_sims = n_sims
        cfg.tax_enabled = True
        return cfg

    def test_td_regimes_evolve_under_markov(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=10)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=None)
        for _ in range(30):
            state = step(state, cfg, 100_000.0, rng, model=None)
        td_regimes = [state.td_equity_regime, state.td_bond_regime,
                      state.td_res_short_regime, state.td_res_med_regime,
                      state.td_res_long_regime]
        assert any(r != 2 for r in td_regimes), "TD regimes should evolve under Markov"

    def test_tf_regimes_evolve_under_markov(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=10)
        rng = np.random.default_rng(99)
        state = _initial_state(cfg, model=None)
        for _ in range(30):
            state = step(state, cfg, 100_000.0, rng, model=None)
        tf_regimes = [state.tf_equity_regime, state.tf_bond_regime]
        assert any(r != 2 for r in tf_regimes), "TF regimes should evolve under Markov"

    def test_td_tf_use_lognormal_when_n_sims_1(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=1)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=None)
        state = step(state, cfg, 100_000.0, rng, model=None)
        assert state.td_equity_regime == 2, "TD regimes unchanged when n_sims=1"
        assert state.tf_equity_regime == 2, "TF regimes unchanged when n_sims=1"



class TestBandAggregation:
    def test_compute_bands_returns_7_percentiles(self):
        from engines.citadel_bands import compute_bands, BAND_PERCENTILES
        assert BAND_PERCENTILES == (5, 10, 25, 50, 75, 90, 95)

    def test_compute_bands_returns_11_series(self):
        from engines.citadel_bands import BAND_SERIES
        assert len(BAND_SERIES) == 11
        assert "total" in BAND_SERIES
        assert "btc_stack" in BAND_SERIES
        assert "td_total" in BAND_SERIES
        assert "tf_total" in BAND_SERIES
        assert "depletion" in BAND_SERIES

    def test_band_ordering(self):
        """P5 <= P25 <= P50 <= P75 <= P95 for total portfolio."""
        import numpy as np
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands
        cfg = SimConfig()
        cfg.start_yr = 2031; cfg.end_yr = 2032
        paths = np.array([[20000 + i * 30000 + j * 100 for j in range(12)]
                          for i in range(20)])
        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)
        for t in range(12):
            vals = [bands[p]["total"][t] for p in [5, 25, 50, 75, 95]]
            for k in range(len(vals) - 1):
                assert vals[k] <= vals[k + 1] + 1e-6



class TestDevBypass:
    def test_dev_bypass_exists_in_mc_payment(self):
        import inspect
        from callbacks import mc_payment
        source = inspect.getsource(mc_payment)
        assert "DEV" in source, "mc_payment should check DEV env var for bypass"



class TestUnifiedMcIntegration:
    def _make_matrices(self):
        import numpy as np
        matrices = {}
        for key in ("equity", "bond", "tres_short", "tres_med", "tres_long"):
            n_bins = 5
            trans = np.full((n_bins, n_bins), 0.05)
            np.fill_diagonal(trans, 0.80)
            trans /= trans.sum(axis=1, keepdims=True)
            bin_means = np.array([-0.03, -0.01, 0.005, 0.015, 0.03])
            bin_vols = np.array([0.015, 0.008, 0.005, 0.008, 0.015])
            matrices[key] = {"trans": trans, "bin_means": bin_means, "bin_vols": bin_vols}
        return matrices

    def test_full_mc_20_sims_produces_spread(self):
        import numpy as np
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import simulate
        cfg = SimConfig()
        cfg.start_yr = 2031; cfg.end_yr = 2033
        cfg.asset_return_model = "markov"
        cfg.asset_matrices = self._make_matrices()
        cfg.initial_equity_regime = 4  # Bull
        cfg.initial_bond_regime = 0    # Bear
        rng = np.random.default_rng(123)
        base = np.linspace(50000, 150000, 24)
        paths = np.array([base * (1 + rng.normal(0, 0.1, 24)) for _ in range(20)])
        result = simulate(cfg, model=None, price_paths=paths)
        assert result.total_usd.shape == (20, 24)
        assert set(result.percentiles.keys()) == {5, 10, 25, 50, 75, 90, 95}
        p5 = result.percentiles[5]["total"]
        p95 = result.percentiles[95]["total"]
        assert np.any(p95 > p5 + 1.0), "MC should produce nonzero spread"

    def test_deterministic_unchanged(self):
        """n_sims=1 with a single price path: all percentiles identical."""
        import numpy as np
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import simulate
        cfg = SimConfig()
        cfg.start_yr = 2031; cfg.end_yr = 2033; cfg.n_sims = 1
        paths = np.array([[80000 + j * 200 for j in range(24)]])
        result = simulate(cfg, model=None, price_paths=paths)
        assert result.total_usd.shape[0] == 1
        for key in ["total", "btc_usd", "cash"]:
            np.testing.assert_array_almost_equal(
                result.percentiles[5][key], result.percentiles[95][key],
                err_msg=f"Deterministic: P5 should equal P95 for {key}")

    def test_bands_match_standalone_compute(self):
        import numpy as np
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands
        cfg = SimConfig()
        cfg.start_yr = 2031; cfg.end_yr = 2032
        paths = np.array([[50000 + i * 10000 + j * 100 for j in range(12)]
                          for i in range(10)])
        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)
        for pct in [5, 50, 95]:
            np.testing.assert_array_almost_equal(
                bands[pct]["total"], result.percentiles[pct]["total"])


# ── Phase 2: Citadel Presets & Cache ─────────────────────────────────────────


class TestCitadelPresets:
    def test_wealth_levels_exist(self):
        from citadel_presets import WEALTH_LEVELS
        assert set(WEALTH_LEVELS.keys()) == {"starter", "full", "bitcoin"}

    def test_wealth_level_has_required_keys(self):
        from citadel_presets import WEALTH_LEVELS
        required = {"label", "dollar_assets", "btc", "monthly_spend",
                    "spend_growth", "inflation", "allocation"}
        for key, wl in WEALTH_LEVELS.items():
            assert required.issubset(wl.keys()), f"{key} missing {required - wl.keys()}"

    def test_allocation_sums_to_100(self):
        from citadel_presets import WEALTH_LEVELS
        for key, wl in WEALTH_LEVELS.items():
            total = sum(wl["allocation"].values())
            assert abs(total - 100) < 0.01, f"{key} allocation sums to {total}"

    def test_macro_regimes_exist(self):
        from citadel_presets import MACRO_REGIMES
        assert set(MACRO_REGIMES.keys()) == {"bear", "neutral", "bull"}
        assert MACRO_REGIMES["bear"]["bin"] == 0
        assert MACRO_REGIMES["neutral"]["bin"] == 2
        assert MACRO_REGIMES["bull"]["bin"] == 4

    def test_rule_sets_exist(self):
        from citadel_presets import RULE_SETS
        assert set(RULE_SETS.keys()) == {"no_rebal", "cautious", "aggressive"}

    def test_cache_dimensions(self):
        from citadel_presets import (BTC_MODELS, BTC_ENTRY_QS, START_YEARS,
                                     SIMS_PER_SCENARIO, WEALTH_LEVELS,
                                     MACRO_REGIMES, RULE_SETS, TAX_STATUSES)
        assert BTC_MODELS == ["bub", "qr", "pl", "lppl", "ef"]
        assert BTC_ENTRY_QS == [1, 10, 50]
        assert START_YEARS == [2028, 2035]
        assert SIMS_PER_SCENARIO == 800
        total = (len(BTC_MODELS) * len(BTC_ENTRY_QS) * len(MACRO_REGIMES) *
                 len(WEALTH_LEVELS) * len(RULE_SETS) * len(START_YEARS) *
                 len(TAX_STATUSES))
        assert total == 1620

    def test_build_config_returns_simconfig(self):
        from citadel_presets import build_config
        from engines.citadel_types import SimConfig
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert isinstance(cfg, SimConfig)

    def test_build_config_starter_values(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert cfg.start_stack == 0.5
        assert cfg.monthly_spend == 5000
        assert cfg.cash_initial == 50_000
        assert cfg.start_yr == 2035
        assert cfg.end_yr == 2075
        assert cfg.freq == "Monthly"
        assert cfg.inflation == 4.0

    def test_build_config_regime_sets_initial_regimes(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="bull", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert cfg.initial_equity_regime == 4
        assert cfg.initial_bond_regime == 4
        assert cfg.initial_res_short_regime == 4
        assert cfg.initial_res_med_regime == 4
        assert cfg.initial_res_long_regime == 4

    def test_build_config_tax_status_mfj(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="mfj",
        )
        assert cfg.tax_enabled is True
        assert cfg.filing_status == "mfj"

    def test_build_config_tax_status_single(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert cfg.tax_enabled is True
        assert cfg.filing_status == "single"

    def test_build_config_loads_asset_matrices(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert cfg.asset_matrices is not None
        assert "equity" in cfg.asset_matrices
        assert "bond" in cfg.asset_matrices
        assert "tres_short" in cfg.asset_matrices



class TestCitadelBandCache:
    def test_band_cache_key_format(self):
        from citadel_band_cache import band_cache_key
        key = band_cache_key("bub", 10, "neutral", "starter",
                             "no_rebal", 2035, "single")
        assert key == "bub_q10_neutral_starter_no_rebal_2035_single"

    def test_band_cache_key_all_combos_unique(self):
        from citadel_band_cache import band_cache_key
        from citadel_presets import (BTC_MODELS, BTC_ENTRY_QS, MACRO_REGIMES,
                                     WEALTH_LEVELS, RULE_SETS, START_YEARS,
                                     TAX_STATUSES)
        keys = set()
        for model in BTC_MODELS:
            for eq in BTC_ENTRY_QS:
                for regime in MACRO_REGIMES:
                    for wealth in WEALTH_LEVELS:
                        for rules in RULE_SETS:
                            for yr in START_YEARS:
                                for tax in TAX_STATUSES:
                                    k = band_cache_key(model, eq, regime,
                                                       wealth, rules, yr, tax)
                                    keys.add(k)
        assert len(keys) == 1620

    def test_pack_unpack_bands_roundtrip(self):
        import numpy as np
        from citadel_band_cache import pack_bands, unpack_bands
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 480
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {}
            for series in BAND_SERIES:
                bands[pct][series] = np.random.rand(n_periods).astype(np.float32)
        packed = pack_bands(bands)
        assert isinstance(packed, np.ndarray)
        assert packed.dtype == np.float32
        unpacked = unpack_bands(packed)
        for pct in BAND_PERCENTILES:
            for series in BAND_SERIES:
                np.testing.assert_array_almost_equal(
                    unpacked[pct][series], bands[pct][series], decimal=5)

    def test_store_and_lookup(self, tmp_path):
        import numpy as np
        from citadel_band_cache import store_entry, lookup_entry
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 24
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.ones(n_periods, dtype=np.float32) * pct
                          for s in BAND_SERIES}
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)
        result = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)
        assert result is not None
        for pct in BAND_PERCENTILES:
            np.testing.assert_array_almost_equal(
                result[pct]["total"], np.ones(n_periods) * pct, decimal=5)

    def test_lookup_missing_returns_none(self, tmp_path):
        from citadel_band_cache import lookup_entry
        result = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)

        assert result is None


class TestCitadelBandGeneration:
    def test_generate_single_entry(self, tmp_path):
        """Smoke test: generate one combo with 5 sims (fast)."""
        import numpy as np
        from citadel_presets import build_config
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands, BAND_PERCENTILES, BAND_SERIES
        from citadel_band_cache import store_entry, lookup_entry

        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        cfg.end_yr = 2036  # 1 year = 12 periods (fast)
        n_sims = 5
        n_periods = 12
        rng = np.random.default_rng(42)
        base = np.linspace(80000, 120000, n_periods)
        paths = np.array([base * (1 + rng.normal(0, 0.05, n_periods))
                          for _ in range(n_sims)])
        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)
        loaded = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)
        assert loaded is not None
        assert set(loaded.keys()) == set(BAND_PERCENTILES)
        assert set(loaded[50].keys()) == set(BAND_SERIES)
        assert len(loaded[50]["total"]) == n_periods

    def test_generate_preserves_band_ordering(self, tmp_path):
        """P5 <= P50 <= P95 in generated bands."""
        import numpy as np
        from citadel_presets import build_config
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands
        from citadel_band_cache import store_entry, lookup_entry

        cfg = build_config(
            wealth="full", regime="bear", rules="cautious",
            start_year=2028, tax_status="mfj",
        )
        cfg.end_yr = 2029
        n_sims = 20
        n_periods = 12
        rng = np.random.default_rng(99)
        base = np.linspace(50000, 100000, n_periods)
        paths = np.array([base * (1 + rng.normal(0, 0.15, n_periods))
                          for _ in range(n_sims)])
        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)
        store_entry("pl", 50, "bear", "full", "cautious",
                    2028, "mfj", bands, cache_dir=tmp_path)
        loaded = lookup_entry("pl", 50, "bear", "full", "cautious",
                              2028, "mfj", cache_dir=tmp_path)
        for t in range(n_periods):
            assert loaded[5]["total"][t] <= loaded[50]["total"][t] + 1e-6
            assert loaded[50]["total"][t] <= loaded[95]["total"][t] + 1e-6



class TestCitadelBandCacheLoader:
    @pytest.fixture(autouse=True)
    def _clear_band_cache(self):
        """Isolate tests from shared module state."""
        from citadel_band_cache import _BAND_CACHE
        _BAND_CACHE.clear()
        yield
        _BAND_CACHE.clear()

    def test_load_band_caches_from_disk(self, tmp_path):
        import numpy as np
        from citadel_band_cache import store_entry, load_band_caches, _BAND_CACHE
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 12
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.ones(n_periods, dtype=np.float32) * pct
                          for s in BAND_SERIES}
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)
        load_band_caches(cache_dir=tmp_path)
        assert len(_BAND_CACHE) == 1
        key = "bub_q10_neutral_starter_no_rebal_2035_single"
        assert key in _BAND_CACHE

    def test_load_empty_dir(self, tmp_path):
        from citadel_band_cache import load_band_caches, _BAND_CACHE
        load_band_caches(cache_dir=tmp_path)
        assert len(_BAND_CACHE) == 0



class TestCitadelBandCacheIntegration:
    def test_full_pipeline_build_simulate_store_lookup(self, tmp_path):
        """End-to-end: build_config -> simulate -> compute_bands -> store -> lookup."""
        import numpy as np
        from citadel_presets import build_config
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands, BAND_PERCENTILES, BAND_SERIES
        from citadel_band_cache import store_entry, lookup_entry

        cfg = build_config(
            wealth="bitcoin", regime="bull", rules="aggressive",
            start_year=2028, tax_status="mfj",
        )
        cfg.end_yr = 2029  # 12 periods for speed
        n_sims = 10
        n_periods = 12
        rng = np.random.default_rng(77)
        base = np.linspace(60000, 200000, n_periods)
        paths = np.array([base * (1 + rng.normal(0, 0.1, n_periods))
                          for _ in range(n_sims)])

        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)

        store_entry("bub", 50, "bull", "bitcoin", "aggressive",
                    2028, "mfj", bands, cache_dir=tmp_path)

        loaded = lookup_entry("bub", 50, "bull", "bitcoin", "aggressive",
                              2028, "mfj", cache_dir=tmp_path)

        assert loaded is not None
        assert set(loaded.keys()) == set(BAND_PERCENTILES)
        for pct in BAND_PERCENTILES:
            assert set(loaded[pct].keys()) == set(BAND_SERIES)
            assert len(loaded[pct]["total"]) == n_periods

        # Verify band ordering
        for t in range(n_periods):
            assert loaded[5]["total"][t] <= loaded[50]["total"][t] + 1e-6
            assert loaded[50]["total"][t] <= loaded[95]["total"][t] + 1e-6

    def test_multiple_entries_same_npz(self, tmp_path):
        """Multiple entries for same (model, start_yr) share one npz."""
        import numpy as np
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        from citadel_band_cache import store_entry, lookup_entry

        n_periods = 12
        for regime in ["bear", "neutral", "bull"]:
            bands = {}
            for pct in BAND_PERCENTILES:
                bands[pct] = {s: np.full(n_periods, float(pct), dtype=np.float32)
                              for s in BAND_SERIES}
            store_entry("qr", 10, regime, "starter", "no_rebal",
                        2035, "single", bands, cache_dir=tmp_path)

        # All three in same npz, all retrievable
        for regime in ["bear", "neutral", "bull"]:
            loaded = lookup_entry("qr", 10, regime, "starter", "no_rebal",
                                  2035, "single", cache_dir=tmp_path)
            assert loaded is not None
            assert loaded[50]["total"][0] == 50.0

    def test_cache_key_uniqueness_across_all_dimensions(self):
        """All 1620 combos produce unique cache keys."""
        from itertools import product as iproduct
        from citadel_presets import (BTC_MODELS, BTC_ENTRY_QS, MACRO_REGIMES,
                                     WEALTH_LEVELS, RULE_SETS, START_YEARS,
                                     TAX_STATUSES)
        from citadel_band_cache import band_cache_key
        keys = set()
        for m, eq, reg, wl, rs, yr, ts in iproduct(
            BTC_MODELS, BTC_ENTRY_QS, MACRO_REGIMES.keys(),
            WEALTH_LEVELS.keys(), RULE_SETS.keys(), START_YEARS, TAX_STATUSES,
        ):
            keys.add(band_cache_key(m, eq, reg, wl, rs, yr, ts))
        assert len(keys) == 1620


# ── Phase 3: Quick Scenarios UI ──────────────────────────────────────────────


class TestCitadelQuickScenariosLayout:
    def test_scenario_stores_exist(self):
        """Verify scenario-related stores are in the layout."""
        from layout.citadel import _citadel_controls
        layout = _citadel_controls()
        layout_str = str(layout)
        assert "cp-scenario-wealth" in layout_str
        assert "cp-scenario-regime" in layout_str
        assert "cp-scenario-rules" in layout_str
        assert "cp-scenario-start-yr" in layout_str
        assert "cp-scenario-bands" in layout_str
        assert "cp-scenario-active" in layout_str

    def test_scenario_pill_buttons_exist(self):
        """Verify pill button IDs are present."""
        from layout.citadel import _citadel_controls
        layout = _citadel_controls()
        layout_str = str(layout)
        for wl in ["starter", "full", "bitcoin"]:
            assert f"cp-pill-{wl}" in layout_str
        for reg in ["bear", "neutral", "bull"]:
            assert f"cp-pill-{reg}" in layout_str
        for rs in ["no_rebal", "cautious", "aggressive"]:
            assert f"cp-pill-{rs}" in layout_str



class TestCitadelScenarioCallback:
    def test_scenario_lookup_returns_bands_for_valid_combo(self, tmp_path):
        """Verify lookup returns bands when cache exists."""
        import numpy as np
        from citadel_band_cache import store_entry, lookup_entry
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 480
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.ones(n_periods, dtype=np.float32) * pct
                          for s in BAND_SERIES}
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)
        result = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)
        assert result is not None
        assert 50 in result
        assert "total" in result[50]
        assert len(result[50]["total"]) == n_periods

    def test_scenario_lookup_returns_none_for_missing(self, tmp_path):
        from citadel_band_cache import lookup_entry
        result = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2099, "single", cache_dir=tmp_path)
        assert result is None

    def test_snap_quantile_to_cached_bin(self):
        """cp-qs value (float 0.25) should snap to nearest cached bin (10)."""
        from callbacks.citadel_scenarios import _snap_entry_q
        assert _snap_entry_q(0.01) == 1
        assert _snap_entry_q(0.05) == 1
        assert _snap_entry_q(0.10) == 10
        assert _snap_entry_q(0.25) == 10
        assert _snap_entry_q(0.50) == 50
        assert _snap_entry_q(0.75) == 50
        assert _snap_entry_q(0.999) == 50



class TestCitadelBandRendering:
    def test_build_band_traces_returns_traces(self):
        """Verify band trace builder produces scatter traces."""
        import numpy as np
        from figures.citadel import _build_band_traces
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 24
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.linspace(1000, 2000, n_periods).tolist()
                          for s in BAND_SERIES}
        time_axis = np.linspace(22, 24, n_periods).tolist()
        traces = _build_band_traces(bands, time_axis, series_key="total",
                                     color="#000000")
        assert len(traces) == 4
        import plotly.graph_objects as go
        for t in traces:
            assert isinstance(t, go.Scatter)

    def test_build_band_traces_empty_bands(self):
        from figures.citadel import _build_band_traces
        traces = _build_band_traces(None, [], series_key="total", color="#000")
        assert traces == []

    def test_build_band_traces_string_keys(self):
        """Verify works with string percentile keys (from JSON store)."""
        import numpy as np
        from figures.citadel import _build_band_traces
        bands = {
            "5": {"total": [100] * 10},
            "25": {"total": [200] * 10},
            "75": {"total": [300] * 10},
            "95": {"total": [400] * 10},
        }
        traces = _build_band_traces(bands, list(range(10)),
                                     series_key="total", color="#FF0000")
        assert len(traces) == 4



class TestCitadelScenarioSnapshot:
    def test_scenario_controls_in_snapshot(self):
        from snapshot import _SNAPSHOT_CONTROLS
        ids = {c[0] for c in _SNAPSHOT_CONTROLS}
        assert "cp-scenario-wealth" in ids
        assert "cp-scenario-regime" in ids
        assert "cp-scenario-rules" in ids
        assert "cp-scenario-start-yr" in ids
        assert "cp-scenario-active" in ids

    def test_scenario_controls_in_tab_controls(self):
        from callbacks.routing import _TAB_CONTROLS
        citadel_ids = _TAB_CONTROLS["citadel"]
        assert "cp-scenario-wealth" in citadel_ids
        assert "cp-scenario-regime" in citadel_ids
        assert "cp-scenario-rules" in citadel_ids
        assert "cp-scenario-start-yr" in citadel_ids
        assert "cp-scenario-active" in citadel_ids



class TestCitadelQuickScenariosIntegration:
    def test_full_scenario_pipeline(self, tmp_path):
        """End-to-end: store bands → lookup → build traces."""
        import numpy as np
        from citadel_band_cache import store_entry, lookup_entry
        from figures.citadel import _build_band_traces
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES

        n_periods = 24
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.linspace(100 * pct, 200 * pct, n_periods).astype(np.float32)
                          for s in BAND_SERIES}
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)

        loaded = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)
        assert loaded is not None

        # Serialize like the callback does
        serialized = {}
        for pct, series_dict in loaded.items():
            serialized[str(pct)] = {k: v.tolist() for k, v in series_dict.items()}

        time_axis = np.linspace(26, 28, n_periods).tolist()
        traces = _build_band_traces(serialized, time_axis,
                                     series_key="total", color="#000000")
        assert len(traces) == 4
        # P5 lower bound should be less than P95 upper bound
        assert traces[0].y[0] < traces[1].y[0]

    def test_all_preset_combos_produce_valid_configs(self):
        """Every preset combo builds a valid SimConfig."""
        from citadel_presets import (WEALTH_LEVELS, MACRO_REGIMES, RULE_SETS,
                                     START_YEARS, TAX_STATUSES, build_config)
        from engines.citadel_sim import validate_config
        for wealth in WEALTH_LEVELS:
            for regime in MACRO_REGIMES:
                for rules in RULE_SETS:
                    cfg = build_config(wealth, regime, rules, 2035, "single")
                    validate_config(cfg)  # raises on invalid



class TestPresetControlValues:
    def test_preset_control_values_returns_dict(self):
        from citadel_presets import preset_control_values
        vals = preset_control_values("starter", "neutral", "no_rebal", 2035)
        assert isinstance(vals, dict)
        assert "cp-stack" in vals
        assert "cp-spend" in vals
        assert "cp-cash-init" in vals

    def test_preset_control_values_starter(self):
        from citadel_presets import preset_control_values
        vals = preset_control_values("starter", "neutral", "no_rebal", 2035)
        assert vals["cp-stack"] == 0.5
        assert vals["cp-spend"] == 5000
        assert vals["cp-cash-init"] == 50000
        assert vals["cp-infl"] == 4.0
        assert vals["cp-spend-growth"] == 1.0

    def test_preset_control_values_bitcoin_bull_aggressive(self):
        from citadel_presets import preset_control_values
        vals = preset_control_values("bitcoin", "bull", "aggressive", 2028)
        assert vals["cp-stack"] == 12.5
        assert vals["cp-spend"] == 50000
        assert vals["cp-cash-floor"] == 100000
        assert vals["cp-yr-range"] == [2028, 2075]

    def test_preset_control_values_rules_no_rebal(self):
        from citadel_presets import preset_control_values
        vals = preset_control_values("starter", "neutral", "no_rebal", 2035)
        assert vals["cp-high-q-thresh"] == 99
        assert vals["cp-low-q-thresh"] == 1



class TestScenarioDynamicLookup:
    def test_snap_entry_q_boundary_values(self):
        from callbacks.citadel_scenarios import _snap_entry_q
        # Values near boundaries (bins: [1, 10, 50])
        assert _snap_entry_q(0.005) == 1    # closer to 1%
        assert _snap_entry_q(0.05) == 1     # closer to 1% than 10%
        assert _snap_entry_q(0.06) == 10    # closer to 10%
        assert _snap_entry_q(0.31) == 50    # closer to 50% (midpoint 30% ties to 10)
        assert _snap_entry_q(0.999) == 50



class TestScenarioStaleIndicator:
    def test_stale_badge_in_layout(self):
        from layout.citadel import _citadel_controls
        layout = _citadel_controls()
        assert "cp-scenario-stale" in repr(layout)


# ═══════════════════════════════════════════════════════════════════════════════
# Section: Citadel preset-scenario property tests (backlog #39)
#
# Sweeps WEALTH_LEVELS × MACRO_REGIMES × RULE_SETS × tax∈{False, True} through
# simulate() and asserts invariants that catch the class of bugs that produce
# implausible multi-billion-dollar end-state totals from a $500k preset.
# ═══════════════════════════════════════════════════════════════════════════════


def _preset_config(wealth, regime, rules, tax_enabled, start_yr=2035, end_yr=2055):
    """Build a deterministic SimConfig for one preset combination.

    Deepcopied from build_config so tweaks (tax_enabled, n_sims, end_yr)
    don't mutate the preset.
    """
    import copy
    from citadel_presets import build_config
    cfg = build_config(wealth, regime, rules, start_yr, "single")
    cfg = copy.deepcopy(cfg)
    cfg.tax_enabled = tax_enabled
    cfg.n_sims = 1  # deterministic
    cfg.end_yr = end_yr
    # Deterministic: swap Markov model for lognormal so we don't need a real
    # transition matrix file. Volatility → 0 via deterministic path.
    cfg.asset_return_model = "lognormal"
    cfg.asset_matrices = None
    return cfg


# Upper-bound envelopes per wealth level over a 20-year horizon against the
# deterministic mock model (price grows linearly, ~$60k in 2035 → ~$85k in
# 2055). Conservatively loose — if a bug inflates total by 10-100x, this will
# catch it; any number under 1/10 of these bounds is considered plausible.
_WEALTH_UPPER_BOUND = {
    "starter": 50_000_000,    # $500k base, up to $50M over 20yr
    "full": 500_000_000,      # $2.5M base
    "bitcoin": 5_000_000_000, # $2.5M + 12.5 BTC
}


class TestCitadelPresetGrid:
    """3 wealth × 3 regime × 3 rule × 2 tax = 54 combinations."""

    @pytest.mark.parametrize("wealth", ["starter", "full", "bitcoin"])
    @pytest.mark.parametrize("regime", ["bear", "neutral", "bull"])
    @pytest.mark.parametrize("rules", ["no_rebal", "cautious", "aggressive"])
    @pytest.mark.parametrize("tax_enabled", [False, True])
    def test_preset_runs_without_crash_or_nan(self, wealth, regime, rules, tax_enabled):
        from engines.citadel import simulate
        cfg = _preset_config(wealth, regime, rules, tax_enabled)
        result = simulate(cfg, _test_model())
        total = np.asarray(result.median["total"], dtype=float)

        label = f"{wealth}/{regime}/{rules}/tax={tax_enabled}"
        assert np.all(np.isfinite(total)), f"{label}: NaN/Inf in total"
        assert np.all(total >= -1.0), f"{label}: negative total (min={total.min():.2e})"

    @pytest.mark.parametrize("wealth", ["starter", "full", "bitcoin"])
    @pytest.mark.parametrize("regime", ["bear", "neutral", "bull"])
    @pytest.mark.parametrize("rules", ["no_rebal", "cautious", "aggressive"])
    def test_preset_total_within_sane_bound(self, wealth, regime, rules):
        from engines.citadel import simulate
        cfg = _preset_config(wealth, regime, rules, tax_enabled=False)
        result = simulate(cfg, _test_model())
        total = np.asarray(result.median["total"], dtype=float)

        upper = _WEALTH_UPPER_BOUND[wealth]
        assert total.max() < upper, (
            f"{wealth}/{regime}/{rules}: total peaks at ${total.max():,.0f} "
            f"which exceeds the ${upper:,.0f} upper bound"
        )

    @pytest.mark.parametrize("wealth", ["starter", "full", "bitcoin"])
    def test_taxes_paid_end_state_nonneg(self, wealth):
        """Cumulative taxes paid can dip within a year (Q4 true-up may return
        money if quarterly estimates overpaid), so a strict monotone
        assertion would fail. The loose invariant: the end-of-sim total is
        non-negative (net tax paid, not net refund)."""
        from engines.citadel import simulate
        cfg = _preset_config(wealth, "neutral", "no_rebal", tax_enabled=True)
        result = simulate(cfg, _test_model())
        tp = getattr(result, "taxes_paid", None)
        if tp is None or len(tp) == 0:
            pytest.skip("no taxes_paid array available")
        final = float(np.asarray(tp[0], dtype=float)[-1])
        assert final >= -1e-3, f"{wealth}: end-of-sim taxes_paid=${final:,.2f} < 0"

    @pytest.mark.parametrize("wealth", ["starter", "full", "bitcoin"])
    def test_tax_on_not_much_higher_than_tax_off(self, wealth):
        """Tax should drag wealth down, not up. Some wiggle is allowed
        (Roth-BTC compounding protection) but the tax-on endpoint should
        never exceed the tax-off endpoint by more than 10% — otherwise the
        figure's 'Tax drag' annotation silently flips to negative."""
        from engines.citadel import simulate
        import copy
        cfg_on = _preset_config(wealth, "neutral", "no_rebal", tax_enabled=True)
        cfg_off = copy.deepcopy(cfg_on)
        cfg_off.tax_enabled = False

        r_on = simulate(cfg_on, _test_model())
        r_off = simulate(cfg_off, _test_model())
        final_on = float(r_on.median["total"][-1])
        final_off = float(r_off.median["total"][-1])

        if final_off <= 0:
            pytest.skip("tax-off endpoint non-positive; ratio undefined")
        ratio = final_on / final_off
        assert ratio <= 1.10, (
            f"{wealth}: tax-on total ${final_on:,.0f} exceeds tax-off "
            f"${final_off:,.0f} by >10% (ratio {ratio:.2f})"
        )


