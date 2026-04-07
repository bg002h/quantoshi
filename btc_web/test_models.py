"""Price models, decomposition, multi-model overlay, palettes."""
import plotly.graph_objects as go
from btc_core import (PriceModel, _FitsBasedModel, BubbleModel, PowerLawModel,
                      S2FModel, QuantileRegressionModel,
                      LogisticModel, BrokenPowerLawModel)
from conftest import (
    M,
    Path,
    _ROOT,
    _CHECKLIST_OPTIONS,
    _SNAPSHOT_CONTROLS,
    _TAB_CONTROLS,
    _app_ctx,
    _build_mc_params,
    _decode_snapshot,
    _encode_snapshot,
    _q3,
    build_bubble_figure,
    build_dca_figure,
    build_retire_figure,
    go,
    np,
    pd,
    pytest,
    qr_price,
    today_t,
)


class TestPriceModelProtocol:
    def test_bubble_implements_protocol(self):
        bub = BubbleModel(M)
        assert isinstance(bub, PriceModel)

    def test_powerlaw_implements_protocol(self):
        pl = PowerLawModel(M.ols_intercept, M.ols_slope, M.price_years,
                           M.price_prices, M.genesis, M.QR_QUANTILES)
        assert isinstance(pl, PriceModel)

    def test_s2f_implements_protocol(self):
        s2f = S2FModel(M.price_years, M.price_prices, M.genesis)
        assert isinstance(s2f, PriceModel)



class TestBubbleModel:
    def setup_method(self):
        self.bub = BubbleModel(M)

    def test_fits_has_quantile_keys(self):
        assert set(self.bub.fits.keys()) == set(M.QR_QUANTILES)

    def test_colors_populated(self):
        assert len(self.bub.colors) > 0
        # Not all quantiles necessarily have colors (e.g. 0.86 may be in fits
        # but not in qr_colors), but colors should be a non-empty subset
        assert set(self.bub.colors.keys()).issubset(set(self.bub.quantiles))

    def test_quantized_true(self):
        assert self.bub.quantized is True

    def test_quantiles_sorted(self):
        assert self.bub.quantiles == sorted(self.bub.quantiles)

    def test_price_at_returns_positive(self):
        q = self.bub.quantiles[len(self.bub.quantiles) // 2]
        t = 10.0
        result = self.bub.price_at(q, t)
        assert float(result) > 0

    def test_price_at_array(self):
        q = self.bub.quantiles[0]
        ts = np.array([5.0, 10.0, 15.0])
        result = self.bub.price_at(q, ts)
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_short_name(self):
        assert self.bub.short_name == "bub"

    def test_name(self):
        assert self.bub.name == "Bubble Model"



class TestPowerLawModel:
    def setup_method(self):
        self.pl = PowerLawModel(M.ols_intercept, M.ols_slope, M.price_years,
                                M.price_prices, M.genesis, M.QR_QUANTILES)

    def test_fits_has_quantile_keys_from_qr_quantiles(self):
        # PL is built from M.QR_QUANTILES, which may differ from M.qr_fits keys
        assert set(self.pl.fits.keys()) == set(M.QR_QUANTILES)

    def test_fits_values_have_intercept_and_slope(self):
        for q, f in self.pl.fits.items():
            assert "intercept" in f
            assert "slope" in f

    def test_all_slopes_equal_ols(self):
        for q, f in self.pl.fits.items():
            np.testing.assert_allclose(f["slope"], M.ols_slope)

    def test_median_intercept_matches_ols(self):
        # Q50% should have z=0, so intercept ≈ ols_intercept
        q50 = min(self.pl.quantiles, key=lambda q: abs(q - 0.5))
        np.testing.assert_allclose(
            self.pl.fits[q50]["intercept"], M.ols_intercept, atol=0.01)

    def test_price_at_returns_positive(self):
        result = self.pl.price_at(0.5, 10.0)
        assert float(result) > 0

    def test_quantized_true(self):
        assert self.pl.quantized is True

    def test_colors_populated(self):
        assert len(self.pl.colors) == len(self.pl.quantiles)

    def test_short_name(self):
        assert self.pl.short_name == "pl"



class TestS2FModel:
    def setup_method(self):
        self.s2f = S2FModel(M.price_years, M.price_prices, M.genesis)

    def test_quantized_false(self):
        assert self.s2f.quantized is False

    def test_fits_is_none(self):
        assert self.s2f.fits is None

    def test_quantiles_empty(self):
        assert self.s2f.quantiles == []

    def test_colors_empty(self):
        assert self.s2f.colors == {}

    def test_price_at_scalar(self):
        result = self.s2f.price_at(0.5, 10.0)
        assert isinstance(result, float)
        assert result > 0

    def test_price_at_array(self):
        ts = np.array([5.0, 10.0, 15.0])
        result = self.s2f.price_at(0.5, ts)
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_find_percentile_returns_half(self):
        assert self.s2f.find_percentile(10.0, 50000) == 0.5

    def test_short_name(self):
        assert self.s2f.short_name == "s2f"



class TestQuantileRegressionModel:
    def setup_method(self):
        self.qr = QuantileRegressionModel(M)

    def test_short_name(self):
        assert self.qr.short_name == "qr"

    def test_fits_are_qr_fits(self):
        assert self.qr.fits is M.qr_fits

    def test_price_at_matches_qr_price(self):
        q = 0.5
        t = 10.0
        expected = qr_price(q, t, M.qr_fits)
        result = self.qr.price_at(q, t)
        np.testing.assert_allclose(result, expected)

    def test_quantized(self):
        assert self.qr.quantized is True



class TestFitsBasedModelMethods:
    def setup_method(self):
        self.bub = BubbleModel(M)

    def test_interp_price_exact_quantile(self):
        q = self.bub.quantiles[5]
        t = 10.0
        expected = float(self.bub.price_at(q, t))
        result = self.bub.interp_price(q, t)
        np.testing.assert_allclose(result, expected)

    def test_interp_price_between_quantiles(self):
        q_lo = self.bub.quantiles[3]
        q_hi = self.bub.quantiles[4]
        q_mid = (q_lo + q_hi) / 2
        t = 10.0
        p_lo = self.bub.interp_price(q_lo, t)
        p_hi = self.bub.interp_price(q_hi, t)
        p_mid = self.bub.interp_price(q_mid, t)
        assert p_lo <= p_mid <= p_hi

    def test_find_percentile_roundtrip(self):
        q = self.bub.quantiles[5]
        t = 10.0
        price = float(self.bub.price_at(q, t))
        recovered_q = self.bub.find_percentile(t, price)
        np.testing.assert_allclose(recovered_q, q, atol=0.01)

    def test_find_percentile_below_min(self):
        t = 10.0
        price = 0.001  # well below any model price
        result = self.bub.find_percentile(t, price)
        assert result == self.bub.quantiles[0]

    def test_find_percentile_above_max(self):
        t = 10.0
        price = 1e20  # well above any model price
        result = self.bub.find_percentile(t, price)
        assert result == self.bub.quantiles[-1]



class TestPriceModelRegistry:
    def test_registry_has_core_entries(self):
        import _app_ctx
        assert len(_app_ctx.PRICE_MODELS) >= 5

    def test_registry_keys(self):
        import _app_ctx
        assert {"bub", "pl", "s2f", "lppl", "exp"}.issubset(set(_app_ctx.PRICE_MODELS.keys()))

    def test_default_model_is_bubble(self):
        import _app_ctx
        assert _app_ctx.DEFAULT_MODEL is _app_ctx.PRICE_MODELS["bub"]

    def test_core_models_quantized(self):
        import _app_ctx
        quantized = {k for k, v in _app_ctx.PRICE_MODELS.items() if v.quantized}
        assert {"bub", "pl", "lppl", "exp"}.issubset(quantized)

    def test_all_models_implement_protocol(self):
        import _app_ctx
        for mdl in _app_ctx.PRICE_MODELS.values():
            assert isinstance(mdl, PriceModel)



class TestHybPPLDDModel:
    """HybPPL (DD) — double-damped, non-excess."""

    def test_instantiates(self):
        from btc_core import HybPPLDDModel
        import _app_ctx
        M = _app_ctx.M
        mdl = HybPPLDDModel(M.price_years, M.price_prices, M.QR_QUANTILES)
        assert mdl.short_name == "hybppl_dd"

    def test_lppl_log10_returns_finite(self):
        from btc_core import HybPPLDDModel
        import _app_ctx
        import numpy as np
        M = _app_ctx.M
        mdl = HybPPLDDModel(M.price_years, M.price_prices, M.QR_QUANTILES)
        for t in (1.0, 5.0, 10.0, 16.0):
            v = mdl._lppl_log10(np.array([t]))
            assert np.isfinite(v).all()
        v10 = mdl._lppl_log10(np.array([10.0]))
        assert 2.0 < v10[0] < 6.0

    def test_dd_included_in_price_models(self):
        import _app_ctx
        assert "hybppl_dd" in _app_ctx.PRICE_MODELS



class TestLPPLComponentDecomposition:
    """LPPL family: sum(components(t)) == _lppl_log10(t) to 1e-10."""

    T_TEST = np.array([1.0, 5.0, 10.0, 16.0, 30.0, 50.0])

    def _assert_invariant(self, model):
        comps = model.components(self.T_TEST)
        assert set(comps.keys()) == set(model.component_names), (
            f"{type(model).__name__}: components() keys != component_names")
        total = sum(comps.values())
        expected = model._lppl_log10(self.T_TEST)
        np.testing.assert_allclose(
            total, expected, rtol=0, atol=1e-10,
            err_msg=f"{type(model).__name__}: sum(components) != _lppl_log10")

    def test_lppl_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["lppl"])

    def test_lppl2_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["lp2"])

    def test_lppl3_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["lp3"])

    def test_lppl4_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["lp4"])

    def test_lppl_weighted_variants_inherit(self):
        import _app_ctx
        for key in ("lppl_w", "lp2_w", "lp3_w", "lp4_w"):
            self._assert_invariant(_app_ctx.PRICE_MODELS[key])

    def test_lppl4_n13_variants_inherit(self):
        import _app_ctx
        for key in ("lp4_n13", "lp4_w_n13"):
            self._assert_invariant(_app_ctx.PRICE_MODELS[key])

    def test_lppl_component_count(self):
        import _app_ctx
        assert len(_app_ctx.PRICE_MODELS["lppl"].component_names) == 3
        assert len(_app_ctx.PRICE_MODELS["lp2"].component_names) == 4
        assert len(_app_ctx.PRICE_MODELS["lp3"].component_names) == 5
        assert len(_app_ctx.PRICE_MODELS["lp4"].component_names) == 6

    def test_linppl_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["linppl"])

    def test_linppl_component_count(self):
        import _app_ctx
        assert len(_app_ctx.PRICE_MODELS["linppl"].component_names) == 3

    def test_hybppl_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["hybppl"])

    def test_hybppl_component_count(self):
        import _app_ctx
        assert len(_app_ctx.PRICE_MODELS["hybppl"].component_names) == 4

    def test_hybppl_dd_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["hybppl_dd"])

    def test_hybppl_dd_component_count(self):
        import _app_ctx
        assert len(_app_ctx.PRICE_MODELS["hybppl_dd"].component_names) == 4



class TestCompositeComponentDecomposition:
    """BM / EF: sum(components(t)) == _composite_log10(t) to 1e-10."""

    T_TEST = np.array([1.0, 5.0, 10.0, 16.0, 30.0, 50.0])

    def _assert_composite_invariant(self, model):
        comps = model.components(self.T_TEST)
        assert set(comps.keys()) == set(model.component_names)
        total = sum(comps.values())
        expected = model._composite_log10(self.T_TEST)
        np.testing.assert_allclose(
            total, expected, rtol=0, atol=1e-10,
            err_msg=f"{type(model).__name__}: sum(components) != _composite_log10")

    def test_bm_invariant(self):
        import _app_ctx
        self._assert_composite_invariant(_app_ctx.PRICE_MODELS["bub"])

    def test_bm_component_count(self):
        import _app_ctx
        assert _app_ctx.PRICE_MODELS["bub"].component_names == ["support", "bubbles"]

    def test_ef_invariant(self):
        import _app_ctx
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded (model_data_ef.pkl absent)")
        self._assert_composite_invariant(ef)

    def test_ef_component_count(self):
        import _app_ctx
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert ef.component_names == ["support", "bubbles"]



class TestDecompRegistry:
    def test_families_keys(self):
        import _app_ctx
        expected = {"bub", "ef", "lppl", "linppl", "hybppl", "hybppl_dd"}
        assert set(_app_ctx.DECOMP_FAMILIES.keys()) == expected

    def test_families_labels(self):
        import _app_ctx
        assert _app_ctx.DECOMP_FAMILIES["bub"] == "BM"
        assert _app_ctx.DECOMP_FAMILIES["lppl"] == "LPPL (family)"
        assert _app_ctx.DECOMP_FAMILIES["hybppl_dd"] == "HybPPL (DD)"

    def test_palette_has_all_four_schemes(self):
        import _app_ctx
        assert set(_app_ctx.DECOMP_COLORS.keys()) == {"default", "cb-brian", "cb-rg", "cb-full"}
        for key, colors in _app_ctx.DECOMP_COLORS.items():
            assert len(colors) == 7, f"{key} palette has {len(colors)} colors, expected 7"
            for c in colors:
                assert c.startswith("#") and len(c) == 7

    def test_sum_color_has_all_four_schemes(self):
        import _app_ctx
        assert set(_app_ctx.DECOMP_SUM_COLOR.keys()) == {"default", "cb-brian", "cb-rg", "cb-full"}



class TestResolveDecompModelKey:
    def test_non_lppl_families_pass_through(self):
        from callbacks.charts import _resolve_decomp_model_key
        assert _resolve_decomp_model_key("bub", [3], [], []) == "bub"
        assert _resolve_decomp_model_key("hybppl_dd", [3], [], []) == "hybppl_dd"
        assert _resolve_decomp_model_key("linppl", [], [], []) == "linppl"
        assert _resolve_decomp_model_key("hybppl", [1, 2], [], []) == "hybppl"
        assert _resolve_decomp_model_key("ef", [3], [], []) == "ef"

    def test_empty_family_returns_none(self):
        from callbacks.charts import _resolve_decomp_model_key
        assert _resolve_decomp_model_key("", [3], [], []) is None
        assert _resolve_decomp_model_key(None, [3], [], []) is None

    def test_lppl_single_nfreq_resolves(self):
        from callbacks.charts import _resolve_decomp_model_key
        assert _resolve_decomp_model_key("lppl", [1], [], []) == "lppl"
        assert _resolve_decomp_model_key("lppl", [2], [], []) == "lp2"
        assert _resolve_decomp_model_key("lppl", [3], [], []) == "lp3"
        assert _resolve_decomp_model_key("lppl", [4], [], []) == "lp4"

    def test_lppl_weighted_modifier(self):
        from callbacks.charts import _resolve_decomp_model_key
        assert _resolve_decomp_model_key("lppl", [1], ["weighted"], []) == "lppl_w"
        assert _resolve_decomp_model_key("lppl", [3], ["weighted"], []) == "lp3_w"
        assert _resolve_decomp_model_key("lppl", [4], ["weighted"], []) == "lp4_w"

    def test_lppl_no13_modifier(self):
        from callbacks.charts import _resolve_decomp_model_key
        assert _resolve_decomp_model_key("lppl", [4], [], ["no13"]) == "lp4_n13"
        assert _resolve_decomp_model_key("lppl", [4], ["weighted"], ["no13"]) == "lp4_w_n13"

    def test_lppl_zero_or_multi_returns_none(self):
        from callbacks.charts import _resolve_decomp_model_key
        assert _resolve_decomp_model_key("lppl", [], [], []) is None
        assert _resolve_decomp_model_key("lppl", [1, 2], [], []) is None
        assert _resolve_decomp_model_key("lppl", [1, 2, 3, 4], [], []) is None
        assert _resolve_decomp_model_key("lppl", None, [], []) is None



class TestUpdateDecompOptions:
    def test_empty_family_hides_body(self):
        from callbacks.charts import update_decomp_options
        opts, warning, style = update_decomp_options("", [3], [], [])
        assert opts == []
        assert warning == []
        assert style == {"display": "none"}

    def test_bm_shows_2_components(self):
        from callbacks.charts import update_decomp_options
        opts, warning, style = update_decomp_options("bub", [3], [], [])
        assert style == {"display": "block"}
        assert warning == []
        values = [o["value"] for o in opts]
        assert values == ["support", "bubbles"]

    def test_hybppl_dd_shows_4_components(self):
        from callbacks.charts import update_decomp_options
        opts, warning, style = update_decomp_options("hybppl_dd", [3], [], [])
        assert len(opts) == 4

    def test_lppl_single_nfreq_shows_components(self):
        from callbacks.charts import update_decomp_options
        opts, warning, style = update_decomp_options("lppl", [3], [], [])
        assert style == {"display": "block"}
        assert warning == []
        assert len(opts) == 5  # LPPL3 = 5 components

    def test_lppl_zero_nfreq_shows_warning(self):
        from callbacks.charts import update_decomp_options
        opts, warning, style = update_decomp_options("lppl", [], [], [])
        assert opts == []
        assert style == {"display": "block"}
        assert warning != []

    def test_lppl_multi_nfreq_shows_warning(self):
        from callbacks.charts import update_decomp_options
        opts, warning, style = update_decomp_options("lppl", [1, 2, 3], [], [])
        assert opts == []
        assert style == {"display": "block"}
        assert warning != []

    def test_lppl_weighted_modifier_resolves(self):
        from callbacks.charts import update_decomp_options
        opts, _, _ = update_decomp_options("lppl", [3], ["weighted"], [])
        assert len(opts) == 5



class TestPruneDecompValue:
    def test_empty_family_clears_value(self):
        from callbacks.charts import _prune_decomp_value
        assert _prune_decomp_value("", [{"value": "a"}], ["a"]) == []

    def test_prune_preserves_valid_values(self):
        from callbacks.charts import _prune_decomp_value
        opts = [{"value": "a"}, {"value": "b"}, {"value": "__sum__"}]
        assert _prune_decomp_value("bub", opts, ["a", "__sum__"]) == ["a", "__sum__"]

    def test_prune_drops_invalid_values(self):
        from callbacks.charts import _prune_decomp_value
        opts = [{"value": "support"}, {"value": "bubbles"}, {"value": "__sum__"}]
        assert _prune_decomp_value("bub", opts, ["damped osc", "support"]) == ["support"]

    def test_prune_empty_current(self):
        from callbacks.charts import _prune_decomp_value
        opts = [{"value": "a"}]
        assert _prune_decomp_value("bub", opts, []) == []
        assert _prune_decomp_value("bub", opts, None) == []



class TestDecompositionTraces:
    """Verify build_bubble_figure renders decomposition traces when active."""

    def _base_p(self, **overrides):
        p = dict(
            selected_qs=[0.5], shade=True, show_ols=False, show_ucl=False,
            show_data=False, show_today=False, show_legend=False,
            minor_grid=False, show_comp=False, show_sup=False,
            xscale="log", yscale="log", xmin=2015, xmax=2030,
            ymin=1, ymax=100000, n_future=0, pt_size=3, pt_alpha=0.3,
            stack=0, show_stack=False, use_lots=False,
            lots=[], legend_pos="outside", comp_color="#FFD700",
            comp_lw=2.0, sup_color="#888888", sup_lw=1.5,
            active_models=[], palette="default", scanner_lines=[],
            user_model=None, qs_mode=[],
            decomp_model="", decomp_components=[], decomp_mode="individual",
            lppl_n_freqs=[], lppl_weighted=[], lppl_no_13=[],
        )
        p.update(overrides)
        return p

    def test_no_model_no_extra_traces(self):
        import _app_ctx
        from figures.bubble import build_bubble_figure
        fig = build_bubble_figure(_app_ctx.M, self._base_p())
        decomp_traces = [t for t in fig.data
                         if getattr(t, 'name', None) and " | " in t.name]
        assert decomp_traces == []

    def test_single_trace_per_decomposition(self):
        """Exactly ONE trace appears when components are selected."""
        import _app_ctx
        from figures.bubble import build_bubble_figure
        fig = build_bubble_figure(_app_ctx.M, self._base_p(
            decomp_model="bub", decomp_components=["support", "bubbles"]))
        trace_names = [t.name for t in fig.data if getattr(t, 'name', None)]
        bm_decomp = [n for n in trace_names if n.startswith("BM | ")]
        assert len(bm_decomp) == 1

    def test_all_components_selected_gives_full_model(self):
        """All components selected → label says 'full model'."""
        import _app_ctx
        from figures.bubble import build_bubble_figure
        fig = build_bubble_figure(_app_ctx.M, self._base_p(
            decomp_model="bub", decomp_components=["support", "bubbles"]))
        trace_names = [t.name for t in fig.data if getattr(t, 'name', None)]
        full = [n for n in trace_names if "full model" in n]
        assert len(full) == 1

    def test_partial_selection_shows_fraction(self):
        """Subset selection label shows fraction (e.g., '1/2 components')."""
        import _app_ctx
        from figures.bubble import build_bubble_figure
        fig = build_bubble_figure(_app_ctx.M, self._base_p(
            decomp_model="bub", decomp_components=["support"]))
        trace_names = [t.name for t in fig.data if getattr(t, 'name', None)]
        partial = [n for n in trace_names if "1/2 components" in n]
        assert len(partial) == 1





@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestModelMCInterface:
    """All quantized models must support find_percentile and interp_price for MC."""

    def test_all_quantized_have_find_percentile(self):
        import _app_ctx
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue
            assert hasattr(mdl, 'find_percentile'), f"{key} missing find_percentile"
            assert callable(mdl.find_percentile), f"{key}.find_percentile not callable"

    def test_all_quantized_have_interp_price(self):
        import _app_ctx
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue
            assert hasattr(mdl, 'interp_price'), f"{key} missing interp_price"
            assert callable(mdl.interp_price), f"{key}.interp_price not callable"

    def test_find_percentile_returns_float(self):
        import _app_ctx
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue
            pct = mdl.find_percentile(16.0, 60000.0)
            assert isinstance(pct, float), f"{key}.find_percentile returned {type(pct)}"
            assert 0.0 <= pct <= 1.0, f"{key} percentile {pct} out of range"

    def test_interp_price_returns_positive(self):
        import _app_ctx
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue
            price = mdl.interp_price(0.5, 16.0)
            assert isinstance(price, float), f"{key}.interp_price returned {type(price)}"
            assert price > 0, f"{key} price {price} not positive"


# ═══════════════════════════════════════════════════════════════════════════════
# Section: Multi-model overlay (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════



class TestMultiModelBubbleFigure:
    """Bubble figure with active_models=["pl"]."""

    def test_pl_overlay_adds_traces(self):
        from datetime import date
        yr_now = date.today().year
        p_base = dict(
            selected_qs=[0.10, 0.50],
            shade=False, show_ols=False, show_data=False, show_today=False,
            show_legend=False, minor_grid=False,
            show_comp=False, show_sup=False,
            xscale="log", yscale="log",
            xmin=2012, xmax=yr_now + 4,
            ymin=0.01, ymax=1e7,
            n_future=0, pt_size=3, pt_alpha=0.3,
            stack=0, show_stack=False, use_lots=False, lots=[],
            comp_color="#FFD700", comp_lw=2.0,
            sup_color="#888888", sup_lw=1.5,
        )
        from figures import build_bubble_figure
        fig_no_pl = build_bubble_figure(M, dict(p_base, active_models=[]))
        fig_with_pl = build_bubble_figure(M, dict(p_base, active_models=["pl"]))
        assert len(fig_with_pl.data) > len(fig_no_pl.data)

    def test_pl_traces_have_dot_dash(self):
        from datetime import date
        yr_now = date.today().year
        from figures import build_bubble_figure
        fig = build_bubble_figure(M, dict(
            selected_qs=[0.50], shade=False, show_ols=False, show_data=False,
            show_today=False, show_legend=False, minor_grid=False,
            show_comp=False, show_sup=False,
            xscale="log", yscale="log",
            xmin=2012, xmax=yr_now + 4,
            ymin=0.01, ymax=1e7, n_future=0, pt_size=3, pt_alpha=0.3,
            stack=0, show_stack=False, use_lots=False, lots=[],
            comp_color="#FFD700", comp_lw=2.0, sup_color="#888888", sup_lw=1.5,
            active_models=["pl"],
        ))
        pl_traces = [t for t in fig.data if t.name and t.legendgroup == "pl"]
        assert len(pl_traces) > 0
        assert pl_traces[0].line.dash == "dot"



class TestMultiModelDcaFigure:
    """DCA figure with active_models=["pl"]."""

    def test_pl_overlay_doesnt_crash(self):
        from datetime import date
        yr_now = date.today().year
        from figures import build_dca_figure
        fig, _ = build_dca_figure(M, dict(
            start_stack=0, use_lots=False,
            amount=100.0, freq="Monthly",
            start_yr=yr_now, end_yr=yr_now + 5,
            disp_mode="btc", log_y=False, show_today=False,
            show_legend=False, minor_grid=False,
            selected_qs=[0.50], lots=[],
            sc_enabled=False, sc_loan_amount=0,
            sc_rate=13.0, sc_loan_type="interest_only",
            sc_term_months=48.0, sc_repeats=0, sc_rollover=False,
            sc_entry_mode="live", sc_custom_price=80000,
            sc_tax_rate=0.33, sc_live_price=None,
            active_models=["pl"],
        ))
        pl_traces = [t for t in fig.data if t.name and t.legendgroup == "pl"]
        assert len(pl_traces) > 0



@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestMcModelSrc:
    """Phase 4: MC model-source dropdown tests."""

    def test_mc_path_key_includes_model_src(self):
        from mc_overlay import _mc_path_key
        key = _mc_path_key({"mc_model_src": "pl"}, "dca")
        assert key["mc_model_src"] == "pl"

    def test_mc_path_key_defaults_to_bub(self):
        from mc_overlay import _mc_path_key
        key = _mc_path_key({}, "dca")
        assert key["mc_model_src"] == "bub"

    def test_resolve_model_returns_bub(self):
        from mc_overlay import _resolve_model
        mdl = _resolve_model({"mc_model_src": "bub"})
        assert mdl is _app_ctx.PRICE_MODELS["bub"]

    def test_resolve_model_default(self):
        from mc_overlay import _resolve_model
        mdl = _resolve_model({})
        assert mdl is _app_ctx.DEFAULT_MODEL

    def test_resolve_model_pl(self):
        from mc_overlay import _resolve_model
        mdl = _resolve_model({"mc_model_src": "pl"})
        assert mdl is _app_ctx.PRICE_MODELS["pl"]

    def test_resolve_model_nonquantized_falls_back(self):
        from mc_overlay import _resolve_model
        mdl = _resolve_model({"mc_model_src": "s2f"})
        assert mdl is _app_ctx.DEFAULT_MODEL

    def test_build_mc_params_includes_model_src(self):
        from callbacks import _build_mc_params
        p = _build_mc_params(
            mc_enable=True, mc_amount=100, mc_infl=0,
            mc_bins=5, mc_sims=100, mc_years=10,
            mc_freq="Monthly", mc_window=None,
            mc_start_yr=2028, mc_entry_q=50,
            mc_cached=None, mc_live_price=0,
            mc_model_src="pl",
        )
        assert p["mc_model_src"] == "pl"

    def test_build_mc_params_defaults_model_src(self):
        from callbacks import _build_mc_params
        p = _build_mc_params(
            mc_enable=True, mc_amount=100, mc_infl=0,
            mc_bins=5, mc_sims=100, mc_years=10,
            mc_freq="Monthly", mc_window=None,
            mc_start_yr=2028, mc_entry_q=50,
            mc_cached=None, mc_live_price=0,
        )
        assert p["mc_model_src"] == "bub"

    def test_snapshot_roundtrip_with_model_src(self):
        from snapshot import _encode_snapshot, _decode_snapshot
        state = {"dca-mc-model-src:value": "pl"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["dca-mc-model-src:value"] == "pl"

    def test_old_snapshot_pads_model_src(self):
        """Old snapshots without mc-model-src fields decode with None (defaults)."""
        from snapshot import _encode_snapshot, _decode_snapshot, _SNAPSHOT_CONTROLS
        # Build state with only old controls (no mc-model-src)
        state = {"main-tabs:active_tab": "dca"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        # mc-model-src fields should not be present (None → skipped)
        assert "dca-mc-model-src:value" not in decoded

    def test_tab_controls_include_model_src(self):
        from callbacks import _TAB_CONTROLS
        assert "hm-mc-model-src" in _TAB_CONTROLS["heatmap"]
        assert "dca-mc-model-src" in _TAB_CONTROLS["dca"]
        assert "ret-mc-model-src" in _TAB_CONTROLS["retire"]
        assert "sc-mc-model-src" in _TAB_CONTROLS["supercharge"]



@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestPhase5Polish:
    """Phase 5: dash styles, S2F overlay, model attributes."""

    def test_dash_styles(self):
        from btc_core import BubbleModel, PowerLawModel, S2FModel
        assert BubbleModel.dash_style == "solid"
        assert PowerLawModel.dash_style == "dot"
        assert S2FModel.dash_style == "dot"

    def test_all_models_have_dash_style(self):
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            assert hasattr(mdl, "dash_style"), f"{key} missing dash_style"
            assert mdl.dash_style in ("solid", "dot", "longdash", "dash", "dashdot", "longdashdot")

    def test_s2f_bubble_overlay(self):
        from figures import build_bubble_figure
        yr_now = pd.Timestamp.today().year
        fig = build_bubble_figure(M, dict(
            selected_qs=[0.5], xscale="log", yscale="log",
            xmin=2012, xmax=yr_now + 4, ymin=0.01, ymax=1e7,
            shade=False, show_ols=False, show_data=False, show_today=False,
            show_legend=False, minor_grid=False,
            show_comp=False, show_sup=False,
            n_future=0, pt_size=3, pt_alpha=0.3,
            stack=0, show_stack=False, use_lots=False, lots=[],
            comp_color="#FFD700", comp_lw=2.0, sup_color="#888888", sup_lw=1.5,
            active_models=["s2f"],
        ))
        s2f_traces = [t for t in fig.data if t.name and t.legendgroup == "s2f"]
        assert len(s2f_traces) == 1
        assert s2f_traces[0].line.dash == "dot"

    def test_s2f_dca_overlay(self):
        from figures import build_dca_figure
        yr_now = pd.Timestamp.today().year
        fig, _ = build_dca_figure(M, dict(
            start_stack=0, use_lots=False,
            amount=100.0, freq="Monthly",
            start_yr=yr_now, end_yr=yr_now + 5,
            disp_mode="btc", log_y=False, show_today=False,
            show_legend=False, minor_grid=False,
            selected_qs=[0.50], lots=[],
            sc_enabled=False, sc_loan_amount=0,
            sc_rate=13.0, sc_loan_type="interest_only",
            sc_term_months=48.0, sc_repeats=0, sc_rollover=False,
            sc_entry_mode="live", sc_custom_price=80000,
            sc_tax_rate=0.33, sc_live_price=None,
            active_models=["s2f"],
        ))
        s2f_traces = [t for t in fig.data if t.name and t.legendgroup == "s2f"]
        assert len(s2f_traces) == 1
        assert s2f_traces[0].line.dash == "dot"

    def test_s2f_retire_overlay(self):
        from figures import build_retire_figure
        fig, _ = build_retire_figure(M, dict(
            start_stack=1.0, use_lots=False,
            wd_amount=5000, freq="Monthly",
            start_yr=2031, end_yr=2040, inflation=4.0,
            disp_mode="btc", log_y=False,
            annotate=False, show_legend=False, minor_grid=False,
            legend_pos="outside",
            selected_qs=[0.5], lots=[],
            active_models=["s2f"],
        ))
        s2f_traces = [t for t in fig.data if t.name and t.legendgroup == "s2f"]
        assert len(s2f_traces) == 1

    def test_pl_uses_dot_dash(self):
        from figures import build_bubble_figure
        yr_now = pd.Timestamp.today().year
        fig = build_bubble_figure(M, dict(
            selected_qs=[0.5], xscale="log", yscale="log",
            xmin=2012, xmax=yr_now + 4, ymin=0.01, ymax=1e7,
            shade=False, show_ols=False, show_data=False, show_today=False,
            show_legend=False, minor_grid=False,
            show_comp=False, show_sup=False,
            n_future=0, pt_size=3, pt_alpha=0.3,
            stack=0, show_stack=False, use_lots=False, lots=[],
            comp_color="#FFD700", comp_lw=2.0, sup_color="#888888", sup_lw=1.5,
            active_models=["pl"],
        ))
        pl_traces = [t for t in fig.data if t.name and t.legendgroup == "pl"]
        assert len(pl_traces) > 0
        assert pl_traces[0].line.dash == "dot"



@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestPalettes:
    """Test palette registry and palette-aware color functions."""

    def test_get_palette_default(self):
        from figures.common import _get_palette
        pal = _get_palette({})
        assert pal is _app_ctx.PALETTES["default"]

    def test_get_palette_cb_rg(self):
        from figures.common import _get_palette
        pal = _get_palette({"palette": "cb-rg"})
        assert pal is _app_ctx.PALETTES["cb-rg"]

    def test_get_palette_unknown_falls_back(self):
        from figures.common import _get_palette
        pal = _get_palette({"palette": "nonexistent"})
        assert pal is _app_ctx.PALETTES["default"]

    def test_thermal_color_default_unchanged(self):
        from figures.common import _thermal_color
        assert _thermal_color(0.50).lower() == "#bdbdbd"

    def test_thermal_color_cb_rg_differs(self):
        from figures.common import _thermal_color
        pal = _app_ctx.PALETTES["cb-rg"]
        assert _thermal_color(0.90) != _thermal_color(0.90, pal)

    def test_all_palettes_have_required_keys(self):
        required = {"thermal_stops", "non_quantized_model", "delay_colors",
                    "annot_colors", "today_line", "hm_c_lo", "hm_c_mid1",
                    "hm_c_mid2", "hm_c_hi", "hm_loss_text", "hm_exceptional_text"}
        for name, pal in _app_ctx.PALETTES.items():
            missing = required - set(pal.keys())
            assert not missing, f"Palette {name!r} missing keys: {missing}"

    def test_all_palettes_thermal_stops_count(self):
        for name, pal in _app_ctx.PALETTES.items():
            assert len(pal["thermal_stops"]) == 12, f"{name} has wrong stop count"

    def test_snapshot_roundtrip_palette(self):
        from snapshot import (_encode_snapshot, _decode_snapshot,
                              _SNAPSHOT_CONTROLS)
        state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
        state["palette-store:data"] = "cb-rg"
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded.get("palette-store:data") == "cb-rg"

    def test_build_bubble_all_palettes(self):
        from figures import build_bubble_figure
        yr_now = pd.Timestamp.today().year
        for pal_key in _app_ctx.PALETTES:
            fig = build_bubble_figure(M, dict(
                selected_qs=[0.5], shade=False, show_data=False,
                show_today=False, show_legend=False, minor_grid=False,
                show_comp=False, show_sup=False, xscale="log", yscale="log",
                xmin=2012, xmax=yr_now + 4, ymin=1, ymax=1e6,
                n_future=1, pt_size=2, pt_alpha=0.2,
                stack=0, show_stack=False, use_lots=False, lots=[],
                comp_color="#FFD700", comp_lw=2, sup_color="#888", sup_lw=1.5,
                palette=pal_key,
            ))
            assert fig is not None, f"bubble failed for {pal_key}"

    def test_build_dca_all_palettes(self):
        from figures import build_dca_figure
        for pal_key in _app_ctx.PALETTES:
            fig, _ = build_dca_figure(M, dict(
                start_stack=0, use_lots=False, amount=100, freq="Monthly",
                start_yr=2024, end_yr=2030, disp_mode="btc",
                log_y=False, show_today=False, show_legend=False,
                minor_grid=False, selected_qs=[0.5], lots=[],
                sc_enabled=False, sc_loan_amount=0, sc_rate=13.0,
                sc_loan_type="interest_only", sc_term_months=48,
                sc_repeats=0, sc_rollover=False, sc_entry_mode="live",
                sc_custom_price=80000, sc_tax_rate=0.33, sc_live_price=None,
                palette=pal_key,
            ))
            assert fig is not None, f"dca failed for {pal_key}"


_EF_PKL = str(_ROOT / "model_data_ef.pkl")
_EF_SKIP = not Path(_EF_PKL).exists()



@pytest.mark.skipif(_EF_SKIP, reason="model_data_ef.pkl not found")
class TestEmpiricalFloorModel:
    """Tests for EmpiricalFloorModel."""

    @pytest.fixture(autouse=True)
    def _load_model(self):
        from btc_core import EmpiricalFloorModel
        self.model = EmpiricalFloorModel(_EF_PKL)

    def test_protocol_fields(self):
        assert self.model.name == "BM Empirical Floor"
        assert self.model.short_name == "ef"
        assert self.model.quantized is True
        assert isinstance(self.model.quantiles, list)
        assert len(self.model.quantiles) > 10
        assert isinstance(self.model.colors, dict)
        assert isinstance(self.model.fits, dict)
        assert 0.5 in self.model.fits

    def test_price_at_scalar(self):
        p = self.model.price_at(0.5, 10.0)
        assert float(p) > 0

    def test_price_at_array(self):
        t = np.array([5.0, 10.0, 15.0])
        prices = self.model.price_at(0.5, t)
        assert len(prices) == 3
        assert all(p > 0 for p in prices)

    def test_quantile_ordering(self):
        p10 = float(self.model.price_at(0.1, 10.0))
        p50 = float(self.model.price_at(0.5, 10.0))
        p90 = float(self.model.price_at(0.9, 10.0))
        assert p10 < p50
        assert p50 < p90

    def test_interp_price(self):
        p = self.model.interp_price(0.37, 10.0)
        assert p > 0

    def test_find_percentile(self):
        t = 12.0
        p50 = float(self.model.price_at(0.5, t))
        q = self.model.find_percentile(t, p50)
        assert abs(q - 0.5) < 0.1

    def test_dash_style(self):
        assert self.model.dash_style == "longdash"



class TestCompositeModelBands:
    """Test asymmetric shrinking Gaussian band behavior."""

    @pytest.fixture(autouse=True)
    def _load_model(self):
        import _app_ctx
        model = _app_ctx.PRICE_MODELS.get("bub")
        if model is None:
            pytest.skip("BubbleModel not available")
        self.model = model

    def test_bands_narrow_over_time(self):
        """σ(t) decreases → ratio of Q50/Q10 should be smaller at late t."""
        p10_early = float(self.model.price_at(0.1, 5.0))
        p50_early = float(self.model.price_at(0.5, 5.0))
        p10_late = float(self.model.price_at(0.1, 30.0))
        p50_late = float(self.model.price_at(0.5, 30.0))
        ratio_early = p50_early / p10_early
        ratio_late = p50_late / p10_late
        assert ratio_early > ratio_late

    def test_asymmetric_bands(self):
        """Downside band narrower than upside at late times."""
        t = 30.0
        p50 = np.log10(float(self.model.price_at(0.5, t)))
        p10 = np.log10(float(self.model.price_at(0.1, t)))
        p90 = np.log10(float(self.model.price_at(0.9, t)))
        down_width = p50 - p10
        up_width = p90 - p50
        assert up_width > down_width

    def test_quantile_ordering_preserved(self):
        """Q1 < Q10 < Q50 < Q90 < Q99 at all times."""
        for t in [3.0, 10.0, 30.0, 50.0]:
            prices = [float(self.model.price_at(q, t))
                      for q in [0.01, 0.1, 0.5, 0.9, 0.99]]
            for i in range(len(prices) - 1):
                assert prices[i] < prices[i + 1]

    def test_q1_never_exceeds_q50(self):
        """The bug this change fixes: Q1% must never exceed Q50%."""
        for t in [10, 20, 30, 40, 50, 60]:
            p1 = float(self.model.price_at(0.01, t))
            p50 = float(self.model.price_at(0.5, t))
            assert p1 < p50



@pytest.mark.skipif(_EF_SKIP, reason="model_data_ef.pkl not found")
class TestEmpiricalFloorComposite:
    """Test EmpiricalFloorModel with _CompositeModel bands."""

    @pytest.fixture(autouse=True)
    def _load_model(self):
        from btc_core import EmpiricalFloorModel
        self.model = EmpiricalFloorModel(_EF_PKL)

    def test_bands_narrow_over_time(self):
        p10_early = float(self.model.price_at(0.1, 5.0))
        p50_early = float(self.model.price_at(0.5, 5.0))
        p10_late = float(self.model.price_at(0.1, 30.0))
        p50_late = float(self.model.price_at(0.5, 30.0))
        ratio_early = p50_early / p10_early
        ratio_late = p50_late / p10_late
        assert ratio_early > ratio_late

    def test_quantile_ordering_preserved(self):
        for t in [3.0, 10.0, 30.0, 50.0]:
            prices = [float(self.model.price_at(q, t))
                      for q in [0.01, 0.1, 0.5, 0.9, 0.99]]
            for i in range(len(prices) - 1):
                assert prices[i] < prices[i + 1]



class TestModelScanner:
    def test_solve_for_quantile(self):
        """Given price and date, find_percentile returns valid quantile."""
        import _app_ctx
        t = today_t(_app_ctx.M.genesis)
        for mdl in _app_ctx.PRICE_MODELS.values():
            pct = mdl.find_percentile(t, 70000)
            assert 0 <= pct <= 1

    def test_solve_for_price(self):
        """Given quantile and date, price_at returns positive price."""
        import _app_ctx
        t = today_t(_app_ctx.M.genesis)
        for mdl in _app_ctx.PRICE_MODELS.values():
            p = float(mdl.price_at(0.5, t))
            assert p > 0

    def test_solve_for_date(self):
        """Root-finding for date works for reasonable inputs."""
        from callbacks.scanner import _solve_date
        import _app_ctx
        for mdl in _app_ctx.PRICE_MODELS.values():
            if not mdl.quantized:
                continue
            result = _solve_date(mdl, 0.5, 1_000_000)
            # Some models may not reach $1M in range — that's OK
            assert isinstance(result, str)

    def test_qr_model_registered(self):
        import _app_ctx
        assert "qr" in _app_ctx.PRICE_MODELS
        assert _app_ctx.PRICE_MODELS["qr"].name == "Quantile Regression"



class TestBubbleModelToggle:
    """bub-model-show includes 'bub' checked by default."""

    def test_bub_in_model_show_options(self):
        """The bubble model appears in Display Models checklist."""
        from layout.bubble import _bubble_controls
        controls = _bubble_controls()
        # Find the bub-model-show checklist
        # Note: Dash components may be falsy (e.g. empty value=[]) so use `is not None`
        def find_checklist(component):
            if hasattr(component, 'id') and component.id == 'bub-model-show':
                return component
            if hasattr(component, 'children'):
                kids = component.children
                if isinstance(kids, list):
                    for c in kids:
                        r = find_checklist(c)
                        if r is not None: return r
                elif kids is not None:
                    r = find_checklist(kids)
                    if r is not None: return r
            return None
        cl = find_checklist(controls)
        assert cl is not None
        option_values = [o["value"] for o in cl.options]
        assert "bub" in option_values
        assert option_values[0] == "bub"  # first in list
        assert "bub" in cl.value  # checked by default



class TestBubbleModelGating:
    """Main BM traces are conditional on 'bub' in active_models."""

    _BASE = dict(
        selected_qs=[0.5] if 0.5 in _app_ctx.DEFAULT_MODEL.fits else [0.10],
        shade=True, show_ols=False, show_ucl=False,
        show_data=False, show_today=False,
        show_legend=False, minor_grid=False,
        show_comp=True, show_sup=True,
        xscale="log", yscale="log",
        xmin=2012, xmax=2030,
        ymin=0.01, ymax=1e7,
        n_future=3, pt_size=3, pt_alpha=0.3,
        stack=0, show_stack=False, use_lots=False, lots=[],
        comp_color="#FFD700", comp_lw=2.0,
        sup_color="#888888", sup_lw=1.5,
        palette="default",
    )

    def test_bub_active_draws_traces(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub"]))
        names = [t.name for t in fig.data if t.name]
        assert any("Bubble composite" in n for n in names)
        assert any("Bubble support" in n for n in names)

    def test_bub_inactive_hides_traces(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=[]))
        # No traces should lack legendgroup (BM traces lack it; overlays always set it)
        bm_traces = [t for t in fig.data if t.name
                     and not getattr(t, "legendgroup", None)
                     and t.name not in ("Price data", "Lots")]
        assert len(bm_traces) == 0, f"BM traces should be hidden, found: {[t.name for t in bm_traces]}"

    def test_bub_inactive_preserves_data_scatter(self):
        """Data scatter, OLS, UCL, today line survive when BM is off."""
        fig = build_bubble_figure(M, dict(self._BASE,
            active_models=[], show_data=True, show_today=True,
            show_ols=True, show_ucl=True))
        names = [t.name for t in fig.data if t.name]
        assert any("Price data" in n for n in names)

    def test_bub_inactive_still_has_axis_config(self):
        """Even with BM hidden, chart should render without error."""
        fig = build_bubble_figure(M, dict(self._BASE, active_models=[]))
        assert isinstance(fig, go.Figure)
        assert fig.layout.xaxis.type in ("log", "linear", "-")



class TestEFCompositeOverlay:
    """EF overlay renders composite/support/future when enabled."""

    _BASE = dict(
        selected_qs=[0.5] if 0.5 in _app_ctx.DEFAULT_MODEL.fits else [0.10],
        shade=False, show_ols=False, show_ucl=False,
        show_data=False, show_today=False,
        show_legend=False, minor_grid=False,
        show_comp=True, show_sup=True,
        xscale="log", yscale="log",
        xmin=2012, xmax=2030,
        ymin=0.01, ymax=1e7,
        n_future=3, pt_size=3, pt_alpha=0.3,
        stack=0, show_stack=False, use_lots=False, lots=[],
        comp_color="#FFD700", comp_lw=2.0,
        sup_color="#888888", sup_lw=1.5,
        palette="default",
    )

    def test_ef_overlay_draws_composite(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["ef"]))
        names = [t.name for t in fig.data if t.name]
        assert any("EF" in n and "composite" in n for n in names)

    def test_ef_overlay_draws_support(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["ef"]))
        names = [t.name for t in fig.data if t.name]
        assert any("EF" in n and "support" in n for n in names)

    def test_ef_composite_uses_own_color(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["ef"]))
        comp_traces = [t for t in fig.data if t.name and "EF" in t.name and "composite" in t.name]
        assert len(comp_traces) > 0
        # EF composite uses palette model color (default palette)
        expected = _app_ctx.PALETTES["default"]["model_colors"]["ef"]
        assert comp_traces[0].line.color == expected

    def test_ef_no_composite_when_show_comp_off(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["ef"], show_comp=False))
        names = [t.name for t in fig.data if t.name]
        assert not any("composite" in n for n in names)

    def test_both_bub_and_ef_composite(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub", "ef"]))
        names = [t.name for t in fig.data if t.name]
        assert any("Bubble composite" in n for n in names)
        assert any("EF" in n and "composite" in n for n in names)





class TestModelR2:
    """All registered models get r2_per_quantile after startup."""

    def test_bubble_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("bub")
        assert hasattr(mdl, "r2_per_quantile")
        assert isinstance(mdl.r2_per_quantile, dict)
        assert len(mdl.r2_per_quantile) > 0
        for q, r2 in mdl.r2_per_quantile.items():
            assert 0 < r2 <= 1.0, f"BM Q{q}: R²={r2}"

    def test_pl_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("pl")
        assert hasattr(mdl, "r2_per_quantile")
        assert len(mdl.r2_per_quantile) > 0
        vals = list(mdl.r2_per_quantile.values())
        assert all(0 < v <= 1.0 for v in vals)

    def test_ef_model_has_r2(self):
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert hasattr(ef, "r2_per_quantile")
        assert len(ef.r2_per_quantile) > 0

    def test_s2f_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("s2f")
        assert hasattr(mdl, "r2_per_quantile")
        assert 0.5 in mdl.r2_per_quantile

    def test_lppl_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("lppl")
        assert hasattr(mdl, "r2_per_quantile")
        assert len(mdl.r2_per_quantile) > 0

    def test_exp_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("exp")
        assert hasattr(mdl, "r2_per_quantile")
        assert len(mdl.r2_per_quantile) > 0

    def test_qr_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("qr")
        assert hasattr(mdl, "r2_per_quantile")
        assert len(mdl.r2_per_quantile) > 0

    def test_ols_r2_on_model_data(self):
        _M = _app_ctx.M
        assert hasattr(_M, "ols_r2")
        assert isinstance(_M.ols_r2, float)
        assert 0.9 < _M.ols_r2 <= 1.0



class TestR2InLegend:
    """Legend labels include R² where available."""

    _BASE = dict(
        selected_qs=[0.5] if 0.5 in _app_ctx.DEFAULT_MODEL.fits else [0.10],
        shade=False, show_ols=True, show_ucl=True,
        show_data=False, show_today=False,
        show_legend=True, minor_grid=False,
        show_comp=True, show_sup=True,
        xscale="log", yscale="log",
        xmin=2012, xmax=2030,
        ymin=0.01, ymax=1e7,
        n_future=3, pt_size=3, pt_alpha=0.3,
        stack=0, show_stack=False, use_lots=False, lots=[],
        comp_color="#FFD700", comp_lw=2.0,
        sup_color="#888888", sup_lw=1.5,
        palette="default",
    )

    def test_bm_quantile_has_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub"]))
        q_traces = [t for t in fig.data if t.name and "Q" in t.name
                    and "%" in t.name and "R\u00b2" in t.name
                    and not getattr(t, "legendgroup", None)]
        assert len(q_traces) > 0, "BM quantile lines should show R²"

    def test_overlay_quantile_has_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub", "pl"]))
        pl_traces = [t for t in fig.data if t.name and t.legendgroup == "pl"
                     and "R\u00b2" in t.name]
        assert len(pl_traces) > 0, "PL overlay lines should show R²"

    def test_ols_has_r2(self):
        fig = build_bubble_figure(_app_ctx.M, dict(self._BASE, active_models=["bub"]))
        ols_traces = [t for t in fig.data if t.name and t.name.startswith("OLS")]
        assert len(ols_traces) > 0
        assert "R\u00b2" in ols_traces[0].name

    def test_s2f_has_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["s2f"]))
        s2f_traces = [t for t in fig.data if t.name and t.legendgroup == "s2f"]
        assert len(s2f_traces) > 0
        assert "R\u00b2" in s2f_traces[0].name

    def test_support_no_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub"]))
        sup_traces = [t for t in fig.data if t.name and "support" in t.name]
        for t in sup_traces:
            assert "R\u00b2" not in t.name, f"Support should not have R²: {t.name}"

    def test_ucl_no_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub"]))
        ucl_traces = [t for t in fig.data if t.name and "Unfairly Cheap" in t.name]
        for t in ucl_traces:
            assert "R\u00b2" not in t.name



class TestSimplifiedQuantilePanel:
    def test_default_quantile_options(self):
        from layout.common import _q_options_default, _DEFAULT_BANDS
        opts = _q_options_default()
        values = [o["value"] for o in opts]
        assert values == [b["value"] for b in _DEFAULT_BANDS]
        assert "median" in values
        assert "inner" in values
        assert "outer" in values

    def test_default_qs_values(self):
        from layout.common import _DEFAULT_QS
        assert _DEFAULT_QS == [0.01, 0.15, 0.50, 0.85, 0.99]

    def test_quantile_mode_toggle_in_bubble(self):
        from layout.bubble import _bubble_controls
        layout_str = repr(_bubble_controls())
        assert "bub-qs-mode" in layout_str
        assert "bub-qs-default-wrap" in layout_str
        assert "bub-qs-advanced-wrap" in layout_str
        assert "bub-qs-adv" in layout_str



class TestQuantileModeSwitch:
    def test_mode_controls_in_snapshot(self):
        from snapshot import _SNAPSHOT_CONTROLS
        ids = {c[0] for c in _SNAPSHOT_CONTROLS}
        for prefix in ["bub", "dca", "ret", "sc"]:
            assert f"{prefix}-qs-mode" in ids, f"{prefix}-qs-mode missing"
            assert f"{prefix}-qs-adv" in ids, f"{prefix}-qs-adv missing"

    def test_mode_controls_in_checklist_options(self):
        from snapshot import _CHECKLIST_OPTIONS
        for prefix in ["bub", "dca", "ret", "sc"]:
            assert f"{prefix}-qs-mode" in _CHECKLIST_OPTIONS
            assert f"{prefix}-qs-adv" in _CHECKLIST_OPTIONS

    def test_mode_controls_in_tab_controls(self):
        from callbacks.routing import _TAB_CONTROLS
        assert "bub-qs-mode" in _TAB_CONTROLS["bubble"]
        assert "dca-qs-mode" in _TAB_CONTROLS["dca"]
        assert "ret-qs-mode" in _TAB_CONTROLS["retire"]
        assert "sc-qs-mode" in _TAB_CONTROLS["supercharge"]

    def test_qs_mode_in_tab_defaults(self):
        from tab_defaults import BUBBLE, DCA, RETIRE, SUPERCHARGE
        assert "qs_mode" in BUBBLE
        assert "qs_mode" in DCA
        assert "qs_mode" in RETIRE
        assert "qs_mode" in SUPERCHARGE



class TestDefaultModeOpacity:
    def test_fallback_q50_has_opacity_in_default_mode(self):
        """Q50% fallback in default mode should have 25% opacity."""
        from figures.bubble import build_bubble_figure
        import _app_ctx
        M = _app_ctx.M
        p = dict(selected_qs=[], shade=False, xscale="log", yscale="log",
                 xmin=2012, xmax=2030, ymin=0, ymax=7, n_future=3,
                 show_comp=False, show_ols=False, show_data=False,
                 show_today=False, pt_size=2, pt_alpha=0.3,
                 stack=0, show_stack=False, lots=[], use_lots=False,
                 show_legend=False, active_models=["bub"],
                 qs_mode=[])
        fig = build_bubble_figure(M, p)
        q50_traces = [t for t in fig.data if hasattr(t, 'name') and t.name and 'Q50%' in str(t.name)]
        assert len(q50_traces) > 0
        assert q50_traces[0].opacity == 0.25

    def test_fallback_q50_full_opacity_in_advanced_mode(self):
        """Q50% fallback in advanced mode should have full opacity."""
        from figures.bubble import build_bubble_figure
        import _app_ctx
        M = _app_ctx.M
        p = dict(selected_qs=[], shade=False, xscale="log", yscale="log",
                 xmin=2012, xmax=2030, ymin=0, ymax=7, n_future=3,
                 show_comp=False, show_ols=False, show_data=False,
                 show_today=False, pt_size=2, pt_alpha=0.3,
                 stack=0, show_stack=False, lots=[], use_lots=False,
                 show_legend=False, active_models=["bub"],
                 qs_mode=["advanced"])
        fig = build_bubble_figure(M, p)
        q50_traces = [t for t in fig.data if hasattr(t, 'name') and t.name and 'Q50%' in str(t.name)]
        assert len(q50_traces) > 0
        assert q50_traces[0].opacity == 1.0  # Q50% = full opacity



class TestSymmetricBandShading:
    def test_symmetric_bands_5_quantiles(self):
        """5 quantiles → 2 bands (outer + inner)."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.01, 0.15, 0.50, 0.85, 0.99]
        prices = {q: np.linspace(100 * (1 + q), 200 * (1 + q), 10) for q in qs}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#000000")
        assert len(traces) == 4  # 2 bands × 2 traces each

    def test_symmetric_bands_3_quantiles(self):
        """3 quantiles → 1 band (outer only)."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.15, 0.50, 0.85]
        prices = {q: np.linspace(100, 200, 10) for q in qs}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#FF0000")
        assert len(traces) == 2

    def test_symmetric_bands_2_quantiles(self):
        """2 quantiles → 1 band."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.15, 0.85]
        prices = {q: np.linspace(100, 200, 10) for q in qs}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#000000")
        assert len(traces) == 2

    def test_symmetric_bands_1_quantile(self):
        """1 quantile → 0 bands."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.50]
        prices = {0.50: np.linspace(100, 200, 10)}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#000000")
        assert len(traces) == 0

    def test_symmetric_bands_outer_lighter_than_inner(self):
        """Outer band should have lower opacity than inner."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.01, 0.15, 0.50, 0.85, 0.99]
        prices = {q: np.linspace(100, 200, 10) for q in qs}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#000000")
        outer_fill = traces[1].fillcolor
        inner_fill = traces[3].fillcolor
        outer_alpha = float(outer_fill.split(",")[-1].rstrip(")"))
        inner_alpha = float(inner_fill.split(",")[-1].rstrip(")"))
        assert outer_alpha < inner_alpha



class TestSymmetricQuantileColors:
    def test_mirror_quantiles_same_color(self):
        """Q15% and Q85% should get the same color."""
        from figures.common import _symmetric_thermal_color
        c15 = _symmetric_thermal_color(0.15)
        c85 = _symmetric_thermal_color(0.85)
        assert c15 == c85

    def test_q50_gets_median_color(self):
        """Q50% should get the median (gray) color."""
        from figures.common import _symmetric_thermal_color
        c50 = _symmetric_thermal_color(0.50)
        assert c50 == "#bdbdbd"



class TestOverlayModelShading:
    def test_overlay_model_bands_use_model_color(self):
        """Overlay model bands should use that model's trace color."""
        from figures.bubble import build_bubble_figure
        import _app_ctx
        M = _app_ctx.M
        p = dict(selected_qs=[0.15, 0.50, 0.85], shade=True,
                 xscale="log", yscale="log",
                 xmin=2012, xmax=2030, ymin=0, ymax=7, n_future=3,
                 show_comp=False, show_ols=False, show_data=False,
                 show_today=False, pt_size=2, pt_alpha=0.3,
                 stack=0, show_stack=False, lots=[], use_lots=False,
                 show_legend=False, active_models=["bub", "pl"],
                 qs_mode=[])
        fig = build_bubble_figure(M, p)
        fill_traces = [t for t in fig.data if t.fill == "tonexty"]
        assert len(fill_traces) >= 2  # at least 1 band per model



class TestColorCodedModelLabels:
    def test_model_labels_have_color_swatch(self):
        """Display Models labels should have colored boxes."""
        from layout.bubble import _bubble_controls
        layout_str = repr(_bubble_controls())
        assert "backgroundColor" in layout_str



class TestResolveLpplMaster:
    """Unit test for the LPPL master -> flavor translation helper."""

    def test_no_master_passes_through(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["bub", "pl"], [3], [], [])
        assert result == ["bub", "pl"]

    def test_master_1_unweighted(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["bub", "lppl"], [1], [], [])
        assert "lppl" in result and "bub" in result
        assert "lp2" not in result

    def test_master_3_weighted(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["lppl"], [3], ["weighted"], [])
        assert result == ["lp3_w"]

    def test_master_3_disabled_by_no_13(self):
        from callbacks.charts import _resolve_lppl_master
        # no_13 disables LP3
        result = _resolve_lppl_master(["lppl"], [3], [], ["no13"])
        assert result == []

    def test_master_4_no_13(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["lppl"], [4], [], ["no13"])
        assert result == ["lp4_n13"]

    def test_master_4_weighted_no_13(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["lppl"], [4], ["weighted"], ["no13"])
        assert result == ["lp4_w_n13"]

    def test_master_all_freqs_unweighted(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["lppl"], [1, 2, 3, 4], [], [])
        assert set(result) == {"lppl", "lp2", "lp3", "lp4"}

    def test_empty_n_freqs_strips_master(self):
        from callbacks.charts import _resolve_lppl_master
        # Master checked but no flavor selected -> master stripped with no replacement
        result = _resolve_lppl_master(["bub", "lppl"], [], [], [])
        assert result == ["bub"]



class TestModelShowChecklistStandardized:
    """Unit tests for _model_show_checklist standardized=True mode."""

    def test_has_lppl_master(self):
        from layout.common import _model_show_checklist
        elems = _model_show_checklist("dca", standardized=True)
        rendered = str(elems).replace("'", '"')
        assert '"value": "lppl"' in rendered

    def test_omits_lppl_variants(self):
        from layout.common import _model_show_checklist
        elems = _model_show_checklist("dca", standardized=True)
        rendered = str(elems).replace("'", '"')
        assert '"value": "lp2"' not in rendered
        assert '"value": "lp3"' not in rendered
        assert '"value": "lp4"' not in rendered
        assert '"value": "lppl_w"' not in rendered

    def test_omits_exp_and_s2f(self):
        from layout.common import _model_show_checklist
        elems = _model_show_checklist("dca", standardized=True)
        rendered = str(elems).replace("'", '"')
        assert '"value": "exp"' not in rendered
        assert '"value": "s2f"' not in rendered

    def test_non_standardized_unchanged(self):
        from layout.common import _model_show_checklist
        elems = _model_show_checklist("dca", standardized=False)
        rendered = str(elems).replace("'", '"')
        assert '"value": "lppl"' in rendered



class TestLpplConfigPanel:
    """Unit test for _lppl_config_panel compact helper."""

    def test_has_activate_and_summary_and_button(self):
        from layout.common import _lppl_config_panel
        card = _lppl_config_panel("dca")
        rendered = str(card)
        assert "dca-lppl-activate" in rendered
        assert "dca-lppl-summary" in rendered
        assert "dca-lppl-configure-btn" in rendered

    def test_no_inline_config_controls(self):
        """The un-prefixed config IDs live in the global modal, not here."""
        from layout.common import _lppl_config_panel
        card = _lppl_config_panel("ret")
        rendered = str(card).replace("'", '"')
        assert '"lppl-n-freqs"' not in rendered
        assert '"lppl-weighted"' not in rendered
        assert '"lppl-no-13"' not in rendered



class TestGlobalLpplModal:
    """Unit test for _global_lppl_modal root-level modal."""

    def test_has_all_config_controls(self):
        from layout.common import _global_lppl_modal
        modal = _global_lppl_modal()
        rendered = str(modal)
        assert "lppl-config-modal" in rendered
        assert "lppl-n-freqs" in rendered
        assert "lppl-weighted" in rendered
        assert "lppl-no-13" in rendered
        assert "lppl-modal-close-btn" in rendered



class TestResolveHmLpplMaster:
    """Unit test for heatmap LPPL master translation."""

    def test_non_lppl_passes_through(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("bub", [3], [], []) == "bub"
        assert _resolve_hm_lppl_master("pl", [3], [], []) == "pl"
        assert _resolve_hm_lppl_master("linppl", [3], [], []) == "linppl"

    def test_lppl_default_n3_unweighted(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [3], [], []) == "lp3"

    def test_lppl_n3_weighted(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [3], ["weighted"], []) == "lp3_w"

    def test_lppl_n4_no_13(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [4], [], ["no13"]) == "lp4_n13"

    def test_lppl_n4_weighted_no_13(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [4], ["weighted"], ["no13"]) == "lp4_w_n13"

    def test_lppl_picks_first_when_multi_selected(self):
        from callbacks.charts import _resolve_hm_lppl_master
        # Heatmap is single-select: takes first entry, ignores rest
        assert _resolve_hm_lppl_master("lppl", [2, 4], [], []) == "lp2"

    def test_lppl_empty_n_freqs_defaults_to_3(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [], [], []) == "lp3"

    def test_lppl_n3_with_no_13_falls_through_to_lppl(self):
        from callbacks.charts import _resolve_hm_lppl_master
        # n=3 and no_13 both set → LP3 disabled → fallback to "lppl"
        assert _resolve_hm_lppl_master("lppl", [3], [], ["no13"]) == "lppl"


# ═══════════════════════════════════════════════════════════════════════════════
# New models: Logistic Growth + Broken Power Law
# ═══════════════════════════════════════════════════════════════════════════════


class TestLogisticModel:
    def setup_method(self):
        self.m = LogisticModel(M.price_years, M.price_prices, M.QR_QUANTILES)

    def test_short_name(self):
        assert self.m.short_name == "gomp"

    def test_quantized(self):
        assert self.m.quantized is True

    def test_fits_has_quantile_keys(self):
        assert set(self.m.fits.keys()) == set(M.QR_QUANTILES)

    def test_price_at_positive(self):
        q = self.m.quantiles[len(self.m.quantiles) // 2]
        assert float(self.m.price_at(q, 10.0)) > 0

    def test_quantile_ordering(self):
        t = 10.0
        prices = [float(self.m.price_at(q, t)) for q in self.m.quantiles]
        assert prices == sorted(prices)

    def test_colors_populated(self):
        assert len(self.m.colors) == len(self.m.quantiles)

    def test_find_percentile(self):
        pct = self.m.find_percentile(10.0, 50000)
        assert 0 < pct < 1

    def test_registered(self):
        assert "gomp" in _app_ctx.PRICE_MODELS


class TestBrokenPowerLawModel:
    def setup_method(self):
        self.m = BrokenPowerLawModel(M.price_years, M.price_prices, M.QR_QUANTILES)

    def test_short_name(self):
        assert self.m.short_name == "bpl"

    def test_quantized(self):
        assert self.m.quantized is True

    def test_fits_has_quantile_keys(self):
        assert set(self.m.fits.keys()) == set(M.QR_QUANTILES)

    def test_price_at_positive(self):
        q = self.m.quantiles[len(self.m.quantiles) // 2]
        assert float(self.m.price_at(q, 10.0)) > 0

    def test_quantile_ordering(self):
        t = 10.0
        prices = [float(self.m.price_at(q, t)) for q in self.m.quantiles]
        assert prices == sorted(prices)

    def test_continuity_at_breakpoint(self):
        """Price should be continuous at t_break."""
        q = self.m.quantiles[len(self.m.quantiles) // 2]
        t_break = self.m._t_break
        p_left = float(self.m.price_at(q, t_break - 0.001))
        p_right = float(self.m.price_at(q, t_break + 0.001))
        import numpy as np
        np.testing.assert_allclose(p_left, p_right, rtol=0.01)

    def test_colors_populated(self):
        assert len(self.m.colors) == len(self.m.quantiles)

    def test_find_percentile(self):
        pct = self.m.find_percentile(10.0, 50000)
        assert 0 < pct < 1

    def test_registered(self):
        assert "bpl" in _app_ctx.PRICE_MODELS
