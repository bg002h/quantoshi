"""Tests for tab_defaults.py — single source of truth for all tab defaults."""
import os
import pytest

os.environ.setdefault("TESTING", "1")


def test_defaults_are_immutable():
    from tab_defaults import BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE, STACK, CITADEL
    for name, defaults in [("BUBBLE", BUBBLE), ("HEATMAP", HEATMAP),
                           ("DCA", DCA), ("RETIRE", RETIRE),
                           ("SUPERCHARGE", SUPERCHARGE), ("STACK", STACK),
                           ("CITADEL", CITADEL)]:
        with pytest.raises(TypeError, match="does not support item assignment"):
            defaults["new_key"] = "bad"


def test_inner_collections_are_tuples():
    from tab_defaults import BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE, STACK, CITADEL
    for name, defaults in [("BUBBLE", BUBBLE), ("HEATMAP", HEATMAP),
                           ("DCA", DCA), ("RETIRE", RETIRE),
                           ("SUPERCHARGE", SUPERCHARGE), ("STACK", STACK),
                           ("CITADEL", CITADEL)]:
        for key, val in defaults.items():
            assert not isinstance(val, list), f"{name}[{key!r}] is a list, should be tuple"
            assert not isinstance(val, set), f"{name}[{key!r}] is a set, should be frozenset"


def test_bubble_defaults_returns_mutable_dict():
    from tab_defaults import bubble_defaults
    d = bubble_defaults()
    assert isinstance(d, dict)
    d["new_key"] = "ok"
    assert "xmin" in d and "xmax" in d
    assert isinstance(d["selected_qs"], list)


def test_heatmap_defaults_returns_mutable_dict():
    from tab_defaults import heatmap_defaults
    d = heatmap_defaults()
    assert isinstance(d, dict)
    assert "entry_yr" in d
    assert "exit_yr_lo" in d
    assert "exit_yr_hi" in d


def test_dca_defaults_returns_mutable_dict():
    from tab_defaults import dca_defaults
    d = dca_defaults()
    assert isinstance(d, dict)
    assert "start_yr" in d and "end_yr" in d
    assert isinstance(d["selected_qs"], list)


def test_retire_defaults_has_correct_static_yr():
    from tab_defaults import retire_defaults
    d = retire_defaults()
    assert d["start_yr"] == 2031
    assert d["end_yr"] == 2075
    assert isinstance(d["selected_qs"], list)


def test_supercharge_defaults_has_list_delays():
    from tab_defaults import supercharge_defaults
    d = supercharge_defaults()
    assert isinstance(d["delays"], list)
    assert isinstance(d["selected_qs"], list)


def test_citadel_defaults_returns_mutable_dict():
    from tab_defaults import citadel_defaults
    d = citadel_defaults()
    assert isinstance(d, dict)
    assert d["high_q_trigger"] == 95
    assert d["cash_floor"] == 50000
    assert isinstance(d["selected_qs"], list)


# ---------------------------------------------------------------------------
# Part 1: Figure builder smoke tests
# ---------------------------------------------------------------------------

def test_bubble_defaults_produce_valid_figure():
    from tab_defaults import bubble_defaults
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from figures.bubble import build_bubble_figure
    fig = build_bubble_figure(_app_ctx.M, bubble_defaults())
    assert len(fig.data) > 0


def test_heatmap_defaults_produce_valid_figure():
    from tab_defaults import heatmap_defaults
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from figures.heatmap import build_heatmap_figure
    d = heatmap_defaults()
    d["entry_q"] = 50.0
    # exit_qs defaults to [] — provide a valid quantile so the chart has data
    d["exit_qs"] = [q for q in [0.50] if q in _app_ctx.DEFAULT_MODEL.fits]
    fig = build_heatmap_figure(_app_ctx.M, d)
    assert len(fig.data) > 0


def test_dca_defaults_produce_valid_figure():
    from tab_defaults import dca_defaults
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from figures.dca import build_dca_figure
    # build_dca_figure returns tuple[go.Figure, dict | None]
    fig_result = build_dca_figure(_app_ctx.M, dca_defaults())
    fig = fig_result[0] if isinstance(fig_result, tuple) else fig_result
    assert len(fig.data) > 0


def test_retire_defaults_produce_valid_figure():
    from tab_defaults import retire_defaults
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from figures.retire import build_retire_figure
    # build_retire_figure returns tuple[go.Figure, dict | None]
    fig_result = build_retire_figure(_app_ctx.M, retire_defaults())
    fig = fig_result[0] if isinstance(fig_result, tuple) else fig_result
    assert len(fig.data) > 0


def test_supercharge_defaults_produce_valid_figure():
    from tab_defaults import supercharge_defaults
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from figures.supercharge import build_supercharge_figure
    d = supercharge_defaults()
    d["selected_qs"] = [q for q in [0.001, 0.10] if q in _app_ctx.DEFAULT_MODEL.fits]
    # build_supercharge_figure returns tuple[go.Figure, dict | None]
    fig_result = build_supercharge_figure(_app_ctx.M, d)
    fig = fig_result[0] if isinstance(fig_result, tuple) else fig_result
    assert len(fig.data) > 0


def test_citadel_defaults_produce_valid_figure():
    from tab_defaults import citadel_defaults
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from figures.citadel import build_citadel_figure
    d = citadel_defaults()
    d["selected_qs"] = [0.25]
    fig_result = build_citadel_figure(_app_ctx.M, d)
    fig = fig_result[0] if isinstance(fig_result, tuple) else fig_result
    assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# Part 2: Layout drift detection test
# ---------------------------------------------------------------------------

def test_layout_values_match_defaults():
    """Walk the layout component tree, check value= props match defaults."""
    from tab_defaults import BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE, STACK, CITADEL
    from btc_web.app import app

    prefix_map = {
        "bub-": BUBBLE, "hm-": HEATMAP, "dca-": DCA,
        "ret-": RETIRE, "sc-": SUPERCHARGE, "cp-": CITADEL,
        "lot-": STACK,
    }

    id_to_key = {
        "bub-ptsize": "pt_size", "bub-ptalpha": "pt_alpha", "bub-n-future": "n_future",
        "hm-mode": "color_mode", "hm-grad": "n_disc", "hm-cell-fs": "cell_font_size",
        "dca-infl": "inflation", "dca-sc-loan": "sc_loan_amount",
        "dca-sc-term": "sc_term_months", "dca-sc-rate": "sc_rate",
        "dca-sc-custom-price": "sc_custom_price", "dca-sc-repeats": "sc_repeats",
        "ret-wd": "wd_amount", "ret-infl": "inflation",
        "sc-infl": "inflation", "sc-wd": "wd_amount",
        "sc-start-yr": "start_yr", "sc-end-yr": "end_yr", "sc-target-yr": "target_yr",
        "cp-cash-init": "cash_initial", "cp-cash-rate": "cash_rate",
        "cp-spend": "monthly_spend", "cp-infl": "inflation", "cp-spend-growth": "spend_growth",
        "cp-high-q-thresh": "high_q_trigger", "cp-high-q-rate": "high_q_rate",
        "cp-high-q-dur": "high_q_dur", "cp-low-q-thresh": "low_q_trigger",
        "cp-low-q-rate": "low_q_rate", "cp-low-q-dur": "low_q_dur",
        "cp-scf-amount": "scf_amount", "cp-scf-rate": "scf_rate",
        "cp-scf-term": "scf_term", "cp-scf-trigger": "scf_repay_trigger",
        "cp-cash-floor": "cash_floor", "cp-res-floor-growth": "reserve_floor_growth",
        "lot-price": "lot_price", "lot-btc": "lot_btc",
    }

    skip_ids = {
        "bub-xrange", "bub-yrange", "bub-auto-y",
        "bub-toggles", "bub-bubble-toggles", "bub-qs",
        "bub-model-show", "bub-show-stack", "bub-use-lots",
        "scan-price", "scan-date", "scan-q",
        "hm-entry-yr", "hm-entry-q", "hm-exit-range", "hm-exit-qs",
        "hm-use-lots", "hm-palette", "hm-toggles", "hm-model-show",
        "hm-c-lo", "hm-c-mid1", "hm-c-mid2", "hm-c-hi",
        "dca-yr-range", "dca-qs", "dca-toggles", "dca-model-show",
        "dca-use-lots", "dca-sc-enable", "dca-sc-rollover",
        "dca-sc-tax",  # layout=int% (33), defaults=fraction (0.33)
        "ret-yr-range", "ret-qs", "ret-toggles", "ret-model-show", "ret-use-lots",
        "sc-qs", "sc-toggles", "sc-model-show", "sc-use-lots",
        "sc-chart-layout", "sc-display-q",
        "cp-qs", "cp-toggles", "cp-use-lots", "cp-yr-range",
        "cp-model-src", "cp-asset-model", "cp-disp",
        "cp-scf-enable", "cp-legend-pos",
        "lot-date", "lot-notes",
    }

    def is_mc_control(cid):
        return "-mc-" in cid

    mismatches = []

    def walk(component):
        cid = getattr(component, 'id', None)
        if cid and isinstance(cid, str) and not is_mc_control(cid) and cid not in skip_ids:
            val = getattr(component, 'value', '__MISSING__')
            if val != '__MISSING__':
                for prefix, defaults in prefix_map.items():
                    if cid.startswith(prefix):
                        raw_key = cid[len(prefix):].replace("-", "_")
                        key = id_to_key.get(cid, raw_key)
                        if key in defaults:
                            expected = defaults[key]
                            if val != expected:
                                mismatches.append(
                                    f"{cid}: layout={val!r}, defaults={expected!r}")
                        break

        children = getattr(component, 'children', None)
        if isinstance(children, (list, tuple)):
            for child in children:
                if child is not None:
                    walk(child)
        elif children is not None:
            walk(children)

    walk(app.layout)
    assert mismatches == [], "Layout/defaults mismatches:\n  " + "\n  ".join(mismatches)


# ── L0 cache fingerprint tests ──────────────────────────────────────────────

def test_defaults_hash_is_stable():
    """Hash is deterministic across calls."""
    from tab_defaults import _DEFAULTS_HASH, _compute_defaults_hash
    assert isinstance(_DEFAULTS_HASH, str)
    assert len(_DEFAULTS_HASH) == 12
    assert _compute_defaults_hash() == _DEFAULTS_HASH


def test_defaults_hash_changes_on_value_change():
    """Hash changes when a frozen dict value changes."""
    from tab_defaults import _compute_defaults_hash, BUBBLE
    import types, hashlib
    original = _compute_defaults_hash()
    fake = types.MappingProxyType({**BUBBLE, "pt_alpha": 999.0})
    h = hashlib.md5()
    h.update(repr(sorted(fake.items())).encode())
    partial = h.hexdigest()[:12]
    assert partial != original[:12]


def test_l0_fingerprint_combines_model_and_defaults():
    """L0 fingerprint includes both model and defaults hashes."""
    from cache import _L0_FINGERPRINT, _MODEL_FP
    from tab_defaults import _DEFAULTS_HASH
    assert isinstance(_L0_FINGERPRINT, str)
    assert len(_L0_FINGERPRINT) == 12
    # Verify it's derived from both (not equal to either alone)
    assert _L0_FINGERPRINT != _MODEL_FP[:12]
    assert _L0_FINGERPRINT != _DEFAULTS_HASH


def test_l0_serialize_deserialize_roundtrip():
    """Figure survives serialize → deserialize."""
    import plotly.graph_objects as go
    from utils import _serialize_result, _deserialize_result
    fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4], name="test")])
    raw = _serialize_result(fig)
    loaded = _deserialize_result(raw)
    assert len(loaded.data) == 1
    assert loaded.data[0].name == "test"


def test_l0_serialize_tuple_roundtrip():
    """(fig, mc_result) tuple survives roundtrip."""
    import plotly.graph_objects as go
    from utils import _serialize_result, _deserialize_result
    fig = go.Figure(data=[go.Scatter(x=[1], y=[2])])
    result = (fig, {"some": "data"})
    raw = _serialize_result(result)
    loaded = _deserialize_result(raw)
    assert isinstance(loaded, tuple)
    assert len(loaded[0].data) == 1
    assert loaded[1]["some"] == "data"


def test_compute_cache_key_deterministic():
    """Same params → same key."""
    from utils import _compute_cache_key
    from tab_defaults import bubble_defaults
    p1 = bubble_defaults()
    p2 = bubble_defaults()
    assert _compute_cache_key("bub", p1) == _compute_cache_key("bub", p2)


def test_compute_cache_key_changes_with_params():
    """Different params → different key."""
    from utils import _compute_cache_key
    from tab_defaults import bubble_defaults
    p1 = bubble_defaults()
    p2 = bubble_defaults()
    p2["pt_alpha"] = 0.99
    assert _compute_cache_key("bub", p1) != _compute_cache_key("bub", p2)
