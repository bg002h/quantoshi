"""Tests for tab_defaults.py — single source of truth for all tab defaults."""
import pytest


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
