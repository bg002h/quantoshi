"""Tab-1 sub-view deep links: /1.2 (CAGR), /1.3 (Residuals), /1.4 (Percentile),
/1.5[.T[.W]] (Occupancy with tail / window presets)."""
import re

import dash
import pytest

from conftest import _app_ctx  # noqa: F401 — app must be initialised
from callbacks.routing import deep_link_bub_view
from layout.bubble import CAGR_DEFAULT_XRANGE

NU = dash.no_update
H = {"display": "none"}
INLINE = {"display": "inline"}

# Output order of deep_link_bub_view (mirrors toggle_bub_view + CAGR/occ extras)
(MODE, PRICE_W, CAGR_W, RESID_W, PCTILE_W, OCC_W,
 PRICE_O, CAGR_O, RESID_O, PCTILE_O, OCC_O,
 SCALE, PANEL, CAGR_CTL, OCC_CTL,
 XRANGE, FWD_YRS, HOVER_TODAY, OCC_TAIL, OCC_WINDOW) = range(20)


def _n_outputs():
    # Module-level @callback registrations live in dash's GLOBAL_CALLBACK_MAP
    # until the server starts; app.callback_map only holds app.* registrations.
    from dash._callback import GLOBAL_CALLBACK_MAP
    cm = {**GLOBAL_CALLBACK_MAP, **_app_ctx.app.callback_map}
    keys = [k for k in cm if "bub-occ-window.value" in k and "bub-view-mode.data" in k
            and "bub-cagr-hover-today.data" in k]
    assert len(keys) == 1, keys
    return keys[0].count("@")   # one "@<hash>" per output in a multi-output key


class TestDeepLinkArity:
    def test_twenty_outputs_and_every_branch_returns_twenty(self):
        assert _n_outputs() == 20
        for path in ("/1.2", "/1.2.5.1", "/1.3", "/1.4", "/1.5", "/1.5.3.1", "/2", None, ""):
            out = deep_link_bub_view(path)
            assert len(out) == 20, path


class TestExistingLinksUnchanged:
    def test_cagr_with_horizon_and_hover(self):
        out = deep_link_bub_view("/1.2.5.1")
        assert out[MODE] == "cagr"
        assert out[CAGR_W] == {} and out[PRICE_W] == H and out[PCTILE_W] == H and out[OCC_W] == H
        assert out[CAGR_O] is False and out[OCC_O] is True and out[PCTILE_O] is True
        assert out[CAGR_CTL] == INLINE and out[OCC_CTL] == H
        assert out[XRANGE] == CAGR_DEFAULT_XRANGE
        assert out[FWD_YRS] == 20 and out[HOVER_TODAY] is True
        assert out[OCC_TAIL] is NU and out[OCC_WINDOW] is NU

    def test_residuals(self):
        out = deep_link_bub_view("/1.3")
        assert out[MODE] == "resid"
        assert out[RESID_W] == {} and out[OCC_W] == H
        assert out[RESID_O] is False and out[OCC_O] is True
        assert out[OCC_CTL] == H and out[CAGR_CTL] == H

    def test_other_paths_are_no_update(self):
        for path in ("/2", "/1", "/9.3", None):
            assert all(v is NU for v in deep_link_bub_view(path)), path


class TestPercentileLink:
    def test_1_4_opens_percentile(self):
        out = deep_link_bub_view("/1.4")
        assert out[MODE] == "percentile"
        assert out[PCTILE_W] == {}
        assert out[PRICE_W] == H and out[CAGR_W] == H and out[RESID_W] == H and out[OCC_W] == H
        assert out[PCTILE_O] is False
        assert out[PRICE_O] is True and out[CAGR_O] is True and out[RESID_O] is True and out[OCC_O] is True
        assert out[SCALE] == {} and out[PANEL] == {}
        assert out[CAGR_CTL] == H and out[OCC_CTL] == H
        assert out[XRANGE] is NU and out[OCC_TAIL] is NU and out[OCC_WINDOW] is NU


class TestOccupancyLink:
    def test_1_5_opens_occupancy_with_default_controls(self):
        out = deep_link_bub_view("/1.5")
        assert out[MODE] == "occupancy"
        assert out[OCC_W] == {} and out[PCTILE_W] == H and out[PRICE_W] == H
        assert out[OCC_O] is False and out[PCTILE_O] is True and out[PRICE_O] is True
        assert out[OCC_CTL] == INLINE and out[CAGR_CTL] == H
        assert out[OCC_TAIL] is NU and out[OCC_WINDOW] is NU

    def test_1_5_T_W_sets_tail_and_window(self):
        # T indexes [5, 10, 25], W indexes [1, 2, 4] — 1-based like /1.2.N
        out = deep_link_bub_view("/1.5.3.1")
        assert out[MODE] == "occupancy"
        assert out[OCC_TAIL] == 25 and out[OCC_WINDOW] == 1
        out = deep_link_bub_view("/1.5.1")
        assert out[OCC_TAIL] == 5 and out[OCC_WINDOW] is NU
        out = deep_link_bub_view("/1.5.2.3")
        assert out[OCC_TAIL] == 10 and out[OCC_WINDOW] == 4

    def test_dash_form_is_equivalent(self):
        assert deep_link_bub_view("/1-5-3-1")[OCC_TAIL] == 25
        assert deep_link_bub_view("/1-5-3-1")[OCC_WINDOW] == 1
        assert deep_link_bub_view("/1-4")[MODE] == "percentile"

    def test_out_of_range_or_garbage_suffix_is_ignored_not_crashed(self):
        for path in ("/1.5.9", "/1.5.0.2", "/1.5.x.y", "/1.5.2.7", "/1.5..1"):
            out = deep_link_bub_view(path)
            assert out[MODE] == "occupancy", path
        assert deep_link_bub_view("/1.5.9")[OCC_TAIL] is NU
        assert deep_link_bub_view("/1.5.2.7")[OCC_WINDOW] is NU
        assert deep_link_bub_view("/1.5.2.7")[OCC_TAIL] == 10


class TestClientsideTabMap:
    def test_tab_map_routes_every_tab1_subview_to_bubble(self):
        scripts = [s for s in _app_ctx.app._inline_scripts
                   if '"/9":"model_info"' in s and "_routingLastPath" in s]
        assert len(scripts) == 1
        js = scripts[0]
        # A single /1.<n> pattern replaces the old per-suffix indexOf checks,
        # so /1.4 and /1.5 (and any future /1.N) land on the bubble tab.
        assert 'indexOf("/1.2")' not in js and 'indexOf("/1.3")' not in js
        assert re.search(r"\^\\/1\\\.\\d\+", js), "expected a /^\\/1\\.\\d+/ regex in the runtime JS"
