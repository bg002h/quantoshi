"""bub_views.VIEW_MODES is the single source of truth for Tab-1 view state.

Three places used to spell the same per-view UI state out by hand — the pill
callback, the clientside sync JS, and the deep-link callback — and the only
thing keeping them in agreement was a reviewer reading all three side by side.
They now all read ``bub_views``; these tests are what makes that structural:

* every producer agrees with ``mode_styles`` for every mode in the table,
* the generated JS decodes back to exactly the table,
* every id in the table exists in the served layout,
* the table's order is the callbacks' Output order (positional outputs — a
  reordered table would silently mis-assign styles, not raise).
"""
import json
import re
from unittest.mock import patch

import dash
import pytest

from conftest import _app_ctx  # noqa: F401 — app must be initialised

import bub_views
from bub_views import (VIEW_MODES, WRAP_IDS, PILL_IDS, CTL_IDS, PANEL_IDS,
                       STYLE_OUTPUT_IDS, HISTORICAL_MODES, DEFAULT_MODE,
                       SYNC_JS_MARKER, mode_styles, mode_for_path)
from callbacks.charts import toggle_bub_view
from callbacks.routing import deep_link_bub_view

NU = dash.no_update
_ALL = list(VIEW_MODES)


class _Ctx:
    """Stand-in for dash.ctx — same shape test_occupancy.py patches in."""

    def __init__(self, triggered_id):
        self.triggered_id = triggered_id


def _callback_map():
    # Module-level @callback registrations live in dash's GLOBAL_CALLBACK_MAP
    # until the server starts; app.callback_map only holds app.* ones.
    from dash._callback import GLOBAL_CALLBACK_MAP
    return {**GLOBAL_CALLBACK_MAP, **_app_ctx.app.callback_map}


def _output_ids(key: str) -> list[str]:
    """Component ids of a multi-output callback key, in Output order.

    Keys look like ``..a.prop@hash...b.prop@hash..`` — split on the ``...``
    separator and take the id in front of the final ``.prop``.
    """
    return [seg.rsplit(".", 1)[0]
            for seg in key.strip(".").split("...") if seg]


def _find_callback(must_contain: tuple[str, ...],
                   must_not_contain: tuple[str, ...] = ()) -> str:
    """The one callback key holding all of `must_contain` and none of the rest.

    All three view-mode callbacks drive the same 14 components, so they can
    only be told apart by their extra outputs.
    """
    keys = [k for k in _callback_map()
            if all(f in k for f in must_contain)
            and not any(f in k for f in must_not_contain)]
    assert len(keys) == 1, (must_contain, must_not_contain, keys)
    return keys[0]


# ═══════════════════════════════════════════════════════════════════════════
# The table itself
# ═══════════════════════════════════════════════════════════════════════════

class TestTableShape:
    def test_ids_are_unique_and_derived_lists_follow_table_order(self):
        assert WRAP_IDS == tuple(v.wrap for v in VIEW_MODES.values())
        assert PILL_IDS == tuple(v.pill for v in VIEW_MODES.values())
        assert CTL_IDS == tuple(v.ctl for v in VIEW_MODES.values()
                                if v.ctl is not None)
        for ids in (WRAP_IDS, PILL_IDS, CTL_IDS):
            assert len(set(ids)) == len(ids), ids
        assert STYLE_OUTPUT_IDS == WRAP_IDS + PILL_IDS + PANEL_IDS + CTL_IDS
        assert len(STYLE_OUTPUT_IDS) == 14

    def test_default_mode_is_in_the_table(self):
        assert DEFAULT_MODE in VIEW_MODES

    def test_deep_links_are_unique(self):
        links = [v.deep_link for v in VIEW_MODES.values() if v.deep_link]
        assert len(set(links)) == len(links), links

    @pytest.mark.parametrize("mode", _ALL)
    def test_mode_styles_shows_exactly_one_wrap_and_fills_one_pill(self, mode):
        st = mode_styles(mode)
        assert len(st) == 14
        spec = VIEW_MODES[mode]
        wraps, pills = st[:5], st[5:10]
        assert [w == {} for w in wraps] == [w == spec.wrap for w in WRAP_IDS]
        assert pills.count(False) == 1
        assert pills[PILL_IDS.index(spec.pill)] is False
        assert st[10] == ({} if spec.scale_controls else {"display": "none"})
        assert st[11] == ({} if spec.bubble_panel else {"display": "none"})
        assert [c == {"display": "inline"} for c in st[12:14]] == \
               [c == spec.ctl for c in CTL_IDS]

    def test_unknown_mode_falls_back_to_default(self):
        assert mode_styles("no-such-view") == mode_styles(DEFAULT_MODE)

    def test_styles_are_fresh_objects_not_shared_mutables(self):
        # These dicts are handed to Dash; a shared instance would let one
        # caller's mutation leak into every later view switch.
        a, b = mode_styles("price"), mode_styles("price")
        assert a == b
        assert all(x is not y for x, y in zip(a, b) if isinstance(x, dict))

    def test_historical_modes_matches_the_table(self):
        assert HISTORICAL_MODES == frozenset(
            m for m, v in VIEW_MODES.items() if v.historical)


class TestModeForPath:
    @pytest.mark.parametrize("mode", _ALL)
    def test_deep_link_resolves_to_its_own_mode(self, mode):
        link = VIEW_MODES[mode].deep_link
        if link is None:
            pytest.skip(f"{mode} has no deep link")
        assert mode_for_path(link) == mode
        assert mode_for_path(link + ".3.1") == mode

    def test_non_view_paths_are_none(self):
        for p in (None, "", "/1", "/2", "/9.3", "/10", "/faq.2"):
            assert mode_for_path(p) is None, p

    def test_longest_prefix_wins(self):
        # If a future row deep-links a suffix of an existing one, the more
        # specific row must win rather than whichever is listed first.
        from bub_views import ViewMode
        table = dict(VIEW_MODES)
        table["zoomed_cagr"] = ViewMode(
            pill="x", wrap="y", historical=False, scale_controls=True,
            bubble_panel=True, ctl=None, deep_link="/1.2.9")
        with patch.object(bub_views, "VIEW_MODES", table):
            assert mode_for_path("/1.2.9") == "zoomed_cagr"
            assert mode_for_path("/1.2.4") == "cagr"


# ═══════════════════════════════════════════════════════════════════════════
# Producer 1: the pill-click server callback
# ═══════════════════════════════════════════════════════════════════════════

class TestToggleBubView:
    @pytest.mark.parametrize("mode", _ALL)
    def test_returns_mode_then_mode_styles(self, mode):
        with patch.multiple("callbacks.charts",
                            ctx=_Ctx(VIEW_MODES[mode].pill)):
            out = toggle_bub_view(0, 0, 0, 0, 0, [2010, 2033])
        assert len(out) == 16
        assert out[0] == mode
        assert tuple(out[1:15]) == mode_styles(mode)

    def test_unknown_trigger_falls_back_to_the_default_view(self):
        with patch.multiple("callbacks.charts", ctx=_Ctx("something-else")):
            out = toggle_bub_view(0, 0, 0, 0, 0, [2010, 2033])
        assert out[0] == DEFAULT_MODE
        assert tuple(out[1:15]) == mode_styles(DEFAULT_MODE)

    def test_xrange_swaps_only_off_the_other_views_default(self):
        from layout.bubble import CAGR_DEFAULT_XRANGE
        with patch.multiple("callbacks.charts", ctx=_Ctx("bub-view-cagr")):
            assert toggle_bub_view(0, 1, 0, 0, 0, [2010, 2033])[15] == \
                CAGR_DEFAULT_XRANGE
            assert toggle_bub_view(0, 1, 0, 0, 0, [2014, 2022])[15] is NU
        for mode in (m for m in _ALL if m != "cagr"):
            with patch.multiple("callbacks.charts",
                                ctx=_Ctx(VIEW_MODES[mode].pill)):
                assert toggle_bub_view(0, 0, 0, 0, 0,
                                       list(CAGR_DEFAULT_XRANGE))[15] == \
                    [2010, 2033], mode
                assert toggle_bub_view(0, 0, 0, 0, 0,
                                       [2014, 2022])[15] is NU, mode

    def test_output_order_is_the_table_order(self):
        key = _find_callback(
            ("bub-view-mode.data", "bub-occ-ctl-wrap.style", "bub-xrange.value"),
            must_not_contain=("bub-occ-window.value",))   # that's the deep link
        ids = _output_ids(key)
        assert ids == ["bub-view-mode", *STYLE_OUTPUT_IDS, "bub-xrange"], ids


# ═══════════════════════════════════════════════════════════════════════════
# Producer 2: the clientside sync JS
# ═══════════════════════════════════════════════════════════════════════════

def _sync_script() -> str:
    hits = [s for s in _app_ctx.app._inline_scripts if SYNC_JS_MARKER in s]
    assert len(hits) == 1, f"{len(hits)} scripts carry {SYNC_JS_MARKER}"
    return hits[0]


class TestClientsideSyncTable:
    def test_embedded_json_is_exactly_the_python_table(self):
        js = _sync_script()
        m = re.search(r"var T = (\{.*\});", js, flags=re.S)
        assert m, js
        assert json.loads(m.group(1)) == {
            mode: list(mode_styles(mode)) for mode in VIEW_MODES}

    def test_every_row_has_one_value_per_output(self):
        # Positional arrays: an Output added without extending every row
        # silently mis-assigns styles rather than raising.
        table = json.loads(re.search(r"var T = (\{.*\});", _sync_script(),
                                     flags=re.S).group(1))
        assert set(table) == set(VIEW_MODES)
        assert {len(v) for v in table.values()} == {14}

    def test_fallback_is_the_default_mode(self):
        assert f'T[{json.dumps(DEFAULT_MODE)}]' in _sync_script()

    def test_output_order_is_the_table_order(self):
        # The sync callback is the only one of the three that does NOT also
        # own bub-view-mode (it is driven *by* it).
        key = _find_callback(("bub-occ-ctl-wrap.style", "bub-price-wrap.style",
                              "bub-bubble-panel.style"),
                             must_not_contain=("bub-view-mode.data",))
        assert _output_ids(key) == list(STYLE_OUTPUT_IDS)


class TestHistoricalOnlyScripts:
    """The N-future hide + the x-range cap both gate on the same mode set."""

    def _scripts(self):
        needle = "mode === '%s'" % sorted(HISTORICAL_MODES)[0]
        return [s for s in _app_ctx.app._inline_scripts
                if needle in s and SYNC_JS_MARKER not in s]

    def test_both_scripts_gate_on_exactly_the_historical_modes(self):
        scripts = self._scripts()
        assert len(scripts) == 2, len(scripts)
        for js in scripts:
            modes = re.findall(r"mode === '(\w+)'", js)
            assert modes == sorted(HISTORICAL_MODES), js

    def test_non_historical_modes_are_absent(self):
        for js in self._scripts():
            for mode in VIEW_MODES:
                if mode not in HISTORICAL_MODES:
                    assert f"mode === '{mode}'" not in js, (mode, js)


# ═══════════════════════════════════════════════════════════════════════════
# Producer 3: the deep-link server callback
# ═══════════════════════════════════════════════════════════════════════════

class TestDeepLinkUsesTheTable:
    @pytest.mark.parametrize("mode", _ALL)
    def test_deep_link_positions_1_to_14_match_mode_styles(self, mode):
        link = VIEW_MODES[mode].deep_link
        if link is None:
            pytest.skip(f"{mode} has no deep link")
        out = deep_link_bub_view(link)
        assert len(out) == 20
        assert out[0] == mode
        assert tuple(out[1:15]) == mode_styles(mode)

    def test_modes_without_a_deep_link_are_unreachable_by_url(self):
        for mode, spec in VIEW_MODES.items():
            if spec.deep_link is None:
                assert all(deep_link_bub_view(p)[0] != mode
                           for p in ("/1", "/1.1", "/2")), mode

    def test_output_order_is_the_table_order_then_the_extras(self):
        key = _find_callback(("bub-view-mode.data", "bub-occ-window.value",
                              "bub-cagr-hover-today.data"))
        assert _output_ids(key) == [
            "bub-view-mode", *STYLE_OUTPUT_IDS,
            "bub-xrange", "bub-cagr-fwd-yrs", "bub-cagr-hover-today",
            "bub-occ-tail", "bub-occ-window"]


# ═══════════════════════════════════════════════════════════════════════════
# Every id in the table is a real component
# ═══════════════════════════════════════════════════════════════════════════

class TestTableIdsExistInLayout:
    @pytest.fixture(scope="class")
    def layout_ids(self):
        # Same full-layout walk the orphan-callback guard uses (builds every
        # tab's content directly — no server, no browser).
        from test_no_orphan_callbacks import _collect_layout_ids
        return _collect_layout_ids()

    @pytest.mark.parametrize("mode", _ALL)
    def test_pill_wrap_and_ctl_ids_exist(self, mode, layout_ids):
        spec = VIEW_MODES[mode]
        for kind, cid in (("pill", spec.pill), ("wrap", spec.wrap),
                          ("ctl", spec.ctl)):
            if cid is not None:
                assert cid in layout_ids, f"{mode}.{kind}={cid} not in layout"

    def test_panel_ids_exist(self, layout_ids):
        for cid in PANEL_IDS:
            assert cid in layout_ids, cid
