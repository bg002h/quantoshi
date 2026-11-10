"""Tab 1: turning MC on widens the X range so the simulation is visible.

A cached MC run starts in 2028/2031/2035 and the default X range ends 2033, so
without this the fan occupies a sliver at the right edge. When the USER ticks
"Activate", the right edge moves out to start year + MC_AUTOWIDEN_YEARS.

Two halves, both tested here:
  * assets/mc_autowiden.js — capture-phase click listener that timestamps a
    real click on #bub-mc-enable (window.__qsMcEnableClickAt);
  * layout.bubble.MC_AUTOWIDEN_JS — the clientside callback, which acts only
    on a fresh timestamp. A share-link restore, Time Machine's force-off and
    every other programmatic write of bub-mc-enable carry no click, so a
    restored link's X range is never overridden.

The JS is executed under node. Browser wiring (does a real tick reach the
callback?) is covered by test_mc_autowiden_e2e.py.
"""
import json
import pathlib
import shutil
import subprocess

import pytest

# conftest imports app -> callbacks/__init__ -> registers every callback. Do
# NOT import callbacks.mc_controls here: see test_axes_presets.py.
from conftest import _app_ctx  # noqa: F401

from layout.bubble import (MC_AUTOWIDEN_JS, MC_AUTOWIDEN_YEARS, XRANGE_MAX,
                           XRANGE_MIN)

_ASSET = pathlib.Path(__file__).parent / "assets" / "mc_autowiden.js"
_NODE = shutil.which("node")
needs_node = pytest.mark.skipif(_NODE is None, reason="node not installed")


# Bound at collection, before any test runs: several test_defaults.py tests
# `from btc_web.app import app`, which executes app.py a SECOND time under
# another module name and re-points _app_ctx.app at a fresh Dash object with
# none of the clientside registrations (callbacks/* is already imported).
_APP = _app_ctx.app


def _entry():
    # _callback_list is the spec Dash ships to the browser; unlike
    # callback_map it carries prevent_initial_call and clientside_function.
    hits = [e for e in _APP._callback_list
            if [i.get("id") for i in e["inputs"]] == ["bub-mc-enable"]
            and "bub-xrange.value" in str(e["output"])]
    assert len(hits) == 1, (
        f"expected exactly one bub-mc-enable -> bub-xrange callback, "
        f"found {len(hits)}")
    return hits[0]


class TestRegistration:
    def test_registered_with_a_single_input(self):
        # Multiple Inputs + allow_duplicate + prevent_initial_call silently
        # no-ops in Dash 4.0 (callbacks/plot_appearance.py:22-28).
        e = _entry()
        assert [(i["id"], i["property"]) for i in e["inputs"]] == [
            ("bub-mc-enable", "value")]

    def test_reads_range_and_start_year_as_state(self):
        e = _entry()
        assert [(s["id"], s["property"]) for s in e["state"]] == [
            ("bub-xrange", "value"), ("bub-mc-start-yr", "value")]

    def test_never_fires_on_page_load(self):
        # allow_duplicate + prevent_initial_call=False crashes gunicorn, and
        # a load-time fire would be a non-user write anyway.
        assert _entry()["prevent_initial_call"] is True

    def test_is_clientside(self):
        # ALARA: no server round-trip to move a slider.
        assert _entry()["clientside_function"]
        assert MC_AUTOWIDEN_JS.strip() in "".join(_APP._inline_scripts)

    def test_span_is_ten_years(self):
        assert MC_AUTOWIDEN_YEARS == 10

    def test_asset_listens_in_the_capture_phase(self):
        src = _ASSET.read_text()
        assert "#bub-mc-enable" in src
        assert "__qsMcEnableClickAt" in src
        assert "__qsMcEnableClickAt" in MC_AUTOWIDEN_JS


# ── the decision logic, executed ────────────────────────────────────────────

_HARNESS = r"""
const NU = '__NU__';
let now = 1000000;
Date.now = () => now;
const window = {dash_clientside: {no_update: NU}};
globalThis.window = window;
const fn = (%(fn)s);
const out = [];
for (const c of %(cases)s) {
    if (c.clickAgoMs === null) window.__qsMcEnableClickAt = 0;
    else if (c.clickAgoMs !== undefined) window.__qsMcEnableClickAt = now - c.clickAgoMs;
    out.push(fn(c.mc, c.xr, c.start));
}
console.log(JSON.stringify(out));
"""


def _run(cases):
    script = _HARNESS % {"fn": MC_AUTOWIDEN_JS, "cases": json.dumps(cases)}
    res = subprocess.run([_NODE, "-e", script], capture_output=True,
                         text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


NU = "__NU__"
ON, OFF = ["yes"], []


@needs_node
class TestDecision:
    def test_user_tick_widens_right_edge_to_start_plus_ten(self):
        assert _run([dict(clickAgoMs=5, mc=ON, xr=[2010, 2033], start=2028)]) \
            == [[2010, 2038]]

    def test_left_edge_is_never_moved(self):
        assert _run([dict(clickAgoMs=5, mc=ON, xr=[2024, 2030], start=2031)]) \
            == [[2024, 2041]]

    def test_a_wider_range_is_never_shrunk(self):
        assert _run([dict(clickAgoMs=5, mc=ON, xr=[2010, 2060], start=2028),
                     dict(clickAgoMs=5, mc=ON, xr=[2010, 2038], start=2028)]) \
            == [NU, NU]

    def test_clamped_to_the_slider_maximum(self):
        start = XRANGE_MAX - 3
        assert _run([dict(clickAgoMs=5, mc=ON, xr=[XRANGE_MIN, 2033], start=start)]) \
            == [[XRANGE_MIN, XRANGE_MAX]]

    def test_write_without_a_click_is_ignored(self):
        """Share-link restore / Time Machine / any programmatic write: the
        restored X range must survive."""
        assert _run([dict(clickAgoMs=None, mc=ON, xr=[2010, 2033], start=2028)]) \
            == [NU]

    def test_stale_click_is_ignored(self):
        assert _run([dict(clickAgoMs=60_000, mc=ON, xr=[2010, 2033], start=2028)]) \
            == [NU]

    def test_one_click_widens_once(self):
        """The timestamp is consumed, so a programmatic write that follows a
        real click inside the window cannot ride on it."""
        assert _run([dict(clickAgoMs=5, mc=ON, xr=[2010, 2033], start=2028),
                     dict(mc=ON, xr=[2010, 2033], start=2028)]) \
            == [[2010, 2038], NU]

    def test_unticking_does_nothing_and_still_consumes_the_click(self):
        assert _run([dict(clickAgoMs=5, mc=OFF, xr=[2010, 2033], start=2028),
                     dict(mc=ON, xr=[2010, 2033], start=2028)]) \
            == [NU, NU]

    @pytest.mark.parametrize("mc,xr,start", [
        (ON, [2010, 2033], None),      # no start year
        (ON, [2010, 2033], "abc"),
        (ON, None, 2028),              # slider not hydrated
        (ON, [2010], 2028),
        (None, [2010, 2033], 2028),    # stale client sends a non-list
        ("yes", [2010, 2033], 2028),
    ])
    def test_malformed_values_are_a_no_op(self, mc, xr, start):
        assert _run([dict(clickAgoMs=5, mc=mc, xr=xr, start=start)]) == [NU]

    def test_numeric_string_start_year_is_accepted(self):
        assert _run([dict(clickAgoMs=5, mc=ON, xr=[2010, 2033], start="2031")]) \
            == [[2010, 2041]]


_LISTENER_HARNESS = r"""
let handler = null, capture = null, evt = null;
globalThis.window = {};
globalThis.document = {addEventListener: (t, h, c) => { evt = t; handler = h; capture = c; }};
let now = 4242;
Date.now = () => now;
%(src)s
const inside = {closest: (sel) => sel === '#bub-mc-enable' ? {} : null};
const outside = {closest: () => null};
const res = {evt, capture};
handler({target: outside});
res.afterOutside = window.__qsMcEnableClickAt || 0;
handler({target: inside});
res.afterInside = window.__qsMcEnableClickAt || 0;
handler({target: null});
handler({target: {}});
console.log(JSON.stringify(res));
"""


@needs_node
def test_listener_stamps_only_clicks_inside_the_mc_enable_checklist():
    script = _LISTENER_HARNESS % {"src": _ASSET.read_text()}
    res = subprocess.run([_NODE, "-e", script], capture_output=True,
                         text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout)
    assert got == {"evt": "click", "capture": True,
                   "afterOutside": 0, "afterInside": 4242}
