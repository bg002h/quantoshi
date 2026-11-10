"""assets/chart_responsive.js: the trace-width override honours an opt-out.

The script polls every chart and restyles EVERY line trace to the user's
trace width (2.5 px, x1.5 on desktop). A trace that sets
``meta.qs_keep_width`` keeps the width the server gave it — the Tab-1 MC
example paths need this, or eight (or 200) of them at 3.75 px bury the fan.

The script is executed under node against a fake document / Plotly.
"""
import json
import pathlib
import shutil
import subprocess

import pytest

_ASSET = pathlib.Path(__file__).parent / "assets" / "chart_responsive.js"
_NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(_NODE is None, reason="node not installed")

_HARNESS = r"""
const calls = [];
let tick = null;
const graphs = %(graphs)s;
globalThis.window = {innerWidth: %(width)d, QS_COLORS: {}, QS_APPEARANCE: {},
                     __chartResponsiveInterval: null};
globalThis.localStorage = {getItem: () => null};
globalThis.document = {getElementById: (id) => {
    const g = graphs[id];
    if (!g) return null;
    return {querySelector: () => g};
}};
globalThis.Plotly = {
    restyle: (g, upd, idx) => {
        calls.push({upd, idx});
        idx.forEach((i, k) => { g.data[i].line.width = upd['line.width'][k]; });
    },
    relayout: () => {},
};
globalThis.setInterval = (fn) => { tick = fn; return 1; };
globalThis.clearInterval = () => {};
%(src)s
const out = [];
for (let n = 0; n < %(ticks)d; n++) {
    calls.length = 0;
    tick();
    out.push(calls.filter(c => c.upd['line.width']).map(c => ({
        idx: c.idx, widths: c.upd['line.width']})));
}
console.log(JSON.stringify({out, widths: graphs['bubble-graph'].data.map(
    t => t.line ? t.line.width : null)}));
"""


def _run(data, width=1400, ticks=1):
    graphs = {"bubble-graph": {"data": data, "layout": {}}}
    script = _HARNESS % {"graphs": json.dumps(graphs), "width": width,
                         "ticks": ticks, "src": _ASSET.read_text()}
    res = subprocess.run([_NODE, "-e", script], capture_output=True,
                         text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _line(width, **extra):
    return {"mode": "lines", "line": {"width": width}, **extra}


KEEP = {"meta": {"qs_keep_width": True}}


def test_ordinary_line_traces_get_the_desktop_width():
    got = _run([_line(2.5), _line(2.0)])
    assert got["out"][0] == [{"idx": [0, 1], "widths": [3.75, 3.75]}]


def test_mobile_width_is_not_multiplied():
    got = _run([_line(2.0)], width=400)
    assert got["widths"] == [2.5]


def test_keep_width_traces_are_left_alone():
    got = _run([_line(2.5), _line(1.0, **KEEP), _line(1.0, **KEEP), _line(2.0)])
    assert got["out"][0] == [{"idx": [0, 3], "widths": [3.75, 3.75]}]
    assert got["widths"] == [3.75, 1.0, 1.0, 3.75]


def test_a_leading_keep_width_trace_does_not_cause_a_restyle_every_poll():
    """needsApply compares the FIRST line trace with the target width. A
    keep-width trace in that slot never matches, which would restyle the
    chart on every 500 ms poll forever."""
    got = _run([_line(1.0, **KEEP), _line(2.5)], ticks=3)
    assert got["out"][0] == [{"idx": [1], "widths": [3.75]}]
    assert got["out"][1] == [] and got["out"][2] == []


def test_meta_without_the_flag_is_not_an_opt_out():
    got = _run([_line(1.0, meta={"other": 1}), _line(1.0, meta="text")])
    assert got["widths"] == [3.75, 3.75]
