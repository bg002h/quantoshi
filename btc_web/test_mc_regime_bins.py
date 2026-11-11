"""MC regime checklists vs the bins control (follow-ups F-24, F-25).

F-24: the bins control on Tabs 2-6 runs 5-10, but the share-link bitmask
tables for their regime checklists were 5 wide, so a ticked bin >= 5 was
dropped on encode (``_list_to_mask`` skips values it does not know). The
tables are positional: widening appends bits and leaves old links' bits 0-4
where they were.

F-25: raising the bin count used to leave the regime value at the old bins,
silently filtering the new top bins. A clientside callback now ticks the new
bins -- only when every old bin was ticked, and only on a real user change of
the bins dropdown (click stamp; never on a share-link restore).
"""
import json
import pathlib
import shutil
import subprocess

import pytest

from conftest import _CHECKLIST_OPTIONS, _TAB_CONTROLS, _list_to_mask, _mask_to_list
from snapshot import _decode_snapshot, _decode_snapshot_v4, _encode_snapshot, _encode_snapshot_v4

# prefix -> tab id, for "current tab only" links
_TABS = {"dca": "dca", "ret": "retire", "hm": "heatmap", "sc": "supercharge",
         "cp": "citadel"}


class TestF24TenBinBitmask:
    @pytest.mark.parametrize("prefix", sorted(_TABS))
    def test_table_reaches_index_nine(self, prefix):
        assert _CHECKLIST_OPTIONS[f"{prefix}-mc-regime"] == list(range(10))

    @pytest.mark.parametrize("prefix", sorted(_TABS))
    @pytest.mark.parametrize("tab_only", [False, True])
    def test_ten_bin_round_trip(self, prefix, tab_only):
        state = {f"{prefix}-mc-enable:value": ["yes"],
                 f"{prefix}-mc-bins:value": 10,
                 f"{prefix}-mc-regime:value": [0, 5, 9]}
        tf = _TAB_CONTROLS[_TABS[prefix]] if tab_only else None
        got = _decode_snapshot_v4(_encode_snapshot_v4(state, tab_filter=tf))
        assert got[f"{prefix}-mc-bins:value"] == 10
        assert got[f"{prefix}-mc-regime:value"] == [0, 5, 9]

    @pytest.mark.parametrize("prefix", sorted(_TABS))
    def test_positional_round_trip(self, prefix):
        state = {f"{prefix}-mc-enable:value": ["yes"],
                 f"{prefix}-mc-regime:value": [1, 6, 8]}
        got = _decode_snapshot(_encode_snapshot(state))
        assert got[f"{prefix}-mc-regime:value"] == [1, 6, 8]

    def test_old_masks_decode_unchanged(self):
        """Every mask an old (5-wide) link could carry decodes to the same
        list under the widened table."""
        old, new = [0, 1, 2, 3, 4], list(range(10))
        for mask in range(32):
            assert _mask_to_list(mask, new) == _mask_to_list(mask, old)
            assert _list_to_mask(_mask_to_list(mask, old), new) == mask


# ── F-25 ────────────────────────────────────────────────────────────────────
_PREFIXES = ("dca", "ret", "hm", "sc", "cp", "bub")


def _bins_callbacks():
    from conftest import _app_ctx
    out = {}
    for e in _app_ctx.app._callback_list:
        ins = [(i["id"], i["property"]) for i in e["inputs"]]
        if len(ins) == 1 and ins[0][0].endswith("-bins") and "-mc-" in ins[0][0] \
                and "-regime.value" in str(e["output"]):
            out[ins[0][0]] = (str(e["output"]), e)
    return out


class TestF25Wiring:
    def test_one_tick_callback_per_bins_control(self):
        cbs = _bins_callbacks()
        want = {f"{p}-mc-bins" for p in _PREFIXES} | {"bub-mc-raw-bins"}
        assert set(cbs) == want

    def test_shape(self):
        """ONE Input, allow_duplicate output, no initial call (Dash 4.0
        silently drops allow_duplicate + several Inputs + prevent_initial_call),
        and the regime value read as State."""
        for bins_id, (key, cb) in _bins_callbacks().items():
            regime = bins_id.replace("-bins", "-regime")
            assert key.startswith(f"{regime}.value@"), key   # allow_duplicate hash
            assert cb.get("prevent_initial_call") is True
            states = [(s["id"], s["property"]) for s in cb.get("state", [])]
            assert states == [(regime, "value")], states


_NODE = shutil.which("node")


def _run_js(body):
    src = (pathlib.Path(__file__).parent / "assets" / "mc_bins_tick.js").read_text()
    prog = f"""
const window = {{dash_clientside: {{no_update: "NO_UPDATE"}}}};
const listeners = [];
const document = {{addEventListener: (t, f, c) => listeners.push([t, f, c]),
                   querySelector: (sel) => null}};
{src}
const tick = window.__qsBinsTick;
const out = [];
{body}
console.log(JSON.stringify(out));
"""
    r = subprocess.run([_NODE, "-e", prog], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_wrapper_names_its_own_bins_id():
    from layout.mc_controls import mc_bins_tick_js
    js = mc_bins_tick_js("ret-mc-bins")
    assert '"ret-mc-bins"' in js and "__qsBinsTick" in js


@pytest.mark.skipif(_NODE is None, reason="node not installed")
class TestF25Logic:
    def test_user_raise_with_all_ticked_ticks_new_bins(self):
        out = _run_js("""
window.__qsBinsUserAt = {"bub-mc-bins": Date.now()};
out.push(tick(8, [0,1,2,3,4], "bub-mc-bins", 5));
""")
        assert out == [[0, 1, 2, 3, 4, 5, 6, 7]]

    def test_unstamped_change_is_left_alone(self):
        """A share-link restore writes bins without a click: no change."""
        out = _run_js("""
out.push(tick(8, [0,1,2,3,4], "bub-mc-bins", 5));
""")
        assert out == ["NO_UPDATE"]

    def test_partial_selection_is_left_alone(self):
        out = _run_js("""
window.__qsBinsUserAt = {"dca-mc-bins": Date.now()};
out.push(tick(8, [0,2], "dca-mc-bins", 5));
""")
        assert out == ["NO_UPDATE"]

    def test_stale_stamp_is_ignored_and_consumed(self):
        out = _run_js("""
window.__qsBinsUserAt = {"bub-mc-raw-bins": Date.now() - 10000};
out.push(tick(8, [0,1,2,3,4], "bub-mc-raw-bins", 5));
out.push(window.__qsBinsUserAt["bub-mc-raw-bins"] === undefined);
""")
        assert out == ["NO_UPDATE", True]

    def test_stamp_is_consumed(self):
        out = _run_js("""
window.__qsBinsUserAt = {"hm-mc-bins": Date.now()};
out.push(tick(7, [0,1,2,3,4], "hm-mc-bins", 5));
out.push(tick(9, [0,1,2,3,4,5,6], "hm-mc-bins", 5));
""")
        assert out == [[0, 1, 2, 3, 4, 5, 6], "NO_UPDATE"]

    def test_previous_bins_tracked_across_restores(self):
        """A restore to 8 (no click) then a user raise to 10 with all 8
        ticked: the previous count is the restored 8, so all 10 get ticked."""
        out = _run_js("""
out.push(tick(8, [0,1,2,3,4,5,6,7], "sc-mc-bins", 5));
window.__qsBinsUserAt = {"sc-mc-bins": Date.now()};
out.push(tick(10, [0,1,2,3,4,5,6,7], "sc-mc-bins", 5));
""")
        assert out == ["NO_UPDATE", list(range(10))]

    def test_restored_deliberate_filter_survives_later_raise_check(self):
        """Restored bins 8 with [0..4] (top bins deliberately off); a user
        raise to 9 must NOT tick them -- not every old (8) bin was ticked."""
        out = _run_js("""
out.push(tick(8, [0,1,2,3,4], "bub-mc-bins", 5));
window.__qsBinsUserAt = {"bub-mc-bins": Date.now()};
out.push(tick(9, [0,1,2,3,4], "bub-mc-bins", 5));
""")
        assert out == ["NO_UPDATE", "NO_UPDATE"]

    def test_lowering_is_left_alone(self):
        out = _run_js("""
window.__qsBinsUserAt = {"cp-mc-bins": Date.now()};
window.__qsBinsPrev = {"cp-mc-bins": 8};
out.push(tick(5, [0,1,2,3,4,5,6,7], "cp-mc-bins", 5));
""")
        assert out == ["NO_UPDATE"]

    def test_string_values_are_accepted(self):
        out = _run_js("""
window.__qsBinsUserAt = {"bub-mc-bins": Date.now()};
out.push(tick("6", ["0","1","2","3","4"], "bub-mc-bins", 5));
""")
        assert out == [[0, 1, 2, 3, 4, 5]]

    def test_click_on_trigger_or_its_popup_stamps(self):
        """A click on the bins trigger, or anywhere inside the popup its
        aria-controls names (Dash 4 portals the option list), stamps that
        bins id; a click elsewhere stamps nothing."""
        out = _run_js("""
const trig = {id: "bub-mc-bins", parentElement: null};
document.querySelector = (sel) => sel === '[aria-controls="pop1"]' ? trig : null;
const popup = {id: "pop1", parentElement: null};
const option = {id: "", parentElement: {id: "", parentElement: popup}};
const other = {id: "x", parentElement: {id: "y", parentElement: null}};
const click = listeners.find(l => l[0] === "click");
out.push(click[2] === true);                    // capture phase
click[1]({target: option});
out.push(typeof window.__qsBinsUserAt["bub-mc-bins"]);
window.__qsBinsUserAt = {};
click[1]({target: other});
out.push(Object.keys(window.__qsBinsUserAt).length);
click[1]({target: {id: "", parentElement: trig}});
out.push(typeof window.__qsBinsUserAt["bub-mc-bins"]);
const key = listeners.find(l => l[0] === "keydown");
window.__qsBinsUserAt = {};
key[1]({key: "Enter", target: option});
out.push(typeof window.__qsBinsUserAt["bub-mc-bins"]);
window.__qsBinsUserAt = {};
key[1]({key: "a", target: option});
out.push(Object.keys(window.__qsBinsUserAt).length);
""")
        assert out == [True, "number", 0, "number", "number", 0]
