"""Tab 1: "Show individual price paths" (``bub-mc-paths``), default OFF.

With MC on, Tab 1 draws the percentile fan. The 200 example paths on top of it
are opt-in: at the default zoom they bury the bands they are meant to
illustrate. What the toggle does to the figure is tested next to the other MC
callback tests (test_callbacks.py::TestUpdateBubbleCallbackMC); this file pins
the plumbing, every piece of which fails *silently* — the chart draws either
way:

* snapshot control list (index-addressed: tail-append only), checklist bitmask
  table, snapshot defaults + fingerprint registry, tab-control set;
* the callback Input (declaration order IS the argument order) and the
  restore short-circuit set;
* the layout control (Tab 1 only);
* share links made before the toggle existed, which showed the paths.
"""
import ast
import base64
import gzip
import json
import pathlib

import pytest

from conftest import (_app_ctx, _SNAPSHOT_CONTROLS, _CHECKLIST_OPTIONS,  # noqa: F401
                      _list_to_mask, _mask_to_list, _TAB_CONTROLS)
from snapshot import (_decode_snapshot, _decode_snapshot_v4, _encode_snapshot,
                      _encode_snapshot_v4, _registry_lookup)
from snapshot_defaults import (SNAPSHOT_DEFAULTS,
                               _compute_snapshot_defaults_fingerprint)

KEY = "bub-mc-paths:value"
# The last fingerprint registered before the toggle existed (Tab-1 sims 8→200).
_OLD_FP = "72f9cada"


def _idx(cid):
    return [c for c, _ in _SNAPSHOT_CONTROLS].index(cid)


def _v4(fp, diffs):
    """A q4 payload exactly as _encode_snapshot_v4 lays it out, for any fp."""
    j = json.dumps([fp, diffs, None], separators=(",", ":"))
    return f"{fp}:" + base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()


class TestSnapshotRegistry:
    def test_appended_at_the_absolute_tail(self):
        """v4 links are index-addressed: anywhere but the tail shifts every
        later field and corrupts already-shipped links."""
        assert _SNAPSHOT_CONTROLS[339] == ("bub-mc-paths", "value")
        # Raw MC (Task 7) appended 8 more at 340-347; nothing may follow
        # them (test_raw_mc_snapshot.py pins that block).
        assert len(_SNAPSHOT_CONTROLS) == 348

    def test_earlier_entries_kept_their_indices(self):
        assert _SNAPSHOT_CONTROLS[337] == ("bub-occ-window", "value")
        assert _SNAPSHOT_CONTROLS[338] == ("bub-ma", "value")

    def test_checklist_options_registered(self):
        assert _CHECKLIST_OPTIONS["bub-mc-paths"] == ["yes"]

    def test_default_is_off(self):
        assert SNAPSHOT_DEFAULTS[KEY] == []

    def test_in_the_bubble_tab_control_set(self):
        """Missing here, a "Current tab only" link silently drops the field."""
        assert "bub-mc-paths" in _TAB_CONTROLS["bubble"]

    def test_current_fingerprint_is_registered(self):
        fp = _compute_snapshot_defaults_fingerprint()
        assert _registry_lookup(fp) is not None, (
            "run tools/update_defaults_registry.py after editing "
            "snapshot_defaults.py")
        assert _registry_lookup(fp)[KEY] == []

    def test_previous_fingerprint_is_still_registered(self):
        old = _registry_lookup(_OLD_FP)
        assert old is not None
        assert KEY not in old, "fixture: _OLD_FP must predate the toggle"


class TestShareLinkRoundTrip:
    @pytest.mark.parametrize("tab_filter", [None, "bubble"])
    def test_paths_on_survives(self, tab_filter):
        tf = _TAB_CONTROLS[tab_filter] if tab_filter else None
        enc = _encode_snapshot_v4({"bub-mc-enable:value": ["yes"],
                                   KEY: ["yes"]}, tab_filter=tf)
        assert _decode_snapshot_v4(enc)[KEY] == ["yes"]

    def test_paths_off_survives_and_costs_nothing(self):
        on = _encode_snapshot_v4({"bub-mc-enable:value": ["yes"], KEY: ["yes"]})
        off = _encode_snapshot_v4({"bub-mc-enable:value": ["yes"], KEY: []})
        base = _encode_snapshot_v4({"bub-mc-enable:value": ["yes"]})
        assert _decode_snapshot_v4(off)[KEY] == []
        assert off == base            # the default is not encoded at all
        assert on != base

    def test_encoded_as_a_bitmask(self):
        fp = _compute_snapshot_defaults_fingerprint()
        enc = _encode_snapshot_v4({KEY: ["yes"]})
        raw = gzip.decompress(base64.urlsafe_b64decode(enc.split(":", 1)[1]))
        got_fp, diffs, lots = json.loads(raw)
        # (the encoder also force-emits unrelated atomic-block fields)
        assert (got_fp, lots) == (fp, None)
        assert diffs[str(_idx("bub-mc-paths"))] == 1


# What a real link carries for Tab 1's MC tick: bub-mc-enable is NOT in
# _CHECKLIST_OPTIONS (unlike the other tabs' *-mc-enable), so it travels as a
# plain list rather than a bitmask.
_MC_ON = ["yes"]


class TestLinksMadeBeforeTheToggle:
    """Before the toggle, MC on Tab 1 always drew the example paths. A link
    from then carries no value for the field; it must keep its paths rather
    than silently restore as fan-only."""

    def test_old_v4_link_with_mc_on_keeps_its_paths(self):
        state = _decode_snapshot_v4(
            _v4(_OLD_FP, {str(_idx("bub-mc-enable")): _MC_ON}))
        assert state["bub-mc-enable:value"] == ["yes"]
        assert state[KEY] == ["yes"]

    def test_old_v4_link_without_mc_is_left_alone(self):
        """No MC, no paths to preserve: the control keeps its layout default
        (off) rather than arriving pre-ticked."""
        state = _decode_snapshot_v4(_v4(_OLD_FP, {}))
        assert state is not None
        assert KEY not in state

    def test_old_positional_link_with_mc_on_keeps_its_paths(self):
        """q3/q2 links are a positional array, shorter than today's list."""
        n_old = _idx("bub-mc-paths")
        values = [None] * n_old
        values[_idx("bub-mc-enable")] = _MC_ON
        j = json.dumps([values, None], separators=(",", ":"))
        enc = base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()
        state = _decode_snapshot(enc)
        assert state["bub-mc-enable:value"] == ["yes"]
        assert state[KEY] == ["yes"]

    def test_new_link_with_mc_on_and_paths_off_stays_off(self):
        """The rule keys on the field being ABSENT, never on its value."""
        fp = _compute_snapshot_defaults_fingerprint()
        state = _decode_snapshot_v4(_v4(fp, {str(_idx("bub-mc-enable")): _MC_ON}))
        assert state[KEY] == []

    def test_new_positional_link_with_paths_off_stays_off(self):
        state = _decode_snapshot(_encode_snapshot(
            {"bub-mc-enable:value": ["yes"], KEY: []}))
        assert state["bub-mc-enable:value"] == ["yes"]
        assert state[KEY] == []


class TestCallbackWiring:
    @staticmethod
    def _fn():
        path = (pathlib.Path(__file__).parent / "callbacks" / "charts"
                / "__init__.py")
        return next(n for n in ast.walk(ast.parse(path.read_text()))
                    if isinstance(n, ast.FunctionDef)
                    and n.name == "update_bubble")

    def _decorator_ids(self):
        dec = next(d for d in self._fn().decorator_list
                   if isinstance(d, ast.Call))
        return [(a.func.id, a.args[0].value) for a in dec.args
                if isinstance(a, ast.Call) and isinstance(a.func, ast.Name)
                and a.func.id in ("Input", "State")]

    def test_is_the_last_input_and_maps_onto_mc_paths(self):
        """Dash maps decorator args onto params by declaration order. Appended
        LAST so a browser holding the previous callback signature still lines
        up for every earlier argument."""
        seq = self._decorator_ids()
        params = [a.arg for a in self._fn().args.args]
        assert len(seq) == len(params), (len(seq), len(params))
        # bub-mc-paths was last until Raw MC (Task 5) appended its 8 Inputs
        # after it, in snapshot order -- nothing else may follow it.
        raw = ["bub-mc-raw-sims", "bub-mc-raw-years", "bub-mc-raw-window",
               "bub-mc-raw-trend", "bub-mc-raw-bins", "bub-mc-raw-regime",
               "bub-mc-lines", "bub-mc-extremes"]
        assert seq[-8:] == [("Input", i) for i in raw]
        assert seq[-9] == ("Input", "bub-mc-paths")
        assert params[-9] == "mc_paths"
        assert seq[-10] == ("Input", "bub-asof-slider")
        assert params[-10] == "asof_slider"

    def test_in_post_restore_triggers(self):
        found = None
        for node in ast.walk(self._fn()):
            if (isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "_POST_RESTORE_TRIGGERS"):
                found = {e.value for e in node.value.elts
                         if isinstance(e, ast.Constant)}
        assert found is not None, "_POST_RESTORE_TRIGGERS assignment not found"
        assert "bub-mc-paths" in found

    def test_registered_on_the_live_callback(self):
        # Server callbacks live in Dash's global map until the first request
        # merges them; _all_callbacks() is the union (see its docstring).
        from test_no_orphan_callbacks import _all_callbacks
        hits = [k for k, e in _all_callbacks().items()
                if "bubble-graph.figure" in k
                and any(i.get("id") == "bub-mc-paths" for i in e["inputs"])]
        assert len(hits) == 1, hits


def _find(node, cid, out):
    if isinstance(node, (list, tuple)):
        for c in node:
            _find(c, cid, out)
        return out
    if getattr(node, "id", None) == cid:
        out.append(node)
    ch = getattr(node, "children", None)
    if ch is not None:
        _find(ch, cid, out)
    return out


class TestLayout:
    def test_one_checklist_on_tab_1_default_off(self):
        from layout.bubble import _bubble_controls
        hits = _find(_bubble_controls(), "bub-mc-paths", [])
        assert len(hits) == 1
        box = hits[0]
        assert [o["value"] for o in box.options] == ["yes"]
        assert box.value == list(SNAPSHOT_DEFAULTS[KEY]) == []
        assert "path" in box.options[0]["label"].lower()

    def test_sits_inside_the_mc_panel_body(self):
        """Hidden with the rest of the MC settings until MC is activated."""
        from layout.bubble import _bubble_controls
        body = _find(_bubble_controls(), "bub-mc-body", [])
        assert len(body) == 1
        assert len(_find(body[0], "bub-mc-paths", [])) == 1

    @pytest.mark.parametrize("prefix", ["dca", "ret", "hm", "sc", "cp"])
    def test_other_tabs_do_not_get_one(self, prefix):
        from layout.mc_controls import _mc_controls
        assert _find(_mc_controls(prefix), f"{prefix}-mc-paths", []) == []

    def test_placeholder_exists_when_mc_is_unavailable(self, monkeypatch):
        """Without the Markov engine the panel is hidden placeholders; the
        callback Input must still resolve or Dash logs a nonexistent-object
        error on every Tab-1 render."""
        from layout import mc_controls
        monkeypatch.setattr(mc_controls._app_ctx, "_HAS_MARKOV", False)
        root = mc_controls._mc_controls("bub", show_paths_toggle=True,
                                        include_raw_src=True)
        hits = _find(root, "bub-mc-paths", [])
        assert len(hits) == 1
        assert hits[0].value == []
        # Raw MC (Task 5): the 9 controls/status plus the mode-class panel.
        for cid in ("bub-mc-raw-sims", "bub-mc-raw-years", "bub-mc-raw-window",
                    "bub-mc-raw-trend", "bub-mc-raw-bins", "bub-mc-raw-regime",
                    "bub-mc-lines", "bub-mc-extremes", "bub-mc-raw-status",
                    "bub-mc-panel"):
            assert len(_find(root, cid, [])) == 1, cid
        for prefix in ("dca", "ret", "hm", "sc", "cp"):
            assert _find(mc_controls._mc_controls(prefix), "bub-mc-panel", []) == []
