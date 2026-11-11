"""Tab-1 Raw MC (model-free Monte Carlo): share-link plumbing.

Eight controls were appended to the tail of ``_SNAPSHOT_CONTROLS`` (indices
340-347) so that a Raw-mode configuration survives a share link. Every piece
of this fails *silently* -- the chart still draws -- so it is pinned here:

* index-addressed tail append (v4 links address fields by array position);
* defaults + the defaults fingerprint (old fingerprint still registered);
* the checklist bitmask tables -- ``bub-mc-raw-regime`` must cover every bin
  index the control can reach (bins run 3-10, so indices 0-9), or a checked
  bin above the table is dropped on encode;
* "Current tab only" links (``_TAB_CONTROLS["bubble"]``);
* links made before the fields existed.
"""
import base64
import gzip
import json

import pytest

from conftest import (_SNAPSHOT_CONTROLS, _CHECKLIST_OPTIONS, _list_to_mask,  # noqa: F401
                      _mask_to_list, _TAB_CONTROLS)
from snapshot import (_decode_snapshot, _decode_snapshot_v4, _encode_snapshot,
                      _encode_snapshot_v4, _registry_lookup)
from snapshot_defaults import (SNAPSHOT_DEFAULTS,
                               _compute_snapshot_defaults_fingerprint)

NEW = ["bub-mc-raw-sims", "bub-mc-raw-years", "bub-mc-raw-window",
       "bub-mc-raw-trend", "bub-mc-raw-bins", "bub-mc-raw-regime",
       "bub-mc-lines", "bub-mc-extremes"]
KEYS = [f"{c}:value" for c in NEW]
# The last fingerprint registered before the Raw fields existed.
_OLD_FP = "04aa50d6"


def _idx(cid):
    return [c for c, _ in _SNAPSHOT_CONTROLS].index(cid)


def _v4(fp, diffs):
    """A q4 payload exactly as _encode_snapshot_v4 lays it out, for any fp."""
    j = json.dumps([fp, diffs, None], separators=(",", ":"))
    return f"{fp}:" + base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()


# A Raw-mode state with every new field off its default. 10 bins with bin 9
# (the highest index the control can reach) ticked: the bitmask table must be
# wide enough to carry it.
RAW_STATE = {
    "bub-mc-enable:value": ["yes"],
    "bub-mc-model-src:value": "raw_cycle",
    "bub-mc-raw-sims:value": 2500,
    "bub-mc-raw-years:value": 20,
    "bub-mc-raw-window:value": [2014, 2024],
    "bub-mc-raw-trend:value": "exp",
    "bub-mc-raw-bins:value": 10,
    "bub-mc-raw-regime:value": [0, 3, 9],
    "bub-mc-lines:value": 75,
    "bub-mc-extremes:value": ["yes"],
}


class TestRegistry:
    def test_appended_at_tail_in_order(self):
        """v4 links are index-addressed: anywhere but the tail shifts every
        later field and corrupts already-shipped links."""
        assert [c for c, _ in _SNAPSHOT_CONTROLS[340:348]] == NEW
        assert all(p == "value" for _, p in _SNAPSHOT_CONTROLS[340:348])
        # The Percentile cycle-fit fields follow at 348-351
        # (test_pctile_cycle.py pins them); nothing may come between.
        assert _SNAPSHOT_CONTROLS[348][0] == "bub-pctile-fit"

    def test_earlier_entries_kept_their_indices(self):
        assert _SNAPSHOT_CONTROLS[339] == ("bub-mc-paths", "value")
        assert _SNAPSHOT_CONTROLS[338] == ("bub-ma", "value")

    def test_raw_ids_in_bubble_tab_controls(self):
        """Missing here, a "Current tab only" link silently drops the field."""
        missing = [c for c in NEW if c not in _TAB_CONTROLS["bubble"]]
        assert not missing, missing

    def test_raw_ids_belong_to_no_other_tab(self):
        for tab, ids in _TAB_CONTROLS.items():
            if tab != "bubble":
                assert not set(NEW) & set(ids), tab

    def test_checklist_options(self):
        assert _CHECKLIST_OPTIONS["bub-mc-extremes"] == ["yes"]
        # bins 3..10 -> the regime checklist can hold indices 0..9
        assert _CHECKLIST_OPTIONS["bub-mc-raw-regime"] == list(range(10))

    def test_only_the_two_checklists_are_bitmasked(self):
        for cid in NEW:
            assert (cid in _CHECKLIST_OPTIONS) == (
                cid in ("bub-mc-raw-regime", "bub-mc-extremes")), cid


class TestDefaults:
    def test_defaults(self):
        got = [SNAPSHOT_DEFAULTS[k] for k in KEYS]
        # raw-window is a static literal, same as bub-mc-window (the layout's
        # slider max is the current year; the default is not re-derived).
        assert got == [1000, 40, [2010, 2026], "power", 5, [0, 1, 2, 3, 4],
                       20, []]

    def test_window_default_matches_the_model_window(self):
        assert (SNAPSHOT_DEFAULTS["bub-mc-raw-window:value"]
                == SNAPSHOT_DEFAULTS["bub-mc-window:value"])

    def test_defaults_agree_with_the_raw_mc_constants(self):
        import raw_mc
        assert SNAPSHOT_DEFAULTS["bub-mc-lines:value"] == raw_mc.RAW_LINES_DEFAULT
        assert (raw_mc.RAW_LINES_MIN <= SNAPSHOT_DEFAULTS["bub-mc-lines:value"]
                <= raw_mc.RAW_LINES_MAX)
        assert SNAPSHOT_DEFAULTS["bub-mc-raw-trend:value"] in raw_mc.RAW_TRENDS

    def test_current_fingerprint_is_registered(self):
        fp = _compute_snapshot_defaults_fingerprint()
        reg = _registry_lookup(fp)
        assert reg is not None, (
            "run tools/update_defaults_registry.py after editing "
            "snapshot_defaults.py")
        for k in KEYS:
            assert reg[k] == SNAPSHOT_DEFAULTS[k], k

    def test_old_fingerprint_registered(self):
        """The pre-change fp still decodes to the old defaults: a link made
        before Raw MC restores exactly as it was, not under the new defaults."""
        assert _compute_snapshot_defaults_fingerprint() != _OLD_FP
        old = _registry_lookup(_OLD_FP)
        assert old is not None
        for k in KEYS:
            assert k not in old, f"fixture: {_OLD_FP} must predate {k}"
        assert old["bub-mc-paths:value"] == []
        assert old["bub-mc-sims:value"] == 200


class TestLayoutAgreesWithDefaults:
    """A default that drifts from the layout makes the sparse diff lie: a
    link would omit a field the layout then restores to a different value."""

    def test_raw_controls_start_at_their_snapshot_defaults(self):
        from layout.bubble import _bubble_controls
        from test_mc_paths_toggle import _find
        root = _bubble_controls()
        for cid in NEW:
            hits = _find(root, cid, [])
            assert len(hits) == 1, cid
            want = SNAPSHOT_DEFAULTS[f"{cid}:value"]
            got = hits[0].value
            if cid == "bub-mc-raw-window":
                # slider max is the live year; the default is a static literal
                assert got[0] == want[0] and got[1] >= want[1], (got, want)
            else:
                assert got == want, (cid, got, want)


class TestRoundTrip:
    @pytest.mark.parametrize("tab_filter", [None, "bubble"])
    def test_round_trip_raw_state(self, tab_filter):
        tf = _TAB_CONTROLS[tab_filter] if tab_filter else None
        state = _decode_snapshot_v4(_encode_snapshot_v4(RAW_STATE, tab_filter=tf))
        for k, v in RAW_STATE.items():
            assert state[k] == v, k

    def test_ten_bin_selection_keeps_index_nine(self):
        """The regime table is as wide as the widest bin count (10)."""
        for sel in ([9], [0, 9], list(range(10)), [1, 4, 8, 9]):
            st = dict(RAW_STATE, **{"bub-mc-raw-regime:value": sel})
            got = _decode_snapshot_v4(_encode_snapshot_v4(st))
            assert got["bub-mc-raw-regime:value"] == sel, sel

    def test_regime_mask_covers_every_reachable_index(self):
        opts = _CHECKLIST_OPTIONS["bub-mc-raw-regime"]
        for n_bins in range(3, 11):
            sel = list(range(n_bins))
            assert _mask_to_list(_list_to_mask(sel, opts), opts) == sel

    def test_positional_q3_round_trip(self):
        """Legacy positional encoding shares the same tables."""
        got = _decode_snapshot(_encode_snapshot(RAW_STATE))
        for k, v in RAW_STATE.items():
            assert got[k] == v, k

    def test_raw_model_src_values_survive(self):
        for src in ("raw_iid", "raw_cycle"):
            st = {"bub-mc-enable:value": ["yes"], "bub-mc-model-src:value": src}
            got = _decode_snapshot_v4(_encode_snapshot_v4(st))
            assert got["bub-mc-model-src:value"] == src

    def test_defaults_cost_nothing(self):
        """A Raw link whose new fields all sit at defaults carries none of
        them -- the whole point of the sparse diff."""
        base = {"bub-mc-enable:value": ["yes"],
                "bub-mc-model-src:value": "raw_iid"}
        st = dict(base, **{k: SNAPSHOT_DEFAULTS[k] for k in KEYS})
        assert _encode_snapshot_v4(st) == _encode_snapshot_v4(base)

    def test_regime_and_extremes_encoded_as_bitmasks(self):
        fp = _compute_snapshot_defaults_fingerprint()
        enc = _encode_snapshot_v4(RAW_STATE)
        raw = gzip.decompress(base64.urlsafe_b64decode(enc.split(":", 1)[1]))
        got_fp, diffs, _lots = json.loads(raw)
        assert got_fp == fp
        assert diffs[str(_idx("bub-mc-raw-regime"))] == (1 << 0) | (1 << 3) | (1 << 9)
        assert diffs[str(_idx("bub-mc-extremes"))] == 1
        assert diffs[str(_idx("bub-mc-raw-window"))] == [2014, 2024]


class TestLinksMadeBeforeRawMc:
    def test_old_v4_link_restores_old_values(self):
        """An old-fp link: its own controls keep the old defaults and the new
        fields are simply absent (the layout default stays in place)."""
        state = _decode_snapshot_v4(_v4(_OLD_FP, {}))
        assert state is not None
        assert state["bub-mc-sims:value"] == 200
        assert not [k for k in KEYS if k in state]

    def test_old_v4_link_diffs_land_on_the_same_controls(self):
        """The tail append moved nothing: an old link's diff for an existing
        index still decodes onto the same control."""
        diffs = {str(_idx("bub-mc-sims")): 400, str(_idx("bub-mc-paths")): 1}
        state = _decode_snapshot_v4(_v4(_OLD_FP, diffs))
        assert state["bub-mc-sims:value"] == 400
        assert state["bub-mc-paths:value"] == ["yes"]
        assert not [k for k in KEYS if k in state]

    def test_old_positional_link_pads_the_new_tail(self):
        """q3 links hold 340 values; the 8 new slots pad to None -> absent."""
        j = json.dumps([[None] * 340, None], separators=(",", ":"))
        enc = base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()
        state = _decode_snapshot(enc)
        assert state is not None
        assert not [k for k in KEYS if state.get(k) is not None]

    def test_model_mode_link_is_unaffected(self):
        """A model-mode link made today restores its model-mode fields and
        never grows a non-default Raw selection."""
        st = {"bub-mc-enable:value": ["yes"], "bub-mc-model-src:value": "bub",
              "bub-mc-sims:value": 800, "bub-mc-regime:value": [0, 2]}
        got = _decode_snapshot_v4(_encode_snapshot_v4(st))
        for k, v in st.items():
            assert got[k] == v, k
        for k in KEYS:
            assert got[k] == SNAPSHOT_DEFAULTS[k], k
