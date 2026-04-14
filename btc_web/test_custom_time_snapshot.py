"""Snapshot encode/decode tests for Custom Time Axis controls."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "btc_web"))

from snapshot import (  # noqa: E402
    _encode_snapshot, _decode_snapshot,
    _SNAPSHOT_CONTROLS, _CHECKLIST_OPTIONS,
)


def _base_state():
    """Minimal state dict with None for every snapshot control key."""
    return {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}


def test_cta_ids_registered():
    ids = {cid for cid, _ in _SNAPSHOT_CONTROLS}
    for expected in ["cta-active", "cta-scale",
                      "cta-t0-cal", "cta-t0-cal-custom",
                      "cta-t0-blk", "cta-t0-blk-custom",
                      "cta-weighting", "cta-models"]:
        assert expected in ids, f"{expected} missing from _SNAPSHOT_CONTROLS"


def test_cta_models_order_frozen():
    """Bitmask encoding depends on this exact order; reordering breaks
    all existing share links."""
    assert _CHECKLIST_OPTIONS["cta-models"] == ["pl", "qr", "bm_floor", "exp"]
    assert _CHECKLIST_OPTIONS["cta-active"] == ["yes"]


def test_cta_active_roundtrip_empty():
    state = _base_state()
    state["cta-active:value"] = []
    encoded = _encode_snapshot(state)
    decoded = _decode_snapshot(encoded)
    assert decoded.get("cta-active:value") in (None, [])


def test_cta_active_roundtrip_set():
    state = _base_state()
    state["cta-active:value"] = ["yes"]
    encoded = _encode_snapshot(state)
    decoded = _decode_snapshot(encoded)
    assert decoded["cta-active:value"] == ["yes"]


def test_cta_models_bitmask_all_combinations():
    """All 16 subsets of {pl, qr, bm_floor, exp} roundtrip."""
    for r in range(5):
        for combo in itertools.combinations(
            ["pl", "qr", "bm_floor", "exp"], r):
            state = _base_state()
            state["cta-models:value"] = list(combo)
            encoded = _encode_snapshot(state)
            decoded = _decode_snapshot(encoded)
            assert set(decoded.get("cta-models:value") or []) == set(combo), combo


def test_cta_weighting_roundtrip():
    for mode in ["none", "inv_t", "inv_sqrt_t", "log_density"]:
        state = _base_state()
        state["cta-weighting:value"] = mode
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["cta-weighting:value"] == mode


def test_cta_t0_cal_custom_date_roundtrip():
    state = _base_state()
    state["cta-t0-cal-custom:date"] = "2012-05-15"
    encoded = _encode_snapshot(state)
    decoded = _decode_snapshot(encoded)
    assert decoded["cta-t0-cal-custom:date"] == "2012-05-15"


def test_all_cta_ids_in_bubble_tab_controls():
    from callbacks.routing import _TAB_CONTROLS
    bubble_set = _TAB_CONTROLS["bubble"]
    for expected in ["cta-active", "cta-scale",
                      "cta-t0-cal", "cta-t0-cal-custom",
                      "cta-t0-blk", "cta-t0-blk-custom",
                      "cta-weighting", "cta-models"]:
        assert expected in bubble_set, \
            f"{expected} not in _TAB_CONTROLS['bubble']"
