"""Snapshot/routing/defaults tests for the bub-sigma-mode control."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

import snapshot  # noqa: E402
import tab_defaults  # noqa: E402
from callbacks.routing import _TAB_CONTROLS  # noqa: E402


def test_sigma_mode_in_snapshot_controls():
    cids = [cid for cid, _ in snapshot._SNAPSHOT_CONTROLS]
    assert "bub-sigma-mode" in cids


def test_sigma_mode_is_value_not_checklist():
    """bub-sigma-mode is a radio (value, not a checklist). It must NOT live
    in _CHECKLIST_OPTIONS or its bitmask encoder would mangle the string."""
    assert "bub-sigma-mode" not in snapshot._CHECKLIST_OPTIONS


def test_sigma_mode_in_bubble_tab_controls():
    assert "bub-sigma-mode" in _TAB_CONTROLS["bubble"]


def test_sigma_mode_not_in_other_tabs():
    for tab_key, ids in _TAB_CONTROLS.items():
        if tab_key == "bubble":
            continue
        assert "bub-sigma-mode" not in ids, tab_key


def test_bubble_default_sigma_mode_is_resqr():
    assert tab_defaults.BUBBLE["sigma_mode"] == "resqr"
    assert tab_defaults.bubble_defaults()["sigma_mode"] == "resqr"


def test_snapshot_roundtrip_preserves_sigma_mode_resqr():
    state = {"bub-sigma-mode:value": "resqr", "main-tabs:active_tab": "bubble"}
    encoded = snapshot._encode_snapshot(state)
    decoded = snapshot._decode_snapshot(encoded)
    assert decoded["bub-sigma-mode:value"] == "resqr"


def test_snapshot_roundtrip_preserves_sigma_mode_constant():
    state = {"bub-sigma-mode:value": "constant", "main-tabs:active_tab": "bubble"}
    encoded = snapshot._encode_snapshot(state)
    decoded = snapshot._decode_snapshot(encoded)
    assert decoded["bub-sigma-mode:value"] == "constant"


def test_snapshot_missing_sigma_mode_absent_on_decode():
    """When sigma_mode is absent (old link), it's not in the decoded dict —
    the restore callback falls back to the default at layout time."""
    state = {"main-tabs:active_tab": "bubble"}
    encoded = snapshot._encode_snapshot(state)
    decoded = snapshot._decode_snapshot(encoded)
    assert "bub-sigma-mode:value" not in decoded


def test_tab_filter_excludes_sigma_mode_for_non_bubble_tabs():
    state = {"bub-sigma-mode:value": "resqr", "main-tabs:active_tab": "heatmap"}
    encoded = snapshot._encode_snapshot(
        state, tab_filter=_TAB_CONTROLS["heatmap"],
    )
    decoded = snapshot._decode_snapshot(encoded)
    assert "bub-sigma-mode:value" not in decoded
