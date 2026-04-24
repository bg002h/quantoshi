# btc_web/test_snapshot_defaults.py
"""Structural invariants for snapshot_defaults SSOT. Phase 1."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))


def test_every_snapshot_control_has_default():
    from snapshot import _SNAPSHOT_CONTROLS
    from snapshot_defaults import SNAPSHOT_DEFAULTS
    missing = [(c, p) for c, p in _SNAPSHOT_CONTROLS
               if f"{c}:{p}" not in SNAPSHOT_DEFAULTS]
    assert not missing, (
        f"{len(missing)} control(s) missing from SNAPSHOT_DEFAULTS: "
        f"{missing[:5]}{'...' if len(missing) > 5 else ''}")


def test_no_phantom_keys_in_snapshot_defaults():
    from snapshot import _SNAPSHOT_CONTROLS
    from snapshot_defaults import SNAPSHOT_DEFAULTS
    valid = {f"{c}:{p}" for c, p in _SNAPSHOT_CONTROLS}
    phantom = [k for k in SNAPSHOT_DEFAULTS if k not in valid]
    assert not phantom, (
        f"SNAPSHOT_DEFAULTS contains keys not in _SNAPSHOT_CONTROLS: {phantom}")


def test_checklist_defaults_are_lists():
    from snapshot import _CHECKLIST_OPTIONS
    from snapshot_defaults import SNAPSHOT_DEFAULTS
    for cid in _CHECKLIST_OPTIONS:
        key = f"{cid}:value"
        if key not in SNAPSHOT_DEFAULTS:
            continue
        v = SNAPSHOT_DEFAULTS[key]
        assert v is None or isinstance(v, list), (
            f"Checklist default {key} must be list or None; "
            f"got {type(v).__name__}: {v!r}")


def test_fingerprint_is_8_hex_chars():
    from snapshot_defaults import _compute_snapshot_defaults_fingerprint
    fp = _compute_snapshot_defaults_fingerprint()
    assert len(fp) == 8
    int(fp, 16)


def test_fingerprint_is_deterministic():
    from snapshot_defaults import _compute_snapshot_defaults_fingerprint
    assert (_compute_snapshot_defaults_fingerprint()
            == _compute_snapshot_defaults_fingerprint())


def test_bub_toggles_default_round_trips_through_bitmask():
    from snapshot import _CHECKLIST_OPTIONS, _list_to_mask, _mask_to_list
    from snapshot_defaults import SNAPSHOT_DEFAULTS
    val = SNAPSHOT_DEFAULTS["bub-toggles:value"]
    if val is None:
        pytest.skip("bub-toggles default is None")
    opts = _CHECKLIST_OPTIONS["bub-toggles"]
    mask = _list_to_mask(val, opts)
    round_tripped = _mask_to_list(mask, opts)
    assert sorted(round_tripped) == sorted(val), (
        f"bub-toggles default {val} does not round-trip: {round_tripped}")


def test_always_encode_members_in_snapshot_controls():
    from snapshot import _SNAPSHOT_CONTROLS
    from snapshot_defaults import ALWAYS_ENCODE
    valid = {f"{c}:{p}" for c, p in _SNAPSHOT_CONTROLS}
    missing = [k for k in ALWAYS_ENCODE if k not in valid]
    assert not missing, (
        f"ALWAYS_ENCODE contains keys not in _SNAPSHOT_CONTROLS: {missing}")


def test_mc_enable_keys_exist_in_snapshot_controls():
    """Phase 2 _mc_null_out_diffs_v4 looks up *-mc-enable keys; missing
    keys would StopIteration. Note: ret-mc-enable defaults to ['yes']
    intentionally (Retire is the MC showcase tab); other tabs default
    disabled. Test asserts existence + valid-checklist shape only."""
    from snapshot import _SNAPSHOT_CONTROLS
    from snapshot_defaults import SNAPSHOT_DEFAULTS
    prefixes = ("dca-mc-", "ret-mc-", "hm-mc-", "sc-mc-", "cp-mc-")
    cids = {c for c, _ in _SNAPSHOT_CONTROLS}
    for pfx in prefixes:
        cid = f"{pfx}enable"
        assert cid in cids, (
            f"_SNAPSHOT_CONTROLS missing {cid} - "
            f"_mc_null_out_diffs_v4 lookup would StopIteration")
        val = SNAPSHOT_DEFAULTS.get(f"{cid}:value")
        assert val is None or isinstance(val, list), (
            f"{cid}:value must be checklist-shaped; got {val!r}")
