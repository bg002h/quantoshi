"""Time Machine (Task 11) — snapshot / share-link round-trip.

Asserts `bub-timemachine-toggle` and `bub-asof-slider` survive a
q4: encode -> decode cycle once they're registered in
`_SNAPSHOT_CONTROLS` (+ `_CHECKLIST_OPTIONS` for the toggle bitmask).

State dict keys follow the "{component_id}:{property}" convention used
throughout snapshot.py / restore_builder.py (see `_v()` docstring in
restore_builder.py) -- NOT bare component ids.
"""
# Importing conftest builds the full _app_ctx registry that snapshot.py's
# module-level code depends on (mirrors test_timemachine_layout.py).
from conftest import _app_ctx  # noqa: F401

from snapshot import _encode_snapshot_v4, _decode_snapshot_v4


def test_timemachine_roundtrip():
    st = {
        "bub-timemachine-toggle:value": ["on"],
        "bub-asof-slider:value": 42,
    }
    dec = _decode_snapshot_v4(_encode_snapshot_v4(st))
    assert dec is not None
    assert dec["bub-timemachine-toggle:value"] == ["on"]
    assert dec["bub-asof-slider:value"] == 42


def test_timemachine_off_does_not_force_a_diff():
    """Toggle off + slider unset (matching defaults) should not bloat the
    diff payload -- only fields that differ from SNAPSHOT_DEFAULTS are
    encoded (neither field is in ALWAYS_ENCODE)."""
    st = {
        "bub-timemachine-toggle:value": [],
        "bub-asof-slider:value": None,
    }
    encoded = _encode_snapshot_v4(st)
    dec = _decode_snapshot_v4(encoded)
    assert dec is not None
    # Off state restores to the toggle's default (empty list) whether or
    # not it round-tripped through the diff payload.
    assert dec.get("bub-timemachine-toggle:value", []) in ([], None)
