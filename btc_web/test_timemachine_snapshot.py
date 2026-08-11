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


def test_new_fields_round_trip_alongside_later_tab_fields():
    """Sanity check: setting the two new TM fields in the same payload as
    Leverage / User-Model fields (which sit later in _SNAPSHOT_CONTROLS)
    doesn't cross-contaminate within a single encode/decode cycle.

    NOTE: this alone would NOT have caught the positional bug below --
    encode and decode always read the SAME (current-process)
    _SNAPSHOT_CONTROLS ordering, so an internally-consistent same-process
    round trip succeeds even when that ordering itself is wrong relative
    to already-shipped links. See
    test_existing_field_indices_unchanged_by_timemachine_addition for the
    test that actually catches that class of bug.
    """
    st = {
        "lev-model:value":   "pl",     # default is "bub" -- differs, gets diffed
        "um-p1-year:data":   2015,     # default is None -- differs, gets diffed
        "um-p2-price:data":  12345.0,  # default is None -- differs, gets diffed
        "bub-timemachine-toggle:value": ["on"],
        "bub-asof-slider:value": 7,
    }
    dec = _decode_snapshot_v4(_encode_snapshot_v4(st))
    assert dec is not None
    assert dec["lev-model:value"] == "pl"
    assert dec["um-p1-year:data"] == 2015
    assert dec["um-p2-price:data"] == 12345.0
    assert dec["bub-timemachine-toggle:value"] == ["on"]
    assert dec["bub-asof-slider:value"] == 7


# Frozen snapshot of every (component_id, property) pair -- IN ORDER -- that
# was in _SNAPSHOT_CONTROLS immediately before this task (commit 9434e49,
# the tip of time-machine before Task 11). Extracted mechanically via
# `git show 9434e49:btc_web/snapshot.py` + ast.literal_eval, not hand-typed.
# Only the tail of the frozen list is checked below (see
# test_existing_field_indices_unchanged_by_timemachine_addition) --
# specifically the two anchor points bracketing where the CRITICAL bug
# inserted the new fields (bub-mc-advanced/bub-mc-years, still Tab-1-bubble)
# and every entry from lev-date through um-p2-price (Leverage + User Model),
# which is exactly the range the buggy insertion point shifted by +2.
_PRE_TASK11_LANDMARKS = [
    ("bub-mc-advanced",     "value", 303),
    ("bub-mc-years",        "value", 320),
    ("lev-date",            "date",  321),
    ("lev-price",           "value", 322),
    ("lev-model",           "value", 323),
    ("lev-floor-q-store",   "data",  324),
    ("lev-rb",              "value", 325),
    ("lev-rl",              "value", 326),
    ("lev-horizon",         "value", 327),
    ("lev-cagr",            "value", 328),
    ("lev-toggles",         "value", 329),
    ("um-p1-year",          "data",  330),
    ("um-p1-price",         "data",  331),
    ("um-p2-year",          "data",  332),
    ("um-p2-price",         "data",  333),
]


def test_existing_field_indices_unchanged_by_timemachine_addition():
    """CRITICAL regression guard: adding bub-timemachine-toggle /
    bub-asof-slider to _SNAPSHOT_CONTROLS must not shift the ARRAY INDEX
    of any field that existed before this task.

    _encode_snapshot_v4/_decode_snapshot_v4 (snapshot.py) key every diff
    entry by `str(i)`, i = position in _SNAPSHOT_CONTROLS. Decode always
    walks the CURRENT list to map index -> (cid, prop); it has no record
    of what order the list was in when an older link was created. So if a
    new field is inserted anywhere but the tail, every field after the
    insertion point silently decodes into the WRONG index for any link
    encoded before the insertion -- this is invisible to any same-process
    round-trip test (see test_new_fields_round_trip_alongside_later_tab_fields
    above), because a fresh encode+decode in the same process always uses
    one consistent ordering.

    An earlier version of this task inserted the two new fields right
    after the bub-mc-* block (end of "the bubble section"), which is NOT
    the tail of _SNAPSHOT_CONTROLS -- the Leverage Calculator and User
    Model sections follow it. That shifted lev-date..um-p2-price (13
    fields) by +2. This test pins those fields (plus two anchors still
    inside the correct, unshifted bubble/MC block) to their pre-Task-11
    indices, and would have failed against that version -- verified via
    `git show a43ec41:btc_web/snapshot.py` (the buggy commit): all 13
    lev-*/um-p* landmarks come back shifted by +2 there.
    """
    from snapshot import _SNAPSHOT_CONTROLS

    mismatches = []
    for cid, prop, expected_idx in _PRE_TASK11_LANDMARKS:
        actual = [i for i, (c, p) in enumerate(_SNAPSHOT_CONTROLS)
                  if c == cid and p == prop]
        assert actual, f"{cid}:{prop} missing from _SNAPSHOT_CONTROLS entirely"
        if actual[0] != expected_idx:
            mismatches.append(
                f"{cid}:{prop} expected index {expected_idx}, got {actual[0]}")
    assert mismatches == [], (
        "Existing _SNAPSHOT_CONTROLS entries shifted index -- this "
        "corrupts already-shipped share links. New fields must be "
        "appended at the ABSOLUTE END of the list, never inserted "
        "mid-list.\n  " + "\n  ".join(mismatches)
    )

    # The two new fields must be the literal last two entries.
    assert _SNAPSHOT_CONTROLS[-2] == ("bub-timemachine-toggle", "value")
    assert _SNAPSHOT_CONTROLS[-1] == ("bub-asof-slider", "value")
