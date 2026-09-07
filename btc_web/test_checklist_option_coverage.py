"""Every checklist option a user can tick must survive a share link.

F-12: `"show_ucl"` (Unfairly Cheap Line) was offered by the Tab-1 Display
checklist but missing from `_CHECKLIST_OPTIONS["bub-toggles"]`, which is the
ordered list the bitmask encoder uses. `_list_to_mask` skips values it does
not know, so the option round-tripped to nothing:

    ['shade', 'show_data', 'show_ucl'] -> mask 5 -> ['shade', 'show_data']

Nine options in the layout, eight bits in the encoder. The user ticked a box,
shared the link, and the recipient silently got a different chart.

The general guard below walks the LAYOUT and compares against the encoder, so
the next option added to any bitmask-encoded checklist cannot repeat this.

ORDER IS THE ENCODING. Each position in `_CHECKLIST_OPTIONS[cid]` is one bit,
so a value may only ever be APPENDED — inserting or reordering silently
changes the meaning of every existing share link. `test_bitmask_order_is_append_only`
pins the prefix that already shipped.
"""
import os

import pytest

os.environ.setdefault("TESTING", "1")


def _layout_checklist_options():
    """{component_id: [option values]} for every dcc.Checklist in the layout."""
    import app  # noqa: F401
    from layout import _build_layout
    from dash import dcc

    found = {}

    def walk(node):
        if node is None:
            return
        if isinstance(node, (list, tuple)):
            for x in node:
                walk(x)
            return
        if isinstance(node, dcc.Checklist):
            cid = getattr(node, "id", None)
            opts = getattr(node, "options", None)
            if isinstance(cid, str) and opts:
                vals = []
                for o in opts:
                    if isinstance(o, dict) and "value" in o:
                        vals.append(o["value"])
                if vals:
                    found[cid] = vals
        for prop in ("children",):
            walk(getattr(node, prop, None))

    walk(_build_layout("bubble"))
    return found


def test_show_ucl_survives_a_share_link():
    """The specific F-12 regression, named so a failure is unambiguous."""
    from snapshot import _CHECKLIST_OPTIONS, _list_to_mask, _mask_to_list

    opts = _CHECKLIST_OPTIONS["bub-toggles"]
    value = ["shade", "show_data", "show_ucl"]
    back = _mask_to_list(_list_to_mask(value, opts), opts)
    assert "show_ucl" in back, (
        f"show_ucl was dropped by the bitmask round trip: {value} -> {back}")


def test_every_layout_checklist_option_is_bitmask_encodable():
    """Any option the layout offers, for a checklist that IS bitmask-encoded,
    must appear in that checklist's encoder list."""
    from snapshot import _CHECKLIST_OPTIONS

    missing = []
    for cid, layout_vals in _layout_checklist_options().items():
        if cid not in _CHECKLIST_OPTIONS:
            continue          # not bitmask-encoded; stored as a plain list
        known = set(_CHECKLIST_OPTIONS[cid])
        for v in layout_vals:
            if v not in known:
                missing.append(f"{cid}: {v!r} offered in layout, absent from "
                               f"_CHECKLIST_OPTIONS")
    assert not missing, (
        "checklist options that cannot survive a share link:\n  "
        + "\n  ".join(missing))


def test_bitmask_order_is_append_only():
    """Each position is a bit. Reordering or inserting rewrites the meaning of
    every share link ever generated, silently."""
    from snapshot import _CHECKLIST_OPTIONS

    shipped_prefix = ["shade", "show_ols", "show_data", "show_today",
                      "show_legend", "minor_grid", "chart_zoom", "show_halvings"]
    actual = _CHECKLIST_OPTIONS["bub-toggles"]
    assert actual[:len(shipped_prefix)] == shipped_prefix, (
        "the first 8 bub-toggles bits are load-bearing for every link issued "
        "before 2026-09-07 — new values append AFTER them")


@pytest.mark.parametrize("value", [
    [],
    ["show_ucl"],
    ["shade", "show_data", "show_today"],
    ["shade", "show_ols", "show_data", "show_today", "show_legend",
     "minor_grid", "chart_zoom", "show_halvings", "show_ucl"],
])
def test_bub_toggles_round_trip(value):
    from snapshot import _CHECKLIST_OPTIONS, _list_to_mask, _mask_to_list

    opts = _CHECKLIST_OPTIONS["bub-toggles"]
    back = _mask_to_list(_list_to_mask(value, opts), opts)
    assert sorted(back) == sorted(value)


def test_old_links_still_decode_the_same():
    """Appending a bit must not disturb the existing eight. A mask built from
    the pre-F-12 list must still mean exactly what it meant."""
    from snapshot import _CHECKLIST_OPTIONS, _mask_to_list

    old_opts = ["shade", "show_ols", "show_data", "show_today",
                "show_legend", "minor_grid", "chart_zoom", "show_halvings"]
    new_opts = _CHECKLIST_OPTIONS["bub-toggles"]
    for mask in range(256):                       # every pre-existing encoding
        assert _mask_to_list(mask, old_opts) == _mask_to_list(mask, new_opts), (
            f"mask {mask} changed meaning when the list grew")
