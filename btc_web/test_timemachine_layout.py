"""Time Machine (Task 9) — layout-only control block on Tab 1.

Asserts the control IDs the Task 10 callbacks will wire up are present in
`_bubble_controls()`. No callback behavior is tested here (Task 10).
"""
# Importing conftest imports app, which builds the full _app_ctx registry
# (PRICE_MODELS etc.) that layout.bubble's module-level code depends on.
from conftest import _app_ctx  # noqa: F401

from layout.bubble import _bubble_controls

_EXPECTED_IDS = {
    "bub-timemachine-toggle",
    "bub-asof-slider",
    "bub-asof-play",
    "bub-asof-interval",
    "bub-asof-label",
    "bub-timemachine-body",
}


def _collect_ids(component):
    """Recursively walk `.children` collecting every `.id` found.

    Dash pattern-matching ids (dicts, e.g. from `_mc_controls`) are not
    hashable, so only string ids are collected into the set.
    """
    ids = set()

    def _walk(node):
        cid = getattr(node, "id", None)
        if isinstance(cid, str):
            ids.add(cid)
        children = getattr(node, "children", None)
        if children is None:
            return
        if not isinstance(children, (list, tuple)):
            children = [children]
        for child in children:
            if hasattr(child, "children") or hasattr(child, "id"):
                _walk(child)

    _walk(component)
    return ids


def test_timemachine_ids_present():
    ids = _collect_ids(_bubble_controls())
    assert _EXPECTED_IDS <= ids, f"missing: {_EXPECTED_IDS - ids}"
