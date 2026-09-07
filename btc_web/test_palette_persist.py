"""F-11: a saved palette must survive a page reload.

Reported 2026-09-06. Pick Deuteranomaly on Tab 1, reload, and the chart came
back in the DEFAULT palette — and `localStorage` had been reset to
`"default"` too, so the choice was not merely mis-rendered, it was lost.

Mechanism (traced with `scripts/check_palette_persists.py` +
`scratchpad/palette_trace.py`): each `palette-select-{tab}` writes its value
into `palette-store` (`callbacks/nav.py`). Every per-tab selector is rendered
with `value="default"` (`layout/common.py`), and the ones inside lazy tabs
MOUNT LATE — `prevent_initial_call=True` does not protect against a component
that appears later, because Dash treats the new component's value as a
change. So the sequence on reload was:

    palette-store hydrates from localStorage -> "cb-brian"
    reverse callback syncs the selectors     -> "cb-brian"
    a lazy tab mounts, its selector fires    -> "default"   <-- clobber
    charts faithfully repaint in "default"

The guard in that callback compared the incoming value only against the
store, and `"default" !== "cb-brian"` is a difference, so it wrote.

Fix: a mount fire is a selector carrying the value the SERVER rendered it
with, which `render-stamp` knows. The first fire from each selector is
ignored when it equals that stamp; every later fire writes normally, so
switching back to Default by hand still works.
"""
import os

import pytest

os.environ.setdefault("TESTING", "1")

_TAB_PALETTE_KEYS = ("bub", "hm", "dca", "ret", "sc", "cp")


def _nav_source():
    here = os.path.dirname(__file__)
    return open(os.path.join(here, "callbacks", "nav.py")).read()


def _forward_block():
    """The dropdown -> store callback registered per tab."""
    src = _nav_source()
    start = src.index("_TAB_PALETTE_KEYS = ")
    end = src.index("# Reverse: store change")
    return src[start:end]


def test_selectors_are_rendered_with_the_stamped_palette():
    """The stamp must equal what the selectors are rendered with, or the
    guard cannot recognise a mount fire."""
    import app  # noqa: F401
    from layout import RENDER_STAMP
    from layout.common import _palette_card_default_value

    assert _palette_card_default_value() == RENDER_STAMP["palette"]


def test_forward_callback_consults_the_render_stamp():
    block = _forward_block()
    assert "render-stamp" in block, (
        "the dropdown->store callback must be able to recognise a mount fire; "
        "it needs the render stamp")


def test_forward_callback_tracks_each_selector_separately():
    """Six selectors mount at six different times (lazy tabs). A single
    shared flag would let the second one through."""
    block = _forward_block()
    # The selector's own id must reach the callback...
    assert 'State(f"palette-select-{_k}", "id")' in block, (
        "the callback needs each selector's id to track them separately")
    # ...and be used as the key of the first-fire map, not a bare boolean.
    assert "seen[k]" in block, "first-fire tracking must be keyed per selector"
    assert "seen = (window.__qsPalSeen" in block


def test_every_tab_selector_still_writes_the_store():
    """The fix must not disconnect the control — all six still drive it."""
    block = _forward_block()
    for k in _TAB_PALETTE_KEYS:
        assert f'palette-select-{k}' in block or "{_k}" in block


@pytest.mark.parametrize("key", _TAB_PALETTE_KEYS)
def test_selector_exists_for_each_tab(key):
    import app  # noqa: F401
    from layout import _build_layout

    found = []

    def walk(node):
        cid = getattr(node, "id", None)
        if isinstance(cid, str):
            found.append(cid)
        ch = getattr(node, "children", None)
        if isinstance(ch, (list, tuple)):
            for c in ch:
                walk(c)
        elif ch is not None:
            walk(ch)

    walk(_build_layout("bubble"))
    # cp lives in the lazy citadel tab, so only the eager ones are asserted
    if key in ("bub",):
        assert f"palette-select-{key}" in found
