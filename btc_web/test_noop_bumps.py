"""No-op writes must not trigger a chart rebuild.

`bubble-first-render` is bumped by five writers; each bump fans out to ~6
POSTs. Measured 2026-09-06 (docs/superpowers/agent-reports/
2026-09-06-r2-first-render-instrumentation.md): a plain `/1` load spends
2 bumps / 16 POSTs / 53 KB and a share-link restore 5 bumps / 30 POSTs /
136 KB — and three of those five bumps carry a value the page had ALREADY
rendered with:

* `palette-store` hydrating to `"default"`, the palette the pre-injected
  figures were built with (`tab_defaults.py`: `palette = sd("palette-store:data")`)
* the lots cascade firing with `effective-lots` going ``[] -> []``
* the snapshot restore writing `bub-sigma-mode = "resqr"` over `"resqr"`

The guards below suppress exactly those. The invariant that makes them safe
is *equality with what was rendered*, never "ignore the first fire" — a
returning CB-palette user hydrates `palette-store` to a value the server did
NOT render with, and that bump is load-bearing. Getting this backwards paints
a colourblind user's charts in the wrong palette, silently.
"""
import os
import re

import pytest

os.environ.setdefault("TESTING", "1")


def _clientside_sources():
    """Every clientside callback's JS source, keyed by first Output."""
    import app  # noqa: F401
    import _app_ctx
    out = {}
    for key, entry in _app_ctx.app.callback_map.items():
        fn = entry.get("callback")
        src = getattr(fn, "__code__", None)
        out[key] = src
    return out


def _read(path):
    here = os.path.dirname(__file__)
    return open(os.path.join(here, path)).read()


# ── The render stamp ────────────────────────────────────────────────────────

def test_render_stamp_matches_what_the_page_is_built_with():
    """The stamp must carry the SAME values the pre-injected figures used,
    or the guards suppress (or fail to suppress) the wrong bumps."""
    import app  # noqa: F401
    from layout import _build_layout, RENDER_STAMP
    from snapshot_defaults import SNAPSHOT_DEFAULTS

    assert RENDER_STAMP["palette"] == SNAPSHOT_DEFAULTS["palette-store:data"]
    assert RENDER_STAMP["sigma_mode"] == SNAPSHOT_DEFAULTS["bub-sigma-mode:value"]

    stores = {}
    _collect(_build_layout("bubble"), stores)
    assert stores["render-stamp"].data == RENDER_STAMP


def _collect(node, out):
    if node is None:
        return
    if isinstance(node, (list, tuple)):
        for x in node:
            _collect(x, out)
        return
    cid = getattr(node, "id", None)
    if isinstance(cid, str):
        out[cid] = node
    child = getattr(node, "children", None)
    if child is not None:
        _collect(child, out)


# ── The three guards ────────────────────────────────────────────────────────

def test_palette_tick_compares_against_the_rendered_palette():
    src = _read("callbacks/charts/_clientside.py")
    block = src[src.index("Palette change"):src.index("Lots / snapshot-lots")]
    assert "render-stamp" in block, (
        "palette tick must compare against the render stamp, not fire blind")
    assert "no_update" in block


def test_lots_tick_does_not_fire_when_lots_stay_empty():
    src = _read("callbacks/charts/_clientside.py")
    block = src[src.index("Lots / snapshot-lots"):]
    block = block[:block.index("prevent_initial_call")]
    # An empty-to-empty cascade must short-circuit. Any non-empty value must
    # still bump — the guard may only test emptiness, never equality of
    # contents (which would need stringifying the whole lots array per fire).
    assert re.search(r"__qsLotsEmpty|lotsWereEmpty|wasEmpty", block), (
        "lots tick must remember whether lots were already empty")


def test_sigma_mode_bump_compares_against_the_rendered_mode():
    src = _read("callbacks/scanner.py")
    block = src[src.index("sigma mode change") if "sigma mode change" in src
                else src.index("σ mode change"):]
    block = block[:block.index("prevent_initial_call")]
    assert "render-stamp" in block, (
        "sigma-mode bump must compare against the render stamp")


@pytest.mark.parametrize("cid", ["render-stamp"])
def test_new_store_is_in_the_layout(cid):
    """A callback referencing a Store that isn't mounted produces Dash
    'nonexistent object' errors — a measured prod cost, see memory
    feedback_nonexistent_input_perf.md."""
    import app  # noqa: F401
    from layout import _build_layout
    stores = {}
    _collect(_build_layout("bubble"), stores)
    assert cid in stores
