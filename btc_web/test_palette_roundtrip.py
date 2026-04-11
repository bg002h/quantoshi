"""Regression test for Display Models palette round-trip invariance.

Bug: switching the palette dropdown (default → cb-* → default) used to leak
HybPPL family variants (hybppl_dd, hyb2l, hyb2c, hyb2b, hyb4d, linppl) and
cfg_*/ecfg_* config instances into the bub-model-show checklist, and
stripped gear icons for the 4 master model entries (bub-bm-gear,
bub-eppl-gear, bub-lppl-gear, bub-hybppl-gear).

These tests pin the `value` list emitted by `update_model_swatches` to the
initial layout's value list for every palette and every transition.
"""
import pytest

import _app_ctx  # noqa: F401  (ensures PRICE_MODELS / PALETTES populated)
from callbacks.charts import update_model_swatches
from layout.bubble import _build_bub_model_options
from layout.common import _model_show_checklist


PALETTE_KEYS = ["default", "cb-brian", "cb-rg", "cb-full"]


def _values(opts):
    return [o["value"] for o in opts]


def _initial_bub_values():
    mc = _app_ctx.PALETTES["default"].get(
        "model_colors", _app_ctx.MODEL_TRACE_COLORS
    )
    return _values(_build_bub_model_options(mc))


def _initial_sim_values(prefix):
    # _model_show_checklist returns a list of dcc.Checklist + friends; grab
    # the Checklist component's options.
    children = _model_show_checklist(prefix, standardized=True, include_mc=False)
    for child in children:
        if hasattr(child, "options") and getattr(child, "id", None) == f"{prefix}-model-show":
            return _values(child.options)
    raise AssertionError(f"Could not locate {prefix}-model-show checklist")


@pytest.mark.parametrize("palette_key", PALETTE_KEYS)
def test_palette_rebuild_matches_initial_bub(palette_key):
    initial = _initial_bub_values()
    bub_opts, *_ = update_model_swatches(palette_key)
    assert _values(bub_opts) == initial, (
        f"bub-model-show values drifted on palette={palette_key}"
    )


@pytest.mark.parametrize("palette_key", PALETTE_KEYS)
@pytest.mark.parametrize("prefix", ["dca", "ret", "sc"])
def test_palette_rebuild_sim_no_leaks(palette_key, prefix):
    """Sim tabs (DCA/Retire/SC) must never leak HybPPL family or cfg_* entries."""
    _, dca_opts, ret_opts, sc_opts = update_model_swatches(palette_key)
    per_prefix = {"dca": dca_opts, "ret": ret_opts, "sc": sc_opts}
    vals = set(_values(per_prefix[prefix]))
    leaks = {v for v in vals
             if v in _app_ctx.HYBPPL_FAMILY_HIDDEN
             or v.startswith("cfg_")
             or v.startswith("ecfg_")}
    assert not leaks, f"{prefix}-model-show leaked {leaks} on palette={palette_key}"


def test_roundtrip_all_palettes_stable():
    """Cycle through every palette transition; value list must stay constant."""
    initial = _initial_bub_values()
    for pk in PALETTE_KEYS + ["default"]:  # ensure default round-trip at end
        bub_opts, *_ = update_model_swatches(pk)
        assert _values(bub_opts) == initial, f"drift at palette={pk}"


def test_gear_icons_preserved_on_palette_rebuild():
    """The 4 master entries must still carry gear icon spans (bub-*-gear)."""
    bub_opts, *_ = update_model_swatches("cb-rg")
    # Each master entry's label is an html.Span; find gear ids nested in it.
    gear_ids = set()

    def _walk(node):
        if hasattr(node, "id") and getattr(node, "id", None):
            gear_ids.add(node.id)
        for attr in ("children",):
            c = getattr(node, attr, None)
            if c is None:
                continue
            if isinstance(c, (list, tuple)):
                for sub in c:
                    _walk(sub)
            else:
                _walk(c)

    for opt in bub_opts:
        _walk(opt["label"])
    expected = {"bub-bm-gear", "bub-eppl-gear", "bub-lppl-gear", "bub-hybppl-gear"}
    missing = expected - gear_ids
    assert not missing, f"missing gear ids after palette rebuild: {missing}"
