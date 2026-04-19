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
from layout.display_models import build_display_models_options


PALETTE_KEYS = ["default", "cb-brian", "cb-rg", "cb-full"]


def _values(opts):
    return [o["value"] for o in opts]


def _initial_bub_values():
    mc = _app_ctx.PALETTES["default"].get(
        "model_colors", _app_ctx.MODEL_TRACE_COLORS
    )
    return _values(build_display_models_options(mc, "bub", include_bm_master=True))


def _walk_component(node, visit):
    """Walk a Dash layout tree, calling `visit(node)` for every component.

    Recurses into:
      - `.children` (scalar, list, or tuple of children)
      - `.options[i]["label"]` for components that have `options` (Checklist
        etc.), because Display Models inline summary spans and gear buttons
        live NESTED inside Checklist option labels — not in `.children`.
      - Arbitrary dict/list structures encountered along the way.
    """
    if node is None:
        return
    if isinstance(node, (list, tuple)):
        for sub in node:
            _walk_component(sub, visit)
        return
    if isinstance(node, dict):
        for v in node.values():
            _walk_component(v, visit)
        return
    if hasattr(node, "id") or hasattr(node, "children"):
        visit(node)
        c = getattr(node, "children", None)
        if c is not None:
            _walk_component(c, visit)
        opts = getattr(node, "options", None)
        if opts is not None and isinstance(opts, (list, tuple)):
            for opt in opts:
                if isinstance(opt, dict):
                    label = opt.get("label")
                    if label is not None:
                        _walk_component(label, visit)


def _live_component_ids(layout):
    """Collect every component id in the layout tree, including ids
    nested inside Checklist option labels (where summary-inline spans
    and gear buttons live)."""
    ids = set()
    def _visit(node):
        nid = getattr(node, "id", None)
        if isinstance(nid, str):
            ids.add(nid)
    _walk_component(layout, _visit)
    return ids


def _live_component_ids_excluding(layout, exclude_id):
    """Collect ids EXCEPT those inside the subtree rooted at `exclude_id`.

    Used by `test_no_mini_card_ids_anywhere` to assert that activate ids
    appear ONLY inside `_defunct-snapshot-placeholders`, not anywhere else
    in the layout.
    """
    ids = set()
    def _walk(node, inside_excluded):
        if node is None:
            return
        if isinstance(node, (list, tuple)):
            for sub in node:
                _walk(sub, inside_excluded)
            return
        if isinstance(node, dict):
            for v in node.values():
                _walk(v, inside_excluded)
            return
        if not (hasattr(node, "id") or hasattr(node, "children")):
            return
        nid = getattr(node, "id", None)
        now_excluded = inside_excluded or (nid == exclude_id)
        if isinstance(nid, str) and not now_excluded:
            ids.add(nid)
        c = getattr(node, "children", None)
        if c is not None:
            _walk(c, now_excluded)
        opts = getattr(node, "options", None)
        if opts is not None and isinstance(opts, (list, tuple)):
            for opt in opts:
                if isinstance(opt, dict):
                    label = opt.get("label")
                    if label is not None:
                        _walk(label, now_excluded)
    _walk(layout, False)
    return ids


@pytest.mark.parametrize("palette_key", PALETTE_KEYS)
def test_palette_rebuild_matches_initial_bub(palette_key):
    initial = _initial_bub_values()
    bub_opts, *_ = update_model_swatches(palette_key, None)
    assert _values(bub_opts) == initial, (
        f"bub-model-show values drifted on palette={palette_key}"
    )


@pytest.mark.parametrize("palette_key", PALETTE_KEYS)
@pytest.mark.parametrize("prefix", ["dca", "ret", "sc"])
def test_palette_rebuild_sim_no_leaks(palette_key, prefix):
    """Sim tabs (DCA/Retire/SC) must never leak HybPPL family or cfg_* entries."""
    _, dca_opts, ret_opts, sc_opts = update_model_swatches(palette_key, None)
    per_prefix = {"dca": dca_opts, "ret": ret_opts, "sc": sc_opts}
    vals = set(_values(per_prefix[prefix]))
    # After Task 5 refactor, "hybppl" is a legitimate MASTER entry in sim tabs.
    # Only its hidden variants (hybppl_dd, hyb2l, hyb2c, hyb2b, hyb4d, linppl)
    # should be considered leaks, plus any cfg_*/ecfg_* config instances.
    hidden_variants = _app_ctx.HYBPPL_FAMILY_HIDDEN - {"hybppl"}
    leaks = {v for v in vals
             if v in hidden_variants
             or v.startswith("cfg_")
             or v.startswith("ecfg_")}
    assert not leaks, f"{prefix}-model-show leaked {leaks} on palette={palette_key}"


def test_roundtrip_all_palettes_stable():
    """Cycle through every palette transition; value list must stay constant."""
    initial = _initial_bub_values()
    for pk in PALETTE_KEYS + ["default"]:  # ensure default round-trip at end
        bub_opts, *_ = update_model_swatches(pk, None)
        assert _values(bub_opts) == initial, f"drift at palette={pk}"


def test_gear_icons_preserved_on_palette_rebuild():
    """The 4 master entries must still carry gear icon spans (bub-*-gear)."""
    bub_opts, *_ = update_model_swatches("cb-rg", None)
    gear_ids = set()
    def _visit(node):
        nid = getattr(node, "id", None)
        if isinstance(nid, str):
            gear_ids.add(nid)
    for opt in bub_opts:
        _walk_component(opt["label"], _visit)
    expected = {"bub-bm-gear", "bub-eppl-gear", "bub-lppl-gear", "bub-hybppl-gear"}
    missing = expected - gear_ids
    assert not missing, f"missing gear ids after palette rebuild: {missing}"


@pytest.mark.parametrize("prefix,palette_key", [
    (p, pk) for p in ("bub", "dca", "ret", "sc")
             for pk in ("default", "cb-brian", "cb-rg", "cb-full")
])
def test_palette_rebuild_matches_initial(prefix, palette_key):
    """Value-list of rebuilt options on palette change matches the initial builder."""
    from layout.display_models import build_display_models_options
    flags = {
        "bub": {"include_bm_master": True},
        "dca": {},
        "ret": {"include_mc": True},
        "sc":  {},
    }[prefix]
    pal = _app_ctx.PALETTES.get(palette_key, _app_ctx.PALETTES["default"])
    mc  = pal.get("model_colors", _app_ctx.MODEL_TRACE_COLORS)
    a = [o["value"] for o in build_display_models_options(mc, prefix, **flags)]
    b = [o["value"] for o in build_display_models_options(mc, prefix, **flags)]
    assert a == b
    mc_default = _app_ctx.PALETTES["default"].get("model_colors", _app_ctx.MODEL_TRACE_COLORS)
    baseline = [o["value"] for o in build_display_models_options(mc_default, prefix, **flags)]
    assert a == baseline


@pytest.mark.parametrize("prefix,family",
    [(p, f) for p in ("bub", "dca", "ret", "sc", "hm")
            for f in ("lppl", "hybppl", "eppl")])
def test_no_mini_card_ids_anywhere(prefix, family):
    """Mini card activate ids must ONLY appear inside the defunct placeholder subtree."""
    import app  # noqa: F401 — registers Dash callbacks
    from layout import _serve_layout
    layout = _serve_layout()
    ids_outside_placeholders = _live_component_ids_excluding(
        layout, "_defunct-snapshot-placeholders"
    )
    activate_id = f"{prefix}-{family}-activate"
    assert activate_id not in ids_outside_placeholders, (
        f"{activate_id} leaked outside _defunct-snapshot-placeholders — "
        f"a real checklist is still emitting it"
    )
    assert f"{prefix}-{family}-configure-btn" not in _live_component_ids(layout)


@pytest.mark.parametrize("prefix,family",
    [(p, f) for p in ("bub", "dca", "ret", "sc")
            for f in ("lppl", "hybppl", "eppl")])
def test_inline_summary_spans_exist(prefix, family):
    """All 12 {prefix}-{family}-summary-inline spans must be emitted by the
    respective tab's content builder.

    After the universal lazy-tab-loading refactor, only the initial tab's
    content is present in _serve_layout() at layout build time — non-active
    tabs show a `Loading...` placeholder until the user visits them. So for
    bubble (default initial_tab), we assert the span appears in layout; for
    dca/ret/sc, we build their tab content directly and assert the span is
    present in THAT subtree. Same coverage, new invariant.
    """
    import app  # noqa: F401 — registers Dash callbacks
    span_id = f"{prefix}-{family}-summary-inline"

    if prefix == "bub":
        from layout import _serve_layout
        layout = _serve_layout()
        assert span_id in _live_component_ids(layout)
    else:
        from callbacks.routing import _build_tab_content
        tab_map = {"dca": "dca", "ret": "retire", "sc": "supercharge"}
        content = _build_tab_content(tab_map[prefix])
        assert span_id in _live_component_ids(content), (
            f"{span_id} not found in {tab_map[prefix]} tab content subtree"
        )


def test_heatmap_status_row_exists():
    """Heatmap has the 4 status-row ids in its tab content.

    Post-lazy-tab refactor: heatmap content is lazy-loaded unless heatmap
    is the initial tab, so we assert these IDs exist in the heatmap
    subtree directly (via _build_tab_content('heatmap')), not in the
    default /1 layout.
    """
    import app  # noqa: F401 — registers Dash callbacks
    from callbacks.routing import _build_tab_content
    hm = _build_tab_content("heatmap")
    ids = _live_component_ids(hm)
    for cid in ("hm-active-family-row", "hm-active-family-label",
                "hm-active-family-summary-inline", "hm-active-family-gear"):
        assert cid in ids, f"missing {cid} in heatmap content subtree"


def test_defunct_placeholders_unconditional():
    """Placeholder div is emitted and contains all 15 activate ids."""
    import app  # noqa: F401 — registers Dash callbacks
    from layout import _serve_layout
    layout = _serve_layout()
    ids = _live_component_ids(layout)
    for prefix in ("bub", "dca", "ret", "sc", "hm"):
        for family in ("lppl", "hybppl", "eppl"):
            assert f"{prefix}-{family}-activate" in ids, \
                f"placeholder {prefix}-{family}-activate missing"


def test_modal_open_callbacks_use_gear_inputs():
    """Static source check: modal-open callbacks no longer reference -configure-btn ids."""
    import pathlib
    charts_dir = pathlib.Path("btc_web/callbacks/charts")
    # charts used to be a single file; it's now a package — read every .py in it.
    src = "\n".join(p.read_text() for p in sorted(charts_dir.glob("*.py")))
    for prefix in ("bub", "dca", "ret", "sc", "hm"):
        for family in ("lppl", "hybppl", "eppl"):
            assert f"{prefix}-{family}-configure-btn" not in src, \
                f"stale {prefix}-{family}-configure-btn reference in charts/"


def test_palette_summary_not_stale():
    """Palette-rebuild path must bake current modal state into re-rendered labels.

    Regression guard for Risk #2 — the whole staleness-fix architecture.
    """
    from layout.display_models import build_display_models_options

    mc = _app_ctx.PALETTES["default"].get("model_colors", _app_ctx.MODEL_TRACE_COLORS)
    custom_summaries = {
        "lppl":   "CUSTOM_LPPL",
        "hybppl": "CUSTOM_HYB",
        "eppl":   "CUSTOM_EPPL",
    }
    opts = build_display_models_options(
        mc, "bub", include_bm_master=True, summaries=custom_summaries,
    )

    collected = {}
    def _visit(node):
        nid = getattr(node, "id", None)
        if isinstance(nid, str) and nid.endswith("-summary-inline"):
            collected[nid] = getattr(node, "children", None)
    for opt in opts:
        _walk_component(opt["label"], _visit)

    assert collected.get("bub-lppl-summary-inline")   == "CUSTOM_LPPL"
    assert collected.get("bub-hybppl-summary-inline") == "CUSTOM_HYB"
    assert collected.get("bub-eppl-summary-inline")   == "CUSTOM_EPPL"


def test_old_snapshot_link_decodes_cleanly():
    """A pre-refactor q3: link with *-activate keys decodes without error."""
    import json, gzip, base64
    from snapshot import _decode_snapshot, _SNAPSHOT_CONTROLS
    values = []
    for cid, prop in _SNAPSHOT_CONTROLS:
        if "-activate" in cid:
            values.append(["yes"])
        else:
            values.append(None)
    payload = [values, []]
    encoded = base64.urlsafe_b64encode(
        gzip.compress(json.dumps(payload, separators=(',', ':')).encode())
    ).decode()
    result = _decode_snapshot(encoded)
    assert result is not None
