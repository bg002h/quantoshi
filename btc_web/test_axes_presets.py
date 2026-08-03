"""One-tap axes presets (Tab 1) — registry, markup, callback registration.

Spec: docs/superpowers/specs/2026-08-03-axes-presets-design.md
"""
# Importing conftest imports app, which imports callbacks/__init__.py and so
# registers every callback. Do NOT `import callbacks.axes_presets` directly:
# that would register the callbacks itself, so the registration test in Task 2
# would pass even when the import line in callbacks/__init__.py is missing --
# the exact failure it exists to catch (spec section 9).
from conftest import _app_ctx  # noqa: F401

from layout.bubble import (AXES_CONTROL_IDS, AXES_DEFAULTS, AXES_PRESETS,
                           _bubble_controls)
from snapshot_defaults import SNAPSHOT_DEFAULTS


def _walk(node):
    """Yield every Dash component in a layout tree."""
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _walk(child)


def _string_ids(component):
    """Every string id in a layout tree.

    Filters to str deliberately: _mc_controls() embeds Dash pattern-matching
    ids, which are dicts -- {"type": "mc-run-btn", "tab": "bub"} and
    {"type": "mc-run-status", "tab": "bub"} (layout/mc_controls.py:127-128).
    dicts are unhashable, so collecting ids into a set without this filter
    raises TypeError: cannot use 'dict' as a set element.
    """
    return {c.id for c in _walk(component)
            if isinstance(getattr(c, "id", None), str)}


class TestAxesPresetRegistry:
    def test_keys_unique(self):
        keys = [p["key"] for p in AXES_PRESETS]
        assert len(keys) == len(set(keys))

    def test_every_entry_is_complete(self):
        assert AXES_PRESETS, "registry must not be empty"
        for p in AXES_PRESETS:
            assert p["key"].strip(), p
            assert p["label"].strip(), p
            assert p["js"].strip(), p
            assert isinstance(p["states"], tuple), p


class TestAxesPresetMarkup:
    def test_button_rendered_for_every_preset(self):
        ids = _string_ids(_bubble_controls())
        for p in AXES_PRESETS:
            assert f"bub-axes-preset-{p['key']}" in ids

    def test_preset_row_present(self):
        assert "bub-axes-presets" in _string_ids(_bubble_controls())


class TestAxesBakedDefaults:
    def test_defaults_match_snapshot_defaults_ssot(self):
        assert AXES_DEFAULTS
        for key, value in AXES_DEFAULTS.items():
            assert SNAPSHOT_DEFAULTS[key] == value, key

    def test_every_control_id_has_a_default(self):
        assert len(AXES_CONTROL_IDS) == 5
        for cid in AXES_CONTROL_IDS:
            assert f"{cid}:value" in SNAPSHOT_DEFAULTS


class TestAxesPresetCallbacks:
    """Guards the two ways this feature fails silently.

    app.clientside_callback populates app.callback_map at import time --
    verified empirically against Dash 4.0.0. NOTE: the comment at
    test_callbacks.py:1812-1815 ("callback_map is only populated after
    app.run()") is true only for SERVER @callback registrations, which sit in
    dash._callback.GLOBAL_CALLBACK_MAP until the _setup_server merge.
    App-method clientside registrations never appear there.
    """

    def _preset_entries(self):
        found = {}
        for entry in _app_ctx.app.callback_map.values():
            ids = [i.get("id") for i in entry["inputs"]]
            for p in AXES_PRESETS:
                if f"bub-axes-preset-{p['key']}" in ids:
                    found[p["key"]] = entry
        return found

    def test_callback_registered_for_every_preset(self):
        found = self._preset_entries()
        missing = [p["key"] for p in AXES_PRESETS if p["key"] not in found]
        assert not missing, (
            f"no callback registered for {missing}. Is "
            "`import callbacks.axes_presets` present in callbacks/__init__.py?")

    def test_each_preset_callback_has_exactly_one_input(self):
        # Multiple Inputs + allow_duplicate + prevent_initial_call silently
        # no-ops in Dash 4.0 (plot_appearance.py:22-28). Never merge these.
        found = self._preset_entries()
        assert set(found) == {p["key"] for p in AXES_PRESETS}, (
            "callback registration is broken -- this test would otherwise "
            "pass vacuously by iterating an empty dict")
        for key, entry in found.items():
            assert len(entry["inputs"]) == 1, (
                f"{key} has {len(entry['inputs'])} Inputs; must be exactly 1")

    def test_every_preset_writes_all_five_axis_controls(self):
        # entry["output"] is a list of Output objects (verified against Dash
        # 4.0.0). No isinstance guard -- a guard that skips on an unexpected
        # shape would turn this into a test that passes without asserting.
        found = self._preset_entries()
        assert set(found) == {p["key"] for p in AXES_PRESETS}, (
            "callback registration is broken -- this test would otherwise "
            "pass vacuously by iterating an empty dict")
        for key, entry in found.items():
            out_ids = {o.component_id for o in entry["output"]}
            assert out_ids == set(AXES_CONTROL_IDS), key


class TestCagrDefaultXrange:
    """[2025, 2050] must have exactly one definition.

    It appears in four places that must agree: the price->CAGR swap and the two
    swap-BACK comparisons in toggle_bub_view, plus the /1.2 deep-link handler in
    routing.py. The swap-back tests exact equality, so a diverged copy silently
    stops CAGR view from restoring [2010, 2033] when you switch back to price.
    """

    def test_no_module_still_hardcodes_the_literal(self):
        import pathlib
        import callbacks.charts as _charts
        import callbacks.routing as _routing

        for mod in (_charts, _routing):
            src = pathlib.Path(mod.__file__).read_text()
            assert "2025, 2050" not in src, (
                f"{mod.__name__} still hardcodes [2025, 2050]. Import "
                "CAGR_DEFAULT_XRANGE from layout.bubble instead -- a second "
                "copy breaks the CAGR<->price swap, which compares for "
                "exact equality.")

    def test_constant_is_a_list_not_a_tuple(self):
        from layout.bubble import CAGR_DEFAULT_XRANGE
        # Compared against JSON-decoded slider values:
        # [2025, 2050] == (2025, 2050) is False, which would break both
        # swap directions silently.
        assert isinstance(CAGR_DEFAULT_XRANGE, list)
        assert CAGR_DEFAULT_XRANGE == [2025, 2050]
