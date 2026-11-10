"""Tab 1: applying a price-regime filter turns "Show individual price paths" on.

Operator request 2026-11-10: the filter's visible effect is the best-matching
paths, so when a user narrows the regimes the paths box ticks itself.

Rules (each a test below):
  * only a REAL click on a regime box does it -- a share-link restore or any
    other programmatic write of bub-mc-regime never touches the paths box
    (a link saved with a filter and paths off must restore that way);
  * only when the filter BECOMES effective (0 < checked < bins, as the server
    reads it) -- so a user who unticks paths while filtered is not overridden
    by their next regime click;
  * it never unticks anything (clearing the filter leaves paths alone).

Same click-stamp pattern as the MC auto-range (test_mc_autowiden.py): an asset
stamps real clicks, the clientside callback consumes the stamp. The JS runs
under node here; browser wiring is test_mc_regime_paths_e2e.py.
"""
import json
import pathlib
import shutil
import subprocess

import pytest

from conftest import _app_ctx  # noqa: F401  (registers every callback)

_APP = _app_ctx.app   # bound at collection: see test_mc_autowiden.py
_ASSET = pathlib.Path(__file__).parent / "assets" / "mc_regime_paths.js"
_NODE = shutil.which("node")
needs_node = pytest.mark.skipif(_NODE is None, reason="node not installed")


def _entry():
    hits = [e for e in _APP._callback_list
            if "bub-mc-paths.value" in str(e["output"])
            and [i.get("id") for i in e["inputs"]] == ["bub-mc-regime"]]
    assert len(hits) == 1, f"expected one regime -> paths callback, got {len(hits)}"
    return hits[0]


class TestRegistration:
    def test_single_input_is_the_regime_checklist(self):
        # Multiple Inputs + allow_duplicate + prevent_initial_call silently
        # no-ops in Dash 4.0 (callbacks/plot_appearance.py:22-28).
        assert [(i["id"], i["property"]) for i in _entry()["inputs"]] == [
            ("bub-mc-regime", "value")]

    def test_bins_and_current_paths_are_state(self):
        assert [(s["id"], s["property"]) for s in _entry()["state"]] == [
            ("bub-mc-bins", "value"), ("bub-mc-paths", "value")]

    def test_paths_output_is_not_allow_duplicate(self):
        """Review I-2: an allow_duplicate Output makes Dash fire update_bubble
        for the regime change AND again for the paths write (2 POSTs per tick
        click). bub-mc-paths.value has no primary writer, so this is it."""
        assert _entry()["output"] == "bub-mc-paths.value"

    def test_never_fires_on_page_load(self):
        assert _entry()["prevent_initial_call"] is True

    def test_is_clientside(self):
        from layout.mc_controls import MC_REGIME_PATHS_JS
        assert _entry()["clientside_function"]
        assert MC_REGIME_PATHS_JS.strip() in "".join(_APP._inline_scripts)

    def test_asset_and_callback_share_the_click_stamp(self):
        from layout.mc_controls import MC_REGIME_PATHS_JS
        src = _ASSET.read_text()
        assert "#bub-mc-regime" in src
        assert "__qsMcRegimeClickAt" in src
        assert "__qsMcRegimeClickAt" in MC_REGIME_PATHS_JS


# ── the decision logic, executed ────────────────────────────────────────────

_HARNESS = r"""
const NU = '__NU__';
let now = 1000000;
Date.now = () => now;
globalThis.window = {dash_clientside: {no_update: NU}};
const fn = (%(fn)s);
const out = [];
for (const c of %(cases)s) {
    if (c.reset) { delete window.__qsRegimeK; window.__qsMcRegimeClickAt = 0; }
    if (c.clickAgoMs === null) window.__qsMcRegimeClickAt = 0;
    else if (c.clickAgoMs !== undefined) window.__qsMcRegimeClickAt = now - c.clickAgoMs;
    out.push(fn(c.checked, c.bins, c.paths));
}
console.log(JSON.stringify(out));
"""

NU = "__NU__"
ALL5, SOME, NONE = [0, 1, 2, 3, 4], [0, 1], []
ON, OFF = ["yes"], []


def _run(cases):
    from layout.mc_controls import MC_REGIME_PATHS_JS
    script = _HARNESS % {"fn": MC_REGIME_PATHS_JS, "cases": json.dumps(cases)}
    res = subprocess.run([_NODE, "-e", script], capture_output=True,
                         text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _c(checked, paths=OFF, bins=5, click=5, reset=False):
    d = dict(checked=checked, bins=bins, paths=paths, reset=reset)
    if click is not None:
        d["clickAgoMs"] = click
    return d


@needs_node
class TestDecision:
    def test_a_click_that_applies_the_filter_ticks_paths(self):
        assert _run([_c(SOME, reset=True)]) == [["yes"]]

    def test_no_click_means_no_change(self):
        """Restore / set_props: no stamp, so the link's own paths value stands."""
        assert _run([_c(SOME, click=None, reset=True)]) == [NU]

    def test_a_stale_click_means_no_change(self):
        assert _run([_c(SOME, click=5000, reset=True)]) == [NU]

    def test_already_on_is_left_alone(self):
        assert _run([_c(SOME, paths=ON, reset=True)]) == [NU]

    @pytest.mark.parametrize("checked,bins", [(ALL5, 5), (NONE, 5), (None, 5),
                                              (SOME, None), (list(range(7)), 7)])
    def test_no_effective_filter_means_no_change(self, checked, bins):
        """All checked, none checked (the server treats it as all allowed),
        or a missing value: nothing is being filtered."""
        assert _run([_c(checked, bins=bins, reset=True)]) == [NU]

    def test_bins_may_arrive_as_a_string(self):
        assert _run([_c(SOME, bins="5", reset=True)]) == [["yes"]]

    def test_narrowing_further_does_not_override_a_manual_untick(self):
        """Filter on (paths auto-ticked) -> user unticks paths -> clicks another
        regime box: the filter was already on, so paths stay as the user left
        them."""
        r = _run([_c(SOME, reset=True),          # applies the filter: ticks
                  _c([0], paths=OFF)])           # narrows; user had unticked
        assert r == [["yes"], NU]

    def test_clearing_then_reapplying_ticks_again(self):
        r = _run([_c(SOME, reset=True), _c(ALL5, paths=ON), _c(SOME, paths=OFF)])
        assert r == [["yes"], NU, ["yes"]]

    def test_clearing_the_filter_never_unticks(self):
        assert _run([_c(SOME, reset=True), _c(ALL5, paths=ON)])[1] == NU

    def test_a_restored_filter_is_remembered_so_the_next_click_is_not_a_transition(self):
        """A link restores with the filter on and paths off (no click). The
        user then clicks a regime box: the filter was already on, so no tick."""
        r = _run([_c(SOME, click=None, reset=True),    # restore: records 'on'
                  _c([0], paths=OFF)])                 # user click, still on
        assert r == [NU, NU]

    def test_lowering_bins_does_not_leave_a_stale_filter_on(self):
        """Review I-1(a). At bins 7 the user checks every box (no filter),
        unchecks one (a true transition: ticks), then lowers bins to 6. The
        callback does not run for the bins change, yet six checked boxes of six
        is no filter now -- so unchecking one is a GENUINE first filter click
        and must tick. (A tracker that kept the last verdict 'on' misses it.)"""
        r = _run([_c(list(range(7)), bins=7, reset=True),        # all checked
                  _c(list(range(6)), bins=7, paths=OFF),         # uncheck one
                  _c([0, 1, 2, 4, 5], bins=6, paths=OFF)])       # bins now 6
        assert r == [NU, ["yes"], ["yes"]]

    def test_raising_bins_makes_the_default_value_a_filter_already(self):
        """Review I-1(b). bins 5 -> 6 leaves the default value [0..4], so the
        server already filters (and the note shows) before any click. The
        first regime click is not a transition: no tick over a manual untick."""
        assert _run([_c([1, 2, 3, 4], bins=6, paths=OFF, reset=True)]) == [NU]

    def test_default_flow_still_ticks_on_the_first_unchecked_box(self):
        assert _run([_c([1, 2, 3, 4], bins=5, paths=OFF, reset=True)]) == [["yes"]]

    def test_the_stamp_is_consumed(self):
        """One click, one chance: a second callback run on the same stamp
        (e.g. the bins callback re-writing options) does nothing."""
        r = _run([_c(SOME, reset=True), _c(ALL5, paths=ON, click=None),
                  _c(SOME, paths=OFF, click=None)])
        assert r == [["yes"], NU, NU]


def test_asset_listens_in_the_capture_phase_on_the_regime_checklist():
    src = _ASSET.read_text()
    assert "addEventListener('click'" in src
    assert "}, true)" in src            # capture phase: runs before React
    assert "Date.now()" in src
