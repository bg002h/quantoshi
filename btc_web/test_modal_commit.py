"""Regression guard for the HybPPL/EPPL modal-close commit-trigger refactor.

Background: the 13 HybPPL + 13 EPPL config radios used to be Inputs on the
bubble / DCA / retire / supercharge chart callbacks, so every click on a
radio re-fired the chart while the user was still browsing options inside
the gear modal. They were demoted to State; two clientside counters
(`hybppl-commit-trigger`, `eppl-commit-trigger`) bump on modal
`is_open=False`, waking each chart once per modal close.

Why not Playwright: the gear icon is an html.Span nested inside a dcc.Checklist
option label. Clicking the span inside a label fights with Dash's n_clicks
propagation and dbc.Modal's lazy-render-on-open behavior in an
environment-dependent way. This static test catches the real regression
risks — Input→State drift and missing commit-trigger wiring — without any
browser.
"""
import ast
import pathlib

import pytest

_CHARTS_PKG = pathlib.Path(__file__).parent / "callbacks" / "charts"
_CLIENTSIDE = _CHARTS_PKG / "_clientside.py"
_INIT       = _CHARTS_PKG / "__init__.py"
_LAYOUT     = pathlib.Path(__file__).parent / "layout" / "__init__.py"


# The 26 radio IDs that must be State (not Input) on every chart callback.
_HYBPPL_IDS = {f"hybppl-cfg-{s}-{field}"
               for s in ("a", "b")
               for field in ("nlog", "ncal", "log1d", "log2d", "cal1d", "cal2d")} | {
    "hybppl-cfg-b-enabled"
}
_EPPL_IDS = {f"eppl-cfg-{s}-{field}"
             for s in ("a", "b")
             for field in ("nlog", "ncal", "log1d", "log2d", "cal1d", "cal2d")} | {
    "eppl-cfg-b-enabled"
}
_CFG_IDS = _HYBPPL_IDS | _EPPL_IDS  # 26 ids total


def _decorator_calls(src_path: pathlib.Path, callback_name: str):
    """Return the list of ast.Call nodes inside the @callback decorator attached
    to the function named `callback_name`.
    """
    tree = ast.parse(src_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == callback_name:
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                func = dec.func
                fname = func.id if isinstance(func, ast.Name) else (
                    func.attr if isinstance(func, ast.Attribute) else None
                )
                if fname == "callback":
                    return dec.args
    raise AssertionError(f"@callback for {callback_name!r} not found in {src_path}")


def _split_io(args):
    """Split a list of ast.Call args into (Output, Input, State) buckets, keyed
    by the first positional arg's string value (the component id).
    """
    out, inp, st = [], [], []
    for a in args:
        if not isinstance(a, ast.Call):
            continue
        name = a.func.id if isinstance(a.func, ast.Name) else None
        if not name or not a.args:
            continue
        first = a.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        cid = first.value
        if name == "Output":
            out.append(cid)
        elif name == "Input":
            inp.append(cid)
        elif name == "State":
            st.append(cid)
    return out, inp, st


# --- the 4 chart callbacks whose radios must NOT be Inputs -------------------

@pytest.mark.parametrize("cb_name", [
    "update_bubble", "update_dca", "update_retire", "update_supercharge",
])
def test_cfg_radios_are_state_not_input(cb_name):
    """The 13 hybppl-cfg-* + 13 eppl-cfg-* ids must appear as State, never Input.

    Fails if anyone re-promotes a radio to Input — which would restore the
    every-click-refires-chart behavior the refactor eliminated.
    """
    _, inputs, states = _split_io(_decorator_calls(_INIT, cb_name))
    input_set = set(inputs)
    state_set = set(states)

    leaked = _CFG_IDS & input_set
    assert not leaked, (
        f"{cb_name}: these radio ids are Input but must be State "
        f"(would re-fire chart on every click): {sorted(leaked)}"
    )

    missing_state = _CFG_IDS - state_set
    assert not missing_state, (
        f"{cb_name}: these radio ids are missing from the State list "
        f"(callback won't see the radio values): {sorted(missing_state)}"
    )


@pytest.mark.parametrize("cb_name", [
    "update_bubble", "update_dca", "update_retire", "update_supercharge",
])
def test_commit_triggers_are_inputs(cb_name):
    """Both commit-trigger stores must be Inputs so modal-close wakes the chart."""
    _, inputs, _ = _split_io(_decorator_calls(_INIT, cb_name))
    input_set = set(inputs)
    for trig in ("hybppl-commit-trigger", "eppl-commit-trigger"):
        assert trig in input_set, (
            f"{cb_name}: '{trig}' is missing from Inputs — modal close will "
            f"no longer re-render the chart."
        )


# --- the two clientside callbacks that bump the counters ---------------------

def test_clientside_has_commit_trigger_callbacks():
    """_clientside.py must register the two counter-bump callbacks.

    Can't easily introspect clientside_callback args via AST (they're
    registered in a side-effect import), so we do a source-string check.
    """
    src = _CLIENTSIDE.read_text()
    # Output must be the trigger store's data.
    assert 'Output("hybppl-commit-trigger", "data")' in src, (
        "Missing clientside callback with Output(hybppl-commit-trigger, data)"
    )
    assert 'Output("eppl-commit-trigger", "data")' in src, (
        "Missing clientside callback with Output(eppl-commit-trigger, data)"
    )
    # Input must be the modal is_open property.
    assert 'Input("hybppl-config-modal", "is_open")' in src, (
        "Clientside must watch hybppl-config-modal.is_open"
    )
    assert 'Input("eppl-config-modal", "is_open")' in src, (
        "Clientside must watch eppl-config-modal.is_open"
    )
    # The JS must bump only on is_open=false, not on every change.
    assert "is_open === false" in src, (
        "Commit trigger JS must guard on `is_open === false` so opens don't "
        "fire the chart."
    )


def test_trigger_stores_declared_in_layout():
    """The two Stores must exist in _serve_layout or they won't be in the DOM."""
    src = _LAYOUT.read_text()
    assert 'dcc.Store(id="hybppl-commit-trigger"' in src
    assert 'dcc.Store(id="eppl-commit-trigger"' in src


# --- sanity: the heatmap callback already uses State and stays that way ------

def test_heatmap_uses_state_for_cfg_radios():
    """Regression safety net — update_heatmap should have had these as State
    from the start, and must keep them that way (its Input trigger is the
    pill bar's hm-active-model store, not these radios).
    """
    _, inputs, states = _split_io(_decorator_calls(_INIT, "update_heatmap"))
    input_set, state_set = set(inputs), set(states)
    # For heatmap, only the "a" side of hybppl and eppl is used.
    heatmap_cfg = {
        "hybppl-cfg-a-nlog", "hybppl-cfg-a-ncal", "hybppl-cfg-a-log1d",
        "hybppl-cfg-a-log2d", "hybppl-cfg-a-cal1d", "hybppl-cfg-a-cal2d",
        "eppl-cfg-a-nlog", "eppl-cfg-a-ncal", "eppl-cfg-a-log1d",
        "eppl-cfg-a-log2d", "eppl-cfg-a-cal1d", "eppl-cfg-a-cal2d",
    }
    leaked = heatmap_cfg & input_set
    assert not leaked, (
        f"update_heatmap: heatmap cfg radios should stay as State, "
        f"these are Input: {sorted(leaked)}"
    )
    missing = heatmap_cfg - state_set
    assert not missing, (
        f"update_heatmap: heatmap cfg radios missing from State: {sorted(missing)}"
    )
