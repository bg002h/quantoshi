"""Guard: no server callback uploads a component subtree to answer a boolean.

A Dash server callback ships the CURRENT VALUE of every Input and State it
declares from the browser to the server on each fire. `State("{tab}-lazy",
"children")` therefore uploads that tab's entire serialized component
subtree — measured at 1,315 KB for `bubble-lazy` once Tab 1's figure is in
it — purely so the callback can evaluate one boolean ("is this still the
'Loading...' placeholder?") and, 9 times out of 10, return `no_update`.

Measured before this guard (recon report
`docs/superpowers/agent-reports/2026-09-06-restore-burst-recon.md`):

* share-link restore: 5,096 KB uploaded, 4,399 KB (86 %) of it these blobs
* ONE tab switch: 14 POSTs, 2,294 KB up / 64 KB down, 13 POSTs returning
  nothing, 2,242 KB of that upload being these blobs — every switch, every
  session, every device

The fix is to answer the boolean with a boolean: a small
`dcc.Store("{tab}-loaded")` written in the SAME output batch as the
children. Same batch matters — the placeholder check is a real correctness
guard (a re-render clobbers user state), so the flag must never lag the
content it describes.

Why `GLOBAL_CALLBACK_MAP`: every server callback in this app is registered
with the module-level `@callback` decorator (52 decorator sites, 96
registrations; `grep -rn "app.callback(" btc_web` finds none), and those
land in `dash._callback.GLOBAL_CALLBACK_MAP`. `app.callback_map` holds the
`app.clientside_callback` registrations, which run in the browser and cost
no upload at all — `charts/_clientside.py` and
`routing.py::_register_first_render_bump` legitimately read
`{tab}-lazy.children` there and must NOT be touched.
"""
import os

import pytest

os.environ.setdefault("TESTING", "1")

_TABS = ("bubble", "heatmap", "dca", "retire", "supercharge",
         "citadel", "leverage", "stack", "model_info", "faq")


def _server_callbacks():
    """{output_key: entry} for every server-side (POST-costing) callback."""
    import app  # noqa: F401  — registers every callback module
    from dash import _callback as dash_callback
    return dash_callback.GLOBAL_CALLBACK_MAP


def _outputs(output_key):
    """[(component_id, property), ...] parsed from a callback_map key.

    Single output: `id.prop` or `id.prop@hash` (allow_duplicate).
    Multi output:  `..id1.prop1...id2.prop2..`
    """
    key = output_key.strip(".")
    parts = key.split("...") if "..." in key else [key]
    out = []
    for p in parts:
        p = p.strip(".")
        if not p or p.startswith("{"):
            continue  # pattern-matching dict id
        cid, _, prop = p.rpartition(".")
        out.append((cid, prop.split("@")[0]))
    return out


def _uploaded_refs():
    """[(output_key, role, id, prop), ...] over every Input/State value that
    a server callback makes the browser upload."""
    refs = []
    for key, entry in _server_callbacks().items():
        for role in ("inputs", "state"):
            for dep in entry.get(role) or []:
                cid = dep.get("id")
                if isinstance(cid, str):
                    refs.append((key, role, cid, dep.get("property")))
    return refs


def _callback_for_output(cid, prop, name=None):
    """(output_key, undecorated function) for the server callback writing
    `cid.prop`, or None. `entry["callback"]` is Dash's `add_context` wrapper,
    which needs an `outputs_list` kwarg; `__wrapped__` is the real function.

    `{tab}-lazy.children` has two writers (`_lazy_load` and `_pf`, which take
    different arguments), so pass `name` to pick one rather than relying on
    the order the two registration loops happen to run in."""
    for key, entry in _server_callbacks().items():
        if (cid, prop) in _outputs(key):
            fn = entry["callback"].__wrapped__
            if name is not None and fn.__name__ != name:
                continue
            return key, fn
    return None


# ── The guard ───────────────────────────────────────────────────────────────

def test_no_server_callback_uploads_a_lazy_tab_subtree():
    """`{tab}-lazy.children` is a whole tab's DOM. Never send it to answer a
    boolean — use the `{tab}-loaded` flag."""
    offenders = [(k, role, cid) for k, role, cid, prop in _uploaded_refs()
                 if cid.endswith("-lazy") and prop == "children"]
    assert offenders == [], (
        f"{len(offenders)} server callback(s) upload a tab subtree:\n" +
        "\n".join(f"  {role:6} {cid}.children  ->  {k[:60]}"
                  for k, role, cid in offenders))


def test_no_server_callback_uploads_the_auto_y_grid():
    """`auto-y-grid.data` is a ~146 KB pre-computed envelope grid, uploaded
    on every tab switch just to test `is not None`."""
    offenders = [(k, role) for k, role, cid, prop in _uploaded_refs()
                 if cid == "auto-y-grid" and prop == "data"]
    assert offenders == [], (
        "auto-y-grid.data is uploaded by: " +
        ", ".join(f"{role} of {k[:60]}" for k, role in offenders))


@pytest.mark.parametrize("tab", _TABS)
def test_lazy_children_and_loaded_flag_are_written_in_one_batch(tab):
    """The flag must not lag the content it describes: a callback that
    writes `{tab}-lazy.children` writes `{tab}-loaded.data` in the same
    output batch, so no interleaved fire can see 'loaded' out of step."""
    writers = [key for key in _server_callbacks()
               if (f"{tab}-lazy", "children") in _outputs(key)]
    assert writers, f"no server callback writes {tab}-lazy.children"
    for key in writers:
        assert (f"{tab}-loaded", "data") in _outputs(key), (
            f"{key[:70]} writes {tab}-lazy.children without "
            f"{tab}-loaded.data in the same batch")


def test_auto_y_grid_writer_also_writes_its_loaded_flag():
    hit = _callback_for_output("auto-y-grid", "data")
    assert hit is not None, "nothing writes auto-y-grid.data"
    key, _fn = hit
    assert ("auto-y-grid-loaded", "data") in _outputs(key), (
        "auto-y-grid.data must be written with auto-y-grid-loaded.data "
        "in the same batch")


# ── Behaviour the flag has to preserve ──────────────────────────────────────

def test_lazy_load_skips_when_flag_says_already_loaded():
    """The placeholder check is a correctness guard: re-rendering a tab the
    user has interacted with clobbers their control state."""
    from dash import no_update
    hit = _callback_for_output("bubble-lazy", "children", "_lazy_load")
    assert hit is not None
    _key, fn = hit
    result = fn("bubble", True)
    assert all(r is no_update for r in result), (
        "lazy load must no-op when the tab is already loaded")


def test_lazy_load_populates_and_raises_the_flag_when_not_loaded():
    hit = _callback_for_output("bubble-lazy", "children", "_lazy_load")
    _key, fn = hit
    children, loaded = fn("bubble", False)
    assert children is not None
    assert loaded is True


def test_lazy_load_ignores_other_tabs():
    from dash import no_update
    _key, fn = _callback_for_output("bubble-lazy", "children", "_lazy_load")
    result = fn("heatmap", False)
    assert all(r is no_update for r in result)


# ── Initial state ───────────────────────────────────────────────────────────

def _stores_by_id(node, out):
    if node is None:
        return
    if isinstance(node, (list, tuple)):
        for x in node:
            _stores_by_id(x, out)
        return
    cid = getattr(node, "id", None)
    if isinstance(cid, str):
        out[cid] = node
    for prop in ("children",):
        child = getattr(node, prop, None)
        if child is not None:
            _stores_by_id(child, out)


@pytest.mark.parametrize("initial", ["heatmap", "bubble"])
def test_eagerly_rendered_tab_starts_loaded_and_others_do_not(initial):
    """The initial tab's content is rendered into the layout directly, so its
    flag must start True — otherwise the first revisit rebuilds it and
    clobbers whatever the user changed.

    The bubble case also pins the auto-Y invariant: `auto-y-grid.data` is
    pre-populated only for an initial bubble load, and its flag must agree
    with that in both directions."""
    import app  # noqa: F401
    from layout import _build_layout
    stores = {}
    _stores_by_id(_build_layout(initial), stores)
    assert stores[f"{initial}-loaded"].data is True
    for tab in _TABS:
        if tab != initial:
            assert stores[f"{tab}-loaded"].data is False, tab
    assert stores["auto-y-grid-loaded"].data is (initial == "bubble")
    assert (stores["auto-y-grid-loaded"].data
            == (stores["auto-y-grid"].data is not None))
