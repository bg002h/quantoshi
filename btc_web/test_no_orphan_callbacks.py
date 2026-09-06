"""Regression guard: every Input/Output/State in a registered callback
must reference a component that actually exists in the layout.

Rationale: see docs + memory `feedback_nonexistent_input_perf.md`.
Orphan refs bloat /_dash-dependencies, cost round-trips, and drown real
errors in console noise. The island-model discipline is that each tab
wires its own controls; cross-tab wires must be deliberate, not
accidental.

How it works:
  1. Build every tab's content by calling its top-level layout function
     directly (bypassing the lazy {tab}-lazy placeholder).
  2. Build the shell layout for navbar + stores + modals.
  3. Walk the trees collecting every component that has an `id`.
  4. Walk BOTH callback registries (`_all_callbacks()`: the server
     `@callback`s in `dash._callback.GLOBAL_CALLBACK_MAP` plus the
     clientside ones in `app.callback_map`) collecting every id referenced
     as an Input / Output / State. Walking only `app.callback_map` — as
     this test did until 2026-09-06 — misses every server callback, which
     measured as 299 component ids over 2,040 reference sites, 65% of all
     references.
  5. Diff. Anything referenced but not in the layout is an orphan.

Allowlist (`_KNOWN_ORPHANS`) documents pre-existing orphans with a
one-line reason. New additions should FIX the orphan rather than grow
the list. Shrink the list opportunistically; it is the ratchet.
"""
import os
import re

import pytest

os.environ.setdefault("TESTING", "1")


# Pre-existing orphans grandfathered in on first green run.
# Each entry: "component-id": "reason — remove when fixed"
# To regenerate: run this test with `--collect-orphans` (see bottom).
_KNOWN_ORPHANS: dict[str, str] = {
    # Restore button lives inside a banner that only renders when a
    # shared link injects snapshot-lots; static layout walk misses it.
    "restore-lots-btn": "banner button — only rendered when snapshot-lots is populated",
}


def _walk_ids(node, out):
    """Recursively collect every component id from a Dash layout tree.

    Descends into `children` AND into any other component-valued prop
    (e.g. `dcc.Checklist.options[i].label` is a common carrier of nested
    Dash components).
    """
    if node is None:
        return
    if isinstance(node, (list, tuple)):
        for x in node:
            _walk_ids(x, out)
        return
    if isinstance(node, dict):
        for v in node.values():
            _walk_ids(v, out)
        return
    cid = getattr(node, "id", None)
    if isinstance(cid, str):
        out.add(cid)
    # Walk every prop that might carry components. Using _prop_names
    # (Dash components expose this) keeps us O(declared props) instead
    # of iterating __dict__.
    props = getattr(node, "_prop_names", None)
    if props:
        for p in props:
            if p == "id":
                continue
            try:
                val = getattr(node, p)
            except AttributeError:
                continue
            _walk_ids(val, out)


def _collect_layout_ids() -> set[str]:
    """Build every tab's content + the shell, return union of component ids."""
    from layout import _build_layout
    from layout.bubble import _bubble_tab
    from layout.heatmap import _heatmap_tab
    from layout.sim_tabs import _dca_tab, _retire_tab
    from layout.supercharge import _supercharge_tab
    from layout.stack import _stack_tracker_tab
    from layout.model_info import _model_info_tab
    from layout.citadel import _citadel_tab
    from layout.faq import _faq_tab
    from layout.leverage import _leverage_tab

    ids: set[str] = set()
    # Shell (navbar, stores, modals, intervals, tabs container)
    _walk_ids(_build_layout("bubble"), ids)
    # Each tab's content, force-rendered
    for fn in (_bubble_tab, _heatmap_tab, _dca_tab, _retire_tab,
               _supercharge_tab, _citadel_tab, _leverage_tab,
               _stack_tracker_tab, _faq_tab):
        _walk_ids(fn(), ids)
    _walk_ids(_model_info_tab(), ids)
    return ids


def _all_callbacks():
    """Every registered callback, keyed by output key.

    Dash keeps two disjoint registries and tests need both:

    * ``dash._callback.GLOBAL_CALLBACK_MAP`` — the server callbacks, i.e.
      every ``@callback`` in this app (`grep -rn "app.callback(" btc_web`
      finds none). These are the ones that cost a POST.
    * ``app.callback_map`` — the ``app.clientside_callback`` registrations.

    Which map holds what depends on test order, so always take the union.
    Dash merges the global map into ``app.callback_map`` during server setup,
    and one test in the suite triggers exactly that:
    ``test_infrastructure.py::TestSourceMapGuard`` builds a Flask test client.
    Measured 2026-09-06, before and after that client is created::

        before   global= 96   clientside=251   union=347
        after    global=  0   clientside=347   union=347

    The merge is lossless, so the union is stable either way — but anything
    asserting on one registry alone passes or fails by xdist worker
    assignment. Do NOT force the merge to unify them either:
    ``Dash._setup_server()`` ``pop()``s the global map, and
    ``test_callbacks.py::test_restore_from_url_does_not_output_bubble_graph``
    reads it directly.
    """
    import _app_ctx
    from dash import _callback as dash_callback
    return {**dash_callback.GLOBAL_CALLBACK_MAP, **_app_ctx.app.callback_map}


def _split_output_key(key):
    """``['id.prop@hash', ...]`` — the output entries of one callback key.

    Single output is ``id.prop`` or ``id.prop@hash`` (allow_duplicate);
    multi-output is ``..id1.prop1...id2.prop2..``. A naive
    ``key.split("...")`` leaves the wrapping ``..`` on the first and last
    entry, so ``'..main-tabs.active_tab'`` never equals
    ``'main-tabs.active_tab'``. Measured 2026-09-06: that mis-parsed 222
    output entries across 154 multi-output callbacks and silently emptied
    three tests. See
    ``docs/superpowers/agent-reports/2026-09-06-orphan-guard-and-syntax-check-recon.md``.
    """
    key = key.strip(".")
    parts = re.split(r"\.\.+", key) if ".." in key else [key]
    return [p for p in (x.strip(".") for x in parts) if p]


_OUTPUT_RE = re.compile(r"([^.]+)\.[^.]+")


def _parse_output_key(key: str) -> list[str]:
    """Outputs are `id.prop` or `..id1.prop1...id2.prop2..` for multi.
    Skip pattern-matching dict ids (they start with `{`)."""
    key = key.strip(".")
    parts = re.split(r"\.\.+", key) if ".." in f".{key}." else [key]
    out = []
    for p in parts:
        if p.startswith("{"):
            continue  # dict-id pattern-matching output
        m = _OUTPUT_RE.match(p)
        if m:
            out.append(m.group(1))
    return out


def _collect_callback_refs() -> dict[str, list[tuple[str, str, str]]]:
    """Returns {component_id: [(callback_key, role, property), ...]}."""
    refs: dict[str, list[tuple[str, str, str]]] = {}
    for key, entry in _all_callbacks().items():
        for cid in _parse_output_key(key):
            refs.setdefault(cid, []).append((key, "Output", "-"))
        for dep in entry.get("inputs", []):
            cid = dep.get("id") if isinstance(dep, dict) else None
            prop = dep.get("property") if isinstance(dep, dict) else "-"
            if isinstance(cid, str) and not cid.startswith("{"):
                refs.setdefault(cid, []).append((key, "Input", prop))
        for dep in entry.get("state", []):
            cid = dep.get("id") if isinstance(dep, dict) else None
            prop = dep.get("property") if isinstance(dep, dict) else "-"
            if isinstance(cid, str) and not cid.startswith("{"):
                refs.setdefault(cid, []).append((key, "State", prop))
    return refs


def test_introspection_sees_both_callback_registries():
    """Non-vacuity guard for every callback-introspecting test in the suite.

    `_all_callbacks()` and `_split_output_key()` are shared by this module,
    `test_callbacks.py` and `test_snapshot.py`. If either silently reverts to
    the clientside-only map or to a naive `key.split("...")`, several of those
    tests stop inspecting anything and pass for free — which is exactly what
    happened between 2026-04 and 2026-09-06 (three tests fully vacuous, two
    partially). Fail loudly here instead.
    """
    union = _all_callbacks()
    outputs = {p.split("@")[0] for k in union for p in _split_output_key(k)}

    # Assert the PROPERTY (server callbacks are visible), not the mechanism
    # (which map they are in). Which map holds them depends on test order —
    # see _all_callbacks() — so asserting on either registry directly makes
    # this test pass or fail according to xdist worker assignment.
    for probe in ("loaded-hash-store.data",      # restore_from_url
                  "snapshot-pending.data",       # restore_from_url + apply_tab_*
                  "bubble-graph.figure"):        # update_bubble
        assert probe in outputs, (
            f"{probe} is not visible to _all_callbacks() — server callbacks "
            f"are being missed ({len(union)} callbacks seen)")

    # And the parser has to strip the wrapping dots off multi-output keys.
    multi = [k for k in union if "..." in k]
    assert multi, "expected at least one multi-output callback"
    for key in multi:
        for part in _split_output_key(key):
            assert not part.startswith("."), (
                f"_split_output_key left a leading dot on {part!r} "
                f"(from {key[:60]!r})")


def test_no_orphan_callback_refs():
    """Every callback Input/Output/State id exists in the full layout."""
    layout_ids = _collect_layout_ids()
    refs = _collect_callback_refs()

    orphans = {cid: sites for cid, sites in refs.items()
               if cid not in layout_ids and cid not in _KNOWN_ORPHANS}

    if orphans:
        lines = ["Orphan callback references (id not in layout):"]
        for cid in sorted(orphans):
            sites = orphans[cid]
            lines.append(f"  {cid}  ({len(sites)} refs)")
            for key, role, prop in sites[:3]:
                short = key if len(key) < 80 else key[:77] + "..."
                lines.append(f"    {role:<6} {prop:<20} <- {short}")
        lines.append("")
        lines.append("Either fix the orphan (preferred) or add to "
                     "_KNOWN_ORPHANS in test_no_orphan_callbacks.py "
                     "with a reason.")
        pytest.fail("\n".join(lines))


def test_known_orphans_still_orphaned():
    """If a grandfathered orphan has been fixed, remove it from the
    allowlist so the test ratchets down."""
    layout_ids = _collect_layout_ids()
    refs = _collect_callback_refs()
    stale = [cid for cid in _KNOWN_ORPHANS
             if cid in layout_ids or cid not in refs]
    if stale:
        pytest.fail(
            "These ids are in _KNOWN_ORPHANS but are no longer orphaned:\n  "
            + "\n  ".join(stale)
            + "\n\nRemove them from _KNOWN_ORPHANS."
        )


if __name__ == "__main__":
    # `python test_no_orphan_callbacks.py` prints a seed allowlist
    # suitable for pasting into `_KNOWN_ORPHANS`.
    layout_ids = _collect_layout_ids()
    refs = _collect_callback_refs()
    orphans = {cid: sites for cid, sites in refs.items() if cid not in layout_ids}
    print(f"# {len(orphans)} orphan ids across {sum(len(s) for s in orphans.values())} refs")
    for cid in sorted(orphans):
        roles = {r for _, r, _ in orphans[cid]}
        print(f'    "{cid}": "{"/".join(sorted(roles))} ref in '
              f'{len(orphans[cid])} callback(s) — TODO",')
