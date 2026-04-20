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
  4. Walk `app.callback_map` collecting every id referenced as an
     Input / Output / State.
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
    # Display-Models consolidation (2026-04-11) removed per-tab inline
    # summary widgets and gear icons, but leftover callbacks still
    # reference them. Harmless (the refs never fire) but bloat
    # /_dash-dependencies; remove when the callbacks are cleaned up.
    "bub-bm-gear":              "defunct gear Input — DM consolidation",
    "bub-lppl-gear":             "defunct gear Input — DM consolidation",
    "bub-hybppl-gear":           "defunct gear Input — DM consolidation",
    "bub-eppl-gear":             "defunct gear Input — DM consolidation",
    "bub-lppl-summary-inline":   "defunct summary Output — DM consolidation",
    "bub-hybppl-summary-inline": "defunct summary Output — DM consolidation",
    "bub-eppl-summary-inline":   "defunct summary Output — DM consolidation",
    "dca-bm-gear":               "defunct gear Input — DM consolidation",
    "dca-lppl-gear":             "defunct gear Input — DM consolidation",
    "dca-hybppl-gear":           "defunct gear Input — DM consolidation",
    "dca-eppl-gear":             "defunct gear Input — DM consolidation",
    "dca-lppl-summary-inline":   "defunct summary Output — DM consolidation",
    "dca-hybppl-summary-inline": "defunct summary Output — DM consolidation",
    "dca-eppl-summary-inline":   "defunct summary Output — DM consolidation",
    "ret-bm-gear":               "defunct gear Input — DM consolidation",
    "ret-lppl-gear":             "defunct gear Input — DM consolidation",
    "ret-hybppl-gear":           "defunct gear Input — DM consolidation",
    "ret-eppl-gear":             "defunct gear Input — DM consolidation",
    "ret-lppl-summary-inline":   "defunct summary Output — DM consolidation",
    "ret-hybppl-summary-inline": "defunct summary Output — DM consolidation",
    "ret-eppl-summary-inline":   "defunct summary Output — DM consolidation",
    "sc-bm-gear":                "defunct gear Input — DM consolidation",
    "sc-lppl-gear":              "defunct gear Input — DM consolidation",
    "sc-hybppl-gear":            "defunct gear Input — DM consolidation",
    "sc-eppl-gear":              "defunct gear Input — DM consolidation",
    "sc-lppl-summary-inline":    "defunct summary Output — DM consolidation",
    "sc-hybppl-summary-inline":  "defunct summary Output — DM consolidation",
    "sc-eppl-summary-inline":    "defunct summary Output — DM consolidation",
    # Snapshot-lots restore button — lives inside a banner that only
    # renders when a shared link injects snapshot-lots; layout walker
    # sees the empty placeholder without the button.
    "restore-lots-btn":          "banner button — only rendered when snapshot-lots is populated",
}


def _walk_ids(node, out):
    """Recursively collect every component id from a Dash layout tree."""
    if node is None:
        return
    if isinstance(node, (list, tuple)):
        for x in node:
            _walk_ids(x, out)
        return
    cid = getattr(node, "id", None)
    if isinstance(cid, str):
        out.add(cid)
    elif isinstance(cid, dict):
        # pattern-matching id — skip
        pass
    children = getattr(node, "children", None)
    if children is not None:
        _walk_ids(children, out)


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
    from _app_ctx import app
    refs: dict[str, list[tuple[str, str, str]]] = {}
    for key, entry in app.callback_map.items():
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
