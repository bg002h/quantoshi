"""One-shot helper for Phase 1 of the SNAPSHOT_DEFAULTS migration.

Imports each tab's layout factory directly (bypassing universal lazy-
tab loading), walks the component tree, and emits a Python dict
literal mapping each (cid, prop) in _SNAPSHOT_CONTROLS to the initial
value found in the layout. Manual review required - unresolved entries
are emitted with TODO markers.
"""
from __future__ import annotations
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "btc_web"))

os.environ["DEV"] = "1"

import app  # noqa: F401
import _app_ctx
from snapshot import _SNAPSHOT_CONTROLS


def _walk(node):
    if node is None:
        return
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for c in children:
            yield from _walk(c)
    else:
        yield from _walk(children)


with _app_ctx.app.server.test_request_context("/1"):
    roots = []
    # Top-level layout
    try:
        roots.append(_app_ctx.app.layout())
    except Exception as e:
        print(f"# WARNING: top layout raised: {e}")
    from layout import bubble as _bubble
    from layout import heatmap as _heatmap
    from layout import sim_tabs as _sim_tabs
    from layout import supercharge as _supercharge
    from layout import citadel as _citadel
    from layout import citadel_tax as _citadel_tax
    from layout import leverage as _leverage
    from layout import stack as _stack
    from layout import display_models as _display_models
    from layout import custom_time as _custom_time
    from layout import faq as _faq
    from layout import model_info as _model_info

    candidates = [
        ("bubble",      getattr(_bubble,      "_bubble_tab",          None)),
        ("heatmap",     getattr(_heatmap,     "_heatmap_tab",         None)),
        ("dca",         getattr(_sim_tabs,    "_dca_tab",             None)),
        ("retire",      getattr(_sim_tabs,    "_retire_tab",          None)),
        ("supercharge", getattr(_supercharge, "_supercharge_tab",     None)),
        ("citadel",     getattr(_citadel,     "_citadel_tab",         None)),
        ("citadel_tax_modal", getattr(_citadel_tax, "tax_config_modal", None)),
        ("citadel_tax_summary", getattr(_citadel_tax, "tax_summary_panel", None)),
        ("leverage",    getattr(_leverage,    "_leverage_tab",        None)),
        ("stack",       getattr(_stack,       "_stack_tracker_tab",   None)),
        ("model_info",  getattr(_model_info,  "_model_info_tab",      None)),
        ("faq",         getattr(_faq,         "_faq_tab",             None)),
        ("custom_time", getattr(_custom_time, "custom_time_panel",    None)),
    ]
    for name, fn in candidates:
        if fn is None:
            print(f"# WARNING: layout factory not found for {name!r}")
            continue
        try:
            roots.append(fn())
        except Exception as e:
            print(f"# WARNING: {name!r} factory raised: {e}")


by_id: dict[str, object] = {}
for root in roots:
    for n in _walk(root):
        nid = getattr(n, "id", None)
        if isinstance(nid, str):
            by_id.setdefault(nid, n)


resolved: dict[str, object] = {}
unresolved: list[tuple[str, str]] = []
for cid, prop in _SNAPSHOT_CONTROLS:
    comp = by_id.get(cid)
    if comp is None:
        unresolved.append((cid, prop))
        continue
    val = getattr(comp, prop, None)
    if isinstance(val, tuple):
        val = list(val)
    resolved[f"{cid}:{prop}"] = val

print("# === RESOLVED (paste into SNAPSHOT_DEFAULTS) ===")
for k in sorted(resolved):
    v = resolved[k]
    print(f"    {k!r}: {v!r},")
print()
print(f"# === UNRESOLVED ({len(unresolved)}) ===")
for cid, prop in unresolved:
    print(f"    {cid!r:40s} {prop!r}")
print()
print(f"# Resolved: {len(resolved)}; Unresolved: {len(unresolved)}; Total controls: {len(_SNAPSHOT_CONTROLS)}")
