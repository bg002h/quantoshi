"""Tab-1 sub-view table — the single source of truth for view-mode UI state.

Tab 1 shows one of five views (Price / Forward CAGR / Residuals / Percentile /
Occupancy), selected through the ``bub-view-mode`` store.  Switching view has to
move a fixed set of UI state in lockstep:

* which graph wrapper is visible (``wrap``),
* which pill is filled vs. outlined (``pill``),
* whether the axis-scale controls and the bubble-composite panel show
  (``scale_controls`` / ``bubble_panel``),
* whether a view-specific control span shows (``ctl``),
* whether the view is historical-only — x-range slider capped at next year and
  the "N future bubbles" slider hidden (``historical``),
* which deep link opens it (``deep_link``).

That state used to be written out by hand in three places that had to agree:
``callbacks.charts.toggle_bub_view`` (pill clicks), the clientside sync callback
right below it (snapshot restore), and ``callbacks.routing.deep_link_bub_view``
(URL deep links).  All three now read this table, so adding a sixth view is one
row here plus its layout + figure callback.

Deliberately import-light: no Dash, no layout imports, so tests and both
callback modules can import it without cycles.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import NamedTuple

__all__ = [
    "ViewMode", "VIEW_MODES", "WRAP_IDS", "PILL_IDS", "CTL_IDS",
    "HISTORICAL_MODES", "DEFAULT_MODE", "mode_styles", "mode_for_path",
    "styles_table_json", "historical_modes_js",
]


class ViewMode(NamedTuple):
    """One Tab-1 sub-view.  Frozen by construction (NamedTuple)."""

    pill: str            # pill button id (`outline` prop is driven)
    wrap: str            # graph wrapper div id (`style` prop is driven)
    historical: bool     # historical-only: x-range capped, N-future hidden
    scale_controls: bool # show the axis scale controls block
    bubble_panel: bool   # show the bubble-composite panel
    ctl: str | None      # view-specific control span id, or None
    deep_link: str | None  # URL prefix that opens this view, or None


# ── The table ────────────────────────────────────────────────────────────────
# Order matters: WRAP_IDS / PILL_IDS / CTL_IDS and the 14-value style tuple are
# derived from it, and the server callbacks' Output order mirrors it exactly.
VIEW_MODES: MappingProxyType[str, ViewMode] = MappingProxyType({
    "price": ViewMode(
        pill="bub-view-price", wrap="bub-price-wrap", historical=False,
        scale_controls=True, bubble_panel=True, ctl=None, deep_link=None),
    "cagr": ViewMode(
        pill="bub-view-cagr", wrap="bub-cagr-wrap", historical=False,
        scale_controls=False, bubble_panel=False,
        ctl="bub-cagr-fwd-wrap", deep_link="/1.2"),
    "resid": ViewMode(
        pill="bub-view-resid", wrap="bub-resid-wrap", historical=True,
        scale_controls=True, bubble_panel=True, ctl=None, deep_link="/1.3"),
    "percentile": ViewMode(
        pill="bub-view-pctile", wrap="bub-pctile-wrap", historical=True,
        scale_controls=True, bubble_panel=True, ctl=None, deep_link="/1.4"),
    "occupancy": ViewMode(
        pill="bub-view-occ", wrap="bub-occ-wrap", historical=True,
        scale_controls=True, bubble_panel=True,
        ctl="bub-occ-ctl-wrap", deep_link="/1.5"),
})

#: The view a page loads in, and the clientside sync's fallback.
DEFAULT_MODE = "price"

#: Ordered id lists, derived from the table (never hand-maintained).
WRAP_IDS: tuple[str, ...] = tuple(v.wrap for v in VIEW_MODES.values())
PILL_IDS: tuple[str, ...] = tuple(v.pill for v in VIEW_MODES.values())
CTL_IDS: tuple[str, ...] = tuple(
    v.ctl for v in VIEW_MODES.values() if v.ctl is not None)

#: Ids whose `style` is driven between the pill outlines and the ctl spans.
PANEL_IDS: tuple[str, str] = ("bub-scale-controls", "bub-bubble-panel")

#: Views that only ever show past data.
HISTORICAL_MODES: frozenset[str] = frozenset(
    m for m, v in VIEW_MODES.items() if v.historical)

#: Full Output order of the 14 style/outline values `mode_styles` returns.
STYLE_OUTPUT_IDS: tuple[str, ...] = WRAP_IDS + PILL_IDS + PANEL_IDS + CTL_IDS


def _hide() -> dict:
    return {"display": "none"}


def _inline() -> dict:
    return {"display": "inline"}


def mode_styles(mode: str) -> tuple:
    """The 14 view-state values for ``mode``, in Output order.

    Order: 5 wrapper styles (:data:`WRAP_IDS`), 5 pill ``outline`` booleans
    (:data:`PILL_IDS`, active pill ``False``), the scale-controls and
    bubble-panel styles, then the control-span styles (:data:`CTL_IDS`).

    Unknown modes fall back to :data:`DEFAULT_MODE` — the same behaviour as the
    clientside sync's ``T[mode] || T["price"]``.
    """
    if mode not in VIEW_MODES:
        mode = DEFAULT_MODE
    spec = VIEW_MODES[mode]
    wraps = [({} if v.wrap == spec.wrap else _hide()) for v in VIEW_MODES.values()]
    pills = [v.pill != spec.pill for v in VIEW_MODES.values()]
    panels = [{} if spec.scale_controls else _hide(),
              {} if spec.bubble_panel else _hide()]
    ctls = [(_inline() if cid == spec.ctl else _hide()) for cid in CTL_IDS]
    return tuple(wraps + pills + panels + ctls)


def mode_for_path(pathname: str | None) -> str | None:
    """Longest deep-link prefix match, or ``None``.

    ``pathname`` must already be normalised (``callbacks.routing._norm`` turns
    ``/1-5-3-1`` into ``/1.5.3.1``); this module stays free of routing imports.
    """
    if not pathname:
        return None
    best: tuple[int, str] | None = None
    for mode, spec in VIEW_MODES.items():
        if spec.deep_link and pathname.startswith(spec.deep_link):
            n = len(spec.deep_link)
            if best is None or n > best[0]:
                best = (n, mode)
    return best[1] if best else None


# ── Generated JavaScript ─────────────────────────────────────────────────────
# The clientside callbacks in callbacks/charts get their tables from here, so a
# new row above reaches the browser without anyone editing JS.

#: Marker string embedded in the sync JS so tests can find that exact script.
SYNC_JS_MARKER = "QS_BUB_VIEW_SYNC_TABLE"


def styles_table_json() -> str:
    """``{mode: [14 values]}`` as JSON, for the clientside sync callback."""
    return json.dumps({m: list(mode_styles(m)) for m in VIEW_MODES},
                      sort_keys=False)


def historical_modes_js() -> str:
    """A JS boolean expression over ``mode`` that is true for historical views.

    Emitted as an explicit ``mode === 'x' || ...`` chain rather than a JSON
    array membership test: it reads the same in a browser devtools breakpoint
    as it does here, and it keeps the two historical-only clientside callbacks
    greppable by mode name (``test_occupancy.py`` pins exactly that).  Either
    way the source of truth is :data:`HISTORICAL_MODES` — this string is
    generated, never typed.
    """
    return " || ".join(f"mode === '{m}'" for m in sorted(HISTORICAL_MODES))
