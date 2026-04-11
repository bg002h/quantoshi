"""Shared Display Models panel + option builder.

Used by tabs 1/3/4/5 (Bubble, DCA, Retire, Supercharger) via
`display_models_panel(prefix, **flags)` for initial layout, and by
`callbacks/charts.py::update_model_swatches` via
`build_display_models_options(mc, ..., summaries=dict)` for palette rebuild.

Heatmap (tab 2) does NOT use this module — it has a pill bar, not a
checklist. Heatmap's status row lives in `layout/heatmap.py`.

See spec: docs/superpowers/specs/2026-04-11-display-models-consolidation-design.md
"""
from __future__ import annotations

from dash import dcc, html

import _app_ctx
from colors import (
    BLACK, FALLBACK_MODEL_GRAY, LOT_MARKER_OUTLINE,
    MODEL_TRACE_COLORS, CITADEL_OVERLAY_COLORS, LINK,
)
from layout.common import (
    _GEAR_STYLE, _MUTED_STYLE,
    _model_info_link, _section_card, _legend_pos_dropdown,
    _INFO_ICON,
)


def build_display_models_options(
    mc: dict,
    prefix: str,
    *,
    include_bm_master: bool = False,
    include_mc: bool = False,
    include_u1: bool = True,
    summaries: dict | None = None,
) -> list[dict]:
    """Pure function. Builds checklist options for a Display Models panel.

    `summaries` keys: "lppl", "hybppl", "eppl". When None, inline summary
    spans emit static defaults that get overwritten by the Store-reader
    clientside callback on page load.
    """
    # FULL implementation lands in Task 2. Stub for Task 1:
    return []


def display_models_panel(
    prefix: str,
    *,
    include_bm_master: bool = False,
    include_mc: bool = False,
    include_u1: bool = True,
    legend_pos_default: str = "bottom-right",
):
    """Return the Display Models section_card for one checklist-style tab."""
    # FULL implementation lands in Task 2. Stub for Task 1:
    return _section_card("Display Models",
        dcc.Checklist(id=f"{prefix}-model-show", options=[], value=[]),
        *_legend_pos_dropdown(prefix, legend_pos_default),
    )
