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
    summaries = summaries or {}
    _LPPL_FAM   = {"lppl", "lp2", "lp3", "lp4"} | set(_app_ctx.LPPL_FAMILY_HIDDEN_FROM_BUBBLE)
    _HYBPPL_FAM = set(_app_ctx.HYBPPL_FAMILY_HIDDEN)
    _PROMOTED   = ("pca", "grdy")
    _DEPRIORITIZED = {"exp", "s2f", "gomp", "bpl"}

    def _swatch_span(color):
        return html.Span(" ", style={
            "display": "inline-block", "width": "12px", "height": "12px",
            "borderRadius": "2px", "verticalAlign": "middle",
            "marginRight": "4px", "backgroundColor": color,
        })

    def _gear_span(gear_id, title):
        return html.Span(
            "\u2699\uFE0F", id=gear_id, n_clicks=0,
            style=_GEAR_STYLE, title=title,
        )

    def _inline_summary(span_id, default):
        return [
            html.Span(" (", style=_MUTED_STYLE),
            html.Span(id=span_id, children=default, style=_MUTED_STYLE),
            html.Span(")", style=_MUTED_STYLE),
        ]

    def _info_link(model_key, label):
        href, exists = _model_info_link(model_key)
        if not exists:
            return None
        return html.A(
            _INFO_ICON, href=href,
            style={"cursor": "pointer", "fontSize": "11px",
                   "marginLeft": "4px", "opacity": "0.6",
                   "textDecoration": "none", "color": LINK},
            title=f"View {label} details on Model Info tab",
        )

    def _master_label(color, name, gear_id, summary_id, summary_default, gear_title):
        return html.Span([
            _swatch_span(color),
            name,
            *_inline_summary(summary_id, summary_default),
            _gear_span(gear_id, gear_title),
        ])

    def _plain_label(color, name, model_key=None):
        children = [_swatch_span(color), name]
        if model_key:
            link = _info_link(model_key, name)
            if link:
                children.append(link)
        return html.Span(children)

    opts = []

    # 1. Bubble Model
    if include_bm_master:
        opts.append({
            "label": html.Span([
                _swatch_span(mc.get("bub", BLACK)),
                "Bubble Model",
                _gear_span(f"{prefix}-bm-gear", "Open Bubble Model settings"),
            ]),
            "value": "bub",
        })
    else:
        opts.append({
            "label": _plain_label(mc.get("bub", BLACK), "Bubble Model"),
            "value": "bub",
        })

    # 2. Entropy PPL master
    opts.append({
        "label": _master_label(
            mc.get("eppl", MODEL_TRACE_COLORS["eppl"]),
            "\U0001FAE0 Entropy PPL",
            gear_id=f"{prefix}-eppl-gear",
            summary_id=f"{prefix}-eppl-summary-inline",
            summary_default=summaries.get("eppl", "1d+1u"),
            gear_title="Configure Entropy PPL",
        ),
        "value": "eppl",
    })

    # 3. LPPL master
    opts.append({
        "label": _master_label(
            mc.get("lppl", MODEL_TRACE_COLORS["lppl"]),
            "LPPL",
            gear_id=f"{prefix}-lppl-gear",
            summary_id=f"{prefix}-lppl-summary-inline",
            summary_default=summaries.get("lppl", "LPPL\u2083"),
            gear_title="Configure LPPL",
        ),
        "value": "lppl",
    })

    # 4. Hybrid PPL master
    opts.append({
        "label": _master_label(
            mc.get("hybppl", CITADEL_OVERLAY_COLORS["reserves_total"]),
            "Hybrid PPL",
            gear_id=f"{prefix}-hybppl-gear",
            summary_id=f"{prefix}-hybppl-summary-inline",
            summary_default=summaries.get("hybppl", "1d+1u"),
            gear_title="Configure Hybrid PPL",
        ),
        "value": "hybppl",
    })

    # 5-7. Non-master model entries
    _HIDDEN = (
        set(_app_ctx.MODEL_SENTINELS)
        | {"bub", "eppl"}
        | _LPPL_FAM
        | _HYBPPL_FAM
    )
    all_models = [
        m for m in _app_ctx.PRICE_MODELS.values()
        if m.short_name not in _HIDDEN
        and not m.short_name.startswith("cfg_")
        and not m.short_name.startswith("ecfg_")
    ]
    promoted = [m for m in all_models if m.short_name in _PROMOTED]
    promoted.sort(key=lambda m: _PROMOTED.index(m.short_name))
    primary  = [m for m in all_models
                if m.short_name not in _PROMOTED
                and m.short_name not in _DEPRIORITIZED]
    deprior  = [m for m in all_models if m.short_name in _DEPRIORITIZED]

    for mdl in promoted + primary + deprior:
        opts.append({
            "label": _plain_label(
                mc.get(mdl.short_name, FALLBACK_MODEL_GRAY),
                mdl.name,
                model_key=mdl.short_name,
            ),
            "value": mdl.short_name,
        })

    # 8. U₁
    if include_u1:
        opts.append({
            "label": _plain_label(
                mc.get("u1", LOT_MARKER_OUTLINE),
                "U\u2081 (User)",
                model_key="u1",
            ),
            "value": "u1",
        })

    # 9. MC
    if include_mc:
        opts.append({
            "label": _plain_label(
                mc.get("mc", FALLBACK_MODEL_GRAY),
                "MC Simulation",
            ),
            "value": "mc",
        })

    return opts


def display_models_panel(
    prefix: str,
    *,
    include_bm_master: bool = False,
    include_mc: bool = False,
    include_u1: bool = True,
    legend_pos_default: str = "bottom-right",
):
    """Return the Display Models section_card for one checklist-style tab."""
    mc = _app_ctx.PALETTES["default"]["model_colors"]
    options = build_display_models_options(
        mc, prefix,
        include_bm_master=include_bm_master,
        include_mc=include_mc,
        include_u1=include_u1,
    )
    return _section_card(
        "Display Models",
        dcc.Checklist(
            id=f"{prefix}-model-show",
            options=options,
            value=[],
            labelStyle={"display": "block"},
            inputStyle={"marginRight": "4px"},
        ),
        *_legend_pos_dropdown(prefix, legend_pos_default),
    )
