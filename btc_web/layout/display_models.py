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
    UI_FONT_MD,
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
    # Deprioritized models appear at the bottom in this explicit order.
    # S2F before Exponential per user preference (2026-04-11).
    _DEPRIORITIZED = ("s2f", "exp", "gomp", "bpl", "plo", "sexp", "logi")

    def _swatch_span(color):
        return html.Span(" ", style={
            "display": "inline-block", "width": "12px", "height": "12px",
            "borderRadius": "2px", "verticalAlign": "middle",
            "marginRight": "4px", "backgroundColor": color,
        })

    def _gear_span(gear_id, title):
        # html.A with href="javascript:void(0)" so it stays interactive
        # content (label skips its checkbox-toggle per spec) but produces
        # no navigation, no scroll, no hash change. A single document-level
        # listener in assets/gear_clicks.js reads the gear's data-family
        # attribute and calls window.dash_clientside.set_props on the
        # matching modal. No Dash callbacks → no phantom-Input risk from
        # lazy tabs.
        family = gear_id.rsplit("-gear", 1)[0].rsplit("-", 1)[-1]
        return html.A(
            "\u2699\uFE0F", id=gear_id, href="javascript:void(0)",
            style=_GEAR_STYLE, title=title, className="qs-gear",
            **{"data-family": family},
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
        # dcc.Link(refresh=False, target="_self") = SPA navigation:
        # pushState + clientside URL router switches Dash tab and opens
        # accordion item N, while preserving control state on other tabs.
        # target="_self" is REQUIRED: dcc.Link's onClick guard only calls
        # preventDefault when target === "_self" OR the URL is internal.
        # Without target, preventDefault never fires → browser does a real
        # navigation → hits Flask /mi.N static route (tab-less page).
        return dcc.Link(
            _INFO_ICON, href=href, refresh=False, target="_self",
            style={"cursor": "pointer", "fontSize": UI_FONT_MD,
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

    # Canonical top-of-list order: BM, PL, QR, Entropy PPL, Hybrid PPL, LPPL.
    # BM first because it's the default flagship model. PL + QR next as the
    # two simplest baselines. Then the three oscillatory family masters in
    # increasing model complexity: EPPL (entropy-damped) → HybPPL (hybrid
    # periodic) → LPPL (full log-periodic power law).
    _ = include_bm_master  # kept as a no-op kwarg for backward compatibility.

    # 1. Bubble Model — gear on ALL tabs (opens global bm-config-modal).
    opts.append({
        "label": html.Span([
            _swatch_span(mc.get("bub", BLACK)),
            "Bubble Model",
            _gear_span(f"{prefix}-bm-gear", "Open Bubble Model settings"),
        ]),
        "value": "bub",
    })

    # 2-3. PL and QR as plain entries. Pulled from PRICE_MODELS for display
    # name; skipped below in the primary-block emission via _HIDDEN.
    for key in ("pl", "qr"):
        mdl = _app_ctx.PRICE_MODELS.get(key)
        if mdl is None:
            continue
        opts.append({
            "label": _plain_label(
                mc.get(key, FALLBACK_MODEL_GRAY),
                mdl.name,
                model_key=key,
            ),
            "value": key,
        })

    # 4. Entropy PPL master
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

    # 5. Hybrid PPL master
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

    # 6. LPPL master
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

    # 7-9. Remaining non-master model entries (pl, qr excluded — already emitted above)
    _HIDDEN = (
        set(_app_ctx.MODEL_SENTINELS)
        | {"bub", "eppl", "pl", "qr"}
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
    deprior.sort(key=lambda m: _DEPRIORITIZED.index(m.short_name))

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


def sigma_mode_section():
    """Bubble-tab-only σ mode radio (constant vs residual QR bands).

    Rendered directly from layout/bubble.py so it can sit below the
    Projection Quantiles card rather than alongside Display Models.
    """
    has_resqr = bool(getattr(_app_ctx, "_HAS_RESQR", False))
    radio_options = [
        {"label": " Constant σ (legacy)", "value": "constant"},
        {"label": " Residual quantile bands", "value": "resqr",
         "disabled": not has_resqr},
    ]
    disclaimer_applicability = html.Div(
        "Applies to Tab 1 (Price & Model Overlays) only — other tabs "
        "use the legacy constant-σ bands regardless of this setting.",
        style={"fontSize": UI_FONT_MD, "color": FALLBACK_MODEL_GRAY,
                "marginTop": "6px"},
    )
    disclaimer_lppl = html.Div(
        "LPPL family is excluded from residual quantile bands (the "
        "3.92-year halving-cycle residual periodicity is not well "
        "captured by the piecewise-linear basis).",
        style={"fontSize": UI_FONT_MD, "color": FALLBACK_MODEL_GRAY,
                "marginTop": "4px"},
    )
    children = [
        dcc.RadioItems(
            id="bub-sigma-mode",
            options=radio_options,
            value="constant",
            labelStyle={"display": "block"},
            inputStyle={"marginRight": "4px"},
        ),
        disclaimer_applicability,
        disclaimer_lppl,
    ]
    if not has_resqr:
        children.append(html.Div(
            "(Residual quantile bands unavailable on this deployment — "
            "run tools/build_bm_model.py to generate them.)",
            style={"fontSize": UI_FONT_MD, "color": FALLBACK_MODEL_GRAY,
                    "marginTop": "4px", "fontStyle": "italic"},
        ))
    return _section_card("σ mode", *children)


def display_models_panel(
    prefix: str,
    *,
    include_bm_master: bool = False,
    include_mc: bool = False,
    include_u1: bool = True,
    default_value: list[str] | None = None,
    legend_pos_default: str = "bottom-right",
):
    """Return the Display Models section_card for one checklist-style tab.

    `default_value` — initial checklist selection. If None, falls back to
    `["bub"] + (["mc"] if include_mc and _HAS_MARKOV else [])` matching the
    pre-refactor behavior of `_model_show_checklist`.
    """
    mc = _app_ctx.PALETTES["default"]["model_colors"]
    options = build_display_models_options(
        mc, prefix,
        include_bm_master=include_bm_master,
        include_mc=include_mc,
        include_u1=include_u1,
    )
    if default_value is None:
        default_value = ["bub"] + (
            ["mc"] if include_mc and _app_ctx._HAS_MARKOV else []
        )
    return _section_card(
        "Display Models",
        dcc.Checklist(
            id=f"{prefix}-model-show",
            options=options,
            value=list(default_value),
            labelStyle={"display": "block"},
            inputStyle={"marginRight": "4px"},
        ),
        *_legend_pos_dropdown(prefix, legend_pos_default),
    )
