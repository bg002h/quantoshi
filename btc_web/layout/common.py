"""Shared layout helpers, style constants, and reusable control builders."""

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from colors import (
    LINK, USER_MODEL_TRACE, FALLBACK_MODEL_GRAY,
    BLACK, MODEL_TRACE_COLORS, CITADEL_OVERLAY_COLORS,
    MODAL_DIVIDER_DARK, PROGRESS_TRACK,
    EPPL_SUMMARY_COLOR, DIM_TEXT, MUTED_TEXT, MUTED_SUMMARY_TEXT,
    DECOMP_ERROR_RED, ERROR_BG, ERROR_BORDER,
    MC_FREE_GREEN, MC_LIVE_AMBER, KNIGHT_GOLD,
    LIGHTBOX_BG, TABLE_HEADER_BG, TABLE_BORDER_LIGHT,
    TABLE_BORDER_MID, TABLE_BORDER_DARK, CODE_BG, BOOTSTRAP_LIGHT_BG,
    CTX_MENU_BG, _hex_alpha, CTX_MENU_BG_ALPHA, CTX_MENU_SHADOW_ALPHA,
    UI_FONT_SM, UI_FONT_MD, UI_FONT_BASE, UI_FONT_LG, UI_FONT_XL,
    Q_OPACITY_FLOOR, Q_OPACITY_RANGE, Q_OPACITY_DECAY,
)


# ── Model Info deep-link helper ────────────────────────────────────────────

_INFO_ICON = "\U0001F4D0"  # 📐 same as Model Info tab


def _modal_header_with_info_link(title: str, model_short_name: str, link_id: str):
    """ModalHeader containing the title plus a 📐 link to the Model Info tab.

    Clicking the icon SPA-navigates to /8.N for that model (no page reload,
    so other tabs' control state is preserved) AND closes the modal via a
    clientside callback in charts.py (see _close_config_modal_on_info).
    If the model has no Model Info entry, the icon is omitted.

    Uses dcc.Link rather than html.A so clicks update dcc.Location.pathname
    via pushState instead of triggering a full page navigation. A real
    browser refresh still hits _serve_layout and resets to defaults —
    i.e. the "refresh to start over" behaviour is preserved.
    """
    href, exists = _model_info_link(model_short_name)
    children = [dbc.ModalTitle(title)]
    if exists:
        children.append(dcc.Link(
            _INFO_ICON,
            id=link_id,
            href=href,
            refresh=False,
            target="_self",  # required for preventDefault; see display_models.py
            style={
                "marginLeft": "8px",
                "fontSize": UI_FONT_XL,
                "textDecoration": "none",
                "color": LINK,
                "cursor": "pointer",
            },
            title=f"View {title} on Model Info tab",
        ))
    return dbc.ModalHeader(html.Div(
        children,
        style={"display": "flex", "alignItems": "center", "gap": "4px"},
    ))


def _model_info_link(short_name):
    """Return (href, exists) for the Model Info deep-link for a model.

    Looks up 'mi-{short_name}' in _MODEL_INFO_ITEMS to find the 1-indexed
    position. Returns ("/mi.N", True) if found, ("", False) if not.
    Tolerant of reordering — computed at layout time from the live list.

    Uses /mi.N (stable name-based route) rather than /8.N so links don't
    break if Model Info changes tab position in the future.
    """
    # Lazy import to avoid circular dependency
    from callbacks.routing import _MODEL_INFO_ITEMS
    mi_key = f"mi-{short_name}"
    try:
        idx = _MODEL_INFO_ITEMS.index(mi_key) + 1
        return f"/mi.{idx}", True
    except ValueError:
        return "", False

# ── Reusable style constants ────────────────────────────────────────────────
_STYLE_HIDDEN     = {"display": "none"}
_STYLE_HINT       = {"color": FALLBACK_MODEL_GRAY, "display": "block", "marginBottom": "4px"}
_STYLE_GRAPH_H    = {"height": "78vh"}
_STYLE_COLOR_H    = {"height": "28px"}
_STYLE_ADDR_CELL  = {"paddingRight": "12px", "whiteSpace": "nowrap", "verticalAlign": "top"}
_STYLE_ADDR_CODE  = {"wordBreak": "break-all", "fontSize": UI_FONT_MD}
_CB_MARGIN        = {"marginRight": "4px"}
_INFL_LABEL       = "Inflation rate (0\u2013100% / yr)"
_Q_HINT_BASE      = "Lower quantiles = more conservative price paths."

# ── Styles for Display Models in-checklist-label controls ───────────
# Moved from layout/bubble.py during display-models consolidation so
# that display_models.py and heatmap.py can share them.
_GEAR_STYLE = {
    "cursor": "pointer", "fontSize": UI_FONT_MD, "marginLeft": "4px",
    "opacity": "0.6", "textDecoration": "none",
}
_MUTED_STYLE = {
    "color": MUTED_SUMMARY_TEXT, "fontSize": UI_FONT_MD, "fontStyle": "italic",
}

def _q_options() -> list[dict]:
    opts = []
    for q in _app_ctx._ALL_QS:
        pct = q * 100
        lbl_text = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
        # Opacity fades with distance from median — lighter for extreme quantiles.
        # Uses the same Q_OPACITY_* parameters as the chart quantile_opacity()
        # formula, but inlined here to avoid circular import from figures/.
        alpha = max(Q_OPACITY_FLOOR,
                    1.0 - abs(q - 0.5) / Q_OPACITY_RANGE * Q_OPACITY_DECAY)
        col = _hex_alpha(DIM_TEXT, alpha)
        lbl = html.Span([
            html.Span("\u25CF ", style={"color": col, "fontSize": UI_FONT_SM}),
            lbl_text,
        ])
        opts.append({"label": lbl, "value": q})
    return opts

_DEFAULT_QS = [0.01, 0.15, 0.50, 0.85, 0.99]

# Band definitions for default mode: each checkbox selects a symmetric pair
_DEFAULT_BANDS = [
    {"value": "inner", "qs": [0.15, 0.85], "label": "Q15% \u2013 Q85%"},
    {"value": "outer", "qs": [0.01, 0.99], "label": "Q1% \u2013 Q99%"},
    {"value": "median", "qs": [0.50], "label": "Q50% (median)"},
]

def _bands_to_qs(band_values: list) -> list[float]:
    """Expand band checkbox values to quantile floats."""
    qs = []
    for b in _DEFAULT_BANDS:
        if b["value"] in (band_values or []):
            qs.extend(b["qs"])
    return sorted(set(qs))

def _q_options_default() -> list[dict]:
    """Band-pair options for default mode (3 checkboxes)."""
    return [{"label": f" {b['label']}", "value": b["value"]}
            for b in _DEFAULT_BANDS]


def _q_panel_with_mode(checklist_id: str, default_value: list,
                       hint: str | None = None):
    """Quantile panel with default/advanced toggle.

    Default mode: 5 options (Q1/15/50/85/99%), max 3 bands.
    Advanced mode: all quantiles, no limit.
    """
    children = []
    if hint:
        children.append(html.Small(hint, style=_STYLE_HINT))

    mode_id = f"{checklist_id}-mode"
    children.append(
        dcc.Checklist(id=mode_id,
                      options=[{"label": " Advanced", "value": "advanced"}],
                      value=[], inputStyle=_CB_MARGIN,
                      className="small mb-1"),
    )
    # Map raw quantile defaults to band values
    default_bands = []
    for b in _DEFAULT_BANDS:
        if any(q in default_value for q in b["qs"]):
            default_bands.append(b["value"])
    if not default_bands:
        default_bands = ["median"]
    children.append(
        html.Div(id=f"{checklist_id}-default-wrap", children=[
            dcc.Checklist(id=checklist_id, options=_q_options_default(),
                          value=default_bands,
                          labelStyle={"display": "block"},
                          inputStyle=_CB_MARGIN),
        ]),
    )
    children.append(
        html.Div(id=f"{checklist_id}-advanced-wrap",
                 style=_STYLE_HIDDEN, children=[
            dcc.Checklist(id=f"{checklist_id}-adv", options=_q_options(),
                          value=default_value, className="q-panel-grid",
                          inputStyle=_CB_MARGIN),
        ]),
    )

    return _section_card("Projection Quantiles", *children)


def _q_panel(checklist_id: str, default_value: list, hint: str | None = None):
    """Quantile checklist — static multi-column grid, no collapse."""
    children = []
    if hint:
        children.append(html.Small(hint,
            style=_STYLE_HINT))
    children.append(
        dcc.Checklist(id=checklist_id, options=_q_options(),
                      value=default_value, className="q-panel-grid",
                      inputStyle=_CB_MARGIN),
    )
    return _section_card("Projection Quantiles", *children)


def _palette_selector(tab_key: str = "tab"):
    """Palette selector widget for a tab's control panel.

    Syncs with palette-store via clientside callbacks in nav.py.
    ID pattern: palette-select-{tab_key} (distinct from navbar's palette-select).
    Every chart tab that renders this helper must pass a unique `tab_key`
    (e.g. "bub", "hm", "dca", "ret", "sc", "cp") so multiple selectors
    can coexist in the pre-rendered layout.
    """
    return _ctrl_card(
        html.Div([
            html.Small("\U0001f3a8 Color palette", className="text-muted me-2"),
            dbc.Select(
                id=f"palette-select-{tab_key}",
                options=[{"label": v, "value": k}
                         for k, v in _app_ctx.PALETTE_LABELS.items()],
                value="default",
                size="sm",
                style={"width": "155px", "fontSize": "0.78rem",
                       "display": "inline-block"},
            ),
        ], className="d-flex align-items-center"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layout helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ctrl_card(*children):
    return dbc.Card(dbc.CardBody(list(children), className="p-2"),
                    className="mb-2 ctrl-card")

_SECTION_ICONS = {}  # Emptied in brand overhaul — no emoji prefixes on section headers

def _section_card(title: str, *children, header_right=None, no_hover=False):
    """Control card with a section header title and optional icon.

    When ``header_right`` is supplied, those widgets render on the right side
    of the header row (used by LPPL/MC model-config panels to place their
    activate-toggle + configure/action buttons inline with the title).

    When ``no_hover=True``, the card uses ``ctrl-card-nohover`` class which
    disables the :hover transform. Required for cards containing
    dcc.Dropdown — the transform creates a stacking context that clips the
    dropdown menu (and intercepts clicks) on iOS Safari.
    """
    if isinstance(title, str):
        icon = _SECTION_ICONS.get(title, "")
        prefix = f"{icon} " if icon else ""
        title_span = html.Span(f"{prefix}{title}")
    else:
        # Caller passed a pre-built component (e.g. html.Span with an id so
        # clientside callbacks can update the header text).
        title_span = title
    if header_right:
        header = html.Div(
            [title_span, html.Div(header_right, className="model-panel-header-right")],
            className="ctrl-section-header model-panel-header",
        )
    else:
        header = html.Div(title_span, className="ctrl-section-header")
    extra_class = " no-hover-transform" if no_hover else ""
    return dbc.Card(
        dbc.CardBody([header, *children], className="p-2"),
        className=f"mb-2 ctrl-card{extra_class}",
    )

def _row(*cols):
    return dbc.Row([dbc.Col(c) for c in cols], className="g-1 mb-1")

def _lbl(text: str):
    return html.Label(text, className="form-label mb-0 small")


def _use_lots_checklist(prefix: str):
    """Standard 'Use Stack Tracker lots' checklist used by bubble/heatmap/citadel."""
    return dcc.Checklist(
        id=f"{prefix}-use-lots",
        options=[{"label": " Use Stack Tracker lots", "value": "yes"}],
        value=[], inputStyle=_CB_MARGIN,
    )


def _plot_appearance_controls(prefix: str):
    """Trace width + grid width/color + data point color controls for a chart tab.

    Each tab gets its own copy of these 6 inputs + a reset button, all with
    prefixed IDs. The JS layer in btc_web/assets/plot_appearance.js owns the
    control plane entirely — it wires up input event listeners, reset button
    clicks, and keeps localStorage["plot-appearance"] in sync across all 5
    tabs. chart_responsive.js reads that same localStorage entry to apply
    trace width / grid / marker color to Plotly figures.

    See docs/superpowers/specs/2026-04-10-plot-appearance-control-plane-design.md.

    Returns a list of components to be spread into a _section_card.
    """
    # Grouped by concept: row 1 = data/trace (thickness + pt color),
    # row 2 = major grid pair (width + color), row 3 = minor grid pair.
    parts = [
        _row(
            html.Div([_lbl("Trace thickness (0.5\u20138)"),
                      dbc.Input(id=f"{prefix}-plot-trace-width", type="number",
                                value=None, min=0.5, max=8, step=0.5, size="sm")]),
            html.Div([_lbl("Data point color"),
                      dbc.Input(id=f"{prefix}-plot-pt-color", type="color",
                                value=None, size="sm",
                                style={"height": "30px", "padding": "2px"})]),
        ),
        _row(
            html.Div([_lbl("Major grid width (0\u20134)"),
                      dbc.Input(id=f"{prefix}-plot-grid-major-width", type="number",
                                value=None, min=0, max=4, step=0.25, size="sm")]),
            html.Div([_lbl("Major grid color"),
                      dbc.Input(id=f"{prefix}-plot-grid-major-color", type="color",
                                value=None, size="sm",
                                style={"height": "30px", "padding": "2px"})]),
        ),
        _row(
            html.Div([_lbl("Minor grid width (0\u20133)"),
                      dbc.Input(id=f"{prefix}-plot-grid-minor-width", type="number",
                                value=None, min=0, max=3, step=0.25, size="sm")]),
            html.Div([_lbl("Minor grid color"),
                      dbc.Input(id=f"{prefix}-plot-grid-minor-color", type="color",
                                value=None, size="sm",
                                style={"height": "30px", "padding": "2px"})]),
        ),
        html.Div(
            dbc.Button("Reset to defaults",
                       id=f"{prefix}-plot-appearance-reset",
                       size="sm", color="secondary",
                       outline=True,
                       className="mt-2",
                       style={"fontSize": UI_FONT_MD, "width": "100%"}),
            style={"marginTop": "8px"},
        ),
    ]
    return parts

def _export_row(tab_id: str):
    """Export row — download triggered client-side via Plotly.downloadImage()."""
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id=f"{tab_id}-fmt", options=["png","svg","jpeg","webp","html"], value="png",
                clearable=False, style={"minWidth": "90px"}), width="auto"),
            dbc.Col(dcc.Dropdown(
                id=f"{tab_id}-scale",
                options=[{"label":"1x (e-mail)","value":1},
                         {"label":"2x (screen)","value":2},
                         {"label":"3x (print)","value":3},
                         {"label":"4x (enlargements)","value":4}],
                value=2, clearable=False, style={"minWidth": "130px"}), width="auto"),
            dbc.Col(dbc.Input(id=f"{tab_id}-fname", value=f"btc_{tab_id}",
                              type="text", size="sm"), width=True),
            dbc.Col(dbc.Button("\u2b07 Download", id=f"{tab_id}-export-btn",
                               size="sm"), width="auto"),
            # dummy store — clientside callback needs an output target
            dcc.Store(id=f"{tab_id}-dl-dummy"),
        ], className="g-1 align-items-center"),
        html.Div("\u2193 Scroll down to configure",
                 className="d-md-none text-center text-muted py-1",
                 style={"fontSize": UI_FONT_MD, "letterSpacing":"0.02em"}),
    ], className="export-row-polished")


_LEGEND_POS_OPTIONS = [
    {"label": "Outside (right)", "value": "outside"},
    {"label": "Top-left",        "value": "top-left"},
    {"label": "Top-right",       "value": "top-right"},
    {"label": "Bottom-left",     "value": "bottom-left"},
    {"label": "Bottom-right",    "value": "bottom-right"},
]

def _chart_toggles(prefix, defaults=None):
    """Reusable chart toggle checklist (Log Y, Annotate, Shade, Legend, Minor grid, Zoom)."""
    opts = [{"label": " Log Y", "value": "log_y"},
            {"label": " Annotate final values", "value": "annotate"},
            {"label": " Discrete steps", "value": "discrete"},
            {"label": " Shade bands", "value": "shade"},
            {"label": " Show legend", "value": "show_legend"},
            {"label": html.Span(" Minor grid", className="minor-grid-opt"),
             "value": "minor_grid"},
            {"label": " Enable chart zoom", "value": "chart_zoom"}]
    return dcc.Checklist(id=f"{prefix}-toggles", options=opts,
                         value=defaults or [], labelStyle={"display": "block"},
                         inputStyle=_CB_MARGIN)


def _display_card(prefix, toggle_defaults=None, extra_children=None):
    """Shared "Display" section — toggle checklist ``{prefix}-toggles`` plus
    optional tab-specific children rendered BELOW the toggle list.

    Every chart tab gets one of these so users find on/off switches in the
    same place. Tabs with specialised toggles (e.g. bubble's
    show_data/show_ols/show_ucl/show_today) should use ``_display_card_custom``
    instead, which accepts a pre-built Checklist directly.
    """
    children = [_chart_toggles(prefix, toggle_defaults)]
    if extra_children:
        children.extend(extra_children)
    return _section_card("Display", *children)


def _display_card_custom(prefix, checklist, extra_children=None):
    """Like ``_display_card`` but uses a caller-supplied Checklist (for tabs
    whose toggle set diverges from the shared 7-option list — notably bubble
    with its show_data/show_ols/show_ucl/show_today appearance flags).
    """
    children = [checklist]
    if extra_children:
        children.extend(extra_children)
    return _section_card("Display", *children)


def _axes_range_card_sim(prefix, yr_min, yr_max, default_start, default_end,
                          *, show_disp=True, disp_kwargs=None, extra_children=None):
    """Shared "Axes & Range" card for DCA / Retire / Supercharge / Citadel.

    Holds the Year range RangeSlider and (optionally) the BTC/USD display-
    mode dropdown. Log-Y lives in the ``Display`` card via `_chart_toggles`.
    """
    children = [
        _lbl("Year range"),
        _year_range_slider(prefix, yr_min, yr_max, default_start, default_end),
    ]
    if show_disp:
        children.append(_btc_usd_dropdown(prefix, **(disp_kwargs or {})))
    if extra_children:
        children.extend(extra_children)
    return _section_card("Axes & Range", *children)


def _btc_usd_dropdown(prefix, btc_label="BTC Balance", default="btc"):
    """Reusable BTC/USD display mode dropdown."""
    return dcc.Dropdown(id=f"{prefix}-disp",
                        options=[{"label": btc_label, "value": "btc"},
                                 {"label": "USD Value", "value": "usd"}],
                        value=default, clearable=False)


def _legend_pos_dropdown(prefix, default="outside"):
    """Legend position dropdown — reused across chart tabs."""
    return [_lbl("Legend position"),
            dcc.Dropdown(id=f"{prefix}-legend-pos",
                         options=_LEGEND_POS_OPTIONS,
                         value=default, clearable=False)]

# ── Shared control builders ──────────────────────────────────────────────────

_BTC_ORANGE = _app_ctx.BTC_ORANGE

def _chart_tab_layout(controls_fn, graph_id, filename, mc_prefix=None,
                      start_collapsed=False):
    """Standard chart tab: 3-col controls (left) + 9-col graph (right).

    mc_prefix: if set, adds an MC overlay div (e.g. "dca" → "dca-mc-overlay").
    start_collapsed: if True, controls drawer starts collapsed on desktop.
    """
    overlay = []
    badge = []
    if mc_prefix:
        overlay = [html.Div(id=f"{mc_prefix}-mc-overlay",
                            style=_STYLE_HIDDEN,
                            className="mc-chart-overlay")]
        badge = [html.Img(id=f"{mc_prefix}-mc-badge",
                          src="/assets/quantoshi_favicon.png",
                          className="mc-premium-badge",
                          style=_STYLE_HIDDEN)]
    _col_cls = "controls-col overflow-auto"
    if start_collapsed:
        _col_cls += " drawer-collapsed"
    # Static placeholder preview image — shows while Plotly hydrates, then
    # is hidden by chart_responsive.js once the interactive chart has data.
    # IMPORTANT: z-index is 0 (BELOW the Plotly chart) so that even if the
    # hide logic ever fails, the rendered chart paints over the preview and
    # the user is never stuck staring at the static PNG. The hide is still
    # wired up (display:none) for screen readers + to stop the PNG from
    # consuming paint cycles, but the visual correctness does not depend
    # on it firing.
    preview_name = graph_id.replace("-graph", "")
    preview_img = html.Img(
        src=f"/assets/{preview_name}_preview.png",
        id=f"{preview_name}-preview-img",
        className="chart-preview-overlay",
        style={"position": "absolute", "top": 0, "left": 0,
               "width": "100%", "height": "100%",
               "objectFit": "contain", "zIndex": 0,
               "pointerEvents": "none"},
    )
    return dbc.Row([
        dbc.Col([
            controls_fn(),
        ], width=3, className=_col_cls,
                style={"maxHeight": "85vh"}),
        dbc.Col([
            html.Div(id=f"{mc_prefix or graph_id}-chart-wrap",
                     style={"position": "relative"}, children=[
                preview_img,
                dcc.Loading(
                    dcc.Graph(id=graph_id, style=_STYLE_GRAPH_H,
                              config={"scrollZoom": False,
                                      "displayModeBar": "hover",
                                      "toImageButtonOptions": {"format": "png", "scale": 2,
                                                               "filename": filename}}),
                    type="default", color=_BTC_ORANGE,
                    delay_show=400,     # skip spinner flash for fast updates
                    show_initially=False,  # never show on initial render (pre-injected figure is already there)
                    overlay_style={"visibility": "visible", "opacity": 0.45},
                ),
                *overlay,
                *badge,
            ]),
            _export_row(graph_id.replace("-graph", "")),
        ], width=9),
    ], className="g-0")


def _chart_tab_layout_with_fab(controls_fn, graph_id, filename):
    """Chart tab with user model input panel + click context menu."""
    import dash_bootstrap_components as dbc

    ctx_menu = html.Div(
        id="um-ctx-menu",
        style={"display": "none", "position": "absolute", "bottom": "14px",
               "left": "14px", "zIndex": 20,
               "backgroundColor": _hex_alpha(CTX_MENU_BG, CTX_MENU_BG_ALPHA), "borderRadius": "8px",
               "padding": "6px 10px", "boxShadow": f"0 2px 12px {_hex_alpha(BLACK, CTX_MENU_SHADOW_ALPHA)}"},
        children=[
            html.Span(id="um-ctx-label",
                      style={"color": USER_MODEL_TRACE, "fontSize": UI_FONT_LG,
                             "fontWeight": "600", "marginRight": "8px"}),
            dbc.Button("P1", id="um-ctx-p1", color="warning", size="sm",
                       outline=True, className="me-1",
                       style={"fontSize": UI_FONT_BASE, "padding": "1px 8px"}),
            dbc.Button("P2", id="um-ctx-p2", color="warning", size="sm",
                       outline=True,
                       style={"fontSize": UI_FONT_BASE, "padding": "1px 8px"}),
        ],
    )

    # Pill bar to switch Price / CAGR views + forward-years input
    view_pills = html.Div([
        dbc.ButtonGroup([
            dbc.Button("Price", id="bub-view-price", color="primary", size="sm"),
            dbc.Button("Forward CAGR", id="bub-view-cagr", outline=True, color="primary", size="sm"),
            dbc.Button("Residuals", id="bub-view-resid", outline=True, color="primary", size="sm"),
        ], size="sm"),
        html.Span(id="bub-cagr-fwd-wrap", style={"display": "none"}, children=[
            dcc.Dropdown(id="bub-cagr-fwd-yrs",
                options=[
                    {"label": "1yr", "value": 1},
                    {"label": "2yr", "value": 2},
                    {"label": "4yr", "value": 4},
                    {"label": "10yr", "value": 10},
                    {"label": "20yr", "value": 20},
                    {"label": "30yr", "value": 30},
                ],
                value=1, clearable=False,
                style={"width": "75px", "fontSize": UI_FONT_BASE,
                       "display": "inline-block", "marginLeft": "8px"}),
        ]),
        dcc.Store(id="bub-view-mode", data="price"),
        dcc.Store(id="bub-cagr-hover-today", data=False),
    ], style={"display": "flex", "alignItems": "center", "justifyContent": "center"},
       className="mb-1")

    return dbc.Row([
        dbc.Col([
            controls_fn(),
        ], width=3, className="controls-col overflow-auto",
                style={"maxHeight": "85vh"}),
        dbc.Col([
            view_pills,
            html.Div(style={"position": "relative"}, children=[
                # Price chart (default visible)
                html.Div(id="bub-price-wrap", children=[
                    html.Div(id=f"{graph_id}-chart-wrap",
                             style={"position": "relative"}, children=[
                        # Static preview PNG — visible during Plotly load,
                        # hidden by chart_responsive.js once figure is rendered.
                        # z-index 0 so Plotly naturally paints over it even if
                        # the hide callback never fires (see _chart_tab_layout).
                        html.Img(src="/assets/bubble_preview.png",
                                 id="bubble-preview-img",
                                 className="chart-preview-overlay",
                                 style={"position": "absolute",
                                        "top": 0, "left": 0,
                                        "width": "100%", "height": "100%",
                                        "objectFit": "contain",
                                        "zIndex": 0,
                                        "pointerEvents": "none"}),
                        dcc.Loading(
                            dcc.Graph(id=graph_id, style=_STYLE_GRAPH_H,
                                      config={"scrollZoom": False,
                                              "displayModeBar": "hover",
                                              "toImageButtonOptions": {"format": "png", "scale": 2,
                                                                       "filename": filename}}),
                            type="default", color=_BTC_ORANGE,
                            delay_show=400,
                            show_initially=False,
                            overlay_style={"visibility": "visible", "opacity": 0.45},
                        ),
                    ]),
                ]),
                # CAGR chart (hidden by default)
                html.Div(id="bub-cagr-wrap", style=_STYLE_HIDDEN, children=[
                    html.Div(style={"position": "relative"}, children=[
                        dcc.Graph(id="bub-cagr-graph", style=_STYLE_GRAPH_H,
                                  config={"scrollZoom": False,
                                          "displayModeBar": "hover",
                                          "toImageButtonOptions": {"format": "png", "scale": 2,
                                                                   "filename": "btc_cagr"}}),
                        # Progress bar overlay — shown while CAGR computes
                        html.Div(id="bub-cagr-progress-wrap", style={"display": "none"}, children=[
                            html.Div(style={
                                "position": "absolute", "top": "50%", "left": "50%",
                                "transform": "translate(-50%, -50%)",
                                "width": "260px", "textAlign": "center",
                            }, children=[
                                html.Div("Computing Forward CAGR\u2026",
                                         style={"color": FALLBACK_MODEL_GRAY, "fontSize": UI_FONT_LG, "marginBottom": "6px"}),
                                html.Div(style={
                                    "height": "6px", "borderRadius": "3px",
                                    "background": PROGRESS_TRACK, "overflow": "hidden",
                                }, children=[
                                    html.Div(id="bub-cagr-progress-bar", style={
                                        "height": "100%", "width": "0%",
                                        "background": _BTC_ORANGE, "borderRadius": "3px",
                                        "transition": "width 0.3s linear",
                                    }),
                                ]),
                            ]),
                        ]),
                    ]),
                    dcc.Store(id="bub-cagr-loading", data=False),
                ]),
                # Residuals chart (hidden by default)
                html.Div(id="bub-resid-wrap", style=_STYLE_HIDDEN, children=[
                    dcc.Graph(id="bub-resid-graph", style=_STYLE_GRAPH_H,
                              config={"scrollZoom": False,
                                      "displayModeBar": "hover",
                                      "toImageButtonOptions": {"format": "png", "scale": 2,
                                                               "filename": "btc_residuals"}}),
                ]),
                ctx_menu,
            ]),
            _export_row(graph_id.replace("-graph", "")),
        ], width=9),
    ], className="g-0")


def _inject_initial_figure(layout, graph_id, fig):
    """Walk the layout tree and set the initial figure on the Graph component."""
    if hasattr(layout, 'children'):
        children = layout.children
        if isinstance(children, list):
            for child in children:
                _inject_initial_figure(child, graph_id, fig)
        elif children is not None:
            _inject_initial_figure(children, graph_id, fig)
    if hasattr(layout, 'id') and layout.id == graph_id:
        layout.figure = fig


def _decade_marks(min_yr, max_yr):
    """Slider mark dict with labels only on decade boundaries: '30, '40, ..."""
    return {y: f"'{y % 100:02d}"
            for y in range(min_yr, max_yr + 1) if y % 10 == 0}


def _year_range_slider(prefix, min_yr, max_yr, default_start, default_end):
    """Year range slider with abbreviated tick marks (decade boundaries only)."""
    return dcc.RangeSlider(
        id=f"{prefix}-yr-range", min=min_yr, max=max_yr,
        value=[default_start, default_end], step=1,
        marks=_decade_marks(min_yr, max_yr),
        tooltip={"always_visible": False},
    )


def _freq_warning_modal():
    """Modal shown when user unlocks frequency editing."""
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Frequency Change")),
        dbc.ModalBody([
            html.P("Changing simulation frequency affects the Markov chain transition "
                   "matrix step size.  Higher frequencies (Daily, Weekly) produce more "
                   "steps per year, which:"),
            html.Ul([
                html.Li("Increases Monte Carlo computation cost proportionally"),
                html.Li("May not match pre-computed cache (cached at Monthly only)"),
            ]),
            html.P([html.B("Monthly"), " is recommended for most analyses."],
                   style={"marginBottom": 0}),
        ]),
        dbc.ModalFooter(
            dbc.Button("OK", id="freq-warning-ok", size="sm", color="primary")),
    ], id="freq-warning-modal", is_open=False, centered=True)


def _global_lppl_modal():
    """Root-level modal holding the n_freqs/weighted/no_13 controls.

    Rendered once in _serve_layout; opened by any tab's
    {prefix}-lppl-configure-btn click.
    """
    return dbc.Modal([
        _modal_header_with_info_link("LPPL Model Configuration", "lppl", "lppl-info-link"),
        dbc.ModalBody([
            _lbl("Oscillation frequencies (N)"),
            dcc.Checklist(id="lppl-n-freqs",
                          options=[
                              {"label": " LPPL\u2081 (1 freq)", "value": 1},
                              {"label": " LPPL\u2082 (2 freqs)", "value": 2},
                              {"label": " LPPL\u2083 (3 freqs) \u2014 recommended",
                               "value": 3},
                              {"label": " LPPL\u2084 (4 freqs) \u2014 \u26A0 likely overfit",
                               "value": 4},
                          ],
                          value=[3],
                          labelStyle={"display": "block"},
                          inputStyle=_CB_MARGIN),
            html.Hr(style={"margin": "6px 0", "borderColor": MODAL_DIVIDER_DARK}),
            dcc.Checklist(id="lppl-weighted",
                          options=[{"label": " Log-time weighted fits",
                                    "value": "weighted"}],
                          value=[], inputStyle=_CB_MARGIN,
                          className="small"),
            html.Small("Emphasizes early-history structure over recent era",
                       style=_STYLE_HINT),
            dcc.Checklist(id="lppl-no-13",
                          options=[{"label": " Exclude \u03c9\u224813 intermod (disables LPPL\u2083)",
                                    "value": "no13"}],
                          value=[], inputStyle=_CB_MARGIN,
                          className="small"),
            html.Small("LP\u2084's \u03c9\u224813 may be an intermodulation artifact",
                       style=_STYLE_HINT),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="lppl-modal-close-btn",
                       size="sm", color="primary"),
        ),
    ], id="lppl-config-modal", is_open=False, centered=True, size="md")


def _two_freq_model_slot(family, slot, damping_label):
    """One model-slot (A or B) inside a two-frequency PPL family config modal.

    family: ID prefix ("hybppl" or "eppl").
    slot: "a" or "b".
    damping_label: displayed-label text for the damped option (differs between
        HybPPL "damped" and EPPL "entropy damped"); value remains "d".
    """
    s = slot
    children = []

    if s == "b":
        children.append(dcc.Checklist(
            id=f"{family}-cfg-b-enabled",
            options=[{"label": " Enable Model B (comparison)", "value": "yes"}],
            value=[], inputStyle=_CB_MARGIN,
        ))
        children.append(html.Hr(style={"margin": "6px 0", "borderColor": MODAL_DIVIDER_DARK}))

    def _damping_radio(freq_key, default_value):
        """Build damping RadioItems for one log/cal frequency."""
        return dcc.RadioItems(
            id=f"{family}-cfg-{s}-{freq_key}",
            options=[{"label": f" {damping_label}", "value": "d"},
                     {"label": " undamped", "value": "u"}],
            value=default_value, inline=True, inputStyle=_CB_MARGIN,
        )

    children.extend([
        _lbl("Log-periodic frequencies"),
        dcc.RadioItems(
            id=f"{family}-cfg-{s}-nlog",
            options=[{"label": " 0", "value": 0},
                     {"label": " 1", "value": 1},
                     {"label": " 2", "value": 2}],
            value=1 if s == "a" else 0,
            inline=True,
            inputStyle=_CB_MARGIN,
        ),
        _lbl("Calendar frequencies"),
        dcc.RadioItems(
            id=f"{family}-cfg-{s}-ncal",
            options=[{"label": " 0", "value": 0},
                     {"label": " 1", "value": 1},
                     {"label": " 2", "value": 2}],
            value=1 if s == "a" else 0,
            inline=True,
            inputStyle=_CB_MARGIN,
        ),
        # Damping controls (visibility toggled by callback)
        html.Div(id=f"{family}-cfg-{s}-log1d-wrap", children=[
            _lbl("Log freq 1 damping"),
            _damping_radio("log1d", "d"),
        ], style=_STYLE_HIDDEN if (s == "b") else {}),
        html.Div(id=f"{family}-cfg-{s}-log2d-wrap", children=[
            _lbl("Log freq 2 damping"),
            _damping_radio("log2d", "d"),
        ], style=_STYLE_HIDDEN),
        html.Div(id=f"{family}-cfg-{s}-cal1d-wrap", children=[
            _lbl("Cal freq 1 damping"),
            _damping_radio("cal1d", "u"),
        ], style=_STYLE_HIDDEN if (s == "b") else {}),
        html.Div(id=f"{family}-cfg-{s}-cal2d-wrap", children=[
            _lbl("Cal freq 2 damping"),
            _damping_radio("cal2d", "u"),
        ], style=_STYLE_HIDDEN),
        html.Div([
            html.Span(id=f"{family}-cfg-{s}-status",
                      style={"fontSize": UI_FONT_MD, "color": FALLBACK_MODEL_GRAY}),
            dcc.Link(id=f"{family}-cfg-{s}-info-link", href="#", refresh=False,
                     style={"fontSize": UI_FONT_MD, "marginLeft": "6px",
                            "color": LINK, "display": "none"},
                     children=_INFO_ICON),
        ], style={"marginTop": "6px"}),
    ])
    return html.Div(children, style={"flex": "1", "minWidth": "200px"})


def _hybppl_model_slot(slot):
    return _two_freq_model_slot("hybppl", slot, damping_label="damped")


def _eppl_model_slot(slot):
    return _two_freq_model_slot("eppl", slot, damping_label="entropy damped")


def _global_two_freq_modal(family, title):
    """Root-level modal holding frequency/damping controls for a two-frequency PPL family.

    Rendered once in _serve_layout; opened by any tab's
    {prefix}-{family}-configure-btn click.
    """
    return dbc.Modal([
        _modal_header_with_info_link(title, family, f"{family}-info-link"),
        dbc.ModalBody([
            html.Div([
                html.Div([
                    html.H6("Model A", style={"fontWeight": "600", "marginBottom": "8px"}),
                    _two_freq_model_slot(family, "a", "damped" if family == "hybppl" else "entropy damped"),
                ], style={"flex": "1", "minWidth": "220px", "paddingRight": "12px"}),
                html.Div(style={"width": "1px", "backgroundColor": MODAL_DIVIDER_DARK,
                                "margin": "0 8px"}),
                html.Div([
                    html.H6("Model B", style={"fontWeight": "600", "marginBottom": "8px",
                                               "color": FALLBACK_MODEL_GRAY}),
                    _two_freq_model_slot(family, "b", "damped" if family == "hybppl" else "entropy damped"),
                ], style={"flex": "1", "minWidth": "220px", "paddingLeft": "12px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px"}),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id=f"{family}-modal-close-btn",
                       size="sm", color="primary"),
        ),
    ], id=f"{family}-config-modal", is_open=False, centered=True, size="lg")


def _global_hybppl_modal():
    return _global_two_freq_modal("hybppl", "Hybrid PPL Configuration")


def _global_eppl_modal():
    return _global_two_freq_modal("eppl", "\U0001FAE0 Entropy PPL Configuration")


def _global_bm_modal():
    """Root-level modal holding Bubble Model settings (composite/support + N future).

    Rendered once in _serve_layout; opened by any tab's {prefix}-bm-gear click.
    """
    from tab_defaults import BUBBLE
    return dbc.Modal([
        _modal_header_with_info_link("Bubble Model Configuration", "bub", "bm-info-link"),
        dbc.ModalBody([
            _lbl("Overlays"),
            dcc.Checklist(
                id="bub-bubble-toggles",
                options=[
                    {"label": " Composite", "value": "show_comp"},
                    {"label": " Support",   "value": "show_sup"},
                ],
                value=["show_comp", "show_sup"],
                labelStyle={"display": "block"},
                inputStyle=_CB_MARGIN,
            ),
            html.Hr(style={"margin": "6px 0", "borderColor": MODAL_DIVIDER_DARK}),
            _lbl("N future bubbles"),
            dcc.Slider(
                id="bub-n-future",
                min=0,
                max=_app_ctx.M.n_future_max,
                value=BUBBLE["n_future"],
                step=1, marks=None,
                tooltip={"always_visible": True},
            ),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="bm-modal-close-btn",
                       size="sm", color="primary"),
        ),
    ], id="bm-config-modal", is_open=False, centered=True, size="md")


def _shared_settings_card(prefix, *, amount_id=None, amount_label="Purchase amount ($)",
                          amount_default=100, infl_default=0, stack_default=0,
                          freq_default="Monthly"):
    """Shared settings panel — controls used by both QR and MC models."""
    children = [
        _lbl("Starting BTC"),
        dbc.Input(id=f"{prefix}-stack", type="number", value=stack_default,
                  min=0, step=0.001, size="sm", debounce=True),
        _use_lots_checklist(prefix),
    ]
    if amount_id:
        children.extend([
            _lbl(amount_label),
            dbc.Input(id=amount_id, type="number", value=amount_default,
                      min=0, max=_app_ctx.MAX_USD, step=1, size="sm",
                      debounce=True),
        ])
    children.extend([
        _lbl("Frequency"),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id=f"{prefix}-freq",
                    options=["Daily", "Weekly", "Monthly", "Quarterly", "Annually"],
                    value=freq_default, clearable=False, disabled=True,
                ),
                width=8,
            ),
            dbc.Col(
                dcc.Checklist(
                    id=f"{prefix}-freq-unlock",
                    options=[{"label": " Unlock", "value": "yes"}],
                    value=[], inputStyle=_CB_MARGIN,
                    style={"fontSize": UI_FONT_MD, "paddingTop": "6px"},
                ),
                width=4,
            ),
        ], className="g-1"),
        _lbl("Inflation rate (0\u2013100% / yr)"),
        dbc.Input(id=f"{prefix}-infl", type="number", value=infl_default,
                  min=0, max=100, step=0.5, size="sm", debounce=True),
    ])
    return _section_card("Your Scenario", *children)


# ── Tab hints (LT-7: collapsible "How to use this tab") ───────────────────────

_TAB_HINTS = {
    "bubble": [
        "Select quantiles (left panel) to see projection channels on the chart.",
        "Toggle between Log and Linear scale to see different perspectives.",
        "Enable 'Show Data' to overlay historical BTC prices.",
        "Adjust 'N Future Bubbles' to extrapolate the bubble model forward.",
        "Enter your BTC stack to see projected USD values in the legend.",
    ],
    "heatmap": [
        "Configure your Quantile Regression model or Markov Simulation.",
        "Configure CAGR heatmap using the chart configuration tab below.",
    ],
    "dca": [
        "Set your periodic DCA amount and frequency (e.g., $100/month).",
        "Configure your Quantile Regression model or Markov Simulation.",
        "Toggle BTC/USD display to see sat count or dollar value.",
        "Enable Stack-celerator to simulate leveraged accumulation using the chart configuration tab below.",
    ],
    "retire": [
        "Set your withdrawal amount, frequency, and inflation rate.",
        "Configure your Quantile Regression model or Markov Simulation.",
        "Depletion arrows mark when your stack hits zero under each scenario.",
        "Adjust the year range to zoom into your planning horizon using the chart configuration tab below.",
    ],
    "supercharge": [
        "Mode A: When does your stack run out at a given withdrawal rate?",
        "Mode B: What is max withdrawal for depletion date.",
        "Configure your Quantile Regression model or Markov Simulation.",
        "Delay offsets compare 'start now' vs 'wait N years' strategies.",
        "Enable quantile bands (shade) to see the full range of outcomes using the chart configuration tab below.",
    ],
    "stack": [
        "Add your BTC purchases with price, date, and amount.",
        "All data stays in your browser \u2014 nothing is sent to the server.",
        "Export/import JSON to back up or transfer your lot data.",
        "Your lots appear on DCA/Retire/SC tabs when 'Use lots' is checked.",
    ],
}


def _tab_hints(tab_id):
    """Collapsible 'How to use this tab' section (native <details>/<summary>)."""
    hints = _TAB_HINTS.get(tab_id, [])
    if not hints:
        return html.Div()
    return html.Details([
        html.Summary("How to use this tab",
                     style={"cursor": "pointer", "fontSize": UI_FONT_LG,
                            "color": MUTED_TEXT, "marginBottom": "6px"}),
        html.Ul([html.Li(h, style={"fontSize": UI_FONT_BASE, "color": DIM_TEXT})
                 for h in hints],
                style={"marginBottom": "8px", "paddingLeft": "20px"}),
    ], style={"marginBottom": "10px"})
