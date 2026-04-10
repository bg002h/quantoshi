"""Shared layout helpers, style constants, and reusable control builders."""

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx


# ── Model Info deep-link helper ────────────────────────────────────────────

_INFO_ICON = "\U0001F4D0"  # 📐 same as Model Info tab

def _model_info_link(short_name):
    """Return (href, exists) for the Model Info deep-link for a model.

    Looks up 'mi-{short_name}' in _MODEL_INFO_ITEMS to find the 1-indexed
    position. Returns ("/8.N", True) if found, ("", False) if not.
    Tolerant of reordering — computed at layout time from the live list.
    """
    # Lazy import to avoid circular dependency
    from callbacks.routing import _MODEL_INFO_ITEMS
    mi_key = f"mi-{short_name}"
    try:
        idx = _MODEL_INFO_ITEMS.index(mi_key) + 1
        return f"/8.{idx}", True
    except ValueError:
        return "", False

# ── Reusable style constants ────────────────────────────────────────────────
_STYLE_HIDDEN     = {"display": "none"}
_STYLE_HINT       = {"color": "#888", "display": "block", "marginBottom": "4px"}
_STYLE_GRAPH_H    = {"height": "78vh"}
_STYLE_COLOR_H    = {"height": "28px"}
_STYLE_ADDR_CELL  = {"paddingRight": "12px", "whiteSpace": "nowrap", "verticalAlign": "top"}
_STYLE_ADDR_CODE  = {"wordBreak": "break-all", "fontSize": "11px"}
_CB_MARGIN        = {"marginRight": "4px"}
_INFL_LABEL       = "Inflation rate (0\u2013100% / yr)"
_Q_HINT_BASE      = "Lower quantiles = more conservative price paths."

def _q_options() -> list[dict]:
    opts = []
    for q in _app_ctx._ALL_QS:
        pct = q * 100
        lbl_text = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
        col = _app_ctx.M.qr_colors.get(q, "#888888")
        lbl = html.Span([
            html.Span("\u25CF ", style={"color": col, "fontSize": "10px"}),
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


def _palette_selector():
    """Palette selector widget for bottom of tab control panels.

    Syncs with palette-store via clientside callbacks in nav.py.
    Uses palette-select-tab ID (distinct from navbar's palette-select).
    """
    return _ctrl_card(
        html.Div([
            html.Small("\U0001f3a8 Color palette", className="text-muted me-2"),
            dbc.Select(
                id="palette-select-tab",
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

_SECTION_ICONS = {
    # Shared chart-tab sections
    "Axes & Range": "\U0001F4D0",            # 📐
    "Display": "\U0001F3A8",                 # 🎨
    "Projection Quantiles": "\U0001F4C9",    # 📉
    "Model Scanner": "\U0001F50D",           # 🔍
    "Data Point Appearance": "\u2728",       # ✨
    "User Model (U\u2081)": "\U0001F3AF",    # 🎯
    # Model-config panels
    "Bubble Model": "\U0001F4CA",            # 📊
    "Component Decomposition": "\U0001F9EC", # 🧬
    "LPPL Models": "\U0001F30A",             # 🌊
    "Hybrid PPL Models": "\U0001F39B\uFE0F",  # 🎛️
    "Monte Carlo Simulation": "\U0001F3B2",  # 🎲
    "Saved Simulation": "\U0001F4BE",        # 💾
    # Citadel tab
    "Chart Settings": "\u2699\uFE0F",        # ⚙️
    "Plan": "\U0001F5D3\uFE0F",              # 🗓️
    "Your Scenario": "\u2699\uFE0F",         # ⚙️
    "Bitcoin Stack": "\u20BF",               # ₿
    "Stack (BTC)": "\u20BF",                 # ₿
    "Starting Stack": "\U0001F4E6",          # 📦 (legacy key)
    "Cash Account": "\U0001F4B5",            # 💵
    "Reserve Fund \u2014 US Treasuries": "\U0001F3DB\uFE0F",  # 🏛️
    "Investment Account": "\U0001F4C8",      # 📈
    "Monthly Spending": "\U0001F4B8",        # 💸
    "Account Floor Rules": "\U0001F6E1\uFE0F",  # 🛡️
    "Global Lump Cooldown": "\u23F1\uFE0F",  # ⏱️
    "Saylor Citadel Fortifier": "\U0001F3F0", # 🏰
    # Stack tracker
    "Add Lot": "\u2795",                     # ➕
    "Export / Import": "\U0001F4C1",         # 📁
}

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
    icon = _SECTION_ICONS.get(title, "")
    prefix = f"{icon} " if icon else ""
    title_span = html.Span(f"{prefix}{title}")
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


def _plot_appearance_controls(prefix: str):
    """Trace width + grid width/color controls for a chart tab.

    Each tab gets its own copy of these controls (IDs prefixed). All copies
    read from and write to the same global 'plot-appearance' localStorage
    store, so changes propagate across tabs.

    Returns a list of components to be spread into a _section_card.
    The store and the reset button are only rendered for the 'bub' prefix
    to avoid duplicate IDs — bubble tab is the canonical location.
    """
    parts = [
        _row(
            html.Div([_lbl("Trace thickness (0.5\u20138)"),
                      dbc.Input(id=f"{prefix}-plot-trace-width", type="number",
                                value=2.5, min=0.5, max=8, step=0.5, size="sm")]),
            html.Div([_lbl("Major grid width (0\u20134)"),
                      dbc.Input(id=f"{prefix}-plot-grid-major-width", type="number",
                                value=1.0, min=0, max=4, step=0.25, size="sm")]),
        ),
        _row(
            html.Div([_lbl("Major grid color"),
                      dbc.Input(id=f"{prefix}-plot-grid-major-color", type="color",
                                value="#888888", size="sm",
                                style={"height": "30px", "padding": "2px"})]),
            html.Div([_lbl("Minor grid width (0\u20133)"),
                      dbc.Input(id=f"{prefix}-plot-grid-minor-width", type="number",
                                value=0.8, min=0, max=3, step=0.25, size="sm")]),
        ),
        _row(
            html.Div([_lbl("Minor grid color"),
                      dbc.Input(id=f"{prefix}-plot-grid-minor-color", type="color",
                                value="#B0B0B0", size="sm",
                                style={"height": "30px", "padding": "2px"})]),
            html.Div([_lbl("Bubble Model color"),
                      dbc.Input(id=f"{prefix}-plot-bm-color", type="color",
                                value="#C8960C", size="sm",
                                style={"height": "30px", "padding": "2px"})]),
        ),
        html.Div(
            dbc.Button("Reset to defaults",
                       id=f"{prefix}-plot-appearance-reset",
                       size="sm", color="secondary",
                       outline=True,
                       className="mt-2",
                       style={"fontSize": "11px", "width": "100%"}),
            style={"marginTop": "8px"},
        ),
    ]
    # Only the bubble tab owns the store (shared across all tabs via localStorage)
    if prefix == "bub":
        parts.append(dcc.Store(id="plot-appearance", storage_type="local", data=None))
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
                 style={"fontSize":"11px", "letterSpacing":"0.02em"}),
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
    # Static placeholder preview image — hidden by chart_responsive.js when
    # the interactive Plotly chart has rendered. Shows users the rough
    # chart shape immediately instead of a blank loading area.
    preview_name = graph_id.replace("-graph", "")
    preview_img = html.Img(
        src=f"/assets/{preview_name}_preview.png",
        id=f"{preview_name}-preview-img",
        className="chart-preview-overlay",
        style={"position": "absolute", "top": 0, "left": 0,
               "width": "100%", "height": "100%",
               "objectFit": "contain", "zIndex": 5,
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
               "backgroundColor": "rgba(30,30,40,0.95)", "borderRadius": "8px",
               "padding": "6px 10px", "boxShadow": "0 2px 12px rgba(0,0,0,0.5)"},
        children=[
            html.Span(id="um-ctx-label",
                      style={"color": "#e67e22", "fontSize": "13px",
                             "fontWeight": "600", "marginRight": "8px"}),
            dbc.Button("P1", id="um-ctx-p1", color="warning", size="sm",
                       outline=True, className="me-1",
                       style={"fontSize": "12px", "padding": "1px 8px"}),
            dbc.Button("P2", id="um-ctx-p2", color="warning", size="sm",
                       outline=True,
                       style={"fontSize": "12px", "padding": "1px 8px"}),
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
                style={"width": "75px", "fontSize": "12px",
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
                        # hidden by chart_responsive.js once figure is rendered
                        html.Img(src="/assets/bubble_preview.png",
                                 id="bubble-preview-img",
                                 className="chart-preview-overlay",
                                 style={"position": "absolute",
                                        "top": 0, "left": 0,
                                        "width": "100%", "height": "100%",
                                        "objectFit": "contain",
                                        "zIndex": 5,
                                        "pointerEvents": "none"}),
                        dcc.Loading(
                            dcc.Graph(id=graph_id, style=_STYLE_GRAPH_H,
                                      config={"scrollZoom": False,
                                              "displayModeBar": "hover",
                                              "toImageButtonOptions": {"format": "png", "scale": 2,
                                                                       "filename": filename}}),
                            type="default", color=_BTC_ORANGE,
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
                                         style={"color": "#888", "fontSize": "13px", "marginBottom": "6px"}),
                                html.Div(style={
                                    "height": "6px", "borderRadius": "3px",
                                    "background": "#e0e0e0", "overflow": "hidden",
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


def _year_range_slider(prefix, min_yr, max_yr, default_start, default_end, mark_step=5):
    """Year range slider with abbreviated tick marks."""
    return dcc.RangeSlider(
        id=f"{prefix}-yr-range", min=min_yr, max=max_yr,
        value=[default_start, default_end], step=1,
        marks={y: f"'{y % 100:02d}" for y in range(min_yr, max_yr + 1, mark_step)},
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


def _model_show_checklist(prefix, standardized=False, include_mc=False):
    """Display models checklist with palette-aware color swatches.

    standardized=True: emits single "LPPL" master (skip individual LPPL
    family variants), omits Exp + S2F. For tabs 1/3/4/5 (Phase 1) and
    tab 2 (Phase 2) that share the standardized UX.
    """
    mc = _app_ctx.PALETTES["default"]["model_colors"]
    _PROMOTED = ("eppl", "pca", "grdy")
    _DEPRIORITIZED = {"exp", "s2f", "gomp", "bpl", "hyb2l", "hyb2c", "hyb2b", "hyb4d"}
    _LPPL_FAM = {"lppl", "lp2", "lp3", "lp4"} | set(
        _app_ctx.LPPL_FAMILY_HIDDEN_FROM_BUBBLE)

    _INFO_STYLE = {
        "cursor": "pointer", "fontSize": "11px", "marginLeft": "4px",
        "opacity": "0.6", "textDecoration": "none", "color": "#1a6fa8",
    }

    def _swatch(color, label, model_key=None):
        children = [
            html.Span(" ", style={
                "display": "inline-block", "width": "12px", "height": "12px",
                "borderRadius": "2px", "verticalAlign": "middle",
                "marginRight": "4px", "backgroundColor": color,
            }),
            label,
        ]
        if model_key:
            href, exists = _model_info_link(model_key)
            if exists:
                children.append(html.A(
                    _INFO_ICON, href=href,
                    style=_INFO_STYLE,
                    title=f"View {label} details on Model Info tab",
                ))
        return html.Span(children)

    opts = [{"label": _swatch(mc.get("bub", "#000"), "Bubble Model"),
             "value": "bub"}]

    if standardized:
        # Inject master LPPL entry right after Bubble Model.
        opts.append({
            "label": _swatch(mc.get("lppl", "#FF6D00"), "LPPL (family)"),
            "value": "lppl",
        })

    _HYBPPL_FAM = _app_ctx.HYBPPL_FAMILY_HIDDEN
    all_models = [mdl for mdl in _app_ctx.PRICE_MODELS.values()
                  if mdl.short_name not in _app_ctx.MODEL_SENTINELS
                  and mdl.short_name != "bub"
                  and mdl.short_name not in _HYBPPL_FAM
                  and not mdl.short_name.startswith("cfg_")
                  and not mdl.short_name.startswith("ecfg_")]
    if standardized:
        all_models = [m for m in all_models
                      if m.short_name not in _LPPL_FAM
                      and m.short_name not in _DEPRIORITIZED]
        promoted = sorted([m for m in all_models if m.short_name in _PROMOTED],
                          key=lambda m: _PROMOTED.index(m.short_name) if m.short_name in _PROMOTED else 99)
        rest = [m for m in all_models if m.short_name not in _PROMOTED]
        ordered = promoted + rest
    else:
        promoted = sorted([m for m in all_models if m.short_name in _PROMOTED],
                          key=lambda m: _PROMOTED.index(m.short_name) if m.short_name in _PROMOTED else 99)
        mid = [m for m in all_models if m.short_name not in _PROMOTED and m.short_name not in _DEPRIORITIZED]
        dep = [m for m in all_models if m.short_name in _DEPRIORITIZED]
        ordered = promoted + mid + dep
    for mdl in ordered:
        opts.append({
            "label": _swatch(mc.get(mdl.short_name, "#888"), mdl.name,
                              model_key=mdl.short_name),
            "value": mdl.short_name,
        })
    if include_mc and _app_ctx._HAS_MARKOV:
        opts.append({"label": " MC Simulation", "value": "mc"})
    return [
        _lbl("Display models"),
        dcc.Checklist(id=f"{prefix}-model-show",
                      options=opts,
                      value=["bub"] + (["mc"] if include_mc and _app_ctx._HAS_MARKOV else []),
                      inline=True,
                      inputStyle=_CB_MARGIN,
                      labelStyle={"marginRight": "12px", "fontSize": "11px"},
                      style={"marginBottom": "8px"}),
    ]


def _lppl_config_panel(prefix):
    """Compact LPPL sub-panel: activate + summary + modal launcher.

    The actual n_freqs/weighted/no_13 controls live in the global
    modal (_global_lppl_modal) so they have unique IDs. Each tab's
    version here links to that one modal.
    """
    activate = dcc.Checklist(
        id=f"{prefix}-lppl-activate",
        options=[{"label": " Activate", "value": "yes"}],
        value=[], inputStyle=_CB_MARGIN,
        className="model-panel-activate",
    )
    configure_btn = dbc.Button(
        "\u2699\ufe0f",
        id=f"{prefix}-lppl-configure-btn",
        size="sm", color="secondary", outline=True,
        title="Configure LPPL",
        className="model-panel-configure-btn",
    )
    return _section_card(
        "LPPL Models",
        html.Div([
            html.Small("Current: ", style={"color": "#888", "fontSize": "11px"}),
            html.Span(id=f"{prefix}-lppl-summary", children="LPPL\u2083",
                      style={"color": "#FF6D00", "fontSize": "11px",
                             "fontWeight": "600"}),
        ], style={"marginTop": "4px", "marginBottom": "4px"}),
        header_right=[activate, configure_btn],
    )


def _global_lppl_modal():
    """Root-level modal holding the n_freqs/weighted/no_13 controls.

    Rendered once in _serve_layout; opened by any tab's
    {prefix}-lppl-configure-btn click.
    """
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("LPPL Model Configuration")),
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
            html.Hr(style={"margin": "6px 0", "borderColor": "#444"}),
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


def _hybppl_config_panel(prefix):
    """Compact HybPPL sub-panel: activate + summary + modal launcher.

    The actual frequency/damping controls live in the global modal
    (_global_hybppl_modal) so they have unique IDs.  Each tab's
    version here links to that one modal.
    """
    activate = dcc.Checklist(
        id=f"{prefix}-hybppl-activate",
        options=[{"label": " Activate", "value": "yes"}],
        value=[], inputStyle=_CB_MARGIN,
        className="model-panel-activate",
    )
    configure_btn = dbc.Button(
        "\u2699\ufe0f",
        id=f"{prefix}-hybppl-configure-btn",
        size="sm", color="secondary", outline=True,
        title="Configure HybPPL",
        className="model-panel-configure-btn",
    )
    return _section_card(
        "Hybrid PPL Models",
        html.Div([
            html.Small("Current: ", style={"color": "#888", "fontSize": "11px"}),
            html.Span(id=f"{prefix}-hybppl-summary", children="1d+1u",
                      style={"color": "#4A90D9", "fontSize": "11px",
                             "fontWeight": "600"}),
        ], style={"marginTop": "4px", "marginBottom": "4px"}),
        header_right=[activate, configure_btn],
    )


def _hybppl_model_slot(slot):
    """One model-slot (A or B) inside the HybPPL config modal."""
    s = slot  # "a" or "b"
    children = []

    if s == "b":
        children.append(dcc.Checklist(
            id="hybppl-cfg-b-enabled",
            options=[{"label": " Enable Model B (comparison)", "value": "yes"}],
            value=[], inputStyle=_CB_MARGIN,
        ))
        children.append(html.Hr(style={"margin": "6px 0", "borderColor": "#444"}))

    children.extend([
        _lbl("Log-periodic frequencies"),
        dcc.RadioItems(
            id=f"hybppl-cfg-{s}-nlog",
            options=[{"label": " 0", "value": 0},
                     {"label": " 1", "value": 1},
                     {"label": " 2", "value": 2}],
            value=1 if s == "a" else 0,
            inline=True,
            inputStyle=_CB_MARGIN,
        ),
        _lbl("Calendar frequencies"),
        dcc.RadioItems(
            id=f"hybppl-cfg-{s}-ncal",
            options=[{"label": " 0", "value": 0},
                     {"label": " 1", "value": 1},
                     {"label": " 2", "value": 2}],
            value=1 if s == "a" else 0,
            inline=True,
            inputStyle=_CB_MARGIN,
        ),
        # Damping controls (visibility toggled by callback)
        html.Div(id=f"hybppl-cfg-{s}-log1d-wrap", children=[
            _lbl("Log freq 1 damping"),
            dcc.RadioItems(
                id=f"hybppl-cfg-{s}-log1d",
                options=[{"label": " damped", "value": "d"},
                         {"label": " undamped", "value": "u"}],
                value="d", inline=True, inputStyle=_CB_MARGIN,
            ),
        ], style=_STYLE_HIDDEN if (s == "b") else {}),
        html.Div(id=f"hybppl-cfg-{s}-log2d-wrap", children=[
            _lbl("Log freq 2 damping"),
            dcc.RadioItems(
                id=f"hybppl-cfg-{s}-log2d",
                options=[{"label": " damped", "value": "d"},
                         {"label": " undamped", "value": "u"}],
                value="d", inline=True, inputStyle=_CB_MARGIN,
            ),
        ], style=_STYLE_HIDDEN),
        html.Div(id=f"hybppl-cfg-{s}-cal1d-wrap", children=[
            _lbl("Cal freq 1 damping"),
            dcc.RadioItems(
                id=f"hybppl-cfg-{s}-cal1d",
                options=[{"label": " damped", "value": "d"},
                         {"label": " undamped", "value": "u"}],
                value="u", inline=True, inputStyle=_CB_MARGIN,
            ),
        ], style=_STYLE_HIDDEN if (s == "b") else {}),
        html.Div(id=f"hybppl-cfg-{s}-cal2d-wrap", children=[
            _lbl("Cal freq 2 damping"),
            dcc.RadioItems(
                id=f"hybppl-cfg-{s}-cal2d",
                options=[{"label": " damped", "value": "d"},
                         {"label": " undamped", "value": "u"}],
                value="u", inline=True, inputStyle=_CB_MARGIN,
            ),
        ], style=_STYLE_HIDDEN),
        html.Div([
            html.Span(id=f"hybppl-cfg-{s}-status",
                      style={"fontSize": "11px", "color": "#888"}),
            html.A(id=f"hybppl-cfg-{s}-info-link", href="#",
                   style={"fontSize": "11px", "marginLeft": "6px",
                          "color": "#1a6fa8", "display": "none"},
                   children="\u2139\uFE0F Model Info"),
        ], style={"marginTop": "6px"}),
    ])
    return html.Div(children, style={"flex": "1", "minWidth": "200px"})


def _global_hybppl_modal():
    """Root-level modal holding HybPPL frequency/damping controls.

    Rendered once in _serve_layout; opened by any tab's
    {prefix}-hybppl-configure-btn click.
    """
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Hybrid PPL Configuration")),
        dbc.ModalBody([
            html.Div([
                html.Div([
                    html.H6("Model A", style={"fontWeight": "600", "marginBottom": "8px"}),
                    _hybppl_model_slot("a"),
                ], style={"flex": "1", "minWidth": "220px", "paddingRight": "12px"}),
                html.Div(style={"width": "1px", "backgroundColor": "#444",
                                "margin": "0 8px"}),
                html.Div([
                    html.H6("Model B", style={"fontWeight": "600", "marginBottom": "8px",
                                               "color": "#888"}),
                    _hybppl_model_slot("b"),
                ], style={"flex": "1", "minWidth": "220px", "paddingLeft": "12px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px"}),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="hybppl-modal-close-btn",
                       size="sm", color="primary"),
        ),
    ], id="hybppl-config-modal", is_open=False, centered=True, size="lg")


def _eppl_config_panel(prefix):
    """Compact EPPL sub-panel: activate + summary + modal launcher.

    The actual frequency/damping controls live in the global modal
    (_global_eppl_modal) so they have unique IDs.  Each tab's
    version here links to that one modal.
    """
    activate = dcc.Checklist(
        id=f"{prefix}-eppl-activate",
        options=[{"label": " Activate", "value": "yes"}],
        value=[], inputStyle=_CB_MARGIN,
        className="model-panel-activate",
    )
    configure_btn = dbc.Button(
        "\u2699\ufe0f",
        id=f"{prefix}-eppl-configure-btn",
        size="sm", color="secondary", outline=True,
        title="Configure Entropy PPL",
        className="model-panel-configure-btn",
    )
    return _section_card(
        "Entropy PPL Models",
        html.Div([
            html.Small("Current: ", style={"color": "#888", "fontSize": "11px"}),
            html.Span(id=f"{prefix}-eppl-summary", children="1d+1u",
                      style={"color": "#148C8C", "fontSize": "11px",
                             "fontWeight": "600"}),
        ], style={"marginTop": "4px", "marginBottom": "4px"}),
        header_right=[activate, configure_btn],
    )


def _eppl_model_slot(slot):
    """One model-slot (A or B) inside the EPPL config modal."""
    s = slot  # "a" or "b"
    children = []

    if s == "b":
        children.append(dcc.Checklist(
            id="eppl-cfg-b-enabled",
            options=[{"label": " Enable Model B (comparison)", "value": "yes"}],
            value=[], inputStyle=_CB_MARGIN,
        ))
        children.append(html.Hr(style={"margin": "6px 0", "borderColor": "#444"}))

    children.extend([
        _lbl("Log-periodic frequencies"),
        dcc.RadioItems(
            id=f"eppl-cfg-{s}-nlog",
            options=[{"label": " 0", "value": 0},
                     {"label": " 1", "value": 1},
                     {"label": " 2", "value": 2}],
            value=1 if s == "a" else 0,
            inline=True,
            inputStyle=_CB_MARGIN,
        ),
        _lbl("Calendar frequencies"),
        dcc.RadioItems(
            id=f"eppl-cfg-{s}-ncal",
            options=[{"label": " 0", "value": 0},
                     {"label": " 1", "value": 1},
                     {"label": " 2", "value": 2}],
            value=1 if s == "a" else 0,
            inline=True,
            inputStyle=_CB_MARGIN,
        ),
        # Damping controls (visibility toggled by callback)
        html.Div(id=f"eppl-cfg-{s}-log1d-wrap", children=[
            _lbl("Log freq 1 damping"),
            dcc.RadioItems(
                id=f"eppl-cfg-{s}-log1d",
                options=[{"label": " entropy damped", "value": "d"},
                         {"label": " undamped", "value": "u"}],
                value="d", inline=True, inputStyle=_CB_MARGIN,
            ),
        ], style=_STYLE_HIDDEN if (s == "b") else {}),
        html.Div(id=f"eppl-cfg-{s}-log2d-wrap", children=[
            _lbl("Log freq 2 damping"),
            dcc.RadioItems(
                id=f"eppl-cfg-{s}-log2d",
                options=[{"label": " entropy damped", "value": "d"},
                         {"label": " undamped", "value": "u"}],
                value="d", inline=True, inputStyle=_CB_MARGIN,
            ),
        ], style=_STYLE_HIDDEN),
        html.Div(id=f"eppl-cfg-{s}-cal1d-wrap", children=[
            _lbl("Cal freq 1 damping"),
            dcc.RadioItems(
                id=f"eppl-cfg-{s}-cal1d",
                options=[{"label": " entropy damped", "value": "d"},
                         {"label": " undamped", "value": "u"}],
                value="u", inline=True, inputStyle=_CB_MARGIN,
            ),
        ], style=_STYLE_HIDDEN if (s == "b") else {}),
        html.Div(id=f"eppl-cfg-{s}-cal2d-wrap", children=[
            _lbl("Cal freq 2 damping"),
            dcc.RadioItems(
                id=f"eppl-cfg-{s}-cal2d",
                options=[{"label": " entropy damped", "value": "d"},
                         {"label": " undamped", "value": "u"}],
                value="u", inline=True, inputStyle=_CB_MARGIN,
            ),
        ], style=_STYLE_HIDDEN),
        html.Div([
            html.Span(id=f"eppl-cfg-{s}-status",
                      style={"fontSize": "11px", "color": "#888"}),
            html.A(id=f"eppl-cfg-{s}-info-link", href="#",
                   style={"fontSize": "11px", "marginLeft": "6px",
                          "color": "#1a6fa8", "display": "none"},
                   children="\u2139\uFE0F Model Info"),
        ], style={"marginTop": "6px"}),
    ])
    return html.Div(children, style={"flex": "1", "minWidth": "200px"})


def _global_eppl_modal():
    """Root-level modal holding EPPL frequency/damping controls.

    Rendered once in _serve_layout; opened by any tab's
    {prefix}-eppl-configure-btn click.
    """
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Entropy PPL Configuration")),
        dbc.ModalBody([
            html.Div([
                html.Div([
                    html.H6("Model A", style={"fontWeight": "600", "marginBottom": "8px"}),
                    _eppl_model_slot("a"),
                ], style={"flex": "1", "minWidth": "220px", "paddingRight": "12px"}),
                html.Div(style={"width": "1px", "backgroundColor": "#444",
                                "margin": "0 8px"}),
                html.Div([
                    html.H6("Model B", style={"fontWeight": "600", "marginBottom": "8px",
                                               "color": "#888"}),
                    _eppl_model_slot("b"),
                ], style={"flex": "1", "minWidth": "220px", "paddingLeft": "12px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px"}),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="eppl-modal-close-btn",
                       size="sm", color="primary"),
        ),
    ], id="eppl-config-modal", is_open=False, centered=True, size="lg")


def _shared_settings_card(prefix, *, amount_id=None, amount_label="Purchase amount ($)",
                          amount_default=100, infl_default=0, stack_default=0,
                          freq_default="Monthly"):
    """Shared settings panel — controls used by both QR and MC models."""
    children = [
        _lbl("Starting BTC"),
        dbc.Input(id=f"{prefix}-stack", type="number", value=stack_default,
                  min=0, step=0.001, size="sm", debounce=True),
        dcc.Checklist(id=f"{prefix}-use-lots",
                      options=[{"label": " Use Stack Tracker lots", "value": "yes"}],
                      value=[], inputStyle=_CB_MARGIN),
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
                    style={"fontSize": "11px", "paddingTop": "6px"},
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
                     style={"cursor": "pointer", "fontSize": "13px",
                            "color": "#666", "marginBottom": "6px"}),
        html.Ul([html.Li(h, style={"fontSize": "12px", "color": "#555"})
                 for h in hints],
                style={"marginBottom": "8px", "paddingLeft": "20px"}),
    ], style={"marginBottom": "10px"})
