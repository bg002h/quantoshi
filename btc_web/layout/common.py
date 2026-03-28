"""Shared layout helpers, style constants, and reusable control builders."""

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx

# ── Reusable style constants ────────────────────────────────────────────────
_STYLE_HIDDEN     = {"display": "none"}
_STYLE_HINT       = {"color": "#888", "display": "block", "marginBottom": "4px"}
_STYLE_GRAPH_H    = {"height": "78vh"}
_STYLE_COLOR_H    = {"height": "28px"}
_STYLE_ADDR_CELL  = {"paddingRight": "12px", "whiteSpace": "nowrap", "verticalAlign": "top"}
_STYLE_ADDR_CODE  = {"wordBreak": "break-all", "fontSize": "11px"}

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

def _q_panel(checklist_id: str, default_value: list, hint: str | None = None):
    """Quantile checklist — static multi-column grid, no collapse."""
    children = []
    if hint:
        children.append(html.Small(hint,
            style=_STYLE_HINT))
    children.append(
        dcc.Checklist(id=checklist_id, options=_q_options(),
                      value=default_value, className="q-panel-grid",
                      inputStyle={"marginRight":"4px"}),
    )
    return _section_card("Projection Quantiles", *children)


# ══════════════════════════════════════════════════════════════════════════════
# Layout helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ctrl_card(*children):
    return dbc.Card(dbc.CardBody(list(children), className="p-2"),
                    className="mb-2 ctrl-card")

_SECTION_ICONS = {
    "Axes & Range": "\U0001F4D0",
    "Display": "\U0001F3A8",
    "Bubble Model": "\U0001F4CA",
    "Projection Quantiles": "\U0001F4C9",
    "Chart Settings": "\u2699\uFE0F",
    "Plan": "\U0001F5D3\uFE0F",
    "Your Scenario": "\u2699\uFE0F",
    "Starting Stack": "\U0001F4E6",
    "Monte Carlo Simulation": "\U0001F3B2",
    "Saved Simulation": "\U0001F4BE",
}

def _section_card(title: str, *children):
    """Control card with a section header title and optional icon."""
    icon = _SECTION_ICONS.get(title, "")
    prefix = f"{icon} " if icon else ""
    return _ctrl_card(html.Div(f"{prefix}{title}", className="ctrl-section-header"), *children)

def _row(*cols):
    return dbc.Row([dbc.Col(c) for c in cols], className="g-1 mb-1")

def _lbl(text: str):
    return html.Label(text, className="form-label mb-0 small")

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
    """Reusable chart toggle checklist (Log Y, Annotate, Legend, Minor grid, Zoom)."""
    opts = [{"label": " Log Y", "value": "log_y"},
            {"label": " Annotate final values", "value": "annotate"},
            {"label": " Discrete steps", "value": "discrete"},
            {"label": " Show legend", "value": "show_legend"},
            {"label": html.Span(" Minor grid", className="minor-grid-opt"),
             "value": "minor_grid"},
            {"label": " Enable chart zoom", "value": "chart_zoom"}]
    return dcc.Checklist(id=f"{prefix}-toggles", options=opts,
                         value=defaults or [], labelStyle={"display": "block"},
                         inputStyle={"marginRight": "5px"})


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

def _chart_tab_layout(controls_fn, graph_id, filename, mc_prefix=None):
    """Standard chart tab: 3-col controls (left) + 9-col graph (right).

    mc_prefix: if set, adds an MC overlay div (e.g. "dca" → "dca-mc-overlay").
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
    return dbc.Row([
        dbc.Col([
            controls_fn(),
        ], width=3, className="controls-col overflow-auto",
                style={"maxHeight": "85vh"}),
        dbc.Col([
            html.Div(id=f"{mc_prefix or graph_id}-chart-wrap",
                     style={"position": "relative"}, children=[
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
    """Chart tab with FAB for user model drawing + confirmation menu + toast."""
    import dash_bootstrap_components as dbc

    fab = html.Button(
        "\u270e",  # ✎ pencil
        id="user-model-fab",
        n_clicks=0,
        style={
            "position": "absolute", "bottom": "14px", "right": "14px",
            "zIndex": 10, "width": "42px", "height": "42px",
            "borderRadius": "50%", "border": "2px solid rgba(255,255,255,0.3)",
            "backgroundColor": "rgba(30,30,40,0.85)",
            "color": "#e67e22", "fontSize": "20px",
            "cursor": "pointer", "display": "flex",
            "alignItems": "center", "justifyContent": "center",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.4)",
            "transition": "all 0.2s ease",
            "lineHeight": "1",
        },
        title="Draw a custom model (click 2 points)",
    )

    confirm_menu = html.Div(
        id="draw-confirm-menu",
        style={"display": "none", "position": "absolute", "bottom": "60px",
               "left": "50%", "transform": "translateX(-50%)", "zIndex": 15,
               "backgroundColor": "rgba(30,30,40,0.95)", "borderRadius": "8px",
               "padding": "8px 12px", "boxShadow": "0 4px 16px rgba(0,0,0,0.5)",
               "whiteSpace": "nowrap"},
        children=[
            dbc.Button("\u2713", id="draw-accept-btn", color="success",
                       size="sm", title="Accept point",
                       style={"fontSize": "14px", "padding": "2px 8px"}),
            dbc.Button("\u21bb", id="draw-adjust-btn", color="warning",
                       size="sm", title="Zoom in to adjust",
                       style={"fontSize": "14px", "padding": "2px 8px"}),
            dbc.Button("\u2715", id="draw-cancel-btn", color="secondary",
                       size="sm", title="Cancel point",
                       style={"fontSize": "14px", "padding": "2px 8px"}),
        ],
    )

    model_menu = html.Div(
        id="draw-model-menu",
        style={"display": "none", "position": "absolute", "bottom": "60px",
               "left": "50%", "transform": "translateX(-50%)", "zIndex": 15,
               "backgroundColor": "rgba(30,30,40,0.95)", "borderRadius": "8px",
               "padding": "8px 12px", "boxShadow": "0 4px 16px rgba(0,0,0,0.5)",
               "whiteSpace": "nowrap"},
        children=[
            dbc.Button("\u270e Redraw", id="draw-redraw-btn", color="warning",
                       size="sm", className="me-2"),
            dbc.Button("\u2715 Delete", id="draw-delete-btn", color="danger",
                       size="sm", className="me-2"),
            dbc.Button("Cancel", id="draw-dismiss-btn", color="secondary",
                       size="sm"),
        ],
    )

    toast = html.Div(
        id="draw-toast",
        style={"display": "none", "position": "absolute", "top": "10px",
               "left": "50%", "transform": "translateX(-50%)", "zIndex": 15,
               "backgroundColor": "rgba(230,126,34,0.9)", "color": "#fff",
               "borderRadius": "6px", "padding": "6px 14px", "fontSize": "13px",
               "fontWeight": "600", "whiteSpace": "nowrap",
               "pointerEvents": "none"},
        children="Tap two points to define your model",
    )

    return dbc.Row([
        dbc.Col([
            controls_fn(),
        ], width=3, className="controls-col overflow-auto",
                style={"maxHeight": "85vh"}),
        dbc.Col([
            # Menus are OUTSIDE chart-wrap so clicks can't trigger Plotly clickData
            html.Div(style={"position": "relative"}, children=[
                html.Div(id=f"{graph_id}-chart-wrap",
                         style={"position": "relative"}, children=[
                    dcc.Loading(
                        dcc.Graph(id=graph_id, style=_STYLE_GRAPH_H,
                                  config={"scrollZoom": False,
                                          "displayModeBar": "hover",
                                          "toImageButtonOptions": {"format": "png", "scale": 2,
                                                                   "filename": filename}}),
                        type="default", color=_BTC_ORANGE,
                    ),
                    fab,
                    toast,
                ]),
                # Positioned over the chart but outside the Plotly DOM tree
                confirm_menu,
                model_menu,
            ]),
            _export_row(graph_id.replace("-graph", "")),
        ], width=9),
    ], className="g-0")


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


def _model_show_checklist(prefix):
    """Display models checklist (QR / MC / PL / S2F) for Chart section."""
    opts = [
        {"label": " Bubble Model", "value": "bub"},
    ]
    if _app_ctx._HAS_MARKOV:
        opts.append({"label": " MC Simulation", "value": "mc"})
    for mdl in _app_ctx.PRICE_MODELS.values():
        if mdl.short_name in ("bub", "s2f"):
            continue  # bub already added above; s2f not quantized
        opts.append({"label": f" {mdl.name}", "value": mdl.short_name})
    return [
        _lbl("Display models"),
        dcc.Checklist(id=f"{prefix}-model-show",
                      options=opts,
                      value=["bub"],
                      inline=True,
                      inputStyle={"marginRight": "4px"},
                      labelStyle={"marginRight": "12px", "fontSize": "11px"},
                      style={"marginBottom": "8px"}),
    ]


def _shared_settings_card(prefix, *, amount_id=None, amount_label="Per-period amount ($)",
                          amount_default=100, infl_default=0, stack_default=0,
                          freq_default="Monthly"):
    """Shared settings panel — controls used by both QR and MC models."""
    children = [
        _lbl("Starting BTC"),
        dbc.Input(id=f"{prefix}-stack", type="number", value=stack_default,
                  min=0, step=0.001, size="sm", debounce=True),
        dcc.Checklist(id=f"{prefix}-use-lots",
                      options=[{"label": " Use Stack Tracker lots", "value": "yes"}],
                      value=[], inputStyle={"marginRight": "5px"}),
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
                    value=[], inputStyle={"marginRight": "4px"},
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
