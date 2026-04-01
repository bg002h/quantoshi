"""Tab 1 — Bubble + QR Overlay layout."""

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from tab_defaults import BUBBLE
from layout.common import (_tab_hints, _section_card, _row, _lbl,
                            _STYLE_HIDDEN, _q_panel, _q_options,
                            _ctrl_card, _legend_pos_dropdown,
                            _chart_tab_layout, _CB_MARGIN)


def _bubble_controls():
    yr_now = pd.Timestamp.today().year
    return html.Div([
        _tab_hints("bubble"),
        _section_card("Axes & Range",
            _row(
                html.Div([_lbl("X scale"), dcc.RadioItems(
                    id="bub-xscale", options=[{"label":"Log","value":"log"},
                                               {"label":"Linear","value":"linear"}],
                    value=BUBBLE["xscale"], inline=True)]),
                html.Div([_lbl("Y scale"), dcc.RadioItems(
                    id="bub-yscale", options=[{"label":"Log","value":"log"},
                                               {"label":"Linear","value":"linear"}],
                    value=BUBBLE["yscale"], inline=True)]),
            ),
            _lbl("X range (year)"),
            dcc.RangeSlider(id="bub-xrange", min=2010, max=2080,
                            value=[2010, 2033], step=1,
                            marks={y: f"'{y % 100:02d}" for y in range(2010, 2081, 10)},
                            tooltip={"always_visible":False}),
            dbc.Row([
                dbc.Col(_lbl("Y range (price)"), width="auto"),
                dbc.Col(dcc.Checklist(
                    id="bub-auto-y",
                    options=[{"label":" Auto","value":"yes"}],
                    value=["yes"], inputStyle=_CB_MARGIN,
                    className="small",
                ), width="auto"),
            ], className="g-0 align-items-center"),
            html.Div(id="bub-yrange-wrap", style=_STYLE_HIDDEN, children=[
                dcc.RangeSlider(id="bub-yrange", min=-2, max=9,
                                value=[-1.5, 6], step=0.5,
                                marks={-2:"1\u00a2", 0:"$1", 2:"$100",
                                        4:"$10K", 6:"$1M", 9:"$1B"},
                                tooltip={"always_visible":False}),
            ]),
        ),
        _section_card("Display",
            dcc.Checklist(id="bub-toggles",
                          options=[{"label":" Shade bands","value":"shade"},
                                   {"label":" Show OLS","value":"show_ols"},
                                   {"label":" Unfairly Cheap Line","value":"show_ucl"},
                                   {"label":" Show data","value":"show_data"},
                                   {"label":" Show today","value":"show_today"},
                                   {"label":" Show legend","value":"show_legend"},
                                   {"label":html.Span(" Minor grid",className="minor-grid-opt"),"value":"minor_grid"},
                                   {"label":" Enable chart zoom","value":"chart_zoom"}],
                          value=["shade","show_data","show_today"],
                          labelStyle={"display":"block"},
                          inputStyle=_CB_MARGIN),
            *_legend_pos_dropdown("bub", BUBBLE["legend_pos"]),
            _lbl("Display models"),
            dcc.Checklist(id="bub-model-show",
                          options=[{"label": " Bubble Model", "value": "bub"}] +
                                  [{"label": f" {mdl.name}", "value": mdl.short_name}
                                   for mdl in _app_ctx.PRICE_MODELS.values()
                                   if mdl.short_name not in _app_ctx.MODEL_SENTINELS
                                   and mdl.short_name != "bub"],
                          value=["bub"], inline=True,
                          inputStyle=_CB_MARGIN,
                          labelStyle={"marginRight": "12px", "fontSize": "11px"},
                          style={"marginBottom": "8px"}),
        ),
        _section_card("Bubble Model",
            _lbl("Bubble"),
            dcc.Checklist(id="bub-bubble-toggles",
                          options=[{"label":" Composite","value":"show_comp"},
                                   {"label":" Support","value":"show_sup"}],
                          value=["show_comp","show_sup"],
                          labelStyle={"display":"block"},
                          inputStyle=_CB_MARGIN),
            _lbl("N future bubbles"),
            dcc.Slider(id="bub-n-future", min=0, max=_app_ctx.M.n_future_max,
                       value=BUBBLE["n_future"], step=1, marks=None,
                       tooltip={"always_visible":True}),
        ),
        _q_panel("bub-qs", [0.5],
                 hint="If none selected, Q50% is shown for active models."),
        _section_card("Model Scanner",
            _row(
                html.Div([
                    _lbl("Price ($)"),
                    dbc.Input(id="scan-price", type="number",
                              placeholder="live", size="sm", debounce=True),
                    html.Small("\u20bf live", id="scan-price-hint",
                               className="text-muted", style={"fontSize": "9px"}),
                ]),
                html.Div([
                    _lbl("Date"),
                    dbc.Input(id="scan-date", type="date",
                              value=pd.Timestamp.today().strftime("%Y-%m-%d"),
                              size="sm", debounce=True),
                ]),
                html.Div([
                    _lbl("Quantile (%)"),
                    dbc.Input(id="scan-q", type="number",
                              min=0.1, max=99.9, step=0.1,
                              size="sm", debounce=True,
                              className="scan-output"),
                ]),
            ),
            dcc.Store(id="scan-output-field", data=["p", "d"]),  # last-2-edited history → q is output
            dcc.Store(id="scan-active-rows", data=[]),
            html.Div(id="scan-results"),
        ),
        _ctrl_card(
            _lbl("Data Point Appearance"),
            _row(
                html.Div([_lbl("Pt size (1\u201320)"),
                          dbc.Input(id="bub-ptsize", type="number",
                                    value=BUBBLE["pt_size"], min=1, max=20, size="sm")]),
                html.Div([_lbl("Alpha (0.1\u20131)"),
                          dbc.Input(id="bub-ptalpha", type="number",
                                    value=BUBBLE["pt_alpha"], min=0.1, max=1.0, step=0.05, size="sm")]),
            ),
        ),
        _ctrl_card(
            _lbl("Stack (BTC)"),
            dbc.InputGroup([
                dbc.Input(id="bub-stack", type="number", value=BUBBLE["stack"],
                          min=0, step=0.001, size="sm", debounce=True),
                dbc.InputGroupText(dcc.Checklist(
                    id="bub-show-stack",
                    options=[{"label":" Show","value":"yes"}],
                    value=[], inputStyle=_CB_MARGIN)),
            ], size="sm"),
            dcc.Checklist(id="bub-use-lots",
                          options=[{"label":" Use Stack Tracker lots","value":"yes"}],
                          value=[], inputStyle=_CB_MARGIN),
        ),
        _section_card("User Model",
            html.Small("Click a data point on the chart, then tap P1 or P2.",
                       style={"color":"#888", "fontSize":"11px"}),
            html.Div([
                html.Label("Point 1", style={"fontWeight":"600", "fontSize":"12px", "marginTop":"6px"}),
                html.Div([
                    _lbl("Year"), html.Span(id="um-p1-year-display",
                        style={"display":"inline-block", "width":"80px", "fontSize":"12px",
                               "fontWeight":"600", "color":"#e67e22",
                               "backgroundColor":"rgba(230,126,34,0.1)",
                               "border":"1px solid rgba(230,126,34,0.3)",
                               "borderRadius":"4px", "padding":"2px 6px",
                               "minHeight":"24px", "verticalAlign":"middle"},
                        children="\u2014"),
                    html.Span(" ", style={"width":"6px", "display":"inline-block"}),
                    _lbl("$"), html.Span(id="um-p1-price-display",
                        style={"display":"inline-block", "width":"100px", "fontSize":"12px",
                               "fontWeight":"600", "color":"#e67e22",
                               "backgroundColor":"rgba(230,126,34,0.1)",
                               "border":"1px solid rgba(230,126,34,0.3)",
                               "borderRadius":"4px", "padding":"2px 6px",
                               "minHeight":"24px", "verticalAlign":"middle"},
                        children="\u2014"),
                ], style={"display":"flex", "alignItems":"center", "gap":"4px", "flexWrap":"wrap"}),
                html.Label("Point 2", style={"fontWeight":"600", "fontSize":"12px", "marginTop":"6px"}),
                html.Div([
                    _lbl("Year"), html.Span(id="um-p2-year-display",
                        style={"display":"inline-block", "width":"80px", "fontSize":"12px",
                               "fontWeight":"600", "color":"#e67e22",
                               "backgroundColor":"rgba(230,126,34,0.1)",
                               "border":"1px solid rgba(230,126,34,0.3)",
                               "borderRadius":"4px", "padding":"2px 6px",
                               "minHeight":"24px", "verticalAlign":"middle"},
                        children="\u2014"),
                    html.Span(" ", style={"width":"6px", "display":"inline-block"}),
                    _lbl("$"), html.Span(id="um-p2-price-display",
                        style={"display":"inline-block", "width":"100px", "fontSize":"12px",
                               "fontWeight":"600", "color":"#e67e22",
                               "backgroundColor":"rgba(230,126,34,0.1)",
                               "border":"1px solid rgba(230,126,34,0.3)",
                               "borderRadius":"4px", "padding":"2px 6px",
                               "minHeight":"24px", "verticalAlign":"middle"},
                        children="\u2014"),
                ], style={"display":"flex", "alignItems":"center", "gap":"4px", "flexWrap":"wrap"}),
            ]),
            # Hidden inputs (callbacks write to these; displays mirror them)
            dcc.Store(id="um-p1-year", data=None),
            dcc.Store(id="um-p1-price", data=None),
            dcc.Store(id="um-p2-year", data=None),
            dcc.Store(id="um-p2-price", data=None),
            html.Div([
                dbc.Button("\u270e Draw", id="um-draw-btn", color="warning", size="sm",
                           className="me-2", style={"marginTop":"8px"}),
                dbc.Button("\u2715 Delete", id="um-delete-btn", color="secondary", size="sm",
                           outline=True, style={"marginTop":"8px"}),
            ]),
        ),
    ])


def _bubble_tab():
    from layout.common import _chart_tab_layout_with_fab
    return _chart_tab_layout_with_fab(_bubble_controls, "bubble-graph", "btc_bubble")
