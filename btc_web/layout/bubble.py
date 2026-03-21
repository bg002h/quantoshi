"""Tab 1 — Bubble + QR Overlay layout."""

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from layout.common import (_tab_hints, _section_card, _row, _lbl,
                            _STYLE_HIDDEN, _q_panel, _q_options,
                            _ctrl_card, _legend_pos_dropdown,
                            _chart_tab_layout)


def _bubble_controls():
    yr_now = pd.Timestamp.today().year
    return html.Div([
        _tab_hints("bubble"),
        _section_card("Axes & Range",
            _row(
                html.Div([_lbl("X scale"), dcc.RadioItems(
                    id="bub-xscale", options=[{"label":"Log","value":"log"},
                                               {"label":"Linear","value":"linear"}],
                    value="log", inline=True)]),
                html.Div([_lbl("Y scale"), dcc.RadioItems(
                    id="bub-yscale", options=[{"label":"Log","value":"log"},
                                               {"label":"Linear","value":"linear"}],
                    value="log", inline=True)]),
            ),
            _lbl("X range (year)"),
            dcc.RangeSlider(id="bub-xrange", min=2010, max=2080,
                            value=[2012, yr_now + 4], step=1,
                            marks={y: f"'{y % 100:02d}" for y in range(2010, 2081, 10)},
                            tooltip={"always_visible":False}),
            dbc.Row([
                dbc.Col(_lbl("Y range (price)"), width="auto"),
                dbc.Col(dcc.Checklist(
                    id="bub-auto-y",
                    options=[{"label":" Auto","value":"yes"}],
                    value=["yes"], inputStyle={"marginRight":"3px"},
                    className="small",
                ), width="auto"),
            ], className="g-0 align-items-center"),
            html.Div(id="bub-yrange-wrap", style=_STYLE_HIDDEN, children=[
                dcc.RangeSlider(id="bub-yrange", min=-2, max=9,
                                value=[0, 7], step=0.5,
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
                          inputStyle={"marginRight":"5px"}),
            *_legend_pos_dropdown("bub", "top-left"),
            _lbl("Display models"),
            dcc.Checklist(id="bub-model-show",
                          options=[{"label": f" {mdl.name}", "value": mdl.short_name}
                                   for mdl in _app_ctx.PRICE_MODELS.values()
                                   if mdl.short_name not in _app_ctx.MODEL_SENTINELS],
                          value=["bub"], inline=True,
                          inputStyle={"marginRight": "4px"},
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
                          inputStyle={"marginRight":"5px"}),
            _lbl("N future bubbles"),
            dcc.Slider(id="bub-n-future", min=0, max=_app_ctx.M.n_future_max,
                       value=3, step=1, marks=None,
                       tooltip={"always_visible":True}),
        ),
        _q_panel("bub-qs", [0.5]),
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
                                    value=3, min=1, max=20, size="sm")]),
                html.Div([_lbl("Alpha (0.1\u20131)"),
                          dbc.Input(id="bub-ptalpha", type="number",
                                    value=0.3, min=0.1, max=1.0, step=0.05, size="sm")]),
            ),
        ),
        _ctrl_card(
            _lbl("Stack (BTC)"),
            dbc.InputGroup([
                dbc.Input(id="bub-stack", type="number", value=0,
                          min=0, step=0.001, size="sm", debounce=True),
                dbc.InputGroupText(dcc.Checklist(
                    id="bub-show-stack",
                    options=[{"label":" Show","value":"yes"}],
                    value=[], inputStyle={"marginRight":"4px"})),
            ], size="sm"),
            dcc.Checklist(id="bub-use-lots",
                          options=[{"label":" Use Stack Tracker lots","value":"yes"}],
                          value=[], inputStyle={"marginRight":"5px"}),
        ),
    ])


def _bubble_tab():
    return _chart_tab_layout(_bubble_controls, "bubble-graph", "btc_bubble")
