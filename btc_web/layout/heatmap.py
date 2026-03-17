"""Tab 2 — CAGR Heatmap layout."""

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from layout.common import (_tab_hints, _section_card, _lbl, _row, _export_row,
                            _STYLE_HIDDEN, _STYLE_HINT, _STYLE_GRAPH_H,
                            _STYLE_COLOR_H, _BTC_ORANGE,
                            _q_options, _model_show_checklist)
from layout.mc_controls import _mc_controls


def _heatmap_controls():
    yr_now = pd.Timestamp.today().year
    return html.Div([
        _tab_hints("heatmap"),
        # ── Quantile Regression Model ──────────────────────────────────
        _section_card("Quantile Regression Model",
            html.Small("Select quantiles to follow.",
                style=_STYLE_HINT),
            html.Small("Select exit quantiles for CAGR projection columns.",
                style=_STYLE_HINT),
            dcc.Checklist(id="hm-exit-qs", options=_q_options(),
                          value=_app_ctx._DEF_QS, className="q-panel-grid",
                          inputStyle={"marginRight":"4px"}),
            html.Hr(className="my-1"),
            _lbl("Entry year"),
            dcc.Slider(id="hm-entry-yr", min=2010, max=2039,
                       value=yr_now, step=1, marks=None,
                       tooltip={"always_visible":True}),
            _lbl("Entry percentile (0.1\u201399.9%)"),
            dbc.Input(id="hm-entry-q", type="number",
                      value=_app_ctx._HM_ENTRY_Q_DEFAULT,
                      min=0.1, max=99.9, step=0.1, size="sm",
                      debounce=True),
            _lbl("Starting BTC (for portfolio display)"),
            dbc.Input(id="hm-stack", type="number", value=0,
                      min=0, step=0.001, size="sm", debounce=True),
            dcc.Checklist(id="hm-use-lots",
                          options=[{"label": " Use Stack Tracker lots", "value": "yes"}],
                          value=[], inputStyle={"marginRight": "5px"}),
        ),
        _mc_controls("hm", show_amount=True, show_inflation=True,
                     show_stack=True, show_mc_entry_q=True, default_entry_q=10,
                     shared_controls={"stack"}),
        # ── Chart ───────────────────────────────────────────────────────
        _section_card("Chart Settings",
            *_model_show_checklist("hm"),
            _lbl("Exit year range"),
            dcc.RangeSlider(id="hm-exit-range", min=2010, max=2060,
                            value=[yr_now, yr_now + 15], step=1,
                            marks={y: f"'{y % 100:02d}" for y in range(2010, 2061, 5)},
                            tooltip={"always_visible":False}),
            _lbl("Color mode"),
            dcc.RadioItems(id="hm-mode",
                           options=[{"label":" Segmented","value":0},
                                    {"label":" Data-Scaled","value":1},
                                    {"label":" Diverging","value":2}],
                           value=0, labelStyle={"display":"block"},
                           inputStyle={"marginRight":"5px"}),
            _lbl("Break 1 (CAGR %, integer)"),
            dbc.Input(id="hm-b1", type="number", value=0,
                      step=1, size="sm"),
            _lbl("Break 2 (CAGR %, integer)"),
            dbc.Input(id="hm-b2", type="number", value=20,
                      step=1, size="sm"),
            _lbl("Color palette"),
            dcc.Dropdown(id="hm-palette",
                options=[
                    {"label": "Forge (dark \u2192 gold)",      "value": "forge"},
                    {"label": "Thermal (blue \u2192 red)",     "value": "thermal"},
                    {"label": "Bitcoin (dark \u2192 orange)",  "value": "bitcoin"},
                    {"label": "Ocean (navy \u2192 cyan)",      "value": "ocean"},
                    {"label": "Monochrome (gray)",        "value": "mono"},
                    {"label": "Custom",                   "value": "custom"},
                ],
                value="forge", clearable=False),
            _row(
                html.Div([_lbl("Lo"), dbc.Input(id="hm-c-lo", type="color",
                           value="#1b0a2e", style=_STYLE_COLOR_H)]),
                html.Div([_lbl("Mid1"), dbc.Input(id="hm-c-mid1", type="color",
                           value="#2c2c3a", style=_STYLE_COLOR_H)]),
                html.Div([_lbl("Mid2"), dbc.Input(id="hm-c-mid2", type="color",
                           value="#1b4332", style=_STYLE_COLOR_H)]),
                html.Div([_lbl("Hi"), dbc.Input(id="hm-c-hi", type="color",
                           value="#ffd700", style=_STYLE_COLOR_H)]),
            ),
            _lbl("Gradient steps"),
            dbc.Input(id="hm-grad", type="number", value=32,
                      min=2, max=64, step=1, size="sm"),
            html.Div("Cell Text", className="ctrl-section-header mt-2"),
            _lbl("Cell text"),
            dcc.Dropdown(id="hm-vfmt",
                options=[
                    {"label":"CAGR %",            "value":"cagr"},
                    {"label":"Exit Price",          "value":"price"},
                    {"label":"CAGR % + Price",      "value":"both"},
                    {"label":"CAGR % + Portfolio",  "value":"stack"},
                    {"label":"Portfolio Value",     "value":"port_only"},
                    {"label":"Multiple (\u00d7)",        "value":"mult_only"},
                    {"label":"CAGR % + Multiple",   "value":"cagr_mult"},
                    {"label":"Multiple + Portfolio","value":"mult_port"},
                    {"label":"None",                "value":"none"},
                ],
                value="cagr", clearable=False),
            _lbl("Cell font size"),
            dbc.Input(id="hm-cell-fs", type="number", value=9,
                      min=5, max=20, step=1, size="sm"),
            dcc.Checklist(id="hm-toggles",
                          options=[{"label":" Show colorbar","value":"colorbar"},
                                   {"label":" Enable chart zoom","value":"chart_zoom"}],
                          value=["colorbar"], labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
        ),
    ])


def _heatmap_tab():
    return dbc.Row([
        dbc.Col([
            _heatmap_controls(),
        ], width=3, className="controls-col overflow-auto",
                style={"maxHeight":"85vh"}),
        dbc.Col([
            # Swipe indicator (hidden when MC disabled)
            html.Div([
                html.Span("\u25c0 ", style={"opacity":"0.5"}),
                html.Span("Quantile Regression", id="hm-sw-qr-lbl",
                           className="fw-bold", style={"cursor":"pointer"}),
                html.Span("  \u00b7  ", style={"opacity":"0.4"}),
                html.Span("Monte Carlo", id="hm-sw-mc-lbl",
                           style={"cursor":"pointer", "opacity":"0.5"}),
                html.Span(" \u25b6", style={"opacity":"0.5"}),
            ], id="hm-swipe-indicator", className="text-center py-1",
               style={"display":"none", "fontSize":"0.85rem", "color":"#6c757d",
                       "userSelect":"none"}),
            # Swipe container
            html.Div([
                html.Div([
                    dcc.Loading(
                        dcc.Graph(id="heatmap-graph", style=_STYLE_GRAPH_H,
                                  config={"scrollZoom":False,
                                          "displayModeBar":"hover",
                                          "toImageButtonOptions":{"format":"png","scale":2,
                                                                   "filename":"btc_heatmap"}}),
                        type="default", color=_BTC_ORANGE,
                    ),
                ], className="hm-swipe-panel"),
                html.Div([
                    dcc.Loading(
                        dcc.Graph(id="hm-mc-graph", style=_STYLE_GRAPH_H,
                                  config={"scrollZoom":False,
                                          "displayModeBar":"hover",
                                          "toImageButtonOptions":{"format":"png","scale":2,
                                                                   "filename":"btc_mc_heatmap"}}),
                        type="default", color=_BTC_ORANGE,
                    ),
                    # MC chart overlay (gray mask when MC not rendered)
                    html.Div(id="hm-mc-overlay", style=_STYLE_HIDDEN,
                             className="mc-chart-overlay"),
                    html.Img(id="hm-mc-badge",
                             src="/assets/quantoshi_favicon.png",
                             className="mc-premium-badge",
                             style=_STYLE_HIDDEN),
                ], className="hm-swipe-panel mc-premium-chart", id="hm-mc-panel",
                   style=_STYLE_HIDDEN),
            ], className="hm-swipe-container", id="hm-swipe-wrap"),
            html.Div(id="hm-swipe-scroll-dummy", style=_STYLE_HIDDEN),
            _export_row("heatmap"),
        ], width=9),
    ], className="g-0")
