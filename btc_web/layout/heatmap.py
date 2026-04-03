"""Tab 2 — CAGR Heatmap layout."""

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from layout.common import (_tab_hints, _section_card, _lbl, _row, _export_row,
                            _STYLE_HIDDEN, _STYLE_HINT, _STYLE_GRAPH_H,
                            _STYLE_COLOR_H, _BTC_ORANGE,
                            _CB_MARGIN, _Q_HINT_BASE,
                            _q_options)
from layout.mc_controls import _mc_controls
from tab_defaults import HEATMAP


def _heatmap_controls():
    yr_now = pd.Timestamp.today().year
    return html.Div([
        _tab_hints("heatmap"),
        # ── Projection Quantiles ──────────────────────────────────────
        _section_card("Projection Quantiles",
            html.Small(_Q_HINT_BASE,
                style=_STYLE_HINT),
            html.Small("Select quantiles for CAGR projection columns.",
                style=_STYLE_HINT),
            dcc.Checklist(id="hm-exit-qs", options=_q_options(),
                          value=_app_ctx._DEF_QS, className="q-panel-grid",
                          inputStyle=_CB_MARGIN),
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
            dbc.Input(id="hm-stack", type="number", value=HEATMAP["stack"],
                      min=0, step=0.001, size="sm", debounce=True),
            dcc.Checklist(id="hm-use-lots",
                          options=[{"label": " Use Stack Tracker lots", "value": "yes"}],
                          value=[], inputStyle=_CB_MARGIN),
        ),
        _mc_controls("hm", show_amount=True, show_inflation=True,
                     show_stack=True, show_mc_entry_q=True, default_entry_q=10,
                     shared_controls={"stack"}),
        # ── Chart ───────────────────────────────────────────────────────
        _section_card("Chart Settings",
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
                           value=HEATMAP["color_mode"], labelStyle={"display":"block"},
                           inputStyle=_CB_MARGIN),
            _lbl("Break 1 (CAGR %, integer)"),
            dbc.Input(id="hm-b1", type="number", value=HEATMAP["b1"],
                      step=1, size="sm"),
            _lbl("Break 2 (CAGR %, integer)"),
            dbc.Input(id="hm-b2", type="number", value=HEATMAP["b2"],
                      step=1, size="sm"),
            _lbl("Color palette"),
            dcc.Dropdown(id="hm-palette",
                options=[
                    {"label": "Finance (red \u2192 green)",    "value": "finance"},
                    {"label": "Forge (dark \u2192 gold)",      "value": "forge"},
                    {"label": "Thermal (blue \u2192 red)",     "value": "thermal"},
                    {"label": "Bitcoin (dark \u2192 orange)",  "value": "bitcoin"},
                    {"label": "Ocean (navy \u2192 cyan)",      "value": "ocean"},
                    {"label": "Monochrome (gray)",        "value": "mono"},
                    {"label": "Custom",                   "value": "custom"},
                ],
                value="mono", clearable=False),
            _row(
                html.Div([_lbl("Lo"), dbc.Input(id="hm-c-lo", type="color",
                           value="#1a1a1a", style=_STYLE_COLOR_H)]),
                html.Div([_lbl("Mid1"), dbc.Input(id="hm-c-mid1", type="color",
                           value="#555555", style=_STYLE_COLOR_H)]),
                html.Div([_lbl("Mid2"), dbc.Input(id="hm-c-mid2", type="color",
                           value="#999999", style=_STYLE_COLOR_H)]),
                html.Div([_lbl("Hi"), dbc.Input(id="hm-c-hi", type="color",
                           value="#e0e0e0", style=_STYLE_COLOR_H)]),
            ),
            _lbl("Gradient steps"),
            dbc.Input(id="hm-grad", type="number", value=HEATMAP["n_disc"],
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
                value=HEATMAP["vfmt"], clearable=False),
            _lbl("Cell font size"),
            dbc.Input(id="hm-cell-fs", type="number", value=HEATMAP["cell_font_size"],
                      min=5, max=20, step=1, size="sm"),
            dcc.Checklist(id="hm-toggles",
                          options=[{"label":" Show colorbar","value":"colorbar"},
                                   {"label":" Enable chart zoom","value":"chart_zoom"}],
                          value=["colorbar"], labelStyle={"display":"block"},
                          inputStyle=_CB_MARGIN),
        ),
        # Hidden placeholder — hm-model-show is referenced by callbacks/snapshot
        # but no longer user-visible (pill bar replaces it on tab 2)
        dcc.Checklist(id="hm-model-show", value=["qr"],
                      style=_STYLE_HIDDEN),
    ])


def _hm_pill_bar():
    """Build model-selector pill bar from registered PRICE_MODELS + optional MC."""
    _pills = []
    for key, mdl in _app_ctx.PRICE_MODELS.items():
        if key == "bub":
            continue
        _pills.append({"label": mdl.name, "value": key})

    buttons = [
        dbc.Button("Bubble Model", id="hm-pill-bub", color="primary",
                   size="sm", className="me-1"),
    ] + [
        dbc.Button(p["label"], id=f"hm-pill-{p['value']}", outline=True,
                   color="primary", size="sm", className="me-1")
        for p in _pills
    ]
    if _app_ctx._HAS_MARKOV:
        buttons.append(
            dbc.Button("Monte Carlo", id="hm-pill-mc", outline=True,
                       color="warning", size="sm"),
        )

    return html.Div([
        dbc.ButtonGroup(buttons, size="sm"),
    ], className="mb-2 text-center")


def _heatmap_tab():
    return dbc.Row([
        dbc.Col([
            _heatmap_controls(),
        ], width=3, className="controls-col overflow-auto",
                style={"maxHeight":"85vh"}),
        dbc.Col([
            # Model selector pills
            _hm_pill_bar(),
            # Active model store
            dcc.Store(id="hm-active-model", storage_type="memory", data="bub"),
            # Swipe indicator (kept as hidden placeholder for existing callbacks)
            html.Div(id="hm-swipe-indicator", style=_STYLE_HIDDEN),
            # Wrap chart + overlay in position:relative so mc-chart-overlay
            # is scoped to the chart area (not the full page)
            html.Div(id="hm-chart-wrap", style={"position": "relative"}, children=[
                # Single chart
                dcc.Loading(
                    dcc.Graph(id="heatmap-graph", style=_STYLE_GRAPH_H,
                              config={"scrollZoom":False,
                                      "displayModeBar":"hover",
                                      "toImageButtonOptions":{"format":"png","scale":2,
                                                               "filename":"btc_heatmap"}}),
                    type="default", color=_BTC_ORANGE,
                ),
                # MC status / overlay elements (kept for MC callbacks)
                html.Div(id="hm-mc-panel", style=_STYLE_HIDDEN,
                         className="mc-premium-chart"),
                html.Div(id="hm-mc-overlay", style=_STYLE_HIDDEN,
                         className="mc-chart-overlay"),
                html.Img(id="hm-mc-badge",
                         src="/assets/quantoshi_favicon.png",
                         className="mc-premium-badge",
                         style=_STYLE_HIDDEN),
            ]),
            html.Div(id="hm-swipe-scroll-dummy", style=_STYLE_HIDDEN),
            _export_row("heatmap"),
        ], width=9),
    ], className="g-0")
