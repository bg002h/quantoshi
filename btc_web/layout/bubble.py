"""Tab 1 — Bubble + QR Overlay layout."""

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from tab_defaults import BUBBLE
from colors import (
    USER_MODEL_TRACE, FALLBACK_MODEL_GRAY,
    DIM_TEXT, BOOTSTRAP_LIGHT_BG,
    _hex_alpha, FONT_MONO,
    UI_FONT_XS, UI_FONT_SM, UI_FONT_MD, UI_FONT_BASE,
    UM_INPUT_BG_ALPHA, UM_INPUT_BORDER_ALPHA,
)
from layout.common import (_tab_hints, _section_card, _row, _lbl,
                            _STYLE_HIDDEN, _STYLE_HINT, _q_panel, _q_panel_with_mode,
                            _q_options, _legend_pos_dropdown,
                            _chart_tab_layout, _CB_MARGIN, _palette_selector,
                            _plot_appearance_controls, _use_lots_checklist)
from layout.display_models import display_models_panel, sigma_mode_section
from layout.custom_time import custom_time_panel


def _bubble_controls():
    yr_now = pd.Timestamp.today().year
    return html.Div([
        _tab_hints("bubble"),
        _section_card("Axes & Range",
            html.Div(id="bub-scale-controls", children=[
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
            ]),
            _lbl("X range (year)"),
            dcc.RangeSlider(id="bub-xrange", min=2010, max=2080,
                            value=[2010, 2033], step=1,
                            marks={y: f"'{y % 100:02d}" for y in range(2010, 2081, 10)},
                            tooltip={"always_visible":False},
                            updatemode="mouseup"),
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
                                value=[-1.5, 6.05], step=0.5,
                                marks={-2:"1\u00a2", 0:"$1", 2:"$100",
                                        4:"$10K", 6:"$1M", 9:"$1B"},
                                tooltip={"always_visible":False},
                                updatemode="mouseup"),
            ]),
        ),
        _section_card("Display",
            dcc.Checklist(id="bub-toggles",
                          options=[{"label":" Show price data","value":"show_data"},
                                   {"label":" Shade bands","value":"shade"},
                                   {"label":" Show OLS","value":"show_ols"},
                                   {"label":" Unfairly Cheap Line","value":"show_ucl"},
                                   {"label":" Show today","value":"show_today"},
                                   {"label":" Show legend","value":"show_legend"},
                                   {"label":html.Span(" Minor grid",className="minor-grid-opt"),"value":"minor_grid"},
                                   {"label":" Enable chart zoom","value":"chart_zoom"}],
                          value=["shade","show_data","show_today"],
                          labelStyle={"display":"block"},
                          inputStyle=_CB_MARGIN),
        ),
        display_models_panel("bub", include_bm_master=True,
                              default_value=["bub"],
                              legend_pos_default=BUBBLE["legend_pos"]),
        _q_panel_with_mode("bub-qs", [0.5],
                           hint=f"If none selected, Q50% is shown at "
                                f"{int(_app_ctx.FALLBACK_Q50_OPACITY * 100)}% opacity."),
        sigma_mode_section(),
        custom_time_panel(),
        # Hidden placeholders — bub-bubble-panel and bub-n-future-wrap are
        # referenced as style outputs by the bub-view-mode toggle callbacks.
        # Real Bubble Model controls now live in the global bm-config-modal.
        html.Div(id="bub-bubble-panel", style=_STYLE_HIDDEN),
        html.Div(id="bub-n-future-wrap", style=_STYLE_HIDDEN),
        _section_card("Model Component Decomposition",
            _lbl("Model"),
            dcc.Dropdown(
                id="bub-decomp-model",
                options=[{"label": "(none)", "value": ""}] +
                        [{"label": label, "value": key}
                         for key, label in _app_ctx.DECOMP_FAMILIES.items()],
                value="", clearable=False,
            ),
            dcc.Checklist(
                id="bub-decomp-show-formulas",
                options=[
                    {"label": " Show full model formula", "value": "full"},
                    {"label": " Show selected formula", "value": "selected"},
                ],
                value=[], inputStyle=_CB_MARGIN,
                labelStyle={"display": "block", "fontSize": UI_FONT_SM, "color": DIM_TEXT},
                style={"marginTop": "4px", "marginBottom": "4px"},
            ),
            html.Div(id="bub-decomp-formula",
                     style={"fontSize": UI_FONT_MD, "marginTop": "6px",
                            "marginBottom": "6px", "overflowX": "auto"}),
            html.Div(id="bub-decomp-active-formula",
                     style={"fontSize": UI_FONT_SM, "marginTop": "4px",
                            "marginBottom": "6px", "padding": "6px",
                            "background": BOOTSTRAP_LIGHT_BG, "borderRadius": "4px",
                            "wordBreak": "break-word", "fontFamily": FONT_MONO}),
            html.Div(id="bub-decomp-body", style=_STYLE_HIDDEN, children=[
                dcc.Checklist(
                    id="bub-decomp-components",
                    options=[], value=[],
                    labelStyle={"display": "block", "fontSize": UI_FONT_MD},
                    inputStyle=_CB_MARGIN,
                ),
                html.Small(
                    "Each checkbox toggles a term on/off. "
                    "log\u2081\u2080(price) = sum of checked terms. "
                    "All checked = full model.",
                    style={"color": FALLBACK_MODEL_GRAY, "fontSize": UI_FONT_SM,
                            "display": "block", "marginTop": "4px"},
                ),
                # Hidden placeholder — preserves bub-decomp-mode snapshot slot.
                dcc.RadioItems(id="bub-decomp-mode", value="individual",
                                style=_STYLE_HIDDEN),
            ]),
            html.Div(id="bub-decomp-warning", children=[]),
            no_hover=True,
        ),
        _section_card(
            html.Span(id="bub-scanner-header",
                      children="Model Scanner \u00b7 Constant \u03c3"),
            _row(
                html.Div([
                    _lbl("Price ($)"),
                    dbc.Input(id="scan-price", type="number",
                              placeholder="live", size="sm", debounce=True),
                    html.Small("\u20bf live", id="scan-price-hint",
                               className="text-muted", style={"fontSize": UI_FONT_XS}),
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
        _section_card("Plot Appearance",
            _row(
                html.Div([_lbl("Pt size (1\u201320)"),
                          dbc.Input(id="bub-ptsize", type="number",
                                    value=BUBBLE["pt_size"], min=1, max=20, size="sm")]),
                html.Div([_lbl("Pt alpha (0.1\u20131)"),
                          dbc.Input(id="bub-ptalpha", type="number",
                                    value=BUBBLE["pt_alpha"], min=0.1, max=1.0, step=0.05, size="sm")]),
            ),
            *_plot_appearance_controls("bub"),
        ),
        _section_card("Stack (BTC)",
            dbc.InputGroup([
                dbc.Input(id="bub-stack", type="number", value=BUBBLE["stack"],
                          min=0, step=0.001, size="sm", debounce=True),
                dbc.InputGroupText(dcc.Checklist(
                    id="bub-show-stack",
                    options=[{"label":" Show","value":"yes"}],
                    value=[], inputStyle=_CB_MARGIN)),
            ], size="sm"),
            _use_lots_checklist("bub"),
        ),
        html.Div(id="um-panel-wrap", style=_STYLE_HIDDEN, children=[
            _section_card("User Model (U\u2081)",
                html.Small("Click a data point on the chart, then tap P1 or P2. "
                           "Model auto-draws when both are set.",
                           style={"color":FALLBACK_MODEL_GRAY, "fontSize": UI_FONT_MD}),
                html.Div([
                    html.Label("Point 1", style={"fontWeight":"600", "fontSize": UI_FONT_BASE, "marginTop":"6px"}),
                    html.Div([
                        _lbl("Year"), html.Span(id="um-p1-year-display",
                            style={"display":"inline-block", "width":"80px", "fontSize": UI_FONT_BASE,
                                   "fontWeight":"600", "color":USER_MODEL_TRACE,
                                   "backgroundColor":_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BG_ALPHA),
                                   "border":f"1px solid {_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BORDER_ALPHA)}",
                                   "borderRadius":"4px", "padding":"2px 6px",
                                   "minHeight":"24px", "verticalAlign":"middle"},
                            children="\u2014"),
                        html.Span(" ", style={"width":"6px", "display":"inline-block"}),
                        _lbl("$"), html.Span(id="um-p1-price-display",
                            style={"display":"inline-block", "width":"100px", "fontSize": UI_FONT_BASE,
                                   "fontWeight":"600", "color":USER_MODEL_TRACE,
                                   "backgroundColor":_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BG_ALPHA),
                                   "border":f"1px solid {_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BORDER_ALPHA)}",
                                   "borderRadius":"4px", "padding":"2px 6px",
                                   "minHeight":"24px", "verticalAlign":"middle"},
                            children="\u2014"),
                    ], style={"display":"flex", "alignItems":"center", "gap":"4px", "flexWrap":"wrap"}),
                    html.Label("Point 2", style={"fontWeight":"600", "fontSize": UI_FONT_BASE, "marginTop":"6px"}),
                    html.Div([
                        _lbl("Year"), html.Span(id="um-p2-year-display",
                            style={"display":"inline-block", "width":"80px", "fontSize": UI_FONT_BASE,
                                   "fontWeight":"600", "color":USER_MODEL_TRACE,
                                   "backgroundColor":_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BG_ALPHA),
                                   "border":f"1px solid {_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BORDER_ALPHA)}",
                                   "borderRadius":"4px", "padding":"2px 6px",
                                   "minHeight":"24px", "verticalAlign":"middle"},
                            children="\u2014"),
                        html.Span(" ", style={"width":"6px", "display":"inline-block"}),
                        _lbl("$"), html.Span(id="um-p2-price-display",
                            style={"display":"inline-block", "width":"100px", "fontSize": UI_FONT_BASE,
                                   "fontWeight":"600", "color":USER_MODEL_TRACE,
                                   "backgroundColor":_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BG_ALPHA),
                                   "border":f"1px solid {_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BORDER_ALPHA)}",
                                   "borderRadius":"4px", "padding":"2px 6px",
                                   "minHeight":"24px", "verticalAlign":"middle"},
                            children="\u2014"),
                    ], style={"display":"flex", "alignItems":"center", "gap":"4px", "flexWrap":"wrap"}),
                ]),
                dcc.Store(id="um-p1-year", data=None),
                dcc.Store(id="um-p1-price", data=None),
                dcc.Store(id="um-p2-year", data=None),
                dcc.Store(id="um-p2-price", data=None),
                # Draw button hidden (auto-draws on P2 set) but kept for callback compatibility
                html.Div(id="um-draw-btn", style=_STYLE_HIDDEN),
                dbc.Button("\u2715 Clear", id="um-delete-btn", color="secondary", size="sm",
                           outline=True, style={"marginTop":"8px"}),
            ),
        ]),
        _palette_selector("bub"),
    ])


def _bubble_tab():
    from layout.common import _chart_tab_layout_with_fab
    return _chart_tab_layout_with_fab(_bubble_controls, "bubble-graph", "btc_bubble")
