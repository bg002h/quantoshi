"""Leverage Calculator tab — control panel + output widgets.

Design spec: docs/superpowers/specs/2026-04-18-leverage-calculator-design.md
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

import _app_ctx
from colors import UI_FONT_MD
from tab_defaults import leverage_defaults


# Curated flagship model list for the "Reversion target model" dropdown.
_LEV_FLAGSHIP_MODELS = ["bub", "pl", "qr", "lppl", "hybppl", "eppl", "pca", "grdy"]


def _model_options():
    """Build dropdown options from PRICE_MODELS, respecting flagship ordering."""
    opts = []
    for key in _LEV_FLAGSHIP_MODELS:
        m = _app_ctx.PRICE_MODELS.get(key)
        if m is not None:
            opts.append({"label": m.name, "value": key})
    if "ef" in _app_ctx.PRICE_MODELS:
        opts.append({"label": _app_ctx.PRICE_MODELS["ef"].name, "value": "ef"})
    return opts


# Floor quantile pill bar — hardcoded at import (parallel to routing.py::_HM_PILL_IDS).
_LEV_FLOOR_QS = [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.50,
                 0.80, 0.85, 0.90, 0.95, 0.99]
_LEV_PILL_IDS = [f"lev-pill-q{int(q*1000):03d}" for q in _LEV_FLOOR_QS]


def _floor_pill_bar():
    """Render 6 pill buttons for floor quantile selection."""
    labels = ["Q0.1%", "Q1%", "Q5%", "Q10%", "Q15%", "Q20%", "Q50%",
              "Q80%", "Q85%", "Q90%", "Q95%", "Q99%"]
    default_q = 0.01
    return html.Div([
        dbc.Button(
            lbl, id=pid, size="sm", className="me-1",
            outline=(q != default_q), color="primary",
            n_clicks=0,
        )
        for pid, lbl, q in zip(_LEV_PILL_IDS, labels, _LEV_FLOOR_QS)
    ], className="d-flex flex-wrap")


def _leverage_tab() -> html.Div:
    """Build the Leverage Calculator tab content."""
    d = leverage_defaults()
    return html.Div([
        # Scenario context row
        dbc.Row([
            dbc.Col([
                html.Label("Current date", style={"fontSize": UI_FONT_MD}),
                dcc.DatePickerSingle(id="lev-date", date=d["lev_date"]),
            ], md=4, xs=12, className="mb-2"),
            dbc.Col([
                html.Label("Current price ($)", style={"fontSize": UI_FONT_MD}),
                dbc.Input(id="lev-price", type="number", min=1, step=0.01,
                          value=d["lev_price"]),
            ], md=4, xs=12, className="mb-2"),
        ], className="mb-3"),

        # Reversion target row
        dbc.Row([
            dbc.Col([
                html.Label("Reversion target model", style={"fontSize": UI_FONT_MD}),
                dcc.Dropdown(id="lev-model", options=_model_options(),
                             value=d["lev_model"], clearable=False),
            ], md=6, xs=12, className="mb-2"),
            dbc.Col([
                html.Label("Floor quantile", style={"fontSize": UI_FONT_MD}),
                _floor_pill_bar(),
                dcc.Store(id="lev-floor-q-store", data=d["lev_floor_q"]),
            ], md=6, xs=12, className="mb-2"),
        ], className="mb-3"),

        # Rate environment row
        dbc.Row([
            dbc.Col([
                html.Label("Borrow rate r_b (0–50 % / yr)",
                           style={"fontSize": UI_FONT_MD}),
                dbc.Input(id="lev-rb", type="number", min=0, max=50, step=0.001,
                          value=d["lev_rb"]),
            ], md=6, xs=12, className="mb-2"),
            dbc.Col([
                html.Label("Opportunity cost r_l (0–50 % / yr)",
                           style={"fontSize": UI_FONT_MD}),
                dbc.Input(id="lev-rl", type="number", min=0, max=50, step=0.001,
                          value=d["lev_rl"]),
            ], md=6, xs=12, className="mb-2"),
        ], className="mb-3"),

        # Your-scenario row
        dbc.Row([
            dbc.Col([
                html.Label("Horizon H (years)", style={"fontSize": UI_FONT_MD}),
                dcc.Slider(id="lev-horizon", min=0.25, max=20, step=0.25,
                           value=d["lev_horizon"],
                           marks={i: str(i) for i in range(0, 21, 2)},
                           tooltip={"always_visible": True, "placement": "top"}),
            ], md=6, xs=12, className="mb-2"),
            dbc.Col([
                html.Label("Target CAGR (%)", style={"fontSize": UI_FONT_MD}),
                dcc.Slider(id="lev-cagr", min=0, max=50, step=0.1,
                           value=d["lev_cagr"],
                           marks={i: f"{i}%" for i in range(0, 51, 10)},
                           tooltip={"always_visible": True, "placement": "top"}),
            ], md=6, xs=12, className="mb-2"),
        ], className="mb-3"),

        # Display options
        dbc.Row([
            dbc.Col([
                dcc.Checklist(
                    id="lev-toggles",
                    options=[{"label": " Show legend", "value": "legend"}],
                    value=list(d["lev_toggles"]),
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    style={"fontSize": UI_FONT_MD},
                ),
            ], xs=12, className="mb-2"),
        ], className="mb-2"),

        # Outputs
        html.Div(id="lev-readout", className="mb-3"),
        dcc.Graph(id="lev-graph", style={"height": "55vh"}),
        html.Div(id="lev-table-wrap",
                 style={"overflowX": "auto", "marginTop": "12px"},
                 children=html.Div(id="lev-table")),
    ], id="leverage-tab-content", className="p-3")
