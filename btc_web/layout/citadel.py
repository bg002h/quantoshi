"""Citadel Planner (Tab 9) — layout with tabbed sub-panels."""

import pandas as pd
from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from layout.common import (
    _section_card, _lbl, _ctrl_card, _q_options,
    _chart_toggles, _btc_usd_dropdown, _legend_pos_dropdown,
    _STYLE_HIDDEN, _STYLE_HINT, _export_row, _chart_tab_layout,
)
from layout.mc_controls import _mc_controls


def _assets_panel():
    """Sub-tab 1: BTC stack, cash, reserves, investments."""
    return html.Div([
        _section_card("Bitcoin Stack",
            _lbl("Starting BTC"),
            dbc.Input(id="cp-stack", type="number", value=1.0, min=0, step=0.01),
            dcc.Checklist(id="cp-use-lots", options=[{"label": " Use Stack Tracker lots", "value": "yes"}],
                          value=[], inputStyle={"marginRight": "5px"}),
        ),
        _section_card("Cash Account",
            _lbl("Initial balance ($)"),
            dbc.Input(id="cp-cash-init", type="number", value=50000, min=0, step=1),
            _lbl("Interest rate (% / yr)"),
            dbc.Input(id="cp-cash-rate", type="number", value=4.0, min=0, step=0.1),
        ),
        _section_card("Reserve Fund — US Treasuries",
            html.Small("3 maturity bins: Short (T-Bills ≤1yr), Medium (T-Notes 2-10yr), Long (T-Bonds 10-30yr)",
                       style=_STYLE_HINT),
            # Header row
            dbc.Row([
                dbc.Col(html.Small("Bin", className="fw-bold"), width=3),
                dbc.Col(html.Small("Initial ($)", className="fw-bold"), width=3),
                dbc.Col(html.Small("Return (%)", className="fw-bold"), width=3),
                dbc.Col(html.Small("Vol (%)", className="fw-bold"), width=3),
            ], className="g-1 mb-1"),
            # Short
            dbc.Row([
                dbc.Col(html.Small("Short"), width=3),
                dbc.Col(dbc.Input(id="cp-res-short-init", type="number", value=50000, min=0, step=1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-res-short-rate", type="number", value=5.0, min=0, step=0.1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-res-short-vol", type="number", value=2.0, min=0, step=0.1, size="sm"), width=3),
            ], className="g-1 mb-1"),
            # Medium
            dbc.Row([
                dbc.Col(html.Small("Medium"), width=3),
                dbc.Col(dbc.Input(id="cp-res-med-init", type="number", value=100000, min=0, step=1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-res-med-rate", type="number", value=4.5, min=0, step=0.1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-res-med-vol", type="number", value=8.0, min=0, step=0.1, size="sm"), width=3),
            ], className="g-1 mb-1"),
            # Long
            dbc.Row([
                dbc.Col(html.Small("Long"), width=3),
                dbc.Col(dbc.Input(id="cp-res-long-init", type="number", value=50000, min=0, step=1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-res-long-rate", type="number", value=4.0, min=0, step=0.1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-res-long-vol", type="number", value=15.0, min=0, step=0.1, size="sm"), width=3),
            ], className="g-1"),
        ),
        _section_card("Investment Account",
            html.Small("2 bins: Equities (stocks) and Bonds", style=_STYLE_HINT),
            dbc.Row([
                dbc.Col(html.Small("Bin", className="fw-bold"), width=3),
                dbc.Col(html.Small("Initial ($)", className="fw-bold"), width=3),
                dbc.Col(html.Small("Return (%)", className="fw-bold"), width=3),
                dbc.Col(html.Small("Vol (%)", className="fw-bold"), width=3),
            ], className="g-1 mb-1"),
            dbc.Row([
                dbc.Col(html.Small("Equities"), width=3),
                dbc.Col(dbc.Input(id="cp-inv-eq-init", type="number", value=200000, min=0, step=1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-inv-eq-rate", type="number", value=10.0, min=0, step=0.1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-inv-eq-vol", type="number", value=16.0, min=0, step=0.1, size="sm"), width=3),
            ], className="g-1 mb-1"),
            dbc.Row([
                dbc.Col(html.Small("Bonds"), width=3),
                dbc.Col(dbc.Input(id="cp-inv-bd-init", type="number", value=100000, min=0, step=1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-inv-bd-rate", type="number", value=5.0, min=0, step=0.1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-inv-bd-vol", type="number", value=7.0, min=0, step=0.1, size="sm"), width=3),
            ], className="g-1"),
        ),
    ])


def _spending_panel():
    """Sub-tab 2: Monthly spending, inflation, growth."""
    return html.Div([
        _section_card("Monthly Spending",
            _lbl("Spending amount ($ / month)"),
            dbc.Input(id="cp-spend", type="number", value=5000, min=0, step=1),
            _lbl("Inflation rate (% / yr)"),
            dbc.Input(id="cp-infl", type="number", value=4.0, min=0, max=100, step=0.5),
            _lbl("Spending growth above inflation (% / yr)"),
            dbc.Input(id="cp-spend-growth", type="number", value=0.0, min=0, max=100, step=0.5),
        ),
    ])


def _trigger_section(prefix, label, default_thresh, default_mode, default_rate, default_dur, split_label):
    """Build a rebalancing trigger section (high-Q or low-Q)."""
    return _section_card(label,
        _lbl("Threshold (quantile %)"),
        dbc.Input(id=f"cp-{prefix}-thresh", type="number", value=default_thresh,
                  min=0, max=100, step=1),
        _lbl("Action mode"),
        dcc.RadioItems(id=f"cp-{prefix}-mode",
            options=[{"label": " Gradual", "value": "gradual"},
                     {"label": " Lump", "value": "lump"}],
            value=default_mode, inline=True,
            inputStyle={"marginRight": "4px"}, labelStyle={"marginRight": "12px"}),
        _lbl("Rate (% per action)"),
        dbc.Input(id=f"cp-{prefix}-rate", type="number", value=default_rate,
                  min=0.1, max=100, step=0.1),
        _lbl("Duration (periods) — gradual only"),
        dbc.Input(id=f"cp-{prefix}-dur", type="number", value=default_dur,
                  min=1, max=120, step=1),
        _lbl(split_label),
        html.Small("Cash / Short / Med / Long / Equities / Bonds — must sum to 100%",
                   style=_STYLE_HINT),
        dbc.Row([
            dbc.Col([html.Small("Cash"), dbc.Input(id=f"cp-{prefix}-split-cash", type="number", value=20, min=0, max=100, step=1, size="sm")], width=2),
            dbc.Col([html.Small("Short"), dbc.Input(id=f"cp-{prefix}-split-rs", type="number", value=20, min=0, max=100, step=1, size="sm")], width=2),
            dbc.Col([html.Small("Med"), dbc.Input(id=f"cp-{prefix}-split-rm", type="number", value=20, min=0, max=100, step=1, size="sm")], width=2),
            dbc.Col([html.Small("Long"), dbc.Input(id=f"cp-{prefix}-split-rl", type="number", value=10, min=0, max=100, step=1, size="sm")], width=2),
            dbc.Col([html.Small("Eq"), dbc.Input(id=f"cp-{prefix}-split-eq", type="number", value=20, min=0, max=100, step=1, size="sm")], width=2),
            dbc.Col([html.Small("Bd"), dbc.Input(id=f"cp-{prefix}-split-bd", type="number", value=10, min=0, max=100, step=1, size="sm")], width=2),
        ], className="g-1"),
    )


def _rules_panel():
    """Sub-tab 3: Rebalancing triggers, floor rules, Saylor Fortifier."""
    return html.Div([
        _trigger_section("high-q", "High-Quantile Trigger — Take Profits",
                        95, "gradual", 2.0, 6, "Proceeds distribution (%)"),
        _trigger_section("low-q", "Low-Quantile Trigger — Accumulate BTC",
                        5, "lump", 10.0, 1, "Source allocation (%)"),
        _section_card("Global Lump Cooldown",
            _lbl("Minimum periods between lump actions"),
            dbc.Input(id="cp-lump-cooldown", type="number", value=12, min=1, step=1),
        ),
        _section_card("Account Floor Rules",
            html.Small("Minimum balances maintained each period", style=_STYLE_HINT),
            _lbl("Cash floor ($)"),
            dbc.Input(id="cp-cash-floor", type="number", value=50000, min=0, step=1),
            _lbl("Reserve Short floor ($)"),
            dbc.Input(id="cp-res-short-floor", type="number", value=0, min=0, step=1),
            _lbl("Reserve Medium floor ($)"),
            dbc.Input(id="cp-res-med-floor", type="number", value=0, min=0, step=1),
            _lbl("Reserve Long floor ($)"),
            dbc.Input(id="cp-res-long-floor", type="number", value=0, min=0, step=1),
            _lbl("Cash floor annual increase (%)"),
            dbc.Input(id="cp-cash-floor-growth", type="number", value=0, min=0, max=50, step=0.5),
            _lbl("Reserve floor annual increase (%)"),
            dbc.Input(id="cp-res-floor-growth", type="number", value=0, min=0, max=50, step=0.5),
        ),
        # Saylor Citadel Fortifier
        dcc.Checklist(id="cp-scf-enable",
            options=[{"label": " Enable Saylor Citadel Fortifier", "value": "yes"}],
            value=[], inputStyle={"marginRight": "5px"},
            style={"marginBottom": "8px"}),
        html.Div(id="cp-scf-body", style=_STYLE_HIDDEN, children=[
            _section_card("Saylor Citadel Fortifier",
                _lbl("Loan amount ($)"),
                dbc.Input(id="cp-scf-amount", type="number", value=100000, min=0, step=1),
                _lbl("Loan type"),
                dcc.RadioItems(id="cp-scf-type",
                    options=[{"label": " Term", "value": "term"},
                             {"label": " Perpetual", "value": "perpetual"}],
                    value="term", inline=True,
                    inputStyle={"marginRight": "4px"}, labelStyle={"marginRight": "12px"}),
                _lbl("Interest rate (% / yr)"),
                dbc.Input(id="cp-scf-rate", type="number", value=8.0, min=0, step=0.1),
                _lbl("Term (months) — term loan only"),
                dbc.Input(id="cp-scf-term", type="number", value=60, min=1, step=1),
                _lbl("Repayment trigger (N x rate) — perpetual only"),
                dbc.Input(id="cp-scf-trigger", type="number", value=1.0, min=0.1, step=0.1),
            ),
        ]),
    ])


def _dd_section(title, *children):
    """Section wrapper for controls with dropdowns — avoids dbc.Card which
    clips dropdown menus on iOS due to transform creating a stacking context."""
    return html.Div([
        html.Div(title, style={"fontWeight": "bold", "fontSize": "12px",
                                "color": "#555", "marginBottom": "4px",
                                "textTransform": "uppercase", "letterSpacing": "0.03em"}),
        *children,
    ], style={"marginBottom": "12px", "padding": "8px",
              "background": "#f8f9fa", "borderRadius": "8px",
              "border": "1px solid #dee2e6"})


def _sim_panel():
    """Sub-tab 4: Simulation settings, quantiles, MC, chart toggles."""
    return html.Div([
        _dd_section("Simulation",
            _lbl("Year range"),
            dcc.RangeSlider(id="cp-yr-range", min=2025, max=2080,
                value=[2031, 2075], step=1,
                marks={y: str(y) for y in range(2025, 2081, 5)}),
            _lbl("Frequency"),
            dcc.Dropdown(id="cp-freq",
                options=[{"label": "Monthly", "value": "Monthly"},
                         {"label": "Quarterly", "value": "Quarterly"},
                         {"label": "Annually", "value": "Annually"}],
                value="Monthly", clearable=False),
            _lbl("Price model"),
            dcc.Dropdown(id="cp-model-src",
                options=[{"label": "Bubble Model", "value": "bub"},
                         {"label": "Power Law", "value": "pl"},
                         {"label": "S2F", "value": "s2f"}],
                value="bub", clearable=False),
        ),
        _dd_section("Dollar Asset Returns",
            html.Small("Lognormal uses your input rates/volatility. "
                       "Markov uses historical regime transitions (S&P 500, bonds, treasuries).",
                       style=_STYLE_HINT),
            dcc.Dropdown(id="cp-asset-model",
                options=[{"label": "Lognormal (user rates)", "value": "lognormal"},
                         {"label": "Markov (historical regimes)", "value": "markov"}],
                value="lognormal", clearable=False),
        ),
        _dd_section("BTC Price Scenario",
            html.Small("Select one quantile for the deterministic BTC price path. "
                       "Lower = more pessimistic, higher = more optimistic.",
                       style=_STYLE_HINT),
            dcc.Dropdown(id="cp-qs",
                options=[{"label": f"Q{q*100:g}%", "value": q}
                         for q in [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25,
                                   0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80,
                                   0.85, 0.90, 0.95, 0.99, 0.999]],
                value=0.25, clearable=False),
        ),
        _dd_section("Display",
            _lbl("Display mode"),
            dcc.Dropdown(id="cp-disp",
                options=[{"label": "USD (total portfolio)", "value": "usd_total"},
                         {"label": "USD (per asset class)", "value": "usd_per_asset"},
                         {"label": "BTC Holdings", "value": "btc"}],
                value="usd_per_asset", clearable=False),
        ),
        _mc_controls("cp", amount_label="Spending per period ($)",
                     amount_default=5000, show_inflation=True, show_stack=True,
                     default_entry_q=10,
                     shared_controls={"amount", "infl", "freq", "stack"}),
        _section_card("Chart Settings",
            _chart_toggles("cp", ["annotate", "log_y", "show_legend", "minor_grid"]),
            *_legend_pos_dropdown("cp", "bottom-right"),
        ),
    ])


def _citadel_controls():
    """All controls in inner tabbed sub-panels."""
    return html.Div([
        # Run button + instructions
        dbc.Button("\u25B6  Run Simulation", id="cp-run-btn",
                   color="warning", className="w-100 mb-2 fw-bold",
                   style={"fontSize": "14px", "letterSpacing": "0.03em"}),
        html.Small("Configure settings below, then click Run Simulation. "
                   "1 deterministic sim is free. MC (multiple sims) requires payment.",
                   style={"color": "#888", "display": "block", "marginBottom": "8px",
                          "fontSize": "11px"}),
        dbc.Tabs([
            dbc.Tab(_assets_panel(), label="Assets", tab_id="cp-assets"),
            dbc.Tab(_spending_panel(), label="Spending", tab_id="cp-spending"),
            dbc.Tab(_rules_panel(), label="Rules", tab_id="cp-rules"),
            dbc.Tab(_sim_panel(), label="Simulation", tab_id="cp-sim"),
        ], id="cp-inner-tabs", active_tab="cp-assets"),
    ])


def _citadel_tab():
    """Tab 9 layout: controls (left) + graph (right) + export row."""
    return _chart_tab_layout(
        _citadel_controls, "citadel-graph", "btc_citadel", mc_prefix="cp",
    )
