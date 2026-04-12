"""Citadel Planner (Tab 9) — layout with tabbed sub-panels."""

import pandas as pd
from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from tab_defaults import CITADEL
from colors import (DIM_TEXT, BOOTSTRAP_LIGHT_BG, BOOTSTRAP_BORDER,
                    FALLBACK_MODEL_GRAY, MODAL_BG, BLACK, _hex_alpha,
                    UI_FONT_SM, UI_FONT_MD, UI_FONT_BASE, UI_FONT_LG, UI_FONT_XL,
                    OVERLAY_CARD_SHADOW_ALPHA, OVERLAY_DIM_ALPHA)
from layout.common import (
    _section_card, _lbl, _ctrl_card, _q_options,
    _chart_toggles, _btc_usd_dropdown, _legend_pos_dropdown,
    _STYLE_HIDDEN, _STYLE_HINT, _export_row, _chart_tab_layout,
    _CB_MARGIN, _INFL_LABEL, _plot_appearance_controls,
    _palette_selector,
)
from layout.mc_controls import _mc_controls
from layout.citadel_tax import tax_toggle_widget, tax_config_modal, tax_summary_panel
from citadel_presets import WEALTH_LEVELS, MACRO_REGIMES, RULE_SETS, START_YEARS


def _assets_panel():
    """Sub-tab 1: BTC stack, cash, reserves, investments."""
    return html.Div([
        _section_card("Bitcoin Stack",
            _lbl("Starting BTC"),
            dbc.Input(id="cp-stack", type="number", value=CITADEL["start_stack"], min=0, step=0.01),
            dcc.Checklist(id="cp-use-lots", options=[{"label": " Use Stack Tracker lots", "value": "yes"}],
                          value=[], inputStyle=_CB_MARGIN),
        ),
        _section_card("Cash Account",
            _lbl("Initial balance ($)"),
            dbc.Input(id="cp-cash-init", type="number", value=CITADEL["cash_initial"], min=0, step=1),
            _lbl("Interest rate (% / yr)"),
            dbc.Input(id="cp-cash-rate", type="number", value=CITADEL["cash_rate"], min=0, step=0.1),
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
                dbc.Col(dbc.Input(id="cp-res-short-init", type="number", value=CITADEL["res_short_init"], min=0, step=1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-res-short-rate", type="number", value=CITADEL["res_short_rate"], min=0, step=0.1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-res-short-vol", type="number", value=CITADEL["res_short_vol"], min=0, step=0.1, size="sm"), width=3),
            ], className="g-1 mb-1"),
            # Medium
            dbc.Row([
                dbc.Col(html.Small("Medium"), width=3),
                dbc.Col(dbc.Input(id="cp-res-med-init", type="number", value=CITADEL["res_med_init"], min=0, step=1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-res-med-rate", type="number", value=CITADEL["res_med_rate"], min=0, step=0.1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-res-med-vol", type="number", value=CITADEL["res_med_vol"], min=0, step=0.1, size="sm"), width=3),
            ], className="g-1 mb-1"),
            # Long
            dbc.Row([
                dbc.Col(html.Small("Long"), width=3),
                dbc.Col(dbc.Input(id="cp-res-long-init", type="number", value=CITADEL["res_long_init"], min=0, step=1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-res-long-rate", type="number", value=CITADEL["res_long_rate"], min=0, step=0.1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-res-long-vol", type="number", value=CITADEL["res_long_vol"], min=0, step=0.1, size="sm"), width=3),
            ], className="g-1"),
        ),
        _section_card("Investment Account",
            html.Small("2 bins: Equities (stocks) and Bonds. Basis = what you originally paid.", style=_STYLE_HINT),
            dbc.Row([
                dbc.Col(html.Small("Bin", className="fw-bold"), width=2),
                dbc.Col(html.Small("Value ($)", className="fw-bold"), width=3),
                dbc.Col(html.Small("Basis ($)", className="fw-bold"), width=2),
                dbc.Col(html.Small("Return (%)", className="fw-bold"), width=3),
                dbc.Col(html.Small("Vol (%)", className="fw-bold"), width=2),
            ], className="g-1 mb-1"),
            dbc.Row([
                dbc.Col(html.Small("Equities"), width=2),
                dbc.Col(dbc.Input(id="cp-inv-eq-init", type="number", value=CITADEL["inv_eq_init"], min=0, step=1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-inv-eq-basis", type="number", value=CITADEL["inv_eq_basis"], min=0, step=1, size="sm"), width=2),
                dbc.Col(dbc.Input(id="cp-inv-eq-rate", type="number", value=CITADEL["inv_eq_rate"], min=0, step=0.1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-inv-eq-vol", type="number", value=CITADEL["inv_eq_vol"], min=0, step=0.1, size="sm"), width=2),
            ], className="g-1 mb-1"),
            dbc.Row([
                dbc.Col(html.Small("Bonds"), width=2),
                dbc.Col(dbc.Input(id="cp-inv-bd-init", type="number", value=CITADEL["inv_bd_init"], min=0, step=1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-inv-bd-basis", type="number", value=CITADEL["inv_bd_basis"], min=0, step=1, size="sm"), width=2),
                dbc.Col(dbc.Input(id="cp-inv-bd-rate", type="number", value=CITADEL["inv_bd_rate"], min=0, step=0.1, size="sm"), width=3),
                dbc.Col(dbc.Input(id="cp-inv-bd-vol", type="number", value=CITADEL["inv_bd_vol"], min=0, step=0.1, size="sm"), width=2),
            ], className="g-1"),
        ),
    ])


def _spending_panel():
    """Sub-tab 2: Monthly spending, inflation, growth."""
    return html.Div([
        _section_card("Monthly Spending",
            _lbl("Monthly spending ($)"),
            dbc.Input(id="cp-spend", type="number", value=CITADEL["monthly_spend"], min=0, step=1),
            _lbl(_INFL_LABEL),
            dbc.Input(id="cp-infl", type="number", value=CITADEL["inflation"], min=0, max=100, step=0.5),
            _lbl("Spending growth above inflation (% / yr)"),
            dbc.Input(id="cp-spend-growth", type="number", value=CITADEL["spend_growth"], min=0, max=100, step=0.5),
        ),
    ])


def _trigger_section(prefix, label, default_thresh, default_mode, default_rate, default_dur, split_label,
                     split_defaults=None):
    """Build a rebalancing trigger section (high-Q or low-Q)."""
    if split_defaults is None:
        split_defaults = (20, 20, 20, 10, 20, 10)
    return _section_card(label,
        dcc.Checklist(id=f"cp-{prefix}-enable",
                      options=[{"label": " Enable", "value": "yes"}],
                      value=["yes"],
                      inputStyle=_CB_MARGIN,
                      style={"marginBottom": "6px"}),
        _lbl("Threshold (quantile %)"),
        dbc.Input(id=f"cp-{prefix}-thresh", type="number", value=default_thresh,
                  min=0, max=100, step=1),
        _lbl("Action mode"),
        dcc.RadioItems(id=f"cp-{prefix}-mode",
            options=[{"label": " Gradual", "value": "gradual"},
                     {"label": " Lump", "value": "lump"}],
            value=default_mode, inline=True,
            inputStyle=_CB_MARGIN, labelStyle={"marginRight": "12px"}),
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
            dbc.Col([html.Small("Cash"), dbc.Input(id=f"cp-{prefix}-split-cash", type="number", value=split_defaults[0], min=0, max=100, step=1, size="sm")], width=2),
            dbc.Col([html.Small("Short"), dbc.Input(id=f"cp-{prefix}-split-rs", type="number", value=split_defaults[1], min=0, max=100, step=1, size="sm")], width=2),
            dbc.Col([html.Small("Med"), dbc.Input(id=f"cp-{prefix}-split-rm", type="number", value=split_defaults[2], min=0, max=100, step=1, size="sm")], width=2),
            dbc.Col([html.Small("Long"), dbc.Input(id=f"cp-{prefix}-split-rl", type="number", value=split_defaults[3], min=0, max=100, step=1, size="sm")], width=2),
            dbc.Col([html.Small("Eq"), dbc.Input(id=f"cp-{prefix}-split-eq", type="number", value=split_defaults[4], min=0, max=100, step=1, size="sm")], width=2),
            dbc.Col([html.Small("Bd"), dbc.Input(id=f"cp-{prefix}-split-bd", type="number", value=split_defaults[5], min=0, max=100, step=1, size="sm")], width=2),
        ], className="g-1"),
    )


def _rules_panel():
    """Sub-tab 3: Rebalancing triggers, floor rules, Saylor Fortifier."""
    return html.Div([
        _trigger_section("high-q", "High-Quantile Trigger — Take Profits",
                        CITADEL["high_q_trigger"], CITADEL["high_q_mode"],
                        CITADEL["high_q_rate"], CITADEL["high_q_dur"],
                        "Proceeds distribution (%)",
                        split_defaults=(
                            CITADEL["high_q_split_cash"], CITADEL["high_q_split_rs"],
                            CITADEL["high_q_split_rm"], CITADEL["high_q_split_rl"],
                            CITADEL["high_q_split_eq"], CITADEL["high_q_split_bd"],
                        )),
        _trigger_section("low-q", "Low-Quantile Trigger — Accumulate BTC",
                        CITADEL["low_q_trigger"], CITADEL["low_q_mode"],
                        CITADEL["low_q_rate"], CITADEL["low_q_dur"],
                        "Source allocation (%)",
                        split_defaults=(
                            CITADEL["low_q_split_cash"], CITADEL["low_q_split_rs"],
                            CITADEL["low_q_split_rm"], CITADEL["low_q_split_rl"],
                            CITADEL["low_q_split_eq"], CITADEL["low_q_split_bd"],
                        )),
        _section_card("Global Lump Cooldown",
            _lbl("Minimum periods between lump actions"),
            dbc.Input(id="cp-lump-cooldown", type="number", value=CITADEL["lump_cooldown"], min=1, step=1),
        ),
        _section_card("Account Floor Rules",
            html.Small("Minimum balances maintained each period", style=_STYLE_HINT),
            _lbl("Cash floor ($)"),
            dbc.Input(id="cp-cash-floor", type="number", value=CITADEL["cash_floor"], min=0, step=1),
            _lbl("Reserve Short floor ($)"),
            dbc.Input(id="cp-res-short-floor", type="number", value=CITADEL["res_short_floor"], min=0, step=1),
            _lbl("Reserve Medium floor ($)"),
            dbc.Input(id="cp-res-med-floor", type="number", value=CITADEL["res_med_floor"], min=0, step=1),
            _lbl("Reserve Long floor ($)"),
            dbc.Input(id="cp-res-long-floor", type="number", value=CITADEL["res_long_floor"], min=0, step=1),
            _lbl("Cash floor annual increase (%)"),
            dbc.Input(id="cp-cash-floor-growth", type="number", value=CITADEL["cash_floor_growth"], min=0, max=50, step=0.5),
            _lbl("Reserve floor annual increase (%)"),
            dbc.Input(id="cp-res-floor-growth", type="number", value=CITADEL["reserve_floor_growth"], min=0, max=50, step=0.5),
        ),
        # Saylor Citadel Fortifier
        dcc.Checklist(id="cp-scf-enable",
            options=[{"label": " Enable Saylor Citadel Fortifier", "value": "yes"}],
            value=[], inputStyle=_CB_MARGIN,
            style={"marginBottom": "8px"}),
        html.Div(id="cp-scf-body", style=_STYLE_HIDDEN, children=[
            _section_card("Saylor Citadel Fortifier",
                _lbl("Loan amount ($)"),
                dbc.Input(id="cp-scf-amount", type="number", value=CITADEL["scf_amount"], min=0, step=1),
                _lbl("Loan type"),
                dcc.RadioItems(id="cp-scf-type",
                    options=[{"label": " Term", "value": "term"},
                             {"label": " Perpetual", "value": "perpetual"}],
                    value=CITADEL["scf_type"], inline=True,
                    inputStyle=_CB_MARGIN, labelStyle={"marginRight": "12px"}),
                _lbl("Interest rate (% / yr)"),
                dbc.Input(id="cp-scf-rate", type="number", value=CITADEL["scf_rate"], min=0, step=0.1),
                _lbl("Term (months) — term loan only"),
                dbc.Input(id="cp-scf-term", type="number", value=CITADEL["scf_term"], min=1, step=1),
                _lbl("Repayment trigger (N x rate) — perpetual only"),
                dbc.Input(id="cp-scf-trigger", type="number", value=CITADEL["scf_repay_trigger"], min=0.1, step=0.1),
            ),
        ]),
    ])


def _dd_section(title, *children):
    """Section wrapper for controls with dropdowns — avoids dbc.Card which
    clips dropdown menus on iOS due to transform creating a stacking context."""
    return html.Div([
        html.Div(title, style={"fontWeight": "bold", "fontSize": UI_FONT_BASE,
                                "color": DIM_TEXT, "marginBottom": "4px",
                                "textTransform": "uppercase", "letterSpacing": "0.03em"}),
        *children,
    ], style={"marginBottom": "12px", "padding": "8px",
              "background": BOOTSTRAP_LIGHT_BG, "borderRadius": "8px",
              "border": f"1px solid {BOOTSTRAP_BORDER}"})


def _sim_panel():
    """Sub-tab 4: Simulation settings, quantiles, MC, chart toggles."""
    return html.Div([
        _dd_section("Simulation",
            _lbl("Year range"),
            dcc.RangeSlider(id="cp-yr-range", min=2025, max=2080,
                value=[CITADEL["start_yr"], CITADEL["end_yr"]], step=1,
                marks={y: f"'{y % 100:02d}" for y in range(2030, 2081, 10)}),
            _lbl("Frequency"),
            dcc.Dropdown(id="cp-freq",
                options=[{"label": "Monthly", "value": "Monthly"},
                         {"label": "Quarterly", "value": "Quarterly"},
                         {"label": "Annually", "value": "Annually"}],
                value=CITADEL["freq"], clearable=False),
            _lbl("Price model"),
            dcc.Dropdown(id="cp-model-src",
                options=[{"label": "Bubble Model", "value": "bub"},
                         {"label": "Power Law", "value": "pl"}],
                value=CITADEL["price_model"], clearable=False),
        ),
        _dd_section("Dollar Asset Returns",
            html.Small("How non-BTC assets grow each period.",
                       style=_STYLE_HINT),
            dcc.Dropdown(id="cp-asset-model",
                options=[{"label": "Fixed Rates (your input rates/vol)", "value": "lognormal"},
                         {"label": "Historical Regimes", "value": "markov"}],
                value=CITADEL["asset_return_model"], clearable=False),
            html.Div(id="cp-asset-model-info",
                     style={"display": "none", "marginTop": "6px"}),
        ),
        tax_toggle_widget(),
        _dd_section("BTC Price Scenario",
            html.Small("Select one quantile for the deterministic BTC price path. "
                       "Lower = more pessimistic, higher = more optimistic.",
                       style=_STYLE_HINT),
            dcc.Dropdown(id="cp-qs",
                options=[{"label": f"Q{q*100:g}%", "value": q}
                         for q in [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25,
                                   0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80,
                                   0.85, 0.90, 0.95, 0.99, 0.999]],
                value=CITADEL["selected_qs"][0], clearable=False),
        ),
        _dd_section("Display",
            _lbl("Display mode"),
            dcc.Dropdown(id="cp-disp",
                options=[{"label": "USD (total portfolio)", "value": "usd_total"},
                         {"label": "USD (per asset class)", "value": "usd_per_asset"},
                         {"label": "BTC Holdings", "value": "btc"}],
                value=CITADEL["disp_mode"], clearable=False),
        ),
        _mc_controls("cp", amount_label="Monthly spending ($)",
                     amount_default=5000, show_inflation=True, show_stack=True,
                     default_entry_q=10,
                     shared_controls={"amount", "infl", "freq", "stack"}),
        _section_card("Chart Settings",
            _chart_toggles("cp", ["annotate", "log_y", "show_legend"]),
            *_legend_pos_dropdown("cp", CITADEL["legend_pos"]),
            html.Div([
                dbc.Button("Show All", id="cp-legend-all", size="sm",
                           color="secondary", outline=True, className="me-1",
                           style={"fontSize": UI_FONT_MD, "padding": "1px 8px"}),
                dbc.Button("Hide All", id="cp-legend-none", size="sm",
                           color="secondary", outline=True,
                           style={"fontSize": UI_FONT_MD, "padding": "1px 8px"}),
            ], style={"marginTop": "4px"}),
        ),
        _palette_selector("cp"),
    ])


def _citadel_controls():
    """All controls in inner tabbed sub-panels."""
    return html.Div([
        # Celery task polling interval (disabled until MC task submitted)
        dcc.Interval(id="cp-celery-poll", interval=3000, disabled=True,
                     max_intervals=100, n_intervals=0),
        # Run button + instructions
        dbc.Button("\u25b6  Run Simulation", id="cp-run-btn",
                   color="warning", className="w-100 mb-2 fw-bold",
                   style={"fontSize": UI_FONT_XL, "letterSpacing": "0.03em"}),
        html.Small("\u25b6 runs a single deterministic projection (free). "
                   "\u26a1 runs Monte Carlo with multiple stochastic paths.",
                   style={"color": FALLBACK_MODEL_GRAY, "display": "block", "marginBottom": "8px",
                          "fontSize": UI_FONT_MD}),
        dbc.Row([
            dbc.Col(
                dbc.Button("\u2193 Save Scenario", id="cp-save-btn",
                           color="secondary", outline=True, size="sm",
                           className="w-100", disabled=True),
                width=6,
            ),
            dbc.Col(
                dcc.Upload(
                    id="cp-scenario-upload",
                    children=dbc.Button("\u2191 Load", color="secondary",
                                        outline=True, size="sm",
                                        className="w-100"),
                    accept=".json",
                ),
                width=6,
            ),
        ], className="mb-2 gx-2"),
        html.Div(id="cp-load-status", className="text-muted small mb-2"),
        dcc.Store(id="cp-save-prep", storage_type="memory"),
        dcc.Store(id="cp-save-dl-dummy", storage_type="memory"),
        dcc.Store(id="cp-load-store", storage_type="memory"),
        html.Div(id="cp-load-overlay", children=[
            html.Div(id="cp-load-overlay-text",
                     children="\u23f3 Loading scenario...",
                     style={"background": MODAL_BG, "borderRadius": "8px",
                            "padding": "18px 32px", "fontSize": "15px",
                            "boxShadow": f"0 2px 16px {_hex_alpha(BLACK, OVERLAY_CARD_SHADOW_ALPHA)}"}),
        ], style={"display": "none", "position": "fixed", "top": 0, "left": 0,
                  "width": "100vw", "height": "100vh", "zIndex": 1060,
                  "background": _hex_alpha(BLACK, OVERLAY_DIM_ALPHA),
                  "display": "none",
                  "justifyContent": "center", "alignItems": "center"}),
        # ── Quick Scenarios ──────────────────────────────────────────────
        _ctrl_card(
            html.Div([
                html.Span("Quick Scenarios (Free)"),
                html.Span(" — stale", id="cp-scenario-stale",
                          className="text-warning small",
                          style={"display": "none"}),
            ], className="ctrl-section-header"),
            # Wealth row
            html.Div([
                html.Small("Wealth", className="text-muted me-2",
                           style={"minWidth": "50px"}),
                dbc.ButtonGroup([
                    dbc.Button(WEALTH_LEVELS[k]["label"].split()[0], id=f"cp-pill-{k}",
                               outline=(k != "starter"), color="primary", size="sm")
                    for k in WEALTH_LEVELS
                ], size="sm"),
            ], className="d-flex align-items-center mb-1"),
            # Regime row
            html.Div([
                html.Small("Regime", className="text-muted me-2",
                           style={"minWidth": "50px"}),
                dbc.ButtonGroup([
                    dbc.Button(MACRO_REGIMES[k]["label"], id=f"cp-pill-{k}",
                               outline=(k != "neutral"), color="primary", size="sm")
                    for k in MACRO_REGIMES
                ], size="sm"),
            ], className="d-flex align-items-center mb-1"),
            # Rules row
            html.Div([
                html.Small("Rules", className="text-muted me-2",
                           style={"minWidth": "50px"}),
                dbc.ButtonGroup([
                    dbc.Button(RULE_SETS[k]["label"], id=f"cp-pill-{k}",
                               outline=(k != "no_rebal"), color="primary", size="sm")
                    for k in RULE_SETS
                ], size="sm"),
            ], className="d-flex align-items-center mb-1"),
            # Start year dropdown
            html.Div([
                html.Small("Start", className="text-muted me-2",
                           style={"minWidth": "50px"}),
                dcc.Dropdown(id="cp-scenario-start-yr",
                    options=[
                        {"label": html.Span(str(y), style={"fontWeight": "bold"}), "value": y}
                        if y in START_YEARS else
                        {"label": str(y), "value": y}
                        for y in range(2025, 2041)
                    ],
                    value=2035, clearable=False,
                    style={"width": "100px", "fontSize": UI_FONT_LG}),
            ], className="d-flex align-items-center mb-1"),
            html.Small("800 simulations per scenario",
                       style={"color": FALLBACK_MODEL_GRAY, "fontSize": UI_FONT_SM,
                              "display": "block", "marginTop": "4px"}),
        ),
        # Scenario stores
        dcc.Store(id="cp-scenario-wealth", data="starter"),
        dcc.Store(id="cp-scenario-regime", data="neutral"),
        dcc.Store(id="cp-scenario-rules", data="no_rebal"),
        dcc.Store(id="cp-scenario-bands", storage_type="memory"),
        dcc.Store(id="cp-scenario-active", data=None),
        dbc.Tabs([
            dbc.Tab(_assets_panel(), label="Assets", tab_id="cp-assets"),
            dbc.Tab(_spending_panel(), label="Spending", tab_id="cp-spending"),
            dbc.Tab(_rules_panel(), label="Rules", tab_id="cp-rules"),
            dbc.Tab(_sim_panel(), label="Simulation", tab_id="cp-sim"),
        ], id="cp-inner-tabs", active_tab="cp-assets"),
        _section_card("Plot Appearance", *_plot_appearance_controls("cp")),
        tax_config_modal(),
    ])


def _citadel_tab():
    """Tab 9 layout: controls (left) + graph (right) + export row.

    Loads a pre-computed default figure from Redis cache (if available)
    so the chart renders instantly on first visit without a callback."""
    import plotly.io as pio
    initial_fig = None
    try:
        from cache import get_citadel_cached
        cached = get_citadel_cached("default:bub:q0.25")
        if cached and cached.get("figure"):
            initial_fig = pio.from_json(cached["figure"])
    except Exception:
        pass

    layout = _chart_tab_layout(
        _citadel_controls, "citadel-graph", "btc_citadel", mc_prefix="cp",
        start_collapsed=False,
    )
    # Inject initial figure into the dcc.Graph component
    if initial_fig is not None:
        from layout.common import _inject_initial_figure
        _inject_initial_figure(layout, "citadel-graph", initial_fig)
    # Insert tax summary panel below the chart (inside the graph column)
    graph_col = layout.children[1]
    graph_col.children.append(tax_summary_panel())
    return layout
