"""Custom Time Axis panel — Tab 1 only.

See spec: docs/superpowers/specs/2026-04-13-custom-time-axis-design.md §2.
"""
from __future__ import annotations

import pandas as pd
from dash import dcc, html
import dash_bootstrap_components as dbc

from colors import DIM_TEXT, UI_FONT_SM
from layout.common import (
    _section_card, _row, _lbl, _STYLE_HIDDEN, _CB_MARGIN, _MUTED_STYLE,
)
from _custom_time_presets import CAL_PRESETS, BLK_PRESETS


def _cal_dropdown_options():
    opts = [{"label": lbl, "value": key} for key, _d, lbl in CAL_PRESETS]
    opts.append({"label": "Custom…", "value": "custom"})
    return opts


def _blk_dropdown_options(block_cap):
    opts = [{"label": lbl, "value": key} for key, _h, lbl in BLK_PRESETS]
    if block_cap is not None:
        opts.append({"label": "Custom…", "value": "custom"})
    return opts


def custom_time_panel():
    """Collapsible Custom Time Axis panel for Tab 1."""
    # Imported here (not at module top) to avoid pulling in custom_fit's
    # heavy imports during bare layout-module loading in tests.
    from engines.custom_fit import _BLOCK_CAP

    body = html.Div(id="cta-body", style=_STYLE_HIDDEN, children=[
        html.Small(
            "\u26a0 Custom Time Axis changes only affect this tab. Other tabs "
            "stay on the default axis (years since 2009-07-25).",
            style={"color": DIM_TEXT, "fontSize": UI_FONT_SM,
                    "display": "block", "marginBottom": "6px"},
        ),

        # Row 2 — Time scale
        _row(
            html.Div([
                _lbl("Time scale"),
                dcc.RadioItems(
                    id="cta-scale",
                    options=[
                        {"label": " Calendar (years)", "value": "calendar"},
                        {"label": " Blockheight (blocks)", "value": "block"},
                    ],
                    value="calendar",
                    labelStyle={"display": "inline-block",
                                 "marginRight": "12px"},
                    inputStyle=_CB_MARGIN,
                ),
            ]),
        ),

        # Row 3a — Calendar t₀
        html.Div(id="cta-t0-cal-wrap", children=[
            _lbl("t\u2080 (calendar)"),
            dcc.Dropdown(id="cta-t0-cal",
                          options=_cal_dropdown_options(),
                          value="optimal", clearable=False),
            html.Div(id="cta-t0-cal-custom-wrap", style=_STYLE_HIDDEN, children=[
                _lbl("Custom date"),
                dcc.DatePickerSingle(
                    id="cta-t0-cal-custom",
                    min_date_allowed="2008-10-31",
                    max_date_allowed="2015-12-31",
                    display_format="YYYY-MM-DD",
                    initial_visible_month=pd.Timestamp("2015-12-01"),
                    placeholder="YYYY-MM-DD",
                ),
            ]),
        ]),

        # Row 3b — Block t₀ (hidden unless scale=block)
        html.Div(id="cta-t0-blk-wrap", style=_STYLE_HIDDEN, children=[
            _lbl("t\u2080 (block)"),
            dcc.Dropdown(id="cta-t0-blk",
                          options=_blk_dropdown_options(_BLOCK_CAP),
                          value="block_0", clearable=False),
            html.Div(id="cta-t0-blk-custom-wrap", style=_STYLE_HIDDEN, children=[
                _lbl("Custom block"),
                dbc.Input(
                    id="cta-t0-blk-custom", type="number",
                    min=0, step=1, max=_BLOCK_CAP or 10**9,
                    debounce=True, placeholder="block number",
                ),
            ]),
        ]),

        # Row 4 — Weighting
        _row(
            html.Div([
                _lbl("Weighting"),
                dcc.Dropdown(
                    id="cta-weighting",
                    options=[
                        {"label": "Unweighted", "value": "none"},
                        {"label": "1/t", "value": "inv_t"},
                        {"label": "1/\u221at", "value": "inv_sqrt_t"},
                        {"label": "Uniform log-t density",
                         "value": "log_density"},
                    ],
                    value="none",
                    clearable=False,
                ),
                html.Small(
                    "Applies to PL, QR, and BM-floor. Exponential ignores "
                    "(linear-t fit; weighting has no effect).",
                    style=_MUTED_STYLE,
                ),
            ]),
        ),

        # Row 5 — Model subset
        _row(
            html.Div([
                _lbl("Models"),
                dcc.Checklist(
                    id="cta-models",
                    options=[
                        {"label": " Power Law", "value": "pl"},
                        {"label": " Quantile Regression", "value": "qr"},
                        {"label": " BM floor", "value": "bm_floor"},
                        {"label": " Exponential", "value": "exp"},
                    ],
                    value=["pl", "qr", "bm_floor"],
                    labelStyle={"display": "block"},
                    inputStyle=_CB_MARGIN,
                ),
            ]),
        ),

        # Footer — Fit status
        html.Div(
            id="cta-status",
            style={"color": DIM_TEXT, "fontSize": UI_FONT_SM,
                    "marginTop": "8px"},
            children="Press Activate to fit.",
        ),

        # $1M projection table link
        dbc.Button(
            "\U0001F4CA $1M Projection Table",
            id="cta-open-proj-modal",
            color="link",
            size="sm",
            style={"padding": "4px 0", "marginTop": "4px",
                    "fontSize": UI_FONT_SM},
        ),

        # Store for router handoff to update_bubble
        dcc.Store(id="bub-redraw-tick", data=0),
    ])

    modal = dbc.Modal(
        id="cta-proj-modal",
        size="xl",
        scrollable=True,
        is_open=False,
        children=[
            dbc.ModalHeader(dbc.ModalTitle(
                "Projected $1M Month + Fit Exponent (all t\u2080 × all weightings)"
            )),
            dbc.ModalBody(
                dcc.Loading(
                    id="cta-proj-loading",
                    type="default",
                    children=html.Div(id="cta-proj-modal-body"),
                ),
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="cta-proj-modal-close",
                            className="ms-auto", n_clicks=0),
            ),
        ],
    )

    panel = _section_card(
        "Custom Time Axis",
        dcc.Checklist(
            id="cta-active",
            options=[{"label": " Activate Custom Time Axis", "value": "yes"}],
            value=[],
            labelStyle={"display": "block", "fontWeight": "bold"},
            inputStyle=_CB_MARGIN,
        ),
        body,
        no_hover=True,
    )
    return html.Div([panel, modal])
