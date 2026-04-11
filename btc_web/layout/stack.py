"""Tab 6 — Stack Tracker layout."""

import pandas as pd

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

import _app_ctx
from layout.common import _tab_hints, _ctrl_card, _section_card, _lbl
from tab_defaults import STACK
from colors import (MODAL_BG, TEXT_COLOR, BOOTSTRAP_BORDER, BOOTSTRAP_LIGHT_BG,
                    BOOTSTRAP_TABLE_SELECT_BG, BOOTSTRAP_TABLE_SELECT_BORDER)


def _stack_tracker_tab():
    return html.Div([
        _tab_hints("stack"),
        html.Div(id="snapshot-lots-banner"),
        dbc.Row([
            # ── table ────────────────────────────────────────────────────────
            dbc.Col([
                dash_table.DataTable(
                    id="lots-table",
                    columns=[
                        {"name":"Date",       "id":"date"},
                        {"name":"BTC",        "id":"btc"},
                        {"name":"Price $/BTC","id":"price"},
                        {"name":"Total Paid", "id":"total_paid"},
                        {"name":"Percentile", "id":"pct_q"},
                        {"name":"Notes",      "id":"notes"},
                    ],
                    data=[],
                    row_selectable="multi",
                    selected_rows=[],
                    style_table={"overflowX":"auto"},
                    style_cell={"backgroundColor":MODAL_BG,"color":TEXT_COLOR,
                                "border":f"1px solid {BOOTSTRAP_BORDER}","padding":"4px 8px",
                                "fontSize":"13px"},
                    style_header={"backgroundColor":BOOTSTRAP_LIGHT_BG,"color":TEXT_COLOR,
                                  "fontWeight":"bold"},
                    style_data_conditional=[
                        {"if":{"state":"selected"},"backgroundColor":BOOTSTRAP_TABLE_SELECT_BG,
                         "border":f"1px solid {BOOTSTRAP_TABLE_SELECT_BORDER}"},
                    ],
                    page_size=20,
                ),
                html.Div(id="lots-summary", className="mt-2 text-muted small"),
            ], width=8),

            # ── controls ─────────────────────────────────────────────────────
            dbc.Col([
                _section_card("Add Lot",
                    _lbl("Date"), dbc.Input(id="lot-date", type="date",
                        value=str(pd.Timestamp.today().date()), size="sm"),
                    _lbl("BTC amount"),
                    dbc.Input(id="lot-btc", type="number", value=STACK["lot_btc"],
                              min=0, step=0.0001, size="sm"),
                    _lbl("Price ($/BTC)"),
                    dbc.Input(id="lot-price", type="number", value=STACK["lot_price"],
                              min=0, step=1, size="sm"),
                    _lbl("Notes"),
                    dbc.Input(id="lot-notes", type="text", value="", size="sm"),
                    html.Div(id="lot-pct-preview", className="mt-1 small text-info"),
                    dbc.Button("Add Lot", id="lot-add-btn", color="primary",
                               size="sm", className="mt-2 w-100"),
                ),
                _ctrl_card(
                    dbc.Button("Delete selected", id="lot-del-btn",
                               color="danger", size="sm", className="w-100 mb-1"),
                    dbc.Button("Clear all", id="lot-clear-btn",
                               color="warning", size="sm", className="w-100"),
                ),
                _section_card("Export / Import",
                    dbc.Button("\u2b07 Export JSON", id="lots-export-btn",
                               color="secondary", size="sm", className="w-100 mb-2"),
                    html.Hr(className="my-1"),
                    dcc.Upload(
                        id="lots-import-upload",
                        children=dbc.Button("\u2b06 Import JSON", color="secondary",
                                            size="sm", className="w-100"),
                        accept=".json",
                        multiple=False,
                    ),
                    html.Div(id="lots-import-status", className="mt-1 small"),
                ),
            ], width=4),
        ]),
    ], className="p-2")
