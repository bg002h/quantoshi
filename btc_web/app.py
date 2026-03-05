"""app.py — Quantoshi web app (Plotly Dash).

Run locally:
    cd /scratch/code/bitcoinprojections
    bash run_web.sh
    → http://localhost:8050  (or http://<your-ip>:8050 from phone/tablet)

Multi-user server deploy (via btc-web.service):
    gunicorn 'btc_web.app:server' -b 0.0.0.0:8050 -w 4

Privacy model:
    - Lot data lives exclusively in each user's browser localStorage.
    - Nothing is persisted server-side; lots only transit server memory
      transiently during Dash callbacks (never written to disk).
"""

import sys
from pathlib import Path

# ── make btc_app/ importable ──────────────────────────────────────────────────
_HERE    = Path(__file__).parent
_ROOT    = _HERE.parent
_BTC_APP = _ROOT / "btc_app"
for _p in (_ROOT, _BTC_APP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json
import gzip
import base64
import math
import urllib.request
import pandas as pd
import plotly.graph_objects as go

import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx, callback, no_update
import dash_bootstrap_components as dbc
from flask import request as flask_request

from btc_core import (load_model_data, _find_lot_percentile, fmt_price,
                      yr_to_t, today_t, leo_weighted_entry, qr_price)
from figures import (build_bubble_figure, build_heatmap_figure,
                     build_dca_figure, build_retire_figure,
                     build_supercharge_figure)

# ── load model (once at startup) ──────────────────────────────────────────────
M = load_model_data()

_ALL_QS = M.QR_QUANTILES   # full list from model
_DEF_QS = [q for q in [0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
           if q in M.qr_fits]


def _nearest_quantile(target, qs):
    """Snap a percentile value to the nearest available quantile."""
    return min(qs, key=lambda q: abs(q - target))


def _startup_heatmap_defaults():
    """Fetch live BTC price at startup; return entry percentile (0–100 scale)."""
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        with urllib.request.urlopen(url, timeout=5) as r:
            price = float(json.loads(r.read())["price"])
        pct = _find_lot_percentile(today_t(M.genesis), price, M.qr_fits)
        if pct is not None:
            return round(pct * 100, 1)   # e.g. 7.5
    except Exception:
        pass
    return 50.0   # fallback


_HM_ENTRY_Q_DEFAULT = _startup_heatmap_defaults()

# ── Snapshot / URL state helpers ───────────────────────────────────────────────

_SNAPSHOT_CONTROLS = [
    ("bub-qs",            "value"),
    ("bub-xscale",        "value"),
    ("bub-yscale",        "value"),
    ("bub-xrange",        "value"),
    ("bub-yrange",        "value"),
    ("bub-toggles",       "value"),
    ("bub-bubble-toggles","value"),
    ("bub-n-future",      "value"),
    ("bub-ptsize",        "value"),
    ("bub-ptalpha",       "value"),
    ("bub-stack",         "value"),
    ("bub-show-stack",    "value"),
    ("bub-use-lots",      "value"),
    ("hm-entry-yr",       "value"),
    ("hm-entry-q",        "value"),
    ("hm-exit-range",     "value"),
    ("hm-exit-qs",        "value"),
    ("hm-mode",           "value"),
    ("hm-b1",             "value"),
    ("hm-b2",             "value"),
    ("hm-c-lo",           "value"),
    ("hm-c-mid1",         "value"),
    ("hm-c-mid2",         "value"),
    ("hm-c-hi",           "value"),
    ("hm-grad",           "value"),
    ("hm-vfmt",           "value"),
    ("hm-cell-fs",        "value"),
    ("hm-toggles",        "value"),
    ("hm-stack",          "value"),
    ("hm-use-lots",       "value"),
    ("dca-stack",         "value"),
    ("dca-use-lots",      "value"),
    ("dca-amount",        "value"),
    ("dca-freq",          "value"),
    ("dca-yr-range",      "value"),
    ("dca-disp",          "value"),
    ("dca-toggles",       "value"),
    ("dca-qs",            "value"),
    ("dca-sc-enable",     "value"),
    ("dca-sc-loan",       "value"),
    ("dca-sc-rate",       "value"),
    ("dca-sc-term",       "value"),
    ("ret-stack",         "value"),
    ("ret-use-lots",      "value"),
    ("ret-wd",            "value"),
    ("ret-freq",          "value"),
    ("ret-yr-range",      "value"),
    ("ret-infl",          "value"),
    ("ret-disp",          "value"),
    ("ret-toggles",       "value"),
    ("ret-qs",            "value"),
    ("sc-stack",          "value"),
    ("sc-use-lots",       "value"),
    ("sc-start-yr",       "value"),
    ("sc-d0",             "value"),
    ("sc-d1",             "value"),
    ("sc-d2",             "value"),
    ("sc-d3",             "value"),
    ("sc-d4",             "value"),
    ("sc-freq",           "value"),
    ("sc-infl",           "value"),
    ("sc-qs",             "value"),
    ("sc-mode",           "value"),
    ("sc-wd",             "value"),
    ("sc-end-yr",         "value"),
    ("sc-target-yr",      "value"),
    ("sc-disp",           "value"),
    ("sc-toggles",        "value"),
    ("sc-chart-layout",   "value"),
    ("sc-display-q",      "value"),
    ("bub-auto-y",        "value"),
    ("main-tabs",         "active_tab"),
]

_SNAP_PREFIX    = "q2:"   # current format
_SNAP_PREFIX_V1 = "q1:"   # legacy format (dict-based), kept for backward compat


def _encode_snapshot(state_dict, tab_filter=None):
    """v2: positional array — no key names, ~50% smaller than v1.

    If tab_filter is a set of component IDs, only those controls (plus
    main-tabs) are encoded; all others become None and fall back to defaults
    on restore.
    """
    values = []
    for cid, prop in _SNAPSHOT_CONTROLS:
        val = state_dict.get(f"{cid}:{prop}")
        if tab_filter is not None and cid != "main-tabs" and cid not in tab_filter:
            val = None
        values.append(val)
    lots   = state_dict.get("_lots")
    payload = [values, lots]
    j = json.dumps(payload, separators=(',', ':'))
    return base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()


def _decode_snapshot(encoded):
    """Decode v2 (positional array) snapshot."""
    try:
        payload = json.loads(gzip.decompress(base64.urlsafe_b64decode(encoded)))
        values, lots = payload
        state = {f"{cid}:{prop}": val
                 for (cid, prop), val in zip(_SNAPSHOT_CONTROLS, values)
                 if val is not None}
        if lots:
            state["_lots"] = lots
        return state
    except Exception:
        return None


def _decode_snapshot_v1(encoded):
    """Decode legacy v1 (dict-based) snapshot."""
    try:
        return json.loads(gzip.decompress(base64.urlsafe_b64decode(encoded)))
    except Exception:
        return None

def _q_options():
    opts = []
    for q in _ALL_QS:
        pct = q * 100
        lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
        opts.append({"label": lbl, "value": q})
    return opts

# ── app init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)
app.title = "Quantoshi"
server = app.server  # for gunicorn

@server.after_request
def _no_cache_dash_internals(response):
    """Prevent browsers caching Dash layout/dependency info between deploys."""
    if flask_request.path in ('/_dash-layout', '/_dash-dependencies'):
        response.headers.update({
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma':        'no-cache',
            'Expires':       '0',
        })
    return response

# ══════════════════════════════════════════════════════════════════════════════
# Layout helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ctrl_card(*children):
    return dbc.Card(dbc.CardBody(list(children), className="p-2"),
                    className="mb-2 ctrl-card")

def _row(*cols):
    return dbc.Row([dbc.Col(c) for c in cols], className="g-1 mb-1")

def _lbl(text):
    return html.Label(text, className="form-label mb-0 small")

def _export_row(tab_id):
    """Export row — download triggered client-side via Plotly.downloadImage()."""
    return dbc.Row([
        dbc.Col(dcc.Dropdown(
            id=f"{tab_id}-fmt", options=["png","svg","jpeg","webp"], value="png",
            clearable=False, style={"minWidth": "90px"}), width="auto"),
        dbc.Col(dbc.Input(id=f"{tab_id}-fname", value=f"btc_{tab_id}",
                          type="text", size="sm"), width=True),
        dbc.Col(dbc.Button("⬇ Download", id=f"{tab_id}-export-btn",
                           size="sm", color="secondary"), width="auto"),
        # dummy store — clientside callback needs an output target
        dcc.Store(id=f"{tab_id}-dl-dummy"),
    ], className="g-1 mt-1 align-items-center")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Bubble + QR Overlay
# ══════════════════════════════════════════════════════════════════════════════

def _bubble_controls():
    yr_now = pd.Timestamp.today().year
    return html.Div([
        _ctrl_card(
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
            dcc.RangeSlider(id="bub-xrange", min=2010, max=2050,
                            value=[2012, yr_now + 4], step=1,
                            marks={y: f"'{y % 100:02d}" for y in range(2010, 2051, 5)},
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
            dcc.RangeSlider(id="bub-yrange", min=-2, max=8,
                            value=[0, 7], step=0.5,
                            marks={-2:"1¢", 0:"$1", 2:"$100",
                                    4:"$10K", 6:"$1M", 8:"$100M"},
                            tooltip={"always_visible":False}),
        ),
        _ctrl_card(
            dcc.Checklist(id="bub-toggles",
                          options=[{"label":" Shade bands","value":"shade"},
                                   {"label":" Show OLS","value":"show_ols"},
                                   {"label":" Show data","value":"show_data"},
                                   {"label":" Show today","value":"show_today"},
                                   {"label":" Show legend","value":"show_legend"}],
                          value=["shade","show_data","show_today"],
                          labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
        ),
        _ctrl_card(
            _lbl("Bubble"),
            dcc.Checklist(id="bub-bubble-toggles",
                          options=[{"label":" Composite","value":"show_comp"},
                                   {"label":" Support","value":"show_sup"}],
                          value=["show_comp","show_sup"],
                          labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
            _lbl("N future bubbles"),
            dcc.Slider(id="bub-n-future", min=0, max=M.n_future_max,
                       value=3, step=1, marks=None,
                       tooltip={"always_visible":True}),
        ),
        _ctrl_card(
            _lbl("Quantiles"),
            dcc.Checklist(id="bub-qs", options=_q_options(),
                          value=[0.05], labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
        ),
        _ctrl_card(
            _row(
                html.Div([_lbl("Pt size (1–20)"),
                          dbc.Input(id="bub-ptsize", type="number",
                                    value=2, min=1, max=20, size="sm")]),
                html.Div([_lbl("Alpha (0.1–1)"),
                          dbc.Input(id="bub-ptalpha", type="number",
                                    value=0.2, min=0.1, max=1.0, step=0.05, size="sm")]),
            ),
        ),
        _ctrl_card(
            _lbl("Stack (BTC)"),
            dbc.InputGroup([
                dbc.Input(id="bub-stack", type="number", value=0,
                          min=0, step=0.001, size="sm"),
                dbc.InputGroupText(dcc.Checklist(
                    id="bub-show-stack",
                    options=[{"label":" Show","value":"yes"}],
                    value=[], inputStyle={"marginRight":"4px"})),
            ], size="sm"),
        ),
        _ctrl_card(
            dcc.Checklist(id="bub-use-lots",
                          options=[{"label":" Use Stack Tracker lots","value":"yes"}],
                          value=[], inputStyle={"marginRight":"5px"}),
        ),
    ])


def _bubble_tab():
    return dbc.Row([
        dbc.Col(_bubble_controls(), width=3, className="controls-col overflow-auto",
                style={"maxHeight":"85vh"}),
        dbc.Col([
            dcc.Graph(id="bubble-graph", style={"height":"78vh"},
                      config={"toImageButtonOptions":{"format":"png","scale":2,
                                                       "filename":"btc_bubble"}}),
            _export_row("bubble"),
        ], width=9),
    ], className="g-0")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — CAGR Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def _heatmap_controls():
    yr_now = pd.Timestamp.today().year
    return html.Div([
        _ctrl_card(
            _lbl(f"Entry year"),
            dcc.Slider(id="hm-entry-yr", min=2010, max=2039,
                       value=yr_now, step=1, marks=None,
                       tooltip={"always_visible":True}),
            _lbl("Entry percentile (0.1–99.9%)"),
            dbc.Input(id="hm-entry-q", type="number",
                      value=_HM_ENTRY_Q_DEFAULT,
                      min=0.1, max=99.9, step=0.1, size="sm"),
        ),
        _ctrl_card(
            _lbl("Exit year range"),
            dcc.RangeSlider(id="hm-exit-range", min=2010, max=2060,
                            value=[yr_now, yr_now + 15], step=1,
                            marks={y: f"'{y % 100:02d}" for y in range(2010, 2061, 5)},
                            tooltip={"always_visible":False}),
            _lbl("Exit quantiles"),
            dcc.Checklist(id="hm-exit-qs", options=_q_options(),
                          value=_DEF_QS, labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
        ),
        _ctrl_card(
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
            _row(
                html.Div([_lbl("Lo"), dbc.Input(id="hm-c-lo", type="color",
                           value=M.CAGR_SEG_C_LO, style={"height":"28px"})]),
                html.Div([_lbl("Mid1"), dbc.Input(id="hm-c-mid1", type="color",
                           value=M.CAGR_SEG_C_MID1, style={"height":"28px"})]),
                html.Div([_lbl("Mid2"), dbc.Input(id="hm-c-mid2", type="color",
                           value=M.CAGR_SEG_C_MID2, style={"height":"28px"})]),
                html.Div([_lbl("Hi"), dbc.Input(id="hm-c-hi", type="color",
                           value=M.CAGR_SEG_C_HI, style={"height":"28px"})]),
            ),
            _lbl("Gradient steps"),
            dbc.Input(id="hm-grad", type="number", value=32,
                      min=2, max=64, step=1, size="sm"),
        ),
        _ctrl_card(
            _lbl("Cell text"),
            dcc.Dropdown(id="hm-vfmt",
                options=[
                    {"label":"CAGR %",            "value":"cagr"},
                    {"label":"Exit Price",          "value":"price"},
                    {"label":"CAGR % + Price",      "value":"both"},
                    {"label":"CAGR % + Portfolio",  "value":"stack"},
                    {"label":"Portfolio Value",     "value":"port_only"},
                    {"label":"Multiple (×)",        "value":"mult_only"},
                    {"label":"CAGR % + Multiple",   "value":"cagr_mult"},
                    {"label":"Multiple + Portfolio","value":"mult_port"},
                    {"label":"None",                "value":"none"},
                ],
                value="cagr", clearable=False),
            _lbl("Cell font size"),
            dbc.Input(id="hm-cell-fs", type="number", value=9,
                      min=5, max=20, step=1, size="sm"),
            dcc.Checklist(id="hm-toggles",
                          options=[{"label":" Show colorbar","value":"colorbar"}],
                          value=["colorbar"], labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
        ),
        _ctrl_card(
            _lbl("Portfolio BTC"),
            dbc.Input(id="hm-stack", type="number", value=0,
                      min=0, step=0.001, size="sm"),
            dcc.Checklist(id="hm-use-lots",
                          options=[{"label":" Use Stack Tracker lots","value":"yes"}],
                          value=[], inputStyle={"marginRight":"5px"}),
        ),
    ])


def _heatmap_tab():
    return dbc.Row([
        dbc.Col(_heatmap_controls(), width=3, className="controls-col overflow-auto",
                style={"maxHeight":"85vh"}),
        dbc.Col([
            dcc.Graph(id="heatmap-graph", style={"height":"78vh"},
                      config={"toImageButtonOptions":{"format":"png","scale":2,
                                                       "filename":"btc_heatmap"}}),
            _export_row("heatmap"),
        ], width=9),
    ], className="g-0")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — BTC Accumulator (DCA)
# ══════════════════════════════════════════════════════════════════════════════

def _dca_controls():
    yr_now = pd.Timestamp.today().year
    return html.Div([
        _ctrl_card(
            _lbl("Starting BTC"),
            dbc.Input(id="dca-stack", type="number", value=0,
                      min=0, step=0.001, size="sm"),
            dcc.Checklist(id="dca-use-lots",
                          options=[{"label":" Use Stack Tracker lots","value":"yes"}],
                          value=[], inputStyle={"marginRight":"5px"}),
        ),
        _ctrl_card(
            _lbl("Per-period amount ($)"),
            dbc.Input(id="dca-amount", type="number", value=100,
                      min=1, step=1, size="sm"),
            _lbl("Frequency"),
            dcc.Dropdown(id="dca-freq",
                         options=["Daily","Weekly","Monthly","Quarterly","Annually"],
                         value="Monthly", clearable=False),
            _lbl("Year range"),
            dcc.RangeSlider(id="dca-yr-range", min=2009, max=2060,
                            value=[yr_now, yr_now + 10], step=1,
                            marks={y: f"'{y % 100:02d}" for y in range(2009, 2061, 5)},
                            tooltip={"always_visible":False}),
        ),
        _ctrl_card(
            _lbl("Display"),
            dcc.Dropdown(id="dca-disp",
                         options=[{"label":"BTC Balance","value":"btc"},
                                  {"label":"USD Value","value":"usd"}],
                         value="btc", clearable=False),
            dcc.Checklist(id="dca-toggles",
                          options=[{"label":" Log Y","value":"log_y"},
                                   {"label":" Show today","value":"show_today"},
                                   {"label":" Dual Y-axis","value":"dual_y"},
                                   {"label":" Show legend","value":"show_legend"}],
                          value=["show_legend","dual_y"], labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
        ),
        _ctrl_card(
            _lbl("Quantiles"),
            html.Small(
                "Price path drives sat accumulation — lower quantile = lower price = more sats/period.",
                style={"color":"#888","display":"block","marginBottom":"4px"},
            ),
            dcc.Checklist(id="dca-qs", options=_q_options(),
                          value=[0.5], labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
        ),
        _stackcellerator_controls(),
    ])


def _stackcellerator_controls():
    return _ctrl_card(
        html.B("Stack-cellerator", style={"fontSize":"12px"}),
        dcc.Checklist(id="dca-sc-enable",
                      options=[{"label":" Enter Saylor Mode","value":"yes"}],
                      value=[], inputStyle={"marginRight":"5px"}),
        html.Div(id="dca-sc-body", style={"display":"none"}, children=[
            _lbl("Loan amount ($)"),
            dbc.Input(id="dca-sc-loan", type="number",
                      value=1200, min=0, step=1, size="sm"),
            _lbl("Annual interest rate (0–100% / yr)"),
            dbc.Input(id="dca-sc-rate", type="number",
                      value=13.0, min=0, max=100, step=0.5, size="sm"),
            _lbl("Loan term (months, rolls over)"),
            dbc.Input(id="dca-sc-term", type="number",
                      value=12, min=1, max=360, step=1, size="sm"),
            html.Div(id="dca-sc-info",
                     style={"fontSize":"11px","color":"#555","marginTop":"4px"}),
        ]),
    )


def _dca_tab():
    return dbc.Row([
        dbc.Col(_dca_controls(), width=3, className="controls-col overflow-auto",
                style={"maxHeight":"85vh"}),
        dbc.Col([
            dcc.Graph(id="dca-graph", style={"height":"78vh"},
                      config={"toImageButtonOptions":{"format":"png","scale":2,
                                                       "filename":"btc_dca"}}),
            _export_row("dca"),
        ], width=9),
    ], className="g-0")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — BTC Retireator
# ══════════════════════════════════════════════════════════════════════════════

def _retire_controls():
    yr_now = pd.Timestamp.today().year
    return html.Div([
        _ctrl_card(
            _lbl("Starting BTC"),
            dbc.Input(id="ret-stack", type="number", value=1.0,
                      min=0, step=0.001, size="sm"),
            dcc.Checklist(id="ret-use-lots",
                          options=[{"label":" Use Stack Tracker lots","value":"yes"}],
                          value=[], inputStyle={"marginRight":"5px"}),
        ),
        _ctrl_card(
            _lbl("Withdrawal/period ($)"),
            dbc.Input(id="ret-wd", type="number", value=5000,
                      min=1, step=1, size="sm"),
            _lbl("Frequency"),
            dcc.Dropdown(id="ret-freq",
                         options=["Daily","Weekly","Monthly","Quarterly","Annually"],
                         value="Monthly", clearable=False),
            _lbl("Year range"),
            dcc.RangeSlider(id="ret-yr-range", min=2024, max=2075,
                            value=[2031, 2075], step=1,
                            marks={y: f"'{y % 100:02d}" for y in range(2025, 2076, 5)},
                            tooltip={"always_visible":False}),
            _lbl("Inflation rate (0–100% / yr)"),
            dbc.Input(id="ret-infl", type="number", value=4,
                      min=0, max=100, step=0.5, size="sm"),
        ),
        _ctrl_card(
            _lbl("Display"),
            dcc.Dropdown(id="ret-disp",
                         options=[{"label":"BTC Remaining","value":"btc"},
                                  {"label":"USD Value","value":"usd"}],
                         value="btc", clearable=False),
            dcc.Checklist(id="ret-toggles",
                          options=[{"label":" Log Y","value":"log_y"},
                                   {"label":" Show today","value":"show_today"},
                                   {"label":" Dual Y-axis","value":"dual_y"},
                                   {"label":" Annotate depletion","value":"annotate"},
                                   {"label":" Show legend","value":"show_legend"}],
                          value=["annotate","log_y","dual_y"], labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
        ),
        _ctrl_card(
            _lbl("Quantiles"),
            dcc.Checklist(id="ret-qs", options=_q_options(),
                          value=_DEF_QS, labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
        ),
    ])


def _retire_tab():
    return dbc.Row([
        dbc.Col(_retire_controls(), width=3, className="controls-col overflow-auto",
                style={"maxHeight":"85vh"}),
        dbc.Col([
            dcc.Graph(id="retire-graph", style={"height":"78vh"},
                      config={"toImageButtonOptions":{"format":"png","scale":2,
                                                       "filename":"btc_retire"}}),
            _export_row("retire"),
        ], width=9),
    ], className="g-0")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — HODL Supercharger
# ══════════════════════════════════════════════════════════════════════════════

def _supercharge_controls():
    yr_now = pd.Timestamp.today().year
    dq_opts = _q_options()
    dq_default = _nearest_quantile(0.05, _ALL_QS)
    return html.Div([
        _ctrl_card(
            _lbl("Mode"),
            dcc.RadioItems(id="sc-mode",
                options=[{"label":" A — Fixed spending (depletion date)","value":"a"},
                         {"label":" B — Fixed depletion (max spending)","value":"b"}],
                value="a", labelStyle={"display":"block"},
                inputStyle={"marginRight":"5px"}),
        ),
        _ctrl_card(
            _lbl("Starting BTC"),
            dbc.Input(id="sc-stack", type="number", value=1.0,
                      min=0, step=0.001, size="sm"),
            dcc.Checklist(id="sc-use-lots",
                          options=[{"label":" Use Stack Tracker lots","value":"yes"}],
                          value=[], inputStyle={"marginRight":"5px"}),
        ),
        _ctrl_card(
            _lbl("Base retirement year"),
            dcc.Slider(id="sc-start-yr", min=yr_now, max=2075,
                       value=yr_now, step=1,
                       marks={y: f"'{y % 100:02d}" for y in range(yr_now, 2076, 5)},
                       tooltip={"always_visible":False}),
        ),
        _ctrl_card(
            _lbl("Delay offsets (years)"),
            dbc.Row([
                dbc.Col(dbc.Input(id="sc-d0", type="number", value=0,
                                  min=0, step=1, size="sm"), width=True),
                dbc.Col(dbc.Input(id="sc-d1", type="number", value=1,
                                  min=0, step=1, size="sm"), width=True),
                dbc.Col(dbc.Input(id="sc-d2", type="number", value=2,
                                  min=0, step=1, size="sm"), width=True),
                dbc.Col(dbc.Input(id="sc-d3", type="number", value=4,
                                  min=0, step=1, size="sm"), width=True),
                dbc.Col(dbc.Input(id="sc-d4", type="number", value=8,
                                  min=0, step=1, size="sm"), width=True),
            ], className="g-1"),
        ),
        _ctrl_card(
            _lbl("Frequency"),
            dcc.Dropdown(id="sc-freq",
                         options=["Daily","Weekly","Monthly","Quarterly","Annually"],
                         value="Monthly", clearable=False),
            _lbl("Inflation rate (0–100% / yr)"),
            dbc.Input(id="sc-infl", type="number", value=4,
                      min=0, max=100, step=0.5, size="sm"),
        ),
        dbc.Collapse([
            _ctrl_card(
                _lbl("Withdrawal/period ($)"),
                dbc.Input(id="sc-wd", type="number", value=5000,
                          min=1, step=1, size="sm"),
                _lbl("End year"),
                dcc.Slider(id="sc-end-yr", min=2030, max=2100,
                           value=2075, step=1,
                           marks={y: f"'{y % 100:02d}" for y in range(2030, 2101, 10)},
                           tooltip={"always_visible":False}),
                _lbl("Display"),
                dcc.Dropdown(id="sc-disp",
                             options=[{"label":"BTC Remaining","value":"btc"},
                                      {"label":"USD Value","value":"usd"}],
                             value="usd", clearable=False),
            ),
        ], id="sc-mode-a-collapse", is_open=True),
        dbc.Collapse([
            _ctrl_card(
                _lbl("Target depletion year"),
                dcc.Slider(id="sc-target-yr", min=2030, max=2100,
                           value=2060, step=1,
                           marks={y: f"'{y % 100:02d}" for y in range(2030, 2101, 10)},
                           tooltip={"always_visible":False}),
            ),
        ], id="sc-mode-b-collapse", is_open=False),
        _ctrl_card(
            _lbl("Quantile band"),
            dcc.Checklist(id="sc-chart-layout",
                options=[{"label":" Shade quantile bands","value":"shade"}],
                value=["shade"],
                inputStyle={"marginRight":"5px"}),
        ),
        dbc.Collapse([
            _ctrl_card(
                _lbl("Display quantile"),
                dcc.Dropdown(id="sc-display-q", options=dq_opts,
                             value=dq_default, clearable=False),
            ),
        ], id="sc-display-q-collapse", is_open=True),
        _ctrl_card(
            dcc.Checklist(id="sc-toggles",
                          options=[{"label":" Annotate depletion","value":"annotate"},
                                   {"label":" Log Y","value":"log_y"},
                                   {"label":" Show today","value":"show_today"},
                                   {"label":" Show legend","value":"show_legend"}],
                          value=["annotate","log_y","show_legend"],
                          labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
        ),
        _ctrl_card(
            _lbl("Quantiles"),
            dcc.Checklist(id="sc-qs", options=_q_options(),
                          value=_DEF_QS, labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
        ),
    ])


def _supercharge_tab():
    return dbc.Row([
        dbc.Col(_supercharge_controls(), width=3, className="controls-col overflow-auto",
                style={"maxHeight":"85vh"}),
        dbc.Col([
            dcc.Graph(id="supercharge-graph", style={"height":"78vh"},
                      config={"toImageButtonOptions":{"format":"png","scale":2,
                                                       "filename":"btc_supercharge"}}),
            _export_row("supercharge"),
        ], width=9),
    ], className="g-0")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6 — FAQ
# ══════════════════════════════════════════════════════════════════════════════

_FAQ = [
    {
        "q": "What does the Share button do?",
        "a": (
            "It's cooler than you might expect. Suppose you've customized a graph and want to "
            "show someone else your plot. You could save an image — or you could send them a "
            "link that takes them directly to Quantoshi with all of your customized configuration "
            "already applied. Every control across all tabs is encoded into the URL, so the "
            "recipient sees exactly what you see. You can optionally include your Stack Tracker "
            "lots in the link too. Your link history is saved in your browser so you can revisit "
            "or re-share any configuration you've generated."
        ),
    },
    {
        "q": "What is quantile regression?",
        "a": (
            "Ordinary regression finds the line that best fits the average of your data. "
            "Quantile regression does something more powerful: it fits a separate line for any "
            "percentile you choose. The 50th percentile line (the median) splits the data in half "
            "— as many points above as below. The 5th percentile line fits the bottom 5% of the "
            "data, and the 95th fits the top 5%. On Quantoshi, quantile regression is applied to "
            "the historical log-log relationship between time and Bitcoin price, giving you a "
            "family of curves that describe not just where Bitcoin has typically been, but how "
            "extreme the highs and lows have historically gotten. The percentile of your purchase "
            "price tells you how cheap or expensive that entry was relative to all historical "
            "prices at that point in Bitcoin's life."
        ),
    },
    {
        "q": "Is Quantoshi predicting future Bitcoin price? What guarantees do I have this will be true?",
        "a": (
            "No. Quantoshi is extrapolating by quantile regression of a power law model. This is math. "
            "Prediction is what YOU do with this math. And as far as a guarantee, there is none (any "
            "guarantee would be worth how much you were required to pay to use this software, which is "
            "free!). Now, if I would have made Quantoshi in 2016, I would have been surprisingly "
            "accurate in 2026, but I would caution anyone working with any dataset against extrapolating "
            "much beyond 1/3 of the dataset. Bitcoin is 17 years old. 17/3 is 5 to a physicist, 6 to a "
            "mathematician, 5.67 to an engineer, and 5-2/3 in US Army Mixed Number Format. Use "
            "caution extrapolating beyond 6 or so years."
        ),
    },
    {
        "q": "Why does this look so bad on [my device / browser]?",
        "a": (
            "I have only been able to test this on Linux and iPhone. I'm a physician, not a "
            "programmer, so I really couldn't tell you anyhow! But if anyone can send me a "
            "screenshot I'd love to fix it for their platform!"
        ),
    },
    {
        "q": "Why do some high-percentile extrapolated quantile projection lines cross in the future?",
        "a": (
            "When a higher-percentile line crosses a lower one (or vice versa), it indicates "
            "a trend in the dataset that renders extrapolation unreliable. For example, because "
            "subsequent Bitcoin bubble peaks have been getting less extreme over time, the 99th "
            "percentile line rises less steeply than the 95th — so much so that these two lines "
            "cross around 2034. It should be noted that the lower-percentile extrapolations "
            "(e.g. the 30th percentile) remain more or less parallel well beyond any reasonable "
            "planning horizon."
        ),
    },
    {
        "q": "Why a Power Law?",
        "a": html.Span([
            "Great question. I did my undergrad in astrophysics and math, so the real question "
            "is: how do you value money? We can't use cash flow analysis on money itself, and "
            "ideally we'd want a scale-invariant model that works at small and large times — "
            "excluding exponential models like the first popular model, the Stock-to-Flow by "
            "the venerable Plan B. Power laws are observed everywhere in nature (literally "
            "everywhere space exists), but for a more detailed discussion please see the ",
            html.A("Scientific Bitcoin Institute",
                   href="https://scientificbitcoininstitute.org/",
                   target="_blank", rel="noopener noreferrer"),
            ". Giovanni Santostasi had the first Bitcoin price model — a Power Law model — "
            "before Plan B's S2F... but as physics people speak differently, it took a while "
            "to catch on :)",
        ]),
    },
    {
        "q": "Can I send you a tip?",
        "a": html.Table([
            html.Tbody([
                html.Tr([html.Td("Bitcoin", style={"paddingRight":"12px","whiteSpace":"nowrap","verticalAlign":"top"}),
                         html.Td(html.Code("bc1qgh6kfnf02uvplq490nyslc7768tnvzftlrw5fe", style={"wordBreak":"break-all","fontSize":"11px"}))]),
                html.Tr([html.Td("Lightning", style={"paddingRight":"12px","whiteSpace":"nowrap","verticalAlign":"top"}),
                         html.Td(html.Code("lno1pgjrzv34xscxyvrp94jrvdej956rgdnp95ukydt9943rxdpkxucrqvpsv5ury93pqgfffll4jmjf0tffqtx47xt886gzp9fajp3966xz96gm2xj9cqedx", style={"wordBreak":"break-all","fontSize":"11px"}))]),
                html.Tr([html.Td("Ecash", style={"paddingRight":"12px","whiteSpace":"nowrap","verticalAlign":"top"}),
                         html.Td(html.Code("creqApGF0gaNhdGRwb3N0YWF4QGh0dHBzOi8vY29pbm9zLmlvL2FwaS9lY2FzaC8xMjU0MGIwYS1kNjcyLTQ0NmEtOWI1ZS1iMzQ2NzAwMDBlODJhZ/dhaXgkMTI1NDBiMGEtZDY3M:", style={"wordBreak":"break-all","fontSize":"11px"}))]),
                html.Tr([html.Td("Liquid BTC", style={"paddingRight":"12px","whiteSpace":"nowrap","verticalAlign":"top"}),
                         html.Td(html.Code("lq1qqfztsa6ffjkspk3qxp4ft8kn2sxu5ja9prn5d9vwuqjjut5g2tzc8rpsgz2pysayplrgemf9dt3vpkqhvsvtkfxvdyk9mlsel", style={"wordBreak":"break-all","fontSize":"11px"}))]),
                html.Tr([html.Td("Liquid USDt", style={"paddingRight":"12px","whiteSpace":"nowrap","verticalAlign":"top"}),
                         html.Td(html.Code("liquidnetwork:lq1qqfjgl0fvv7a5prqd7d0k4x80kq2v0cngzxuj7hz3pdhuj0xg57tuzk9q0knrsuevsrywqys92ttefak83xzqq6uqmngkkaa74?assetid=ce091c998b83c78bb71a632313ba3760f176", style={"wordBreak":"break-all","fontSize":"11px"}))]),
            ])
        ], style={"width":"100%","borderCollapse":"collapse","marginTop":"4px"}),
    },
    {
        "q": "I see you modeled up to 3 future bubbles in the first tab... What is your model / how did you model it?",
        "a": (
            "Interesting question. I modeled Bitcoin price as a power law running through the "
            "bottom roughly 30% of the data, and then modeled each bubble separately in "
            "log-log space as something like a trapezoid. I then looked at the shape of each "
            "of the trapezoids and noticed how they changed from a tall triangle, to a "
            "medium-height trapezoid, and then to a very long, short, almost table-like "
            "trapezoid (kinda like a Japanese low table, a chabudai)... Anyhow, in "
            "mathematical terms, I parameterized each bubble and took the trend through time "
            "on each part of each shape and extrapolated that trend (along with the timing "
            "trend) to up to three future bubbles. The result is underwhelming — the bubbles "
            "converge somewhat rapidly on the support trendline... which is part of what "
            "everyone means when they say Bitcoin is getting less volatile over time. I only use "
            "the last three bubbles to extrapolate over; adding the very first bubble massively "
            "screws up the trend, and we were just kids back then, so it shouldn't really "
            "count :)"
        ),
    },
    {
        "q": "Why did you make this?",
        "a": (
            "Everyone needs bitcoin... and Bitcoin is for everyone. The more clearly people "
            "can see the past, the more accurately they can model the future. I'm just doing "
            "a tiny part in helping people see the bright orange future we are racing towards."
        ),
    },
    {
        "q": "Do you have a podcast?",
        "a": html.Span([
            "No. I'm ugly and I work in the dark all day (only half of this is true)... "
            "if you are looking for a podcast recommendation, see ",
            html.A("porkopolis.io",
                   href="https://www.porkopolis.io/youtube/",
                   target="_blank", rel="noopener noreferrer"),
            ". Mezenskis has way nicer charts... in fact, I'm waiting to subscribe to "
            "his charts myself!",
        ]),
    },
    {
        "q": "I see you have a Bitcoin price ticker in the header... does this reveal my IP address to a third party?",
        "a": "No. It reveals the IP address of Quantoshi to Binance.",
    },
    {
        "q": "Can I link directly to a tab?",
        "a": html.Span([
            "YES! Just add a /1 to the URL to get to the first tab, a /2 to get to the "
            "second, and so on. For example, ",
            html.A("quantoshi.xyz/4",
                   href="https://quantoshi.xyz/4",
                   target="_blank", rel="noopener noreferrer"),
            " will take you directly to the retirement extrapolator.",
        ]),
    },
    {
        "q": "Can I run my own Quantoshi? Is it Open Source?",
        "a": html.Span([
            "Yes and yes. Quantoshi is free as in beer and free as in speech — BSD-2 licensed "
            "open source code available at ",
            html.A("github.com/bg002h/quantoshi",
                   href="https://github.com/bg002h/quantoshi",
                   target="_blank", rel="noopener noreferrer"),
            ". You are welcome to do anything with the code... or nothing. "
            "There's also a native Linux app compiled as an x86 AppImage there too, "
            "but it's a few iterations behind.",
        ]),
    },
    {
        "q": "If I enter my purchases in Quantoshi as Stack Tracker lots, where does that data go?",
        "a": html.Span([
            "The data is stored long-term in your browser's \"localStorage\" — it is never stored "
            "long term on the server. That said, all charts are rendered server-side. "
            "Logs are intentionally deleted every 27 days to prevent aggregation of data "
            "by authorities, some of whom are allowed to demand data thirty days or older "
            "without a warrant. So for optimal privacy, use the onion version of this website: ",
            html.A("Stay dark, Anon",
                   href="http://u5dprelc4ti7xoczb5sbtye6qidlji2l6psmkx35anvxgjyqrkmu32ad.onion",
                   target="_blank", rel="noopener noreferrer"),
            ".",
        ]),
    },
    {
        "q": "Is there any way to contact someone about this app?",
        "a": html.Span([
            "Email: ",
            html.A("bcg@pm.me", href="mailto:bcg@pm.me"),
            " or Nostr: ",
            html.A("npub1fa8c9pr…qanthnd",
                   href="https://nostr.com/npub1fa8c9prxnrlkdtjl48adfsxyaduz8tas075l2n4f6903y9awjmxqanthnd",
                   target="_blank", rel="noopener noreferrer"),
        ]),
    },
]


def _faq_tab():
    items = []
    for i, entry in enumerate(_FAQ):
        items.append(
            dbc.AccordionItem(
                html.P(entry["a"], className="mb-0"),
                title=entry["q"],
                item_id=f"faq-{i}",
            )
        )
    return html.Div([
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H5("Frequently Asked Questions", className="mb-3 mt-2"),
                    dbc.Accordion(items, id="faq-accordion", start_collapsed=True, flush=True),
                ]),
                width={"size": 8, "offset": 2},
            )
        ),
    ], className="p-3")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Stack Tracker
# ══════════════════════════════════════════════════════════════════════════════


def _stack_tracker_tab():
    return html.Div([
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
                    style_cell={"backgroundColor":"#fff","color":"#222",
                                "border":"1px solid #dee2e6","padding":"4px 8px",
                                "fontSize":"13px"},
                    style_header={"backgroundColor":"#f8f9fa","color":"#222",
                                  "fontWeight":"bold"},
                    style_data_conditional=[
                        {"if":{"state":"selected"},"backgroundColor":"#cce5ff",
                         "border":"1px solid #99caff"},
                    ],
                    page_size=20,
                ),
                html.Div(id="lots-summary", className="mt-2 text-muted small"),
            ], width=8),

            # ── controls ─────────────────────────────────────────────────────
            dbc.Col([
                _ctrl_card(
                    html.H6("Add Lot", className="mb-2"),
                    _lbl("Date"), dbc.Input(id="lot-date", type="date",
                        value=str(pd.Timestamp.today().date()), size="sm"),
                    _lbl("BTC amount"),
                    dbc.Input(id="lot-btc", type="number", value=0.01,
                              min=0, step=0.0001, size="sm"),
                    _lbl("Price ($/BTC)"),
                    dbc.Input(id="lot-price", type="number", value=69420,
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
                _ctrl_card(
                    html.H6("Export / Import", className="mb-2"),
                    dbc.Button("⬇ Export JSON", id="lots-export-btn",
                               color="secondary", size="sm", className="w-100 mb-2"),
                    html.Hr(className="my-1"),
                    dcc.Upload(
                        id="lots-import-upload",
                        children=dbc.Button("⬆ Import JSON", color="secondary",
                                            size="sm", className="w-100"),
                        accept=".json",
                        multiple=False,
                    ),
                    html.Div(id="lots-import-status", className="mt-1 small"),
                ),
            ], width=4),
        ]),
    ], className="p-2")


# ══════════════════════════════════════════════════════════════════════════════
# App layout
# ══════════════════════════════════════════════════════════════════════════════

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/png" href="/assets/quantoshi_logo.png">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

app.layout = dbc.Container([
    dcc.Interval(id="price-interval", interval=20*60*1000, n_intervals=0),
    dcc.Store(id="btc-price-store", storage_type="memory", data=None),
    dcc.Store(id="lots-store", storage_type="local", data=[]),
    dcc.Store(id="lots-export-dummy"),
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="snapshot-lots",     storage_type="memory", data=None),
    dcc.Store(id="effective-lots",    storage_type="memory", data=[]),
    dcc.Store(id="link-history",      storage_type="local",  data=[]),
    dcc.Store(id="loaded-hash-store", storage_type="memory"),
    dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col(html.Img(src="/assets/quantoshi_logo.png", height="40px"),
                        width="auto"),
                dbc.Col(dbc.NavbarBrand("Quantoshi", className="ms-2 fw-bold fs-4",
                                        style={"fontFamily":"Palatino Linotype, Palatino, Book Antiqua, serif"}),
                        width="auto"),
                dbc.Col(
                    html.Div(id="price-ticker",
                             style={"fontSize":"12px", "color":"rgba(255,255,255,0.85)",
                                    "whiteSpace":"nowrap", "fontFamily":"monospace"}),
                    className="ms-3", width="auto",
                ),
                dbc.Col(
                    html.Div([
                        html.Span("Stay dark, Anon →",
                                  style={"fontSize":"9px", "color":"rgba(255,255,255,0.4)",
                                         "whiteSpace":"nowrap", "marginRight":"6px"}),
                        html.A(
                            "🧅 Tor onion",
                            href="http://u5dprelc4ti7xoczb5sbtye6qidlji2l6psmkx35anvxgjyqrkmu32ad.onion",
                            target="_blank",
                            rel="noopener noreferrer",
                            className="text-white-50 small text-decoration-none",
                        ),
                    ], style={"display":"flex", "alignItems":"center"}),
                    className="ms-auto", width="auto",
                ),
                dbc.Col(
                    html.Div([
                        dbc.Button("📸 Share", id="share-btn", color="light", size="sm",
                                   outline=True, className="ms-2"),
                        html.Div("▲ Cooler than you think",
                                 style={"fontSize":"9px", "color":"rgba(255,255,255,0.4)",
                                        "whiteSpace":"nowrap", "textAlign":"center",
                                        "marginTop":"1px"}),
                    ]),
                    width="auto",
                ),
            ], align="center", className="g-0 w-100"),
        ], fluid=True),
        color="#2c3e50", dark=True, className="mb-0 py-1",
    ),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Share / Restore Configuration")),
        dbc.ModalBody([
            html.Div("Scope:", className="fw-semibold small mb-1"),
            dcc.RadioItems(
                id="share-scope",
                options=[
                    {"label": " All tabs — full state, longer link", "value": "all"},
                    {"label": " Current tab only — shorter link",    "value": "tab"},
                ],
                value="all",
                inputStyle={"marginRight": "5px"},
                className="mb-2 small",
            ),
            dcc.Checklist(
                id="share-include-lots",
                options=[{"label": " Include Stack Tracker lots in link", "value": "yes"}],
                value=[], inputStyle={"marginRight": "5px"},
                className="small",
            ),
            dbc.Button("Generate link", id="share-copy-btn",
                       color="primary", size="sm", className="mt-2 mb-3 w-100"),
            dbc.InputGroup([
                dbc.Input(id="share-url-display", type="text", readonly=True,
                          placeholder="Click 'Generate link' above…", size="sm"),
                dcc.Clipboard(target_id="share-url-display",
                              style={"cursor":"pointer","fontSize":"18px",
                                     "padding":"4px 8px"}),
            ], size="sm"),
            html.Hr(className="my-3"),
            html.Div([
                html.Span("Link History", className="fw-semibold small"),
                html.Span(" (your browser only — no duplicates)",
                          className="text-muted small ms-1"),
            ], className="mb-2"),
            html.Div(id="link-history-display"),
            dbc.Button("🗑 Clear history", id="clear-history-btn",
                       color="link", size="sm", className="text-danger mt-2 p-0"),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="share-modal-close", className="ms-auto", size="sm")
        ),
    ], id="share-modal", is_open=False, size="lg", scrollable=True),
    dbc.Tabs([
        dbc.Tab(_bubble_tab(),       label="Bubble + QR Overlay", tab_id="bubble"),
        dbc.Tab(_heatmap_tab(),      label="CAGR Heatmap",        tab_id="heatmap"),
        dbc.Tab(_dca_tab(),          label="BTC Accumulator",     tab_id="dca"),
        dbc.Tab(_retire_tab(),       label="BTC RetireMentator",  tab_id="retire"),
        dbc.Tab(_supercharge_tab(),  label="HODL Supercharger",   tab_id="supercharge"),
        dbc.Tab(_stack_tracker_tab(),label="Stack Tracker",       tab_id="stack"),
        dbc.Tab(_faq_tab(),          label="FAQ",                 tab_id="faq"),
    ], id="main-tabs", active_tab="bubble"),
], fluid=True, className="px-2 py-1")


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks — chart updates
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("bubble-graph", "figure"),
    Input("bub-qs",            "value"),
    Input("bub-toggles",       "value"),
    Input("bub-bubble-toggles","value"),
    Input("bub-xscale",        "value"),
    Input("bub-yscale",        "value"),
    Input("bub-xrange",        "value"),
    Input("bub-yrange",        "value"),
    Input("bub-n-future",      "value"),
    Input("bub-ptsize",        "value"),
    Input("bub-ptalpha",       "value"),
    Input("bub-stack",         "value"),
    Input("bub-show-stack",    "value"),
    Input("bub-use-lots",      "value"),
    Input("effective-lots",    "data"),
)
def update_bubble(sel_qs, toggles, bubble_toggles,
                  xscale, yscale, xrange, yrange,
                  n_future, ptsize, ptalpha, stack, show_stack, use_lots, lots_data):
    toggles        = toggles or []
    bubble_toggles = bubble_toggles or []
    yrange         = yrange or [0, 7]
    xrange         = xrange or [2012, 2030]

    return build_bubble_figure(M, dict(
        selected_qs = sel_qs or [],
        shade       = "shade"     in toggles,
        show_ols    = "show_ols"  in toggles,
        show_data   = "show_data"   in toggles,
        show_today  = "show_today"  in toggles,
        show_legend = "show_legend" in toggles,
        show_comp   = "show_comp" in bubble_toggles,
        show_sup    = "show_sup"  in bubble_toggles,
        xscale      = xscale or "linear",
        yscale      = yscale or "log",
        xmin        = int(xrange[0]), xmax = int(xrange[1]),
        ymin        = 10 ** yrange[0], ymax = 10 ** yrange[1],
        n_future    = int(n_future or 0),
        pt_size     = int(ptsize or 3),
        pt_alpha    = float(ptalpha or 0.6),
        stack       = float(stack or 0),
        show_stack  = bool(show_stack),
        use_lots    = bool(use_lots),
        lots        = lots_data or [],
        comp_color  = "#FFD700", comp_lw = 2.0,
        sup_color   = "#888888", sup_lw  = 1.5,
    ))


@callback(
    Output("bub-yrange", "value", allow_duplicate=True),
    Input("bub-xrange",  "value"),
    Input("bub-auto-y",  "value"),
    Input("bub-yscale",  "value"),
    State("bub-qs",      "value"),
    prevent_initial_call=True,
)
def auto_bubble_yrange(xrange, auto_y, yscale, sel_qs):
    if not auto_y or not xrange:
        raise dash.exceptions.PreventUpdate
    xmin, xmax = int(xrange[0]), int(xrange[1])
    qs = sorted([float(q) for q in (sel_qs or []) if float(q) in M.qr_fits])
    if not qs:
        qs = sorted(M.qr_fits.keys())
    t_lo = max(yr_to_t(xmin, M.genesis), 0.1)
    t_hi = yr_to_t(xmax, M.genesis)
    p_lo = float(qr_price(qs[0],  t_lo, M.qr_fits))
    p_hi = float(qr_price(qs[-1], t_hi, M.qr_fits))
    if (yscale or "log") == "log":
        y_lo = math.floor(math.log10(max(p_lo, 1e-10)) * 2) / 2 - 0.5
        y_hi = math.ceil( math.log10(max(p_hi, 1e-10)) * 2) / 2 + 0.5
        y_lo = max(-2.0, min(y_lo, 6.0))
        y_hi = min(8.0,  max(y_hi, 1.0))
    else:  # linear — floor near zero, ceiling at highest quantile + 10% headroom
        y_lo = -2.0
        y_hi = math.ceil(math.log10(max(p_hi * 1.1, 1e-10)) * 2) / 2
        y_hi = min(8.0, max(y_hi, 1.0))
    return [round(y_lo, 1), round(y_hi, 1)]


@callback(
    Output("heatmap-graph", "figure"),
    Input("hm-entry-yr",  "value"),
    Input("hm-entry-q",   "value"),
    Input("hm-exit-range","value"),
    Input("hm-exit-qs",   "value"),
    Input("hm-mode",      "value"),
    Input("hm-b1",        "value"),
    Input("hm-b2",        "value"),
    Input("hm-c-lo",      "value"),
    Input("hm-c-mid1",    "value"),
    Input("hm-c-mid2",    "value"),
    Input("hm-c-hi",      "value"),
    Input("hm-grad",      "value"),
    Input("hm-vfmt",      "value"),
    Input("hm-cell-fs",   "value"),
    Input("hm-toggles",   "value"),
    Input("hm-stack",     "value"),
    Input("hm-use-lots",  "value"),
    Input("effective-lots","data"),
    State("btc-price-store", "data"),
)
def update_heatmap(entry_yr, entry_q, exit_range, exit_qs, mode,
                   b1, b2, c_lo, c_mid1, c_mid2, c_hi, grad,
                   vfmt, cell_fs, toggles, stack, use_lots, lots_data, live_price):
    exit_range = exit_range or [entry_yr or 2025, (entry_yr or 2025) + 10]
    toggles    = toggles or []
    yr_now = pd.Timestamp.today().year
    return build_heatmap_figure(M, dict(
        entry_yr     = int(entry_yr or yr_now),
        entry_q      = float(entry_q or 50),
        live_price   = float(live_price) if live_price and int(entry_yr or yr_now) == yr_now else None,
        exit_yr_lo   = int(exit_range[0]),
        exit_yr_hi   = int(exit_range[1]),
        exit_qs      = exit_qs or [],
        color_mode   = int(mode or 0),
        b1           = float(b1 or M.CAGR_SEG_B1),
        b2           = float(b2 or M.CAGR_SEG_B2),
        c_lo         = c_lo   or M.CAGR_SEG_C_LO,
        c_mid1       = c_mid1 or M.CAGR_SEG_C_MID1,
        c_mid2       = c_mid2 or M.CAGR_SEG_C_MID2,
        c_hi         = c_hi   or M.CAGR_SEG_C_HI,
        n_disc       = int(grad or M.CAGR_GRAD_STEPS),
        vfmt         = vfmt or "cagr",
        cell_font_size = int(cell_fs or 9),
        show_colorbar = "colorbar" in toggles,
        stack        = float(stack or 0),
        use_lots     = bool(use_lots),
        lots         = lots_data or [],
    ))


@callback(
    Output("dca-graph", "figure"),
    Input("dca-stack",    "value"),
    Input("dca-use-lots", "value"),
    Input("dca-amount",   "value"),
    Input("dca-freq",     "value"),
    Input("dca-yr-range", "value"),
    Input("dca-disp",     "value"),
    Input("dca-toggles",  "value"),
    Input("dca-qs",       "value"),
    Input("effective-lots","data"),
    Input("dca-sc-enable","value"),
    Input("dca-sc-loan",  "value"),
    Input("dca-sc-rate",  "value"),
    Input("dca-sc-term",  "value"),
    State("btc-price-store","data"),
)
def update_dca(stack, use_lots, amount, freq, yr_range, disp, toggles, sel_qs, lots_data,
               sc_enable, sc_loan, sc_rate, sc_term, price_data):
    toggles    = toggles or []
    yr_range   = yr_range or [2024, 2034]
    live_price = float(price_data or 0)
    return build_dca_figure(M, dict(
        start_stack    = float(stack or 0),
        use_lots       = bool(use_lots),
        amount         = float(amount or 100),
        freq           = freq or "Monthly",
        start_yr       = int(yr_range[0]),
        end_yr         = int(yr_range[1]),
        disp_mode      = disp or "btc",
        log_y          = "log_y"      in toggles,
        show_today     = "show_today" in toggles,
        dual_y         = "dual_y"     in toggles,
        show_legend    = "show_legend" in toggles,
        selected_qs    = sel_qs or [],
        lots           = lots_data or [],
        sc_enabled     = bool(sc_enable),
        sc_loan_amount = float(sc_loan or 0),
        sc_rate        = float(sc_rate) if sc_rate is not None else 13.0,
        sc_live_price  = live_price,
    ))


@callback(Output("dca-sc-body","style"), Input("dca-sc-enable","value"))
def _toggle_dca_sc_body(val):
    return {} if val else {"display": "none"}


@callback(
    Output("dca-sc-info","children"),
    Input("dca-amount",   "value"),
    Input("dca-freq",     "value"),
    Input("dca-sc-enable","value"),
    Input("dca-sc-loan",  "value"),
    Input("dca-sc-rate",  "value"),
    Input("dca-sc-term",  "value"),
    State("btc-price-store","data"),
)
def update_sc_info(amount, freq, enabled, sc_loan, rate, term, price_data):
    if not enabled:
        return ""
    FREQ_PPY = {"Daily":365,"Weekly":52,"Monthly":12,"Quarterly":4,"Annually":1}
    ppy       = FREQ_PPY.get(freq or "Monthly", 12)
    amount    = float(amount or 100)
    principal = float(sc_loan or 0)
    rate      = float(rate) if rate is not None else 13.0
    payment   = principal * (rate / 100.0) / ppy
    reduced   = amount - payment
    live      = float(price_data or 0)
    lump_btc  = principal / live if live > 0 else None
    lines = [
        f"Loan: {fmt_price(principal)}",
        f"Payment: {fmt_price(payment)}/period  \u2192  DCA: {fmt_price(max(reduced, 0))}/period",
    ]
    if lump_btc:
        lines.append(f"Buys \u2248 {lump_btc:.5f} BTC @ {fmt_price(live)}")
    if reduced <= 0:
        lines.append("\u26a0 Payment \u2265 DCA amount \u2014 reduce loan or rate")
    return [html.Div(l) for l in lines]


@callback(
    Output("retire-graph", "figure"),
    Input("ret-stack",    "value"),
    Input("ret-use-lots", "value"),
    Input("ret-wd",       "value"),
    Input("ret-freq",     "value"),
    Input("ret-yr-range", "value"),
    Input("ret-infl",     "value"),
    Input("ret-disp",     "value"),
    Input("ret-toggles",  "value"),
    Input("ret-qs",       "value"),
    Input("effective-lots","data"),
)
def update_retire(stack, use_lots, wd, freq, yr_range, infl, disp, toggles, sel_qs, lots_data):
    toggles  = toggles or []
    yr_range = yr_range or [2025, 2045]
    return build_retire_figure(M, dict(
        start_stack  = float(stack or 1.0),
        use_lots     = bool(use_lots),
        wd_amount    = float(wd or 5000),
        freq         = freq or "Monthly",
        start_yr     = int(yr_range[0]),
        end_yr       = int(yr_range[1]),
        inflation    = float(infl or 0),
        disp_mode    = disp or "btc",
        log_y        = "log_y"     in toggles,
        show_today   = "show_today" in toggles,
        dual_y       = "dual_y"    in toggles,
        annotate     = "annotate"  in toggles,
        show_legend  = "show_legend" in toggles,
        selected_qs  = sel_qs or [],
        lots         = lots_data or [],
    ))


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks — HODL Supercharger
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("supercharge-graph", "figure"),
    Input("sc-stack",        "value"),
    Input("sc-use-lots",     "value"),
    Input("sc-start-yr",     "value"),
    Input("sc-d0",           "value"),
    Input("sc-d1",           "value"),
    Input("sc-d2",           "value"),
    Input("sc-d3",           "value"),
    Input("sc-d4",           "value"),
    Input("sc-freq",         "value"),
    Input("sc-infl",         "value"),
    Input("sc-qs",           "value"),
    Input("sc-mode",         "value"),
    Input("sc-wd",           "value"),
    Input("sc-end-yr",       "value"),
    Input("sc-target-yr",    "value"),
    Input("sc-disp",         "value"),
    Input("sc-toggles",      "value"),
    Input("sc-chart-layout", "value"),
    Input("sc-display-q",    "value"),
    Input("effective-lots",  "data"),
)
def update_supercharge(stack, use_lots, start_yr,
                       d0, d1, d2, d3, d4,
                       freq, infl, sel_qs, mode,
                       wd, end_yr, target_yr, disp,
                       toggles, chart_layout, display_q, lots_data):
    delays  = [float(x) for x in [d0, d1, d2, d3, d4] if x is not None]
    toggles = toggles or []
    yr_now  = pd.Timestamp.today().year
    # chart_layout is now a checklist list; legacy snapshots may send an int
    _cl = (2 if "shade" in (chart_layout or []) else 0) \
          if isinstance(chart_layout, list) \
          else (int(chart_layout) if chart_layout is not None else 2)
    return build_supercharge_figure(M, dict(
        mode         = mode or "a",
        start_stack  = float(stack or 1.0),
        start_yr     = int(start_yr or yr_now),
        delays       = delays if delays else [0, 1, 2, 4, 8],
        freq         = freq or "Monthly",
        inflation    = float(infl or 4.0),
        selected_qs  = sel_qs or [],
        chart_layout = _cl,
        display_q    = float(display_q) if display_q is not None
                       else _nearest_quantile(0.5, _ALL_QS),
        wd_amount    = float(wd or 5000),
        end_yr       = int(end_yr or 2075),
        disp_mode    = disp or "usd",
        log_y        = "log_y"      in toggles,
        annotate     = "annotate"   in toggles,
        show_today   = "show_today" in toggles,
        show_legend  = "show_legend" in toggles,
        target_yr    = int(target_yr or 2060),
        lots         = lots_data or [],
        use_lots     = bool(use_lots),
    ))


@callback(
    Output("sc-mode-a-collapse", "is_open"),
    Output("sc-mode-b-collapse", "is_open"),
    Input("sc-mode", "value"),
)
def toggle_sc_mode(mode):
    return mode == "a", mode == "b"


@callback(
    Output("sc-display-q-collapse", "is_open"),
    Input("sc-chart-layout", "value"),
)
def toggle_sc_display_q(layout):
    return "shade" not in (layout or [])


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks — Stack Tracker
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("lot-pct-preview", "children"),
    Input("lot-date",  "value"),
    Input("lot-price", "value"),
)
def preview_percentile(date_str, price):
    if not date_str or not price or float(price) <= 0:
        return ""
    try:
        t   = (pd.Timestamp(date_str) - M.genesis).days / 365.25
        pct = _find_lot_percentile(t, float(price), M.qr_fits)
        return f"Q{pct*100:.2f}%"
    except Exception:
        return ""


@callback(
    Output("lots-store",        "data"),
    Output("lots-table",        "data"),
    Output("lots-table",        "selected_rows"),
    Output("lots-summary",      "children"),
    Output("lots-import-status","children"),
    Input("lot-add-btn",        "n_clicks"),
    Input("lot-del-btn",        "n_clicks"),
    Input("lot-clear-btn",      "n_clicks"),
    Input("lots-import-upload", "contents"),
    State("lot-date",           "value"),
    State("lot-btc",            "value"),
    State("lot-price",          "value"),
    State("lot-notes",          "value"),
    State("lots-table",         "selected_rows"),
    State("lots-store",         "data"),
    prevent_initial_call=True,
)
def manage_lots(add_n, del_n, clear_n, import_contents,
                date_str, btc_amt, price_val, notes,
                selected_rows, lots_data):
    triggered = ctx.triggered_id
    lots = list(lots_data or [])
    import_status = dash.no_update

    if triggered == "lot-add-btn":
        if not date_str or not btc_amt or not price_val:
            raise dash.exceptions.PreventUpdate
        try:
            btc   = float(btc_amt)
            price = float(price_val)
            if btc <= 0 or price <= 0:
                raise ValueError
            t     = (pd.Timestamp(date_str) - M.genesis).days / 365.25
            pct_q = _find_lot_percentile(t, price, M.qr_fits)
            lots.append({
                "date":  date_str,
                "btc":   round(btc, 8),
                "price": round(price, 2),
                "pct_q": round(pct_q, 6),
                "notes": (notes or "").strip(),
            })
            lots.sort(key=lambda l: l["date"])
        except Exception:
            raise dash.exceptions.PreventUpdate

    elif triggered == "lot-del-btn":
        if selected_rows:
            lots = [l for i, l in enumerate(lots) if i not in selected_rows]

    elif triggered == "lot-clear-btn":
        lots = []

    elif triggered == "lots-import-upload" and import_contents:
        try:
            _hdr, b64 = import_contents.split(",", 1)
            raw  = base64.b64decode(b64).decode("utf-8")
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("expected a JSON array")
            # recompute pct_q in case file came from a different model version
            parsed = []
            for row in data:
                t     = (pd.Timestamp(row["date"]) - M.genesis).days / 365.25
                pct_q = _find_lot_percentile(t, float(row["price"]), M.qr_fits)
                parsed.append({
                    "date":  row["date"],
                    "btc":   round(float(row["btc"]), 8),
                    "price": round(float(row["price"]), 2),
                    "pct_q": round(pct_q, 6),
                    "notes": str(row.get("notes", "")).strip(),
                })
            parsed.sort(key=lambda l: l["date"])
            lots = parsed
            import_status = f"Imported {len(lots)} lot(s) ✓"
        except Exception as e:
            import_status = f"Import failed: {e}"
            raise dash.exceptions.PreventUpdate

    table_data = [
        {**l,
         "total_paid": fmt_price(l["btc"] * l["price"]),
         "pct_q":      f"Q{l['pct_q']*100:.2f}%"}
        for l in lots
    ]
    return lots, table_data, [], _lots_summary(lots), import_status


@callback(
    Output("lots-table",   "data",    allow_duplicate=True),
    Output("lots-summary", "children", allow_duplicate=True),
    Input("lots-store",    "data"),
    prevent_initial_call=True,
)
def sync_table_on_load(lots_data):
    lots = lots_data or []
    table_data = [
        {**l,
         "total_paid": fmt_price(l["btc"] * l["price"]),
         "pct_q":      f"Q{l['pct_q']*100:.2f}%"}
        for l in lots
    ]
    return table_data, _lots_summary(lots)


def _lots_summary(lots):
    if not lots:
        return "No lots."
    total_btc  = sum(l["btc"] for l in lots)
    total_paid = sum(l["btc"] * l["price"] for l in lots)
    avg_price  = total_paid / total_btc if total_btc else 0
    avg_pct    = sum(l["pct_q"] * l["btc"] for l in lots) / total_btc if total_btc else 0
    return (f"{len(lots)} lot(s)  |  {total_btc:.8g} BTC  |  "
            f"Avg {fmt_price(avg_price)}/BTC  |  "
            f"Total paid {fmt_price(total_paid)}  |  "
            f"Avg Q{avg_pct*100:.2f}%")


# ── Stack Tracker: clientside JSON export (data never leaves the browser) ─────
app.clientside_callback(
    """
    function(n_clicks, lots_data) {
        if (!n_clicks) return window.dash_clientside.no_update;
        var data = lots_data || [];
        var json = JSON.stringify(data, null, 2);
        var blob = new Blob([json], {type: 'application/json'});
        var url  = URL.createObjectURL(blob);
        var a    = document.createElement('a');
        a.href     = url;
        a.download = 'btc_lots.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return window.dash_clientside.no_update;
    }
    """,
    Output("lots-export-dummy", "data"),
    Input("lots-export-btn",    "n_clicks"),
    State("lots-store",         "data"),
    prevent_initial_call=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Callback — live BTC price ticker (Binance, refreshes every 5 min)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_btc_price():
    """Fetch current BTC price — Binance primary, CoinGecko fallback."""
    sources = [
        ("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
         lambda d: float(d["price"])),
        ("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
         lambda d: float(d["bitcoin"]["usd"])),
    ]
    for url, parse in sources:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return parse(json.loads(r.read()))
        except Exception:
            continue
    return None


@callback(
    Output("price-ticker",    "children"),
    Output("btc-price-store", "data"),
    Output("hm-entry-q",      "value", allow_duplicate=True),
    Input("price-interval", "n_intervals"),
    prevent_initial_call="initial_duplicate",
)
def update_price_ticker(_):
    price = _fetch_btc_price()
    if price is None:
        return "₿ —", no_update, no_update
    pct = _find_lot_percentile(today_t(M.genesis), price, M.qr_fits)
    pct_str = f"Q{pct*100:.1f}%" if pct is not None else "—"
    pct_val = round(pct * 100, 1) if pct is not None else no_update
    return f"₿ {fmt_price(price)}  ·  {pct_str}", price, pct_val


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks — image export (client-side, no kaleido/Chrome needed)
# Uses Plotly.downloadImage() which renders in the browser.
# ══════════════════════════════════════════════════════════════════════════════

_EXPORT_TABS = [
    ("bubble",  "bubble-graph"),
    ("heatmap", "heatmap-graph"),
    ("dca",     "dca-graph"),
    ("retire",  "retire-graph"),
    ("supercharge", "supercharge-graph"),
]

for _tab_id, _graph_id in _EXPORT_TABS:
    app.clientside_callback(
        f"""
        function(n_clicks, fmt, fname, figure) {{
            if (!n_clicks) return window.dash_clientside.no_update;
            if (!figure)   return window.dash_clientside.no_update;
            Plotly.downloadImage(figure, {{
                format:   fmt   || 'png',
                width:    1920,
                height:   1080,
                scale:    2,
                filename: fname || '{_tab_id}'
            }});
            return window.dash_clientside.no_update;
        }}
        """,
        Output(f"{_tab_id}-dl-dummy", "data"),
        Input(f"{_tab_id}-export-btn", "n_clicks"),
        State(f"{_tab_id}-fmt",        "value"),
        State(f"{_tab_id}-fname",      "value"),
        State(f"{_tab_id}-graph",      "figure"),
        prevent_initial_call=True,
    )



# ══════════════════════════════════════════════════════════════════════════════
# Callback — pathname-based tab routing (/1 … /6)
# ══════════════════════════════════════════════════════════════════════════════

_PATH_TO_TAB = {
    "/1": "bubble", "/2": "heatmap", "/3": "dca",
    "/4": "retire",  "/5": "supercharge", "/6": "stack", "/7": "faq",
}
_TAB_TO_PATH = {v: k for k, v in _PATH_TO_TAB.items()}

# Component IDs that belong to each tab (for single-tab share links)
_TAB_CONTROLS = {
    "bubble":      {"bub-qs","bub-xscale","bub-yscale","bub-xrange","bub-yrange",
                    "bub-toggles","bub-bubble-toggles","bub-n-future","bub-ptsize",
                    "bub-ptalpha","bub-stack","bub-show-stack","bub-use-lots","bub-auto-y"},
    "heatmap":     {"hm-entry-yr","hm-entry-q","hm-exit-range","hm-exit-qs","hm-mode",
                    "hm-b1","hm-b2","hm-c-lo","hm-c-mid1","hm-c-mid2","hm-c-hi",
                    "hm-grad","hm-vfmt","hm-cell-fs","hm-toggles","hm-stack","hm-use-lots"},
    "dca":         {"dca-stack","dca-use-lots","dca-amount","dca-freq","dca-yr-range",
                    "dca-disp","dca-toggles","dca-qs",
                    "dca-sc-enable","dca-sc-loan","dca-sc-rate","dca-sc-term"},
    "retire":      {"ret-stack","ret-use-lots","ret-wd","ret-freq","ret-yr-range",
                    "ret-infl","ret-disp","ret-toggles","ret-qs"},
    "supercharge": {"sc-stack","sc-use-lots","sc-start-yr","sc-d0","sc-d1","sc-d2",
                    "sc-d3","sc-d4","sc-freq","sc-infl","sc-qs","sc-mode","sc-wd",
                    "sc-end-yr","sc-target-yr","sc-disp","sc-toggles","sc-chart-layout",
                    "sc-display-q"},
    "stack":       set(),
    "faq":         set(),
}

app.clientside_callback(
    """
    function(pathname) {
        var map = {"/1":"bubble","/2":"heatmap","/3":"dca",
                   "/4":"retire","/5":"supercharge","/6":"stack","/7":"faq"};
        if (pathname && /^\\/7\\.\\d+$/.test(pathname)) { return "faq"; }
        var tab = map[pathname];
        return tab ? tab : window.dash_clientside.no_update;
    }
    """,
    Output("main-tabs", "active_tab", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call="initial_duplicate",
)


@callback(
    Output("faq-accordion", "active_item"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def open_faq_item(pathname):
    """Open a specific FAQ accordion item when pathname is /7.N (1-indexed)."""
    if not pathname or not pathname.startswith("/7."):
        return no_update
    try:
        n = int(pathname[3:])
        if 1 <= n <= len(_FAQ):
            return f"faq-{n - 1}"
    except (ValueError, IndexError):
        pass
    return no_update


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks — Snapshot / Share
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("share-modal", "is_open"),
    Input("share-btn",          "n_clicks"),
    Input("share-modal-close",  "n_clicks"),
    State("share-modal",        "is_open"),
    prevent_initial_call=True,
)
def toggle_share_modal(n1, n2, is_open):
    return not is_open


@callback(
    *[Output(cid, prop) for cid, prop in _SNAPSHOT_CONTROLS],
    Output("snapshot-lots",     "data"),
    Output("loaded-hash-store", "data"),
    Input("url", "hash"),
    prevent_initial_call=False,
)
def restore_from_url(hash_str):
    n_outs = len(_SNAPSHOT_CONTROLS) + 2
    blank  = [no_update] * n_outs
    if not hash_str:
        return blank
    h = hash_str.lstrip("#")
    if h.startswith(_SNAP_PREFIX):
        state = _decode_snapshot(h[len(_SNAP_PREFIX):])
    elif h.startswith(_SNAP_PREFIX_V1):
        state = _decode_snapshot_v1(h[len(_SNAP_PREFIX_V1):])
    else:
        return blank
    if not state:
        return blank
    results = [state.get(f"{cid}:{prop}", no_update)
               for cid, prop in _SNAPSHOT_CONTROLS]
    results.append(state.get("_lots", None))  # snapshot-lots
    results.append(hash_str)                  # loaded-hash-store
    return results


@callback(
    Output("share-url-display", "value"),
    Output("link-history",      "data"),
    Input("share-copy-btn",    "n_clicks"),
    Input("loaded-hash-store", "data"),
    State("share-scope",        "value"),
    State("share-include-lots", "value"),
    State("lots-store",         "data"),
    *[State(cid, prop) for cid, prop in _SNAPSHOT_CONTROLS],
    State("link-history",       "data"),
    prevent_initial_call=True,
)
def manage_snapshot(n_btn, loaded_hash, share_scope, include_lots, lots_data, *rest):
    *ctrl_vals, history = rest
    history  = list(history or [])
    existing = {h["hash"] for h in history}
    triggered = ctx.triggered_id

    if triggered == "share-copy-btn":
        state = {f"{cid}:{prop}": val
                 for (cid, prop), val in zip(_SNAPSHOT_CONTROLS, ctrl_vals)}
        if include_lots and lots_data:
            state["_lots"] = lots_data
        # Determine active tab and its URL path
        active_tab = state.get("main-tabs:active_tab") or "bubble"
        tab_path   = _TAB_TO_PATH.get(active_tab, "/1")
        # Apply tab filter for single-tab links
        scope      = share_scope or "all"
        tab_filter = _TAB_CONTROLS.get(active_tab) if scope == "tab" else None
        encoded    = _encode_snapshot(state, tab_filter=tab_filter)
        base_url   = flask_request.host_url.rstrip("/")
        full_url   = f"{base_url}{tab_path}#q2:{encoded}"
        if encoded not in existing:
            history.insert(0, {
                "hash": encoded, "url": full_url,
                "ts": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "includes_lots": bool(include_lots and lots_data),
                "scope": scope,
                "tab": active_tab,
            })
            history = history[:50]
        return full_url, history

    if triggered == "loaded-hash-store" and loaded_hash:
        h = loaded_hash.lstrip("#")
        if h.startswith(_SNAP_PREFIX):
            encoded = h[len(_SNAP_PREFIX):]
            state   = _decode_snapshot(encoded)
            prefix  = _SNAP_PREFIX
        elif h.startswith(_SNAP_PREFIX_V1):
            encoded = h[len(_SNAP_PREFIX_V1):]
            state   = _decode_snapshot_v1(encoded)
            prefix  = _SNAP_PREFIX_V1
        else:
            return no_update, no_update
        if not state:
            return no_update, no_update
        active_tab = state.get("main-tabs:active_tab") or "bubble"
        tab_path   = _TAB_TO_PATH.get(active_tab, "/1")
        base_url   = flask_request.host_url.rstrip("/")
        full_url   = f"{base_url}{tab_path}#{prefix}{encoded}"
        if encoded not in existing:
            history.insert(0, {
                "hash": encoded, "url": full_url,
                "ts": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "includes_lots": "_lots" in state,
                "scope": "unknown",
                "tab": active_tab,
            })
            history = history[:50]
            return no_update, history

    return no_update, no_update


@callback(
    Output("effective-lots", "data"),
    Input("lots-store",    "data"),
    Input("snapshot-lots", "data"),
)
def update_effective_lots(local_lots, snapshot_lots):
    return snapshot_lots if snapshot_lots is not None else (local_lots or [])


@callback(
    Output("snapshot-lots-banner", "children"),
    Input("snapshot-lots", "data"),
)
def update_snapshot_banner(snapshot_lots):
    if not snapshot_lots:
        return []
    n = len(snapshot_lots)
    return dbc.Alert([
        html.Span(f"Showing {n} lot(s) from a shared link.  "),
        dbc.Button("Restore my lots", id="restore-lots-btn",
                   color="link", size="sm", className="p-0 ms-1 align-baseline"),
    ], color="info", className="py-1 px-3 mb-2 d-flex align-items-center")


@callback(
    Output("snapshot-lots", "data", allow_duplicate=True),
    Input("restore-lots-btn", "n_clicks"),
    prevent_initial_call=True,
)
def restore_my_lots(_):
    return None


@callback(
    Output("link-history-display", "children"),
    Input("link-history", "data"),
)
def render_link_history(history):
    if not history:
        return html.Small("No links yet.", className="text-muted")
    items = []
    for entry in history:
        badge = (dbc.Badge("lots", color="info", pill=True, className="me-1")
                 if entry.get("includes_lots") else None)
        items.append(dbc.ListGroupItem([
            html.Div([
                html.Small(entry.get("ts", ""), className="text-muted me-2"),
                badge,
            ], className="mb-1"),
            dbc.InputGroup([
                dbc.Input(value=entry.get("url", ""), readonly=True, size="sm",
                          style={"fontFamily":"monospace","fontSize":"11px"}),
            ], size="sm"),
            html.Div(
                html.A("↩ Restore this configuration",
                       href=("#" + entry["url"].split("#", 1)[1])
                            if "#" in entry.get("url", "")
                            else f"#{_SNAP_PREFIX}{entry['hash']}",
                       className="small"),
                className="mt-1",
            ),
        ], className="py-2"))
    return dbc.ListGroup(items, flush=True,
                         style={"maxHeight":"300px","overflowY":"auto"})


@callback(
    Output("link-history", "data", allow_duplicate=True),
    Input("clear-history-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_history(_):
    return []

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, socket
    port = int(os.environ.get("PORT", 8050))
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"
    print(f"\n  Quantoshi Web App")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{local_ip}:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
