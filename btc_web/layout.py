"""Layout builders: tab controls, main layout assembly, splash modal."""
#
# Sections:
#   Imports & helpers ............... ~1-100
#   Shared control builders ........ ~100-170
#   Tab 1: Bubble + QR Overlay ..... ~173-270
#   Tab 2: CAGR Heatmap ............ ~274-418
#   MC controls (shared) ........... ~420-608
#   Tab 3: BTC Accumulator (DCA) ... ~611-700
#   Tab 4: BTC RetireMentator ...... ~705-760
#   Tab 5: HODL Supercharger ....... ~762-868
#   Tab 6: FAQ ..................... ~871-1165
#   Tab 7: Stack Tracker ........... ~1170-1250
#   Splash quotes .................. ~1274-1610
#   App layout assembly ............ ~1612-end

import json
import pandas as pd

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

import _app_ctx
from utils import _nearest_quantile
from snapshot import _SNAPSHOT_CONTROLS
from mc_cache import (CACHED_START_YRS, WD_AMOUNTS, STACK_SIZES,
                      ENTRY_PCT_BINS, MC_YEARS_OPTIONS, INFL_OPTIONS)
from figures import _LOGO_B64_ALL

_PRICE_INTERVAL_MS = 20 * 60 * 1000   # live price ticker refresh (20 minutes)
_MC_POLL_INTERVAL_MS = 3000            # MC payment status poll interval (3 seconds)
_MC_POLL_MAX = 300                     # max poll intervals (300 × 3s = 15 min timeout)

def _q_options():
    opts = []
    for q in _app_ctx._ALL_QS:
        pct = q * 100
        lbl_text = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
        col = _app_ctx.M.qr_colors.get(q, "#888888")
        lbl = html.Span([
            html.Span("\u25CF ", style={"color": col, "fontSize": "10px"}),
            lbl_text,
        ])
        opts.append({"label": lbl, "value": q})
    return opts

_Q_COLLAPSED_HEIGHT = "7.5em"   # ~4 rows visible when collapsed

def _q_panel(checklist_id, default_value, hint=None):
    """Quantile checklist with expand/collapse toggle.

    Clicking anywhere in the panel expands it. After 4 s of no checkbox
    interaction the panel collapses back automatically.
    """
    children = []
    if hint:
        children.append(html.Small(hint,
            style={"color":"#888","display":"block","marginBottom":"4px"}))
    toggle_id = f"{checklist_id}-expand"
    children.extend([
        html.Div(
            dcc.Checklist(id=checklist_id, options=_q_options(),
                          value=default_value, labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
            id=f"{checklist_id}-wrap", className="q-panel-wrap",
            style={"maxHeight": _Q_COLLAPSED_HEIGHT, "overflow": "hidden",
                   "transition": "max-height 0.25s ease"},
        ),
        html.Span("Show all \u25be", id=toggle_id, n_clicks=0,
                  style={"fontSize": "11px", "color": "#1a6fa8", "display": "block",
                         "marginTop": "2px", "cursor": "pointer",
                         "userSelect": "none"}),
    ])
    return _section_card("Quantile Regression Model", *children)


# ══════════════════════════════════════════════════════════════════════════════
# Layout helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ctrl_card(*children):
    return dbc.Card(dbc.CardBody(list(children), className="p-2"),
                    className="mb-2 ctrl-card")

def _section_card(title, *children):
    """Control card with a section header title."""
    return _ctrl_card(html.Div(title, className="ctrl-section-header"), *children)

def _row(*cols):
    return dbc.Row([dbc.Col(c) for c in cols], className="g-1 mb-1")

def _lbl(text):
    return html.Label(text, className="form-label mb-0 small")

def _export_row(tab_id):
    """Export row — download triggered client-side via Plotly.downloadImage()."""
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id=f"{tab_id}-fmt", options=["png","svg","jpeg","webp"], value="png",
                clearable=False, style={"minWidth": "90px"}), width="auto"),
            dbc.Col(dcc.Dropdown(
                id=f"{tab_id}-scale",
                options=[{"label":"1x (e-mail)","value":1},
                         {"label":"2x (screen)","value":2},
                         {"label":"3x (print)","value":3},
                         {"label":"4x (enlargements)","value":4}],
                value=2, clearable=False, style={"minWidth": "130px"}), width="auto"),
            dbc.Col(dbc.Input(id=f"{tab_id}-fname", value=f"btc_{tab_id}",
                              type="text", size="sm"), width=True),
            dbc.Col(dbc.Button("\u2b07 Download", id=f"{tab_id}-export-btn",
                               size="sm"), width="auto"),
            # dummy store — clientside callback needs an output target
            dcc.Store(id=f"{tab_id}-dl-dummy"),
        ], className="g-1 align-items-center"),
        html.Div("\u2193 Scroll down to configure",
                 className="d-md-none text-center text-muted py-1",
                 style={"fontSize":"11px", "letterSpacing":"0.02em"}),
    ], className="export-row-polished")


# ── Shared control builders ──────────────────────────────────────────────────

_BTC_ORANGE = _app_ctx.BTC_ORANGE

def _chart_tab_layout(controls_fn, graph_id, filename, mc_prefix=None):
    """Standard chart tab: 3-col controls (left) + 9-col graph (right).

    mc_prefix: if set, adds an MC overlay div (e.g. "dca" → "dca-mc-overlay").
    """
    overlay = []
    badge = []
    if mc_prefix:
        overlay = [html.Div(id=f"{mc_prefix}-mc-overlay",
                            style={"display": "none"},
                            className="mc-chart-overlay")]
        badge = [html.Img(id=f"{mc_prefix}-mc-badge",
                          src="/assets/quantoshi_favicon.png",
                          className="mc-premium-badge",
                          style={"display": "none"})]
    return dbc.Row([
        dbc.Col(controls_fn(), width=3, className="controls-col overflow-auto",
                style={"maxHeight": "85vh"}),
        dbc.Col([
            html.Div(id=f"{mc_prefix or graph_id}-chart-wrap",
                     style={"position": "relative"}, children=[
                dcc.Loading(
                    dcc.Graph(id=graph_id, style={"height": "78vh"},
                              config={"toImageButtonOptions": {"format": "png", "scale": 2,
                                                               "filename": filename}}),
                    type="default", color=_BTC_ORANGE,
                ),
                *overlay,
                *badge,
            ]),
            _export_row(graph_id.replace("-graph", "")),
        ], width=9),
    ], className="g-0")


def _year_range_slider(prefix, min_yr, max_yr, default_start, default_end, mark_step=5):
    """Year range slider with abbreviated tick marks."""
    return dcc.RangeSlider(
        id=f"{prefix}-yr-range", min=min_yr, max=max_yr,
        value=[default_start, default_end], step=1,
        marks={y: f"'{y % 100:02d}" for y in range(min_yr, max_yr + 1, mark_step)},
        tooltip={"always_visible": False},
    )


def _freq_dropdown(prefix, default="Monthly"):
    """Frequency dropdown (Daily/Weekly/Monthly/Quarterly/Annually)."""
    return dcc.Dropdown(
        id=f"{prefix}-freq",
        options=["Daily", "Weekly", "Monthly", "Quarterly", "Annually"],
        value=default, clearable=False,
    )


def _stack_control_card(prefix, default_btc=0, header=True):
    """Starting Stack card with BTC input and Use Stack Tracker lots checkbox."""
    children = []
    if header:
        children.append(html.Div("Starting Stack", className="ctrl-section-header"))
    children.extend([
        _lbl("Starting BTC"),
        dbc.Input(id=f"{prefix}-stack", type="number", value=default_btc,
                  min=0, step=0.001, size="sm"),
        dcc.Checklist(id=f"{prefix}-use-lots",
                      options=[{"label": " Use Stack Tracker lots", "value": "yes"}],
                      value=[], inputStyle={"marginRight": "5px"}),
    ])
    return _ctrl_card(*children)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Bubble + QR Overlay
# ══════════════════════════════════════════════════════════════════════════════

def _bubble_controls():
    yr_now = pd.Timestamp.today().year
    return html.Div([
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
            html.Div(id="bub-yrange-wrap", style={"display":"none"}, children=[
                dcc.RangeSlider(id="bub-yrange", min=-2, max=8,
                                value=[0, 7], step=0.5,
                                marks={-2:"1¢", 0:"$1", 2:"$100",
                                        4:"$10K", 6:"$1M", 8:"$100M"},
                                tooltip={"always_visible":False}),
            ]),
        ),
        _section_card("Display",
            dcc.Checklist(id="bub-toggles",
                          options=[{"label":" Shade bands","value":"shade"},
                                   {"label":" Show OLS","value":"show_ols"},
                                   {"label":" Show data","value":"show_data"},
                                   {"label":" Show today","value":"show_today"},
                                   {"label":" Show legend","value":"show_legend"},
                                   {"label":html.Span(" Minor grid",className="minor-grid-opt"),"value":"minor_grid"}],
                          value=["shade","show_data","show_today"],
                          labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
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
        _q_panel("bub-qs", []),
        _ctrl_card(
            _lbl("Data Point Appearance"),
            _row(
                html.Div([_lbl("Pt size (1–20)"),
                          dbc.Input(id="bub-ptsize", type="number",
                                    value=3, min=1, max=20, size="sm")]),
                html.Div([_lbl("Alpha (0.1–1)"),
                          dbc.Input(id="bub-ptalpha", type="number",
                                    value=0.3, min=0.1, max=1.0, step=0.05, size="sm")]),
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
            dcc.Checklist(id="bub-use-lots",
                          options=[{"label":" Use Stack Tracker lots","value":"yes"}],
                          value=[], inputStyle={"marginRight":"5px"}),
        ),
    ])


def _bubble_tab():
    return _chart_tab_layout(_bubble_controls, "bubble-graph", "btc_bubble")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — CAGR Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def _heatmap_controls():
    yr_now = pd.Timestamp.today().year
    return html.Div([
        _section_card("Entry Point",
            _lbl("Entry year"),
            dcc.Slider(id="hm-entry-yr", min=2010, max=2039,
                       value=yr_now, step=1, marks=None,
                       tooltip={"always_visible":True}),
            _lbl("Entry percentile (0.1–99.9%)"),
            dbc.Input(id="hm-entry-q", type="number",
                      value=_app_ctx._HM_ENTRY_Q_DEFAULT,
                      min=0.1, max=99.9, step=0.1, size="sm"),
        ),
        _section_card("Exit Range",
            _lbl("Exit year range"),
            dcc.RangeSlider(id="hm-exit-range", min=2010, max=2060,
                            value=[yr_now, yr_now + 15], step=1,
                            marks={y: f"'{y % 100:02d}" for y in range(2010, 2061, 5)},
                            tooltip={"always_visible":False}),
        ),
        _q_panel("hm-exit-qs", _app_ctx._DEF_QS),
        _section_card("Color & Style",
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
                           value=_app_ctx.M.CAGR_SEG_C_LO, style={"height":"28px"})]),
                html.Div([_lbl("Mid1"), dbc.Input(id="hm-c-mid1", type="color",
                           value=_app_ctx.M.CAGR_SEG_C_MID1, style={"height":"28px"})]),
                html.Div([_lbl("Mid2"), dbc.Input(id="hm-c-mid2", type="color",
                           value=_app_ctx.M.CAGR_SEG_C_MID2, style={"height":"28px"})]),
                html.Div([_lbl("Hi"), dbc.Input(id="hm-c-hi", type="color",
                           value=_app_ctx.M.CAGR_SEG_C_HI, style={"height":"28px"})]),
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
        # MC controls at the very bottom, below all quantile config
        _mc_controls("hm", show_amount=False, show_mc_entry_q=True, default_entry_q=10),
    ])


def _heatmap_tab():
    return dbc.Row([
        dbc.Col(_heatmap_controls(), width=3, className="controls-col overflow-auto",
                style={"maxHeight":"85vh"}),
        dbc.Col([
            # Swipe indicator (hidden when MC disabled)
            html.Div([
                html.Span("◀ ", style={"opacity":"0.5"}),
                html.Span("Quantile Regression", id="hm-sw-qr-lbl",
                           className="fw-bold", style={"cursor":"pointer"}),
                html.Span("  ·  ", style={"opacity":"0.4"}),
                html.Span("Monte Carlo", id="hm-sw-mc-lbl",
                           style={"cursor":"pointer", "opacity":"0.5"}),
                html.Span(" ▶", style={"opacity":"0.5"}),
            ], id="hm-swipe-indicator", className="text-center py-1",
               style={"display":"none", "fontSize":"0.85rem", "color":"#6c757d",
                       "userSelect":"none"}),
            # Swipe container
            html.Div([
                html.Div([
                    dcc.Loading(
                        dcc.Graph(id="heatmap-graph", style={"height":"78vh"},
                                  config={"scrollZoom":False,
                                          "toImageButtonOptions":{"format":"png","scale":2,
                                                                   "filename":"btc_heatmap"}}),
                        type="default", color=_BTC_ORANGE,
                    ),
                ], className="hm-swipe-panel"),
                html.Div([
                    dcc.Loading(
                        dcc.Graph(id="hm-mc-graph", style={"height":"78vh"},
                                  config={"scrollZoom":False,
                                          "toImageButtonOptions":{"format":"png","scale":2,
                                                                   "filename":"btc_mc_heatmap"}}),
                        type="default", color=_BTC_ORANGE,
                    ),
                    # MC chart overlay (gray mask when MC not rendered)
                    html.Div(id="hm-mc-overlay", style={"display": "none"},
                             className="mc-chart-overlay"),
                    html.Img(id="hm-mc-badge",
                             src="/assets/quantoshi_favicon.png",
                             className="mc-premium-badge",
                             style={"display": "none"}),
                ], className="hm-swipe-panel mc-premium-chart", id="hm-mc-panel",
                   style={"display":"none"}),
            ], className="hm-swipe-container", id="hm-swipe-wrap"),
            html.Div(id="hm-swipe-scroll-dummy", style={"display":"none"}),
            _export_row("heatmap"),
        ], width=9),
    ], className="g-0")


# ── Shared MC controls (DCA + Retire) ────────────────────────────────────────

_QUANT_FONT = {"fontFamily": '"Palatino Linotype", Palatino, "Book Antiqua", serif',
               "color": "#000", "letterSpacing": "1px"}
_MC_CACHED_START_YRS = set(CACHED_START_YRS)
_MC_CACHED_ENTRY_QS = {int(v * 100) for v in ENTRY_PCT_BINS}   # {10,20,...,90}
_MC_CACHED_YEARS    = set(MC_YEARS_OPTIONS)                      # {10,20,30,40}
_MC_CACHED_WD       = set(WD_AMOUNTS)
_MC_CACHED_INFL     = set(INFL_OPTIONS)
_MC_CACHED_STACKS   = set(STACK_SIZES)

def _bold_opts(values, fmt, cached_set):
    """Build dropdown options, bolding+enlarging values in the pre-computed cache."""
    return [
        {"label": html.Span(fmt(v), style={"fontWeight": "bold", "fontSize": "16px"})
                  if v in cached_set else fmt(v),
         "value": v}
        for v in values
    ]

# ── MC pricing (sats) ────────────────────────────────────────────────────────
# Cached: pre-computed paths on server, instant lookup
# Non-cached: live Markov chain simulation (~1-3s compute)
_MC_PRICE_CACHED = {10: 100, 20: 200, 30: 300, 40: 400}
_MC_PRICE_LIVE   = {10: 500, 20: 1000, 30: 1500, 40: 2000}
_MC_START_YR_OPTIONS = _bold_opts(range(2026, 2051), str, _MC_CACHED_START_YRS)
_MC_ENTRY_Q_OPTIONS  = _bold_opts(
    [int(v * 100) for v in ENTRY_PCT_BINS],
    lambda v: f"{v}%", _MC_CACHED_ENTRY_QS)
_MC_ENTRY_Q_OPTIONS_ADV = _bold_opts(
    list(range(1, 100)),
    lambda v: f"{v}%", _MC_CACHED_ENTRY_QS)
_MC_YEARS_OPTIONS    = _bold_opts(MC_YEARS_OPTIONS, lambda v: f"{v} yr", _MC_CACHED_YEARS)
_MC_WD_OPTIONS       = _bold_opts(WD_AMOUNTS, lambda v: f"${v:,}/mo", _MC_CACHED_WD)
_MC_INFL_OPTIONS     = _bold_opts(INFL_OPTIONS, lambda v: f"{v}%", _MC_CACHED_INFL)
_MC_STACK_OPTIONS    = _bold_opts(STACK_SIZES, lambda v: f"{v} BTC", _MC_CACHED_STACKS)

def _mc_controls(prefix, amount_label="Per-period amount ($)", amount_default=100,
                  show_inflation=False, show_amount=True,
                  amount_dropdown=False, show_stack=False, show_mc_entry_q=False,
                  default_entry_q=50):
    """Monte Carlo simulation controls, reusable across tabs."""
    yr_now = pd.Timestamp.today().year
    if not _app_ctx._HAS_MARKOV:
        # Hidden placeholders so callback IDs exist even without markov module
        return html.Div(style={"display": "none"}, children=[
            dcc.Checklist(id=f"{prefix}-mc-enable", value=[]),
            dcc.Checklist(id=f"{prefix}-mc-advanced", value=[]),
            dbc.Input(id=f"{prefix}-mc-amount", value=amount_default),
            dbc.Input(id=f"{prefix}-mc-infl", value=4),
            dbc.Input(id=f"{prefix}-mc-bins", value=5),
            dcc.Dropdown(id=f"{prefix}-mc-sims", value=800),
            dcc.Dropdown(id=f"{prefix}-mc-years", value=10),
            dcc.Dropdown(id=f"{prefix}-mc-freq", value="Monthly"),
            dbc.Input(id=f"{prefix}-mc-ppy", value="12/yr"),
            dcc.RangeSlider(id=f"{prefix}-mc-window", min=2010,
                            max=yr_now,
                            value=[2010, yr_now]),
            html.Div(id=f"{prefix}-mc-adv-body"),
            html.Div(id=f"{prefix}-mc-cost"),
            dcc.Store(id=f"{prefix}-mc-price-val", storage_type="memory", data=0),
            html.Div(id=f"{prefix}-mc-body"),
            html.Div(id=f"{prefix}-mc-status"),
            dbc.Button(id=f"{prefix}-mc-dl-btn", style={"display":"none"}),
            dbc.Button(id=f"{prefix}-mc-run-btn", style={"display":"none"}),
            html.Div(id=f"{prefix}-mc-run-status"),
            dcc.Store(id=f"{prefix}-mc-rendered-key", storage_type="memory"),
            html.Div(id=f"{prefix}-mc-match"),
            dcc.Upload(id=f"{prefix}-mc-upload"),
            html.Div(id=f"{prefix}-mc-upload-status"),
            dcc.Slider(id=f"{prefix}-mc-entry-yr", value=yr_now),
            dcc.Dropdown(id=f"{prefix}-mc-entry-q",
                        value=max(10, min(90, round(_app_ctx._HM_ENTRY_Q_DEFAULT / 10) * 10 or 50))),
            dcc.Dropdown(id=f"{prefix}-mc-stack", value=1.0),
            dcc.Dropdown(id=f"{prefix}-mc-start-yr", value=2031),
        ])
    yr_now = pd.Timestamp.today().year
    return html.Div(style={"position": "relative"}, children=[
        html.Span([
            html.Span("\u2694", style={"fontSize": "16px", "marginRight": "3px"}),
            html.Span("NEW", style={"position": "relative", "top": "-2px"}),
        ], className="mc-new-badge", style={
            "position": "absolute", "top": "4px", "right": "-2px",
            "fontWeight": "900", "color": "#c0c0c0",
            "fontFamily": "'Impact', 'Arial Black', sans-serif",
            "textTransform": "uppercase",
            "backgroundColor": "#1a1a1a",
            "borderRadius": "5px", "transform": "rotate(18deg)",
            "zIndex": "1", "lineHeight": "1.2",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1)",
            "textShadow": "0 0 4px rgba(139,0,0,0.6)",
        }),
        _ctrl_card(
        html.Span([
            html.B("Monte Carlo", style={"fontSize": "12px"}),
            html.Span([
                html.Span("\u26a1", style={"fontSize": "15px"}),
                " Paid feature",
            ], style={"fontSize": "10px", "color": "#6b5300",
                      "marginLeft": "6px", "fontWeight": "normal",
                      "backgroundColor": "rgba(184,134,11,0.12)",
                      "padding": "1px 6px", "borderRadius": "4px"}),
        ]),
        dcc.Checklist(id=f"{prefix}-mc-enable",
                      options=[{"label": " Activate Markov chain stochastic engine", "value": "yes"}],
                      value=[], inputStyle={"marginRight": "5px"}),
        html.Div(id=f"{prefix}-mc-body", style={"display": "none"}, children=[
            dcc.Checklist(id=f"{prefix}-mc-advanced",
                          options=[{"label": " Advanced simulator options", "value": "yes"}],
                          value=[], inputStyle={"marginRight": "5px"},
                          style={"fontSize": "11px", "color": "#666", "marginBottom": "6px"}),
            html.Div(dcc.Slider(id=f"{prefix}-mc-entry-yr", value=yr_now),
                     style={"display": "none"}),
            *([ _lbl("Retirement start year (bold = cached)"),
                dcc.Dropdown(id=f"{prefix}-mc-start-yr",
                             options=_MC_START_YR_OPTIONS,
                             value=2031, clearable=False),
                _lbl("Entry percentile (10% steps, cache-aligned)"),
                dcc.Dropdown(id=f"{prefix}-mc-entry-q",
                             options=_MC_ENTRY_Q_OPTIONS,
                             value=default_entry_q, clearable=False),
            ] if show_stack else [
                _lbl("MC start year (bold = cached)"),
                dcc.Dropdown(id=f"{prefix}-mc-start-yr",
                             options=_MC_START_YR_OPTIONS,
                             value=2026, clearable=False),
                *([_lbl("Entry percentile (10% steps, cache-aligned)"),
                   dcc.Dropdown(id=f"{prefix}-mc-entry-q",
                                options=_MC_ENTRY_Q_OPTIONS,
                                value=default_entry_q, clearable=False),
                ] if show_mc_entry_q else []),
            ]),
            *([_lbl(amount_label),
               dcc.Dropdown(id=f"{prefix}-mc-amount",
                            options=_MC_WD_OPTIONS,
                            value=amount_default, clearable=False),
              ] if show_amount and amount_dropdown else
              [_lbl(amount_label),
               dbc.Input(id=f"{prefix}-mc-amount", type="number",
                         value=amount_default, min=1, step=1, size="sm"),
              ] if show_amount else [
               dbc.Input(id=f"{prefix}-mc-amount", type="number",
                         value=amount_default, style={"display": "none"}),
              ]),
            *([ _lbl("Inflation rate (% / yr)"),
                dcc.Dropdown(id=f"{prefix}-mc-infl",
                             options=_MC_INFL_OPTIONS,
                             value=4, clearable=False),
            ] if show_inflation else [
                dcc.Dropdown(id=f"{prefix}-mc-infl", value=0,
                             style={"display": "none"}),
            ]),
            *([ _lbl("Starting BTC stack"),
                dcc.Dropdown(id=f"{prefix}-mc-stack",
                             options=_MC_STACK_OPTIONS,
                             value=1.0, clearable=False),
            ] if show_stack else [
                dcc.Dropdown(id=f"{prefix}-mc-stack", value=1.0,
                             style={"display": "none"}),
            ]),
            _lbl("Years to model"),
            dcc.Dropdown(id=f"{prefix}-mc-years",
                         options=_MC_YEARS_OPTIONS,
                         value=10, clearable=False),
            # Advanced controls (hidden until checkbox toggled)
            html.Div(id=f"{prefix}-mc-adv-body", style={"display": "none"}, children=[
                _lbl("Markov transition matrix dimension"),
                dcc.Dropdown(id=f"{prefix}-mc-bins",
                             options=_bold_opts(
                                 list(range(5, 11)),
                                 lambda v: f"{v}×{v}", {5}),
                             value=5, clearable=False),
                _lbl("Simulations"),
                dcc.Dropdown(id=f"{prefix}-mc-sims",
                             options=_bold_opts(
                                 [100, 200, 400, 800, 1600, 3200],
                                 str, {100, 200, 400, 800}),
                             value=800, clearable=False),
                _lbl("Frequency"),
                dcc.Dropdown(id=f"{prefix}-mc-freq",
                             options=_bold_opts(
                                 ["Monthly", "Weekly", "Daily"],
                                 str, {"Monthly"}),
                             value="Monthly", clearable=False),
                _lbl("Periods per year"),
                dbc.Input(id=f"{prefix}-mc-ppy", value="12/yr", size="sm",
                          disabled=True),
                _lbl("Historical window"),
                dcc.RangeSlider(id=f"{prefix}-mc-window", min=2010,
                                max=yr_now, value=[2010, yr_now],
                                marks={y: str(y) for y in range(2010, yr_now + 1, 5)}),
            ]),
            html.Div(id=f"{prefix}-mc-cost",
                     style={"fontSize": "11px", "color": "#555", "marginTop": "6px",
                            "lineHeight": "1.4"}),
            dcc.Store(id=f"{prefix}-mc-price-val", storage_type="memory", data=0),
            # ── Run Simulation button (payment-gated when BTCPay active) ──
            dbc.Button(
                [html.Span("\u26a1 ", style={"fontSize": "14px"}), "Run Simulation"],
                id=f"{prefix}-mc-run-btn", size="sm", color="warning",
                className="w-100 mt-2",
                style={"fontWeight": "600"},
            ),
            html.Div(id=f"{prefix}-mc-run-status",
                     style={"fontSize": "10px", "color": "#555", "marginTop": "4px",
                            "textAlign": "center"}),
            dcc.Store(id=f"{prefix}-mc-rendered-key", storage_type="memory"),
            html.Div(id=f"{prefix}-mc-match",
                     style={"fontSize": "10px", "marginTop": "4px",
                            "textAlign": "center"}),
            html.Hr(className="my-2"),
            html.Div("Saved Simulation", className="ctrl-section-header"),
            html.Div(id=f"{prefix}-mc-status",
                     style={"fontSize": "10px", "color": "#555", "marginBottom": "4px"}),
            dbc.Row([
                dbc.Col(
                    dbc.Button("\u2b07 Save", id=f"{prefix}-mc-dl-btn",
                               size="sm", color="secondary", className="w-100"),
                    width=6),
                dbc.Col(
                    dcc.Upload(
                        id=f"{prefix}-mc-upload",
                        children=dbc.Button("\u2b06 Load", size="sm",
                                            color="secondary", className="w-100"),
                        accept=".json", multiple=False,
                    ),
                    width=6),
            ], className="g-1"),
            html.Div(id=f"{prefix}-mc-upload-status", className="mt-1",
                     style={"fontSize": "10px"}),
        ]),
    ),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — BTC Accumulator (DCA)
# ══════════════════════════════════════════════════════════════════════════════

def _dca_controls():
    yr_now = pd.Timestamp.today().year
    return html.Div([
        _section_card("DCA Plan",
            _lbl("Per-period amount ($)"),
            dbc.Input(id="dca-amount", type="number", value=100,
                      min=1, step=1, size="sm"),
            _lbl("Frequency"),
            _freq_dropdown("dca"),
            _lbl("Year range"),
            _year_range_slider("dca", 2009, 2060, yr_now, yr_now + 10),
        ),
        _section_card("Display",
            dcc.Dropdown(id="dca-disp",
                         options=[{"label":"BTC Balance","value":"btc"},
                                  {"label":"USD Value","value":"usd"}],
                         value="btc", clearable=False),
            dcc.Checklist(id="dca-toggles",
                          options=[{"label":" Log Y","value":"log_y"},
                                   {"label":" Dual Y-axis","value":"dual_y"},
                                   {"label":" Show legend","value":"show_legend"},
                                   {"label":html.Span(" Minor grid",className="minor-grid-opt"),"value":"minor_grid"}],
                          value=["show_legend","dual_y"], labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
        ),
        _q_panel("dca-qs", [0.5],
                 hint="Price path drives sat accumulation — lower quantile = lower price = more sats/period."),
        _stack_control_card("dca", default_btc=0),
        _stackcelerator_controls(),
        _mc_controls("dca", amount_label="DCA amount per period ($)", amount_default=100,
                     show_mc_entry_q=True),
    ])


def _stackcelerator_controls():
    return _ctrl_card(
        html.B("Stack-celerator", style={"fontSize":"12px"}),
        dcc.Checklist(id="dca-sc-enable",
                      options=[{"label":" Activate Saylor Mode","value":"yes"}],
                      value=[], inputStyle={"marginRight":"5px"}),
        # Why html.Div(display:none) instead of dbc.Collapse: Dash Collapse
        # unmounts its children, destroying component state on toggle.
        html.Div(id="dca-sc-body", style={"display":"none"}, children=[
            _lbl("Loan amount ($)"),
            dbc.Input(id="dca-sc-loan", type="number",
                      value=1200, min=0, step=1, size="sm"),
            _lbl("Entry price (1st cycle)"),
            dcc.Dropdown(id="dca-sc-entry-mode",
                         options=[{"label":"Live ticker","value":"live"},
                                  {"label":"Model price","value":"model"},
                                  {"label":"Custom price","value":"custom"}],
                         value="live", clearable=False),
            html.Div(id="dca-sc-custom-price-row", style={"display":"none"}, children=[
                _lbl("Custom entry price ($)"),
                dbc.Input(id="dca-sc-custom-price", type="number",
                          value=80000, min=1, step=1, size="sm"),
            ]),
            _lbl("Loan type"),
            dcc.Dropdown(id="dca-sc-type",
                         options=[{"label":"Interest-only","value":"interest_only"},
                                  {"label":"Amortizing","value":"amortizing"}],
                         value="interest_only", clearable=False),
            html.Div(id="dca-sc-rollover-row", children=[
                dbc.Checklist(id="dca-sc-rollover",
                              options=[{"label":" Roll over (refinance; no BTC sold between cycles)",
                                        "value":"yes"}],
                              value=[], inputStyle={"marginRight":"5px"}),
            ]),
            _lbl("Annual interest rate (0–100% / yr)"),
            dbc.Input(id="dca-sc-rate", type="number",
                      value=13.0, min=0, max=100, step=0.5, size="sm"),
            _lbl("Loan term (months)"),
            dbc.Input(id="dca-sc-term", type="number",
                      value=12, min=1, max=360, step=1, size="sm"),
            _lbl("Additional loan cycles (0 = one loan)"),
            dbc.Input(id="dca-sc-repeats", type="number",
                      value=0, min=0, step=1, size="sm"),
            _lbl("Capital gains tax on repayment (0–100%)"),
            dbc.Input(id="dca-sc-tax", type="number",
                      value=33, min=0, max=99, step=0.5, size="sm"),
            html.Div(id="dca-sc-info",
                     style={"fontSize":"11px","color":"#555","marginTop":"4px"}),
        ]),
    )


def _dca_tab():
    return _chart_tab_layout(_dca_controls, "dca-graph", "btc_dca", mc_prefix="dca")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — BTC Retireator
# ══════════════════════════════════════════════════════════════════════════════

def _retire_controls():
    yr_now = pd.Timestamp.today().year
    return html.Div([
        _section_card("Withdrawal Plan",
            _lbl("Withdrawal/period ($)"),
            dbc.Input(id="ret-wd", type="number", value=5000,
                      min=1, step=1, size="sm"),
            _lbl("Frequency"),
            _freq_dropdown("ret"),
            _lbl("Year range"),
            _year_range_slider("ret", 2024, 2075, 2031, 2075),
            _lbl("Inflation rate (0–100% / yr)"),
            dbc.Input(id="ret-infl", type="number", value=4,
                      min=0, max=100, step=0.5, size="sm"),
        ),
        _section_card("Display",
            dcc.Dropdown(id="ret-disp",
                         options=[{"label":"BTC Remaining","value":"btc"},
                                  {"label":"USD Value","value":"usd"}],
                         value="btc", clearable=False),
            dcc.Checklist(id="ret-toggles",
                          options=[{"label":" Log Y","value":"log_y"},
                                   {"label":" Dual Y-axis","value":"dual_y"},
                                   {"label":" Annotate depletion","value":"annotate"},
                                   {"label":" Show today","value":"show_today"},
                                   {"label":" Show legend","value":"show_legend"},
                                   {"label":html.Span(" Minor grid",className="minor-grid-opt"),"value":"minor_grid"}],
                          value=["annotate","log_y","dual_y","minor_grid"],
                          labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
            _lbl("Legend position"),
            dcc.Dropdown(id="ret-legend-pos",
                         options=[{"label":"Outside (right)","value":"outside"},
                                  {"label":"Top-left","value":"top-left"},
                                  {"label":"Top-right","value":"top-right"},
                                  {"label":"Bottom-left","value":"bottom-left"},
                                  {"label":"Bottom-right","value":"bottom-right"}],
                         value="outside", clearable=False),
        ),
        _q_panel("ret-qs", [0.01, 0.10, 0.25],
                 hint="Lower quantile = lower price = faster depletion \u2014 worst-case planning."),
        _stack_control_card("ret", default_btc=1.0),
        _mc_controls("ret", amount_label="Withdrawal per period ($)", amount_default=5000,
                     show_inflation=True, amount_dropdown=True, show_stack=True),
    ])


def _retire_tab():
    return _chart_tab_layout(_retire_controls, "retire-graph", "btc_retire", mc_prefix="ret")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — HODL Supercharger
# ══════════════════════════════════════════════════════════════════════════════

def _supercharge_controls():
    yr_now = pd.Timestamp.today().year
    display_q_opts = _q_options()
    display_q_default = _nearest_quantile(0.05, _app_ctx._ALL_QS)
    return html.Div([
        # ── Scenario card: mode + timeline + delays + freq/infl + mode-specific ──
        _section_card("Scenario",
            dcc.RadioItems(id="sc-mode",
                options=[{"label":" A \u2014 Fixed spending (depletion date)","value":"a"},
                         {"label":" B \u2014 Fixed depletion (max spending)","value":"b"}],
                value="a", labelStyle={"display":"block"},
                inputStyle={"marginRight":"5px"}),
            dbc.Collapse(
                html.Div(
                    "\u2248YYYY annotations mark the year each scenario\u2019s BTC stack reaches zero \u2014 savings exhausted.",
                    style={"fontSize":"10px","color":"#888","marginTop":"6px",
                           "lineHeight":"1.4"},
                ),
                id="sc-depl-note-collapse", is_open=True,
            ),
            html.Hr(className="my-2"),
            _lbl("Base retirement year"),
            dcc.Slider(id="sc-start-yr", min=yr_now, max=2075,
                       value=2033, step=1,
                       marks={y: f"'{y % 100:02d}" for y in range(yr_now, 2076, 5)},
                       tooltip={"always_visible":False}),
            _lbl("Delay offsets (years)"),
            dbc.Row([
                dbc.Col(dbc.Input(id="sc-d0", type="number", value=0,
                                  min=0, step=1, size="sm"), width=True),
                dbc.Col(dbc.Input(id="sc-d1", type="number", value=0,
                                  min=0, step=1, size="sm"), width=True),
                dbc.Col(dbc.Input(id="sc-d2", type="number", value=0,
                                  min=0, step=1, size="sm"), width=True),
                dbc.Col(dbc.Input(id="sc-d3", type="number", value=1,
                                  min=0, step=1, size="sm"), width=True),
                dbc.Col(dbc.Input(id="sc-d4", type="number", value=2,
                                  min=0, step=1, size="sm"), width=True),
            ], className="g-1"),
            _lbl("Frequency"),
            _freq_dropdown("sc", default="Annually"),
            _lbl("Inflation rate (0\u2013100% / yr)"),
            dbc.Input(id="sc-infl", type="number", value=4,
                      min=0, max=100, step=0.5, size="sm"),
            html.Hr(className="my-2"),
            dbc.Collapse([
                _lbl("Withdrawal/period ($)"),
                dbc.Input(id="sc-wd", type="number", value=100000,
                          min=1, step=1, size="sm"),
                _lbl("End year"),
                html.Div(dcc.Slider(id="sc-end-yr", min=2030, max=2100,
                           value=2075, step=1,
                           marks={y: f"'{y % 100:02d}" for y in range(2030, 2101, 10)},
                           tooltip={"always_visible":False})),
            ], id="sc-mode-a-collapse", is_open=True),
            dbc.Collapse([
                _lbl("Target depletion year"),
                html.Div(dcc.Slider(id="sc-target-yr", min=2030, max=2100,
                           value=2060, step=1,
                           marks={y: f"'{y % 100:02d}" for y in range(2030, 2101, 10)},
                           tooltip={"always_visible":False})),
            ], id="sc-mode-b-collapse", is_open=False),
        ),
        # ── Display card: chart layout + display-q + BTC/USD + toggles ──
        _section_card("Display",
            dcc.Checklist(id="sc-chart-layout",
                options=[{"label":" Shade quantile bands","value":"shade"}],
                value=["shade"],
                inputStyle={"marginRight":"5px"}),
            dbc.Collapse([
                _lbl("Display quantile"),
                dcc.Dropdown(id="sc-display-q", options=display_q_opts,
                             value=display_q_default, clearable=False),
            ], id="sc-display-q-collapse", is_open=True),
            dcc.Dropdown(id="sc-disp",
                         options=[{"label":"BTC Remaining","value":"btc"},
                                  {"label":"USD Value","value":"usd"}],
                         value="usd", clearable=False),
            dcc.Checklist(id="sc-toggles",
                          options=[{"label":" Annotate depletion","value":"annotate"},
                                   {"label":" Show today","value":"show_today"},
                                   {"label":" Log Y","value":"log_y"},
                                   {"label":" Show legend","value":"show_legend"},
                                   {"label":html.Span(" Minor grid",className="minor-grid-opt"),"value":"minor_grid"}],
                          value=["annotate","log_y","show_legend","minor_grid"],
                          labelStyle={"display":"block"},
                          inputStyle={"marginRight":"5px"}),
        ),
        _q_panel("sc-qs", [q for q in [0.001, 0.10] if q in _app_ctx.M.qr_fits],
                 hint="Lower quantile = earlier depletion \u2014 use multiple quantiles to see the range."),
        _stack_control_card("sc", default_btc=1.0),
        # Hidden MC controls — keeps component IDs alive for callbacks
        html.Div(_mc_controls("sc", amount_label="Withdrawal per period ($)",
                              amount_default=5000, show_inflation=True,
                              amount_dropdown=True, show_stack=True),
                 style={"display": "none"}),
    ])


def _supercharge_tab():
    return _chart_tab_layout(_supercharge_controls, "supercharge-graph", "btc_supercharge")


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
        "q": "What is the difference between Quantile Regression and Markov Chain Monte Carlo?",
        "a": html.Span([
            "Quantile Regression (QR) fits smooth percentile curves to the historical log-log "
            "relationship between time and Bitcoin price. It is deterministic: given a percentile "
            "and a date, it returns exactly one price. The computation is a single matrix solve — "
            "fast enough to run in your browser in milliseconds. QR tells you ",
            html.I("where"),
            " a given percentile has historically fallen, but it assumes the future follows the "
            "same smooth power-law trend.",
            html.Br(), html.Br(),
            "Markov Chain Monte Carlo (MCMC) simulation is fundamentally different. It models "
            "Bitcoin's price as a random walk governed by a transition matrix estimated from "
            "historical returns. At each time step the simulation draws a random move from the "
            "learned distribution, so every run produces a different price path — and each path "
            "consists of hundreds of sequential transitions. To get stable "
            "statistics (median, percentile bands), we need to repeat this hundreds of times — "
            "Quantoshi runs several hundred simulations per scenario.",
            html.Br(), html.Br(),
            "This is why MCMC is computationally expensive: each path requires stepping through "
            "hundreds of time periods in sequence, and we repeat this hundreds of times. A single QR "
            "lookup is O(1); a single MCMC fan requires O(paths \u00d7 periods) floating-point "
            "operations — hundreds of paths times hundreds of steps per path.",
            html.Br(), html.Br(),
            "To keep this tractable, Quantoshi pre-computes a cache of over 45,000 scenarios "
            "covering different entry percentiles, time horizons, withdrawal amounts, inflation "
            "rates, and stack sizes. The full cache occupies roughly 834 MB of RAM, loaded at "
            "server startup from compressed arrays on disk. A compiled Cython engine generates "
            "the cache offline; at runtime, lookups are instantaneous.",
            html.Br(), html.Br(),
            "When you choose parameters outside the pre-computed cache — a custom withdrawal "
            "amount, a different inflation rate, or a start year we haven't cached — Quantoshi "
            "has to run a fresh simulation on the server in real time. This means generating "
            "hundreds of full price paths from scratch, each stepping through hundreds of "
            "transitions, and then computing withdrawal overlays on top of them. That is why "
            "custom simulations cost a small lightning payment and may take a few seconds to "
            "return: you are paying for dedicated compute time that cannot be amortized across "
            "other users.",
        ]),
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
        "q": "What is the Stack-celerator on the DCA tab?",
        "a": html.Span([
            html.Span(
                "It's \u201cActivate Saylor Mode\u201d \u2014 a strategy popularized by Michael Saylor and "
                "MicroStrategy: instead of only dollar-cost averaging, you also borrow money "
                "and use the loan proceeds to buy a lump sum of Bitcoin up front. You then "
                "service the loan from your regular DCA contributions. If Bitcoin appreciates "
                "faster than your interest rate — historically a very safe bet — you end up "
                "with significantly more Bitcoin than plain DCA would have gotten you. "
                "The dashed overlay lines on the chart show your projected stack with the loan "
                "versus without (solid lines). The Stack-celeration factor in the chart title "
                "tells you how many times better the loan strategy performs versus plain DCA "
                "at the median."
            ),
            html.Br(), html.Br(),
            html.Span(
                "Two loan types are available. "
                "Interest-only: you pay just the interest each period and repay the full "
                "principal at the end of each term by selling some Bitcoin — subject to capital "
                "gains tax. "
                "Amortizing: like a standard mortgage, each payment covers both interest and a "
                "slice of the principal. No Bitcoin needs to be sold; the loan is paid off "
                "entirely in fiat from your DCA contributions."
            ),
            html.Br(), html.Br(),
            html.Span(
                "For interest-only loans you can also enable Roll over, which is the more "
                "realistic HODLer approach: instead of selling Bitcoin to repay at term end, "
                "you refinance into a new loan. Your Bitcoin is never sold mid-simulation — "
                "only once, at the very end of the final term."
            ),
            html.Br(), html.Br(),
            html.Span(
                "Please note: it is possible to compound losses by using a loan if you buy "
                "Bitcoin at a high percentile and sell at a lower percentile \u2014 sometimes "
                "even many years in the future, according to quantile regression extrapolations. "
                "Be careful when you choose to predict Bitcoin price from historical price data. "
                "Past performance is not a guarantee of future returns."
            ),
        ]),
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
    # Note: item_id uses loop index -- reordering _FAQ entries will break
    # direct links (/7.N) since they reference items by position.
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

_app_ctx.app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/png" href="/assets/quantoshi_favicon.png">
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

# ── Splash quotes ─────────────────────────────────────────────────────────────
_SPLASH_QUOTES = [
    # Satoshi
    ("If you don't believe me or don't get it, I don't have time to try to convince you, sorry.",
     "Satoshi Nakamoto"),
    ("The root problem with conventional currency is all the trust that's required to make it work.",
     "Satoshi Nakamoto"),
    ("It might make sense just to get some in case it catches on.",
     "Satoshi Nakamoto"),
    ("Lost coins only make everyone else's coins worth slightly more. Think of it as a donation to everyone.",
     "Satoshi Nakamoto"),
    ("I've been working on a new electronic cash system that's fully peer-to-peer, with no trusted third party.",
     "Satoshi Nakamoto"),
    # Cypherpunk / sovereignty
    ("Privacy is necessary for an open society in the electronic age.",
     "Eric Hughes, A Cypherpunk's Manifesto"),
    ("We must defend our own privacy if we expect to have any.",
     "Eric Hughes, A Cypherpunk's Manifesto"),
    ("Bitcoin is a remarkable cryptographic achievement, and the ability to create something "
     "that is not duplicable in the digital world has enormous value.",
     "Eric Schmidt"),
    ("The computer can be used as a tool to liberate and protect people, rather than to control them.",
     "Hal Finney"),
    ("Running bitcoin.", "Hal Finney"),
    # Sound money
    ("Gold is money. Everything else is credit.", "J.P. Morgan, 1912"),
    ("Money is a guarantee that we may have what we want in the future. "
     "Though we need nothing at the moment, it insures the possibility of satisfying a new desire when it arises.",
     "Aristotle"),
    ("Inflation is taxation without legislation.", "Milton Friedman"),
    ("The curious task of economics is to demonstrate to men how little they really know "
     "about what they imagine they can design.", "F.A. Hayek"),
    ("I don't believe we shall ever have a good money again before we take the thing out of "
     "the hands of government.", "F.A. Hayek"),
    ("There is no subtler, no surer means of overturning the existing basis of society than "
     "to debauch the currency.", "John Maynard Keynes"),
    ("The single most important thing in money is durability.", "Nick Szabo"),
    # HODL culture / wisdom
    ("The stock market is a device for transferring money from the impatient to the patient.",
     "Warren Buffett"),
    ("In the short run, the market is a voting machine, but in the long run, it is a weighing machine.",
     "Benjamin Graham"),
    ("Be fearful when others are greedy, and greedy when others are fearful.", "Warren Buffett"),
    ("The best time to plant a tree was 20 years ago. The second best time is now.",
     "Chinese Proverb"),
    ("Compound interest is the eighth wonder of the world. He who understands it, earns it; "
     "he who doesn't, pays it.", "Attributed to Albert Einstein"),
    # Freedom / conviction
    ("Those who would give up essential liberty, to purchase a little temporary safety, "
     "deserve neither liberty nor safety.", "Benjamin Franklin"),
    ("The only way to deal with an unfree world is to become so absolutely free "
     "that your very existence is an act of rebellion.", "Albert Camus"),
    ("First they ignore you, then they laugh at you, then they fight you, then you win.",
     "Attributed to Mahatma Gandhi"),
    ("In a world of universal deceit, telling the truth is a revolutionary act.",
     "Attributed to George Orwell"),
    # Trace Mayer
    ("Bitcoin is the highest form of property rights mankind has ever invented.",
     "Trace Mayer"),
    ("He who has the gold makes the rules. Bitcoin is the gold of the digital age.",
     "Trace Mayer"),
    # Adam Back
    ("Bitcoin is the one technology that could actually limit the power of government in a meaningful way.",
     "Adam Back"),
    ("Hashcash was my proof of work. Bitcoin was Satoshi's masterpiece.",
     "Adam Back"),
    # American HODL
    ("Bitcoin is the exit. Everything else is a trap.",
     "American HODL"),
    ("Stay toxic. Stay humble. Stack sats. The signal will find you.",
     "American HODL"),
    # Lyn Alden
    ("The system is built on constant growth. Like a shark, it dies if it stops swimming.",
     "Lyn Alden"),
    ("Bitcoin is the best at what it does. And in a world of negative real rates "
     "and a host of currency failures in emerging markets, what it does has utility.",
     "Lyn Alden"),
    # Preston Pysh
    ("The fact that you're watching the explosion in demand for stablecoins "
     "is not them winning. That is them losing epically against Bitcoin.",
     "Preston Pysh"),
    ("To hand off the baton from legacy finance to the future Bitcoin system, "
     "the systems have to match frequency.",
     "Preston Pysh"),
    # Jason Lowery
    ("Bitcoin is not just money. It is a novel form of power projection — "
     "an electro-cyber defense system that converts energy into security.",
     "Jason Lowery, Softwar"),
    ("Proof of work imposes real physical costs on digital actions. "
     "That changes everything about how we think about cybersecurity.",
     "Jason Lowery"),
    # Bruce Fenton
    ("Maximalists might not be that diplomatic, but they're also protecting my bitcoins "
     "because I know they'll never compromise.",
     "Bruce Fenton"),
    ("Bitcoin is the most powerful and important open-source project out there.",
     "Bruce Fenton"),
    # Tuur Demeester
    ("Bitcoin and the cryptocurrencies are the greatest investment opportunity of our day and age.",
     "Tuur Demeester, 2013 (BTC at $100)"),
    ("Bitcoin has the qualities to make for an ideal money — it is designed for the internet, "
     "mobile and fast, with personal privacy, discrete and nonconfiscatable.",
     "Tuur Demeester"),
    # Samson Mow
    ("$1M Bitcoin was already decided when the ETFs were approved. "
     "We're just coasting along now.",
     "Samson Mow"),
    ("We haven't even really started the price run yet. "
     "Faces won't be melted, they will be atomized.",
     "Samson Mow"),
    # Peter Todd
    ("I am Satoshi, as is everyone else.",
     "Peter Todd"),
    ("The point is to make bitcoin the global currency.",
     "Peter Todd"),
    # Jameson Lopp
    ("Bitcoin is a very interesting experiment that if successful could not only "
     "revolutionize money, but revolutionize how we think about governance.",
     "Jameson Lopp"),
    ("Cryptographic protocols are powerful tools that provide asymmetric defense "
     "capabilities to normal people.",
     "Jameson Lopp"),
    # Martti Malmi
    ("Pursuing something greater than yourself brings meaning in life.",
     "Martti Malmi (sirius), Bitcoin's second developer"),
    ("That is regretful, but then again, with the early Bitcoiners we set in motion "
     "something greater than personal gain.",
     "Martti Malmi"),
    # Trace Mayer — Proof of Keys
    ("I started Proof of Keys as a celebration of our monetary sovereignty. "
     "January 3rd — withdraw your coins, hold your own keys, run your own node. "
     "Grow a spine, have some personal power, be free.",
     "Trace Mayer, Proof of Keys Day"),
    # Matt Odell
    ("Stay humble, stack sats.",
     "Matt Odell"),
    ("Privacy is a prerequisite for freedom. If you don't have privacy, "
     "people can use your private information to control you.",
     "Matt Odell"),
    ("My focus will continue to be freedom tech, forever. "
     "Our politicians are corrupt and our institutions are broken — "
     "freedom tech is the only real option we have.",
     "Matt Odell"),
    # Matthew Mezinskis (Porkopolis Economics)
    ("Bitcoin isn't exponential — it's a power curve. "
     "And that's stronger than inflation and money printing.",
     "Matthew Mezinskis, Porkopolis Economics"),
    ("The monetary base is to the core of the entire fiat financial system "
     "as 21 million bitcoins are to the core of the Bitcoin protocol.",
     "Matthew Mezinskis"),
    ("95% of what we see in Bitcoin's price is the power curve of network adoption itself. "
     "It has nothing to do with the Fed or interest rates.",
     "Matthew Mezinskis"),
    ("Not your keys, not your coins.", "Bitcoin Proverb"),
    ("We are all Satoshi.", "Bitcoin Community"),
    ("Fix the money, fix the world.", "Bitcoin Community"),
    ("Tick tock, next block.", "Bitcoin Community"),
    # Peter Schiff (with BTC price at time of quote)
    ("Keep dreaming. Bitcoin is never going to hit $100,000!",
     "Peter Schiff, November 8, 2019 (BTC: $9,273)"),
    ("Bitcoin is digital fool's gold. It's a natural Ponzi scheme "
     "where new buyers keep it afloat.",
     "Peter Schiff, December 2017 (BTC: $17,000)"),
    ("Bitcoin will fall to $1,000. Sell your Bitcoins before it happens.",
     "Peter Schiff, 2018 (BTC: $6,200)"),
    # Bitcoin obituaries — declared dead 470+ times
    ("Bitcoin is probably rat poison squared.",
     "Warren Buffett, May 5, 2018 (BTC: $9,671)"),
    ("Bitcoin is the biggest bubble in human history.",
     "Nouriel Roubini, Bloomberg, February 2, 2018 (BTC: $9,641)"),
    ("Bitcoin is the greatest scam in history.",
     "Forbes, April 24, 2018 (BTC: $8,892)"),
    ("Bitcoin has pretty much failed as a currency.",
     "Bank of England Governor, February 19, 2018 (BTC: $10,825)"),
    ("Bitcoin has failed.",
     "European Central Bank, February 22, 2024 (BTC: $51,305)"),
    # Historical moments
    ("How's this for a disruptive technology? An anonymous Internet group has "
     "created a [working currency](https://news.slashdot.org/story/10/07/11/1747245/bitcoin-releases-version-03) "
     "with no central authority, no banks, and no charge-backs.",
     "Slashdot, July 11, 2010"),
    ("Bitcoin P2P e-cash paper — "
     "[I've been working on a new electronic cash system](https://www.metzdowd.com/pipermail/cryptography/2008-October/014810.html) "
     "that's fully peer-to-peer, with no trusted third party.",
     "Satoshi Nakamoto, Cryptography Mailing List, October 31, 2008"),
    ("I'll pay 10,000 bitcoins for a couple of pizzas.. like maybe 2 large ones "
     "so I have some left over for the next day.",
     "Laszlo Hanyecz, BitcoinTalk, May 18, 2010"),
    ("Bitcoin breaks $1 for the first time on Mt. Gox. "
     "A mass of new users floods the [BitcoinTalk forums](https://bitcointalk.org/index.php?topic=3664.0) "
     "as the media takes notice.",
     "February 9, 2011"),
    ("WikiLeaks has kicked the hornet's nest, and the swarm is headed towards us.",
     "Satoshi Nakamoto, December 11, 2010"),
    ("After a four-year struggle, the SEC approves "
     "[spot Bitcoin ETFs](https://www.sec.gov/newsroom/press-releases/2024-10) "
     "— eleven funds begin trading January 11, 2024.",
     "U.S. Securities and Exchange Commission, January 10, 2024"),
    ("It's Halving Day. Block reward drops from 6.25 to 3.125 BTC. "
     "840,000 blocks mined. Tick tock.",
     "Bitcoin Network, April 19, 2024"),
    ("\"yay accidental hardfork?\" — Luke Dashjr spots a chain split on IRC caused by a "
     "database change (BDB\u2009\u2192\u2009LevelDB). Developers convince miners to "
     "[downgrade to v0.7](https://bitcoin.org/en/alert/2013-03-11-chain-fork), "
     "saving the network.",
     "March 12, 2013"),
    ("\U0001f4a9 [Namecoin](https://bitcointalk.org/index.php?topic=6017.0) launches "
     "as the first ever altcoin — a decentralized DNS built on Bitcoin's code. "
     "The shitcoin era begins.",
     "\U0001f4a9 April 18, 2011"),
    ("\U0001f4a9 [SolidCoin](https://bitcointalk.org/index.php?topic=38453.0) announced: "
     "\"new and improved block chain, secure from pools.\" "
     "Bitcoin miners DoS it into oblivion. It doesn't survive.",
     "\U0001f4a9 August 2011"),
    ("[\"I AM HODLING\"](https://bitcointalk.org/index.php?topic=375643.0) — "
     "BitcoinTalk user GameKyuubi, drunk on whiskey during a crash from $1,100, "
     "misspells \"holding\" and accidentally creates the most enduring meme in Bitcoin.",
     "December 18, 2013"),
    # NVK (Rodolfo Novak)
    ("Bitcoin already won. Everyone is just catching up.",
     "NVK (Rodolfo Novak), Coinkite"),
    ("We all have reasonably similar needs for keeping bitcoins secure. "
     "No compromises in privacy and security.",
     "NVK"),
    # bg002h
    ("Hardforks aren't that hard. It's getting others to use them that's hard.",
     "bg002h, BitcoinTalk (member since July 2010)"),
    ("I stopped mining cause it was gonna take roughly a week to mine the next block of 50 bitcoins "
     "— and I never even tried GPU mining.",
     "bg002h, BitcoinTalk"),
    ("It's fun to look back at this and think \"I was a small part in making it happen.\"",
     "bg002h, BitcoinTalk"),
    ("Continue converting a small amount of dollars every 2 weeks... "
     "the only thing that'll be different is that I'll get more BTC.",
     "bg002h, BitcoinTalk"),
    ("I'll let you know my exit strategy in 10 years.",
     "bg002h, BitcoinTalk"),
    ("Trust? Why? You have nothing to gain and only stand to lose.",
     "bg002h, BitcoinTalk"),
    # Ross Ulbricht
    ("Bitcoin's power comes from the fact that any one of us can mine, "
     "any one of us can generate addresses, any one of us can send bitcoin to anyone else. "
     "With Bitcoin, we are all free.",
     "Ross Ulbricht"),
    ("I made Silk Road because I thought I was furthering the things I cared about: "
     "freedom, privacy, equality. I was impatient. I rushed ahead with my first idea.",
     "Ross Ulbricht"),
    ("Stay united. Those that oppose decentralization and freedom love it when we're divided. "
     "So long as we can agree that we deserve freedom, the future is ours.",
     "Ross Ulbricht, Bitcoin 2025"),
    # Donald Trump
    ("If you vote for me, on Day 1, I will commute the sentence of Ross Ulbricht. "
     "He's already served 11 years. We're gonna get him home.",
     "Donald Trump, Libertarian National Convention, May 2024"),
    ("I will ensure that the future of crypto and Bitcoin will be made in the USA. "
     "I will support the right to self-custody and never allow the creation of a CBDC.",
     "Donald Trump, Bitcoin 2024 Conference"),
    # Charlie Shrem
    ("The first person to walk through the door always gets shot, "
     "and then everyone else can come through.",
     "Charlie Shrem"),
    ("Bitcoin is cash with wings.",
     "Charlie Shrem"),
    # Ryan Selkis
    ("Bitcoin is the least risky crypto asset and has the fewest headwinds. "
     "It will be the first to cross the chasm to mainstream adoption.",
     "Ryan Selkis (TwoBitIdiot), Messari"),
    ("I started blogging as the Two-Bit Idiot in 2013 and broke the Mt. Gox story. "
     "Transparency is the only way this industry survives.",
     "Ryan Selkis"),
    ("Charlie Shrem, CEO of BitInstant — which processed 30% of all Bitcoin transactions — "
     "is [sentenced to two years](https://www.justice.gov/usao-sdny/pr/"
     "former-ceo-bitcoin-exchange-company-sentenced-manhattan-federal-court-two-years-prison) "
     "for aiding unlicensed money transmission tied to Silk Road. Bitcoin's first felon.",
     "December 19, 2014"),
    ("[This is gentlemen.](https://bitcointalk.org/index.php?topic=855789.0) "
     "An overexcited Bitcoiner meant to type \"this is it, gentlemen\" during a price rally "
     "but left out a word — and accidentally created a battle cry.",
     "November 11, 2014"),
    ("Trendon Shavers (pirateat40) promises 7% weekly returns on "
     "[Bitcoin Savings & Trust](https://bitcointalk.org/index.php?topic=50822.0). "
     "It's a Ponzi scheme. He's [convicted](https://www.justice.gov/usao-sdny/pr/"
     "texas-man-sentenced-operating-bitcoin-ponzi-scheme) — "
     "Bitcoin's first major fraud.",
     "2011\u20132016"),
    ("Slush and Stick announce [Trezor](https://bitcointalk.org/index.php?topic=122438.0) "
     "on BitcoinTalk — the world's first Bitcoin hardware wallet. "
     "Self-custody just got a whole lot easier.",
     "2013"),
    ("NVK launches [Coldcard](https://bitcointalk.org/index.php?topic=5033058.0) "
     "— a no-compromise, air-gapped Bitcoin hardware wallet. "
     "The cypherpunk DIY ethos in a signing device.",
     "2018"),
    # Bitcoin documentaries
    ("Required viewing: [Banking on Bitcoin](https://www.imdb.com/title/tt5033790/) (2016) "
     "— the most disruptive invention since the Internet, "
     "and the ideological battle over its future.",
     "Documentary"),
    ("Required viewing: [The Rise and Rise of Bitcoin](https://www.imdb.com/title/tt2821314/) (2014) "
     "— a programmer's journey into the rabbit hole, "
     "featuring Vitalik Buterin and Julian Assange.",
     "Documentary"),
    ("Required viewing: [Bitcoin: The End of Money as We Know It]"
     "(https://www.imdb.com/title/tt4654844/) (2015) "
     "— how Bitcoin challenges everything we thought we knew about currency.",
     "Documentary"),
    ("Required viewing: [The Great Reset and the Rise of Bitcoin]"
     "(https://www.imdb.com/title/tt17999542/) (2022) "
     "— how the monetary system broke and why Bitcoin is the fix.",
     "Documentary"),
]

def _splash_quote_index():
    """Deterministic pseudo-random quote index: rotates every 6 hours, same for all users."""
    import random as _rnd
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    epoch_6h = int(now.timestamp()) // (6 * 3600)
    # Seed with epoch so all users get the same quote, but order looks random
    indices = list(range(len(_SPLASH_QUOTES)))
    _rnd.Random(epoch_6h).shuffle(indices)
    return indices[0]

_SPLASH_IDX = _splash_quote_index()
_SPLASH_Q, _SPLASH_A = _SPLASH_QUOTES[_SPLASH_IDX]

# Genesis block quote always shown first in splash modal
_GENESIS_QUOTE = ("The Times 03/Jan/2009 Chancellor on brink of second bailout for banks.",
                  "Bitcoin Genesis Block")
# Build JSON for clientside quote cycling (genesis first, then rest in shuffled order)
import json as _json
import random as _rnd
_shuffled = list(_SPLASH_QUOTES)
_rnd.Random(42).shuffle(_shuffled)
_SPLASH_QUOTES_JS = _json.dumps(
    [list(_GENESIS_QUOTE)] + [list(q) for q in _shuffled]
)

_app_ctx.app.layout = dbc.Container([
    dcc.Interval(id="price-interval", interval=_PRICE_INTERVAL_MS, n_intervals=0),
    dcc.Store(id="btc-price-store", storage_type="memory", data=None),
    dcc.Store(id="splash-ts-store", storage_type="local", data=None),
    dcc.Store(id="lots-store", storage_type="local", data=[]),
    dcc.Store(id="lots-export-dummy"),
    dcc.Store(id="wm-b64-store", storage_type="memory", data=_LOGO_B64_ALL),
    # MC simulation result stores (localStorage — survives page reloads)
    dcc.Store(id="dca-mc-results", storage_type="memory", data=None),
    dcc.Store(id="ret-mc-results", storage_type="memory", data=None),
    dcc.Store(id="hm-mc-results",  storage_type="memory", data=None),
    dcc.Store(id="sc-mc-results",  storage_type="memory", data=None),
    # Trigger stores: incremented on MC upload to force figure redraw
    dcc.Store(id="dca-mc-loaded", storage_type="memory", data=0),
    dcc.Store(id="ret-mc-loaded", storage_type="memory", data=0),
    dcc.Store(id="hm-mc-loaded",  storage_type="memory", data=0),
    dcc.Store(id="sc-mc-loaded",  storage_type="memory", data=0),
    dcc.Store(id="dca-mc-dl-dummy"),
    dcc.Store(id="ret-mc-dl-dummy"),
    dcc.Store(id="hm-mc-dl-dummy"),
    dcc.Store(id="sc-mc-dl-dummy"),
    # ── MC payment stores + polling ──────────────────────────────────────
    dcc.Store(id="mc-pay-invoice", storage_type="memory", data=None),
    dcc.Store(id="mc-pay-token",   storage_type="memory", data=None),
    dcc.Store(id="mc-pay-trigger", storage_type="memory", data=0),
    dcc.Interval(id="mc-pay-poll", interval=_MC_POLL_INTERVAL_MS, disabled=True,
                 max_intervals=_MC_POLL_MAX, n_intervals=0),
    # MC save prompt modal — shown after cache miss (new simulation)
    dcc.Store(id="mc-save-tab", storage_type="memory", data=None),
    dcc.Store(id="mc-save-modal-dummy"),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Monte Carlo Simulation Complete")),
        dbc.ModalBody("This simulation took a while to compute. "
                      "Save it now so you don't have to wait again."),
        dbc.ModalFooter([
            dbc.Button("\u2b07 Save simulation", id="mc-save-modal-dl",
                       color="warning", className="me-2"),
            dbc.Button("Dismiss", id="mc-save-modal-dismiss",
                       color="secondary"),
        ]),
    ], id="mc-save-modal", is_open=False, backdrop="static", centered=True),
    # ── MC payment modal ─────────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Pay to Run Simulation")),
        dbc.ModalBody([
            html.Div(id="mc-pay-info",
                     style={"fontSize": "13px", "marginBottom": "10px"}),
            # Iframe container — shown for onion users (Tor Browser)
            html.Div(id="mc-pay-iframe-wrap", style={"display": "none"}, children=[
                html.Iframe(id="mc-pay-iframe",
                            style={"width": "100%", "height": "420px",
                                   "border": "none", "borderRadius": "8px"},
                            src="about:blank"),
            ]),
            # Payment details — shown for clearnet users
            html.Div(id="mc-pay-details", style={"display": "none"}, children=[
                dbc.ButtonGroup([
                    dbc.Button([html.Span("\u26a1 "), "Lightning"],
                               id="mc-pay-ln-btn", color="warning",
                               outline=False, size="sm", active=True),
                    dbc.Button([html.Span("\u20bf "), "On-chain"],
                               id="mc-pay-chain-btn", color="warning",
                               outline=True, size="sm", active=False),
                ], size="sm", className="w-100 mb-3"),
                html.Div(
                    html.Img(id="mc-pay-qr",
                             style={"maxWidth": "220px", "width": "100%"}),
                    style={"textAlign": "center", "margin": "10px 0"},
                ),
                html.Div([
                    html.Code(id="mc-pay-dest",
                              style={"fontSize": "10px", "wordBreak": "break-all",
                                     "display": "block", "padding": "8px",
                                     "backgroundColor": "#f5f5f5",
                                     "borderRadius": "4px",
                                     "fontFamily": "monospace",
                                     "userSelect": "all", "lineHeight": "1.4",
                                     "maxHeight": "80px", "overflow": "auto"}),
                    dbc.Button("Copy", id="mc-pay-copy-btn", size="sm",
                               color="secondary", outline=True,
                               className="mt-1",
                               style={"fontSize": "11px"}),
                ]),
                html.Div(id="mc-pay-amount-info",
                         style={"fontSize": "12px", "color": "#555",
                                "marginTop": "6px", "textAlign": "center"}),
            ]),
            dcc.Store(id="mc-pay-methods", storage_type="memory", data=None),
            html.Div(id="mc-pay-status",
                     style={"fontSize": "12px", "color": "#555",
                            "textAlign": "center", "marginTop": "8px"}),
        ]),
        dbc.ModalFooter(
            dbc.Button("Cancel", id="mc-pay-cancel", color="secondary"),
        ),
    ], id="mc-pay-modal", is_open=False, backdrop="static", centered=True,
       size="lg"),
    # ── Quant-tier cost warning modal (>50k sats) ──
    dbc.Modal([
        dbc.ModalHeader(html.Span(
            "\u2694\ufe0f Entering Quant Territory \u2694\ufe0f",
            style={**_QUANT_FONT, "fontWeight": "bold", "fontSize": "20px",
                   "textAlign": "center", "width": "100%"})),
        dbc.ModalBody([
            html.P("I see your model costs have left the realm of mere mortals "
                   "and entered the realm of Wall St. Quants.",
                   style={**_QUANT_FONT, "fontSize": "15px", "lineHeight": "1.6",
                          "textAlign": "center"}),
            html.P(id="mc-quant-cost-info",
                   style={"fontFamily": "'Courier New', Courier, monospace",
                          "fontSize": "14px", "color": "#555", "textAlign": "center",
                          "letterSpacing": "1px"}),
            html.P("Are you sure you want to continue?",
                   style={**_QUANT_FONT, "fontWeight": "bold",
                          "fontSize": "15px", "marginTop": "10px",
                          "textAlign": "center"}),
            html.P(id="mc-quant-onchain-note",
                   style={"fontStyle": "italic", "fontSize": "12px",
                          "textAlign": "center", "color": "#555"}),
        ]),
        dbc.ModalFooter([
            dbc.Button("Take me back", id="mc-quant-cancel",
                       color="secondary", className="me-auto",
                       style=_QUANT_FONT),
            dbc.Button("\u26a1 Proceed, I am Sir Baller", id="mc-quant-proceed",
                       color="warning",
                       style={**_QUANT_FONT, "fontWeight": "600"}),
        ]),
    ], id="mc-quant-modal", is_open=False, centered=True),
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="snapshot-lots",     storage_type="memory", data=None),
    dcc.Store(id="effective-lots",    storage_type="memory", data=[]),
    dcc.Store(id="link-history",      storage_type="local",  data=[]),
    dcc.Store(id="loaded-hash-store", storage_type="memory"),
    dcc.Store(id="journey-store",    storage_type="local",  data=None),
    # ── Splash quote modal ────────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalBody([
            html.Div([
                html.Img(src="/assets/quantoshi_logo_nav.png", height="50px",
                         style={"opacity":"0.9"}),
                html.Div([html.Span("Q", className="brand-q"),
                          "uantoshi"],
                         style={"fontFamily":"Palatino Linotype, Palatino, Book Antiqua, serif",
                                "fontSize":"1.5rem", "fontWeight":"700",
                                "color":"#2c3e50", "marginLeft":"10px"}),
            ], style={"display":"flex", "alignItems":"center",
                      "justifyContent":"center", "marginBottom":"20px"}),
            html.Div([
                dcc.Markdown(id="splash-quote-text",
                             style={"fontSize":"16px", "fontStyle":"italic",
                                    "color":"#2c3e50", "lineHeight":"1.5",
                                    "textAlign":"center", "marginBottom":"10px"},
                             link_target="_blank"),
                html.Div(id="splash-quote-attr",
                         style={"fontSize":"13px", "color":"#666",
                                "textAlign":"center"}),
            ], style={"padding":"10px 20px"}),
            html.Div(id="journey-stats",
                     style={"textAlign":"center", "fontSize":"12px",
                            "color":"#888", "marginTop":"16px",
                            "lineHeight":"1.7", "display":"none"}),
            html.Div([
                dbc.Button("Let's go", id="splash-dismiss", size="lg",
                           className="btn-share-accent",
                           style={"padding":"8px 40px", "fontSize":"15px",
                                  "borderRadius":"8px"}),
                html.Span(
                    dbc.Button("More quotes", id="splash-next", size="sm",
                               outline=True, color="secondary",
                               style={"marginLeft":"12px", "fontSize":"13px",
                                      "borderRadius":"8px"}),
                    id="splash-next-wrap", style={"display":"none"},
                ),
            ], style={"textAlign":"center", "marginTop":"24px"}),
        ], style={"padding":"30px 20px 24px"}),
    ], id="splash-modal", is_open=False, centered=True, backdrop="static",
       className="splash-modal"),
    dbc.Navbar(
        dbc.Container([
            # ── Desktop navbar (hidden on mobile portrait) ────────────────
            html.Div([
                html.Div([
                    # Left: logo + brand + ticker
                    html.Div([
                        html.Img(src="/assets/quantoshi_logo_nav.png", height="40px",
                                 id="logo-easter-egg", className="logo-glow",
                                 style={"cursor":"pointer"}),
                        dbc.NavbarBrand([html.Span("Q", className="brand-q"),
                                         html.Span("uantoshi", className="brand-uantoshi")],
                                        className="ms-2 fw-bold fs-2 brand-name",
                                        style={"fontFamily":"Palatino Linotype, Palatino, Book Antiqua, serif"}),
                        html.Div(id="price-ticker",
                                 style={"fontSize":"19px", "fontWeight":"600",
                                        "color":"rgba(255,255,255,0.9)",
                                        "whiteSpace":"nowrap", "fontFamily":"monospace",
                                        "marginLeft":"14px"}),
                    ], style={"display":"flex", "alignItems":"center"}),
                    # Right: collapsible drawer (stacked vertically) + toggle
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Span("Stay dark, Anon ▶ ",
                                          style={"fontSize":"9px", "color":"rgba(255,255,255,0.4)",
                                                 "whiteSpace":"nowrap"}),
                                html.A(
                                    "🧅 Tor onion",
                                    href="http://u5dprelc4ti7xoczb5sbtye6qidlji2l6psmkx35anvxgjyqrkmu32ad.onion",
                                    target="_blank",
                                    rel="noopener noreferrer",
                                    className="text-decoration-none",
                                    style={"fontSize":"15px", "color":"rgba(255,255,255,0.75)"},
                                ),
                            ], style={"display":"flex", "alignItems":"center",
                                      "justifyContent":"flex-end"}),
                            html.Div([
                                html.Span("Cooler than you think ▶ ",
                                          style={"fontSize":"9px", "color":"rgba(255,255,255,0.4)",
                                                 "whiteSpace":"nowrap"}),
                                dbc.Button("📸 Share", id="share-btn", size="sm",
                                           className="btn-share-accent"),
                            ], style={"display":"flex", "alignItems":"center",
                                      "justifyContent":"flex-end",
                                      "marginTop":"2px"}),
                        ], id="desktop-nav-drawer", className="desktop-nav-drawer"),
                        html.Div("⋯", id="desktop-nav-toggle",
                                 className="desktop-nav-toggle desktop-nav-toggle-hidden",
                                 style={"color":"rgba(255,255,255,0.5)",
                                        "fontSize":"18px", "cursor":"pointer",
                                        "letterSpacing":"2px", "padding":"4px 8px"}),
                    ], style={"display":"flex", "alignItems":"center",
                              "marginLeft":"auto"}),
                ], style={"display":"flex", "alignItems":"center",
                          "justifyContent":"space-between", "width":"100%"}),
            ], className="d-none d-md-block w-100"),
            # ── Mobile navbar (hidden on desktop) ─────────────────────────
            html.Div([
                # Row 1: logo+brand left, [toggle when collapsed], ticker right
                html.Div([
                    html.Div([
                        html.Img(src="/assets/quantoshi_logo_nav.png", height="34px",
                                 id="logo-easter-egg-mobile", className="logo-glow",
                                 style={"cursor":"pointer"}),
                        html.Span([html.Span("Q", className="brand-q"),
                                   html.Span("uantoshi", className="brand-uantoshi")],
                                  className="fw-bold ms-2 brand-name",
                                  style={"fontFamily":"Palatino Linotype, Palatino, Book Antiqua, serif",
                                         "fontSize":"1.75rem", "color":"#fff"}),
                    ], style={"display":"flex", "alignItems":"center"}),
                    html.Div("⋯", id="mobile-nav-toggle",
                             className="mobile-nav-toggle mobile-nav-toggle-hidden",
                             style={"color":"rgba(255,255,255,0.5)",
                                    "fontSize":"18px", "cursor":"pointer",
                                    "lineHeight":"1", "letterSpacing":"2px",
                                    "padding":"4px 8px"}),
                    html.Div(id="price-ticker-mobile",
                             style={"fontSize":"18px", "fontWeight":"700",
                                    "color":"rgba(255,255,255,0.95)",
                                    "whiteSpace":"nowrap", "fontFamily":"monospace"}),
                ], style={"display":"flex", "alignItems":"center",
                          "justifyContent":"space-between", "width":"100%"}),
                # Row 2: collapsible drawer — full content, auto-hides after 3s
                html.Div([
                    html.Hr(style={"borderColor":"rgba(255,255,255,0.12)",
                                   "margin":"3px 0"}),
                    html.Div([
                        html.A([
                            html.Span("🧅", style={"fontSize":"20px",
                                                    "lineHeight":"1"}),
                            html.Span(" ◂ Stay dark, Anon",
                                      style={"fontSize":"11px",
                                             "color":"rgba(255,255,255,0.5)",
                                             "marginLeft":"5px"}),
                        ], href="http://u5dprelc4ti7xoczb5sbtye6qidlji2l6psmkx35anvxgjyqrkmu32ad.onion",
                           target="_blank", rel="noopener noreferrer",
                           className="text-decoration-none",
                           style={"display":"flex", "alignItems":"center"}),
                        dbc.Button("📸 Share", id="share-btn-mobile", size="sm",
                                   className="btn-share-accent",
                                   style={"fontSize":"10px", "padding":"2px 8px"}),
                    ], style={"display":"flex", "alignItems":"center",
                              "justifyContent":"space-between", "width":"100%"}),
                ], id="mobile-nav-drawer", className="mobile-nav-drawer"),
            ], className="d-md-none w-100"),
        ], fluid=True),
        color="#2c3e50", dark=True, className="mb-0 py-1 mt-1 navbar-parallax",
        id="main-navbar",
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
                value="tab",
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
                       size="sm", className="mt-2 mb-3 w-100 btn-generate-accent"),
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
    # ── Footer: block height + halving countdown ──────────────────────────
    html.Div([
        html.Span(id="footer-block-height",
                  style={"marginRight": "16px"}),
        html.Span(id="footer-halving-countdown"),
    ], className="site-footer",
       style={"textAlign": "center", "fontSize": "11px",
              "color": "rgba(0,0,0,0.35)", "padding": "10px 0 14px",
              "fontFamily": "monospace", "letterSpacing": "0.5px"}),
], fluid=True, className="px-2 py-1")

