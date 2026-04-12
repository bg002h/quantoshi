"""Tab 2 — CAGR Heatmap layout."""

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from layout.common import (_tab_hints, _section_card, _lbl, _row, _export_row,
                            _STYLE_HIDDEN, _STYLE_HINT, _STYLE_GRAPH_H,
                            _STYLE_COLOR_H, _BTC_ORANGE,
                            _CB_MARGIN, _Q_HINT_BASE, _GEAR_STYLE, _MUTED_STYLE,
                            _q_options, _palette_selector)
from layout.mc_controls import _mc_controls
from tab_defaults import HEATMAP
from colors import (NEAR_BLACK, DIM_TEXT, SPINE_COLOR_FALLBACK,
                    PROGRESS_TRACK, FALLBACK_MODEL_GRAY)


# ── Heatmap pill set ───────────────────────────────────────────────────────
# Source of truth for the heatmap model-selector pill bar. Both this layout
# module and callbacks/routing.py (for _hm_pill_click / _hm_pill_sync Inputs
# and Outputs) import from here so the rendered pill ids and the callback
# binding set stay in sync. Any divergence causes Dash to raise
# ReferenceError on pill click and silently abort the callback.
_HM_PILL_MODELS_BASE = [
    "bub", "pl", "lppl", "hybppl",
    "pca", "grdy", "eppl", "gomp", "bpl",
]

# Short display labels rendered inside each pill button.
_HM_PILL_LABELS = {
    "bub": "BM", "pl": "PL", "lppl": "LPPL",
    "linppl": "LinPPL", "hybppl": "HybPPL",
    "hyb2l": "H2L", "hyb2c": "H2C", "hyb2b": "H2B", "hyb4d": "H4D",
    "pca": "PCA", "grdy": "Grdy", "eppl": "EPPL",
    "gomp": "Gomp", "bpl": "BPL",
    "ef": "EF", "u1": "U\u2081", "mc": "MC",
}


def _hm_pill_models():
    """Return the full ordered pill-model list including conditional entries
    (ef / u1 / mc). Mirrors the logic in callbacks/routing.py — any change
    here must be matched there (or routing.py can import this helper)."""
    keys = list(_HM_PILL_MODELS_BASE)
    if "ef" in _app_ctx.PRICE_MODELS:
        keys.append("ef")
    if "u1" in _app_ctx.PRICE_MODELS:
        keys.append("u1")
    if _app_ctx._HAS_MARKOV:
        keys.append("mc")
    return keys


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
                            value=[yr_now, yr_now + 10], step=1,
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
                    {"label": "Red \u2192 White \u2192 Green",  "value": "rwg"},
                    {"label": "Red \u2192 Black \u2192 Green",  "value": "rbg"},
                    {"label": "Blue \u2192 White \u2192 Orange", "value": "bwo"},
                    {"label": "Monochrome (gray)",               "value": "mono"},
                ],
                value=HEATMAP["hm_palette"], clearable=False),
            _row(
                html.Div([_lbl("Lo"), dbc.Input(id="hm-c-lo", type="color",
                           value=HEATMAP["c_lo"], style=_STYLE_COLOR_H)]),
                html.Div([_lbl("Mid1"), dbc.Input(id="hm-c-mid1", type="color",
                           value=HEATMAP["c_mid1"], style=_STYLE_COLOR_H)]),
                html.Div([_lbl("Mid2"), dbc.Input(id="hm-c-mid2", type="color",
                           value=HEATMAP["c_mid2"], style=_STYLE_COLOR_H)]),
                html.Div([_lbl("Hi"), dbc.Input(id="hm-c-hi", type="color",
                           value=HEATMAP["c_hi"], style=_STYLE_COLOR_H)]),
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
        _palette_selector("hm"),
    ])


def _hm_pill_bar():
    """Model-selector pill bar — standardized set (BM, PL, LPPL master,
    LinPPL, HybPPL, EF, U1, MC). LPPL family variants collapse into the
    single LPPL master pill; the actual flavor is chosen via the global
    LPPL config modal."""
    mc = _app_ctx.PALETTES["default"]["model_colors"]

    def _pill_label(key, display_name):
        return html.Span([
            html.Span(" ", style={
                "display": "inline-block", "width": "8px", "height": "8px",
                "borderRadius": "2px", "verticalAlign": "middle",
                "marginRight": "4px",
                "backgroundColor": mc.get(key, FALLBACK_MODEL_GRAY),
            }),
            display_name,
        ])

    # Iterate the full pill-model list (base + conditional ef/u1/mc) so the
    # rendered buttons exactly match the callback Input/Output set declared
    # in callbacks/routing.py. Any model listed but missing here triggers a
    # Dash "nonexistent object" ReferenceError and aborts pill clicks
    # entirely — see git history around 2026-04-11 for the regression.
    buttons = []
    for i, key in enumerate(_hm_pill_models()):
        label_text = _HM_PILL_LABELS.get(key, key)
        if key == "mc":
            buttons.append(
                dbc.Button(label_text, id=f"hm-pill-{key}",
                           outline=True, color="warning", size="sm"),
            )
        else:
            buttons.append(
                dbc.Button(_pill_label(key, label_text),
                           id=f"hm-pill-{key}",
                           color="primary", size="sm",
                           outline=(i != 0)),
            )

    pill_bar = html.Div([
        dbc.ButtonGroup(buttons, size="sm"),
    ], className="mb-1 text-center")

    status_row = html.Div(
        id="hm-active-family-row",
        style={"display": "none",           # hidden until gated clientside
               "alignItems": "center",
               "gap": "4px",
               "marginTop": "6px",
               "fontSize": "11px"},
        children=[
            html.Span("Active: ", style={"color": FALLBACK_MODEL_GRAY}),
            html.Span(id="hm-active-family-label",
                      style={"fontWeight": "600"}),
            html.Span(" \u00b7 (", style=_MUTED_STYLE),
            html.Span(id="hm-active-family-summary-inline",
                      children="", style=_MUTED_STYLE),
            html.Span(") ", style=_MUTED_STYLE),
            html.Span("\u2699\uFE0F",
                      id="hm-active-family-gear", n_clicks=0,
                      style=_GEAR_STYLE,
                      title="Configure active model"),
        ],
    )

    return html.Div([pill_bar, status_row])


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
                    dcc.Graph(id="heatmap-graph", style={"height": "100vh"},
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
