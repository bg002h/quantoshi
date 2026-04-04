"""Layout package — tab controls, main layout assembly, splash modal.

Re-exports symbols consumed by callbacks.py and app.py.
"""
#
# Sections:
#   Re-exports ........................ ~1-20
#   App layout assembly ............... ~25-end

import json

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from figures import _LOGO_B64_ALL
from snapshot import _SNAPSHOT_CONTROLS

# ── Re-exports (consumed by callbacks.py, app.py, etc.) ─────────────────────
from layout.common import (_STYLE_HIDDEN, _STYLE_HINT, _STYLE_GRAPH_H,
                            _STYLE_COLOR_H, _CB_MARGIN, _freq_warning_modal)
from layout.mc_controls import (_bold_opts, _regime_options,
                                 _MC_CACHED_START_YRS, _MC_CACHED_YEARS,
                                 _MC_CACHED_ENTRY_QS, _MC_CACHED_WD, _MC_CACHED_INFL,
                                 _MC_PRICE_LIVE,
                                 _MC_ENTRY_Q_OPTIONS, _MC_ENTRY_Q_OPTIONS_ADV,
                                 _QUANT_FONT)
from layout.faq import _FAQ
from layout.splash import (_SPLASH_QUOTES_JS,
                            _SPLASH_IDX, _SPLASH_Q, _SPLASH_A,
                            _GENESIS_QUOTE)

# ── Tab builders (used only in layout assembly below) ────────────────────────
from layout.bubble import _bubble_tab
from layout.heatmap import _heatmap_tab
from layout.sim_tabs import _dca_tab, _retire_tab
from layout.supercharge import _supercharge_tab
from layout.stack import _stack_tracker_tab
from layout.model_info import _model_info_tab
from layout.citadel import _citadel_tab
from layout.faq import _faq_tab
from layout.common import _export_row, _BTC_ORANGE

# ── Constants used in layout assembly ────────────────────────────────────────
_PRICE_INTERVAL_MS = 20 * 60 * 1000   # live price ticker refresh (20 minutes)
_MC_POLL_INTERVAL_MS = 3000            # MC payment status poll interval (3 seconds)
_MC_POLL_MAX = 300                     # max poll intervals (300 × 3s = 15 min timeout)

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

_PATH_TO_TAB = {
    "/1": "bubble", "/2": "heatmap", "/3": "dca", "/4": "retire",
    "/5": "supercharge", "/6": "stack", "/7": "model_info", "/8": "faq",
    "/9": "citadel",
}

_TAB_TO_GRAPH = {
    "bubble": "bubble-graph", "heatmap": "heatmap-graph",
    "dca": "dca-graph", "retire": "retire-graph",
    "supercharge": "supercharge-graph", "citadel": "citadel-graph",
}
_TAB_TO_FIG_FN = {}  # populated lazily below


def _get_initial_figure(tab):
    """Get pre-computed figure for a tab from L1 cache. Returns fig or None."""
    if not _TAB_TO_FIG_FN:
        # Lazy init — imports only available after app.py finishes setup
        try:
            from utils import (_get_bubble_fig, _get_heatmap_fig, _get_dca_fig,
                               _get_retire_fig, _get_supercharge_fig, _get_citadel_fig)
            from tab_defaults import (bubble_defaults, heatmap_defaults, dca_defaults,
                                      retire_defaults, supercharge_defaults, citadel_defaults)
            _TAB_TO_FIG_FN["bubble"] = (_get_bubble_fig, bubble_defaults)
            _TAB_TO_FIG_FN["heatmap"] = (_get_heatmap_fig, heatmap_defaults)
            _TAB_TO_FIG_FN["dca"] = (_get_dca_fig, dca_defaults)
            _TAB_TO_FIG_FN["retire"] = (_get_retire_fig, retire_defaults)
            _TAB_TO_FIG_FN["supercharge"] = (_get_supercharge_fig, supercharge_defaults)
            _TAB_TO_FIG_FN["citadel"] = (_get_citadel_fig, citadel_defaults)
        except Exception:
            return None

    entry = _TAB_TO_FIG_FN.get(tab)
    if not entry:
        return None
    get_fn, defaults_fn = entry
    try:
        result = get_fn(defaults_fn())
        return result[0] if isinstance(result, tuple) else result
    except Exception:
        return None


def _serve_layout():
    """Layout function — called on each page request.

    Reads the URL path to set the initial active tab, so visiting /9
    builds the layout with active_tab='citadel' from the start.
    Injects the pre-computed figure for the active tab into the HTML
    so it renders before any callbacks fire.
    """
    import flask
    path = flask.request.path if flask.has_request_context() else "/"
    clean = path.rstrip("/").split(".")[0] or "/"
    initial_tab = _PATH_TO_TAB.get(clean, "bubble")
    layout = _build_layout(initial_tab)

    # Inject pre-computed figures for ALL chart tabs (L1 cache hit, ~0ms each).
    # This prevents the flash/resize when switching to a new tab.
    from layout.common import _inject_initial_figure
    for tab, gid in _TAB_TO_GRAPH.items():
        fig = _get_initial_figure(tab)
        if fig:
            _inject_initial_figure(layout, gid, fig)

    return layout

_app_ctx.app.layout = _serve_layout


def _build_layout(initial_tab="bubble"):
    return dbc.Container([
    _freq_warning_modal(),
    dcc.Interval(id="price-interval", interval=_PRICE_INTERVAL_MS, n_intervals=0),
    dcc.Store(id="btc-price-store", storage_type="memory", data=None),
    dcc.Store(id="model-percentiles-store", storage_type="memory", data=None),
    dcc.Store(id="ticker-model-idx", storage_type="memory", data=0),
    dcc.Store(id="user-model-store", storage_type="memory", data=None),
    dcc.Store(id="um-clicked-point", storage_type="memory", data=None),
    dcc.Store(id="viewport-width", storage_type="memory", data=1200),
    dcc.Store(id="snapshot-state-store", storage_type="memory", data=None),
    dcc.Store(id="ticker-mode-store", storage_type="local", data="usd"),
    dcc.Store(id="splash-ts-store", storage_type="local", data=None),
    dcc.Store(id="lots-store", storage_type="local", data=[]),
    dcc.Store(id="lots-export-dummy"),
    dcc.Store(id="wm-b64-store", storage_type="memory", data=_LOGO_B64_ALL),
    # Per-tab render triggers — start at 1 (figures already injected at layout time).
    # Clientside trigger only fires for cur=0 (unvisited), so no callback fires
    # until the user changes a control. Double-click tab forces reload.
    *[dcc.Store(id=f"{tab}-first-render", storage_type="memory", data=1)
      for tab in ("bubble", "heatmap", "dca", "retire", "supercharge", "citadel")],
    # MC per-tab stores (results, unblocked cache, loaded trigger, download dummy)
    *[dcc.Store(id=f"{pfx}-mc-results", storage_type="memory", data=None)
      for pfx in ("dca", "ret", "hm", "sc", "cp")],
    *[dcc.Store(id=f"{pfx}-mc-unblocked", storage_type="memory", data=None)
      for pfx in ("dca", "ret", "hm", "sc", "cp")],
    *[dcc.Store(id=f"{pfx}-mc-loaded", storage_type="memory", data=0)
      for pfx in ("dca", "ret", "hm", "sc", "cp")],
    *[dcc.Store(id=f"{pfx}-mc-dl-dummy") for pfx in ("dca", "ret", "hm", "sc", "cp")],
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
            html.Div(id="mc-pay-iframe-wrap", style=_STYLE_HIDDEN, children=[
                html.Iframe(id="mc-pay-iframe",
                            style={"width": "100%", "height": "420px",
                                   "border": "none", "borderRadius": "8px"},
                            src="about:blank"),
            ]),
            # Payment details — shown for clearnet users
            html.Div(id="mc-pay-details", style=_STYLE_HIDDEN, children=[
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
    html.Div(id="copy-toast-container"),
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="snapshot-lots",     storage_type="memory", data=None),
    dcc.Store(id="effective-lots",    storage_type="memory", data=[]),
    dcc.Store(id="link-history",      storage_type="local",  data=[]),
    dcc.Store(id="loaded-hash-store", storage_type="memory"),
    dcc.Store(id="palette-store",    storage_type="local",  data="default"),
    # Model warning dismissal (localStorage — persists across sessions, no cookies)
    dcc.Store(id="model-warn-dismissed", storage_type="local", data={}),
    dcc.ConfirmDialog(id="s2f-warn-dialog",
        message="Remember: all models are wrong, some are useful, and some are no longer useful."),
    dcc.ConfirmDialog(id="exp-warn-dialog",
        message="WARNING: this model is not useful at extremes of past and future."),
    dcc.ConfirmDialog(id="u1-warn-dialog",
        message="User Model (U\u2081): Right-click the chart to set P1 and P2 points. "
                "A power law is fitted through both points to generate quantile bands. "
                "The model is session-only \u2014 it disappears on page refresh."),
    dcc.ConfirmDialog(id="lp4-warn-dialog",
        message="LPPL\u2084 is probably NOT that smart. "
                "The 4th frequency (ω\u224813 or 17, depending on constraints) is likely an "
                "intermodulation artifact of the 3 robust oscillations, not a real structural "
                "feature. See Model Info \u2192 LPPL Weighting & Regime Shifts (/7.6) for why. "
                "Use LPPL\u2083 if you want a physically defensible fit."),
    dcc.Store(id="lp4-warn-prev", storage_type="memory", data=[3]),
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
            html.Div(id="knight-welcome",
                     style={"fontSize":"13px", "color":"#b8860b",
                            "textAlign":"center", "marginTop":"8px"}),
            html.Div(
                html.A("\u2694\ufe0f Replay ceremony", href="#",
                       id="replay-knight-link",
                       style={"cursor": "pointer", "fontSize": "12px",
                              "color": "#b8860b", "textDecoration": "none"}),
                id="replay-knight-wrap",
                style={"textAlign": "center", "marginTop": "4px",
                       "display": "none"}),
            html.Div(
                dbc.Button("\u2694\ufe0f Accept Knighthood", id="onion-knight-btn",
                           color="warning", size="lg",
                           style={"fontFamily":"Palatino Linotype, Palatino, Book Antiqua, serif",
                                  "fontWeight":"700", "padding":"10px 30px",
                                  "fontSize":"16px", "borderRadius":"8px"}),
                id="onion-knight-wrap",
                style={"textAlign":"center", "marginTop":"16px", "display":"none"}),
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
                    id="splash-next-wrap", style=_STYLE_HIDDEN,
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
                        html.Div([
                            html.Span(id="price-ticker",
                                      style={"fontSize":"23px", "fontWeight":"600",
                                             "color":"rgba(255,255,255,0.9)",
                                             "whiteSpace":"nowrap",
                                             "fontFamily":"'SF Mono', 'Cascadia Code', 'JetBrains Mono', 'Fira Code', Menlo, Consolas, monospace",
                                             "fontVariantNumeric":"tabular-nums"}),
                            html.Span(id="ticker-pct", n_clicks=0,
                                      style={"fontSize":"27px", "fontWeight":"700",
                                             "whiteSpace":"nowrap", "cursor":"pointer",
                                             "marginLeft":"6px", "userSelect":"none",
                                             "fontFamily":"'SF Mono', 'Cascadia Code', 'JetBrains Mono', 'Fira Code', Menlo, Consolas, monospace",
                                             "fontVariantNumeric":"tabular-nums"}),
                            html.Span(id="price-sparkline",
                                      style={"display":"inline-block", "verticalAlign":"middle"}),
                            html.Span(id="ticker-mode-toggle",
                                      n_clicks=0,
                                      style={"fontSize":"13px", "color":"rgba(255,255,255,0.45)",
                                             "cursor":"pointer", "marginLeft":"8px",
                                             "verticalAlign":"middle", "userSelect":"none"}),
                        ], style={"display":"flex", "alignItems":"center",
                                  "marginLeft":"14px"}),
                    ], style={"display":"flex", "alignItems":"center"}),
                    # Right: collapsible drawer (stacked vertically) + toggle
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Span("Stay dark, Anon \u25b6 ",
                                          style={"fontSize":"9px", "color":"rgba(255,255,255,0.4)",
                                                 "whiteSpace":"nowrap"}),
                                html.A(
                                    "\U0001f9c5 Tor onion",
                                    href="http://u5dprelc4ti7xoczb5sbtye6qidlji2l6psmkx35anvxgjyqrkmu32ad.onion",
                                    target="_blank",
                                    rel="noopener noreferrer",
                                    className="text-decoration-none",
                                    style={"fontSize":"15px", "color":"rgba(255,255,255,0.75)"},
                                ),
                            ], style={"display":"flex", "alignItems":"center",
                                      "justifyContent":"flex-end"}),
                            html.Div([
                                dbc.Select(
                                    id="palette-select",
                                    options=[{"label": v, "value": k}
                                             for k, v in _app_ctx.PALETTE_LABELS.items()],
                                    value="default",
                                    size="sm",
                                    style={"width": "155px", "fontSize": "0.78rem",
                                           "display": "inline-block", "marginRight": "8px"},
                                ),
                                html.Span("Cooler than you think \u25b6 ",
                                          style={"fontSize":"9px", "color":"rgba(255,255,255,0.4)",
                                                 "whiteSpace":"nowrap"}),
                                dbc.Button("\U0001f4f8 Share", id="share-btn", size="sm",
                                           className="btn-share-accent"),
                            ], style={"display":"flex", "alignItems":"center",
                                      "justifyContent":"flex-end",
                                      "marginTop":"2px"}),
                        ], id="desktop-nav-drawer", className="desktop-nav-drawer"),
                        html.Div("\u22ef", id="desktop-nav-toggle",
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
                    html.Div("\u22ef", id="mobile-nav-toggle",
                             className="mobile-nav-toggle mobile-nav-toggle-hidden",
                             style={"color":"rgba(255,255,255,0.5)",
                                    "fontSize":"18px", "cursor":"pointer",
                                    "lineHeight":"1", "letterSpacing":"2px",
                                    "padding":"4px 8px"}),
                    html.Span(id="price-ticker-mobile",
                             style={"fontSize":"20px", "fontWeight":"700",
                                    "color":"rgba(255,255,255,0.95)",
                                    "whiteSpace":"nowrap",
                                    "fontFamily":"'SF Mono', 'Cascadia Code', 'JetBrains Mono', 'Fira Code', Menlo, Consolas, monospace",
                                    "fontVariantNumeric":"tabular-nums"}),
                    html.Span(id="ticker-pct-mobile", n_clicks=0,
                             style={"fontSize":"24px", "fontWeight":"700",
                                    "whiteSpace":"nowrap", "cursor":"pointer",
                                    "marginLeft":"4px", "userSelect":"none",
                                    "fontFamily":"'SF Mono', 'Cascadia Code', 'JetBrains Mono', 'Fira Code', Menlo, Consolas, monospace",
                                    "fontVariantNumeric":"tabular-nums"}),
                ], style={"display":"flex", "alignItems":"center",
                          "justifyContent":"space-between", "width":"100%"}),
                # Row 2: collapsible drawer — full content, auto-hides after 3s
                html.Div([
                    html.Hr(style={"borderColor":"rgba(255,255,255,0.12)",
                                   "margin":"3px 0"}),
                    html.Div([
                        html.A([
                            html.Span("\U0001f9c5", style={"fontSize":"20px",
                                                    "lineHeight":"1"}),
                            html.Span(" \u25c2 Stay dark, Anon",
                                      style={"fontSize":"11px",
                                             "color":"rgba(255,255,255,0.5)",
                                             "marginLeft":"5px"}),
                        ], href="http://u5dprelc4ti7xoczb5sbtye6qidlji2l6psmkx35anvxgjyqrkmu32ad.onion",
                           target="_blank", rel="noopener noreferrer",
                           className="text-decoration-none",
                           style={"display":"flex", "alignItems":"center"}),
                        dbc.Button("\U0001f4f8 Share", id="share-btn-mobile", size="sm",
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
                inputStyle=_CB_MARGIN,
                className="mb-2 small",
            ),
            dcc.Checklist(
                id="share-include-lots",
                options=[{"label": " Include Stack Tracker lots in link", "value": "yes"}],
                value=[], inputStyle=_CB_MARGIN,
                className="small",
            ),
            dbc.Button("Generate link", id="share-copy-btn",
                       size="sm", className="mt-2 mb-3 w-100 btn-generate-accent"),
            dbc.InputGroup([
                dbc.Input(id="share-url-display", type="text", readonly=True,
                          placeholder="Click 'Generate link' above\u2026", size="sm"),
                dcc.Clipboard(id="share-clipboard",
                              target_id="share-url-display",
                              style={"cursor":"pointer","fontSize":"18px",
                                     "padding":"4px 8px"}),
            ], size="sm"),
            html.Div([
                html.Img(id="share-qr-img", src="", style={"display": "none"}),
                html.Div("Scan to open on another device",
                         id="share-qr-label",
                         style={"display": "none", "fontSize": "10px",
                                "color": "#888", "textAlign": "center",
                                "marginBottom": "8px"}),
            ]),
            html.Img(id="share-preview-thumb", src="", className="share-preview-thumb",
                     style={"display": "none"}),
            html.Hr(className="my-3"),
            html.Div([
                html.Span("Link History", className="fw-semibold small"),
                html.Span(" (your browser only — no duplicates)",
                          className="text-muted small ms-1"),
            ], className="mb-2"),
            html.Div(id="link-history-display"),
            dbc.Button("\U0001f5d1 Clear history", id="clear-history-btn",
                       color="link", size="sm", className="text-danger mt-2 p-0"),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="share-modal-close", className="ms-auto", size="sm")
        ),
    ], id="share-modal", is_open=False, size="lg", scrollable=True),
    dbc.Tabs([
        dbc.Tab(_bubble_tab(),       label="\U0001F4C8 Bubble + QR Overlay", tab_id="bubble"),
        dbc.Tab(_heatmap_tab(),      label="\U0001F525 CAGR Heatmap",        tab_id="heatmap"),
        dbc.Tab(_dca_tab(),          label="\U0001F4B0 BTC Accumulator",     tab_id="dca"),
        dbc.Tab(_retire_tab(),       label="\U0001F3D6\uFE0F BTC RetireMentator",  tab_id="retire"),
        dbc.Tab(_supercharge_tab(),  label="\u26A1 HODL Supercharger",   tab_id="supercharge"),
        dbc.Tab(_stack_tracker_tab(),label="\U0001F5DD\uFE0F Stack Tracker",       tab_id="stack"),
        dbc.Tab(_model_info_tab(),   label="\U0001F4D0 Model Info",      tab_id="model_info"),
        dbc.Tab(_faq_tab(),          label="\u2753 FAQ",                 tab_id="faq"),
        dbc.Tab(_citadel_tab(),      label="\U0001F3F0 Citadel Planner", tab_id="citadel"),
    ], id="main-tabs", active_tab=initial_tab),
    # ── Footer: block height + halving countdown + doc links ──────────────
    html.Div([
        html.Span(id="footer-block-height",
                  style={"marginRight": "16px"}),
        html.Span(id="footer-halving-countdown"),
        html.Span([
            html.Span(" · ", className="footer-sep-inline"),
            html.A("Architecture", href="/docs/architecture",
                   className="footer-link"),
            html.Span(" · "),
            html.A("User Manual", href="/docs/user-manual",
                   className="footer-link"),
        ], className="footer-docs"),
    ], className="site-footer",
       style={"textAlign": "center", "fontSize": "11px",
              "color": "rgba(0,0,0,0.35)", "padding": "10px 0 14px",
              "fontFamily": "monospace", "letterSpacing": "0.5px"}),
], fluid=True, className="px-2 py-1")
