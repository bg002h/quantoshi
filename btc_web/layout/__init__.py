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
# _LOGO_B64_ALL removed from layout — loaded on demand by export callback
from snapshot import _SNAPSHOT_CONTROLS

# ── Re-exports (consumed by callbacks.py, app.py, etc.) ─────────────────────
from layout.common import (_global_lppl_modal, _global_hybppl_modal,
                             _global_eppl_modal, _global_bm_modal)
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
from layout.leverage import _leverage_tab
from layout.common import _export_row, _BTC_ORANGE
from colors import (
    WHITE, FALLBACK_MODEL_GRAY,
    CODE_BG, DIM_TEXT, MUTED_TEXT, SPLASH_BRAND_DARK, KNIGHT_GOLD,
    BLACK, _hex_alpha,
    FONT_BRAND, FONT_MONO,
    UI_FONT_XS, UI_FONT_SM, UI_FONT_MD, UI_FONT_BASE,
    UI_FONT_LG, UI_FONT_XL, UI_FONT_XXL, UI_FONT_HEADING,
    TICKER_TEXT_ALPHA, NAV_DIM_ALPHA, NAV_TAGLINE_ALPHA,
    NAV_TOGGLE_ALPHA, NAV_HR_ALPHA, MOBILE_TICKER_ALPHA,
    FOOTER_TEXT_ALPHA,
)

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
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        {%css%}
        <script>
  (function() {
    try {
      var raw = localStorage.getItem("palette-store");
      if (raw) {
        var key;
        try { key = JSON.parse(raw); } catch(e) { key = raw; }
        if (typeof key === "string") {
          document.documentElement.dataset.palette = key;
        }
      }
    } catch(e) {}
  })();
</script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
            <script>
            window.addEventListener('load',function(){
                setTimeout(function(){
                    ['knighting','wizard'].forEach(function(f){
                        var s=document.createElement('script');
                        s.src='/assets/.deferred/'+f+'.js';
                        s.defer=true;
                        document.body.appendChild(s);
                    });
                },3000);
            });
            </script>
        </footer>
    </body>
</html>"""

# Canonical copy in callbacks/routing.py — keep in sync
_PATH_TO_TAB = {
    "/1": "bubble", "/2": "heatmap", "/3": "dca",
    "/4": "retire", "/5": "supercharge", "/6": "citadel",
    "/7": "leverage", "/8": "stack", "/9": "model_info", "/10": "faq",
    "/leverage": "leverage",
    "/faq": "faq", "/mi": "model_info",
}

_TAB_TO_GRAPH = {
    "bubble": "bubble-graph", "heatmap": "heatmap-graph",
    "dca": "dca-graph", "retire": "retire-graph",
    "supercharge": "supercharge-graph", "citadel": "citadel-graph",
    "leverage": "lev-graph",
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
            from figures.leverage import build_leverage_figure
            from tab_defaults import leverage_defaults
            _TAB_TO_FIG_FN["leverage"] = (build_leverage_figure, leverage_defaults)
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
    # /_dash-layout is a separate XHR — use Referer to determine target tab
    if path == "/_dash-layout":
        referer = flask.request.headers.get("Referer", "")
        # Extract path from referer URL (e.g., "https://quantoshi.xyz/3" → "/3")
        from urllib.parse import urlparse
        path = urlparse(referer).path or "/"
    clean = path.rstrip("/").split(".")[0] or "/"
    initial_tab = _PATH_TO_TAB.get(clean, "bubble")
    layout = _build_layout(initial_tab)

    # Inject pre-computed figure ONLY for the target tab to reduce layout JSON size.
    # All 6 figures were ~1.5MB of the 2.2MB layout; injecting one cuts it to ~600KB.
    # Non-target tabs get their figures via first-render callback on first visit.
    from layout.common import _inject_initial_figure
    target_gid = _TAB_TO_GRAPH.get(initial_tab)
    if target_gid:
        fig = _get_initial_figure(initial_tab)
        if fig:
            _inject_initial_figure(layout, target_gid, fig)

    return layout

_app_ctx.app.layout = _serve_layout


def _pick_splash_quote():
    """Pick a random quote at layout time so it's pre-populated in the layout JSON.

    Returns (children_list, attribution). The children_list is a list of
    Dash components (strings and html.A for markdown links) ready to pass
    to html.P — avoids dcc.Markdown which loads from an async JS chunk.
    """
    import json as _json, random as _rnd, time as _time, re as _re
    from layout.splash import _SPLASH_QUOTES_JS
    try:
        quotes = _json.loads(_SPLASH_QUOTES_JS)
        # Genesis (index 0) is always first; the callback may replace it later
        q_text, q_attr = quotes[0]
    except Exception:
        return [""], ""

    # Parse markdown [text](url) links into html.A components
    parts = []
    last = 0
    for m in _re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", q_text):
        if m.start() > last:
            parts.append(q_text[last:m.start()])
        parts.append(html.A(m.group(1), href=m.group(2),
                            target="_blank", rel="noopener"))
        last = m.end()
    if last < len(q_text):
        parts.append(q_text[last:])
    return ['"'] + parts + ['"'], q_attr


def _build_layout(initial_tab="bubble"):
    _splash_q, _splash_a = _pick_splash_quote()

    def _t(tab_id, label, content_fn):
        """Eager-render if this is the active tab; else lazy placeholder.

        All tab content is wrapped in a {tab_id}-lazy Div that the
        corresponding lazy-load callback in callbacks/routing.py populates
        on first active-tab match.
        """
        lazy_id = f"{tab_id}-lazy"
        if tab_id == initial_tab:
            body = html.Div(content_fn(), id=lazy_id)
        else:
            body = html.Div(
                html.P("Loading...", className="text-muted p-4"),
                id=lazy_id,
            )
        return dbc.Tab(body, label=label, tab_id=tab_id)

    return dbc.Container([
    _freq_warning_modal(),
    # ── Display Models family summary Store ────────────────────────────
    # Single source of truth for LPPL/HybPPL/EPPL summary strings.
    # Written by compute_family_summaries callback (added in Task 2); read
    # by update_model_swatches + inline Store-reader clientside callbacks +
    # heatmap status row.
    dcc.Store(
        id="display-model-summaries",
        storage_type="memory",
        data={"lppl": "LPPL\u2083", "hybppl": "1d+1u", "eppl": "1d+1u"},
    ),
    # ── Display Models consolidation placeholder block (always emitted) ──
    # These 15 hidden dcc.Checklist components satisfy callback registration
    # for _SNAPSHOT_CONTROLS tuples that are defunct after the display-models
    # refactor but MUST remain in _SNAPSHOT_CONTROLS to preserve positional
    # bit-index stability for old q3: share links.
    # See spec: 2026-04-11-display-models-consolidation-design.md, Task 0 finding 3.
    html.Div(
        id="_defunct-snapshot-placeholders",
        style={"display": "none"},
        children=[
            dcc.Checklist(
                id=f"{prefix}-{family}-activate",
                options=[{"label": "", "value": "yes"}],
                value=[],
            )
            for prefix in ("bub", "dca", "ret", "sc", "hm")
            for family in ("lppl", "hybppl", "eppl")
        ],
    ),
    dcc.Interval(id="price-interval", interval=_PRICE_INTERVAL_MS, n_intervals=0),
    # Per-tab staggered prefetch timers. Each fires once (max_intervals=1) at
    # its own delay, so non-active tabs get quietly populated in the
    # background starting ~1.5s after load. Stagger prevents large simultaneous
    # React reconciliations that could stutter the active tab.
    *[dcc.Interval(id=f"{_pf_tid}-prefetch-iv",
                   interval=1500 + _pf_idx * 400,
                   max_intervals=1, n_intervals=0)
      for _pf_idx, _pf_tid in enumerate(
          ("bubble","heatmap","dca","retire","supercharge",
           "citadel","leverage","stack","model_info","faq"))],
    dcc.Store(id="prefetch-ready", storage_type="memory", data=0),
    # Set by restore_from_url to loaded_hash when it directly built the
    # active tab's figure via restore_builder._build_bubble_figure_from_state.
    # The splash listener watches this so prefetch-ready=1 (non-active-tab
    # warming) only fires AFTER the active chart has actually been drawn.
    # update_bubble reads it as State + uses it to short-circuit redundant
    # rebuilds when CTA's tick bump re-fires the callback post-restore.
    # See memory/restore_callback_architecture.md.
    dcc.Store(id="active-chart-committed", storage_type="memory", data=None),
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
    dcc.Store(id="wm-b64-store", storage_type="memory", data=None),  # no longer embedded in layout
    dcc.Store(id="auto-y-grid", storage_type="memory",
              data=_app_ctx.AUTO_Y_GRID if initial_tab == "bubble" else None),
    # Per-tab render triggers — 1 for the target tab (figure pre-injected),
    # 0 for others (callback fires on first visit to fetch their figure).
    *[dcc.Store(id=f"{tab}-first-render", storage_type="memory",
                data=1 if tab == initial_tab else 0)
      for tab in ("bubble", "heatmap", "dca", "retire", "supercharge", "citadel", "leverage")],
    # Shared tick store: palette / lots writes bump this; a clientside
    # router then bumps the active tab's first-render. Replaces the old
    # bridge callback that fired on initial Store hydration.
    dcc.Store(id="active-tab-bump-tick", storage_type="memory", data=0),
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
    # Hybppl/Eppl modal-close commit triggers. Chart callbacks demote the 26
    # radio Inputs to State and watch these counters so mid-edit radio clicks
    # don't re-fire charts until the user closes the gear modal.
    dcc.Store(id="hybppl-commit-trigger", storage_type="memory", data=0),
    dcc.Store(id="eppl-commit-trigger",   storage_type="memory", data=0),
    dcc.Store(id="bm-commit-trigger",     storage_type="memory", data=0),
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
                     style={"fontSize": UI_FONT_LG, "marginBottom": "10px"}),
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
                              style={"fontSize": UI_FONT_SM, "wordBreak": "break-all",
                                     "display": "block", "padding": "8px",
                                     "backgroundColor": CODE_BG,
                                     "borderRadius": "4px",
                                     "fontFamily": FONT_MONO,
                                     "userSelect": "all", "lineHeight": "1.4",
                                     "maxHeight": "80px", "overflow": "auto"}),
                    dbc.Button("Copy", id="mc-pay-copy-btn", size="sm",
                               color="secondary", outline=True,
                               className="mt-1",
                               style={"fontSize": UI_FONT_MD}),
                ]),
                html.Div(id="mc-pay-amount-info",
                         style={"fontSize": UI_FONT_BASE, "color": DIM_TEXT,
                                "marginTop": "6px", "textAlign": "center"}),
            ]),
            dcc.Store(id="mc-pay-methods", storage_type="memory", data=None),
            html.Div(id="mc-pay-status",
                     style={"fontSize": UI_FONT_BASE, "color": DIM_TEXT,
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
                   style={"fontFamily": FONT_MONO,
                          "fontSize": UI_FONT_XL, "color": DIM_TEXT, "textAlign": "center",
                          "letterSpacing": "1px"}),
            html.P("Are you sure you want to continue?",
                   style={**_QUANT_FONT, "fontWeight": "bold",
                          "fontSize": "15px", "marginTop": "10px",
                          "textAlign": "center"}),
            html.P(id="mc-quant-onchain-note",
                   style={"fontStyle": "italic", "fontSize": UI_FONT_BASE,
                          "textAlign": "center", "color": DIM_TEXT}),
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
    # Gate for single-redraw-per-snapshot; armed by restore_from_url,
    # released by apply_tab_{tab} or 3s safety timer. See spec
    # 2026-04-24-single-redraw-per-snapshot-design.md.
    dcc.Store(id="snapshot-pending", storage_type="memory", data=False),
    # Per-tab flag: records the loaded_hash each tab last applied.
    # Prevents apply_tab_{tab} from re-writing stale snapshot values when
    # {tab}-first-render bumps again for unrelated reasons
    # (e.g. sigma-mode radio change).
    *[dcc.Store(id=f"{t}-snap-applied", storage_type="memory", data=None)
      for t in ("bubble", "heatmap", "dca", "retire",
                "supercharge", "citadel", "leverage")],
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
                         style={"fontFamily": FONT_BRAND,
                                "fontSize":"1.5rem", "fontWeight":"700",
                                "color":SPLASH_BRAND_DARK, "marginLeft":"10px"}),
            ], style={"display":"flex", "alignItems":"center",
                      "justifyContent":"center", "marginBottom":"20px"}),
            html.Div([
                html.P(id="splash-quote-text",
                       children=_splash_q,
                       style={"fontSize": UI_FONT_XXL, "fontStyle":"italic",
                              "color":SPLASH_BRAND_DARK, "lineHeight":"1.5",
                              "textAlign":"center", "marginBottom":"10px"}),
                html.Div(id="splash-quote-attr",
                         children=(f"\u2014 {_splash_a}" if _splash_a else ""),
                         style={"fontSize": UI_FONT_LG, "color":MUTED_TEXT,
                                "textAlign":"center"}),
            ], style={"padding":"10px 20px"}),
            html.Div(id="journey-stats",
                     style={"textAlign":"center", "fontSize": UI_FONT_BASE,
                            "color":FALLBACK_MODEL_GRAY, "marginTop":"16px",
                            "lineHeight":"1.7", "display":"none"}),
            html.Div(id="knight-welcome",
                     style={"fontSize": UI_FONT_LG, "color":KNIGHT_GOLD,
                            "textAlign":"center", "marginTop":"8px"}),
            html.Div(
                html.A("\u2694\ufe0f Replay ceremony", href="#",
                       id="replay-knight-link",
                       style={"cursor": "pointer", "fontSize": UI_FONT_BASE,
                              "color": KNIGHT_GOLD, "textDecoration": "none"}),
                id="replay-knight-wrap",
                style={"textAlign": "center", "marginTop": "4px",
                       "display": "none"}),
            html.Div(
                dbc.Button("\u2694\ufe0f Accept Knighthood", id="onion-knight-btn",
                           color="warning", size="lg",
                           style={"fontFamily": FONT_BRAND,
                                  "fontWeight":"700", "padding":"10px 30px",
                                  "fontSize": UI_FONT_XXL, "borderRadius":"8px"}),
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
                               style={"marginLeft":"12px", "fontSize": UI_FONT_LG,
                                      "borderRadius":"8px"}),
                    id="splash-next-wrap", style=_STYLE_HIDDEN,
                ),
            ], style={"textAlign":"center", "marginTop":"24px"}),
        ], style={"padding":"30px 20px 24px"}),
    ], id="splash-modal", is_open=False, centered=True, backdrop="static",
       className="splash-modal"),
    dbc.Modal([
        dbc.ModalBody([
            html.Div([
                html.Img(src="/assets/quantoshi_logo_nav.png",
                         style={"width": "80px", "height": "auto",
                                "marginBottom": "16px", "opacity": "0.9"}),
                html.Div([
                    dbc.Spinner(size="sm", color="primary",
                                spinner_style={"marginRight": "10px"}),
                    html.Span("Restoring your shared view…"),
                ], style={"display": "flex", "alignItems": "center",
                          "justifyContent": "center",
                          "fontSize": UI_FONT_BASE, "color": MUTED_TEXT,
                          "marginTop": "4px"}),
            ], style={"textAlign": "center", "padding": "24px"}),
        ]),
    ], id="restore-progress-modal", is_open=False, centered=True, size="sm",
       backdrop="static", keyboard=False),
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
                                        style={"fontFamily": FONT_BRAND}),
                        html.Div([
                            html.Span(id="price-ticker",
                                      style={"fontSize":"23px", "fontWeight":"600",
                                             "color":_hex_alpha(WHITE, TICKER_TEXT_ALPHA),
                                             "whiteSpace":"nowrap",
                                             "fontFamily": FONT_MONO,
                                             "fontVariantNumeric":"tabular-nums"}),
                            html.Span(id="ticker-pct", n_clicks=0,
                                      style={"fontSize":"27px", "fontWeight":"700",
                                             "whiteSpace":"nowrap", "cursor":"pointer",
                                             "marginLeft":"6px", "userSelect":"none",
                                             "fontFamily": FONT_MONO,
                                             "fontVariantNumeric":"tabular-nums"}),
                            html.Span(id="price-sparkline",
                                      style={"display":"inline-block", "verticalAlign":"middle"}),
                            html.Span(id="ticker-mode-toggle",
                                      n_clicks=0,
                                      style={"fontSize": UI_FONT_LG, "color":_hex_alpha(WHITE, NAV_DIM_ALPHA),
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
                                          style={"fontSize": UI_FONT_XS, "color":_hex_alpha(WHITE, NAV_TAGLINE_ALPHA),
                                                 "whiteSpace":"nowrap"}),
                                html.A(
                                    "\U0001f9c5 Tor onion",
                                    href="http://u5dprelc4ti7xoczb5sbtye6qidlji2l6psmkx35anvxgjyqrkmu32ad.onion",
                                    target="_blank",
                                    rel="noopener noreferrer",
                                    className="text-decoration-none",
                                    style={"fontSize":"15px", "color":_hex_alpha(WHITE, 0.75)},
                                ),
                            ], style={"display":"flex", "alignItems":"center",
                                      "justifyContent":"flex-end"}),
                            html.Div([
                                # Navbar palette selector hidden — per-tab selectors at
                                # the bottom of each chart tab's controls are the primary
                                # palette switcher. This component stays in the DOM (hidden)
                                # because nav.py sync callbacks reference its id.
                                dbc.Select(
                                    id="palette-select",
                                    options=[{"label": v, "value": k}
                                             for k, v in _app_ctx.PALETTE_LABELS.items()],
                                    value="default",
                                    size="sm",
                                    style={"display": "none"},
                                ),
                                html.Span("Cooler than you think \u25b6 ",
                                          style={"fontSize": UI_FONT_XS, "color":_hex_alpha(WHITE, NAV_TAGLINE_ALPHA),
                                                 "whiteSpace":"nowrap"}),
                                dbc.Button("\U0001f4f8 Share", id="share-btn", size="sm",
                                           className="btn-share-accent"),
                            ], style={"display":"flex", "alignItems":"center",
                                      "justifyContent":"flex-end",
                                      "marginTop":"2px"}),
                        ], id="desktop-nav-drawer", className="desktop-nav-drawer"),
                        html.Div("\u22ef", id="desktop-nav-toggle",
                                 className="desktop-nav-toggle desktop-nav-toggle-hidden",
                                 style={"color":_hex_alpha(WHITE, NAV_TOGGLE_ALPHA),
                                        "fontSize": UI_FONT_HEADING, "cursor":"pointer",
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
                                  style={"fontFamily": FONT_BRAND,
                                         "fontSize":"1.75rem", "color":WHITE}),
                    ], style={"display":"flex", "alignItems":"center"}),
                    html.Div("\u22ef", id="mobile-nav-toggle",
                             className="mobile-nav-toggle mobile-nav-toggle-hidden",
                             style={"color":_hex_alpha(WHITE, NAV_TOGGLE_ALPHA),
                                    "fontSize": UI_FONT_HEADING, "cursor":"pointer",
                                    "lineHeight":"1", "letterSpacing":"2px",
                                    "padding":"4px 8px"}),
                    html.Span(id="price-ticker-mobile",
                             style={"fontSize":"20px", "fontWeight":"700",
                                    "color":_hex_alpha(WHITE, MOBILE_TICKER_ALPHA),
                                    "whiteSpace":"nowrap",
                                    "fontFamily": FONT_MONO,
                                    "fontVariantNumeric":"tabular-nums"}),
                    html.Span(id="ticker-pct-mobile", n_clicks=0,
                             style={"fontSize":"24px", "fontWeight":"700",
                                    "whiteSpace":"nowrap", "cursor":"pointer",
                                    "marginLeft":"4px", "userSelect":"none",
                                    "fontFamily": FONT_MONO,
                                    "fontVariantNumeric":"tabular-nums"}),
                ], style={"display":"flex", "alignItems":"center",
                          "justifyContent":"space-between", "width":"100%"}),
                # Row 2: collapsible drawer — full content, auto-hides after 3s
                html.Div([
                    html.Hr(style={"borderColor":_hex_alpha(WHITE, NAV_HR_ALPHA),
                                   "margin":"3px 0"}),
                    html.Div([
                        html.A([
                            html.Span("\U0001f9c5", style={"fontSize":"20px",
                                                    "lineHeight":"1"}),
                            html.Span(" \u25c2 Stay dark, Anon",
                                      style={"fontSize": UI_FONT_MD,
                                             "color":_hex_alpha(WHITE, NAV_TOGGLE_ALPHA),
                                             "marginLeft":"5px"}),
                        ], href="http://u5dprelc4ti7xoczb5sbtye6qidlji2l6psmkx35anvxgjyqrkmu32ad.onion",
                           target="_blank", rel="noopener noreferrer",
                           className="text-decoration-none",
                           style={"display":"flex", "alignItems":"center"}),
                        dbc.Button("\U0001f4f8 Share", id="share-btn-mobile", size="sm",
                                   className="btn-share-accent",
                                   style={"fontSize": UI_FONT_SM, "padding":"2px 8px"}),
                    ], style={"display":"flex", "alignItems":"center",
                              "justifyContent":"space-between", "width":"100%"}),
                ], id="mobile-nav-drawer", className="mobile-nav-drawer"),
            ], className="d-md-none w-100"),
        ], fluid=True),
        color=SPLASH_BRAND_DARK, dark=True, className="mb-0 py-1 mt-1 navbar-parallax",
        id="main-navbar",
    ),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Share / Restore Configuration")),
        dbc.ModalBody([
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
                              style={"cursor":"pointer","fontSize": UI_FONT_HEADING,
                                     "padding":"4px 8px"}),
            ], size="sm"),
            html.Div([
                html.Img(id="share-qr-img", src="", style={"display": "none"}),
                html.Div("Scan to open on another device",
                         id="share-qr-label",
                         style={"display": "none", "fontSize": UI_FONT_SM,
                                "color": FALLBACK_MODEL_GRAY, "textAlign": "center",
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
    # All non-active tabs are lazy-loaded: only initial_tab's content is
    # materialized at layout build time; others show a placeholder until
    # the user switches to them (or a lazy callback fires on active_tab
    # change). This keeps /N layout JSON small — a visit to /1 doesn't
    # download heatmap/dca/retire/SC control panels, etc.
    dbc.Tabs([
        _t("bubble",       "Price Models",    _bubble_tab),
        _t("heatmap",      "Heatmap",         _heatmap_tab),
        _t("dca",          "Accumulator",     _dca_tab),
        _t("retire",       "RetireMentator",  _retire_tab),
        _t("supercharge",  "Supercharger",    _supercharge_tab),
        _t("citadel",      "Citadel",         _citadel_tab),
        _t("leverage",     "Max Pay-Price",   _leverage_tab),
        _t("stack",        "Stack",           _stack_tracker_tab),
        _t("model_info",   "Model Info",      lambda: _model_info_tab().children),
        _t("faq",          "FAQ",             _faq_tab),
    ], id="main-tabs", active_tab=initial_tab),
    _global_lppl_modal(),
    _global_hybppl_modal(),
    _global_eppl_modal(),
    _global_bm_modal(),
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
       style={"textAlign": "center", "fontSize": UI_FONT_MD,
              "color": _hex_alpha(BLACK, FOOTER_TEXT_ALPHA), "padding": "10px 0 14px",
              "fontFamily": FONT_MONO, "letterSpacing": "0.5px"}),
], fluid=True, className="px-2 py-1")
