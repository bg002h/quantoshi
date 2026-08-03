"""Tab 1 — Bubble + QR Overlay layout."""

import json

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from tab_defaults import BUBBLE
from snapshot_defaults import SNAPSHOT_DEFAULTS
from colors import (
    USER_MODEL_TRACE, FALLBACK_MODEL_GRAY,
    DIM_TEXT, BOOTSTRAP_LIGHT_BG,
    _hex_alpha, FONT_MONO,
    UI_FONT_XS, UI_FONT_SM, UI_FONT_MD, UI_FONT_BASE,
    UM_INPUT_BG_ALPHA, UM_INPUT_BORDER_ALPHA,
)
from layout.common import (_tab_hints, _section_card, _row, _lbl,
                            _STYLE_HIDDEN, _STYLE_HINT, _q_panel, _q_panel_with_mode,
                            _q_options, _legend_pos_dropdown,
                            _chart_tab_layout, _CB_MARGIN, _palette_selector,
                            _plot_appearance_controls, _use_lots_checklist)
from layout.display_models import display_models_panel, sigma_mode_section
from layout.custom_time import custom_time_panel
from layout.mc_controls import _mc_controls


# ── Axes presets (Tab 1) ─────────────────────────────────────────────────
# Registry is the single source of truth for the preset row. Callbacks in
# callbacks/axes_presets.py import it and register one clientside callback per
# entry. Adding a preset = one entry here + its JS body.
# Spec: docs/superpowers/specs/2026-08-03-axes-presets-design.md

# The five Axes & Range controls a preset may write, in Output order.
# Every preset JS body returns a list in exactly this order.
AXES_CONTROL_IDS = ("bub-xscale", "bub-yscale", "bub-xrange",
                    "bub-yrange", "bub-auto-y")

# System defaults, read from the SSOT and baked into the preset JS at import
# time. test_axes_presets.py guards against drift.
AXES_DEFAULTS = {f"{cid}:value": SNAPSHOT_DEFAULTS[f"{cid}:value"]
                 for cid in AXES_CONTROL_IDS}

# X range that CAGR view uses, where the X slider means exit years rather than
# calendar years. SINGLE SOURCE: callbacks/charts/__init__.py (toggle_bub_view,
# 3 sites) and callbacks/routing.py (the /1.2 deep link) import this. Those
# sites compare it for EXACT EQUALITY to swap back to the price-view range, so
# a second copy silently breaks the CAGR<->price round-trip.
# MUST stay a list -- [2025, 2050] == (2025, 2050) is False in Python.
CAGR_DEFAULT_XRANGE = [2025, 2050]

# Domain of the bub-xrange slider. SINGLE SOURCE: the slider definition below
# and every preset JS body that clamps to it read these. Previously the bounds
# were literals in two places; a third copy was about to be added for the
# "+1 year" preset, which is the same drift hazard CAGR_DEFAULT_XRANGE fixed.
XRANGE_MIN = 2010
XRANGE_MAX = 2080

# Sets the X window to the current calendar year. Y is deliberately left alone:
# when auto-Y is on the existing clientside recompute fits it, and when auto-Y
# is off the user has taken manual control (spec D2).
_JS_CUR_YEAR = """
function(n) {
    var NU = window.dash_clientside.no_update;
    if (!n) { return [NU, NU, NU, NU, NU]; }
    var y = new Date().getFullYear();
    /* clamp so y+1 stays inside the slider domain */
    y = Math.max(%(xmin)s, Math.min(y, %(xmax)s - 1));
    return [NU, NU, [y, y + 1], NU, NU];
}
""" % {"xmin": XRANGE_MIN, "xmax": XRANGE_MAX}

# ── Step presets: move one edge of the X window by a year ────────────────
# These are the RELATIVE presets: they read the live bub-xrange as State rather
# than computing an absolute answer, so repeated taps accumulate. Reading a prop
# that the same callback also writes is an established pattern here --
# auto_bubble_yrange does exactly that with bub-yrange
# (callbacks/charts/__init__.py:833,841).
#
# One template, so a new step preset cannot drift from its siblings. Each
# returns no_update when the move would leave the slider domain or invert the
# window, rather than rewriting an unchanged value -- a tap at a boundary
# therefore costs no chart redraw.
_JS_STEP_TEMPLATE = """
function(n, xrange) {
    var NU = window.dash_clientside.no_update;
    if (!n || !xrange || xrange.length !== 2) { return [NU, NU, NU, NU, NU]; }
    var lo = xrange[0], hi = xrange[1];
    %(mutate)s
    if (lo < %(xmin)s || hi > %(xmax)s || lo >= hi) {
        return [NU, NU, NU, NU, NU];
    }
    return [NU, NU, [lo, hi], NU, NU];
}
"""

# Extends the END forward a year; the start stays put.
_JS_PLUS_1Y = _JS_STEP_TEMPLATE % {
    "mutate": "hi = hi + 1;", "xmin": XRANGE_MIN, "xmax": XRANGE_MAX}

# Extends the START back a year; the end stays put. The default window already
# begins at XRANGE_MIN, so this preset is inert on a freshly loaded page -- it
# hides itself in that state rather than offering a button that does nothing
# (see _JS_HIDE_AT_FLOOR).
_JS_MINUS_1Y = _JS_STEP_TEMPLATE % {
    "mutate": "lo = lo - 1;", "xmin": XRANGE_MIN, "xmax": XRANGE_MAX}

# Optional per-preset visibility rule. A preset that carries "hide_js" gets a
# second clientside callback wired to bub-xrange.value, driving its button's
# `style`, so a preset that would be inert simply is not shown. Fires on page
# load (no prevent_initial_call), so the initial state is correct without a tap.
# The Output is not allow_duplicate -- nothing else writes these buttons'
# style -- which is what makes firing on load safe here.
_JS_HIDE_AT_FLOOR = """
function(xrange) {
    var atFloor = (!xrange || xrange.length !== 2 || xrange[0] <= %(xmin)s);
    return atFloor ? {display: "none"} : {};
}
""" % {"xmin": XRANGE_MIN}

# Restores the axes the page loaded with: the share-link URL's values when the
# page came from one, system defaults otherwise. View-awareness applies ONLY to
# the fallback -- a link's X range wins in every view.
#
# The presence test must be `!== undefined && !== null`, never `||`:
# bub-auto-y's legitimate "off" value is [], which is falsy.
#
# bub-xrange gets one further narrowing, deliberately NOT applied to the other
# four fields. Production always emits q4 links (snapshot_cb.py), and q4
# decode back-fills EVERY control from the historical-defaults registry
# (snapshot.py::_decode_snapshot_v4) -- so snap["bub-xrange:value"] is present
# in the store even when the link itself never set it, and "present" alone
# can no longer mean "the link supplied this". For bub-xrange only, a
# supplied value that is JSON-equal to the baked-in system default is
# therefore treated as NOT supplied, which lets the CAGR-view fallback
# ([2025, 2050]) fire on ordinary q4 links. The other four fields have no
# view-dependent fallback, so applying this narrowing to them would only add
# risk for no benefit.
_JS_DEFAULT = """
function(n, snap, view_mode) {
    var NU = window.dash_clientside.no_update;
    if (!n) { return [NU, NU, NU, NU, NU]; }
    var IDS = %(ids)s;
    var DEFAULTS = %(defaults)s;
    var out = [];
    for (var i = 0; i < IDS.length; i++) {
        var k = IDS[i] + ":value";
        var v = snap ? snap[k] : undefined;
        var supplied = (v !== undefined && v !== null);
        if (supplied && IDS[i] === "bub-xrange"
            && JSON.stringify(v) === JSON.stringify(DEFAULTS[k])) {
            supplied = false;   /* q4 back-fills every control; a value equal to
                                   the default is indistinguishable from absent */
        }
        if (supplied) {
            out.push(v);
        } else if (IDS[i] === "bub-xrange" && view_mode === "cagr") {
            out.push(%(cagr_xrange)s);
        } else {
            out.push(DEFAULTS[k]);
        }
    }
    return out;
}
""" % {
    "ids": json.dumps(list(AXES_CONTROL_IDS)),
    "defaults": json.dumps(AXES_DEFAULTS),
    "cagr_xrange": json.dumps(CAGR_DEFAULT_XRANGE),
}

AXES_PRESETS = (
    {"key": "default", "label": "Default",
     "js": _JS_DEFAULT,
     "states": (("snapshot-state-store", "data"), ("bub-view-mode", "data"))},
    {"key": "minus1y", "label": "-1Y",
     "js": _JS_MINUS_1Y, "states": (("bub-xrange", "value"),),
     "hide_js": _JS_HIDE_AT_FLOOR},
    {"key": "cur_year", "label": "Current year",
     "js": _JS_CUR_YEAR, "states": ()},
    {"key": "plus1y", "label": "+1Y",
     "js": _JS_PLUS_1Y, "states": (("bub-xrange", "value"),)},
)


def _bubble_controls():
    yr_now = pd.Timestamp.today().year
    return html.Div([
        _tab_hints("bubble"),
        _section_card("Axes & Range",
            html.Div(id="bub-scale-controls", children=[
                _row(
                    html.Div([_lbl("X scale"), dcc.RadioItems(
                        id="bub-xscale", options=[{"label":"Log","value":"log"},
                                                   {"label":"Linear","value":"linear"}],
                        value=BUBBLE["xscale"], inline=True)]),
                    html.Div([_lbl("Y scale"), dcc.RadioItems(
                        id="bub-yscale", options=[{"label":"Log","value":"log"},
                                                   {"label":"Linear","value":"linear"}],
                        value=BUBBLE["yscale"], inline=True)]),
                ),
            ]),
            _lbl("X range (year)"),
            dcc.RangeSlider(id="bub-xrange", min=XRANGE_MIN, max=XRANGE_MAX,
                            value=[2010, 2033], step=1,
                            marks={y: f"'{y % 100:02d}"
                                   for y in range(XRANGE_MIN, XRANGE_MAX + 1, 10)},
                            tooltip={"always_visible":False}),
            dbc.Row([
                dbc.Col(_lbl("Y range (price)"), width="auto"),
                dbc.Col(dcc.Checklist(
                    id="bub-auto-y",
                    options=[{"label":" Auto","value":"yes"}],
                    value=["yes"], inputStyle=_CB_MARGIN,
                    className="small",
                ), width="auto"),
            ], className="g-0 align-items-center"),
            html.Div(id="bub-yrange-wrap", style=_STYLE_HIDDEN, children=[
                dcc.RangeSlider(id="bub-yrange", min=-2, max=9,
                                value=[-1.5, 6.05], step=0.5,
                                marks={-2:"1\u00a2", 0:"$1", 2:"$100",
                                        4:"$10K", 6:"$1M", 9:"$1B"},
                                tooltip={"always_visible":False}),
            ]),
            _lbl("Presets"),
            html.Div(
                id="bub-axes-presets",
                className="d-flex flex-wrap gap-1",
                children=[
                    dbc.Button(p["label"], id=f"bub-axes-preset-{p['key']}",
                               size="sm", color="secondary", outline=True,
                               className="flex-fill")
                    for p in AXES_PRESETS
                ],
            ),
        ),
        _section_card("Display",
            dcc.Checklist(id="bub-toggles",
                          options=[{"label":" Show price data","value":"show_data"},
                                   {"label":" Shade bands","value":"shade"},
                                   {"label":" Show OLS","value":"show_ols"},
                                   {"label":" Unfairly Cheap Line","value":"show_ucl"},
                                   {"label":" Show today","value":"show_today"},
                                   {"label":" Show legend","value":"show_legend"},
                                   {"label":html.Span(" Minor grid",className="minor-grid-opt"),"value":"minor_grid"},
                                   {"label":" Enable chart zoom","value":"chart_zoom"}],
                          value=["shade","show_data","show_today"],
                          labelStyle={"display":"block"},
                          inputStyle=_CB_MARGIN),
        ),
        display_models_panel("bub", include_bm_master=True,
                              default_value=["bub"],
                              legend_pos_default=BUBBLE["legend_pos"]),
        _q_panel_with_mode("bub-qs", [0.5],
                           hint=f"If none selected, Q50% is shown at "
                                f"{int(_app_ctx.FALLBACK_Q50_OPACITY * 100)}% opacity."),
        # Tab-1 MC: opt-in (default off). Reuses the panel from Tabs 3-5.
        # show_amount/show_inflation/show_stack=False because Tab 1 is
        # price-space (no withdrawal amount, no inflation, no accumulation).
        _mc_controls("bub",
                     show_amount=False,
                     show_inflation=False,
                     show_stack=False,
                     show_mc_entry_q=True,
                     default_entry_q=10,
                     amount_default=100,
                     mc_enabled_default=False,
                     # Tab 1 spaghetti reads sims directly — small defaults
                     # are visually clearer + still free (cache holds 200).
                     # User can type any integer 1-3200 in the input; datalist
                     # shows these presets as autocomplete suggestions.
                     default_sims=8,
                     sims_options=[8, 16, 32, 64, 128, 200, 400, 800, 1600, 3200]),
        sigma_mode_section(),
        custom_time_panel(),
        # Hidden placeholders — bub-bubble-panel and bub-n-future-wrap are
        # referenced as style outputs by the bub-view-mode toggle callbacks.
        # Real Bubble Model controls now live in the global bm-config-modal.
        html.Div(id="bub-bubble-panel", style=_STYLE_HIDDEN),
        html.Div(id="bub-n-future-wrap", style=_STYLE_HIDDEN),
        # Tab-1 MC overlay/badge placeholders. The bubble tab uses
        # _chart_tab_layout_with_fab (not _chart_tab_layout), so these IDs
        # aren't auto-created next to the chart. Hosting them here as hidden
        # placeholders satisfies the match-indicator clientside callback in
        # callbacks/mc_controls.py. Tasks 7-12 will reposition them over
        # the chart for the actual MC visual overlay.
        html.Div(id="bub-mc-overlay", style=_STYLE_HIDDEN,
                 className="mc-chart-overlay"),
        html.Img(id="bub-mc-badge",
                 src="/assets/quantoshi_favicon.png",
                 className="mc-premium-badge",
                 style=_STYLE_HIDDEN),
        _section_card("Model Component Decomposition",
            _lbl("Model"),
            dcc.Dropdown(
                id="bub-decomp-model",
                options=[{"label": "(none)", "value": ""}] +
                        [{"label": label, "value": key}
                         for key, label in _app_ctx.DECOMP_FAMILIES.items()],
                value="", clearable=False,
            ),
            dcc.Checklist(
                id="bub-decomp-show-formulas",
                options=[
                    {"label": " Show full model formula", "value": "full"},
                    {"label": " Show selected formula", "value": "selected"},
                ],
                value=[], inputStyle=_CB_MARGIN,
                labelStyle={"display": "block", "fontSize": UI_FONT_SM, "color": DIM_TEXT},
                style={"marginTop": "4px", "marginBottom": "4px"},
            ),
            html.Div(id="bub-decomp-formula",
                     style={"fontSize": UI_FONT_MD, "marginTop": "6px",
                            "marginBottom": "6px", "overflowX": "auto"}),
            html.Div(id="bub-decomp-active-formula",
                     style={"fontSize": UI_FONT_SM, "marginTop": "4px",
                            "marginBottom": "6px", "padding": "6px",
                            "background": BOOTSTRAP_LIGHT_BG, "borderRadius": "4px",
                            "wordBreak": "break-word", "fontFamily": FONT_MONO}),
            html.Div(id="bub-decomp-body", style=_STYLE_HIDDEN, children=[
                dcc.Checklist(
                    id="bub-decomp-components",
                    options=[], value=[],
                    labelStyle={"display": "block", "fontSize": UI_FONT_MD},
                    inputStyle=_CB_MARGIN,
                ),
                html.Small(
                    "Each checkbox toggles a term on/off. "
                    "log\u2081\u2080(price) = sum of checked terms. "
                    "All checked = full model.",
                    style={"color": FALLBACK_MODEL_GRAY, "fontSize": UI_FONT_SM,
                            "display": "block", "marginTop": "4px"},
                ),
                # Hidden placeholder — preserves bub-decomp-mode snapshot slot.
                dcc.RadioItems(id="bub-decomp-mode", value="individual",
                                style=_STYLE_HIDDEN),
            ]),
            html.Div(id="bub-decomp-warning", children=[]),
            no_hover=True,
        ),
        _section_card(
            html.Span(id="bub-scanner-header",
                      children="Model Scanner \u00b7 Constant \u03c3"),
            _row(
                html.Div([
                    _lbl("Price ($)"),
                    dbc.Input(id="scan-price", type="number",
                              placeholder="live", size="sm", debounce=True),
                    html.Small("\u20bf live", id="scan-price-hint",
                               className="text-muted", style={"fontSize": UI_FONT_XS}),
                ]),
                html.Div([
                    _lbl("Date"),
                    dbc.Input(id="scan-date", type="date",
                              value=pd.Timestamp.today().strftime("%Y-%m-%d"),
                              size="sm", debounce=True),
                ]),
                html.Div([
                    _lbl("Quantile (%)"),
                    dbc.Input(id="scan-q", type="number",
                              min=0.1, max=99.9, step=0.1,
                              size="sm", debounce=True,
                              className="scan-output"),
                ]),
            ),
            dcc.Store(id="scan-output-field", data=["p", "d"]),  # last-2-edited history → q is output
            dcc.Store(id="scan-active-rows", data=[]),
            html.Div(id="scan-results"),
        ),
        _section_card("Plot Appearance",
            _row(
                html.Div([_lbl("Pt size (1\u201320)"),
                          dbc.Input(id="bub-ptsize", type="number",
                                    value=BUBBLE["pt_size"], min=1, max=20, size="sm")]),
                html.Div([_lbl("Pt alpha (0.1\u20131)"),
                          dbc.Input(id="bub-ptalpha", type="number",
                                    value=BUBBLE["pt_alpha"], min=0.1, max=1.0, step=0.05, size="sm")]),
            ),
            *_plot_appearance_controls("bub"),
        ),
        _section_card("Stack (BTC)",
            dbc.InputGroup([
                dbc.Input(id="bub-stack", type="number", value=BUBBLE["stack"],
                          min=0, step=0.001, size="sm", debounce=True),
                dbc.InputGroupText(dcc.Checklist(
                    id="bub-show-stack",
                    options=[{"label":" Show","value":"yes"}],
                    value=[], inputStyle=_CB_MARGIN)),
            ], size="sm"),
            _use_lots_checklist("bub"),
        ),
        html.Div(id="um-panel-wrap", style=_STYLE_HIDDEN, children=[
            _section_card("User Model (U\u2081)",
                html.Small("Click a data point on the chart, then tap P1 or P2. "
                           "Model auto-draws when both are set.",
                           style={"color":FALLBACK_MODEL_GRAY, "fontSize": UI_FONT_MD}),
                html.Div([
                    html.Label("Point 1", style={"fontWeight":"600", "fontSize": UI_FONT_BASE, "marginTop":"6px"}),
                    html.Div([
                        _lbl("Year"), html.Span(id="um-p1-year-display",
                            style={"display":"inline-block", "width":"80px", "fontSize": UI_FONT_BASE,
                                   "fontWeight":"600", "color":USER_MODEL_TRACE,
                                   "backgroundColor":_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BG_ALPHA),
                                   "border":f"1px solid {_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BORDER_ALPHA)}",
                                   "borderRadius":"4px", "padding":"2px 6px",
                                   "minHeight":"24px", "verticalAlign":"middle"},
                            children="\u2014"),
                        html.Span(" ", style={"width":"6px", "display":"inline-block"}),
                        _lbl("$"), html.Span(id="um-p1-price-display",
                            style={"display":"inline-block", "width":"100px", "fontSize": UI_FONT_BASE,
                                   "fontWeight":"600", "color":USER_MODEL_TRACE,
                                   "backgroundColor":_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BG_ALPHA),
                                   "border":f"1px solid {_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BORDER_ALPHA)}",
                                   "borderRadius":"4px", "padding":"2px 6px",
                                   "minHeight":"24px", "verticalAlign":"middle"},
                            children="\u2014"),
                    ], style={"display":"flex", "alignItems":"center", "gap":"4px", "flexWrap":"wrap"}),
                    html.Label("Point 2", style={"fontWeight":"600", "fontSize": UI_FONT_BASE, "marginTop":"6px"}),
                    html.Div([
                        _lbl("Year"), html.Span(id="um-p2-year-display",
                            style={"display":"inline-block", "width":"80px", "fontSize": UI_FONT_BASE,
                                   "fontWeight":"600", "color":USER_MODEL_TRACE,
                                   "backgroundColor":_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BG_ALPHA),
                                   "border":f"1px solid {_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BORDER_ALPHA)}",
                                   "borderRadius":"4px", "padding":"2px 6px",
                                   "minHeight":"24px", "verticalAlign":"middle"},
                            children="\u2014"),
                        html.Span(" ", style={"width":"6px", "display":"inline-block"}),
                        _lbl("$"), html.Span(id="um-p2-price-display",
                            style={"display":"inline-block", "width":"100px", "fontSize": UI_FONT_BASE,
                                   "fontWeight":"600", "color":USER_MODEL_TRACE,
                                   "backgroundColor":_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BG_ALPHA),
                                   "border":f"1px solid {_hex_alpha(USER_MODEL_TRACE, UM_INPUT_BORDER_ALPHA)}",
                                   "borderRadius":"4px", "padding":"2px 6px",
                                   "minHeight":"24px", "verticalAlign":"middle"},
                            children="\u2014"),
                    ], style={"display":"flex", "alignItems":"center", "gap":"4px", "flexWrap":"wrap"}),
                ]),
                dcc.Store(id="um-p1-year", data=None),
                dcc.Store(id="um-p1-price", data=None),
                dcc.Store(id="um-p2-year", data=None),
                dcc.Store(id="um-p2-price", data=None),
                # Draw button hidden (auto-draws on P2 set) but kept for callback compatibility
                html.Div(id="um-draw-btn", style=_STYLE_HIDDEN),
                dbc.Button("\u2715 Clear", id="um-delete-btn", color="secondary", size="sm",
                           outline=True, style={"marginTop":"8px"}),
            ),
        ]),
        _palette_selector("bub"),
    ])


def _bubble_tab():
    from layout.common import _chart_tab_layout_with_fab
    return _chart_tab_layout_with_fab(_bubble_controls, "bubble-graph", "btc_bubble")
