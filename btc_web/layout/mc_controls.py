"""Monte Carlo simulation controls — shared across DCA, Retire, Heatmap, Supercharger tabs."""

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from mc_cache import (CACHED_START_YRS, WD_AMOUNTS,
                      ENTRY_PCT_BINS, MC_YEARS_OPTIONS, INFL_OPTIONS)
from mc_overlay import bin_regime_labels

from layout.common import (_section_card, _ctrl_card, _row, _lbl,
                            _STYLE_HIDDEN, _STYLE_HINT)

_QUANT_FONT = {"fontFamily": '"Palatino Linotype", Palatino, "Book Antiqua", serif',
               "color": "#000", "letterSpacing": "1px"}
_MC_CACHED_START_YRS = set(CACHED_START_YRS)
_MC_CACHED_ENTRY_QS = {int(v * 100) for v in ENTRY_PCT_BINS}   # {10,20,...,90}
_MC_CACHED_YEARS    = set(MC_YEARS_OPTIONS)                      # {10,20,30,40}
_MC_CACHED_WD       = set(WD_AMOUNTS)
_MC_CACHED_INFL     = set(INFL_OPTIONS)


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
_MC_PRICE_LIVE   = {10: 500, 20: 1000, 30: 1500, 40: 2000}
_MC_START_YR_OPTIONS = _bold_opts(range(2026, 2051), str, _MC_CACHED_START_YRS)
# Entry Q dropdown: show 1%–90% in 10% steps, with 1%/10%/50% bolded
_MC_ENTRY_Q_VALS = sorted(set([1] + [int(v * 100) for v in ENTRY_PCT_BINS] + list(range(10, 100, 10))))
_MC_ENTRY_Q_OPTIONS  = _bold_opts(
    _MC_ENTRY_Q_VALS,
    lambda v: f"{v}%", _MC_CACHED_ENTRY_QS)
_MC_ENTRY_Q_OPTIONS_ADV = _bold_opts(
    [round(i / 10, 1) for i in range(1, 1000)],
    lambda v: f"{v}%", _MC_CACHED_ENTRY_QS)
_MC_YEARS_OPTIONS    = _bold_opts([10, 20, 30, 40], lambda v: f"{v} yr", _MC_CACHED_YEARS)
_MC_WD_OPTIONS       = _bold_opts(WD_AMOUNTS, lambda v: f"${v:,}/mo", _MC_CACHED_WD)
_MC_INFL_OPTIONS     = _bold_opts(INFL_OPTIONS, lambda v: f"{v}%", _MC_CACHED_INFL)

def _regime_options(n_bins=5):
    """Build checklist options for the bin regime filter."""
    labels = bin_regime_labels(n_bins)
    return [{"label": labels[i], "value": i} for i in range(n_bins)]

_MC_REGIME_OPTIONS_5 = _regime_options(5)   # pre-computed for default 5-bin case

def _mc_controls(prefix, amount_label="Per-period amount ($)", amount_default=100,
                  show_inflation=False, show_amount=True,
                  show_stack=False, show_mc_entry_q=False,
                  default_entry_q=50, start_yr_label=None,
                  shared_controls=frozenset()):
    """Monte Carlo simulation controls, reusable across tabs."""
    yr_now = pd.Timestamp.today().year
    if not _app_ctx._HAS_MARKOV:
        # Hidden placeholders so callback IDs exist even without markov module
        _ph = []  # placeholder components
        _ph.append(dcc.Checklist(id=f"{prefix}-mc-enable", value=[]))
        _ph.append(dcc.Checklist(id=f"{prefix}-mc-advanced", value=[]))
        if "amount" not in shared_controls:
            _ph.append(dbc.Input(id=f"{prefix}-mc-amount", value=amount_default))
        if "infl" not in shared_controls:
            _ph.append(dbc.Input(id=f"{prefix}-mc-infl", value=4))
        _ph.append(dbc.Input(id=f"{prefix}-mc-bins", value=5))
        _ph.append(dcc.Checklist(id=f"{prefix}-mc-regime", value=list(range(5))))
        _ph.append(dcc.Dropdown(id=f"{prefix}-mc-sims", value=200))
        _ph.append(dcc.Dropdown(id=f"{prefix}-mc-years", value=40))
        if "freq" not in shared_controls:
            _ph.append(dcc.Dropdown(id=f"{prefix}-mc-freq", value="Monthly"))
            _ph.append(dbc.Input(id=f"{prefix}-mc-ppy", value="12/yr"))
        _ph.append(dcc.RangeSlider(id=f"{prefix}-mc-window", min=2010,
                                    max=yr_now, value=[2010, yr_now]))
        _ph.append(html.Div(id=f"{prefix}-mc-adv-body"))
        _ph.append(html.Div(id=f"{prefix}-mc-cost"))
        _ph.append(dcc.Store(id=f"{prefix}-mc-price-val", storage_type="memory", data=0))
        _ph.append(html.Div(id=f"{prefix}-mc-body"))
        _ph.append(html.Div(id=f"{prefix}-mc-status"))
        _ph.append(dbc.Button(id=f"{prefix}-mc-dl-btn", style=_STYLE_HIDDEN))
        _ph.append(dbc.Button(id=f"{prefix}-mc-run-btn", style=_STYLE_HIDDEN))
        _ph.append(html.Div(id=f"{prefix}-mc-run-status"))
        _ph.append(dcc.Store(id=f"{prefix}-mc-rendered-key", storage_type="memory"))
        _ph.append(html.Div(id=f"{prefix}-mc-match"))
        _ph.append(dbc.Button(id=f"{prefix}-mc-restore-btn", style=_STYLE_HIDDEN))
        _ph.append(dcc.Upload(id=f"{prefix}-mc-upload"))
        _ph.append(html.Div(id=f"{prefix}-mc-upload-status"))
        _ph.append(dcc.Slider(id=f"{prefix}-mc-entry-yr", value=yr_now))
        _ph.append(dcc.Dropdown(id=f"{prefix}-mc-entry-q",
                    value=max(10, min(90, round(_app_ctx._HM_ENTRY_Q_DEFAULT / 10) * 10 or 50))))
        if "stack" not in shared_controls:
            _ph.append(dbc.Input(id=f"{prefix}-mc-stack", type="number", value=1.0))
        _ph.append(dcc.Dropdown(id=f"{prefix}-mc-start-yr", value=2031))
        _ph.append(dcc.Dropdown(id=f"{prefix}-mc-model-src", value="bub"))
        return html.Div(style=_STYLE_HIDDEN, children=_ph)
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
            html.B("Monte Carlo Simulation", style={"fontSize": "12px"}),
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
        html.Div(id=f"{prefix}-mc-body", style=_STYLE_HIDDEN, children=[
            dcc.Checklist(id=f"{prefix}-mc-advanced",
                          options=[{"label": " Advanced simulator options", "value": "yes"}],
                          value=[], inputStyle={"marginRight": "5px"},
                          style={"fontSize": "11px", "color": "#666", "marginBottom": "6px"}),
            html.Div(dcc.Slider(id=f"{prefix}-mc-entry-yr", value=yr_now),
                     style=_STYLE_HIDDEN),
            _lbl((start_yr_label or "MC start year") + " (bold = cached)"),
            dcc.Dropdown(id=f"{prefix}-mc-start-yr",
                         options=_MC_START_YR_OPTIONS,
                         value=2031 if show_stack else 2028, clearable=False),
            _lbl("Entry percentile (10% steps, cache-aligned)"),
            dcc.Dropdown(id=f"{prefix}-mc-entry-q",
                         options=_MC_ENTRY_Q_OPTIONS,
                         value=default_entry_q, clearable=False),
            _lbl("Model source (quantile bands)"),
            dcc.Dropdown(id=f"{prefix}-mc-model-src",
                         options=[{"label": f" {mdl.name}", "value": mdl.short_name}
                                  for mdl in _app_ctx.PRICE_MODELS.values()
                                  if mdl.quantized],
                         value="bub", clearable=False),
            *([_lbl(amount_label + " (0–4,294,967,295)"),
               dbc.Input(id=f"{prefix}-mc-amount", type="number",
                         value=amount_default, min=0, max=_app_ctx.MAX_USD,
                         step=1, size="sm", debounce=True),
              ] if show_amount and "amount" not in shared_controls else (
              [dbc.Input(id=f"{prefix}-mc-amount", type="number",
                         value=amount_default, style=_STYLE_HIDDEN),
              ] if "amount" not in shared_controls else [])),
            *([ _lbl("Inflation rate (% / yr)"),
                dcc.Dropdown(id=f"{prefix}-mc-infl",
                             options=_MC_INFL_OPTIONS,
                             value=4, clearable=False),
            ] if show_inflation and "infl" not in shared_controls else (
            [   dcc.Dropdown(id=f"{prefix}-mc-infl", value=0,
                             style=_STYLE_HIDDEN),
            ] if "infl" not in shared_controls else [])),
            *([ _lbl("Starting BTC stack (0–1000)"),
                dbc.Input(id=f"{prefix}-mc-stack", type="number",
                          min=0, max=1000, step=0.01, value=1.0, size="sm",
                          debounce=True),
            ] if show_stack and "stack" not in shared_controls else (
            [   dbc.Input(id=f"{prefix}-mc-stack", type="number",
                          value=1.0, style=_STYLE_HIDDEN),
            ] if "stack" not in shared_controls else [])),
            _lbl("Years to model"),
            dcc.Dropdown(id=f"{prefix}-mc-years",
                         options=_MC_YEARS_OPTIONS,
                         value=40, clearable=False),
            # Advanced controls (hidden until checkbox toggled)
            html.Div(id=f"{prefix}-mc-adv-body", style=_STYLE_HIDDEN, children=[
                _lbl("Markov transition matrix dimension"),
                dcc.Dropdown(id=f"{prefix}-mc-bins",
                             options=_bold_opts(
                                 list(range(5, 11)),
                                 lambda v: f"{v}\u00d7{v}", {5}),
                             value=5, clearable=False),
                _lbl("Price regime filter"),
                dcc.Checklist(
                    id=f"{prefix}-mc-regime",
                    options=_MC_REGIME_OPTIONS_5,
                    value=list(range(5)),
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"display": "block", "fontSize": "11px",
                                "lineHeight": "1.6", "color": "#444"},
                    style={"marginBottom": "6px"},
                ),
                _lbl("Simulations"),
                dcc.Dropdown(id=f"{prefix}-mc-sims",
                             options=_bold_opts(
                                 [100, 200, 400, 800, 1600, 3200],
                                 str, {200}),
                             value=200, clearable=False),
                *([ _lbl("Frequency"),
                    dcc.Dropdown(id=f"{prefix}-mc-freq",
                                 options=_bold_opts(
                                     ["Daily", "Weekly", "Monthly", "Quarterly", "Annually"],
                                     str, {"Monthly"}),
                                 value="Monthly", clearable=False),
                    _lbl("Periods per year"),
                    dbc.Input(id=f"{prefix}-mc-ppy", value="12/yr", size="sm",
                              disabled=True),
                ] if "freq" not in shared_controls else []),
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
                [html.Span("\u26a1 ", style={"fontSize": "14px"}), "Run MC Simulation"],
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
            dbc.Button("\u21a9 Restore last settings",
                       id=f"{prefix}-mc-restore-btn",
                       color="warning", size="sm", outline=True,
                       className="w-100 mt-1",
                       style=_STYLE_HIDDEN),
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
