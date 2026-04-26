"""Monte Carlo simulation controls — shared across DCA, Retire, Heatmap, Supercharger tabs."""

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from mc_cache import (CACHED_START_YRS, WD_AMOUNTS,
                      ENTRY_PCT_BINS, MC_YEARS_OPTIONS, INFL_OPTIONS,
                      _CACHED_MODEL_KEYS)
from mc_overlay import bin_regime_labels

from layout.common import (_section_card, _ctrl_card, _row, _lbl,
                            _STYLE_HIDDEN, _STYLE_HINT, _CB_MARGIN)
from colors import (BLACK, SILVER, NEAR_BLACK, MUTED_TEXT,
                    MODAL_DIVIDER_DARK, DIM_TEXT, WHITE, BADGE_GLOW_RED, _hex_alpha,
                    FONT_BRAND, UI_FONT_SM, UI_FONT_MD, UI_FONT_LG, UI_FONT_XL, UI_FONT_XXL,
                    CTX_MENU_SHADOW_ALPHA, BADGE_INSET_ALPHA, BADGE_GLOW_ALPHA)

_QUANT_FONT = {"fontFamily": FONT_BRAND,
               "color": BLACK, "letterSpacing": "1px"}


def _mc_model_src_options():
    """Curated master-only set for the MC quantile-bands dropdown.

    Mirrors the heatmap pill set (`layout.heatmap._HM_PILL_MODELS_BASE`,
    minus 'mc') so configurable families (LPPL/HybPPL/EPPL) appear once
    as a master entry. The specific config variant is picked up from the
    global LPPL/HybPPL/EPPL config modals via `_resolve_mc_model_src`
    in the callbacks before `_mc_setup` runs.

    Models present in the precomputed MC cache (free tier) render bold;
    others require live compute (paid).
    """
    from layout.heatmap import _HM_PILL_MODELS_BASE
    keys = list(_HM_PILL_MODELS_BASE)
    if "ef" in _app_ctx.PRICE_MODELS:
        keys.append("ef")
    if "u1" in _app_ctx.PRICE_MODELS:
        keys.append("u1")
    out = []
    for k in keys:
        if k not in _app_ctx.PRICE_MODELS:
            continue
        name = _app_ctx.PRICE_MODELS[k].name
        if k in _CACHED_MODEL_KEYS:
            label = html.Span(f" {name}", style={"fontWeight": "bold"})
        else:
            label = f" {name}"
        out.append({"label": label, "value": k})
    return out
_MC_CACHED_START_YRS = set(CACHED_START_YRS)
_MC_CACHED_ENTRY_QS = {int(v * 100) for v in ENTRY_PCT_BINS}   # {10,20,...,90}
_MC_CACHED_YEARS    = set(MC_YEARS_OPTIONS)                      # {10,20,30,40}
_MC_CACHED_WD       = set(WD_AMOUNTS)
_MC_CACHED_INFL     = set(INFL_OPTIONS)


def _bold_opts(values, fmt, cached_set):
    """Build dropdown options, bolding+enlarging values in the pre-computed cache."""
    return [
        {"label": html.Span(fmt(v), style={"fontWeight": "bold", "fontSize": UI_FONT_XXL})
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

def _mc_controls(prefix, amount_label="Purchase amount ($)", amount_default=100,
                  show_inflation=False, show_amount=True,
                  show_stack=False, show_mc_entry_q=False,
                  default_entry_q=50, start_yr_label=None,
                  shared_controls=frozenset(), mc_enabled_default=False,
                  default_sims=200, sims_options=None):
    """Monte Carlo simulation controls, reusable across tabs."""
    yr_now = pd.Timestamp.today().year
    _mc_enable_val = ["yes"] if mc_enabled_default else []
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
        _ph.append(dcc.Dropdown(id=f"{prefix}-mc-sims", value=default_sims))
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
        _ph.append(dbc.Button(id={"type": "mc-run-btn", "tab": prefix}, style=_STYLE_HIDDEN))
        _ph.append(html.Div(id={"type": "mc-run-status", "tab": prefix}))
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
            html.Span("\u2694", style={"fontSize": UI_FONT_XXL, "marginRight": "3px"}),
            html.Span("NEW", style={"position": "relative", "top": "-2px"}),
        ], className="mc-new-badge", style={
            "position": "absolute", "top": "4px", "right": "-2px",
            "fontWeight": "900", "color": SILVER,
            "fontFamily": "'Impact', 'Arial Black', sans-serif",
            "textTransform": "uppercase",
            "backgroundColor": NEAR_BLACK,
            "borderRadius": "5px", "transform": "rotate(18deg)",
            "zIndex": "1", "lineHeight": "1.2",
            "boxShadow": f"0 2px 6px {_hex_alpha(BLACK, CTX_MENU_SHADOW_ALPHA)}, inset 0 1px 0 {_hex_alpha(WHITE, BADGE_INSET_ALPHA)}",
            "textShadow": f"0 0 4px {_hex_alpha(BADGE_GLOW_RED, BADGE_GLOW_ALPHA)}",
        }),
        _section_card(
        "Monte Carlo Simulation",
        html.Div(id=f"{prefix}-mc-body", style=_STYLE_HIDDEN, children=[
            dcc.Checklist(id=f"{prefix}-mc-advanced",
                          options=[{"label": " Advanced simulator options", "value": "yes"}],
                          value=[], inputStyle=_CB_MARGIN,
                          style={"fontSize": UI_FONT_MD, "color": MUTED_TEXT, "marginBottom": "6px"}),
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
                         options=_mc_model_src_options(),
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
                    inputStyle=_CB_MARGIN,
                    labelStyle={"display": "block", "fontSize": UI_FONT_MD,
                                "lineHeight": "1.6", "color": MODAL_DIVIDER_DARK},
                    style={"marginBottom": "2px"},
                ),
                html.Small(
                    "Free-tier: cached paths sorted by alignment with your "
                    "selection (each path may visit blocked regimes briefly). "
                    "Filter only materially changes output when sims < 200 "
                    "(cache size); at 200, paths are reordered but the same "
                    "set is used — percentiles unchanged. Paid live MC: paths "
                    "cannot leave selected regimes.",
                    style={"display": "block", "fontSize": UI_FONT_SM,
                           "color": DIM_TEXT, "lineHeight": "1.3",
                           "marginBottom": "8px"},
                ),
                _lbl("Simulations (1–3200; ≤200 is free)"),
                # Number input + HTML5 datalist: type any integer 1-3200,
                # or pick from the preset list. ≤200 is free-tier (cache
                # holds 200 paths). >200 triggers paid flow per is_free_tier.
                html.Datalist(
                    id=f"{prefix}-mc-sims-presets",
                    children=[html.Option(value=str(v))
                              for v in (sims_options
                                        or [8, 16, 32, 64, 128, 200, 400, 800, 1600, 3200])],
                ),
                dbc.Input(id=f"{prefix}-mc-sims", type="number",
                          value=default_sims,
                          min=1, max=3200, step=1, size="sm",
                          debounce=True, list=f"{prefix}-mc-sims-presets"),
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
                     style={"fontSize": UI_FONT_MD, "color": DIM_TEXT, "marginTop": "6px",
                            "lineHeight": "1.4"}),
            dcc.Store(id=f"{prefix}-mc-price-val", storage_type="memory", data=0),
            # ── Run Simulation button (payment-gated when BTCPay active) ──
            dbc.Button(
                [html.Span("\u26a1 ", style={"fontSize": UI_FONT_XL}), "Run MC Simulation"],
                id={"type": "mc-run-btn", "tab": prefix}, size="sm", color="warning",
                className="w-100 mt-2",
                style={"fontWeight": "600"},
            ),
            html.Div(id={"type": "mc-run-status", "tab": prefix},
                     style={"fontSize": UI_FONT_SM, "color": DIM_TEXT, "marginTop": "4px",
                            "textAlign": "center"}),
            dcc.Store(id=f"{prefix}-mc-rendered-key", storage_type="memory"),
            html.Div(id=f"{prefix}-mc-match",
                     style={"fontSize": UI_FONT_SM, "marginTop": "4px",
                            "textAlign": "center"}),
            dbc.Button("\u21a9 Restore last settings",
                       id=f"{prefix}-mc-restore-btn",
                       color="warning", size="sm", outline=True,
                       className="w-100 mt-1",
                       style=_STYLE_HIDDEN),
            html.Hr(className="my-2"),
            _section_card("Saved Simulation",
                html.Div(id=f"{prefix}-mc-status",
                         style={"fontSize": UI_FONT_SM, "color": DIM_TEXT, "marginBottom": "4px"}),
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
                         style={"fontSize": UI_FONT_SM}),
            ),
        ]),
        header_right=[
            dcc.Checklist(
                id=f"{prefix}-mc-enable",
                options=[{"label": " Activate", "value": "yes"}],
                value=_mc_enable_val, inputStyle=_CB_MARGIN,
                className="model-panel-activate",
            ),
            html.Span([
                html.Span("\u26a1", style={"fontSize": UI_FONT_LG}),
                " Paid",
            ], className="model-panel-paid-badge"),
        ],
    ),
    ])
