"""Tab 3 — BTC Accumulator (DCA) and Tab 4 — BTC RetireMentator layout."""

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from layout.common import (_tab_hints, _section_card, _lbl,
                            _STYLE_HIDDEN, _STYLE_HINT,
                            _q_options, _model_show_checklist,
                            _shared_settings_card, _year_range_slider,
                            _btc_usd_dropdown, _chart_toggles,
                            _legend_pos_dropdown, _ctrl_card,
                            _chart_tab_layout)
from layout.mc_controls import _mc_controls


def _accum_withdraw_controls(prefix, tab_key, q_hint, q_defaults,
                              shared_kwargs, mc_kwargs, yr_range,
                              chart_toggle_defaults, btc_usd_kwargs=None,
                              extra_sections=None):
    """Shared builder for DCA (tab 3) and Retire (tab 4) controls."""
    children = [
        _tab_hints(tab_key),
        _shared_settings_card(prefix, **shared_kwargs),
        _section_card("Quantile Regression Model",
            html.Small("Select quantiles to follow.", style=_STYLE_HINT),
            html.Small(q_hint, style=_STYLE_HINT),
            dcc.Checklist(id=f"{prefix}-qs", options=_q_options(),
                          value=q_defaults, className="q-panel-grid",
                          inputStyle={"marginRight":"4px"}),
        ),
    ]
    if extra_sections:
        children.extend(extra_sections)
    children.append(_mc_controls(prefix, **mc_kwargs))
    yr_now = pd.Timestamp.today().year
    children.append(
        _section_card("Chart Settings",
            *_model_show_checklist(prefix),
            _lbl("Year range"),
            _year_range_slider(prefix, *yr_range),
            _btc_usd_dropdown(prefix, **(btc_usd_kwargs or {})),
            _chart_toggles(prefix, chart_toggle_defaults),
            *_legend_pos_dropdown(prefix, "bottom-right"),
        ),
    )
    return html.Div(children)


def _stackcelerator_controls():
    return _ctrl_card(
        html.B("Stack-celerator", style={"fontSize":"12px"}),
        dcc.Checklist(id="dca-sc-enable",
                      options=[{"label":" Activate Saylor Mode","value":"yes"}],
                      value=[], inputStyle={"marginRight":"5px"}),
        # Why html.Div(display:none) instead of dbc.Collapse: Dash Collapse
        # unmounts its children, destroying component state on toggle.
        html.Div(id="dca-sc-body", style=_STYLE_HIDDEN, children=[
            _lbl("Loan amount ($)"),
            dbc.Input(id="dca-sc-loan", type="number",
                      value=1200, min=0, max=_app_ctx.MAX_USD, step=1, size="sm",
                      debounce=True),
            _lbl("Entry price (1st cycle)"),
            dcc.Dropdown(id="dca-sc-entry-mode",
                         options=[{"label":"Live ticker","value":"live"},
                                  {"label":"Model price","value":"model"},
                                  {"label":"Custom price","value":"custom"}],
                         value="live", clearable=False),
            html.Div(id="dca-sc-custom-price-row", style=_STYLE_HIDDEN, children=[
                _lbl("Custom entry price ($)"),
                dbc.Input(id="dca-sc-custom-price", type="number",
                          value=_app_ctx.SC_DEFAULT_PRICE, min=1, step=1, size="sm",
                          debounce=True),
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
            _lbl("Annual interest rate (0\u2013100% / yr)"),
            dbc.Input(id="dca-sc-rate", type="number",
                      value=_app_ctx.SC_DEFAULT_RATE, min=0, max=100, step=0.5, size="sm",
                      debounce=True),
            _lbl("Loan term (months)"),
            dbc.Input(id="dca-sc-term", type="number",
                      value=_app_ctx.SC_DEFAULT_TERM, min=1, max=360, step=1, size="sm",
                      debounce=True),
            _lbl("Additional loan cycles (0 = one loan)"),
            dbc.Input(id="dca-sc-repeats", type="number",
                      value=0, min=0, step=1, size="sm",
                      debounce=True),
            _lbl("Capital gains tax on repayment (0\u2013100%)"),
            dbc.Input(id="dca-sc-tax", type="number",
                      value=_app_ctx.SC_DEFAULT_TAX, min=0, max=99, step=0.5, size="sm",
                      debounce=True),
            html.Div(id="dca-sc-info",
                     style={"fontSize":"11px","color":"#555","marginTop":"4px"}),
        ]),
    )


def _dca_controls():
    yr_now = pd.Timestamp.today().year
    return _accum_withdraw_controls(
        "dca", "dca",
        q_hint="Price path drives sat accumulation \u2014 lower quantile = lower price = more sats/period.",
        q_defaults=[0.5],
        shared_kwargs=dict(amount_id="dca-amount", amount_label="Per-period amount ($)",
                           amount_default=100, infl_default=0, stack_default=0),
        mc_kwargs=dict(amount_label="DCA amount per period ($)", amount_default=100,
                       show_inflation=True, show_stack=True,
                       show_mc_entry_q=True, default_entry_q=10,
                       shared_controls={"amount", "infl", "freq", "stack"}),
        yr_range=(2009, 2080, yr_now, yr_now + 10),
        chart_toggle_defaults=["annotate"],
        extra_sections=[_stackcelerator_controls()],
    )


def _dca_tab():
    return _chart_tab_layout(_dca_controls, "dca-graph", "btc_dca", mc_prefix="dca")


def _retire_controls():
    return _accum_withdraw_controls(
        "ret", "retire",
        q_hint="Lower quantile = lower price = faster depletion \u2014 worst-case planning.",
        q_defaults=[0.01, 0.10, 0.25],
        shared_kwargs=dict(amount_id="ret-wd", amount_label="Withdrawal/period ($)",
                           amount_default=5000, infl_default=4, stack_default=1.0),
        mc_kwargs=dict(amount_label="Withdrawal per period ($)", amount_default=5000,
                       show_inflation=True, show_stack=True, default_entry_q=10,
                       start_yr_label="Retirement start year",
                       shared_controls={"amount", "infl", "freq", "stack"}),
        yr_range=(2024, 2080, 2031, 2075),
        chart_toggle_defaults=["annotate", "log_y", "minor_grid"],
        btc_usd_kwargs={"btc_label": "BTC Remaining"},
    )


def _retire_tab():
    return _chart_tab_layout(_retire_controls, "retire-graph", "btc_retire", mc_prefix="ret")
