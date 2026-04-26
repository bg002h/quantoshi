"""Tab 5 — HODL Supercharger layout."""

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
from utils import _nearest_quantile
from tab_defaults import SUPERCHARGE
from layout.common import (_tab_hints, _section_card, _lbl, _palette_selector,
                            _STYLE_HINT, _q_options, _q_panel_with_mode,
                            _shared_settings_card,
                            _btc_usd_dropdown, _chart_toggles,
                            _chart_tab_layout,
                            _CB_MARGIN, _Q_HINT_BASE, _plot_appearance_controls,
                            _display_card_custom)
from layout.display_models import display_models_panel
from layout.mc_controls import _mc_controls
from colors import STATIC_PAGE_MUTED, UI_FONT_SM


def _supercharge_controls():
    yr_now = pd.Timestamp.today().year
    display_q_opts = _q_options()
    display_q_default = _nearest_quantile(SUPERCHARGE["display_q"], _app_ctx._ALL_QS)
    return html.Div([
        _tab_hints("supercharge"),
        _shared_settings_card("sc", infl_default=SUPERCHARGE["inflation"], stack_default=SUPERCHARGE["start_stack"]),
        # ── Plan — strategy (mode + delays; mode-conditional withdrawal +
        # horizon). sc-start-yr (base retirement year) moved to Axes &
        # Range below since it's mode-agnostic and maps naturally onto the
        # chart's x-axis start. End/target years stay here because they
        # are mode-conditional.
        _section_card("Plan",
            dcc.RadioItems(id="sc-mode",
                options=[{"label":" A \u2014 Fixed spending (depletion date)","value":"a"},
                         {"label":" B \u2014 Fixed depletion (max spending)","value":"b"}],
                value=SUPERCHARGE["mode"], labelStyle={"display":"block"},
                inputStyle=_CB_MARGIN),
            dbc.Collapse(
                html.Div(
                    "\u2248YYYY annotations mark the year each scenario\u2019s BTC stack reaches zero \u2014 savings exhausted.",
                    style={"fontSize": UI_FONT_SM,"color":STATIC_PAGE_MUTED,"marginTop":"6px",
                           "lineHeight":"1.4"},
                ),
                id="sc-depl-note-collapse", is_open=True,
            ),
            html.Hr(className="my-2"),
            _lbl("Delay offsets (years)"),
            dbc.Row([
                dbc.Col(dbc.Input(id="sc-d0", type="number", value=SUPERCHARGE["delays"][0],
                                  min=0, step=1, size="sm"), width=True),
                dbc.Col(dbc.Input(id="sc-d1", type="number", value=SUPERCHARGE["delays"][1],
                                  min=0, step=1, size="sm"), width=True),
                dbc.Col(dbc.Input(id="sc-d2", type="number", value=SUPERCHARGE["delays"][2],
                                  min=0, step=1, size="sm"), width=True),
                dbc.Col(dbc.Input(id="sc-d3", type="number", value=SUPERCHARGE["delays"][3],
                                  min=0, step=1, size="sm"), width=True),
                dbc.Col(dbc.Input(id="sc-d4", type="number", value=SUPERCHARGE["delays"][4],
                                  min=0, step=1, size="sm"), width=True),
            ], className="g-1"),
            html.Hr(className="my-2"),
            dbc.Collapse([
                _lbl("Withdrawal amount ($)"),
                dbc.Input(id="sc-wd", type="number", value=SUPERCHARGE["wd_amount"],
                          min=0, max=_app_ctx.MAX_USD, step=1, size="sm",
                          debounce=True),
                _lbl("End year"),
                html.Div(dcc.Slider(id="sc-end-yr", min=2030, max=2100,
                           value=SUPERCHARGE["end_yr"], step=1,
                           marks={y: f"'{y % 100:02d}" for y in range(2030, 2101, 10)},
                           tooltip={"always_visible":False})),
            ], id="sc-mode-a-collapse", is_open=True),
            dbc.Collapse([
                _lbl("Target depletion year"),
                html.Div(dcc.Slider(id="sc-target-yr", min=2030, max=2100,
                           value=SUPERCHARGE["target_yr"], step=1,
                           marks={y: f"'{y % 100:02d}" for y in range(2030, 2101, 10)},
                           tooltip={"always_visible":False})),
            ], id="sc-mode-b-collapse", is_open=False),
        ),
        _mc_controls("sc", amount_label="Withdrawal amount per period ($)",
                     amount_default=5000, show_inflation=True,
                     show_stack=True, default_entry_q=10,
                     start_yr_label="Withdrawal start year",
                     shared_controls={"amount", "infl", "freq", "stack"},
                     mc_enabled_default=True),
        display_models_panel("sc", legend_pos_default=SUPERCHARGE["legend_pos"],
                             include_mc=True),
        _q_panel_with_mode("sc-qs",
                           [0.15, 0.85],
                           hint="Lower prices mean earlier depletion."),
        # ── Axes & Range — sc-start-yr (base retirement year = chart
        # x-axis start) + disp_mode. End/target years remain in Plan
        # because they're mode-conditional.
        _section_card("Axes & Range",
            _lbl("Base retirement year"),
            dcc.Slider(id="sc-start-yr", min=yr_now, max=2075,
                       value=SUPERCHARGE["start_yr"], step=1,
                       marks={y: f"'{y % 100:02d}"
                              for y in range(yr_now, 2076) if y % 10 == 0},
                       tooltip={"always_visible":False}),
            _btc_usd_dropdown("sc", btc_label="BTC Remaining", default="usd"),
        ),
        # ── Display (toggles + shade layout + collapsed display-q) ──────
        _display_card_custom(
            "sc",
            _chart_toggles("sc", ["annotate", "log_y", "shade"]),
            extra_children=[
                dcc.Checklist(id="sc-chart-layout",
                    options=[{"label":" Shade quantile bands","value":"shade"}],
                    value=["shade"],
                    inputStyle=_CB_MARGIN),
                dbc.Collapse([
                    _lbl("Display quantile"),
                    dcc.Dropdown(id="sc-display-q", options=display_q_opts,
                                 value=display_q_default, clearable=False),
                ], id="sc-display-q-collapse", is_open=True),
            ],
        ),
        _section_card("Plot Appearance", *_plot_appearance_controls("sc")),
        _palette_selector("sc"),
    ])


def _supercharge_tab():
    return _chart_tab_layout(_supercharge_controls, "supercharge-graph", "btc_supercharge",
                             mc_prefix="sc")
