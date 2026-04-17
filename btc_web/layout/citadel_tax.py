"""Citadel Planner — Tax configuration modal and toggle widgets."""

from dash import dcc, html
import dash_bootstrap_components as dbc

from layout.common import _CB_MARGIN, _lbl, _STYLE_HINT
from engines.tax_data import STATE_TAX_RATES
from colors import (DIM_TEXT, BOOTSTRAP_LIGHT_BG, BOOTSTRAP_BORDER,
                    UI_FONT_MD, UI_FONT_BASE, UI_FONT_LG)

# ── State dropdown options (sorted by name) ─────────────────────────────────

_STATE_NAMES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
    "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

_STATE_OPTIONS = sorted(
    [
        {"label": f"{name} ({STATE_TAX_RATES[code]:.2f}%)", "value": code}
        for code, name in _STATE_NAMES.items()
    ],
    key=lambda o: o["label"],
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _account_asset_grid(prefix: str):
    """Shared asset inputs for tax-deferred / tax-free account cards."""
    from tab_defaults import CITADEL
    # Map prefix "cp-td" → key prefix "td_", "cp-tf" → "tf_"
    _kp = prefix.replace("cp-", "") + "_"  # "td_" or "tf_"
    def _d(suffix):
        return CITADEL.get(f"{_kp}{suffix}", 0)
    return html.Div([
        _lbl("BTC Stack"),
        dbc.Input(id=f"{prefix}-btc", type="number", min=0, step=0.001,
                  value=_d("btc"), size="sm"),
        _lbl("Cash ($)"),
        dbc.Input(id=f"{prefix}-cash", type="number", min=0, step=1000,
                  value=_d("cash"), size="sm"),
        html.Hr(style={"margin": "6px 0"}),
        html.Small("Reserves", className="fw-bold d-block mb-1",
                   style={"fontSize": UI_FONT_MD}),
        dbc.Row([
            dbc.Col([html.Small("Short"),
                     dbc.Input(id=f"{prefix}-res-short", type="number",
                               min=0, step=1000, value=_d("res_short"), size="sm")], width=4),
            dbc.Col([html.Small("Med"),
                     dbc.Input(id=f"{prefix}-res-med", type="number",
                               min=0, step=1000, value=_d("res_med"), size="sm")], width=4),
            dbc.Col([html.Small("Long"),
                     dbc.Input(id=f"{prefix}-res-long", type="number",
                               min=0, step=1000, value=_d("res_long"), size="sm")], width=4),
        ], className="g-1 mb-1"),
        html.Small("Investments", className="fw-bold d-block mb-1",
                   style={"fontSize": UI_FONT_MD}),
        dbc.Row([
            dbc.Col([html.Small("Equities"),
                     dbc.Input(id=f"{prefix}-inv-eq", type="number",
                               min=0, step=1000, value=_d("inv_eq"), size="sm")], width=6),
            dbc.Col([html.Small("Bonds"),
                     dbc.Input(id=f"{prefix}-inv-bd", type="number",
                               min=0, step=1000, value=_d("inv_bd"), size="sm")], width=6),
        ], className="g-1"),
    ])


# ── Bracket reference table ─────────────────────────────────────────────────

def _bracket_table(title: str, brackets: list[tuple[float, float]]) -> html.Div:
    """Render a bracket table as an html.Table."""
    rows = []
    prev = 0
    for upper, rate in brackets:
        if upper == float("inf"):
            bracket_str = f"${prev:,.0f}+"
        else:
            bracket_str = f"${prev:,.0f} - ${upper:,.0f}"
        rows.append(html.Tr([
            html.Td(bracket_str, style={"paddingRight": "16px", "fontSize": UI_FONT_BASE}),
            html.Td(f"{rate * 100:.1f}%", style={"fontSize": UI_FONT_BASE, "textAlign": "right"}),
        ]))
        prev = upper
    return html.Div([
        html.Strong(title, style={"fontSize": UI_FONT_BASE, "display": "block",
                                   "marginBottom": "4px"}),
        html.Table([html.Tbody(rows)],
                   style={"marginBottom": "12px"}),
    ])


def _bracket_reference_section():
    """Accordion with federal bracket tables for reference."""
    from engines.tax_data import (FEDERAL_BRACKETS_TCJA, FEDERAL_BRACKETS_SUNSET,
                                   LTCG_BRACKETS)
    return dbc.Accordion([
        dbc.AccordionItem([
            dbc.Row([
                dbc.Col([
                    _bracket_table("Single (TCJA)", FEDERAL_BRACKETS_TCJA["single"]),
                    _bracket_table("MFJ (TCJA)", FEDERAL_BRACKETS_TCJA["mfj"]),
                ], md=4),
                dbc.Col([
                    _bracket_table("Single (Sunset)", FEDERAL_BRACKETS_SUNSET["single"]),
                    _bracket_table("MFJ (Sunset)", FEDERAL_BRACKETS_SUNSET["mfj"]),
                ], md=4),
                dbc.Col([
                    _bracket_table("LTCG — Single", LTCG_BRACKETS["single"]),
                    _bracket_table("LTCG — MFJ", LTCG_BRACKETS["mfj"]),
                ], md=4),
            ]),
        ], title="Tax Rate Reference (2025 brackets)", item_id="tax-ref"),
    ], start_collapsed=True, id="cp-tax-ref-accordion")


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def tax_toggle_widget():
    """Compact section for the Simulation sub-tab — title + toggle + config button + stores."""
    _TITLE_STYLE = {"fontWeight": "bold", "fontSize": UI_FONT_BASE,
                    "color": DIM_TEXT, "marginBottom": "4px",
                    "textTransform": "uppercase", "letterSpacing": "0.03em"}
    return html.Div([
        html.Div("Taxation", style=_TITLE_STYLE),
        dbc.Switch(id="cp-tax-toggle",
                   label="Enable taxation & tax advantaged retirement accounts",
                   value=False,
                   style={"marginBottom": "4px", "transform": "scale(1.2)",
                          "transformOrigin": "left center"}),
        dbc.Button("Configure Tax Settings\u2026", id="cp-tax-config-btn",
                   color="secondary", size="sm", outline=True,
                   style={"display": "none"}),
        dcc.Store(id="cp-tax-config", storage_type="memory", data={}),
        dcc.Store(id="cp-tax-annual-data", data=[]),
    ], style={"marginBottom": "12px", "padding": "8px",
              "background": BOOTSTRAP_LIGHT_BG, "borderRadius": "8px",
              "border": f"1px solid {BOOTSTRAP_BORDER}"})


def tax_config_modal():
    """Full-screen modal with all tax configuration controls."""
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Tax Configuration"), close_button=True),
        dbc.ModalBody([

            # ── Section A: Filing & Rates ────────────────────────────────
            dbc.Card(dbc.CardBody([
                html.H6("Filing & Rates", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        _lbl("Filing Status"),
                        dbc.RadioItems(id="cp-tax-filing",
                            options=[{"label": "Single", "value": "single"},
                                     {"label": "Married Filing Jointly", "value": "mfj"}],
                            value="single", inline=True,
                            inputStyle=_CB_MARGIN,
                            labelStyle={"marginRight": "12px"}),
                    ], md=6),
                    dbc.Col([
                        _lbl("Tax Law"),
                        dbc.RadioItems(id="cp-tax-tcja",
                            options=[{"label": "Current law (TCJA)", "value": "tcja"},
                                     {"label": "Scheduled sunset", "value": "sunset"}],
                            value="tcja", inline=True,
                            inputStyle=_CB_MARGIN,
                            labelStyle={"marginRight": "12px"}),
                    ], md=6),
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col([
                        _lbl("State"),
                        dbc.Select(id="cp-tax-state", options=_STATE_OPTIONS,
                                   value="TX"),
                    ], md=4),
                    dbc.Col([
                        _lbl("State Rate (%)"),
                        dbc.Input(id="cp-tax-state-rate", type="number",
                                  step=0.01, value=0.0, size="sm"),
                    ], md=2),
                    dbc.Col([
                        _lbl("Birth Year"),
                        dbc.Input(id="cp-tax-birth-year", type="number",
                                  min=1900, max=2099, step=1,
                                  placeholder="Skip RMDs", size="sm"),
                    ], md=2),
                    dbc.Col([
                        _lbl("Cost Basis Method"),
                        dbc.RadioItems(id="cp-tax-basis-method",
                            options=[{"label": "FIFO (oldest first)", "value": "fifo"},
                                     {"label": "LIFO (newest first)", "value": "lifo"}],
                            value="fifo", inline=True,
                            inputStyle=_CB_MARGIN,
                            labelStyle={"marginRight": "12px"}),
                    ], md=4),
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col([
                        _lbl("Other Annual Income ($)"),
                        dbc.Input(id="cp-tax-other-income", type="number",
                                  min=0, step=1000, value=0, size="sm"),
                    ], md=4),
                    dbc.Col([
                        _lbl("Income Growth (%/yr)"),
                        dbc.Input(id="cp-tax-other-income-growth", type="number",
                                  min=0, max=20, step=0.5, value=0, size="sm"),
                    ], md=4),
                ], className="mb-0"),
            ]), className="mb-3"),

            # ── Section B: Account Wrappers ──────────────────────────────
            dbc.Card(dbc.CardBody([
                html.H6("Account Wrappers", className="mb-3"),
                dbc.Row([
                    # Taxable (read-only info)
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.Strong("Taxable Account",
                                    style={"fontSize": UI_FONT_LG, "display": "block",
                                           "marginBottom": "6px"}),
                        html.Small("Uses existing Citadel asset configuration "
                                   "from the Assets sub-tab.",
                                   style=_STYLE_HINT),
                        html.Small("BTC & investment sales: capital gains tax "
                                   "(ST or LT based on holding period). "
                                   "Cash/reserve withdrawals: no tax event. "
                                   "Interest earned: ordinary income.",
                                   style=_STYLE_HINT),
                    ], className="p-2"), color="light"), md=4),

                    # Tax-Deferred (Traditional IRA/401k)
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.Strong("Tax-Deferred (Trad IRA / 401k)",
                                    style={"fontSize": UI_FONT_LG, "display": "block",
                                           "marginBottom": "6px"}),
                        html.Small("Withdrawals taxed as ordinary income. "
                                   "Subject to RMDs at age 73+.",
                                   style=_STYLE_HINT),
                        _account_asset_grid("cp-td"),
                    ], className="p-2")), md=4),

                    # Tax-Free (Roth IRA/401k)
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.Strong("Tax-Free (Roth IRA / 401k)",
                                    style={"fontSize": UI_FONT_LG, "display": "block",
                                           "marginBottom": "6px"}),
                        html.Small("Qualified withdrawals are tax-free. "
                                   "No RMDs required.",
                                   style=_STYLE_HINT),
                        _account_asset_grid("cp-tf"),
                    ], className="p-2")), md=4),
                ]),
            ]), className="mb-3"),

            # ── Section C: Bracket Reference ─────────────────────────────
            _bracket_reference_section(),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="cp-tax-cancel", color="secondary",
                       className="me-2"),
            dbc.Button("Save Tax Settings", id="cp-tax-save", color="primary"),
        ]),
    ], id="cp-tax-modal", fullscreen=True, is_open=False)


def tax_summary_panel():
    """Collapsible summary table shown below the chart when tax is enabled."""
    return dbc.Collapse(
        dbc.Card(dbc.CardBody([
            html.H6("Tax Summary", className="mb-2",
                     style={"fontSize": UI_FONT_LG}),
            dbc.Table(id="cp-tax-summary-table", bordered=True, size="sm",
                      style={"fontSize": UI_FONT_BASE}),
        ], className="p-2"), className="mb-2"),
        id="cp-tax-summary", is_open=False,
    )
