"""Tax configuration modal callbacks for the Citadel Planner."""

from dash import Input, Output, State, callback, ctx, html, no_update

import _app_ctx

# ── Toggle config button visibility (clientside) ─────────────────────────────
_app_ctx.app.clientside_callback(
    "function(v) { return v ? {} : {display:'none'}; }",
    Output("cp-tax-config-btn", "style"),
    Input("cp-tax-toggle", "value"),
)

# ── Update Run button label (clientside) ─────────────────────────────────────
_app_ctx.app.clientside_callback(
    'function(v) { return v ? "\u25b6 Run Simulation (with Tax)" : "\u25b6 Run Simulation"; }',
    Output("cp-run-btn", "children", allow_duplicate=True),
    Input("cp-tax-toggle", "value"),
    prevent_initial_call=True,
)


# ── Open modal ────────────────────────────────────────────────────────────────
@callback(
    Output("cp-tax-modal", "is_open", allow_duplicate=True),
    Input("cp-tax-config-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _open_tax_modal(n):
    return True


# ── State dropdown auto-fill rate ─────────────────────────────────────────────
@callback(
    Output("cp-tax-state-rate", "value"),
    Input("cp-tax-state", "value"),
    prevent_initial_call=True,
)
def _update_state_rate(state_code):
    from engines.tax_data import STATE_TAX_RATES
    if not state_code:
        return no_update
    return STATE_TAX_RATES.get(state_code, 0.0)


# ── Save config + close modal ─────────────────────────────────────────────────
@callback(
    Output("cp-tax-config", "data"),
    Output("cp-tax-modal", "is_open"),
    Input("cp-tax-save", "n_clicks"),
    Input("cp-tax-cancel", "n_clicks"),
    State("cp-tax-filing", "value"),
    State("cp-tax-state", "value"),
    State("cp-tax-state-rate", "value"),
    State("cp-tax-birth-year", "value"),
    State("cp-tax-other-income", "value"),
    State("cp-tax-other-income-growth", "value"),
    State("cp-tax-tcja", "value"),
    State("cp-tax-basis-method", "value"),
    State("cp-td-btc", "value"),
    State("cp-td-cash", "value"),
    State("cp-td-res-short", "value"),
    State("cp-td-res-med", "value"),
    State("cp-td-res-long", "value"),
    State("cp-td-inv-eq", "value"),
    State("cp-td-inv-bd", "value"),
    State("cp-tf-btc", "value"),
    State("cp-tf-cash", "value"),
    State("cp-tf-res-short", "value"),
    State("cp-tf-res-med", "value"),
    State("cp-tf-res-long", "value"),
    State("cp-tf-inv-eq", "value"),
    State("cp-tf-inv-bd", "value"),
    prevent_initial_call=True,
)
def _save_or_cancel(save_clicks, cancel_clicks, filing, state, state_rate, birth_year,
                    other_income, other_income_growth, tcja, basis_method,
                    td_btc, td_cash, td_rs, td_rm, td_rl, td_eq, td_bd,
                    tf_btc, tf_cash, tf_rs, tf_rm, tf_rl, tf_eq, tf_bd):
    triggered = ctx.triggered_id
    if triggered == "cp-tax-cancel":
        return no_update, False

    config = {
        "filing_status": filing or "single",
        "state_code": state or "TX",
        "state_rate_override": float(state_rate) if state_rate is not None else None,
        "birth_year": int(birth_year) if birth_year else None,
        "other_income": float(other_income or 0),
        "other_income_growth": float(other_income_growth or 0),
        "tcja_sunset": (tcja == "sunset"),
        "cost_basis_method": basis_method or "fifo",
        "td_btc": float(td_btc or 0),
        "td_cash": float(td_cash or 0),
        "td_res_short": float(td_rs or 0),
        "td_res_med": float(td_rm or 0),
        "td_res_long": float(td_rl or 0),
        "td_inv_eq": float(td_eq or 0),
        "td_inv_bd": float(td_bd or 0),
        "tf_btc": float(tf_btc or 0),
        "tf_cash": float(tf_cash or 0),
        "tf_res_short": float(tf_rs or 0),
        "tf_res_med": float(tf_rm or 0),
        "tf_res_long": float(tf_rl or 0),
        "tf_inv_eq": float(tf_eq or 0),
        "tf_inv_bd": float(tf_bd or 0),
    }
    return config, False


# ── Tax summary table builder ────────────────────────────────────────────────
@callback(
    Output("cp-tax-summary", "is_open"),
    Output("cp-tax-summary-table", "children"),
    Input("cp-tax-annual-data", "data"),
    prevent_initial_call=True,
)
def _build_tax_summary(annual_data):
    if not annual_data:
        return False, []

    header = html.Thead(html.Tr([
        html.Th("Year"), html.Th("Ordinary"), html.Th("ST Gains"),
        html.Th("LT Gains"), html.Th("Federal"), html.Th("NIIT"),
        html.Th("State"), html.Th("Total"), html.Th("Eff. Rate"),
    ]))

    rows = []
    for yr in annual_data:
        fed = yr.get("federal_ordinary", 0) + yr.get("federal_ltcg", 0)
        eff = yr.get("effective_rate", 0)
        rows.append(html.Tr([
            html.Td(yr.get("year", "")),
            html.Td(f"${yr.get('ordinary_income', 0):,.0f}"),
            html.Td(f"${yr.get('st_gains', 0):,.0f}"),
            html.Td(f"${yr.get('lt_gains', 0):,.0f}"),
            html.Td(f"${fed:,.0f}"),
            html.Td(f"${yr.get('niit', 0):,.0f}"),
            html.Td(f"${yr.get('state', 0):,.0f}"),
            html.Td(f"${yr.get('total', 0):,.0f}"),
            html.Td(f"{eff * 100:.1f}%" if isinstance(eff, (int, float)) else "0.0%"),
        ]))

    return True, [header, html.Tbody(rows)]


# ── Helper for testing ────────────────────────────────────────────────────────
def _state_to_rate(state_code: str) -> float:
    from engines.tax_data import STATE_TAX_RATES
    return STATE_TAX_RATES.get(state_code, 0.0)
