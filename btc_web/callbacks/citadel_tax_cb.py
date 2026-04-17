"""Tax configuration modal callbacks for the Citadel Planner."""

import json

from dash import Input, Output, State, callback, ctx, no_update

import _app_ctx
from engines.tax_data import STATE_TAX_RATES

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


# ── Open modal (clientside) ───────────────────────────────────────────────────
_app_ctx.app.clientside_callback(
    "function(n) { return true; }",
    Output("cp-tax-modal", "is_open", allow_duplicate=True),
    Input("cp-tax-config-btn", "n_clicks"),
    prevent_initial_call=True,
)


# ── State dropdown auto-fill rate (clientside) ────────────────────────────────
_app_ctx.app.clientside_callback(
    "function(code) {"
    "  var rates = " + json.dumps(STATE_TAX_RATES) + ";"
    "  if (!code) { return window.dash_clientside.no_update; }"
    "  var r = rates[code];"
    "  return (r === undefined) ? 0.0 : r;"
    "}",
    Output("cp-tax-state-rate", "value"),
    Input("cp-tax-state", "value"),
    prevent_initial_call=True,
)


# ── Save config + close modal ─────────────────────────────────────────────────
# The header × ("cp-tax-header-save") shares the save path with the footer
# Save button — mobile users often can't reach the footer buttons when the
# modal body overflows the viewport, so the always-visible header × is the
# primary confirm affordance.  Escape still closes the modal (via dbc
# default) without firing any callback, which matches Cancel semantics.
@callback(
    Output("cp-tax-config", "data"),
    Output("cp-tax-modal", "is_open"),
    Input("cp-tax-save", "n_clicks"),
    Input("cp-tax-header-save", "n_clicks"),
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
def _save_or_cancel(save_clicks, header_save_clicks, cancel_clicks,
                    filing, state, state_rate, birth_year,
                    other_income, other_income_growth, tcja, basis_method,
                    td_btc, td_cash, td_rs, td_rm, td_rl, td_eq, td_bd,
                    tf_btc, tf_cash, tf_rs, tf_rm, tf_rl, tf_eq, tf_bd):
    triggered = ctx.triggered_id
    if triggered == "cp-tax-cancel":
        return no_update, False

    def _pos(v): return max(float(v or 0), 0.0)
    _by = None
    if birth_year:
        _by = max(1900, min(int(birth_year), 2099))

    config = {
        "filing_status": filing or "single",
        "state_code": state or "TX",
        "state_rate_override": max(float(state_rate), 0.0) if state_rate is not None else None,
        "birth_year": _by,
        "other_income": _pos(other_income),
        "other_income_growth": max(float(other_income_growth or 0), 0.0),
        "tcja_sunset": (tcja == "sunset"),
        "cost_basis_method": basis_method or "fifo",
        "td_btc": _pos(td_btc),
        "td_cash": _pos(td_cash),
        "td_res_short": _pos(td_rs),
        "td_res_med": _pos(td_rm),
        "td_res_long": _pos(td_rl),
        "td_inv_eq": _pos(td_eq),
        "td_inv_bd": _pos(td_bd),
        "tf_btc": _pos(tf_btc),
        "tf_cash": _pos(tf_cash),
        "tf_res_short": _pos(tf_rs),
        "tf_res_med": _pos(tf_rm),
        "tf_res_long": _pos(tf_rl),
        "tf_inv_eq": _pos(tf_eq),
        "tf_inv_bd": _pos(tf_bd),
    }
    return config, False


# ── Tax summary table builder (clientside) ───────────────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(annual_data) {
        if (!annual_data || (Array.isArray(annual_data) && annual_data.length === 0)) {
            return [false, []];
        }
        // Flatten list-of-lists (one per sim) → flat list of year dicts
        if (Array.isArray(annual_data[0])) {
            annual_data = annual_data[0];
        }
        function H(tag, children, props) {
            var p = props || {};
            p.children = children;
            return {namespace: "dash_html_components", type: tag, props: p};
        }
        function fmtUSD(n) {
            var v = (n == null) ? 0 : n;
            var sign = v < 0 ? "-" : "";
            v = Math.abs(Math.round(v));
            var s = v.toString();
            // Insert thousands separators
            s = s.replace(/\\B(?=(\\d{3})+(?!\\d))/g, ",");
            return "$" + sign + s;
        }
        var header = H("Thead", [H("Tr", [
            H("Th", "Year"), H("Th", "Ordinary"), H("Th", "ST Gains"),
            H("Th", "LT Gains"), H("Th", "Federal"), H("Th", "NIIT"),
            H("Th", "State"), H("Th", "Total"), H("Th", "Eff. Rate")
        ])]);
        var rows = [];
        for (var i = 0; i < annual_data.length; i++) {
            var yr = annual_data[i];
            var fed = (yr.federal_ordinary || 0) + (yr.federal_ltcg || 0);
            var eff = yr.effective_rate;
            var effStr = (typeof eff === "number") ? (eff * 100).toFixed(1) + "%" : "0.0%";
            rows.push(H("Tr", [
                H("Td", String(yr.year == null ? "" : yr.year)),
                H("Td", fmtUSD(yr.ordinary_income)),
                H("Td", fmtUSD(yr.st_gains)),
                H("Td", fmtUSD(yr.lt_gains)),
                H("Td", fmtUSD(fed)),
                H("Td", fmtUSD(yr.niit)),
                H("Td", fmtUSD(yr.state)),
                H("Td", fmtUSD(yr.total)),
                H("Td", effStr),
            ]));
        }
        var body = H("Tbody", rows);
        return [true, [header, body]];
    }
    """,
    Output("cp-tax-summary", "is_open"),
    Output("cp-tax-summary-table", "children"),
    Input("cp-tax-annual-data", "data"),
    prevent_initial_call=True,
)


# ── Helper for testing ────────────────────────────────────────────────────────
def _state_to_rate(state_code: str) -> float:
    from engines.tax_data import STATE_TAX_RATES
    return STATE_TAX_RATES.get(state_code, 0.0)


def _build_tax_summary(annual_data):
    """Python mirror of the clientside tax-summary-table builder above.

    Used only by tests -- production path is the JS clientside callback
    registered with `cp-tax-annual-data` as its Input. Signature matches
    the JS return shape so tests can assert on both behaviors identically.

    Returns: (is_open: bool, children: list).
    - Empty or None input  -> (False, [])
    - Non-empty            -> (True, [header_thead, body_tbody]) (2 children)

    Accepts either a flat list of year dicts, or a list-of-lists where the
    first element is the list of year dicts (matching the engine's MC output
    shape).
    """
    from dash import html

    if not annual_data:
        return False, []

    # Flatten list-of-lists: MC sims may wrap annual data in an outer list.
    if isinstance(annual_data, list) and annual_data and isinstance(annual_data[0], list):
        annual_data = annual_data[0]

    if not annual_data:
        return False, []

    def _fmt_usd(n):
        v = n if n is not None else 0
        return f"${v:,.0f}" if v >= 0 else f"-${abs(v):,.0f}"

    header = html.Thead(html.Tr([
        html.Th("Year"), html.Th("Ordinary"), html.Th("ST Gains"),
        html.Th("LT Gains"), html.Th("Federal"), html.Th("NIIT"),
        html.Th("State"), html.Th("Total"), html.Th("Eff. Rate"),
    ]))
    rows = []
    for yr in annual_data:
        fed = (yr.get("federal_ordinary") or 0) + (yr.get("federal_ltcg") or 0)
        eff = yr.get("effective_rate")
        eff_str = f"{eff * 100:.1f}%" if isinstance(eff, (int, float)) else "0.0%"
        rows.append(html.Tr([
            html.Td(str(yr.get("year", ""))),
            html.Td(_fmt_usd(yr.get("ordinary_income"))),
            html.Td(_fmt_usd(yr.get("st_gains"))),
            html.Td(_fmt_usd(yr.get("lt_gains"))),
            html.Td(_fmt_usd(fed)),
            html.Td(_fmt_usd(yr.get("niit"))),
            html.Td(_fmt_usd(yr.get("state"))),
            html.Td(_fmt_usd(yr.get("total"))),
            html.Td(eff_str),
        ]))
    body = html.Tbody(rows)
    return True, [header, body]
