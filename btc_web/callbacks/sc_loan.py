"""Saylor Mode / Stack-celerator loan info and toggles."""

import json

from dash import html, Input, Output, State, callback

import _app_ctx
from btc_core import fmt_price
from tab_defaults import DCA
from callbacks.coerce import _ci, _cf
from figures import FREQ_PPY


# ── Saylor Mode: first-time quote toast ──────────────────────────────────────
_SAYLOR_QUOTES = [
    "There is no second best.",
    "If you\u2019ve got a billion-dollar problem, you need a trillion-dollar solution.",
    "Buy bitcoin. Then go figure out what you sold.",
    "You don\u2019t need a Plan B. You just need more Bitcoin.",
    "Buy Bitcoin and wait. Not complicated.",
    "The best time to buy bitcoin was yesterday. The second best time is today.",
    "There is no top, because fiat has no bottom.",
    "I decided to put all my eggs in one basket, and that basket is Bitcoin.",
]
_SAYLOR_QUOTES_JS = json.dumps(_SAYLOR_QUOTES)

_app_ctx.app.clientside_callback(
    f"""
    function(val) {{
        var NU = window.dash_clientside.no_update;
        if (!val || !val.length) return NU;
        var WK = "wizard-flags";
        var isDev = (location.hostname !== "quantoshi.xyz" &&
                     !location.hostname.endsWith(".onion"));
        try {{
            var f = JSON.parse(localStorage.getItem(WK)) || {{}};
            var now = Date.now();
            var day = 24 * 3600 * 1000;
            if (f.saylor_toast_ts && (now - f.saylor_toast_ts < day) && !isDev) return NU;
            f.saylor_toast_ts = now;
            localStorage.setItem(WK, JSON.stringify(f));
        }} catch(e) {{ return NU; }}
        var quotes = {_SAYLOR_QUOTES_JS};
        var q = quotes[Math.floor(Math.random() * quotes.length)];
        var el = document.createElement("div");
        el.className = "ambient-toast";
        el.textContent = "\\u201c" + q + "\\u201d \\u2014 Michael Saylor";
        document.body.appendChild(el);
        setTimeout(function() {{
            if (el.parentNode) el.parentNode.removeChild(el);
        }}, 6500);
        return NU;
    }}
    """,
    Output("dca-sc-body", "style", allow_duplicate=True),
    Input("dca-sc-enable", "value"),
    prevent_initial_call="initial_duplicate",
)


_app_ctx.app.clientside_callback(
    """
    function(mode) {
        return (mode === 'custom') ? {} : {display: 'none'};
    }
    """,
    Output("dca-sc-custom-price-row", "style"),
    Input("dca-sc-entry-mode", "value"),
)


_app_ctx.app.clientside_callback(
    """
    function(loan_type) {
        return ((loan_type || 'interest_only') === 'interest_only') ? {} : {display: 'none'};
    }
    """,
    Output("dca-sc-rollover-row", "style"),
    Input("dca-sc-type", "value"),
)


@callback(
    Output("dca-sc-info","children"),
    Input("dca-amount",     "value"),
    Input("dca-freq",       "value"),
    Input("dca-sc-enable",  "value"),
    Input("dca-sc-loan",    "value"),
    Input("dca-sc-rate",    "value"),
    Input("dca-sc-term",    "value"),
    Input("dca-sc-type",         "value"),
    Input("dca-sc-repeats",      "value"),
    Input("dca-sc-entry-mode",   "value"),
    Input("dca-sc-custom-price", "value"),
    Input("dca-sc-tax",          "value"),
    Input("dca-sc-rollover",     "value"),
    State("btc-price-store","data"),
)
def update_sc_info(amount, freq, enabled, sc_loan, rate, term, loan_type, repeats,
                   entry_mode, custom_price, tax, rollover, price_data):
    if not enabled:
        return ""
    from engines.sc_math import compute_sc_loan as _compute_sc_loan

    ppy          = FREQ_PPY.get(freq or "Monthly", 12)
    amount       = _ci(amount, 100, lo=0, hi=_app_ctx.MAX_USD)
    principal    = _cf(sc_loan, 0)
    rate         = _cf(rate, 13.0)
    term         = _cf(term, 12)
    loan_type    = loan_type or "interest_only"
    sc_rollover  = bool(rollover) and loan_type == "interest_only"
    n_repeats    = _ci(repeats, 0)
    n_cycles     = 1 + n_repeats
    r            = rate / 100.0 / ppy
    term_periods = max(1, round(term * ppy / 12))
    entry_mode   = entry_mode or "live"
    tax_rate     = _cf(tax, 33, lo=0, hi=100) / 100.0
    live         = _cf(price_data, 0)

    principal, pmt, capped = _compute_sc_loan(principal, amount, r, term_periods, loan_type)

    if loan_type == "amortizing":
        total_interest = pmt * term_periods - principal
        type_lbl = "Amortizing"
    else:
        total_interest = pmt * term_periods
        type_lbl = "Interest-only (rollover)" if sc_rollover else "Interest-only"

    reduced = amount - pmt

    # Entry price for display
    if entry_mode == "live":
        ep = live
        ep_lbl = f"Live ticker ({fmt_price(live)})" if live > 0 else "Live ticker"
    elif entry_mode == "custom":
        ep = _cf(custom_price, DCA["sc_custom_price"])
        ep_lbl = f"Custom ({fmt_price(ep)})"
    else:
        ep = 0.0
        ep_lbl = "Model price"

    lump_btc  = principal / ep if ep > 0 else None
    active_mo = n_cycles * int(term)

    # Tax cost line — tax applies only to the capital gain (sell_price − cost_basis),
    # not to the full sale proceeds.  Actual BTC sold depends on future price.
    if loan_type == "interest_only":
        basis_str = fmt_price(ep) if ep > 0 else "model price at cycle start"
        if sc_rollover:
            tax_lbl = (f"Tax @{tax_rate*100:.4g}%: on gain at simulation end "
                       f"(cost basis: {basis_str})")
        elif tax_rate > 0:
            tax_lbl = (f"Tax @{tax_rate*100:.4g}%: on gain at each cycle-end repayment "
                       f"(cost basis: {basis_str})")
        else:
            tax_lbl = None
    else:  # amortizing
        tax_lbl = f"Tax @{tax_rate*100:.4g}%: N/A — principal repaid in fiat (no BTC sold)"

    loan_lbl = fmt_price(principal)
    if capped:
        loan_lbl += f"  (capped — max for {fmt_price(amount)}/period DCA)"
    lines = [
        f"Loan: {loan_lbl}  \u00b7  {type_lbl}",
        f"Entry: {ep_lbl}",
        f"Payment: {fmt_price(pmt)}/period  \u2192  DCA: {fmt_price(reduced)}/period",
        f"Total interest/cycle: {fmt_price(total_interest)}  (over {int(term)} mo)",
    ]
    if lump_btc:
        cycle_note = "first cycle only @ entry price" if sc_rollover else "each cycle @ entry price"
        lines.append(f"Buys \u2248 {lump_btc:.5f} BTC  ({cycle_note})")
    if tax_lbl:
        lines.append(tax_lbl)
    lines.append(f"Cycles: {n_cycles} total  \u00b7  Loan active {active_mo} mo")
    return [html.Div(l) for l in lines]
