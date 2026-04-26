"""Leverage Calculator — main callback.

Wires inputs (date, price, model, reversion-q store, rates, H, CAGR sliders)
to outputs (plot, readout, table).

Registered at import time (side-effect) — import from `callbacks/__init__.py`.
"""
from __future__ import annotations

import datetime as _dt

from dash import Input, Output, State, callback, html
from dash.exceptions import PreventUpdate

from figures.leverage import (
    build_leverage_figure, reversion_price, P_max, implied_cagr, implied_quantile,
    _parse_date,
)
from colors import BORDER_MUTED, TABLE_ROW_HIGHLIGHT_BG


def _readout(buy_date, P_now, sell_date, sell_price, H_yr, c, max_pay, implied_c, model, q):
    """Render the scenario + decision block (spec §6.2)."""
    q_label = f"Q{q*100:g}%"
    is_buy = P_now <= max_pay
    if P_now > 0:
        delta_pct = (max_pay - P_now) / P_now * 100
    else:
        delta_pct = 0
    badge_text = (
        f"✓ BUY — {delta_pct:+.1f}% under your max" if is_buy
        else f"⚠ ABOVE MAX — {-delta_pct:.1f}% over; raise H or lower target to flip"
    )
    badge_class = "alert alert-success" if is_buy else "alert alert-danger"

    implied_str = f"{implied_c*100:.1f}%" if implied_c is not None else "—"
    return html.Div([
        html.Div([
            html.Span(f"Buy:  {buy_date.isoformat()}  @ "),
            html.B(f"${P_now:,.0f}"),
            html.Span("  (current price)"),
        ]),
        html.Div([
            html.Span(f"Sell: {sell_date.isoformat()}  @ "),
            html.B(f"${sell_price:,.0f}"),
            html.Span(f"  ({model} {q_label} reversion)"),
        ]),
        html.Div(f"Horizon H = {H_yr:.2f} yr"),
        html.Hr(),
        html.Div([
            html.Span("Your target: "),
            html.B(f"{c*100:.1f}% CAGR"),
        ]),
        html.Div([
            html.Span("Max pay-price today: "),
            html.B(f"${max_pay:,.0f}", style={"fontSize": "1.25em"}),
        ]),
        html.Div(badge_text, className=badge_class, style={"marginTop": "8px"}),
        html.Div(f"Implied CAGR at ${P_now:,.0f}: {implied_str}"),
    ], style={"border": f"1px solid {BORDER_MUTED}", "borderRadius": "6px", "padding": "12px"})


def _fmt_q(q_val):
    pct = q_val * 100
    return f"Q{pct:.2g}%" if pct >= 1 else f"Q{pct:.2g}%"


def _table(buy_date, model, q, r_b, r_l, c, H_slider):
    """Render the 7-row canonical table (spec §6.3)."""
    horizons = [1, 2, 3, 4, 5, 8, 10]
    header = html.Tr([html.Th(h) for h in
                      ["H (yr)", "Sell date", "Sell price", "Sell quantile",
                       "Max pay @ 0%", f"@ r_l ({r_l*100:.2f}%)",
                       f"@ r_b ({r_b*100:.2f}%)", f"@ your ({c*100:.1f}%)"]])
    rows = []
    for H in horizons:
        sell_d = buy_date + _dt.timedelta(days=int(round(H * 365.25)))
        sp = reversion_price(model, q,sell_d)
        sell_q = implied_quantile(model, sp, sell_d)
        row_style = {"backgroundColor": TABLE_ROW_HIGHLIGHT_BG} if abs(H - H_slider) < 0.5 else {}
        rows.append(html.Tr([
            html.Td(H),
            html.Td(sell_d.isoformat()),
            html.Td(f"${sp:,.0f}"),
            html.Td(_fmt_q(sell_q)),
            html.Td(f"${P_max(sp, H, 0.0):,.0f}"),
            html.Td(f"${P_max(sp, H, r_l):,.0f}"),
            html.Td(f"${P_max(sp, H, r_b):,.0f}"),
            html.Td(f"${P_max(sp, H, c):,.0f}"),
        ], style=row_style))
    return html.Table([html.Thead(header), html.Tbody(rows)],
                      className="table table-sm", style={"width": "100%"})


@callback(
    Output("lev-graph", "figure"),
    Output("lev-readout", "children"),
    Output("lev-table", "children"),
    Input("leverage-first-render", "data"),
    Input("lev-date", "date"),
    Input("lev-price", "value"),
    Input("lev-model", "value"),
    Input("lev-floor-q-store", "data"),
    Input("lev-rb", "value"),
    Input("lev-rl", "value"),
    Input("lev-horizon", "value"),
    Input("lev-cagr", "value"),
    Input("lev-toggles", "value"),
    State("snapshot-pending", "data"),
    prevent_initial_call=False,
)
def update_leverage(first_render, date_val, price_val, model, q,
                    rb_val, rl_val, H_val, c_val, toggles,
                    snapshot_pending=False):
    # Snapshot gate — see spec 2026-04-24-single-redraw-per-snapshot.
    if snapshot_pending:
        from dash import no_update
        return no_update, no_update, no_update
    if not first_render:
        raise PreventUpdate

    # Coerce with falsy-zero-safe pattern (CLAUDE.md §"Falsy-zero")
    price = float(price_val) if price_val is not None else 65000.0
    rb    = float(rb_val)    if rb_val    is not None else 13.0
    rl    = float(rl_val)    if rl_val    is not None else 4.5
    H_yr  = float(H_val)     if H_val     is not None else 4.0
    c_pct = float(c_val)     if c_val     is not None else 20.0
    model = str(model or "bub")
    q     = float(q) if q is not None else 0.01

    # Guards (spec §5.5)
    H_yr  = max(H_yr, 0.01)
    price = max(price, 1.0)

    # Parse date
    buy_date = _parse_date(date_val) if date_val else _dt.date.today()

    # Compute core outputs
    c_dec = c_pct / 100.0
    r_b_dec = rb / 100.0
    r_l_dec = rl / 100.0

    sell_date = buy_date + _dt.timedelta(days=int(round(H_yr * 365.25)))
    try:
        sp = reversion_price(model, q,sell_date)
    except (KeyError, AttributeError, ValueError) as e:
        return (
            {}, html.Div(f"Model unavailable: {e}", className="alert alert-warning"),
            html.Div()
        )

    max_pay = P_max(sp, H_yr, c_dec)
    implied_c = implied_cagr(sp, price, H_yr)

    p = {
        "lev_price": price, "lev_date": buy_date,
        "lev_model": model, "lev_reversion_q": q,
        "lev_rb": rb, "lev_rl": rl,
        "lev_horizon": H_yr, "lev_cagr": c_pct,
        "lev_toggles": tuple(toggles or ()),
        "palette": "default",
    }
    fig = build_leverage_figure(p)
    ro = _readout(buy_date, price, sell_date, sp, H_yr, c_dec, max_pay, implied_c, model, q)
    tbl = _table(buy_date, model, q, r_b_dec, r_l_dec, c_dec, H_yr)
    return fig, ro, tbl


import _app_ctx


# Input → Store: integer percent (1–99) → fractional float (0.01–0.99).
# HTML5 step=1/min=1/max=99 enforces validity; out-of-range values arrive
# as null and are ignored.
_app_ctx.app.clientside_callback(
    """function(pct) {
        if (pct === null || pct === undefined) {
            return window.dash_clientside.no_update;
        }
        var n = Math.round(Number(pct));
        if (!isFinite(n) || n < 1 || n > 99) {
            return window.dash_clientside.no_update;
        }
        return n / 100.0;
    }""",
    Output("lev-floor-q-store", "data", allow_duplicate=True),
    Input("lev-reversion-q-input", "value"),
    prevent_initial_call=True,
)


# Store → Input: fractional float → integer percent (snapshot restore path).
# Old share-links with sub-1% values (e.g. 0.001 = 0.1%) clamp to 1.
_app_ctx.app.clientside_callback(
    """function(q) {
        if (q === null || q === undefined) {
            return window.dash_clientside.no_update;
        }
        var pct = Math.round(Number(q) * 100.0);
        if (!isFinite(pct)) return window.dash_clientside.no_update;
        if (pct < 1) pct = 1;
        if (pct > 99) pct = 99;
        return pct;
    }""",
    Output("lev-reversion-q-input", "value", allow_duplicate=True),
    Input("lev-floor-q-store", "data"),
    prevent_initial_call="initial_duplicate",
)
