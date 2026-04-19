"""Leverage Calculator — main callback.

Wires inputs (date, price, model, floor-q store, rates, H, CAGR sliders)
to outputs (plot, readout, table).

Registered at import time (side-effect) — import from `callbacks/__init__.py`.
"""
from __future__ import annotations

import datetime as _dt

from dash import Input, Output, callback, html
from dash.exceptions import PreventUpdate

from figures.leverage import (
    build_leverage_figure, floor_price, P_max, implied_cagr, _parse_date,
)


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
            html.Span(f"  ({model} {q_label} floor)"),
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
    ], style={"border": "1px solid #ccc", "borderRadius": "6px", "padding": "12px"})


def _table(buy_date, model, q, r_b, r_l, c, H_slider):
    """Render the 7-row canonical table (spec §6.3)."""
    horizons = [1, 2, 3, 4, 5, 8, 10]
    header = html.Tr([html.Th(h) for h in
                      ["H (yr)", "Sell date", "Sell price",
                       "Max pay @ 0%", f"@ r_l ({r_l*100:.2f}%)",
                       f"@ r_b ({r_b*100:.2f}%)", f"@ your ({c*100:.1f}%)"]])
    rows = []
    for H in horizons:
        sell_d = buy_date + _dt.timedelta(days=int(round(H * 365.25)))
        sp = floor_price(model, q, sell_d)
        row_style = {"backgroundColor": "#ffe"} if abs(H - H_slider) < 0.5 else {}
        rows.append(html.Tr([
            html.Td(H),
            html.Td(sell_d.isoformat()),
            html.Td(f"${sp:,.0f}"),
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
    prevent_initial_call=True,
)
def update_leverage(first_render, date_val, price_val, model, q,
                    rb_val, rl_val, H_val, c_val):
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
        sp = floor_price(model, q, sell_date)
    except (KeyError, AttributeError, ValueError) as e:
        return (
            {}, html.Div(f"Model unavailable: {e}", className="alert alert-warning"),
            html.Div()
        )

    max_pay = P_max(sp, H_yr, c_dec)
    implied_c = implied_cagr(sp, price, H_yr)

    p = {
        "lev_price": price, "lev_date": buy_date,
        "lev_model": model, "lev_floor_q": q,
        "lev_rb": rb, "lev_rl": rl,
        "lev_horizon": H_yr, "lev_cagr": c_pct,
        "palette": "default",
    }
    fig = build_leverage_figure(p)
    ro = _readout(buy_date, price, sell_date, sp, H_yr, c_dec, max_pay, implied_c, model, q)
    tbl = _table(buy_date, model, q, r_b_dec, r_l_dec, c_dec, H_yr)
    return fig, ro, tbl
