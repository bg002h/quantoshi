"""Model Scanner callbacks — bidirectional price/date/quantile lookup."""

import numpy as np
import pandas as pd
from dash import html, Input, Output, State, callback, no_update, ctx, ALL

import _app_ctx
from btc_core import today_t, fmt_price


def _solve_date(model, q_frac, target_price):
    """Root-find t where model.price_at(q, t) = target_price."""
    from scipy.optimize import brentq
    log_target = np.log10(max(target_price, 1e-10))
    def f(t):
        return np.log10(max(float(model.price_at(q_frac, t)), 1e-10)) - log_target
    try:
        t = brentq(f, 0.5, 72.0)
        genesis = _app_ctx.M.genesis
        date = genesis + pd.Timedelta(days=t * 365.25)
        return date.strftime("%Y-%m-%d")
    except (ValueError, RuntimeError):
        return "\u2014"


def _output_from_history(history):
    """The output field is whichever of p/d/q is NOT in the last-two-edited list."""
    all_fields = {"p", "d", "q"}
    remaining = all_fields - set(history[-2:] if len(history) >= 2 else history)
    if len(remaining) == 1:
        return remaining.pop()
    # Ambiguous — default to q
    return "q"


@callback(
    Output("scan-output-field", "data"),
    Output("scan-results", "children"),
    Output("scan-price-hint", "style"),
    Input("scan-price", "value"),
    Input("scan-date", "value"),
    Input("scan-q", "value"),
    State("scan-output-field", "data"),
    Input("btc-price-store", "data"),
    prevent_initial_call=False,
)
def update_scanner(price_val, date_val, q_val, edit_history, live_price):
    """Compute the missing variable across all models."""
    trigger = ctx.triggered_id if ctx.triggered_id else None
    genesis = _app_ctx.M.genesis

    # edit_history is a list of the last 2 edited fields, e.g. ["p", "d"]
    if not isinstance(edit_history, list):
        edit_history = ["p", "d"]  # default: price+date edited → q is output

    # Update history based on what was just edited
    trigger_map = {"scan-price": "p", "scan-date": "d", "scan-q": "q"}
    if trigger in trigger_map:
        field = trigger_map[trigger]
        # Remove this field if already in history, then append
        edit_history = [f for f in edit_history if f != field]
        edit_history.append(field)
        # Keep only last 2
        edit_history = edit_history[-2:]

    output_field = _output_from_history(edit_history)

    # Resolve defaults
    use_live = (price_val is None or price_val == "")
    if use_live:
        price = float(live_price) if live_price else None
    else:
        price = float(price_val)

    hint_style = {"fontSize": "9px"} if use_live else {"fontSize": "9px", "display": "none"}

    if date_val is None or date_val == "":
        date_val = pd.Timestamp.today().strftime("%Y-%m-%d")

    t = (pd.to_datetime(date_val) - genesis).days / 365.25
    if t <= 0:
        t = 0.5

    q_frac = float(q_val) / 100.0 if q_val is not None and q_val != "" else None

    rows = []

    if output_field == "q" and price is not None:
        # Unfairly Cheap Line
        ucl_price = 10 ** (_app_ctx.UCL_INTERCEPT + _app_ctx.UCL_SLOPE * np.log10(t))
        ucl_ratio = price / ucl_price
        ucl_style = {"fontSize": "11px", "color": "#ff6b6b"}
        rows.append(html.Tr([
            html.Td("Unfairly Cheap Line", style=ucl_style),
            html.Td(f"{ucl_ratio:.2f}x above", style={**ucl_style, "fontWeight": "bold"}),
        ]))

        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue  # skip S2F etc — quantiles don't apply
            pct = mdl.find_percentile(t, price)
            rows.append(html.Tr([
                html.Td(mdl.name, style={"fontSize": "11px"}),
                html.Td(f"Q{pct*100:.1f}%", style={"fontSize": "11px",
                         "fontWeight": "bold"}),
            ], id={"type": "scan-row", "model": key},
               style={"cursor": "pointer"}))

    elif output_field == "p" and q_frac is not None:
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue
            try:
                p = float(mdl.price_at(q_frac, t))
                price_str = fmt_price(p)
            except Exception:
                price_str = "\u2014"
            rows.append(html.Tr([
                html.Td(mdl.name, style={"fontSize": "11px"}),
                html.Td(price_str, style={"fontSize": "11px",
                         "fontWeight": "bold"}),
            ], id={"type": "scan-row", "model": key},
               style={"cursor": "pointer"}))

    elif output_field == "d" and price is not None and q_frac is not None:
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue
            date_str = _solve_date(mdl, q_frac, price)
            rows.append(html.Tr([
                html.Td(mdl.name, style={"fontSize": "11px"}),
                html.Td(date_str, style={"fontSize": "11px",
                         "fontWeight": "bold"}),
            ], id={"type": "scan-row", "model": key},
               style={"cursor": "pointer"}))

    header_map = {"q": "Quantile", "p": "Price", "d": "Date"}
    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Model", style={"fontSize": "11px", "paddingRight": "12px"}),
            html.Th(header_map.get(output_field, ""),
                     style={"fontSize": "11px"}),
        ])),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse",
              "marginTop": "6px"}) if rows else html.Small(
                  "Enter values above", className="text-muted")

    return (edit_history, table, hint_style)


@callback(
    Output("scan-active-rows", "data"),
    Input({"type": "scan-row", "model": ALL}, "n_clicks"),
    State("scan-active-rows", "data"),
    prevent_initial_call=True,
)
def toggle_scanner_row(n_clicks_list, active):
    """Toggle a model's scanner line on/off when its row is clicked."""
    if not ctx.triggered_id:
        return no_update
    model_key = ctx.triggered_id["model"]
    active = active or []
    if model_key in active:
        active.remove(model_key)
    else:
        active.append(model_key)
    return active
