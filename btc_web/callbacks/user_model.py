"""User-defined model: input fields + click-to-set context menu."""

from dash import Input, Output, State, callback, no_update, ctx
from btc_core import fmt_price

import _app_ctx
from btc_core import UserModel

_GENESIS_YR = 2009.56  # 2009-07-25
_HIDDEN = {"display": "none"}
_CTX_VISIBLE = {
    "display": "flex", "alignItems": "center",
    "position": "absolute", "bottom": "14px", "left": "14px", "zIndex": 20,
    "backgroundColor": "rgba(30,30,40,0.95)", "borderRadius": "8px",
    "padding": "6px 10px", "boxShadow": "0 2px 12px rgba(0,0,0,0.5)",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Click data point → show context menu with year+price
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("um-clicked-point", "data"),
    Output("um-ctx-menu", "style"),
    Output("um-ctx-label", "children"),
    Input("bubble-graph", "clickData"),
    prevent_initial_call=True,
)
def on_data_click(click_data):
    if not click_data or not click_data.get("points"):
        return no_update, no_update, no_update

    pt = click_data["points"][0]
    t_val = pt["x"]
    price = pt["y"]
    year = round(_GENESIS_YR + t_val, 1)

    label = f"{year}  ${price:,.0f}" if price >= 1 else f"{year}  ${price:.4f}"
    return {"year": year, "price": price}, _CTX_VISIBLE, label


# ══════════════════════════════════════════════════════════════════════════════
# 2. "P1" / "P2" buttons → fill inputs + hide menu
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("um-p1-year", "value", allow_duplicate=True),
    Output("um-p1-price", "value", allow_duplicate=True),
    Output("um-ctx-menu", "style", allow_duplicate=True),
    Input("um-ctx-p1", "n_clicks"),
    State("um-clicked-point", "data"),
    prevent_initial_call=True,
)
def set_p1(n, pt):
    if not pt:
        return no_update, no_update, no_update
    return pt["year"], round(pt["price"], 2), _HIDDEN


@callback(
    Output("um-p2-year", "value", allow_duplicate=True),
    Output("um-p2-price", "value", allow_duplicate=True),
    Output("um-ctx-menu", "style", allow_duplicate=True),
    Input("um-ctx-p2", "n_clicks"),
    State("um-clicked-point", "data"),
    prevent_initial_call=True,
)
def set_p2(n, pt):
    if not pt:
        return no_update, no_update, no_update
    return pt["year"], round(pt["price"], 2), _HIDDEN


# ══════════════════════════════════════════════════════════════════════════════
# 3. Draw button → construct UserModel
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("user-model-store", "data", allow_duplicate=True),
    Output("bub-model-show", "value", allow_duplicate=True),
    Input("um-draw-btn", "n_clicks"),
    State("um-p1-year", "value"),
    State("um-p1-price", "value"),
    State("um-p2-year", "value"),
    State("um-p2-price", "value"),
    State("bub-model-show", "value"),
    prevent_initial_call=True,
)
def draw_user_model(n_clicks, p1y, p1p, p2y, p2p, cur_model_show):
    if not all([p1y, p1p, p2y, p2p]):
        return no_update, no_update

    M = _app_ctx.M
    # Direct conversion: year = _GENESIS_YR + t, so t = year - _GENESIS_YR
    # This is the exact reverse of how the context menu computes the year
    t1 = float(p1y) - _GENESIS_YR
    t2 = float(p2y) - _GENESIS_YR

    model = UserModel.from_points(
        t1=t1, p1=float(p1p),
        t2=t2, p2=float(p2p),
        price_years=M.price_years,
        price_prices=M.price_prices,
        quantiles=list(M.QR_QUANTILES),
    )
    store_data = model.to_store_dict()

    ms = list(cur_model_show or [])
    if "u1" not in ms:
        ms.append("u1")
    return store_data, ms


# ══════════════════════════════════════════════════════════════════════════════
# 4. Delete button → clear model + inputs
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("user-model-store", "data", allow_duplicate=True),
    Output("um-p1-year", "value", allow_duplicate=True),
    Output("um-p1-price", "value", allow_duplicate=True),
    Output("um-p2-year", "value", allow_duplicate=True),
    Output("um-p2-price", "value", allow_duplicate=True),
    Input("um-delete-btn", "n_clicks"),
    prevent_initial_call=True,
)
def delete_user_model(n_clicks):
    return None, None, None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# 5. Dynamic Display Models checklist injection
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_SHOW_PREFIXES = ["bub", "dca", "ret", "sc"]


@callback(
    [Output(f"{p}-model-show", "options", allow_duplicate=True) for p in _MODEL_SHOW_PREFIXES],
    Input("user-model-store", "data"),
    [State(f"{p}-model-show", "options") for p in _MODEL_SHOW_PREFIXES],
    prevent_initial_call=True,
)
def inject_user_model_option(user_data, *current_options_list):
    results = []
    u1_opt = {"label": " U\u2081 (User)", "value": "u1"}
    for opts in current_options_list:
        opts = list(opts or [])
        opts = [o for o in opts if o.get("value") != "u1"]
        if user_data:
            opts.append(u1_opt)
        results.append(opts)
    return results
