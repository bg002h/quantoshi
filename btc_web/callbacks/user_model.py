"""User-defined model: simple input fields + Draw/Delete buttons."""

from dash import Input, Output, State, callback, no_update, ctx

import _app_ctx
from btc_core import UserModel

_GENESIS_YR = 2009.56  # 2009-07-25


# ══════════════════════════════════════════════════════════════════════════════
# 1. Click data point → auto-fill the next empty input pair
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("um-p1-year", "value", allow_duplicate=True),
    Output("um-p1-price", "value", allow_duplicate=True),
    Output("um-p2-year", "value", allow_duplicate=True),
    Output("um-p2-price", "value", allow_duplicate=True),
    Input("bubble-graph", "clickData"),
    State("um-p1-year", "value"),
    State("um-p1-price", "value"),
    State("um-p2-year", "value"),
    State("um-p2-price", "value"),
    prevent_initial_call=True,
)
def autofill_from_click(click_data, p1y, p1p, p2y, p2p):
    """Click a data point → fill the next empty point pair."""
    if not click_data or not click_data.get("points"):
        return no_update, no_update, no_update, no_update

    pt = click_data["points"][0]
    year = round(_GENESIS_YR + pt["x"], 1)
    price = round(pt["y"], 2)

    # Fill point 1 if empty, else point 2
    if not p1y and not p1p:
        return year, price, no_update, no_update
    elif not p2y and not p2p:
        return no_update, no_update, year, price
    else:
        # Both filled — overwrite point 2 (user is adjusting)
        return no_update, no_update, year, price


# ══════════════════════════════════════════════════════════════════════════════
# 2. Draw button → construct UserModel from inputs
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

    from btc_core import yr_to_t
    M = _app_ctx.M
    t1 = yr_to_t(float(p1y), M.genesis)
    t2 = yr_to_t(float(p2y), M.genesis)

    model = UserModel.from_points(
        t1=t1, p1=float(p1p),
        t2=t2, p2=float(p2p),
        price_years=M.price_years,
        price_prices=M.price_prices,
        quantiles=list(M.QR_QUANTILES),
    )
    store_data = model.to_store_dict()

    # Auto-check U1 in Display Models
    ms = list(cur_model_show or [])
    if "u1" not in ms:
        ms.append("u1")
    return store_data, ms


# ══════════════════════════════════════════════════════════════════════════════
# 3. Delete button → clear model
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
# 4. Dynamic Display Models checklist injection
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_SHOW_PREFIXES = ["bub", "dca", "ret", "sc"]  # heatmap uses pill bar, not checklist


@callback(
    [Output(f"{p}-model-show", "options", allow_duplicate=True) for p in _MODEL_SHOW_PREFIXES],
    Input("user-model-store", "data"),
    [State(f"{p}-model-show", "options") for p in _MODEL_SHOW_PREFIXES],
    prevent_initial_call=True,
)
def inject_user_model_option(user_data, *current_options_list):
    """Add or remove U1 option from Display Models checklists on all tabs."""
    results = []
    u1_opt = {"label": " U1 (User)", "value": "u1"}
    for opts in current_options_list:
        opts = list(opts or [])
        opts = [o for o in opts if o.get("value") != "u1"]
        if user_data:
            opts.append(u1_opt)
        results.append(opts)
    return results
