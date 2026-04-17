"""User-defined model: input fields + click-to-set context menu."""

from dash import Input, Output, State, callback, no_update

import _app_ctx
from btc_core import UserModel
from colors import CTX_MENU_BG, BLACK, _hex_alpha, CTX_MENU_BG_ALPHA, CTX_MENU_SHADOW_ALPHA

_GENESIS_YR = 2009.56  # 2009-07-25
_HIDDEN = {"display": "none"}
_CTX_VISIBLE = {
    "display": "flex", "alignItems": "center",
    "position": "absolute", "bottom": "14px", "left": "14px", "zIndex": 20,
    "backgroundColor": _hex_alpha(CTX_MENU_BG, CTX_MENU_BG_ALPHA), "borderRadius": "8px",
    "padding": "6px 10px", "boxShadow": f"0 2px 12px {_hex_alpha(BLACK, CTX_MENU_SHADOW_ALPHA)}",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Click data point → show context menu with year+price
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("um-clicked-point", "data"),
    Output("um-ctx-menu", "style"),
    Output("um-ctx-label", "children"),
    Input("bubble-graph", "clickData"),
    State("bub-model-show", "value"),
    prevent_initial_call=True,
)
def on_data_click(click_data, model_show):
    if not click_data or not click_data.get("points"):
        return no_update, no_update, no_update
    # Only show P1/P2 picker when U1 is enabled
    if "u1" not in (model_show or []):
        return no_update, _HIDDEN, no_update

    pt = click_data["points"][0]
    t_val = pt["x"]
    price = pt["y"]
    year = round(_GENESIS_YR + t_val, 2)

    label = f"{year}  ${price:,.2f}" if price >= 100 else f"{year}  ${price:.4f}" if price >= 1 else f"{year}  ${price:.6f}"
    return {"year": year, "price": price}, _CTX_VISIBLE, label


# ══════════════════════════════════════════════════════════════════════════════
# 2. "P1" / "P2" buttons → fill inputs + hide menu
# ══════════════════════════════════════════════════════════════════════════════

_SET_POINT_JS = """function(n, pt) {
    if (!pt) return [window.dash_clientside.no_update,
                     window.dash_clientside.no_update,
                     window.dash_clientside.no_update,
                     window.dash_clientside.no_update,
                     window.dash_clientside.no_update];
    var yr = pt.year;
    var pr = Math.round(pt.price * 1e6) / 1e6;
    var fmt;
    if (pr >= 100) fmt = pr.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    else if (pr >= 1) fmt = pr.toFixed(4);
    else fmt = pr.toFixed(6);
    return [yr, pr, String(yr), fmt, {display: 'none'}];
}"""

_app_ctx.app.clientside_callback(
    _SET_POINT_JS,
    Output("um-p1-year", "data", allow_duplicate=True),
    Output("um-p1-price", "data", allow_duplicate=True),
    Output("um-p1-year-display", "children"),
    Output("um-p1-price-display", "children"),
    Output("um-ctx-menu", "style", allow_duplicate=True),
    Input("um-ctx-p1", "n_clicks"),
    State("um-clicked-point", "data"),
    prevent_initial_call=True,
)

_app_ctx.app.clientside_callback(
    _SET_POINT_JS,
    Output("um-p2-year", "data", allow_duplicate=True),
    Output("um-p2-price", "data", allow_duplicate=True),
    Output("um-p2-year-display", "children"),
    Output("um-p2-price-display", "children"),
    Output("um-ctx-menu", "style", allow_duplicate=True),
    Input("um-ctx-p2", "n_clicks"),
    State("um-clicked-point", "data"),
    prevent_initial_call=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Delete button → clear model + inputs
# ══════════════════════════════════════════════════════════════════════════════

_app_ctx.app.clientside_callback(
    """function(n) {
        return [null, null, null, null, null,
                '\\u2014', '\\u2014', '\\u2014', '\\u2014'];
    }""",
    Output("user-model-store", "data", allow_duplicate=True),
    Output("um-p1-year", "data", allow_duplicate=True),
    Output("um-p1-price", "data", allow_duplicate=True),
    Output("um-p2-year", "data", allow_duplicate=True),
    Output("um-p2-price", "data", allow_duplicate=True),
    Output("um-p1-year-display", "children", allow_duplicate=True),
    Output("um-p1-price-display", "children", allow_duplicate=True),
    Output("um-p2-year-display", "children", allow_duplicate=True),
    Output("um-p2-price-display", "children", allow_duplicate=True),
    Input("um-delete-btn", "n_clicks"),
    prevent_initial_call=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Dynamic Display Models checklist injection
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_SHOW_PREFIXES = ["bub", "dca", "ret", "sc"]


# U₁ option is now statically included in the Display Models checklist.
# Show/hide the P1/P2 panel based on whether U1 is checked.
_app_ctx.app.clientside_callback(
    """function(model_show) {
        var has_u1 = (model_show || []).indexOf('u1') >= 0;
        return has_u1 ? {} : {display: 'none'};
    }""",
    Output("um-panel-wrap", "style"),
    Input("bub-model-show", "value"),
)


# Auto-draw: when both P1 and P2 exist, auto-construct the model
@callback(
    Output("user-model-store", "data", allow_duplicate=True),
    Input("um-p1-year", "data"),
    Input("um-p1-price", "data"),
    Input("um-p2-year", "data"),
    Input("um-p2-price", "data"),
    prevent_initial_call=True,
)
def auto_draw(p1y, p1p, p2y, p2p):
    """Auto-construct UserModel when both P1 and P2 are set."""
    if not all([p1y, p1p, p2y, p2p]):
        return no_update
    M = _app_ctx.M
    t1 = float(p1y) - _GENESIS_YR
    t2 = float(p2y) - _GENESIS_YR
    model = UserModel.from_points(
        t1=t1, p1=float(p1p), t2=t2, p2=float(p2p),
        price_years=M.price_years, price_prices=M.price_prices,
        quantiles=list(M.QR_QUANTILES),
    )
    return model.to_store_dict()
