"""User-defined model: draw-mode state machine + model construction."""

from dash import Input, Output, State, callback, clientside_callback, no_update, ctx

import _app_ctx
from btc_core import UserModel

_HIDDEN = {"display": "none"}
_MENU_STYLE = {
    "display": "flex", "position": "absolute", "bottom": "60px",
    "left": "50%", "transform": "translateX(-50%)", "zIndex": 15,
    "backgroundColor": "rgba(30,30,40,0.95)", "borderRadius": "8px",
    "padding": "8px 12px", "boxShadow": "0 4px 16px rgba(0,0,0,0.5)",
    "whiteSpace": "nowrap",
}


def _idle_state():
    """Return an idle draw-mode state."""
    return {"phase": "idle", "point1": None, "point2": None,
            "pre_draw_zoom": None}


# ══════════════════════════════════════════════════════════════════════════════
# 1. FAB click → toggle draw mode or show model menu
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("draw-mode-store", "data", allow_duplicate=True),
    Output("draw-confirm-menu", "style", allow_duplicate=True),
    Output("draw-model-menu", "style", allow_duplicate=True),
    Output("draw-toast", "className", allow_duplicate=True),
    Output("user-model-fab", "className", allow_duplicate=True),
    Input("user-model-fab", "n_clicks"),
    State("draw-mode-store", "data"),
    State("user-model-store", "data"),
    prevent_initial_call=True,
)
def on_fab_click(n_clicks, draw_state, model_data):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update

    phase = (draw_state or {}).get("phase", "idle")

    # Any draw phase → abort
    if phase in ("placing_p1", "confirming_p1", "placing_p2", "confirming_p2"):
        return _idle_state(), _HIDDEN, _HIDDEN, "", ""

    # showing_menu → dismiss
    if phase == "showing_menu":
        return _idle_state(), _HIDDEN, _HIDDEN, "", ""

    # idle + has model → show redraw/delete menu
    if model_data:
        new_state = dict(draw_state or {})
        new_state["phase"] = "showing_menu"
        return new_state, _HIDDEN, _MENU_STYLE, "", ""

    # idle + no model → enter draw mode
    new_state = {"phase": "placing_p1", "point1": None, "point2": None,
                 "pre_draw_zoom": None}
    return new_state, _HIDDEN, _HIDDEN, "visible", "draw-active"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Redraw / Delete / Dismiss buttons
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("draw-mode-store", "data", allow_duplicate=True),
    Output("user-model-store", "data", allow_duplicate=True),
    Output("draw-model-menu", "style", allow_duplicate=True),
    Output("draw-toast", "className", allow_duplicate=True),
    Output("user-model-fab", "className", allow_duplicate=True),
    Input("draw-redraw-btn", "n_clicks"),
    Input("draw-delete-btn", "n_clicks"),
    Input("draw-dismiss-btn", "n_clicks"),
    State("draw-mode-store", "data"),
    prevent_initial_call=True,
)
def on_model_menu(redraw_clicks, delete_clicks, dismiss_clicks, draw_state):
    triggered = ctx.triggered_id

    if triggered == "draw-redraw-btn":
        new_state = {"phase": "placing_p1", "point1": None, "point2": None,
                     "pre_draw_zoom": None}
        return new_state, None, _HIDDEN, "visible", "draw-active"

    if triggered == "draw-delete-btn":
        return _idle_state(), None, _HIDDEN, "", ""

    # dismiss
    return _idle_state(), no_update, _HIDDEN, "", ""


# ══════════════════════════════════════════════════════════════════════════════
# 3. Accept / Adjust / Cancel buttons
# ══════════════════════════════════════════════════════════════════════════════

# Genesis year fraction for t→year conversion
_GENESIS_YR = 2009.56  # 2009-07-25 ≈ 2009.56

@callback(
    Output("draw-mode-store", "data", allow_duplicate=True),
    Output("draw-confirm-menu", "style", allow_duplicate=True),
    Output("user-model-store", "data", allow_duplicate=True),
    Output("user-model-fab", "className", allow_duplicate=True),
    Output("draw-toast", "className", allow_duplicate=True),
    Output("bub-xrange", "value", allow_duplicate=True),
    Output("bub-yrange", "value", allow_duplicate=True),
    Output("bub-model-show", "value", allow_duplicate=True),
    Input("draw-accept-btn", "n_clicks"),
    Input("draw-adjust-btn", "n_clicks"),
    Input("draw-cancel-btn", "n_clicks"),
    State("draw-mode-store", "data"),
    State("bub-xrange", "value"),
    State("bub-yrange", "value"),
    State("bub-model-show", "value"),
    prevent_initial_call=True,
)
def on_confirm_action(accept, adjust, cancel, draw_state, cur_xrange, cur_yrange, cur_model_show):
    triggered = ctx.triggered_id
    phase = (draw_state or {}).get("phase", "idle")
    cur_xrange = cur_xrange or [2012, 2030]
    cur_yrange = cur_yrange or [0, 7]

    # no_update shorthand for model-show (8th output)
    _MS = no_update

    if triggered == "draw-cancel-btn":
        orig = (draw_state or {}).get("pre_draw_zoom")
        xr = orig["xrange"] if orig else no_update
        yr = orig["yrange"] if orig else no_update
        if phase == "confirming_p1":
            new_state = dict(draw_state)
            new_state["phase"] = "placing_p1"
            new_state["point1"] = None
            return new_state, _HIDDEN, no_update, "draw-active", "", xr, yr, _MS
        elif phase == "confirming_p2":
            new_state = dict(draw_state)
            new_state["phase"] = "placing_p2"
            new_state["point2"] = None
            return new_state, _HIDDEN, no_update, "draw-active", "", xr, yr, _MS
        return no_update, _HIDDEN, no_update, "draw-active", "", no_update, no_update, _MS

    if triggered == "draw-adjust-btn":
        import math
        pt = None
        if phase == "confirming_p1" and (draw_state or {}).get("point1"):
            pt = draw_state["point1"]
        elif phase == "confirming_p2" and (draw_state or {}).get("point2"):
            pt = draw_state["point2"]

        if pt:
            new_state = dict(draw_state)
            if not new_state.get("pre_draw_zoom"):
                new_state["pre_draw_zoom"] = {"xrange": list(cur_xrange), "yrange": list(cur_yrange)}

            click_year = _GENESIS_YR + pt["t"]
            click_log_price = math.log10(max(pt["price"], 1e-10))

            x_lo, x_hi = float(cur_xrange[0]), float(cur_xrange[1])
            y_lo, y_hi = float(cur_yrange[0]), float(cur_yrange[1])
            x_half = (x_hi - x_lo) / 4
            y_half = (y_hi - y_lo) / 4

            new_xrange = [max(2010, round(click_year - x_half)),
                          min(2080, round(click_year + x_half))]
            new_yrange = [max(-2, round(click_log_price - y_half, 1)),
                          min(9, round(click_log_price + y_half, 1))]

            new_state["phase"] = "placing_p1" if phase == "confirming_p1" else "placing_p2"
            new_state.pop("_adjust_zoom", None)
            return new_state, _HIDDEN, no_update, "draw-active", "", new_xrange, new_yrange, _MS

        new_state = dict(draw_state)
        new_state["phase"] = "placing_p1" if phase == "confirming_p1" else "placing_p2"
        return new_state, _HIDDEN, no_update, "draw-active", "", no_update, no_update, _MS

    if triggered == "draw-accept-btn":
        if phase == "confirming_p1":
            new_state = dict(draw_state)
            new_state["phase"] = "placing_p2"
            orig = new_state.get("pre_draw_zoom")
            xr = orig["xrange"] if orig else no_update
            yr = orig["yrange"] if orig else no_update
            new_state["pre_draw_zoom"] = None
            return new_state, _HIDDEN, no_update, "draw-active", "", xr, yr, _MS
        elif phase == "confirming_p2":
            p1 = draw_state["point1"]
            p2 = draw_state["point2"]
            M = _app_ctx.M
            model = UserModel.from_points(
                t1=p1["t"], p1=p1["price"],
                t2=p2["t"], p2=p2["price"],
                price_years=M.price_years,
                price_prices=M.price_prices,
                quantiles=list(M.QR_QUANTILES),
            )
            store_data = model.to_store_dict()
            orig = (draw_state or {}).get("pre_draw_zoom")
            xr = orig["xrange"] if orig else no_update
            yr = orig["yrange"] if orig else no_update
            # Auto-check U1 in Display Models
            ms = list(cur_model_show or [])
            if "u1" not in ms:
                ms.append("u1")
            return _idle_state(), _HIDDEN, store_data, "", "", xr, yr, ms

    return no_update, _HIDDEN, no_update, "", "", no_update, no_update, _MS


# ══════════════════════════════════════════════════════════════════════════════
# 4. Tab switch → auto-cancel draw mode
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("draw-mode-store", "data", allow_duplicate=True),
    Output("draw-confirm-menu", "style", allow_duplicate=True),
    Output("draw-model-menu", "style", allow_duplicate=True),
    Output("user-model-fab", "className", allow_duplicate=True),
    Input("main-tabs", "active_tab"),
    State("draw-mode-store", "data"),
    prevent_initial_call=True,
)
def on_tab_switch(active_tab, draw_state):
    phase = (draw_state or {}).get("phase", "idle")
    if phase != "idle":
        return _idle_state(), _HIDDEN, _HIDDEN, ""
    return no_update, no_update, no_update, no_update


# ══════════════════════════════════════════════════════════════════════════════
# 5. Chart click → capture point during draw mode
# ══════════════════════════════════════════════════════════════════════════════

# Bridge: hidden button click → read global coords → write to store
clientside_callback(
    """
    function(n) {
        if (window._rawChartClick) {
            var data = window._rawChartClick;
            window._rawChartClick = null;
            return data;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("raw-chart-click", "data"),
    Input("raw-click-trigger", "n_clicks"),
    prevent_initial_call=True,
)


@callback(
    Output("draw-mode-store", "data", allow_duplicate=True),
    Output("draw-confirm-menu", "style", allow_duplicate=True),
    Input("raw-chart-click", "data"),
    State("draw-mode-store", "data"),
    prevent_initial_call=True,
)
def on_chart_click(click_data, draw_state):
    """Capture raw chart click (from JS pixel→data conversion, no trace snapping)."""
    phase = (draw_state or {}).get("phase", "idle")

    if phase not in ("placing_p1", "placing_p2"):
        return no_update, no_update

    if not click_data or not click_data.get("t") or not click_data.get("price"):
        return no_update, no_update

    clicked = {"t": click_data["t"], "price": click_data["price"]}
    new_draw = dict(draw_state)
    new_draw.pop("_adjust_zoom", None)

    if phase == "placing_p1":
        new_draw["point1"] = clicked
        new_draw["phase"] = "confirming_p1"
    else:
        new_draw["point2"] = clicked
        new_draw["phase"] = "confirming_p2"

    return new_draw, _MENU_STYLE


# ══════════════════════════════════════════════════════════════════════════════
# 6. Dynamic Display Models checklist injection
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
        # Remove any existing u1 option
        opts = [o for o in opts if o.get("value") != "u1"]
        # Add if user model exists
        if user_data:
            opts.append(u1_opt)
        results.append(opts)
    return results
