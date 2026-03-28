"""User-defined model: draw-mode state machine + model construction."""

from dash import Input, Output, State, callback, no_update, ctx

import _app_ctx
from btc_core import UserModel

_HIDDEN = {"display": "none"}
_MENU_STYLE = {
    "display": "flex", "position": "absolute", "bottom": "14px",
    "left": "14px", "zIndex": 15,
    "backgroundColor": "rgba(30,30,40,0.95)", "borderRadius": "8px",
    "padding": "4px 8px", "boxShadow": "0 2px 8px rgba(0,0,0,0.4)",
    "whiteSpace": "nowrap", "gap": "4px",
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
    Output("bub-model-show", "value", allow_duplicate=True),
    Input("draw-accept-btn", "n_clicks"),
    Input("draw-adjust-btn", "n_clicks"),
    Input("draw-cancel-btn", "n_clicks"),
    State("draw-mode-store", "data"),
    State("bub-model-show", "value"),
    prevent_initial_call=True,
)
def on_confirm_action(accept, adjust, cancel, draw_state, cur_model_show):
    triggered = ctx.triggered_id
    phase = (draw_state or {}).get("phase", "idle")
    _MS = no_update  # model-show shorthand

    if triggered == "draw-cancel-btn":
        new_state = dict(draw_state)
        new_state.pop("_zoom_range", None)  # clear zoom
        if phase == "confirming_p1":
            new_state["phase"] = "placing_p1"
            new_state["point1"] = None
            return new_state, _HIDDEN, no_update, "draw-active", "", _MS
        elif phase == "confirming_p2":
            new_state["phase"] = "placing_p2"
            new_state["point2"] = None
            return new_state, _HIDDEN, no_update, "draw-active", "", _MS
        return no_update, _HIDDEN, no_update, "draw-active", "", _MS

    if triggered == "draw-adjust-btn":
        import math
        from btc_core import yr_to_t
        pt = None
        if phase == "confirming_p1" and (draw_state or {}).get("point1"):
            pt = draw_state["point1"]
        elif phase == "confirming_p2" and (draw_state or {}).get("point2"):
            pt = draw_state["point2"]

        if pt:
            new_state = dict(draw_state)
            # Get current visible range (from previous zoom or full chart)
            prev_zr = new_state.get("_zoom_range")
            if prev_zr:
                cur_t_lo, cur_t_hi = prev_zr["t_lo"], prev_zr["t_hi"]
                cur_y_lo, cur_y_hi = prev_zr["y_lo"], prev_zr["y_hi"]
            else:
                # Full chart range — approximate from typical defaults
                cur_t_lo = yr_to_t(2010, _app_ctx.M.genesis)
                cur_t_hi = yr_to_t(2080, _app_ctx.M.genesis)
                cur_y_lo = 0.01  # 10^-2
                cur_y_hi = 1e9   # 10^9

            # 2x zoom in log-space, centered on point
            log_t_lo = math.log10(max(cur_t_lo, 0.01))
            log_t_hi = math.log10(max(cur_t_hi, 0.02))
            log_y_lo = math.log10(max(cur_y_lo, 1e-10))
            log_y_hi = math.log10(max(cur_y_hi, 1e-10))
            log_cx = math.log10(max(pt["t"], 0.01))
            log_cy = math.log10(max(pt["price"], 1e-10))

            t_half = (log_t_hi - log_t_lo) / 4
            y_half = (log_y_hi - log_y_lo) / 4

            new_state["_zoom_range"] = {
                "t_lo": 10 ** (log_cx - t_half),
                "t_hi": 10 ** (log_cx + t_half),
                "y_lo": 10 ** (log_cy - y_half),
                "y_hi": 10 ** (log_cy + y_half),
            }
            new_state["phase"] = "placing_p1" if phase == "confirming_p1" else "placing_p2"
            return new_state, _HIDDEN, no_update, "draw-active", "", _MS

        new_state = dict(draw_state)
        new_state["phase"] = "placing_p1" if phase == "confirming_p1" else "placing_p2"
        return new_state, _HIDDEN, no_update, "draw-active", "", _MS

    if triggered == "draw-accept-btn":
        if phase == "confirming_p1":
            new_state = dict(draw_state)
            new_state["phase"] = "placing_p2"
            new_state.pop("_zoom_range", None)  # reset zoom for point 2
            return new_state, _HIDDEN, no_update, "draw-active", "", _MS
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
            # Auto-check U1 in Display Models
            ms = list(cur_model_show or [])
            if "u1" not in ms:
                ms.append("u1")
            return _idle_state(), _HIDDEN, store_data, "", "", ms

    return no_update, _HIDDEN, no_update, "", "", _MS


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

@callback(
    Output("draw-mode-store", "data", allow_duplicate=True),
    Output("draw-confirm-menu", "style", allow_duplicate=True),
    Input("bubble-graph", "clickData"),
    State("draw-mode-store", "data"),
    prevent_initial_call=True,
)
def on_chart_click(click_data, draw_state):
    """Capture chart click via Plotly clickData (snaps to 200×200 background grid)."""
    phase = (draw_state or {}).get("phase", "idle")

    if phase not in ("placing_p1", "placing_p2"):
        return no_update, no_update

    if not click_data or not click_data.get("points"):
        return no_update, no_update

    pt = click_data["points"][0]
    clicked = {"t": pt["x"], "price": pt["y"]}
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
