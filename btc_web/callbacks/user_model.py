"""User-defined model: draw-mode state machine + model construction."""

from dash import Input, Output, State, callback, no_update, ctx

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

@callback(
    Output("draw-mode-store", "data", allow_duplicate=True),
    Output("draw-confirm-menu", "style", allow_duplicate=True),
    Output("user-model-store", "data", allow_duplicate=True),
    Output("user-model-fab", "className", allow_duplicate=True),
    Output("draw-toast", "className", allow_duplicate=True),
    Input("draw-accept-btn", "n_clicks"),
    Input("draw-adjust-btn", "n_clicks"),
    Input("draw-cancel-btn", "n_clicks"),
    State("draw-mode-store", "data"),
    prevent_initial_call=True,
)
def on_confirm_action(accept, adjust, cancel, draw_state):
    triggered = ctx.triggered_id
    phase = (draw_state or {}).get("phase", "idle")

    if triggered == "draw-cancel-btn":
        if phase == "confirming_p1":
            new_state = dict(draw_state)
            new_state["phase"] = "placing_p1"
            new_state["point1"] = None
            return new_state, _HIDDEN, no_update, "draw-active", ""
        elif phase == "confirming_p2":
            new_state = dict(draw_state)
            new_state["phase"] = "placing_p2"
            new_state["point2"] = None
            return new_state, _HIDDEN, no_update, "draw-active", ""
        return no_update, _HIDDEN, no_update, "draw-active", ""

    if triggered == "draw-adjust-btn":
        if phase == "confirming_p1":
            new_state = dict(draw_state)
            new_state["phase"] = "placing_p1"
            new_state["_adjust_zoom"] = True
            new_state["_zoom_level"] = (draw_state or {}).get("_zoom_level", 0) + 1
            return new_state, _HIDDEN, no_update, "draw-active", ""
        elif phase == "confirming_p2":
            new_state = dict(draw_state)
            new_state["phase"] = "placing_p2"
            new_state["_adjust_zoom"] = True
            new_state["_zoom_level"] = (draw_state or {}).get("_zoom_level", 0) + 1
            return new_state, _HIDDEN, no_update, "draw-active", ""
        return no_update, _HIDDEN, no_update, "draw-active", ""

    if triggered == "draw-accept-btn":
        if phase == "confirming_p1":
            new_state = dict(draw_state)
            new_state["phase"] = "placing_p2"
            new_state.pop("_adjust_zoom", None)
            new_state.pop("_zoom_level", None)  # reset zoom for point 2
            return new_state, _HIDDEN, no_update, "draw-active", ""
        elif phase == "confirming_p2":
            # Construct model from two accepted points
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
            return _idle_state(), _HIDDEN, store_data, "", ""

    return no_update, _HIDDEN, no_update, "", ""


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
    phase = (draw_state or {}).get("phase", "idle")

    if phase not in ("placing_p1", "placing_p2"):
        return no_update, no_update

    if not click_data or not click_data.get("points"):
        return no_update, no_update

    pt = click_data["points"][0]
    clicked = {"t": pt["x"], "price": pt["y"]}
    new_draw = dict(draw_state)
    new_draw.pop("_adjust_zoom", None)  # clear zoom flag from previous adjust

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
