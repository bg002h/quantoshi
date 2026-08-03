"""One-tap axes presets (Tab 1) — one clientside callback per preset.

MUST be one single-Input clientside callback per preset: Dash 4.0 silently
fails to fire a clientside callback that combines allow_duplicate=True with
MULTIPLE Inputs and prevent_initial_call. See callbacks/plot_appearance.py:22-28.

Registration must go through _app_ctx.app.clientside_callback, not the
module-level dash.clientside_callback, which registers into
dash._callback.GLOBAL_CALLBACK_MAP instead of app.callback_map.

Spec: docs/superpowers/specs/2026-08-03-axes-presets-design.md
"""
from dash import Input, Output, State

import _app_ctx
from layout.bubble import AXES_CONTROL_IDS, AXES_PRESETS

for _preset in AXES_PRESETS:
    _app_ctx.app.clientside_callback(
        _preset["js"],
        # Fresh Output objects per iteration -- do not hoist and reuse.
        *[Output(cid, "value", allow_duplicate=True)
          for cid in AXES_CONTROL_IDS],
        Input(f"bub-axes-preset-{_preset['key']}", "n_clicks"),
        *[State(cid, prop) for cid, prop in _preset["states"]],
        prevent_initial_call=True,
    )

    # Optional visibility rule: a preset that would be inert in some X-range
    # state hides its own button rather than offering a dead control.
    # Deliberately NOT prevent_initial_call -- the button must already be in the
    # right state on page load, before any tap. Safe to fire on load because
    # this Output is not allow_duplicate (nothing else writes these styles);
    # allow_duplicate + prevent_initial_call=False crashes gunicorn.
    if _preset.get("hide_js"):
        _app_ctx.app.clientside_callback(
            _preset["hide_js"],
            Output(f"bub-axes-preset-{_preset['key']}", "style"),
            Input("bub-xrange", "value"),
        )
