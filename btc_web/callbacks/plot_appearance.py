"""Plot Appearance — pt_size / pt_alpha reset callback.

The rest of the Plot Appearance control plane (the 30 inputs and 5 reset
buttons across the 5 chart tabs) is owned by btc_web/assets/plot_appearance.js
as a pure JS/DOM implementation against localStorage["plot-appearance"].
See docs/superpowers/specs/2026-04-10-plot-appearance-control-plane-design.md.

This file keeps only the bubble-tab pt_size / pt_alpha reset callback, which
must stay in Dash because pt_size and pt_alpha are server-side bubble figure
parameters read via Dash Input bub-ptsize.value / bub-ptalpha.value — not
from localStorage.
"""
from dash import Input, Output
import _app_ctx


# Any of the 5 reset buttons resets the bubble tab's pt_size / pt_alpha
# sliders to their factory defaults (size=10, alpha=0.5). The JS layer
# handles the other 6 Plot Appearance fields directly; this callback is
# the one remaining Dash-side responsibility.
#
# NOTE: registered as 5 SEPARATE single-Input callbacks instead of one
# multi-Input callback. Dash 4.0 has a bug where a clientside callback
# combining allow_duplicate=True + multiple Inputs + prevent_initial_call
# silently fails to fire even when n_clicks increments. The probe
# callback confirmed n_clicks was propagating correctly to Dash's
# internal store; only the multi-input version of this callback was
# being skipped. Splitting into per-button callbacks sidesteps the bug.
_RESET_BODY = """
function(n) {
    if (!n) {
        return [window.dash_clientside.no_update,
                window.dash_clientside.no_update];
    }
    return [10, 0.5];
}
"""

for _prefix in ("bub", "dca", "ret", "sc", "cp"):
    _app_ctx.app.clientside_callback(
        _RESET_BODY,
        Output("bub-ptsize", "value", allow_duplicate=True),
        Output("bub-ptalpha", "value", allow_duplicate=True),
        Input(f"{_prefix}-plot-appearance-reset", "n_clicks"),
        prevent_initial_call=True,
    )
