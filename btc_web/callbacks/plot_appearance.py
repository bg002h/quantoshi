"""Plot Appearance controls — trace thickness, grid width/color, data point color.

Rendered on every chart tab (bubble, DCA, retire, supercharger, citadel).
All tabs share a single 'plot-appearance' localStorage store, so changes
propagate across tabs. The chart_responsive.js asset reads the store
and applies values via Plotly.restyle/Plotly.relayout.
"""
import json as _json
from dash import Input, Output
import _app_ctx


_DEFAULTS = {
    "trace_width": 2.5,
    "grid_major_width": 1.0,
    "grid_major_color": "#888888",
    "grid_minor_width": 0.8,
    "grid_minor_color": "#B0B0B0",
    "pt_color": "#2C3E50",
    "pt_size": 10,
    "pt_alpha": 0.5,
}
_DEFAULTS_JSON = _json.dumps(_DEFAULTS)

_PREFIXES = ("bub", "dca", "ret", "sc", "cp")


for _prefix in _PREFIXES:
    # ── Persist control values → localStorage store ──────────────────────
    _app_ctx.app.clientside_callback(
        """
        function(tw, gmw, gmc, gnw, gnc, ptc) {
            if (tw == null && gmw == null && gmc == null && gnw == null && gnc == null && ptc == null) {
                return window.dash_clientside.no_update;
            }
            /* Preserve existing pt_size/pt_alpha */
            var cur = null;
            try { cur = JSON.parse(localStorage.getItem("plot-appearance")); } catch(e) {}
            cur = cur || {};
            return {
                trace_width: tw,
                grid_major_width: gmw,
                grid_major_color: gmc,
                grid_minor_width: gnw,
                grid_minor_color: gnc,
                pt_color: ptc,
                pt_size: cur.pt_size || 10,
                pt_alpha: cur.pt_alpha || 0.5,
            };
        }
        """,
        Output("plot-appearance", "data", allow_duplicate=True),
        Input(f"{_prefix}-plot-trace-width", "value"),
        Input(f"{_prefix}-plot-grid-major-width", "value"),
        Input(f"{_prefix}-plot-grid-major-color", "value"),
        Input(f"{_prefix}-plot-grid-minor-width", "value"),
        Input(f"{_prefix}-plot-grid-minor-color", "value"),
        Input(f"{_prefix}-plot-pt-color", "value"),
        prevent_initial_call=True,
    )

    # ── Restore control values from store on page load / cross-tab sync ──
    _app_ctx.app.clientside_callback(
        """
        function(data) {
            var NU = window.dash_clientside.no_update;
            if (!data) return [NU, NU, NU, NU, NU, NU];
            return [data.trace_width, data.grid_major_width, data.grid_major_color,
                    data.grid_minor_width, data.grid_minor_color,
                    data.pt_color || "#2C3E50"];
        }
        """,
        Output(f"{_prefix}-plot-trace-width", "value"),
        Output(f"{_prefix}-plot-grid-major-width", "value"),
        Output(f"{_prefix}-plot-grid-major-color", "value"),
        Output(f"{_prefix}-plot-grid-minor-width", "value"),
        Output(f"{_prefix}-plot-grid-minor-color", "value"),
        Output(f"{_prefix}-plot-pt-color", "value"),
        Input("plot-appearance", "data"),
        prevent_initial_call="initial_duplicate",
    )


# ══════════════════════════════════════════════════════════════════════
# Reset button: single global callback triggered by any of the 5 reset
# buttons, writes _DEFAULTS to the store. The restore callbacks sync
# each tab's 6 shared controls.
# ══════════════════════════════════════════════════════════════════════
_app_ctx.app.clientside_callback(
    f"""
    function(bub, dca, ret, sc, cp) {{
        var trig = (window.dash_clientside.callback_context || {{}}).triggered;
        if (!trig || !trig.length || trig[0].value == null) {{
            return window.dash_clientside.no_update;
        }}
        return {_DEFAULTS_JSON};
    }}
    """,
    Output("plot-appearance", "data", allow_duplicate=True),
    Input("bub-plot-appearance-reset", "n_clicks"),
    Input("dca-plot-appearance-reset", "n_clicks"),
    Input("ret-plot-appearance-reset", "n_clicks"),
    Input("sc-plot-appearance-reset", "n_clicks"),
    Input("cp-plot-appearance-reset", "n_clicks"),
    prevent_initial_call=True,
)

# Bubble-tab pt_size/pt_alpha reset (these live in the card but aren't
# in the shared store since they're bubble-only)
_app_ctx.app.clientside_callback(
    """
    function(n) {
        if (!n) return [window.dash_clientside.no_update,
                         window.dash_clientside.no_update];
        return [10, 0.5];
    }
    """,
    Output("bub-ptsize", "value", allow_duplicate=True),
    Output("bub-ptalpha", "value", allow_duplicate=True),
    Input("bub-plot-appearance-reset", "n_clicks"),
    prevent_initial_call=True,
)
