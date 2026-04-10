"""Plot Appearance controls — trace thickness, grid width/color, BM color.

Rendered on every chart tab (bubble, DCA, retire, supercharger, citadel).
Each tab has its own input IDs prefixed by the tab's key. All tabs
read from and write to the same global 'plot-appearance' localStorage
store, so changes propagate across tabs.

The chart_responsive.js asset reads the store directly and applies
values via Plotly.restyle/Plotly.relayout on every chart.
"""
from dash import Input, Output
import _app_ctx


_DEFAULTS = {
    "trace_width": 2.5,
    "grid_major_width": 1.0,
    "grid_major_color": "#888888",
    "grid_minor_width": 0.8,
    "grid_minor_color": "#B0B0B0",
    "bm_color": "#C8960C",
}

_PREFIXES = ("bub", "dca", "ret", "sc", "cp")


for _prefix in _PREFIXES:
    # ── Persist control values → localStorage store ──────────────────────
    _app_ctx.app.clientside_callback(
        """
        function(tw, gmw, gmc, gnw, gnc, bmc) {
            if (tw == null && gmw == null && gmc == null && gnw == null && gnc == null && bmc == null) {
                return window.dash_clientside.no_update;
            }
            return {
                trace_width: tw,
                grid_major_width: gmw,
                grid_major_color: gmc,
                grid_minor_width: gnw,
                grid_minor_color: gnc,
                bm_color: bmc,
            };
        }
        """,
        Output("plot-appearance", "data", allow_duplicate=True),
        Input(f"{_prefix}-plot-trace-width", "value"),
        Input(f"{_prefix}-plot-grid-major-width", "value"),
        Input(f"{_prefix}-plot-grid-major-color", "value"),
        Input(f"{_prefix}-plot-grid-minor-width", "value"),
        Input(f"{_prefix}-plot-grid-minor-color", "value"),
        Input(f"{_prefix}-plot-bm-color", "value"),
        prevent_initial_call=True,
    )

    # ── Restore control values from store on page load ────────────────────
    _app_ctx.app.clientside_callback(
        """
        function(data) {
            var NU = window.dash_clientside.no_update;
            if (!data) return [NU, NU, NU, NU, NU, NU];
            return [data.trace_width, data.grid_major_width, data.grid_major_color,
                    data.grid_minor_width, data.grid_minor_color,
                    data.bm_color || "#C8960C"];
        }
        """,
        Output(f"{_prefix}-plot-trace-width", "value"),
        Output(f"{_prefix}-plot-grid-major-width", "value"),
        Output(f"{_prefix}-plot-grid-major-color", "value"),
        Output(f"{_prefix}-plot-grid-minor-width", "value"),
        Output(f"{_prefix}-plot-grid-minor-color", "value"),
        Output(f"{_prefix}-plot-bm-color", "value"),
        Input("plot-appearance", "data"),
        prevent_initial_call="initial_duplicate",
    )

    # ── Reset button → controls back to defaults ──────────────────────────
    _app_ctx.app.clientside_callback(
        f"""
        function(n) {{
            var NU = window.dash_clientside.no_update;
            if (!n) return [NU, NU, NU, NU, NU, NU];
            return [{_DEFAULTS['trace_width']}, {_DEFAULTS['grid_major_width']},
                    "{_DEFAULTS['grid_major_color']}", {_DEFAULTS['grid_minor_width']},
                    "{_DEFAULTS['grid_minor_color']}", "{_DEFAULTS['bm_color']}"];
        }}
        """,
        Output(f"{_prefix}-plot-trace-width", "value", allow_duplicate=True),
        Output(f"{_prefix}-plot-grid-major-width", "value", allow_duplicate=True),
        Output(f"{_prefix}-plot-grid-major-color", "value", allow_duplicate=True),
        Output(f"{_prefix}-plot-grid-minor-width", "value", allow_duplicate=True),
        Output(f"{_prefix}-plot-grid-minor-color", "value", allow_duplicate=True),
        Output(f"{_prefix}-plot-bm-color", "value", allow_duplicate=True),
        Input(f"{_prefix}-plot-appearance-reset", "n_clicks"),
        prevent_initial_call=True,
    )
