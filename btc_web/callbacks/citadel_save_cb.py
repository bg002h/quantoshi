"""Citadel Planner — save scenario (server prep + clientside download)."""

import datetime
import json
import logging

from dash import callback, Input, Output, State

import _app_ctx
from snapshot import _SNAPSHOT_CONTROLS
from callbacks.routing import _TAB_CONTROLS

log = logging.getLogger(__name__)

# ── Build the list of (component_id, property) for all Citadel controls ──
def _citadel_control_ids():
    cp_ids = _TAB_CONTROLS.get("citadel", set())
    return [(cid, prop) for cid, prop in _SNAPSHOT_CONTROLS if cid in cp_ids]

_CP_CONTROLS = _citadel_control_ids()

# Map from component id to index in _CP_CONTROLS for fast lookup
_CID_IDX = {cid: i for i, (cid, _) in enumerate(_CP_CONTROLS)}


def _build_filename(controls_dict):
    """Build a descriptive filename from control values."""
    def _g(cid, default=""):
        return controls_dict.get(cid, default)

    yr = _g("cp-yr-range", [2031, 2075])
    start, end = (yr[0], yr[1]) if isinstance(yr, (list, tuple)) and len(yr) >= 2 else (2031, 2075)

    freq = str(_g("cp-freq", "Monthly")).lower()[:3]

    qs = _g("cp-qs", [])
    if isinstance(qs, list) and qs:
        q_str = str(qs[0]).replace(".", "p")
    else:
        q_str = "q50"

    stack = _g("cp-stack", 1.0)
    try:
        stack = float(stack)
    except (TypeError, ValueError):
        stack = 1.0

    cash = _g("cp-cash-init", 0)
    try:
        cash = int(float(cash))
    except (TypeError, ValueError):
        cash = 0

    spend = _g("cp-spend", 5000)
    try:
        spend = int(float(spend))
    except (TypeError, ValueError):
        spend = 5000

    parts = [
        "citadel",
        f"{start}-{end}",
        freq,
        q_str,
        f"stack-{stack:g}",
        f"cash-{cash}",
        f"spend-{spend}",
    ]

    # Optional tax suffix
    tax_config = _g("cp-tax-config")
    tax_toggle = _g("cp-tax-toggle")
    if tax_toggle and tax_config and isinstance(tax_config, dict):
        state_code = tax_config.get("state", "")
        if state_code:
            parts.append(f"tax-{state_code}")

    # Optional MC suffix
    mc_enable = _g("cp-mc-enable")
    if mc_enable and isinstance(mc_enable, list) and mc_enable:
        mc_sims = _g("cp-mc-sims", 0)
        try:
            mc_sims = int(mc_sims)
        except (TypeError, ValueError):
            mc_sims = 0
        if mc_sims > 1:
            parts.append(f"mc-{mc_sims}s")

    parts.append(datetime.date.today().isoformat())
    return "_".join(parts) + ".json"


# ── Server-side save prep callback ──────────────────────────────────────────

# Extra stores to capture (not in _SNAPSHOT_CONTROLS)
_EXTRA_STATES = [
    ("citadel-graph",      "figure"),
    ("cp-mc-results",      "data"),
    ("cp-tax-config",      "data"),
    ("cp-tax-annual-data", "data"),
]

# Note: cp-tax-config appears both in _CP_CONTROLS (snapshot) and _EXTRA_STATES.
# We include it in _EXTRA_STATES so we always get it regardless of snapshot order.

@callback(
    Output("cp-save-prep", "data"),
    Input("cp-save-btn", "n_clicks"),
    [State(cid, prop) for cid, prop in _CP_CONTROLS] +
    [State(cid, prop) for cid, prop in _EXTRA_STATES],
    prevent_initial_call=True,
)
def _save_prep(n_clicks, *args):
    if not n_clicks:
        return None

    n_cp = len(_CP_CONTROLS)

    # Build controls dict: {component_id: value}
    controls = {}
    for i, (cid, _prop) in enumerate(_CP_CONTROLS):
        controls[cid] = args[i]

    # Build extra data dict
    extra = {}
    for j, (cid, prop) in enumerate(_EXTRA_STATES):
        extra[f"{cid}.{prop}"] = args[n_cp + j]

    # Strip heavy MC per-sim arrays, keep summary
    mc_data = extra.get("cp-mc-results.data")
    if mc_data and isinstance(mc_data, dict):
        keep_keys = {
            "time_axis", "percentiles", "median", "depletion_period",
            "annual_taxes", "rebal_events", "n_sims", "cumulative_spend",
            "path_key", "overlay_key", "created", "tab", "metadata",
        }
        mc_data = {k: v for k, v in mc_data.items() if k in keep_keys}
        extra["cp-mc-results.data"] = mc_data

    # Build scenario dict
    scenario = {
        "app": "Quantoshi",
        "type": "citadel_scenario",
        "version": 1,
        "created": datetime.datetime.utcnow().isoformat() + "Z",
        "controls": controls,
        "figure": extra.get("citadel-graph.figure"),
        "mc_results": extra.get("cp-mc-results.data"),
        "tax_config": extra.get("cp-tax-config.data"),
        "tax_annual": extra.get("cp-tax-annual-data.data"),
    }

    filename = _build_filename(controls)
    return {"filename": filename, "data": scenario}


# ── Clientside download: watches cp-save-prep, triggers browser download ────

_app_ctx.app.clientside_callback(
    """
    function(prep) {
        if (!prep || !prep.data) return window.dash_clientside.no_update;
        var json = JSON.stringify(prep.data, null, 2);
        var blob = new Blob([json], {type: 'application/json'});
        var url  = URL.createObjectURL(blob);
        var a    = document.createElement('a');
        a.href     = url;
        a.download = prep.filename || 'citadel_scenario.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return '';
    }
    """,
    Output("cp-load-status", "children"),
    Input("cp-save-prep", "data"),
    prevent_initial_call=True,
)


# ── Enable/disable Save button based on graph data ─────────────────────────

_app_ctx.app.clientside_callback(
    """
    function(fig) {
        if (!fig || !fig.data || fig.data.length === 0) return true;
        return false;
    }
    """,
    Output("cp-save-btn", "disabled"),
    Input("citadel-graph", "figure"),
    prevent_initial_call=True,
)
