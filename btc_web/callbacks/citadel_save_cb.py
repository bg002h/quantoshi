"""Citadel Planner — save/load scenario callbacks."""

import base64
import datetime
import json
import logging

import dash
from dash import callback, Input, Output, State, no_update

import _app_ctx
from snapshot import _SNAPSHOT_CONTROLS
from callbacks.routing import _TAB_CONTROLS

log = logging.getLogger(__name__)

# ── Build the list of (component_id, property) for all Citadel controls ──
def _citadel_control_ids():
    cp_ids = _TAB_CONTROLS.get("citadel", set())
    return [(cid, prop) for cid, prop in _SNAPSHOT_CONTROLS if cid in cp_ids]

_CP_CONTROLS = _citadel_control_ids()

# JSON map of [{id, prop}, ...] for the clientside load callback
_CP_CONTROLS_JSON = json.dumps([{"id": cid, "prop": prop}
                                 for cid, prop in _CP_CONTROLS])


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
        return window.dash_clientside.no_update;
    }
    """,
    Output("cp-save-dl-dummy", "data"),
    Input("cp-save-prep", "data"),
    prevent_initial_call=True,
)



# Save button enabled via the main Citadel callback's running= parameter.
# Background callbacks use running= to set component properties during/after
# execution. We add cp-save-btn.disabled to the citadel_cb.py running= list
# instead of using a separate clientside callback (which can't detect
# background callback figure updates).


# ── Load: server-side parse → single store → clientside set_props ────────────
#
# The original approach used ~100 Output(..., allow_duplicate=True) to restore
# each control directly. This broke Dash 4's client-side callback graph and
# prevented tab routing from working (backlog #22).
#
# Fix: server callback parses JSON into a single store, then a clientside
# callback uses dash_clientside.set_props() to distribute values — zero
# allow_duplicate outputs needed.

# Open the loading modal immediately when the user picks a file.
# Uses 'filename' (not 'contents') because the server callback clears
# 'contents' to None, which would re-trigger a 'contents'-based callback.
_app_ctx.app.clientside_callback(
    """
    function(fn) {
        if (!fn) return window.dash_clientside.no_update;
        window.dash_clientside.set_props("cp-load-modal-body",
            {children: "\\u23f3 Loading scenario..."});
        return true;
    }
    """,
    Output("cp-load-modal", "is_open"),
    Input("cp-scenario-upload", "filename"),
    prevent_initial_call=True,
)


@callback(
    Output("cp-scenario-upload", "contents"),
    Output("cp-load-store", "data"),
    Input("cp-scenario-upload", "contents"),
    prevent_initial_call=True,
)
def _load_scenario(contents):
    if not contents:
        raise dash.exceptions.PreventUpdate

    def _err(msg):
        return None, {"error": msg}

    try:
        _content_type, content_string = contents.split(",", 1)
        raw = base64.b64decode(content_string)
        if len(raw) > 2_000_000:
            return _err("File too large (max 2 MB)")
        data = json.loads(raw)
    except Exception as e:
        return _err(f"Parse error: {e}")

    if data.get("type") != "citadel_scenario":
        return _err("Not a Quantoshi Citadel scenario file")
    if data.get("version", 0) > 1:
        return _err(f"Unsupported version: {data.get('version')}")

    created = data.get("created", "unknown")[:19]
    return None, {
        "controls": data.get("controls", {}),
        "figure": data.get("figure"),
        "mc_results": data.get("mc_results"),
        "tax_config": data.get("tax_config"),
        "tax_annual": data.get("tax_annual"),
        "created": created,
    }


# Clientside callback: distributes loaded scenario to individual controls
# via set_props (no allow_duplicate outputs needed), manages loading modal.
_app_ctx.app.clientside_callback(
    """
    function(scenario) {
        if (!scenario) return window.dash_clientside.no_update;
        var sp = window.dash_clientside.set_props;
        var NU = window.dash_clientside.no_update;

        if (scenario.error) {
            // Show error in modal, close after 2s
            sp("cp-load-modal-body", {children: scenario.error});
            setTimeout(function() { sp("cp-load-modal", {is_open: false}); }, 2000);
            return NU;
        }

        var controls = scenario.controls || {};
        var map = %s;

        // Restore each control via set_props
        for (var i = 0; i < map.length; i++) {
            var cid = map[i].id;
            var prop = map[i].prop;
            if (controls.hasOwnProperty(cid)) {
                var obj = {};
                obj[prop] = controls[cid];
                sp(cid, obj);
            }
        }

        // Restore figure, MC results, tax config, tax annual
        if (scenario.figure)     sp("citadel-graph",      {figure: scenario.figure});
        if (scenario.mc_results) sp("cp-mc-results",      {data: scenario.mc_results});
        if (scenario.tax_config) sp("cp-tax-config",      {data: scenario.tax_config});
        if (scenario.tax_annual) sp("cp-tax-annual-data", {data: scenario.tax_annual});

        // Clear any loading overlays that set_props triggers.
        // dcc.Loading adds a full-screen spinner and/or dims the content.
        // The overlay can appear at different times depending on figure size
        // and network latency, so we sweep repeatedly.
        function clearLoadingOverlays() {
            // Remove full-screen spinners
            document.querySelectorAll(".dash-spinner-container").forEach(
                function(el) { el.remove(); });
            // Remove any lingering modal backdrops
            document.querySelectorAll(".modal-backdrop").forEach(
                function(el) { el.remove(); });
            // Force the dcc.Loading wrapper to show content normally
            var wrap = document.getElementById("cp-chart-wrap");
            if (wrap) {
                wrap.querySelectorAll("[data-dash-is-loading]").forEach(
                    function(el) { el.removeAttribute("data-dash-is-loading"); });
                // Reset any inline visibility/opacity the Loading component set
                wrap.querySelectorAll("div").forEach(function(el) {
                    if (el.style.visibility === "hidden") el.style.visibility = "";
                    if (parseFloat(el.style.opacity) < 1) el.style.opacity = "";
                });
            }
        }
        // Sweep at multiple intervals to catch late-appearing overlays
        [100, 300, 600, 1200, 2500].forEach(function(ms) {
            setTimeout(clearLoadingOverlays, ms);
        });

        // Reset the file input so the same file can be re-loaded
        setTimeout(function() {
            var inp = document.querySelector("#cp-scenario-upload input[type=file]");
            if (inp) inp.value = "";
        }, 100);

        // Show completion in modal, then close
        sp("cp-load-modal-body", {children:
            "\\u2705 Scenario loaded: " + (scenario.created || "unknown")
        });
        setTimeout(function() { sp("cp-load-modal", {is_open: false}); }, 1200);

        // Status message (persists after modal closes)
        sp("cp-load-status", {children: "Loaded: " + (scenario.created || "unknown")});

        return NU;
    }
    """ % _CP_CONTROLS_JSON,
    Output("cp-load-store", "data", allow_duplicate=True),  # dummy output
    Input("cp-load-store", "data"),
    prevent_initial_call=True,
)
