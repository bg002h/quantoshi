"""Citadel Quick Scenarios — pill button callbacks and band loading."""
from __future__ import annotations

import json as _json

from dash import callback, no_update, Input, Output, State

import _app_ctx
from citadel_presets import WEALTH_LEVELS, MACRO_REGIMES, RULE_SETS, BTC_ENTRY_QS


# ── Quantile snapping ────────────────────────────────────────────────────────

def _snap_entry_q(q_float: float) -> int:
    """Snap a quantile float (0.01–0.999) to the nearest cached entry bin (1/10/50)."""
    q_pct = round(q_float * 100)
    return min(BTC_ENTRY_QS, key=lambda b: abs(q_pct - b))


# ── Pill button click → update store + toggle outline ────────────────────────

def _register_pill_cb(group_name, keys):
    """Register a clientside callback that updates a store and toggles pill outlines."""
    outputs = [Output(f"cp-scenario-{group_name}", "data")]
    outputs += [Output(f"cp-pill-{k}", "outline") for k in keys]
    inputs = [Input(f"cp-pill-{k}", "n_clicks") for k in keys]
    keys_json = _json.dumps(keys)
    _app_ctx.app.clientside_callback(
        f"""function() {{
            var keys = {keys_json};
            var tid = dash_clientside.callback_context.triggered_id;
            if (!tid) return Array(1 + keys.length).fill(window.dash_clientside.no_update);
            var selected = tid.replace('cp-pill-', '');
            var outlines = keys.map(function(k) {{ return k !== selected; }});
            return [selected].concat(outlines);
        }}""",
        *outputs, *inputs,
        prevent_initial_call=True,
    )


_register_pill_cb("wealth", list(WEALTH_LEVELS.keys()))
_register_pill_cb("regime", list(MACRO_REGIMES.keys()))
_register_pill_cb("rules", list(RULE_SETS.keys()))


# ── Load cached bands when any scenario dimension changes ────────────────────

@callback(
    Output("cp-scenario-bands", "data"),
    Output("cp-scenario-active", "data"),
    Input("cp-scenario-wealth", "data"),
    Input("cp-scenario-regime", "data"),
    Input("cp-scenario-rules", "data"),
    Input("cp-scenario-start-yr", "value"),
    State("cp-qs", "value"),
    State("cp-model-src", "value"),
    State("cp-tax-config", "data"),
    prevent_initial_call=True,
)
def load_scenario_bands(wealth, regime, rules, start_yr, quantile, model_src, tax_config):
    """Look up pre-computed bands for the selected scenario."""
    if not all([wealth, regime, rules, start_yr]):
        return no_update, no_update

    from citadel_band_cache import lookup_entry

    model_key = model_src or "bub"
    entry_q = _snap_entry_q(float(quantile or 0.25))
    tax_status = tax_config.get("filing_status", "single") if isinstance(tax_config, dict) else "single"

    bands = lookup_entry(model_key, entry_q, regime, wealth, rules,
                         int(start_yr), tax_status)
    if bands is None:
        return None, None

    # Serialize for dcc.Store (numpy arrays → lists)
    serialized = {}
    for pct, series_dict in bands.items():
        serialized[str(pct)] = {k: v.tolist() for k, v in series_dict.items()}

    active_key = f"{model_key}_q{entry_q}_{regime}_{wealth}_{rules}_{start_yr}_{tax_status}"
    return serialized, active_key


# ── Auto-fill controls when scenario loads ───────────────────────────────────

_FILL_OUTPUTS = [
    "cp-stack", "cp-cash-init",
    "cp-res-short-init", "cp-res-med-init", "cp-res-long-init",
    "cp-inv-eq-init", "cp-inv-bd-init",
    "cp-spend", "cp-infl", "cp-spend-growth",
    "cp-cash-floor", "cp-high-q-thresh", "cp-low-q-thresh",
    "cp-yr-range", "cp-asset-model",
]


@callback(
    [Output(cid, "value", allow_duplicate=True) for cid in _FILL_OUTPUTS]
    + [Output("cp-scenario-applied", "data", allow_duplicate=True)],
    Input("cp-scenario-active", "data"),
    State("cp-scenario-wealth", "data"),
    State("cp-scenario-regime", "data"),
    State("cp-scenario-rules", "data"),
    State("cp-scenario-start-yr", "value"),
    State("cp-scenario-applied", "data"),
    prevent_initial_call=True,
)
def auto_fill_controls(active_key, wealth, regime, rules, start_yr, applied_tick):
    """Fill Citadel controls with preset values when a scenario loads.

    On success, also bumps cp-scenario-applied (a tick counter). update_citadel
    listens to that store as Input and re-renders the chart with the freshly
    auto-filled control values. This gives the user one-click scenarios:
    pick Bull/Bear/Wealth-tier → chart updates without manually clicking Run.
    Without the tick store, update_citadel would race auto_fill_controls and
    read stale State values.
    """
    if not active_key or not wealth:
        return [no_update] * len(_FILL_OUTPUTS) + [no_update]
    from citadel_presets import preset_control_values
    vals = preset_control_values(wealth, regime, rules, int(start_yr or 2035))
    next_tick = (applied_tick or 0) + 1
    return [vals[cid] for cid in _FILL_OUTPUTS] + [next_tick]


# ── Stale indicator ──────────────────────────────────────────────────────────

def is_scenario_stale(stack, spend, cash_init, infl, cash_floor,
                      active_key, wealth, regime, rules, start_yr) -> bool:
    """Return True when any of the 5 watched controls differs from the preset.

    Public helper so the chart callback can suppress cached scenario bands
    when the user has modified inputs away from the preset. Same logic
    used by detect_stale (which drives the visible badge).
    """
    if not active_key or not wealth:
        return False
    from citadel_presets import preset_control_values
    vals = preset_control_values(wealth, regime, rules, int(start_yr or 2035))
    for cid, cur in [("cp-stack", stack), ("cp-spend", spend),
                      ("cp-cash-init", cash_init), ("cp-infl", infl),
                      ("cp-cash-floor", cash_floor)]:
        if cur is not None and float(cur) != vals[cid]:
            return True
    return False


@callback(
    Output("cp-scenario-stale", "style"),
    Input("cp-stack", "value"),
    Input("cp-spend", "value"),
    Input("cp-cash-init", "value"),
    Input("cp-infl", "value"),
    Input("cp-cash-floor", "value"),
    State("cp-scenario-active", "data"),
    State("cp-scenario-wealth", "data"),
    State("cp-scenario-regime", "data"),
    State("cp-scenario-rules", "data"),
    State("cp-scenario-start-yr", "value"),
    prevent_initial_call=True,
)
def detect_stale(stack, spend, cash_init, infl, cash_floor,
                 active_key, wealth, regime, rules, start_yr):
    """Show stale indicator when controls differ from loaded preset."""
    stale = is_scenario_stale(stack, spend, cash_init, infl, cash_floor,
                              active_key, wealth, regime, rules, start_yr)
    return {} if stale else {"display": "none"}
