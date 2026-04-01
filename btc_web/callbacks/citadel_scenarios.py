"""Citadel Quick Scenarios — pill button callbacks and band loading."""
from __future__ import annotations

from dash import callback, no_update, Input, Output, State, ctx

from citadel_presets import WEALTH_LEVELS, MACRO_REGIMES, RULE_SETS, BTC_ENTRY_QS


# ── Quantile snapping ────────────────────────────────────────────────────────

def _snap_entry_q(q_float: float) -> int:
    """Snap a quantile float (0.01–0.999) to the nearest cached entry bin (1/10/50)."""
    q_pct = round(q_float * 100)
    return min(BTC_ENTRY_QS, key=lambda b: abs(q_pct - b))


# ── Pill button click → update store + toggle outline ────────────────────────

def _register_pill_cb(group_name, keys):
    """Register a callback that updates a store and toggles pill outlines."""
    outputs = [Output(f"cp-scenario-{group_name}", "data")]
    outputs += [Output(f"cp-pill-{k}", "outline") for k in keys]
    inputs = [Input(f"cp-pill-{k}", "n_clicks") for k in keys]

    @callback(outputs, inputs, prevent_initial_call=True)
    def _on_pill_click(*n_clicks_args):
        triggered = ctx.triggered_id
        if not triggered:
            return [no_update] * (1 + len(keys))
        selected = triggered.replace("cp-pill-", "")
        outlines = [k != selected for k in keys]
        return [selected] + outlines


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
    prevent_initial_call=True,
)
def load_scenario_bands(wealth, regime, rules, start_yr, quantile):
    """Look up pre-computed bands for the selected scenario."""
    if not all([wealth, regime, rules, start_yr]):
        return no_update, no_update

    from citadel_band_cache import lookup_entry

    model_key = "bub"
    entry_q = _snap_entry_q(float(quantile or 0.25))

    bands = lookup_entry(model_key, entry_q, regime, wealth, rules,
                         int(start_yr), "single")
    if bands is None:
        return None, None

    # Serialize for dcc.Store (numpy arrays → lists)
    serialized = {}
    for pct, series_dict in bands.items():
        serialized[str(pct)] = {k: v.tolist() for k, v in series_dict.items()}

    active_key = f"{model_key}_q{entry_q}_{regime}_{wealth}_{rules}_{start_yr}_single"
    return serialized, active_key
